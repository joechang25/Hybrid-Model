import json
import os
import time
import math
import argparse

import numpy
import pandas as pd
from PIL import Image
import cv2
import tensorflow as tf
from tensorflow.keras import mixed_precision
from timeit import default_timer as timer

import models
from models.utils import *
from utils.resize_utils import *
from utils.resize_dataset import *

import tvm
from tvm import te
import tvm.relay as relay
from tvm import autotvm

model_names = sorted(models.model_class.keys())
supported_optimizer = ['sgd', 'sgdw', 'adam', 'adamw']
supported_print_format = ['tf', 'torch']
supported_dataset = ['dogcat', 'food', 'imagenet']
data_size = {"dogcat" : 5000, "food" : 25250, "imagenet" : 50000}
class_size = {"dogcat" : 2, "food" : 101, "imagenet" : 1000}

parser = argparse.ArgumentParser(description='Tensorflow ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet50)')
parser.add_argument('-e', '--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=1, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128)')
parser.add_argument('-s', '--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-j', '--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 0=auto)')
parser.add_argument('-t', '--test', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--optim', default='sgdw', type=str, metavar='OPT',
                    help='optimizer: ' +
                         ' | '.join(supported_optimizer) +
                         ' (default: sgdw)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='WD', help='initial weight decay', dest='wd')
parser.add_argument('--dropout', default=0.0, type=float, metavar='D',
                    help='dropout rate')
parser.add_argument('--image-size', default=224, type=int,
                    metavar='I', help='image size (default: 224)')
parser.add_argument('--print-format', default='tf', type=str, metavar='F',
                    help="print format: tf | torch (default: tf)")
parser.add_argument('--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to the checkpoint (default: none)')
parser.add_argument('--save-dir', default='saved_model', type=str, metavar='PATH',
                    help='path to save checkpoint (default: ./saved_model)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--use-xla', dest='use_xla', action='store_true',
                    help='use xla compilation')
parser.add_argument('--mixed', dest='mixed', action='store_true',
                    help='use mixed precision')
parser.add_argument('--torch-init', dest='torch_init', action='store_true',
                    help='use torch layer initialization')
parser.add_argument('--summary', dest='summary', action='store_true',
                    help='print model summary')
parser.add_argument('--grad-accum', default=1, type=int, metavar='N',
                    help='number of batches for gradient accumulation (default: 1)')
parser.add_argument('--dataset', dest='dataset',default='dogcat',type=str, help='dataset: ' + '|'.join(supported_dataset))
parser.add_argument('--difficulty', dest='difficulty', default='224x112',type=str, help='choose the size of difficulty, from large to small size. e.g. 224x112 or 224x112x56')
parser.add_argument('--conf', dest='conf', type=float, default=0.5, help='set confidence level of the difficulty model')

def parse_options():
    args = parser.parse_args()

    if args.optim not in supported_optimizer:
        raise ValueError("Invalid optimizer '{}'".format(args.optim))
    if args.print_format not in supported_print_format:
        raise ValueError("Invalid print format '{}'".format(args.print_format))
    if args.resume and not os.path.exists(args.resume):
        raise ValueError('Resume but model file does not exist')
    if args.workers < 0:
        raise ValueError("Invalid number of workers '{}'".format(args.workers))
    if args.grad_accum < 1:
        raise ValueError("Invalid gradient accumulation number '{}'".format(args.grad_accum))

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    else:
        print('  => using CPU, this will be slow')

    if args.mixed:
        print('  => using mixed precision')
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)

    if args.grad_accum > 1:
        args.batch_size *= args.grad_accum
        print("  => using gradient accumulation; resetting "
              "batch size to '{}'".format(args.batch_size))

    if args.use_xla:
        tf.keras.backend.clear_session()
        tf.config.optimizer.set_jit(True)

    if args.print_format == 'tf':
        args.print_freq = 1

    return args


def main():
    args = parse_options()
    image_size = args.image_size

    loss_fn = get_loss_fn(args)
    resize_loss_fn = get_resize_loss_fn(args)
    optimizer = get_optimizer(args)
    difficulty = args.difficulty.split("x")
    arch = args.arch    


    # Create model
    #args.arch = "mobilenet_v3_small"
    args.arch = arch
    valid_dir = os.path.join(args.data, 'val')
    resize_valid_ds = get_eval_dataset(valid_dir, image_size, class_size[args.dataset], args)

    custom_ob = {'hard_swish' : models.mobilenet_v3.hard_swish}
    resize_model = tf.keras.models.load_model("diff_" + args.dataset + "/" + arch + "_" + args.difficulty, custom_objects=custom_ob, compile=False)
    #resize_model.summary() 
    shape_dict = {"input_" + str(len(difficulty) + 1): (1, 3, image_size, image_size)}
    mod, params = relay.frontend.from_keras(resize_model, shape_dict)
    # compile the model
    target = "llvm"
    dev = tvm.cpu(0)

    with tvm.transform.PassContext(opt_level=3):
        resize_model = relay.build_module.create_executor("graph", mod, dev, target, params).evaluate()
    print("Finished compiling resize model")

    difficulty.reverse()
    size = difficulty
    pretrained_model = []
    for i in range(len(size)):
        print("Compiling model size : ", size[i])
        args.arch = arch
        size[i] = int(size[i])
        args.image_size = size[i]
        args.resume = None
        args.resize = None
        model = get_model(class_size[args.dataset], args)
        model.summary()
        pretrained_model.append(model)
        pretrained_model[i].load_weights(args.dataset + "_model/" + arch + "_" + str(size[i]) + ".h5")
        name = "input_" + str(i + 1)
        shape_dict = { name : (1, 3, size[i], size[i])}
        mod, params = relay.frontend.from_keras(pretrained_model[i], shape_dict)
        # compile the model
        target = "llvm"
        dev = tvm.cpu(0)

        model_name = "autotune_logs/" + arch + "_" + str(size[i])
        log_file = "%s.log" % model_name
        graph_opt_sch_file = "%s_graph_opt.log" % model_name
        with autotvm.apply_graph_best(graph_opt_sch_file):
            with tvm.transform.PassContext(opt_level=3):
                pretrained_model[i] = relay.build_module.create_executor("graph", mod, dev, target, params).evaluate()
        print("Finished compiling " + arch + " model : ", size[i])

    # Data loading code
    print("Loading code...")
    valid_dir = os.path.join(args.data, 'val')
    valid_ds = get_eval_dataset(valid_dir, image_size, class_size[args.dataset], args)
    print("Finished loading code")
    
    if args.evaluate:
        validate(resize_valid_ds, valid_ds, resize_model, pretrained_model, loss_fn, resize_loss_fn, args)
        return

    print("Start Training")
    for epoch in range(args.start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(train_ds, model, loss_fn, optimizer, epoch, args)

        # evaluate on validation set
        validate(valid_ds, model, loss_fn, args)

        #save_checkpoint(model, epoch, args)

    # Training finished
    model.save('{}/{}.h5'.format(args.save_dir, args.arch))


def save_checkpoint(model, epoch, args):
    model.save('{}/{}-ckpt-{}.h5'.format(args.save_dir, args.arch, epoch))


@tf.function
def train_step(model, loss_fn, optimizer, images, target):
    with tf.GradientTape() as tape:
        output = model(images, training=True)
        loss = loss_fn(target, output)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return output, loss


@tf.function
def train_step_mixed(model, loss_fn, optimizer, images, target):
    with tf.GradientTape() as tape:
        output = model(images, training=True)
        loss = loss_fn(target, output)
        scaled_loss = optimizer.get_scaled_loss(loss)

    scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
    gradients = optimizer.get_unscaled_gradients(scaled_gradients)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return output, loss


@tf.function
def compute_gradients(model, loss_fn, optimizer, scale, images, target):
    with tf.GradientTape() as tape:
        output = model(images, training=True)
        loss = loss_fn(target, output)
        loss *= scale

    gradients = tape.gradient(loss, model.trainable_variables)
    return output, loss, gradients


@tf.function
def compute_gradients_mixed(model, loss_fn, optimizer, scale, images, target):
    with tf.GradientTape() as tape:
        output = model(images, training=True)
        loss = loss_fn(target, output)
        loss *= scale
        scaled_loss = optimizer.get_scaled_loss(loss)

    scaled_gradients = tape.gradient(scaled_loss, model.trainable_variables)
    gradients = optimizer.get_unscaled_gradients(scaled_gradients)
    return output, loss, gradients


@tf.function
def apply_gradients(model, optimizer, gradients):
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def train_step_grad_accum(model, loss_fn, optimizer, images, target, args):
    batch_size = args.batch_size
    micro_batch_size = batch_size // args.grad_accum
    num_images = len(images)
    num_micro_batch = math.ceil(num_images / micro_batch_size)

    comp_fn = compute_gradients_mixed if args.mixed else compute_gradients

    sizes = []
    scales = []
    for i in range(num_micro_batch):
        size = min(num_images, micro_batch_size)
        sizes.append(size)
        scales.append(tf.constant(size / batch_size, dtype=tf.float32))
        num_images -= size

    split_images = tf.split(images, sizes, axis=0)
    split_target = tf.split(target, sizes, axis=0)

    # compute gradients of the first micro-batch
    output, loss, gradients = comp_fn(model, loss_fn, optimizer, scales[0],
                                      split_images[0], split_target[0])

    # accumulate gradients of the remaining micro-batches
    for i in range(1, num_micro_batch):
        o, l, g = comp_fn(model, loss_fn, optimizer, scales[i],
                          split_images[i], split_target[i])
        output = tf.concat([output, o], axis=0)
        loss += l
        for j in range(len(gradients)):
            gradients[j] += g[j]

    # apply gradients
    apply_gradients(model, optimizer, gradients)

    return output, loss


def do_train(model, loss_fn, optimizer, images, target, args):
    if args.grad_accum > 1:
        return train_step_grad_accum(model, loss_fn, optimizer, images,
                                     target, args)

    train_fn = train_step_mixed if args.mixed else train_step
    return train_fn(model, loss_fn, optimizer, images, target)


def train(train_ds, model, loss_fn, optimizer, epoch, args):
    num_batch = math.ceil(1281167 / args.batch_size)
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        num_batch,
        [batch_time, data_time, losses, top1],
        prefix='Epoch {}: '.format(epoch),
        fmt=args.print_format)

    end = time.time()
    for i, (images, target) in enumerate(train_ds):
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output
        output, loss = do_train(model, loss_fn, optimizer, images, target,
                                args)
        # measure accuracy and record loss
        acc1 = accuracy(output, target, topk=(1))
        losses.update(loss.numpy(), len(images))
        top1.update(acc1, len(images))
        top5.update(acc5, len(images))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)
        if i + 1 == num_batch:
            break


@tf.function
def test_step(model, loss_fn, images, target):
    output = model(images, training=False)
    loss = loss_fn(target, output)
    return output, loss


def test_step_grad_accum(model, loss_fn, images, target, args):
    batch_size = args.batch_size
    micro_batch_size = batch_size // args.grad_accum
    num_images = len(images)
    num_micro_batch = math.ceil(num_images / micro_batch_size)

    sizes = []
    scales = []
    for i in range(num_micro_batch):
        size = min(num_images, micro_batch_size)
        sizes.append(size)
        scales.append(tf.constant(size / batch_size, dtype=tf.float32))
        num_images -= size

    split_images = tf.split(images, sizes, axis=0)
    split_target = tf.split(target, sizes, axis=0)

    # compute the first micro-batch
    output, loss = test_step(model, loss_fn, split_images[0], split_target[0])
    loss = tf.cast(loss, tf.float32) * scales[0]

    # compute the remaining micro-batches
    for i in range(1, num_micro_batch):
        o, l = test_step(model, loss_fn, split_images[i], split_target[i])

        output = tf.concat([output, o], axis=0)
        loss += tf.cast(l, tf.float32) * scales[i]

    return output, loss


def do_test(model, loss_fn, images, target, args):
    if args.grad_accum > 1:
        return test_step_grad_accum(model, loss_fn, images, target, args)

    return test_step(model, loss_fn, images, target)

def test_top(output, conf, num_difficulty):
    output = output.numpy()
    for i in range(num_difficulty):
        if output[0][i] >= conf:
            return i
        else:
            output[0][i + 1] += output[0][i]
    #Should not go here
    return num_difficulty - 1

def preprocess_images(resize_valid_ds, valid_ds, start, end):
    resize_images = []
    images = []
    target = []
    #print('Preprocessing resize images...')
    cnt = 0
    for x, y in resize_valid_ds:
        im = x
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        im *= std
        im += mean
        im *= 255.
        im /= 127.5
        im -= 1.0
        x = im
        if cnt >= start and cnt < end:
            img = x.numpy().transpose([0, 3, 1, 2])
            resize_images.append(tvm.nd.array(img))
        if cnt >= end:
            break
        cnt += 1
        
    #print("Finished resize images")

    #print('Preprocessing inference images...')
    cnt = 0
    for x, y in valid_ds:
        if cnt >= start and cnt < end:
            img = x.numpy()
            tar = y
            images.append(img)
            target.append(tar)
        if cnt >= end:
            break
        cnt += 1
    #print('Finished inference images')
    return resize_images, images, target


def validate(resize_valid_ds, valid_ds, resize_model, pretrained_model, loss_fn, resize_loss_fn, args):
    
    num_batch = math.ceil(data_size[args.dataset] / args.batch_size)
    batch_time = AverageMeter('Time', ':6.3f')
    resize_time = AverageMeter(':6.3f')
    inference_time = AverageMeter(':6.3f')
    total_time = AverageMeter(':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    correct_95 = AverageMeter(':6.2f')
    total_95 = AverageMeter(':6.2f')
    correct_190 = AverageMeter(':6.2f')
    total_190 = AverageMeter(':6.2f')
    correct_380 = AverageMeter(':6.2f')
    total_380 = AverageMeter(':6.2f')

    progress = ProgressMeter(
        num_batch,
        [batch_time, resize_time, inference_time, total_time, None, losses, top1, top5, correct_95, total_95, correct_190, total_190, correct_380, total_380],
        prefix='Test: ',
        fmt=args.print_format)

    end = time.time()
    resize_num = [0, 0, 0]
    resize_cor = [0, 0, 0]
    #resize_images = []
    #images = []
    #target = []
    difficulty = args.difficulty.split("x")
    difficulty.reverse()
    difficulty = list(map(int, difficulty))
    num_diff = len(difficulty)

    
    chunk = 10
    print('Start Inference...')
    pri_i = 0
    for j in range(chunk):
        resize_images, images, target = preprocess_images(resize_valid_ds, valid_ds, j * int(data_size[args.dataset] / chunk), (j + 1) * int(data_size[args.dataset] / chunk))

        for i in range(int(data_size[args.dataset] / chunk)):
            total_start = time.time()
            resize_start = time.time()
            resize_output = resize_model(resize_images[i])
            resize_end = time.time()
            num = test_top(resize_output, args.conf, num_diff)
            resize_output = difficulty[num]

            if resize_output < 380:
                res = tf.image.resize(images[i], [resize_output, resize_output])
                res = res.numpy().transpose([0, 3, 1, 2])
                nd_arr = tvm.nd.array(res)
                in_start = time.time()
                output = pretrained_model[num](nd_arr)
                in_end = time.time()
            else:
                in_start = time.time()
                output = pretrained_model[num](tvm.nd.array(images[i].transpose([0, 3, 1, 2])))
                in_end = time.time()
    
            total_end = time.time()

            output = output.numpy()
            loss = loss_fn(target[i], output)
            resize_num[int(resize_output/190)] += 1
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target[i], topk=(1, 2))
            losses.update(loss.numpy(), 1)
            top1.update(acc1, 1)

            # measure elapsed time
            batch_time.update(time.time() - end)
            resize_time.update(resize_end - resize_start)
            inference_time.update(in_end - in_start)
            total_time.update(total_end - total_start)
            end = time.time()

            num = int(resize_output/190)
            if acc1 == 100.0:
                resize_cor[num] += 1
            if resize_num[0] != 0:
                correct_95.update(resize_cor[0] / resize_num[0] * 100)
            total_95.update(resize_num[0] / data_size[args.dataset] * 100)
            if resize_num[1] != 0:
                correct_190.update(resize_cor[1] / resize_num[1] * 100)
            total_190.update(resize_num[1] / data_size[args.dataset] * 100)
            if resize_num[2] != 0:
                correct_380.update(resize_cor[2]/ resize_num[2] * 100)
            total_380.update(resize_num[2] / data_size[args.dataset] * 100)

            if pri_i % args.print_freq == 0:
                progress.display(pri_i)
            pri_i += 1

    return top1.avg

class AverageMeter():
    # Computes and stores the average and current value
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)



class ProgressMeter():
    def __init__(self, num_batches, meters, prefix="", fmt="tf"):
        self.num_batches = num_batches
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches-1)
        self.meters = meters
        self.prefix = prefix
        self.display_fn = self.display_tf if fmt == "tf" else self.display_torch

    def display(self, batch):
        self.display_fn(batch)

    def display_tf(self, batch):
        batch_time, resize_time, inf_time, total_time, _, losses, top1, top5, cor_95, total_95, cor_190, total_190, cor_380, total_380 = self.meters
        terimal_width = os.get_terminal_size()[0]
        if batch + 1 == self.num_batches:
            end = '\n'
            total = time.strftime('%H:%M:%S', time.gmtime(batch_time.sum))
            s = self.prefix + self.batch_fmtstr.format(batch) + \
                    ' - Resize_time: {:.5f}'.format(resize_time.avg) + \
                    ' - Inference_time: {:.5f}'.format(inf_time.avg) + \
                    ' - total_time: {:.5f}'.format(total_time.avg) + \
                    ' - 95_correct: {:2.3f}'.format(cor_95.val) + \
                    ' - 95_total: {:2.3f}'.format(total_95.val) + \
                    ' - 190_correct: {:2.3f}'.format(cor_190.val) + \
                    ' - 190_total: {:2.3f}'.format(total_190.val) + \
                    ' - 380_correct: {:2.3f}'.format(cor_380.val) + \
                    ' - 380_total: {:2.3f}'.format(total_380.val) + \
                ' - time: {}'.format(total) + \
                ' - loss: {:.4f}'.format(losses.avg) + \
                ' - acc1: {:2.2f}'.format(top1.avg) + \
                ' - acc5: {:2.2f}'.format(top5.avg)
        else:
            end = '\r'
            eta = time.strftime('%H:%M:%S',
                    time.gmtime(batch_time.avg * (self.num_batches-batch)))
            s = self.prefix + self.batch_fmtstr.format(batch) + \
                    ' - Resize_time: {:.5f}'.format(resize_time.avg) + \
                    ' - Inference_time: {:.5f}'.format(inf_time.avg) + \
                    ' - total_time: {:.5f}'.format(total_time.avg) + \
                    ' - 95_correct: {:2.3f}'.format(cor_95.val) + \
                    ' - 95_total: :{:2.3f}'.format(total_95.val) + \
                    ' - 190_correct: {:2.3f}'.format(cor_190.val) + \
                    ' - 190_total: {:2.3f}'.format(total_190.val) + \
                    ' - 380_correct: {:2.3f}'.format(cor_380.val) + \
                    ' - 380_total: {:2.3f}'.format(total_380.val) + \
                ' - ETA: {}'.format(eta) + \
                ' - loss: {:.4f}'.format(losses.avg) + \
                ' - acc1/5: {:2.2f}/{:2.2f}'.format(top1.avg, top5.avg)
            lines = (len(s) + terimal_width - 1) // terimal_width
            if lines > 1:
                s += '\x1b[{}A'.format(lines - 1)

        print(s.ljust(terimal_width), end=end)

    def display_torch(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters if meter]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    # Computes accuracy over the k top predictions for the specified values of k
    maxk = max(topk)
    batch_size = len(target)
    pred = tf.math.top_k(output, maxk, True)
    target = tf.math.argmax(target, axis=-1, output_type=tf.int32)
    target = tf.reshape(target, shape=(-1, 1))
    correct = tf.math.equal(pred.indices, target)

    res = []
    for k in topk:
        correct_k = tf.math.reduce_sum(tf.cast(correct[:,:k], tf.int32))
        res.append(correct_k.numpy() * (100.0 / batch_size))
    return res

if __name__ == '__main__':
    main()
