import json
import os
import time
import math
import argparse

import numpy
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras import mixed_precision
from keras import backend as K
import models
from models.utils import *
from utils.utils import *
from utils.dataset import *
from keras.backend import manual_variable_initialization 



model_names = sorted(models.model_class.keys())
supported_optimizer = ['sgd', 'sgdw', 'adam', 'adamw', 'rms']
supported_print_format = ['tf', 'torch']
supported_dataset = ['dogcat', 'food', 'imagenet']
data_size = {"dogcat" : 20000, "food" : 75750, "imagenet" : 1281167}
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
parser.add_argument('-b', '--batch-size', default=128, type=int,
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
parser.add_argument('--resize', dest='resize', action='store_true')
parser.add_argument('--conf', dest='conf', type=float, default=0.5, help='set confidence level of the difficulty model')
parser.add_argument('--dataset', dest='dataset',default='dogcat',type=str, help='dataset: ' + '|'.join(supported_dataset))
parser.add_argument('--difficulty', dest='difficulty', default='224x112',type=str, help='choose the size of difficulty, from large to small size. e.g. 224x112 or 224x112x56')

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


def get_classifier(args, image_size):
    image_size_orig = args.image_size 
    args.image_size = image_size
    classifier = get_model(class_size[args.dataset], args)
    print("Load weight, size :", image_size)
    classifier.load_weights(args.dataset + "_model/" + args.arch + "_" + str(image_size) + ".h5")
    
    for layer in classifier.layers:
        layer.trainable = False
    args.image_size = image_size_orig
    return tf.function(classifier)


class ResizeNet(tf.keras.Model):
    def __init__(self, args, sizes):
        """
        - args: input args of this program.
        - sizes: desired sizes to re-size. The sizes must
          be listed from large to small.
        """
        super(ResizeNet, self).__init__()
        self.num_sizes = len(sizes)
        self.sizes = sizes
        self.classifiers = [get_classifier(args, s) for s in sizes]
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        self.conf = args.conf


    def lookup_label(self, images, target):
        target = tf.math.argmax(target, axis=-1, output_type=tf.int32)
        indices = tf.reshape(target, shape=(-1, 1))
        size = self.sizes[0]
        label = tf.zeros_like(indices, dtype=tf.int32)
        pred = self.classifiers[0](tf.image.resize(images, [size, size]))
        prob = tf.gather_nd(pred, indices, batch_dims=1)
        confident = tf.greater_equal(prob, self.conf)
        for i in range(1, self.num_sizes):
            size = self.sizes[i]
            pred = self.classifiers[i](tf.image.resize(images, [size, size]))
            prob = tf.gather_nd(pred, indices, batch_dims=1)
            pred = tf.math.argmax(pred, axis=-1, output_type=tf.int32)
            correct = tf.equal(pred, target)
            confident = tf.logical_and(confident, tf.greater_equal(prob, self.conf))
            l = tf.where(tf.logical_and(correct, confident), i, 0)
            l = tf.cast(l, dtype=tf.int32)
            l = tf.reshape(l, shape=(-1, 1))
            label = tf.concat([label, l], axis=-1)

        label = tf.reduce_max(label, axis=-1)
        label = self.num_sizes - label - 1
        return tf.one_hot(label, self.num_sizes, dtype=tf.float32)


def main():
    args = parse_options()
    image_size = args.image_size
    difficulty = args.difficulty.split("x")
    
    loss_fn = get_loss_fn(args)
    optimizer = get_optimizer(args)
    
    for i in range(len(difficulty)):
        difficulty[i] = int(difficulty[i])
    
    # Create model
    model = ResizeNet(args, difficulty)
    model.compile(optimizer=optimizer, loss=loss_fn)

    if args.summary:
        resizer.summary()
        print('')

	# Data loading code
    print("Loading data...")
    train_dir = os.path.join(args.data, 'train') 
    valid_ds = get_eval_dataset(train_dir, image_size, class_size[args.dataset], args)
    
    validate(valid_ds, model, loss_fn, args)


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


def validate(valid_ds, model, loss_fn, args):

    num_batch = math.ceil(data_size[args.dataset] / args.batch_size)
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    num0 = AverageMeter('Easy', ':6.2f')
    num1 = AverageMeter('Medium', ':6.2f')
    num2 = AverageMeter('Hard', ':6.2f')
    #top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        num_batch,
        [batch_time, None, losses, top1, num0, num1, num2],
        prefix='Test: ',
        fmt=args.print_format)

    end = time.time()
    image_size = args.image_size
    
    for i, (images, target) in enumerate(valid_ds):
        
        label = model.lookup_label(images, target)    
        
        mean = [103.939, 116.779, 123.68]
        images = images[..., ::-1]
        images += mean
        images /= 127.5
        images -= 1.0

        acc1, choice = accuracy(label, label, topk=(1))

        #losses.update(loss.numpy(), len(images))
        top1.update(acc1, len(images))

        num0.update(choice[0])
        num1.update(choice[1])
        num2.update(choice[2])

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

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
        self.total = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.total += val

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter():
    def __init__(self, num_batches, meters, prefix="", fmt="tf"):
        self.num_batches = num_batches
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches - 1)
        self.meters = meters
        self.prefix = prefix
        self.display_fn = self.display_tf if fmt == "tf" else self.display_torch

    def display(self, batch):
        self.display_fn(batch)

    def display_tf(self, batch):
        batch_time, _, losses, top1, num0, num1, num2 = self.meters
        terimal_width = os.get_terminal_size()[0]
        if batch + 1 == self.num_batches:
            end = '\n'
            total = time.strftime('%H:%M:%S', time.gmtime(batch_time.sum))
            s = self.prefix + self.batch_fmtstr.format(batch) + \
                ' - time: {}'.format(total) + \
                ' - Easy: {:2.2f}'.format(num0.total) + \
                ' - Medium: {:2.2f}'.format(num1.total) + \
                ' - Hard: {:2.2f}'.format(num2.total)
        else:
            end = '\r'
            eta = time.strftime('%H:%M:%S',
                                time.gmtime(batch_time.avg * (self.num_batches - batch)))
            s = self.prefix + self.batch_fmtstr.format(batch) + \
                ' - ETA: {}'.format(eta) + \
                ' - Easy: {:2.2f}'.format(num0.total) + \
                ' - Medium: {:2.2f}'.format(num1.total) + \
                ' - Hard: {:2.2f}'.format(num2.total)
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
    batch_size = len(target)
    
    pred = tf.math.top_k(output, 1, True)
    target = tf.math.argmax(target, axis=-1, output_type=tf.int32)
    target = tf.reshape(target, shape=(-1, 1))
    
    indices = pred.indices.numpy()
    targets = target.numpy()
    
    choice = [0, 0, 0, 0]
    for i in range(4):
        choice[i] = tf.reduce_sum(tf.cast(tf.math.equal(indices, i), tf.int32))
    correct = tf.math.equal(indices, targets)    
    correct_k = tf.math.reduce_sum(tf.cast(correct[:,:1], tf.int32))
    res = correct_k.numpy() * (100.0 / batch_size)
    
    return res, choice


if __name__ == '__main__':
    main()
