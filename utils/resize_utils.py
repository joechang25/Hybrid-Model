import os
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import mixed_precision


def get_learning_rate(args, epoch):
    return args.lr * (0.1 ** (epoch // 30))


def get_weight_decay(args, epoch):
    return args.wd * (0.1 ** (epoch // 30))


def adjust_learning_rate(optimizer, epoch, args):
    lr = get_learning_rate(args, epoch)
    optimizer.lr.assign(lr)

    if args.optim in ['sgdw', 'adamw']:
        """For optimizers with weight decay, adjust the weight decay as well"""
        wd = get_weight_decay(args, epoch)
        optimizer.weight_decay.assign(wd)
        print("Epoch {}: lr = {:.2e} - wd = {:.2e}".format(epoch, lr, wd))
    else:
        print("Epoch {}: lr = {:.2e}".format(epoch, lr))


def get_resize_loss_fn(args):
    return tf.keras.losses.MeanSquaredError()

def get_loss_fn(args):
    return tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    
def get_optimizer(args):
    print("  => using optimizer '{}'".format(args.optim))

    lr = get_learning_rate(args, 0)
    if args.optim == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr,
                                            momentum=args.momentum)
    elif args.optim == 'sgdw':
        optimizer = tfa.optimizers.SGDW(learning_rate=lr,
                                        momentum=args.momentum,
                                        weight_decay=args.wd)
    elif args.optim == 'adam':
        epsilon = 1e-4 if args.mixed else 1e-7
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr,
                                             epsilon=epsilon)
    elif args.optim == 'adamw':
        epsilon = 1e-4 if args.mixed else 1e-7
        optimizer = tfa.optimizers.AdamW(learning_rate=lr,
                                         epsilon=epsilon,
                                         weight_decay=args.wd)

    if args.mixed:
        optimizer = mixed_precision.LossScaleOptimizer(optimizer)
    return optimizer
