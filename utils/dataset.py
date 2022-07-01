import os
import sys
import numpy as np
from functools import partial
import tensorflow as tf
import pandas as pd
from models.utils import get_normalize_fn



def do_scale(image, size):
    """Rescale the image by scaling the smaller spatial dimension to `size`."""
    shape = tf.cast(tf.shape(image), tf.float32)
    w_greater = tf.greater(shape[0], shape[1])
    shape = tf.cond(w_greater,
                    lambda: tf.cast([shape[0] / shape[1] * size, size], tf.int32),
                    lambda: tf.cast([size, shape[1] / shape[0] * size], tf.int32))

    return tf.image.resize(image, shape)


def center_crop(image, crop_height, crop_width):
    """Crops to center of image with specified `crop_height, crop_width`."""
    shape = tf.shape(image)
    height, width = shape[0], shape[1]

    amount_to_be_cropped_h = (height - crop_height)
    crop_top = amount_to_be_cropped_h // 2
    amount_to_be_cropped_w = (width - crop_width)
    crop_left = amount_to_be_cropped_w // 2
    return tf.slice(image, [crop_top, crop_left, 0], [crop_height, crop_width, -1])


def random_crop_and_flip(image):
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                       dtype=tf.float32,
                       shape=[1, 1, 4])

    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=0.1,
        aspect_ratio_range=(3. / 4, 4. / 3.),
        area_range=(0.08, 1.0),
        max_attempts=100,
        use_image_if_no_bounding_boxes=True
    )

    bbox_begin, bbox_size, _ = sample_distorted_bounding_box
    offset_y, offset_x, _ = tf.unstack(bbox_begin)
    target_height, target_width, _ = tf.unstack(bbox_size)
    cropped = tf.image.crop_to_bounding_box(image, offset_y, offset_x,
                                            target_height, target_width)
    cropped = tf.image.random_flip_left_right(cropped)
    return cropped


def preprocess_for_train(image, image_size):
    image = random_crop_and_flip(image)
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.cast(image, tf.float32)
    return image


def preprocess_for_eval(image, image_size):
    if image_size <= 256:
        image = do_scale(image, 256)
    else:
        image = do_scale(image, image_size)
    image = center_crop(image, image_size, image_size) 
    image = tf.image.resize(image, [image_size, image_size])
    image = tf.cast(image, tf.float32)
    return image


def read_tfrecord(example, image_size, class_size, normalize_fn, preprocess_fn):
    tfrecord_format = (
        {
            # Must be consistent with imagenet_to_gcs.py.
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
            'image/colorspace': tf.io.FixedLenFeature([], tf.string),
            'image/channels': tf.io.FixedLenFeature([], tf.int64),
            'image/class/label': tf.io.FixedLenFeature([], tf.int64),
            'image/class/synset': tf.io.FixedLenFeature([], tf.string),
            'image/format': tf.io.FixedLenFeature([], tf.string),
            'image/filename': tf.io.FixedLenFeature([], tf.string),
            #'image/class/size': tf.io.FixedLenFeature([], tf.int64),
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
        }
    )
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = tf.image.decode_jpeg(example['image/encoded'], channels=3)
    image = preprocess_fn(image, image_size)
    image = normalize_fn(image)
    
    label = tf.one_hot(example['image/class/label'], class_size, dtype=tf.float32)
    #filename = example['image/filename']
    return image, label


def read_tfrecord_val(example, image_size, class_size, normalize_fn, preprocess_fn):
    tfrecord_format = ( 
        {
            # Must be consistent with imagenet_to_gcs.py.
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
            'image/colorspace': tf.io.FixedLenFeature([], tf.string),
            'image/channels': tf.io.FixedLenFeature([], tf.int64),
            'image/class/label': tf.io.FixedLenFeature([], tf.int64),
            'image/class/synset': tf.io.FixedLenFeature([], tf.string),
            'image/format': tf.io.FixedLenFeature([], tf.string),
            'image/filename': tf.io.FixedLenFeature([], tf.string),
            'image/class/size': tf.io.FixedLenFeature([], tf.int64),
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
        }
    )
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = tf.image.decode_jpeg(example['image/encoded'], channels=3)
    image = preprocess_fn(image, image_size)
    image = normalize_fn(image)

    lists = tf.one_hot(example['image/class/size'], class_size, dtype=tf.float32)
    return image, lists

def get_train_dataset(path, image_size, class_size, args):
    batch_size = args.batch_size
    workers = args.workers if args.workers > 0 else tf.data.AUTOTUNE
    normalize_fn = get_normalize_fn(args.arch)

    files = tf.io.gfile.glob(os.path.join(path, '*'))
    dataset = tf.data.Dataset.from_tensor_slices(files)
    dataset = dataset.apply(tf.data.experimental.ignore_errors())
    dataset = dataset.interleave(tf.data.TFRecordDataset,
                                 cycle_length=10,
                                 block_length=8)
    dataset = dataset.shuffle(buffer_size=4096, seed=None)
    dataset = dataset.repeat()
    dataset = dataset.map(partial(read_tfrecord,
                                  image_size=image_size,
                                  class_size=class_size,
                                  normalize_fn=normalize_fn,
                                  preprocess_fn=preprocess_for_train),
                          num_parallel_calls=workers)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def get_eval_dataset(path, image_size, class_size, args):
    batch_size = args.batch_size
    workers = args.workers if args.workers > 0 else tf.data.AUTOTUNE
    normalize_fn = get_normalize_fn(args.arch)

    files = tf.io.gfile.glob(os.path.join(path, '*'))
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.map(partial(read_tfrecord,
                                  image_size=image_size,
                                  class_size=class_size,
                                  normalize_fn=normalize_fn,
                                  preprocess_fn=preprocess_for_eval),
                          num_parallel_calls=workers)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset
