#!/bin/bash

export IMAGENET_HOME=`pwd`
python dog_cat_to_gcs.py \
  --raw_data_dir=$IMAGENET_HOME \
  --local_scratch_dir=$IMAGENET_HOME/../tfrecords_dogs_cats\
  --nogcs_upload
