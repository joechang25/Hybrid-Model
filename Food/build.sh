#!/bin/bash

export IMAGENET_HOME=`pwd`
python food_to_gcs.py \
  --raw_data_dir=$IMAGENET_HOME \
  --local_scratch_dir=$IMAGENET_HOME/../tfrecords_food\
  --nogcs_upload
