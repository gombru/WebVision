#!/usr/bin/env sh
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs
set -e

EXAMPLE=/home/rgomez/datasets/WebVision/example_lmdb #outputpath
TRAIN_TXT=/home/rgomez/datasets/WebVision/info/train_filelist_google_1000.txt #filelist and labels txt path

TOOLS=/home/rgomez/caffe-master/build/tools #caffe path

DATA_ROOT=/home/rgomez/datasets/WebVision/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$DATA_ROOT" ]; then
  echo "Error: DATA_ROOT is not a path to a directory: $DATA_ROOT"
  echo "Set the DATA_ROOT variable in create_imagenet.sh to the path" \
       "where the ImageNet training data is stored."
  exit 1
fi

echo "Creating lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $DATA_ROOT \
    $TRAIN_TXT \
    $EXAMPLE

echo "Done."
