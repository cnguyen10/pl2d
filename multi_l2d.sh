#!/bin/sh
DEVICE_ID=1  # which GPU is going to be used
export CUDA_VISIBLE_DEVICES=${DEVICE_ID}

python3 -m multi_l2d.multi_l2d