#!/bin/sh
DEVICE_ID=1  # which GPU is going to be used
export CUDA_VISIBLE_DEVICES=${DEVICE_ID}

apptainer exec --nv -B /sda2:/sda2 apptainer.sif python3 "main.py"