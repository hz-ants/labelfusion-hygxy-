#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 ./tools/train.py  --dataset_root /home/fapsros/Desktop/posenetown/datasets\
  --resume_posenet pose_model_current.pth\
  --resume_refinenet pose_refine_model_current.pth\
  --start_epoch 454

