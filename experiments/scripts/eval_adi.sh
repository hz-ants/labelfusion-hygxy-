#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

python3 ./tools/eval_adi.py --dataset_root ./datasets/\
  --model trained_models/pose_model_current.pth\
  --refine_model trained_models/pose_refine_model_current.pth
