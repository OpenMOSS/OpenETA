#! /bin/bash

TASK_NAME=$1
CONFIG_NAME=${2:-"demo"}
GPU=${3:-0}
NUM_PROCESSES=${4:-3}

export CUDA_VISIBLE_DEVICES=$GPU
python scripts/parallel_collect_data.py $TASK_NAME $CONFIG_NAME \
    --workers=$NUM_PROCESSES