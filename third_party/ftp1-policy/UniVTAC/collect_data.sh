#! /bin/bash

TASK_NAME=$1
CONFIG_NAME=${2}
GPU=${3:-0}
START_SEED=${4:--1}
MAX_SEED=${5:--1}
EPISODE=${6:--1}

python scripts/collect_data.py \
    $TASK_NAME $CONFIG_NAME \
    --start_seed $START_SEED \
    --max_seed $MAX_SEED \
    --episode_num $EPISODE \
    --gpu $GPU