#!/bin/bash
TASK_NAME=${1}
TASK_CONFIG=${2}
POLICY_CONIFG=${3}
GPU=${4}

export CUDA_VISIBLE_DEVICES=$GPU
python scripts/eval_policy.py $TASK_NAME $TASK_CONFIG $POLICY_CONIFG

# bash eval_policy.sh TactileACT/deploy_policy_insert_lean 0