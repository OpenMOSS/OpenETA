#!/usr/bin/env bash
# bash scripts/shell/eval_ftp1_batch.sh FTP1

# Batch-run FTP1 evaluation on UniVTAC tasks sequentially.
# Usage (from UniVTAC/):
#   bash scripts/shell/eval_ftp1_batch.sh
#   bash scripts/shell/eval_ftp1_batch.sh FTP1
#
# Optional overrides:
#   TOTAL_NUM=20 WORKERS=2 GPU=0 MODEL_NAME=MyModel bash scripts/shell/eval_ftp1_batch.sh
# Notes:
#   This script always runs in per-task mode:
#   TASKS[idx] / DOMAIN_NAMES[idx] / CKPT_DIRS[idx] / RUN_SUFFIXES[idx] are matched by index.

set -euo pipefail
cd "$(dirname "$0")/../.."

TOTAL_NUM=${TOTAL_NUM:-100}
WORKERS=${WORKERS:-1}
TASK_CONFIG=${TASK_CONFIG:-"contact.yml"}
GPU=${GPU:-"0"}
LOW_MEMORY=${LOW_MEMORY:-0}
NO_VIDEO=${NO_VIDEO:-0}
MODEL_NAME=${1:-${MODEL_NAME:-FTP1}}
SAVE_ROOT=${SAVE_ROOT:-"eval_results/${MODEL_NAME}"}
SAVE_INFER_INPUT_DIR=${SAVE_INFER_INPUT_DIR:-"${SAVE_ROOT}/infer_input_sample"}
ACTION_REP=${ACTION_REP:-auto}
CHUNK_FIRST_N=${CHUNK_FIRST_N:-20}
VIDEO_FREQ=${VIDEO_FREQ:-2}
RESUME=${RESUME:-0}
BATCH_LOG_DIR=${BATCH_LOG_DIR:-"${SAVE_ROOT}/_batch_logs"}


TASKS=(
  "pull_out_key"
  "insert_hole"
  "insert_tube"
  "lift_bottle"
  "lift_can"
  "put_bottle_in_shelf"
)

DOMAIN_NAMES=(
  "UniVTAC_pull_out_key"
  "UniVTAC_insert_hole"
  "UniVTAC_insert_tube"
  "UniVTAC_lift_bottle"
  "UniVTAC_lift_can"
  "UniVTAC_put_bottle"
)

CKPT_DIRS=(
  "/cephfs/shared/yuanchengbo/rtac1_cache/checkpoints/rtac1/FTP1_UniVTAC_pull_out_key_expert_gsmall_ftp1/19999"
  "/cephfs/shared/yuanchengbo/rtac1_cache/checkpoints/rtac/FTP1_UniVTAC_insert_hole_expert_gsmall_ftp1/19999"
  "/cephfs/shared/yuanchengbo/rtac1_cache/checkpoints/rtac/FTP1_UniVTAC_insert_tube_expert_gsmall_ftp1/19999"
  "/cephfs/shared/yuanchengbo/rtac1_cache/checkpoints/rtac/FTP1_UniVTAC_lift_bottle_expert_gsmall_ftp1/19999"
  "/cephfs/shared/yuanchengbo/rtac1_cache/checkpoints/rtac/FTP1_UniVTAC_lift_can_expert_gsmall_ftp1/19999"
  "/cephfs/shared/yuanchengbo/rtac1_cache/checkpoints/rtac/FTP1_UniVTAC_put_bottle_expert_gsmall_ftp1/19999"
)
RUN_SUFFIXES=(
  "FTP1_UniVTAC_pull_out_key_expert_gsmall_ftp1"
  "FTP1_UniVTAC_insert_hole_expert_gsmall_ftp1"
  "FTP1_UniVTAC_insert_tube_expert_gsmall_ftp1"
  "FTP1_UniVTAC_lift_bottle_expert_gsmall_ftp1"
  "FTP1_UniVTAC_lift_can_expert_gsmall_ftp1"
  "FTP1_UniVTAC_put_bottle_expert_gsmall_ftp1"
)

n=${#CKPT_DIRS[@]}
if [[ ${#TASKS[@]} -ne $n || ${#DOMAIN_NAMES[@]} -ne $n || ${#RUN_SUFFIXES[@]} -ne $n ]]; then
  echo "[eval_ftp1_batch.sh] array length mismatch for per-task mode" >&2
  echo "  TASKS=${#TASKS[@]} DOMAIN_NAMES=${#DOMAIN_NAMES[@]} CKPT_DIRS=${#CKPT_DIRS[@]} RUN_SUFFIXES=${#RUN_SUFFIXES[@]}" >&2
  exit 1
fi

echo "[eval_ftp1_batch.sh] model_name=$MODEL_NAME save_root=$SAVE_ROOT total_runs=$n total_num=$TOTAL_NUM workers=$WORKERS gpu=$GPU resume=$RESUME"

batch_ts=$(date +"%Y-%m-%d_%H-%M-%S")
mkdir -p "$BATCH_LOG_DIR"
exit_summary="$BATCH_LOG_DIR/batch_exit_codes_${batch_ts}.tsv"
printf "idx\ttask\tdomain\tcheckpoint_dir\trun_suffix\texit_code\tstatus\tduration_sec\tlog_file\n" > "$exit_summary"

overall_failed=0

for idx in "${!CKPT_DIRS[@]}"; do
  task_name="${TASKS[$idx]}"
  domain_name="${DOMAIN_NAMES[$idx]}"
  ckpt_dir="${CKPT_DIRS[$idx]}"
  run_suffix="${RUN_SUFFIXES[$idx]}"

  if [[ -z "$task_name" ]]; then
    echo "[eval_ftp1_batch.sh] empty task entry at idx=$idx" >&2
    exit 1
  fi
  if [[ -z "$domain_name" || -z "$ckpt_dir" || -z "$run_suffix" ]]; then
    echo "[eval_ftp1_batch.sh] empty domain/checkpoint/run_suffix at idx=$idx" >&2
    exit 1
  fi

  log_file="$BATCH_LOG_DIR/${batch_ts}_$((idx + 1))_${task_name}.log"
  echo "[eval_ftp1_batch.sh] [$((idx + 1))/$n] task=$task_name domain=$domain_name checkpoint_dir=$ckpt_dir run_suffix=$run_suffix log_file=$log_file"
  start_epoch=$(date +%s)

  set +e
  CHECKPOINT_DIR="$ckpt_dir" \
  DOMAIN_NAME="$domain_name" \
  RUN_SUFFIX="$run_suffix" \
  TASK_LIST="$task_name" \
  MODEL_NAME="$MODEL_NAME" \
  TOTAL_NUM="$TOTAL_NUM" \
  WORKERS="$WORKERS" \
  TASK_CONFIG="$TASK_CONFIG" \
  GPU="$GPU" \
  LOW_MEMORY="$LOW_MEMORY" \
  NO_VIDEO="$NO_VIDEO" \
  SAVE_INFER_INPUT_DIR="$SAVE_INFER_INPUT_DIR" \
  ACTION_REP="$ACTION_REP" \
  CHUNK_FIRST_N="$CHUNK_FIRST_N" \
  VIDEO_FREQ="$VIDEO_FREQ" \
  RESUME="$RESUME" \
  bash scripts/shell/eval_ftp1.sh 2>&1 | tee "$log_file"
  run_exit=${PIPESTATUS[0]}
  set -e

  end_epoch=$(date +%s)
  duration_sec=$((end_epoch - start_epoch))
  status_label="ok"
  if [[ $run_exit -ne 0 ]]; then
    status_label="failed"
    overall_failed=1
  fi

  printf "%d\t%s\t%s\t%s\t%s\t%d\t%s\t%d\t%s\n" \
    "$((idx + 1))" "$task_name" "$domain_name" "$ckpt_dir" "$run_suffix" "$run_exit" "$status_label" "$duration_sec" "$log_file" \
    >> "$exit_summary"
  echo "[eval_ftp1_batch.sh] [$((idx + 1))/$n] finished status=$status_label exit_code=$run_exit duration=${duration_sec}s"

done

echo "[eval_ftp1_batch.sh] all runs finished; exit_summary=$exit_summary"
if [[ $overall_failed -ne 0 ]]; then
  echo "[eval_ftp1_batch.sh] one or more runs failed" >&2
  exit 1
fi
