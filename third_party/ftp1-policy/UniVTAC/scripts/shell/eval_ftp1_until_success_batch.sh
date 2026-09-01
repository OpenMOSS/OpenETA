#!/usr/bin/env bash
# Run all configured UniVTAC FTP1 tasks sequentially until each task reaches N successes.
# This script is independent from the original eval_ftp1.sh / eval_ftp1.py pipeline.
#
# Usage:
#   bash scripts/shell/eval_ftp1_until_success_batch.sh
#   TARGET_SUCCESSES=5 GPU=0 bash scripts/shell/eval_ftp1_until_success_batch.sh FTP1

set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL_NAME=${1:-${MODEL_NAME:-FTP1}}
TARGET_SUCCESSES=${TARGET_SUCCESSES:-5}
MAX_EPISODES=${MAX_EPISODES:-0}
TASK_CONFIG=${TASK_CONFIG:-"contact.yml"}
GPU=${GPU:-"0"}
LOW_MEMORY=${LOW_MEMORY:-0}
SAVE_ROOT=${SAVE_ROOT:-"eval_results/${MODEL_NAME}_success_runs"}
ACTION_REP=${ACTION_REP:-auto}
CHUNK_FIRST_N=${CHUNK_FIRST_N:-20}
VIDEO_FREQ=${VIDEO_FREQ:-2}
VIDEO_FPS=${VIDEO_FPS:-10}
MAX_STEPS=${MAX_STEPS:-1000}
START_SEED=${START_SEED:-1000000}
NUM_INFERENCE_STEPS=${NUM_INFERENCE_STEPS:-10}
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
  "/cephfs/shared/yuanchengbo/ftp1_cache/checkpoints/ftp1/FTP50k_UniVTAC_pull_out_key_expert_gsmall_ftp1/19999"
  "/cephfs/shared/yuanchengbo/ftp1_cache/checkpoints/ftp1/FTP50k_UniVTAC_insert_hole_expert_gsmall_ftp1/19999"
  "/cephfs/shared/yuanchengbo/ftp1_cache/checkpoints/ftp1/FTP50k_UniVTAC_insert_tube_expert_gsmall_ftp1/19999"
  "/cephfs/shared/yuanchengbo/ftp1_cache/checkpoints/ftp1/FTP50k_UniVTAC_lift_bottle_expert_gsmall_ftp1/19999"
  "/cephfs/shared/yuanchengbo/ftp1_cache/checkpoints/ftp1/FTP50k_UniVTAC_lift_can_expert_gsmall_ftp1/19999"
  "/cephfs/shared/yuanchengbo/ftp1_cache/checkpoints/ftp1/FTP50k_UniVTAC_put_bottle_expert_gsmall_ftp1/19999"
)

RUN_SUFFIXES=(
  "FTP50k_UniVTAC_pull_out_key_expert_gsmall_ftp1"
  "FTP50k_UniVTAC_insert_hole_expert_gsmall_ftp1"
  "FTP50k_UniVTAC_insert_tube_expert_gsmall_ftp1"
  "FTP50k_UniVTAC_lift_bottle_expert_gsmall_ftp1"
  "FTP50k_UniVTAC_lift_can_expert_gsmall_ftp1"
  "FTP50k_UniVTAC_put_bottle_expert_gsmall_ftp1"
)

n=${#CKPT_DIRS[@]}
if [[ ${#TASKS[@]} -ne $n || ${#DOMAIN_NAMES[@]} -ne $n || ${#RUN_SUFFIXES[@]} -ne $n ]]; then
  echo "[eval_ftp1_until_success_batch.sh] array length mismatch" >&2
  echo "  TASKS=${#TASKS[@]} DOMAIN_NAMES=${#DOMAIN_NAMES[@]} CKPT_DIRS=${#CKPT_DIRS[@]} RUN_SUFFIXES=${#RUN_SUFFIXES[@]}" >&2
  exit 1
fi

echo "[eval_ftp1_until_success_batch.sh] model_name=$MODEL_NAME save_root=$SAVE_ROOT tasks=$n target_successes=$TARGET_SUCCESSES gpu=$GPU"

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

  if [[ -z "$task_name" || -z "$domain_name" || -z "$ckpt_dir" || -z "$run_suffix" ]]; then
    echo "[eval_ftp1_until_success_batch.sh] empty task config at idx=$idx" >&2
    exit 1
  fi

  log_file="$BATCH_LOG_DIR/${batch_ts}_$((idx + 1))_${task_name}.log"
  echo "[eval_ftp1_until_success_batch.sh] [$((idx + 1))/$n] task=$task_name domain=$domain_name checkpoint_dir=$ckpt_dir log_file=$log_file"
  start_epoch=$(date +%s)

  cmd=(
    python scripts/eval_ftp1_until_success.py
    --checkpoint_dir "$ckpt_dir"
    --domain_name "$domain_name"
    --task_name "$task_name"
    --task_config "$TASK_CONFIG"
    --target_successes "$TARGET_SUCCESSES"
    --start_seed "$START_SEED"
    --gpu "$GPU"
    --num_inference_steps "$NUM_INFERENCE_STEPS"
    --chunk_first_n "$CHUNK_FIRST_N"
    --action_rep "$ACTION_REP"
    --video_frequency "$VIDEO_FREQ"
    --video_fps "$VIDEO_FPS"
    --max_steps "$MAX_STEPS"
    --save_root "$SAVE_ROOT"
    --run_suffix "$run_suffix"
  )

  if [[ "$MAX_EPISODES" != "0" ]]; then
    cmd+=(--max_episodes "$MAX_EPISODES")
  fi
  if [[ "$LOW_MEMORY" == "1" ]]; then
    cmd+=(--low_memory)
  fi

  set +e
  "${cmd[@]}" 2>&1 | tee "$log_file"
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
  echo "[eval_ftp1_until_success_batch.sh] [$((idx + 1))/$n] finished status=$status_label exit_code=$run_exit duration=${duration_sec}s"
done

echo "[eval_ftp1_until_success_batch.sh] all runs finished; exit_summary=$exit_summary"
if [[ $overall_failed -ne 0 ]]; then
  echo "[eval_ftp1_until_success_batch.sh] one or more runs failed" >&2
  exit 1
fi
