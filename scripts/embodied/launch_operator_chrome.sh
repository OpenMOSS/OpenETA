#!/usr/bin/env bash
set -euo pipefail

# Chrome-first embodied Operator launcher.
#
# This owns one fresh LIBERO simulator, one episode dashboard/Viser pair, one
# persistent Gateway, and one fresh Codex Operator. It intentionally does not
# create a Kitty window or terminal-image mirror. When the Operator exits, all
# world-changing services stop while the Chrome replay remains available.

REPO_ROOT="${OPENETA_REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
# shellcheck source=service_lifecycle.sh
source "$REPO_ROOT/scripts/embodied/service_lifecycle.sh"

ROOT_ARG="${1:-$REPO_ROOT/artifacts/embodied/operator-chrome-$(date +%Y%m%d-%H%M%S)}"
ROOT="$ROOT_ARG"
[[ "$ROOT" = /* ]] || ROOT="$REPO_ROOT/$ROOT"
if [[ -e "$ROOT/events.jsonl" ]]; then
  printf 'Refusing to append to existing episode root: %s\n' "$ROOT" >&2
  exit 2
fi
mkdir -p "$ROOT"
openeta_claim_episode_root "$ROOT"
if ! "$REPO_ROOT/scripts/embodied/replay_dashboard.sh" start; then
  printf 'Warning: global Replay Hub failed to start; episode launch continues.\n' >&2
fi

SIM_PORT="${OPENETA_SIM_PORT:-9320}"
MCP_PORT="${OPENETA_OPERATOR_MCP_PORT:-$((SIM_PORT + 1))}"
CONTROL_PORT="${OPENETA_GATEWAY_CONTROL_PORT:-$((SIM_PORT + 2))}"
VISER_PORT="${OPENETA_VISER_PORT:-$((SIM_PORT + 3))}"
INSPECTOR_PORT="${OPENETA_GRASP_INSPECTOR_PORT:-$((SIM_PORT + 4))}"
WEB_PORT="${OPENETA_EPISODE_WEB_PORT:-$((SIM_PORT + 5))}"

export OPENETA_REPO_ROOT="$REPO_ROOT"
export OPENETA_EMBODIED_ROOT="$ROOT"
export OPENETA_GATEWAY_ROOT="$ROOT"
EPISODE_ROOT_ID="$(
  printf '%s' "$ROOT" | sha256sum | cut -c1-12
)"
if [[ -n "${OPENETA_OPERATOR_ROOT:-}" ]]; then
  export OPENETA_OPERATOR_ROOT_OWNED="${OPENETA_OPERATOR_ROOT_OWNED:-0}"
else
  export OPENETA_OPERATOR_ROOT="${XDG_RUNTIME_DIR:-/tmp}/openeta-operator-workspace-$(basename "$ROOT")-$EPISODE_ROOT_ID"
  export OPENETA_OPERATOR_ROOT_OWNED=1
fi
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OPENETA_LIBERO_PYTHON="${OPENETA_LIBERO_PYTHON:-$REPO_ROOT/sim/venvs/libero/bin/python}"
export OPENETA_LIBERO_ENV_ID="${OPENETA_LIBERO_ENV_ID:-openeta/libero_libero_spatial_task0-v0}"
export OPENETA_LIBERO_SEED="${OPENETA_LIBERO_SEED:-17}"
export OPENETA_OPERATOR_TASK="${OPENETA_OPERATOR_TASK:-pick up the black bowl between the plate and the ramekin and place it on the plate}"
export OPENETA_LIBERO_MAX_EPISODE_STEPS="${OPENETA_LIBERO_MAX_EPISODE_STEPS:-5000}"
export OPENETA_OPERATOR_IMAGE_WIDTH="${OPENETA_OPERATOR_IMAGE_WIDTH:-512}"
export OPENETA_OPERATOR_IMAGE_HEIGHT="${OPENETA_OPERATOR_IMAGE_HEIGHT:-512}"
export OPENETA_SIM_PORT="$SIM_PORT"
export OPENETA_SIM_URL="${OPENETA_SIM_URL:-http://127.0.0.1:$SIM_PORT/sse}"
export OPENETA_OPERATOR_MCP_PORT="$MCP_PORT"
export OPENETA_GATEWAY_CONTROL_PORT="$CONTROL_PORT"
export OPENETA_VISER_PORT="$VISER_PORT"
export OPENETA_GRASP_INSPECTOR_PORT="$INSPECTOR_PORT"
export OPENETA_GRASP_INSPECTOR_URL="${OPENETA_GRASP_INSPECTOR_URL:-http://127.0.0.1:$INSPECTOR_PORT}"
export OPENETA_EPISODE_WEB_PORT="$WEB_PORT"
export OPENETA_OPERATOR_MODEL="${OPENETA_OPERATOR_MODEL:-gpt-5.6-terra}"
export OPENETA_OPERATOR_POINTCLOUD_MODE="${OPENETA_OPERATOR_POINTCLOUD_MODE:-live-multiview-consensus}"
export OPENETA_OPERATOR_CONTEXT_PROFILE="${OPENETA_OPERATOR_CONTEXT_PROFILE:-openeta-light}"
export OPENETA_MCP_CONFIG="${OPENETA_MCP_CONFIG:-$REPO_ROOT/.mcp.json}"
export LIBERO_DIR="${LIBERO_DIR:-$REPO_ROOT/third_party/LIBERO}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

simulator_wrapper_pid=""
logger_wrapper_pid=""
operator_pid=""

cleanup_operator_workspace() {
  [[ "${OPENETA_OPERATOR_ROOT_OWNED:-0}" == "1" ]] || return 0
  case "$OPENETA_OPERATOR_ROOT" in
    "${XDG_RUNTIME_DIR:-/tmp}"/openeta-operator-workspace-*)
      rm -rf -- "$OPENETA_OPERATOR_ROOT"
      ;;
    *)
      printf 'Refusing to remove non-OpenETA Operator workspace: %s\n' \
        "$OPENETA_OPERATOR_ROOT" >&2
      return 1
      ;;
  esac
}

stop_wrapper() {
  local pid="$1"
  [[ -n "$pid" ]] || return 0
  kill -0 "$pid" 2>/dev/null || return 0
  kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 30); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 0.1
  done
  kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
}

cleanup() {
  local status=$?
  stop_wrapper "$operator_pid"
  stop_wrapper "$simulator_wrapper_pid"
  # logger_pane deliberately leaves dashboard/Viser alive after startup.
  stop_wrapper "$logger_wrapper_pid"
  openeta_stop_episode_execution_services "$ROOT"
  cleanup_operator_workspace
  printf 'Per-episode replay retained temporarily: http://127.0.0.1:%s\n' "$WEB_PORT"
  printf 'Durable Replay Hub: http://127.0.0.1:%s\n' \
    "${OPENETA_REPLAY_HUB_PORT:-9295}"
  printf 'Final cleanup: %s/scripts/embodied/stop_episode_services.sh %s\n' \
    "$REPO_ROOT" "$ROOT"
  return "$status"
}
trap cleanup EXIT INT TERM

for spec in \
  "$SIM_PORT:LIBERO simulator MCP" \
  "$MCP_PORT:Operator Gateway MCP" \
  "$CONTROL_PORT:Operator Gateway control" \
  "$VISER_PORT:Viser viewer" \
  "$INSPECTOR_PORT:Viser control" \
  "$WEB_PORT:Chrome dashboard"; do
  port="${spec%%:*}"
  label="${spec#*:}"
  openeta_assert_or_reclaim_port "$ROOT" "$port" "$label"
done

printf 'Starting Chrome-first embodied Operator\n'
printf '  root=%s\n' "$ROOT"
printf '  env=%s seed=%s\n' "$OPENETA_LIBERO_ENV_ID" "$OPENETA_LIBERO_SEED"
printf '  task=%s\n' "$OPENETA_OPERATOR_TASK"
printf '  dashboard=http://127.0.0.1:%s\n' "$WEB_PORT"

setsid bash "$REPO_ROOT/scripts/embodied/simulator_pane.sh" \
  >"$ROOT/simulator-pane.log" 2>&1 &
simulator_wrapper_pid="$!"

setsid bash "$REPO_ROOT/scripts/embodied/logger_pane.sh" \
  >"$ROOT/logger-pane.log" 2>&1 &
logger_wrapper_pid="$!"

dashboard_ready=0
for _ in $(seq 1 200); do
  if ! kill -0 "$simulator_wrapper_pid" 2>/dev/null; then
    printf 'Simulator launcher exited. See %s\n' "$ROOT/simulator-pane.log" >&2
    exit 2
  fi
  status="$(curl --noproxy '*' -sS -o /dev/null -w '%{http_code}' \
    --max-time 1 "http://127.0.0.1:$WEB_PORT/" 2>/dev/null || true)"
  if [[ "$status" == "200" ]]; then
    dashboard_ready=1
    break
  fi
  sleep 0.25
done
if [[ "$dashboard_ready" != "1" ]]; then
  printf 'Chrome dashboard did not become ready. See %s\n' "$ROOT/logger-pane.log" >&2
  exit 2
fi

printf 'Chrome dashboard ready: http://127.0.0.1:%s\n' "$WEB_PORT"
if [[ "${OPENETA_OPERATOR_FOREGROUND:-0}" == "1" ]]; then
  operator_status=0
  bash "$REPO_ROOT/scripts/embodied/operator_pane.sh" || operator_status=$?
  exit "$operator_status"
fi

setsid bash "$REPO_ROOT/scripts/embodied/operator_pane.sh" \
  >"$ROOT/operator-launch.log" 2>&1 &
operator_pid="$!"

operator_status=0
wait "$operator_pid" || operator_status=$?
operator_pid=""
exit "$operator_status"
