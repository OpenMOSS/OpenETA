#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${OPENETA_REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
# shellcheck source=service_lifecycle.sh
source "$REPO_ROOT/scripts/embodied/service_lifecycle.sh"
PYTHON_BIN="${OPENETA_LIBERO_PYTHON:-$REPO_ROOT/sim/venvs/libero/bin/python}"
ROOT="${OPENETA_EMBODIED_ROOT:?OPENETA_EMBODIED_ROOT is required}"
SIM_PORT="${OPENETA_SIM_PORT:-8765}"

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# The simulator process owns environment construction, so the horizon must be
# present here even when the Gateway/manual harness is launched separately.
export OPENETA_LIBERO_MAX_EPISODE_STEPS="${OPENETA_LIBERO_MAX_EPISODE_STEPS:-5000}"
mkdir -p "$ROOT"
SERVER_LOG="$ROOT/simulator.log"
server_pid=""
cleanup() {
  if [[ -n "$server_pid" ]] && kill -0 "$server_pid" 2>/dev/null; then
    openeta_stop_owned_process "$ROOT" "$server_pid" \
      "$(openeta_process_start_ticks "$server_pid" 2>/dev/null || true)" \
      "$(openeta_process_group "$server_pid")" || true
    wait "$server_pid" 2>/dev/null || true
  fi
  [[ -n "$server_pid" ]] && openeta_unregister_service "$ROOT" "$server_pid"
}
trap cleanup EXIT INT TERM

openeta_assert_or_reclaim_port "$ROOT" "$SIM_PORT" "LIBERO simulator MCP"

setsid uv run --no-project --python "$PYTHON_BIN" \
  -m sim.mcp_server \
  --transport sse \
  --port "$SIM_PORT" \
  >"$SERVER_LOG" 2>&1 &
server_pid=$!
openeta_register_service "$ROOT" simulator "$server_pid" "$SIM_PORT"

uv run --no-project --python 3.10 \
  "$REPO_ROOT/scripts/embodied/kitty_simulator_view.py" \
  --root "$ROOT" \
  --server-pid "$server_pid" \
  --port "$SIM_PORT" \
  --interval "${OPENETA_SIM_VIEW_INTERVAL:-0.25}"
