#!/usr/bin/env bash
set -euo pipefail

# Start the Chrome-only manual harness.  This intentionally does not launch
# Codex Operator: the human can exercise the same observe/mark_point/move_to
# control path first.
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
# shellcheck source=service_lifecycle.sh
source "$REPO_ROOT/scripts/embodied/service_lifecycle.sh"
ROOT_ARG="${1:-$REPO_ROOT/artifacts/embodied/manual-$(date +%Y%m%d-%H%M%S)}"
ROOT="$ROOT_ARG"
[[ "$ROOT" = /* ]] || ROOT="$REPO_ROOT/$ROOT"
if [[ -e "$ROOT/events.jsonl" ]]; then
  printf 'Refusing to append to existing episode root: %s\n' "$ROOT" >&2
  exit 2
fi
mkdir -p "$ROOT"

SIM_URL="${OPENETA_SIM_URL:-http://127.0.0.1:8825/sse}"
SIM_PORT="${OPENETA_SIM_PORT:-8825}"
WEB_PORT="${OPENETA_EPISODE_WEB_PORT:-8923}"
CONTROL_PORT="${OPENETA_GATEWAY_CONTROL_PORT:-8790}"
MCP_PORT="${OPENETA_MANUAL_MCP_PORT:-8781}"
TASK="${OPENETA_OPERATOR_TASK:-pick up the black bowl between the plate and the ramekin and place it on the plate}"
ENV_ID="${OPENETA_LIBERO_ENV_ID:-openeta/libero_libero_spatial_task0-v0}"
SEED="${OPENETA_LIBERO_SEED:-17}"

export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# Manual point/move validation exercises several closed-loop Cartesian
# primitives. Match the Operator launcher instead of silently inheriting the
# 500-step benchmark horizon from an independently started simulator.
export OPENETA_LIBERO_MAX_EPISODE_STEPS="${OPENETA_LIBERO_MAX_EPISODE_STEPS:-5000}"
export OPENETA_OPERATOR_POINTCLOUD_MODE="${OPENETA_OPERATOR_POINTCLOUD_MODE:-single-agentview}"
export OPENETA_OPERATOR_CONTEXT_PROFILE="${OPENETA_OPERATOR_CONTEXT_PROFILE:-openeta-light}"
printf '%s\n' "$ROOT" > /tmp/openeta-current-root

openeta_claim_episode_root "$ROOT"
if ! "$REPO_ROOT/scripts/embodied/replay_dashboard.sh" start; then
  printf 'Warning: global Replay Hub failed to start; manual launch continues.\n' >&2
fi
cleanup() {
  # Keep the completed or interrupted episode inspectable in Chrome. Runtime
  # services stop here; the explicit stop command printed below performs final
  # observability cleanup. The global Replay Hub keeps artifacts available
  # after the per-episode Dashboard reaches its terminal TTL.
  openeta_stop_episode_execution_services "$ROOT"
}
trap cleanup EXIT INT TERM

openeta_assert_or_reclaim_port "$ROOT" "$MCP_PORT" "manual Gateway MCP"
openeta_assert_or_reclaim_port "$ROOT" "$CONTROL_PORT" "manual Gateway control"
openeta_assert_or_reclaim_port "$ROOT" "$WEB_PORT" "manual Chrome dashboard"

setsid "$REPO_ROOT/.venv/bin/python3" -u -m tools.embodied_mcp_server \
  --transport sse --host 127.0.0.1 --port "$MCP_PORT" \
  --root "$ROOT" --env-id "$ENV_ID" --task "$TASK" --seed "$SEED" \
  --image-width 512 --image-height 512 --sim-url "$SIM_URL" \
  --control-port "$CONTROL_PORT" >"$ROOT/gateway.log" 2>&1 < /dev/null &
gateway_pid="$!"
echo "$gateway_pid" > "$ROOT/gateway.pid"
openeta_register_service "$ROOT" gateway "$gateway_pid" "$MCP_PORT,$CONTROL_PORT"

setsid "$REPO_ROOT/.venv/bin/python3" -u "$REPO_ROOT/scripts/embodied/episode_web_dashboard.py" \
  --root "$ROOT" --host 0.0.0.0 --port "$WEB_PORT" \
  --viser-url "http://127.0.0.1:$((SIM_PORT + 100))" \
  --control-url "http://127.0.0.1:$CONTROL_PORT" >"$ROOT/dashboard.log" 2>&1 < /dev/null &
dashboard_pid="$!"
echo "$dashboard_pid" > "$ROOT/dashboard.pid"
openeta_register_service "$ROOT" dashboard "$dashboard_pid" "$WEB_PORT"

sleep 1
if ! kill -0 "$gateway_pid" 2>/dev/null; then
  printf 'Manual Gateway exited during startup. See %s\n' "$ROOT/gateway.log" >&2
  exit 2
fi
if ! kill -0 "$dashboard_pid" 2>/dev/null; then
  printf 'Manual dashboard exited during startup. See %s\n' "$ROOT/dashboard.log" >&2
  exit 2
fi
printf 'Manual Chrome harness is ready (Operator is NOT running).\n'
printf 'dashboard: http://127.0.0.1:%s\n' "$WEB_PORT"
printf 'replay hub: http://127.0.0.1:%s\n' "${OPENETA_REPLAY_HUB_PORT:-9295}"
printf 'viser:     http://127.0.0.1:%s\n' "$((SIM_PORT + 100))"
printf 'root:      %s\n' "$ROOT"
printf 'first action: click observe / refresh RGB-D\n'
printf 'stop:      %s/scripts/embodied/stop_episode_services.sh %s\n' "$REPO_ROOT" "$ROOT"

# Remain the execution lifecycle owner. Closing or interrupting this launcher
# stops the Gateway while preserving the dashboard as a read-only/offline
# replay. The explicit stop command tears down both execution and observability.
while kill -0 "$gateway_pid" 2>/dev/null && kill -0 "$dashboard_pid" 2>/dev/null; do
  wait -n "$gateway_pid" "$dashboard_pid" 2>/dev/null || true
done
