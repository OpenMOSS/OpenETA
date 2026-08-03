#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${OPENETA_REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
# shellcheck source=service_lifecycle.sh
source "$REPO_ROOT/scripts/embodied/service_lifecycle.sh"
ROOT="${OPENETA_EMBODIED_ROOT:?OPENETA_EMBODIED_ROOT is required}"
WEB_PORT="${OPENETA_EPISODE_WEB_PORT:-8890}"
VISER_PORT="${OPENETA_VISER_PORT:-8081}"
INSPECTOR_PORT="${OPENETA_GRASP_INSPECTOR_PORT:-8082}"

cd "$REPO_ROOT"
supervisor_pid=""
dashboard_pid=""
startup_complete=0
openeta_assert_or_reclaim_port "$ROOT" "$WEB_PORT" "episode Chrome dashboard"
if [[ "${OPENETA_VISER_SUPERVISOR:-0}" == "1" ]]; then
  openeta_assert_or_reclaim_port "$ROOT" "$VISER_PORT" "Viser viewer"
  openeta_assert_or_reclaim_port "$ROOT" "$INSPECTOR_PORT" "grasp inspector control"
fi

setsid uv run --no-project --python 3.10 \
  "$REPO_ROOT/scripts/embodied/episode_web_dashboard.py" \
  --root "$ROOT" \
  --port "$WEB_PORT" \
  --viser-url "http://127.0.0.1:$VISER_PORT" \
  --control-url "http://127.0.0.1:${OPENETA_GATEWAY_CONTROL_PORT:-8790}" \
  >"$ROOT/episode-web.log" 2>&1 &
dashboard_pid="$!"
openeta_register_service "$ROOT" dashboard "$dashboard_pid" "$WEB_PORT"
if [[ "${OPENETA_VISER_SUPERVISOR:-0}" == "1" ]]; then
  supervisor_root="${OPENETA_VISER_ARTIFACT_ROOT:-$ROOT/grasp-inspector}"
  setsid uv run --no-project --python 3.10 \
    "$REPO_ROOT/scripts/embodied/grasp_inspector_supervisor.py" \
    --episode-root "$ROOT" \
    --artifact-root "$supervisor_root" \
    --viewer-python "${OPENETA_GRASPGENX_PYTHON:-python3}" \
    --viewer-pythonpath "${OPENETA_GRASPGENX_ROOT:-}" \
    --port "$VISER_PORT" \
    --control-port "$INSPECTOR_PORT" \
    >"$supervisor_root.log" 2>&1 &
  supervisor_pid="$!"
  openeta_register_service "$ROOT" viser-supervisor "$supervisor_pid" "$VISER_PORT,$INSPECTOR_PORT"
fi
cleanup_failed_startup() {
  [[ "$startup_complete" == "0" ]] || return 0
  if [[ -n "$supervisor_pid" ]]; then
    openeta_stop_owned_process "$ROOT" "$supervisor_pid" \
      "$(openeta_process_start_ticks "$supervisor_pid" 2>/dev/null || true)" \
      "$(openeta_process_group "$supervisor_pid")" || true
    openeta_unregister_service "$ROOT" "$supervisor_pid"
  fi
  if [[ -n "$dashboard_pid" ]]; then
    openeta_stop_owned_process "$ROOT" "$dashboard_pid" \
      "$(openeta_process_start_ticks "$dashboard_pid" 2>/dev/null || true)" \
      "$(openeta_process_group "$dashboard_pid")" || true
    openeta_unregister_service "$ROOT" "$dashboard_pid"
  fi
}
trap cleanup_failed_startup EXIT INT TERM
printf 'Episode Chrome dashboard: http://127.0.0.1:%s\n' "$WEB_PORT"
printf 'Per-episode Dashboard remains available for the terminal replay TTL.\n'
printf 'Durable Replay Hub: http://127.0.0.1:%s\n' \
  "${OPENETA_REPLAY_HUB_PORT:-9295}"
printf 'Explicit stop: %s/scripts/embodied/stop_episode_services.sh %s\n' \
  "$REPO_ROOT" "$ROOT"
startup_complete=1
uv run --no-project --python 3.10 \
  "$REPO_ROOT/scripts/embodied/kitty_logger_view.py" \
  --root "$ROOT" \
  --interval "${OPENETA_LOGGER_INTERVAL:-0.25}"
