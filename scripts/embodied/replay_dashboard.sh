#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${OPENETA_REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
# shellcheck source=service_lifecycle.sh
source "$REPO_ROOT/scripts/embodied/service_lifecycle.sh"

EPISODES_ROOT="${OPENETA_REPLAY_ROOT:-$REPO_ROOT/artifacts/embodied}"
PORT="${OPENETA_REPLAY_HUB_PORT:-9295}"
TTL="${OPENETA_REPLAY_TERMINAL_TTL_SECONDS:-900}"
ORPHAN_TTL="${OPENETA_REPLAY_ORPHAN_TTL_SECONDS:-3600}"
STATE_ROOT="$EPISODES_ROOT/.replay-hub"
PID_FILE="$STATE_ROOT/pid"
START_FILE="$STATE_ROOT/start_ticks"
LOG_FILE="$STATE_ROOT/hub.log"

hub_pid() {
  [[ -r "$PID_FILE" ]] || return 1
  local pid
  pid="$(<"$PID_FILE")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  printf '%s\n' "$pid"
}

hub_running() {
  local pid expected current
  pid="$(hub_pid 2>/dev/null || true)"
  [[ -n "$pid" ]] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  expected="$(cat "$START_FILE" 2>/dev/null || true)"
  current="$(openeta_process_start_ticks "$pid" 2>/dev/null || true)"
  [[ -n "$expected" && "$expected" == "$current" ]] || return 1
  openeta_is_embodied_process "$pid"
}

start_hub() {
  if hub_running; then
    printf 'Replay Hub already running: http://127.0.0.1:%s\n' "$PORT"
    return 0
  fi
  mkdir -p "$STATE_ROOT"
  openeta_assert_or_reclaim_port "" "$PORT" "global Replay Hub"
  setsid uv run --no-project --python 3.10 \
    "$REPO_ROOT/scripts/embodied/replay_hub.py" \
    --episodes-root "$EPISODES_ROOT" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --terminal-ttl-seconds "$TTL" \
    --orphan-ttl-seconds "$ORPHAN_TTL" \
    >"$LOG_FILE" 2>&1 < /dev/null &
  local pid="$!"
  printf '%s\n' "$pid" >"$PID_FILE"
  printf '%s\n' "$(openeta_process_start_ticks "$pid")" >"$START_FILE"
  for _ in $(seq 1 100); do
    if curl --noproxy '*' -fsS --max-time 1 "http://127.0.0.1:$PORT/" >/dev/null 2>&1; then
      printf 'Replay Hub ready: http://127.0.0.1:%s\n' "$PORT"
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      printf 'Replay Hub exited during startup. See %s\n' "$LOG_FILE" >&2
      return 2
    fi
    sleep 0.1
  done
  printf 'Replay Hub did not become ready. See %s\n' "$LOG_FILE" >&2
  return 2
}

stop_hub() {
  if ! hub_running; then
    printf 'Replay Hub is not running.\n'
    return 0
  fi
  local pid start pgid
  pid="$(hub_pid)"
  start="$(<"$START_FILE")"
  pgid="$(openeta_process_group "$pid")"
  openeta_stop_owned_process "" "$pid" "$start" "$pgid"
  rm -f "$PID_FILE" "$START_FILE"
  printf 'Replay Hub stopped.\n'
}

resolve_episode() {
  local value="$1"
  [[ "$value" = /* ]] || value="$REPO_ROOT/$value"
  value="$(realpath -m "$value")"
  case "$value" in
    "$(realpath -m "$EPISODES_ROOT")"/*) ;;
    *) printf 'Episode must be under %s\n' "$EPISODES_ROOT" >&2; return 2 ;;
  esac
  [[ -f "$value/episode.json" ]] || {
    printf 'Not an episode root: %s\n' "$value" >&2
    return 2
  }
  printf '%s\n' "$value"
}

command="${1:-status}"
case "$command" in
  start)
    start_hub
    ;;
  stop)
    stop_hub
    ;;
  restart)
    stop_hub
    start_hub
    ;;
  status)
    if hub_running; then
      printf 'running pid=%s url=http://127.0.0.1:%s terminal_ttl_seconds=%s orphan_ttl_seconds=%s\n' \
        "$(hub_pid)" "$PORT" "$TTL" "$ORPHAN_TTL"
    else
      printf 'stopped\n'
      exit 1
    fi
    ;;
  sweep)
    uv run --no-project --python 3.10 \
      "$REPO_ROOT/scripts/embodied/replay_hub.py" \
      --episodes-root "$EPISODES_ROOT" \
      --terminal-ttl-seconds "$TTL" \
      --orphan-ttl-seconds "$ORPHAN_TTL" \
      --sweep-once
    ;;
  pin)
    root="$(resolve_episode "${2:?episode root required}")"
    touch "$root/.replay-pin"
    printf 'Pinned replay services for %s\n' "$root"
    ;;
  unpin)
    root="$(resolve_episode "${2:?episode root required}")"
    rm -f "$root/.replay-pin"
    printf 'Unpinned replay services for %s\n' "$root"
    ;;
  *)
    printf 'Usage: %s {start|stop|restart|status|sweep|pin <root>|unpin <root>}\n' "$0" >&2
    exit 2
    ;;
esac
