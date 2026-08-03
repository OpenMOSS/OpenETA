#!/usr/bin/env bash

# Shared lifecycle helpers for local embodied services.
#
# The registry is deliberately line-oriented so it can be inspected and
# repaired without Python.  A process is stopped only when both its recorded
# Linux start time and its current command line still identify an OpenETA
# embodied service.  This prevents PID reuse and unrelated port occupants from
# being killed.

if [[ -z "${BASH_VERSION:-}" ]]; then
  printf 'service_lifecycle.sh requires Bash; execute its launcher/stop script instead of sourcing it from another shell.\n' >&2
  return 2 2>/dev/null || exit 2
fi

_openeta_lifecycle_source="${BASH_SOURCE[0]:-$0}"
OPENETA_LIFECYCLE_REPO_ROOT="${OPENETA_REPO_ROOT:-$(cd "$(dirname "$_openeta_lifecycle_source")/../.." && pwd)}"
unset _openeta_lifecycle_source

openeta_service_registry() {
  printf '%s/services.tsv\n' "$1"
}

openeta_process_start_ticks() {
  local pid="$1"
  [[ -r "/proc/$pid/stat" ]] || return 1
  awk '{print $22}' "/proc/$pid/stat"
}

openeta_process_command() {
  local pid="$1"
  [[ -r "/proc/$pid/cmdline" ]] || return 1
  tr '\0' ' ' <"/proc/$pid/cmdline"
}

openeta_process_group() {
  ps -o pgid= -p "$1" 2>/dev/null | tr -d ' '
}

openeta_is_embodied_process() {
  local pid="$1"
  local episode_root="${2:-}"
  local command
  command="$(openeta_process_command "$pid" 2>/dev/null || true)"
  [[ -n "$command" ]] || return 1

  case "$command" in
    *"$OPENETA_LIFECYCLE_REPO_ROOT/scripts/embodied/episode_web_dashboard.py"* | \
    *"scripts/embodied/episode_web_dashboard.py"* | \
    *"$OPENETA_LIFECYCLE_REPO_ROOT/scripts/embodied/replay_hub.py"* | \
    *"scripts/embodied/replay_hub.py"* | \
    *"$OPENETA_LIFECYCLE_REPO_ROOT/scripts/embodied/grasp_inspector_supervisor.py"* | \
    *"scripts/embodied/grasp_inspector_supervisor.py"* | \
    *"$OPENETA_LIFECYCLE_REPO_ROOT/scripts/embodied/operator_app_server.py"* | \
    *"scripts/embodied/operator_app_server.py"* | \
    *"-m tools.embodied_mcp_server"* | \
    *"-m sim.mcp_server"* | \
    *"$OPENETA_LIFECYCLE_REPO_ROOT/sim/bench_worker.py"* | \
    *"sim/bench_worker.py"*)
      ;;
    *)
      return 1
      ;;
  esac

  # Services whose command includes an episode root must agree with it.
  if [[ -n "$episode_root" ]] && [[ "$command" == *"--root "* || "$command" == *"--episode-root "* ]]; then
    [[ "$command" == *"$episode_root"* ]] || return 1
  fi
}

openeta_register_service() {
  local episode_root="$1"
  local role="$2"
  local pid="$3"
  local ports="${4:--}"
  local pgid start command registry
  pgid="$(openeta_process_group "$pid")"
  start="$(openeta_process_start_ticks "$pid")"
  command="$(openeta_process_command "$pid" | tr '\t\n' '  ')"
  registry="$(openeta_service_registry "$episode_root")"
  mkdir -p "$episode_root"
  (
    flock -x 9
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$role" "$pid" "$pgid" "$start" "$ports" "$command" >&9
  ) 9>>"$registry"
}

openeta_unregister_service() {
  local episode_root="$1"
  local pid="$2"
  local registry tmp
  registry="$(openeta_service_registry "$episode_root")"
  [[ -e "$registry" ]] || return 0
  tmp="${registry}.tmp.$$"
  (
    flock -x 9
    awk -F '\t' -v pid="$pid" '$2 != pid' "$registry" >"$tmp"
    mv "$tmp" "$registry"
  ) 9>>"${registry}.lock"
}

openeta_stop_owned_process() {
  local episode_root="$1"
  local pid="$2"
  local expected_start="${3:-}"
  local pgid="${4:-}"
  local current_start
  kill -0 "$pid" 2>/dev/null || return 0
  current_start="$(openeta_process_start_ticks "$pid" 2>/dev/null || true)"
  if [[ -n "$expected_start" && "$current_start" != "$expected_start" ]]; then
    printf 'Refusing to stop reused PID %s (start time changed).\n' "$pid" >&2
    return 1
  fi
  if ! openeta_is_embodied_process "$pid" "$episode_root"; then
    printf 'Refusing to stop PID %s: it is not a matching OpenETA embodied service.\n' "$pid" >&2
    return 1
  fi
  [[ -n "$pgid" ]] || pgid="$(openeta_process_group "$pid")"
  # Kill a whole session only when its leader is itself a recognized OpenETA
  # service. Older launches may share a shell/Kitty process group; killing
  # that group would be wider than the recorded service ownership.
  local stop_group=0
  if [[ -n "$pgid" ]] && openeta_is_embodied_process "$pgid" "$episode_root"; then
    stop_group=1
  fi
  if [[ "$stop_group" == "1" ]]; then
    kill -TERM -- "-$pgid" 2>/dev/null || true
  else
    kill -TERM "$pid" 2>/dev/null || true
  fi
  for _ in $(seq 1 50); do
    kill -0 "$pid" 2>/dev/null || return 0
    sleep 0.1
  done
  if [[ "$stop_group" == "1" ]]; then
    kill -KILL -- "-$pgid" 2>/dev/null || true
  else
    kill -KILL "$pid" 2>/dev/null || true
  fi
}

openeta_stop_episode_services() {
  local episode_root="$1"
  local registry line role pid pgid start ports command
  registry="$(openeta_service_registry "$episode_root")"
  if [[ ! -f "$registry" ]]; then
    rm -rf "$episode_root/.service-launch-claim" "$episode_root/.operator-launch-claim"
    return 0
  fi
  mapfile -t lines <"$registry"
  local index
  for ((index=${#lines[@]} - 1; index >= 0; index--)); do
    line="${lines[$index]}"
    IFS=$'\t' read -r role pid pgid start ports command <<<"$line"
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    openeta_stop_owned_process "$episode_root" "$pid" "$start" "$pgid" || true
  done
  : >"$registry"
  rm -rf "$episode_root/.service-launch-claim" "$episode_root/.operator-launch-claim"
}

openeta_service_is_observability() {
  local role="$1"
  case "$role" in
    dashboard | viser-supervisor)
      return 0
      ;;
    *)
      return 1
      ;;
  esac
}

openeta_stop_episode_execution_services() {
  # Stop world-changing/runtime services while deliberately retaining the
  # browser dashboard and replay viewer. The retained services are still
  # registered and can be removed later with openeta_stop_episode_services.
  local episode_root="$1"
  local registry line role pid pgid start ports command
  registry="$(openeta_service_registry "$episode_root")"
  if [[ ! -f "$registry" ]]; then
    rm -rf "$episode_root/.service-launch-claim" "$episode_root/.operator-launch-claim"
    return 0
  fi
  mapfile -t lines <"$registry"
  local index
  for ((index=${#lines[@]} - 1; index >= 0; index--)); do
    line="${lines[$index]}"
    IFS=$'\t' read -r role pid pgid start ports command <<<"$line"
    openeta_service_is_observability "$role" && continue
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    openeta_stop_owned_process "$episode_root" "$pid" "$start" "$pgid" || true
    openeta_unregister_service "$episode_root" "$pid"
  done
  rm -rf "$episode_root/.service-launch-claim" "$episode_root/.operator-launch-claim"
}

openeta_stop_episode_observability_services() {
  # Stop only the per-episode Dashboard/Viser processes. Durable artifacts are
  # untouched and remain available through the global Replay Hub.
  local episode_root="$1"
  local registry line role pid pgid start ports command
  registry="$(openeta_service_registry "$episode_root")"
  if [[ ! -f "$registry" ]]; then
    rm -f "$episode_root/.replay-lease-until"
    return 0
  fi
  mapfile -t lines <"$registry"
  local index
  for ((index=${#lines[@]} - 1; index >= 0; index--)); do
    line="${lines[$index]}"
    IFS=$'\t' read -r role pid pgid start ports command <<<"$line"
    openeta_service_is_observability "$role" || continue
    [[ "$pid" =~ ^[0-9]+$ ]] || continue
    openeta_stop_owned_process "$episode_root" "$pid" "$start" "$pgid" || true
    openeta_unregister_service "$episode_root" "$pid"
  done
  rm -f "$episode_root/.replay-lease-until"
}

openeta_port_pids() {
  local port="$1"
  lsof -nP -t -iTCP:"$port" -sTCP:LISTEN 2>/dev/null | sort -u
}

openeta_assert_or_reclaim_port() {
  local episode_root="$1"
  local port="$2"
  local label="$3"
  local pids pid pgid
  pids="$(openeta_port_pids "$port" || true)"
  [[ -z "$pids" ]] && return 0

  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    if ! openeta_is_embodied_process "$pid"; then
      printf 'Port %s (%s) is occupied by unrelated PID %s: %s\n' \
        "$port" "$label" "$pid" "$(openeta_process_command "$pid" 2>/dev/null || printf unknown)" >&2
      return 2
    fi
  done <<<"$pids"

  printf 'Reclaiming stale OpenETA service on port %s (%s): PID(s) %s\n' \
    "$port" "$label" "$(tr '\n' ' ' <<<"$pids")" >&2
  while IFS= read -r pid; do
    [[ -n "$pid" ]] || continue
    pgid="$(openeta_process_group "$pid")"
    openeta_stop_owned_process "" "$pid" "" "$pgid"
  done <<<"$pids"

  if [[ -n "$(openeta_port_pids "$port" || true)" ]]; then
    printf 'Port %s (%s) is still occupied after stale-service cleanup.\n' "$port" "$label" >&2
    return 2
  fi
}

openeta_claim_episode_root() {
  local episode_root="$1"
  local claim="$episode_root/.service-launch-claim"
  mkdir -p "$episode_root"
  if ! mkdir "$claim" 2>/dev/null; then
    local old_pid=""
    old_pid="$(cat "$claim/launcher.pid" 2>/dev/null || true)"
    if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
      printf 'Refusing duplicate launch for episode root %s (launcher PID %s).\n' \
        "$episode_root" "$old_pid" >&2
      return 2
    fi
    openeta_stop_episode_services "$episode_root"
    mkdir "$claim"
  fi
  printf '%s\n' "$$" >"$claim/launcher.pid"
  printf '%s\n' "$(date -Iseconds)" >"$claim/created_at"
}

openeta_claim_component() {
  local episode_root="$1"
  local component="$2"
  local claim="$episode_root/.${component}-launch-claim"
  mkdir -p "$episode_root"
  if ! mkdir "$claim" 2>/dev/null; then
    local old_pid=""
    old_pid="$(cat "$claim/launcher.pid" 2>/dev/null || true)"
    if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
      printf 'Refusing duplicate %s launch for episode root %s (launcher PID %s).\n' \
        "$component" "$episode_root" "$old_pid" >&2
      return 2
    fi
    rm -rf "$claim"
    mkdir "$claim"
  fi
  printf '%s\n' "$$" >"$claim/launcher.pid"
  printf '%s\n' "$(date -Iseconds)" >"$claim/created_at"
}
