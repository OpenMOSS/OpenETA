#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${OPENETA_REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
# shellcheck source=service_lifecycle.sh
source "$REPO_ROOT/scripts/embodied/service_lifecycle.sh"

if [[ "$#" -ne 1 ]]; then
  printf '{"success":false,"reason":"episode_root_required"}\n'
  exit 2
fi
ROOT="$1"
[[ "$ROOT" = /* ]] || ROOT="$REPO_ROOT/$ROOT"
ROOT="$(realpath -m "$ROOT")"
[[ -f "$ROOT/episode.json" ]] || {
  printf '{"success":false,"reason":"not_an_episode_root"}\n'
  exit 2
}

TTL="${OPENETA_REPLAY_VISER_TTL_SECONDS:-900}"
existing_port=""
if [[ -f "$ROOT/services.tsv" ]]; then
  while IFS=$'\t' read -r role pid _pgid _start ports _command; do
    [[ "$role" == "viser-supervisor" && "$pid" =~ ^[0-9]+$ ]] || continue
    if kill -0 "$pid" 2>/dev/null; then
      existing_port="${ports%%,*}"
      break
    fi
  done <"$ROOT/services.tsv"
fi

if [[ -n "$existing_port" ]]; then
  printf '%s\n' "$(( $(date +%s) + TTL ))" >"$ROOT/.replay-lease-until"
  printf '{"success":true,"reused":true,"viser_url":"http://127.0.0.1:%s","ttl_seconds":%s}\n' \
    "$existing_port" "$TTL"
  exit 0
fi

VISER_PORT=""
CONTROL_PORT=""
for candidate in $(seq 9801 2 9997); do
  control=$((candidate + 1))
  if [[ -z "$(openeta_port_pids "$candidate" || true)" && -z "$(openeta_port_pids "$control" || true)" ]]; then
    VISER_PORT="$candidate"
    CONTROL_PORT="$control"
    break
  fi
done
[[ -n "$VISER_PORT" ]] || {
  printf '{"success":false,"reason":"no_free_viser_port_pair"}\n'
  exit 2
}

artifact_root="$ROOT/grasp-inspector-replay"
mkdir -p "$artifact_root"
setsid uv run --no-project --python 3.10 \
  "$REPO_ROOT/scripts/embodied/grasp_inspector_supervisor.py" \
  --episode-root "$ROOT" \
  --artifact-root "$artifact_root" \
  --viewer-python "${OPENETA_GRASPGENX_PYTHON:-python3}" \
  --viewer-pythonpath "${OPENETA_GRASPGENX_ROOT:-}" \
  --port "$VISER_PORT" \
  --control-port "$CONTROL_PORT" \
  --no-open-browser \
  >"$artifact_root.log" 2>&1 < /dev/null &
pid="$!"
openeta_register_service "$ROOT" viser-supervisor "$pid" "$VISER_PORT,$CONTROL_PORT"
printf '%s\n' "$(( $(date +%s) + TTL ))" >"$ROOT/.replay-lease-until"
printf '{"success":true,"reused":false,"viser_url":"http://127.0.0.1:%s","ttl_seconds":%s}\n' \
  "$VISER_PORT" "$TTL"
