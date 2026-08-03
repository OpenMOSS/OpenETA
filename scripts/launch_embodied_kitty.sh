#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
# shellcheck source=embodied/service_lifecycle.sh
source "$REPO_ROOT/scripts/embodied/service_lifecycle.sh"
ROOT_ARG="${1:-$REPO_ROOT/artifacts/embodied/kitty-$(date +%Y%m%d-%H%M%S)}"
SEED="${2:-17}"
STEPS="${3:-8}"
INTERVAL="${4:-0.5}"

if [[ "$ROOT_ARG" = /* ]]; then
  ROOT="$ROOT_ARG"
else
  ROOT="$REPO_ROOT/$ROOT_ARG"
fi
if ! command -v kitty >/dev/null 2>&1; then
  printf 'kitty is required for the three-pane launcher\n' >&2
  exit 127
fi
if [[ -e "$ROOT/events.jsonl" ]]; then
  printf 'Refusing to append to existing episode root: %s\n' "$ROOT" >&2
  exit 2
fi
mkdir -p "$ROOT"
openeta_claim_episode_root "$ROOT"
if ! "$REPO_ROOT/scripts/embodied/replay_dashboard.sh" start; then
  printf 'Warning: global Replay Hub failed to start; Kitty launch continues.\n' >&2
fi

export OPENETA_REPO_ROOT="$REPO_ROOT"
export OPENETA_EMBODIED_ROOT="$ROOT"
# Kitty panes use Kitty's graphics protocol directly.  If the launcher is
# invoked from tmux, propagating TMUX makes `kitten icat` probe tmux's
# passthrough settings instead of Kitty's; older tmux versions then emit
# `invalid option: allow-passthrough` and the pane appears to have no image.
# The visible Kitty session is intentionally independent of the parent tmux
# client, so do not pass tmux routing state into its children.
unset TMUX TMUX_PANE
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OPENETA_LIBERO_PYTHON="$REPO_ROOT/sim/venvs/libero/bin/python"
export OPENETA_LIBERO_ENV_ID="${OPENETA_LIBERO_ENV_ID:-openeta/libero_libero_spatial_task0-v0}"
export OPENETA_LIBERO_SEED="$SEED"
export OPENETA_LIBERO_STEPS="$STEPS"
export OPENETA_LIBERO_INTERVAL="$INTERVAL"
# Cartesian Operator primitives expand into many physical LIBERO steps.  Give
# visible interactive runs enough room for perception retries and transport;
# normal tests and evaluation retain the project default when launched outside
# this workflow.
export OPENETA_LIBERO_MAX_EPISODE_STEPS="${OPENETA_LIBERO_MAX_EPISODE_STEPS:-5000}"
export OPENETA_OPERATOR_IMAGE_WIDTH="${OPENETA_OPERATOR_IMAGE_WIDTH:-512}"
export OPENETA_OPERATOR_IMAGE_HEIGHT="${OPENETA_OPERATOR_IMAGE_HEIGHT:-512}"
export OPENETA_SIM_PORT="${OPENETA_SIM_PORT:-8765}"
export OPENETA_SIM_URL="${OPENETA_SIM_URL:-http://127.0.0.1:${OPENETA_SIM_PORT}/sse}"
# A viewer/control pair is owned by one episode.  Deriving it from the
# simulator port keeps concurrent Kitty sessions isolated: otherwise a newer
# run silently finds the previous run's Viser at 8082 and, correctly, rejects
# it as a stale scene.
export OPENETA_VISER_PORT="${OPENETA_VISER_PORT:-$((OPENETA_SIM_PORT + 100))}"
export OPENETA_GRASP_INSPECTOR_PORT="${OPENETA_GRASP_INSPECTOR_PORT:-$((OPENETA_SIM_PORT + 101))}"
export OPENETA_GRASP_INSPECTOR_URL="${OPENETA_GRASP_INSPECTOR_URL:-http://127.0.0.1:${OPENETA_GRASP_INSPECTOR_PORT}}"
export OPENETA_EPISODE_WEB_PORT="${OPENETA_EPISODE_WEB_PORT:-$((OPENETA_SIM_PORT + 102))}"
export OPENETA_MCP_CONFIG="${OPENETA_MCP_CONFIG:-$REPO_ROOT/.mcp.json}"
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
export OPENETA_OPERATOR_TASK="${OPENETA_OPERATOR_TASK:-pick up the black bowl and place it on the plate}"
export LIBERO_DIR="${LIBERO_DIR:-$REPO_ROOT/third_party/LIBERO}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

printf 'Starting Kitty embodied layout\n  root=%s\n  env=%s\n  seed=%s steps=%s interval=%ss\n' \
  "$ROOT" "$OPENETA_LIBERO_ENV_ID" "$SEED" "$STEPS" "$INTERVAL"
printf '  Chrome dashboard=http://127.0.0.1:%s\n' "$OPENETA_EPISODE_WEB_PORT"
printf '  Stop command=%s/scripts/embodied/stop_episode_services.sh %s\n' "$REPO_ROOT" "$ROOT"
cd "$REPO_ROOT"
kitty_args=()
if [[ "${OPENETA_KITTY_DETACH:-0}" == "1" ]]; then
  printf 'OPENETA_KITTY_DETACH=1 is deprecated and no longer launches an episode.\n' >&2
  printf 'Use the Chrome-first launcher instead:\n  %s/scripts/embodied/launch_operator_chrome.sh %s\n' \
    "$REPO_ROOT" "$ROOT" >&2
  exit 2
fi
cleanup() {
  # Closing Kitty ends the world-changing runtime but intentionally leaves
  # Chrome observability/replay alive for its terminal TTL. Durable artifacts
  # remain browseable through the global Replay Hub after that process exits.
  openeta_stop_episode_execution_services "$ROOT"
  if [[ "${OPENETA_OPERATOR_ROOT_OWNED:-0}" == "1" ]]; then
    case "$OPENETA_OPERATOR_ROOT" in
      "${XDG_RUNTIME_DIR:-/tmp}"/openeta-operator-workspace-*)
        rm -rf -- "$OPENETA_OPERATOR_ROOT"
        ;;
      *)
        printf 'Refusing to remove non-OpenETA Operator workspace: %s\n' \
          "$OPENETA_OPERATOR_ROOT" >&2
        ;;
    esac
  fi
}
trap cleanup EXIT INT TERM
kitty "${kitty_args[@]}" --session "$REPO_ROOT/scripts/embodied/kitty-session.conf"
