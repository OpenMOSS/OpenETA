#!/usr/bin/env bash
set -euo pipefail

# Start a fresh interactive Operator Codex.  This process deliberately does
# not use the historical OpenETA runtime, resume/fork, project memory, or the
# source repository as its working directory.  Its only external control
# surface is the one semantic embodied MCP gateway for this episode.

REPO_ROOT="${OPENETA_REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
# shellcheck source=service_lifecycle.sh
source "$REPO_ROOT/scripts/embodied/service_lifecycle.sh"
EPISODE_ROOT="${OPENETA_GATEWAY_ROOT:-${OPENETA_EMBODIED_ROOT:?OPENETA_EMBODIED_ROOT is required}}"
OPERATOR_ROOT="${OPENETA_OPERATOR_ROOT:-$EPISODE_ROOT/operator-workspace}"
EPISODE_NAME="$(basename "$EPISODE_ROOT")"
EPISODE_ROOT_ID="$(
  printf '%s' "$EPISODE_ROOT" | sha256sum | cut -c1-12
)"
if [[ -n "${OPENETA_OPERATOR_CODEX_HOME:-}" ]]; then
  OPERATOR_CODEX_HOME_OWNED="${OPENETA_OPERATOR_CODEX_HOME_OWNED:-0}"
else
  OPERATOR_CODEX_HOME_OWNED=1
fi
OPERATOR_CODEX_HOME="${OPENETA_OPERATOR_CODEX_HOME:-${XDG_RUNTIME_DIR:-/tmp}/openeta-codex-home-$EPISODE_NAME-$EPISODE_ROOT_ID}"
TASK="${OPENETA_OPERATOR_TASK:-pick up the black bowl between the plate and the ramekin and place it on the plate}"
export OPENETA_POINT_ONLY_OPERATOR="${OPENETA_POINT_ONLY_OPERATOR:-1}"
ENV_ID="${OPENETA_LIBERO_ENV_ID:-openeta/libero_libero_spatial_task0-v0}"
SEED="${OPENETA_LIBERO_SEED:-17}"
# Direct Operator launches (without the Kitty wrapper) must get the same
# interactive horizon as the visible launcher.  A LIBERO motion primitive can
# consume hundreds of physical steps; the benchmark default of 500 otherwise
# terminates the environment during recovery before the episode is finished.
export OPENETA_LIBERO_MAX_EPISODE_STEPS="${OPENETA_LIBERO_MAX_EPISODE_STEPS:-5000}"
IMAGE_WIDTH="${OPENETA_OPERATOR_IMAGE_WIDTH:-512}"
IMAGE_HEIGHT="${OPENETA_OPERATOR_IMAGE_HEIGHT:-512}"
MCP_CONFIG_PATH="${OPENETA_MCP_CONFIG:-$REPO_ROOT/.mcp.json}"
CODEX_BIN="${OPENETA_CODEX_BIN:-codex}"
CODEX_MODE="${OPENETA_CODEX_MODE:-interactive}"
OPERATOR_RUNTIME="${OPENETA_OPERATOR_RUNTIME:-app-server}"
OPERATOR_MODEL="${OPENETA_OPERATOR_MODEL:-gpt-5.6-terra}"
OPERATOR_REASONING_EFFORT="${OPENETA_OPERATOR_REASONING_EFFORT:-medium}"
OPERATOR_MODEL_PROVIDER="${OPENETA_OPERATOR_MODEL_PROVIDER:-}"
OPERATOR_PROVIDER_CONFIG="${OPENETA_OPERATOR_PROVIDER_CONFIG:-${HOME}/.codex/config.toml}"
OPERATOR_YOLO="${OPENETA_OPERATOR_YOLO:-1}"
OPERATOR_PROMPT="${OPENETA_OPERATOR_PROMPT:-}"
OPERATOR_CONTEXT_PROFILE="${OPENETA_OPERATOR_CONTEXT_PROFILE:-openeta-light}"
GATEWAY_HOST="${OPENETA_GATEWAY_HOST:-127.0.0.1}"
GATEWAY_PORT="${OPENETA_OPERATOR_MCP_PORT:-8780}"
GATEWAY_CONTROL_PORT="${OPENETA_GATEWAY_CONTROL_PORT:-8790}"

openeta_claim_component "$EPISODE_ROOT" operator

mcp_url() {
  local name="$1"
  local fallback="$2"
  local value
  value="$(
    PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
      uv run --no-project --python "$REPO_ROOT/.venv/bin/python" python -c '
import sys
from agent.tools.mcp_registry import load_mcp_server_url

print(load_mcp_server_url(sys.argv[1], path=sys.argv[2]))
' "$name" "$MCP_CONFIG_PATH" 2>/dev/null || true
  )"
  printf '%s\n' "${value:-$fallback}"
}

# The Kitty simulator pane owns a local LIBERO server.  Perception services
# are normally remote registrations from .mcp.json, with explicit environment
# variables taking precedence for deployments and CI.
SIM_URL="${OPENETA_SIM_URL:-http://127.0.0.1:8765/sse}"
SAM3_URL="${OPENETA_SAM3_URL:-$(mcp_url openeta-sam3 http://127.0.0.1:8773/sse)}"
ANYGRASP_URL="${OPENETA_ANYGRASP_URL:-$(mcp_url openeta-anygrasp http://127.0.0.1:8774/sse)}"
GRASP_INSPECTOR_URL="${OPENETA_GRASP_INSPECTOR_URL:-http://127.0.0.1:${OPENETA_GRASP_INSPECTOR_PORT:-8082}}"

cleanup_operator_codex_home() {
  [[ "$OPERATOR_CODEX_HOME_OWNED" == "1" ]] || return 0
  case "$OPERATOR_CODEX_HOME" in
    "${XDG_RUNTIME_DIR:-/tmp}"/openeta-codex-home-*)
      rm -rf -- "$OPERATOR_CODEX_HOME"
      ;;
    *)
      printf 'Refusing to remove non-OpenETA Operator CODEX_HOME: %s\n' \
        "$OPERATOR_CODEX_HOME" >&2
      return 1
      ;;
  esac
}

mkdir -p "$OPERATOR_ROOT" "$OPERATOR_CODEX_HOME"
trap cleanup_operator_codex_home EXIT INT TERM

# A listening TCP port is not sufficient evidence that it is the requested
# simulator: old Viser control servers also use ordinary HTTP ports. Refuse to
# start a fresh Operator until the configured SSE endpoint completes an MCP
# handshake and advertises the simulator's observe tool.
probe_simulator_mcp() {
  timeout "${OPENETA_SIM_READY_PROBE_TIMEOUT_S:-8}" \
    env -u ALL_PROXY -u all_proxy -u HTTP_PROXY -u http_proxy \
      -u HTTPS_PROXY -u https_proxy \
      PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
      uv run --no-project --python "$REPO_ROOT/.venv/bin/python" python -c '
import asyncio
import sys

from mcp import ClientSession
from mcp.client.sse import sse_client

async def main():
    async with sse_client(sys.argv[1]) as streams:
        async with ClientSession(streams[0], streams[1]) as session:
            await session.initialize()
            tools = {tool.name for tool in (await session.list_tools()).tools}
            required = {"create_env", "observe_env", "move_to"}
            missing = sorted(required - tools)
            if missing:
                raise SystemExit("missing simulator tools: " + ",".join(missing))

asyncio.run(main())
' "$SIM_URL"
}

simulator_ready=0
simulator_ready_deadline=$((SECONDS + ${OPENETA_SIM_READY_TIMEOUT_S:-90}))
while (( SECONDS < simulator_ready_deadline )); do
  if probe_simulator_mcp >/dev/null 2>&1; then
    simulator_ready=1
    break
  fi
  sleep "${OPENETA_SIM_READY_INTERVAL_S:-0.5}"
done
if [[ "$simulator_ready" != "1" ]]; then
  printf 'Refusing to start Operator: simulator MCP is not ready at %s or does not advertise observe.\n' \
    "$SIM_URL" >&2
  exit 2
fi

# Codex requires a trusted working root.  Initialize only this empty operator
# workspace; it is not the source repository and contains no repository history.
if [[ ! -d "$OPERATOR_ROOT/.git" ]]; then
  git -C "$OPERATOR_ROOT" init -q
fi

if [[ -e "$OPERATOR_CODEX_HOME/config.toml" ]]; then
  printf 'Refusing a non-fresh Operator CODEX_HOME with config.toml: %s\n' "$OPERATOR_CODEX_HOME" >&2
  exit 2
fi

# Keep authentication available without inheriting ~/.codex/config.toml,
# saved sessions, memories, or unrelated MCP registrations.
USER_CODEX_HOME="${HOME}/.codex"
if [[ -f "$USER_CODEX_HOME/auth.json" && ! -e "$OPERATOR_CODEX_HOME/auth.json" ]]; then
  ln -s "$USER_CODEX_HOME/auth.json" "$OPERATOR_CODEX_HOME/auth.json"
fi

toml_string() {
  local value="$1"
  value="${value//\\/\\\\}"
  value="${value//\"/\\\"}"
  printf '"%s"' "$value"
}

toml_array() {
  local rendered="["
  local arg
  for arg in "$@"; do
    rendered+="$(toml_string "$arg"),"
  done
  rendered="${rendered%,}]"
  printf '%s' "$rendered"
}

# The interactive CLI asks whether a newly-created project is trusted before
# it accepts the initial prompt.  Keep that decision inside the isolated
# Operator CODEX_HOME instead of inheriting the user's global config or
# requiring a human to answer inside Kitty.
: > "$OPERATOR_CODEX_HOME/config.toml"
printf 'model_reasoning_effort = %s\n' \
  "$(toml_string "$OPERATOR_REASONING_EFFORT")" \
  >> "$OPERATOR_CODEX_HOME/config.toml"
if [[ -n "$OPERATOR_MODEL_PROVIDER" ]]; then
  uv run --no-project --python "$REPO_ROOT/.venv/bin/python" \
    "$REPO_ROOT/scripts/embodied/operator_provider_config.py" \
    --source "$OPERATOR_PROVIDER_CONFIG" \
    --provider "$OPERATOR_MODEL_PROVIDER" \
    >> "$OPERATOR_CODEX_HOME/config.toml"
  printf '\n' >> "$OPERATOR_CODEX_HOME/config.toml"
fi
printf '[projects.%s]\ntrust_level = "trusted"\n' \
  "$(toml_string "$OPERATOR_ROOT")" >> "$OPERATOR_CODEX_HOME/config.toml"

gateway_args=(
  run --no-project
  --python "$REPO_ROOT/.venv/bin/python"
  -m tools.embodied_mcp_server
  --transport stdio
  --root "$EPISODE_ROOT"
  --env-id "$ENV_ID"
  --task "$TASK"
  --seed "$SEED"
  --image-width "$IMAGE_WIDTH"
  --image-height "$IMAGE_HEIGHT"
  --sim-url "$SIM_URL"
  --sam3-url "$SAM3_URL"
  --anygrasp-url "$ANYGRASP_URL"
  --grasp-inspector-url "$GRASP_INSPECTOR_URL"
  --control-port "$GATEWAY_CONTROL_PORT"
)
# The parent shell uses a SOCKS proxy for the model gateway.  The MCP client
# inside this isolated child must connect directly to the simulator and the
# private SAM3/AnyGrasp hosts; otherwise httpx tries to construct a SOCKS
# transport even when NO_PROXY matches and fails if socksio is unavailable.
gateway_process_env=(
  -u ALL_PROXY -u all_proxy
  -u HTTP_PROXY -u http_proxy
  -u HTTPS_PROXY -u https_proxy
  "OPENETA_POINT_ONLY_OPERATOR=${OPENETA_POINT_ONLY_OPERATOR}"
  "OPENETA_OPERATOR_CONTEXT_PROFILE=${OPERATOR_CONTEXT_PROFILE}"
  "OPENETA_OPERATOR_POINTCLOUD_MODE=${OPENETA_OPERATOR_POINTCLOUD_MODE:-single-agentview}"
  "OPENETA_LIBERO_SETTLE_STEPS=${OPENETA_LIBERO_SETTLE_STEPS:-10}"
  # Preserve privileged object poses only in host-side episode artifacts.
  # They are not projected into the Operator tool response or prompt.
  "OPENETA_BUILDER_OBJECT_DIAGNOSTICS=${OPENETA_BUILDER_OBJECT_DIAGNOSTICS:-1}"
  "OPENETA_OPERATOR_ROOT=${OPERATOR_ROOT}"
)
gateway_env_args=(
  "${gateway_process_env[@]}"
  uv
  "${gateway_args[@]}"
)
gateway_args_toml="$(toml_array "${gateway_env_args[@]}")"

export CODEX_HOME="$OPERATOR_CODEX_HOME"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

printf '\033[1;36mClean Operator Codex\033[0m\n'
printf 'Task: %s\n' "$TASK"
printf 'Episode artifacts: %s\n' "$EPISODE_ROOT"
printf 'Simulator MCP: %s\n' "$SIM_URL"
printf 'SAM3 MCP: %s\n' "$SAM3_URL"
printf 'AnyGrasp MCP: %s\n' "$ANYGRASP_URL"
printf 'Grasp inspector control: %s\n' "$GRASP_INSPECTOR_URL"
printf 'Operator RGB-D: %sx%s\n' "$IMAGE_WIDTH" "$IMAGE_HEIGHT"
printf 'Codex model: %s\n' "$OPERATOR_MODEL"
printf 'Codex reasoning effort: %s\n' "$OPERATOR_REASONING_EFFORT"
printf 'Codex model provider: %s\n' "${OPERATOR_MODEL_PROVIDER:-openai}"
printf 'Codex runtime: %s\n' "$OPERATOR_RUNTIME"
printf 'Codex mode: %s\n' "$( [[ "$OPERATOR_YOLO" == "1" ]] && printf 'YOLO' || printf 'approval-aware' )"
printf 'Context profile: %s\n' "$OPERATOR_CONTEXT_PROFILE"
printf 'Operator MCP tools: observe, mark_point, move_to, check_task, report_issue, finish_episode.\n\n'

if [[ -z "$OPERATOR_PROMPT" ]]; then
  OPERATOR_PROMPT="$(
    OPENETA_OPERATOR_CONTEXT_PROFILE="$OPERATOR_CONTEXT_PROFILE" \
      uv run --no-project --python "$REPO_ROOT/.venv/bin/python" python -c '
import sys
from tools.operator_context_profiles import load_profile

print(load_profile(sys.argv[1]).render_prompt(sys.argv[2]))
' "$OPERATOR_CONTEXT_PROFILE" "$TASK"
  )"
fi
OPENETA_OPERATOR_CONTEXT_PROFILE="$OPERATOR_CONTEXT_PROFILE" \
  uv run --no-project --python "$REPO_ROOT/.venv/bin/python" python -c '
import pathlib
import sys
from tools.operator_context_profiles import load_profile, write_initial_contract

write_initial_contract(
    pathlib.Path(sys.argv[1]),
    profile=load_profile(sys.argv[2]),
    task=sys.argv[3],
    prompt=sys.argv[4],
    model=sys.argv[5],
    reasoning_effort=sys.argv[6],
    operator_root=sys.argv[7],
    yolo=sys.argv[8] == "1",
)
' "$EPISODE_ROOT/operator_context_contract.json" "$OPERATOR_CONTEXT_PROFILE" \
  "$TASK" "$OPERATOR_PROMPT" "$OPERATOR_MODEL" "$OPERATOR_REASONING_EFFORT" \
  "$OPERATOR_ROOT" "$OPERATOR_YOLO"

gateway_pid=""
operator_pid=""
stop_persistent_gateway() {
  if [[ -n "$gateway_pid" ]] && kill -0 "$gateway_pid" 2>/dev/null; then
    openeta_stop_owned_process "$EPISODE_ROOT" "$gateway_pid" \
      "$(openeta_process_start_ticks "$gateway_pid" 2>/dev/null || true)" \
      "$(openeta_process_group "$gateway_pid")" || true
    wait "$gateway_pid" 2>/dev/null || true
  fi
  [[ -n "$gateway_pid" ]] && openeta_unregister_service "$EPISODE_ROOT" "$gateway_pid"
  rm -rf "$EPISODE_ROOT/.operator-launch-claim"
}
stop_operator_runtime() {
  # A model-stream failure is not an Operator decision.  If the episode is
  # still live, ask the persistent Gateway to record an external abort before
  # its process is torn down; this keeps Replay Hub status truthful.
  if [[ -n "$gateway_pid" ]] && kill -0 "$gateway_pid" 2>/dev/null; then
    episode_status="$(
      OPENETA_EPISODE_ROOT="$EPISODE_ROOT" \
        uv run --no-project --python "$REPO_ROOT/.venv/bin/python" python -c '
import json, os
from pathlib import Path
p = Path(os.environ["OPENETA_EPISODE_ROOT"]) / "current.json"
try:
    value = json.loads(p.read_text())
except Exception:
    value = {}
print(value.get("status", "unknown"))
' 2>/dev/null || printf 'unknown'
    )"
    case "$episode_status" in
      running)
        curl --noproxy '*' -sS --max-time 20 \
          -H 'Content-Type: application/json' \
          -X POST "http://${GATEWAY_HOST}:${GATEWAY_CONTROL_PORT}/abort" \
          -d '{"reason":"Operator stream stopped before terminal decision."}' \
          >/dev/null 2>&1 || true
        ;;
    esac
  fi
  if [[ -n "$operator_pid" ]] && kill -0 "$operator_pid" 2>/dev/null; then
    openeta_stop_owned_process "$EPISODE_ROOT" "$operator_pid" \
      "$(openeta_process_start_ticks "$operator_pid" 2>/dev/null || true)" \
      "$(openeta_process_group "$operator_pid")" || true
    wait "$operator_pid" 2>/dev/null || true
  fi
  [[ -n "$operator_pid" ]] && openeta_unregister_service "$EPISODE_ROOT" "$operator_pid"
  stop_persistent_gateway
}

if [[ "$OPERATOR_RUNTIME" == "app-server" ]]; then
  # Codex app-server may recreate command-backed stdio MCP clients at a turn
  # boundary. The embodied Gateway owns the simulator handle and episode
  # writer, so it must instead outlive every automatic continuation turn.
  # Run one local SSE Gateway for the whole Operator process and let Codex
  # reconnect only its transport client.
  persistent_gateway_args=("${gateway_args[@]}")
  persistent_gateway_args[7]="streamable-http"
  persistent_gateway_args+=(
    --host "$GATEWAY_HOST"
    --port "$GATEWAY_PORT"
  )
  openeta_assert_or_reclaim_port "$EPISODE_ROOT" "$GATEWAY_PORT" "Operator Gateway MCP"
  openeta_assert_or_reclaim_port "$EPISODE_ROOT" "$GATEWAY_CONTROL_PORT" "Operator Gateway control"
  setsid env "${gateway_process_env[@]}" uv "${persistent_gateway_args[@]}" \
    >"$EPISODE_ROOT/gateway.log" 2>&1 &
  gateway_pid="$!"
  printf '%s\n' "$gateway_pid" > "$EPISODE_ROOT/gateway.pid"
  openeta_register_service "$EPISODE_ROOT" gateway "$gateway_pid" "$GATEWAY_PORT,$GATEWAY_CONTROL_PORT"
  trap 'stop_persistent_gateway; cleanup_operator_codex_home' EXIT INT TERM
  gateway_url="http://${GATEWAY_HOST}:${GATEWAY_PORT}/mcp"
  gateway_ready=0
  for _ in $(seq 1 100); do
    if ! kill -0 "$gateway_pid" 2>/dev/null; then
      printf 'Persistent Operator Gateway exited during startup. See %s\n' \
        "$EPISODE_ROOT/gateway.log" >&2
      exit 2
    fi
    gateway_status="$(
      curl --noproxy '*' -sS -o /dev/null -w '%{http_code}' \
        --max-time 1 "$gateway_url" 2>/dev/null || true
    )"
    # A real FastMCP Streamable HTTP endpoint rejects a bare GET without an
    # MCP session as 400/405/406.  A generic 404 can belong to an unrelated
    # process already occupying the requested port and must never count as
    # readiness.
    if [[ "$gateway_status" =~ ^(200|400|405|406)$ ]]; then
      gateway_ready=1
      break
    fi
    sleep 0.1
  done
  if [[ "$gateway_ready" != "1" ]] || ! kill -0 "$gateway_pid" 2>/dev/null; then
    printf 'Persistent Operator Gateway is unavailable. See %s\n' \
      "$EPISODE_ROOT/gateway.log" >&2
    exit 2
  fi
  {
    # Some OpenAI-compatible Responses gateways reject zstd request bodies.
    # Keep the isolated Operator transport on ordinary JSON.
    printf '\n[features]\nmemories = false\nenable_request_compression = false\n'
    printf '\n[memories]\nuse_memories = false\ngenerate_memories = false\n'
    printf '\n[history]\npersistence = "none"\n'
    printf '\n[mcp_servers.operator]\nurl = %s\nrequired = true\ntool_timeout_sec = 600\n' \
      "$(toml_string "$gateway_url")"
  } >> "$OPERATOR_CODEX_HOME/config.toml"
  setsid uv run --no-project --python "$REPO_ROOT/.venv/bin/python" \
    "$REPO_ROOT/scripts/embodied/operator_app_server.py" \
    --episode-root "$EPISODE_ROOT" \
    --cwd "$OPERATOR_ROOT" \
    --prompt "$OPERATOR_PROMPT" \
    --model "$OPERATOR_MODEL" \
    --reasoning-effort "$OPERATOR_REASONING_EFFORT" \
    --codex-bin "$CODEX_BIN" &
  operator_pid="$!"
  openeta_register_service "$EPISODE_ROOT" operator-app-server "$operator_pid" "-"
  trap 'stop_operator_runtime; cleanup_operator_codex_home' EXIT INT TERM
  operator_status=0
  wait "$operator_pid" || operator_status="$?"
  stop_operator_runtime
  cleanup_operator_codex_home
  trap - EXIT INT TERM
  exit "$operator_status"
fi

codex_config_args=(
  -c 'features.memories=false'
  -c 'features.enable_request_compression=false'
  -c 'memories.use_memories=false'
  -c 'memories.generate_memories=false'
  -c 'history.persistence="none"'
  -c "mcp_servers.operator.command=$(toml_string env)"
  -c "mcp_servers.operator.args=$gateway_args_toml"
)

codex_yolo_args=()
if [[ "$OPERATOR_YOLO" == "1" ]]; then
  codex_yolo_args=(--dangerously-bypass-approvals-and-sandbox)
fi

case "$CODEX_MODE" in
  interactive)
    operator_status=0
    "$CODEX_BIN" \
      -C "$OPERATOR_ROOT" \
      --no-alt-screen \
      -m "$OPERATOR_MODEL" \
      "${codex_yolo_args[@]}" \
      "${codex_config_args[@]}" \
      "$OPERATOR_PROMPT" || operator_status="$?"
    ;;
  exec)
    # Headless proof mode for tty/CI.  The model and MCP boundary are the
    # same as interactive mode; only the Codex presentation is JSONL.
    operator_status=0
    "$CODEX_BIN" \
      -m "$OPERATOR_MODEL" \
      "${codex_yolo_args[@]}" \
      exec \
      -C "$OPERATOR_ROOT" \
      -s read-only \
      --ephemeral \
      --ignore-user-config \
      --ignore-rules \
      --skip-git-repo-check \
      --json \
      "${codex_config_args[@]}" \
      "$OPERATOR_PROMPT" || operator_status="$?"
    ;;
  *)
    printf 'Unsupported OPENETA_CODEX_MODE=%s (expected interactive or exec)\n' "$CODEX_MODE" >&2
    exit 2
    ;;
esac

cleanup_operator_codex_home
trap - EXIT INT TERM
exit "$operator_status"
