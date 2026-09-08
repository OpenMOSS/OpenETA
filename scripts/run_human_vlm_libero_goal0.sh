#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
openeta_bin="${repo_root}/.venv/bin/openeta"

manual_vlm_host="127.0.0.1"
manual_vlm_port="${OPENETA_MANUAL_VLM_PORT:-8099}"
sim_host="127.0.0.1"
sim_port="${OPENETA_LOCAL_SIM_PORT:-8766}"
simulator_timeout_s="${OPENETA_SIMULATOR_TIMEOUT_S:-300}"
max_turns="${OPENETA_HUMAN_MAX_TURNS:-200}"

if [[ ! -x "${openeta_bin}" ]]; then
  echo "OpenETA CLI not found: ${openeta_bin}" >&2
  echo "Run 'uv sync --extra dev' in ${repo_root} first." >&2
  exit 1
fi

if ! curl --fail --silent --show-error --max-time 2 \
  "http://${manual_vlm_host}:${manual_vlm_port}/v1/models" >/dev/null; then
  echo "Human VLM Console is not listening on ${manual_vlm_host}:${manual_vlm_port}." >&2
  echo "Start it first with:" >&2
  echo "  bash scripts/start_human_vlm_console.sh" >&2
  exit 1
fi

if ! nc -z -w 2 "${sim_host}" "${sim_port}"; then
  echo "Local OpenETA simulator is not listening on ${sim_host}:${sim_port}." >&2
  echo "Selected simulator endpoint: http://${sim_host}:${sim_port}/sse." >&2
  exit 1
fi

cd "${repo_root}"

export OPENETA_LLM_PROVIDER="manual-vlm"
export OPENETA_LLM_MODEL="human-vlm"
export OPENETA_LLM_API_BASE="http://${manual_vlm_host}:${manual_vlm_port}/v1"
export OPENETA_LLM_API_KEY="local-placeholder"
export OPENETA_LLM_TIMEOUT_S="86400"
export OPENETA_LLM_MAX_ATTEMPTS="1"
export OPENETA_LLM_RETRY_BACKOFF_S="0"

# Keep fallback on the same human-operated endpoint. This prevents values in a
# local .env from silently handing a cancelled or retried request to a model.
export OPENETA_LLM_FALLBACK_PROVIDER="manual-vlm"
export OPENETA_LLM_FALLBACK_MODEL="human-vlm"
export OPENETA_LLM_FALLBACK_API_BASE="http://${manual_vlm_host}:${manual_vlm_port}/v1"
export OPENETA_LLM_FALLBACK_API_KEY="local-placeholder"
export OPENETA_LLM_FALLBACK_TIMEOUT_S="86400"

task="Complete LIBERO Goal task 0 with seed 0. Use the exact environment ID openeta/libero_libero_goal_task0-v0. The benchmark instruction is: open the middle drawer of the cabinet. Create that simulator environment through the registered environment tool, use only the provider-visible observations, artifacts, python_exec sandbox, and registered tools, and stop only after trusted environment evidence reports success or after you determine that the harness cannot express or execute a necessary action."

echo "Starting Human VLM run: LIBERO Goal task 0, seed 0."
echo "Environment: openeta/libero_libero_goal_task0-v0"
echo "Instruction: open the middle drawer of the cabinet"
echo "Turn budget: ${max_turns}"
echo "Answer each pending planner request in http://${manual_vlm_host}:${manual_vlm_port}/"

exec "${openeta_bin}" --simulator-timeout-s "${simulator_timeout_s}" \
  --simulator-mcp-url "http://${sim_host}:${sim_port}/sse" \
  --once "${task}" --max-turns "${max_turns}"
