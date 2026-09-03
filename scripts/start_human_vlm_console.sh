#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"
python_bin="${repo_root}/.venv/bin/python"

port="${OPENETA_MANUAL_VLM_PORT:-8099}"
trace_dir="${OPENETA_MANUAL_VLM_TRACE_DIR:-tmp/manual-vlm-traces/libero-goal-0-seed0}"
open_browser="${OPENETA_MANUAL_VLM_OPEN_BROWSER:-1}"

if [[ ! -x "${python_bin}" ]]; then
  echo "OpenETA Python environment not found: ${python_bin}" >&2
  echo "Run 'uv sync --extra dev' in ${repo_root} first." >&2
  exit 1
fi

cd "${repo_root}"
mkdir -p "${trace_dir}"

echo "Starting Human VLM Console for LIBERO Goal task 0."
echo "GUI: http://127.0.0.1:${port}/"
echo "Provider API: http://127.0.0.1:${port}/v1"
echo "Wire traces: ${repo_root}/${trace_dir}"
echo "Keep this terminal open; press Ctrl-C to stop."

open_args=()
if [[ "${open_browser}" != "0" ]]; then
  open_args+=(--open)
fi

exec "${python_bin}" -m tools.manual_vlm_proxy \
  --host 127.0.0.1 \
  --port "${port}" \
  --decision-timeout 0 \
  --record-dir "${trace_dir}" \
  "${open_args[@]}"
