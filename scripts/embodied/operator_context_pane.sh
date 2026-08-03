#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${OPENETA_REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
ROOT="${OPENETA_EMBODIED_ROOT:?OPENETA_EMBODIED_ROOT is required}"

cd "$REPO_ROOT"
exec uv run --no-project --python 3.10 \
  "$REPO_ROOT/scripts/embodied/kitty_operator_context_view.py" \
  --root "$ROOT" \
  --interval "${OPENETA_CONTEXT_INTERVAL:-0.2}"
