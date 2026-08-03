#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${OPENETA_REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"

# Keep the simulator, Gateway, and Replay services owned by the existing
# launcher while running the ordinary Codex CLI interactively in this terminal.
export OPENETA_OPERATOR_RUNTIME=cli
export OPENETA_CODEX_MODE=interactive
export OPENETA_OPERATOR_FOREGROUND=1

exec "$REPO_ROOT/scripts/embodied/launch_operator_chrome.sh" "$@"
