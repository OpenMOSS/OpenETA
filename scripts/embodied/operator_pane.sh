#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${OPENETA_REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
ROOT="${OPENETA_EMBODIED_ROOT:?OPENETA_EMBODIED_ROOT is required}"

printf '\033[1;36mOpenETA Operator Codex pane\033[0m\n'
printf 'This pane must receive only the task, embodied tool docs, and current visual results.\n'
printf 'Episode root: %s\n' "$ROOT"
printf 'Host diagnostics stay outside this pane and read logger artifacts.\n'
printf 'A fresh Operator Codex is started here; set OPENETA_OPERATOR_COMMAND only for an explicit launcher override.\n\n'

if [[ -n "${OPENETA_OPERATOR_COMMAND:-}" && "${OPENETA_ALLOW_OPERATOR_OVERRIDE:-0}" != "1" ]]; then
  printf 'OPENETA_OPERATOR_COMMAND requires OPENETA_ALLOW_OPERATOR_OVERRIDE=1\n' >&2
  exit 2
fi

if [[ -n "${OPENETA_OPERATOR_COMMAND:-}" ]]; then
  exec zsh -lc "$OPENETA_OPERATOR_COMMAND"
fi
exec "$REPO_ROOT/scripts/embodied/operator_codex.sh"
