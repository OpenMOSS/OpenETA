#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${OPENETA_REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
# shellcheck source=service_lifecycle.sh
source "$REPO_ROOT/scripts/embodied/service_lifecycle.sh"

if [[ "$#" -ne 1 ]]; then
  printf 'Usage: %s <episode-root>\n' "$0" >&2
  exit 2
fi

ROOT="$1"
[[ "$ROOT" = /* ]] || ROOT="$REPO_ROOT/$ROOT"
openeta_stop_episode_services "$ROOT"
printf 'Stopped registered OpenETA services for %s\n' "$ROOT"
