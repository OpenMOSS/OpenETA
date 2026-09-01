#!/usr/bin/env bash
# One-shot setup: create a single conda env (UniVTAC, Python 3.10) with
# Isaac Sim 4.5, Isaac Lab 2.1.1, TacEx (from third_party), and RTac1 (openpi).
# Run from the **rtac1** repo root (so that UniVTAC/ and pyproject.toml exist).
#
# Prerequisites:
#   - conda installed
#   - Isaac Lab cloned and checked out to v2.1.1, e.g.:
#       git clone https://github.com/isaac-sim/IsaacLab
#       cd IsaacLab && git checkout v2.1.1
#   - Set ISAACLAB_PATH to Isaac Lab repo (or default: ~/IsaacLab)
#
# Usage:
#   cd /path/to/rtac1
#   export ISAACLAB_PATH=~/IsaacLab   # optional if not ~/IsaacLab
#   bash UniVTAC/scripts/setup_univtac_rtac1_env.sh

set -euo pipefail

RTAC1_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
UNIVTAC_ROOT="${RTAC1_ROOT}/UniVTAC"
TACEX_ROOT="${UNIVTAC_ROOT}/third_party/TacEx"
ISAACLAB_PATH="${ISAACLAB_PATH:-$HOME/IsaacLab}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-UniVTAC}"
INSTALL_RTAC1="${INSTALL_RTAC1:-true}"

echo "[setup] RTac1 root: $RTAC1_ROOT"
echo "[setup] UniVTAC root: $UNIVTAC_ROOT"
echo "[setup] Isaac Lab path: $ISAACLAB_PATH"
echo "[setup] Conda env: $CONDA_ENV_NAME"

if [[ ! -f "$RTAC1_ROOT/pyproject.toml" ]]; then
  echo "[ERROR] Run this script from the rtac1 repo root (parent of UniVTAC/)." >&2
  exit 1
fi

if ! command -v conda &>/dev/null; then
  echo "[ERROR] conda not found. Install Miniconda/Anaconda first." >&2
  exit 1
fi

# Ensure conda is available in this shell
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true

# Create conda env if not exists
if conda env list | grep -q "^${CONDA_ENV_NAME}\s"; then
  echo "[INFO] Conda env '$CONDA_ENV_NAME' already exists."
else
  echo "[INFO] Creating conda env '$CONDA_ENV_NAME' with Python 3.10..."
  conda create -n "$CONDA_ENV_NAME" python=3.10 -y
fi

conda activate "$CONDA_ENV_NAME"

# Isaac Sim 4.5 (pip) + PyTorch cu118
echo "[INFO] Installing PyTorch (cu118) and Isaac Sim 4.5..."
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
pip install --upgrade pip
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com

# Isaac Lab 2.1.1
if [[ -d "$ISAACLAB_PATH" ]]; then
  echo "[INFO] Installing Isaac Lab from $ISAACLAB_PATH..."
  export ISAACLAB_PATH
  cd "$ISAACLAB_PATH"
  git checkout v2.1.1 2>/dev/null || true
  ./isaaclab.sh --install
  cd "$RTAC1_ROOT"
else
  echo "[WARN] Isaac Lab not found at $ISAACLAB_PATH. Clone it and run: ./isaaclab.sh --install"
fi

# TacEx from third_party (requires ISAACLAB_PATH for tacex.sh)
if [[ -d "$TACEX_ROOT" ]] && [[ -n "${ISAACLAB_PATH:-}" ]] && [[ -d "$ISAACLAB_PATH" ]]; then
  echo "[INFO] Installing TacEx from $TACEX_ROOT..."
  export TACEX_PATH="$TACEX_ROOT"
  export ISAACLAB_PATH
  cd "$TACEX_ROOT"
  ./tacex.sh -i
  cd "$RTAC1_ROOT"
else
  echo "[WARN] Skipping TacEx (TACEX_ROOT or ISAACLAB_PATH missing). Install manually: cd third_party/TacEx && ./tacex.sh -i"
fi

# RTac1 (openpi) in the same env
if [[ "$INSTALL_RTAC1" == "true" ]]; then
  echo "[INFO] Installing RTac1 (openpi) in editable mode..."
  cd "$RTAC1_ROOT"
  pip install -e . --no-build-isolation 2>/dev/null || pip install -e .
  echo "[INFO] RTac1 installed. Verify with: python -c \"import torch; print('RTac1 import OK')\""
fi

echo "[INFO] Done. Activate with: conda activate $CONDA_ENV_NAME"
