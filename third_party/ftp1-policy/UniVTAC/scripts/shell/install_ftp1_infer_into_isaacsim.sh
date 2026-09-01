#!/usr/bin/env bash
# Install FTP1 inference **into the current (Isaac Sim) environment** without touching torch.
# Keeps Isaac Sim's torch 2.5.1; only adds FTP1 deps (transformers, jax, openpi, etc.).
#
# Use after Isaac Sim + Isaac Lab are already installed in this env.
# Run from ftp1 repo root:
#   conda activate <your_isaacsim_env>
#   bash UniVTAC/scripts/shell/install_ftp1_infer_into_isaacsim.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIVTAC_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
FTP1_ROOT="$(cd "$UNIVTAC_ROOT/.." && pwd)"
REQUIREMENTS="$UNIVTAC_ROOT/requirements-univtac-infer-into-isaacsim.txt"

if [[ ! -f "$FTP1_ROOT/pyproject.toml" ]] || [[ ! -d "$FTP1_ROOT/UniVTAC" ]]; then
  echo "[ERROR] ftp1 repo root not found." >&2
  exit 1
fi

if [[ ! -f "$REQUIREMENTS" ]]; then
  echo "[ERROR] Requirements file not found: $REQUIREMENTS" >&2
  exit 1
fi

echo "[install] FTP1 inference into current env (do not upgrade torch)"
echo "[install] Python: $(which python) ($(python -c 'import sys; print(sys.version)'))"
echo "[install] PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not found')"
echo ""

# 1) Install FTP1 inference deps (no torch in this file)
echo "[install] Step 1/4: pip install -r requirements-univtac-infer-into-isaacsim.txt"
pip install -r "$REQUIREMENTS"

# 2) Install openpi (ftp1) editable, no deps
echo "[install] Step 2/4: pip install --no-deps -e . (openpi)"
pip install --no-deps -e "$FTP1_ROOT"

# 3) Install openpi-client
echo "[install] Step 3/4: pip install --no-deps -e packages/openpi-client"
pip install --no-deps -e "$FTP1_ROOT/packages/openpi-client"

# 4) Verify
echo "[install] Step 4/4: Verify FTP1 inference import..."
cd "$FTP1_ROOT"
python -c "
from openpi.policies.ftp1_inference_wrapper import FTP1InferenceWrapper
from openpi.models_pytorch.ftp1_pytorch import FTP1Pytorch
print('FTP1 inference import OK')
"

echo ""
echo "[install] Done. FTP1 inference is available in this (Isaac Sim) environment."
