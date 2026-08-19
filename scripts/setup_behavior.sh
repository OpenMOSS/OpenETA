#!/bin/bash
# OpenETA — BEHAVIOR-1K / OmniGibson setup for a fresh machine.
#
# Split out of setup_envs.sh because BEHAVIOR is the one backend that cannot
# live in a uv venv: Isaac Sim ships conda-only binary extensions, so the
# runtime is a conda env and the OpenETA worker reaches it through a symlink.
#
# Idempotent: every step checks for its own completion marker first, so a
# re-run after a failure resumes instead of redoing the 37 GB download.
#
# Usage:
#   scripts/setup_behavior.sh                 # full setup
#   scripts/setup_behavior.sh --verify        # check an existing install
#   BEHAVIOR_ROOT=/data/BEHAVIOR-1K scripts/setup_behavior.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

BEHAVIOR_ROOT="${BEHAVIOR_ROOT:-/home/yfzhang/nvme1/BEHAVIOR-1K}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-behavior}"
BEHAVIOR_TAG="${BEHAVIOR_TAG:-v3.9.1}"
# Upstream setup.sh pins torch 2.7.0+cu128; the CUDA toolkit must match that
# minor version or OmniGibson's primitives extension refuses to build.
CUDA_VERSION="${CUDA_VERSION:-12.8}"
VENV_LINK="$REPO_ROOT/sim/venvs/behavior"

VERIFY_ONLY=false
[ "${1:-}" = "--verify" ] && VERIFY_ONLY=true

say()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
ok()   { printf '    \033[32mOK\033[0m   %s\n' "$*"; }
warn() { printf '    \033[33mWARN\033[0m %s\n' "$*"; }
die()  { printf '    \033[31mFAIL\033[0m %s\n' "$*" >&2; exit 1; }

CONDA_BASE="$(conda info --base 2>/dev/null || echo "$HOME/anaconda3")"
ENV_PREFIX="$CONDA_BASE/envs/$CONDA_ENV_NAME"
PY="$ENV_PREFIX/bin/python"

# ── 0. preflight ────────────────────────────────────────────────────
say "Preflight"
command -v conda >/dev/null || die "conda not found; Isaac Sim requires a conda env"
command -v git   >/dev/null || die "git not found"
nvidia-smi >/dev/null 2>&1  || die "nvidia-smi failed; an NVIDIA GPU + driver is required"

DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
CAP="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)"
NGPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
ok "driver $DRIVER, compute_cap $CAP, ${NGPU} GPU(s)"
# Verified working on compute_cap 12.0 (Blackwell / RTX 5090) with Isaac Sim
# 5.1.0 -- recorded because Blackwell is newer than Isaac Sim's release notes.

if [ -d "/usr/local/cuda-${CUDA_VERSION%.*}" ] || [ -d "/usr/local/cuda-$CUDA_VERSION" ]; then
    ok "CUDA toolkit $CUDA_VERSION present"
else
    warn "No /usr/local/cuda-$CUDA_VERSION; only needed for --primitives"
fi

FREE_GB="$(df -BG --output=avail "$(dirname "$BEHAVIOR_ROOT")" 2>/dev/null | tail -1 | tr -dc '0-9' || echo 0)"
if [ "${FREE_GB:-0}" -lt 60 ] && [ ! -d "$BEHAVIOR_ROOT/datasets/behavior-1k-assets" ]; then
    warn "Only ${FREE_GB}G free at $(dirname "$BEHAVIOR_ROOT"); assets need ~40G"
fi

if $VERIFY_ONLY; then
    say "Verify-only mode"
    [ -x "$PY" ] || die "no interpreter at $PY"
    OMNI_KIT_ACCEPT_EULA=YES OMNIGIBSON_HEADLESS=1 "$PY" - <<'PYCODE' || die "import check failed"
import importlib
for m in ("torch", "omnigibson", "bddl", "numpy", "av", "lerobot"):
    mod = importlib.import_module(m)
    print(f"    OK   {m} {getattr(mod, '__version__', '?')}")
import torch
print(f"    OK   cuda avail={torch.cuda.is_available()} devices={torch.cuda.device_count()}")
PYCODE
    [ -L "$VENV_LINK" ] && ok "worker symlink -> $(readlink "$VENV_LINK")" || warn "missing $VENV_LINK"
    exit 0
fi

# ── 1. source tree ──────────────────────────────────────────────────
say "Source tree at $BEHAVIOR_ROOT"
if [ -d "$BEHAVIOR_ROOT/.git" ]; then
    ok "already cloned ($(git -C "$BEHAVIOR_ROOT" describe --tags 2>/dev/null || echo 'no tag'))"
else
    mkdir -p "$(dirname "$BEHAVIOR_ROOT")"
    git clone --branch "$BEHAVIOR_TAG" --depth 1 \
        https://github.com/StanfordVL/BEHAVIOR-1K.git "$BEHAVIOR_ROOT"
    ok "cloned $BEHAVIOR_TAG"
fi

# ── 2. conda env + upstream installer ───────────────────────────────
say "Conda env '$CONDA_ENV_NAME' (Python 3.11)"
if [ -x "$PY" ]; then
    ok "exists: $($PY --version 2>&1)"
else
    conda create -y -n "$CONDA_ENV_NAME" python=3.11
    ok "created"
fi

say "Upstream installer (omnigibson + bddl + Isaac Sim + assets)"
# The upstream script owns the isaacsim==5.1.0 pin, the torch/cu wheel index,
# and asset decryption.  Reimplementing any of that here would silently drift
# from the pinned checkout, so drive it instead of duplicating it.
if OMNI_KIT_ACCEPT_EULA=YES "$PY" -c "import omnigibson" >/dev/null 2>&1; then
    ok "omnigibson already importable; skipping installer"
else
    DATASET_FLAG="--dataset"
    if [ -d "$BEHAVIOR_ROOT/datasets/behavior-1k-assets" ]; then
        # Reuse the 37 GB already on disk rather than re-downloading it.
        DATASET_FLAG=""
        ok "assets present ($(du -sh "$BEHAVIOR_ROOT/datasets" 2>/dev/null | cut -f1)); skipping download"
    fi
    (
        cd "$BEHAVIOR_ROOT"
        eval "$(conda shell.bash hook)"
        conda activate "$CONDA_ENV_NAME"
        unset EXP_PATH CARB_APP_PATH ISAAC_PATH
        export OMNI_KIT_ACCEPT_EULA=YES CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes
        # shellcheck disable=SC2086
        ./setup.sh --omnigibson --bddl $DATASET_FLAG \
            --cuda-version "$CUDA_VERSION" \
            --accept-conda-tos --accept-nvidia-eula --accept-dataset-tos
    )
    ok "installer finished"
fi

# ── 3. dependency pin that upstream misses ──────────────────────────
say "PyAV pin"
# lerobot 0.5.2 declares av>=15,<16, but nothing in the install chain enforces
# it.  With av 18 installed, `av.option` is gone, lerobot.datasets exports
# nothing, and omnigibson.envs dies on its eager LeRobotDataWrapper import --
# which surfaces as "cannot import name LeRobotDataset", nowhere near the
# real cause.  Pin it explicitly.
AV_OK=$("$PY" - <<'PYCODE'
try:
    import av
    v = tuple(int(x) for x in av.__version__.split(".")[:1])
    print("yes" if 15 <= v[0] < 16 else "no")
except Exception:
    print("no")
PYCODE
)
if [ "$AV_OK" = "yes" ]; then
    ok "av in range"
else
    "$PY" -m pip install -q 'av>=15.0.0,<16.0.0'
    ok "av pinned to 15.x"
fi

# ── 4. worker symlink ───────────────────────────────────────────────
say "OpenETA worker link"
# sim/env_registry.py finds a bench runtime under sim/venvs/<bench>; BEHAVIOR's
# runtime is the conda env, so link it in.  Path is gitignored.
mkdir -p "$REPO_ROOT/sim/venvs"
if [ -L "$VENV_LINK" ]; then
    ok "already linked -> $(readlink "$VENV_LINK")"
elif [ -e "$VENV_LINK" ]; then
    warn "$VENV_LINK exists and is not a symlink; leaving it alone"
else
    ln -s "$ENV_PREFIX" "$VENV_LINK"
    ok "linked $VENV_LINK -> $ENV_PREFIX"
fi

cat > "$REPO_ROOT/sim/venvs/behavior_activate_extra.sh" << SH
# Sourced by the BEHAVIOR worker.  OMNI_KIT_ACCEPT_EULA must be set before the
# first isaacsim import or the process blocks forever on an interactive prompt
# that has no tty to read from.
export OMNI_KIT_ACCEPT_EULA=YES
export OMNIGIBSON_HEADLESS=1
export BEHAVIOR_ROOT=$BEHAVIOR_ROOT
# OMNIGIBSON_DATA_PATH is the name macros.determine_data_path() actually reads.
# OMNIGIBSON_DATASET_PATH is read by nothing, so setting that instead leaves the
# default in force -- a path relative to the omnigibson module -- and env
# creation dies on "Data path ... does not exist" only once a worker boots.
export OMNIGIBSON_DATA_PATH=$BEHAVIOR_ROOT/datasets
SH
ok "wrote behavior_activate_extra.sh"

# ── 5. smoke test ───────────────────────────────────────────────────
say "Smoke test (boots Isaac Sim; ~40 s cold)"
OMNI_KIT_ACCEPT_EULA=YES OMNIGIBSON_HEADLESS=1 "$PY" - <<'PYCODE' || die "smoke test failed"
import omnigibson as og
env = og.Environment(configs={
    "env": {"action_frequency": 30, "physics_frequency": 120},
    "scene": {"type": "Scene"},
    "robots": [{"type": "R1Pro", "obs_modalities": ["rgb"],
                "action_normalize": True, "grasping_mode": "physical",
                "default_reset_mode": "tuck"}],
})
robot = env.robots[0]
groups = {k: len(v) for k, v in robot.controller_action_idx.items()}
print(f"    OK   R1Pro action_dim={robot.action_dim} groups={groups}")
assert robot.action_dim == 21, f"expected 21 (IK default), got {robot.action_dim}"
print("    OK   smoke test passed")
import sys; sys.stdout.flush()
try:
    og.shutdown()          # hard-exits; nothing after this line runs
except BaseException:
    pass
PYCODE

say "Done"
cat <<EOF
    Source:   $BEHAVIOR_ROOT
    Runtime:  $ENV_PREFIX
    Worker:   $VENV_LINK
    Verify:   scripts/setup_behavior.sh --verify
EOF
