#!/bin/bash
# OpenETA — per-environment uv virtual environments
# Each benchmark gets its own isolated venv under sim/venvs/<name>/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENVS_DIR="$REPO_ROOT/sim/venvs"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

usage() {
    cat << 'EOF'
Usage: setup_envs.sh [BENCH]...

Create isolated uv virtual environments for OpenETA simulation backends.

Benchmarks:
  metaworld     MetaWorld — 100 Sawyer manipulation tasks (MuJoCo)
  maniskill     ManiSkill 3 — 74 manipulation/locomotion tasks (SAPIEN)
  libero        LIBERO — 60+ Franka manipulation tasks (robosuite/MuJoCo)
  robocasa      RoboCasa365 — 317 kitchen tasks (PandaOmron/MuJoCo)
  behavior      BEHAVIOR-1K v3.9 — 1000 household activities (R1Pro/Isaac Sim)
  genesis       Genesis — Franka cube pick (GPU physics engine)
  d4rl          D4RL — 9 locomotion tasks (MuJoCo)
  all           All of the above

Examples:
  setup_envs.sh metaworld
  setup_envs.sh metaworld,maniskill
  setup_envs.sh libero maniskill all

Each bench creates:  sim/venvs/<name>/
Activate with:       source sim/venvs/<name>/bin/activate
Only registered envs for that bench are available after activation.
EOF
    exit 0
}

# ── Parse args ──────────────────────────────────────────────────
if [ $# -eq 0 ] || [ "$1" = "-h" ] || [ "$1" = "--help" ]; then
    usage
fi

BENCHES=""
for arg in "$@"; do
    case "$arg" in
        all) BENCHES="$BENCHES metaworld maniskill libero robocasa behavior genesis d4rl" ;;
        *)   for b in ${arg//,/ }; do BENCHES="$BENCHES $b"; done ;;
    esac
done

mkdir -p "$VENVS_DIR"

echo "================================================================"
echo "OpenETA Per-Environment Setup"
echo "Targets: $BENCHES"
echo "venvs dir: $VENVS_DIR"
echo "================================================================"

for BENCH in $BENCHES; do

    VENV="$VENVS_DIR/$BENCH"
    case "$BENCH" in

    # ═══════════════ MetaWorld ══════════════════════════════════
    metaworld)
        echo ""
        echo "==> MetaWorld → $VENV"
        uv venv -q --python "$PYTHON_VERSION" "$VENV" 2>&1 || true
        source "$VENV/bin/activate"
        uv pip install -q gymnasium numpy torch cloudpickle packaging metaworld 2>&1 | tail -1
        uv pip install -q -e "$REPO_ROOT" 2>&1 | tail -1

        cat > "$VENV/activate_extra.sh" << 'SH'
export MUJOCO_GL=egl
SH
        echo "   Done. Activate: source sim/venvs/metaworld/bin/activate"
        echo "   NOTE: mujoco_py may fail on GCC14. If so, use conda."
        ;;

    # ═══════════════ ManiSkill ═════════════════════════════════
    maniskill)
        echo ""
        echo "==> ManiSkill → $VENV"
        uv venv -q --python "$PYTHON_VERSION" "$VENV" 2>&1 || true
        source "$VENV/bin/activate"
        uv pip install -q gymnasium numpy torch cloudpickle mani_skill 2>&1 | tail -1
        uv pip install -q -e "$REPO_ROOT" 2>&1 | tail -1

        cat > "$VENV/activate_extra.sh" << 'SH'
export MS_SKIP_ASSET_DOWNLOAD_PROMPT=1
SH

        # Download robot assets
        echo "   Downloading robot assets..."
        for task in "PickCube-v1" "MS-AntRun-v1" "AnymalC-Reach-v1" "UnitreeG1Stand-v1"; do
            echo -n "     $task ... "
            timeout 60 python -c "
import os; os.environ['MS_SKIP_ASSET_DOWNLOAD_PROMPT']='1'
import gymnasium as gym; import mani_skill.envs
gym.make('\$task', obs_mode='state', render_mode='rgb_array', num_envs=1).reset()
" 2>/dev/null && echo "OK" || echo "SKIP"
        done
        echo "   Done. Activate: source sim/venvs/maniskill/bin/activate"
        ;;

    # ═══════════════ LIBERO ════════════════════════════════════
    libero)
        LIBERO_DIR="${LIBERO_DIR:-/tmp/LIBERO}"
        LIBERO_ASSETS="${LIBERO_ASSETS:-/tmp/libero_assets}"
        echo ""
        echo "==> LIBERO → $VENV"

        uv venv -q --python "$PYTHON_VERSION" "$VENV" 2>&1 || true
        source "$VENV/bin/activate"
        uv pip install -q gymnasium numpy torch cloudpickle h5py 2>&1 | tail -1

        if [ ! -d "$LIBERO_DIR" ]; then
            echo "   Cloning LIBERO..."
            git clone -q https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_DIR" --depth 1
        fi

        # CRITICAL: LIBERO needs specific MuJoCo version
        uv pip install -q mujoco==3.3.0 'robosuite<1.5' bddl easydict matplotlib gym 2>&1 | tail -1
        uv pip install -q -e "$LIBERO_DIR" 2>&1 | tail -1
        uv pip install -q -e "$REPO_ROOT" 2>&1 | tail -1

        # Store assets inside the venv (not /tmp)
        LIBERO_ASSETS="${VENV}/assets/datasets"
        mkdir -p "${VENV}/assets/datasets"

        # Pre-create LIBERO config file (avoids interactive prompt on import)
        mkdir -p ~/.libero
        cat > ~/.libero/config.yaml << YAML
benchmark_root: ${LIBERO_DIR}/libero/libero
bddl_files: ${LIBERO_DIR}/libero/libero/bddl_files
init_states: ${LIBERO_DIR}/libero/libero/init_files
datasets: ${VENV}/assets/datasets
assets: ${LIBERO_DIR}/libero/libero/assets
YAML

        cat > "$VENV/activate_extra.sh" << SH
export MUJOCO_GL=egl
export LIBERO_DATASET_PATH=${VENV}/assets/datasets
export LIBERO_DIR=${LIBERO_DIR}
SH

        # Download datasets
        mkdir -p "$LIBERO_ASSETS"
        echo "   Downloading LIBERO datasets..."
        for ds in libero_spatial libero_object libero_goal libero_10; do
            echo -n "     $ds ... "
            echo "N" | timeout 120 python -c "
import os, sys; sys.path.insert(0, '$LIBERO_DIR')
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
from libero.libero.utils.download_utils import libero_dataset_download
libero_dataset_download(datasets='$ds', download_dir='$LIBERO_ASSETS', use_huggingface=True)
" 2>/dev/null && echo "OK" || echo "SKIP"
        done
        echo "   Done. Activate: source sim/venvs/libero/bin/activate"
        ;;

    # ═══════════════ RoboCasa365 ═══════════════════════════════
    robocasa)
        ROBOCASA_PYTHON_VERSION="${ROBOCASA_PYTHON_VERSION:-3.11}"
        ROBOCASA_COMMIT="${ROBOCASA_COMMIT:-b4684e6ee37d377cc392e98302a6b916d588b415}"
        ROBOSUITE_COMMIT="${ROBOSUITE_COMMIT:-5ce6643f3092639d08f7b0f90ed1c6a84f50552c}"
        SOURCE_DIR="$VENV/src"
        ROBOCASA_DIR="$SOURCE_DIR/robocasa"
        ROBOSUITE_DIR="$SOURCE_DIR/robosuite"
        echo ""
        echo "==> RoboCasa365 → $VENV"

        if [ ! -x "$VENV/bin/python" ]; then
            uv venv -q --python "$ROBOCASA_PYTHON_VERSION" "$VENV"
        fi
        source "$VENV/bin/activate"
        mkdir -p "$SOURCE_DIR"

        if [ ! -d "$ROBOSUITE_DIR/.git" ]; then
            echo "   Cloning robosuite..."
            git clone -q https://github.com/ARISE-Initiative/robosuite.git "$ROBOSUITE_DIR"
        fi
        git -C "$ROBOSUITE_DIR" fetch -q origin "$ROBOSUITE_COMMIT"
        git -C "$ROBOSUITE_DIR" checkout -q "$ROBOSUITE_COMMIT"

        if [ ! -d "$ROBOCASA_DIR/.git" ]; then
            echo "   Cloning RoboCasa..."
            git clone -q https://github.com/robocasa/robocasa.git "$ROBOCASA_DIR"
        fi
        git -C "$ROBOCASA_DIR" fetch -q origin "$ROBOCASA_COMMIT"
        git -C "$ROBOCASA_DIR" checkout -q "$ROBOCASA_COMMIT"

        # RoboCasa 1.0.1 pins Gymnasium <1 through LeRobot while OpenETA's
        # agent process uses Gymnasium >=1.  Keeping this worker isolated is
        # therefore a correctness requirement, not just a convenience.
        uv pip install -q -e "$ROBOSUITE_DIR"
        uv pip install -q -e "$ROBOCASA_DIR"
        uv pip install -q 'mcp>=1.0' 'Pillow>=10' 'starlette>=0.27' 'uvicorn>=0.23'
        # Do not install OpenETA's distribution metadata into this venv: its
        # gymnasium>=1 requirement intentionally conflicts with RoboCasa's
        # official gymnasium<1 stack.  bench_worker adds REPO_ROOT to sys.path.
        if [ ! -f "$ROBOSUITE_DIR/robosuite/macros_private.py" ]; then
            python -m robosuite.scripts.setup_macros
        fi
        if [ ! -f "$ROBOCASA_DIR/robocasa/macros_private.py" ]; then
            python -m robocasa.scripts.setup_macros
        fi

        cat > "$VENV/activate_extra.sh" << SH
export MUJOCO_GL=egl
export ROBOCASA_ROOT=${ROBOCASA_DIR}
export ROBOSUITE_ROOT=${ROBOSUITE_DIR}
SH

        ASSET_MARKER="$VENV/.robocasa_assets_all_complete"
        if [ ! -f "$ASSET_MARKER" ]; then
            echo "   Downloading all official RoboCasa assets (about 23 GB extracted)..."
            echo y | python -m robocasa.scripts.download_kitchen_assets --type all
            touch "$ASSET_MARKER"
        fi
        echo "   Done. Activate: source sim/venvs/robocasa/bin/activate"
        ;;

    # ═══════════════ BEHAVIOR-1K / OmniGibson ══════════════════
    behavior)
        BEHAVIOR_TAG="${BEHAVIOR_TAG:-v3.9.0}"
        BEHAVIOR_COMMIT="${BEHAVIOR_COMMIT:-6559858f7c814143f08be27830d24fac16a12058}"
        SOURCE_DIR="$VENV/src/BEHAVIOR-1K"
        RUNTIME="$VENV/runtime"
        echo ""
        echo "==> BEHAVIOR-1K $BEHAVIOR_TAG → $VENV"

        mkdir -p "$VENV/src"
        if [ ! -d "$SOURCE_DIR/.git" ]; then
            git clone --branch "$BEHAVIOR_TAG" --depth 1 \
                https://github.com/StanfordVL/BEHAVIOR-1K.git "$SOURCE_DIR"
        fi
        git -C "$SOURCE_DIR" fetch -q origin "$BEHAVIOR_TAG"
        git -C "$SOURCE_DIR" checkout -q "$BEHAVIOR_COMMIT"

        if [ ! -x "$RUNTIME/bin/python" ]; then
            source "$(conda info --base)/etc/profile.d/conda.sh"
            conda create -p "$RUNTIME" python=3.11 pip 'setuptools>=71,<81' wheel -c conda-forge -y
        fi
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate "$RUNTIME"
        unset EXP_PATH CARB_APP_PATH ISAAC_PATH
        export CONDA_PLUGINS_AUTO_ACCEPT_TOS=yes
        export OMNI_KIT_ACCEPT_EULA=YES
        (
            cd "$SOURCE_DIR"
            ./setup.sh --omnigibson --bddl --dataset \
                --accept-nvidia-eula --accept-dataset-tos
        )
        uv pip install -q --python "$RUNTIME/bin/python" \
            'mcp>=1.0' 'starlette==0.45.3' 'uvicorn==0.29.0' 'Pillow==11.3.0'

        cat > "$VENV/activate_extra.sh" << SH
export OMNI_KIT_ACCEPT_EULA=YES
export OMNIGIBSON_HEADLESS=True
export OMNIGIBSON_DATA_PATH=${SOURCE_DIR}/datasets
SH
        echo "   Done. Preflight: $RUNTIME/bin/python scripts/behavior_preflight.py"
        ;;

    # ═══════════════ Genesis ═══════════════════════════════════
    genesis)
        echo ""
        echo "==> Genesis → $VENV"
        uv venv -q --python "$PYTHON_VERSION" "$VENV" 2>&1 || true
        source "$VENV/bin/activate"
        uv pip install -q gymnasium numpy torch cloudpickle genesis-world 2>&1 | tail -1
        uv pip install -q -e "$REPO_ROOT" 2>&1 | tail -1
        echo "   Done. Activate: source sim/venvs/genesis/bin/activate"
        ;;

    # ═══════════════ D4RL ═════════════════════════════════════
    d4rl)
        echo ""
        echo "==> D4RL → $VENV"
        echo "   NOTE: Needs MuJoCo 210 binary at ~/.mujoco/mujoco210"
        uv venv -q --python "$PYTHON_VERSION" "$VENV" 2>&1 || true
        source "$VENV/bin/activate"
        uv pip install -q gymnasium numpy torch cloudpickle d4rl 2>&1 | tail -1
        uv pip install -q -e "$REPO_ROOT" 2>&1 | tail -1

        cat > "$VENV/activate_extra.sh" << 'SH'
export LD_LIBRARY_PATH=$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia:$LD_LIBRARY_PATH
export MUJOCO_GL=egl
SH
        echo "   Done. Activate: source sim/venvs/d4rl/bin/activate"
        ;;

    *)
        echo "Unknown bench: '$BENCH'"
        echo "Available: metaworld maniskill libero robocasa behavior genesis d4rl all"
        ;;
    esac
done

# ── Summary ────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "Setup complete!"
echo "================================================================"
echo ""
echo "Installed venvs:"
for d in "$VENVS_DIR"/*/; do
    [ -d "$d" ] && echo "  $(basename "$d")  →  source $d/bin/activate"
done
echo ""
echo "Usage:"
echo "  source sim/venvs/metaworld/bin/activate"
echo "  python -c \"import gymnasium as gym; import sim.env_registry; env = gym.make('openeta/metaworld_50_assembly-v3-v0', render_mode='rgb_array'); env.reset()\""
echo ""
echo "Only the active bench's environments are registered."
