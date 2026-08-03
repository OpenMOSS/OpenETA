# Simulation Environment Setup Guide

本指南覆盖 `sim/envs/` 中 14 个仿真环境的依赖和安装方法。

---

## 快速开始（最小安装）

```bash
# 创建 uv 管理的虚拟环境
uv venv --python 3.10
source .venv/bin/activate

# 核心依赖（所有 env 都需要）
uv pip install gymnasium numpy torch omegaconf

# OpenETA 自身（开发模式）
cd /path/to/openeta
uv pip install -e ".[dev]"
```

验证：

```bash
python -c "
import gymnasium as gym
import sim.env_registry
env = gym.make('openeta/dummy_sim-v0')
obs, _ = env.reset()
print('OK:', obs['task'])
"
```

---

## 环境总览

| 环境 | 复杂度 | GPU | 额外仿真器 | 安装难度 |
|------|--------|-----|-----------|---------|
| `dummy` | ★ | 否 | 无 | 开箱即用 |
| `d4rl` | ★ | 否 | MuJoCo (自带) | `uv pip install d4rl` |
| `metaworld` | ★★ | 否 | MuJoCo (自带) | `uv pip install metaworld` |
| `genesis` | ★★ | 是 | Genesis | `uv pip install genesis-world` |
| `maniskill` | ★★★ | 是 | SAPIEN | `uv pip install mani-skill` |
| `frankasim` | ★★★ | 是 | MuJoCo | `uv pip install frankasim` |
| `libero` | ★★★★ | 是 | PyBullet + LIBERO assets | 需要 clone + asset 下载 |
| `calvin` | ★★★★ | 是 | PyBullet + CALVIN assets | 需要 clone + dataset |
| `habitat` | ★★★★ | 是 | Habitat-Sim | 需要 conda 安装 |
| `robocasa` | ★★★★★ | 是 | robosuite + assets | 需要 clone + asset 下载 |
| `robotwin` | ★★★★★ | 是 | RoboTwin + digital twin | 需要 clone + asset 下载 |
| `roboverse` | ★★★★★ | 是 | RoboVerse + MetaSim | 需要 clone + asset 下载 |
| `behavior` | ★★★★★ | 是 | OmniGibson + Isaac Sim | 需要 NVIDIA Omniverse |
| `isaaclab` | ★★★★★ | 是 | Isaac Sim | 需要 Isaac Sim 安装 |
| `polaris` | ★★★★★ | 是 | Isaac Sim + PolaRiS | 同 IsaacLab |
| `embodichain` | ★★★ | 否 | 无（纯数值） | `uv pip install embodichain` |

---

## 详细安装步骤

### 通用前提

```bash
# 创建项目 venv
uv venv --python 3.10
source .venv/bin/activate

# 基础依赖
uv pip install gymnasium numpy torch omegaconf cloudpickle

# OpenETA
uv pip install -e .
```

### D4RL (locomotion, CPU 可跑)

```bash
uv pip install d4rl
```

**验证：**
```bash
python -c "
import gymnasium as gym
import sim.env_registry
env = gym.make('openeta/d4rl_halfcheetah-medium-v2')
obs, _ = env.reset()
print('OK:', obs.shape)
"
```

### MetaWorld (manipulation, CPU 可跑)

```bash
uv pip install metaworld
```

**验证：**
```bash
python -c "
import gymnasium as gym
import sim.env_registry
env = gym.make('openeta/metaworld_50_assembly-v0')
obs, _ = env.reset()
print('OK')
"
```

### Genesis (GPU required)

Genesis 是目前**最容易安装的 GPU 仿真器**（纯 pip）。

```bash
uv pip install genesis-world
```

**已知问题：**
- 需要 CUDA 12.x + 兼容的 NVIDIA 驱动 (>= 525)
- `nvJitLink error` → Genesis 版本与 CUDA 版本不匹配，升级 `genesis-world` 或降级 CUDA toolkit
- `pip install --upgrade genesis-world` 通常能解决
- Genesis 要求 `torch` 已安装且有 CUDA 支持

**验证：**
```bash
python -c "
import genesis; print('Genesis:', genesis.__version__)
import torch; print('CUDA:', torch.cuda.is_available())
# 如果上面通过，测试 env
import gymnasium as gym
import sim.env_registry
env = gym.make('openeta/genesis_cube_pick-v0')
obs, _ = env.reset()
print('OK')
"
```

### ManiSkill (GPU required)

```bash
uv pip install mani-skill
```

**注意：**
- ManiSkill 依赖 SAPIEN 物理引擎，后者需要特定 CUDA 版本
- 官方推荐 CUDA 11.8 或 12.1
- 如果 `pip install mani-skill` 失败，尝试 `pip install mani-skill --find-links https://maniskill2.github.io/whl/`

**验证：**
```bash
python -c "
import mani_skill
print('ManiSkill:', mani_skill.__version__)
# ManiSkill 环境会自动注册到 gymnasium
from mani_skill.utils.registration import REGISTERED_ENVS
print('Tasks:', len(REGISTERED_ENVS))
"
```

### LIBERO (GPU required)

```bash
# 1. Clone LIBERO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO

# 2. 安装
uv pip install -e .

# 3. 下载 assets
python -m libero.scripts.download_libero_assets

# 4. 设置环境变量
export LIBERO_ASSET_ROOT=$(python -c "import libero; import os; print(os.path.join(os.path.dirname(libero.__file__), 'assets'))")
```

**验证：**
```bash
python -c "
from libero.libero.benchmark import Benchmark
bench = Benchmark(task_suite_name='libero_spatial')
print('Tasks:', bench.get_num_tasks())
"
```

### CALVIN (GPU required)

```bash
# 1. Clone CALVIN
git clone https://github.com/mees/calvin.git
cd calvin

# 2. 安装
uv pip install -e .
uv pip install calvin-models calvin-env

# 3. 下载 dataset
# (见 https://github.com/mees/calvin 说明)
```

### RoboCasa365

```bash
# 创建 Python 3.11 隔离环境，checkout 已验证的 RoboCasa/robosuite commit，
# 安装 worker 依赖并下载全部官方 assets（解压后约 23 GB）
./scripts/setup_envs.sh robocasa
source sim/venvs/robocasa/bin/activate
source sim/venvs/robocasa/activate_extra.sh

# 验证 317 tasks / 634 split-specific env IDs
python -m scripts.robocasa_benchmark task-sets
python -c "from sim.env_registry import list_envs; print(len([x for x in list_envs() if x.env_type == 'robocasa']))"
```

这里必须使用独立 venv：RoboCasa 1.0.1 的官方依赖链需要
`gymnasium==0.29.x`，而 OpenETA 主环境需要 `gymnasium>=1.0`。MCP server
会自动用 `sim/venvs/robocasa/bin/python` 启动 worker，主 agent 不会导入
RoboCasa/MuJoCo。完整 benchmark 用法见 `sim/ROBOCASA.md`。

### RoboTwin (GPU required)

```bash
git clone https://github.com/RoboTwin-Platform/RoboTwin.git
cd RoboTwin
uv pip install -e .
```

### RoboVerse (GPU required)

```bash
git clone https://github.com/RoboVerseOrg/RoboVerse.git
cd RoboVerse
uv pip install -e .
# MetaSim 作为仿真后端
uv pip install metasim
```

### BEHAVIOR-1K v3.9 / OmniGibson (GPU + Isaac Sim required)

The supported path is automated and version-pinned:

```bash
bash scripts/setup_envs.sh behavior
python scripts/behavior_preflight.py --output outputs/behavior/preflight.json
source sim/venvs/behavior/activate_extra.sh
sim/venvs/behavior/runtime/bin/python scripts/behavior_smoke.py
```

See `sim/BEHAVIOR.md` for the exact versions, assets, worker design, and GPU
diagnostics. The environment uses Python 3.11 in an isolated conda prefix and
does not share packages with the main Agent process.

### IsaacLab / Polaris (GPU + Isaac Sim required)

```bash
# Isaac Sim 必须先安装
git clone https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab
./isaaclab.sh --install
```

### Habitat (GPU required, 推荐 conda)

```bash
# Habitat 推荐 conda 安装
conda install habitat-sim -c conda-forge -c aihabitat
uv pip install habitat-lab habitat-baselines
```

### FrankaSim (GPU required)

```bash
uv pip install frankasim
```

### EmbodiChain (CPU 可跑)

```bash
uv pip install embodichain
```

---

## 完整 uv 环境示例 (D4RL + MetaWorld + Genesis)

```bash
# 创建环境
uv venv --python 3.10
source .venv/bin/activate

# 核心
uv pip install gymnasium numpy torch omegaconf cloudpickle

# 轻量仿真器 (CPU)
uv pip install d4rl metaworld

# GPU 仿真器
uv pip install genesis-world

# OpenETA
uv pip install -e /path/to/openeta

# 验证所有可用环境
python -c "
from sim.env_registry import list_envs
for s in list_envs():
    print(f'  {s.id:60s} [{s.env_type}]')
"
```

---

## 常见问题

### `ModuleNotFoundError: No module named 'omegaconf'`

```bash
uv pip install omegaconf
```

### `AssertionError: An empty Dict observation space is not allowed`

Dummy env 的 gymnasium 兼容性警告，不影响使用。

### Genesis `nvJitLink error: Internal error`

Genesis 与当前 CUDA 版本不兼容。尝试：
```bash
uv pip install --upgrade genesis-world torch
```
或使用 RLinf 官方的 Docker 镜像（预装所有依赖）。

### `from sim.envs.venv import ...` 导入失败

`sim/envs/venv/__init__.py` 是精简桩实现，仅支持 `num_envs=1`。
如需多进程并行环境 (`num_envs > 1`)，安装完整 RLinf：
```bash
uv pip install rlinf[embodied]
```

---

## RLinf 官方的安装方式（参考）

RLinf 官方使用 Docker 作为主要分发方式，支持以下预构建镜像：

```bash
# 运行 (以 Genesis 为例)
docker run -it --rm --gpus all --shm-size 32g --network host \
    --name rlinf -v .:/workspace/RLinf \
    rlinf/rlinf:agentic-rlinf0.3-genesis
```

Docker 镜像 tag 格式：`rlinf/rlinf:agentic-rlinf0.3-<env_name>`

RLinf 也提供了安装脚本（不在 vendored subset 中）：
```bash
bash requirements/install.sh embodied --env genesis        # genesis only
bash requirements/install.sh embodied --env maniskill_libero  # ManiSkill + Libero
bash requirements/install.sh embodied --env behavior       # BEHAVIOR (OmniGibson)
```

这些脚本内部会自动处理 conda 环境、CUDA 兼容性和 Python 版本 (3.10)。

---

## Python 版本要求

| 环境 | Python |
|------|--------|
| 大部分 env | >= 3.10 |
| BEHAVIOR | 3.10 only |
| D4RL | 3.10 only |
| RLinf 默认 | 3.11.14 |
| OpenETA 推荐 | 3.10（最大兼容性） |
