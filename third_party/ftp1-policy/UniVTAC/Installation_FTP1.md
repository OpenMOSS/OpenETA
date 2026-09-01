# UniVTAC Installation Guide

This document summarizes the installation procedure for the current UniVTAC version in this repository.
It is based on the original installation notes in `UniVTAC/docs/Installation_org.md`, but reorganized into a single English guide with the current repository layout and scripts.

## Overview

UniVTAC depends on the following major components:

- Linux with an NVIDIA GPU
- Python 3.10
- NVIDIA Isaac Sim `4.5.0`
- Isaac Lab `2.1.1`
- cuRobo
- TacEx from the local modified source at `UniVTAC/third_party/TacEx`
- FTP1 inference integrated into the Isaac Sim environment

The recommended setup is to use one Conda environment for:

- Isaac Sim
- Isaac Lab
- TacEx
- FTP1 inference

This guide assumes the environment name is `uni`.

## Prerequisites

Before starting, make sure you have:

- a Linux machine with NVIDIA drivers installed
- `conda`
- `git`
- `gcc` and `g++`
- enough disk space for Isaac Sim, Isaac Lab, TacEx build artifacts, and model dependencies

## Step 0: Remove an old environment if needed

If you want to rebuild the environment from scratch:

```bash
conda env remove -n uni
```

## Step 1: Clone the repository


Clone the repository:

```bash
# git clone https://github.com/byml-c/UniVTAC.git
cd UniVTAC
```

## Step 2: Create the Conda environment

Create and activate the recommended shared environment:

```bash
CONDA_SOLVER=classic conda create -n uni python=3.10 -y
conda activate uni
```

All remaining steps in this guide should be executed inside:

```bash
conda activate uni
```

## Step 3: Install Isaac Sim 4.5.0

Install CUDA-enabled PyTorch first:

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
pip install --upgrade pip
```

Install system packages required by Isaac Sim:

```bash
sudo apt-get update
sudo apt-get install gcc-11 g++-11
```

Install Isaac Sim:

```bash
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
```

Verify that Isaac Sim launches:

```bash
isaacsim --allow-root --headless
```

## Step 3.1: Fix Vulkan issues if Isaac Sim fails to start

On some machines, Isaac Sim fails because of Vulkan configuration issues. If that happens, install the Vulkan-related packages first:

```bash
sudo apt-get update
sudo apt-get install -y libvulkan1 vulkan-tools
sudo apt-get install -y libnvidia-gl-$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1) 2>/dev/null || true
sudo apt-get install -y nvidia-vulkan-icd 2>/dev/null || true
```

You can inspect possible NVIDIA Vulkan libraries with:

```bash
find /usr -name "*nvidia*vulkan*" 2>/dev/null
ls /usr/lib/x86_64-linux-gnu/libnvidia-glcore.so* 2>/dev/null
```

If needed, create an NVIDIA ICD file:

```bash
sudo mkdir -p /usr/share/vulkan/icd.d
echo '{"file_format_version": "1.0.0", "ICD": {"library_path": "/usr/lib/x86_64-linux-gnu/libnvidia-glcore.so.$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)", "api_version": "1.1.0"}}' | sudo tee /usr/share/vulkan/icd.d/nvidia_icd.json
```

Verify Vulkan:

```bash
# With display
vulkaninfo --summary

# Headless
DISPLAY= vulkaninfo --summary 2>&1 | head -30
```

## Step 4: Install Isaac Lab 2.1.1

Install additional system dependencies:

```bash
sudo apt install cmake build-essential
```

Clone and install Isaac Lab:

```bash
git clone https://github.com/isaac-sim/IsaacLab
cd IsaacLab
git fetch origin tag v2.1.1 && git checkout v2.1.1
./isaaclab.sh --install
```

Before editable installation, update the `flatdict` requirement:

- open `source/isaaclab/setup.py`
- replace `flatdict==4.0.1` with `flatdict==4.1.0`

Then install Isaac Lab into the current environment:

```bash
pip install --upgrade setuptools
pip install -e source/isaaclab
```

Verify Isaac Lab:

```bash
conda activate uni
cd /path/to/IsaacLab
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py -- --task=Isaac-Ant-v0 --headless
```

## Step 5: Install FTP1 inference into the Isaac Sim environment

First update line 6 of `../pyproject.toml` to `requires-python = ">=3.10"`.

Then from the `ftp1` repository root, run:

```bash
bash scripts/shell/install_ftp1_infer_into_isaacsim.sh
```

This is the recommended method for UniVTAC because it installs FTP1 inference without replacing Isaac Sim's PyTorch stack.

After installation, the same `uni` environment should support both:

- Isaac Lab / TacEx simulation
- `FTP1InferenceWrapper`-based inference

## Step 6: Install TacEx from the local modified source

UniVTAC requires the modified TacEx source shipped inside this repository:

```bash
cd /path/to/ftp1/UniVTAC/third_party/TacEx
```

Do **not** install TacEx from the public upstream repository for this project setup.

### Step 6.1: Install TacEx core

Make sure you are still inside the Isaac Sim / Isaac Lab environment:

```bash
conda activate uni
```

Install TacEx core packages:

```bash
./tacex.sh -i
```

Verify TacEx with a simple demo:

```bash
python ./scripts/demos/tactile_sim_approaches/check_taxim_sim.py --debug_vis
```

You can also test an RL example:

```bash
python ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Ball-Rolling-Tactile-RGB-v0 --num_envs 512 --enable_cameras
```

If the simulation runs correctly, you can inspect tactile outputs in the Isaac Lab UI under:

`Scene Debug Visualization > Observations > sensor_output`

### Step 6.2: Install TacEx UIPC support

The `tacex_uipc` package is responsible for UIPC-based tactile simulation in TacEx.

#### 6.2.1 Install Vcpkg

If Vcpkg is not installed yet:

```bash
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
sudo apt-get update
sudo apt-get install -y curl zip unzip tar
./bootstrap-vcpkg.sh -disableMetrics
```

Install `spdlog` through Vcpkg:

```bash
./vcpkg install spdlog:x64-linux
```

#### 6.2.2 Export `CMAKE_TOOLCHAIN_FILE`

Set the Vcpkg toolchain path so that CMake can find it:

```bash
export CMAKE_TOOLCHAIN_FILE="${YOUR_PATH}/Toolchain/vcpkg/scripts/buildsystems/vcpkg.cmake"
```

You may want to put the export into `~/.bashrc`.

#### 6.2.3 Install the build environment for libuipc

TacEx UIPC currently expects:

- CMake `3.26`
- GCC `11.4`
- CUDA `12.4`

Install the conda environment additions from TacEx:

```bash
cd /path/to/ftp1/UniVTAC/third_party/TacEx
conda activate uni
conda env update -n uni --file ./source/tacex_uipc/libuipc/conda/env.yaml
```

If CUDA 12.4 does not work on your machine, update your NVIDIA driver or adjust the TacEx environment file to use an older CUDA version such as 12.2.

#### 6.2.4 Install `tacex_uipc`

Make sure `CMAKE_TOOLCHAIN_FILE` is set, then install:

```bash
conda activate uni
rm -rf source/tacex_uipc/build
export CMAKE_TOOLCHAIN_FILE="${YOUR_PATH}/Toolchain/vcpkg/scripts/buildsystems/vcpkg.cmake"
pip install -e source/tacex_uipc -v
```

You may also install all TacEx packages through:

```bash
./tacex.sh -i all
```

#### 6.2.5 Verify `tacex_uipc`

Run:

```bash
python ./scripts/benchmarking/tactile_sim_performance/run_ball_rolling_experiment.py --num_envs 1 --debug_vis --env uipc
```

## Step 7: Install cuRobo

UniVTAC uses cuRobo for GPU-accelerated collision-aware motion planning.

Follow the official cuRobo guide:

- https://curobo.org/get_started/1_install_instructions.html

## Step 8: Fix `transformers_replace` inside the Conda environment

If FTP1 evaluation reports an error similar to:

> `transformers_replace is not installed correctly`

copy the custom transformer replacement files into the active Conda environment instead of a `.venv`.

For Python `3.10`, run from the `ftp1` repository root:

```bash
conda activate uni
cd /path/to/ftp1
cp -r ./src/openpi/models_pytorch/transformers_replace/* "$CONDA_PREFIX/lib/python3.10/site-packages/transformers/"
```

If your environment uses a different Python version, replace `python3.10` with the actual version path.

## Step 9: Install ffmpeg

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
ffmpeg -version
```

## Step 10: Fix `torch-scatter` if needed

If `torch_scatter` has binary compatibility issues in the current environment:

```bash
conda activate uni
pip uninstall torch_scatter
pip install torch_scatter --no-binary :all: --no-build-isolation
```

## Recommended verification checklist

Before using UniVTAC for collection or evaluation, verify the following in order:

1. `isaacsim --allow-root --headless` launches successfully.
2. Isaac Lab runs a basic training script.
3. `bash UniVTAC/scripts/shell/install_ftp1_infer_into_isaacsim.sh` completes without replacing `torch`.
4. FTP1 inference imports correctly in the `uni` environment.
5. TacEx core demo runs.
6. `tacex_uipc` example runs if UIPC simulation is needed.
7. `ffmpeg -version` works.

## Notes

- The current installation flow is designed around a single shared Conda environment called `uni`.
- For this repository version, the local TacEx source under `UniVTAC/third_party/TacEx` is the required source of truth.
- The FTP1 integration path used here is the script:
  `UniVTAC/scripts/shell/install_ftp1_infer_into_isaacsim.sh`
- This guide intentionally keeps the troubleshooting commands from the original notes because they are still useful on real deployment machines.
