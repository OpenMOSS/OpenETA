If you have a working Isaac Lab environment, you can directly [install TacEx](Local-Installation#installing-tacex).\
Otherwise, **you need to install Isaac Sim 4.5 and Isaac Lab 2.1.1**.
Below is a quick summary, but here is the [full installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).

<details>
<summary>Quick summary for Installing Isaac Sim and Isaac Lab for Ubuntu 22.04</summary>

> [!note]
> To install Isaac Sim for Ubuntu 20.04 follow the [binary installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/binaries_installation.html).

### Isaac Sim - Linux pip installation

```bash
# create virtual environment
conda create -n env_isaaclab python=3.10
conda activate env_isaaclab
# install cuda-enabled pytorch
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
pip install --upgrade pip
# install isaac sim packages
pip install 'isaacsim[all,extscache]==4.5.0' --extra-index-url https://pypi.nvidia.com
```

> verify that the Isaac Sim installation works by calling `isaacsim` in the terminal

### Isaac Lab

```bash
# install dependencies via apt (Ubuntu)
sudo apt install cmake build-essential
git clone https://github.com/isaac-sim/IsaacLab
cd IsaacLab
# use Isaac Lab version 2.1.1
git checkout v2.1.1
# activate the Isaac Sim python env
conda activate env_isaaclab
# install isaaclab extensions (with --editable flag)
./isaaclab.sh --install # or "./isaaclab.sh -i"
```

To verify the Isaac Lab Installation:

```bash
conda activate env_isaaclab
python scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Ant-v0 --headless
```

</details>

# Installing TacEx [Core]

**0.** If you haven't already done so, clone the repository and its submodules:

```bash
# Need it for the USD assets
git lfs install
git clone --recurse-submodules https://github.com/DH-Ng/TacEx.git
cd TacEx
```
**1.** Activate the Isaac Env
```bash
conda activate env_isaaclab
```

**2.** Install the core packages of TacEx
```bash
# Script will pip install core TacEx packages with --editable flag)
./tacex.sh -i
```

> You can install the extensions one by one via e.g. `python -m pip install -e source/tacex_uipc`

**3.** Verify that TacEx works by running an example:

```bash
python ./scripts/demos/tactile_sim_approaches/check_taxim_sim.py --debug_vis
```

And here is an RL example:
```bash
python ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Ball-Rolling-Tactile-RGB-v0 --num_envs 512 --enable_cameras
```
> You can view the sensor output in the IsaacLab Tab: `Scene Debug Visualization > Observations > sensor_output`

# Installing TacEx [UIPC]
The `tacex_uipc` package is responsible for the [UIPC](https://spirimirror.github.io/libuipc-doc/) simulation in TacEx.

**1.** Install the [libuipc dependencies](https://spirimirror.github.io/libuipc-doc/build_install/linux/):
* If not installed yet, install Vcpkg

```bash
mkdir ~/Toolchain
cd ~/Toolchain
git clone https://github.com/microsoft/vcpkg.git
cd vcpkg
./bootstrap-vcpkg.sh -disableMetrics
```

* Set the System Environment Variable  `CMAKE_TOOLCHAIN_FILE` to let CMake detect Vcpkg. If you installed it like above, you can do this:

```bash
# Write in ~/.bashrc
export CMAKE_TOOLCHAIN_FILE="$HOME/Toolchain/vcpkg/scripts/buildsystems/vcpkg.cmake"
```

* We also need `CMake 3.26`, `GCC 11.4` and `Cuda 12.4` to build libuipc. Install this into the Isaac Sim python env:

```bash
# Inside the root dir of TacEx repo
conda activate env_isaaclab
conda env update -n env_isaaclab --file ./source/tacex_uipc/libuipc/conda/env.yaml
```
> If Cuda 12.4 does not work for, try updating your Nvidia drivers or try to use an older Cuda version by adjusting the env.yaml file (e.g. Cuda 12.2).

**2.** Install `tacex_uipc`
```bash
# This also builds `libuipc` and pip installs the python bindings.
conda activate env_isaaclab
pip install -e source/tacex_uipc -v
```
> You can also install all TacEx packages with `./tacex.sh -i all`.

**3.** Verify that the `tacex_uipc` works by running an example:

```bash
python ./scripts/benchmarking/tactile_sim_performance/run_ball_rolling_experiment.py --num_envs 1 --debug_vis --env uipc
```

# Code formatting

There is a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

# Setup Vscode

Full setup Guide can be found [here](https://isaac-sim.github.io/IsaacLab/main/source/overview/developer-guide/vs_code.html#setting-up-visual-studio-code).

In a nutshell:

1. Run VSCode Tasks, by pressing `Ctrl+Shift+P`and selecting `Tasks: Run Task`
2. Select `setup_python_env` in the drop down menu.

Now you should have

- `.vscode/launch.json`, which contains the launch configurations for debugging python code.
- `.vscode/settings.json`, which contains the settings for the python interpreter and the python environment.
