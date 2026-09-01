For the TacEx container **you need the Isaac Lab docker image with Isaac Sim 4.5**. If you have, go directly to [Build TacEx Image](Docker-Container-Setup#build-tacex-image).\
Otherwise, you need to build the docker image with **Isaac Lab 2.1.1**.
Below is a short summary for this and the full guide can be found [here](https://isaac-sim.github.io/IsaacLab/main/source/deployment/docker.html#docker-guide).

<details><summary> Build Isaac Lab Base Image</summary>

> [!note] Prerequisites
>
> You need the Nvidia drivers, Docker and the Nvidia Container Toolkit installed (see [container setup](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/install_container.html#container-setup))
>
> - Helpful for Driver Installation on Linux: [Driver Installation Linux](https://docs.omniverse.nvidia.com/dev-guide/latest/linux-troubleshooting.html#q1-how-to-install-a-driver)
>
> For downloading the Isaac Sim container, you need to setup your Nvidia API key:
>
> - Install the ngc client https://org.ngc.nvidia.com/setup/installers/cli (for the config just press enter twice. i.e. use the default config)
> - For setting up the API key: [api key setup](https://org.ngc.nvidia.com/setup/api-key) ![image](https://github.com/user-attachments/assets/f773bcc2-fed0-4266-9fb2-10e23b9f874f)

You need to have the Isaac Lab repo cloned:
```bash
git clone https://github.com/isaac-sim/IsaacLab
cd IsaacLab
# use Isaac Lab version 2.1.1
git checkout v2.1.1
```

Then you can build the base image via:
```bash
# call this inside the root directory of Isaac Lab repo
./docker/container.py start
```

Once you have built the base Isaac Lab image, you can check it exists via:

```bash
docker images

# Output should look like this:
#
# REPOSITORY                       TAG       IMAGE ID       CREATED          SIZE
# isaac-lab-base                   latest    28be62af627e   32 minutes ago   16.9GB
```
</details>

# Build TacEx Image
When you have the Isaac Lab container, you can build the docker container for this project.
Go into the source directory of the TacEx repository and run
```bash
# assuming you are in the root directory of the TacEx Repo
./docker/container.py build # this command also starts the container in the background
```

> [!note]
>
> For simplicity we use the container script from Isaac Lab (slightly modified) for building, starting, entering and stopping the container. Additional features, such as different container profiles, are currently not supported here.

Verify that the image (`isaac-lab-tacex`) was built successfully:

```bash
docker images

# Output should look something like:
#
# REPOSITORY                       TAG       IMAGE ID       CREATED             SIZE
# isaac-lab-tacex                  latest    1eaf4cd9dce3   20 seconds ago      22.6GB
# isaac-lab-base                   latest    892938acb55c   About an hour ago   16.9GB
```

> [!note]
>
> If you, e.g., want to use the Isaac Lab ROS image instead of the base image `isaac-lab-base` for the TacEx container, then you need to adjust the name `ISAACLAB_BASE_IMAGE` in the `docker/.env.base` file of this repository.

# Running the container

Start the container with:

```bash
#  will start the services defined in our `docker-compose.yaml` file in detached mode.
./docker/container.py start
```
To enter the container use

```bash
./docker/container.py enter
```

and to stop it

```bash
./docker/container.py stop
```

<!---
> [!tip]
>
> The container script can be found in `./docker/container.py`. Just setup an alias in your `~/.bashrc` file for conveniently calling it. For example via `alias tacex="/path_to_repo/docker/container.py"`.-->

# Install TacEx inside container

Start the container and enter it:
```bash
./docker/container.py start
./docker/container.py enter
```
Then install the TacEx pip packages:
```bash
# This command pip installs every pip package in the `tacex/source` directory.
./tacex.sh -i all
```
> Delete the `source/tacex_uipc/build` folder if tacex_uipc cannot be build and try the installation again.

To verify the installation, run
```bash
python /workspace/tacex/scripts/benchmarking/tactile_sim_performance/run_ball_rolling_experiment.py --num_envs 1 --debug_vis --env uipc
```

# Livestreaming

Use the [streaming client](https://docs.isaacsim.omniverse.nvidia.com/4.5.0/installation/download.html#isaac-sim-latest-release) to view the Isaac Sim GUI.

Example usage:
- open the Streaming Client
- run a command
```bash
# --livestream 1 if you use the Omniverse Streaming Client
python /workspace/tacex/scripts/benchmarking/tactile_sim_performance/run_ball_rolling_experiment.py --livestream 2
```
- connect


# Set up VScode (Optional)

For development inside the container you need the [Remote Explorer Extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode.remote-explorer). With this, you can just attach to the running container. Once you did, just open the TacEx folder `/workspace/tacex` and you are almost good to go.

<!--- make sure that env variables `${ISAACLAB_PATH}` and `ISAACLAB_EXTENSION_TEMPLATE_PATH` are set properly.
This is done automatically in the docker setup. You can set it manually like this:
```bash
export ISAACLAB_PATH="/path_to/isaaclab"
export ISAACLAB_EXTENSION_TEMPLATE_PATH="/path_to/tacex"
```-->

Some more additional steps for a smoother development experience:

- Create a symbolic link to easily access the Isaac Lab files in the container:

```bash
ln -s /workspace/isaaclab /workspace/tacex/_isaaclab
```

- Run VScode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu. When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

> For the docker setup this is `/workspace/isaaclab/_isaac_sim`, if you added the symbolic link

Now you should have

- `.vscode/launch.json`: Contains the launch configurations for debugging python code.
- `.vscode/settings.json`: Contains the settings for the python interpreter and the python environment.

If IsaacLab and IsaacSim python modules are not indexed correctly (i.e., the IDE cannot find them when being in the code editor), then you need to adjust the `"python.analysis.extraPaths"` in the `.vscode/settings.json` file.

> [!note]
>
> For the docker setup you can also just use the `/docker/settings.json` file of this repo to replace the one in the `.vscode` folder for indexing IsaacLab and TacEx modules. (Idk why the ones from IsaacSim don't work right now.)

## Code Formatting

To use pre-commit in the docker container, install it first:

```bash
# `pip install pre-commit` only works with a virtual env
apt-get update
apt-get -y install pre-commit
```

Pre-commit also needs some extra modules you need to reinstall due to the docker setup:

```bash
python -m pip install pyyaml filelock --force-reinstall
```

Check that the installation was successful:

```bash
pre-commit --version
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```

## Debugging

If you did the setup correctly, you should have a launch config "Docker: Current File" for debugging.
For more complex commands, such as RL training commands, you can add custom launch configs, which look like this:

```bash
        {
            "name": "Docker: Train Environment",
            "type": "debugpy",
            "request": "launch",
            "args" : ["--task", "Template-Isaac-Velocity-Flat-Anymal-D-v0", "--num_envs", "4096", "--headless"],
            "program": "${workspaceFolder}/scripts/rsl_rl/train.py",
            "console": "integratedTerminal",
            "env": {
                "PYTHONPATH": "${env:PYTHONPATH}:${workspaceFolder}"
            }
        },
```

with `args` and `program` adjusted.

Instead of doing this for every command, you can also just add one launch config like this:

```bash
        {
            "name": "Python Debugger: Remote Attach",
            "type": "debugpy",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 3000
            },
            "pathMappings": [
                {
                    "localRoot": "${workspaceFolder}",
                    "remoteRoot": "."
                }
            ]
        },
```

then run

```bash
lab -p -m debugpy --listen 3000 --wait-for-client _your_command_
```

and attach via your VScode debugger (launch the config above).

# Interacting with a running container

If you want to run commands inside the running container from "outside" the container, you can use the `exec` command:

```bash
docker exec --interactive --tty -e DISPLAY=${DISPLAY} isaac-lab-tacex /bin/bash
```

## Shutting down the container

When you are done or want to stop the running containers, you can bring down the services:

```bash
docker compose --env-file .env.base --file docker-compose.yaml down
```

This stops and removes the running containers, but keeps the images.
