# Adjusted version from https://github.com/isaac-sim/IsaacLab/tree/9d6321463067c541ce1a24531ff87f99a18fd8f7/source/isaaclab_tasks/test

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Launch Isaac Sim Simulator first."""

import sys

# Import pinocchio in the main script to force the use of the dependencies installed by IsaacLab and not the one installed by Isaac Sim
# pinocchio is required by the Pink IK controller
if sys.platform != "win32":
    import pinocchio  # noqa: F401

from isaaclab.app import AppLauncher

# launch the simulator
app_launcher = AppLauncher(headless=True, enable_cameras=True)  # with headless, our camera envs don't seem to work
# app_launcher = AppLauncher(headless=False, enable_cameras=True)  # this does not work in docker container
simulation_app = app_launcher.app


"""Rest everything follows."""

import pytest
from env_test_utils import _run_environments, setup_environment

# import isaaclab_tasks  # noqa: F401
import tacex_tasks  # noqa: F401


@pytest.mark.parametrize("num_envs, device", [(32, "cuda"), (1, "cuda")])
@pytest.mark.parametrize("task_name", setup_environment(include_play=False, factory_envs=False, multi_agent=False))
def test_environments(task_name, num_envs, device):
    # run environments without stage in memory
    _run_environments(task_name, device, num_envs, create_stage_in_memory=False)
