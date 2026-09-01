# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions specific to the in-hand dexterous manipulation environments."""

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

from tacex import GelSightSensor

if TYPE_CHECKING:
    from .commands import InHandReOrientationCommand


def goal_quat_diff(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, make_quat_unique: bool
) -> torch.Tensor:
    """Goal orientation relative to the asset's root frame.

    The quaternion is represented as (w, x, y, z). The real part is always positive.
    """
    # extract useful elements
    asset: RigidObject = env.scene[asset_cfg.name]
    command_term: InHandReOrientationCommand = env.command_manager.get_term(command_name)

    # obtain the orientations
    goal_quat_w = command_term.command[:, 3:7]
    asset_quat_w = asset.data.root_quat_w

    # compute quaternion difference
    quat = math_utils.quat_mul(asset_quat_w, math_utils.quat_conjugate(goal_quat_w))
    # make sure the quaternion real-part is always positive
    return math_utils.quat_unique(quat) if make_quat_unique else quat


def mean_marker_motion(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Mean (x,y) of the initial and current tactile marker displacements."""

    gsmini_ring: GelSightSensor = env.scene["gsmini_ring"]
    marker_motion_data = gsmini_ring.data.output["marker_motion"]
    x_mean_ring = torch.mean(marker_motion_data[:, :, :, 0], dim=2)
    y_mean_ring = torch.mean(marker_motion_data[:, :, :, 1], dim=2)

    gsmini_middle: GelSightSensor = env.scene["gsmini_middle"]
    marker_motion_data = gsmini_middle.data.output["marker_motion"]
    x_mean_middle = torch.mean(marker_motion_data[:, :, :, 0], dim=2)
    y_mean_middle = torch.mean(marker_motion_data[:, :, :, 1], dim=2)

    gsmini_index: GelSightSensor = env.scene["gsmini_index"]
    marker_motion_data = gsmini_index.data.output["marker_motion"]
    x_mean_index = torch.mean(marker_motion_data[:, :, :, 0], dim=2)
    y_mean_index = torch.mean(marker_motion_data[:, :, :, 1], dim=2)

    gsmini_thumb: GelSightSensor = env.scene["gsmini_thumb"]
    marker_motion_data = gsmini_thumb.data.output["marker_motion"]
    x_mean_thumb = torch.mean(marker_motion_data[:, :, :, 0], dim=2)
    y_mean_thumb = torch.mean(marker_motion_data[:, :, :, 1], dim=2)

    mean_curr_marker = torch.cat(
        (
            x_mean_ring,
            y_mean_ring,
            x_mean_middle,
            y_mean_middle,
            x_mean_index,
            y_mean_index,
            x_mean_thumb,
            y_mean_thumb,
        ),
        dim=-1,
    )

    return mean_curr_marker
