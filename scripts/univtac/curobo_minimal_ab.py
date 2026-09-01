#!/usr/bin/env python3
"""Replay the native Insert Hole reset failure directly through cuRobo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import warp as wp
import yaml

import curobo
from curobo.geom.sdf.world import CollisionCheckerType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
    PoseCostMetric,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROBOT_CONFIG = (
    REPO_ROOT
    / "third_party/ftp1-policy/UniVTAC/assets/embodiments/franka/curobo.yml"
)
QPOS = [
    -0.047443192452192307,
    0.09495621174573898,
    0.00030604409403167665,
    -1.7838352918624878,
    -0.00006868993659736589,
    1.8912733793258667,
    0.7383537292480469,
]
QVEL = [
    -0.004970676731318235,
    -0.002374921692535281,
    -0.0010871444828808308,
    0.0002643370535224676,
    -0.0000395484094042331,
    0.0014430003939196467,
    0.00018310020095668733,
]
# Curobo Pose.from_list uses [x, y, z, qw, qx, qy, qz].
GOAL = [
    0.7831483324359159,
    -0.0555652499277238,
    0.5949957847076629,
    0.006293902277035414,
    0.00030904584593184425,
    0.99998014527387,
    -0.000018516001605822785,
]
CONSTRAINT_POSE = [1, 1, 1, 1, 1, 0]
PRE_DIS = 0.008
JOINT_NAMES = [f"panda_joint{index}" for index in range(1, 8)]


def _scalar(value):
    if isinstance(value, torch.Tensor):
        flat = value.detach().cpu().reshape(-1)
        return flat.tolist() if flat.numel() != 1 else flat.item()
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curobo-commit", required=True)
    parser.add_argument("--robot-config", type=Path, default=DEFAULT_ROBOT_CONFIG)
    args = parser.parse_args()

    robot_config_path = args.robot_config.expanduser().resolve()
    with robot_config_path.open("r", encoding="utf-8") as stream:
        robot_config = yaml.safe_load(stream)
    config_dir = robot_config_path.parent
    kinematics = robot_config["robot_cfg"]["kinematics"]
    kinematics["urdf_path"] = str(config_dir / kinematics["urdf_path"])
    kinematics["collision_spheres"] = str(
        config_dir / kinematics["collision_spheres"]
    )

    warp_compat_alias = not hasattr(wp, "torch")
    if warp_compat_alias:
        # Isaac App registers this namespace; standalone Warp 1.16 exposes the
        # same conversion functions directly on the top-level module.
        wp.torch = wp

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        robot_cfg=robot_config,
        world_model={
            "cuboid": {
                "table": {
                    "dims": [0.5, 0.0, 0.0],
                    "pose": [-1000.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                }
            }
        },
        interpolation_dt=1.0 / 120.0,
        position_threshold=0.001,
        rotation_threshold=0.01,
        high_precision=True,
        collision_checker_type=CollisionCheckerType.MESH,
        collision_activation_distance=0.4,
    )
    motion_gen = MotionGen(motion_gen_config)
    tensor_args = motion_gen.tensor_args
    position = tensor_args.to_device(QPOS).reshape(1, -1)
    velocity = tensor_args.to_device(QVEL).reshape(1, -1)
    start = JointState(
        position=position,
        velocity=velocity,
        acceleration=torch.zeros_like(position),
        jerk=torch.zeros_like(position),
        joint_names=JOINT_NAMES,
    )
    plan_config = MotionGenPlanConfig(max_attempts=10, time_dilation_factor=1.0)
    plan_config.pose_cost_metric = PoseCostMetric(
        hold_partial_pose=True,
        hold_vec_weight=tensor_args.to_device(CONSTRAINT_POSE),
        offset_position=tensor_args.to_device([0.0, 0.0, PRE_DIS]),
    )
    result = motion_gen.plan_single(start, Pose.from_list(GOAL), plan_config)

    print(
        json.dumps(
            {
                "schema_version": "openeta.univtac.curobo_minimal_ab.v1",
                "source_planning_call_index": 4,
                "curobo_file": str(Path(curobo.__file__).resolve()),
                "curobo_version": curobo.__version__,
                "curobo_commit": args.curobo_commit,
                "robot_config": str(robot_config_path),
                "constraint_pose": CONSTRAINT_POSE,
                "pre_dis": PRE_DIS,
                "monkey_patch_loaded": warp_compat_alias,
                "monkey_patch_scope": (
                    "standalone Warp 1.16 wp.torch API alias only"
                    if warp_compat_alias
                    else None
                ),
                "success": _scalar(result.success),
                "status": repr(result.status),
                "valid_query": _scalar(result.valid_query),
                "attempts": _scalar(result.attempts),
                "solve_time": _scalar(result.solve_time),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
