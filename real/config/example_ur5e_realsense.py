"""Example hardware config: one UR5e + one wrist RealSense.

Copy and adapt for your cell. Build an env from this with::

    from real.config.example_ur5e_realsense import build_env
    env = build_env()
    obs = env.reset(task="pick up the red block")
"""

from __future__ import annotations

from real.cameras.base import CameraConfig
from real.env import RealRobotEnv
from real.registry import make_camera, make_robot
from real.robots.base import RobotConfig

# -- cameras --------------------------------------------------------------
WRIST_CAMERA = CameraConfig(
    name="wrist",
    width=1280,
    height=720,
    fps=30,
    serial=None,  # set to disambiguate multiple RealSense devices
    depth_enabled=True,
    align_depth_to_color=True,
)

# -- robot ----------------------------------------------------------------
UR5E = RobotConfig(
    name="ur5e",
    ip="127.0.0.1",  # UR controller address
    dof=6,
    home_joints=[0.0, -1.57, -1.57, -1.57, 1.57, 0.0],
    max_velocity=0.25,
    max_acceleration=0.5,
    has_gripper=True,
    extra={"gripper": "robotiq_2f85"},
)


def build_env(task: str = "") -> RealRobotEnv:
    camera = make_camera("realsense", WRIST_CAMERA)
    robot = make_robot("ur5e", UR5E)
    return RealRobotEnv(cameras=[camera], robot=robot, task=task)
