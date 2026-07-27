"""OpenETA real-robot deployment package.

Hardware abstraction for driving real cameras and robot arms through the same
``adapter.protocol`` / ``adapter.sim`` contract the simulator uses.

Layout::

    real/cameras/   RGB-D camera drivers (base + RealSense)
    real/robots/    robot-arm drivers (base + UR5e, Franka stub)
    real/registry.py  string-keyed driver registries
    real/env.py       RealRobotEnv (SimulatorAdapter implementation)
    real/config/      example hardware configs
    real/examples/    runnable smoke scripts

Vendor SDKs (pyrealsense2, ur_rtde, ...) are optional and imported lazily; see
the ``real`` optional-dependency extra in ``pyproject.toml``.
"""

from real.env import RealRobotEnv
from real.registry import (
    available_cameras,
    available_robots,
    make_camera,
    make_robot,
    register_camera,
    register_robot,
)

__all__ = [
    "RealRobotEnv",
    "available_cameras",
    "available_robots",
    "make_camera",
    "make_robot",
    "register_camera",
    "register_robot",
]
