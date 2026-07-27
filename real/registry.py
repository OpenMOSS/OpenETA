"""Registries mapping string keys to camera and robot driver classes.

New hardware is added by writing a driver in ``real/cameras`` or ``real/robots``
and registering it here (or at import time via :func:`register_camera` /
:func:`register_robot`). Config files and the env builder reference drivers by
these keys so no user code imports vendor SDKs directly.
"""

from __future__ import annotations

from real.cameras.base import Camera, CameraConfig
from real.cameras.realsense import RealSenseCamera
from real.cameras.webcam import WebcamCamera
from real.robots.base import RobotArm, RobotConfig
from real.robots.franka import FrankaArm
from real.robots.ur5e import UR5eArm

_CAMERA_REGISTRY: dict[str, type[Camera]] = {
    "realsense": RealSenseCamera,
    # L515 and D400 share the same driver; alias for clarity in configs.
    "realsense_l515": RealSenseCamera,
    "realsense_d435": RealSenseCamera,
    # RGB-only sources (USB/UVC webcams, RTSP/HTTP streams) via OpenCV.
    "webcam": WebcamCamera,
    "opencv": WebcamCamera,
    "rtsp": WebcamCamera,
}

_ROBOT_REGISTRY: dict[str, type[RobotArm]] = {
    "ur5e": UR5eArm,
    "franka": FrankaArm,
}


def register_camera(key: str, cls: type[Camera]) -> None:
    _CAMERA_REGISTRY[key] = cls


def register_robot(key: str, cls: type[RobotArm]) -> None:
    _ROBOT_REGISTRY[key] = cls


def available_cameras() -> list[str]:
    return sorted(_CAMERA_REGISTRY)


def available_robots() -> list[str]:
    return sorted(_ROBOT_REGISTRY)


def make_camera(driver: str, config: CameraConfig) -> Camera:
    try:
        cls = _CAMERA_REGISTRY[driver]
    except KeyError:
        raise KeyError(
            f"Unknown camera driver {driver!r}. Available: {available_cameras()}"
        ) from None
    return cls(config)


def make_robot(driver: str, config: RobotConfig) -> RobotArm:
    try:
        cls = _ROBOT_REGISTRY[driver]
    except KeyError:
        raise KeyError(
            f"Unknown robot driver {driver!r}. Available: {available_robots()}"
        ) from None
    return cls(config)
