"""Camera hardware abstraction for real-robot deployment.

A ``Camera`` produces :class:`adapter.protocol.CameraFrame` objects so the rest
of OpenETA (point-cloud builders, grasp/placement tools, loggers) consumes real
sensor data through the exact same contract the simulator uses.

Concrete drivers (RealSense, etc.) live in sibling modules and import their
vendor SDK lazily so this package stays importable without hardware libraries.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from adapter.protocol import CameraFrame, JsonDict


@dataclass(slots=True)
class CameraConfig:
    """Static configuration for a single camera.

    Attributes
    ----------
    name:
        Stable frame id, e.g. ``"wrist"`` or ``"front"``. Becomes
        ``CameraFrame.frame_id``.
    width, height, fps:
        Requested stream format. Drivers pick the closest supported profile.
    depth_width, depth_height, depth_fps:
        Optional separate depth-stream format. Some sensors (e.g. L515) share no
        common resolution between color and depth, so depth must be requested at
        its own supported size and aligned into color. ``0``/``None`` means
        "reuse the color width/height/fps".
    serial:
        Optional device serial to disambiguate multiple identical cameras.
    depth_enabled:
        Whether to stream and align a depth frame.
    align_depth_to_color:
        Align depth into the color optical frame so RGB and depth share
        intrinsics (matches the point-cloud builders' expectations). Ignored
        by RGB-only cameras.
    device:
        Source selector for non-RealSense cameras: an OpenCV device index
        (``0``, ``1``, ...) or a URL/path (RTSP/HTTP stream, video file).
    intrinsics:
        Optional pre-calibrated intrinsics (``fx, fy, cx, cy`` and ``scale``).
        RGB-only cameras cannot self-report intrinsics, so supply them here from
        calibration when downstream geometry is needed.
    mount:
        Physical mounting: ``"fixed"`` for a stationary third-person camera
        (extrinsics are ``T_base_cam``) or ``"wrist"`` for a camera on the
        end-effector (extrinsics are ``T_gripper_cam``). Selects the hand-eye
        calibration mode and how downstream code interprets ``extrinsics``.
    extrinsics:
        Optional camera-to-robot-base transform, populated from calibration.
        Self-describing dict (see ``adapter.protocol.CameraFrame.to_mcp_dict``).
    """

    name: str
    width: int = 1280
    height: int = 720
    fps: int = 30
    depth_width: int = 0
    depth_height: int = 0
    depth_fps: int = 0
    serial: str | None = None
    depth_enabled: bool = True
    align_depth_to_color: bool = True
    device: int | str | None = None
    mount: str = "fixed"
    intrinsics: JsonDict = field(default_factory=dict)
    extrinsics: JsonDict = field(default_factory=dict)


class Camera(ABC):
    """Base class for a real RGB-D camera."""

    def __init__(self, config: CameraConfig) -> None:
        self.config = config
        self._started = False

    @property
    def name(self) -> str:
        return self.config.name

    @abstractmethod
    def start(self) -> None:
        """Open the device and begin streaming. Idempotent."""

    @abstractmethod
    def read(self) -> CameraFrame:
        """Grab the latest aligned RGB-D frame as a ``CameraFrame``.

        ``depth`` is linear metric depth in **metres**, and ``intrinsics``
        carries at least ``fx, fy, cx, cy, scale`` so the shared point-cloud
        builders (``tools.anygrasp_core.build_point_cloud_from_rgbd``) can
        backproject without conversion.
        """

    @abstractmethod
    def stop(self) -> None:
        """Stop streaming and release the device. Idempotent."""

    # -- context manager sugar --------------------------------------------
    def __enter__(self) -> "Camera":
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
