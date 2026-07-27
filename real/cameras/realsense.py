"""Intel RealSense RGB-D camera driver (D400 series + L515).

Depends on ``pyrealsense2`` from the optional ``real`` extra. The import is
deferred to :meth:`RealSenseCamera.start` so the module stays importable in the
base agent venv and under unit tests without the SDK installed.
"""

from __future__ import annotations

import time

from adapter.protocol import CameraFrame
from real.cameras.base import Camera, CameraConfig


class RealSenseCamera(Camera):
    """RealSense RGB-D camera exposing aligned metric-depth frames.

    Supports the D400 stereo family and the L515 solid-state LiDAR. The color
    stream defines the shared optical frame; depth is aligned to it when
    ``config.align_depth_to_color`` is set.
    """

    def __init__(self, config: CameraConfig) -> None:
        super().__init__(config)
        self._pipeline = None
        self._align = None
        self._depth_scale = 1.0
        self._intrinsics: dict[str, float] = {}

    def start(self) -> None:
        if self._started:
            return
        try:
            import pyrealsense2 as rs  # noqa: F401
        except ImportError as exc:  # pragma: no cover - hardware path
            raise RuntimeError(
                "pyrealsense2 is required for RealSenseCamera. Install the "
                "'real' extra: uv sync --extra real"
            ) from exc

        cfg = self.config
        pipeline = rs.pipeline()
        rs_config = rs.config()
        if cfg.serial:
            rs_config.enable_device(cfg.serial)
        rs_config.enable_stream(
            rs.stream.color, cfg.width, cfg.height, rs.format.bgr8, cfg.fps
        )
        if cfg.depth_enabled:
            # Some sensors (L515) share no common color/depth resolution, so the
            # depth stream can be requested at its own size and aligned to color.
            d_w = cfg.depth_width or cfg.width
            d_h = cfg.depth_height or cfg.height
            d_fps = cfg.depth_fps or cfg.fps
            rs_config.enable_stream(
                rs.stream.depth, d_w, d_h, rs.format.z16, d_fps
            )

        profile = pipeline.start(rs_config)
        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale = float(depth_sensor.get_depth_scale())

        if cfg.depth_enabled and cfg.align_depth_to_color:
            self._align = rs.align(rs.stream.color)

        # Intrinsics of the frame the point cloud is built in (color when aligned)
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        self._intrinsics = {
            "fx": float(intr.fx),
            "fy": float(intr.fy),
            "cx": float(intr.ppx),
            "cy": float(intr.ppy),
            "width": int(intr.width),
            "height": int(intr.height),
            # depth PNG is encoded as uint16 millimetres by _encode_pixels_to_base64
            # (mode="depth"); MCP contract is depth_m = raw_depth / scale.
            "scale": 1000.0,
        }
        self._pipeline = pipeline
        self._started = True

    def read(self) -> CameraFrame:
        if not self._started or self._pipeline is None:
            raise RuntimeError("RealSenseCamera.read() called before start().")
        import numpy as np

        frames = self._pipeline.wait_for_frames()
        if self._align is not None:
            frames = self._align.process(frames)

        color = frames.get_color_frame()
        rgb = np.asanyarray(color.get_data())[:, :, ::-1]  # BGR -> RGB

        depth_m = None
        if self.config.depth_enabled:
            depth = frames.get_depth_frame()
            if depth:
                raw = np.asanyarray(depth.get_data())
                depth_m = (raw.astype("float32") * self._depth_scale)

        return CameraFrame(
            frame_id=self.name,
            rgb=rgb.tolist(),
            depth=depth_m.tolist() if depth_m is not None else None,
            intrinsics=dict(self._intrinsics),
            extrinsics=dict(self.config.extrinsics),
            timestamp_s=time.time(),
        )

    def stop(self) -> None:
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            finally:
                self._pipeline = None
        self._started = False
