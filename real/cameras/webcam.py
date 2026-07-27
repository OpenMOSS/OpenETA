"""Generic RGB webcam / IP-camera driver (via OpenCV).

Covers USB/UVC webcams, laptop built-in cameras, and network streams
(RTSP/HTTP) — any source ``cv2.VideoCapture`` can open. These are **RGB-only**:
``CameraFrame.depth`` is ``None`` and intrinsics come from ``config.intrinsics``
(calibration) rather than the device.

Depends on ``opencv-python`` from the optional ``real`` extra; the import is
deferred to :meth:`WebcamCamera.start` so the module stays importable without it.
"""

from __future__ import annotations

import time

from adapter.protocol import CameraFrame
from real.cameras.base import Camera, CameraConfig


class WebcamCamera(Camera):
    """RGB-only camera backed by ``cv2.VideoCapture``.

    ``config.device`` selects the source: an integer index (``0`` for the
    default camera) or a URL/path for an RTSP/HTTP stream or video file. Depth
    is never produced; ``config.depth_enabled`` is forced off.
    """

    def __init__(self, config: CameraConfig) -> None:
        config.depth_enabled = False
        super().__init__(config)
        self._cap = None

    def start(self) -> None:
        if self._started:
            return
        try:
            import cv2
        except ImportError as exc:  # pragma: no cover - hardware path
            raise RuntimeError(
                "opencv-python is required for WebcamCamera. Install the "
                "'real' extra: uv sync --extra real"
            ) from exc

        cfg = self.config
        source = cfg.device if cfg.device is not None else 0
        # Prefer the V4L2 backend for /dev/* devices; OpenCV otherwise tries
        # GStreamer first and logs a spurious uridecodebin error.
        if isinstance(source, str) and source.startswith("/dev/"):
            cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
        else:
            cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"WebcamCamera could not open source {source!r}.")

        # Many UVC webcams only expose MJPG at speed; request it explicitly.
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
        cap.set(cv2.CAP_PROP_FPS, cfg.fps)

        self._cap = cap
        self._started = True

    def read(self) -> CameraFrame:
        if not self._started or self._cap is None:
            raise RuntimeError("WebcamCamera.read() called before start().")
        ok, frame_bgr = self._cap.read()
        if not ok or frame_bgr is None:
            raise RuntimeError(f"WebcamCamera failed to grab a frame from {self.name!r}.")
        # BGR -> RGB, then to nested lists: CameraFrame.rgb is typed
        # list[list[list[int]]] and to_mcp_dict() truthiness-checks it, so a
        # raw ndarray would raise "truth value is ambiguous".
        rgb = frame_bgr[:, :, ::-1].tolist()
        return CameraFrame(
            frame_id=self.name,
            rgb=rgb,
            depth=None,
            intrinsics=dict(self.config.intrinsics),
            extrinsics=dict(self.config.extrinsics),
            timestamp_s=time.time(),
        )

    def stop(self) -> None:
        if self._cap is not None:
            try:
                self._cap.release()
            finally:
                self._cap = None
        self._started = False
