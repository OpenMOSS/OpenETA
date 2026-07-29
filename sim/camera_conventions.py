"""Deterministic camera-frame conversions used at simulator boundaries.

Simulator renderers commonly expose a camera pose in an OpenGL/USD frame
(+X right, +Y up, camera looking along -Z), while RGB-D perception tools use
the OpenCV optical frame (+X right, +Y down, +Z forward).  The simulator
adapter owns that distinction and publishes an OpenCV camera-to-world
transform so downstream tools never need to infer conventions from a backend
name.
"""

from __future__ import annotations

from typing import Any

import numpy as np

_OPENCV_TO_RENDERER = np.diag([1.0, -1.0, -1.0])
_RENDERER_FRAME_ALIASES = {
    "opengl",
    "opengl_renderer",
    "omnigibson_usd",
    "renderer",
    "usd",
}
_OPENCV_FRAME_ALIASES = {"opencv", "opencv_optical"}


def quaternion_xyzw_to_rotation_matrix(quaternion_xyzw: Any) -> np.ndarray:
    """Return a finite 3x3 active rotation from an ``[x, y, z, w]`` quaternion."""

    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(-1)
    if quaternion.size != 4 or not np.isfinite(quaternion).all():
        raise ValueError("camera quaternion must contain four finite xyzw values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("camera quaternion must have non-zero norm")
    x, y, z, w = quaternion / norm
    return np.asarray(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=np.float64,
    )


def normalise_camera_to_world_opencv(
    *,
    position_xyz: Any,
    rotation_camera_to_world: Any,
    source_camera_frame: str,
    normalized_from: str,
) -> dict[str, Any]:
    """Build the canonical OpenCV optical-frame camera-to-world packet.

    ``rotation_camera_to_world`` maps vectors expressed in
    ``source_camera_frame`` into world coordinates.  Renderer/USD frames are
    converted with ``diag(1, -1, -1)`` on the right:

    ``R_world_opencv = R_world_renderer @ R_renderer_from_opencv``.
    """

    position = np.asarray(position_xyz, dtype=np.float64).reshape(-1)
    rotation = np.asarray(rotation_camera_to_world, dtype=np.float64)
    if position.size != 3 or not np.isfinite(position).all():
        raise ValueError("camera position must contain three finite values")
    if rotation.size != 9:
        raise ValueError("camera rotation must contain nine values")
    rotation = rotation.reshape(3, 3)
    if not np.isfinite(rotation).all():
        raise ValueError("camera rotation must contain only finite values")

    source = str(source_camera_frame or "").strip().lower().replace("-", "_")
    if source in _RENDERER_FRAME_ALIASES:
        rotation_opencv = rotation @ _OPENCV_TO_RENDERER
    elif source in _OPENCV_FRAME_ALIASES:
        rotation_opencv = rotation
    else:
        raise ValueError(f"unsupported source camera frame: {source_camera_frame!r}")

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_opencv
    transform[:3, 3] = position[:3]
    return {
        "pos": position[:3].tolist(),
        "mat": rotation_opencv.reshape(-1).tolist(),
        "camera_to_world": transform.tolist(),
        "matrix_layout": "row_major",
        "frame_transform": "camera_to_world",
        "camera_frame": "opencv",
        "image_origin": "top_left",
        "raw_camera_convention": source,
        "normalized_from": str(normalized_from),
    }
