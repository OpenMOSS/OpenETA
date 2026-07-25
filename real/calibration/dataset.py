"""Load eye-to-hand calibration samples from prepared multi-camera datasets.

Consumes the output of the teleop dataset-prep tool (see
the companion teleoperation checkout's dataset preparation tool):

    <dataset_dir>/
        <camera_name>/
            <sample_id>_frame.png
            <sample_id>_proprio.json
        _manifest.json   (optional)

Each ``*_proprio.json`` follows the teleop schema:
    {"sample": {"ee": {"<arm_key>": {"position": {x,y,z},
                                      "orientation": {x,y,z,w},
                                      "frame_id": "base_link"}}}}

Grouping by camera-name subdirectory lets one dataset calibrate every fixed
camera (L515, D435i, ...) in a single run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from real.calibration.eye_to_hand import Sample
from real.calibration.geometry import (
    quaternion_xyzw_to_rotation_matrix,
    transform_from_rt,
)


def _pose_to_transform(ee: Dict) -> np.ndarray:
    pos = ee["position"]
    quat = ee["orientation"]
    R = quaternion_xyzw_to_rotation_matrix(quat["x"], quat["y"], quat["z"], quat["w"])
    t = np.array([pos["x"], pos["y"], pos["z"]], dtype=np.float64)
    return transform_from_rt(R, t)


def load_camera_samples(
    camera_dir: Path,
    arm_key: str = "arm_left",
    load_images: bool = True,
) -> List[Sample]:
    """Load all (image, proprio) sample pairs for one camera subdirectory."""
    frame_paths = sorted(camera_dir.glob("*_frame.png"))
    samples: List[Sample] = []

    for frame_path in frame_paths:
        sample_id = frame_path.name[: -len("_frame.png")]
        proprio_path = camera_dir / f"{sample_id}_proprio.json"

        if not proprio_path.exists():
            samples.append(
                Sample(
                    sample_id=sample_id,
                    T_base_ee=np.eye(4),
                    reject_reason="missing_proprio_json",
                )
            )
            continue

        payload = json.loads(proprio_path.read_text(encoding="utf-8"))
        ee = payload.get("sample", {}).get("ee", {}).get(arm_key)
        if not ee:
            samples.append(
                Sample(
                    sample_id=sample_id,
                    T_base_ee=np.eye(4),
                    reject_reason=f"missing_ee_{arm_key}",
                )
            )
            continue

        image = None
        if load_images:
            image = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)

        samples.append(
            Sample(
                sample_id=sample_id,
                T_base_ee=_pose_to_transform(ee),
                frame_id=ee.get("frame_id", ""),
                image=image,
            )
        )

    return samples


def discover_cameras(dataset_dir: Path) -> List[str]:
    """Return camera-name subdirectories that contain at least one frame."""
    cameras = []
    for child in sorted(dataset_dir.iterdir()):
        if child.is_dir() and any(child.glob("*_frame.png")):
            cameras.append(child.name)
    return cameras
