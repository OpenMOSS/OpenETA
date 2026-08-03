#!/usr/bin/env python3
"""Export one post-action observation as a local execution Viser sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from scripts.embodied.export_libero_graspgenx_sample import (
    _agentview,
    _camera_pose_opencv,
    _observation,
    _write_json,
)


def run(args: argparse.Namespace) -> int:
    episode_root = args.episode_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    sample_dir = output_root / "00"
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"Refusing non-empty output root: {output_root}")
    sample_dir.mkdir(parents=True, exist_ok=True)

    comparison = json.loads(args.comparison.read_text(encoding="utf-8"))
    observation_id = str(comparison["observation_id"])
    observation = _observation(episode_root, observation_id)
    frame = _agentview(episode_root, observation)
    metadata = frame.get("metadata", {})
    intrinsics = metadata.get("intrinsics", {})
    extrinsics = metadata.get("extrinsics", {})
    rgb_path = Path(frame["rgb_path"])
    depth_path = Path(frame["depth_path"])
    rgb = Image.open(rgb_path).convert("RGB")
    depth_mm = np.asarray(Image.open(depth_path), dtype=np.uint16)
    depth_m = depth_mm.astype(np.float32) / float(intrinsics.get("scale", 1000.0))
    fx, fy = float(intrinsics["fx"]), float(intrinsics["fy"])
    cx, cy = float(intrinsics["cx"]), float(intrinsics["cy"])
    camera_pose = _camera_pose_opencv(dict(extrinsics))

    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    rows, columns = np.where(valid)
    z = depth_m[rows, columns]
    camera_points = np.stack(
        [(columns - cx) * z / fx, (rows - cy) * z / fy, z], axis=1
    )
    world_points = (
        camera_points @ camera_pose[:3, :3].T + camera_pose[:3, 3]
    )
    target = np.asarray(comparison["target_world_from_grip_site"], dtype=np.float64)
    actual = np.asarray(comparison["actual_world_from_grip_site"], dtype=np.float64)
    center = (target[:3, 3] + actual[:3, 3]) / 2.0
    distances = np.linalg.norm(world_points - center, axis=1)
    selected = distances <= float(args.focus_radius_m)
    if int(selected.sum()) < args.min_points:
        nearest = np.argsort(distances)[: min(args.min_points, len(distances))]
        selected = np.zeros(len(distances), dtype=bool)
        selected[nearest] = True
    segmentation = np.zeros(depth_m.shape, dtype=np.uint8)
    segmentation[rows[selected], columns[selected]] = 101

    lower = world_points.min(axis=0) - 0.05
    upper = world_points.max(axis=0) + 0.05
    rgb.save(sample_dir / "rgb.png")
    np.save(sample_dir / "depth.npy", depth_m)
    Image.fromarray(segmentation, mode="L").save(sample_dir / "seg.png")
    _write_json(
        sample_dir / "meta_data.json",
        {
            "intrinsics": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
            "camera_pose": camera_pose.tolist(),
            "label_map": {"ground": 0, "execution_focus": 101},
            "scene_bounds": [*lower.tolist(), *upper.tolist()],
            "scene_mode": "execution",
            "observation_id": observation_id,
            "frame_id": frame.get("frame_id"),
            "action_id": comparison.get("action_id"),
            "camera_frame": "opencv",
            "focus_center_world": center.tolist(),
        },
    )
    _write_json(
        output_root / "export.json",
        {
            "episode_root": str(episode_root),
            "observation_id": observation_id,
            "frame_id": frame.get("frame_id"),
            "comparison": str(args.comparison),
            "rgb_source": str(rgb_path),
            "depth_source": str(depth_path),
            "camera_pose_opencv_to_world": camera_pose.tolist(),
            "focus_center_world": center.tolist(),
            "focus_radius_m": args.focus_radius_m,
            "selected_point_count": int(selected.sum()),
            "sample_dir": str(sample_dir),
        },
    )
    print((output_root / "export.json").read_text(encoding="utf-8"))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--focus-radius-m", type=float, default=0.16)
    parser.add_argument("--min-points", type=int, default=300)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
