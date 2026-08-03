"""Export one retained LIBERO RGB-D mask as a GraspGen-X real-world sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _observation(root: Path, observation_id: str) -> dict[str, Any]:
    events = root / "events.jsonl"
    for line in events.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        payload = event.get("payload", {})
        if event.get("kind") == "observation" and payload.get("observation_id") == observation_id:
            return payload
    raise RuntimeError(f"Observation {observation_id!r} not found in {events}")


def _agentview(root: Path, observation: dict[str, Any]) -> dict[str, Any]:
    for frame in observation.get("frames", []):
        if isinstance(frame, dict) and frame.get("camera_id") == "agentview":
            value = dict(frame)
            for key in ("rgb_path", "depth_path"):
                path = Path(str(value[key]))
                value[key] = str(path if path.is_absolute() else root / path)
            return value
    raise RuntimeError("Observation has no agentview frame")


def _camera_pose_opencv(extrinsics: dict[str, Any]) -> np.ndarray:
    if extrinsics.get("frame_transform") != "camera_to_world":
        raise RuntimeError("Expected camera_to_world extrinsics")
    if extrinsics.get("camera_frame") != "opengl":
        raise RuntimeError("Expected retained LIBERO OpenGL camera frame")
    flat = np.asarray(extrinsics.get("mat"), dtype=np.float64)
    if flat.shape != (9,):
        raise RuntimeError("Expected row-major 3x3 camera rotation")
    world_from_gl = flat.reshape(3, 3)
    opencv_to_opengl = np.diag([1.0, -1.0, -1.0])
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = world_from_gl @ opencv_to_opengl
    pose[:3, 3] = np.asarray(extrinsics.get("pos"), dtype=np.float64)
    return pose


def run(args: argparse.Namespace) -> int:
    episode_root = args.episode_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    sample_dir = output_root / "00"
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"Refusing non-empty output root: {output_root}")
    sample_dir.mkdir(parents=True, exist_ok=True)

    observation = _observation(episode_root, args.observation_id)
    frame = _agentview(episode_root, observation)
    metadata = frame.get("metadata", {})
    intrinsics = metadata.get("intrinsics", {})
    extrinsics = metadata.get("extrinsics", {})

    rgb_path = Path(frame["rgb_path"])
    depth_path = Path(frame["depth_path"])
    mask_path = args.mask.expanduser().resolve()
    rgb = Image.open(rgb_path).convert("RGB")
    depth_mm = np.asarray(Image.open(depth_path), dtype=np.uint16)
    mask = np.asarray(Image.open(mask_path).convert("L"), dtype=np.uint8) > 0
    if depth_mm.shape != mask.shape or rgb.size != (mask.shape[1], mask.shape[0]):
        raise RuntimeError("RGB, depth, and target mask dimensions differ")
    depth_m = depth_mm.astype(np.float32) / 1000.0
    valid_target = mask & np.isfinite(depth_m) & (depth_m > 0.0)
    if int(valid_target.sum()) < args.min_object_points:
        raise RuntimeError(
            f"Target mask has only {int(valid_target.sum())} valid depth points; "
            f"need {args.min_object_points}"
        )

    fx = float(intrinsics["fx"])
    fy = float(intrinsics["fy"])
    cx = float(intrinsics["cx"])
    cy = float(intrinsics["cy"])
    camera_pose = _camera_pose_opencv(dict(extrinsics))
    v, u = np.where(depth_m > 0.0)
    z = depth_m[v, u]
    camera_points = np.stack(
        [(u - cx) * z / fx, (v - cy) * z / fy, z], axis=1
    )
    world_points = (
        camera_points @ camera_pose[:3, :3].T + camera_pose[:3, 3]
    )
    lower = world_points.min(axis=0) - 0.05
    upper = world_points.max(axis=0) + 0.05

    rgb.save(sample_dir / "rgb.png")
    np.save(sample_dir / "depth.npy", depth_m)
    segmentation = np.zeros(mask.shape, dtype=np.uint8)
    segmentation[valid_target] = 101
    Image.fromarray(segmentation, mode="L").save(sample_dir / "seg.png")
    meta = {
        "intrinsics": [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        "camera_pose": camera_pose.tolist(),
        "label_map": {"ground": 0, "obj_1": 101},
        "scene_bounds": [*lower.tolist(), *upper.tolist()],
        "scene_mode": "proposal",
        "observation_id": args.observation_id,
        "frame_id": frame.get("frame_id"),
        "camera_frame": "opencv",
    }
    _write_json(sample_dir / "meta_data.json", meta)
    _write_json(
        output_root / "export.json",
        {
            "episode_root": str(episode_root),
            "observation_id": args.observation_id,
            "frame_id": frame.get("frame_id"),
            "rgb_source": str(rgb_path),
            "depth_source": str(depth_path),
            "mask_source": str(mask_path),
            "valid_object_points": int(valid_target.sum()),
            "depth_min_m": float(depth_m[valid_target].min()),
            "depth_max_m": float(depth_m[valid_target].max()),
            "camera_input_frame": "opengl",
            "camera_output_frame": "opencv",
            "camera_pose_opencv_to_world": camera_pose.tolist(),
            "sample_dir": str(sample_dir),
        },
    )
    print(json.dumps(json.loads((output_root / "export.json").read_text()), indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--observation-id", required=True)
    parser.add_argument("--mask", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--min-object-points", type=int, default=100)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
