#!/usr/bin/env python3
"""Offline multi-camera eye-to-hand calibration CLI.

Given a prepared dataset directory with one subdirectory per fixed camera,
solve ``T_base_cam`` for every camera against a checkerboard mounted on the
robot end-effector, and write a combined JSON report.

Intrinsics resolution per camera (first match wins):
  1. --intrinsics <json>  entry keyed by camera name
  2. RealSense SDK, matched by the camera's serial in --config
  3. --fx/--fy/--cx/--cy   (applies to all cameras lacking the above)

Example:
    python -m real.calibration.run_offline_calibration \
        --dataset-dir /tmp/eye2hand_20260722 \
        --config real/config/ur5e_bench.json \
        --inner-corners 11x8 --square-size-m 0.02 \
        --arm-key arm_left --method PARK
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

from real.calibration.dataset import discover_cameras, load_camera_samples
from real.calibration.eye_to_hand import (
    CheckerboardSpec,
    calibrate_camera,
    intrinsics_matrix,
)


def parse_inner_corners(text: str) -> Tuple[int, int]:
    parts = text.lower().replace(" ", "").split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("inner-corners must look like 11x8")
    return int(parts[0]), int(parts[1])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, help="ur5e_bench.json, to map camera->serial and to patch extrinsics.")
    p.add_argument("--intrinsics", type=Path, help="JSON mapping camera_name -> {fx,fy,cx,cy,dist?}.")
    p.add_argument("--fx", type=float)
    p.add_argument("--fy", type=float)
    p.add_argument("--cx", type=float)
    p.add_argument("--cy", type=float)
    p.add_argument("--dist", type=str, default="0,0,0,0,0")
    p.add_argument("--inner-corners", type=parse_inner_corners, default="11x8",
                   help="Inner corners cols x rows (squares-1). Default 11x8 for a 12x9 board.")
    p.add_argument("--square-size-m", type=float, default=0.02)
    p.add_argument("--arm-key", type=str, default="arm_left")
    p.add_argument("--method", type=str, default="PARK",
                   choices=["TSAI", "PARK", "HORAUD", "ANDREFF", "DANIILIDIS"])
    p.add_argument("--max-reprojection-px", type=float, default=3.0)
    p.add_argument("--max-consistency-translation-mm", type=float, default=40.0)
    p.add_argument("--max-consistency-rotation-deg", type=float, default=6.0)
    p.add_argument("--cameras", type=str, help="Comma-separated subset of camera names to calibrate.")
    p.add_argument("--eye-in-hand", type=str, default="",
                   help="Comma-separated camera names mounted on the wrist (eye-in-hand). "
                        "These solve T_gripper_cam; all others solve T_base_cam. "
                        "For eye-in-hand the checkerboard must be FIXED in the scene.")
    p.add_argument("--output", type=Path, help="Report path. Default: <dataset-dir>/eye_to_hand_report.json")
    p.add_argument("--write-config", action="store_true",
                   help="Patch T_base_cam into --config camera extrinsics (writes a .bak first).")
    return p.parse_args()


def parse_number_list(text: str) -> list:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def load_config_serials(config_path: Optional[Path]) -> Dict[str, Optional[str]]:
    """Map camera name -> serial (or None) from ur5e_bench.json."""
    if not config_path or not config_path.exists():
        return {}
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    result = {}
    for cam in cfg.get("cameras", []):
        result[cam.get("name")] = cam.get("serial")
    return result


def realsense_intrinsics_by_serial() -> Dict[str, np.ndarray]:
    """Query connected RealSense devices; return serial -> K. Empty on failure."""
    try:
        import pyrealsense2 as rs
    except Exception:
        return {}
    out: Dict[str, np.ndarray] = {}
    try:
        ctx = rs.context()
        for dev in ctx.query_devices():
            serial = dev.get_info(rs.camera_info.serial_number)
            pipeline = rs.pipeline()
            cfg = rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(rs.stream.color)
            try:
                profile = pipeline.start(cfg)
                stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
                intr = stream.get_intrinsics()
                out[serial] = intrinsics_matrix(intr.fx, intr.fy, intr.ppx, intr.ppy)
            finally:
                pipeline.stop()
    except Exception as exc:
        print(f"[warn] RealSense intrinsics query failed: {exc}", file=sys.stderr)
    return out


def resolve_intrinsics(
    camera_name: str,
    args: argparse.Namespace,
    intr_map: Dict[str, dict],
    serial_by_cam: Dict[str, Optional[str]],
    rs_by_serial: Dict[str, np.ndarray],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Resolve (K, D) for a camera, or None if nothing matched."""
    dist_default = np.asarray(parse_number_list(args.dist), dtype=np.float64).reshape(-1, 1)

    if camera_name in intr_map:
        m = intr_map[camera_name]
        K = intrinsics_matrix(m["fx"], m["fy"], m["cx"], m["cy"])
        D = np.asarray(m.get("dist", [0, 0, 0, 0, 0]), dtype=np.float64).reshape(-1, 1)
        return K, D

    serial = serial_by_cam.get(camera_name)
    if serial and serial in rs_by_serial:
        return rs_by_serial[serial], dist_default

    if None not in (args.fx, args.fy, args.cx, args.cy):
        return intrinsics_matrix(args.fx, args.fy, args.cx, args.cy), dist_default

    return None


def patch_config(config_path: Path, results: Dict[str, dict]) -> None:
    """Write solved extrinsics into each camera's extrinsics field.

    ``results`` maps camera name -> {"key": "T_base_cam"|"T_gripper_cam",
    "matrix": 4x4 list, "frame": reference frame}.
    """
    backup = config_path.with_suffix(config_path.suffix + ".bak")
    backup.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    for cam in cfg.get("cameras", []):
        name = cam.get("name")
        if name in results:
            entry = results[name]
            cam["extrinsics"] = {
                "type": entry["key"],
                "frame": entry["frame"],
                entry["key"]: entry["matrix"],
            }
    config_path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    print(f"Patched extrinsics into {config_path} (backup: {backup})")


def main() -> int:
    args = parse_args()
    dataset_dir: Path = args.dataset_dir
    if not dataset_dir.is_dir():
        print(f"dataset-dir not found: {dataset_dir}", file=sys.stderr)
        return 2

    board = CheckerboardSpec(
        inner_corners_x=args.inner_corners[0],
        inner_corners_y=args.inner_corners[1],
        square_size_m=args.square_size_m,
    )

    all_cameras = discover_cameras(dataset_dir)
    if args.cameras:
        wanted = {c.strip() for c in args.cameras.split(",") if c.strip()}
        all_cameras = [c for c in all_cameras if c in wanted]
    if not all_cameras:
        print("No camera subdirectories with frames found.", file=sys.stderr)
        return 2

    intr_map = json.loads(args.intrinsics.read_text(encoding="utf-8")) if args.intrinsics else {}
    serial_by_cam = load_config_serials(args.config)
    rs_by_serial = realsense_intrinsics_by_serial() if serial_by_cam else {}

    eye_in_hand_cams = {c.strip() for c in args.eye_in_hand.split(",") if c.strip()}

    from real.calibration.eye_to_hand import EYE_IN_HAND, EYE_TO_HAND

    combined = {"dataset_dir": str(dataset_dir), "cameras": {}}
    solved: Dict[str, dict] = {}

    for camera_name in all_cameras:
        intr = resolve_intrinsics(camera_name, args, intr_map, serial_by_cam, rs_by_serial)
        if intr is None:
            print(f"[{camera_name}] no intrinsics resolved; skipping.", file=sys.stderr)
            combined["cameras"][camera_name] = {"error": "no_intrinsics"}
            continue
        K, D = intr

        mode = EYE_IN_HAND if camera_name in eye_in_hand_cams else EYE_TO_HAND
        samples = load_camera_samples(dataset_dir / camera_name, arm_key=args.arm_key)
        try:
            result = calibrate_camera(
                camera_name, samples, K, D, board,
                method=args.method,
                mode=mode,
                max_reprojection_px=args.max_reprojection_px,
                max_consistency_translation_mm=args.max_consistency_translation_mm,
                max_consistency_rotation_deg=args.max_consistency_rotation_deg,
            )
        except ValueError as exc:
            print(f"[{camera_name}] calibration failed: {exc}", file=sys.stderr)
            combined["cameras"][camera_name] = {"error": str(exc), "mode": mode}
            continue

        combined["cameras"][camera_name] = result.report
        cam_key = "T_base_cam" if mode == EYE_TO_HAND else "T_gripper_cam"
        frame = "base_link" if mode == EYE_TO_HAND else "gripper"
        solved[camera_name] = {
            "key": cam_key,
            "frame": frame,
            "matrix": np.asarray(result.T_cam, dtype=float).round(12).tolist(),
        }

        q = result.report["quality_metrics"]
        print(f"\n=== {camera_name} [{mode}] ===")
        print(f"  final valid samples: {result.report['counts']['final_valid_samples']}"
              f" / {result.report['counts']['total_samples']}")
        print(f"  pose consistency translation [mm]: {q['pose_consistency_translation_error_mm']}")
        print(f"  pose consistency rotation   [deg]: {q['pose_consistency_rotation_error_deg']}")
        print(f"  {cam_key}:")
        print(np.array2string(result.T_cam, formatter={"float_kind": lambda x: f"{x: .6f}"}))

    output_path = args.output or (dataset_dir / "eye_to_hand_report.json")
    output_path.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote combined report: {output_path}")

    if args.write_config:
        if not args.config:
            print("--write-config requires --config", file=sys.stderr)
            return 2
        if not solved:
            print("Nothing solved; not patching config.", file=sys.stderr)
            return 3
        patch_config(args.config, solved)

    return 0 if solved else 4


if __name__ == "__main__":
    sys.exit(main())
