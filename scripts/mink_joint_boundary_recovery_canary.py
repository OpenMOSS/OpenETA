#!/usr/bin/env python3
"""Replay the r18 wrist target, then verify seeded preserve-current retreat."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scipy.spatial.transform import Rotation

from agent.tools.sim_mcp import SseSimulatorMcpTransport


R18_WRIST_XYZ = [-0.042991370826, -0.346145939798, 0.182968491111]
R18_WRIST_ROTATION = [
    [0.007337702961, 0.999973078695, 0.0],
    [0.999973078695, -0.007337702961, 0.0],
    [0.0, 0.0, -1.0],
]


def _eef_xyz(payload: dict[str, Any]) -> list[float]:
    observation = payload.get("observation", payload)
    robot = observation.get("robot") if isinstance(observation, dict) else None
    pose = robot.get("end_effector_pose") if isinstance(robot, dict) else None
    xyz = pose.get("xyz") if isinstance(pose, dict) else None
    return [float(value) for value in xyz[:3]] if isinstance(xyz, list) else []


def _preview_seed(preview: dict[str, Any], receipt_id: str) -> dict[str, Any]:
    candidate = preview.get("best_candidate")
    joints = candidate.get("joint_positions") if isinstance(candidate, dict) else None
    if preview.get("status") != "reachable" or not isinstance(joints, list):
        raise RuntimeError(f"IK preview did not return a reachable joint seed: {preview}")
    return {
        "schema_version": "openeta.ik_execution_seed.v1",
        "receipt_id": receipt_id,
        "joint_positions": joints,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    transport = SseSimulatorMcpTransport(args.url)
    created: dict[str, Any] = {}
    cleanup: dict[str, Any] = {}
    report: dict[str, Any] = {}
    try:
        created = transport.call_tool(
            "create_env",
            {
                "env_id": args.env_id,
                "seed": args.seed,
                "include_objects": True,
                "image_width": args.image_size,
                "image_height": args.image_size,
            },
            timeout_s=args.timeout_s,
        )
        common = {
            "handle": str(created["handle"]),
            "session_id": str(created["session_id"]),
        }
        reset = transport.call_tool(
            "reset_env", {**common, "seed": args.seed}, timeout_s=args.timeout_s
        )
        euler = Rotation.from_matrix(R18_WRIST_ROTATION).as_euler(
            "xyz", degrees=True
        ).tolist()
        wrist_preview = transport.call_tool(
            "ik_preview_check",
            {
                **common,
                "x": R18_WRIST_XYZ[0],
                "y": R18_WRIST_XYZ[1],
                "z": R18_WRIST_XYZ[2],
                "roll": euler[0],
                "pitch": euler[1],
                "yaw": euler[2],
                "preserve_current_orientation": False,
                "position_tolerance_m": args.tolerance_m,
                "orientation_tolerance_rad": args.orientation_tolerance_rad,
            },
            timeout_s=args.timeout_s,
        )
        wrist_move = transport.call_tool(
            "move_to",
            {
                **common,
                "x": R18_WRIST_XYZ[0],
                "y": R18_WRIST_XYZ[1],
                "z": R18_WRIST_XYZ[2],
                "roll": euler[0],
                "pitch": euler[1],
                "yaw": euler[2],
                "tolerance": args.tolerance_m,
                "ori_tolerance": args.orientation_tolerance_rad,
                "num_steps": args.max_steps,
                "enable_collision_check": True,
                "ik_execution_seed": _preview_seed(
                    wrist_preview, "r18-wrist-viewpoint"
                ),
            },
            timeout_s=args.timeout_s,
        )
        reached_xyz = _eef_xyz(wrist_move)
        retreat_xyz = [reached_xyz[0], reached_xyz[1], reached_xyz[2] + args.retreat_z]
        retreat_preview = transport.call_tool(
            "ik_preview_check",
            {
                **common,
                "x": retreat_xyz[0],
                "y": retreat_xyz[1],
                "z": retreat_xyz[2],
                "preserve_current_orientation": True,
                "position_tolerance_m": args.tolerance_m,
                "orientation_tolerance_rad": args.orientation_tolerance_rad,
            },
            timeout_s=args.timeout_s,
        )
        retreat_move = transport.call_tool(
            "move_to",
            {
                **common,
                "x": retreat_xyz[0],
                "y": retreat_xyz[1],
                "z": retreat_xyz[2],
                "tolerance": args.tolerance_m,
                "ori_tolerance": args.orientation_tolerance_rad,
                "num_steps": args.max_steps,
                "enable_collision_check": True,
                "ik_execution_seed": _preview_seed(
                    retreat_preview, "r18-preserve-current-retreat"
                ),
            },
            timeout_s=args.timeout_s,
        )
        wrist_receipt = wrist_move.get("controller_receipt") or {}
        retreat_receipt = retreat_move.get("controller_receipt") or {}
        checks = {
            "wrist_preview_reachable": wrist_preview.get("status") == "reachable",
            "wrist_target_reached": wrist_move.get("reached_target") is True,
            "wrist_seed_bound": (
                wrist_receipt.get("ik_execution_seed_receipt_id")
                == "r18-wrist-viewpoint"
            ),
            "retreat_preview_reachable": retreat_preview.get("status") == "reachable",
            "retreat_target_reached": retreat_move.get("reached_target") is True,
            "preserve_current_seed_bound": (
                retreat_receipt.get("ik_execution_seed_receipt_id")
                == "r18-preserve-current-retreat"
            ),
            "no_constraint_escape_deadlock": (
                (retreat_move.get("controller_failure") or {}).get("code")
                != "constraint_escape_preview_rejected"
            ),
        }
        report = {
            "schema_version": "openeta.mink_joint_boundary_recovery_canary.v1",
            "url": args.url,
            "env_id": args.env_id,
            "seed": args.seed,
            "reset_eef_xyz": _eef_xyz(reset),
            "wrist_target_xyz": R18_WRIST_XYZ,
            "wrist_preview": wrist_preview,
            "wrist_move": wrist_move,
            "retreat_target_xyz": retreat_xyz,
            "retreat_preview": retreat_preview,
            "retreat_move": retreat_move,
            "checks": checks,
            "passed": all(checks.values()),
        }
    finally:
        if created.get("handle") and created.get("session_id"):
            try:
                cleanup = transport.call_tool(
                    "close_env",
                    {
                        "handle": created["handle"],
                        "session_id": created["session_id"],
                    },
                    timeout_s=min(args.timeout_s, 30.0),
                )
            except Exception as exc:  # noqa: BLE001 - preserve canary evidence.
                cleanup = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    report["cleanup"] = cleanup
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8769/sse")
    parser.add_argument(
        "--env-id", default="openeta/libero_libero_object_task7-v0"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--retreat-z", type=float, default=0.05)
    parser.add_argument("--tolerance-m", type=float, default=0.01)
    parser.add_argument("--orientation-tolerance-rad", type=float, default=0.1)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--timeout-s", type=float, default=240.0)
    parser.add_argument(
        "--output", default="tmp/mink-joint-boundary-recovery-canary.json"
    )
    args = parser.parse_args()
    report = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"passed": report.get("passed"), "checks": report.get("checks")}, indent=2))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
