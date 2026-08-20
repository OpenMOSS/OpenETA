#!/usr/bin/env python3
"""Replay a difficult full-pose target through preview-bound Mink execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent.tools.sim_mcp import SseSimulatorMcpTransport


def _summary(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in {"cameras", "observation"}
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    transport = SseSimulatorMcpTransport(args.url)
    created: dict[str, Any] = {}
    report: dict[str, Any] = {}
    cleanup: dict[str, Any] = {}
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
        handle = str(created["handle"])
        session_id = str(created["session_id"])
        common = {"handle": handle, "session_id": session_id}
        reset = transport.call_tool(
            "reset_env",
            {**common, "seed": args.seed},
            timeout_s=args.timeout_s,
        )
        target = [args.x, args.y, args.z]
        euler = [args.roll, args.pitch, args.yaw]
        preview = transport.call_tool(
            "ik_preview_check",
            {
                **common,
                "x": target[0],
                "y": target[1],
                "z": target[2],
                "roll": euler[0],
                "pitch": euler[1],
                "yaw": euler[2],
                "preserve_current_orientation": False,
                "position_tolerance_m": args.position_tolerance_m,
                "orientation_tolerance_rad": args.orientation_tolerance_rad,
                "check_endpoint_collision": False,
                "timeout_s": args.ik_timeout_s,
            },
            timeout_s=args.timeout_s,
        )
        candidate = preview.get("best_candidate")
        joints = (
            candidate.get("joint_positions")
            if isinstance(candidate, dict)
            else None
        )
        if preview.get("status") != "reachable" or not isinstance(joints, list):
            raise RuntimeError(f"full-pose preview did not return a seed: {preview}")
        seed_receipt = {
            "schema_version": "openeta.ik_execution_seed.v1",
            "receipt_id": "mink-explicit-pose-canary",
            "joint_positions": joints,
            "preview_tolerances": {
                "position_tolerance_m": args.position_tolerance_m,
                "orientation_tolerance_rad": args.orientation_tolerance_rad,
            },
        }
        move = transport.call_tool(
            "move_to",
            {
                **common,
                "x": target[0],
                "y": target[1],
                "z": target[2],
                "roll": euler[0],
                "pitch": euler[1],
                "yaw": euler[2],
                "tolerance": args.position_tolerance_m,
                "ori_tolerance": args.orientation_tolerance_rad,
                "num_steps": args.max_steps,
                "enable_collision_check": not args.disable_collision_check,
                "ik_execution_seed": seed_receipt,
            },
            timeout_s=args.timeout_s,
        )
        receipt = move.get("controller_receipt")
        receipt = receipt if isinstance(receipt, dict) else {}
        collision = move.get("collision")
        collision = collision if isinstance(collision, dict) else {}
        checks = {
            "preview_reachable": preview.get("status") == "reachable",
            "seeded_execution_policy_used": (
                receipt.get("execution_policy")
                == "ik_preview_seeded_cartesian_goal"
            ),
            "seed_receipt_bound": (
                receipt.get("ik_execution_seed_receipt_id")
                == "mink-explicit-pose-canary"
            ),
            "target_reached": move.get("reached_target") is True,
            "position_within_tolerance": (
                isinstance(move.get("max_axis_position_error_m"), int | float)
                and float(move["max_axis_position_error_m"])
                <= args.position_tolerance_m
            ),
            "orientation_within_tolerance": (
                isinstance(move.get("orientation_error_rad"), int | float)
                and float(move["orientation_error_rad"])
                <= args.orientation_tolerance_rad
            ),
            "no_collision": collision.get("detected") is not True,
        }
        report = {
            "schema_version": "openeta.mink_explicit_pose_canary.v1",
            "url": args.url,
            "env_id": args.env_id,
            "seed": args.seed,
            "target": {"xyz": target, "euler_xyz_deg": euler},
            "create": _summary(created),
            "reset": _summary(reset),
            "preview": _summary(preview),
            "move": _summary(move),
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
    parser.add_argument("--url", default="http://127.0.0.1:8877/sse")
    parser.add_argument(
        "--env-id", default="openeta/libero_libero_object_task2-v0"
    )
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=128)
    # Exact difficult target exposed by live run r6 on 2026-08-19.
    parser.add_argument("--x", type=float, default=0.119494411745)
    parser.add_argument("--y", type=float, default=-0.178418484208)
    parser.add_argument("--z", type=float, default=0.264357553633)
    parser.add_argument("--roll", type=float, default=-170.56168639027211)
    parser.add_argument("--pitch", type=float, default=-35.19625500974049)
    parser.add_argument("--yaw", type=float, default=116.06621440910317)
    parser.add_argument("--position-tolerance-m", type=float, default=0.002)
    parser.add_argument("--orientation-tolerance-rad", type=float, default=0.3)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--disable-collision-check", action="store_true")
    parser.add_argument("--ik-timeout-s", type=float, default=20.0)
    parser.add_argument("--timeout-s", type=float, default=240.0)
    parser.add_argument(
        "--output", default="tmp/mink-explicit-pose-canary-20260819.json"
    )
    args = parser.parse_args()
    report = run(args)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": report.get("passed"),
                "checks": report.get("checks"),
                "move": report.get("move"),
            },
            indent=2,
        )
    )
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
