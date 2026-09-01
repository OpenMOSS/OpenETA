#!/usr/bin/env python3
"""Verify safe Mink recovery when physics starts inside a collision boundary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent.tools.sim_mcp import SseSimulatorMcpTransport


def _eef_xyz(payload: dict[str, Any]) -> list[float]:
    observation = payload.get("observation", payload)
    robot = observation.get("robot") if isinstance(observation, dict) else None
    pose = robot.get("end_effector_pose") if isinstance(robot, dict) else None
    xyz = pose.get("xyz") if isinstance(pose, dict) else None
    return [float(value) for value in xyz[:3]] if isinstance(xyz, list) else []


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
        handle = str(created["handle"])
        session_id = str(created["session_id"])
        common = {"handle": handle, "session_id": session_id}
        reset = transport.call_tool(
            "reset_env", {**common, "seed": args.seed}, timeout_s=args.timeout_s
        )
        opened = transport.call_tool("gripper_open", common, timeout_s=args.timeout_s)
        start = _eef_xyz(reset)
        seed_target = [args.x, args.y, args.seed_z]
        retreat_target = [args.x, args.y, args.retreat_z]

        # This first call is test setup only: it deliberately creates the exact
        # adverse state that the production collision controller must recover
        # from. Agent paths should keep collision checking enabled.
        seeded = transport.call_tool(
            "move_to",
            {
                **common,
                "x": seed_target[0],
                "y": seed_target[1],
                "z": seed_target[2],
                "num_steps": args.max_steps,
                "tolerance": args.tolerance_m,
                "enable_collision_check": False,
            },
            timeout_s=args.timeout_s,
        )
        seeded_xyz = _eef_xyz(seeded)
        retreated = transport.call_tool(
            "move_to",
            {
                **common,
                "x": retreat_target[0],
                "y": retreat_target[1],
                "z": retreat_target[2],
                "num_steps": args.max_steps,
                "tolerance": args.tolerance_m,
                "enable_collision_check": True,
            },
            timeout_s=args.timeout_s,
        )
        retreated_xyz = _eef_xyz(retreated)
        collision = retreated.get("collision")
        collision = collision if isinstance(collision, dict) else {}
        recovery = collision.get("constraint_boundary_recovery")
        recovery = recovery if isinstance(recovery, dict) else {}
        checks = {
            "controller_is_mink": (
                created.get("control_spec", {}).get("controller", {}).get(
                    "controller_id"
                )
                == "mink.robosuite_joint_velocity"
            ),
            # Floor contact can physically prevent the disabled-collision setup
            # move from reaching the requested z.  The setup contract is that it
            # actuated into a low pose; the retreat receipt below is the actual
            # proof that the controller began inside its recovery boundary.
            "adverse_setup_motion_executed": (
                len(seeded_xyz) == 3
                and int(seeded.get("steps_executed") or 0) > 0
                and seeded_xyz[2] < 0.02
            ),
            "verified_escape_used": recovery.get("used") is True,
            "verified_escape_executed_steps": int(
                recovery.get("verified_escape_steps") or 0
            )
            > 0,
            "retreat_reached": retreated.get("reached_target") is True,
            "retreat_increased_height": (
                len(seeded_xyz) == len(retreated_xyz) == 3
                and retreated_xyz[2] > seeded_xyz[2] + 0.05
            ),
            "no_collision_at_retreat_end": collision.get("detected") is False,
        }
        report = {
            "schema_version": "openeta.mink_penetration_escape_canary.v1",
            "url": args.url,
            "env_id": args.env_id,
            "seed": args.seed,
            "start_xyz": start,
            "seed_target_xyz": seed_target,
            "seeded_xyz": seeded_xyz,
            "retreat_target_xyz": retreat_target,
            "retreated_xyz": retreated_xyz,
            "seed_motion": seeded,
            "retreat_motion": retreated,
            "gripper_open": opened,
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
                    timeout_s=args.timeout_s,
                )
            except Exception as exc:  # noqa: BLE001
                cleanup = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    report["cleanup"] = cleanup
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8877/sse")
    parser.add_argument("--env-id", default="openeta/libero_libero_object_task2-v0")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--x", type=float, default=-0.08)
    parser.add_argument("--y", type=float, default=-0.10)
    parser.add_argument("--seed-z", type=float, default=0.0)
    parser.add_argument("--retreat-z", type=float, default=0.20)
    parser.add_argument("--tolerance-m", type=float, default=0.005)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument(
        "--output", default="tmp/mink-penetration-escape-canary-20260819.json"
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
