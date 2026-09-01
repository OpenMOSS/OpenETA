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

        def preview_pose(xyz: list[float]) -> dict[str, Any]:
            return transport.call_tool(
                "ik_preview_check",
                {
                    **common,
                    "x": xyz[0],
                    "y": xyz[1],
                    "z": xyz[2],
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

        def execute_pose(
            xyz: list[float],
            preview_result: dict[str, Any],
            *,
            receipt_id: str,
            contact_authorization: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            candidate = preview_result.get("best_candidate")
            joints = (
                candidate.get("joint_positions")
                if isinstance(candidate, dict)
                else None
            )
            if preview_result.get("status") != "reachable" or not isinstance(
                joints, list
            ):
                raise RuntimeError(
                    f"full-pose preview did not return a seed: {preview_result}"
                )
            parameters: dict[str, Any] = {
                **common,
                "x": xyz[0],
                "y": xyz[1],
                "z": xyz[2],
                "roll": euler[0],
                "pitch": euler[1],
                "yaw": euler[2],
                "tolerance": args.position_tolerance_m,
                "ori_tolerance": args.orientation_tolerance_rad,
                "num_steps": args.max_steps,
                "enable_collision_check": not args.disable_collision_check,
                "ik_execution_seed": {
                    "schema_version": "openeta.ik_execution_seed.v1",
                    "receipt_id": receipt_id,
                    "joint_positions": joints,
                    "preview_tolerances": {
                        "position_tolerance_m": args.position_tolerance_m,
                        "orientation_tolerance_rad": args.orientation_tolerance_rad,
                    },
                },
            }
            if contact_authorization is not None:
                parameters["contact_authorization"] = contact_authorization
            return transport.call_tool(
                "move_to",
                parameters,
                timeout_s=args.timeout_s,
            )

        pre_target = (
            [args.pre_x, args.pre_y, args.pre_z]
            if all(value is not None for value in (args.pre_x, args.pre_y, args.pre_z))
            else None
        )
        if any(value is not None for value in (args.pre_x, args.pre_y, args.pre_z)) and (
            pre_target is None
        ):
            raise ValueError("--pre-x, --pre-y, and --pre-z must be supplied together")
        pre_preview: dict[str, Any] | None = None
        pre_move: dict[str, Any] | None = None
        intermediate_results: list[dict[str, Any]] = []
        if pre_target is not None:
            pre_preview = preview_pose(pre_target)
            pre_move = execute_pose(
                pre_target,
                pre_preview,
                receipt_id="mink-explicit-pose-canary-pre",
            )
            for index, fraction in enumerate(args.intermediate_fractions):
                if not 0.0 < fraction < 1.0:
                    raise ValueError("--intermediate-fraction values must be in (0, 1)")
                xyz = [
                    start + fraction * (end - start)
                    for start, end in zip(pre_target, target)
                ]
                intermediate_preview = preview_pose(xyz)
                intermediate_move = execute_pose(
                    xyz,
                    intermediate_preview,
                    receipt_id=f"mink-explicit-pose-canary-intermediate-{index}",
                )
                intermediate_results.append(
                    {
                        "fraction": fraction,
                        "xyz": xyz,
                        "preview": _summary(intermediate_preview),
                        "move": _summary(intermediate_move),
                    }
                )

        contact_authorization: dict[str, Any] | None = None
        if args.authorize_final_contact:
            target_object = next(
                (
                    item
                    for item in reset.get("objects", []) or []
                    if isinstance(item, dict)
                    and item.get("category") == args.target_category
                ),
                None,
            )
            if target_object is None:
                raise RuntimeError(
                    f"target category {args.target_category!r} not found for contact authorization"
                )
            contact_authorization = {
                "schema_version": "openeta.contact_authorization.v1",
                "compiled_grasp_id": "mink-explicit-pose-canary",
                "waypoint_role": "grasp_contact",
                "target_anchor_world_xyz": target_object["position"][:3],
                "target_evidence_id": "mink-explicit-pose-canary",
                "object_scene_epoch": 0,
            }
        preview = preview_pose(target)
        move = execute_pose(
            target,
            preview,
            receipt_id="mink-explicit-pose-canary",
            contact_authorization=contact_authorization,
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
            "pre_target": (
                {"xyz": pre_target, "preview": _summary(pre_preview or {}), "move": _summary(pre_move or {})}
                if pre_target is not None
                else None
            ),
            "intermediate_results": intermediate_results,
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
    parser.add_argument("--pre-x", type=float)
    parser.add_argument("--pre-y", type=float)
    parser.add_argument("--pre-z", type=float)
    parser.add_argument(
        "--intermediate-fraction",
        dest="intermediate_fractions",
        action="append",
        type=float,
        default=[],
        help="Optional fraction along pre-target to target; may be repeated.",
    )
    parser.add_argument("--roll", type=float, default=-170.56168639027211)
    parser.add_argument("--pitch", type=float, default=-35.19625500974049)
    parser.add_argument("--yaw", type=float, default=116.06621440910317)
    parser.add_argument("--position-tolerance-m", type=float, default=0.002)
    parser.add_argument("--orientation-tolerance-rad", type=float, default=0.3)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--disable-collision-check", action="store_true")
    parser.add_argument("--authorize-final-contact", action="store_true")
    parser.add_argument("--target-category", default="milk")
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
