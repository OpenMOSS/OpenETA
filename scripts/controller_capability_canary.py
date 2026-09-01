#!/usr/bin/env python3
"""Live simulator canary for controller capability and execution receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agent.tools.sim_mcp import SseSimulatorMcpTransport, close_simulator_mcp_env


def _eef_xyz(payload: Any) -> list[float]:
    if isinstance(payload, dict):
        end_effector = payload.get("end_effector_pose")
        xyz = end_effector.get("xyz") if isinstance(end_effector, dict) else None
        if (
            isinstance(xyz, list | tuple)
            and len(xyz) >= 3
            and all(isinstance(value, int | float) for value in xyz[:3])
        ):
            return [float(value) for value in xyz[:3]]
        for value in payload.values():
            found = _eef_xyz(value)
            if found:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _eef_xyz(value)
            if found:
                return found
    return []


def run(args: argparse.Namespace) -> dict[str, Any]:
    transport = SseSimulatorMcpTransport(args.url)
    handle = ""
    session_id = ""
    report: dict[str, Any] = {
        "schema_version": "openeta.controller_capability_canary.v1",
        "url": args.url,
        "env_id": args.env_id,
        "seed": args.seed,
    }
    try:
        created = transport.call_tool(
            "create_env",
            {
                "env_id": args.env_id,
                "seed": args.seed,
                "image_width": args.image_size,
                "image_height": args.image_size,
                "include_objects": False,
            },
            timeout_s=args.timeout_s,
        )
        handle = str(created.get("handle") or "")
        session_id = str(created.get("session_id") or "")
        if not handle:
            raise RuntimeError(f"create_env did not return a handle: {created}")
        control_spec = created.get("control_spec")
        controller = (
            control_spec.get("controller") if isinstance(control_spec, dict) else None
        )
        report["create"] = {
            "handle": handle,
            "session_id": session_id,
            "backend": created.get("backend"),
            "control_spec": control_spec,
        }
        expected_contracts = {
            "robosuite.osc_pose": (
                "normalized_cartesian_delta_pose",
                "openeta.outer_closed_loop_cartesian.v1",
            ),
            "mink.robosuite_joint_velocity": (
                "joint_velocity",
                "openeta.worker_mink_goal.v1",
            ),
        }
        expected_interface, expected_executor = expected_contracts[
            args.expected_controller_id
        ]
        expected_controller = bool(
            isinstance(controller, dict)
            and controller.get("controller_id") == args.expected_controller_id
            and controller.get("command_interface") == expected_interface
            and controller.get("goal_executor") == expected_executor
        )
        reset_args: dict[str, Any] = {"handle": handle, "seed": args.seed}
        if session_id:
            reset_args["session_id"] = session_id
        reset = transport.call_tool("reset_env", reset_args, timeout_s=args.timeout_s)
        start = _eef_xyz(reset)
        if len(start) != 3:
            raise RuntimeError("reset_env did not expose the current EEF xyz")
        target = [start[0], start[1], start[2] + args.delta_z_m]
        common = {"handle": handle, "session_id": session_id}
        preview = transport.call_tool(
            "ik_preview_check",
            {
                **common,
                "x": target[0],
                "y": target[1],
                "z": target[2],
                "preserve_current_orientation": True,
                "position_tolerance_m": args.tolerance_m,
            },
            timeout_s=args.timeout_s,
        )
        move = transport.call_tool(
            "move_to",
            {
                **common,
                "x": target[0],
                "y": target[1],
                "z": target[2],
                "tolerance": args.tolerance_m,
                "num_steps": args.max_steps,
                "enable_collision_check": not args.disable_collision_check,
            },
            timeout_s=args.timeout_s,
        )
        receipt = move.get("controller_receipt")
        receipt_consistent = bool(
            isinstance(receipt, dict)
            and receipt.get("schema_version")
            == "openeta.controller_execution_receipt.v1"
            and receipt.get("controller_id") == args.expected_controller_id
            and receipt.get("steps_executed") == move.get("steps_executed")
            and receipt.get("stop_reason") == move.get("stop_reason")
            and receipt.get("reached_target") == move.get("reached_target")
        )
        report["probe"] = {
            "start_xyz": start,
            "target_xyz": target,
            "ik_status": preview.get("status"),
            "ik_reason_code": preview.get("reason_code"),
            "move_reached_target": move.get("reached_target"),
            "move_stop_reason": move.get("stop_reason"),
            "move_code": move.get("code"),
            "move_steps_executed": move.get("steps_executed"),
            "move_position_error_m": move.get("position_error_m"),
            "controller_receipt": receipt,
        }
        move_check = (
            move.get("code") == args.expected_move_rejection_code
            and move.get("steps_executed") == 0
            and move.get("reached_target") is False
            if args.expected_move_rejection_code
            else move.get("reached_target") is True
        )
        report["checks"] = {
            "declared_expected_controller": expected_controller,
            "ik_reachable": preview.get("status") == "reachable",
            (
                "move_safely_rejected"
                if args.expected_move_rejection_code
                else "move_reached_target"
            ): move_check,
            **(
                {"controller_receipt_consistent": receipt_consistent}
                if not args.expected_move_rejection_code
                else {}
            ),
        }
        report["passed"] = all(report["checks"].values())
    finally:
        if handle:
            report["cleanup"] = close_simulator_mcp_env(
                transport,
                handle=handle,
                session_id=session_id,
                timeout_s=min(args.timeout_s, 30.0),
            )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8766/sse")
    parser.add_argument(
        "--env-id",
        default="openeta/libero_libero_object_task2-v0",
    )
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--delta-z-m", type=float, default=0.02)
    parser.add_argument("--tolerance-m", type=float, default=0.003)
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument(
        "--expected-controller-id",
        choices=("robosuite.osc_pose", "mink.robosuite_joint_velocity"),
        default="robosuite.osc_pose",
    )
    parser.add_argument("--disable-collision-check", action="store_true")
    parser.add_argument("--expected-move-rejection-code", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    report = run(args)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
