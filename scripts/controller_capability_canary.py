#!/usr/bin/env python3
"""Live simulator canary for controller capability and execution receipts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
from uuid import uuid4

from agent.tools.sim_mcp import SseSimulatorMcpTransport, close_simulator_mcp_env
from adapter.environment_lifecycle import close_response_error


def _collision_coverage_ok(value: Any, *, endpoint: bool = False) -> bool:
    """Require explicit coverage, not merely a requested check or no detected hit.

    This validates endpoint/post-step monitoring receipts, not swept-path safety.
    """
    return bool(
        isinstance(value, dict)
        and value.get("detected") is False
        and value.get("available") is not False
        and value.get("world_checked") is True
        and value.get("self_checked") is True
        and (not endpoint or (
            value.get("checked") is True
            and value.get("scene_objects_included") is True
        ))
    )


def _eef_xyz(payload: Any) -> list[float]:
    if isinstance(payload, dict):
        end_effector = payload.get("end_effector_pose")
        xyz = end_effector.get("xyz") if isinstance(end_effector, dict) else None
        if (
            isinstance(xyz, list | tuple)
            and len(xyz) >= 3
            and all(
                isinstance(value, int | float)
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in xyz[:3]
            )
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
    require_seed = bool(getattr(args, "require_ik_seed", False))
    through_runtime = bool(getattr(args, "through_agent_runtime", False))
    if through_runtime and (not require_seed or args.disable_collision_check):
        raise ValueError("Agent runtime probe requires --require-ik-seed and collision checks")
    if require_seed and (
        args.expected_controller_id != "mink.robosuite_joint_velocity"
        or args.expected_move_rejection_code
    ):
        raise ValueError("Seed consumption probe requires Mink and a successful nonzero motion")
    # Reject unbounded/ineffective probes before creating a remote environment.
    if (
        type(args.max_steps) is not int
        or args.max_steps <= 0
        or not math.isfinite(args.timeout_s)
        or args.timeout_s <= 0
        or not math.isfinite(args.delta_z_m)
        or not math.isfinite(args.tolerance_m)
        or args.tolerance_m <= 0
        or abs(args.delta_z_m) <= args.tolerance_m
    ):
        raise ValueError("Probe needs positive finite budgets/tolerance and delta outside tolerance")
    transport = SseSimulatorMcpTransport(args.url)
    handle = ""
    session_id = ""
    report: dict[str, Any] = {
        "schema_version": "openeta.controller_capability_canary.v1",
        "url": args.url,
        "env_id": args.env_id,
        "seed": args.seed,
        "passed": False,
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
            raise RuntimeError("create_env did not return a handle; remote outcome needs reconciliation")
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
        report["checks"] = {"declared_expected_controller": expected_controller}
        if not expected_controller:
            raise RuntimeError("Controller contract mismatch; no reset or motion was requested")
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
                "orientation_tolerance_rad": 0.05,
                "check_endpoint_collision": not args.disable_collision_check,
                "include_scene_objects": not args.disable_collision_check,
            },
            timeout_s=args.timeout_s,
        )
        report["checks"]["ik_reachable"] = preview.get("status") == "reachable"
        report["preview"] = {
            "status": preview.get("status"),
            "reason_code": preview.get("reason_code"),
            "collision": preview.get("collision"),
            "path": preview.get("path"),
        }
        if not report["checks"]["ik_reachable"]:
            raise RuntimeError("Preview did not report reachable; no motion was requested")
        endpoint_collision_ok = _collision_coverage_ok(preview.get("collision"), endpoint=True)
        report["checks"]["endpoint_collision_coverage"] = endpoint_collision_ok
        if not args.disable_collision_check and not endpoint_collision_ok:
            raise RuntimeError("Endpoint collision coverage unavailable or unsafe; no motion was requested")
        execution_seed = None
        if require_seed:
            candidate = preview.get("best_candidate")
            joints = candidate.get("joint_positions") if isinstance(candidate, dict) else None
            if (not isinstance(joints, list) or len(joints) != 7 or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) for value in joints
            )):
                raise RuntimeError("Reachable preview did not return seven finite seed joints; no motion requested")
            execution_seed = {
                "schema_version": "openeta.ik_execution_seed.v1",
                "receipt_id": f"controller-canary-{uuid4().hex}",
                "joint_positions": [float(value) for value in joints],
                "preview_tolerances": {
                    "position_tolerance_m": args.tolerance_m,
                    "orientation_tolerance_rad": 0.05,
                },
            }
            report["seed_probe"] = {
                **execution_seed, "source": "direct_preview_not_agent_receipt_resolution",
            }
        move_arguments = {
                **common,
                "x": target[0],
                "y": target[1],
                "z": target[2],
                "tolerance": args.tolerance_m,
                "ori_tolerance": 0.05,
                "num_steps": args.max_steps,
                "enable_collision_check": not args.disable_collision_check,
                **({"ik_execution_seed": execution_seed} if execution_seed else {}),
        }
        if through_runtime:
            from scripts.controller_runtime_probe import runtime_seed_move

            move, execution_seed, runtime_report = runtime_seed_move(
                transport=transport, preview=preview, target=target,
                handle=handle, session_id=session_id, tolerance_m=args.tolerance_m,
                max_steps=args.max_steps, timeout_s=args.timeout_s,
            )
            report["agent_runtime_probe"] = runtime_report
            report["seed_probe"] = {
                **execution_seed, "source": "agent_runtime_exact_receipt_resolution",
            }
        else:
            move = transport.call_tool("move_to", move_arguments, timeout_s=args.timeout_s)
        receipt = move.get("controller_receipt")
        receipt_consistent = bool(
            isinstance(receipt, dict)
            and receipt.get("schema_version")
            == "openeta.controller_execution_receipt.v1"
            and receipt.get("controller_id") == args.expected_controller_id
            and receipt.get("command_interface") == expected_interface
            and receipt.get("goal_executor") == expected_executor
            and type(receipt.get("steps_executed")) is int
            and type(move.get("steps_executed")) is int
            and receipt.get("steps_executed") == move.get("steps_executed")
            and receipt.get("stop_reason") == move.get("stop_reason")
            and receipt.get("reached_target") == move.get("reached_target")
        )
        end_xyz = _eef_xyz({"end_effector_pose": move.get("end")})
        endpoint_matches = bool(
            len(end_xyz) == 3
            and max(abs(actual - desired) for actual, desired in zip(end_xyz, target))
            <= args.tolerance_m
        )
        report["probe"] = {
            "start_xyz": start,
            "target_xyz": target,
            "actual_end": move.get("end"),
            "ik_status": preview.get("status"),
            "ik_reason_code": preview.get("reason_code"),
            "move_reached_target": move.get("reached_target"),
            "move_stop_reason": move.get("stop_reason"),
            "move_code": move.get("code"),
            "move_steps_executed": move.get("steps_executed"),
            "move_position_error_m": move.get("position_error_m"),
            "controller_receipt": receipt,
            "collision": move.get("collision"),
        }
        move_check = (
            move.get("code") == args.expected_move_rejection_code
            and type(move.get("steps_executed")) is int
            and move.get("steps_executed") == 0
            and move.get("reached_target") is False
            if args.expected_move_rejection_code
            else (
                move.get("reached_target") is True
                and type(move.get("steps_executed")) is int
                and 0 < move["steps_executed"] <= args.max_steps
            )
        )
        report["checks"] = {
            "declared_expected_controller": expected_controller,
            "ik_reachable": preview.get("status") == "reachable",
            "collision_checks_requested": not args.disable_collision_check,
            "endpoint_collision_coverage": endpoint_collision_ok,
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
            **({"measured_endpoint_within_tolerance": endpoint_matches}
               if not args.expected_move_rejection_code else {}),
            **({"motion_collision_coverage": _collision_coverage_ok(move.get("collision"))}
               if not args.expected_move_rejection_code else {}),
            **({
                "ik_seed_receipt_bound": isinstance(receipt, dict) and (
                    receipt.get("ik_execution_seed_receipt_id") == execution_seed["receipt_id"]
                ),
                "ik_seed_execution_policy_used": isinstance(receipt, dict) and (
                    receipt.get("execution_policy") == "ik_preview_seeded_cartesian_goal"
                ),
            } if execution_seed else {}),
            **({"agent_pipeline_executed": report["agent_runtime_probe"]["pipeline_status"] == "executed"}
               if through_runtime else {}),
        }
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if handle:
            try:
                report["cleanup"] = close_simulator_mcp_env(
                    transport,
                    handle=handle,
                    session_id=session_id,
                    timeout_s=min(args.timeout_s, 30.0),
                )
            except Exception as exc:
                report["cleanup"] = {
                    "ok": False,
                    "handle": handle,
                    "session_id": session_id,
                    "error": {"type": type(exc).__name__, "message": str(exc)},
                }
            report.setdefault("checks", {})["cleanup_confirmed"] = (
                close_response_error(report["cleanup"]) is None
            )
    report["passed"] = bool(
        not report.get("error")
        and report.get("checks")
        and all(report["checks"].values())
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
    parser.add_argument("--require-ik-seed", action="store_true",
                        help="Require a finite preview seed and bound Mink seeded-execution receipt")
    parser.add_argument("--through-agent-runtime", action="store_true",
                        help="Dispatch one deterministic move via production Agent receipt resolution")
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
