#!/usr/bin/env python3
"""Deterministic simulator-only canary for motion conditions A, B, and C."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Any

from agent.tools.sim_mcp import SseSimulatorMcpTransport, close_simulator_mcp_env


ORIENTATION_TOLERANCE_RAD = 0.30


def _find_pose(payload: object) -> tuple[list[float], list[float]]:
    if isinstance(payload, dict):
        pose = payload.get("end_effector_pose")
        if isinstance(pose, dict):
            xyz = pose.get("xyz")
            quat = pose.get("quat_xyzw")
            if (
                isinstance(xyz, list)
                and len(xyz) >= 3
                and isinstance(quat, list)
                and len(quat) >= 4
            ):
                return [float(value) for value in xyz[:3]], [
                    float(value) for value in quat[:4]
                ]
        for value in payload.values():
            xyz, quat = _find_pose(value)
            if xyz:
                return xyz, quat
    elif isinstance(payload, list):
        for value in payload:
            xyz, quat = _find_pose(value)
            if xyz:
                return xyz, quat
    return [], []


def _find_objects(payload: object) -> dict[str, list[float]]:
    found: dict[str, list[float]] = {}
    if isinstance(payload, dict):
        objects = payload.get("objects")
        if isinstance(objects, list):
            for index, item in enumerate(objects):
                if not isinstance(item, dict):
                    continue
                xyz = item.get("position", item.get("xyz"))
                if not isinstance(xyz, list) or len(xyz) < 3:
                    continue
                if not all(
                    isinstance(value, int | float) and not isinstance(value, bool)
                    for value in xyz[:3]
                ):
                    continue
                name = str(item.get("name") or f"object_{index}")
                found[name] = [float(value) for value in xyz[:3]]
        for value in payload.values():
            found.update(_find_objects(value))
    elif isinstance(payload, list):
        for value in payload:
            found.update(_find_objects(value))
    return found


def _distance(left: object, right: object) -> float | None:
    if (
        not isinstance(left, list | tuple)
        or not isinstance(right, list | tuple)
        or len(left) < 3
        or len(right) < 3
    ):
        return None
    return math.sqrt(sum((left[index] - right[index]) ** 2 for index in range(3)))


def _object_displacement(
    before: dict[str, list[float]],
    after: dict[str, list[float]],
) -> dict[str, Any]:
    values = {
        name: _distance(before[name], after[name])
        for name in sorted(before.keys() & after.keys())
    }
    values = {name: value for name, value in values.items() if value is not None}
    maximum = max(values.values(), default=None)
    return {
        "object_count": len(values),
        "max_displacement_m": maximum,
        "objects_over_5mm": sorted(
            name for name, value in values.items() if value > 0.005
        ),
        "per_object_m": values,
    }


def _preview(
    transport: SseSimulatorMcpTransport,
    common: dict[str, Any],
    pose: dict[str, Any],
    *,
    timeout_s: float,
) -> dict[str, Any]:
    xyz = pose["xyz"]
    return transport.call_tool(
        "ik_preview_check",
        {
            **common,
            "x": xyz[0],
            "y": xyz[1],
            "z": xyz[2],
            "preserve_current_orientation": True,
            "position_tolerance_m": 0.01,
            "orientation_tolerance_rad": ORIENTATION_TOLERANCE_RAD,
            "check_endpoint_collision": True,
        },
        timeout_s=timeout_s,
    )


def _seed(preview: dict[str, Any], source_id: str) -> dict[str, Any] | None:
    candidate = preview.get("best_candidate")
    joints = candidate.get("joint_positions") if isinstance(candidate, dict) else None
    if not isinstance(joints, list) or len(joints) != 7:
        return None
    return {
        "schema_version": "openeta.ik_execution_seed.v1",
        "receipt_id": source_id,
        "pose_policy_signature": f"abc-canary:{source_id}",
        "joint_positions": [float(value) for value in joints],
        "preview_tolerances": {
            "position_tolerance_m": 0.01,
            "orientation_tolerance_rad": ORIENTATION_TOLERANCE_RAD,
        },
    }


def _route_bundle(
    route: list[dict[str, Any]],
    previews: list[dict[str, Any]],
) -> dict[str, Any]:
    entries = []
    for index, (pose, preview) in enumerate(zip(route, previews, strict=True)):
        source_id = hashlib.sha256(
            json.dumps(
                {"index": index, "pose": pose, "preview": preview.get("target")},
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:20]
        private_pose = dict(pose)
        target = preview.get("target")
        target = target if isinstance(target, dict) else {}
        quat = target.get("quat_xyzw")
        if isinstance(quat, list) and len(quat) == 4:
            private_pose["quat_xyzw"] = [float(value) for value in quat]
        candidate = preview.get("best_candidate")
        candidate = candidate if isinstance(candidate, dict) else {}
        entries.append(
            {
                "index": index,
                "source_ik_receipt_id": source_id,
                "target_pose": private_pose,
                "orientation_policy": "preserve_current",
                "source_captured_quat_xyzw": (
                    [float(value) for value in quat]
                    if isinstance(quat, list) and len(quat) == 4
                    else None
                ),
                "tolerances": {
                    "position_tolerance_m": 0.01,
                    "orientation_tolerance_rad": ORIENTATION_TOLERANCE_RAD,
                },
                "source_seed_hint": {
                    "joint_positions": list(candidate.get("joint_positions") or []),
                },
            }
        )
    return {
        "schema_version": "openeta.experimental_route_execution_bundle.v1",
        "condition": "C",
        "authority": "host_memory_exact_receipt_resolution",
        "sequential_preview_policy": "just_in_time_from_actual_segment_end",
        "path_collision_authority": "controller_per_step_only",
        "entries": entries,
    }


def _motion_metrics(
    response: dict[str, Any],
    *,
    target_xyz: list[float],
    elapsed_s: float,
    displacement: dict[str, Any],
) -> dict[str, Any]:
    end = response.get("end")
    end_xyz = end.get("xyz") if isinstance(end, dict) else []
    collision = response.get("collision")
    collision = collision if isinstance(collision, dict) else {}
    receipt = response.get("controller_receipt")
    receipt = receipt if isinstance(receipt, dict) else {}
    chain = response.get("sequential_route_preview")
    chain = chain if isinstance(chain, dict) else {}
    waypoint_results = response.get("waypoint_results")
    waypoint_results = waypoint_results if isinstance(waypoint_results, list) else []
    stable_counts = [
        item.get("controller_receipt", {}).get("stable_steps_completed")
        for item in waypoint_results
        if isinstance(item, dict) and isinstance(item.get("controller_receipt"), dict)
    ]
    return {
        "reached_target": response.get("reached_target") is True,
        "stop_reason": response.get("stop_reason") or receipt.get("stop_reason"),
        "code": response.get("code"),
        "steps_executed": response.get("steps_executed"),
        "waypoints_requested": response.get("waypoints_requested"),
        "waypoints_completed": response.get("waypoints_completed"),
        "end_xyz": end_xyz,
        "position_error_m": _distance(target_xyz, end_xyz),
        "collision_detected": collision.get("detected") is True,
        "minimum_distance_m": collision.get("minimum_distance_m"),
        "motion_execution_profile": response.get("motion_execution_profile")
        or receipt.get("motion_execution_profile"),
        "stable_steps_completed": receipt.get("stable_steps_completed"),
        "waypoint_stable_steps_completed": stable_counts,
        "joint_velocity_max_abs_rad_s": receipt.get(
            "joint_velocity_max_abs_rad_s"
        ),
        "convergence_stall_kind": receipt.get("convergence_stall_kind"),
        "progress_diagnostics": receipt.get("progress_diagnostics"),
        "sequential_waypoints_previewed": chain.get("waypoints_previewed"),
        "sequential_waypoints_authorized": chain.get("waypoints_authorized"),
        "elapsed_s": elapsed_s,
        "object_displacement": displacement,
    }


def _run_scenario(
    transport: SseSimulatorMcpTransport,
    args: argparse.Namespace,
    scenario: str,
) -> dict[str, Any]:
    handle = ""
    session_id = ""
    report: dict[str, Any] = {"scenario": scenario}
    try:
        created = transport.call_tool(
            "create_env",
            {
                "env_id": args.env_id,
                "seed": args.seed,
                "image_width": 256,
                "image_height": 256,
                "include_objects": True,
            },
            timeout_s=args.timeout_s,
        )
        handle = str(created.get("handle") or "")
        session_id = str(created.get("session_id") or "")
        if not handle:
            raise RuntimeError(f"create_env returned no handle: {created}")
        common = {"handle": handle, "session_id": session_id}
        reset = transport.call_tool(
            "reset_env", {**common, "seed": args.seed}, timeout_s=args.timeout_s
        )
        start_xyz, _ = _find_pose(reset)
        initial_observed = transport.call_tool(
            "observe_env", common, timeout_s=args.timeout_s
        )
        before_objects = (
            _find_objects(initial_observed)
            or _find_objects(reset)
            or _find_objects(created)
        )
        if not start_xyz:
            raise RuntimeError("reset_env returned no EEF pose")
        if scenario == "short_up":
            route = [
                {
                    "frame": "world",
                    "xyz": [start_xyz[0], start_xyz[1], start_xyz[2] + 0.02],
                }
            ]
        elif scenario == "clear_route":
            route = [
                {"frame": "world", "xyz": [start_xyz[0], start_xyz[1], 0.40]},
                {"frame": "world", "xyz": [0.10, -0.32, 0.40]},
                {"frame": "world", "xyz": [0.10, -0.32, 0.12]},
            ]
        elif scenario == "collision_crossing":
            route = [
                {"frame": "world", "xyz": [-0.25, 0.0, 0.18]},
                {"frame": "world", "xyz": [-0.25, 0.0, 0.12]},
                {"frame": "world", "xyz": [0.20, 0.0, 0.12]},
            ]
        else:
            raise ValueError(f"unsupported scenario {scenario!r}")
        previews = [
            _preview(transport, common, pose, timeout_s=args.timeout_s)
            for pose in route
        ]
        report["preview_statuses"] = [preview.get("status") for preview in previews]
        report["start_xyz"] = start_xyz
        report["route"] = route
        if any(preview.get("status") != "reachable" for preview in previews):
            report["skipped"] = "initial endpoint preview was not reachable"
            return report
        started = time.monotonic()
        if scenario == "short_up":
            pose = route[0]
            source_id = "abc-short-up"
            arguments: dict[str, Any] = {
                **common,
                "x": pose["xyz"][0],
                "y": pose["xyz"][1],
                "z": pose["xyz"][2],
                "num_steps": 100,
                "tolerance": 0.003,
                "ori_tolerance": ORIENTATION_TOLERANCE_RAD,
                "enable_collision_check": True,
            }
            seed = _seed(previews[0], source_id)
            if seed is not None:
                arguments["ik_execution_seed"] = seed
            response = transport.call_tool(
                "move_to", arguments, timeout_s=args.timeout_s
            )
        else:
            arguments = {
                **common,
                "trajectory": route,
                "num_steps_per_waypoint": 100,
                "tolerance": 0.01,
                "ori_tolerance": ORIENTATION_TOLERANCE_RAD,
                "enable_collision_check": True,
            }
            if args.condition == "C":
                arguments["route_execution_bundle"] = _route_bundle(route, previews)
            response = transport.call_tool(
                "follow_eef_trajectory", arguments, timeout_s=args.timeout_s
            )
        elapsed = time.monotonic() - started
        observed = transport.call_tool(
            "observe_env", common, timeout_s=args.timeout_s
        )
        displacement = _object_displacement(before_objects, _find_objects(observed))
        report["metrics"] = _motion_metrics(
            response,
            target_xyz=route[-1]["xyz"],
            elapsed_s=elapsed,
            displacement=displacement,
        )
        report["response"] = response
        return report
    finally:
        if handle:
            report["cleanup"] = close_simulator_mcp_env(
                transport,
                handle=handle,
                session_id=session_id,
                timeout_s=min(args.timeout_s, 30.0),
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8770/sse")
    parser.add_argument("--condition", choices=("A", "B", "C"), required=True)
    parser.add_argument(
        "--env-id", default="openeta/libero_libero_object_task2-v0"
    )
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument(
        "--scenario",
        action="append",
        choices=("short_up", "clear_route", "collision_crossing"),
        help="Run only the selected scenario; repeat to select multiple.",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    transport = SseSimulatorMcpTransport(args.url)
    scenarios = args.scenario or ["short_up", "clear_route", "collision_crossing"]
    report = {
        "schema_version": "openeta.motion_control_abc_canary.v1",
        "condition": args.condition,
        "url": args.url,
        "env_id": args.env_id,
        "seed": args.seed,
        "scenarios": [
            _run_scenario(transport, args, scenario)
            for scenario in scenarios
        ],
    }
    report["passed"] = all(
        scenario.get("metrics", {}).get("reached_target") is True
        and scenario.get("metrics", {}).get("collision_detected") is False
        for scenario in report["scenarios"]
        if not scenario.get("skipped")
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "condition": args.condition,
                "passed": report["passed"],
                "scenarios": [
                    {
                        "scenario": scenario.get("scenario"),
                        "preview_statuses": scenario.get("preview_statuses"),
                        "skipped": scenario.get("skipped"),
                        "metrics": scenario.get("metrics"),
                    }
                    for scenario in report["scenarios"]
                ],
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
