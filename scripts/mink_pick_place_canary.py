#!/usr/bin/env python3
"""Deterministic physical pick/place canary for one LIBERO object task.

This is an isolation diagnostic, not an Agent policy or runtime task script. It
uses simulator object coordinates to prove whether the controller, carried-object
collision model, gripper latch, receptacle geometry, and official task predicate
can complete a known top-down pick followed by a conservative drop placement.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from agent.tools.sim_mcp import SseSimulatorMcpTransport


def _summary(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in {"cameras", "observation"}
    }


def _observation(value: dict[str, Any]) -> dict[str, Any]:
    observation = value.get("observation", value)
    return observation if isinstance(observation, dict) else {}


def _find_object(
    observation: dict[str, Any],
    *,
    category: str,
) -> dict[str, Any] | None:
    for item in observation.get("objects", []) or []:
        if isinstance(item, dict) and item.get("category") == category:
            return item
    return None


def _object_xyz(observation: dict[str, Any], object_name: str) -> list[float]:
    for item in observation.get("objects", []) or []:
        if isinstance(item, dict) and item.get("name") == object_name:
            position = item.get("position")
            if isinstance(position, list) and len(position) >= 3:
                return [float(value) for value in position[:3]]
    return []


def _latest_reward(*results: dict[str, Any]) -> float:
    rewards = [
        float(result.get("reward") or 0.0)
        for result in results
        if isinstance(result.get("reward"), int | float)
    ]
    return max(rewards, default=0.0)


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
            "reset_env",
            {**common, "seed": args.seed},
            timeout_s=args.timeout_s,
        )
        target = _find_object(reset, category=args.target_category)
        receptacle = _find_object(reset, category=args.receptacle_category)
        if target is None or receptacle is None:
            categories = sorted(
                str(item.get("category") or "")
                for item in reset.get("objects", []) or []
                if isinstance(item, dict)
            )
            raise RuntimeError(
                "target or receptacle category was not found; available="
                + ",".join(categories)
            )

        target_name = str(target["name"])
        receptacle_name = str(receptacle["name"])
        target_start = [float(value) for value in target["position"][:3]]
        receptacle_start = [float(value) for value in receptacle["position"][:3]]
        contact = [
            target_start[0] + args.contact_x_offset_m,
            target_start[1] + args.contact_y_offset_m,
            target_start[2] + args.contact_z_offset_m,
        ]
        waypoints = {
            "clearance": [target_start[0], target_start[1], target_start[2] + 0.19],
            "precontact": [
                target_start[0],
                target_start[1],
                target_start[2] + 0.105,
            ],
            "contact": contact,
            "lift": [contact[0], contact[1], contact[2] + args.lift_distance_m],
            "transit": [
                contact[0],
                args.transit_y_m,
                args.transport_height_m,
            ],
            "above_receptacle": [
                receptacle_start[0],
                receptacle_start[1],
                args.transport_height_m,
            ],
            "retreat": [
                receptacle_start[0],
                receptacle_start[1],
                args.retreat_height_m,
            ],
        }
        orientation = {
            "roll": args.roll,
            "pitch": args.pitch,
            "yaw": args.yaw,
            "preserve_current_orientation": False,
            "orientation_tolerance_rad": args.orientation_tolerance_rad,
        }

        def preview(xyz: list[float]) -> dict[str, Any]:
            return transport.call_tool(
                "ik_preview_check",
                {
                    **common,
                    "x": xyz[0],
                    "y": xyz[1],
                    "z": xyz[2],
                    **orientation,
                    "position_tolerance_m": args.tolerance_m,
                    "check_endpoint_collision": False,
                },
                timeout_s=args.timeout_s,
            )

        authorization = {
            "schema_version": "openeta.contact_authorization.v1",
            "compiled_grasp_id": "mink-pick-place-physical-canary",
            "waypoint_role": "grasp_contact",
            "target_anchor_world_xyz": target_start,
            "target_evidence_id": "mink-pick-place-physical-canary",
            "object_scene_epoch": 0,
        }

        def move(
            xyz: list[float],
            *,
            contact_authorization: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            parameters: dict[str, Any] = {
                **common,
                "x": xyz[0],
                "y": xyz[1],
                "z": xyz[2],
                "num_steps": args.max_steps,
                "tolerance": args.tolerance_m,
                "ori_tolerance": args.orientation_tolerance_rad,
                "enable_collision_check": True,
                **orientation,
            }
            if contact_authorization is not None:
                parameters["contact_authorization"] = contact_authorization
            return transport.call_tool("move_to", parameters, timeout_s=args.timeout_s)

        opened = transport.call_tool("gripper_open", common, timeout_s=args.timeout_s)
        previews = {name: preview(xyz) for name, xyz in waypoints.items()}
        motions: dict[str, dict[str, Any]] = {}
        for name in ("clearance", "precontact"):
            motions[name] = move(waypoints[name])
            if motions[name].get("reached_target") is not True:
                break
        else:
            motions["contact"] = move(
                waypoints["contact"],
                contact_authorization=authorization,
            )
            if motions["contact"].get("reached_target") is True:
                closed = transport.call_tool(
                    "gripper_close",
                    {**common, "contact_authorization": authorization},
                    timeout_s=args.timeout_s,
                )
                motions["close"] = closed
                for name in ("lift", "transit", "above_receptacle"):
                    motions[name] = move(waypoints[name])
                    if motions[name].get("reached_target") is not True:
                        break
                else:
                    released = transport.call_tool(
                        "gripper_open", common, timeout_s=args.timeout_s
                    )
                    motions["release"] = released
                    if released.get("terminated") is not True:
                        motions["retreat"] = move(waypoints["retreat"])

        final_observation_result = transport.call_tool(
            "observe_env", common, timeout_s=args.timeout_s
        )
        final_observation = _observation(final_observation_result)
        target_final = _object_xyz(final_observation, target_name)
        receptacle_final = _object_xyz(final_observation, receptacle_name)
        reward = _latest_reward(*motions.values(), final_observation_result)
        terminated = any(
            result.get("terminated") is True for result in motions.values()
        ) or final_observation_result.get("terminated") is True
        target_receptacle_xy_distance_m = (
            math.dist(target_final[:2], receptacle_final[:2])
            if len(target_final) == len(receptacle_final) == 3
            else None
        )
        checks = {
            "controller_is_mink": (
                created.get("control_spec", {}).get("controller", {}).get(
                    "controller_id"
                )
                == "mink.robosuite_joint_velocity"
            ),
            "all_waypoints_ik_reachable": all(
                result.get("status") == "reachable" for result in previews.values()
            ),
            "contact_reached": motions.get("contact", {}).get("reached_target") is True,
            "nonempty_close": (
                float(
                    motions.get("close", {})
                    .get("gripper_actuation_receipt", {})
                    .get("measured_open_fraction")
                    or 0.0
                )
                >= args.nonempty_close_threshold
            ),
            "lift_reached": motions.get("lift", {}).get("reached_target") is True,
            "transit_reached": motions.get("transit", {}).get("reached_target") is True,
            "above_receptacle_reached": (
                motions.get("above_receptacle", {}).get("reached_target") is True
            ),
            "release_executed": "release" in motions,
            "target_near_receptacle_xy": (
                target_receptacle_xy_distance_m is not None
                and target_receptacle_xy_distance_m <= args.receptacle_xy_tolerance_m
            ),
            "official_reward_success": reward > 0.0,
            "official_termination_success": terminated and reward > 0.0,
        }
        report = {
            "schema_version": "openeta.mink_pick_place_physical_canary.v1",
            "diagnostic_only": True,
            "not_agent_policy": True,
            "url": args.url,
            "env_id": args.env_id,
            "seed": args.seed,
            "target": {
                "name": target_name,
                "category": args.target_category,
                "start_xyz": target_start,
                "final_xyz": target_final,
            },
            "receptacle": {
                "name": receptacle_name,
                "category": args.receptacle_category,
                "start_xyz": receptacle_start,
                "final_xyz": receptacle_final,
            },
            "waypoints": waypoints,
            "previews": {name: _summary(value) for name, value in previews.items()},
            "motions": {name: _summary(value) for name, value in motions.items()},
            "measurements": {
                "target_receptacle_xy_distance_m": target_receptacle_xy_distance_m,
                "reward": reward,
                "terminated": terminated,
            },
            "checks": checks,
            "passed": all(checks.values()),
            "create": _summary(created),
            "reset_objects": reset.get("objects", []),
            "initial_gripper_open": _summary(opened),
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
    parser.add_argument("--url", default="http://127.0.0.1:8766/sse")
    parser.add_argument(
        "--env-id", default="openeta/libero_libero_object_task7-v0"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target-category", default="milk")
    parser.add_argument("--receptacle-category", default="basket")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--contact-x-offset-m", type=float, default=0.0)
    parser.add_argument("--contact-y-offset-m", type=float, default=0.0)
    parser.add_argument("--contact-z-offset-m", type=float, default=0.02)
    parser.add_argument("--lift-distance-m", type=float, default=0.12)
    parser.add_argument("--transport-height-m", type=float, default=0.35)
    parser.add_argument("--retreat-height-m", type=float, default=0.42)
    parser.add_argument("--transit-y-m", type=float, default=0.05)
    parser.add_argument("--roll", type=float, default=180.0)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--orientation-tolerance-rad", type=float, default=0.1)
    parser.add_argument("--tolerance-m", type=float, default=0.005)
    parser.add_argument("--nonempty-close-threshold", type=float, default=0.2)
    parser.add_argument("--receptacle-xy-tolerance-m", type=float, default=0.12)
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--timeout-s", type=float, default=240.0)
    parser.add_argument(
        "--output", default="tmp/mink-task7-milk-pick-place-physical-canary.json"
    )
    args = parser.parse_args()
    report = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": report.get("passed"),
                "checks": report.get("checks"),
                "measurements": report.get("measurements"),
                "output": str(output),
            },
            indent=2,
        )
    )
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
