#!/usr/bin/env python3
"""Deterministic LIBERO approach/contact/close/lift canary for worker Mink."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from agent.tools.sim_mcp import SseSimulatorMcpTransport


def _xyz(observation: dict[str, Any], *, object_name: str = "") -> list[float]:
    if object_name:
        for item in observation.get("objects", []) or []:
            if isinstance(item, dict) and item.get("name") == object_name:
                return [float(value) for value in item.get("position", [])[:3]]
        return []
    robot = observation.get("robot") if isinstance(observation, dict) else None
    pose = robot.get("end_effector_pose") if isinstance(robot, dict) else None
    xyz = pose.get("xyz") if isinstance(pose, dict) else None
    return [float(value) for value in xyz[:3]] if isinstance(xyz, list) else []


def _observation(result: dict[str, Any]) -> dict[str, Any]:
    value = result.get("observation", result)
    return value if isinstance(value, dict) else {}


def _summary(result: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in result.items()
        if key not in {"cameras", "observation"}
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
        handle = str(created["handle"])
        session_id = str(created["session_id"])
        common = {"handle": handle, "session_id": session_id}
        reset = transport.call_tool(
            "reset_env", {**common, "seed": args.seed}, timeout_s=args.timeout_s
        )
        target = next(
            (
                item
                for item in reset.get("objects", []) or []
                if isinstance(item, dict) and item.get("category") == args.target_category
            ),
            None,
        )
        if target is None:
            raise RuntimeError(f"target category {args.target_category!r} not found")
        object_name = str(target["name"])
        object_start = [float(value) for value in target["position"][:3]]
        clearance = [object_start[0], object_start[1], object_start[2] + 0.19]
        precontact = [object_start[0], object_start[1], object_start[2] + 0.105]
        contact = [
            object_start[0] + args.contact_x_offset_m,
            object_start[1] + args.contact_y_offset_m,
            object_start[2] + args.contact_z_offset_m,
        ]
        lift = [contact[0], contact[1], contact[2] + args.lift_distance_m]

        opened = transport.call_tool("gripper_open", common, timeout_s=args.timeout_s)

        explicit_orientation = all(
            value is not None for value in (args.roll, args.pitch, args.yaw)
        )
        if any(value is not None for value in (args.roll, args.pitch, args.yaw)) and not (
            explicit_orientation
        ):
            raise ValueError("--roll, --pitch, and --yaw must be supplied together")

        def orientation_parameters() -> dict[str, Any]:
            if not explicit_orientation:
                return {"preserve_current_orientation": True}
            return {
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
                    **orientation_parameters(),
                    "position_tolerance_m": args.tolerance_m,
                    "check_endpoint_collision": False,
                },
                timeout_s=args.timeout_s,
            )

        def move(
            xyz: list[float],
            *,
            authorization: dict[str, Any] | None = None,
        ) -> dict[str, Any]:
            payload: dict[str, Any] = {
                **common,
                "x": xyz[0],
                "y": xyz[1],
                "z": xyz[2],
                "num_steps": args.max_steps,
                "tolerance": args.tolerance_m,
                "enable_collision_check": True,
                **orientation_parameters(),
            }
            if explicit_orientation:
                payload["ori_tolerance"] = args.orientation_tolerance_rad
            if authorization is not None:
                payload["contact_authorization"] = authorization
            return transport.call_tool("move_to", payload, timeout_s=args.timeout_s)

        previews = {
            "clearance": preview(clearance),
            "precontact": preview(precontact),
            "contact": preview(contact),
            "lift": preview(lift),
        }
        clearance_result = move(clearance)
        precontact_result = move(precontact)
        unauthorized_contact_result = move(contact)
        authorization = {
            "schema_version": "openeta.contact_authorization.v1",
            "compiled_grasp_id": "controller-contact-canary",
            "waypoint_role": "grasp_contact",
            "target_anchor_world_xyz": object_start,
            "target_evidence_id": "controller-contact-canary",
            "object_scene_epoch": 0,
        }
        contact_result = move(contact, authorization=authorization)
        closed = transport.call_tool(
            "gripper_close",
            {**common, "contact_authorization": authorization},
            timeout_s=args.timeout_s,
        )
        close_observation = _observation(closed)
        object_after_close = _xyz(close_observation, object_name=object_name)
        lift_result = move(lift)
        lift_observation = _observation(lift_result)
        object_after_lift = _xyz(lift_observation, object_name=object_name)
        eef_after_lift = _xyz(lift_observation)
        object_lift_m = (
            object_after_lift[2] - object_after_close[2]
            if len(object_after_close) == len(object_after_lift) == 3
            else None
        )
        object_eef_distance_m = (
            math.dist(object_after_lift, eef_after_lift)
            if len(object_after_lift) == len(eef_after_lift) == 3
            else None
        )
        collision_receipts = {
            "clearance": clearance_result.get("collision"),
            "precontact": precontact_result.get("collision"),
            "contact": contact_result.get("collision"),
            "unauthorized_contact": unauthorized_contact_result.get("collision"),
            "lift": lift_result.get("collision"),
        }
        lift_collision = collision_receipts["lift"]
        lift_collision = (
            lift_collision if isinstance(lift_collision, dict) else {}
        )
        attached_coverage = lift_collision.get("attached_object_coverage")
        attached_coverage = (
            attached_coverage if isinstance(attached_coverage, dict) else {}
        )
        attachment = lift_collision.get("contact_policy", {}).get("attachment")
        attachment = attachment if isinstance(attachment, dict) else {}
        attachment_dims = attachment.get("dims")
        attachment_dims = (
            [float(value) for value in attachment_dims[:3]]
            if isinstance(attachment_dims, list) and len(attachment_dims) >= 3
            else []
        )
        close_attachment = closed.get("attachment_proxy_receipt")
        close_attachment = (
            close_attachment if isinstance(close_attachment, dict) else {}
        )
        lift_attachment_receipt = lift_result.get("attachment_proxy_receipt")
        lift_attachment_receipt = (
            lift_attachment_receipt
            if isinstance(lift_attachment_receipt, dict)
            else {}
        )
        checks = {
            "controller_is_mink": (
                created.get("control_spec", {}).get("controller", {}).get("controller_id")
                == "mink.robosuite_joint_velocity"
            ),
            "all_ik_reachable": all(
                result.get("status") == "reachable" for result in previews.values()
            ),
            "clearance_reached": clearance_result.get("reached_target") is True,
            "precontact_reached": precontact_result.get("reached_target") is True,
            "unauthorized_contact_stopped": (
                unauthorized_contact_result.get("reached_target") is False
                and int(unauthorized_contact_result.get("steps_executed") or 0) > 0
            ),
            "contact_reached": contact_result.get("reached_target") is True,
            "contact_bound_to_target": (
                isinstance(contact_result.get("collision"), dict)
                and contact_result["collision"].get("contact_policy", {}).get(
                    "authorized_target_object"
                )
                == object_name
            ),
            "close_attachment_bound_to_target": (
                close_attachment.get("status") == "tentative"
                and close_attachment.get("target_object_name") == object_name
                and close_attachment.get("binding_source")
                == "host_compiled_target_provenance"
            ),
            "close_does_not_claim_attachment_proof": (
                close_attachment.get("attachment_proven") is False
            ),
            "lift_reached": lift_result.get("reached_target") is True,
            "object_lifted": object_lift_m is not None and object_lift_m >= 0.015,
            "object_remained_near_eef": (
                object_eef_distance_m is not None and object_eef_distance_m <= 0.12
            ),
            "lift_attached_trajectory_checked": (
                lift_collision.get("trajectory_checked") is True
                and attached_coverage.get("trajectory_checked") is True
                and attached_coverage.get("predicted_step_checked") is True
                and attached_coverage.get("actual_step_checked") is True
            ),
            "lift_attachment_bound_to_target": (
                attachment.get("object_name") == object_name
            ),
            "lift_keeps_conservative_proxy_without_claiming_proof": (
                lift_attachment_receipt.get("status") == "tentative"
                and lift_attachment_receipt.get("reason")
                == "awaiting_independent_co_motion_evidence"
                and lift_attachment_receipt.get("target_object_name") == object_name
                and lift_attachment_receipt.get("attachment_proven") is False
            ),
            "attachment_bounds_are_shape_aware": (
                len(attachment_dims) == 3
                and max(attachment_dims) <= 0.30
                and min(attachment_dims) <= 0.10
                and max(attachment_dims) >= min(attachment_dims) * 1.5
            ),
        }
        report = {
            "schema_version": "openeta.mink_contact_canary.v1",
            "url": args.url,
            "env_id": args.env_id,
            "seed": args.seed,
            "target": {
                "name": object_name,
                "category": args.target_category,
                "start_xyz": object_start,
            },
            "waypoints": {
                "clearance": clearance,
                "precontact": precontact,
                "contact": contact,
                "lift": lift,
            },
            "orientation": {
                "policy": (
                    "explicit_rpy" if explicit_orientation else "preserve_current"
                ),
                "roll_pitch_yaw": (
                    [args.roll, args.pitch, args.yaw]
                    if explicit_orientation
                    else None
                ),
            },
            "create": _summary(created),
            "previews": {name: _summary(value) for name, value in previews.items()},
            "motions": {
                "clearance": _summary(clearance_result),
                "precontact": _summary(precontact_result),
                "unauthorized_contact": _summary(unauthorized_contact_result),
                "contact": _summary(contact_result),
                "close": _summary(closed),
                "lift": _summary(lift_result),
            },
            "collision_receipts": collision_receipts,
            "measurements": {
                "object_after_close_xyz": object_after_close,
                "object_after_lift_xyz": object_after_lift,
                "eef_after_lift_xyz": eef_after_lift,
                "object_lift_m": object_lift_m,
                "object_eef_distance_m": object_eef_distance_m,
                "attachment_dims_m": attachment_dims,
            },
            "checks": checks,
            "passed": all(checks.values()),
            "gripper_open_response": _summary(opened),
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
            except Exception as exc:  # noqa: BLE001 - preserve the canary result.
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
    parser.add_argument("--target-category", default="salad_dressing")
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--contact-x-offset-m", type=float, default=0.0)
    parser.add_argument("--contact-y-offset-m", type=float, default=0.0)
    parser.add_argument("--contact-z-offset-m", type=float, default=0.02)
    parser.add_argument("--roll", type=float)
    parser.add_argument("--pitch", type=float)
    parser.add_argument("--yaw", type=float)
    parser.add_argument("--orientation-tolerance-rad", type=float, default=0.1)
    parser.add_argument("--lift-distance-m", type=float, default=0.12)
    parser.add_argument("--tolerance-m", type=float, default=0.004)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument(
        "--output", default="tmp/mink-contact-canary-20260819.json"
    )
    args = parser.parse_args()
    report = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
