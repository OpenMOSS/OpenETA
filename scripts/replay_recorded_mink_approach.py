#!/usr/bin/env python3
"""Local-only controller isolation: replay the first two recorded seeded moves.

No planner, perception service, oracle target lookup, grasp, or release is used.
Recorded poses, tolerances, IK joints and compiled contact anchor are unchanged.
This is controller evidence, never an autonomous task-performance sample.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import urlparse

from agent.tools.sim_mcp import SseSimulatorMcpTransport, _extract_orientation_arguments


def recorded_moves(rows: list[dict]) -> list[dict]:
    receipts: dict[str, dict] = {}
    compiled: dict[str, dict] = {}
    moves = []
    pending = None
    for row in rows:
        event = row.get("event", {})
        name, phase = event.get("name"), event.get("phase")
        outputs = event.get("details", {}).get("outputs", {})
        if phase == "end" and name == "compile_grasp_seed" and event.get("success"):
            compiled[outputs["compiled_grasp_id"]] = outputs
        if phase == "end" and name == "ik_preview_check" and event.get("success"):
            receipt = outputs.get("ik_preview_receipt", {})
            receipts[receipt["receipt_id"]] = receipt
        if phase == "start" and name == "move_to":
            pending = event["parameters"]
        if phase != "end" or name != "move_to" or pending is None:
            continue
        pose = pending["target_pose"]
        receipt = receipts[pending["ik_receipt_id"]]
        if receipt.get("classification") != "feasible" or receipt.get("target_pose") != pose:
            raise ValueError("recorded move is not bound to an exact feasible IK target")
        joints = receipt["best_candidate"]["joint_positions"]
        if len(joints) != 7 or pending.get("enable_collision_check") is not True:
            raise ValueError("replay requires seven preview joints and collision checking")
        budget = outputs["motion_summary"]["controller_receipt"]["iteration_budget"]
        if not 1 <= budget <= 150:
            raise ValueError("replay only accepts recorded controller budgets up to 150")
        params = {
            **dict(zip(("x", "y", "z"), pose["xyz"], strict=True)),
            **_extract_orientation_arguments(pending, tool_name="move_to"),
            "tolerance": pending["tolerance"],
            "ori_tolerance": pending["ori_tolerance"],
            "num_steps": budget,
            "enable_collision_check": True,
            "ik_execution_seed": {
                "schema_version": "openeta.ik_execution_seed.v1",
                "receipt_id": "replay-" + receipt["receipt_id"],
                "joint_positions": joints,
                "preview_tolerances": {
                    "position_tolerance_m": pending["tolerance"],
                    "orientation_tolerance_rad": pending["ori_tolerance"],
                },
            },
        }
        if pose.get("waypoint_role") == "grasp_contact":
            geometry = compiled[pose["compiled_grasp_id"]]
            params["contact_authorization"] = {
                "schema_version": "openeta.contact_authorization.v1",
                "compiled_grasp_id": pose["compiled_grasp_id"],
                "waypoint_role": "grasp_contact",
                "target_anchor_world_xyz": geometry["target_anchor_world_xyz"],
                "target_evidence_id": "recorded-controller-isolation",
                "object_scene_epoch": geometry["scene_epoch"],
            }
        elif pose.get("waypoint_role") != "grasp_clearance":
            raise ValueError("only initial grasp clearance/contact may be replayed")
        moves.append(params)
        pending = None
        if len(moves) == 2:
            break
    if len(moves) != 2 or "contact_authorization" in moves[0] or "contact_authorization" not in moves[1]:
        raise ValueError("expected exactly a recorded clearance followed by contact")
    return moves


def run(args, *, transport=None):
    parsed = urlparse(args.url)
    if parsed.hostname not in {"127.0.0.1", "localhost", "::1"} or parsed.username or parsed.password:
        raise ValueError("controller replay is loopback-only")
    moves = recorded_moves([json.loads(line) for line in Path(args.tool_calls).read_text().splitlines() if line.strip()])
    client = transport or SseSimulatorMcpTransport(args.url)
    common = {}
    report = {"schema_version": "openeta.recorded_mink_approach_replay.v1",
              "agent_performance_sample": False, "source": str(args.tool_calls),
              "env_id": args.env_id, "seed": args.seed, "moves": [], "passed": False}
    try:
        created = client.call_tool("create_env", {"env_id": args.env_id, "seed": args.seed,
            "image_width": 128, "image_height": 128, "include_objects": False}, timeout_s=300)
        common = {"handle": created["handle"], "session_id": created["session_id"]}
        client.call_tool("reset_env", {**common, "seed": args.seed}, timeout_s=300)
        for params in moves:
            result = client.call_tool("move_to", {**common, **params}, timeout_s=300)
            summary = {k: v for k, v in result.items() if k not in {"observation", "cameras"}}
            report["moves"].append({"request": params, "result": summary})
            receipt = result.get("controller_receipt", {})
            passed = (result.get("reached_target") is True
                      and result.get("collision", {}).get("detected") is False
                      and receipt.get("ik_execution_seed_receipt_id") == params["ik_execution_seed"]["receipt_id"])
            if not passed:
                break
        report["passed"] = len(report["moves"]) == 2 and passed
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)[:500]}
    finally:
        if common:
            try:
                report["cleanup"] = client.call_tool("close_env", common, timeout_s=30)
            except Exception as exc:
                report["cleanup"] = {"ok": False, "error_type": type(exc).__name__}
        else:
            report["cleanup"] = {"ok": False, "reason": "no_created_handle"}
        report["passed"] = report["passed"] and report["cleanup"].get("ok") is True
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool-calls", required=True)
    parser.add_argument("--url", default="http://127.0.0.1:18766/sse")
    parser.add_argument("--env-id", default="openeta/libero_libero_object_task0-v0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = run(args)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"passed": result["passed"], "output": args.output, "cleanup": result["cleanup"]}))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
