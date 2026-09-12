import json
from argparse import Namespace

import pytest

from scripts.replay_recorded_mink_approach import recorded_moves, run


def rows():
    records = [{"event": {"name": "compile_grasp_seed", "phase": "end", "success": True,
        "details": {"outputs": {"compiled_grasp_id": "compiled", "scene_epoch": 0,
                               "target_anchor_world_xyz": [0, 0, 0.05]}}}}]
    for index, role in enumerate(("grasp_clearance", "grasp_contact")):
        pose = {"xyz": [0, 0, 0.2 - index * 0.15], "frame": "world",
                "rotation_matrix": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                "compiled_grasp_id": "compiled", "waypoint_role": role}
        receipt = {"receipt_id": str(index), "classification": "feasible", "target_pose": pose,
                   "best_candidate": {"joint_positions": [0] * 7}}
        records.extend([
            {"event": {"name": "ik_preview_check", "phase": "end", "success": True,
                       "details": {"outputs": {"ik_preview_receipt": receipt}}}},
            {"event": {"name": "move_to", "phase": "start", "parameters": {
                "target_pose": pose, "ik_receipt_id": str(index), "enable_collision_check": True,
                "tolerance": 0.01 if index == 0 else 0.005, "ori_tolerance": 0.1}}},
            {"event": {"name": "move_to", "phase": "end", "details": {"outputs": {
                "motion_summary": {"controller_receipt": {"iteration_budget": 150}}}}}},
        ])
    return records


def test_replay_preserves_seeds_tolerances_and_contact_scope():
    moves = recorded_moves(rows())
    assert [m["tolerance"] for m in moves] == [0.01, 0.005]
    assert all(m["enable_collision_check"] for m in moves)
    assert "contact_authorization" not in moves[0]
    assert moves[1]["contact_authorization"]["target_anchor_world_xyz"] == [0, 0, 0.05]
    bad = rows()
    bad[2]["event"]["parameters"]["enable_collision_check"] = False
    with pytest.raises(ValueError, match="collision checking"):
        recorded_moves(bad)


@pytest.mark.parametrize("outcome", ["pass", "not_reached", "timeout"])
def test_controller_replay_is_bounded_and_closes_after_failure(tmp_path, outcome):
    path = tmp_path / "calls.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows()))
    calls = []
    class Transport:
        def call_tool(self, name, parameters, **kwargs):
            calls.append(name)
            if name == "create_env":
                return {"handle": "fixture", "session_id": "fixture"}
            if name == "move_to":
                if outcome == "timeout":
                    raise TimeoutError("fixture")
                return {"reached_target": outcome == "pass", "collision": {"detected": False},
                        "controller_receipt": {"ik_execution_seed_receipt_id": parameters["ik_execution_seed"]["receipt_id"]}}
            return {"ok": True}
    args = Namespace(url="http://127.0.0.1:18766/sse", tool_calls=str(path), env_id="fixture", seed=0)
    result = run(args, transport=Transport())
    assert result["passed"] is (outcome == "pass")
    assert result["agent_performance_sample"] is False
    assert calls[-1] == "close_env"
    assert calls.count("move_to") == (2 if outcome == "pass" else 1)
    assert "gripper_close" not in calls
    args.url = "https://example.invalid"
    with pytest.raises(ValueError, match="loopback-only"):
        run(args, transport=Transport())
