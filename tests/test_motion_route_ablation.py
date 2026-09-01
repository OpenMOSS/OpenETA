from agent.evals.motion_route_ablation import summarize_motion_route_job


def _event(name, *, success=True, parameters=None, motion=None, response=None):
    outputs = {"motion_summary": motion or {}}
    if response is not None:
        outputs["response"] = response
    return {
        "event": {
            "phase": "end",
            "name": name,
            "success": success,
            "parameters": parameters or {},
            "details": {"outputs": outputs},
        }
    }


def _job(method, *, pair="clear", target=(0.1, -0.32, 0.12)):
    return {
        "job_id": f"{pair}-{method}",
        "source_episode_id": f"{pair}-{method}",
        "repeat_index": 0,
        "episode": {
            "metadata": {
                "ablation_pair_id": pair,
                "route_method": method,
                "target_xyz": list(target),
                "position_tolerance_m": 0.01,
            }
        },
    }


def _model_row(prompt=100, completion=10):
    return {
        "result": {
            "details": {
                "usage": {
                    "prompt_tokens": prompt,
                    "completion_tokens": completion,
                    "cached_tokens": 5,
                    "total_tokens": prompt + completion,
                }
            }
        }
    }


def test_summarize_direct_motion_success_and_adherence():
    rows = [
        _event("ik_preview_check"),
        _event(
            "move_to",
            parameters={"ik_receipt_id": "ik-1"},
            motion={
                "reached_target": True,
                "end": {"xyz": [0.101, -0.319, 0.119]},
                "collision": {"detected": False, "minimum_distance_m": 0.02},
                "steps_executed": 15,
                "stop_reason": "target_reached",
            },
        ),
    ]

    result = summarize_motion_route_job(
        job=_job("direct"),
        final_result={"status": "success", "outcome": {"status": "success"}},
        tool_rows=rows,
        model_rows=[_model_row(), _model_row(50, 5)],
    )

    assert result["method_adherent"] is True
    assert result["instruction_exact"] is True
    assert result["execution_success"] is True
    assert result["selected_waypoint_count"] == 0
    assert result["usage"]["total_tokens"] == 165


def test_summarize_agent_waypoint_motion_uses_receipt_count():
    rows = [
        _event("ik_preview_check", parameters={"target_pose": {"xyz": [0, 0, 0.3]}}),
        _event("ik_preview_check", parameters={"target_pose": {"xyz": [0.1, 0, 0.3]}}),
        _event("ik_preview_check", parameters={"target_pose": {"xyz": [0.1, -0.32, 0.12]}}),
        _event(
            "follow_eef_trajectory",
            parameters={"ik_receipt_ids": ["ik-1", "ik-2", "ik-3"]},
            motion={
                "reached_target": True,
                "end": {"xyz": [0.1, -0.32, 0.12]},
                "collision": {"detected": False},
                "motion_execution_profile": {"condition": "C"},
                "steps_executed": 40,
                "waypoints_requested": 3,
                "waypoints_completed": 3,
                "waypoint_results": [
                    {
                        "reached_target": True,
                        "controller_receipt": {
                            "stable_steps_completed": 3,
                            "ik_execution_seed_receipt_id": "seed-1",
                        },
                    },
                    {
                        "reached_target": True,
                        "controller_receipt": {
                            "stable_steps_completed": 3,
                            "ik_execution_seed_receipt_id": "seed-2",
                        },
                    },
                    {
                        "reached_target": True,
                        "controller_receipt": {
                            "stable_steps_completed": 3,
                            "ik_execution_seed_receipt_id": "seed-3",
                        },
                    },
                ],
                "sequential_route_preview": {
                    "waypoints_previewed": 3,
                    "waypoints_authorized": 3,
                },
            },
        ),
    ]

    result = summarize_motion_route_job(
        job=_job("agent_waypoints"),
        final_result={"status": "success", "outcome": {"status": "success"}},
        tool_rows=rows,
        model_rows=[],
    )

    assert result["method_adherent"] is True
    assert result["execution_success"] is True
    assert result["selected_waypoint_count"] == 3
    assert result["motion_condition"] == "C"
    assert result["stable_waypoint_count"] == 3
    assert result["sequential_waypoints_authorized"] == 3


def test_summarize_reports_extra_tool_without_invalidating_motion_method():
    rows = [
        _event("ik_preview_check"),
        _event("sam3"),
        _event(
            "move_to",
            motion={
                "reached_target": False,
                "end": {"xyz": [0.0, 0.0, 0.0]},
                "collision": {"detected": True},
            },
        ),
    ]

    result = summarize_motion_route_job(
        job=_job("direct"),
        final_result={"status": "fail", "outcome": {"status": "fail"}},
        tool_rows=rows,
        model_rows=[],
    )

    assert result["method_adherent"] is True
    assert result["instruction_exact"] is False
    assert result["execution_success"] is False
    assert result["extra_tool_names"] == ["sam3"]


def test_summarize_recovers_condition_from_full_controller_receipt(tmp_path):
    response_path = tmp_path / "response.json"
    response_path.write_text(
        """{
          "reached_target": false,
          "end": {"xyz": [0.08, -0.31, 0.39]},
          "controller_receipt": {
            "motion_execution_profile": {"condition": "B"}
          },
          "waypoints_requested": 3,
          "waypoints_completed": 1,
          "waypoint_results": [
            {"reached_target": true, "controller_receipt": {"stable_steps_completed": 3}},
            {"reached_target": false, "controller_receipt": {"stable_steps_completed": 0}}
          ]
        }""",
        encoding="utf-8",
    )
    rows = [
        _event("ik_preview_check"),
        _event("ik_preview_check"),
        _event(
            "follow_eef_trajectory",
            parameters={"ik_receipt_ids": ["ik-1", "ik-2"]},
            motion={
                "reached_target": False,
                "end": {"xyz": [0.08, -0.31, 0.39]},
                "controller_receipt": {"stable_arrival_enabled": True},
            },
            response={"response_path": str(response_path)},
        ),
    ]

    result = summarize_motion_route_job(
        job=_job("agent_waypoints"),
        final_result={"status": "fail", "outcome": {"status": "fail"}},
        tool_rows=rows,
        model_rows=[],
    )

    assert result["motion_condition"] == "B"
    assert result["waypoints_requested"] == 3
    assert result["waypoints_completed"] == 1
    assert result["stable_waypoint_count"] == 1
