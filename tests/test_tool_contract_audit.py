from __future__ import annotations

import json
from pathlib import Path

from agent.evals.tool_contract_audit import audit_tool_event_file


def _write_events(path: Path, events: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )


def test_tool_contract_audit_finds_repeated_view_and_missing_motion_receipts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tool_calls.jsonl"
    sam_details = {
        "operational_success": True,
        "semantic_outcome": "detections_available",
        "outputs": {"source_packet_id": "packet-1"},
        "diagnostics": [],
        "recovery_options": [],
    }
    _write_events(
        path,
        [
            {
                "seq": 1,
                "event": {
                    "phase": "end",
                    "name": "sam3",
                    "effect": "read_only",
                    "success": True,
                    "parameters": {
                        "source_packet_id": "packet-1",
                        "camera_frame_id": "wrist",
                        "prompt": "milk",
                    },
                    "details": sam_details,
                },
            },
            {
                "seq": 2,
                "event": {
                    "phase": "end",
                    "name": "sam3",
                    "effect": "read_only",
                    "success": True,
                    "parameters": {
                        "source_packet_id": "packet-1",
                        "camera_frame_id": "wrist",
                        "prompt": "milk",
                    },
                    "details": sam_details,
                },
            },
            {
                "seq": 3,
                "event": {
                    "phase": "end",
                    "name": "move_to",
                    "effect": "world_mutating",
                    "success": True,
                    "parameters": {"target_pose": {"xyz": [0.1, 0.2, 0.3]}},
                    "details": {
                        "operational_success": True,
                        "semantic_outcome": "success",
                        "outputs": {},
                        "diagnostics": [],
                        "recovery_options": [],
                    },
                },
            },
        ],
    )

    report = audit_tool_event_file(path)

    assert report["tool_call_count"] == 3
    assert report["issue_code_counts"]["repeated_perception_on_unchanged_view"] == 1
    assert report["issue_code_counts"]["motion_without_pose_feedback"] == 1
    assert report["issue_code_counts"]["motion_without_collision_coverage"] == 1
    assert report["issue_code_counts"][
        "motion_without_exact_typed_ik_handoff"
    ] == 1
    assert report["issue_code_counts"]["mutation_without_environment_receipt"] == 1
    assert report["contract_pass"] is False
    assert report["quality_checks"]["motions_have_pose_and_controller_receipts"] is False


def test_tool_contract_audit_treats_new_packet_ids_without_mutation_as_same_view(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tool_calls.jsonl"
    events = []
    for seq, packet_id in ((2, "packet-1"), (8, "packet-2")):
        events.append(
            {
                "seq": seq,
                "event": {
                    "phase": "end",
                    "name": "sam3",
                    "effect": "read_only",
                    "success": True,
                    "parameters": {
                        "source_packet_id": packet_id,
                        "camera_frame_id": "wrist",
                        "prompt": "salad dressing bottle",
                        **({"mode": "text"} if seq == 2 else {}),
                    },
                    "details": {
                        "operational_success": True,
                        "semantic_outcome": "detections_available",
                        "outputs": {"source_packet_id": packet_id},
                        "diagnostics": [],
                        "recovery_options": [],
                    },
                },
            }
        )
    _write_events(path, events)

    report = audit_tool_event_file(path)

    assert report["issue_code_counts"]["repeated_perception_on_unchanged_view"] == 1
    issue = next(
        item
        for item in report["issues"]
        if item["code"] == "repeated_perception_on_unchanged_view"
    )
    assert issue["evidence"]["previous_seq"] == 2


def test_tool_contract_audit_accepts_complete_motion_contract(tmp_path: Path) -> None:
    path = tmp_path / "tool_calls.jsonl"
    _write_events(
        path,
        [
            {
                "seq": 1,
                "event": {
                    "phase": "end",
                    "name": "ik_preview_check",
                    "effect": "read_only",
                    "success": True,
                    "parameters": {"target_pose": {"xyz": [0.1, 0.2, 0.3]}},
                    "details": {
                        "operational_success": True,
                        "semantic_outcome": "ik_feasible",
                        "outputs": {
                            "ik_preview_receipt": {
                                "receipt_id": "ik-complete",
                                "classification": "feasible",
                            },
                            "collision_coverage": {"coverage_complete": True},
                        },
                        "diagnostics": [],
                        "recovery_options": [],
                    },
                },
            },
            {
                "seq": 2,
                "event": {
                    "phase": "end",
                    "name": "move_to",
                    "effect": "world_mutating",
                    "success": True,
                    "parameters": {
                        "target_pose": {"xyz": [0.1, 0.2, 0.3]},
                        "ik_receipt_id": "ik-complete",
                    },
                    "details": {
                        "operational_success": True,
                        "semantic_outcome": "success",
                        "environment_receipt": {"receipt_id": "receipt-1"},
                        "outputs": {
                            "pose_feedback": {"reached_target": True},
                            "motion_summary": {
                                "reached_target": True,
                                "steps_executed": 12,
                                "stop_reason": "target_reached",
                                "controller_receipt": {
                                    "schema_version": "openeta.controller_execution_receipt.v1",
                                    "controller_id": "robosuite.osc_pose",
                                    "command_interface": "normalized_cartesian_delta_pose",
                                    "goal_executor": "openeta.outer_closed_loop_cartesian.v1",
                                    "execution_location": "mcp_server",
                                    "orientation_policy": "preserve_current",
                                    "iteration_budget": 100,
                                    "steps_executed": 12,
                                    "stop_reason": "target_reached",
                                    "reached_target": True,
                                },
                            },
                            "collision_coverage": {
                                "coverage_complete": True,
                                "coverage_status": "trajectory_and_world",
                            },
                        },
                        "diagnostics": [],
                        "recovery_options": [],
                    },
                },
            }
        ],
    )

    report = audit_tool_event_file(path)

    assert report["issue_count"] == 0
    assert report["contract_pass"] is True
    assert all(report["quality_checks"].values())


def test_tool_contract_audit_rejects_motion_reference_from_failed_ik(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tool_calls.jsonl"
    _write_events(
        path,
        [
            {
                "seq": 1,
                "event": {
                    "phase": "end",
                    "name": "ik_preview_check",
                    "effect": "read_only",
                    "success": False,
                    "content": (
                        "IK preview unreachable (full_pose_infeasible). "
                        "Execution reference: ik_receipt_id=ik-rejected; pass only "
                        "this id to move_to."
                    ),
                    "parameters": {
                        "target_pose": {
                            "xyz": [0.1, 0.2, 0.3],
                            "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
                        }
                    },
                    "details": {
                        "operational_success": False,
                        "semantic_outcome": "ik_repairable",
                        "outputs": {
                            "ik_preview_receipt": {
                                "receipt_id": "ik-rejected",
                                "classification": "repairable",
                                "reason_code": "full_pose_infeasible",
                            },
                            "motion_execution_ref": {
                                "tool": "move_to",
                                "ik_receipt_id": "ik-rejected",
                            },
                            "collision_coverage": {"coverage_complete": False},
                        },
                        "diagnostics": [{"code": "full_pose_infeasible"}],
                        "recovery_options": [
                            {
                                "action": "preview_modified_pose",
                                "reason": "change the pose before retrying",
                            }
                        ],
                    },
                },
            }
        ],
    )

    report = audit_tool_event_file(path)

    assert report["issue_code_counts"][
        "nonexecutable_ik_exposes_motion_reference"
    ] == 1
    assert report["quality_checks"][
        "ik_receipts_expose_consistent_execution_authority"
    ] is False
    assert report["contract_pass"] is False


def test_tool_contract_audit_rejects_deferred_ik_without_collision_owner(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tool_calls.jsonl"
    _write_events(
        path,
        [
            {
                "seq": 1,
                "event": {
                    "phase": "end",
                    "name": "ik_preview_check",
                    "effect": "read_only",
                    "success": True,
                    "content": (
                        "IK preview endpoint_collision_check_unavailable. "
                        "Execution reference: ik_receipt_id=ik-deferred."
                    ),
                    "parameters": {
                        "target_pose": {
                            "xyz": [0.1, 0.2, 0.3],
                            "orientation_policy": "preserve_current_orientation",
                        }
                    },
                    "details": {
                        "operational_success": True,
                        "semantic_outcome": "ik_inconclusive",
                        "outputs": {
                            "ik_preview_receipt": {
                                "receipt_id": "ik-deferred",
                                "classification": (
                                    "kinematically_feasible_collision_deferred"
                                ),
                                "reason_code": (
                                    "endpoint_collision_check_unavailable"
                                ),
                                "motion_collision_delegation": {
                                    "applicable": True,
                                    "available_for_matching_move": False,
                                },
                            },
                            "response": {
                                "motion_execution_ref": {
                                    "tool": "move_to",
                                    "ik_receipt_id": "ik-deferred",
                                }
                            },
                            "collision_coverage": {"coverage_complete": False},
                        },
                        "diagnostics": [
                            {"code": "endpoint_collision_check_unavailable"}
                        ],
                        "recovery_options": [
                            {
                                "action": "use_collision_owning_controller",
                                "reason": "motion must own trajectory collision checks",
                            }
                        ],
                    },
                },
            }
        ],
    )

    report = audit_tool_event_file(path)

    assert report["issue_code_counts"][
        "nonexecutable_ik_exposes_motion_reference"
    ] == 1
    assert report["quality_checks"][
        "ik_receipts_expose_consistent_execution_authority"
    ] is False
    assert report["contract_pass"] is False


def test_tool_contract_audit_flags_failed_motion_after_fragile_ik(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tool_calls.jsonl"
    target = {"frame": "world", "xyz": [0.1, 0.2, 0.3]}
    _write_events(
        path,
        [
            {
                "seq": 1,
                "event": {
                    "phase": "end",
                    "name": "ik_preview_check",
                    "effect": "read_only",
                    "success": True,
                    "parameters": {"target_pose": target},
                    "details": {
                        "operational_success": True,
                        "semantic_outcome": "ik_feasible",
                        "outputs": {
                            "ik_preview_receipt": {
                                "receipt_id": "ik-fragile",
                                "classification": "feasible",
                                "target_pose": target,
                                "best_candidate": {"joint_margin_min_rad": 0.081},
                            },
                            "collision_coverage": {"coverage_complete": True},
                        },
                        "diagnostics": [],
                        "recovery_options": [],
                    },
                },
            },
            {
                "seq": 2,
                "event": {
                    "phase": "end",
                    "name": "move_to",
                    "effect": "world_mutating",
                    "success": False,
                    "parameters": {
                        "target_pose": target,
                        "ik_receipt_id": "ik-fragile",
                    },
                    "details": {
                        "operational_success": False,
                        "semantic_outcome": "target_not_reached",
                        "outputs": {
                            "motion_summary": {
                                "reached_target": False,
                                "steps_executed": 5,
                            }
                        },
                        "diagnostics": [{"code": "target_not_reached"}],
                        "recovery_options": [
                            {"action": "select_another_grasp_candidate"}
                        ],
                    },
                },
            },
        ],
    )

    report = audit_tool_event_file(path)

    assert report["issue_code_counts"][
        "failed_motion_after_execution_fragile_ik"
    ] == 1
    issue = next(
        item
        for item in report["issues"]
        if item["code"] == "failed_motion_after_execution_fragile_ik"
    )
    assert issue["severity"] == "warning"
    assert issue["evidence"]["risk_level"] == "elevated"
    assert issue["evidence"]["selected_joint_margin_rad"] == 0.081
    assert report["quality_checks"][
        "failed_motions_avoid_execution_fragile_ik"
    ] is False


def test_tool_contract_audit_requires_executable_anyplace_handoff(tmp_path: Path) -> None:
    path = tmp_path / "tool_calls.jsonl"
    _write_events(
        path,
        [
            {
                "seq": 1,
                "event": {
                    "phase": "end",
                    "name": "anyplace",
                    "effect": "planning",
                    "success": True,
                    "parameters": {"bundle_id": "anyplace:bundle"},
                    "details": {
                        "operational_success": True,
                        "semantic_outcome": "success",
                        "outputs": {"placement_candidates": []},
                        "diagnostics": [],
                        "recovery_options": [],
                    },
                },
            },
            {
                "seq": 2,
                "event": {
                    "phase": "end",
                    "name": "camera_pose_to_world",
                    "effect": "read_only",
                    "success": True,
                    "parameters": {
                        "camera_pose": {
                            "id": "place_grasp_000",
                            "source_grasp_id": "grasp_000",
                        },
                        "camera_extrinsics": {"pos": [0, 0, 0]},
                    },
                    "details": {
                        "operational_success": True,
                        "semantic_outcome": "success",
                        "outputs": {},
                        "diagnostics": [],
                        "recovery_options": [],
                    },
                },
            },
        ],
    )

    report = audit_tool_event_file(path)

    assert report["issue_code_counts"][
        "anyplace_missing_executable_placement_handoff"
    ] == 1
    assert report["issue_code_counts"][
        "anyplace_pose_transformed_without_short_id_resolution"
    ] == 1


def test_tool_contract_audit_accepts_reference_bound_anyplace_transform(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tool_calls.jsonl"
    result_id = "anyplace-result:abc"
    _write_events(
        path,
        [
            {
                "seq": 1,
                "event": {
                    "phase": "end",
                    "name": "anyplace",
                    "effect": "planning",
                    "success": True,
                    "parameters": {"bundle_id": "anyplace:bundle"},
                    "details": {
                        "operational_success": True,
                        "semantic_outcome": "success",
                        "outputs": {
                            "result_id": result_id,
                            "source_packet_id": "obs-0001",
                            "camera_frame_id": "agentview",
                            "camera_pose_to_world_handoff": {
                                "tool": "camera_pose_to_world",
                                "placement_result_id": result_id,
                                "required_parameters": [
                                    "placement_result_id",
                                    "candidate_id",
                                ],
                            },
                        },
                        "diagnostics": [],
                        "recovery_options": [],
                    },
                },
            },
            {
                "seq": 2,
                "event": {
                    "phase": "end",
                    "name": "camera_pose_to_world",
                    "effect": "read_only",
                    "success": True,
                    "parameters": {
                        "placement_result_id": result_id,
                        "candidate_id": "placement_002",
                        "camera_pose": {
                            "id": "place_grasp_002",
                            "source_grasp_id": "grasp_000",
                        },
                        "camera_extrinsics": {"pos": [0, 0, 0]},
                    },
                    "details": {
                        "operational_success": True,
                        "semantic_outcome": "success",
                        "outputs": {
                            "placement_result_id": result_id,
                            "candidate_id": "placement_002",
                            "placement_reference": {
                                "schema_version": (
                                    "openeta.placement_world_reference.v1"
                                ),
                                "execution_authorized": False,
                                "placement_result_id": result_id,
                                "candidate_id": "placement_002",
                            },
                        },
                        "diagnostics": [],
                        "recovery_options": [],
                    },
                },
            },
        ],
    )

    report = audit_tool_event_file(path)

    assert "anyplace_missing_executable_placement_handoff" not in report[
        "issue_code_counts"
    ]
    assert "anyplace_pose_transformed_without_short_id_resolution" not in report[
        "issue_code_counts"
    ]
    assert "placement_transform_identity_not_preserved" not in report[
        "issue_code_counts"
    ]


def test_tool_contract_audit_accepts_deferred_ik_with_complete_motion_collision(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tool_calls.jsonl"
    target = {"xyz": [0.1, 0.2, 0.3]}
    _write_events(
        path,
        [
            {
                "seq": 1,
                "event": {
                    "phase": "end",
                    "name": "ik_preview_check",
                    "effect": "read_only",
                    "success": True,
                    "parameters": {
                        "target_pose": target,
                        "check_endpoint_collision": True,
                    },
                    "details": {
                        "operational_success": True,
                        "semantic_outcome": (
                            "ik_kinematically_feasible_collision_deferred"
                        ),
                        "outputs": {
                            "ik_preview_receipt": {
                                "classification": "inconclusive",
                                "reason_code": (
                                    "endpoint_collision_check_unavailable"
                                ),
                                "best_candidate": {"joint_positions": [0.0] * 7},
                            },
                            "collision_coverage": {"coverage_complete": False},
                        },
                        "diagnostics": [],
                        "recovery_options": [
                            {
                                "action": (
                                    "delegate_collision_to_verified_motion_controller"
                                )
                            }
                        ],
                    },
                },
            },
            {
                "seq": 2,
                "event": {
                    "phase": "end",
                    "name": "move_to",
                    "effect": "world_mutating",
                    "success": True,
                    "parameters": {
                        "target_pose": target,
                        "enable_collision_check": True,
                    },
                    "details": {
                        "operational_success": True,
                        "semantic_outcome": "success",
                        "environment_receipt": {"receipt_id": "receipt-1"},
                        "outputs": {
                            "pose_feedback": {"reached_target": True},
                            "motion_summary": {
                                "reached_target": True,
                                "steps_executed": 12,
                                "stop_reason": "target_reached",
                                "controller_receipt": {
                                    "schema_version": (
                                        "openeta.controller_execution_receipt.v1"
                                    ),
                                    "controller_id": "mink.robosuite_joint_velocity",
                                    "command_interface": "joint_velocity",
                                    "goal_executor": "openeta.worker_mink_goal.v1",
                                    "execution_location": "bench_worker",
                                    "orientation_policy": "preserve_current",
                                    "iteration_budget": 100,
                                    "steps_executed": 12,
                                    "stop_reason": "target_reached",
                                    "reached_target": True,
                                },
                            },
                            "collision_coverage": {
                                "coverage_complete": True,
                                "coverage_status": "trajectory_and_world",
                            },
                        },
                        "diagnostics": [],
                        "recovery_options": [],
                    },
                },
            },
        ],
    )

    report = audit_tool_event_file(path)

    assert "motion_without_matching_feasible_ik_receipt" not in report[
        "issue_code_counts"
    ]


def test_tool_contract_audit_flags_redundant_no_collision_ik_after_deferred(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tool_calls.jsonl"
    target = {"xyz": [0.1, 0.2, 0.3]}
    _write_events(
        path,
        [
            {
                "seq": 1,
                "event": {
                    "phase": "end",
                    "name": "ik_preview_check",
                    "effect": "read_only",
                    "success": True,
                    "parameters": {
                        "target_pose": target,
                        "check_endpoint_collision": True,
                    },
                    "details": {
                        "operational_success": True,
                        "semantic_outcome": (
                            "ik_kinematically_feasible_collision_deferred"
                        ),
                        "outputs": {
                            "ik_preview_receipt": {
                                "classification": (
                                    "kinematically_feasible_collision_deferred"
                                ),
                                "reason_code": (
                                    "endpoint_collision_check_unavailable"
                                ),
                            },
                            "collision_coverage": {"coverage_complete": False},
                        },
                        "diagnostics": [],
                        "recovery_options": [
                            {
                                "action": (
                                    "delegate_collision_to_verified_motion_controller"
                                )
                            }
                        ],
                    },
                },
            },
            {
                "seq": 2,
                "event": {
                    "phase": "end",
                    "name": "ik_preview_check",
                    "effect": "read_only",
                    "success": True,
                    "parameters": {
                        "target_pose": target,
                        "check_endpoint_collision": False,
                    },
                    "details": {
                        "operational_success": True,
                        "semantic_outcome": "ik_feasible",
                        "outputs": {
                            "ik_preview_receipt": {"classification": "feasible"},
                            "collision_coverage": {"coverage_complete": False},
                        },
                        "diagnostics": [],
                        "recovery_options": [],
                    },
                },
            },
        ],
    )

    report = audit_tool_event_file(path)

    assert report["issue_code_counts"][
        "redundant_kinematics_only_ik_after_deferred_solution"
    ] == 1


def test_tool_contract_audit_detects_inconsistent_controller_receipt(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tool_calls.jsonl"
    _write_events(
        path,
        [
            {
                "seq": 2,
                "event": {
                    "phase": "end",
                    "name": "move_to",
                    "effect": "world_mutating",
                    "success": True,
                    "parameters": {"target_pose": {"xyz": [0.1, 0.2, 0.3]}},
                    "details": {
                        "operational_success": False,
                        "semantic_outcome": "target_not_reached",
                        "environment_receipt": {"receipt_id": "receipt-1"},
                        "outputs": {
                            "pose_feedback": {"reached_target": False},
                            "collision_coverage": {"coverage_complete": True},
                            "motion_summary": {
                                "reached_target": False,
                                "steps_executed": 100,
                                "stop_reason": "iteration_limit",
                                "controller_receipt": {
                                    "schema_version": "openeta.controller_execution_receipt.v1",
                                    "controller_id": "robosuite.osc_pose",
                                    "command_interface": "normalized_cartesian_delta_pose",
                                    "goal_executor": "openeta.outer_closed_loop_cartesian.v1",
                                    "execution_location": "mcp_server",
                                    "orientation_policy": "preserve_current",
                                    "iteration_budget": 100,
                                    "steps_executed": 0,
                                    "stop_reason": "target_reached",
                                    "reached_target": True,
                                },
                            },
                        },
                        "diagnostics": [{"code": "target_not_reached"}],
                        "recovery_options": [{"action": "replan_from_actual_pose"}],
                    },
                },
            }
        ],
    )

    report = audit_tool_event_file(path)

    issue = next(
        item
        for item in report["issues"]
        if item["code"] == "controller_execution_receipt_inconsistent"
    )
    assert set(issue["evidence"]["mismatches"]) == {
        "steps_executed",
        "stop_reason",
        "reached_target",
    }


def test_tool_contract_audit_rejects_target_miss_marked_operational_success(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tool_calls.jsonl"
    _write_events(
        path,
        [
            {
                "seq": 1,
                "event": {
                    "phase": "end",
                    "name": "move_to",
                    "effect": "world_mutating",
                    "success": True,
                    "details": {
                        "operational_success": True,
                        "semantic_outcome": "target_not_reached",
                        "environment_receipt": {"receipt_id": "receipt-1"},
                        "outputs": {
                            "pose_feedback": {"reached_target": False},
                            "collision_coverage": {"coverage_complete": True},
                        },
                        "diagnostics": [{"code": "target_not_reached"}],
                        "recovery_options": [{"action": "replan_from_actual_pose"}],
                    },
                },
            }
        ],
    )

    report = audit_tool_event_file(path)

    assert report["issue_code_counts"][
        "target_not_reached_marked_operational_success"
    ] == 1


def test_tool_contract_audit_detects_packet_churn_planning_loop(tmp_path: Path) -> None:
    path = tmp_path / "tool_calls.jsonl"
    events = []
    for seq, packet_id in ((2, "packet-1"), (4, "packet-2")):
        events.append(
            {
                "seq": seq,
                "event": {
                    "phase": "end",
                    "name": "propose_wrist_viewpoints",
                    "effect": "read_only",
                    "success": True,
                    "parameters": {
                        "compiled_grasp_id": "compiled-1",
                        "camera_frame_id": "wrist",
                        "source_packet_id": packet_id,
                    },
                    "details": {
                        "operational_success": True,
                        "semantic_outcome": "completed",
                        "outputs": {"proposal_id": "wrist_viewpoint:stable"},
                        "diagnostics": [],
                        "recovery_options": [],
                    },
                },
            }
        )
    _write_events(path, events)

    report = audit_tool_event_file(path)

    assert report["issue_code_counts"]["repeated_equivalent_read_only_call"] == 1


def test_tool_contract_audit_requires_exact_viewpoint_ik_binding(tmp_path: Path) -> None:
    path = tmp_path / "tool_calls.jsonl"
    _write_events(
        path,
        [
            {
                "seq": 2,
                "event": {
                    "phase": "end",
                    "name": "propose_wrist_viewpoints",
                    "effect": "read_only",
                    "success": True,
                    "parameters": {"compiled_grasp_id": "compiled-1"},
                    "details": {
                        "operational_success": True,
                        "semantic_outcome": "completed",
                        "outputs": {
                            "proposal_id": "wrist_viewpoint:one",
                            "candidates": [
                                {
                                    "candidate_id": "wrist_view_00",
                                    "target_pose": {
                                        "xyz": [0.1, 0.2, 0.3],
                                        "rotation_matrix": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                                    },
                                }
                            ],
                        },
                        "diagnostics": [],
                        "recovery_options": [],
                    },
                },
            },
            {
                "seq": 4,
                "event": {
                    "phase": "end",
                    "name": "ik_preview_check",
                    "effect": "read_only",
                    "success": True,
                    "parameters": {"target_pose": {"xyz": [0.1, 0.2, 0.3]}},
                    "details": {
                        "operational_success": True,
                        "semantic_outcome": "ik_feasible",
                        "outputs": {"collision_coverage": {"coverage_complete": True}},
                        "diagnostics": [],
                        "recovery_options": [],
                    },
                },
            },
        ],
    )

    report = audit_tool_event_file(path)

    assert report["issue_code_counts"][
        "viewpoint_ik_without_exact_full_pose_binding"
    ] == 1


def test_tool_contract_audit_expires_viewpoint_attribution_after_unrelated_tool(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tool_calls.jsonl"
    _write_events(
        path,
        [
            {
                "seq": 2,
                "event": {
                    "phase": "end",
                    "name": "propose_wrist_viewpoints",
                    "effect": "read_only",
                    "success": True,
                    "parameters": {"compiled_grasp_id": "compiled-1"},
                    "details": {
                        "operational_success": True,
                        "semantic_outcome": "completed",
                        "outputs": {
                            "proposal_id": "wrist_viewpoint:one",
                            "candidates": [
                                {
                                    "candidate_id": "wrist_view_00",
                                    "target_pose": {
                                        "xyz": [0.1, 0.2, 0.3],
                                        "rotation_matrix": [
                                            [1, 0, 0],
                                            [0, -1, 0],
                                            [0, 0, -1],
                                        ],
                                    },
                                }
                            ],
                        },
                        "diagnostics": [],
                        "recovery_options": [],
                    },
                },
            },
            {
                "seq": 4,
                "event": {
                    "phase": "end",
                    "name": "compile_grasp_seed",
                    "effect": "read_only",
                    "success": True,
                    "parameters": {
                        "grasp_result_id": "gpe-1",
                        "candidate_id": "gpe-1-000",
                    },
                    "details": {
                        "operational_success": True,
                        "semantic_outcome": "completed",
                        "outputs": {},
                        "diagnostics": [],
                        "recovery_options": [],
                    },
                },
            },
            {
                "seq": 6,
                "event": {
                    "phase": "end",
                    "name": "ik_preview_check",
                    "effect": "read_only",
                    "success": True,
                    "parameters": {
                        "target_pose": {
                            "xyz": [0.4, 0.5, 0.6],
                            "rotation_matrix": [
                                [0, 1, 0],
                                [1, 0, 0],
                                [0, 0, -1],
                            ],
                        }
                    },
                    "details": {
                        "operational_success": True,
                        "semantic_outcome": "ik_feasible",
                        "outputs": {"collision_coverage": {"coverage_complete": True}},
                        "diagnostics": [],
                        "recovery_options": [],
                    },
                },
            },
        ],
    )

    report = audit_tool_event_file(path)

    assert "viewpoint_ik_without_exact_full_pose_binding" not in report[
        "issue_code_counts"
    ]


def test_tool_contract_audit_accepts_output_level_failure_diagnostics(tmp_path: Path) -> None:
    path = tmp_path / "tool_calls.jsonl"
    _write_events(
        path,
        [
            {
                "seq": 2,
                "event": {
                    "phase": "end",
                    "name": "sam3",
                    "effect": "read_only",
                    "success": False,
                    "parameters": {"source_packet_id": "bad-packet"},
                    "details": {
                        "operational_success": False,
                        "semantic_outcome": "operational_failure",
                        "outputs": {
                            "reason": "unknown_source_packet_id",
                            "diagnostics": [
                                {
                                    "code": "unknown_source_packet_id",
                                    "message": "copy one of recent_source_packets",
                                }
                            ],
                        },
                        "diagnostics": [],
                        "recovery_options": [{"action": "inspect_diagnostics"}],
                    },
                },
            }
        ],
    )

    report = audit_tool_event_file(path)

    assert "failed_without_diagnostics" not in report["issue_code_counts"]
    assert report["issue_code_counts"]["failure_without_actionable_recovery"] == 1
    assert report["issue_code_counts"]["agent_visible_reference_unresolvable"] == 1


def test_tool_contract_audit_accepts_output_level_actionable_recovery(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tool_calls.jsonl"
    _write_events(
        path,
        [
            {
                "seq": 2,
                "event": {
                    "phase": "end",
                    "name": "retrieve_asset_reference",
                    "effect": "read_only",
                    "success": False,
                    "details": {
                        "operational_success": False,
                        "semantic_outcome": "reference_service_unavailable",
                        "outputs": {
                            "diagnostics": [{"code": "object_memory_retrieval_failed"}],
                            "recovery_options": [
                                {
                                    "action": "continue_with_current_visual_evidence",
                                    "reason": "the optional service does not block vision",
                                }
                            ],
                        },
                        "diagnostics": [],
                        "recovery_options": [{"action": "inspect_diagnostics"}],
                    },
                },
            }
        ],
    )

    report = audit_tool_event_file(path)

    assert "failure_without_actionable_recovery" not in report["issue_code_counts"]
    assert "semantic_failure_without_recovery" not in report["issue_code_counts"]


def test_tool_contract_audit_detects_opaque_controller_failure(tmp_path: Path) -> None:
    path = tmp_path / "tool_calls.jsonl"
    _write_events(
        path,
        [
            {
                "seq": 4,
                "event": {
                    "phase": "end",
                    "name": "move_to",
                    "effect": "world_mutating",
                    "success": False,
                    "content": "Worker-local controller goal failed: AssertionError: ",
                    "details": {
                        "operational_success": False,
                        "semantic_outcome": "operational_failure",
                        "outputs": {
                            "motion_summary": {
                                "reached_target": False,
                                "steps_executed": 0,
                                "stop_reason": "controller_error",
                            }
                        },
                        "diagnostics": [
                            {
                                "code": "simulator_mcp_target_not_reached",
                                "message": "Simulator motion did not reach the requested target.",
                            }
                        ],
                        "recovery_options": [{"action": "inspect_diagnostics"}],
                    },
                },
            }
        ],
    )

    report = audit_tool_event_file(path)

    assert report["issue_code_counts"]["opaque_controller_failure"] == 1
    assert report["issue_code_counts"]["failure_without_actionable_recovery"] == 1


def test_tool_contract_audit_marks_zero_step_motion_as_noop(tmp_path: Path) -> None:
    path = tmp_path / "tool_calls.jsonl"
    target = {"xyz": [0.1, 0.2, 0.3]}
    _write_events(
        path,
        [
            {
                "seq": 1,
                "event": {
                    "phase": "end",
                    "name": "ik_preview_check",
                    "effect": "read_only",
                    "success": True,
                    "parameters": {"target_pose": target},
                    "details": {
                        "operational_success": True,
                        "semantic_outcome": "ik_feasible",
                        "outputs": {
                            "ik_preview_receipt": {"classification": "feasible"},
                            "collision_coverage": {"coverage_complete": True},
                        },
                        "diagnostics": [],
                        "recovery_options": [],
                    },
                },
            },
            {
                "seq": 2,
                "event": {
                    "phase": "end",
                    "name": "move_to",
                    "effect": "world_mutating",
                    "success": True,
                    "parameters": {"target_pose": target},
                    "details": {
                        "operational_success": True,
                        "semantic_outcome": "mutation_acknowledged",
                        "environment_receipt": {"receipt_id": "receipt-2"},
                        "outputs": {
                            "pose_feedback": {"reached_target": True},
                            "motion_summary": {
                                "reached_target": True,
                                "steps_executed": 0,
                                "stop_reason": "target_reached",
                                "start": {"xyz": [0.09, 0.2, 0.3]},
                                "end": {"xyz": [0.09, 0.2, 0.3]},
                                "controller_receipt": {
                                    "schema_version": "openeta.controller_execution_receipt.v1",
                                    "controller_id": "mink.robosuite_joint_velocity",
                                    "command_interface": "joint_velocity",
                                    "goal_executor": "openeta.worker_mink_goal.v1",
                                    "execution_location": "bench_worker",
                                    "orientation_policy": "preserve_current",
                                    "iteration_budget": 100,
                                    "steps_executed": 0,
                                    "stop_reason": "target_reached",
                                    "reached_target": True,
                                },
                            },
                            "collision_coverage": {"coverage_complete": True},
                        },
                        "diagnostics": [],
                        "recovery_options": [],
                    },
                },
            },
        ],
    )

    report = audit_tool_event_file(path)

    assert report["issue_code_counts"]["motion_acknowledged_without_state_change"] == 1
    assert "motion_without_matching_feasible_ik_receipt" not in report["issue_code_counts"]

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[1]["event"]["details"][
        "semantic_outcome"
    ] = "target_already_within_tolerance"
    rows[1]["event"]["details"]["recovery_options"] = [
        {"action": "propose_materially_distinct_checked_endpoint"}
    ]
    _write_events(path, rows)

    corrected = audit_tool_event_file(path)

    assert "motion_acknowledged_without_state_change" not in corrected[
        "issue_code_counts"
    ]

    rows[1]["event"]["details"]["operational_success"] = False
    rows[1]["event"]["details"]["semantic_outcome"] = "target_not_reached"
    rows[1]["event"]["details"]["outputs"]["pose_feedback"][
        "reached_target"
    ] = False
    rows[1]["event"]["details"]["outputs"]["motion_summary"].update(
        {
            "reached_target": False,
            "stop_reason": "collision_detected",
        }
    )
    _write_events(path, rows)

    collision_stop = audit_tool_event_file(path)

    assert "motion_acknowledged_without_state_change" not in collision_stop[
        "issue_code_counts"
    ]


def test_tool_contract_audit_accepts_machine_noise_between_ik_and_move(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tool_calls.jsonl"
    preview_target = {
        "xyz": [-0.180802723606, -0.268571997861, 0.264845266286],
        "rotation_matrix": [
            [-0.078173076226, 0.912272530112, 0.402054475895],
            [0.989011708475, 0.020205899029, 0.146449808532],
            [0.125478270196, 0.409085026859, -0.903827792236],
        ],
    }
    move_target = json.loads(json.dumps(preview_target))
    move_target["rotation_matrix"][1][2] = 0.146449806626
    _write_events(
        path,
        [
            {
                "seq": 1,
                "event": {
                    "phase": "end",
                    "name": "ik_preview_check",
                    "effect": "read_only",
                    "success": True,
                    "parameters": {
                        "target_pose": preview_target,
                        "check_endpoint_collision": True,
                    },
                    "details": {
                        "operational_success": True,
                        "semantic_outcome": (
                            "ik_kinematically_feasible_collision_deferred"
                        ),
                        "outputs": {
                            "ik_preview_receipt": {
                                "classification": (
                                    "kinematically_feasible_collision_deferred"
                                ),
                                "reason_code": "endpoint_collision_check_unavailable",
                                "orientation_policy": "explicit_orientation",
                                "target_pose": preview_target,
                            },
                            "collision_coverage": {"coverage_complete": False},
                        },
                        "diagnostics": [],
                        "recovery_options": [],
                    },
                },
            },
            {
                "seq": 2,
                "event": {
                    "phase": "end",
                    "name": "move_to",
                    "effect": "world_mutating",
                    "success": True,
                    "parameters": {
                        "target_pose": move_target,
                        "enable_collision_check": True,
                    },
                    "details": {
                        "operational_success": True,
                        "semantic_outcome": "mutation_acknowledged",
                        "environment_receipt": {"receipt_id": "receipt-2"},
                        "outputs": {
                            "pose_feedback": {"reached_target": True},
                            "motion_summary": {
                                "reached_target": True,
                                "steps_executed": 5,
                                "stop_reason": "target_reached",
                                "controller_receipt": {
                                    "schema_version": (
                                        "openeta.controller_execution_receipt.v1"
                                    ),
                                    "controller_id": "mink.robosuite_joint_velocity",
                                    "command_interface": "joint_velocity",
                                    "goal_executor": "openeta.worker_mink_goal.v1",
                                    "execution_location": "bench_worker",
                                    "orientation_policy": "explicit",
                                    "iteration_budget": 100,
                                    "steps_executed": 5,
                                    "stop_reason": "target_reached",
                                    "reached_target": True,
                                },
                            },
                            "collision_coverage": {
                                "coverage_complete": True,
                                "trajectory_checked": True,
                                "world_checked": True,
                            },
                        },
                        "diagnostics": [],
                        "recovery_options": [],
                    },
                },
            },
        ],
    )

    report = audit_tool_event_file(path)

    assert "motion_without_matching_feasible_ik_receipt" not in report[
        "issue_code_counts"
    ]


def test_tool_contract_audit_detects_repeated_ik_capability_gap(tmp_path: Path) -> None:
    path = tmp_path / "tool_calls.jsonl"
    target = {"xyz": [0.1, 0.2, 0.3]}
    events = []
    for seq in (2, 6):
        events.append(
            {
                "seq": seq,
                "event": {
                    "phase": "end",
                    "name": "ik_preview_check",
                    "effect": "read_only",
                    "success": True,
                    "parameters": {
                        "target_pose": target,
                        "check_endpoint_collision": True,
                    },
                    "details": {
                        "operational_success": True,
                        "semantic_outcome": "ik_inconclusive",
                        "outputs": {
                            "ik_preview_receipt": {
                                "classification": "inconclusive",
                                "reason_code": "endpoint_collision_check_unavailable",
                            },
                            "collision_coverage": {"coverage_complete": False},
                        },
                        "diagnostics": [{"code": "collision_coverage_incomplete"}],
                        "recovery_options": [
                            {"action": "repeat_exact_pose_with_kinematics_only"}
                        ],
                    },
                },
            }
        )
    _write_events(path, events)

    report = audit_tool_event_file(path)

    assert report["issue_code_counts"]["repeated_unresolved_ik_capability_gap"] == 1


def test_tool_contract_audit_detects_unpreviewed_motion_and_zero_step_deadlock(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tool_calls.jsonl"

    def failed_move(seq: int, target_y: float, obstacle: str) -> dict:
        return {
            "seq": seq,
            "event": {
                "phase": "end",
                "name": "move_to",
                "effect": "world_mutating",
                "success": True,
                "parameters": {"target_pose": {"xyz": [0.1, target_y, 0.2]}},
                "details": {
                    "operational_success": False,
                    "semantic_outcome": "target_not_reached",
                    "environment_receipt": {"receipt_id": f"receipt-{seq}"},
                    "outputs": {
                        "pose_feedback": {
                            "reached_target": False,
                            "actual_xyz": [0.1, 0.0, 0.15],
                        },
                        "collision_coverage": {
                            "coverage_complete": False,
                            "collision_detected": True,
                        },
                        "motion_summary": {
                            "reached_target": False,
                            "steps_executed": 0,
                            "stop_reason": "collision_detected",
                            "end": {"xyz": [0.1, 0.0, 0.15]},
                            "collision": {
                                "collision_type": "attached_object_world",
                                "obstacle": obstacle,
                            },
                        },
                    },
                    "diagnostics": [{"code": "simulator_mcp_collision"}],
                    "recovery_options": [{"action": "replan_from_actual_pose"}],
                },
            },
        }

    _write_events(path, [failed_move(2, -0.1, "box_1"), failed_move(4, 0.2, "milk_1")])

    report = audit_tool_event_file(path)

    assert report["issue_code_counts"][
        "motion_without_matching_feasible_ik_receipt"
    ] == 2
    assert report["issue_code_counts"]["repeated_zero_step_motion_deadlock"] == 1
    assert "collision_coverage_incomplete" not in report["issue_code_counts"]


def test_tool_contract_audit_detects_repeated_failed_motion_attractor(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tool_calls.jsonl"
    target = {
        "xyz": [0.04, -0.10, 0.16],
        "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    events: list[dict] = []
    for base_seq, actual in (
        (1, [-0.0071, -0.1613, 0.1285]),
        (3, [-0.0076, -0.1610, 0.1279]),
    ):
        events.extend(
            [
                {
                    "seq": base_seq,
                    "event": {
                        "phase": "end",
                        "name": "ik_preview_check",
                        "effect": "read_only",
                        "success": True,
                        "parameters": {"target_pose": target},
                        "details": {
                            "operational_success": True,
                            "semantic_outcome": "ik_feasible",
                            "outputs": {
                                "ik_preview_receipt": {"classification": "feasible"},
                                "collision_coverage": {"coverage_complete": True},
                            },
                            "diagnostics": [],
                            "recovery_options": [],
                        },
                    },
                },
                {
                    "seq": base_seq + 1,
                    "event": {
                        "phase": "end",
                        "name": "move_to",
                        "effect": "world_mutating",
                        "success": True,
                        "parameters": {"target_pose": target},
                        "details": {
                            "operational_success": False,
                            "semantic_outcome": "target_not_reached",
                            "environment_receipt": {
                                "receipt_id": f"receipt-{base_seq}"
                            },
                            "outputs": {
                                "pose_feedback": {
                                    "reached_target": False,
                                    "actual_xyz": actual,
                                },
                                "collision_coverage": {"coverage_complete": True},
                                "motion_summary": {
                                    "reached_target": False,
                                    "steps_executed": 150,
                                    "position_error_m": 0.083,
                                    "stop_reason": "iteration_limit",
                                    "end": {"xyz": actual},
                                },
                            },
                            "diagnostics": [{"code": "target_not_reached"}],
                            "recovery_options": [
                                {"action": "replan_from_actual_pose"}
                            ],
                        },
                    },
                },
            ]
        )
    _write_events(path, events)

    report = audit_tool_event_file(path)

    assert report["issue_code_counts"]["repeated_failed_motion_attractor"] == 1


def test_tool_contract_audit_does_not_require_receipts_from_interrupted_motion(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tool_calls.jsonl"
    _write_events(
        path,
        [
            {
                "seq": 2,
                "event": {
                    "phase": "end",
                    "name": "move_to",
                    "effect": "world_mutating",
                    "success": False,
                    "parameters": {"target_pose": {"xyz": [0.1, 0.2, 0.3]}},
                    "details": {
                        "operational_success": False,
                        "semantic_outcome": "operational_failure",
                        "outputs": {},
                        "diagnostics": [{"code": "execution_cancelled"}],
                        "recovery_options": [{"action": "observe"}],
                    },
                },
            }
        ],
    )

    report = audit_tool_event_file(path)

    assert "motion_without_pose_feedback" not in report["issue_code_counts"]
    assert "motion_without_collision_coverage" not in report["issue_code_counts"]


def test_tool_contract_audit_detects_unbound_proxy_and_probable_lift_slip(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tool_calls.jsonl"
    _write_events(
        path,
        [
            {
                "seq": 2,
                "event": {
                    "phase": "end",
                    "name": "gripper_control",
                    "effect": "world_mutating",
                    "success": True,
                    "parameters": {"position": 0},
                    "details": {
                        "operational_success": True,
                        "semantic_outcome": "completed",
                        "outputs": {
                            "attachment_proxy_receipt": {
                                "schema_version": "openeta.attachment_proxy_receipt.v1",
                                "status": "tentative",
                                "binding_source": "nearest_object_fallback",
                                "target_object_name": "distractor_1",
                            },
                            "observation_summary": {
                                "robot": {
                                    "gripper_state": {"openness": 0.15}
                                }
                            },
                        },
                        "environment_receipt": {"receipt_id": "close"},
                        "diagnostics": [],
                        "recovery_options": [],
                    },
                },
            },
            {
                "seq": 4,
                "event": {
                    "phase": "end",
                    "name": "move_to",
                    "effect": "world_mutating",
                    "success": True,
                    "parameters": {
                        "target_pose": {"xyz": [0.0, 0.0, 0.20]}
                    },
                    "details": {
                        "operational_success": True,
                        "semantic_outcome": "completed",
                        "outputs": {
                            "pose_feedback": {"reached_target": True},
                            "collision_coverage": {"coverage_complete": True},
                            "attachment_proxy_receipt": {
                                "schema_version": "openeta.attachment_proxy_receipt.v1",
                                "status": "retired",
                                "reason": "aperture_collapsed_to_empty_close",
                                "target_object_name": "distractor_1",
                                "measured_open_fraction": 0.03,
                                "attachment_proven": False,
                            },
                            "motion_summary": {
                                "reached_target": True,
                                "steps_executed": 4,
                                "start": {"xyz": [0.0, 0.0, 0.10]},
                                "end": {"xyz": [0.0, 0.0, 0.20]},
                                "controller_receipt": {
                                    "schema_version": "openeta.controller_execution_receipt.v1",
                                    "controller_id": "mink",
                                    "command_interface": "joint_velocity",
                                    "goal_executor": "goal",
                                    "execution_location": "worker",
                                    "orientation_policy": "preserve_current",
                                    "iteration_budget": 100,
                                    "steps_executed": 4,
                                    "stop_reason": "target_reached",
                                    "reached_target": True,
                                },
                            },
                            "observation_summary": {
                                "robot": {
                                    "gripper_state": {"openness": 0.03}
                                }
                            },
                        },
                        "environment_receipt": {"receipt_id": "lift"},
                        "diagnostics": [],
                        "recovery_options": [],
                    },
                },
            },
        ],
    )

    report = audit_tool_event_file(path)

    assert report["issue_code_counts"][
        "attachment_proxy_not_bound_to_compiled_target"
    ] == 1
    assert report["issue_code_counts"][
        "probable_grasp_slip_after_lift_probe"
    ] == 1


def test_tool_contract_audit_requires_attachment_refresh_after_lift_probe(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tool_calls.jsonl"
    _write_events(
        path,
        [
            {
                "seq": 2,
                "event": {
                    "phase": "end",
                    "name": "gripper_control",
                    "effect": "world_mutating",
                    "success": True,
                    "parameters": {"position": 0},
                    "details": {
                        "operational_success": True,
                        "semantic_outcome": "completed",
                        "outputs": {
                            "attachment_proxy_receipt": {
                                "status": "tentative",
                                "binding_source": "host_compiled_target_provenance",
                                "target_object_name": "milk_1",
                            },
                            "observation_summary": {
                                "robot": {"gripper_state": {"openness": 0.2}}
                            },
                        },
                        "environment_receipt": {"receipt_id": "close"},
                        "diagnostics": [],
                        "recovery_options": [],
                    },
                },
            },
            {
                "seq": 4,
                "event": {
                    "phase": "end",
                    "name": "move_to",
                    "effect": "world_mutating",
                    "success": True,
                    "parameters": {"target_pose": {"xyz": [0.0, 0.0, 0.2]}},
                    "details": {
                        "operational_success": True,
                        "semantic_outcome": "completed",
                        "outputs": {
                            "pose_feedback": {"reached_target": True},
                            "collision_coverage": {"coverage_complete": True},
                            "motion_summary": {
                                "reached_target": True,
                                "steps_executed": 4,
                                "start": {"xyz": [0.0, 0.0, 0.1]},
                                "end": {"xyz": [0.0, 0.0, 0.2]},
                            },
                        },
                        "environment_receipt": {"receipt_id": "lift"},
                        "diagnostics": [],
                        "recovery_options": [],
                    },
                },
            },
        ],
    )

    report = audit_tool_event_file(path)

    assert report["issue_code_counts"][
        "lift_probe_without_attachment_refresh_receipt"
    ] == 1


def test_tool_contract_audit_flags_oversized_clearance_as_attachment_probe(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tool_calls.jsonl"
    _write_events(
        path,
        [
            {
                "seq": 2,
                "event": {
                    "phase": "end",
                    "name": "gripper_control",
                    "effect": "world_mutating",
                    "success": True,
                    "parameters": {"position": 0},
                    "details": {
                        "operational_success": True,
                        "semantic_outcome": "requires_attachment_probe",
                        "outputs": {
                            "attachment_proxy_receipt": {
                                "status": "tentative",
                                "binding_source": "host_compiled_target_provenance",
                                "target_object_name": "milk_1",
                            },
                            "observation_summary": {
                                "robot": {"gripper_state": {"openness": 0.7}}
                            },
                        },
                        "environment_receipt": {"receipt_id": "close"},
                        "diagnostics": [],
                        "recovery_options": [{"action": "small_lift_probe"}],
                    },
                },
            },
            {
                "seq": 4,
                "event": {
                    "phase": "end",
                    "name": "move_to",
                    "effect": "world_mutating",
                    "success": True,
                    "parameters": {
                        "target_pose": {
                            "xyz": [-0.08, -0.05, 0.12],
                            "waypoint_role": "grasp_clearance",
                        }
                    },
                    "details": {
                        "operational_success": True,
                        "semantic_outcome": "no_attachment_evidence",
                        "outputs": {
                            "pose_feedback": {"reached_target": True},
                            "collision_coverage": {"coverage_complete": True},
                            "attachment_proxy_receipt": {
                                "status": "retired",
                                "reason": "aperture_collapsed_to_empty_close",
                                "target_object_name": "milk_1",
                                "measured_open_fraction": 0.01,
                            },
                            "motion_summary": {
                                "reached_target": True,
                                "steps_executed": 10,
                                "start": {"xyz": [0.0, 0.0, 0.0]},
                                "end": {"xyz": [-0.08, -0.05, 0.12]},
                            },
                        },
                        "environment_receipt": {"receipt_id": "lift"},
                        "diagnostics": [],
                        "recovery_options": [{"action": "reopen_and_repair_contact"}],
                    },
                },
            },
        ],
    )

    report = audit_tool_event_file(path)

    issue = next(
        item
        for item in report["issues"]
        if item["code"] == "oversized_attachment_probe_after_close"
    )
    assert issue["evidence"]["waypoint_role"] == "grasp_clearance"
    assert issue["evidence"]["eef_displacement_m"] > 0.14


def test_tool_contract_audit_flags_lift_after_explicit_empty_close(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tool_calls.jsonl"
    _write_events(
        path,
        [
            {
                "seq": 2,
                "event": {
                    "phase": "end",
                    "name": "gripper_control",
                    "effect": "world_mutating",
                    "success": True,
                    "parameters": {"position": 0},
                    "details": {
                        "operational_success": True,
                        "semantic_outcome": "no_attachment_evidence",
                        "outputs": {
                            "attachment_proxy_receipt": {
                                "status": "not_armed",
                                "reason": "empty_close_or_no_measurable_contact",
                                "measured_open_fraction": 0.03,
                                "attachment_proven": False,
                            },
                            "observation_summary": {
                                "robot": {"gripper_state": {"openness": 0.03}}
                            },
                        },
                        "environment_receipt": {"receipt_id": "close"},
                        "diagnostics": [],
                        "recovery_options": [{"action": "reopen_and_repair_contact"}],
                    },
                },
            },
            {
                "seq": 4,
                "event": {
                    "phase": "end",
                    "name": "move_to",
                    "effect": "world_mutating",
                    "success": True,
                    "parameters": {"target_pose": {"xyz": [0.0, 0.0, 0.2]}},
                    "details": {
                        "operational_success": True,
                        "semantic_outcome": "completed",
                        "outputs": {
                            "pose_feedback": {"reached_target": True},
                            "collision_coverage": {"coverage_complete": True},
                            "motion_summary": {
                                "reached_target": True,
                                "steps_executed": 4,
                                "start": {"xyz": [0.0, 0.0, 0.1]},
                                "end": {"xyz": [0.0, 0.0, 0.2]},
                            },
                        },
                        "environment_receipt": {"receipt_id": "lift"},
                        "diagnostics": [],
                        "recovery_options": [],
                    },
                },
            },
        ],
    )

    report = audit_tool_event_file(path)

    assert report["issue_code_counts"]["lift_after_explicit_empty_close"] == 1
    assert "lift_probe_without_attachment_refresh_receipt" not in report[
        "issue_code_counts"
    ]


def test_tool_contract_audit_flags_contact_residual_above_compiler_guidance(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tool_calls.jsonl"
    _write_events(
        path,
        [
            {
                "seq": 2,
                "event": {
                    "phase": "end",
                    "name": "move_to",
                    "effect": "world_mutating",
                    "success": True,
                    "parameters": {
                        "target_pose": {
                            "xyz": [0.1, 0.2, 0.3],
                            "compiled_grasp_id": "compiled-1",
                            "waypoint_role": "grasp_contact",
                        }
                    },
                    "details": {
                        "operational_success": True,
                        "semantic_outcome": "completed",
                        "outputs": {
                            "pose_feedback": {"reached_target": True},
                            "collision_coverage": {"coverage_complete": True},
                            "motion_summary": {
                                "reached_target": True,
                                "max_axis_position_error_m": 0.0079,
                                "position_error_m": 0.012,
                                "orientation_error_rad": 0.085,
                                "steps_executed": 7,
                                "start": {"xyz": [0.1, 0.2, 0.4]},
                                "end": {"xyz": [0.094, 0.193, 0.305]},
                                "controller_receipt": {
                                    "schema_version": "openeta.controller_execution_receipt.v1",
                                    "controller_id": "mink",
                                    "command_interface": "joint_velocity",
                                    "goal_executor": "goal",
                                    "execution_location": "worker",
                                    "orientation_policy": "explicit",
                                    "iteration_budget": 100,
                                    "steps_executed": 7,
                                    "stop_reason": "target_reached",
                                    "reached_target": True,
                                },
                            },
                        },
                        "environment_receipt": {"receipt_id": "contact"},
                        "diagnostics": [],
                        "recovery_options": [],
                    },
                },
            }
        ],
    )

    report = audit_tool_event_file(path)

    issue = next(
        value
        for value in report["issues"]
        if value["code"] == "contact_execution_residual_exceeds_guidance"
    )
    assert issue["evidence"]["max_axis_position_error_m"] == 0.0079
    assert issue["evidence"]["recommended_max_axis_error_m"] == 0.005


def test_tool_contract_audit_flags_mutually_exclusive_sam3_prompts(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tool_calls.jsonl"
    _write_events(
        path,
        [
            {
                "seq": 2,
                "event": {
                    "phase": "end",
                    "name": "sam3",
                    "effect": "read_only",
                    "success": False,
                    "parameters": {
                        "source_packet_id": "obs-0001",
                        "mode": "points",
                        "points": [{"x": 10, "y": 10, "label": 1}],
                        "roi_bbox_xyxy": [0, 0, 20, 20],
                    },
                    "details": {
                        "operational_success": False,
                        "semantic_outcome": "operational_failure",
                        "outputs": {},
                        "diagnostics": [{"code": "conflicting_prompts"}],
                        "recovery_options": [{"action": "remove_roi_or_points"}],
                    },
                },
            }
        ],
    )

    report = audit_tool_event_file(path)

    assert report["issue_code_counts"][
        "mutually_exclusive_perception_prompts"
    ] == 1


def test_tool_contract_audit_resolves_session_directory(tmp_path: Path) -> None:
    path = tmp_path / "session" / "rollout" / "tool_calls.jsonl"
    path.parent.mkdir(parents=True)
    _write_events(path, [])

    report = audit_tool_event_file(tmp_path / "session")

    assert report["source"] == str(path)
    assert report["tool_call_count"] == 0


def test_tool_contract_audit_reads_blocked_pipeline_trace_and_finds_lost_contact_receipt(
    tmp_path: Path,
) -> None:
    session = tmp_path / "session"
    path = session / "rollout" / "tool_calls.jsonl"
    path.parent.mkdir(parents=True)
    _write_events(
        path,
        [
            {
                "seq": 7,
                "timestamp_s": 10.0,
                "event": {
                    "phase": "end",
                    "name": "move_to",
                    "effect": "world_mutating",
                    "success": True,
                    "parameters": {
                        "target_pose": {
                            "xyz": [0.1, 0.2, 0.1],
                            "compiled_grasp_id": "compiled-contact",
                            "waypoint_role": "grasp_contact",
                        }
                    },
                    "details": {
                        "operational_success": True,
                        "semantic_outcome": "completed",
                        "environment_receipt": {},
                        "outputs": {
                            "motion_summary": {
                                "reached_target": True,
                                "steps_executed": 4,
                            }
                        },
                        "diagnostics": [],
                        "recovery_options": [],
                    },
                },
            }
        ],
    )
    _write_events(
        session / "trace.jsonl",
        [
            {
                "event_type": "pipeline_plan",
                "timestamp_s": 11.0,
                "payload": {
                    "status": "blocked",
                    "request": {
                        "kind": "tool_call",
                        "name": "gripper_control",
                        "parameters": {"position": 0},
                    },
                    "metadata": {
                        "repair_bundle": {
                            "code": "compiled_contact_receipt_missing",
                            "latest_ik_preview": {
                                "target_pose": {
                                    "compiled_grasp_id": "compiled-contact"
                                }
                            },
                            "allowed_next_calls": [
                                {"tool": "observe", "parameters": {}}
                            ],
                        }
                    },
                },
            }
        ],
    )

    report = audit_tool_event_file(session)

    assert report["trace_source"] == str(session / "trace.jsonl")
    assert report["pipeline_blocked_count"] == 1
    assert report["issue_code_counts"][
        "gate_rejected_close_after_successful_contact"
    ] == 1
    assert report["quality_checks"][
        "pipeline_gate_blocks_are_causally_consistent"
    ] is False


def test_tool_contract_audit_rejects_double_nested_outputs(tmp_path: Path) -> None:
    path = tmp_path / "tool_calls.jsonl"
    _write_events(
        path,
        [
            {
                "seq": 1,
                "event": {
                    "phase": "end",
                    "name": "estimate_depth_prior",
                    "effect": "read_only",
                    "success": True,
                    "parameters": {"source_packet_id": "obs-0001"},
                    "details": {
                        "operational_success": True,
                        "semantic_outcome": "success",
                        "outputs": {
                            "outputs": {
                                "source_packet_id": "obs-0001",
                                "prior_depth": [[0.4]],
                            }
                        },
                        "diagnostics": [],
                        "recovery_options": [],
                    },
                },
            }
        ],
    )

    report = audit_tool_event_file(path)

    assert report["issue_code_counts"]["double_nested_tool_outputs"] == 1
    assert report["quality_checks"][
        "tool_outputs_use_canonical_single_envelope"
    ] is False
    assert report["contract_pass"] is False


def test_tool_contract_audit_refuses_ambiguous_directory(tmp_path: Path) -> None:
    for session_name in ("session-a", "session-b"):
        path = tmp_path / session_name / "rollout" / "tool_calls.jsonl"
        path.parent.mkdir(parents=True)
        _write_events(path, [])

    try:
        audit_tool_event_file(tmp_path)
    except ValueError as exc:
        assert "found 2 rollouts" in str(exc)
        assert "pass one session" in str(exc)
    else:
        raise AssertionError("ambiguous directory should fail closed")
