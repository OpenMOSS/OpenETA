from __future__ import annotations

from adapter.protocol import EnvAction, EnvObservation, RobotState
from agent.runtime.memory import AgentMemory
from agent.tools.grasp_geometry import DEFAULT_GRASP_PROFILE, compile_grasp_seed

import json


def _candidate(candidate_id: str, score: float) -> dict:
    return {
        "id": candidate_id,
        "frame": "camera",
        "camera_frame": "opencv",
        "score": score,
        "width": 0.06,
        "translation_xyz": [0.1, 0.2, 0.3],
        "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    }


def _tool_action(
    name: str,
    parameters: dict,
    *,
    success: bool = True,
    outputs: dict | None = None,
    grasp_outcome: str = "",
) -> EnvAction:
    details = {"parameters": parameters, "outputs": dict(outputs or {})}
    if grasp_outcome:
        details["supervision"] = {"details": {"grasp_outcome": grasp_outcome}}
    return EnvAction(
        action_type="tool_call",
        command={
            "request": {"kind": "tool_call", "name": name, "parameters": parameters},
            "status": "executed" if success else "failed",
            "tool_calls": [
                {
                    "name": name,
                    "status": "executed" if success else "failed",
                    "result": {"success": success, "details": details},
                }
            ],
        },
    )


def _memory_with_candidates() -> AgentMemory:
    memory = AgentMemory()
    memory.start_session(task="pick up alphabat soup and place it into basket")
    memory.add_action(
        _tool_action(
            "anygrasp",
            {},
            outputs={
                "result_id": "anygrasp-1",
                "grasp_candidates": [
                    _candidate("grasp_001", 0.7),
                    _candidate("grasp_000", 0.9),
                ],
            },
        )
    )
    return memory


def _compiled(memory: AgentMemory) -> dict:
    profile = json.loads(DEFAULT_GRASP_PROFILE.read_text(encoding="utf-8"))
    return compile_grasp_seed(
        {
            "camera_pose": memory.anygrasp_candidate_policy()["active_candidate"],
            "camera_extrinsics": {
                "pos": [0.0, 0.0, 0.0],
                "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            },
            "camera_frame_id": "agentview",
            "target_class": "upright_can",
            "scene_epoch": memory.scene_epoch(),
        },
        profile=profile,
        profile_sha256="profile-sha",
    )


def test_select_grasp_candidate_repoints_active_and_gate_follows() -> None:
    memory = _memory_with_candidates()
    policy = memory.grasp_candidate_policy()
    # Greedy default: highest score is active.
    assert policy["active_candidate"]["id"] == "grasp_000"
    # The gate blocks compiling a non-active candidate before selection.
    assert "grasp_000" in memory.grasp_candidate_gate_error(
        tool_name="compile_grasp_seed",
        parameters={"camera_pose": {"id": "grasp_001"}, "scene_epoch": memory.scene_epoch()},
    )

    selection = memory.select_grasp_candidate(
        candidate_id="grasp_001",
        reason="more top-down approach for the target",
    )

    assert selection["candidate"]["id"] == "grasp_001"
    assert selection["rank"] == 1
    assert selection["previous_candidate_id"] == "grasp_000"
    updated = memory.grasp_candidate_policy()
    assert updated["active_candidate"]["id"] == "grasp_001"
    assert updated["active_rank"] == 1
    assert updated["status"] == "active"
    assert updated["remaining_candidate_ids"] == []
    assert updated["last_agent_selection"]["reason"] == (
        "more top-down approach for the target"
    )
    # After selection, the gate accepts the chosen candidate and rejects the old one.
    assert (
        memory.grasp_candidate_gate_error(
            tool_name="compile_grasp_seed",
            parameters={
                "camera_pose": {"id": "grasp_001"},
                "scene_epoch": memory.scene_epoch(),
            },
        )
        is None
    )
    assert "grasp_001" in memory.grasp_candidate_gate_error(
        tool_name="compile_grasp_seed",
        parameters={"camera_pose": {"id": "grasp_000"}, "scene_epoch": memory.scene_epoch()},
    )


def test_select_grasp_candidate_rejects_out_of_list_id() -> None:
    memory = _memory_with_candidates()
    try:
        memory.select_grasp_candidate(candidate_id="grasp_999", reason="nope")
    except ValueError as exc:
        assert "grasp_999" in str(exc)
        assert "physically-valid" in str(exc)
    else:  # pragma: no cover - selection must reject unknown ids
        raise AssertionError("expected ValueError for unknown candidate id")


def test_select_grasp_candidate_requires_reason() -> None:
    memory = _memory_with_candidates()
    try:
        memory.select_grasp_candidate(candidate_id="grasp_001", reason="  ")
    except ValueError as exc:
        assert "reason" in str(exc)
    else:  # pragma: no cover - selection must require a reason
        raise AssertionError("expected ValueError for missing reason")


def test_select_grasp_candidate_records_audit_trail() -> None:
    memory = _memory_with_candidates()
    memory.select_grasp_candidate(candidate_id="grasp_001", reason="first pick")
    memory.select_grasp_candidate(candidate_id="grasp_000", reason="revert to rank 0")

    policy = memory.grasp_candidate_policy()
    selections = policy["agent_selections"]
    assert [entry["candidate_id"] for entry in selections] == ["grasp_001", "grasp_000"]
    assert selections[0]["previous_candidate_id"] == "grasp_000"
    assert selections[1]["previous_candidate_id"] == "grasp_001"
    assert policy["active_candidate"]["id"] == "grasp_000"


def test_runtime_select_grasp_candidate_tool_delegates_to_memory() -> None:
    from agent.runtime.runtime import OpenEtaAgentRuntime
    from agent.tools.registry import (
        ToolExecutionContext,
        build_default_tool_registry,
    )

    memory = _memory_with_candidates()
    runtime = OpenEtaAgentRuntime(memory=memory, rollout_enabled=False)
    spec = build_default_tool_registry().get("select_grasp_candidate")

    context = ToolExecutionContext(
        name="select_grasp_candidate",
        spec=spec,
        parameters={"candidate_id": "grasp_001", "reason": "clearer top-down approach"},
    )
    result = runtime._select_grasp_candidate_tool(context)

    assert result.success is True
    assert memory.grasp_candidate_policy()["active_candidate"]["id"] == "grasp_001"

    bad = ToolExecutionContext(
        name="select_grasp_candidate",
        spec=spec,
        parameters={"candidate_id": "grasp_404", "reason": "invalid"},
    )
    failure = runtime._select_grasp_candidate_tool(bad)
    assert failure.success is False
    # A failed selection leaves the prior active candidate untouched.
    assert memory.grasp_candidate_policy()["active_candidate"]["id"] == "grasp_001"


def test_anygrasp_requires_compiler_but_anyplace_pose_keeps_generic_transform() -> None:
    memory = _memory_with_candidates()

    assert "compile_grasp_seed" in memory.grasp_candidate_gate_error(
        tool_name="camera_pose_to_world",
        parameters={"camera_pose": {"id": "grasp_000"}},
    )
    assert (
        memory.grasp_candidate_gate_error(
            tool_name="camera_pose_to_world",
            parameters={"camera_pose": {"id": "place_grasp_000", "source_grasp_id": "grasp_000"}},
        )
        is None
    )


def test_exhausted_candidate_queue_selects_highest_score_fallback() -> None:
    memory = _memory_with_candidates()
    policy = memory.anygrasp_candidate_policy()
    policy["candidates"] = [policy["active_candidate"]]
    policy["remaining_candidate_ids"] = []
    policy["target_detection"] = {
        "target_prompt": "black bowl",
        "source_image": "/tmp/current.rgb.png",
    }
    memory.save_fact("grasp_candidate_policy", policy, source="test")

    candidate_id = policy["active_candidate"]["id"]
    memory.add_action(
        _tool_action(
            "move_to",
            {"target_pose": {"id": candidate_id}},
            success=False,
            outputs={
                "motion_summary": {
                    "reached_target": False,
                    "end": {"xyz": [0.1, 0.2, 0.3]},
                }
            },
        )
    )

    policy = memory.grasp_candidate_policy()
    assert policy["status"] == "active"
    assert policy["candidate_fallback"] is True
    assert policy["fallback_reason"] == "all_ranked_candidates_failed"
    assert policy["active_candidate"]["id"] == "grasp_000"
    assert policy["active_candidate"]["candidate_fallback"] is True
    assert memory.grasp_reestimation() is None
    assert memory.grasp_recovery() is None


def test_candidate_attempt_limit_selects_highest_score_before_using_fourth_candidate() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick bowl")
    memory.add_action(
        _tool_action(
            "anygrasp",
            {},
            outputs={
                "result_id": "four-candidates",
                "grasp_candidates": [
                    _candidate(f"grasp_{index:03d}", 1.0 - index * 0.1)
                    for index in range(4)
                ],
            },
        )
    )
    policy = memory.anygrasp_candidate_policy()
    policy["candidate_attempt_count"] = 2
    policy["target_detection"] = {
        "target_prompt": "black bowl",
        "source_image": "/tmp/current.rgb.png",
    }
    memory.save_fact("grasp_candidate_policy", policy, source="test")

    active = policy["active_candidate"]
    memory.add_action(
        _tool_action(
            "move_to",
            {"target_pose": {"id": active["id"]}},
            success=False,
            outputs={
                "motion_summary": {
                    "reached_target": False,
                    "end": {"xyz": [0.1, 0.2, 0.3]},
                }
            },
        )
    )

    policy = memory.anygrasp_candidate_policy()
    assert policy["status"] == "active"
    assert policy["candidate_attempt_count"] == 3
    assert policy["candidate_fallback"] is True
    assert policy["active_candidate"]["id"] == "grasp_000"
    assert policy["active_candidate"]["candidate_fallback"] is True
    assert memory.grasp_reestimation() is None
    assert memory.grasp_recovery() is None


def test_anygrasp_filters_non_executable_width_before_score_ranking() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick can")
    too_wide = _candidate("grasp_000", 0.99)
    too_wide["width"] = 0.1
    executable = _candidate("grasp_001", 0.5)

    memory.add_action(
        _tool_action(
            "anygrasp",
            {},
            outputs={
                "result_id": "anygrasp-width-filter",
                "grasp_candidates": [too_wide, executable],
            },
        )
    )

    policy = memory.anygrasp_candidate_policy()
    assert policy["raw_candidate_count"] == 2
    assert policy["candidate_count"] == 1
    assert policy["active_candidate"]["id"] == "grasp_001"
    assert policy["active_candidate"]["rank"] == 0
    assert policy["rejected_candidates"] == [
        {
            "candidate_id": "grasp_000",
            "rank": None,
            "score": 0.99,
            "reason": (
                "candidate width exceeds calibration max_gripper_width_m 0.0800 m"
            ),
            "source": "physical_gripper_width_filter",
        }
    ]


def test_selected_mask_geometry_becomes_task_strategy_compile_hint() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick alphabet soup")
    memory.save_fact(
        "pending_sam3_selection",
        {
            "result_id": "sam3-geometry",
            "candidates": [{"id": "detection_000", "mask_ref": "mask.png"}],
        },
        source="sam3",
    )
    selected = memory.resolve_sam3_selection(
        result_id="sam3-geometry",
        detection_id="detection_000",
        selection_source="main_agent_vlm",
        target_geometry_family="upright_can",
    )
    assert selected["target_geometry_family"] == "upright_can"

    memory.add_action(
        _tool_action(
            "grasp_pose_estimate",
            {},
            outputs={
                "result_id": "grasp-geometry",
                "mode": "targeted",
                "selected_backend": "anygrasp",
                "grasp_candidates": [_candidate("grasp_000", 0.9)],
            },
        )
    )

    policy = memory.grasp_candidate_policy()
    assert policy["target_detection"]["id"] == "detection_000"
    assert policy["compile_hints"] == {
        "target_geometry_family": "upright_can",
    }


def test_new_grasp_queue_clears_prior_release_but_keeps_completed_ledger() -> None:
    memory = AgentMemory()
    memory.start_session(task="put both cans in the basket")
    memory.save_fact(
        "placement_release",
        {
            "status": "retreated",
            "candidate_id": "first-grasp",
            "placement_pose_id": "first-place",
        },
        source="test",
    )
    completed = {
        "items": [
            {
                "candidate_id": "first-grasp",
                "placement_pose_id": "first-place",
                "target_object": "alphabet soup",
            }
        ]
    }
    memory.save_fact("completed_placement_subgoals", completed, source="test")

    memory.add_action(
        _tool_action(
            "grasp_pose_estimate",
            {},
            outputs={
                "result_id": "second-grasp-queue",
                "mode": "targeted",
                "selected_backend": "anygrasp",
                "grasp_candidates": [_candidate("second-grasp", 0.9)],
            },
        )
    )

    assert memory.placement_release() is None
    assert memory.get_memory("completed_placement_subgoals", namespace="facts")["facts"][
        "completed_placement_subgoals"
    ]["value"] == completed


def test_denied_host_release_invalidates_dropped_grasp_instead_of_replaying() -> None:
    memory = _memory_with_candidates()
    memory.save_fact(
        "grasp_execution",
        {"status": "completed", "stage": "attached", "candidate_id": "grasp_000"},
        source="test",
    )
    memory.save_fact(
        "attachment_gate",
        {"status": "resolved", "verdict": "PASS", "candidate_id": "grasp_000"},
        source="test",
    )
    memory.save_fact(
        "placement_release",
        {
            "schema_version": "openeta.placement_release.v1",
            "status": "ready",
            "candidate_id": "grasp_000",
            "placement_pose_id": "place-1",
            "release_pose": {"frame": "world", "xyz": [0.1, 0.2, 0.3]},
        },
        source="test",
    )
    memory.save_fact(
        "selected_sam3_detection",
        {"id": "basket-mask", "label": "basket"},
        source="test",
    )
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "gripper_control",
                    "parameters": {"position": 1},
                },
                "status": "failed",
                "metadata": {
                    "planner_metadata": {
                        "host_obligation": {
                            "stage": "release",
                            "tool": "gripper_control",
                        }
                    }
                },
                "tool_calls": [
                    {
                        "name": "gripper_control",
                        "status": "failed",
                        "result": {
                            "success": False,
                            "content": "Target detached before the valid release pose.",
                            "details": {
                                "diagnostics": [{"code": "supervision_denied"}],
                            },
                        },
                    }
                ],
            },
        )
    )

    assert memory.placement_release()["status"] == "failed"
    assert memory.grasp_execution() is None
    assert memory.attachment_gate() is None
    assert memory.grasp_candidate_policy() is None
    assert memory.selected_sam3_detection() is None
    assert any(event.event_type == "placement_release_failed" for event in memory.events)


def test_host_grasp_execution_accepts_bounded_pose_adjustment_at_contact() -> None:
    memory = _memory_with_candidates()
    compiled = _compiled(memory)
    memory.add_action(_tool_action("compile_grasp_seed", {"scene_epoch": 0}, outputs=compiled))
    assert memory.grasp_execution()["stage"] == "open"
    assert memory.grasp_candidate_policy()["compile_hints"] == {
        "target_geometry_family": "upright_can",
        "strategy_id": "top-down-vertical-panda-p8",
        "pregrasp_distance_m": 0.25,
    }

    required = memory.grasp_execution()["required_action"]
    memory.add_action(_tool_action(required["name"], required["parameters"]))
    assert memory.grasp_execution()["stage"] == "hover"
    required = memory.grasp_execution()["required_action"]
    reference_xyz = required["parameters"]["target_pose"]["xyz"]
    adjusted = {
        "target_pose": {
            **required["parameters"]["target_pose"],
            "xyz": [reference_xyz[0] + 0.02, reference_xyz[1], reference_xyz[2]],
        }
    }
    assert memory.grasp_execution_gate_error(tool_name="move_to", parameters=adjusted) is None
    outside_envelope = {"target_pose": {**required["parameters"]["target_pose"], "xyz": [9, 9, 9]}}
    assert "closed-loop envelope" in memory.grasp_execution_gate_error(
        tool_name="move_to", parameters=outside_envelope
    )
    memory.add_action(_tool_action(required["name"], adjusted))
    assert memory.grasp_execution()["stage"] == "align"

    compiled_id = compiled["compiled_grasp_id"]
    alignment = {
        "schema_version": "openeta.wrist_alignment.v1",
        "alignment_id": "align-1",
        "compiled_grasp_id": compiled_id,
        "candidate_id": "grasp_000",
        "scene_epoch": memory.scene_epoch(),
        "aligned_hover_pose": {**compiled["hover_pose"], "xyz": [0.0, 0.0, 0.6]},
        "adjusted_precontact_pose": {
            **compiled["contact_pose"],
            "xyz": [0.0, 0.0, 0.55],
            "grasp_stage": "precontact",
        },
        "adjusted_contact_pose": {**compiled["contact_pose"], "xyz": [0.0, 0.0, 0.5]},
    }
    memory.add_action(_tool_action("compute_wrist_alignment", {}, outputs=alignment))
    assert memory.grasp_execution()["stage"] == "align_move"
    for expected_stage in ("precontact", "descend", "close"):
        required = memory.grasp_execution()["required_action"]
        memory.add_action(_tool_action(required["name"], required["parameters"]))
        assert memory.grasp_execution()["stage"] == expected_stage
    assert memory.anygrasp_candidate_policy()["status"] == "accepted"


def test_acknowledged_binary_gripper_state_is_latched_and_skips_redundant_open() -> None:
    memory = _memory_with_candidates()
    memory.add_action(_tool_action("gripper_control", {"position": 1}))

    assert memory.gripper_command_state()["position"] == 1
    assert memory.gripper_command_state()["latched"] is True

    compiled = _compiled(memory)
    memory.add_action(
        _tool_action(
            "compile_grasp_seed",
            {"scene_epoch": memory.scene_epoch()},
            outputs={**compiled, "scene_epoch": memory.scene_epoch()},
        )
    )

    execution = memory.grasp_execution()
    assert execution["stage"] == "hover"
    assert execution["required_action"]["name"] == "move_to"


def test_close_timeout_reconciliation_accepts_object_between_fingers() -> None:
    memory = _memory_with_candidates()
    compiled = _compiled(memory)
    memory.add_action(_tool_action("compile_grasp_seed", {"scene_epoch": 0}, outputs=compiled))
    execution = memory.grasp_execution()
    execution.update(
        {
            "stage": "close",
            "required_action": {"name": "gripper_control", "parameters": {"position": 0}},
        }
    )
    memory.save_fact("grasp_execution", execution, source="test")
    memory.add_action(
        _tool_action(
            "gripper_control",
            {"position": 0},
            success=False,
            outputs={"motion_outcome": "unknown", "reconciliation_required": True},
        )
    )

    memory.add_observation(
        EnvObservation(
            task="pick",
            cameras=[],
            robot=RobotState(gripper_state={"open": True, "openness": 0.45}),
        )
    )

    assert memory.motion_reconciliation()["status"] == "completed"
    assert memory.gripper_command_state()["position"] == 0
    assert memory.grasp_execution()["stage"] == "probe"


def test_lift_probe_reconciliation_advances_to_attachment_gate() -> None:
    memory = _memory_with_candidates()
    compiled = _compiled(memory)
    memory.add_action(_tool_action("compile_grasp_seed", {"scene_epoch": 0}, outputs=compiled))
    execution = memory.grasp_execution()
    execution.update({"stage": "probe", "required_action": None})
    memory.save_fact("grasp_execution", execution, source="test")
    required = {
        "target_pose": {
            "frame": "world",
            "probe_type": "grasp_lift",
            "source_grasp_id": "grasp_000",
            "xyz": [0.1, 0.2, 0.4],
        }
    }
    memory.save_fact(
        "grasp_lift_probe",
        {
            "status": "required",
            "candidate_id": "grasp_000",
            "required_parameters": required,
        },
        source="test",
    )
    memory.add_action(
        _tool_action(
            "move_to",
            required,
            success=False,
            outputs={"motion_outcome": "unknown", "reconciliation_required": True},
        )
    )

    memory.add_observation(
        EnvObservation(
            task="pick",
            cameras=[],
            robot=RobotState(end_effector_pose={"xyz": [0.102, 0.2, 0.399]}),
        )
    )

    assert memory.motion_reconciliation()["status"] == "completed"
    assert memory.grasp_lift_probe()["status"] == "completed"
    assert memory.grasp_lift_probe()["last_attempt_status"] == "reconciled"
    assert memory.grasp_execution()["stage"] == "attachment"
    assert memory.attachment_gate()["verdict"] == "UNKNOWN"
    assert memory.grasp_execution()["attachment_actions"]["pass"]["parameters"][
        "target_pose"
    ]["xyz"] == [0.1, 0.2, 0.48000000000000004]


def test_robotiq_object_detection_resolves_attachment_and_completes_full_lift() -> None:
    memory = AgentMemory()
    full_lift = {
        "name": "move_to",
        "parameters": {
            "target_pose": {
                "frame": "world",
                "xyz": [0.1, 0.2, 0.4],
                "source_grasp_id": "grasp_000",
                "grasp_stage": "full_lift",
            }
        },
    }
    memory.save_fact(
        "gripper_command_state",
        {"position": 0, "latch": "closed"},
        source="test",
    )
    memory.save_fact(
        "grasp_execution",
        {
            "schema_version": "openeta.grasp_execution.v1",
            "status": "required",
            "stage": "attachment",
            "candidate_id": "grasp_000",
            "required_action": None,
            "attachment_actions": {
                "pass": full_lift,
                "fail": {"name": "gripper_control", "parameters": {"position": 1}},
            },
        },
        source="test",
    )
    memory.save_fact(
        "attachment_gate",
        {
            "status": "pending",
            "verdict": "UNKNOWN",
            "candidate_id": "grasp_000",
        },
        source="test",
    )

    memory.add_observation(
        EnvObservation(
            task="pick",
            cameras=[],
            robot=RobotState(
                gripper_state={
                    "model": "robotiq",
                    "object_detection": "object_detected_closing",
                    "position": 159,
                    "position_normalized": 0.6235,
                    "requested_position": 255,
                }
            ),
        )
    )

    assert memory.attachment_gate()["verdict"] == "PASS"
    memory.add_action(
        _tool_action(
            full_lift["name"],
            full_lift["parameters"],
            grasp_outcome="unknown",
        )
    )

    assert memory.attachment_gate()["verdict"] == "PASS"
    assert memory.attachment_gate()["pass_action_completed"] is True
    assert memory.grasp_execution()["status"] == "completed"
    assert memory.grasp_execution()["stage"] == "attached"


def test_robotiq_at_position_no_object_remains_unknown_at_any_width() -> None:
    memory = AgentMemory()
    memory.save_fact(
        "gripper_command_state",
        {"position": 0, "latch": "closed"},
        source="test",
    )
    memory.save_fact(
        "grasp_execution",
        {
            "schema_version": "openeta.grasp_execution.v1",
            "status": "required",
            "stage": "attachment",
            "candidate_id": "grasp_000",
            "required_action": None,
            "attachment_actions": {
                "pass": {
                    "name": "move_to",
                    "parameters": {
                        "target_pose": {
                            "frame": "world",
                            "xyz": [0.1, 0.2, 0.4],
                            "source_grasp_id": "grasp_000",
                            "grasp_stage": "full_lift",
                        }
                    },
                },
                "fail": {"name": "gripper_control", "parameters": {"position": 1}},
            },
        },
        source="test",
    )
    memory.save_fact(
        "attachment_gate",
        {"status": "pending", "verdict": "UNKNOWN", "candidate_id": "grasp_000"},
        source="test",
    )

    memory.add_observation(
        EnvObservation(
            task="pick",
            cameras=[],
            robot=RobotState(
                gripper_state={
                    "model": "robotiq",
                    "object_detection": "at_position_no_object",
                    "position_normalized": 0.82,
                    "openness": 0.18,
                }
            ),
        )
    )

    assert memory.attachment_gate()["verdict"] == "UNKNOWN"
    memory.add_observation(
        EnvObservation(
            task="pick",
            cameras=[],
            robot=RobotState(
                gripper_state={
                    "model": "robotiq",
                    "object_detection": "at_position_no_object",
                    "position_normalized": 0.99,
                    "openness": 0.01,
                }
            ),
        )
    )

    assert memory.attachment_gate()["verdict"] == "UNKNOWN"


def test_attachment_fail_recovery_open_advances_to_next_candidate() -> None:
    memory = _memory_with_candidates()
    active_candidate_id = memory.grasp_candidate_policy()["active_candidate"]["id"]
    memory.save_fact(
        "grasp_execution",
        {
            "schema_version": "openeta.grasp_execution.v1",
            "status": "required",
            "stage": "attachment",
            "candidate_id": active_candidate_id,
            "required_action": None,
            "attachment_actions": {
                "pass": {
                    "name": "move_to",
                    "parameters": {
                        "target_pose": {
                            "frame": "world",
                            "xyz": [0.1, 0.2, 0.4],
                            "source_grasp_id": active_candidate_id,
                            "grasp_stage": "full_lift",
                        }
                    },
                },
                "fail": {"name": "gripper_control", "parameters": {"position": 1}},
            },
        },
        source="test",
    )
    memory.save_fact(
        "attachment_gate",
        {
            "status": "resolved",
            "verdict": "FAIL",
            "candidate_id": active_candidate_id,
        },
        source="test",
    )

    memory.add_action(_tool_action("gripper_control", {"position": 1}))

    policy = memory.grasp_candidate_policy()
    assert policy["active_candidate"]["id"] == "grasp_001"
    assert policy["rejected_candidates"][0]["source"] == "attachment_gate_rejected"
    assert memory.grasp_execution() is None
    assert memory.attachment_gate() is None


def test_attachment_full_lift_waits_for_observation_without_replaying() -> None:
    memory = AgentMemory()
    full_lift = {
        "name": "move_to",
        "parameters": {
            "target_pose": {
                "frame": "world",
                "xyz": [0.1, 0.2, 0.4],
                "source_grasp_id": "grasp_000",
                "grasp_stage": "full_lift",
            }
        },
    }
    memory.save_fact(
        "gripper_command_state",
        {"position": 0, "latch": "closed"},
        source="test",
    )
    memory.save_fact(
        "grasp_execution",
        {
            "schema_version": "openeta.grasp_execution.v1",
            "status": "required",
            "stage": "attachment",
            "candidate_id": "grasp_000",
            "required_action": None,
            "attachment_actions": {
                "pass": full_lift,
                "fail": {"name": "gripper_control", "parameters": {"position": 1}},
            },
        },
        source="test",
    )
    memory.save_fact(
        "attachment_gate",
        {
            "status": "pending",
            "verdict": "UNKNOWN",
            "candidate_id": "grasp_000",
        },
        source="test",
    )

    memory.add_action(_tool_action(full_lift["name"], full_lift["parameters"]))

    assert memory.grasp_execution()["status"] == "required"
    assert memory.attachment_gate()["pass_action_completed"] is True
    assert memory.attachment_gate()["pass_action_attempt_count"] == 1

    memory.add_observation(
        EnvObservation(
            task="pick",
            cameras=[],
            robot=RobotState(
                gripper_state={
                    "object_detection": "object_detected_closing",
                    "position": 159,
                    "requested_position": 255,
                }
            ),
        )
    )

    assert memory.attachment_gate()["verdict"] == "PASS"
    assert memory.grasp_execution()["stage"] == "attached"


def test_unreached_hover_rejects_candidate_without_advancing_execution_stage() -> None:
    memory = _memory_with_candidates()
    compiled = _compiled(memory)
    memory.add_action(_tool_action("compile_grasp_seed", {"scene_epoch": 0}, outputs=compiled))
    open_action = memory.grasp_execution()["required_action"]
    memory.add_action(_tool_action(open_action["name"], open_action["parameters"]))
    hover_action = memory.grasp_execution()["required_action"]

    memory.add_action(
        _tool_action(
            hover_action["name"],
            hover_action["parameters"],
            outputs={"motion_summary": {"reached_target": False}},
        )
    )

    assert memory.grasp_execution() is None
    assert memory.anygrasp_candidate_policy()["active_candidate"]["id"] == "grasp_001"
    assert memory.scene_epoch() == 2
    assert memory.transition_ledger()[-1]["verdict"] == "FAIL"


def test_near_unreached_hover_advances_to_alignment_without_rejecting_candidate() -> None:
    memory = _memory_with_candidates()
    compiled = _compiled(memory)
    memory.add_action(_tool_action("compile_grasp_seed", {"scene_epoch": 0}, outputs=compiled))
    open_action = memory.grasp_execution()["required_action"]
    memory.add_action(_tool_action(open_action["name"], open_action["parameters"]))
    hover_action = memory.grasp_execution()["required_action"]
    target_xyz = hover_action["parameters"]["target_pose"]["xyz"]
    end_xyz = [target_xyz[0] + 0.03, target_xyz[1], target_xyz[2]]

    memory.add_action(
        _tool_action(
            hover_action["name"],
            hover_action["parameters"],
            outputs={
                "motion_summary": {"reached_target": False},
                "response": {
                    "motion_summary": {
                        "reached_target": False,
                        "collision": {"detected": False},
                        "end": {"xyz": end_xyz},
                        "target": {
                            "x": target_xyz[0],
                            "y": target_xyz[1],
                            "z": target_xyz[2],
                        },
                    }
                },
            },
        )
    )

    assert memory.grasp_execution()["stage"] == "align"
    assert memory.anygrasp_candidate_policy()["active_candidate"]["id"] == "grasp_000"
    assert memory.transition_ledger()[-1]["verdict"] == "PASS"


def test_transport_timeout_reconciles_gripper_and_partial_motion() -> None:
    memory = _memory_with_candidates()
    compiled = _compiled(memory)
    memory.add_action(_tool_action("compile_grasp_seed", {"scene_epoch": 0}, outputs=compiled))
    required = memory.grasp_execution()["required_action"]
    timeout = _tool_action(
        required["name"],
        required["parameters"],
        success=False,
        outputs={"motion_outcome": "unknown", "reconciliation_required": True},
    )
    memory.add_action(timeout)
    assert memory.motion_reconciliation()["status"] == "required"
    assert memory.grasp_execution_gate_error(tool_name="move_to", parameters={}) is not None

    memory.add_observation(
        EnvObservation(
            task="pick",
            cameras=[],
            robot=RobotState(gripper_state={"open": True, "openness": 0.95}),
        )
    )
    assert memory.motion_reconciliation()["status"] == "completed"
    assert memory.grasp_execution()["stage"] == "hover"

    hover = memory.grasp_execution()["required_action"]
    memory.add_action(
        _tool_action(
            hover["name"],
            hover["parameters"],
            success=False,
            outputs={"motion_outcome": "unknown", "reconciliation_required": True},
        )
    )
    target = hover["parameters"]["target_pose"]["xyz"]
    memory.add_observation(
        EnvObservation(
            task="pick",
            cameras=[],
            robot=RobotState(end_effector_pose={"xyz": [target[0] + 0.05, target[1], target[2]]}),
        )
    )
    assert memory.motion_reconciliation()["status"] == "required"
    assert (
        memory.grasp_execution_gate_error(tool_name=hover["name"], parameters=hover["parameters"])
        is not None
    )
    assert memory.grasp_execution_gate_error(tool_name="observe", parameters={}) is None
    assert (
        memory.grasp_execution_gate_error(tool_name="gripper_control", parameters={"position": 0})
        is not None
    )

    stable_partial = EnvObservation(
        task="pick",
        cameras=[],
        robot=RobotState(end_effector_pose={"xyz": [target[0] + 0.05, target[1], target[2]]}),
    )
    memory.add_observation(stable_partial)
    assert memory.motion_reconciliation()["status"] == "required"
    memory.add_observation(stable_partial)

    assert memory.motion_reconciliation()["status"] == "failed"
    assert memory.grasp_execution() is None
    assert memory.anygrasp_candidate_policy()["active_candidate"]["id"] == "grasp_001"
    assert memory.anygrasp_candidate_policy()["last_rejection"] == {
        "source": "reconciled_candidate_motion_rejected",
        "target_tool": "move_to",
        "reason": "reconciled_target_not_reached",
    }


def test_transport_timeout_continues_reconciling_after_unresolved_observation() -> None:
    memory = _memory_with_candidates()
    compiled = _compiled(memory)
    memory.add_action(_tool_action("compile_grasp_seed", {"scene_epoch": 0}, outputs=compiled))
    open_action = memory.grasp_execution()["required_action"]
    memory.add_action(_tool_action(open_action["name"], open_action["parameters"]))
    hover = memory.grasp_execution()["required_action"]
    memory.add_action(
        _tool_action(
            hover["name"],
            hover["parameters"],
            success=False,
            outputs={"motion_outcome": "unknown", "reconciliation_required": True},
        )
    )
    target = hover["parameters"]["target_pose"]["xyz"]
    stable_far = EnvObservation(
        task="pick",
        cameras=[],
        robot=RobotState(end_effector_pose={"xyz": [target[0] + 0.3, target[1], target[2]]}),
    )

    memory.add_observation(stable_far)
    assert memory.motion_reconciliation()["status"] == "unresolved"
    memory.add_observation(stable_far)
    memory.add_observation(stable_far)

    assert memory.motion_reconciliation()["status"] == "failed"
    assert memory.grasp_execution() is None
    assert memory.anygrasp_candidate_policy()["active_candidate"]["id"] == "grasp_001"


def test_probe_stage_allows_exact_host_generated_lift_only() -> None:
    memory = _memory_with_candidates()
    compiled = _compiled(memory)
    memory.add_action(_tool_action("compile_grasp_seed", {"scene_epoch": 0}, outputs=compiled))
    execution = memory.grasp_execution()
    execution.update({"stage": "probe", "required_action": None})
    memory.save_fact("grasp_execution", execution, source="test")
    required = {
        "target_pose": {
            "frame": "world",
            "probe_type": "grasp_lift",
            "source_grasp_id": "grasp_000",
            "xyz": [0.1, 0.2, 0.4],
        }
    }
    memory.save_fact(
        "grasp_lift_probe",
        {"status": "required", "candidate_id": "grasp_000", "required_parameters": required},
        source="test",
    )

    assert memory.grasp_execution_gate_error(tool_name="move_to", parameters=required) is None
    assert (
        memory.grasp_execution_gate_error(
            tool_name="move_to",
            parameters={"target_pose": {"xyz": [0.1, 0.2, 0.41]}},
        )
        is not None
    )
