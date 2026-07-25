from __future__ import annotations

from adapter.protocol import CameraFrame, EnvAction, EnvObservation, RobotState
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


def _candidate_with_approach(candidate_id: str, score: float, approach: str) -> dict:
    candidate = _candidate(candidate_id, score)
    candidate["rotation_matrix"] = {
        "top_down": [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        "front": [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]],
        "side": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    }[approach]
    return candidate


def _record_handle_observation(memory: AgentMemory, *, task: str) -> None:
    memory.add_observation(
        EnvObservation(
            task=task,
            cameras=[
                CameraFrame(
                    frame_id="agentview",
                    rgb=[[[0, 0, 0]]],
                    extrinsics={
                        "pos": [0.0, 0.0, 0.0],
                        "mat": [0.0, 0.0, -1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
                    },
                )
            ],
            robot=RobotState(),
            metadata={
                "image_artifacts": [
                    {
                        "kind": "rgb",
                        "frame_id": "agentview",
                        "path": "tmp/agentview.rgb.png",
                    }
                ]
            },
        )
    )


def _memory_with_articulated_handle_candidates(
    *,
    geometry_family: str = "articulated_handle",
    task: str = "open the middle drawer of the cabinet",
    prompt: str = "middle drawer handle",
) -> AgentMemory:
    memory = AgentMemory()
    memory.start_session(task=task)
    _record_handle_observation(memory, task=task)
    memory.save_fact(
        "selected_sam3_detection",
        {
            "id": "detection_handle",
            "mask_ref": "tmp/handle.mask.png",
            "source_image": "tmp/agentview.rgb.png",
            "target_prompt": prompt,
            "target_geometry_family": geometry_family,
            "scene_epoch": memory.scene_epoch(),
        },
        source="test",
    )
    memory.add_action(
        _tool_action(
            "grasp_pose_estimate",
            {},
            outputs={
                "result_id": "handle-grasps",
                "mode": "targeted",
                "selected_backend": "anygrasp",
                "source_rgb": "tmp/agentview.rgb.png",
                "target_mask": "tmp/handle.mask.png",
                "camera_frame_id": "agentview",
                "source": {
                    "mode": "targeted",
                    "rgb": "tmp/agentview.rgb.png",
                    "object_mask": "tmp/handle.mask.png",
                    "camera_frame_id": "agentview",
                },
                "grasp_candidates": [
                    _candidate_with_approach("top-0", 0.95, "top_down"),
                    _candidate_with_approach("top-1", 0.90, "top_down"),
                    _candidate_with_approach("top-2", 0.85, "top_down"),
                    _candidate_with_approach("top-3", 0.80, "top_down"),
                    _candidate_with_approach("front-0", 0.75, "front"),
                    _candidate_with_approach("side-0", 0.70, "side"),
                ],
            },
        )
    )
    return memory


def _reject_active_candidate(memory: AgentMemory) -> None:
    policy = memory.grasp_candidate_policy()
    active = policy["active_candidate"]
    memory.add_action(
        _tool_action(
            "move_to",
            {"target_pose": {"source_grasp_id": active["id"]}},
            success=False,
            outputs={
                "motion_summary": {
                    "reached_target": False,
                    "end": {"xyz": [0.1, 0.2, 0.3]},
                }
            },
        )
    )


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


def _memory_at_articulated_close() -> AgentMemory:
    memory = AgentMemory()
    memory.start_session(task="open the microwave")
    memory.save_fact(
        "grasp_candidate_policy",
        {
            "status": "accepted",
            "interaction_family": "articulated_handle",
            "active_candidate": {"id": "handle-1", "rank": 0, "score": 0.8},
        },
        source="test",
    )
    memory.save_fact(
        "grasp_execution",
        {
            "schema_version": "openeta.grasp_execution.v1",
            "status": "required",
            "stage": "close",
            "candidate_id": "handle-1",
            "compiled_grasp_id": "compiled-1",
            "scene_epoch": memory.scene_epoch(),
            "required_action": {"name": "gripper_control", "parameters": {"position": 0}},
        },
        source="test",
    )
    return memory


def test_articulated_close_routes_to_prepare_probe_not_lift() -> None:
    memory = _memory_at_articulated_close()

    memory.add_action(_tool_action("gripper_control", {"position": 0}))

    execution = memory.grasp_execution()
    assert execution["stage"] == "prepare_probe"
    assert execution["probe_kind"] == "articulated_attachment"
    assert memory.grasp_lift_probe() is None


def test_articulated_probe_pass_keeps_endpoint_and_completes_attachment() -> None:
    memory = _memory_at_articulated_close()
    memory.add_action(_tool_action("gripper_control", {"position": 0}))
    required_action = {
        "name": "move_to",
        "parameters": {
            "target_pose": {
                "frame": "world",
                "xyz": [0.15, 0.2, 0.3],
                "probe_type": "articulated_attachment",
                "source_grasp_id": "handle-1",
                "compiled_grasp_id": "compiled-1",
                "scene_epoch": memory.scene_epoch(),
                "probe_path_sha256": "a" * 64,
            }
        },
    }
    prepared = {
        "schema_version": "openeta.articulated_attachment_probe.v1",
        "status": "prepared",
        "candidate_id": "handle-1",
        "compiled_grasp_id": "compiled-1",
        "scene_epoch": memory.scene_epoch(),
        "motion_type": "linear",
        "distance_m": 0.05,
        "path_sha256": "a" * 64,
        "required_action": required_action,
        "pre_probe_image_paths": ["before-agent.png", "before-wrist.png"],
    }
    memory.add_action(
        _tool_action("prepare_attachment_probe", {}, outputs=prepared)
    )
    assert memory.grasp_execution()["stage"] == "probe"

    memory.add_action(_tool_action(required_action["name"], required_action["parameters"]))

    assert memory.articulated_attachment_probe()["status"] == "completed"
    assert memory.grasp_execution()["stage"] == "attachment"
    assert "pass" not in memory.grasp_execution()["attachment_actions"]

    memory.add_action(
        _tool_action(
            "assess_attachment_probe",
            {},
            outputs={
                "schema_version": "openeta.articulated_attachment_assessment.v1",
                "candidate_id": "handle-1",
                "verdict": "PASS",
                "reason": "handle co-moved",
            },
        )
    )

    assert memory.attachment_gate()["verdict"] == "PASS"
    assert memory.grasp_execution()["status"] == "completed"
    assert memory.grasp_execution()["stage"] == "attached"


def _memory_at_articulated_attachment() -> tuple[AgentMemory, dict]:
    memory = _memory_at_articulated_close()
    memory.add_action(_tool_action("gripper_control", {"position": 0}))
    required_action = {
        "name": "move_to",
        "parameters": {
            "target_pose": {
                "frame": "world",
                "xyz": [0.15, 0.2, 0.3],
                "probe_type": "articulated_attachment",
                "source_grasp_id": "handle-1",
                "compiled_grasp_id": "compiled-1",
                "scene_epoch": memory.scene_epoch(),
                "probe_path_sha256": "b" * 64,
            },
            "enable_collision_check": True,
        },
    }
    memory.add_action(
        _tool_action(
            "prepare_attachment_probe",
            {},
            outputs={
                "schema_version": "openeta.articulated_attachment_probe.v1",
                "status": "prepared",
                "candidate_id": "handle-1",
                "compiled_grasp_id": "compiled-1",
                "scene_epoch": memory.scene_epoch(),
                "motion_type": "linear",
                "distance_m": 0.05,
                "path_sha256": "b" * 64,
                "required_action": required_action,
                "pre_probe_image_paths": ["before-agent.png", "before-wrist.png"],
            },
        )
    )
    memory.add_action(_tool_action(required_action["name"], required_action["parameters"]))
    return memory, required_action


def test_articulated_unknown_allows_one_fresh_observation_then_stays_bounded() -> None:
    memory, _ = _memory_at_articulated_attachment()
    unknown = {
        "schema_version": "openeta.articulated_attachment_assessment.v1",
        "candidate_id": "handle-1",
        "verdict": "UNKNOWN",
        "reason": "wrist view is occluded",
    }
    memory.add_action(_tool_action("assess_attachment_probe", {}, outputs=unknown))
    assert memory.attachment_gate()["refresh_required"] is True

    memory.add_action(_tool_action("observe", {}))
    memory.add_observation(
        EnvObservation(task="open the microwave", cameras=[], robot=RobotState())
    )
    gate = memory.attachment_gate()
    assert gate["unknown_refresh_completed"] is True
    assert gate["refresh_required"] is False

    memory.add_action(_tool_action("assess_attachment_probe", {}, outputs=unknown))
    gate = memory.attachment_gate()
    assert gate["assessment_count"] == 2
    assert gate["verdict"] == "UNKNOWN"
    assert gate["refresh_required"] is False


def test_articulated_fail_open_advances_candidate_without_reviewer_metadata() -> None:
    memory, _ = _memory_at_articulated_attachment()
    policy = memory.grasp_candidate_policy()
    policy["candidates"] = [
        {"id": "handle-1", "rank": 0, "score": 0.8},
        {"id": "handle-2", "rank": 1, "score": 0.7},
    ]
    policy["approach_mode_order"] = ["front"]
    policy["approach_mode_queues"] = {"front": ["handle-1", "handle-2"]}
    policy["active_approach_mode"] = "front"
    policy["active_mode_rank"] = 0
    policy["mode_attempt_count"] = 0
    policy["mode_max_candidate_attempts"] = 3
    memory.save_fact("grasp_candidate_policy", policy, source="test")
    memory.add_action(
        _tool_action(
            "assess_attachment_probe",
            {},
            outputs={
                "schema_version": "openeta.articulated_attachment_assessment.v1",
                "candidate_id": "handle-1",
                "verdict": "FAIL",
                "reason": "handle stayed fixed while the gripper moved",
            },
        )
    )

    memory.add_action(_tool_action("gripper_control", {"position": 1}))

    assert memory.grasp_candidate_policy()["active_candidate"]["id"] == "handle-2"
    assert memory.grasp_execution() is None
    assert memory.articulated_attachment_probe() is None


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


def test_exhausted_motion_candidate_queue_requires_fresh_reestimation() -> None:
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
    assert policy["status"] == "exhausted"
    assert policy["active_candidate"] is None
    assert policy["reestimate_required"]["status"] == "pending_recovery"
    assert memory.grasp_reestimation()["status"] == "pending_recovery"
    assert memory.grasp_recovery()["status"] == "required"


def test_candidate_attempt_limit_requires_fresh_reestimation() -> None:
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
    assert policy["status"] == "exhausted"
    assert policy["candidate_attempt_count"] == 3
    assert policy["active_candidate"] is None
    assert policy["reestimate_required"]["reason"] == "candidate_retry_limit_exceeded"
    assert memory.grasp_reestimation()["status"] == "pending_recovery"
    assert memory.grasp_recovery()["status"] == "required"


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


def test_articulated_handle_uses_bounded_mode_queues_then_global_fallback() -> None:
    memory = _memory_with_articulated_handle_candidates()
    policy = memory.grasp_candidate_policy()

    assert policy["interaction_family"] == "articulated_handle"
    assert policy["approach_mode_order"] == ["top_down", "front", "side"]
    assert policy["active_candidate"]["id"] == "top-0"
    assert policy["compile_hints"] == {
        "target_geometry_family": "articulated_handle",
        "approach_mode": "top_down",
        "strategy_id": "top-down-drawer-handle-panda-p8",
    }

    _reject_active_candidate(memory)
    _reject_active_candidate(memory)
    _reject_active_candidate(memory)
    policy = memory.grasp_candidate_policy()
    assert policy["active_candidate"]["id"] == "front-0"
    assert policy["active_approach_mode"] == "front"
    assert "top-3" not in {
        item["candidate_id"] for item in policy["rejected_candidates"]
    }
    assert policy["compile_hints"]["strategy_id"] == (
        "native-front-articulated-handle-panda-p8"
    )

    _reject_active_candidate(memory)
    policy = memory.grasp_candidate_policy()
    assert policy["active_candidate"]["id"] == "side-0"
    assert policy["compile_hints"]["strategy_id"] == (
        "native-side-articulated-handle-panda-p8"
    )
    _reject_active_candidate(memory)
    policy = memory.grasp_candidate_policy()
    assert policy["candidate_fallback"] is True
    assert policy["fallback_reason"] == "all_approach_modes_failed"
    assert policy["active_candidate"]["id"] == "top-0"
    assert policy["active_candidate"]["candidate_fallback"] is True

    _reject_active_candidate(memory)
    policy = memory.grasp_candidate_policy()
    assert policy["status"] == "exhausted"
    assert policy["reestimate_required"]["reason"] == (
        "articulated_handle_fallback_failed"
    )
    assert memory.grasp_reestimation()["status"] == "pending_recovery"


def test_legacy_drawer_handle_and_conservative_task_fallback_enable_modes() -> None:
    legacy = _memory_with_articulated_handle_candidates(geometry_family="drawer_handle")
    assert legacy.grasp_candidate_policy()["interaction_family"] == "articulated_handle"

    fallback = _memory_with_articulated_handle_candidates(
        geometry_family="unknown",
        task="open the microwave",
        prompt="microwave door handle",
    )
    assert fallback.grasp_candidate_policy()["interaction_family"] == "articulated_handle"


def test_non_articulated_tasks_and_non_handle_targets_keep_original_candidate_policy() -> None:
    cases = [
        ("open the bottle", "bottle cap", "other"),
        ("pull the basket closer", "basket handle", "other"),
        ("put the bowl inside the open microwave", "black bowl", "bowl"),
    ]
    for task, prompt, family in cases:
        memory = _memory_with_articulated_handle_candidates(
            geometry_family=family,
            task=task,
            prompt=prompt,
        )
        policy = memory.grasp_candidate_policy()
        assert "interaction_family" not in policy
        assert policy["active_candidate"]["id"] == "top-0"
        assert policy["compile_hints"] == {"target_geometry_family": family}


def test_explicit_articulated_label_does_not_override_portable_handle_exclusion() -> None:
    memory = _memory_with_articulated_handle_candidates(
        geometry_family="articulated_handle",
        task="put the mug in the cabinet",
        prompt="mug handle",
    )

    assert "interaction_family" not in memory.grasp_candidate_policy()


def test_missing_matching_extrinsics_fails_closed_to_original_policy() -> None:
    memory = AgentMemory()
    memory.start_session(task="open the middle drawer")
    memory.save_fact(
        "selected_sam3_detection",
        {
            "id": "detection_handle",
            "mask_ref": "tmp/handle.mask.png",
            "source_image": "tmp/agentview.rgb.png",
            "target_prompt": "drawer handle",
            "target_geometry_family": "articulated_handle",
            "scene_epoch": memory.scene_epoch(),
        },
        source="test",
    )
    memory.add_action(
        _tool_action(
            "grasp_pose_estimate",
            {},
            outputs={
                "result_id": "handle-grasps-no-extrinsics",
                "mode": "targeted",
                "selected_backend": "anygrasp",
                "source_rgb": "tmp/agentview.rgb.png",
                "target_mask": "tmp/handle.mask.png",
                "camera_frame_id": "agentview",
                "grasp_candidates": [
                    _candidate_with_approach("top-0", 0.95, "top_down")
                ],
            },
        )
    )

    policy = memory.grasp_candidate_policy()
    assert "interaction_family" not in policy
    assert policy["compile_hints"] == {
        "target_geometry_family": "articulated_handle"
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
        "pregrasp_distance_m": 0.15,
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
