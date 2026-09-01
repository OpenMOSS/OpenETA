from __future__ import annotations

import pytest

from adapter.protocol import CameraFrame, EnvObservation, RobotState
from agent.backends.planner import CallablePlannerBackend, PlannerBackendResult, StaticPlannerBackend
from agent.tools.attachment_probe import (
    ARTICULATED_ATTACHMENT_ASSESSMENT_PROMPT,
    ARTICULATED_ATTACHMENT_PROBE_DISTANCE_M,
    AttachmentProbeError,
    assess_attachment_probe,
    prepare_attachment_probe,
)
from agent.tools.registry import ToolExecutionContext, build_default_tool_registry


def _observation() -> EnvObservation:
    return EnvObservation(
        task="open the microwave",
        cameras=[
            CameraFrame(frame_id="agentview", rgb=[]),
            CameraFrame(frame_id="wrist", rgb=[]),
        ],
        robot=RobotState(
            end_effector_pose={
                "xyz": [0.1, 0.2, 0.3],
                "rotation_matrix": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            },
            gripper_state={"open": False, "openness": 0.4},
        ),
        metadata={
            "image_artifacts": [
                {"kind": "rgb", "frame_id": "agentview", "path": "before-agent.png"},
                {"kind": "rgb", "frame_id": "wrist", "path": "before-wrist.png"},
            ]
        },
    )


def _role_observation() -> EnvObservation:
    return EnvObservation(
        task="open the cabinet",
        cameras=[
            CameraFrame(frame_id="zed_head", role="scene_primary", rgb=[]),
            CameraFrame(frame_id="wrist_left", role="wrist_secondary", rgb=[]),
            CameraFrame(frame_id="wrist_right", role="wrist_primary", rgb=[]),
        ],
        robot=_observation().robot,
        metadata={
            "image_artifacts": [
                {
                    "kind": "rgb",
                    "frame_id": "zed_head",
                    "role": "scene_primary",
                    "path": "before-zed.png",
                },
                {
                    "kind": "rgb",
                    "frame_id": "wrist_left",
                    "role": "wrist_secondary",
                    "path": "before-left-wrist.png",
                },
                {
                    "kind": "rgb",
                    "frame_id": "wrist_right",
                    "role": "wrist_primary",
                    "path": "before-right-wrist.png",
                },
            ]
        },
    )


def _memory_context(*, freshness: str = "current_object_scene") -> dict:
    return {
        "memory": {
            "scene_epoch": 4,
            "gripper_command_state": {
                "position": 0,
                "state": "closed",
                "attachment_proxy_receipt": {
                    "schema_version": "openeta.attachment_proxy_receipt.v1",
                    "status": "tentative",
                    "reason": "non_empty_close_with_tentative_safety_proxy",
                },
            },
            "provenance_evidence_graph": {
                "schema_version": "openeta.provenance_evidence_graph.v1",
                "nodes": [
                    {
                        "kind": "compiled_targeted_grasp",
                        "compiled_grasp_id": "compiled-1",
                        "candidate_id": "handle-1",
                        "freshness": freshness,
                    }
                ],
                "edges": [],
            },
        }
    }


def _prepare(parameters: dict, *, observation: EnvObservation | None = None) -> dict:
    return prepare_attachment_probe(
        {"compiled_grasp_id": "compiled-1", **parameters},
        observation=observation or _observation(),
        supervision_context=_memory_context(),
    )


def test_prepare_linear_probe_freezes_exact_five_centimetres() -> None:
    result = _prepare(
        {
            "motion_type": "linear",
            "direction_world_xyz": [2.0, 0.0, 0.0],
            "reason": "drawer front moves toward camera",
        }
    )

    assert result["motion_type"] == "linear"
    assert result["frozen_motion"]["name"] == "move_to"
    endpoint = result["frozen_motion"]["parameters"]["target_pose"]["xyz"]
    assert endpoint == pytest.approx([0.15, 0.2, 0.3])
    assert result["distance_m"] == ARTICULATED_ATTACHMENT_PROBE_DISTANCE_M
    assert result["pre_probe_image_paths"] == ["before-agent.png", "before-wrist.png"]
    assert result["frozen_motion"]["parameters"]["enable_collision_check"] is True
    assert result["ik_preview_requests"] == [
        {
            "tool": "ik_preview_check",
            "parameters": {
                "probe_id": result["probe_id"],
                "waypoint_index": 0,
                "position_tolerance_m": 0.01,
                "orientation_tolerance_rad": 0.10,
                "check_endpoint_collision": True,
            },
        }
    ]
    assert result["execution_handoff"]["tool"] == "move_to"
    assert set(result["execution_handoff"]["parameters"]) == {
        "ik_receipt_id",
        "enable_collision_check",
    }
    assert result["probe_id"] == f"probe:{result['path_sha256']}"


def test_prepare_linear_probe_preserves_quaternion_orientation() -> None:
    observation = _observation()
    observation.robot.end_effector_pose = {
        "xyz": [0.1, 0.2, 0.3],
        "quat_xyzw": [0.0, 0.0, 0.0, 2.0],
    }

    result = _prepare(
        {"motion_type": "linear", "direction_world_xyz": [1.0, 0.0, 0.0]},
        observation=observation,
    )

    assert result["frozen_motion"]["parameters"]["target_pose"]["quat_xyzw"] == [
        0.0,
        0.0,
        0.0,
        1.0,
    ]


def test_prepare_probe_uses_backend_neutral_scene_and_wrist_roles() -> None:
    result = _prepare(
        {"motion_type": "linear", "direction_world_xyz": [1.0, 0.0, 0.0]},
        observation=_role_observation(),
    )
    assert result["pre_probe_image_paths"] == ["before-zed.png", "before-right-wrist.png"]


def test_prepare_arc_probe_preserves_waypoints_and_bounds() -> None:
    result = _prepare(
        {
            "motion_type": "arc",
            "waypoint_offsets_world_xyz": [
                [0.0125, 0.0, 0.0],
                [0.025, 0.0, 0.0],
                [0.0375, 0.0, 0.0],
                [0.05, 0.0, 0.0],
            ],
            "reason": "local door arc",
        }
    )

    assert result["frozen_motion"]["name"] == "follow_eef_trajectory"
    trajectory = result["frozen_motion"]["parameters"]["trajectory"]
    assert len(trajectory) == 4
    assert trajectory[-1]["xyz"] == pytest.approx([0.15, 0.2, 0.3])
    assert all(pose["probe_path_sha256"] == result["path_sha256"] for pose in trajectory)
    assert [request["parameters"]["probe_id"] for request in result["ik_preview_requests"]] == [
        result["probe_id"]
    ] * 4
    assert [
        request["parameters"]["waypoint_index"]
        for request in result["ik_preview_requests"]
    ] == [0, 1, 2, 3]
    assert result["execution_handoff"]["tool"] == "follow_eef_trajectory"
    assert len(result["execution_handoff"]["parameters"]["ik_receipt_ids"]) == 4


@pytest.mark.parametrize(
    "offsets",
    [
        [[0.025, 0.0, 0.0], [0.05, 0.0, 0.0]],
        [[0.01, 0.0, 0.0], [0.02, 0.0, 0.0]],
    ],
)
def test_prepare_arc_probe_rejects_long_segment_or_wrong_total(offsets) -> None:
    with pytest.raises(AttachmentProbeError):
        _prepare(
            {
                "motion_type": "arc",
                "waypoint_offsets_world_xyz": offsets,
                "reason": "invalid",
            }
        )


def test_prepare_probe_requires_explicit_current_compiled_grasp_evidence() -> None:
    parameters = {
        "compiled_grasp_id": "missing",
        "motion_type": "linear",
        "direction_world_xyz": [1, 0, 0],
    }
    with pytest.raises(AttachmentProbeError, match="provenance_evidence_graph"):
        prepare_attachment_probe(
            parameters,
            observation=_observation(),
            supervision_context=_memory_context(),
        )

    with pytest.raises(AttachmentProbeError, match="superseded"):
        prepare_attachment_probe(
            {**parameters, "compiled_grasp_id": "compiled-1"},
            observation=_observation(),
            supervision_context=_memory_context(freshness="superseded_target_evidence"),
        )


def test_prepare_probe_accepts_same_lineage_after_gripper_close_epoch_change() -> None:
    result = prepare_attachment_probe(
        {
            "compiled_grasp_id": "compiled-1",
            "motion_type": "linear",
            "direction_world_xyz": [1, 0, 0],
        },
        observation=_observation(),
        supervision_context=_memory_context(freshness="stale_object_scene"),
    )

    assert result["compiled_grasp_id"] == "compiled-1"
    assert result["frozen_motion"]["name"] == "move_to"


def test_prepare_probe_requires_scene_and_wrist_rgb() -> None:
    observation = _observation()
    observation.metadata["image_artifacts"] = [
        {"kind": "rgb", "frame_id": "agentview", "path": "before-agent.png"}
    ]
    with pytest.raises(AttachmentProbeError, match="agentview and wrist"):
        _prepare(
            {"motion_type": "linear", "direction_world_xyz": [1, 0, 0]},
            observation=observation,
        )


def test_prepare_probe_rejects_open_or_empty_close_gripper_evidence() -> None:
    parameters = {
        "compiled_grasp_id": "compiled-1",
        "motion_type": "linear",
        "direction_world_xyz": [0, 0, 1],
    }
    opened = _observation()
    opened.robot.gripper_state = {"open": True, "openness": 0.998}
    with pytest.raises(AttachmentProbeError, match="measured gripper state"):
        prepare_attachment_probe(
            parameters,
            observation=opened,
            supervision_context=_memory_context(),
        )

    memory = _memory_context()
    memory["memory"]["gripper_command_state"]["attachment_proxy_receipt"] = {
        "status": "not_armed",
        "reason": "empty_close_or_no_measurable_contact",
    }
    with pytest.raises(AttachmentProbeError, match="tentative non-empty close receipt"):
        prepare_attachment_probe(
            parameters,
            observation=_observation(),
            supervision_context=memory,
        )


def test_prepare_probe_prefers_continuous_aperture_over_coarse_open_flag() -> None:
    observation = _observation()
    observation.robot.gripper_state = {"open": True, "openness": 0.5258}

    result = prepare_attachment_probe(
        {
            "compiled_grasp_id": "compiled-1",
            "motion_type": "linear",
            "direction_world_xyz": [0, 0, 1],
        },
        observation=observation,
        supervision_context=_memory_context(),
    )

    assert result["gripper_evidence"]["measured_open"] is True
    assert result["gripper_evidence"]["measured_openness"] == pytest.approx(0.5258)


def _assessment_context(probe: dict, observation: EnvObservation) -> ToolExecutionContext:
    memory = _memory_context()["memory"]
    memory["articulated_attachment_probe"] = {**probe, "status": "completed"}
    tools = build_default_tool_registry()
    return ToolExecutionContext(
        name="assess_attachment_probe",
        spec=tools.get("assess_attachment_probe"),
        parameters={"probe_id": probe["probe_id"]},
        observation=observation,
        metadata={"task": observation.task, "supervision_context": {"memory": memory}},
    )


def test_assessment_uses_explicit_probe_and_before_after_multiview() -> None:
    probe = _prepare({"motion_type": "linear", "direction_world_xyz": [1, 0, 0]})
    after = _observation()
    after.metadata["image_artifacts"] = [
        {"kind": "rgb", "frame_id": "agentview", "path": "after-agent.png"},
        {"kind": "rgb", "frame_id": "wrist", "path": "after-wrist.png"},
    ]
    context = _assessment_context(probe, after)
    requests = []

    def decide(request):
        requests.append(request)
        return PlannerBackendResult(
            payload={"verdict": "PASS", "reason": "handle co-moved"},
            provider="unit",
            model="attachment-reviewer",
        )

    result = assess_attachment_probe(context, backend=CallablePlannerBackend(decide))

    assert result["verdict"] == "PASS"
    assert result["probe_id"] == probe["probe_id"]
    assert requests[0].tool_context["vision_image_paths"] == [
        "before-agent.png",
        "before-wrist.png",
        "after-agent.png",
        "after-wrist.png",
    ]
    assert requests[0].system_prompt == ARTICULATED_ATTACHMENT_ASSESSMENT_PROMPT


def test_assessment_rejects_wrong_probe_id_and_missing_after_view() -> None:
    probe = _prepare({"motion_type": "linear", "direction_world_xyz": [1, 0, 0]})
    context = _assessment_context(probe, _observation())
    context.parameters["probe_id"] = "probe:wrong"
    with pytest.raises(AttachmentProbeError, match="probe_id"):
        assess_attachment_probe(
            context,
            backend=StaticPlannerBackend({"verdict": "PASS", "reason": "unused"}),
        )

    after = _observation()
    after.metadata["image_artifacts"] = [
        {"kind": "rgb", "frame_id": "agentview", "path": "after-agent.png"}
    ]
    context = _assessment_context(probe, after)
    with pytest.raises(AttachmentProbeError, match="agentview and one wrist"):
        assess_attachment_probe(
            context,
            backend=StaticPlannerBackend({"verdict": "PASS", "reason": "unused"}),
        )


def test_assessment_rejects_visual_pass_when_gripper_is_measured_open() -> None:
    probe = _prepare({"motion_type": "linear", "direction_world_xyz": [0, 0, 1]})
    after = _observation()
    after.robot.gripper_state = {"open": True, "openness": 0.998}
    context = _assessment_context(probe, after)

    with pytest.raises(AttachmentProbeError, match="measured gripper state"):
        assess_attachment_probe(
            context,
            backend=StaticPlannerBackend(
                {"verdict": "PASS", "reason": "must not override proprioception"}
            ),
        )
