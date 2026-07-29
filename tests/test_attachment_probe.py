from __future__ import annotations

import math

import pytest

from adapter.protocol import CameraFrame, EnvObservation, RobotState
from agent.backends.planner import PlannerBackendResult, StaticPlannerBackend
from agent.backends.planner import CallablePlannerBackend
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
            }
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
            CameraFrame(
                frame_id="zed_head",
                role="scene_primary",
                rgb=[],
            ),
            CameraFrame(
                frame_id="wrist_left",
                role="wrist_secondary",
                rgb=[],
            ),
            CameraFrame(
                frame_id="wrist_right",
                role="wrist_primary",
                rgb=[],
            ),
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


def _memory_context(*, stage: str = "prepare_probe") -> dict:
    return {
        "memory": {
            "scene_epoch": 4,
            "grasp_execution": {
                "status": "required",
                "stage": stage,
                "candidate_id": "handle-1",
                "compiled_grasp_id": "compiled-1",
                "scene_epoch": 4,
                "attachment_mode": "articulated_handle" if stage == "attachment" else None,
            },
            "grasp_candidate_policy": {
                "interaction_family": "articulated_handle",
                "active_candidate": {"id": "handle-1"},
            },
        }
    }


def test_prepare_linear_probe_freezes_exact_five_centimetres() -> None:
    result = prepare_attachment_probe(
        {
            "motion_type": "linear",
            "direction_world_xyz": [2.0, 0.0, 0.0],
            "reason": "drawer front moves toward camera",
        },
        observation=_observation(),
        supervision_context=_memory_context(),
    )

    assert result["motion_type"] == "linear"
    assert result["required_action"]["name"] == "move_to"
    endpoint = result["required_action"]["parameters"]["target_pose"]["xyz"]
    assert endpoint == pytest.approx([0.15, 0.2, 0.3])
    assert result["distance_m"] == ARTICULATED_ATTACHMENT_PROBE_DISTANCE_M
    assert result["pre_probe_image_paths"] == ["before-agent.png", "before-wrist.png"]
    assert result["required_action"]["parameters"]["enable_collision_check"] is True
    assert len(result["path_sha256"]) == 64


def test_prepare_linear_probe_preserves_quaternion_orientation() -> None:
    observation = _observation()
    observation.robot.end_effector_pose = {
        "xyz": [0.1, 0.2, 0.3],
        "quat_xyzw": [0.0, 0.0, 0.0, 2.0],
    }

    result = prepare_attachment_probe(
        {"motion_type": "linear", "direction_world_xyz": [1.0, 0.0, 0.0]},
        observation=observation,
        supervision_context=_memory_context(),
    )

    assert result["required_action"]["parameters"]["target_pose"]["quat_xyzw"] == [
        0.0,
        0.0,
        0.0,
        1.0,
    ]


def test_prepare_probe_uses_backend_neutral_scene_and_wrist_roles() -> None:
    result = prepare_attachment_probe(
        {"motion_type": "linear", "direction_world_xyz": [1.0, 0.0, 0.0]},
        observation=_role_observation(),
        supervision_context=_memory_context(),
    )

    assert result["pre_probe_image_paths"] == [
        "before-zed.png",
        "before-right-wrist.png",
    ]


def test_prepare_arc_probe_preserves_waypoints_and_bounds() -> None:
    result = prepare_attachment_probe(
        {
            "motion_type": "arc",
            "waypoint_offsets_world_xyz": [
                [0.0125, 0.0, 0.0],
                [0.025, 0.0, 0.0],
                [0.0375, 0.0, 0.0],
                [0.05, 0.0, 0.0],
            ],
            "reason": "local door arc",
        },
        observation=_observation(),
        supervision_context=_memory_context(),
    )

    assert result["required_action"]["name"] == "follow_eef_trajectory"
    trajectory = result["required_action"]["parameters"]["trajectory"]
    assert len(trajectory) == 4
    assert trajectory[-1]["xyz"] == pytest.approx([0.15, 0.2, 0.3])
    assert all(
        pose["probe_path_sha256"] == result["path_sha256"] for pose in trajectory
    )


@pytest.mark.parametrize(
    "offsets",
    [
        [[0.025, 0.0, 0.0], [0.05, 0.0, 0.0]],
        [[0.01, 0.0, 0.0], [0.02, 0.0, 0.0]],
    ],
)
def test_prepare_arc_probe_rejects_long_segment_or_wrong_total(offsets) -> None:
    with pytest.raises(AttachmentProbeError):
        prepare_attachment_probe(
            {
                "motion_type": "arc",
                "waypoint_offsets_world_xyz": offsets,
                "reason": "invalid",
            },
            observation=_observation(),
            supervision_context=_memory_context(),
        )


def test_prepare_probe_rejects_non_articulated_policy() -> None:
    context = _memory_context()
    context["memory"]["grasp_candidate_policy"]["interaction_family"] = "portable_object"
    with pytest.raises(AttachmentProbeError, match="not an articulated handle"):
        prepare_attachment_probe(
            {"motion_type": "linear", "direction_world_xyz": [1, 0, 0]},
            observation=_observation(),
            supervision_context=context,
        )


def test_prepare_probe_requires_agentview_and_wrist_rgb() -> None:
    observation = _observation()
    observation.metadata["image_artifacts"] = [
        {"kind": "rgb", "frame_id": "agentview", "path": "before-agent.png"}
    ]
    with pytest.raises(AttachmentProbeError, match="agentview and wrist"):
        prepare_attachment_probe(
            {"motion_type": "linear", "direction_world_xyz": [1, 0, 0]},
            observation=observation,
            supervision_context=_memory_context(),
        )


def test_assessment_uses_before_and_after_multiview() -> None:
    probe = prepare_attachment_probe(
        {"motion_type": "linear", "direction_world_xyz": [1, 0, 0]},
        observation=_observation(),
        supervision_context=_memory_context(),
    )
    memory = _memory_context(stage="attachment")["memory"]
    memory["articulated_attachment_probe"] = {**probe, "status": "completed"}
    memory["attachment_gate"] = {
        "status": "pending",
        "verdict": "UNKNOWN",
        "assessment_count": 0,
    }
    after = _observation()
    after.metadata["image_artifacts"] = [
        {"kind": "rgb", "frame_id": "agentview", "path": "after-agent.png"},
        {"kind": "rgb", "frame_id": "wrist", "path": "after-wrist.png"},
    ]
    tools = build_default_tool_registry()
    context = ToolExecutionContext(
        name="assess_attachment_probe",
        spec=tools.get("assess_attachment_probe"),
        observation=after,
        metadata={"task": after.task, "supervision_context": {"memory": memory}},
    )
    requests = []

    def decide(request):
        requests.append(request)
        return PlannerBackendResult(
            payload={
                "verdict": "PASS",
                "reason": "handle co-moved and remains between fingers",
            },
            provider="unit",
            model="attachment-reviewer",
        )

    backend = CallablePlannerBackend(decide)

    result = assess_attachment_probe(context, backend=backend)

    assert result["verdict"] == "PASS"
    assert result["checked_by"] == "independent_attachment_reviewer"
    assert requests[0].tool_context["vision_image_paths"] == [
        "before-agent.png",
        "before-wrist.png",
        "after-agent.png",
        "after-wrist.png",
    ]
    assert [row["role"] for row in requests[0].tool_context["image_order"]] == [
        "before_agentview",
        "before_wrist",
        "after_agentview",
        "after_wrist",
    ]
    assert requests[0].system_prompt == ARTICULATED_ATTACHMENT_ASSESSMENT_PROMPT
    assert "before/after agentview and wrist images" in requests[0].system_prompt


def test_assessment_fails_closed_without_after_wrist_rgb() -> None:
    probe = prepare_attachment_probe(
        {"motion_type": "linear", "direction_world_xyz": [1, 0, 0]},
        observation=_observation(),
        supervision_context=_memory_context(),
    )
    memory = _memory_context(stage="attachment")["memory"]
    memory["articulated_attachment_probe"] = {**probe, "status": "completed"}
    memory["attachment_gate"] = {
        "status": "pending",
        "verdict": "UNKNOWN",
        "assessment_count": 0,
    }
    after = _observation()
    after.metadata["image_artifacts"] = [
        {"kind": "rgb", "frame_id": "agentview", "path": "after-agent.png"}
    ]
    tools = build_default_tool_registry()
    context = ToolExecutionContext(
        name="assess_attachment_probe",
        spec=tools.get("assess_attachment_probe"),
        observation=after,
        metadata={"task": after.task, "supervision_context": {"memory": memory}},
    )

    with pytest.raises(AttachmentProbeError, match="agentview and one wrist"):
        assess_attachment_probe(
            context,
            backend=StaticPlannerBackend({"verdict": "PASS", "reason": "unused"}),
        )


def test_assessment_rejects_calls_after_two_attempts() -> None:
    probe = prepare_attachment_probe(
        {"motion_type": "linear", "direction_world_xyz": [1, 0, 0]},
        observation=_observation(),
        supervision_context=_memory_context(),
    )
    memory = _memory_context(stage="attachment")["memory"]
    memory["articulated_attachment_probe"] = {**probe, "status": "completed"}
    memory["attachment_gate"] = {
        "status": "pending",
        "verdict": "UNKNOWN",
        "assessment_count": 2,
        "unknown_refresh_completed": True,
    }
    context = ToolExecutionContext(
        name="assess_attachment_probe",
        spec=build_default_tool_registry().get("assess_attachment_probe"),
        observation=_observation(),
        metadata={"supervision_context": {"memory": memory}},
    )

    with pytest.raises(AttachmentProbeError, match="budget is exhausted"):
        assess_attachment_probe(
            context,
            backend=StaticPlannerBackend({"verdict": "PASS", "reason": "unused"}),
        )
