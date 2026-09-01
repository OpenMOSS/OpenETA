from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from adapter.protocol import CameraFrame, EnvAction, EnvObservation, RobotState
from agent.runtime.actions import PipelineStatus
from agent.runtime.memory import (
    ARTICULATED_ATTACHMENT_PROBE_KEY,
    ATTACHMENT_EVIDENCE_KEY,
    AgentMemory,
    GRIPPER_COMMAND_STATE_KEY,
    GRASP_PROVENANCE_KEY,
    ROBOT_MOTION_EPOCH_KEY,
)
from agent.runtime.pipeline import ActionPipeline
from agent.runtime.planner import (
    PlannerDecision,
    _validate_compiled_grasp_target_freshness,
    build_tool_context,
)
from agent.runtime.skills import (
    SkillRegistry,
    SkillSpec,
    build_default_skill_registry,
    lint_skill_contracts,
)
from agent.tools.registry import (
    ToolExecutionContext,
    ToolResult,
    build_default_tool_registry,
    make_tool_result,
)
from agent.tools.sim_mcp import (
    SimulatorMcpToolProxyConfig,
    bind_simulator_mcp_tool_handlers,
)


def _observation() -> EnvObservation:
    return EnvObservation(
        task="pick the cube",
        cameras=[
            CameraFrame(
                frame_id="agentview",
                role="scene_primary",
                rgb=[[[0, 0, 0]]],
                depth=[[1.0]],
                intrinsics={
                    "fx": 100.0,
                    "fy": 100.0,
                    "cx": 0.5,
                    "cy": 0.5,
                    "scale": 1000,
                },
            )
        ],
        robot=RobotState(gripper_state={"open": True}),
        metadata={
            "step_idx": 4,
            "image_artifacts": [
                {
                    "kind": "rgb",
                    "frame_id": "agentview",
                    "role": "scene_primary",
                    "path": "/session/rgb.png",
                    "packet_id": "packet-4",
                },
                {
                    "kind": "depth",
                    "frame_id": "agentview",
                    "role": "scene_primary",
                    "path": "/session/depth.png",
                    "packet_id": "packet-4",
                },
            ],
        },
    )


def _record_ik_receipt(
    memory: AgentMemory,
    *,
    receipt_id: str,
    target_pose: dict,
    classification: str = "feasible",
    captured_quat_xyzw: list[float] | None = None,
    joint_positions: list[float] | None = None,
) -> None:
    """Record one host-owned IK handoff for Agent-facing move_to tests."""

    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "ik_preview_check",
                    "parameters": {"target_pose": dict(target_pose)},
                },
                "status": "executed",
                "tool_calls": [
                    {
                        "name": "ik_preview_check",
                        "status": "executed",
                        "parameters": {"target_pose": dict(target_pose)},
                        "result": {
                            "success": True,
                            "details": {
                                "outputs": {
                                    "ik_preview_receipt": {
                                        "schema_version": "openeta.ik_preview_receipt.v1",
                                        "receipt_id": receipt_id,
                                        "classification": classification,
                                        "target_pose": dict(target_pose),
                                        "orientation_policy": (
                                            "explicit_orientation"
                                            if "rotation_matrix" in target_pose
                                            else "preserve_current"
                                        ),
                                        "tolerances": {
                                            "position_tolerance_m": 0.01,
                                            "orientation_tolerance_rad": 0.1,
                                        },
                                        **(
                                            {
                                                "reachability": {
                                                    "target": {
                                                        "xyz": list(target_pose["xyz"]),
                                                        "quat_xyzw": list(
                                                            captured_quat_xyzw
                                                        ),
                                                    },
                                                },
                                                "best_candidate": {
                                                    "joint_positions": list(
                                                        joint_positions or []
                                                    )
                                                },
                                            }
                                            if captured_quat_xyzw is not None
                                            else {}
                                        ),
                                    }
                                }
                            },
                        },
                    }
                ],
            },
        )
    )


def test_compiled_grasp_waypoint_is_host_resolved_for_ik_preview() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    pose = {
        "frame": "world",
        "xyz": [0.1, 0.2, 0.3],
        "rotation_matrix": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "compiled_grasp_id": "compiled-ref",
        "waypoint_role": "grasp_clearance",
    }
    memory.save_artifact(
        "compiled-ref",
        {
            "type": "compiled_grasp",
            "compiled_grasp_id": "compiled-ref",
            "scene_epoch": 0,
            "hover_pose": dict(pose),
        },
        source="compile_grasp_seed",
    )
    tools = build_default_tool_registry()
    dispatched: list[dict] = []

    def record_ik(context: ToolExecutionContext) -> ToolResult:
        dispatched.append(dict(context.parameters))
        return ToolResult(True)

    tools.bind_handler("ik_preview_check", record_ik)

    plan = ActionPipeline().compile(
        PlannerDecision(
            action_type="tool_call",
            action="ik_preview_check",
            parameters={
                "compiled_grasp_id": "compiled-ref",
                "waypoint_role": "grasp_clearance",
                "check_endpoint_collision": True,
            },
        ),
        observation=_observation(),
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )

    assert plan.status is PipelineStatus.EXECUTED
    assert dispatched[0]["target_pose"] == pose
    assert dispatched[0]["check_endpoint_collision"] is True
    assert "target_pose" not in plan.request.parameters


def test_agent_chosen_compiled_path_sample_preserves_full_grasp_pose() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    rotation = [
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
    ]
    memory.save_artifact(
        "compiled-path-ref",
        {
            "type": "compiled_grasp",
            "compiled_grasp_id": "compiled-path-ref",
            "scene_epoch": 0,
            "hover_pose": {
                "frame": "world",
                "xyz": [0.1, 0.2, 0.3],
                "rotation_matrix": rotation,
                "compiled_grasp_id": "compiled-path-ref",
                "waypoint_role": "grasp_clearance",
            },
            "contact_pose": {
                "frame": "world",
                "xyz": [0.2, 0.1, 0.1],
                "rotation_matrix": rotation,
                "compiled_grasp_id": "compiled-path-ref",
                "waypoint_role": "grasp_contact",
            },
        },
        source="compile_grasp_seed",
    )
    tools = build_default_tool_registry()
    dispatched: list[dict] = []
    tools.bind_handler(
        "ik_preview_check",
        lambda context: dispatched.append(dict(context.parameters)) or ToolResult(True),
    )

    plan = ActionPipeline().compile(
        PlannerDecision(
            action_type="tool_call",
            action="ik_preview_check",
            parameters={
                "compiled_grasp_id": "compiled-path-ref",
                "path_fraction": 0.8,
                "position_tolerance_m": 0.005,
                "orientation_tolerance_rad": 0.1,
                "check_endpoint_collision": True,
            },
        ),
        observation=_observation(),
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )

    assert plan.status is PipelineStatus.EXECUTED
    target = dispatched[0]["target_pose"]
    assert target["xyz"] == [0.18, 0.12, 0.14]
    assert target["rotation_matrix"] == rotation
    assert target["waypoint_role"] == "grasp_path_sample"
    assert target["path_fraction"] == 0.8
    assert target["segment_start_role"] == "grasp_clearance"
    assert target["segment_end_role"] == "grasp_contact"
    assert dispatched[0]["check_endpoint_collision"] is True
    assert "target_pose" not in plan.request.parameters


def test_compiled_grasp_position_only_ik_preserves_policy_without_rotation() -> None:
    memory = AgentMemory()
    memory.start_session(task="inspect the cube near the grasp anchor")
    pose = {
        "frame": "world",
        "xyz": [0.1, 0.2, 0.3],
        "rotation_matrix": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "compiled_grasp_id": "compiled-position-only-ref",
        "waypoint_role": "grasp_clearance",
    }
    memory.save_artifact(
        "compiled-position-only-ref",
        {
            "type": "compiled_grasp",
            "compiled_grasp_id": "compiled-position-only-ref",
            "scene_epoch": 0,
            "hover_pose": dict(pose),
        },
        source="compile_grasp_seed",
    )
    tools = build_default_tool_registry()
    dispatched: list[dict] = []

    def record_ik(context: ToolExecutionContext) -> ToolResult:
        dispatched.append(dict(context.parameters))
        return ToolResult(True)

    tools.bind_handler("ik_preview_check", record_ik)

    plan = ActionPipeline().compile(
        PlannerDecision(
            action_type="tool_call",
            action="ik_preview_check",
            parameters={
                "compiled_grasp_id": "compiled-position-only-ref",
                "waypoint_role": "grasp_clearance",
                "preserve_current_orientation": True,
                "check_endpoint_collision": True,
            },
        ),
        observation=_observation(),
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )

    assert plan.status is PipelineStatus.EXECUTED
    resolved = dispatched[0]
    assert resolved["preserve_current_orientation"] is True
    assert resolved["check_endpoint_collision"] is True
    assert resolved["target_pose"]["xyz"] == pose["xyz"]
    assert resolved["target_pose"]["compiled_grasp_id"] == (
        "compiled-position-only-ref"
    )
    assert "rotation_matrix" not in resolved["target_pose"]
    assert "target_pose" not in plan.request.parameters


def test_move_to_resolves_exact_pose_from_ik_receipt_id() -> None:
    memory = AgentMemory()
    memory.start_session(task="move safely")
    pose = {
        "frame": "world",
        "xyz": [0.21, -0.04, 0.31],
        "rotation_matrix": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    }
    _record_ik_receipt(memory, receipt_id="ik-exact-ref", target_pose=pose)
    tools = build_default_tool_registry()
    dispatched: list[dict] = []

    def record_move(context: ToolExecutionContext) -> ToolResult:
        dispatched.append(dict(context.parameters))
        return ToolResult(True)

    tools.bind_handler("move_to", record_move)

    plan = ActionPipeline().compile(
        PlannerDecision(
            action_type="tool_call",
            action="move_to",
            parameters={
                "ik_receipt_id": "ik-exact-ref",
                "enable_collision_check": True,
            },
        ),
        observation=_observation(),
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )

    assert plan.status is PipelineStatus.EXECUTED
    assert plan.request.parameters == {
        "ik_receipt_id": "ik-exact-ref",
        "enable_collision_check": True,
    }
    resolved = dispatched[0]
    assert resolved["target_pose"] == pose
    assert resolved["ik_receipt_id"] == "ik-exact-ref"
    assert resolved["enable_collision_check"] is True


def test_move_to_rejects_copied_pose_and_unknown_receipt_with_actionable_ids() -> None:
    memory = AgentMemory()
    memory.start_session(task="move safely")
    pose = {"frame": "world", "xyz": [0.1, 0.2, 0.3]}
    _record_ik_receipt(memory, receipt_id="ik-known-ref", target_pose=pose)
    tools = build_default_tool_registry()
    tools.bind_handler("move_to", lambda _context: ToolResult(True))

    copied = ActionPipeline().compile(
        PlannerDecision(
            action_type="tool_call",
            action="move_to",
            parameters={"target_pose": pose},
        ),
        observation=_observation(),
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )
    unknown = ActionPipeline().compile(
        PlannerDecision(
            action_type="tool_call",
            action="move_to",
            parameters={"ik_receipt_id": "ik-typo-ref"},
        ),
        observation=_observation(),
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )

    assert copied.status is PipelineStatus.BLOCKED
    assert copied.metadata["repair_bundle"]["code"] == "invalid_ik_receipt_reference"
    assert "no longer accepts model-copied target_pose" in copied.tool_calls[0].reason
    assert unknown.status is PipelineStatus.BLOCKED
    assert "ik-known-ref" in unknown.tool_calls[0].reason


def test_typed_move_capture_uses_trusted_host_execution_receipt() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    target_pose = {
        "frame": "world",
        "xyz": [0.10, 0.20, 0.30],
        "compiled_grasp_id": "compiled-typed",
        "source_grasp_id": "candidate-typed",
        "waypoint_role": "grasp_contact",
    }
    memory.save_fact(
        GRASP_PROVENANCE_KEY,
        {
            "schema_version": "openeta.grasp_provenance.v1",
            "compiled_grasp_id": "compiled-typed",
            "candidate_id": "candidate-typed",
            "object_scene_epoch": 0,
            "robot_motion_epoch": 0,
        },
        source="unit_test",
    )
    memory.save_artifact(
        "compiled-typed",
        {
            "type": "compiled_grasp",
            "compiled_grasp_id": "compiled-typed",
            "candidate_id": "candidate-typed",
            "scene_epoch": 0,
            "contact_pose": dict(target_pose),
        },
        source="unit_test",
    )
    _record_ik_receipt(
        memory,
        receipt_id="ik-typed-contact",
        target_pose=target_pose,
    )

    class MotionTransport:
        def call_tool(self, _name, _arguments, *, timeout_s=None):
            del timeout_s
            return {
                "success": True,
                "reached_target": True,
                "steps_executed": 4,
                "stop_reason": "target_reached",
                "start": {"xyz": [0.10, 0.20, 0.34]},
                "end": {"xyz": [0.10, 0.20, 0.30]},
                "target": {"xyz": [0.10, 0.20, 0.30]},
                "position_error_m": 0.0,
            }

    tools = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(),
        transport=MotionTransport(),
        config=SimulatorMcpToolProxyConfig(
            session_id="sim-typed",
            handle="env-typed",
        ),
        tool_names=("move_to",),
    )
    pipeline = ActionPipeline()
    plan = pipeline.compile(
        PlannerDecision(
            action_type="tool_call",
            action="move_to",
            parameters={
                "ik_receipt_id": "ik-typed-contact",
                "enable_collision_check": True,
            },
        ),
        observation=_observation(),
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )

    assert plan.status is PipelineStatus.EXECUTED
    assert plan.tool_calls[0].parameters == {
        "ik_receipt_id": "ik-typed-contact",
        "enable_collision_check": True,
    }
    details = plan.tool_calls[0].result["details"]
    assert details["parameters"] == plan.tool_calls[0].parameters
    assert details["host_execution_receipt"]["parameters"]["target_pose"] == (
        target_pose
    )
    memory.add_action(plan.to_env_action())

    receipt = memory.latest_compiled_contact_execution()
    assert receipt is not None
    assert receipt["compiled_grasp_id"] == "compiled-typed"
    assert receipt["requested_xyz"] == [0.10, 0.20, 0.30]
    assert memory.transition_ledger()[-1]["candidate_id"] == "candidate-typed"

    tools.bind_handler("gripper_control", lambda _context: ToolResult(True))
    close = pipeline.compile(
        PlannerDecision(
            action_type="tool_call",
            action="gripper_control",
            parameters={"position": 0},
        ),
        observation=_observation(),
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )
    assert close.status is not PipelineStatus.BLOCKED


def test_typed_move_transport_unknown_reconciles_host_resolved_target() -> None:
    memory = AgentMemory()
    memory.start_session(task="move safely")
    target_pose = {"frame": "world", "xyz": [0.10, 0.20, 0.30]}
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "move_to",
                    "parameters": {"ik_receipt_id": "ik-transport-unknown"},
                },
                "status": "executed",
                "tool_calls": [
                    {
                        "name": "move_to",
                        "status": "failed",
                        "parameters": {
                            "target_pose": target_pose,
                            "ik_receipt_id": "ik-transport-unknown",
                        },
                        "result": {
                            "success": False,
                            "details": {
                                "outputs": {
                                    "motion_outcome": "unknown",
                                    "reconciliation_required": True,
                                }
                            },
                        },
                    }
                ],
            },
        )
    )

    reconciliation = memory.motion_reconciliation()
    assert reconciliation is not None
    assert reconciliation["intended_parameters"]["target_pose"] == target_pose
    assert reconciliation["intended_parameters"]["ik_receipt_id"] == (
        "ik-transport-unknown"
    )


def test_follow_trajectory_resolves_ordered_ik_receipts_without_model_copied_poses() -> None:
    memory = AgentMemory()
    memory.start_session(task="move through a checked path")
    poses = [
        {"frame": "world", "xyz": [0.10, 0.20, 0.30]},
        {"frame": "world", "xyz": [0.11, 0.20, 0.31]},
    ]
    for index, pose in enumerate(poses):
        _record_ik_receipt(
            memory,
            receipt_id=f"ik-path-{index}",
            target_pose=pose,
        )
    tools = build_default_tool_registry()
    dispatched: list[dict] = []

    def record_trajectory(context: ToolExecutionContext) -> ToolResult:
        dispatched.append(dict(context.parameters))
        return ToolResult(True)

    tools.bind_handler("follow_eef_trajectory", record_trajectory)
    plan = ActionPipeline().compile(
        PlannerDecision(
            action_type="tool_call",
            action="follow_eef_trajectory",
            parameters={
                "ik_receipt_ids": ["ik-path-0", "ik-path-1"],
                "enable_collision_check": True,
            },
        ),
        observation=_observation(),
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )

    assert plan.status is PipelineStatus.EXECUTED
    assert plan.request.parameters == {
        "ik_receipt_ids": ["ik-path-0", "ik-path-1"],
        "enable_collision_check": True,
    }
    assert dispatched == [
        {
            "trajectory": poses,
            "ik_receipt_ids": ["ik-path-0", "ik-path-1"],
            "enable_collision_check": True,
        }
    ]


def test_host_builds_condition_c_route_bundle_from_current_ik_receipts() -> None:
    memory = AgentMemory()
    memory.start_session(task="move through a checked path")
    poses = [
        {"frame": "world", "xyz": [0.10, 0.20, 0.30]},
        {"frame": "world", "xyz": [0.11, 0.20, 0.31]},
    ]
    captured = [0.0, 0.0, 0.70710678, 0.70710678]
    joints = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7]
    for index, pose in enumerate(poses):
        _record_ik_receipt(
            memory,
            receipt_id=f"ik-route-{index}",
            target_pose=pose,
            captured_quat_xyzw=captured,
            joint_positions=joints,
        )

    bundle = memory.resolve_ik_trajectory_execution_bundle(
        {"ik_receipt_ids": ["ik-route-0", "ik-route-1"]}
    )

    assert bundle is not None
    assert bundle["condition"] == "C"
    assert bundle["authority"] == "host_memory_exact_receipt_resolution"
    assert bundle["sequential_preview_policy"] == (
        "just_in_time_from_actual_segment_end"
    )
    assert [entry["source_ik_receipt_id"] for entry in bundle["entries"]] == [
        "ik-route-0",
        "ik-route-1",
    ]
    assert bundle["entries"][0]["target_pose"]["quat_xyzw"] == captured
    assert bundle["entries"][0]["source_captured_quat_xyzw"] == captured
    assert bundle["entries"][0]["source_seed_hint"]["joint_positions"] == joints


def test_condition_c_route_bundle_rejects_stale_receipt() -> None:
    memory = AgentMemory()
    memory.start_session(task="move through a checked path")
    _record_ik_receipt(
        memory,
        receipt_id="ik-route-stale",
        target_pose={"frame": "world", "xyz": [0.10, 0.20, 0.30]},
    )
    memory._advance_runtime_epochs(
        tool="move_to",
        object_scene_changed=False,
        source="unit_test_motion",
    )

    with pytest.raises(ValueError, match="stale"):
        memory.resolve_ik_trajectory_execution_bundle(
            {"ik_receipt_ids": ["ik-route-stale"]}
        )


def test_follow_trajectory_rejects_copied_path_and_infeasible_waypoint() -> None:
    memory = AgentMemory()
    memory.start_session(task="move through a checked path")
    pose = {"frame": "world", "xyz": [0.10, 0.20, 0.30]}
    _record_ik_receipt(
        memory,
        receipt_id="ik-hard-infeasible",
        target_pose=pose,
        classification="hard_infeasible",
    )
    tools = build_default_tool_registry()
    tools.bind_handler("follow_eef_trajectory", lambda _context: ToolResult(True))

    copied = ActionPipeline().compile(
        PlannerDecision(
            action_type="tool_call",
            action="follow_eef_trajectory",
            parameters={"trajectory": [pose]},
        ),
        observation=_observation(),
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )
    infeasible = ActionPipeline().compile(
        PlannerDecision(
            action_type="tool_call",
            action="follow_eef_trajectory",
            parameters={"ik_receipt_ids": ["ik-hard-infeasible"]},
        ),
        observation=_observation(),
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )

    assert copied.status is PipelineStatus.BLOCKED
    assert copied.metadata["repair_bundle"]["code"] == (
        "invalid_ik_trajectory_reference"
    )
    assert "no longer accepts a model-copied trajectory" in copied.tool_calls[0].reason
    assert infeasible.status is PipelineStatus.BLOCKED
    assert "trajectory_waypoint_0" in infeasible.tool_calls[0].reason
    assert "hard_infeasible" in infeasible.tool_calls[0].reason


def test_typed_trajectory_completes_matching_frozen_attachment_probe() -> None:
    memory = AgentMemory()
    memory.start_session(task="open the drawer")
    marker = "frozen-probe-path"
    poses = [
        {
            "frame": "world",
            "xyz": [0.10, 0.20, 0.30],
            "probe_path_sha256": marker,
        },
        {
            "frame": "world",
            "xyz": [0.11, 0.20, 0.31],
            "probe_path_sha256": marker,
        },
    ]
    memory.save_fact(
        ARTICULATED_ATTACHMENT_PROBE_KEY,
        {
            "schema_version": "openeta.articulated_attachment_probe.v1",
            "status": "prepared",
            "probe_id": f"probe:{marker}",
            "path_sha256": marker,
            "frozen_motion": {
                "name": "follow_eef_trajectory",
                "parameters": {
                    "trajectory": poses,
                    "enable_collision_check": True,
                },
            },
        },
        source="unit_test",
    )
    for index, pose in enumerate(poses):
        _record_ik_receipt(
            memory,
            receipt_id=f"ik-probe-{index}",
            target_pose=pose,
        )
    tools = build_default_tool_registry()
    tools.bind_handler("follow_eef_trajectory", lambda _context: ToolResult(True))

    plan = ActionPipeline().compile(
        PlannerDecision(
            action_type="tool_call",
            action="follow_eef_trajectory",
            parameters={
                "ik_receipt_ids": ["ik-probe-0", "ik-probe-1"],
                "enable_collision_check": True,
            },
        ),
        observation=_observation(),
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )
    assert plan.status is PipelineStatus.EXECUTED

    memory.add_action(plan.to_env_action())
    probe = memory.articulated_attachment_probe()
    assert probe is not None
    assert probe["status"] == "completed"
    assert probe["last_attempt_status"] == "executed"


def test_ik_resolves_exact_attachment_probe_waypoint_from_short_id() -> None:
    memory = AgentMemory()
    memory.start_session(task="lift the held object")
    marker = "probe-waypoint-short-id"
    target_pose = {
        "frame": "world",
        "xyz": [0.10, 0.20, 0.35],
        "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
        "probe_path_sha256": marker,
    }
    memory.save_fact(
        ARTICULATED_ATTACHMENT_PROBE_KEY,
        {
            "schema_version": "openeta.articulated_attachment_probe.v1",
            "status": "prepared",
            "probe_id": f"probe:{marker}",
            "path_sha256": marker,
            "scene_epoch": 0,
            "robot_motion_epoch": 0,
            "frozen_path": [target_pose],
            "frozen_motion": {
                "name": "move_to",
                "parameters": {
                    "target_pose": target_pose,
                    "enable_collision_check": True,
                },
            },
        },
        source="unit_test",
    )
    captured: list[dict] = []
    tools = build_default_tool_registry()
    tools.bind_handler(
        "ik_preview_check",
        lambda context: captured.append(dict(context.parameters)) or ToolResult(True),
    )

    request_parameters = {
        "probe_id": f"probe:{marker}",
        "waypoint_index": 0,
        "position_tolerance_m": 0.01,
        "orientation_tolerance_rad": 0.10,
        "check_endpoint_collision": True,
    }
    plan = ActionPipeline().compile(
        PlannerDecision(
            action_type="tool_call",
            action="ik_preview_check",
            parameters=request_parameters,
        ),
        observation=_observation(),
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )

    assert plan.status is PipelineStatus.EXECUTED
    assert captured == [
        {
            "target_pose": target_pose,
            "position_tolerance_m": 0.01,
            "orientation_tolerance_rad": 0.10,
            "check_endpoint_collision": True,
        }
    ]
    assert plan.tool_calls[0].parameters == request_parameters


@pytest.mark.parametrize(
    ("parameters", "message"),
    [
        ({"probe_id": "probe:wrong", "waypoint_index": 0}, "active probe_id"),
        ({"probe_id": "probe:current", "waypoint_index": 2}, "valid indices=[0]"),
    ],
)
def test_ik_probe_reference_rejection_returns_actionable_feedback(
    parameters: dict,
    message: str,
) -> None:
    memory = AgentMemory()
    memory.start_session(task="lift the held object")
    marker = "current"
    pose = {
        "frame": "world",
        "xyz": [0.1, 0.2, 0.35],
        "probe_path_sha256": marker,
    }
    memory.save_fact(
        ARTICULATED_ATTACHMENT_PROBE_KEY,
        {
            "status": "prepared",
            "probe_id": "probe:current",
            "path_sha256": marker,
            "scene_epoch": 0,
            "robot_motion_epoch": 0,
            "frozen_path": [pose],
            "frozen_motion": {
                "name": "move_to",
                "parameters": {"target_pose": pose, "enable_collision_check": True},
            },
        },
        source="unit_test",
    )

    plan = ActionPipeline().compile(
        PlannerDecision(
            action_type="tool_call",
            action="ik_preview_check",
            parameters=parameters,
        ),
        observation=_observation(),
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
        memory=memory,
    )

    assert plan.status is PipelineStatus.BLOCKED
    assert plan.metadata["repair_bundle"]["code"] == (
        "invalid_attachment_probe_reference"
    )
    assert message in plan.tool_calls[0].reason


def test_attachment_probe_waypoint_reference_expires_after_robot_motion() -> None:
    memory = AgentMemory()
    memory.start_session(task="lift the held object")
    marker = "stale-after-motion"
    pose = {
        "frame": "world",
        "xyz": [0.1, 0.2, 0.35],
        "probe_path_sha256": marker,
    }
    memory.save_fact(
        ARTICULATED_ATTACHMENT_PROBE_KEY,
        {
            "status": "prepared",
            "probe_id": f"probe:{marker}",
            "path_sha256": marker,
            "scene_epoch": 0,
            "robot_motion_epoch": 0,
            "frozen_path": [pose],
        },
        source="unit_test",
    )
    memory.save_fact(ROBOT_MOTION_EPOCH_KEY, {"epoch": 1}, source="unit_test")

    with pytest.raises(ValueError, match="robot_motion_epoch=0, current=1"):
        memory.resolve_attachment_probe_waypoint(
            probe_id=f"probe:{marker}",
            waypoint_index=0,
        )


def _record_target_selection(
    memory: AgentMemory,
    *,
    artifact_root: Path | None = None,
    rgb: str = "/session/rgb.png",
    depth: str = "/session/depth.png",
    mask: str = "/session/mask.png",
    extrinsics: dict | None = None,
) -> None:
    if artifact_root is not None:
        rgb_path = artifact_root / "rgb.png"
        depth_path = artifact_root / "depth.png"
        mask_path = artifact_root / "mask.png"
        Image.new("RGB", (8, 8), (0, 0, 0)).save(rgb_path)
        Image.new("I;16", (8, 8), 1200).save(depth_path)
        target_mask = Image.new("L", (8, 8), 0)
        for y in range(2, 6):
            for x in range(2, 6):
                target_mask.putpixel((x, y), 255)
        target_mask.save(mask_path)
        rgb, depth, mask = str(rgb_path), str(depth_path), str(mask_path)
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "tool_calls": [
                    {
                        "name": "sam3",
                        "status": "executed",
                        "parameters": {
                            "source_packet_id": "packet-4",
                            "camera_frame_id": "agentview",
                            "prompt": "cube",
                        },
                        "result": {
                            "success": True,
                            "details": {
                                "outputs": {
                                    "result_id": "sam3-target-1",
                                    "prompt": "cube",
                                    "source_image": rgb,
                                    "source_packet_id": "packet-4",
                                    "source_observation": {
                                        "packet_id": "packet-4",
                                        "frame_id": "agentview",
                                        "rgb": rgb,
                                        "depth": depth,
                                        "intrinsics": {
                                            "fx": 100.0,
                                            "fy": 100.0,
                                            "cx": 0.5,
                                            "cy": 0.5,
                                            "scale": 1000,
                                        },
                                        **(
                                            {"extrinsics": extrinsics}
                                            if extrinsics is not None
                                            else {}
                                        ),
                                    },
                                    "detection_count": 1,
                                    "detections": [
                                        {
                                            "id": "detection_001",
                                            "score": 0.9,
                                            "mask_ref": mask,
                                        }
                                    ],
                                }
                            },
                        },
                    }
                ]
            },
        )
    )
    memory.resolve_sam3_selection(
        result_id="sam3-target-1",
        detection_id="detection_001",
        selection_source="main_agent_vlm",
        reason="cube identity verified",
    )
    # Selection is performed by a bookkeeping tool before the enclosing action is
    # appended; the next action boundary refreshes host-resolved evidence bundles.
    memory.add_action(EnvAction(action_type="tool_call", command={"tool_calls": []}))


def test_agent_owned_pipeline_does_not_turn_reference_work_into_a_task_gate(
    tmp_path: Path,
) -> None:
    memory = AgentMemory()
    memory.start_session(task="pick alphabet soup")
    rgb = tmp_path / "rgb.png"
    depth = tmp_path / "depth.png"
    Image.new("RGB", (8, 8), (0, 0, 0)).save(rgb)
    Image.new("I;16", (8, 8), 1000).save(depth)
    observation = _observation()
    observation.metadata["image_artifacts"][0]["path"] = str(rgb)
    observation.metadata["image_artifacts"][1]["path"] = str(depth)
    memory.add_observation(observation)
    memory.save_fact(
        "pending_reference_localization",
        {
            "scene_image": "/session/rgb.png",
            "source_packet_id": "packet-4",
            "camera_frame_id": "agentview",
            "target_object": "alphabet soup",
            "required_parameter": "positive_points",
            "positive_points": [{"x": 20, "y": 30, "label": 1}],
        },
        source="test",
    )
    tools = build_default_tool_registry()
    tools.bind_handler("retrieve_asset_reference", lambda _context: ToolResult(True))
    decision = PlannerDecision(
        action_type="tool_call",
        action="retrieve_asset_reference",
        parameters={
            "environment": "libero",
            "target_object": "alphabet soup",
            "source_packet_id": "packet-4",
            "camera_frame_id": "agentview",
        },
    )

    plan = ActionPipeline().compile(
        decision,
        observation=observation,
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )
    assert plan.status != PipelineStatus.BLOCKED


def test_grasp_bundle_resolves_aligned_inputs_without_model_copying(
    tmp_path: Path,
) -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    _record_target_selection(memory, artifact_root=tmp_path)
    public = memory.grasp_input_bundle()
    assert public is not None and public["status"] == "ready"
    bundle_id = public["bundle_id"]
    captured: list[dict] = []
    tools = build_default_tool_registry()

    def handler(context: ToolExecutionContext) -> ToolResult:
        captured.append(dict(context.parameters))
        return ToolResult(True, content="resolved")

    tools.bind_handler("grasp_pose_estimate", handler)
    plan = ActionPipeline().compile(
        PlannerDecision(
            action_type="tool_call",
            action="grasp_pose_estimate",
            parameters={"bundle_id": bundle_id},
        ),
        observation=_observation(),
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )

    assert plan.status == PipelineStatus.EXECUTED
    assert captured[0]["rgb"] == str(tmp_path / "rgb.png")
    assert captured[0]["depth"] == str(tmp_path / "depth.png")
    assert captured[0]["object_mask"]["mask_ref"] == str(tmp_path / "mask.png")
    assert captured[0]["camera_frame_id"] == "agentview"
    assert plan.metadata["provenance_bundle_resolution"]["bundle_id"] == bundle_id
    receipt = plan.metadata["host_resolution_receipt"]
    assert receipt["status"] == "resolved"
    assert receipt["resolver_id"] == (
        "openeta.host_resolver.grasp_pose_estimate.v1"
    )
    assert receipt["dispatch_authority"] == "tool_contract"
    assert receipt["public_parameter_keys"] == ["bundle_id"]
    assert "rgb" in receipt["resolved_parameter_keys"]


def test_grasp_bundle_preserves_only_agent_backend_order_override(
    tmp_path: Path,
) -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    _record_target_selection(memory, artifact_root=tmp_path)
    public = memory.grasp_input_bundle()
    assert public is not None and public["status"] == "ready"
    captured: list[dict] = []
    tools = build_default_tool_registry()
    tools.bind_handler(
        "grasp_pose_estimate",
        lambda context: captured.append(dict(context.parameters)) or ToolResult(True),
    )

    plan = ActionPipeline().compile(
        PlannerDecision(
            action_type="tool_call",
            action="grasp_pose_estimate",
            parameters={
                "bundle_id": public["bundle_id"],
                "backend_preference": ["graspgenx", "anygrasp"],
            },
        ),
        observation=_observation(),
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )

    assert plan.status == PipelineStatus.EXECUTED
    assert captured[0]["backend_preference"] == ["graspgenx", "anygrasp"]
    assert captured[0]["rgb"] == str(tmp_path / "rgb.png")
    assert captured[0]["object_mask"]["mask_ref"] == str(tmp_path / "mask.png")
    assert plan.tool_calls[0].parameters == {
        "bundle_id": public["bundle_id"],
        "backend_preference": ["graspgenx", "anygrasp"],
    }


def test_grasp_bundle_preserves_host_depth_compatibility_hint(tmp_path: Path) -> None:
    rgb = tmp_path / "rgb.png"
    depth = tmp_path / "depth.png"
    mask = tmp_path / "mask.png"
    Image.new("RGB", (6, 6), (0, 0, 0)).save(rgb)
    Image.new("I;16", (6, 6), 1200).save(depth)
    target_mask = Image.new("L", (6, 6), 0)
    for y in range(2, 5):
        for x in range(2, 5):
            target_mask.putpixel((x, y), 255)
    target_mask.save(mask)
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    _record_target_selection(
        memory,
        rgb=str(rgb),
        depth=str(depth),
        mask=str(mask),
    )

    public = memory.grasp_input_bundle()
    assert public is not None and public["status"] == "ready"
    resolved = memory.resolve_grasp_input_bundle(public["bundle_id"])

    assert resolved["parameters"]["hints"]["depth_cutoff_factor"] == 1.333333
    assert resolved["parameters"]["hints"]["max_gripper_width_m"] == 0.08
    assert resolved["parameters"]["hints"]["execution_reference_point"] == (
        "translation_xyz"
    )


def test_grasp_bundle_derives_packet_owned_camera_up_direction(tmp_path: Path) -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    _record_target_selection(
        memory,
        artifact_root=tmp_path,
        extrinsics={
            "camera_frame": "opengl",
            "frame_transform": "camera_to_world",
            "matrix_layout": "row_major",
            "pos": [0.0, 0.0, 0.0],
            "mat": [1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0],
        },
    )

    public = memory.grasp_input_bundle()
    resolved = memory.resolve_grasp_input_bundle(public["bundle_id"])

    assert resolved["parameters"]["hints"]["up_direction_camera"] == pytest.approx(
        [0.0, -1.0, 0.0]
    )


def test_grasp_bundle_rejects_clipped_target_mask_with_view_recovery(
    tmp_path: Path,
) -> None:
    rgb = tmp_path / "rgb.png"
    depth = tmp_path / "depth.png"
    mask = tmp_path / "mask.png"
    Image.new("RGB", (8, 8), (0, 0, 0)).save(rgb)
    Image.new("I;16", (8, 8), 1200).save(depth)
    clipped = Image.new("L", (8, 8), 0)
    for y in range(2, 6):
        for x in range(0, 3):
            clipped.putpixel((x, y), 255)
    clipped.save(mask)
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    _record_target_selection(
        memory,
        rgb=str(rgb),
        depth=str(depth),
        mask=str(mask),
    )

    public = memory.grasp_input_bundle()

    assert public is not None
    assert public["status"] == "requires_better_view"
    assert public["semantic_outcome"] == "requires_better_view"
    assert public["target_mask_quality"]["status"] == "clipped_mask"
    assert "bundle_id" not in public
    assert {item["action"] for item in public["recovery_options"]} == {
        "propose_wrist_viewpoints",
        "segment_from_scene_primary",
    }


def test_pipeline_resolves_molmopoint_packet_sources_without_model_paths(
    tmp_path: Path,
) -> None:
    rgb = tmp_path / "rgb.png"
    depth = tmp_path / "depth.png"
    Image.new("RGB", (8, 8), (0, 0, 0)).save(rgb)
    Image.new("I;16", (8, 8), 1000).save(depth)
    observation = _observation()
    observation.metadata["image_artifacts"][0]["path"] = str(rgb)
    observation.metadata["image_artifacts"][1]["path"] = str(depth)
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    memory.add_observation(observation)
    captured: list[dict] = []
    tools = build_default_tool_registry()
    tools.bind_handler(
        "molmopoint",
        lambda context: captured.append(dict(context.parameters)) or ToolResult(True),
    )

    plan = ActionPipeline().compile(
        PlannerDecision(
            action_type="tool_call",
            action="molmopoint",
            parameters={
                "sources": [
                    {
                        "source_packet_id": "packet-4",
                        "camera_frame_id": "agentview",
                    }
                ],
                "prompt": "point to the cube",
            },
        ),
        observation=observation,
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )

    assert plan.status == PipelineStatus.EXECUTED
    assert plan.tool_calls[0].parameters == {
        "sources": [
            {"source_packet_id": "packet-4", "camera_frame_id": "agentview"}
        ],
        "prompt": "point to the cube",
    }
    assert captured[0]["images"] == [str(rgb)]
    assert captured[0]["_source_observations"][0]["packet_id"] == "packet-4"


def test_same_view_recovery_allows_identical_refresh_and_rejects_changed_pixels(
    tmp_path: Path,
) -> None:
    rgb = tmp_path / "rgb.png"
    depth = tmp_path / "depth.png"
    Image.new("RGB", (8, 8), (0, 0, 0)).save(rgb)
    Image.new("I;16", (8, 8), 1000).save(depth)
    observation = _observation()
    observation.metadata["image_artifacts"][0]["path"] = str(rgb)
    observation.metadata["image_artifacts"][1]["path"] = str(depth)
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    memory.add_observation(observation)
    source = memory.resolve_observation_packet("packet-4", "agentview")
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "sam3",
                    "parameters": {
                        "source_packet_id": "packet-4",
                        "camera_frame_id": "agentview",
                        "prompt": "cube",
                    },
                },
                "status": "executed",
                "tool_calls": [
                    {
                        "name": "sam3",
                        "status": "executed",
                        "parameters": {
                            "source_packet_id": "packet-4",
                            "camera_frame_id": "agentview",
                            "prompt": "cube",
                        },
                        "result": {
                            "success": True,
                            "details": {
                                "parameters": {
                                    "source_packet_id": "packet-4",
                                    "camera_frame_id": "agentview",
                                    "prompt": "cube",
                                },
                                "outputs": {
                                    "result_id": "sam-no-detection",
                                    "detections": [],
                                    "prompt": "cube",
                                    "source_packet_id": "packet-4",
                                    "source_frame_id": "agentview",
                                    "source_observation": source,
                                },
                            },
                        },
                    }
                ],
            },
        )
    )
    refreshed = _observation()
    for artifact in refreshed.metadata["image_artifacts"]:
        artifact["packet_id"] = "packet-5"
        artifact["path"] = str(rgb if artifact["kind"] == "rgb" else depth)
    memory.add_observation(refreshed)
    tools = build_default_tool_registry()
    tools.bind_handler("molmopoint", lambda _context: ToolResult(True))
    tools.bind_handler("sam3", lambda _context: ToolResult(True))

    identical = ActionPipeline().compile(
        PlannerDecision(
            action_type="tool_call",
            action="molmopoint",
            parameters={
                "sources": [
                    {
                        "source_packet_id": "packet-5",
                        "camera_frame_id": "agentview",
                    }
                ],
                "prompt": "Point to the cube in Image 1.",
            },
        ),
        observation=refreshed,
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )

    assert identical.status is PipelineStatus.EXECUTED

    changed_rgb = tmp_path / "changed-rgb.png"
    Image.new("RGB", (8, 8), (255, 0, 0)).save(changed_rgb)
    changed = _observation()
    for artifact in changed.metadata["image_artifacts"]:
        artifact["packet_id"] = "packet-6"
        artifact["path"] = str(changed_rgb if artifact["kind"] == "rgb" else depth)
    memory.add_observation(changed)

    blocked = ActionPipeline().compile(
        PlannerDecision(
            action_type="tool_call",
            action="molmopoint",
            parameters={
                "sources": [
                    {
                        "source_packet_id": "packet-6",
                        "camera_frame_id": "agentview",
                    }
                ],
                "prompt": "Point to the cube in Image 1.",
            },
        ),
        observation=changed,
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )

    assert blocked.status is PipelineStatus.BLOCKED
    assert blocked.metadata["repair_bundle"]["code"] == "same_view_packet_mismatch"
    assert "exact SAM3 no-detection source packet 'packet-4'" in (
        blocked.tool_calls[0].reason
    )
    repair_calls = blocked.metadata["repair_bundle"]["allowed_next_calls"]
    exact = next(call for call in repair_calls if call["tool"] == "molmopoint")
    assert exact["parameters"]["sources"] == [
        {"source_packet_id": "packet-4", "camera_frame_id": "agentview"}
    ]

    exact_plan = ActionPipeline().compile(
        PlannerDecision(
            action_type="tool_call",
            action="molmopoint",
            parameters=exact["parameters"],
        ),
        observation=changed,
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )

    assert exact_plan.status is PipelineStatus.EXECUTED
    assert memory.same_view_point_grounding_source_error(
        [{"packet_id": "wrist-packet-1", "frame_id": "wrist"}]
    ) is None

    blocked_sam3 = ActionPipeline().compile(
        PlannerDecision(
            action_type="tool_call",
            action="sam3",
            parameters={
                "source_packet_id": "packet-6",
                "camera_frame_id": "agentview",
                "mode": "points",
                "points": [{"x": 4, "y": 4, "label": 1}],
                "evidence_role": "target_object",
            },
        ),
        observation=changed,
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )

    assert blocked_sam3.status is PipelineStatus.BLOCKED
    assert blocked_sam3.metadata["repair_bundle"]["code"] == (
        "same_view_packet_mismatch"
    )
    exact_sam3 = next(
        call
        for call in blocked_sam3.metadata["repair_bundle"]["allowed_next_calls"]
        if call["tool"] == "sam3" and call["parameters"].get("mode") == "points"
    )
    assert exact_sam3["parameters"]["source_packet_id"] == "packet-4"
    assert exact_sam3["parameters"]["points"] == [{"x": 4, "y": 4, "label": 1}]


def test_pipeline_resolves_depth_prior_packet_without_model_paths(tmp_path: Path) -> None:
    rgb = tmp_path / "rgb.png"
    depth = tmp_path / "depth.png"
    Image.new("RGB", (8, 8), (0, 0, 0)).save(rgb)
    Image.new("I;16", (8, 8), 1000).save(depth)
    observation = _observation()
    observation.metadata["image_artifacts"][0]["path"] = str(rgb)
    observation.metadata["image_artifacts"][1]["path"] = str(depth)
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    memory.add_observation(observation)
    captured: list[dict] = []
    tools = build_default_tool_registry()
    tools.bind_handler(
        "estimate_depth_prior",
        lambda context: captured.append(dict(context.parameters)) or ToolResult(True),
    )

    plan = ActionPipeline().compile(
        PlannerDecision(
            action_type="tool_call",
            action="estimate_depth_prior",
            parameters={
                "source_packet_id": "packet-4",
                "camera_frame_id": "agentview",
                "resolution_level": 4,
            },
        ),
        observation=observation,
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )

    assert plan.status == PipelineStatus.EXECUTED
    assert plan.tool_calls[0].parameters == {
        "source_packet_id": "packet-4",
        "camera_frame_id": "agentview",
        "resolution_level": 4,
    }
    assert captured[0]["rgb"] == str(rgb)
    assert captured[0]["intrinsics"]["fx"] == 100.0
    assert captured[0]["source_packet_id"] == "packet-4"
    assert captured[0]["resolution_level"] == 4
    assert plan.metadata["host_resolution_receipt"] == {
        "schema_version": "openeta.host_resolution_receipt.v1",
        "status": "resolved",
        "tool": "estimate_depth_prior",
        "resolver_id": "openeta.host_resolver.estimate_depth_prior.v1",
        "implementation": (
            "agent.tools.runtime_contract_bindings."
            "_resolve_estimate_depth_prior_input"
        ),
        "dispatch_authority": "tool_contract",
        "public_parameter_keys": [
            "camera_frame_id",
            "resolution_level",
            "source_packet_id",
        ],
        "resolved_parameter_keys": [
            "_source_observation",
            "bundle_id",
            "camera_frame_id",
            "camera_id",
            "intrinsics",
            "resolution_level",
            "rgb",
            "source_packet_id",
        ],
    }


def test_pipeline_resolves_depth_enhancement_and_matching_prior(tmp_path: Path) -> None:
    rgb = tmp_path / "rgb.png"
    depth = tmp_path / "depth.png"
    prior = tmp_path / "prior.npy"
    confidence = tmp_path / "confidence.npy"
    Image.new("RGB", (8, 8), (0, 0, 0)).save(rgb)
    Image.new("I;16", (8, 8), 1000).save(depth)
    prior.write_bytes(b"prior")
    confidence.write_bytes(b"confidence")
    observation = _observation()
    observation.metadata["image_artifacts"][0]["path"] = str(rgb)
    observation.metadata["image_artifacts"][1]["path"] = str(depth)
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    memory.add_observation(observation)
    memory.save_artifact(
        "depth-prior-agentview",
        {
            "type": "depth_prior",
            "source_rgb": str(rgb),
            "prior_depth": str(prior),
            "prior_confidence": str(confidence),
            "prior_confidence_semantics": "higher_is_better",
        },
        source="test",
    )
    captured: list[dict] = []
    tools = build_default_tool_registry()
    tools.bind_handler(
        "enhance_depth",
        lambda context: captured.append(dict(context.parameters)) or ToolResult(True),
    )

    plan = ActionPipeline().compile(
        PlannerDecision(
            action_type="tool_call",
            action="enhance_depth",
            parameters={
                "source_packet_id": "packet-4",
                "camera_frame_id": "agentview",
            },
        ),
        observation=observation,
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )

    assert plan.status == PipelineStatus.EXECUTED
    assert plan.tool_calls[0].parameters == {
        "source_packet_id": "packet-4",
        "camera_frame_id": "agentview",
    }
    assert captured[0]["rgb"] == str(rgb)
    assert captured[0]["depth"] == str(depth)
    assert captured[0]["prior_depth"] == str(prior)
    assert captured[0]["prior_confidence"] == str(confidence)
    assert captured[0]["source_packet_id"] == "packet-4"


def test_pipeline_resolves_fresh_wrist_viewpoint_inputs(tmp_path: Path) -> None:
    rgb = tmp_path / "wrist-rgb.png"
    depth = tmp_path / "wrist-depth.png"
    Image.new("RGB", (8, 8), (0, 0, 0)).save(rgb)
    Image.new("I;16", (8, 8), 1000).save(depth)
    observation = EnvObservation(
        task="pick the cube",
        cameras=[
            CameraFrame(
                frame_id="wrist",
                role="wrist",
                rgb=[[[0, 0, 0]]],
                depth=[[1.0]],
                intrinsics={"fx": 100.0, "fy": 100.0, "cx": 4.0, "cy": 4.0},
                extrinsics={
                    "camera_frame": "opencv",
                    "frame_transform": "camera_to_world",
                    "camera_to_world": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                },
            )
        ],
        robot=RobotState(
            end_effector_pose={
                "xyz": [0.0, 0.0, 0.0],
                "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            gripper_state={"open": True},
        ),
        metadata={
            "image_artifacts": [
                {
                    "kind": "rgb",
                    "frame_id": "wrist",
                    "role": "wrist",
                    "path": str(rgb),
                    "packet_id": "packet-wrist",
                },
                {
                    "kind": "depth",
                    "frame_id": "wrist",
                    "role": "wrist",
                    "path": str(depth),
                    "packet_id": "packet-wrist",
                },
            ]
        },
    )
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    memory.add_observation(observation)
    memory.artifacts["compiled"] = {
        "value": {
            "type": "compiled_grasp",
            "schema_version": "openeta.compiled_grasp_seed.v1",
            "compiled_grasp_id": "compiled-1",
            "scene_epoch": 0,
            "target_anchor_world_xyz": [0.1, 0.2, 0.3],
        }
    }
    captured: list[dict] = []
    tools = build_default_tool_registry()
    tools.bind_handler(
        "propose_wrist_viewpoints",
        lambda context: captured.append(dict(context.parameters)) or ToolResult(True),
    )

    plan = ActionPipeline().compile(
        PlannerDecision(
            action_type="tool_call",
            action="propose_wrist_viewpoints",
            parameters={
                "compiled_grasp_id": "compiled-1",
                "source_packet_id": "packet-wrist",
                "camera_frame_id": "wrist",
            },
        ),
        observation=observation,
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )

    assert plan.status == PipelineStatus.EXECUTED
    assert captured[0]["compiled_grasp"]["compiled_grasp_id"] == "compiled-1"
    assert captured[0]["source_packet_id"] == "packet-wrist"
    assert captured[0]["camera_frame_id"] == "wrist"


def test_compile_grasp_seed_resolves_candidate_and_calibration_from_short_ids(
    tmp_path: Path,
) -> None:
    rgb = tmp_path / "rgb.png"
    depth = tmp_path / "depth.png"
    mask = tmp_path / "mask.png"
    rgb.write_bytes(b"rgb")
    depth.write_bytes(b"depth")
    mask.write_bytes(b"mask")
    observation = EnvObservation(
        task="pick the cube",
        cameras=[
            CameraFrame(
                frame_id="agentview",
                role="scene_primary",
                rgb=[],
                depth=[],
                intrinsics={"fx": 100.0, "fy": 100.0, "cx": 1.0, "cy": 1.0},
                extrinsics={
                    "camera_frame": "opencv",
                    "frame_transform": "camera_to_world",
                    "camera_to_world": [
                        [1.0, 0.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ],
                },
            )
        ],
        robot=RobotState(),
        metadata={
            "image_artifacts": [
                {
                    "kind": "rgb",
                    "frame_id": "agentview",
                    "role": "scene_primary",
                    "path": str(rgb),
                    "packet_id": "packet-grasp",
                },
                {
                    "kind": "depth",
                    "frame_id": "agentview",
                    "role": "scene_primary",
                    "path": str(depth),
                    "packet_id": "packet-grasp",
                },
            ]
        },
    )
    candidate = {
        "id": "candidate-7",
        "translation_xyz": [0.0, 0.0, 1.0],
        "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "width": 0.05,
    }
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    memory.add_observation(observation)
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {"kind": "tool_call", "name": "grasp_pose_estimate"},
                "status": "executed",
                "tool_calls": [
                    {
                        "name": "grasp_pose_estimate",
                        "status": "executed",
                        "result": {
                            "success": True,
                            "details": {
                                "outputs": {
                                    "result_id": "grasp-result-1",
                                    "scene_epoch": 0,
                                    "candidate_count": 1,
                                    "target_mask_quality": {
                                        "status": "usable",
                                        "area_fraction": 0.008,
                                    },
                                    "grasp_selection_advice": {
                                        "status": "skipped_single_candidate"
                                    },
                                    "grasp_candidates": [candidate],
                                    "source": {
                                        "mode": "targeted",
                                        "source_packet_id": "packet-grasp",
                                        "camera_frame_id": "agentview",
                                        "scene_epoch": 0,
                                        "rgb": str(rgb),
                                        "depth": str(depth),
                                        "object_mask": str(mask),
                                    },
                                }
                            },
                        },
                    }
                ],
            },
        )
    )
    captured: list[dict] = []
    tools = build_default_tool_registry()
    tools.bind_handler(
        "compile_grasp_seed",
        lambda context: captured.append(dict(context.parameters))
        or ToolResult(
            True,
            details={
                "schema_version": "openeta.compiled_grasp_seed.v1",
                "compiled_grasp_id": "compiled-short-ref-1",
                "candidate_id": "candidate-7",
                "scene_epoch": 0,
                "target_anchor_world_xyz": [0.1, 0.2, 0.12],
                "hover_pose": {
                    "frame": "world",
                    "xyz": [0.1, 0.2, 0.3],
                    "compiled_grasp_id": "compiled-short-ref-1",
                    "waypoint_role": "grasp_clearance",
                },
                "contact_pose": {
                    "frame": "world",
                    "xyz": [0.1, 0.2, 0.16],
                    "compiled_grasp_id": "compiled-short-ref-1",
                    "waypoint_role": "grasp_contact",
                },
            },
        ),
    )

    plan = ActionPipeline().compile(
        PlannerDecision(
            action_type="tool_call",
            action="compile_grasp_seed",
            parameters={
                "grasp_result_id": "grasp-result-1",
                "candidate_id": "candidate-7",
                "target_geometry_family": "boxed_item",
            },
        ),
        observation=observation,
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )

    assert plan.status == PipelineStatus.EXECUTED
    assert captured[0]["camera_pose"] == candidate
    assert captured[0]["camera_extrinsics"] == observation.cameras[0].extrinsics
    assert captured[0]["grasp_result_id"] == "grasp-result-1"
    assert captured[0]["target_geometry_family"] == "boxed_item"
    assert captured[0]["target_mask_quality"]["area_fraction"] == 0.008
    assert captured[0]["grasp_candidate_count"] == 1
    assert captured[0]["grasp_selection_advice"]["status"] == (
        "skipped_single_candidate"
    )
    # Memory consumes the same sanitized Agent action that is written to the
    # conversation ledger. It must reconstruct provenance from short ids rather
    # than depending on private host-expanded camera_pose fields.
    memory.add_action(plan.to_env_action())
    assert plan.tool_calls[0].parameters == {
        "grasp_result_id": "grasp-result-1",
        "candidate_id": "candidate-7",
        "target_geometry_family": "boxed_item",
    }
    retained = memory.retained_targeted_grasp()
    assert retained is not None
    assert retained["compiled_grasp_id"] == "compiled-short-ref-1"
    authorization = memory.resolve_compiled_contact_authorization(
        {
            "frame": "world",
            "xyz": [0.1, 0.2, 0.16],
            "compiled_grasp_id": "compiled-short-ref-1",
            "waypoint_role": "grasp_contact",
        }
    )
    assert authorization is not None
    assert authorization["compiled_grasp_id"] == "compiled-short-ref-1"


def test_ik_resolves_exact_full_viewpoint_pose_from_short_ids() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    target_pose = {
        "frame": "world",
        "xyz": [0.1, 0.2, 0.3],
        "rotation_matrix": [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
    }
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {"kind": "tool_call", "name": "propose_wrist_viewpoints"},
                "status": "executed",
                "tool_calls": [
                    {
                        "name": "propose_wrist_viewpoints",
                        "status": "executed",
                        "result": {
                            "success": True,
                            "details": {
                                "outputs": {
                                    "proposal_id": "wrist_viewpoint:one",
                                    "object_scene_epoch": 0,
                                    "robot_motion_epoch": 0,
                                    "candidates": [
                                        {
                                            "candidate_id": "wrist_view_00",
                                            "target_pose": target_pose,
                                        }
                                    ],
                                }
                            },
                        },
                    }
                ],
            },
        )
    )
    captured: list[dict] = []
    tools = build_default_tool_registry()
    tools.bind_handler(
        "ik_preview_check",
        lambda context: captured.append(dict(context.parameters)) or ToolResult(True),
    )

    plan = ActionPipeline().compile(
        PlannerDecision(
            action_type="tool_call",
            action="ik_preview_check",
            parameters={
                "viewpoint_proposal_id": "wrist_viewpoint:one",
                "candidate_id": "wrist_view_00",
            },
        ),
        observation=_observation(),
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )

    assert plan.status == PipelineStatus.EXECUTED
    assert captured[0]["target_pose"] == target_pose
    assert captured[0]["viewpoint_proposal_id"] == "wrist_viewpoint:one"


def test_tool_result_separates_operational_success_from_semantic_outcome() -> None:
    tools = build_default_tool_registry()

    def handler(context: ToolExecutionContext) -> ToolResult:
        return make_tool_result(
            context,
            success=True,
            outputs={
                "detection_count": 0,
                "detections": [],
                "same_view_recovery_handoff": {
                    "source_packet_id": "packet-4",
                    "camera_frame_id": "wrist",
                    "evidence_role": "target_object",
                },
            },
        )

    tools.bind_handler("sam3", handler)
    result = tools.call(
        "sam3",
        {"source_packet_id": "packet-4", "prompt": "cube"},
    )

    assert result.success is True
    assert result.details["operational_success"] is True
    assert result.details["semantic_outcome"] == "no_detection"
    assert result.details["facts_produced"] == [
        "outputs.detection_count",
        "outputs.same_view_recovery_handoff",
    ]
    assert {option["action"] for option in result.details["recovery_options"]} == {
        "change_viewpoint_or_distance",
        "preserve_same_view_for_point_grounding",
        "refine_grounding",
        "use_reference_or_point_prompt",
    }


def test_decision_state_is_bounded_index_over_packets_bundles_and_last_effect(
    tmp_path: Path,
) -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    _record_target_selection(memory, artifact_root=tmp_path)
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "status": "executed",
                "tool_calls": [
                    {
                        "name": "sam3",
                        "status": "executed",
                        "result": {
                            "success": True,
                            "details": {
                                "effect": "read_only",
                                "operational_success": True,
                                "semantic_outcome": "detections_available",
                                "facts_produced": ["outputs.detections"],
                                "recovery_options": [],
                                "outputs": {
                                    "result_id": "sam3-result-1",
                                    "detection_count": 1,
                                    "grasp_selection_advice": {
                                        "schema_version": "openeta.grasp_selection_advice.v1",
                                        "recommended_candidate_id": "gpe-001",
                                        "confidence": 0.8,
                                    },
                                    "grasp_selection_bundle": {
                                        "schema_version": "openeta.grasp_selection_bundle.v1",
                                        "bundle_id": "grasp-selection:example",
                                        "bundle_ref": "/session/selection_bundle.json",
                                    },
                                },
                                "artifacts": [
                                    {"path": "/session/selection.contact-sheet.png"}
                                ],
                            },
                        },
                    }
                ],
            },
        )
    )
    tools = build_default_tool_registry()
    tools.bind_handler("grasp_pose_estimate", lambda _context: ToolResult(True))
    observation = _observation()
    observation.cameras[0].extrinsics = {
        "frame_transform": "camera_to_world",
        "pos": [0.8, 0.0, 0.6],
        "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    }
    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=tools,
        skills=build_default_skill_registry(),
    )["agent_context"]
    state = context["decision_state"]

    assert state["schema_version"] == "openeta.decision_state.v1"
    assert state["current_observation_packet"]["packet_ids"] == ["packet-4"]
    assert state["current_observation_packet"]["camera_calibrations"][0][
        "extrinsics"
    ]["frame_transform"] == "camera_to_world"
    assert state["active_bundles"]["grasp_pose_estimate"]["status"] == "ready"
    assert state["last_action_effect"]["semantic_outcome"] == "detections_available"
    assert state["last_action_effect"]["outputs"]["result_id"] == "sam3-result-1"
    assert state["last_action_effect"]["grasp_selection_advice"][
        "recommended_candidate_id"
    ] == "gpe-001"
    assert state["last_action_effect"]["grasp_selection_bundle"]["bundle_id"] == (
        "grasp-selection:example"
    )
    assert state["last_action_effect"]["artifact_refs"] == [
        "/session/selection.contact-sheet.png"
    ]
    assert "grasp_pose_estimate" in state["available_tools"]


def test_superseded_target_blocks_old_contact_but_allows_clearance_waypoint() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    memory.save_fact(
        "selected_sam3_detection",
        {
            "result_id": "sam3-current",
            "id": "detection_002",
            "mask_ref": "/session/current-mask.png",
            "source_image": "/session/current.png",
            "scene_epoch": 0,
        },
        source="select_sam3_detection",
    )
    memory.save_fact(
        "grasp_provenance",
        {
            "schema_version": "openeta.grasp_provenance.v1",
            "evidence_id": "grasp:old",
            "target_evidence_id": "sam3:sam3-old:detection_001",
            "candidate_id": "grasp-old-0",
            "compiled_grasp_id": "compiled-old",
            "object_scene_epoch": 0,
            "robot_motion_epoch": 1,
        },
        source="compile_grasp_seed_evidence_graph",
    )
    memory.save_artifact(
        "compiled-old",
        {
            "type": "compiled_grasp",
            "compiled_grasp_id": "compiled-old",
            "hover_pose": {"frame": "world", "xyz": [0.1, 0.2, 0.30]},
            "contact_pose": {"frame": "world", "xyz": [0.1, 0.2, 0.15]},
        },
        source="compile_grasp_seed",
    )
    memory.save_fact(
        "grasp_input_bundles",
        {
            "schema_version": "openeta.grasp_input_bundle_store.v1",
            "active_bundle_id": "grasp:new-target",
            "public": {
                "status": "ready",
                "bundle_id": "grasp:new-target",
                "target_evidence_id": "sam3:sam3-current:detection_002",
            },
            "bundles": {},
        },
        source="host_provenance_bundle_resolver",
    )
    graph = memory.provenance_evidence_graph()
    tool_context = {
        "provenance_evidence_graph": graph,
        "grasp_input_bundle": {
            "status": "ready",
            "bundle_id": "grasp:new-target",
        },
    }

    contact_errors = _validate_compiled_grasp_target_freshness(
        PlannerDecision(
            action_type="tool_call",
            action="move_to",
            parameters={
                "target_pose": {"frame": "world", "xyz": [0.1, 0.2, 0.15]}
            },
        ),
        tool_context=tool_context,
    )
    clearance_errors = _validate_compiled_grasp_target_freshness(
        PlannerDecision(
            action_type="tool_call",
            action="move_to",
            parameters={
                "target_pose": {"frame": "world", "xyz": [0.1, 0.2, 0.30]}
            },
        ),
        tool_context=tool_context,
    )
    close_errors = _validate_compiled_grasp_target_freshness(
        PlannerDecision(
            action_type="tool_call",
            action="gripper_control",
            parameters={"position": 0},
        ),
        tool_context=tool_context,
    )

    assert graph["inconsistencies"][0]["code"] == "compiled_grasp_target_superseded"
    assert "sam3:sam3-old:detection_001" in contact_errors[0]
    assert "sam3:sam3-current:detection_002" in contact_errors[0]
    assert "grasp:new-target" in contact_errors[0]
    assert clearance_errors == []
    assert close_errors

    tools = build_default_tool_registry()
    tools.bind_handler("move_to", lambda _context: ToolResult(True))
    _record_ik_receipt(
        memory,
        receipt_id="ik-old-contact",
        target_pose={"frame": "world", "xyz": [0.1, 0.2, 0.15]},
    )
    blocked_plan = ActionPipeline().compile(
        PlannerDecision(
            action_type="tool_call",
            action="move_to",
            parameters={"ik_receipt_id": "ik-old-contact"},
        ),
        observation=_observation(),
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )

    assert blocked_plan.status is PipelineStatus.BLOCKED
    assert blocked_plan.metadata["provenance_integrity_gate"]["blocked"] is True
    assert "sam3:sam3-current:detection_002" in blocked_plan.tool_calls[0].reason


def test_failed_compiled_contact_receipt_blocks_gripper_close() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    memory.save_fact(
        "selected_sam3_detection",
        {
            "result_id": "sam3-current",
            "id": "detection_000",
            "mask_ref": "/session/current-mask.png",
            "source_image": "/session/current.png",
            "scene_epoch": 0,
        },
        source="select_sam3_detection",
    )
    memory.save_fact(
        GRASP_PROVENANCE_KEY,
        {
            "schema_version": "openeta.grasp_provenance.v1",
            "evidence_id": "grasp:current",
            "target_evidence_id": "sam3:sam3-current:detection_000",
            "candidate_id": "grasp-current-0",
            "compiled_grasp_id": "compiled-current",
            "object_scene_epoch": 0,
            "robot_motion_epoch": 0,
        },
        source="compile_grasp_seed_evidence_graph",
    )
    memory.save_artifact(
        "compiled-current",
        {
            "type": "compiled_grasp",
            "compiled_grasp_id": "compiled-current",
            "candidate_id": "grasp-current-0",
            "scene_epoch": 0,
            "target_anchor_world_xyz": [0.1, 0.2, 0.1],
            "hover_pose": {"frame": "world", "xyz": [0.1, 0.2, 0.25]},
            "contact_pose": {
                "frame": "world",
                "compiled_grasp_id": "compiled-current",
                "source_grasp_id": "grasp-current-0",
                "waypoint_role": "grasp_contact",
                "xyz": [0.1, 0.2, 0.1],
            },
        },
        source="compile_grasp_seed",
    )

    def contact_action(
        *,
        reached: bool,
        steps: int,
        collision: dict,
        actual_xyz: list[float] | None = None,
        position_error_m: float | None = None,
        max_axis_position_error_m: float | None = None,
        orientation_error_rad: float = 0.0,
    ) -> EnvAction:
        target_pose = {
            "frame": "world",
            "compiled_grasp_id": "compiled-current",
            "source_grasp_id": "grasp-current-0",
            "waypoint_role": "grasp_contact",
            "xyz": [0.1, 0.2, 0.1],
        }
        actual = actual_xyz or [0.1, 0.2, 0.1 if reached else 0.16]
        position_error = (
            position_error_m
            if position_error_m is not None
            else (0.0 if reached else 0.06)
        )
        max_axis_error = (
            max_axis_position_error_m
            if max_axis_position_error_m is not None
            else position_error
        )
        details = {
            "operational_success": reached,
            "outputs": {
                "motion_summary": {
                    "reached_target": reached,
                    "steps_executed": steps,
                    "stop_reason": "target_reached" if reached else "collision_detected",
                    "start": {"xyz": [0.1, 0.2, 0.2]},
                    "end": {"xyz": actual},
                    "position_error_m": position_error,
                    "max_axis_position_error_m": max_axis_error,
                    "orientation_error_rad": orientation_error_rad,
                    "collision": collision,
                }
            },
            "diagnostics": (
                []
                if reached
                else [{"code": "simulator_mcp_collision", "candidate_rejection": False}]
            ),
        }
        return EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "move_to",
                    "parameters": {"target_pose": target_pose},
                },
                "status": "executed",
                "tool_calls": [
                    {
                        "name": "move_to",
                        "status": "executed",
                        "parameters": {"target_pose": target_pose},
                        "result": {"success": True, "details": details},
                    }
                ],
            },
        )

    memory.add_action(
        contact_action(
            reached=False,
            steps=4,
            collision={
                "detected": True,
                "geom1_name": "gripper0_hand_collision",
                "geom2_name": "tomato_sauce_1_g4",
            },
        )
    )
    receipt = memory.latest_compiled_contact_execution()
    assert receipt["reached_target"] is False
    assert receipt["actual_xyz"] == [0.1, 0.2, 0.16]

    visible_context = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )["agent_context"]
    visible_receipt = visible_context["decision_state"][
        "latest_compiled_contact_execution"
    ]
    assert visible_receipt["reached_target"] is False
    assert visible_receipt["collision"]["geom2_name"] == "tomato_sauce_1_g4"

    tools = build_default_tool_registry()
    tools.bind_handler("gripper_control", lambda _context: ToolResult(True))
    pipeline = ActionPipeline()
    blocked = pipeline.compile(
        PlannerDecision(
            action_type="tool_call",
            action="gripper_control",
            parameters={"position": 0},
        ),
        observation=_observation(),
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )

    assert blocked.status is PipelineStatus.BLOCKED
    assert blocked.metadata["repair_bundle"]["code"] == "compiled_contact_not_reached"
    reason = blocked.tool_calls[0].reason
    assert "reached_target=false" in reason
    assert "actual_eef_xyz=[0.1, 0.2, 0.16]" in reason
    assert "position_error_m=0.06 (limit=0.01)" in reason
    assert "Collision diagnostics from the preceding arm motion do not block" in reason

    memory.add_action(
        contact_action(
            reached=False,
            steps=17,
            actual_xyz=[0.1003, 0.2002, 0.1001],
            position_error_m=0.0088,
            max_axis_position_error_m=0.0063,
            orientation_error_rad=0.019,
            collision={
                "detected": True,
                "geom1_name": "robot0_link7_collision",
                "geom2_name": "milk_1_g1",
            },
        )
    )
    near_contact_receipt = memory.latest_compiled_contact_execution()
    assert near_contact_receipt["max_axis_position_error_m"] == 0.0063
    assert near_contact_receipt["orientation_error_rad"] == 0.019
    collision_does_not_block_finger_close = pipeline.compile(
        PlannerDecision(
            action_type="tool_call",
            action="gripper_control",
            parameters={"position": 0},
        ),
        observation=_observation(),
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )
    assert collision_does_not_block_finger_close.status is not PipelineStatus.BLOCKED

    memory.add_action(
        contact_action(
            reached=False,
            steps=8,
            actual_xyz=[0.1003, 0.2002, 0.1001],
            position_error_m=0.0004,
            max_axis_position_error_m=0.0003,
            orientation_error_rad=0.301,
            collision={"detected": False},
        )
    )
    excessive_orientation_error = pipeline.compile(
        PlannerDecision(
            action_type="tool_call",
            action="gripper_control",
            parameters={"position": 0},
        ),
        observation=_observation(),
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )
    assert excessive_orientation_error.status is PipelineStatus.BLOCKED
    assert "orientation_error_rad=0.301 (limit=0.3)" in (
        excessive_orientation_error.tool_calls[0].reason or ""
    )

    memory.add_action(contact_action(reached=True, steps=3, collision={"detected": False}))
    allowed = pipeline.compile(
        PlannerDecision(
            action_type="tool_call",
            action="gripper_control",
            parameters={"position": 0},
        ),
        observation=_observation(),
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )
    assert allowed.status is not PipelineStatus.BLOCKED

    memory.save_artifact(
        "compiled-next",
        {
            "type": "compiled_grasp",
            "compiled_grasp_id": "compiled-next",
            "scene_epoch": 0,
            "target_anchor_world_xyz": [0.1, 0.2, 0.1],
            "contact_pose": {
                "frame": "world",
                "compiled_grasp_id": "compiled-next",
                "waypoint_role": "grasp_contact",
                "xyz": [0.1, 0.2, 0.1],
            },
        },
        source="compile_grasp_seed",
    )
    memory.save_fact(
        GRASP_PROVENANCE_KEY,
        {
            "schema_version": "openeta.grasp_provenance.v1",
            "compiled_grasp_id": "compiled-next",
            "target_evidence_id": "sam3:sam3-current:detection_000",
            "object_scene_epoch": 0,
        },
        source="compile_grasp_seed_evidence_graph",
    )
    same_target_new_plan = pipeline.compile(
        PlannerDecision(
            action_type="tool_call",
            action="gripper_control",
            parameters={"position": 0},
        ),
        observation=_observation(),
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )
    assert same_target_new_plan.status is not PipelineStatus.BLOCKED
    assert memory.resolve_active_attachment_candidate()["compiled_grasp_id"] == (
        "compiled-current"
    )

    memory.save_fact(
        GRASP_PROVENANCE_KEY,
        {
            "schema_version": "openeta.grasp_provenance.v1",
            "compiled_grasp_id": "compiled-next",
            "target_evidence_id": "sam3:other-target:detection_001",
            "object_scene_epoch": 0,
        },
        source="compile_grasp_seed_evidence_graph",
    )
    cross_target_mismatch = pipeline.compile(
        PlannerDecision(
            action_type="tool_call",
            action="gripper_control",
            parameters={"position": 0},
        ),
        observation=_observation(),
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )
    assert cross_target_mismatch.status is PipelineStatus.BLOCKED
    assert cross_target_mismatch.metadata["repair_bundle"]["code"] == (
        "compiled_contact_receipt_mismatch"
    )


def test_release_is_blocked_after_failed_attached_motion_with_actionable_evidence() -> None:
    memory = AgentMemory()
    memory.start_session(task="place the milk in the basket")
    memory.save_fact(
        GRIPPER_COMMAND_STATE_KEY,
        {"position": 0, "state": "closed", "latched": True},
        source="test",
    )
    memory.save_fact(
        ATTACHMENT_EVIDENCE_KEY,
        {"verdict": "PASS", "compiled_grasp_id": "compiled-milk"},
        source="test",
    )
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {"kind": "tool_call", "name": "move_to", "parameters": {}},
                "status": "executed",
                "tool_calls": [
                    {
                        "name": "move_to",
                        "status": "executed",
                        "parameters": {},
                        "result": {
                            "success": True,
                            "details": {
                                "semantic_outcome": "target_not_reached",
                                "outputs": {
                                    "motion_summary": {
                                        "stop_reason": "collision_detected",
                                        "reached_target": False,
                                        "collision": {
                                            "detected": True,
                                            "geom1_name": "milk_1_g1",
                                            "geom2_name": "basket_1_g4",
                                        },
                                    },
                                    "pose_feedback": {
                                        "actual_xyz": [-0.05, 0.26, 0.25],
                                        "requested_xyz": [-0.07, 0.39, 0.14],
                                        "reached_target": False,
                                    },
                                },
                            },
                        },
                    }
                ],
            },
        )
    )
    tools = build_default_tool_registry()
    tools.bind_handler("gripper_control", lambda _context: ToolResult(True))

    blocked = ActionPipeline().compile(
        PlannerDecision(
            action_type="tool_call",
            action="gripper_control",
            parameters={"position": 1},
        ),
        observation=_observation(),
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )

    assert blocked.status is PipelineStatus.BLOCKED
    assert blocked.metadata["repair_bundle"]["code"] == (
        "attached_release_after_failed_motion"
    )
    reason = blocked.tool_calls[0].reason or ""
    assert "actual_eef_xyz=[-0.05, 0.26, 0.25]" in reason
    assert "milk_1_g1" in reason and "basket_1_g4" in reason
    assert "successful subsequent carrying motion clears this check" in reason


def test_camera_pose_to_world_reference_remains_in_world_evidence() -> None:
    memory = AgentMemory()
    memory.start_session(task="place the milk in the basket")
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "camera_pose_to_world",
                    "parameters": {
                        "placement_result_id": "anyplace-result:test",
                        "candidate_id": "placement_000",
                    },
                },
                "status": "executed",
                "tool_calls": [
                    {
                        "name": "camera_pose_to_world",
                        "status": "executed",
                        "result": {
                            "success": True,
                            "details": {
                                "outputs": {
                                    "placement_result_id": "anyplace-result:test",
                                    "candidate_id": "placement_000",
                                    "world_pose": {
                                        "frame": "world",
                                        "translation_xyz": [-0.07, 0.39, 0.135],
                                        "rotation_matrix": [
                                            [1.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0],
                                            [0.0, 0.0, 1.0],
                                        ],
                                    },
                                    "placement_reference": {
                                        "schema_version": (
                                            "openeta.placement_world_reference.v1"
                                        ),
                                        "semantic_role": "low_release_geometric_reference",
                                        "execution_authorized": False,
                                        "required_before_motion": ["agent_authored_waypoint"],
                                        "agent_release_options": [
                                            {
                                                "mode": (
                                                    "gravity_assisted_open_container_drop"
                                                )
                                            }
                                        ],
                                    },
                                }
                            },
                        },
                    }
                ],
            },
        )
    )

    reference = memory.placement_world_reference()
    assert reference is not None
    assert reference["world_pose"]["translation_xyz"] == [-0.07, 0.39, 0.135]
    assert reference["agent_release_options"][0]["mode"] == (
        "gravity_assisted_open_container_drop"
    )
    projected = memory.planning_context()["world_evidence"]
    assert projected["placement_world_reference"]["value"]["candidate_id"] == (
        "placement_000"
    )


def test_compiled_contact_rejects_cross_axis_sweep_without_requiring_a_stage() -> None:
    contact_pose = {
        "frame": "world",
        "compiled_grasp_id": "compiled-corridor",
        "source_grasp_id": "candidate-0",
        "waypoint_role": "grasp_contact",
        "xyz": [0.15, 0.0, 0.10],
    }
    compiled = {
        "type": "compiled_grasp",
        "compiled_grasp_id": "compiled-corridor",
        "scene_epoch": 0,
        "approach_world_xyz": [1.0, 0.0, 0.0],
        "hover_pose": {
            "frame": "world",
            "compiled_grasp_id": "compiled-corridor",
            "waypoint_role": "grasp_clearance",
            "xyz": [0.0, 0.0, 0.10],
        },
        "contact_pose": dict(contact_pose),
    }

    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    observation = _observation()
    observation.robot.end_effector_pose = {"xyz": [0.0, 0.08, 0.25]}
    memory.add_observation(observation)
    memory.save_artifact("compiled-corridor", compiled, source="compile_grasp_seed")

    blocked = memory.compiled_grasp_target_gate_error(
        tool_name="move_to",
        parameters={"target_pose": dict(contact_pose)},
    )

    assert blocked is not None
    assert blocked.startswith("compiled_contact_approach_misaligned:")
    assert "current_eef_xyz=[0.0, 0.08, 0.25]" in blocked
    assert "lateral_offset_m=" in blocked
    assert "not a required hover/descend task stage" in blocked

    aligned_memory = AgentMemory()
    aligned_memory.start_session(task="pick the cube")
    aligned_observation = _observation()
    aligned_observation.robot.end_effector_pose = {"xyz": [0.0, 0.0, 0.10]}
    aligned_memory.add_observation(aligned_observation)
    aligned_memory.save_artifact(
        "compiled-corridor", compiled, source="compile_grasp_seed"
    )
    assert aligned_memory.compiled_grasp_target_gate_error(
        tool_name="move_to",
        parameters={"target_pose": dict(contact_pose)},
    ) is None

    near_memory = AgentMemory()
    near_memory.start_session(task="pick the cube")
    near_observation = _observation()
    near_observation.robot.end_effector_pose = {"xyz": [0.13, 0.025, 0.10]}
    near_memory.add_observation(near_observation)
    near_memory.save_artifact(
        "compiled-corridor", compiled, source="compile_grasp_seed"
    )
    assert near_memory.compiled_grasp_target_gate_error(
        tool_name="move_to",
        parameters={"target_pose": dict(contact_pose)},
    ) is None


def test_compiled_contact_rejects_large_pending_rotation_without_task_stage() -> None:
    contact_pose = {
        "frame": "world",
        "compiled_grasp_id": "compiled-orientation-entry",
        "source_grasp_id": "candidate-0",
        "waypoint_role": "grasp_contact",
        "xyz": [0.15, 0.0, 0.10],
        "rotation_matrix": [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    }
    compiled = {
        "type": "compiled_grasp",
        "compiled_grasp_id": "compiled-orientation-entry",
        "scene_epoch": 0,
        "approach_world_xyz": [1.0, 0.0, 0.0],
        "contact_pose": dict(contact_pose),
    }
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    observation = _observation()
    observation.robot.end_effector_pose = {
        "xyz": [0.0, 0.0, 0.10],
        "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    memory.add_observation(observation)
    memory.save_artifact(
        "compiled-orientation-entry",
        compiled,
        source="compile_grasp_seed",
    )

    blocked = memory.compiled_grasp_target_gate_error(
        tool_name="move_to",
        parameters={"target_pose": dict(contact_pose)},
    )

    assert blocked is not None
    assert blocked.startswith("compiled_contact_orientation_misaligned:")
    assert "orientation_delta_rad=1.5708 (limit=0.30)" in blocked
    assert "not a required alignment stage" in blocked

    contact_pose["rotation_matrix"] = [
        [0.98006658, -0.19866933, 0.0],
        [0.19866933, 0.98006658, 0.0],
        [0.0, 0.0, 1.0],
    ]
    aligned_memory = AgentMemory()
    aligned_memory.start_session(task="pick the cube")
    aligned_memory.add_observation(observation)
    aligned_memory.save_artifact(
        "compiled-orientation-entry",
        {**compiled, "contact_pose": dict(contact_pose)},
        source="compile_grasp_seed",
    )
    assert aligned_memory.compiled_grasp_target_gate_error(
        tool_name="move_to",
        parameters={"target_pose": dict(contact_pose)},
    ) is None


def test_failed_agent_chosen_clearance_receipt_blocks_dependent_contact() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    compiled_id = "compiled-clearance"
    clearance_pose = {
        "frame": "world",
        "compiled_grasp_id": compiled_id,
        "source_grasp_id": "grasp-0",
        "waypoint_role": "grasp_clearance",
        "xyz": [0.1, 0.2, 0.25],
    }
    contact_pose = {
        "frame": "world",
        "compiled_grasp_id": compiled_id,
        "source_grasp_id": "grasp-0",
        "waypoint_role": "grasp_contact",
        "xyz": [0.1, 0.2, 0.1],
    }
    memory.save_artifact(
        compiled_id,
        {
            "type": "compiled_grasp",
            "compiled_grasp_id": compiled_id,
            "candidate_id": "grasp-0",
            "scene_epoch": 0,
            "hover_pose": dict(clearance_pose),
            "contact_pose": dict(contact_pose),
        },
        source="compile_grasp_seed",
    )

    def clearance_action(
        *,
        reached: bool,
        position_error_m: float = 0.04,
        orientation_error_rad: float = 0.45,
    ) -> EnvAction:
        details = {
            "operational_success": reached,
            "outputs": {
                "motion_summary": {
                    "reached_target": reached,
                    "steps_executed": 100 if not reached else 4,
                    "stop_reason": "target_reached" if reached else "iteration_limit",
                    "start": {"xyz": [0.0, 0.0, 0.2]},
                    "end": {"xyz": [0.098, 0.203, 0.247]},
                    "position_error_m": position_error_m,
                    "orientation_error_rad": (
                        orientation_error_rad if not reached else 0.05
                    ),
                }
            },
            "diagnostics": (
                []
                if reached
                else [{"code": "simulator_mcp_target_not_reached"}]
            ),
        }
        return EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "move_to",
                    "parameters": {"target_pose": dict(clearance_pose)},
                },
                "status": "executed",
                "tool_calls": [
                    {
                        "name": "move_to",
                        "status": "executed",
                        "parameters": {"target_pose": dict(clearance_pose)},
                        "result": {"success": True, "details": details},
                    }
                ],
            },
        )

    memory.add_action(clearance_action(reached=False))
    receipt = memory.latest_compiled_clearance_execution()
    assert receipt["reached_target"] is False
    assert receipt["actual_xyz"] == [0.098, 0.203, 0.247]

    tools = build_default_tool_registry()
    tools.bind_handler("move_to", lambda _context: ToolResult(True))
    _record_ik_receipt(
        memory,
        receipt_id="ik-contact-after-failed-clearance",
        target_pose=contact_pose,
    )
    blocked = ActionPipeline().compile(
        PlannerDecision(
            action_type="tool_call",
            action="move_to",
            parameters={"ik_receipt_id": "ik-contact-after-failed-clearance"},
        ),
        observation=_observation(),
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )

    assert blocked.status is PipelineStatus.BLOCKED
    assert blocked.metadata["repair_bundle"]["code"] == (
        "compiled_clearance_not_reached"
    )
    reason = blocked.tool_calls[0].reason
    assert "reached_target=false" in reason
    assert "actual_eef_xyz=[0.098, 0.203, 0.247]" in reason
    assert "orientation_error_rad=0.45" in reason
    assert "position<=0.010m and orientation<=0.30rad" in reason
    assert blocked.metadata["repair_bundle"]["execution_evidence"][
        "latest_compiled_clearance_execution"
    ]["stop_reason"] == "iteration_limit"

    memory.add_action(clearance_action(reached=True))
    assert (
        memory.compiled_grasp_target_gate_error(
            tool_name="move_to",
            parameters={"target_pose": dict(contact_pose)},
        )
        is None
    )


def test_near_clearance_residual_allows_agent_chosen_contact_without_stage() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    compiled_id = "compiled-near-clearance"
    clearance_pose = {
        "frame": "world",
        "compiled_grasp_id": compiled_id,
        "source_grasp_id": "grasp-0",
        "waypoint_role": "grasp_clearance",
        "xyz": [0.1, 0.2, 0.25],
    }
    contact_pose = {
        "frame": "world",
        "compiled_grasp_id": compiled_id,
        "source_grasp_id": "grasp-0",
        "waypoint_role": "grasp_contact",
        "xyz": [0.1, 0.2, 0.1],
    }
    memory.save_artifact(
        compiled_id,
        {
            "type": "compiled_grasp",
            "compiled_grasp_id": compiled_id,
            "candidate_id": "grasp-0",
            "scene_epoch": 0,
            "hover_pose": dict(clearance_pose),
            "contact_pose": dict(contact_pose),
        },
        source="compile_grasp_seed",
    )
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "move_to",
                    "parameters": {"target_pose": dict(clearance_pose)},
                },
                "status": "executed",
                "tool_calls": [
                    {
                        "name": "move_to",
                        "status": "executed",
                        "parameters": {"target_pose": dict(clearance_pose)},
                        "result": {
                            "success": True,
                            "details": {
                                "operational_success": False,
                                "outputs": {
                                    "motion_summary": {
                                        "reached_target": False,
                                        "steps_executed": 100,
                                        "stop_reason": "iteration_limit",
                                        "start": {"xyz": [0.0, 0.0, 0.2]},
                                        "end": {"xyz": [0.098, 0.203, 0.247]},
                                        "position_error_m": 0.0052,
                                        "orientation_error_rad": 0.253,
                                    }
                                },
                            },
                        },
                    }
                ],
            },
        )
    )

    assert (
        memory._compiled_contact_clearance_gate_error(
            tool_name="move_to",
            parameters={"target_pose": dict(contact_pose)},
        )
        is None
    )


def test_pipeline_reports_compiled_grasp_residual_budget_repair() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    memory.add_observation(_observation())
    memory.save_artifact(
        "compiled-adjustment",
        {
            "type": "compiled_grasp",
            "compiled_grasp_id": "compiled-adjustment",
            "scene_epoch": 0,
            "contact_pose": {
                "frame": "world",
                "xyz": [0.1, 0.2, 0.15],
                "compiled_grasp_id": "compiled-adjustment",
                "waypoint_role": "grasp_contact",
            },
        },
        source="compile_grasp_seed",
    )
    tools = build_default_tool_registry()
    tools.bind_handler("move_to", lambda _context: ToolResult(True))
    _record_ik_receipt(
        memory,
        receipt_id="ik-over-budget-adjustment",
        target_pose={
            "frame": "world",
            "xyz": [0.13, 0.2, 0.15],
            "compiled_grasp_id": "compiled-adjustment",
            "waypoint_role": "grasp_contact",
        },
    )

    blocked = ActionPipeline().compile(
        PlannerDecision(
            action_type="tool_call",
            action="move_to",
            parameters={"ik_receipt_id": "ik-over-budget-adjustment"},
        ),
        observation=_observation(),
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )

    assert blocked.status is PipelineStatus.BLOCKED
    assert blocked.metadata["repair_bundle"]["code"] == (
        "compiled_grasp_adjustment_out_of_bounds"
    )
    assert "per-call limit of 0.020 m" in blocked.tool_calls[0].reason
    assert "Last accepted residual" in blocked.tool_calls[0].reason
    assert "not a limit on travel distance from the current EEF" in (
        blocked.tool_calls[0].reason
    )
    assert "exact host reference xyz [0.1, 0.2, 0.15]" in blocked.tool_calls[0].reason
    assert "omit compiled_grasp_id and waypoint_role" in blocked.tool_calls[0].reason


def test_position_only_compiled_move_requires_matching_feasible_ik_policy() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    memory.add_observation(_observation())
    compiled_id = "compiled-position-only"
    target_pose = {
        "frame": "world",
        "xyz": [0.1, 0.2, 0.15],
        "compiled_grasp_id": compiled_id,
        "waypoint_role": "grasp_contact",
    }
    memory.save_artifact(
        compiled_id,
        {
            "type": "compiled_grasp",
            "compiled_grasp_id": compiled_id,
            "scene_epoch": 0,
            "contact_pose": {
                **target_pose,
                "rotation_matrix": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
            },
        },
        source="compile_grasp_seed",
    )

    unverified = memory.compiled_grasp_target_gate_error(
        tool_name="move_to",
        parameters={"target_pose": target_pose, "tolerance": 0.01},
    )
    assert unverified is not None
    assert "unverified_orientation_policy" in unverified

    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "ik_preview_check",
                    "parameters": {
                        "target_pose": target_pose,
                        "preserve_current_orientation": True,
                        "position_tolerance_m": 0.01,
                    },
                },
                "status": "executed",
                "tool_calls": [
                    {
                        "name": "ik_preview_check",
                        "status": "executed",
                        "parameters": {
                            "target_pose": target_pose,
                            "preserve_current_orientation": True,
                            "position_tolerance_m": 0.01,
                        },
                        "result": {
                            "success": True,
                            "details": {
                                "outputs": {
                                    "ik_preview_receipt": {
                                        "schema_version": "openeta.ik_preview_receipt.v1",
                                        "receipt_id": "ik-preserve-current-1",
                                        "classification": "feasible",
                                        "orientation_policy": "preserve_current",
                                    }
                                }
                            },
                        },
                    }
                ],
            },
        )
    )

    assert memory.compiled_grasp_target_gate_error(
        tool_name="move_to",
        parameters={"target_pose": target_pose, "tolerance": 0.01},
    ) is None


def test_contact_authorization_is_resolved_from_active_host_grasp_evidence() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    memory.add_observation(_observation())
    compiled_id = "compiled-contact-auth"
    memory.save_artifact(
        compiled_id,
        {
            "type": "compiled_grasp",
            "compiled_grasp_id": compiled_id,
            "scene_epoch": 0,
            "target_anchor_world_xyz": [0.1, 0.2, 0.12],
            "contact_pose": {
                "frame": "world",
                "xyz": [0.1, 0.2, 0.16],
                "compiled_grasp_id": compiled_id,
                "waypoint_role": "grasp_contact",
            },
        },
        source="compile_grasp_seed",
    )
    memory.save_fact(
        GRASP_PROVENANCE_KEY,
        {
            "schema_version": "openeta.grasp_provenance.v1",
            "compiled_grasp_id": compiled_id,
            "target_evidence_id": "sam3:result:target",
            "object_scene_epoch": 0,
        },
        source="host_provenance_bundle_resolver",
    )

    authorization = memory.resolve_compiled_contact_authorization(
        {
            "frame": "world",
            "xyz": [0.1, 0.2, 0.16],
            "compiled_grasp_id": compiled_id,
            "waypoint_role": "grasp_contact",
        }
    )

    assert authorization == {
        "schema_version": "openeta.contact_authorization.v1",
        "compiled_grasp_id": compiled_id,
        "waypoint_role": "grasp_contact",
        "target_anchor_world_xyz": [0.1, 0.2, 0.12],
        "target_evidence_id": "sam3:result:target",
        "object_scene_epoch": 0,
    }
    assert memory.resolve_compiled_contact_authorization(
        {
            "frame": "world",
            "xyz": [0.1, 0.2, 0.20],
            "compiled_grasp_id": compiled_id,
            "waypoint_role": "grasp_clearance",
        }
    ) is None
    assert memory.resolve_active_attachment_candidate() == authorization


def test_gripper_reopen_invalidates_contact_geometry_but_preserves_identity_lineage() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    compiled_id = "compiled-contact-retired"
    memory.save_artifact(
        compiled_id,
        {
            "type": "compiled_grasp",
            "compiled_grasp_id": compiled_id,
            "scene_epoch": 0,
            "target_anchor_world_xyz": [0.1, 0.2, 0.12],
            "contact_pose": {
                "frame": "world",
                "xyz": [0.1, 0.2, 0.16],
                "compiled_grasp_id": compiled_id,
                "waypoint_role": "grasp_contact",
            },
        },
        source="compile_grasp_seed",
    )
    memory.save_fact(
        GRASP_PROVENANCE_KEY,
        {
            "schema_version": "openeta.grasp_provenance.v1",
            "evidence_id": "grasp:retired",
            "compiled_grasp_id": compiled_id,
            "target_evidence_id": "sam3:result:target",
            "target_identity_anchor_id": "target-anchor-1",
            "candidate": {"id": "candidate-retired"},
            "source": {"mode": "targeted"},
            "object_scene_epoch": 0,
        },
        source="host_provenance_bundle_resolver",
    )
    memory.save_fact(
        GRIPPER_COMMAND_STATE_KEY,
        {"position": 0, "state": "closed", "latched": True},
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
                "status": "executed",
                "tool_calls": [
                    {
                        "name": "gripper_control",
                        "status": "executed",
                        "parameters": {"position": 1},
                        "result": {"success": True, "details": {"outputs": {}}},
                    }
                ],
            },
        )
    )

    assert memory.retained_targeted_grasp() is None
    graph = memory.provenance_evidence_graph()
    node = next(item for item in graph["nodes"] if item["kind"] == "compiled_targeted_grasp")
    assert node["freshness"] == "invalidated_contact_geometry"
    assert node["target_identity_anchor_id"] == "target-anchor-1"
    assert memory.resolve_active_attachment_candidate() is None
    with pytest.raises(ValueError, match="contact geometry was invalidated"):
        memory.resolve_compiled_contact_authorization(
            {
                "frame": "world",
                "xyz": [0.1, 0.2, 0.16],
                "compiled_grasp_id": compiled_id,
                "waypoint_role": "grasp_contact",
            }
        )


def test_gripper_close_state_retains_tentative_proxy_receipt_for_probe_gate() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    receipt = {
        "schema_version": "openeta.attachment_proxy_receipt.v1",
        "status": "tentative",
        "reason": "non_empty_close_with_tentative_safety_proxy",
    }
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "gripper_control",
                    "parameters": {"position": 0},
                },
                "status": "executed",
                "tool_calls": [
                    {
                        "name": "gripper_control",
                        "status": "executed",
                        "parameters": {"position": 0},
                        "result": {
                            "success": True,
                            "details": {
                                "outputs": {"attachment_proxy_receipt": receipt}
                            },
                        },
                    }
                ],
            },
        )
    )

    assert memory.gripper_command_state()["attachment_proxy_receipt"] == receipt


def test_feasible_full_pose_ik_receipt_resolves_private_execution_seed() -> None:
    memory = AgentMemory()
    memory.start_session(task="reach the target")
    target_pose = {
        "frame": "world",
        "xyz": [0.1, -0.2, 0.3],
        "rotation_matrix": [
            [-0.051366706614, 0.998300348825, 0.027532976902],
            [0.995270593464, 0.04889581027, 0.083938390428],
            [0.082449461935, 0.031714418796, -0.996090569651],
        ],
    }
    parameters = {
        "target_pose": target_pose,
        "orientation_tolerance_rad": 0.3,
    }
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "ik_preview_check",
                    "parameters": parameters,
                },
                "status": "executed",
                "tool_calls": [
                    {
                        "name": "ik_preview_check",
                        "status": "executed",
                        "parameters": parameters,
                        "result": {
                            "success": True,
                            "details": {
                                "outputs": {
                                    "ik_preview_receipt": {
                                        "schema_version": "openeta.ik_preview_receipt.v1",
                                        "receipt_id": "ik-seed-1",
                                        "classification": "feasible",
                                        "orientation_policy": "explicit_orientation",
                                        "tolerances": {
                                            "position_tolerance_m": 0.002,
                                            "orientation_tolerance_rad": 0.3,
                                        },
                                        "best_candidate": {
                                            "joint_positions": [0.1] * 7,
                                        },
                                    }
                                }
                            },
                        },
                    }
                ],
            },
        )
    )

    seed = memory.resolve_ik_execution_seed(parameters)

    assert seed == {
        "schema_version": "openeta.ik_execution_seed.v1",
        "receipt_id": "ik-seed-1",
        "pose_policy_signature": memory.ik_preview_receipts()["latest"][
            "pose_policy_signature"
        ],
        "joint_positions": [0.1] * 7,
        "preview_tolerances": {
            "position_tolerance_m": 0.002,
            "orientation_tolerance_rad": 0.3,
        },
        "object_scene_epoch": memory.object_scene_epoch(),
        "robot_motion_epoch": memory.robot_motion_epoch(),
    }
    json_roundtripped_parameters = {
        "target_pose": {
            "frame": "world",
            "xyz": [0.1000005, -0.2, 0.3],
            "rotation_matrix": [
                [-0.051366706614, 0.998300348825, 0.027532976902],
                [0.995270593464, 0.048895810428, 0.083938390428],
                [0.082449461935, 0.031714418796, -0.996090499351],
            ],
        },
        "orientation_tolerance_rad": 0.3,
        "enable_collision_check": True,
    }
    assert memory.ik_execution_gate_error(
        tool_name="move_to",
        parameters=json_roundtripped_parameters,
    ) is None
    roundtripped_seed = memory.resolve_ik_execution_seed(
        json_roundtripped_parameters
    )
    assert roundtripped_seed is not None
    assert roundtripped_seed["receipt_id"] == "ik-seed-1"

    materially_changed_parameters = {
        **json_roundtripped_parameters,
        "target_pose": {
            **json_roundtripped_parameters["target_pose"],
            "xyz": [0.10002, -0.2, 0.3],
        },
    }
    assert memory.resolve_ik_execution_seed(materially_changed_parameters) is None
    assert "ik_preview_required" in str(
        memory.ik_execution_gate_error(
            tool_name="move_to",
            parameters=materially_changed_parameters,
        )
    )
    assert (
        memory.resolve_ik_execution_seed(
            {
                "target_pose": {
                    "frame": "world",
                    "xyz": [0.1, -0.2, 0.3],
                },
                "preserve_current_orientation": True,
            }
        )
        is None
    )


def test_ik_receipt_matches_semantically_equivalent_matrix_and_quaternion() -> None:
    memory = AgentMemory()
    memory.start_session(task="reach the wrist viewpoint")
    matrix_pose = {
        "frame": "world",
        "xyz": [-0.173573143204, -0.219657273542, 0.21657810518],
        "rotation_matrix": [
            [0.999999879052, 0.000491829688, 0.0],
            [0.000491829688, -0.999999879052, 0.0],
            [0.0, 0.0, -1.0],
        ],
    }
    preview_parameters = {"target_pose": matrix_pose}
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "ik_preview_check",
                    "parameters": preview_parameters,
                },
                "status": "executed",
                "tool_calls": [
                    {
                        "name": "ik_preview_check",
                        "status": "executed",
                        "parameters": preview_parameters,
                        "result": {
                            "success": True,
                            "details": {
                                "outputs": {
                                    "ik_preview_receipt": {
                                        "schema_version": "openeta.ik_preview_receipt.v1",
                                        "receipt_id": "ik-cross-representation-1",
                                        "classification": "feasible",
                                        "orientation_policy": "explicit_orientation",
                                        "best_candidate": {
                                            "joint_positions": [0.1] * 7,
                                        },
                                    }
                                }
                            },
                        },
                    }
                ],
            },
        )
    )
    quaternion_parameters = {
        "target_pose": {
            "frame": "world",
            "xyz": matrix_pose["xyz"],
            "quat_xyzw": [
                0.9999999697629425,
                0.0002459148514356854,
                1.505794178367545e-20,
                6.123233810588188e-17,
            ],
        },
        "enable_collision_check": True,
    }

    assert memory.ik_execution_gate_error(
        tool_name="move_to",
        parameters=quaternion_parameters,
    ) is None
    seed = memory.resolve_ik_execution_seed(quaternion_parameters)
    assert seed is not None
    assert seed["receipt_id"] == "ik-cross-representation-1"

    changed_quaternion = {
        **quaternion_parameters,
        "target_pose": {
            **quaternion_parameters["target_pose"],
            "quat_xyzw": [0.99995, 0.01, 0.0, 0.0],
        },
    }
    assert memory.resolve_ik_execution_seed(changed_quaternion) is None


def test_preserve_current_ik_receipt_resolves_epoch_bound_execution_seed() -> None:
    memory = AgentMemory()
    memory.start_session(task="retreat while preserving the current wrist orientation")
    target_pose = {
        "frame": "world",
        "translation_xyz": [-0.05, -0.34, 0.226],
        "preserve_current_orientation": True,
    }
    preview_parameters = {
        "target_pose": target_pose,
        "preserve_current_orientation": True,
        "position_tolerance_m": 0.005,
        "orientation_tolerance_rad": 0.1,
    }
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "tool_calls": [
                    {
                        "name": "ik_preview_check",
                        "status": "executed",
                        "parameters": preview_parameters,
                        "result": {
                            "success": True,
                            "details": {
                                "outputs": {
                                    "ik_preview_receipt": {
                                        "schema_version": "openeta.ik_preview_receipt.v1",
                                        "receipt_id": "ik-preserve-current-1",
                                        "classification": "feasible",
                                        "orientation_policy": "preserve_current",
                                        "best_candidate": {
                                            "joint_positions": [0.2] * 7,
                                        },
                                    }
                                }
                            },
                        },
                    }
                ]
            },
        )
    )

    move_parameters = {
        **preview_parameters,
        "ik_receipt_id": "ik-preserve-current-1",
        "enable_collision_check": True,
    }
    seed = memory.resolve_ik_execution_seed(move_parameters)

    assert seed is not None
    assert seed["receipt_id"] == "ik-preserve-current-1"
    assert seed["joint_positions"] == [0.2] * 7
    assert seed["object_scene_epoch"] == memory.object_scene_epoch()
    assert seed["robot_motion_epoch"] == memory.robot_motion_epoch()


def test_deferred_collision_ik_requires_verified_motion_collision_owner() -> None:
    memory = AgentMemory()
    memory.start_session(task="reach the target")
    observation = _observation()
    observation.metadata["create_env"] = {
        "control_spec": {
            "schema_version": "openeta.sim_control.v1",
            "controller": {
                "controller_id": "mink.robosuite_joint_velocity",
                "goal_executor": "openeta.worker_mink_goal.v1",
                "collision_callback": True,
                "collision_scope": (
                    "worker_per_step_pre_actuation_and_post_step_configuration"
                ),
            },
        }
    }
    memory.add_observation(observation)
    assert memory.controller_capabilities()[
        "motion_owns_trajectory_world_collision"
    ] is True

    target_pose = {
        "frame": "world",
        "xyz": [0.1, -0.2, 0.3],
        "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    preview_parameters = {
        "target_pose": target_pose,
        "check_endpoint_collision": True,
    }
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "ik_preview_check",
                    "parameters": preview_parameters,
                },
                "status": "executed",
                "tool_calls": [
                    {
                        "name": "ik_preview_check",
                        "status": "executed",
                        "parameters": preview_parameters,
                        "result": {
                            "success": True,
                            "details": {
                                "outputs": {
                                    "ik_preview_receipt": {
                                        "schema_version": "openeta.ik_preview_receipt.v1",
                                        "receipt_id": "ik-deferred-1",
                                        "classification": (
                                            "kinematically_feasible_collision_deferred"
                                        ),
                                        "reason_code": (
                                            "endpoint_collision_check_unavailable"
                                        ),
                                        "best_candidate": {
                                            "joint_positions": [0.2] * 7,
                                        },
                                    }
                                }
                            },
                        },
                    }
                ],
            },
        )
    )

    safe_move = {"target_pose": target_pose, "enable_collision_check": True}
    assert memory.ik_execution_gate_error(
        tool_name="move_to", parameters=safe_move
    ) is None
    seed = memory.resolve_ik_execution_seed(safe_move)
    assert seed is not None
    assert seed["receipt_id"] == "ik-deferred-1"

    for unsafe_move in (
        {"target_pose": target_pose},
        {"target_pose": target_pose, "enable_collision_check": False},
    ):
        error = memory.ik_execution_gate_error(
            tool_name="move_to", parameters=unsafe_move
        )
        assert error is not None
        assert error.startswith("ik_collision_delegation_not_authorized:")
        assert memory.resolve_ik_execution_seed(unsafe_move) is None


def test_deferred_collision_ik_fails_closed_without_controller_capability() -> None:
    memory = AgentMemory()
    memory.start_session(task="reach the target")
    target_pose = {
        "frame": "world",
        "xyz": [0.1, -0.2, 0.3],
        "quat_xyzw": [0.0, 0.0, 0.0, 1.0],
    }
    parameters = {"target_pose": target_pose, "check_endpoint_collision": True}
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "tool_calls": [
                    {
                        "name": "ik_preview_check",
                        "status": "executed",
                        "parameters": parameters,
                        "result": {
                            "success": True,
                            "details": {
                                "outputs": {
                                    "ik_preview_receipt": {
                                        "receipt_id": "ik-deferred-no-owner",
                                        "classification": (
                                            "kinematically_feasible_collision_deferred"
                                        ),
                                        "reason_code": (
                                            "endpoint_collision_check_unavailable"
                                        ),
                                        "best_candidate": {
                                            "joint_positions": [0.2] * 7,
                                        },
                                    }
                                }
                            },
                        },
                    }
                ]
            },
        )
    )

    move = {"target_pose": target_pose, "enable_collision_check": True}
    error = memory.ik_execution_gate_error(tool_name="move_to", parameters=move)
    assert error is not None
    assert "motion_owns_trajectory_world_collision=None" in error
    assert memory.resolve_ik_execution_seed(move) is None


def test_skill_contract_lint_rejects_stale_tool_references() -> None:
    skills = SkillRegistry()
    skills.register(
        SkillSpec(
            name="broken",
            description="broken contract",
            content="Call `ghost_tool` and then call `observe`.",
            allowed_tools=("ghost_tool",),
        )
    )

    issues = lint_skill_contracts(skills, build_default_tool_registry())

    assert {issue["code"] for issue in issues} == {
        "unknown_allowed_tool",
        "unknown_documented_tool",
        "documented_tool_not_allowed",
    }
