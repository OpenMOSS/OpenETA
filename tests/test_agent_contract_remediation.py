from __future__ import annotations

from pathlib import Path

from PIL import Image

from adapter.protocol import CameraFrame, EnvAction, EnvObservation, RobotState
from agent.runtime.actions import PipelineStatus
from agent.runtime.memory import AgentMemory
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


def _record_target_selection(
    memory: AgentMemory,
    *,
    rgb: str = "/session/rgb.png",
    depth: str = "/session/depth.png",
    mask: str = "/session/mask.png",
) -> None:
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "tool_calls": [
                    {
                        "name": "sam3",
                        "status": "executed",
                        "parameters": {"image": rgb, "prompt": "cube"},
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


def test_agent_owned_pipeline_does_not_turn_reference_work_into_a_task_gate() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick alphabet soup")
    memory.save_fact(
        "pending_reference_localization",
        {
            "scene_image": "/session/rgb.png",
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
            "scene_image": "/session/rgb.png",
        },
    )

    plan = ActionPipeline().compile(
        decision,
        observation=_observation(),
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )
    assert plan.status != PipelineStatus.BLOCKED


def test_grasp_bundle_resolves_aligned_inputs_without_model_copying() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    _record_target_selection(memory)
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
    assert captured[0]["rgb"] == "/session/rgb.png"
    assert captured[0]["depth"] == "/session/depth.png"
    assert captured[0]["object_mask"]["mask_ref"] == "/session/mask.png"
    assert captured[0]["camera_frame_id"] == "agentview"
    assert plan.metadata["provenance_bundle_resolution"]["bundle_id"] == bundle_id


def test_grasp_bundle_preserves_host_depth_compatibility_hint(tmp_path: Path) -> None:
    rgb = tmp_path / "rgb.png"
    depth = tmp_path / "depth.png"
    mask = tmp_path / "mask.png"
    Image.new("RGB", (4, 4), (0, 0, 0)).save(rgb)
    Image.new("I;16", (4, 4), 1200).save(depth)
    Image.new("L", (4, 4), 255).save(mask)
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


def test_tool_result_separates_operational_success_from_semantic_outcome() -> None:
    tools = build_default_tool_registry()

    def handler(context: ToolExecutionContext) -> ToolResult:
        return make_tool_result(
            context,
            success=True,
            outputs={"detection_count": 0, "detections": []},
        )

    tools.bind_handler("sam3", handler)
    result = tools.call("sam3", {"image": "/session/rgb.png", "prompt": "cube"})

    assert result.success is True
    assert result.details["operational_success"] is True
    assert result.details["semantic_outcome"] == "no_detection"
    assert result.details["facts_produced"] == ["outputs.detection_count"]
    assert {option["action"] for option in result.details["recovery_options"]} == {
        "refine_grounding",
        "use_reference_or_point_prompt",
    }


def test_decision_state_is_bounded_index_over_packets_bundles_and_last_effect() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick the cube")
    _record_target_selection(memory)
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
    blocked_plan = ActionPipeline().compile(
        PlannerDecision(
            action_type="tool_call",
            action="move_to",
            parameters={
                "target_pose": {"frame": "world", "xyz": [0.1, 0.2, 0.15]}
            },
        ),
        observation=_observation(),
        tools=tools,
        skills=build_default_skill_registry(),
        memory=memory,
    )

    assert blocked_plan.status is PipelineStatus.BLOCKED
    assert blocked_plan.metadata["provenance_integrity_gate"]["blocked"] is True
    assert "grasp:new-target" in blocked_plan.tool_calls[0].reason


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

    blocked = ActionPipeline().compile(
        PlannerDecision(
            action_type="tool_call",
            action="move_to",
            parameters={
                "target_pose": {
                    "frame": "world",
                    "xyz": [0.13, 0.2, 0.15],
                    "compiled_grasp_id": "compiled-adjustment",
                    "waypoint_role": "grasp_contact",
                }
            },
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
