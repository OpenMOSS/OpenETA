from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path

import pytest
from PIL import Image

from adapter.protocol import CameraFrame, EnvAction, EnvObservation, RobotState
from agent.backends.planner import (
    CallablePlannerBackend,
    OpenAICompatiblePlannerBackend,
    OpenAICompatiblePlannerBackendConfig,
    PlannerBackendRequest,
    ProviderHttpError,
    StaticPlannerBackend,
    extract_context_window_tokens,
)
from agent.backends.provider_config import PlannerProviderConfig, read_apikey_file
from agent.runtime.checkers import CHECKER_RESULT_SCHEMA_VERSION, CheckerSubagentConfig
from agent.runtime.episode import DummyEpisodeEnvironment, OpenEtaEpisodeRunner
from agent.runtime.memory import AgentMemory
from agent.runtime.memory_store import JsonMemoryStore
from agent.runtime.pipeline import ActionPipeline
from agent.runtime.planner import (
    PlannerDecision,
    PlannerContextConfig,
    ToolCallingPlanner,
    _agent_owned_tool_planner_system_prompt,
    _validate_official_reward_completion,
    build_tool_context,
)
from agent.runtime.promoted_memory import PromotedMemoryStore
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.runtime.skills import (
    SkillRegistry,
    SkillSpec,
    build_default_skill_registry,
    load_skill_markdown,
)
from agent.runtime.token_counting import (
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    estimate_json_tokens,
    estimate_text_tokens,
)
from agent.tools.handlers import bind_dummy_tool_handlers
from agent.tools.contracts import (
    build_default_tool_contract_catalog,
    project_agent_tool_contract,
)
from agent.tools.registry import (
    TOOL_RESULT_SCHEMA_VERSION,
    ToolExecutionContext,
    ToolRegistry,
    ToolResult,
    ToolSpec,
    build_default_tool_registry,
)


def _observation() -> EnvObservation:
    return EnvObservation(
        task="find the cube",
        cameras=[
            CameraFrame(
                frame_id="front",
                rgb=[[[0, 0, 0]]],
                depth=[[1.0]],
            )
        ],
        robot=RobotState(end_effector_pose={"xyz": [0.0, 0.0, 0.5]}),
        objects=[{"name": "cube"}],
        metadata={
            "step_idx": 1,
            "image_artifacts": [
                {
                    "kind": "rgb",
                    "frame_id": "front",
                    "path": "front-rgb.png",
                    "packet_id": "packet-front",
                },
                {
                    "kind": "depth",
                    "frame_id": "front",
                    "path": "front-depth.png",
                    "packet_id": "packet-front",
                },
            ],
        },
    )


def _record_test_ik_receipt(
    memory: AgentMemory,
    *,
    receipt_id: str,
    target_pose: dict,
) -> None:
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
                                        "classification": "feasible",
                                        "target_pose": dict(target_pose),
                                        "orientation_policy": "preserve_current",
                                        "tolerances": {
                                            "position_tolerance_m": 0.01,
                                            "orientation_tolerance_rad": 0.1,
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


def _observation_with_packet_files(tmp_path: Path) -> EnvObservation:
    """Return the common fixture with production-valid immutable RGB-D files."""

    rgb_path = tmp_path / "front-rgb.png"
    depth_path = tmp_path / "front-depth.png"
    Image.new("RGB", (2, 2), color=(0, 0, 0)).save(rgb_path)
    Image.new("I;16", (2, 2), color=1000).save(depth_path)
    observation = _observation()
    observation.metadata["image_artifacts"][0]["path"] = str(rgb_path)
    observation.metadata["image_artifacts"][1]["path"] = str(depth_path)
    return observation


def _rgbd_observation(
    *,
    task: str,
    views: list[tuple[str, Path, Path]],
    with_extrinsics: bool = False,
) -> EnvObservation:
    intrinsics = {"fx": 100.0, "fy": 100.0, "cx": 0.5, "cy": 0.5, "scale": 1000}
    return EnvObservation(
        task=task,
        cameras=[
            CameraFrame(
                frame_id=frame_id,
                rgb=[[[0, 0, 0]]],
                depth=[[1.0]],
                intrinsics=dict(intrinsics),
                extrinsics=(
                    {
                        "camera_to_world": [
                            [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ]
                    }
                    if with_extrinsics
                    else {}
                ),
            )
            for frame_id, _, _ in views
        ],
        robot=RobotState(),
        metadata={
            "image_artifacts": [
                artifact
                for frame_id, rgb, depth in views
                for artifact in (
                    {
                        "kind": "rgb",
                        "frame_id": frame_id,
                        "path": str(rgb),
                        "packet_id": "packet-rgbd",
                    },
                    {
                        "kind": "depth",
                        "frame_id": frame_id,
                        "path": str(depth),
                        "packet_id": "packet-rgbd",
                    },
                )
            ]
        },
    )


def _tools_with_handlers(*names: str) -> ToolRegistry:
    tools = build_default_tool_registry()
    for name in names:
        if name not in {spec.name for spec in tools.list()}:
            tools.register(
                ToolSpec(
                    name=name,
                    category="test_fixture",
                    description="Test-only non-default tool.",
                    parameters={},
                )
            )
        tools.bind_handler(name, lambda context: ToolResult(True, content="ok"))
    return tools


def _record_pending_sam3_selection(
    memory: AgentMemory,
    *,
    original_image_ref: str = "agentview.png",
    contact_sheet_ref: str = "selection.png",
    segmentation_mode: str = "point_prompt",
    source_observation: dict | None = None,
    evidence_role: str = "target_object",
    result_id: str = "sam3-run-selection",
    prompt: str = "alphabet soup",
) -> None:
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request_name": "sam3",
                "tool_calls": [
                    {
                        "name": "sam3",
                        "status": "executed",
                        "result": {
                            "success": True,
                            "details": {
                                "outputs": {
                                    "result_id": result_id,
                                    "prompt": prompt,
                                    "evidence_role": evidence_role,
                                    "source_image": original_image_ref,
                                    **(
                                        {"source_observation": source_observation}
                                        if source_observation is not None
                                        else {}
                                    ),
                                    "segmentation_mode": segmentation_mode,
                                    "ranking": "score_descending",
                                    "detection_count": 2,
                                    "detections": [
                                        {
                                            "id": "detection_000",
                                            "rank": 0,
                                            "backend_index": 1,
                                            "score": 0.91,
                                            "mask_ref": "tmp/mask_000.png",
                                        },
                                        {
                                            "id": "detection_001",
                                            "rank": 1,
                                            "backend_index": 0,
                                            "score": 0.78,
                                            "mask_ref": "tmp/mask_001.png",
                                        },
                                    ],
                                    "selection_required": True,
                                    "selected_detection": None,
                                    "selection_bundle": {
                                        "original_image_ref": original_image_ref,
                                        "contact_sheet_ref": contact_sheet_ref,
                                        "candidate_count": 2,
                                        "candidates": [],
                                    },
                                },
                                "artifacts": [],
                            },
                        },
                    }
                ],
            },
        )
    )


def _record_grasp_candidates(
    memory: AgentMemory,
    *,
    source_tool: str = "anygrasp",
    camera_frame_id: str | None = None,
) -> None:
    candidate_prefix = "graspgenx" if source_tool == "graspgenx" else "grasp"
    candidates = [
        {
            "id": f"{candidate_prefix}_000",
            "rank": 0,
            "backend_index": 1,
            "frame": "camera",
            "camera_frame": "opencv",
            "score": 0.9,
            "translation_xyz": [0.1, 0.2, 0.3],
            "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "depth": 0.03,
            "width": 0.06,
            "height": 0.03,
            "gripper_tip_position_xyz": [0.13, 0.2, 0.3],
        },
        {
            "id": f"{candidate_prefix}_001",
            "rank": 1,
            "backend_index": 0,
            "frame": "camera",
            "camera_frame": "opencv",
            "score": 0.7,
            "translation_xyz": [0.2, 0.1, 0.3],
            "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "depth": 0.03,
            "width": 0.06,
            "height": 0.03,
            "gripper_tip_position_xyz": [0.23, 0.1, 0.3],
        },
    ]
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {"kind": "tool_call", "name": source_tool, "parameters": {}},
                "status": "executed",
                "tool_calls": [
                    {
                        "name": source_tool,
                        "status": "executed",
                        "result": {
                            "success": True,
                            "details": {
                                "outputs": {
                                    "result_id": f"{source_tool}-run-001",
                                    "ranking": "score_descending",
                                    "candidate_count": 2,
                                    "grasp_candidates": candidates,
                                    "source_rgb": "tmp/rgb.png",
                                    "source_depth": "tmp/depth.png",
                                    "target_mask": "tmp/object-mask.png",
                                    "source": {
                                        "mode": "targeted",
                                        "rgb": "tmp/rgb.png",
                                        "depth": "tmp/depth.png",
                                        "object_mask": "tmp/object-mask.png",
                                        **(
                                            {"camera_frame_id": camera_frame_id}
                                            if camera_frame_id
                                            else {}
                                        ),
                                        "intrinsics": {
                                            "fx": 600.0,
                                            "fy": 600.0,
                                            "cx": 256.0,
                                            "cy": 256.0,
                                            "scale": 1000.0,
                                        },
                                    },
                                }
                            },
                        },
                    }
                ],
            },
        )
    )


def _prepare_agent_owned_anyplace_bundle(memory: AgentMemory) -> tuple[str, dict]:
    _record_grasp_candidates(memory)
    retained_artifact = memory.artifacts["anygrasp_grasp_candidates_latest"]["value"]
    candidate = retained_artifact["best_grasp_candidate"]
    _record_pending_sam3_selection(
        memory,
        original_image_ref="tmp/rgb.png",
        evidence_role="placement_region",
        prompt="basket",
    )
    memory.resolve_sam3_selection(
        result_id="sam3-run-selection",
        detection_id="detection_000",
        selection_source="main_agent_vlm",
    )
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "compile_grasp_seed",
                    "parameters": {"camera_pose": candidate, "scene_epoch": 0},
                },
                "status": "executed",
                "tool_calls": [
                    {
                        "name": "compile_grasp_seed",
                        "status": "executed",
                        "result": {
                            "success": True,
                            "details": {
                                "outputs": {
                                    "schema_version": "openeta.compiled_grasp_seed.v1",
                                    "compiled_grasp_id": "compiled-agent-owned-1",
                                    "candidate_id": candidate["id"],
                                    "scene_epoch": 0,
                                }
                            },
                        },
                    }
                ],
            },
        )
    )
    public = memory.anyplace_input_bundle()
    assert public is not None
    assert public["status"] == "ready"
    return public["bundle_id"], candidate


def _record_overwidth_grasp_policy(
    memory: AgentMemory,
    *,
    backend: str,
    source_rgb: str,
    camera_frame_id: str,
    widths: tuple[float, ...] = (0.09, 0.12),
) -> None:
    candidates = [
        {
            "id": f"{backend}-overwidth-{index}",
            "frame": "camera",
            "camera_frame": "opencv",
            "score": 0.9 - index * 0.1,
            "translation_xyz": [0.1, 0.2, 0.3],
            "rotation_matrix": [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            "depth": 0.03,
            "width": width,
            "height": 0.03,
            "gripper_tip_position_xyz": [0.13, 0.2, 0.3],
        }
        for index, width in enumerate(widths)
    ]
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "tool_calls": [
                    {
                        "name": "grasp_pose_estimate",
                        "status": "executed",
                        "result": {
                            "success": True,
                            "details": {
                                "outputs": {
                                    "result_id": f"{backend}-overwidth-result",
                                    "selected_backend": backend,
                                    "mode": "targeted",
                                    "grasp_candidates": candidates,
                                    "source_rgb": source_rgb,
                                    "camera_frame_id": camera_frame_id,
                                    "source": {
                                        "mode": "targeted",
                                        "rgb": source_rgb,
                                        "camera_frame_id": camera_frame_id,
                                    },
                                }
                            },
                        },
                    }
                ]
            },
        )
    )


def test_static_planner_backend_executes_registered_tool_handler(tmp_path: Path) -> None:
    tools = build_default_tool_registry()

    def sam3_handler(context: ToolExecutionContext) -> ToolResult:
        assert context.name == "sam3"
        assert context.metadata["session_id"] == runtime.memory.session_id
        assert context.observation is not None
        assert context.observation.cameras[0].frame_id == "front"
        return ToolResult(
            True,
            content="segmented cube",
            details={"mask_id": "mask-1", "prompt": context.parameters["prompt"]},
        )

    tools.bind_handler("sam3", sam3_handler)
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {
                "kind": "tool_call",
                "name": "sam3",
                "parameters": {"source_packet_id": "packet-front", "prompt": "cube"},
                "reasoning": "Need segmentation before grasp planning.",
            }
        )
    )
    runtime = OpenEtaAgentRuntime(planner=planner, tools=tools)
    runtime.start_session(task="find the cube")

    action = runtime.act(_observation_with_packet_files(tmp_path))

    command = action.command
    assert action.action_type == "tool_call"
    assert command["status"] == "executed"
    assert command["tool_calls"][0]["name"] == "sam3"
    assert command["tool_calls"][0]["result"]["content"] == "segmented cube"
    assert command["tool_calls"][0]["result"]["details"]["schema_version"] == (
        TOOL_RESULT_SCHEMA_VERSION
    )
    assert command["tool_calls"][0]["result"]["details"]["result_type"] == "perception"
    assert command["tool_calls"][0]["result"]["details"]["outputs"]["mask_id"] == "mask-1"


def test_tool_registry_emits_realtime_start_and_end_events() -> None:
    tools = build_default_tool_registry()
    events = []
    tools.add_listener(events.append)
    tools.bind_handler("observe", lambda context: ToolResult(True, content="objects"))

    result = tools.call("observe", {"reason": "event test"}, observation=_observation())

    assert result.success is True
    assert [event["phase"] for event in events] == ["start", "end"]
    assert [event["name"] for event in events] == ["observe", "observe"]
    assert events[0]["parameters"] == {"reason": "event test"}
    assert events[1]["success"] is True
    assert events[1]["content"] == "objects"


def test_planner_backend_validation_retries_until_valid_payload() -> None:
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            [
                {
                    "kind": "tool_call",
                    "name": "missing_tool",
                    "parameters": {},
                },
                {
                    "kind": "response",
                    "name": "talk",
                    "parameters": {},
                    "reasoning": "Fallback after validation feedback.",
                },
            ]
        ),
        max_validation_retries=1,
    )
    memory = AgentMemory()
    memory.start_session(task="find the cube")

    decision = planner.plan(
        _observation(),
        memory=memory,
        tools=_tools_with_handlers("sam3"),
        skills=build_default_skill_registry(),
    )

    assert decision.action_type == "response"
    assert decision.action == "talk"
    assert [
        item["decision"]["name"] for item in decision.metadata["validation_attempt_history"]
    ] == ["missing_tool", "talk"]
    assert decision.metadata["validation_attempt_history"][0]["validation_errors"]
    assert decision.metadata["validation_attempt_history"][1]["validation_errors"] == []


def test_planner_context_uses_environment_assigned_task_as_active_objective() -> None:
    memory = AgentMemory()
    memory.start_session(task="Create an environment and complete its assigned task.")
    assigned_task = "pick up alphabet soup and place it into basket"
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "status": "executed",
                "request": {
                    "kind": "tool_call",
                    "name": "create_simulator_env",
                    "parameters": {"env_id": "openeta/libero-task0-v0"},
                },
                "tool_calls": [
                    {
                        "name": "create_simulator_env",
                        "status": "executed",
                        "result": {
                            "success": True,
                            "details": {
                                "outputs": {
                                    "assigned_task": assigned_task,
                                    "environment": {
                                        "env_id": "openeta/libero-task0-v0",
                                        "handle": "env-1",
                                        "session_id": "sim-session-1",
                                    },
                                }
                            },
                        },
                    }
                ],
            },
        )
    )

    context = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=_tools_with_handlers("observe"),
        skills=build_default_skill_registry(),
    )

    assert context["task"] == assigned_task
    assert context["active_environment_task"]["task"] == assigned_task
    assert context["memory"]["current_user_request"] == (
        "Create an environment and complete its assigned task."
    )


def test_planner_context_can_keep_session_request_authoritative_for_probe() -> None:
    probe_task = "Move to the requested endpoint, report the receipt, and stop."
    assigned_task = "pick up alphabet soup and place it into basket"
    memory = AgentMemory()
    memory.start_session(
        task=probe_task,
        metadata={"task_authority": "session_user_request"},
    )
    memory.save_fact(
        "active_environment_task",
        {"task": assigned_task},
        source="simulator_observation",
    )

    context = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=_tools_with_handlers("observe"),
        skills=build_default_skill_registry(),
    )

    assert context["task"] == probe_task
    assert context["task_authority"] == "session_user_request"
    assert context["active_environment_task"]["task"] == assigned_task
    assert context["memory"]["current_user_request"] == probe_task


def test_motion_probe_can_disable_official_reward_completion_gate() -> None:
    errors = _validate_official_reward_completion(
        PlannerDecision(
            action_type="response",
            action="task_complete",
            parameters={"message": "motion subgoal reached"},
        ),
        tool_context={
            "memory": {
                "metadata": {
                    "source": "ParallelEpisodeHarness",
                    "require_official_reward": False,
                }
            },
            "latest_environment_receipt": {"reward": 0.0, "info": {}},
        },
    )

    assert errors == []


def test_planner_validation_exhaustion_returns_structured_internal_failure() -> None:
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            [
                {"kind": "tool_call", "name": "missing_tool", "parameters": {}},
                {"kind": "tool_call", "name": "missing_tool", "parameters": {}},
            ]
        ),
        max_validation_retries=1,
    )
    memory = AgentMemory()
    memory.start_session(task="find the cube")

    decision = planner.plan(
        _observation(),
        memory=memory,
        tools=_tools_with_handlers("graspgenx"),
        skills=build_default_skill_registry(),
    )

    assert decision.action_type == "response"
    assert decision.action == "talk"
    assert decision.parameters["code"] == "planner_validation_failed"
    assert decision.parameters["validation_attempts"] == 2
    assert decision.metadata["validation_attempts"] == 2
    assert len(decision.metadata["validation_attempt_history"]) == 2
    assert all(
        item["provider_attempts"] == 1 for item in decision.metadata["validation_attempt_history"]
    )


def test_legacy_top_level_command_kinds_are_rejected() -> None:
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {
                "kind": "skill_call",
                "name": "pick",
                "parameters": {"target": "cube"},
                "reasoning": "Legacy schema should no longer be accepted.",
            }
        ),
        max_validation_retries=0,
    )
    memory = AgentMemory()
    memory.start_session(task="pick cube")

    decision = planner.plan(
        _observation(),
        memory=memory,
        tools=_tools_with_handlers("anyplace"),
        skills=build_default_skill_registry(),
    )

    assert decision.action_type == "response"
    assert decision.action == "talk"
    assert decision.parameters["code"] == "planner_validation_failed"
    assert "Unsupported command kind" in decision.parameters["validation_errors"][0]
    assert decision.metadata["validation_attempts"] == 1
    assert decision.metadata["validation_attempt_history"][0]["decision"]["name"] == "pick"


def test_noop_response_is_not_planner_facing() -> None:
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {
                "kind": "response",
                "name": "noop",
                "parameters": {},
                "reasoning": "No-op is no longer part of the agreed response surface.",
            }
        ),
        max_validation_retries=0,
    )
    memory = AgentMemory()
    memory.start_session(task="wait")

    decision = planner.plan(
        _observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )

    assert decision.action_type == "response"
    assert decision.action == "talk"
    assert decision.parameters["code"] == "planner_validation_failed"
    assert "Unsupported response name" in decision.parameters["validation_errors"][0]


def test_environment_lifecycle_interface_belongs_to_tool_contract() -> None:
    prompt = _agent_owned_tool_planner_system_prompt()
    tools = build_default_tool_registry()
    catalog = build_default_tool_contract_catalog(tools.list())
    create_tool = project_agent_tool_contract(catalog.get("create_simulator_env"))
    close_tool = project_agent_tool_contract(catalog.get("close_simulator_env"))

    assert "create_simulator_env" not in prompt
    assert "exclusive_environment_creation_path" in create_tool["semantic_limits"]
    assert "exclusive_environment_cleanup_path" in close_tool["semantic_limits"]






def test_reference_guided_sam3_accepts_exact_source_packet_id(tmp_path: Path) -> None:
    localized_scene = tmp_path / "wrist-0014.png"
    rematerialized_scene = tmp_path / "wrist-0013.png"
    localized_scene.write_bytes(b"same-static-wrist-scene")
    rematerialized_scene.write_bytes(b"same-static-wrist-scene")
    points = [{"x": 212.0, "y": 308.0, "label": 1}]
    memory = AgentMemory()
    memory.start_session(task="pick alphabet soup")
    memory.save_fact(
        "pending_reference_localization",
        {
            "scene_image": str(localized_scene),
            "source_packet_id": "packet-reference",
            "camera_frame_id": "wrist",
            "target_object": "alphabet_soup",
            "positive_points": points,
            "required_parameter": "positive_points",
        },
        source="retrieve_asset_reference",
    )
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {
                "kind": "tool_call",
                "name": "sam3",
                "parameters": {
                    "source_packet_id": "packet-reference",
                    "camera_frame_id": "wrist",
                    "positive_points": points,
                },
            }
        )
    )

    decision = planner.plan(
        _observation(),
        memory=memory,
        tools=_tools_with_handlers("sam3"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "sam3"
    assert decision.metadata["validation_attempt_history"][0]["validation_errors"] == []


def test_verified_reference_evidence_binds_to_matching_sam3_result() -> None:
    memory = AgentMemory()
    scene = "tmp/scene.png"
    points = [{"x": 130.0, "y": 251.0, "label": 1}]
    memory.add_observation(
        _rgbd_observation(
            task="pick alphabet soup",
            views=[("agentview", Path(scene), Path("tmp/depth.png"))],
        )
    )
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "tool_calls": [
                    {
                        "name": "retrieve_asset_reference",
                        "result": {
                            "success": True,
                            "details": {
                                "outputs": {
                                    "environment": "libero",
                                    "target_object": "alphabet_soup",
                                    "localization_bundle": {
                                        "scene_image_ref": scene,
                                        "reference_image_refs": ["front.png", "side.png"],
                                        "positive_points": points,
                                        "memory_query_key": "libero/alphabet_soup",
                                    },
                                    "localizer": {
                                        "verification": {
                                            "decision": "match",
                                            "confidence": 0.98,
                                            "reason": "blue and orange label matches",
                                            "candidate_crop": "candidate.png",
                                        }
                                    },
                                }
                            },
                        },
                    }
                ]
            },
        )
    )
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "tool_calls": [
                    {
                        "name": "sam3",
                        "result": {
                            "success": True,
                            "details": {
                                "parameters": {
                                    "source_packet_id": "packet-rgbd",
                                    "camera_frame_id": "agentview",
                                    "positive_points": points,
                                },
                                "outputs": {
                                    "result_id": "sam-verified",
                                    "source_packet_id": "packet-rgbd",
                                    "source_image": scene,
                                    "detections": [
                                        {
                                            "id": "detection_000",
                                            "rank": 0,
                                            "score": 0.97,
                                        }
                                    ],
                                },
                            },
                        },
                    }
                ]
            },
        )
    )

    pending = memory.pending_sam3_selection()
    assert pending["result_id"] == "sam-verified"
    assert pending["reference_verification"]["decision"] == "match"
    assert pending["reference_verification"]["candidate_crop"] == "candidate.png"






def test_code_policy_validation_feedback_points_to_simulator_creation_tool() -> None:
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {
                "kind": "tool_call",
                "name": "code_policy",
                "parameters": {"tool": "search_envs", "query": "libero"},
                "reasoning": "Incorrectly trying to use code_policy for MCP orchestration.",
            }
        ),
        max_validation_retries=0,
    )
    memory = AgentMemory()
    memory.start_session(task="create a libero simulator environment")

    decision = planner.plan(
        _observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "talk"
    assert decision.parameters["code"] == "planner_validation_failed"
    validation_error = decision.parameters["validation_errors"][0]
    assert "tool_call::create_simulator_env" in validation_error


def test_grasp_bundle_accepts_valid_backend_preference_and_rejects_unknown() -> None:
    tools = build_default_tool_registry()
    tools.bind_handler("grasp_pose_estimate", lambda _context: ToolResult(True))
    memory = AgentMemory()
    memory.start_session(task="pick the cube")

    accepted = ToolCallingPlanner(
        StaticPlannerBackend(
            {
                "kind": "tool_call",
                "name": "grasp_pose_estimate",
                "parameters": {
                    "bundle_id": "grasp:host-issued",
                    "backend_preference": ["graspgenx", "anygrasp"],
                },
            }
        ),
        max_validation_retries=0,
    ).plan(
        _observation(),
        memory=memory,
        tools=tools,
        skills=build_default_skill_registry(),
    )
    rejected = ToolCallingPlanner(
        StaticPlannerBackend(
            {
                "kind": "tool_call",
                "name": "grasp_pose_estimate",
                "parameters": {
                    "bundle_id": "grasp:host-issued",
                    "backend_preference": ["mystery_estimator"],
                },
            }
        ),
        max_validation_retries=0,
    ).plan(
        _observation(),
        memory=memory,
        tools=tools,
        skills=build_default_skill_registry(),
    )

    assert accepted.action == "grasp_pose_estimate"
    assert accepted.parameters["backend_preference"] == ["graspgenx", "anygrasp"]
    assert rejected.action == "talk"
    assert rejected.parameters["code"] == "planner_validation_failed"
    assert "mystery_estimator" in rejected.parameters["validation_errors"][0]


def test_sam3_point_validation_rejects_molmopoint_fields_then_accepts_xy() -> None:
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            [
                {
                    "kind": "tool_call",
                    "name": "sam3",
                    "parameters": {
                        "mode": "points",
                        "source_packet_id": "packet-front",
                        "points": [
                            {
                                "image_index": 1,
                                "pixel_x": 466.0,
                                "pixel_y": 480.0,
                                "label": 1,
                            }
                        ],
                    },
                },
                {
                    "kind": "tool_call",
                    "name": "sam3",
                    "parameters": {
                        "mode": "points",
                        "source_packet_id": "packet-front",
                        "points": [{"x": 466.0, "y": 480.0, "label": 1}],
                    },
                },
            ]
        ),
        max_validation_retries=1,
    )
    memory = AgentMemory()
    memory.start_session(task="segment the grounded object")

    decision = planner.plan(
        _observation(),
        memory=memory,
        tools=_tools_with_handlers("sam3"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "sam3"
    assert decision.parameters["points"] == [{"x": 466.0, "y": 480.0, "label": 1}]
    assert decision.metadata["validation_attempts"] == 2


def test_planner_prompt_explains_molmopoint_to_sam3_point_mapping() -> None:
    prompt = build_default_skill_registry().get("pick").content

    assert "point-grounding capability" in prompt
    assert "original scene" in prompt
    assert "Do not add category guesses" in prompt


def test_anygrasp_validation_rejects_placeholder_mask_and_incomplete_intrinsics() -> None:
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            [
                {
                    "kind": "tool_call",
                    "name": "anygrasp",
                    "parameters": {
                        "mode": "targeted",
                        "rgb": "front-rgb.png",
                        "depth": "front-depth.png",
                        "target_mask": "latest_sam3_mask",
                        "intrinsics": {"camera_index": 0, "frame_id": "agentview"},
                    },
                    "reasoning": "Incorrectly using placeholder outputs.",
                },
                {
                    "kind": "tool_call",
                    "name": "anygrasp",
                    "parameters": {
                        "mode": "targeted",
                        "rgb": "front-rgb.png",
                        "depth": "front-depth.png",
                        "target_mask": "tmp/image/sam3/run/mask_001.png",
                        "intrinsics": {
                            "fx": 1.0,
                            "fy": 1.0,
                            "cx": 0.5,
                            "cy": 0.5,
                            "scale": 1000.0,
                        },
                    },
                    "reasoning": "Retry with concrete SAM3 mask path and camera intrinsics.",
                },
            ]
        ),
        max_validation_retries=1,
    )
    memory = AgentMemory()
    memory.start_session(task="pick milk")

    decision = planner.plan(
        _observation(),
        memory=memory,
        tools=_tools_with_handlers("anygrasp"),
        skills=build_default_skill_registry(),
    )

    assert decision.action_type == "tool_call"
    assert decision.action == "anygrasp"
    assert decision.parameters["target_mask"] == "tmp/image/sam3/run/mask_001.png"
    assert decision.metadata["validation_attempts"] == 2


def test_anygrasp_validation_feedback_mentions_concrete_mask_path() -> None:
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {
                "kind": "tool_call",
                "name": "anygrasp",
                "parameters": {
                    "mode": "targeted",
                    "rgb": "front-rgb.png",
                    "depth": "front-depth.png",
                    "target_mask": "latest_sam3_mask",
                    "intrinsics": {"camera_index": 0},
                },
            }
        ),
        max_validation_retries=0,
    )
    memory = AgentMemory()
    memory.start_session(task="pick milk")

    decision = planner.plan(
        _observation(),
        memory=memory,
        tools=_tools_with_handlers("anygrasp"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "talk"
    assert decision.parameters["code"] == "planner_validation_failed"
    errors = "\n".join(decision.parameters["validation_errors"])
    assert "details.outputs.selected_detection.mask_ref" in errors
    assert "details.outputs.detections[i].mask_ref" in errors
    assert "detections[0]" not in errors
    assert "fx/fy/cx/cy/scale" in errors


def test_contact_graspnet_is_not_an_agent_tool_or_facade_backend() -> None:
    from agent.tools.registry import GRASP_POSE_BACKENDS

    public_names = {item.name for item in build_default_tool_registry().list()}
    assert "contact_graspnet" not in public_names
    assert "contact_graspnet" not in GRASP_POSE_BACKENDS


def test_graspgenx_validation_requires_complete_targeted_inputs() -> None:
    valid_parameters = {
        "rgb": "tmp/rgb.png",
        "depth": "tmp/depth.png",
        "object_mask": {
            "mask_ref": "tmp/object-mask.png",
            "source_image": "tmp/rgb.png",
            "label": "bottle",
        },
        "intrinsics": {
            "fx": 1.0,
            "fy": 1.0,
            "cx": 0.5,
            "cy": 0.5,
            "scale": 1000.0,
        },
        "gripper_name": "franka_panda",
        "up_direction_camera": [0.0, 0.0, -1.0],
    }
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            [
                {
                    "kind": "tool_call",
                    "name": "graspgenx",
                    "parameters": {
                        "rgb": "latest_rgb",
                        "depth": "latest_depth",
                        "object_mask": "latest_mask",
                        "intrinsics": {},
                        "gripper_name": "<gripper>",
                        "up_direction_camera": [0.0, float("nan"), 0.0],
                    },
                },
                {
                    "kind": "tool_call",
                    "name": "graspgenx",
                    "parameters": valid_parameters,
                },
            ]
        ),
        max_validation_retries=1,
    )
    memory = AgentMemory()
    memory.start_session(task="predict grasps for the configured gripper")

    decision = planner.plan(
        _observation(),
        memory=memory,
        tools=_tools_with_handlers("graspgenx"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "graspgenx"
    assert decision.parameters == valid_parameters
    assert decision.metadata["validation_attempts"] == 2


def test_anyplace_validation_rejects_placeholders_then_accepts_structured_handoff() -> None:
    valid_parameters = {"bundle_id": "anyplace:host-issued-bundle"}
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            [
                {
                    "kind": "tool_call",
                    "name": "anyplace",
                    "parameters": {
                        "rgb": "latest_rgb",
                        "depth": "latest_depth",
                        "object_mask": "latest_mask",
                        "placement_region_mask": {"mask_ref": "mask_ref"},
                        "intrinsics": {},
                        "selected_grasp": {},
                    },
                },
                {"kind": "tool_call", "name": "anyplace", "parameters": valid_parameters},
            ]
        ),
        max_validation_retries=1,
    )
    memory = AgentMemory()
    memory.start_session(task="place object")

    decision = planner.plan(
        _observation(),
        memory=memory,
        tools=_tools_with_handlers("anyplace"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "anyplace"
    assert decision.parameters == valid_parameters
    assert decision.metadata["validation_attempts"] == 2


def test_camera_pose_to_world_validation_rejects_mixed_placement_handoff() -> None:
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            [
                {
                    "kind": "tool_call",
                    "name": "camera_pose_to_world",
                    "parameters": {
                        "placement_result_id": "anyplace-result:abc",
                        "candidate_id": "placement_000",
                        "camera_extrinsics": {"invented": True},
                    },
                },
                {
                    "kind": "tool_call",
                    "name": "camera_pose_to_world",
                    "parameters": {
                        "placement_result_id": "anyplace-result:abc",
                        "candidate_id": "placement_000",
                    },
                },
            ]
        ),
        max_validation_retries=1,
    )
    memory = AgentMemory()
    memory.start_session(task="place object")

    decision = planner.plan(
        _observation(),
        memory=memory,
        tools=_tools_with_handlers("camera_pose_to_world"),
        skills=build_default_skill_registry(),
    )

    assert decision.parameters == {
        "placement_result_id": "anyplace-result:abc",
        "candidate_id": "placement_000",
    }
    assert decision.metadata["validation_attempts"] == 2


def test_gripper_control_validation_rejects_fractional_aperture_command() -> None:
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            [
                {
                    "kind": "tool_call",
                    "name": "gripper_control",
                    "parameters": {"position": 0.67},
                },
                {
                    "kind": "tool_call",
                    "name": "gripper_control",
                    "parameters": {"position": 0},
                },
            ]
        ),
        max_validation_retries=1,
    )
    memory = AgentMemory()
    memory.start_session(task="close the gripper")

    decision = planner.plan(
        _observation(),
        memory=memory,
        tools=_tools_with_handlers("gripper_control"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "gripper_control"
    assert decision.parameters == {"position": 0}
    assert decision.metadata["validation_attempts"] == 2


def test_anyplace_validation_rejects_model_supplied_provenance_packet() -> None:
    intrinsics = {"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "scale": 1000.0}
    parameters = {
        "rgb": "tmp/rgb.png",
        "depth": "tmp/depth.png",
        "object_mask": "tmp/object-mask.png",
        "placement_region_mask": {
            "mask_ref": "tmp/placement-mask.png",
            "source_image": "tmp/rgb.png",
        },
        "intrinsics": intrinsics,
        "selected_grasp": {
            "candidate": {
                "id": "graspgenx_000",
                "frame": "camera",
                "camera_frame": "opencv",
                "score": 0.8,
                "translation_xyz": [0.1, 0.2, 0.3],
                "rotation_matrix": [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ],
                "gripper_tip_position_xyz": [0.2, 0.2, 0.3],
                "depth": 0.1,
                "width": 0.08,
                "height": 0.04,
                "gripper_name": "franka_panda",
            },
            "source": {
                "source_tool": "graspgenx",
                "mode": "targeted",
                "rgb": "tmp/rgb.png",
                "depth": "tmp/depth.png",
                "object_mask": "tmp/object-mask.png",
                "intrinsics": intrinsics,
                "gripper_name": "franka_panda",
                "up_direction_camera": [0.0, 0.0, -1.0],
            },
        },
    }
    planner = ToolCallingPlanner(
        StaticPlannerBackend({"kind": "tool_call", "name": "anyplace", "parameters": parameters})
    )
    memory = AgentMemory()
    memory.start_session(task="validate a normalized predictor packet")

    decision = planner.plan(
        _observation(),
        memory=memory,
        tools=_tools_with_handlers("anyplace"),
        skills=build_default_skill_registry(),
    )

    assert decision.action_type == "response"
    assert decision.action == "talk"
    assert decision.parameters["code"] == "planner_validation_failed"
    assert "bundle_id" in decision.parameters["validation_errors"][0]


def test_tool_handler_exception_is_structured_result() -> None:
    tools: ToolRegistry = build_default_tool_registry()

    def failing_handler(context: ToolExecutionContext) -> ToolResult:
        raise RuntimeError(f"bad prompt: {context.parameters['prompt']}")

    tools.bind_handler("sam3", failing_handler)
    result = tools.call("sam3", {"prompt": "cube"}, observation=_observation())

    assert result.success is False
    assert "Tool handler failed: sam3" in result.content
    assert result.details["schema_version"] == TOOL_RESULT_SCHEMA_VERSION
    assert result.details["diagnostics"][0]["error_type"] == "RuntimeError"


def test_callable_planner_backend_accepts_xml_string_payload() -> None:
    def model_wrapper(request: PlannerBackendRequest) -> str:
        assert "tool_references" in request.tool_context
        return """
        ```xml
        <decision>
          <kind>tool_call</kind><name>get_memory</name>
          <parameters><namespace>all</namespace></parameters>
          <reasoning>Need a reference pose.</reasoning>
        </decision>
        ```
        """

    tools = build_default_tool_registry()
    tools.bind_handler("get_memory", lambda context: {"content": "memory read"})
    planner = ToolCallingPlanner(
        CallablePlannerBackend(model_wrapper, provider="unit", model="xml-string")
    )
    runtime = OpenEtaAgentRuntime(planner=planner, tools=tools)
    runtime.start_session(task="pick cube")

    action = runtime.act(_observation())

    assert action.command["status"] == "executed"
    assert action.command["tool_calls"][0]["name"] == "get_memory"
    assert action.command["tool_calls"][0]["result"]["content"] == "memory read"


def test_skill_call_returns_guidance_without_hidden_tool_expansion() -> None:
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {
                "kind": "tool_call",
                "name": "skill_call",
                "parameters": {"name": "pick", "target": "cube"},
                "reasoning": "Need the pick guidance before selecting tools.",
            }
        )
    )
    runtime = OpenEtaAgentRuntime(planner=planner)
    runtime.start_session(task="pick cube")

    action = runtime.act(_observation())

    command = action.command
    assert action.action_type == "tool_call"
    assert command["request"]["kind"] == "tool_call"
    assert command["request"]["name"] == "skill_call"
    assert command["status"] == "planned"
    assert command["tool_calls"] == []
    assert command["safety_checks"] == []
    assert command["skill_call"]["name"] == "pick"
    assert "macro" in command["skill_call"]["result"]["content"]
    skill_parameters = command["skill_call"]["parameters"]
    available = set(skill_parameters["available_allowed_tools"])
    unavailable = set(skill_parameters["unavailable_allowed_tools"])
    assert available.isdisjoint(unavailable)
    assert available | unavailable == set(skill_parameters["allowed_tools"])
    assert "python_exec" in available
    assert "sam3" in unavailable
    assert "never retry an unbound tool" in skill_parameters["tool_availability_rule"]
    assert command["metadata"]["execution_rule"]["mode"] == "skill_guidance_only"


def test_agent_visible_tools_use_contract_schema_and_keep_host_only_audit() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube")
    tools = _tools_with_handlers(
        "grasp_pose_estimate",
        "camera_pose_to_world",
        "gripper_control",
    )

    context = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=tools,
        skills=build_default_skill_registry(),
    )

    available = {
        item["name"]: item for item in context["agent_context"]["available_tools"]
    }
    gripper = available["gripper_control"]
    assert context["agent_context"]["available_tools_schema_version"] == (
        "openeta.agent_tool_contract.v2"
    )
    assert gripper["parameters"]["required"] == ["position"]
    assert gripper["parameters"]["properties"]["position"]["enum"] == [
        0,
        1,
        False,
        True,
    ]
    assert "position" not in gripper["parameters"]

    audit = context["tool_contract_projection_audit"]
    assert audit["authoritative_projection"] == "tool_contract"
    assert audit["runtime_authority"] == "tool_registry_handler_binding"
    assert audit["tool_count"] == 3
    assert audit["matching_tool_count"] == 1
    assert {item["tool"] for item in audit["mismatches"]} == {
        "camera_pose_to_world",
        "grasp_pose_estimate",
    }
    assert "tool_contract_projection_audit" not in context["agent_context"]


def test_skill_references_are_text_guidance_not_required_tool_macros() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube")
    context = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
        config=PlannerContextConfig(max_skill_content_chars=8000),
    )

    pick = next(skill for skill in context["skill_references"] if skill["name"] == "pick")
    selected_pick = next(
        skill for skill in context["selected_skill_guidance"] if skill["name"] == "pick"
    )

    assert context["schema_version"] == "openeta.planner_context.v1"
    assert "content" not in pick
    assert "content" in selected_pick
    assert "## Grasp estimation and selection" in selected_pick["content"]
    assert "allowed_tools" in pick
    assert "required_tools" not in pick
    assert "safety_checks" not in pick
    assert "move_to" in pick["allowed_tools"]
    assert selected_pick["allowed_tools"] == pick["allowed_tools"]
    assert selected_pick["available_allowed_tools"] == []
    assert set(selected_pick["unavailable_allowed_tools"]) == set(
        selected_pick["allowed_tools"]
    )
    assert pick["available_allowed_tools"] == []
    assert "tool_availability_rule" not in pick
    assert "unbound tool" in context["skill_usage"]["tool_availability_rule"]
    assert {skill["name"] for skill in context["skill_references"]} == {
        skill["name"] for skill in context["selected_skill_guidance"]
    }
    assert context["skill_usage"]["inspection_recommended"][0] == "pick"
    assert context["skill_usage"]["inspection_required"] == []


def test_skill_guidance_distinguishes_declared_from_executable_tools() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube")
    tools = build_default_tool_registry()
    tools.bind_handler("observe", lambda _context: ToolResult(True))
    tools.bind_handler("move_to", lambda _context: ToolResult(True))

    context = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=tools,
        skills=build_default_skill_registry(),
        config=PlannerContextConfig(max_skill_content_chars=8000),
    )

    pick = next(
        skill for skill in context["selected_skill_guidance"] if skill["name"] == "pick"
    )
    assert pick["available_allowed_tools"] == ["observe", "move_to"]
    assert "sam3" in pick["unavailable_allowed_tools"]
    assert "sam3" in pick["allowed_tools"]


def test_planner_context_attaches_primary_current_rgb_artifact() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube")
    observation = _observation()
    observation.metadata["image_artifacts"] = [
        {
            "kind": "rgb",
            "frame_id": "wrist",
            "path": "/exact/session/cameras.1.wrist.rgb.png",
            "width": 512,
            "height": 512,
        },
        {
            "kind": "depth",
            "frame_id": "agentview",
            "packet_id": "packet-current",
            "path": "/exact/session/cameras.0.agentview.depth.png",
        },
        {
            "kind": "rgb",
            "frame_id": "render",
            "path": "/exact/session/render.rgb.png",
        },
        {
            "kind": "rgb",
            "frame_id": "agentview",
            "packet_id": "packet-current",
            "path": "/exact/session/cameras.0.agentview.rgb.png",
            "format": "png",
        },
    ]

    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )

    assert context["vision_image_paths"] == [
        "/exact/session/cameras.0.agentview.rgb.png",
        "/exact/session/cameras.1.wrist.rgb.png",
    ]
    assert context["current_camera_artifacts"][0]["packet_id"] == "packet-current"
    assert [item["evidence_id"] for item in context["vision_evidence"]] == [
        "current_observation:1:agentview",
        "current_observation:1:wrist",
    ]
    assert all(item["freshness"] == "current" for item in context["vision_evidence"])
    assert [item["frame_id"] for item in context["current_camera_artifacts"]] == [
        "agentview",
        "agentview",
        "render",
        "wrist",
    ]
    assert [item["kind"] for item in context["current_camera_artifacts"]] == [
        "rgb",
        "depth",
        "rgb",
        "rgb",
    ]
    assert context["current_camera_artifacts"][1]["path"].endswith("cameras.0.agentview.depth.png")




def test_skill_usage_stops_recommending_inspection_after_skill_call() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube")
    memory.record(
        "action",
        {
            "command": {
                "request": {
                    "kind": "tool_call",
                    "name": "skill_call",
                    "parameters": {"name": "pick"},
                }
            }
        },
    )

    context = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
        config=PlannerContextConfig(max_skill_content_chars=4000),
    )

    assert "pick" in context["skill_usage"]["selected_skills"]
    assert "pick" in context["skill_usage"]["inspected_skills"]
    assert "pick" not in context["skill_usage"]["inspection_recommended"]
    assert "pick" not in context["skill_usage"]["inspection_required"]


def test_truncated_skill_guidance_requires_explicit_inspection() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube")
    observation = _observation()
    observation.task = "pick cube"

    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
        config=PlannerContextConfig(max_selected_skills=1, max_skill_content_chars=48),
    )

    assert context["selected_skill_guidance"][0]["name"] == "pick"
    assert context["selected_skill_guidance"][0]["content_truncated"] is True
    assert context["skill_usage"]["inspection_required"] == ["pick"]


def test_pick_skill_fits_complete_default_context_without_exception() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube")
    observation = _observation()
    observation.task = "pick cube"

    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
        config=PlannerContextConfig(max_selected_skills=1),
    )

    selected = context["selected_skill_guidance"][0]
    assert selected["name"] == "pick"
    assert selected["content_char_count"] <= 8000
    assert selected["content_truncated"] is False
    assert context["skill_usage"]["inspection_required"] == []


def test_open_drawer_task_selects_pull_skill() -> None:
    memory = AgentMemory()
    memory.start_session(task="open the middle drawer of the cabinet")
    observation = _observation()
    observation.task = "open the middle drawer of the cabinet"

    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
        config=PlannerContextConfig(max_selected_skills=2, max_skill_content_chars=4000),
    )

    assert "pull" in context["skill_usage"]["selected_skills"]


def test_planner_redirects_world_mutation_to_required_skill_inspection() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube")
    observation = _observation()
    observation.task = "pick cube"
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {
                "kind": "tool_call",
                "name": "move_to",
                "parameters": {"ik_receipt_id": "ik-skill-redirect"},
            }
        ),
        max_validation_retries=0,
        context_config=PlannerContextConfig(
            max_selected_skills=1,
            max_skill_content_chars=48,
        ),
    )

    decision = planner.plan(
        observation,
        memory=memory,
        tools=_tools_with_handlers("move_to"),
        skills=build_default_skill_registry(),
    )

    assert decision.action_type == "tool_call"
    assert decision.action == "skill_call"
    assert decision.parameters == {"skill": "pick"}
    assert decision.metadata["validation_attempts"] == 1
    assert len(decision.metadata["validation_attempt_history"]) == 1
    assert "must be inspected" in decision.metadata["validation_errors"][0]
    assert decision.metadata["policy_redirect"] == {
        "code": "required_skill_inspection",
        "skill": "pick",
        "blocked_action": {"kind": "tool_call", "name": "move_to"},
    }


def test_current_chinese_pick_task_outranks_stale_simulator_session_task() -> None:
    memory = AgentMemory()
    memory.start_session(task="请帮我创建一个新的libero仿真环境")
    observation = _observation()
    observation.task = "好，请帮我抓起来桌上的 alphabet soup"

    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )

    selected = context["selected_skill_guidance"]
    assert selected[0]["name"] == "pick"
    assert selected[0]["selection_score"] > next(
        skill["selection_score"] for skill in selected if skill["name"] == "sim_mcp"
    )


def test_planner_context_selects_sim_mcp_skill_for_chinese_sim_task() -> None:
    memory = AgentMemory()
    memory.start_session(task="创建一个libero+panda机械臂的仿真环境")
    observation = _observation()
    observation.task = "让libero环境中的机械臂向左移动一点"

    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
        config=PlannerContextConfig(max_skill_content_chars=4000),
    )

    selected = {skill["name"]: skill for skill in context["selected_skill_guidance"]}
    assert "sim_mcp" in selected
    assert "create_simulator_env" in selected["sim_mcp"]["allowed_tools"]
    assert "python_exec" in selected["sim_mcp"]["allowed_tools"]


def test_planner_context_selects_embodiment_explore_only_for_profile_work() -> None:
    memory = AgentMemory()
    memory.start_session(task="calibrate a new robot profile")
    observation = _observation()
    observation.task = "calibrate a new robot profile and discover controller parameters"

    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )

    selected = context["selected_skill_guidance"]
    assert selected[0]["name"] == "embodiment_explore"
    assert selected[0]["content_truncated"] is False
    assert context["skill_usage"]["inspection_required"] == []
    assert "update_skill" in selected[0]["allowed_tools"]

    normal_memory = AgentMemory()
    normal_memory.start_session(task="pick cube")
    normal_context = build_tool_context(
        observation=_observation(),
        memory=normal_memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )
    assert "embodiment_explore" not in {
        skill["name"] for skill in normal_context["selected_skill_guidance"]
    }


def test_calibration_tools_require_explicit_embodiment_explore_scope() -> None:
    attempted = {
        "kind": "tool_call",
        "name": "propose_calibration_profile",
        "parameters": {
            "profile": {},
            "profile_fingerprint": {},
            "rationale": "exercise calibration scope validation",
        },
    }
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            [
                attempted,
                {
                    "kind": "response",
                    "name": "talk",
                    "parameters": {"message": "calibration is out of scope"},
                },
            ]
        ),
        max_validation_retries=1,
    )
    normal_memory = AgentMemory()
    normal_memory.start_session(task="pick cube")

    rejected = planner.plan(
        _observation(),
        memory=normal_memory,
        tools=_tools_with_handlers("propose_calibration_profile"),
        skills=build_default_skill_registry(),
    )

    assert rejected.action == "talk"
    errors = rejected.metadata["validation_attempt_history"][0]["validation_errors"]
    assert any("explicit embodiment_explore session" in error for error in errors)

    explore_memory = AgentMemory()
    explore_memory.start_session(task="calibrate a new robot profile")
    explore_observation = _observation()
    explore_observation.task = "calibrate a new robot profile"
    allowed = ToolCallingPlanner(StaticPlannerBackend(attempted)).plan(
        explore_observation,
        memory=explore_memory,
        tools=_tools_with_handlers("propose_calibration_profile"),
        skills=build_default_skill_registry(),
    )
    assert allowed.action == "propose_calibration_profile"


def test_compact_current_sim_skill_needs_no_forced_inspection() -> None:
    memory = AgentMemory()
    memory.start_session(task="请帮我创建一个libero仿真环境")
    observation = _observation()
    observation.task = "请帮我创建一个libero仿真环境"

    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )

    selected = context["selected_skill_guidance"][0]
    assert selected["name"] == "sim_mcp"
    assert selected["content_truncated"] is False
    assert context["skill_usage"]["inspection_required"] == []


def test_planner_context_only_exposes_tools_with_executable_handlers() -> None:
    tools = build_default_tool_registry()
    tools.bind_handler("observe", lambda context: ToolResult(True, content="observed"))
    memory = AgentMemory()
    memory.start_session(task="inspect scene")

    context = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=tools,
        skills=build_default_skill_registry(),
    )

    visible = {reference["name"] for reference in context["tool_references"]}
    assert visible == {"observe"}
    assert {
        "ik_preview_check",
        "obstacle_avoidance",
        "anydexgrasp",
        "slam",
    }.isdisjoint(visible)

    tools.register(
        ToolSpec(
            name="test_map_query",
            category="test_fixture",
            description="Test-only dynamically registered map query.",
        ),
        lambda context: ToolResult(True, content="map ready"),
    )
    rebound_context = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=tools,
        skills=build_default_skill_registry(),
    )
    assert "test_map_query" in {
        reference["name"] for reference in rebound_context["tool_references"]
    }


def test_planner_rejects_registered_tool_without_handler() -> None:
    planner = ToolCallingPlanner(
        StaticPlannerBackend({"kind": "tool_call", "name": "observe", "parameters": {}}),
        max_validation_retries=0,
    )
    memory = AgentMemory()
    memory.start_session(task="navigate")

    decision = planner.plan(
        _observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "talk"
    assert decision.parameters["code"] == "planner_validation_failed"
    assert decision.parameters["validation_errors"] == [
        "Tool requested by planner is not executable: observe."
    ]


def test_current_sim_creation_task_can_use_complete_projected_skill() -> None:
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {
                "kind": "tool_call",
                "name": "create_simulator_env",
                "parameters": {"env_id": "openeta/libero_libero_10_task0-v0"},
            }
        )
    )
    memory = AgentMemory()
    memory.start_session(task="请帮我创建一个libero仿真环境")
    observation = _observation()
    observation.task = "请帮我创建一个libero仿真环境"

    decision = planner.plan(
        observation,
        memory=memory,
        tools=_tools_with_handlers("create_simulator_env"),
        skills=build_default_skill_registry(),
    )

    assert decision.action_type == "tool_call"
    assert decision.action == "create_simulator_env"
    assert decision.parameters["env_id"] == "openeta/libero_libero_10_task0-v0"
    assert decision.metadata["validation_attempts"] == 1
    assert decision.metadata["validation_attempt_history"][0]["validation_errors"] == []


def test_planner_context_compacts_previous_action_metadata() -> None:
    huge_payload = "x" * 10000
    observation = _observation()
    observation.metadata["previous_action"] = {
        "action_type": "tool_call",
        "request_kind": "tool_call",
        "request_name": "python_exec",
        "status": "executed",
        "tool_calls": [
            {
                "name": "python_exec",
                "status": "executed",
                "result": {
                    "success": True,
                    "content": "python_exec completed",
                    "details": {
                        "outputs": {"result": {"large": huge_payload}},
                        "artifacts": [{"preview": huge_payload}],
                    },
                },
            }
        ],
    }
    memory = AgentMemory()
    memory.start_session(task="inspect previous action")

    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )

    serialized = json.dumps(context, ensure_ascii=False)
    previous_action = context["observation"]["metadata"]["previous_action"]
    assert huge_payload not in serialized
    assert previous_action["request_name"] == "python_exec"
    assert previous_action["tool_calls"][0]["result"]["success"] is True


def test_planner_context_bounds_selected_skill_guidance_content() -> None:
    memory = AgentMemory()
    memory.start_session(task="inspect long skill")
    skills = SkillRegistry()
    skills.register(
        SkillSpec(
            name="inspect",
            description="Inspect a target object.",
            content="0123456789" * 20,
            task_patterns=("inspect <object>",),
            allowed_tools=("observe",),
        )
    )

    context = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=skills,
        config=PlannerContextConfig(max_selected_skills=1, max_skill_content_chars=48),
    )

    selected = context["selected_skill_guidance"][0]
    assert selected["name"] == "inspect"
    assert selected["content_truncated"] is True
    assert selected["content"].endswith("[truncated]")
    assert selected["content_char_count"] == 200


def test_pick_skill_is_loaded_from_markdown_guidance() -> None:
    skills = build_default_skill_registry()
    pick = skills.get("pick")

    assert pick.source == "markdown:skills/pick.md"
    assert "concise English visual phrase" in pick.content
    assert "Scores rank proposals but do not prove identity" in pick.content
    assert "## Grasp estimation and selection" in pick.content
    assert "## Approach and near-field refinement" in pick.content
    assert "transport stability" in pick.content
    assert "wrist-view grasp estimate" in pick.content
    assert "co-motion plus vacancy at the source location" in pick.content
    assert "live tool contracts exclusively define" in pick.content
    assert "source_packet_id" not in pick.content
    assert "bundle_id" not in pick.content
    assert "ik_receipt_id" not in pick.content
    assert "LIBERO" not in pick.content
    assert pick.allowed_tools[:7] == (
        "observe",
        "retrieve_asset_reference",
        "sam3",
        "select_sam3_detection",
        "estimate_depth_prior",
        "enhance_depth",
        "grasp_pose_estimate",
    )
    assert "move_to" in pick.allowed_tools
    assert "follow_eef_trajectory" in pick.allowed_tools


def test_builtin_task_skills_are_loaded_from_markdown_guidance() -> None:
    skills = build_default_skill_registry()

    for name in ("pick", "place", "push", "pull", "stack"):
        skill = skills.get(name)
        assert skill.source == f"markdown:skills/{name}.md"
        assert skill.editable is True
        assert skill.version == "v1"
        assert skill.task_patterns
        assert skill.allowed_tools
        assert "guidance" in skill.content.lower()
        assert "executable" in skill.content
        assert "macro" in skill.content


def test_pick_skill_keeps_domain_guidance_without_tool_interface_copy() -> None:
    prompt = build_default_skill_registry().get("pick").content

    assert "concise English visual phrase" in prompt
    assert "Scores rank proposals but do not prove identity" in prompt
    assert "host-owned task phases" in prompt
    assert "ordinary geometric" in prompt
    assert "live tool contracts exclusively define" in prompt


def test_skill_selection_smoke_includes_relevant_markdown_guidance() -> None:
    memory = AgentMemory()
    memory.start_session(task="place cube into basket")
    observation = EnvObservation(
        task="place cube into basket",
        cameras=[
            CameraFrame(
                frame_id="front",
                rgb=[[[0, 0, 0]]],
                depth=[[1.0]],
            )
        ],
        robot=RobotState(end_effector_pose={"xyz": [0.0, 0.0, 0.5]}),
        objects=[{"name": "cube"}, {"name": "basket"}],
        metadata={"step_idx": 1},
    )

    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )

    selected = context["selected_skill_guidance"]
    place = next(skill for skill in selected if skill["name"] == "place")
    assert place["source"] == "markdown:skills/place.md"
    assert "## Plan placement evidence early" in place["content"]
    assert "Never substitute a grasp pose on the receptacle" in place["content"]
    assert "Stop lateral motion before release" in place["content"]
    assert "move_to" in place["allowed_tools"]
    assert "anyplace" in place["allowed_tools"]
    assert "content" not in next(
        skill for skill in context["skill_references"] if skill["name"] == "place"
    )


def test_memory_extract_skill_is_guidance_for_memory_tools() -> None:
    skills = build_default_skill_registry()
    skill = skills.get("memory_extract")

    assert skill.source == "markdown:skills/memory_extract.md"
    assert skill.allowed_tools == ("get_memory", "save_memory", "compact_memory")
    assert "text guidance only" in skill.content
    assert "Do not write directly to `agent/memory/`" in skill.content


def test_skill_markdown_loader_accepts_frontmatter(tmp_path) -> None:
    path = tmp_path / "demo.md"
    path.write_text(
        """---
name: demo
description: Demo skill.
version: v2
editable: false
task_patterns:
  - demo <object>
allowed_tools:
  - observe
---
# Demo

Call `observe`.
""",
        encoding="utf-8",
    )

    skill = load_skill_markdown(path)

    assert skill.name == "demo"
    assert skill.description == "Demo skill."
    assert skill.version == "v2"
    assert skill.editable is False
    assert skill.task_patterns == ("demo <object>",)
    assert skill.allowed_tools == ("observe",)
    assert "Call `observe`" in skill.content


def test_default_tools_are_atomic_and_do_not_include_pick_place_macros() -> None:
    tools = build_default_tool_registry()
    tool_names = {tool.name for tool in tools.list()}

    assert "pick" not in tool_names
    assert "place" not in tool_names
    assert {"observe", "move_to", "gripper_control"}.issubset(tool_names)


def test_agent_memory_tracks_working_facts_artifacts_skill_notes_and_compaction() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube")

    memory.save_fact("target", {"name": "cube"}, source="unit")
    memory.save_artifact(
        "mask",
        {
            "id": "mask-1",
            "tool": "sam3",
            "path": "/Users/kazusa/Documents/openeta/tmp/tool_result/mask.json",
            "grep_hint": "grep -n '<pattern>' /Users/kazusa/Documents/openeta/tmp/tool_result/mask.json",
            "dashboard_url": "http://sim.example/session/session-1",
            "images": [{"path": "/Users/kazusa/Documents/openeta/tmp/image/rgb/front.png"}],
        },
        source="unit",
    )
    memory.save_skill_note("pick", {"failure": "empty mask"}, source="unit")
    summary = memory.compact(max_events=3)

    context = memory.planning_context()

    assert context["working_memory"]["facts"]["target"]["value"]["name"] == "cube"
    assert context["working_memory"]["artifacts"]["mask"]["id"] == "mask-1"
    assert context["working_memory"]["artifacts"]["mask"]["path"].endswith("mask.json")
    assert context["working_memory"]["artifacts"]["mask"]["dashboard_url"] == (
        "http://sim.example/session/session-1"
    )
    assert context["working_memory"]["artifacts"]["mask"]["image_paths"] == [
        "/Users/kazusa/Documents/openeta/tmp/image/rgb/front.png"
    ]
    assert context["working_memory"]["skill_notes"]["pick"][0]["note"]["failure"] == "empty mask"
    assert (
        "facts=['scene_epoch', 'object_scene_epoch', 'robot_motion_epoch', 'target']"
        in summary
    )
    assert context["working_memory"]["compact_summary"] == summary


def test_agent_memory_keeps_latest_human_answer_for_current_episode() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick milk")
    memory.record("episode_start", {"task": "pick milk"})
    memory.add_external_event(
        {
            "type": "human_answer",
            "question": "Should I pick the cube instead?",
            "answer": "Yes, pick the cube.",
        }
    )

    context = memory.planning_context()

    assert context["latest_human_interaction"]["question"] == ("Should I pick the cube instead?")
    assert context["latest_human_interaction"]["answer"] == "Yes, pick the cube."
    human_event = next(
        event for event in context["recent_events"] if event["type"] == "human_answer"
    )
    assert human_event["payload"]["answer"] == "Yes, pick the cube."

    for index in range(12):
        memory.record("diagnostic", {"index": index})

    assert memory.planning_context()["latest_human_interaction"]["answer"] == (
        "Yes, pick the cube."
    )

    memory.record("episode_start", {"task": "create simulator"})
    assert memory.planning_context()["latest_human_interaction"] is None


def test_agent_memory_captures_tool_result_artifacts() -> None:
    memory = AgentMemory()
    memory.start_session(task="remember image")
    artifact = {
        "type": "image",
        "kind": "rgb",
        "index": "front.rgb",
        "path": "/tmp/openeta/front.png",
    }
    action = EnvAction(
        action_type="tool_call",
        command={
            "request": {"kind": "tool_call", "name": "python_exec"},
            "tool_calls": [
                {
                    "name": "python_exec",
                    "status": "executed",
                    "result": {
                        "success": True,
                        "details": {"artifacts": [artifact]},
                    },
                }
            ],
        },
    )

    memory.add_action(action)

    stored = memory.get_memory(namespace="artifacts")["artifacts"]
    assert len(stored) == 1
    saved = next(iter(stored.values()))
    assert saved["source"] == "tool_result"
    assert saved["value"]["path"] == "/tmp/openeta/front.png"
    assert saved["value"]["tool"] == "python_exec"


def test_agent_memory_derives_observe_camera_packets_for_anygrasp(tmp_path) -> None:
    response_path = tmp_path / "render_env-response.json"
    response_path.write_text(
        json.dumps(
            {
                "cameras": [
                    {
                        "frame_id": "agentview",
                        "width": 512,
                        "height": 512,
                        "rgb_path": "/tmp/openeta/cameras.0.agentview.rgb.png",
                        "depth_path": "/tmp/openeta/cameras.0.agentview.depth.png",
                        "depth_min": 0.55,
                        "depth_max": 2.697,
                        "intrinsics": {
                            "fx": 618.0386719675123,
                            "fy": 618.0386719675123,
                            "cx": 256,
                            "cy": 256,
                        },
                        "extrinsics": {
                            "camera_frame": "opengl",
                            "frame_transform": "camera_to_world",
                            "matrix_layout": "row_major",
                            "pos": [0.6, 0.0, 0.96],
                            "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    memory = AgentMemory()
    memory.start_session(task="pick can")
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {"kind": "tool_call", "name": "observe"},
                "tool_calls": [
                    {
                        "name": "observe",
                        "status": "executed",
                        "result": {
                            "success": True,
                            "details": {
                                "outputs": {
                                    "response": {
                                        "response_path": str(response_path),
                                        "response_omitted": True,
                                    }
                                },
                                "artifacts": [],
                            },
                        },
                    }
                ],
            },
        )
    )

    artifacts = memory.get_memory(namespace="artifacts")["artifacts"]
    packet = artifacts["observe_camera_packet_agentview"]["value"]
    assert packet["frame_id"] == "agentview"
    assert packet["rgb_path"].endswith("agentview.rgb.png")
    assert packet["depth_path"].endswith("agentview.depth.png")
    assert packet["anygrasp_intrinsics"] == {
        "fx": 618.0386719675123,
        "fy": 618.0386719675123,
        "cx": 256,
        "cy": 256,
        "scale": 1000.0,
    }
    assert packet["depth_scale_source"] == "default_png_millimeters"

    context = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )
    summary = context["memory"]["working_memory"]["artifacts"]["observe_camera_packet_agentview"]
    assert summary["anygrasp_intrinsics"]["scale"] == 1000.0
    assert summary["intrinsics"]["fx"] == 618.0386719675123
    assert summary["intrinsics"]["scale"] == 1000.0
    assert summary["extrinsics"]["mat"] == [
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        1.0,
    ]


def test_planner_context_preserves_simulator_observation_and_motion_summaries() -> None:
    observation_summary = {
        "robot": {
            "end_effector_pose": {"xyz": [-0.1, 0.06, 0.6]},
            "gripper_state": {"open": False},
        },
        "object_count": 1,
        "objects": [
            {
                "name": "alphabet_soup_1",
                "category": "alphabet_soup",
                "position": [-0.11, -0.17, 0.475],
            }
        ],
    }
    motion_summary = {
        "collision": {"detected": True, "world_collision": True},
        "end": {"xyz": [-0.08, 0.07, 0.61]},
        "target": {"x": -0.08, "y": 0.07, "z": 0.46},
        "steps_executed": 3,
        "reached_target": False,
    }
    memory = AgentMemory()
    memory.start_session(task="抓起来 alphabet soup")
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request_name": "move_to",
                "tool_calls": [
                    {
                        "name": "move_to",
                        "status": "failed",
                        "result": {
                            "success": False,
                            "details": {
                                "outputs": {
                                    "observation_summary": observation_summary,
                                    "motion_summary": motion_summary,
                                },
                                "state_delta": {
                                    "observation": observation_summary,
                                    "motion": motion_summary,
                                },
                                "diagnostics": [{"code": "simulator_mcp_collision"}],
                            },
                        },
                    }
                ],
            },
        )
    )

    context = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )
    event = next(item for item in context["memory"]["recent_events"] if item["type"] == "action")
    details = event["payload"]["command"]["tool_calls"][0]["result"]["details"]
    assert details["outputs"]["observation_summary"]["objects"][0]["position"] == [
        -0.11,
        -0.17,
        0.475,
    ]
    assert details["outputs"]["motion_summary"]["collision"]["detected"] is True
    assert details["state_delta"]["motion"]["reached_target"] is False


def test_planner_context_recent_events_do_not_embed_prior_full_tool_context() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube")
    memory.record(
        "action",
        {
            "command": {
                "request": {"kind": "tool_call", "name": "move_to", "parameters": {}},
                "metadata": {
                    "planner_metadata": {
                        "tool_context": {"large": "x" * 5000},
                        "backend_details": {"usage": {"prompt_tokens": 123}},
                    }
                },
            }
        },
    )

    context = memory.planning_context()
    rendered = json.dumps(context["recent_events"], ensure_ascii=False)

    assert "x" * 100 not in rendered
    assert "<omitted>" in rendered
    assert len(rendered) < 2000


def test_tool_calling_planner_metadata_keeps_context_summary_not_full_context() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube")
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {
                "kind": "response",
                "name": "talk",
                "parameters": {"message": "ok"},
                "reasoning": "test",
            }
        )
    )

    decision = planner.plan(
        _observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )

    assert "tool_context" not in decision.metadata
    assert decision.metadata["tool_context_summary"]["schema_version"] == (
        "openeta.planner_context_summary.v1"
    )
    assert "context_budget" in decision.metadata["tool_context_summary"]


def test_planner_context_projects_without_mutating_history_when_budget_is_reached() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube")
    memory.save_fact("large_note", {"content": "x" * 1200}, source="unit")

    context = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
        config=PlannerContextConfig(
            context_window_tokens=100,
            auto_compact_trigger_ratio=0.5,
            approx_chars_per_token=4,
        ),
    )

    assert not any(event.event_type == "memory_compacted" for event in memory.events)
    assert context["context_budget"]["schema_version"] == "openeta.context_budget.v2"
    assert context["context_budget"]["auto_compact_triggered"] is True
    projection = context["context_budget"]["projection"]
    assert projection["policy"] == "elastic_total_token_budget"
    assert projection["durable_history_mutated"] is False


def test_planner_context_uses_default_one_million_context_window() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube")
    memory.save_fact("large_note", {"content": "x" * 1200}, source="unit")

    context = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )

    assert not any(event.event_type == "memory_compacted" for event in memory.events)
    assert context["context_budget"]["context_window_tokens"] == DEFAULT_CONTEXT_WINDOW_TOKENS
    assert context["context_budget"]["auto_compact_triggered"] is False
    assert context["context_budget"]["trigger_tokens"] == (
        int(DEFAULT_CONTEXT_WINDOW_TOKENS * 0.9) - 4096
    )


def test_planner_projects_bounded_recent_layers_without_mutating_durable_history() -> None:
    memory = AgentMemory()
    memory.start_session(task="inspect a long manipulation trace")
    for index in range(40):
        memory.add_action(
            EnvAction(
                action_type="tool_call",
                command={
                    "status": "executed",
                    "request": {
                        "kind": "tool_call",
                        "name": "python_exec",
                        "parameters": {"code": f"result = {index}"},
                    },
                    "tool_calls": [
                        {
                            "name": "python_exec",
                            "status": "executed",
                            "result": {
                                "success": True,
                                "content": f"result {index}",
                                "details": {"outputs": {"result": index}},
                            },
                        }
                    ],
                },
            )
        )
        memory.add_observation(_observation())

    context = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
        config=PlannerContextConfig(context_window_tokens=1_000_000),
    )

    assert len(memory.model_conversation_messages()) == 81
    recent = context["agent_context"]["recent_transitions"]
    assert len(recent) == 3
    assert {event["type"] for event in recent} == {"observation"}
    assert len(context["agent_context"]["transition_ledger"]) == 40
    assert context["context_budget"]["projection"]["triggered"] is False

    requests: list[PlannerBackendRequest] = []

    def capture(request: PlannerBackendRequest) -> dict:
        requests.append(request)
        return {
            "kind": "response",
            "name": "talk",
            "parameters": {"message": "history inspected"},
        }

    ToolCallingPlanner(
        CallablePlannerBackend(capture),
        context_config=PlannerContextConfig(context_window_tokens=1_000_000),
    ).plan(
        _observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )

    assert len(requests) == 1
    # Initial user task + one compact history index + four recent action/result
    # pairs. The append-only canonical conversation remains complete in memory.
    assert len(requests[0].conversation_messages) == 10
    assert "compacted transcript summary" in requests[0].conversation_messages[1][
        "content"
    ]
    assert len(requests[0].tool_context["recent_transitions"]) == 3
    assert len(requests[0].tool_context["transition_ledger"]) == 40

    constrained = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
        config=PlannerContextConfig(context_window_tokens=10_000),
    )["context_budget"]["projection"]
    assert constrained["triggered"] is True
    assert constrained["entries_removed"] is True
    assert constrained["fits_target"] is True
    assert len(memory.model_conversation_messages()) == 81
    assert len(memory.recent_events(None)) >= 81


def test_layered_projection_does_not_replay_old_large_tool_results() -> None:
    memory = AgentMemory()
    memory.start_session(task="inspect a long manipulation trace")
    for index in range(40):
        memory.add_action(
            EnvAction(
                action_type="tool_call",
                command={
                    "status": "executed",
                    "request": {
                        "kind": "tool_call",
                        "name": "python_exec",
                        "parameters": {"code": f"result = {index}"},
                    },
                    "tool_calls": [
                        {
                            "name": "python_exec",
                            "status": "executed",
                            "result": {
                                "success": True,
                                "content": "ok",
                                "details": {
                                    "outputs": {
                                        "result": {
                                            "index": index,
                                            "payload": f"marker-{index}-" + "x" * 4_000,
                                        }
                                    }
                                },
                            },
                        }
                    ],
                },
            )
        )

    requests: list[PlannerBackendRequest] = []

    def capture(request: PlannerBackendRequest) -> dict:
        requests.append(request)
        return {
            "kind": "response",
            "name": "talk",
            "parameters": {"message": "history inspected"},
        }

    ToolCallingPlanner(
        CallablePlannerBackend(capture),
        context_config=PlannerContextConfig(context_window_tokens=1_000_000),
    ).plan(
        _observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )

    request = requests[0]
    durable_messages = memory.model_conversation_messages()
    projected_text = json.dumps(
        {
            "conversation": request.conversation_messages,
            "context": request.tool_context,
        },
        ensure_ascii=False,
    )
    assert "marker-0-" not in projected_text
    assert "marker-39-" in projected_text
    assert len(request.conversation_messages) == 10
    assert estimate_json_tokens(
        {
            "conversation": request.conversation_messages,
            "context": request.tool_context,
        }
    ).tokens < estimate_json_tokens({"conversation": durable_messages}).tokens // 2
    assert len(memory.model_conversation_messages()) == 81


def test_planner_context_can_disable_context_window_threshold() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube")
    memory.save_fact("large_note", {"content": "x" * 1200}, source="unit")

    context = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
        config=PlannerContextConfig(context_window_tokens=None),
    )

    assert not any(event.event_type == "memory_compacted" for event in memory.events)
    assert context["context_budget"]["context_window_tokens"] is None
    assert context["context_budget"]["trigger_tokens"] is None


def test_token_estimator_reports_method_metadata() -> None:
    estimate = estimate_text_tokens("hello world", model="unknown-provider-model")

    assert estimate.tokens > 0
    assert estimate.chars == len("hello world")
    assert estimate.estimator["method"] in {
        "tiktoken",
        "json_chars_div_approx_chars_per_token",
    }


def test_json_memory_store_persists_session_trace_and_working_memory(tmp_path) -> None:
    store = JsonMemoryStore(tmp_path / ".openeta_memory")
    memory = AgentMemory(store=store)
    memory.start_session(task="pick cube", metadata={"env": "dummy"})

    memory.save_fact("target", {"name": "cube"}, source="unit")
    memory.save_artifact("mask", {"id": "mask-1"}, source="unit")
    memory.save_skill_note("pick", {"lesson": "retry mask"}, source="unit")
    summary = memory.compact(max_events=2)

    assert memory.session_id is not None
    session_path = store.session_path(memory.session_id)
    lines = [json.loads(line) for line in session_path.read_text(encoding="utf-8").splitlines()]

    assert lines[0]["event_type"] == "session_start"
    assert lines[-1]["event_type"] == "memory_compacted"
    assert lines[-1]["payload"]["summary"] == summary

    working_dir = store.working_dir_for(memory.session_id)
    facts = json.loads((working_dir / "facts.json").read_text(encoding="utf-8"))
    artifacts = json.loads((working_dir / "artifacts.json").read_text(encoding="utf-8"))
    skill_notes = json.loads((working_dir / "skill_notes.json").read_text(encoding="utf-8"))
    compact = json.loads((working_dir / "compact_summary.json").read_text(encoding="utf-8"))

    assert facts["target"]["value"]["name"] == "cube"
    assert artifacts["mask"]["value"]["id"] == "mask-1"
    assert skill_notes["pick"][0]["note"]["lesson"] == "retry mask"
    assert compact["summary"] == summary
    sessions = store.list_sessions()
    assert sessions[0]["session_id"] == memory.session_id
    assert sessions[0]["working_dir"] == str(working_dir)


def test_json_memory_store_migrates_legacy_layout(tmp_path) -> None:
    root = tmp_path / ".openeta_memory"
    legacy_sessions = root / "sessions"
    legacy_sessions.mkdir(parents=True)
    legacy_session_path = legacy_sessions / "legacy-session.jsonl"
    legacy_session_path.write_text(
        json.dumps(
            {
                "event_type": "session_start",
                "timestamp_s": 10.0,
                "payload": {"task": "pick milk"},
            },
            sort_keys=True,
        )
        + "\n"
        + json.dumps(
            {
                "event_type": "tool_result",
                "timestamp_s": 12.0,
                "payload": {"type": "move_to"},
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    legacy_working = root / "working"
    legacy_working.mkdir()
    (legacy_working / "facts.json").write_text(
        json.dumps({"target": {"value": "legacy milk"}}, sort_keys=True),
        encoding="utf-8",
    )

    store = JsonMemoryStore(root)

    migrated_trace = root / "sessions" / "legacy-session" / "trace.jsonl"
    assert migrated_trace.exists()
    assert not legacy_session_path.exists()
    assert not legacy_working.exists()
    archived_working_dirs = list((root / "legacy" / "working").iterdir())
    assert len(archived_working_dirs) == 1
    assert (archived_working_dirs[0] / "facts.json").exists()

    sessions = store.list_sessions()
    assert sessions[0]["session_id"] == "legacy-session"
    assert sessions[0]["task"] == "pick milk"
    assert sessions[0]["event_count"] == 2
    assert sessions[0]["session_path"] == str(migrated_trace)
    assert sessions[0]["working_dir"] == str(root / "sessions" / "legacy-session" / "working")
    assert sessions[0]["metadata"]["migrated_from_layout"].endswith("sessions/legacy-session.jsonl")


def test_json_memory_store_serializes_concurrent_index_updates(tmp_path) -> None:
    root = tmp_path / ".openeta_memory"
    session_ids = [f"session-{index:02d}" for index in range(24)]

    def start_session(session_id: str) -> None:
        JsonMemoryStore(root).start_session(
            session_id=session_id,
            task=f"task for {session_id}",
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(start_session, session_ids))

    store = JsonMemoryStore(root)
    indexed = {entry["session_id"] for entry in store.list_sessions()}
    assert indexed == set(session_ids)
    assert json.loads(store.index_path.read_text(encoding="utf-8"))["sessions"]


def test_promoted_memory_store_appends_reviewed_project_memory(tmp_path) -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube")
    memory.save_fact("target", {"name": "cube"}, source="unit")

    result = PromotedMemoryStore(tmp_path / "agent_memory").promote(
        memory,
        namespace="facts",
        key="target",
        reviewer="unit",
        note="keep target fact",
    )

    text = result.path.read_text(encoding="utf-8")
    assert result.path.name == "project_memory.md"
    assert result.namespace == "facts"
    assert result.key == "target"
    assert "reviewed_by: unit" in text
    assert "note: keep target fact" in text
    assert '"target"' in text
    assert any(event.event_type == "memory_promoted" for event in memory.events)


def test_promoted_memory_store_rejects_targets_outside_agent_memory(tmp_path) -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube")
    memory.save_fact("target", {"name": "cube"}, source="unit")

    with pytest.raises(ValueError, match="must stay under agent/memory"):
        PromotedMemoryStore(tmp_path / "agent_memory").promote(
            memory,
            namespace="facts",
            key="target",
            target="../outside.md",
        )


def test_agent_memory_scopes_working_memory_to_resumed_session(tmp_path) -> None:
    root = tmp_path / ".openeta_memory"
    first = AgentMemory(store=JsonMemoryStore(root))
    first.start_session(task="pick cube")
    first.save_fact("target", {"name": "cube"}, source="unit")
    first_session_id = first.session_id
    assert first_session_id is not None

    second = AgentMemory(store=JsonMemoryStore(root))
    second.start_session(task="place cube")

    assert "target" not in second.facts
    assert second.events[0].event_type == "session_start"
    assert second.task == "place cube"

    resumed = AgentMemory(store=JsonMemoryStore(root))
    resumed.resume_session(first_session_id)

    assert resumed.facts["target"]["value"]["name"] == "cube"
    assert resumed.task == "pick cube"
    assert any(event.event_type == "session_resumed" for event in resumed.events)


def test_provider_config_roundtrips_context_window_tokens_and_retry_policy(
    tmp_path,
) -> None:
    from agent.backends.provider_config import load_planner_provider_config, write_env_file

    env_path = tmp_path / ".env"
    write_env_file(
        PlannerProviderConfig(
            provider="openai-compatible",
            model="demo",
            api_base="https://example.test",
            api_key="sk-test",
            max_attempts=4,
            retry_backoff_s=0.25,
            context_window_tokens=128000,
        ),
        env_path,
    )

    loaded = load_planner_provider_config(
        dotenv_path=env_path,
        apikey_path=tmp_path / "none.md",
    )

    assert loaded.context_window_tokens == 128000
    assert loaded.max_attempts == 4
    assert loaded.retry_backoff_s == 0.25
    assert loaded.redacted()["context_window_tokens"] == 128000


def test_provider_config_defaults_context_window_to_one_million(tmp_path) -> None:
    from agent.backends.provider_config import load_planner_provider_config

    loaded = load_planner_provider_config(
        env={},
        dotenv_path=tmp_path / "missing.env",
        apikey_path=tmp_path / "missing.md",
    )

    assert loaded.context_window_tokens == DEFAULT_CONTEXT_WINDOW_TOKENS


def test_extract_context_window_tokens_from_provider_metadata() -> None:
    assert extract_context_window_tokens({"context_length": "128,000"}) == 128000
    assert extract_context_window_tokens({"metadata": {"context_window": 64000}}) == 64000
    assert extract_context_window_tokens({"id": "model-without-metadata"}) is None


def test_runtime_memory_tools_are_bound_and_visible_to_planner_context() -> None:
    runtime = OpenEtaAgentRuntime(
        planner=ToolCallingPlanner(StaticPlannerBackend({"kind": "response", "name": "talk"}))
    )
    runtime.start_session(task="pick cube")

    result = runtime.tools.call(
        "save_memory",
        {
            "namespace": "artifacts",
            "key": "grasp_candidates",
            "content": {"id": "grasp-1", "tool": "anygrasp"},
        },
        observation=_observation(),
    )
    loaded = runtime.tools.call(
        "get_memory",
        {"namespace": "artifacts", "key": "grasp_candidates"},
        observation=_observation(),
    )

    assert result.success is True
    assert loaded.details["result_type"] == "bookkeeping"
    assert loaded.details["outputs"]["artifacts"]["grasp_candidates"]["value"]["id"] == ("grasp-1")
    context = build_tool_context(
        observation=_observation(),
        memory=runtime.memory,
        tools=runtime.tools,
        skills=runtime.skills,
    )
    assert context["memory"]["working_memory"]["artifacts"]["grasp_candidates"]["id"] == "grasp-1"


def test_planner_context_preserves_recent_sam3_mask_refs() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick milk box")
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request_name": "sam3",
                "tool_calls": [
                    {
                        "name": "sam3",
                        "status": "executed",
                        "result": {
                            "success": True,
                            "content": "SAM3 segmentation completed.",
                            "details": {
                                "schema_version": TOOL_RESULT_SCHEMA_VERSION,
                                "tool": "sam3",
                                "category": "perception",
                                "effect": "read_only",
                                "result_type": "perception",
                                "success": True,
                                "parameters": {
                                    "image": "agentview.png",
                                    "prompt": "milk box",
                                },
                                "outputs": {
                                    "detection_count": 1,
                                    "detections": [
                                        {
                                            "label": "milk box",
                                            "score": 0.66,
                                            "mask_ref": "tmp/image/sam3/run/mask_001.png",
                                        }
                                    ],
                                },
                                "artifacts": [],
                                "state_delta": {},
                                "diagnostics": [],
                            },
                        },
                    }
                ],
            },
        )
    )

    context = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )

    action_event = next(
        event for event in context["memory"]["recent_events"] if event["type"] == "action"
    )
    mask_ref = action_event["payload"]["command"]["tool_calls"][0]["result"]["details"]["outputs"][
        "detections"
    ][0]["mask_ref"]
    assert mask_ref == "tmp/image/sam3/run/mask_001.png"


def test_planner_context_preserves_recent_python_exec_result() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick can")
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request_name": "python_exec",
                "tool_calls": [
                    {
                        "name": "python_exec",
                        "status": "executed",
                        "result": {
                            "success": True,
                            "content": "python_exec completed",
                            "details": {
                                "schema_version": TOOL_RESULT_SCHEMA_VERSION,
                                "tool": "python_exec",
                                "category": "coding",
                                "effect": "world_mutating",
                                "result_type": "world_mutating",
                                "success": True,
                                "outputs": {
                                    "result": {
                                        "rgb": "/tmp/openeta/agentview.rgb.png",
                                        "depth": "/tmp/openeta/agentview.depth.png",
                                        "intrinsics": {
                                            "fx": 618.0,
                                            "fy": 618.0,
                                            "cx": 256,
                                            "cy": 256,
                                            "scale": 1000.0,
                                        },
                                        "mask_paths": [
                                            "tmp/image/sam3/run/mask_000.png",
                                        ],
                                        "candidates": [
                                            {"id": "g0", "width": 0.081},
                                            {"id": "g1", "width": 0.079},
                                        ],
                                    }
                                },
                                "artifacts": [],
                                "state_delta": {},
                                "diagnostics": [],
                            },
                        },
                    }
                ],
            },
        )
    )

    context = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )

    action_event = next(
        event for event in context["memory"]["recent_events"] if event["type"] == "action"
    )
    extracted = action_event["payload"]["command"]["tool_calls"][0]["result"]["details"]["outputs"][
        "result"
    ]
    assert extracted["intrinsics"]["scale"] == 1000.0
    assert extracted["mask_paths"][0] == "tmp/image/sam3/run/mask_000.png"
    assert extracted["candidates"][0]["width"] == 0.081
    assert extracted["candidates"][1]["id"] == "g1"


def test_planner_context_preserves_anygrasp_candidates_for_followup_motion() -> None:
    long_session_root = "tmp/" + ("session-segment/" * 24)
    candidate = {
        "id": "grasp_000",
        "frame": "camera",
        "camera_frame": "opencv",
        "score": 0.92,
        "translation_xyz": [0.1, 0.2, 0.3],
        "rotation_matrix": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "depth": 0.03,
        "width": 0.06,
        "height": 0.03,
        "gripper_tip_position_xyz": [0.1, 0.22, 0.3],
    }
    memory = AgentMemory()
    memory.start_session(task="pick can")
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request_name": "anygrasp",
                "tool_calls": [
                    {
                        "name": "anygrasp",
                        "status": "executed",
                        "result": {
                            "success": True,
                            "content": "AnyGrasp grasp detection completed.",
                            "details": {
                                "schema_version": TOOL_RESULT_SCHEMA_VERSION,
                                "tool": "anygrasp",
                                "result_type": "planning",
                                "outputs": {
                                    "source_rgb": f"{long_session_root}agentview.rgb.png",
                                    "source_depth": f"{long_session_root}agentview.depth.png",
                                    "target_mask": f"{long_session_root}mask_000.png",
                                    "source": {
                                        "mode": "targeted",
                                        "rgb": f"{long_session_root}agentview.rgb.png",
                                        "depth": f"{long_session_root}agentview.depth.png",
                                        "object_mask": f"{long_session_root}mask_000.png",
                                        "intrinsics": {
                                            "fx": 1.0,
                                            "fy": 1.0,
                                            "cx": 0.5,
                                            "cy": 0.5,
                                            "scale": 1000.0,
                                        },
                                    },
                                    "candidate_count": 1,
                                    "best_grasp_candidate": candidate,
                                    "grasp_candidates": [candidate],
                                    "ranking": "score_descending",
                                },
                                "artifacts": [],
                            },
                        },
                    }
                ],
            },
        )
    )

    context = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )

    action_event = next(
        event for event in context["memory"]["recent_events"] if event["type"] == "action"
    )
    details = action_event["payload"]["command"]["tool_calls"][0]["result"]["details"]
    outputs = details["outputs"]
    assert outputs["candidate_count"] == 1
    assert outputs["best_grasp_candidate"]["id"] == "grasp_000"
    assert outputs["ranking"] == "score_descending"
    assert outputs["grasp_candidates"][0]["translation_xyz"] == [0.1, 0.2, 0.3]
    assert outputs["grasp_candidates"][0]["rotation_matrix"][0] == [1.0, 0.0, 0.0]

    grasp_artifact = context["memory"]["working_memory"]["artifacts"][
        "anygrasp_grasp_candidates_latest"
    ]
    assert grasp_artifact["candidate_count"] == 1
    assert grasp_artifact["best_grasp_candidate"]["id"] == "grasp_000"
    assert grasp_artifact["selected_grasp_source"]["mode"] == "targeted"
    assert grasp_artifact["selected_grasp_source"]["intrinsics"]["scale"] == 1000.0
    assert "next_tool_hint" not in grasp_artifact

    # Ranked estimator output is evidence; the host must not silently promote
    # rank 0 into an Agent-selected target.
    assert context["retained_targeted_grasp"] is None


def test_planner_context_preserves_anyplace_candidates_for_post_pick_motion() -> None:
    place_pose = {
        "id": "place_grasp_000",
        "source_grasp_id": "grasp_000",
        "frame": "camera",
        "camera_frame": "opencv",
        "translation_xyz": [0.2, 0.1, 0.4],
        "rotation_matrix": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    }
    memory = AgentMemory()
    memory.start_session(task="pick can and place it in basket")
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request_name": "anyplace",
                "tool_calls": [
                    {
                        "name": "anyplace",
                        "status": "executed",
                        "result": {
                            "success": True,
                            "content": "AnyPlace placement prediction completed.",
                            "details": {
                                "schema_version": TOOL_RESULT_SCHEMA_VERSION,
                                "tool": "anyplace",
                                "result_type": "planning",
                                "outputs": {
                                    "candidate_count": 1,
                                    "selected_grasp_id": "grasp_000",
                                    "placement_candidates": [
                                        {
                                            "id": "placement_000",
                                            "source_grasp_id": "grasp_000",
                                            "place_grasp_pose": place_pose,
                                        }
                                    ],
                                },
                                "artifacts": [],
                            },
                        },
                    }
                ],
            },
        )
    )

    context = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )

    artifact = context["memory"]["working_memory"]["artifacts"][
        "anyplace_placement_candidates_latest"
    ]
    assert artifact["selected_grasp_id"] == "grasp_000"
    assert artifact["placement_candidates"][0]["place_grasp_pose"]["id"] == ("place_grasp_000")
    assert "next_tool_hint" not in artifact












def test_asset_reference_strips_agent_scene_image_path() -> None:
    exact_path = "/tmp/session/hash/agentview.rgb.png"
    observation = _observation()
    observation.metadata["image_artifacts"] = [
        {"kind": "rgb", "frame_id": "agentview", "path": exact_path}
    ]
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            [
                {
                    "kind": "tool_call",
                    "name": "retrieve_asset_reference",
                    "parameters": {
                        "environment": "libero",
                        "target_object": "alphabet soup",
                        "source_packet_id": "obs-0001",
                        "scene_image": "/tmp/session/agentview.rgb.png",
                    },
                },
                {
                    "kind": "tool_call",
                    "name": "retrieve_asset_reference",
                    "parameters": {
                        "environment": "libero",
                        "target_object": "alphabet soup",
                        "source_packet_id": "obs-0001",
                        "scene_image": exact_path,
                    },
                },
            ]
        ),
        max_validation_retries=1,
    )

    decision = planner.plan(
        observation,
        memory=AgentMemory(),
        tools=_tools_with_handlers("retrieve_asset_reference"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "retrieve_asset_reference"
    assert decision.parameters == {
        "environment": "libero",
        "target_object": "alphabet soup",
        "source_packet_id": "obs-0001",
    }
    canonicalizations = decision.metadata["host_parameter_canonicalizations"]
    assert canonicalizations[0]["reason"] == "strip_agent_visual_transport_path"


def test_asset_reference_accepts_byte_identical_scene_rematerialization(
    tmp_path: Path,
) -> None:
    previous_path = tmp_path / "previous" / "wrist.rgb.png"
    current_path = tmp_path / "current" / "wrist.rgb.png"
    previous_path.parent.mkdir()
    current_path.parent.mkdir()
    previous_path.write_bytes(b"same-wrist-scene")
    current_path.write_bytes(b"same-wrist-scene")
    observation = _observation()
    observation.metadata["image_artifacts"] = [
        {"kind": "rgb", "frame_id": "wrist", "path": str(current_path)}
    ]
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {
                "kind": "tool_call",
                "name": "retrieve_asset_reference",
                "parameters": {
                    "environment": "libero",
                    "target_object": "alphabet soup",
                    "source_packet_id": "obs-0001",
                    "scene_image": str(previous_path),
                },
            }
        )
    )

    decision = planner.plan(
        observation,
        memory=AgentMemory(),
        tools=_tools_with_handlers("retrieve_asset_reference"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "retrieve_asset_reference"
    assert "scene_image" not in decision.parameters
    assert decision.parameters["source_packet_id"] == "obs-0001"
    assert decision.metadata["validation_attempt_history"][0]["validation_errors"] == []

























































































































































def test_planner_context_preserves_sam3_multi_detection_selection_signal() -> None:
    memory = AgentMemory()
    memory.start_session(task="抓起来 alphabet soup")
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request_name": "sam3",
                "tool_calls": [
                    {
                        "name": "sam3",
                        "status": "executed",
                        "result": {
                            "success": True,
                            "details": {
                                "outputs": {
                                    "detection_count": 2,
                                    "detections": [
                                        {
                                            "id": "detection_000",
                                            "mask_ref": "tmp/mask_000.png",
                                            "score": 0.8,
                                        },
                                        {
                                            "id": "detection_001",
                                            "mask_ref": "tmp/mask_001.png",
                                            "score": 0.7,
                                        },
                                    ],
                                    "selection_required": True,
                                    "selected_detection": None,
                                },
                                "artifacts": [],
                            },
                        },
                    }
                ],
            },
        )
    )

    context = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )
    event = next(item for item in context["memory"]["recent_events"] if item["type"] == "action")
    outputs = event["payload"]["command"]["tool_calls"][0]["result"]["details"]["outputs"]
    assert outputs["selection_required"] is True
    assert outputs["selected_detection"] is None
    assert outputs["detections"][1]["mask_ref"] == "tmp/mask_001.png"




def test_sam3_semantic_roles_preserve_target_while_selecting_placement() -> None:
    runtime = OpenEtaAgentRuntime(
        tools=bind_dummy_tool_handlers(build_default_tool_registry())
    )
    runtime.start_session(task="pick alphabet soup and place it in the basket")
    _record_pending_sam3_selection(
        runtime.memory,
        result_id="sam3-target",
        evidence_role="target_object",
        prompt="alphabet soup",
    )

    target_action = runtime.pipeline.compile(
        PlannerDecision(
            action_type="tool_call",
            action="select_sam3_detection",
            parameters={
                "sam3_result_id": "sam3-target",
                "detection_id": "detection_001",
                "evidence_role": "target_object",
                "reason": "The crop matches the soup package.",
            },
        ),
        observation=_observation(),
        tools=runtime.tools,
        skills=runtime.skills,
        memory=runtime.memory,
    )
    assert target_action.status.value == "executed"
    target = dict(runtime.memory.selected_sam3_detection() or {})
    assert target["result_id"] == "sam3-target"
    assert target["evidence_role"] == "target_object"

    _record_pending_sam3_selection(
        runtime.memory,
        result_id="sam3-placement",
        evidence_role="placement_region",
        prompt="basket",
    )
    assert runtime.memory.pending_sam3_selection()["evidence_role"] == (
        "placement_region"
    )
    assert runtime.memory.selected_sam3_detection() == target

    placement_action = runtime.pipeline.compile(
        PlannerDecision(
            action_type="tool_call",
            action="select_sam3_detection",
            parameters={
                "sam3_result_id": "sam3-placement",
                "detection_id": "detection_000",
                "evidence_role": "placement_region",
                "reason": "The mask covers the basket interior.",
            },
        ),
        observation=_observation(),
        tools=runtime.tools,
        skills=runtime.skills,
        memory=runtime.memory,
    )

    assert placement_action.status.value == "executed"
    selections = runtime.memory.selected_sam3_detections()
    assert selections["target_object"] == target
    assert selections["placement_region"]["result_id"] == "sam3-placement"
    assert runtime.memory.selected_sam3_detection() == target
    world = runtime.memory.world_evidence_context()
    assert world["selected_target"]["value"] == target
    assert world["placement_region"]["value"]["target_prompt"] == "basket"


def test_select_sam3_detection_rejects_role_mismatch() -> None:
    runtime = OpenEtaAgentRuntime(
        tools=bind_dummy_tool_handlers(build_default_tool_registry())
    )
    runtime.start_session(task="pick alphabet soup and place it in the basket")
    _record_pending_sam3_selection(
        runtime.memory,
        result_id="sam3-placement",
        evidence_role="placement_region",
        prompt="basket",
    )

    action = runtime.pipeline.compile(
        PlannerDecision(
            action_type="tool_call",
            action="select_sam3_detection",
            parameters={
                "sam3_result_id": "sam3-placement",
                "detection_id": "detection_000",
                "evidence_role": "target_object",
                "reason": "This is the basket.",
            },
        ),
        observation=_observation(),
        tools=runtime.tools,
        skills=runtime.skills,
        memory=runtime.memory,
    )

    assert action.status.value == "failed"
    assert runtime.memory.pending_sam3_selection() is not None
    assert runtime.memory.selected_sam3_detection("placement_region") is None


def test_select_sam3_detection_reports_exact_pending_id_for_copy_repair() -> None:
    runtime = OpenEtaAgentRuntime(
        tools=bind_dummy_tool_handlers(build_default_tool_registry())
    )
    runtime.start_session(task="pick the salad dressing")
    _record_pending_sam3_selection(
        runtime.memory,
        result_id="20260820T150027609877Z-5089a74a",
        evidence_role="target_object",
        prompt="salad dressing bottle",
    )

    action = runtime.pipeline.compile(
        PlannerDecision(
            action_type="tool_call",
            action="select_sam3_detection",
            parameters={
                "sam3_result_id": "20260820T150027609877Z-5089a74",
                "detection_id": "detection_000",
                "evidence_role": "target_object",
                "reason": "The green bottle is the requested target.",
            },
        ),
        observation=_observation(),
        tools=runtime.tools,
        skills=runtime.skills,
        memory=runtime.memory,
    )

    assert action.status.value == "failed"
    content = str(action.tool_calls[0].result["content"])
    assert "expected='20260820T150027609877Z-5089a74a'" in content
    assert "received='20260820T150027609877Z-5089a74'" in content
    assert "available detection_ids=['detection_000', 'detection_001']" in content
    assert "without rerunning SAM3" in content


def test_wrist_target_selection_exposes_alignment_consumer_handoff() -> None:
    runtime = OpenEtaAgentRuntime(
        tools=bind_dummy_tool_handlers(build_default_tool_registry())
    )
    runtime.start_session(task="pick the milk")
    _record_pending_sam3_selection(
        runtime.memory,
        result_id="sam3-wrist-target",
        source_observation={
            "packet_id": "obs-0011",
            "frame_id": "wrist",
            "role": "wrist",
        },
        evidence_role="target_object",
        prompt="milk carton",
    )

    action = runtime.pipeline.compile(
        PlannerDecision(
            action_type="tool_call",
            action="select_sam3_detection",
            parameters={
                "sam3_result_id": "sam3-wrist-target",
                "detection_id": "detection_000",
                "evidence_role": "target_object",
                "reason": "The wrist mask covers the same milk carton.",
            },
        ),
        observation=_observation(),
        tools=runtime.tools,
        skills=runtime.skills,
        memory=runtime.memory,
    )

    assert action.status.value == "executed"
    result = action.tool_calls[0].result
    assert result is not None
    handoff = result["details"]["outputs"]["downstream_consumer_handoff"]
    assert handoff["inspect"] == "host_resolved_inputs.wrist_alignment"
    assert handoff["primary_consumer"] == "compute_wrist_alignment"
    assert handoff["source_packet_id"] == "obs-0011"
    assert "materializes its downstream input" in result["content"]


def test_rejected_placement_selection_does_not_invalidate_target() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick alphabet soup and place it in the basket")
    _record_pending_sam3_selection(memory, result_id="sam3-target")
    memory.resolve_sam3_selection(
        result_id="sam3-target",
        detection_id="detection_001",
        selection_source="main_agent_vlm",
    )
    target = dict(memory.selected_sam3_detection() or {})
    _record_pending_sam3_selection(
        memory,
        result_id="sam3-placement",
        evidence_role="placement_region",
        prompt="basket",
    )

    rejected = memory.reject_sam3_detections(
        result_id="sam3-placement",
        reason="No candidate covers the basket interior.",
    )

    assert rejected["evidence_role"] == "placement_region"
    assert memory.selected_sam3_detection() == target
    assert memory.sam3_no_detection() is None
    assert memory.sam3_no_detection("placement_region")["result_id"] == (
        "sam3-placement"
    )


def test_runtime_can_reject_all_pending_sam3_detections() -> None:
    runtime = OpenEtaAgentRuntime(tools=bind_dummy_tool_handlers(build_default_tool_registry()))
    runtime.start_session(task="pick alphabet soup")
    _record_pending_sam3_selection(runtime.memory)

    rejected = runtime.pipeline.compile(
        PlannerDecision(
            action_type="tool_call",
            action="reject_sam3_detections",
            parameters={
                "sam3_result_id": "sam3-run-selection",
                "reason": "All masks cover neighboring objects, not the soup can.",
            },
        ),
        observation=_observation(),
        tools=runtime.tools,
        skills=runtime.skills,
        memory=runtime.memory,
    )

    assert rejected.status.value == "executed"
    assert runtime.memory.pending_sam3_selection() is None
    no_detection = runtime.memory.sam3_no_detection()
    assert no_detection["reason"] == "semantic_candidates_rejected"
    assert no_detection["rejected_detection_ids"] == [
        "detection_000",
        "detection_001",
    ]


def test_direct_grasp_backends_are_not_registered_agent_tools() -> None:
    names = {tool.name for tool in build_default_tool_registry().list()}

    assert "grasp_pose_estimate" in names
    assert {"anygrasp", "graspgenx", "list_graspgenx_grippers"}.isdisjoint(names)


def test_memory_requires_semantic_selection_for_single_sam3_detection() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube")
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "tool_calls": [
                    {
                        "name": "sam3",
                        "result": {
                            "success": True,
                            "details": {
                                "outputs": {
                                    "result_id": "sam3-single",
                                    "detections": [
                                        {
                                            "id": "detection_000",
                                            "score": 0.93,
                                            "mask_ref": "tmp/cube-mask.png",
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

    pending = memory.pending_sam3_selection()
    assert pending["result_id"] == "sam3-single"
    assert pending["candidate_count"] == 1
    assert pending["candidates"][0]["id"] == "detection_000"
    assert memory.selected_sam3_detection() is None






def test_planner_context_preserves_camera_pose_transform_for_move_to() -> None:
    world_pose = {
        "id": "grasp_000",
        "frame": "world",
        "score": 0.92,
        "translation_xyz": [-0.12, -0.13, 0.48],
        "rotation_matrix": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "gripper_tip_position_xyz": [-0.12, -0.11, 0.5],
    }
    memory = AgentMemory()
    memory.start_session(task="pick alphabet soup")
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request_name": "camera_pose_to_world",
                "tool_calls": [
                    {
                        "name": "camera_pose_to_world",
                        "status": "executed",
                        "result": {
                            "success": True,
                            "content": "camera-frame pose transformed to world frame",
                            "details": {
                                "schema_version": TOOL_RESULT_SCHEMA_VERSION,
                                "tool": "camera_pose_to_world",
                                "result_type": "planning",
                                "outputs": {
                                    "frame": "world",
                                    "camera_frame_id": "agentview",
                                    "world_pose": world_pose,
                                    "translation_xyz": world_pose["translation_xyz"],
                                    "rotation_matrix": world_pose["rotation_matrix"],
                                    "gripper_tip_position_xyz": world_pose[
                                        "gripper_tip_position_xyz"
                                    ],
                                },
                                "artifacts": [],
                            },
                        },
                    }
                ],
            },
        )
    )

    context = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )

    action_event = next(
        event for event in context["memory"]["recent_events"] if event["type"] == "action"
    )
    outputs = action_event["payload"]["command"]["tool_calls"][0]["result"]["details"]["outputs"]
    assert outputs["world_pose"]["translation_xyz"] == [-0.12, -0.13, 0.48]
    assert outputs["translation_xyz"] == [-0.12, -0.13, 0.48]

    pose_artifact = context["memory"]["working_memory"]["artifacts"][
        "camera_pose_to_world_world_pose_latest"
    ]
    assert pose_artifact["world_pose"]["translation_xyz"] == [-0.12, -0.13, 0.48]
    assert pose_artifact["camera_frame_id"] == "agentview"
    assert "next_tool_hint" not in pose_artifact


def test_dummy_tool_handlers_return_standard_result_envelopes() -> None:
    tools = bind_dummy_tool_handlers(build_default_tool_registry())
    tools.register(
        ToolSpec(
            name="test_planning",
            category="manipulation",
            description="Test-only planning result fixture.",
            effect="planning",
        ),
        lambda context: ToolResult(
            True,
            content="plan ready",
            details={"grasp_candidates": [{"id": "grasp-1"}]},
        ),
    )
    packet_observation = _rgbd_observation(
        task="find the cube",
        views=[
            (
                "front",
                Path("tests/fixtures/sam3/sam_test.png"),
                Path("tests/fixtures/sam3/sam_test.png"),
            )
        ],
    )

    perception = tools.call(
        "sam3",
        {"source_packet_id": "packet-rgbd", "prompt": "cube"},
        observation=packet_observation,
    )
    planning = tools.call(
        "test_planning",
        {},
        observation=_observation(),
    )
    safety = tools.call(
        "ik_preview_check",
        {"target_pose": {"xyz": [0.4, 0.0, 0.2]}},
        observation=_observation(),
    )
    world = tools.call(
        "move_to",
        {"target_pose": {"xyz": [0.4, 0.0, 0.2]}},
        observation=_observation(),
    )
    memory = OpenEtaAgentRuntime().tools.call(
        "save_memory",
        {"namespace": "facts", "key": "target", "content": {"name": "cube"}},
        observation=_observation(),
    )

    assert perception.details["schema_version"] == TOOL_RESULT_SCHEMA_VERSION
    assert perception.details["result_type"] == "perception"
    assert perception.details["outputs"]["masks"][0]["mask_id"] == "mask-cube-001"
    assert planning.details["result_type"] == "planning"
    assert planning.details["outputs"]["grasp_candidates"][0]["id"] == "grasp-1"
    assert safety.details["result_type"] == "safety"
    assert safety.details["outputs"]["feasible"] is True
    assert world.details["result_type"] == "world_mutating"
    assert world.details["requires_observation_after_call"] is True
    assert world.details["state_delta"]["eef_pose"]["xyz"] == [0.4, 0.0, 0.2]
    assert memory.details["result_type"] == "bookkeeping"
    assert memory.details["outputs"]["namespace"] == "facts"


def test_registry_promotes_legacy_tool_artifacts_into_standard_envelope() -> None:
    tools = build_default_tool_registry()
    tools.bind_handler(
        "sam3",
        lambda context: ToolResult(
            True,
            content="mask generated",
            details={
                "detections": [{"mask_ref": "cube-mask.png"}],
                "artifacts": [
                    {
                        "type": "segmentation_mask",
                        "kind": "mask",
                        "tool": "sam3",
                        "path": "cube-mask.png",
                    }
                ],
            },
        ),
    )

    result = tools.call(
        "sam3",
        {"source_packet_id": "packet-front", "prompt": "cube"},
        observation=_observation(),
    )

    assert result.details["schema_version"] == TOOL_RESULT_SCHEMA_VERSION
    assert result.details["outputs"]["detections"][0]["mask_ref"] == "cube-mask.png"
    assert result.details["artifacts"][0]["path"] == "cube-mask.png"


def test_registry_flattens_legacy_explicit_outputs_instead_of_double_nesting() -> None:
    tools = build_default_tool_registry()
    tools.bind_handler(
        "estimate_depth_prior",
        lambda _context: ToolResult(
            True,
            content="prior ready",
            details={
                "tool": "estimate_depth_prior",
                "backend": "depth_prior_mcp",
                "outputs": {
                    "source_packet_id": "obs-0000",
                    "prior_depth": "prior.npy",
                },
                "artifacts": [{"type": "depth_prior", "path": "prior.npy"}],
            },
        ),
    )

    result = tools.call(
        "estimate_depth_prior",
        {"source_packet_id": "obs-0000", "camera_frame_id": "agentview"},
    )

    assert result.details["outputs"]["source_packet_id"] == "obs-0000"
    assert result.details["outputs"]["prior_depth"] == "prior.npy"
    assert result.details["outputs"]["backend"] == "depth_prior_mcp"
    assert "outputs" not in result.details["outputs"]
    assert result.details["artifacts"][0]["path"] == "prior.npy"


def test_registry_preserves_legacy_domain_schema_version_as_output() -> None:
    tools = build_default_tool_registry()
    tools.bind_handler(
        "compile_grasp_seed",
        lambda _context: ToolResult(
            True,
            details={
                "schema_version": "openeta.compiled_grasp_seed.v1",
                "compiled_grasp_id": "compiled-1",
                "candidate_id": "candidate-1",
            },
        ),
    )

    result = tools.call("compile_grasp_seed", {})

    assert result.details["schema_version"] == TOOL_RESULT_SCHEMA_VERSION
    assert result.details["outputs"]["schema_version"] == (
        "openeta.compiled_grasp_seed.v1"
    )
    assert result.details["outputs"]["compiled_grasp_id"] == "compiled-1"


def test_registry_promotes_legacy_diagnostics_and_recovery_contract() -> None:
    tools = build_default_tool_registry()
    tools.bind_handler(
        "grasp_pose_estimate",
        lambda context: ToolResult(
            False,
            content="all candidates collide",
            details={
                "reason": "all_grasps_colliding",
                "diagnostics": [
                    {"code": "grasp_pose_estimate_failed", "retryable": False}
                ],
                "recovery_options": [
                    {
                        "action": "acquire_materially_different_view_or_backend",
                        "reason": "the unchanged request cannot produce a new result",
                    }
                ],
            },
        ),
    )

    result = tools.call("grasp_pose_estimate", {"bundle_id": "grasp:test"})

    assert result.details["diagnostics"][0]["code"] == "grasp_pose_estimate_failed"
    assert result.details["recovery_options"][0]["action"] == (
        "acquire_materially_different_view_or_backend"
    )
    assert result.details["outputs"]["reason"] == "all_grasps_colliding"


def test_registry_projects_actionable_recovery_from_standard_failure() -> None:
    tools = build_default_tool_registry()
    tools.bind_handler(
        "sam3",
        lambda context: ToolResult(
            False,
            content="missing image",
            details={
                "schema_version": TOOL_RESULT_SCHEMA_VERSION,
                "diagnostics": [{"code": "missing_image"}],
                "outputs": {"reason": "missing_image"},
            },
        ),
    )

    result = tools.call("sam3", {"source_packet_id": "obs-0001", "prompt": "cube"})

    assert result.details["recovery_options"] == [
        {
            "action": "correct_parameters_from_diagnostic_and_tool_schema",
            "reason": (
                "Repair sam3 inputs using the reported code and current host-resolved "
                "values; do not invent missing geometry or provenance."
            ),
        }
    ]


def test_registry_synthesizes_diagnostic_for_opaque_handler_failure() -> None:
    tools = build_default_tool_registry()
    tools.bind_handler(
        "sam3",
        lambda context: ToolResult(
            False,
            content="backend returned no usable response",
            details={"reason": "invalid_mcp_response"},
        ),
    )

    result = tools.call("sam3", {"source_packet_id": "obs-0001", "prompt": "cube"})

    assert result.details["diagnostics"] == [
        {
            "code": "invalid_mcp_response",
            "message": "backend returned no usable response",
            "reason": "invalid_mcp_response",
        }
    ]
    assert result.details["recovery_options"][0]["action"] == (
        "stop_repeating_and_report_backend_contract_mismatch"
    )


def test_pipeline_allows_planner_requested_safe_check_tool_call() -> None:
    tools = bind_dummy_tool_handlers(build_default_tool_registry())
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {
                "kind": "tool_call",
                "name": "safe_check",
                "parameters": {
                    "tool": "ik_preview_check",
                    "target_pose": {"xyz": [0.4, 0.0, 0.2]},
                },
            }
        )
    )
    runtime = OpenEtaAgentRuntime(planner=planner, tools=tools)
    runtime.start_session(task="preview whether a pose is safe")

    action = runtime.act(_observation())

    command = action.command
    assert command["request"]["name"] == "safe_check"
    assert command["status"] == "executed"
    assert command["safety_checks"][0]["name"] == "ik_preview_check"
    assert command["safety_checks"][0]["reason"] == "Planner-requested safety check."
    assert command["safety_checks"][0]["result"]["details"]["result_type"] == "safety"
    assert command["safety_checks"][0]["result"]["details"]["outputs"]["feasible"] is True
    assert command["tool_calls"] == []


def test_pipeline_runs_pre_safety_checker_before_configured_tool_call() -> None:
    tools = bind_dummy_tool_handlers(build_default_tool_registry())
    target_pose = {"frame": "world", "xyz": [0.4, 0.0, 0.2]}
    pipeline = ActionPipeline(
        checker_subagents=CheckerSubagentConfig(pre_safety_checks={"move_to": "ik_preview_check"})
    )
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {
                "kind": "tool_call",
                "name": "move_to",
                "parameters": {"ik_receipt_id": "ik-pre-safety-pass"},
            }
        )
    )
    runtime = OpenEtaAgentRuntime(planner=planner, tools=tools, pipeline=pipeline)
    runtime.start_session(task="move to pose")
    _record_test_ik_receipt(
        runtime.memory,
        receipt_id="ik-pre-safety-pass",
        target_pose=target_pose,
    )

    action = runtime.act(_observation())

    command = action.command
    assert command["status"] == "executed"
    assert command["safety_checks"][0]["name"] == "ik_preview_check"
    assert command["safety_checks"][0]["result"]["details"]["outputs"]["feasible"] is True
    assert command["tool_calls"][0]["name"] == "move_to"
    assert command["tool_calls"][0]["status"] == "executed"
    assert command["metadata"]["checker_results"]["pre_safety_checks"][0]["name"] == (
        "ik_preview_check"
    )


def test_pipeline_blocks_tool_call_when_pre_safety_checker_fails() -> None:
    tools = bind_dummy_tool_handlers(build_default_tool_registry())
    target_pose = {"frame": "world", "xyz": [9.0, 0.0, 0.2]}

    def unsafe_ik(context: ToolExecutionContext) -> ToolResult:
        return ToolResult(
            False,
            content="IK target is infeasible",
            details={"feasible": False, "reason": "outside_workspace"},
        )

    tools.bind_handler("ik_preview_check", unsafe_ik, replace=True)
    pipeline = ActionPipeline(
        checker_subagents=CheckerSubagentConfig(pre_safety_checks={"move_to": "ik_preview_check"})
    )
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {
                "kind": "tool_call",
                "name": "move_to",
                "parameters": {"ik_receipt_id": "ik-pre-safety-fail"},
            }
        )
    )
    runtime = OpenEtaAgentRuntime(planner=planner, tools=tools, pipeline=pipeline)
    runtime.start_session(task="unsafe move")
    _record_test_ik_receipt(
        runtime.memory,
        receipt_id="ik-pre-safety-fail",
        target_pose=target_pose,
    )

    action = runtime.act(_observation())

    command = action.command
    assert command["status"] == "blocked"
    assert command["safety_checks"][0]["status"] == "failed"
    assert command["safety_checks"][0]["result"]["details"]["outputs"]["feasible"] is False
    assert command["tool_calls"][0]["name"] == "move_to"
    assert command["tool_calls"][0]["status"] == "skipped"
    assert "IK target is infeasible" in command["tool_calls"][0]["reason"]
    repair = command["metadata"]["repair_bundle"]
    assert repair["schema_version"] == "openeta.gate_repair.v1"
    assert repair["code"] == "ik_preview_not_feasible"
    assert repair["checker_evidence"][0]["name"] == "ik_preview_check"
    assert any(call["tool"] == "observe" for call in repair["allowed_next_calls"])
    shadow = repair["contract_shadow_validation"]
    assert shadow["evaluated"] is True
    assert shadow["enforcing"] is False
    assert shadow["authoritative_gate"] == "legacy_runtime"
    assert shadow["conformant"] is True
    assert {
        item["check_id"] for item in shadow["matched_gate_bindings"]
    } == {
        "runtime.ik_execution_authorization",
        "runtime.pre_safety_checker",
    }


def test_pipeline_runs_post_failure_checker_after_configured_tool_call(
    tmp_path: Path,
) -> None:
    tools = build_default_tool_registry()
    tools.bind_handler(
        "sam3",
        lambda context: ToolResult(
            False,
            content="mask generation failed",
            details={"reason": "empty_mask"},
        ),
    )
    pipeline = ActionPipeline(
        checker_subagents=CheckerSubagentConfig(post_failure_checks=("sam3",))
    )
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {
                "kind": "tool_call",
                "name": "sam3",
                "parameters": {"source_packet_id": "packet-front", "prompt": "cube"},
            }
        )
    )
    runtime = OpenEtaAgentRuntime(planner=planner, tools=tools, pipeline=pipeline)
    runtime.start_session(task="segment cube")

    action = runtime.act(_observation_with_packet_files(tmp_path))

    command = action.command
    post_checks = command["metadata"]["checker_results"]["post_failure_checks"]
    assert command["status"] == "failed"
    assert post_checks[0]["name"] == "failure_check"
    assert post_checks[0]["result"]["details"]["schema_version"] == (CHECKER_RESULT_SCHEMA_VERSION)
    assert post_checks[0]["result"]["details"]["target_tool"] == "sam3"
    assert post_checks[0]["result"]["details"]["verdict"] == "failed"
    recovery_events = [
        event for event in runtime.memory.events if event.event_type == "recovery_feedback"
    ]
    assert len(recovery_events) == 1
    recovery_context = runtime.memory.planning_context()["recent_events"]
    recovery_summary = next(
        event for event in recovery_context if event["type"] == "recovery_feedback"
    )
    assert recovery_summary["payload"]["command"]["status"] == "failed"
    assert recovery_summary["payload"]["command"]["request"]["name"] == "sam3"


def test_pipeline_does_not_run_post_failure_checker_after_success(
    tmp_path: Path,
) -> None:
    tools = build_default_tool_registry()
    tools.bind_handler("sam3", lambda context: ToolResult(True, content="mask generated"))
    pipeline = ActionPipeline(
        checker_subagents=CheckerSubagentConfig(post_failure_checks=("sam3",))
    )
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {
                "kind": "tool_call",
                "name": "sam3",
                "parameters": {"source_packet_id": "packet-front", "prompt": "cube"},
            }
        )
    )
    runtime = OpenEtaAgentRuntime(planner=planner, tools=tools, pipeline=pipeline)
    runtime.start_session(task="segment cube")

    action = runtime.act(_observation_with_packet_files(tmp_path))

    assert action.command["status"] == "executed"
    assert action.command["metadata"]["checker_results"]["post_failure_checks"] == []
    assert not any(event.event_type == "recovery_feedback" for event in runtime.memory.events)


def test_episode_runner_executes_three_closed_loop_tool_turns() -> None:
    tools = build_default_tool_registry()
    tools.bind_handler(
        "observe",
        lambda context: ToolResult(
            True,
            content="objects detected",
            details={"objects": ["cube"], "step_idx": context.observation.metadata["step_idx"]},
        ),
    )
    tools.bind_handler(
        "sam3",
        lambda context: ToolResult(
            True,
            content="mask generated",
            details={"mask_id": "mask-cube", "prompt": context.parameters["prompt"]},
        ),
    )
    tools.bind_handler(
        "get_memory",
        lambda context: ToolResult(
            True,
            content="grasp candidates generated",
            details={
                "grasp_candidates": [{"id": "grasp-1"}],
                "namespace": context.parameters["namespace"],
            },
        ),
    )
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            [
                {
                    "kind": "tool_call",
                    "name": "observe",
                    "parameters": {"reason": "inspect scene"},
                    "reasoning": "List objects before segmentation.",
                },
                {
                    "kind": "tool_call",
                    "name": "sam3",
                    "parameters": {
                        "source_packet_id": "packet-front",
                        "prompt": "cube",
                    },
                    "reasoning": "Segment the target object.",
                },
                {
                    "kind": "tool_call",
                    "name": "get_memory",
                    "parameters": {"namespace": "all"},
                    "reasoning": "Inspect accumulated working memory.",
                },
            ]
        )
    )
    runtime = OpenEtaAgentRuntime(planner=planner, tools=tools)
    runner = OpenEtaEpisodeRunner(
        runtime=runtime,
        environment=DummyEpisodeEnvironment(),
    )

    result = runner.run(task="pick cube", max_turns=3, metadata={"source": "unit"})

    assert len(result.steps) == 3
    assert [step.turn_index for step in result.steps] == [1, 2, 3]
    assert [step.action.command["request"]["name"] for step in result.steps] == [
        "observe",
        "sam3",
        "get_memory",
    ]
    assert [step.observation.metadata["step_idx"] for step in result.steps] == [0, 1, 2]
    assert result.steps[0].action.command["tool_calls"][0]["result"]["content"] == (
        "objects detected"
    )
    assert result.steps[2].step_result.info["previous_action"]["request_name"] == "get_memory"
    assert runtime.memory.session_id == result.session_id
    event_types = [event.event_type for event in runtime.memory.events]
    assert event_types.count("episode_step") == 3
    assert "episode_start" in event_types
    assert "episode_result" in event_types


def test_episode_runner_stops_when_agent_reports_task_complete() -> None:
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            [
                {
                    "kind": "tool_call",
                    "name": "observe",
                    "parameters": {"reason": "locate cube"},
                },
                {
                    "kind": "response",
                    "name": "task_complete",
                    "parameters": {"success": True, "summary": "cube located"},
                    "reasoning": "The objective is satisfied.",
                },
                {
                    "kind": "tool_call",
                    "name": "sam3",
                    "parameters": {
                        "source_packet_id": "packet-front",
                        "prompt": "cube",
                    },
                },
            ]
        )
    )
    tools = build_default_tool_registry()
    tools.bind_handler(
        "observe",
        lambda context: ToolResult(True, content="objects detected"),
    )
    runtime = OpenEtaAgentRuntime(planner=planner, tools=tools)
    runner = OpenEtaEpisodeRunner(runtime=runtime, environment=DummyEpisodeEnvironment())

    result = runner.run(task="find cube", max_turns=10)

    assert len(result.steps) == 2
    assert result.terminated is True
    assert result.truncated is False
    assert result.metadata["stop_reason"] == "task_complete"
    assert result.steps[-1].action.command["request"]["kind"] == "response"
    assert result.steps[-1].action.command["request"]["name"] == "task_complete"
    assert result.steps[-1].step_result.info["termination_source"] == "agent"


def test_episode_runner_stops_when_agent_talks_to_user() -> None:
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            [
                {
                    "kind": "response",
                    "name": "talk",
                    "parameters": {"message": "No image path is available."},
                    "reasoning": "Report the result to the user.",
                },
                {
                    "kind": "response",
                    "name": "talk",
                    "parameters": {"message": "This should not repeat."},
                },
            ]
        )
    )
    runtime = OpenEtaAgentRuntime(planner=planner, tools=build_default_tool_registry())
    runner = OpenEtaEpisodeRunner(runtime=runtime, environment=DummyEpisodeEnvironment())

    result = runner.run(task="find image path", max_turns=10)

    assert len(result.steps) == 1
    assert result.terminated is True
    assert result.metadata["stop_reason"] == "status_report"
    assert result.steps[0].action.command["status"] == "executed"
    assert result.steps[0].step_result.info["termination_reason"] == "status_report"
    assert result.steps[0].step_result.info["response_name"] == "talk"


def test_episode_runner_pauses_when_agent_asks_human() -> None:
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            [
                {
                    "kind": "response",
                    "name": "ask_human",
                    "parameters": {"question": "Which LIBERO task should I create?"},
                    "reasoning": "Need operator choice.",
                },
                {
                    "kind": "tool_call",
                    "name": "observe",
                    "parameters": {"reason": "continue after answer"},
                },
            ]
        )
    )
    tools = build_default_tool_registry()
    tools.bind_handler(
        "observe",
        lambda context: ToolResult(True, content="objects detected"),
    )
    runtime = OpenEtaAgentRuntime(planner=planner, tools=tools)
    runner = OpenEtaEpisodeRunner(runtime=runtime, environment=DummyEpisodeEnvironment())

    result = runner.run(task="create libero env", max_turns=10)

    assert len(result.steps) == 1
    assert result.terminated is False
    assert result.truncated is False
    assert result.metadata["stop_reason"] == "ask_human"
    assert result.metadata["waiting_for_human"] is True
    assert result.steps[0].action.command["request"]["name"] == "ask_human"
    assert result.steps[0].step_result.info["pause_reason"] == "ask_human"

    runner.resume_after_human()
    continued = runner.continue_run(max_turns=1)

    assert len(continued.steps) == 1
    assert continued.steps[0].action.command["request"]["name"] == "observe"


def test_episode_runner_excludes_human_wait_from_timeout_budget() -> None:
    now_s = [0.0]
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            [
                {
                    "kind": "response",
                    "name": "ask_human",
                    "parameters": {"question": "Should I pick the cube?"},
                },
                {
                    "kind": "tool_call",
                    "name": "close_simulator_env",
                    "parameters": {},
                },
            ]
        )
    )
    tools = _tools_with_handlers("close_simulator_env")
    runtime = OpenEtaAgentRuntime(planner=planner, tools=tools)
    runner = OpenEtaEpisodeRunner(
        runtime=runtime,
        environment=DummyEpisodeEnvironment(),
        clock=lambda: now_s[0],
    )

    paused = runner.run(task="pick milk", max_turns=10, timeout_s=10.0)
    assert paused.metadata["waiting_for_human"] is True

    now_s[0] = 120.0
    runtime.update_memory(
        {
            "type": "human_answer",
            "question": "Should I pick the cube?",
            "answer": "No, close this simulator environment.",
        }
    )
    runner.resume_after_human()
    continued = runner.continue_run(max_turns=2)

    assert continued.truncated is False
    assert continued.metadata["failure_reason"] == {}
    assert continued.metadata["usage"]["elapsed_s"] == 0.0
    assert continued.metadata["usage"]["human_wait_s"] == 120.0
    assert continued.steps[-1].action.command["request"]["name"] == "close_simulator_env"


def test_episode_runner_truncates_at_safety_turn_limit() -> None:
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {
                "kind": "tool_call",
                "name": "observe",
                "parameters": {"reason": "inspect scene"},
            }
        )
    )
    tools = build_default_tool_registry()
    tools.bind_handler(
        "observe",
        lambda context: ToolResult(True, content="objects detected"),
    )
    runtime = OpenEtaAgentRuntime(planner=planner, tools=tools)
    runner = OpenEtaEpisodeRunner(runtime=runtime, environment=DummyEpisodeEnvironment())

    result = runner.run(task="find cube", max_turns=2)

    assert len(result.steps) == 2
    assert result.terminated is False
    assert result.truncated is True
    assert result.metadata["stop_reason"] == "max_turns"
    assert result.metadata["remaining_turns"] == 0


def test_openai_compatible_backend_uses_chat_completions_transport() -> None:
    captured = {}

    def fake_transport(url, body, headers, timeout_s):
        captured["url"] = url
        captured["body"] = body
        captured["headers"] = headers
        captured["timeout_s"] = timeout_s
        return {
            "id": "chatcmpl-test",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            '{"kind": "tool_call", "name": "sam3", '
                            '"parameters": {"source_packet_id": "packet-front", '
                            '"prompt": "cube"}, '
                            '"reasoning": "Need segmentation."}'
                        )
                    },
                }
            ],
            "usage": {"total_tokens": 42},
        }

    backend = OpenAICompatiblePlannerBackend(
        OpenAICompatiblePlannerBackendConfig(
            model="test-model",
            api_base="https://api.example.test",
            api_key="secret-key",
            timeout_s=3.0,
        ),
        transport=fake_transport,
    )
    request = PlannerBackendRequest(
        tool_context={"task": "find cube", "tool_references": []},
        system_prompt="return json",
    )

    result = backend.decide(request)

    assert captured["url"] == "https://api.example.test/v1/chat/completions"
    assert captured["body"]["model"] == "test-model"
    assert captured["headers"]["Authorization"] == "Bearer secret-key"
    assert captured["timeout_s"] == 3.0
    assert result.payload.startswith('{"kind": "tool_call"')
    assert result.details["usage"]["total_tokens"] == 42
    assert result.details["usage_source"] == "provider"
    assert result.details["provider_attempts"] == 1


def test_openai_compatible_backend_retries_transient_provider_timeouts() -> None:
    calls = 0
    sleeps = []

    def flaky_transport(url, body, headers, timeout_s):
        nonlocal calls
        del url, body, headers, timeout_s
        calls += 1
        if calls < 3:
            raise TimeoutError("provider read timed out")
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": '{"kind":"response","name":"talk"}'},
                }
            ],
            "usage": {"total_tokens": 8},
        }

    backend = OpenAICompatiblePlannerBackend(
        OpenAICompatiblePlannerBackendConfig(
            model="test-model",
            api_base="https://api.example.test",
            api_key="secret-key",
            max_attempts=3,
            retry_backoff_s=0.5,
        ),
        transport=flaky_transport,
        sleep=sleeps.append,
    )

    result = backend.decide(
        PlannerBackendRequest(tool_context={"task": "test"}, system_prompt="json")
    )

    assert result.status.value == "planned"
    assert calls == 3
    assert sleeps == [0.5, 1.0]
    assert result.details["provider_attempts"] == 3
    assert [item["attempt"] for item in result.details["retry_errors"]] == [1, 2]


def test_openai_compatible_backend_retries_cloudflare_gateway_errors() -> None:
    calls = 0

    def flaky_transport(url, body, headers, timeout_s):
        nonlocal calls
        del url, body, headers, timeout_s
        calls += 1
        if calls == 1:
            raise ProviderHttpError(522, "connection timed out")
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": '{"kind":"response","name":"talk"}'},
                }
            ],
            "usage": {"total_tokens": 8},
        }

    backend = OpenAICompatiblePlannerBackend(
        OpenAICompatiblePlannerBackendConfig(
            model="test-model",
            api_base="https://api.example.test",
            api_key="secret-key",
            max_attempts=2,
            retry_backoff_s=0,
        ),
        transport=flaky_transport,
    )

    result = backend.decide(
        PlannerBackendRequest(tool_context={"task": "test"}, system_prompt="json")
    )

    assert result.status.value == "planned"
    assert calls == 2
    assert result.details["provider_attempts"] == 2
    assert result.details["retry_errors"][0]["error_type"] == "ProviderHttpError"


def test_openai_compatible_backend_does_not_retry_non_transient_errors() -> None:
    calls = 0

    def invalid_transport(url, body, headers, timeout_s):
        nonlocal calls
        del url, body, headers, timeout_s
        calls += 1
        raise ValueError("invalid provider request")

    backend = OpenAICompatiblePlannerBackend(
        OpenAICompatiblePlannerBackendConfig(
            model="test-model",
            api_base="https://api.example.test",
            api_key="secret-key",
        ),
        transport=invalid_transport,
        sleep=lambda _delay: None,
    )

    result = backend.decide(
        PlannerBackendRequest(tool_context={"task": "test"}, system_prompt="json")
    )

    assert result.status.value == "failed"
    assert result.payload["name"] == "ask_human"
    assert result.details["provider_attempts"] == 1
    assert result.details["retry_errors"] == []
    assert calls == 1


def test_openai_compatible_backend_asks_human_after_timeout_retries_exhausted() -> None:
    calls = 0

    def timed_out_transport(url, body, headers, timeout_s):
        nonlocal calls
        del url, body, headers, timeout_s
        calls += 1
        raise TimeoutError("provider read timed out")

    backend = OpenAICompatiblePlannerBackend(
        OpenAICompatiblePlannerBackendConfig(
            model="test-model",
            api_base="https://api.example.test",
            api_key="secret-key",
            max_attempts=3,
            retry_backoff_s=0,
        ),
        transport=timed_out_transport,
    )

    result = backend.decide(
        PlannerBackendRequest(tool_context={"task": "test"}, system_prompt="json")
    )

    assert result.status.value == "failed"
    assert result.payload["name"] == "ask_human"
    assert result.payload["parameters"]["provider_attempts"] == 3
    assert result.details["provider_attempts"] == 3
    assert len(result.details["retry_errors"]) == 2
    assert calls == 3


def test_openai_compatible_backend_attaches_pending_selection_images(tmp_path: Path) -> None:
    from PIL import Image

    original = tmp_path / "original.png"
    contact_sheet = tmp_path / "selection.png"
    Image.new("RGB", (8, 8), "white").save(original)
    Image.new("RGB", (16, 8), "blue").save(contact_sheet)
    captured = {}

    def fake_transport(url, body, headers, timeout_s):
        del url, headers, timeout_s
        captured["body"] = body
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            '{"kind":"tool_call","name":"select_sam3_detection",'
                            '"parameters":{"sam3_result_id":"sam3-run-selection",'
                            '"detection_id":"detection_001"}}'
                        )
                    },
                }
            ],
            "usage": {"total_tokens": 10},
        }

    backend = OpenAICompatiblePlannerBackend(
        OpenAICompatiblePlannerBackendConfig(
            model="vision-model",
            api_base="https://api.example.test",
            api_key="secret-key",
        ),
        transport=fake_transport,
    )
    result = backend.decide(
        PlannerBackendRequest(
            tool_context={
                "task": "pick alphabet soup",
                    "pending_target_selection": {
                    "result_id": "sam3-run-selection",
                    "selection_bundle": {
                        "original_image_ref": str(original),
                        "contact_sheet_ref": str(contact_sheet),
                    },
                },
            },
            system_prompt="return json",
        )
    )

    user_content = captured["body"]["messages"][1]["content"]
    assert isinstance(user_content, list)
    assert [part["type"] for part in user_content] == [
        "text",
        "image_url",
        "image_url",
        "text",
    ]
    assert all(
        part["image_url"]["url"].startswith("data:image/png;base64,")
        for part in user_content[1:3]
    )
    assert [item["path"] for item in result.details["vision_attachments"]] == [
        str(original),
        str(contact_sheet),
    ]
    assert "base64" not in json.dumps(result.details)


def test_openai_compatible_backend_labels_reviewer_vision_evidence(tmp_path: Path) -> None:
    from PIL import Image

    current = tmp_path / "current.png"
    baseline = tmp_path / "baseline.png"
    Image.new("RGB", (8, 8), "white").save(current)
    Image.new("RGB", (8, 8), "blue").save(baseline)
    captured = {}

    def fake_transport(url, body, headers, timeout_s):
        del url, headers, timeout_s
        captured["body"] = body
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            '{"decision":"approve","reason":"consistent",'
                            '"grasp_outcome":"not_assessed","candidate_id":""}'
                        )
                    },
                }
            ],
            "usage": {"total_tokens": 10},
        }

    backend = OpenAICompatiblePlannerBackend(
        OpenAICompatiblePlannerBackendConfig(
            model="vision-model",
            api_base="https://api.example.test",
            api_key="secret-key",
        ),
        transport=fake_transport,
    )
    result = backend.decide(
        PlannerBackendRequest(
            tool_context={
                "vision_image_paths": [str(current), str(baseline)],
                "vision_evidence": [
                    {"role": "current_scene", "path": str(current)},
                    {
                        "role": "target_source_before_grasp",
                        "path": str(baseline),
                    },
                ],
            },
            system_prompt="review action",
            metadata={"isolated_context": True},
        )
    )

    user_content = captured["body"]["messages"][1]["content"]
    assert [part["type"] for part in user_content] == [
        "text",
        "text",
        "image_url",
        "text",
        "image_url",
        "text",
    ]
    assert user_content[1]["text"] == (
        "Image #1 role: current_scene. This is the current state used for action review."
    )
    assert user_content[3]["text"] == (
        "Image #2 role: target_source_before_grasp. "
        "This is a historical baseline, not the current state."
    )
    assert [item["role"] for item in result.details["vision_attachments"]] == [
        "current_scene",
        "target_source_before_grasp",
    ]


def test_main_agent_prompt_hides_visual_paths_but_still_attaches_images(
    tmp_path: Path,
) -> None:
    from PIL import Image

    current = tmp_path / "current-wrist.png"
    Image.new("RGB", (8, 8), "white").save(current)
    captured = {}

    def fake_transport(url, body, headers, timeout_s):
        del url, headers, timeout_s
        captured["body"] = body
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            '{"kind":"response","name":"talk",'
                            '"parameters":{"message":"observed"}}'
                        )
                    },
                }
            ],
            "usage": {"total_tokens": 10},
        }

    backend = OpenAICompatiblePlannerBackend(
        OpenAICompatiblePlannerBackendConfig(
            model="vision-model",
            api_base="https://api.example.test",
            api_key="secret-key",
        ),
        transport=fake_transport,
    )
    result = backend.decide(
        PlannerBackendRequest(
            tool_context={
                "schema_version": "openeta.agent_context.v2",
                "vision_image_paths": [str(current)],
                "vision_evidence": [
                    {
                        "role": "current_scene",
                        "frame_id": "wrist",
                        "packet_id": "obs-0009",
                        "path": str(current),
                    }
                ],
                "current_observation": {
                    "visual_evidence": [
                        {
                            "frame_id": "wrist",
                            "packet_id": "obs-0009",
                            "path": str(current),
                        }
                    ]
                },
                "decision_state": {
                    "current_observation_packet": {
                        "packet_ids": ["obs-0009"],
                        "camera_artifacts": [
                            {
                                "frame_id": "wrist",
                                "packet_id": "obs-0009",
                                "path": str(current),
                            }
                        ],
                    }
                },
            },
            system_prompt="return json",
        )
    )

    user_message = next(
        item
        for item in reversed(captured["body"]["messages"])
        if item["role"] == "user"
    )
    assert isinstance(user_message["content"], list)
    prompt_text = user_message["content"][-1]["text"]
    assert str(current) not in prompt_text
    assert "vision_image_paths" not in prompt_text
    assert "obs-0009" in prompt_text
    assert [item["path"] for item in result.details["vision_attachments"]] == [
        str(current)
    ]


def test_openai_compatible_backend_attaches_scene_and_asset_reference(tmp_path: Path) -> None:
    from PIL import Image

    scene = tmp_path / "scene.png"
    reference = tmp_path / "reference.png"
    Image.new("RGB", (32, 24), "blue").save(scene)
    Image.new("RGB", (12, 10), "red").save(reference)
    captured = {}

    def fake_transport(url, body, headers, timeout_s):
        del url, headers, timeout_s
        captured["body"] = body
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            '{"kind":"tool_call","name":"sam3",'
                            '"parameters":{"source_packet_id":"packet-scene",'
                            '"prompt":"alphabet soup can",'
                            '"roi_bbox_xyxy":[2,3,20,18]}}'
                        )
                    },
                }
            ],
            "usage": {"total_tokens": 10},
        }

    backend = OpenAICompatiblePlannerBackend(
        OpenAICompatiblePlannerBackendConfig(
            model="vision-model",
            api_base="https://api.example.test",
            api_key="secret-key",
        ),
        transport=fake_transport,
    )
    result = backend.decide(
        PlannerBackendRequest(
            tool_context={
                "task": "pick alphabet soup",
                    "pending_reference_localization": {
                    "scene_image": str(scene),
                    "source_packet_id": "packet-scene",
                    "reference_images": [str(reference)],
                },
            },
            system_prompt="return json",
        )
    )

    user_content = captured["body"]["messages"][1]["content"]
    assert [part["type"] for part in user_content] == [
        "text",
        "image_url",
        "image_url",
        "text",
    ]
    assert [item["path"] for item in result.details["vision_attachments"]] == [
        str(scene),
        str(reference),
    ]


def test_openai_compatible_backend_estimates_tokens_when_usage_is_missing() -> None:
    def fake_transport(url, body, headers, timeout_s):
        del url, body, headers, timeout_s
        return {
            "id": "chatcmpl-no-usage",
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "content": (
                            '{"kind":"response","name":"task_complete",'
                            '"parameters":{"success":true}}'
                        )
                    },
                }
            ],
        }

    backend = OpenAICompatiblePlannerBackend(
        OpenAICompatiblePlannerBackendConfig(
            model="unknown-provider-model",
            api_base="https://api.example.test",
            api_key="secret-key",
        ),
        transport=fake_transport,
    )

    result = backend.decide(
        PlannerBackendRequest(tool_context={"task": "test"}, system_prompt="json")
    )

    assert result.details["usage_source"] == "estimated"
    assert result.details["usage"]["prompt_tokens"] > 0
    assert result.details["usage"]["completion_tokens"] > 0
    assert result.details["usage"]["total_tokens"] == (
        result.details["usage"]["prompt_tokens"] + result.details["usage"]["completion_tokens"]
    )
    assert result.details["usage_estimator"]["prompt"]


def test_openai_compatible_backend_derives_total_from_partial_usage() -> None:
    def fake_transport(url, body, headers, timeout_s):
        del url, body, headers, timeout_s
        return {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {"content": '{"kind":"response","name":"talk"}'},
                }
            ],
            "usage": {"prompt_tokens": "12", "completion_tokens": 3},
        }

    backend = OpenAICompatiblePlannerBackend(
        OpenAICompatiblePlannerBackendConfig(
            model="test-model",
            api_base="https://api.example.test",
            api_key="secret-key",
        ),
        transport=fake_transport,
    )

    result = backend.decide(
        PlannerBackendRequest(tool_context={"task": "test"}, system_prompt="json")
    )

    assert result.details["usage_source"] == "provider_derived"
    assert result.details["usage"]["total_tokens"] == 15


def test_apikey_file_loader_reads_newapi_channel_without_printing_secret(tmp_path) -> None:
    apikey_path = tmp_path / "apikey.md"
    apikey_path.write_text(
        'sk-local-secret\n{"_type":"newapi_channel_conn",'
        '"key":"sk-json-secret","url":"https://open.example.test"}\n',
        encoding="utf-8",
    )

    config: PlannerProviderConfig = read_apikey_file(apikey_path)

    assert config.provider == "openai-compatible"
    assert config.api_base == "https://open.example.test"
    assert config.api_key == "sk-json-secret"
    assert config.redacted()["api_key"] != "sk-json-secret"
