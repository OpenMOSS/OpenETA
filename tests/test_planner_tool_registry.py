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
    _default_tool_planner_system_prompt,
    _host_obligation_decision,
    _select_anyplace_candidate,
    _matching_depth_enhancement,
    _grasp_compile_obligation,
    _grasp_sensor_safety_obligation,
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
from agent.runtime.token_counting import DEFAULT_CONTEXT_WINDOW_TOKENS, estimate_text_tokens
from agent.tools.handlers import bind_dummy_tool_handlers
from agent.tools.registry import (
    TOOL_RESULT_SCHEMA_VERSION,
    ToolExecutionContext,
    ToolRegistry,
    ToolResult,
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
        metadata={"step_idx": 1},
    )


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
                    {"kind": "rgb", "frame_id": frame_id, "path": str(rgb)},
                    {"kind": "depth", "frame_id": frame_id, "path": str(depth)},
                )
            ]
        },
    )


def _tools_with_handlers(*names: str) -> ToolRegistry:
    tools = build_default_tool_registry()
    for name in names:
        tools.bind_handler(name, lambda context: ToolResult(True, content="ok"))
    return tools


def _record_pending_sam3_selection(
    memory: AgentMemory,
    *,
    original_image_ref: str = "agentview.png",
    contact_sheet_ref: str = "selection.png",
    segmentation_mode: str = "point_prompt",
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
                                    "result_id": "sam3-run-selection",
                                    "prompt": "alphabet soup",
                                    "source_image": original_image_ref,
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


def _record_anygrasp_candidate_policy(
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


def test_static_planner_backend_executes_registered_tool_handler() -> None:
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
                "parameters": {"image": "front", "prompt": "cube"},
                "reasoning": "Need segmentation before grasp planning.",
            }
        )
    )
    runtime = OpenEtaAgentRuntime(planner=planner, tools=tools)
    runtime.start_session(task="find the cube")

    action = runtime.act(_observation())

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
    tools.bind_handler("scene_detector", lambda context: ToolResult(True, content="objects"))

    result = tools.call("scene_detector", {"image": "front"}, observation=_observation())

    assert result.success is True
    assert [event["phase"] for event in events] == ["start", "end"]
    assert [event["name"] for event in events] == ["scene_detector", "scene_detector"]
    assert events[0]["parameters"] == {"image": "front"}
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
    assert decision.metadata["validation_attempts"] == 2
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


def test_default_planner_prompt_uses_first_class_simulator_creation_tool() -> None:
    prompt = _default_tool_planner_system_prompt()

    assert "tool_call::create_simulator_env" in prompt
    assert "only environment-creation path" in prompt
    assert "Do not invoke create_env or close_env through python_exec or code_policy." in prompt


def test_planner_enforces_reference_guided_sam3_roi_obligation() -> None:
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            [
                {
                    "kind": "tool_call",
                    "name": "observe",
                    "parameters": {},
                },
                {
                    "kind": "tool_call",
                    "name": "sam3",
                    "parameters": {
                        "image": "scene.png",
                        "prompt": "alphabet soup can",
                        "roi_bbox_xyxy": [12, 18, 90, 110],
                    },
                },
            ]
        ),
        max_validation_retries=1,
    )
    memory = AgentMemory()
    memory.start_session(task="pick alphabet soup")
    memory.save_fact(
        "pending_reference_localization",
        {
            "scene_image": "scene.png",
            "reference_images": ["reference.png"],
            "target_object": "alphabet_soup",
        },
        source="retrieve_asset_reference",
    )

    decision = planner.plan(
        _observation(),
        memory=memory,
        tools=_tools_with_handlers("observe", "sam3"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "sam3"
    assert decision.parameters["roi_bbox_xyxy"] == [12, 18, 90, 110]
    history = decision.metadata["validation_attempt_history"]
    assert "reference localization obligation" in history[0]["validation_errors"][0]
    assert history[1]["validation_errors"] == []


def test_planner_enforces_exact_reference_guided_sam3_positive_point() -> None:
    points = [{"x": 212.0, "y": 308.0, "label": 1}]
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            [
                {
                    "kind": "tool_call",
                    "name": "sam3",
                    "parameters": {
                        "image": "scene.png",
                        "positive_points": [{"x": 213.0, "y": 308.0, "label": 1}],
                    },
                },
                {
                    "kind": "tool_call",
                    "name": "sam3",
                    "parameters": {
                        "image": "scene.png",
                        "positive_points": points,
                    },
                },
            ]
        ),
        max_validation_retries=1,
    )
    memory = AgentMemory()
    memory.start_session(task="pick alphabet soup")
    memory.save_fact(
        "pending_reference_localization",
        {
            "scene_image": "scene.png",
            "target_object": "alphabet_soup",
            "positive_points": points,
            "required_parameter": "positive_points",
        },
        source="retrieve_asset_reference",
    )

    decision = planner.plan(
        _observation(),
        memory=memory,
        tools=_tools_with_handlers("sam3"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "sam3"
    assert decision.parameters["positive_points"] == points
    history = decision.metadata["validation_attempt_history"]
    assert "exact positive_points" in history[0]["validation_errors"][0]
    assert history[1]["validation_errors"] == []


def test_reference_guided_sam3_accepts_byte_identical_scene_copy(tmp_path: Path) -> None:
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
                    "image": str(rematerialized_scene),
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
                                    "image": scene,
                                    "positive_points": points,
                                },
                                "outputs": {
                                    "result_id": "sam-verified",
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


def test_host_selects_decisive_sam3_mask_for_verified_reference_point() -> None:
    memory = AgentMemory()
    memory.save_fact(
        "pending_sam3_selection",
        {
            "result_id": "sam-verified",
            "reference_verification": {
                "decision": "match",
                "confidence": 0.98,
            },
            "candidates": [
                {"id": "detection_000", "rank": 0, "score": 0.976},
                {"id": "detection_001", "rank": 1, "score": 0.672},
                {"id": "detection_002", "rank": 2, "score": 0.320},
            ],
        },
        source="sam3",
    )
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {"kind": "response", "name": "ask_human", "parameters": {"question": "unused"}}
        )
    )

    decision = planner.plan(
        _observation(),
        memory=memory,
        tools=_tools_with_handlers("select_sam3_detection"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "select_sam3_detection"
    assert decision.parameters["sam3_result_id"] == "sam-verified"
    assert decision.parameters["detection_id"] == "detection_000"
    assert decision.metadata["execution_model"] == "host_obligation_dispatch"
    assert decision.metadata["host_obligation"]["schema_version"] == (
        "openeta.reference_verified_selection.v1"
    )


def test_host_keeps_ambiguous_verified_sam3_masks_for_model_review() -> None:
    memory = AgentMemory()
    memory.save_fact(
        "pending_sam3_selection",
        {
            "result_id": "sam-ambiguous",
            "reference_verification": {"decision": "match"},
            "candidates": [
                {"id": "detection_000", "rank": 0, "score": 0.91},
                {"id": "detection_001", "rank": 1, "score": 0.82},
            ],
        },
        source="sam3",
    )
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {
                "kind": "tool_call",
                "name": "select_sam3_detection",
                "parameters": {
                    "sam3_result_id": "sam-ambiguous",
                    "detection_id": "detection_001",
                    "selection_confidence": 0.7,
                    "reason": "mask boundary review",
                },
            }
        )
    )

    decision = planner.plan(
        _observation(),
        memory=memory,
        tools=_tools_with_handlers("select_sam3_detection"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "select_sam3_detection"
    assert decision.parameters["detection_id"] == "detection_001"
    assert decision.metadata["execution_model"] == "closed_loop_tool_calling"


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


def test_sam3_point_validation_rejects_molmopoint_fields_then_accepts_xy() -> None:
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            [
                {
                    "kind": "tool_call",
                    "name": "sam3",
                    "parameters": {
                        "mode": "points",
                        "image": "tmp/scene.jpg",
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
                        "image": "tmp/scene.jpg",
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
    prompt = _default_tool_planner_system_prompt()

    assert "SAM3 also supports mode=points" in prompt
    assert "pixel_x/pixel_y to SAM3 x/y" in prompt
    assert "image_sources[image_index]" in prompt


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


def test_contact_graspnet_validation_requires_concrete_sam3_artifact() -> None:
    intrinsics = {"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "scale": 1000.0}
    valid_parameters = {
        "rgb": "tmp/rgb.png",
        "depth": "tmp/depth.png",
        "object_mask": {
            "mask_ref": "tmp/object-mask.png",
            "source_image": "tmp/rgb.png",
            "label": "bottle",
        },
        "intrinsics": intrinsics,
    }
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            [
                {
                    "kind": "tool_call",
                    "name": "contact_graspnet",
                    "parameters": {
                        "rgb": "latest_rgb",
                        "depth": "latest_depth",
                        "object_mask": "latest_mask",
                        "intrinsics": {},
                    },
                },
                {
                    "kind": "tool_call",
                    "name": "contact_graspnet",
                    "parameters": valid_parameters,
                },
            ]
        ),
        max_validation_retries=1,
    )
    memory = AgentMemory()
    memory.start_session(task="predict targeted Panda grasps")

    decision = planner.plan(
        _observation(),
        memory=memory,
        tools=_tools_with_handlers("contact_graspnet"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "contact_graspnet"
    assert decision.parameters == valid_parameters
    assert decision.metadata["validation_attempts"] == 2


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
    valid_intrinsics = {"fx": 1.0, "fy": 1.0, "cx": 0.5, "cy": 0.5, "scale": 1000.0}
    candidate = {
        "id": "grasp_000",
        "frame": "camera",
        "camera_frame": "opencv",
        "score": 0.5,
        "translation_xyz": [0.1, 0.2, 0.3],
        "rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        "gripper_tip_position_xyz": [0.13, 0.2, 0.3],
        "depth": 0.03,
        "width": 0.06,
        "height": 0.03,
    }
    valid_parameters = {
        "rgb": "tmp/rgb.png",
        "depth": "tmp/depth.png",
        "object_mask": "tmp/object-mask.png",
        "placement_region_mask": {
            "mask_ref": "tmp/placement-mask.png",
            "source_image": "tmp/rgb.png",
            "label": "rack slot",
        },
        "intrinsics": valid_intrinsics,
        "selected_grasp": {
            "candidate": candidate,
            "source": {
                "mode": "targeted",
                "rgb": "tmp/rgb.png",
                "depth": "tmp/depth.png",
                "object_mask": "tmp/object-mask.png",
                "intrinsics": valid_intrinsics,
            },
        },
    }
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


def test_anyplace_validation_accepts_complete_graspgenx_source() -> None:
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

    assert decision.action == "anyplace"
    assert decision.parameters == parameters


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


def test_callable_planner_backend_accepts_json_string_payload() -> None:
    def model_wrapper(request: PlannerBackendRequest) -> str:
        assert "tool_references" in request.tool_context
        return """
        ```json
        {"kind": "tool_call", "name": "hand_pose_database",
         "parameters": {"object": "cube", "task": "pick"},
         "reasoning": "Need a reference pose."}
        ```
        """

    tools = build_default_tool_registry()
    tools.bind_handler(
        "hand_pose_database",
        lambda context: {"content": "pose found", "details": {"object": "cube"}},
    )
    planner = ToolCallingPlanner(
        CallablePlannerBackend(model_wrapper, provider="unit", model="json-string")
    )
    runtime = OpenEtaAgentRuntime(planner=planner, tools=tools)
    runtime.start_session(task="pick cube")

    action = runtime.act(_observation())

    assert action.command["status"] == "executed"
    assert action.command["tool_calls"][0]["name"] == "hand_pose_database"
    assert action.command["tool_calls"][0]["result"]["content"] == "pose found"


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
    assert command["metadata"]["execution_rule"]["mode"] == "skill_guidance_only"


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
    assert "Recommended Tool Sequence" in selected_pick["content"]
    assert "allowed_tools" in pick
    assert "required_tools" not in pick
    assert "safety_checks" not in pick
    assert "move_to" in pick["allowed_tools"]
    assert selected_pick["allowed_tools"] == pick["allowed_tools"]
    assert {skill["name"] for skill in context["skill_references"]} == {
        skill["name"] for skill in context["selected_skill_guidance"]
    }
    assert context["skill_usage"]["inspection_recommended"][0] == "pick"
    assert context["skill_usage"]["inspection_required"] == []


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

    assert context["vision_image_paths"] == ["/exact/session/cameras.0.agentview.rgb.png"]
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
                "parameters": {"target_pose": {"xyz": [0.1, 0.2, 0.3]}},
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
        "parameters": {"profile": {}},
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


def test_default_context_includes_current_sim_skill_without_hard_inspection_gate() -> None:
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

    tools.bind_handler("slam", lambda context: ToolResult(True, content="map ready"))
    rebound_context = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=tools,
        skills=build_default_skill_registry(),
    )
    assert "slam" in {reference["name"] for reference in rebound_context["tool_references"]}


def test_planner_rejects_registered_tool_without_handler() -> None:
    planner = ToolCallingPlanner(
        StaticPlannerBackend({"kind": "tool_call", "name": "slam", "parameters": {}}),
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
        "Tool requested by planner is not executable: slam."
    ]


def test_current_sim_creation_task_accepts_first_valid_world_mutating_decision() -> None:
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
    assert "Call `observe`" in pick.content
    assert "Extract the target phrase from the user task" in pick.content
    assert "Call `sam3`" in pick.content
    assert "把桌上的罐子抓起来" in pick.content
    assert "can" in pick.content
    assert "Do not pass a non-English user phrase directly to `sam3`" in pick.content
    assert "Stop after `sam3` and inspect its result" in pick.content
    assert "do not" in pick.content.lower()
    assert "default to `detections[0]`" in pick.content
    assert "static post-close image is not evidence" in pick.content.lower()
    assert "exact target asset name from the task" in pick.content
    assert "Do not add" in pick.content
    assert "use the\n`embodiment_explore` skill" in pick.content
    assert "does not silently recalibrate one" in pick.content
    assert "grasp candidate list" in pick.content
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


def test_builtin_task_skills_are_loaded_from_markdown_guidance() -> None:
    skills = build_default_skill_registry()

    for name in ("pick", "place", "push", "pull", "stack"):
        skill = skills.get(name)
        assert skill.source == f"markdown:skills/{name}.md"
        assert skill.editable is True
        assert skill.version == "v1"
        assert skill.task_patterns
        assert skill.allowed_tools
        assert "text guidance only" in skill.content
        assert "executable" in skill.content


def test_planner_prompt_guards_pick_against_direct_motion_and_localizes_sam3_prompt() -> None:
    prompt = _default_tool_planner_system_prompt()

    assert "do not start with move_to or gripper_control" in prompt
    assert "prior perception/grasp tool result" in prompt
    assert "`罐子` -> `can`" in prompt
    assert "before calling SAM3" in prompt
    assert "never run grasp estimation on the basket" in prompt


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
    assert "Recommended Tool Sequence" in place["content"]
    assert "Never run grasp estimation on the receptacle" in place["content"]
    assert "Call `gripper_control`" in place["content"]
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
    assert {"scene_detector", "move_to", "gripper_control"}.issubset(tool_names)


def test_agent_memory_tracks_working_facts_artifacts_skill_notes_and_compaction() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube")

    memory.save_fact("target", {"name": "cube"}, source="unit")
    memory.save_artifact(
        "mask",
        {
            "id": "mask-1",
            "tool": "sam3",
            "path": "/tmp/openeta-test/tool_result/mask.json",
            "grep_hint": "grep -n '<pattern>' /tmp/openeta-test/tool_result/mask.json",
            "dashboard_url": "http://sim.example/session/session-1",
            "images": [{"path": "/tmp/openeta-test/image/rgb/front.png"}],
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
        "/tmp/openeta-test/image/rgb/front.png"
    ]
    assert context["working_memory"]["skill_notes"]["pick"][0]["note"]["failure"] == "empty mask"
    assert "facts=['scene_epoch', 'target']" in summary
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


def test_planner_context_auto_compacts_when_budget_threshold_is_reached() -> None:
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

    assert any(event.event_type == "memory_compacted" for event in memory.events)
    assert context["context_budget"]["schema_version"] == "openeta.context_budget.v1"
    assert context["context_budget"]["auto_compact_triggered"] is True
    assert context["memory"]["working_memory"]["compact_summary"]


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
    assert context["context_budget"]["trigger_tokens"] == int(DEFAULT_CONTEXT_WINDOW_TOKENS * 0.9)


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
    assert "compile_grasp_seed" in grasp_artifact["next_tool_hint"]

    retained = context["retained_targeted_grasp"]
    assert retained["candidate"]["id"] == "grasp_000"
    assert retained["source"]["mode"] == "targeted"
    assert retained["source"]["rgb"] == f"{long_session_root}agentview.rgb.png"
    assert retained["source"]["depth"] == f"{long_session_root}agentview.depth.png"
    assert retained["source"]["object_mask"] == f"{long_session_root}mask_000.png"
    assert "[truncated]" not in json.dumps(retained)


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
    assert "camera_pose_to_world" in artifact["next_tool_hint"]


def test_anygrasp_policy_activates_highest_score_candidate() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube")
    _record_anygrasp_candidate_policy(memory)

    context = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )

    policy = context["grasp_candidate_policy"]
    assert policy["status"] == "active"
    assert policy["active_rank"] == 0
    assert policy["active_candidate"]["id"] == "grasp_000"
    assert policy["remaining_candidate_ids"] == ["grasp_001"]


def test_combined_pick_place_allows_grasp_compilation_before_anyplace() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube and place it in basket")
    _record_anygrasp_candidate_policy(memory)
    active = memory.anygrasp_candidate_policy()["active_candidate"]
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {
                "kind": "tool_call",
                "name": "compile_grasp_seed",
                "parameters": {
                    "camera_pose": active,
                    "camera_extrinsics": {
                        "pos": [0.0, 0.0, 0.0],
                        "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                    },
                    "scene_epoch": 0,
                    "target_class": "boxed_item",
                },
            }
        )
    )
    observation = _observation()
    observation.task = "pick cube and place it in basket"

    decision = planner.plan(
        observation,
        memory=memory,
        tools=_tools_with_handlers("compile_grasp_seed", "sam3", "anyplace"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "compile_grasp_seed"
    assert decision.metadata["validation_attempts"] == 1
    assert decision.metadata["validation_attempt_history"][0]["validation_errors"] == []


def test_anyplace_waits_for_final_attachment_pass() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube and place it in basket")
    _record_anygrasp_candidate_policy(memory)
    active = memory.anygrasp_candidate_policy()["active_candidate"]
    retained = memory.retained_targeted_grasp()
    anyplace_parameters = {
        "rgb": retained["source"]["rgb"],
        "depth": retained["source"]["depth"],
        "object_mask": retained["source"]["object_mask"],
        "intrinsics": retained["source"]["intrinsics"],
        "placement_region_mask": {
            "mask_ref": "tmp/mask_000.png",
            "source_image": retained["source"]["rgb"],
        },
        "selected_grasp": {
            "candidate": retained["candidate"],
            "source": retained["source"],
        },
        "depth_truncation": 2.0,
    }
    compile_parameters = {
        "camera_pose": active,
        "camera_extrinsics": {
            "pos": [0.0, 0.0, 0.0],
            "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        },
        "scene_epoch": 0,
        "target_class": "boxed_item",
    }
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            [
                {"kind": "tool_call", "name": "anyplace", "parameters": anyplace_parameters},
                {
                    "kind": "tool_call",
                    "name": "compile_grasp_seed",
                    "parameters": compile_parameters,
                },
            ]
        ),
        max_validation_retries=1,
    )
    observation = _observation()
    observation.task = "pick cube and place it in basket"

    decision = planner.plan(
        observation,
        memory=memory,
        tools=_tools_with_handlers("anyplace", "compile_grasp_seed"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "compile_grasp_seed"
    first_errors = decision.metadata["validation_attempt_history"][0]["validation_errors"]
    assert any("must wait until the final grasp candidate" in error for error in first_errors)


def test_combined_pick_place_requires_placement_mask_on_retained_rgb() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube and place it in basket")
    _record_anygrasp_candidate_policy(memory)
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            [
                {
                    "kind": "tool_call",
                    "name": "sam3",
                    "parameters": {"image": "tmp/latest.png", "prompt": "basket"},
                },
                {
                    "kind": "tool_call",
                    "name": "sam3",
                    "parameters": {"image": "tmp/rgb.png", "prompt": "basket"},
                },
            ]
        ),
        max_validation_retries=1,
    )
    observation = _observation()
    observation.task = "pick cube and place it in basket"

    decision = planner.plan(
        observation,
        memory=memory,
        tools=_tools_with_handlers("sam3", "anyplace"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "sam3"
    assert decision.parameters["image"] == "tmp/rgb.png"
    assert decision.metadata["validation_attempts"] == 1
    canonicalizations = decision.metadata["host_parameter_canonicalizations"]
    assert canonicalizations[0]["reason"] == ("freeze_placement_mask_to_targeted_grasp_rgb")


def test_placement_mask_accepts_byte_identical_same_epoch_rgb_copy(tmp_path) -> None:
    retained_rgb = tmp_path / "observation-0004.png"
    rematerialized_rgb = tmp_path / "observation-0005.png"
    retained_rgb.write_bytes(b"same-scene-rgb")
    rematerialized_rgb.write_bytes(b"same-scene-rgb")
    memory = AgentMemory()
    memory.start_session(task="pick cube and place it in basket")
    _record_anygrasp_candidate_policy(memory)
    artifact = memory.artifacts["anygrasp_grasp_candidates_latest"]["value"]
    artifact["selected_grasp_source"]["rgb"] = str(retained_rgb)
    artifact["source_rgb"] = str(retained_rgb)
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {
                "kind": "tool_call",
                "name": "sam3",
                "parameters": {"image": str(rematerialized_rgb), "prompt": "basket"},
            }
        )
    )
    observation = _observation()
    observation.task = "pick cube and place it in basket"

    decision = planner.plan(
        observation,
        memory=memory,
        tools=_tools_with_handlers("sam3", "anyplace"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "sam3"
    assert decision.parameters["image"] == str(rematerialized_rgb)
    assert decision.metadata["validation_attempt_history"][0]["validation_errors"] == []


def test_asset_reference_canonicalizes_current_scene_image_path() -> None:
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
                        "scene_image": "/tmp/session/agentview.rgb.png",
                    },
                },
                {
                    "kind": "tool_call",
                    "name": "retrieve_asset_reference",
                    "parameters": {
                        "environment": "libero",
                        "target_object": "alphabet soup",
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
    assert decision.parameters["scene_image"] == exact_path
    assert decision.metadata["validation_attempts"] == 1
    canonicalizations = decision.metadata["host_parameter_canonicalizations"]
    assert canonicalizations[0]["reason"] == ("bind_reference_localizer_to_current_camera_rgb")


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
    assert decision.metadata["validation_attempt_history"][0]["validation_errors"] == []


def test_targeted_grasp_obligation_joins_selected_mask_to_current_rgbd(
    tmp_path: Path,
) -> None:
    selected_rgb = tmp_path / "selected" / "agentview.rgb.png"
    current_rgb = tmp_path / "current" / "agentview.rgb.png"
    current_depth = tmp_path / "current" / "agentview.depth.png"
    selected_rgb.parent.mkdir()
    current_rgb.parent.mkdir()
    selected_rgb.write_bytes(b"same-agentview-scene")
    current_rgb.write_bytes(b"same-agentview-scene")
    current_depth.write_bytes(b"depth")
    memory = AgentMemory()
    _record_pending_sam3_selection(memory, original_image_ref=str(selected_rgb))
    memory.resolve_sam3_selection(
        result_id="sam3-run-selection",
        detection_id="detection_000",
        selection_source="main_agent_vlm",
    )
    observation = EnvObservation(
        task="pick alphabet soup",
        cameras=[
            CameraFrame(
                frame_id="agentview",
                rgb=[[[0, 0, 0]]],
                depth=[[1.0]],
                intrinsics={"fx": 100.0, "fy": 100.0, "cx": 0.5, "cy": 0.5, "scale": 1000},
            )
        ],
        robot=RobotState(),
        metadata={
            "image_artifacts": [
                {"kind": "rgb", "frame_id": "agentview", "path": str(current_rgb)},
                {"kind": "depth", "frame_id": "agentview", "path": str(current_depth)},
            ]
        },
    )

    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=_tools_with_handlers("grasp_pose_estimate"),
        skills=build_default_skill_registry(),
    )

    obligation = context["targeted_grasp_obligation"]
    assert obligation["required_parameters"] == {
        "mode": "targeted",
        "rgb": str(current_rgb),
        "depth": str(current_depth),
        "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 0.5, "cy": 0.5, "scale": 1000},
        "object_mask": {
            "mask_ref": "tmp/mask_000.png",
            "source_image": str(current_rgb),
            "result_id": "sam3-run-selection",
            "detection_id": "detection_000",
        },
        "camera_frame_id": "agentview",
        "scene_epoch": 0,
        "hints": {"depth_cutoff_factor": 1.0},
    }
    assert obligation["source_rematerialized"] is True

    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {"kind": "response", "name": "talk", "parameters": {"message": "unused"}}
        )
    )
    decision = planner.plan(
        observation,
        memory=memory,
        tools=_tools_with_handlers("grasp_pose_estimate"),
        skills=build_default_skill_registry(),
    )
    assert decision.action == "grasp_pose_estimate"
    assert decision.parameters == obligation["required_parameters"]
    assert decision.metadata["execution_model"] == "host_obligation_dispatch"


def test_targeted_grasp_obligation_prefers_usable_enhanced_depth(
    tmp_path: Path,
) -> None:
    selected_rgb = tmp_path / "selected" / "agentview.rgb.png"
    current_rgb = tmp_path / "current" / "agentview.rgb.png"
    current_depth = tmp_path / "current" / "agentview.depth.png"
    fused_depth = tmp_path / "enhanced" / "agentview.fused.png"
    selected_rgb.parent.mkdir()
    current_rgb.parent.mkdir()
    fused_depth.parent.mkdir()
    selected_rgb.write_bytes(b"same-agentview-scene")
    current_rgb.write_bytes(b"same-agentview-scene")
    current_depth.write_bytes(b"raw-depth")
    fused_depth.write_bytes(b"enhanced-depth")
    memory = AgentMemory()
    _record_pending_sam3_selection(memory, original_image_ref=str(selected_rgb))
    memory.resolve_sam3_selection(
        result_id="sam3-run-selection",
        detection_id="detection_000",
        selection_source="main_agent_vlm",
    )
    memory.save_artifact(
        "enhance_depth_depth_enhancement_agentview",
        {
            "type": "depth_enhancement",
            "tool": "enhance_depth",
            "index": "agentview",
            "camera_id": "agentview",
            "source_rgb": str(current_rgb),
            "source_depth": str(current_depth),
            "fused_depth_png": str(fused_depth),
            "report_path": str(tmp_path / "report.json"),
            "provenance_mask_png": str(tmp_path / "provenance.png"),
            "point_cloud_npz": str(tmp_path / "points.npz"),
            "quality": {
                "use_for_grasp_candidate_generation": True,
                "use_for_collision_clearance": False,
            },
        },
        source="enhance_depth",
    )
    observation = EnvObservation(
        task="pick cube",
        cameras=[
            CameraFrame(
                frame_id="agentview",
                rgb=[[[0, 0, 0]]],
                depth=[[1.0]],
                intrinsics={"fx": 100.0, "fy": 100.0, "cx": 0.5, "cy": 0.5, "scale": 1000},
            )
        ],
        robot=RobotState(),
        metadata={
            "image_artifacts": [
                {"kind": "rgb", "frame_id": "agentview", "path": str(current_rgb)},
                {"kind": "depth", "frame_id": "agentview", "path": str(current_depth)},
            ]
        },
    )

    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=_tools_with_handlers("grasp_pose_estimate"),
        skills=build_default_skill_registry(),
    )

    required = context["targeted_grasp_obligation"]["required_parameters"]
    assert required["depth"] == str(fused_depth)
    assert required["hints"]["depth_source"] == "enhanced_depth"
    assert required["hints"]["depth_enhancement"]["provenance_mask_png"] == str(
        tmp_path / "provenance.png"
    )
    assert (
        required["hints"]["depth_enhancement"]["quality"]["use_for_collision_clearance"]
        is False
    )
    assert required["hints"]["collision_check"] is False
    assert required["hints"]["depth_enhancement"]["requires_sensor_safety_check"] is True


def test_matching_depth_enhancement_rejects_stale_digest_and_epoch(
    tmp_path: Path,
) -> None:
    rgb = tmp_path / "rgb.png"
    depth = tmp_path / "depth.png"
    candidate = tmp_path / "candidate.png"
    rgb.write_bytes(b"rgb-v1")
    depth.write_bytes(b"depth-v1")
    candidate.write_bytes(b"candidate")
    artifact = {
        "type": "depth_enhancement",
        "camera_id": "wrist",
        "source_rgb": str(rgb),
        "source_depth": str(depth),
        "source_rgb_sha256": sha256(rgb.read_bytes()).hexdigest(),
        "source_depth_sha256": sha256(depth.read_bytes()).hexdigest(),
        "scene_epoch": 7,
        "fused_depth_png": str(candidate),
        "quality": {"use_for_grasp_candidate_generation": True},
    }

    assert (
        _matching_depth_enhancement(
            {"enhancement": artifact},
            frame_id="wrist",
            source_rgb=str(rgb),
            source_depth=str(depth),
            scene_epoch=7,
        )
        is not None
    )
    assert (
        _matching_depth_enhancement(
            {"enhancement": artifact},
            frame_id="wrist",
            source_rgb=str(rgb),
            source_depth=str(depth),
            scene_epoch=8,
        )
        is None
    )
    depth.write_bytes(b"depth-v2")
    assert (
        _matching_depth_enhancement(
            {"enhancement": artifact},
            frame_id="wrist",
            source_rgb=str(rgb),
            source_depth=str(depth),
            scene_epoch=7,
        )
        is None
    )


def test_overwidth_grasps_retry_same_target_on_alternate_camera(tmp_path: Path) -> None:
    agent_rgb = tmp_path / "agentview.rgb.png"
    agent_depth = tmp_path / "agentview.depth.png"
    waist_rgb = tmp_path / "waist.rgb.png"
    waist_depth = tmp_path / "waist.depth.png"
    for index, path in enumerate((agent_rgb, agent_depth, waist_rgb, waist_depth)):
        path.write_bytes(f"artifact-{index}".encode())
    memory = AgentMemory()
    _record_pending_sam3_selection(memory, original_image_ref=str(agent_rgb))
    memory.resolve_sam3_selection(
        result_id="sam3-run-selection",
        detection_id="detection_000",
        selection_source="main_agent_vlm",
    )
    _record_overwidth_grasp_policy(
        memory,
        backend="anygrasp",
        source_rgb=str(agent_rgb),
        camera_frame_id="agentview",
    )
    observation = _rgbd_observation(
        task="pick alphabet soup",
        views=[
            ("agentview", agent_rgb, agent_depth),
            ("waist", waist_rgb, waist_depth),
        ],
    )
    tools = _tools_with_handlers("sam3", "grasp_pose_estimate")
    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=tools,
        skills=build_default_skill_registry(),
    )

    fallback = context["grasp_estimation_fallback_obligation"]
    assert fallback["stage"] == "alternate_camera_segmentation"
    assert fallback["required_parameters"] == {
        "mode": "text",
        "image": str(waist_rgb),
        "prompt": "alphabet soup",
    }
    decision = ToolCallingPlanner(StaticPlannerBackend([])).plan(
        observation,
        memory=memory,
        tools=tools,
        skills=build_default_skill_registry(),
    )
    assert decision.action == "sam3"
    assert decision.parameters == fallback["required_parameters"]

    _record_pending_sam3_selection(memory, original_image_ref=str(waist_rgb))
    memory.resolve_sam3_selection(
        result_id="sam3-run-selection",
        detection_id="detection_000",
        selection_source="main_agent_vlm",
    )
    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=tools,
        skills=build_default_skill_registry(),
    )
    fallback = context["grasp_estimation_fallback_obligation"]
    assert fallback["stage"] == "alternate_camera_estimation"
    assert fallback["required_parameters"]["rgb"] == str(waist_rgb)
    assert "excluded_backends" not in fallback["required_parameters"]["hints"]


def test_passive_views_exhaust_before_active_wrist_refinement(tmp_path: Path) -> None:
    paths = {
        name: (tmp_path / f"{name}.rgb.png", tmp_path / f"{name}.depth.png")
        for name in ("agentview", "waist", "wrist")
    }
    for index, (rgb, depth) in enumerate(paths.values()):
        rgb.write_bytes(f"rgb-{index}".encode())
        depth.write_bytes(f"depth-{index}".encode())
    memory = AgentMemory()
    _record_pending_sam3_selection(
        memory,
        original_image_ref=str(paths["agentview"][0]),
    )
    memory.resolve_sam3_selection(
        result_id="sam3-run-selection",
        detection_id="detection_000",
        selection_source="main_agent_vlm",
    )
    _record_overwidth_grasp_policy(
        memory,
        backend="anygrasp",
        source_rgb=str(paths["agentview"][0]),
        camera_frame_id="agentview",
    )
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "tool_calls": [
                    {
                        "name": "sam3",
                        "status": "executed",
                        "result": {
                            "success": True,
                            "details": {
                                "parameters": {
                                    "image": str(paths["waist"][0]),
                                    "prompt": "alphabet soup",
                                },
                                "outputs": {
                                    "result_id": "sam3-waist-empty",
                                    "prompt": "alphabet soup",
                                    "source_image": str(paths["waist"][0]),
                                    "detections": [],
                                },
                            },
                        },
                    }
                ]
            },
        )
    )
    observation = _rgbd_observation(
        task="pick alphabet soup",
        views=[(name, *view) for name, view in paths.items()],
        with_extrinsics=True,
    )
    tools = _tools_with_handlers(
        "sam3",
        "grasp_pose_estimate",
        "ik_preview_check",
        "obstacle_avoidance",
        "move_to",
    )
    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=tools,
        skills=build_default_skill_registry(),
    )

    attempts = context["grasp_candidate_policy"]["fallback_attempts"]
    assert attempts[-1]["outcome"] == "segmentation_no_detection"
    fallback = context["grasp_estimation_fallback_obligation"]
    assert fallback["stage"] == "wrist_refinement_ik"
    hover_pose = fallback["required_parameters"]["target_pose"]
    assert hover_pose["grasp_stage"] == "grasp_estimation_refinement_hover"
    assert hover_pose["xyz"] == pytest.approx([0.1, -0.2, -0.05])
    decision = ToolCallingPlanner(StaticPlannerBackend([])).plan(
        observation,
        memory=memory,
        tools=tools,
        skills=build_default_skill_registry(),
    )
    assert decision.action == "ik_preview_check"
    assert decision.parameters == fallback["required_parameters"]

    def record_call(name: str, parameters: dict[str, object], outputs: dict[str, object]) -> None:
        memory.add_action(
            EnvAction(
                action_type="tool_call",
                command={
                    "request": {
                        "kind": "tool_call",
                        "name": name,
                        "parameters": parameters,
                    },
                    "status": "executed",
                    "tool_calls": [
                        {
                            "name": name,
                            "status": "executed",
                            "result": {
                                "success": True,
                                "details": {
                                    "parameters": parameters,
                                    "outputs": outputs,
                                },
                            },
                        }
                    ],
                },
            )
        )

    record_call("ik_preview_check", fallback["required_parameters"], {"feasible": True})
    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=tools,
        skills=build_default_skill_registry(),
    )
    fallback = context["grasp_estimation_fallback_obligation"]
    assert fallback["stage"] == "wrist_refinement_collision_check"

    record_call("obstacle_avoidance", fallback["required_parameters"], {"clear": True})
    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=tools,
        skills=build_default_skill_registry(),
    )
    fallback = context["grasp_estimation_fallback_obligation"]
    assert fallback["stage"] == "wrist_refinement_move"
    decision = ToolCallingPlanner(StaticPlannerBackend([])).plan(
        observation,
        memory=memory,
        tools=tools,
        skills=build_default_skill_registry(),
    )
    assert decision.action == "move_to"
    assert decision.parameters == fallback["required_parameters"]
    assert (
        memory.grasp_candidate_gate_error(
            tool_name="move_to",
            parameters=fallback["required_parameters"],
        )
        is None
    )

    record_call(
        "move_to",
        fallback["required_parameters"],
        {"motion_summary": {"reached_target": True}},
    )
    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=tools,
        skills=build_default_skill_registry(),
    )
    fallback = context["grasp_estimation_fallback_obligation"]
    assert fallback["stage"] == "wrist_refinement_segmentation"
    assert fallback["required_parameters"]["image"] == str(paths["wrist"][0])


def test_wrist_refinement_stops_when_ik_is_infeasible(tmp_path: Path) -> None:
    rgb = tmp_path / "agentview.rgb.png"
    depth = tmp_path / "agentview.depth.png"
    wrist_rgb = tmp_path / "wrist.rgb.png"
    wrist_depth = tmp_path / "wrist.depth.png"
    for path in (rgb, depth, wrist_rgb, wrist_depth):
        path.write_bytes(path.name.encode())
    memory = AgentMemory()
    _record_pending_sam3_selection(memory, original_image_ref=str(rgb))
    memory.resolve_sam3_selection(
        result_id="sam3-run-selection",
        detection_id="detection_000",
        selection_source="main_agent_vlm",
    )
    _record_overwidth_grasp_policy(
        memory,
        backend="anygrasp",
        source_rgb=str(rgb),
        camera_frame_id="agentview",
    )
    observation = _rgbd_observation(
        task="pick alphabet soup",
        views=[
            ("agentview", rgb, depth),
            ("wrist", wrist_rgb, wrist_depth),
        ],
        with_extrinsics=True,
    )
    tools = _tools_with_handlers(
        "sam3",
        "grasp_pose_estimate",
        "ik_preview_check",
        "obstacle_avoidance",
        "move_to",
    )
    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=tools,
        skills=build_default_skill_registry(),
    )
    fallback = context["grasp_estimation_fallback_obligation"]
    assert fallback["stage"] == "wrist_refinement_ik"
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "ik_preview_check",
                    "parameters": fallback["required_parameters"],
                },
                "status": "executed",
                "tool_calls": [
                    {
                        "name": "ik_preview_check",
                        "status": "executed",
                        "result": {
                            "success": True,
                            "details": {"outputs": {"feasible": False}},
                        },
                    }
                ],
            },
        )
    )

    recovery = memory.grasp_estimation_recovery()
    assert recovery["status"] == "blocked"
    assert recovery["last_failure"]["hard_rejection"] == "ik_unreachable"
    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=tools,
        skills=build_default_skill_registry(),
    )
    assert context["grasp_estimation_fallback_obligation"]["status"] == "blocked"


def test_overwidth_grasps_switch_backend_after_all_camera_views(tmp_path: Path) -> None:
    agent_rgb = tmp_path / "agentview.rgb.png"
    agent_depth = tmp_path / "agentview.depth.png"
    waist_rgb = tmp_path / "waist.rgb.png"
    waist_depth = tmp_path / "waist.depth.png"
    for index, path in enumerate((agent_rgb, agent_depth, waist_rgb, waist_depth)):
        path.write_bytes(f"artifact-{index}".encode())
    memory = AgentMemory()
    _record_pending_sam3_selection(memory, original_image_ref=str(agent_rgb))
    memory.resolve_sam3_selection(
        result_id="sam3-run-selection",
        detection_id="detection_000",
        selection_source="main_agent_vlm",
    )
    _record_overwidth_grasp_policy(
        memory,
        backend="anygrasp",
        source_rgb=str(agent_rgb),
        camera_frame_id="agentview",
    )
    policy = memory.grasp_candidate_policy()
    assert policy is not None
    policy["fallback_attempts"].append(
        {
            "backend": "anygrasp",
            "source_rgb": str(waist_rgb),
            "outcome": "segmentation_no_detection",
            "raw_candidate_count": 0,
            "width_limit_m": 0.08,
        }
    )
    memory.save_fact("grasp_candidate_policy", policy, source="test")
    observation = _rgbd_observation(
        task="pick alphabet soup",
        views=[
            ("agentview", agent_rgb, agent_depth),
            ("waist", waist_rgb, waist_depth),
        ],
    )
    tools = _tools_with_handlers("sam3", "grasp_pose_estimate")
    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=tools,
        skills=build_default_skill_registry(),
    )

    fallback = context["grasp_estimation_fallback_obligation"]
    assert fallback["stage"] == "alternate_backend"
    assert fallback["excluded_backends"] == ["anygrasp"]
    assert fallback["required_parameters"]["hints"]["excluded_backends"] == [
        "anygrasp"
    ]
    decision = ToolCallingPlanner(StaticPlannerBackend([])).plan(
        observation,
        memory=memory,
        tools=tools,
        skills=build_default_skill_registry(),
    )
    assert decision.action == "grasp_pose_estimate"
    assert decision.parameters == fallback["required_parameters"]


def test_grasp_width_filter_uses_session_calibration_profile(tmp_path: Path) -> None:
    profile = tmp_path / "wide-gripper.json"
    profile.write_text(
        json.dumps(
            {
                "calibration_id": "test-wide-gripper",
                "max_gripper_width_m": 0.11,
            }
        ),
        encoding="utf-8",
    )
    memory = AgentMemory()
    memory.start_session(
        task="pick object",
        metadata={"workspace": {"grasp_profile_path": str(profile)}},
    )

    _record_overwidth_grasp_policy(
        memory,
        backend="anygrasp",
        source_rgb="tmp/rgb.png",
        camera_frame_id="wrist",
    )

    policy = memory.grasp_candidate_policy()
    assert policy["status"] == "active"
    assert policy["physical_width_limit_m"] == pytest.approx(0.11)
    assert policy["grasp_calibration_id"] == "test-wide-gripper"
    assert policy["active_candidate"]["width"] == pytest.approx(0.09)
    assert [item["candidate_id"] for item in policy["rejected_candidates"]] == [
        "anygrasp-overwidth-1"
    ]
    assert "0.1100 m" in policy["rejected_candidates"][0]["reason"]


def test_all_overwidth_backends_activate_highest_scoring_final_candidate(
    tmp_path: Path,
) -> None:
    rgb = tmp_path / "agentview.rgb.png"
    depth = tmp_path / "agentview.depth.png"
    profile = tmp_path / "wide-gripper.json"
    rgb.write_bytes(b"rgb")
    depth.write_bytes(b"depth")
    profile.write_text(
        json.dumps(
            {
                "calibration_id": "test-wide-gripper",
                "max_gripper_width_m": 0.11,
            }
        ),
        encoding="utf-8",
    )
    memory = AgentMemory()
    memory.start_session(
        task="pick alphabet soup",
        metadata={"workspace": {"grasp_profile_path": str(profile)}},
    )
    _record_pending_sam3_selection(memory, original_image_ref=str(rgb))
    memory.resolve_sam3_selection(
        result_id="sam3-run-selection",
        detection_id="detection_000",
        selection_source="main_agent_vlm",
    )
    for backend in ("anygrasp", "contact_graspnet", "graspgenx"):
        _record_overwidth_grasp_policy(
            memory,
            backend=backend,
            source_rgb=str(rgb),
            camera_frame_id="agentview",
            widths=(0.12, 0.14),
        )
    recovery = memory.grasp_estimation_recovery()
    for entry in recovery["fallback_candidates"]:
        if entry["source_backend"] == "contact_graspnet":
            entry["candidate"]["score"] = 0.99
    memory.save_fact("grasp_estimation_recovery", recovery, source="test")
    observation = _rgbd_observation(
        task="pick alphabet soup",
        views=[("agentview", rgb, depth)],
        with_extrinsics=True,
    )
    tools = build_default_tool_registry()
    runtime = OpenEtaAgentRuntime(
        planner=ToolCallingPlanner(StaticPlannerBackend([])),
        memory=memory,
        tools=tools,
        skills=build_default_skill_registry(),
    )
    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=tools,
        skills=build_default_skill_registry(),
    )

    fallback = context["grasp_estimation_fallback_obligation"]
    assert fallback["status"] == "required"
    assert fallback["stage"] == "final_candidate_activation"
    assert fallback["required_tool"] == "activate_final_grasp_candidate"
    assert fallback["excluded_backends"] == [
        "anygrasp",
        "contact_graspnet",
        "graspgenx",
    ]
    action = runtime.act(observation)
    assert action.command["request"]["name"] == "activate_final_grasp_candidate"
    policy = memory.grasp_candidate_policy()
    assert policy["status"] == "active"
    assert policy["final_refinable_fallback"] is True
    candidate = policy["active_candidate"]
    assert candidate["original_candidate_id"] == "contact_graspnet-overwidth-0"
    assert candidate["width"] == pytest.approx(0.11)
    assert candidate["estimated_width_m"] == pytest.approx(0.12)
    assert candidate["max_gripper_width_m"] == pytest.approx(0.11)
    assert candidate["grasp_calibration_id"] == "test-wide-gripper"
    assert candidate["width_clamped_to_physical_limit"] is True
    assert policy["physical_width_limit_m"] == pytest.approx(0.11)
    assert policy["grasp_calibration_profile_path"] == str(profile)
    assert memory.grasp_estimation_recovery()["status"] == "final_candidate_activated"
    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=tools,
        skills=build_default_skill_registry(),
    )
    compile_obligation = context["grasp_compile_obligation"]
    assert compile_obligation["required_tool"] == "compile_grasp_seed"
    assert compile_obligation["required_parameters"]["camera_pose"]["id"] == candidate["id"]
    assert (
        compile_obligation["required_parameters"]["camera_pose"]["final_refinable_fallback"]
        is True
    )

    failed_parameters = {
        "target_pose": {
            "frame": "world",
            "source_grasp_id": candidate["id"],
            "xyz": [0.1, 0.2, 0.3],
        }
    }
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "move_to",
                    "parameters": failed_parameters,
                },
                "status": "failed",
                "tool_calls": [
                    {
                        "name": "move_to",
                        "status": "failed",
                        "result": {
                            "success": False,
                            "content": "IK unreachable",
                            "details": {
                                "diagnostics": [
                                    {
                                        "code": "grasp_candidate_unreachable",
                                        "candidate_rejection": True,
                                    }
                                ]
                            },
                        },
                    }
                ],
            },
        )
    )
    assert memory.grasp_candidate_policy()["status"] == "exhausted"
    assert memory.grasp_estimation_recovery()["status"] == "final_candidate_activated"
    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=tools,
        skills=build_default_skill_registry(),
    )
    assert context["grasp_estimation_fallback_obligation"] is None


def test_enhanced_grasp_requires_matching_sensor_safety_evidence(
    tmp_path: Path,
) -> None:
    safety_depth = tmp_path / "safety.png"
    safety_cloud = tmp_path / "safety.npz"
    report = tmp_path / "report.json"
    for path in (safety_depth, safety_cloud, report):
        path.write_bytes(b"fixture")
    candidate = {"id": "gpe-1", "frame": "camera"}
    policy = {
        "status": "active",
        "source_tool": "grasp_pose_estimate",
        "active_candidate": candidate,
    }
    retained = {
        "source": {
            "camera_frame_id": "wrist",
            "requires_sensor_safety_check": True,
            "depth_enhancement": {
                "safety_depth_png": str(safety_depth),
                "safety_point_cloud_npz": str(safety_cloud),
                "report_path": str(report),
            },
        }
    }
    observation = EnvObservation(
        task="pick",
        cameras=[
            CameraFrame(
                frame_id="wrist",
                rgb=[[[0, 0, 0]]],
                depth=[[1.0]],
                extrinsics={"camera_to_world": [1.0] * 16},
            )
        ],
        robot=RobotState(),
    )

    obligation = _grasp_sensor_safety_obligation(
        grasp_policy=policy,
        retained=retained,
        execution=None,
        scene_epoch=3,
        working_artifacts={},
    )
    assert obligation is not None
    assert obligation["required_tool"] == "obstacle_avoidance"
    assert (
        _grasp_compile_obligation(
            observation,
            grasp_policy=policy,
            retained=retained,
            execution=None,
            scene_epoch=3,
            asset_reference=None,
            working_artifacts={},
        )
        is None
    )

    request = obligation["required_parameters"]["path"]
    evidence = {
        "value": {
            "type": "enhanced_grasp_sensor_safety_check",
            **request,
            "clear": True,
        }
    }
    assert (
        _grasp_sensor_safety_obligation(
            grasp_policy=policy,
            retained=retained,
            execution=None,
            scene_epoch=3,
            working_artifacts={"safety": evidence},
        )
        is None
    )
    assert (
        _grasp_compile_obligation(
            observation,
            grasp_policy=policy,
            retained=retained,
            execution=None,
            scene_epoch=3,
            asset_reference=None,
            working_artifacts={"safety": evidence},
        )
        is not None
    )


def test_camera_frame_grasp_without_extrinsics_forces_observation_refresh() -> None:
    memory = AgentMemory()
    _record_anygrasp_candidate_policy(
        memory,
        source_tool="grasp_pose_estimate",
        camera_frame_id="agentview",
    )
    observation = EnvObservation(
        task="pick alphabet soup",
        cameras=[
            CameraFrame(
                frame_id="agentview",
                rgb=[[[0, 0, 0]]],
                intrinsics={"fx": 100.0, "fy": 100.0, "cx": 0.5, "cy": 0.5},
            )
        ],
        robot=RobotState(),
    )
    tools = _tools_with_handlers("observe")
    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=tools,
        skills=build_default_skill_registry(),
    )

    assert context["grasp_calibration_refresh_obligation"] == {
        "schema_version": "openeta.grasp_calibration_refresh_obligation.v1",
        "required_tool": "observe",
        "required_parameters": {},
        "camera_frame_id": "agentview",
        "candidate_id": "grasp_000",
        "reason": "matching_camera_extrinsics_missing",
    }

    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {"kind": "response", "name": "ask_human", "parameters": {"message": "unused"}}
        )
    )
    decision = planner.plan(
        observation,
        memory=memory,
        tools=tools,
        skills=build_default_skill_registry(),
    )

    assert decision.action == "observe"
    assert decision.parameters == {}
    assert decision.metadata["execution_model"] == "host_obligation_dispatch"
    assert decision.metadata["host_obligation"]["stage"] == "grasp_calibration_refresh"


def test_camera_frame_grasp_with_extrinsics_does_not_request_refresh() -> None:
    memory = AgentMemory()
    _record_anygrasp_candidate_policy(
        memory,
        source_tool="grasp_pose_estimate",
        camera_frame_id="agentview",
    )
    observation = EnvObservation(
        task="pick alphabet soup",
        cameras=[
            CameraFrame(
                frame_id="agentview",
                rgb=[[[0, 0, 0]]],
                intrinsics={"fx": 100.0, "fy": 100.0, "cx": 0.5, "cy": 0.5},
                extrinsics={
                    "pos": [0.0, 0.0, 0.0],
                    "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                },
            )
        ],
        robot=RobotState(),
    )

    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=_tools_with_handlers("observe"),
        skills=build_default_skill_registry(),
    )

    assert context["grasp_calibration_refresh_obligation"] is None
    assert context["current_camera_calibrations"] == [
        {
            "frame_id": "agentview",
            "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 0.5, "cy": 0.5},
            "extrinsics": {
                "pos": [0.0, 0.0, 0.0],
                "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            },
        }
    ]


def test_grasp_compile_canonicalizes_host_numeric_state_without_changing_semantics() -> None:
    memory = AgentMemory()
    _record_anygrasp_candidate_policy(
        memory,
        source_tool="grasp_pose_estimate",
        camera_frame_id="agentview",
    )
    active = memory.grasp_candidate_policy()["active_candidate"]
    exact_extrinsics = {
        "pos": [0.8965773716836134, 5.216182733499864e-07, 0.65],
        "mat": [
            -1.7233905013069872e-06,
            -0.5287697435529835,
            0.8487653140297038,
            0.9999999999985034,
            -7.823149652530503e-07,
            1.5430955956352577e-06,
            -1.5194045527300304e-07,
            -0.848765314031093,
            0.5287697435535403,
        ],
    }
    observation = EnvObservation(
        task="pick alphabet soup",
        cameras=[
            CameraFrame(
                frame_id="agentview",
                rgb=[[[0, 0, 0]]],
                extrinsics=exact_extrinsics,
            )
        ],
        robot=RobotState(),
    )
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {
                "kind": "tool_call",
                "name": "compile_grasp_seed",
                "parameters": {
                    "camera_pose": {**active, "translation_xyz": [9.0, 9.0, 9.0]},
                    "camera_extrinsics": {
                        **exact_extrinsics,
                        "mat": [*exact_extrinsics["mat"][:6], -1.5194045527300304, *exact_extrinsics["mat"][7:]],
                    },
                    "camera_frame_id": "stale-camera",
                    "scene_epoch": 7,
                    "target_geometry_family": "upright_can",
                },
            }
        )
    )

    decision = planner.plan(
        observation,
        memory=memory,
        tools=_tools_with_handlers("compile_grasp_seed"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "compile_grasp_seed"
    assert decision.parameters["camera_pose"] == active
    assert decision.parameters["camera_extrinsics"] == exact_extrinsics
    assert decision.parameters["camera_frame_id"] == "agentview"
    assert decision.parameters["scene_epoch"] == 0
    assert decision.parameters["target_geometry_family"] == "upright_can"
    canonicalized = {
        entry["field"] for entry in decision.metadata["host_parameter_canonicalizations"]
    }
    assert canonicalized == {
        "camera_pose",
        "camera_extrinsics",
        "camera_frame_id",
        "scene_epoch",
    }


def test_fallback_grasp_candidate_reuses_semantics_via_host_compile() -> None:
    memory = AgentMemory()
    _record_anygrasp_candidate_policy(
        memory,
        source_tool="grasp_pose_estimate",
        camera_frame_id="agentview",
    )
    policy = memory.grasp_candidate_policy()
    policy["compile_hints"] = {
        "target_geometry_family": "upright_can",
        "pregrasp_distance_m": 0.08,
    }
    memory.save_fact("grasp_candidate_policy", policy, source="test")
    exact_extrinsics = {
        "pos": [0.0, 0.0, 0.0],
        "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    }
    observation = EnvObservation(
        task="pick alphabet soup",
        cameras=[
            CameraFrame(
                frame_id="agentview",
                rgb=[[[0, 0, 0]]],
                extrinsics=exact_extrinsics,
            )
        ],
        robot=RobotState(),
    )
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {"kind": "response", "name": "ask_human", "parameters": {"message": "unused"}}
        )
    )

    decision = planner.plan(
        observation,
        memory=memory,
        tools=_tools_with_handlers("compile_grasp_seed"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "compile_grasp_seed"
    assert decision.parameters == {
        "camera_pose": policy["active_candidate"],
        "camera_extrinsics": exact_extrinsics,
        "camera_frame_id": "agentview",
        "scene_epoch": 0,
        "target_geometry_family": "upright_can",
        "pregrasp_distance_m": 0.08,
    }
    assert decision.metadata["execution_model"] == "host_obligation_dispatch"
    assert decision.metadata["host_obligation"]["stage"] == "grasp_compile"


def test_verified_reference_geometry_drives_initial_host_grasp_compile() -> None:
    memory = AgentMemory()
    _record_anygrasp_candidate_policy(
        memory,
        source_tool="grasp_pose_estimate",
        camera_frame_id="agentview",
    )
    memory.save_fact(
        "target_asset_reference",
        {
            "target_object": "alphabet_soup",
            "exact_instance_verification": {
                "decision": "match",
                "grasp_geometry_family": "upright_can",
            },
        },
        source="test",
    )
    exact_extrinsics = {
        "pos": [0.0, 0.0, 0.0],
        "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    }
    observation = EnvObservation(
        task="pick alphabet soup",
        cameras=[
            CameraFrame(
                frame_id="agentview",
                rgb=[[[0, 0, 0]]],
                extrinsics=exact_extrinsics,
            )
        ],
        robot=RobotState(),
    )
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {"kind": "response", "name": "ask_human", "parameters": {"message": "unused"}}
        )
    )

    decision = planner.plan(
        observation,
        memory=memory,
        tools=_tools_with_handlers("compile_grasp_seed"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "compile_grasp_seed"
    assert decision.parameters["target_geometry_family"] == "upright_can"
    assert decision.parameters["camera_extrinsics"] == exact_extrinsics
    assert decision.metadata["execution_model"] == "host_obligation_dispatch"


def test_grasp_open_precedes_stale_reference_recovery_obligation() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick up alphabet soup and place it into basket")
    memory.save_fact(
        "grasp_execution",
        {
            "schema_version": "openeta.grasp_execution.v1",
            "status": "required",
            "stage": "open",
            "candidate_id": "grasp-1",
            "required_action": {
                "name": "gripper_control",
                "parameters": {"position": 1},
            },
        },
        source="test",
    )
    memory.save_fact(
        "sam3_no_detection",
        {
            "result_id": "empty-wrist",
            "source_image": "tmp/wrist.rgb.png",
            "target_prompt": "alphabet soup",
        },
        source="test",
    )
    observation = EnvObservation(
        task="pick up alphabet soup and place it into basket",
        cameras=[CameraFrame(frame_id="wrist", rgb=[[[0, 0, 0]]])],
        robot=RobotState(),
        metadata={
            "env_id": "openeta/libero_libero_object_task0-v0",
            "image_artifacts": [
                {
                    "kind": "rgb",
                    "frame_id": "wrist",
                    "path": "tmp/wrist.rgb.png",
                }
            ],
        },
    )
    tools = _tools_with_handlers("gripper_control", "retrieve_asset_reference")
    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=tools,
        skills=build_default_skill_registry(),
    )
    assert context["target_reference_obligation"]["required_tool"] == (
        "retrieve_asset_reference"
    )
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {"kind": "response", "name": "ask_human", "parameters": {"message": "unused"}}
        )
    )

    decision = planner.plan(
        observation,
        memory=memory,
        tools=tools,
        skills=build_default_skill_registry(),
    )

    assert decision.action == "gripper_control"
    assert decision.parameters == {"position": 1}
    assert decision.metadata["host_obligation"]["stage"] == "open"


def test_targeted_grasp_obligation_adapts_fixed_anygrasp_depth_cutoff(
    tmp_path: Path,
) -> None:
    selected_rgb = tmp_path / "selected.rgb.png"
    current_rgb = tmp_path / "current.rgb.png"
    depth = tmp_path / "current.depth.png"
    mask = tmp_path / "mask.png"
    selected_rgb.write_bytes(b"same-scene")
    current_rgb.write_bytes(b"same-scene")
    Image.new("I;16", (4, 4), 1200).save(depth)
    Image.new("L", (4, 4), 255).save(mask)
    memory = AgentMemory()
    memory.save_fact(
        "pending_sam3_selection",
        {
            "result_id": "sam-depth-cutoff",
            "source_image": str(selected_rgb),
            "candidates": [
                {
                    "id": "detection_000",
                    "rank": 0,
                    "score": 0.97,
                    "mask_ref": str(mask),
                }
            ],
        },
        source="sam3",
    )
    memory.resolve_sam3_selection(
        result_id="sam-depth-cutoff",
        detection_id="detection_000",
        selection_source="host",
    )
    observation = EnvObservation(
        task="pick alphabet soup",
        cameras=[
            CameraFrame(
                frame_id="agentview",
                rgb=[[[0, 0, 0]]],
                depth=[[1.2]],
                intrinsics={
                    "fx": 100.0,
                    "fy": 100.0,
                    "cx": 0.5,
                    "cy": 0.5,
                    "scale": 1000,
                },
            )
        ],
        robot=RobotState(),
        metadata={
            "image_artifacts": [
                {"kind": "rgb", "frame_id": "agentview", "path": str(current_rgb)},
                {"kind": "depth", "frame_id": "agentview", "path": str(depth)},
            ]
        },
    )

    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=_tools_with_handlers("grasp_pose_estimate"),
        skills=build_default_skill_registry(),
    )

    required = context["targeted_grasp_obligation"]["required_parameters"]
    assert required["depth"] == str(depth)
    assert required["object_mask"]["mask_ref"] == str(mask)
    assert required["intrinsics"]["scale"] == 1000
    assert required["hints"]["depth_cutoff_factor"] == pytest.approx(1.333333)


def test_fresh_selection_restarts_an_exhausted_anygrasp_queue(tmp_path: Path) -> None:
    rgb = tmp_path / "agentview.rgb.png"
    depth = tmp_path / "agentview.depth.png"
    mask = tmp_path / "mask.png"
    rgb.write_bytes(b"fresh-scene")
    Image.new("I;16", (4, 4), 800).save(depth)
    Image.new("L", (4, 4), 255).save(mask)
    memory = AgentMemory()
    _record_pending_sam3_selection(memory, original_image_ref=str(rgb))
    memory.resolve_sam3_selection(
        result_id="sam3-run-selection",
        detection_id="detection_000",
        selection_source="host",
    )
    _record_anygrasp_candidate_policy(memory)
    exhausted = memory.anygrasp_candidate_policy()
    exhausted.update(
        {
            "status": "exhausted",
            "active_candidate": None,
            "active_rank": None,
        }
    )
    memory.save_fact("anygrasp_candidate_policy", exhausted, source="test")
    memory.save_fact(
        "pending_sam3_selection",
        {
            "result_id": "sam3-fresh-selection",
            "source_image": str(rgb),
            "candidates": [
                {
                    "id": "detection_000",
                    "rank": 0,
                    "score": 0.98,
                    "mask_ref": str(mask),
                }
            ],
        },
        source="sam3",
    )
    memory.resolve_sam3_selection(
        result_id="sam3-fresh-selection",
        detection_id="detection_000",
        selection_source="host",
    )
    observation = EnvObservation(
        task="pick alphabet soup",
        cameras=[
            CameraFrame(
                frame_id="agentview",
                rgb=[[[0, 0, 0]]],
                depth=[[0.8]],
                intrinsics={
                    "fx": 100.0,
                    "fy": 100.0,
                    "cx": 0.5,
                    "cy": 0.5,
                    "scale": 1000,
                },
            )
        ],
        robot=RobotState(),
        metadata={
            "image_artifacts": [
                {"kind": "rgb", "frame_id": "agentview", "path": str(rgb)},
                {"kind": "depth", "frame_id": "agentview", "path": str(depth)},
            ]
        },
    )

    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=_tools_with_handlers("grasp_pose_estimate"),
        skills=build_default_skill_registry(),
    )

    obligation = context["targeted_grasp_obligation"]
    assert obligation["sam3_result_id"] == "sam3-fresh-selection"
    assert obligation["required_parameters"]["object_mask"]["mask_ref"] == str(mask)


@pytest.mark.parametrize(
    ("stage", "position"),
    [("open", 1), ("close", 0)],
)
def test_exact_gripper_grasp_stages_use_host_dispatch(stage: str, position: int) -> None:
    memory = AgentMemory()
    memory.save_fact(
        "grasp_execution",
        {
            "schema_version": "openeta.grasp_execution.v1",
            "status": "required",
            "stage": stage,
            "candidate_id": "grasp_003",
            "required_action": {
                "name": "gripper_control",
                "parameters": {"position": position},
            },
        },
        source="test",
    )
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {"kind": "response", "name": "talk", "parameters": {"message": "unused"}}
        )
    )

    decision = planner.plan(
        _observation(),
        memory=memory,
        tools=_tools_with_handlers("gripper_control"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "gripper_control"
    assert decision.parameters == {"position": position}
    assert decision.metadata["execution_model"] == "host_obligation_dispatch"
    assert decision.metadata["host_obligation"]["stage"] == stage


@pytest.mark.parametrize("stage", ["hover", "align_move"])
def test_grasp_motion_obligation_precedes_stale_target_reference(stage: str) -> None:
    required = {
        "target_pose": {
            "frame": "world",
            "xyz": [0.1, 0.2, 0.3],
            "grasp_stage": stage,
        }
    }
    decision = _host_obligation_decision(
        {
            "grasp_execution": {
                "schema_version": "openeta.grasp_execution.v1",
                "status": "required",
                "stage": stage,
                "required_action": {"name": "move_to", "parameters": required},
            },
            "target_reference_obligation": {
                "schema_version": "openeta.target_reference_obligation.v1",
                "required_tool": "retrieve_asset_reference",
                "required_parameters": {
                    "environment": "libero",
                    "target_object": "alphabet soup",
                    "scene_image": "stale-wrist.png",
                },
            },
        },
        tools=_tools_with_handlers("move_to", "retrieve_asset_reference"),
    )

    assert decision is not None
    assert decision.action == "move_to"
    assert decision.parameters == required
    assert decision.metadata["host_obligation"]["stage"] == stage


def test_pending_semantic_selection_precedes_grasp_hover() -> None:
    decision = _host_obligation_decision(
        {
            "selection_obligation": {
                "result_id": "wrist-sam3",
                "reference_verification": {
                    "decision": "match",
                    "grasp_geometry_family": "boxed_item",
                },
                "candidates": [
                    {"id": "detection_000", "rank": 0, "score": 0.97},
                    {"id": "detection_001", "rank": 1, "score": 0.40},
                ],
            },
            "grasp_execution": {
                "schema_version": "openeta.grasp_execution.v1",
                "status": "required",
                "stage": "hover",
                "required_action": {
                    "name": "move_to",
                    "parameters": {
                        "target_pose": {
                            "frame": "world",
                            "xyz": [0.1, 0.2, 0.3],
                            "grasp_stage": "hover",
                        }
                    },
                },
            },
        },
        tools=_tools_with_handlers("select_sam3_detection", "move_to"),
    )

    assert decision is not None
    assert decision.action == "select_sam3_detection"
    assert decision.parameters["sam3_result_id"] == "wrist-sam3"
    assert decision.parameters["target_geometry_family"] == "boxed_item"


def test_ambiguous_semantic_selection_blocks_host_hover_dispatch() -> None:
    decision = _host_obligation_decision(
        {
            "selection_obligation": {
                "result_id": "ambiguous-wrist-sam3",
                "candidates": [
                    {"id": "detection_000", "rank": 0, "score": 0.81},
                    {"id": "detection_001", "rank": 1, "score": 0.78},
                ],
            },
            "grasp_execution": {
                "schema_version": "openeta.grasp_execution.v1",
                "status": "required",
                "stage": "hover",
                "required_action": {
                    "name": "move_to",
                    "parameters": {
                        "target_pose": {
                            "frame": "world",
                            "xyz": [0.1, 0.2, 0.3],
                            "grasp_stage": "hover",
                        }
                    },
                },
            },
        },
        tools=_tools_with_handlers("select_sam3_detection", "move_to"),
    )

    assert decision is None


def test_fixed_lift_probe_uses_host_dispatch() -> None:
    required = {
        "target_pose": {
            "frame": "world",
            "xyz": [0.1, 0.2, 0.3],
            "source_grasp_id": "grasp_003",
            "grasp_stage": "lift_probe",
        }
    }
    memory = AgentMemory()
    memory.save_fact(
        "grasp_execution",
        {
            "schema_version": "openeta.grasp_execution.v1",
            "status": "required",
            "stage": "probe",
            "candidate_id": "grasp_003",
            "required_action": None,
        },
        source="test",
    )
    memory.save_fact(
        "grasp_lift_probe",
        {
            "schema_version": "openeta.grasp_lift_probe.v1",
            "status": "required",
            "candidate_id": "grasp_003",
            "required_parameters": required,
        },
        source="test",
    )
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {"kind": "response", "name": "talk", "parameters": {"message": "unused"}}
        )
    )

    decision = planner.plan(
        _observation(),
        memory=memory,
        tools=_tools_with_handlers("move_to"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "move_to"
    assert decision.parameters == required
    assert decision.metadata["execution_model"] == "host_obligation_dispatch"
    assert decision.metadata["host_obligation"]["stage"] == "probe"


def test_attachment_full_lift_uses_host_dispatch_for_independent_review() -> None:
    required = {
        "target_pose": {
            "frame": "world",
            "xyz": [0.1, 0.2, 0.4],
            "source_grasp_id": "grasp_003",
            "grasp_stage": "full_lift",
        }
    }
    memory = AgentMemory()
    memory.save_fact(
        "grasp_execution",
        {
            "schema_version": "openeta.grasp_execution.v1",
            "status": "required",
            "stage": "attachment",
            "candidate_id": "grasp_003",
            "required_action": None,
            "attachment_actions": {
                "pass": {"name": "move_to", "parameters": required},
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
            "candidate_id": "grasp_003",
        },
        source="test",
    )
    planner = ToolCallingPlanner(
        StaticPlannerBackend({"kind": "tool_call", "name": "observe", "parameters": {}})
    )

    decision = planner.plan(
        _observation(),
        memory=memory,
        tools=_tools_with_handlers("move_to", "observe"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "move_to"
    assert decision.parameters == required
    assert decision.metadata["execution_model"] == "host_obligation_dispatch"
    assert decision.metadata["host_obligation"]["stage"] == "attachment"

    empty_observation = _observation()
    empty_observation.robot.gripper_state = {"open": False, "openness": 0.02}
    decision = planner.plan(
        empty_observation,
        memory=memory,
        tools=_tools_with_handlers("move_to", "gripper_control"),
        skills=build_default_skill_registry(),
    )
    assert decision.action == "move_to"
    assert decision.parameters == required
    assert decision.metadata["host_obligation"]["stage"] == "attachment"

    ambiguous_observation = _observation()
    ambiguous_observation.robot.gripper_state = {"open": False, "openness": 0.06}
    decision = planner.plan(
        ambiguous_observation,
        memory=memory,
        tools=_tools_with_handlers("move_to", "gripper_control"),
        skills=build_default_skill_registry(),
    )
    assert decision.action == "move_to"
    assert decision.parameters == required
    assert decision.metadata["host_obligation"]["stage"] == "attachment"

    memory.save_fact(
        "attachment_gate",
        {
            "status": "resolved",
            "verdict": "PASS",
            "candidate_id": "grasp_003",
        },
        source="test",
    )
    decision = planner.plan(
        ambiguous_observation,
        memory=memory,
        tools=_tools_with_handlers("move_to", "gripper_control"),
        skills=build_default_skill_registry(),
    )
    assert decision.action == "move_to"
    assert decision.parameters == required

    memory.save_fact(
        "attachment_gate",
        {
            "status": "resolved",
            "verdict": "FAIL",
            "candidate_id": "grasp_003",
        },
        source="test",
    )
    decision = planner.plan(
        empty_observation,
        memory=memory,
        tools=_tools_with_handlers("move_to", "gripper_control"),
        skills=build_default_skill_registry(),
    )
    assert decision.action == "gripper_control"
    assert decision.parameters == {"position": 1}
    assert decision.metadata["host_obligation"]["stage"] == "attachment_recovery"

    memory.save_fact(
        "attachment_gate",
        {
            "status": "pending",
            "verdict": "UNKNOWN",
            "candidate_id": "grasp_003",
            "pass_action_completed": True,
        },
        source="test",
    )
    decision = planner.plan(
        ambiguous_observation,
        memory=memory,
        tools=_tools_with_handlers("move_to", "gripper_control"),
        skills=build_default_skill_registry(),
    )
    assert decision.action == "ask_human"
    assert decision.parameters["failure_code"] == "attachment_verification_unknown"
    assert decision.metadata["host_obligation"]["stage"] == "attachment_verification"


@pytest.mark.parametrize("stage", ["hover", "align_move"])
def test_host_generated_safe_grasp_motion_uses_host_dispatch(stage: str) -> None:
    memory = AgentMemory()
    required = {"target_pose": {"frame": "world", "xyz": [0.1, 0.2, 0.3]}}
    memory.save_fact(
        "grasp_execution",
        {
            "schema_version": "openeta.grasp_execution.v1",
            "status": "required",
            "stage": stage,
            "candidate_id": "grasp_003",
            "required_action": {
                "name": "move_to",
                "parameters": required,
            },
        },
        source="test",
    )
    planner = ToolCallingPlanner(
        StaticPlannerBackend({"kind": "tool_call", "name": "observe", "parameters": {}})
    )

    decision = planner.plan(
        _observation(),
        memory=memory,
        tools=_tools_with_handlers("move_to", "observe"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "move_to"
    assert decision.parameters == required
    assert decision.metadata["execution_model"] == "host_obligation_dispatch"
    assert decision.metadata["host_obligation"]["stage"] == stage


def test_adjustable_contact_descend_remains_model_planned() -> None:
    memory = AgentMemory()
    memory.save_fact(
        "grasp_execution",
        {
            "schema_version": "openeta.grasp_execution.v1",
            "status": "required",
            "stage": "descend",
            "candidate_id": "grasp_003",
            "required_action": {
                "name": "move_to",
                "parameters": {"target_pose": {"frame": "world", "xyz": [0.1, 0.2, 0.3]}},
            },
        },
        source="test",
    )
    planner = ToolCallingPlanner(
        StaticPlannerBackend({"kind": "tool_call", "name": "observe", "parameters": {}})
    )

    decision = planner.plan(
        _observation(),
        memory=memory,
        tools=_tools_with_handlers("move_to", "observe"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "observe"
    assert decision.metadata["execution_model"] == "closed_loop_tool_calling"


@pytest.mark.parametrize("failure_reason", ["empty_target_mask"])
def test_failed_anygrasp_mask_invalidates_sam3_selection_before_retry(
    failure_reason: str,
) -> None:
    memory = AgentMemory()
    _record_pending_sam3_selection(memory, original_image_ref="tmp/rgb.png")
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
                    "name": "anygrasp",
                    "parameters": {"mode": "targeted"},
                },
                "status": "failed",
                "tool_calls": [
                    {
                        "name": "anygrasp",
                        "status": "failed",
                        "result": {
                            "success": False,
                            "details": {
                                "outputs": {
                                    "reason": failure_reason,
                                    "source_rgb": "tmp/rgb.png",
                                }
                            },
                        },
                    }
                ],
            },
        )
    )

    assert memory.selected_sam3_detection() is None
    assert memory.sam3_no_detection()["reason"] == failure_reason
    assert any(event.event_type == "sam3_detection_invalidated" for event in memory.events)

    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            [
                {
                    "kind": "tool_call",
                    "name": "anygrasp",
                    "parameters": {
                        "mode": "targeted",
                        "rgb": "tmp/rgb.png",
                        "depth": "tmp/depth.png",
                        "target_mask": "tmp/mask_000.png",
                        "intrinsics": {
                            "fx": 1.0,
                            "fy": 1.0,
                            "cx": 0.5,
                            "cy": 0.5,
                            "scale": 1000.0,
                        },
                    },
                },
                {
                    "kind": "tool_call",
                    "name": "sam3",
                    "parameters": {"image": "tmp/rgb.png", "prompt": "alphabet soup"},
                },
            ]
        ),
        max_validation_retries=1,
    )
    decision = planner.plan(
        _observation(),
        memory=memory,
        tools=_tools_with_handlers("anygrasp", "sam3"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "sam3"
    first_errors = decision.metadata["validation_attempt_history"][0]["validation_errors"]
    assert any("fresh select_sam3_detection" in error for error in first_errors)


@pytest.mark.parametrize(
    "reported_reason",
    ["all_backends_failed", "insufficient_object_points"],
)
def test_unified_no_candidates_retries_same_verified_mask_once_with_dense_sampling(
    tmp_path: Path,
    reported_reason: str,
) -> None:
    selected_rgb = tmp_path / "selected" / "agentview.rgb.png"
    current_rgb = tmp_path / "current" / "agentview.rgb.png"
    current_depth = tmp_path / "current" / "agentview.depth.png"
    selected_rgb.parent.mkdir()
    current_rgb.parent.mkdir()
    selected_rgb.write_bytes(b"same-agentview-scene")
    current_rgb.write_bytes(b"same-agentview-scene")
    current_depth.write_bytes(b"depth")
    memory = AgentMemory()
    _record_pending_sam3_selection(memory, original_image_ref=str(selected_rgb))
    memory.resolve_sam3_selection(
        result_id="sam3-run-selection",
        detection_id="detection_000",
        selection_source="main_agent_vlm",
    )
    selected = memory.selected_sam3_detection()
    selected["bbox_xyxy"] = [205, 227, 225, 253]
    memory.save_fact("selected_sam3_detection", selected, source="test")
    failed_call = {
        "name": "grasp_pose_estimate",
        "status": "failed",
        "parameters": {"mode": "targeted", "hints": {}},
        "result": {
            "success": False,
            "details": {
                "outputs": {
                    "reason": reported_reason,
                    "backend_attempts": [
                        {
                            "backend": "anygrasp",
                            "status": "failed",
                            "reason": "no_grasp_candidates",
                        },
                        {
                            "backend": "contact_graspnet",
                            "status": "failed",
                            "reason": "no_grasp_candidates",
                        },
                    ],
                }
            },
        },
    }
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "grasp_pose_estimate",
                    "parameters": {"mode": "targeted", "hints": {}},
                },
                "status": "failed",
                "tool_calls": [failed_call],
            },
        )
    )
    assert memory.selected_sam3_detection()["dense_grasp_retry_required"] is True
    assert memory.sam3_no_detection() is None
    observation = EnvObservation(
        task="pick alphabet soup",
        cameras=[
            CameraFrame(
                frame_id="agentview",
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
        robot=RobotState(),
        metadata={
            "image_artifacts": [
                {"kind": "rgb", "frame_id": "agentview", "path": str(current_rgb)},
                {"kind": "depth", "frame_id": "agentview", "path": str(current_depth)},
            ]
        },
    )
    planner = ToolCallingPlanner(StaticPlannerBackend([]))

    decision = planner.plan(
        observation,
        memory=memory,
        tools=_tools_with_handlers("grasp_pose_estimate"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "grasp_pose_estimate"
    assert decision.parameters["hints"]["dense_sampling"] is True

    dense_parameters = dict(decision.parameters)
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "grasp_pose_estimate",
                    "parameters": dense_parameters,
                },
                "status": "failed",
                "tool_calls": [{**failed_call, "parameters": dense_parameters}],
            },
        )
    )
    assert memory.selected_sam3_detection() is None
    assert memory.sam3_no_detection()["reason"] == "no_grasp_candidates"
    assert memory.sam3_no_detection()["segmentation_mode"] == "point_prompt"
    assert memory.sam3_no_detection()["bbox_xyxy"] == [205, 227, 225, 253]


def test_unified_grasp_backend_failure_retries_once_then_opens_circuit(
    tmp_path: Path,
) -> None:
    selected_rgb = tmp_path / "selected" / "agentview.rgb.png"
    current_rgb = tmp_path / "current" / "agentview.rgb.png"
    current_depth = tmp_path / "current" / "agentview.depth.png"
    selected_rgb.parent.mkdir()
    current_rgb.parent.mkdir()
    selected_rgb.write_bytes(b"same-agentview-scene")
    current_rgb.write_bytes(b"same-agentview-scene")
    current_depth.write_bytes(b"depth")
    memory = AgentMemory()
    _record_pending_sam3_selection(memory, original_image_ref=str(selected_rgb))
    memory.resolve_sam3_selection(
        result_id="sam3-run-selection",
        detection_id="detection_000",
        selection_source="main_agent_vlm",
    )
    failed_call = {
        "name": "grasp_pose_estimate",
        "status": "failed",
        "parameters": {"mode": "targeted", "hints": {}},
        "result": {
            "success": False,
            "details": {
                "outputs": {
                    "reason": "all_backends_failed",
                    "backend_attempts": [
                        {
                            "backend": "anygrasp",
                            "status": "failed",
                            "reason": "model_inference_failed",
                        },
                        {
                            "backend": "contact_graspnet",
                            "status": "failed",
                            "reason": "mcp_call_failed",
                        },
                    ],
                }
            },
        },
    }

    def record_failure() -> None:
        memory.add_action(
            EnvAction(
                action_type="tool_call",
                command={
                    "request": {
                        "kind": "tool_call",
                        "name": "grasp_pose_estimate",
                        "parameters": {"mode": "targeted", "hints": {}},
                    },
                    "status": "failed",
                    "tool_calls": [failed_call],
                },
            )
        )

    observation = EnvObservation(
        task="pick alphabet soup",
        cameras=[
            CameraFrame(
                frame_id="agentview",
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
        robot=RobotState(),
        metadata={
            "image_artifacts": [
                {"kind": "rgb", "frame_id": "agentview", "path": str(current_rgb)},
                {"kind": "depth", "frame_id": "agentview", "path": str(current_depth)},
            ]
        },
    )

    record_failure()
    first = memory.selected_sam3_detection()["grasp_estimator_backend_failure"]
    assert first == {
        "reason": "model_inference_failed",
        "error_type": None,
        "attempt_count": 1,
        "max_attempts": 2,
        "status": "retry_required",
    }
    planner = ToolCallingPlanner(StaticPlannerBackend([]))
    decision = planner.plan(
        observation,
        memory=memory,
        tools=_tools_with_handlers("grasp_pose_estimate"),
        skills=build_default_skill_registry(),
    )
    assert decision.action == "grasp_pose_estimate"
    retry_parameters = dict(decision.parameters)

    record_failure()
    second = memory.selected_sam3_detection()["grasp_estimator_backend_failure"]
    assert second["attempt_count"] == 2
    assert second["status"] == "exhausted"
    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=_tools_with_handlers("grasp_pose_estimate"),
        skills=build_default_skill_registry(),
    )
    assert context["targeted_grasp_obligation"] is None

    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            [
                {
                    "kind": "tool_call",
                    "name": "grasp_pose_estimate",
                    "parameters": retry_parameters,
                },
                {
                    "kind": "response",
                    "name": "talk",
                    "parameters": {"message": "grasp backend unavailable"},
                },
            ]
        ),
        max_validation_retries=1,
    )
    decision = planner.plan(
        observation,
        memory=memory,
        tools=_tools_with_handlers("grasp_pose_estimate"),
        skills=build_default_skill_registry(),
    )
    assert decision.action == "talk"
    errors = decision.metadata["validation_attempt_history"][0]["validation_errors"]
    assert any("exhausted its bounded retry budget" in error for error in errors)


@pytest.mark.parametrize(
    ("segmentation_mode", "expects_retry"),
    [("point_prompt", True), ("roi_attention", False)],
)
def test_no_grasp_candidates_uses_at_most_one_same_scene_roi_retry(
    tmp_path: Path,
    segmentation_mode: str,
    expects_retry: bool,
) -> None:
    failed_scene = tmp_path / "previous" / "agentview.rgb.png"
    current_scene = tmp_path / "current" / "agentview.rgb.png"
    failed_scene.parent.mkdir()
    current_scene.parent.mkdir()
    failed_scene.write_bytes(b"same-static-scene")
    current_scene.write_bytes(b"same-static-scene")
    bbox_xyxy = [183.0, 299.0, 212.0, 332.0]
    memory = AgentMemory()
    memory.start_session(task="pick up cream cheese and place it into basket.")
    memory.save_fact(
        "sam3_no_detection",
        {
            "result_id": "failed-target-mask",
            "source_image": str(failed_scene),
            "target_prompt": "cream cheese",
            "reason": "no_grasp_candidates",
            "segmentation_mode": segmentation_mode,
        },
        source="anygrasp",
    )
    memory.save_fact(
        "target_asset_reference",
        {
            "environment": "libero",
            "target_object": "cream_cheese",
            "scene_image": str(failed_scene),
            "bbox_xyxy": bbox_xyxy,
        },
        source="retrieve_asset_reference",
    )
    observation = _observation()
    observation.task = "pick up cream cheese and place it into basket."
    observation.metadata = {
        "env_id": "openeta/libero_libero_object_task1-v0",
        "image_artifacts": [
            {
                "kind": "rgb",
                "frame_id": "agentview",
                "path": str(current_scene),
            }
        ],
    }
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {"kind": "response", "name": "talk", "parameters": {"message": "stop"}}
        )
    )

    decision = planner.plan(
        observation,
        memory=memory,
        tools=_tools_with_handlers("sam3", "retrieve_asset_reference"),
        skills=build_default_skill_registry(),
    )

    if not expects_retry:
        assert decision.action == "talk"
        return
    assert decision.action == "sam3"
    assert decision.parameters == {
        "image": str(current_scene),
        "prompt": "cream cheese",
        "roi_bbox_xyxy": bbox_xyxy,
    }
    assert decision.metadata["execution_model"] == "host_obligation_dispatch"
    assert decision.metadata["host_obligation"]["retry_mode"] == ("roi_after_no_grasp_candidates")


def test_sparse_point_mask_uses_selected_bbox_for_roi_retry(tmp_path: Path) -> None:
    failed_scene = tmp_path / "previous" / "agentview.rgb.png"
    current_scene = tmp_path / "current" / "agentview.rgb.png"
    failed_scene.parent.mkdir()
    current_scene.parent.mkdir()
    failed_scene.write_bytes(b"same-static-scene")
    current_scene.write_bytes(b"same-static-scene")
    bbox_xyxy = [205, 227, 225, 253]
    memory = AgentMemory()
    memory.start_session(task="pick up alphabet soup and place it into basket.")
    memory.save_fact(
        "sam3_no_detection",
        {
            "result_id": "sparse-point-mask",
            "source_image": str(failed_scene),
            "target_prompt": "alphabet soup",
            "reason": "no_grasp_candidates",
            "segmentation_mode": "point_prompt",
            "bbox_xyxy": bbox_xyxy,
        },
        source="grasp_pose_estimate",
    )
    observation = _observation()
    observation.task = "pick up alphabet soup and place it into basket."
    observation.metadata = {
        "env_id": "libero-env",
        "image_artifacts": [
            {"kind": "rgb", "frame_id": "agentview", "path": str(current_scene)}
        ],
    }
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {"kind": "response", "name": "talk", "parameters": {"message": "unused"}}
        )
    )

    decision = planner.plan(
        observation,
        memory=memory,
        tools=_tools_with_handlers("sam3"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "sam3"
    assert decision.parameters == {
        "image": str(current_scene),
        "prompt": "alphabet soup",
        "roi_bbox_xyxy": bbox_xyxy,
    }
    assert decision.metadata["host_obligation"]["retry_mode"] == (
        "roi_after_no_grasp_candidates"
    )


def test_placement_obligation_joins_receptacle_mask_to_frozen_grasp() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube and place it in basket")
    _record_anygrasp_candidate_policy(memory)
    _record_pending_sam3_selection(
        memory,
        original_image_ref="tmp/rgb.png",
    )
    memory.resolve_sam3_selection(
        result_id="sam3-run-selection",
        detection_id="detection_000",
        selection_source="main_agent_vlm",
    )
    pre_attachment_context = build_tool_context(
        observation=_observation(),
        memory=memory,
        tools=_tools_with_handlers("anyplace"),
        skills=build_default_skill_registry(),
    )
    assert pre_attachment_context["placement_obligation"] is None
    active_candidate = memory.anygrasp_candidate_policy()["active_candidate"]
    memory.save_fact(
        "grasp_execution",
        {
            "status": "completed",
            "stage": "attached",
            "candidate_id": active_candidate["id"],
        },
        source="test",
    )
    memory.save_fact(
        "attachment_gate",
        {
            "status": "resolved",
            "verdict": "PASS",
            "candidate_id": active_candidate["id"],
        },
        source="test",
    )

    observation = _observation()
    observation.task = "pick cube and place it in basket"
    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=_tools_with_handlers("anyplace"),
        skills=build_default_skill_registry(),
    )

    retained = context["retained_targeted_grasp"]
    obligation = context["placement_obligation"]
    assert obligation["required_tool"] == "anyplace"
    assert obligation["required_parameters"] == {
        "rgb": retained["source"]["rgb"],
        "depth": retained["source"]["depth"],
        "object_mask": retained["source"]["object_mask"],
        "placement_region_mask": {
            "mask_ref": "tmp/mask_000.png",
            "source_image": retained["source"]["rgb"],
        },
        "intrinsics": retained["source"]["intrinsics"],
        "selected_grasp": {
            "candidate": retained["candidate"],
            "source": retained["source"],
        },
        "depth_truncation": 2.0,
    }

    exact = obligation["required_parameters"]
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {"kind": "response", "name": "talk", "parameters": {"message": "unused"}}
        ),
    )

    decision = planner.plan(
        observation,
        memory=memory,
        tools=_tools_with_handlers("anyplace"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "anyplace"
    assert decision.parameters == exact
    assert decision.metadata["execution_model"] == "host_obligation_dispatch"
    assert decision.metadata["host_obligation"]["tool"] == "anyplace"

    configured = build_tool_context(
        observation=observation,
        memory=memory,
        tools=_tools_with_handlers("anyplace"),
        skills=build_default_skill_registry(),
        config=PlannerContextConfig(anyplace_depth_truncation_m=1.5),
    )
    assert configured["placement_obligation"]["required_parameters"][
        "depth_truncation"
    ] == 1.5


def test_placement_transform_obligation_joins_rank_zero_pose_after_attachment() -> None:
    place_pose = {
        "id": "place_grasp_000",
        "source_grasp_id": "grasp_003",
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
                            "details": {
                                "outputs": {
                                    "candidate_count": 1,
                                    "selected_grasp_id": "grasp_003",
                                    "placement_candidates": [
                                        {
                                            "id": "placement_000",
                                            "source_grasp_id": "grasp_003",
                                            "place_grasp_pose": place_pose,
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
    memory.save_fact(
        "grasp_execution",
        {
            "status": "completed",
            "stage": "attached",
            "candidate_id": "grasp_005",
            "compiled_grasp": {"camera_frame_id": "agentview"},
        },
        source="test",
    )
    memory.save_fact(
        "attachment_gate",
        {"status": "resolved", "verdict": "PASS"},
        source="test",
    )
    extrinsics = {
        "camera_frame": "opengl",
        "frame_transform": "camera_to_world",
        "matrix_layout": "row_major",
        "pos": [0.8, 0.0, 0.65],
        "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    }
    observation = EnvObservation(
        task="pick can and place it in basket",
        cameras=[
            CameraFrame(
                frame_id="agentview",
                rgb=[[[0, 0, 0]]],
                extrinsics=extrinsics,
            )
        ],
        robot=RobotState(),
    )
    mismatched_context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=_tools_with_handlers("camera_pose_to_world"),
        skills=build_default_skill_registry(),
    )
    assert mismatched_context["placement_transform_obligation"] is None
    memory.save_fact(
        "grasp_execution",
        {
            "status": "completed",
            "stage": "attached",
            "candidate_id": "grasp_003",
            "compiled_grasp": {"camera_frame_id": "agentview"},
        },
        source="test",
    )
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {"kind": "response", "name": "talk", "parameters": {"message": "unused"}}
        )
    )

    decision = planner.plan(
        observation,
        memory=memory,
        tools=_tools_with_handlers("camera_pose_to_world"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "camera_pose_to_world"
    assert decision.parameters == {
        "camera_pose": place_pose,
        "camera_extrinsics": extrinsics,
        "camera_frame_id": "agentview",
    }
    assert decision.metadata["execution_model"] == "host_obligation_dispatch"
    assert decision.metadata["host_obligation"]["tool"] == "camera_pose_to_world"

    memory.artifacts["camera_pose_to_world_world_pose_latest"] = {
        "source": "tool_result",
        "value": {
            "tool": "camera_pose_to_world",
            "type": "world_pose",
            "source_grasp_id": "place_grasp_000",
        },
    }
    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=_tools_with_handlers("camera_pose_to_world"),
        skills=build_default_skill_registry(),
    )
    assert context["placement_transform_obligation"] is None


def test_anyplace_candidate_selection_prefers_receptacle_interior_clearance(
    tmp_path: Path,
) -> None:
    mask = tmp_path / "basket-mask.png"
    image = Image.new("L", (100, 100), 0)
    for y in range(10, 91):
        for x in range(10, 91):
            image.putpixel((x, y), 255)
    image.save(mask)
    candidates = [
        {
            "id": "placement_edge",
            "place_grasp_pose": {
                "source_grasp_id": "grasp_003",
                "frame": "camera",
                "gripper_tip_position_xyz": [-0.3, -0.3, 1.0],
            },
        },
        {
            "id": "placement_center",
            "place_grasp_pose": {
                "source_grasp_id": "grasp_003",
                "frame": "camera",
                "gripper_tip_position_xyz": [0.0, 0.0, 1.0],
            },
        },
    ]

    selected, selection = _select_anyplace_candidate(
        candidates,
        anyplace_output={
            "source": {
                "placement_region_mask": {"mask_ref": str(mask)},
                "intrinsics": {"fx": 100.0, "fy": 100.0, "cx": 50.0, "cy": 50.0},
            }
        },
        source_grasp_id="grasp_003",
    )

    assert selected["id"] == "placement_center"
    assert selection["policy"] == "max_receptacle_mask_bbox_clearance"
    assert selection["original_rank"] == 1
    assert selection["projected_pixel_xy"] == [50.0, 50.0]


def test_placement_motion_requires_high_hover_before_vertical_descend() -> None:
    release_pose = {
        "id": "place_grasp_000",
        "source_grasp_id": "grasp_003",
        "frame": "world",
        "translation_xyz": [0.07, 0.30, 0.13],
        "rotation_matrix": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    }
    memory = AgentMemory()
    memory.save_fact(
        "grasp_execution",
        {
            "schema_version": "openeta.grasp_execution.v1",
            "status": "completed",
            "stage": "attached",
            "candidate_id": "grasp_003",
        },
        source="test",
    )
    memory.save_fact(
        "attachment_gate",
        {"status": "resolved", "verdict": "PASS", "candidate_id": "grasp_003"},
        source="test",
    )
    memory.artifacts["camera_pose_to_world_world_pose_latest"] = {
        "source": "tool_result",
        "value": {
            "tool": "camera_pose_to_world",
            "type": "world_pose",
            "source_grasp_id": "place_grasp_000",
            "world_pose": release_pose,
        },
    }
    observation = EnvObservation(
        task="pick can and place it in basket",
        cameras=[],
        robot=RobotState(end_effector_pose={"xyz": [0.13, 0.04, 0.22]}),
    )
    final_hover_pose = {
        "frame": "world",
        "xyz": [0.07, 0.30, 0.23],
        "source_grasp_id": "grasp_003",
        "placement_pose_id": "place_grasp_000",
        "placement_stage": "carry_hover_final",
    }
    carry_distance = math.hypot(0.07 - 0.13, 0.30 - 0.04)
    carry_ratio = 0.08 / carry_distance
    hover_pose = {
        "frame": "world",
        "xyz": [
            0.13 + (0.07 - 0.13) * carry_ratio,
            0.04 + (0.30 - 0.04) * carry_ratio,
            0.23,
        ],
        "source_grasp_id": "grasp_003",
        "placement_pose_id": "place_grasp_000",
        "placement_stage": "carry_hover",
    }
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            [
                {
                    "kind": "tool_call",
                    "name": "move_to",
                    "parameters": {"target_pose": release_pose},
                },
                {
                    "kind": "tool_call",
                    "name": "move_to",
                    "parameters": {"target_pose": hover_pose},
                },
            ]
        ),
        max_validation_retries=1,
    )

    decision = planner.plan(
        observation,
        memory=memory,
        tools=_tools_with_handlers("move_to"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "move_to"
    assert decision.parameters == {"target_pose": hover_pose}
    assert decision.metadata["execution_model"] == "host_obligation_dispatch"
    assert decision.metadata["host_obligation"]["stage"] == "carry_hover"
    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=_tools_with_handlers("move_to"),
        skills=build_default_skill_registry(),
    )
    assert context["placement_motion_guidance"]["stage"] == "carry_hover"
    assert context["placement_motion_guidance"]["safe_hover_pose"] == hover_pose
    assert context["placement_motion_guidance"]["final_hover_pose"] == final_hover_pose
    assert context["placement_motion_guidance"]["carry_max_step_m"] == 0.08
    assert context["placement_motion_guidance"]["release_pose"]["translation_xyz"] == [
        0.07,
        0.30,
        0.13 + 0.08,
    ]
    assert context["placement_motion_guidance"]["anyplace_reference_pose"] == release_pose

    observation.robot.end_effector_pose = {"xyz": [0.13, 0.04, 0.15]}
    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=_tools_with_handlers("move_to"),
        skills=build_default_skill_registry(),
    )
    assert context["placement_motion_guidance"]["stage"] == "carry_raise"
    assert context["placement_motion_guidance"]["safe_hover_pose"]["xyz"] == [
        0.13,
        0.04,
        0.23,
    ]

    observation.robot.end_effector_pose = {"xyz": final_hover_pose["xyz"]}
    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=_tools_with_handlers("move_to"),
        skills=build_default_skill_registry(),
    )
    assert context["placement_motion_guidance"]["stage"] == "descend"
    assert context["placement_motion_guidance"]["safe_hover_pose"]["xyz"] == pytest.approx(
        [0.07, 0.30, 0.21]
    )
    decision = planner.plan(
        observation,
        memory=memory,
        tools=_tools_with_handlers("move_to"),
        skills=build_default_skill_registry(),
    )
    assert decision.action == "move_to"
    assert decision.parameters["target_pose"]["xyz"] == pytest.approx([0.07, 0.30, 0.21])
    assert decision.metadata["host_obligation"]["stage"] == "descend"

    observation.robot.end_effector_pose = {"xyz": [0.07, 0.30, 0.13 + 0.08]}
    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=_tools_with_handlers("gripper_control"),
        skills=build_default_skill_registry(),
    )
    assert context["placement_motion_guidance"]["stage"] == "release"
    decision = planner.plan(
        observation,
        memory=memory,
        tools=_tools_with_handlers("move_to"),
        skills=build_default_skill_registry(),
    )
    assert decision.action == "move_to"
    assert decision.parameters["target_pose"]["xyz"] == pytest.approx([0.07, 0.30, 0.21])
    assert decision.metadata["host_obligation"]["stage"] == "release"

    observation.robot.gripper_state = {"open": False, "openness": 0.02}
    decision = planner.plan(
        observation,
        memory=memory,
        tools=_tools_with_handlers("gripper_control"),
        skills=build_default_skill_registry(),
    )
    assert decision.action == "gripper_control"
    assert decision.parameters == {"position": 1}
    assert decision.metadata["execution_model"] == "host_obligation_dispatch"
    assert decision.metadata["host_obligation"]["stage"] == "placement_drop_detected"


def test_successful_placement_descend_is_immediately_release_ready() -> None:
    memory = AgentMemory()
    memory.save_fact(
        "grasp_execution",
        {
            "schema_version": "openeta.grasp_execution.v1",
            "status": "completed",
            "stage": "attached",
            "candidate_id": "grasp_003",
        },
        source="test",
    )
    descend_pose = {
        "frame": "world",
        "xyz": [0.07, 0.30, 0.21],
        "source_grasp_id": "grasp_003",
        "placement_pose_id": "place_grasp_000",
        "placement_stage": "descend",
    }

    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "move_to",
                    "parameters": {"target_pose": descend_pose},
                },
                "status": "executed",
                "tool_calls": [
                    {
                        "name": "move_to",
                        "status": "executed",
                        "result": {"success": True, "content": "target reached"},
                    }
                ],
            },
        )
    )

    release = memory.placement_release()
    assert release["status"] == "ready"
    assert release["arrival_stage"] == "descend"
    assert release["release_pose"]["placement_stage"] == "release"


def test_collision_free_descend_stalled_above_receptacle_is_release_ready() -> None:
    memory = AgentMemory()
    descend_pose = {
        "frame": "world",
        "xyz": [-0.0727, 0.2470, 0.5786],
        "source_grasp_id": "grasp_003",
        "placement_pose_id": "place_grasp_000",
        "placement_stage": "descend",
    }
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "move_to",
                    "parameters": {"target_pose": descend_pose},
                },
                "status": "executed",
                "tool_calls": [
                    {
                        "name": "move_to",
                        "status": "executed",
                        "result": {
                            "success": True,
                            "details": {
                                "outputs": {
                                    "motion_summary": {
                                        "reached_target": False,
                                        "collision": {"detected": False},
                                        "end": {"xyz": [-0.0562, 0.2611, 0.6201]},
                                    }
                                }
                            },
                        },
                    }
                ],
            },
        )
    )

    release = memory.placement_release()
    assert release["status"] == "ready"
    assert release["arrival_stage"] == "descend_near_receptacle"
    assert release["release_pose"]["xyz"] == [-0.0562, 0.2611, 0.6201]
    assert release["release_pose"]["adaptive_release"] == {
        "reason": "controller_stalled_safely_above_receptacle",
        "xy_error_m": pytest.approx(0.021704, abs=1e-6),
        "height_above_requested_m": pytest.approx(0.0415, abs=1e-6),
    }


def test_detachment_over_receptacle_completes_subgoal_without_candidate_fallback() -> None:
    memory = AgentMemory()
    memory.save_fact(
        "grasp_candidate_policy",
        {
            "status": "accepted",
            "active_rank": 0,
            "active_candidate": {"id": "grasp_003"},
            "candidates": [{"id": "grasp_003"}, {"id": "grasp_004"}],
            "target_detection": {
                "id": "detection_000",
                "label": "alphabet soup can",
                "target_prompt": "alphabet soup",
            },
        },
        source="test",
    )
    memory.save_fact(
        "selected_sam3_detection",
        {"id": "detection_000", "target_prompt": "alphabet soup"},
        source="test",
    )
    memory.save_fact(
        "grasp_execution",
        {
            "schema_version": "openeta.grasp_execution.v1",
            "status": "completed",
            "stage": "attached",
            "candidate_id": "grasp_003",
        },
        source="test",
    )
    memory.save_fact(
        "attachment_gate",
        {"status": "resolved", "verdict": "PASS", "candidate_id": "grasp_003"},
        source="test",
    )
    memory.save_fact(
        "grasp_lift_probe",
        {"status": "completed", "candidate_id": "grasp_003"},
        source="test",
    )
    memory.artifacts["anyplace_placement_candidates_latest"] = {"value": {}}
    memory.artifacts["camera_pose_to_world_world_pose_latest"] = {"value": {}}

    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "gripper_control",
                    "parameters": {"position": 1},
                },
                "metadata": {
                    "planner_metadata": {
                        "host_obligation": {
                            "schema_version": "openeta.placement_motion_guidance.v1",
                            "stage": "placement_drop_detected",
                            "candidate_id": "grasp_003",
                            "placement_pose_id": "place_grasp_000",
                        }
                    }
                },
                "status": "executed",
                "tool_calls": [
                    {
                        "name": "gripper_control",
                        "status": "executed",
                        "result": {
                            "success": True,
                            "content": "empty gripper normalized open",
                            "details": {
                                "supervision": {
                                    "allowed": True,
                                    "reason": "The target detached in the basket.",
                                    "details": {
                                        "grasp_outcome": "fail",
                                        "candidate_id": "grasp_003",
                                    },
                                }
                            },
                        },
                    }
                ],
            },
        )
    )

    release = memory.placement_release()
    assert release["status"] == "retreated"
    assert release["release_mode"] == "detached_over_receptacle"
    assert memory.grasp_candidate_policy() is None
    assert memory.selected_sam3_detection() is None
    assert memory.grasp_execution() is None
    assert memory.attachment_gate() is None
    completed = memory.facts["completed_placement_subgoals"]["value"]["items"]
    assert completed[-1]["target_object"] == "alphabet soup"
    assert completed[-1]["release_mode"] == "detached_over_receptacle"
    assert "anyplace_placement_candidates_latest" not in memory.artifacts
    assert "camera_pose_to_world_world_pose_latest" not in memory.artifacts


def test_successful_placement_release_clears_stale_attachment_state() -> None:
    memory = AgentMemory()
    memory.save_fact(
        "anygrasp_candidate_policy",
        {
            "status": "accepted",
            "active_candidate": {"id": "grasp_003"},
            "accepted_candidate": {"id": "grasp_003"},
        },
        source="test",
    )
    memory.save_fact(
        "grasp_execution",
        {
            "schema_version": "openeta.grasp_execution.v1",
            "status": "completed",
            "stage": "attached",
            "candidate_id": "grasp_003",
        },
        source="test",
    )
    memory.save_fact(
        "attachment_gate",
        {"status": "resolved", "verdict": "PASS", "candidate_id": "grasp_003"},
        source="test",
    )
    memory.save_fact(
        "grasp_lift_probe",
        {"status": "completed", "candidate_id": "grasp_003"},
        source="test",
    )
    memory.artifacts["anyplace_placement_candidates_latest"] = {"value": {}}
    memory.artifacts["camera_pose_to_world_world_pose_latest"] = {"value": {}}
    release_pose = {
        "frame": "world",
        "xyz": [0.07, 0.30, 0.21],
        "source_grasp_id": "grasp_003",
        "placement_pose_id": "place_grasp_000",
        "placement_stage": "release",
    }
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "move_to",
                    "parameters": {"target_pose": release_pose},
                },
                "status": "executed",
                "tool_calls": [
                    {
                        "name": "move_to",
                        "status": "executed",
                        "result": {"success": True, "content": "target reached"},
                    }
                ],
            },
        )
    )

    assert memory.placement_release()["status"] == "ready"
    assert memory.attachment_gate()["verdict"] == "PASS"
    ready_context = build_tool_context(
        observation=EnvObservation(
            task="pick can and place it in basket",
            cameras=[],
            robot=RobotState(
                end_effector_pose={"xyz": release_pose["xyz"]},
                gripper_state={"open": False, "openness": 0.7},
            ),
        ),
        memory=memory,
        tools=_tools_with_handlers("move_to", "gripper_control"),
        skills=build_default_skill_registry(),
    )
    assert ready_context["placement_release_obligation"] == {
        "schema_version": "openeta.placement_release_obligation.v1",
        "status": "required",
        "stage": "release",
        "required_action": {
            "name": "gripper_control",
            "parameters": {"position": 1},
        },
        "rule": (
            "The retained grasp reached the derived release pose. Open the "
            "gripper immediately; do not rerun target localization or insert "
            "another placement motion."
        ),
    }
    planner = ToolCallingPlanner(StaticPlannerBackend([]))
    decision = planner.plan(
        EnvObservation(
            task="pick can and place it in basket",
            cameras=[],
            robot=RobotState(
                end_effector_pose={"xyz": release_pose["xyz"]},
                gripper_state={"open": False, "openness": 0.7},
            ),
        ),
        memory=memory,
        tools=_tools_with_handlers("move_to", "gripper_control"),
        skills=build_default_skill_registry(),
    )
    assert decision.action == "gripper_control"
    assert decision.parameters == {"position": 1}
    assert decision.metadata["execution_model"] == "host_obligation_dispatch"
    assert decision.metadata["host_obligation"]["stage"] == "release"

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
                        "result": {"success": True, "content": "gripper opened"},
                    }
                ],
            },
        )
    )

    assert memory.placement_release()["status"] == "released"
    assert memory.anygrasp_candidate_policy() is None
    assert memory.grasp_lift_probe() is None
    assert memory.grasp_execution() is None
    assert memory.attachment_gate() is None
    assert "anyplace_placement_candidates_latest" not in memory.artifacts
    assert "camera_pose_to_world_world_pose_latest" not in memory.artifacts
    context = build_tool_context(
        observation=EnvObservation(
            task="pick can and place it in basket",
            cameras=[],
            robot=RobotState(
                end_effector_pose={"xyz": release_pose["xyz"]},
                gripper_state={"open": True, "openness": 1.0},
            ),
        ),
        memory=memory,
        tools=_tools_with_handlers("move_to", "gripper_control"),
        skills=build_default_skill_registry(),
    )
    assert context["placement_release"]["status"] == "released"
    assert context["placement_motion_guidance"] is None
    retreat_pose = {
        "frame": "world",
        "xyz": [0.07, 0.30, 0.31],
        "source_grasp_id": "grasp_003",
        "placement_pose_id": "place_grasp_000",
        "placement_stage": "retreat",
    }
    assert context["placement_release_obligation"] == {
        "schema_version": "openeta.placement_release_obligation.v1",
        "status": "required",
        "stage": "retreat",
        "required_action": {
            "name": "move_to",
            "parameters": {"target_pose": retreat_pose},
        },
        "retreat_distance_m": 0.10,
        "rule": (
            "Retreat vertically with the gripper open before judging placement. "
            "Use the resulting same-episode environment receipt as official reward evidence."
        ),
    }
    decision = planner.plan(
        EnvObservation(
            task="pick can and place it in basket",
            cameras=[],
            robot=RobotState(
                end_effector_pose={"xyz": release_pose["xyz"]},
                gripper_state={"open": True, "openness": 1.0},
            ),
        ),
        memory=memory,
        tools=_tools_with_handlers("move_to"),
        skills=build_default_skill_registry(),
    )
    assert decision.action == "move_to"
    assert decision.parameters == {"target_pose": retreat_pose}
    assert decision.metadata["execution_model"] == "host_obligation_dispatch"
    assert decision.metadata["host_obligation"]["stage"] == "retreat"

    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "move_to",
                    "parameters": {"target_pose": retreat_pose},
                },
                "status": "executed",
                "tool_calls": [
                    {
                        "name": "move_to",
                        "status": "executed",
                        "result": {
                            "success": True,
                            "details": {
                                "outputs": {
                                    "motion_summary": {"reached_target": True},
                                }
                            },
                        },
                    }
                ],
            },
        )
    )
    assert memory.placement_release()["status"] == "retreated"
    context = build_tool_context(
        observation=EnvObservation(
            task="pick can and place it in basket",
            cameras=[],
            robot=RobotState(
                end_effector_pose={"xyz": retreat_pose["xyz"]},
                gripper_state={"open": True, "openness": 1.0},
            ),
        ),
        memory=memory,
        tools=_tools_with_handlers("move_to"),
        skills=build_default_skill_registry(),
    )
    assert context["placement_release_obligation"] is None


def test_wrist_alignment_obligation_joins_current_geometry(tmp_path: Path) -> None:
    selected_rgb = tmp_path / "selected" / "wrist.rgb.png"
    current_rgb = tmp_path / "current" / "wrist.rgb.png"
    current_depth = tmp_path / "current" / "wrist.depth.png"
    selected_rgb.parent.mkdir()
    current_rgb.parent.mkdir()
    selected_rgb.write_bytes(b"same-wrist-scene")
    current_rgb.write_bytes(b"same-wrist-scene")
    current_depth.write_bytes(b"depth")
    memory = AgentMemory()
    _record_pending_sam3_selection(memory, original_image_ref=str(selected_rgb))
    memory.resolve_sam3_selection(
        result_id="sam3-run-selection",
        detection_id="detection_000",
        selection_source="main_agent_vlm",
    )
    memory.save_fact(
        "grasp_execution",
        {
            "status": "required",
            "stage": "align",
            "compiled_grasp": {"schema_version": "openeta.compiled_grasp_seed.v1"},
        },
        source="test",
    )
    memory.save_fact("scene_epoch", {"epoch": 3}, source="test")
    observation = EnvObservation(
        task="pick alphabet soup",
        cameras=[
            CameraFrame(
                frame_id="wrist",
                rgb=[[[0, 0, 0]]],
                depth=[[1.0]],
                intrinsics={"fx": 100.0, "fy": 100.0, "cx": 0.5, "cy": 0.5, "scale": 1000},
                extrinsics={
                    "pos": [0.0, 0.0, 0.5],
                    "mat": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                },
            )
        ],
        robot=RobotState(end_effector_pose={"xyz": [0.1, 0.2, 0.3]}),
        metadata={
            "image_artifacts": [
                {"kind": "rgb", "frame_id": "wrist", "path": str(current_rgb)},
                {"kind": "depth", "frame_id": "wrist", "path": str(current_depth)},
            ]
        },
    )

    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=_tools_with_handlers("compute_wrist_alignment"),
        skills=build_default_skill_registry(),
    )

    required = context["wrist_alignment_obligation"]["required_parameters"]
    assert required["target_mask"] == "tmp/mask_000.png"
    assert required["depth"] == str(current_depth)
    assert required["scene_epoch"] == 3
    assert required["desired_pixel_xy"] == [0.5, 0.5]
    assert required["current_eef_pose"]["xyz"] == [0.1, 0.2, 0.3]

    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {"kind": "response", "name": "talk", "parameters": {"message": "unused"}}
        )
    )
    decision = planner.plan(
        observation,
        memory=memory,
        tools=_tools_with_handlers("compute_wrist_alignment"),
        skills=build_default_skill_registry(),
    )
    assert decision.action == "compute_wrist_alignment"
    assert decision.parameters == required
    assert decision.metadata["execution_model"] == "host_obligation_dispatch"


def test_motion_reconciliation_preempts_pending_host_grasp_move() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick alphabet soup")
    target_pose = {
        "frame": "world",
        "grasp_stage": "hover",
        "source_grasp_id": "grasp_007",
        "xyz": [0.1, 0.2, 0.3],
    }
    memory.save_fact(
        "grasp_execution",
        {
            "schema_version": "openeta.grasp_execution.v1",
            "status": "required",
            "stage": "hover",
            "candidate_id": "grasp_007",
            "required_action": {
                "name": "move_to",
                "parameters": {"target_pose": target_pose},
            },
        },
        source="test",
    )
    memory.save_fact(
        "motion_reconciliation",
        {
            "status": "required",
            "tool": "move_to",
            "candidate_id": "grasp_007",
            "intended_parameters": {"target_pose": target_pose},
        },
        source="test",
    )
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {"kind": "tool_call", "name": "move_to", "parameters": {"target_pose": target_pose}}
        )
    )

    decision = planner.plan(
        _observation(),
        memory=memory,
        tools=_tools_with_handlers("observe", "move_to"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "observe"
    assert decision.parameters == {}
    assert decision.metadata["execution_model"] == "host_obligation_dispatch"
    assert decision.metadata["host_obligation"]["unknown_tool"] == "move_to"


def test_stale_prehover_wrist_mask_dispatches_current_wrist_sam3(tmp_path: Path) -> None:
    stale_rgb = tmp_path / "prehover" / "wrist.rgb.png"
    current_rgb = tmp_path / "hover" / "wrist.rgb.png"
    stale_rgb.parent.mkdir()
    current_rgb.parent.mkdir()
    stale_rgb.write_bytes(b"prehover-wrist-scene")
    current_rgb.write_bytes(b"current-hover-wrist-scene")
    memory = AgentMemory()
    _record_pending_sam3_selection(memory, original_image_ref=str(stale_rgb))
    memory.resolve_sam3_selection(
        result_id="sam3-run-selection",
        detection_id="detection_000",
        selection_source="main_agent_vlm",
    )
    memory.save_fact(
        "grasp_execution",
        {
            "status": "required",
            "stage": "align",
            "compiled_grasp": {"schema_version": "openeta.compiled_grasp_seed.v1"},
        },
        source="test",
    )
    observation = _observation()
    observation.metadata["image_artifacts"] = [
        {"kind": "rgb", "frame_id": "wrist", "path": str(current_rgb)}
    ]
    tools = _tools_with_handlers("sam3", "compute_wrist_alignment")

    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=tools,
        skills=build_default_skill_registry(),
    )

    obligation = context["wrist_segmentation_obligation"]
    assert obligation["required_parameters"] == {
        "image": str(current_rgb),
        "prompt": "alphabet soup",
    }
    assert context["wrist_alignment_obligation"] is None

    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {"kind": "response", "name": "talk", "parameters": {"message": "unused"}}
        )
    )
    decision = planner.plan(
        observation,
        memory=memory,
        tools=tools,
        skills=build_default_skill_registry(),
    )
    assert decision.action == "sam3"
    assert decision.parameters == obligation["required_parameters"]
    assert decision.metadata["execution_model"] == "host_obligation_dispatch"


def test_empty_wrist_sam3_requires_canonical_reference_fallback(tmp_path: Path) -> None:
    previous_wrist = tmp_path / "previous" / "wrist.rgb.png"
    current_wrist = tmp_path / "current" / "wrist.rgb.png"
    previous_wrist.parent.mkdir()
    current_wrist.parent.mkdir()
    previous_wrist.write_bytes(b"same-wrist-scene")
    current_wrist.write_bytes(b"same-wrist-scene")
    memory = AgentMemory()
    memory.save_fact(
        "grasp_execution",
        {"status": "required", "stage": "align"},
        source="test",
    )
    memory.save_fact(
        "target_asset_reference",
        {"environment": "libero", "target_object": "alphabet_soup"},
        source="test",
    )
    memory.save_fact(
        "sam3_no_detection",
        {"result_id": "empty-wrist", "source_image": str(previous_wrist)},
        source="test",
    )
    observation = _observation()
    observation.metadata["image_artifacts"] = [
        {"kind": "rgb", "frame_id": "wrist", "path": str(current_wrist)}
    ]
    expected = {
        "environment": "libero",
        "target_object": "alphabet_soup",
        "scene_image": str(current_wrist),
    }
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {"kind": "response", "name": "talk", "parameters": {"message": "unused"}}
        )
    )

    decision = planner.plan(
        observation,
        memory=memory,
        tools=_tools_with_handlers("sam3", "retrieve_asset_reference"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "retrieve_asset_reference"
    assert decision.parameters == expected
    assert decision.metadata["execution_model"] == "host_obligation_dispatch"


def test_empty_wrist_sam3_derives_reference_without_prior_asset_lookup(
    tmp_path: Path,
) -> None:
    wrist = tmp_path / "wrist.rgb.png"
    wrist.write_bytes(b"current-wrist-scene")
    memory = AgentMemory()
    memory.save_fact(
        "grasp_execution",
        {"status": "required", "stage": "align"},
        source="test",
    )
    memory.save_fact(
        "sam3_no_detection",
        {
            "result_id": "empty-wrist",
            "source_image": str(wrist),
            "target_prompt": "salad dressing",
        },
        source="sam3",
    )
    observation = _observation()
    observation.task = "pick up the salad dressing and place it in the basket"
    observation.metadata = {
        "env_id": "openeta/libero_libero_object_task2-v0",
        "image_artifacts": [{"kind": "rgb", "frame_id": "wrist", "path": str(wrist)}],
    }
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {"kind": "response", "name": "talk", "parameters": {"message": "unused"}}
        )
    )

    decision = planner.plan(
        observation,
        memory=memory,
        tools=_tools_with_handlers("sam3", "retrieve_asset_reference"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "retrieve_asset_reference"
    assert decision.parameters == {
        "environment": "openeta/libero_libero_object_task2-v0",
        "target_object": "salad dressing",
        "scene_image": str(wrist),
    }
    assert decision.metadata["execution_model"] == "host_obligation_dispatch"


def test_stale_empty_wrist_result_does_not_ground_current_frame(
    tmp_path: Path,
) -> None:
    stale_wrist = tmp_path / "stale" / "wrist.rgb.png"
    current_wrist = tmp_path / "current" / "wrist.rgb.png"
    stale_wrist.parent.mkdir()
    current_wrist.parent.mkdir()
    stale_wrist.write_bytes(b"stale-wrist-scene")
    current_wrist.write_bytes(b"current-wrist-scene")
    memory = AgentMemory()
    memory.save_fact(
        "grasp_execution",
        {"status": "required", "stage": "align"},
        source="test",
    )
    memory.save_fact(
        "sam3_no_detection",
        {
            "result_id": "empty-stale-wrist",
            "source_image": str(stale_wrist),
            "target_prompt": "salad dressing",
        },
        source="sam3",
    )
    observation = _observation()
    observation.task = "pick up the salad dressing and place it in the basket"
    observation.metadata = {
        "env_id": "openeta/libero_libero_object_task2-v0",
        "image_artifacts": [
            {"kind": "rgb", "frame_id": "wrist", "path": str(current_wrist)}
        ],
    }
    tools = _tools_with_handlers("sam3", "retrieve_asset_reference")
    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=tools,
        skills=build_default_skill_registry(),
    )

    assert context["wrist_reference_obligation"] is None

    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {
                "kind": "tool_call",
                "name": "sam3",
                "parameters": {
                    "image": str(current_wrist),
                    "prompt": "salad dressing",
                },
            }
        )
    )
    decision = planner.plan(
        observation,
        memory=memory,
        tools=tools,
        skills=build_default_skill_registry(),
    )

    assert decision.action == "sam3"
    assert decision.parameters["image"] == str(current_wrist)


def test_wrist_align_allows_molmopoint_when_reference_cannot_be_derived(
    tmp_path: Path,
) -> None:
    wrist = tmp_path / "wrist.rgb.png"
    wrist.write_bytes(b"current-wrist-scene")
    memory = AgentMemory()
    memory.save_fact(
        "grasp_execution",
        {"status": "required", "stage": "align"},
        source="test",
    )
    memory.save_fact(
        "sam3_no_detection",
        {"result_id": "empty-wrist", "source_image": str(wrist)},
        source="sam3",
    )
    observation = _observation()
    observation.metadata = {
        "image_artifacts": [{"kind": "rgb", "frame_id": "wrist", "path": str(wrist)}]
    }
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {
                "kind": "tool_call",
                "name": "molmopoint",
                "parameters": {
                    "images": [str(wrist)],
                    "prompt": "Point to the target object in Image 1.",
                },
            }
        )
    )

    decision = planner.plan(
        observation,
        memory=memory,
        tools=_tools_with_handlers("molmopoint", "sam3", "compute_wrist_alignment"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "molmopoint"
    assert (
        memory.grasp_execution_gate_error(
            tool_name="molmopoint",
            parameters=decision.parameters,
        )
        is None
    )


def test_wrist_align_allows_compute_wrist_alignment_when_no_obligation_pending(
    tmp_path: Path,
) -> None:
    # Regression: the planner validator's align fallback used to omit
    # compute_wrist_alignment from its whitelist while the memory.py tool gate
    # allowed it. When no wrist_alignment_obligation exists (empty wrist SAM3),
    # the planner rejected the model's compute_wrist_alignment call, exhausted
    # retries, and terminated the episode. The two gates must agree.
    wrist = tmp_path / "wrist.rgb.png"
    wrist.write_bytes(b"current-wrist-scene")
    memory = AgentMemory()
    memory.save_fact(
        "grasp_execution",
        {"status": "required", "stage": "align"},
        source="test",
    )
    observation = _observation()
    observation.metadata = {
        "image_artifacts": [{"kind": "rgb", "frame_id": "wrist", "path": str(wrist)}]
    }
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {
                "kind": "tool_call",
                "name": "compute_wrist_alignment",
                "parameters": {},
            }
        )
    )

    decision = planner.plan(
        observation,
        memory=memory,
        tools=_tools_with_handlers("compute_wrist_alignment"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "compute_wrist_alignment"
    # The planner validator and the tool-execution gate must both allow it.
    assert (
        memory.grasp_execution_gate_error(
            tool_name="compute_wrist_alignment",
            parameters=decision.parameters,
        )
        is None
    )


def test_empty_initial_sam3_requires_exact_task_reference_before_prompt_broadening(
    tmp_path: Path,
) -> None:
    failed_scene = tmp_path / "previous" / "agentview.rgb.png"
    current_scene = tmp_path / "current" / "agentview.rgb.png"
    failed_scene.parent.mkdir()
    current_scene.parent.mkdir()
    failed_scene.write_bytes(b"same-static-scene")
    current_scene.write_bytes(b"same-static-scene")
    memory = AgentMemory()
    memory.start_session(task="pick up alphabet soup and place it into basket.")
    memory.save_fact(
        "sam3_no_detection",
        {
            "result_id": "empty-exact-target",
            "source_image": str(failed_scene),
            "target_prompt": "alphabet soup can",
        },
        source="sam3",
    )
    observation = _observation()
    observation.task = "pick up alphabet soup and place it into basket."
    observation.metadata = {
        "env_id": "openeta/libero_libero_object_task0-v0",
        "image_artifacts": [
            {
                "kind": "rgb",
                "frame_id": "agentview",
                "path": str(current_scene),
            }
        ],
    }
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {
                "kind": "tool_call",
                "name": "sam3",
                "parameters": {
                    "image": str(current_scene),
                    "prompt": "soup can",
                },
            }
        )
    )

    decision = planner.plan(
        observation,
        memory=memory,
        tools=_tools_with_handlers("sam3", "retrieve_asset_reference"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "retrieve_asset_reference"
    assert decision.parameters == {
        "environment": "openeta/libero_libero_object_task0-v0",
        "target_object": "alphabet soup",
        "scene_image": str(current_scene),
    }
    assert decision.metadata["execution_model"] == "host_obligation_dispatch"
    obligation = decision.metadata["host_obligation"]
    assert obligation["empty_sam3_result_id"] == "empty-exact-target"


@pytest.mark.parametrize(
    ("task", "target"),
    [
        ("pick up the alphabet soup and place it in the basket", "alphabet soup"),
        ("pick cube and place it into basket", "cube"),
        ("please grasp a milk box", "milk box"),
        (
            "pick up the black bowl between the plate and the ramekin and place it on the plate",
            "black bowl",
        ),
    ],
)
def test_empty_sam3_reference_extracts_exact_pick_target(
    tmp_path: Path,
    task: str,
    target: str,
) -> None:
    scene = tmp_path / "agentview.png"
    scene.write_bytes(b"scene")
    memory = AgentMemory()
    memory.save_fact(
        "sam3_no_detection",
        {"result_id": "empty", "source_image": str(scene), "target_prompt": target},
        source="sam3",
    )
    observation = _observation()
    observation.task = task
    observation.metadata = {
        "env_id": "libero-env",
        "image_artifacts": [{"kind": "rgb", "frame_id": "agentview", "path": str(scene)}],
    }

    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=_tools_with_handlers("retrieve_asset_reference"),
        skills=build_default_skill_registry(),
    )

    assert context["target_reference_obligation"]["required_parameters"]["target_object"] == target


def test_empty_sam3_reference_strips_scene_relation_for_object_memory(
    tmp_path: Path,
) -> None:
    scene = tmp_path / "agentview.png"
    scene.write_bytes(b"scene")
    memory = AgentMemory()
    memory.save_fact(
        "sam3_no_detection",
        {"result_id": "empty", "source_image": str(scene), "target_prompt": "black bowl"},
        source="sam3",
    )
    observation = _observation()
    observation.task = "pick up the black bowl on the cookie box and place it on the plate"
    observation.metadata = {
        "env_id": "openeta/libero_libero_spatial_task3-v0",
        "image_artifacts": [{"kind": "rgb", "frame_id": "agentview", "path": str(scene)}],
    }

    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=_tools_with_handlers("retrieve_asset_reference"),
        skills=build_default_skill_registry(),
    )

    assert context["target_reference_obligation"]["required_parameters"]["target_object"] == (
        "black bowl"
    )


def test_failed_reference_localization_is_not_automatically_replayed(
    tmp_path: Path,
) -> None:
    scene = tmp_path / "agentview.png"
    scene.write_bytes(b"occluded-scene")
    memory = AgentMemory()
    memory.save_fact(
        "sam3_no_detection",
        {
            "result_id": "empty-current-scene",
            "source_image": str(scene),
            "target_prompt": "alphabet soup",
        },
        source="sam3",
    )
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "tool_calls": [
                    {
                        "name": "retrieve_asset_reference",
                        "status": "failed",
                        "result": {
                            "success": False,
                            "content": "Object memory localization failed.",
                            "details": {"outputs": {"reason": "object_memory_localization_failed"}},
                        },
                    }
                ]
            },
        )
    )
    observation = _observation()
    observation.task = "pick up alphabet soup and place it into basket."
    observation.metadata = {
        "env_id": "libero-env",
        "image_artifacts": [{"kind": "rgb", "frame_id": "agentview", "path": str(scene)}],
    }

    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=_tools_with_handlers("retrieve_asset_reference", "molmopoint"),
        skills=build_default_skill_registry(),
    )

    assert context["target_reference_obligation"] is None
    fallback = context["molmopoint_fallback_obligation"]
    assert fallback["status"] == "required"
    assert fallback["attempt"] == 1
    assert fallback["required_parameters"]["images"] == [str(scene)]

    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {"kind": "response", "name": "talk", "parameters": {"message": "unused"}}
        )
    )
    decision = planner.plan(
        observation,
        memory=memory,
        tools=_tools_with_handlers("retrieve_asset_reference", "molmopoint"),
        skills=build_default_skill_registry(),
    )
    assert decision.action == "molmopoint"
    assert decision.parameters["images"] == [str(scene)]
    assert decision.metadata["execution_model"] == "host_obligation_dispatch"


def test_molmopoint_fallback_stops_after_bounded_failures(tmp_path: Path) -> None:
    scene = tmp_path / "agentview.png"
    scene.write_bytes(b"scene")
    memory = AgentMemory()
    memory.save_fact(
        "sam3_no_detection",
        {
            "result_id": "empty-bounded",
            "source_image": str(scene),
            "target_prompt": "alphabet soup",
        },
        source="sam3",
    )
    memory.save_fact(
        "reference_localization_failure",
        {
            "sam3_result_id": "empty-bounded",
            "target_object": "alphabet soup",
            "scene_image": str(scene),
            "molmopoint_attempts": 0,
        },
        source="retrieve_asset_reference",
    )
    for _ in range(2):
        memory.add_action(
            EnvAction(
                action_type="tool_call",
                command={
                    "tool_calls": [
                        {
                            "name": "molmopoint",
                            "result": {
                                "success": False,
                                "content": "MolmoPoint failed: backend unavailable.",
                            },
                        }
                    ]
                },
            )
        )
    assert memory.reference_localization_failure()["molmopoint_attempts"] == 2

    observation = _observation()
    observation.task = "pick up alphabet soup and place it into basket."
    observation.metadata = {
        "env_id": "libero-env",
        "image_artifacts": [{"kind": "rgb", "frame_id": "agentview", "path": str(scene)}],
    }
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {"kind": "response", "name": "talk", "parameters": {"message": "unused"}}
        )
    )

    decision = planner.plan(
        observation,
        memory=memory,
        tools=_tools_with_handlers("retrieve_asset_reference", "molmopoint"),
        skills=build_default_skill_registry(),
    )

    assert decision.action_type == "response"
    assert decision.action == "ask_human"
    assert decision.parameters["failure_code"] == "target_localization_exhausted"
    assert decision.metadata["execution_model"] == "host_obligation_dispatch"


def test_molmopoint_fallback_budget_survives_successful_wrong_point_cycle(
    tmp_path: Path,
) -> None:
    scene = tmp_path / "agentview.png"
    scene.write_bytes(b"scene")
    memory = AgentMemory()
    memory.save_fact(
        "sam3_no_detection",
        {
            "result_id": "empty-first",
            "source_image": str(scene),
            "target_prompt": "alphabet soup",
        },
        source="sam3",
    )
    memory.save_fact(
        "reference_localization_failure",
        {
            "sam3_result_id": "empty-first",
            "target_object": "alphabet soup",
            "scene_image": str(scene),
            "molmopoint_attempts": 0,
        },
        source="retrieve_asset_reference",
    )
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "tool_calls": [
                    {"name": "molmopoint", "result": {"success": True, "details": {}}}
                ]
            },
        )
    )
    assert memory.reference_localization_failure()["molmopoint_attempts"] == 1

    memory.save_fact(
        "sam3_no_detection",
        {
            "result_id": "empty-second",
            "source_image": str(scene),
            "target_prompt": "alphabet soup",
            "rejection_reason": "the selected mask was the green bottle",
        },
        source="reject_sam3_detections",
    )
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "name": "retrieve_asset_reference",
                    "parameters": {
                        "target_object": "alphabet soup",
                        "scene_image": str(scene),
                    },
                },
                "tool_calls": [
                    {
                        "name": "retrieve_asset_reference",
                        "parameters": {
                            "target_object": "alphabet soup",
                            "scene_image": str(scene),
                        },
                        "result": {"success": False},
                    }
                ],
            },
        )
    )

    assert memory.reference_localization_failure()["sam3_result_id"] == "empty-second"
    assert memory.reference_localization_failure()["molmopoint_attempts"] == 1
    observation = _observation()
    observation.task = "pick up alphabet soup"
    observation.metadata = {
        "env_id": "libero-env",
        "image_artifacts": [{"kind": "rgb", "frame_id": "agentview", "path": str(scene)}],
    }
    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=_tools_with_handlers("retrieve_asset_reference", "molmopoint"),
        skills=build_default_skill_registry(),
    )
    fallback = context["molmopoint_fallback_obligation"]
    assert fallback["status"] == "required"
    assert fallback["attempt"] == 2
    assert "green bottle" in fallback["required_parameters"]["prompt"]


def test_molmopoint_fallback_budget_uses_scene_epoch_not_render_path(
    tmp_path: Path,
) -> None:
    first_scene = tmp_path / "observation-1.png"
    second_scene = tmp_path / "observation-2.png"
    first_scene.write_bytes(b"same-world")
    second_scene.write_bytes(b"same-world")
    memory = AgentMemory()
    memory.save_fact("scene_epoch", {"epoch": 7}, source="test")
    memory.save_fact(
        "sam3_no_detection",
        {
            "result_id": "empty-first",
            "source_image": str(first_scene),
            "target_prompt": "alphabet soup",
        },
        source="sam3",
    )
    memory.save_fact(
        "reference_localization_failure",
        {
            "sam3_result_id": "empty-first",
            "target_object": "alphabet soup",
            "scene_image": str(first_scene),
            "scene_epoch": 7,
            "molmopoint_attempts": 0,
        },
        source="retrieve_asset_reference",
    )
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "tool_calls": [
                    {"name": "molmopoint", "result": {"success": True, "details": {}}}
                ]
            },
        )
    )

    memory.save_fact(
        "sam3_no_detection",
        {
            "result_id": "empty-second",
            "source_image": str(second_scene),
            "target_prompt": "alphabet soup",
        },
        source="sam3",
    )
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "name": "retrieve_asset_reference",
                    "parameters": {
                        "target_object": "alphabet soup",
                        "scene_image": str(second_scene),
                    },
                },
                "tool_calls": [
                    {
                        "name": "retrieve_asset_reference",
                        "parameters": {
                            "target_object": "alphabet soup",
                            "scene_image": str(second_scene),
                        },
                        "result": {"success": False},
                    }
                ],
            },
        )
    )

    failure = memory.reference_localization_failure()
    assert failure["scene_image"] == str(second_scene)
    assert failure["scene_epoch"] == 7
    assert failure["molmopoint_attempts"] == 1


def test_pending_wrist_reference_localization_suppresses_duplicate_retrieve(
    tmp_path: Path,
) -> None:
    wrist = tmp_path / "wrist.rgb.png"
    wrist.write_bytes(b"wrist-scene")
    points = [{"x": 12.0, "y": 18.0, "label": 1}]
    memory = AgentMemory()
    memory.save_fact(
        "grasp_execution",
        {"status": "required", "stage": "align"},
        source="test",
    )
    memory.save_fact(
        "target_asset_reference",
        {"environment": "libero", "target_object": "alphabet_soup"},
        source="test",
    )
    memory.save_fact(
        "sam3_no_detection",
        {"result_id": "empty-wrist", "source_image": str(wrist)},
        source="test",
    )
    memory.save_fact(
        "pending_reference_localization",
        {
            "scene_image": str(wrist),
            "target_object": "alphabet_soup",
            "positive_points": points,
            "required_parameter": "positive_points",
            "required_next_tool": "sam3",
        },
        source="retrieve_asset_reference",
    )
    observation = _observation()
    observation.metadata["image_artifacts"] = [
        {"kind": "rgb", "frame_id": "wrist", "path": str(wrist)}
    ]

    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=_tools_with_handlers("sam3", "retrieve_asset_reference"),
        skills=build_default_skill_registry(),
    )

    assert context["wrist_reference_obligation"] is None
    assert context["reference_localization_obligation"]["positive_points"] == points

    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {"kind": "response", "name": "talk", "parameters": {"message": "unused"}}
        )
    )
    decision = planner.plan(
        observation,
        memory=memory,
        tools=_tools_with_handlers("sam3", "retrieve_asset_reference"),
        skills=build_default_skill_registry(),
    )
    assert decision.action == "sam3"
    assert decision.parameters == {"image": str(wrist), "positive_points": points}
    assert decision.metadata["execution_model"] == "host_obligation_dispatch"


def test_wrist_reference_uses_remaining_exact_multi_object_task_name(
    tmp_path: Path,
) -> None:
    wrist = tmp_path / "wrist.rgb.png"
    wrist.write_bytes(b"wrist-scene")
    memory = AgentMemory()
    memory.save_fact(
        "grasp_execution",
        {"status": "required", "stage": "align"},
        source="test",
    )
    memory.save_fact(
        "sam3_no_detection",
        {
            "result_id": "empty-wrist",
            "source_image": str(wrist),
            "target_prompt": "tomato sauce bottle",
        },
        source="test",
    )
    memory.save_fact(
        "completed_placement_subgoals",
        {"items": [{"target_object": "alphabet soup can"}]},
        source="test",
    )
    observation = _observation()
    observation.task = "put both the alphabet soup and the tomato sauce in the basket"
    observation.metadata = {
        "env_id": "openeta/libero_libero_10_task0-v0",
        "image_artifacts": [{"kind": "rgb", "frame_id": "wrist", "path": str(wrist)}],
    }

    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=_tools_with_handlers("retrieve_asset_reference"),
        skills=build_default_skill_registry(),
    )

    obligation = context["wrist_reference_obligation"]
    assert obligation["required_parameters"] == {
        "environment": "openeta/libero_libero_10_task0-v0",
        "target_object": "tomato sauce",
        "scene_image": str(wrist),
    }


def test_failed_wrist_reference_localization_advances_anygrasp_candidate() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick alphabet soup")
    _record_anygrasp_candidate_policy(memory)
    memory.save_fact(
        "grasp_execution",
        {
            "schema_version": "openeta.grasp_execution.v1",
            "status": "required",
            "stage": "align",
            "candidate_id": "grasp_000",
            "required_action": None,
        },
        source="test",
    )

    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "retrieve_asset_reference",
                    "parameters": {
                        "environment": "libero",
                        "target_object": "alphabet soup",
                        "scene_image": "/tmp/wrist.png",
                    },
                },
                "status": "failed",
                "tool_calls": [
                    {
                        "name": "retrieve_asset_reference",
                        "status": "failed",
                        "result": {
                            "success": False,
                            "content": "Object memory localization failed.",
                            "details": {
                                "outputs": {"reason": "object_memory_localization_failed"},
                                "diagnostics": [{"code": "object_memory_localization_failed"}],
                            },
                        },
                    }
                ],
            },
        )
    )

    policy = memory.anygrasp_candidate_policy()
    assert policy["active_candidate"]["id"] == "grasp_001"
    assert policy["active_rank"] == 1
    assert policy["rejected_candidates"][0]["source"] == ("wrist_reference_localization_rejected")
    assert memory.grasp_execution() is None


def test_anyplace_host_dispatches_exact_final_grasp_packet() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube and place it in basket")
    _record_anygrasp_candidate_policy(memory)
    active = memory.anygrasp_candidate_policy()["active_candidate"]
    memory.save_fact(
        "grasp_execution",
        {
            "status": "completed",
            "stage": "attached",
            "candidate_id": active["id"],
        },
        source="test",
    )
    memory.save_fact(
        "attachment_gate",
        {"status": "resolved", "verdict": "PASS", "candidate_id": active["id"]},
        source="test",
    )
    _record_pending_sam3_selection(memory, original_image_ref="tmp/rgb.png")
    memory.resolve_sam3_selection(
        result_id="sam3-run-selection",
        detection_id="detection_000",
        selection_source="main_agent_vlm",
    )
    retained = memory.retained_targeted_grasp()
    exact = {
        "rgb": retained["source"]["rgb"],
        "depth": retained["source"]["depth"],
        "object_mask": retained["source"]["object_mask"],
        "intrinsics": retained["source"]["intrinsics"],
        "placement_region_mask": {
            "mask_ref": "tmp/mask_000.png",
            "source_image": retained["source"]["rgb"],
        },
        "selected_grasp": {
            "candidate": retained["candidate"],
            "source": retained["source"],
        },
        "depth_truncation": 2.0,
    }
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {"kind": "response", "name": "talk", "parameters": {"message": "unused"}}
        )
    )
    observation = _observation()
    observation.task = "pick cube and place it in basket"

    decision = planner.plan(
        observation,
        memory=memory,
        tools=_tools_with_handlers("anyplace"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "anyplace"
    assert decision.parameters == exact
    assert decision.metadata["execution_model"] == "host_obligation_dispatch"


def test_anyplace_host_does_not_repeat_a_deterministic_failure() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube and place it in basket")
    _record_anygrasp_candidate_policy(memory)
    active = memory.anygrasp_candidate_policy()["active_candidate"]
    memory.save_fact(
        "grasp_execution",
        {
            "status": "completed",
            "stage": "attached",
            "candidate_id": active["id"],
        },
        source="test",
    )
    memory.save_fact(
        "attachment_gate",
        {"status": "resolved", "verdict": "PASS", "candidate_id": active["id"]},
        source="test",
    )
    _record_pending_sam3_selection(memory, original_image_ref="tmp/rgb.png")
    memory.resolve_sam3_selection(
        result_id="sam3-run-selection",
        detection_id="detection_000",
        selection_source="main_agent_vlm",
    )
    memory.record(
        "recovery_feedback",
        {
            "source": "action_pipeline",
            "command": {
                "status": "failed",
                "tool_calls": [
                    {
                        "name": "anyplace",
                        "status": "failed",
                        "result": {
                            "success": False,
                            "content": (
                                "AnyPlace placement prediction failed: empty_object_pointcloud."
                            ),
                        },
                    }
                ],
            },
        },
    )
    observation = _observation()
    observation.task = "pick cube and place it in basket"

    context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=_tools_with_handlers("anyplace"),
        skills=build_default_skill_registry(),
    )

    assert context["placement_obligation"] is None


def test_combined_pick_place_blocks_receptacle_segmentation_before_anygrasp() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube and place it in basket")
    _record_pending_sam3_selection(memory)
    memory.resolve_sam3_selection(
        result_id="sam3-run-selection",
        detection_id="detection_000",
        selection_source="main_agent_vlm",
        confidence=0.99,
        reason="The selected mask is the task target cube.",
    )
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            [
                {
                    "kind": "tool_call",
                    "name": "sam3",
                    "parameters": {"image": "agentview.png", "prompt": "basket interior"},
                },
                {
                    "kind": "tool_call",
                    "name": "anygrasp",
                    "parameters": {
                        "rgb": "agentview.png",
                        "depth": "agentview.depth.png",
                        "intrinsics": {
                            "fx": 600.0,
                            "fy": 600.0,
                            "cx": 256.0,
                            "cy": 256.0,
                            "scale": 1000.0,
                        },
                        "target_mask": "tmp/mask_000.png",
                        "mode": "targeted",
                    },
                },
            ]
        ),
        max_validation_retries=1,
    )
    observation = _observation()
    observation.task = "pick cube and place it in basket"

    decision = planner.plan(
        observation,
        memory=memory,
        tools=_tools_with_handlers("sam3", "anygrasp", "anyplace"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "anygrasp"
    first_errors = decision.metadata["validation_attempt_history"][0]["validation_errors"]
    assert any("one active SAM3 selection slot" in error for error in first_errors)


def test_replacement_anygrasp_requires_reopening_after_accepted_motion() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube")
    _record_anygrasp_candidate_policy(memory)
    active = memory.anygrasp_candidate_policy()["active_candidate"]
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "move_to",
                    "parameters": {"target_pose": active},
                },
                "status": "executed",
                "safety_checks": [],
                "tool_calls": [
                    {
                        "name": "move_to",
                        "status": "executed",
                        "result": {"success": True, "content": "target reached"},
                    }
                ],
            },
        )
    )
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            [
                {
                    "kind": "tool_call",
                    "name": "anygrasp",
                    "parameters": {
                        "rgb": "agentview.png",
                        "depth": "agentview.depth.png",
                        "intrinsics": {
                            "fx": 600.0,
                            "fy": 600.0,
                            "cx": 256.0,
                            "cy": 256.0,
                            "scale": 1000.0,
                        },
                        "target_mask": "tmp/mask.png",
                        "mode": "targeted",
                    },
                },
                {
                    "kind": "tool_call",
                    "name": "gripper_control",
                    "parameters": {"position": 1},
                },
            ]
        ),
        max_validation_retries=1,
    )
    observation = _observation()
    observation.robot.gripper_state = {"open": False, "openness": 0.05}

    decision = planner.plan(
        observation,
        memory=memory,
        tools=_tools_with_handlers("anygrasp", "gripper_control"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "gripper_control"
    assert decision.parameters == {"position": 1}
    first_errors = decision.metadata["validation_attempt_history"][0]["validation_errors"]
    assert any("retained ranked grasp-estimation result" in error for error in first_errors)


def test_active_anygrasp_queue_blocks_resegmentation_from_replacing_candidates() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube")
    _record_anygrasp_candidate_policy(memory)
    active_id = memory.anygrasp_candidate_policy()["active_candidate"]["id"]
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            [
                {
                    "kind": "tool_call",
                    "name": "anygrasp",
                    "parameters": {
                        "rgb": "fresh.png",
                        "depth": "fresh.depth.png",
                        "intrinsics": {
                            "fx": 600.0,
                            "fy": 600.0,
                            "cx": 256.0,
                            "cy": 256.0,
                            "scale": 1000.0,
                        },
                        "target_mask": "fresh.mask.png",
                        "mode": "targeted",
                    },
                },
                {
                    "kind": "tool_call",
                    "name": "observe",
                    "parameters": {"reason": "retain the active candidate queue"},
                },
            ]
        ),
        max_validation_retries=1,
    )

    decision = planner.plan(
        _observation(),
        memory=memory,
        tools=_tools_with_handlers("anygrasp", "observe"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "observe"
    first_errors = decision.metadata["validation_attempt_history"][0]["validation_errors"]
    assert any(active_id in error for error in first_errors)
    assert any("Rerun grasp_pose_estimate only after" in error for error in first_errors)


def test_graspgenx_replaces_active_policy_and_uses_existing_greedy_order() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube")
    _record_anygrasp_candidate_policy(memory)
    assert memory.anygrasp_candidate_policy()["source_tool"] == "anygrasp"

    _record_anygrasp_candidate_policy(memory, source_tool="graspgenx")

    policy = memory.anygrasp_candidate_policy()
    assert policy["source_tool"] == "graspgenx"
    assert policy["status"] == "active"
    assert policy["active_rank"] == 0
    assert policy["active_candidate"]["id"] == "graspgenx_000"
    assert policy["remaining_candidate_ids"] == ["graspgenx_001"]
    error = memory.grasp_candidate_gate_error(
        tool_name="camera_pose_to_world",
        parameters={"camera_pose": {"id": "graspgenx_001"}},
    )
    assert "Greedy GraspGenX policy" in error


def test_pipeline_blocks_skipping_ahead_in_anygrasp_candidate_queue() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube")
    _record_anygrasp_candidate_policy(memory)
    pipeline = ActionPipeline()

    plan = pipeline.compile(
        PlannerDecision(
            action_type="tool_call",
            action="camera_pose_to_world",
            parameters={
                "camera_pose": {"id": "grasp_001", "frame": "camera"},
                "camera_extrinsics": {"pos": [0, 0, 0], "mat": [1, 0, 0, 0, 1, 0, 0, 0, 1]},
            },
        ),
        observation=_observation(),
        tools=bind_dummy_tool_handlers(build_default_tool_registry()),
        skills=build_default_skill_registry(),
        memory=memory,
    )

    assert plan.status.value == "blocked"
    assert "require compile_grasp_seed" in plan.tool_calls[0].reason
    assert memory.anygrasp_candidate_policy()["active_candidate"]["id"] == "grasp_000"


def test_failed_pre_safety_check_advances_anygrasp_candidate() -> None:
    tools = bind_dummy_tool_handlers(build_default_tool_registry())

    def unsafe_ik(context: ToolExecutionContext) -> ToolResult:
        return ToolResult(
            False,
            content="IK target is infeasible",
            details={"feasible": False, "reason": "outside_workspace"},
        )

    tools.bind_handler("ik_preview_check", unsafe_ik, replace=True)
    required_parameters = {
        "target_pose": {
            "frame": "world",
            "xyz": [0.1, 0.2, 0.3],
            "source_grasp_id": "grasp_000",
            "compiled_grasp_id": "compiled-000",
            "scene_epoch": 0,
        }
    }
    runtime = OpenEtaAgentRuntime(
        planner=ToolCallingPlanner(
            StaticPlannerBackend(
                {
                    "kind": "tool_call",
                    "name": "move_to",
                    "parameters": required_parameters,
                }
            )
        ),
        tools=tools,
        pipeline=ActionPipeline(
            checker_subagents=CheckerSubagentConfig(
                pre_safety_checks={"move_to": "ik_preview_check"}
            )
        ),
    )
    runtime.start_session(task="pick cube")
    _record_anygrasp_candidate_policy(runtime.memory)
    runtime.memory.save_fact(
        "grasp_execution",
        {
            "schema_version": "openeta.grasp_execution.v1",
            "status": "required",
            "stage": "hover",
            "candidate_id": "grasp_000",
            "scene_epoch": 0,
            "required_action": {"name": "move_to", "parameters": required_parameters},
        },
        source="unit",
    )

    action = runtime.act(_observation())

    assert action.command["status"] == "blocked"
    policy = runtime.memory.anygrasp_candidate_policy()
    assert policy["active_candidate"]["id"] == "grasp_001"
    assert policy["active_rank"] == 1
    assert policy["rejected_candidates"][0]["candidate_id"] == "grasp_000"
    assert policy["rejected_candidates"][0]["source"] == "safety_check_rejected"


@pytest.mark.parametrize("review_decision", ["reject", "abstain"])
def test_independent_precontact_review_denial_advances_anygrasp_candidate(
    review_decision: str,
) -> None:
    memory = AgentMemory()
    memory.start_session(task="pick alphabet soup")
    _record_anygrasp_candidate_policy(memory)
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "move_to",
                    "parameters": {
                        "target_pose": {
                            "frame": "world",
                            "grasp_stage": "hover",
                            "source_grasp_id": "grasp_000",
                            "xyz": [0.1, 0.2, 0.3],
                        }
                    },
                },
                "status": "failed",
                "tool_calls": [
                    {
                        "name": "move_to",
                        "status": "failed",
                        "result": {
                            "success": False,
                            "content": "The hover targets the milk carton.",
                            "details": {
                                "outputs": {
                                    "supervision": {
                                        "allowed": False,
                                        "source": "independent_reviewer",
                                        "reason": "The hover targets the milk carton.",
                                        "details": {
                                            "decision": review_decision,
                                            "grasp_outcome": "not_assessed",
                                            "candidate_id": "",
                                        },
                                    }
                                },
                                "diagnostics": [{"code": "supervision_denied"}],
                            },
                        },
                    }
                ],
            },
        )
    )

    policy = memory.anygrasp_candidate_policy()
    assert policy["active_candidate"]["id"] == "grasp_001"
    assert policy["active_rank"] == 1
    assert policy["rejected_candidates"][0]["source"] == ("independent_precontact_review_rejected")
    assert policy["rejected_candidates"][0]["reason"] == ("The hover targets the milk carton.")


def test_structured_uncertain_review_exhaustion_triggers_refinement() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick alphabet soup")
    _record_anygrasp_candidate_policy(memory, camera_frame_id="agentview")
    policy = memory.grasp_candidate_policy()
    policy["fallback_target_prompt"] = "alphabet soup"
    memory.save_fact("grasp_candidate_policy", policy, source="test")

    for _ in range(2):
        active = memory.grasp_candidate_policy()["active_candidate"]
        parameters = {
            "target_pose": {
                "frame": "world",
                "grasp_stage": "hover",
                "source_grasp_id": active["id"],
                "xyz": [0.1, 0.2, 0.3],
            }
        }
        memory.add_action(
            EnvAction(
                action_type="tool_call",
                command={
                    "request": {
                        "kind": "tool_call",
                        "name": "move_to",
                        "parameters": parameters,
                    },
                    "status": "failed",
                    "tool_calls": [
                        {
                            "name": "move_to",
                            "status": "failed",
                            "result": {
                                "success": False,
                                "content": "The distant view is too occluded to review.",
                                "details": {
                                    "outputs": {
                                        "supervision": {
                                            "allowed": False,
                                            "source": "independent_reviewer",
                                            "reason": (
                                                "The distant view is too occluded to review."
                                            ),
                                            "details": {
                                                "decision": "reject",
                                                "grasp_outcome": "not_assessed",
                                                "recovery_class": "uncertain_review",
                                            },
                                        }
                                    },
                                    "diagnostics": [{"code": "supervision_denied"}],
                                },
                            },
                        }
                    ],
                },
            )
        )

    policy = memory.grasp_candidate_policy()
    assert policy["status"] == "exhausted"
    assert policy["exhaustion_reason"] == "uncertain_review"
    recovery = memory.grasp_estimation_recovery()
    assert recovery["status"] == "required"
    assert recovery["trigger_class"] == "uncertain_review"


def test_host_descend_review_abstention_advances_anygrasp_candidate() -> None:
    """Regression for the Object0 pre-contact abstention deadlock."""

    memory = AgentMemory()
    memory.start_session(task="pick alphabet soup")
    _record_anygrasp_candidate_policy(memory)
    required_parameters = {
        "target_pose": {
            "frame": "world",
            "grasp_stage": "contact",
            "source_grasp_id": "grasp_000",
            "xyz": [0.1, 0.2, 0.1],
        }
    }
    memory.save_fact(
        "grasp_execution",
        {
            "schema_version": "openeta.grasp_execution.v1",
            "status": "required",
            "stage": "descend",
            "candidate_id": "grasp_000",
            "required_action": {
                "name": "move_to",
                "parameters": required_parameters,
            },
        },
        source="test",
    )

    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "move_to",
                    "parameters": required_parameters,
                },
                "status": "failed",
                "tool_calls": [
                    {
                        "name": "move_to",
                        "status": "failed",
                        "result": {
                            "success": False,
                            "content": (
                                "The target appears laterally offset, so geometric "
                                "support is ambiguous."
                            ),
                            "details": {
                                "outputs": {
                                    "supervision": {
                                        "allowed": False,
                                        "source": "independent_reviewer",
                                        "reason": (
                                            "The target appears laterally offset, so "
                                            "geometric support is ambiguous."
                                        ),
                                        "details": {
                                            "decision": "abstain",
                                            "grasp_outcome": "not_assessed",
                                            "candidate_id": "",
                                        },
                                    }
                                },
                                "diagnostics": [
                                    {
                                        "code": "supervision_denied",
                                        "source": "independent_reviewer",
                                    }
                                ],
                            },
                        },
                    }
                ],
            },
        )
    )

    policy = memory.anygrasp_candidate_policy()
    assert policy["candidate_attempt_count"] == 1
    assert policy["active_candidate"]["id"] == "grasp_001"
    assert policy["active_rank"] == 1
    assert policy["rejected_candidates"][0]["source"] == (
        "independent_host_stage_review_rejected"
    )
    assert policy["last_rejection"]["grasp_stage"] == "descend"
    assert memory.grasp_execution() is None


def test_host_close_review_rejection_is_attributed_to_active_candidate() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick alphabet soup")
    _record_anygrasp_candidate_policy(memory)
    policy = memory.anygrasp_candidate_policy()
    policy["status"] = "accepted"
    policy["accepted_candidate"] = dict(policy["active_candidate"])
    memory.save_fact("anygrasp_candidate_policy", policy, source="test")
    memory.save_fact(
        "grasp_execution",
        {
            "schema_version": "openeta.grasp_execution.v1",
            "status": "required",
            "stage": "close",
            "candidate_id": "grasp_000",
            "required_action": {
                "name": "gripper_control",
                "parameters": {"position": 0},
            },
        },
        source="test",
    )

    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "gripper_control",
                    "parameters": {"position": 0},
                },
                "status": "failed",
                "tool_calls": [
                    {
                        "name": "gripper_control",
                        "status": "failed",
                        "result": {
                            "success": False,
                            "content": "The active candidate targets the wrong object.",
                            "details": {
                                "outputs": {
                                    "supervision": {
                                        "allowed": False,
                                        "source": "independent_reviewer",
                                        "reason": (
                                            "The active candidate targets the wrong object."
                                        ),
                                        "details": {
                                            "decision": "reject",
                                            "grasp_outcome": "not_assessed",
                                            "candidate_id": "",
                                        },
                                    }
                                },
                                "diagnostics": [{"code": "supervision_denied"}],
                            },
                        },
                    }
                ],
            },
        )
    )

    policy = memory.anygrasp_candidate_policy()
    assert policy["status"] == "active"
    assert policy["active_candidate"]["id"] == "grasp_001"
    assert "accepted_candidate" not in policy
    assert policy["rejected_candidates"][0]["source"] == ("independent_host_stage_review_rejected")
    assert policy["last_rejection"]["grasp_stage"] == "close"
    assert memory.grasp_execution() is None


def test_candidate_width_compile_failure_advances_anygrasp_candidate() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick alphabet soup")
    _record_anygrasp_candidate_policy(memory)
    active = memory.anygrasp_candidate_policy()["active_candidate"]

    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "compile_grasp_seed",
                    "parameters": {
                        "camera_pose": dict(active),
                        "camera_extrinsics": {"mat": [1.0] * 9, "pos": [0.0] * 3},
                        "camera_frame_id": "agentview",
                        "scene_epoch": 0,
                        "target_class": "upright_can",
                    },
                },
                "status": "failed",
                "tool_calls": [
                    {
                        "name": "compile_grasp_seed",
                        "status": "failed",
                        "result": {
                            "success": False,
                            "content": (
                                "grasp seed compilation failed: candidate width "
                                "0.0796 m is outside restricted bounds [0.0200, 0.0750]"
                            ),
                            "details": {
                                "diagnostics": [
                                    {
                                        "code": "grasp_seed_compile_failed",
                                        "error_type": "GraspGeometryError",
                                        "message": (
                                            "candidate width 0.0796 m is outside "
                                            "restricted bounds [0.0200, 0.0750]"
                                        ),
                                    }
                                ],
                                "outputs": {"reason": "grasp_seed_compile_failed"},
                            },
                        },
                    }
                ],
            },
        )
    )

    policy = memory.anygrasp_candidate_policy()
    assert policy["active_candidate"]["id"] == "grasp_001"
    assert policy["active_rank"] == 1
    assert policy["rejected_candidates"][0]["source"] == ("grasp_seed_geometry_rejected")


def test_structured_strategy_filter_exhaustion_triggers_perception_refinement() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick bowl")
    _record_anygrasp_candidate_policy(memory, camera_frame_id="agentview")
    policy = memory.grasp_candidate_policy()
    policy["fallback_target_prompt"] = "bowl"
    memory.save_fact("grasp_candidate_policy", policy, source="test")

    for _ in range(2):
        active = memory.anygrasp_candidate_policy()["active_candidate"]
        parameters = {
            "camera_pose": dict(active),
            "camera_extrinsics": {"mat": [1.0] * 9, "pos": [0.0] * 3},
            "camera_frame_id": "agentview",
            "scene_epoch": 0,
            "target_class": "bowl",
        }
        memory.add_action(
            EnvAction(
                action_type="tool_call",
                command={
                    "request": {
                        "kind": "tool_call",
                        "name": "compile_grasp_seed",
                        "parameters": parameters,
                    },
                    "status": "failed",
                    "tool_calls": [
                        {
                            "name": "compile_grasp_seed",
                            "status": "failed",
                            "result": {
                                "success": False,
                                "content": "grasp seed candidate rejected",
                                "details": {
                                    "diagnostics": [
                                        {
                                            "code": "grasp_seed_candidate_rejected",
                                            "candidate_rejection": True,
                                            "candidate_id": active["id"],
                                            "message": (
                                                "native approach below strategy minimum"
                                            ),
                                            "rejection_code": (
                                                "strategy_alignment_rejected"
                                            ),
                                            "recovery_class": "perception_refinable",
                                        }
                                    ],
                                    "outputs": {
                                        "reason": "grasp_seed_candidate_rejected",
                                        "candidate_rejection": True,
                                        "candidate_id": active["id"],
                                        "rejection_code": "strategy_alignment_rejected",
                                        "recovery_class": "perception_refinable",
                                    },
                                },
                            },
                        }
                    ],
                },
            )
        )

    policy = memory.anygrasp_candidate_policy()
    assert policy["status"] == "exhausted"
    assert policy["fallback_required"] is True
    assert policy["rejected_candidates"][0]["source"] == (
        "grasp_seed_geometry_rejected"
    )
    assert policy["rejected_candidates"][0]["recovery_class"] == "perception_refinable"
    recovery = memory.grasp_estimation_recovery()
    assert recovery["status"] == "required"
    assert recovery["trigger_class"] == "perception_refinable"
    assert recovery["seed_candidate"]["id"] == "grasp_000"


def test_candidate_specific_motion_rejection_advances_then_selects_score_fallback() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube")
    _record_anygrasp_candidate_policy(memory)

    def reject(candidate_id: str) -> None:
        memory.add_action(
            EnvAction(
                action_type="tool_call",
                command={
                    "request": {
                        "kind": "tool_call",
                        "name": "move_to",
                        "parameters": {
                            "target_pose": {
                                "id": candidate_id,
                                "frame": "world",
                                "translation_xyz": [0.1, 0.2, 0.3],
                            },
                        },
                    },
                    "status": "failed",
                    "safety_checks": [],
                    "tool_calls": [
                        {
                            "name": "move_to",
                            "status": "failed",
                            "result": {
                                "success": False,
                                "content": "motion collision",
                                "details": {
                                    "diagnostics": [
                                        {
                                            "code": "grasp_candidate_collision",
                                            "candidate_rejection": True,
                                        }
                                    ]
                                },
                            },
                        }
                    ],
                },
            )
        )

    reject("grasp_000")
    assert memory.anygrasp_candidate_policy()["active_candidate"]["id"] == "grasp_001"
    reject("grasp_001")

    policy = memory.anygrasp_candidate_policy()
    assert policy["status"] == "active"
    assert policy["candidate_fallback"] is True
    assert policy["active_candidate"]["id"] == "grasp_000"
    assert policy["active_candidate"]["candidate_fallback"] is True
    assert len(policy["rejected_candidates"]) == 2
    assert memory.grasp_estimation_recovery() is None
    assert "All AnyGrasp candidates" in memory.grasp_candidate_gate_error(
        tool_name="camera_pose_to_world",
        parameters={"camera_pose": {"id": "grasp_000"}},
    )


def test_motion_rejection_requires_fresh_observation_for_alternate_view_reestimate() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube")
    _record_anygrasp_candidate_policy(memory)
    active = memory.anygrasp_candidate_policy()["active_candidate"]
    required_parameters = {
        "target_pose": {
            "id": active["id"],
            "frame": "world",
            "xyz": [0.30, 0.20, 0.05],
        }
    }
    memory.save_fact(
        "grasp_execution",
        {
            "schema_version": "openeta.grasp_execution.v1",
            "status": "required",
            "stage": "descend",
            "candidate_id": active["id"],
            "required_action": {
                "name": "move_to",
                "parameters": required_parameters,
            },
        },
        source="test",
    )
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "move_to",
                    "parameters": required_parameters,
                },
                "status": "executed",
                "tool_calls": [
                    {
                        "name": "move_to",
                        "status": "executed",
                        "result": {
                            "success": True,
                            "content": "target not reached",
                            "details": {
                                "parameters": required_parameters,
                                "outputs": {
                                    "motion_summary": {
                                        "reached_target": False,
                                        "end": {"xyz": [0.10, 0.20, 0.04]},
                                    }
                                },
                            },
                        },
                    }
                ],
            },
        )
    )

    assert memory.grasp_recovery() is None
    assert memory.anygrasp_candidate_policy()["active_candidate"]["id"] == "grasp_001"
    return

def test_unclassified_motion_collision_keeps_active_anygrasp_candidate() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube")
    _record_anygrasp_candidate_policy(memory)
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "move_to",
                    "parameters": {"target_pose": {"id": "grasp_000"}},
                },
                "status": "failed",
                "safety_checks": [],
                "tool_calls": [
                    {
                        "name": "move_to",
                        "status": "failed",
                        "result": {
                            "success": False,
                            "content": "motion collision",
                            "details": {"reason": "collision"},
                        },
                    }
                ],
            },
        )
    )

    policy = memory.anygrasp_candidate_policy()
    assert policy["active_candidate"]["id"] == "grasp_000"
    assert policy["rejected_candidates"] == []


def test_successful_motion_accepts_policy_and_releases_later_motion_gate() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube")
    _record_anygrasp_candidate_policy(memory)
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "move_to",
                    "parameters": {
                        "target_pose": {
                            "id": "grasp_000",
                            "frame": "world",
                            "translation_xyz": [0.1, 0.2, 0.3],
                        },
                    },
                },
                "status": "executed",
                "safety_checks": [],
                "tool_calls": [
                    {
                        "name": "move_to",
                        "status": "executed",
                        "result": {"success": True, "content": "target reached"},
                    }
                ],
            },
        )
    )

    policy = memory.anygrasp_candidate_policy()
    assert policy["status"] == "accepted"
    assert policy["accepted_candidate"]["id"] == "grasp_000"
    assert (
        memory.grasp_candidate_gate_error(
            tool_name="move_to",
            parameters={"target_pose": {"xyz": [0.4, 0.0, 0.5]}},
        )
        is None
    )


def test_independent_failed_grasp_outcome_advances_accepted_candidate() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube")
    _record_anygrasp_candidate_policy(memory)
    active = memory.anygrasp_candidate_policy()["active_candidate"]
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "move_to",
                    "parameters": {"target_pose": active},
                },
                "status": "executed",
                "safety_checks": [],
                "tool_calls": [
                    {
                        "name": "move_to",
                        "status": "executed",
                        "result": {"success": True, "content": "target reached"},
                    }
                ],
            },
        )
    )
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
                        "result": {"success": True, "content": "gripper closed"},
                    }
                ],
            },
        )
    )
    memory.add_observation(
        EnvObservation(
            task="pick cube",
            cameras=[],
            robot=RobotState(
                end_effector_pose={"xyz": [0.18, 0.02, 0.06]},
                gripper_state={"open": False, "openness": 0.2},
            ),
        )
    )
    probe = memory.grasp_lift_probe()
    assert probe["status"] == "required"
    assert probe["required_parameters"]["target_pose"]["xyz"] == pytest.approx([0.18, 0.02, 0.14])
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "move_to",
                    "parameters": probe["required_parameters"],
                },
                "status": "executed",
                "tool_calls": [
                    {
                        "name": "move_to",
                        "status": "executed",
                        "result": {"success": True, "content": "lift target reached"},
                    }
                ],
            },
        )
    )
    assert memory.grasp_lift_probe()["status"] == "completed"
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
                "safety_checks": [],
                "tool_calls": [
                    {
                        "name": "gripper_control",
                        "status": "executed",
                        "result": {
                            "success": True,
                            "content": "gripper opened",
                            "details": {
                                "supervision": {
                                    "allowed": True,
                                    "source": "independent_reviewer",
                                    "reason": "The target stayed on the table.",
                                    "details": {
                                        "grasp_outcome": "failed",
                                        "candidate_id": "grasp_000",
                                    },
                                }
                            },
                        },
                    }
                ],
            },
        )
    )

    policy = memory.anygrasp_candidate_policy()
    assert policy["status"] == "active"
    assert policy["active_candidate"]["id"] == "grasp_001"
    assert "accepted_candidate" not in policy
    assert policy["rejected_candidates"][0]["source"] == ("independent_grasp_outcome_rejected")


def test_denied_placement_motion_advances_candidate_after_failed_grasp_review() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube")
    _record_anygrasp_candidate_policy(memory)
    active = memory.anygrasp_candidate_policy()["active_candidate"]
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "move_to",
                    "parameters": {"target_pose": active},
                },
                "status": "executed",
                "tool_calls": [
                    {
                        "name": "move_to",
                        "status": "executed",
                        "result": {"success": True, "content": "target reached"},
                    }
                ],
            },
        )
    )
    memory.artifacts["anyplace_placement_candidates_latest"] = {"value": {}}
    memory.artifacts["camera_pose_to_world_world_pose_latest"] = {"value": {}}
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
                        "result": {"success": True, "content": "gripper closed"},
                    }
                ],
            },
        )
    )
    memory.add_observation(
        EnvObservation(
            task="pick cube",
            cameras=[],
            robot=RobotState(
                end_effector_pose={"xyz": [0.18, 0.02, 0.06]},
                gripper_state={"open": False, "openness": 0.2},
            ),
        )
    )
    probe = memory.grasp_lift_probe()
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "move_to",
                    "parameters": probe["required_parameters"],
                },
                "status": "executed",
                "tool_calls": [
                    {
                        "name": "move_to",
                        "status": "executed",
                        "result": {"success": True, "content": "lift target reached"},
                    }
                ],
            },
        )
    )
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "move_to",
                    "parameters": {
                        "target_pose": {
                            "frame": "world",
                            "translation_xyz": [0.1, 0.1, 0.25],
                        }
                    },
                },
                "status": "failed",
                "tool_calls": [
                    {
                        "name": "move_to",
                        "status": "failed",
                        "result": {
                            "success": False,
                            "content": "The target stayed on the source surface.",
                            "details": {
                                "outputs": {
                                    "supervision": {
                                        "allowed": False,
                                        "source": "independent_reviewer",
                                        "reason": "The target stayed on the source surface.",
                                        "details": {
                                            "decision": "reject",
                                            "grasp_outcome": "fail",
                                            "candidate_id": "grasp_000",
                                        },
                                    }
                                },
                                "diagnostics": [{"code": "supervision_denied"}],
                            },
                        },
                    }
                ],
            },
        )
    )

    policy = memory.anygrasp_candidate_policy()
    assert policy["status"] == "active"
    assert policy["active_candidate"]["id"] == "grasp_001"
    assert "accepted_candidate" not in policy
    assert policy["rejected_candidates"][0]["source"] == ("independent_grasp_outcome_rejected")
    assert policy["last_rejection"]["target_tool"] == "move_to"
    assert memory.grasp_lift_probe() is None
    assert memory.grasp_execution() is None
    assert memory.grasp_recovery() is None
    assert "anyplace_placement_candidates_latest" not in memory.artifacts
    assert "camera_pose_to_world_world_pose_latest" not in memory.artifacts


def test_failed_grasp_review_cannot_advance_candidate_before_lift_probe() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube")
    _record_anygrasp_candidate_policy(memory)
    active = memory.anygrasp_candidate_policy()["active_candidate"]
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "move_to",
                    "parameters": {"target_pose": active},
                },
                "status": "executed",
                "tool_calls": [
                    {
                        "name": "move_to",
                        "status": "executed",
                        "result": {"success": True, "content": "target reached"},
                    }
                ],
            },
        )
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
                        "result": {
                            "success": True,
                            "details": {
                                "supervision": {
                                    "reason": "Static post-close image looked uncertain.",
                                    "details": {
                                        "grasp_outcome": "failed",
                                        "candidate_id": "grasp_000",
                                    },
                                }
                            },
                        },
                    }
                ],
            },
        )
    )

    policy = memory.anygrasp_candidate_policy()
    assert policy["status"] == "accepted"
    assert policy["active_candidate"]["id"] == "grasp_000"
    assert policy["rejected_candidates"] == []


def test_planner_retries_gripper_open_as_exact_required_lift_probe() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube")
    _record_anygrasp_candidate_policy(memory)
    active = memory.anygrasp_candidate_policy()["active_candidate"]
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "move_to",
                    "parameters": {"target_pose": active},
                },
                "status": "executed",
                "tool_calls": [
                    {
                        "name": "move_to",
                        "status": "executed",
                        "result": {"success": True, "content": "target reached"},
                    }
                ],
            },
        )
    )
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
                        "result": {"success": True, "content": "gripper closed"},
                    }
                ],
            },
        )
    )
    observation = EnvObservation(
        task="pick cube",
        cameras=[],
        robot=RobotState(
            end_effector_pose={"xyz": [0.18, 0.02, 0.06]},
            gripper_state={"open": False, "openness": 0.2},
        ),
    )
    memory.add_observation(observation)
    required = memory.grasp_lift_probe()["required_parameters"]
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            [
                {
                    "kind": "tool_call",
                    "name": "gripper_control",
                    "parameters": {"position": 1},
                },
                {
                    "kind": "tool_call",
                    "name": "move_to",
                    "parameters": required,
                },
            ]
        ),
        max_validation_retries=1,
    )

    decision = planner.plan(
        observation,
        memory=memory,
        tools=_tools_with_handlers("move_to", "gripper_control"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "move_to"
    assert decision.parameters == required
    first_errors = decision.metadata["validation_attempt_history"][0]["validation_errors"]
    assert any("requires the fixed lift probe" in error for error in first_errors)


def test_structured_target_not_reached_advances_candidate_despite_success_envelope() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube")
    _record_anygrasp_candidate_policy(memory)

    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "move_to",
                    "parameters": {"target_pose": {"id": "grasp_000"}},
                },
                "status": "executed",
                "safety_checks": [],
                "tool_calls": [
                    {
                        "name": "move_to",
                        "status": "executed",
                        "result": {
                            "success": True,
                            "content": "legacy simulator success envelope",
                            "details": {
                                "outputs": {
                                    "response": {"motion_summary": {"reached_target": False}}
                                }
                            },
                        },
                    }
                ],
            },
        )
    )

    policy = memory.anygrasp_candidate_policy()
    assert policy["status"] == "active"
    assert policy["active_rank"] == 1
    assert policy["active_candidate"]["id"] == "grasp_001"
    assert policy["rejected_candidates"][0]["reason"] == (
        "Simulator motion summary reports that the target was not reached."
    )


def test_unrelated_tool_failure_does_not_advance_anygrasp_candidate() -> None:
    memory = AgentMemory()
    memory.start_session(task="pick cube")
    _record_anygrasp_candidate_policy(memory)
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "gripper_control",
                    "parameters": {"position": 0},
                },
                "status": "failed",
                "safety_checks": [],
                "tool_calls": [
                    {
                        "name": "gripper_control",
                        "status": "failed",
                        "result": {"success": False, "content": "gripper timeout"},
                    }
                ],
            },
        )
    )
    memory.add_action(
        EnvAction(
            action_type="tool_call",
            command={
                "request": {
                    "kind": "tool_call",
                    "name": "move_to",
                    "parameters": {
                        "target_pose": {
                            "id": "grasp_000",
                            "frame": "world",
                            "translation_xyz": [0.1, 0.2, 0.3],
                        }
                    },
                },
                "status": "failed",
                "safety_checks": [],
                "tool_calls": [
                    {
                        "name": "move_to",
                        "status": "failed",
                        "result": {
                            "success": False,
                            "content": "MCP transport timeout",
                            "details": {"reason": "mcp_call_failed"},
                        },
                    }
                ],
            },
        )
    )

    policy = memory.anygrasp_candidate_policy()
    assert policy["active_candidate"]["id"] == "grasp_000"
    assert policy["rejected_candidates"] == []


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


def test_runtime_selection_tool_resolves_obligation_and_unblocks_anygrasp() -> None:
    tools = bind_dummy_tool_handlers(build_default_tool_registry())
    runtime = OpenEtaAgentRuntime(tools=tools)
    runtime.start_session(task="pick alphabet soup")
    _record_pending_sam3_selection(runtime.memory)

    context = build_tool_context(
        observation=_observation(),
        memory=runtime.memory,
        tools=runtime.tools,
        skills=runtime.skills,
    )
    assert context["selection_obligation"]["result_id"] == "sam3-run-selection"
    blocked = runtime.pipeline.compile(
        PlannerDecision(
            action_type="tool_call",
            action="anygrasp",
            parameters={
                "mode": "targeted",
                "rgb": "rgb.png",
                "depth": "depth.png",
                "intrinsics": {"fx": 1, "fy": 1, "cx": 1, "cy": 1, "scale": 1},
                "target_mask": "tmp/mask_000.png",
            },
        ),
        observation=_observation(),
        tools=runtime.tools,
        skills=runtime.skills,
        memory=runtime.memory,
    )
    assert blocked.status.value == "blocked"
    assert blocked.tool_calls[0].status.value == "skipped"
    scene_mode = runtime.pipeline.compile(
        PlannerDecision(
            action_type="tool_call",
            action="anygrasp",
            parameters={
                "mode": "scene",
                "rgb": "rgb.png",
                "depth": "depth.png",
                "intrinsics": {"fx": 1, "fy": 1, "cx": 1, "cy": 1, "scale": 1},
            },
        ),
        observation=_observation(),
        tools=runtime.tools,
        skills=runtime.skills,
        memory=runtime.memory,
    )
    assert scene_mode.status.value == "executed"
    blocked_motion = runtime.pipeline.compile(
        PlannerDecision(
            action_type="tool_call",
            action="move_to",
            parameters={"target_pose": {"xyz": [0.1, 0.2, 0.3]}},
        ),
        observation=_observation(),
        tools=runtime.tools,
        skills=runtime.skills,
        memory=runtime.memory,
    )
    assert blocked_motion.status.value == "blocked"

    selected = runtime.pipeline.compile(
        PlannerDecision(
            action_type="tool_call",
            action="select_sam3_detection",
            parameters={
                "sam3_result_id": "sam3-run-selection",
                "detection_id": "detection_001",
                "selection_confidence": 0.84,
                "target_geometry_family": "upright_can",
                "reason": "The crop matches the alphabet soup package.",
            },
        ),
        observation=_observation(),
        tools=runtime.tools,
        skills=runtime.skills,
        memory=runtime.memory,
    )
    assert selected.status.value == "executed"
    assert runtime.memory.pending_sam3_selection() is None
    resolved = runtime.memory.selected_sam3_detection()
    assert resolved["id"] == "detection_001"
    assert resolved["selection_source"] == "main_agent_vlm"
    assert resolved["target_geometry_family"] == "upright_can"

    wrong_mask = runtime.pipeline.compile(
        PlannerDecision(
            action_type="tool_call",
            action="anygrasp",
            parameters={
                "mode": "targeted",
                "rgb": "rgb.png",
                "depth": "depth.png",
                "intrinsics": {"fx": 1, "fy": 1, "cx": 1, "cy": 1, "scale": 1},
                "target_mask": "tmp/mask_000.png",
            },
        ),
        observation=_observation(),
        tools=runtime.tools,
        skills=runtime.skills,
        memory=runtime.memory,
    )
    assert wrong_mask.status.value == "blocked"

    allowed = runtime.pipeline.compile(
        PlannerDecision(
            action_type="tool_call",
            action="anygrasp",
            parameters={
                "mode": "targeted",
                "rgb": "rgb.png",
                "depth": "depth.png",
                "intrinsics": {"fx": 1, "fy": 1, "cx": 1, "cy": 1, "scale": 1},
                "target_mask": "tmp/mask_001.png",
            },
        ),
        observation=_observation(),
        tools=runtime.tools,
        skills=runtime.skills,
        memory=runtime.memory,
    )
    assert allowed.status.value == "executed"


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


def test_runtime_selection_gate_applies_to_graspgenx_mask_artifact() -> None:
    tools = bind_dummy_tool_handlers(build_default_tool_registry())
    tools.bind_handler(
        "graspgenx",
        lambda context: ToolResult(
            True,
            content="grasp candidates generated",
            details={"grasp_candidates": [{"id": "graspgenx_000", "score": 0.9}]},
        ),
    )
    runtime = OpenEtaAgentRuntime(tools=tools)
    runtime.start_session(task="pick alphabet soup with GraspGenX")
    _record_pending_sam3_selection(runtime.memory)
    base_parameters = {
        "rgb": "rgb.png",
        "depth": "depth.png",
        "intrinsics": {"fx": 1, "fy": 1, "cx": 1, "cy": 1, "scale": 1},
        "gripper_name": "franka_panda",
        "up_direction_camera": [0.0, 0.0, -1.0],
    }

    blocked = runtime.pipeline.compile(
        PlannerDecision(
            action_type="tool_call",
            action="graspgenx",
            parameters={
                **base_parameters,
                "object_mask": {
                    "mask_ref": "tmp/mask_001.png",
                    "source_image": "rgb.png",
                },
            },
        ),
        observation=_observation(),
        tools=runtime.tools,
        skills=runtime.skills,
        memory=runtime.memory,
    )
    assert blocked.status.value == "blocked"

    runtime.memory.resolve_sam3_selection(
        result_id="sam3-run-selection",
        detection_id="detection_001",
        selection_source="main_agent_vlm",
        confidence=0.9,
    )
    wrong = runtime.pipeline.compile(
        PlannerDecision(
            action_type="tool_call",
            action="graspgenx",
            parameters={
                **base_parameters,
                "object_mask": {
                    "mask_ref": "tmp/mask_000.png",
                    "source_image": "rgb.png",
                },
            },
        ),
        observation=_observation(),
        tools=runtime.tools,
        skills=runtime.skills,
        memory=runtime.memory,
    )
    assert wrong.status.value == "blocked"

    allowed = runtime.pipeline.compile(
        PlannerDecision(
            action_type="tool_call",
            action="graspgenx",
            parameters={
                **base_parameters,
                "object_mask": {
                    "mask_ref": "tmp/mask_001.png",
                    "source_image": "rgb.png",
                },
            },
        ),
        observation=_observation(),
        tools=runtime.tools,
        skills=runtime.skills,
        memory=runtime.memory,
    )
    assert allowed.status.value == "executed"


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


def test_planner_retries_pending_anygrasp_as_explicit_detection_selection() -> None:
    tools = bind_dummy_tool_handlers(build_default_tool_registry())
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            [
                {
                    "kind": "tool_call",
                    "name": "anygrasp",
                    "parameters": {
                        "mode": "targeted",
                        "rgb": "rgb.png",
                        "depth": "depth.png",
                        "intrinsics": {"fx": 1, "fy": 1, "cx": 1, "cy": 1, "scale": 1},
                        "target_mask": "tmp/mask_000.png",
                    },
                },
                {
                    "kind": "tool_call",
                    "name": "select_sam3_detection",
                    "parameters": {
                        "sam3_result_id": "sam3-run-selection",
                        "detection_id": "detection_001",
                        "selection_confidence": 0.9,
                        "reason": "Visual package match.",
                    },
                },
            ]
        ),
        max_validation_retries=1,
    )
    runtime = OpenEtaAgentRuntime(planner=planner, tools=tools)
    runtime.start_session(task="pick alphabet soup")
    _record_pending_sam3_selection(runtime.memory)

    action = runtime.act(_observation())

    assert action.command["request"]["name"] == "select_sam3_detection"
    assert action.command["status"] == "executed"
    assert runtime.memory.selected_sam3_detection()["id"] == "detection_001"


def test_planner_cannot_overwrite_pending_selection_with_another_sam3() -> None:
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            [
                {
                    "kind": "tool_call",
                    "name": "sam3",
                    "parameters": {
                        "image": "new-scene.png",
                        "prompt": "alphabet soup can",
                    },
                },
                {
                    "kind": "tool_call",
                    "name": "select_sam3_detection",
                    "parameters": {
                        "sam3_result_id": "sam3-run-selection",
                        "detection_id": "detection_001",
                        "selection_confidence": 0.9,
                        "reason": "The verified package appearance matches.",
                    },
                },
            ]
        ),
        max_validation_retries=1,
    )
    memory = AgentMemory()
    memory.start_session(task="pick alphabet soup")
    _record_pending_sam3_selection(memory)

    decision = planner.plan(
        _observation(),
        memory=memory,
        tools=_tools_with_handlers("sam3", "select_sam3_detection"),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "select_sam3_detection"
    history = decision.metadata["validation_attempt_history"]
    assert "do not overwrite it with another SAM3 request" in history[0]["validation_errors"][0]
    assert history[1]["validation_errors"] == []


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
    assert "move_to.target_pose" in pose_artifact["next_tool_hint"]
    assert "without changing" in pose_artifact["next_tool_hint"]


def test_dummy_tool_handlers_return_standard_result_envelopes() -> None:
    tools = bind_dummy_tool_handlers(build_default_tool_registry())

    perception = tools.call(
        "sam3",
        {"image": "front", "prompt": "cube"},
        observation=_observation(),
    )
    planning = tools.call(
        "anygrasp",
        {
            "rgb": "front-rgb.png",
            "depth": "front-depth.png",
            "target_mask": "cube-mask.png",
            "intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0, "scale": 1000.0},
        },
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
        {"image": "front-rgb.png", "prompt": "cube"},
        observation=_observation(),
    )

    assert result.details["schema_version"] == TOOL_RESULT_SCHEMA_VERSION
    assert result.details["outputs"]["detections"][0]["mask_ref"] == "cube-mask.png"
    assert result.details["artifacts"][0]["path"] == "cube-mask.png"


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
    pipeline = ActionPipeline(
        checker_subagents=CheckerSubagentConfig(pre_safety_checks={"move_to": "ik_preview_check"})
    )
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {
                "kind": "tool_call",
                "name": "move_to",
                "parameters": {"target_pose": {"xyz": [0.4, 0.0, 0.2]}},
            }
        )
    )
    runtime = OpenEtaAgentRuntime(planner=planner, tools=tools, pipeline=pipeline)
    runtime.start_session(task="move to pose")

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
                "parameters": {"target_pose": {"xyz": [9.0, 0.0, 0.2]}},
            }
        )
    )
    runtime = OpenEtaAgentRuntime(planner=planner, tools=tools, pipeline=pipeline)
    runtime.start_session(task="unsafe move")

    action = runtime.act(_observation())

    command = action.command
    assert command["status"] == "blocked"
    assert command["safety_checks"][0]["status"] == "failed"
    assert command["safety_checks"][0]["result"]["details"]["outputs"]["feasible"] is False
    assert command["tool_calls"][0]["name"] == "move_to"
    assert command["tool_calls"][0]["status"] == "skipped"


def test_pipeline_runs_post_failure_checker_after_configured_tool_call() -> None:
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
                "parameters": {"image": "front", "prompt": "cube"},
            }
        )
    )
    runtime = OpenEtaAgentRuntime(planner=planner, tools=tools, pipeline=pipeline)
    runtime.start_session(task="segment cube")

    action = runtime.act(_observation())

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


def test_pipeline_does_not_run_post_failure_checker_after_success() -> None:
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
                "parameters": {"image": "front", "prompt": "cube"},
            }
        )
    )
    runtime = OpenEtaAgentRuntime(planner=planner, tools=tools, pipeline=pipeline)
    runtime.start_session(task="segment cube")

    action = runtime.act(_observation())

    assert action.command["status"] == "executed"
    assert action.command["metadata"]["checker_results"]["post_failure_checks"] == []
    assert not any(event.event_type == "recovery_feedback" for event in runtime.memory.events)


def test_episode_runner_executes_three_closed_loop_tool_turns() -> None:
    tools = build_default_tool_registry()
    tools.bind_handler(
        "scene_detector",
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
        "anygrasp",
        lambda context: ToolResult(
            True,
            content="grasp candidates generated",
            details={
                "grasp_candidates": [{"id": "grasp-1"}],
                "target_mask": context.parameters["target_mask"],
            },
        ),
    )
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            [
                {
                    "kind": "tool_call",
                    "name": "scene_detector",
                    "parameters": {"image": "front"},
                    "reasoning": "List objects before segmentation.",
                },
                {
                    "kind": "tool_call",
                    "name": "sam3",
                    "parameters": {"image": "front", "prompt": "cube"},
                    "reasoning": "Segment the target object.",
                },
                {
                    "kind": "tool_call",
                    "name": "anygrasp",
                    "parameters": {
                        "rgb": "front-rgb.png",
                        "depth": "front-depth.png",
                        "target_mask": "cube-mask.png",
                        "intrinsics": {
                            "fx": 1.0,
                            "fy": 1.0,
                            "cx": 0.0,
                            "cy": 0.0,
                            "scale": 1000.0,
                        },
                    },
                    "reasoning": "Generate grasp candidates from RGBD inputs.",
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
        "scene_detector",
        "sam3",
        "anygrasp",
    ]
    assert [step.observation.metadata["step_idx"] for step in result.steps] == [0, 1, 2]
    assert result.steps[0].action.command["tool_calls"][0]["result"]["content"] == (
        "objects detected"
    )
    assert result.steps[2].step_result.info["previous_action"]["request_name"] == "anygrasp"
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
                    "name": "scene_detector",
                    "parameters": {"image": "front"},
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
                    "parameters": {"image": "front", "prompt": "cube"},
                },
            ]
        )
    )
    tools = build_default_tool_registry()
    tools.bind_handler(
        "scene_detector",
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
                    "name": "scene_detector",
                    "parameters": {"image": "front"},
                },
            ]
        )
    )
    tools = build_default_tool_registry()
    tools.bind_handler(
        "scene_detector",
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
    assert continued.steps[0].action.command["request"]["name"] == "scene_detector"


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
    continued = runner.continue_run(max_turns=1)

    assert continued.truncated is False
    assert continued.metadata["failure_reason"] == {}
    assert continued.metadata["usage"]["elapsed_s"] == 0.0
    assert continued.metadata["usage"]["human_wait_s"] == 120.0
    assert continued.steps[0].action.command["request"]["name"] == ("close_simulator_env")


def test_episode_runner_truncates_at_safety_turn_limit() -> None:
    planner = ToolCallingPlanner(
        StaticPlannerBackend(
            {
                "kind": "tool_call",
                "name": "scene_detector",
                "parameters": {"image": "front"},
            }
        )
    )
    tools = build_default_tool_registry()
    tools.bind_handler(
        "scene_detector",
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
                            '"parameters": {"image": "front", "prompt": "cube"}, '
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
                "selection_obligation": {
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
    ]
    assert all(
        part["image_url"]["url"].startswith("data:image/png;base64,") for part in user_content[1:]
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
                            '"parameters":{"image":"scene.png",'
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
                "reference_localization_obligation": {
                    "scene_image": str(scene),
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
        'test-local-api-key\n{"_type":"newapi_channel_conn",'
        '"key":"test-json-api-key","url":"https://open.example.test"}\n',
        encoding="utf-8",
    )

    config: PlannerProviderConfig = read_apikey_file(apikey_path)

    assert config.provider == "openai-compatible"
    assert config.api_base == "https://open.example.test"
    assert config.api_key == "test-json-api-key"
    assert config.redacted()["api_key"] != "test-json-api-key"
