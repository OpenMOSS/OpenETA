from __future__ import annotations

import json
from pathlib import Path

import pytest

from adapter.protocol import CameraFrame, EnvAction, EnvObservation, RobotState
from agent.backends.planner import (
    CallablePlannerBackend,
    OpenAICompatiblePlannerBackend,
    OpenAICompatiblePlannerBackendConfig,
    PlannerBackendRequest,
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


def _record_pending_sam3_selection(
    memory: AgentMemory,
    *,
    original_image_ref: str = "agentview.png",
    contact_sheet_ref: str = "selection.png",
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


def _record_anygrasp_candidate_policy(memory: AgentMemory) -> None:
    candidates = [
        {
            "id": "grasp_000",
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
            "id": "grasp_001",
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
                "request": {"kind": "tool_call", "name": "anygrasp", "parameters": {}},
                "status": "executed",
                "tool_calls": [
                    {
                        "name": "anygrasp",
                        "status": "executed",
                        "result": {
                            "success": True,
                            "details": {
                                "outputs": {
                                    "result_id": "anygrasp-run-001",
                                    "ranking": "score_descending",
                                    "candidate_count": 2,
                                    "grasp_candidates": candidates,
                                }
                            },
                        },
                    }
                ],
            },
        )
    )


def test_static_planner_backend_executes_registered_tool_handler() -> None:
    tools = build_default_tool_registry()

    def sam3_handler(context: ToolExecutionContext) -> ToolResult:
        assert context.name == "sam3"
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
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )

    assert decision.action_type == "response"
    assert decision.action == "talk"
    assert decision.metadata["validation_attempts"] == 2


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
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )

    assert decision.action_type == "response"
    assert decision.action == "ask_human"
    assert "Unsupported command kind" in decision.parameters["validation_errors"][0]


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
    assert decision.action == "ask_human"
    assert "Unsupported response name" in decision.parameters["validation_errors"][0]


def test_default_planner_prompt_uses_first_class_simulator_creation_tool() -> None:
    prompt = _default_tool_planner_system_prompt()

    assert "tool_call::create_simulator_env" in prompt
    assert "only environment-creation path" in prompt
    assert "Do not invoke create_env or close_env through python_exec or code_policy." in prompt


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

    assert decision.action == "ask_human"
    validation_error = decision.parameters["validation_errors"][0]
    assert "tool_call::create_simulator_env" in validation_error


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
        tools=build_default_tool_registry(),
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
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "ask_human"
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
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "contact_graspnet"
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
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )

    assert decision.action == "anyplace"
    assert decision.parameters == valid_parameters
    assert decision.metadata["validation_attempts"] == 2


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


def test_planner_blocks_world_mutation_until_truncated_primary_skill_is_inspected() -> None:
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
        tools=build_default_tool_registry(),
        skills=build_default_skill_registry(),
    )

    assert decision.action_type == "response"
    assert decision.action == "ask_human"
    assert "must be inspected" in decision.parameters["validation_errors"][0]


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
    assert "closed gripper alone is not success" in pick.content.lower()
    assert "Do not call `scene_detector` in the default pick flow" in pick.content
    assert "grasp candidate list" in pick.content
    assert pick.allowed_tools[:4] == (
        "observe",
        "scene_detector",
        "sam3",
        "select_sam3_detection",
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
    assert "Call `scene_detector`" in place["content"]
    assert "Call `gripper_control`" in place["content"]
    assert "move_to" in place["allowed_tools"]
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
            "path": "/tmp/openeta/tool_result/mask.json",
            "grep_hint": "grep -n '<pattern>' /tmp/openeta/tool_result/mask.json",
            "dashboard_url": "http://sim.example/session/session-1",
            "images": [{"path": "/tmp/openeta/image/rgb/front.png"}],
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
        "/tmp/openeta/image/rgb/front.png"
    ]
    assert context["working_memory"]["skill_notes"]["pick"][0]["note"]["failure"] == "empty mask"
    assert "facts=['target']" in summary
    assert context["working_memory"]["compact_summary"] == summary


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
    summary = context["memory"]["working_memory"]["artifacts"][
        "observe_camera_packet_agentview"
    ]
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
                                "diagnostics": [
                                    {"code": "simulator_mcp_collision"}
                                ],
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
    event = next(
        item for item in context["memory"]["recent_events"] if item["type"] == "action"
    )
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
    lines = [
        json.loads(line)
        for line in session_path.read_text(encoding="utf-8").splitlines()
    ]

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
    assert sessions[0]["metadata"]["migrated_from_layout"].endswith(
        "sessions/legacy-session.jsonl"
    )


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
    assert loaded.details["outputs"]["artifacts"]["grasp_candidates"]["value"]["id"] == (
        "grasp-1"
    )
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
    mask_ref = action_event["payload"]["command"]["tool_calls"][0]["result"]["details"][
        "outputs"
    ]["detections"][0]["mask_ref"]
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
    extracted = action_event["payload"]["command"]["tool_calls"][0]["result"]["details"][
        "outputs"
    ]["result"]
    assert extracted["intrinsics"]["scale"] == 1000.0
    assert extracted["mask_paths"][0] == "tmp/image/sam3/run/mask_000.png"


def test_planner_context_preserves_anygrasp_candidates_for_followup_motion() -> None:
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
                                    "source_rgb": "tmp/image/rgb/agentview.png",
                                    "source_depth": "tmp/image/depth/agentview.png",
                                    "target_mask": "tmp/image/sam3/mask_000.png",
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
    assert "camera_pose_to_world" in grasp_artifact["next_tool_hint"]


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
    assert "current active candidate" in plan.tool_calls[0].reason
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
    runtime = OpenEtaAgentRuntime(
        planner=ToolCallingPlanner(
            StaticPlannerBackend(
                {
                    "kind": "tool_call",
                    "name": "move_to",
                    "parameters": {
                        "target_pose": {
                            "id": "grasp_000",
                            "frame": "world",
                            "translation_xyz": [0.1, 0.2, 0.3],
                        }
                    },
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

    action = runtime.act(_observation())

    assert action.command["status"] == "blocked"
    policy = runtime.memory.anygrasp_candidate_policy()
    assert policy["active_candidate"]["id"] == "grasp_001"
    assert policy["active_rank"] == 1
    assert policy["rejected_candidates"][0]["candidate_id"] == "grasp_000"
    assert policy["rejected_candidates"][0]["source"] == "safety_check_rejected"


def test_candidate_specific_motion_rejection_advances_then_exhausts_candidates() -> None:
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
    assert policy["status"] == "exhausted"
    assert policy["active_candidate"] is None
    assert len(policy["rejected_candidates"]) == 2
    assert "All AnyGrasp candidates" in memory.grasp_candidate_gate_error(
        tool_name="camera_pose_to_world",
        parameters={"camera_pose": {"id": "grasp_001"}},
    )


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
    assert memory.grasp_candidate_gate_error(
        tool_name="move_to",
        parameters={"target_pose": {"xyz": [0.4, 0.0, 0.5]}},
    ) is None


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
    event = next(
        item for item in context["memory"]["recent_events"] if item["type"] == "action"
    )
    outputs = event["payload"]["command"]["tool_calls"][0]["result"]["details"][
        "outputs"
    ]
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
    outputs = action_event["payload"]["command"]["tool_calls"][0]["result"]["details"][
        "outputs"
    ]
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
        checker_subagents=CheckerSubagentConfig(
            pre_safety_checks={"move_to": "ik_preview_check"}
        )
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
        checker_subagents=CheckerSubagentConfig(
            pre_safety_checks={"move_to": "ik_preview_check"}
        )
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
    assert post_checks[0]["result"]["details"]["schema_version"] == (
        CHECKER_RESULT_SCHEMA_VERSION
    )
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
    assert not any(
        event.event_type == "recovery_feedback" for event in runtime.memory.events
    )


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
        part["image_url"]["url"].startswith("data:image/png;base64,")
        for part in user_content[1:]
    )
    assert [item["path"] for item in result.details["vision_attachments"]] == [
        str(original),
        str(contact_sheet),
    ]
    assert "base64" not in json.dumps(result.details)


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
        result.details["usage"]["prompt_tokens"]
        + result.details["usage"]["completion_tokens"]
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
