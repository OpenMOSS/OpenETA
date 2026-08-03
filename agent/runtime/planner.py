"""Planner interfaces for the lightweight OpenETA agent runtime."""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from adapter.protocol import EnvObservation, JsonDict
from agent.runtime.actions import CommandKind
from agent.backends.code_policy import (
    CodePolicyBackend,
    CodePolicyGenerationRequest,
    PlaceholderCodePolicyBackend,
)
from agent.runtime.memory import AgentMemory, summarize_observation
from agent.backends.planner import (
    PlaceholderPlannerBackend,
    PlannerBackend,
    PlannerBackendRequest,
    PlannerBackendResult,
)
from agent.runtime.skills import SkillRegistry, SkillSpec
from agent.runtime.token_counting import DEFAULT_CONTEXT_WINDOW_TOKENS, estimate_json_tokens
from agent.tools.registry import ToolRegistry, ToolSpec


_SKILL_MATCH_STOPWORDS = {
    "a",
    "an",
    "and",
    "by",
    "for",
    "from",
    "in",
    "into",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
    "object",
    "target",
}


@dataclass(slots=True)
class PlannerDecision:
    """One planner decision before conversion to `EnvAction`."""

    action_type: str
    action: str
    parameters: JsonDict = field(default_factory=dict)
    reasoning: str = ""
    skill: str | None = None
    code: str | None = None
    metadata: JsonDict = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PlannerContextConfig:
    """Controls bounded planner-facing context assembly."""

    max_memory_events: int = 8
    max_selected_skills: int = 3
    max_skill_content_chars: int = 1200
    auto_compact_enabled: bool = True
    context_window_tokens: int | None = DEFAULT_CONTEXT_WINDOW_TOKENS
    auto_compact_trigger_ratio: float = 0.9
    auto_compact_max_events: int = 8
    approx_chars_per_token: int = 4
    token_estimator_model: str | None = None


class BasePlanner(ABC):
    """Planner interface for one-step embodied decisions."""

    @abstractmethod
    def plan(
        self,
        observation: EnvObservation,
        *,
        memory: AgentMemory,
        tools: ToolRegistry,
        skills: SkillRegistry,
    ) -> PlannerDecision:
        """Plan exactly one next action from the current observation."""


class ToolCallingPlanner(BasePlanner):
    """Default closed-loop tool-calling planner bridge.

    This planner does not hard-code task flows. It packages the current
    observation, memory summary, available tools, and skills into a decision
    context. A real agent backend can then choose exactly one next `tool_call`
    or `response` command from that context.
    """

    def __init__(
        self,
        backend: PlannerBackend | None = None,
        *,
        max_validation_retries: int = 1,
        system_prompt: str = "",
        context_config: PlannerContextConfig | None = None,
    ) -> None:
        self.backend = backend or PlaceholderPlannerBackend()
        self.max_validation_retries = max(0, max_validation_retries)
        self.system_prompt = system_prompt or _default_tool_planner_system_prompt()
        self.context_config = context_config or PlannerContextConfig()

    def plan(
        self,
        observation: EnvObservation,
        *,
        memory: AgentMemory,
        tools: ToolRegistry,
        skills: SkillRegistry,
    ) -> PlannerDecision:
        tool_context = build_tool_context(
            observation=observation,
            memory=memory,
            tools=tools,
            skills=skills,
            config=self.context_config,
        )
        if isinstance(self.backend, PlaceholderPlannerBackend) and not any(
            event.event_type == "observation" for event in memory.events[:-1]
        ):
            return PlannerDecision(
                action_type="tool_call",
                action="sense",
                parameters={},
                reasoning="Start the closed-loop run by requesting/confirming observation.",
                metadata=_planner_metadata(
                    planner=self,
                    tool_context=tool_context,
                    backend=self.backend,
                ),
            )
        validation_errors: list[str] = []
        last_result: PlannerBackendResult | None = None
        backend_usage: JsonDict = {}
        backend_usage_sources: JsonDict = {}
        for attempt in range(1, self.max_validation_retries + 2):
            request = PlannerBackendRequest(
                tool_context=tool_context,
                system_prompt=self.system_prompt,
                attempt=attempt,
                validation_errors=validation_errors,
                metadata={"schema_version": "openeta.planner_decision.v1"},
            )
            last_result = self.backend.decide(request)
            backend_usage = _merge_backend_usage(backend_usage, last_result.details)
            usage_source = str(last_result.details.get("usage_source") or "unknown")
            backend_usage_sources[usage_source] = int(
                backend_usage_sources.get(usage_source) or 0
            ) + 1
            decision, validation_errors = _decision_from_backend_result(
                last_result,
                tools=tools,
                skills=skills,
            )
            if not validation_errors:
                validation_errors.extend(
                    _validate_required_skill_inspection(
                        decision,
                        tools=tools,
                        tool_context=tool_context,
                    )
                )
            if not validation_errors:
                validation_errors.extend(
                    _validate_detection_selection_obligation(
                        decision,
                        tools=tools,
                        tool_context=tool_context,
                    )
                )
            if not validation_errors:
                validation_errors.extend(
                    _validate_anygrasp_candidate_policy(
                        decision,
                        tool_context=tool_context,
                    )
                )
            if not validation_errors:
                decision.metadata.update(
                    _planner_metadata(
                        planner=self,
                        tool_context=tool_context,
                        backend=self.backend,
                        backend_result=last_result,
                        backend_usage=backend_usage,
                        backend_usage_sources=backend_usage_sources,
                        validation_attempts=attempt,
                    )
                )
                return decision

        return PlannerDecision(
            action_type="response",
            action="ask_human",
            parameters={
                "message": "Planner backend returned an invalid action request.",
                "validation_errors": validation_errors,
            },
            reasoning="Planner backend failed schema validation after retries.",
            metadata=_planner_metadata(
                planner=self,
                tool_context=tool_context,
                backend=self.backend,
                backend_result=last_result,
                backend_usage=backend_usage,
                backend_usage_sources=backend_usage_sources,
                validation_errors=validation_errors,
            ),
        )


class CodePolicyPlanner(BasePlanner):
    """Optional Code-as-Policy planner bridge.

    This planner does not hard-code task flows. It packages the current
    observation, memory summary, available tools, skills, and environment API
    references into a policy context. Use this only for short-horizon,
    locally-verifiable policy snippets; it is not the default OpenETA loop.
    """

    def __init__(self, backend: CodePolicyBackend | None = None) -> None:
        self.backend = backend or PlaceholderCodePolicyBackend()

    def plan(
        self,
        observation: EnvObservation,
        *,
        memory: AgentMemory,
        tools: ToolRegistry,
        skills: SkillRegistry,
    ) -> PlannerDecision:
        policy_context = build_policy_context(
            observation=observation,
            memory=memory,
            tools=tools,
            skills=skills,
        )
        generated = self.backend.generate(
            CodePolicyGenerationRequest(policy_context=policy_context)
        )
        return PlannerDecision(
            action_type="tool_call",
            action="code_policy",
            parameters={"policy_context": policy_context},
            code=generated.code,
            reasoning=(
                "OpenETA delegates this bounded action to an agent-generated "
                "Code-as-Policy snippet."
            ),
            metadata={
                "planner": type(self).__name__,
                "backend": self.backend.descriptor(),
                "generation_status": generated.status.value,
                "generation_details": generated.details,
                "execution_model": "optional_bounded_code_policy",
            },
        )


class RuleBasedPlanner(BasePlanner):
    """Deterministic bootstrap planner.

    This is not intended to solve real tasks. It makes the runtime executable
    before a model-backed tool-calling planner is connected. Keep it as a
    fallback or smoke-test planner, not as the primary embodied policy.
    """

    def plan(
        self,
        observation: EnvObservation,
        *,
        memory: AgentMemory,
        tools: ToolRegistry,
        skills: SkillRegistry,
    ) -> PlannerDecision:
        del tools, skills
        task = observation.task.lower()

        if _contains_any(task, ("pick", "grasp", "take", "拿", "抓", "取")):
            target = _select_target_object(observation)
            return PlannerDecision(
                action_type="tool_call",
                action="sam3",
                parameters={"image": _first_camera_id(observation), "prompt": target},
                reasoning=(
                    "Task asks for object acquisition; start with atomic "
                    f"segmentation of target `{target}`."
                ),
            )

        if _contains_any(task, ("place", "put", "放", "放置")):
            return PlannerDecision(
                action_type="tool_call",
                action="scene_detector",
                parameters={"image": _first_camera_id(observation)},
                reasoning="Task asks for placement; first locate candidate receptacles.",
            )

        if _contains_any(task, ("navigate", "go to", "move to", "room", "导航", "移动")):
            return PlannerDecision(
                action_type="tool_call",
                action="slam",
                parameters={"target_location": "task-specified location"},
                reasoning="Task asks for navigation or base movement; query spatial map first.",
            )

        if _contains_any(task, ("wait", "等待")):
            return PlannerDecision(
                action_type="response",
                action="talk",
                parameters={"message": "Waiting for the task-specified condition."},
                reasoning="Task asks the agent to wait; report that state without a tool call.",
            )

        if not any(event.event_type == "observation" for event in memory.events):
            return PlannerDecision(
                action_type="tool_call",
                action="sense",
                parameters={},
                reasoning="No previous observation is available in memory.",
            )

        return PlannerDecision(
            action_type="response",
            action="talk",
            parameters={"message": "No bootstrap rule matched the task."},
            reasoning="No bootstrap rule matched the task.",
        )


def _decision_from_backend_result(
    result: PlannerBackendResult,
    *,
    tools: ToolRegistry,
    skills: SkillRegistry,
) -> tuple[PlannerDecision, list[str]]:
    payload, parse_errors = _parse_backend_payload(result.payload)
    if parse_errors:
        return _invalid_decision(parse_errors), parse_errors

    decision, build_errors = _build_planner_decision(payload)
    validation_errors = [*build_errors, *_validate_planner_decision(decision, tools, skills)]
    if validation_errors:
        return decision, validation_errors
    return decision, []


def _parse_backend_payload(payload: JsonDict | str) -> tuple[JsonDict, list[str]]:
    if isinstance(payload, dict):
        if isinstance(payload.get("decision"), dict):
            return dict(payload["decision"]), []
        if isinstance(payload.get("action"), dict):
            return dict(payload["action"]), []
        return dict(payload), []

    if not isinstance(payload, str):
        return {}, [f"Planner backend payload must be dict or JSON string, got {type(payload)}."]

    text = _strip_json_code_fence(payload)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            return {}, ["Planner backend returned text without a JSON object."]
        try:
            parsed = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            return {}, [f"Planner backend returned invalid JSON: {exc}"]

    if not isinstance(parsed, dict):
        return {}, ["Planner backend JSON must decode to an object."]
    if isinstance(parsed.get("decision"), dict):
        return dict(parsed["decision"]), []
    if isinstance(parsed.get("action"), dict):
        return dict(parsed["action"]), []
    return dict(parsed), []


def _strip_json_code_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            return "\n".join(lines[1:-1]).strip()
    return stripped


def _build_planner_decision(payload: JsonDict) -> tuple[PlannerDecision, list[str]]:
    errors: list[str] = []
    raw_action_type = payload.get("kind", payload.get("action_type", payload.get("type")))
    if not isinstance(raw_action_type, str) or not raw_action_type.strip():
        errors.append("Decision field `kind` or `action_type` must be a non-empty string.")
        raw_action_type = "response"

    raw_name = payload.get("name", payload.get("tool", payload.get("skill", payload.get("action"))))
    if not isinstance(raw_name, str) or not raw_name.strip():
        if raw_action_type == "tool_call" and isinstance(payload.get("calls"), list):
            raw_name = "tool_batch"
        else:
            errors.append("Decision field `name`, `tool`, `skill`, or `action` is required.")
            raw_name = "invalid"

    parameters = payload.get("parameters", {})
    if "calls" in payload and raw_action_type == "tool_call" and raw_name in {"tool_batch", "batch"}:
        parameters = {"calls": payload["calls"]}
    if parameters is None:
        parameters = {}
    if not isinstance(parameters, dict):
        errors.append("Decision field `parameters` must be an object.")
        parameters = {"value": parameters}

    raw_reasoning = payload.get("reasoning", payload.get("reason", ""))
    reasoning = raw_reasoning if isinstance(raw_reasoning, str) else str(raw_reasoning)
    raw_code = payload.get("code")
    code = raw_code if isinstance(raw_code, str) else None
    skill = payload.get("skill") if isinstance(payload.get("skill"), str) else None

    return (
        PlannerDecision(
            action_type=raw_action_type,
            action=raw_name,
            parameters=parameters,
            reasoning=reasoning,
            skill=skill,
            code=code,
            metadata={"raw_backend_payload": payload},
        ),
        errors,
    )


def _validate_planner_decision(
    decision: PlannerDecision,
    tools: ToolRegistry,
    skills: SkillRegistry,
) -> list[str]:
    errors: list[str] = []
    kind = _planner_kind_alias(decision.action_type, decision.skill)
    if kind is None:
        errors.append(f"Unsupported command kind: {decision.action_type!r}.")
        return errors

    if kind == CommandKind.TOOL_CALL:
        if _is_skill_decision(decision):
            name = _skill_decision_name(decision)
            try:
                skills.get(name)
            except KeyError:
                errors.append(f"Unknown skill requested by planner: {name}.")
        elif _is_safety_decision(decision):
            tool_name = _safety_decision_tool_name(decision)
            try:
                spec = tools.get(tool_name)
            except KeyError:
                errors.append(f"Unknown safety tool requested by planner: {tool_name}.")
            else:
                if spec.category != "safety":
                    errors.append(f"safe_check requested non-safety tool: {tool_name}.")
        elif _is_code_policy_decision(decision):
            if not decision.code:
                errors.append(
                    "code_policy is reserved for bounded policy snippets and requires a "
                    "top-level `code` string. Use tool_call::create_simulator_env for "
                    "environment creation and stable simulator tools for control."
                )
        elif decision.action in {"sense"}:
            pass
        elif decision.action in {"tool_batch", "batch"}:
            errors.extend(_validate_tool_batch(decision.parameters, tools))
        else:
            try:
                tools.get(decision.action)
            except KeyError:
                errors.append(f"Unknown tool requested by planner: {decision.action}.")
            else:
                errors.extend(_validate_tool_parameters(decision.action, decision.parameters))

    if kind == CommandKind.RESPONSE and decision.action not in {
        "ask_human",
        "talk",
        "task_complete",
    }:
        errors.append(f"Unsupported response name: {decision.action!r}.")

    return errors


def _validate_tool_batch(parameters: JsonDict, tools: ToolRegistry) -> list[str]:
    errors: list[str] = []
    calls = parameters.get("calls")
    if not isinstance(calls, list) or not calls:
        return ["tool_batch requires a non-empty `parameters.calls` list."]
    for idx, call in enumerate(calls):
        if not isinstance(call, dict):
            errors.append(f"tool_batch call {idx} must be an object.")
            continue
        name = call.get("name", call.get("tool"))
        if not isinstance(name, str) or not name:
            errors.append(f"tool_batch call {idx} requires a tool `name`.")
            continue
        try:
            tools.get(name)
        except KeyError:
            errors.append(f"tool_batch call {idx} requested unknown tool: {name}.")
    return errors


def _validate_tool_parameters(tool_name: str, parameters: JsonDict) -> list[str]:
    if tool_name == "anyplace":
        return _validate_anyplace_parameters(parameters)
    if tool_name == "contact_graspnet":
        return _validate_contact_graspnet_parameters(parameters)
    if tool_name != "anygrasp":
        return []

    errors: list[str] = []
    mode = str(parameters.get("mode") or "targeted").strip().lower()
    if mode in {"", "targeted"}:
        target_mask = parameters.get("target_mask")
        if not isinstance(target_mask, str) or not target_mask.strip():
            errors.append(
                "anygrasp targeted mode requires `parameters.target_mask` as a concrete "
                "local mask image path from the previous sam3 result."
            )
        elif _looks_like_placeholder_mask_path(target_mask):
            errors.append(
                "anygrasp `parameters.target_mask` must be the exact SAM3 mask path, "
                "such as `details.outputs.selected_detection.mask_ref` for a single "
                "detection or the explicitly disambiguated "
                "`details.outputs.detections[i].mask_ref` for multiple detections; do not use "
                f"placeholder values like {target_mask!r}."
            )

    intrinsics = parameters.get("intrinsics")
    required_intrinsics = ("fx", "fy", "cx", "cy", "scale")
    if not isinstance(intrinsics, dict):
        errors.append(
            "anygrasp requires `parameters.intrinsics` copied from the same camera "
            "metadata as rgb/depth, with fx, fy, cx, cy, and scale."
        )
    else:
        missing = [key for key in required_intrinsics if key not in intrinsics]
        if missing:
            errors.append(
                "anygrasp `parameters.intrinsics` is missing required camera fields: "
                + ", ".join(missing)
                + ". Copy fx/fy/cx/cy/scale from the same observe/render camera metadata."
            )
    return errors


def _validate_contact_graspnet_parameters(parameters: JsonDict) -> list[str]:
    errors: list[str] = []
    for key in ("rgb", "depth"):
        value = parameters.get(key)
        if not isinstance(value, str) or not value.strip() or _looks_like_placeholder_path(value):
            errors.append(
                f"contact_graspnet requires `parameters.{key}` as a concrete local file path."
            )

    object_mask = parameters.get("object_mask")
    if not isinstance(object_mask, dict):
        errors.append(
            "contact_graspnet requires `parameters.object_mask` as a SAM3 artifact "
            "containing mask_ref and source_image; bare mask paths are not accepted."
        )
    else:
        for key in ("mask_ref", "source_image"):
            value = object_mask.get(key)
            if (
                not isinstance(value, str)
                or not value.strip()
                or _looks_like_placeholder_path(value)
            ):
                errors.append(
                    f"contact_graspnet object_mask requires a concrete `{key}` local path."
                )

    _validate_required_intrinsics(
        parameters.get("intrinsics"),
        label="contact_graspnet `parameters.intrinsics`",
        errors=errors,
    )
    return errors


def _validate_required_skill_inspection(
    decision: PlannerDecision,
    *,
    tools: ToolRegistry,
    tool_context: JsonDict,
) -> list[str]:
    skill_usage = tool_context.get("skill_usage")
    if not isinstance(skill_usage, dict):
        return []
    required = skill_usage.get("inspection_required")
    if not isinstance(required, list) or not required:
        return []
    required_name = str(required[0] or "").strip()
    if not required_name:
        return []
    if _is_skill_decision(decision) and _skill_decision_name(decision) == required_name:
        return []
    if decision.action_type.lower().strip() != "tool_call":
        return []
    try:
        spec = tools.get(decision.action)
    except KeyError:
        return []
    if not spec.requires_observation_after_call:
        return []
    return [
        f"Selected skill {required_name!r} is truncated and must be inspected with "
        "tool_call::skill_call before world-mutating control."
    ]


def _validate_anyplace_parameters(parameters: JsonDict) -> list[str]:
    errors: list[str] = []
    for key in ("rgb", "depth", "object_mask"):
        value = parameters.get(key)
        if not isinstance(value, str) or not value.strip() or _looks_like_placeholder_path(value):
            errors.append(f"anyplace requires `parameters.{key}` as a concrete local file path.")

    placement_mask = parameters.get("placement_region_mask")
    if not isinstance(placement_mask, dict):
        errors.append(
            "anyplace requires `parameters.placement_region_mask` as a SAM3 artifact "
            "containing mask_ref and source_image."
        )
    else:
        for key in ("mask_ref", "source_image"):
            value = placement_mask.get(key)
            if (
                not isinstance(value, str)
                or not value.strip()
                or _looks_like_placeholder_path(value)
            ):
                errors.append(
                    f"anyplace placement_region_mask requires a concrete `{key}` local path."
                )

    _validate_required_intrinsics(
        parameters.get("intrinsics"),
        label="anyplace `parameters.intrinsics`",
        errors=errors,
    )
    selected = parameters.get("selected_grasp")
    if not isinstance(selected, dict):
        errors.append(
            "anyplace requires `parameters.selected_grasp` with candidate and source objects."
        )
        return errors
    candidate = selected.get("candidate")
    if not isinstance(candidate, dict):
        errors.append("anyplace selected_grasp requires an AnyGrasp `candidate` object.")
    else:
        required_candidate = (
            "id",
            "frame",
            "camera_frame",
            "score",
            "translation_xyz",
            "rotation_matrix",
            "gripper_tip_position_xyz",
            "depth",
            "width",
            "height",
        )
        missing = [key for key in required_candidate if key not in candidate]
        if missing:
            errors.append(
                "anyplace selected_grasp.candidate is missing required fields: "
                + ", ".join(missing)
                + "."
            )
    source = selected.get("source")
    if not isinstance(source, dict) or source.get("mode") != "targeted":
        errors.append("anyplace selected_grasp.source must come from targeted AnyGrasp.")
    else:
        for key in ("rgb", "depth", "object_mask"):
            value = source.get(key)
            if (
                not isinstance(value, str)
                or not value.strip()
                or _looks_like_placeholder_path(value)
            ):
                errors.append(
                    f"anyplace selected_grasp.source requires a concrete `{key}` local path."
                )
        _validate_required_intrinsics(
            source.get("intrinsics"),
            label="anyplace `parameters.selected_grasp.source.intrinsics`",
            errors=errors,
        )
    return errors


def _validate_required_intrinsics(value: object, *, label: str, errors: list[str]) -> None:
    required = ("fx", "fy", "cx", "cy", "scale")
    if not isinstance(value, dict):
        errors.append(f"{label} must contain fx, fy, cx, cy, and scale.")
        return
    missing = [key for key in required if key not in value]
    if missing:
        errors.append(f"{label} is missing required fields: " + ", ".join(missing) + ".")


def _looks_like_placeholder_mask_path(value: str) -> bool:
    normalized = value.strip().lower()
    if not normalized:
        return True
    placeholders = {
        "latest_sam3_mask",
        "latest_mask",
        "sam3_mask",
        "target_mask",
        "mask_ref",
        "mask_path",
        "<mask_ref>",
        "<target_mask>",
    }
    if normalized in placeholders:
        return True
    if normalized.startswith("<") and normalized.endswith(">"):
        return True
    return "latest" in normalized and "mask" in normalized


def _looks_like_placeholder_path(value: str) -> bool:
    normalized = value.strip().lower()
    if not normalized:
        return True
    if normalized.startswith("<") and normalized.endswith(">"):
        return True
    return normalized in {
        "rgb",
        "depth",
        "object_mask",
        "mask_ref",
        "source_image",
        "latest_rgb",
        "latest_depth",
        "latest_mask",
        "latest_sam3_mask",
    }


def _is_skill_decision(decision: PlannerDecision) -> bool:
    return decision.action_type.lower().strip() == "tool_call" and decision.action == "skill_call"


def _skill_decision_name(decision: PlannerDecision) -> str:
    if decision.skill:
        return decision.skill
    if decision.action != "skill_call":
        return decision.action
    for key in ("skill", "name"):
        value = decision.parameters.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return decision.action


def _is_safety_decision(decision: PlannerDecision) -> bool:
    return decision.action_type.lower().strip() == "tool_call" and decision.action == "safe_check"


def _safety_decision_tool_name(decision: PlannerDecision) -> str:
    if decision.action != "safe_check":
        return decision.action
    for key in ("tool", "name", "target"):
        value = decision.parameters.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return decision.action


def _is_code_policy_decision(decision: PlannerDecision) -> bool:
    return decision.action_type.lower().strip() == "tool_call" and decision.action == "code_policy"


def _planner_kind_alias(action_type: str, skill: str | None) -> CommandKind | None:
    del skill
    normalized = action_type.lower().strip()
    if normalized == "tool_call":
        return CommandKind.TOOL_CALL
    if normalized == "response":
        return CommandKind.RESPONSE
    return None


def _invalid_decision(errors: list[str]) -> PlannerDecision:
    return PlannerDecision(
        action_type="response",
        action="ask_human",
        parameters={
            "message": "Planner backend output could not be parsed.",
            "validation_errors": errors,
        },
        reasoning="Planner backend output could not be parsed.",
    )


def _default_tool_planner_system_prompt() -> str:
    return (
        "You are the OpenETA closed-loop embodied planner. Return exactly one "
        "JSON object with fields: kind, name, parameters, reasoning. Valid "
        "top-level kinds are tool_call and response. For tool_call, choose one "
        "registered tool by name, such as create_simulator_env, close_simulator_env, "
        "observe, python_exec, sam3, select_sam3_detection, anygrasp, "
        "contact_graspnet, anyplace, "
        "camera_pose_to_world, move_to, gripper_control, save_memory, "
        "or get_memory. Use "
        "tool_call::create_simulator_env as the only environment-creation path. "
        "Do not invoke create_env or close_env through python_exec or code_policy. For "
        "response, use name ask_human, talk, or task_complete. Choose one "
        "state-changing tool at most; observe again after any world-mutating "
        "action. Tools are atomic stable capabilities. Skills are editable text "
        "guidance documents, not executable macros. When a runtime tool returns "
        "a live tool catalog, docstring, or input schema, treat that runtime "
        "documentation as authoritative over skill examples. If a tool call "
        "fails, inspect the runtime catalog, docstring, input schema, and error "
        "response before retrying. Use tool_call::skill_call only to inspect "
        "guidance. When planner context contains selected_skill_guidance, treat "
        "it as task-level procedure guidance. If skill_usage.inspection_required "
        "is non-empty, inspect that skill with tool_call::skill_call before any "
        "world-mutating control because the selected guidance is truncated. If "
        "only inspection_recommended is non-empty, inspection may be skipped when "
        "the selected guidance already contains the complete procedure. For "
        "pick/grasp/acquire "
        "tasks, do not start with move_to or gripper_control unless a prior "
        "perception/grasp tool result in the current session already provides "
        "a concrete target pose; follow the skill sequence "
        "observe -> scene/object detection -> segmentation -> grasp proposal -> "
        "control/check. For perception prompts, normalize non-English user "
        "object names to concise English visual phrases when possible, such as "
        "`罐子` -> `can`, before calling SAM3. Do not batch dependent tool calls when later parameters "
        "come from earlier outputs in the same command; for example, call sam3, "
        "observe its result, then call anygrasp with the returned mask_ref. "
        "Never treat the number of SAM3 detections as semantic validation. When "
        "selection_obligation is present, including for a single detection, visually "
        "inspect the attached original/contact-sheet images and call "
        "select_sam3_detection with the exact sam3_result_id and detection_id before "
        "calling anygrasp or a world-mutating tool. Treat score as a ranking hint, "
        "not proof of target identity; rerun sam3, observe, or ask_human when uncertain. "
        "A closed gripper is not proof of a successful grasp: verify target-object "
        "motion, end-effector state, reward, or visual evidence before lifting. "
        "Never invent placeholders such as latest_sam3_mask; use exact paths "
        "from prior tool results or artifacts. AnyGrasp candidates follow an explicit "
        "greedy fallback policy: use grasp_candidate_policy.active_candidate, pass the "
        "complete candidate through camera_pose_to_world, and pass the complete returned "
        "world_pose to move_to without editing its translated world-frame pose. Close "
        "the gripper with a separate gripper_control call only after move_to succeeds. "
        "Do not skip to a lower rank. Only a structured candidate-specific safety or "
        "failure-check rejection advances "
        "the policy. Transport failures, interruption, unclassified path collisions, and "
        "calibration errors must keep the current candidate active for diagnosis. When the "
        "queue is exhausted, observe and rerun perception instead of reusing a rejected pose."
    )


def _validate_detection_selection_obligation(
    decision: PlannerDecision,
    *,
    tools: ToolRegistry,
    tool_context: JsonDict,
) -> list[str]:
    if decision.action_type.lower().strip() != "tool_call":
        return []
    pending = tool_context.get("selection_obligation")
    selected = tool_context.get("selected_sam3_detection")
    if decision.action == "select_sam3_detection":
        if not isinstance(pending, dict):
            return ["select_sam3_detection requested without a pending SAM3 selection."]
        result_id = str(decision.parameters.get("sam3_result_id") or "")
        expected_result_id = str(pending.get("result_id") or "")
        if result_id != expected_result_id:
            return [
                "select_sam3_detection must use the exact pending sam3_result_id "
                f"{expected_result_id!r}."
            ]
        detection_id = str(decision.parameters.get("detection_id") or "")
        candidate_ids = {
            str(candidate.get("id") or "")
            for candidate in (pending.get("candidates") or [])
            if isinstance(candidate, dict)
        }
        if not detection_id or detection_id not in candidate_ids:
            return [
                "select_sam3_detection detection_id must identify one candidate from "
                "the pending SAM3 result."
            ]
        return []
    try:
        spec = tools.get(decision.action)
    except KeyError:
        return []
    if isinstance(pending, dict):
        mode = str(decision.parameters.get("mode") or "targeted").strip().lower()
        if decision.action == "anygrasp" and mode != "scene":
            return [
                "Targeted AnyGrasp is blocked until select_sam3_detection resolves "
                "the pending SAM3 semantic-verification obligation."
            ]
        if spec.effect.value == "world_mutating":
            return [
                "World-mutating tools are blocked while a SAM3 detection selection "
                "obligation is pending."
            ]
    if decision.action != "anygrasp" or not isinstance(selected, dict):
        return []
    mode = str(decision.parameters.get("mode") or "targeted").strip().lower()
    if mode == "scene":
        return []
    expected_mask = str(selected.get("mask_ref") or "")
    supplied_mask = str(decision.parameters.get("target_mask") or "")
    if expected_mask and supplied_mask != expected_mask:
        return [
            "Targeted AnyGrasp must use the mask_ref returned by the recorded "
            "select_sam3_detection result."
        ]
    return []


def _validate_anygrasp_candidate_policy(
    decision: PlannerDecision,
    *,
    tool_context: JsonDict,
) -> list[str]:
    if decision.action_type.lower().strip() != "tool_call":
        return []
    policy = tool_context.get("grasp_candidate_policy")
    if not isinstance(policy, dict):
        return []
    if str(policy.get("status") or "") == "accepted":
        return []
    target_tool = (
        _safety_decision_tool_name(decision)
        if _is_safety_decision(decision)
        else decision.action
    )
    if target_tool not in {"camera_pose_to_world", "move_to"}:
        return []
    active = policy.get("active_candidate")
    if str(policy.get("status") or "") == "exhausted" or not isinstance(active, dict):
        return [
            "All AnyGrasp candidates are exhausted. Observe and rerun AnyGrasp before "
            "requesting another grasp-derived transform, safety check, or motion."
        ]
    active_id = str(active.get("id") or "")
    supplied_id = _planner_grasp_candidate_id(decision.parameters)
    if not supplied_id:
        return [
            f"{target_tool} must preserve the active AnyGrasp candidate id {active_id!r}. "
            "Pass the complete candidate to camera_pose_to_world and the complete "
            "world_pose result to safety/motion tools."
        ]
    if supplied_id != active_id:
        return [
            f"Greedy AnyGrasp policy requires active candidate {active_id!r}; "
            f"candidate {supplied_id!r} cannot be used until earlier candidates are "
            "rejected by a linked safety check or motion failure."
        ]
    return []


def _planner_grasp_candidate_id(parameters: JsonDict) -> str:
    for key in ("source_grasp_id", "grasp_candidate_id"):
        value = parameters.get(key)
        if isinstance(value, str) and value:
            return value
    for key in ("camera_pose", "target_pose", "pose", "eef_pose"):
        pose = parameters.get(key)
        if not isinstance(pose, dict):
            continue
        for id_key in ("id", "source_grasp_id", "grasp_candidate_id"):
            value = pose.get(id_key)
            if isinstance(value, str) and value:
                return value
    target_parameters = parameters.get("target_parameters")
    if isinstance(target_parameters, dict):
        return _planner_grasp_candidate_id(target_parameters)
    return ""


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _select_target_object(observation: EnvObservation) -> str:
    if observation.objects:
        name = observation.objects[0].get("name")
        if isinstance(name, str) and name:
            return name
    return "task-specified object"


def _first_camera_id(observation: EnvObservation) -> str | None:
    if not observation.cameras:
        return None
    return observation.cameras[0].frame_id


def build_policy_context(
    *,
    observation: EnvObservation,
    memory: AgentMemory,
    tools: ToolRegistry,
    skills: SkillRegistry,
    config: PlannerContextConfig | None = None,
) -> JsonDict:
    """Build the agent-visible context for bounded Code-as-Policy generation."""

    tool_context = build_tool_context(
        observation=observation,
        memory=memory,
        tools=tools,
        skills=skills,
        config=config,
    )
    return {
        **tool_context,
        "env_api_reference": _env_api_reference(),
        "safety_constraints": [
            "Code policy is optional and must be short-horizon.",
            "Run feasibility and collision checks before physical motion.",
            "Observe/checkpoint after any simulator or robot state change.",
            "Ask a human when task targets, receptacles, or constraints are ambiguous.",
        ],
    }


def build_tool_context(
    *,
    observation: EnvObservation,
    memory: AgentMemory,
    tools: ToolRegistry,
    skills: SkillRegistry,
    config: PlannerContextConfig | None = None,
) -> JsonDict:
    """Build the agent-visible context for closed-loop tool selection."""

    context_config = config or PlannerContextConfig()
    context = _build_tool_context_payload(
        observation=observation,
        memory=memory,
        tools=tools,
        skills=skills,
        config=context_config,
    )
    budget = _context_budget_status(
        context,
        config=context_config,
        auto_compact_triggered=False,
    )
    if budget["should_auto_compact"]:
        memory.compact(max_events=context_config.auto_compact_max_events)
        context = _build_tool_context_payload(
            observation=observation,
            memory=memory,
            tools=tools,
            skills=skills,
            config=context_config,
        )
        budget = _context_budget_status(
            context,
            config=context_config,
            auto_compact_triggered=True,
        )
    context["context_budget"] = budget
    return context


def _build_tool_context_payload(
    *,
    observation: EnvObservation,
    memory: AgentMemory,
    tools: ToolRegistry,
    skills: SkillRegistry,
    config: PlannerContextConfig,
) -> JsonDict:
    selected_skill_guidance = _selected_skill_guidance(
        skills.list(),
        observation=observation,
        memory=memory,
        config=config,
    )
    skill_usage = _skill_usage_guidance(selected_skill_guidance, memory)
    memory_context = memory.planning_context(max_events=config.max_memory_events)
    return {
        "schema_version": "openeta.planner_context.v1",
        "task": observation.task,
        "observation": _observation_summary(observation),
        "memory": memory_context,
        "selection_obligation": memory_context.get("selection_obligation"),
        "selected_sam3_detection": memory_context.get("selected_sam3_detection"),
        "grasp_candidate_policy": memory_context.get("grasp_candidate_policy"),
        "tool_references": [_tool_reference(tool) for tool in tools.list()],
        "registered_tool_handlers": tools.handler_names(),
        "skill_references": [_selected_skill_reference(skill) for skill in selected_skill_guidance],
        "available_skill_count": len(skills.list()),
        "selected_skill_guidance": selected_skill_guidance,
        "skill_usage": skill_usage,
        "execution_rules": _tool_calling_rules(),
    }


def _context_budget_status(
    context: JsonDict,
    *,
    config: PlannerContextConfig,
    auto_compact_triggered: bool,
) -> JsonDict:
    estimate = estimate_json_tokens(
        context,
        model=config.token_estimator_model,
        approx_chars_per_token=config.approx_chars_per_token,
    )
    estimated_chars = estimate.chars
    estimated_tokens = estimate.tokens
    trigger_ratio = min(max(config.auto_compact_trigger_ratio, 0.0), 1.0)
    trigger_tokens = (
        int(config.context_window_tokens * trigger_ratio)
        if config.context_window_tokens is not None
        else None
    )
    tokens_until_auto_compact = (
        max(0, trigger_tokens - estimated_tokens) if trigger_tokens is not None else None
    )
    should_auto_compact = (
        config.auto_compact_enabled
        and not auto_compact_triggered
        and trigger_tokens is not None
        and estimated_tokens >= trigger_tokens
    )
    return {
        "schema_version": "openeta.context_budget.v1",
        "auto_compact_enabled": config.auto_compact_enabled,
        "auto_compact_triggered": auto_compact_triggered,
        "should_auto_compact": should_auto_compact,
        "context_window_tokens": config.context_window_tokens,
        "trigger_ratio": trigger_ratio,
        "trigger_tokens": trigger_tokens,
        "estimated_chars": estimated_chars,
        "estimated_tokens": estimated_tokens,
        "tokens_until_auto_compact": tokens_until_auto_compact,
        "estimator": estimate.estimator,
    }


def _planner_metadata(
    *,
    planner: ToolCallingPlanner,
    tool_context: JsonDict,
    backend: PlannerBackend,
    backend_result: PlannerBackendResult | None = None,
    backend_usage: JsonDict | None = None,
    backend_usage_sources: JsonDict | None = None,
    validation_attempts: int | None = None,
    validation_errors: list[str] | None = None,
) -> JsonDict:
    metadata: JsonDict = {
        "planner": type(planner).__name__,
        "tool_context_summary": _tool_context_summary(tool_context),
        "backend": backend.descriptor(),
        "execution_model": "closed_loop_tool_calling",
    }
    if backend_result is not None:
        metadata.update(
            {
                "backend_status": backend_result.status.value,
                "backend_provider": backend_result.provider,
                "backend_model": backend_result.model,
                "backend_details": backend_result.details,
            }
        )
    if backend_usage:
        metadata["backend_usage"] = dict(backend_usage)
    if backend_usage_sources:
        metadata["backend_usage_sources"] = dict(backend_usage_sources)
    if validation_attempts is not None:
        metadata["validation_attempts"] = validation_attempts
    if validation_errors is not None:
        metadata["validation_errors"] = list(validation_errors)
    return metadata


def _merge_backend_usage(accumulated: JsonDict, details: JsonDict) -> JsonDict:
    usage = details.get("usage")
    if not isinstance(usage, dict):
        return dict(accumulated)
    normalized = {
        str(key): max(0, int(value))
        for key, value in usage.items()
        if not isinstance(value, bool) and isinstance(value, (int, float))
    }
    if "total_tokens" not in normalized:
        prompt = int(normalized.get("prompt_tokens") or 0)
        completion = int(normalized.get("completion_tokens") or 0)
        if prompt or completion:
            normalized["total_tokens"] = prompt + completion
    merged = dict(accumulated)
    for key, value in normalized.items():
        merged[key] = merged.get(key, 0) + value
    return merged


def _tool_context_summary(context: JsonDict) -> JsonDict:
    budget = context.get("context_budget")
    selected_skills = context.get("selected_skill_guidance", [])
    if not isinstance(selected_skills, list):
        selected_skills = []
    observation = context.get("observation")
    if not isinstance(observation, dict):
        observation = {}
    memory = context.get("memory")
    if not isinstance(memory, dict):
        memory = {}
    recent_events = memory.get("recent_events", [])
    if not isinstance(recent_events, list):
        recent_events = []
    return {
        "schema_version": "openeta.planner_context_summary.v1",
        "task": context.get("task"),
        "observation": {
            "camera_count": len(observation.get("camera_ids", []) or []),
            "object_count": len(observation.get("objects", []) or []),
            "metadata_keys": sorted((observation.get("metadata") or {}).keys())
            if isinstance(observation.get("metadata"), dict)
            else [],
        },
        "memory": {
            "recent_event_count": len(recent_events),
            "has_compact_summary": bool(
                ((memory.get("working_memory") or {}).get("compact_summary"))
                if isinstance(memory.get("working_memory"), dict)
                else False
            ),
        },
        "tool_count": len(context.get("tool_references", []) or []),
        "registered_handler_count": len(context.get("registered_tool_handlers", []) or []),
        "skill_count": len(context.get("skill_references", []) or []),
        "selected_skills": [
            {
                "name": skill.get("name"),
                "score": skill.get("selection_score"),
                "current_task_score": skill.get("current_task_score"),
                "content_char_count": skill.get("content_char_count"),
                "content_truncated": skill.get("content_truncated"),
            }
            for skill in selected_skills
            if isinstance(skill, dict)
        ],
        "skill_usage": dict(context.get("skill_usage"))
        if isinstance(context.get("skill_usage"), dict)
        else {},
        "context_budget": dict(budget) if isinstance(budget, dict) else {},
    }


def _observation_summary(observation: EnvObservation) -> JsonDict:
    summary = summarize_observation(observation)
    summary.pop("task", None)
    return summary


def _tool_reference(tool: ToolSpec) -> JsonDict:
    return {
        "name": tool.name,
        "category": tool.category,
        "description": tool.description,
        "parameters": tool.parameters,
        "safe_by_default": tool.safe_by_default,
        "effect": tool.effect.value,
        "batchable": tool.allows_batched_observation,
        "requires_observation_after_call": tool.requires_observation_after_call,
    }


def _skill_reference(skill: SkillSpec) -> JsonDict:
    return {
        "name": skill.name,
        "description": skill.description,
        "task_patterns": list(skill.task_patterns),
        "allowed_tools": list(skill.allowed_tools),
        "source": skill.source,
        "version": skill.version,
        "editable": skill.editable,
        "metadata": skill.metadata,
    }


def _selected_skill_reference(skill: JsonDict) -> JsonDict:
    return {
        key: value
        for key, value in skill.items()
        if key
        in {
            "name",
            "description",
            "task_patterns",
            "allowed_tools",
            "source",
            "version",
            "editable",
            "metadata",
            "selection_score",
            "current_task_score",
            "selection_reason",
            "content_char_count",
            "content_truncated",
        }
    }


def _selected_skill_guidance(
    skills: list[SkillSpec],
    *,
    observation: EnvObservation,
    memory: AgentMemory,
    config: PlannerContextConfig,
) -> list[JsonDict]:
    scored = [
        (score, _skill_text_relevance_score(skill, observation.task.lower()), skill)
        for skill in skills
        if (score := _skill_relevance_score(skill, observation, memory)) > 0
    ]
    scored.sort(key=lambda item: (-item[0], item[2].name))
    return [
        _skill_guidance_reference(
            skill,
            score=score,
            current_task_score=current_task_score,
            config=config,
        )
        for score, current_task_score, skill in scored[: config.max_selected_skills]
    ]


def _skill_guidance_reference(
    skill: SkillSpec,
    *,
    score: int,
    current_task_score: int,
    config: PlannerContextConfig,
) -> JsonDict:
    content, truncated = _truncate_text(skill.content, config.max_skill_content_chars)
    payload = _skill_reference(skill)
    payload.update(
        {
            "content": content,
            "content_truncated": truncated,
            "content_char_count": len(skill.content),
            "selection_score": score,
            "current_task_score": current_task_score,
            "selection_reason": "Matched current task, scene, or working memory.",
        }
    )
    return payload


def _skill_usage_guidance(selected_skill_guidance: list[JsonDict], memory: AgentMemory) -> JsonDict:
    selected = [
        str(skill.get("name")).strip()
        for skill in selected_skill_guidance
        if isinstance(skill.get("name"), str) and str(skill.get("name")).strip()
    ]
    inspected = _inspected_skill_names(memory)
    inspection_recommended = [name for name in selected if name not in inspected]
    primary = selected_skill_guidance[0] if selected_skill_guidance else {}
    primary_name = str(primary.get("name") or "").strip()
    inspection_required = (
        [primary_name]
        if primary_name
        and primary.get("content_truncated") is True
        and int(primary.get("current_task_score") or 0) > 0
        and primary_name not in inspected
        else []
    )
    return {
        "selected_skills": selected,
        "inspected_skills": sorted(inspected),
        "inspection_recommended": inspection_recommended,
        "inspection_required": inspection_required,
        "rule": (
            "If inspection_required is non-empty, call tool_call::skill_call for "
            "the first listed skill before world-mutating control because the "
            "selected guidance is truncated. Otherwise, when inspection_recommended "
            "is non-empty, inspect or explicitly follow the complete selected guidance."
        ),
    }


def _inspected_skill_names(memory: AgentMemory) -> set[str]:
    inspected: set[str] = set()
    for event in memory.events:
        payload = event.payload
        if not isinstance(payload, dict):
            continue
        command = payload.get("command")
        if not isinstance(command, dict):
            continue
        skill_call = command.get("skill_call")
        if isinstance(skill_call, dict):
            name = skill_call.get("name")
            if isinstance(name, str) and name.strip():
                inspected.add(name.strip())
        request = command.get("request")
        if isinstance(request, dict) and request.get("name") == "skill_call":
            parameters = request.get("parameters")
            if isinstance(parameters, dict):
                name = parameters.get("name") or parameters.get("skill")
                if isinstance(name, str) and name.strip():
                    inspected.add(name.strip())
    return inspected


def _skill_relevance_score(
    skill: SkillSpec,
    observation: EnvObservation,
    memory: AgentMemory,
) -> int:
    current_query = observation.task.lower()
    supporting_query = _skill_query_text(observation, memory, include_current_task=False)
    current_score = _skill_text_relevance_score(skill, current_query)
    supporting_score = _skill_text_relevance_score(skill, supporting_query)
    score = current_score * 3 + supporting_score
    if current_score == 0 and memory.task and memory.task.lower() != current_query:
        score += min(3, _skill_text_relevance_score(skill, memory.task.lower()))
    if skill.name in memory.skill_notes:
        score += 2
    return score


def _skill_text_relevance_score(skill: SkillSpec, query: str) -> int:
    query_tokens = set(_word_tokens(query))
    score = 0
    name = skill.name.lower()
    if name in query:
        score += 8
    for token in _word_tokens(name):
        if token in query_tokens:
            score += 4
    for pattern in skill.task_patterns:
        normalized_pattern = pattern.strip().lower()
        if normalized_pattern and normalized_pattern in query:
            score += 8
            continue
        pattern_anchor = re.sub(r"<[^>]+>", "", normalized_pattern).strip()
        if pattern_anchor and pattern_anchor in query:
            score += 8
            continue
        pattern_tokens = [
            token for token in _word_tokens(pattern) if token not in _SKILL_MATCH_STOPWORDS
        ]
        if pattern_tokens and all(token in query_tokens for token in pattern_tokens):
            score += 6
        elif any(token in query_tokens for token in pattern_tokens):
            score += 3
    description_tokens = {
        token
        for token in _word_tokens(skill.description)
        if token not in _SKILL_MATCH_STOPWORDS
    }
    score += min(3, len(query_tokens & description_tokens))
    return score


def _skill_query_text(
    observation: EnvObservation,
    memory: AgentMemory,
    *,
    include_current_task: bool = True,
) -> str:
    object_names = [
        str(obj.get("name", ""))
        for obj in observation.objects
        if isinstance(obj, dict) and obj.get("name")
    ]
    return " ".join(
        [
            observation.task if include_current_task else "",
            *object_names,
            memory.compact_summary,
            *memory.skill_notes.keys(),
        ]
    ).lower()


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    if max_chars <= 0:
        return "", bool(text)
    if len(text) <= max_chars:
        return text, False
    marker = "\n\n[truncated]"
    return text[: max(0, max_chars - len(marker))].rstrip() + marker, True


def _tool_calling_rules() -> JsonDict:
    return {
        "primary_loop": "observe -> decide one tool(parameter) -> execute -> observe result",
        "default": "One state-changing tool per planner turn.",
        "batching": {
            "allowed_effects": ["read_only", "bookkeeping", "planning"],
            "blocked_effects": ["world_mutating"],
            "rule": (
                "Batching is only allowed for read-only sensing/query, bookkeeping, "
                "and pure planning helpers. Any world-mutating actuator/control "
                "tool must return control to the planner with a fresh observation."
            ),
        },
        "skills": (
            "Skills are editable text guidance documents. They may recommend a "
            "tool sequence, but the runtime will not auto-expand or execute that "
            "sequence. The planner must choose each atomic tool_call explicitly "
            "after observing the previous result."
        ),
        "dependent_tool_calls": (
            "Only batch independent read-only/planning tools. If tool B needs "
            "paths, ids, intrinsics, masks, poses, or candidates produced by "
            "tool A, call A first, inspect its result in the next planner turn, "
            "then call B with concrete parameters."
        ),
        "runtime_tool_docs": (
            "Runtime-discovered tool catalogs, docstrings, and input schemas are "
            "authoritative for parameter names, required fields, and current "
            "interface availability. If skill text or examples conflict with "
            "runtime tool documentation, follow the runtime documentation. "
            "When a runtime tool call fails, first inspect the relevant catalog, "
            "docstring, input schema, and error response before retrying with "
            "changed parameters."
        ),
        "code_policy": (
            "Code policy is an optional atomic-tool backend for bounded, locally "
            "verifiable snippets, not the main task execution loop."
        ),
    }


def _env_api_reference() -> JsonDict:
    return {
        "sandbox": {
            "root": "sim/",
            "backend": "RLinf-backed Gymnasium environment",
            "env_registry": "sim.envs.get_env_cls(env_type, env_cfg)",
            "runtime_protocol": ["reset", "step", "chunk_step", "close"],
            "wrappers": "No shared wrapper package is currently exposed; simulator adapters own optional recording instrumentation",
        },
        "api.observe()": "Return the latest observation cached by the code-policy API.",
        "api.step(action)": "Apply one low-level env action through the OpenETA env facade.",
        "api.chunk_step(actions)": "Apply an action chunk when the backend supports chunk stepping.",
        "api.reset(**kwargs)": "Reset the underlying env through the OpenETA env facade.",
        "call_tool(name, **kwargs)": "Call a registered perception/control/safety tool.",
        "safe_check(name, **kwargs)": "Run a registered safety preflight check.",
        "move_arm(target_pose, *, preview=True)": "Future helper that should compile to one or more api.step/api.chunk_step calls.",
        "move_base(target_pose, *, preview=True)": "Future helper that should compile to one or more api.step/api.chunk_step calls.",
        "open_gripper()": "Future helper that should compile to an env action.",
        "close_gripper()": "Future helper that should compile to an env action.",
        "ask_human(message)": "Request clarification from an operator.",
        "talk(message)": "Emit human-readable status.",
    }
