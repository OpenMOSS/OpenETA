"""Planner interfaces for the lightweight OpenETA agent runtime."""

from __future__ import annotations

import json
import math
import re
import time
import urllib.parse
import xml.etree.ElementTree as ET
from functools import lru_cache
from hashlib import sha256
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

from adapter.protocol import EnvObservation, JsonDict
from agent.runtime.actions import CommandKind
from agent.backends.code_policy import (
    CodePolicyBackend,
    CodePolicyGenerationRequest,
    PlaceholderCodePolicyBackend,
)
from agent.runtime.memory import (
    AgentMemory,
    summarize_observation,
)
from agent.runtime.planner_prompts import compose_main_planner_prompt
from agent.runtime.rollout import RolloutRecorder, public_backend_details
from agent.backends.planner import (
    PlaceholderPlannerBackend,
    PlannerBackend,
    PlannerBackendRequest,
    PlannerBackendResult,
)
from agent.runtime.skills import SkillRegistry, SkillSpec
from agent.runtime.task_playbooks import (
    DEFAULT_TASK_PLAYBOOK_ROOT,
    TaskPlaybookError,
    load_task_playbooks,
    select_task_playbook,
)
from agent.runtime.token_counting import (
    DEFAULT_CONTEXT_WINDOW_TOKENS,
    TokenEstimate,
    estimate_json_tokens,
)
from agent.runtime.visual_history import (
    VisualHistoryConfig,
    build_visual_history_projection,
)
from agent.tools.registry import GRASP_POSE_BACKENDS, ToolRegistry, ToolSpec
from agent.tools.contracts import (
    ToolContractCatalog,
    ToolContractRuntimePolicy,
    check_tool_request_conformance,
)


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

_CAMERA_ROLE_PREFERENCE = {
    "scene_primary": 0,
    "scene_secondary": 1,
    "wrist_primary": 2,
    "wrist_secondary": 3,
}

DEFAULT_MAX_SKILL_CONTENT_CHARS = 8000
DEFAULT_RECENT_CONVERSATION_ACTION_GROUPS = 4
DEFAULT_RECENT_TRANSITION_OBSERVATIONS = 3
TOOL_CONTRACT_SHADOW_VALIDATION_SCHEMA_VERSION = (
    "openeta.tool_contract_shadow_validation.v1"
)


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
    """Controls the bounded model projection of an unbounded durable session."""

    # ``max_memory_events`` is a legacy caller override on the in-memory source
    # projection. Normal Planner requests read the durable event stream and then
    # apply the semantic high-fidelity window below.
    max_memory_events: int | None = None
    recent_conversation_action_groups: int = DEFAULT_RECENT_CONVERSATION_ACTION_GROUPS
    recent_transition_observations: int = DEFAULT_RECENT_TRANSITION_OBSERVATIONS
    max_selected_skills: int = 3
    max_skill_content_chars: int | None = DEFAULT_MAX_SKILL_CONTENT_CHARS
    auto_compact_enabled: bool = True
    context_window_tokens: int | None = DEFAULT_CONTEXT_WINDOW_TOKENS
    auto_compact_trigger_ratio: float = 0.9
    reserved_output_tokens: int = 4096
    approx_chars_per_token: int = 4
    approx_tokens_per_image: int = 2048
    token_estimator_model: str | None = None
    visual_history: VisualHistoryConfig = field(
        default_factory=lambda: VisualHistoryConfig(enabled=False)
    )


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
        tool_contract_catalog: ToolContractCatalog | None = None,
        tool_contract_policy: ToolContractRuntimePolicy | None = None,
    ) -> None:
        self.backend = backend or PlaceholderPlannerBackend()
        self.max_validation_retries = max(0, max_validation_retries)
        self.context_config = context_config or PlannerContextConfig()
        self.tool_contract_catalog = (
            tool_contract_catalog or _default_tool_contract_catalog()
        )
        self.tool_contract_policy = tool_contract_policy or ToolContractRuntimePolicy()
        self.tool_contract_policy.ensure_valid(self.tool_contract_catalog)
        base_prompt = system_prompt or _agent_owned_tool_planner_system_prompt()
        self.system_prompt, self.prompt_metadata = compose_main_planner_prompt(base_prompt)
        self.rollout_recorder: RolloutRecorder | None = None

    def set_rollout_recorder(self, recorder: RolloutRecorder | None) -> None:
        """Attach the training-data recorder owned by the runtime."""

        self.rollout_recorder = recorder

    def plan(
        self,
        observation: EnvObservation,
        *,
        memory: AgentMemory,
        tools: ToolRegistry,
        skills: SkillRegistry,
    ) -> PlannerDecision:
        tool_context, conversation_messages = _build_budgeted_tool_context(
            observation=observation,
            memory=memory,
            tools=tools,
            skills=skills,
            config=self.context_config,
            system_prompt=self.system_prompt,
        )
        host_obligation = _invariant_obligation_decision(
            tool_context,
            tools=tools,
        )
        if host_obligation is not None:
            host_obligation.metadata.update(
                _planner_metadata(
                    planner=self,
                    tool_context=tool_context,
                    backend=self.backend,
                )
            )
            host_obligation.metadata["execution_model"] = "host_invariant_dispatch"
            return host_obligation
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
        validation_attempt_history: list[JsonDict] = []
        for attempt in range(1, self.max_validation_retries + 2):
            agent_context = tool_context.get("agent_context")
            request = PlannerBackendRequest(
                tool_context=(
                    dict(agent_context) if isinstance(agent_context, dict) else tool_context
                ),
                system_prompt=self.system_prompt,
                conversation_messages=conversation_messages,
                conversation_summary=memory.conversation_checkpoint_summary(),
                attempt=attempt,
                validation_errors=validation_errors,
                metadata={"schema_version": "openeta.planner_decision.v1"},
            )
            model_started_at_s = time.time()
            last_result = self.backend.decide(request)
            model_completed_at_s = time.time()
            backend_usage = _merge_backend_usage(backend_usage, last_result.details)
            usage_source = str(last_result.details.get("usage_source") or "unknown")
            backend_usage_sources[usage_source] = (
                int(backend_usage_sources.get(usage_source) or 0) + 1
            )
            decision, validation_errors = _decision_from_backend_result(
                last_result,
                tools=tools,
                skills=skills,
                tool_contract_catalog=self.tool_contract_catalog,
                tool_contract_policy=self.tool_contract_policy,
                tool_context=tool_context,
            )
            required_skill = ""
            if not validation_errors:
                required_skill = _required_skill_inspection_name(
                    decision,
                    tools=tools,
                    tool_context=tool_context,
                )
                if required_skill:
                    validation_errors.append(_required_skill_inspection_error(required_skill))
            if not validation_errors:
                validation_errors.extend(
                    _validate_calibration_permission(
                        decision,
                        tool_context=tool_context,
                    )
                )
            if not validation_errors:
                validation_errors.extend(
                    _validate_perception_artifact_provenance(
                        decision,
                        tool_context=tool_context,
                    )
                )
            if not validation_errors:
                validation_errors.extend(
                    _validate_compiled_grasp_target_freshness(
                        decision,
                        tool_context=tool_context,
                    )
                )
            if not validation_errors:
                validation_errors.extend(
                    _validate_official_reward_completion(
                        decision,
                        tool_context=tool_context,
                    )
                )
            if self.rollout_recorder is not None:
                self.rollout_recorder.record_model_call(
                    request=request,
                    result=last_result,
                    decision=decision,
                    validation_errors=validation_errors,
                    backend=self.backend.descriptor(),
                    started_at_s=model_started_at_s,
                    completed_at_s=model_completed_at_s,
                )
            validation_attempt_history.append(
                _planner_validation_attempt_record(
                    attempt=attempt,
                    result=last_result,
                    decision=decision,
                    validation_errors=validation_errors,
                )
            )
            if required_skill:
                redirected = PlannerDecision(
                    action_type="tool_call",
                    action="skill_call",
                    parameters={"skill": required_skill},
                    reasoning=(
                        f"Inspect required skill guidance {required_skill!r} before executing "
                        f"the blocked world-mutating tool {decision.action!r}."
                    ),
                )
                redirected.metadata.update(
                    _planner_metadata(
                        planner=self,
                        tool_context=tool_context,
                        backend=self.backend,
                        backend_result=last_result,
                        backend_usage=backend_usage,
                        backend_usage_sources=backend_usage_sources,
                        validation_attempts=attempt,
                        validation_attempt_history=validation_attempt_history,
                        validation_errors=validation_errors,
                        policy_redirect={
                            "code": "required_skill_inspection",
                            "skill": required_skill,
                            "blocked_action": {
                                "kind": decision.action_type,
                                "name": decision.action,
                            },
                        },
                    )
                )
                return redirected
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
                        validation_attempt_history=validation_attempt_history,
                    )
                )
                return decision

        return PlannerDecision(
            action_type="response",
            action="talk",
            parameters={
                "message": "Planner could not produce a valid action request.",
                "code": "planner_validation_failed",
                "validation_errors": validation_errors,
                "validation_attempts": len(validation_attempt_history),
            },
            reasoning="Planner backend failed schema validation after retries.",
            metadata=_planner_metadata(
                planner=self,
                tool_context=tool_context,
                backend=self.backend,
                backend_result=last_result,
                backend_usage=backend_usage,
                backend_usage_sources=backend_usage_sources,
                validation_attempts=len(validation_attempt_history),
                validation_attempt_history=validation_attempt_history,
                validation_errors=validation_errors,
            ),
        )


def _invariant_obligation_decision(
    tool_context: JsonDict,
    *,
    tools: ToolRegistry,
) -> PlannerDecision | None:
    """Dispatch only host-owned observation and transport safety obligations."""

    refresh = tool_context.get("fresh_observation_obligation")
    if (
        isinstance(refresh, dict)
        and refresh.get("required") is True
        and tools.can_execute("observe")
    ):
        return PlannerDecision(
            action_type="tool_call",
            action="observe",
            parameters={"reason": "host_refresh_after_world_mutation"},
            reasoning=(
                "The previous environment mutation returned no fresh observation; "
                "refresh the same environment before any model-directed control."
            ),
            metadata={
                "host_invariant": {
                    "schema_version": "openeta.fresh_observation_obligation.v1",
                    "tool": "observe",
                    "attempt": refresh.get("attempt"),
                }
            },
        )

    reconciliation = tool_context.get("motion_reconciliation")
    if (
        isinstance(reconciliation, dict)
        and reconciliation.get("status") in {"required", "unresolved"}
        and tools.can_execute("observe")
    ):
        return PlannerDecision(
            action_type="tool_call",
            action="observe",
            parameters={},
            reasoning=(
                "The previous simulator action has transport-unknown outcome; observe "
                "the same handle before dispatching another world mutation."
            ),
            metadata={
                "host_invariant": {
                    "schema_version": "openeta.motion_reconciliation.v1",
                    "tool": "observe",
                    "unknown_tool": reconciliation.get("tool"),
                }
            },
        )
    return None

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
        task = _effective_task_text(observation, memory).lower()

        if _contains_any(task, ("pick", "grasp", "take", "拿", "抓", "取")):
            target = _select_target_object(observation)
            return PlannerDecision(
                action_type="tool_call",
                action="sam3",
                parameters={
                    "source_packet_id": _first_observation_packet_id(observation),
                    "prompt": target,
                },
                reasoning=(
                    "Task asks for object acquisition; start with atomic "
                    f"segmentation of target `{target}`."
                ),
            )

        if _contains_any(task, ("place", "put", "放", "放置")):
            return PlannerDecision(
                action_type="tool_call",
                action="observe",
                parameters={"reason": "locate candidate placement receptacles"},
                reasoning="Task asks for placement; first inspect the current scene.",
            )

        if _contains_any(task, ("navigate", "go to", "move to", "room", "导航", "移动")):
            return PlannerDecision(
                action_type="response",
                action="talk",
                parameters={
                    "message": (
                        "No base-navigation tool is registered in this runtime; "
                        "I cannot execute the requested navigation safely."
                    )
                },
                reasoning="The default runtime has no executable base-navigation facade.",
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
    tool_contract_catalog: ToolContractCatalog | None = None,
    tool_contract_policy: ToolContractRuntimePolicy | None = None,
    tool_context: JsonDict | None = None,
) -> tuple[PlannerDecision, list[str]]:
    payload, parse_errors = _parse_backend_payload(result.payload)
    if parse_errors:
        return _invalid_decision(parse_errors), parse_errors

    decision, build_errors = _build_planner_decision(payload)
    canonicalizations = _canonicalize_host_parameters(
        decision,
        tool_context=tool_context or {},
    )
    if canonicalizations:
        decision.metadata["host_parameter_canonicalizations"] = canonicalizations
    catalog = tool_contract_catalog or _default_tool_contract_catalog()
    policy = tool_contract_policy or ToolContractRuntimePolicy()
    validation_errors = [
        *build_errors,
        *_validate_planner_decision(
            decision,
            tools,
            skills,
            tool_contract_catalog=catalog,
            tool_contract_policy=policy,
        ),
    ]
    contract_shadow = _tool_contract_shadow_validation(
        decision,
        tools=tools,
        tool_contract_catalog=catalog,
        tool_contract_policy=policy,
    )
    if contract_shadow is not None:
        decision.metadata["tool_contract_shadow_validation"] = contract_shadow
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
        return {}, [f"Planner backend payload must be dict or XML string, got {type(payload)}."]

    return _parse_xml_decision(payload)


# Leaf element names whose text must remain verbatim. This prevents identifiers
# and code-like text from being inferred as numbers merely because they happen
# to contain a numeric-looking value.
_XML_STRING_LEAF_NAMES = frozenset(
    {
        "camera_frame_id",
        "code",
        "kind",
        "message",
        "name",
        "prompt",
        "query",
        "reasoning",
        "skill",
        "source_packet_id",
        "target_mask",
        "tool",
        "url",
    }
)

# These child names are the explicit list vocabulary used by the planner wire
# contract. Other repeated child tags are objects and are rejected downstream
# if they do not match the requested ToolContract schema.
_XML_LIST_ITEM_NAMES = frozenset({"item", "call"})

# Some single-item arrays are naturally emitted as an XML object without an
# extra ``<item>`` wrapper. Keep this vocabulary deliberately narrow: these
# are wire-level collection fields whose element object is unambiguous. The
# downstream ToolContract validator remains the authority for the contents.
_XML_SINGLE_OBJECT_ARRAY_NAMES = frozenset({"points"})

_XML_TOKEN_PATTERN = re.compile(
    r"<!\[CDATA\[.*?\]\]>|<!--.*?-->|<[^>]+>",
    re.DOTALL,
)
_XML_OPEN_TAG_PATTERN = re.compile(r"<\s*([A-Za-z_][\w.:-]*)\b[^>]*>", re.DOTALL)
_XML_CLOSE_TAG_PATTERN = re.compile(r"</\s*([A-Za-z_][\w.:-]*)\s*>")


def _parse_xml_decision(payload: str) -> tuple[JsonDict, list[str]]:
    """Parse one main-planner ``<decision>`` into the existing decision shape.

    Host-generated failure payloads remain dictionaries and bypass this parser.
    Main-model strings are intentionally XML-only so a malformed or stale JSON
    response is visible to the existing validation-retry loop instead of being
    silently accepted through a compatibility path.
    """

    text = _strip_code_fence(payload)
    start = text.find("<decision")
    end = text.rfind("</decision>")
    if start == -1 or end == -1:
        return {}, ["Planner backend returned text without a <decision> element."]
    document = text[start : end + len("</decision>")]
    repair_tags: list[str] = []
    try:
        root = ET.fromstring(document)
    except ET.ParseError as original_exc:
        repaired_document, repair_tags = _repair_unclosed_xml_list_containers(document)
        if not repair_tags:
            return {}, [f"Planner backend returned invalid XML: {original_exc}"]
        try:
            root = ET.fromstring(repaired_document)
        except ET.ParseError:
            return {}, [f"Planner backend returned invalid XML: {original_exc}"]
    if root.tag != "decision":
        return {}, [f"Planner backend XML root must be <decision>, got <{root.tag}>."]

    value = _xml_element_to_value(root)
    if not isinstance(value, dict):
        return {}, ["Planner backend <decision> must decode to an object."]
    if repair_tags:
        value["_xml_wire_repair"] = {
            "schema_version": "openeta.planner_xml_repair.v1",
            "kind": "close_unclosed_list_container",
            "inserted_closing_tags": repair_tags,
            "count": len(repair_tags),
        }
    return value, []


def _repair_unclosed_xml_list_containers(document: str) -> tuple[str, list[str]]:
    """Close only list containers whose omitted end tag is unambiguous.

    The XML wire contract defines a container with ``<item>`` or ``<call>``
    children as a list.  Models occasionally emit a complete list and begin a
    sibling field without closing that container, for example ``<xyz><item>``
    followed by ``<quat_xyzw>``.  Re-parenting the sibling by inserting
    ``</xyz>`` is deterministic.  No object element, scalar, or arbitrary tag
    mismatch is repaired here; those remain visible to the validation retry
    loop.
    """

    output: list[str] = []
    stack: list[dict[str, object]] = []
    repaired: list[str] = []
    cursor = 0

    def close_intervening_lists(next_name: str) -> None:
        while stack and stack[-1]["is_list"] is True and next_name not in _XML_LIST_ITEM_NAMES:
            name = str(stack.pop()["name"])
            output.append(f"</{name}>")
            repaired.append(name)

    for match in _XML_TOKEN_PATTERN.finditer(document):
        output.append(document[cursor : match.start()])
        token = match.group(0)
        cursor = match.end()

        if token.startswith("<![CDATA[") or token.startswith("<!--"):
            output.append(token)
            continue
        if token.startswith("<?") or token.startswith("<!"):
            output.append(token)
            continue

        closing = _XML_CLOSE_TAG_PATTERN.fullmatch(token)
        if closing is not None:
            name = closing.group(1)
            while stack and str(stack[-1]["name"]) != name:
                if stack[-1]["is_list"] is not True:
                    break
                missing = str(stack.pop()["name"])
                output.append(f"</{missing}>")
                repaired.append(missing)
            output.append(token)
            if stack and str(stack[-1]["name"]) == name:
                stack.pop()
            continue

        opening = _XML_OPEN_TAG_PATTERN.fullmatch(token)
        if opening is None:
            output.append(token)
            continue
        name = opening.group(1)
        close_intervening_lists(name)
        output.append(token)
        if stack and name in _XML_LIST_ITEM_NAMES:
            stack[-1]["is_list"] = True
        if not token.rstrip().endswith("/>"):
            stack.append({"name": name, "is_list": False})

    output.append(document[cursor:])
    if not repaired:
        return document, []
    return "".join(output), repaired


def _xml_element_to_value(element: ET.Element) -> object:
    """Convert the XML wire vocabulary into JSON-equivalent Python values."""

    declared = (element.get("type") or "").strip().lower()
    children = list(element)
    if not children:
        if declared == "object":
            return {}
        if declared == "array":
            return []
        if element.tag == "parameters" and (element.text or "").strip() == "":
            return {}
        return _coerce_xml_scalar(element)

    child_tags = {child.tag for child in children}
    if declared == "array" or child_tags & _XML_LIST_ITEM_NAMES:
        return [_xml_element_to_value(child) for child in children]
    if element.tag in _XML_SINGLE_OBJECT_ARRAY_NAMES:
        if len(child_tags) == 1 and next(iter(child_tags)) in {"point"}:
            return [_xml_element_to_value(child) for child in children]
        return [
            {child.tag: _xml_element_to_value(child) for child in children}
        ]

    result: JsonDict = {}
    for child in children:
        result[child.tag] = _xml_element_to_value(child)
    return result


def _coerce_xml_scalar(element: ET.Element) -> object:
    """Coerce one XML leaf while preserving free text and CDATA verbatim.

    Explicit XML ``type`` attributes remain authoritative, while the natural
    scalar spellings ``true``, ``false``, and ``null`` are also inferred. This
    keeps planner XML equivalent to the former JSON wire format when a model
    omits a redundant type attribute. Reserved free-text fields are checked
    first and therefore remain strings even when they contain those words.
    """

    raw = element.text if element.text is not None else ""
    declared = (element.get("type") or "").strip().lower()
    if declared == "null":
        return None
    if declared == "boolean":
        return raw.strip().lower() in {"true", "1", "yes"}
    if declared == "string":
        return raw
    if declared == "integer":
        try:
            return int(raw.strip())
        except ValueError:
            return raw
    if declared == "number":
        try:
            return float(raw.strip())
        except ValueError:
            return raw
    if element.tag in _XML_STRING_LEAF_NAMES:
        return raw

    stripped = raw.strip()
    if stripped == "":
        return raw
    lowered = stripped.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        return raw


def _strip_code_fence(text: str) -> str:
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
    if (
        "calls" in payload
        and raw_action_type == "tool_call"
        and raw_name in {"tool_batch", "batch"}
    ):
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
    *,
    tool_contract_catalog: ToolContractCatalog | None = None,
    tool_contract_policy: ToolContractRuntimePolicy | None = None,
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
                elif not tools.can_execute(tool_name):
                    errors.append(
                        f"Safety tool requested by planner is not executable: {tool_name}."
                    )
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
                if not tools.can_execute(decision.action):
                    errors.append(
                        f"Tool requested by planner is not executable: {decision.action}."
                    )
                else:
                    policy = tool_contract_policy or ToolContractRuntimePolicy()
                    if policy.request_is_authoritative(decision.action):
                        catalog = tool_contract_catalog or _default_tool_contract_catalog()
                        contract = catalog.get(decision.action)
                        errors.extend(
                            _format_tool_contract_request_violations(
                                decision.action,
                                check_tool_request_conformance(
                                    contract,
                                    decision.parameters,
                                ),
                            )
                        )
                    else:
                        errors.extend(
                            _validate_tool_parameters(
                                decision.action,
                                decision.parameters,
                            )
                        )

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
        else:
            if not tools.can_execute(name):
                errors.append(f"tool_batch call {idx} requested unbound tool: {name}.")
    return errors


_STATIC_TOOL_PARAMETER_RULES: dict[str, JsonDict] = {
    "observe": {
        "types": {"reason": "string"},
    },
    "create_simulator_env": {
        "required": ("env_id",),
        "types": {
            "env_id": "string",
            "seed": "integer",
            "task": "string",
            "render_mode": "string",
            "image_width": "integer",
            "image_height": "integer",
            "session_id": "string",
            "include_objects": "boolean",
        },
    },
    "close_simulator_env": {"types": {}},
    "retrieve_asset_reference": {
        "required": ("environment", "target_object", "source_packet_id"),
        "types": {
            "environment": "string",
            "target_object": "string",
            "source_packet_id": "string",
            "camera_frame_id": "string",
        },
    },
    "select_sam3_detection": {
        "required": ("sam3_result_id", "detection_id"),
        "types": {
            "sam3_result_id": "string",
            "detection_id": "string",
            "selection_confidence": "number",
            "reason": "string",
            "identity_anchor_id": "string",
            "identity_relation": "string",
            "evidence_role": "string",
            "target_geometry_family": "string",
        },
        "enums": {
            "identity_relation": {"same_instance", "replace_misidentified_anchor"},
            "evidence_role": {"target_object", "placement_region"},
        },
    },
    "reject_sam3_detections": {
        "required": ("sam3_result_id", "reason"),
        "types": {"sam3_result_id": "string", "reason": "string"},
    },
    "compile_grasp_seed": {
        "required": ("grasp_result_id", "candidate_id"),
        "types": {
            "grasp_result_id": "string",
            "candidate_id": "string",
            "target_geometry_family": "string",
            "target_class": "string",
            "strategy_id": "string",
            "articulated_handle_options": "object",
            "pregrasp_distance_m": "number",
        },
    },
    "compute_wrist_alignment": {
        "required": ("bundle_id",),
        "types": {"bundle_id": "string", "max_correction_m": "number"},
    },
    "propose_wrist_viewpoints": {
        "required": ("compiled_grasp_id", "source_packet_id", "camera_frame_id"),
        "types": {
            "compiled_grasp_id": "string",
            "source_packet_id": "string",
            "camera_frame_id": "string",
        },
    },
    "prepare_attachment_probe": {
        "required": ("compiled_grasp_id", "motion_type"),
        "types": {
            "compiled_grasp_id": "string",
            "motion_type": "string",
            "direction_world_xyz": "array",
            "waypoint_offsets_world_xyz": "array",
            "reason": "string",
        },
        "enums": {"motion_type": {"linear", "arc"}},
    },
    "assess_attachment_probe": {
        "required": ("probe_id",),
        "types": {"probe_id": "string"},
    },
    "move_to": {
        "required": ("ik_receipt_id",),
        "types": {
            "ik_receipt_id": "string",
            "num_steps": "integer",
            "tolerance": "number",
            "ori_tolerance": "number",
            "enable_collision_check": "boolean",
        },
    },
    "follow_eef_trajectory": {
        "required": ("ik_receipt_ids",),
        "types": {
            "ik_receipt_ids": "array",
            "num_steps_per_waypoint": "integer",
            "tolerance": "number",
            "ori_tolerance": "number",
            "enable_collision_check": "boolean",
        },
    },
    "ik_preview_check": {
        "types": {
            "target_pose": "object",
            "compiled_grasp_id": "string",
            "waypoint_role": "string",
            "path_fraction": "number",
            "viewpoint_proposal_id": "string",
            "candidate_id": "string",
            "probe_id": "string",
            "waypoint_index": "integer",
            "position_tolerance_m": "number",
            "orientation_tolerance_rad": "number",
            "preserve_current_orientation": "boolean",
            "check_endpoint_collision": "boolean",
        },
        "enums": {
            "waypoint_role": {
                "grasp_clearance",
                "grasp_precontact",
                "grasp_alignment_reference",
                "grasp_contact",
            }
        },
        "one_of_required": (
            ("target_pose",),
            ("compiled_grasp_id", "waypoint_role"),
            ("compiled_grasp_id", "path_fraction"),
            ("viewpoint_proposal_id", "candidate_id"),
            ("probe_id", "waypoint_index"),
        ),
    },
}


def _validate_static_tool_parameters(
    tool_name: str,
    parameters: JsonDict,
) -> list[str]:
    rules = _STATIC_TOOL_PARAMETER_RULES[tool_name]
    types = rules.get("types")
    types = types if isinstance(types, dict) else {}
    errors = _unsupported_parameter_errors(tool_name, parameters, set(types))
    for name in rules.get("required", ()):
        if name not in parameters:
            errors.append(f"{tool_name} requires `parameters.{name}`.")
    one_of = rules.get("one_of_required")
    if isinstance(one_of, tuple):
        matched = sum(
            all(name in parameters for name in group)
            for group in one_of
            if isinstance(group, tuple)
        )
        if matched != 1:
            errors.append(f"{tool_name} requires exactly one supported reference shape.")
    for name, expected in types.items():
        if name not in parameters:
            continue
        value = parameters[name]
        valid = {
            "string": isinstance(value, str),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "boolean": isinstance(value, bool),
            "array": isinstance(value, list),
            "object": isinstance(value, dict),
        }.get(str(expected), True)
        if not valid:
            errors.append(f"{tool_name} `parameters.{name}` must be {expected}.")
            continue
        if expected == "string" and not value:
            errors.append(f"{tool_name} `parameters.{name}` must be non-empty.")
    enums = rules.get("enums")
    enums = enums if isinstance(enums, dict) else {}
    for name, allowed in enums.items():
        if name in parameters and parameters[name] not in allowed:
            errors.append(f"{tool_name} `parameters.{name}` has an unsupported value.")
    return errors


def _validate_tool_parameters(tool_name: str, parameters: JsonDict) -> list[str]:
    if tool_name in _STATIC_TOOL_PARAMETER_RULES:
        return _validate_static_tool_parameters(tool_name, parameters)
    if tool_name == "python_exec":
        return _validate_python_exec_parameters(parameters)
    if tool_name == "web_search":
        return _validate_web_search_parameters(parameters)
    if tool_name == "web_fetch":
        return _validate_web_fetch_parameters(parameters)
    if tool_name == "sam3":
        return _validate_sam3_parameters(parameters)
    if tool_name == "molmopoint":
        return _validate_molmopoint_parameters(parameters)
    if tool_name == "estimate_depth_prior":
        return _validate_depth_packet_parameters(
            tool_name,
            parameters,
            optional_fields={"resolution_level"},
        )
    if tool_name == "enhance_depth":
        return _validate_depth_packet_parameters(
            tool_name,
            parameters,
            optional_fields={"config"},
        )
    if tool_name == "anyplace":
        return _validate_anyplace_parameters(parameters)
    if tool_name == "camera_pose_to_world":
        return _validate_camera_pose_to_world_parameters(parameters)
    if tool_name == "gripper_control":
        return _validate_gripper_control_parameters(parameters)
    if tool_name == "grasp_pose_estimate":
        return _validate_grasp_pose_estimate_parameters(parameters)
    if tool_name == "propose_calibration_profile":
        return _validate_proposal_parameters(
            tool_name,
            parameters,
            payload_field="profile",
            extra_fields={"profile_fingerprint": dict, "validation_gates": list, "ledger": list},
            required_extra=("profile_fingerprint",),
        )
    if tool_name == "propose_grasp_strategy":
        return _validate_proposal_parameters(
            tool_name,
            parameters,
            payload_field="strategy",
            extra_fields={
                "base_strategy_sha256": str,
                "rollout_summary": dict,
                "ledger": list,
            },
        )
    if tool_name in {"promote_calibration_profile", "promote_grasp_strategy"}:
        return _validate_promotion_parameters(tool_name, parameters)
    if tool_name in {"save_memory", "get_memory", "delete_memory", "compact_memory"}:
        return _validate_memory_tool_parameters(tool_name, parameters)
    if tool_name in {"register_skill", "update_skill"}:
        return _validate_skill_management_parameters(tool_name, parameters)
    if tool_name == "graspgenx":
        return _validate_graspgenx_parameters(parameters)
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


def _validate_python_exec_parameters(parameters: JsonDict) -> list[str]:
    errors = _unsupported_parameter_errors(
        "python_exec",
        parameters,
        {"code", "sandbox", "timeout_s"},
    )
    code = parameters.get("code")
    if not isinstance(code, str) or not code:
        errors.append("python_exec requires a non-empty string `parameters.code`.")
    sandbox = parameters.get("sandbox")
    if sandbox is not None and sandbox not in {"sandbox", "outside_sandbox"}:
        errors.append("python_exec sandbox must be sandbox or outside_sandbox.")
    timeout_s = parameters.get("timeout_s")
    if timeout_s is not None and (
        isinstance(timeout_s, bool)
        or not isinstance(timeout_s, (int, float))
        or not 0 < float(timeout_s) <= 600
    ):
        errors.append("python_exec timeout_s must be a number in (0, 600].")
    return errors


def _validate_proposal_parameters(
    tool_name: str,
    parameters: JsonDict,
    *,
    payload_field: str,
    extra_fields: dict[str, type],
    required_extra: tuple[str, ...] = (),
) -> list[str]:
    allowed = {payload_field, "rationale", *extra_fields}
    errors = _unsupported_parameter_errors(tool_name, parameters, allowed)
    payload = parameters.get(payload_field)
    if not isinstance(payload, dict):
        errors.append(f"{tool_name} requires object `parameters.{payload_field}`.")
    rationale = parameters.get("rationale")
    if not isinstance(rationale, str) or not rationale:
        errors.append(f"{tool_name} requires non-empty string `parameters.rationale`.")
    for name in required_extra:
        if name not in parameters:
            errors.append(f"{tool_name} requires `parameters.{name}`.")
    for name, expected_type in extra_fields.items():
        value = parameters.get(name)
        if value is not None and not isinstance(value, expected_type):
            errors.append(
                f"{tool_name} `parameters.{name}` must be {expected_type.__name__}."
            )
    return errors


def _validate_promotion_parameters(
    tool_name: str,
    parameters: JsonDict,
) -> list[str]:
    errors = _unsupported_parameter_errors(
        tool_name,
        parameters,
        {"proposal_id", "target_status", "evidence"},
    )
    proposal_id = parameters.get("proposal_id")
    if not isinstance(proposal_id, str) or not proposal_id:
        errors.append(f"{tool_name} requires non-empty string `parameters.proposal_id`.")
    if parameters.get("target_status") not in {"candidate", "validated"}:
        errors.append(f"{tool_name} target_status must be candidate or validated.")
    evidence = parameters.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        errors.append(f"{tool_name} requires a non-empty evidence list.")
    else:
        for index, item in enumerate(evidence):
            if not isinstance(item, dict):
                errors.append(f"{tool_name} evidence[{index}] must be an object.")
                continue
            if set(item) - {"path", "split"}:
                errors.append(f"{tool_name} evidence[{index}] has unsupported fields.")
            if not isinstance(item.get("path"), str) or not item.get("path"):
                errors.append(f"{tool_name} evidence[{index}].path must be non-empty.")
            if item.get("split") not in {"canary", "held_out"}:
                errors.append(
                    f"{tool_name} evidence[{index}].split must be canary or held_out."
                )
    return errors


def _validate_memory_tool_parameters(
    tool_name: str,
    parameters: JsonDict,
) -> list[str]:
    allowed_by_tool = {
        "save_memory": {"namespace", "key", "content", "tags"},
        "get_memory": {"namespace", "key"},
        "delete_memory": {"namespace", "key"},
        "compact_memory": {"max_events"},
    }
    errors = _unsupported_parameter_errors(
        tool_name,
        parameters,
        allowed_by_tool[tool_name],
    )
    if tool_name == "compact_memory":
        max_events = parameters.get("max_events")
        if max_events is not None and (
            isinstance(max_events, bool)
            or not isinstance(max_events, int)
            or max_events < 1
        ):
            errors.append("compact_memory max_events must be an integer >= 1.")
        return errors
    namespace = parameters.get("namespace")
    allowed_namespaces = {"facts", "artifacts", "skill_notes"}
    if tool_name != "save_memory":
        allowed_namespaces.add("all")
    if namespace is not None and namespace not in allowed_namespaces:
        errors.append(
            f"{tool_name} namespace must be one of {sorted(allowed_namespaces)}."
        )
    key = parameters.get("key")
    if key is not None and (not isinstance(key, str) or not key):
        errors.append(f"{tool_name} key must be a non-empty string when supplied.")
    if tool_name in {"save_memory", "delete_memory"} and "key" not in parameters:
        errors.append(f"{tool_name} requires `parameters.key`.")
    if tool_name == "save_memory":
        if "content" not in parameters:
            errors.append("save_memory requires `parameters.content`.")
        tags = parameters.get("tags")
        if tags is not None and (
            not isinstance(tags, list)
            or any(not isinstance(tag, str) for tag in tags)
        ):
            errors.append("save_memory tags must be an array of strings.")
    return errors


def _validate_skill_management_parameters(
    tool_name: str,
    parameters: JsonDict,
) -> list[str]:
    allowed = (
        {
            "name",
            "goal",
            "description",
            "requirements",
            "examples",
            "content",
            "task_patterns",
            "allowed_tools",
        }
        if tool_name == "register_skill"
        else {"name", "requested_changes", "examples", "requirements", "content"}
    )
    errors = _unsupported_parameter_errors(tool_name, parameters, allowed)
    name = parameters.get("name")
    if not isinstance(name, str) or not name:
        errors.append(f"{tool_name} requires non-empty string `parameters.name`.")
    elif tool_name == "register_skill" and re.fullmatch(
        r"[a-z0-9]+(?:-[a-z0-9]+)*", name
    ) is None:
        errors.append("register_skill name must be a lowercase hyphenated slug.")
    string_fields = (
        {"goal", "description", "content"}
        if tool_name == "register_skill"
        else {"requested_changes", "content"}
    )
    for field in string_fields:
        value = parameters.get(field)
        if value is not None and not isinstance(value, str):
            errors.append(f"{tool_name} {field} must be a string when supplied.")
    if tool_name == "register_skill":
        for field in ("task_patterns", "allowed_tools"):
            value = parameters.get(field)
            if value is not None and (
                not isinstance(value, list)
                or any(not isinstance(item, str) for item in value)
            ):
                errors.append(f"register_skill {field} must be an array of strings.")
    return errors


def _unsupported_parameter_errors(
    tool_name: str,
    parameters: JsonDict,
    allowed: set[str],
) -> list[str]:
    extras = sorted(set(parameters) - allowed)
    if not extras:
        return []
    return [f"{tool_name} received unsupported parameters: {', '.join(extras)}."]


def _tool_contract_shadow_validation(
    decision: PlannerDecision,
    *,
    tools: ToolRegistry,
    tool_contract_catalog: ToolContractCatalog | None = None,
    tool_contract_policy: ToolContractRuntimePolicy | None = None,
) -> JsonDict | None:
    """Record legacy/contract parity and the selected per-tool authority."""

    if decision.action_type.strip().lower() != "tool_call":
        return None
    if decision.action in {
        "sense",
        "tool_batch",
        "batch",
        "safe_check",
        "code_policy",
        "skill_call",
    }:
        return None
    try:
        tools.get(decision.action)
    except KeyError:
        return None

    from agent.tools.contracts import ContractMaturity

    try:
        contract = (tool_contract_catalog or _default_tool_contract_catalog()).get(
            decision.action
        )
    except KeyError:
        return None
    if contract.maturity is ContractMaturity.INFERRED:
        return {
            "schema_version": TOOL_CONTRACT_SHADOW_VALIDATION_SCHEMA_VERSION,
            "tool": decision.action,
            "contract_maturity": contract.maturity.value,
            "evaluated": False,
            "enforcing": False,
            "authoritative_validator": "legacy_planner",
            "reason": "inferred contracts are inventory-only",
        }

    policy = tool_contract_policy or ToolContractRuntimePolicy()
    contract_authoritative = policy.request_is_authoritative(decision.action)
    legacy_errors = _validate_tool_parameters(decision.action, decision.parameters)
    contract_violations = check_tool_request_conformance(
        contract,
        decision.parameters,
    )
    legacy_accepted = not legacy_errors
    contract_accepted = not contract_violations
    return {
        "schema_version": TOOL_CONTRACT_SHADOW_VALIDATION_SCHEMA_VERSION,
        "tool": decision.action,
        "contract_maturity": contract.maturity.value,
        "evaluated": True,
        "enforcing": contract_authoritative,
        "authoritative_validator": (
            "tool_contract" if contract_authoritative else "legacy_planner"
        ),
        "legacy_accepted": legacy_accepted,
        "contract_accepted": contract_accepted,
        "acceptance_match": legacy_accepted == contract_accepted,
        "legacy_errors": list(legacy_errors),
        "contract_violations": [
            violation.to_dict() for violation in contract_violations
        ],
    }


def _format_tool_contract_request_violations(
    tool_name: str,
    violations: tuple[object, ...],
) -> list[str]:
    """Turn structured contract failures into concise Agent repair feedback."""

    errors: list[str] = []
    for violation in violations:
        code = str(getattr(violation, "code", "request_contract_violation"))
        path = str(getattr(violation, "path", "parameters"))
        message = str(getattr(violation, "message", "request does not match schema"))
        errors.append(
            f"{tool_name} request violates ToolContract at `{path}` ({code}): {message}"
        )
    return errors


@lru_cache(maxsize=1)
def _default_tool_contract_catalog():
    """Cache immutable default declarations used by non-enforcing shadow checks."""

    from agent.tools.contracts import build_default_tool_contract_catalog
    from agent.tools.registry import build_default_tool_registry

    return build_default_tool_contract_catalog(build_default_tool_registry().list())


def _validate_gripper_control_parameters(parameters: JsonDict) -> list[str]:
    """Keep the Agent-facing latch command explicitly binary.

    The simulator reports a continuous measured aperture, but that observation
    must never leak back into the command contract as a fractional target.
    """

    if set(parameters) != {"position"}:
        return [
            "gripper_control requires exactly `parameters.position`; use binary "
            "0=closed or 1=open, not a measured aperture or fractional command."
        ]
    position = parameters.get("position")
    if isinstance(position, bool):
        return []
    if isinstance(position, int | float) and not isinstance(position, bool):
        numeric = float(position)
        if math.isfinite(numeric) and numeric in {0.0, 1.0}:
            return []
    return [
        "gripper_control `parameters.position` must be exactly binary 0=closed "
        "or 1=open; measured gripper aperture is observation-only."
    ]


def _validate_web_search_parameters(parameters: JsonDict) -> list[str]:
    errors = _unsupported_parameter_errors(
        "web_search",
        parameters,
        {"query", "max_results", "language", "time_range"},
    )
    query = parameters.get("query")
    if not isinstance(query, str) or not query.strip() or len(query.strip()) > 512:
        errors.append(
            "web_search requires `parameters.query` as 1-512 characters of public "
            "search terms without secrets or private user data."
        )
    max_results = parameters.get("max_results", 5)
    if (
        isinstance(max_results, bool)
        or not isinstance(max_results, int)
        or not 1 <= max_results <= 10
    ):
        errors.append("web_search `parameters.max_results` must be an integer from 1 to 10.")
    time_range = str(parameters.get("time_range") or "").strip().lower()
    if time_range not in {"", "day", "month", "year"}:
        errors.append("web_search `parameters.time_range` must be empty, day, month, or year.")
    return errors


def _validate_web_fetch_parameters(parameters: JsonDict) -> list[str]:
    errors = _unsupported_parameter_errors(
        "web_fetch",
        parameters,
        {"url", "max_chars"},
    )
    url = parameters.get("url")
    if not isinstance(url, str) or not url.strip() or len(url.strip()) > 2048:
        errors.append(
            "web_fetch requires `parameters.url` as an absolute public HTTPS URL of "
            "at most 2048 characters."
        )
    else:
        parsed = urllib.parse.urlsplit(url.strip())
        if (
            parsed.scheme.lower() != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
        ):
            errors.append(
                "web_fetch `parameters.url` must be an absolute HTTPS URL without "
                "embedded credentials."
            )
    max_chars = parameters.get("max_chars", 12_000)
    if (
        isinstance(max_chars, bool)
        or not isinstance(max_chars, int)
        or not 1 <= max_chars <= 40_000
    ):
        errors.append("web_fetch `parameters.max_chars` must be an integer from 1 to 40000.")
    return errors


def _validate_sam3_parameters(parameters: JsonDict) -> list[str]:
    errors = _unsupported_parameter_errors(
        "sam3",
        parameters,
        {
            "source_packet_id",
            "camera_frame_id",
            "mode",
            "prompt",
            "points",
            "positive_points",
            "roi_bbox_xyxy",
            "evidence_role",
        },
    )
    source_packet_id = parameters.get("source_packet_id")
    if not isinstance(source_packet_id, str) or not source_packet_id.strip():
        errors.append(
            "sam3 requires `parameters.source_packet_id` copied exactly from visible "
            "observation evidence. Local image paths are not accepted."
        )
    if "image" in parameters:
        errors.append(
            "sam3 no longer accepts `parameters.image`; use source_packet_id so the "
            "host can resolve session-owned artifacts and provenance."
        )
    camera_frame_id = parameters.get("camera_frame_id")
    if camera_frame_id is not None and (
        not isinstance(camera_frame_id, str) or not camera_frame_id.strip()
    ):
        errors.append("sam3 `parameters.camera_frame_id` must be a non-empty string when set.")
    legacy_points = parameters.get("positive_points")
    mode = (
        str(parameters.get("mode") or ("points" if legacy_points is not None else "text"))
        .strip()
        .lower()
    )
    if mode not in {"text", "points"}:
        errors.append("sam3 `parameters.mode` must be `text` or `points`.")
        return errors
    prompt = parameters.get("prompt")
    points = parameters.get("points")
    if points is None and legacy_points is not None:
        points = legacy_points
    if mode == "text":
        if (
            not isinstance(prompt, str)
            or not prompt.strip()
            or _looks_like_placeholder_prompt(prompt)
        ):
            errors.append(
                "sam3 text mode requires `parameters.prompt` as a concrete visual phrase."
            )
        if isinstance(points, list) and points:
            errors.append("sam3 text mode must not include non-empty `parameters.points`.")
        return errors
    if isinstance(prompt, str) and prompt.strip():
        errors.append("sam3 points mode must not include a non-empty `parameters.prompt`.")
    if not isinstance(points, list) or not 1 <= len(points) <= 64:
        errors.append(
            "sam3 points mode requires `parameters.points` as a list of one to 64 points."
        )
        return errors
    foreground_count = 0
    for point_index, point in enumerate(points):
        if not isinstance(point, dict) or set(point) != {"x", "y", "label"}:
            errors.append(f"sam3 point {point_index} must contain exactly x, y, and label.")
            continue
        x = point.get("x")
        y = point.get("y")
        label = point.get("label")
        if (
            isinstance(x, bool)
            or isinstance(y, bool)
            or not isinstance(x, (int, float))
            or not isinstance(y, (int, float))
            or not math.isfinite(float(x))
            or not math.isfinite(float(y))
            or float(x) < 0
            or float(y) < 0
            or isinstance(label, bool)
            or not isinstance(label, int)
            or label not in {0, 1}
        ):
            errors.append(f"sam3 point {point_index} requires finite numeric x/y and label 0 or 1.")
            continue
        foreground_count += int(label == 1)
    if foreground_count == 0:
        errors.append("sam3 points mode requires at least one foreground point with label=1.")
    return errors


def _validate_molmopoint_parameters(parameters: JsonDict) -> list[str]:
    errors = _unsupported_parameter_errors(
        "molmopoint",
        parameters,
        {"sources", "prompt"},
    )
    if "images" in parameters:
        errors.append(
            "molmopoint no longer accepts `parameters.images`; use ordered packet "
            "sources so the host resolves session-owned images and provenance."
        )
    sources = parameters.get("sources")
    if not isinstance(sources, list) or not 1 <= len(sources) <= 4:
        errors.append(
            "molmopoint requires `parameters.sources` as an ordered list of one to "
            "four objects containing source_packet_id and optional camera_frame_id."
        )
    else:
        for source_index, source in enumerate(sources):
            if not isinstance(source, dict):
                errors.append(
                    f"molmopoint sources[{source_index}] must be an object containing "
                    "source_packet_id and optional camera_frame_id."
                )
                continue
            extra = sorted(set(source) - {"source_packet_id", "camera_frame_id"})
            if extra:
                errors.append(
                    f"molmopoint sources[{source_index}] contains unsupported fields: "
                    + ", ".join(extra)
                    + ". Do not pass paths or copied observation payloads."
                )
            source_packet_id = source.get("source_packet_id")
            if not isinstance(source_packet_id, str) or not source_packet_id.strip():
                errors.append(
                    f"molmopoint sources[{source_index}].source_packet_id must be a "
                    "non-empty id copied exactly from visible observation evidence."
                )
            camera_frame_id = source.get("camera_frame_id")
            if camera_frame_id is not None and (
                not isinstance(camera_frame_id, str) or not camera_frame_id.strip()
            ):
                errors.append(
                    f"molmopoint sources[{source_index}].camera_frame_id must be a "
                    "non-empty string when set."
                )
    prompt = parameters.get("prompt")
    if (
        not isinstance(prompt, str)
        or not prompt.strip()
        or len(prompt.strip()) > 1024
        or _looks_like_placeholder_prompt(prompt)
    ):
        errors.append(
            "molmopoint requires `parameters.prompt` as a complete pointing instruction "
            "of at most 1024 characters, not a placeholder."
        )
    return errors


def _validate_depth_packet_parameters(
    tool_name: str,
    parameters: JsonDict,
    *,
    optional_fields: set[str],
) -> list[str]:
    errors: list[str] = []
    allowed = {"source_packet_id", "camera_frame_id", *optional_fields}
    extra = sorted(set(parameters) - allowed)
    if extra:
        errors.append(
            f"{tool_name} accepts observation packet references, not model-supplied "
            "paths or calibration payloads; unsupported fields: "
            + ", ".join(extra)
            + "."
        )
    source_packet_id = parameters.get("source_packet_id")
    if not isinstance(source_packet_id, str) or not source_packet_id.strip():
        errors.append(
            f"{tool_name} requires `parameters.source_packet_id` copied exactly "
            "from visible observation evidence."
        )
    camera_frame_id = parameters.get("camera_frame_id")
    if camera_frame_id is not None and (
        not isinstance(camera_frame_id, str) or not camera_frame_id.strip()
    ):
        errors.append(
            f"{tool_name} `parameters.camera_frame_id` must be a non-empty string when set."
        )
    if "resolution_level" in optional_fields and "resolution_level" in parameters:
        value = parameters.get("resolution_level")
        if (
            isinstance(value, bool)
            or not isinstance(value, int)
            or not 0 <= value < 10
        ):
            errors.append(
                "estimate_depth_prior `parameters.resolution_level` must be an integer "
                "in [0, 10)."
            )
    if "config" in optional_fields and "config" in parameters:
        if not isinstance(parameters.get("config"), dict):
            errors.append("enhance_depth `parameters.config` must be an object when set.")
    return errors


def _validate_grasp_pose_estimate_parameters(parameters: JsonDict) -> list[str]:
    bundle_id = parameters.get("bundle_id")
    if isinstance(bundle_id, str) and bundle_id.strip():
        extra = sorted(
            str(key)
            for key in parameters
            if key not in {"bundle_id", "backend_preference"}
        )
        if extra:
            return [
                "grasp_pose_estimate bundle references cannot be overridden with "
                "model-supplied fields: "
                + ", ".join(extra)
                + "."
            ]
        return _validate_grasp_backend_preference(parameters.get("backend_preference"))
    errors: list[str] = []
    mode = str(parameters.get("mode") or "targeted").strip().lower()
    if mode not in {"targeted", "scene"}:
        errors.append("grasp_pose_estimate mode must be targeted or scene.")
    for key in ("rgb", "depth"):
        value = parameters.get(key)
        if not isinstance(value, str) or not value.strip() or _looks_like_placeholder_path(value):
            errors.append(
                f"grasp_pose_estimate requires `parameters.{key}` as a concrete local path."
            )
    object_mask = parameters.get("object_mask")
    if mode == "targeted":
        if not isinstance(object_mask, dict):
            errors.append(
                "grasp_pose_estimate targeted mode requires `parameters.object_mask` "
                "as a complete SAM3 artifact with mask_ref and source_image."
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
                        f"grasp_pose_estimate object_mask requires a concrete `{key}` local path."
                    )
    elif object_mask is not None:
        errors.append("grasp_pose_estimate scene mode does not accept object_mask.")
    _validate_required_intrinsics(
        parameters.get("intrinsics"),
        label="grasp_pose_estimate `parameters.intrinsics`",
        errors=errors,
    )
    frame_id = parameters.get("camera_frame_id")
    if not isinstance(frame_id, str) or not frame_id.strip():
        errors.append("grasp_pose_estimate requires a concrete camera_frame_id.")
    scene_epoch = parameters.get("scene_epoch")
    if isinstance(scene_epoch, bool) or not isinstance(scene_epoch, int) or scene_epoch < 0:
        errors.append("grasp_pose_estimate requires the current non-negative scene_epoch.")
    hints = parameters.get("hints")
    if hints is not None and not isinstance(hints, dict):
        errors.append("grasp_pose_estimate hints must be an object when provided.")
    errors.extend(_validate_grasp_backend_preference(parameters.get("backend_preference")))
    return errors


def _validate_grasp_backend_preference(value: object) -> list[str]:
    if value is None:
        return []
    allowed = ", ".join(GRASP_POSE_BACKENDS)
    if not isinstance(value, list) or not value:
        return [
            "grasp_pose_estimate backend_preference must be a non-empty ordered "
            f"list chosen from: {allowed}."
        ]
    if any(not isinstance(item, str) or not item.strip() for item in value):
        return [
            "grasp_pose_estimate backend_preference entries must be non-empty "
            f"backend names chosen from: {allowed}."
        ]
    normalized = [str(item).strip().lower() for item in value]
    unknown = sorted(set(normalized).difference(GRASP_POSE_BACKENDS))
    if unknown:
        return [
            "grasp_pose_estimate backend_preference contains unknown backend(s): "
            + ", ".join(unknown)
            + f". Allowed backends: {allowed}."
        ]
    duplicates = sorted({item for item in normalized if normalized.count(item) > 1})
    if duplicates:
        return [
            "grasp_pose_estimate backend_preference must not repeat backend(s): "
            + ", ".join(duplicates)
            + "."
        ]
    return []


def _validate_graspgenx_parameters(parameters: JsonDict) -> list[str]:
    errors: list[str] = []
    for key in ("rgb", "depth"):
        value = parameters.get(key)
        if not isinstance(value, str) or not value.strip() or _looks_like_placeholder_path(value):
            errors.append(f"graspgenx requires `parameters.{key}` as a concrete local file path.")
    object_mask = parameters.get("object_mask")
    if not isinstance(object_mask, dict):
        errors.append(
            "graspgenx requires `parameters.object_mask` as a SAM3 artifact "
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
                errors.append(f"graspgenx object_mask requires a concrete `{key}` local path.")
    _validate_required_intrinsics(
        parameters.get("intrinsics"),
        label="graspgenx `parameters.intrinsics`",
        errors=errors,
    )
    gripper_name = parameters.get("gripper_name")
    if (
        not isinstance(gripper_name, str)
        or not gripper_name.strip()
        or (gripper_name.strip().startswith("<") and gripper_name.strip().endswith(">"))
    ):
        errors.append("graspgenx requires `parameters.gripper_name` from list_graspgenx_grippers.")
    up = parameters.get("up_direction_camera")
    if (
        not isinstance(up, list)
        or len(up) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in (up if isinstance(up, list) else [])
        )
    ):
        errors.append(
            "graspgenx requires `parameters.up_direction_camera` as three finite numbers."
        )
    return errors




def _required_skill_inspection_name(
    decision: PlannerDecision,
    *,
    tools: ToolRegistry,
    tool_context: JsonDict,
) -> str:
    skill_usage = tool_context.get("skill_usage")
    if not isinstance(skill_usage, dict):
        return ""
    required = skill_usage.get("inspection_required")
    if not isinstance(required, list) or not required:
        return ""
    required_name = str(required[0] or "").strip()
    if not required_name:
        return ""
    if _is_skill_decision(decision) and _skill_decision_name(decision) == required_name:
        return ""
    if decision.action_type.lower().strip() != "tool_call":
        return ""
    try:
        spec = tools.get(decision.action)
    except KeyError:
        return ""
    if not spec.requires_observation_after_call:
        return ""
    return required_name


def _required_skill_inspection_error(required_name: str) -> str:
    return (
        f"Selected skill {required_name!r} is truncated and must be inspected with "
        "tool_call::skill_call before world-mutating control."
    )


def _validate_anyplace_parameters(parameters: JsonDict) -> list[str]:
    bundle_id = parameters.get("bundle_id")
    if not isinstance(bundle_id, str) or not bundle_id.strip():
        return [
            "anyplace requires only the host-issued `parameters.bundle_id`; inspect "
            "host_resolved_inputs.anyplace and do not reconstruct RGB-D, mask, "
            "intrinsics, or selected_grasp fields."
        ]
    extra = sorted(str(key) for key in parameters if key != "bundle_id")
    if extra:
        return [
            "anyplace bundle references cannot be overridden with model-supplied fields: "
            + ", ".join(extra)
            + "."
        ]
    return []


def _validate_camera_pose_to_world_parameters(parameters: JsonDict) -> list[str]:
    """Keep AnyPlace handoff opaque while preserving the generic geometry path."""

    result_id = parameters.get("placement_result_id")
    candidate_id = parameters.get("candidate_id")
    uses_placement_reference = result_id is not None or candidate_id is not None
    if not uses_placement_reference:
        return []
    if not isinstance(result_id, str) or not result_id.strip():
        return [
            "camera_pose_to_world AnyPlace handoff requires a non-empty "
            "`placement_result_id` returned by anyplace."
        ]
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        return [
            "camera_pose_to_world AnyPlace handoff requires an exact non-empty "
            "`candidate_id` from that placement result."
        ]
    extra = sorted(
        str(key)
        for key in parameters
        if key not in {"placement_result_id", "candidate_id"}
    )
    if extra:
        return [
            "camera_pose_to_world placement references cannot be overridden with "
            "model-supplied pose or calibration fields: "
            + ", ".join(extra)
            + "."
        ]
    return []


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


def _looks_like_placeholder_prompt(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized.startswith("<") and normalized.endswith(">"):
        return True
    return normalized in {"prompt", "pointing_prompt", "your prompt", "insert prompt here"}


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


def _agent_owned_tool_planner_system_prompt() -> str:
    """Return the task-agnostic planner role and response contract.

    Closed-loop invariants are appended from ``embodied_closed_loop.md``.
    Task procedures belong to selected skills/playbooks, while exact request,
    result, receipt, and bundle semantics belong to the live tool contracts.
    """

    return (
        "You are the OpenETA closed-loop embodied planner. Return exactly one XML "
        "<decision> element with child elements kind, name, reasoning, and parameters. "
        "kind must be tool_call or response. name is the executable tool name for a "
        "tool_call, or ask_human, talk, or task_complete for a response. Put tool "
        "arguments inside <parameters> as named child elements and use an empty "
        "<parameters/> when there are none. Encode arrays as repeated <item> children "
        "and nested objects as named child elements. Wrap code and all multi-line or "
        "quoted text in CDATA so literal newlines survive; do not encode them as "
        "backslash-n. Numbers may use type=\"integer\" or type=\"number\"; booleans "
        "and nulls require type=\"boolean\" and type=\"null\". A tool_batch encodes "
        "<calls> with repeated <call> children, each containing name and parameters. "
        "Example: <decision><kind>tool_call</kind><name>python_exec</name>"
        "<reasoning>Inspect an artifact.</reasoning><parameters><code><![CDATA["
        "\nresult = artifacts.read_json('/workspace/result.json')\n"
        "]]></code></parameters></decision>. "
        "For tool_call choose exactly one executable atomic "
        "tool from available_tools; tool_references is only a legacy name index. "
        "You own task decomposition, progress assessment, recovery choice, and the next "
        "task action. The host does not provide a task phase or required next action. "
        "Use the supplied current-state projection as an index and query durable artifacts "
        "or memory only when it is insufficient. Relevant skills are reusable task-domain "
        "guidance; an exact-task playbook, when present, is only a prior. Neither is an "
        "executable macro or host-authored phase machine. Live tool contracts exclusively "
        "define request fields, structured outputs, opaque references, receipts, bundles, "
        "and repair payloads."
    )


def _validate_official_reward_completion(
    decision: PlannerDecision,
    *,
    tool_context: JsonDict,
) -> list[str]:
    if decision.action_type.lower().strip() != "response" or decision.action != "task_complete":
        return []
    memory = tool_context.get("memory")
    metadata = memory.get("metadata") if isinstance(memory, dict) else None
    if not isinstance(metadata, dict) or metadata.get("source") != "ParallelEpisodeHarness":
        return []
    if metadata.get("require_official_reward") is False:
        return []
    receipt = tool_context.get("latest_environment_receipt")
    info = receipt.get("info") if isinstance(receipt, dict) else None
    try:
        reward = float(receipt.get("reward")) if isinstance(receipt, dict) else 0.0
    except (TypeError, ValueError):
        reward = 0.0
    if (
        reward > 0
        and isinstance(info, dict)
        and info.get("environment_receipt_trusted") is True
        and info.get("official_reward") is True
    ):
        return []
    return [
        "LIBERO batch completion requires an official positive reward from the same "
        "episode. Continue with settle/retreat/observe instead of declaring task_complete."
    ]


def _validate_calibration_permission(
    decision: PlannerDecision,
    *,
    tool_context: JsonDict,
) -> list[str]:
    """Keep calibration lifecycle writes inside an explicitly selected scope."""

    if decision.action_type != "tool_call" or decision.action not in {
        "propose_calibration_profile",
        "promote_calibration_profile",
    }:
        return []
    selected = tool_context.get("selected_skill_guidance")
    selected_names = {
        str(item.get("name") or "")
        for item in selected
        if isinstance(item, dict)
    } if isinstance(selected, list) else set()
    if "embodiment_explore" in selected_names:
        return []
    return [
        f"{decision.action} is a calibration lifecycle write and requires an "
        "explicit embodiment_explore session; ordinary task execution may inspect "
        "calibration evidence but may not modify the profile."
    ]


def _canonicalize_host_parameters(
    decision: PlannerDecision,
    *,
    tool_context: JsonDict,
) -> list[JsonDict]:
    """Canonicalize host-owned image paths while preserving semantic choices."""

    if decision.action_type.lower().strip() != "tool_call":
        return []
    if decision.action == "grasp_pose_estimate":
        parameters = dict(decision.parameters)
        if str(parameters.get("mode") or "targeted").strip().lower() == "scene":
            return []
        selected = tool_context.get("selected_sam3_detection")
        object_mask = parameters.get("object_mask")
        if not isinstance(selected, dict) or not isinstance(object_mask, dict):
            return []
        selected_mask = str(selected.get("mask_ref") or "")
        supplied_mask = str(object_mask.get("mask_ref") or "")
        source_observation = selected.get("source_observation")
        selected_epoch = selected.get("scene_epoch")
        current_epoch = tool_context.get("scene_epoch")
        if (
            selected_mask
            and supplied_mask == selected_mask
            and isinstance(source_observation, dict)
            and selected_epoch == current_epoch
        ):
            source_rgb = source_observation.get("rgb")
            source_depth = source_observation.get("depth")
            source_intrinsics = source_observation.get("intrinsics")
            source_frame_id = source_observation.get("frame_id")
            if (
                isinstance(source_rgb, str)
                and source_rgb
                and isinstance(source_depth, str)
                and source_depth
                and isinstance(source_intrinsics, dict)
                and source_intrinsics
                and isinstance(source_frame_id, str)
                and source_frame_id
            ):
                canonicalizations: list[JsonDict] = []

                def bind(field: str, canonical: object) -> None:
                    supplied = parameters.get(field)
                    if supplied == canonical:
                        return
                    parameters[field] = canonical
                    canonicalizations.append(
                        {
                            "field": field,
                            "tool": "grasp_pose_estimate",
                            "reason": "bind_selected_mask_to_source_observation_packet",
                            "supplied": supplied,
                            "canonical": canonical,
                        }
                    )

                bind("rgb", source_rgb)
                hints = parameters.get("hints")
                enhanced_depth = (
                    hints.get("depth_enhancement") if isinstance(hints, dict) else None
                )
                if not isinstance(enhanced_depth, dict):
                    bind("depth", source_depth)
                    bind("intrinsics", dict(source_intrinsics))
                bind("camera_frame_id", source_frame_id)
                canonical_mask = dict(object_mask)
                supplied_source = canonical_mask.get("source_image")
                if supplied_source != source_rgb:
                    canonical_mask["source_image"] = source_rgb
                    canonicalizations.append(
                        {
                            "field": "object_mask.source_image",
                            "tool": "grasp_pose_estimate",
                            "reason": "bind_selected_mask_to_source_observation_packet",
                            "supplied": supplied_source,
                            "canonical": source_rgb,
                        }
                    )
                parameters["object_mask"] = canonical_mask
                decision.parameters = parameters
                return canonicalizations
        rgb = parameters.get("rgb")
        depth = parameters.get("depth")
        source_image = object_mask.get("source_image")
        if (
            not selected_mask
            or supplied_mask != selected_mask
            or not isinstance(rgb, str)
            or not isinstance(depth, str)
            or not isinstance(source_image, str)
            or _same_resolved_local_path(rgb, source_image)
            or not _same_local_artifact(rgb, source_image)
            or not _aligned_rgb_depth_packet(rgb, depth, parameters=parameters)
        ):
            return []
        canonical_mask = dict(object_mask)
        canonical_mask["source_image"] = rgb
        parameters["object_mask"] = canonical_mask
        decision.parameters = parameters
        return [
            {
                "field": "object_mask.source_image",
                "tool": "grasp_pose_estimate",
                "reason": "bind_byte_identical_mask_source_to_selected_rgbd_packet",
                "supplied": source_image,
                "canonical": rgb,
            }
        ]
    if decision.action == "retrieve_asset_reference":
        supplied_image = decision.parameters.get("scene_image")
        if supplied_image is None:
            return []
        parameters = dict(decision.parameters)
        parameters.pop("scene_image", None)
        decision.parameters = parameters
        return [
            {
                "field": "scene_image",
                "tool": "retrieve_asset_reference",
                "reason": "strip_agent_visual_transport_path",
                "supplied": supplied_image,
                "canonical": "host_resolved_from_source_packet_id",
            }
        ]
    return []


def _validate_perception_artifact_provenance(
    decision: PlannerDecision,
    *,
    tool_context: JsonDict,
) -> list[str]:
    """Keep mask/RGB-D provenance strict without prescribing a task phase."""

    if (
        decision.action_type.lower().strip() != "tool_call"
        or decision.action != "grasp_pose_estimate"
        or str(decision.parameters.get("mode") or "targeted").strip().lower() == "scene"
    ):
        return []
    parameters = decision.parameters
    object_mask = parameters.get("object_mask")
    if not isinstance(object_mask, dict):
        return []
    rgb = parameters.get("rgb")
    depth = parameters.get("depth")
    source_image = object_mask.get("source_image")
    errors: list[str] = []
    if not _same_resolved_local_path(rgb, source_image):
        errors.append(
            "grasp_pose_estimate object_mask.source_image must be the exact RGB "
            "artifact passed as rgb. A mask cannot be combined with a later render "
            "unless that render is byte-identical and host-canonicalized."
        )
    if not _aligned_rgb_depth_packet(rgb, depth, parameters=parameters):
        errors.append(
            "grasp_pose_estimate rgb and depth must come from the same observation "
            "packet; copy both paths from one recent transition."
        )
    selected = tool_context.get("selected_sam3_detection")
    if isinstance(selected, dict):
        selected_epoch = selected.get("scene_epoch")
        current_epoch = tool_context.get("scene_epoch")
        if (
            isinstance(selected_epoch, int)
            and not isinstance(selected_epoch, bool)
            and isinstance(current_epoch, int)
            and not isinstance(current_epoch, bool)
            and selected_epoch != current_epoch
        ):
            errors.append(
                "grasp_pose_estimate cannot reuse a selected mask from a stale "
                "scene epoch; segment the current scene first."
            )
        expected_mask = str(selected.get("mask_ref") or "")
        supplied_mask = str(object_mask.get("mask_ref") or "")
        if expected_mask and supplied_mask != expected_mask:
            errors.append(
                "grasp_pose_estimate object_mask.mask_ref must preserve the currently "
                "selected SAM3 detection."
            )
    return errors


def _validate_compiled_grasp_target_freshness(
    decision: PlannerDecision,
    *,
    tool_context: JsonDict,
) -> list[str]:
    """Reject only stale contact/close actions, while keeping safe waypoints usable."""

    if decision.action_type.lower().strip() != "tool_call":
        return []
    graph = tool_context.get("provenance_evidence_graph")
    graph = graph if isinstance(graph, dict) else {}
    inconsistencies = graph.get("inconsistencies")
    mismatch = next(
        (
            item
            for item in inconsistencies or []
            if isinstance(item, dict)
            and item.get("code") == "compiled_grasp_target_superseded"
        ),
        None,
    )
    if not isinstance(mismatch, dict):
        return []

    unsafe = False
    if decision.action == "gripper_control":
        try:
            unsafe = int(decision.parameters.get("position")) == 0
        except (TypeError, ValueError):
            unsafe = False
    elif decision.action in {"move_to", "follow_eef_trajectory"}:
        compiled_id = str(mismatch.get("compiled_grasp_id") or "")
        node = next(
            (
                item
                for item in graph.get("nodes", []) or []
                if isinstance(item, dict)
                and item.get("kind") == "compiled_targeted_grasp"
                and str(item.get("compiled_grasp_id") or "") == compiled_id
            ),
            {},
        )
        contact_pose = node.get("contact_pose") if isinstance(node, dict) else None
        contact_xyz = contact_pose.get("xyz") if isinstance(contact_pose, dict) else None
        if decision.action == "move_to":
            requested_poses = [decision.parameters.get("target_pose")]
        else:
            trajectory = decision.parameters.get("trajectory")
            requested_poses = trajectory if isinstance(trajectory, list) else []
        unsafe = any(
            _pose_xyz_within(pose, contact_xyz, tolerance_m=0.08)
            for pose in requested_poses
        )
    if not unsafe:
        return []

    bundle = tool_context.get("grasp_input_bundle")
    bundle_id = bundle.get("bundle_id") if isinstance(bundle, dict) else None
    recovery = (
        f" Call grasp_pose_estimate with bundle_id={bundle_id!r}, choose a current "
        "candidate, and compile it before contact or gripper close."
        if isinstance(bundle_id, str) and bundle_id
        else " Re-segment the current target, estimate a new grasp, and compile it before contact."
    )
    return [
        "compiled_grasp_target_superseded: compiled grasp "
        f"{mismatch.get('compiled_grasp_id')!r} is bound to target evidence "
        f"{mismatch.get('compiled_target_evidence_id')!r}, but the current selected "
        f"target is {mismatch.get('current_target_evidence_id')!r}. The requested "
        "contact/close action cannot use the old pose. Safe retreat and clearance "
        "waypoints remain allowed." + recovery
    ]


def _pose_xyz_within(
    pose: object,
    reference_xyz: object,
    *,
    tolerance_m: float,
) -> bool:
    if not isinstance(pose, dict):
        return False
    xyz = pose.get("xyz")
    if not (
        isinstance(xyz, list | tuple)
        and isinstance(reference_xyz, list | tuple)
        and len(xyz) == 3
        and len(reference_xyz) == 3
    ):
        return False
    try:
        distance = math.sqrt(
            sum(
                (float(xyz[index]) - float(reference_xyz[index])) ** 2
                for index in range(3)
            )
        )
    except (TypeError, ValueError):
        return False
    return math.isfinite(distance) and distance <= tolerance_m


def _same_resolved_local_path(left: object, right: object) -> bool:
    if not isinstance(left, str) or not isinstance(right, str) or not left or not right:
        return False
    try:
        return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()
    except (OSError, ValueError):
        return False


def _path_packet_id(path_value: object) -> str:
    if not isinstance(path_value, str) or not path_value:
        return ""
    try:
        return Path(path_value).parent.name
    except (OSError, ValueError):
        return ""


def _aligned_rgb_depth_packet(
    rgb: object,
    depth: object,
    *,
    parameters: JsonDict,
) -> bool:
    if not isinstance(rgb, str) or not isinstance(depth, str) or not rgb or not depth:
        return False
    hints = parameters.get("hints")
    if isinstance(hints, dict) and isinstance(hints.get("depth_enhancement"), dict):
        return True
    rgb_packet = _path_packet_id(rgb)
    depth_packet = _path_packet_id(depth)
    return bool(rgb_packet and depth_packet and rgb_packet == depth_packet)


def _same_local_artifact(left: object, right: object) -> bool:
    if not isinstance(left, str) or not isinstance(right, str) or not left or not right:
        return False
    if left == right:
        return True
    try:
        left_path = Path(left)
        right_path = Path(right)
        if not left_path.is_file() or not right_path.is_file():
            return False
        if left_path.stat().st_size != right_path.stat().st_size:
            return False
        return sha256(left_path.read_bytes()).digest() == sha256(right_path.read_bytes()).digest()
    except OSError:
        return False


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


def _first_observation_packet_id(observation: EnvObservation) -> str:
    artifacts = observation.metadata.get("image_artifacts")
    if not isinstance(artifacts, list):
        return ""
    for artifact in artifacts:
        if not isinstance(artifact, dict) or artifact.get("kind") != "rgb":
            continue
        packet_id = artifact.get("packet_id")
        if isinstance(packet_id, str) and packet_id:
            return packet_id
    return ""


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
    context = {
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

    context, _conversation_messages = _build_budgeted_tool_context(
        observation=observation,
        memory=memory,
        tools=tools,
        skills=skills,
        config=config,
        system_prompt="",
    )
    return context


def _build_budgeted_tool_context(
    *,
    observation: EnvObservation,
    memory: AgentMemory,
    tools: ToolRegistry,
    skills: SkillRegistry,
    config: PlannerContextConfig | None,
    system_prompt: str,
) -> tuple[JsonDict, list[JsonDict]]:
    """Build and jointly project runtime evidence plus canonical chat history."""

    context_config = config or PlannerContextConfig()
    context = _build_tool_context_payload(
        observation=observation,
        memory=memory,
        tools=tools,
        skills=skills,
        config=context_config,
    )
    conversation_messages = memory.model_conversation_messages(
        max_action_groups=max(0, context_config.recent_conversation_action_groups)
    )
    conversation_messages, budget = _project_planner_input_to_budget(
        context,
        config=context_config,
        conversation_messages=conversation_messages,
        system_prompt=system_prompt,
    )
    context["context_budget"] = budget
    agent_context = context.get("agent_context")
    if isinstance(agent_context, dict):
        agent_context["context_budget"] = budget
    return context, conversation_messages


def _build_tool_context_payload(
    *,
    observation: EnvObservation,
    memory: AgentMemory,
    tools: ToolRegistry,
    skills: SkillRegistry,
    config: PlannerContextConfig,
) -> JsonDict:
    executable_tools = [tool for tool in tools.list() if tools.can_execute(tool.name)]
    tool_references, tool_contract_projection_audit = (
        _contract_driven_tool_references(executable_tools)
    )
    executable_tool_names = {tool.name for tool in executable_tools}
    selected_skill_guidance = _selected_skill_guidance(
        skills.list(),
        observation=observation,
        memory=memory,
        config=config,
    )
    for skill_guidance in selected_skill_guidance:
        _annotate_skill_tool_availability(
            skill_guidance,
            executable_tool_names=executable_tool_names,
        )
    skill_usage = _skill_usage_guidance(selected_skill_guidance, memory)
    memory_context = memory.planning_context(max_events=config.max_memory_events)
    pending_ik_execution_index = _pending_ik_execution_index(memory)
    tool_loop_warning = _motion_failure_attractor_warning(
        memory
    ) or _conversation_no_progress_warning(memory)
    effective_task = _effective_task_text(observation, memory)
    task_playbook = _matched_task_playbook(
        observation=observation,
        memory=memory,
        task=effective_task,
    )
    camera_artifacts = _current_camera_artifacts(observation, memory=memory)
    visual_history: JsonDict | None = None
    if config.visual_history.enabled:
        visual_projection = build_visual_history_projection(
            observation=observation,
            memory=memory,
            config=config.visual_history,
            current_camera_artifacts=camera_artifacts,
        )
        vision_image_paths = list(visual_projection["vision_image_paths"])
        vision_evidence = list(visual_projection["vision_evidence"])
        visual_history = dict(visual_projection["visual_history"])
    else:
        vision_image_paths = [
            artifact["path"]
            for artifact in camera_artifacts
            if artifact["kind"] == "rgb" and _is_primary_planner_camera(artifact)
        ][:2]
        vision_evidence = _current_vision_evidence(
            observation,
            image_paths=vision_image_paths,
            camera_artifacts=camera_artifacts,
        )
    context: JsonDict = {
        "schema_version": "openeta.planner_context.v1",
        "task": effective_task,
        "task_authority": memory.metadata.get("task_authority") or "environment_task",
        "active_environment_task": memory_context.get("active_environment_task"),
        "task_playbook": task_playbook,
        "observation": _observation_summary(
            observation,
            gripper_command_state=memory_context.get("gripper_command_state"),
        ),
        "vision_image_paths": vision_image_paths,
        "vision_evidence": vision_evidence,
        "visual_history": visual_history,
        "current_camera_artifacts": camera_artifacts,
        "current_camera_calibrations": _current_camera_calibrations(observation),
        "memory": memory_context,
        "pending_target_selection": memory_context.get("pending_target_selection"),
        "selected_sam3_detection": memory_context.get("selected_sam3_detection"),
        "selected_sam3_detections": memory_context.get("selected_sam3_detections"),
        "sam3_no_detection": memory_context.get("sam3_no_detection"),
        "sam3_no_detections": memory_context.get("sam3_no_detections"),
        "pending_reference_localization": memory_context.get(
            "pending_reference_localization"
        ),
        "target_identity_anchor": memory_context.get("target_identity_anchor"),
        "retained_targeted_grasp": memory_context.get("retained_targeted_grasp"),
        "provenance_evidence_graph": memory_context.get(
            "provenance_evidence_graph"
        ),
        "grasp_adjustment_budget": memory_context.get("grasp_adjustment_budget"),
        "grasp_input_bundle": memory_context.get("grasp_input_bundle"),
        "wrist_alignment_bundle": memory_context.get("wrist_alignment_bundle"),
        "anyplace_input_bundle": memory_context.get("anyplace_input_bundle"),
        "articulated_attachment_probe": memory_context.get(
            "articulated_attachment_probe"
        ),
        "gripper_command_state": memory_context.get("gripper_command_state"),
        "attachment_evidence": memory_context.get("attachment_evidence"),
        "motion_reconciliation": memory_context.get("motion_reconciliation"),
        "ik_preview_receipts": memory_context.get("ik_preview_receipts"),
        "pending_ik_execution_index": pending_ik_execution_index,
        "fresh_observation_obligation": {
            "schema_version": "openeta.fresh_observation_obligation.v1",
            "required": True,
            "attempt": int(observation.metadata.get("fresh_observation_attempts") or 0) + 1,
        }
        if observation.metadata.get("fresh_observation_required") is True
        else None,
        "scene_epoch": memory_context.get("scene_epoch"),
        "object_scene_epoch": memory_context.get("object_scene_epoch"),
        "robot_motion_epoch": memory_context.get("robot_motion_epoch"),
        "transition_ledger": memory_context.get("transition_ledger"),
        "latest_compiled_clearance_execution": memory_context.get(
            "latest_compiled_clearance_execution"
        ),
        "latest_compiled_contact_execution": memory_context.get(
            "latest_compiled_contact_execution"
        ),
        "latest_environment_receipt": memory_context.get("latest_environment_receipt"),
        "tool_references": tool_references,
        # Host-only migration evidence. The Agent receives the contract-driven
        # references above; this audit remains outside agent_context and records
        # where legacy ToolSpec parameter names intentionally differ.
        "tool_contract_projection_audit": tool_contract_projection_audit,
        "registered_tool_handlers": tools.handler_names(),
        "skill_references": [_selected_skill_reference(skill) for skill in selected_skill_guidance],
        "available_skill_count": len(skills.list()),
        "selected_skill_guidance": selected_skill_guidance,
        "skill_usage": skill_usage,
        "execution_rules": _tool_calling_rules(),
        "tool_loop_warning": tool_loop_warning,
    }
    context["agent_context"] = _build_agent_decision_context(context, config=config)
    return context


def _build_agent_decision_context(
    runtime_context: JsonDict,
    *,
    config: PlannerContextConfig,
) -> JsonDict:
    """Project runtime evidence into the smaller context owned by the Agent.

    Host task phases and required-next-action obligations deliberately stay out
    of this projection. Runtime keeps them temporarily for compatibility and
    safety checks while the main VLM receives observations, evidence, and open
    questions from which it can choose its own next action.
    """

    memory = runtime_context.get("memory")
    memory = memory if isinstance(memory, dict) else {}
    working = memory.get("working_memory")
    working = working if isinstance(working, dict) else {}
    working_facts = working.get("facts")
    working_facts = working_facts if isinstance(working_facts, dict) else {}

    explicit_agent_state = memory.get("agent_working_state")
    if isinstance(explicit_agent_state, dict):
        agent_facts = dict(explicit_agent_state)
    else:
        agent_facts = {
            str(key): dict(raw_entry)
            for key, raw_entry in working_facts.items()
            if isinstance(raw_entry, dict) and raw_entry.get("source") == "save_memory"
        }

    explicit_world_evidence = memory.get("world_evidence")
    runtime_evidence = (
        _project_world_evidence(dict(explicit_world_evidence))
        if isinstance(explicit_world_evidence, dict)
        else {}
    )

    open_questions: JsonDict = {}
    question_fields = {
        "target_selection": "pending_target_selection",
        "reference_localization": "pending_reference_localization",
        "perception_failure": "sam3_no_detection",
    }
    for output_key, context_key in question_fields.items():
        value = memory.get(context_key)
        if value is not None:
            open_questions[output_key] = (
                _project_perception_failure(
                    value,
                    retained_grasp=memory.get("retained_targeted_grasp"),
                )
                if output_key == "perception_failure" and isinstance(value, dict)
                else value
            )

    recent_events = memory.get("recent_events")
    recent_events = recent_events if isinstance(recent_events, list) else []
    recent_transitions = _recent_high_fidelity_transitions(
        recent_events,
        observation_turns=max(0, config.recent_transition_observations),
    )

    observation = runtime_context.get("observation")
    observation = observation if isinstance(observation, dict) else {}
    visual_evidence = runtime_context.get("vision_evidence")
    visual_evidence = visual_evidence if isinstance(visual_evidence, list) else []
    current_visual_evidence = [
        item
        for item in visual_evidence
        if isinstance(item, dict) and item.get("role") == "current_scene"
    ]
    artifacts = working.get("artifacts")
    artifacts = _bounded_artifact_index(
        artifacts if isinstance(artifacts, dict) else {}
    )

    active_bundles = {
        name: bundle
        for name, bundle in {
            "grasp_pose_estimate": memory.get("grasp_input_bundle"),
            "anyplace": memory.get("anyplace_input_bundle"),
        }.items()
        if isinstance(bundle, dict)
    }
    unresolved_obligations: JsonDict = dict(open_questions)
    if runtime_context.get("fresh_observation_obligation") is not None:
        unresolved_obligations["fresh_observation"] = runtime_context.get(
            "fresh_observation_obligation"
        )
    motion_reconciliation = runtime_evidence.get("motion_reconciliation")
    if isinstance(motion_reconciliation, dict) and motion_reconciliation.get(
        "status"
    ) in {"required", "unresolved"}:
        unresolved_obligations["motion_reconciliation"] = motion_reconciliation
    evidence_graph = memory.get("provenance_evidence_graph")
    evidence_graph = evidence_graph if isinstance(evidence_graph, dict) else {}
    provenance_inconsistencies = evidence_graph.get("inconsistencies")
    if isinstance(provenance_inconsistencies, list) and provenance_inconsistencies:
        unresolved_obligations["provenance_integrity"] = provenance_inconsistencies
    tool_loop_warning = runtime_context.get("tool_loop_warning")
    if isinstance(tool_loop_warning, dict):
        unresolved_obligations["no_progress_tool_loop"] = tool_loop_warning
    pending_ik_execution_index = runtime_context.get("pending_ik_execution_index")
    if isinstance(pending_ik_execution_index, dict) and pending_ik_execution_index.get(
        "receipts"
    ):
        unresolved_obligations["preview_execution_gap"] = {
            "pending_receipt_ids": [
                item.get("receipt_id")
                for item in pending_ik_execution_index["receipts"]
                if isinstance(item, dict) and item.get("receipt_id")
            ],
            "resolution": (
                "Execute an exact receipt reference, intentionally supersede it with "
                "materially changed evidence/geometry, or explain why it is no longer "
                "useful before repeating same-state perception."
            ),
        }
    camera_artifacts = runtime_context.get("current_camera_artifacts")
    camera_artifacts = camera_artifacts if isinstance(camera_artifacts, list) else []
    packet_ids = list(
        dict.fromkeys(
            str(item.get("packet_id"))
            for item in camera_artifacts
            if isinstance(item, dict) and item.get("packet_id")
        )
    )
    tool_references = runtime_context.get("tool_references", [])
    decision_state = {
        "schema_version": "openeta.decision_state.v1",
        "current_observation_packet": {
            "packet_ids": packet_ids,
            "step_idx": observation.get("metadata", {}).get("step_idx")
            if isinstance(observation.get("metadata"), dict)
            else None,
            "camera_artifacts": camera_artifacts,
            "camera_calibrations": runtime_context.get(
                "current_camera_calibrations", []
            ),
        },
        "active_bundles": active_bundles,
        "grasp_adjustment_budget": memory.get("grasp_adjustment_budget"),
        # Execution receipt only: an Agent-chosen clearance is optional, but an
        # explicit miss cannot be silently treated as a successful contact premise.
        "latest_compiled_clearance_execution": memory.get(
            "latest_compiled_clearance_execution"
        ),
        # This is an execution receipt, not a host-owned task phase.  Keeping it
        # in the compact decision index lets the Agent reason from the actual
        # contact outcome even before a dependent close is attempted.
        "latest_compiled_contact_execution": memory.get(
            "latest_compiled_contact_execution"
        ),
        "pending_execution_receipts": pending_ik_execution_index,
        "unresolved_obligations": unresolved_obligations,
        "last_action_effect": _latest_action_effect(recent_events),
        "no_progress_tool_loop": tool_loop_warning,
        "tool_health": memory.get("tool_health", {}),
        "available_tools": [
            reference.get("name")
            for reference in tool_references
            if isinstance(reference, dict) and reference.get("name")
        ],
    }

    return {
        "schema_version": "openeta.agent_context.v2",
        "objective": {
            "task": runtime_context.get("task"),
            "task_authority": runtime_context.get("task_authority"),
            "active_environment_task": runtime_context.get("active_environment_task"),
            "latest_human_interaction": memory.get("latest_human_interaction"),
        },
        "current_observation": {
            "summary": observation,
            "visual_evidence": current_visual_evidence,
            "status": (
                "available" if current_visual_evidence else "visual_evidence_not_supplied"
            ),
        },
        "visual_history": runtime_context.get("visual_history"),
        "recent_transitions": recent_transitions,
        "transition_ledger": _project_transition_ledger(
            memory.get("transition_ledger", [])
        ),
        "world_evidence": runtime_evidence,
        "evidence_graph": evidence_graph,
        "host_resolved_inputs": {
            "grasp_pose_estimate": memory.get("grasp_input_bundle"),
            "wrist_alignment": memory.get("wrist_alignment_bundle"),
            "anyplace": memory.get("anyplace_input_bundle"),
        },
        "decision_state": decision_state,
        "open_questions": open_questions,
        "agent_working_state": {
            "facts": agent_facts,
            "skill_notes": working.get("skill_notes", {}),
            "compact_summary": working.get("compact_summary", ""),
        },
        "artifacts": artifacts,
        "task_playbook": runtime_context.get("task_playbook"),
        "relevant_skills": runtime_context.get("selected_skill_guidance", []),
        "skill_usage": runtime_context.get("skill_usage", {}),
        # Full schemas remain in the canonical context once. The provider backend
        # moves this cache-stable block ahead of growing conversation history; the
        # legacy tool_references field stays a name-only compatibility index.
        "available_tools_schema_version": "openeta.agent_tool_contract.v2",
        "available_tools": runtime_context.get("tool_references", []),
        "tool_references": [
            {"name": reference.get("name")}
            for reference in runtime_context.get("tool_references", [])
            if isinstance(reference, dict) and reference.get("name")
        ],
        "operational_constraints": {
            "fresh_observation_required": (
                runtime_context.get("fresh_observation_obligation") is not None
            ),
            "motion_reconciliation": runtime_evidence.get("motion_reconciliation"),
            "rules": runtime_context.get("execution_rules", {}),
        },
        "vision_image_paths": runtime_context.get("vision_image_paths", []),
        "vision_evidence": visual_evidence,
    }


def _pending_ik_execution_index(memory: AgentMemory) -> JsonDict | None:
    """Project current executable IK receipts that have not been dispatched.

    The index is deliberately capability-oriented: it does not choose a target,
    force motion, or advance a host-owned stage.  It merely makes the causal gap
    between a successful read-only preview and physical execution hard to miss.
    """

    state = memory.ik_preview_receipts() or {}
    receipts = [
        item for item in state.get("receipts", []) if isinstance(item, dict)
    ]
    if not receipts:
        return None
    current_object_epoch = memory.object_scene_epoch()
    current_robot_epoch = memory.robot_motion_epoch()
    latest_disposition: dict[str, str] = {}
    for event in reversed(memory.events[-240:]):
        if event.event_type == "ik_preview_receipt":
            if (
                _planner_epoch(event.payload.get("object_scene_epoch"))
                == current_object_epoch
                and _planner_epoch(event.payload.get("robot_motion_epoch"))
                == current_robot_epoch
            ):
                receipt_id = str(event.payload.get("receipt_id") or "")
                if receipt_id:
                    latest_disposition.setdefault(receipt_id, "previewed")
            continue
        if event.event_type != "action":
            continue
        anchor = event.payload.get("input_state_anchor")
        anchor = anchor if isinstance(anchor, dict) else {}
        if not anchor or (
            _planner_epoch(anchor.get("object_scene_epoch")) != current_object_epoch
            or _planner_epoch(anchor.get("robot_motion_epoch")) != current_robot_epoch
        ):
            continue
        command = event.payload.get("command")
        command = command if isinstance(command, dict) else {}
        request = command.get("request")
        request = request if isinstance(request, dict) else {}
        name = str(request.get("name") or "")
        parameters = request.get("parameters")
        parameters = parameters if isinstance(parameters, dict) else {}
        if name == "move_to":
            receipt_id = str(parameters.get("ik_receipt_id") or "")
            if receipt_id:
                latest_disposition.setdefault(receipt_id, "attempted")
        elif name == "follow_eef_trajectory":
            receipt_ids = parameters.get("ik_receipt_ids")
            if isinstance(receipt_ids, list):
                for value in receipt_ids:
                    receipt_id = str(value or "")
                    if receipt_id:
                        latest_disposition.setdefault(receipt_id, "attempted")

    projected: list[JsonDict] = []
    seen: set[str] = set()
    for receipt in reversed(receipts):
        receipt_id = str(receipt.get("receipt_id") or "")
        if (
            not receipt_id
            or receipt_id in seen
            or latest_disposition.get(receipt_id) == "attempted"
        ):
            continue
        seen.add(receipt_id)
        if (
            _planner_epoch(receipt.get("object_scene_epoch")) != current_object_epoch
            or _planner_epoch(receipt.get("robot_motion_epoch")) != current_robot_epoch
        ):
            continue
        classification = str(receipt.get("classification") or "")
        delegation = receipt.get("motion_collision_delegation")
        delegated = bool(
            classification == "kinematically_feasible_collision_deferred"
            and isinstance(delegation, dict)
            and delegation.get("available_for_matching_move") is True
        )
        if classification != "feasible" and not delegated:
            continue
        execution_parameters: JsonDict = {"ik_receipt_id": receipt_id}
        if delegated:
            execution_parameters["enable_collision_check"] = True
        item: JsonDict = {
            "receipt_id": receipt_id,
            "status": "previewed_not_executed",
            "classification": classification,
            "orientation_policy": receipt.get("orientation_policy"),
            "target_signature": receipt.get("target_signature"),
            "execution_reference": {
                "tool": "move_to",
                "parameters": execution_parameters,
            },
        }
        request_reference = receipt.get("request_reference")
        if isinstance(request_reference, dict) and request_reference:
            item["request_reference"] = dict(request_reference)
        projected.append(item)
        if len(projected) >= 3:
            break
    if not projected:
        return None
    return {
        "schema_version": "openeta.decision_state.pending_execution_index.v1",
        "receipts": projected,
        "interpretation": (
            "Each entry is an executable IK preview that has not yet been dispatched. "
            "The Agent may execute the exact short reference, supersede it with "
            "materially changed evidence/geometry, or decline it with an explicit reason."
        ),
        "host_policy": "capability_index_only; no next action is forced or gated",
    }


def _planner_epoch(value: object) -> int:
    if isinstance(value, bool):
        return -1
    try:
        return int(value)
    except (TypeError, ValueError):
        return -1


def _conversation_no_progress_warning(memory: AgentMemory) -> JsonDict | None:
    """Detect repeated semantically equivalent tool requests without host progress.

    Observation packet ids are provenance handles, not task progress. Read-only
    turns can mint a new handle for unchanged geometry, so the signature omits
    packet ids while retaining the actual tool, prompt, camera, bundle, and
    candidate identifiers. This is Agent-visible reflection evidence, not a
    required-next-action state machine or an execution gate.
    """

    actions: list[tuple[str, str, JsonDict]] = []
    for item in reversed(memory.conversation.items):
        if item.role != "assistant" or item.kind != "action":
            continue
        request = item.data.get("request")
        request = request if isinstance(request, dict) else {}
        if str(request.get("kind") or "") != "tool_call":
            break
        name = str(request.get("name") or "")
        parameters = request.get("parameters")
        parameters = parameters if isinstance(parameters, dict) else {}
        stable_parameters = _without_packet_provenance(parameters)
        signature = sha256(
            json.dumps(
                {"name": name, "parameters": stable_parameters},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:20]
        actions.append((name, signature, parameters))
        if len(actions) >= 8:
            break
    if not actions:
        return None
    latest_name, latest_signature, _ = actions[0]
    repeated = 0
    packet_ids: list[str] = []
    for name, signature, parameters in actions:
        if name != latest_name or signature != latest_signature:
            break
        repeated += 1
        packet_id = parameters.get("source_packet_id")
        if isinstance(packet_id, str) and packet_id:
            packet_ids.append(packet_id)
    if repeated >= 2:
        repeated_anchors = _recent_equivalent_request_anchors(
            memory,
            tool_name=latest_name,
            semantic_signature=latest_signature,
            limit=repeated,
        )
        if (
            len(repeated_anchors) >= 2
            and all(anchor.get("visual_signature") for anchor in repeated_anchors)
            and not all(
                _same_semantic_state_anchor(anchor, repeated_anchors[0])
                for anchor in repeated_anchors[1:]
            )
        ):
            return None
        return {
            "schema_version": "openeta.no_progress_tool_loop.v1",
            "tool": latest_name,
            "equivalent_call_count": repeated,
            "semantic_signature": latest_signature,
            "packet_ids_changed": len(set(packet_ids)) > 1,
            "interpretation": (
                "The same semantic tool request was repeated without an intervening "
                "different action. Packet-id-only refresh is provenance churn, not new "
                "geometry or task progress. Inspect and consume the latest result, choose "
                "a materially different input/action, or explain the observed change."
            ),
            "host_policy": "reflection_warning_only; no tool is forced or blocked",
        }
    return _interleaved_no_progress_warning(memory)


def _recent_equivalent_request_anchors(
    memory: AgentMemory,
    *,
    tool_name: str,
    semantic_signature: str,
    limit: int,
) -> list[JsonDict]:
    anchors: list[JsonDict] = []
    for event in reversed(memory.events[-160:]):
        if event.event_type != "action":
            continue
        command = event.payload.get("command")
        command = command if isinstance(command, dict) else {}
        request = command.get("request")
        request = request if isinstance(request, dict) else {}
        name = str(request.get("name") or "")
        parameters = request.get("parameters")
        parameters = parameters if isinstance(parameters, dict) else {}
        signature = sha256(
            json.dumps(
                {"name": name, "parameters": _without_packet_provenance(parameters)},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:20]
        if name != tool_name or signature != semantic_signature:
            break
        anchor = event.payload.get("input_state_anchor")
        anchors.append(dict(anchor) if isinstance(anchor, dict) else {})
        if len(anchors) >= limit:
            break
    return anchors


def _interleaved_no_progress_warning(memory: AgentMemory) -> JsonDict | None:
    """Detect equivalent calls hidden inside a short read-only/planning cycle.

    A common failure mode is ``sam3 -> select -> sam3 -> select``.  Adjacent-call
    detection misses it even though no world mutation occurred.  This detector
    deliberately remains advisory: it exposes a ready bundle and asks the Agent
    to consume it, but never blocks a legitimate refresh after visible change.
    """

    semantic_state_warning = _semantic_state_cycle_warning(memory)
    if semantic_state_warning is not None:
        return semantic_state_warning

    actions: list[tuple[str, str, JsonDict]] = []
    world_mutating = {"move_to", "follow_eef_trajectory", "gripper_control"}
    for event in reversed(memory.events[-160:]):
        if event.event_type == "observed_object_scene_change":
            break
        if event.event_type != "action":
            continue
        command = event.payload.get("command")
        command = command if isinstance(command, dict) else {}
        request = command.get("request")
        request = request if isinstance(request, dict) else {}
        if str(request.get("kind") or "") != "tool_call":
            continue
        name = str(request.get("name") or "")
        if name in world_mutating:
            break
        parameters = request.get("parameters")
        parameters = parameters if isinstance(parameters, dict) else {}
        stable_parameters = _semantic_repeat_parameters(name, parameters)
        signature = sha256(
            json.dumps(
                {"name": name, "parameters": stable_parameters},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:20]
        actions.append((name, signature, parameters))
        if len(actions) >= 12:
            break
    if len(actions) < 2:
        return None
    latest_name, latest_signature, _ = actions[0]
    matches = [
        (name, parameters)
        for name, signature, parameters in actions
        if name == latest_name and signature == latest_signature
    ]
    if len(matches) < 2:
        return None
    intervening_tools = [name for name, _, _ in actions[1:] if name != latest_name]
    reusable_bundles: JsonDict = {}
    for key, bundle in {
        "grasp_pose_estimate": memory.grasp_input_bundle(),
        "anyplace": memory.anyplace_input_bundle(),
    }.items():
        if isinstance(bundle, dict) and bundle.get("status") == "ready":
            reusable_bundles[key] = {
                "bundle_id": bundle.get("bundle_id"),
                "status": bundle.get("status"),
                "target_evidence_id": bundle.get("target_evidence_id"),
            }
    packet_ids = [
        str(parameters.get("source_packet_id"))
        for _, parameters in matches
        if parameters.get("source_packet_id")
    ]
    return {
        "schema_version": "openeta.no_progress_tool_loop.v1",
        "tool": latest_name,
        "trigger_type": "interleaved_equivalent_read_only_cycle",
        "equivalent_call_count": len(matches),
        "semantic_signature": latest_signature,
        "packet_ids_changed": len(set(packet_ids)) > 1,
        "intervening_tools": list(dict.fromkeys(intervening_tools)),
        "reusable_bundles": reusable_bundles,
        "interpretation": (
            "The same semantic request recurred inside a read-only/planning cycle with "
            "no world change. Consume the latest successful result or a ready bundle. "
            "Repeat localization only when current visual/VDM evidence shows object "
            "motion, occlusion, mask invalidity, or another material input change."
        ),
        "host_policy": "reflection_warning_only; no tool is forced or blocked",
    }


def _semantic_state_cycle_warning(memory: AgentMemory) -> JsonDict | None:
    """Detect a repeated negative outcome under identical visual/epoch evidence."""

    actions: list[JsonDict] = []
    world_mutating = {
        "move_to",
        "follow_eef_trajectory",
        "gripper_control",
        "lower_body_control_policy",
    }
    for event in reversed(memory.events[-240:]):
        if event.event_type == "observed_object_scene_change":
            break
        if event.event_type != "action":
            continue
        command = event.payload.get("command")
        command = command if isinstance(command, dict) else {}
        request = command.get("request")
        request = request if isinstance(request, dict) else {}
        if str(request.get("kind") or "") != "tool_call":
            continue
        name = str(request.get("name") or "")
        if name in world_mutating:
            break
        anchor = event.payload.get("input_state_anchor")
        anchor = anchor if isinstance(anchor, dict) else {}
        visual_signature = str(anchor.get("visual_signature") or "")
        if not visual_signature:
            continue
        semantic_outcome = _event_tool_semantic_outcome(command, name=name)
        actions.append(
            {
                "name": name,
                "semantic_outcome": semantic_outcome,
                "anchor": anchor,
            }
        )
        if len(actions) >= 16:
            break
    if len(actions) < 2:
        return None
    latest = actions[0]
    latest_outcome = str(latest.get("semantic_outcome") or "")
    if not _is_no_progress_semantic_outcome(latest_outcome):
        return None
    latest_anchor = latest["anchor"]
    match_index: int | None = None
    repeat_count = 1
    for index, candidate in enumerate(actions[1:], start=1):
        if (
            candidate.get("name") == latest.get("name")
            and candidate.get("semantic_outcome") == latest_outcome
            and _same_semantic_state_anchor(candidate.get("anchor"), latest_anchor)
        ):
            repeat_count += 1
            if match_index is None:
                match_index = index
    if match_index is None:
        return None
    intervening_tools = [
        str(item.get("name") or "") for item in reversed(actions[1:match_index])
    ]
    pending = _pending_ik_execution_index(memory)
    pending_ids = []
    if isinstance(pending, dict):
        pending_ids = [
            str(item.get("receipt_id") or "")
            for item in pending.get("receipts", [])
            if isinstance(item, dict) and item.get("receipt_id")
        ]
    return {
        "schema_version": "openeta.no_progress_tool_loop.v1",
        "trigger_type": "semantic_state_cycle_without_world_change",
        "tool": latest.get("name"),
        "semantic_outcome": latest_outcome,
        "equivalent_outcome_count": repeat_count,
        "intervening_tools": list(dict.fromkeys(intervening_tools)),
        "state_anchor": {
            "visual_signature": latest_anchor.get("visual_signature"),
            "object_scene_epoch": latest_anchor.get("object_scene_epoch"),
            "robot_motion_epoch": latest_anchor.get("robot_motion_epoch"),
        },
        "pending_execution_receipt_ids": pending_ids,
        "interpretation": (
            "The same non-progress outcome recurred after a multi-tool read-only "
            "cycle while the visual observation hash and both world epochs stayed "
            "unchanged. Additional same-state perception is unlikely to add evidence. "
            "Consume a useful pending execution reference, materially change the "
            "geometry/evidence, choose another strategy, or explain why retry is useful."
        ),
        "host_policy": "reflection_warning_only; no tool is forced or blocked",
    }


def _event_tool_semantic_outcome(command: JsonDict, *, name: str) -> str:
    calls = command.get("tool_calls")
    calls = calls if isinstance(calls, list) else []
    for call in reversed(calls):
        if not isinstance(call, dict) or str(call.get("name") or "") != name:
            continue
        result = call.get("result")
        result = result if isinstance(result, dict) else {}
        details = result.get("details")
        details = details if isinstance(details, dict) else {}
        outcome = details.get("semantic_outcome")
        if isinstance(outcome, str):
            return outcome.strip().lower()
    return ""


def _is_no_progress_semantic_outcome(value: str) -> bool:
    normalized = str(value or "").strip().lower()
    if not normalized:
        return False
    exact = {
        "requires_better_view",
        "no_detection",
        "not_found",
        "target_not_reached",
        "no_material_view_change",
        "unchanged",
        "inconclusive",
        "retry_required",
        "blocked",
        "rejected",
        "failed",
    }
    return normalized in exact or any(
        marker in normalized
        for marker in ("not_reached", "no_detection", "better_view", "infeasible")
    )


def _same_semantic_state_anchor(left: object, right: object) -> bool:
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    return bool(left.get("visual_signature")) and all(
        left.get(field_name) == right.get(field_name)
        for field_name in (
            "visual_signature",
            "object_scene_epoch",
            "robot_motion_epoch",
        )
    )


def _semantic_repeat_parameters(name: str, parameters: JsonDict) -> JsonDict:
    stable = _without_packet_provenance(parameters)
    stable = stable if isinstance(stable, dict) else {}
    if name == "sam3" and not stable.get("mode"):
        stable["mode"] = "text"
    if name == "select_sam3_detection":
        return {
            key: stable.get(key)
            for key in (
                "evidence_role",
                "identity_anchor_id",
                "identity_relation",
                "target_geometry_family",
            )
            if stable.get(key) is not None
        }
    return stable


def _motion_failure_attractor_warning(memory: AgentMemory) -> JsonDict | None:
    """Expose repeated controller convergence to the same wrong endpoint.

    This is outcome-based rather than request-adjacency-based: perception,
    preview, and recovery calls may occur between two attempts.  It remains a
    reflection warning only and never selects or blocks an action.
    """

    failures: dict[str, JsonDict] = {}
    for event in reversed(memory.events[-160:]):
        if event.event_type == "observed_object_scene_change":
            break
        if event.event_type != "action":
            continue
        command = event.payload.get("command")
        command = command if isinstance(command, dict) else {}
        calls = command.get("tool_calls")
        calls = calls if isinstance(calls, list) else []
        for call in reversed(calls):
            if not isinstance(call, dict) or str(call.get("name") or "") != "move_to":
                continue
            result = call.get("result")
            result = result if isinstance(result, dict) else {}
            details = result.get("details")
            details = details if isinstance(details, dict) else {}
            if details.get("operational_success") is not False:
                continue
            outputs = details.get("outputs")
            outputs = outputs if isinstance(outputs, dict) else {}
            motion = outputs.get("motion_summary")
            motion = motion if isinstance(motion, dict) else {}
            if motion.get("reached_target") is not False:
                continue
            parameters = call.get("parameters")
            parameters = parameters if isinstance(parameters, dict) else {}
            pose_signature = _motion_pose_policy_signature(parameters)
            end = motion.get("end")
            end = end if isinstance(end, dict) else {}
            actual_xyz = end.get("xyz")
            if not pose_signature or not _finite_planner_xyz(actual_xyz):
                continue
            current = {
                "actual_xyz": [float(value) for value in actual_xyz[:3]],
                "position_error_m": motion.get("position_error_m"),
                "steps_executed": motion.get("steps_executed"),
                "stop_reason": motion.get("stop_reason"),
            }
            newer = failures.get(pose_signature)
            if isinstance(newer, dict) and _planner_xyz_distance(
                newer.get("actual_xyz"), current["actual_xyz"]
            ) <= 0.005:
                return {
                    "schema_version": "openeta.no_progress_tool_loop.v1",
                    "trigger_type": "repeated_failed_motion_attractor",
                    "tool": "move_to",
                    "semantic_signature": pose_signature,
                    "attempt_count": 2,
                    "actual_endpoint_a_xyz": newer.get("actual_xyz"),
                    "actual_endpoint_b_xyz": current["actual_xyz"],
                    "position_error_m": newer.get("position_error_m"),
                    "steps_executed": newer.get("steps_executed"),
                    "stop_reason": newer.get("stop_reason"),
                    "interpretation": (
                        "Multiple executions of the same endpoint/orientation policy "
                        "converged to the same wrong EEF pose within 5 mm. More retries "
                        "are unlikely to add evidence. Change candidate, orientation "
                        "policy, controller-compatible waypoint, or recovery strategy."
                    ),
                    "host_policy": "reflection_warning_only; no tool is forced or blocked",
                }
            failures[pose_signature] = current
    return None


def _motion_pose_policy_signature(parameters: JsonDict) -> str:
    target = parameters.get("target_pose")
    if not isinstance(target, dict):
        return ""
    xyz = target.get("xyz", target.get("translation_xyz"))
    if not _finite_planner_xyz(xyz):
        return ""
    orientation = {
        key: target.get(key)
        for key in (
            "rotation_matrix",
            "quat_xyzw",
            "quaternion",
            "rotvec",
            "roll",
            "pitch",
            "yaw",
        )
        if target.get(key) is not None
    }
    preserve_current = parameters.get("preserve_current_orientation")
    if preserve_current is None:
        preserve_current = not orientation
    canonical = {
        "target_xyz": [float(value) for value in xyz[:3]],
        "orientation_policy": (
            "preserve_current" if preserve_current is True else "explicit_orientation"
        ),
        "orientation": orientation,
    }
    return sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]


def _finite_planner_xyz(value: object) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) >= 3
        and all(
            isinstance(item, (int, float))
            and not isinstance(item, bool)
            and math.isfinite(float(item))
            for item in value[:3]
        )
    )


def _planner_xyz_distance(left: object, right: object) -> float:
    if not _finite_planner_xyz(left) or not _finite_planner_xyz(right):
        return float("inf")
    return math.sqrt(sum((float(left[i]) - float(right[i])) ** 2 for i in range(3)))


def _without_packet_provenance(value: object) -> object:
    if isinstance(value, dict):
        return {
            str(key): _without_packet_provenance(item)
            for key, item in value.items()
            if str(key)
            not in {
                "source_packet_id",
                "source_packet_ids",
                "source_observation",
                "source_observations",
            }
        }
    if isinstance(value, (list, tuple)):
        return [_without_packet_provenance(item) for item in value]
    return value


def _latest_action_effect(recent_events: list[JsonDict]) -> JsonDict | None:
    """Project only the most recent action outcome into the bounded decision state."""

    for event in reversed(recent_events):
        if not isinstance(event, dict) or event.get("type") != "action":
            continue
        payload = event.get("payload")
        payload = payload if isinstance(payload, dict) else {}
        command = payload.get("command")
        command = command if isinstance(command, dict) else {}
        calls = command.get("tool_calls")
        calls = calls if isinstance(calls, list) else []
        call = next((item for item in reversed(calls) if isinstance(item, dict)), None)
        metadata = command.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        if call is None:
            return {
                "request": command.get("request"),
                "status": command.get("status"),
                "repair_bundle": metadata.get("repair_bundle"),
            }
        result = call.get("result")
        result = result if isinstance(result, dict) else {}
        details = result.get("details")
        details = details if isinstance(details, dict) else {}
        outputs = details.get("outputs")
        outputs = outputs if isinstance(outputs, dict) else {}
        grasp_selection_advice = details.get("grasp_selection_advice")
        if not isinstance(grasp_selection_advice, dict):
            grasp_selection_advice = outputs.get("grasp_selection_advice")
        grasp_selection_bundle = details.get("grasp_selection_bundle")
        if not isinstance(grasp_selection_bundle, dict):
            grasp_selection_bundle = outputs.get("grasp_selection_bundle")
        artifacts = details.get("artifacts")
        artifacts = artifacts if isinstance(artifacts, list) else []
        return {
            "tool": call.get("name"),
            "status": call.get("status"),
            "content": str(result.get("content") or "")[:4_000],
            "operational_success": details.get(
                "operational_success", result.get("success")
            ),
            "semantic_outcome": details.get("semantic_outcome"),
            "effect": details.get("effect"),
            "facts_produced": details.get("facts_produced", []),
            "recovery_options": details.get("recovery_options", []),
            "outputs": _project_latest_tool_outputs(
                str(call.get("name") or ""),
                outputs,
            ),
            "grasp_selection_advice": _bounded_decision_value(
                grasp_selection_advice
            )
            if isinstance(grasp_selection_advice, dict)
            else None,
            "grasp_selection_bundle": _bounded_decision_value(
                grasp_selection_bundle
            )
            if isinstance(grasp_selection_bundle, dict)
            else None,
            "artifact_refs": [
                ref
                for artifact in artifacts[:12]
                if isinstance(artifact, dict)
                for key in ("path", "mask_ref", "overlay_ref", "crop_ref", "response_path")
                if isinstance((ref := artifact.get(key)), str) and ref
            ][:12],
            "repair_bundle": metadata.get("repair_bundle"),
        }
    return None


def _recent_high_fidelity_transitions(
    events: list[JsonDict],
    *,
    observation_turns: int,
) -> list[JsonDict]:
    """Project recent non-conversation evidence without replaying full actions.

    Action requests and host ToolResults already live in the bounded canonical
    conversation, while action/environment outcomes have a compact durable
    ``transition_ledger`` representation.  This window therefore carries only
    observation and recovery evidence that those two layers do not represent.
    Complete events remain in the append-only session trace.
    """

    if observation_turns <= 0:
        return []
    relevant = [
        (
            _project_recovery_feedback_event(event)
            if event.get("type") == "recovery_feedback"
            else event
        )
        for event in events
        if isinstance(event, dict)
        and event.get("type") in {"observation", "recovery_feedback"}
    ]
    observation_indices = [
        index
        for index, event in enumerate(relevant)
        if event.get("type") == "observation"
    ]
    if observation_indices:
        start = observation_indices[max(0, len(observation_indices) - observation_turns)]
        return relevant[start:]

    # Recovery feedback can precede the first canonical observation (for
    # example, a preflight rejection). Keep a small bounded fallback rather than
    # silently hiding the only actionable evidence.
    return relevant[-observation_turns:]


def _project_recovery_feedback_event(event: JsonDict) -> JsonDict:
    """Keep actionable failure evidence without replaying a full ToolResult."""

    payload = event.get("payload")
    payload = payload if isinstance(payload, dict) else {}
    command = payload.get("command")
    command = command if isinstance(command, dict) else {}
    effect = _latest_action_effect(
        [{"type": "action", "payload": {"command": command}}]
    )
    request = command.get("request")
    request = request if isinstance(request, dict) else {}
    return {
        "type": "recovery_feedback",
        "timestamp_s": event.get("timestamp_s"),
        "payload": {
            "source": payload.get("source"),
            "keys": payload.get("keys"),
            "request": {
                "name": request.get("name"),
                "parameters": _bounded_decision_value(
                    request.get("parameters", {}),
                    depth=4,
                    max_items=16,
                    max_string_chars=500,
                ),
            },
            "outcome": effect,
            "durable_trace": "full recovery event remains in the session event log",
        },
    }


def _project_latest_tool_outputs(tool: str, outputs: JsonDict) -> object:
    """Bound the latest result by semantic fields, not arbitrary raw size."""

    if tool == "prepare_attachment_probe":
        return _bounded_decision_value(
            {
                key: outputs[key]
                for key in (
                    "schema_version",
                    "status",
                    "probe_id",
                    "compiled_grasp_id",
                    "scene_epoch",
                    "robot_motion_epoch",
                    "motion_type",
                    "path_sha256",
                    "ik_preview_requests",
                    "execution_handoff",
                )
                if key in outputs
            },
            depth=7,
            max_items=16,
            max_string_chars=1_500,
        )
    projected = _bounded_decision_value(outputs)
    serialized = json.dumps(projected, ensure_ascii=False, separators=(",", ":"))
    if len(serialized) <= 8_000:
        return projected
    priority = {
        "schema_version",
        "result_id",
        "candidate_count",
        "best_grasp_candidate",
        "selected_grasp_source",
        "grasp_selection_advice",
        "grasp_selection_bundle",
        "compiled_grasp_id",
        "contact_pose",
        "hover_pose",
        "precontact_pose",
        "execution_guidance",
        "ik_receipt",
        "receipt",
        "reachability",
        "classification",
        "reason_code",
        "message",
        "suggestions",
        "motion_summary",
        "collision_coverage",
        "pose_feedback",
        "attachment_proxy_receipt",
        "gripper_actuation_receipt",
        "observation_summary",
        "response_path",
        "raw_output_ref",
        "complete_outputs_artifact",
    }
    selected = {
        key: value
        for key, value in outputs.items()
        if key in priority
    }
    compact = _bounded_decision_value(
        selected,
        depth=4,
        max_items=12,
        max_string_chars=750,
    )
    if isinstance(compact, dict):
        compact["projection"] = {
            "tool": tool,
            "full_output_omitted": True,
            "available_via": "artifact path or durable tool receipt",
        }
    return compact


def _project_world_evidence(evidence: JsonDict) -> JsonDict:
    """Project large evidence banks while preserving their durable runtime form."""

    projected = dict(evidence)
    ik_entry = projected.get("ik_preview_receipts")
    if isinstance(ik_entry, dict):
        value = ik_entry.get("value")
        if isinstance(value, dict):
            receipts = value.get("receipts")
            receipts = receipts if isinstance(receipts, list) else []
            compact_receipts = [
                _compact_ik_receipt(receipt)
                for receipt in receipts
                if isinstance(receipt, dict)
            ]
            projected["ik_preview_receipts"] = {
                **ik_entry,
                "value": {
                    "schema_version": "openeta.ik_preview_receipt_projection.v1",
                    "receipt_count": len(receipts),
                    "latest": _bounded_decision_value(
                        value.get("latest", {}),
                        depth=5,
                        max_items=24,
                        max_string_chars=1_000,
                    ),
                    "index": compact_receipts,
                    "query": (
                        "Use the exact receipt_id in ik_preview_check/move_to. Full "
                        "receipts remain in durable memory and rollout artifacts."
                    ),
                },
            }
    candidate_entry = projected.get("grasp_candidates")
    if isinstance(candidate_entry, dict):
        value = candidate_entry.get("value")
        if isinstance(value, dict):
            candidates = value.get("grasp_candidates")
            candidates = candidates if isinstance(candidates, list) else []
            advice = value.get("grasp_selection_advice")
            advice = advice if isinstance(advice, dict) else {}
            recommended_id = str(advice.get("recommended_candidate_id") or "")
            visible_ids = {
                recommended_id,
                *(
                    str(item)
                    for item in advice.get("alternatives", [])
                    if isinstance(item, str)
                ),
            }
            visible = [
                candidate
                for candidate in candidates
                if isinstance(candidate, dict)
                and str(candidate.get("id") or "") in visible_ids
            ]
            if not visible:
                visible = [item for item in candidates[:3] if isinstance(item, dict)]
            projected["grasp_candidates"] = {
                **candidate_entry,
                "value": {
                    key: value.get(key)
                    for key in (
                        "type",
                        "result_id",
                        "candidate_count",
                        "scene_epoch",
                        "source_tool",
                        "source_backend",
                        "source_rgb",
                        "source_depth",
                        "target_mask",
                        "raw_output_ref",
                        "complete_outputs_artifact",
                        "query_hint",
                    )
                    if value.get(key) is not None
                },
            }
            projected_value = projected["grasp_candidates"]["value"]
            projected_value["visible_candidates"] = _bounded_decision_value(
                visible[:4], depth=5, max_items=16, max_string_chars=750
            )
            projected_value["grasp_selection_advice"] = {
                key: advice.get(key)
                for key in (
                    "status",
                    "decision",
                    "recommended_candidate_id",
                    "alternatives",
                    "confidence",
                    "reasons",
                    "uncertainties",
                    "bundle_id",
                )
                if advice.get(key) is not None
            }
            projected_value["projection"] = {
                "visible_candidate_count": len(visible[:4]),
                "full_candidate_count": len(candidates),
                "full_bank_available_via": (
                    value.get("complete_outputs_artifact")
                    or value.get("raw_output_ref")
                ),
            }
    return projected


def _compact_ik_receipt(receipt: JsonDict) -> JsonDict:
    target = receipt.get("target_pose")
    target = target if isinstance(target, dict) else {}
    return {
        key: receipt.get(key)
        for key in (
            "receipt_id",
            "classification",
            "reason_code",
            "orientation_policy",
            "pose_policy_signature",
            "object_scene_epoch",
            "robot_motion_epoch",
        )
        if receipt.get(key) is not None
    } | {
        "target": {
            key: target.get(key)
            for key in ("xyz", "waypoint_role", "compiled_grasp_id")
            if target.get(key) is not None
        }
    }


def _project_perception_failure(
    failure: JsonDict,
    *,
    retained_grasp: object,
) -> JsonDict:
    projected = dict(failure)
    frame_id = str(
        failure.get("frame_id")
        or failure.get("camera_frame_id")
        or ""
    )
    if "wrist" not in frame_id.lower():
        return projected
    retained = retained_grasp if isinstance(retained_grasp, dict) else {}
    projected["workflow_impact"] = {
        "classification": "optional_wrist_refinement_unavailable",
        "coarse_grasp_invalidated": False,
        "retained_compiled_grasp_id": retained.get("compiled_grasp_id"),
        "guidance": (
            "A wrist-view segmentation miss does not invalidate current-epoch "
            "scene-view grasp evidence. Inspect the fresh dual view and either use "
            "the retained grasp as a reference, make an Agent-owned bounded residual "
            "adjustment, or pursue same-view point grounding only when refinement is "
            "actually needed. Do not restart the entire scene pipeline by default."
        ),
        "host_policy": "advisory_only",
    }
    return projected


def _project_transition_ledger(value: object) -> list[JsonDict]:
    """Drop repeated zero-reward receipts from the model projection only."""

    rows = (
        [dict(row) for row in value if isinstance(row, dict)]
        if isinstance(value, list)
        else []
    )
    zero_environment_indices = [
        index
        for index, row in enumerate(rows)
        if row.get("tool") == "environment_receipt"
        and float(row.get("reward") or 0.0) <= 0
        and row.get("terminated") is not True
        and row.get("truncated") is not True
    ]
    latest_zero_environment_index = (
        zero_environment_indices[-1] if zero_environment_indices else None
    )
    projected: list[JsonDict] = []
    for index, row in enumerate(rows):
        if row.get("tool") != "environment_receipt":
            projected.append(row)
            continue
        if (
            float(row.get("reward") or 0.0) > 0
            or row.get("terminated") is True
            or row.get("truncated") is True
        ):
            projected.append(row)
            continue
        if index == latest_zero_environment_index:
            projected.append(
                {
                    **row,
                    "projection_note": (
                        f"latest of {len(zero_environment_indices)} repeated zero-reward "
                        "environment receipts; full ledger remains durable"
                    ),
                }
            )
    return projected


def _bounded_artifact_index(
    artifacts: JsonDict,
    *,
    max_entries: int = 24,
    max_total_chars: int = 30_000,
) -> JsonDict:
    """Expose recent artifact metadata without reinjecting the full durable bank."""

    selected: list[tuple[str, object]] = []
    used_chars = 0
    for key, value in reversed(list(artifacts.items())):
        if len(selected) >= max_entries:
            break
        projected = _bounded_decision_value(
            value,
            depth=6,
            max_items=16,
            max_string_chars=1_000,
        )
        serialized = json.dumps(projected, ensure_ascii=False, separators=(",", ":"))
        if len(serialized) > 6_000:
            projected = _bounded_decision_value(
                value,
                depth=3,
                max_items=8,
                max_string_chars=500,
            )
            serialized = json.dumps(projected, ensure_ascii=False, separators=(",", ":"))
        if selected and used_chars + len(serialized) > max_total_chars:
            break
        selected.append((str(key), projected))
        used_chars += len(serialized)

    selected.reverse()
    projection: JsonDict = {
        "schema_version": "openeta.artifact_index_projection.v1",
        "total_count": len(artifacts),
        "visible_count": len(selected),
        "truncated": len(selected) < len(artifacts),
    }
    if projection["truncated"]:
        projection["query"] = {
            "metadata": "call get_memory with namespace='artifacts'",
            "files": (
                "call python_exec and assign result = artifacts.list_files(pattern='*')"
            ),
            "policy": "use the exact returned path; do not invent artifact aliases",
        }
    return {"__index__": projection, **dict(selected)}


def _bounded_decision_value(
    value: object,
    *,
    depth: int = 5,
    max_items: int = 16,
    max_string_chars: int = 1_500,
) -> object:
    """Keep the latest tool result actionable without embedding large artifacts."""

    if depth <= 0:
        return "<omitted>"
    if isinstance(value, str):
        if value.startswith("data:image/") or value.startswith("data:application/"):
            return "<inline_artifact_omitted>"
        return (
            value
            if len(value) <= max_string_chars
            else value[:max_string_chars] + "...[truncated]"
        )
    if isinstance(value, dict):
        return {
            str(key): _bounded_decision_value(
                item,
                depth=depth - 1,
                max_items=max_items,
                max_string_chars=max_string_chars,
            )
            for key, item in list(value.items())[:max_items]
        }
    if isinstance(value, (list, tuple)):
        return [
            _bounded_decision_value(
                item,
                depth=depth - 1,
                max_items=max_items,
                max_string_chars=max_string_chars,
            )
            for item in value[:max_items]
        ]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:max_string_chars]


def _matched_task_playbook(
    *,
    observation: EnvObservation,
    memory: AgentMemory,
    task: str,
) -> JsonDict | None:
    metadata = observation.metadata
    environment_id = str(metadata.get("env_id") or memory.metadata.get("env_id") or "")
    suite = str(metadata.get("suite") or memory.metadata.get("suite") or "")
    task_index = metadata.get("task_index", memory.metadata.get("task_index"))
    if not environment_id or not suite or isinstance(task_index, bool) or not isinstance(task_index, int):
        return None
    workspace = memory.metadata.get("workspace")
    workspace = workspace if isinstance(workspace, dict) else {}
    root = Path(str(workspace.get("task_playbook_root") or DEFAULT_TASK_PLAYBOOK_ROOT))
    calibration_id = str(
        metadata.get("calibration_profile_id")
        or memory.metadata.get("calibration_profile_id")
        or workspace.get("grasp_profile_id")
        or ""
    )
    try:
        playbooks = load_task_playbooks(root)
        return select_task_playbook(
            playbooks,
            environment_id=environment_id,
            suite=suite,
            task_index=task_index,
            task=task,
            calibration_id=calibration_id,
        )
    except (OSError, json.JSONDecodeError, TaskPlaybookError):
        return None


def _current_camera_artifacts(
    observation: EnvObservation,
    *,
    memory: AgentMemory,
) -> list[JsonDict]:
    """Return current RGB/depth artifacts in stable planner preference order."""

    if observation.metadata.get("fresh_observation_required") is True:
        return []
    raw_artifacts = observation.metadata.get("image_artifacts")
    if not isinstance(raw_artifacts, list):
        return []
    preferred_frames = {"agentview": 0, "render": 1, "wrist": 2}
    artifacts: list[JsonDict] = []
    for index, raw in enumerate(raw_artifacts):
        if not isinstance(raw, dict) or raw.get("kind") not in {"rgb", "depth"}:
            continue
        path = raw.get("path")
        if not isinstance(path, str) or not path:
            continue
        frame_id = str(raw.get("frame_id") or "")
        kind = str(raw["kind"])
        artifact: JsonDict = {
            "frame_id": frame_id,
            "kind": kind,
            "path": path,
        }
        role = str(raw.get("role") or "")
        if role:
            artifact["role"] = role
        for artifact_field in ("packet_id", "width", "height", "format", "index"):
            value = raw.get(artifact_field)
            if value is not None:
                artifact[artifact_field] = value
        packet_reference = memory.observation_packet_reference_for_path(path)
        if packet_reference.get("source_packet_id"):
            artifact["packet_id"] = packet_reference["source_packet_id"]
        artifact["_sort_key"] = (
            _CAMERA_ROLE_PREFERENCE.get(role, preferred_frames.get(frame_id, 3)),
            0 if kind == "rgb" else 1,
            index,
        )
        artifacts.append(artifact)
    artifacts.sort(key=lambda artifact: artifact["_sort_key"])
    for artifact in artifacts:
        artifact.pop("_sort_key", None)
    return artifacts


def _current_vision_evidence(
    observation: EnvObservation,
    *,
    image_paths: list[str],
    camera_artifacts: list[JsonDict],
) -> list[JsonDict]:
    """Label every planner image as current evidence with stable provenance."""

    cameras = {camera.frame_id: camera for camera in observation.cameras}
    step_idx = observation.metadata.get("step_idx")
    evidence: list[JsonDict] = []
    for image_index, path in enumerate(image_paths):
        artifact = next(
            (
                item
                for item in camera_artifacts
                if item.get("kind") == "rgb" and item.get("path") == path
            ),
            {},
        )
        frame_id = str(artifact.get("frame_id") or f"camera_{image_index}")
        camera = cameras.get(frame_id)
        camera_role = str(artifact.get("role") or getattr(camera, "role", "") or "scene")
        timestamp_s = getattr(camera, "timestamp_s", None)
        evidence_id = f"current_observation:{step_idx if step_idx is not None else 'na'}:{frame_id}"
        item: JsonDict = {
            "evidence_id": evidence_id,
            "role": "current_scene",
            "camera_role": camera_role,
            "frame_id": frame_id,
            "path": path,
            "freshness": "current",
        }
        if step_idx is not None:
            item["observation_step"] = step_idx
        if timestamp_s is not None:
            item["timestamp_s"] = timestamp_s
        evidence.append(item)
    return evidence


def _current_camera_calibrations(observation: EnvObservation) -> list[JsonDict]:
    """Expose current numeric calibration without pixel payloads."""

    if observation.metadata.get("fresh_observation_required") is True:
        return []
    calibrations: list[JsonDict] = []
    for camera in observation.cameras:
        if not camera.intrinsics and not camera.extrinsics:
            continue
        calibration: JsonDict = {
            "frame_id": camera.frame_id,
            "intrinsics": dict(camera.intrinsics),
            "extrinsics": dict(camera.extrinsics),
        }
        if camera.role:
            calibration["role"] = camera.role
        calibrations.append(calibration)
    return calibrations


def _camera_item_role(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("role") or "")
    return str(getattr(value, "role", "") or "")


def _camera_item_frame_id(value: object) -> str:
    if isinstance(value, dict):
        return str(value.get("camera_frame_id") or value.get("frame_id") or "")
    return str(getattr(value, "frame_id", "") or "")


def _camera_matches(
    value: object,
    *,
    roles: set[str],
    legacy_frames: set[str],
) -> bool:
    role = _camera_item_role(value)
    if role in _CAMERA_ROLE_PREFERENCE:
        return role in roles
    return _camera_item_frame_id(value) in legacy_frames


def _is_primary_planner_camera(value: object) -> bool:
    return _camera_matches(
        value,
        roles={"scene_primary", "wrist_primary"},
        legacy_frames={"agentview", "wrist"},
    )


def _context_budget_status(
    context: JsonDict,
    *,
    config: PlannerContextConfig,
    auto_compact_triggered: bool,
    conversation_messages: list[JsonDict] | None = None,
    system_prompt: str = "",
    projection: JsonDict | None = None,
) -> JsonDict:
    conversation_messages = conversation_messages or []
    agent_context = context.get("agent_context")
    budget_context = agent_context if isinstance(agent_context, dict) else context
    estimate = _planner_input_estimate(
        budget_context,
        conversation_messages,
        system_prompt=system_prompt,
        config=config,
    )
    estimated_chars = estimate.chars
    estimated_tokens = estimate.tokens
    trigger_ratio = min(max(config.auto_compact_trigger_ratio, 0.0), 1.0)
    trigger_tokens = (
        max(
            1,
            int(config.context_window_tokens * trigger_ratio)
            - max(0, config.reserved_output_tokens),
        )
        if config.context_window_tokens is not None
        else None
    )
    tokens_until_auto_compact = (
        max(0, trigger_tokens - estimated_tokens) if trigger_tokens is not None else None
    )
    should_auto_compact = False
    return {
        "schema_version": "openeta.context_budget.v2",
        "auto_compact_enabled": config.auto_compact_enabled,
        "auto_compact_triggered": auto_compact_triggered,
        "should_auto_compact": should_auto_compact,
        "context_window_tokens": config.context_window_tokens,
        "trigger_ratio": trigger_ratio,
        "trigger_tokens": trigger_tokens,
        "estimated_chars": estimated_chars,
        "estimated_tokens": estimated_tokens,
        "conversation_message_count": len(conversation_messages),
        "tokens_until_auto_compact": tokens_until_auto_compact,
        "reserved_output_tokens": max(0, config.reserved_output_tokens),
        "projection": dict(projection or {}),
        "estimator": estimate.estimator,
    }


def _project_planner_input_to_budget(
    context: JsonDict,
    *,
    config: PlannerContextConfig,
    conversation_messages: list[JsonDict],
    system_prompt: str,
) -> tuple[list[JsonDict], JsonDict]:
    """Fit elastic history to one prompt budget without mutating durable memory."""

    messages = [dict(message) for message in conversation_messages]
    window = config.context_window_tokens
    trigger_ratio = min(max(config.auto_compact_trigger_ratio, 0.0), 1.0)
    target_tokens = (
        max(1, int(window * trigger_ratio) - max(0, config.reserved_output_tokens))
        if window is not None
        else None
    )
    agent_context = context.get("agent_context")
    agent_context = agent_context if isinstance(agent_context, dict) else context
    initial = _planner_input_estimate(
        agent_context,
        messages,
        system_prompt=system_prompt,
        config=config,
    )
    dropped = {
        "recent_transitions": 0,
        "transition_ledger": 0,
        "visual_deltas": 0,
        "conversation_messages": 0,
    }

    def over_budget() -> bool:
        if target_tokens is None:
            return False
        return _planner_input_estimate(
            agent_context,
            messages,
            system_prompt=system_prompt,
            config=config,
        ).tokens > target_tokens

    while config.auto_compact_enabled and over_budget():
        transitions = agent_context.get("recent_transitions")
        if isinstance(transitions, list) and len(transitions) > 1:
            transitions.pop(0)
            dropped["recent_transitions"] += 1
            continue
        ledger = agent_context.get("transition_ledger")
        if isinstance(ledger, list) and len(ledger) > 1:
            ledger.pop(0)
            dropped["transition_ledger"] += 1
            continue
        visual_history = agent_context.get("visual_history")
        deltas = (
            visual_history.get("compressed_deltas")
            if isinstance(visual_history, dict)
            else None
        )
        if isinstance(deltas, list) and len(deltas) > 1:
            deltas.pop(0)
            dropped["visual_deltas"] += 1
            continue
        removed = _drop_oldest_conversation_action_group(messages)
        if removed:
            dropped["conversation_messages"] += removed
            continue
        break

    final = _planner_input_estimate(
        agent_context,
        messages,
        system_prompt=system_prompt,
        config=config,
    )
    projection = {
        "policy": "elastic_total_token_budget",
        "triggered": target_tokens is not None and initial.tokens > target_tokens,
        "entries_removed": any(dropped.values()),
        "target_input_tokens": target_tokens,
        "initial_estimated_tokens": initial.tokens,
        "final_estimated_tokens": final.tokens,
        "fits_target": target_tokens is None or final.tokens <= target_tokens,
        "dropped": dropped,
        "durable_history_mutated": False,
    }
    budget = _context_budget_status(
        context,
        config=config,
        auto_compact_triggered=projection["triggered"],
        conversation_messages=messages,
        system_prompt=system_prompt,
        projection=projection,
    )
    return messages, budget


def _planner_input_estimate(
    agent_context: JsonDict,
    messages: list[JsonDict],
    *,
    system_prompt: str,
    config: PlannerContextConfig,
) -> TokenEstimate:
    text_estimate = estimate_json_tokens(
        {
            "system_prompt": system_prompt,
            "conversation_messages": messages,
            "tool_context": agent_context,
        },
        model=config.token_estimator_model,
        approx_chars_per_token=config.approx_chars_per_token,
    )
    image_paths = agent_context.get("vision_image_paths")
    unique_image_count = len(
        {
            str(path)
            for path in (image_paths if isinstance(image_paths, list) else [])
            if isinstance(path, str) and path
        }
    )
    image_tokens = unique_image_count * max(0, config.approx_tokens_per_image)
    return TokenEstimate(
        tokens=text_estimate.tokens + image_tokens,
        chars=text_estimate.chars,
        estimator={
            **text_estimate.estimator,
            "image_estimate": {
                "image_count": unique_image_count,
                "approx_tokens_per_image": max(0, config.approx_tokens_per_image),
                "estimated_image_tokens": image_tokens,
            },
        },
    )


def _drop_oldest_conversation_action_group(messages: list[JsonDict]) -> int:
    """Drop one old action/result pair while preserving dialogue constraints."""

    for index, message in enumerate(messages[:-1]):
        content = str(message.get("content") or "")
        if message.get("role") != "assistant" or '"openeta_action"' not in content:
            continue
        removed = 1
        if index + 1 < len(messages):
            following = messages[index + 1]
            if (
                following.get("role") == "user"
                and "OpenETA host execution evidence" in str(following.get("content") or "")
            ):
                removed = 2
        del messages[index : index + removed]
        return removed
    # If only dialogue remains, retain the initial task and latest message.
    if len(messages) > 2:
        del messages[1]
        return 1
    return 0


def _planner_metadata(
    *,
    planner: ToolCallingPlanner,
    tool_context: JsonDict,
    backend: PlannerBackend,
    backend_result: PlannerBackendResult | None = None,
    backend_usage: JsonDict | None = None,
    backend_usage_sources: JsonDict | None = None,
    validation_attempts: int | None = None,
    validation_attempt_history: list[JsonDict] | None = None,
    validation_errors: list[str] | None = None,
    policy_redirect: JsonDict | None = None,
) -> JsonDict:
    metadata: JsonDict = {
        "planner": type(planner).__name__,
        "tool_context_summary": _tool_context_summary(tool_context),
        "backend": backend.descriptor(),
        "execution_model": "closed_loop_tool_calling",
        "planner_prompt": dict(planner.prompt_metadata),
    }
    if backend_result is not None:
        metadata.update(
            {
                "backend_status": backend_result.status.value,
                "backend_provider": backend_result.provider,
                "backend_model": backend_result.model,
                "backend_details": public_backend_details(backend_result.details),
            }
        )
    if backend_usage:
        metadata["backend_usage"] = dict(backend_usage)
    if backend_usage_sources:
        metadata["backend_usage_sources"] = dict(backend_usage_sources)
    if validation_attempts is not None:
        metadata["validation_attempts"] = validation_attempts
    if validation_attempt_history is not None:
        metadata["validation_attempt_history"] = [dict(item) for item in validation_attempt_history]
    if validation_errors is not None:
        metadata["validation_errors"] = list(validation_errors)
    if policy_redirect is not None:
        metadata["policy_redirect"] = dict(policy_redirect)
    return metadata


def _planner_validation_attempt_record(
    *,
    attempt: int,
    result: PlannerBackendResult,
    decision: PlannerDecision,
    validation_errors: list[str],
) -> JsonDict:
    details = result.details if isinstance(result.details, dict) else {}
    record: JsonDict = {
        "attempt": attempt,
        "backend_status": result.status.value,
        "provider": result.provider,
        "model": result.model,
        "decision": {
            "kind": decision.action_type,
            "name": decision.action,
        },
        "validation_errors": list(validation_errors),
        "provider_attempts": max(1, int(details.get("provider_attempts") or 1)),
    }
    for key in ("response_id", "usage_source", "finish_reason"):
        value = details.get(key)
        if isinstance(value, (str, int, float, bool)) and value != "":
            record[key] = value
    usage = details.get("usage")
    if isinstance(usage, dict):
        record["usage"] = {
            str(key): value
            for key, value in usage.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
    retry_errors = details.get("retry_errors")
    if isinstance(retry_errors, list) and retry_errors:
        record["provider_retry_errors"] = [
            dict(item) for item in retry_errors if isinstance(item, dict)
        ]
    return record


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
    projection_audit = context.get("tool_contract_projection_audit")
    projection_audit = (
        projection_audit if isinstance(projection_audit, dict) else {}
    )
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
            "has_latest_human_interaction": isinstance(
                memory.get("latest_human_interaction"),
                dict,
            ),
            "has_compact_summary": bool(
                ((memory.get("working_memory") or {}).get("compact_summary"))
                if isinstance(memory.get("working_memory"), dict)
                else False
            ),
        },
        "tool_count": len(context.get("tool_references", []) or []),
        "registered_handler_count": len(context.get("registered_tool_handlers", []) or []),
        "tool_contract_projection": {
            "authoritative_projection": projection_audit.get(
                "authoritative_projection"
            ),
            "tool_count": projection_audit.get("tool_count", 0),
            "matching_tool_count": projection_audit.get("matching_tool_count", 0),
            "mismatch_count": projection_audit.get("mismatch_count", 0),
            "mismatches": list(projection_audit.get("mismatches") or []),
        },
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


def _observation_summary(
    observation: EnvObservation,
    *,
    gripper_command_state: object = None,
) -> JsonDict:
    summary = summarize_observation(observation)
    summary.pop("task", None)
    measured = observation.robot.gripper_state
    measured_aperture: JsonDict = {}
    if isinstance(measured, dict):
        openness = measured.get("openness")
        if isinstance(openness, int | float) and not isinstance(openness, bool):
            measured_aperture["open_fraction"] = float(openness)
        legacy_open = measured.get("open")
        if isinstance(legacy_open, bool):
            measured_aperture["legacy_threshold_open"] = legacy_open
    commanded = (
        dict(gripper_command_state)
        if isinstance(gripper_command_state, dict)
        else None
    )
    summary["gripper_evidence"] = {
        "measured_aperture": measured_aperture,
        "commanded_state": commanded,
        "attachment_status": "unknown_without_co_motion_evidence",
        "semantics": (
            "measured_aperture.open_fraction is continuous sensor feedback, not a "
            "command and not attachment proof. legacy_threshold_open is retained only "
            "for compatibility and must not be read as an open command or empty grasp. "
            "commanded_state is the last acknowledged binary latch command. Determine "
            "attachment from post-lift co-motion/source-vacancy evidence."
        ),
    }
    return summary


def _contract_driven_tool_references(
    tools: list[ToolSpec],
) -> tuple[list[JsonDict], JsonDict]:
    """Build Agent-visible schemas from ToolContract, not duplicate ToolSpec prose."""

    from agent.tools.contracts import (
        audit_agent_tool_projection,
        build_default_tool_contract_catalog,
        project_agent_tool_contract,
    )

    catalog = build_default_tool_contract_catalog(tools)
    references: list[JsonDict] = []
    audit_rows: list[JsonDict] = []
    for spec in tools:
        contract = catalog.get(spec.name)
        references.append(project_agent_tool_contract(contract))
        audit_rows.append(audit_agent_tool_projection(contract, spec))
    mismatches = [row for row in audit_rows if row.get("matches") is not True]
    return references, {
        "schema_version": "openeta.agent_tool_contract_projection_catalog_audit.v1",
        "authoritative_projection": "tool_contract",
        "runtime_authority": "tool_registry_handler_binding",
        "tool_count": len(audit_rows),
        "matching_tool_count": len(audit_rows) - len(mismatches),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "interpretation": (
            "Parameter-name mismatches are migration evidence; they do not change "
            "legacy Planner validation or runtime gate authority."
        ),
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
            "available_allowed_tools",
            "unavailable_allowed_tools",
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


def _annotate_skill_tool_availability(
    skill: JsonDict,
    *,
    executable_tool_names: set[str],
) -> None:
    """Project one static skill contract onto this runtime's bound handlers.

    ``allowed_tools`` remains the durable authoring declaration.  It must not be
    interpreted as proof that an optional MCP/backend is configured for the
    current process, so the Agent receives the executable intersection and the
    unavailable remainder explicitly on every turn.
    """

    declared = [
        str(name)
        for name in skill.get("allowed_tools", [])
        if isinstance(name, str) and name
    ]
    skill["available_allowed_tools"] = [
        name for name in declared if name in executable_tool_names
    ]
    skill["unavailable_allowed_tools"] = [
        name for name in declared if name not in executable_tool_names
    ]


def _selected_skill_guidance(
    skills: list[SkillSpec],
    *,
    observation: EnvObservation,
    memory: AgentMemory,
    config: PlannerContextConfig,
) -> list[JsonDict]:
    effective_task = _effective_task_text(observation, memory)
    scored = [
        (score, _skill_text_relevance_score(skill, effective_task.lower()), skill)
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
    content_limit = config.max_skill_content_chars
    declared_limit = skill.metadata.get("context_char_limit")
    # A skill may request a narrowly scoped production exception when its full
    # safety contract no longer fits the default bound. Explicitly smaller
    # planner configs continue to win, which keeps bounded-context tests and
    # deployments deterministic.
    if (
        content_limit is not None
        and content_limit >= DEFAULT_MAX_SKILL_CONTENT_CHARS
        and isinstance(declared_limit, int)
        and declared_limit > content_limit
    ):
        content_limit = declared_limit
    if content_limit is None:
        content, truncated = skill.content, False
    else:
        content, truncated = _truncate_text(skill.content, content_limit)
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
        "tool_availability_rule": (
            "A skill's allowed_tools is authoring guidance. Call only its "
            "available_allowed_tools that also appear in current tool_references; "
            "report an unavailable capability instead of retrying an unbound tool."
        ),
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
    current_query = _effective_task_text(observation, memory).lower()
    supporting_query = _skill_query_text(observation, memory, include_current_task=False)
    current_score = _skill_text_relevance_score(skill, current_query)
    supporting_score = _skill_text_relevance_score(skill, supporting_query)
    score = current_score * 3 + supporting_score
    current_request = memory.current_user_request
    if current_score == 0 and current_request and current_request.lower() != current_query:
        score += min(3, _skill_text_relevance_score(skill, current_request.lower()))
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
        token for token in _word_tokens(skill.description) if token not in _SKILL_MATCH_STOPWORDS
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
            _effective_task_text(observation, memory) if include_current_task else "",
            *object_names,
            memory.compact_summary,
            *memory.skill_notes.keys(),
        ]
    ).lower()


def _effective_task_text(observation: EnvObservation, memory: AgentMemory) -> str:
    # Benchmark environments often expose their native manipulation instruction
    # in every observation.  Most runs intentionally use that as the objective,
    # but focused evaluation/probe episodes may provide a narrower user request
    # (for example, execute one motion and stop).  Keep that authority explicit
    # in run metadata instead of guessing from the wording of either task.
    if memory.metadata.get("task_authority") == "session_user_request":
        requested = memory.current_user_request or memory.task
        if isinstance(requested, str) and requested.strip():
            return requested.strip()
    active = memory.active_environment_task()
    task = active.get("task") if isinstance(active, dict) else None
    return task.strip() if isinstance(task, str) and task.strip() else observation.task


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
