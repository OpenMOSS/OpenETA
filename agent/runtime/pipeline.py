"""Safe/tool/skill command pipeline for OpenETA agent decisions."""

from __future__ import annotations

from adapter.protocol import EnvObservation, JsonDict
from agent.runtime.actions import (
    CommandKind,
    CommandPipelinePlan,
    CommandRequest,
    PipelineCall,
    PipelineStatus,
)
from agent.runtime.checkers import (
    CheckerSubagentConfig,
    build_failure_check_call,
    safety_check_parameters,
)
from agent.runtime.interfaces import ActionInterfaceRegistry, build_default_action_interfaces
from agent.runtime.memory import AgentMemory
from agent.runtime.planner import PlannerDecision
from agent.runtime.skills import SkillRegistry
from agent.tools.registry import ToolRegistry, ToolResult


class ActionPipeline:
    """Compile planner decisions into structured `EnvAction.command` payloads."""

    def __init__(
        self,
        *,
        execute_safe_checks: bool = True,
        checker_subagents: CheckerSubagentConfig | None = None,
        interfaces: ActionInterfaceRegistry | None = None,
    ) -> None:
        self.execute_safe_checks = execute_safe_checks
        self.checker_subagents = checker_subagents or CheckerSubagentConfig()
        self.interfaces = interfaces or build_default_action_interfaces()

    def compile(
        self,
        decision: PlannerDecision,
        *,
        observation: EnvObservation,
        tools: ToolRegistry,
        skills: SkillRegistry,
        memory: AgentMemory | None = None,
    ) -> CommandPipelinePlan:
        request = _decision_to_request(decision)

        if request.kind == CommandKind.TOOL_CALL:
            if _is_skill_call_request(request):
                return self._compile_skill_call(
                    request,
                    observation=observation,
                    tools=tools,
                    skills=skills,
                    planner_metadata=decision.metadata,
                )
            if _is_safety_check_request(request):
                safe_name = _named_tool_target(request)
                grasp_gate_error = (
                    memory.grasp_candidate_gate_error(
                        tool_name=safe_name,
                        parameters=request.parameters,
                    )
                    if memory is not None
                    else None
                )
                if grasp_gate_error:
                    safe_call = _skipped_tool_call(
                        safe_name,
                        request.parameters,
                        reason=grasp_gate_error,
                    )
                    return CommandPipelinePlan(
                        request=request,
                        status=PipelineStatus.BLOCKED,
                        safety_checks=[safe_call],
                        metadata={
                            "interface": self.interfaces.descriptor(
                                request.kind,
                                request.name,
                            ),
                            "planner_metadata": decision.metadata,
                            "grasp_candidate_gate": {
                                "blocked": True,
                                "reason": grasp_gate_error,
                            },
                        },
                    )
                safe_call = self._compile_safety_check(
                    safe_name,
                    request.parameters,
                    tools=tools,
                    observation=observation,
                    reason="Planner-requested safety check.",
                )
                return CommandPipelinePlan(
                    request=request,
                    status=_aggregate_status([safe_call]),
                    safety_checks=[safe_call],
                    metadata={
                        "interface": self.interfaces.descriptor(request.kind, request.name),
                        "planner_metadata": decision.metadata,
                    },
                )
            if _is_direct_tool_like_request(request):
                interface_descriptor = self.interfaces.descriptor(
                    request.kind,
                    request.name,
                )
                return CommandPipelinePlan(
                    request=request,
                    status=_direct_request_status(
                        request.kind,
                        request.name,
                        interface_descriptor,
                    ),
                    metadata={
                        "interface": interface_descriptor,
                        "observation_step": observation.metadata.get("step_idx"),
                        "planner_metadata": decision.metadata,
                    },
                )
            if _is_tool_batch_request(request):
                return self._compile_tool_batch(
                    request,
                    tools=tools,
                    memory=memory,
                    planner_metadata=decision.metadata,
                )

            selection_gate_error = _detection_selection_gate_error(
                request,
                tools=tools,
                memory=memory,
            )
            if selection_gate_error:
                tool_call = _skipped_tool_call(
                    request.name,
                    request.parameters,
                    reason=selection_gate_error,
                )
                return CommandPipelinePlan(
                    request=request,
                    status=PipelineStatus.BLOCKED,
                    tool_calls=[tool_call],
                    metadata={
                        "interface": self.interfaces.descriptor(request.kind, request.name),
                        "planner_metadata": decision.metadata,
                        "execution_rule": _tool_execution_rule(tool_call, tools),
                        "selection_gate": {
                            "blocked": True,
                            "reason": selection_gate_error,
                        },
                    },
                )

            grasp_gate_error = (
                memory.grasp_candidate_gate_error(
                    tool_name=request.name,
                    parameters=request.parameters,
                )
                if memory is not None
                else None
            )
            if grasp_gate_error:
                tool_call = _skipped_tool_call(
                    request.name,
                    request.parameters,
                    reason=grasp_gate_error,
                )
                return CommandPipelinePlan(
                    request=request,
                    status=PipelineStatus.BLOCKED,
                    tool_calls=[tool_call],
                    metadata={
                        "interface": self.interfaces.descriptor(request.kind, request.name),
                        "planner_metadata": decision.metadata,
                        "execution_rule": _tool_execution_rule(tool_call, tools),
                        "grasp_candidate_gate": {
                            "blocked": True,
                            "reason": grasp_gate_error,
                        },
                    },
                )

            safety_checks = self._compile_pre_safety_checks(
                request.name,
                request.parameters,
                tools=tools,
                observation=observation,
            )
            if safety_checks and not _checks_allow_tool_execution(safety_checks):
                tool_call = _skipped_tool_call(
                    request.name,
                    request.parameters,
                    reason="Tool call skipped because a pre-tool safety checker did not pass.",
                )
                return CommandPipelinePlan(
                    request=request,
                    status=PipelineStatus.BLOCKED,
                    safety_checks=safety_checks,
                    tool_calls=[tool_call],
                    metadata={
                        "interface": self.interfaces.descriptor(request.kind, request.name),
                        "planner_metadata": decision.metadata,
                        "execution_rule": _tool_execution_rule(tool_call, tools),
                        "checker_results": {
                            "pre_safety_checks": [
                                call.to_dict() for call in safety_checks
                            ],
                            "post_failure_checks": [],
                        },
                    },
                )

            tool_call = self._compile_tool_call(
                request.name,
                request.parameters,
                tools=tools,
                observation=observation,
                reason="Direct planner-requested tool call.",
            )
            post_failure_checks = self._compile_post_failure_checks(tool_call)
            return CommandPipelinePlan(
                request=request,
                status=_aggregate_status([*safety_checks, tool_call]),
                safety_checks=safety_checks,
                tool_calls=[tool_call],
                metadata={
                    "interface": self.interfaces.descriptor(request.kind, request.name),
                    "planner_metadata": decision.metadata,
                    "execution_rule": _tool_execution_rule(tool_call, tools),
                    "checker_results": {
                        "pre_safety_checks": [call.to_dict() for call in safety_checks],
                        "post_failure_checks": [
                            call.to_dict() for call in post_failure_checks
                        ],
                    },
                },
            )

        interface_descriptor = self.interfaces.descriptor(request.kind, request.name)
        return CommandPipelinePlan(
            request=request,
            status=_direct_request_status(request.kind, request.name, interface_descriptor),
            metadata={
                "interface": interface_descriptor,
                "observation_step": observation.metadata.get("step_idx"),
                "planner_metadata": decision.metadata,
            },
        )

    def _compile_skill_call(
        self,
        request: CommandRequest,
        *,
        observation: EnvObservation,
        tools: ToolRegistry,
        skills: SkillRegistry,
        planner_metadata: JsonDict,
    ) -> CommandPipelinePlan:
        skill_name = _skill_call_name(request)
        try:
            skill = skills.get(skill_name)
        except KeyError as exc:
            failed_call = PipelineCall(
                kind=CommandKind.TOOL_CALL,
                name=skill_name,
                parameters=request.parameters,
                status=PipelineStatus.FAILED,
                reason=str(exc),
            )
            return CommandPipelinePlan(
                request=request,
                status=PipelineStatus.FAILED,
                skill_call=failed_call,
            )

        del observation, tools
        skill_call = PipelineCall(
            kind=CommandKind.TOOL_CALL,
            name=skill.name,
            parameters={
                "requested_parameters": request.parameters,
                "task_patterns": list(skill.task_patterns),
                "allowed_tools": list(skill.allowed_tools),
            },
            status=PipelineStatus.PLANNED,
            result={
                "success": True,
                "content": skill.content,
                "details": {
                    "description": skill.description,
                    "source": skill.source,
                    "version": skill.version,
                    "editable": skill.editable,
                    "metadata": skill.metadata,
                },
            },
            reason=request.reasoning
            or "Skill guidance selected; planner must choose atomic tools explicitly.",
        )
        return CommandPipelinePlan(
            request=request,
            status=PipelineStatus.PLANNED,
            skill_call=skill_call,
            metadata={
                "interface": self.interfaces.descriptor(request.kind, request.name),
                "skill_description": skill.description,
                "planner_metadata": planner_metadata,
                "execution_rule": {
                    "mode": "skill_guidance_only",
                    "summary": (
                        "Skills are editable text guidance. The runtime does not "
                        "auto-expand them into hidden tool calls; the planner must "
                        "select each atomic tool in the closed-loop process."
                    ),
                },
            },
        )

    def _compile_safety_check(
        self,
        name: str,
        parameters: JsonDict,
        *,
        tools: ToolRegistry,
        observation: EnvObservation | None,
        reason: str,
    ) -> PipelineCall:
        if not self.execute_safe_checks:
            return PipelineCall(
                kind=CommandKind.TOOL_CALL,
                name=name,
                parameters=parameters,
                status=PipelineStatus.PENDING,
                reason=reason,
            )
        return self._compile_tool_call(
            name,
            parameters,
            tools=tools,
            observation=observation,
            kind=CommandKind.TOOL_CALL,
            reason=reason,
        )

    def _compile_pre_safety_checks(
        self,
        target_tool: str,
        target_parameters: JsonDict,
        *,
        tools: ToolRegistry,
        observation: EnvObservation | None,
    ) -> list[PipelineCall]:
        checker_tool = self.checker_subagents.safety_tool_for(target_tool)
        if not checker_tool:
            return []
        return [
            self._compile_safety_check(
                checker_tool,
                safety_check_parameters(
                    checker_tool=checker_tool,
                    target_tool=target_tool,
                    target_parameters=target_parameters,
                ),
                tools=tools,
                observation=observation,
                reason=f"Pre-tool safety checker for `{target_tool}`.",
            )
        ]

    def _compile_post_failure_checks(self, tool_call: PipelineCall) -> list[PipelineCall]:
        if not self.checker_subagents.should_run_failure_check(tool_call.name):
            return []
        if tool_call.status != PipelineStatus.FAILED:
            return []
        return [
            build_failure_check_call(
                checker_name=self.checker_subagents.failure_checker_name,
                target_call=tool_call,
            )
        ]

    def _compile_tool_call(
        self,
        name: str,
        parameters: JsonDict,
        *,
        tools: ToolRegistry,
        observation: EnvObservation | None = None,
        reason: str,
        kind: CommandKind = CommandKind.TOOL_CALL,
    ) -> PipelineCall:
        try:
            tools.get(name)
        except KeyError as exc:
            return PipelineCall(
                kind=kind,
                name=name,
                parameters=parameters,
                status=PipelineStatus.FAILED,
                reason=str(exc),
            )

        if not tools.can_execute(name):
            return PipelineCall(
                kind=kind,
                name=name,
                parameters=parameters,
                status=PipelineStatus.PENDING,
                reason=f"{reason} No handler registered yet.",
            )

        result = tools.call(
            name,
            parameters,
            observation=observation,
            metadata={"pipeline_kind": kind.value, "reason": reason},
        )
        return PipelineCall(
            kind=kind,
            name=name,
            parameters=parameters,
            status=PipelineStatus.EXECUTED if result.success else PipelineStatus.FAILED,
            result=_tool_result_to_dict(result),
            reason=reason,
        )

    def _compile_tool_batch(
        self,
        request: CommandRequest,
        *,
        tools: ToolRegistry,
        memory: AgentMemory | None,
        planner_metadata: JsonDict,
    ) -> CommandPipelinePlan:
        calls = _tool_batch_calls(request)
        compiled_calls: list[PipelineCall] = []
        blocked = False

        for call in calls:
            name = str(call.get("name", ""))
            parameters = call.get("parameters", {})
            if not isinstance(parameters, dict):
                parameters = {"value": parameters}
            reason = "Batched planner-requested tool call."
            try:
                spec = tools.get(name)
            except KeyError as exc:
                compiled_calls.append(
                    PipelineCall(
                        kind=CommandKind.TOOL_CALL,
                        name=name,
                        parameters=parameters,
                        status=PipelineStatus.FAILED,
                        reason=str(exc),
                    )
                )
                blocked = True
                continue

            selection_gate_error = (
                memory.detection_selection_gate_error(
                    tool_name=name,
                    parameters=parameters,
                    world_mutating=spec.effect.value == "world_mutating",
                )
                if memory is not None
                else None
            )
            if selection_gate_error:
                compiled_calls.append(
                    PipelineCall(
                        kind=CommandKind.TOOL_CALL,
                        name=name,
                        parameters=parameters,
                        status=PipelineStatus.BLOCKED,
                        reason=selection_gate_error,
                    )
                )
                blocked = True
                continue

            grasp_gate_error = (
                memory.grasp_candidate_gate_error(
                    tool_name=name,
                    parameters=parameters,
                )
                if memory is not None
                else None
            )
            if grasp_gate_error:
                compiled_calls.append(
                    PipelineCall(
                        kind=CommandKind.TOOL_CALL,
                        name=name,
                        parameters=parameters,
                        status=PipelineStatus.BLOCKED,
                        reason=grasp_gate_error,
                    )
                )
                blocked = True
                continue

            if not spec.allows_batched_observation:
                compiled_calls.append(
                    PipelineCall(
                        kind=CommandKind.TOOL_CALL,
                        name=name,
                        parameters=parameters,
                        status=PipelineStatus.BLOCKED,
                        reason=(
                            f"{reason} Tool effect `{spec.effect.value}` requires "
                            "a fresh observation before another tool is selected."
                        ),
                    )
                )
                blocked = True
                continue

            compiled_calls.append(
                self._compile_tool_call(
                    name,
                    parameters,
                    tools=tools,
                    observation=None,
                    reason=reason,
                )
            )

        status = PipelineStatus.BLOCKED if blocked else _aggregate_status(compiled_calls)
        return CommandPipelinePlan(
            request=request,
            status=status,
            tool_calls=compiled_calls,
            metadata={
                "interface": self.interfaces.descriptor(request.kind, request.name),
                "planner_metadata": planner_metadata,
                "execution_rule": {
                    "mode": "batched_read_only_tools",
                    "allowed_effects": ["read_only", "bookkeeping", "planning"],
                    "blocked_effects": ["world_mutating"],
                    "requires_observation_after_batch": False,
                },
            },
        )


def _decision_to_request(decision: PlannerDecision) -> CommandRequest:
    kind = _normalize_command_kind(decision.action_type, decision.skill)
    name = decision.action
    parameters = dict(decision.parameters)
    return CommandRequest(
        kind=kind,
        name=name,
        parameters=parameters,
        reasoning=decision.reasoning,
        code=decision.code,
    )


def _normalize_command_kind(action_type: str, skill: str | None) -> CommandKind:
    del skill
    normalized = action_type.lower().strip()
    if normalized == "response":
        return CommandKind.RESPONSE
    if normalized != "tool_call":
        raise ValueError(f"Unsupported command kind: {action_type!r}")
    return CommandKind.TOOL_CALL


def _is_tool_batch_request(request: CommandRequest) -> bool:
    return request.name in {"batch", "tool_batch"} or isinstance(
        request.parameters.get("calls"), list
    )


def _is_skill_call_request(request: CommandRequest) -> bool:
    return request.name == "skill_call"


def _is_safety_check_request(request: CommandRequest) -> bool:
    return request.name == "safe_check"


def _is_direct_tool_like_request(request: CommandRequest) -> bool:
    return request.name in {"sense", "code_policy"}


def _skill_call_name(request: CommandRequest) -> str:
    for key in ("skill", "name"):
        value = request.parameters.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return request.name


def _named_tool_target(request: CommandRequest) -> str:
    for key in ("tool", "target", "name"):
        value = request.parameters.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return request.name


def _tool_batch_calls(request: CommandRequest) -> list[JsonDict]:
    calls = request.parameters.get("calls", [])
    if not isinstance(calls, list):
        return []
    return [call for call in calls if isinstance(call, dict)]


def _tool_execution_rule(call: PipelineCall, tools: ToolRegistry) -> JsonDict:
    try:
        spec = tools.get(call.name)
    except KeyError:
        return {"mode": "unknown_tool"}
    return {
        "mode": "single_tool_closed_loop",
        "effect": spec.effect.value,
        "batchable": spec.allows_batched_observation,
        "requires_observation_after_call": spec.requires_observation_after_call,
    }


def _tool_result_to_dict(result: ToolResult) -> JsonDict:
    return {
        "success": result.success,
        "content": result.content,
        "details": result.details,
    }


def _checks_allow_tool_execution(calls: list[PipelineCall]) -> bool:
    for call in calls:
        if call.status != PipelineStatus.EXECUTED:
            return False
        if isinstance(call.result, dict) and call.result.get("success") is False:
            return False
    return True


def _detection_selection_gate_error(
    request: CommandRequest,
    *,
    tools: ToolRegistry,
    memory: AgentMemory | None,
) -> str | None:
    if memory is None:
        return None
    try:
        spec = tools.get(request.name)
    except KeyError:
        return None
    return memory.detection_selection_gate_error(
        tool_name=request.name,
        parameters=request.parameters,
        world_mutating=spec.effect.value == "world_mutating",
    )


def _skipped_tool_call(name: str, parameters: JsonDict, *, reason: str) -> PipelineCall:
    return PipelineCall(
        kind=CommandKind.TOOL_CALL,
        name=name,
        parameters=parameters,
        status=PipelineStatus.SKIPPED,
        reason=reason,
    )


def _aggregate_status(calls: list[PipelineCall]) -> PipelineStatus:
    if any(call.status == PipelineStatus.FAILED for call in calls):
        return PipelineStatus.FAILED
    if any(call.status == PipelineStatus.BLOCKED for call in calls):
        return PipelineStatus.BLOCKED
    if any(call.status == PipelineStatus.PENDING for call in calls):
        return PipelineStatus.PENDING
    if calls and all(call.status == PipelineStatus.EXECUTED for call in calls):
        return PipelineStatus.EXECUTED
    return PipelineStatus.PLANNED


def _direct_request_status(
    kind: CommandKind,
    name: str,
    interface_descriptor: JsonDict,
) -> PipelineStatus:
    if kind == CommandKind.RESPONSE and name in {"talk", "task_complete"}:
        return PipelineStatus.EXECUTED
    if not interface_descriptor.get("implemented", False):
        return PipelineStatus.PENDING
    return PipelineStatus.PLANNED
