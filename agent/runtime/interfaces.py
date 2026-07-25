"""Reserved execution interfaces for OpenETA command subtypes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from adapter.protocol import EnvObservation, JsonDict
from agent.runtime.actions import CommandKind, CommandRequest, PipelineStatus
from agent.runtime.sandbox import (
    CodePolicyExecutionRequest,
    RlinfCodePolicySandbox,
)


@dataclass(slots=True)
class ActionExecutionContext:
    """Context passed to an action interface implementation."""

    request: CommandRequest
    observation: EnvObservation
    memory_context: JsonDict = field(default_factory=dict)
    metadata: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class ActionExecutionResult:
    """Result returned by future action interface implementations."""

    status: PipelineStatus
    content: str = ""
    details: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "status": self.status.value,
            "content": self.content,
            "details": self.details,
        }


class ActionInterface(ABC):
    """Base interface for executing one command subtype.

    Current implementations are placeholders. Real implementations can later be
    registered without changing planner or simulator bridge code.
    """

    kind: CommandKind
    name: str
    description: str

    @abstractmethod
    def execute(self, context: ActionExecutionContext) -> ActionExecutionResult:
        """Execute, simulate, or delegate an action request."""

    def descriptor(self) -> JsonDict:
        return {
            "kind": self.kind.value,
            "name": self.name,
            "description": self.description,
            "implemented": not isinstance(self, PendingActionInterface),
        }


class PendingActionInterface(ActionInterface):
    """Placeholder interface for action kinds that are not implemented yet."""

    def execute(self, context: ActionExecutionContext) -> ActionExecutionResult:
        return ActionExecutionResult(
            status=PipelineStatus.PENDING,
            content=f"Interface `{self.name}` is reserved but not implemented.",
            details={"request": context.request.to_dict()},
        )


class SkillCallInterface(PendingActionInterface):
    kind = CommandKind.TOOL_CALL
    name = "skill_call"
    description = "Select or inspect a text-guidance skill; does not execute tools."


class ToolCallInterface(PendingActionInterface):
    kind = CommandKind.TOOL_CALL
    name = "tool_call"
    description = "Execute a perception, manipulation, navigation, or control tool."


class SafeCheckInterface(PendingActionInterface):
    kind = CommandKind.TOOL_CALL
    name = "safe_check"
    description = "Run a safety preflight check before action execution."


class CodePolicyInterface(ActionInterface):
    kind = CommandKind.TOOL_CALL
    name = "code_policy"
    description = "Execute or evaluate bounded generated Code-as-Policy snippets."

    def __init__(self, sandbox: RlinfCodePolicySandbox | None = None) -> None:
        self.sandbox = sandbox or RlinfCodePolicySandbox()

    def execute(self, context: ActionExecutionContext) -> ActionExecutionResult:
        result = self.sandbox.execute(
            CodePolicyExecutionRequest(
                code=context.request.code or "",
                policy_context=context.request.parameters.get("policy_context", {}),
                metadata=context.metadata,
            )
        )
        return ActionExecutionResult(
            status=result.status,
            content=result.content,
            details=result.details,
        )

    def descriptor(self) -> JsonDict:
        sandbox_descriptor = self.sandbox.descriptor()
        return {
            "kind": self.kind.value,
            "name": self.name,
            "description": self.description,
            "implemented": bool(sandbox_descriptor.get("implemented", False)),
            "sandbox": sandbox_descriptor,
        }


class AskHumanInterface(PendingActionInterface):
    kind = CommandKind.RESPONSE
    name = "ask_human"
    description = "Ask a human/operator for missing task information."


class SenseInterface(PendingActionInterface):
    kind = CommandKind.TOOL_CALL
    name = "sense"
    description = "Request or acquire a fresh environment observation."


class TalkInterface(PendingActionInterface):
    kind = CommandKind.RESPONSE
    name = "talk"
    description = "Send a spoken or textual status message."


class TaskCompleteInterface(ActionInterface):
    kind = CommandKind.RESPONSE
    name = "task_complete"
    description = "End the current episode because the task is complete."

    def execute(self, context: ActionExecutionContext) -> ActionExecutionResult:
        return ActionExecutionResult(
            status=PipelineStatus.EXECUTED,
            content="Task completion acknowledged.",
            details={"request": context.request.to_dict()},
        )


class ActionInterfaceRegistry:
    """Registry for command subtype interfaces."""

    def __init__(self) -> None:
        self._interfaces: dict[str, ActionInterface] = {}

    def register(self, interface: ActionInterface) -> None:
        self._interfaces[interface.name] = interface

    def get(self, kind: CommandKind, name: str | None = None) -> ActionInterface:
        if name:
            interface = self._interfaces.get(name)
            if interface is not None:
                return interface
        if kind == CommandKind.RESPONSE:
            raise KeyError(f"No command interface registered for {kind.value}::{name}")
        fallback_name = "tool_call"
        try:
            return self._interfaces[fallback_name]
        except KeyError as exc:
            raise KeyError(
                f"No command interface registered for {kind.value}::{name or fallback_name}"
            ) from exc

    def descriptor(self, kind: CommandKind, name: str | None = None) -> JsonDict:
        return self.get(kind, name).descriptor()

    def list(self) -> list[ActionInterface]:
        return list(self._interfaces.values())


def build_default_action_interfaces() -> ActionInterfaceRegistry:
    """Build placeholder interfaces for current command subtypes."""

    registry = ActionInterfaceRegistry()
    for interface in [
        SkillCallInterface(),
        ToolCallInterface(),
        SafeCheckInterface(),
        CodePolicyInterface(),
        AskHumanInterface(),
        SenseInterface(),
        TalkInterface(),
        TaskCompleteInterface(),
    ]:
        registry.register(interface)
    return registry
