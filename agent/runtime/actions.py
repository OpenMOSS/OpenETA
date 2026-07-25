"""Structured command schema for OpenETA agent decisions.

`CommandKind` is intentionally small: an agent turn either calls one tool-like
capability or returns a response to the human/session. More specific operations
such as skill lookup, safety checks, code policy, sensing, ask-human, and talk
are represented by `CommandRequest.name`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from adapter.protocol import EnvAction, JsonDict


COMMAND_SCHEMA_VERSION = "openeta.agent_command.v1"


class CommandKind(str, Enum):
    """Top-level agent command categories exchanged through `EnvAction.command`."""

    TOOL_CALL = "tool_call"
    RESPONSE = "response"


class PipelineStatus(str, Enum):
    """Pipeline compilation or execution status."""

    PLANNED = "planned"
    PENDING = "pending"
    EXECUTED = "executed"
    BLOCKED = "blocked"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(slots=True)
class CommandRequest:
    """Planner-selected command before simulator execution."""

    kind: CommandKind
    name: str
    parameters: JsonDict = field(default_factory=dict)
    reasoning: str = ""
    code: str | None = None

    def to_dict(self) -> JsonDict:
        payload: JsonDict = {
            "kind": self.kind.value,
            "name": self.name,
            "parameters": self.parameters,
        }
        if self.reasoning:
            payload["reasoning"] = self.reasoning
        if self.code is not None:
            payload["code"] = self.code
        return payload


@dataclass(slots=True)
class PipelineCall:
    """A planned or executed tool/safety/skill call in the command pipeline."""

    kind: CommandKind
    name: str
    parameters: JsonDict = field(default_factory=dict)
    status: PipelineStatus = PipelineStatus.PLANNED
    result: JsonDict | None = None
    reason: str = ""

    def to_dict(self) -> JsonDict:
        payload: JsonDict = {
            "kind": self.kind.value,
            "name": self.name,
            "parameters": self.parameters,
            "status": self.status.value,
        }
        if self.result is not None:
            payload["result"] = self.result
        if self.reason:
            payload["reason"] = self.reason
        return payload


@dataclass(slots=True)
class CommandPipelinePlan:
    """Compiled command plan sent to the simulator adapter."""

    request: CommandRequest
    status: PipelineStatus
    safety_checks: list[PipelineCall] = field(default_factory=list)
    tool_calls: list[PipelineCall] = field(default_factory=list)
    skill_call: PipelineCall | None = None
    metadata: JsonDict = field(default_factory=dict)

    def to_command(self) -> JsonDict:
        command: JsonDict = {
            "schema_version": COMMAND_SCHEMA_VERSION,
            "request": self.request.to_dict(),
            "status": self.status.value,
            "safety_checks": [call.to_dict() for call in self.safety_checks],
            "tool_calls": [call.to_dict() for call in self.tool_calls],
            "metadata": self.metadata,
        }
        if self.skill_call is not None:
            command["skill_call"] = self.skill_call.to_dict()
        return command

    def to_env_action(self) -> EnvAction:
        return EnvAction(
            action_type=self.request.kind.value,
            code=self.request.code,
            command=self.to_command(),
            metadata={
                "source": "OpenEtaAgentRuntime",
                "schema_version": COMMAND_SCHEMA_VERSION,
                "pipeline_status": self.status.value,
            },
        )
