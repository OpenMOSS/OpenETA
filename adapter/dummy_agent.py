"""Dummy code-agent adapter used to validate the OpenETA bridge."""

from __future__ import annotations

from adapter.agent import AgentAdapter
from adapter.protocol import EnvAction, EnvObservation, JsonDict


class DummyAgentAdapter(AgentAdapter):
    """Deterministic no-op agent with a tiny in-memory event log."""

    def __init__(self) -> None:
        self.task: str | None = None
        self.memory: list[JsonDict] = []

    def start_session(self, *, task: str, metadata: JsonDict | None = None) -> None:
        self.task = task
        self.memory.clear()
        if metadata:
            self.memory.append({"type": "session_metadata", "metadata": metadata})

    def act(self, observation: EnvObservation) -> EnvAction:
        self.memory.append({"type": "observation", "step": observation.metadata.get("step_idx")})
        return EnvAction(
            action_type="response",
            command={
                "schema_version": "openeta.agent_command.v1",
                "request": {
                    "kind": "response",
                    "name": "task_complete",
                    "parameters": {"message": "Dummy adapter completed its test turn."},
                    "reasoning": "The deterministic dummy adapter performs no world action.",
                },
                "status": "executed",
                "safety_checks": [],
                "tool_calls": [],
            },
            metadata={"source": "DummyAgentAdapter"},
        )

    def update_memory(self, event: JsonDict) -> None:
        self.memory.append(event)
