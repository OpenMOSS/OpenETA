"""Adapter for the lightweight OpenETA agent runtime."""

from __future__ import annotations

from adapter.agent import AgentAdapter
from adapter.protocol import EnvAction, EnvObservation, JsonDict
from agent.runtime.runtime import OpenEtaAgentRuntime


class OpenEtaAgentAdapter(AgentAdapter):
    """AgentAdapter backed by the lightweight Python OpenETA runtime."""

    def __init__(self, runtime: OpenEtaAgentRuntime | None = None) -> None:
        self.runtime = runtime or OpenEtaAgentRuntime()

    def start_session(self, *, task: str, metadata: JsonDict | None = None) -> None:
        self.runtime.start_session(task=task, metadata=metadata)

    def act(self, observation: EnvObservation) -> EnvAction:
        return self.runtime.act(observation)

    def update_memory(self, event: JsonDict) -> None:
        self.runtime.update_memory(event)
