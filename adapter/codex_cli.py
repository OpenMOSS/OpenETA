"""Legacy Codex CLI adapter placeholder.

OpenETA now targets a lightweight Python agent runtime. The Codex submodule is
kept as a reference for now, but this adapter is not on the primary path.
"""

from __future__ import annotations

from adapter.agent import AgentAdapter
from adapter.protocol import EnvAction, EnvObservation, JsonDict


class CodexCliAgentAdapter(AgentAdapter):
    """Legacy placeholder for a Codex CLI-backed agent."""

    def __init__(self, *, codex_path: str = "agent/codex") -> None:
        self.codex_path = codex_path
        self.task: str | None = None

    def start_session(self, *, task: str, metadata: JsonDict | None = None) -> None:
        del metadata
        self.task = task

    def act(self, observation: EnvObservation) -> EnvAction:
        raise NotImplementedError(
            "The legacy Codex CLI adapter is not wired for execution. "
            "Use OpenEtaAgentAdapter for the primary agent workflow."
        )
