"""Agent-side adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from adapter.protocol import EnvAction, EnvObservation, JsonDict


class AgentAdapter(ABC):
    """Common interface for OpenETA-owned and external agent runtimes."""

    @abstractmethod
    def start_session(self, *, task: str, metadata: JsonDict | None = None) -> None:
        """Start or reset an agent session."""

    @abstractmethod
    def act(self, observation: EnvObservation) -> EnvAction:
        """Produce the next simulator action from an observation."""

    def update_memory(self, event: JsonDict) -> None:
        """Persist an interaction event for future turns."""
        return None

    def close(self) -> None:
        """Release agent resources."""
        return None
