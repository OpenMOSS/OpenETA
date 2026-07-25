"""Simulator-side adapter interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from adapter.protocol import EnvAction, EnvObservation, StepResult


class SimulatorAdapter(ABC):
    """Common interface for RLinf-backed and dummy simulators."""

    @abstractmethod
    def reset(self, *, task: str | None = None, seed: int | None = None) -> EnvObservation:
        """Reset the simulator and return the first observation."""

    @abstractmethod
    def observe(self) -> EnvObservation:
        """Return the latest simulator observation without stepping."""

    @abstractmethod
    def step(self, action: EnvAction) -> StepResult:
        """Apply an agent action and return the environment result."""

    def close(self) -> None:
        """Release simulator resources."""
        return None

