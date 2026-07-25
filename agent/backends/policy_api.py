"""Narrow API exposed to bounded generated Code-as-Policy snippets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from adapter.protocol import JsonDict
from agent.runtime.env_facade import RlinfEnvFacade, RlinfStepResult


@dataclass(slots=True)
class PolicyApiTrace:
    """Trace of env-facing calls made by generated policy code."""

    calls: list[JsonDict] = field(default_factory=list)

    def record(self, name: str, payload: JsonDict) -> None:
        self.calls.append({"name": name, "payload": payload})


class CodePolicyApi:
    """Controlled runtime object for bounded Code-as-Policy execution.

    Generated code should operate through this object instead of receiving a raw
    simulator environment. High-level helper methods can be added here later,
    but the final actuation boundary remains `step(action)` or
    `chunk_step(actions)`.
    """

    def __init__(self, env: RlinfEnvFacade) -> None:
        self.env = env
        self.trace = PolicyApiTrace()
        self.last_observation: Any = None
        self.last_step: RlinfStepResult | None = None

    def reset(self, **kwargs: Any) -> Any:
        self.trace.record("reset", {"kwargs": kwargs})
        self.last_observation = self.env.reset(**kwargs)
        return self.last_observation

    def observe(self) -> Any:
        self.trace.record("observe", {})
        return self.last_observation

    def step(self, action: Any, **kwargs: Any) -> RlinfStepResult:
        self.trace.record("step", {"action": action, "kwargs": kwargs})
        self.last_step = self.env.step(action, **kwargs)
        self.last_observation = self.last_step.observation
        return self.last_step

    def chunk_step(self, actions: Any, **kwargs: Any) -> RlinfStepResult:
        self.trace.record("chunk_step", {"actions": actions, "kwargs": kwargs})
        self.last_step = self.env.chunk_step(actions, **kwargs)
        self.last_observation = self.last_step.observation
        return self.last_step

    def descriptor(self) -> JsonDict:
        return {
            "name": type(self).__name__,
            "env": self.env.descriptor(),
            "calls": len(self.trace.calls),
        }
