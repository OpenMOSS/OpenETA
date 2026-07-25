"""OpenETA facade over RLinf-backed environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from adapter.protocol import JsonDict


@dataclass(slots=True)
class RlinfEnvSpec:
    """Declarative RLinf environment construction request.

    `env_type` is resolved by `sim.envs.get_env_cls(env_type, env_cfg)`.
    The remaining fields mirror the constructor shape used by most RLinf env
    classes in this vendored subset.
    """

    env_type: str
    env_cfg: Any
    num_envs: int = 1
    seed_offset: int = 0
    total_num_processes: int = 1
    worker_info: Any | None = None
    constructor_kwargs: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class RlinfStepResult:
    """Normalized result from a single env interaction call."""

    observation: Any
    reward: Any = None
    terminated: Any = None
    truncated: Any = None
    info: Any = None

    def to_dict(self) -> JsonDict:
        return {
            "observation": self.observation,
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "info": self.info,
        }


class RlinfEnvFacade:
    """Narrow OpenETA control facade for bounded generated Code-as-Policy.

    This class is intentionally smaller than RLinf's raw environment classes.
    Generated policies should receive methods like these instead of direct
    access to arbitrary simulator objects.
    """

    def __init__(self, env: Any, *, spec: RlinfEnvSpec | None = None) -> None:
        self.env = env
        self.spec = spec

    @classmethod
    def from_spec(cls, spec: RlinfEnvSpec) -> "RlinfEnvFacade":
        """Construct a facade by resolving the RLinf env class lazily."""

        from sim.envs import get_env_cls

        env_cls = get_env_cls(spec.env_type, spec.env_cfg)
        env = _construct_env(env_cls, spec)
        return cls(env, spec=spec)

    def reset(self, **kwargs: Any) -> Any:
        """Reset the environment and return the raw RLinf observation payload."""

        return self.env.reset(**kwargs)

    def step(self, action: Any, **kwargs: Any) -> RlinfStepResult:
        """Execute one action and normalize Gym/Gymnasium-style return tuples."""

        return _normalize_step_result(self.env.step(action, **kwargs))

    def chunk_step(self, chunk_actions: Any, **kwargs: Any) -> RlinfStepResult:
        """Execute an action chunk when the backend supports chunk stepping."""

        if not hasattr(self.env, "chunk_step"):
            raise AttributeError(f"{type(self.env).__name__} does not support chunk_step")
        return _normalize_step_result(self.env.chunk_step(chunk_actions, **kwargs))

    def close(self) -> None:
        if hasattr(self.env, "close"):
            self.env.close()

    def descriptor(self) -> JsonDict:
        return {
            "name": type(self).__name__,
            "env_type": self.spec.env_type if self.spec else None,
            "env_class": type(self.env).__name__,
            "protocol": {
                "reset": hasattr(self.env, "reset"),
                "step": hasattr(self.env, "step"),
                "chunk_step": hasattr(self.env, "chunk_step"),
                "close": hasattr(self.env, "close"),
            },
        }


def _construct_env(env_cls: type, spec: RlinfEnvSpec) -> Any:
    kwargs = dict(spec.constructor_kwargs)
    kwargs.setdefault("cfg", spec.env_cfg)
    kwargs.setdefault("num_envs", spec.num_envs)
    kwargs.setdefault("seed_offset", spec.seed_offset)
    kwargs.setdefault("total_num_processes", spec.total_num_processes)
    if spec.worker_info is not None:
        kwargs.setdefault("worker_info", spec.worker_info)
    return env_cls(**kwargs)


def _normalize_step_result(result: Any) -> RlinfStepResult:
    if not isinstance(result, tuple):
        return RlinfStepResult(observation=result)
    if len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return RlinfStepResult(obs, reward, terminated, truncated, info)
    if len(result) == 4:
        obs, reward, done, info = result
        return RlinfStepResult(obs, reward, done, False, info)
    return RlinfStepResult(observation=result)
