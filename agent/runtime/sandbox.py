"""Code-policy execution sandbox boundary for RLinf simulator environments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from adapter.protocol import JsonDict
from agent.runtime.actions import PipelineStatus
from agent.runtime.env_facade import RlinfEnvFacade
from agent.backends.policy_api import CodePolicyApi


DEFAULT_SIM_ROOT = Path(__file__).resolve().parents[2] / "sim"
DEFAULT_RLINF_ENVS_MODULE = "sim.envs"
DEFAULT_RLINF_ENV_RESOLVER = "get_env_cls"
DEFAULT_RLINF_WRAPPERS_MODULE: str | None = None


@dataclass(slots=True)
class RlinfSandboxConfig:
    """Configuration for the simulator-side code-policy sandbox."""

    sim_root: Path = DEFAULT_SIM_ROOT
    envs_module: str = DEFAULT_RLINF_ENVS_MODULE
    env_class_resolver: str = DEFAULT_RLINF_ENV_RESOLVER
    wrappers_module: str | None = DEFAULT_RLINF_WRAPPERS_MODULE
    use_collect_episode: bool = False
    use_record_video: bool = False
    output_dir: Path | None = None
    metadata: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class CodePolicyExecutionRequest:
    """Request to execute generated policy code against an environment."""

    code: str
    policy_context: JsonDict
    env: RlinfEnvFacade | Any | None = None
    api: CodePolicyApi | None = None
    metadata: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class CodePolicyExecutionResult:
    """Result returned by the code-policy sandbox."""

    status: PipelineStatus
    content: str = ""
    details: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        return {
            "status": self.status.value,
            "content": self.content,
            "details": self.details,
        }


class RlinfCodePolicySandbox:
    """Simulator-side boundary for executing generated policy code.

    The intended sandbox is the RLinf-backed environment tree under `sim/`.
    The local environment registry is `sim.envs.get_env_cls`. This repository
    currently has no shared wrapper package; instrumentation must be provided
    explicitly by a simulator adapter. Real execution should
    happen through an explicit simulator adapter that provides the env instance
    and allowed helper APIs.
    """

    def __init__(self, config: RlinfSandboxConfig | None = None) -> None:
        self.config = config or RlinfSandboxConfig()

    def descriptor(self) -> JsonDict:
        return {
            "name": type(self).__name__,
            "sim_root": str(self.config.sim_root),
            "envs_module": self.config.envs_module,
            "env_class_resolver": self.config.env_class_resolver,
            "runtime_protocol": ["reset", "step", "chunk_step", "close"],
            "wrappers_module": self.config.wrappers_module,
            "wrappers_role": (
                "optional adapter-provided instrumentation"
                if self.config.wrappers_module
                else "unavailable in the current repository"
            ),
            "use_collect_episode": self.config.use_collect_episode,
            "use_record_video": self.config.use_record_video,
            "output_dir": str(self.config.output_dir) if self.config.output_dir else None,
            "implemented": False,
        }

    def wrap_env(self, env: Any, *, video_cfg: Any | None = None) -> Any:
        """Return the env unchanged unless unavailable wrapper flags are requested.

        Recording instrumentation belongs to the simulator adapter until a
        shared local wrapper contract is added.
        """

        if self.config.use_collect_episode or self.config.use_record_video:
            raise RuntimeError(
                "Shared rollout/video wrappers are not available in the current "
                "sim.envs tree; provide instrumentation through the simulator adapter."
            )
        del video_cfg
        return env

    def execute(self, request: CodePolicyExecutionRequest) -> CodePolicyExecutionResult:
        """Reserve the execution hook for generated code.

        Actual code execution needs a narrower API surface than raw Python
        `exec`: allowed env methods, timeout/resource limits, safety preflight,
        and trajectory logging must be defined first.
        """

        api = request.api
        if api is None and isinstance(request.env, RlinfEnvFacade):
            api = CodePolicyApi(request.env)

        return CodePolicyExecutionResult(
            status=PipelineStatus.PENDING,
            content="RLinf code-policy sandbox interface is reserved but not executable yet.",
            details={
                "sandbox": self.descriptor(),
                "code_size": len(request.code),
                "policy_api": api.descriptor() if api is not None else None,
                "env_facade": (
                    request.env.descriptor()
                    if isinstance(request.env, RlinfEnvFacade)
                    else None
                ),
                "policy_context_keys": sorted(request.policy_context.keys()),
            },
        )
