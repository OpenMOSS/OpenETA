from __future__ import annotations

import pytest

import sim.envs
from agent.runtime.env_facade import RlinfEnvFacade, RlinfEnvSpec
from agent.runtime.sandbox import RlinfCodePolicySandbox, RlinfSandboxConfig


def test_env_facade_resolves_classes_from_sim_envs(monkeypatch) -> None:
    captured = {}

    class FakeEnv:
        def __init__(self, **kwargs) -> None:
            captured.update(kwargs)

        def reset(self, **kwargs):
            return {"reset": kwargs}

    monkeypatch.setattr(sim.envs, "get_env_cls", lambda env_type, env_cfg: FakeEnv)
    spec = RlinfEnvSpec(env_type="libero", env_cfg={"task": 0}, num_envs=2)

    facade = RlinfEnvFacade.from_spec(spec)

    assert facade.reset(seed=7) == {"reset": {"seed": 7}}
    assert captured["cfg"] == {"task": 0}
    assert captured["num_envs"] == 2


def test_code_policy_sandbox_reports_current_env_registry() -> None:
    descriptor = RlinfCodePolicySandbox().descriptor()

    assert descriptor["envs_module"] == "sim.envs"
    assert descriptor["wrappers_module"] is None
    assert descriptor["wrappers_role"] == "unavailable in the current repository"


def test_code_policy_sandbox_rejects_unavailable_shared_wrappers() -> None:
    sandbox = RlinfCodePolicySandbox(
        RlinfSandboxConfig(use_record_video=True),
    )

    with pytest.raises(RuntimeError, match="not available"):
        sandbox.wrap_env(object(), video_cfg={})
