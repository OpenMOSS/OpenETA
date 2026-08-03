from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import gymnasium as gym
import pytest

from sim.env_config import build_libero_cfg
from sim.env_registry import _make_libero_direct
from sim.libero_runtime import get_libero_max_episode_steps


def test_libero_horizon_defaults_to_benchmark_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENETA_LIBERO_MAX_EPISODE_STEPS", raising=False)

    assert get_libero_max_episode_steps() == 5000
    assert build_libero_cfg("libero_spatial").max_episode_steps == 5000


def test_libero_horizon_rejects_invalid_values(monkeypatch: pytest.MonkeyPatch) -> None:
    for value in ("0", "-1", "many"):
        monkeypatch.setenv("OPENETA_LIBERO_MAX_EPISODE_STEPS", value)
        with pytest.raises(ValueError, match="must be a positive integer"):
            get_libero_max_episode_steps()


def test_direct_libero_env_receives_interactive_horizon(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    created: dict[str, object] = {}

    class FakeOffScreenRenderEnv:
        observation_space = gym.spaces.Dict({})

        def __init__(self, **kwargs) -> None:
            created.update(kwargs)

        def close(self) -> None:
            pass

    libero_package = ModuleType("libero")
    libero_package.__path__ = []  # type: ignore[attr-defined]
    libero_module = ModuleType("libero.libero")
    libero_module.__path__ = []  # type: ignore[attr-defined]
    libero_module.get_libero_path = lambda _name: str(tmp_path)  # type: ignore[attr-defined]
    envs_module = ModuleType("libero.libero.envs")
    envs_module.OffScreenRenderEnv = FakeOffScreenRenderEnv  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "libero", libero_package)
    monkeypatch.setitem(sys.modules, "libero.libero", libero_module)
    monkeypatch.setitem(sys.modules, "libero.libero.envs", envs_module)
    monkeypatch.setenv("OPENETA_LIBERO_MAX_EPISODE_STEPS", "5000")

    env = _make_libero_direct(
        SimpleNamespace(problem_folder="suite", bddl_file="task.bddl"),
    )

    assert created["horizon"] == 5000
    env.close()
