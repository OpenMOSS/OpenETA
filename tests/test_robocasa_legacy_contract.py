from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

# The legacy RLinf vector path intentionally lives in the RoboCasa training
# environment, where these optional dependencies are installed.  Keep the base
# OpenETA test environment lightweight while still running the contract tests
# in that supported environment.
pytest.importorskip("torch")
pytest.importorskip("omegaconf")

from sim.envs.robocasa.robocasa_env import RobocasaEnv
from sim.envs.robocasa.utils import get_image_space


def _legacy_env(*, num_envs: int = 4) -> RobocasaEnv:
    env = object.__new__(RobocasaEnv)
    env.num_envs = num_envs
    env.current_raw_obs = None
    env.current_info_list = None
    env._is_start = False
    return env


def test_default_image_space_is_the_three_camera_contract() -> None:
    assert get_image_space("default") == [
        "observation/image",
        "observation/wrist_image",
        "observation/extra_view_image",
    ]
    with pytest.raises(ValueError, match="Unknown RoboCasa image_space"):
        get_image_space("missing")


def test_group_members_repeat_task_and_scenario_seed() -> None:
    env = _legacy_env(num_envs=8)
    env.group_size = 2
    env.num_group = 4
    env.num_tasks = 3
    env.seed_offset = 1
    env.cfg = SimpleNamespace(seed=100)

    env._init_reset_state_ids()
    env._init_task_ids()

    np.testing.assert_array_equal(
        env.env_seeds,
        np.array([104, 104, 105, 105, 106, 106, 107, 107]),
    )
    np.testing.assert_array_equal(
        env.task_ids,
        np.array([0, 0, 1, 1, 2, 2, 0, 0]),
    )


def test_spawned_factory_imports_robocasa_and_uses_captured_camera_contract(
    monkeypatch,
) -> None:
    calls = []
    fake_robocasa = ModuleType("robocasa")
    fake_robosuite = ModuleType("robosuite")
    fake_controllers = ModuleType("robosuite.controllers")

    def load_controller(*, controller, robot):
        return {"controller": controller, "robot": robot}

    def make(**kwargs):
        assert "robocasa" in sys.modules
        calls.append(kwargs)
        return {"created": True}

    fake_controllers.load_composite_controller_config = load_controller
    fake_robosuite.make = make
    monkeypatch.setitem(sys.modules, "robocasa", fake_robocasa)
    monkeypatch.setitem(sys.modules, "robosuite", fake_robosuite)
    monkeypatch.setitem(sys.modules, "robosuite.controllers", fake_controllers)

    env = _legacy_env(num_envs=1)
    env.task_ids = np.array([0])
    env.task_names = ["PnPCounterToCab"]
    env.env_seeds = np.array([41])
    env.cfg = SimpleNamespace(
        image_space="default",
        robot_name="PandaOmron",
        init_params=SimpleNamespace(camera_widths=128, camera_heights=96),
    )

    created = env.get_env_fns()[0]()

    assert created == {"created": True}
    assert calls[0]["env_name"] == "PnPCounterToCab"
    assert calls[0]["camera_names"] == [
        "robot0_agentview_left",
        "robot0_eye_in_hand",
        "robot0_agentview_right",
    ]


def test_first_partial_reset_fails_fast_instead_of_miswrapping_batch() -> None:
    env = _legacy_env()
    with pytest.raises(RuntimeError, match="first RoboCasa legacy vector reset"):
        env._merge_reset_batch(
            np.array([1, 3]),
            [{"value": 10}, {"value": 30}],
            [{"lang": "ten"}, {"lang": "thirty"}],
        )


def test_partial_reset_merges_into_full_cached_batch() -> None:
    env = _legacy_env()
    env.current_raw_obs = [{"value": index} for index in range(env.num_envs)]
    env.current_info_list = [{"lang": str(index)} for index in range(env.num_envs)]

    indices = env._merge_reset_batch(
        np.array([1, 3]),
        [{"value": 10}, {"value": 30}],
        [{"lang": "ten"}, {"lang": "thirty"}],
    )

    np.testing.assert_array_equal(indices, np.array([1, 3]))
    assert [item["value"] for item in env.current_raw_obs] == [0, 10, 2, 30]
    assert [item["lang"] for item in env.current_info_list] == [
        "0",
        "ten",
        "2",
        "thirty",
    ]


def test_partial_reset_returns_a_full_observation_batch(monkeypatch) -> None:
    class FakeVectorEnv:
        def reset(self, *, id):
            return (
                [{"value": 100 + int(index)} for index in id],
                [{"lang": f"reset-{int(index)}"} for index in id],
            )

    env = _legacy_env()
    env.env = FakeVectorEnv()
    env.current_raw_obs = [{"value": index} for index in range(env.num_envs)]
    env.current_info_list = [{"lang": str(index)} for index in range(env.num_envs)]
    monkeypatch.setattr(
        env,
        "_wrap_obs",
        lambda raw, _info: {"values": [item["value"] for item in raw]},
    )
    monkeypatch.setattr(env, "_reset_metrics", lambda _indices: None)

    observation, info = env.reset(env_idx=[1, 3])

    assert observation == {"values": [0, 101, 2, 103]}
    assert info == {}


def test_step_and_chunk_reject_unbatched_or_empty_actions() -> None:
    env = _legacy_env()
    with pytest.raises(ValueError, match="vector actions"):
        env.step(np.zeros(12, dtype=np.float32))
    with pytest.raises(ValueError, match="chunk_actions"):
        env.chunk_step(np.zeros((env.num_envs, 0, 12), dtype=np.float32))
    with pytest.raises(ValueError, match="chunk_actions"):
        env.chunk_step(np.zeros((1, 2, 12), dtype=np.float32))


def test_metrics_count_only_the_first_success_step() -> None:
    env = _legacy_env(num_envs=2)
    env.prev_step_reward = np.zeros(2)
    env._elapsed_steps = np.array([1, 1], dtype=np.int32)
    env._init_metrics()

    env._record_metrics(
        np.array([1.0, 0.0]),
        np.array([True, False]),
        {},
    )
    env._elapsed_steps += 1
    env._record_metrics(
        np.array([1.0, 0.0]),
        np.array([True, False]),
        {},
    )

    np.testing.assert_array_equal(env.returns, np.array([1.0, 0.0]))
    np.testing.assert_array_equal(env.success_episode_len, np.array([1, 0]))
