"""Pure-CPU contract tests for the distributed BEHAVIOR adapter."""

from __future__ import annotations

import random
import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from sim.env_config import build_behavior_cfg
from sim.envs.behavior.behavior_env import (
    BehaviorEnv,
    BehaviorProcessPool,
    _physical_worker_index,
    _reset_vector_rows,
    _validate_reset_isolation,
)
from sim.envs.behavior.instance_loader import (
    ActivityInstanceFile,
    ActivityInstanceLoader,
)
from sim.envs.behavior.seeding import derive_behavior_seed


class _ScriptedPool:
    activity_name = "turning_on_radio"

    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.steps: list[tuple] = []
        self.reset_calls: list[tuple[list[int], list[int]]] = []
        self.action_devices: list[torch.device] = []
        self._reset_generation = 0

    def env_reset_slice(
        self,
        _global_start,
        num_envs,
        env_indices=None,
        reset_seeds=None,
    ):
        selected = list(range(num_envs)) if env_indices is None else list(env_indices)
        seeds = [] if reset_seeds is None else list(reset_seeds)
        self.reset_calls.append((selected, seeds))
        self._reset_generation += 1
        observations = [
            {"value": torch.tensor(self._reset_generation * 100 + idx)}
            for idx in selected
        ]
        infos = [{"reset_seed": seed} for seed in seeds]
        return observations, infos

    def env_chunk_step_slice(
        self,
        _global_start,
        slice_num_envs,
        chunk_actions,
    ):
        assert slice_num_envs == self.num_envs
        self.action_devices.append(chunk_actions.device)
        chunk_size = chunk_actions.shape[1]
        scripted = self.steps[:chunk_size]
        del self.steps[:chunk_size]
        assert len(scripted) == chunk_size
        return tuple([step[field] for step in scripted] for field in range(5))


class _ResetChild:
    def __init__(self, index: int):
        self.index = index
        self.calls = 0
        self.rng_samples: list[tuple[float, float, float]] = []

    def reset(self, *, get_obs: bool):
        self.calls += 1
        self.rng_samples.append(
            (
                random.random(),
                float(np.random.random()),
                float(torch.rand(()).item()),
            )
        )
        if not get_obs:
            return None
        return {"row": self.index}, {"reset": self.index}


def _make_env(
    monkeypatch,
    *,
    num_envs=2,
    auto_reset=False,
    ignore_terminations=False,
    record_metrics=True,
):
    pool = _ScriptedPool(num_envs)
    monkeypatch.setattr(
        BehaviorProcessPool,
        "acquire_shared",
        classmethod(lambda cls, *_args, **_kwargs: (pool, 0)),
    )
    monkeypatch.setattr(
        BehaviorProcessPool,
        "release_shared",
        classmethod(lambda cls, *_args: None),
    )
    monkeypatch.setattr(
        BehaviorEnv,
        "_load_tasks_cfg",
        lambda self, _activity_name: setattr(self, "task_description", "test task"),
    )

    def _wrap_obs(self, raw_obs):
        return {
            "states": torch.stack(
                [torch.as_tensor(obs["value"]) for obs in raw_obs]
            ).cpu(),
            "task_descriptions": [self.task_description] * self.num_envs,
        }

    monkeypatch.setattr(BehaviorEnv, "_wrap_obs", _wrap_obs)
    cfg = OmegaConf.create(
        {
            "seed": 10,
            "reward_coef": 1.0,
            "ignore_terminations": ignore_terminations,
            "use_rel_reward": False,
            "auto_reset": auto_reset,
            "max_episode_steps": 50,
            "use_fixed_reset_state_ids": False,
            "group_size": 1,
            "video_cfg": {},
        }
    )
    env = BehaviorEnv(
        cfg=cfg,
        num_envs=num_envs,
        seed_offset=3,
        total_num_processes=1,
        worker_info=None,
        record_metrics=record_metrics,
    )
    env.reset()
    return env, pool


def _step(
    values,
    rewards,
    terminations,
    truncations,
    infos,
):
    return (
        [{"value": torch.tensor(value)} for value in values],
        torch.tensor(rewards),
        torch.tensor(terminations),
        torch.tensor(truncations),
        infos,
    )


def test_behavior_step_metrics_disabled_preserves_info_and_cpu(monkeypatch):
    env, pool = _make_env(monkeypatch, record_metrics=False)
    pool.steps.append(
        _step(
            [1, 2],
            [0.5, 1.0],
            [False, False],
            [False, False],
            [{"native": "a"}, {"native": "b"}],
        )
    )

    obs, rewards, terminated, truncated, infos = env.step(
        torch.zeros((2, 4))
    )

    assert obs["states"].tolist() == [1, 2]
    assert infos["native"] == ["a", "b"]
    assert "episode" not in infos
    assert env.elapsed_steps.tolist() == [1, 1]
    assert pool.action_devices == [torch.device("cpu")]
    assert all(
        tensor.device.type == "cpu"
        for tensor in (rewards, terminated, truncated, env.elapsed_steps)
    )


def test_pinned_omnigibson_partial_reset_targets_only_selected_children():
    children = [_ResetChild(index) for index in range(3)]
    vector_env = SimpleNamespace(envs=children)

    observations, infos = _reset_vector_rows(
        vector_env,
        reset_indices=[2, 0],
        reset_seeds=[17, 29],
        get_obs=True,
        supports_env_indices=False,
    )

    assert observations == [{"row": 2}, {"row": 0}]
    assert infos == [{"reset": 2}, {"reset": 0}]
    assert [child.calls for child in children] == [1, 0, 1]

    _reset_vector_rows(
        vector_env,
        reset_indices=[2, 0],
        reset_seeds=[17, 29],
        get_obs=True,
        supports_env_indices=False,
    )
    assert children[2].rng_samples[0] == children[2].rng_samples[1]
    assert children[0].rng_samples[0] == children[0].rng_samples[1]


def test_explicit_partial_seed_does_not_rebase_unselected_rows():
    env = object.__new__(BehaviorEnv)
    env.num_envs = 3
    env.seed_offset = 5
    env.seed = 105
    env.pool_offset = 0
    env._reset_counts = torch.zeros(3, dtype=torch.int64)
    env._reset_seed_bases = torch.tensor([100, 100, 100], dtype=torch.int64)

    initial = env._next_reset_seeds([0, 1, 2])
    assert len(set(initial)) == 3
    explicit = env._next_reset_seeds([1], seed=7)
    assert explicit == [
        derive_behavior_seed(
            7,
            worker_seed_offset=5,
            row_index=1,
            reset_count=0,
            stream=2,
        )
    ]
    assert env.seed == 105
    assert env._next_reset_seeds([0, 2]) == [
        derive_behavior_seed(
            100,
            worker_seed_offset=5,
            row_index=idx,
            reset_count=1,
            stream=2,
        )
        for idx in (0, 2)
    ]
    assert env._next_reset_seeds([1]) == [
        derive_behavior_seed(
            7,
            worker_seed_offset=5,
            row_index=1,
            reset_count=1,
            stream=2,
        )
    ]


def test_behavior_distributed_seed_identity_separates_workers_and_stages():
    seeds = {
        derive_behavior_seed(
            100,
            worker_seed_offset=worker,
            row_index=row,
            reset_count=reset_count,
            stream=2,
        )
        for worker in range(3)
        for row in range(4)
        for reset_count in range(3)
    }

    assert len(seeds) == 36


def test_behavior_pool_identity_groups_pipeline_stages_by_physical_worker():
    assert [
        _physical_worker_index(offset, pipeline_stage_num=3)
        for offset in range(6)
    ] == [0, 0, 0, 1, 1, 1]


def test_pinned_omnigibson_partial_reset_requires_one_child_per_actor():
    assert _validate_reset_isolation(None, num_envs=2) == [0, 1]
    assert _validate_reset_isolation([0], num_envs=1) == [0]
    with pytest.raises(RuntimeError, match="simulator-global physics step"):
        _validate_reset_isolation([0], num_envs=2)


def test_behavior_pool_lease_allocator_reuses_released_gap():
    leases = {0: 2, 4: 2}

    assert BehaviorProcessPool._find_lease_offset(6, leases, 2) == 2
    assert BehaviorProcessPool._find_lease_offset(6, leases, 3) is None


def test_behavior_chunk_preserves_done_timeline_without_auto_reset(monkeypatch):
    env, pool = _make_env(monkeypatch, auto_reset=False)
    pool.steps.extend(
        [
            _step(
                [1, 2],
                [0, 0],
                [True, False],
                [False, False],
                [{"native": 1}, {"native": 2}],
            ),
            _step(
                [3, 4],
                [0, 0],
                [False, False],
                [False, True],
                [{"native": 3}, {"native": 4}],
            ),
        ]
    )

    _obs, _rewards, terminated, truncated, _infos = env.chunk_step(
        torch.zeros((2, 2, 4))
    )

    assert terminated.tolist() == [[True, False], [False, False]]
    assert truncated.tolist() == [[False, False], [False, True]]


def test_behavior_auto_reset_is_partial_and_keeps_final_payload(monkeypatch):
    env, pool = _make_env(monkeypatch, num_envs=3, auto_reset=True)
    pool.reset_calls.clear()
    pool.steps.append(
        _step(
            [10, 20, 30],
            [1, 2, 3],
            [False, True, False],
            [False, False, False],
            [
                {"native": "left"},
                {"native": "done", "done": {"success": True}},
                {"native": "right"},
            ],
        )
    )

    obs, _rewards, terminated, truncated, infos = env.step(
        torch.zeros((3, 4))
    )

    assert pool.reset_calls[0][0] == [1]
    assert obs["states"].tolist() == [10, 201, 30]
    assert infos["final_observation"]["states"].tolist() == [10, 20, 30]
    assert infos["final_info"]["native"] == ["left", "done", "right"]
    assert infos["_final_info"].tolist() == [False, True, False]
    assert infos["reset_seed"] == [
        None,
        derive_behavior_seed(
            10,
            worker_seed_offset=3,
            row_index=1,
            reset_count=1,
            stream=2,
        ),
        None,
    ]
    assert infos["_reset_seed"].tolist() == [False, True, False]
    assert terminated.tolist() == [False, True, False]
    assert truncated.tolist() == [False, False, False]
    assert env.elapsed_steps.tolist() == [1, 0, 1]
    assert env.returns.tolist() == [1, 0, 3]


def test_behavior_ignore_success_does_not_turn_it_into_truncation(monkeypatch):
    env, pool = _make_env(
        monkeypatch,
        auto_reset=True,
        ignore_terminations=True,
    )
    pool.reset_calls.clear()
    pool.steps.append(
        _step(
            [1, 2],
            [0, 0],
            [True, False],
            [False, False],
            [{"done": {"success": True}}, {}],
        )
    )

    _obs, _rewards, terminated, truncated, _infos = env.step(
        torch.zeros((2, 4))
    )

    assert not terminated.any()
    assert not truncated.any()
    assert pool.reset_calls == []


def test_behavior_metrics_freeze_at_first_success(monkeypatch):
    env, _pool = _make_env(monkeypatch, auto_reset=False)

    env._elapsed_steps += 1
    first = env._record_metrics(
        torch.tensor([1.0, 0.5]),
        torch.tensor([True, False]),
        {},
    )
    env._elapsed_steps += 1
    second = env._record_metrics(
        torch.tensor([5.0, 0.5]),
        torch.tensor([True, False]),
        {},
    )

    assert first["episode"]["return"].tolist() == [1.0, 0.5]
    assert second["episode"]["return"].tolist() == [1.0, 1.0]
    assert second["episode"]["reward"].tolist() == [1.0, 0.5]
    assert env.success_episode_len.tolist() == [1, 0]


def test_behavior_config_uses_r1pro_schema_and_preserves_task_defaults():
    cfg = build_behavior_cfg(
        "turning_on_radio",
        omni_config={"task": {"activity_definition_id": 2}},
    )

    assert cfg.omni_config.robots[0].model == "r1pro"
    assert cfg.omni_config.task.activity_name == "turning_on_radio"
    assert cfg.omni_config.task.activity_definition_id == 2
    assert cfg.omni_config.task.instance_resample_mode == "disabled"


def test_instance_loader_resamples_only_selected_rows_deterministically(monkeypatch):
    files = tuple(
        ActivityInstanceFile(idx, f"/tmp/{idx}.json", "template")
        for idx in range(3)
    )
    loader = ActivityInstanceLoader(
        omni_cfg=OmegaConf.create({"task": {}}),
        activity_name="task",
        activity_instance_id=0,
        instance_resample_mode="offline",
        activity_instances=files,
        seed=0,
    )
    captured = {}

    def _capture(_vec_env, selected_files, env_indices=None, seeds=None):
        captured["ids"] = [item.instance_id for item in selected_files]
        captured["indices"] = env_indices
        captured["seeds"] = seeds

    monkeypatch.setattr(loader, "_apply_instance_files", _capture)
    vec_env = SimpleNamespace(envs=[object(), object(), object(), object()])
    loader.prepare_reset(vec_env, env_indices=[1, 3], seeds=[7, 11])

    expected = [
        __import__("random").Random(seed).choice(files).instance_id
        for seed in (7, 11)
    ]
    assert captured == {
        "ids": expected,
        "indices": [1, 3],
        "seeds": [7, 11],
    }


def test_instance_loader_rejects_partial_tro_state_hot_switch():
    files = (
        ActivityInstanceFile(0, "/tmp/0.json", "tro_state"),
        ActivityInstanceFile(1, "/tmp/1.json", "tro_state"),
    )
    loader = ActivityInstanceLoader(
        omni_cfg=OmegaConf.create({"task": {}}),
        activity_name="task",
        activity_instance_id=0,
        instance_resample_mode="offline",
        activity_instances=files,
        seed=0,
    )
    vec_env = SimpleNamespace(envs=[object(), object()])

    with pytest.raises(RuntimeError, match="global physics steps"):
        loader.prepare_reset(vec_env, env_indices=[1], seeds=[7])


def test_template_partial_reload_post_loads_only_selected_children(monkeypatch):
    class Child:
        def __init__(self):
            self.reload_count = 0
            self.post_load_count = 0

        def reload(self, _config):
            self.reload_count += 1

        def post_play_load(self):
            self.post_load_count += 1

    fake_og = ModuleType("omnigibson")
    fake_og.sim = SimpleNamespace(
        is_stopped=lambda: True,
        stop=lambda: None,
        play=lambda: None,
    )
    monkeypatch.setitem(sys.modules, "omnigibson", fake_og)
    seeded: list[int] = []
    monkeypatch.setattr(
        "sim.envs.behavior.instance_loader.seed_behavior_reset_rngs",
        seeded.append,
    )
    children = [Child(), Child()]
    loader = ActivityInstanceLoader(
        omni_cfg=OmegaConf.create(
            {
                "task": {},
                "scene": {},
            }
        ),
        activity_name="task",
        activity_instance_id=0,
        instance_resample_mode="offline",
        activity_instances=(
            ActivityInstanceFile(0, "/tmp/0.json", "template"),
        ),
        seed=0,
    )

    loader._load_template_instances(
        SimpleNamespace(envs=children),
        [children[1]],
        [loader.activity_instances[0]],
        seeds=[13],
    )

    assert children[0].reload_count == 0
    assert children[0].post_load_count == 0
    assert children[1].reload_count == 1
    assert children[1].post_load_count == 1
    assert seeded == [13, 13]


def test_behavior_fixed_reset_ids_fail_fast(monkeypatch):
    cfg = build_behavior_cfg(
        "turning_on_radio",
        use_fixed_reset_state_ids=True,
    )
    with pytest.raises(ValueError, match="does not support fixed reset_state_ids"):
        BehaviorEnv(
            cfg=cfg,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=None,
        )
