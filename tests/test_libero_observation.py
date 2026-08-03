from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from logger.observability import EpisodeObservability
from sim.bench_worker import _reset_with_image
from sim.env_registry import _LibEnvWrapper
from sim.libero_observation import LiberoObservationFacade


class _FakeLiberoEnv:
    def __init__(self) -> None:
        self.step_idx = 0
        self.closed = False

    def _observation(self) -> dict:
        return {
            "cameras": {
                "agentview": {
                    "rgb": np.full((4, 5, 3), self.step_idx, dtype=np.uint8),
                    "depth": np.full((4, 5), 0.5 + self.step_idx, dtype=np.float32),
                    "intrinsics": {"fx": 10.0, "fy": 10.0, "cx": 2.0, "cy": 2.0},
                    "extrinsics": {"frame_transform": "camera_to_world"},
                }
            },
            "proprio": {"ee_pose": np.arange(7, dtype=np.float32)},
            "task_description": "pick up the cube",
        }

    def reset(self, **_kwargs):
        self.step_idx = 0
        return self._observation(), {"backend": "fake-libero"}

    def step(self, _action):
        self.step_idx += 1
        return self._observation(), 1.25, False, False, {}

    def render(self):
        return self._observation()["cameras"]["agentview"]["rgb"]

    def check_success(self):
        return True

    def get_sim_state(self):
        return {"sim_step": self.step_idx}

    def close(self):
        self.closed = True


def _events(root: Path) -> list[dict]:
    return [json.loads(line) for line in (root / "events.jsonl").read_text().splitlines()]


def test_libero_facade_retains_rgbd_and_correlates_action_to_post_frame(tmp_path: Path) -> None:
    root = tmp_path / "episode"
    writer = EpisodeObservability(root, episode_id="libero-ep", env_id="libero-task0")
    env = _FakeLiberoEnv()
    facade = LiberoObservationFacade(env, writer)

    observation, info = facade.reset(seed=3)
    assert observation["task_description"] == "pick up the cube"
    first_record = info["observation_record"]
    first_frame = first_record["frames"][0]
    assert first_frame["camera_id"] == "agentview"
    assert first_frame["metadata"]["depth_unit"] == "metres_float32"
    assert (root / first_frame["rgb_path"]).is_file()
    assert (root / first_frame["depth_path"]).is_file()

    _, reward, terminated, truncated, step_info = facade.step(np.zeros(7, dtype=np.float32))
    assert reward == 1.25
    assert not terminated and not truncated
    second_record = step_info["observation_record"]
    assert second_record["frame_ids"] != first_record["frame_ids"]

    events = _events(root)
    action = next(event for event in events if event["kind"] == "action")
    result = next(event for event in events if event["kind"] == "tool_result")
    assert action["frame_refs"]["input"] == first_record["frame_ids"]
    assert result["action_id"] == action["action_id"]
    assert result["frame_refs"]["post"] == second_record["frame_ids"]
    assert result["payload"]["result"]["reward"] == 1.25

    assert facade.check_success() is True
    assert facade.get_sim_state() == {"sim_step": 1}
    facade.close()
    assert env.closed is True


def test_libero_wrapper_seeds_native_env_and_does_not_mark_done_as_truncated() -> None:
    class RawEnv:
        def __init__(self) -> None:
            self.seed_value = None

        def seed(self, value):
            self.seed_value = value

        def reset(self):
            return {"agentview_image": np.zeros((2, 2, 3), dtype=np.uint8)}

        def step(self, _action):
            return self.reset(), 0.0, True, {"native": True}

        def close(self):
            pass

    raw = RawEnv()
    wrapper = _LibEnvWrapper(raw)
    wrapper.reset(seed=17)
    result = wrapper.step(np.zeros(7, dtype=np.float32))
    assert raw.seed_value == 17
    assert result[2] is True
    assert result[3] is False


def test_libero_wrapper_applies_official_init_state_and_records_selection() -> None:
    class RawEnv:
        def __init__(self) -> None:
            self.seed_value = None
            self.reset_count = 0
            self.applied_init_states: list[object] = []

        def seed(self, value):
            self.seed_value = value

        def reset(self):
            self.reset_count += 1
            return {"agentview_image": np.zeros((2, 2, 3), dtype=np.uint8)}

        def set_init_state(self, state):
            self.applied_init_states.append(state)
            return {
                "agentview_image": np.full((2, 2, 3), len(self.applied_init_states), dtype=np.uint8)
            }

        def close(self):
            pass

    raw = RawEnv()
    wrapper = _LibEnvWrapper(
        raw,
        benchmark_init_states=["state-0", "state-1", "state-2"],
        benchmark_init_provenance={"mode": "official_benchmark_init_state"},
        default_init_seed=0,
    )

    observation, info = wrapper.reset(seed=17)

    assert raw.seed_value == 17
    assert raw.reset_count == 1
    assert raw.applied_init_states == ["state-2"]
    assert observation["agentview_image"][0, 0, 0] == 1
    assert info["initialization"] == {
        "mode": "official_benchmark_init_state",
        "init_state_index": 2,
        "selection": "seed_modulo_available_init_states",
        "reset_seed": 17,
        "selection_seed": 17,
    }

    _, explicit_info = wrapper.reset(options={"init_state_index": 1})

    assert raw.applied_init_states == ["state-2", "state-1"]
    assert explicit_info["initialization"]["init_state_index"] == 1
    assert explicit_info["initialization"]["selection"] == "explicit_builder_init_state_index"
    assert explicit_info["initialization"]["selection_seed"] is None


def test_bench_reset_keeps_official_initialization_provenance_in_metadata() -> None:
    class Env:
        def reset(self, *, seed=None):
            assert seed == 24
            return {
                "task_description": "pick up the alphabet soup",
                "cameras": {
                    "agentview": {
                        "rgb": np.zeros((2, 2, 3), dtype=np.uint8),
                    }
                },
            }, {
                "initialization": {
                    "mode": "official_benchmark_init_state",
                    "init_state_index": 24,
                }
            }

        def render(self):
            return None

    result = _reset_with_image(Env(), seed=24)

    assert result["metadata"]["reset_info"] == {
        "initialization": {
            "mode": "official_benchmark_init_state",
            "init_state_index": 24,
        }
    }
