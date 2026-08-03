from __future__ import annotations

import json
import threading
import time

import pytest

import agent.cli.batch_eval as batch_eval
from agent.backends.provider_config import PlannerProviderConfig
from agent.cli.batch_eval import (
    build_mcp_episode_worker_factory,
    load_parallel_episode_manifest,
    main,
)
from agent.runtime.episode import EpisodeResult
from agent.runtime.parallel import (
    MAX_PARALLEL_EPISODES,
    ParallelEpisodeHarness,
    ParallelEpisodeSpec,
    ParallelEpisodeWorker,
)


class FakeRunner:
    def __init__(
        self,
        spec: ParallelEpisodeSpec,
        state: dict,
        lock: threading.Lock,
        *,
        fail: bool = False,
    ) -> None:
        self.spec = spec
        self.state = state
        self.lock = lock
        self.fail = fail

    def run(
        self,
        *,
        task,
        max_turns,
        max_tool_calls,
        timeout_s,
        max_total_tokens,
        metadata,
    ):
        assert task == self.spec.task
        assert max_turns == self.spec.max_turns
        assert max_tool_calls == self.spec.max_tool_calls
        assert timeout_s == self.spec.timeout_s
        assert max_total_tokens == self.spec.max_total_tokens
        assert metadata["episode_id"] == self.spec.episode_id
        with self.lock:
            self.state["active"] += 1
            self.state["max_active"] = max(
                self.state["max_active"], self.state["active"]
            )
        try:
            time.sleep(0.03)
            if self.fail:
                raise RuntimeError(f"failed {self.spec.episode_id}")
            return EpisodeResult(
                task=task,
                session_id=f"session-{self.spec.episode_id}",
                terminated=True,
                metadata={"stop_reason": "task_complete", "waiting_for_human": False},
            )
        finally:
            with self.lock:
                self.state["active"] -= 1


def _spec(index: int) -> ParallelEpisodeSpec:
    return ParallelEpisodeSpec(
        episode_id=f"episode-{index}",
        task=f"task {index}",
        env_id=f"openeta/test-{index}-v0",
        max_turns=3,
    )


def test_batch_worker_factory_does_not_gate_on_remote_move_to_schema(
    monkeypatch,
) -> None:
    class OldMoveToTransport:
        instances = 0

        def __init__(self, url: str) -> None:
            self.url = url
            type(self).instances += 1

        def list_tools(self, *, timeout_s=None):
            del timeout_s
            tools = []
            for name in (
                "create_env",
                "reset_env",
                "render_env",
                "close_env",
                "move_to",
                "gripper_open",
                "gripper_close",
            ):
                tool = {"name": name}
                if name == "move_to":
                    tool["input_schema"] = {
                        "type": "object",
                        "properties": {"x": {}, "y": {}, "z": {}},
                    }
                tools.append(tool)
            return {"tools": tools}

    monkeypatch.setattr(
        batch_eval,
        "load_planner_provider_config",
        lambda: PlannerProviderConfig(
            model="test-model",
            api_base="http://provider.example/v1",
            api_key="test-key",
        ),
    )
    monkeypatch.setattr(
        batch_eval,
        "SseSimulatorMcpTransport",
        OldMoveToTransport,
    )

    factory = build_mcp_episode_worker_factory(
        sim_url="http://sim.example/sse",
        sam3_url="http://sam3.example/sse",
        anygrasp_url="http://anygrasp.example/sse",
    )

    assert callable(factory)
    assert OldMoveToTransport.instances == 0


def test_parallel_harness_bounds_concurrency_preserves_order_and_cleans_up() -> None:
    state = {"active": 0, "max_active": 0, "closed": []}
    lock = threading.Lock()

    def factory(spec, batch_id):
        assert batch_id == "batch-test"
        return ParallelEpisodeWorker(
            runner=FakeRunner(spec, state, lock),
            close=lambda: _record_close(state, lock, spec.episode_id),
        )

    result = ParallelEpisodeHarness(factory, concurrency=2).run(
        [_spec(index) for index in range(5)],
        batch_id="batch-test",
    )

    assert state["max_active"] == 2
    assert sorted(state["closed"]) == [f"episode-{index}" for index in range(5)]
    assert [outcome.spec.episode_id for outcome in result.outcomes] == [
        f"episode-{index}" for index in range(5)
    ]
    assert result.failed_count == 0
    payload = result.to_dict()
    assert payload["schema_version"] == "openeta.parallel_episode_batch.v2"
    assert payload["autonomous_success_count"] == 5
    assert payload["rates"] == {
        "success_rate": 1.0,
        "autonomous_success_rate": 1.0,
        "assisted_success_rate": 0.0,
        "intervention_rate": 0.0,
    }


def test_parallel_harness_isolates_worker_failure_and_still_cleans_up() -> None:
    state = {"active": 0, "max_active": 0, "closed": []}
    lock = threading.Lock()

    def factory(spec, batch_id):
        del batch_id
        return ParallelEpisodeWorker(
            runner=FakeRunner(spec, state, lock, fail=spec.episode_id == "episode-1"),
            close=lambda: _record_close(state, lock, spec.episode_id),
        )

    result = ParallelEpisodeHarness(factory, concurrency=3).run(
        [_spec(index) for index in range(3)]
    )

    assert [outcome.status for outcome in result.outcomes] == [
        "success",
        "fail",
        "success",
    ]
    assert result.outcomes[1].error["type"] == "RuntimeError"
    assert sorted(state["closed"]) == ["episode-0", "episode-1", "episode-2"]


def test_parallel_harness_closes_environment_after_need_human() -> None:
    state = {"active": 0, "max_active": 0, "closed": []}
    lock = threading.Lock()
    spec = _spec(0)

    class NeedHumanRunner(FakeRunner):
        def run(self, **kwargs):
            task = kwargs["task"]
            return EpisodeResult(
                task=task,
                session_id="session-human",
                metadata={"waiting_for_human": True, "stop_reason": "ask_human"},
            )

    result = ParallelEpisodeHarness(
        lambda spec, batch_id: ParallelEpisodeWorker(
            runner=NeedHumanRunner(spec, state, lock),
            close=lambda: _record_close(state, lock, spec.episode_id),
            pause=lambda episode: {
                "session_id": episode.session_id,
                "interaction_id": "interaction-1",
                "question": "Which object?",
                "terminal": False,
            },
        ),
        concurrency=1,
    ).run([spec])

    assert result.outcomes[0].status == "need_human"
    assert result.need_human_count == 1
    assert result.outcomes[0].interaction["session_id"] == "session-human"
    assert result.outcomes[0].cleanup == {"ok": True}
    assert state["closed"] == ["episode-0"]


def test_parallel_harness_enforces_hard_concurrency_limit() -> None:
    with pytest.raises(ValueError, match="concurrency"):
        ParallelEpisodeHarness(lambda spec, batch_id: None, concurrency=0)
    with pytest.raises(ValueError, match="concurrency"):
        ParallelEpisodeHarness(
            lambda spec, batch_id: None,
            concurrency=MAX_PARALLEL_EPISODES + 1,
        )


def test_parallel_manifest_loader_and_validate_only(tmp_path, capsys) -> None:
    manifest = tmp_path / "batch.json"
    manifest.write_text(
        json.dumps(
            {
                "episodes": [
                    {
                        "episode_id": "libero-0",
                        "task": "pick up the soup can",
                        "env_id": "openeta/libero_object_task0-v0",
                        "seed": 7,
                        "max_turns": 12,
                        "max_tool_calls": 50,
                        "timeout_s": 600,
                        "max_total_tokens": 5_000_000,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    specs = load_parallel_episode_manifest(manifest)

    assert specs[0].seed == 7
    assert specs[0].max_turns == 12
    assert specs[0].max_tool_calls == 50
    assert specs[0].timeout_s == 600
    assert specs[0].max_total_tokens == 5_000_000
    assert main(["--manifest", str(manifest), "--validate-only"]) == 0
    assert json.loads(capsys.readouterr().out)["episode_count"] == 1


def _record_close(state: dict, lock: threading.Lock, episode_id: str) -> dict:
    with lock:
        state["closed"].append(episode_id)
    return {"ok": True}
