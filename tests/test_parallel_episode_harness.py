from __future__ import annotations

import json
import threading
import time

import pytest

import agent.cli.batch_eval as batch_eval
import agent.runtime.runtime_assembly as runtime_assembly
from adapter.protocol import EnvAction, EnvObservation, RobotState, StepResult
from agent.backends.provider_config import PlannerProviderConfig
from agent.cli.batch_eval import (
    build_mcp_episode_worker_factory,
    load_parallel_episode_manifest,
    main,
)
from agent.runtime.episode import EpisodeResult, EpisodeStep
from agent.runtime.parallel import (
    MAX_PARALLEL_EPISODES,
    ParallelEpisodeHarness,
    ParallelEpisodeSpec,
    ParallelEpisodeWorker,
    classify_episode_result,
)
from agent.tools.registry import ToolResult


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
            self.state.setdefault("metadata", []).append(dict(metadata))
            self.state["active"] += 1
            self.state["max_active"] = max(self.state["max_active"], self.state["active"])
        try:
            time.sleep(0.03)
            if self.fail:
                raise RuntimeError(f"failed {self.spec.episode_id}")
            return EpisodeResult(
                task=task,
                session_id=f"session-{self.spec.episode_id}",
                steps=[
                    EpisodeStep(
                        turn_index=1,
                        observation=EnvObservation(
                            task=task,
                            cameras=[],
                            robot=RobotState(),
                        ),
                        action=EnvAction(
                            action_type="response",
                            command={
                                "request": {
                                    "kind": "response",
                                    "name": "task_complete",
                                    "parameters": {},
                                }
                            },
                        ),
                        step_result=StepResult(
                            observation=EnvObservation(
                                task=task,
                                cameras=[],
                                robot=RobotState(),
                            ),
                            reward=1.0,
                            terminated=True,
                        ),
                    )
                ],
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


def test_libero_task_complete_without_official_reward_is_failure() -> None:
    episode = EpisodeResult(
        task="pick object",
        session_id="session-no-reward",
        terminated=True,
        metadata={"stop_reason": "task_complete"},
    )

    assert (
        classify_episode_result(
            episode,
            env_id="openeta/libero_libero_10_task0-v0",
        )
        == "fail"
    )
    assert classify_episode_result(episode, env_id="openeta/test-v0") == "success"


def test_unattended_need_human_fails_without_persisting_pause() -> None:
    pauses: list[str] = []

    class NeedHumanRunner:
        def run(self, **kwargs):
            return EpisodeResult(
                task=kwargs["task"],
                session_id="session-unattended",
                metadata={"waiting_for_human": True},
            )

    spec = ParallelEpisodeSpec(
        episode_id="episode-unattended",
        task="pick cube",
        env_id="env",
        metadata={"on_need_human": "fail"},
    )
    harness = ParallelEpisodeHarness(
        lambda _spec, _batch_id: ParallelEpisodeWorker(
            runner=NeedHumanRunner(),
            close=lambda: {"ok": True},
            pause=lambda result: pauses.append(str(result.session_id)) or {},
        ),
        concurrency=1,
    )

    result = harness.run([spec])

    assert result.fail_count == 1
    assert result.outcomes[0].error["code"] == "need_human_in_unattended_run"
    assert pauses == []


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


def test_batch_worker_factory_binds_configured_anyplace(
    monkeypatch,
    tmp_path,
) -> None:
    class FakeTransport:
        def __init__(self, url: str) -> None:
            self.url = url

    seen_urls: list[str] = []
    monkeypatch.setattr(
        batch_eval,
        "load_planner_provider_config",
        lambda: PlannerProviderConfig(
            model="test-model",
            api_base="http://provider.example/v1",
            api_key="test-key",
        ),
    )
    monkeypatch.setattr(batch_eval, "load_mcp_server_url", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(batch_eval, "SseSimulatorMcpTransport", FakeTransport)
    monkeypatch.setattr(
        runtime_assembly,
        "build_sse_anyplace_mcp_placer",
        lambda *, url: seen_urls.append(url) or (lambda _request: {}),
    )
    spec = ParallelEpisodeSpec(
        episode_id="episode-anyplace",
        task="pick cube and place it in basket",
        env_id="openeta/test-v0",
        metadata={"workspace_parent": str(tmp_path)},
    )

    worker = build_mcp_episode_worker_factory(
        sim_url="http://sim.example/sse",
        anyplace_url="http://anyplace.example/sse",
    )(spec, "batch-anyplace")

    assert seen_urls == ["http://anyplace.example/sse"]
    assert worker.runner.runtime.tools.can_execute("anyplace") is True
    session_id = worker.runner.initial_session_id
    store = worker.runner.runtime.memory.store
    assert store.root == tmp_path
    assert store.session_dir(session_id) == tmp_path / "sessions" / session_id


def test_batch_worker_factory_binds_configured_molmopoint(
    monkeypatch,
    tmp_path,
) -> None:
    class FakeTransport:
        def __init__(self, url: str) -> None:
            self.url = url

    seen_urls: list[str] = []
    pointer = object()
    monkeypatch.setattr(
        batch_eval,
        "load_planner_provider_config",
        lambda: PlannerProviderConfig(
            model="test-model",
            api_base="http://provider.example/v1",
            api_key="test-key",
        ),
    )
    monkeypatch.setattr(batch_eval, "load_mcp_server_url", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(batch_eval, "SseSimulatorMcpTransport", FakeTransport)
    monkeypatch.setattr(
        runtime_assembly,
        "build_sse_molmopoint_mcp_pointer",
        lambda *, url: seen_urls.append(url) or pointer,
    )
    monkeypatch.setattr(
        runtime_assembly,
        "build_molmopoint_handler",
        lambda point_images, **_kwargs: (
            (lambda _context: ToolResult(True, content="pointed"))
            if point_images is pointer
            else pytest.fail("unexpected MolmoPoint callable")
        ),
    )
    spec = ParallelEpisodeSpec(
        episode_id="episode-molmopoint",
        task="locate the target",
        env_id="openeta/test-v0",
        metadata={"workspace_parent": str(tmp_path)},
    )

    worker = build_mcp_episode_worker_factory(
        sim_url="http://sim.example/sse",
        molmopoint_url="http://molmopoint.example/sse",
    )(spec, "batch-molmopoint")

    assert seen_urls == ["http://molmopoint.example/sse"]
    assert worker.runner.runtime.tools.can_execute("molmopoint") is True


def test_reviewed_batch_skill_author_sees_runtime_bound_internal_tools(
    monkeypatch,
    tmp_path,
) -> None:
    class FakeTransport:
        def __init__(self, url: str) -> None:
            self.url = url

    monkeypatch.setattr(
        batch_eval,
        "load_planner_provider_config",
        lambda: PlannerProviderConfig(
            model="test-model",
            api_base="http://provider.example/v1",
            api_key="test-key",
        ),
    )
    monkeypatch.setattr(batch_eval, "load_mcp_server_url", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(batch_eval, "SseSimulatorMcpTransport", FakeTransport)
    spec = ParallelEpisodeSpec(
        episode_id="episode-reviewed-tools",
        task="pick cube",
        env_id="openeta/test-v0",
        metadata={"workspace_parent": str(tmp_path)},
    )

    worker = build_mcp_episode_worker_factory(
        sim_url="http://sim.example/sse",
        supervision_profile="reviewed_autonomy",
    )(spec, "batch-reviewed-tools")

    applier = worker.runner.runtime.self_improvement_reviewer.auto_applier
    executable_names = {tool_spec.name for tool_spec in applier.executable_tools}
    assert "select_sam3_detection" in executable_names
    assert "scene_detector" not in executable_names


def test_batch_worker_factory_binds_graspgenx_behind_unified_tool(
    monkeypatch,
) -> None:
    class FakeTransport:
        def __init__(self, url: str) -> None:
            assert url == "http://sim.example/sse"

        def list_tools(self, *, timeout_s=None):
            del timeout_s
            return {
                "tools": [
                    {"name": name}
                    for name in (
                        "create_env",
                        "reset_env",
                        "render_env",
                        "close_env",
                        "move_to",
                        "gripper_open",
                        "gripper_close",
                    )
                ]
            }

    predictor = object()
    lister = object()
    monkeypatch.setattr(
        batch_eval,
        "load_planner_provider_config",
        lambda: PlannerProviderConfig(
            model="test-model",
            api_base="http://provider.example/v1",
            api_key="test-key",
        ),
    )
    monkeypatch.setattr(batch_eval, "load_mcp_server_url", lambda *_args, **_kwargs: "")
    monkeypatch.setattr(batch_eval, "SseSimulatorMcpTransport", FakeTransport)
    monkeypatch.setattr(
        runtime_assembly,
        "build_sse_graspgenx_mcp_predictor",
        lambda *, url: (
            predictor
            if url == "http://graspgenx.example/sse"
            else pytest.fail("unexpected GraspGenX URL")
        ),
    )
    monkeypatch.setattr(
        runtime_assembly,
        "build_sse_graspgenx_mcp_gripper_lister",
        lambda *, url: (
            lister
            if url == "http://graspgenx.example/sse"
            else pytest.fail("unexpected GraspGenX URL")
        ),
    )
    monkeypatch.setattr(
        runtime_assembly,
        "build_graspgenx_handler",
        lambda predict, list_grippers, **_kwargs: (
            (lambda _context: ToolResult(True, content="predicted"))
            if predict is predictor and list_grippers is lister
            else pytest.fail("unexpected GraspGenX callables")
        ),
    )
    facade_backends = {}
    monkeypatch.setattr(
        runtime_assembly,
        "build_grasp_pose_estimate_handler",
        lambda handlers: facade_backends.update(handlers)
        or (lambda _context: ToolResult(True, content="predicted")),
    )

    factory = build_mcp_episode_worker_factory(
        sim_url="http://sim.example/sse",
        graspgenx_url="http://graspgenx.example/sse",
    )
    worker = factory(_spec(0), "batch-graspgenx")

    assert worker.runner.runtime.tools.can_execute("grasp_pose_estimate") is True
    assert worker.runner.runtime.tools.can_execute("graspgenx") is False
    assert worker.runner.runtime.tools.can_execute("list_graspgenx_grippers") is False
    assert set(facade_backends) == {"graspgenx"}


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
        "agent_assisted_success_rate": 0.0,
        "human_assisted_success_rate": 0.0,
        "intervention_rate": 0.0,
    }


def test_parallel_worker_host_metadata_overrides_manifest_metadata() -> None:
    state = {"active": 0, "max_active": 0, "closed": []}
    lock = threading.Lock()
    spec = ParallelEpisodeSpec(
        episode_id="profile-run",
        task="calibrate profile",
        env_id="env",
        metadata={"calibration_profile_sha256": "untrusted-manifest-value"},
    )

    def factory(current_spec, _batch_id):
        return ParallelEpisodeWorker(
            runner=FakeRunner(current_spec, state, lock),
            close=lambda: _record_close(state, lock, current_spec.episode_id),
            run_metadata={"calibration_profile_sha256": "host-staged-profile-hash"},
        )

    ParallelEpisodeHarness(factory, concurrency=1).run([spec], batch_id="profile-batch")

    assert state["metadata"][0]["calibration_profile_sha256"] == ("host-staged-profile-hash")


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


def test_parallel_harness_interrupt_propagates_to_active_runner() -> None:
    started = threading.Event()
    interrupted = threading.Event()
    closed: list[str] = []

    class InterruptibleRunner:
        def run(self, **kwargs):
            started.set()
            assert interrupted.wait(timeout=2.0)
            return EpisodeResult(
                task=kwargs["task"],
                session_id="session-interrupted",
                metadata={
                    "stop_reason": "episode_interrupted",
                    "failure_reason": {"code": "parallel_batch_interrupted"},
                },
            )

        def interrupt(self, *, code):
            assert code == "parallel_batch_interrupted"
            interrupted.set()
            return {"ok": True, "closed": True}

    harness = ParallelEpisodeHarness(
        lambda spec, _batch_id: ParallelEpisodeWorker(
            runner=InterruptibleRunner(),
            close=lambda: closed.append(spec.episode_id) or {"ok": True},
        ),
        concurrency=1,
    )
    results = []
    thread = threading.Thread(target=lambda: results.append(harness.run([_spec(0)])))
    thread.start()
    assert started.wait(timeout=1.0)

    cleanup = harness.interrupt()
    thread.join(timeout=2.0)

    assert thread.is_alive() is False
    assert cleanup == [{"index": 0, "ok": True, "closed": True}]
    assert results[0].fail_count == 1
    assert closed == ["episode-0"]


def test_parallel_harness_preserves_structured_exception_code() -> None:
    class StructuredFailure(RuntimeError):
        code = "provider_queue_timeout"

    def factory(spec, batch_id):
        del spec, batch_id
        raise StructuredFailure("queue timed out")

    result = ParallelEpisodeHarness(factory, concurrency=1).run([_spec(0)])

    assert result.outcomes[0].error == {
        "type": "StructuredFailure",
        "message": "queue timed out",
        "code": "provider_queue_timeout",
    }


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
                        "max_tool_calls": 200,
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
    assert specs[0].max_tool_calls == 200
    assert specs[0].timeout_s == 600
    assert specs[0].max_total_tokens == 5_000_000
    assert main(["--manifest", str(manifest), "--validate-only"]) == 0
    assert json.loads(capsys.readouterr().out)["episode_count"] == 1


def _record_close(state: dict, lock: threading.Lock, episode_id: str) -> dict:
    with lock:
        state["closed"].append(episode_id)
    return {"ok": True}
