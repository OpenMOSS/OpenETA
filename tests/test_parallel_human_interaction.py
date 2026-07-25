from __future__ import annotations

import pytest

import agent.cli.batch_eval as batch_eval
from agent.backends.planner import StaticPlannerBackend
from agent.runtime.episode import OpenEtaEpisodeRunner
from agent.runtime.interactions import PausedEpisodeRecord, PausedEpisodeStore
from agent.runtime.memory import AgentMemory
from agent.runtime.memory_store import JsonMemoryStore
from agent.runtime.parallel import ParallelEpisodeWorker
from agent.runtime.planner import ToolCallingPlanner
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.tools.sim_mcp import (
    SimulatorMcpEpisodeConfig,
    SimulatorMcpEpisodeEnvironment,
)


class ResumeTransport:
    def __init__(self) -> None:
        self.calls = []

    def call_tool(self, name, arguments, *, timeout_s=None):
        self.calls.append({"name": name, "arguments": dict(arguments)})
        if name == "create_env":
            return {
                "success": True,
                "handle": "new-sim-handle",
                "session_id": "new-sim-session",
            }
        if name == "reset_env":
            return {"success": True, "cameras": [], "robot": {}}
        if name == "close_env":
            return {"ok": True}
        raise AssertionError(name)


def _paused_record(tmp_path) -> PausedEpisodeRecord:
    return PausedEpisodeRecord(
        batch_id="batch-human",
        episode_id="episode-human",
        session_id="session-human",
        interaction_id="interaction-current",
        question="Which object should I pick?",
        task="pick one object",
        env_id="openeta/test-v0",
        seed=3,
        max_turns=5,
        turn_index=1,
        memory_root=str(tmp_path / "memory"),
        artifact_root=str(tmp_path / "artifacts"),
        tool_call_count=7,
        total_tokens=1234,
        token_usage_sources={"provider": 2},
    )


def test_resume_paused_episode_records_answer_and_finishes_success(
    monkeypatch, tmp_path
) -> None:
    record = _paused_record(tmp_path)
    store = PausedEpisodeStore(tmp_path / "interactions")
    store.save(record)
    memory = AgentMemory(store=JsonMemoryStore(root=record.memory_root))
    runtime = OpenEtaAgentRuntime(
        planner=ToolCallingPlanner(
            StaticPlannerBackend(
                {
                    "kind": "response",
                    "name": "task_complete",
                    "parameters": {"success": True, "summary": "object selected"},
                }
            )
        ),
        memory=memory,
    )
    runtime.start_session(task=record.task, session_id=record.session_id)
    transport = ResumeTransport()
    environment = SimulatorMcpEpisodeEnvironment(
        transport=transport,
        config=SimulatorMcpEpisodeConfig(env_id=record.env_id, seed=record.seed),
    )
    worker = ParallelEpisodeWorker(
        runner=OpenEtaEpisodeRunner(runtime=runtime, environment=environment),
        close=environment.close,
    )
    monkeypatch.setattr(batch_eval, "PausedEpisodeStore", lambda: store)
    monkeypatch.setattr(
        batch_eval,
        "build_mcp_episode_worker_factory",
        lambda **kwargs: lambda spec, batch_id: worker,
    )

    payload = batch_eval.resume_paused_episode(
        session_id=record.session_id,
        interaction_id=record.interaction_id,
        answer="Pick the red cube.",
    )

    outcome = payload["outcome"]
    assert outcome["status"] == "success"
    assert outcome["assistance"] == {
        "assisted": True,
        "human_intervention_count": 1,
        "guidance_intervention_count": 0,
    }
    assert outcome["episode"]["metadata"]["usage"]["tool_call_count"] == 7
    assert outcome["episode"]["metadata"]["usage"]["total_tokens"] == 1234
    assert outcome["episode"]["metadata"]["usage"]["token_usage_sources"] == {
        "provider": 2
    }
    assert [call["name"] for call in transport.calls] == [
        "create_env",
        "reset_env",
        "close_env",
    ]
    assert transport.calls[0]["arguments"]["env_id"] == record.env_id
    assert transport.calls[0]["arguments"]["seed"] == record.seed
    assert transport.calls[0]["arguments"]["task"] == record.task
    assert transport.calls[1]["arguments"]["seed"] == record.seed
    assert not store.path_for(record.session_id).exists()
    human_answers = [
        event for event in runtime.memory.events if event.event_type == "human_answer"
    ]
    assert human_answers[-1].payload["answer"] == "Pick the red cube."


def test_resume_paused_episode_rejects_stale_interaction(monkeypatch, tmp_path) -> None:
    record = _paused_record(tmp_path)
    store = PausedEpisodeStore(tmp_path / "interactions")
    store.save(record)
    monkeypatch.setattr(batch_eval, "PausedEpisodeStore", lambda: store)

    with pytest.raises(ValueError, match="stale"):
        batch_eval.resume_paused_episode(
            session_id=record.session_id,
            interaction_id="interaction-old",
            answer="Pick the cube.",
        )


def test_resume_can_pause_again_with_new_interaction_id(monkeypatch, tmp_path) -> None:
    record = _paused_record(tmp_path)
    store = PausedEpisodeStore(tmp_path / "interactions")
    store.save(record)
    memory = AgentMemory(store=JsonMemoryStore(root=record.memory_root))
    runtime = OpenEtaAgentRuntime(
        planner=ToolCallingPlanner(
            StaticPlannerBackend(
                {
                    "kind": "response",
                    "name": "ask_human",
                    "parameters": {"question": "Where should I place it?"},
                }
            )
        ),
        memory=memory,
    )
    runtime.start_session(task=record.task, session_id=record.session_id)
    transport = ResumeTransport()
    environment = SimulatorMcpEpisodeEnvironment(
        transport=transport,
        config=SimulatorMcpEpisodeConfig(env_id=record.env_id, seed=record.seed),
    )
    worker = ParallelEpisodeWorker(
        runner=OpenEtaEpisodeRunner(runtime=runtime, environment=environment),
        close=environment.close,
    )
    monkeypatch.setattr(batch_eval, "PausedEpisodeStore", lambda: store)
    monkeypatch.setattr(
        batch_eval,
        "build_mcp_episode_worker_factory",
        lambda **kwargs: lambda spec, batch_id: worker,
    )

    payload = batch_eval.resume_paused_episode(
        session_id=record.session_id,
        interaction_id=record.interaction_id,
        answer="Pick the red cube.",
    )

    outcome = payload["outcome"]
    assert outcome["status"] == "need_human"
    assert outcome["interaction"]["interaction_id"] != record.interaction_id
    assert outcome["assistance"]["human_intervention_count"] == 1
    updated = store.load(record.session_id)
    assert updated.human_intervention_count == 1
    assert updated.interaction_history[0]["answer"] == "Pick the red cube."
    assert updated.resume_mode == "restart_environment"
    assert updated.tool_call_count == 7
    assert updated.total_tokens == 1234
    assert updated.token_usage_sources == {"provider": 2}
    assert [call["name"] for call in transport.calls] == [
        "create_env",
        "reset_env",
        "close_env",
    ]


def test_paused_record_does_not_persist_expiring_simulator_handle(tmp_path) -> None:
    payload = _paused_record(tmp_path).to_dict()

    assert payload["schema_version"] == "openeta.paused_episode.v2"
    assert payload["resume_mode"] == "restart_environment"
    assert payload["max_tool_calls"] == 200
    assert payload["timeout_s"] == 3600
    assert payload["max_total_tokens"] == 5_000_000
    assert payload["tool_call_count"] == 7
    assert payload["total_tokens"] == 1234
    assert payload["token_usage_sources"] == {"provider": 2}
    assert "sim_handle" not in payload
    assert "sim_session_id" not in payload


def test_legacy_paused_record_is_migrated_to_environment_restart(tmp_path) -> None:
    payload = _paused_record(tmp_path).to_dict()
    payload["schema_version"] = "openeta.paused_episode.v1"
    payload.pop("resume_mode")
    payload.pop("tool_call_count")
    payload.pop("total_tokens")
    payload.pop("token_usage_sources")
    payload["sim_handle"] = "expired-handle"
    payload["sim_session_id"] = "expired-session"

    record = PausedEpisodeRecord.from_dict(payload)

    assert record.resume_mode == "restart_environment"
    assert record.tool_call_count == 0
    assert record.total_tokens == 0
    assert record.token_usage_sources == {}
    assert "sim_handle" not in record.to_dict()
