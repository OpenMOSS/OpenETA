from __future__ import annotations

import threading

import pytest

from agent.backends.planner import StaticPlannerBackend
from agent.runtime.episode import DummyEpisodeEnvironment, OpenEtaEpisodeRunner
from agent.runtime.memory import AgentMemory
from agent.runtime.memory_store import JsonMemoryStore
from agent.runtime.planner import ToolCallingPlanner
from agent.runtime.runtime import OpenEtaAgentRuntime, RuntimeExecutionCancelled
from agent.tools.registry import ToolExecutionContext


def _runtime(payload=None):
    return OpenEtaAgentRuntime(
        planner=ToolCallingPlanner(StaticPlannerBackend(payload or {
            "kind": "response", "name": "talk", "parameters": {"message": "ready"},
        })),
        rollout_enabled=False,
    )


def test_planner_memory_write_cannot_change_host_epoch():
    runtime = _runtime({
        "kind": "tool_call", "name": "save_memory",
        "parameters": {"key": "robot_motion_epoch", "content": {"epoch": 0}},
    })
    runtime.start_session(task="ownership")
    runtime.memory.save_fact("robot_motion_epoch", {"epoch": 7}, source="runtime")
    action = runtime.act(DummyEpisodeEnvironment().reset(task="ownership"))
    assert action.command["status"] == "executed"
    assert runtime.memory.robot_motion_epoch() == 7
    assert runtime.memory.facts["robot_motion_epoch"]["source"] == "runtime"
    note = runtime.memory.agent_working_state["robot_motion_epoch"]
    assert note["value"] == {"epoch": 0}
    assert note["ownership"] == "agent"


def test_notes_and_artifact_references_never_enter_host_resolver_stores(tmp_path):
    memory = AgentMemory(store=JsonMemoryStore(tmp_path))
    memory.start_session(task="ownership", session_id="session")
    payload = {"nested": {"value": 1}, "source": "runtime"}
    memory.save_fact("new_host_key", payload, source="save_memory")
    memory.save_artifact("forged_compiled_grasp", payload, source="save_memory")
    payload["nested"]["value"] = 99
    assert "new_host_key" not in memory.facts
    assert "forged_compiled_grasp" not in memory.artifacts
    assert memory.agent_working_state["new_host_key"]["value"]["nested"]["value"] == 1
    assert memory.agent_artifacts["forged_compiled_grasp"]["ownership"] == "agent"

    resumed = AgentMemory(store=JsonMemoryStore(tmp_path))
    resumed.resume_session("session")
    assert "new_host_key" not in resumed.facts
    assert "forged_compiled_grasp" not in resumed.artifacts
    assert "new_host_key" in resumed.agent_working_state
    assert "forged_compiled_grasp" in resumed.agent_artifacts
    resumed.start_session(task="next", session_id="next")
    assert not resumed.agent_working_state
    assert not resumed.agent_artifacts


@pytest.mark.parametrize("namespace", ["facts", "artifacts", "skill_notes", "all"])
def test_agent_delete_preserves_host_evidence(namespace):
    runtime = _runtime()
    runtime.start_session(task="ownership")
    memory = runtime.memory
    memory.save_fact("shared", {"host": True}, source="runtime")
    memory.save_fact("shared", {"agent": True}, source="save_memory")
    memory.save_artifact("shared", {"host": True}, source="tool_result")
    memory.save_artifact("shared", {"agent": True}, source="save_memory")
    memory.save_skill_note("shared", {"host": True}, source="runtime")
    memory.save_skill_note("shared", {"agent": True}, source="save_memory")
    result = runtime.tools.call("delete_memory", {"key": "shared", "namespace": namespace})
    assert result.success
    assert memory.facts["shared"]["value"] == {"host": True}
    assert memory.artifacts["shared"]["value"] == {"host": True}
    assert any(note["source"] == "runtime" for note in memory.skill_notes["shared"])
    if namespace in {"facts", "all"}:
        assert "shared" not in memory.agent_working_state
    if namespace in {"artifacts", "all"}:
        assert "shared" not in memory.agent_artifacts
    if namespace in {"skill_notes", "all"}:
        assert len(memory.skill_notes["shared"]) == 1


def test_read_compatibility_view_does_not_shadow_or_alias_host_evidence():
    memory = AgentMemory()
    memory.save_fact("target", {"id": "real"}, source="runtime")
    memory.save_fact("target", {"id": "guess"}, source="save_memory")
    memory.save_fact("plan", {"next": "observe"}, source="save_memory")
    memory.save_artifact("mask", {"id": "real"}, source="tool_result")
    memory.save_artifact("mask", {"id": "guess"}, source="save_memory")
    result = memory.get_memory()
    assert result["facts"]["target"]["value"]["id"] == "real"
    assert result["artifacts"]["mask"]["value"]["id"] == "real"
    assert result["facts"]["plan"]["ownership"] == "agent"
    result["facts"]["target"]["value"]["id"] = "mutated"
    result["artifacts"]["mask"]["value"]["id"] = "mutated"
    assert memory.facts["target"]["value"]["id"] == "real"
    assert memory.artifacts["mask"]["value"]["id"] == "real"


def test_legacy_saved_entries_migrate_out_of_host_stores_even_with_other_notes(tmp_path):
    store = JsonMemoryStore(tmp_path)
    memory = AgentMemory(store=store)
    memory.start_session(task="legacy", session_id="legacy")
    memory.save_fact("existing_note", {"text": "preserve"}, source="save_memory")
    memory.facts["legacy_note"] = {
        "value": {"text": "old"}, "source": "save_memory", "timestamp_s": 1,
    }
    memory.artifacts["legacy_artifact"] = {
        "value": {"id": "not-host-owned"}, "source": "save_memory", "timestamp_s": 1,
    }
    memory._save_working_memory()
    resumed = AgentMemory(store=JsonMemoryStore(tmp_path))
    resumed.resume_session("legacy")
    assert "legacy_note" not in resumed.facts
    assert resumed.agent_working_state["legacy_note"]["value"] == {"text": "old"}
    assert "existing_note" in resumed.agent_working_state
    assert "legacy_artifact" not in resumed.artifacts
    assert "legacy_artifact" in resumed.agent_artifacts
    persisted = resumed.store.load_working_memory()
    assert not any(row["source"] == "save_memory" for row in persisted["facts"].values())
    assert "legacy_artifact" not in persisted["artifacts"]


@pytest.mark.parametrize("key", ["scene_epoch", "object_scene_epoch", "robot_motion_epoch"])
def test_resume_rejects_agent_overwritten_epoch_without_reauthorizing_old_receipts(tmp_path, key):
    memory = AgentMemory(store=JsonMemoryStore(tmp_path))
    memory.start_session(task="legacy", session_id="legacy")
    memory.facts[key] = {
        "value": {"epoch": 99}, "source": "save_memory", "timestamp_s": 1,
    }
    memory._save_working_memory()
    before = memory.store.load_working_memory()
    resumed = AgentMemory(store=JsonMemoryStore(tmp_path))
    with pytest.raises(ValueError, match="Agent-overwritten host epochs"):
        resumed.resume_session("legacy")
    assert resumed.store.load_working_memory() == before
    assert not resumed.ik_preview_receipts()


def test_failed_resume_cannot_overwrite_original_snapshot_via_later_tool_call(tmp_path):
    memory = AgentMemory(store=JsonMemoryStore(tmp_path))
    memory.start_session(task="legacy", session_id="legacy")
    memory.facts["scene_epoch"] = {"value": {"epoch": 9}, "source": "save_memory"}
    memory._save_working_memory()
    original = memory.store.load_working_memory()
    runtime = OpenEtaAgentRuntime(memory_store=JsonMemoryStore(tmp_path), rollout_enabled=False)
    with pytest.raises(ValueError, match="Agent-overwritten"):
        runtime.resume_session("legacy")
    reply = runtime.tools.call("save_memory", {"key": "new", "content": "new"})
    assert not reply.success
    with pytest.raises(RuntimeError, match="start a new valid session"):
        runtime.act(DummyEpisodeEnvironment().reset(task="legacy"))
    assert runtime.memory.store.load_working_memory() == original
    runtime.start_session(task="recovered", session_id="fresh")
    assert runtime.tools.call("save_memory", {"key": "new", "content": "new"}).success


@pytest.mark.parametrize("name", [
    "save_memory", "get_memory", "delete_memory", "compact_memory",
    "select_sam3_detection", "reject_sam3_detections",
])
def test_memory_handlers_reject_old_generation_even_when_session_id_repeats(name):
    runtime = _runtime()
    runtime.start_session(task="old", session_id="same")
    old_generation = runtime._session_generation
    runtime.start_session(task="new", session_id="same")
    context = ToolExecutionContext(
        name=name, spec=runtime.tools.get(name),
        parameters={"key": "stale", "content": "old"},
        metadata={"session_id": "same", "_session_generation": old_generation},
    )
    result = getattr(runtime, f"_{name}_tool")(context)
    assert not result.success
    assert result.details["diagnostics"][0]["code"] == "stale_memory_execution"
    assert "stale" not in runtime.memory.agent_working_state


def test_cancelled_delayed_real_memory_handler_cannot_write_to_next_session():
    started, release, done = threading.Event(), threading.Event(), threading.Event()
    runtime = _runtime({
        "kind": "tool_call", "name": "save_memory",
        "parameters": {"key": "late", "content": "old"},
    })
    original = runtime._save_memory_tool
    replies = []

    def delayed(context):
        started.set()
        assert release.wait(3)
        try:
            replies.append(original(context))
            return replies[-1]
        finally:
            done.set()

    runtime.tools.bind_handler("save_memory", delayed, replace=True)
    runner = OpenEtaEpisodeRunner(runtime=runtime, environment=DummyEpisodeEnvironment())
    runner.start(task="old", timeout_s=5, max_turns=2)
    failures = []

    def run():
        try:
            runner.continue_run()
        except RuntimeExecutionCancelled as exc:
            failures.append(exc)

    thread = threading.Thread(target=run)
    thread.start()
    try:
        assert started.wait(2)
        runner.interrupt(code="user_interrupt")
        thread.join(2)
        assert not thread.is_alive()
        assert runner.wait_for_idle(timeout_s=0.3)
        runtime.start_session(task="new", session_id="new")
        release.set()
        assert done.wait(2)
        assert not replies[0].success
        assert "late" not in runtime.memory.agent_working_state
        assert "late" not in runtime.memory.facts
    finally:
        release.set()
        thread.join(3)
