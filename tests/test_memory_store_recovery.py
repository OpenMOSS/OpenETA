from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from adapter.protocol import EnvAction

from agent.runtime.memory import AgentMemory
from agent.runtime.memory_journal import MemoryStoreCorruption, read_journal
from agent.runtime.memory_snapshot import MemoryStoreConflict, OBJECT_FIELDS, read_snapshot
from agent.runtime.memory_store import JsonMemoryStore


def content(value):
    return SimpleNamespace(**{key: {"generation_value": value} for key in OBJECT_FIELDS},
                           compact_summary=str(value))


def started(root):
    store = JsonMemoryStore(root)
    store.start_session(session_id="session", task="recover")
    return store


def event(index):
    return SimpleNamespace(event_type="fixture", timestamp_s=float(index), payload={"index": index})


def assert_generation(memory, expected):
    assert all(memory[key] == {"generation_value": expected} for key in OBJECT_FIELDS)
    assert memory["compact_summary"] == str(expected)


def test_snapshot_transaction_remains_complete_when_export_update_fails(tmp_path, monkeypatch):
    store = started(tmp_path)
    store.save_working_memory(content(1))
    original = store._write_json

    def fail(filename, payload, **kwargs):
        if filename == "agent_working_state.json":
            raise OSError("injected export failure")
        original(filename, payload, **kwargs)

    monkeypatch.setattr(store, "_write_json", fail)
    store.save_working_memory(content(2))
    # This deliberately leaves mixed compatibility exports, not a mixed restore.
    assert json.loads((store.working_dir / "facts.json").read_text())["generation_value"] == 2
    assert json.loads((store.working_dir / "agent_working_state.json").read_text())["generation_value"] == 1
    assert store.recovery_reports[-1]["code"] == "snapshot_export_failed"
    assert_generation(started(tmp_path).load_working_memory(), 2)


@pytest.mark.parametrize("stage,expected", [("before", 1), ("after", 2)])
def test_real_process_exit_at_snapshot_publication_never_exposes_mixed_generation(tmp_path, stage, expected):
    store = started(tmp_path)
    store.save_working_memory(content(1))
    script = """
import os, sys
from pathlib import Path
from types import SimpleNamespace
from agent.runtime.memory_store import JsonMemoryStore
from agent.runtime.memory_snapshot import OBJECT_FIELDS
store = JsonMemoryStore(sys.argv[1])
store.start_session(session_id='session', task='recover')
original = Path.replace
def replace(path, target):
    if Path(target).name == 'snapshot.json':
        if sys.argv[2] == 'before': os._exit(73)
        original(path, target)
        os._exit(73)
    return original(path, target)
Path.replace = replace
store.save_working_memory(SimpleNamespace(**{key: {'generation_value': 2} for key in OBJECT_FIELDS}, compact_summary='2'))
"""
    completed = subprocess.run([sys.executable, "-c", script, str(tmp_path), stage],
                               cwd=Path(__file__).resolve().parents[1], timeout=10, capture_output=True)
    assert completed.returncode == 73, completed.stderr.decode()
    assert_generation(started(tmp_path).load_working_memory(), expected)


def test_corrupt_or_missing_managed_snapshot_never_falls_back_to_exports(tmp_path):
    store = started(tmp_path)
    store.save_working_memory(content(1))
    path = store.working_dir / "snapshot.json"
    original = path.read_bytes()
    value = json.loads(original)
    value["memory"]["facts"]["generation_value"] = 99
    path.write_text(json.dumps(value))
    with pytest.raises(MemoryStoreCorruption, match="checksum"):
        started(tmp_path)
    path.write_bytes(original)
    path.unlink()
    with pytest.raises(MemoryStoreCorruption, match="not a fallback"):
        started(tmp_path)


def test_interrupted_first_snapshot_commit_preserves_legacy_data_but_fails_closed(tmp_path, monkeypatch):
    from agent.runtime import memory_snapshot
    store = started(tmp_path)
    legacy = store.working_dir / "facts.json"
    legacy.write_text('{"old": {"epoch": 7}}')
    original = memory_snapshot.atomic_write

    def fail(path, data):
        if path.name == "snapshot.json":
            raise OSError("injected failure")
        original(path, data)

    monkeypatch.setattr(memory_snapshot, "atomic_write", fail)
    with pytest.raises(OSError):
        store.save_working_memory(content(1))
    assert legacy.read_text() == '{"old": {"epoch": 7}}'
    with pytest.raises(MemoryStoreCorruption, match="Missing authoritative snapshot"):
        started(tmp_path)


def test_legacy_snapshot_migrates_once_then_exports_lose_restore_authority(tmp_path):
    store = started(tmp_path)
    (store.working_dir / "facts.json").write_text('{"legacy": 3}')
    assert store.load_working_memory()["facts"] == {"legacy": 3}
    store.save_working_memory(content(2))
    (store.working_dir / "facts.json").write_text('{"forged": 99}')
    assert_generation(started(tmp_path).load_working_memory(), 2)


def test_stale_store_cannot_overwrite_a_newer_snapshot(tmp_path):
    first = started(tmp_path)
    first.save_working_memory(content(1))
    second = started(tmp_path)
    first.save_working_memory(content(2))
    with pytest.raises(MemoryStoreConflict):
        second.save_working_memory(content(3))
    assert_generation(second.load_working_memory(), 2)
    second.save_working_memory(content(3))
    assert_generation(first.load_working_memory(), 3)


def test_lockless_snapshot_readers_always_observe_one_complete_generation(tmp_path):
    store = started(tmp_path)
    store.save_working_memory(content(0))

    def write():
        for number in range(1, 12):
            store.save_working_memory(content(number))

    with ThreadPoolExecutor(max_workers=2) as pool:
        writer = pool.submit(write)
        for _ in range(120):
            generation, memory, _ = read_snapshot(store.working_dir, "session")
            assert_generation(memory, generation - 1)
        writer.result(timeout=10)


@pytest.mark.parametrize("tail", [b'{"event_type":', b'{"text":"\xe6\x9d'])
def test_partial_tail_reads_are_nonmutating_and_append_quarantines_exact_bytes(tmp_path, tail):
    store = started(tmp_path)
    for number in range(3):
        store.append_event(event(number))
    path = store.session_path("session")
    valid = path.read_bytes()
    path.write_bytes(valid + tail)
    assert len(store.load_events("session", limit=1)) == 1
    assert path.read_bytes() == valid + tail
    assert store.recovery_reports[-1]["code"] == "journal_partial_tail_detected"
    store.append_event(event(3))
    rows = store.load_events("session")
    assert [row["_store_seq"] for row in rows] == [1, 2, 3, 4]
    assert path.read_bytes().startswith(valid)
    report = next(report for report in store.recovery_reports if report["code"].endswith("recovered"))
    assert Path(report["quarantine_path"]).read_bytes() == tail
    assert report["offset"] == len(valid)
    assert (path.parent / "journal-recovery-required.json").is_file()


@pytest.mark.parametrize("suffix", [b'not-json\n', b'{}\nnot-json\n{}\n', b'[]\n'])
def test_complete_or_middle_corruption_is_not_silently_skipped(tmp_path, suffix):
    store = started(tmp_path)
    path = store.session_path("session")
    path.write_bytes(b'{"event_type":"legacy"}\n' + suffix)
    before = path.read_bytes()
    with pytest.raises(MemoryStoreCorruption):
        store.load_events("session")
    with pytest.raises(MemoryStoreCorruption):
        store.append_event(event(0))
    assert path.read_bytes() == before


@pytest.mark.parametrize("operation", ["checksum", "gap", "legacy_after_sequence"])
def test_journal_sequence_and_checksum_detect_valid_json_corruption(tmp_path, operation):
    store = started(tmp_path)
    for number in range(3):
        store.append_event(event(number))
    path = store.session_path("session")
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if operation == "checksum":
        rows[1]["payload"]["index"] = 99
    elif operation == "gap":
        rows.pop(1)
    else:
        rows.append({"event_type": "legacy"})
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(MemoryStoreCorruption):
        store.load_events("session")


def test_valid_legacy_record_without_newline_is_preserved_on_append(tmp_path):
    store = started(tmp_path)
    path = store.session_path("session")
    original = b'{"event_type":"legacy"}'
    path.write_bytes(original)
    store.append_event(event(0))
    assert path.read_bytes().startswith(original + b"\n")
    assert store.load_events("session")[-1]["_store_seq"] == 2


def test_conversation_uses_same_strict_recovery_rules_as_trace(tmp_path):
    store = started(tmp_path)
    store.append_conversation_record({"record_type": "fixture"})
    path = store.conversation_path("session")
    valid = path.read_bytes()
    path.write_bytes(valid + b'{"record_type":')
    assert len(store.load_conversation_records("session")) == 1
    store.append_conversation_record({"record_type": "next"})
    assert len(store.load_conversation_records("session")) == 2
    path.write_bytes(valid + b'invalid\n')
    with pytest.raises(MemoryStoreCorruption):
        store.load_conversation_records("session")


def test_concurrent_store_instances_append_contiguous_records(tmp_path):
    stores = [started(tmp_path) for _ in range(4)]

    def write(item):
        index, store = item
        for number in range(12):
            store.append_event(event(index * 12 + number))

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(write, enumerate(stores)))
    rows = stores[0].load_events("session")
    assert [row["_store_seq"] for row in rows] == list(range(1, 49))
    assert {row["payload"]["index"] for row in rows} == set(range(48))


def test_resume_preserves_valid_history_and_invalidates_epochs_after_torn_trace(tmp_path):
    memory = AgentMemory(store=JsonMemoryStore(tmp_path))
    memory.start_session(task="keep original instruction", session_id="session")
    memory.save_fact("robot_motion_epoch", {"epoch": 7}, source="runtime")
    path = memory.store.session_path("session")
    path.write_bytes(path.read_bytes() + b'{"event_type":"action",')
    resumed = AgentMemory(store=JsonMemoryStore(tmp_path))
    resumed.resume_session("session")
    assert resumed.current_user_request == "keep original instruction"
    assert resumed.robot_motion_epoch() == 8
    assert any(event.event_type == "session_journal_recovery" for event in resumed.events)
    assert not any(event.event_type == "world_epochs_advanced"
                   and event.payload.get("tool") == "session_recovery" for event in resumed.events)
    assert not read_journal(path)[1].partial_tail
    # The durable fence also protects a second resume, including a process loss
    # between journal repair and the runtime's epoch snapshot commit.
    again = AgentMemory(store=JsonMemoryStore(tmp_path))
    again.resume_session("session")
    assert again.robot_motion_epoch() == 9


def test_repair_does_not_touch_either_journal_if_other_has_middle_corruption(tmp_path):
    store = started(tmp_path)
    trace = store.session_path("session")
    conversation = store.conversation_path("session")
    trace.write_bytes(b'{"torn":')
    conversation.write_bytes(b'invalid\n')
    with pytest.raises(MemoryStoreCorruption):
        store.recover_session_journals("session")
    assert trace.read_bytes() == b'{"torn":'
    assert not (trace.parent / "recovery").exists()


@pytest.mark.parametrize("journal", ["trace", "conversation"])
def test_deleted_complete_tail_is_detected_by_snapshot_cursor(tmp_path, journal):
    store = started(tmp_path)
    if journal == "trace":
        store.append_event(event(0))
        path = store.session_path("session")
    else:
        store.append_conversation_record({"record_type": "fixture"})
        path = store.conversation_path("session")
    store.save_working_memory(content(1))
    path.write_bytes(b"")
    with pytest.raises(MemoryStoreCorruption, match="shorter than committed snapshot"):
        store.load_working_memory()
    with pytest.raises(MemoryStoreCorruption, match="shorter than committed snapshot"):
        store.save_working_memory(content(2))
    with pytest.raises(MemoryStoreCorruption, match="shorter than committed snapshot"):
        store.recover_session_journals("session")


def test_complete_journal_ahead_of_checkpoint_invalidates_old_epoch_on_resume(tmp_path):
    memory = AgentMemory(store=JsonMemoryStore(tmp_path))
    memory.start_session(task="recover", session_id="session")
    memory.save_fact("robot_motion_epoch", {"epoch": 7}, source="runtime")
    memory.record("uncheckpointed_action", {"outcome": "unknown"})
    resumed = AgentMemory(store=JsonMemoryStore(tmp_path))
    resumed.resume_session("session")
    assert resumed.robot_motion_epoch() == 8
    report = next(event for event in resumed.events if event.event_type == "session_journal_recovery")
    assert any(row["code"] == "journal_ahead_of_snapshot" for row in report.payload["reports"])


def test_recovery_fence_survives_process_loss_before_epoch_checkpoint(tmp_path):
    memory = AgentMemory(store=JsonMemoryStore(tmp_path))
    memory.start_session(task="recover", session_id="session")
    memory.save_fact("robot_motion_epoch", {"epoch": 7}, source="runtime")
    path = memory.store.session_path("session")
    path.write_bytes(path.read_bytes() + b'{"torn":')
    memory.store.recover_session_journals("session")
    # Drop all in-memory state without recording/snapshotting new epochs.
    del memory
    resumed = AgentMemory(store=JsonMemoryStore(tmp_path))
    resumed.resume_session("session")
    assert resumed.robot_motion_epoch() == 8


def test_same_store_append_does_not_rescan_whole_trace_each_time(tmp_path, monkeypatch):
    from agent.runtime import memory_store
    store = started(tmp_path)
    original = memory_store.read_journal
    calls = []

    def read(path, **kwargs):
        calls.append(path)
        return original(path, **kwargs)

    monkeypatch.setattr(memory_store, "read_journal", read)
    for number in range(15):
        store.append_event(event(number))
    assert calls == [store.session_path("session")]


@pytest.mark.parametrize("broken", [b'{"sessions":', b'[]', b'{"sessions": []}'])
def test_corrupt_session_index_is_not_silently_replaced(tmp_path, broken):
    store = started(tmp_path)
    store.index_path.write_bytes(broken)
    with pytest.raises(MemoryStoreCorruption):
        store.list_sessions()
    with pytest.raises(MemoryStoreCorruption):
        store.start_session(session_id="other", task="do not discard index")
    assert store.index_path.read_bytes() == broken


def test_recovered_trace_cannot_reauthorize_a_precrash_feasible_ik_seed(tmp_path):
    memory = AgentMemory(store=JsonMemoryStore(tmp_path))
    memory.start_session(task="reach", session_id="session")
    parameters = {"target_pose": {"frame": "world", "xyz": [0.1, -0.2, 0.3],
                                   "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
                  "orientation_tolerance_rad": 0.3}
    memory.add_action(EnvAction(action_type="tool_call", command={
        "status": "executed",
        "request": {"kind": "tool_call", "name": "ik_preview_check", "parameters": parameters},
        "tool_calls": [{"name": "ik_preview_check", "status": "executed", "parameters": parameters,
                        "result": {"success": True, "details": {"outputs": {"ik_preview_receipt": {
                            "schema_version": "openeta.ik_preview_receipt.v1",
                            "receipt_id": "precrash", "classification": "feasible",
                            "orientation_policy": "explicit_orientation",
                            "tolerances": {"position_tolerance_m": 0.002, "orientation_tolerance_rad": 0.3},
                            "best_candidate": {"joint_positions": [0.1] * 7},
                        }}}}}],
    }))
    assert memory.ik_execution_gate_error(tool_name="move_to", parameters=parameters) is None
    assert memory.resolve_ik_execution_seed(parameters)["receipt_id"] == "precrash"
    memory._save_working_memory()
    path = memory.store.session_path("session")
    path.write_bytes(path.read_bytes() + b'{"torn":')
    resumed = AgentMemory(store=JsonMemoryStore(tmp_path))
    resumed.resume_session("session")
    assert resumed.ik_execution_gate_error(tool_name="move_to", parameters=parameters) is not None
    assert resumed.resolve_ik_execution_seed(parameters) is None


def test_invalid_existing_recovery_fence_is_not_overwritten_by_another_tail_repair(tmp_path):
    store = started(tmp_path)
    path = store.session_path("session")
    path.write_bytes(b'{"torn":')
    fence = path.parent / "journal-recovery-required.json"
    fence.write_bytes(b'corrupt-fence')
    with pytest.raises(MemoryStoreCorruption, match="recovery fence"):
        store.recover_session_journals("session")
    with pytest.raises(MemoryStoreCorruption, match="recovery fence"):
        store.append_event(event(0))
    assert path.read_bytes() == b'{"torn":'
    assert fence.read_bytes() == b'corrupt-fence'
