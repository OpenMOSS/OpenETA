"""Filesystem-backed memory stores for OpenETA agent runtime."""

from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from weakref import WeakValueDictionary

from adapter.protocol import JsonDict
from agent.runtime.memory_journal import (
    MemoryStoreCorruption, append_record, atomic_write, ensure_directory, json_bytes, read_journal,
    read_recovery_fence, recover_tail, sync_directory,
)
from agent.runtime.memory_snapshot import (
    MemoryStoreConflict, OBJECT_FIELDS, read_snapshot, write_snapshot,
)

try:
    import fcntl
except ImportError:  # pragma: no cover - OpenETA deployment targets are POSIX.
    fcntl = None


@dataclass(frozen=True, slots=True)
class JsonMemoryStoreConfig:
    """Configuration for the local JSON/JSONL memory store."""

    root: Path | str = ".openeta_memory"


class JsonMemoryStore:
    """Persist session trace and working memory under a per-session directory.

    The store is intentionally narrow: session events are append-only local
    runtime state, while curated project memory remains a separate explicit
    artifact under ``agent/memory``.
    """

    _index_thread_lock = threading.RLock()
    _session_locks_guard = threading.Lock()
    _session_locks = WeakValueDictionary()

    def __init__(self, root: Path | str | None = None) -> None:
        self.config = JsonMemoryStoreConfig(root=root or ".openeta_memory")
        self.root = Path(self.config.root)
        self.sessions_dir = self.root / "sessions"
        self.index_path = self.root / "session_index.json"
        self.current_session_id: str | None = None
        self.recovery_reports: list[JsonDict] = []
        self._snapshot_generations: dict[str, int] = {}
        self._journal_cache: dict[Path, tuple[tuple, Any]] = {}
        self._migrate_legacy_layout()

    def start_session(
        self,
        *,
        session_id: str,
        task: str,
        metadata: JsonDict | None = None,
    ) -> None:
        with self._locked_session(session_id):
            self.current_session_id = session_id
            self.working_dir_for(session_id).mkdir(parents=True, exist_ok=True)
            generation, _, _ = read_snapshot(self.working_dir_for(session_id), session_id)
            self._snapshot_generations[session_id] = generation
            self.session_path(session_id).touch(exist_ok=True)
            sync_directory(self.session_dir(session_id))
        self._upsert_index_entry(
            session_id=session_id,
            task=task,
            metadata=metadata or {},
            status="active",
        )

    def append_event(self, event: Any) -> None:
        if self.current_session_id is None:
            return
        session_id = self.current_session_id
        payload = {
            "event_type": str(getattr(event, "event_type")),
            "timestamp_s": float(getattr(event, "timestamp_s")),
            "payload": dict(getattr(event, "payload")),
        }
        self._append_journal(session_id, self.session_path(session_id), payload)
        self._touch_index_entry(session_id, event=payload)

    def append_conversation_record(self, record: JsonDict) -> None:
        if self.current_session_id is None:
            return
        session_id = self.current_session_id
        self._append_journal(session_id, self.conversation_path(session_id), record)
        self._touch_index_entry(session_id)

    def load_conversation_records(self, session_id: str) -> list[JsonDict]:
        return self._load_journal(session_id, self.conversation_path(session_id))

    def load_working_memory(self) -> JsonDict:
        if self.current_session_id is None:
            return _empty_working_memory()
        session_id = self.current_session_id
        working = self.working_dir_for(session_id)
        with self._locked_session(session_id):
            generation, memory, cursors = read_snapshot(working, session_id)
            self._validate_checkpoint_journals(session_id, cursors)
            self._snapshot_generations[session_id] = generation
            if memory is not None:
                return memory
            return {**{key: self._read_json_object(f"{key}.json", working=working) for key in OBJECT_FIELDS},
                    "compact_summary": self._read_compact_summary(working=working)}

    def save_working_memory(self, memory: Any) -> None:
        if self.current_session_id is None:
            return
        session_id = self.current_session_id
        working = self.working_dir_for(session_id)
        with self._locked_session(session_id):
            generation, _, cursors = read_snapshot(working, session_id)
            self._validate_checkpoint_journals(session_id, cursors)
            if self._snapshot_generations.get(session_id) != generation:
                raise MemoryStoreConflict("Snapshot changed in another store; reload before saving")
            # Serialize to a detached tree before publication or export writes.
            payload = json.loads(json_bytes({
                **{key: dict(getattr(memory, key, {})) for key in OBJECT_FIELDS},
                "compact_summary": str(getattr(memory, "compact_summary", "")),
            }))
            cursors = {name: self._journal_state(path).count for name, path in (
                ("trace", self.session_path(session_id)),
                ("conversation", self.conversation_path(session_id)),
            )}
            write_snapshot(working, session_id, payload, generation=generation + 1,
                           journal_cursors=cursors)
            self._snapshot_generations[session_id] = generation + 1
            # Compatibility exports are for inspection only. They never provide
            # restore authority once snapshot.json has been published.
            try:
                for key in OBJECT_FIELDS:
                    self._write_json(f"{key}.json", payload[key], working=working)
                self._write_json("compact_summary.json", {"summary": payload["compact_summary"]}, working=working)
            except OSError as exc:
                self._report({"code": "snapshot_export_failed", "session_id": session_id,
                              "generation": generation + 1, "message": str(exc)})
        self._touch_index_entry(session_id)

    @property
    def working_dir(self) -> Path:
        if self.current_session_id is None:
            return self.root / "sessions" / "(no-session)" / "working"
        return self.working_dir_for(self.current_session_id)

    def session_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "trace.jsonl"

    def conversation_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "conversation.jsonl"

    def session_dir(self, session_id: str) -> Path:
        if (not session_id or session_id in {".", ".."} or Path(session_id).name != session_id
                or "/" in session_id or "\\" in session_id):
            raise ValueError("session_id must be one directory name")
        return self.sessions_dir / session_id

    def working_dir_for(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "working"

    def list_sessions(self) -> list[JsonDict]:
        index = self._read_index()
        sessions = list(index.get("sessions", {}).values())
        sessions.sort(key=lambda item: float(item.get("updated_at_s") or 0.0), reverse=True)
        return sessions

    def session_exists(self, session_id: str) -> bool:
        return self.session_path(session_id).exists()

    def load_session_metadata(self, session_id: str) -> JsonDict:
        sessions = self._read_index().get("sessions", {})
        entry = sessions.get(session_id)
        return dict(entry) if isinstance(entry, dict) else {}

    def load_events(self, session_id: str, *, limit: int | None = None) -> list[JsonDict]:
        return self._load_journal(session_id, self.session_path(session_id), limit=limit)

    def recover_session_journals(self, session_id: str) -> list[JsonDict]:
        """Validate both journals before repairing either; never used by queries."""
        reports = []
        with self._locked_session(session_id):
            report = read_recovery_fence(self.session_dir(session_id))
            if report is not None:
                reports.append(report)
            states = [(path, read_journal(path, limit=1)[1]) for path in
                      (self.session_path(session_id), self.conversation_path(session_id))]
            _, _, cursors = read_snapshot(self.working_dir_for(session_id), session_id)
            for name, (path, state) in zip(("trace", "conversation"), states):
                if name in cursors and state.count < cursors[name]:
                    raise MemoryStoreCorruption(f"Journal shorter than committed snapshot: {path}")
                if name in cursors and state.count > cursors[name]:
                    reports.append({"code": "journal_ahead_of_snapshot", "path": str(path),
                                    "snapshot_sequence": cursors[name], "journal_sequence": state.count})
            for path, state in states:
                report = recover_tail(path, state)
                self._journal_cache.pop(path, None)
                if report:
                    self._report(report)
                    reports.append(report)
        return reports

    @contextmanager
    def _locked_session(self, session_id: str):
        directory = self.session_dir(session_id)
        key = str(directory.resolve())
        with self._session_locks_guard:
            lock = self._session_locks.setdefault(key, threading.RLock())
        with lock:
            ensure_directory(directory)
            with (directory / ".store.lock").open("a+b") as handle:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _report(self, report: JsonDict) -> None:
        if report not in self.recovery_reports:
            self.recovery_reports.append(report)
            logging.getLogger(__name__).warning("Memory store recovery: %s", report)

    def _load_journal(self, session_id: str, path: Path, *, limit=None) -> list[JsonDict]:
        with self._locked_session(session_id):
            rows, state = read_journal(path, limit=limit)
            if state.partial_tail:
                self._report({"code": "journal_partial_tail_detected", "path": str(path),
                              "offset": state.valid_bytes, "byte_count": len(state.partial_tail)})
            return rows

    @staticmethod
    def _journal_signature(path: Path) -> tuple:
        if not path.exists():
            return ()
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns

    def _journal_state(self, path: Path):
        signature = self._journal_signature(path)
        cached = self._journal_cache.get(path)
        if cached is not None and cached[0] == signature:
            return cached[1]
        _, state = read_journal(path, limit=1)
        self._journal_cache[path] = signature, state
        return state

    def _validate_checkpoint_journals(self, session_id: str, cursors: JsonDict) -> None:
        for name, path in (("trace", self.session_path(session_id)),
                           ("conversation", self.conversation_path(session_id))):
            if name in cursors and self._journal_state(path).count < cursors[name]:
                raise MemoryStoreCorruption(f"Journal shorter than committed snapshot: {path}")

    def _append_journal(self, session_id: str, path: Path, payload: JsonDict) -> None:
        with self._locked_session(session_id):
            signature = self._journal_signature(path)
            state = self._journal_state(path)
            self._journal_cache.pop(path, None)
            report = recover_tail(path, state)
            if report:
                self._report(report)
            append_record(path, payload, state)
            if not signature:
                sync_directory(path.parent)
            self._journal_cache[path] = self._journal_signature(path), state

    def _migrate_legacy_layout(self) -> None:
        """Move pre-session-scoped memory files into the current layout."""
        if not self.root.exists():
            return

        legacy_session_files = (
            sorted(self.sessions_dir.glob("*.jsonl")) if self.sessions_dir.exists() else []
        )
        for legacy_path in legacy_session_files:
            session_id = legacy_path.stem
            target_path = self.session_path(session_id)
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if target_path.exists():
                archive_path = self._unique_path(
                    self.root / "legacy" / "sessions" / legacy_path.name
                )
                archive_path.parent.mkdir(parents=True, exist_ok=True)
                legacy_path.replace(archive_path)
                continue
            legacy_path.replace(target_path)
            self._upsert_legacy_session_index(
                session_id=session_id,
                trace_path=target_path,
                legacy_path=legacy_path,
            )

        legacy_working_dir = self.root / "working"
        if legacy_working_dir.exists() and legacy_working_dir.is_dir():
            archive_dir = self._unique_path(
                self.root / "legacy" / "working" / str(int(time.time()))
            )
            archive_dir.parent.mkdir(parents=True, exist_ok=True)
            legacy_working_dir.replace(archive_dir)

    def _read_json_object(self, filename: str, *, working: Path | None = None) -> JsonDict:
        path = (working or self.working_dir) / filename
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise MemoryStoreCorruption(f"Legacy memory object has invalid shape: {path}")
        return data

    def _read_compact_summary(self, *, working: Path | None = None) -> str:
        path = (working or self.working_dir) / "compact_summary.json"
        if not path.exists():
            return ""
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return str(data.get("summary", ""))
        if isinstance(data, str):
            return data
        raise MemoryStoreCorruption(f"Legacy summary has invalid shape: {path}")

    def _write_json(self, filename: str, payload: JsonDict, *, working: Path | None = None) -> None:
        path = (working or self.working_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(path, json_bytes(payload) + b"\n")

    def _read_index(self) -> JsonDict:
        if not self.index_path.exists():
            return {"sessions": {}}
        try:
            data = json.loads(self.index_path.read_text(encoding="utf-8"))
        except (ValueError, UnicodeError) as exc:
            raise MemoryStoreCorruption(f"Invalid session index: {self.index_path}") from exc
        if (not isinstance(data, dict) or not isinstance(data.get("sessions"), dict)
                or any(not isinstance(entry, dict) for entry in data["sessions"].values())):
            raise MemoryStoreCorruption(f"Invalid session index shape: {self.index_path}")
        return data

    def _write_index(self, index: JsonDict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        atomic_write(self.index_path, json_bytes(index) + b"\n")

    @contextmanager
    def _locked_index(self):
        self.root.mkdir(parents=True, exist_ok=True)
        lock_path = self.root / ".session_index.lock"
        with self._index_thread_lock:
            with lock_path.open("a+", encoding="utf-8") as lock_file:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _upsert_index_entry(
        self,
        *,
        session_id: str,
        task: str,
        metadata: JsonDict,
        status: str,
    ) -> None:
        with self._locked_index():
            now = time.time()
            index = self._read_index()
            sessions = index.setdefault("sessions", {})
            if not isinstance(sessions, dict):
                sessions = {}
                index["sessions"] = sessions
            current = sessions.get(session_id)
            entry = dict(current) if isinstance(current, dict) else {}
            entry.setdefault("created_at_s", now)
            entry.update(
                {
                    "session_id": session_id,
                    "task": task,
                    "metadata": metadata,
                    "status": status,
                    "updated_at_s": now,
                    "session_path": str(self.session_path(session_id)),
                    "conversation_path": str(self.conversation_path(session_id)),
                    "working_dir": str(self.working_dir_for(session_id)),
                }
            )
            sessions[session_id] = entry
            self._write_index(index)

    def _upsert_legacy_session_index(
        self,
        *,
        session_id: str,
        trace_path: Path,
        legacy_path: Path,
    ) -> None:
        events = self._read_trace_events(trace_path)
        first_event = events[0] if events else {}
        last_event = events[-1] if events else {}
        task = _payload_string(first_event, "task") or _payload_string(last_event, "task")
        created_at = _event_timestamp(first_event) or time.time()
        updated_at = _event_timestamp(last_event) or created_at

        with self._locked_index():
            index = self._read_index()
            sessions = index.setdefault("sessions", {})
            if not isinstance(sessions, dict):
                sessions = {}
                index["sessions"] = sessions
            current = sessions.get(session_id)
            entry = dict(current) if isinstance(current, dict) else {}
            entry.setdefault("created_at_s", created_at)
            entry.update(
                {
                    "session_id": session_id,
                    "task": task,
                    "metadata": {
                        **dict(entry.get("metadata") or {}),
                        "migrated_from_layout": str(legacy_path),
                    },
                    "status": entry.get("status") or "migrated",
                    "updated_at_s": updated_at,
                    "event_count": len(events),
                    "session_path": str(trace_path),
                    "conversation_path": str(self.conversation_path(session_id)),
                    "working_dir": str(self.working_dir_for(session_id)),
                }
            )
            sessions[session_id] = entry
            self._write_index(index)

    def _read_trace_events(self, path: Path) -> list[JsonDict]:
        return self._load_journal(path.parent.name, path)

    def _unique_path(self, path: Path) -> Path:
        if not path.exists():
            return path
        for idx in range(1, 1000):
            candidate = path.with_name(f"{path.name}-{idx}")
            if not candidate.exists():
                return candidate
        raise RuntimeError(f"could not find free legacy archive path under {path.parent}")

    def _touch_index_entry(self, session_id: str, *, event: JsonDict | None = None) -> None:
        with self._locked_index():
            index = self._read_index()
            sessions = index.setdefault("sessions", {})
            if not isinstance(sessions, dict):
                return
            entry = sessions.get(session_id)
            if not isinstance(entry, dict):
                return
            entry["updated_at_s"] = time.time()
            entry["session_path"] = str(self.session_path(session_id))
            entry["conversation_path"] = str(self.conversation_path(session_id))
            entry["working_dir"] = str(self.working_dir_for(session_id))
            if event is not None:
                entry["event_count"] = int(entry.get("event_count") or 0) + 1
                payload = event.get("payload")
                if isinstance(payload, dict):
                    preview = (
                        payload.get("task")
                        or payload.get("type")
                        or payload.get("event_type")
                    )
                    if isinstance(preview, str) and preview.strip():
                        entry["preview"] = preview.strip()[:160]
            sessions[session_id] = entry
            self._write_index(index)


def _empty_working_memory() -> JsonDict:
    return {
        "facts": {},
        "agent_working_state": {},
        "agent_artifacts": {},
        "artifacts": {},
        "skill_notes": {},
        "compact_summary": "",
    }


def _event_timestamp(event: JsonDict) -> float | None:
    value = event.get("timestamp_s")
    if isinstance(value, int | float):
        return float(value)
    return None


def _payload_string(event: JsonDict, key: str) -> str:
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return ""
    value = payload.get(key)
    return value if isinstance(value, str) else ""
