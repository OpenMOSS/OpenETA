"""Filesystem-backed memory stores for OpenETA agent runtime."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adapter.protocol import JsonDict


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

    def __init__(self, root: Path | str | None = None) -> None:
        self.config = JsonMemoryStoreConfig(root=root or ".openeta_memory")
        self.root = Path(self.config.root)
        self.sessions_dir = self.root / "sessions"
        self.index_path = self.root / "session_index.json"
        self.current_session_id: str | None = None
        self._migrate_legacy_layout()

    def start_session(
        self,
        *,
        session_id: str,
        task: str,
        metadata: JsonDict | None = None,
    ) -> None:
        self.current_session_id = session_id
        self.session_dir(session_id).mkdir(parents=True, exist_ok=True)
        self.working_dir_for(session_id).mkdir(parents=True, exist_ok=True)
        self.session_path(session_id).touch(exist_ok=True)
        self._upsert_index_entry(
            session_id=session_id,
            task=task,
            metadata=metadata or {},
            status="active",
        )

    def append_event(self, event: Any) -> None:
        if self.current_session_id is None:
            return
        self.session_dir(self.current_session_id).mkdir(parents=True, exist_ok=True)
        payload = {
            "event_type": str(getattr(event, "event_type")),
            "timestamp_s": float(getattr(event, "timestamp_s")),
            "payload": dict(getattr(event, "payload")),
        }
        with self.session_path(self.current_session_id).open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
        self._touch_index_entry(self.current_session_id, event=payload)

    def load_working_memory(self) -> JsonDict:
        if self.current_session_id is None:
            return _empty_working_memory()
        return {
            "facts": self._read_json_object("facts.json"),
            "artifacts": self._read_json_object("artifacts.json"),
            "skill_notes": self._read_json_object("skill_notes.json"),
            "compact_summary": self._read_compact_summary(),
        }

    def save_working_memory(self, memory: Any) -> None:
        if self.current_session_id is None:
            return
        self.working_dir.mkdir(parents=True, exist_ok=True)
        self._write_json("facts.json", dict(getattr(memory, "facts", {})))
        self._write_json("artifacts.json", dict(getattr(memory, "artifacts", {})))
        self._write_json("skill_notes.json", dict(getattr(memory, "skill_notes", {})))
        self._write_json(
            "compact_summary.json",
            {"summary": str(getattr(memory, "compact_summary", ""))},
        )
        self._touch_index_entry(self.current_session_id)

    @property
    def working_dir(self) -> Path:
        if self.current_session_id is None:
            return self.root / "sessions" / "(no-session)" / "working"
        return self.working_dir_for(self.current_session_id)

    def session_path(self, session_id: str) -> Path:
        return self.session_dir(session_id) / "trace.jsonl"

    def session_dir(self, session_id: str) -> Path:
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
        path = self.session_path(session_id)
        if not path.exists():
            return []
        rows: list[JsonDict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            if isinstance(data, dict):
                rows.append(data)
        if limit is not None and limit > 0:
            return rows[-limit:]
        return rows

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

    def _read_json_object(self, filename: str) -> JsonDict:
        path = self.working_dir / filename
        if not path.exists():
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}

    def _read_compact_summary(self) -> str:
        path = self.working_dir / "compact_summary.json"
        if not path.exists():
            return ""
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return str(data.get("summary", ""))
        if isinstance(data, str):
            return data
        return ""

    def _write_json(self, filename: str, payload: JsonDict) -> None:
        path = self.working_dir / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(path)

    def _read_index(self) -> JsonDict:
        if not self.index_path.exists():
            return {"sessions": {}}
        data = json.loads(self.index_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return {"sessions": {}}
        sessions = data.get("sessions")
        if not isinstance(sessions, dict):
            data["sessions"] = {}
        return data

    def _write_index(self, index: JsonDict) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        tmp_path = self.index_path.with_suffix(self.index_path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(index, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(self.index_path)

    def _upsert_index_entry(
        self,
        *,
        session_id: str,
        task: str,
        metadata: JsonDict,
        status: str,
    ) -> None:
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
                "working_dir": str(self.working_dir_for(session_id)),
            }
        )
        sessions[session_id] = entry
        self._write_index(index)

    def _read_trace_events(self, path: Path) -> list[JsonDict]:
        events: list[JsonDict] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                events.append(data)
        return events

    def _unique_path(self, path: Path) -> Path:
        if not path.exists():
            return path
        for idx in range(1, 1000):
            candidate = path.with_name(f"{path.name}-{idx}")
            if not candidate.exists():
                return candidate
        raise RuntimeError(f"could not find free legacy archive path under {path.parent}")

    def _touch_index_entry(self, session_id: str, *, event: JsonDict | None = None) -> None:
        index = self._read_index()
        sessions = index.setdefault("sessions", {})
        if not isinstance(sessions, dict):
            return
        entry = sessions.get(session_id)
        if not isinstance(entry, dict):
            return
        entry["updated_at_s"] = time.time()
        entry["session_path"] = str(self.session_path(session_id))
        entry["working_dir"] = str(self.working_dir_for(session_id))
        if event is not None:
            entry["event_count"] = int(entry.get("event_count") or 0) + 1
            payload = event.get("payload")
            if isinstance(payload, dict):
                preview = payload.get("task") or payload.get("type") or payload.get("event_type")
                if isinstance(preview, str) and preview.strip():
                    entry["preview"] = preview.strip()[:160]
        sessions[session_id] = entry
        self._write_index(index)


def _empty_working_memory() -> JsonDict:
    return {
        "facts": {},
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
