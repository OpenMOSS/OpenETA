"""One authoritative, checksummed working-memory transaction per generation."""

from __future__ import annotations

import json
from pathlib import Path

from agent.runtime.memory_journal import MemoryStoreCorruption, atomic_write, digest, json_bytes

SCHEMA = "openeta.working_memory_snapshot.v1"
OBJECT_FIELDS = ("facts", "agent_working_state", "agent_artifacts", "artifacts", "skill_notes")


class MemoryStoreConflict(RuntimeError):
    """Another store committed since this instance last loaded its snapshot."""


def read_snapshot(working: Path, session_id: str) -> tuple[int, dict | None, dict]:
    path = working / "snapshot.json"
    if not path.exists():
        if (working / ".snapshot-managed").exists():
            raise MemoryStoreCorruption(f"Missing authoritative snapshot: {path}; legacy exports are not a fallback")
        return 0, None, {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise MemoryStoreCorruption(f"Invalid authoritative snapshot: {path}") from exc
    if not isinstance(value, dict):
        raise MemoryStoreCorruption(f"Invalid snapshot envelope: {path}")
    checked = {key: item for key, item in value.items() if key != "sha256"}
    generation = value.get("generation")
    memory = value.get("memory")
    cursors = value.get("journal_cursors")
    if (value.get("schema_version") != SCHEMA or value.get("session_id") != session_id
            or type(generation) is not int or generation < 1
            or not isinstance(memory, dict) or value.get("sha256") != digest(checked)
            or any(not isinstance(memory.get(key), dict) for key in OBJECT_FIELDS)
            or not isinstance(memory.get("compact_summary"), str)
            or not isinstance(cursors, dict)
            or any(type(cursors.get(key)) is not int or cursors[key] < 0
                   for key in ("trace", "conversation"))):
        raise MemoryStoreCorruption(f"Invalid snapshot identity, shape or checksum: {path}")
    return generation, memory, cursors


def write_snapshot(working: Path, session_id: str, memory: dict, *, generation: int,
                   journal_cursors: dict) -> None:
    envelope = {"schema_version": SCHEMA, "session_id": session_id,
                "generation": generation, "memory": memory, "journal_cursors": journal_cursors}
    envelope["sha256"] = digest(envelope)
    encoded = json_bytes(envelope) + b"\n"
    marker = working / ".snapshot-managed"
    if not marker.exists():
        # An interrupted first migration cannot silently restore stale legacy
        # exports. Originals remain available for explicit forensic recovery.
        atomic_write(marker, (SCHEMA + "\n").encode())
    atomic_write(working / "snapshot.json", encoded)
