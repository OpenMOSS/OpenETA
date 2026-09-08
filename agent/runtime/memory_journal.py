"""Durable JSON/JSONL primitives; callers serialize readers and writers.

Only an unterminated final record may be recovered. Complete malformed records,
sequence gaps, and checksum mismatches are corruption, never silently skipped.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4


class MemoryStoreCorruption(ValueError):
    pass


def json_bytes(value) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(json_bytes(value)).hexdigest()


def sync_directory(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def ensure_directory(path: Path) -> None:
    missing = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    path.mkdir(parents=True, exist_ok=True)
    for directory in reversed(missing):
        sync_directory(directory.parent)


def atomic_write(path: Path, data: bytes) -> None:
    """Unique same-directory temporary, fsync data, replace, fsync directory."""
    ensure_directory(path.parent)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass
class JournalState:
    count: int = 0
    valid_bytes: int = 0
    needs_newline: bool = False
    partial_tail: bytes = b""


def read_journal(path: Path, *, limit: int | None = None) -> tuple[list[dict], JournalState]:
    rows = deque(maxlen=limit if limit is not None and limit > 0 else None)
    state = JournalState()
    sequenced = False
    if not path.exists():
        return [], state
    with path.open("rb") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                state.valid_bytes += len(raw)
                state.needs_newline = not raw.endswith(b"\n")
                continue
            try:
                value = json.loads(raw.decode("utf-8"))
            except (ValueError, UnicodeError) as exc:
                if not raw.endswith(b"\n"):
                    state.partial_tail = raw
                    break
                raise MemoryStoreCorruption(f"Malformed journal record: {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise MemoryStoreCorruption(f"Non-object journal record: {path}:{line_number}")
            sequence = value.get("_store_seq")
            if sequence is not None:
                expected = state.count + 1
                if type(sequence) is not int or sequence != expected:
                    raise MemoryStoreCorruption(f"Journal sequence mismatch: {path}:{line_number}")
                checksum = value.get("_store_sha256")
                checked = {key: item for key, item in value.items() if key != "_store_sha256"}
                if checksum != digest(checked):
                    raise MemoryStoreCorruption(f"Journal checksum mismatch: {path}:{line_number}")
                sequenced = True
            elif sequenced or "_store_sha256" in value or "_store_seq" in value:
                raise MemoryStoreCorruption(f"Unsequenced record after journal migration: {path}:{line_number}")
            rows.append(value)
            state.count += 1
            state.valid_bytes += len(raw)
            state.needs_newline = not raw.endswith(b"\n")
    return list(rows), state


def read_recovery_fence(directory: Path) -> dict | None:
    fence = directory / "journal-recovery-required.json"
    if not fence.exists():
        return None
    try:
        report = json.loads(fence.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise MemoryStoreCorruption(f"Invalid journal recovery fence: {fence}") from exc
    if not isinstance(report, dict) or report.get("code") != "journal_partial_tail_recovered":
        raise MemoryStoreCorruption(f"Invalid journal recovery fence: {fence}")
    return report


def recover_tail(path: Path, state: JournalState) -> dict | None:
    """Preserve exact torn bytes and their offset durably before truncating."""
    if not state.partial_tail:
        return None
    read_recovery_fence(path.parent)
    recovery = path.parent / "recovery"
    recovery.mkdir(exist_ok=True)
    sync_directory(path.parent)
    tail_path = recovery / f"{path.name}.{uuid4().hex}.tail"
    report = {"code": "journal_partial_tail_recovered", "path": str(path),
              "offset": state.valid_bytes, "byte_count": len(state.partial_tail),
              "sha256": hashlib.sha256(state.partial_tail).hexdigest(),
              "quarantine_path": str(tail_path)}
    atomic_write(tail_path, state.partial_tail)
    atomic_write(tail_path.with_suffix(".json"), json_bytes(report) + b"\n")
    # A crash after truncation but before runtime epoch invalidation must not
    # erase the fact that history was incomplete. This sticky fence is retained
    # across resumes; it is not automatically cleared by a journal append.
    atomic_write(path.parent / "journal-recovery-required.json", json_bytes(report) + b"\n")
    with path.open("r+b") as handle:
        handle.truncate(state.valid_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    state.partial_tail = b""
    return report


def append_record(path: Path, value: dict, state: JournalState) -> None:
    row = {**value, "_store_seq": state.count + 1}
    row.pop("_store_sha256", None)
    row["_store_sha256"] = digest(row)
    encoded = (b"\n" if state.needs_newline else b"") + json_bytes(row) + b"\n"
    with path.open("ab") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    state.count += 1
    state.valid_bytes += len(encoded)
    state.needs_newline = False
