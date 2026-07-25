"""Reviewed promotion of runtime memory into repository memory."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from adapter.protocol import JsonDict
from agent.runtime.memory import AgentMemory


DEFAULT_PROMOTED_MEMORY_ROOT = Path(__file__).resolve().parents[1] / "memory"
DEFAULT_PROMOTED_MEMORY_FILE = "project_memory.md"


@dataclass(frozen=True, slots=True)
class PromotedMemoryResult:
    """Result of promoting reviewed working memory into `agent/memory`."""

    path: Path
    entry_id: str
    namespace: str
    key: str
    payload: JsonDict


class PromotedMemoryStore:
    """Append-only writer for reviewed project memory markdown files."""

    def __init__(self, root: Path | str = DEFAULT_PROMOTED_MEMORY_ROOT) -> None:
        self.root = Path(root)

    def promote(
        self,
        memory: AgentMemory,
        *,
        namespace: str,
        key: str,
        reviewer: str = "human",
        note: str = "",
        target: str = DEFAULT_PROMOTED_MEMORY_FILE,
    ) -> PromotedMemoryResult:
        namespace = namespace.strip()
        key = key.strip()
        if not namespace:
            raise ValueError("namespace is required")
        if not key and namespace != "compact_summary":
            raise ValueError("key is required")

        payload = select_promoted_memory_payload(memory, namespace=namespace, key=key)
        path = self._target_path(target)
        entry_id = _entry_id(namespace, key or "summary")
        entry = _format_promoted_entry(
            entry_id=entry_id,
            namespace=namespace,
            key=key or "summary",
            payload=payload,
            session_id=memory.session_id or "",
            reviewer=reviewer,
            note=note,
        )
        self.root.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("# OpenETA Promoted Memory\n\n", encoding="utf-8")
        with path.open("a", encoding="utf-8") as handle:
            handle.write(entry)
        memory.record(
            "memory_promoted",
            {
                "namespace": namespace,
                "key": key or "summary",
                "target": str(path),
                "entry_id": entry_id,
                "reviewer": reviewer,
            },
        )
        return PromotedMemoryResult(
            path=path,
            entry_id=entry_id,
            namespace=namespace,
            key=key or "summary",
            payload=payload,
        )

    def _target_path(self, target: str) -> Path:
        rel = Path(target or DEFAULT_PROMOTED_MEMORY_FILE)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("promoted memory target must stay under agent/memory")
        if rel.suffix != ".md":
            raise ValueError("promoted memory target must be a markdown file")
        root = self.root.resolve()
        path = (self.root / rel).resolve()
        if root != path and root not in path.parents:
            raise ValueError("promoted memory target must stay under agent/memory")
        return path


def select_promoted_memory_payload(
    memory: AgentMemory,
    *,
    namespace: str,
    key: str,
) -> JsonDict:
    """Select one reviewed memory item for promotion."""

    if namespace == "compact_summary":
        if not memory.compact_summary:
            raise KeyError("compact_summary is empty")
        return {"summary": memory.compact_summary}
    if namespace not in {"facts", "artifacts", "skill_notes"}:
        raise ValueError("namespace must be facts, artifacts, skill_notes, or compact_summary")
    selected = memory.get_memory(key, namespace=namespace).get(namespace)
    if not isinstance(selected, dict) or key not in selected:
        raise KeyError(f"memory item not found: {namespace}:{key}")
    return {key: selected[key]}


def _format_promoted_entry(
    *,
    entry_id: str,
    namespace: str,
    key: str,
    payload: JsonDict,
    session_id: str,
    reviewer: str,
    note: str,
) -> str:
    promoted_at = datetime.now(timezone.utc).isoformat()
    metadata = [
        f"- promoted_at: {promoted_at}",
        f"- namespace: {namespace}",
        f"- key: {key}",
        f"- source_session_id: {session_id or 'unknown'}",
        f"- reviewed_by: {reviewer or 'human'}",
    ]
    if note:
        metadata.append(f"- note: {note}")
    payload_text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    return (
        f"\n## {entry_id}\n\n"
        + "\n".join(metadata)
        + "\n\n```json\n"
        + payload_text
        + "\n```\n"
    )


def _entry_id(namespace: str, key: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_key = re.sub(r"[^a-zA-Z0-9_.-]+", "-", key.strip()).strip("-") or "summary"
    return f"{timestamp}-{namespace}-{safe_key}"
