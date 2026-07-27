"""Shared path ownership helpers for session-scoped runtime artifacts."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Any, Mapping


_MAX_COMPONENT_LENGTH = 96


def artifact_session_id(metadata: Mapping[str, Any] | None) -> str:
    """Return the local Agent session that owns artifacts for one tool call."""

    if not isinstance(metadata, Mapping):
        return ""
    return str(metadata.get("session_id") or "").strip()


def artifact_session_root(
    output_root: str | Path,
    session_id: str | None,
) -> Path:
    """Return an output root isolated to one Agent session when available."""

    root = Path(output_root)
    session = str(session_id or "").strip()
    if not session:
        return root
    return root / safe_artifact_component(session, fallback="session")


def safe_artifact_component(value: object, *, fallback: str = "artifact") -> str:
    """Convert an external identifier into one collision-resistant path component."""

    raw = str(value or "").strip()
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", raw).strip(".-")
    if not cleaned:
        cleaned = fallback
    changed = cleaned != raw or len(cleaned) > _MAX_COMPONENT_LENGTH
    if len(cleaned) > _MAX_COMPONENT_LENGTH:
        cleaned = cleaned[:_MAX_COMPONENT_LENGTH].rstrip(".-") or fallback
    if changed:
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:10]
        cleaned = f"{cleaned}-{digest}"
    return cleaned
