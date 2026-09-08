"""Materialize long textual payloads into local artifact files."""

from __future__ import annotations

import re
import hashlib
import json
import os
import stat
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from adapter.protocol import JsonDict
from agent.runtime.artifact_paths import artifact_session_root, safe_artifact_component


DEFAULT_TEXT_ARTIFACT_OUTPUT_ROOT = Path("tmp") / "tool_result" / "text"
DEFAULT_MAX_INLINE_TEXT_CHARS = 2000
DEFAULT_TEXT_PREVIEW_CHARS = 600


@dataclass(frozen=True, slots=True)
class TextArtifact:
    """Local reference for one materialized long text field."""

    index: str
    path: str
    chars: int
    preview: str
    grep_hint: str

    def to_dict(self) -> JsonDict:
        return {
            "type": "text",
            "index": self.index,
            "path": self.path,
            "chars": self.chars,
            "preview": self.preview,
            "grep_hint": self.grep_hint,
        }


@dataclass(frozen=True, slots=True)
class TextArtifactBundle:
    """Result from replacing long inline text fields with local references."""

    payload: JsonDict
    artifacts: list[TextArtifact]
    artifact_root: str
    bundle_id: str

    def to_dict(self) -> JsonDict:
        return {
            "payload": self.payload,
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
            "artifact_root": self.artifact_root,
            "bundle_id": self.bundle_id,
        }


def materialize_long_texts(
    payload: JsonDict,
    *,
    output_root: str | Path | None = None,
    bundle_id: str | None = None,
    max_inline_chars: int = DEFAULT_MAX_INLINE_TEXT_CHARS,
    preview_chars: int = DEFAULT_TEXT_PREVIEW_CHARS,
    session_id: str | None = None,
) -> TextArtifactBundle:
    """Write long strings to text files and return a lightweight payload.

    The returned payload keeps short previews inline and adds ``*_text_path``
    fields for dict values. List string entries are replaced by small reference
    objects. This keeps planner context searchable without embedding large MCP
    text responses directly.
    """

    if not isinstance(payload, dict):
        raise TypeError("materialize_long_texts expects a dict payload")
    bundle = _safe_token(bundle_id or str(uuid4()))
    root = artifact_session_root(
        output_root or DEFAULT_TEXT_ARTIFACT_OUTPUT_ROOT,
        session_id,
    ) / bundle
    artifacts: list[TextArtifact] = []
    scrubbed = _materialize_value(
        payload,
        root=root,
        path_parts=[],
        artifacts=artifacts,
        max_inline_chars=max_inline_chars,
        preview_chars=preview_chars,
    )
    return TextArtifactBundle(
        payload=scrubbed if isinstance(scrubbed, dict) else {"value": scrubbed},
        artifacts=artifacts,
        artifact_root=str(root.resolve()),
        bundle_id=bundle,
    )


def grep_text_artifact(
    path: str | Path,
    pattern: str,
    *,
    max_matches: int = 20,
    ignore_case: bool = True,
    cursor: JsonDict | None = None,
) -> JsonDict:
    """Page matching lines, with match-centred snippets and an N+1 lookahead."""

    _positive_limit(max_matches, "max_matches", 200)
    text_path = Path(path)
    flags = re.IGNORECASE if ignore_case else 0
    regex = re.compile(pattern, flags)
    query = hashlib.sha256(json.dumps([pattern, ignore_case]).encode()).hexdigest()
    matches: list[JsonDict] = []
    next_cursor = None
    with _text_snapshot(text_path) as (handle, version):
        offset = _cursor_offset(cursor, version)
        if cursor is not None and cursor.get("query") != query:
            raise ValueError("Grep cursor belongs to a different query")
        lineno, column = _skip_text(handle, offset)
        if column:
            raise ValueError("Grep cursor must point to the start of a line")
        for raw_line in handle:
            line = raw_line.removesuffix("\n")
            match = regex.search(line)
            if match is not None:
                if len(matches) == max_matches:
                    next_cursor = {"version": version, "char_offset": offset, "query": query}
                    break
                left = max(0, match.start() - 160)
                right = min(len(line), max(match.end(), match.start() + 1) + 160, left + 500)
                matches.append({
                    "line": lineno,
                    "text": line[left:right],
                    "snippet_start_column": left,
                    "snippet_clipped": left > 0 or right < len(line),
                    "match_start_column": match.start(),
                    "match_end_column": match.end(),
                    "match_clipped": match.end() > right,
                    "read_cursor": {"version": version, "char_offset": offset + match.start()},
                })
            offset += len(raw_line)
            lineno += 1
    return {
        "schema_version": "openeta.text_artifact_search.v1",
        "path": str(text_path),
        "pattern": pattern,
        "match_count": len(matches),
        "match_unit": "matching_line",
        "total_match_count": len(matches) if cursor is None and next_cursor is None else None,
        "truncated": next_cursor is not None,
        "next_cursor": next_cursor,
        "offset_unit": "unicode_codepoint_after_universal_newline_decoding",
        "matches": matches,
    }


def read_text_artifact(
    path: str | Path, *, max_chars: int = 1500, cursor: JsonDict | None = None,
) -> JsonDict:
    """Read a bounded page, including the interior of a single long JSONL line."""
    _positive_limit(max_chars, "max_chars", 1_000_000)
    text_path = Path(path)
    with _text_snapshot(text_path) as (handle, version):
        offset = _cursor_offset(cursor, version)
        line, column = _skip_text(handle, offset)
        page = handle.read(max_chars + 1)
        content = page[:max_chars]
        truncated = len(page) > max_chars
    return {
        "schema_version": "openeta.text_artifact_page.v1",
        "path": str(text_path),
        "text": content,
        "char_offset": offset,
        "line": line,
        "column": column,
        "chars_returned": len(content),
        "offset_unit": "unicode_codepoint_after_universal_newline_decoding",
        "truncated": truncated,
        "next_cursor": {"version": version, "char_offset": offset + len(content)} if truncated else None,
    }


def _positive_limit(value: int, name: str, maximum: int) -> None:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1, {maximum}]")


def _cursor_offset(cursor: JsonDict | None, version: str) -> int:
    if cursor is None:
        return 0
    if not isinstance(cursor, dict) or cursor.get("version") != version:
        raise ValueError("Artifact changed or cursor belongs to a different file; restart reading")
    offset = cursor.get("char_offset")
    if type(offset) is not int or offset < 0:
        raise ValueError("Cursor char_offset must be a non-negative integer")
    return offset


def _skip_text(handle, offset: int) -> tuple[int, int]:
    remaining, line, column = offset, 1, 0
    while remaining:
        chunk = handle.read(min(remaining, 65536))
        if not chunk:
            raise ValueError("Cursor exceeds the current file")
        remaining -= len(chunk)
        line += chunk.count("\n")
        column = len(chunk.rsplit("\n", 1)[-1]) if "\n" in chunk else column + len(chunk)
    return line, column


def _text_version(path: Path, snapshot: os.stat_result, digest: str) -> str:
    fields = [str(path.resolve()), snapshot.st_dev, snapshot.st_ino, snapshot.st_size,
              snapshot.st_mtime_ns, snapshot.st_ctime_ns, digest]
    return hashlib.sha256(json.dumps(fields).encode()).hexdigest()


def _text_digest(handle) -> str:
    # Metadata timestamps can have coarse resolution (including same-size writes).
    handle.seek(0)
    digest = hashlib.sha256()
    while chunk := handle.buffer.read(65536):
        digest.update(chunk)
    handle.seek(0)
    return digest.hexdigest()


@contextmanager
def _text_snapshot(path: Path):
    # Nonblocking open also lets us reject FIFOs without waiting for a writer.
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    with os.fdopen(fd, "r", encoding="utf-8", errors="replace", newline=None) as handle:
        before = os.fstat(handle.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError("Artifact text reader requires a regular file")
        version = _text_version(path, before, _text_digest(handle))
        yield handle, version
        after_digest = _text_digest(handle)
        if (version != _text_version(path, os.fstat(handle.fileno()), after_digest)
                or version != _text_version(path, path.stat(), after_digest)):
            raise ValueError("Artifact changed during reading; restart reading")


def _materialize_value(
    value: Any,
    *,
    root: Path,
    path_parts: list[str],
    artifacts: list[TextArtifact],
    max_inline_chars: int,
    preview_chars: int,
) -> Any:
    if isinstance(value, dict):
        return _materialize_dict(
            value,
            root=root,
            path_parts=path_parts,
            artifacts=artifacts,
            max_inline_chars=max_inline_chars,
            preview_chars=preview_chars,
        )
    if isinstance(value, list):
        return [
            _materialize_list_item(
                item,
                root=root,
                path_parts=[*path_parts, str(idx)],
                artifacts=artifacts,
                max_inline_chars=max_inline_chars,
                preview_chars=preview_chars,
            )
            for idx, item in enumerate(value)
        ]
    if (
        isinstance(value, str)
        and len(value) > max_inline_chars
        and _should_materialize_path(path_parts)
    ):
        artifact = _write_text_artifact(
            value,
            root=root,
            path_parts=path_parts or ["text"],
            artifacts=artifacts,
            preview_chars=preview_chars,
        )
        return _inline_text_preview(value, artifact=artifact, preview_chars=preview_chars)
    return value


def _materialize_list_item(
    value: Any,
    *,
    root: Path,
    path_parts: list[str],
    artifacts: list[TextArtifact],
    max_inline_chars: int,
    preview_chars: int,
) -> Any:
    if (
        isinstance(value, str)
        and len(value) > max_inline_chars
        and _should_materialize_path(path_parts)
    ):
        artifact = _write_text_artifact(
            value,
            root=root,
            path_parts=path_parts,
            artifacts=artifacts,
            preview_chars=preview_chars,
        )
        return {
            "text_preview": artifact.preview,
            "text_chars": artifact.chars,
            "text_ref": artifact.index,
            "text_path": artifact.path,
            "text_omitted": True,
            "grep_hint": artifact.grep_hint,
        }
    return _materialize_value(
        value,
        root=root,
        path_parts=path_parts,
        artifacts=artifacts,
        max_inline_chars=max_inline_chars,
        preview_chars=preview_chars,
    )


def _materialize_dict(
    value: JsonDict,
    *,
    root: Path,
    path_parts: list[str],
    artifacts: list[TextArtifact],
    max_inline_chars: int,
    preview_chars: int,
) -> JsonDict:
    payload = dict(value)
    for key, item in list(value.items()):
        key_str = str(key)
        parts = [*path_parts, key_str]
        if (
            isinstance(item, str)
            and len(item) > max_inline_chars
            and _should_materialize_text_key(key_str)
        ):
            artifact = _write_text_artifact(
                item,
                root=root,
                path_parts=parts,
                artifacts=artifacts,
                preview_chars=preview_chars,
            )
            payload[key] = _inline_text_preview(
                item,
                artifact=artifact,
                preview_chars=preview_chars,
            )
            payload[f"{key_str}_text_ref"] = artifact.index
            payload[f"{key_str}_text_path"] = artifact.path
            payload[f"{key_str}_text_chars"] = artifact.chars
            payload[f"{key_str}_text_omitted"] = True
            payload[f"{key_str}_grep_hint"] = artifact.grep_hint
            continue
        if isinstance(item, str):
            payload[key] = item
            continue
        payload[key] = _materialize_value(
            item,
            root=root,
            path_parts=parts,
            artifacts=artifacts,
            max_inline_chars=max_inline_chars,
            preview_chars=preview_chars,
        )
    return payload


def _write_text_artifact(
    text: str,
    *,
    root: Path,
    path_parts: list[str],
    artifacts: list[TextArtifact],
    preview_chars: int,
) -> TextArtifact:
    root.mkdir(parents=True, exist_ok=True)
    index = ".".join(_safe_token(part) for part in path_parts if part) or "text"
    path = root / f"{index}.txt"
    if path.exists():
        path = root / f"{index}.{len(artifacts):03d}.txt"
    path.write_text(text, encoding="utf-8")
    artifact = TextArtifact(
        index=index,
        path=str(path.resolve()),
        chars=len(text),
        preview=_preview(text, preview_chars),
        grep_hint=f"grep -n '<pattern>' {path.resolve()}",
    )
    artifacts.append(artifact)
    return artifact


def _inline_text_preview(text: str, *, artifact: TextArtifact, preview_chars: int) -> str:
    return (
        f"{_preview(text, preview_chars)}\n"
        f"...[truncated {len(text)} chars; full text saved to {artifact.path}; "
        f"use grep: {artifact.grep_hint}]"
    )


def _preview(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit)].rstrip()


def _safe_token(value: str) -> str:
    return safe_artifact_component(value, fallback="item")


def _should_materialize_text_key(key: str) -> bool:
    lowered = key.lower()
    return not (
        _looks_like_image_payload_key(lowered)
        or "base64" in lowered
        or lowered in {"rgb", "depth", "image", "pixels", "array"}
        or lowered.endswith("_base64_omitted")
        or lowered.endswith("_path")
        or lowered.endswith("_ref")
        or lowered.endswith("_chars")
        or lowered.endswith("_omitted")
        or lowered.endswith("_grep_hint")
        or lowered in {"path", "artifact_root", "bundle_id", "grep_hint"}
    )


def _should_materialize_path(path_parts: list[str]) -> bool:
    return not any(_looks_like_image_payload_key(str(part).lower()) for part in path_parts)


def _looks_like_image_payload_key(key: str) -> bool:
    lowered = key.lower()
    return (
        "base64" in lowered
        or lowered in {"rgb", "depth", "image", "pixels", "array"}
        or lowered.endswith("_base64")
        or lowered.endswith("_base64_omitted")
    )
