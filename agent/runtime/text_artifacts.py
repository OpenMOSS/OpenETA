"""Materialize long textual payloads into local artifact files."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from adapter.protocol import JsonDict


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
    root = Path(output_root or DEFAULT_TEXT_ARTIFACT_OUTPUT_ROOT) / bundle
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
) -> JsonDict:
    """Search one materialized text artifact and return bounded matches."""

    text_path = Path(path)
    flags = re.IGNORECASE if ignore_case else 0
    regex = re.compile(pattern, flags)
    matches: list[JsonDict] = []
    lines = text_path.read_text(encoding="utf-8", errors="replace").splitlines()
    for lineno, line in enumerate(lines, 1):
        if regex.search(line):
            matches.append({"line": lineno, "text": line[:500]})
            if len(matches) >= max_matches:
                break
    return {
        "path": str(text_path),
        "pattern": pattern,
        "match_count": len(matches),
        "truncated": len(matches) >= max_matches,
        "matches": matches,
    }


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
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._-")
    return token[:80] or "item"


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
