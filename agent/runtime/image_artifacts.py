"""Materialize MCP image payloads into local artifact files."""

from __future__ import annotations

import base64
import binascii
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from adapter.protocol import JsonDict

DEFAULT_MCP_IMAGE_OUTPUT_ROOT = Path("tmp") / "image"

_MCP_IMAGE_FIELDS = {
    "rgb_base64": "rgb",
    "depth_base64": "depth",
    "image_base64": "image",
}


@dataclass(slots=True)
class ImageArtifact:
    """Local reference for one materialized MCP image field."""

    index: str
    path: str
    kind: str
    source_field: str
    format: str = "png"
    frame_id: str = ""
    width: int | None = None
    height: int | None = None
    byte_size: int = 0

    def to_dict(self) -> JsonDict:
        payload: JsonDict = {
            "index": self.index,
            "path": self.path,
            "kind": self.kind,
            "source_field": self.source_field,
            "format": self.format,
            "byte_size": self.byte_size,
        }
        if self.frame_id:
            payload["frame_id"] = self.frame_id
        if self.width is not None:
            payload["width"] = self.width
        if self.height is not None:
            payload["height"] = self.height
        return payload


@dataclass(slots=True)
class MaterializedImageBundle:
    """Result from replacing inline MCP base64 images with local references."""

    bundle_id: str
    artifact_root: str
    payload: JsonDict
    images: list[ImageArtifact] = field(default_factory=list)

    def to_dict(self) -> JsonDict:
        return {
            "bundle_id": self.bundle_id,
            "artifact_root": self.artifact_root,
            "payload": self.payload,
            "images": [image.to_dict() for image in self.images],
        }


def materialize_mcp_images(
    payload: JsonDict,
    *,
    output_root: str | Path = DEFAULT_MCP_IMAGE_OUTPUT_ROOT,
    bundle_id: str | None = None,
) -> MaterializedImageBundle:
    """Write MCP base64 image fields to disk and return a scrubbed payload.

    The simulator MCP transport uses compact base64 fields such as
    ``rgb_base64`` and ``depth_base64``. Those fields are useful on the wire but
    too large for planner context. This helper keeps the payload shape intact
    while replacing each image field with a stable local reference.
    """

    if not isinstance(payload, dict):
        raise TypeError("materialize_mcp_images expects a dict payload")

    resolved_bundle_id = bundle_id or _new_bundle_id()
    root = Path(output_root)
    images: list[ImageArtifact] = []
    scrubbed = _materialize_value(
        json.loads(json.dumps(payload)),
        root=root,
        bundle_id=resolved_bundle_id,
        path_parts=[],
        images=images,
    )
    return MaterializedImageBundle(
        bundle_id=resolved_bundle_id,
        artifact_root=str(root.resolve()),
        payload=scrubbed if isinstance(scrubbed, dict) else {},
        images=images,
    )


def _materialize_value(
    value: Any,
    *,
    root: Path,
    bundle_id: str,
    path_parts: list[str],
    images: list[ImageArtifact],
) -> Any:
    if isinstance(value, dict):
        return _materialize_dict(
            value,
            root=root,
            bundle_id=bundle_id,
            path_parts=path_parts,
            images=images,
        )
    if isinstance(value, list):
        return [
            _materialize_value(
                item,
                root=root,
                bundle_id=bundle_id,
                path_parts=[*path_parts, str(idx)],
                images=images,
            )
            for idx, item in enumerate(value)
        ]
    return value


def _materialize_dict(
    payload: JsonDict,
    *,
    root: Path,
    bundle_id: str,
    path_parts: list[str],
    images: list[ImageArtifact],
) -> JsonDict:
    frame_id = str(payload.get("frame_id") or payload.get("camera") or "")
    if not frame_id and path_parts:
        frame_id = path_parts[-1]

    for field, kind in _MCP_IMAGE_FIELDS.items():
        encoded = payload.get(field)
        if not isinstance(encoded, str) or not encoded.strip():
            continue
        fmt = _format_for_field(payload, field)
        index = _image_index(path_parts=path_parts, frame_id=frame_id, kind=kind)
        path = root / kind / bundle_id / f"{_safe_filename(index)}.{fmt}"
        data = _decode_base64_image(encoded)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

        artifact = ImageArtifact(
            index=index,
            path=str(path.resolve()),
            kind=kind,
            source_field=field,
            format=fmt,
            frame_id=frame_id,
            width=_optional_int(payload.get("width")),
            height=_optional_int(payload.get("height")),
            byte_size=len(data),
        )
        images.append(artifact)
        payload.pop(field, None)
        payload[f"{kind}_ref"] = artifact.index
        payload[f"{kind}_path"] = artifact.path
        payload[f"{field}_omitted"] = True

    for key, value in list(payload.items()):
        payload[key] = _materialize_value(
            value,
            root=root,
            bundle_id=bundle_id,
            path_parts=[*path_parts, str(key)],
            images=images,
        )
    return payload


def _decode_base64_image(value: str) -> bytes:
    encoded = value.strip()
    if "," in encoded and encoded.lower().startswith("data:"):
        encoded = encoded.split(",", 1)[1]
    try:
        return base64.b64decode(encoded, validate=True)
    except binascii.Error as exc:
        raise ValueError("invalid base64 image payload") from exc


def _format_for_field(payload: JsonDict, field: str) -> str:
    explicit = payload.get("format") or payload.get(f"{field}_format")
    fmt = str(explicit or "png").lower().lstrip(".")
    if fmt in {"jpg", "jpeg"}:
        return "jpg"
    if fmt == "png":
        return "png"
    return "bin"


def _image_index(*, path_parts: list[str], frame_id: str, kind: str) -> str:
    prefix = ".".join(_safe_token(part) for part in path_parts if part)
    frame = _safe_token(frame_id) if frame_id else "image"
    if prefix:
        return f"{prefix}.{frame}.{kind}"
    return f"{frame}.{kind}"


def _safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "image"


def _safe_token(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
    return token or "item"


def _optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _new_bundle_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{stamp}-{uuid4().hex[:8]}"
