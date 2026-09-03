"""Load a staged observation-only context without importing simulator packages."""

from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

from agent.runtime.observation_interfaces import (
    ReadOnlyImageInput,
    ReadOnlyObservationContext,
    ReadOnlyObservationError,
    validate_readonly_context,
)

_EXPECTED_FILES = {
    "context.json",
    "images/head_rgb.png",
    "images/wrist_rgb.png",
    "images/left_tactile_rgb_marker.png",
    "images/right_tactile_rgb_marker.png",
}


def load_staged_observation_context(context_root: Path) -> ReadOnlyObservationContext:
    root = context_root.expanduser().resolve()
    files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if files != _EXPECTED_FILES or any(path.is_symlink() for path in root.rglob("*")):
        raise ReadOnlyObservationError("staged Agent input must contain exactly five regular files")
    payload = json.loads((root / "context.json").read_text(encoding="utf-8"))
    if set(payload) != {
        "task_instruction",
        "step_identifiers",
        "proprio",
        "visual_descriptors",
        "images",
    }:
        raise ReadOnlyObservationError("staged context fields do not match read-only schema")
    images: list[ReadOnlyImageInput] = []
    for record in payload["images"]:
        relative = PurePosixPath(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ReadOnlyObservationError("staged image path must remain relative")
        path = root.joinpath(*relative.parts).resolve(strict=True)
        if not path.is_relative_to(root):
            raise ReadOnlyObservationError("staged image escapes Agent input root")
        images.append(
            ReadOnlyImageInput(
                label=str(record["label"]),
                media_type=str(record["media_type"]),
                path=path,
                width=int(record["width"]),
                height=int(record["height"]),
                channels=int(record["channels"]),
                dtype=str(record["dtype"]),
            )
        )
    context = ReadOnlyObservationContext(
        instruction=str(payload["task_instruction"]),
        step_identifiers=dict(payload["step_identifiers"]),
        proprio=dict(payload["proprio"]),
        visual_descriptors=tuple(dict(item) for item in payload["visual_descriptors"]),
        images=tuple(images),
    )
    validate_readonly_context(context)
    return context
