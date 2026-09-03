"""Project UniVTAC operator-visible snapshots into staged read-only Agent input."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
from PIL import Image

from agent.runtime.observation_interfaces import (
    EXPECTED_IMAGE_LABELS,
    ReadOnlyImageInput,
    ReadOnlyObservationContext,
    validate_readonly_context,
)
from sim.envs.univtac.contract import (
    UniVTACContractError,
    UniVTACTaskSnapshot,
    validate_operator_visible,
)

_IMAGE_SPECS = (
    ("camera/head/rgb", "cameras", "head", "rgb", "head_rgb.png", (270, 480, 3)),
    ("camera/wrist/rgb", "cameras", "wrist", "rgb", "wrist_rgb.png", (270, 480, 3)),
    (
        "tactile/left_tactile/rgb_marker",
        "tactile",
        "left_tactile",
        "rgb_marker",
        "left_tactile_rgb_marker.png",
        (240, 320, 3),
    ),
    (
        "tactile/right_tactile/rgb_marker",
        "tactile",
        "right_tactile",
        "rgb_marker",
        "right_tactile_rgb_marker.png",
        (240, 320, 3),
    ),
)
_EXPECTED_STAGED_FILES = {
    "context.json",
    "images/head_rgb.png",
    "images/wrist_rgb.png",
    "images/left_tactile_rgb_marker.png",
    "images/right_tactile_rgb_marker.png",
}


def load_validated_pre_action_snapshot(
    snapshot_path: Path,
    artifact_root: Path,
) -> dict[str, Any]:
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise UniVTACContractError("snapshot JSON must contain an object")
    snapshot = UniVTACTaskSnapshot.from_dict(payload)
    if snapshot.phase != "pre_action":
        raise UniVTACContractError("read-only Agent handoff requires a pre_action snapshot")
    validate_operator_visible(snapshot.operator_visible)
    project_operator_visible_context(snapshot.operator_visible, artifact_root)
    return snapshot.operator_visible


def _resolve_visible_png(artifact_root: Path, descriptor: Mapping[str, Any]) -> Path:
    relative_text = str(descriptor.get("path", ""))
    relative = PurePosixPath(relative_text)
    if not relative_text or relative.is_absolute() or ".." in relative.parts:
        raise UniVTACContractError(f"operator artifact path must be relative: {relative_text!r}")
    root = artifact_root.expanduser().resolve()
    candidate = root.joinpath(*relative.parts)
    if candidate.is_symlink():
        raise UniVTACContractError(f"operator artifact must not be a symlink: {relative_text}")
    try:
        resolved = candidate.resolve(strict=True)
    except FileNotFoundError as exc:
        raise UniVTACContractError(f"operator artifact is missing: {relative_text}") from exc
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise UniVTACContractError(f"operator artifact escapes its root: {relative_text}")
    return resolved


def project_operator_visible_context(
    operator_visible: Mapping[str, Any],
    artifact_root: Path,
) -> ReadOnlyObservationContext:
    validate_operator_visible(operator_visible)
    if set(operator_visible["cameras"]) != {"head", "wrist"}:
        raise UniVTACContractError("read-only gate requires exactly head and wrist cameras")
    if set(operator_visible["tactile"]) != {"left_tactile", "right_tactile"}:
        raise UniVTACContractError("read-only gate requires exactly two named tactile sensors")
    images: list[ReadOnlyImageInput] = []
    descriptors: list[dict[str, Any]] = []
    for label, group, name, field, _, expected_shape in _IMAGE_SPECS:
        descriptor = operator_visible[group][name][field]
        path = _resolve_visible_png(artifact_root, descriptor)
        if descriptor.get("encoding") != "png" or descriptor.get("stored_dtype") != "uint8":
            raise UniVTACContractError(f"{label} is not stored as uint8 PNG")
        with Image.open(path) as image:
            image.load()
            array = np.asarray(image)
            if image.mode != "RGB" or array.dtype != np.uint8:
                raise UniVTACContractError(f"{label} must decode as RGB uint8")
        if tuple(array.shape) != expected_shape:
            raise UniVTACContractError(
                f"{label} expected decoded shape {expected_shape}, got {tuple(array.shape)}"
            )
        if tuple(descriptor.get("shape", ())) != expected_shape:
            raise UniVTACContractError(f"{label} descriptor shape does not match decoded PNG")
        images.append(
            ReadOnlyImageInput(
                label=label,
                media_type="image/png",
                path=path,
                width=expected_shape[1],
                height=expected_shape[0],
                channels=expected_shape[2],
                dtype="uint8",
            )
        )
        descriptors.append(
            {
                "label": label,
                "shape": list(expected_shape),
                "dtype": "uint8",
                "value_range": list(descriptor["value_range"]),
                "media_type": "image/png",
            }
        )
    context = ReadOnlyObservationContext(
        instruction=str(operator_visible["task_instruction"]),
        step_identifiers=dict(operator_visible["step_identifiers"]),
        proprio=dict(operator_visible["proprio"]),
        visual_descriptors=tuple(descriptors),
        images=tuple(images),
    )
    validate_readonly_context(context)
    return context


def materialize_operator_visible_input(
    context: ReadOnlyObservationContext,
    context_root: Path,
) -> ReadOnlyObservationContext:
    validate_readonly_context(context)
    context_root = context_root.expanduser().resolve()
    if context_root.exists():
        raise UniVTACContractError(f"Agent input directory must be fresh: {context_root}")
    images_dir = context_root / "images"
    images_dir.mkdir(parents=True, mode=0o700)
    os.chmod(context_root, 0o700)
    staged_images: list[ReadOnlyImageInput] = []
    image_records: list[dict[str, Any]] = []
    for image, spec in zip(context.images, _IMAGE_SPECS, strict=True):
        filename = spec[4]
        destination = images_dir / filename
        shutil.copyfile(image.path, destination)
        staged_images.append(
            ReadOnlyImageInput(
                label=image.label,
                media_type=image.media_type,
                path=destination,
                width=image.width,
                height=image.height,
                channels=image.channels,
                dtype=image.dtype,
            )
        )
        image_records.append(
            {
                "label": image.label,
                "media_type": image.media_type,
                "path": f"images/{filename}",
                "width": image.width,
                "height": image.height,
                "channels": image.channels,
                "dtype": image.dtype,
            }
        )
    payload = {
        "task_instruction": context.instruction,
        "step_identifiers": dict(context.step_identifiers),
        "proprio": dict(context.proprio),
        "visual_descriptors": [dict(item) for item in context.visual_descriptors],
        "images": image_records,
    }
    (context_root / "context.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    validate_agent_input_directory(context_root)
    staged = ReadOnlyObservationContext(
        instruction=context.instruction,
        step_identifiers=dict(context.step_identifiers),
        proprio=dict(context.proprio),
        visual_descriptors=tuple(dict(item) for item in context.visual_descriptors),
        images=tuple(staged_images),
    )
    validate_readonly_context(staged)
    return staged


def validate_agent_input_directory(context_root: Path) -> None:
    root = context_root.expanduser().resolve()
    files = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if files != _EXPECTED_STAGED_FILES:
        raise UniVTACContractError(
            f"Agent input directory must contain exactly five files, got {sorted(files)}"
        )
    if any(path.is_symlink() for path in root.rglob("*")):
        raise UniVTACContractError("Agent input directory must not contain symlinks")


def load_materialized_operator_context(context_root: Path) -> ReadOnlyObservationContext:
    validate_agent_input_directory(context_root)
    root = context_root.expanduser().resolve()
    payload = json.loads((root / "context.json").read_text(encoding="utf-8"))
    if set(payload) != {
        "task_instruction",
        "step_identifiers",
        "proprio",
        "visual_descriptors",
        "images",
    }:
        raise UniVTACContractError("staged context fields do not match the read-only contract")
    images: list[ReadOnlyImageInput] = []
    for record in payload["images"]:
        relative = PurePosixPath(record["path"])
        path = root.joinpath(*relative.parts).resolve(strict=True)
        if not path.is_relative_to(root):
            raise UniVTACContractError("staged image escapes Agent input root")
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


assert tuple(spec[0] for spec in _IMAGE_SPECS) == EXPECTED_IMAGE_LABELS
