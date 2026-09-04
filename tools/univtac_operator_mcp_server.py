#!/usr/bin/env python3
"""Expose one validated UniVTAC pre-action snapshot as a read-only MCP tool."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.types import Image
from mcp.types import TextContent, ToolAnnotations

from sim.envs.univtac.agent_context import (
    load_validated_pre_action_snapshot,
    project_operator_visible_context,
)

_CAMERA_LABELS = ("camera/head/rgb", "camera/wrist/rgb")
_TACTILE_LABELS = (
    "tactile/left_tactile/rgb_marker",
    "tactile/right_tactile/rgb_marker",
)
_STRUCTURED_SCALAR_FIELDS = {
    "active_pixel_ratio",
    "mean_absolute_change",
    "p95_absolute_change",
    "salient_mass",
}
_STRUCTURED_REGION_FIELDS = _STRUCTURED_SCALAR_FIELDS | {
    "normalized_horizontal_saliency"
}
_MCP_MODES = (
    "observe",
    "staged",
    "simultaneous_fusion",
    "image_first_no_commit",
    "image_first_with_commit",
)


def _validated_structured_summary(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("structured tactile summary must be a mapping")
    keys = set(payload)
    if keys == {"vector_order", "left_tactile", "right_tactile"}:
        expected_fields = _STRUCTURED_REGION_FIELDS
        if payload["vector_order"] != ["left", "center", "right"]:
            raise ValueError("structured tactile vector order is invalid")
    elif keys == {"left_tactile", "right_tactile"}:
        expected_fields = _STRUCTURED_SCALAR_FIELDS
    else:
        raise ValueError("structured tactile summary fields are invalid")
    for sensor in ("left_tactile", "right_tactile"):
        values = payload[sensor]
        if not isinstance(values, dict) or set(values) != expected_fields:
            raise ValueError(f"structured tactile fields are invalid for {sensor}")
        if "normalized_horizontal_saliency" in values:
            vector = values["normalized_horizontal_saliency"]
            if not isinstance(vector, list) or len(vector) != 3:
                raise ValueError("normalized horizontal saliency must contain three values")
            if any(not isinstance(value, (int, float)) for value in vector):
                raise ValueError("normalized horizontal saliency must be numeric")
        if any(
            not isinstance(values[field], (int, float))
            for field in expected_fields - {"normalized_horizontal_saliency"}
        ):
            raise ValueError("structured tactile metrics must be numeric")
    return payload


def _relative_to_episode(path: Path, episode_root: Path) -> str:
    resolved = path.resolve(strict=True)
    root = episode_root.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"MCP image is outside the episode root: {path}")
    return resolved.relative_to(root).as_posix()


def build_observe_blocks(
    *,
    episode_root: Path,
    snapshot_path: Path,
    condition_manifest: Path | None = None,
    include_structured_summary: bool = True,
) -> list[Any]:
    """Return the exact text and native image blocks visible to Codex."""

    root = episode_root.expanduser().resolve(strict=True)
    snapshot = snapshot_path.expanduser().resolve(strict=True)
    if condition_manifest is None and not snapshot.is_relative_to(root):
        raise ValueError("snapshot_pre.json must be inside the episode root")
    manifest: dict[str, Any] | None = None
    if condition_manifest is not None:
        manifest_path = condition_manifest.expanduser().resolve(strict=True)
        if not manifest_path.is_relative_to(root):
            raise ValueError("condition manifest must be inside the episode root")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        simulator_root = Path(str(manifest["simulator_root"])).resolve(strict=True)
    else:
        simulator_root = root / "simulator"
    operator_visible = load_validated_pre_action_snapshot(snapshot, simulator_root)
    context = project_operator_visible_context(operator_visible, simulator_root)
    images = list(context.images)
    if manifest is not None:
        requested = manifest.get("images")
        if not isinstance(requested, list):
            raise ValueError("condition manifest images must be a list")
        expected_labels = list(_CAMERA_LABELS)
        if len(requested) == 4:
            expected_labels.extend(_TACTILE_LABELS)
        elif len(requested) == 6:
            labels = [str(item.get("label")) for item in requested]
            if labels[:2] != list(_CAMERA_LABELS) or any(
                not label.startswith("tactile/") for label in labels[2:]
            ):
                raise ValueError("six-image condition labels are invalid")
            expected_labels = labels
        elif len(requested) != 2:
            raise ValueError("condition observe must return two, four, or six images")
        if [item.get("label") for item in requested] != expected_labels:
            raise ValueError("condition manifest image labels are invalid")
        originals = {image.label: image for image in context.images}
        selected = []
        for item in requested:
            staged = root / str(item["path"])
            if "shape" in item:
                shape = tuple(int(value) for value in item["shape"])
                if len(shape) != 3:
                    raise ValueError("condition image shape must be HxWxC")
                source = context.images[0]
                media_type = str(item.get("media_type", "image/png"))
                dtype = str(item.get("dtype", "uint8"))
                height, width, channels = shape
            else:
                source = originals[str(item["source_label"])]
                media_type = source.media_type
                dtype = source.dtype
                height, width, channels = source.height, source.width, source.channels
            selected.append(
                type(source)(
                    label=str(item["label"]),
                    media_type=media_type,
                    path=staged,
                    width=width,
                    height=height,
                    channels=channels,
                    dtype=dtype,
                )
            )
        images = selected
    text_payload = {
        "task_instruction": context.instruction,
        "step_identifiers": dict(context.step_identifiers),
        "proprio": dict(context.proprio),
        "images": [
            {
                "label": image.label,
                "shape": [image.height, image.width, image.channels],
                "dtype": image.dtype,
            }
            for image in images
        ],
    }
    if (
        include_structured_summary
        and manifest is not None
        and "model_visible_structured_tactile_summary" in manifest
    ):
        text_payload["tactile_change_summary"] = _validated_structured_summary(
            manifest["model_visible_structured_tactile_summary"]
        )
    return [
        TextContent(
            type="text",
            text=json.dumps(text_payload, ensure_ascii=False, separators=(",", ":")),
        ),
        *(Image(path=image.path) for image in images),
    ]


def record_operator_context(
    *,
    episode_root: Path,
    blocks: list[Any],
    timestamp_s: float | None = None,
) -> dict[str, Any]:
    """Persist the precise MCP projection without retaining image bytes."""

    root = episode_root.expanduser().resolve(strict=True)
    trace_path = root / "operator_context.jsonl"
    existing_rows = (
        [line for line in trace_path.read_text(encoding="utf-8").splitlines() if line]
        if trace_path.is_file()
        else []
    )
    if existing_rows:
        raise RuntimeError("observe may be called exactly once in this episode")
    return _append_operator_context(
        episode_root=root,
        tool="observe",
        arguments={},
        blocks=blocks,
        timestamp_s=timestamp_s,
    )


def _append_operator_context(
    *,
    episode_root: Path,
    tool: str,
    arguments: dict[str, Any],
    blocks: list[Any],
    timestamp_s: float | None = None,
) -> dict[str, Any]:
    root = episode_root.expanduser().resolve(strict=True)
    trace_path = root / "operator_context.jsonl"
    existing_rows = (
        [line for line in trace_path.read_text(encoding="utf-8").splitlines() if line]
        if trace_path.is_file()
        else []
    )
    row = {
        "seq": len(existing_rows) + 1,
        "timestamp_s": time.time() if timestamp_s is None else timestamp_s,
        "tool": tool,
        "arguments": arguments,
        "response_text_blocks": [
            block.text for block in blocks if isinstance(block, TextContent)
        ],
        "response_image_paths": [
            _relative_to_episode(Path(block.path), root)
            for block in blocks
            if isinstance(block, Image)
        ],
    }
    with trace_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return row


class StagedGroundingSession:
    """Enforce image-first commitment before revealing structured guidance."""

    def __init__(self, *, episode_root: Path, snapshot_path: Path, condition_manifest: Path) -> None:
        self.episode_root = episode_root.expanduser().resolve(strict=True)
        self.snapshot_path = snapshot_path.expanduser().resolve(strict=True)
        self.condition_manifest = condition_manifest.expanduser().resolve(strict=True)
        if not self.condition_manifest.is_relative_to(self.episode_root):
            raise ValueError("condition manifest must be inside the episode root")
        manifest = json.loads(self.condition_manifest.read_text(encoding="utf-8"))
        if "model_visible_structured_tactile_summary" not in manifest:
            raise ValueError("staged grounding requires structured tactile guidance")
        self.guidance = _validated_structured_summary(
            manifest["model_visible_structured_tactile_summary"]
        )
        self.state = "START"

    def observe_images(self) -> list[Any]:
        if self.state != "START":
            raise RuntimeError("observe_images is available exactly once at START")
        blocks = build_observe_blocks(
            episode_root=self.episode_root,
            snapshot_path=self.snapshot_path,
            condition_manifest=self.condition_manifest,
            include_structured_summary=False,
        )
        _append_operator_context(
            episode_root=self.episode_root,
            tool="observe_images",
            arguments={},
            blocks=blocks,
        )
        self.state = "IMAGES_OBSERVED"
        return blocks

    def record_image_judgment(
        self,
        *,
        tactile_image_access: Literal["available", "unavailable", "uncertain"],
        left_tactile_state: Literal[
            "clear_change", "little_or_no_change", "uncertain"
        ],
        right_tactile_state: Literal[
            "clear_change", "little_or_no_change", "uncertain"
        ],
        tactile_changed_side: Literal["left", "right", "both", "neither", "uncertain"],
        left_visual_cue: Literal[
            "localized_colored_disturbance",
            "regular_grid",
            "diffuse_or_ambiguous",
            "unavailable",
        ],
        right_visual_cue: Literal[
            "localized_colored_disturbance",
            "regular_grid",
            "diffuse_or_ambiguous",
            "unavailable",
        ],
        evidence_summary: str,
    ) -> list[Any]:
        if self.state != "IMAGES_OBSERVED":
            raise RuntimeError("record_image_judgment requires one completed observe_images call")
        if not evidence_summary.strip():
            raise ValueError("image judgment evidence_summary must be non-empty")
        judgment = {
            "tactile_image_access": tactile_image_access,
            "left_tactile_state": left_tactile_state,
            "right_tactile_state": right_tactile_state,
            "tactile_changed_side": tactile_changed_side,
            "left_visual_cue": left_visual_cue,
            "right_visual_cue": right_visual_cue,
            "evidence_summary": evidence_summary,
        }
        (self.episode_root / "image_judgment.json").write_text(
            json.dumps(judgment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        blocks = [
            TextContent(
                type="text",
                text=json.dumps(
                    {"status": "image_judgment_committed"},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )
        ]
        _append_operator_context(
            episode_root=self.episode_root,
            tool="record_image_judgment",
            arguments=judgment,
            blocks=blocks,
        )
        self.state = "IMAGE_JUDGMENT_COMMITTED"
        return blocks

    def observe_structured_guidance(self) -> list[Any]:
        if self.state != "IMAGE_JUDGMENT_COMMITTED":
            raise RuntimeError(
                "observe_structured_guidance requires a committed image judgment"
            )
        blocks = [
            TextContent(
                type="text",
                text=json.dumps(
                    {"tactile_change_summary": self.guidance},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )
        ]
        _append_operator_context(
            episode_root=self.episode_root,
            tool="observe_structured_guidance",
            arguments={},
            blocks=blocks,
        )
        self.state = "GUIDANCE_OBSERVED"
        return blocks


class ImageFirstNoCommitSession:
    """Reveal guidance only after images, without exposing a commitment tool."""

    def __init__(
        self, *, episode_root: Path, snapshot_path: Path, condition_manifest: Path
    ) -> None:
        self.episode_root = episode_root.expanduser().resolve(strict=True)
        self.snapshot_path = snapshot_path.expanduser().resolve(strict=True)
        self.condition_manifest = condition_manifest.expanduser().resolve(strict=True)
        if not self.condition_manifest.is_relative_to(self.episode_root):
            raise ValueError("condition manifest must be inside the episode root")
        manifest = json.loads(self.condition_manifest.read_text(encoding="utf-8"))
        if "model_visible_structured_tactile_summary" not in manifest:
            raise ValueError("image-first mode requires structured tactile guidance")
        self.guidance = _validated_structured_summary(
            manifest["model_visible_structured_tactile_summary"]
        )
        self.state = "START"

    def observe_images(self) -> list[Any]:
        if self.state != "START":
            raise RuntimeError("observe_images is available exactly once at START")
        blocks = build_observe_blocks(
            episode_root=self.episode_root,
            snapshot_path=self.snapshot_path,
            condition_manifest=self.condition_manifest,
            include_structured_summary=False,
        )
        _append_operator_context(
            episode_root=self.episode_root,
            tool="observe_images",
            arguments={},
            blocks=blocks,
        )
        self.state = "IMAGES_OBSERVED"
        return blocks

    def observe_structured_guidance(self) -> list[Any]:
        if self.state != "IMAGES_OBSERVED":
            raise RuntimeError(
                "observe_structured_guidance requires one completed observe_images call"
            )
        blocks = [
            TextContent(
                type="text",
                text=json.dumps(
                    {"tactile_change_summary": self.guidance},
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )
        ]
        _append_operator_context(
            episode_root=self.episode_root,
            tool="observe_structured_guidance",
            arguments={},
            blocks=blocks,
        )
        self.state = "GUIDANCE_OBSERVED"
        return blocks


def _validate_simultaneous_manifest(path: Path, episode_root: Path) -> None:
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_relative_to(episode_root):
        raise ValueError("condition manifest must be inside the episode root")
    manifest = json.loads(resolved.read_text(encoding="utf-8"))
    images = manifest.get("images")
    if not isinstance(images, list) or [item.get("label") for item in images] != [
        *_CAMERA_LABELS,
        *_TACTILE_LABELS,
    ]:
        raise ValueError("simultaneous fusion requires exactly four canonical images")
    if "model_visible_structured_tactile_summary" not in manifest:
        raise ValueError("simultaneous fusion requires structured tactile guidance")
    _validated_structured_summary(manifest["model_visible_structured_tactile_summary"])


def build_server(
    *,
    episode_root: Path,
    snapshot_path: Path,
    condition_manifest: Path | None = None,
    mode: Literal[
        "observe",
        "staged",
        "simultaneous_fusion",
        "image_first_no_commit",
        "image_first_with_commit",
    ] = "observe",
) -> FastMCP:
    root = episode_root.expanduser().resolve(strict=True)
    snapshot = snapshot_path.expanduser().resolve(strict=True)
    if mode not in _MCP_MODES:
        raise ValueError(f"unsupported UniVTAC MCP mode: {mode}")
    commit_mode = mode in {"staged", "image_first_with_commit"}
    no_commit_mode = mode == "image_first_no_commit"
    simultaneous_mode = mode == "simultaneous_fusion"
    if commit_mode:
        instructions = (
            "Use observe_images, record_image_judgment, then observe_structured_guidance."
        )
    elif no_commit_mode:
        instructions = "Use observe_images, then observe_structured_guidance."
    else:
        instructions = "One validated UniVTAC pre-action observation is available through observe."
    server = FastMCP("univtac-readonly", instructions=instructions, log_level="WARNING")

    if commit_mode:
        if condition_manifest is None:
            raise ValueError("staged mode requires a condition manifest")
        session = StagedGroundingSession(
            episode_root=root,
            snapshot_path=snapshot,
            condition_manifest=condition_manifest,
        )

        @server.tool(
            name="observe_images",
            description="Return task context and four native images without structured guidance.",
            annotations=ToolAnnotations(
                readOnlyHint=True, idempotentHint=False, destructiveHint=False
            ),
            structured_output=False,
        )
        def observe_images() -> list[Any]:
            return session.observe_images()

        @server.tool(
            name="record_image_judgment",
            description="Commit the image-only tactile judgment before guidance is available.",
            annotations=ToolAnnotations(
                readOnlyHint=True, idempotentHint=False, destructiveHint=False
            ),
            structured_output=False,
        )
        def record_image_judgment(
            tactile_image_access: Literal["available", "unavailable", "uncertain"],
            left_tactile_state: Literal[
                "clear_change", "little_or_no_change", "uncertain"
            ],
            right_tactile_state: Literal[
                "clear_change", "little_or_no_change", "uncertain"
            ],
            tactile_changed_side: Literal[
                "left", "right", "both", "neither", "uncertain"
            ],
            left_visual_cue: Literal[
                "localized_colored_disturbance",
                "regular_grid",
                "diffuse_or_ambiguous",
                "unavailable",
            ],
            right_visual_cue: Literal[
                "localized_colored_disturbance",
                "regular_grid",
                "diffuse_or_ambiguous",
                "unavailable",
            ],
            evidence_summary: str,
        ) -> list[Any]:
            return session.record_image_judgment(
                tactile_image_access=tactile_image_access,
                left_tactile_state=left_tactile_state,
                right_tactile_state=right_tactile_state,
                tactile_changed_side=tactile_changed_side,
                left_visual_cue=left_visual_cue,
                right_visual_cue=right_visual_cue,
                evidence_summary=evidence_summary,
            )

        @server.tool(
            name="observe_structured_guidance",
            description="Reveal structured tactile guidance after the image judgment is committed.",
            annotations=ToolAnnotations(
                readOnlyHint=True, idempotentHint=False, destructiveHint=False
            ),
            structured_output=False,
        )
        def observe_structured_guidance() -> list[Any]:
            return session.observe_structured_guidance()

        return server

    if no_commit_mode:
        if condition_manifest is None:
            raise ValueError("image-first no-commit mode requires a condition manifest")
        no_commit_session = ImageFirstNoCommitSession(
            episode_root=root,
            snapshot_path=snapshot,
            condition_manifest=condition_manifest,
        )

        @server.tool(
            name="observe_images",
            description="Return task context and four native images without structured guidance.",
            annotations=ToolAnnotations(
                readOnlyHint=True, idempotentHint=False, destructiveHint=False
            ),
            structured_output=False,
        )
        def observe_images() -> list[Any]:
            return no_commit_session.observe_images()

        @server.tool(
            name="observe_structured_guidance",
            description="Reveal structured tactile guidance after images were observed.",
            annotations=ToolAnnotations(
                readOnlyHint=True, idempotentHint=False, destructiveHint=False
            ),
            structured_output=False,
        )
        def observe_structured_guidance() -> list[Any]:
            return no_commit_session.observe_structured_guidance()

        return server

    if simultaneous_mode:
        if condition_manifest is None:
            raise ValueError("simultaneous fusion mode requires a condition manifest")
        _validate_simultaneous_manifest(condition_manifest, root)

    @server.tool(
        name="observe",
        description=(
            "Return the task instruction, step identifiers, proprioception, head and wrist RGB, "
            "and the available left and right tactile image evidence for this pre-action episode."
        ),
        annotations=ToolAnnotations(
            readOnlyHint=True,
            idempotentHint=True,
            destructiveHint=False,
        ),
        structured_output=False,
    )
    def observe() -> list[Any]:
        blocks = build_observe_blocks(
            episode_root=root,
            snapshot_path=snapshot,
            condition_manifest=condition_manifest,
        )
        record_operator_context(episode_root=root, blocks=blocks)
        return blocks

    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--snapshot-pre", type=Path, required=True)
    parser.add_argument("--condition-manifest", type=Path)
    parser.add_argument("--mode", choices=_MCP_MODES, default="observe")
    args = parser.parse_args(argv)
    server = build_server(
        episode_root=args.episode_root,
        snapshot_path=args.snapshot_pre,
        condition_manifest=args.condition_manifest,
        mode=args.mode,
    )
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
