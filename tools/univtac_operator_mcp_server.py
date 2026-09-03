#!/usr/bin/env python3
"""Expose one validated UniVTAC pre-action snapshot as a read-only MCP tool."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.types import Image
from mcp.types import TextContent, ToolAnnotations

from sim.envs.univtac.agent_context import (
    load_validated_pre_action_snapshot,
    project_operator_visible_context,
)


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
) -> list[Any]:
    """Return the exact text and four native image blocks visible to Codex."""

    root = episode_root.expanduser().resolve(strict=True)
    snapshot = snapshot_path.expanduser().resolve(strict=True)
    if not snapshot.is_relative_to(root):
        raise ValueError("snapshot_pre.json must be inside the episode root")
    simulator_root = root / "simulator"
    operator_visible = load_validated_pre_action_snapshot(snapshot, simulator_root)
    context = project_operator_visible_context(operator_visible, simulator_root)
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
            for image in context.images
        ],
    }
    return [
        TextContent(
            type="text",
            text=json.dumps(text_payload, ensure_ascii=False, separators=(",", ":")),
        ),
        *(Image(path=image.path) for image in context.images),
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
    text_blocks = [block.text for block in blocks if isinstance(block, TextContent)]
    image_paths = [
        _relative_to_episode(Path(block.path), root) for block in blocks if isinstance(block, Image)
    ]
    row = {
        "seq": 1,
        "timestamp_s": time.time() if timestamp_s is None else timestamp_s,
        "tool": "observe",
        "arguments": {},
        "response_text_blocks": text_blocks,
        "response_image_paths": image_paths,
    }
    with trace_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    return row


def build_server(*, episode_root: Path, snapshot_path: Path) -> FastMCP:
    root = episode_root.expanduser().resolve(strict=True)
    snapshot = snapshot_path.expanduser().resolve(strict=True)
    server = FastMCP(
        "univtac-readonly",
        instructions="One validated UniVTAC pre-action observation is available through observe.",
        log_level="WARNING",
    )

    @server.tool(
        name="observe",
        description=(
            "Return the task instruction, step identifiers, proprioception, head and wrist RGB, "
            "and left and right tactile rgb_marker images for this pre-action episode."
        ),
        annotations=ToolAnnotations(
            readOnlyHint=True,
            idempotentHint=True,
            destructiveHint=False,
        ),
        structured_output=False,
    )
    def observe() -> list[Any]:
        blocks = build_observe_blocks(episode_root=root, snapshot_path=snapshot)
        record_operator_context(episode_root=root, blocks=blocks)
        return blocks

    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--snapshot-pre", type=Path, required=True)
    args = parser.parse_args(argv)
    server = build_server(
        episode_root=args.episode_root,
        snapshot_path=args.snapshot_pre,
    )
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
