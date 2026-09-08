#!/usr/bin/env python3
"""Expose one live UniVTAC episode through three bounded MCP tools."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.types import Image
from mcp.types import TextContent, ToolAnnotations

from sim.envs.univtac.live_operation import MCP_TOOL_NAMES, validate_worker_projection


def _post(worker_url: str, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{worker_url.rstrip('/')}{endpoint}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"UniVTAC worker rejected {endpoint}: {body}") from exc
    if not isinstance(result, dict) or result.get("ok") is not True:
        raise RuntimeError(f"invalid UniVTAC worker response for {endpoint}: {result!r}")
    return result


def _relative_image(path: str, episode_root: Path) -> Path:
    candidate = (episode_root / path).resolve(strict=True)
    root = episode_root.resolve(strict=True)
    if not candidate.is_relative_to(root):
        raise ValueError(f"worker image is outside episode root: {path}")
    return candidate


def _observation_blocks(
    observation: dict[str, Any], episode_root: Path
) -> list[Any]:
    projection = validate_worker_projection(observation)
    text_payload = dict(projection)
    text_payload["images"] = [
        {key: value for key, value in descriptor.items() if key != "path"}
        for descriptor in projection["images"]
    ]
    return [
        TextContent(
            type="text",
            text=json.dumps(text_payload, ensure_ascii=False, separators=(",", ":")),
        ),
        *(
            Image(path=_relative_image(str(descriptor["path"]), episode_root))
            for descriptor in projection["images"]
        ),
    ]


def _record_context(
    *, episode_root: Path, tool: str, arguments: dict[str, Any], blocks: list[Any]
) -> None:
    trace_path = episode_root / "operator_context.jsonl"
    existing = (
        [line for line in trace_path.read_text(encoding="utf-8").splitlines() if line]
        if trace_path.is_file()
        else []
    )
    root = episode_root.resolve(strict=True)
    def recorded_path(block):
        path = Path(block.path).resolve(strict=True)
        if tool == "review_demonstrations" and not path.is_relative_to(root):
            # Shared immutable expert media are not copied into each query episode.
            return path.as_posix()
        return path.relative_to(root).as_posix()

    row = {
        "seq": len(existing) + 1,
        "timestamp_s": time.time(),
        "tool": tool,
        "arguments": arguments,
        "response_text_blocks": [
            block.text for block in blocks if isinstance(block, TextContent)
        ],
        "response_image_paths": [
            recorded_path(block)
            for block in blocks
            if isinstance(block, Image)
        ],
    }
    with trace_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def build_server(*, episode_root: Path, worker_url: str) -> FastMCP:
    root = episode_root.expanduser().resolve(strict=True)
    server = FastMCP("UniVTAC Pull Out Key Operation")

    @server.tool(
        name="observe",
        description=(
            "Observe the live Pull Out Key scene before acting. Returns task text, visible phase, "
            "joint/EE state, head and wrist RGB, and left/right tactile rgb_marker images."
        ),
        annotations=ToolAnnotations(
            readOnlyHint=True, idempotentHint=False, destructiveHint=False
        ),
        structured_output=False,
    )
    def observe() -> list[Any]:
        result = _post(worker_url, "/observe", {})
        blocks = _observation_blocks(result["observation"], root)
        _record_context(episode_root=root, tool="observe", arguments={}, blocks=blocks)
        return blocks

    @server.tool(
        name="execute_skill",
        description=(
            "Execute one bounded native Pull Out Key manipulation skill exactly once and return a "
            "fresh visual/tactile observation. No pose, joint, displacement, or other parameters "
            "are accepted."
        ),
        annotations=ToolAnnotations(
            readOnlyHint=False, idempotentHint=False, destructiveHint=True
        ),
        structured_output=False,
    )
    def execute_skill(
        skill: Literal["align_key", "settle_alignment", "pull_key_out"],
    ) -> list[Any]:
        arguments = {"skill": skill}
        result = _post(worker_url, "/execute", arguments)
        blocks = _observation_blocks(result["observation"], root)
        _record_context(
            episode_root=root,
            tool="execute_skill",
            arguments=arguments,
            blocks=blocks,
        )
        return blocks

    @server.tool(
        name="finish_episode",
        description=(
            "Finish the live episode. Native task evaluation happens privately after this call; "
            "no success or reward is returned to the operator."
        ),
        annotations=ToolAnnotations(
            readOnlyHint=False, idempotentHint=False, destructiveHint=False
        ),
        structured_output=False,
    )
    def finish_episode() -> list[Any]:
        result = _post(worker_url, "/finish", {})
        text = TextContent(
            type="text",
            text=json.dumps(
                {
                    "status": result["status"],
                    "executed_skills": result["executed_skills"],
                    "remaining_skills": result["remaining_skills"],
                },
                separators=(",", ":"),
            ),
        )
        blocks = [text]
        _record_context(
            episode_root=root,
            tool="finish_episode",
            arguments={},
            blocks=blocks,
        )
        return blocks

    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--worker-url", required=True)
    args = parser.parse_args(argv)
    server = build_server(episode_root=args.episode_root, worker_url=args.worker_url)
    registered = tuple(sorted(server._tool_manager._tools))
    if registered != tuple(sorted(MCP_TOOL_NAMES)):
        raise RuntimeError(f"unexpected live UniVTAC MCP tools: {registered}")
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
