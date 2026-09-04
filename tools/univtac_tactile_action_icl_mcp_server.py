#!/usr/bin/env python3
"""Expose one live R1.1 tactile-action ICL episode through bounded MCP tools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.types import Image
from mcp.types import TextContent, ToolAnnotations

from sim.envs.univtac.tactile_action_icl import (
    R11_MCP_TOOLS,
    validate_r11_projection,
)
from tools.univtac_operation_mcp_server import _post, _record_context, _relative_image


def _observation_blocks(
    observation: dict[str, Any], episode_root: Path, image_mode: str
) -> list[Any]:
    projection = validate_r11_projection(observation, image_mode=image_mode)
    text_payload = dict(projection)
    text_payload["images"] = [
        {key: value for key, value in descriptor.items() if key != "path"}
        for descriptor in projection["images"]
    ]
    return [
        TextContent(type="text", text=json.dumps(text_payload, separators=(",", ":"))),
        *(
            Image(path=_relative_image(str(descriptor["path"]), episode_root))
            for descriptor in projection["images"]
        ),
    ]


def build_server(*, episode_root: Path, worker_url: str) -> FastMCP:
    root = episode_root.expanduser().resolve(strict=True)
    condition = json.loads((root / "condition.json").read_text(encoding="utf-8"))
    visible = condition["agent_visible"]
    image_mode = str(visible["image_mode"])
    server = FastMCP("UniVTAC Tactile Action ICL")

    @server.tool(
        name="review_demonstrations",
        description="Review the episode's in-context examples before observing the live query.",
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=False),
        structured_output=False,
    )
    def review_demonstrations() -> list[Any]:
        demos = visible["demonstrations"]
        if not demos:
            blocks: list[Any] = [
                TextContent(
                    type="text", text="No demonstrations are provided for this episode."
                )
            ]
        else:
            blocks = []
            for demo in demos:
                blocks.extend(
                    [
                        TextContent(
                            type="text",
                            text=(
                                f"Example {demo['example']}\n\n"
                                f"Action:\n{demo['action']}\n\n"
                                "Execution:\ncompleted\n\n"
                                "Episode outcome:\nnative success"
                            ),
                        ),
                        Image(path=_relative_image(demo["image_path"], root)),
                    ]
                )
        _record_context(
            episode_root=root,
            tool="review_demonstrations",
            arguments={},
            blocks=blocks,
        )
        return blocks

    @server.tool(
        name="observe",
        description="Observe the current live query state using this condition's modalities.",
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=False),
        structured_output=False,
    )
    def observe() -> list[Any]:
        result = _post(worker_url, "/observe", {})
        blocks = _observation_blocks(result["observation"], root, image_mode)
        _record_context(episode_root=root, tool="observe", arguments={}, blocks=blocks)
        return blocks

    @server.tool(
        name="execute_skill",
        description=(
            "Execute one opaque bounded manipulation procedure and return a fresh observation. "
            "No pose, joint, displacement, or other parameters are accepted."
        ),
        annotations=ToolAnnotations(
            readOnlyHint=False, idempotentHint=False, destructiveHint=True
        ),
        structured_output=False,
    )
    def execute_skill(
        skill: Literal["skill_mica", "skill_onyx", "skill_quartz"],
    ) -> list[Any]:
        arguments = {"skill": skill}
        result = _post(worker_url, "/execute", arguments)
        blocks = _observation_blocks(result["observation"], root, image_mode)
        _record_context(
            episode_root=root, tool="execute_skill", arguments=arguments, blocks=blocks
        )
        return blocks

    @server.tool(
        name="finish_episode",
        description="Finish interaction; native evaluation remains private.",
        annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=False),
        structured_output=False,
    )
    def finish_episode() -> list[Any]:
        result = _post(worker_url, "/finish", {})
        blocks = [TextContent(type="text", text=json.dumps({"status": result["status"]}))]
        _record_context(
            episode_root=root, tool="finish_episode", arguments={}, blocks=blocks
        )
        return blocks

    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--worker-url", required=True)
    args = parser.parse_args(argv)
    server = build_server(episode_root=args.episode_root, worker_url=args.worker_url)
    if tuple(sorted(server._tool_manager._tools)) != tuple(sorted(R11_MCP_TOOLS)):
        raise RuntimeError("unexpected R1.1 MCP tool list")
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
