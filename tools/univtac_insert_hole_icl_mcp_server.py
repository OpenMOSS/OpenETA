#!/usr/bin/env python3
"""Expose one static R1.3 Insert Hole ICL decision through bounded MCP tools."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.types import Image
from mcp.types import TextContent, ToolAnnotations

from sim.envs.univtac.insert_hole_tactile_icl import (
    R13_MCP_TOOLS,
    OneChoiceBudget,
    validate_visible_payload,
)
from sim.envs.univtac.trace import write_json
from tools.univtac_operation_mcp_server import _record_context, _relative_image


def _image_blocks(phases: list[dict[str, Any]], root: Path) -> list[Any]:
    blocks: list[Any] = []
    for phase in phases:
        text = {
            "phase": phase["phase"],
            "proprio": phase["proprio"],
            "images": [
                {key: value for key, value in descriptor.items() if key != "path"}
                for descriptor in phase["images"]
            ],
        }
        blocks.append(TextContent(type="text", text=json.dumps(text, separators=(",", ":"))))
        blocks.extend(
            Image(path=_relative_image(str(descriptor["path"]), root))
            for descriptor in phase["images"]
        )
    return blocks


def build_server(*, episode_root: Path) -> FastMCP:
    root = episode_root.expanduser().resolve(strict=True)
    manifest = json.loads((root / "condition.json").read_text(encoding="utf-8"))
    visible = validate_visible_payload(manifest["agent_visible"])
    budget = OneChoiceBudget()
    server = FastMCP("UniVTAC Insert Hole Tactile ICL")

    @server.tool(
        name="review_demonstrations",
        description="Review the available successful contact-transition examples.",
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=False),
        structured_output=False,
    )
    def review_demonstrations() -> list[Any]:
        demos = visible["demonstrations"]
        if not demos:
            blocks: list[Any] = [
                TextContent(type="text", text="No demonstrations are available for this episode.")
            ]
        else:
            blocks = []
            for index, demo in enumerate(demos, start=1):
                blocks.append(
                    TextContent(
                        type="text",
                        text=(
                            f"Example {index}\n"
                            f"Opaque procedure: {demo['action']}\n"
                            "Execution: completed\n"
                            "Episode outcome: native success"
                        ),
                    )
                )
                blocks.extend(
                    _image_blocks(
                        [
                            demo["before_contact"],
                            demo["after_contact"],
                            demo["outcome_observation"],
                        ],
                        root,
                    )
                )
        _record_context(
            episode_root=root,
            tool="review_demonstrations",
            arguments={},
            blocks=blocks,
        )
        return blocks

    @server.tool(
        name="observe_query",
        description="Inspect the current before/after-contact observation pair.",
        annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=False),
        structured_output=False,
    )
    def observe_query() -> list[Any]:
        query = visible["query"]
        header = TextContent(
            type="text",
            text=json.dumps(
                {
                    "task_instruction": query["task_instruction"],
                    "available_skills": query["available_skills"],
                },
                separators=(",", ":"),
            ),
        )
        blocks = [
            header,
            *_image_blocks([query["before_contact"], query["after_contact"]], root),
        ]
        _record_context(
            episode_root=root,
            tool="observe_query",
            arguments={},
            blocks=blocks,
        )
        return blocks

    @server.tool(
        name="choose_skill",
        description=(
            "Submit exactly one opaque manipulation procedure. This records the decision; "
            "it does not execute a robot action."
        ),
        annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=False),
        structured_output=False,
    )
    def choose_skill(
        skill: Literal["skill_slate", "skill_ember"],
        confidence: Literal["low", "medium", "high"],
        reason: str,
    ) -> list[Any]:
        budget.reserve(skill)
        decision = {
            "schema_version": "openeta.univtac.r13.codex_decision.v1",
            "skill": skill,
            "confidence": confidence,
            "reason": str(reason),
            "choice_count": 1,
        }
        write_json(root / "decision.json", decision)
        arguments = {"skill": skill, "confidence": confidence, "reason": str(reason)}
        blocks = [
            TextContent(
                type="text",
                text=json.dumps(
                    {"status": "choice_recorded", "skill": skill}, separators=(",", ":")
                ),
            )
        ]
        _record_context(
            episode_root=root,
            tool="choose_skill",
            arguments=arguments,
            blocks=blocks,
        )
        return blocks

    return server


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode-root", type=Path, required=True)
    args = parser.parse_args(argv)
    server = build_server(episode_root=args.episode_root)
    if tuple(sorted(server._tool_manager._tools)) != tuple(sorted(R13_MCP_TOOLS)):
        raise RuntimeError("unexpected R1.3 MCP tool list")
    server.run(transport="stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
