#!/usr/bin/env python3
"""Print the exact Operator prompt, tools, schemas, and resolved contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tools.operator_context_profiles import canonical_sha256, load_profile


def build_surface(*, include_content: bool = False) -> dict[str, Any]:
    profile = load_profile()

    # Import after resolving the active profile because the MCP server binds
    # its public tool set and schemas at module initialization.
    from tools.embodied_mcp_server import mcp

    registered = sorted(mcp._tool_manager.list_tools(), key=lambda tool: tool.name)
    tools = {
        tool.name: {
            "description_sha256": canonical_sha256(tool.description),
            "input_schema_sha256": canonical_sha256(tool.parameters),
            **(
                {
                    "description": tool.description,
                    "input_schema": tool.parameters,
                }
                if include_content
                else {}
            ),
        }
        for tool in registered
    }
    payload: dict[str, Any] = {
        "schema_version": "openeta.operator_release_surface.v1",
        "profile": profile.label,
        "status": profile.status,
        "composition_sha256": profile.composition_sha256,
        "manifest_sha256": profile.manifest_sha256,
        "prompt_sha256": canonical_sha256(profile.prompt_template),
        "tool_descriptions_sha256": canonical_sha256(profile.tool_descriptions),
        "resolved_invariants_sha256": canonical_sha256(
            profile.manifest["invariants"]
        ),
        "public_operator_tools": profile.public_operator_tools,
        "tools": tools,
    }
    if include_content:
        payload.update(
            {
                "prompt_template": profile.prompt_template,
                "tool_descriptions": profile.tool_descriptions,
                "resolved_invariants": profile.manifest["invariants"],
            }
        )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-content", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    rendered = json.dumps(
        build_surface(include_content=args.include_content),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
