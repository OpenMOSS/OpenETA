#!/usr/bin/env python3
"""Check typed-fact compatibility for an ordered list of OpenETA tools."""

from __future__ import annotations

import argparse
import json

from agent.tools.contracts import (
    build_default_tool_contract_catalog,
    check_tool_chain_compatibility,
)
from agent.tools.registry import build_default_tool_registry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "tools",
        help="comma-separated tool names; order describes a proposed composition",
    )
    parser.add_argument(
        "--initial-fact",
        action="append",
        default=[],
        help="host/external typed fact already available at the start; repeatable",
    )
    args = parser.parse_args()
    tool_names = tuple(name.strip() for name in args.tools.split(",") if name.strip())
    catalog = build_default_tool_contract_catalog(build_default_tool_registry().list())
    report = check_tool_chain_compatibility(
        catalog,
        tool_names,
        initial_facts=args.initial_fact,
    )
    print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
    return 0 if report.compatible else 1


if __name__ == "__main__":
    raise SystemExit(main())
