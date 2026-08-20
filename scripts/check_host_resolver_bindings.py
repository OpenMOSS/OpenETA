#!/usr/bin/env python3
"""Audit stable and contract-driven host resolver runtime bindings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.tools.contracts import build_default_tool_contract_catalog
from agent.tools.registry import build_default_tool_registry
from agent.tools.runtime_contract_bindings import audit_host_resolver_bindings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)
    catalog = build_default_tool_contract_catalog(build_default_tool_registry().list())
    report = audit_host_resolver_bindings(catalog)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0 if report["conformant"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
