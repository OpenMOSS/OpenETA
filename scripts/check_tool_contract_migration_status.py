#!/usr/bin/env python3
"""Generate the local ToolContract migration completion audit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.evals.tool_contract_migration_status import (
    audit_tool_contract_migration_status,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--output", default="")
    parser.add_argument("--test-passed", type=int, default=0)
    parser.add_argument("--test-skipped", type=int, default=0)
    parser.add_argument("--test-warnings", type=int, default=0)
    parser.add_argument("--harness-revision", type=int)
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="also fail while external review or post-review work remains",
    )
    args = parser.parse_args(argv)
    report = audit_tool_contract_migration_status(
        args.repo_root,
        test_passed=args.test_passed,
        test_skipped=args.test_skipped,
        test_warnings=args.test_warnings,
        harness_revision=args.harness_revision,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    if not report["internal_conformant"]:
        return 1
    if args.require_complete and not report["goal_complete"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

