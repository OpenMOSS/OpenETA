#!/usr/bin/env python3
"""Append one human-readable GPT-Pro/Codex round summary for the dashboard."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def build_entry(args: argparse.Namespace) -> dict[str, Any]:
    experiment_root = args.experiment_root
    commit = args.commit
    push = args.push
    entry: dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "round": args.round,
        "status": args.status,
        "pro_instruction_summary": args.pro_instruction_summary,
        "codex_work_summary": args.codex_work_summary,
        "result_summary": args.result_summary,
        "experiment_root": experiment_root,
        "commit": commit,
        "push": push,
        "pro_question": getattr(args, "pro_question", "") or args.pro_instruction_summary,
        "method_summary": getattr(args, "method_summary", "") or args.codex_work_summary,
        "observed_summary": getattr(args, "observed_summary", "") or args.result_summary,
        "implication_summary": getattr(args, "implication_summary", ""),
        "one_line_conclusion": getattr(args, "one_line_conclusion", "")
        or args.result_summary,
        "evidence": {
            "commit": commit,
            "push": push,
            "tests": getattr(args, "tests", []) or [],
            "experiment_root": experiment_root,
            "artifact_paths": getattr(args, "artifact_paths", []) or [],
        },
    }
    return entry


def append_entry(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--round", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--pro-instruction-summary", required=True)
    parser.add_argument("--codex-work-summary", required=True)
    parser.add_argument("--result-summary", required=True)
    parser.add_argument("--pro-question", default="")
    parser.add_argument("--method-summary", default="")
    parser.add_argument("--observed-summary", default="")
    parser.add_argument("--implication-summary", default="")
    parser.add_argument("--one-line-conclusion", default="")
    parser.add_argument("--experiment-root", default="")
    parser.add_argument("--commit", default="")
    parser.add_argument("--push", default="")
    parser.add_argument("--test", dest="tests", action="append", default=[])
    parser.add_argument(
        "--artifact-path", dest="artifact_paths", action="append", default=[]
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    append_entry(args.output.expanduser().resolve(), build_entry(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
