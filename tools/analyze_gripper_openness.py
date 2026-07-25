#!/usr/bin/env python3
"""Find the smallest gripper openness values in OpenETA trajectory traces."""

from __future__ import annotations

import argparse
import heapq
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True, slots=True)
class OpennessRecord:
    value: float
    trace_path: Path
    line_number: int
    json_path: str
    event_type: str
    timestamp_s: float | None


def _iter_openness(value: Any, *, json_path: str = "$") -> Iterator[tuple[str, float]]:
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{json_path}.{key}"
            if key == "openness" and isinstance(child, (int, float)) and not isinstance(
                child, bool
            ):
                parsed = float(child)
                if math.isfinite(parsed):
                    yield child_path, parsed
            yield from _iter_openness(child, json_path=child_path)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _iter_openness(child, json_path=f"{json_path}[{index}]")


def _experiment_id(trace_path: Path) -> str:
    parts = trace_path.parts
    try:
        index = parts.index("experiments")
    except ValueError:
        return ""
    return parts[index + 1] if index + 1 < len(parts) else ""


def scan_traces(root: Path, *, top: int) -> tuple[list[OpennessRecord], dict[str, int]]:
    smallest: list[tuple[float, int, OpennessRecord]] = []
    sequence = 0
    stats = {
        "trace_count": 0,
        "line_count": 0,
        "invalid_json_lines": 0,
        "openness_count": 0,
    }
    for trace_path in sorted(root.rglob("trace.jsonl")):
        stats["trace_count"] += 1
        with trace_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                stats["line_count"] += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    stats["invalid_json_lines"] += 1
                    continue
                if not isinstance(row, dict):
                    continue
                event_type = str(row.get("event_type") or "")
                timestamp = row.get("timestamp_s")
                timestamp_s = (
                    float(timestamp)
                    if isinstance(timestamp, (int, float))
                    and not isinstance(timestamp, bool)
                    and math.isfinite(float(timestamp))
                    else None
                )
                for json_path, openness in _iter_openness(row):
                    stats["openness_count"] += 1
                    record = OpennessRecord(
                        value=openness,
                        trace_path=trace_path,
                        line_number=line_number,
                        json_path=json_path,
                        event_type=event_type,
                        timestamp_s=timestamp_s,
                    )
                    # Keep a bounded max-heap using negative openness values.
                    item = (-openness, sequence, record)
                    sequence += 1
                    if len(smallest) < top:
                        heapq.heappush(smallest, item)
                    elif openness < -smallest[0][0]:
                        heapq.heapreplace(smallest, item)
    records = sorted((item[2] for item in smallest), key=lambda item: item.value)
    return records, stats


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively scan OpenETA trace.jsonl files and report the smallest "
            "numeric values stored under keys named 'openness'."
        )
    )
    parser.add_argument(
        "root",
        nargs="?",
        type=Path,
        default=Path(".openeta_memory/experiments"),
        help="Root containing experiment trajectories (default: .openeta_memory/experiments).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="Number of smallest records to print (default: 20).",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    if args.top <= 0:
        raise SystemExit("--top must be greater than zero")
    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"trajectory root does not exist or is not a directory: {root}")

    records, stats = scan_traces(root, top=args.top)
    print(
        "scanned "
        f"{stats['trace_count']} traces, {stats['line_count']} JSONL lines, "
        f"{stats['openness_count']} openness values"
    )
    if stats["invalid_json_lines"]:
        print(
            f"warning: skipped {stats['invalid_json_lines']} invalid JSON lines",
            file=sys.stderr,
        )
    if not records:
        print("no numeric openness values found")
        return 1

    minimum = records[0]
    print(f"global minimum openness: {minimum.value:.12g}")
    print(f"experiment: {_experiment_id(minimum.trace_path)}")
    print(f"trace: {minimum.trace_path}")
    print(f"line: {minimum.line_number}")
    print(f"event_type: {minimum.event_type}")
    print(f"json_path: {minimum.json_path}")
    if minimum.timestamp_s is not None:
        print(f"timestamp_s: {minimum.timestamp_s:.6f}")

    print("\nsmallest records:")
    print("rank\topenness\texperiment\tevent_type\tline\tjson_path\ttrace")
    for rank, record in enumerate(records, start=1):
        print(
            f"{rank}\t{record.value:.12g}\t{_experiment_id(record.trace_path)}\t"
            f"{record.event_type}\t{record.line_number}\t{record.json_path}\t"
            f"{record.trace_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
