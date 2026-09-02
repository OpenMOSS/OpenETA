#!/usr/bin/env python3
"""Aggregate the R0.9.5.1 wheelhouse result into its top-level manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output = args.output_root.resolve()
    top = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    wheelhouse = json.loads((output / "wheelhouse_pipeline/run_manifest.json").read_text(encoding="utf-8"))
    if top.get("classification") != "legacy_sdist_reproducible_wheel_bridge_validated":
        raise RuntimeError("legacy source-to-wheel bridge is not validated")
    for stage in ("L1", "L2", "D0", "D1", "D2"):
        top["stages"][stage] = wheelhouse["stages"].get(stage, "not_run_due_to_gate")
    top["transport_attempts"] = wheelhouse.get("transport_attempts", 0)
    top["wheel_cache_created"] = True
    if wheelhouse.get("status") == "completed":
        top["status"] = "closed_wheelhouse_validated"
        top["classification"] = "closed_wheelhouse_validated"
    else:
        top["status"] = "failed"
        top["classification"] = wheelhouse.get("classification") or "blocked_by_external_resources"
        top["failure"] = wheelhouse.get("failure")
    write(output / "run_manifest.json", top)
    write(output / "summary.json", top)


if __name__ == "__main__":
    main()
