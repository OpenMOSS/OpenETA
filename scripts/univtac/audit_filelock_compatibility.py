#!/usr/bin/env python3
"""Audit installed reverse dependencies against filelock 3.13.1."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = ROOT / "sim/envs/univtac/filelock_compatibility.py"
SPEC = importlib.util.spec_from_file_location("univtac_filelock_compatibility", HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(HELPER_PATH)
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)
AFTER_VERSION = HELPER.AFTER_VERSION
active_reverse_dependencies = HELPER.active_reverse_dependencies


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    distributions = [
        {
            "name": distribution.metadata.get("Name"),
            "version": distribution.version,
            "requires": list(distribution.requires or ()),
        }
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    ]
    audit = active_reverse_dependencies(distributions, candidate_version=AFTER_VERSION)
    payload = {
        "schema_version": "openeta.univtac.filelock_reverse_dependency_audit.v1",
        "installed_filelock": importlib.metadata.version("filelock"),
        "candidate_filelock": AFTER_VERSION,
        "audit": audit,
        "success": audit["success"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    raise SystemExit(0 if payload["success"] else 1)


if __name__ == "__main__":
    main()
