#!/usr/bin/env python3
"""Audit installed packaging requirements and fixed-source packaging imports."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import re
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = ROOT / "sim/envs/univtac/packaging_compatibility.py"
SPEC = importlib.util.spec_from_file_location("univtac_packaging_compatibility", HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"cannot load packaging compatibility helper: {HELPER_PATH}")
helper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helper)


IMPORT_PATTERN = re.compile(r"^\s*(?:import\s+packaging(?:\.|\s|$)|from\s+packaging(?:\.|\s))", re.MULTILINE)


def distribution_records() -> list[dict[str, object]]:
    return [
        {"name": item.metadata.get("Name"), "version": item.version, "requires": list(item.requires or [])}
        for item in importlib.metadata.distributions()
        if item.metadata.get("Name")
    ]


def source_imports(specifications: list[str]) -> list[dict[str, object]]:
    records = []
    for specification in specifications:
        label, raw_path = specification.split("=", 1)
        root = Path(raw_path).resolve()
        matches = []
        if root.exists():
            for path in sorted(root.rglob("*.py")):
                try:
                    text = path.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):
                    continue
                for match in IMPORT_PATTERN.finditer(text):
                    matches.append({"path": str(path), "line": text.count("\n", 0, match.start()) + 1, "statement": match.group(0).strip()})
        records.append({"label": label, "root": str(root), "matches": matches})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        installed = importlib.metadata.version("packaging")
        requirements = helper.audit_requirements(distribution_records())
        payload = {
            "schema_version": "openeta.univtac.packaging_requirements_inventory.v1",
            "python_executable": str(Path(sys.executable).resolve()),
            "installed_packaging_version": installed,
            "requirements": requirements["requirements"],
            "blockers": requirements["blockers"],
            "source_imports": source_imports(args.source_root),
            "success": installed == helper.NATIVE_VERSION and requirements["success"],
        }
    except BaseException as exc:
        payload = {"success": False, "error_class": type(exc).__name__, "error_message": str(exc), "traceback": traceback.format_exc()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    raise SystemExit(0 if payload["success"] else 1)


if __name__ == "__main__":
    main()
