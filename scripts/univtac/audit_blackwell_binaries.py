#!/usr/bin/env python3
"""Audit CUDA architecture and dynamic linking of selected native binaries."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from sim.envs.univtac.native_build_contract import sha256_file


def _run(command: list[str]) -> dict[str, object]:
    completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=300)
    return {"returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}


def audit(path: Path) -> dict[str, object]:
    record: dict[str, object] = {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    combined = ""
    for name, arguments in (
        ("readelf_dynamic", ["readelf", "-d"]),
        ("readelf_notes", ["readelf", "-n"]),
        ("ldd", ["ldd"]),
        ("cuobjdump_elf", ["cuobjdump", "--list-elf"]),
        ("cuobjdump_ptx", ["cuobjdump", "--dump-ptx"]),
    ):
        executable = shutil.which(arguments[0])
        if executable is None:
            record[name] = {"available": False}
            continue
        result = _run([executable, *arguments[1:], str(path)])
        text = str(result["stdout"]) + "\n" + str(result["stderr"])
        combined += "\n" + text
        record[name] = {
            "available": True,
            "returncode": result["returncode"],
            "output_tail": text[-6000:],
        }
    tokens = sorted(set(re.findall(r"(?:sm|compute)_[0-9]+[a-z]?", combined)))
    record["architecture_tokens"] = tokens
    record["has_sm120_or_compute120"] = "sm_120" in tokens or "compute_120" in tokens
    record["has_forbidden_fallback"] = any(token in tokens for token in ("sm_89", "compute_89", "sm_90", "compute_90", "sm_120a"))
    record["missing_dependencies"] = sorted(
        line.strip() for line in combined.splitlines() if "not found" in line.lower()
    )
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("binaries", nargs="+", type=Path)
    args = parser.parse_args()
    records = [audit(path) for path in args.binaries if path.is_file()]
    payload = {
        "schema_version": "openeta.univtac.blackwell_binary_audit.v1",
        "runtime_variant": "blackwell_compat_adaptation_v1",
        "records": records,
        "has_sm120_or_compute120": any(item["has_sm120_or_compute120"] for item in records),
        "binaries_with_forbidden_fallback_tokens": [
            item["path"] for item in records if item["has_forbidden_fallback"]
        ],
        "missing_dependencies": sorted({dep for item in records for dep in item["missing_dependencies"]}),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raise SystemExit(0 if records and payload["has_sm120_or_compute120"] and not payload["missing_dependencies"] else 1)


if __name__ == "__main__":
    main()
