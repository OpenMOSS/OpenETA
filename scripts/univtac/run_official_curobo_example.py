#!/usr/bin/env python3
"""Run one unmodified cuRobo example and record its loaded CUDA extensions."""

from __future__ import annotations

import argparse
import json
import runpy
import traceback
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", type=Path, required=True)
    parser.add_argument("--expected-root", type=Path, required=True)
    parser.add_argument("--expected-extension", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    example = args.example.resolve()
    expected_root = args.expected_root.resolve()
    payload = {
        "success": False,
        "example": str(example),
        "example_provenance_ok": example.is_relative_to(expected_root),
        "expected_extensions": sorted(args.expected_extension),
    }
    try:
        import torch

        if not payload["example_provenance_ok"]:
            raise ValueError("official example is outside the fixed cuRobo checkout")
        runpy.run_path(str(example), run_name="__main__")
        torch.cuda.synchronize()
        payload["example_returned"] = True
    except BaseException as exc:  # noqa: BLE001
        payload.update({"error_class": type(exc).__name__, "error_message": str(exc), "traceback": traceback.format_exc()})
    maps = Path("/proc/self/maps").read_text(encoding="utf-8")
    payload["loaded_extensions"] = sorted(
        {line.split()[-1] for line in maps.splitlines() if "curobolib/" in line and "_cu" in line}
    )
    payload["extension_provenance_ok"] = all(
        Path(path).resolve().is_relative_to(expected_root) for path in payload["loaded_extensions"]
    )
    loaded_names = {Path(path).name.split(".", 1)[0] for path in payload["loaded_extensions"]}
    payload["expected_extensions_loaded"] = set(args.expected_extension).issubset(loaded_names)
    payload["success"] = bool(
        payload.get("example_returned")
        and payload["example_provenance_ok"]
        and payload["extension_provenance_ok"]
        and payload["expected_extensions_loaded"]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raise SystemExit(0 if payload["success"] else 1)


if __name__ == "__main__":
    main()
