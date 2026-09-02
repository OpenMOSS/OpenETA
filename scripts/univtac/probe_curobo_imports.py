#!/usr/bin/env python3
"""Import the five fixed cuRobo CUDA extensions in an isolated process."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "sim/envs/univtac/elf_loader_closure.py"
helper_spec = importlib.util.spec_from_file_location("univtac_elf_loader_closure", HELPER)
if helper_spec is None or helper_spec.loader is None:
    raise ImportError(f"cannot load loader helper: {HELPER}")
helper = importlib.util.module_from_spec(helper_spec)
helper_spec.loader.exec_module(helper)

MODULES = tuple(
    f"curobo.curobolib.{name}"
    for name in ("lbfgs_step_cu", "kinematics_fused_cu", "line_search_cu", "tensor_step_cu", "geom_cu")
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-root", type=Path, required=True)
    parser.add_argument("--forbidden-root", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {"success": False, "modules": {}}
    try:
        import torch

        torch.arange(4, device="cuda").sum()
        torch.cuda.synchronize()
        memory_before = {
            "allocated_bytes": torch.cuda.memory_allocated(),
            "reserved_bytes": torch.cuda.memory_reserved(),
        }
        import curobo

        payload.update(
            {
                "curobo_realpath": str(Path(curobo.__file__).resolve()),
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "device_capability": list(torch.cuda.get_device_capability()),
            }
        )
        for name in MODULES:
            started = time.monotonic()
            module = importlib.import_module(name)
            payload["modules"][name] = {
                "realpath": str(Path(module.__file__).resolve()),
                "latency_seconds": time.monotonic() - started,
            }
        torch.cuda.synchronize()
        records = helper.parse_proc_maps(Path("/proc/self/maps").read_text(encoding="utf-8"))
        objects = helper.loaded_objects(records)
        payload["gpu_memory"] = {
            "before_imports": memory_before,
            "after_imports": {
                "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
            },
        }
        payload["loaded_objects"] = objects
        payload["legacy_paths"] = helper.legacy_paths(objects, args.forbidden_root)
        expected = str(args.expected_root.resolve()) + "/"
        payload["module_provenance_ok"] = all(
            str(record["realpath"]).startswith(expected) for record in payload["modules"].values()
        )
        payload["success"] = (
            len(payload["modules"]) == len(MODULES)
            and payload["module_provenance_ok"]
            and not payload["legacy_paths"]
        )
    except BaseException as exc:  # noqa: BLE001 - persist native import failures
        payload.update({"error_class": type(exc).__name__, "error_message": str(exc), "traceback": traceback.format_exc()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raise SystemExit(0 if payload["success"] else 1)


if __name__ == "__main__":
    main()
