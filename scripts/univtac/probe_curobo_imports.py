#!/usr/bin/env python3
"""Import the five fixed cuRobo CUDA extensions in an isolated process."""

from __future__ import annotations

import argparse
import importlib
import json
import time
import traceback
from pathlib import Path

MODULES = tuple(
    f"curobo.curobolib.{name}"
    for name in ("lbfgs_step_cu", "kinematics_fused_cu", "line_search_cu", "tensor_step_cu", "geom_cu")
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {"success": False, "modules": {}}
    try:
        import curobo
        import torch

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
        payload["success"] = len(payload["modules"]) == len(MODULES)
    except BaseException as exc:  # noqa: BLE001 - persist native import failures
        payload.update({"error_class": type(exc).__name__, "error_message": str(exc), "traceback": traceback.format_exc()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raise SystemExit(0 if payload["success"] else 1)


if __name__ == "__main__":
    main()
