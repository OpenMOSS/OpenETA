#!/usr/bin/env python3
"""Load every fixed cuRobo CUDA extension and run its official FK example."""

from __future__ import annotations

import argparse
import importlib
import json
import runpy
import sys
import traceback
from pathlib import Path


EXTENSIONS = (
    "curobo.curobolib.lbfgs_step_cu",
    "curobo.curobolib.kinematics_fused_cu",
    "curobo.curobolib.line_search_cu",
    "curobo.curobolib.tensor_step_cu",
    "curobo.curobolib.geom_cu",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "schema_version": "openeta.univtac.blackwell_curobo_probe.v1",
        "runtime_variant": "blackwell_compat_adaptation_v1",
        "extensions": {},
        "official_smoke": "unresolved",
        "success": False,
    }
    try:
        import torch
        import curobo

        payload["torch_version"] = torch.__version__
        payload["torch_cuda_version"] = torch.version.cuda
        payload["curobo_realpath"] = str(Path(curobo.__file__).resolve())
        for name in EXTENSIONS:
            module = importlib.import_module(name)
            payload["extensions"][name] = str(Path(module.__file__).resolve())
        example = args.source_root / "examples" / "kinematics_example.py"
        if example.is_file():
            runpy.run_path(str(example), run_name="__main__")
            payload["official_smoke"] = "passed"
            payload["official_smoke_path"] = str(example.resolve())
        payload["success"] = len(payload["extensions"]) == len(EXTENSIONS) and payload["official_smoke"] == "passed"
        payload["classification"] = "passed" if payload["success"] else "blackwell_native_bridge_partially_viable"
    except BaseException as exc:
        payload.update(
            {
                "classification": "curobo_sm120_build_failed",
                "error_class": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    payload["forbidden_modules_loaded"] = sorted(
        name for name in sys.modules if name.split(".")[0] in {"isaacsim", "isaaclab", "omni", "carb", "tacex_uipc", "openeta"}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raise SystemExit(0 if payload.get("success") and not payload["forbidden_modules_loaded"] else 1)


if __name__ == "__main__":
    main()
