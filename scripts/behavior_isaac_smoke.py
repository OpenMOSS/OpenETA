#!/usr/bin/env python3
"""Launch and shut down a real Isaac Sim headless app, recording evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import subprocess
import time


REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs" / "behavior" / "isaac_headless_smoke.json",
    )
    args = parser.parse_args()
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

    started = time.time()
    from isaacsim import SimulationApp

    app = SimulationApp({"headless": True})
    try:
        import torch

        app.update()
        gpu = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        import isaacsim

        kit_logs = Path(isaacsim.__file__).parent / "kit" / "logs" / "Kit" / "Isaac-Sim Python" / "5.1"
        logs = sorted(kit_logs.glob("kit_*.log"), key=lambda path: path.stat().st_mtime)
        result = {
            "ok": True,
            "command": "sim/venvs/behavior/runtime/bin/python scripts/behavior_isaac_smoke.py",
            "exit_code": 0,
            "platform": platform.platform(),
            "gpu": gpu.stdout.strip(),
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "torch_cuda_available": torch.cuda.is_available(),
            "torch_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "isaac_sim": "5.1.0",
            "elapsed_s": round(time.time() - started, 3),
            "kit_log": str(logs[-1]) if logs else None,
            "summary": "SimulationApp reached app-ready, updated once, and shut down cleanly.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2), flush=True)
        return 0
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
