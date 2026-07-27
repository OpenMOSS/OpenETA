#!/usr/bin/env python3
"""Exercise BEHAVIOR through the same isolated worker transport used by MCP."""

from __future__ import annotations

import json
from pathlib import Path
import time

import numpy as np

from sim.mcp_server.worker_mgr import BenchWorkerManager


OUTPUT = Path("outputs/behavior/worker_smoke.json")


def _obs_summary(payload: dict) -> dict:
    obs = payload.get("observation", payload)
    cameras = obs.get("cameras", []) if isinstance(obs, dict) else []
    return {
        "keys": sorted(obs) if isinstance(obs, dict) else [],
        "camera_names": sorted(
            camera.get("frame_id", camera.get("name", ""))
            for camera in cameras
            if isinstance(camera, dict)
        ),
        "reward": payload.get("reward"),
        "terminated": payload.get("terminated"),
        "truncated": payload.get("truncated"),
    }


def main() -> int:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    manager = BenchWorkerManager()
    worker = None
    handle = None
    started = time.monotonic()
    try:
        env_id = "openeta/behavior_picking_up_trash-v0"
        created, worker = manager.create_env_on_worker(
            env_id,
            {
                "env_id": env_id,
                "seed": 0,
                "render_mode": "rgb_array",
                "image_width": 128,
                "image_height": 128,
                "robot": "R1Pro",
            },
        )
        if "error" in created:
            raise RuntimeError(created["error"])
        handle = created["handle"]
        reset = worker.proxy("POST", f"/env/{handle}/reset", {"seed": 0})
        step = worker.proxy(
            "POST",
            f"/env/{handle}/step",
            {"action": np.zeros(created["action_dim"], dtype=np.float32).tolist(), "render": True},
        )
        closed = worker.proxy("DELETE", f"/env/{handle}")
        if not closed.get("ok"):
            raise RuntimeError(f"close failed: {closed}")
        manager.release_worker(worker.base_url)
        worker = None
        handle = None
        result = {
            "ok": True,
            "elapsed_s": round(time.monotonic() - started, 3),
            "create": created,
            "reset": _obs_summary(reset),
            "step": _obs_summary(step),
            "close": closed,
            "worker_retired_after_close": True,
        }
        OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2))
        return 0
    except BaseException as exc:
        failure = {
            "ok": False,
            "elapsed_s": round(time.monotonic() - started, 3),
            "phase": "worker create/reset/step/close",
            "error": f"{type(exc).__name__}: {exc}",
        }
        OUTPUT.write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        raise
    finally:
        if worker is not None and handle is not None:
            try:
                worker.proxy("DELETE", f"/env/{handle}")
            except Exception:
                pass
        manager.stop_all()


if __name__ == "__main__":
    raise SystemExit(main())
