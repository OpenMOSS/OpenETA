#!/usr/bin/env python3
"""Run an isolated CUDA probe or one bounded official TacEx UIPC step."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.envs.univtac.cuda_device_diagnostics import collect_cuda_diagnostic
from sim.envs.univtac.resource_sanitation import utc_now, write_json


DEFAULT_CANONICAL = Path(
    "/home/ubuntu/wybcode/.worktrees/ftp1-policy/r02-official-89fa681/UniVTAC"
)
OFFICIAL_EXAMPLE_RELATIVE = Path(
    "third_party/TacEx/source/tacex_uipc/examples/libuipc-samples/1_hello_libuipc.py"
)


def run_cuda(output: Path, mode_label: str) -> int:
    payload = collect_cuda_diagnostic()
    payload["mode"] = mode_label
    write_json(output, payload)
    return 0 if payload["success"] else 1


def _add_tacex_paths(task_root: Path) -> None:
    source = task_root / "third_party/TacEx/source"
    for path in (
        task_root,
        source,
        source / "tacex",
        source / "tacex_assets",
        source / "tacex_tasks",
        source / "tacex_uipc",
    ):
        if not path.is_dir():
            raise FileNotFoundError(f"required canonical runtime path is missing: {path}")
        sys.path.insert(0, str(path))


def run_uipc(output: Path, task_root: Path, mode_label: str) -> int:
    example = (task_root / OFFICIAL_EXAMPLE_RELATIVE).resolve()
    if not example.is_file():
        raise FileNotFoundError(f"official UIPC example is missing: {example}")
    _add_tacex_paths(task_root)
    original_argv = sys.argv[:]
    original_cwd = Path.cwd()
    module: Any = None
    steps = 0
    loop_checks = 0
    error: dict[str, str] | None = None
    try:
        os.chdir(task_root / "third_party/TacEx")
        sys.argv = [str(example), "--headless", "--livestream", "0"]
        spec = importlib.util.spec_from_file_location("r04_official_hello_libuipc", example)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"could not load official example: {example}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        original_step = module.UipcSim.step
        original_is_running = module.simulation_app.is_running

        def counted_step(instance, *args, **kwargs):
            nonlocal steps
            returned = original_step(instance, *args, **kwargs)
            steps += 1
            return returned

        def bounded_is_running() -> bool:
            nonlocal loop_checks
            loop_checks += 1
            return steps < 1 and loop_checks <= 20 and bool(original_is_running())

        module.UipcSim.step = counted_step
        module.simulation_app.is_running = bounded_is_running
        module.main()
    except BaseException as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        if module is not None and getattr(module, "simulation_app", None) is not None:
            try:
                module.simulation_app.close()
            except Exception as exc:
                if error is None:
                    error = {"type": type(exc).__name__, "message": str(exc)}
        os.chdir(original_cwd)
        sys.argv = original_argv
    payload = {
        "schema_version": "openeta.univtac.uipc_sentinel.v1",
        "captured_at": utc_now(),
        "mode": mode_label,
        "official_example": str(example),
        "official_source_unmodified": True,
        "bounded_harness_only": True,
        "completed_steps": steps,
        "completed_step": steps >= 1,
        "loop_checks": loop_checks,
        "ftp1_model_loaded": False,
        "openeta_imported": False,
        "error": error,
        "success": steps >= 1 and error is None,
    }
    write_json(output, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["success"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", choices=("cuda", "uipc"), required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode-label", required=True)
    parser.add_argument("--canonical-task-root", default=str(DEFAULT_CANONICAL))
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    if args.probe == "cuda":
        return run_cuda(output, args.mode_label)
    return run_uipc(
        output,
        Path(args.canonical_task_root).expanduser().resolve(),
        args.mode_label,
    )


if __name__ == "__main__":
    raise SystemExit(main())
