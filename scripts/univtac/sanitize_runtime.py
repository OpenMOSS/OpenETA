#!/usr/bin/env python3
"""Inventory and narrowly clean stale UniVTAC/Isaac runtime processes."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.envs.univtac.resource_sanitation import (
    apply_cleanup_plan,
    attach_gpu_usage,
    build_cleanup_plan,
    collect_gpu_inventory,
    collect_inotify_inventory,
    collect_process_inventory,
    safe_device_environment,
    utc_now,
    write_json,
)


DEFAULT_OUTPUT = REPO_ROOT / "outputs/univtac-runtime-r04"
DEFAULT_CANONICAL = Path(
    "/home/ubuntu/wybcode/.worktrees/ftp1-policy/r02-official-89fa681/UniVTAC"
)


def _related_roots(canonical: Path) -> list[Path]:
    return [
        REPO_ROOT,
        canonical,
        REPO_ROOT.parents[1] / "worktrees",
    ]


def _inventory(output_root: Path, canonical: Path, phase: str) -> dict[str, object]:
    gpu = collect_gpu_inventory()
    processes = attach_gpu_usage(
        collect_process_inventory(related_roots=_related_roots(canonical)), gpu
    )
    inotify = collect_inotify_inventory()
    if phase == "pre_cleanup":
        target = output_root / "pre_cleanup"
        write_json(target / "process_inventory.json", processes)
        write_json(target / "gpu_inventory.json", gpu)
        write_json(target / "inotify_inventory.json", inotify)
        write_json(
            target / "runtime_environment.json",
            {
                "captured_at": utc_now(),
                "uid": os.getuid(),
                "cwd": str(Path.cwd()),
                "python_executable": sys.executable,
                "device_environment": safe_device_environment(),
                "canonical_task_root": str(canonical.resolve()),
                "repo_root": str(REPO_ROOT),
            },
        )
    else:
        target = output_root / "cleanup"
        write_json(target / "post_cleanup_process_inventory.json", processes)
        write_json(target / "post_cleanup_gpu_inventory.json", gpu)
        write_json(target / "post_cleanup_inotify_inventory.json", inotify)
    return {"processes": processes, "gpu": gpu, "inotify": inotify}


def run_dry_run(output_root: Path, canonical: Path) -> int:
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    inventory = _inventory(output_root, canonical, "pre_cleanup")
    plan = build_cleanup_plan(inventory["processes"])
    write_json(output_root / "cleanup/cleanup_plan.json", plan)
    write_json(
        output_root / "run_manifest.json",
        {
            "schema_version": "openeta.univtac.runtime_sanitation_run.v1",
            "started_at": utc_now(),
            "status": "dry_run_complete",
            "dry_run": True,
            "cleanup_counts": plan["counts"],
        },
    )
    return 0


def run_apply(output_root: Path, canonical: Path, wait_seconds: float) -> int:
    plan_path = output_root / "cleanup/cleanup_plan.json"
    if not plan_path.is_file():
        raise FileNotFoundError(f"dry-run cleanup plan is missing: {plan_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    actions = apply_cleanup_plan(plan, term_wait_seconds=wait_seconds)
    action_path = output_root / "cleanup/cleanup_actions.jsonl"
    action_path.parent.mkdir(parents=True, exist_ok=True)
    action_path.write_text(
        "".join(json.dumps(item, sort_keys=True) + "\n" for item in actions),
        encoding="utf-8",
    )
    time.sleep(15)
    post = _inventory(output_root, canonical, "post_cleanup")
    remaining_eligible = [
        item
        for item in post["processes"]["processes"]
        if item.get("eligible_for_cleanup")
    ]
    manifest_path = output_root / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "cleanup_applied_at": utc_now(),
            "dry_run": False,
            "status": (
                "cleanup_complete" if not remaining_eligible else "cleanup_incomplete"
            ),
            "cleanup_action_count": len(actions),
            "remaining_eligible_pids": [item["pid"] for item in remaining_eligible],
        }
    )
    write_json(manifest_path, manifest)
    return 0 if not remaining_eligible else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True)
    mode.add_argument("--apply", action="store_true")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--canonical-task-root", default=str(DEFAULT_CANONICAL))
    parser.add_argument("--term-wait-seconds", type=float, default=15.0)
    args = parser.parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    canonical = Path(args.canonical_task_root).expanduser().resolve()
    if args.apply:
        return run_apply(output_root, canonical, args.term_wait_seconds)
    return run_dry_run(output_root, canonical)


if __name__ == "__main__":
    raise SystemExit(main())
