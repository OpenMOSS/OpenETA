#!/usr/bin/env python3
"""Run the fixed three-seed Pull Out Key native expert baseline."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.envs.univtac.native_operation import (
    EXPERT_SEEDS,
    summarize_native_expert_results,
    validate_native_expert_config,
)
from sim.envs.univtac.scoped_isaac51_launcher import (
    ScopedIsaac51LaunchError,
    ScopedIsaac51LaunchSpec,
    run_scoped_isaac51_command,
)
from sim.envs.univtac.trace import write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


def _git_value(cwd: Path, *args: str) -> str | None:
    import subprocess

    completed = subprocess.run(
        ["git", *args], cwd=cwd, check=False, capture_output=True, text=True
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _result_row(
    *, seed: int, seed_root: Path, lifecycle: dict[str, Any] | None, error: str | None
) -> dict[str, Any]:
    child = _read_json(seed_root / "child_result.json")
    final = _read_json(seed_root / "final_result.json")
    cleanup = child.get("cleanup", {}) if child else {}
    infrastructure_valid = bool(
        lifecycle
        and lifecycle.get("returncode") == 0
        and not lifecycle.get("timed_out")
        and lifecycle.get("cleanup_complete")
        and child
        and child.get("status") == "completed"
        and cleanup.get("task_close")
    )
    return {
        "seed": seed,
        "infrastructure_valid": infrastructure_valid,
        "classification": (
            child.get("classification") if child else "native_runtime_abort"
        ),
        "plan_success": bool(final and final.get("plan_success")),
        "native_check_success": bool(final and final.get("native_check_success")),
        "native_check_early_stop": (
            final.get("native_check_early_stop") if final else None
        ),
        "expert_episode_success": bool(final and final.get("expert_episode_success")),
        "play_once_call_count": (
            child.get("counters", {}).get("play_once_call_count", 0) if child else 0
        ),
        "move_call_count": (
            child.get("counters", {}).get("move_call_count", 0) if child else 0
        ),
        "check_success_call_count": (
            child.get("counters", {}).get("check_success_call_count", 0) if child else 0
        ),
        "check_early_stop_call_count": (
            child.get("counters", {}).get("check_early_stop_call_count", 0)
            if child
            else 0
        ),
        "transition_count": int(final.get("transition_count", 0)) if final else 0,
        "semantic_segments": final.get("semantic_segments", []) if final else [],
        "cleanup_complete": bool(lifecycle and lifecycle.get("cleanup_complete")),
        "error": error or (child.get("error") if child else "missing child result"),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "seed",
        "infrastructure_valid",
        "classification",
        "plan_success",
        "native_check_success",
        "native_check_early_stop",
        "expert_episode_success",
        "play_once_call_count",
        "move_call_count",
        "check_success_call_count",
        "check_early_stop_call_count",
        "transition_count",
        "cleanup_complete",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.expanduser().resolve(strict=True)
    runtime_python = args.runtime_python.expanduser().resolve(strict=True)
    source_root = args.source_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"output root must be fresh: {output_root}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("native expert config must be a mapping")
    config = validate_native_expert_config(payload)
    output_root.mkdir(parents=True)
    expert_root = output_root / "expert"
    expert_root.mkdir()
    run_manifest = {
        "schema_version": "openeta.univtac.r10_run.v1",
        "round": "R1.0",
        "status": "running_native_expert",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "repo_head": _git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "repo_branch": _git_value(REPO_ROOT, "branch", "--show-current"),
        "source_root": str(source_root),
        "source_head": _git_value(source_root, "rev-parse", "HEAD"),
        "runtime_python": str(runtime_python),
        "config": str(config_path),
        "seeds": list(EXPERT_SEEDS),
        "simulator_invocation_count": 0,
        "agent_process_count": 0,
        "agent_action_count": 0,
    }
    write_json(output_root / "run_manifest.json", run_manifest)

    rows: list[dict[str, Any]] = []
    all_transitions: list[dict[str, Any]] = []
    for seed in EXPERT_SEEDS:
        seed_root = expert_root / f"seed_{seed}"
        command = [
            str(REPO_ROOT / "scripts" / "univtac" / "probe_pull_out_key_native_expert.py"),
            "--config",
            str(config_path),
            "--source-root",
            str(source_root),
            "--repo-root",
            str(REPO_ROOT),
            "--output-root",
            str(seed_root),
            "--seed",
            str(seed),
        ]
        if args.headless:
            command.append("--headless")
        spec = ScopedIsaac51LaunchSpec(
            python_executable=runtime_python,
            command=tuple(command),
            cwd=source_root,
            output_root=seed_root,
            timeout_seconds=config["timeout_seconds"],
        )
        lifecycle = None
        error = None
        try:
            run_manifest["simulator_invocation_count"] += 1
            write_json(output_root / "run_manifest.json", run_manifest)
            lifecycle = run_scoped_isaac51_command(spec).to_dict()
        except ScopedIsaac51LaunchError as exc:
            error = f"{type(exc).__name__}: {exc}"
        row = _result_row(seed=seed, seed_root=seed_root, lifecycle=lifecycle, error=error)
        rows.append(row)
        write_json(seed_root / "baseline_row.json", row)
        if lifecycle is not None:
            write_json(seed_root / "launcher_lifecycle.json", lifecycle)
        for transition in _read_jsonl(seed_root / "action_trace.jsonl"):
            all_transitions.append(transition)
            _append_jsonl(output_root / "demonstration_transitions.jsonl", transition)

    _write_csv(output_root / "expert_results.csv", rows)
    summary = summarize_native_expert_results(rows)
    summary["all_infrastructure_valid"] = all(row["infrastructure_valid"] for row in rows)
    summary["total_transition_count"] = len(all_transitions)
    write_json(output_root / "expert_summary.json", summary)
    run_manifest.update(
        {
            "status": "native_expert_complete",
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "native_expert_success_count": summary["native_expert_success_count"],
            "agent_stage_allowed": summary["agent_stage_allowed"],
        }
    )
    write_json(output_root / "run_manifest.json", run_manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["all_infrastructure_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
