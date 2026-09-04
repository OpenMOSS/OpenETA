#!/usr/bin/env python3
"""Run the fixed R1.2 branching-task qualification protocol."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.envs.univtac.branching_qualification import (
    MAX_SIMULATOR_EPISODES,
    SEEDS,
    TASKS,
    qualify_task,
    select_counterfactual,
    select_tactile_icl_task,
    validate_branching_config,
    validate_counterfactual_pair,
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
    parser.add_argument("--aggregate-existing", action="store_true")
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


def _git_value(cwd: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ["git", *args], cwd=cwd, check=False, capture_output=True, text=True
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _episode_row(
    *,
    task: str,
    seed: int,
    condition: str,
    episode_root: Path,
    lifecycle: dict[str, Any] | None,
    error: str | None,
) -> dict[str, Any]:
    child = _read_json(episode_root / "child_result.json")
    final = _read_json(episode_root / "final_result.json")
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
        "task": task,
        "seed": seed,
        "condition": condition,
        "infrastructure_valid": infrastructure_valid,
        "classification": child.get("classification") if child else "native_runtime_abort",
        "plan_success": bool(final and final.get("plan_success")),
        "native_check_success": bool(final and final.get("native_check_success")),
        "native_check_early_stop": final.get("native_check_early_stop") if final else None,
        "expert_episode_success": bool(final and final.get("expert_episode_success")),
        "decision_class": final.get("decision_class") if final else None,
        "check_mid_success": final.get("check_mid_success") if final else None,
        "native_x_move": final.get("native_x_move") if final else None,
        "native_z_move": final.get("native_z_move") if final else None,
        "applied_x_move": final.get("applied_x_move") if final else None,
        "applied_z_move": final.get("applied_z_move") if final else None,
        "final_insert_z": final.get("final_insert_z") if final else None,
        "corrective_time_dilation": final.get("corrective_time_dilation") if final else None,
        "final_time_dilation": final.get("final_time_dilation") if final else None,
        "corrective_moves_executed": final.get("corrective_moves_executed") if final else None,
        "play_once_call_count": child.get("counters", {}).get("play_once_call_count", 0) if child else 0,
        "move_call_count": child.get("counters", {}).get("move_call_count", 0) if child else 0,
        "check_success_call_count": child.get("counters", {}).get("check_success_call_count", 0) if child else 0,
        "check_early_stop_call_count": child.get("counters", {}).get("check_early_stop_call_count", 0) if child else 0,
        "decision_snapshot_complete": bool(final and final.get("decision_snapshot_complete")),
        "action_trace_complete": bool(final and final.get("action_trace_complete")),
        "transition_count": int(final.get("transition_count", 0)) if final else 0,
        "semantic_segments": final.get("semantic_segments", []) if final else [],
        "cleanup_complete": bool(lifecycle and lifecycle.get("cleanup_complete")),
        "episode_path": episode_root.as_posix(),
        "error": error or (child.get("error") if child else "missing child result"),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "task",
        "seed",
        "condition",
        "infrastructure_valid",
        "classification",
        "plan_success",
        "native_check_success",
        "native_check_early_stop",
        "expert_episode_success",
        "decision_class",
        "check_mid_success",
        "native_x_move",
        "native_z_move",
        "applied_x_move",
        "applied_z_move",
        "corrective_moves_executed",
        "play_once_call_count",
        "move_call_count",
        "decision_snapshot_complete",
        "action_trace_complete",
        "cleanup_complete",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _run_episode(
    *,
    config_path: Path,
    config: dict[str, Any],
    runtime_python: Path,
    source_root: Path,
    output_root: Path,
    task: str,
    seed: int,
    condition: str,
    headless: bool,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    command = [
        str(REPO_ROOT / "scripts/univtac/probe_branching_task.py"),
        "--config",
        str(config_path),
        "--source-root",
        str(source_root),
        "--repo-root",
        str(REPO_ROOT),
        "--output-root",
        str(output_root),
        "--task",
        task,
        "--seed",
        str(seed),
        "--condition",
        condition,
    ]
    if headless:
        command.append("--headless")
    spec = ScopedIsaac51LaunchSpec(
        python_executable=runtime_python,
        command=tuple(command),
        cwd=source_root,
        output_root=output_root,
        timeout_seconds=config["timeout_seconds"],
    )
    lifecycle = None
    error = None
    try:
        lifecycle = run_scoped_isaac51_command(spec).to_dict()
    except ScopedIsaac51LaunchError as exc:
        error = f"{type(exc).__name__}: {exc}"
    row = _episode_row(
        task=task,
        seed=seed,
        condition=condition,
        episode_root=output_root,
        lifecycle=lifecycle,
        error=error,
    )
    write_json(output_root / "result_row.json", row)
    if lifecycle is not None:
        write_json(output_root / "launcher_lifecycle.json", lifecycle)
    return row, lifecycle


def _collect_episode_evidence(
    *, output_root: Path, episode_root: Path, row: dict[str, Any]
) -> None:
    decision = _read_json(episode_root / "decision_state.json")
    if decision:
        decision["episode_path"] = episode_root.relative_to(output_root).as_posix()
        _append_jsonl(output_root / "decision_states.jsonl", decision)
    for transition in _read_jsonl(episode_root / "action_trace.jsonl"):
        transition["episode_path"] = episode_root.relative_to(output_root).as_posix()
        transition["condition"] = row["condition"]
        _append_jsonl(output_root / "native_transitions.jsonl", transition)


def _aggregate_existing(output_root: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    task_rows: dict[str, list[dict[str, Any]]] = {task: [] for task in TASKS}
    for task in TASKS:
        for seed in SEEDS:
            episode_root = output_root / task / f"seed_{seed}" / "expert"
            lifecycle = _read_json(episode_root / "launcher_lifecycle.json")
            row = _episode_row(
                task=task,
                seed=seed,
                condition="expert",
                episode_root=episode_root,
                lifecycle=lifecycle,
                error=None,
            )
            write_json(episode_root / "result_row.json", row)
            rows.append(row)
            task_rows[task].append(row)
    task_summaries = {task: qualify_task(task, task_rows[task]) for task in TASKS}
    counterfactual = select_counterfactual(task_summaries)
    counterfactual_rows: list[dict[str, Any]] = []
    selected = "none"
    selection_reason = "no_branching_candidate"
    if counterfactual is not None:
        for condition in ("correct", "wrong"):
            episode_root = (
                output_root
                / "counterfactual"
                / str(counterfactual["task"])
                / f"seed_{counterfactual['seed']}"
                / condition
            )
            lifecycle = _read_json(episode_root / "launcher_lifecycle.json")
            row = _episode_row(
                task=str(counterfactual["task"]),
                seed=int(counterfactual["seed"]),
                condition=condition,
                episode_root=episode_root,
                lifecycle=lifecycle,
                error=None,
            )
            write_json(episode_root / "result_row.json", row)
            rows.append(row)
            counterfactual_rows.append(row)
        if all(row["infrastructure_valid"] for row in counterfactual_rows):
            validate_counterfactual_pair(*counterfactual_rows)
            selected, selection_reason = select_tactile_icl_task(
                counterfactual, *counterfactual_rows
            )
        else:
            selection_reason = "counterfactual_infrastructure_incomplete"
    summary = {
        "schema_version": "openeta.univtac.r12_summary.v1",
        "classification": "tactile_branching_task_qualification_completed",
        "task_summaries": task_summaries,
        "counterfactual_task": counterfactual["task"] if counterfactual else None,
        "counterfactual_seed": counterfactual["seed"] if counterfactual else None,
        "correct_continuation_success": (
            counterfactual_rows[0]["expert_episode_success"]
            if len(counterfactual_rows) == 2
            else None
        ),
        "wrong_continuation_success": (
            counterfactual_rows[1]["expert_episode_success"]
            if len(counterfactual_rows) == 2
            else None
        ),
        "selected_tactile_icl_task": selected,
        "selection_reason": selection_reason,
        "simulator_invocation_count": len(rows),
        "codex_process_count": 0,
        "icl_trial_count": 0,
        "all_infrastructure_valid": all(row["infrastructure_valid"] for row in rows),
        "results": rows,
    }
    _write_csv(output_root / "task_results.csv", rows)
    write_json(output_root / "summary.json", summary)
    manifest = _read_json(output_root / "run_manifest.json") or {}
    manifest.update(
        {
            "status": "completed",
            "classification": summary["classification"],
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "simulator_invocation_count": len(rows),
            "selected_tactile_icl_task": selected,
            "aggregation_recomputed_without_simulator": True,
        }
    )
    write_json(output_root / "run_manifest.json", manifest)
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.expanduser().resolve(strict=True)
    runtime_python = args.runtime_python.expanduser().resolve(strict=True)
    source_root = args.source_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and not args.aggregate_existing:
        raise FileExistsError(f"output root must be fresh: {output_root}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("branching config must be a mapping")
    config = validate_branching_config(payload)
    source_head = _git_value(source_root, "rev-parse", "HEAD")
    source_dirty = _git_value(source_root, "status", "--porcelain")
    if source_head != config["source_commit"]:
        raise RuntimeError(
            f"pinned source HEAD mismatch: expected {config['source_commit']}, got {source_head}"
        )
    if source_dirty:
        raise RuntimeError("pinned source checkout must remain clean")

    if args.aggregate_existing:
        if not output_root.is_dir():
            raise FileNotFoundError(f"existing output root is unavailable: {output_root}")
        summary = _aggregate_existing(output_root)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0 if summary["all_infrastructure_valid"] else 1

    output_root.mkdir(parents=True)
    run_manifest = {
        "schema_version": "openeta.univtac.r12_run.v1",
        "round": "R1.2",
        "classification": "tactile_branching_task_qualification_running",
        "status": "running_experts",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "repo_head": _git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "repo_branch": _git_value(REPO_ROOT, "branch", "--show-current"),
        "source_root": str(source_root),
        "source_head": source_head,
        "runtime_python": str(runtime_python),
        "config": str(config_path),
        "tasks": list(TASKS),
        "seeds": list(SEEDS),
        "simulator_episode_budget": MAX_SIMULATOR_EPISODES,
        "simulator_invocation_count": 0,
        "codex_process_count": 0,
        "icl_trial_count": 0,
    }
    write_json(output_root / "run_manifest.json", run_manifest)

    rows: list[dict[str, Any]] = []
    task_rows: dict[str, list[dict[str, Any]]] = {task: [] for task in TASKS}
    for task in TASKS:
        for seed in SEEDS:
            episode_root = output_root / task / f"seed_{seed}" / "expert"
            run_manifest["simulator_invocation_count"] += 1
            write_json(output_root / "run_manifest.json", run_manifest)
            row, _ = _run_episode(
                config_path=config_path,
                config=config,
                runtime_python=runtime_python,
                source_root=source_root,
                output_root=episode_root,
                task=task,
                seed=seed,
                condition="expert",
                headless=args.headless,
            )
            rows.append(row)
            task_rows[task].append(row)
            _collect_episode_evidence(
                output_root=output_root, episode_root=episode_root, row=row
            )

    task_summaries = {
        task: qualify_task(task, task_rows[task]) for task in TASKS
    }
    counterfactual = select_counterfactual(task_summaries)
    counterfactual_rows: list[dict[str, Any]] = []
    selected = "none"
    selection_reason = "no_branching_candidate"
    if counterfactual is not None:
        run_manifest["status"] = "running_counterfactual"
        for condition in ("correct", "wrong"):
            episode_root = (
                output_root
                / "counterfactual"
                / counterfactual["task"]
                / f"seed_{counterfactual['seed']}"
                / condition
            )
            run_manifest["simulator_invocation_count"] += 1
            if run_manifest["simulator_invocation_count"] > MAX_SIMULATOR_EPISODES:
                raise RuntimeError("R1.2 simulator episode budget exceeded")
            write_json(output_root / "run_manifest.json", run_manifest)
            row, _ = _run_episode(
                config_path=config_path,
                config=config,
                runtime_python=runtime_python,
                source_root=source_root,
                output_root=episode_root,
                task=str(counterfactual["task"]),
                seed=int(counterfactual["seed"]),
                condition=condition,
                headless=args.headless,
            )
            rows.append(row)
            counterfactual_rows.append(row)
            _collect_episode_evidence(
                output_root=output_root, episode_root=episode_root, row=row
            )
        if all(row["infrastructure_valid"] for row in counterfactual_rows):
            validate_counterfactual_pair(*counterfactual_rows)
            selected, selection_reason = select_tactile_icl_task(
                counterfactual, *counterfactual_rows
            )
        else:
            selection_reason = "counterfactual_infrastructure_incomplete"

    summary = {
        "schema_version": "openeta.univtac.r12_summary.v1",
        "classification": "tactile_branching_task_qualification_completed",
        "task_summaries": task_summaries,
        "counterfactual_task": counterfactual["task"] if counterfactual else None,
        "counterfactual_seed": counterfactual["seed"] if counterfactual else None,
        "correct_continuation_success": (
            counterfactual_rows[0]["expert_episode_success"]
            if len(counterfactual_rows) == 2
            else None
        ),
        "wrong_continuation_success": (
            counterfactual_rows[1]["expert_episode_success"]
            if len(counterfactual_rows) == 2
            else None
        ),
        "selected_tactile_icl_task": selected,
        "selection_reason": selection_reason,
        "simulator_invocation_count": run_manifest["simulator_invocation_count"],
        "codex_process_count": 0,
        "icl_trial_count": 0,
        "all_infrastructure_valid": all(row["infrastructure_valid"] for row in rows),
        "results": rows,
    }
    _write_csv(output_root / "task_results.csv", rows)
    write_json(output_root / "summary.json", summary)
    run_manifest.update(
        {
            "status": "completed",
            "classification": summary["classification"],
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "selected_tactile_icl_task": selected,
        }
    )
    write_json(output_root / "run_manifest.json", run_manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["all_infrastructure_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
