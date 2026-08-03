#!/usr/bin/env python3
"""Plan and run fresh Operators across LIBERO tasks or task/seed matrices."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import signal
import subprocess
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from scripts.embodied.episode_summary import summarize_episode


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARTIFACTS_ROOT = REPO_ROOT / "artifacts" / "embodied"
TERMINAL_STATES = {"completed", "failed"}
_SUMMARY_LOCK = threading.Lock()


def _load_value(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _load_json(path: Path) -> dict[str, Any]:
    value = _load_value(path)
    return value if isinstance(value, dict) else {}


def _stop_episode_services(root: str | None) -> None:
    if not root:
        return
    episode = Path(root)
    marker = episode / ".batch-replay-services-stopped"
    if marker.exists():
        return
    result = subprocess.run(
        [str(REPO_ROOT / "scripts/embodied/stop_episode_services.sh"), root],
        cwd=REPO_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode == 0:
        marker.touch()


def _authoritative_tasks(libero_python: Path, libero_dir: Path) -> list[dict[str, Any]]:
    code = """
import json
from sim.env_registry import hot_activate, list_envs

if not hot_activate("libero"):
    raise SystemExit("LIBERO is unavailable")
tasks = [
    {
        "env_id": spec.id,
        "suite": spec.suite,
        "task": spec.task_description,
    }
    for spec in list_envs(env_type="libero")
]
print("OPENETA_LIBERO_TASKS=" + json.dumps(tasks, ensure_ascii=False))
"""
    env = os.environ.copy()
    env.update(
        {
            "LIBERO_DIR": str(libero_dir),
            "PYTHONPATH": str(REPO_ROOT),
        }
    )
    completed = subprocess.run(
        [str(libero_python), "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    marker = "OPENETA_LIBERO_TASKS="
    line = next(
        (
            item
            for item in reversed(completed.stdout.splitlines())
            if item.startswith(marker)
        ),
        None,
    )
    if line is None:
        raise RuntimeError("LIBERO task enumeration returned no task manifest")
    tasks = json.loads(line.removeprefix(marker))
    if not isinstance(tasks, list) or len(tasks) != 130:
        raise RuntimeError(
            f"expected 130 authoritative LIBERO tasks, got {len(tasks)}"
        )
    return tasks


def _observed_model_providers(root: Path) -> set[str]:
    app_events = root / "operator_app_server.jsonl"
    model_providers: set[str] = set()
    if not app_events.is_file():
        return model_providers
    for line in app_events.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        result = event.get("result")
        if isinstance(result, dict):
            provider = result.get("modelProvider")
            if isinstance(provider, str):
                model_providers.add(provider)
    return model_providers


def _observed_reasoning_efforts(root: Path) -> set[str]:
    path = root / "operator_app_server.jsonl"
    efforts: set[str] = set()
    if not path.is_file():
        return efforts
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        result = event.get("result")
        if isinstance(result, dict):
            effort = result.get("reasoningEffort")
            if isinstance(effort, str):
                efforts.add(effort)
    return efforts


def _valid_operator_episode(
    root: Path,
    *,
    expected_provider: str | None = None,
    expected_reasoning_effort: str | None = None,
) -> bool:
    episode = _load_json(root / "episode.json")
    context = root / "operator_context.jsonl"
    contract = root / "operator_context_contract.json"
    if (
        episode.get("status") not in TERMINAL_STATES
        or not contract.is_file()
        or not context.is_file()
        or not context.read_text(encoding="utf-8", errors="replace").strip()
    ):
        return False
    summary = summarize_episode(root)
    valid = bool(
        summary.get("infrastructure_valid")
        and summary.get("finish_episode_count") == 1
    )
    if expected_provider is not None:
        valid = valid and _observed_model_providers(root) == {
            expected_provider
        }
    if expected_reasoning_effort is not None:
        valid = valid and _observed_reasoning_efforts(root) == {
            expected_reasoning_effort
        }
    return valid


def _operator_configuration_matches(
    root: Path,
    *,
    profile: str,
    model: str,
    model_provider: str,
    reasoning_effort: str | None,
    pointcloud_mode: str,
) -> bool:
    contract = _load_json(root / "operator_context_contract.json")
    context_profile = contract.get("context_profile")
    current = _load_json(root / "current.json")
    return bool(
        contract.get("model") == model
        and (
            reasoning_effort is None
            or contract.get("reasoning_effort") == reasoning_effort
        )
        and isinstance(context_profile, dict)
        and context_profile.get("profile_id") == profile
        and _observed_model_providers(root) == {model_provider}
        and (
            reasoning_effort is None
            or _observed_reasoning_efforts(root) == {reasoning_effort}
        )
        and current.get("pointcloud_mode") == pointcloud_mode
    )


def _historical_coverage(artifacts_root: Path) -> dict[str, list[str]]:
    covered: dict[str, list[str]] = defaultdict(list)
    for episode_path in artifacts_root.rglob("episode.json"):
        episode = _load_json(episode_path)
        env_id = episode.get("env_id")
        if (
            isinstance(env_id, str)
            and env_id.startswith("openeta/libero_")
            and _valid_operator_episode(episode_path.parent)
        ):
            covered[env_id].append(str(episode_path.parent))
    return dict(covered)


def _matching_attempts(
    artifacts_root: Path,
    *,
    profile: str,
    model: str,
    model_provider: str,
    reasoning_effort: str | None,
    pointcloud_mode: str,
) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for contract_path in artifacts_root.rglob("operator_context_contract.json"):
        root = contract_path.parent
        if not _operator_configuration_matches(
            root,
            profile=profile,
            model=model,
            model_provider=model_provider,
            reasoning_effort=reasoning_effort,
            pointcloud_mode=pointcloud_mode,
        ):
            continue
        episode = _load_json(root / "episode.json")
        env_id = episode.get("env_id")
        seed = episode.get("seed")
        if (
            not isinstance(env_id, str)
            or not env_id.startswith("openeta/libero_")
            or not isinstance(seed, int)
            or not _valid_operator_episode(
                root,
                expected_provider=model_provider,
                expected_reasoning_effort=reasoning_effort,
            )
        ):
            continue
        summary = summarize_episode(root)
        attempts.append(
            {
                "episode_root": str(root.resolve()),
                "env_id": env_id,
                "seed": seed,
                "episode_status": summary.get("episode_status"),
                "episode_success": summary.get("episode_success"),
                "infrastructure_valid": summary.get("infrastructure_valid"),
                "finish_episode_count": summary.get("finish_episode_count"),
                "tool_calls": summary.get("tool_calls"),
                "resolved_context_sha256": _load_json(
                    root / "operator_context_contract.json"
                ).get("resolved_context_sha256"),
                "source": "bootstrap",
            }
        )
    return sorted(
        attempts,
        key=lambda row: (
            str(row["env_id"]),
            int(row["seed"]),
            str(row["episode_root"]),
        ),
    )


def _interleave_suites(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        groups[str(task["suite"])].append(task)
    for values in groups.values():
        values.sort(key=lambda item: item["env_id"])
    order = [
        "libero_10",
        "libero_goal",
        "libero_object",
        "libero_spatial",
        "libero_90",
    ]
    result: list[dict[str, Any]] = []
    while any(groups.values()):
        for suite in order:
            if groups[suite]:
                result.append(groups[suite].pop(0))
    return result


def _episode_slug(env_id: str) -> str:
    return (
        env_id.removeprefix("openeta/libero_")
        .removesuffix("-v0")
        .replace("/", "-")
    )


def _run_key(item: dict[str, Any]) -> str:
    run_id = item.get("run_id")
    if isinstance(run_id, str) and run_id:
        return run_id
    return f"ordinal-{int(item['ordinal']):03d}"


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def plan_coverage(args: argparse.Namespace) -> int:
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "coverage_manifest.json"
    if manifest_path.exists() and not args.force:
        raise SystemExit(f"refusing to replace existing manifest: {manifest_path}")
    all_tasks = _authoritative_tasks(args.libero_python, args.libero_dir)
    historical = _historical_coverage(args.artifacts_root)
    untested = [
        task for task in all_tasks if task["env_id"] not in historical
    ]
    ordered = _interleave_suites(untested)
    planned = []
    for index, task in enumerate(ordered):
        planned.append(
            {
                **task,
                "ordinal": index,
                "seed": args.seed_base + index,
                "slug": _episode_slug(task["env_id"]),
            }
        )
    manifest = {
        "schema_version": "openeta.libero_operator_coverage.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "authoritative_task_count": len(all_tasks),
        "historically_tested_task_count": len(historical),
        "planned_task_count": len(planned),
        "profile": args.profile,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "model_provider": args.model_provider,
        "pointcloud_mode": args.pointcloud_mode,
        "image_size": [args.image_width, args.image_height],
        "historical_coverage": historical,
        "tasks": planned,
    }
    _write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "authoritative": len(all_tasks),
                "historically_tested": len(historical),
                "planned": len(planned),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def plan_matrix(args: argparse.Namespace) -> int:
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "coverage_manifest.json"
    if manifest_path.exists() and not args.force:
        raise SystemExit(f"refusing to replace existing manifest: {manifest_path}")

    all_tasks = _authoritative_tasks(args.libero_python, args.libero_dir)
    suite_tasks = sorted(
        (task for task in all_tasks if task["suite"] == args.suite),
        key=lambda task: task["env_id"],
    )
    if not suite_tasks:
        raise SystemExit(f"authoritative registry has no tasks for suite {args.suite!r}")
    seeds = list(dict.fromkeys(args.seeds))
    if not seeds:
        raise SystemExit("at least one seed is required")

    planned: list[dict[str, Any]] = []
    for seed in seeds:
        for task in suite_tasks:
            slug = f"{_episode_slug(task['env_id'])}-seed-{seed}"
            planned.append(
                {
                    **task,
                    "ordinal": len(planned),
                    "seed": seed,
                    "slug": slug,
                    "run_id": slug,
                }
            )
    manifest = {
        "schema_version": "openeta.libero_operator_matrix.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "authoritative_task_count": len(all_tasks),
        "suite": args.suite,
        "suite_task_count": len(suite_tasks),
        "seeds": seeds,
        "planned_task_count": len(planned),
        "profile": args.profile,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "model_provider": args.model_provider,
        "pointcloud_mode": args.pointcloud_mode,
        "image_size": [args.image_width, args.image_height],
        "tasks": planned,
    }
    _write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "suite": args.suite,
                "tasks": len(suite_tasks),
                "seeds": seeds,
                "planned": len(planned),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def plan_first_success(args: argparse.Namespace) -> int:
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "first_success_manifest.json"
    if manifest_path.exists() and not args.force:
        raise SystemExit(f"refusing to replace existing manifest: {manifest_path}")

    all_tasks = _authoritative_tasks(args.libero_python, args.libero_dir)
    ordered = _interleave_suites(all_tasks)
    planned = [
        {
            **task,
            "ordinal": index,
            "slug": _episode_slug(task["env_id"]),
        }
        for index, task in enumerate(ordered)
    ]
    bootstrap_attempts = _matching_attempts(
        args.artifacts_root,
        profile=args.profile,
        model=args.model,
        model_provider=args.model_provider,
        reasoning_effort=args.reasoning_effort,
        pointcloud_mode=args.pointcloud_mode,
    )
    successful_envs = {
        str(row["env_id"])
        for row in bootstrap_attempts
        if row.get("episode_success") is True
    }
    manifest = {
        "schema_version": "openeta.libero_operator_first_success.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "authoritative_task_count": len(all_tasks),
        "suite_task_counts": {
            suite: sum(task["suite"] == suite for task in all_tasks)
            for suite in sorted({str(task["suite"]) for task in all_tasks})
        },
        "seed_schedule": list(dict.fromkeys(args.seeds)),
        "stop_condition": "first_operator_finalized_native_success",
        "strict_infrastructure_validity_reported_separately": True,
        "profile": args.profile,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "model_provider": args.model_provider,
        "pointcloud_mode": args.pointcloud_mode,
        "image_size": [args.image_width, args.image_height],
        "bootstrap_artifacts_root": str(args.artifacts_root.resolve()),
        "bootstrap_attempts": bootstrap_attempts,
        "bootstrap_successful_task_count": len(successful_envs),
        "tasks": planned,
    }
    _write_json(manifest_path, manifest)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "tasks": len(planned),
                "suite_task_counts": manifest["suite_task_counts"],
                "seed_schedule": manifest["seed_schedule"],
                "bootstrap_attempts": len(bootstrap_attempts),
                "bootstrap_successful_tasks": len(successful_envs),
                "remaining_tasks": len(planned) - len(successful_envs),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _attempt_root(
    output: Path,
    task: dict[str, Any],
    *,
    expected_provider: str,
    expected_reasoning_effort: str | None,
) -> tuple[Path, bool]:
    task_root = (
        output
        / "episodes"
        / f"{int(task['ordinal']):03d}-{task['slug']}"
    )
    attempts = sorted(task_root.glob("attempt-*"))
    for attempt in attempts:
        if _valid_operator_episode(
            attempt,
            expected_provider=expected_provider,
            expected_reasoning_effort=expected_reasoning_effort,
        ):
            return attempt, True
    next_number = len(attempts) + 1
    return task_root / f"attempt-{next_number:03d}", False


def _summarize_attempt(
    root: Path,
    *,
    task: dict[str, Any],
    returncode: int | None,
    timed_out: bool,
    expected_provider: str,
    expected_reasoning_effort: str | None,
) -> dict[str, Any]:
    if (root / "operator_context_contract.json").is_file():
        summary = summarize_episode(root)
    else:
        summary = {
            "episode_root": str(root),
            "episode_status": None,
            "episode_success": None,
            "infrastructure_valid": False,
            "infrastructure_invalid_reasons": ["no_operator_context_contract"],
            "tool_calls": 0,
        }
    summary.update(
        {
            "env_id": task["env_id"],
            "suite": task["suite"],
            "task": task["task"],
            "ordinal": task["ordinal"],
            "seed": task["seed"],
            "run_id": _run_key(task),
            "launcher_returncode": returncode,
            "launcher_timed_out": timed_out,
        }
    )
    model_providers = _observed_model_providers(root)
    summary["observed_model_providers"] = sorted(model_providers)
    summary["expected_model_provider"] = expected_provider
    summary["model_provider_match"] = model_providers == {
        expected_provider
    }
    if not summary["model_provider_match"]:
        summary["infrastructure_valid"] = False
        reasons = list(summary.get("infrastructure_invalid_reasons") or [])
        reasons.append("operator_model_provider_mismatch")
        summary["infrastructure_invalid_reasons"] = sorted(set(reasons))
    reasoning_efforts = _observed_reasoning_efforts(root)
    summary["observed_reasoning_efforts"] = sorted(reasoning_efforts)
    summary["expected_reasoning_effort"] = expected_reasoning_effort
    summary["reasoning_effort_match"] = (
        expected_reasoning_effort is None
        or reasoning_efforts == {expected_reasoning_effort}
    )
    if not summary["reasoning_effort_match"]:
        summary["infrastructure_valid"] = False
        reasons = list(summary.get("infrastructure_invalid_reasons") or [])
        reasons.append("operator_reasoning_effort_mismatch")
        summary["infrastructure_invalid_reasons"] = sorted(set(reasons))
    return summary


def _run_one(
    output: Path,
    manifest: dict[str, Any],
    task: dict[str, Any],
    *,
    base_port: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    expected_provider = str(manifest["model_provider"])
    expected_reasoning_effort = manifest.get("reasoning_effort")
    expected_reasoning_effort = (
        str(expected_reasoning_effort)
        if expected_reasoning_effort is not None
        else None
    )
    root, complete = _attempt_root(
        output,
        task,
        expected_provider=expected_provider,
        expected_reasoning_effort=expected_reasoning_effort,
    )
    if complete:
        return _summarize_attempt(
            root,
            task=task,
            returncode=0,
            timed_out=False,
            expected_provider=expected_provider,
            expected_reasoning_effort=expected_reasoning_effort,
        )
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "coverage-launch.log"
    port = base_port + int(task["ordinal"]) * 10
    env = os.environ.copy()
    env.update(
        {
            "OPENETA_OPERATOR_CONTEXT_PROFILE": str(manifest["profile"]),
            "OPENETA_OPERATOR_MODEL": str(manifest["model"]),
            "OPENETA_OPERATOR_REASONING_EFFORT": str(
                manifest.get("reasoning_effort") or "medium"
            ),
            "OPENETA_OPERATOR_MODEL_PROVIDER": str(
                manifest["model_provider"]
            ),
            "OPENETA_LIBERO_SEED": str(task["seed"]),
            "OPENETA_OPERATOR_TASK": str(task["task"]),
            "OPENETA_LIBERO_ENV_ID": str(task["env_id"]),
            "OPENETA_OPERATOR_POINTCLOUD_MODE": str(
                manifest["pointcloud_mode"]
            ),
            "OPENETA_OPERATOR_IMAGE_WIDTH": str(manifest["image_size"][0]),
            "OPENETA_OPERATOR_IMAGE_HEIGHT": str(manifest["image_size"][1]),
            "OPENETA_SIM_PORT": str(port),
        }
    )
    timed_out = False
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                str(
                    REPO_ROOT
                    / "scripts"
                    / "embodied"
                    / "launch_operator_chrome.sh"
                ),
                str(root),
            ],
            cwd=REPO_ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                returncode = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait(timeout=10)
    return _summarize_attempt(
        root,
        task=task,
        returncode=returncode,
        timed_out=timed_out,
        expected_provider=expected_provider,
        expected_reasoning_effort=expected_reasoning_effort,
    )


def _write_summary(output: Path, rows: list[dict[str, Any]]) -> None:
    with _SUMMARY_LOCK:
        ordered = sorted(rows, key=lambda row: int(row["ordinal"]))
        _write_json(output / "coverage_summary.json", ordered)


def _discover_first_success_attempts(
    output: Path,
    manifest: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    bootstrap = manifest.get("bootstrap_attempts")
    if isinstance(bootstrap, list):
        for row in bootstrap:
            if isinstance(row, dict) and row.get("episode_root"):
                rows[str(row["episode_root"])] = row

    tasks = manifest.get("tasks")
    task_by_env = {
        str(task["env_id"]): task
        for task in tasks
        if isinstance(task, dict) and task.get("env_id")
    }
    for episode_path in (output / "episodes").glob("*/*/episode.json"):
        root = episode_path.parent
        episode = _load_json(episode_path)
        env_id = episode.get("env_id")
        seed = episode.get("seed")
        task = task_by_env.get(str(env_id))
        if task is None or not isinstance(seed, int):
            continue
        if not _operator_configuration_matches(
            root,
            profile=str(manifest["profile"]),
            model=str(manifest["model"]),
            model_provider=str(manifest["model_provider"]),
            reasoning_effort=(
                str(manifest["reasoning_effort"])
                if manifest.get("reasoning_effort") is not None
                else None
            ),
            pointcloud_mode=str(manifest["pointcloud_mode"]),
        ):
            continue
        row = _summarize_attempt(
            root,
            task={**task, "seed": seed},
            returncode=None,
            timed_out=False,
            expected_provider=str(manifest["model_provider"]),
            expected_reasoning_effort=(
                str(manifest["reasoning_effort"])
                if manifest.get("reasoning_effort") is not None
                else None
            ),
        )
        row["source"] = "first_success"
        rows[str(root.resolve())] = row
    return sorted(
        rows.values(),
        key=lambda row: (
            int(task_by_env.get(str(row.get("env_id")), {}).get("ordinal", 10**9)),
            int(row.get("seed", 10**9)),
            str(row.get("episode_root")),
        ),
    )


def _first_success_task_rows(
    manifest: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    attempts_by_env: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in attempts:
        attempts_by_env[str(row.get("env_id"))].append(row)
    rows: list[dict[str, Any]] = []
    for task in manifest["tasks"]:
        task_attempts = sorted(
            attempts_by_env.get(str(task["env_id"]), []),
            key=lambda row: (
                int(row.get("seed", 10**9)),
                str(row.get("episode_root")),
            ),
        )
        valid_attempts = [
            row
            for row in task_attempts
            if row.get("infrastructure_valid") is True
            and row.get("finish_episode_count") == 1
        ]
        finalized_native_successes = [
            row
            for row in task_attempts
            if row.get("episode_status") == "completed"
            and row.get("episode_success") is True
            and row.get("finish_episode_count") == 1
            and row.get("model_provider_match") is not False
        ]
        strict_successes = [
            row
            for row in finalized_native_successes
            if row.get("infrastructure_valid") is True
        ]
        rows.append(
            {
                **task,
                "success": bool(finalized_native_successes),
                "successful_seed": (
                    int(finalized_native_successes[0]["seed"])
                    if finalized_native_successes
                    else None
                ),
                "successful_attempt_infrastructure_valid": (
                    bool(strict_successes)
                    if finalized_native_successes
                    else None
                ),
                "valid_attempt_count": len(valid_attempts),
                "valid_failed_seed_count": len(
                    {
                        int(row["seed"])
                        for row in valid_attempts
                        if row.get("episode_success") is not True
                    }
                ),
                "attempts": task_attempts,
            }
        )
    return rows


def _write_first_success_state(
    output: Path,
    manifest: dict[str, Any],
    attempts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = _first_success_task_rows(manifest, attempts)
    _write_json(output / "first_success_attempts.json", attempts)
    _write_json(output / "first_success_summary.json", rows)
    suite_summary: dict[str, dict[str, int]] = {}
    for suite in sorted({str(row["suite"]) for row in rows}):
        suite_rows = [row for row in rows if row["suite"] == suite]
        suite_summary[suite] = {
            "tasks": len(suite_rows),
            "successful": sum(row["success"] for row in suite_rows),
            "strict_infrastructure_valid_successful": sum(
                row["successful_attempt_infrastructure_valid"] is True
                for row in suite_rows
            ),
            "remaining": sum(not row["success"] for row in suite_rows),
        }
    _write_json(
        output / "first_success_progress.json",
        {
            "schema_version": "openeta.libero_operator_first_success_progress.v1",
            "task_count": len(rows),
            "successful_task_count": sum(row["success"] for row in rows),
            "strict_infrastructure_valid_successful_task_count": sum(
                row["successful_attempt_infrastructure_valid"] is True
                for row in rows
            ),
            "remaining_task_count": sum(not row["success"] for row in rows),
            "infrastructure_valid_attempt_count": sum(
                row.get("infrastructure_valid") is True for row in attempts
            ),
            "suite_summary": suite_summary,
        },
    )
    return rows


def run_first_success(args: argparse.Namespace) -> int:
    output = args.output.expanduser().resolve()
    manifest = _load_json(output / "first_success_manifest.json")
    if not manifest:
        raise SystemExit(f"missing first-success manifest under {output}")
    tasks = manifest.get("tasks")
    seeds = manifest.get("seed_schedule")
    if not isinstance(tasks, list) or not isinstance(seeds, list):
        raise SystemExit("first-success manifest has no tasks or seed schedule")

    attempts = _discover_first_success_attempts(output, manifest)
    rows = _write_first_success_state(output, manifest, attempts)
    started = time.monotonic()
    task_count = len(tasks)
    new_attempts = 0
    for seed_index, seed in enumerate(seeds):
        attempted_seeds = {
            (str(row.get("env_id")), int(row["seed"]))
            for row in attempts
            if isinstance(row.get("seed"), int)
            and (
                row.get("episode_status")
                in {"completed", "failed", "aborted", "stopped"}
                or row.get("launcher_timed_out") is True
            )
        }
        successful = {
            str(row["env_id"]) for row in rows if row.get("success") is True
        }
        selected = [
            {
                **task,
                "seed": int(seed),
                "ordinal": int(task["ordinal"]) + seed_index * task_count,
                "slug": f"{task['slug']}-seed-{int(seed)}",
                "run_id": f"{task['slug']}-seed-{int(seed)}",
            }
            for task in tasks
            if str(task["env_id"]) not in successful
            and (str(task["env_id"]), int(seed)) not in attempted_seeds
        ]
        if args.max_new_attempts is not None:
            remaining = args.max_new_attempts - new_attempts
            if remaining <= 0:
                return 0
            selected = selected[:remaining]
        if not selected:
            continue
        print(
            json.dumps(
                {
                    "seed_pass": int(seed),
                    "scheduled": len(selected),
                    "already_successful": len(successful),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.jobs
        ) as executor:
            future_to_task = {
                executor.submit(
                    _run_one,
                    output,
                    manifest,
                    task,
                    base_port=args.base_port,
                    timeout_seconds=args.timeout_seconds,
                ): task
                for task in selected
            }
            for future in concurrent.futures.as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    row = future.result()
                except Exception as exc:
                    row = {
                        **task,
                        "episode_root": None,
                        "episode_status": None,
                        "episode_success": None,
                        "infrastructure_valid": False,
                        "infrastructure_invalid_reasons": [
                            "first_success_runner_error:"
                            f"{type(exc).__name__}:{exc}"
                        ],
                    }
                row["source"] = "first_success"
                _stop_episode_services(row.get("episode_root"))
                attempts.append(row)
                rows = _write_first_success_state(output, manifest, attempts)
                progress = _load_json(output / "first_success_progress.json")
                print(
                    json.dumps(
                        {
                            "env_id": task["env_id"],
                            "suite": task["suite"],
                            "seed": task["seed"],
                            "status": row.get("episode_status"),
                            "success": row.get("episode_success"),
                            "infrastructure_valid": row.get(
                                "infrastructure_valid"
                            ),
                            "successful_tasks": progress.get(
                                "successful_task_count"
                            ),
                            "remaining_tasks": progress.get(
                                "remaining_task_count"
                            ),
                            "elapsed_s": round(time.monotonic() - started, 1),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        new_attempts += len(selected)
    return 0


def run_coverage(args: argparse.Namespace) -> int:
    output = args.output.expanduser().resolve()
    manifest = _load_json(output / "coverage_manifest.json")
    if not manifest:
        raise SystemExit(f"missing coverage manifest under {output}")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list):
        raise SystemExit("coverage manifest has no task list")
    selected = tasks
    if args.start is not None:
        selected = [
            task for task in selected if int(task["ordinal"]) >= args.start
        ]
    if args.limit is not None:
        selected = selected[: args.limit]
    prior = _load_value(output / "coverage_summary.json")
    rows_by_run: dict[str, dict[str, Any]] = {}
    if isinstance(prior, list):
        rows_by_run.update(
            {
                _run_key(row): row
                for row in prior
                if isinstance(row, dict)
                and row.get("env_id")
                and row.get("ordinal") is not None
            }
        )
    selected = [
        task
        for task in selected
        if not (
            _run_key(task) in rows_by_run
            and rows_by_run[_run_key(task)].get("infrastructure_valid") is True
            and rows_by_run[_run_key(task)].get("episode_status")
            in TERMINAL_STATES
            and rows_by_run[_run_key(task)].get("finish_episode_count") == 1
            and rows_by_run[_run_key(task)].get("observed_model_providers")
            == [str(manifest["model_provider"])]
        )
    ]
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.jobs
    ) as executor:
        future_to_task = {
            executor.submit(
                _run_one,
                output,
                manifest,
                task,
                base_port=args.base_port,
                timeout_seconds=args.timeout_seconds,
            ): task
            for task in selected
        }
        for future in concurrent.futures.as_completed(future_to_task):
            task = future_to_task[future]
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    **task,
                    "episode_root": None,
                    "episode_status": None,
                    "episode_success": None,
                    "infrastructure_valid": False,
                    "infrastructure_invalid_reasons": [
                        f"coverage_runner_error:{type(exc).__name__}:{exc}"
                    ],
                }
            _stop_episode_services(row.get("episode_root"))
            rows_by_run[_run_key(task)] = row
            rows = list(rows_by_run.values())
            _write_summary(output, rows)
            print(
                json.dumps(
                    {
                        "completed": len(rows_by_run),
                        "planned": len(tasks),
                        "env_id": task["env_id"],
                        "seed": task["seed"],
                        "status": row.get("episode_status"),
                        "success": row.get("episode_success"),
                        "infrastructure_valid": row.get(
                            "infrastructure_valid"
                        ),
                        "provider": row.get("observed_model_providers"),
                        "elapsed_s": round(time.monotonic() - started, 1),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan")
    plan.add_argument("--output", type=Path, required=True)
    plan.add_argument("--artifacts-root", type=Path, default=DEFAULT_ARTIFACTS_ROOT)
    plan.add_argument(
        "--libero-python",
        type=Path,
        default=REPO_ROOT / "sim" / "venvs" / "libero" / "bin" / "python",
    )
    plan.add_argument(
        "--libero-dir",
        type=Path,
        default=Path(os.environ.get("LIBERO_DIR", "third_party/LIBERO")),
    )
    plan.add_argument(
        "--profile",
        default="openeta-light",
    )
    plan.add_argument("--model", default="gpt-5.6-terra")
    plan.add_argument("--reasoning-effort", default="medium")
    plan.add_argument("--model-provider", default="openai")
    plan.add_argument(
        "--pointcloud-mode",
        default="live-multiview-consensus",
    )
    plan.add_argument("--image-width", type=int, default=512)
    plan.add_argument("--image-height", type=int, default=512)
    plan.add_argument("--seed-base", type=int, default=101)
    plan.add_argument("--force", action="store_true")
    plan.set_defaults(function=plan_coverage)

    matrix = subparsers.add_parser("plan-matrix")
    matrix.add_argument("--output", type=Path, required=True)
    matrix.add_argument(
        "--libero-python",
        type=Path,
        default=REPO_ROOT / "sim" / "venvs" / "libero" / "bin" / "python",
    )
    matrix.add_argument(
        "--libero-dir",
        type=Path,
        default=Path(os.environ.get("LIBERO_DIR", "third_party/LIBERO")),
    )
    matrix.add_argument("--suite", required=True)
    matrix.add_argument("--seeds", type=int, nargs="+", required=True)
    matrix.add_argument(
        "--profile",
        default="openeta-light",
    )
    matrix.add_argument("--model", default="gpt-5.6-terra")
    matrix.add_argument("--reasoning-effort", default="medium")
    matrix.add_argument("--model-provider", default="openai")
    matrix.add_argument(
        "--pointcloud-mode",
        default="live-multiview-consensus",
    )
    matrix.add_argument("--image-width", type=int, default=512)
    matrix.add_argument("--image-height", type=int, default=512)
    matrix.add_argument("--force", action="store_true")
    matrix.set_defaults(function=plan_matrix)

    first_success = subparsers.add_parser("plan-first-success")
    first_success.add_argument("--output", type=Path, required=True)
    first_success.add_argument(
        "--artifacts-root",
        type=Path,
        default=DEFAULT_ARTIFACTS_ROOT,
    )
    first_success.add_argument(
        "--libero-python",
        type=Path,
        default=REPO_ROOT / "sim" / "venvs" / "libero" / "bin" / "python",
    )
    first_success.add_argument(
        "--libero-dir",
        type=Path,
        default=Path(os.environ.get("LIBERO_DIR", "third_party/LIBERO")),
    )
    first_success.add_argument("--seeds", type=int, nargs="+", required=True)
    first_success.add_argument(
        "--profile",
        default="openeta-light",
    )
    first_success.add_argument("--model", default="gpt-5.6-terra")
    first_success.add_argument("--reasoning-effort", default="medium")
    first_success.add_argument("--model-provider", default="openai")
    first_success.add_argument(
        "--pointcloud-mode",
        default="live-multiview-consensus",
    )
    first_success.add_argument("--image-width", type=int, default=512)
    first_success.add_argument("--image-height", type=int, default=512)
    first_success.add_argument("--force", action="store_true")
    first_success.set_defaults(function=plan_first_success)

    run = subparsers.add_parser("run")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--jobs", type=int, default=1)
    run.add_argument("--base-port", type=int, default=11000)
    run.add_argument("--timeout-seconds", type=float, default=1800.0)
    run.add_argument("--start", type=int)
    run.add_argument("--limit", type=int)
    run.set_defaults(function=run_coverage)

    run_first = subparsers.add_parser("run-first-success")
    run_first.add_argument("--output", type=Path, required=True)
    run_first.add_argument("--jobs", type=int, default=1)
    run_first.add_argument("--base-port", type=int, default=14000)
    run_first.add_argument("--timeout-seconds", type=float, default=5400.0)
    run_first.add_argument("--max-new-attempts", type=int)
    run_first.set_defaults(function=run_first_success)

    args = parser.parse_args()
    return args.function(args)


if __name__ == "__main__":
    raise SystemExit(main())
