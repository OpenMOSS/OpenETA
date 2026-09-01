from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any


def get_git_commit_short() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def sanitize_path_component(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    return normalized.replace("/", "_").replace("\\", "_")


def build_run_id(timestamp: str | None = None, commit: str | None = None) -> str:
    timestamp = timestamp or time.strftime(r"%Y-%m-%d_%H:%M:%S")
    commit = sanitize_path_component(commit) or sanitize_path_component(get_git_commit_short())
    return f"{timestamp}_{commit}" if commit else timestamp


def worker_name(index: int) -> str:
    return f"worker_{index}"


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=4)


def load_worker_metadata(metadata_dir: str | Path) -> dict[str, dict[str, Any]]:
    metadata_dir = Path(metadata_dir)
    worker_payloads: dict[str, dict[str, Any]] = {}
    if not metadata_dir.exists():
        return worker_payloads

    for path in sorted(metadata_dir.glob("worker_*.json")):
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            payload = {}
        worker_payloads[path.stem] = payload if isinstance(payload, dict) else {}
    return worker_payloads


def summarize_worker_metadata(worker_payloads: dict[str, dict[str, Any]]) -> dict[str, Any]:
    total_episodes = 0
    success = 0
    failed = 0
    error = 0
    other_results: dict[str, int] = {}
    total_cost_time = 0.0
    per_worker: dict[str, Any] = {}

    for worker, payload in sorted(worker_payloads.items()):
        result_counts: dict[str, int] = {}
        worker_cost_time = 0.0
        for _, meta in payload.items():
            if not isinstance(meta, dict):
                continue
            result = str(meta.get("result", "unknown"))
            result_counts[result] = result_counts.get(result, 0) + 1

            cost_time = meta.get("cost_time")
            if isinstance(cost_time, (int, float)):
                worker_cost_time += float(cost_time)
                total_cost_time += float(cost_time)

        worker_success = result_counts.get("success", 0)
        worker_failed = result_counts.get("failed", 0)
        worker_error = result_counts.get("error", 0)
        episodes = worker_success + worker_failed + worker_error
        total_episodes += episodes
        success += worker_success
        failed += worker_failed
        error += worker_error
        for result, count in result_counts.items():
            if result not in {"success", "failed", "error"}:
                other_results[result] = other_results.get(result, 0) + count

        per_worker[worker] = {
            "episodes": episodes,
            "success": worker_success,
            "failed": worker_failed,
            "error": worker_error,
            "other_results": {k: v for k, v in result_counts.items() if k not in {"success", "failed", "error"}},
            "success_rate": (result_counts.get("success", 0) / episodes) if episodes > 0 else 0.0,
            "total_cost_time": worker_cost_time,
        }

    return {
        "total_episodes": total_episodes,
        "success": success,
        "failed": failed,
        "error": error,
        "other_results": other_results,
        "success_rate": (success / total_episodes) if total_episodes > 0 else 0.0,
        "total_cost_time": total_cost_time,
        "workers": per_worker,
    }
