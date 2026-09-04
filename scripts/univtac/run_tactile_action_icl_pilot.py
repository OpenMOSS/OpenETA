#!/usr/bin/env python3
"""Run the fixed R1.1 live tactile-action-outcome ICL continuation pilot."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.error
import urllib.request
from concurrent.futures import Future, ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.univtac.run_codex_readonly_observation import _run_to_files
from sim.envs.univtac.codex_readonly import read_jsonl, summarize_codex_exec
from sim.envs.univtac.scoped_isaac51_launcher import (
    ScopedIsaac51LaunchResult,
    ScopedIsaac51LaunchSpec,
    run_scoped_isaac51_command,
)
from sim.envs.univtac.tactile_action_icl import (
    CORRUPTED_LABEL,
    QUERY_SPECS,
    R11_MCP_TOOLS,
    build_demonstration_bank,
    build_r11_codex_command,
    stage_r11_condition,
    summarize_r11_results,
    validate_r11_config,
)
from sim.envs.univtac.trace import write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--r10-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--codex-bin", default=os.environ.get("OPENETA_CODEX_BIN", "codex"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--headless", action="store_true")
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _post(worker_url: str, endpoint: str) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{worker_url.rstrip('/')}{endpoint}",
        data=b"{}",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict) or payload.get("ok") is not True:
        raise RuntimeError(f"worker {endpoint} returned invalid response: {payload!r}")
    return payload


def _wait_for_worker(
    ready_path: Path,
    future: Future[ScopedIsaac51LaunchResult],
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if ready_path.is_file():
            return _read_json(ready_path)
        if future.done():
            lifecycle = future.result()
            raise RuntimeError(
                f"live worker exited before ready with returncode {lifecycle.returncode}"
            )
        time.sleep(1)
    raise TimeoutError("R1.1 worker did not reach ready state")


def _action_started(attempt_root: Path) -> bool:
    trace = attempt_root / "action_trace.jsonl"
    return trace.is_file() and bool(trace.read_text(encoding="utf-8").strip())


def _validate_operator_rows(
    rows: list[dict[str, Any]], *, image_mode: str, has_demos: bool
) -> dict[str, Any]:
    tools = [str(row.get("tool")) for row in rows]
    if tools[:2] != ["review_demonstrations", "observe"]:
        raise RuntimeError("R1.1 must review demonstrations then observe")
    if tools.count("review_demonstrations") != 1 or tools.count("observe") != 1:
        raise RuntimeError("R1.1 review and observe must each be called exactly once")
    if tools.count("finish_episode") != 1 or tools[-1] != "finish_episode":
        raise RuntimeError("R1.1 finish_episode must be called exactly once and last")
    if set(tools) - set(R11_MCP_TOOLS):
        raise RuntimeError("unexpected R1.1 MCP tool")
    expected_observation_images = 2 if image_mode == "visual_only" else 4
    for row in rows:
        tool = row["tool"]
        count = len(row.get("response_image_paths", []))
        expected = (
            3
            if tool == "review_demonstrations" and has_demos
            else expected_observation_images
            if tool in {"observe", "execute_skill"}
            else 0
        )
        if count != expected:
            raise RuntimeError(f"{tool} returned {count} images, expected {expected}")
    sequence = [row["arguments"]["skill"] for row in rows if row["tool"] == "execute_skill"]
    if len(sequence) > 3 or len(sequence) != len(set(sequence)):
        raise RuntimeError("R1.1 Agent skill budget or once-only rule was violated")
    return {"tool_sequence": tools, "agent_skill_sequence": sequence}


def _invalid_execute_count(attempt_root: Path) -> int:
    return sum(
        row.get("path") == "/execute" and row.get("ok") is False
        for row in read_jsonl(attempt_root / "worker_requests.jsonl")
    )


def _result_from_evidence(
    attempt_root: Path, *, native_evaluation_required: bool
) -> dict[str, Any]:
    condition = _read_json(attempt_root / "condition.json")
    visible = condition["agent_visible"]
    host = condition["host_only"]
    operator_rows = read_jsonl(attempt_root / "operator_context.jsonl")
    selected = [
        row["arguments"]["skill"]
        for row in operator_rows
        if row.get("tool") == "execute_skill"
    ]
    expected = list(host["expected_remaining_sequence"])
    expected_first = expected[0]
    first = selected[0] if selected else None
    final_path = attempt_root / "final_result.json"
    final_result = _read_json(final_path) if final_path.is_file() else {}
    if native_evaluation_required and not final_result:
        raise RuntimeError("native evaluation evidence is missing")
    action_rows = read_jsonl(attempt_root / "action_trace.jsonl")
    plan_success = (
        final_result.get("plan_success")
        if final_result
        else action_rows[-1].get("plan_success_after")
        if action_rows
        else None
    )
    codex_summary = summarize_codex_exec(read_jsonl(attempt_root / "codex_exec.jsonl"))
    lifecycle_path = attempt_root / "codex_lifecycle.json"
    codex_lifecycle = _read_json(lifecycle_path) if lifecycle_path.is_file() else {}
    return {
        "seed": int(host["seed"]),
        "query_start_state": host["query_start_state"],
        "support_seed": int(host["support_seed"]),
        "condition": visible["condition"],
        "expected_first_skill": expected_first,
        "agent_first_skill": first,
        "first_skill_correct": first == expected_first,
        "expected_remaining_sequence": expected,
        "agent_skill_sequence": selected,
        "exact_sequence": selected == expected,
        "corrupted_label_followed": (
            first == CORRUPTED_LABEL[expected_first]
            if visible["condition"] == "action_swapped_icl_multimodal"
            else False
        ),
        "agent_skill_count": len(selected),
        "plan_success": plan_success,
        "native_check_success": final_result.get("native_check_success"),
        "native_early_stop": final_result.get("native_check_early_stop"),
        "native_continuation_success": bool(
            final_result.get("agent_operation_smoke_success", False)
        ),
        "native_evaluation_available": bool(final_result),
        "codex_completed": bool(
            codex_lifecycle.get("returncode") == 0
            and codex_lifecycle.get("timed_out") is False
        ),
        "invalid_or_repeated_skill_count": _invalid_execute_count(attempt_root),
        "duration_seconds": _read_json(attempt_root / "episode.json").get(
            "duration_seconds"
        ),
        "token_usage": codex_summary.get("usage"),
        "run": attempt_root.as_posix(),
    }


def _run_attempt(
    *,
    attempt_root: Path,
    config_path: Path,
    config: dict[str, Any],
    runtime_python: Path,
    source_root: Path,
    codex_bin: str,
    auth_path: Path,
    headless: bool,
) -> dict[str, Any]:
    condition = _read_json(attempt_root / "condition.json")
    visible = condition["agent_visible"]
    host = condition["host_only"]
    seed = int(host["seed"])
    started = datetime.now(timezone.utc).isoformat()
    episode = {
        "schema_version": "openeta.univtac.r11.episode.v1",
        "round": "R1.1",
        "task": "pull_out_key",
        "seed": seed,
        "condition": visible["condition"],
        "support_seed": int(host["support_seed"]),
        "model": config["model"],
        "reasoning_effort": config["reasoning_effort"],
        "status": "starting_worker",
        "started_at": started,
        "agent_action_count": 0,
        "error": None,
    }
    worker_command = [
        str(REPO_ROOT / "scripts/univtac/serve_pull_out_key_live_worker.py"),
        "--config",
        str(config_path),
        "--source-root",
        str(source_root),
        "--repo-root",
        str(REPO_ROOT),
        "--output-root",
        str(attempt_root),
        "--icl-condition",
        str(attempt_root / "condition.json"),
        "--port",
        "0",
    ]
    if headless:
        worker_command.append("--headless")
    spec = ScopedIsaac51LaunchSpec(
        python_executable=runtime_python,
        command=tuple(worker_command),
        cwd=source_root,
        output_root=attempt_root,
        timeout_seconds=config["worker_timeout_seconds"],
    )
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(run_scoped_isaac51_command, spec)
    worker_url: str | None = None
    codex_home: Path | None = None
    try:
        ready = _wait_for_worker(
            attempt_root / "ready.json", future, config["worker_timeout_seconds"]
        )
        worker_url = str(ready["worker_url"])
        workspace = attempt_root / "operator-workspace"
        codex_home = attempt_root / "runtime/codex-home"
        workspace.mkdir()
        codex_home.mkdir(parents=True)
        subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
        (codex_home / "auth.json").symlink_to(auth_path)
        command = build_r11_codex_command(
            codex_bin=codex_bin,
            workspace=workspace,
            repo_root=REPO_ROOT,
            episode_root=attempt_root,
            worker_url=worker_url,
            final_response_path=attempt_root / "agent_final.md",
            model=config["model"],
            reasoning_effort=config["reasoning_effort"],
        )
        write_json(attempt_root / "codex_command.json", {"command": command})
        environment = dict(os.environ)
        environment["CODEX_HOME"] = str(codex_home)
        episode["status"] = "codex_running"
        write_json(attempt_root / "episode.json", episode)
        codex_lifecycle = _run_to_files(
            command,
            cwd=workspace,
            environment=environment,
            stdout_path=attempt_root / "codex_exec.jsonl",
            stderr_path=attempt_root / "codex_stderr.log",
            timeout_seconds=config["codex_timeout_seconds"],
        )
        write_json(attempt_root / "codex_lifecycle.json", codex_lifecycle)
        if codex_lifecycle["returncode"] != 0 or codex_lifecycle["timed_out"]:
            raise RuntimeError("Codex exec failed during R1.1 operation")
        if not future.done():
            _post(worker_url, "/host_finalize")
        lifecycle = future.result(timeout=120).to_dict()
        write_json(attempt_root / "worker_lifecycle.json", lifecycle)
        if lifecycle["returncode"] != 0 or lifecycle["timed_out"] or not lifecycle["cleanup_complete"]:
            raise RuntimeError("R1.1 worker did not exit cleanly")
        child = _read_json(attempt_root / "child_result.json")
        if child["status"] != "completed":
            raise RuntimeError("R1.1 worker did not complete host evaluation")
        operator_rows = read_jsonl(attempt_root / "operator_context.jsonl")
        trace = _validate_operator_rows(
            operator_rows,
            image_mode=visible["image_mode"],
            has_demos=bool(visible["demonstrations"]),
        )
        selected = trace["agent_skill_sequence"]
        row = _result_from_evidence(attempt_root, native_evaluation_required=True)
        row["duration_seconds"] = child.get("duration_seconds")
        episode.update(
            {
                "status": "completed",
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": child.get("duration_seconds"),
                "agent_action_count": len(selected),
                "tool_sequence": trace["tool_sequence"],
                "result": row,
            }
        )
        write_json(attempt_root / "episode.json", episode)
        write_json(attempt_root / "result.json", row)
        return row
    except Exception as exc:
        error = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(attempt_root / "exception.json", error)
        episode.update(
            {
                "status": "failed",
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "agent_action_count": len(
                    read_jsonl(attempt_root / "action_trace.jsonl")
                ),
                "error": error,
            }
        )
        write_json(attempt_root / "episode.json", episode)
        raise
    finally:
        if not future.done() and worker_url is not None:
            try:
                _post(worker_url, "/host_finalize")
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                pass
        executor.shutdown(wait=True)
        if codex_home is not None and codex_home.is_dir():
            (codex_home / "auth.json").unlink(missing_ok=True)
            shutil.rmtree(codex_home)


def _write_results(output_root: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "seed", "query_start_state", "support_seed", "condition",
        "expected_first_skill", "agent_first_skill", "first_skill_correct",
        "expected_remaining_sequence", "agent_skill_sequence", "exact_sequence",
        "corrupted_label_followed", "agent_skill_count", "plan_success",
        "native_check_success", "native_early_stop", "native_continuation_success",
        "native_evaluation_available", "codex_completed",
        "invalid_or_repeated_skill_count",
        "duration_seconds", "token_usage", "run",
    ]
    with (output_root / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: json.dumps(value) if isinstance(value, (dict, list)) else value for key, value in row.items()}
            )
    if len(rows) == 12:
        write_json(output_root / "summary.json", summarize_r11_results(rows))
    with (output_root / "raw_answers.md").open("w", encoding="utf-8") as stream:
        for row in rows:
            answer = Path(row["run"]) / "agent_final.md"
            stream.write(f"## seed {row['seed']} · {row['condition']}\n\n")
            text = answer.read_text(encoding="utf-8").strip() if answer.is_file() else "unavailable"
            stream.write(text + "\n\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.expanduser().resolve(strict=True)
    runtime_python = args.runtime_python.expanduser().resolve(strict=True)
    source_root = args.source_root.expanduser().resolve(strict=True)
    r10_root = args.r10_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and not args.resume:
        raise FileExistsError(f"R1.1 output root must be fresh: {output_root}")
    config = validate_r11_config(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    codex_bin = shutil.which(args.codex_bin)
    auth_path = Path.home() / ".codex/auth.json"
    if codex_bin is None or not auth_path.is_file():
        raise FileNotFoundError("Codex CLI or existing authentication is unavailable")
    output_root.mkdir(parents=True, exist_ok=args.resume)
    bank_root = output_root / "demonstration_bank"
    if args.resume:
        bank = _read_json(bank_root / "manifest.json")
    else:
        bank = build_demonstration_bank(r10_root=r10_root, output_root=bank_root)
        write_json(bank_root / "manifest.json", bank)
        write_json(
            output_root / "run_manifest.json",
            {
                "round": "R1.1",
                "status": "running",
                "semantic_episode_count": 12,
                "model": config["model"],
                "reasoning_effort": config["reasoning_effort"],
                "started_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    rows = []
    for result_path in sorted(output_root.rglob("result.json")):
        previous = _read_json(result_path)
        row = _result_from_evidence(
            result_path.parent, native_evaluation_required=False
        )
        if row["duration_seconds"] is None:
            row["duration_seconds"] = previous.get("duration_seconds")
        write_json(result_path, row)
        rows.append(row)
    completed = {(int(row["seed"]), row["condition"]) for row in rows}
    for seed, spec in QUERY_SPECS.items():
        for condition_name in spec["condition_order"]:
            cell_root = output_root / "episodes" / f"seed_{seed}" / condition_name
            if (seed, condition_name) in completed:
                continue
            existing = sorted(cell_root.glob("attempt_*/episode.json"))
            action_attempt = next(
                (path.parent for path in existing if _action_started(path.parent)), None
            )
            if action_attempt is not None:
                row = _result_from_evidence(
                    action_attempt, native_evaluation_required=False
                )
                write_json(action_attempt / "result.json", row)
                rows.append(row)
                completed.add((seed, condition_name))
                _write_results(output_root, rows)
                continue
            final_error: Exception | None = None
            for attempt in range(len(existing) + 1, 3):
                attempt_root = cell_root / f"attempt_{attempt:02d}"
                stage_r11_condition(
                    bank=bank,
                    seed=seed,
                    condition=condition_name,
                    bank_root=bank_root,
                    episode_root=attempt_root,
                )
                try:
                    row = _run_attempt(
                        attempt_root=attempt_root,
                        config_path=config_path,
                        config=config,
                        runtime_python=runtime_python,
                        source_root=source_root,
                        codex_bin=codex_bin,
                        auth_path=auth_path,
                        headless=args.headless,
                    )
                    rows.append(row)
                    completed.add((seed, condition_name))
                    _write_results(output_root, rows)
                    final_error = None
                    break
                except Exception as exc:
                    final_error = exc
                    if _action_started(attempt_root) or attempt == 2:
                        raise
            if final_error is not None:
                raise final_error
    _write_results(output_root, rows)
    manifest = _read_json(output_root / "run_manifest.json")
    manifest.update(
        {
            "status": "completed",
            "completed_semantic_episode_count": len(rows),
            "ended_at": datetime.now(timezone.utc).isoformat(),
        }
    )
    write_json(output_root / "run_manifest.json", manifest)
    print(json.dumps(_read_json(output_root / "summary.json"), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
