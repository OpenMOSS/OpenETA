#!/usr/bin/env python3
"""Run one live Codex-controlled bounded Pull Out Key episode."""

from __future__ import annotations

import argparse
import json
import os
import shutil
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
from sim.envs.univtac.live_operation import (
    LIVE_SEED,
    MCP_TOOL_NAMES,
    build_live_codex_command,
    validate_agent_attempt_boundary,
    validate_live_operation_config,
)
from sim.envs.univtac.scoped_isaac51_launcher import (
    ScopedIsaac51LaunchResult,
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
    parser.add_argument("--codex-bin", default=os.environ.get("OPENETA_CODEX_BIN", "codex"))
    parser.add_argument("--attempt", type=int, default=1)
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
    raise TimeoutError("live worker did not reach ready state")


def _validate_operator_trace(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tools = [str(row.get("tool")) for row in rows]
    if not tools or tools[0] != "observe":
        raise RuntimeError("Codex must call observe before any operation skill")
    if tools.count("observe") != 1 or tools.count("finish_episode") != 1:
        raise RuntimeError("Codex must call observe and finish_episode exactly once")
    if tools[-1] != "finish_episode":
        raise RuntimeError("finish_episode must be the final MCP call")
    unexpected = sorted(set(tools) - set(MCP_TOOL_NAMES))
    if unexpected:
        raise RuntimeError(f"unexpected live MCP tools were called: {unexpected}")
    execute_rows = [row for row in rows if row.get("tool") == "execute_skill"]
    if len(execute_rows) > 3:
        raise RuntimeError("Codex exceeded the three-skill action budget")
    for row in rows:
        expected_images = 4 if row.get("tool") in {"observe", "execute_skill"} else 0
        if len(row.get("response_image_paths", [])) != expected_images:
            raise RuntimeError(f"{row.get('tool')} returned an unexpected image count")
    return {
        "tool_sequence": tools,
        "tool_call_count": len(rows),
        "action_tool_call_count": len(execute_rows),
        "selected_skill_sequence": [row["arguments"]["skill"] for row in execute_rows],
    }


def _count_codex_attempts(output_root: Path) -> int:
    """Count persisted Codex invocations instead of trusting stale metadata."""
    return sum(1 for path in (output_root / "agent").rglob("codex_exec.jsonl") if path.is_file())


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.expanduser().resolve(strict=True)
    runtime_python = args.runtime_python.expanduser().resolve(strict=True)
    source_root = args.source_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve(strict=True)
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config_payload, dict):
        raise TypeError("live operation config must be a mapping")
    config = validate_live_operation_config(config_payload)
    validate_agent_attempt_boundary(output_root, args.attempt)
    expert_summary = _read_json(output_root / "expert_summary.json")
    successful = [
        int(row["seed"])
        for row in expert_summary["results"]
        if row["expert_episode_success"]
    ]
    if not successful:
        raise RuntimeError("native expert success count is zero; Agent stage is forbidden")
    selected_seed = min(successful)
    if selected_seed != LIVE_SEED or config["seed"] != selected_seed:
        raise RuntimeError("live Agent must use the lowest successful expert seed")

    codex_bin = shutil.which(args.codex_bin)
    auth_path = Path.home() / ".codex/auth.json"
    if codex_bin is None:
        raise FileNotFoundError(f"Codex CLI is unavailable: {args.codex_bin}")
    if not auth_path.is_file():
        raise FileNotFoundError("existing Codex authentication is unavailable")

    agent_root = (
        output_root / "agent" / f"seed_{selected_seed}"
        if args.attempt == 1
        else output_root / "agent" / f"attempt_{args.attempt:02d}" / f"seed_{selected_seed}"
    )
    if agent_root.exists():
        raise FileExistsError(f"live Agent output root must be fresh: {agent_root}")
    started_at = datetime.now(timezone.utc).isoformat()
    episode = {
        "schema_version": "openeta.univtac.live_codex_episode.v1",
        "round": "R1.0",
        "task": "pull_out_key",
        "seed": selected_seed,
        "attempt": args.attempt,
        "model": config["model"],
        "reasoning_effort": config["reasoning_effort"],
        "status": "starting_worker",
        "started_at": started_at,
        "mcp_tools": list(MCP_TOOL_NAMES),
        "simulator_invocation_count": 1,
        "codex_process_count": 0,
        "agent_action_count": 0,
        "error": None,
    }
    worker_command = [
        str(REPO_ROOT / "scripts" / "univtac" / "serve_pull_out_key_live_worker.py"),
        "--config",
        str(config_path),
        "--source-root",
        str(source_root),
        "--repo-root",
        str(REPO_ROOT),
        "--output-root",
        str(agent_root),
        "--port",
        "0",
    ]
    if args.headless:
        worker_command.append("--headless")
    spec = ScopedIsaac51LaunchSpec(
        python_executable=runtime_python,
        command=tuple(worker_command),
        cwd=source_root,
        output_root=agent_root,
        timeout_seconds=config["worker_timeout_seconds"],
    )
    codex_home: Path | None = None
    worker_url: str | None = None
    future: Future[ScopedIsaac51LaunchResult] | None = None
    executor: ThreadPoolExecutor | None = None
    lifecycle: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    try:
        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(run_scoped_isaac51_command, spec)
        ready = _wait_for_worker(
            agent_root / "ready.json", future, config["worker_timeout_seconds"]
        )
        worker_url = str(ready["worker_url"])
        write_json(agent_root / "episode.json", episode)

        workspace = agent_root / "operator-workspace"
        codex_home = agent_root / "runtime" / "codex-home"
        workspace.mkdir()
        codex_home.mkdir(parents=True)
        import subprocess

        subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
        (codex_home / "auth.json").symlink_to(auth_path)
        final_path = agent_root / "agent_final.md"
        codex_command = build_live_codex_command(
            codex_bin=codex_bin,
            workspace=workspace,
            repo_root=REPO_ROOT,
            episode_root=agent_root,
            worker_url=worker_url,
            final_response_path=final_path,
            model=config["model"],
            reasoning_effort=config["reasoning_effort"],
        )
        write_json(agent_root / "codex_command.json", {"command": codex_command})
        codex_environment = dict(os.environ)
        codex_environment["CODEX_HOME"] = str(codex_home)
        episode["status"] = "codex_running"
        episode["codex_process_count"] = 1
        write_json(agent_root / "episode.json", episode)
        codex_lifecycle = _run_to_files(
            codex_command,
            cwd=workspace,
            environment=codex_environment,
            stdout_path=agent_root / "codex_exec.jsonl",
            stderr_path=agent_root / "codex_stderr.log",
            timeout_seconds=config["codex_timeout_seconds"],
        )
        write_json(agent_root / "codex_lifecycle.json", codex_lifecycle)
        if codex_lifecycle["returncode"] != 0 or codex_lifecycle["timed_out"]:
            raise RuntimeError("Codex exec failed during live operation")
        if not future.done():
            try:
                _post(worker_url, "/host_finalize")
            except urllib.error.URLError:
                if not future.done():
                    raise
        lifecycle = future.result(timeout=120).to_dict()
        executor.shutdown(wait=True)
        executor = None

        write_json(agent_root / "worker_lifecycle.json", lifecycle)
        child = _read_json(agent_root / "child_result.json")
        final_result = _read_json(agent_root / "final_result.json")
        operator_rows = read_jsonl(agent_root / "operator_context.jsonl")
        trace = _validate_operator_trace(operator_rows)
        codex_rows = read_jsonl(agent_root / "codex_exec.jsonl")
        codex_summary = summarize_codex_exec(codex_rows)
        write_json(agent_root / "codex_trace_summary.json", codex_summary)
        final_text = (agent_root / "agent_final.md").read_text(encoding="utf-8").strip()
        if not final_text:
            raise RuntimeError("Codex final response is empty")
        if lifecycle["returncode"] != 0 or lifecycle["timed_out"] or not lifecycle["cleanup_complete"]:
            raise RuntimeError("live simulator worker did not exit cleanly")
        if child["status"] != "completed":
            raise RuntimeError("live worker did not complete host evaluation")
        if trace["selected_skill_sequence"] != final_result["selected_skill_sequence"]:
            raise RuntimeError("Codex and worker selected skill sequences differ")
        if trace["action_tool_call_count"] != final_result["world_changing_skill_count"]:
            raise RuntimeError("Codex and worker action counts differ")
        episode.update(
            {
                "status": "completed",
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "classification": "first_univtac_closed_loop_operation_completed",
                "tool_sequence": trace["tool_sequence"],
                "tool_call_count": trace["tool_call_count"],
                "selected_skill_sequence": trace["selected_skill_sequence"],
                "agent_action_count": trace["action_tool_call_count"],
                "native_outcome": final_result,
                "agent_operation_smoke_success": final_result[
                    "agent_operation_smoke_success"
                ],
                "final_response_path": "agent_final.md",
            }
        )
        write_json(agent_root / "episode.json", episode)
        run_manifest = _read_json(output_root / "run_manifest.json")
        run_manifest.update(
            {
                "status": "completed",
                "agent_process_count": _count_codex_attempts(output_root),
                "agent_action_count": trace["action_tool_call_count"],
                "agent_seed": selected_seed,
                "agent_operation_smoke_success": final_result[
                    "agent_operation_smoke_success"
                ],
                "ended_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        write_json(output_root / "run_manifest.json", run_manifest)
        print(json.dumps(episode, indent=2, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - retain first live failure
        error = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        agent_root.mkdir(parents=True, exist_ok=True)
        write_json(agent_root / "exception.json", error)
        episode.update(
            {
                "status": "failed",
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "classification": "first_univtac_closed_loop_operation_failed",
                "error": error,
            }
        )
        write_json(agent_root / "episode.json", episode)
        manifest_path = output_root / "run_manifest.json"
        if manifest_path.is_file():
            run_manifest = _read_json(manifest_path)
            run_manifest.update(
                {
                    "status": "agent_attempt_failed",
                    "agent_process_count": _count_codex_attempts(output_root),
                    "last_agent_attempt": args.attempt,
                    "last_agent_attempt_action_count": int(episode["agent_action_count"]),
                }
            )
            write_json(manifest_path, run_manifest)
        return 1
    finally:
        if future is not None and not future.done() and worker_url is not None:
            try:
                _post(worker_url, "/host_finalize")
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                agent_root.mkdir(parents=True, exist_ok=True)
                write_json(
                    agent_root / "host_finalize_error.json",
                    {"error_type": type(exc).__name__, "error": str(exc)},
                )
        if executor is not None:
            executor.shutdown(wait=True)
        if codex_home is not None and codex_home.is_dir():
            (codex_home / "auth.json").unlink(missing_ok=True)
            shutil.rmtree(codex_home)


if __name__ == "__main__":
    raise SystemExit(main())
