#!/usr/bin/env python3
"""Run one fresh Pull Out Key snapshot through Codex and a read-only MCP."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.envs.univtac.codex_readonly import (
    build_codex_exec_command,
    load_operator_context,
    read_jsonl,
    summarize_codex_exec,
)
from sim.envs.univtac.trace import write_json

SUCCESS_CLASSIFICATION = "codex_native_univtac_readonly_observation_and_replay_passed"
SIMULATOR_SUCCESS = "scoped_launcher_and_pull_out_key_seed1000000_gate_passed"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_to_files(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    started = time.monotonic()
    started_at = _utc_now()
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        timed_out = False
        try:
            returncode = process.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                returncode = process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait(timeout=15)
    return {
        "command": command,
        "returncode": returncode,
        "timed_out": timed_out,
        "started_at": started_at,
        "ended_at": _utc_now(),
        "elapsed_seconds": time.monotonic() - started,
        "stdout_path": stdout_path.name,
        "stderr_path": stderr_path.name,
    }


def _snapshot_id(snapshot_path: Path) -> str:
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    return str(payload["snapshot_id"])


def _final_answer_covers_requested_views(text: str) -> bool:
    lowered = text.lower()
    return (
        all(token in lowered for token in ("wrist", "left", "right"))
        and any(token in lowered for token in ("head", "scene", "visual"))
        and any(token in lowered for token in ("uncertain", "ambigu", "unclear"))
    )


def finalize_existing_run(output_root: Path) -> dict[str, Any]:
    """Finalize an already-completed simulator and Codex trace without rerunning either."""

    episode_path = output_root / "episode.json"
    episode = json.loads(episode_path.read_text(encoding="utf-8"))
    simulator_lifecycle = json.loads(
        (output_root / "lifecycle/simulator.json").read_text(encoding="utf-8")
    )
    codex_lifecycle = json.loads((output_root / "lifecycle/codex.json").read_text(encoding="utf-8"))
    simulator_summary = json.loads(
        (output_root / "simulator/summary.json").read_text(encoding="utf-8")
    )
    if (
        simulator_lifecycle["returncode"] != 0
        or simulator_lifecycle["timed_out"]
        or simulator_summary.get("classification") != SIMULATOR_SUCCESS
    ):
        raise RuntimeError("fresh Pull Out Key simulator gate failed")
    if codex_lifecycle["returncode"] != 0 or codex_lifecycle["timed_out"]:
        raise RuntimeError("Codex exec failed")
    snapshot_path = output_root / "simulator/pull_out_key_seed1000000/snapshot_pre.json"
    if (
        not snapshot_path.is_file()
        or (snapshot_path.parent / "snapshot_post.json").exists()
        or (snapshot_path.parent / "transition.json").exists()
    ):
        raise RuntimeError("read-only snapshot artifacts are invalid")
    operator_rows = load_operator_context(output_root / "operator_context.jsonl")
    codex_rows = read_jsonl(output_root / "codex_exec.jsonl")
    trace_summary = summarize_codex_exec(codex_rows)
    write_json(output_root / "codex_trace_summary.json", trace_summary)
    final_path = output_root / "agent_final.md"
    final_text = final_path.read_text(encoding="utf-8").strip()
    if len(operator_rows) != 1:
        raise RuntimeError("Codex did not call observe exactly once")
    if not final_text or not _final_answer_covers_requested_views(final_text):
        raise RuntimeError("Codex final answer did not cover all requested observation areas")
    exception_path = output_root / "exception.json"
    if exception_path.is_file():
        previous = json.loads(exception_path.read_text(encoding="utf-8"))
        write_json(
            output_root / "resolved_harness_error.json",
            {
                "previous_error_type": previous.get("error_type"),
                "previous_error": previous.get("error"),
                "resolution": "accepted the model's uncertainty synonym 'unclear'",
                "simulator_rerun": False,
                "codex_rerun": False,
            },
        )
        exception_path.unlink()
    episode.pop("error", None)
    episode.update(
        {
            "status": "completed",
            "ended_at": codex_lifecycle["ended_at"],
            "codex_started_at": codex_lifecycle["started_at"],
            "codex_ended_at": codex_lifecycle["ended_at"],
            "snapshot_id": _snapshot_id(snapshot_path),
            "tool_call_count": 1,
            "operator_context_row_count": 1,
            "classification": SUCCESS_CLASSIFICATION,
            "duration_seconds": (
                simulator_lifecycle["elapsed_seconds"] + codex_lifecycle["elapsed_seconds"]
            ),
        }
    )
    write_json(episode_path, episode)
    return episode


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--gate-config",
        type=Path,
        default=REPO_ROOT / "configs/univtac/pull_out_key_seed1000000_gate.yaml",
    )
    parser.add_argument("--codex-bin", default=os.environ.get("OPENETA_CODEX_BIN", "codex"))
    parser.add_argument(
        "--model",
        default=os.environ.get("OPENETA_OPERATOR_MODEL", "gpt-5.6-terra"),
    )
    parser.add_argument(
        "--reasoning-effort",
        default=os.environ.get("OPENETA_OPERATOR_REASONING_EFFORT", "medium"),
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--finalize-existing", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root.expanduser().resolve()
    if args.finalize_existing:
        episode = finalize_existing_run(output_root)
        print(json.dumps(episode, indent=2, sort_keys=True))
        return 0
    if output_root.exists():
        raise FileExistsError(f"output root must be fresh: {output_root}")
    codex_bin = shutil.which(args.codex_bin)
    auth_path = Path.home() / ".codex/auth.json"
    if codex_bin is None:
        raise FileNotFoundError(f"Codex CLI is unavailable: {args.codex_bin}")
    if not auth_path.is_file():
        raise FileNotFoundError("existing Codex authentication is unavailable")

    output_root.mkdir(parents=True)
    started_at = _utc_now()
    episode = {
        "round": "R0.9.13",
        "task": "pull_out_key",
        "seed": 1_000_000,
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "status": "starting",
        "started_at": started_at,
        "ended_at": None,
        "snapshot_id": None,
        "tool_call_count": 0,
        "operator_context_row_count": 0,
        "mcp_tools": ["observe"],
        "simulator_invocation_count": 0,
        "codex_process_count": 0,
        "snapshot_post_count": 0,
        "transition_count": 0,
        "action_tool_call_count": 0,
        "simulator_step_after_snapshot_count": 0,
        "final_response_path": "agent_final.md",
        "classification": None,
    }
    write_json(output_root / "episode.json", episode)
    simulator_root = output_root / "simulator"
    simulator_command = [
        sys.executable,
        str(REPO_ROOT / "scripts/univtac/run_pull_out_key_gate.py"),
        "--config",
        str(args.gate_config.expanduser().resolve()),
        "--runtime-python",
        str(args.runtime_python.expanduser().resolve()),
        "--source-root",
        str(args.source_root.expanduser().resolve()),
        "--output-root",
        str(simulator_root),
    ]
    if args.headless:
        simulator_command.append("--headless")
    lifecycle_dir = output_root / "lifecycle"
    error: dict[str, Any] | None = None
    codex_home: Path | None = None
    try:
        episode["status"] = "simulator_running"
        episode["simulator_invocation_count"] = 1
        write_json(output_root / "episode.json", episode)
        simulator_lifecycle = _run_to_files(
            simulator_command,
            cwd=REPO_ROOT,
            environment=dict(os.environ),
            stdout_path=lifecycle_dir / "simulator_stdout.log",
            stderr_path=lifecycle_dir / "simulator_stderr.log",
            timeout_seconds=1260,
        )
        write_json(lifecycle_dir / "simulator.json", simulator_lifecycle)
        simulator_summary = json.loads(
            (simulator_root / "summary.json").read_text(encoding="utf-8")
        )
        if (
            simulator_lifecycle["returncode"] != 0
            or simulator_lifecycle["timed_out"]
            or simulator_summary.get("classification") != SIMULATOR_SUCCESS
        ):
            raise RuntimeError("fresh Pull Out Key simulator gate failed")

        snapshot_path = simulator_root / "pull_out_key_seed1000000/snapshot_pre.json"
        if not snapshot_path.is_file():
            raise FileNotFoundError("fresh snapshot_pre.json is missing")
        if (snapshot_path.parent / "snapshot_post.json").exists() or (
            snapshot_path.parent / "transition.json"
        ).exists():
            raise RuntimeError("read-only simulator produced post-action artifacts")
        episode["snapshot_id"] = _snapshot_id(snapshot_path)
        write_json(output_root / "episode.json", episode)

        workspace = output_root / "operator-workspace"
        codex_home = output_root / "runtime/codex-home"
        workspace.mkdir()
        codex_home.mkdir(parents=True)
        subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
        (codex_home / "auth.json").symlink_to(auth_path)
        final_path = output_root / "agent_final.md"
        codex_command = build_codex_exec_command(
            codex_bin=codex_bin,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            workspace=workspace,
            repo_root=REPO_ROOT,
            episode_root=output_root,
            snapshot_path=snapshot_path,
            final_response_path=final_path,
        )
        codex_environment = dict(os.environ)
        codex_environment["CODEX_HOME"] = str(codex_home)
        episode["status"] = "codex_running"
        episode["codex_started_at"] = _utc_now()
        episode["codex_process_count"] = 1
        write_json(output_root / "episode.json", episode)
        codex_lifecycle = _run_to_files(
            codex_command,
            cwd=workspace,
            environment=codex_environment,
            stdout_path=output_root / "codex_exec.jsonl",
            stderr_path=lifecycle_dir / "codex_stderr.log",
            timeout_seconds=900,
        )
        write_json(lifecycle_dir / "codex.json", codex_lifecycle)
        episode["codex_ended_at"] = _utc_now()
        if codex_lifecycle["returncode"] != 0 or codex_lifecycle["timed_out"]:
            raise RuntimeError("Codex exec failed")

        episode = finalize_existing_run(output_root)
        print(json.dumps(episode, indent=2, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001 - persist the real integration failure
        error = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(output_root / "exception.json", error)
        episode.update(
            {
                "status": "failed",
                "ended_at": _utc_now(),
                "classification": "codex_native_univtac_readonly_observation_failed",
                "error": error,
            }
        )
        write_json(output_root / "episode.json", episode)
        return 1
    finally:
        if codex_home is not None and codex_home.is_dir():
            (codex_home / "auth.json").unlink(missing_ok=True)
            shutil.rmtree(codex_home)


if __name__ == "__main__":
    raise SystemExit(main())
