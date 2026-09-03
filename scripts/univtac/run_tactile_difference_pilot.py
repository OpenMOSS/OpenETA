#!/usr/bin/env python3
"""Run the R0.9.15 Pull Out Key temporal tactile-difference pilot."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.univtac.run_codex_readonly_observation import _run_to_files, _utc_now
from scripts.univtac.run_tactile_causal_pilot import _archive_failed_attempt
from sim.envs.univtac.codex_readonly import (
    build_codex_exec_command,
    load_operator_context,
    read_jsonl,
    summarize_codex_exec,
)
from sim.envs.univtac.pull_out_key_gate import seed_dir_name, success_classification
from sim.envs.univtac.tactile_difference_pilot import (
    CONDITION_ORDER,
    MODEL,
    PILOT_PROMPT,
    REASONING_EFFORT,
    SEEDS,
    add_press_depth_reference,
    build_pair_artifacts,
    parse_prediction,
    stage_condition,
    summarize_predictions,
    validate_config,
)
from sim.envs.univtac.trace import write_json


class ConditionRunError(RuntimeError):
    def __init__(self, message: str, *, attempts: int, infrastructure_failures: int) -> None:
        super().__init__(message)
        self.attempts = attempts
        self.infrastructure_failures = infrastructure_failures


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--codex-bin", default=os.environ.get("OPENETA_CODEX_BIN", "codex"))
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _run_seed_capture(
    *,
    seed: int,
    base_gate: dict[str, Any],
    runtime_python: Path,
    source_root: Path,
    output_root: Path,
    headless: bool,
) -> dict[str, Any]:
    simulator_root = output_root / "simulator" / f"seed_{seed}"
    gate_config = dict(base_gate)
    gate_config.update({"seed": seed, "capture_tactile_pair": True})
    config_path = output_root / "simulator" / f"seed_{seed}_gate.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(gate_config, sort_keys=False), encoding="utf-8")
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts/univtac/run_pull_out_key_gate.py"),
        "--config",
        str(config_path),
        "--runtime-python",
        str(runtime_python),
        "--source-root",
        str(source_root),
        "--output-root",
        str(simulator_root),
    ]
    if headless:
        command.append("--headless")
    lifecycle = _run_to_files(
        command,
        cwd=REPO_ROOT,
        environment=dict(os.environ),
        stdout_path=output_root / "lifecycle" / f"simulator_seed_{seed}_stdout.log",
        stderr_path=output_root / "lifecycle" / f"simulator_seed_{seed}_stderr.log",
        timeout_seconds=float(gate_config["timeout_seconds"]) + 60,
    )
    write_json(output_root / "lifecycle" / f"simulator_seed_{seed}.json", lifecycle)
    seed_root = simulator_root / seed_dir_name(seed)
    summary = (
        _load_json(simulator_root / "summary.json")
        if (simulator_root / "summary.json").is_file()
        else {}
    )
    baseline = seed_root / "snapshot_pre_grasp_baseline.json"
    current = seed_root / "snapshot_post_pre_move.json"
    capture = seed_root / "pair_capture_summary.json"
    valid = (
        lifecycle["returncode"] == 0
        and not lifecycle["timed_out"]
        and summary.get("classification") == success_classification(seed)
        and baseline.is_file()
        and current.is_file()
        and capture.is_file()
    )
    return {
        "seed": seed,
        "valid": valid,
        "simulator_root": str(simulator_root),
        "baseline_snapshot": str(baseline),
        "current_snapshot": str(current),
        "capture_summary": str(capture),
        "classification": summary.get("classification", "runtime_error"),
        "error": None if valid else summary.get("launcher_error") or "pair capture gate failed",
    }


def _semantic_evidence_exists(episode_root: Path) -> bool:
    context = episode_root / "operator_context.jsonl"
    final = episode_root / "agent_final.md"
    return bool(
        (context.is_file() and context.read_text(encoding="utf-8").strip())
        or (final.is_file() and final.read_text(encoding="utf-8").strip())
    )


def _execute_codex_attempt(
    *,
    seed: int,
    condition: str,
    episode_root: Path,
    snapshot: Path,
    codex_bin: str,
    timeout_seconds: float,
    auth_path: Path,
    attempt_index: int,
) -> dict[str, Any]:
    episode = {
        "round": "R0.9.15",
        "task": "pull_out_key",
        "seed": seed,
        "condition": condition,
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "status": "codex_running",
        "started_at": _utc_now(),
        "ended_at": None,
        "tool_call_count": 0,
        "operator_context_row_count": 0,
        "mcp_tools": ["observe"],
        "simulator_invocation_count": 0,
        "codex_process_count": 1,
        "codex_attempt_index": attempt_index,
        "snapshot_post_count": 0,
        "transition_count": 0,
        "action_tool_call_count": 0,
        "simulator_step_after_snapshot_count": 0,
        "final_response_path": "agent_final.md",
    }
    write_json(episode_root / "episode.json", episode)
    workspace = episode_root / "operator-workspace"
    codex_home = episode_root / "runtime/codex-home"
    workspace.mkdir()
    codex_home.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    (codex_home / "auth.json").symlink_to(auth_path)
    command = build_codex_exec_command(
        codex_bin=codex_bin,
        model=MODEL,
        reasoning_effort=REASONING_EFFORT,
        workspace=workspace,
        repo_root=REPO_ROOT,
        episode_root=episode_root,
        snapshot_path=snapshot,
        final_response_path=episode_root / "agent_final.md",
        condition_manifest=episode_root / "condition.json",
        prompt=PILOT_PROMPT,
    )
    environment = dict(os.environ)
    environment["CODEX_HOME"] = str(codex_home)
    try:
        lifecycle = _run_to_files(
            command,
            cwd=workspace,
            environment=environment,
            stdout_path=episode_root / "codex_exec.jsonl",
            stderr_path=episode_root / "codex_stderr.log",
            timeout_seconds=timeout_seconds,
        )
        write_json(episode_root / "codex_lifecycle.json", lifecycle)
        context_rows = load_operator_context(
            episode_root / "operator_context.jsonl", expected_image_count=6
        )
        final_text = (episode_root / "agent_final.md").read_text(encoding="utf-8").strip()
        prediction = parse_prediction(final_text)
        trace_summary = summarize_codex_exec(read_jsonl(episode_root / "codex_exec.jsonl"))
        if lifecycle["returncode"] != 0 or lifecycle["timed_out"] or len(context_rows) != 1:
            raise RuntimeError("Codex condition did not complete with exactly one observe")
        episode.update(
            {
                "status": "completed",
                "ended_at": lifecycle["ended_at"],
                "duration_seconds": lifecycle["elapsed_seconds"],
                "tool_call_count": 1,
                "operator_context_row_count": 1,
                "parse_status": "parsed",
                "usage": trace_summary["usage"],
            }
        )
        write_json(episode_root / "episode.json", episode)
        return {
            "seed": seed,
            "condition": condition,
            "model": MODEL,
            **prediction,
            "duration_seconds": lifecycle["elapsed_seconds"],
            "usage": trace_summary["usage"],
            "tool_call_count": 1,
            "parse_status": "parsed",
            "raw_response_path": str(
                (episode_root / "agent_final.md").relative_to(episode_root.parents[2])
            ),
        }
    except Exception as exc:
        write_json(
            episode_root / "exception.json",
            {
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        episode.update({"status": "failed", "ended_at": _utc_now(), "parse_status": "failed"})
        write_json(episode_root / "episode.json", episode)
        raise
    finally:
        (codex_home / "auth.json").unlink(missing_ok=True)
        shutil.rmtree(codex_home, ignore_errors=True)


def _run_condition(
    *,
    seed: int,
    condition: str,
    pair: dict[str, Any],
    source: dict[str, Any],
    output_root: Path,
    codex_bin: str,
    timeout_seconds: float,
    auth_path: Path,
    retry_limit: int,
) -> tuple[dict[str, Any], int, int]:
    episode_root = output_root / "runs" / f"seed_{seed}" / condition
    episode_root.mkdir(parents=True, exist_ok=False)
    stage_condition(
        condition=condition,
        pair=pair,
        episode_root=episode_root,
        output_root=output_root,
        simulator_root=Path(source["simulator_root"]),
    )
    attempts = 0
    infrastructure_failures = 0
    while True:
        attempts += 1
        try:
            prediction = _execute_codex_attempt(
                seed=seed,
                condition=condition,
                episode_root=episode_root,
                snapshot=Path(source["current_snapshot"]),
                codex_bin=codex_bin,
                timeout_seconds=timeout_seconds,
                auth_path=auth_path,
                attempt_index=attempts,
            )
            return prediction, attempts, infrastructure_failures
        except Exception as exc:
            if _semantic_evidence_exists(episode_root) or infrastructure_failures >= retry_limit:
                raise ConditionRunError(
                    str(exc),
                    attempts=attempts,
                    infrastructure_failures=infrastructure_failures,
                ) from exc
            infrastructure_failures += 1
            _archive_failed_attempt(episode_root)


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_results(path: Path, rows: list[dict[str, Any]], references: dict[str, Any]) -> None:
    fields = [
        "seed",
        "condition",
        "left_change",
        "right_change",
        "left_change_region",
        "right_change_region",
        "stronger_change_side",
        "left_reference_region",
        "right_reference_region",
        "reference_stronger_side",
        "duration_seconds",
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
        "parse_status",
        "tool_call_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            reference = references[str(row["seed"])]
            usage = row.get("usage") or {}
            writer.writerow(
                {
                    **{field: row.get(field) for field in fields},
                    "left_reference_region": reference["sensors"]["left_tactile"]["metrics"][
                        "centroid_region"
                    ],
                    "right_reference_region": reference["sensors"]["right_tactile"]["metrics"][
                        "centroid_region"
                    ],
                    "reference_stronger_side": reference["reference_stronger_side"],
                    "input_tokens": usage.get("input_tokens"),
                    "cached_input_tokens": usage.get("cached_input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                    "reasoning_output_tokens": usage.get("reasoning_output_tokens"),
                }
            )


def _write_raw_answers(output_root: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# R0.9.15 raw Codex answers", ""]
    for row in rows:
        path = output_root / row["raw_response_path"]
        lines.extend(
            [
                f"## seed {row['seed']} — {row['condition']}",
                "",
                "```json",
                path.read_text(encoding="utf-8").strip(),
                "```",
                "",
            ]
        )
    (output_root / "raw_answers.md").write_text("\n".join(lines), encoding="utf-8")


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
        raise TypeError("difference pilot config must be a mapping")
    config = validate_config(payload)
    codex_bin = shutil.which(args.codex_bin)
    auth_path = Path.home() / ".codex/auth.json"
    if codex_bin is None or not auth_path.is_file():
        raise RuntimeError("Codex CLI or existing authentication is unavailable")
    output_root.mkdir(parents=True)
    (output_root / "prompt.txt").write_text(PILOT_PROMPT, encoding="utf-8")
    pilot = {
        "round": "R0.9.15",
        "task": "pull_out_key",
        "seeds": list(SEEDS),
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "status": "running",
        "started_at": _utc_now(),
        "simulator_invocation_count": 0,
        "codex_process_attempts": 0,
        "pre_observe_infrastructure_failures": 0,
        "semantic_retries_after_observe": 0,
        "action_tool_call_count": 0,
    }
    write_json(output_root / "pilot.json", pilot)
    base_gate_path = (REPO_ROOT / str(config["gate_config"])).resolve(strict=True)
    base_gate = yaml.safe_load(base_gate_path.read_text(encoding="utf-8"))
    sources = []
    pairs: dict[str, Any] = {}
    for seed in SEEDS:
        pilot["simulator_invocation_count"] += 1
        write_json(output_root / "pilot.json", pilot)
        source = _run_seed_capture(
            seed=seed,
            base_gate=base_gate,
            runtime_python=runtime_python,
            source_root=source_root,
            output_root=output_root,
            headless=bool(config["headless"]),
        )
        sources.append(source)
        if source["valid"]:
            baseline = _load_json(Path(source["baseline_snapshot"]))
            current = _load_json(Path(source["current_snapshot"]))
            pairs[str(seed)] = build_pair_artifacts(
                seed=seed,
                baseline_snapshot=baseline,
                current_snapshot=current,
                simulator_root=Path(source["simulator_root"]),
                output_root=output_root,
            )
    write_json(
        output_root / "capture_summary.json",
        {
            "sources": [
                {
                    **source,
                    "pair_capture": (
                        _load_json(Path(source["capture_summary"])) if source["valid"] else None
                    ),
                }
                for source in sources
            ]
        },
    )
    predictions: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for source in sources:
        if not source["valid"]:
            continue
        seed = int(source["seed"])
        for condition in CONDITION_ORDER[seed]:
            try:
                prediction, attempts, failures = _run_condition(
                    seed=seed,
                    condition=condition,
                    pair=pairs[str(seed)],
                    source=source,
                    output_root=output_root,
                    codex_bin=codex_bin,
                    timeout_seconds=float(config["codex_timeout_seconds"]),
                    auth_path=auth_path,
                    retry_limit=int(config["pre_observe_retry_limit"]),
                )
                predictions.append(prediction)
                pilot["codex_process_attempts"] += attempts
                pilot["pre_observe_infrastructure_failures"] += failures
            except Exception as exc:  # noqa: BLE001 - persist each real condition failure
                pilot["codex_process_attempts"] += int(getattr(exc, "attempts", 1))
                pilot["pre_observe_infrastructure_failures"] += int(
                    getattr(exc, "infrastructure_failures", 0)
                )
                errors.append(
                    {
                        "seed": seed,
                        "condition": condition,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            _write_predictions(output_root / "predictions.jsonl", predictions)
            write_json(output_root / "pilot.json", pilot)
    references: dict[str, Any] = {}
    for source in sources:
        if source["valid"]:
            seed = str(source["seed"])
            references[seed] = add_press_depth_reference(
                pair=pairs[seed],
                baseline_snapshot=_load_json(Path(source["baseline_snapshot"])),
                current_snapshot=_load_json(Path(source["current_snapshot"])),
                simulator_root=Path(source["simulator_root"]),
                output_root=output_root,
            )
    write_json(output_root / "difference_metrics.json", {"seeds": references})
    predictions.sort(key=lambda row: (int(row["seed"]), str(row["condition"])))
    _write_predictions(output_root / "predictions.jsonl", predictions)
    _write_results(output_root / "results.csv", predictions, references)
    _write_raw_answers(output_root, predictions)
    summary = summarize_predictions(predictions, references)
    complete = len(predictions) == 9 and not errors and len(references) == 3
    summary.update(
        {
            "classification": (
                "tactile_temporal_difference_pilot_completed"
                if complete
                else "tactile_temporal_difference_pilot_incomplete"
            ),
            "valid_seed_count": len(references),
            "completed_semantic_trials": len(predictions),
            "codex_process_attempts": pilot["codex_process_attempts"],
            "pre_observe_infrastructure_failures": pilot["pre_observe_infrastructure_failures"],
            "semantic_retries_after_observe": 0,
            "simulator_invocation_count": pilot["simulator_invocation_count"],
            "action_tool_call_count": 0,
            "errors": errors,
        }
    )
    write_json(output_root / "summary.json", summary)
    dashboard = output_root / "dashboard"
    dashboard.mkdir()
    (dashboard / "index.html").write_text(
        '<!doctype html><meta charset="utf-8">'
        f'<a href="http://127.0.0.1:9399/pilot/{output_root.name}">'
        "Open the live R0.9.15 difference pilot dashboard</a>\n",
        encoding="utf-8",
    )
    pilot.update(
        {
            "status": "completed" if complete else "incomplete",
            "ended_at": _utc_now(),
            "classification": summary["classification"],
            "difference_signal": summary["difference_signal"],
        }
    )
    write_json(output_root / "pilot.json", pilot)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
