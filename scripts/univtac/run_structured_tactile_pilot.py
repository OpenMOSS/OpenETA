#!/usr/bin/env python3
"""Run the zero-simulator R0.9.16 structured tactile-guidance pilot."""

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
from scripts.univtac.run_tactile_difference_pilot import (
    ConditionRunError,
    _semantic_evidence_exists,
)
from sim.envs.univtac.codex_readonly import (
    build_codex_exec_command,
    load_operator_context,
    read_jsonl,
    summarize_codex_exec,
)
from sim.envs.univtac.structured_tactile_pilot import (
    CONDITION_ORDER,
    MODEL,
    PILOT_PROMPT,
    REASONING_EFFORT,
    SEEDS,
    build_structured_reference,
    parse_prediction,
    stage_condition,
    summarize_predictions,
    validate_config,
)
from sim.envs.univtac.trace import write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--codex-bin", default=os.environ.get("OPENETA_CODEX_BIN", "codex"))
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _load_sources(source_root: Path) -> dict[int, dict[str, Any]]:
    capture = _load_json(source_root / "capture_summary.json")
    records = capture.get("sources")
    if not isinstance(records, list):
        raise TypeError("R0.9.15 capture summary source records must be a list")
    sources = {int(record["seed"]): dict(record) for record in records}
    if tuple(sorted(sources)) != SEEDS:
        raise ValueError("R0.9.15 source must contain exactly the three fixed seeds")
    for seed, record in sources.items():
        if not record.get("valid"):
            raise ValueError(f"R0.9.15 seed {seed} is not a valid pair capture")
        for field in ("baseline_snapshot", "current_snapshot", "simulator_root"):
            if not Path(str(record[field])).exists():
                raise FileNotFoundError(f"missing R0.9.15 {field} for seed {seed}")
    return sources


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
        "round": "R0.9.16",
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
        contexts = load_operator_context(
            episode_root / "operator_context.jsonl", expected_image_count=6
        )
        final_text = (episode_root / "agent_final.md").read_text(encoding="utf-8").strip()
        prediction = parse_prediction(final_text)
        trace_summary = summarize_codex_exec(read_jsonl(episode_root / "codex_exec.jsonl"))
        if lifecycle["returncode"] != 0 or lifecycle["timed_out"] or len(contexts) != 1:
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
    source: dict[str, Any],
    legacy_reference: dict[str, Any],
    structured_reference: dict[str, Any],
    source_root: Path,
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
        source_root=source_root,
        legacy_reference=legacy_reference,
        structured_reference=structured_reference,
        episode_root=episode_root,
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
                    str(exc), attempts=attempts, infrastructure_failures=infrastructure_failures
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
                    "left_reference_region": reference["sensors"]["left_tactile"][
                        "structured_metrics"
                    ]["reference_region"],
                    "right_reference_region": reference["sensors"]["right_tactile"][
                        "structured_metrics"
                    ]["reference_region"],
                    "reference_stronger_side": reference["reference_stronger_side"],
                    "input_tokens": usage.get("input_tokens"),
                    "cached_input_tokens": usage.get("cached_input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                    "reasoning_output_tokens": usage.get("reasoning_output_tokens"),
                }
            )


def _write_raw_answers(output_root: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# R0.9.16 raw Codex answers", ""]
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
    source_root = args.source_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"output root must be fresh: {output_root}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("structured pilot config must be a mapping")
    config = validate_config(payload)
    codex_bin = shutil.which(args.codex_bin)
    auth_path = Path.home() / ".codex/auth.json"
    if codex_bin is None or not auth_path.is_file():
        raise RuntimeError("Codex CLI or existing authentication is unavailable")
    source_pilot = _load_json(source_root / "pilot.json")
    source_summary = _load_json(source_root / "summary.json")
    if (
        source_pilot.get("round") != "R0.9.15"
        or source_summary.get("classification")
        != "tactile_temporal_difference_pilot_completed"
    ):
        raise ValueError("source root is not a completed R0.9.15 pilot")
    sources = _load_sources(source_root)
    legacy_references = _load_json(source_root / "difference_metrics.json").get("seeds")
    if not isinstance(legacy_references, dict):
        raise TypeError("R0.9.15 difference metrics seeds must be a mapping")
    output_root.mkdir(parents=True)
    (output_root / "prompt.txt").write_text(PILOT_PROMPT, encoding="utf-8")
    references = {
        str(seed): build_structured_reference(
            seed=seed,
            source_root=source_root,
            legacy_reference=legacy_references[str(seed)],
        )
        for seed in SEEDS
    }
    write_json(
        output_root / "structured_reference.json",
        {
            "source_round": "R0.9.15",
            "active_threshold": 12,
            "saliency_percentile": 90,
            "seeds": references,
        },
    )
    write_json(
        output_root / "secondary_diagnostics.json",
        {
            "r0915_summary": source_summary,
            "r0915_predictions": read_jsonl(source_root / "predictions.jsonl"),
            "note": "Old global centroid and press-depth delta remain secondary diagnostics.",
        },
    )
    pilot = {
        "round": "R0.9.16",
        "task": "pull_out_key",
        "seeds": list(SEEDS),
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "status": "running",
        "started_at": _utc_now(),
        "source_root": str(source_root),
        "simulator_invocation_count": 0,
        "codex_process_attempts": 0,
        "pre_observe_infrastructure_failures": 0,
        "semantic_retries_after_observe": 0,
        "action_tool_call_count": 0,
    }
    write_json(output_root / "pilot.json", pilot)
    predictions: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for seed in SEEDS:
        for condition in CONDITION_ORDER[seed]:
            try:
                prediction, attempts, failures = _run_condition(
                    seed=seed,
                    condition=condition,
                    source=sources[seed],
                    legacy_reference=legacy_references[str(seed)],
                    structured_reference=references[str(seed)],
                    source_root=source_root,
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
    predictions.sort(key=lambda row: (int(row["seed"]), str(row["condition"])))
    _write_predictions(output_root / "predictions.jsonl", predictions)
    _write_results(output_root / "results.csv", predictions, references)
    _write_raw_answers(output_root, predictions)
    summary = summarize_predictions(predictions, references)
    complete = len(predictions) == 9 and not errors
    summary.update(
        {
            "classification": (
                "tactile_structured_guidance_pilot_completed"
                if complete
                else "tactile_structured_guidance_pilot_incomplete"
            ),
            "completed_semantic_trials": len(predictions),
            "codex_process_attempts": pilot["codex_process_attempts"],
            "pre_observe_infrastructure_failures": pilot[
                "pre_observe_infrastructure_failures"
            ],
            "semantic_retries_after_observe": 0,
            "simulator_invocation_count": 0,
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
        "Open the live R0.9.16 structured tactile dashboard</a>\n",
        encoding="utf-8",
    )
    pilot.update(
        {
            "status": "completed" if complete else "incomplete",
            "ended_at": _utc_now(),
            "classification": summary["classification"],
            "structured_signal": summary["structured_signal"],
        }
    )
    write_json(output_root / "pilot.json", pilot)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
