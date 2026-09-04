#!/usr/bin/env python3
"""Run the zero-simulator R0.9.17 unilateral tactile guidance-conflict pilot."""

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
from scripts.univtac.run_structured_tactile_pilot import _load_sources
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
from sim.envs.univtac.trace import write_json
from sim.envs.univtac.unilateral_tactile_pilot import (
    CONDITION_ORDER,
    MODEL,
    PILOT_PROMPT,
    REASONING_EFFORT,
    SEEDS,
    build_source_mapping,
    parse_prediction,
    stage_condition,
    summarize_predictions,
    validate_config,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--structured-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--codex-bin", default=os.environ.get("OPENETA_CODEX_BIN", "codex"))
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


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
        "round": "R0.9.17",
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
        "counterfactual_multimodal_input": True,
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
            episode_root / "operator_context.jsonl", expected_image_count=4
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
    source_mapping: dict[str, Any],
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
        source_mapping=source_mapping,
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


def _write_results(
    path: Path, rows: list[dict[str, Any]], source_mappings: dict[str, Any]
) -> None:
    fields = [
        "seed",
        "condition",
        "left_image_source",
        "right_image_source",
        "expected_image_change_side",
        "tactile_image_access",
        "left_tactile_state",
        "right_tactile_state",
        "tactile_changed_side",
        "overall_contact_state",
        "visual_tactile_consistency",
        "parse_status",
        "duration_seconds",
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "reasoning_output_tokens",
        "tool_call_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            mapping = source_mappings[str(row["seed"])]
            usage = row.get("usage") or {}
            writer.writerow(
                {
                    **{field: row.get(field) for field in fields},
                    "left_image_source": mapping["image_source_mapping"]["left_tactile"],
                    "right_image_source": mapping["image_source_mapping"]["right_tactile"],
                    "expected_image_change_side": mapping["expected_image_change_side"],
                    "input_tokens": usage.get("input_tokens"),
                    "cached_input_tokens": usage.get("cached_input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                    "reasoning_output_tokens": usage.get("reasoning_output_tokens"),
                }
            )


def _write_raw_answers(output_root: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# R0.9.17 raw Codex answers", ""]
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
    structured_root = args.structured_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"output root must be fresh: {output_root}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("unilateral pilot config must be a mapping")
    config = validate_config(payload)
    codex_bin = shutil.which(args.codex_bin)
    auth_path = Path.home() / ".codex/auth.json"
    if codex_bin is None or not auth_path.is_file():
        raise RuntimeError("Codex CLI or existing authentication is unavailable")
    source_pilot = _load_json(source_root / "pilot.json")
    structured_pilot = _load_json(structured_root / "pilot.json")
    if source_pilot.get("round") != "R0.9.15" or source_pilot.get("status") != "completed":
        raise ValueError("source root is not a completed R0.9.15 pilot")
    if (
        structured_pilot.get("round") != "R0.9.16"
        or structured_pilot.get("status") != "completed"
    ):
        raise ValueError("structured root is not a completed R0.9.16 pilot")
    sources = _load_sources(source_root)
    legacy_references = _load_json(source_root / "difference_metrics.json").get("seeds")
    structured_references = _load_json(structured_root / "structured_reference.json").get(
        "seeds"
    )
    if not isinstance(legacy_references, dict) or not isinstance(structured_references, dict):
        raise TypeError("source references must be mappings")
    source_mappings = {
        str(seed): build_source_mapping(
            seed=seed,
            source_root=source_root,
            legacy_reference=legacy_references[str(seed)],
            structured_reference=structured_references[str(seed)],
        )
        for seed in SEEDS
    }
    output_root.mkdir(parents=True)
    resolved = {
        **config,
        "source_root": str(source_root),
        "structured_root": str(structured_root),
        "output_root": str(output_root),
        "simulator_invocation_count": 0,
    }
    write_json(output_root / "config_resolved.json", resolved)
    write_json(
        output_root / "source_mapping.json",
        {
            "counterfactual_multimodal_input": True,
            "claim_boundary": "model_input_level_tactile_causal_intervention_only",
            "seeds": source_mappings,
        },
    )
    (output_root / "prompt.txt").write_text(PILOT_PROMPT, encoding="utf-8")
    pilot = {
        "round": "R0.9.17",
        "task": "pull_out_key",
        "seeds": list(SEEDS),
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "status": "running",
        "started_at": _utc_now(),
        "source_root": str(source_root),
        "structured_root": str(structured_root),
        "counterfactual_multimodal_input": True,
        "simulator_invocation_count": 0,
        "new_reset_count": 0,
        "new_snapshot_count": 0,
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
                    source_mapping=source_mappings[str(seed)],
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
    _write_results(output_root / "results.csv", predictions, source_mappings)
    _write_raw_answers(output_root, predictions)
    complete = len(predictions) == 9 and not errors
    summary = (
        summarize_predictions(predictions, source_mappings)
        if complete
        else {
            "classification": "unilateral_tactile_guidance_conflict_pilot_incomplete",
            "unilateral_signal": None,
        }
    )
    summary.update(
        {
            "classification": (
                "unilateral_tactile_guidance_conflict_pilot_completed"
                if complete
                else "unilateral_tactile_guidance_conflict_pilot_incomplete"
            ),
            "completed_semantic_trials": len(predictions),
            "codex_process_attempts": pilot["codex_process_attempts"],
            "pre_observe_infrastructure_failures": pilot[
                "pre_observe_infrastructure_failures"
            ],
            "semantic_retries_after_observe": 0,
            "simulator_invocation_count": 0,
            "new_reset_count": 0,
            "new_snapshot_count": 0,
            "action_tool_call_count": 0,
            "counterfactual_multimodal_input": True,
            "errors": errors,
        }
    )
    write_json(output_root / "summary.json", summary)
    dashboard = output_root / "dashboard"
    dashboard.mkdir()
    (dashboard / "index.html").write_text(
        '<!doctype html><meta charset="utf-8">'
        f'<a href="http://127.0.0.1:9399/pilot/{output_root.name}">'
        "Open the live R0.9.17 unilateral tactile dashboard</a>\n",
        encoding="utf-8",
    )
    pilot.update(
        {
            "status": "completed" if complete else "incomplete",
            "ended_at": _utc_now(),
            "classification": summary["classification"],
            "unilateral_signal": summary["unilateral_signal"],
        }
    )
    write_json(output_root / "pilot.json", pilot)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
