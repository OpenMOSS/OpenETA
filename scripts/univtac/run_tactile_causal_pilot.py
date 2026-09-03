#!/usr/bin/env python3
"""Run the three-seed Codex-native Pull Out Key tactile causal pilot."""

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

from scripts.univtac.run_codex_readonly_observation import _run_to_files, _snapshot_id, _utc_now
from sim.envs.univtac.agent_context import (
    load_validated_pre_action_snapshot,
    project_operator_visible_context,
)
from sim.envs.univtac.codex_readonly import (
    build_codex_exec_command,
    load_operator_context,
    read_jsonl,
    summarize_codex_exec,
)
from sim.envs.univtac.pull_out_key_gate import seed_dir_name, success_classification
from sim.envs.univtac.tactile_causal_pilot import (
    CONDITION_ORDER,
    MODEL,
    PILOT_PROMPT,
    REASONING_EFFORT,
    SEEDS,
    parse_prediction,
    press_depth_reference,
    stage_condition,
    summarize_pilot,
    validate_pilot_config,
)
from sim.envs.univtac.trace import write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--reuse-seed1000000-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--codex-bin", default=os.environ.get("OPENETA_CODEX_BIN", "codex"))
    parser.add_argument("--resume-failed", action="store_true")
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _source_record(simulator_root: Path, seed: int) -> dict[str, Any]:
    seed_root = simulator_root / seed_dir_name(seed)
    snapshot = seed_root / "snapshot_pre.json"
    summary = _load_json(simulator_root / "summary.json")
    valid = (
        summary.get("classification") == success_classification(seed)
        and snapshot.is_file()
        and not (seed_root / "snapshot_post.json").exists()
        and not (seed_root / "transition.json").exists()
    )
    return {
        "seed": seed,
        "valid": valid,
        "simulator_root": str(simulator_root),
        "snapshot_pre": str(snapshot),
        "classification": summary.get("classification"),
        "error": None if valid else summary.get("launcher_error") or "reset-only gate failed",
    }


def _run_seed_gate(
    *,
    seed: int,
    base_config: dict[str, Any],
    runtime_python: Path,
    source_root: Path,
    output_root: Path,
    headless: bool,
) -> dict[str, Any]:
    gate_root = output_root / "simulator" / f"seed_{seed}"
    gate_config = dict(base_config)
    gate_config["seed"] = seed
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
        str(gate_root),
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
    if not (gate_root / "summary.json").is_file():
        return {
            "seed": seed,
            "valid": False,
            "simulator_root": str(gate_root),
            "snapshot_pre": None,
            "classification": "runtime_error",
            "error": f"gate exited {lifecycle['returncode']} without summary",
        }
    record = _source_record(gate_root, seed)
    if lifecycle["returncode"] != 0 or lifecycle["timed_out"]:
        record["valid"] = False
        record["error"] = (
            f"gate returncode={lifecycle['returncode']} timed_out={lifecycle['timed_out']}"
        )
    return record


def _run_codex_condition(
    *,
    seed: int,
    condition: str,
    source: dict[str, Any],
    output_root: Path,
    codex_bin: str,
    timeout_seconds: float,
    auth_path: Path,
    reuse_staged: bool = False,
) -> dict[str, Any]:
    episode_root = output_root / "runs" / f"seed_{seed}" / condition
    snapshot = Path(source["snapshot_pre"]).resolve(strict=True)
    simulator_root = Path(source["simulator_root"]).resolve(strict=True)
    if reuse_staged:
        if (
            not (episode_root / "condition.json").is_file()
            or not (episode_root / "images").is_dir()
        ):
            raise FileNotFoundError("failed condition lacks its staged manifest or images")
        _archive_failed_attempt(episode_root)
    else:
        episode_root.mkdir(parents=True, exist_ok=False)
        visible = load_validated_pre_action_snapshot(snapshot, simulator_root)
        context = project_operator_visible_context(visible, simulator_root)
        image_map = {image.label: image.path for image in context.images}
        stage_condition(
            condition=condition,
            context_images=image_map,
            episode_root=episode_root,
            simulator_root=simulator_root,
        )
    expected_images = 2 if condition == "visual_only" else 4
    episode = {
        "round": "R0.9.14",
        "task": "pull_out_key",
        "seed": seed,
        "condition": condition,
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "status": "codex_running",
        "started_at": _utc_now(),
        "ended_at": None,
        "snapshot_id": _snapshot_id(snapshot),
        "tool_call_count": 0,
        "operator_context_row_count": 0,
        "mcp_tools": ["observe"],
        "simulator_invocation_count": 0,
        "codex_process_count": 1,
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
        rows = load_operator_context(
            episode_root / "operator_context.jsonl", expected_image_count=expected_images
        )
        codex_rows = read_jsonl(episode_root / "codex_exec.jsonl")
        codex_summary = summarize_codex_exec(codex_rows)
        final_text = (episode_root / "agent_final.md").read_text(encoding="utf-8").strip()
        prediction = parse_prediction(final_text)
        if lifecycle["returncode"] != 0 or lifecycle["timed_out"] or len(rows) != 1:
            raise RuntimeError("Codex condition did not complete with exactly one observe")
        episode.update(
            {
                "status": "completed",
                "ended_at": lifecycle["ended_at"],
                "duration_seconds": lifecycle["elapsed_seconds"],
                "tool_call_count": 1,
                "operator_context_row_count": 1,
                "parse_status": "parsed",
                "usage": codex_summary["usage"],
            }
        )
        write_json(episode_root / "episode.json", episode)
        return {
            "seed": seed,
            "condition": condition,
            "model": MODEL,
            **prediction,
            "duration_seconds": lifecycle["elapsed_seconds"],
            "usage": codex_summary["usage"],
            "tool_call_count": 1,
            "parse_status": "parsed",
            "raw_response_path": str((episode_root / "agent_final.md").relative_to(output_root)),
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


def _archive_failed_attempt(episode_root: Path) -> None:
    attempts_root = episode_root / "attempts"
    attempts_root.mkdir(exist_ok=True)
    index = 1
    while (attempts_root / f"attempt_{index}").exists():
        index += 1
    archive = attempts_root / f"attempt_{index}"
    archive.mkdir()
    for name in (
        "episode.json",
        "codex_exec.jsonl",
        "codex_stderr.log",
        "codex_lifecycle.json",
        "exception.json",
        "operator_context.jsonl",
        "agent_final.md",
        "operator-workspace",
        "runtime",
    ):
        source = episode_root / name
        if source.exists():
            shutil.move(str(source), archive / name)


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    scalar_rows = [
        {
            key: json.dumps(value, sort_keys=True) if isinstance(value, dict) else value
            for key, value in row.items()
        }
        for row in rows
    ]
    fieldnames = list(scalar_rows[0]) if scalar_rows else ["seed", "condition", "parse_status"]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scalar_rows)


def _write_raw_answers(output_root: Path, rows: list[dict[str, Any]]) -> None:
    sections = ["# R0.9.14 raw Codex answers", ""]
    for row in rows:
        response_path = output_root / str(row["raw_response_path"])
        sections.extend(
            (
                f"## seed {row['seed']} — {row['condition']}",
                "",
                "```json",
                response_path.read_text(encoding="utf-8").strip(),
                "```",
                "",
            )
        )
    (output_root / "raw_answers.md").write_text("\n".join(sections), encoding="utf-8")


def _finish_outputs(
    *,
    output_root: Path,
    predictions: list[dict[str, Any]],
    references: dict[str, Any],
    sources: list[dict[str, Any]],
    condition_errors: list[dict[str, Any]],
    pilot: dict[str, Any],
) -> dict[str, Any]:
    predictions = sorted(predictions, key=lambda row: (int(row["seed"]), str(row["condition"])))
    _write_predictions(output_root / "predictions.jsonl", predictions)
    _write_results_csv(output_root / "results.csv", predictions)
    _write_raw_answers(output_root, predictions)
    summary = summarize_pilot(predictions, references)
    complete = len(predictions) == 9 and not condition_errors
    summary.update(
        {
            "classification": (
                "pull_out_key_tactile_causal_pilot_completed"
                if complete
                else "pull_out_key_tactile_causal_pilot_incomplete"
            ),
            "pilot_signal": summary["pilot_signal"] if complete else "unavailable_incomplete",
            "valid_seeds": [row["seed"] for row in sources if row["valid"]],
            "invalid_seeds": [row["seed"] for row in sources if not row["valid"]],
            "completed_call_count": len(predictions),
            "condition_errors": condition_errors,
            "new_simulator_invocation_count": pilot["new_simulator_invocation_count"],
            "codex_process_count": pilot["codex_process_count"],
            "observe_tool_call_count": sum(row["tool_call_count"] for row in predictions),
            "action_tool_call_count": 0,
            "raw_answers_path": "raw_answers.md",
        }
    )
    write_json(output_root / "summary.json", summary)
    pilot.update(
        {
            "status": "completed" if complete else "incomplete",
            "ended_at": _utc_now(),
            "classification": summary["classification"],
            "pilot_signal": summary["pilot_signal"],
        }
    )
    write_json(output_root / "pilot.json", pilot)
    return summary


def _resume_failed(
    *,
    output_root: Path,
    config: dict[str, Any],
    codex_bin: str,
    auth_path: Path,
) -> int:
    pilot = _load_json(output_root / "pilot.json")
    sources = list(_load_json(output_root / "snapshot_sources.json")["sources"])
    predictions = read_jsonl(output_root / "predictions.jsonl")
    completed = {(int(row["seed"]), str(row["condition"])) for row in predictions}
    failed = [
        (seed, condition)
        for seed in SEEDS
        for condition in CONDITION_ORDER[seed]
        if (seed, condition) not in completed
    ]
    source_by_seed = {int(source["seed"]): source for source in sources}
    condition_errors: list[dict[str, Any]] = []
    resumed = 0
    for seed, condition in failed:
        pilot["codex_process_count"] = int(pilot.get("codex_process_count", 0)) + 1
        resumed += 1
        write_json(output_root / "pilot.json", pilot)
        try:
            predictions.append(
                _run_codex_condition(
                    seed=seed,
                    condition=condition,
                    source=source_by_seed[seed],
                    output_root=output_root,
                    codex_bin=codex_bin,
                    timeout_seconds=float(config["codex_timeout_seconds"]),
                    auth_path=auth_path,
                    reuse_staged=True,
                )
            )
        except Exception as exc:  # noqa: BLE001 - persist the real resume failure
            condition_errors.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
            )
            stderr = output_root / "runs" / f"seed_{seed}" / condition / "codex_stderr.log"
            if stderr.is_file() and "404 Not Found" in stderr.read_text(
                encoding="utf-8", errors="replace"
            ):
                break
    pilot["resume_codex_process_count"] = int(pilot.get("resume_codex_process_count", 0)) + resumed
    summary = _finish_outputs(
        output_root=output_root,
        predictions=predictions,
        references=_load_json(output_root / "host_reference.json"),
        sources=sources,
        condition_errors=condition_errors,
        pilot=pilot,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["classification"].endswith("_completed") else 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.expanduser().resolve(strict=True)
    runtime_python = args.runtime_python.expanduser().resolve(strict=True)
    source_root = args.source_root.expanduser().resolve(strict=True)
    reuse_root = args.reuse_seed1000000_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and not args.resume_failed:
        raise FileExistsError(f"output root must be fresh: {output_root}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("pilot config must be a mapping")
    config = validate_pilot_config(payload)
    codex_bin = shutil.which(args.codex_bin)
    auth_path = Path.home() / ".codex/auth.json"
    if codex_bin is None or not auth_path.is_file():
        raise RuntimeError("Codex CLI or existing authentication is unavailable")
    if args.resume_failed:
        return _resume_failed(
            output_root=output_root,
            config=config,
            codex_bin=codex_bin,
            auth_path=auth_path,
        )
    output_root.mkdir(parents=True)
    (output_root / "prompt.txt").write_text(PILOT_PROMPT, encoding="utf-8")
    pilot = {
        "round": "R0.9.14",
        "task": "pull_out_key",
        "seeds": list(SEEDS),
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "status": "running",
        "started_at": _utc_now(),
        "new_simulator_invocation_count": 0,
        "codex_process_count": 0,
        "action_tool_call_count": 0,
    }
    write_json(output_root / "pilot.json", pilot)
    base_gate_path = (REPO_ROOT / str(config["gate_config"])).resolve(strict=True)
    base_gate = yaml.safe_load(base_gate_path.read_text(encoding="utf-8"))
    sources = [_source_record(reuse_root / "simulator", SEEDS[0])]
    for seed in SEEDS[1:]:
        pilot["new_simulator_invocation_count"] += 1
        write_json(output_root / "pilot.json", pilot)
        sources.append(
            _run_seed_gate(
                seed=seed,
                base_config=base_gate,
                runtime_python=runtime_python,
                source_root=source_root,
                output_root=output_root,
                headless=bool(config["headless"]),
            )
        )
    write_json(output_root / "snapshot_sources.json", {"sources": sources})
    predictions: list[dict[str, Any]] = []
    condition_errors: list[dict[str, Any]] = []
    for source in sources:
        if not source["valid"]:
            continue
        seed = int(source["seed"])
        for condition in CONDITION_ORDER[seed]:
            pilot["codex_process_count"] += 1
            write_json(output_root / "pilot.json", pilot)
            try:
                predictions.append(
                    _run_codex_condition(
                        seed=seed,
                        condition=condition,
                        source=source,
                        output_root=output_root,
                        codex_bin=codex_bin,
                        timeout_seconds=float(config["codex_timeout_seconds"]),
                        auth_path=auth_path,
                    )
                )
            except Exception as exc:  # noqa: BLE001 - persist each real condition failure
                condition_errors.append(
                    {
                        "seed": seed,
                        "condition": condition,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    }
                )
            _write_predictions(output_root / "predictions.jsonl", predictions)
    references: dict[str, Any] = {}
    for source in sources:
        if source["valid"]:
            contact = _load_json(
                Path(source["simulator_root"])
                / seed_dir_name(int(source["seed"]))
                / "contact_summary.json"
            )
            references[str(source["seed"])] = press_depth_reference(contact)
    write_json(output_root / "host_reference.json", references)
    summary = _finish_outputs(
        output_root=output_root,
        predictions=predictions,
        references=references,
        sources=sources,
        condition_errors=condition_errors,
        pilot=pilot,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if pilot["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
