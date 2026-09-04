#!/usr/bin/env python3
"""Run the fixed R1.3 causal Insert Hole tactile-action ICL pilot."""

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
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.univtac.run_codex_readonly_observation import _run_to_files
from sim.envs.univtac.codex_readonly import read_jsonl, summarize_codex_exec
from sim.envs.univtac.insert_hole_tactile_icl import (
    CONDITIONS,
    MAX_SIMULATOR_EPISODES,
    QUERY_SEEDS,
    R13_MCP_TOOLS,
    SUPPORT_SEEDS,
    SWAPPED_SKILL,
    build_balanced_support_bank,
    build_canonical_queries,
    build_r13_codex_command,
    stage_condition,
    summarize_r13_results,
    validate_r13_config,
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
    parser.add_argument("--r12-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--codex-bin", default=os.environ.get("OPENETA_CODEX_BIN", "codex"))
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


def _git_value(cwd: Path, *args: str) -> str | None:
    completed = subprocess.run(
        ["git", *args], cwd=cwd, check=False, capture_output=True, text=True
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _run_simulator(
    *,
    config_path: Path,
    config: dict[str, Any],
    runtime_python: Path,
    source_root: Path,
    episode_root: Path,
    seed: int,
    condition: str,
    headless: bool,
    skill: str | None = None,
    canonical_class: str | None = None,
) -> dict[str, Any]:
    command = [
        str(REPO_ROOT / "scripts/univtac/probe_branching_task.py"),
        "--config",
        str(config_path),
        "--source-root",
        str(source_root),
        "--repo-root",
        str(REPO_ROOT),
        "--output-root",
        str(episode_root),
        "--task",
        "insert_hole",
        "--seed",
        str(seed),
        "--condition",
        condition,
    ]
    if skill is not None:
        command.extend(("--skill", skill))
    if canonical_class is not None:
        command.extend(("--canonical-class", canonical_class))
    if headless:
        command.append("--headless")
    spec = ScopedIsaac51LaunchSpec(
        python_executable=runtime_python,
        command=tuple(command),
        cwd=source_root,
        output_root=episode_root,
        timeout_seconds=config["simulator_timeout_seconds"],
    )
    lifecycle = None
    error = None
    try:
        lifecycle = run_scoped_isaac51_command(spec).to_dict()
    except ScopedIsaac51LaunchError as exc:
        error = f"{type(exc).__name__}: {exc}"
    if lifecycle is not None:
        write_json(episode_root / "launcher_lifecycle.json", lifecycle)
    child_path = episode_root / "child_result.json"
    child = _read_json(child_path) if child_path.is_file() else {}
    final_path = episode_root / "final_result.json"
    final = _read_json(final_path) if final_path.is_file() else {}
    cleanup = child.get("cleanup", {})
    infrastructure_valid = bool(
        lifecycle
        and lifecycle.get("returncode") == 0
        and lifecycle.get("timed_out") is False
        and lifecycle.get("cleanup_complete")
        and child.get("status") == "completed"
        and cleanup.get("task_close")
    )
    row = {
        "seed": seed,
        "condition": condition,
        "infrastructure_valid": infrastructure_valid,
        "classification": child.get("classification", "native_runtime_abort"),
        "plan_success": final.get("plan_success"),
        "native_check_success": final.get("native_check_success"),
        "native_check_early_stop": final.get("native_check_early_stop"),
        "native_episode_success": bool(final.get("expert_episode_success")),
        "decision_class": final.get("decision_class"),
        "native_x_move": final.get("native_x_move"),
        "native_z_move": final.get("native_z_move"),
        "selected_skill": final.get("selected_skill"),
        "applied_x_move": final.get("applied_x_move"),
        "applied_z_move": final.get("applied_z_move"),
        "execution_state_mismatch": final.get("execution_state_mismatch"),
        "agent_world_changing_choice_count": final.get(
            "agent_world_changing_choice_count", 0
        ),
        "play_once_call_count": child.get("counters", {}).get("play_once_call_count", 0),
        "check_success_call_count": child.get("counters", {}).get(
            "check_success_call_count", 0
        ),
        "check_early_stop_call_count": child.get("counters", {}).get(
            "check_early_stop_call_count", 0
        ),
        "action_trace_complete": final.get("action_trace_complete", False),
        "decision_snapshot_complete": final.get("decision_snapshot_complete", False),
        "episode_root": str(episode_root),
        "error": error or child.get("error"),
    }
    write_json(episode_root / "r13_result_row.json", row)
    return row


def _flatten_image_paths(value: Any) -> list[str]:
    paths: list[str] = []
    if isinstance(value, dict):
        if {"label", "path", "shape", "dtype"}.issubset(value):
            paths.append(str(value["path"]))
        else:
            for child in value.values():
                paths.extend(_flatten_image_paths(child))
    elif isinstance(value, list):
        for child in value:
            paths.extend(_flatten_image_paths(child))
    return paths


def _assert_same_condition_images(left_root: Path, right_root: Path) -> None:
    left = _read_json(left_root / "condition.json")["agent_visible"]
    right = _read_json(right_root / "condition.json")["agent_visible"]
    left_paths = _flatten_image_paths(left)
    right_paths = _flatten_image_paths(right)
    if len(left_paths) != len(right_paths):
        raise RuntimeError("paired condition image counts differ")
    for left_path, right_path in zip(left_paths, right_paths, strict=True):
        left_array = np.asarray(Image.open(left_root / left_path))
        right_array = np.asarray(Image.open(right_root / right_path))
        if not np.array_equal(left_array, right_array):
            raise RuntimeError("paired condition image arrays differ")


def _descriptors(value: Any) -> list[dict[str, Any]]:
    found: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if {"label", "path", "shape", "dtype"}.issubset(value):
            found.append(value)
        else:
            for child in value.values():
                found.extend(_descriptors(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_descriptors(child))
    return found


def _assert_query_control(cell_roots: dict[str, Path]) -> None:
    payloads = {
        name: _read_json(root / "condition.json")["agent_visible"]["query"]
        for name, root in cell_roots.items()
    }
    reference = payloads["correct_icl_multimodal"]
    for name in (
        "no_demo_multimodal",
        "correct_icl_multimodal",
        "action_swapped_icl_multimodal",
    ):
        payload = payloads[name]
        if (
            payload["before_contact"]["proprio"]
            != reference["before_contact"]["proprio"]
            or payload["after_contact"]["proprio"]
            != reference["after_contact"]["proprio"]
        ):
            raise RuntimeError("R1.3 canonical query proprio differs across conditions")
        left = _descriptors(reference)
        right = _descriptors(payload)
        if [row["label"] for row in left] != [row["label"] for row in right]:
            raise RuntimeError("R1.3 canonical query labels differ across multimodal conditions")
        for left_row, right_row in zip(left, right, strict=True):
            left_array = np.asarray(
                Image.open(cell_roots["correct_icl_multimodal"] / left_row["path"])
            )
            right_array = np.asarray(Image.open(cell_roots[name] / right_row["path"]))
            if not np.array_equal(left_array, right_array):
                raise RuntimeError("R1.3 canonical query arrays differ across conditions")
    vision = payloads["correct_icl_vision_only"]
    if (
        vision["before_contact"]["proprio"] != reference["before_contact"]["proprio"]
        or vision["after_contact"]["proprio"] != reference["after_contact"]["proprio"]
    ):
        raise RuntimeError("R1.3 vision-only query changed proprio")
    expected = [row for row in _descriptors(reference) if "tactile" not in row["label"]]
    actual = _descriptors(vision)
    if [row["label"] for row in expected] != [row["label"] for row in actual]:
        raise RuntimeError("R1.3 vision-only query changed more than tactile images")
    for expected_row, actual_row in zip(expected, actual, strict=True):
        expected_array = np.asarray(
            Image.open(cell_roots["correct_icl_multimodal"] / expected_row["path"])
        )
        actual_array = np.asarray(
            Image.open(cell_roots["correct_icl_vision_only"] / actual_row["path"])
        )
        if not np.array_equal(expected_array, actual_array):
            raise RuntimeError("R1.3 vision-only query changed a retained visual image")


def _assert_condition_interventions(cell_roots: dict[str, Path]) -> None:
    correct = _read_json(
        cell_roots["correct_icl_multimodal"] / "condition.json"
    )["agent_visible"]
    swapped = _read_json(
        cell_roots["action_swapped_icl_multimodal"] / "condition.json"
    )["agent_visible"]
    swapped_normalized = json.loads(json.dumps(swapped))
    for correct_demo, swapped_demo in zip(
        correct["demonstrations"],
        swapped_normalized["demonstrations"],
        strict=True,
    ):
        if swapped_demo["action"] != SWAPPED_SKILL[correct_demo["action"]]:
            raise RuntimeError("R1.3 swapped demonstration label is invalid")
        swapped_demo["action"] = correct_demo["action"]
    if swapped_normalized != correct:
        raise RuntimeError("R1.3 C1/C2 differ by more than opaque action labels")

    vision = _read_json(
        cell_roots["correct_icl_vision_only"] / "condition.json"
    )["agent_visible"]

    def remove_tactile(value: Any) -> Any:
        if isinstance(value, dict):
            if {"label", "path", "shape", "dtype"}.issubset(value):
                return None if "tactile" in str(value["label"]) else value
            return {key: remove_tactile(child) for key, child in value.items()}
        if isinstance(value, list):
            return [
                cleaned
                for child in value
                if (cleaned := remove_tactile(child)) is not None
            ]
        return value

    if remove_tactile(correct) != vision:
        raise RuntimeError("R1.3 C1/C3 differ by more than tactile image removal")


def _validate_operator_rows(
    rows: list[dict[str, Any]], *, has_demos: bool, image_mode: str
) -> None:
    tools = [str(row.get("tool")) for row in rows]
    if tools != ["review_demonstrations", "observe_query", "choose_skill"]:
        raise RuntimeError(f"R1.3 tool order/count is invalid: {tools}")
    if set(tools) != set(R13_MCP_TOOLS):
        raise RuntimeError("R1.3 exposed or called an unexpected MCP tool")
    review_images = len(rows[0].get("response_image_paths", []))
    expected_review = 0 if not has_demos else 12 if image_mode == "visual_only" else 24
    if review_images != expected_review:
        raise RuntimeError(
            f"R1.3 review returned {review_images} images, expected {expected_review}"
        )
    query_images = len(rows[1].get("response_image_paths", []))
    expected_query = 4 if image_mode == "visual_only" else 8
    if query_images != expected_query:
        raise RuntimeError(
            f"R1.3 query returned {query_images} images, expected {expected_query}"
        )
    if rows[2].get("response_image_paths"):
        raise RuntimeError("R1.3 choose_skill must not return images")


def _run_codex_decision(
    *,
    episode_root: Path,
    config: dict[str, Any],
    codex_bin: str,
    auth_path: Path,
) -> dict[str, Any]:
    condition = _read_json(episode_root / "condition.json")
    host = condition["host_only"]
    visible = condition["agent_visible"]
    workspace = episode_root / "operator-workspace"
    codex_home = episode_root / "runtime/codex-home"
    workspace.mkdir()
    codex_home.mkdir(parents=True)
    subprocess.run(["git", "init", "-q"], cwd=workspace, check=True)
    (codex_home / "auth.json").symlink_to(auth_path)
    command = build_r13_codex_command(
        codex_bin=codex_bin,
        workspace=workspace,
        repo_root=REPO_ROOT,
        episode_root=episode_root,
        final_response_path=episode_root / "agent_final.md",
        model=config["model"],
        reasoning_effort=config["reasoning_effort"],
    )
    write_json(episode_root / "codex_command.json", {"command": command})
    started = time.monotonic()
    lifecycle: dict[str, Any] = {}
    try:
        environment = dict(os.environ)
        environment["CODEX_HOME"] = str(codex_home)
        lifecycle = _run_to_files(
            command,
            cwd=workspace,
            environment=environment,
            stdout_path=episode_root / "codex_exec.jsonl",
            stderr_path=episode_root / "codex_stderr.log",
            timeout_seconds=config["codex_timeout_seconds"],
        )
        write_json(episode_root / "codex_lifecycle.json", lifecycle)
        decision_path = episode_root / "decision.json"
        if not decision_path.is_file():
            raise RuntimeError("Codex did not submit an R1.3 skill choice")
        decision = _read_json(decision_path)
        operator_rows = read_jsonl(episode_root / "operator_context.jsonl")
        _validate_operator_rows(
            operator_rows,
            has_demos=bool(visible["demonstrations"]),
            image_mode=host["image_mode"],
        )
        expected = str(host["expected_skill"])
        selected = str(decision["skill"])
        row = {
            "seed": int(host["seed"]),
            "condition": str(host["condition"]),
            "canonical_query_id": host["canonical_query_id"],
            "canonical_decision_class": host["canonical_decision_class"],
            "expected_skill": expected,
            "selected_skill": selected,
            "selection_correct": selected == expected,
            "confidence": decision["confidence"],
            "reason": decision["reason"],
            "choice_count": decision["choice_count"],
            "corrupted_demo_followed": (
                host["condition"] == "action_swapped_icl_multimodal"
                and selected == SWAPPED_SKILL[expected]
            ),
            "codex_completed": bool(
                lifecycle.get("returncode") == 0 and not lifecycle.get("timed_out")
            ),
            "duration_seconds": time.monotonic() - started,
            "token_usage": summarize_codex_exec(
                read_jsonl(episode_root / "codex_exec.jsonl")
            ).get("usage"),
            "run": str(episode_root),
        }
        write_json(episode_root / "decision_result.json", row)
        write_json(
            episode_root / "episode.json",
            {
                "schema_version": "openeta.univtac.r13.decision_episode.v1",
                "round": "R1.3",
                "task": "insert_hole",
                "seed": int(host["seed"]),
                "condition": str(host["condition"]),
                "model": config["model"],
                "reasoning_effort": config["reasoning_effort"],
                "status": "completed",
                "duration_seconds": row["duration_seconds"],
                "tool_call_count": len(operator_rows),
                "choice_count": decision["choice_count"],
            },
        )
        return row
    finally:
        (codex_home / "auth.json").unlink(missing_ok=True)
        shutil.rmtree(codex_home, ignore_errors=True)


def _archive_failed_prechoice_attempt(episode_root: Path) -> None:
    if (episode_root / "decision.json").exists():
        return
    lifecycle = episode_root / "codex_lifecycle.json"
    if not lifecycle.is_file():
        return
    archive = episode_root / "attempts/attempt_01_failed_before_choice"
    if archive.exists():
        return
    archive.mkdir(parents=True)
    for name in (
        "agent_final.md",
        "codex_command.json",
        "codex_exec.jsonl",
        "codex_lifecycle.json",
        "codex_stderr.log",
        "operator_context.jsonl",
        "operator-workspace",
        "runtime",
    ):
        source = episode_root / name
        if source.exists():
            shutil.move(str(source), archive / name)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "seed",
        "condition",
        "canonical_decision_class",
        "expected_skill",
        "selected_skill",
        "selection_correct",
        "confidence",
        "native_continuation_success",
        "execution_state_mismatch",
        "codex_completed",
        "duration_seconds",
        "token_usage",
        "reason",
        "run",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row.get(key))
                    if isinstance(row.get(key), (dict, list))
                    else row.get(key)
                    for key in fields
                }
            )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.expanduser().resolve(strict=True)
    runtime_python = args.runtime_python.expanduser().resolve(strict=True)
    source_root = args.source_root.expanduser().resolve(strict=True)
    r12_root = args.r12_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and not args.resume:
        raise FileExistsError(f"R1.3 output root must be fresh: {output_root}")
    config = validate_r13_config(yaml.safe_load(config_path.read_text(encoding="utf-8")))
    codex_bin = shutil.which(args.codex_bin)
    auth_path = Path.home() / ".codex/auth.json"
    if codex_bin is None or not auth_path.is_file():
        raise FileNotFoundError("Codex CLI or existing authentication is unavailable")
    source_head = _git_value(source_root, "rev-parse", "HEAD")
    if source_head != config["source_commit"]:
        raise RuntimeError(f"pinned source HEAD mismatch: {source_head}")
    if _git_value(source_root, "status", "--porcelain"):
        raise RuntimeError("pinned source checkout must remain clean")

    if args.resume:
        if not output_root.is_dir():
            raise FileNotFoundError(f"R1.3 resume root is unavailable: {output_root}")
        if (output_root / "executions").exists():
            raise RuntimeError("R1.3 resume is only allowed before physical continuations")
        manifest = _read_json(output_root / "run_manifest.json")
        support = {
            "agent_visible": _read_json(
                output_root / "balanced_support_bank/agent_visible.json"
            ),
            "host_only": _read_json(
                output_root / "balanced_support_bank/host_only.json"
            ),
        }
        queries = {
            "agent_visible": _read_json(
                output_root / "canonical_query_states/agent_visible.json"
            ),
            "host_only": _read_json(
                output_root / "canonical_query_states/host_only.json"
            ),
        }
        oracle_summary = _read_json(output_root / "query_oracle_summary.json")
        oracle_rows = list(oracle_summary["rows"])
        oracle_success = int(oracle_summary["success_count"])
        manifest["status"] = "resuming_after_prechoice_infrastructure_failure"
        manifest["resume_count"] = int(manifest.get("resume_count", 0)) + 1
        write_json(output_root / "run_manifest.json", manifest)
    else:
        output_root.mkdir(parents=True)
        manifest = {
            "schema_version": "openeta.univtac.r13.run.v1",
            "round": "R1.3",
            "status": "running_query_oracle",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "repo_head": _git_value(REPO_ROOT, "rev-parse", "HEAD"),
            "repo_branch": _git_value(REPO_ROOT, "branch", "--show-current"),
            "source_head": source_head,
            "model": config["model"],
            "reasoning_effort": config["reasoning_effort"],
            "support_seeds": list(SUPPORT_SEEDS),
            "query_seeds": list(QUERY_SEEDS),
            "simulator_invocation_count": 0,
            "codex_decision_count": 0,
            "max_simulator_episodes": MAX_SIMULATOR_EPISODES,
        }
        write_json(output_root / "run_manifest.json", manifest)
        support = build_balanced_support_bank(
            r12_root=r12_root, output_root=output_root / "balanced_support_bank"
        )
        oracle_rows = []
        for seed in QUERY_SEEDS:
            episode_root = output_root / "query_oracle" / f"seed_{seed}"
            manifest["simulator_invocation_count"] += 1
            write_json(output_root / "run_manifest.json", manifest)
            row = _run_simulator(
                config_path=config_path,
                config=config,
                runtime_python=runtime_python,
                source_root=source_root,
                episode_root=episode_root,
                seed=seed,
                condition="expert",
                headless=args.headless,
            )
            oracle_rows.append(row)
        oracle_success = sum(row["native_episode_success"] for row in oracle_rows)
        oracle_summary = {
            "success_count": oracle_success,
            "success_rate": oracle_success / 3,
            "class_distribution": dict(
                sorted(
                    Counter(str(row["decision_class"]) for row in oracle_rows).items()
                )
            ),
            "class_coverage_limited": len(
                {row["decision_class"] for row in oracle_rows}
            )
            < 2,
            "rows": oracle_rows,
        }
        write_json(output_root / "query_oracle_summary.json", oracle_summary)
    if oracle_success < 2 or not all(row["infrastructure_valid"] for row in oracle_rows):
        manifest.update(
            {
                "status": "stopped_at_query_qualification",
                "ended_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        write_json(output_root / "run_manifest.json", manifest)
        write_json(
            output_root / "summary.json",
            {
                "classification": "insert_hole_query_qualification_failed",
                "query_oracle": oracle_summary,
                "codex_decision_count": 0,
                "simulator_invocation_count": 3,
            },
        )
        return 1

    if not args.resume:
        queries = build_canonical_queries(
            oracle_root=output_root / "query_oracle",
            output_root=output_root / "canonical_query_states",
        )
    manifest["status"] = "running_codex_decisions"
    write_json(output_root / "run_manifest.json", manifest)
    decisions: list[dict[str, Any]] = []
    for seed in QUERY_SEEDS:
        cell_roots = {}
        for condition in CONDITIONS:
            cell = output_root / "decisions" / f"seed_{seed}" / condition
            if not (cell / "condition.json").is_file():
                stage_condition(
                    support_bank=support,
                    support_root=output_root / "balanced_support_bank",
                    query_bank=queries,
                    query_root=output_root / "canonical_query_states",
                    seed=seed,
                    condition=condition,
                    episode_root=cell,
                )
            cell_roots[condition] = cell
        _assert_same_condition_images(
            cell_roots["correct_icl_multimodal"],
            cell_roots["action_swapped_icl_multimodal"],
        )
        _assert_query_control(cell_roots)
        _assert_condition_interventions(cell_roots)
        for condition in CONDITIONS:
            cell = cell_roots[condition]
            result_path = cell / "decision_result.json"
            if result_path.is_file():
                row = _read_json(result_path)
            else:
                _archive_failed_prechoice_attempt(cell)
                row = _run_codex_decision(
                    episode_root=cell,
                    config=config,
                    codex_bin=codex_bin,
                    auth_path=auth_path,
                )
            decisions.append(row)
            manifest["codex_decision_count"] = len(decisions)
            write_json(output_root / "run_manifest.json", manifest)

    if len(decisions) != 12 or any(row["choice_count"] != 1 for row in decisions):
        raise RuntimeError("R1.3 requires twelve saved one-choice Codex decisions")
    (output_root / "codex_decisions.jsonl").unlink(missing_ok=True)
    for row in decisions:
        _append_jsonl(output_root / "codex_decisions.jsonl", row)

    manifest["status"] = "running_physical_continuations"
    write_json(output_root / "run_manifest.json", manifest)
    results: list[dict[str, Any]] = []
    for decision in decisions:
        seed = int(decision["seed"])
        condition = str(decision["condition"])
        episode_root = output_root / "executions" / f"seed_{seed}" / condition
        manifest["simulator_invocation_count"] += 1
        if manifest["simulator_invocation_count"] > MAX_SIMULATOR_EPISODES:
            raise RuntimeError("R1.3 simulator episode budget exceeded")
        write_json(output_root / "run_manifest.json", manifest)
        physical = _run_simulator(
            config_path=config_path,
            config=config,
            runtime_python=runtime_python,
            source_root=source_root,
            episode_root=episode_root,
            seed=seed,
            condition="selected",
            headless=args.headless,
            skill=str(decision["selected_skill"]),
            canonical_class=str(decision["canonical_decision_class"]),
        )
        if physical["agent_world_changing_choice_count"] != 1:
            raise RuntimeError("R1.3 physical continuation must execute one Agent choice")
        result = {
            **decision,
            "native_continuation_success": physical["native_episode_success"],
            "execution_state_mismatch": physical["execution_state_mismatch"],
            "execution_decision_class": physical["decision_class"],
            "native_x_move": physical["native_x_move"],
            "native_z_move": physical["native_z_move"],
            "applied_x_move": physical["applied_x_move"],
            "applied_z_move": physical["applied_z_move"],
            "physical_infrastructure_valid": physical["infrastructure_valid"],
            "execution_root": str(episode_root),
        }
        results.append(result)
        _append_jsonl(output_root / "execution_results.jsonl", result)

    summary = summarize_r13_results(results)
    summary.update(
        {
            "query_oracle": oracle_summary,
            "support_bank": _read_json(
                output_root / "balanced_support_bank/host_only.json"
            ),
            "simulator_invocation_count": manifest["simulator_invocation_count"],
            "codex_decision_count": manifest["codex_decision_count"],
            "all_physical_infrastructure_valid": all(
                row["physical_infrastructure_valid"] for row in results
            ),
            "execution_state_mismatch_count": sum(
                bool(row["execution_state_mismatch"]) for row in results
            ),
        }
    )
    _write_csv(output_root / "results.csv", results)
    write_json(output_root / "summary.json", summary)
    with (output_root / "raw_answers.md").open("w", encoding="utf-8") as stream:
        for row in decisions:
            answer = Path(row["run"]) / "agent_final.md"
            stream.write(f"## seed {row['seed']} · {row['condition']}\n\n")
            stream.write(
                (answer.read_text(encoding="utf-8").strip() if answer.is_file() else "unavailable")
                + "\n\n"
            )
    manifest.update(
        {
            "status": "completed",
            "classification": summary["classification"],
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "action_icl_signal": summary["action_icl_signal"],
            "tactile_icl_signal": summary["tactile_icl_signal"],
        }
    )
    write_json(output_root / "run_manifest.json", manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["all_physical_infrastructure_valid"] else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"R1.3 failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
        raise
