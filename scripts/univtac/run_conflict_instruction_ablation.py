#!/usr/bin/env python3
"""Run the zero-simulator R0.9.20 conflict-aware grounding interaction ablation."""

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
    read_jsonl,
    summarize_codex_exec,
)
from sim.envs.univtac.conflict_instruction_ablation import (
    COMMON_CONFLICT_INSTRUCTION,
    MCP_MODES,
    MCP_TOOLS,
    MODEL,
    PROTOCOL_ORDER,
    PROTOCOLS,
    REASONING_EFFORT,
    SEEDS,
    assert_paired_inputs_equal,
    parse_final_prediction,
    prompt_for,
    stage_protocol,
    summarize_predictions,
    validate_config,
    validate_trace,
)
from sim.envs.univtac.grounding_mechanism_ablation import EXPECTED_IMAGE_SIDE
from sim.envs.univtac.trace import write_json


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--r0918-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--codex-bin", default=os.environ.get("OPENETA_CODEX_BIN", "codex"))
    return parser.parse_args(argv)


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _snapshot(manifest_path: Path, seed: int) -> Path:
    manifest = _load_json(manifest_path)
    return (
        Path(str(manifest["simulator_root"]))
        / f"pull_out_key_seed{seed}"
        / "snapshot_post_pre_move.json"
    ).resolve(strict=True)


def _execute_attempt(
    *,
    seed: int,
    protocol: str,
    episode_root: Path,
    snapshot: Path,
    codex_bin: str,
    timeout_seconds: float,
    auth_path: Path,
    attempt_index: int,
) -> dict[str, Any]:
    tool_count = len(MCP_TOOLS[protocol])
    episode = {
        "round": "R0.9.20",
        "task": "pull_out_key",
        "seed": seed,
        "protocol": protocol,
        "model": MODEL,
        "reasoning_effort": REASONING_EFFORT,
        "status": "codex_running",
        "started_at": _utc_now(),
        "ended_at": None,
        "tool_call_count": 0,
        "operator_context_row_count": 0,
        "mcp_tools": MCP_TOOLS[protocol],
        "mcp_mode": MCP_MODES[protocol],
        "simulator_invocation_count": 0,
        "new_reset_count": 0,
        "new_snapshot_count": 0,
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
        mcp_mode=MCP_MODES[protocol],
        prompt=prompt_for(protocol),
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
        validate_trace(episode_root / "operator_context.jsonl", protocol)
        final_text = (episode_root / "agent_final.md").read_text(encoding="utf-8").strip()
        prediction = parse_final_prediction(final_text)
        trace_summary = summarize_codex_exec(read_jsonl(episode_root / "codex_exec.jsonl"))
        if lifecycle["returncode"] != 0 or lifecycle["timed_out"]:
            raise RuntimeError("Codex conflict-aware trial did not complete")
        episode.update(
            {
                "status": "completed",
                "ended_at": lifecycle["ended_at"],
                "duration_seconds": lifecycle["elapsed_seconds"],
                "tool_call_count": tool_count,
                "operator_context_row_count": tool_count,
                "parse_status": "parsed",
                "usage": trace_summary["usage"],
            }
        )
        write_json(episode_root / "episode.json", episode)
        return {
            "seed": seed,
            "protocol": protocol,
            "model": MODEL,
            "duration_seconds": lifecycle["elapsed_seconds"],
            "usage": trace_summary["usage"],
            "tool_call_count": tool_count,
            "parse_status": "parsed",
            **prediction,
            "raw_response_path": str(
                (episode_root / "agent_final.md").relative_to(episode_root.parents[2])
            ),
        }
    except Exception as exc:
        write_json(
            episode_root / "exception.json",
            {"error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()},
        )
        episode.update({"status": "failed", "ended_at": _utc_now(), "parse_status": "failed"})
        write_json(episode_root / "episode.json", episode)
        raise
    finally:
        (codex_home / "auth.json").unlink(missing_ok=True)
        shutil.rmtree(codex_home, ignore_errors=True)


def _run_protocol(
    *,
    seed: int,
    protocol: str,
    output_root: Path,
    codex_bin: str,
    timeout_seconds: float,
    auth_path: Path,
    retry_limit: int,
) -> tuple[dict[str, Any], int, int]:
    root = output_root / "runs" / f"seed_{seed}" / protocol
    snapshot = _snapshot(root / "condition.json", seed)
    attempts = failures = 0
    while True:
        attempts += 1
        try:
            return (
                _execute_attempt(
                    seed=seed,
                    protocol=protocol,
                    episode_root=root,
                    snapshot=snapshot,
                    codex_bin=codex_bin,
                    timeout_seconds=timeout_seconds,
                    auth_path=auth_path,
                    attempt_index=attempts,
                ),
                attempts,
                failures,
            )
        except Exception as exc:
            if _semantic_evidence_exists(root) or failures >= retry_limit:
                raise ConditionRunError(
                    str(exc), attempts=attempts, infrastructure_failures=failures
                ) from exc
            failures += 1
            _archive_failed_attempt(root)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_results(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "seed", "protocol", "expected_image_side", "text_indicated_side",
        "guidance_consistency", "final_changed_side", "final_evidence_basis",
        "follow_image", "follow_text", "duration_seconds", "input_tokens",
        "cached_input_tokens", "output_tokens", "reasoning_output_tokens", "tool_call_count",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            seed = int(row["seed"])
            expected = EXPECTED_IMAGE_SIDE[seed]
            text_side = "right" if expected == "left" else "left"
            usage = row.get("usage") or {}
            writer.writerow(
                {
                    "seed": seed, "protocol": row["protocol"],
                    "expected_image_side": expected, "text_indicated_side": text_side,
                    "guidance_consistency": row["guidance_consistency"],
                    "final_changed_side": row["final_changed_side"],
                    "final_evidence_basis": row["final_evidence_basis"],
                    "follow_image": row["final_changed_side"] == expected,
                    "follow_text": row["final_changed_side"] == text_side,
                    "duration_seconds": row["duration_seconds"],
                    "input_tokens": usage.get("input_tokens"),
                    "cached_input_tokens": usage.get("cached_input_tokens"),
                    "output_tokens": usage.get("output_tokens"),
                    "reasoning_output_tokens": usage.get("reasoning_output_tokens"),
                    "tool_call_count": row["tool_call_count"],
                }
            )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.expanduser().resolve(strict=True)
    source_root = args.source_root.expanduser().resolve(strict=True)
    r0918_root = args.r0918_root.expanduser().resolve(strict=True)
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        raise FileExistsError(f"output root must be fresh: {output_root}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("conflict instruction config must be a mapping")
    config = validate_config(payload)
    codex_bin = shutil.which(args.codex_bin)
    auth_path = Path.home() / ".codex/auth.json"
    if codex_bin is None or not auth_path.is_file():
        raise RuntimeError("Codex CLI or existing authentication is unavailable")
    source_pilot = _load_json(source_root / "pilot.json")
    r0919_summary = _load_json(source_root / "summary.json")
    r0918_summary = _load_json(r0918_root / "summary.json")
    if source_pilot.get("round") != "R0.9.19" or source_pilot.get("status") != "completed":
        raise ValueError("source root is not a completed R0.9.19 pilot")
    output_root.mkdir(parents=True)
    write_json(output_root / "config_resolved.json", {**config, "source_root": str(source_root)})
    (output_root / "common_conflict_instruction.txt").write_text(
        COMMON_CONFLICT_INSTRUCTION, encoding="utf-8"
    )
    prompts = output_root / "protocol_prompts"
    prompts.mkdir()
    for protocol in PROTOCOLS:
        (prompts / f"{protocol}.txt").write_text(prompt_for(protocol), encoding="utf-8")
    for seed in SEEDS:
        for protocol in PROTOCOLS:
            episode = output_root / "runs" / f"seed_{seed}" / protocol
            episode.mkdir(parents=True, exist_ok=False)
            stage_protocol(seed=seed, protocol=protocol, source_root=source_root, episode_root=episode)
    assert_paired_inputs_equal(output_root)
    pilot = {
        "round": "R0.9.20", "task": "pull_out_key", "seeds": list(SEEDS),
        "protocols": list(PROTOCOLS), "model": MODEL, "reasoning_effort": REASONING_EFFORT,
        "status": "running", "started_at": _utc_now(), "source_root": str(source_root),
        "simulator_invocation_count": 0, "new_reset_count": 0, "new_snapshot_count": 0,
        "codex_process_attempts": 0, "pre_image_infrastructure_failures": 0,
        "semantic_retries": 0, "action_tool_call_count": 0,
    }
    write_json(output_root / "pilot.json", pilot)
    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for seed in SEEDS:
        for protocol in PROTOCOL_ORDER[seed]:
            try:
                row, attempts, failures = _run_protocol(
                    seed=seed, protocol=protocol, output_root=output_root, codex_bin=codex_bin,
                    timeout_seconds=float(config["codex_timeout_seconds"]), auth_path=auth_path,
                    retry_limit=int(config["pre_image_tool_retry_limit"]),
                )
                rows.append(row)
                pilot["codex_process_attempts"] += attempts
                pilot["pre_image_infrastructure_failures"] += failures
            except Exception as exc:  # noqa: BLE001 - persist the real trial failure
                pilot["codex_process_attempts"] += int(getattr(exc, "attempts", 1))
                pilot["pre_image_infrastructure_failures"] += int(
                    getattr(exc, "infrastructure_failures", 0)
                )
                errors.append({"seed": seed, "protocol": protocol, "error": str(exc)})
            _write_jsonl(output_root / "predictions.jsonl", rows)
            write_json(output_root / "pilot.json", pilot)
    rows.sort(key=lambda row: (int(row["seed"]), PROTOCOLS.index(str(row["protocol"]))))
    _write_jsonl(output_root / "predictions.jsonl", rows)
    _write_results(output_root / "results.csv", rows)
    lines = ["# R0.9.20 conflict-aware interaction ablation answers", ""]
    for row in rows:
        lines.extend([
            f"## seed {row['seed']} — {row['protocol']}", "", "```json",
            (output_root / row["raw_response_path"]).read_text(encoding="utf-8").strip(), "```", "",
        ])
    (output_root / "raw_answers.md").write_text("\n".join(lines), encoding="utf-8")
    complete = len(rows) == 6 and not errors
    summary = (
        summarize_predictions(rows, r0919_summary, r0918_summary)
        if complete
        else {"classification": "tactile_conflict_instruction_interaction_ablation_incomplete", "minimal_skill_candidate": None}
    )
    summary.update({
        "completed_semantic_trials": len(rows), "codex_process_attempts": pilot["codex_process_attempts"],
        "pre_image_infrastructure_failures": pilot["pre_image_infrastructure_failures"],
        "semantic_retries": 0, "simulator_invocation_count": 0, "new_reset_count": 0,
        "new_snapshot_count": 0, "action_tool_call_count": 0, "errors": errors,
    })
    write_json(output_root / "summary.json", summary)
    dashboard = output_root / "dashboard"
    dashboard.mkdir()
    (dashboard / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><a href="http://127.0.0.1:9399/pilot/'
        f'{output_root.name}">Open the live R0.9.20 dashboard</a>\n', encoding="utf-8"
    )
    pilot.update({
        "status": "completed" if complete else "incomplete", "ended_at": _utc_now(),
        "classification": summary["classification"],
        "minimal_skill_candidate": summary["minimal_skill_candidate"],
    })
    write_json(output_root / "pilot.json", pilot)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
