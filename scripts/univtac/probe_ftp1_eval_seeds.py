#!/usr/bin/env python3
"""Audit FTP-1 and run reset-only probes on its first evaluation seeds."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.univtac.audit_ftp1_protocol import (
    CANONICAL_COMMIT,
    audit_protocol,
    render_markdown,
)
from sim.envs.univtac.resource_sanitation import run_managed_process
from sim.envs.univtac.trace import write_json


FIXED_PROBE_SEEDS = (1000000, 1000001, 1000002)
PRIMARY_TASKS = ("insert_hole", "insert_tube", "pull_out_key")
CONDITIONAL_CONTROL_TASK = "lift_can"
LAUNCHER_EXACT = "ftp1_exact_startup"
LAUNCHER_PARITY = "ftp1_protocol_headless_parity"
DEFAULT_CONFIG = REPO_ROOT / "configs/univtac/ftp1_eval_seed_probe.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/ftp1-eval-seed-probe"
PROBE_IMPORT_FLAGS = {
    "ftp1_model_loaded": False,
    "openpi_imported": False,
    "openeta_imported": False,
    "planner_behavior_modified": False,
}
SECRET_PATTERN = re.compile(
    r"(?i)(api[_-]?key|authorization|bearer|credential|password|secret|token)"
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _git_head(path: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def validate_probe_config(config: dict[str, Any]) -> None:
    if tuple(config.get("seeds", ())) != FIXED_PROBE_SEEDS:
        raise ValueError(
            f"FTP-1 probe seeds must be exactly {list(FIXED_PROBE_SEEDS)}"
        )
    if tuple(config.get("tasks", ())) != PRIMARY_TASKS:
        raise ValueError(f"primary tasks must be exactly {list(PRIMARY_TASKS)}")
    control = config.get("conditional_control", {})
    if control.get("task") != CONDITIONAL_CONTROL_TASK:
        raise ValueError("conditional control must be lift_can")
    if tuple(control.get("seeds", ())) != FIXED_PROBE_SEEDS:
        raise ValueError("conditional control must use the fixed paired seeds")
    if int(config.get("required_valid_seeds", 0)) != len(FIXED_PROBE_SEEDS):
        raise ValueError("required_valid_seeds must equal three")
    if config.get("canonical_commit") != CANONICAL_COMMIT:
        raise ValueError("canonical FTP-1 commit does not match the frozen revision")
    if int(config.get("gpu", -1)) != 0:
        raise ValueError("FTP-1 parity probe must use public shell default GPU=0")


def reset_is_valid(record: dict[str, Any]) -> bool:
    """Evaluate only the reset/observation contract, never task or policy success."""

    return bool(
        record.get("reset_returned") is True
        and record.get("plan_success") is True
        and record.get("observation_available") is True
        and len(record.get("tactile_sensor_names", ())) == 2
        and record.get("rgb_marker_valid") is True
    )


def should_run_conditional_control(records: Iterable[dict[str, Any]]) -> bool:
    records = list(records)
    return len(records) == len(PRIMARY_TASKS) * len(FIXED_PROBE_SEEDS) and all(
        record.get("reached_reset") is True and not reset_is_valid(record)
        for record in records
    )


def _descriptor_values(value: Any) -> list[float] | None:
    if isinstance(value, list) and all(isinstance(item, (int, float)) for item in value):
        return [float(item) for item in value]
    if isinstance(value, dict):
        values = value.get("values")
        if isinstance(values, list) and all(
            isinstance(item, (int, float)) for item in values
        ):
            return [float(item) for item in values]
    return None


def _translation_delta(planner_failure: dict[str, Any] | None) -> list[float] | None:
    if not planner_failure:
        return None
    target = planner_failure.get("failed_target_ee_pose", {}).get("position")
    current = (
        planner_failure.get("robot_state_before_failure", {})
        .get("current_ee_pose", {})
        .get("position")
    )
    target_values = _descriptor_values(target)
    current_values = _descriptor_values(current)
    if not target_values or not current_values or len(target_values) != len(current_values):
        return None
    return [target - current for target, current in zip(target_values, current_values)]


def _native_field(planner_failure: dict[str, Any] | None, name: str) -> Any:
    if not planner_failure:
        return None
    value = planner_failure.get("native_result", {}).get(name)
    if isinstance(value, dict) and "value" in value:
        return value["value"]
    return value


def normalize_seed_result(
    *,
    task: str,
    seed: int,
    task_root: Path,
    task_output: Path,
    launcher_mode: str,
    process_run: dict[str, Any] | None = None,
) -> dict[str, Any]:
    seed_dir = task_output / f"seed_{seed}"
    diagnostic_path = seed_dir / "reset_diagnostic.json"
    run_path = task_output / "diagnostic_run.json"
    run = _read_json(run_path) if run_path.is_file() else {}
    if not diagnostic_path.is_file():
        error = run.get("error") or _native_process_error(process_run)
        reached_reset = bool(run.get("reached_reset"))
        return {
            "task": task,
            "seed": seed,
            "source_commit": _git_head(task_root.parent),
            "task_module_realpath": str((task_root / "envs" / f"{task}.py").resolve()),
            "task_config_realpath": run.get("task_config"),
            "asset_root": run.get("asset_root_requested"),
            "launcher_mode": launcher_mode,
            "reset_started": reached_reset,
            "reset_returned": False,
            "plan_success": False,
            "observation_available": False,
            "tactile_sensor_names": [],
            "tactile_metadata": {},
            "rgb_marker_valid": False,
            "reset_valid": False,
            "cleanup_success": False,
            "failure_stage": "reset_runtime_error" if reached_reset else "startup_before_reset",
            "error_class": error.get("type"),
            "error_message": error.get("message"),
            "planning_call_count": 0,
            "first_failed_call": None,
            "move_call_index": None,
            "status": None,
            "valid_query": None,
            "target_current_translation": None,
            **PROBE_IMPORT_FLAGS,
            "reached_task_construction": bool(run.get("reached_task_construction")),
            "reached_reset": reached_reset,
            "failed_before_reset": not reached_reset,
            "process_returncode": (
                process_run.get("returncode") if process_run else None
            ),
        }

    diagnostic = _read_json(diagnostic_path)
    failure_path = seed_dir / "planner_failure.json"
    planner_failure = _read_json(failure_path) if failure_path.is_file() else None
    exception = diagnostic.get("reset_exception_summary") or {}
    record = {
        "task": task,
        "seed": seed,
        "source_commit": _git_head(task_root.parent),
        "task_module_realpath": diagnostic.get("task_module_realpath"),
        "task_config_realpath": diagnostic.get("task_config_realpath"),
        "asset_root": diagnostic.get("asset_root"),
        "launcher_mode": launcher_mode,
        "reset_started": bool(diagnostic.get("reset_started")),
        "reset_returned": diagnostic.get("reset_returned") is True,
        "plan_success": diagnostic.get("plan_success") is True,
        "observation_available": diagnostic.get("observation_checked") is True,
        "tactile_sensor_names": diagnostic.get("tactile_sensor_names", []),
        "tactile_metadata": diagnostic.get("tactile_metadata", {}),
        "rgb_marker_valid": diagnostic.get("rgb_marker_valid") is True,
        "cleanup_success": diagnostic.get("cleanup_ok") is True,
        "failure_stage": diagnostic.get("failure_stage"),
        "error_class": exception.get("type"),
        "error_message": exception.get("message"),
        "planning_call_count": diagnostic.get("planner_call_count", 0),
        "first_failed_call": (
            planner_failure.get("failed_planning_call_index")
            if planner_failure
            else None
        ),
        "move_call_index": (
            planner_failure.get("failed_move_call_index") if planner_failure else None
        ),
        "status": (
            planner_failure.get("native_result_status") if planner_failure else None
        ),
        "valid_query": _native_field(planner_failure, "valid_query"),
        "attempts": _native_field(planner_failure, "attempts"),
        "ik_time": _native_field(planner_failure, "ik_time"),
        "graph_time": _native_field(planner_failure, "graph_time"),
        "trajopt_time": _native_field(planner_failure, "trajopt_time"),
        "solve_time": _native_field(planner_failure, "solve_time"),
        "target_current_translation": _translation_delta(planner_failure),
        "constraint_pose": (
            _failed_planner_call(seed_dir).get("constraint_pose")
            if planner_failure
            else None
        ),
        **PROBE_IMPORT_FLAGS,
        "reached_task_construction": True,
        "reached_reset": True,
        "failed_before_reset": False,
        "process_returncode": process_run.get("returncode") if process_run else None,
    }
    record["reset_valid"] = reset_is_valid(record)
    return record


def _native_process_error(process_run: dict[str, Any] | None) -> dict[str, str]:
    fallback = {
        "type": "MissingResetDiagnostic",
        "message": "the process ended before a per-seed reset diagnostic was written",
    }
    if not process_run:
        return fallback
    log_path = Path(str(process_run.get("log_path", "")))
    if not log_path.is_file():
        return fallback
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if "cudaErrorInvalidDevice" in text:
        return {
            "type": "NativeProcessAbort",
            "message": (
                "TacEx UIPC aborted with cudaErrorInvalidDevice in "
                "muda CUB DeviceRadixSort before reset returned"
            ),
        }
    if "Fatal Python error: Aborted" in text and "_wait_for_viewport" in text:
        return {
            "type": "NativeProcessAbort",
            "message": (
                "Isaac SimulationApp aborted while waiting for the viewport; "
                "the log also reports inotify change-watch exhaustion"
            ),
        }
    if process_run.get("timed_out"):
        return {
            "type": "ProcessTimeout",
            "message": "the reset probe process exceeded its configured timeout",
        }
    return {
        "type": "NativeProcessExit",
        "message": f"native process exited with return code {process_run.get('returncode')}",
    }


def _failed_planner_call(seed_dir: Path) -> dict[str, Any]:
    failure_path = seed_dir / "planner_failure.json"
    if not failure_path.is_file():
        return {}
    failure = _read_json(failure_path)
    index = failure.get("failed_planning_call_index")
    calls_path = seed_dir / "planner_calls.jsonl"
    if not calls_path.is_file():
        return {}
    for line in calls_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record.get("planning_call_index") == index:
            return record
    return {}


def classify_task(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    records = list(records)
    valid_count = sum(reset_is_valid(record) for record in records)
    if len(records) != len(FIXED_PROBE_SEEDS):
        classification = "runtime_error"
    elif valid_count == len(FIXED_PROBE_SEEDS):
        classification = "all_three_reset_valid"
    elif 0 < valid_count < len(FIXED_PROBE_SEEDS):
        classification = "mixed_reset_validity"
    elif any(record.get("failure_stage") == "startup_before_reset" for record in records):
        classification = "startup_blocked"
    elif any(record.get("failure_stage") == "reset_runtime_error" for record in records):
        classification = "runtime_error"
    else:
        signatures = {
            (
                record.get("first_failed_call"),
                record.get("status"),
                json.dumps(record.get("constraint_pose"), sort_keys=True),
            )
            for record in records
        }
        all_planner = all(
            record.get("failure_stage") == "reset_pre_move_planner"
            for record in records
        )
        if all_planner and len(signatures) == 1:
            classification = "all_three_same_planner_failure"
        elif all(record.get("failure_stage") for record in records):
            classification = "all_three_different_failures"
        else:
            classification = "runtime_error"
    return {
        "classification": classification,
        "suspected_structural_failure": classification
        == "all_three_same_planner_failure",
        "reset_valid_count": valid_count,
        "seeds": {str(record["seed"]): record for record in records},
    }


def _run_subprocess(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    timeout: int,
    environment_overrides: dict[str, str] | None = None,
    unset_environment: tuple[str, ...] = (),
) -> dict[str, Any]:
    child_environment = os.environ.copy()
    for name in unset_environment:
        child_environment.pop(name, None)
    child_environment.update(environment_overrides or {})
    result = run_managed_process(
        command,
        cwd=cwd,
        log_path=log_path,
        timeout_seconds=timeout,
        environment=child_environment,
    )
    result.update(
        {
            "environment_overrides": dict(
                sorted((environment_overrides or {}).items())
            ),
            "unset_environment": sorted(unset_environment),
        }
    )
    return result


def _materialize_seed_evidence(
    *, raw_output: Path, task_output: Path, seed: int, record: dict[str, Any]
) -> None:
    destination = task_output / f"seed_{seed}"
    destination.mkdir(parents=True, exist_ok=False)
    native_seed = raw_output / f"seed_{seed}"
    if native_seed.is_dir():
        for source in native_seed.iterdir():
            if source.is_file():
                shutil.copy2(source, destination / source.name)
    for source_name, destination_name in (
        ("diagnostic_run.json", "process_run.json"),
        ("exception.json", "process_exception.json"),
        ("simulation_app_close_exception.json", "process_close_exception.json"),
    ):
        source = raw_output / source_name
        if source.is_file():
            shutil.copy2(source, destination / destination_name)
    write_json(destination / "result.json", record)
    for manifest_name in ("runtime_manifest.json", "asset_manifest.json"):
        source = raw_output / manifest_name
        destination_manifest = task_output / manifest_name
        if source.is_file() and not destination_manifest.exists():
            shutil.copy2(source, destination_manifest)


def _diagnostic_command(
    *, config_path: Path, source_root: Path, asset_root: Path, output: Path,
    task: str, seeds: Iterable[int], launcher_mode: str
) -> list[str]:
    return [
        sys.executable,
        str(REPO_ROOT / "scripts/univtac/diagnose_reset.py"),
        "--task",
        task,
        "--seeds",
        ",".join(str(seed) for seed in seeds),
        "--task-mode",
        "eval",
        "--config-profile",
        "official-demo",
        "--config",
        str(config_path),
        "--source-root",
        str(source_root),
        "--asset-root",
        str(asset_root),
        "--output-root",
        str(output),
        "--launcher-mode",
        launcher_mode,
    ]


def _redact_text(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if SECRET_PATTERN.search(line):
            lines.append("[redacted credential-shaped line]")
        else:
            lines.append(line)
    return "\n".join(lines)


def validate_author_bundle(bundle: Path) -> None:
    allowed = {".md", ".sh", ".json", ".txt"}
    for path in bundle.iterdir():
        if not path.is_file() or path.suffix not in allowed:
            raise ValueError(f"author bundle contains a disallowed file: {path.name}")
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError(f"author bundle file is too large: {path.name}")
        text = path.read_text(encoding="utf-8")
        if SECRET_PATTERN.search(text):
            raise ValueError(f"author bundle contains credential-shaped text: {path.name}")


def _create_author_bundle(
    *, output_root: Path, audit: dict[str, Any], source_manifest: dict[str, Any],
    comparison: dict[str, Any], runs: list[dict[str, Any]], config: dict[str, Any]
) -> None:
    bundle = output_root / "author_bundle"
    bundle.mkdir(parents=True, exist_ok=False)
    official = comparison["tasks"]
    failed = [
        record
        for task in PRIMARY_TASKS
        for record in official[task]["seeds"].values()
        if not reset_is_valid(record)
    ]
    first_with_call = next((record for record in failed if record["first_failed_call"]), None)
    runtime_source = next(
        (
            output_root / "headless_parity" / task / "runtime_manifest.json"
            for task in PRIMARY_TASKS
            if (output_root / "headless_parity" / task / "runtime_manifest.json").is_file()
        ),
        None,
    )
    runtime = _read_json(runtime_source) if runtime_source else {}
    write_json(bundle / "runtime_manifest.json", runtime)
    write_json(bundle / "source_manifest.json", source_manifest)
    write_json(bundle / "task_seed_results.json", {"tasks": official})
    write_json(bundle / "failed_call_summary.json", {"failed_runs": failed})

    call_input: dict[str, Any] = {"available": False}
    if first_with_call:
        task = first_with_call["task"]
        seed = first_with_call["seed"]
        call = _failed_planner_call(
            output_root / "headless_parity" / task / f"seed_{seed}"
        )
        call_input = {
            "available": True,
            "task": task,
            "seed": seed,
            "planning_call_index": call.get("planning_call_index"),
            "current_ee_pose": call.get("current_ee_pose"),
            "target_ee_pose": call.get("target_ee_pose"),
            "current_arm_joint_position": call.get("current_arm_joint_position"),
            "current_arm_joint_velocity": call.get("current_arm_joint_velocity"),
            "pre_dis": call.get("pre_dis"),
            "constraint_pose": call.get("constraint_pose"),
            "time_dilation_factor": call.get("time_dilation_factor"),
        }
    write_json(bundle / "call4_minimal_input.json", call_input)

    exact = comparison["exact_startup_sentinel"]
    _write_text(
        bundle / "README.md",
        "\n".join(
            [
                "# FTP-1 UniVTAC reset reproduction",
                f"- Official commit: `{CANONICAL_COMMIT}`.",
                f"- Runtime: `{runtime.get('python', {}).get('version', 'see runtime_manifest.json')}`.",
                f"- Tasks: {', '.join(PRIMARY_TASKS)}; seeds: {list(FIXED_PROBE_SEEDS)}.",
                f"- Exact launcher reached reset: `{exact.get('reached_reset')}`.",
                "- Headless parity results are in `task_seed_results.json`.",
                f"- First planner failure: `{first_with_call.get('first_failed_call') if first_with_call else None}`.",
                "- Planner, task, constraints, poses, thresholds, and retries were not changed.",
                "- Reproduce with `bash minimal_commands.sh`.",
                "- Please confirm the intended runtime/patch set and whether these seeds should reset on the public release.",
            ]
        ),
    )
    _write_text(
        bundle / "protocol_summary.md",
        render_markdown(audit),
    )
    _write_text(
        bundle / "minimal_commands.sh",
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd {REPO_ROOT}",
                f"{sys.executable} scripts/univtac/probe_ftp1_eval_seeds.py \\",
                f"  --config {Path(config['_config_path'])} \\",
                "  --output-root outputs/ftp1-eval-seed-probe-reproduction",
            ]
        ),
    )
    tails: list[str] = []
    for run in runs:
        path = Path(run["log_path"])
        if path.is_file():
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            tails.append(f"## {path.name}\n" + "\n".join(lines[-100:]))
    _write_text(bundle / "relevant_log_tail.txt", _redact_text("\n\n".join(tails)))
    os.chmod(bundle / "minimal_commands.sh", 0o755)
    validate_author_bundle(bundle)


def run_probe(args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("probe config must be a mapping")
    config["_config_path"] = str(config_path)
    validate_probe_config(config)
    output_root = Path(args.output_root).expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    checkout = Path(config["canonical_checkout"]).expanduser().resolve()
    source_root = Path(config["task_source_root"]).expanduser().resolve()
    asset_root = Path(config["asset_root"]).expanduser().resolve()
    if _git_head(checkout) != CANONICAL_COMMIT:
        raise RuntimeError("canonical FTP-1 checkout is not at the frozen commit")
    if source_root != checkout / "UniVTAC":
        raise RuntimeError("task_source_root is not the canonical checkout's UniVTAC")

    audit, source_manifest = audit_protocol(checkout)
    audit_root = output_root / "protocol_audit"
    write_json(audit_root / "ftp1_protocol_audit.json", audit)
    write_json(audit_root / "source_manifest.json", source_manifest)
    _write_text(audit_root / "ftp1_protocol_audit.md", render_markdown(audit))

    launcher_audit = {
        "exact_ftp1": {
            "enable_cameras": True,
            "livestream": 2,
            "headless": False,
            "render_frequency": 0,
            "video_frequency": 2,
        },
        "headless_parity_override": {
            "HEADLESS": 1,
            "livestream": 0,
            "render_frequency": 1,
            "all_other_contact_yml_semantics_unchanged": True,
        },
    }
    run_manifest: dict[str, Any] = {
        "schema_version": "openeta.ftp1_eval_seed_probe_run.v1",
        "started_at": _utc_now(),
        "ended_at": None,
        "status": "running",
        "canonical_commit": CANONICAL_COMMIT,
        "canonical_checkout": str(checkout),
        "task_source_root": str(source_root),
        "asset_root": str(asset_root),
        "tasks": list(PRIMARY_TASKS),
        "seeds": list(FIXED_PROBE_SEEDS),
        **PROBE_IMPORT_FLAGS,
        "launcher_audit": launcher_audit,
        "runs": [],
    }
    write_json(output_root / "run_manifest.json", run_manifest)

    exact_output = output_root / "exact_startup_sentinel/insert_hole"
    exact_log = output_root / ".logs/exact_insert_hole_seed_1000000.log"
    exact_command = _diagnostic_command(
        config_path=config_path,
        source_root=source_root,
        asset_root=asset_root,
        output=exact_output,
        task="insert_hole",
        seeds=(FIXED_PROBE_SEEDS[0],),
        launcher_mode="ftp1-exact",
    )
    exact_run = _run_subprocess(
        exact_command,
        cwd=REPO_ROOT,
        log_path=exact_log,
        timeout=int(config["exact_startup_timeout_seconds"]),
        environment_overrides={"ACCEPT_EULA": "Y", "CUDA_VISIBLE_DEVICES": "0"},
    )
    run_manifest["runs"].append(exact_run)
    exact_record = normalize_seed_result(
        task="insert_hole",
        seed=FIXED_PROBE_SEEDS[0],
        task_root=source_root,
        task_output=exact_output,
        launcher_mode=LAUNCHER_EXACT,
        process_run=exact_run,
    )
    exact_seed_dir = exact_output / f"seed_{FIXED_PROBE_SEEDS[0]}"
    exact_seed_dir.mkdir(parents=True, exist_ok=True)
    write_json(exact_seed_dir / "result.json", exact_record)
    if (exact_output / "diagnostic_run.json").is_file():
        shutil.copy2(
            exact_output / "diagnostic_run.json", exact_seed_dir / "process_run.json"
        )
    write_json(exact_output / "summary.json", {"seed_1000000": exact_record})

    task_records: dict[str, list[dict[str, Any]]] = {}
    for task in PRIMARY_TASKS:
        task_output = output_root / "headless_parity" / task
        task_output.mkdir(parents=True, exist_ok=False)
        records: list[dict[str, Any]] = []
        for seed in FIXED_PROBE_SEEDS:
            raw_output = output_root / ".raw/headless_parity" / task / f"seed_{seed}"
            log_path = output_root / ".logs" / f"headless_{task}_seed_{seed}.log"
            command = _diagnostic_command(
                config_path=config_path,
                source_root=source_root,
                asset_root=asset_root,
                output=raw_output,
                task=task,
                seeds=(seed,),
                launcher_mode="ftp1-protocol-headless-parity",
            )
            run = _run_subprocess(
                command,
                cwd=REPO_ROOT,
                log_path=log_path,
                timeout=int(config["task_timeout_seconds"]),
                environment_overrides={
                    "ACCEPT_EULA": "Y",
                    "CUDA_VISIBLE_DEVICES": "0",
                    "HEADLESS": "1",
                    "LIVESTREAM": "0",
                },
            )
            run_manifest["runs"].append(run)
            record = normalize_seed_result(
                task=task,
                seed=seed,
                task_root=source_root,
                task_output=raw_output,
                launcher_mode=LAUNCHER_PARITY,
                process_run=run,
            )
            records.append(record)
            _materialize_seed_evidence(
                raw_output=raw_output,
                task_output=task_output,
                seed=seed,
                record=record,
            )
            write_json(output_root / "run_manifest.json", run_manifest)
        task_records[task] = records
        write_json(task_output / "summary.json", classify_task(records))

    lift_triggered = should_run_conditional_control(
        record for records in task_records.values() for record in records
    )
    lift_records: list[dict[str, Any]] = []
    if lift_triggered:
        task = CONDITIONAL_CONTROL_TASK
        task_output = output_root / "headless_parity" / task
        task_output.mkdir(parents=True, exist_ok=False)
        for seed in FIXED_PROBE_SEEDS:
            raw_output = output_root / ".raw/headless_parity" / task / f"seed_{seed}"
            log_path = output_root / ".logs" / f"headless_{task}_seed_{seed}.log"
            command = _diagnostic_command(
                config_path=config_path,
                source_root=source_root,
                asset_root=asset_root,
                output=raw_output,
                task=task,
                seeds=(seed,),
                launcher_mode="ftp1-protocol-headless-parity",
            )
            run = _run_subprocess(
                command,
                cwd=REPO_ROOT,
                log_path=log_path,
                timeout=int(config["task_timeout_seconds"]),
                environment_overrides={
                    "ACCEPT_EULA": "Y",
                    "CUDA_VISIBLE_DEVICES": "0",
                    "HEADLESS": "1",
                    "LIVESTREAM": "0",
                },
            )
            run_manifest["runs"].append(run)
            record = normalize_seed_result(
                task=task,
                seed=seed,
                task_root=source_root,
                task_output=raw_output,
                launcher_mode=LAUNCHER_PARITY,
                process_run=run,
            )
            lift_records.append(record)
            _materialize_seed_evidence(
                raw_output=raw_output,
                task_output=task_output,
                seed=seed,
                record=record,
            )
        write_json(task_output / "summary.json", classify_task(lift_records))

    comparison: dict[str, Any] = {
        "schema_version": "openeta.ftp1_eval_seed_probe_comparison.v1",
        "exact_startup_sentinel": exact_record,
        "tasks": {task: classify_task(records) for task, records in task_records.items()},
        "conditional_lift_can": {
            "triggered": lift_triggered,
            "result": classify_task(lift_records) if lift_records else None,
        },
        "smoke": {"triggered": False, "reason": None, "result_path": None},
    }

    insert_valid = all(reset_is_valid(record) for record in task_records["insert_hole"])
    if insert_valid:
        selected_config = output_root / "insert_hole_smoke_ftp1_seeds.yaml"
        smoke_config = {
            "task": "insert_hole",
            "seeds": list(FIXED_PROBE_SEEDS),
            "probe_delta_mm": [0.0, 0.0, -2.0],
            "probe_time_dilation_factor": 0.1,
            "save_host_only": True,
            "strict_two_tactile_sensors": True,
            "fail_on_missing_rgb_marker": True,
            "vendor_task_config": "task_config/contact.yml",
            "runtime_render_frequency_override": 1,
        }
        selected_config.write_text(
            yaml.safe_dump(smoke_config, sort_keys=False), encoding="utf-8"
        )
        smoke_output = output_root / "insert_hole_smoke"
        smoke_log = output_root / ".logs/insert_hole_smoke.log"
        command = [
            sys.executable,
            str(REPO_ROOT / "scripts/univtac/smoke_insert_hole.py"),
            "--config",
            str(selected_config),
            "--output-root",
            str(smoke_output),
            "--source-root",
            str(source_root),
            "--asset-root",
            str(asset_root),
            "--headless",
            "--livestream",
            "0",
        ]
        run = _run_subprocess(
            command,
            cwd=REPO_ROOT,
            log_path=smoke_log,
            timeout=int(config["task_timeout_seconds"]),
            environment_overrides={
                "ACCEPT_EULA": "Y",
                "CUDA_VISIBLE_DEVICES": "0",
                "HEADLESS": "1",
                "LIVESTREAM": "0",
            },
        )
        run_manifest["runs"].append(run)
        comparison["smoke"] = {
            "triggered": True,
            "reason": "all three Insert Hole FTP-1 seeds were reset-valid",
            "result_path": str(smoke_output / "summary.json"),
            "returncode": run["returncode"],
        }
    else:
        comparison["smoke"]["reason"] = (
            "Insert Hole did not have three reset-valid FTP-1 seeds"
        )

    write_json(output_root / "comparison.json", comparison)
    author_triggered = any(
        all(not reset_is_valid(record) for record in task_records[task])
        for task in ("insert_hole", "insert_tube")
    )
    if author_triggered:
        _create_author_bundle(
            output_root=output_root,
            audit=audit,
            source_manifest=source_manifest,
            comparison=comparison,
            runs=run_manifest["runs"],
            config=config,
        )

    missing_required_diagnostics = any(
        record.get("failure_stage") == "startup_before_reset"
        for records in task_records.values()
        for record in records
    )
    native_runtime_errors = any(
        record.get("failure_stage") == "reset_runtime_error"
        for records in task_records.values()
        for record in records
    )
    timed_out = any(run.get("timed_out") for run in run_manifest["runs"])
    smoke_failed = bool(
        comparison["smoke"]["triggered"]
        and comparison["smoke"].get("returncode") != 0
    )
    orchestration_failed = (
        missing_required_diagnostics
        or native_runtime_errors
        or timed_out
        or smoke_failed
    )
    run_manifest["ended_at"] = _utc_now()
    run_manifest["status"] = "failed" if orchestration_failed else "complete"
    run_manifest["conditional_lift_can_triggered"] = lift_triggered
    run_manifest["smoke_triggered"] = insert_valid
    run_manifest["author_bundle_triggered"] = author_triggered
    write_json(output_root / "run_manifest.json", run_manifest)
    return 1 if orchestration_failed else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    return run_probe(args)


if __name__ == "__main__":
    raise SystemExit(main())
