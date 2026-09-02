#!/usr/bin/env python3
"""Run R0.4 CUDA/UIPC sentinels and gated clean-reset diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.univtac.probe_ftp1_eval_seeds import (
    LAUNCHER_EXACT,
    LAUNCHER_PARITY,
    _diagnostic_command,
    _materialize_seed_evidence,
    _run_subprocess,
    normalize_seed_result,
    reset_is_valid,
)
from sim.envs.univtac.cuda_device_diagnostics import (
    FIXED_FTP1_SEEDS,
    classify_uipc_runs,
    compare_cuda_modes,
    gate_followups,
    should_continue_after_gate,
)
from sim.envs.univtac.resource_sanitation import (
    build_restore_ready,
    collect_gpu_inventory,
    collect_inotify_inventory,
    collect_process_inventory,
    decide_inotify_change,
    read_sysctl,
    run_managed_process,
    run_noninteractive_sysctl,
    safe_device_environment,
    utc_now,
    write_json,
)


DEFAULT_CONFIG = REPO_ROOT / "configs/univtac/runtime_sanitation_probe.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/univtac-runtime-r04"
TASK_ORDER = ("lift_can", "pull_out_key", "insert_hole", "insert_tube")


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


def _validate_external_inotify(
    inventory: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    minimums = config.get("external_inotify_minimums", {})
    current_instances = int(inventory.get("max_user_instances") or 0)
    current_watches = int(inventory.get("max_user_watches") or 0)
    required_instances = int(minimums.get("max_user_instances", 1024))
    required_watches = int(minimums.get("max_user_watches", 524288))
    instance_ratio = inventory.get("instance_usage_ratio")
    watch_ratio = inventory.get("watch_usage_ratio")
    failures: list[str] = []
    if current_instances < required_instances:
        failures.append("max_user_instances_below_external_minimum")
    if current_watches < required_watches:
        failures.append("max_user_watches_below_external_minimum")
    if instance_ratio is None or float(instance_ratio) >= 0.75:
        failures.append("inotify_instance_usage_at_or_above_75_percent")
    if watch_ratio is None or float(watch_ratio) >= 0.75:
        failures.append("inotify_watch_usage_at_or_above_75_percent")
    return {
        "satisfied": not failures,
        "failures": failures,
        "required": {
            "max_user_instances": required_instances,
            "max_user_watches": required_watches,
        },
        "observed": {
            "max_user_instances": current_instances,
            "max_user_watches": current_watches,
            "instance_usage_ratio": instance_ratio,
            "watch_usage_ratio": watch_ratio,
        },
    }


def _residual_group_members(*payloads: Mapping[str, Any]) -> list[int]:
    residual: set[int] = set()

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            members = value.get("final_process_group_members")
            if isinstance(members, list):
                residual.update(int(pid) for pid in members)
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    for payload in payloads:
        visit(payload)
    return sorted(residual)


def _load_config(path: Path) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError("runtime sanitation config must be a mapping")
    if tuple(config.get("seeds", ())) != FIXED_FTP1_SEEDS:
        raise ValueError(f"seeds must be exactly {list(FIXED_FTP1_SEEDS)}")
    canonical = Path(config["canonical_checkout"]).expanduser().resolve()
    if _git_head(canonical) != config["canonical_commit"]:
        raise RuntimeError("canonical FTP-1 checkout commit changed")
    if Path(config["canonical_task_root"]).expanduser().resolve() != canonical / "UniVTAC":
        raise RuntimeError("canonical task root does not belong to the frozen checkout")
    return config


def _base_environment(overrides: Mapping[str, str], *, unset_visible: bool = False) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "ACCEPT_EULA": "Y",
            "HEADLESS": "1",
            "LIVESTREAM": "0",
        }
    )
    if unset_visible:
        environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment.update({str(key): str(value) for key, value in overrides.items()})
    return environment


def _run_cuda_mode(
    *, mode: str, overrides: Mapping[str, str], config: Mapping[str, Any], output_root: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    directory = output_root / "cuda_device" / mode
    result_path = directory / "result.json"
    command = [
        str(config["python_executable"]),
        str(REPO_ROOT / "scripts/univtac/probe_uipc_device.py"),
        "--probe",
        "cuda",
        "--mode-label",
        mode,
        "--output",
        str(result_path),
    ]
    run = run_managed_process(
        command,
        cwd=REPO_ROOT,
        log_path=directory / "console.log",
        timeout_seconds=120,
        environment=_base_environment(overrides, unset_visible=mode.startswith("d1_")),
        cleanup_wait_seconds=float(config["cleanup_wait_seconds"]),
    )
    result = _read_json(result_path) if result_path.is_file() else {
        "mode": mode,
        "success": False,
        "error": "result_missing",
    }
    write_json(directory / "process_run.json", run)
    return result, run


def _run_uipc(
    *, label: str, mode: str, overrides: Mapping[str, str], config: Mapping[str, Any],
    output_root: Path
) -> dict[str, Any]:
    directory = output_root / "uipc_sentinel" / label
    result_path = directory / "result.json"
    stage_path = directory / "stage.json"
    for stale_path in (result_path, stage_path):
        if stale_path.is_file():
            stale_path.unlink()
    command = [
        str(config["python_executable"]),
        str(REPO_ROOT / "scripts/univtac/probe_uipc_device.py"),
        "--probe",
        "uipc",
        "--mode-label",
        mode,
        "--canonical-task-root",
        str(config["canonical_task_root"]),
        "--output",
        str(result_path),
    ]
    run = run_managed_process(
        command,
        cwd=REPO_ROOT,
        log_path=directory / "console.log",
        timeout_seconds=float(config["uipc_timeout_seconds"]),
        environment=_base_environment(overrides, unset_visible=mode.startswith("d1_")),
        cleanup_wait_seconds=float(config["cleanup_wait_seconds"]),
    )
    log_text = (
        (directory / "console.log").read_text(encoding="utf-8", errors="replace")
        if (directory / "console.log").is_file()
        else ""
    )
    native = _read_json(result_path) if result_path.is_file() else {}
    stage = _read_json(stage_path) if stage_path.is_file() else {}
    record = {
        **run,
        "label": label,
        "mode": mode,
        "completed_step": native.get("completed_step") is True,
        "completed_steps": native.get("completed_steps", 0),
        "invalid_device": "cudaErrorInvalidDevice" in log_text,
        "cuda_malloc_async_failure": "cudaMallocAsync" in log_text,
        "native_result": native,
        "native_result_missing": not result_path.is_file(),
        "stage": stage.get("stage"),
        "stage_evidence": stage,
    }
    write_json(directory / "process_run.json", record)
    return record


def _new_gate_summary() -> dict[str, Any]:
    return {
        "schema_version": "openeta.univtac.clean_reset_gates.v1",
        "tasks": {
            task: {
                str(seed): {
                    "status": "not_run_due_to_gate",
                    "reason": "gate_not_reached",
                }
                for seed in (
                    (FIXED_FTP1_SEEDS[0],)
                    if task == "lift_can"
                    else FIXED_FTP1_SEEDS
                )
            }
            for task in TASK_ORDER
        },
        "exact_launcher": {"status": "not_run_due_to_gate", "reason": "gate_not_reached"},
    }


def _run_reset(
    *, task: str, seed: int, config_path: Path, config: Mapping[str, Any],
    output_root: Path, launcher_mode: str = "ftp1-protocol-headless-parity"
) -> dict[str, Any]:
    task_output = output_root / "reset_gates" / task
    task_output.mkdir(parents=True, exist_ok=True)
    raw = output_root / ".raw/reset_gates" / task / f"seed_{seed}"
    log = output_root / ".logs" / f"reset_{task}_seed_{seed}.log"
    command = _diagnostic_command(
        config_path=Path(config["diagnostic_config_resolved"]),
        source_root=Path(config["canonical_task_root"]),
        asset_root=Path(config["asset_root"]),
        output=raw,
        task=task,
        seeds=(seed,),
        launcher_mode=launcher_mode,
    )
    mode = config["selected_cuda_mode"]
    overrides = config["cuda_modes"][mode]
    run = _run_subprocess(
        command,
        cwd=REPO_ROOT,
        log_path=log,
        timeout=int(config["reset_timeout_seconds"]),
        environment_overrides={
            **{str(key): str(value) for key, value in overrides.items()},
            "ACCEPT_EULA": "Y",
            "HEADLESS": "0" if launcher_mode == "ftp1-exact" else "1",
            "LIVESTREAM": "2" if launcher_mode == "ftp1-exact" else "0",
        },
        unset_environment=("CUDA_VISIBLE_DEVICES",) if mode.startswith("d1_") else (),
    )
    record = normalize_seed_result(
        task=task,
        seed=seed,
        task_root=Path(config["canonical_task_root"]),
        task_output=raw,
        launcher_mode=LAUNCHER_EXACT if launcher_mode == "ftp1-exact" else LAUNCHER_PARITY,
        process_run=run,
    )
    record["process_cleanup_complete"] = run.get("cleanup_complete") is True
    record["new_gpu_pids_after"] = run.get("new_gpu_pids_after", [])
    record["inotify_instance_delta"] = run.get("inotify_instance_delta")
    record["process_group_id"] = run.get("process_group_id")
    record["final_process_group_members"] = run.get(
        "final_process_group_members", []
    )
    _materialize_seed_evidence(
        raw_output=raw, task_output=task_output, seed=seed, record=record
    )
    return record


def _collect_logs(output_root: Path) -> str:
    sections: list[str] = []
    for path in sorted(output_root.rglob("*.log")):
        if "author_bundle_v2" in path.parts:
            continue
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        sections.append(f"## {path.relative_to(output_root)}\n" + "\n".join(lines[-80:]))
    return "\n\n".join(sections)


def _create_author_bundle(
    *, output_root: Path, classification: str, system_changes: Mapping[str, Any],
    cuda_comparison: Mapping[str, Any], uipc_summary: Mapping[str, Any],
    gate_summary: Mapping[str, Any], clean_runtime: Mapping[str, Any],
    bundle_name: str = "author_bundle_v2",
) -> None:
    bundle = output_root / bundle_name
    bundle.mkdir(parents=True, exist_ok=True)
    r03_protocol = REPO_ROOT / "outputs/ftp1-eval-seed-probe/author_bundle/protocol_summary.md"
    if r03_protocol.is_file():
        shutil.copy2(r03_protocol, bundle / "protocol_summary.md")
    else:
        _write_text(bundle / "protocol_summary.md", "# Protocol summary\nR0.3 protocol audit unavailable.")
    pre = _read_json(output_root / "pre_cleanup/process_inventory.json")
    plan = _read_json(output_root / "cleanup/cleanup_plan.json")
    post = _read_json(output_root / "cleanup/post_cleanup_process_inventory.json")
    actions_path = output_root / "cleanup/cleanup_actions.jsonl"
    actions = [
        json.loads(line)
        for line in actions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if actions_path.is_file() else []
    def process_counts(inventory: Mapping[str, Any]) -> dict[str, int]:
        processes = inventory.get("processes", [])
        return {
            "related": len(processes),
            "eligible": sum(
                1 for item in processes if item.get("eligible_for_cleanup")
            ),
        }

    cleanup = {
        "pre_counts": process_counts(pre),
        "plan_counts": plan.get("counts", {}),
        "terminated_targets": [
            {"pid": entry.get("pid"), "start_ticks": entry.get("start_ticks")}
            for entry in plan.get("entries", [])
            if entry.get("action") == "terminate"
        ],
        "actions": actions,
        "post_counts": process_counts(post),
        "remaining_eligible_pids": [
            item.get("pid")
            for item in post.get("processes", [])
            if item.get("eligible_for_cleanup")
        ],
    }
    write_json(bundle / "process_cleanup_summary.json", cleanup)
    def public_inotify(inventory: Mapping[str, Any] | None) -> dict[str, Any]:
        inventory = inventory or {}
        return {
            key: inventory.get(key)
            for key in (
                "captured_at",
                "instance_count",
                "instance_usage_ratio",
                "max_user_instances",
                "max_user_watches",
                "max_queued_events",
                "watch_count",
                "watch_count_reliable",
            )
        }

    public_system_changes = dict(system_changes.get("inotify", {}))
    public_system_changes["cleanup_usage"] = public_inotify(
        public_system_changes.get("cleanup_usage")
    )
    write_json(bundle / "inotify_summary.json", public_system_changes)
    write_json(bundle / "cuda_device_matrix.json", cuda_comparison)
    write_json(bundle / "uipc_sentinel_results.json", uipc_summary)
    write_json(bundle / "reset_gate_results.json", gate_summary)
    public_runtime = dict(clean_runtime)
    public_runtime["final_inotify"] = public_inotify(clean_runtime.get("final_inotify"))
    final_gpu = clean_runtime.get("final_gpu", {})
    public_runtime["final_gpu"] = {
        "captured_at": final_gpu.get("captured_at"),
        "gpus": final_gpu.get("gpus", []),
        "compute_process_count": len(final_gpu.get("compute_processes", [])),
        "compute_memory_mib": sum(
            int(item.get("gpu_memory_mib") or 0)
            for item in final_gpu.get("compute_processes", [])
        ),
    }
    write_json(bundle / "runtime_manifest_clean.json", public_runtime)
    _write_text(
        bundle / "clean_runtime_summary.md",
        "\n".join(
            [
                "# Clean runtime summary",
                f"- Classification: `{classification}`.",
                f"- Cleanup complete: `{clean_runtime.get('cleanup_complete')}`.",
                f"- UIPC consecutive two-pass: `{uipc_summary.get('consecutive_two_pass')}`.",
                f"- Planner reached: `{clean_runtime.get('planner_reached')}`.",
            ]
        ),
    )
    _write_text(
        bundle / "README.md",
        "\n".join(
            [
                "# UniVTAC R0.4 runtime reproduction",
                "- R0.3 observed 74 visible Isaac/Kit-related processes and 127/128 inotify instances.",
                f"- R0.4 cleanup complete: `{clean_runtime.get('cleanup_complete')}`.",
                f"- Temporary inotify change applied: `{system_changes.get('inotify', {}).get('applied_success')}`.",
                f"- Clean-state UIPC sentinel classification: `{uipc_summary.get('classification')}`.",
                f"- Clean-state planner reached: `{clean_runtime.get('planner_reached')}`.",
                "- This bundle does not claim an upstream bug beyond the recorded evidence.",
            ]
        ),
    )
    _write_text(
        bundle / "minimal_commands.sh",
        "\n".join(
            [
                "#!/usr/bin/env bash",
                "set -euo pipefail",
                f"cd '{REPO_ROOT}'",
                "uv run --frozen --extra dev python -m pytest -q tests/univtac",
                "# Review runtime_sanitation_probe.yaml, then run the sanitation and gate scripts.",
            ]
        ),
    )
    os.chmod(bundle / "minimal_commands.sh", 0o755)
    _write_text(bundle / "relevant_log_tail.txt", _collect_logs(output_root))
    write_json(
        bundle / "system_restore_verification.json",
        {
            "original": system_changes.get("inotify", {}).get("original"),
            "final": system_changes.get("inotify", {}).get("final"),
            "restore_required": system_changes.get("inotify", {}).get("restore_required"),
            "restore_success": system_changes.get("inotify", {}).get("restore_success"),
        },
    )
    _write_text(
        bundle / "suggested_author_message.md",
        "\n".join(
            [
                "# Suggested message to UniVTAC / FTP-1 authors",
                "We reproduced the public reset path at the frozen FTP-1 commit and are attaching a minimal clean-runtime evidence bundle.",
                f"R0.4 classification: `{classification}`.",
                "No task, planner, constraint, gripper, TacEx, cuRobo, or Isaac source was modified.",
                "Could you confirm the runtime/inotify prerequisites and the exact dependency environment used for the public UniVTAC evaluation?",
            ]
        ),
    )


def run(args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    config = _load_config(config_path)
    config["diagnostic_config_resolved"] = str(
        (REPO_ROOT / str(config["diagnostic_config"])).resolve()
    )
    cleanup_manifest = output_root / "run_manifest.json"
    post_process = output_root / "cleanup/post_cleanup_process_inventory.json"
    post_gpu = output_root / "cleanup/post_cleanup_gpu_inventory.json"
    post_inotify = output_root / "cleanup/post_cleanup_inotify_inventory.json"
    for required in (cleanup_manifest, post_process, post_gpu, post_inotify):
        if not required.is_file():
            raise FileNotFoundError(f"completed sanitation evidence is missing: {required}")

    manifest = _read_json(cleanup_manifest)
    manifest.pop("diagnostic_error", None)
    manifest.update(
        {
            "r04_started_at": utc_now(),
            "status": "running_diagnostics",
            "git_head": _git_head(REPO_ROOT),
            "canonical_commit": config["canonical_commit"],
            "ftp1_model_loaded": False,
            "openeta_imported": False,
        }
    )
    write_json(cleanup_manifest, manifest)
    system_changes: dict[str, Any] = {
        "inotify": {
            "management": config.get("inotify_management", "automatic"),
            "original": {
                "max_user_instances": read_sysctl("fs.inotify.max_user_instances"),
                "max_user_watches": read_sysctl("fs.inotify.max_user_watches"),
            },
            "cleanup_usage": _read_json(post_inotify),
            "change_required": False,
            "applied": {},
            "applied_success": False,
            "restore_required": False,
            "restored": {},
            "restore_success": None,
        }
    }
    write_json(output_root / "system_changes.json", system_changes)
    classification = "blocked_by_external_resources"
    cuda_comparison: dict[str, Any] = {"status": "not_run_due_to_gate"}
    uipc_summary: dict[str, Any] = {
        "status": "not_run_due_to_gate",
        "consecutive_two_pass": False,
    }
    gate_summary = _new_gate_summary()
    planner_reached = False
    cleanup_complete = True
    diagnostic_error: dict[str, str] | None = None
    temporary_instances_applied = False
    try:
        remaining = [
            item
            for item in _read_json(post_process)["processes"]
            if item.get("eligible_for_cleanup")
        ]
        gpu = _read_json(post_gpu)
        gpu_free_ratio = min(
            (
                item["memory_free_mib"] / item["memory_total_mib"]
                for item in gpu.get("gpus", [])
                if item.get("memory_total_mib")
            ),
            default=0.0,
        )
        if remaining:
            raise RuntimeError(f"eligible stale processes remain: {[item['pid'] for item in remaining]}")
        if gpu_free_ratio < float(config["gpu_free_ratio_required"]):
            raise RuntimeError(
                f"GPU free ratio {gpu_free_ratio:.3f} is below required threshold"
            )

        inotify_management = config.get("inotify_management", "automatic")
        cleanup_inotify = _read_json(post_inotify)
        if inotify_management == "external":
            external = _validate_external_inotify(cleanup_inotify, config)
            system_changes["inotify"]["external_precondition"] = external
            system_changes["inotify"]["change_required"] = False
            system_changes["inotify"]["trigger_reason"] = "externally_managed"
            system_changes["inotify"]["restore_success"] = None
            if not external["satisfied"]:
                raise RuntimeError(
                    "manual_inotify_precondition_missing: "
                    + ", ".join(external["failures"])
                )
        else:
            decision = decide_inotify_change(cleanup_inotify)
            system_changes["inotify"]["change_required"] = decision[
                "instances_change_required"
            ]
            system_changes["inotify"]["trigger_reason"] = decision["reason"]
            if decision["instances_change_required"]:
                applied = run_noninteractive_sysctl(
                    "fs.inotify.max_user_instances",
                    int(config["inotify_instances_temporary_limit"]),
                )
                system_changes["inotify"]["applied"]["max_user_instances"] = applied
                system_changes["inotify"]["applied_success"] = applied["success"]
                system_changes["inotify"]["restore_required"] = applied["success"]
                temporary_instances_applied = applied["success"]
                if not applied["success"]:
                    raise RuntimeError("sysctl_change_unavailable")
        write_json(output_root / "system_changes.json", system_changes)

        cuda_results: dict[str, dict[str, Any]] = {}
        cuda_runs: dict[str, dict[str, Any]] = {}
        for mode, overrides in config["cuda_modes"].items():
            result, process_run = _run_cuda_mode(
                mode=mode,
                overrides=overrides,
                config=config,
                output_root=output_root,
            )
            cuda_results[mode] = result
            cuda_runs[mode] = process_run
            if not process_run.get("cleanup_complete"):
                cleanup_complete = False
                classification = "simulator_cleanup_leak"
                raise RuntimeError("CUDA diagnostic process cleanup leak")
        d0_name, d1_name = tuple(config["cuda_modes"])
        cuda_comparison = compare_cuda_modes(cuda_results[d0_name], cuda_results[d1_name])
        cuda_comparison["results"] = cuda_results
        cuda_comparison["process_runs"] = cuda_runs
        write_json(output_root / "cuda_device/comparison.json", cuda_comparison)
        if cuda_comparison["classification"] == "cuda_device_ordinal_mismatch":
            classification = "cuda_device_ordinal_mismatch"
            raise RuntimeError("CUDA D0/D1 map to different physical devices")
        if not cuda_comparison["d0_success"] or not cuda_comparison["d1_success"]:
            raise RuntimeError("basic CUDA diagnostic failed")

        uipc_runs: list[dict[str, Any]] = []
        for label, mode in (("d0_run_1", d0_name), ("d1_run_1", d1_name)):
            uipc_runs.append(
                _run_uipc(
                    label=label,
                    mode=mode,
                    overrides=config["cuda_modes"][mode],
                    config=config,
                    output_root=output_root,
                )
            )
            if not uipc_runs[-1].get("cleanup_complete"):
                cleanup_complete = False
                classification = "simulator_cleanup_leak"
                raise RuntimeError("UIPC sentinel cleanup leak")
        passing_modes = [
            record["mode"]
            for record in uipc_runs
            if record["returncode"] == 0
            and record["completed_step"]
            and not record["invalid_device"]
        ]
        if not passing_modes:
            uipc_summary = classify_uipc_runs(uipc_runs)
            classification = str(uipc_summary["classification"])
            write_json(output_root / "uipc_sentinel/summary.json", uipc_summary)
            raise RuntimeError("no UIPC mode passed the first sentinel")
        selected = d0_name if d0_name in passing_modes else passing_modes[0]
        config["selected_cuda_mode"] = selected
        uipc_runs.append(
            _run_uipc(
                label="selected_mode_run_2",
                mode=selected,
                overrides=config["cuda_modes"][selected],
                config=config,
                output_root=output_root,
            )
        )
        uipc_summary = classify_uipc_runs(uipc_runs)
        uipc_summary["selected_mode"] = selected
        write_json(output_root / "uipc_sentinel/summary.json", uipc_summary)
        if not uipc_summary["consecutive_two_pass"]:
            classification = (
                "persistent_uipc_invalid_device_clean_state"
                if uipc_summary["classification"]
                == "persistent_uipc_invalid_device_clean_state"
                else "blocked_by_external_resources"
            )
            raise RuntimeError("UIPC sentinel did not pass twice")

        lift = _run_reset(
            task="lift_can",
            seed=FIXED_FTP1_SEEDS[0],
            config_path=config_path,
            config=config,
            output_root=output_root,
        )
        gate_summary["tasks"]["lift_can"][str(FIXED_FTP1_SEEDS[0])] = lift
        if not lift.get("process_cleanup_complete"):
            cleanup_complete = False
            classification = "simulator_cleanup_leak"
            raise RuntimeError("Lift Can cleanup leak")
        if not should_continue_after_gate(lift, require_reset_valid=True):
            classification = "blocked_by_external_resources"
            raise RuntimeError("Lift Can did not become reset-valid")

        seed_zero_records: dict[str, dict[str, Any]] = {}
        for task in ("pull_out_key", "insert_hole", "insert_tube"):
            record = _run_reset(
                task=task,
                seed=FIXED_FTP1_SEEDS[0],
                config_path=config_path,
                config=config,
                output_root=output_root,
            )
            gate_summary["tasks"][task][str(FIXED_FTP1_SEEDS[0])] = record
            seed_zero_records[task] = record
            planner_reached = planner_reached or int(record.get("planning_call_count", 0)) > 0
            if not record.get("process_cleanup_complete"):
                cleanup_complete = False
                classification = "simulator_cleanup_leak"
                raise RuntimeError(f"{task} cleanup leak")
            if record.get("failure_stage") in {"startup_before_reset", "reset_runtime_error"}:
                classification = "blocked_by_external_resources"
                raise RuntimeError(f"{task} shared runtime failure")

        for task, seed_zero in seed_zero_records.items():
            for seed in gate_followups(seed_zero):
                record = _run_reset(
                    task=task,
                    seed=seed,
                    config_path=config_path,
                    config=config,
                    output_root=output_root,
                )
                gate_summary["tasks"][task][str(seed)] = record
                planner_reached = planner_reached or int(record.get("planning_call_count", 0)) > 0
                if not record.get("process_cleanup_complete"):
                    cleanup_complete = False
                    classification = "simulator_cleanup_leak"
                    raise RuntimeError(f"{task}/{seed} cleanup leak")

        classification = (
            "planner_reached_after_runtime_cleanup"
            if planner_reached
            else "runtime_pollution_resolved"
        )
        if (
            uipc_summary["consecutive_two_pass"]
            and reset_is_valid(lift)
            and cleanup_complete
            and collect_inotify_inventory()["instance_usage_ratio"] < 0.75
        ):
            exact = _run_reset(
                task="insert_hole",
                seed=FIXED_FTP1_SEEDS[0],
                config_path=config_path,
                config=config,
                output_root=output_root,
                launcher_mode="ftp1-exact",
            )
            gate_summary["exact_launcher"] = exact
            if not exact.get("process_cleanup_complete"):
                cleanup_complete = False
                classification = "simulator_cleanup_leak"
                raise RuntimeError("exact launcher cleanup leak")
            if not reset_is_valid(exact):
                classification = "exact_launcher_failed_clean_state"
    except Exception as exc:
        diagnostic_error = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        manifest["diagnostic_error"] = diagnostic_error
    finally:
        if temporary_instances_applied:
            original = int(system_changes["inotify"]["original"]["max_user_instances"])
            restored = run_noninteractive_sysctl("fs.inotify.max_user_instances", original)
            system_changes["inotify"]["restored"]["max_user_instances"] = restored
            restored_value = read_sysctl("fs.inotify.max_user_instances")
            system_changes["inotify"]["restored_value"] = restored_value
            system_changes["inotify"]["restore_success"] = (
                restored["success"] and restored_value == original
            )
            if not system_changes["inotify"]["restore_success"]:
                cleanup_complete = False
                classification = "cleanup_incomplete"
        elif config.get("inotify_management", "automatic") != "external":
            system_changes["inotify"]["restore_success"] = True
        system_changes["inotify"]["final"] = {
            "max_user_instances": read_sysctl("fs.inotify.max_user_instances"),
            "max_user_watches": read_sysctl("fs.inotify.max_user_watches"),
        }
        write_json(output_root / "system_changes.json", system_changes)
        write_json(output_root / "reset_gates/summary.json", gate_summary)
        final_process = collect_process_inventory(
            related_roots=[REPO_ROOT, Path(config["canonical_task_root"]), REPO_ROOT.parents[1] / "worktrees"]
        )
        final_gpu = collect_gpu_inventory()
        final_inotify = collect_inotify_inventory()
        write_json(output_root / "final_resource_state/process_inventory.json", final_process)
        write_json(output_root / "final_resource_state/gpu_inventory.json", final_gpu)
        write_json(output_root / "final_resource_state/inotify_inventory.json", final_inotify)
        manual_original = config.get("manual_restore_original", {})
        restore_ready = build_restore_ready(
            original_instances=int(manual_original.get("max_user_instances", 128)),
            original_watches=int(manual_original.get("max_user_watches", 65536)),
            inotify_inventory=final_inotify,
            process_inventory=final_process,
            gpu_inventory=final_gpu,
            process_group_residual=_residual_group_members(
                cuda_comparison, uipc_summary, gate_summary
            ),
        )
        restore_ready["original_file_warning"] = (
            "recorded_original_matches_temporary_limits; using R0.4 observed originals"
        )
        write_json(output_root / "restore_ready.json", restore_ready)
        clean_runtime = {
            "captured_at": utc_now(),
            "git_head": _git_head(REPO_ROOT),
            "classification": classification,
            "cleanup_complete": cleanup_complete,
            "planner_reached": planner_reached,
            "diagnostic_error": diagnostic_error,
            "device_environment": safe_device_environment(),
            "final_inotify": final_inotify,
            "final_gpu": final_gpu,
            "final_related_process_count": len(final_process["processes"]),
        }
        _create_author_bundle(
            output_root=output_root,
            classification=classification,
            system_changes=system_changes,
            cuda_comparison=cuda_comparison,
            uipc_summary=uipc_summary,
            gate_summary=gate_summary,
            clean_runtime=clean_runtime,
            bundle_name=(
                "author_bundle_uipc"
                if classification
                in {
                    "persistent_uipc_invalid_device_clean_state",
                    "uipc_initialization_early_exit_clean_state",
                    "uipc_no_step_clean_state",
                    "uipc_sentinel_timeout_clean_state",
                }
                else "author_bundle_planner"
                if classification == "planner_reached_after_runtime_cleanup"
                else "author_bundle_v2"
            ),
        )
        manifest.update(
            {
                "r04_ended_at": utc_now(),
                "status": (
                    "complete"
                    if cleanup_complete and diagnostic_error is None
                    else "complete_with_exact_failure"
                    if cleanup_complete and classification == "exact_launcher_failed_clean_state"
                    else "blocked"
                    if cleanup_complete
                    else "cleanup_incomplete"
                ),
                "classification": classification,
                "cleanup_complete": cleanup_complete,
                "planner_reached": planner_reached,
            }
        )
        write_json(cleanup_manifest, manifest)
    return 0 if cleanup_complete and diagnostic_error is None else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
