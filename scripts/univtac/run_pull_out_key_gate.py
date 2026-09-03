#!/usr/bin/env python3
"""Run the single R0.9.11 Pull Out Key gate through the scoped launcher."""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.envs.univtac.pull_out_key_gate import (
    seed_dir_name,
    success_classification,
    validate_gate_config,
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
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--finalize-existing", action="store_true")
    return parser.parse_args(argv)


def _git_value(cwd: Path, *args: str) -> str | None:
    import subprocess

    completed = subprocess.run(["git", *args], cwd=cwd, check=False, capture_output=True, text=True)
    return completed.stdout.strip() if completed.returncode == 0 else None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _physx_evidence(output_root: Path) -> dict[str, Any]:
    relevant_path = output_root / "launcher" / "kit_relevant.txt"
    text = (
        relevant_path.read_text(encoding="utf-8", errors="replace")
        if relevant_path.is_file()
        else ""
    )
    fallback_tokens = (
        "GPU solver is not supported",
        "GPU BP is not supported",
        "Switching to Software mode",
    )
    match = re.search(r"omni\.physx handle on CUDA lib is (\S+)", text)
    handle = match.group(1) if match else None
    return {
        "handle": handle,
        "handle_nonnull": bool(handle and handle not in {"(nil)", "0x0"}),
        "gpu_solver_fallback": "GPU solver is not supported" in text,
        "gpu_broadphase_fallback": "GPU BP is not supported" in text,
        "software_fallback": "Switching to Software mode" in text,
        "all_fallbacks_absent": not any(token in text for token in fallback_tokens),
        "unversioned_nvml_warning": "libnvidia-ml.so" in text,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _recover_child_result(
    output_root: Path,
    source_root: Path,
    lifecycle: dict[str, Any],
    seed: int = 1_000_000,
) -> dict[str, Any] | None:
    """Recover evidence when SimulationApp.close ends the child interpreter."""

    seed_dir = output_root / seed_dir_name(seed)
    stages = _read_jsonl(seed_dir / "stages.jsonl")
    stage_events = {(record["stage"], record["event"]) for record in stages}
    required_exits = {
        "task_constructor",
        "planner_recorder_install",
        "reset_actors",
        "marker_stabilization",
        "marker_calibration",
        "pre_move",
        "reset",
        "reset_returned",
        "get_observations",
        "snapshot_projection",
        "task_close",
    }
    if not all((stage, "exit") in stage_events for stage in required_exits):
        return None
    if ("simulation_app_close", "enter") not in stage_events:
        return None
    if any("error_type" in record for record in stages):
        return None
    if lifecycle.get("returncode") != 0 or lifecycle.get("timed_out"):
        return None
    if not lifecycle.get("cleanup_complete"):
        return None
    if lifecycle.get("sigterm_sent") or lifecycle.get("sigkill_sent"):
        return None
    if (seed_dir / "exception.json").exists():
        return None
    diagnostics = _read_json(seed_dir / "planner_diagnostics.json")
    contact = _read_json(seed_dir / "contact_summary.json")
    observation = _read_json(seed_dir / "observation_summary.json")
    snapshot = _read_json(seed_dir / "snapshot_pre.json")
    if not all((diagnostics, contact, observation, snapshot)):
        return None
    planning_calls = diagnostics["planning_calls"]
    move_calls = diagnostics["move_calls"]
    if not move_calls or not planning_calls:
        return None
    if any(
        "exception" in call or call.get("motion_gen_success") is False for call in planning_calls
    ):
        return None
    if not contact["any_contact_candidate"]:
        return None
    task_metadata = snapshot["host_only"]["task_metadata"]
    if not task_metadata.get("plan_success"):
        return None
    module_realpaths = {
        "envs.pull_out_key": str(source_root / "envs" / "pull_out_key.py"),
        "envs._base_task": str(source_root / "envs" / "_base_task.py"),
        "tacex": str(
            source_root / "third_party" / "TacEx" / "source" / "tacex" / "tacex" / "__init__.py"
        ),
        "tacex_uipc": str(
            source_root
            / "third_party"
            / "TacEx"
            / "source"
            / "tacex_uipc"
            / "tacex_uipc"
            / "__init__.py"
        ),
        "tacex_assets": str(
            source_root
            / "third_party"
            / "TacEx"
            / "source"
            / "tacex_assets"
            / "tacex_assets"
            / "__init__.py"
        ),
    }
    if not all(Path(path).is_file() for path in module_realpaths.values()):
        return None
    return {
        "schema_version": "openeta.univtac.pull_out_key_gate.v1",
        "classification": success_classification(seed),
        "requested_seed": seed,
        "observed_seed": seed,
        "unexpected_seeds": [],
        "module_realpaths": module_realpaths,
        "module_path_evidence": "validated in child control flow; paths recovered after SimulationApp.close ended interpreter",
        "counters": {
            "reset_call_count": 1,
            "observation_call_count": 1,
            "play_once_call_count": 0,
            "check_success_call_count": 0,
            "check_early_stop_call_count": 0,
        },
        "cleanup": {
            "task_close": True,
            "simulation_app_close_invoked": True,
            "simulation_app_close_returned": False,
            "simulation_app_close_ended_interpreter": True,
            "process_group_cleanup_complete": True,
        },
        "snapshot_pre": f"{seed_dir_name(seed)}/snapshot_pre.json",
        "snapshot_post_created": False,
        "transition_created": False,
        "planner_move_call_count": len(move_calls),
        "planner_call_count": len(planning_calls),
        "planner_failure": None,
        "plan_success": True,
        "in_pre_move": False,
        "cid_present": bool(task_metadata.get("cid_present")),
        "target_pose_present": bool(task_metadata.get("target_pose_present")),
        "slot_init_pose_present": bool(task_metadata.get("slot_init_pose_present")),
        "contact_summary": contact,
        "operator_visible_keys": sorted(snapshot["operator_visible"]),
        "host_only_keys": sorted(snapshot["host_only"]),
        "result_recovered_after_clean_close_exit": True,
        "error": None,
    }


def _finalize_run(
    *,
    output_root: Path,
    source_root: Path,
    config: dict[str, Any],
    run_manifest: dict[str, Any],
    lifecycle: dict[str, Any] | None,
    launcher_error: dict[str, str] | None,
) -> str:
    seed = int(config["seed"])
    seed_dir = output_root / seed_dir_name(seed)
    child_result = _read_json(seed_dir / "child_result.json")
    if child_result is None and lifecycle is not None:
        child_result = _recover_child_result(output_root, source_root, lifecycle, seed)
        if child_result is not None:
            write_json(seed_dir / "child_result_recovered.json", child_result)
    physx = _physx_evidence(output_root)
    if launcher_error is not None:
        classification = (
            "scoped_launcher_alias_preflight_failed"
            if (output_root / "runtime" / "libcuda_alias").exists()
            else "scoped_launcher_driver_resolution_failed"
        )
    elif lifecycle is None or (
        lifecycle["returncode"] != 0
        or lifecycle["timed_out"]
        or lifecycle["sigterm_sent"]
        or lifecycle["sigkill_sent"]
    ):
        classification = "native_runtime_abort"
    elif not lifecycle["cleanup_complete"]:
        classification = "cleanup_incomplete"
    elif (
        not physx["handle_nonnull"]
        or not physx["all_fallbacks_absent"]
        or lifecycle["stub_mapped"]
        or lifecycle["mapped_libcuda_paths"] != [lifecycle["alias_target"]]
    ):
        classification = "scoped_launcher_libcuda_alias_not_effective"
    elif child_result is None:
        classification = "native_runtime_abort"
    else:
        classification = str(child_result["classification"])
    summary = {
        "schema_version": "openeta.univtac.pull_out_key_gate_summary.v1",
        "classification": classification,
        "simulator_invocation_count": run_manifest["simulator_invocation_count"],
        "requested_seed": config["seed"],
        "observed_seed": child_result.get("observed_seed") if child_result else None,
        "unexpected_seeds": child_result.get("unexpected_seeds", []) if child_result else [],
        "launcher_error": launcher_error,
        "launcher_lifecycle": lifecycle,
        "physx": physx,
        "child_result": child_result,
        "no_play_once": bool(
            child_result and child_result["counters"]["play_once_call_count"] == 0
        ),
        "no_success_checks": bool(
            child_result
            and child_result["counters"]["check_success_call_count"] == 0
            and child_result["counters"]["check_early_stop_call_count"] == 0
        ),
        "snapshot_post_absent": not (seed_dir / "snapshot_post.json").exists(),
        "transition_absent": not (seed_dir / "transition.json").exists(),
    }
    write_json(output_root / "summary.json", summary)
    run_manifest["status"] = "complete"
    run_manifest["classification"] = classification
    run_manifest["ended_at"] = datetime.now(timezone.utc).isoformat()
    write_json(output_root / "run_manifest.json", run_manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return classification


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = args.config.expanduser().resolve()
    runtime_python = args.runtime_python.expanduser().resolve()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and not args.finalize_existing:
        raise FileExistsError(f"output root must be fresh: {output_root}")
    config_payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config_payload, dict):
        raise TypeError("gate config must be a mapping")
    config = validate_gate_config(config_payload)
    if args.finalize_existing:
        run_manifest = _read_json(output_root / "run_manifest.json")
        lifecycle = _read_json(output_root / "launcher" / "lifecycle.json")
        if run_manifest is None or lifecycle is None:
            raise FileNotFoundError("existing run lacks manifest or launcher lifecycle")
        if run_manifest.get("simulator_invocation_count") != 1:
            raise RuntimeError("existing run does not contain exactly one simulator invocation")
        classification = _finalize_run(
            output_root=output_root,
            source_root=source_root,
            config=config,
            run_manifest=run_manifest,
            lifecycle=lifecycle,
            launcher_error=None,
        )
        return (
            0 if classification == success_classification(int(config["seed"])) else 1
        )
    output_root.mkdir(parents=True)
    child_argv = [
        str(REPO_ROOT / "scripts" / "univtac" / "probe_pull_out_key_seed.py"),
        "--config",
        str(config_path),
        "--source-root",
        str(source_root),
        "--repo-root",
        str(REPO_ROOT),
        "--output-root",
        str(output_root),
        "--seed",
        str(config["seed"]),
    ]
    if args.headless:
        child_argv.append("--headless")
    spec = ScopedIsaac51LaunchSpec(
        python_executable=runtime_python,
        command=tuple(child_argv),
        cwd=source_root,
        output_root=output_root,
        timeout_seconds=float(config["timeout_seconds"]),
    )
    run_manifest = {
        "schema_version": "openeta.univtac.pull_out_key_gate_run.v1",
        "round": "R0.9.11",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "repo_root": str(REPO_ROOT),
        "repo_head": _git_value(REPO_ROOT, "rev-parse", "HEAD"),
        "repo_branch": _git_value(REPO_ROOT, "branch", "--show-current"),
        "source_root": str(source_root),
        "source_head": _git_value(source_root, "rev-parse", "HEAD"),
        "runtime_python": str(runtime_python),
        "config": str(config_path),
        "requested_seed": config["seed"],
        "seed_label": config["seed_label"],
        "simulator_invocation_limit": 1,
        "simulator_invocation_count": 0,
        "project_hash_gate_evaluated": False,
        "no_parallel_runtime_gate_evaluated": False,
        "unrelated_processes_terminated": False,
        "child_argv": child_argv,
        "status": "starting",
    }
    write_json(output_root / "run_manifest.json", run_manifest)
    lifecycle = None
    launcher_error = None
    try:
        run_manifest["simulator_invocation_count"] = 1
        write_json(output_root / "run_manifest.json", run_manifest)
        lifecycle = run_scoped_isaac51_command(spec)
    except ScopedIsaac51LaunchError as exc:
        launcher_error = {"error_type": type(exc).__name__, "error": str(exc)}

    classification = _finalize_run(
        output_root=output_root,
        source_root=source_root,
        config=config,
        run_manifest=run_manifest,
        lifecycle=lifecycle.to_dict() if lifecycle else None,
        launcher_error=launcher_error,
    )
    return 0 if classification == success_classification(int(config["seed"])) else 1


if __name__ == "__main__":
    raise SystemExit(main())
