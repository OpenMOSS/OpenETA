#!/usr/bin/env python3
"""Diagnose native UniVTAC reset/pre_move planning without changing behavior."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.envs.univtac.observation import validate_tactile_packets
from sim.envs.univtac.planner_diagnostics import PlannerDiagnosticRecorder
from sim.envs.univtac.runtime import (
    assert_task_not_imported,
    collect_asset_manifest,
    collect_runtime_manifest,
    import_task_after_launcher,
    manifest_contains_secret_keys,
)
from sim.envs.univtac.seed_eligibility import (
    ResetEligibilityInput,
    should_stop_scan,
    summarize_seed_eligibility,
)
from sim.envs.univtac.trace import write_exception, write_json


DEFAULT_CONFIG = REPO_ROOT / "configs/univtac/insert_hole_reset_diagnostic.yaml"
DEFAULT_SOURCE_ROOT = REPO_ROOT / "third_party/ftp1-policy/UniVTAC"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "outputs/univtac-reset-diagnostic/insert_hole"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_seeds(text: str) -> list[int]:
    try:
        seeds = [int(part.strip()) for part in text.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--seeds must be comma-separated integers") from exc
    if not seeds or seeds != sorted(seeds) or len(seeds) != len(set(seeds)):
        raise argparse.ArgumentTypeError("--seeds must be unique and strictly ascending")
    return seeds


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"diagnostic config must be a mapping: {path}")
    required = {
        "task_config",
        "strict_two_tactile_sensors",
        "fail_on_missing_rgb_marker",
        "required_valid_seeds",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"diagnostic config is missing keys: {missing}")
    return payload


def _load_task_profile(task_root: Path, relative_path: str) -> dict[str, Any]:
    path = (task_root / relative_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"task profile is missing: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"task profile must be a mapping: {path}")
    return payload


def _configure_task(
    *,
    task_module: Any,
    task_root: Path,
    config: dict[str, Any],
    config_profile: str,
    native_root: Path,
    device: str | None,
):
    profile = _load_task_profile(task_root, str(config["task_config"]))
    env_cfg = task_module.TaskCfg()
    env_cfg.save_dir = str(native_root)
    env_cfg.worker_name = f"reset_diagnostic_{config_profile}"
    env_cfg.scene.num_envs = 1
    env_cfg.decimation = int(profile.get("decimation", env_cfg.decimation))
    env_cfg.obs_data_type = dict(profile.get("observations", {}))
    env_cfg.random_texture = bool(profile.get("random_texture", False))
    env_cfg.tactile_sensor_type = str(profile.get("sensor_type", "gsmini"))
    if config_profile == "official-demo":
        env_cfg.save_frequency = int(
            profile.get("save_frequency", env_cfg.save_frequency)
        )
        env_cfg.video_frequency = int(
            profile.get("video_frequency", env_cfg.video_frequency)
        )
        env_cfg.render_frequency = int(
            profile.get("render_frequency", env_cfg.render_frequency)
        )
    elif config_profile == "smoke":
        env_cfg.save_frequency = 0
        env_cfg.video_frequency = 0
        env_cfg.render_frequency = 0
    else:
        raise ValueError(f"unsupported config profile: {config_profile}")
    if device:
        env_cfg.sim.device = device
    return env_cfg


def _array_metadata(value: Any) -> dict[str, Any]:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    array = np.asarray(value)
    return {"shape": list(array.shape), "dtype": str(array.dtype)}


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = "".join(
        json.dumps(
            record,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
        for record in records
    )
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _planner_summary(recorder: PlannerDiagnosticRecorder) -> list[dict[str, Any]]:
    return [
        {
            "planning_call_index": record.get("planning_call_index"),
            "move_call_index": record.get("enclosing_move_call_index"),
            "action_type": record.get("action_type"),
            "action_repr": record.get("action_repr"),
            "success": record.get("motion_gen_success"),
            "status": record.get("motion_gen_status"),
            "step_count": record.get("task_step_count"),
            "elapsed_seconds": record.get("planning_elapsed_seconds"),
        }
        for record in recorder.planning_calls
    ]


def _eligibility_input(
    *,
    seed: int,
    reset_exception: bool,
    plan_success: bool,
    sensor_names: list[str],
    rgb_marker_valid: bool,
    failure_stage: str | None,
    planner_failure: dict[str, Any] | None,
) -> ResetEligibilityInput:
    return ResetEligibilityInput(
        seed=seed,
        reset_exception=reset_exception,
        plan_success=plan_success,
        tactile_sensor_count=len(sensor_names),
        rgb_marker_valid=rgb_marker_valid,
        failure_stage=failure_stage,
        planner_status=(
            planner_failure.get("native_result_status") if planner_failure else None
        ),
        failed_move_call_index=(
            planner_failure.get("failed_move_call_index") if planner_failure else None
        ),
        failed_planning_call_index=(
            planner_failure.get("failed_planning_call_index")
            if planner_failure
            else None
        ),
    )


def run_diagnostic(args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    task_root = Path(args.source_root).expanduser().resolve()
    asset_root = Path(args.asset_root).expanduser().resolve()
    config = _load_config(config_path)
    seeds = list(args.seeds) if args.seeds is not None else [
        int(seed) for seed in config.get("candidate_order", [])
    ]
    if not seeds or seeds != sorted(seeds) or len(seeds) != len(set(seeds)):
        raise ValueError("candidate seeds must be unique and strictly ascending")
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    run_state: dict[str, Any] = {
        "schema_version": "openeta.univtac.reset_diagnostic_run.v1",
        "task": args.task,
        "task_mode": args.task_mode,
        "config_profile": args.config_profile,
        "candidate_order": seeds,
        "source_root": str(task_root),
        "asset_root_requested": str(asset_root),
        "started_at": _utc_now(),
        "ended_at": None,
        "status": "starting",
        "error": None,
    }
    write_json(output_root / "diagnostic_run.json", run_state)

    simulation_app = None
    task = None
    original_cwd = Path.cwd()
    runtime_directory = tempfile.TemporaryDirectory(prefix="openeta-univtac-reset-")
    records: list[ResetEligibilityInput] = []
    overall_error = False
    cleanup_abort = False
    try:
        assert_task_not_imported(args.task)
        from isaaclab.app import AppLauncher

        app_launcher = AppLauncher(args)
        simulation_app = app_launcher.app
        if simulation_app is None:
            raise RuntimeError("AppLauncher returned no simulation_app")

        os.chdir(task_root)
        task_module, context = import_task_after_launcher(
            simulation_app=simulation_app,
            task_root=task_root,
            task_name=args.task,
            runtime_root=Path(runtime_directory.name),
            asset_root=asset_root,
        )
        runtime_manifest = collect_runtime_manifest(
            repo_root=REPO_ROOT,
            current_task_root=DEFAULT_SOURCE_ROOT,
            context=context,
            task_name=args.task,
        )
        if manifest_contains_secret_keys(runtime_manifest):
            raise RuntimeError("runtime manifest contains a credential-shaped key")
        write_json(output_root / "runtime_manifest.json", runtime_manifest)
        write_json(output_root / "asset_manifest.json", collect_asset_manifest(context))

        env_cfg = _configure_task(
            task_module=task_module,
            task_root=task_root,
            config=config,
            config_profile=args.config_profile,
            native_root=Path(runtime_directory.name) / "native",
            device=getattr(args, "device", None),
        )
        task = task_module.Task(env_cfg, mode=args.task_mode)
        recorder = PlannerDiagnosticRecorder(task)
        recorder.install()
        run_state["status"] = "running"
        write_json(output_root / "diagnostic_run.json", run_state)

        for seed in seeds:
            seed_dir = output_root / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=False)
            recorder.start_seed(seed)
            reset_exception = False
            reset_exception_payload: dict[str, Any] | None = None
            plan_success = False
            observation_checked = False
            sensor_names: list[str] = []
            tactile_metadata: dict[str, Any] = {}
            rgb_marker_valid = False
            failure_stage: str | None = None
            cleanup_ok = False
            reset_started = time.perf_counter()
            try:
                task.reset(seed=seed)
                plan_success = task.plan_success is True
                if not plan_success:
                    failure_stage = "reset_pre_move_planner"
                else:
                    observation = task._get_observations()
                    observation_checked = True
                    discovered = validate_tactile_packets(
                        observation,
                        strict_two_tactile_sensors=bool(
                            config["strict_two_tactile_sensors"]
                        ),
                        fail_on_missing_rgb_marker=bool(
                            config["fail_on_missing_rgb_marker"]
                        ),
                    )
                    sensor_names = list(discovered)
                    tactile_metadata = {
                        name: {
                            "rgb_marker": _array_metadata(
                                observation["tactile"][name]["rgb_marker"]
                            )
                        }
                        for name in sensor_names
                    }
                    rgb_marker_valid = True
            except Exception as exc:
                reset_exception = True
                failure_stage = (
                    "observation_contract"
                    if plan_success and observation_checked
                    else "reset_exception"
                )
                reset_exception_payload = write_exception(seed_dir / "exception.json", exc)
            reset_elapsed = time.perf_counter() - reset_started
            planner_failure = recorder.planner_failure()
            if planner_failure is not None:
                write_json(seed_dir / "planner_failure.json", planner_failure)
            _write_jsonl(seed_dir / "planner_calls.jsonl", recorder.planning_calls)

            eligibility = _eligibility_input(
                seed=seed,
                reset_exception=reset_exception,
                plan_success=plan_success,
                sensor_names=sensor_names,
                rgb_marker_valid=rgb_marker_valid,
                failure_stage=failure_stage,
                planner_failure=planner_failure,
            )
            try:
                task.clean_cache(result=None)
                cleanup_ok = True
            except Exception as exc:
                overall_error = True
                cleanup_abort = True
                write_exception(seed_dir / "cleanup_exception.json", exc)
                failure_stage = "cleanup"

            diagnostic = {
                "schema_version": "openeta.univtac.reset_diagnostic.v1",
                "seed": seed,
                "task": args.task,
                "task_mode": args.task_mode,
                "config_profile": args.config_profile,
                "reset_elapsed_seconds": reset_elapsed,
                "reset_exception": reset_exception,
                "reset_exception_summary": (
                    {
                        "type": reset_exception_payload["error_type"],
                        "message": reset_exception_payload["error"],
                    }
                    if reset_exception_payload
                    else None
                ),
                "plan_success": plan_success,
                "observation_checked": observation_checked,
                "tactile_sensor_names": sensor_names,
                "tactile_sensor_count": len(sensor_names),
                "tactile_metadata": tactile_metadata,
                "rgb_marker_valid": rgb_marker_valid,
                "reset_valid": eligibility.reset_valid,
                "eligibility_included": cleanup_ok,
                "eligibility_excluded_reason": None if cleanup_ok else "cleanup_failed",
                "failure_stage": failure_stage,
                "planner_call_count": len(recorder.planning_calls),
                "move_call_count": len(recorder.move_calls),
                "planner_call_summary": _planner_summary(recorder),
                "move_calls": recorder.move_calls,
                "plan_success_transitions": recorder.plan_success_transitions,
                "planner_failure_path": (
                    f"seed_{seed}/planner_failure.json" if planner_failure else None
                ),
                "planner_calls_path": f"seed_{seed}/planner_calls.jsonl",
                "cleanup_ok": cleanup_ok,
            }
            write_json(seed_dir / "reset_diagnostic.json", diagnostic)
            if cleanup_ok:
                records.append(eligibility)
            eligibility_summary = summarize_seed_eligibility(
                records,
                seeds,
                required=int(config["required_valid_seeds"]),
            )
            write_json(output_root / "seed_eligibility.json", eligibility_summary)

            if cleanup_abort or should_stop_scan(
                records,
                seeds,
                required=int(config["required_valid_seeds"]),
            ):
                break

        final_eligibility = summarize_seed_eligibility(
            records,
            seeds,
            required=int(config["required_valid_seeds"]),
        )
        write_json(output_root / "seed_eligibility.json", final_eligibility)
        if len(final_eligibility["selected_smoke_seeds"]) < int(
            config["required_valid_seeds"]
        ):
            overall_error = True
    except Exception as exc:
        overall_error = True
        fatal = write_exception(output_root / "exception.json", exc)
        run_state["error"] = {
            "type": fatal["error_type"],
            "message": fatal["error"],
            "traceback_path": "exception.json",
        }
    finally:
        close_errors: list[dict[str, str]] = []
        if task is not None:
            try:
                task.close()
            except Exception as exc:
                close_errors.append({"component": "task", "error": str(exc)})
        os.chdir(original_cwd)
        if close_errors:
            overall_error = True
            write_json(output_root / "close_errors.json", {"errors": close_errors})
        run_state["ended_at"] = _utc_now()
        run_state["status"] = "failed" if overall_error else "passed"
        write_json(output_root / "diagnostic_run.json", run_state)
        if simulation_app is not None:
            run_state["simulation_app_close_attempted"] = True
            write_json(output_root / "diagnostic_run.json", run_state)
            try:
                simulation_app.close()
            except Exception as exc:
                overall_error = True
                write_exception(output_root / "simulation_app_close_exception.json", exc)
                run_state["status"] = "failed"
                run_state["error"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback_path": "simulation_app_close_exception.json",
                }
                write_json(output_root / "diagnostic_run.json", run_state)
        runtime_directory.cleanup()
    return 1 if overall_error else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="insert_hole")
    parser.add_argument("--seeds", type=_parse_seeds, default=None)
    parser.add_argument("--task-mode", choices=("eval", "collect"), default="eval")
    parser.add_argument(
        "--config-profile", choices=("smoke", "official-demo"), default="smoke"
    )
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--source-root", default=str(DEFAULT_SOURCE_ROOT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))

    assert_task_not_imported("insert_hole")
    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True
    args.num_envs = 1
    return run_diagnostic(args)


if __name__ == "__main__":
    raise SystemExit(main())
