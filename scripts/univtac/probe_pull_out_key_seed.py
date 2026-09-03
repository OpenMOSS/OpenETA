#!/usr/bin/env python3
"""Run the single Isaac51 Pull Out Key pre-action observation probe."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
import types
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StageRecorder:
    def __init__(self, path: Path, started: float) -> None:
        self.path = path
        self.started = started
        self.task: Any | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, stage: str, event: str, **extra: Any) -> None:
        task = self.task
        payload = {
            "stage": stage,
            "event": event,
            "utc_timestamp": datetime.now(timezone.utc).isoformat(),
            "monotonic_elapsed_seconds": time.monotonic() - self.started,
            "pid": os.getpid(),
            "process_group_id": os.getpgrp(),
            "step_count": int(getattr(task, "step_count", 0)) if task else None,
            "take_action_count": int(getattr(task, "take_action_cnt", 0)) if task else None,
            "atom_id": int(getattr(task, "atom_id", 0)) if task else None,
            "atom_tag": str(getattr(task, "atom_tag", "")) if task else None,
            "plan_success": bool(getattr(task, "plan_success", False)) if task else None,
            **extra,
        }
        with self.path.open("a", encoding="utf-8", buffering=1) as stream:
            stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def wrap_method(self, owner: Any, method_name: str, stage: str) -> None:
        original = getattr(owner, method_name)

        def wrapper(_self, *args, **kwargs):
            self.record(stage, "enter")
            try:
                result = original(*args, **kwargs)
            except BaseException as exc:
                self.record(
                    stage,
                    "exit",
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                raise
            self.record(stage, "exit")
            return result

        setattr(owner, method_name, types.MethodType(wrapper, owner))


def parse_base_args(
    argv: list[str] | None = None,
) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    base_args, _ = parser.parse_known_args(argv)
    return parser, base_args


def _write_stdlib_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _module_realpath(module: Any) -> Path:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError(f"module {module.__name__} has no file path")
    return Path(module_file).resolve()


def _install_forbidden_call_guards(
    task: Any,
    counters: dict[str, int],
) -> None:
    for method_name, counter_name in (
        ("play_once", "play_once_call_count"),
        ("_play_once", "play_once_call_count"),
        ("check_success", "check_success_call_count"),
        ("check_early_stop", "check_early_stop_call_count"),
    ):
        if not hasattr(task, method_name):
            continue

        def forbidden(_self, *args, _counter=counter_name, _name=method_name, **kwargs):
            counters[_counter] += 1
            raise RuntimeError(f"forbidden method was called during pre-action gate: {_name}")

        setattr(task, method_name, types.MethodType(forbidden, task))


def main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    parser, base_args = parse_base_args(argv)
    source_root = base_args.source_root.expanduser().resolve()
    repo_root = base_args.repo_root.expanduser().resolve()
    output_root = base_args.output_root.expanduser().resolve()
    seed_dir = output_root / "pull_out_key_seed1000000"
    seed_dir.mkdir(parents=True, exist_ok=False)
    stages = StageRecorder(seed_dir / "stages.jsonl", started)
    task = None
    simulation_app = None
    recorder = None
    cleanup = {
        "task_close": False,
        "simulation_app_close_invoked": False,
        "simulation_app_close_returned": False,
    }
    counters = {
        "reset_call_count": 0,
        "observation_call_count": 0,
        "play_once_call_count": 0,
        "check_success_call_count": 0,
        "check_early_stop_call_count": 0,
    }
    result: dict[str, Any] = {
        "schema_version": "openeta.univtac.pull_out_key_gate.v1",
        "classification": "app_launcher_failed",
        "requested_seed": 1_000_000,
        "observed_seed": None,
        "unexpected_seeds": [],
        "module_realpaths": {},
        "counters": counters,
        "cleanup": cleanup,
        "snapshot_pre": None,
        "snapshot_post_created": False,
        "transition_created": False,
        "error": None,
    }
    try:
        stages.record("app_launcher_construct", "enter")
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args(argv)
        args.enable_cameras = True
        args.num_envs = 1
        args.device = "cuda:0"
        app_launcher = AppLauncher(args)
        simulation_app = app_launcher.app
        if simulation_app is None:
            raise RuntimeError("AppLauncher returned no simulation_app")
        stages.record("app_launcher_construct", "exit")
        stages.record("app_launcher_started", "enter")
        stages.record("app_launcher_started", "exit")

        for path in (repo_root, source_root):
            while str(path) in sys.path:
                sys.path.remove(str(path))
            sys.path.insert(0, str(path))

        stages.record("post_launcher_imports", "enter")
        import envs._base_task as base_task_module
        import tacex
        import tacex_assets
        import tacex_uipc
        import yaml
        from envs.utils.env_parser import build_task_env_cfg, load_task_config

        from sim.envs.univtac.contract import validate_operator_visible
        from sim.envs.univtac.observation import capture_snapshot
        from sim.envs.univtac.planner_diagnostics import PlannerDiagnosticRecorder
        from sim.envs.univtac.pull_out_key_gate import (
            summarize_pull_out_key_observation,
            validate_gate_config,
        )
        from sim.envs.univtac.trace import verify_artifacts, write_json, write_snapshot

        stages.record("post_launcher_imports", "exit")

        config_payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if not isinstance(config_payload, dict):
            raise TypeError("gate config must be a mapping")
        gate_config = validate_gate_config(config_payload)
        stages.record("task_config_load", "enter")
        native_config, native_config_path = load_task_config(
            source_root / "task_config" / f"{gate_config['task_config']}.yml"
        )
        task_module, env_cfg, timing, native_save_dir = build_task_env_cfg(
            gate_config["task"],
            native_config,
            gate_config["task_config"],
            "collect",
            device=gate_config["device"],
            save_dir=seed_dir / "native",
        )
        env_cfg.save_frequency = 0
        env_cfg.video_frequency = 0
        stages.record("task_config_load", "exit")

        imported_modules = {
            "envs.pull_out_key": task_module,
            "envs._base_task": base_task_module,
            "tacex": tacex,
            "tacex_uipc": tacex_uipc,
            "tacex_assets": tacex_assets,
        }
        allowed_source_roots = (
            source_root,
            source_root / "third_party" / "TacEx" / "source",
        )
        module_realpaths: dict[str, str] = {}
        for module_name, module in imported_modules.items():
            module_path = _module_realpath(module)
            if not any(module_path.is_relative_to(root) for root in allowed_source_roots):
                raise RuntimeError(
                    f"{module_name} resolved outside pinned Isaac51 source: {module_path}"
                )
            module_realpaths[module_name] = str(module_path)
        result["module_realpaths"] = module_realpaths
        _write_stdlib_json(seed_dir / "module_realpaths.json", module_realpaths)

        stages.record("task_constructor", "enter")
        task = task_module.Task(env_cfg, mode="collect")
        stages.task = task
        stages.record("task_constructor", "exit")
        result["classification"] = "pull_out_key_seed1000000_reset_before_pre_move_failed"
        result["task_config"] = {
            "native_config_path": str(native_config_path),
            "mode": "collect",
            "device": str(env_cfg.sim.device),
            "num_envs": int(env_cfg.scene.num_envs),
            "save_frequency": int(env_cfg.save_frequency),
            "video_frequency": int(env_cfg.video_frequency),
            "render_frequency": int(env_cfg.render_frequency),
            "timing": repr(timing),
            "native_save_dir": str(native_save_dir),
        }

        stages.record("planner_recorder_install", "enter")
        recorder = PlannerDiagnosticRecorder(task)
        recorder.install()
        recorder.start_seed(gate_config["seed"])
        stages.record("planner_recorder_install", "exit")

        stages.wrap_method(task, "_reset_actors", "reset_actors")
        stages.wrap_method(
            task,
            "_stabilize_and_calibrate_marker_references",
            "marker_stabilization",
        )
        stages.wrap_method(
            task._tactile_manager,
            "calibrate_marker_references",
            "marker_calibration",
        )
        stages.wrap_method(task, "pre_move", "pre_move")
        stages.wrap_method(task, "delay", "pre_move_delay")
        stages.wrap_method(task.atom, "grasp_actor", "pre_move_grasp_actor")
        _install_forbidden_call_guards(task, counters)

        stages.record("reset", "enter")
        counters["reset_call_count"] += 1
        result["observed_seed"] = gate_config["seed"]
        try:
            task.reset(seed=gate_config["seed"])
        except Exception as exc:
            stages.record("reset", "exit", error_type=type(exc).__name__, error=str(exc))
            if any(
                record.get("stage") == "pre_move" and record.get("event") == "enter"
                for record in _read_stage_records(stages.path)
            ):
                result["classification"] = "pull_out_key_seed1000000_pre_move_exception"
            raise
        stages.record("reset", "exit")
        stages.record("reset_returned", "enter")
        stages.record("reset_returned", "exit")

        diagnostics = json.loads(recorder.to_json())
        write_json(seed_dir / "planner_diagnostics.json", diagnostics)
        planner_failure = recorder.planner_failure()
        planner_ok = (
            bool(recorder.move_calls)
            and bool(recorder.planning_calls)
            and planner_failure is None
            and all("exception" not in call for call in recorder.planning_calls)
            and all(call.get("motion_gen_success") is not False for call in recorder.planning_calls)
            and task.plan_success is True
        )
        if not planner_ok:
            result["classification"] = "pull_out_key_seed1000000_pre_move_planner_failed"
            raise RuntimeError(f"pre_move planner gate failed: {planner_failure}")
        if task.in_pre_move is not False:
            raise RuntimeError("task.in_pre_move did not return to false")
        for attribute in ("cid", "target_pose", "slot_init_pose"):
            if not hasattr(task, attribute):
                raise RuntimeError(f"Pull Out Key reset did not create task.{attribute}")

        stages.record("get_observations", "enter")
        counters["observation_call_count"] += 1
        observation = task._get_observations()
        stages.record("get_observations", "exit")
        try:
            observation_summary, contact_summary = summarize_pull_out_key_observation(observation)
        except Exception:
            result["classification"] = "pull_out_key_seed1000000_observation_contract_failed"
            raise
        write_json(seed_dir / "native_observation_tree.json", observation_summary["key_tree"])
        write_json(seed_dir / "observation_summary.json", observation_summary)
        write_json(seed_dir / "contact_summary.json", contact_summary)
        if not contact_summary["any_contact_candidate"]:
            result["classification"] = "pull_out_key_seed1000000_tactile_contact_not_observed"
            raise RuntimeError("no positive press_depth contact candidate was observed")

        stages.record("snapshot_projection", "enter")
        capture = capture_snapshot(
            observation,
            output_root=output_root,
            seed_dir=seed_dir,
            task_name="pull_out_key",
            seed=gate_config["seed"],
            phase="pre_action",
            action_id="pull-out-key-seed-1000000-ready",
            simulator_step=int(task.step_count),
            take_action_count=int(task.take_action_cnt),
            task_instruction=gate_config["task_instruction"],
            task_metadata={
                "seed_label": gate_config["seed_label"],
                "plan_success": bool(task.plan_success),
                "cid_present": hasattr(task, "cid"),
                "target_pose_present": hasattr(task, "target_pose"),
                "slot_init_pose_present": hasattr(task, "slot_init_pose"),
            },
            native_check_success=None,
            save_host_only=True,
            strict_two_tactile_sensors=True,
            fail_on_missing_rgb_marker=True,
        )
        validate_operator_visible(capture.snapshot.operator_visible)
        write_snapshot(seed_dir / "snapshot_pre.json", capture.snapshot)
        verify_artifacts(output_root, capture.snapshot.to_dict())
        stages.record("snapshot_projection", "exit")
        result.update(
            {
                "classification": "scoped_launcher_and_pull_out_key_seed1000000_gate_passed",
                "snapshot_pre": str((seed_dir / "snapshot_pre.json").relative_to(output_root)),
                "planner_move_call_count": len(recorder.move_calls),
                "planner_call_count": len(recorder.planning_calls),
                "planner_failure": None,
                "plan_success": bool(task.plan_success),
                "in_pre_move": bool(task.in_pre_move),
                "cid_present": hasattr(task, "cid"),
                "target_pose_present": hasattr(task, "target_pose"),
                "slot_init_pose_present": hasattr(task, "slot_init_pose"),
                "contact_summary": contact_summary,
                "operator_visible_keys": sorted(capture.snapshot.operator_visible),
                "host_only_keys": sorted(capture.snapshot.host_only),
            }
        )
    except Exception as exc:  # noqa: BLE001 - persist native simulator failures
        result["error"] = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_stdlib_json(seed_dir / "exception.json", result["error"])
        if recorder is not None:
            try:
                _write_stdlib_json(
                    seed_dir / "planner_diagnostics.json",
                    json.loads(recorder.to_json()),
                )
                result["planner_move_call_count"] = len(recorder.move_calls)
                result["planner_call_count"] = len(recorder.planning_calls)
                result["planner_failure"] = recorder.planner_failure()
                if recorder.planner_failure() is not None:
                    result["classification"] = "pull_out_key_seed1000000_pre_move_planner_failed"
            except Exception as diagnostic_exc:  # noqa: BLE001 - preserve primary failure
                result["planner_diagnostic_error"] = {
                    "error_type": type(diagnostic_exc).__name__,
                    "error": str(diagnostic_exc),
                }
    finally:
        if task is not None:
            stages.record("task_close", "enter")
            try:
                task.close()
                cleanup["task_close"] = True
                stages.record("task_close", "exit")
            except Exception as exc:  # noqa: BLE001 - cleanup evidence must survive
                stages.record("task_close", "exit", error_type=type(exc).__name__, error=str(exc))
                result["classification"] = "cleanup_incomplete"
        stages.task = None
        if simulation_app is not None:
            stages.record("simulation_app_close", "enter")
            cleanup["simulation_app_close_invoked"] = True
            result["counters"] = counters
            result["cleanup"] = cleanup
            result["snapshot_post_created"] = (seed_dir / "snapshot_post.json").exists()
            result["transition_created"] = (seed_dir / "transition.json").exists()
            _write_stdlib_json(seed_dir / "child_result.json", result)
            try:
                simulation_app.close()
                cleanup["simulation_app_close_returned"] = True
                stages.record("simulation_app_close", "exit")
            except Exception as exc:  # noqa: BLE001 - cleanup evidence must survive
                stages.record(
                    "simulation_app_close", "exit", error_type=type(exc).__name__, error=str(exc)
                )
                result["classification"] = "cleanup_incomplete"
        result["counters"] = counters
        result["cleanup"] = cleanup
        result["snapshot_post_created"] = (seed_dir / "snapshot_post.json").exists()
        result["transition_created"] = (seed_dir / "transition.json").exists()
        _write_stdlib_json(seed_dir / "child_result.json", result)
    passed = (
        result["classification"] == "scoped_launcher_and_pull_out_key_seed1000000_gate_passed"
        and cleanup["task_close"]
        and cleanup["simulation_app_close_invoked"]
    )
    return 0 if passed else 1


def _read_stage_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


if __name__ == "__main__":
    raise SystemExit(main())
