#!/usr/bin/env python3
"""Run one native Pull Out Key expert episode inside Isaac Sim."""

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
        row = {
            "stage": stage,
            "event": event,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic() - self.started,
            "pid": os.getpid(),
            "step_count": int(getattr(task, "step_count", 0)) if task else None,
            "take_action_count": int(getattr(task, "take_action_cnt", 0)) if task else None,
            "atom_id": int(getattr(task, "atom_id", 0)) if task else None,
            "atom_tag": str(getattr(task, "atom_tag", "")) if task else None,
            "plan_success": bool(getattr(task, "plan_success", False)) if task else None,
            **extra,
        }
        _append_jsonl(self.path, row)


def _parse_base_args(
    argv: list[str] | None,
) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    base_args, _ = parser.parse_known_args(argv)
    return parser, base_args


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", buffering=1) as stream:
        stream.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _module_path(module: Any) -> str:
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError(f"module {module.__name__} has no file path")
    return str(Path(module_file).resolve())


def main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    parser, base = _parse_base_args(argv)
    source_root = base.source_root.expanduser().resolve()
    repo_root = base.repo_root.expanduser().resolve()
    episode_root = base.output_root.expanduser().resolve()
    seed = int(base.seed)
    episode_root.mkdir(parents=True, exist_ok=True)
    if (episode_root / "child_result.json").exists():
        raise FileExistsError(f"expert episode root is not fresh: {episode_root}")
    stages = StageRecorder(episode_root / "stages.jsonl", started)
    task = None
    simulation_app = None
    planner = None
    transitions: list[dict[str, Any]] = []
    cleanup = {
        "task_close": False,
        "simulation_app_close_invoked": False,
        "simulation_app_close_returned": False,
    }
    counters = {
        "reset_call_count": 0,
        "play_once_call_count": 0,
        "move_call_count": 0,
        "observation_call_count": 0,
        "check_success_call_count": 0,
        "check_early_stop_call_count": 0,
        "explicit_post_pull_delay_count": 0,
    }
    result: dict[str, Any] = {
        "schema_version": "openeta.univtac.native_expert_episode.v1",
        "task": "pull_out_key",
        "seed": seed,
        "status": "starting",
        "classification": "app_launcher_failed",
        "counters": counters,
        "cleanup": cleanup,
        "transitions": [],
        "final_result": None,
        "error": None,
    }

    try:
        stages.record("app_launcher", "enter")
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args(argv)
        args.enable_cameras = True
        args.num_envs = 1
        args.device = "cuda:0"
        simulation_app = AppLauncher(args).app
        if simulation_app is None:
            raise RuntimeError("AppLauncher returned no simulation_app")
        stages.record("app_launcher", "exit")

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
        from sim.envs.univtac.native_operation import (
            EXPERT_SEEDS,
            EXPERT_SEGMENTS,
            attach_native_outcome,
            build_native_move_transition,
            expert_episode_success,
            segment_name,
            serialize_native_action,
            validate_native_expert_config,
        )
        from sim.envs.univtac.observation import capture_snapshot
        from sim.envs.univtac.planner_diagnostics import (
            PlannerDiagnosticRecorder,
            safe_diagnostic_value,
        )
        from sim.envs.univtac.pull_out_key_gate import summarize_pull_out_key_observation
        from sim.envs.univtac.trace import verify_artifacts, write_json, write_snapshot

        stages.record("post_launcher_imports", "exit")

        config_payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if not isinstance(config_payload, dict):
            raise TypeError("native expert config must be a mapping")
        config = validate_native_expert_config(config_payload)
        if seed not in EXPERT_SEEDS:
            raise RuntimeError(f"unexpected expert seed: {seed}")

        native_config, native_config_path = load_task_config(
            source_root / "task_config" / f"{config['task_config']}.yml"
        )
        task_module, env_cfg, timing, native_save_dir = build_task_env_cfg(
            config["task"],
            native_config,
            config["task_config"],
            config["mode"],
            device=config["device"],
            save_dir=episode_root / "native",
        )
        env_cfg.save_frequency = 0
        env_cfg.video_frequency = 0

        modules = {
            "envs.pull_out_key": task_module,
            "envs._base_task": base_task_module,
            "tacex": tacex,
            "tacex_assets": tacex_assets,
            "tacex_uipc": tacex_uipc,
        }
        module_paths = {name: _module_path(module) for name, module in modules.items()}
        allowed_roots = (source_root, source_root / "third_party" / "TacEx" / "source")
        for name, value in module_paths.items():
            path = Path(value)
            if not any(path.is_relative_to(root) for root in allowed_roots):
                raise RuntimeError(f"{name} resolved outside pinned source: {path}")
        write_json(episode_root / "module_realpaths.json", module_paths)

        stages.record("task_constructor", "enter")
        task = task_module.Task(env_cfg, mode=config["mode"])
        stages.task = task
        stages.record("task_constructor", "exit")
        result["classification"] = "native_reset_failed"
        result["task_config"] = {
            "native_config_path": str(native_config_path),
            "mode": config["mode"],
            "device": str(env_cfg.sim.device),
            "num_envs": int(env_cfg.scene.num_envs),
            "save_frequency": int(env_cfg.save_frequency),
            "video_frequency": int(env_cfg.video_frequency),
            "render_frequency": int(env_cfg.render_frequency),
            "timing": repr(timing),
            "native_save_dir": str(native_save_dir),
        }

        planner = PlannerDiagnosticRecorder(task)
        planner.install()
        planner.start_seed(seed)

        stages.record("reset", "enter")
        counters["reset_call_count"] += 1
        task.reset(seed=seed)
        stages.record("reset", "exit")
        if not task.plan_success:
            result["classification"] = "native_pre_move_planner_failed"
            raise RuntimeError("native reset/pre_move returned with plan_success=False")

        def capture(
            *, root: Path, phase: str, action_id: str, metadata: dict[str, Any]
        ):
            counters["observation_call_count"] += 1
            observation = task._get_observations()
            observation_summary, contact_summary = summarize_pull_out_key_observation(
                observation
            )
            captured = capture_snapshot(
                observation,
                output_root=episode_root,
                seed_dir=root,
                task_name="pull_out_key",
                seed=seed,
                phase=phase,
                action_id=action_id,
                simulator_step=int(task.step_count),
                take_action_count=int(task.take_action_cnt),
                task_instruction=config["task_instruction"],
                task_metadata={
                    **metadata,
                    "plan_success": bool(task.plan_success),
                    "contact_summary": contact_summary,
                },
                native_check_success=None,
                save_host_only=config["save_host_only"],
                strict_two_tactile_sensors=config["strict_two_tactile_sensors"],
                fail_on_missing_rgb_marker=True,
            )
            validate_operator_visible(captured.snapshot.operator_visible)
            verify_artifacts(episode_root, captured.snapshot.to_dict())
            write_json(root / f"observation_{phase}.json", observation_summary)
            return captured

        pre_task = capture(
            root=episode_root / "pre_task",
            phase="pre_action",
            action_id=f"native-expert-{seed}-pre-task",
            metadata={"stage": "post_pre_move_pre_play_once"},
        )
        write_snapshot(episode_root / "snapshot_pre_task.json", pre_task.snapshot)

        original_move = task.move
        original_delay = task.delay
        active_move: dict[str, Any] | None = None
        pending_pull: dict[str, Any] | None = None
        final_capture = None

        def finalize_move(context: dict[str, Any], post_settle_delay_steps: int) -> None:
            nonlocal final_capture
            segment = str(context["semantic_segment"])
            transition_dir = episode_root / "transitions" / segment
            after = capture(
                root=transition_dir,
                phase="post_action",
                action_id=str(context["action_id"]),
                metadata={
                    "semantic_segment": segment,
                    "native_move_call_index": context["move_index"],
                    "post_settle_delay_steps": post_settle_delay_steps,
                },
            )
            write_snapshot(transition_dir / "snapshot_after.json", after.snapshot)
            planner_end = len(planner.planning_calls)
            transition = build_native_move_transition(
                output_root=episode_root,
                transition_dir=transition_dir,
                seed=seed,
                move_index=int(context["move_index"]),
                semantic_segment=segment,
                before=context["before"],
                after=after,
                native_actions_before=context["native_actions_before"],
                native_actions_after=context["native_actions_after"],
                move_args=context["move_args"],
                move_kwargs=context["move_kwargs"],
                move_returned=context.get("move_returned"),
                plan_success_before=bool(context["plan_success_before"]),
                plan_success_after=bool(task.plan_success),
                planner_call_indices=range(context["planner_start"] + 1, planner_end + 1),
                post_settle_delay_steps=post_settle_delay_steps,
            )
            write_json(transition_dir / "transition.json", transition)
            verify_artifacts(
                episode_root,
                context["before"].snapshot.to_dict(),
                after.snapshot.to_dict(),
                transition,
            )
            transitions.append(transition)
            _append_jsonl(episode_root / "action_trace.jsonl", transition)
            context["after"] = after
            if segment == "pull_key_out":
                final_capture = after
            stages.record(segment, "transition_saved")

        def move_wrapper(_self, actions, *move_args, **move_kwargs):
            nonlocal active_move, pending_pull
            counters["move_call_count"] += 1
            move_index = counters["move_call_count"]
            semantic_segment = segment_name(move_index)
            transition_dir = episode_root / "transitions" / semantic_segment
            action_id = f"native-expert-{seed}-{semantic_segment}"
            action_list = list(actions)
            before = capture(
                root=transition_dir,
                phase="pre_action",
                action_id=action_id,
                metadata={
                    "semantic_segment": semantic_segment,
                    "native_move_call_index": move_index,
                },
            )
            write_snapshot(transition_dir / "snapshot_before.json", before.snapshot)
            context = {
                "move_index": move_index,
                "semantic_segment": semantic_segment,
                "action_id": action_id,
                "before": before,
                "native_actions_before": [serialize_native_action(item) for item in action_list],
                "move_args": safe_diagnostic_value(move_args),
                "move_kwargs": safe_diagnostic_value(move_kwargs),
                "planner_start": len(planner.planning_calls),
                "plan_success_before": bool(task.plan_success),
            }
            active_move = context
            stages.record(semantic_segment, "move_enter")
            try:
                returned = original_move(action_list, *move_args, **move_kwargs)
                context["move_returned"] = returned
                return returned
            finally:
                context["native_actions_after"] = [
                    serialize_native_action(item) for item in action_list
                ]
                stages.record(semantic_segment, "move_exit")
                active_move = None
                if semantic_segment == "pull_key_out":
                    pending_pull = context
                else:
                    finalize_move(context, 0)

        def delay_wrapper(_self, *delay_args, **delay_kwargs):
            nonlocal pending_pull
            returned = original_delay(*delay_args, **delay_kwargs)
            if active_move is None and pending_pull is not None:
                steps = delay_kwargs.get("steps", delay_args[0] if delay_args else 20)
                counters["explicit_post_pull_delay_count"] += 1
                context = pending_pull
                pending_pull = None
                finalize_move(context, int(steps))
            return returned

        task.move = types.MethodType(move_wrapper, task)
        task.delay = types.MethodType(delay_wrapper, task)

        result["classification"] = "native_play_once_failed"
        stages.record("play_once", "enter")
        counters["play_once_call_count"] += 1
        task.play_once()
        stages.record("play_once", "exit")
        if pending_pull is not None:
            context = pending_pull
            pending_pull = None
            finalize_move(context, 0)

        if counters["play_once_call_count"] != 1:
            raise RuntimeError("native task.play_once must be called exactly once")
        if counters["move_call_count"] != len(EXPERT_SEGMENTS):
            raise RuntimeError(
                f"native play_once expected {len(EXPERT_SEGMENTS)} moves, "
                f"observed {counters['move_call_count']}"
            )
        if [item["semantic_segment"] for item in transitions] != list(EXPERT_SEGMENTS):
            raise RuntimeError("native move transitions do not match the three fixed segments")
        if counters["explicit_post_pull_delay_count"] != 1:
            raise RuntimeError("native play_once must execute one explicit post-pull delay")
        if transitions[-1]["post_settle_delay_steps"] != 20:
            raise RuntimeError("native pull post-settle delay must remain 20 steps")

        if final_capture is None:
            raise RuntimeError("native pull transition did not produce a final observation")
        write_json(episode_root / "planner_diagnostics.json", json.loads(planner.to_json()))
        write_snapshot(episode_root / "snapshot_final.json", final_capture.snapshot)

        plan_success = bool(task.plan_success)
        counters["check_success_call_count"] += 1
        native_check_success = bool(task.check_success())
        native_check_early_stop: bool | None = None
        if plan_success and native_check_success:
            counters["check_early_stop_call_count"] += 1
            native_check_early_stop = bool(task.check_early_stop())
        episode_success = expert_episode_success(
            plan_success=plan_success,
            native_check_success=native_check_success,
            native_check_early_stop=native_check_early_stop,
        )
        native_outcome = {
            "plan_success": plan_success,
            "native_check_success": native_check_success,
            "native_check_early_stop": native_check_early_stop,
            "expert_episode_success": episode_success,
        }
        transitions = [attach_native_outcome(item, native_outcome) for item in transitions]
        for transition in transitions:
            segment = transition["semantic_segment"]
            write_json(episode_root / "transitions" / segment / "transition.json", transition)
        (episode_root / "action_trace.jsonl").unlink(missing_ok=True)
        for transition in transitions:
            _append_jsonl(episode_root / "action_trace.jsonl", transition)

        final_result = {
            "schema_version": "openeta.univtac.native_expert_result.v1",
            "task": "pull_out_key",
            "seed": seed,
            **native_outcome,
            "transition_count": len(transitions),
            "semantic_segments": [item["semantic_segment"] for item in transitions],
            "final_simulator_step": int(task.step_count),
            "final_take_action_count": int(task.take_action_cnt),
        }
        write_json(episode_root / "final_result.json", final_result)
        episode = {
            "schema_version": "openeta.univtac.native_expert_episode.v1",
            "task": "pull_out_key",
            "seed": seed,
            "status": "completed",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "duration_seconds": time.monotonic() - started,
            "counters": counters,
            "transition_count": len(transitions),
            "semantic_segments": list(EXPERT_SEGMENTS),
            "native_outcome": native_outcome,
            "operator_visible_success_exposed": False,
        }
        write_json(episode_root / "episode.json", episode)
        result.update(
            {
                "status": "completed",
                "classification": (
                    "native_expert_episode_success"
                    if episode_success
                    else "native_expert_episode_task_failure"
                ),
                "transitions": [
                    f"transitions/{item['semantic_segment']}/transition.json"
                    for item in transitions
                ],
                "final_result": final_result,
            }
        )
    except Exception as exc:  # noqa: BLE001 - persist native runtime evidence
        result["status"] = "failed"
        result["error"] = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(episode_root / "exception.json", result["error"])
        if planner is not None:
            try:
                _write_json(
                    episode_root / "planner_diagnostics.json",
                    json.loads(planner.to_json()),
                )
            except Exception as diagnostic_exc:  # noqa: BLE001
                result["planner_diagnostic_error"] = str(diagnostic_exc)
    finally:
        if task is not None:
            stages.record("task_close", "enter")
            try:
                task.close()
                cleanup["task_close"] = True
                stages.record("task_close", "exit")
            except Exception as exc:  # noqa: BLE001
                result["classification"] = "cleanup_incomplete"
                stages.record("task_close", "exit", error_type=type(exc).__name__, error=str(exc))
        stages.task = None
        result["cleanup"] = cleanup
        result["counters"] = counters
        _write_json(episode_root / "child_result.json", result)
        if simulation_app is not None:
            cleanup["simulation_app_close_invoked"] = True
            _write_json(episode_root / "child_result.json", result)
            stages.record("simulation_app_close", "enter")
            try:
                simulation_app.close()
                cleanup["simulation_app_close_returned"] = True
                stages.record("simulation_app_close", "exit")
            except Exception as exc:  # noqa: BLE001
                result["classification"] = "cleanup_incomplete"
                stages.record(
                    "simulation_app_close", "exit", error_type=type(exc).__name__, error=str(exc)
                )
        result["cleanup"] = cleanup
        _write_json(episode_root / "child_result.json", result)

    completed = result["status"] == "completed" and cleanup["task_close"]
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
