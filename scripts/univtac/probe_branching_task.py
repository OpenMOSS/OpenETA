#!/usr/bin/env python3
"""Run one R1.2 native expert or controlled branching continuation."""

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
        path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, stage: str, event: str, **extra: Any) -> None:
        task = self.task
        _append_jsonl(
            self.path,
            {
                "stage": stage,
                "event": event,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.monotonic() - self.started,
                "pid": os.getpid(),
                "step_count": int(getattr(task, "step_count", 0)) if task else None,
                "take_action_count": int(getattr(task, "take_action_cnt", 0)) if task else None,
                "plan_success": bool(getattr(task, "plan_success", False)) if task else None,
                **extra,
            },
        )


def _parse_base_args(
    argv: list[str] | None,
) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--condition",
        choices=("expert", "correct", "wrong", "selected"),
        required=True,
    )
    parser.add_argument("--skill", choices=("skill_slate", "skill_ember"))
    parser.add_argument("--canonical-class")
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
    task_name = str(base.task)
    seed = int(base.seed)
    condition = str(base.condition)
    selected_skill = str(base.skill) if base.skill else None
    canonical_class = str(base.canonical_class) if base.canonical_class else None
    episode_root.mkdir(parents=True, exist_ok=True)
    if (episode_root / "child_result.json").exists():
        raise FileExistsError(f"episode root is not fresh: {episode_root}")

    stages = StageRecorder(episode_root / "stages.jsonl", started)
    task = None
    simulation_app = None
    planner = None
    transitions: list[dict[str, Any]] = []
    decision_state: dict[str, Any] | None = None
    lift_mid_success: bool | None = None
    insert_native: dict[str, Any] | None = None
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
        "decision_snapshot_count": 0,
        "check_mid_success_call_count": 0,
        "check_success_call_count": 0,
        "check_early_stop_call_count": 0,
    }
    result: dict[str, Any] = {
        "schema_version": "openeta.univtac.branching_episode.v1",
        "task": task_name,
        "seed": seed,
        "condition": condition,
        "status": "starting",
        "classification": "app_launcher_failed",
        "counters": counters,
        "cleanup": cleanup,
        "decision_state": None,
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
        import numpy as np
        import tacex
        import tacex_assets
        import tacex_uipc
        import yaml
        from envs.utils.env_parser import build_task_env_cfg, load_task_config

        from sim.envs.univtac.branching_qualification import (
            SEEDS,
            TASKS,
            classify_insert_correction,
            classify_lift_branch,
            expert_segment_name,
            validate_branching_config,
        )
        from sim.envs.univtac.contract import validate_operator_visible
        from sim.envs.univtac.insert_hole_tactile_icl import (
            OPAQUE_TO_CLASS,
            validate_r13_config,
        )
        from sim.envs.univtac.insert_hole_tactile_icl import (
            QUERY_SEEDS as R13_QUERY_SEEDS,
        )
        from sim.envs.univtac.native_operation import (
            attach_native_outcome,
            build_native_move_transition,
            expert_episode_success,
            serialize_native_action,
        )
        from sim.envs.univtac.observation import capture_snapshot
        from sim.envs.univtac.planner_diagnostics import (
            PlannerDiagnosticRecorder,
            safe_diagnostic_value,
        )
        from sim.envs.univtac.trace import verify_artifacts, write_json, write_snapshot

        stages.record("post_launcher_imports", "exit")

        config_payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if not isinstance(config_payload, dict):
            raise TypeError("branching config must be a mapping")
        is_r13 = config_payload.get("round") == "R1.3"
        if is_r13:
            config = validate_r13_config(config_payload)
            allowed_tasks = ("insert_hole",)
            allowed_seeds = R13_QUERY_SEEDS
            if condition not in {"expert", "selected"}:
                raise RuntimeError("R1.3 child supports only expert or selected condition")
            if condition == "selected" and (
                selected_skill not in OPAQUE_TO_CLASS or canonical_class is None
            ):
                raise RuntimeError(
                    "R1.3 selected continuation requires skill and canonical class"
                )
        else:
            config = validate_branching_config(config_payload)
            allowed_tasks = TASKS
            allowed_seeds = SEEDS
            if condition == "selected":
                raise RuntimeError("selected condition is reserved for R1.3")
        if task_name not in allowed_tasks or seed not in allowed_seeds:
            raise RuntimeError(f"unexpected task/seed: {task_name}/{seed}")

        native_config, native_config_path = load_task_config(
            source_root / "task_config" / f"{config['task_config']}.yml"
        )
        task_module, env_cfg, timing, native_save_dir = build_task_env_cfg(
            task_name,
            native_config,
            config["task_config"],
            config["mode"],
            device=config["device"],
            save_dir=episode_root / "native",
        )
        env_cfg.save_frequency = 0
        env_cfg.video_frequency = 0
        modules = {
            f"envs.{task_name}": task_module,
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
            captured = capture_snapshot(
                observation,
                output_root=episode_root,
                seed_dir=root,
                task_name=task_name,
                seed=seed,
                phase=phase,
                action_id=action_id,
                simulator_step=int(task.step_count),
                take_action_count=int(task.take_action_cnt),
                task_instruction=config["task_instructions"][task_name],
                task_metadata={**metadata, "plan_success": bool(task.plan_success)},
                native_check_success=None,
                save_host_only=config["save_host_only"],
                strict_two_tactile_sensors=config["strict_two_tactile_sensors"],
                fail_on_missing_rgb_marker=True,
            )
            validate_operator_visible(captured.snapshot.operator_visible)
            visible = captured.snapshot.operator_visible
            if not {"head", "wrist"}.issubset(visible["cameras"]):
                raise RuntimeError("decision evidence requires head and wrist cameras")
            if len(visible["tactile"]) != 2:
                raise RuntimeError("decision evidence requires two tactile images")
            verify_artifacts(episode_root, captured.snapshot.to_dict())
            write_json(root / f"observation_tree_{phase}.json", captured.observation_key_tree)
            return captured

        pre_task = capture(
            root=episode_root / "pre_task",
            phase="pre_action",
            action_id=f"r12-{task_name}-{seed}-{condition}-pre-task",
            metadata={"stage": "post_pre_move_pre_episode", "condition": condition},
        )
        write_snapshot(episode_root / "snapshot_pre_task.json", pre_task.snapshot)

        original_move = task.move
        original_check_mid = task.check_mid_success if task_name == "lift_bottle" else None

        def capture_decision_insert() -> None:
            nonlocal decision_state, insert_native
            if decision_state is not None:
                raise RuntimeError("Insert Hole decision snapshot was captured more than once")
            rel_pose = task.prism.get_pose().rebase(task.target_pose)
            gripper_dis = task._robot_manager.get_gripper_center_pose().rebase(
                task.prism.get_pose()
            )[2]
            x_move = -gripper_dis * np.sin(rel_pose.euler[1])
            z_move = gripper_dis * (np.cos(rel_pose.euler[1]) - 1)
            insert_native = {
                "relative_pose": safe_diagnostic_value(rel_pose.tolist()),
                "gripper_dis": float(gripper_dis),
                "native_x_move": float(x_move),
                "native_z_move": float(z_move),
                "decision_class": classify_insert_correction(float(x_move)),
            }
            captured = capture(
                root=episode_root / "decision",
                phase="pre_action",
                action_id=f"r12-{task_name}-{seed}-{condition}-decision",
                metadata={"decision_point": "after_first_downward_before_corrective"},
            )
            counters["decision_snapshot_count"] += 1
            write_snapshot(episode_root / "snapshot_decision.json", captured.snapshot)
            decision_state = {
                "schema_version": "openeta.univtac.branching_decision.v1",
                "task": task_name,
                "seed": seed,
                "condition": condition,
                "decision_point": "after_first_downward_before_corrective",
                "operator_visible": captured.snapshot.operator_visible,
                "snapshot_path": "snapshot_decision.json",
                "host_only": copy_dict(insert_native),
                "native_actions_before_decision": [
                    item["semantic_segment"] for item in transitions
                ],
                "native_next_action": None,
                "final_native_outcome": None,
            }
            write_json(episode_root / "decision_state.json", decision_state)

        def segment_for_move(move_index: int) -> str:
            if task_name == "insert_hole":
                return expert_segment_name(task=task_name, move_index=move_index)
            if condition == "wrong" and move_index == 6:
                return "open_gripper"
            return expert_segment_name(
                task=task_name,
                move_index=move_index,
                lift_mid_success=lift_mid_success,
            )

        def move_wrapper(_self, actions, *move_args, **move_kwargs):
            counters["move_call_count"] += 1
            move_index = counters["move_call_count"]
            segment = segment_for_move(move_index)
            if segment.startswith(("unexpected_", "post_rotate_")):
                raise RuntimeError(f"unexpected native move {move_index} for {task_name}/{condition}")
            transition_dir = episode_root / "transitions" / f"{move_index:02d}_{segment}"
            action_id = f"r12-{task_name}-{seed}-{condition}-{move_index:02d}-{segment}"
            action_refs = list(actions)
            before = capture(
                root=transition_dir,
                phase="pre_action",
                action_id=action_id,
                metadata={"semantic_segment": segment, "native_move_call_index": move_index},
            )
            write_snapshot(transition_dir / "snapshot_before.json", before.snapshot)
            serialized_before = [serialize_native_action(item) for item in action_refs]
            planner_start = len(planner.planning_calls)
            plan_before = bool(task.plan_success)
            stages.record(segment, "move_enter")
            returned = original_move(actions, *move_args, **move_kwargs)
            stages.record(segment, "move_exit")
            after = capture(
                root=transition_dir,
                phase="post_action",
                action_id=action_id,
                metadata={"semantic_segment": segment, "native_move_call_index": move_index},
            )
            write_snapshot(transition_dir / "snapshot_after.json", after.snapshot)
            transition = build_native_move_transition(
                output_root=episode_root,
                transition_dir=transition_dir,
                seed=seed,
                move_index=move_index,
                semantic_segment=segment,
                before=before,
                after=after,
                native_actions_before=serialized_before,
                native_actions_after=[serialize_native_action(item) for item in action_refs],
                move_args=safe_diagnostic_value(move_args),
                move_kwargs=safe_diagnostic_value(move_kwargs),
                move_returned=returned,
                plan_success_before=plan_before,
                plan_success_after=bool(task.plan_success),
                planner_call_indices=range(planner_start + 1, len(planner.planning_calls) + 1),
                post_settle_delay_steps=0,
                enforce_native_order=False,
                task_name=task_name,
            )
            write_json(transition_dir / "transition.json", transition)
            transitions.append(transition)
            _append_jsonl(episode_root / "action_trace.jsonl", transition)
            if task_name == "insert_hole" and move_index == 1:
                capture_decision_insert()
            return returned

        task.move = types.MethodType(move_wrapper, task)

        if task_name == "lift_bottle":
            def check_mid_wrapper(_self):
                nonlocal decision_state, lift_mid_success
                counters["check_mid_success_call_count"] += 1
                if counters["check_mid_success_call_count"] != 1:
                    raise RuntimeError("check_mid_success must be called exactly once")
                returned = bool(original_check_mid())
                lift_mid_success = returned
                rel_pose = task.bottle.get_pose().rebase(task.target_pose)
                captured = capture(
                    root=episode_root / "decision",
                    phase="pre_action",
                    action_id=f"r12-{task_name}-{seed}-{condition}-decision",
                    metadata={"decision_point": "check_mid_success"},
                )
                counters["decision_snapshot_count"] += 1
                write_snapshot(episode_root / "snapshot_decision.json", captured.snapshot)
                decision_state = {
                    "schema_version": "openeta.univtac.branching_decision.v1",
                    "task": task_name,
                    "seed": seed,
                    "condition": condition,
                    "decision_point": "check_mid_success",
                    "operator_visible": captured.snapshot.operator_visible,
                    "snapshot_path": "snapshot_decision.json",
                    "host_only": {
                        "check_mid_success": returned,
                        "decision_class": classify_lift_branch(returned),
                        "mid_relative_pose": safe_diagnostic_value(rel_pose.tolist()),
                    },
                    "native_actions_before_decision": [
                        item["semantic_segment"] for item in transitions
                    ],
                    "native_next_action": None,
                    "final_native_outcome": None,
                }
                write_json(episode_root / "decision_state.json", decision_state)
                return returned

            task.check_mid_success = types.MethodType(check_mid_wrapper, task)

        result["classification"] = "native_episode_failed"
        stages.record(condition, "episode_enter")
        applied: dict[str, Any] = {}
        if condition == "expert":
            counters["play_once_call_count"] += 1
            task.play_once()
        elif task_name == "insert_hole":
            task.move(task.atom.move_by_displacement(z=-0.03), time_dilation_factor=0.2)
            if insert_native is None:
                raise RuntimeError("Insert Hole decision calculation is unavailable")
            x_move = float(insert_native["native_x_move"])
            if condition == "wrong":
                x_move = -x_move
            elif condition == "selected":
                desired_class = OPAQUE_TO_CLASS[selected_skill]
                x_move = abs(x_move) * (
                    1.0 if desired_class == "positive_x_correction" else -1.0
                )
            z_move = float(insert_native["native_z_move"])
            task.move(
                task.atom.move_by_displacement(x=x_move, z=z_move),
                time_dilation_factor=0.5,
            )
            task.move(
                task.atom.move_by_displacement(z=-0.04, xyz_coord=task.prism.get_pose()),
                time_dilation_factor=0.5,
            )
            task.delay(20, is_save=False)
            applied = {
                "applied_x_move": x_move,
                "applied_z_move": z_move,
                "final_insert_z": -0.04,
                "corrective_time_dilation": 0.5,
                "final_time_dilation": 0.5,
                "selected_skill": selected_skill,
                "canonical_decision_class": canonical_class,
                "execution_decision_class": insert_native["decision_class"],
                "execution_state_mismatch": (
                    condition == "selected"
                    and insert_native["decision_class"] != canonical_class
                ),
                "agent_world_changing_choice_count": (
                    1 if condition == "selected" else 0
                ),
            }
        else:
            task.move(task.atom.close_gripper())
            task.gripper_rotate(task.bottle, 70 / 180 * np.pi, steps=4)
            mid = bool(task.check_mid_success())
            if mid:
                raise RuntimeError("Lift Bottle counterfactual requires check_mid_success=False")
            corrective = condition == "correct"
            if corrective:
                task.move(
                    task.atom.move_by_displacement(
                        rpy=[0, np.pi / 6, 0], rpy_coord="gripper"
                    ),
                    time_dilation_factor=0.5,
                )
                task.move(
                    task.atom.move_by_displacement(
                        x=task.target_pose[0] - task.bottle.get_pose()[0] + 0.02
                    ),
                    time_dilation_factor=0.5,
                )
            task.move(task.atom.open_gripper(0.5))
            task.delay(30, is_save=False)
            applied = {"corrective_moves_executed": corrective}
        stages.record(condition, "episode_exit")

        if counters["play_once_call_count"] != (1 if condition == "expert" else 0):
            raise RuntimeError("native play_once count violates expert/counterfactual protocol")
        if counters["decision_snapshot_count"] != 1 or decision_state is None:
            raise RuntimeError("episode must capture exactly one decision snapshot")
        if task_name == "lift_bottle" and counters["check_mid_success_call_count"] != 1:
            raise RuntimeError("Lift Bottle must call native check_mid_success exactly once")
        expected_moves = 3 if task_name == "insert_hole" else (6 if lift_mid_success or condition == "wrong" else 8)
        if counters["move_call_count"] != expected_moves:
            raise RuntimeError(
                f"expected {expected_moves} native moves, observed {counters['move_call_count']}"
            )

        final_capture = capture(
            root=episode_root / "final",
            phase="post_action",
            action_id=f"r12-{task_name}-{seed}-{condition}-final",
            metadata={"stage": "post_episode_pre_evaluator", "condition": condition},
        )
        write_snapshot(episode_root / "snapshot_final.json", final_capture.snapshot)
        write_json(episode_root / "planner_diagnostics.json", json.loads(planner.to_json()))

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
        outcome = {
            "plan_success": plan_success,
            "native_check_success": native_check_success,
            "native_check_early_stop": native_check_early_stop,
            "expert_episode_success": episode_success,
        }
        transitions = [attach_native_outcome(item, outcome) for item in transitions]
        (episode_root / "action_trace.jsonl").unlink(missing_ok=True)
        for transition in transitions:
            _append_jsonl(episode_root / "action_trace.jsonl", transition)
        decision_state["native_next_action"] = (
            transitions[len(decision_state["native_actions_before_decision"])]
            if len(transitions) > len(decision_state["native_actions_before_decision"])
            else None
        )
        decision_state["final_native_outcome"] = outcome
        write_json(episode_root / "decision_state.json", decision_state)
        decision_host = decision_state["host_only"]
        final_result = {
            "schema_version": "openeta.univtac.branching_result.v1",
            "task": task_name,
            "seed": seed,
            "condition": condition,
            **outcome,
            **applied,
            "decision_class": decision_host["decision_class"],
            "check_mid_success": decision_host.get("check_mid_success"),
            "native_x_move": decision_host.get("native_x_move"),
            "native_z_move": decision_host.get("native_z_move"),
            "transition_count": len(transitions),
            "semantic_segments": [item["semantic_segment"] for item in transitions],
            "decision_snapshot_complete": True,
            "action_trace_complete": len(transitions) == expected_moves,
            "final_simulator_step": int(task.step_count),
            "final_take_action_count": int(task.take_action_cnt),
        }
        write_json(episode_root / "final_result.json", final_result)
        write_json(
            episode_root / "episode.json",
            {
                "schema_version": "openeta.univtac.branching_episode.v1",
                "round": "R1.3" if is_r13 else "R1.2",
                "task": task_name,
                "seed": seed,
                "condition": condition,
                "status": "completed",
                "duration_seconds": time.monotonic() - started,
                "counters": counters,
                "native_outcome": outcome,
                "operator_visible_decision_label_exposed": False,
            },
        )
        result.update(
            {
                "status": "completed",
                "classification": (
                    "native_episode_success" if episode_success else "native_episode_task_failure"
                ),
                "decision_state": decision_state,
                "final_result": final_result,
            }
        )
    except Exception as exc:  # noqa: BLE001 - persist simulator evidence
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
                    episode_root / "planner_diagnostics.json", json.loads(planner.to_json())
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

    return 0 if result["status"] == "completed" and cleanup["task_close"] else 1


def copy_dict(value: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(value, allow_nan=False))


if __name__ == "__main__":
    raise SystemExit(main())
