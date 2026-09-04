#!/usr/bin/env python3
"""Serve one live Pull Out Key task through a bounded loopback skill API."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any


def _parse_base_args(
    argv: list[str] | None,
) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=0)
    base, _ = parser.parse_known_args(argv)
    return parser, base


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


def _read_request(handler: BaseHTTPRequestHandler) -> dict[str, Any]:
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length)
    payload = json.loads(raw.decode("utf-8")) if raw else {}
    if not isinstance(payload, dict):
        raise TypeError("worker request must be a JSON object")
    return payload


def _respond(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


def main(argv: list[str] | None = None) -> int:
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    parser, base = _parse_base_args(argv)
    source_root = base.source_root.expanduser().resolve()
    repo_root = base.repo_root.expanduser().resolve()
    output_root = base.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    task = None
    simulation_app = None
    server = None
    planner = None
    cleanup = {
        "task_close": False,
        "server_close": False,
        "simulation_app_close_invoked": False,
        "simulation_app_close_returned": False,
    }
    counters = {
        "reset_call_count": 0,
        "play_once_call_count": 0,
        "observe_call_count": 0,
        "execute_skill_call_count": 0,
        "world_changing_skill_count": 0,
        "finish_episode_call_count": 0,
        "check_success_call_count": 0,
        "check_early_stop_call_count": 0,
        "over_rotate_sample_count": 0,
    }
    result: dict[str, Any] = {
        "schema_version": "openeta.univtac.live_worker_episode.v1",
        "status": "starting",
        "classification": "app_launcher_failed",
        "task": "pull_out_key",
        "seed": 1_000_000,
        "started_at": started_at,
        "counters": counters,
        "cleanup": cleanup,
        "selected_skill_sequence": [],
        "final_result": None,
        "error": None,
    }

    try:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args(argv)
        args.enable_cameras = True
        args.num_envs = 1
        args.device = "cuda:0"
        simulation_app = AppLauncher(args).app
        if simulation_app is None:
            raise RuntimeError("AppLauncher returned no simulation_app")

        for path in (repo_root, source_root):
            while str(path) in sys.path:
                sys.path.remove(str(path))
            sys.path.insert(0, str(path))

        import tacex
        import tacex_assets
        import tacex_uipc
        import yaml
        from envs.utils.env_parser import build_task_env_cfg, load_task_config

        from sim.envs.univtac.contract import UniVTACContractError, validate_operator_visible
        from sim.envs.univtac.live_operation import (
            LIVE_SEED,
            PullOutKeySkillBudget,
            validate_live_operation_config,
            validate_worker_projection,
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
        from sim.envs.univtac.pull_out_key_gate import summarize_pull_out_key_observation
        from sim.envs.univtac.trace import verify_artifacts, write_json, write_snapshot

        config_payload = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if not isinstance(config_payload, dict):
            raise TypeError("live operation config must be a mapping")
        config = validate_live_operation_config(config_payload)

        native_config, native_config_path = load_task_config(
            source_root / "task_config" / f"{config['task_config']}.yml"
        )
        task_module, env_cfg, timing, native_save_dir = build_task_env_cfg(
            config["task"],
            native_config,
            config["task_config"],
            config["mode"],
            device=config["device"],
            save_dir=output_root / "native",
        )
        env_cfg.save_frequency = 0
        env_cfg.video_frequency = 0
        module_paths = {
            "envs.pull_out_key": str(Path(task_module.__file__).resolve()),
            "tacex": str(Path(tacex.__file__).resolve()),
            "tacex_assets": str(Path(tacex_assets.__file__).resolve()),
            "tacex_uipc": str(Path(tacex_uipc.__file__).resolve()),
        }
        allowed_roots = (source_root, source_root / "third_party" / "TacEx" / "source")
        for name, value in module_paths.items():
            if not any(Path(value).is_relative_to(root) for root in allowed_roots):
                raise RuntimeError(f"{name} resolved outside pinned source: {value}")
        write_json(output_root / "module_realpaths.json", module_paths)

        task = task_module.Task(env_cfg, mode=config["mode"])
        planner = PlannerDiagnosticRecorder(task)
        planner.install()
        planner.start_seed(LIVE_SEED)
        counters["reset_call_count"] += 1
        result["classification"] = "live_reset_failed"
        task.reset(seed=LIVE_SEED)
        if not task.plan_success:
            raise RuntimeError("live reset/pre_move returned with plan_success=False")

        budget = PullOutKeySkillBudget(max_calls=config["max_world_changing_skills"])
        counters["over_rotate_sample_count"] += 1
        over_rotate = float(task.rng.uniform(0.09, 0.16))
        task.metadata["over_rotate"] = over_rotate
        state: dict[str, Any] = {
            "finished": False,
            "finish_reason": None,
            "initial_observe_completed": False,
            "observation_index": 0,
            "selected_skill_sequence": [],
            "transitions": [],
            "fatal_error": None,
        }

        def capture_projection(
            *,
            phase: str,
            action_id: str,
            visible_phase: str,
            skill_execution: dict[str, Any],
            root: Path | None = None,
        ) -> tuple[Any, dict[str, Any]]:
            state["observation_index"] += 1
            observation = task._get_observations()
            observation_summary, contact_summary = summarize_pull_out_key_observation(
                observation
            )
            snapshot_root = root or (
                output_root
                / "observations"
                / f"{state['observation_index']:03d}_{visible_phase}"
            )
            captured = capture_snapshot(
                observation,
                output_root=output_root,
                seed_dir=snapshot_root,
                task_name="pull_out_key",
                seed=LIVE_SEED,
                phase=phase,
                action_id=action_id,
                simulator_step=int(task.step_count),
                take_action_count=int(task.take_action_cnt),
                task_instruction=config["task_instruction"],
                task_metadata={
                    "visible_phase": visible_phase,
                    "plan_success": bool(task.plan_success),
                    "contact_summary": contact_summary,
                },
                native_check_success=None,
                save_host_only=True,
                strict_two_tactile_sensors=True,
                fail_on_missing_rgb_marker=True,
            )
            validate_operator_visible(captured.snapshot.operator_visible)
            verify_artifacts(output_root, captured.snapshot.to_dict())
            write_snapshot(snapshot_root / f"snapshot_{phase}.json", captured.snapshot)
            write_json(snapshot_root / f"observation_{phase}.json", observation_summary)
            visible = captured.snapshot.operator_visible
            images = []
            for camera_name in ("head", "wrist"):
                descriptor = visible["cameras"][camera_name]["rgb"]
                images.append(
                    {
                        "label": f"camera/{camera_name}/rgb",
                        "path": descriptor["path"],
                        "shape": descriptor["shape"],
                        "dtype": descriptor["dtype"],
                    }
                )
            for sensor_name in ("left_tactile", "right_tactile"):
                descriptor = visible["tactile"][sensor_name]["rgb_marker"]
                images.append(
                    {
                        "label": f"tactile/{sensor_name}/rgb_marker",
                        "path": descriptor["path"],
                        "shape": descriptor["shape"],
                        "dtype": descriptor["dtype"],
                    }
                )
            projection = validate_worker_projection(
                {
                    "task_instruction": visible["task_instruction"],
                    "visible_phase": visible_phase,
                    "step_identifiers": visible["step_identifiers"],
                    "proprio": visible["proprio"],
                    "images": images,
                    "skill_execution": skill_execution,
                    "remaining_skills": budget.remaining,
                }
            )
            write_json(snapshot_root / "operator_projection.json", projection)
            return captured, projection

        def native_skill_call(skill: str):
            if skill == "align_key":
                actions = task.atom.move_by_displacement(
                    rpy=[0, 0, -task.key_rotation + over_rotate], xyz_coord="local"
                )
                kwargs = {
                    "time_dilation_factor": 0.5,
                    "constraint_pose": [0, 0, 0, 1, 1, 1],
                    "delay": False,
                }
                post_delay = 0
            elif skill == "settle_alignment":
                actions = task.atom.move_by_displacement(
                    rpy=[0, 0, -over_rotate + 0.05], xyz_coord="local"
                )
                kwargs = {
                    "constraint_pose": [0, 0, 0, 1, 1, 1],
                    "delay": False,
                }
                post_delay = 0
            elif skill == "pull_key_out":
                actions = task.atom.move_by_displacement(z=-0.03, xyz_coord="local")
                kwargs = {"time_dilation_factor": 0.2}
                post_delay = 20
            else:
                raise UniVTACContractError(f"unknown Pull Out Key skill: {skill}")
            return list(actions), kwargs, post_delay

        def execute_skill(skill: str) -> dict[str, Any]:
            call_index = budget.reserve(skill)
            counters["execute_skill_call_count"] += 1
            counters["world_changing_skill_count"] += 1
            state["selected_skill_sequence"].append(skill)
            transition_root = output_root / "transitions" / f"{call_index:02d}_{skill}"
            action_id = f"live-{LIVE_SEED}-{call_index}-{skill}"
            before, _ = capture_projection(
                phase="pre_action",
                action_id=action_id,
                visible_phase=f"before_{skill}",
                skill_execution={"skill": skill, "status": "pending"},
                root=transition_root,
            )
            write_snapshot(transition_root / "snapshot_before.json", before.snapshot)
            actions, kwargs, post_delay = native_skill_call(skill)
            actions_before = [serialize_native_action(action) for action in actions]
            plan_before = bool(task.plan_success)
            planner_start = len(planner.planning_calls)
            move_returned = task.move(actions, **kwargs)
            if post_delay:
                task.delay(post_delay)
            actions_after = [serialize_native_action(action) for action in actions]
            status = "completed" if move_returned and task.plan_success else "failed"
            after, projection = capture_projection(
                phase="post_action",
                action_id=action_id,
                visible_phase=f"after_{skill}",
                skill_execution={"skill": skill, "status": status},
                root=transition_root,
            )
            write_snapshot(transition_root / "snapshot_after.json", after.snapshot)
            transition = build_native_move_transition(
                output_root=output_root,
                transition_dir=transition_root,
                seed=LIVE_SEED,
                move_index=call_index,
                semantic_segment=skill,
                before=before,
                after=after,
                native_actions_before=actions_before,
                native_actions_after=actions_after,
                move_args=[],
                move_kwargs=safe_diagnostic_value(kwargs),
                move_returned=move_returned,
                plan_success_before=plan_before,
                plan_success_after=bool(task.plan_success),
                planner_call_indices=range(planner_start + 1, len(planner.planning_calls) + 1),
                post_settle_delay_steps=post_delay,
                enforce_native_order=False,
            )
            write_json(transition_root / "transition.json", transition)
            state["transitions"].append(transition)
            _append_jsonl(output_root / "action_trace.jsonl", transition)
            return projection

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, _format: str, *log_args: Any) -> None:
                return

            def do_POST(self) -> None:
                request: dict[str, Any] = {}
                try:
                    request = _read_request(self)
                    if self.path == "/observe":
                        if request:
                            raise UniVTACContractError("observe accepts no arguments")
                        if state["initial_observe_completed"]:
                            raise UniVTACContractError("initial observe may be called exactly once")
                        state["initial_observe_completed"] = True
                        counters["observe_call_count"] += 1
                        _, projection = capture_projection(
                            phase="pre_action",
                            action_id=f"live-{LIVE_SEED}-initial-observe",
                            visible_phase="pre_action",
                            skill_execution={"skill": None, "status": "not_started"},
                        )
                        response = {"ok": True, "observation": projection}
                    elif self.path == "/execute":
                        if set(request) != {"skill"} or not isinstance(request["skill"], str):
                            raise UniVTACContractError("execute requires only a string skill")
                        projection = execute_skill(request["skill"])
                        response = {"ok": True, "observation": projection}
                    elif self.path in {"/finish", "/host_finalize"}:
                        if request:
                            raise UniVTACContractError("finish accepts no arguments")
                        if self.path == "/finish":
                            counters["finish_episode_call_count"] += 1
                            state["finish_reason"] = "codex_finish_episode"
                        else:
                            state["finish_reason"] = "parent_finalize"
                        state["finished"] = True
                        response = {
                            "ok": True,
                            "status": "finish_accepted_host_evaluation_pending",
                            "executed_skills": list(budget.executed),
                            "remaining_skills": budget.remaining,
                        }
                    else:
                        raise UniVTACContractError(f"unknown worker endpoint: {self.path}")
                    _append_jsonl(
                        output_root / "worker_requests.jsonl",
                        {"path": self.path, "request": request, "ok": True},
                    )
                    _respond(self, 200, response)
                except UniVTACContractError as exc:
                    _append_jsonl(
                        output_root / "worker_requests.jsonl",
                        {"path": self.path, "request": request, "ok": False, "error": str(exc)},
                    )
                    _respond(self, 400, {"ok": False, "error": str(exc)})
                except Exception as exc:  # noqa: BLE001 - preserve live failure
                    state["fatal_error"] = {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                    state["finished"] = True
                    _respond(self, 500, {"ok": False, "error": str(exc)})

        server = HTTPServer(("127.0.0.1", args.port), Handler)
        server.timeout = 1.0
        worker_url = f"http://127.0.0.1:{server.server_address[1]}"
        result.update(
            {
                "status": "ready",
                "classification": "live_worker_ready",
                "worker_url": worker_url,
                "task_config": {
                    "native_config_path": str(native_config_path),
                    "timing": repr(timing),
                    "native_save_dir": str(native_save_dir),
                },
            }
        )
        _write_json(
            output_root / "ready.json",
            {
                "status": "ready",
                "worker_url": worker_url,
                "task": "pull_out_key",
                "seed": LIVE_SEED,
                "mcp_tools": ["observe", "execute_skill", "finish_episode"],
            },
        )
        _write_json(output_root / "child_result.json", result)

        deadline = time.monotonic() + config["worker_timeout_seconds"]
        while not state["finished"] and time.monotonic() < deadline:
            server.handle_request()
        if not state["finished"]:
            raise TimeoutError("live worker timed out before finish_episode")
        if state["fatal_error"] is not None:
            raise RuntimeError(f"live worker request failed: {state['fatal_error']['error']}")

        plan_success = bool(task.plan_success)
        counters["check_success_call_count"] += 1
        native_check_success = bool(task.check_success())
        native_check_early_stop: bool | None = None
        if plan_success and native_check_success:
            counters["check_early_stop_call_count"] += 1
            native_check_early_stop = bool(task.check_early_stop())
        operation_success = expert_episode_success(
            plan_success=plan_success,
            native_check_success=native_check_success,
            native_check_early_stop=native_check_early_stop,
        )
        outcome = {
            "plan_success": plan_success,
            "native_check_success": native_check_success,
            "native_check_early_stop": native_check_early_stop,
            "agent_operation_smoke_success": operation_success,
        }
        state["transitions"] = [
            attach_native_outcome(transition, outcome)
            for transition in state["transitions"]
        ]
        (output_root / "action_trace.jsonl").unlink(missing_ok=True)
        for index, transition in enumerate(state["transitions"], start=1):
            write_json(
                output_root
                / "transitions"
                / f"{index:02d}_{transition['semantic_segment']}"
                / "transition.json",
                transition,
            )
            _append_jsonl(output_root / "action_trace.jsonl", transition)
        write_json(output_root / "planner_diagnostics.json", json.loads(planner.to_json()))
        final_result = {
            "schema_version": "openeta.univtac.live_operation_result.v1",
            "task": "pull_out_key",
            "seed": LIVE_SEED,
            "selected_skill_sequence": list(state["selected_skill_sequence"]),
            "world_changing_skill_count": counters["world_changing_skill_count"],
            "action_budget": config["max_world_changing_skills"],
            "finish_reason": state["finish_reason"],
            **outcome,
        }
        write_json(output_root / "final_result.json", final_result)
        result.update(
            {
                "status": "completed",
                "classification": "live_agent_episode_completed",
                "selected_skill_sequence": list(state["selected_skill_sequence"]),
                "final_result": final_result,
                "ended_at": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": time.monotonic() - started,
            }
        )
    except Exception as exc:  # noqa: BLE001 - persist native live failures
        result["status"] = "failed"
        result["error"] = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        _write_json(output_root / "exception.json", result["error"])
    finally:
        if server is not None:
            server.server_close()
            cleanup["server_close"] = True
        if task is not None:
            try:
                task.close()
                cleanup["task_close"] = True
            except Exception as exc:  # noqa: BLE001
                result["classification"] = "cleanup_incomplete"
                result.setdefault("cleanup_errors", []).append(str(exc))
        result["cleanup"] = cleanup
        result["counters"] = counters
        _write_json(output_root / "child_result.json", result)
        if simulation_app is not None:
            cleanup["simulation_app_close_invoked"] = True
            _write_json(output_root / "child_result.json", result)
            try:
                simulation_app.close()
                cleanup["simulation_app_close_returned"] = True
            except Exception as exc:  # noqa: BLE001
                result["classification"] = "cleanup_incomplete"
                result.setdefault("cleanup_errors", []).append(str(exc))
        result["cleanup"] = cleanup
        _write_json(output_root / "child_result.json", result)

    completed = result["status"] == "completed" and cleanup["task_close"]
    return 0 if completed else 1


if __name__ == "__main__":
    raise SystemExit(main())
