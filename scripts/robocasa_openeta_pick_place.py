#!/usr/bin/env python3
"""Run one fixed-base RoboCasa pick/place attempt with OpenETA skills/tools.

The task checker is always RoboCasa's native checker.  The policy first tries
the configured OpenETA perception stack (SAM3, AnyGrasp, AnyPlace).  This
machine currently lacks the AnyGrasp / AnyPlace SDKs, so their structured
failures are retained and a clearly-labelled privileged pose fallback is used
to continue the physical rollout.  A video and JSON trace are written even if
the rollout raises or the checker never succeeds.
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path
from typing import Any, Callable

import imageio.v2 as imageio
import numpy as np

from adapter.protocol import EnvObservation
from agent.backends.planner import StaticPlannerBackend
from agent.runtime.planner import ToolCallingPlanner
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.tools.handlers import (
    build_anygrasp_handler,
    build_anyplace_handler,
    build_sam3_handler,
    build_sse_sam3_mcp_segmenter,
)
from agent.tools.registry import (
    ToolExecutionContext,
    ToolRegistry,
    ToolResult,
    build_default_tool_registry,
    make_tool_result,
)
from sim.envs.robocasa.direct_env import RoboCasaDirectEnv


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _world_to_base(vector: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = quat_xyzw
    rotation = np.asarray(
        [
            [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
            [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
            [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
        ],
        dtype=float,
    )
    return rotation.T @ vector


def _eef_position(env: RoboCasaDirectEnv) -> np.ndarray:
    raw = env.unwrapped_env
    return raw.sim.data.site_xpos[raw.robots[0].eef_site_id["right"]].copy()


def _base_quaternion(env: RoboCasaDirectEnv) -> np.ndarray:
    raw = env.unwrapped_env
    body_id = raw.sim.model.body_name2id(raw.robots[0].robot_model.root_body)
    return np.roll(raw.sim.data.body_xquat[body_id], -1).copy()


def _object_position(env: RoboCasaDirectEnv, name: str = "obj") -> np.ndarray:
    raw = env.unwrapped_env
    obj = raw.objects[name]
    return raw.sim.data.body_xpos[raw.sim.model.body_name2id(obj.root_body)].copy()


def _workspace_frame(env: RoboCasaDirectEnv, *, width: int, height: int) -> np.ndarray:
    """Render a fixed workcell overview derived from robot-base proprioception."""
    raw = env.unwrapped_env
    base = raw.sim.data.body_xpos[
        raw.sim.model.body_name2id(raw.robots[0].robot_model.root_body)
    ].copy()
    context = raw.sim._render_context_offscreen
    # robosuite's observation render switches this shared camera back to a
    # named fixed camera after every env.step().  Restore the free-camera mode
    # before applying the workcell overview parameters.
    context.cam.type = 0  # mujoco.mjtCamera.mjCAMERA_FREE
    context.cam.fixedcamid = -1
    lookat = getattr(
        env,
        "_openeta_video_lookat",
        base + np.asarray([-0.50, 0.0, 0.30]),
    )
    context.cam.lookat[:] = np.asarray(lookat, dtype=float)
    context.cam.distance = 1.35
    context.cam.azimuth = 115.0
    context.cam.elevation = -28.0
    context.render(width, height, camera_id=-1)
    return np.flipud(np.asarray(context.read_pixels(width, height)))[..., :3].copy()


def _video_frame(env: RoboCasaDirectEnv, obs: dict[str, Any], *, size: int) -> np.ndarray:
    import cv2

    snapshots = getattr(env, "_openeta_state_snapshots", None)
    if isinstance(snapshots, list):
        raw = env.unwrapped_env
        snapshots.append(
            {
                "qpos": np.asarray(raw.sim.data.qpos, dtype=float).copy(),
                "qvel": np.asarray(raw.sim.data.qvel, dtype=float).copy(),
            }
        )
    overview = _workspace_frame(env, width=size, height=size)
    wrist = np.flipud(np.asarray(obs["robot0_eye_in_hand_image"]))[..., :3].copy()
    wrist = cv2.resize(wrist, (size, size), interpolation=cv2.INTER_AREA)
    return np.concatenate([overview, wrist], axis=1)


def _observation(obs: dict[str, Any], step_idx: int) -> EnvObservation:
    return EnvObservation.from_dict(
        {
            "task_description": str(obs.get("_openeta_task_description", "")),
            "proprio": {
                "eef_pose": np.concatenate(
                    [np.asarray(obs["robot0_eef_pos"]), np.asarray(obs["robot0_eef_quat"])]
                ).tolist(),
                "base_pose": {
                    "xyz": np.asarray(obs["robot0_base_pos"]).tolist(),
                    "quat_xyzw": np.asarray(obs["robot0_base_quat"]).tolist(),
                },
            },
            "metadata": {"step_idx": step_idx, "benchmark": "robocasa365"},
        }
    )


def _raise_missing(name: str, root_hint: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    def missing(_: dict[str, Any]) -> dict[str, Any]:
        raise RuntimeError(f"{name} unavailable: missing SDK / weights at {root_hint}")
    return missing


def run(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    video_path = out / "episode.mp4"
    trace_path = out / "trace.json"
    result_path = out / "result.json"
    experiment_path = out / "experiment.md"
    perception_dir = out / "perception"
    perception_dir.mkdir()

    frames: list[np.ndarray] = []
    trace: list[dict[str, Any]] = []
    total_steps = 0
    reward = 0.0
    terminated = truncated = False
    info: dict[str, Any] = {}
    obs: dict[str, Any] = {}
    reset_info: dict[str, Any] = {}
    fallback_used = False
    tools: ToolRegistry = build_default_tool_registry()
    env = RoboCasaDirectEnv(
        "PickPlaceCounterToSink",
        robot="Panda",
        split="target",
        seed=args.seed,
        image_width=args.size,
        image_height=args.size,
        camera_depths=True,
    )

    def record_tool(name: str, parameters: dict[str, Any]) -> ToolResult:
        result = tools.call(name, parameters, observation=_observation(obs, total_steps))
        trace.append(
            {
                "turn": len(trace) + 1,
                "kind": "tool_call",
                "name": name,
                "parameters": parameters,
                "result": {
                    "success": result.success,
                    "content": result.content,
                    "details": result.details,
                },
            }
        )
        return result

    def bind(name: str, handler: Callable[[ToolExecutionContext], ToolResult]) -> None:
        tools.bind_handler(name, handler, replace=tools.can_execute(name))

    try:
        obs, reset_info = env.reset(seed=args.seed)
        raw = env.unwrapped_env
        env._openeta_state_snapshots = []
        env._openeta_video_lookat = np.asarray(obs["robot0_base_pos"], dtype=float) + np.asarray(
            [-0.50, 0.0, 0.30]
        )
        frames.extend([_video_frame(env, obs, size=args.size)] * 10)

        # Exercise the real OpenETA runtime and editable skill registry.  Skill
        # calls select guidance only; atomic calls below remain explicit.
        runtime = OpenEtaAgentRuntime(
            planner=ToolCallingPlanner(
                StaticPlannerBackend(
                    [
                        {
                            "kind": "tool_call",
                            "name": "skill_call",
                            "parameters": {"name": "pick", "target": "mug"},
                            "reasoning": "Select the OpenETA pick guidance.",
                        },
                        {
                            "kind": "tool_call",
                            "name": "skill_call",
                            "parameters": {"name": "place", "target": "sink"},
                            "reasoning": "Select the OpenETA place guidance.",
                        },
                    ]
                )
            ),
            tools=tools,
        )
        runtime.start_session(
            task=str(obs.get("_openeta_task_description", "")),
            metadata={"benchmark": "robocasa365", "robot": "Panda", "seed": args.seed},
        )
        for skill_name in ("pick", "place"):
            action = runtime.act(_observation(obs, total_steps))
            trace.append(
                {
                    "turn": len(trace) + 1,
                    "kind": "openeta_runtime_action",
                    "selected_skill": skill_name,
                    "action": action.to_dict(),
                }
            )

        # Save a generic workcell image for the real SAM3 call.  The first-party
        # fixed Panda agentview is separately known to be self-occluded.
        rgb_path = perception_dir / "workspace_rgb.png"
        imageio.imwrite(rgb_path, _workspace_frame(env, width=args.size, height=args.size))

        bind(
            "observe",
            lambda context: make_tool_result(
                context,
                success=True,
                content="workspace RGB-only observation captured",
                outputs={
                    "rgb": str(rgb_path.resolve()),
                    "depth_available": False,
                    "diagnostic": "custom overview renderer has no calibrated metric depth",
                },
                artifacts=[{"type": "rgb", "path": str(rgb_path.resolve())}],
            ),
        )
        record_tool("observe", {"camera": "workspace_overview"})

        bind(
            "sam3",
            build_sam3_handler(
                build_sse_sam3_mcp_segmenter(url=args.sam3_url),
                output_root=perception_dir / "sam3",
                result_output_root=perception_dir / "sam3",
            ),
        )
        sam3_obj = record_tool("sam3", {"image": str(rgb_path.resolve()), "prompt": "mug"})
        sam3_sink = record_tool("sam3", {"image": str(rgb_path.resolve()), "prompt": "sink basin"})

        # Bind real-format handlers to unavailable backends so the trace keeps
        # the same structured contract as a deployed installation.
        bind(
            "anygrasp",
            build_anygrasp_handler(
                _raise_missing("AnyGrasp", "configured AnyGrasp backend"),
                output_root=perception_dir / "anygrasp",
            ),
        )
        bind(
            "anyplace",
            build_anyplace_handler(
                _raise_missing("AnyPlace", "configured AnyPlace backend"),
                output_root=perception_dir / "anyplace",
            ),
        )
        grasp = record_tool(
            "anygrasp",
            {
                "mode": "targeted",
                "rgb": str(rgb_path.resolve()),
                "target_mask": "sam3 mask" if sam3_obj.success else "unavailable",
            },
        )
        placement = record_tool(
            "anyplace",
            {
                "rgb": str(rgb_path.resolve()),
                "object_mask": "sam3 mug mask" if sam3_obj.success else "unavailable",
                "placement_region_mask": {"mask_ref": "sam3 sink mask"} if sam3_sink.success else {},
                "selected_grasp": {},
            },
        )

        # Without valid RGB-D grasp and placement transforms, continue with an
        # explicitly privileged calibration fallback.  Native checker output
        # remains the only success criterion.
        fallback_used = not (grasp.success and placement.success)
        obj_xyz = _object_position(env)
        sink_xyz = np.asarray(raw.sink.pos, dtype=float).copy()
        sink_xyz[2] = max(float(sink_xyz[2]), 0.99)
        base_xyz = np.asarray(obs["robot0_base_pos"], dtype=float)
        base_quat = _base_quaternion(env)
        targets = {
            "pick_approach": obj_xyz + np.asarray([0.0, 0.0, 0.16]),
            "pick_contact": obj_xyz + np.asarray(
                [args.grasp_offset_x, args.grasp_offset_y, args.grasp_offset_z]
            ),
            "pick_lift": obj_xyz + np.asarray([0.0, 0.0, 0.22]),
            "place_approach": sink_xyz + np.asarray([0.0, 0.0, 0.22]),
            # The fixed Panda becomes stiff while carrying the mug into the
            # basin.  Release from the centered, collision-free hover and let
            # gravity settle the object before retreating.
            "place_release": sink_xyz + np.asarray([0.0, 0.0, 0.22]),
            "retreat": sink_xyz + np.asarray([0.20, 0.0, 0.28]),
        }

        def ik_handler(context: ToolExecutionContext) -> ToolResult:
            target = np.asarray(context.parameters.get("target_xyz", []), dtype=float)
            radial = float(np.linalg.norm(target[:2] - base_xyz[:2])) if target.size >= 3 else 99.0
            allowed = bool(target.size >= 3 and radial < 0.82 and 0.78 < target[2] < 1.35)
            return make_tool_result(
                context,
                success=allowed,
                content="workspace reachability preview passed" if allowed else "target outside calibrated workspace",
                outputs={"radial_distance_m": radial, "allowed": allowed},
            )

        def obstacle_handler(context: ToolExecutionContext) -> ToolResult:
            target = np.asarray(context.parameters.get("target_xyz", []), dtype=float)
            allowed = bool(target.size >= 3 and target[2] >= 0.94)
            return make_tool_result(
                context,
                success=allowed,
                content="height gate passed" if allowed else "target below counter safety floor",
                outputs={"allowed": allowed, "minimum_z": 0.94},
            )

        def move_handler(context: ToolExecutionContext) -> ToolResult:
            nonlocal obs, reward, terminated, truncated, info, total_steps
            target = np.asarray(context.parameters["target_xyz"], dtype=float)
            stage_start = total_steps
            for _ in range(args.max_steps_per_move):
                error = target - _eef_position(env)
                if np.max(np.abs(error)) < args.tolerance:
                    break
                action = np.zeros(7, dtype=np.float32)
                action[:3] = np.clip(
                    _world_to_base(error, base_quat) / args.action_scale, -1.0, 1.0
                )
                action[6] = -1.0 if context.parameters.get("gripper", "open") == "open" else 1.0
                obs, reward, terminated, truncated, info = env.step(action)
                total_steps += 1
                if total_steps % args.frame_stride == 0:
                    frames.append(_video_frame(env, obs, size=args.size))
                if terminated or truncated:
                    break
            end = _eef_position(env)
            # Contact motion may stop a few centimetres early because the
            # fingers have physically reached the object; that is a valid
            # terminal condition for the following close command.
            stage_name = context.parameters.get("stage")
            if stage_name == "pick_contact":
                reach_tolerance = args.contact_tolerance
            elif stage_name in ("place_approach", "place_release"):
                reach_tolerance = args.carrying_tolerance
            else:
                reach_tolerance = args.tolerance * 1.5
            reached = bool(np.max(np.abs(target - end)) < reach_tolerance)
            return make_tool_result(
                context,
                success=reached,
                content="closed-loop OSC move completed" if reached else "move budget ended before tolerance",
                state_delta={
                    "start_step": stage_start,
                    "end_step": total_steps,
                    "end_eef_xyz": end.tolist(),
                    "object_xyz": _object_position(env).tolist(),
                    "object_minus_eef_xyz": (_object_position(env) - end).tolist(),
                    "native_success": bool(raw._check_success()),
                },
            )

        def gripper_handler(context: ToolExecutionContext) -> ToolResult:
            nonlocal obs, reward, terminated, truncated, info, total_steps
            command = str(context.parameters.get("command", "open"))
            action = np.zeros(7, dtype=np.float32)
            action[6] = -1.0 if command == "open" else 1.0
            for _ in range(args.gripper_steps):
                obs, reward, terminated, truncated, info = env.step(action)
                total_steps += 1
                if total_steps % args.frame_stride == 0:
                    frames.append(_video_frame(env, obs, size=args.size))
                if terminated or truncated:
                    break
            gripper = raw.robots[0].gripper["right"]
            contact_geoms = set(gripper.contact_geoms)
            contacts: list[list[str]] = []
            for index in range(raw.sim.data.ncon):
                contact = raw.sim.data.contact[index]
                name1 = raw.sim.model.geom_id2name(contact.geom1) or str(contact.geom1)
                name2 = raw.sim.model.geom_id2name(contact.geom2) or str(contact.geom2)
                if name1 in contact_geoms or name2 in contact_geoms:
                    contacts.append([name1, name2])
            return make_tool_result(
                context,
                success=True,
                content=f"gripper {command} command executed",
                state_delta={
                    "object_xyz": _object_position(env).tolist(),
                    "object_minus_eef_xyz": (_object_position(env) - _eef_position(env)).tolist(),
                    "gripper_qpos": np.asarray(obs.get("robot0_gripper_qpos", [])).tolist(),
                    "gripper_contacts": contacts,
                    "native_success": bool(raw._check_success()),
                },
            )

        bind("ik_preview_check", ik_handler)
        bind("obstacle_avoidance", obstacle_handler)
        bind("move_to", move_handler)
        bind("gripper_control", gripper_handler)

        record_tool("gripper_control", {"command": "open"})
        for stage in (
            "pick_approach",
            "pick_contact",
            "pick_lift",
            "place_approach",
            "place_release",
            "retreat",
        ):
            target = targets[stage]
            for safety_name in ("ik_preview_check", "obstacle_avoidance"):
                check = record_tool(safety_name, {"stage": stage, "target_xyz": target.tolist()})
                if not check.success:
                    raise RuntimeError(f"{safety_name} rejected {stage}: {check.content}")
            held = stage not in ("pick_approach", "pick_contact", "retreat")
            move_result = record_tool(
                "move_to",
                {"stage": stage, "target_xyz": target.tolist(), "gripper": "closed" if held else "open"},
            )
            frames.append(_video_frame(env, obs, size=args.size))
            if not move_result.success:
                raise RuntimeError(f"move_to failed at {stage}: {move_result.content}")
            if stage == "pick_contact":
                record_tool("gripper_control", {"command": "close"})
            if stage == "place_release":
                record_tool("gripper_control", {"command": "open"})

            # RoboCasa terminates immediately when its native checker turns
            # true.  Do not issue another actuator command into a completed
            # episode merely to satisfy the nominal retreat waypoint.
            if bool(info.get("success", False)) or bool(raw._check_success()):
                break

        frames.extend([_video_frame(env, obs, size=args.size)] * 14)
        success = bool(info.get("success", False) or raw._check_success())
        stop_reason = "native_success" if success else "native_checker_not_satisfied"
    except Exception as exc:  # always retain video and trace
        success = bool(info.get("success", False))
        try:
            success = success or bool(env.unwrapped_env._check_success())
        except Exception:
            pass
        stop_reason = "exception"
        trace.append(
            {
                "turn": len(trace) + 1,
                "kind": "exception",
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        if obs:
            frames.extend([_video_frame(env, obs, size=args.size)] * 14)
    finally:
        if not frames:
            frames.append(np.zeros((args.size, args.size * 2, 3), dtype=np.uint8))
        # Re-render the recorded simulator states after the control rollout.
        # There are no intervening env.step() camera renders during this pass,
        # so the fixed free-camera overview remains authoritative and stable.
        snapshots = getattr(env, "_openeta_state_snapshots", [])
        if snapshots:
            replay_frames: list[np.ndarray] = []
            replay_env: RoboCasaDirectEnv | None = None
            try:
                # Use an isolated renderer/model instance.  The rollout's
                # named camera observations mutate robosuite's shared render
                # state; a fresh env has not executed those control-time
                # renders and can replay all saved qpos/qvel states cleanly.
                replay_env = RoboCasaDirectEnv(
                    "PickPlaceCounterToSink",
                    robot="Panda",
                    split="target",
                    seed=args.seed,
                    image_width=64,
                    image_height=64,
                    camera_depths=False,
                )
                replay_obs, _ = replay_env.reset(seed=args.seed)
                replay_env._openeta_video_lookat = np.asarray(
                    replay_obs["robot0_base_pos"], dtype=float
                ) + np.asarray([-0.50, 0.0, 0.30])
                replay_raw = replay_env.unwrapped_env
                for snapshot in snapshots:
                    if len(replay_raw.sim.data.qpos) != len(snapshot["qpos"]):
                        raise RuntimeError("replay model qpos shape differs from rollout model")
                    replay_raw.sim.data.qpos[:] = snapshot["qpos"]
                    replay_raw.sim.data.qvel[:] = snapshot["qvel"]
                    replay_raw.sim.forward()
                    replay_frames.append(
                        _workspace_frame(replay_env, width=args.size, height=args.size)
                    )
                replay_frames.extend([replay_frames[-1].copy()] * 14)
                frames = replay_frames
            finally:
                if replay_env is not None:
                    replay_env.close()
        imageio.mimsave(video_path, frames, fps=args.fps, macro_block_size=None)
        try:
            env.close()
        except Exception:
            pass

    result = {
        "schema_version": "openeta.robocasa_pick_place_attempt.v1",
        "benchmark": "RoboCasa365",
        "task": "PickPlaceCounterToSink",
        "language": str(obs.get("_openeta_task_description", "")),
        "split": "target",
        "seed": args.seed,
        "robot": "Panda",
        "fixed_base": True,
        "base_action_used": False,
        "action_dim": 7,
        "success": success,
        "stop_reason": stop_reason,
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "physical_steps": total_steps,
        "openeta_runtime": "OpenEtaAgentRuntime + ToolCallingPlanner(static fallback)",
        "planner_provider_configured": False,
        "skills_attempted": ["pick", "place"],
        "tools_attempted": sorted({entry.get("name") for entry in trace if entry.get("name")}),
        "privileged_pose_fallback_used": fallback_used,
        "success_authority": "RoboCasa native _check_success",
        "native_reset_info": reset_info,
        "native_final_info": info,
        "video": str(video_path.resolve()),
        "trace": str(trace_path.resolve()),
        "experiment": str(experiment_path.resolve()),
    }
    trace_path.write_text(json.dumps(_jsonable(trace), indent=2) + "\n", encoding="utf-8")
    result_path.write_text(json.dumps(_jsonable(result), indent=2) + "\n", encoding="utf-8")
    experiment_path.write_text(
        "\n".join(
            [
                "# RoboCasa OpenETA fixed-base pick/place experiment",
                "",
                f"- Task: `PickPlaceCounterToSink` (target split, seed {args.seed})",
                "- Robot: fixed-base Panda; no base action exists in the 7D action space",
                f"- Native checker success: `{success}`",
                f"- Stop reason: `{stop_reason}`",
                f"- Physical steps: `{total_steps}`",
                f"- Privileged pose fallback: `{fallback_used}`",
                "- Planner: deterministic static backend because no LLM/VLM provider is configured",
                "- Perception: real SAM3 endpoint attempted; AnyGrasp and AnyPlace structured failures retained",
                f"- Video: `{video_path.resolve()}`",
                f"- Trace: `{trace_path.resolve()}`",
                "",
                "Success is reported only from RoboCasa's native task checker.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=384)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--max-steps-per-move", type=int, default=120)
    parser.add_argument("--gripper-steps", type=int, default=16)
    parser.add_argument("--tolerance", type=float, default=0.008)
    parser.add_argument("--contact-tolerance", type=float, default=0.035)
    parser.add_argument("--carrying-tolerance", type=float, default=0.055)
    parser.add_argument("--action-scale", type=float, default=0.05)
    parser.add_argument("--grasp-offset-x", type=float, default=0.008)
    parser.add_argument("--grasp-offset-y", type=float, default=0.0)
    parser.add_argument("--grasp-offset-z", type=float, default=-0.015)
    parser.add_argument("--sam3-url", default="http://127.0.0.1:8773/sse")
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
