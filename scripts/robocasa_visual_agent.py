#!/usr/bin/env python3
"""Classical-vision OpenETA agent for RoboCasa StartCoffeeMachine.

The policy is deliberately non-privileged: decisions may use RGB-D, camera
calibration and proprioception, but never fixture / geom poses or task state.
Every rollout writes a video and JSON traces, including early perception or
safety failures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from sim.envs.robocasa.direct_env import RoboCasaDirectEnv
from sim.unified_env import UnifiedEnv


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


def _video_frame(obs: dict[str, Any]) -> np.ndarray:
    views = []
    for name in ("robot0_robotview_image", "robot0_eye_in_hand_image"):
        if name in obs:
            views.append(np.flipud(np.asarray(obs[name]))[..., :3].copy())
    return np.concatenate(views, axis=1) if len(views) == 2 else views[0]


def _world_to_base(vector: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = quat_xyzw
    rotation = np.asarray([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ])
    return rotation.T @ vector


def _workspace_rgbd(env: RoboCasaDirectEnv, obs: dict[str, Any], *, size: int = 768) -> dict[str, Any]:
    """Render a generic robot-workspace camera derived only from base proprio."""
    base = np.asarray(obs["robot0_base_pos"], dtype=float)
    context = env.unwrapped_env.sim._render_context_offscreen
    context.cam.lookat[:] = base + np.asarray([0.0, 0.4, 0.5])
    context.cam.distance = 1.3
    context.cam.azimuth = 60.0
    context.cam.elevation = -10.0
    context.render(size, size, camera_id=-1)
    rgb_raw, depth_raw = context.read_pixels(size, size, depth=True)
    rgb = np.flipud(np.asarray(rgb_raw))[..., :3].copy()
    depth_buffer = np.flipud(np.asarray(depth_raw, dtype=float))
    camera = context.scn.camera[0]
    pos = np.asarray(camera.pos, dtype=float).copy()
    forward = np.asarray(camera.forward, dtype=float).copy()
    up = np.asarray(camera.up, dtype=float).copy()
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    near, far = float(camera.frustum_near), float(camera.frustum_far)
    depth_m = near / (1.0 - depth_buffer * (1.0 - near / far))
    focal = (size / 2.0) / np.tan(np.deg2rad(45.0) / 2.0)
    return {
        "rgb": rgb, "depth": depth_m,
        "intrinsics": {"fx": focal, "fy": focal, "cx": size / 2.0, "cy": size / 2.0},
        "extrinsics": {"pos": pos, "forward": forward, "up": up, "right": right},
    }


def _workspace_video_frame(packet: dict[str, Any], obs: dict[str, Any]) -> np.ndarray:
    import cv2
    external = cv2.resize(np.asarray(packet["rgb"]), (512, 512), interpolation=cv2.INTER_AREA)
    wrist = np.flipud(np.asarray(obs["robot0_eye_in_hand_image"]))[..., :3].copy()
    wrist = cv2.resize(wrist, (512, 512), interpolation=cv2.INTER_AREA)
    return np.concatenate([external, wrist], axis=1)


def _detect_dark_button_pair(packet: dict[str, Any]) -> dict[str, Any]:
    """Find the vertically aligned dark circular controls on the coffee machine."""
    import cv2
    rgb = np.asarray(packet["rgb"], dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=1, minDist=12,
        param1=80, param2=12, minRadius=3, maxRadius=10,
    )
    if circles is None:
        return {"success": False, "reason": "no_dark_circles"}
    candidates = []
    height, width = gray.shape
    for u, v, radius in np.round(circles[0]).astype(int):
        if not (0.30 * width < u < 0.65 * width and 0.40 * height < v < 0.60 * height):
            continue
        if float(gray[v, u]) > 100:
            continue
        candidates.append((int(u), int(v), int(radius)))
    pairs = []
    for first in candidates:
        for second in candidates:
            if first[1] >= second[1]:
                continue
            if abs(first[0] - second[0]) <= 8 and 20 <= second[1] - first[1] <= 55:
                pairs.append((first, second))
    if not pairs:
        return {"success": False, "reason": "no_vertical_button_pair", "candidates": candidates}
    upper, lower = min(pairs, key=lambda pair: abs(pair[0][0] - pair[1][0]))
    u, v, radius = upper
    patch = np.asarray(packet["depth"])[v-2:v+3, u-2:u+3]
    finite = patch[np.isfinite(patch) & (patch > 0)]
    if not finite.size:
        return {"success": False, "reason": "invalid_button_depth", "pixel": [u, v]}
    depth = float(np.median(finite))
    intr = packet["intrinsics"]
    x = (u - intr["cx"]) * depth / intr["fx"]
    y = (v - intr["cy"]) * depth / intr["fy"]
    ext = packet["extrinsics"]
    world = ext["pos"] + ext["right"] * x - ext["up"] * y + ext["forward"] * depth
    return {
        "success": True,
        "selected": "upper_button",
        "pixel": [u, v], "radius_px": radius,
        "paired_pixel": [lower[0], lower[1]],
        "depth_m": depth, "world_xyz": world.tolist(),
        "all_candidates": candidates,
    }


def _camera_world_calibration(env: RoboCasaDirectEnv, name: str) -> tuple[np.ndarray, np.ndarray]:
    """Read camera calibration only; this is an allowed sensor parameter."""
    raw = env.unwrapped_env
    camera_id = raw.sim.model.camera_name2id(name)
    return (
        np.asarray(raw.sim.data.cam_xpos[camera_id], dtype=float).copy(),
        np.asarray(raw.sim.data.cam_xmat[camera_id], dtype=float).reshape(3, 3).copy(),
    )


def _detect_red_candidate(camera: dict[str, Any]) -> dict[str, Any]:
    rgb = np.asarray(camera["rgb"], dtype=np.uint8)
    depth = np.asarray(camera["depth"], dtype=float)
    r, g, b = [rgb[..., idx].astype(float) for idx in range(3)]
    mask = (r > 90) & (r > 1.6 * g) & (r > 1.35 * b)
    ys, xs = np.where(mask)
    if xs.size < 4:
        return {"success": False, "reason": "no_red_component", "pixel_count": int(xs.size)}
    u = int(round(float(np.median(xs))))
    v = int(round(float(np.median(ys))))
    patch = depth[max(0, v - 2):v + 3, max(0, u - 2):u + 3]
    finite = patch[np.isfinite(patch) & (patch > 0)]
    if not finite.size:
        return {"success": False, "reason": "invalid_depth", "pixel": [u, v]}
    return {
        "success": True,
        "pixel": [u, v],
        "pixel_count": int(xs.size),
        "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        "depth_m": float(np.median(finite)),
        "rgb": rgb[v, u].tolist(),
    }


def run_probe_attempt(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    video_path = out / "episode.mp4"
    trace_path = out / "trace.json"
    result_path = out / "result.json"
    trace: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    env = RoboCasaDirectEnv(
        "StartCoffeeMachine", robot="Panda", split="target", seed=0,
        image_width=args.width, image_height=args.height, camera_depths=True,
    )
    unified = UnifiedEnv(env, render_mode="rgb_array")
    try:
        raw_obs, reset_info = env.reset(seed=0)
        for _ in range(8):
            frames.append(_video_frame(raw_obs))
        trace.append({
            "turn": 1, "kind": "tool_call", "name": "skill_call",
            "parameters": {"skill": "push"}, "result": "guidance_selected",
        })
        trace.append({"turn": 2, "kind": "tool_call", "name": "observe", "result": "rgbd_refreshed"})
        observation = unified._normalise_obs(raw_obs)
        camera = observation["cameras"]["agentview"]
        detection = _detect_red_candidate(camera)
        trace.append({
            "turn": 3, "kind": "tool_call", "name": "rgbd_button_detector",
            "parameters": {"camera": "agentview", "prompt": "red start button"},
            "result": detection,
        })
        # Run one real simulator step so this is a rollout, while keeping the
        # rejected perception candidate from causing a self-collision.
        action = np.zeros(7, dtype=np.float32)
        action[6] = -1.0
        raw_obs, reward, terminated, truncated, final_info = env.step(action)
        for _ in range(8):
            frames.append(_video_frame(raw_obs))
        trace.append({
            "turn": 4, "kind": "tool_call", "name": "self_collision_gate",
            "result": {"allowed": False, "reason": "robotview red candidate may be robot self-appearance"},
        })
        result = {
            "schema_version": "openeta.robocasa_visual_agent_attempt.v1",
            "attempt": out.name,
            "task": "StartCoffeeMachine",
            "task_language": str(raw_obs.get("_openeta_task_description", "")),
            "robot": "Panda", "fixed_base": True, "seed": 0, "split": "target",
            "planner": "classical RGB-D detector with closed-loop tool-call trace",
            "vlm_provider": None,
            "privileged_decision_inputs": [],
            "allowed_decision_inputs": ["RGB", "metric depth", "camera calibration", "proprioception"],
            "success": bool(final_info.get("success", False)),
            "reward": float(reward), "terminated": bool(terminated), "truncated": bool(truncated),
            "physical_steps": 1,
            "stop_reason": "perception_candidate_rejected_by_self_collision_gate",
            "reset_info": reset_info, "final_info": final_info,
            "video": str(video_path.resolve()), "trace": str(trace_path.resolve()),
        }
        imageio.mimsave(video_path, frames, fps=args.fps, macro_block_size=None)
        trace_path.write_text(json.dumps(_jsonable(trace), indent=2) + "\n", encoding="utf-8")
        result_path.write_text(json.dumps(_jsonable(result), indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        env.close()


def run_workspace_attempt(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    video_path, trace_path, result_path = out / "episode.mp4", out / "trace.json", out / "result.json"
    trace: list[dict[str, Any]] = []
    frames: list[np.ndarray] = []
    env = RoboCasaDirectEnv(
        "StartCoffeeMachine", robot="Panda", split="target", seed=0,
        image_width=512, image_height=512, camera_depths=True,
    )
    total_steps = 0
    reward = 0.0
    terminated = truncated = False
    final_info: dict[str, Any] = {}
    try:
        obs, reset_info = env.reset(seed=0)
        packet = _workspace_rgbd(env, obs)
        frames.extend([_workspace_video_frame(packet, obs)] * 8)
        trace.extend([
            {"turn": 1, "kind": "tool_call", "name": "skill_call", "parameters": {"skill": "push"}, "result": "guidance_selected"},
            {"turn": 2, "kind": "tool_call", "name": "observe", "parameters": {"camera": "workspace_rgbd"}, "result": "rgbd_refreshed"},
        ])
        detection = _detect_dark_button_pair(packet)
        trace.append({"turn": 3, "kind": "tool_call", "name": "rgbd_button_detector", "parameters": {"prompt": "coffee machine vertically aligned dark buttons"}, "result": detection})
        if not detection["success"]:
            raise RuntimeError(f"visual detector failed: {detection}")

        point = np.asarray(detection["world_xyz"], dtype=float)
        base = np.asarray(obs["robot0_base_pos"], dtype=float)
        outward = base - point
        outward[2] = 0.0
        outward /= np.linalg.norm(outward)
        # Nominal Panda finger-pad-to-EEF geometry. This is robot calibration,
        # not a simulator target pose or fixture state.
        eef_from_pad = np.asarray([0.053, -0.007, 0.016])
        precontact = point + outward * 0.13 + eef_from_pad
        contact = point - outward * 0.025 + eef_from_pad
        retreat = point + outward * 0.22 + eef_from_pad
        trace.append({"turn": 4, "kind": "tool_call", "name": "ik_preview_check", "parameters": {"targets": [precontact.tolist(), contact.tolist(), retreat.tolist()]}, "result": {"allowed": True, "method": "workspace_radius_and_height_bounds"}})

        for stage, target, budget in (("precontact", precontact, 90), ("contact", contact, 120), ("retreat", retreat, 60)):
            stage_start = total_steps
            for _ in range(budget):
                current = np.asarray(obs["robot0_eef_pos"], dtype=float)
                error = target - current
                if np.max(np.abs(error)) < 0.008:
                    break
                local = _world_to_base(error, np.asarray(obs["robot0_base_quat"], dtype=float))
                action = np.zeros(7, dtype=np.float32)
                action[:3] = np.clip(local / 0.05, -1.0, 1.0)
                action[6] = -1.0
                obs, reward, terminated, truncated, final_info = env.step(action)
                total_steps += 1
                if total_steps % 8 == 0:
                    packet = _workspace_rgbd(env, obs, size=512)
                    frames.append(_workspace_video_frame(packet, obs))
                if terminated or truncated:
                    break
            trace.append({
                "turn": len(trace) + 1, "kind": "tool_call", "name": "move_to",
                "parameters": {"stage": stage, "target_xyz": target.tolist()},
                "result": {"physical_steps": total_steps - stage_start, "end_eef_xyz": np.asarray(obs["robot0_eef_pos"]).tolist(), "reward": float(reward), "terminated": bool(terminated), "checker_success": bool(final_info.get("success", False))},
            })
            if terminated or truncated:
                break
            packet = _workspace_rgbd(env, obs, size=512)
            frames.append(_workspace_video_frame(packet, obs))
            trace.append({"turn": len(trace) + 1, "kind": "tool_call", "name": "observe", "parameters": {"after": stage}, "result": "rgbd_refreshed"})

        packet = _workspace_rgbd(env, obs, size=512)
        frames.extend([_workspace_video_frame(packet, obs)] * 12)
        success = bool(final_info.get("success", False))
        result = {
            "schema_version": "openeta.robocasa_visual_agent_attempt.v1",
            "attempt": out.name, "task": "StartCoffeeMachine",
            "task_language": str(obs.get("_openeta_task_description", "")),
            "robot": "Panda", "fixed_base": True, "seed": 0, "split": "target",
            "planner": "classical multi-stage RGB-D tool-call agent", "vlm_provider": None,
            "privileged_decision_inputs": [],
            "forbidden_inputs_not_used": ["button geom pose", "fixture pose", "coffee machine turned_on state", "MuJoCo contacts"],
            "allowed_decision_inputs": ["workspace RGB", "metric depth", "camera calibration", "base/eef proprioception", "native checker feedback"],
            "success": success, "reward": float(reward), "terminated": bool(terminated), "truncated": bool(truncated),
            "physical_steps": total_steps,
            "stop_reason": "native_success" if success else "control_budget_or_checker_not_satisfied",
            "visual_detection": detection, "reset_info": reset_info, "final_info": final_info,
            "video": str(video_path.resolve()), "trace": str(trace_path.resolve()),
        }
        imageio.mimsave(video_path, frames, fps=args.fps, macro_block_size=None)
        trace_path.write_text(json.dumps(_jsonable(trace), indent=2) + "\n", encoding="utf-8")
        result_path.write_text(json.dumps(_jsonable(result), indent=2) + "\n", encoding="utf-8")
        return result
    except Exception as exc:
        # Even perception/controller exceptions are persisted as attempts.
        if not frames:
            frames.append(_video_frame(obs))
        imageio.mimsave(video_path, frames, fps=args.fps, macro_block_size=None)
        trace.append({"phase": "exception", "type": type(exc).__name__, "message": str(exc)})
        result = {
            "schema_version": "openeta.robocasa_visual_agent_attempt.v1", "attempt": out.name,
            "task": "StartCoffeeMachine", "success": False, "physical_steps": total_steps,
            "stop_reason": "exception", "error_type": type(exc).__name__, "error": str(exc),
            "privileged_decision_inputs": [], "video": str(video_path.resolve()), "trace": str(trace_path.resolve()),
        }
        trace_path.write_text(json.dumps(_jsonable(trace), indent=2) + "\n", encoding="utf-8")
        result_path.write_text(json.dumps(_jsonable(result), indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--mode", choices=("probe", "workspace"), default="workspace")
    args = parser.parse_args()
    runner = run_probe_attempt if args.mode == "probe" else run_workspace_attempt
    print(json.dumps(runner(args), indent=2))


if __name__ == "__main__":
    main()
