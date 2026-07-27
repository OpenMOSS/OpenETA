#!/usr/bin/env python3
"""Run a deterministic fixed-base Panda RoboCasa button task and save video.

This is an integration/calibration demo, not a vision-general policy: the
controller reads the authoritative button geom pose, then uses closed-loop OSC
position deltas. Task success is still decided only by RoboCasa's native
``_check_success`` implementation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np

from sim.envs.robocasa.direct_env import RoboCasaDirectEnv


def _world_to_base(vector: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = quat_xyzw
    rotation = np.asarray(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )
    return rotation.T @ vector


def _eef_position(env: RoboCasaDirectEnv) -> np.ndarray:
    raw = env.unwrapped_env
    site_id = raw.robots[0].eef_site_id["right"]
    return raw.sim.data.site_xpos[site_id].copy()


def _base_quaternion(env: RoboCasaDirectEnv) -> np.ndarray:
    raw = env.unwrapped_env
    body_name = raw.robots[0].robot_model.root_body
    body_id = raw.sim.model.body_name2id(body_name)
    return np.roll(raw.sim.data.body_xquat[body_id], -1).copy()


def _video_frame(obs: dict[str, Any], env: RoboCasaDirectEnv) -> np.ndarray | None:
    """Compose an external fixed view and wrist close-up into one frame."""

    views: list[np.ndarray] = []
    wrist = obs.get("robot0_eye_in_hand_image")
    if wrist is not None:
        wrist_view = np.flipud(np.asarray(wrist))[..., :3].copy()
        height, width = wrist_view.shape[:2]
        try:
            raw = env.unwrapped_env
            context = raw.sim._render_context_offscreen
            context.cam.lookat[:] = env._openeta_video_lookat
            context.cam.distance = 1.3
            context.cam.azimuth = 120.0
            context.cam.elevation = -15.0
            context.render(width, height, camera_id=-1)
            views.append(np.flipud(context.read_pixels(width, height)))
        except Exception:
            pass
        views.append(wrist_view)
    if len(views) == 2:
        return np.concatenate(views, axis=1)
    if views:
        return views[0]
    return env.render()


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{args.task}_fixed_panda.mp4"
    result_path = output_dir / "result.json"

    env = RoboCasaDirectEnv(
        args.task,
        robot="Panda",
        split=args.split,
        seed=args.seed,
        image_width=args.width,
        image_height=args.height,
        camera_depths=False,
    )
    frames: list[np.ndarray] = []
    trace: list[dict[str, Any]] = []
    total_steps = 0
    reward = 0.0
    terminated = False
    truncated = False
    info: dict[str, Any] = {}

    try:
        obs, reset_info = env.reset(seed=args.seed)
        raw = env.unwrapped_env
        if hasattr(raw, "microwave"):
            appliance = raw.microwave
            button_geom = appliance.naming_prefix + "start_button"
        elif hasattr(raw, "coffee_machine"):
            appliance = raw.coffee_machine
            button_geom = appliance.naming_prefix + appliance._start_button_names[0]
        else:
            raise RuntimeError(f"Task {args.task!r} has no supported button fixture")
        button_id = raw.sim.model.geom_name2id(button_geom)
        button = raw.sim.data.geom_xpos[button_id].copy()
        env._openeta_video_lookat = button.copy()
        frame = _video_frame(obs, env)
        if frame is not None:
            frames.extend([frame.copy()] * 8)
        start = _eef_position(env)
        base_quat = _base_quaternion(env)

        # The microwave button is a thin geom whose outward normal is world
        # -X for this fixture orientation. First align the finger pad with the
        # button without touching the panel. Once there, measure the actual
        # pad-to-grip-site offset (it depends on wrist orientation), then press
        # along the surface normal.
        button_rotation = raw.sim.data.geom_xmat[button_id].reshape(3, 3)
        thin_axis = int(np.argmin(raw.sim.model.geom_size[button_id]))
        approach = button_rotation[:, thin_axis].copy()
        if float(np.dot(approach, start - button)) < 0.0:
            approach *= -1.0
        targets = [("pre_contact", button + 0.10 * approach)]

        for stage, target in targets:
            stage_start = total_steps
            for _ in range(args.max_steps_per_stage):
                current = _eef_position(env)
                error = target - current
                if np.max(np.abs(error)) < args.tolerance:
                    break
                local_error = _world_to_base(error, base_quat)
                action = np.zeros(7, dtype=np.float32)
                action[:3] = np.clip(local_error / args.action_scale, -1.0, 1.0)
                # Keep the fingers open so one finger can press the small
                # button without the palm or the opposing finger hitting the
                # microwave face first.
                action[6] = -1.0
                obs, reward, terminated, truncated, info = env.step(action)
                total_steps += 1
                if total_steps % args.frame_stride == 0:
                    frame = _video_frame(obs, env)
                    if frame is not None:
                        frames.append(frame.copy())
                if terminated or truncated:
                    break
            trace.append(
                {
                    "stage": stage,
                    "target_xyz": target.tolist(),
                    "end_xyz": _eef_position(env).tolist(),
                    "steps": total_steps - stage_start,
                    "turned_on": bool(appliance.get_state()["turned_on"]),
                    "native_success": bool(raw._check_success()),
                }
            )
            if stage == "contact":
                gripper_geoms = set(raw.robots[0].gripper["right"].contact_geoms)
                contact_pairs: list[dict[str, Any]] = []
                for index in range(raw.sim.data.ncon):
                    contact = raw.sim.data.contact[index]
                    name1 = raw.sim.model.geom_id2name(contact.geom1) or str(contact.geom1)
                    name2 = raw.sim.model.geom_id2name(contact.geom2) or str(contact.geom2)
                    if name1 in gripper_geoms or name2 in gripper_geoms:
                        contact_pairs.append(
                            {
                                "geoms": [name1, name2],
                                "contact_xyz": np.asarray(contact.pos).tolist(),
                                "geom1_xyz": raw.sim.data.geom_xpos[contact.geom1].tolist(),
                                "geom2_xyz": raw.sim.data.geom_xpos[contact.geom2].tolist(),
                            }
                        )
                trace[-1]["gripper_contacts"] = contact_pairs
                trace[-1]["eef_quat_xyzw"] = np.asarray(
                    obs["robot0_eef_quat"]
                ).tolist()
            if stage == "pre_contact":
                gripper = raw.robots[0].gripper["right"]
                eef = _eef_position(env)
                contact_geoms = list(gripper.contact_geoms)
                geom_offsets = {
                    name: (
                        raw.sim.data.geom_xpos[raw.sim.model.geom_name2id(name)] - eef
                    ).copy()
                    for name in contact_geoms
                }
                # Align the furthest-forward collision geom itself with the
                # small button. This prevents a neighbouring finger link from
                # hitting the broad microwave face first and stalling OSC.
                finger2 = next(
                    (
                        name
                        for name in contact_geoms
                        if name.endswith("finger2_collision")
                    ),
                    "",
                )
                leading_geom = finger2 or max(
                    geom_offsets, key=lambda name: geom_offsets[name][0]
                )
                contact_target = (
                    button - geom_offsets[leading_geom] - 0.030 * approach
                )
                targets.extend(
                    [
                        ("contact", contact_target),
                        ("retreat", button + 0.22 * approach),
                    ]
                )
                trace[-1]["leading_contact_geom"] = leading_geom
                trace[-1]["leading_contact_geom_offset"] = geom_offsets[
                    leading_geom
                ].tolist()
            if terminated or truncated:
                break

        final_frame = _video_frame(obs, env)
        if final_frame is not None:
            frames.extend([final_frame.copy()] * 12)
        success = bool(raw._check_success())
        result = {
            "schema_version": "openeta.robocasa_fixed_panda_demo.v1",
            "benchmark": "robocasa365",
            "task": args.task,
            "language": str(obs.get("_openeta_task_description", "")),
            "split": args.split,
            "seed": args.seed,
            "robot": "Panda",
            "fixed_base": True,
            "action_dim": int(env.action_space.shape[0]),
            "success": success,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "steps": total_steps,
            "button_xyz": button.tolist(),
            "start_eef_xyz": start.tolist(),
            "final_eef_xyz": _eef_position(env).tolist(),
            "final_button_distance": float(np.linalg.norm(_eef_position(env) - button)),
            "turned_on": bool(appliance.get_state()["turned_on"]),
            "native_reset_info": reset_info,
            "native_final_info": info,
            "trace": trace,
            "video": str(video_path.resolve()),
            "policy": "privileged deterministic button-pose calibration controller",
        }
        imageio.mimsave(video_path, frames, fps=args.fps, macro_block_size=None)
        result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        return result
    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="StartCoffeeMachine")
    parser.add_argument("--split", choices=("pretrain", "target"), default="target")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs/robocasa/fixed_panda_coffee_seed0")
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--max-steps-per-stage", type=int, default=120)
    parser.add_argument("--tolerance", type=float, default=0.008)
    parser.add_argument("--action-scale", type=float, default=0.05)
    args = parser.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == "__main__":
    main()
