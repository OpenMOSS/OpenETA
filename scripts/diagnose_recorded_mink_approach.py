#!/usr/bin/env python3
"""Private, local-only QP telemetry for a recorded two-move controller replay.

Run in the LIBERO worker environment. No provider, network client, planner,
gripper close or task acceptance. The original solver and safety checks run
unchanged; telemetry can contain simulator state and must remain local.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def run(args):
    # Importing the worker applies its validated dependency overlay and renderer
    # setup, exactly as the server-spawned LIBERO worker does.
    from sim import bench_worker as worker
    import mink
    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation
    from sim.controllers.mink_goal import execute_libero_mink_goal, _libero_runtime

    source = json.loads(Path(args.replay_report).read_text())
    requests = [entry["request"] for entry in source["moves"]]
    if (len(requests) != 2 or source.get("agent_performance_sample") is not False
            or any(p.get("enable_collision_check") is not True
                   or not 1 <= p["num_steps"] <= 150 for p in requests)):
        raise ValueError("requires a bounded, collision-checked two-move isolation report")
    report = {"agent_performance_sample": False, "private_local_telemetry": True,
              "condition": args.condition, "open_during_approach": args.open_during_approach,
              "open_before_approach": args.open_before_approach,
              "moves": [], "cleanup": {"ok": False}}
    env = None
    original_solve = mink.solve_ik
    try:
        env = worker._make_env_locked(source["env_id"], seed=source["seed"],
                                      image_width=128, image_height=128)
        worker._reset_with_image(env, seed=source["seed"])
        raw, robot = _libero_runtime(env)
        model = raw.sim.model._model
        arm_indices = np.asarray(robot._ref_joint_vel_indexes)
        arm_positions = np.asarray(robot._ref_joint_pos_indexes)
        if args.open_before_approach:
            # Match the existing gripper_open tool's 40-step stationary horizon.
            action = np.zeros(8, dtype=np.float32)
            action[-1] = -1.0
            for _ in range(40):
                opened = worker._step_with_image(env, action, render=False)
                if opened.get("error") or opened.get("terminated") or opened.get("truncated"):
                    raise RuntimeError("opening precondition did not complete")
            report["opened_gripper_state"] = opened.get("observation", {}).get("robot", {}).get("gripper_state")
        for request in requests:
            entry = {"request": request, "qp_calls": [], "steps": []}
            report["moves"].append(entry)

            def solve(configuration, tasks, dt, **kwargs):
                violations = []
                for joint in range(model.njnt):
                    if not model.jnt_limited[joint]:
                        continue
                    q = float(configuration.q[model.jnt_qposadr[joint]])
                    low, high = model.jnt_range[joint]
                    if q < low or q > high:
                        violations.append({"joint": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint),
                                           "q": q, "lower": float(low), "upper": float(high),
                                           "arm_controlled": bool(model.jnt_dofadr[joint] in arm_indices)})
                sample = {"step": len(entry["steps"]), "joint_violations": violations,
                          "limits": [type(limit).__name__ for limit in kwargs.get("limits", [])],
                          "q_arm": configuration.q[arm_positions].tolist()}
                entry["qp_calls"].append(sample)
                try:
                    velocity = original_solve(configuration, tasks, dt, **kwargs)
                    sample.update({"solved": True, "commanded_arm_velocity": velocity[arm_indices].tolist()})
                    return velocity
                except Exception as exc:
                    sample.update({"solved": False, "error_type": type(exc).__name__})
                    raise

            def step(action, render):
                result = worker._step_with_image(env, action, render=render)
                contacts = []
                for contact in raw.sim.data.contact[:raw.sim.data.ncon]:
                    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(g)) or "" for g in (contact.geom1, contact.geom2)]
                    if any("gripper" in name or "robot" in name for name in names):
                        contacts.append({"geoms": names, "distance": float(contact.dist)})
                entry["steps"].append({"action": action.tolist(),
                    "q_arm": raw.sim.data.qpos[arm_positions].tolist(),
                    "actual_arm_velocity": raw.sim.data.qvel[arm_indices].tolist(),
                    "controller_goal_velocity": robot.controller.goal_vel.tolist(),
                    "robot_contacts": contacts[:30]})
                return result

            mink.solve_ik = solve
            # Target identity has already been resolved by the private server
            # in the original local replay. Reuse that resolution, not an oracle
            # target lookup or a new semantic claim.
            index = len(report["moves"]) - 1
            authorization = source["moves"][index]["result"].get("collision", {}).get("contact_policy", {}).get("authorization")
            result = execute_libero_mink_goal(env,
                target_xyz=[request[k] for k in ("x", "y", "z")],
                target_quat_xyzw=Rotation.from_euler("xyz", [request[k] for k in ("roll", "pitch", "yaw")], degrees=True).as_quat().tolist(),
                preserve_current_orientation=False, max_steps=request["num_steps"],
                position_tolerance_m=request["tolerance"], orientation_tolerance_rad=request["ori_tolerance"],
                gripper_command=-1.0 if (args.open_during_approach or args.open_before_approach) else 0.0, enable_collision_check=True,
                contact_authorization=authorization, attachment_proxy=None,
                ik_execution_seed=request["ik_execution_seed"], step_callback=step,
                motion_execution_condition=args.condition)
            entry["result"] = {k: v for k, v in result.items() if k != "observation"}
            print(json.dumps({"move": index, "steps": result["steps_executed"],
                              "stop": result["stop_reason"], "position_error_m": result["position_error_m"]}), flush=True)
            if not result["reached_target"]:
                break
    finally:
        mink.solve_ik = original_solve
        if env is not None:
            env.close()
            report["cleanup"] = {"ok": True}
        Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-report", required=True)
    parser.add_argument("--condition", choices=["A", "B", "C"], default="A")
    parser.add_argument("--open-during-approach", action="store_true",
                        help="Local ablation only: explicitly open fingers during the same two moves")
    parser.add_argument("--open-before-approach", action="store_true",
                        help="Local ablation only: run the existing 40-step opening horizon first")
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
