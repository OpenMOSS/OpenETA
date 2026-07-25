"""Smoke test: build the example env, take one observation, print a summary.

Requires real hardware and the ``real`` extra installed::

    uv sync --extra real
    uv run python -m real.examples.smoke_observe
"""

from __future__ import annotations

from real.config.example_ur5e_realsense import build_env


def main() -> None:
    env = build_env(task="smoke: observe scene")
    try:
        obs = env.reset()
        print(f"task: {obs.task!r}")
        print(f"cameras: {[c.frame_id for c in obs.cameras]}")
        for cam in obs.cameras:
            h = len(cam.rgb)
            w = len(cam.rgb[0]) if h else 0
            has_depth = cam.depth is not None
            print(f"  {cam.frame_id}: rgb={w}x{h} depth={'yes' if has_depth else 'no'} "
                  f"intr={ {k: cam.intrinsics.get(k) for k in ('fx', 'fy', 'cx', 'cy')} }")
        print(f"robot joints: {obs.robot.joint_positions}")
        print(f"robot ee: {obs.robot.end_effector_pose}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
