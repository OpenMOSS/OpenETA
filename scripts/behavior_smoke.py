#!/usr/bin/env python3
"""Create, reset, step and render one real OpenETA BEHAVIOR-1K env."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
os.environ.setdefault("OMNIGIBSON_HEADLESS", "True")


def main() -> int:
    parser = argparse.ArgumentParser()
    # This is OmniGibson's pinned r1pro_behavior.yaml default and therefore
    # the most stable offline-instance smoke target.
    parser.add_argument("--task", default="picking_up_trash")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "outputs" / "behavior" / "smoke")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    import gymnasium as gym
    import imageio.v3 as iio
    import sim.env_registry  # noqa: F401

    env_id = f"openeta/behavior_{args.task}-v0"
    env = gym.make(env_id, seed=args.seed, render_mode="rgb_array", disable_env_checker=True)
    try:
        obs, reset_info = env.reset(seed=args.seed)
        frame0 = env.render()
        action = np.zeros(env.action_space.shape, dtype=np.float32)
        obs, reward, terminated, truncated, step_info = env.step(action)
        frame1 = env.render()
        frames = [np.asarray(x, dtype=np.uint8) for x in (frame0, frame1) if x is not None]
        if frames:
            iio.imwrite(args.output / "smoke.mp4", np.stack(frames), fps=2)
        result = {
            "env_id": env_id,
            "seed": args.seed,
            "action_dim": int(env.action_space.shape[0]),
            "camera_names": sorted(obs.get("cameras", {})),
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "reset_info_keys": sorted(reset_info),
            "step_info_keys": sorted(step_info),
            "video": str(args.output / "smoke.mp4") if frames else None,
        }
        (args.output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
        return 0
    finally:
        env.close()


if __name__ == "__main__":
    raise SystemExit(main())
