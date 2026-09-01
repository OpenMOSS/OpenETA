"""
Ball Rolling Environments:
Goal is to roll a ball to a random target position.
"""

import gymnasium as gym

from . import agents
from .base_env import (  # need to import BallRollingEnv here, otherwise class will not be detected for entry point
    PoleBalancingEnv,
    PoleBalancingEnvCfg,
)

# ---
# Register Gym environments.
# ---


# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Pole-Balancing-Base-v0 --num_envs 1000 --enable_cameras
gym.register(
    id="TacEx-Pole-Balancing-Base-v0",
    entry_point=f"{__name__}.base_env:PoleBalancingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": PoleBalancingEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_camera_cfg.yaml",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)
