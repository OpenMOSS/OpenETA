"""
Ball Rolling Environments:
Goal is to roll a ball to a random target position.
"""

import gymnasium as gym

from . import agents
from .base_env import (  # need to import BallRollingEnv here, otherwise class will not be detected for entry point
    BallRollingEnv,
    BallRollingEnvCfg,
)

# ---
# Register Gym environments.
# ---


# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Ball-Rolling-Privileged-v0 --num_envs 1000
gym.register(
    id="TacEx-Ball-Rolling-Privileged-v0",
    entry_point=f"{__name__}.base_env:BallRollingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BallRollingEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.BallRollingPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

from .reset_with_IK_solver import (  # need to import BallRollingEnv here, otherwise class will not be detected for entry point
    BallRollingIKResetEnv,
    BallRollingIKResetEnvCfg,
)

# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Ball-Rolling-Privileged-Reset-with-IK-solver_v0 --num_envs 1000
gym.register(
    id="TacEx-Ball-Rolling-Privileged-Reset-with-IK-solver_v0",
    entry_point=f"{__name__}.reset_with_IK_solver:BallRollingIKResetEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BallRollingIKResetEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.BallRollingPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

from .without_reaching import (
    BallRollingWithoutReachingEnv,
    BallRollingWithoutReachingEnvCfg,
)

# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Ball-Rolling-Privileged-Without-Reaching_v0 --num_envs 1000
gym.register(
    id="TacEx-Ball-Rolling-Privileged-Without-Reaching_v0",
    entry_point=f"{__name__}.without_reaching:BallRollingWithoutReachingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BallRollingWithoutReachingEnvCfg,
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": agents.rsl_rl_cfg.BallRollingPPORunnerCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)
