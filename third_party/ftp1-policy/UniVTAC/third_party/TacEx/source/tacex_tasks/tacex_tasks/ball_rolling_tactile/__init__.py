"""
Ball Rolling Environments:
Goal is to roll a ball to a random target position.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##
from .ball_rolling_depth import (  # need to import BallRollingEnv here, otherwise class will not be detected for entry point
    BallRollingEnv,
    BallRollingEnvCfg,
)

# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Ball-Rolling-Tactile-Depth-v1 --num_envs 1024 --enable_cameras
gym.register(
    id="TacEx-Ball-Rolling-Tactile-Depth-v1",
    entry_point=f"{__name__}.ball_rolling_depth:BallRollingEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BallRollingEnvCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_camera_cfg.yaml",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)


from .ball_rolling_tactile_rgb import BallRollingTactileRGBCfg, BallRollingTactileRGBEnv

# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Ball-Rolling-Tactile-RGB-v0 --num_envs 512 --enable_cameras
gym.register(
    id="TacEx-Ball-Rolling-Tactile-RGB-v0",
    entry_point=f"{__name__}.ball_rolling_tactile_rgb:BallRollingTactileRGBEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BallRollingTactileRGBCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_tactile_rgb_cfg.yaml",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

from .ball_rolling_taxim_fots import BallRollingTaximFotsCfg, BallRollingTaximFotsEnv

# isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Ball-Rolling-Taxim-Fots-v0 --num_envs 100 --enable_cameras
gym.register(
    id="TacEx-Ball-Rolling-Taxim-Fots-v0",
    entry_point=f"{__name__}.ball_rolling_taxim_fots:BallRollingTaximFotsEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": BallRollingTaximFotsCfg,
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_tactile_rgb_cfg.yaml",
        # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
    },
)

try:
    from .ball_rolling_tactile_rgb_uipc import BallRollingTactileRGBUipcCfg, BallRollingTactileRGBUipcEnv

    # isaaclab -p ./scripts/reinforcement_learning/skrl/train.py --task TacEx-Ball-Rolling-Tactile-RGB-Uipc-v0 --num_envs 1 --enable_cameras
    gym.register(
        id="TacEx-Ball-Rolling-Tactile-RGB-Uipc-v0",
        entry_point=f"{__name__}.ball_rolling_tactile_rgb_uipc:BallRollingTactileRGBUipcEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": BallRollingTactileRGBUipcCfg,
            "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_tactile_rgb_cfg.yaml",
            # "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
            "skrl_sac_cfg_entry_point": f"{agents.__name__}:skrl_sac_cfg.yaml",
        },
    )
except ImportError:
    pass
