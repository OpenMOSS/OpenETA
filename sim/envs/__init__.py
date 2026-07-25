# Copyright 2025 The RLinf Authors.
# Copyright 2026 The OpenETA Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pure gymnasium ``gym.Env`` wrappers extracted from RLinf.

This module contains *only* the environment-side code: no training
infrastructure, no scheduler, no data pipelines.  ``realworld`` and
``world_model`` envs have been removed (they depend on the scheduler /
data layers).
"""

from enum import Enum


class SupportedEnvType(Enum):
    MANISKILL = "maniskill"
    LIBERO = "libero"
    ROBOTWIN = "robotwin"
    ISAACLAB = "isaaclab"
    METAWORLD = "metaworld"
    BEHAVIOR = "behavior"
    CALVIN = "calvin"
    ROBOCASA = "robocasa"
    FRANKASIM = "frankasim"
    HABITAT = "habitat"
    GENESIS = "genesis"
    EMBODICHAIN = "embodichain"
    ROBOVERSE = "roboverse"
    D4RL = "d4rl"
    POLARIS = "polaris"


def get_env_cls(env_type: str, env_cfg=None):
    """Get environment class based on environment type.

    Args:
        env_type: Type of environment (e.g. ``"maniskill"``, ``"libero"``).
        env_cfg: Optional environment configuration. Required for ``"isaaclab"``.

    Returns:
        Environment class corresponding to the environment type.
    """
    env_type = SupportedEnvType(env_type)

    if env_type == SupportedEnvType.MANISKILL:
        from sim.envs.maniskill.maniskill_env import ManiskillEnv

        return ManiskillEnv

    elif env_type == SupportedEnvType.LIBERO:
        from sim.envs.libero.libero_env import LiberoEnv

        return LiberoEnv

    elif env_type == SupportedEnvType.ROBOTWIN:
        from sim.envs.robotwin.robotwin_env import RoboTwinEnv

        return RoboTwinEnv

    elif env_type == SupportedEnvType.ISAACLAB:
        from sim.envs.isaaclab import REGISTER_ISAACLAB_ENVS

        if env_cfg is None:
            raise ValueError(
                "env_cfg is required for isaaclab environment type. "
                "Please provide env_cfg.init_params.id to select the task."
            )

        task_id = env_cfg.init_params.id
        if task_id not in REGISTER_ISAACLAB_ENVS:
            raise KeyError(
                f"Task type {task_id} has not been registered! "
                f"Available tasks: {list(REGISTER_ISAACLAB_ENVS.keys())}"
            )
        return REGISTER_ISAACLAB_ENVS[task_id]

    elif env_type == SupportedEnvType.METAWORLD:
        from sim.envs.metaworld.metaworld_env import MetaWorldEnv

        return MetaWorldEnv

    elif env_type == SupportedEnvType.BEHAVIOR:
        from sim.envs.behavior.behavior_env import BehaviorEnv

        return BehaviorEnv

    elif env_type == SupportedEnvType.CALVIN:
        from sim.envs.calvin.calvin_gym_env import CalvinEnv

        return CalvinEnv

    elif env_type == SupportedEnvType.ROBOCASA:
        from sim.envs.robocasa.robocasa_env import RobocasaEnv

        return RobocasaEnv

    elif env_type == SupportedEnvType.HABITAT:
        from sim.envs.habitat.habitat_env import HabitatEnv

        return HabitatEnv

    elif env_type == SupportedEnvType.FRANKASIM:
        from sim.envs.frankasim.frankasim_env import FrankaSimEnv

        return FrankaSimEnv

    elif env_type == SupportedEnvType.GENESIS:
        from sim.envs.genesis.genesis_env import GenesisEnv

        return GenesisEnv

    elif env_type == SupportedEnvType.EMBODICHAIN:
        from sim.envs.embodichain.embodichain_env import EmbodiChainEnv

        return EmbodiChainEnv

    elif env_type == SupportedEnvType.ROBOVERSE:
        from sim.envs.roboverse.roboverse_env import RoboVerseEnv

        return RoboVerseEnv

    elif env_type == SupportedEnvType.D4RL:
        from sim.envs.d4rl.d4rl_env import D4RLEnv

        return D4RLEnv

    elif env_type == SupportedEnvType.POLARIS:
        from sim.envs.polaris.polaris_env import PolarisEnv

        return PolarisEnv

    else:
        raise NotImplementedError(f"Environment type {env_type} not implemented")
