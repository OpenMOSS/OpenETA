# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""
import argparse
import copy
import sys

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="SAC",
    choices=["SAC"],
    help="The RL algorithm used for training the skrl agent.",
)

parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path to model.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

"""Rest everything follows."""

from omni.isaac.core.utils.extensions import enable_extension

enable_extension(
    "omni.isaac.debug_draw"
)  # otherwise running headless on the cluster is not possible (some GIPC classes import debug_draw)

import gymnasium as gym
import os
import random
import torch
import torch.nn as nn
from datetime import datetime

import skrl
from packaging import version

# import the skrl components to build the RL system
from skrl.agents.torch.sac import SAC, SAC_DEFAULT_CONFIG
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.trainers.torch import SequentialTrainer

# check for minimum supported skrl version
SKRL_VERSION = "1.3.0"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch") or args_cli.ml_framework.startswith("jax"):
    pass

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab.envs import (
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from omni.isaac.lab.utils.dict import print_dict
from omni.isaac.lab.utils.io import dump_pickle, dump_yaml
from omni.isaac.lab_tasks.utils.hydra import hydra_task_config
from omni.isaac.lab_tasks.utils.wrappers.skrl import SkrlVecEnvWrapper


# define models (stochastic and deterministic models) using mixins
class StochasticActor(GaussianMixin, Model):
    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-5,
        max_log_std=2,
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std)
        # self.features_extractor = nn.Sequential(nn.Conv2d(3, 32, kernel_size=8, stride=3),
        #                                         nn.ReLU(),
        #                                         nn.Conv2d(32, 64, kernel_size=4, stride=2),
        #                                         nn.ReLU(),
        #                                         nn.Conv2d(64, 64, kernel_size=2, stride=1),
        #                                         nn.ReLU(),
        #                                         nn.Flatten(),
        #                                         nn.Linear(7744, 512),
        #                                         nn.ReLU(),
        #                                         nn.Linear(512, 8),
        #                                         nn.Tanh())
        # self.features_extractor = nn.Sequential(
        #     nn.LazyConv2d(out_channels=32, kernel_size=8, stride=4, padding=0),
        #     nn.ReLU(),
        #     nn.LazyConv2d(out_channels=64, kernel_size=4, stride=2, padding=0),
        #     nn.ReLU(),
        #     nn.LazyConv2d(out_channels=64, kernel_size=3, stride=1, padding=0),
        #     nn.ReLU(),
        #     nn.Flatten(),
        # )
        self.features_extractor = nn.Sequential(
            nn.LazyConv2d(out_channels=4, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=16, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # self.net = nn.Sequential(nn.Linear(2334, 32), # why is the input 2334?, 2304 from feature extractor, 30 from propio -> why 2304?
        #                          nn.ReLU(),
        #                          nn.Linear(32, 32),
        #                          nn.ReLU(),
        #                          nn.Linear(32, self.num_actions))

        self.net = nn.Sequential(
            nn.Linear(270, 32),  # why is the input 2334?, 2304 from feature extractor, 30 from propio -> why 2304?
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_actions),
        )

        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

    def compute(self, inputs, role):
        # states = unflatten_tensorized_space(self.observation_space, inputs["states"])
        states = inputs["states"]
        # The dm_control.manipulation tasks have as observation/state spec a `collections.OrderedDict` object as follows:
        # OrderedDict([('front_close', BoundedArray(shape=(1, 84, 84, 3), dtype=dtype('uint8'), name='front_close', minimum=0, maximum=255)),
        #              ('jaco_arm/joints_pos', Array(shape=(1, 6, 2), dtype=dtype('float64'), name='jaco_arm/joints_pos')),
        #              ('jaco_arm/joints_torque', Array(shape=(1, 6), dtype=dtype('float64'), name='jaco_arm/joints_torque')),
        #              ('jaco_arm/joints_vel', Array(shape=(1, 6), dtype=dtype('float64'), name='jaco_arm/joints_vel')),
        #              ('jaco_arm/jaco_hand/joints_pos', Array(shape=(1, 3), dtype=dtype('float64'), name='jaco_arm/jaco_hand/joints_pos')),
        #              ('jaco_arm/jaco_hand/joints_vel', Array(shape=(1, 3), dtype=dtype('float64'), name='jaco_arm/jaco_hand/joints_vel')),
        #              ('jaco_arm/jaco_hand/pinch_site_pos', Array(shape=(1, 3), dtype=dtype('float64'), name='jaco_arm/jaco_hand/pinch_site_pos')),
        #              ('jaco_arm/jaco_hand/pinch_site_rmat', Array(shape=(1, 9), dtype=dtype('float64'), name='jaco_arm/jaco_hand/pinch_site_rmat'))])

        # This spec is converted to a `gym.spaces.Dict` space by the `wrap_env` function as follows:
        # Dict(front_close: Box(0, 255, (1, 84, 84, 3), uint8),
        #      jaco_arm/jaco_hand/joints_pos: Box(-inf, inf, (1, 3), float64),
        #      jaco_arm/jaco_hand/joints_vel: Box(-inf, inf, (1, 3), float64),
        #      jaco_arm/jaco_hand/pinch_site_pos: Box(-inf, inf, (1, 3), float64),
        #      jaco_arm/jaco_hand/pinch_site_rmat: Box(-inf, inf, (1, 9), float64),
        #      jaco_arm/joints_pos: Box(-inf, inf, (1, 6, 2), float64),
        #      jaco_arm/joints_torque: Box(-inf, inf, (1, 6), float64),
        #      jaco_arm/joints_vel: Box(-inf, inf, (1, 6), float64))

        # The `spaces` parameter is a flat tensor of the flattened observation/state space with shape (batch_size, size_of_flat_space).
        # Using the model's method `tensor_to_space` we can convert the flattened tensor to the original space.
        # https://skrl.readthedocs.io/en/latest/modules/skrl.models.base_class.html#skrl.models.torch.base.Model.tensor_to_space
        space = self.tensor_to_space(states, self.observation_space)

        # For this case, the `space` variable is a Python dictionary with the following structure and shapes:
        # {'front_close': torch.Tensor(shape=[batch_size, 1, 84, 84, 3], dtype=torch.float32),
        #  'jaco_arm/jaco_hand/joints_pos': torch.Tensor(shape=[batch_size, 1, 3], dtype=torch.float32)
        #  'jaco_arm/jaco_hand/joints_vel': torch.Tensor(shape=[batch_size, 1, 3], dtype=torch.float32)
        #  'jaco_arm/jaco_hand/pinch_site_pos': torch.Tensor(shape=[batch_size, 1, 3], dtype=torch.float32)
        #  'jaco_arm/jaco_hand/pinch_site_rmat': torch.Tensor(shape=[batch_size, 1, 9], dtype=torch.float32)
        #  'jaco_arm/joints_pos': torch.Tensor(shape=[batch_size, 1, 6, 2], dtype=torch.float32)
        #  'jaco_arm/joints_torque': torch.Tensor(shape=[batch_size, 1, 6], dtype=torch.float32)
        #  'jaco_arm/joints_vel': torch.Tensor(shape=[batch_size, 1, 6], dtype=torch.float32)}

        # permute and normalize the images (samples, width, height, channels) -> (samples, channels, width, height)
        # ? our space['vision_obs'] (images) here are just (width, height, channels)? -> why no samples?
        # print("vision_obs shape ", space['vision_obs'].shape)
        # print("vision_obs shape after ", space['vision_obs'].permute(0, 3, 1, 2).shape)
        features = self.features_extractor(space["vision_obs"].permute(0, 3, 1, 2) / 255.0)

        # print("features shape ", features.shape) # 2304, with propio = 30 -> num_observ = 2334
        mean_actions = torch.tanh(self.net(torch.cat([features, space["proprioceptive_obs"]], dim=-1)))

        return mean_actions, self.log_std_parameter, {}


class Critic(DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions)

        # self.features_extractor = nn.Sequential(nn.Conv2d(3, 32, kernel_size=8, stride=3),
        #                                         nn.ReLU(),
        #                                         nn.Conv2d(32, 64, kernel_size=4, stride=2),
        #                                         nn.ReLU(),
        #                                         nn.Conv2d(64, 64, kernel_size=2, stride=1),
        #                                         nn.ReLU(),
        #                                         nn.Flatten(),
        #                                         nn.Linear(7744, 512),
        #                                         nn.ReLU(),
        #                                         nn.Linear(512, 8),
        #                                         nn.Tanh())
        self.features_extractor = nn.Sequential(
            nn.LazyConv2d(out_channels=4, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=16, kernel_size=2, stride=2, padding=0),
            nn.ReLU(),
            nn.LazyConv2d(out_channels=16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # self.net = nn.Sequential(nn.Linear(2334+self.num_actions, 32), #2334 is hardcoded to the feature extractor and the observations -> how to make it based on env information?
        #                          nn.ReLU(),
        #                          nn.Linear(32, 32),
        #                          nn.ReLU(),
        #                          nn.Linear(32, self.num_actions))
        self.net = nn.Sequential(
            nn.Linear(
                270 + self.num_actions, 32
            ),  # 2334 is hardcoded to the feature extractor and the observations -> how to make it based on env information?
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, self.num_actions),
        )

    # def compute(self, inputs, role):
    #     return self.net(torch.cat([inputs["states"], inputs["taken_actions"]], dim=1)), {}
    def compute(self, inputs, role):
        states = inputs["states"]

        # map the observations/states to the original space.
        # See the explanation above (StochasticActor.compute)
        space = self.tensor_to_space(states, self.observation_space)

        # permute and normalize the images (samples, width, height, channels) -> (samples, channels, width, height)
        # ? we dont have samples dim, instead just (width, height, channels)
        features = self.features_extractor(space["vision_obs"].permute(0, 3, 1, 2) / 255.0)

        return self.net(torch.cat([features, space["proprioceptive_obs"], inputs["taken_actions"]], dim=-1)), {}


# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"


def _process_cfg(cfg: dict) -> dict:
    """Convert simple types to skrl classes/components
    :param cfg: A configuration dictionary
    :return: Updated dictionary
    """
    _direct_eval = [
        "learning_rate_scheduler",
        "shared_state_preprocessor",
        "state_preprocessor",
        "value_preprocessor",
    ]

    def reward_shaper_function(scale):
        def reward_shaper(rewards, *args, **kwargs):
            return rewards * scale

        return reward_shaper

    def update_dict(d):
        for key, value in d.items():
            if isinstance(value, dict):
                update_dict(value)
            else:
                if key in _direct_eval:
                    if type(d[key]) is str:
                        d[key] = eval(value)
                elif key.endswith("_kwargs"):
                    d[key] = value if value is not None else {}
                elif key in ["rewards_shaper_scale"]:
                    d["rewards_shaper"] = reward_shaper_function(value)
        return d

    return update_dict(copy.deepcopy(cfg))


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f"_{agent_cfg['agent']['experiment']['experiment_name']}"
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # get checkpoint path
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # # configure and instantiate the skrl runner
    # # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    # runner = Runner(env, agent_cfg)

    device = env.device

    # instantiate a memory as rollout buffer (any memory can be used for this)
    memory = RandomMemory(memory_size=agent_cfg["memory_size"], num_envs=env.num_envs, device=device)

    # instantiate the agent's models (function approximators).
    # SAC requires 5 models, visit its documentation for more details
    # https://skrl.readthedocs.io/en/latest/api/agents/sac.html#models
    models = {
        "policy": StochasticActor(env.observation_space, env.action_space, device),
        "critic_1": Critic(env.observation_space, env.action_space, device),
        "critic_2": Critic(env.observation_space, env.action_space, device),
        "target_critic_1": Critic(env.observation_space, env.action_space, device),
        "target_critic_2": Critic(env.observation_space, env.action_space, device),
    }

    cfg = SAC_DEFAULT_CONFIG.copy()
    cfg.update(_process_cfg(agent_cfg["agent"]))

    agent = SAC(
        models=models,
        memory=memory,
        cfg=cfg,  # agent_cfg,
        observation_space=env.observation_space,
        action_space=env.action_space,
        device=device,
    )

    if args_cli.checkpoint:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        agent.load(resume_path)
    # configure and instantiate the RL trainer
    cfg_trainer = {
        "timesteps": agent_cfg["trainer"]["timesteps"],
        "headless": True,
    }  # headless command gets overridden by IsaacLab argument
    trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)
    # # start training
    # trainer.train()

    # do the training "manually" for better logging of variables
    # -> see `single_agent_train` method of Trainer as reference and https://github.com/Toni-SM/skrl/discussions/84
    import tqdm

    # reset env
    states, infos = trainer.env.reset()
    for timestep in tqdm.tqdm(
        range(trainer.initial_timestep, trainer.timesteps), disable=trainer.disable_progressbar, file=sys.stdout
    ):
        # pre-interaction
        trainer.agents.pre_interaction(timestep=timestep, timesteps=trainer.timesteps)

        with torch.no_grad():
            # compute actions
            actions = trainer.agents.act(states, timestep=timestep, timesteps=trainer.timesteps)[0]

            # step the environments
            next_states, rewards, terminated, truncated, infos = trainer.env.step(actions)

            # render scene
            if not trainer.headless:
                trainer.env.render()

            # record the environments' transitions
            trainer.agents.record_transition(
                states=states,
                actions=actions,
                rewards=rewards,
                next_states=next_states,
                terminated=terminated,
                truncated=truncated,
                infos=infos,
                timestep=timestep,
                timesteps=trainer.timesteps,
            )
            # log extra info from IsaacLab env
            if "log" in infos:
                for k, v in infos["log"].items():
                    if isinstance(v, torch.Tensor) and v.numel() == 1:
                        trainer.agents.track_data(f"IsaacLab Extra Log / {k}", v.item())

            # log environment info
            if trainer.environment_info in infos:
                for k, v in infos[trainer.environment_info].items():
                    if isinstance(v, torch.Tensor) and v.numel() == 1:
                        trainer.agents.track_data(f"Info / {k}", v.item())

        # post-interaction
        trainer.agents.post_interaction(timestep=timestep, timesteps=trainer.timesteps)

        # reset environments
        if trainer.env.num_envs > 1:
            states = next_states
        else:
            if terminated.any() or truncated.any():
                with torch.no_grad():
                    states, infos = trainer.env.reset()
            else:
                states = next_states

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
