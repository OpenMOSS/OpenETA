# Copyright 2025 The RLinf Authors.
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
import copy
from typing import Optional, Union

import gymnasium as gym
import numpy as np
import torch
from omegaconf import OmegaConf

from sim.envs.robocasa.utils import (
    OBS_KEY_CAMERA_NAME_MAPPING,
    OBS_KEY_ROBOCASA_IMAGE_MAPPING,
    get_image_space,
)
from sim.envs.utils import (
    list_of_dict_to_dict_of_list,
    to_tensor,
)


class RobocasaEnv(gym.Env):
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        self.seed_offset = seed_offset
        self.cfg = cfg
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.num_envs = num_envs
        self.seed = self.cfg.seed + seed_offset
        self._is_start = True
        self.group_size = int(self.cfg.group_size)
        if self.num_envs <= 0:
            raise ValueError("RoboCasa legacy vector env requires num_envs > 0")
        if self.group_size <= 0:
            raise ValueError("RoboCasa group_size must be positive")
        if self.num_envs % self.group_size != 0:
            raise ValueError(
                f"RoboCasa num_envs ({self.num_envs}) must be divisible by "
                f"group_size ({self.group_size})"
            )
        self.num_group = self.num_envs // self.group_size
        self.use_fixed_reset_state_ids = cfg.get("use_fixed_reset_state_ids", False)

        self.ignore_terminations = cfg.ignore_terminations
        self.auto_reset = cfg.auto_reset

        self._generator = np.random.default_rng(seed=self.seed)

        # Get task list from config
        # Convert OmegaConf ListConfig to standard Python list
        task_names_raw = (
            OmegaConf.to_container(cfg.task_names, resolve=True)
            if OmegaConf.is_config(cfg.task_names)
            else cfg.task_names
        )
        self.task_names = (
            task_names_raw if isinstance(task_names_raw, list) else [task_names_raw]
        )
        if not self.task_names or any(not str(task).strip() for task in self.task_names):
            raise ValueError("RoboCasa task_names must contain at least one non-empty task")
        self.num_tasks = len(self.task_names)

        # Initialize reset state IDs for group_size repetition
        # Each unique scenario (num_group) will be repeated group_size times
        self._init_reset_state_ids()
        self._init_task_ids()

        self._init_env()

        self.prev_step_reward = np.zeros(self.num_envs)
        self.use_rel_reward = cfg.use_rel_reward

        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)

        self.video_cfg = cfg.video_cfg
        self.current_raw_obs = None
        self.current_info_list = None

    @property
    def camera_names(self):
        """Set the camaera names given image_space"""
        # Idealy, the env shall provide the full info. However, the number of cameras would lower the efficiency.
        # SO we only setup cameras in need.
        image_space = get_image_space(self.cfg.image_space)

        camera_names = [
            OBS_KEY_CAMERA_NAME_MAPPING[obs_key]
            for obs_key in image_space
            if obs_key in OBS_KEY_CAMERA_NAME_MAPPING
        ]

        return camera_names

    def _init_reset_state_ids(self):
        """Initialize reset state IDs - simplified version.

        For robocasa, we don't use dynamic reset_state_ids because:
        1. Robocasa doesn't support changing scenes via reset options
        2. Each environment is created with a fixed seed

        We simply assign each parallel environment a unique, fixed seed.
        """
        local_group_ids = np.arange(self.num_group, dtype=np.int64)
        global_group_ids = self.seed_offset * self.num_group + local_group_ids
        group_seeds = (self.cfg.seed + global_group_ids).astype(np.int64)
        self.env_seeds = group_seeds.repeat(self.group_size)

    def update_reset_state_ids(self):
        """Update reset state IDs for the next rollout.

        For robocasa, we use fixed seeds, so this is a no-op.
        """
        pass

    def _init_task_ids(self):
        """Assign one task per group, repeated for every group member."""

        group_task_ids = np.arange(self.num_group, dtype=np.int64) % self.num_tasks
        self.task_ids = group_task_ids.repeat(self.group_size)

    def _init_env(self):
        """Initialize robocasa environments using subprocess isolation."""
        try:
            from sim.envs.robocasa.venv import RobocasaSubprocEnv
        except ModuleNotFoundError as exc:
            if exc.name == "sim.envs.venv":
                raise RuntimeError(
                    "The legacy RoboCasa vector path requires the optional RLinf "
                    "vector helpers (`sim.envs.venv`). Use the registered "
                    "RoboCasaDirectEnv benchmark path, or install the full RLinf "
                    "vector runtime."
                ) from exc
            raise

        import robocasa  # noqa: F401 Robocasa must be imported to register envs

        # Create environment factory functions for subprocess isolation
        env_fns = self.get_env_fns()

        # Use subprocess vector environment to avoid OpenGL context sharing
        self.env = RobocasaSubprocEnv(env_fns)

    def get_env_fns(self):
        """Create environment factory functions for each parallel environment."""
        env_fns = []

        for env_id in range(self.num_envs):
            task_idx = self.task_ids[env_id]
            task_name = self.task_names[task_idx]
            env_seed = self.env_seeds[env_id]

            # Convert OmegaConf configs to standard Python types
            camera_widths_raw = self.cfg.init_params.camera_widths
            camera_heights_raw = self.cfg.init_params.camera_heights
            camera_widths = (
                OmegaConf.to_container(camera_widths_raw, resolve=True)
                if OmegaConf.is_config(camera_widths_raw)
                else camera_widths_raw
            )
            camera_heights = (
                OmegaConf.to_container(camera_heights_raw, resolve=True)
                if OmegaConf.is_config(camera_heights_raw)
                else camera_heights_raw
            )
            robot_name = self.cfg.robot_name
            camera_names = tuple(self.camera_names)

            def env_fn(
                task=task_name,
                seed=env_seed,
                width=camera_widths,
                height=camera_heights,
                robot=robot_name,
                cameras=camera_names,
            ):
                """Factory function to create a robosuite environment in subprocess."""
                import robocasa  # noqa: F401 - register tasks in spawned workers
                import robosuite
                from robosuite.controllers import load_composite_controller_config

                controller_config = load_composite_controller_config(
                    controller=None,
                    robot=robot,
                )

                env = robosuite.make(
                    env_name=task,
                    robots=robot,
                    controller_configs=controller_config,
                    camera_names=list(cameras),
                    camera_widths=width,
                    camera_heights=height,
                    has_renderer=False,
                    has_offscreen_renderer=True,
                    ignore_done=True,
                    use_object_obs=True,
                    use_camera_obs=True,
                    camera_depths=False,
                    seed=seed,
                    translucent_robot=False,
                    render_camera="robot0_agentview_center",  # Use same camera as observation
                )
                return env

            env_fns.append(env_fn)

        return env_fns

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def info_logging_keys(self):
        return []

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    def _init_metrics(self):
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs)
        self.success_episode_len = np.zeros(self.num_envs, dtype=np.int32)

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = np.zeros(self.num_envs, dtype=bool)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0
            self.success_episode_len[mask] = 0
            self._elapsed_steps[env_idx] = 0
        else:
            self.prev_step_reward[:] = 0
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self.success_episode_len[:] = 0
            self._elapsed_steps[:] = 0

    def _record_metrics(self, step_reward, terminations, infos):
        episode_info = {}
        self.returns += step_reward * (~self.success_once)
        new_success_mask = terminations & ~self.success_once
        if new_success_mask.any():
            self.success_episode_len[new_success_mask] = self.elapsed_steps[
                new_success_mask
            ]
        self.success_once = self.success_once | terminations
        episode_info["success_once"] = self.success_once.copy()
        episode_info["return"] = self.returns.copy()
        episode_info["episode_len"] = self.elapsed_steps.copy()
        episode_len_for_reward = np.where(
            self.success_once, self.success_episode_len, self.elapsed_steps
        )
        episode_info["reward"] = episode_info["return"] / np.maximum(
            episode_len_for_reward, 1
        )
        infos["episode"] = to_tensor(episode_info)
        return infos

    def _extract_image_and_state(self, obs):
        """Extract images and states from robocasa observations.

        Pi0 expects:
        - Three 224x224 images: robot0_agentview_left_image, robot0_eye_in_hand_image, robot0_agentview_right_image
        - 25D state matching training data (padded to 32D internally by Pi0)

        Based on dataset analysis and norm_stats.json, Pi0 expects 16D state:
        [0:3]   robot0_eef_pos (x, y, z) - 3D
        [3:7]   robot0_eef_quat (w, x, y, z) - 4D
        [7:9]   robot0_gripper_qpos (l, r) - 2D
        [9:11]  robot0_gripper_qvel (l, r) - 2D
        [11:14] robot0_base_to_eef_pos (x, y, z) - 3D
        [14:18] robot0_base_to_eef_quat (w, x, y, z) - 4D
        [18:21] robot0_base_pos - (x, y, z) 3D
        [21:25] robot0_base_quat - (w, x, y, z) 4D
        """
        left_images = []
        wrist_images = []
        right_images = []
        states = []

        for env_id in range(len(obs)):
            # Get camera images
            left_img = obs[env_id].get("robot0_agentview_left_image")
            wrist_img = obs[env_id].get("robot0_eye_in_hand_image")
            right_img = obs[env_id].get("robot0_agentview_right_image")

            # Flip images vertically (OpenGL coordinates are upside down)

            if left_img is not None:
                left_img = left_img[::-1]
            if wrist_img is not None:
                wrist_img = wrist_img[::-1]
            if right_img is not None:
                right_img = right_img[::-1]

            left_images.append(left_img)
            wrist_images.append(wrist_img)
            right_images.append(right_img)

            # Construct full 25D state matching Pi0's training format
            state_25d = np.zeros(25, dtype=np.float32)
            state_25d[0:3] = obs[env_id]["robot0_eef_pos"]
            state_25d[3:7] = obs[env_id]["robot0_eef_quat"]
            state_25d[7:9] = obs[env_id]["robot0_gripper_qpos"]
            state_25d[9:11] = obs[env_id]["robot0_gripper_qvel"]
            state_25d[11:14] = obs[env_id]["robot0_base_to_eef_pos"]
            state_25d[14:18] = obs[env_id]["robot0_base_to_eef_quat"]
            state_25d[18:21] = obs[env_id]["robot0_base_pos"]
            state_25d[21:25] = obs[env_id]["robot0_base_quat"]

            states.append(state_25d)

        return {
            "robot0_agentview_left_image": np.array(left_images),
            "robot0_eye_in_hand_image": np.array(wrist_images),
            "robot0_agentview_right_image": np.array(
                right_images
            ),  # can be [None, None, ...]
            "state": np.array(states),
        }

    def _extract_task_description(self, info_list):
        return [info.get("ep_meta", {}).get("lang", "") for info in info_list]

    def _wrap_obs(self, obs_list, info_list):
        if len(obs_list) != self.num_envs or len(info_list) != self.num_envs:
            raise RuntimeError(
                "RoboCasa observation wrapping requires a full vector batch; "
                f"got {len(obs_list)} observations and {len(info_list)} infos "
                f"for num_envs={self.num_envs}"
            )
        extracted_obs = self._extract_image_and_state(obs_list)
        task_description_list = self._extract_task_description(info_list)

        images_and_states_list = []
        for idx in range(self.num_envs):
            images_and_states = {
                "robot0_agentview_left_image": extracted_obs[
                    "robot0_agentview_left_image"
                ][idx],
                "robot0_eye_in_hand_image": extracted_obs["robot0_eye_in_hand_image"][
                    idx
                ],
                "robot0_agentview_right_image": extracted_obs[
                    "robot0_agentview_right_image"
                ][idx],
                "state": extracted_obs["state"][idx],
            }
            images_and_states_list.append(images_and_states)

        images_and_states_tensor = to_tensor(
            list_of_dict_to_dict_of_list(images_and_states_list)
        )

        states = images_and_states_tensor["state"]

        # Flatten structure to match libero format
        obs = {
            "states": states,
            "task_descriptions": task_description_list,
        }

        # Convert images from [H, W, C] -> [B, H, W, C]
        for obs_key_name, img_name in OBS_KEY_ROBOCASA_IMAGE_MAPPING.items():
            if images_and_states_tensor[img_name][0] is None:
                img_tensor = None
            else:
                img_tensor = torch.stack(
                    [value.clone() for value in images_and_states_tensor[img_name]]
                )
            obs.update({obs_key_name: img_tensor})

        return obs

    def _normalise_env_indices(self, env_idx) -> np.ndarray:
        if env_idx is None:
            return np.arange(self.num_envs, dtype=np.int64)
        if isinstance(env_idx, (int, np.integer)):
            indices = np.asarray([int(env_idx)], dtype=np.int64)
        else:
            raw = np.asarray(env_idx)
            if raw.dtype == bool:
                if raw.shape != (self.num_envs,):
                    raise ValueError(
                        "Boolean RoboCasa env_idx mask must have shape "
                        f"({self.num_envs},), got {raw.shape}"
                    )
                indices = np.flatnonzero(raw).astype(np.int64)
            else:
                indices = raw.astype(np.int64, copy=False).reshape(-1)
        if indices.size == 0:
            raise ValueError("RoboCasa env_idx must select at least one environment")
        if np.any(indices < 0) or np.any(indices >= self.num_envs):
            raise IndexError(
                f"RoboCasa env_idx values must be in [0, {self.num_envs - 1}]"
            )
        if len(np.unique(indices)) != len(indices):
            raise ValueError("RoboCasa env_idx must not contain duplicates")
        return indices

    def _merge_reset_batch(self, env_idx, raw_obs, info_list):
        indices = self._normalise_env_indices(env_idx)
        raw_batch = list(raw_obs)
        info_batch = list(info_list)
        if len(raw_batch) != len(indices) or len(info_batch) != len(indices):
            raise RuntimeError(
                "RoboCasa vector reset returned a batch with the wrong size: "
                f"selected={len(indices)}, observations={len(raw_batch)}, "
                f"infos={len(info_batch)}"
            )

        if self.current_raw_obs is None or self.current_info_list is None:
            if len(indices) != self.num_envs or set(indices.tolist()) != set(
                range(self.num_envs)
            ):
                raise RuntimeError(
                    "The first RoboCasa legacy vector reset must initialise all "
                    "environments before a partial reset can be merged"
                )
            self.current_raw_obs = [None] * self.num_envs
            self.current_info_list = [None] * self.num_envs

        for batch_index, env_id in enumerate(indices.tolist()):
            self.current_raw_obs[env_id] = raw_batch[batch_index]
            self.current_info_list[env_id] = info_batch[batch_index]
        if any(item is None for item in self.current_raw_obs) or any(
            item is None for item in self.current_info_list
        ):
            raise RuntimeError("RoboCasa vector observation cache is incomplete")
        return indices

    def _cache_step_batch(self, raw_obs, info_list) -> None:
        raw_batch = list(raw_obs)
        info_batch = list(info_list)
        if len(raw_batch) != self.num_envs or len(info_batch) != self.num_envs:
            raise RuntimeError(
                "RoboCasa vector step must return one item per environment; "
                f"got {len(raw_batch)} observations and {len(info_batch)} infos "
                f"for num_envs={self.num_envs}"
            )
        self.current_raw_obs = raw_batch
        self.current_info_list = info_batch

    def reset(
        self,
        env_idx: Optional[Union[int, list[int], np.ndarray]] = None,
        options: Optional[dict] = None,
    ):
        del options
        env_idx = self._normalise_env_indices(env_idx)

        if self.is_start:
            self._is_start = False

        # Reset using vectorized environment (subprocess isolation avoids OpenGL issues)
        # Use libero's SubprocVectorEnv reset interface
        raw_obs, info_list = self.env.reset(id=env_idx)
        env_idx = self._merge_reset_batch(env_idx, raw_obs, info_list)
        obs = self._wrap_obs(self.current_raw_obs, self.current_info_list)
        self._reset_metrics(env_idx)
        infos = {}
        return obs, infos

    def step(self, actions=None, auto_reset=True):
        if actions is None:
            assert self._is_start, "Actions must be provided after the first reset."
        if self.is_start:
            # Initial reset at the start of evaluation
            obs, infos = self.reset()
            terminations = np.zeros(self.num_envs, dtype=bool)
            truncations = np.zeros(self.num_envs, dtype=bool)
            rewards = np.zeros(self.num_envs, dtype=np.float32)

            return (
                obs,
                to_tensor(rewards),
                to_tensor(terminations),
                to_tensor(truncations),
                infos,
            )

        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions)
        if actions.ndim != 2 or actions.shape[0] != self.num_envs:
            raise ValueError(
                "RoboCasa vector actions must have shape "
                f"(num_envs, action_dim); got {actions.shape}"
            )

        self._elapsed_steps += 1

        # Use vectorized environment step (subprocess isolation avoids OpenGL issues)
        # Robosuite returns 4 values: (obs, reward, done, info)
        raw_obs, rewards, dones, info_lists = self.env.step(actions)
        self._cache_step_batch(raw_obs, info_lists)

        # Extract success from infos
        terminations = np.array(
            [info.get("success", False) for info in info_lists]
        ).astype(bool)
        truncations = self._elapsed_steps >= self.cfg.max_episode_steps
        obs = self._wrap_obs(self.current_raw_obs, self.current_info_list)

        step_reward = self._calc_step_reward(terminations)

        infos = list_of_dict_to_dict_of_list(info_lists)
        infos = self._record_metrics(step_reward, terminations, infos)
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = to_tensor(terminations)
            terminations[:] = False

        dones = terminations | truncations
        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)
        return (
            obs,
            to_tensor(step_reward),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        if isinstance(chunk_actions, torch.Tensor):
            shape = tuple(chunk_actions.shape)
        else:
            chunk_actions = np.asarray(chunk_actions)
            shape = chunk_actions.shape
        if len(shape) != 3 or shape[0] != self.num_envs or shape[1] <= 0:
            raise ValueError(
                "RoboCasa chunk_actions must have shape "
                f"(num_envs, chunk_size, action_dim) with chunk_size > 0; got {shape}"
            )
        chunk_size = chunk_actions.shape[1]
        obs_list = []
        infos_list = []

        chunk_rewards = []

        raw_chunk_terminations = []
        raw_chunk_truncations = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=False
            )
            obs_list.append(extracted_obs)
            infos_list.append(infos)

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_terminations = torch.stack(
            raw_chunk_terminations, dim=1
        )  # [num_envs, chunk_steps]
        raw_chunk_truncations = torch.stack(
            raw_chunk_truncations, dim=1
        )  # [num_envs, chunk_steps]

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones.cpu().numpy(), obs_list[-1], infos_list[-1]
            )

        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_chunk_terminations)
            chunk_terminations[:, -1] = past_terminations

            chunk_truncations = torch.zeros_like(raw_chunk_truncations)
            chunk_truncations[:, -1] = past_truncations
        else:
            chunk_terminations = raw_chunk_terminations.clone()
            chunk_truncations = raw_chunk_truncations.clone()
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _handle_auto_reset(self, dones, _final_obs, infos):
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = copy.deepcopy(infos)
        obs, infos = self.reset(env_idx=env_idx)
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    def _calc_step_reward(self, terminations):
        reward = self.cfg.reward_coef * terminations
        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward

        if self.use_rel_reward:
            return reward_diff
        else:
            return reward

    def close(self):
        """Close all environments."""
        if hasattr(self, "env"):
            self.env.close()
