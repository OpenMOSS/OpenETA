"""Direct RoboCasa365 environment used by the OpenETA benchmark worker.

The older :mod:`sim.envs.robocasa.robocasa_env` module is an RLinf vector
environment.  It is useful for training, but it assumes scheduler state and a
subprocess-vector API that are not present in OpenETA's interactive worker.
This module intentionally wraps RoboCasa's official ``create_env`` helper
instead.  The resulting environment has one scenario, one authoritative task
checker, and a flat action vector whose layout matches the official
PandaOmron controller.

RoboCasa is imported lazily so importing the main OpenETA package never pulls
MuJoCo / robosuite into the agent process.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np


ROBOCASA_ACTION_DIM = 12
ROBOCASA_FIXED_ACTION_DIM = 7
ROBOCASA_ACTION_LAYOUT: dict[str, tuple[int, int]] = {
    "eef_position_delta": (0, 3),
    "eef_rotation_delta": (3, 6),
    "gripper": (6, 7),
    "base_motion": (7, 11),
    "base_velocity": (7, 10),
    "torso": (10, 11),
    "base_mode": (11, 12),
}
ROBOCASA_FIXED_ACTION_LAYOUT: dict[str, tuple[int, int]] = {
    "eef_position_delta": (0, 3),
    "eef_rotation_delta": (3, 6),
    "gripper": (6, 7),
}


def _install_fixed_panda_compat() -> None:
    """Teach RoboCasa 1.0.1's kitchen reset how to place a fixed Panda.

    RoboCasa's public constructor accepts ``robots="Panda"``, but its kitchen
    reset unconditionally writes PandaOmron mobile-base joints.  A fixed Panda
    has no such joints and its root body was deliberately compiled at
    ``[10, 10, z]`` by the kitchen loader.  For fixed robots, move that static
    root body to the task's sampled base anchor and mount it at the same
    0.7-m arm height as PandaOmron.  Mobile and humanoid robots continue to use
    the official implementation unchanged.
    """

    import robocasa.utils.env_utils as env_utils

    if getattr(env_utils, "_openeta_fixed_panda_compat", False):
        return

    original_set_robot_base = env_utils.set_robot_base
    original_set_robot_to_position = env_utils.set_robot_to_position
    mount_height = 0.7
    yaw_offset = -np.pi / 2.0

    def is_fixed(env: Any) -> bool:
        return "mobilebase0_joint_mobile_yaw" not in env.sim.model.joint_names

    def place_fixed(env: Any, position: Any) -> np.ndarray:
        floor_position = np.asarray(position, dtype=float).copy()
        mounted_position = floor_position.copy()
        mounted_position[2] += mount_height
        robot_model = env.robots[0].robot_model
        body_id = env.sim.model.body_name2id(robot_model.root_body)
        env.sim.model.body_pos[body_id] = mounted_position

        anchor_ori = np.asarray(env.init_robot_base_ori_anchor, dtype=float)
        yaw = float(anchor_ori[2] + yaw_offset)
        env.sim.model.body_quat[body_id] = np.asarray(
            [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=float
        )
        env.sim.forward()
        # RoboCasa serializes this return value into episode metadata and may
        # feed it back through set_robot_to_position on a later reset. Keep it
        # in the original floor-anchor convention to avoid adding 0.7 m twice.
        return floor_position

    def set_robot_base(env: Any, anchor_pos: Any, anchor_ori: Any,
                       rot_dev: float, pos_dev_x: float, pos_dev_y: float) -> Any:
        if not is_fixed(env):
            return original_set_robot_base(
                env, anchor_pos, anchor_ori, rot_dev, pos_dev_x, pos_dev_y
            )
        del anchor_ori, rot_dev, pos_dev_x, pos_dev_y
        return place_fixed(env, anchor_pos)

    def set_robot_to_position(env: Any, global_pos: Any) -> Any:
        if not is_fixed(env):
            return original_set_robot_to_position(env, global_pos)
        return place_fixed(env, global_pos)

    env_utils.set_robot_base = set_robot_base
    env_utils.set_robot_to_position = set_robot_to_position
    env_utils._openeta_fixed_panda_compat = True


class RoboCasaDirectEnv(gym.Env):
    """Single-scenario RoboCasa365 environment with official success checks.

    Parameters mirror the official benchmark variables.  ``seed`` identifies
    the sampled scenario, ``split`` selects the disjoint scene/object split,
    and ``horizon`` defaults to RoboCasa's v1.0.1 task registry value.

    The official RoboCasa365 datasets and Gym wrapper use ``PandaOmron``.  We
    keep it as the benchmark default and expose the controller's actual flat
    action limits rather than synthesising a generic 7-D arm action.
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        task_name: str,
        *,
        split: str = "pretrain",
        seed: int = 0,
        robot: str = "PandaOmron",
        render_mode: str | None = "rgb_array",
        image_width: int = 256,
        image_height: int = 256,
        camera_depths: bool = True,
        horizon: int | None = None,
        **env_kwargs: Any,
    ) -> None:
        super().__init__()
        if split not in {"pretrain", "target"}:
            raise ValueError("RoboCasa split must be 'pretrain' or 'target'")
        if not camera_depths:
            raise ValueError(
                "RoboCasa DirectEnv requires camera_depths=True for the "
                "agent-facing calibrated RGB-D contract"
            )

        # Lazy imports keep RoboCasa's pinned Gymnasium/MuJoCo dependencies in
        # its dedicated worker process.
        import robocasa  # noqa: F401 - registers the kitchen task classes
        from robosuite import macros as robosuite_macros
        from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
        from robocasa.utils.dataset_registry_utils import get_task_horizon
        from robocasa.utils.env_utils import create_env

        registered_tasks = set(TASK_SET_REGISTRY["all_tasks"])
        if task_name not in registered_tasks:
            raise KeyError(
                f"Unknown RoboCasa benchmark task {task_name!r}; "
                f"expected one of {len(registered_tasks)} registered tasks"
            )

        self.task_name = task_name
        self.split = split
        self.scenario_seed = int(seed)
        self.robot = robot
        self.render_mode = render_mode
        self.horizon = int(horizon if horizon is not None else get_task_horizon(task_name))
        self.elapsed_steps = 0
        self._last_raw_obs: dict[str, Any] = {}
        self._robosuite_macros = robosuite_macros
        self._image_convention = str(
            getattr(robosuite_macros, "IMAGE_CONVENTION", "opengl")
        ).strip().lower()
        if self._image_convention not in {"opengl", "opencv"}:
            raise RuntimeError(
                "Unsupported robosuite IMAGE_CONVENTION "
                f"{self._image_convention!r}; expected 'opengl' or 'opencv'."
            )

        fixed_base = robot != "PandaOmron"
        if fixed_base:
            _install_fixed_panda_compat()
            camera_names = ["robot0_robotview", "robot0_eye_in_hand"]
            env_kwargs.setdefault("render_camera", "robot0_robotview")
        else:
            camera_names = [
                "robot0_agentview_left",
                "robot0_agentview_right",
                "robot0_eye_in_hand",
            ]
        self._camera_names = camera_names
        self._env = create_env(
            env_name=task_name,
            robots=robot,
            split=split,
            seed=self.scenario_seed,
            render_onscreen=False,
            camera_names=camera_names,
            camera_widths=int(image_width),
            camera_heights=int(image_height),
            camera_depths=bool(camera_depths),
            **env_kwargs,
        )

        # robosuite defers robot/controller construction until its first
        # reset, so ``action_spec`` is not valid immediately after make().
        # RoboCasa's own Gym wrapper performs the same initialization reset.
        self._env.reset()
        low, high = self._env.action_spec
        low_arr = np.asarray(low, dtype=np.float32)
        high_arr = np.asarray(high, dtype=np.float32)
        self.action_space = gym.spaces.Box(low=low_arr, high=high_arr, dtype=np.float32)
        expected_dim = ROBOCASA_FIXED_ACTION_DIM if fixed_base else ROBOCASA_ACTION_DIM
        if self.action_space.shape != (expected_dim,):
            raise RuntimeError(
                f"Official {robot} controller changed action layout: "
                f"expected {expected_dim} dimensions, got {self.action_space.shape}. "
                "Update the OpenETA RoboCasa action codec before running evaluations."
            )

        # Raw robosuite observations are dynamic dictionaries.  UnifiedEnv
        # builds the canonical observation space after the first reset.
        self.observation_space = gym.spaces.Dict({})

    @property
    def unwrapped_env(self) -> Any:
        """Return the authoritative RoboCasa/robosuite environment."""

        return self._env

    @property
    def action_layout(self) -> dict[str, tuple[int, int]]:
        if self.action_space.shape == (ROBOCASA_FIXED_ACTION_DIM,):
            return dict(ROBOCASA_FIXED_ACTION_LAYOUT)
        return dict(ROBOCASA_ACTION_LAYOUT)

    def _set_seed(self, seed: int) -> None:
        self.scenario_seed = int(seed)
        # RoboCasa's own Gym wrapper applies reset seeds this way.  Scene and
        # object sampling both consume this generator during reset.
        self._env.rng = np.random.default_rng(self.scenario_seed)

    def _checked_image_convention(self) -> str:
        """Return the construction-time convention, rejecting global drift."""

        macros = getattr(self, "_robosuite_macros", None)
        if macros is not None:
            current = str(
                getattr(macros, "IMAGE_CONVENTION", "opengl")
            ).strip().lower()
            if current != self._image_convention:
                raise RuntimeError(
                    "robosuite IMAGE_CONVENTION changed after RoboCasa "
                    "environment construction; refusing an ambiguous RGB-D packet"
                )
        return self._image_convention

    def _annotate(self, raw_obs: dict[str, Any]) -> dict[str, Any]:
        obs = dict(raw_obs)
        missing_rgbd = [
            field
            for camera_name in self._camera_names
            for field in (f"{camera_name}_image", f"{camera_name}_depth")
            if obs.get(field) is None
        ]
        if missing_rgbd:
            raise RuntimeError(
                "RoboCasa DirectEnv observation is missing configured RGB-D "
                f"channels: {missing_rgbd}"
            )
        if "robot0_base_pos" not in obs:
            try:
                robot_model = self._env.robots[0].robot_model
                body_id = self._env.sim.model.body_name2id(robot_model.root_body)
                obs["robot0_base_pos"] = self._env.sim.data.body_xpos[body_id].copy()
                quat_wxyz = self._env.sim.data.body_xquat[body_id]
                obs["robot0_base_quat"] = np.roll(quat_wxyz, -1).copy()
            except Exception:
                pass
        language = ""
        try:
            language = str(self._env.get_ep_meta().get("lang", ""))
        except Exception:
            language = ""
        obs["_openeta_task_description"] = language
        # Freeze the convention observed when this environment was created.
        # Do not mutate the process-global robosuite macro: a worker may also
        # host a LIBERO environment whose established image path must remain
        # untouched.
        obs["_openeta_image_convention"] = self._checked_image_convention()
        obs["_openeta_robocasa_camera_names"] = list(self._camera_names)
        obs["_openeta_benchmark"] = {
            "name": "robocasa365",
            "task": self.task_name,
            "split": self.split,
            "scenario_seed": self.scenario_seed,
            "horizon": self.horizon,
            "elapsed_steps": self.elapsed_steps,
            "robot": self.robot,
            "action_dim": int(self.action_space.shape[0]),
        }
        self._last_raw_obs = obs
        return obs

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        del options
        if seed is not None:
            self._set_seed(seed)
        self.elapsed_steps = 0
        raw_obs = self._env.reset()
        success = bool(self._env._check_success())
        return self._annotate(raw_obs), {
            "success": success,
            "task": self.task_name,
            "split": self.split,
            "scenario_seed": self.scenario_seed,
            "horizon": self.horizon,
            "elapsed_steps": self.elapsed_steps,
        }

    def step(
        self,
        action: np.ndarray | list[float] | tuple[float, ...],
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        act = np.asarray(action, dtype=np.float32).reshape(-1)
        if act.shape != self.action_space.shape:
            raise ValueError(
                f"RoboCasa action must have shape {self.action_space.shape}, got {act.shape}"
            )
        act = np.clip(act, self.action_space.low, self.action_space.high)
        raw_obs, _reward, env_done, info = self._env.step(act)
        self.elapsed_steps += 1

        success = bool(self._env._check_success())
        terminated = bool(env_done or success)
        truncated = bool(self.elapsed_steps >= self.horizon and not terminated)
        reward = 1.0 if success else 0.0
        safe_info = dict(info) if isinstance(info, dict) else {"raw_info": str(info)}
        safe_info.update(
            {
                "success": success,
                "task": self.task_name,
                "split": self.split,
                "scenario_seed": self.scenario_seed,
                "horizon": self.horizon,
                "elapsed_steps": self.elapsed_steps,
            }
        )
        return self._annotate(raw_obs), reward, terminated, truncated, safe_info

    def render(self) -> np.ndarray | None:
        raw = self._last_raw_obs
        for camera_name in self._camera_names:
            image = raw.get(f"{camera_name}_image")
            if image is not None:
                array = np.asarray(image)
                if self._checked_image_convention() == "opengl":
                    array = np.flipud(array)
                return array[..., :3].copy()
        return None

    def close(self) -> None:
        self._env.close()

    def __getattr__(self, name: str) -> Any:
        # Preserve access to sim/model/objects for calibration and optional
        # privileged debug observations in UnifiedEnv.
        if name.startswith("__"):
            raise AttributeError(name)
        env = self.__dict__.get("_env")
        if env is None:
            raise AttributeError(name)
        return getattr(env, name)
