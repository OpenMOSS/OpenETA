"""OpenETA environment registry — ``gym.make()`` entry point for all envs.

Usage::

    import gymnasium as gym
    import sim.env_registry  # noqa: F401 — triggers registration

    env = gym.make("openeta/dummy_sim-v0", task="pick up the cube")
    obs, info = env.reset()
    # ...

    # List discoverable envs
    from sim.env_registry import list_envs, get_env_spec, search
    for spec in list_envs():
        print(spec.id, spec.task_description)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import gymnasium as gym
import numpy as np

# ── Sentinel ───────────────────────────────────────────────────────
_UNSET = object()

# ── Global registry ─────────────────────────────────────────────────
_registry: dict[str, EnvSpec] = {}
_initialized: bool = False
_active_bench: str | None = None


# ══════════════════════════════════════════════════════════════════════
# Hot-loading — dynamically activate a benchmark's site-packages
# ══════════════════════════════════════════════════════════════════════

_HOT_BENCHES: dict[str, str] = {
    "dummy":      "",                         # always available
    "behavior":   "",                         # always available (static task list)
    "metaworld":  "metaworld",
    "maniskill":  "mani_skill",
    "libero":     "libero",
    "robocasa":   "robocasa",
    "genesis":    "genesis",
    "d4rl":       "d4rl",
}


def _add_bench_venv_deps(bench: str) -> None:
    """Add the venv site-packages for *bench* to sys.path if it exists.

    Some benches (libero) need older versions of deps (robosuite) that may
    conflict with system packages.  To avoid venv copies (PIL, numpy,
    PyTorch) shadowing their system counterparts, we eagerly import known
    shared packages *before* inserting the venv path.
    """
    import sys, os
    venv_sp = os.path.join(os.path.dirname(__file__), "venvs", bench, "lib")
    if not os.path.isdir(venv_sp):
        return
    py_dirs = sorted([d for d in os.listdir(venv_sp) if d.startswith("python")], reverse=True)
    if not py_dirs:
        return
    site_packages = os.path.join(venv_sp, py_dirs[0], "site-packages")
    if not os.path.isdir(site_packages) or site_packages in sys.path:
        return
    # Eagerly import packages we want from the *system* before the venv
    # path takes precedence (avoids Python 3.10→3.13 ABI mismatches).
    for _mod in ("PIL", "numpy", "cv2", "numba", "torch", "torchvision",
                 "torchaudio", "scipy", "matplotlib", "pandas", "pygame",
                 "PIL.Image", "PIL._imaging"):
        try:
            __import__(_mod)
        except ImportError:
            pass
    sys.path.insert(0, site_packages)


def hot_activate(bench: str) -> bool:
    """Activate a benchmark at runtime by adding its site-packages to sys.path::

        from sim.env_registry import hot_activate
        hot_activate("libero")
        env = gym.make("openeta/libero_libero_spatial_task0-v0")

    Returns ``True`` if successful, False if the bench is not installed.
    """
    import sys, os, importlib

    if bench in ("dummy", "behavior"):
        return True

    # Map bench name → python import name
    _BENCH_PKG: dict[str, str] = {
        "metaworld": "metaworld", "maniskill": "mani_skill",
        "libero": "libero", "robocasa": "robocasa",
        "genesis": "genesis", "d4rl": "d4rl",
    }
    pkg = _BENCH_PKG.get(bench, bench)
    if bench == "libero":
        lib_dir = os.environ.get("LIBERO_DIR", "/tmp/LIBERO")
        if lib_dir not in sys.path:
            sys.path.insert(0, lib_dir)

    # 1. Try direct import
    try:
        __import__(pkg)
        _register_bench(bench)
        return True
    except ImportError:
        pass

    # 2. Try venv site-packages
    venv_sp = os.path.join(os.path.dirname(__file__), "venvs", bench, "lib")
    py_dirs = []
    if os.path.isdir(venv_sp):
        py_dirs = sorted([d for d in os.listdir(venv_sp) if d.startswith("python")], reverse=True)
    if not py_dirs:
        return False

    site_packages = os.path.join(venv_sp, py_dirs[0], "site-packages")
    if not os.path.isdir(site_packages):
        return False

    if site_packages not in sys.path:
        sys.path.insert(0, site_packages)
    importlib.invalidate_caches()

    _modules_before = set(sys.modules)
    try:
        __import__(pkg)
        _register_bench(bench)
        return True
    except ImportError:
        if site_packages in sys.path:
            sys.path.remove(site_packages)
        # Clean up partially-imported modules (e.g. torch C extension
        # state that would corrupt subsequent imports by other benches).
        _new_modules = [k for k in sys.modules if k not in _modules_before]
        for mod_name in _new_modules:
            sys.modules.pop(mod_name, None)
        return False


def _register_bench(bench: str) -> None:
    """Call the registration function for *bench* (idempotent)."""
    _fn = {
        "metaworld": _register_metaworld_envs,
        "maniskill": _register_maniskill_envs,
        "libero":    _register_libero_envs,
        "robocasa":  _register_robocasa_envs,
        "genesis":   _register_genesis_envs,
        "d4rl":      _register_d4rl_envs,
    }.get(bench)
    if _fn is None:
        return
    try:
        _fn()
    except Exception:
        import traceback
        traceback.print_exc()


def hot_list_available() -> dict[str, bool]:
    """Return which benches are installable (venv on disk)."""
    import os
    result = {"dummy": True, "behavior": True}
    venvs_dir = os.path.join(os.path.dirname(__file__), "venvs")
    for bench in ["metaworld", "maniskill", "libero", "robocasa", "genesis", "d4rl"]:
        result[bench] = os.path.isdir(os.path.join(venvs_dir, bench))
    return result


# ══════════════════════════════════════════════════════════════════════
# EnvSpec
# ══════════════════════════════════════════════════════════════════════


@dataclass
class EnvSpec:
    """Metadata for one registered OpenETA environment."""

    id: str
    """Full gym ID (e.g. ``"openeta/behavior_turning_on_radio-v0"``)."""

    env_type: str
    """RLinf ``SupportedEnvType`` value (lowercase), or ``"dummy"``."""

    task_slug: str
    """URL-safe short task name (e.g. ``"turning_on_radio"``)."""

    task_description: str
    """Natural-language task description."""

    suite: str | None = None
    """Parent suite name for multi-task envs, ``None`` for single-task."""

    default_robot: str | None = None
    """Default robot model. ``None`` when not applicable."""

    available_robots: list[str] = field(default_factory=list)
    """Robot models this task can switch between (empty = fixed)."""

    action_dim: int | None = None
    """Action space dimension (``None`` = unknown until instantiation)."""

    max_episode_steps: int = 1000
    """Default maximum episode length."""

    requires_gpu: bool = False
    """Whether a CUDA-capable GPU is required."""

    requires_sim_install: bool = False
    """Whether an external simulator package must be installed."""


# ══════════════════════════════════════════════════════════════════════
# make_env — gymnasium entry_point
# ══════════════════════════════════════════════════════════════════════


def make_env(
    env_type: str,
    task_name: str = "",
    *,
    num_envs: int = 1,
    seed: int = 0,
    robot: str | None = None,
    render_mode: str | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
    include_objects: bool = False,
    split: str = "pretrain",
    **overrides: Any,
) -> gym.Env:
    """Build a single-task environment with unified observation interface.

    This is the ``entry_point`` callable registered with
    :func:`gymnasium.register`.  All envs are automatically wrapped in
    :class:`sim.unified_env.UnifiedEnv` to guarantee a consistent
    observation structure.

    Args:
        env_type: RLinf ``SupportedEnvType`` value, or ``"dummy"``.
        task_name: Task identifier specific to this env type.
        num_envs: Number of parallel instances (default ``1``).
        seed: Random seed offset.
        robot: Override the default robot (only for robots-in-config envs).
        render_mode: ``"human"`` (interactive viewer), ``"rgb_array"``
            (render to numpy), or ``None`` (no rendering).
        **overrides: Extra config keys merged into the DictConfig.

    Returns:
        A :class:`sim.unified_env.UnifiedEnv` wrapping the simulator.
    """
    from sim.unified_env import UnifiedEnv

    if env_type == "dummy":
        raw = _make_dummy_env(task_name, num_envs=num_envs, seed=seed, **overrides)
        return UnifiedEnv(raw, render_mode=render_mode)

    # ── Direct gym.make path (MetaWorld / ManiSkill / LIBERO) ──
    # These envs bypass the RLinf wrapper for direct gym.make()
    if env_type == "metaworld":
        raw = _make_metaworld_direct(task_name, render_mode=render_mode,
                                     image_width=image_width, image_height=image_height)
        ue = UnifiedEnv(raw, render_mode=render_mode)
        ue._include_objects = include_objects
        return ue
    if env_type == "maniskill":
        raw = _make_maniskill_direct(task_name, render_mode=render_mode,
                                     image_width=image_width, image_height=image_height)
        ue = UnifiedEnv(raw, render_mode=render_mode)
        ue._include_objects = include_objects
        return ue
    if env_type == "libero":
        # task_name is "suite_name/task_index"
        _add_bench_venv_deps("libero")  # libero needs robosuite deps from venv
        parts = task_name.rsplit("/", 1)
        suite_name = parts[0]
        task_idx = int(parts[1]) if len(parts) > 1 else 0
        raw = _make_libero_direct_worker(suite_name, task_idx, task_name,
                                         image_width=image_width, image_height=image_height)
        ue = UnifiedEnv(raw, render_mode=render_mode)
        ue._include_objects = include_objects
        return ue
    if env_type == "robocasa":
        # Use RoboCasa365's official single-environment constructor.  The
        # vendored RLinf wrapper is training-oriented and requires scheduler
        # state that the interactive OpenETA worker does not provide.
        raw = _make_robocasa_direct(
            task_name,
            split=split,
            seed=seed,
            robot=robot or "PandaOmron",
            render_mode=render_mode,
            image_width=image_width,
            image_height=image_height,
            **overrides,
        )
        ue = UnifiedEnv(raw, render_mode=render_mode)
        ue._include_objects = include_objects
        return ue
    if env_type == "behavior":
        from sim.envs.behavior.direct_env import BehaviorDirectEnv

        raw = BehaviorDirectEnv(
            task_name,
            seed=seed,
            robot=robot or "R1Pro",
            render_mode=render_mode,
            image_width=image_width or 128,
            image_height=image_height or 128,
            **overrides,
        )
        ue = UnifiedEnv(raw, render_mode=render_mode)
        ue._include_objects = include_objects
        return ue

    # ── RLinf-backed env ───────────────────────────────────────
    from sim.envs import get_env_cls
    from sim.env_config import build_config as _build_config

    cfg_overrides = dict(overrides)
    if robot is not None:
        cfg_overrides.setdefault("robot", robot)
    # Pass render_mode into config builder
    cfg_overrides.setdefault("render_mode", render_mode)

    cfg = _build_config(env_type, task_name, **cfg_overrides)
    env_cls = get_env_cls(env_type, cfg)
    raw = env_cls(
        cfg=cfg,
        num_envs=num_envs,
        seed_offset=seed,
        total_num_processes=1,
        worker_info=None,
    )
    return UnifiedEnv(raw, render_mode=render_mode)


def _make_metaworld_direct(env_name: str, render_mode: str | None = "rgb_array",
                            image_width: int | None = None, image_height: int | None = None) -> gym.Env:
    """Create a MetaWorld gym env directly (bypasses MetaWorldEnv vectorisation).

    Note: Requires mujoco_py (MuJoCo 210) to be installed with working
    Cython extensions.  On GCC >= 14, mujoco_py Cython compilation fails.
    Use conda or Docker in that case.
    """
    import metaworld  # noqa: F401 — registers "Meta-World" namespace with gymnasium
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        metaworld.register_mw_envs()
    rm = render_mode if render_mode is not None else "rgb_array"
    kwargs: dict[str, Any] = {}
    if image_width is not None:
        kwargs["width"] = image_width
    if image_height is not None:
        kwargs["height"] = image_height
    env = gym.make(
        "Meta-World/MT1",
        env_name=env_name,
        render_mode=rm,
        camera_id=2,
        disable_env_checker=True,
        **kwargs,
    )
    # ── fix depth clip planes ──────────────────────────────────
    try:
        env.unwrapped.model.vis.map.znear = 0.05
        env.unwrapped.model.vis.map.zfar = 3.0
    except Exception:
        pass
    # Attach task description for UnifiedEnv normaliser
    try:
        from sim.envs.metaworld import MetaWorldBenchmark
        bm = MetaWorldBenchmark("metaworld_50")
        descs = bm.get_task_description()
        idx = bm.get_env_names().index(env_name)
        env._task_description = descs[idx]
    except Exception:
        env._task_description = str(env_name)
    return env


def _make_maniskill_direct(task_id: str, render_mode: str | None = "rgb_array",
                            image_width: int | None = None, image_height: int | None = None) -> gym.Env:
    """Create a ManiSkill 3 env directly via gymnasium.make."""
    import mani_skill.envs  # noqa: F401 — triggers gym registration
    rm = render_mode if render_mode is not None else "rgb_array"
    # obs_mode='rgbd' gives camera RGB + depth + agent.qpos (joint positions)
    kwargs: dict[str, Any] = {"obs_mode": "rgbd",
                               "control_mode": "pd_ee_delta_pose",
                               "render_mode": rm, "num_envs": 1}
    if image_width is not None or image_height is not None:
        hr_cfg: dict[str, Any] = {}
        sensor_cfg: dict[str, Any] = {}
        if image_width is not None:
            hr_cfg["width"] = image_width
            sensor_cfg["width"] = image_width
        if image_height is not None:
            hr_cfg["height"] = image_height
            sensor_cfg["height"] = image_height
        kwargs["human_render_camera_configs"] = {"render_camera": hr_cfg}
        kwargs["sensor_configs"] = {"base_camera": sensor_cfg}
    try:
        env = gym.make(task_id, **kwargs)
        return env
    except Exception:
        kwargs.pop("human_render_camera_configs", None)
        kwargs.pop("sensor_configs", None)
        return gym.make(task_id, **kwargs)


def _make_libero_direct(task: Any, render_mode: str | None = "rgb_array",
                         image_width: int | None = None, image_height: int | None = None) -> gym.Env:
    """Create a LIBERO OffScreenRenderEnv from a Benchmark Task."""
    import sys, os
    lib_dir = os.environ.get("LIBERO_DIR", "/tmp/LIBERO")
    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)
    from libero.libero import get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    bddl_path = os.path.join(
        get_libero_path("bddl_files"), task.problem_folder, task.bddl_file
    )
    kwargs: dict[str, Any] = {"bddl_file_name": bddl_path, "camera_depths": True}
    if image_width is not None:
        kwargs["camera_widths"] = image_width
    if image_height is not None:
        kwargs["camera_heights"] = image_height
    raw_env = OffScreenRenderEnv(**kwargs)
    return _LibEnvWrapper(raw_env)


class _LibEnvWrapper(gym.Env):
    """Adapt old gym.Env (4-tuple step) to gymnasium (5-tuple step + reset kwargs)."""
    def __init__(self, raw_env):
        super().__init__()
        self._env = raw_env
        self.action_space = gym.spaces.Box(-1, 1, (7,), dtype=np.float32)
        self.observation_space = raw_env.observation_space if hasattr(raw_env, "observation_space") else gym.spaces.Dict({})
        self.reward_range = (-float("inf"), float("inf"))
        self.metadata = {"render_modes": ["rgb_array"]}
        self._task_description = getattr(raw_env, "_task_description", "")
        self._last_frame: Any = None

    def reset(self, *, seed=None, options=None):
        obs = self._env.reset()

        self._last_frame = obs.get("agentview_image") if isinstance(obs, dict) else None
        return obs, {}

    def step(self, action):
        ret = self._env.step(action)
        if len(ret) == 4:
            obs, rew, done, info = ret
            self._last_frame = obs.get("agentview_image") if isinstance(obs, dict) else None
            return obs, rew, done, done, info
        self._last_frame = ret[0].get("agentview_image") if isinstance(ret[0], dict) else None
        return ret

    def render(self):
        return self._last_frame

    def close(self):
        self._env.close()


def _make_libero_direct_worker(suite_name: str, task_idx: int, _unused: str,
                                image_width: int | None = None, image_height: int | None = None) -> gym.Env:
    """Create a LIBERO env from suite name + task index."""
    import sys, os
    lib_dir = os.environ.get("LIBERO_DIR", "/tmp/LIBERO")
    if lib_dir not in sys.path:
        sys.path.insert(0, lib_dir)
    from libero.libero.benchmark import get_benchmark
    b = get_benchmark(suite_name)()
    t = b.get_task(task_idx)
    env = _make_libero_direct(t, image_width=image_width, image_height=image_height)
    env._task_description = t.language
    env._env._task_description = t.language
    return env


def _make_robocasa_direct(
    task_name: str,
    *,
    split: str,
    seed: int,
    robot: str,
    render_mode: str | None,
    image_width: int | None,
    image_height: int | None,
    **overrides: Any,
) -> gym.Env:
    """Create one official RoboCasa365 benchmark scenario."""

    from sim.envs.robocasa.direct_env import RoboCasaDirectEnv

    # ``bench_worker`` carries a legacy free-form task override.  RoboCasa's
    # registered task is authoritative, so never forward that unknown kwarg.
    overrides.pop("task", None)
    return RoboCasaDirectEnv(
        task_name=task_name,
        split=split,
        seed=seed,
        robot=robot,
        render_mode=render_mode,
        image_width=image_width or 256,
        image_height=image_height or 256,
        **overrides,
    )


def _register_libero_envs() -> None:
    """Register LIBERO tasks from all benchmark suites."""
    import sys, os
    if "/tmp/LIBERO" not in sys.path:
        sys.path.insert(0, "/tmp/LIBERO")
    try:
        from libero.libero.benchmark import get_benchmark_dict
    except ImportError:
        return

    for suite_name, cls in sorted(get_benchmark_dict().items()):
        try:
            b = cls()
            n = b.get_num_tasks()
        except Exception:
            continue

        for i in range(n):
            t = b.get_task(i)
            task_name = f"{suite_name}/{i}"
            env_id = f"openeta/libero_{suite_name}_task{i}-v0"
            _register_one(
                env_id,
                EnvSpec(
                    id=env_id,
                    env_type="libero",
                    task_slug=f"{suite_name}_task{i}",
                    task_description=t.language,
                    suite=suite_name,
                    default_robot="franka_panda",
                    max_episode_steps=500,
                    requires_gpu=False,
                    requires_sim_install=True,
                ),
                env_type="libero",
                task_name=task_name,
            )


def _register_robocasa_envs() -> None:
    """Register RoboCasa365 tasks for both official evaluation splits."""

    try:
        from robocasa.utils.dataset_registry import (
            ATOMIC_TASK_DATASETS,
            TASK_SET_REGISTRY,
        )
        from robocasa.utils.dataset_registry_utils import get_task_horizon
    except ImportError:
        return

    import re

    atomic_tasks = set(ATOMIC_TASK_DATASETS)
    for task_name in TASK_SET_REGISTRY["all_tasks"]:
        words = re.sub(r"(?<!^)(?=[A-Z])", " ", task_name).lower()
        family = "atomic" if task_name in atomic_tasks else "composite"
        horizon = int(get_task_horizon(task_name))
        for split in ("pretrain", "target"):
            env_id = f"openeta/robocasa_{split}_{task_name}-v0"
            _register_one(
                env_id,
                EnvSpec(
                    id=env_id,
                    env_type="robocasa",
                    task_slug=f"{split}_{task_name}",
                    task_description=f"RoboCasa365 {split} task: {words}",
                    suite=f"robocasa365_{family}",
                    default_robot="PandaOmron",
                    available_robots=["PandaOmron", "Panda"],
                    action_dim=12,
                    max_episode_steps=horizon,
                    requires_gpu=False,
                    requires_sim_install=True,
                ),
                env_type="robocasa",
                task_name=task_name,
                split=split,
            )


def _make_dummy_env(
    kind: str,
    num_envs: int = 1,
    seed: int = 0,
    **overrides: Any,
) -> gym.Env:
    """Create a dummy sim or dummy agent adapter wrapped as gym.Env."""
    from adapter.dummy_agent import DummyAgentAdapter
    from adapter.dummy_sim import DummySimulatorAdapter

    if kind == "sim":
        return DummySimEnv(
            task=overrides.get("task", "dummy task"),
            seed=seed,
        )
    if kind == "agent":
        return DummyAgentEnv(agent=DummyAgentAdapter(), seed=seed)
    raise ValueError(f"Unknown dummy env kind: {kind!r}. Use 'sim' or 'agent'.")


# ══════════════════════════════════════════════════════════════════════
# Lightweight gym.Env wrappers for the dummy implementations
# ══════════════════════════════════════════════════════════════════════


class DummySimEnv(gym.Env):
    """Thin gym.Env wrapper around :class:`adapter.dummy_sim.DummySimulatorAdapter`.

    This is the simplest possible environment for testing the registry
    without any external simulator installed.
    """

    metadata = {"render_modes": []}

    def __init__(self, task: str = "dummy task", seed: int = 0) -> None:
        from adapter.dummy_sim import DummySimulatorAdapter

        self._sim = DummySimulatorAdapter()
        self._task = task
        self._seed = seed
        self._episode_ended = False

        self.observation_space = gym.spaces.Dict(
            {
                "task": gym.spaces.Text(4096),
                "cameras": gym.spaces.Sequence(
                    gym.spaces.Dict(
                        {
                            "frame_id": gym.spaces.Text(64),
                            "rgb": gym.spaces.Box(0, 255, shape=(2, 3, 3), dtype=int),
                            "depth": gym.spaces.Box(-np.inf, np.inf, shape=(2, 2), dtype=float),
                            "intrinsics": gym.spaces.Dict({}),
                            "extrinsics": gym.spaces.Dict({}),
                        }
                    )
                ),
                "robot_joint_positions": gym.spaces.Box(-10.0, 10.0, shape=(3,), dtype=float),
                "robot_joint_velocities": gym.spaces.Box(-10.0, 10.0, shape=(3,), dtype=float),
                "end_effector_pose": gym.spaces.Dict({}),
                "gripper_state": gym.spaces.Dict({}),
                "objects": gym.spaces.Sequence(
                    gym.spaces.Dict(
                        {
                            "name": gym.spaces.Text(128),
                            "position": gym.spaces.Box(-100.0, 100.0, shape=(3,), dtype=float),
                        }
                    )
                ),
                "metadata": gym.spaces.Dict(
                    {
                        "step_idx": gym.spaces.Discrete(1_000_000),
                        "last_action": gym.spaces.Text(64, min_length=0),
                    }
                ),
            }
        )
        self.action_space = gym.spaces.Dict(
            {
                "action_type": gym.spaces.Text(32),
                "code": gym.spaces.Text(4096),
            }
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._seed = seed
        obs = self._sim.reset(task=self._task, seed=self._seed)
        self._episode_ended = False
        return self._obs_to_dict(obs), {}

    def step(self, action):
        from adapter.protocol import EnvAction

        # Accept dict or EnvAction
        if isinstance(action, dict):
            action = EnvAction(
                action_type=action.get("action_type", "code"),
                code=action.get("code", ""),
            )

        result = self._sim.step(action)
        self._episode_ended = result.terminated or result.truncated
        return (
            self._obs_to_dict(result.observation),
            result.reward,
            result.terminated,
            result.truncated,
            result.info,
        )

    def _obs_to_dict(self, observation):
        return {
            "task": observation.task,
            "cameras": [
                {
                    "frame_id": c.frame_id,
                    "rgb": c.rgb,
                    "depth": c.depth,
                    "intrinsics": c.intrinsics,
                    "extrinsics": c.extrinsics,
                    "timestamp_s": c.timestamp_s,
                }
                for c in observation.cameras
            ],
            "robot_joint_positions": observation.robot.joint_positions,
            "robot_joint_velocities": observation.robot.joint_velocities,
            "end_effector_pose": observation.robot.end_effector_pose,
            "gripper_state": observation.robot.gripper_state,
            "objects": observation.objects,
            "metadata": observation.metadata,
        }

    def close(self):
        self._sim.close()


class DummyAgentEnv(gym.Env):
    """gym.Env that wraps a DummyAgentAdapter for agent-side testing."""

    metadata = {"render_modes": []}

    def __init__(self, agent=None, seed: int = 0) -> None:
        from adapter.dummy_agent import DummyAgentAdapter

        self._agent = agent or DummyAgentAdapter()
        self._seed = seed
        self._current_obs = None

        self.observation_space = gym.spaces.Dict(
            {"agent_state": gym.spaces.Text(4096, min_length=0)}
        )
        self.action_space = gym.spaces.Dict(
            {
                "action_type": gym.spaces.Text(32),
                "code": gym.spaces.Text(4096),
            }
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._seed = seed
        task = options.get("task", "dummy task") if options else "dummy task"
        self._agent.start_session(task=task, metadata={"seed": self._seed})
        self._current_obs = None
        return {"agent_state": "ready"}, {}

    def step(self, action):
        return {"agent_state": "done"}, 0.0, True, False, {"info": "DummyAgentEnv does not run a sim"}


# ══════════════════════════════════════════════════════════════════════
# Registration
# ══════════════════════════════════════════════════════════════════════


def _register_one(
    env_id: str,
    spec: EnvSpec,
    *,
    env_type: str,
    task_name: str,
    **kwargs: Any,
) -> None:
    """Register *env_id* with gymnasium and store its EnvSpec (idempotent)."""
    if env_id in _registry:
        return  # already registered
    gym.register(
        id=env_id,
        entry_point="sim.env_registry:make_env",
        kwargs={"env_type": env_type, "task_name": task_name, **kwargs},
        order_enforce=False,
        disable_env_checker=True,
    )
    _registry[env_id] = spec


def _register_dummy_envs() -> None:
    """Register dummy sim and agent envs (always available)."""
    _register_one(
        "openeta/dummy_sim-v0",
        EnvSpec(
            id="openeta/dummy_sim-v0",
            env_type="dummy",
            task_slug="sim",
            task_description="Dummy simulator with synthetic RGBD frames (1-step termination).",
            max_episode_steps=1,
        ),
        env_type="dummy",
        task_name="sim",
    )
    _register_one(
        "openeta/dummy_agent-v0",
        EnvSpec(
            id="openeta/dummy_agent-v0",
            env_type="dummy",
            task_slug="agent",
            task_description="Dummy code agent that returns a no-op action on every observation.",
        ),
        env_type="dummy",
        task_name="agent",
    )


# ── Behavior (BEHAVIOR-1K via OmniGibson) ────────────────────────────


def _register_behavior_envs() -> None:
    """Register all BDDL activities pinned by the BEHAVIOR v3.9 checkout.

    Requires ``omnigibson`` + Isaac Sim to actually instantiate.
    """
    import json
    import os

    descriptions_path = os.path.join(
        os.path.dirname(__file__),
        "envs",
        "behavior",
        "behavior_task.jsonl",
    )
    activities_path = os.path.join(
        os.path.dirname(__file__),
        "envs",
        "behavior",
        "behavior_activities.txt",
    )
    if not os.path.exists(activities_path):
        return

    descriptions: dict[str, str] = {}
    if os.path.exists(descriptions_path):
        with open(descriptions_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    descriptions[entry["task_name"]] = entry["task"]

    with open(activities_path, encoding="utf-8") as f:
        for line in f:
            task_name = line.strip()
            if not task_name:
                continue
            desc = descriptions.get(task_name, task_name.replace("_", " ").capitalize() + ".")
            env_id = f"openeta/behavior_{task_name}-v0"
            _register_one(
                env_id,
                EnvSpec(
                    id=env_id,
                    env_type="behavior",
                    task_slug=task_name,
                    task_description=desc,
                    default_robot="R1Pro",
                    available_robots=["R1Pro"],
                    max_episode_steps=1000,
                    requires_gpu=True,
                    requires_sim_install=True,
                ),
                env_type="behavior",
                task_name=task_name,
            )


# ── ManiSkill ───────────────────────────────────────────────────────


def _register_maniskill_envs() -> None:
    """Register ManiSkill tasks (requires ``mani_skill`` package)."""
    try:
        import mani_skill.envs  # noqa: F401 — triggers gym registration
        from mani_skill.utils.registration import REGISTERED_ENVS
    except ImportError:
        return

    # Tasks excluded from OpenETA — locomotion, humanoid, quadruped, dexterous hand
    _EXCLUDED_MANISKILL = {
        "MS-AntRun-v1", "MS-AntWalk-v1",
        "MS-HopperHop-v1", "MS-HopperStand-v1",
        "MS-CartpoleBalance-v1", "MS-CartpoleSwingUp-v1",
        "MS-HumanoidRun-v1", "MS-HumanoidStand-v1", "MS-HumanoidWalk-v1",
        "AnymalC-Reach-v1", "AnymalC-Spin-v1",
        "UnitreeG1PlaceAppleInBowl-v1", "UnitreeG1Stand-v1", "UnitreeG1TransportBox-v1",
        "UnitreeGo2-Reach-v1", "UnitreeH1Stand-v1",
        "PutEggplantInBasketScene-v1",
        "TriFingerRotateCubeLevel0-v1", "TriFingerRotateCubeLevel1-v1",
        "TriFingerRotateCubeLevel2-v1", "TriFingerRotateCubeLevel3-v1",
        "TriFingerRotateCubeLevel4-v1",
    }

    for task_id in sorted(REGISTERED_ENVS.keys()):
        if task_id in _EXCLUDED_MANISKILL:
            continue
        env_id = f"openeta/maniskill_{task_id}-v0"
        _register_one(
            env_id,
            EnvSpec(
                id=env_id,
                env_type="maniskill",
                task_slug=task_id,
                task_description=f"ManiSkill task: {task_id}",
                max_episode_steps=200,
                requires_gpu=True,
                requires_sim_install=True,
            ),
            env_type="maniskill",
            task_name=task_id,
        )


# ── Genesis ─────────────────────────────────────────────────────────


def _register_genesis_envs() -> None:
    """Register Genesis tasks (requires ``genesis`` package)."""
    try:
        import genesis  # noqa: F401
    except ImportError:
        return

    # Genesis tasks are defined in rlinf.envs.genesis.tasks
    try:
        from sim.envs.genesis.tasks import _TASK_REGISTRY
    except ImportError:
        _TASK_REGISTRY = {}

    for task_name in _TASK_REGISTRY:
        env_id = f"openeta/genesis_{task_name}-v0"
        _register_one(
            env_id,
            EnvSpec(
                id=env_id,
                env_type="genesis",
                task_slug=task_name,
                task_description=f"Genesis task: {task_name}",
                default_robot="franka_panda",
                max_episode_steps=200,
                requires_gpu=True,
                requires_sim_install=True,
            ),
            env_type="genesis",
            task_name=task_name,
        )


# ── MetaWorld ───────────────────────────────────────────────────────


def _register_metaworld_envs() -> None:
    """Register MetaWorld tasks from manifest (avoids broken mujoco_py import)."""
    import json, os

    config_path = os.path.join(
        os.path.dirname(__file__), "envs", "metaworld", "metaworld_config.json"
    )
    if not os.path.exists(config_path):
        return

    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except Exception:
        return

    task_descriptions = cfg.get("TASK_DESCRIPTIONS", {})
    if not task_descriptions:
        return

    ml45 = cfg.get("ML45", {})

    # Exclude sports/non-manipulation tasks (soccer, basketball)
    _EXCLUDED_METAWORLD = {"soccer-v3", "basketball-v3"}

    suites = [
        ("metaworld_50", list(task_descriptions.keys()), task_descriptions),
        ("metaworld_45_ind", ml45.get("train", []), task_descriptions),
        ("metaworld_45_ood", ml45.get("test", []), task_descriptions),
    ]

    for suite, env_names, descs in suites:
        short = suite.split("_")[-1]
        for env_name in env_names:
            if env_name in _EXCLUDED_METAWORLD:
                continue
            desc = descs.get(env_name, "")
            env_id = f"openeta/metaworld_{short}_{env_name}-v0"
            _register_one(
                env_id,
                EnvSpec(
                    id=env_id,
                    env_type="metaworld",
                    task_slug=env_name,
                    task_description=desc,
                    suite=suite,
                    default_robot="sawyer",
                    max_episode_steps=500,
                    requires_gpu=False,
                    requires_sim_install=True,
                ),
                env_type="metaworld",
                task_name=env_name,
            )


# ── D4RL ────────────────────────────────────────────────────────────


def _register_d4rl_envs() -> None:
    """Register D4RL locomotion tasks (requires ``d4rl`` package)."""
    try:
        import d4rl  # noqa: F401
    except ImportError:
        return

    _D4RL_TASKS = [
        "halfcheetah-medium-v2",
        "halfcheetah-medium-replay-v2",
        "halfcheetah-medium-expert-v2",
        "hopper-medium-v2",
        "hopper-medium-replay-v2",
        "hopper-medium-expert-v2",
        "walker2d-medium-v2",
        "walker2d-medium-replay-v2",
        "walker2d-medium-expert-v2",
    ]
    for task_id in _D4RL_TASKS:
        env_id = f"openeta/d4rl_{task_id}-v0"
        _register_one(
            env_id,
            EnvSpec(
                id=env_id,
                env_type="d4rl",
                task_slug=task_id,
                task_description=f"D4RL locomotion: {task_id}",
                requires_gpu=False,
                requires_sim_install=True,
            ),
            env_type="d4rl",
            task_name=task_id,
        )


# ══════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════


def _venv_active(env_type: str) -> bool:
    """Check if the current Python is running from ``sim/venvs/<env_type>/``."""
    import os, sys
    venv_path = os.path.join(os.path.dirname(__file__), "venvs", env_type)
    venv_python = os.path.join(venv_path, "bin", "python")
    return os.path.isfile(venv_python) and sys.executable.startswith(venv_path)


def _pkg_available(import_name: str) -> bool:
    """Check if *import_name* can be imported in the current Python."""
    try:
        __import__(import_name)
        return True
    except ImportError:
        pass
    # LIBERO editable install — needs LIBERO_DIR on sys.path temporarily
    if import_name == "libero":
        try:
            import sys, os
            lib_path = os.environ.get("LIBERO_DIR", "/tmp/LIBERO")
            need_remove = lib_path not in sys.path
            if need_remove:
                sys.path.insert(0, lib_path)
            __import__(import_name)
            if need_remove:
                sys.path.remove(lib_path)
            return True
        except ImportError:
            pass
    return False


def _init_registry() -> None:
    """Lazily register all discoverable environments (idempotent).

    - ``dummy`` / ``behavior``: always registered
    - all others: auto-detected if running from their venv,
      or manually via :func:`hot_activate`.
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    _register_dummy_envs()
    # Registration is static and intentionally does not import OmniGibson.
    # Instantiation still requires the isolated Isaac Sim worker environment.
    _register_behavior_envs()

    # Auto-register if running from a per-bench venv
    for bench in ["metaworld", "maniskill", "libero", "robocasa", "genesis", "d4rl"]:
        if _venv_active(bench):
            _register_bench(bench)

    # Note: to register additional benches from other venvs,
    # call hot_activate("bench_name") before gym.make().


def list_envs(
    env_type: str | None = None,
    suite: str | None = None,
    robot: str | None = None,
) -> list[EnvSpec]:
    """Return registered environments, optionally filtered.

    Args:
        env_type: Filter by env type (e.g. ``"behavior"``).
        suite: Filter by suite name (multi-task envs only).
        robot: Filter by default robot model.

    Returns:
        List of matching ``EnvSpec`` objects.
    """
    _init_registry()
    result = list(_registry.values())
    if env_type is not None:
        result = [s for s in result if s.env_type == env_type]
    if suite is not None:
        result = [s for s in result if s.suite == suite]
    if robot is not None:
        result = [s for s in result if s.default_robot == robot]
    return result


def get_env_spec(env_id: str) -> EnvSpec | None:
    """Return the ``EnvSpec`` for *env_id*, or ``None`` if unknown."""
    _init_registry()
    return _registry.get(env_id)


def search(query: str) -> list[EnvSpec]:
    """Case-insensitive substring search across task descriptions and slugs.

    Results are ranked: exact slug match first, then description matches.
    """
    _init_registry()
    q = query.lower()
    scored: list[tuple[int, EnvSpec]] = []
    for spec in _registry.values():
        score = 0
        if q == spec.task_slug.lower():
            score = 100
        elif q in spec.task_slug.lower():
            score = 50
        elif q in spec.task_description.lower():
            score = 10
        if score > 0:
            scored.append((score, spec))
    scored.sort(key=lambda x: -x[0])
    return [s for _, s in scored]


# ── Eager registration on import ────────────────────────────────────
_init_registry()
