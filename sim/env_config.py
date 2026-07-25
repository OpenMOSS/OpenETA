"""Minimal config builders for each environment type.

Every builder accepts ``render_mode`` as a unified kwarg and maps it
to the backend-specific config key:

=============== ==================== ===================================
render_mode      Genesis              ManiSkill / MetaWorld / D4RL
=============== ==================== ===================================
``"human"``      show_viewer=True     render_mode="human"
``"rgb_array"``  BatchRenderer        render_mode="rgb_array" (default)
``None``         no renderer          render_mode="rgb_array"
=============== ====================

For BEHAVIOR: ``render_mode="human"`` → ``macro.headless=False``.
"""

from __future__ import annotations

from typing import Any

from omegaconf import DictConfig, OmegaConf


# ── Common defaults shared by all envs ──────────────────────────────

_COMMON_FIELDS: dict[str, Any] = {
    "seed": 0,
    "max_episode_steps": 1000,
    "auto_reset": True,
    "ignore_terminations": False,
    "use_rel_reward": False,
    "reward_coef": 1.0,
    "group_size": 1,
    "use_fixed_reset_state_ids": False,
    "video_cfg": {
        "info_on_video": False,
    },
}


def _base_cfg(**overrides: Any) -> DictConfig:
    merged = dict(_COMMON_FIELDS)
    merged.update(overrides)
    return OmegaConf.create(merged)


# ── Per-environment-type builders ───────────────────────────────────

def build_maniskill_cfg(task_id: str, **overrides: Any) -> DictConfig:
    render_mode = overrides.pop("render_mode", "rgb_array")
    init_params = {
        "id": task_id,
        "obs_mode": "state",
        "control_mode": "pd_ee_delta_pose",
        "render_mode": render_mode if render_mode is not None else "rgb_array",
    }
    init_params.update(overrides.pop("init_params", {}))
    return _base_cfg(init_params=init_params, **overrides)


# ── Genesis ──────────────────────────────────────────────────────────

def build_genesis_cfg(task_name: str, **overrides: Any) -> DictConfig:
    render_mode = overrides.pop("render_mode", "rgb_array")
    init_params = {
        "task_name": task_name,
        "backend": "cuda",
        # Genesis-specific: show_viewer is not an init_param but we
        # carry it so UnifiedEnv._apply_render_mode can read it.
        "show_viewer": (render_mode == "human"),
    }
    init_params.update(overrides.pop("init_params", {}))
    return _base_cfg(max_episode_steps=200, init_params=init_params, **overrides)


# ── MetaWorld ─────────────────────────────────────────────────────────

def build_metaworld_cfg(env_name: str, **overrides: Any) -> DictConfig:
    """Build config for a MetaWorld environment.

    Args:
        env_name: Specific MetaWorld task name (e.g. ``"assembly-v3"``).
            The suite is auto-detected from the registration.
    """
    render_mode = overrides.pop("render_mode", "rgb_array")
    # Detect suite from env_name or use override
    suite = overrides.pop("task_suite_name", "metaworld_50")
    init_params = {
        "render_mode": render_mode if render_mode is not None else "rgb_array",
    }
    init_params.update(overrides.pop("init_params", {}))
    return _base_cfg(
        max_episode_steps=500,
        task_suite_name=suite,
        task_name=env_name,          # specific env_name
        use_fixed_reset_state_ids=True,
        is_eval=False,
        use_ordered_reset_state_ids=False,
        init_params=init_params,
        **overrides,
    )


# ── D4RL ──────────────────────────────────────────────────────────────

def build_d4rl_cfg(task_name: str, **overrides: Any) -> DictConfig:
    render_mode = overrides.pop("render_mode", "rgb_array")
    return _base_cfg(
        max_episode_steps=1000,
        task_name=task_name,
        render_mode=render_mode if render_mode is not None else "rgb_array",
        **overrides,
    )


# ── BEHAVIOR (OmniGibson) ────────────────────────────────────────────

def build_behavior_cfg(
    activity_name: str,
    *,
    robot_model: str = "franka_panda",
    base_config_name: str = "r1pro_behavior",
    **overrides: Any,
) -> DictConfig:
    render_mode = overrides.pop("render_mode", "rgb_array")
    omni_config: dict[str, Any] = {
        "task": {"activity_name": activity_name},
        "robots": [{"model": robot_model}],
    }
    if render_mode == "human":
        omni_config.setdefault("macro", {})
        omni_config["macro"]["headless"] = False
        omni_config["macro"]["render_viewer_camera"] = True
    omni_config.update(overrides.pop("omni_config", {}))
    return _base_cfg(
        max_episode_steps=1000,
        base_config_name=base_config_name,
        omni_config=omni_config,
        **overrides,
    )


# ── Libero ────────────────────────────────────────────────────────────

def build_libero_cfg(
    task_suite_name: str,
    *,
    task_id: int | None = None,
    **overrides: Any,
) -> DictConfig:
    render_mode = overrides.pop("render_mode", "rgb_array")
    extra: dict[str, Any] = {}
    if task_id is not None:
        extra["specific_reset_id"] = task_id
    return _base_cfg(
        max_episode_steps=500,
        task_suite_name=task_suite_name,
        init_params={
            "bddl_file_name": "",
            "has_offscreen_renderer": (render_mode != "human"),
            "use_camera_obs": True,
            "control_freq": 20,
        },
        **extra,
        **overrides,
    )


# ── RoboCasa ──────────────────────────────────────────────────────────

def build_robocasa_cfg(
    task_name: str,
    *,
    robot_name: str = "Panda",
    **overrides: Any,
) -> DictConfig:
    render_mode = overrides.pop("render_mode", "rgb_array")
    return _base_cfg(
        max_episode_steps=500,
        robot_name=robot_name,
        task_names=[task_name],
        image_space="default",
        init_params={
            "camera_widths": [224, 224, 224],
            "camera_heights": [224, 224, 224],
            "has_renderer": (render_mode == "human"),
            "has_offscreen_renderer": (render_mode != "human"),
            "use_camera_obs": True,
        },
        **overrides,
    )


# ── RoboTwin ──────────────────────────────────────────────────────────

def build_robotwin_cfg(task_name: str, **overrides: Any) -> DictConfig:
    overrides.pop("render_mode", None)
    return _base_cfg(
        max_episode_steps=500,
        task_config={"task_name": task_name},
        **overrides,
    )


# ── RoboVerse ─────────────────────────────────────────────────────────

def build_roboverse_cfg(
    task_name: str,
    *,
    robot_name: str = "franka",
    **overrides: Any,
) -> DictConfig:
    overrides.pop("render_mode", None)
    return _base_cfg(
        max_episode_steps=500,
        task_name=task_name,
        robot_name=robot_name,
        **overrides,
    )


# ── FrankaSim ─────────────────────────────────────────────────────────

def build_frankasim_cfg(task_name: str, **overrides: Any) -> DictConfig:
    render_mode = overrides.pop("render_mode", None)
    init_params = {}
    if render_mode is not None:
        init_params["render_mode"] = render_mode
    init_params.update(overrides.pop("init_params", {}))
    return _base_cfg(
        max_episode_steps=500,
        task_config={"task_name": task_name},
        init_params=init_params,
        **overrides,
    )


# ── Calvin ────────────────────────────────────────────────────────────

def build_calvin_cfg(task_suite_name: str, **overrides: Any) -> DictConfig:
    overrides.pop("render_mode", None)
    return _base_cfg(
        max_episode_steps=500,
        task_suite_name=task_suite_name,
        is_eval=False,
        use_ordered_reset_state_ids=False,
        **overrides,
    )


# ── IsaacLab / Polaris ────────────────────────────────────────────────

def build_isaaclab_cfg(task_id: str, **overrides: Any) -> DictConfig:
    overrides.pop("render_mode", None)
    init_params = {"id": task_id}
    init_params.update(overrides.pop("init_params", {}))
    return _base_cfg(init_params=init_params, **overrides)


# ── Habitat ───────────────────────────────────────────────────────────

def build_habitat_cfg(task_key: str, **overrides: Any) -> DictConfig:
    r = overrides.pop("render_mode", None)
    extra = {}
    if r == "human":
        extra["show_gui"] = True
    return _base_cfg(
        max_episode_steps=1000,
        task_key=task_key,
        max_steps_per_rollout_epoch=1000,
        **extra,
        **overrides,
    )


# ── Dispatch table ────────────────────────────────────────────────────

_BUILDERS: dict[str, Any] = {
    "maniskill":  build_maniskill_cfg,
    "genesis":    build_genesis_cfg,
    "metaworld":  build_metaworld_cfg,
    "d4rl":       build_d4rl_cfg,
    "behavior":   build_behavior_cfg,
    "libero":     build_libero_cfg,
    "robocasa":   build_robocasa_cfg,
    "robotwin":   build_robotwin_cfg,
    "roboverse":  build_roboverse_cfg,
    "frankasim":  build_frankasim_cfg,
    "calvin":     build_calvin_cfg,
    "isaaclab":   build_isaaclab_cfg,
    "polaris":    build_isaaclab_cfg,
    "habitat":    build_habitat_cfg,
}


def build_config(env_type: str, task_name: str, **overrides: Any) -> DictConfig:
    """Build a minimal DictConfig for *env_type* / *task_name*.

    All builders accept a unified ``render_mode`` kwarg:
    ``"human"``, ``"rgb_array"``, or ``None``.
    """
    builder = _BUILDERS.get(env_type)
    if builder is None:
        raise KeyError(
            f"No config builder for env_type={env_type!r}. "
            f"Supported: {sorted(_BUILDERS)}"
        )
    return builder(task_name, **overrides)
