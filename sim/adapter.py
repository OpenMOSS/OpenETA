"""UnifiedSimulatorAdapter — bridges UnifiedEnv to OpenETA protocol.

Converts UnifiedEnv dict obs → EnvObservation (CameraFrame + RobotState
+ objects + task), and passes EnvAction to the underlying env.
No code-policy sandbox — pure sim bridge.
"""

from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np

from adapter.protocol import EnvAction, EnvObservation, StepResult
from adapter.sim import SimulatorAdapter
from sim.unified_env import UnifiedEnv


class UnifiedSimulatorAdapter(SimulatorAdapter):
    """SimulatorAdapter backed by a :class:`UnifiedEnv`.

    Usage::

        adapter = make_sim_adapter("openeta/metaworld_50_assembly-v3-v0")
        obs = adapter.reset(task="assembly")
        result = adapter.step(EnvAction(action_type="code_policy", code="", command={}))
    """

    def __init__(self, env: UnifiedEnv) -> None:
        self._env = env
        self._last_obs: dict[str, Any] | None = None
        self._task: str = ""

    # ── SimulatorAdapter ABC ────────────────────────────────────

    def reset(self, *, task: str | None = None, seed: int | None = None) -> EnvObservation:
        self._task = task or ""
        obs_dict, _info = self._env.reset(seed=seed)
        self._last_obs = obs_dict
        if not obs_dict.get("task_description") and self._task:
            obs_dict["task_description"] = self._task
        return self._dict_to_env_obs(obs_dict)

    def observe(self) -> EnvObservation:
        if self._last_obs is None:
            raise RuntimeError("Must call reset() before observe()")
        return self._dict_to_env_obs(self._last_obs)

    def step(self, action: EnvAction) -> StepResult:
        raw_action = self._env_action_to_raw(action)
        obs_dict, reward, terminated, truncated, info = self._env.step(raw_action)
        self._last_obs = obs_dict
        merged_info: dict = info if isinstance(info, dict) else {"raw_info": info}
        return StepResult(
            observation=self._dict_to_env_obs(obs_dict),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=merged_info,
        )

    def close(self) -> None:
        self._env.close()

    # ── action conversion ───────────────────────────────────────

    def _env_action_to_raw(self, action: EnvAction) -> Any:
        """Map leader's EnvAction to what the underlying env expects."""
        # Command may carry action_vector
        if action.command:
            av = action.command.get("action") or action.command.get("action_vector")
            if av is not None:
                return np.asarray(av, dtype=np.float32)
        # Dict action space → pass dict
        try:
            aspace = self._detect_space()
            if isinstance(aspace, gym.spaces.Dict):
                return {"action_type": action.action_type, "code": action.code or ""}
        except Exception:
            pass
        # Box → sample
        return self._sample_action()

    def _sample_action(self) -> np.ndarray:
        try:
            aspace = self._detect_space()
            if isinstance(aspace, gym.spaces.Box):
                return aspace.sample().astype(np.float32)
        except Exception:
            pass
        return np.zeros(7, dtype=np.float32)

    def _detect_space(self) -> gym.spaces.Space | None:
        try:
            raw_env = self._env._env
            while hasattr(raw_env, "env") and not isinstance(raw_env, UnifiedEnv):
                raw_env = raw_env.env
            return getattr(raw_env, "action_space", None)
        except Exception:
            return None

    # ── properties ──────────────────────────────────────────────

    @property
    def action_dim(self) -> int:
        try:
            aspace = self._detect_space()
            if isinstance(aspace, gym.spaces.Box):
                s = aspace.shape
                return int(s[0]) if s else 1
        except Exception:
            pass
        return 7

    @property
    def action_space_desc(self) -> dict:
        try:
            aspace = self._detect_space()
            if aspace is None:
                return {"type": "unknown"}
            desc: dict = {"type": type(aspace).__name__}
            if isinstance(aspace, gym.spaces.Box):
                desc.update({
                    "shape": list(aspace.shape) if aspace.shape else [],
                    "dtype": str(aspace.dtype),
                    "low": float(aspace.low[0]) if aspace.low.size else None,
                    "high": float(aspace.high[0]) if aspace.high.size else None,
                })
            return desc
        except Exception:
            return {"type": "unknown"}

    # ── EnvObservation conversion ───────────────────────────────

    def _dict_to_env_obs(self, obs: dict) -> EnvObservation:
        """Convert UnifiedEnv observation dict → :class:`EnvObservation`.

        Delegates to ``EnvObservation.from_dict()`` which handles both
        UnifiedEnv raw format (numpy arrays, ``cameras`` as dict, ``proprio``)
        and the serialised format transparently.
        """
        return EnvObservation.from_dict(obs, task=self._task or None)

    @staticmethod
    def _env_obs_to_dict(env_obs: EnvObservation) -> dict:
        """Reverse of :meth:`_dict_to_env_obs` — for JSON serialisation."""
        return env_obs.to_dict()


def make_sim_adapter(env_id: str, *, task: str = "", seed: int = 0,
                     render_mode: str | None = "rgb_array", **kwargs) -> UnifiedSimulatorAdapter:
    import sim.env_registry  # noqa: F401
    if task:
        kwargs.setdefault("task", task)
    env = gym.make(env_id, seed=seed, render_mode=render_mode, **kwargs)
    if not isinstance(env, UnifiedEnv):
        env = UnifiedEnv(env, render_mode=render_mode)
    return UnifiedSimulatorAdapter(env)
