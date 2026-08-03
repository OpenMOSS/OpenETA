"""LIBERO observation-bound facade for the visual embodied harness.

The facade deliberately sits above :class:`sim.unified_env.UnifiedEnv`.
LIBERO-specific RGB-D normalization and camera calibration remain in the
existing unified wrapper; this module only adds episode correlation, retained
visual artifacts, and explicit access to simulator-native evaluation hooks.

The normal Operator Codex path consumes the returned observation and the
materialized frame record.  ``check_success`` and ``get_sim_state`` are
separate methods for host/evaluator diagnostics and are never injected into
the normal visual observation automatically.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from logger.observability import EpisodeObservability


class LiberoObservationFacade:
    """Record canonical LIBERO observations around an existing environment.

    Parameters
    ----------
    env:
        Usually an ``openeta/...libero...`` environment returned by
        ``gym.make`` and therefore already wrapped by ``UnifiedEnv``.
    observability:
        The episode-local :class:`EpisodeObservability` writer.
    record_privileged_state:
        If true, include any optional ``objects`` field in event metadata.
        The default is false so a visual Operator episode does not silently
        receive privileged scene state through its trace payload.
    """

    def __init__(
        self,
        env: Any,
        observability: EpisodeObservability,
        *,
        record_privileged_state: bool = False,
    ) -> None:
        self.env = env
        self.observability = observability
        self.record_privileged_state = record_privileged_state
        self.sim_step = 0
        self.last_observation: dict[str, Any] | None = None
        self.last_record: dict[str, Any] | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        """Reset LIBERO and retain the initial RGB-D observation."""
        kwargs: dict[str, Any] = {}
        if seed is not None:
            kwargs["seed"] = seed
        if options is not None:
            kwargs["options"] = options
        result = self.env.reset(**kwargs)
        observation, info = self._unpack_reset(result)
        self.sim_step = 0
        self.last_observation = observation
        self.last_record = self._record(observation, source="reset")
        return observation, self._with_record(info, self.last_record)

    def observe(self, *, source: str = "observe") -> dict[str, Any]:
        """Retain the latest canonical observation without stepping."""
        if self.last_observation is None:
            raise RuntimeError("observe() requires reset() first")
        self.last_record = self._record(self.last_observation, source=source)
        return self.last_record

    def step(
        self,
        action: Any,
    ) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        """Execute one normal simulator action and retain its post-frame."""
        input_frames = list((self.last_record or {}).get("frame_ids", []))
        action_record = self.observability.record_action(
            request={"action": action},
            input_frames=input_frames,
        )
        try:
            raw_result = self.env.step(action)
        except Exception as exc:
            self.observability.record_tool_result(
                tool="sim.step",
                success=False,
                action_id=action_record["action_id"],
                input_frames=input_frames,
                result={"error_type": type(exc).__name__, "error": str(exc)},
            )
            raise
        observation, reward, terminated, truncated, info = self._unpack_step(raw_result)
        self.sim_step += 1
        self.last_observation = observation
        self.last_record = self._record(observation, source="post_step")
        self.observability.record_tool_result(
            tool="sim.step",
            success=True,
            action_id=action_record["action_id"],
            input_frames=input_frames,
            post_frames=self.last_record["frame_ids"],
            result={
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
            },
        )
        return (
            observation,
            reward,
            terminated,
            truncated,
            self._with_record(info, self.last_record),
        )

    def render_observation(self, *, source: str = "render") -> dict[str, Any]:
        """Retain a render-only RGB frame when the backend exposes one.

        This is intentionally separate from ``observe``: a render-only frame
        has no fresh depth or calibration guarantee and must not be mistaken
        for the RGB-D snapshot used by perception tools.
        """
        frame = self.env.render()
        if frame is None:
            return self.observe(source=source)
        record = self.observability.record_observation(
            [{"camera_id": "render", "rgb": np.asarray(frame)}],
            source=source,
            sim_step=self.sim_step,
        )
        self.last_record = record
        return record

    def check_success(self) -> bool | None:
        """Return the native LIBERO checker result, or ``None`` if unavailable."""
        method = self._find_method("check_success")
        if method is None:
            return None
        result = method()
        if isinstance(result, np.generic):
            result = result.item()
        return bool(result) if isinstance(result, (bool, int, float)) else result

    def get_sim_state(self) -> Any:
        """Return native simulator state for host-side diagnostics only."""
        method = self._find_method("get_sim_state")
        return method() if method is not None else None

    def close(self) -> None:
        self.env.close()

    def _record(self, observation: Mapping[str, Any], *, source: str) -> dict[str, Any]:
        cameras = []
        for camera_id, camera in (observation.get("cameras") or {}).items():
            if not isinstance(camera, Mapping):
                continue
            metadata = {
                "intrinsics": camera.get("intrinsics", {}),
                "extrinsics": camera.get("extrinsics", {}),
            }
            if camera.get("rgb") is not None:
                metadata["rgb_shape"] = list(np.asarray(camera["rgb"]).shape)
            if camera.get("depth") is not None:
                metadata["depth_shape"] = list(np.asarray(camera["depth"]).shape)
                metadata["depth_unit"] = "metres_float32"
            cameras.append(
                {
                    "camera_id": str(camera_id),
                    "rgb": camera.get("rgb"),
                    "depth": camera.get("depth"),
                    "metadata": metadata,
                }
            )

        metadata: dict[str, Any] = {
            "task_description": observation.get("task_description", ""),
            "proprio": observation.get("proprio", {}),
            "metadata": observation.get("metadata", {}),
        }
        if self.record_privileged_state and "objects" in observation:
            metadata["objects"] = observation["objects"]
        return self.observability.record_observation(
            cameras,
            source=source,
            sim_step=self.sim_step,
            metadata=metadata,
        )

    @staticmethod
    def _unpack_reset(result: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        if isinstance(result, tuple) and len(result) == 2:
            observation, info = result
            return dict(observation), dict(info or {})
        return dict(result), {}

    @staticmethod
    def _unpack_step(result: Any) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if len(result) == 5:
            observation, reward, terminated, truncated, info = result
        elif len(result) == 4:
            observation, reward, terminated, info = result
            truncated = False
        else:
            raise ValueError(f"Expected a 4- or 5-tuple from env.step(), got {len(result)}")
        return dict(observation), float(reward), bool(terminated), bool(truncated), dict(info or {})

    @staticmethod
    def _with_record(info: Mapping[str, Any], record: Mapping[str, Any]) -> dict[str, Any]:
        result = dict(info)
        result["observation_record"] = dict(record)
        return result

    def _find_method(self, name: str):
        seen: set[int] = set()
        pending = [self.env]
        while pending:
            candidate = pending.pop(0)
            if candidate is None or id(candidate) in seen:
                continue
            seen.add(id(candidate))
            method = getattr(candidate, name, None)
            if callable(method):
                return method
            for attr in ("unwrapped", "_env", "env"):
                child = getattr(candidate, attr, None)
                if child is not None and id(child) not in seen:
                    pending.append(child)
        return None


__all__ = ["LiberoObservationFacade"]
