"""Structured episode logger with media stored beside compact JSONL events."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import time
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from adapter.protocol import EnvAction, EnvObservation, StepResult


EPISODE_SCHEMA_VERSION = "openeta.episode.v1"
STEP_SCHEMA_VERSION = "openeta.episode_step.v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "camera"


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else 0.0
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except Exception:
            pass
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return result.stdout.strip()
    except Exception:
        return ""


class EpisodeLogger:
    """Write one episode as metadata JSON, step JSONL, RGB and depth files."""

    def __init__(self, root: str | Path = "artifacts/episodes") -> None:
        self.root = Path(root)
        self.episode_id = ""
        self.episode_dir: Path | None = None
        self._metadata: dict[str, Any] = {}
        self._steps = 0
        self._started_monotonic = 0.0
        self._stream = None

    @property
    def active(self) -> bool:
        return self._stream is not None

    def start_episode(
        self,
        *,
        task: str,
        environment: str = "",
        seed: int | None = None,
        episode_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        if self.active:
            raise RuntimeError("EpisodeLogger already has an active episode")
        self.episode_id = episode_id or (
            datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        )
        self.episode_dir = self.root / _safe_name(self.episode_id)
        self.episode_dir.mkdir(parents=True, exist_ok=False)
        (self.episode_dir / "media").mkdir()
        self._stream = (self.episode_dir / "steps.jsonl").open("a", encoding="utf-8")
        self._steps = 0
        self._started_monotonic = time.monotonic()
        self._metadata = {
            "schema_version": EPISODE_SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "task": task,
            "environment": environment,
            "seed": seed,
            "git_commit": _git_commit(),
            "started_at": _utc_now(),
            "status": "running",
            "success": None,
            "step_count": 0,
            "metadata": _json_safe(metadata or {}),
        }
        self._write_metadata()
        return self.episode_dir

    def _require_active(self) -> Path:
        if not self.active or self.episode_dir is None:
            raise RuntimeError("Call start_episode() before logging")
        return self.episode_dir

    def _write_metadata(self) -> None:
        episode_dir = self._require_active()
        target = episode_dir / "episode.json"
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(_json_safe(self._metadata), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(temporary, target)

    def _save_observation_media(
        self, observation: EnvObservation, *, label: str
    ) -> list[dict[str, Any]]:
        episode_dir = self._require_active()
        media_dir = episode_dir / "media" / _safe_name(label)
        media_dir.mkdir(parents=True, exist_ok=True)
        refs: list[dict[str, Any]] = []
        for camera in observation.cameras:
            name = _safe_name(camera.frame_id)
            entry: dict[str, Any] = {"frame_id": camera.frame_id}
            rgb = np.asarray(camera.rgb, dtype=np.uint8)
            if rgb.size:
                rgb_path = media_dir / f"{name}.rgb.png"
                Image.fromarray(rgb[..., :3], mode="RGB").save(rgb_path)
                entry["rgb"] = rgb_path.relative_to(episode_dir).as_posix()
            if camera.depth is not None:
                depth = np.asarray(camera.depth, dtype=np.float32)
                depth = np.nan_to_num(depth, nan=0.0, posinf=65.535, neginf=0.0)
                depth = np.where(depth >= 0.0, depth, 0.0).astype(np.float32)
                depth_mm = np.clip(np.round(depth * 1000.0), 0, 65535).astype(np.uint16)
                depth_path = media_dir / f"{name}.depth.png"
                Image.fromarray(depth_mm).save(depth_path)
                entry["depth"] = depth_path.relative_to(episode_dir).as_posix()
                finite = depth[np.isfinite(depth)]
                entry["depth_unit"] = "millimetres_uint16_png"
                entry["depth_min_m"] = float(finite.min()) if finite.size else 0.0
                entry["depth_max_m"] = float(finite.max()) if finite.size else 0.0
            entry["intrinsics"] = _json_safe(camera.intrinsics)
            entry["extrinsics"] = _json_safe(camera.extrinsics)
            refs.append(entry)
        return refs

    @staticmethod
    def _observation_record(
        observation: EnvObservation, camera_refs: list[dict[str, Any]]
    ) -> dict[str, Any]:
        return {
            "task": observation.task,
            "robot": _json_safe(observation.robot.to_dict()),
            "objects": _json_safe(observation.objects),
            "metadata": _json_safe(observation.metadata),
            "camera_refs": camera_refs,
        }

    def log_step(
        self,
        *,
        step_index: int,
        observation: EnvObservation,
        action: EnvAction,
        result: StepResult | None,
        plan: dict[str, Any] | None = None,
        safety_verdict: dict[str, Any] | None = None,
        failure_verdict: dict[str, Any] | None = None,
        latency_ms: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        self._require_active()
        camera_refs = self._save_observation_media(
            observation, label=f"step_{int(step_index):06d}"
        )
        event: dict[str, Any] = {
            "schema_version": STEP_SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "step_index": int(step_index),
            "timestamp_s": time.time(),
            "observation": self._observation_record(observation, camera_refs),
            "plan": _json_safe(plan or {}),
            "safety_verdict": _json_safe(safety_verdict or {}),
            "action": _json_safe(action.to_dict()),
            "step_result": None,
            "failure_verdict": _json_safe(failure_verdict or {}),
            "latency_ms": _json_safe(latency_ms or {}),
            "metadata": _json_safe(metadata or {}),
        }
        if result is not None:
            event["step_result"] = {
                "reward": float(result.reward),
                "terminated": bool(result.terminated),
                "truncated": bool(result.truncated),
                "info": _json_safe(result.info),
                "observation": {
                    "task": result.observation.task,
                    "robot": _json_safe(result.observation.robot.to_dict()),
                    "objects": _json_safe(result.observation.objects),
                    "metadata": _json_safe(result.observation.metadata),
                },
            }
        assert self._stream is not None
        self._stream.write(json.dumps(event, ensure_ascii=False) + "\n")
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._steps = max(self._steps, int(step_index) + 1)
        return event

    def finish_episode(
        self,
        *,
        status: str,
        success: bool | None,
        final_observation: EnvObservation | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        episode_dir = self._require_active()
        final_refs: list[dict[str, Any]] = []
        if final_observation is not None:
            final_refs = self._save_observation_media(final_observation, label="final")
        self._metadata.update(
            {
                "status": status,
                "success": success,
                "step_count": self._steps,
                "finished_at": _utc_now(),
                "duration_s": time.monotonic() - self._started_monotonic,
                "final_camera_refs": final_refs,
            }
        )
        if metadata:
            current = self._metadata.setdefault("metadata", {})
            if isinstance(current, dict):
                current.update(_json_safe(metadata))
        self._write_metadata()
        assert self._stream is not None
        self._stream.close()
        self._stream = None
        return episode_dir

    def abort_episode(self, error: BaseException) -> Path:
        return self.finish_episode(
            status="error",
            success=False,
            metadata={"error": {"type": type(error).__name__, "message": str(error)}},
        )
