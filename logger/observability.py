"""Small canonical observability writer for embodied episodes.

This module is intentionally independent from the existing OpenETA runtime.
It provides the correlation layer that can later be used by simulator MCP,
perception tools, and the Operator Codex episode runner without replacing the
legacy logger or response-artifact formats.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np
from PIL import Image


JsonValue = Any


def _json_default(value: Any) -> str:
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


@dataclass(frozen=True)
class FrameRecord:
    """One retained camera frame, distinct from the camera stream id."""

    frame_id: str
    observation_id: str
    camera_id: str
    source: str
    captured_at_s: float
    sim_step: int | None
    rgb_path: str | None
    depth_path: str | None
    metadata: dict[str, JsonValue]

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "frame_id": self.frame_id,
            "observation_id": self.observation_id,
            "camera_id": self.camera_id,
            "source": self.source,
            "captured_at_s": self.captured_at_s,
            "sim_step": self.sim_step,
            "rgb_path": self.rgb_path,
            "depth_path": self.depth_path,
            "metadata": _jsonable(self.metadata),
        }


class EpisodeObservability:
    """Write correlated episode metadata, frames, and events to disk.

    The writer does not own a simulator and does not interpret tool payloads.
    It only assigns stable ids, copies existing image files into collision-free
    episode paths when they exist, and appends an event envelope to JSONL.
    """

    schema_version = "openeta.observability.v1"

    def __init__(
        self,
        root: str | Path,
        *,
        episode_id: str,
        session_id: str = "",
        task: str = "",
        env_id: str = "",
        seed: int | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.frames_root = self.root / "media" / "frames"
        self.frames_root.mkdir(parents=True, exist_ok=True)
        self.episode_path = self.root / "episode.json"
        self.events_path = self.root / "events.jsonl"
        self.episode_id = episode_id
        self.session_id = session_id
        self._clock = clock
        self._seq = 0
        self._counters: dict[str, int] = {}
        self._episode = {
            "schema_version": self.schema_version,
            "episode_id": episode_id,
            "session_id": session_id,
            "task": task,
            "env_id": env_id,
            "seed": seed,
            "status": "running",
            "success": None,
            "started_at_s": float(clock()),
            "metadata": _jsonable(dict(metadata or {})),
        }
        self._write_episode()
        self._append_event(
            "episode_start",
            payload={"episode": dict(self._episode)},
        )

    def _next_id(self, prefix: str) -> str:
        value = self._counters.get(prefix, 0) + 1
        self._counters[prefix] = value
        return f"{prefix}-{value:06d}"

    def _write_episode(self) -> None:
        self.episode_path.write_text(
            json.dumps(self._episode, ensure_ascii=False, indent=2, default=_json_default)
            + "\n",
            encoding="utf-8",
        )

    def _append_event(
        self,
        kind: str,
        *,
        payload: Mapping[str, JsonValue] | None = None,
        turn_id: int | str | None = None,
        action_id: str | None = None,
        tool_call_id: str | None = None,
        parent_call_id: str | None = None,
        input_frames: Iterable[str] = (),
        output_frames: Iterable[str] = (),
        post_frames: Iterable[str] = (),
        artifact_refs: Iterable[str] = (),
    ) -> dict[str, JsonValue]:
        self._seq += 1
        event = {
            "schema_version": self.schema_version,
            "event_id": f"evt-{self._seq:06d}",
            "seq": self._seq,
            "episode_id": self.episode_id,
            "session_id": self.session_id,
            "turn_id": turn_id,
            "timestamp_s": float(self._clock()),
            "kind": kind,
            "action_id": action_id,
            "tool_call_id": tool_call_id,
            "parent_call_id": parent_call_id,
            "frame_refs": {
                "input": list(input_frames),
                "output": list(output_frames),
                "post": list(post_frames),
            },
            "artifact_refs": list(artifact_refs),
            "payload": _jsonable(dict(payload or {})),
        }
        with self.events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False, default=_json_default) + "\n")
        return event

    def _copy_artifact(self, source: str | Path | None, frame_id: str, channel: str) -> str | None:
        if source is None:
            return None
        source_path = Path(source)
        if not source_path.exists() or not source_path.is_file():
            return str(source)
        suffix = source_path.suffix or ".bin"
        target = self.frames_root / f"{frame_id}.{channel}{suffix}"
        shutil.copy2(source_path, target)
        return str(target.relative_to(self.root))

    def _write_array_artifact(self, value: Any, frame_id: str, channel: str) -> str | None:
        """Materialize an in-memory RGB or metric-depth array as a PNG.

        Observation facades normally receive numpy arrays directly from a
        simulator.  Keeping this conversion here means every retained frame
        still goes through the same collision-free naming and event path as a
        pre-existing artifact file.
        """
        if value is None:
            return None
        array = np.asarray(value)
        if array.size == 0:
            return None
        if channel == "rgb":
            if array.ndim != 3 or array.shape[-1] < 3:
                raise ValueError(f"RGB observation must have shape (H, W, 3+), got {array.shape}")
            array = np.clip(array[..., :3], 0, 255).astype(np.uint8)
            image = Image.fromarray(array)
        elif channel == "depth":
            if array.ndim != 2:
                array = np.squeeze(array)
            if array.ndim != 2:
                raise ValueError(f"Depth observation must have shape (H, W), got {array.shape}")
            # Canonical OpenETA depth is metric metres.  PNG is retained as
            # uint16 millimetres so it is loss-bounded and easy to decode.
            array = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            array = np.clip(array * 1000.0, 0.0, 65535.0).astype(np.uint16)
            image = Image.fromarray(array)
        else:
            raise ValueError(f"Unsupported observation channel: {channel!r}")

        target = self.frames_root / f"{frame_id}.{channel}.png"
        image.save(target)
        return str(target.relative_to(self.root))

    def _materialize_value(self, value: Any, frame_id: str, channel: str) -> str | None:
        if value is None:
            return None
        if isinstance(value, (str, Path)):
            return self._copy_artifact(value, frame_id, channel)
        return self._write_array_artifact(value, frame_id, channel)

    def record_observation(
        self,
        cameras: Iterable[Mapping[str, JsonValue]],
        *,
        source: str = "observe",
        sim_step: int | None = None,
        captured_at_s: float | None = None,
        turn_id: int | str | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> dict[str, JsonValue]:
        """Record one observation, potentially containing multiple cameras."""

        observation_id = self._next_id("obs")
        capture_time = float(self._clock() if captured_at_s is None else captured_at_s)
        frames: list[FrameRecord] = []
        for camera in cameras:
            camera_id = str(camera.get("camera_id") or camera.get("frame_id") or "camera")
            frame_id = f"{self._next_id('frame')}-{camera_id}"
            frames.append(
                FrameRecord(
                    frame_id=frame_id,
                    observation_id=observation_id,
                    camera_id=camera_id,
                    source=source,
                    captured_at_s=capture_time,
                    sim_step=sim_step,
                    rgb_path=self._materialize_value(
                        camera.get("rgb_path", camera.get("rgb")), frame_id, "rgb"
                    ),
                    depth_path=self._materialize_value(
                        camera.get("depth_path", camera.get("depth")), frame_id, "depth"
                    ),
                    metadata=dict(camera.get("metadata") or {}),
                )
            )
        frame_dicts = [frame.to_dict() for frame in frames]
        event = self._append_event(
            "observation",
            turn_id=turn_id,
            output_frames=[frame.frame_id for frame in frames],
            payload={
                "observation_id": observation_id,
                "source": source,
                "sim_step": sim_step,
                "captured_at_s": capture_time,
                "frames": frame_dicts,
                "metadata": dict(metadata or {}),
            },
        )
        return {
            "observation_id": observation_id,
            "frame_ids": [frame.frame_id for frame in frames],
            "frames": frame_dicts,
            "event_id": event["event_id"],
        }

    def record_action(
        self,
        *,
        request: Mapping[str, JsonValue],
        input_frames: Iterable[str] = (),
        turn_id: int | str | None = None,
        action_id: str | None = None,
    ) -> dict[str, JsonValue]:
        action_id = action_id or self._next_id("action")
        event = self._append_event(
            "action",
            turn_id=turn_id,
            action_id=action_id,
            input_frames=input_frames,
            payload={"request": dict(request)},
        )
        return {"action_id": action_id, "event_id": event["event_id"]}

    def record_tool_start(
        self,
        *,
        tool: str,
        action_id: str | None = None,
        tool_call_id: str | None = None,
        input_frames: Iterable[str] = (),
        turn_id: int | str | None = None,
        parent_call_id: str | None = None,
        parameters: Mapping[str, JsonValue] | None = None,
    ) -> dict[str, JsonValue]:
        tool_call_id = tool_call_id or self._next_id("call")
        event = self._append_event(
            "tool_start",
            turn_id=turn_id,
            action_id=action_id,
            tool_call_id=tool_call_id,
            parent_call_id=parent_call_id,
            input_frames=input_frames,
            payload={"tool": tool, "parameters": dict(parameters or {})},
        )
        return {"tool_call_id": tool_call_id, "event_id": event["event_id"]}

    def record_tool_result(
        self,
        *,
        tool: str,
        success: bool,
        action_id: str | None = None,
        tool_call_id: str | None = None,
        input_frames: Iterable[str] = (),
        output_frames: Iterable[str] = (),
        post_frames: Iterable[str] = (),
        artifact_refs: Iterable[str] = (),
        turn_id: int | str | None = None,
        result: Mapping[str, JsonValue] | None = None,
    ) -> dict[str, JsonValue]:
        event = self._append_event(
            "tool_result",
            turn_id=turn_id,
            action_id=action_id,
            tool_call_id=tool_call_id,
            input_frames=input_frames,
            output_frames=output_frames,
            post_frames=post_frames,
            artifact_refs=artifact_refs,
            payload={"tool": tool, "success": bool(success), "result": dict(result or {})},
        )
        return {"event_id": event["event_id"], "tool_call_id": tool_call_id}

    def record_failure_case(
        self,
        *,
        category: str,
        component: str,
        code: str,
        message: str,
        tool: str | None = None,
        action_id: str | None = None,
        input_frames: Iterable[str] = (),
        artifact_refs: Iterable[str] = (),
        details: Mapping[str, JsonValue] | None = None,
    ) -> dict[str, JsonValue]:
        """Persist the first terminal failure for this episode.

        Tool failures remain useful local evidence, but they are not enough to
        tell a completed episode from an abandoned one.  This event is the
        explicit episode-level failure boundary.  The full backend payload
        stays in its normal artifact, while this record keeps the stable
        component/code/message and the references needed for replay.
        """

        failure_case_id = self._next_id("failure")
        failure = {
            "failure_case_id": failure_case_id,
            "category": str(category),
            "component": str(component),
            "code": str(code),
            "message": str(message),
            "tool": str(tool) if tool else None,
            "terminal": True,
            "details": _jsonable(dict(details or {})),
        }
        event = self._append_event(
            "failure_case",
            action_id=action_id,
            input_frames=input_frames,
            artifact_refs=artifact_refs,
            payload=failure,
        )
        self._episode.update(
            {
                "status": "failed",
                "success": False,
                "failure_case": failure,
            }
        )
        self._write_episode()
        return {"event_id": event["event_id"], **failure}

    def record_issue(
        self,
        *,
        category: str,
        component: str,
        code: str,
        message: str,
        tool: str | None = None,
        action_id: str | None = None,
        input_frames: Iterable[str] = (),
        artifact_refs: Iterable[str] = (),
        details: Mapping[str, JsonValue] | None = None,
    ) -> dict[str, JsonValue]:
        """Persist a nonterminal attempt/tool issue without ending the episode."""

        issue_id = self._next_id("issue")
        issue = {
            "issue_id": issue_id,
            "category": str(category),
            "component": str(component),
            "code": str(code),
            "message": str(message),
            "tool": str(tool) if tool else None,
            "terminal": False,
            "details": _jsonable(dict(details or {})),
        }
        event = self._append_event(
            "issue",
            action_id=action_id,
            input_frames=input_frames,
            artifact_refs=artifact_refs,
            payload=issue,
        )
        return {"event_id": event["event_id"], **issue}

    def record_operator_choice(
        self,
        *,
        choice_type: str,
        choice_id: str,
        input_frames: Iterable[str] = (),
        turn_id: int | str | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> dict[str, JsonValue]:
        """Record a semantic Operator choice without duplicating raw pose data.

        The Operator chooses stable visual labels such as ``detection_000`` or
        ``G0``.  The host-side episode trace records that decision and its
        input frame, while the full candidate/detection payload remains in the
        retained tool result artifact.
        """

        event = self._append_event(
            "operator_choice",
            turn_id=turn_id,
            input_frames=input_frames,
            payload={
                "choice_type": str(choice_type),
                "choice_id": str(choice_id),
                "metadata": dict(metadata or {}),
            },
        )
        return {"event_id": event["event_id"], "choice_id": str(choice_id)}

    def finish(
        self,
        *,
        status: str,
        success: bool | None,
        final_frames: Iterable[str] = (),
        result: Mapping[str, JsonValue] | None = None,
    ) -> dict[str, JsonValue]:
        final_frame_list = list(final_frames)
        event = self._append_event(
            "episode_end",
            output_frames=final_frame_list,
            payload={"status": status, "success": success, "result": dict(result or {})},
        )
        self._episode.update(
            {
                "status": status,
                "success": success,
                "finished_at_s": float(self._clock()),
                "final_frame_ids": final_frame_list,
                "result": _jsonable(dict(result or {})),
            }
        )
        self._write_episode()
        return {"event_id": event["event_id"], "status": status, "success": success}
