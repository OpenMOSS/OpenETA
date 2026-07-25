"""Host-owned depth-prior prefetching for perception tool composition."""

from __future__ import annotations

import copy
import json
import math
import threading
import time
from collections import OrderedDict
from concurrent.futures import Future
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from adapter.protocol import EnvObservation, JsonDict
from agent.tools.registry import (
    ToolExecutionContext,
    ToolHandler,
    ToolResult,
    ToolSpec,
    make_tool_result,
)


DEFAULT_DEPTH_PREFETCH_CACHE_ENTRIES = 64


@dataclass(slots=True)
class _PrefetchEntry:
    future: Future[ToolResult]
    created_at_s: float


class DepthPriorPrefetchCoordinator:
    """Deduplicate background depth-prior calls for the same camera frame."""

    def __init__(
        self,
        handler: ToolHandler,
        *,
        spec: ToolSpec,
        max_entries: int = DEFAULT_DEPTH_PREFETCH_CACHE_ENTRIES,
    ) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self._handler = handler
        self._spec = spec
        self._max_entries = int(max_entries)
        self._entries: OrderedDict[tuple[Any, ...], _PrefetchEntry] = OrderedDict()
        self._entries_lock = threading.Lock()
        self._inference_slot = threading.BoundedSemaphore(1)

    def prefetch_for_sam3(
        self,
        context: ToolExecutionContext,
        source_rgb: str,
    ) -> JsonDict:
        """Start one depth prediction without delaying the SAM3 call."""

        parameters = _depth_parameters_from_observation(
            source_rgb=source_rgb,
            observation=context.observation,
        )
        if parameters is None:
            return {
                "status": "skipped",
                "reason": "matching_camera_intrinsics_unavailable",
            }
        depth_context = ToolExecutionContext(
            name=self._spec.name,
            spec=self._spec,
            parameters=parameters,
            observation=context.observation,
            metadata=dict(context.metadata),
        )
        entry, cache_hit = self._get_or_start(depth_context)
        status = "running"
        if entry.future.done():
            try:
                status = "ready" if entry.future.result().success else "failed"
            except Exception:  # noqa: BLE001 - background failures stay isolated.
                status = "failed"
        return {
            "status": status,
            "cache_hit": cache_hit,
            "source_rgb": parameters["rgb"],
            "camera_id": parameters["camera_id"],
            "next_tool_hint": (
                "Call estimate_depth_prior with this same rgb, intrinsics, and "
                "camera_id before enhance_depth; the host will reuse this prefetch."
            ),
        }

    def handler(self, context: ToolExecutionContext) -> ToolResult:
        """Return a prefetched result or run the ordinary handler once."""

        request_context = _enrich_depth_context(context)
        key = _request_key(request_context)
        if key is None:
            return _coerce_handler_result(
                self._handler(request_context),
                context=request_context,
            )
        started = time.monotonic()
        entry, cache_hit = self._get_or_start(request_context)
        retry_after_prefetch_failure = False
        try:
            result = copy.deepcopy(entry.future.result())
        except Exception as exc:  # noqa: BLE001 - failed futures become ToolResult.
            if not cache_hit:
                return _prefetch_exception_result(
                    request_context,
                    exc=exc,
                    cache_hit=False,
                    started=started,
                )
            self._discard(key, entry)
            retry_after_prefetch_failure = True
            retry_entry, _ = self._get_or_start(request_context)
            try:
                result = copy.deepcopy(retry_entry.future.result())
            except Exception as retry_exc:  # noqa: BLE001 - structured failure.
                return _prefetch_exception_result(
                    request_context,
                    exc=retry_exc,
                    cache_hit=True,
                    started=started,
                    retried=True,
                )
        if cache_hit and not retry_after_prefetch_failure and not result.success:
            self._discard(key, entry)
            retry_after_prefetch_failure = True
            retry_entry, _ = self._get_or_start(request_context)
            try:
                result = copy.deepcopy(retry_entry.future.result())
            except Exception as retry_exc:  # noqa: BLE001 - structured failure.
                return _prefetch_exception_result(
                    request_context,
                    exc=retry_exc,
                    cache_hit=True,
                    started=started,
                    retried=True,
                )
        details = dict(result.details)
        outputs = details.get("outputs")
        outputs = dict(outputs) if isinstance(outputs, Mapping) else {}
        outputs["prefetch"] = {
            "cache_hit": cache_hit,
            "retried_after_prefetch_failure": retry_after_prefetch_failure,
            "wait_seconds": round(time.monotonic() - started, 6),
        }
        details["outputs"] = outputs
        result.details = details
        return result

    def _get_or_start(
        self,
        context: ToolExecutionContext,
    ) -> tuple[_PrefetchEntry, bool]:
        key = _request_key(context)
        if key is None:
            raise ValueError("depth prefetch requires a valid local RGB and intrinsics")
        with self._entries_lock:
            entry = self._entries.get(key)
            if entry is not None:
                self._entries.move_to_end(key)
                return entry, True
            self._prune_completed_locked()
            future: Future[ToolResult] = Future()
            entry = _PrefetchEntry(future=future, created_at_s=time.monotonic())
            self._entries[key] = entry

        worker_context = ToolExecutionContext(
            name=context.name,
            spec=context.spec,
            parameters=copy.deepcopy(context.parameters),
            observation=context.observation,
            metadata=dict(context.metadata),
        )
        threading.Thread(
            target=self._run,
            args=(future, worker_context),
            name="openeta-depth-prefetch",
            daemon=True,
        ).start()
        return entry, False

    def _discard(
        self,
        key: tuple[Any, ...],
        entry: _PrefetchEntry,
    ) -> None:
        with self._entries_lock:
            if self._entries.get(key) is entry:
                self._entries.pop(key, None)

    def _run(
        self,
        future: Future[ToolResult],
        context: ToolExecutionContext,
    ) -> None:
        try:
            with self._inference_slot:
                value = self._handler(context)
                result = _coerce_handler_result(value, context=context)
        except BaseException as exc:  # noqa: BLE001 - Future must always settle.
            future.set_exception(exc)
        else:
            future.set_result(result)

    def _prune_completed_locked(self) -> None:
        if len(self._entries) < self._max_entries:
            return
        completed = [key for key, entry in self._entries.items() if entry.future.done()]
        while len(self._entries) >= self._max_entries and completed:
            self._entries.pop(completed.pop(0), None)


def _depth_parameters_from_observation(
    *,
    source_rgb: str,
    observation: EnvObservation | None,
) -> JsonDict | None:
    if observation is None:
        return None
    resolved_rgb = _resolved_existing_path(source_rgb)
    if resolved_rgb is None:
        return None
    frame_id = _artifact_frame_id(observation, resolved_rgb)
    camera = next(
        (
            candidate
            for candidate in observation.cameras
            if frame_id and candidate.frame_id == frame_id
        ),
        None,
    )
    if camera is None and len(observation.cameras) == 1:
        camera = observation.cameras[0]
    if camera is None or _normalized_intrinsics(camera.intrinsics) is None:
        return None
    metadata = observation.metadata if isinstance(observation.metadata, dict) else {}
    return {
        "rgb": resolved_rgb,
        "intrinsics": dict(camera.intrinsics),
        "camera_id": camera.frame_id or frame_id or "camera",
        "camera_model": str(metadata.get("camera_model") or "pinhole"),
        "calibration_profile_id": str(metadata.get("calibration_profile_id") or ""),
        "bundle_id": Path(resolved_rgb).parent.name,
    }


def _enrich_depth_context(context: ToolExecutionContext) -> ToolExecutionContext:
    rgb = context.parameters.get("rgb")
    derived = (
        _depth_parameters_from_observation(
            source_rgb=rgb,
            observation=context.observation,
        )
        if isinstance(rgb, str)
        else None
    )
    if derived is None:
        return context
    parameters = dict(derived)
    parameters.update(context.parameters)
    return ToolExecutionContext(
        name=context.name,
        spec=context.spec,
        parameters=parameters,
        observation=context.observation,
        metadata=dict(context.metadata),
    )


def _artifact_frame_id(observation: EnvObservation, resolved_rgb: str) -> str:
    artifacts = observation.metadata.get("image_artifacts")
    if not isinstance(artifacts, list):
        return ""
    for artifact in artifacts:
        if not isinstance(artifact, Mapping) or artifact.get("kind") != "rgb":
            continue
        path = artifact.get("path")
        if not isinstance(path, str):
            continue
        resolved = _resolved_existing_path(path)
        if resolved == resolved_rgb:
            return str(artifact.get("frame_id") or "")
    return ""


def _request_key(context: ToolExecutionContext) -> tuple[Any, ...] | None:
    rgb = context.parameters.get("rgb")
    intrinsics = _normalized_intrinsics(context.parameters.get("intrinsics"))
    if not isinstance(rgb, str) or intrinsics is None:
        return None
    resolved_rgb = _resolved_existing_path(rgb)
    if resolved_rgb is None:
        return None
    stat = Path(resolved_rgb).stat()
    resolution = context.parameters.get("resolution_level")
    try:
        normalized_resolution = None if resolution is None else int(resolution)
    except (TypeError, ValueError):
        normalized_resolution = json.dumps(resolution, sort_keys=True, default=str)
    return (
        str(context.metadata.get("session_id") or ""),
        resolved_rgb,
        stat.st_mtime_ns,
        stat.st_size,
        tuple(intrinsics[key] for key in ("fx", "fy", "cx", "cy")),
        str(context.parameters.get("camera_id") or "camera"),
        str(context.parameters.get("camera_model") or "pinhole"),
        str(context.parameters.get("calibration_profile_id") or ""),
        normalized_resolution,
    )


def _normalized_intrinsics(value: object) -> dict[str, float] | None:
    if not isinstance(value, Mapping):
        return None
    normalized: dict[str, float] = {}
    for key in ("fx", "fy", "cx", "cy"):
        try:
            number = float(value[key])
        except (KeyError, TypeError, ValueError):
            return None
        if not math.isfinite(number) or (key in {"fx", "fy"} and number <= 0):
            return None
        normalized[key] = number
    return normalized


def _resolved_existing_path(value: str) -> str | None:
    path = Path(value).expanduser()
    if not path.is_file():
        return None
    return str(path.resolve())


def _coerce_handler_result(
    value: ToolResult | JsonDict | str | None,
    *,
    context: ToolExecutionContext,
) -> ToolResult:
    if isinstance(value, ToolResult):
        return value
    if isinstance(value, dict):
        details = value.get("details")
        return ToolResult(
            bool(value.get("success", True)),
            content=str(value.get("content") or ""),
            details=dict(details) if isinstance(details, Mapping) else dict(value),
        )
    if isinstance(value, str):
        return ToolResult(True, content=value)
    if value is None:
        return make_tool_result(
            context,
            success=False,
            content="Depth prior handler returned no result.",
            diagnostics=[{"code": "empty_handler_result"}],
        )
    raise TypeError(f"unsupported depth prior handler result: {type(value).__name__}")


def _prefetch_exception_result(
    context: ToolExecutionContext,
    *,
    exc: Exception,
    cache_hit: bool,
    started: float,
    retried: bool = False,
) -> ToolResult:
    return make_tool_result(
        context,
        success=False,
        content=f"Depth prior prefetch failed: {exc}",
        outputs={
            "reason": "prefetch_failed",
            "prefetch": {
                "cache_hit": cache_hit,
                "retried_after_prefetch_failure": retried,
                "wait_seconds": round(time.monotonic() - started, 6),
            },
        },
        diagnostics=[
            {
                "code": "prefetch_failed",
                "error_type": type(exc).__name__,
                "message": str(exc),
            }
        ],
    )
