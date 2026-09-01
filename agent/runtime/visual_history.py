"""Deterministic raw-image windows and main-view visual differencing."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from adapter.protocol import EnvObservation, JsonDict
from agent.backends.planner import PlannerBackend, PlannerBackendRequest
from agent.runtime.actions import PipelineStatus

if TYPE_CHECKING:
    from agent.runtime.memory import AgentMemory
    from agent.runtime.rollout import RolloutRecorder


VISUAL_DELTA_SCHEMA_VERSION = "openeta.visual_delta.v1"
VISUAL_HISTORY_SCHEMA_VERSION = "openeta.visual_history.v1"
VISUAL_DELTA_PROMPT = """You are OpenETA's isolated visual differencing module.
Compare only the labelled previous and current fixed-main-camera images for the
given embodied task. Report objective visible changes. Do not infer hidden
causes, do not rely on an expected action, and explicitly report occlusion or
uncertainty. Return exactly one JSON object with four array-of-string fields:
visible_changes, task_progress_evidence, completion_evidence, uncertainties.
Use empty arrays when no evidence is visible. Do not write code or markdown.
"""


@dataclass(frozen=True, slots=True)
class VisualHistoryConfig:
    """Agent-local policy for raw visual context and derived VDM evidence."""

    enabled: bool = True
    main_camera_role: str = "agentview"
    recent_main_turns: int = 3
    include_initial_main: bool = True
    include_current_wrist: bool = True
    include_vdm: bool = True
    vdm_max_output_items: int = 8
    vdm_max_item_chars: int = 500
    # None keeps all compact textual deltas until the planner's total token
    # projector needs to evict old entries.  Explicit evaluation overrides may
    # still request the legacy fixed-window ablation.
    vdm_recent_delta_limit: int | None = None

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
    ) -> "VisualHistoryConfig":
        source = env if env is not None else os.environ
        return cls(
            enabled=_env_bool(source, "OPENETA_VISUAL_HISTORY_ENABLED", True),
            main_camera_role=(
                str(source.get("OPENETA_VDM_CAMERA_ROLE") or "agentview").strip()
                or "agentview"
            ),
            recent_main_turns=_env_positive_int(
                source,
                "OPENETA_VISUAL_RECENT_MAIN_TURNS",
                3,
            ),
            include_initial_main=_env_bool(
                source,
                "OPENETA_VISUAL_INCLUDE_INITIAL_MAIN",
                True,
            ),
            include_current_wrist=_env_bool(
                source,
                "OPENETA_VISUAL_INCLUDE_CURRENT_WRIST",
                True,
            ),
            include_vdm=_env_bool(
                source,
                "OPENETA_VISUAL_VDM_ENABLED",
                True,
            ),
            vdm_recent_delta_limit=(
                _env_positive_int(
                    source,
                    "OPENETA_VISUAL_VDM_RECENT_DELTA_LIMIT",
                    6,
                )
                if str(source.get("OPENETA_VISUAL_VDM_RECENT_DELTA_LIMIT") or "").strip()
                else None
            ),
        )

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, object] | None,
        *,
        base: "VisualHistoryConfig | None" = None,
    ) -> "VisualHistoryConfig":
        """Apply strict host-owned evaluation overrides to a base policy.

        Evaluation plans carry JSON, while interactive runs continue to use
        ``from_env``. Keeping parsing here avoids teaching the generic eval
        scheduler about visual-history fields.
        """

        current = base or cls.from_env()
        if value is None:
            return current
        allowed = {
            "enabled",
            "main_camera_role",
            "recent_main_turns",
            "include_initial_main",
            "include_current_wrist",
            "include_vdm",
            "vdm_max_output_items",
            "vdm_max_item_chars",
            "vdm_recent_delta_limit",
        }
        unknown = sorted(str(key) for key in value if key not in allowed)
        if unknown:
            raise ValueError(
                "unsupported visual_history evaluation override(s): "
                + ", ".join(unknown)
            )

        booleans: dict[str, bool] = {}
        for key in (
            "enabled",
            "include_initial_main",
            "include_current_wrist",
            "include_vdm",
        ):
            if key not in value:
                booleans[key] = bool(getattr(current, key))
                continue
            observed = value[key]
            if not isinstance(observed, bool):
                raise ValueError(f"visual_history.{key} must be boolean")
            booleans[key] = observed

        integers: dict[str, int] = {}
        for key in (
            "recent_main_turns",
            "vdm_max_output_items",
            "vdm_max_item_chars",
        ):
            observed = value.get(key, getattr(current, key))
            if isinstance(observed, bool) or not isinstance(observed, int) or observed < 1:
                raise ValueError(f"visual_history.{key} must be a positive integer")
            integers[key] = observed

        raw_delta_limit = value.get(
            "vdm_recent_delta_limit", current.vdm_recent_delta_limit
        )
        if raw_delta_limit is not None and (
            isinstance(raw_delta_limit, bool)
            or not isinstance(raw_delta_limit, int)
            or raw_delta_limit < 1
        ):
            raise ValueError(
                "visual_history.vdm_recent_delta_limit must be null or a positive integer"
            )

        role = value.get("main_camera_role", current.main_camera_role)
        if not isinstance(role, str) or not role.strip():
            raise ValueError("visual_history.main_camera_role must be non-empty text")
        return cls(
            enabled=booleans["enabled"],
            main_camera_role=role.strip(),
            recent_main_turns=integers["recent_main_turns"],
            include_initial_main=booleans["include_initial_main"],
            include_current_wrist=booleans["include_current_wrist"],
            include_vdm=booleans["include_vdm"],
            vdm_max_output_items=integers["vdm_max_output_items"],
            vdm_max_item_chars=integers["vdm_max_item_chars"],
            vdm_recent_delta_limit=raw_delta_limit,
        )

    @property
    def planner_raw_image_capacity(self) -> int:
        return (
            self.recent_main_turns
            + int(self.include_initial_main)
            + int(self.include_current_wrist)
        )

    def to_dict(self) -> JsonDict:
        return {
            "schema_version": VISUAL_HISTORY_SCHEMA_VERSION,
            "enabled": self.enabled,
            "main_camera_role": self.main_camera_role,
            "recent_main_turns": self.recent_main_turns,
            "include_initial_main": self.include_initial_main,
            "include_current_wrist": self.include_current_wrist,
            "include_vdm": self.include_vdm,
            "vdm_camera_roles": [self.main_camera_role] if self.include_vdm else [],
            "vdm_recent_delta_limit": self.vdm_recent_delta_limit,
        }


class VisualHistoryManager:
    """Generate one durable main-view delta for each adjacent observation."""

    def __init__(
        self,
        *,
        config: VisualHistoryConfig,
        backend: PlannerBackend | None,
    ) -> None:
        self.config = config
        self.backend = backend
        self.rollout_recorder: RolloutRecorder | None = None

    def set_rollout_recorder(self, recorder: "RolloutRecorder | None") -> None:
        self.rollout_recorder = recorder

    def descriptor(self) -> JsonDict:
        return {
            **self.config.to_dict(),
            "backend": self.backend.descriptor() if self.backend is not None else None,
            "prompt_sha256": _prompt_sha256(),
        }

    def observe(self, observation: EnvObservation, *, memory: "AgentMemory") -> JsonDict | None:
        """Generate or explicitly fail one adjacent visual delta.

        ``AgentMemory.add_observation`` must run first so the raw observation
        reference is durable before a derived record is attempted.
        """

        if not self.config.enabled or not self.config.include_vdm:
            return None
        observations = observation_history(memory)
        if len(observations) < 2:
            return None
        previous, current = observations[-2:]
        delta_id = _delta_id(previous, current, self.config.main_camera_role)
        existing = next(
            (
                item
                for item in visual_delta_history(memory)
                if item.get("delta_id") == delta_id
            ),
            None,
        )
        if existing is not None:
            return existing

        previous_main = _select_main_artifact(previous, self.config.main_camera_role)
        current_main = _select_main_artifact(current, self.config.main_camera_role)
        envelope = _delta_envelope(
            previous,
            current,
            previous_main=previous_main,
            current_main=current_main,
            camera_role=self.config.main_camera_role,
        )
        if previous_main is None or current_main is None:
            record = {
                **envelope,
                "status": "unavailable",
                "visible_changes": [],
                "task_progress_evidence": [],
                "completion_evidence": [],
                "uncertainties": [
                    "configured fixed-main RGB artifact is unavailable for one or both turns"
                ],
                "failure": {
                    "code": "main_view_artifact_unavailable",
                    "missing": [
                        label
                        for label, value in (
                            ("previous", previous_main),
                            ("current", current_main),
                        )
                        if value is None
                    ],
                },
            }
            memory.record("visual_delta", record)
            return record

        # Read-only tool turns still receive a freshly persisted observation,
        # but the fixed main-camera pixels are commonly byte-identical.  A VLM
        # call cannot add evidence in that case.  Preserve the adjacent-delta
        # coverage record while resolving the no-change result deterministically.
        comparison_started_at_s = time.time()
        previous_sha256 = _artifact_content_sha256(previous_main)
        current_sha256 = _artifact_content_sha256(current_main)
        if (
            previous_sha256
            and current_sha256
            and previous_sha256 == current_sha256
        ):
            record = {
                **envelope,
                "status": "no_visible_change",
                "visible_changes": [],
                "task_progress_evidence": [],
                "completion_evidence": [],
                "uncertainties": [],
                "comparison": {
                    "method": "sha256",
                    "identical": True,
                    "content_sha256": current_sha256,
                },
                "derived_by": "host_identical_image_check",
                "duration_s": max(0.0, time.time() - comparison_started_at_s),
            }
            memory.record("visual_delta", record)
            return record
        if self.backend is None:
            record = {
                **envelope,
                "status": "failed",
                "visible_changes": [],
                "task_progress_evidence": [],
                "completion_evidence": [],
                "uncertainties": ["visual differencing backend is not configured"],
                "failure": {"code": "vdm_backend_unavailable"},
            }
            memory.record("visual_delta", record)
            return record

        task = observation.task or memory.current_user_request or memory.task or ""
        request = PlannerBackendRequest(
            tool_context={
                "schema_version": "openeta.visual_delta_request.v1",
                "task": task,
                "from_observation": _public_observation_ref(previous, previous_main),
                "to_observation": _public_observation_ref(current, current_main),
                "vision_image_paths": [previous_main["path"], current_main["path"]],
                "vision_evidence": [
                    _vdm_image_evidence(previous, previous_main, role="previous_main_view"),
                    _vdm_image_evidence(current, current_main, role="current_main_view"),
                ],
                "required_output": {
                    "visible_changes": "array[string]",
                    "task_progress_evidence": "array[string]",
                    "completion_evidence": "array[string]",
                    "uncertainties": "array[string]",
                },
            },
            system_prompt=VISUAL_DELTA_PROMPT,
            metadata={
                "schema_version": "openeta.visual_delta_request.v1",
                "isolated_context": True,
                "role": "visual_differencing",
            },
        )
        started_at_s = time.time()
        try:
            result = self.backend.decide(request)
            completed_at_s = time.time()
            validation_errors: list[str] = []
            try:
                fields = _parse_delta_payload(
                    result.payload,
                    max_items=self.config.vdm_max_output_items,
                    max_chars=self.config.vdm_max_item_chars,
                )
                if result.status == PipelineStatus.FAILED:
                    raise ValueError("VDM backend returned failed status")
            except Exception as exc:  # noqa: BLE001 - recorded before outer degradation.
                validation_errors = [str(exc)[:500]]
                self._record_model_call(
                    request=request,
                    result=result,
                    decision=None,
                    validation_errors=validation_errors,
                    started_at_s=started_at_s,
                    completed_at_s=completed_at_s,
                )
                raise
            self._record_model_call(
                request=request,
                result=result,
                decision=fields,
                validation_errors=validation_errors,
                started_at_s=started_at_s,
                completed_at_s=completed_at_s,
            )
        except Exception as exc:  # noqa: BLE001 - VDM failure must not block the Agent.
            record = {
                **envelope,
                "status": "failed",
                "visible_changes": [],
                "task_progress_evidence": [],
                "completion_evidence": [],
                "uncertainties": ["visual differencing request failed"],
                "failure": {
                    "code": "vdm_request_failed",
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:500],
                },
                "duration_s": max(0.0, time.time() - started_at_s),
            }
            memory.record("visual_delta", record)
            return record

        status = (
            "available"
            if fields["visible_changes"]
            or fields["task_progress_evidence"]
            or fields["completion_evidence"]
            else "no_visible_change"
        )
        record = {
            **envelope,
            "status": status,
            **fields,
            "provider": result.provider,
            "model": result.model,
            "usage": _compact_usage(result.details),
            "duration_s": max(0.0, completed_at_s - started_at_s),
        }
        memory.record("visual_delta", record)
        return record

    def _record_model_call(
        self,
        *,
        request: PlannerBackendRequest,
        result: object,
        decision: JsonDict | None,
        validation_errors: list[str],
        started_at_s: float,
        completed_at_s: float,
    ) -> None:
        if self.rollout_recorder is None:
            return
        self.rollout_recorder.record_model_call(
            request=request,
            result=result,
            decision=decision,
            validation_errors=validation_errors,
            backend=(self.backend.descriptor() if self.backend is not None else {}),
            started_at_s=started_at_s,
            completed_at_s=completed_at_s,
        )


def build_visual_history_projection(
    *,
    observation: EnvObservation,
    memory: "AgentMemory",
    config: VisualHistoryConfig,
    current_camera_artifacts: list[JsonDict],
) -> JsonDict:
    """Build the bounded raw window and compressed delta bridge."""

    observations = observation_history(memory)
    if not observations:
        observations = [
            _synthetic_current_observation(
                observation,
                current_camera_artifacts=current_camera_artifacts,
            )
        ]

    current_index = len(observations) - 1
    recent_start = max(0, len(observations) - config.recent_main_turns)
    selected_indices: list[int] = []
    if config.include_initial_main:
        selected_indices.append(0)
    selected_indices.extend(range(recent_start, len(observations)))
    selected_indices = list(dict.fromkeys(selected_indices))

    paths: list[str] = []
    evidence: list[JsonDict] = []
    raw_evidence: list[JsonDict] = []
    for index in selected_indices:
        record = observations[index]
        artifact = _select_main_artifact(record, config.main_camera_role)
        if artifact is None:
            continue
        if index == current_index:
            role = "current_scene"
            freshness = "current"
        elif index == 0:
            role = "historical_anchor"
            freshness = "initial_anchor"
        else:
            role = "historical_scene"
            freshness = "recent_history"
        item = _planner_image_evidence(record, artifact, role=role, freshness=freshness)
        if artifact["path"] not in paths:
            paths.append(artifact["path"])
            evidence.append(item)
            raw_evidence.append(item)

    if config.include_current_wrist:
        wrist = _select_wrist_artifact(observations[-1])
        if wrist is not None and wrist["path"] not in paths:
            item = _planner_image_evidence(
                observations[-1],
                wrist,
                role="current_scene",
                freshness="current",
            )
            paths.append(wrist["path"])
            evidence.append(item)
            raw_evidence.append(item)

    deltas = visual_delta_history(memory) if config.include_vdm else []
    bridge_target = recent_start if recent_start > 1 else 0
    compressed = [
        delta
        for delta in deltas
        if _delta_to_index(delta) is not None
        and 0 < int(_delta_to_index(delta) or 0) <= bridge_target
    ]
    compressed.sort(key=lambda item: int(_delta_to_index(item) or 0))
    available_targets = {
        int(target)
        for item in compressed
        if (target := _delta_to_index(item)) is not None
        and item.get("status") in {"available", "no_visible_change"}
    }
    expected_targets = (
        set(range(1, bridge_target + 1)) if config.include_vdm else set()
    )

    if config.vdm_recent_delta_limit is None:
        projected_deltas = compressed
        compacted_deltas: list[JsonDict] = []
    else:
        projected_deltas = compressed[-config.vdm_recent_delta_limit :]
        compacted_deltas = compressed[: -config.vdm_recent_delta_limit]

    return {
        "vision_image_paths": paths,
        "vision_evidence": evidence,
        "visual_history": {
            "schema_version": VISUAL_HISTORY_SCHEMA_VERSION,
            "policy": config.to_dict(),
            "raw_evidence": raw_evidence,
            "compressed_deltas": [_model_delta(delta) for delta in projected_deltas],
            "compressed_delta_summary": _compact_delta_summary(
                compacted_deltas,
                max_items=config.vdm_max_output_items * 2,
                max_chars=config.vdm_max_item_chars,
            ),
            "coverage": {
                "from_observation_index": 0,
                "through_observation_index": current_index,
                "compressed_through_observation_index": bridge_target,
                "missing_delta_observation_indices": sorted(
                    expected_targets - available_targets
                ),
            },
        },
    }


def observation_history(memory: "AgentMemory") -> list[JsonDict]:
    return [
        dict(payload)
        for event_type, payload in _history_events(memory)
        if event_type == "observation" and isinstance(payload, dict)
    ]


def visual_delta_history(memory: "AgentMemory") -> list[JsonDict]:
    return [
        dict(payload)
        for event_type, payload in _history_events(memory)
        if event_type == "visual_delta" and isinstance(payload, dict)
    ]


def _history_events(memory: "AgentMemory") -> list[tuple[str, JsonDict]]:
    store = memory.store
    if store is not None and memory.session_id is not None:
        try:
            rows = store.load_events(memory.session_id, limit=None)
        except (OSError, ValueError, json.JSONDecodeError):
            rows = []
        if rows:
            return [
                (
                    str(row.get("event_type") or ""),
                    dict(row.get("payload") or {}),
                )
                for row in rows
                if isinstance(row, dict) and isinstance(row.get("payload"), dict)
            ]
    return [(event.event_type, dict(event.payload)) for event in memory.events]


def _delta_envelope(
    previous: JsonDict,
    current: JsonDict,
    *,
    previous_main: JsonDict | None,
    current_main: JsonDict | None,
    camera_role: str,
) -> JsonDict:
    return {
        "schema_version": VISUAL_DELTA_SCHEMA_VERSION,
        "delta_id": _delta_id(previous, current, camera_role),
        "from_observation": _public_observation_ref(previous, previous_main),
        "to_observation": _public_observation_ref(current, current_main),
        "source_camera_roles": [camera_role],
        "excluded_camera_roles": ["wrist"],
        "prompt_sha256": _prompt_sha256(),
        "created_at_s": time.time(),
    }


def _delta_id(previous: JsonDict, current: JsonDict, camera_role: str) -> str:
    return (
        f"visual_delta:{_observation_index(previous)}:"
        f"{_observation_index(current)}:{camera_role}"
    )


def _public_observation_ref(record: JsonDict, artifact: JsonDict | None) -> JsonDict:
    ref: JsonDict = {
        "observation_index": _observation_index(record),
        "environment_step": _environment_step(record),
    }
    if artifact is not None:
        ref.update(
            {
                "evidence_id": _evidence_id(record, artifact),
                "frame_id": artifact.get("frame_id", ""),
                "camera_role": artifact.get("role", ""),
                "path": artifact.get("path", ""),
            }
        )
    return ref


def _select_main_artifact(record: JsonDict, role: str) -> JsonDict | None:
    rgb = _rgb_artifacts(record)
    exact_role = next((item for item in rgb if item.get("role") == role), None)
    if exact_role is not None:
        return exact_role
    return next((item for item in rgb if item.get("frame_id") == role), None)


def _select_wrist_artifact(record: JsonDict) -> JsonDict | None:
    return next(
        (
            item
            for item in _rgb_artifacts(record)
            if str(item.get("role") or "").startswith("wrist")
            or "wrist" in str(item.get("frame_id") or "").lower()
        ),
        None,
    )


def _rgb_artifacts(record: JsonDict) -> list[JsonDict]:
    artifacts = record.get("visual_artifacts")
    if not isinstance(artifacts, list):
        artifacts = record.get("runtime_camera_sources")
    return [
        _normalize_artifact(dict(item))
        for item in artifacts or []
        if isinstance(item, dict)
        and str(item.get("kind") or "rgb") == "rgb"
        and isinstance(item.get("path") or item.get("rgb_path"), str)
        and (item.get("path") or item.get("rgb_path"))
    ]


def _normalize_artifact(artifact: JsonDict) -> JsonDict:
    normalized = dict(artifact)
    if not normalized.get("path") and normalized.get("rgb_path"):
        normalized["path"] = normalized["rgb_path"]
    return normalized


def _artifact_content_sha256(artifact: JsonDict) -> str:
    path_value = artifact.get("path") or artifact.get("rgb_path")
    if not isinstance(path_value, str) or not path_value:
        return ""
    try:
        return sha256(Path(path_value).read_bytes()).hexdigest()
    except OSError:
        return ""


def _planner_image_evidence(
    record: JsonDict,
    artifact: JsonDict,
    *,
    role: str,
    freshness: str,
) -> JsonDict:
    artifact = _normalize_artifact(artifact)
    item: JsonDict = {
        "evidence_id": _evidence_id(record, artifact),
        "role": role,
        "camera_role": artifact.get("role") or artifact.get("frame_id") or "scene",
        "frame_id": artifact.get("frame_id", ""),
        "path": artifact["path"],
        "freshness": freshness,
        "observation_index": _observation_index(record),
    }
    environment_step = _environment_step(record)
    if environment_step is not None:
        item["observation_step"] = environment_step
    if artifact.get("timestamp_s") is not None:
        item["timestamp_s"] = artifact["timestamp_s"]
    return item


def _vdm_image_evidence(record: JsonDict, artifact: JsonDict, *, role: str) -> JsonDict:
    item = _planner_image_evidence(
        record,
        artifact,
        role=role,
        freshness="historical_source",
    )
    item["camera_role"] = artifact.get("role") or artifact.get("frame_id")
    return item


def _evidence_id(record: JsonDict, artifact: JsonDict) -> str:
    return (
        f"observation:{_observation_index(record)}:"
        f"{artifact.get('frame_id') or artifact.get('role') or 'camera'}"
    )


def _observation_index(record: JsonDict) -> int:
    value = record.get("observation_index")
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


def _environment_step(record: JsonDict) -> int | None:
    value = record.get("environment_step")
    if value is None:
        metadata = record.get("metadata")
        value = metadata.get("step_idx") if isinstance(metadata, dict) else None
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _delta_to_index(record: JsonDict) -> int | None:
    target = record.get("to_observation")
    value = target.get("observation_index") if isinstance(target, dict) else None
    return int(value) if isinstance(value, int) and not isinstance(value, bool) else None


def _model_delta(record: JsonDict) -> JsonDict:
    return {
        key: record.get(key)
        for key in (
            "schema_version",
            "delta_id",
            "from_observation",
            "to_observation",
            "source_camera_roles",
            "excluded_camera_roles",
            "status",
            "visible_changes",
            "task_progress_evidence",
            "completion_evidence",
            "uncertainties",
            "prompt_sha256",
        )
        if record.get(key) is not None
    }


def _compact_delta_summary(
    records: list[JsonDict],
    *,
    max_items: int,
    max_chars: int,
) -> JsonDict | None:
    """Bound older VDM text while the full adjacent-delta history stays durable."""

    if not records:
        return None
    per_field_limit = max(1, max_items // 4)
    fields: JsonDict = {}
    for key in (
        "visible_changes",
        "task_progress_evidence",
        "completion_evidence",
        "uncertainties",
    ):
        selected: list[str] = []
        seen: set[str] = set()
        for record in reversed(records):
            values = record.get(key)
            if not isinstance(values, list):
                continue
            for raw in reversed(values):
                if not isinstance(raw, str):
                    continue
                value = raw.strip()[:max_chars]
                if not value or value in seen:
                    continue
                selected.append(value)
                seen.add(value)
                if len(selected) >= per_field_limit:
                    break
            if len(selected) >= per_field_limit:
                break
        fields[key] = list(reversed(selected))
    status_counts: dict[str, int] = {}
    for record in records:
        status = str(record.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    first_index = _delta_to_index(records[0])
    last_index = _delta_to_index(records[-1])
    return {
        "schema_version": "openeta.visual_delta_summary.v1",
        "summary_kind": "deterministic_bounded_rollup",
        "compacted_delta_count": len(records),
        "through_observation_index": last_index,
        "from_observation_index": max(0, int(first_index or 1) - 1),
        "status_counts": status_counts,
        **fields,
        "durable_history_query": (
            "Full visual_delta records remain in the session event store and may be "
            "queried with python_exec when this rollup is insufficient."
        ),
    }


def _parse_delta_payload(payload: JsonDict | str, *, max_items: int, max_chars: int) -> JsonDict:
    if isinstance(payload, str):
        parsed = json.loads(payload)
    else:
        parsed = payload
    if not isinstance(parsed, dict):
        raise ValueError("VDM response must be a JSON object")
    fields: JsonDict = {}
    for key in (
        "visible_changes",
        "task_progress_evidence",
        "completion_evidence",
        "uncertainties",
    ):
        value = parsed.get(key)
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise ValueError(f"VDM field {key} must be an array of strings")
        fields[key] = [item.strip()[:max_chars] for item in value[:max_items] if item.strip()]
    return fields


def _compact_usage(details: JsonDict) -> JsonDict:
    usage = details.get("usage")
    compact = dict(usage) if isinstance(usage, dict) else {}
    if "total_tokens" not in compact:
        prompt = compact.get("prompt_tokens")
        completion = compact.get("completion_tokens")
        if isinstance(prompt, int) and isinstance(completion, int):
            compact["total_tokens"] = max(0, prompt) + max(0, completion)
    for key in ("usage_source", "provider_attempts", "provider_role", "response_id"):
        if details.get(key) is not None:
            compact[key] = details[key]
    return compact


def _prompt_sha256() -> str:
    return sha256(VISUAL_DELTA_PROMPT.encode("utf-8")).hexdigest()


def _synthetic_current_observation(
    observation: EnvObservation,
    *,
    current_camera_artifacts: list[JsonDict],
) -> JsonDict:
    return {
        "observation_index": 0,
        "environment_step": observation.metadata.get("step_idx"),
        "visual_artifacts": [dict(item) for item in current_camera_artifacts],
    }


def _env_bool(source: Mapping[str, str], key: str, default: bool) -> bool:
    raw = source.get(key)
    if raw is None or not str(raw).strip():
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _env_positive_int(
    source: Mapping[str, str],
    key: str,
    default: int,
) -> int:
    try:
        value = int(str(source.get(key) or default))
    except ValueError:
        return default
    return max(1, value)
