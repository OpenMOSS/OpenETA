"""Session memory for lightweight embodied agents."""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from adapter.protocol import EnvAction, EnvObservation, JsonDict
from agent.runtime.conversation import (
    ConversationHistory,
    ConversationItem,
    checkpoint_record,
    item_record,
)
from agent.runtime.calibration_registry import (
    DEFAULT_GRASP_CALIBRATION_PROFILE,
    load_grasp_calibration_capabilities,
)


PENDING_SAM3_SELECTION_KEY = "pending_sam3_selection"
SELECTED_SAM3_DETECTION_KEY = "selected_sam3_detection"
PENDING_REFERENCE_LOCALIZATION_KEY = "pending_reference_localization"
REFERENCE_LOCALIZATION_FAILURE_KEY = "reference_localization_failure"
TARGET_LOCALIZATION_BUDGET_KEY = "target_localization_budget"
TARGET_ASSET_REFERENCE_KEY = "target_asset_reference"
SAM3_NO_DETECTION_KEY = "sam3_no_detection"
GRASP_CANDIDATE_POLICY_KEY = "grasp_candidate_policy"
LEGACY_ANYGRASP_CANDIDATE_POLICY_KEY = "anygrasp_candidate_policy"
GRASP_REESTIMATION_KEY = "grasp_reestimation"
GRASP_LIFT_PROBE_KEY = "grasp_lift_probe"
ARTICULATED_ATTACHMENT_PROBE_KEY = "articulated_attachment_probe"
GRASP_EXECUTION_KEY = "grasp_execution"
GRASP_RECOVERY_KEY = "grasp_recovery"
GRASP_ESTIMATION_RECOVERY_KEY = "grasp_estimation_recovery"
GRIPPER_COMMAND_STATE_KEY = "gripper_command_state"
ATTACHMENT_GATE_KEY = "attachment_gate"
PLACEMENT_RELEASE_KEY = "placement_release"
COMPLETED_PLACEMENT_SUBGOALS_KEY = "completed_placement_subgoals"
MOTION_RECONCILIATION_KEY = "motion_reconciliation"
SCENE_EPOCH_KEY = "scene_epoch"
TRANSITION_LEDGER_KEY = "transition_ledger"
ACTIVE_ENVIRONMENT_TASK_KEY = "active_environment_task"
GRASP_LIFT_PROBE_DISTANCE_M = 0.08
GRASP_FULL_LIFT_DISTANCE_M = 0.08
GRASP_RECOVERY_RETREAT_DISTANCE_M = 0.12
GRASP_CANDIDATE_MAX_ATTEMPTS = 3
ARTICULATED_HANDLE_APPROACH_MODES = ("top_down", "front", "side")
GRASP_REFERENCE_POSITION_TOLERANCE_M = 0.05
GRASP_REFERENCE_ORIENTATION_TOLERANCE_DEG = 20.0
TRANSITION_LEDGER_LIMIT = 32
GRASP_GEOMETRY_FAMILIES = {
    "upright_can",
    "upright_bottle",
    "boxed_item",
    "bowl",
    "apple",
    "articulated_handle",
    "drawer_handle",
    "other",
    "unknown",
}


class MemoryStore(Protocol):
    """Persistence boundary for session trace and working memory."""

    def start_session(
        self,
        *,
        session_id: str,
        task: str,
        metadata: JsonDict | None = None,
    ) -> None: ...

    def append_event(self, event: "MemoryEvent") -> None: ...

    def append_conversation_record(self, record: JsonDict) -> None: ...

    def load_conversation_records(self, session_id: str) -> list[JsonDict]: ...

    def load_working_memory(self) -> JsonDict: ...

    def save_working_memory(self, memory: "AgentMemory") -> None: ...

    def load_events(self, session_id: str, *, limit: int | None = None) -> list[JsonDict]: ...

    def load_session_metadata(self, session_id: str) -> JsonDict: ...


@dataclass(slots=True)
class MemoryEvent:
    """One durable-enough event in an agent session."""

    event_type: str
    payload: JsonDict
    timestamp_s: float = field(default_factory=time.time)


class AgentMemory:
    """Session log plus working memory.

    Without a store this remains an in-process object. With a store, session
    events are written to JSONL and working memory is loaded/saved as JSON.
    """

    def __init__(self, *, store: MemoryStore | None = None) -> None:
        self.store = store
        self.session_id: str | None = None
        self.task: str | None = None
        self.current_user_request: str = ""
        self.metadata: JsonDict = {}
        self.events: list[MemoryEvent] = []
        self.conversation = ConversationHistory()
        self.facts: dict[str, JsonDict] = {}
        self.artifacts: dict[str, JsonDict] = {}
        self.skill_notes: dict[str, list[JsonDict]] = {}
        self.compact_summary: str = ""

    def start_session(
        self,
        *,
        task: str,
        metadata: JsonDict | None = None,
        session_id: str | None = None,
    ) -> None:
        boot_facts = dict(self.facts) if self.session_id is None else {}
        boot_artifacts = dict(self.artifacts) if self.session_id is None else {}
        self.session_id = session_id or str(uuid4())
        self.task = task
        self.current_user_request = ""
        self.metadata = dict(metadata or {})
        self.events.clear()
        self.conversation.clear()
        self.facts = boot_facts
        self.facts.setdefault(
            SCENE_EPOCH_KEY,
            _memory_fact_entry({"epoch": 0}, source="runtime"),
        )
        self.artifacts = boot_artifacts
        self.skill_notes.clear()
        self.compact_summary = ""
        if self.store is not None:
            self.store.start_session(
                session_id=self.session_id,
                task=task,
                metadata=self.metadata,
            )
            self._save_working_memory()
        self.record(
            "session_start",
            {
                "session_id": self.session_id,
                "task": task,
                "metadata": self.metadata,
            },
        )
        self.begin_user_turn(task, source="session_start")

    def resume_session(
        self,
        session_id: str,
        *,
        task: str = "",
        metadata: JsonDict | None = None,
        max_events: int | None = 64,
    ) -> None:
        self.session_id = session_id
        stored_metadata: JsonDict = {}
        if self.store is not None:
            stored_metadata = self.store.load_session_metadata(session_id)
        self.task = task or str(stored_metadata.get("task") or "")
        self.metadata = {
            **(
                stored_metadata.get("metadata")
                if isinstance(stored_metadata.get("metadata"), dict)
                else {}
            ),
            **dict(metadata or {}),
        }
        self.events.clear()
        self.conversation.clear()
        self.facts.clear()
        self.artifacts.clear()
        self.skill_notes.clear()
        self.compact_summary = ""
        if self.store is not None:
            self.store.start_session(
                session_id=session_id,
                task=self.task or "(resumed)",
                metadata=self.metadata,
            )
            self._load_working_memory()
            for row in self.store.load_events(session_id, limit=max_events):
                event_type = str(row.get("event_type") or "event")
                payload = row.get("payload") if isinstance(row.get("payload"), dict) else {}
                timestamp = row.get("timestamp_s")
                try:
                    timestamp_s = float(timestamp)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    timestamp_s = time.time()
                self.events.append(
                    MemoryEvent(
                        event_type=event_type,
                        payload=dict(payload),
                        timestamp_s=timestamp_s,
                    )
                )
            records = self.store.load_conversation_records(session_id)
            if records:
                self.conversation.replay(records)
            else:
                self._reconstruct_legacy_conversation(
                    self.store.load_events(session_id, limit=None)
                )
                for item in self.conversation.items:
                    self._append_conversation_record(item_record(item))
        self.current_user_request = self.conversation.current_user_request or (self.task or "")
        self.record(
            "session_resumed",
            {
                "session_id": session_id,
                "task": self.task,
                "loaded_event_count": len(self.events),
            },
        )

    def begin_user_turn(self, text: str, *, source: str = "user") -> ConversationItem:
        """Record one exact user instruction as a durable conversation turn."""

        item = self.conversation.begin_user_turn(text, source=source)
        self.current_user_request = item.content
        self._append_conversation_record(item_record(item))
        self.record(
            "user_message",
            {
                "turn_id": item.turn_id,
                "source": source,
                "text": item.content,
            },
        )
        return item

    def record(self, event_type: str, payload: JsonDict | None = None) -> MemoryEvent:
        event = MemoryEvent(event_type=event_type, payload=dict(payload or {}))
        self.events.append(event)
        if self.store is not None:
            self.store.append_event(event)
        return event

    def add_observation(self, observation: EnvObservation) -> None:
        summary = summarize_observation(observation)
        summary["scene_epoch"] = self.scene_epoch()
        summary["runtime_camera_calibrations"] = [
            {
                "frame_id": camera.frame_id,
                "extrinsics": dict(camera.extrinsics),
            }
            for camera in observation.cameras
            if isinstance(camera.extrinsics, dict) and camera.extrinsics
        ]
        runtime_sources: list[JsonDict] = []
        seen_sources: set[tuple[str, str]] = set()
        for artifact in observation.metadata.get("image_artifacts", []):
            if (
                not isinstance(artifact, dict)
                or artifact.get("kind") != "rgb"
                or not isinstance(artifact.get("path"), str)
            ):
                continue
            frame_id = str(artifact.get("frame_id") or "")
            rgb_path = str(artifact["path"])
            key = (frame_id, rgb_path)
            if key not in seen_sources:
                seen_sources.add(key)
                runtime_sources.append({"frame_id": frame_id, "rgb_path": rgb_path})
        for artifact in self.artifacts.values():
            value = artifact.get("value") if isinstance(artifact, dict) else None
            if not isinstance(value, dict) or value.get("kind") != "rgbd_camera":
                continue
            frame_id = str(value.get("frame_id") or "")
            rgb_path = str(value.get("rgb_path") or "")
            key = (frame_id, rgb_path)
            if rgb_path and key not in seen_sources:
                seen_sources.add(key)
                runtime_sources.append({"frame_id": frame_id, "rgb_path": rgb_path})
        summary["runtime_camera_sources"] = runtime_sources
        self.record("observation", summary)
        self._capture_grasp_reestimation_observation()
        reconciliation_updated = self._reconcile_unknown_motion(observation)
        attachment_updated = self._capture_attachment_observation_verdict(observation)
        if (
            self._capture_grasp_lift_probe_obligation(observation)
            or reconciliation_updated
            or attachment_updated
        ):
            self._save_working_memory()

    def _capture_grasp_reestimation_observation(self) -> bool:
        reestimate = _memory_fact_value(self.facts.get(GRASP_REESTIMATION_KEY))
        if not isinstance(reestimate, dict) or reestimate.get("status") != "pending_observation":
            return False
        reestimate.update({"status": "ready", "observed_at_s": time.time()})
        self.facts[GRASP_REESTIMATION_KEY] = _memory_fact_entry(
            reestimate,
            source="candidate_reestimate_observation",
        )
        for key in (
            PENDING_SAM3_SELECTION_KEY,
            SELECTED_SAM3_DETECTION_KEY,
            GRASP_CANDIDATE_POLICY_KEY,
            LEGACY_ANYGRASP_CANDIDATE_POLICY_KEY,
            GRASP_LIFT_PROBE_KEY,
            ARTICULATED_ATTACHMENT_PROBE_KEY,
            GRASP_EXECUTION_KEY,
            ATTACHMENT_GATE_KEY,
        ):
            self.facts.pop(key, None)
        for key in (
            "anygrasp_grasp_candidates_latest",
            "grasp_pose_estimate_grasp_candidates_latest",
            "graspgenx_grasp_candidates_latest",
            "contact_graspnet_grasp_candidates_latest",
            "anyplace_placement_candidates_latest",
            "camera_pose_to_world_world_pose_latest",
        ):
            self.artifacts.pop(key, None)
        self.record("grasp_reestimate_ready", dict(reestimate))
        self._save_working_memory()
        return True

    def add_action(self, action: EnvAction) -> None:
        environment_task_updated = self._capture_active_environment_task(action)
        self._capture_reference_localization_state(action)
        self._capture_sam3_selection_state(action)
        target_mask_invalidated = self._invalidate_failed_anygrasp_target_mask(action)
        self._capture_anygrasp_candidate_policy(action)
        self._capture_compiled_grasp(action)
        self._capture_wrist_alignment(action)
        captured_artifacts = _extract_action_artifacts(action)
        for artifact in captured_artifacts:
            key = _artifact_memory_key(artifact, fallback_index=len(self.artifacts))
            self.artifacts[key] = {
                "value": artifact,
                "source": "tool_result",
                "timestamp_s": time.time(),
            }
        world_mutated = self._record_successful_world_mutation(action)
        gripper_state_updated = self._capture_gripper_command_state(action)
        placement_release_denied = self._capture_placement_release_denial(action)
        placement_release_updated = self._capture_placement_release(action)
        attachment_updated = self._capture_attachment_verdict(action)
        articulated_assessment_updated = self._capture_articulated_attachment_assessment(
            action
        )
        articulated_probe_prepared = self._capture_articulated_attachment_probe(action)
        articulated_probe_updated = self._capture_articulated_attachment_probe_result(action)
        lift_probe_updated = self._capture_grasp_lift_probe_result(action)
        execution_updated = self._advance_grasp_execution(action)
        reconciliation_updated = self._capture_motion_reconciliation(action)
        recovery_updated = self._advance_grasp_recovery(action)
        estimation_recovery_updated = self._advance_grasp_estimation_recovery(action)
        candidate_advanced = self._advance_anygrasp_candidate_after_rejection(action)
        candidate_accepted = (
            False if candidate_advanced else self._accept_anygrasp_candidate_after_motion(action)
        )
        if candidate_advanced:
            self.facts.pop(GRASP_LIFT_PROBE_KEY, None)
            self.facts.pop(ARTICULATED_ATTACHMENT_PROBE_KEY, None)
            self.facts.pop(GRASP_EXECUTION_KEY, None)
            self.facts.pop(ATTACHMENT_GATE_KEY, None)
            invalidated = [
                key
                for key in (
                    "anyplace_placement_candidates_latest",
                    "camera_pose_to_world_world_pose_latest",
                )
                if self.artifacts.pop(key, None) is not None
            ]
            if invalidated:
                self.record(
                    "placement_plan_invalidated",
                    {
                        "reason": "grasp_candidate_changed",
                        "artifact_keys": invalidated,
                    },
                )
        self._append_transition_ledger(action)
        if (
            captured_artifacts
            or world_mutated
            or placement_release_denied
            or placement_release_updated
            or lift_probe_updated
            or articulated_probe_prepared
            or articulated_probe_updated
            or execution_updated
            or attachment_updated
            or articulated_assessment_updated
            or reconciliation_updated
            or recovery_updated
            or estimation_recovery_updated
            or gripper_state_updated
            or candidate_advanced
            or candidate_accepted
            or target_mask_invalidated
            or environment_task_updated
        ):
            self._save_working_memory()
        self.record(
            "action",
            {
                "action_type": action.action_type,
                "command": action.command,
                "has_code": action.code is not None,
                "metadata": action.metadata,
                "captured_artifact_count": len(captured_artifacts),
            },
        )
        for conversation_item in self.conversation.add_action(action):
            self._append_conversation_record(item_record(conversation_item))

    def _invalidate_failed_anygrasp_target_mask(self, action: EnvAction) -> bool:
        call = _tool_call(action, "grasp_pose_estimate") or _tool_call(action, "anygrasp")
        if not isinstance(call, dict) or _call_result_success(call):
            return False
        outputs = _tool_call_outputs(call)
        reason = str(outputs.get("reason") or "")
        if reason in {"insufficient_object_points", "empty_point_cloud"}:
            reason = "no_grasp_candidates"
        if reason == "all_backends_failed":
            attempts = outputs.get("backend_attempts")
            failed_reasons = (
                {
                    str(attempt.get("reason") or "")
                    for attempt in attempts
                    if isinstance(attempt, dict) and attempt.get("status") == "failed"
                }
                if isinstance(attempts, list)
                else set()
            )
            if failed_reasons == {"no_grasp_candidates"}:
                reason = "no_grasp_candidates"
            elif failed_reasons and failed_reasons.issubset(
                {
                    "mcp_call_failed",
                    "model_inference_failed",
                    "model_load_failed",
                    "unknown_error",
                }
            ):
                reason = "model_inference_failed"
        if reason not in {
            "empty_target_mask",
            "model_inference_failed",
            "no_grasp_candidates",
        }:
            return False
        selected = self.selected_sam3_detection()
        if not isinstance(selected, dict):
            return False
        parameters = call.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {}
        if reason == "model_inference_failed":
            unified = str(call.get("name") or "") == "grasp_pose_estimate"
            failure_key = (
                "grasp_estimator_backend_failure" if unified else "anygrasp_backend_failure"
            )
            previous = selected.get(failure_key)
            previous_attempts = (
                int(previous.get("attempt_count") or 0) if isinstance(previous, dict) else 0
            )
            metadata = outputs.get("metadata")
            error_type = str(metadata.get("error_type") or "") if isinstance(metadata, dict) else ""
            attempt_count = previous_attempts + 1
            exhausted = attempt_count >= 2
            selected[failure_key] = {
                "reason": reason,
                "error_type": error_type or None,
                "attempt_count": attempt_count,
                "max_attempts": 2,
                "status": "exhausted" if exhausted else "retry_required",
            }
            self.facts[SELECTED_SAM3_DETECTION_KEY] = _memory_fact_entry(
                selected,
                source=failure_key,
            )
            self.record(
                "grasp_estimator_backend_retry_exhausted"
                if exhausted
                else "grasp_estimator_backend_retry_required",
                {
                    "result_id": selected.get("result_id"),
                    "detection_id": selected.get("id"),
                    "reason": reason,
                    "error_type": error_type or None,
                    "attempt_count": attempt_count,
                },
            )
            return True
        hints = parameters.get("hints")
        dense_sampling = hints.get("dense_sampling") if isinstance(hints, dict) else None
        if (
            reason == "no_grasp_candidates"
            and parameters.get("dense_grasp") is not True
            and dense_sampling is not True
        ):
            selected["dense_grasp_retry_required"] = True
            self.facts[SELECTED_SAM3_DETECTION_KEY] = _memory_fact_entry(
                selected,
                source="anygrasp_dense_retry",
            )
            self.record(
                "anygrasp_dense_retry_required",
                {
                    "result_id": selected.get("result_id"),
                    "detection_id": selected.get("id"),
                },
            )
            return True
        source_image = outputs.get("source_rgb") or selected.get("source_image")
        self.facts.pop(SELECTED_SAM3_DETECTION_KEY, None)
        self.facts.pop(PENDING_SAM3_SELECTION_KEY, None)
        self.facts[SAM3_NO_DETECTION_KEY] = _memory_fact_entry(
            {
                "result_id": selected.get("result_id"),
                "source_image": source_image,
                "target_prompt": selected.get("target_prompt"),
                "reason": reason,
                "segmentation_mode": selected.get("segmentation_mode"),
                "bbox_xyxy": selected.get("bbox_xyxy"),
            },
            source="anygrasp",
        )
        self.record(
            "sam3_detection_invalidated",
            {
                "result_id": selected.get("result_id"),
                "detection_id": selected.get("id"),
                "reason": reason,
            },
        )
        return True

    def add_external_event(self, event: JsonDict) -> None:
        event_type = str(event.get("type", "external"))
        self.record(event_type, event)
        if event_type == "human_answer":
            answer = event.get("answer")
            if isinstance(answer, str) and answer.strip():
                self.begin_user_turn(answer, source="human_answer")

    def save_fact(self, key: str, value: JsonDict, *, source: str = "") -> None:
        if key == LEGACY_ANYGRASP_CANDIDATE_POLICY_KEY:
            key = GRASP_CANDIDATE_POLICY_KEY
            self.facts.pop(LEGACY_ANYGRASP_CANDIDATE_POLICY_KEY, None)
        self.facts[key] = {"value": dict(value), "source": source, "timestamp_s": time.time()}
        self.record("memory_fact_saved", {"key": key, "source": source})
        self._save_working_memory()

    def save_artifact(self, key: str, value: JsonDict, *, source: str = "") -> None:
        self.artifacts[key] = {"value": dict(value), "source": source, "timestamp_s": time.time()}
        self.record("memory_artifact_saved", {"key": key, "source": source})
        self._save_working_memory()

    def save_skill_note(self, skill_name: str, note: JsonDict, *, source: str = "") -> None:
        entry = {"note": dict(note), "source": source, "timestamp_s": time.time()}
        self.skill_notes.setdefault(skill_name, []).append(entry)
        self.record("memory_skill_note_saved", {"skill": skill_name, "source": source})
        self._save_working_memory()

    def get_memory(
        self,
        key: str | None = None,
        *,
        namespace: str = "all",
    ) -> JsonDict:
        if namespace == "facts":
            return {"facts": _select_memory(self.facts, key)}
        if namespace == "artifacts":
            return {"artifacts": _select_memory(self.artifacts, key)}
        if namespace == "skill_notes":
            if key is None:
                return {"skill_notes": self.skill_notes}
            return {"skill_notes": {key: self.skill_notes.get(key, [])}}
        return {
            "facts": _select_memory(self.facts, key),
            "artifacts": _select_memory(self.artifacts, key),
            "skill_notes": (
                self.skill_notes if key is None else {key: self.skill_notes.get(key, [])}
            ),
            "compact_summary": self.compact_summary,
        }

    def pending_sam3_selection(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(PENDING_SAM3_SELECTION_KEY))

    def selected_sam3_detection(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(SELECTED_SAM3_DETECTION_KEY))

    def pending_reference_localization(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(PENDING_REFERENCE_LOCALIZATION_KEY))

    def target_asset_reference(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(TARGET_ASSET_REFERENCE_KEY))

    def reference_localization_failure(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(REFERENCE_LOCALIZATION_FAILURE_KEY))

    def sam3_no_detection(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(SAM3_NO_DETECTION_KEY))

    def grasp_candidate_policy(self) -> JsonDict | None:
        return _memory_fact_value(
            self.facts.get(GRASP_CANDIDATE_POLICY_KEY)
            or self.facts.get(LEGACY_ANYGRASP_CANDIDATE_POLICY_KEY)
        )

    def grasp_reestimation(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(GRASP_REESTIMATION_KEY))

    def anygrasp_candidate_policy(self) -> JsonDict | None:
        """Backward-compatible accessor for pre-facade callers."""

        return self.grasp_candidate_policy()

    def retained_targeted_grasp(self) -> JsonDict | None:
        """Return the active targeted grasp and its exact aligned source packet."""

        policy = self.grasp_candidate_policy() or {}
        final_source = policy.get("final_fallback_source")
        final_candidate = policy.get("active_candidate")
        if (
            policy.get("final_refinable_fallback") is True
            and isinstance(final_source, dict)
            and isinstance(final_candidate, dict)
        ):
            source = final_source.get("source")
            source = dict(source) if isinstance(source, dict) else {}
            for source_key, policy_key in (
                ("rgb", "source_rgb"),
                ("depth", "source_depth"),
            ):
                if not isinstance(source.get(source_key), str) or not source[source_key]:
                    value = final_source.get(policy_key)
                    if isinstance(value, str) and value:
                        source[source_key] = value
            frame_id = final_source.get("camera_frame_id")
            if isinstance(frame_id, str) and frame_id:
                source["camera_frame_id"] = frame_id
            return {
                "artifact_key": "grasp_estimation_recovery.final_candidate",
                "result_id": policy.get("result_id"),
                "status": policy.get("status", "retained"),
                "candidate": dict(final_candidate),
                "source": source,
            }

        artifact_key = next(
            (
                key
                for key in (
                    "grasp_pose_estimate_grasp_candidates_latest",
                    "anygrasp_grasp_candidates_latest",
                    "graspgenx_grasp_candidates_latest",
                    "contact_graspnet_grasp_candidates_latest",
                )
                if key in self.artifacts
            ),
            "",
        )
        entry = self.artifacts.get(artifact_key)
        if not isinstance(entry, dict):
            return None
        artifact = entry.get("value")
        if not isinstance(artifact, dict):
            return None
        source_value = artifact.get("selected_grasp_source")
        source = dict(source_value) if isinstance(source_value, dict) else {}
        if str(source.get("mode") or "") != "targeted":
            return None

        for source_key, artifact_key in (
            ("rgb", "source_rgb"),
            ("depth", "source_depth"),
            ("object_mask", "target_mask"),
        ):
            if not isinstance(source.get(source_key), str) or not source[source_key]:
                value = artifact.get(artifact_key)
                if isinstance(value, str) and value:
                    source[source_key] = value

        candidate = policy.get("active_candidate")
        if not isinstance(candidate, dict):
            candidate = artifact.get("best_grasp_candidate")
        if not isinstance(candidate, dict):
            return None

        return {
            "artifact_key": artifact_key,
            "result_id": policy.get("result_id"),
            "status": policy.get("status", "retained"),
            "candidate": dict(candidate),
            "source": source,
        }

    def activate_final_grasp_candidate(self, *, recovery_id: str) -> JsonDict:
        recovery = self.grasp_estimation_recovery()
        policy = self.grasp_candidate_policy()
        if (
            not isinstance(recovery, dict)
            or recovery.get("status") != "required"
            or str(recovery.get("recovery_id") or "") != recovery_id
        ):
            raise ValueError("final grasp fallback requires the active recovery_id.")
        if str(recovery.get("trigger_class") or "") not in {
            "perception_refinable",
            "uncertain_review",
        }:
            raise ValueError("final grasp fallback is not allowed for a hard rejection.")
        if not isinstance(policy, dict) or policy.get("status") != "exhausted":
            raise ValueError("final grasp fallback requires an exhausted candidate policy.")
        entries = [
            dict(item)
            for item in recovery.get("fallback_candidates", [])
            if isinstance(item, dict) and isinstance(item.get("candidate"), dict)
        ]
        if not entries:
            raise ValueError("final grasp fallback has no refinable candidate.")
        selected = min(
            entries,
            key=lambda item: _grasp_candidate_sort_key(dict(item["candidate"])),
        )
        candidate = dict(selected["candidate"])
        original_id = str(candidate.get("id") or "grasp")
        candidate["original_candidate_id"] = original_id
        candidate["id"] = f"{original_id}-final-{recovery_id[-8:]}"
        candidate["final_refinable_fallback"] = True
        capabilities = _fallback_candidate_capabilities(
            selected,
            default=self._active_grasp_calibration_capabilities(),
        )
        max_gripper_width = float(capabilities["max_gripper_width_m"])
        try:
            estimated_width = float(candidate.get("width"))
        except (TypeError, ValueError):
            estimated_width = max_gripper_width
        if not math.isfinite(estimated_width):
            estimated_width = max_gripper_width
        candidate["estimated_width_m"] = estimated_width
        candidate["width"] = min(max(0.0, estimated_width), max_gripper_width)
        candidate["max_gripper_width_m"] = max_gripper_width
        candidate["grasp_calibration_id"] = capabilities["calibration_id"]
        candidate["width_clamped_to_physical_limit"] = (
            estimated_width > max_gripper_width
        )
        policy.update(
            {
                "result_id": selected.get("source_result_id") or policy.get("result_id"),
                "source_tool": selected.get("source_tool") or policy.get("source_tool"),
                "source_backend": (
                    selected.get("source_backend") or policy.get("source_backend")
                ),
                "status": "active",
                "candidate_count": 1,
                "active_rank": 0,
                "active_candidate": candidate,
                "remaining_candidate_ids": [],
                "candidates": [candidate],
                "source_rgb": selected.get("source_rgb"),
                "source_depth": selected.get("source_depth"),
                "camera_frame_id": selected.get("camera_frame_id"),
                "target_detection": selected.get("target_detection"),
                "final_refinable_fallback": True,
                "final_fallback_source": selected,
                "final_fallback_recovery_id": recovery_id,
                "physical_width_limit_m": max_gripper_width,
                "grasp_calibration_id": capabilities["calibration_id"],
                "grasp_calibration_profile_path": capabilities["profile_path"],
                "activated_at_s": time.time(),
            }
        )
        for key in ("fallback_required", "exhaustion_reason", "refinement_seed_candidate"):
            policy.pop(key, None)
        recovery.update(
            {
                "status": "final_candidate_activated",
                "final_candidate_id": candidate["id"],
                "final_source_result_id": selected.get("source_result_id"),
                "completed_at_s": time.time(),
            }
        )
        self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
            policy,
            source="final_refinable_grasp_fallback",
        )
        self.facts[GRASP_ESTIMATION_RECOVERY_KEY] = _memory_fact_entry(
            recovery,
            source="final_refinable_grasp_fallback",
        )
        self.facts.pop(GRASP_EXECUTION_KEY, None)
        self.facts.pop(GRASP_LIFT_PROBE_KEY, None)
        self.facts.pop(ARTICULATED_ATTACHMENT_PROBE_KEY, None)
        self.facts.pop(ATTACHMENT_GATE_KEY, None)
        self.record(
            "final_refinable_grasp_candidate_activated",
            {
                "recovery_id": recovery_id,
                "candidate_id": candidate["id"],
                "original_candidate_id": original_id,
                "score": candidate.get("score"),
                "source_backend": policy.get("source_backend"),
                "width_clamped": candidate["width_clamped_to_physical_limit"],
            },
        )
        self._save_working_memory()
        return {
            "recovery_id": recovery_id,
            "candidate": candidate,
            "source_backend": policy.get("source_backend"),
            "source_result_id": policy.get("result_id"),
            "max_gripper_width_m": max_gripper_width,
            "grasp_calibration_id": capabilities["calibration_id"],
        }

    def _active_grasp_calibration_capabilities(self) -> JsonDict:
        workspace = self.metadata.get("workspace")
        workspace = workspace if isinstance(workspace, dict) else {}
        profile_path = (
            workspace.get("grasp_profile_path")
            or self.metadata.get("grasp_profile_path")
            or self.metadata.get("calibration_profile_path")
            or DEFAULT_GRASP_CALIBRATION_PROFILE
        )
        return load_grasp_calibration_capabilities(str(profile_path))

    def grasp_lift_probe(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(GRASP_LIFT_PROBE_KEY))

    def articulated_attachment_probe(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(ARTICULATED_ATTACHMENT_PROBE_KEY))

    def grasp_execution(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(GRASP_EXECUTION_KEY))

    def grasp_recovery(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(GRASP_RECOVERY_KEY))

    def grasp_estimation_recovery(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(GRASP_ESTIMATION_RECOVERY_KEY))

    def gripper_command_state(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(GRIPPER_COMMAND_STATE_KEY))

    def attachment_gate(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(ATTACHMENT_GATE_KEY))

    def placement_release(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(PLACEMENT_RELEASE_KEY))

    def motion_reconciliation(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(MOTION_RECONCILIATION_KEY))

    def scene_epoch(self) -> int:
        value = _memory_fact_value(self.facts.get(SCENE_EPOCH_KEY)) or {}
        try:
            return max(0, int(value.get("epoch") or 0))
        except (TypeError, ValueError):
            return 0

    def transition_ledger(self) -> list[JsonDict]:
        value = _memory_fact_value(self.facts.get(TRANSITION_LEDGER_KEY)) or {}
        rows = value.get("rows")
        return (
            [dict(row) for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
        )

    def latest_environment_receipt(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get("latest_environment_receipt"))

    def active_environment_task(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(ACTIVE_ENVIRONMENT_TASK_KEY))

    def _capture_active_environment_task(self, action: EnvAction) -> bool:
        close_call = _successful_tool_call(action, "close_simulator_env")
        if close_call is not None:
            removed = self.facts.pop(ACTIVE_ENVIRONMENT_TASK_KEY, None)
            gripper_removed = self.facts.pop(GRIPPER_COMMAND_STATE_KEY, None)
            self.record(
                "active_environment_task_cleared",
                {"reason": "simulator_environment_closed"},
            )
            return removed is not None or gripper_removed is not None

        gripper_removed = False
        call = _successful_tool_call(action, "create_simulator_env")
        if call is not None:
            gripper_removed = self.facts.pop(GRIPPER_COMMAND_STATE_KEY, None) is not None
        if call is None:
            call = _successful_tool_call(action, "observe")
        if call is None:
            return False

        extracted = _assigned_task_from_tool_call(call)
        if extracted is None:
            if str(call.get("name") or "") != "create_simulator_env":
                return False
            removed = self.facts.pop(ACTIVE_ENVIRONMENT_TASK_KEY, None)
            self.record(
                "active_environment_task_missing",
                {"source_tool": "create_simulator_env"},
            )
            return removed is not None or gripper_removed

        task, source_field = extracted
        outputs = _tool_call_outputs(call)
        environment = outputs.get("environment")
        environment = environment if isinstance(environment, dict) else {}
        mcp = outputs.get("mcp")
        mcp = mcp if isinstance(mcp, dict) else {}
        previous = self.active_environment_task() or {}
        if str(call.get("name") or "") == "observe" and previous.get("task") == task:
            return False
        value = {
            "task": task,
            "env_id": environment.get("env_id") or previous.get("env_id"),
            "handle": (environment.get("handle") or mcp.get("handle") or previous.get("handle")),
            "session_id": (
                environment.get("session_id") or mcp.get("session_id") or previous.get("session_id")
            ),
            "source_tool": str(call.get("name") or ""),
            "source_field": source_field,
            "scene_epoch": self.scene_epoch(),
            "updated_at_s": time.time(),
        }
        value = {key: item for key, item in value.items() if item not in (None, "")}
        self.facts[ACTIVE_ENVIRONMENT_TASK_KEY] = _memory_fact_entry(
            value,
            source="simulator_tool_result",
        )
        if previous != value:
            self.record("active_environment_task_updated", value)
        return previous != value or gripper_removed

    def grasp_candidate_gate_error(
        self,
        *,
        tool_name: str,
        parameters: JsonDict,
    ) -> str | None:
        policy = self.grasp_candidate_policy()
        if policy is None:
            return None
        if tool_name not in {
            "compile_grasp_seed",
            "camera_pose_to_world",
            "move_to",
            "follow_eef_trajectory",
            "ik_preview_check",
            "obstacle_avoidance",
        }:
            return None
        recovery = self.grasp_recovery()
        recovery_action = recovery.get("required_action") if isinstance(recovery, dict) else None
        if (
            isinstance(recovery, dict)
            and recovery.get("status") == "required"
            and isinstance(recovery_action, dict)
            and tool_name == recovery_action.get("name")
            and parameters == recovery_action.get("parameters")
        ):
            return None
        estimation_recovery = self.grasp_estimation_recovery()
        if _is_grasp_estimation_recovery_action(
            tool_name,
            parameters,
            recovery=estimation_recovery,
        ):
            return None
        status = str(policy.get("status") or "")
        if status == "accepted":
            return None
        source_tool = str(policy.get("source_tool") or "grasp_pose_estimate")
        source_backend = str(policy.get("source_backend") or source_tool)
        source_label = _grasp_backend_label(source_backend)
        active = policy.get("active_candidate")
        if status == "exhausted" or not isinstance(active, dict):
            return (
                f"All {source_label} candidates were rejected. Observe and rerun "
                f"{source_label} "
                "instead of reusing an exhausted pose."
            )
        active_id = str(active.get("id") or "")
        supplied_id = _parameters_grasp_candidate_id(parameters)
        if tool_name == "camera_pose_to_world":
            if _is_anyplace_pose(parameters):
                return None
            if source_tool in {"anygrasp", "grasp_pose_estimate"}:
                return (
                    "AnyGrasp candidates require compile_grasp_seed; "
                    "camera_pose_to_world does not apply the GraspNet-to-EEF "
                    "calibration."
                )
            if not supplied_id:
                return (
                    f"camera_pose_to_world must receive the active {source_label} "
                    f"candidate {active_id!r}, including its id."
                )
            if supplied_id != active_id:
                return (
                    f"Greedy {source_label} policy requires the current active "
                    f"candidate {active_id!r}; later candidates are available only "
                    "after a candidate-linked safety or motion rejection."
                )
            return None
        if tool_name == "compile_grasp_seed":
            if source_tool not in {"anygrasp", "grasp_pose_estimate"}:
                return (
                    f"{source_label} candidates use camera_pose_to_world rather than "
                    "the AnyGrasp-specific compile_grasp_seed calibration."
                )
            if supplied_id != active_id:
                return (
                    "compile_grasp_seed must receive the complete active AnyGrasp "
                    f"candidate {active_id!r}."
                )
            try:
                supplied_epoch = int(parameters.get("scene_epoch"))
            except (TypeError, ValueError):
                supplied_epoch = -1
            if supplied_epoch != self.scene_epoch():
                return (
                    "compile_grasp_seed must use the current host scene_epoch "
                    f"{self.scene_epoch()}."
                )
            return None
        if (
            source_tool in {"anygrasp", "grasp_pose_estimate"}
            and tool_name in {"move_to", "follow_eef_trajectory"}
            and self.grasp_execution() is None
        ):
            return (
                "Raw AnyGrasp motion is blocked. Compile the active candidate with "
                "compile_grasp_seed and follow the host-generated grasp_execution stages."
            )
        if supplied_id and supplied_id != active_id:
            return (
                f"Tool {tool_name!r} references {source_label} candidate {supplied_id!r}, "
                f"but the current active candidate is {active_id!r}."
            )
        return None

    def grasp_execution_gate_error(
        self,
        *,
        tool_name: str,
        parameters: JsonDict,
    ) -> str | None:
        reconciliation = self.motion_reconciliation()
        if isinstance(reconciliation, dict) and reconciliation.get("status") in {
            "required",
            "unresolved",
        }:
            if tool_name == "observe":
                return None
            return (
                "A simulator action has transport-unknown outcome. Observe the same "
                "environment handle before issuing another action. Do not resend a "
                "partial move because the original controller may still be running."
            )
        execution = self.grasp_execution()
        if not isinstance(execution, dict) or execution.get("status") != "required":
            return None
        stage = str(execution.get("stage") or "")
        if tool_name == "observe":
            return None
        if stage == "align":
            if tool_name in {
                "sam3",
                "select_sam3_detection",
                "retrieve_asset_reference",
                "molmopoint",
                "compute_wrist_alignment",
            }:
                return None
            return (
                "The grasp is at safe hover and requires fresh wrist alignment. "
                "Observe/segment the wrist view, then call compute_wrist_alignment."
            )
        if stage == "prepare_probe":
            if tool_name == "prepare_attachment_probe":
                return None
            return (
                "The closed articulated handle requires prepare_attachment_probe. "
                "Propose one bounded linear direction or short arc from the current "
                "multi-view observation before any further world mutation."
            )
        if stage == "attachment":
            if execution.get("attachment_mode") == "articulated_handle":
                gate = self.attachment_gate() or {}
                verdict = str(gate.get("verdict") or "UNKNOWN").upper()
                assessment_count = int(gate.get("assessment_count") or 0)
                refresh_required = gate.get("refresh_required") is True
                refresh_completed = gate.get("unknown_refresh_completed") is True
                if verdict == "UNKNOWN":
                    if tool_name == "assess_attachment_probe" and (
                        assessment_count == 0
                        or (assessment_count == 1 and refresh_completed)
                    ):
                        return None
                    if (
                        tool_name == "observe"
                        and assessment_count == 1
                        and refresh_required
                        and not refresh_completed
                    ):
                        return None
            actions = execution.get("attachment_actions")
            allowed = [
                action
                for action in (actions.values() if isinstance(actions, dict) else [])
                if isinstance(action, dict)
            ]
            if any(
                tool_name == action.get("name") and parameters == action.get("parameters")
                for action in allowed
            ):
                return None
            return (
                "Attachment evidence is pending. Keep the gripper closed and use only "
                "the host-owned action for the current verdict. Articulated UNKNOWN "
                "permits one fresh observation; portable-object PASS uses full lift, "
                "and structured FAIL permits the exact recovery open."
            )
        if stage == "probe":
            articulated_probe = self.articulated_attachment_probe()
            articulated_required = (
                articulated_probe.get("required_action")
                if isinstance(articulated_probe, dict)
                else None
            )
            if (
                isinstance(articulated_required, dict)
                and articulated_probe.get("status") == "required"
                and tool_name == articulated_required.get("name")
                and parameters == articulated_required.get("parameters")
            ):
                return None
            probe = self.grasp_lift_probe()
            required_parameters = (
                probe.get("required_parameters") if isinstance(probe, dict) else None
            )
            if (
                isinstance(required_parameters, dict)
                and probe.get("status") == "required"
                and tool_name == "move_to"
                and parameters == required_parameters
            ):
                return None
            return (
                "The attachment probe requires the exact host-generated motion "
                "parameters while keeping the gripper closed."
            )
        required = execution.get("required_action")
        if not isinstance(required, dict):
            return "The host-generated grasp execution obligation is malformed."
        return grasp_reference_action_error(
            stage=stage,
            tool_name=tool_name,
            parameters=parameters,
            required_action=required,
        )

    def resolve_sam3_selection(
        self,
        *,
        result_id: str,
        detection_id: str,
        selection_source: str,
        confidence: float | None = None,
        reason: str = "",
        target_geometry_family: str = "",
    ) -> JsonDict:
        pending = self.pending_sam3_selection()
        if pending is None:
            raise ValueError("No SAM3 detection selection is pending.")
        expected_result_id = str(pending.get("result_id") or "")
        if not result_id or result_id != expected_result_id:
            raise ValueError("select_sam3_detection requires the exact pending sam3_result_id.")
        candidates = pending.get("candidates")
        if not isinstance(candidates, list):
            candidates = []
        selected = next(
            (
                dict(candidate)
                for candidate in candidates
                if isinstance(candidate, dict) and str(candidate.get("id") or "") == detection_id
            ),
            None,
        )
        if selected is None:
            raise ValueError("detection_id does not belong to the pending SAM3 result.")
        geometry_family = target_geometry_family.strip().lower()
        if geometry_family and geometry_family not in GRASP_GEOMETRY_FAMILIES:
            raise ValueError(
                "target_geometry_family must be one of "
                + ", ".join(sorted(GRASP_GEOMETRY_FAMILIES))
                + "."
            )
        selected.update(
            {
                "result_id": result_id,
                "source_image": pending.get("source_image"),
                "target_prompt": pending.get("target_prompt"),
                "segmentation_mode": pending.get("segmentation_mode"),
                "selection_source": selection_source or "main_agent_vlm",
                "selection_confidence": confidence,
                "selection_reason": reason,
                "selected_at_s": time.time(),
                "scene_epoch": _optional_int(
                    pending.get("scene_epoch"),
                    default=self.scene_epoch(),
                ),
            }
        )
        if geometry_family and geometry_family != "unknown":
            selected["target_geometry_family"] = geometry_family
        self.facts.pop(PENDING_SAM3_SELECTION_KEY, None)
        self.facts[SELECTED_SAM3_DETECTION_KEY] = _memory_fact_entry(
            selected,
            source="select_sam3_detection",
        )
        self.facts.pop(TARGET_LOCALIZATION_BUDGET_KEY, None)
        self.record(
            "sam3_detection_selected",
            {
                "result_id": result_id,
                "detection_id": detection_id,
                "selection_source": selected["selection_source"],
                "selection_confidence": confidence,
            },
        )
        self._save_working_memory()
        return selected

    def reject_sam3_detections(self, *, result_id: str, reason: str) -> JsonDict:
        pending = self.pending_sam3_selection()
        if pending is None:
            raise ValueError("No SAM3 detection selection is pending.")
        expected_result_id = str(pending.get("result_id") or "")
        if not result_id or result_id != expected_result_id:
            raise ValueError("reject_sam3_detections requires the pending sam3_result_id.")
        rejection_reason = reason.strip()
        if not rejection_reason:
            raise ValueError("reject_sam3_detections requires a visual reason.")
        candidates = pending.get("candidates")
        rejected_ids = [
            str(candidate.get("id") or "")
            for candidate in (candidates if isinstance(candidates, list) else [])
            if isinstance(candidate, dict) and str(candidate.get("id") or "")
        ]
        no_detection = {
            "result_id": result_id,
            "source_image": pending.get("source_image"),
            "target_prompt": pending.get("target_prompt"),
            "reason": "semantic_candidates_rejected",
            "segmentation_mode": pending.get("segmentation_mode"),
            "rejected_detection_ids": rejected_ids,
            "rejection_reason": rejection_reason,
        }
        self.facts.pop(PENDING_SAM3_SELECTION_KEY, None)
        self.facts.pop(SELECTED_SAM3_DETECTION_KEY, None)
        self.facts[SAM3_NO_DETECTION_KEY] = _memory_fact_entry(
            no_detection,
            source="reject_sam3_detections",
        )
        self.record("sam3_detections_rejected", dict(no_detection))
        self._save_working_memory()
        return no_detection

    def detection_selection_gate_error(
        self,
        *,
        tool_name: str,
        parameters: JsonDict,
        world_mutating: bool = False,
    ) -> str | None:
        no_detection = self.sam3_no_detection()
        reference_failure = _memory_fact_value(
            self.facts.get(REFERENCE_LOCALIZATION_FAILURE_KEY)
        )
        if (
            tool_name == "retrieve_asset_reference"
            and isinstance(no_detection, dict)
            and isinstance(reference_failure, dict)
            and str(reference_failure.get("sam3_result_id") or "")
            == str(no_detection.get("result_id") or "")
        ):
            return (
                "Object-memory localization already failed for the current empty SAM3 "
                "result. Use molmopoint on sam3_no_detection.source_image, then feed its "
                "exact pixel point to SAM3; do not retry retrieve_asset_reference until "
                "a new SAM3 result or changed scene exists."
            )
        localization = self.pending_reference_localization()
        if localization is not None:
            required_parameter = str(localization.get("required_parameter") or "roi_bbox_xyxy")
            if tool_name != "sam3":
                return (
                    "Reference-guided target localization is pending. The next "
                    f"perception action must call SAM3 with {required_parameter}."
                )
            expected_image = str(localization.get("scene_image") or "")
            if str(parameters.get("image") or "") != expected_image:
                return (
                    "Reference-guided SAM3 must use the exact scene image from the "
                    "pending asset-reference result."
                )
            if required_parameter == "positive_points":
                expected_points = localization.get("positive_points")
                if parameters.get("positive_points") != expected_points:
                    return (
                        "Reference-guided SAM3 requires the exact positive_points "
                        "returned by the isolated reference localizer."
                    )
            else:
                bbox = parameters.get("roi_bbox_xyxy")
                if not isinstance(bbox, list) or len(bbox) != 4:
                    return (
                        "Reference-guided SAM3 requires roi_bbox_xyxy in original-image "
                        "pixel coordinates."
                    )
        pending = self.pending_sam3_selection()
        mode = str(parameters.get("mode") or "targeted").strip().lower()
        if pending is not None and tool_name == "anygrasp" and mode != "scene":
            return (
                "Targeted AnyGrasp is blocked until the pending SAM3 candidates are "
                "resolved with select_sam3_detection."
            )
        if pending is not None and tool_name == "graspgenx":
            return (
                "GraspGenX is blocked until the pending SAM3 candidates are resolved "
                "with select_sam3_detection."
            )
        if pending is not None and world_mutating:
            return (
                "World-mutating tools are blocked while a SAM3 detection selection "
                "obligation is pending."
            )
        if tool_name not in {"anygrasp", "graspgenx"}:
            return None
        if tool_name == "anygrasp" and mode == "scene":
            return None
        selected = self.selected_sam3_detection()
        if selected is None:
            return None
        expected_mask = str(selected.get("mask_ref") or "")
        if tool_name == "graspgenx":
            object_mask = parameters.get("object_mask")
            supplied_mask = (
                str(object_mask.get("mask_ref") or "") if isinstance(object_mask, dict) else ""
            )
        else:
            supplied_mask = str(parameters.get("target_mask") or "")
        if expected_mask and supplied_mask != expected_mask:
            if tool_name == "anygrasp":
                return (
                    "Targeted AnyGrasp must use the mask_ref from the recorded "
                    "select_sam3_detection result."
                )
            return "GraspGenX must use the mask_ref from the recorded select_sam3_detection result."
        return None

    def _capture_sam3_selection_state(self, action: EnvAction) -> None:
        command = action.command if isinstance(action.command, dict) else {}
        for call in command.get("tool_calls", []) or []:
            if not isinstance(call, dict) or str(call.get("name") or "") != "sam3":
                continue
            result = call.get("result")
            if not isinstance(result, dict) or not bool(result.get("success")):
                continue
            details = result.get("details")
            if not isinstance(details, dict):
                continue
            outputs = details.get("outputs")
            if not isinstance(outputs, dict):
                outputs = details
            detections = outputs.get("detections")
            if not isinstance(detections, list):
                continue
            self.facts.pop(GRASP_REESTIMATION_KEY, None)
            candidates = [
                dict(candidate) for candidate in detections if isinstance(candidate, dict)
            ]
            result_id = str(outputs.get("result_id") or "")
            if not result_id:
                result_id = f"sam3-{int(time.time() * 1000)}"
            selection_bundle = outputs.get("selection_bundle")
            if not isinstance(selection_bundle, dict):
                selection_bundle = {}
            parameters = details.get("parameters")
            if not isinstance(parameters, dict):
                parameters = {}
            source_camera_role = str(
                outputs.get("source_camera_role")
                or details.get("source_camera_role")
                or ""
            )
            source_frame_id = (
                outputs.get("frame_id")
                or outputs.get("source_frame_id")
                or (
                    details.get("source_frame_id")
                    if source_camera_role
                    else None
                )
                or parameters.get("frame_id")
            )
            base = {
                "result_id": result_id,
                "target_prompt": outputs.get("prompt") or parameters.get("prompt"),
                "source_image": outputs.get("source_image") or parameters.get("image"),
                "frame_id": source_frame_id,
                "ranking": outputs.get("ranking") or "score_descending",
                "candidate_count": len(candidates),
                "candidates": candidates,
                "selection_bundle": dict(selection_bundle),
                "segmentation_mode": outputs.get("segmentation_mode"),
                "scene_epoch": self.scene_epoch(),
            }
            if source_camera_role:
                base["camera_role"] = source_camera_role
            self.facts.pop(REFERENCE_LOCALIZATION_FAILURE_KEY, None)
            asset_reference = self.target_asset_reference()
            if isinstance(asset_reference, dict):
                verification = asset_reference.get("exact_instance_verification")
                if (
                    isinstance(verification, dict)
                    and str(verification.get("decision") or "").lower() == "match"
                    and str(parameters.get("image") or "")
                    == str(asset_reference.get("scene_image") or "")
                    and parameters.get("positive_points") == asset_reference.get("positive_points")
                ):
                    base["reference_verification"] = dict(verification)
            self.facts.pop(PENDING_SAM3_SELECTION_KEY, None)
            self.facts.pop(SELECTED_SAM3_DETECTION_KEY, None)
            if candidates:
                self.facts.pop(SAM3_NO_DETECTION_KEY, None)
                self.facts[PENDING_SAM3_SELECTION_KEY] = _memory_fact_entry(
                    base,
                    source="sam3",
                )
                self.record(
                    "sam3_detection_selection_required",
                    {
                        "result_id": result_id,
                        "candidate_count": len(candidates),
                        "verification_scope": (
                            "single_detection" if len(candidates) == 1 else "multiple_detections"
                        ),
                    },
                )
            else:
                self.facts[SAM3_NO_DETECTION_KEY] = _memory_fact_entry(
                    base,
                    source="sam3",
                )
                self._capture_grasp_fallback_segmentation_failure(base)
                self.record("sam3_no_detection", {"result_id": result_id})
            self._save_working_memory()

    def _capture_grasp_fallback_segmentation_failure(self, sam3_result: JsonDict) -> None:
        policy = self.grasp_candidate_policy()
        if not isinstance(policy, dict) or policy.get("fallback_required") is not True:
            return
        target_prompt = str(sam3_result.get("target_prompt") or "").strip()
        fallback_prompt = str(policy.get("fallback_target_prompt") or "").strip()
        if target_prompt and fallback_prompt and target_prompt != fallback_prompt:
            return
        source_image = str(sam3_result.get("source_image") or "")
        if not source_image:
            return
        attempts = _grasp_fallback_attempts(policy)
        max_gripper_width = _policy_max_gripper_width(
            policy,
            default=float(
                self._active_grasp_calibration_capabilities()["max_gripper_width_m"]
            ),
        )
        _append_grasp_fallback_attempt(
            attempts,
            {
                "backend": str(policy.get("source_backend") or "anygrasp"),
                "camera_frame_id": _camera_frame_for_source_image(
                    policy,
                    source_image,
                ),
                "source_rgb": source_image,
                "outcome": "segmentation_no_detection",
                "raw_candidate_count": 0,
                "width_limit_m": max_gripper_width,
            },
        )
        policy["fallback_attempts"] = attempts
        self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
            policy,
            source="grasp_width_fallback",
        )
        self.record(
            "grasp_estimation_fallback_camera_exhausted",
            {
                "backend": str(policy.get("source_backend") or "anygrasp"),
                "source_rgb": source_image,
                "reason": "segmentation_no_detection",
            },
        )

    def _capture_reference_localization_state(self, action: EnvAction) -> None:
        command = action.command if isinstance(action.command, dict) else {}
        for call in command.get("tool_calls", []) or []:
            if not isinstance(call, dict):
                continue
            name = str(call.get("name") or "")
            result = call.get("result")
            if (
                name == "retrieve_asset_reference"
                and isinstance(result, dict)
                and not bool(result.get("success"))
            ):
                no_detection = self.sam3_no_detection()
                if isinstance(no_detection, dict):
                    parameters = call.get("parameters")
                    if not isinstance(parameters, dict):
                        request = command.get("request")
                        parameters = (
                            request.get("parameters") if isinstance(request, dict) else None
                        )
                    parameters = parameters if isinstance(parameters, dict) else {}
                    target_object = (
                        parameters.get("target_object")
                        or no_detection.get("target_prompt")
                    )
                    scene_image = (
                        parameters.get("scene_image")
                        or no_detection.get("source_image")
                    )
                    budget = _memory_fact_value(
                        self.facts.get(TARGET_LOCALIZATION_BUDGET_KEY)
                    )
                    current_epoch = self.scene_epoch()
                    same_grounding_request = (
                        isinstance(budget, dict)
                        and str(budget.get("target_object") or "").strip().lower()
                        == str(target_object or "").strip().lower()
                        and budget.get("scene_epoch") == current_epoch
                    )
                    failure = {
                        "sam3_result_id": no_detection.get("result_id"),
                        "target_object": target_object,
                        "scene_image": scene_image,
                        "scene_epoch": current_epoch,
                        "failed_at_s": time.time(),
                        "molmopoint_attempts": (
                            int(budget.get("molmopoint_attempts") or 0)
                            if same_grounding_request
                            else 0
                        ),
                    }
                    self.facts[REFERENCE_LOCALIZATION_FAILURE_KEY] = _memory_fact_entry(
                        failure,
                        source=name,
                    )
                    self.record("asset_reference_localization_failed", dict(failure))
                    self._save_working_memory()
                continue
            if name == "molmopoint":
                no_detection = self.sam3_no_detection()
                if isinstance(no_detection, dict):
                    failure = self.reference_localization_failure() or {}
                    if str(failure.get("sam3_result_id") or "") != str(
                        no_detection.get("result_id") or ""
                    ):
                        failure = {
                            "sam3_result_id": no_detection.get("result_id"),
                            "target_object": no_detection.get("target_prompt"),
                            "scene_image": no_detection.get("source_image"),
                            "failed_at_s": time.time(),
                            "molmopoint_attempts": 0,
                        }
                    failure["molmopoint_attempts"] = int(
                        failure.get("molmopoint_attempts") or 0
                    ) + 1
                    failure["scene_epoch"] = self.scene_epoch()
                    if not isinstance(result, dict) or not bool(result.get("success")):
                        failure["last_molmopoint_error"] = (
                            str(result.get("content") or "MolmoPoint failed.")
                            if isinstance(result, dict)
                            else "MolmoPoint failed."
                        )
                        failure["last_molmopoint_failed_at_s"] = time.time()
                    else:
                        failure["last_molmopoint_succeeded_at_s"] = time.time()
                    self.facts[REFERENCE_LOCALIZATION_FAILURE_KEY] = _memory_fact_entry(
                        failure,
                        source=name,
                    )
                    self.facts[TARGET_LOCALIZATION_BUDGET_KEY] = _memory_fact_entry(
                        {
                            "target_object": failure.get("target_object"),
                            "scene_epoch": failure.get("scene_epoch"),
                            "molmopoint_attempts": failure["molmopoint_attempts"],
                            "last_scene_image": failure.get("scene_image"),
                            "updated_at_s": time.time(),
                        },
                        source=name,
                    )
                    self.record(
                        (
                            "molmopoint_localization_succeeded"
                            if isinstance(result, dict) and bool(result.get("success"))
                            else "molmopoint_localization_failed"
                        ),
                        dict(failure),
                    )
                    self._save_working_memory()
                if not isinstance(result, dict) or not bool(result.get("success")):
                    continue
            if not isinstance(result, dict) or not bool(result.get("success")):
                continue
            details = result.get("details")
            if not isinstance(details, dict):
                continue
            outputs = details.get("outputs")
            if not isinstance(outputs, dict):
                outputs = details
            if name == "retrieve_asset_reference":
                self.facts.pop(REFERENCE_LOCALIZATION_FAILURE_KEY, None)
                self.facts.pop(TARGET_LOCALIZATION_BUDGET_KEY, None)
                bundle = outputs.get("localization_bundle")
                if not isinstance(bundle, dict):
                    continue
                scene_image = str(bundle.get("scene_image_ref") or outputs.get("scene_image") or "")
                references = bundle.get("reference_image_refs")
                if not isinstance(references, list):
                    references = outputs.get("reference_images")
                reference_images = [
                    str(item) for item in (references or []) if isinstance(item, str) and item
                ]
                if not scene_image or not reference_images:
                    continue
                positive_points = bundle.get("positive_points")
                if not isinstance(positive_points, list):
                    positive_points = outputs.get("positive_points")
                bbox_xyxy = bundle.get("bbox_xyxy")
                if not isinstance(bbox_xyxy, list):
                    bbox_xyxy = outputs.get("bbox_xyxy")
                point_prompt = isinstance(positive_points, list) and bool(positive_points)
                localizer = outputs.get("localizer")
                if not isinstance(localizer, dict):
                    localizer = {}
                verification = localizer.get("verification")
                exact_instance_verification = (
                    {
                        "decision": "match",
                        "confidence": verification.get("confidence"),
                        "reason": verification.get("reason"),
                        "candidate_crop": verification.get("candidate_crop"),
                        "reference_geometry": verification.get("reference_geometry"),
                        "candidate_geometry": verification.get("candidate_geometry"),
                        "grasp_geometry_family": verification.get("grasp_geometry_family"),
                    }
                    if isinstance(verification, dict)
                    and str(verification.get("decision") or "").lower() == "match"
                    else None
                )
                obligation = {
                    "environment": outputs.get("environment") or bundle.get("environment"),
                    "target_object": outputs.get("target_object") or bundle.get("target_object"),
                    "scene_image": scene_image,
                    "reference_images": reference_images,
                    "marked_scene_image": bundle.get("marked_scene_image_ref")
                    or outputs.get("marked_scene_image"),
                    "positive_points": positive_points if point_prompt else None,
                    "bbox_xyxy": bbox_xyxy,
                    "localization_bundle": dict(bundle),
                    "exact_instance_verification": exact_instance_verification,
                    "required_next_tool": "sam3",
                    "required_parameter": ("positive_points" if point_prompt else "roi_bbox_xyxy"),
                }
                self.facts[PENDING_REFERENCE_LOCALIZATION_KEY] = _memory_fact_entry(
                    obligation,
                    source=name,
                )
                self.facts[TARGET_ASSET_REFERENCE_KEY] = _memory_fact_entry(
                    {
                        "environment": obligation["environment"],
                        "target_object": obligation["target_object"],
                        "memory_query_key": bundle.get("memory_query_key"),
                        "resolved_asset_key": (
                            bundle.get("resolved_asset_key") or outputs.get("resolved_asset_key")
                        ),
                        "memory_resolution": (
                            bundle.get("memory_resolution") or outputs.get("memory_resolution")
                        ),
                        "scene_image": scene_image,
                        "reference_images": reference_images,
                        "positive_points": positive_points if point_prompt else None,
                        "bbox_xyxy": bbox_xyxy,
                        "exact_instance_verification": exact_instance_verification,
                    },
                    source=name,
                )
                self.facts.pop(PENDING_SAM3_SELECTION_KEY, None)
                self.facts.pop(SELECTED_SAM3_DETECTION_KEY, None)
                self.record(
                    "asset_reference_localization_required",
                    {
                        "environment": obligation["environment"],
                        "target_object": obligation["target_object"],
                        "reference_count": len(reference_images),
                    },
                )
                self._save_working_memory()
                continue
            if name == "molmopoint":
                no_detection = self.sam3_no_detection()
                points = outputs.get("points")
                image_sources = outputs.get("image_sources")
                if (
                    not isinstance(no_detection, dict)
                    or not isinstance(points, list)
                    or not isinstance(image_sources, list)
                ):
                    continue
                normalized: list[JsonDict] = []
                scene_image = ""
                selected_index: int | None = None
                for point in points:
                    if not isinstance(point, dict):
                        continue
                    image_index = point.get("image_index")
                    x = point.get("pixel_x")
                    y = point.get("pixel_y")
                    if (
                        not isinstance(image_index, int)
                        or isinstance(image_index, bool)
                        or not 0 <= image_index < len(image_sources)
                        or not _finite_number(x)
                        or not _finite_number(y)
                    ):
                        continue
                    source = image_sources[image_index]
                    if not isinstance(source, str) or not source:
                        continue
                    if selected_index is None:
                        selected_index = image_index
                        scene_image = source
                    if image_index != selected_index:
                        continue
                    normalized.append({"x": float(x), "y": float(y), "label": 1})
                if not scene_image or not normalized:
                    continue
                obligation = {
                    "environment": None,
                    "target_object": no_detection.get("target_prompt"),
                    "scene_image": scene_image,
                    "reference_images": [],
                    "marked_scene_image": None,
                    "positive_points": normalized,
                    "bbox_xyxy": None,
                    "localization_bundle": {
                        "source": "molmopoint",
                        "image_index": selected_index,
                    },
                    "exact_instance_verification": None,
                    "required_next_tool": "sam3",
                    "required_parameter": "positive_points",
                }
                self.facts[PENDING_REFERENCE_LOCALIZATION_KEY] = _memory_fact_entry(
                    obligation,
                    source=name,
                )
                self.facts.pop(REFERENCE_LOCALIZATION_FAILURE_KEY, None)
                self.facts.pop(PENDING_SAM3_SELECTION_KEY, None)
                self.facts.pop(SELECTED_SAM3_DETECTION_KEY, None)
                self.record(
                    "molmopoint_localization_required",
                    {
                        "target_object": obligation["target_object"],
                        "scene_image": scene_image,
                        "point_count": len(normalized),
                    },
                )
                self._save_working_memory()
                continue
            if name != "sam3" or PENDING_REFERENCE_LOCALIZATION_KEY not in self.facts:
                continue
            parameters = details.get("parameters")
            if not isinstance(parameters, dict):
                parameters = call.get("parameters")
            if not isinstance(parameters, dict):
                parameters = {}
            pending = self.pending_reference_localization() or {}
            required_parameter = str(pending.get("required_parameter") or "roi_bbox_xyxy")
            geometry_matches = (
                parameters.get("positive_points") == pending.get("positive_points")
                if required_parameter == "positive_points"
                else parameters.get("roi_bbox_xyxy") is not None
            )
            if (
                str(parameters.get("image") or "") == str(pending.get("scene_image") or "")
                and geometry_matches
            ):
                self.facts.pop(PENDING_REFERENCE_LOCALIZATION_KEY, None)
                self.record(
                    "asset_reference_localization_resolved",
                    {
                        "target_object": pending.get("target_object"),
                        required_parameter: parameters.get(required_parameter),
                    },
                )
                self._save_working_memory()

    def _capture_anygrasp_candidate_policy(self, action: EnvAction) -> None:
        command = action.command if isinstance(action.command, dict) else {}
        for call in command.get("tool_calls", []) or []:
            source_tool = str(call.get("name") or "") if isinstance(call, dict) else ""
            if not isinstance(call, dict) or source_tool not in {
                "grasp_pose_estimate",
                "anygrasp",
                "graspgenx",
                "contact_graspnet",
            }:
                continue
            result = call.get("result")
            if not isinstance(result, dict) or not bool(result.get("success")):
                continue
            details = result.get("details")
            if not isinstance(details, dict):
                continue
            outputs = details.get("outputs")
            if not isinstance(outputs, dict):
                outputs = details
            source_backend = str(outputs.get("selected_backend") or source_tool)
            candidates_value = outputs.get("grasp_candidates")
            if not isinstance(candidates_value, list):
                continue
            raw_candidates = [
                dict(candidate)
                for candidate in candidates_value
                if isinstance(candidate, dict) and str(candidate.get("id") or "")
            ]
            if not raw_candidates:
                continue
            capabilities = self._active_grasp_calibration_capabilities()
            max_gripper_width = float(capabilities["max_gripper_width_m"])
            width_rejections = [
                candidate
                for candidate in raw_candidates
                if not _candidate_fits_gripper(
                    candidate,
                    max_gripper_width_m=max_gripper_width,
                )
            ]
            candidates = [
                candidate
                for candidate in raw_candidates
                if _candidate_fits_gripper(
                    candidate,
                    max_gripper_width_m=max_gripper_width,
                )
            ]
            candidates.sort(key=_grasp_candidate_sort_key)
            for rank, candidate in enumerate(candidates):
                candidate["rank"] = rank
            result_id = str(outputs.get("result_id") or "")
            if not result_id:
                result_id = f"{source_tool}-{int(time.time() * 1000)}"
            selected_target = self.selected_sam3_detection()
            source = outputs.get("source")
            if not isinstance(source, dict):
                source = {}
            source_rgb = str(outputs.get("source_rgb") or source.get("rgb") or "")
            source_depth = str(outputs.get("source_depth") or source.get("depth") or "")
            camera_frame_id = str(
                outputs.get("camera_frame_id") or source.get("camera_frame_id") or ""
            )
            if not camera_frame_id:
                camera_frame_id = self._latest_camera_frame_for_source_image(
                    source_rgb,
                    scene_epoch=self.scene_epoch(),
                )
            previous_policy = self.grasp_candidate_policy()
            existing_recovery = self.grasp_estimation_recovery()
            target_prompt = (
                str(selected_target.get("target_prompt") or "").strip()
                if isinstance(selected_target, dict)
                else ""
            )
            matching_recovery = (
                isinstance(existing_recovery, dict)
                and bool(target_prompt)
                and str(existing_recovery.get("target_prompt") or "").strip()
                == target_prompt
            )
            previous_prompt = (
                str(previous_policy.get("fallback_target_prompt") or "").strip()
                if isinstance(previous_policy, dict)
                else ""
            )
            fallback_attempts = (
                _grasp_fallback_attempts(previous_policy)
                if target_prompt and target_prompt == previous_prompt
                else []
            )
            all_candidates_over_width = bool(raw_candidates) and len(width_rejections) == len(
                raw_candidates
            )
            if all_candidates_over_width:
                _append_grasp_fallback_attempt(
                    fallback_attempts,
                    {
                        "backend": source_backend,
                        "camera_frame_id": camera_frame_id,
                        "source_rgb": source_rgb,
                        "outcome": "all_candidates_over_width",
                        "raw_candidate_count": len(raw_candidates),
                        "width_limit_m": max_gripper_width,
                    },
                )
            policy = {
                "result_id": result_id,
                "source_tool": source_tool,
                "source_backend": source_backend,
                "ranking": "score_descending",
                "status": "active" if candidates else "exhausted",
                "candidate_count": len(candidates),
                "raw_candidate_count": len(raw_candidates),
                "active_rank": 0 if candidates else None,
                "active_candidate": candidates[0] if candidates else None,
                "remaining_candidate_ids": [
                    str(candidate.get("id")) for candidate in candidates[1:]
                ],
                "candidates": candidates,
                "source_rgb": source_rgb,
                "source_depth": source_depth,
                "camera_frame_id": camera_frame_id,
                "grasp_source": dict(source),
                "physical_width_limit_m": max_gripper_width,
                "grasp_calibration_id": capabilities["calibration_id"],
                "grasp_calibration_profile_path": capabilities["profile_path"],
                "candidate_attempt_count": 0,
                "max_candidate_attempts": GRASP_CANDIDATE_MAX_ATTEMPTS,
                "candidate_fallback": False,
                "rejected_candidates": [
                    {
                        "candidate_id": candidate.get("id"),
                        "rank": candidate.get("rank"),
                        "score": candidate.get("score"),
                        "reason": (
                            "candidate width exceeds calibration "
                            f"max_gripper_width_m {max_gripper_width:.4f} m"
                        ),
                        "source": "physical_gripper_width_filter",
                    }
                    for candidate in width_rejections
                ],
                "scene_epoch": self.scene_epoch(),
                "activated_at_s": time.time(),
            }
            if target_prompt:
                policy["fallback_target_prompt"] = target_prompt
            if fallback_attempts:
                policy["fallback_attempts"] = fallback_attempts
            if all_candidates_over_width:
                seed_candidate = min(raw_candidates, key=_grasp_candidate_sort_key)
                policy.update(
                    {
                        "exhaustion_reason": "physical_gripper_width_filter",
                        "fallback_required": True,
                        "refinement_seed_candidate": dict(seed_candidate),
                    }
                )
            elif candidates and matching_recovery:
                policy["recovery_lineage"] = {
                    "recovery_id": existing_recovery.get("recovery_id"),
                    "trigger_class": existing_recovery.get("trigger_class"),
                    "trigger_scope": existing_recovery.get("trigger_scope"),
                    "source_result_id": existing_recovery.get("source_result_id"),
                }
            if isinstance(selected_target, dict) and str(outputs.get("mode") or "targeted") == (
                "targeted"
            ):
                policy["target_detection"] = dict(selected_target)
                geometry_family = str(
                    selected_target.get("target_geometry_family") or ""
                ).strip()
                if geometry_family:
                    policy["compile_hints"] = {
                        "target_geometry_family": geometry_family,
                    }
                handle_policy = _articulated_handle_candidate_policy(
                    candidates=candidates,
                    selected_target=selected_target,
                    task=_active_task_text(self),
                    source_tool=source_tool,
                    source_rgb=source_rgb,
                    source_mask=str(
                        outputs.get("target_mask")
                        or outputs.get("object_mask")
                        or source.get("object_mask")
                        or ""
                    ),
                    camera_frame_id=camera_frame_id,
                    scene_epoch=self.scene_epoch(),
                    camera_extrinsics=self._latest_camera_extrinsics(
                        camera_frame_id=camera_frame_id,
                        scene_epoch=self.scene_epoch(),
                    ),
                )
                if handle_policy is not None:
                    policy.update(handle_policy)
            previous_release = self.placement_release()
            if isinstance(previous_release, dict) and str(
                previous_release.get("status") or ""
            ) in {"retreated", "failed", "invalidated"}:
                self.facts.pop(PLACEMENT_RELEASE_KEY, None)
                self.record(
                    "placement_release_cleared_for_new_grasp",
                    {
                        "previous_candidate_id": previous_release.get("candidate_id"),
                        "previous_placement_pose_id": previous_release.get(
                            "placement_pose_id"
                        ),
                        "new_result_id": result_id,
                    },
                )
            self.facts.pop(LEGACY_ANYGRASP_CANDIDATE_POLICY_KEY, None)
            self.facts.pop(GRASP_REESTIMATION_KEY, None)
            self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
                policy,
                source=source_tool,
            )
            if all_candidates_over_width:
                self._schedule_grasp_estimation_recovery(
                    policy=policy,
                    seed_candidate=policy.get("refinement_seed_candidate"),
                    source=dict(source),
                    rejection={
                        "source": "physical_gripper_width_filter",
                        "reason": "all candidates exceed the physical gripper width limit",
                        "recovery_class": "perception_refinable",
                        "recovery_scope": "view",
                    },
                )
            elif candidates and isinstance(existing_recovery, dict):
                self.facts.pop(GRASP_ESTIMATION_RECOVERY_KEY, None)
                self.record(
                    (
                        "grasp_estimation_recovery_completed"
                        if matching_recovery
                        else "grasp_estimation_recovery_superseded"
                    ),
                    {
                        "recovery_id": existing_recovery.get("recovery_id"),
                        "result_id": result_id,
                        "candidate_count": len(candidates),
                        "source_backend": source_backend,
                        "camera_frame_id": camera_frame_id,
                    },
                )
            self.record(
                f"{source_tool}_candidate_activated",
                {
                    "result_id": result_id,
                    "candidate_id": candidates[0].get("id") if candidates else None,
                    "rank": 0 if candidates else None,
                    "score": candidates[0].get("score") if candidates else None,
                    "candidate_count": len(candidates),
                    "width_rejection_count": len(width_rejections),
                },
            )
            self._save_working_memory()

    def _latest_camera_extrinsics(
        self,
        *,
        camera_frame_id: str,
        scene_epoch: int,
    ) -> JsonDict | None:
        if not camera_frame_id:
            return None
        for event in reversed(self.events):
            if event.event_type != "observation":
                continue
            try:
                observed_epoch = int(event.payload.get("scene_epoch"))
            except (TypeError, ValueError):
                continue
            if observed_epoch != scene_epoch:
                continue
            calibrations = event.payload.get("runtime_camera_calibrations")
            if not isinstance(calibrations, list):
                continue
            for calibration in calibrations:
                if not isinstance(calibration, dict):
                    continue
                if str(calibration.get("frame_id") or "") != camera_frame_id:
                    continue
                extrinsics = calibration.get("extrinsics")
                return dict(extrinsics) if isinstance(extrinsics, dict) else None
        return None

    def _latest_camera_frame_for_source_image(
        self,
        source_image: str,
        *,
        scene_epoch: int,
    ) -> str:
        if not source_image:
            return ""
        for event in reversed(self.events):
            if event.event_type != "observation":
                continue
            try:
                observed_epoch = int(event.payload.get("scene_epoch"))
            except (TypeError, ValueError):
                continue
            if observed_epoch != scene_epoch:
                continue
            sources = event.payload.get("runtime_camera_sources")
            if not isinstance(sources, list):
                continue
            for source in sources:
                if not isinstance(source, dict):
                    continue
                if str(source.get("rgb_path") or "") == source_image:
                    return str(source.get("frame_id") or "")
        return ""

    def _advance_anygrasp_candidate_after_rejection(self, action: EnvAction) -> bool:
        policy = self.grasp_candidate_policy()
        if policy is None or str(policy.get("status") or "") not in {"active", "accepted"}:
            return False
        active = policy.get("active_candidate")
        if not isinstance(active, dict):
            return False
        active_candidate_id = str(active.get("id") or "")
        rejection = _wrist_reference_localization_rejection(
            action,
            active_candidate_id=active_candidate_id,
            execution=self.grasp_execution(),
        )
        if rejection is None:
            rejection = _host_stage_precontact_review_rejection(
                action,
                active_candidate_id=active_candidate_id,
                execution=self.grasp_execution(),
            )
        if rejection is None:
            if str(policy.get("status") or "") == "accepted":
                rejection = _candidate_linked_grasp_outcome_rejection(
                    action,
                    active_candidate_id=active_candidate_id,
                    lift_probe=self.grasp_lift_probe(),
                    articulated_probe=self.articulated_attachment_probe(),
                    attachment_gate=self.attachment_gate(),
                    execution=self.grasp_execution(),
                )
            else:
                rejection = _candidate_linked_rejection(
                    action,
                    active_candidate_id=active_candidate_id,
                    artifacts=self.artifacts,
                    final_refinable_fallback=(
                        policy.get("final_refinable_fallback") is True
                    ),
                )
        if rejection is None:
            return False

        advanced = self._apply_anygrasp_candidate_rejection(
            policy=policy,
            active=active,
            rejection=rejection,
        )
        # Retained ranked candidates are already in the same perception packet;
        # switch directly to the next candidate. Only an exhausted queue needs a
        # fresh observation and alternate-view re-estimation.
        if advanced and isinstance(policy.get("reestimate_required"), dict):
            self._schedule_grasp_recovery(
                xyz=self._action_end_effector_xyz(action),
                rejection=rejection,
                candidate_id=active_candidate_id,
            )
            recovery_class = _grasp_estimation_recovery_class(rejection)
            if (
                str(policy.get("status") or "") == "exhausted"
                and policy.get("final_refinable_fallback") is not True
                and recovery_class in {"perception_refinable", "uncertain_review"}
            ):
                rejection = {
                    **rejection,
                    "recovery_class": recovery_class,
                    "recovery_scope": "candidate",
                }
                policy.update(
                    {
                        "fallback_required": True,
                        "exhaustion_reason": recovery_class,
                        "refinement_seed_candidate": dict(active),
                    }
                )
                attempts = _grasp_fallback_attempts(policy)
                _append_grasp_fallback_attempt(
                    attempts,
                    {
                        "backend": str(policy.get("source_backend") or "anygrasp"),
                        "camera_frame_id": policy.get("camera_frame_id"),
                        "source_rgb": policy.get("source_rgb"),
                        "outcome": f"all_candidates_{recovery_class}",
                        "raw_candidate_count": policy.get("raw_candidate_count"),
                    },
                )
                policy["fallback_attempts"] = attempts
                self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
                    policy,
                    source="grasp_estimation_recovery",
                )
                self._schedule_grasp_estimation_recovery(
                    policy=policy,
                    seed_candidate=_highest_refinable_candidate(policy) or active,
                    source=_policy_grasp_source(policy),
                    rejection=rejection,
                )
        return advanced

    def _schedule_grasp_recovery(
        self,
        *,
        xyz: object,
        rejection: JsonDict,
        candidate_id: str,
    ) -> bool:
        if str(rejection.get("source") or "") not in {
            "candidate_motion_rejected",
            "independent_grasp_outcome_rejected",
            "reconciled_candidate_motion_rejected",
            "safety_check_rejected",
            "grasp_seed_geometry_rejected",
            "independent_precontact_review_rejected",
            "independent_host_stage_review_rejected",
            "wrist_reference_localization_rejected",
            "articulated_attachment_assessment_failed",
        }:
            return False
        target_detection = (self.grasp_candidate_policy() or {}).get("target_detection")
        target_detection = target_detection if isinstance(target_detection, dict) else {}
        previous_view = str(target_detection.get("frame_id") or "agentview")
        recovery = {
            "schema_version": "openeta.grasp_recovery.v1",
            "status": "required",
            "candidate_id": candidate_id,
            "rejection_source": rejection.get("source"),
            "rejection_reason": rejection.get("reason"),
            "purpose": (
                "candidate_reestimate"
                if isinstance(
                    (self.grasp_candidate_policy() or {}).get("reestimate_required"),
                    dict,
                )
                else "candidate_fallback"
            ),
            "reestimate_strategy": "alternate_camera_view",
            "previous_view": previous_view,
            "observation_views": ["agentview", "wrist", "render"],
            "scene_epoch": self.scene_epoch(),
            "required_action": {
                "name": "observe",
                "parameters": {},
            },
            "created_at_s": time.time(),
        }
        self.facts[GRASP_RECOVERY_KEY] = _memory_fact_entry(
            recovery,
            source="structured_candidate_rejection",
        )
        self.record("grasp_recovery_required", dict(recovery))
        return True

    def _advance_grasp_recovery(self, action: EnvAction) -> bool:
        recovery = self.grasp_recovery()
        if not isinstance(recovery, dict) or recovery.get("status") != "required":
            return False
        required = recovery.get("required_action")
        if not isinstance(required, dict) or not _action_matches(action, required):
            return False
        name = str(required.get("name") or "")
        call = _tool_call(action, name)
        if not isinstance(call, dict):
            return False
        completed = _call_result_success(call) and not _motion_call_rejects_candidate(call)
        recovery.update(
            {
                "status": "completed" if completed else "failed",
                "completed_at_s": time.time(),
                "scene_epoch": self.scene_epoch(),
                "result": {
                    "tool": name,
                    "reached_target": completed,
                    "reason": "" if completed else _call_failure_reason(call),
                },
            }
        )
        self.facts[GRASP_RECOVERY_KEY] = _memory_fact_entry(
            recovery,
            source="candidate_reestimate_observation",
        )
        self.record(
            "grasp_recovery_completed" if completed else "grasp_recovery_failed",
            dict(recovery),
        )
        if completed and recovery.get("purpose") == "candidate_reestimate":
            reestimate = (self.grasp_candidate_policy() or {}).get("reestimate_required")
            if isinstance(reestimate, dict):
                reestimate.update(
                    {
                        "status": "pending_observation",
                        "recovery_completed_at_s": time.time(),
                        "previous_view": recovery.get("previous_view"),
                        "reestimate_strategy": recovery.get("reestimate_strategy"),
                    }
                )
                self.facts[GRASP_REESTIMATION_KEY] = _memory_fact_entry(
                    reestimate,
                    source="candidate_reestimate_recovery_completed",
                )
                self.record("grasp_reestimate_observation_required", dict(reestimate))
        return True

    def _schedule_grasp_estimation_recovery(
        self,
        *,
        policy: JsonDict,
        seed_candidate: object,
        source: object,
        rejection: JsonDict,
    ) -> bool:
        trigger_class = str(rejection.get("recovery_class") or "")
        if trigger_class not in {"perception_refinable", "uncertain_review"}:
            return False
        if not isinstance(seed_candidate, dict):
            return False
        target_prompt = str(policy.get("fallback_target_prompt") or "").strip()
        if not target_prompt:
            return False
        existing = self.grasp_estimation_recovery()
        same_target = (
            isinstance(existing, dict)
            and str(existing.get("target_prompt") or "").strip() == target_prompt
        )
        if same_target and isinstance(existing, dict) and existing.get("status") == "blocked":
            return False
        recovery_id = (
            str(existing.get("recovery_id") or "")
            if same_target and isinstance(existing, dict)
            else ""
        ) or f"grasp-refinement-{uuid4()}"
        recovery: JsonDict = {
            "schema_version": "openeta.grasp_estimation_recovery.v1",
            "status": "required",
            "recovery_id": recovery_id,
            "trigger_class": trigger_class,
            "trigger_scope": str(rejection.get("recovery_scope") or "candidate"),
            "trigger_source": rejection.get("source"),
            "trigger_reason": rejection.get("reason"),
            "source_result_id": policy.get("result_id"),
            "source_backend": policy.get("source_backend"),
            "source_camera_frame_id": policy.get("camera_frame_id"),
            "source_rgb": policy.get("source_rgb"),
            "target_prompt": target_prompt,
            "seed_candidate": dict(seed_candidate),
            "created_scene_epoch": self.scene_epoch(),
            "created_at_s": time.time(),
        }
        fallback_candidates = (
            [
                dict(item)
                for item in existing.get("fallback_candidates", [])
                if isinstance(item, dict)
            ]
            if same_target and isinstance(existing, dict)
            else []
        )
        candidate_entry = {
            "candidate": dict(seed_candidate),
            "source_result_id": policy.get("result_id"),
            "source_tool": policy.get("source_tool"),
            "source_backend": policy.get("source_backend"),
            "camera_frame_id": policy.get("camera_frame_id"),
            "source_rgb": policy.get("source_rgb"),
            "source_depth": policy.get("source_depth"),
            "source": dict(source) if isinstance(source, dict) else {},
            "target_detection": (
                dict(policy["target_detection"])
                if isinstance(policy.get("target_detection"), dict)
                else {}
            ),
            "trigger_class": trigger_class,
            "grasp_calibration": {
                "calibration_id": policy.get("grasp_calibration_id"),
                "profile_path": policy.get("grasp_calibration_profile_path"),
                "max_gripper_width_m": policy.get("physical_width_limit_m"),
            },
        }
        _append_refinable_fallback_candidate(fallback_candidates, candidate_entry)
        recovery["fallback_candidates"] = fallback_candidates
        if (
            same_target
            and isinstance(existing, dict)
            and existing.get("hover_completed_scene_epoch") is not None
        ):
            for key in (
                "ik_passed",
                "collision_check_passed",
                "hover_completed_scene_epoch",
                "hover_target_pose",
                "last_failure",
            ):
                if key in existing:
                    recovery[key] = existing[key]
        if (
            str(rejection.get("source") or "") == "wrist_reference_localization_rejected"
            or str(rejection.get("grasp_stage") or "") == "align"
        ):
            recovery["hover_completed_scene_epoch"] = self.scene_epoch()
        self.facts[GRASP_ESTIMATION_RECOVERY_KEY] = _memory_fact_entry(
            recovery,
            source="structured_grasp_estimation_rejection",
        )
        self.record("grasp_estimation_recovery_required", dict(recovery))
        return True

    def _advance_grasp_estimation_recovery(self, action: EnvAction) -> bool:
        recovery = self.grasp_estimation_recovery()
        if not isinstance(recovery, dict) or recovery.get("status") != "required":
            return False
        command = action.command if isinstance(action.command, dict) else {}
        request = command.get("request")
        if not isinstance(request, dict):
            return False
        name = str(request.get("name") or "")
        parameters = request.get("parameters")
        if not isinstance(parameters, dict) or not _is_grasp_estimation_recovery_action(
            name,
            parameters,
            recovery=recovery,
        ):
            return False
        call = _tool_call(action, name)
        if not isinstance(call, dict):
            return False
        outputs = _tool_call_outputs(call)
        success = _call_result_success(call)
        if name == "ik_preview_check":
            passed = success and outputs.get("feasible") is True
            recovery["ik_passed"] = passed
        elif name == "obstacle_avoidance":
            passed = success and outputs.get("clear") is True
            recovery["collision_check_passed"] = passed
        elif name == "move_to":
            passed = success and not _motion_call_rejects_candidate(call)
            if passed:
                recovery["hover_completed_scene_epoch"] = self.scene_epoch()
        else:
            return False
        if not passed:
            recovery.update(
                {
                    "status": "blocked",
                    "last_failure": {
                        "tool": name,
                        "reason": _call_failure_reason(call),
                        "hard_rejection": (
                            "ik_unreachable"
                            if name == "ik_preview_check"
                            else (
                                "collision_or_unsafe_path"
                                if name == "obstacle_avoidance"
                                else "refinement_hover_motion_failed"
                            )
                        ),
                    },
                    "completed_at_s": time.time(),
                }
            )
        self.facts[GRASP_ESTIMATION_RECOVERY_KEY] = _memory_fact_entry(
            recovery,
            source="grasp_estimation_recovery_check",
        )
        self.record(
            (
                "grasp_estimation_recovery_blocked"
                if recovery.get("status") == "blocked"
                else "grasp_estimation_recovery_advanced"
            ),
            {
                "recovery_id": recovery.get("recovery_id"),
                "tool": name,
                "passed": passed,
                "scene_epoch": self.scene_epoch(),
            },
        )
        return True

    def _action_end_effector_xyz(self, action: EnvAction) -> object:
        call = _tool_call(action, "move_to") or _tool_call(action, "follow_eef_trajectory")
        outputs = _tool_call_outputs(call) if isinstance(call, dict) else {}
        motion = outputs.get("motion_summary")
        end = motion.get("end") if isinstance(motion, dict) else None
        xyz = end.get("xyz") if isinstance(end, dict) else None
        return xyz if _finite_xyz(xyz) else self._latest_observed_end_effector_xyz()

    def _latest_observed_end_effector_xyz(self) -> object:
        for event in reversed(self.events):
            if event.event_type != "observation":
                continue
            robot = event.payload.get("robot")
            pose = robot.get("end_effector_pose") if isinstance(robot, dict) else None
            xyz = pose.get("xyz") if isinstance(pose, dict) else None
            if _finite_xyz(xyz):
                return xyz
        return None

    def _apply_anygrasp_candidate_rejection(
        self,
        *,
        policy: JsonDict,
        active: JsonDict,
        rejection: JsonDict,
    ) -> bool:
        """Advance one ranked candidate from linked deterministic evidence."""

        if policy.get("interaction_family") == "articulated_handle":
            return self._apply_articulated_handle_candidate_rejection(
                policy=policy,
                active=active,
                rejection=rejection,
            )

        candidates = policy.get("candidates")
        if not isinstance(candidates, list):
            candidates = []
        current_rank = int(policy.get("active_rank") or 0)
        rejected = policy.get("rejected_candidates")
        if not isinstance(rejected, list):
            rejected = []
        rejected.append(
            {
                "candidate_id": active.get("id"),
                "rank": current_rank,
                "score": active.get("score"),
                "reason": rejection.get("reason"),
                "source": rejection.get("source"),
                "recovery_class": _grasp_estimation_recovery_class(rejection),
                "rejected_at_s": time.time(),
            }
        )
        attempts = int(policy.get("candidate_attempt_count") or 0) + 1
        max_attempts = int(
            policy.get("max_candidate_attempts") or GRASP_CANDIDATE_MAX_ATTEMPTS
        )
        retry_exhausted = attempts >= max_attempts
        next_rank = current_rank + 1
        next_candidate = (
            dict(candidates[next_rank])
            if not retry_exhausted
            and next_rank < len(candidates)
            and isinstance(candidates[next_rank], dict)
            else None
        )
        source_tool = str(policy.get("source_tool") or "grasp_pose_estimate")
        policy.update(
            {
                "status": "active" if next_candidate is not None else "exhausted",
                "active_rank": next_rank if next_candidate is not None else None,
                "active_candidate": next_candidate,
                "remaining_candidate_ids": [
                    str(candidate.get("id"))
                    for candidate in candidates[next_rank + 1 :]
                    if isinstance(candidate, dict)
                ]
                if next_candidate is not None
                else [],
                "rejected_candidates": rejected,
                "candidate_attempt_count": attempts,
                "last_candidate_attempt": {
                    "candidate_id": active.get("id"),
                    "attempt_count": attempts,
                    "max_attempts": max_attempts,
                    "retry_exhausted": retry_exhausted,
                },
                "last_rejection": dict(rejection),
            }
        )
        if next_candidate is None:
            # Once the bounded candidate budget is exhausted, keep the highest
            # scoring candidate only when compilation rejected a strategy-level
            # geometry preference. Motion, safety, and structured perception
            # rejections must continue through recovery instead of reviving a pose
            # that was already shown to be unusable.
            fallback_candidate = (
                dict(candidates[0])
                if candidates and isinstance(candidates[0], dict)
                else None
            )
            recovery_class = _grasp_estimation_recovery_class(rejection)
            score_fallback_allowed = (
                str(rejection.get("source") or "") == "grasp_seed_geometry_rejected"
                and recovery_class == "none"
                and policy.get("final_refinable_fallback") is not True
            )
            if (
                fallback_candidate is not None
                and policy.get("candidate_fallback") is not True
                and score_fallback_allowed
            ):
                fallback_candidate["candidate_fallback"] = True
                policy.update(
                    {
                        "status": "active",
                        "active_rank": int(fallback_candidate.get("rank") or 0),
                        "active_candidate": fallback_candidate,
                        "remaining_candidate_ids": [],
                        "candidate_fallback": True,
                        "fallback_reason": "all_ranked_candidates_failed",
                        "fallback_selected_at_s": time.time(),
                    }
                )
                policy.pop("reestimate_required", None)
                self.facts.pop(GRASP_REESTIMATION_KEY, None)
                self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
                    policy,
                    source=f"{source_tool}_score_fallback",
                )
                self.record(
                    f"{source_tool}_candidate_fallback_activated",
                    {
                        "result_id": policy.get("result_id"),
                        "candidate_id": fallback_candidate.get("id"),
                        "rank": fallback_candidate.get("rank"),
                        "score": fallback_candidate.get("score"),
                        "attempt_count": attempts,
                        "max_attempts": max_attempts,
                        "reason": "all_ranked_candidates_failed",
                        "alignment_filter_bypassed": True,
                    },
                )
                return True
            if policy.get("final_refinable_fallback") is True:
                policy.pop("reestimate_required", None)
                self.facts.pop(GRASP_REESTIMATION_KEY, None)
                self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
                    policy,
                    source=f"{source_tool}_final_fallback_exhausted",
                )
                self.record(
                    f"{source_tool}_final_fallback_exhausted",
                    {
                        "result_id": policy.get("result_id"),
                        "candidate_id": active.get("id"),
                        "attempt_count": attempts,
                        "reason": rejection.get("reason"),
                    },
                )
                return True
            policy["reestimate_required"] = {
                "status": "pending_recovery",
                "reason": (
                    "candidate_retry_limit_exceeded"
                    if retry_exhausted
                    else "ranked_candidate_queue_exhausted"
                ),
                "candidate_id": active.get("id"),
                "attempt_count": attempts,
                "max_attempts": max_attempts,
                "scene_epoch": self.scene_epoch(),
                "created_at_s": time.time(),
                "target_prompt": ((policy.get("target_detection") or {}).get("target_prompt")),
                "source_image": ((policy.get("target_detection") or {}).get("source_image")),
            }
            self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
                policy,
                source=f"{source_tool}_reestimate_required",
            )
            self.facts[GRASP_REESTIMATION_KEY] = _memory_fact_entry(
                dict(policy["reestimate_required"]),
                source=f"{source_tool}_reestimate_required",
            )
            self.record(
                f"{source_tool}_reestimate_required",
                dict(policy["reestimate_required"]),
            )
            return True
        policy.pop("accepted_candidate", None)
        policy.pop("accepted_at_s", None)
        self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
            policy,
            source=f"{source_tool}_greedy_fallback",
        )
        self.record(
            f"{source_tool}_candidate_rejected",
            {
                "result_id": policy.get("result_id"),
                "candidate_id": active.get("id"),
                "rank": current_rank,
                "reason": rejection.get("reason"),
                "source": rejection.get("source"),
                "next_candidate_id": (
                    next_candidate.get("id") if isinstance(next_candidate, dict) else None
                ),
                "exhausted": next_candidate is None,
            },
        )
        if next_candidate is not None:
            self.record(
                f"{source_tool}_candidate_activated",
                {
                    "result_id": policy.get("result_id"),
                    "candidate_id": next_candidate.get("id"),
                    "rank": next_rank,
                    "score": next_candidate.get("score"),
                    "activation_source": "previous_candidate_rejected",
                },
            )
        return True

    def _apply_articulated_handle_candidate_rejection(
        self,
        *,
        policy: JsonDict,
        active: JsonDict,
        rejection: JsonDict,
    ) -> bool:
        """Advance bounded top/front/side queues before one score fallback."""

        candidates_value = policy.get("candidates")
        candidates = (
            [dict(candidate) for candidate in candidates_value if isinstance(candidate, dict)]
            if isinstance(candidates_value, list)
            else []
        )
        by_id = {str(candidate.get("id") or ""): candidate for candidate in candidates}
        active_id = str(active.get("id") or "")
        active_mode = str(policy.get("active_approach_mode") or active.get("approach_mode") or "")
        rejected = policy.get("rejected_candidates")
        if not isinstance(rejected, list):
            rejected = []
        rejected.append(
            {
                "candidate_id": active_id,
                "rank": active.get("rank"),
                "score": active.get("score"),
                "approach_mode": active_mode,
                "reason": rejection.get("reason"),
                "source": rejection.get("source"),
                "recovery_class": _grasp_estimation_recovery_class(rejection),
                "rejected_at_s": time.time(),
            }
        )
        attempts = int(policy.get("candidate_attempt_count") or 0) + 1
        mode_attempts = int(policy.get("mode_attempt_count") or 0) + 1
        max_mode_attempts = int(
            policy.get("mode_max_candidate_attempts") or GRASP_CANDIDATE_MAX_ATTEMPTS
        )
        policy.update(
            {
                "rejected_candidates": rejected,
                "candidate_attempt_count": attempts,
                "mode_attempt_count": mode_attempts,
                "last_candidate_attempt": {
                    "candidate_id": active_id,
                    "attempt_count": attempts,
                    "mode_attempt_count": mode_attempts,
                    "max_mode_attempts": max_mode_attempts,
                    "approach_mode": active_mode,
                },
                "last_rejection": dict(rejection),
            }
        )
        source_tool = str(policy.get("source_tool") or "grasp_pose_estimate")
        if policy.get("candidate_fallback") is True:
            return self._mark_articulated_handle_reestimate_required(
                policy=policy,
                active=active,
                attempts=attempts,
                source_tool=source_tool,
            )

        queues = policy.get("approach_mode_queues")
        mode_order = policy.get("approach_mode_order")
        if not isinstance(queues, dict) or not isinstance(mode_order, list):
            return self._mark_articulated_handle_reestimate_required(
                policy=policy,
                active=active,
                attempts=attempts,
                source_tool=source_tool,
            )
        queue = queues.get(active_mode)
        queue = [str(value) for value in queue] if isinstance(queue, list) else []
        current_mode_rank = int(policy.get("active_mode_rank") or 0)
        next_mode_rank = current_mode_rank + 1
        next_candidate: JsonDict | None = None
        next_mode = active_mode
        activation_source = "previous_candidate_rejected"
        if mode_attempts < max_mode_attempts and next_mode_rank < len(queue):
            next_candidate = by_id.get(queue[next_mode_rank])
        else:
            try:
                active_mode_index = [str(value) for value in mode_order].index(active_mode)
            except ValueError:
                active_mode_index = -1
            for value in mode_order[active_mode_index + 1 :]:
                candidate_ids = queues.get(value)
                if not isinstance(candidate_ids, list) or not candidate_ids:
                    continue
                candidate = by_id.get(str(candidate_ids[0]))
                if candidate is None:
                    continue
                next_mode = str(value)
                next_mode_rank = 0
                next_candidate = candidate
                mode_attempts = 0
                activation_source = "approach_mode_degraded"
                break

        if next_candidate is not None:
            next_candidate = dict(next_candidate)
            next_queue = queues.get(next_mode)
            next_queue = (
                [str(value) for value in next_queue]
                if isinstance(next_queue, list)
                else []
            )
            policy.update(
                {
                    "status": "active",
                    "active_approach_mode": next_mode,
                    "active_mode_rank": next_mode_rank,
                    "active_rank": int(next_candidate.get("rank") or 0),
                    "active_candidate": next_candidate,
                    "remaining_candidate_ids": next_queue[next_mode_rank + 1 :],
                    "mode_attempt_count": mode_attempts,
                    "compile_hints": {
                        "target_geometry_family": "articulated_handle",
                        "approach_mode": next_mode,
                        "strategy_id": _articulated_handle_strategy_id(next_mode),
                    },
                }
            )
            policy.pop("accepted_candidate", None)
            policy.pop("accepted_at_s", None)
            self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
                policy,
                source=f"{source_tool}_articulated_handle_fallback",
            )
            self.record(
                f"{source_tool}_candidate_rejected",
                {
                    "result_id": policy.get("result_id"),
                    "candidate_id": active_id,
                    "rank": active.get("rank"),
                    "approach_mode": active_mode,
                    "reason": rejection.get("reason"),
                    "source": rejection.get("source"),
                    "next_candidate_id": next_candidate.get("id"),
                    "next_approach_mode": next_mode,
                    "activation_source": activation_source,
                },
            )
            self.record(
                f"{source_tool}_candidate_activated",
                {
                    "result_id": policy.get("result_id"),
                    "candidate_id": next_candidate.get("id"),
                    "rank": next_candidate.get("rank"),
                    "score": next_candidate.get("score"),
                    "approach_mode": next_mode,
                    "activation_source": activation_source,
                },
            )
            return True

        fallback_candidate = dict(candidates[0]) if candidates else None
        if fallback_candidate is not None:
            fallback_mode = str(fallback_candidate.get("approach_mode") or "side")
            fallback_candidate.update(
                {
                    "candidate_fallback": True,
                    "fallback_reason": "all_approach_modes_failed",
                }
            )
            policy.update(
                {
                    "status": "active",
                    "active_approach_mode": fallback_mode,
                    "active_mode_rank": None,
                    "active_rank": int(fallback_candidate.get("rank") or 0),
                    "active_candidate": fallback_candidate,
                    "remaining_candidate_ids": [],
                    "candidate_fallback": True,
                    "fallback_reason": "all_approach_modes_failed",
                    "fallback_selected_at_s": time.time(),
                    "compile_hints": {
                        "target_geometry_family": "articulated_handle",
                        "approach_mode": fallback_mode,
                        "strategy_id": _articulated_handle_strategy_id(fallback_mode),
                        "candidate_fallback": True,
                        "fallback_reason": "all_approach_modes_failed",
                    },
                }
            )
            policy.pop("reestimate_required", None)
            self.facts.pop(GRASP_REESTIMATION_KEY, None)
            self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
                policy,
                source=f"{source_tool}_articulated_handle_score_fallback",
            )
            self.record(
                f"{source_tool}_candidate_fallback_activated",
                {
                    "result_id": policy.get("result_id"),
                    "candidate_id": fallback_candidate.get("id"),
                    "rank": fallback_candidate.get("rank"),
                    "score": fallback_candidate.get("score"),
                    "reason": "all_approach_modes_failed",
                    "approach_mode": fallback_mode,
                    "alignment_filter_bypassed": True,
                },
            )
            return True
        return self._mark_articulated_handle_reestimate_required(
            policy=policy,
            active=active,
            attempts=attempts,
            source_tool=source_tool,
        )

    def _mark_articulated_handle_reestimate_required(
        self,
        *,
        policy: JsonDict,
        active: JsonDict,
        attempts: int,
        source_tool: str,
    ) -> bool:
        policy["status"] = "exhausted"
        policy["active_candidate"] = None
        policy["active_rank"] = None
        policy["remaining_candidate_ids"] = []
        policy["reestimate_required"] = {
            "status": "pending_recovery",
            "reason": "articulated_handle_fallback_failed",
            "candidate_id": active.get("id"),
            "attempt_count": attempts,
            "scene_epoch": self.scene_epoch(),
            "created_at_s": time.time(),
            "target_prompt": ((policy.get("target_detection") or {}).get("target_prompt")),
            "source_image": ((policy.get("target_detection") or {}).get("source_image")),
        }
        self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
            policy,
            source=f"{source_tool}_reestimate_required",
        )
        self.facts[GRASP_REESTIMATION_KEY] = _memory_fact_entry(
            dict(policy["reestimate_required"]),
            source=f"{source_tool}_reestimate_required",
        )
        self.record(
            f"{source_tool}_reestimate_required",
            dict(policy["reestimate_required"]),
        )
        return True

    def _capture_compiled_grasp(self, action: EnvAction) -> bool:
        call = _successful_tool_call(action, "compile_grasp_seed")
        if call is None:
            return False
        outputs = _tool_call_outputs(call)
        if outputs.get("schema_version") != "openeta.compiled_grasp_seed.v1":
            return False
        policy = self.anygrasp_candidate_policy() or {}
        active = policy.get("active_candidate")
        if not isinstance(active, dict):
            return False
        candidate_id = str(outputs.get("candidate_id") or "")
        if candidate_id != str(active.get("id") or ""):
            return False
        if _optional_int(outputs.get("scene_epoch"), default=-1) != self.scene_epoch():
            return False
        compile_hints: JsonDict = {
            field: outputs.get(field)
            for field in (
                "target_geometry_family",
                "strategy_id",
                "pregrasp_distance_m",
                "approach_mode",
                "fallback_reason",
            )
            if outputs.get(field) not in (None, "")
        }
        if outputs.get("candidate_fallback") is True:
            compile_hints["candidate_fallback"] = True
        if compile_hints:
            policy["compile_hints"] = compile_hints
            self.facts.pop(LEGACY_ANYGRASP_CANDIDATE_POLICY_KEY, None)
            self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
                policy,
                source="compile_grasp_seed",
            )
        hover = outputs.get("hover_pose")
        gripper_state = self.gripper_command_state() or {}
        already_open = gripper_state.get("position") == 1
        if already_open and isinstance(hover, dict):
            initial_stage = "hover"
            required_action = {
                "name": "move_to",
                "parameters": {"target_pose": _pose_for_epoch(hover, self.scene_epoch())},
            }
        else:
            initial_stage = "open"
            required_action = {
                "name": "gripper_control",
                "parameters": {"position": 1},
            }
        execution = {
            "schema_version": "openeta.grasp_execution.v1",
            "status": "required",
            "stage": initial_stage,
            "candidate_id": candidate_id,
            "compiled_grasp_id": outputs.get("compiled_grasp_id"),
            "scene_epoch": self.scene_epoch(),
            "compiled_grasp": dict(outputs),
            "required_action": required_action,
            "transition_conditions": {
                "hover": (
                    "reach the host pose offset at least 0.15 m opposite "
                    "approach_world_xyz before wrist alignment"
                ),
                "close": (
                    "acknowledge binary gripper position=0; this confirms only the "
                    "latched command, never object attachment"
                ),
            },
            "created_at_s": time.time(),
        }
        self.facts[GRASP_EXECUTION_KEY] = _memory_fact_entry(
            execution,
            source="compile_grasp_seed",
        )
        self.facts.pop(ATTACHMENT_GATE_KEY, None)
        self.facts.pop(GRASP_LIFT_PROBE_KEY, None)
        self.facts.pop(ARTICULATED_ATTACHMENT_PROBE_KEY, None)
        self.record(
            "grasp_execution_started",
            {
                "candidate_id": candidate_id,
                "compiled_grasp_id": outputs.get("compiled_grasp_id"),
                "calibration_status": outputs.get("calibration_status"),
                "scene_epoch": self.scene_epoch(),
            },
        )
        return True

    def _capture_articulated_attachment_probe(self, action: EnvAction) -> bool:
        call = _successful_tool_call(action, "prepare_attachment_probe")
        if call is None:
            return False
        outputs = _tool_call_outputs(call)
        if outputs.get("schema_version") != "openeta.articulated_attachment_probe.v1":
            return False
        execution = self.grasp_execution()
        policy = self.grasp_candidate_policy()
        if (
            not isinstance(execution, dict)
            or execution.get("status") != "required"
            or execution.get("stage") != "prepare_probe"
            or not isinstance(policy, dict)
            or policy.get("interaction_family") != "articulated_handle"
        ):
            return False
        if str(outputs.get("candidate_id") or "") != str(execution.get("candidate_id") or ""):
            return False
        if str(outputs.get("compiled_grasp_id") or "") != str(
            execution.get("compiled_grasp_id") or ""
        ):
            return False
        if _optional_int(outputs.get("scene_epoch"), default=-1) != self.scene_epoch():
            return False
        required_action = outputs.get("required_action")
        if not isinstance(required_action, dict):
            return False
        name = str(required_action.get("name") or "")
        parameters = required_action.get("parameters")
        if name not in {"move_to", "follow_eef_trajectory"} or not isinstance(
            parameters, dict
        ):
            return False
        probe = {
            **dict(outputs),
            "status": "required",
            "prepared_at_s": time.time(),
        }
        execution.update(
            {
                "stage": "probe",
                "scene_epoch": self.scene_epoch(),
                "required_action": {"name": name, "parameters": dict(parameters)},
                "probe_kind": "articulated_attachment",
            }
        )
        self.facts[ARTICULATED_ATTACHMENT_PROBE_KEY] = _memory_fact_entry(
            probe,
            source="prepare_attachment_probe",
        )
        self.facts[GRASP_EXECUTION_KEY] = _memory_fact_entry(
            execution,
            source="prepare_attachment_probe",
        )
        self.record("articulated_attachment_probe_prepared", dict(probe))
        return True

    def _capture_articulated_attachment_probe_result(self, action: EnvAction) -> bool:
        probe = self.articulated_attachment_probe()
        if not isinstance(probe, dict) or probe.get("status") != "required":
            return False
        required = probe.get("required_action")
        if not isinstance(required, dict):
            return False
        command = action.command if isinstance(action.command, dict) else {}
        request = command.get("request")
        if not isinstance(request, dict):
            return False
        name = str(required.get("name") or "")
        parameters = required.get("parameters")
        if (
            str(request.get("name") or "") != name
            or not isinstance(parameters, dict)
            or request.get("parameters") != parameters
        ):
            return False
        call = _tool_call(action, name)
        probe["attempt_count"] = int(probe.get("attempt_count") or 0) + 1
        if (
            not isinstance(call, dict)
            or not _call_result_success(call)
            or _motion_call_rejects_candidate(call)
        ):
            probe["last_attempt_status"] = "failed"
            self.facts[ARTICULATED_ATTACHMENT_PROBE_KEY] = _memory_fact_entry(
                probe,
                source="runtime_articulated_attachment_probe",
            )
            self.record(
                "articulated_attachment_probe_failed",
                {
                    "candidate_id": probe.get("candidate_id"),
                    "attempt_count": probe["attempt_count"],
                },
            )
            return True
        probe.update(
            {
                "status": "completed",
                "completed_at_s": time.time(),
                "last_attempt_status": "executed",
            }
        )
        self.facts[ARTICULATED_ATTACHMENT_PROBE_KEY] = _memory_fact_entry(
            probe,
            source="runtime_articulated_attachment_probe",
        )
        self.record("articulated_attachment_probe_completed", dict(probe))
        execution = self.grasp_execution()
        if isinstance(execution, dict) and execution.get("stage") == "probe":
            self._advance_articulated_probe_to_attachment(execution)
        return True

    def _capture_wrist_alignment(self, action: EnvAction) -> bool:
        execution = self.grasp_execution()
        if not isinstance(execution, dict) or execution.get("stage") != "align":
            return False
        call = _successful_tool_call(action, "compute_wrist_alignment")
        if call is None:
            return False
        outputs = _tool_call_outputs(call)
        if outputs.get("schema_version") != "openeta.wrist_alignment.v1":
            return False
        if outputs.get("compiled_grasp_id") != execution.get("compiled_grasp_id"):
            return False
        if _optional_int(outputs.get("scene_epoch"), default=-1) != self.scene_epoch():
            return False
        aligned_pose = outputs.get("aligned_hover_pose")
        adjusted_contact = outputs.get("adjusted_contact_pose")
        adjusted_precontact = outputs.get("adjusted_precontact_pose")
        if not isinstance(aligned_pose, dict) or not isinstance(adjusted_contact, dict):
            return False
        execution.update(
            {
                "stage": "align_move",
                "scene_epoch": self.scene_epoch(),
                "alignment": dict(outputs),
                "adjusted_contact_pose": dict(adjusted_contact),
                "adjusted_precontact_pose": (
                    dict(adjusted_precontact)
                    if isinstance(adjusted_precontact, dict)
                    else None
                ),
                "required_action": {
                    "name": "move_to",
                    "parameters": {
                        "target_pose": _pose_for_epoch(aligned_pose, self.scene_epoch())
                    },
                },
            }
        )
        self.facts[GRASP_EXECUTION_KEY] = _memory_fact_entry(
            execution,
            source="compute_wrist_alignment",
        )
        self.record(
            "grasp_wrist_alignment_ready",
            {
                "candidate_id": execution.get("candidate_id"),
                "alignment_id": outputs.get("alignment_id"),
                "correction_world_xyz": outputs.get("correction_world_xyz"),
            },
        )
        return True

    def _record_successful_world_mutation(self, action: EnvAction) -> bool:
        command = action.command if isinstance(action.command, dict) else {}
        request = command.get("request")
        name = str(request.get("name") or "") if isinstance(request, dict) else ""
        if name not in {
            "move_to",
            "follow_eef_trajectory",
            "gripper_control",
            "lower_body_control_policy",
        }:
            return False
        if _successful_tool_call(action, name) is None:
            return False
        epoch = self.scene_epoch() + 1
        self.facts[SCENE_EPOCH_KEY] = _memory_fact_entry(
            {"epoch": epoch},
            source="successful_world_mutation",
        )
        self.record("scene_epoch_advanced", {"scene_epoch": epoch, "tool": name})
        return True

    def _capture_gripper_command_state(self, action: EnvAction) -> bool:
        call = _successful_tool_call(action, "gripper_control")
        if call is None:
            return False
        command = action.command if isinstance(action.command, dict) else {}
        request = command.get("request")
        parameters = request.get("parameters") if isinstance(request, dict) else None
        requested_position = (
            parameters.get("position", parameters.get("open"))
            if isinstance(parameters, dict)
            else None
        )
        position = (
            _binary_gripper_position(requested_position) if isinstance(parameters, dict) else None
        )
        if position is None:
            return False
        self._set_gripper_command_state(position, source="acknowledged_gripper_command")
        return True

    def _set_gripper_command_state(self, position: int, *, source: str) -> None:
        state = {
            "schema_version": "openeta.gripper_command_state.v1",
            "position": position,
            "state": "open" if position == 1 else "closed",
            "latched": True,
            "scene_epoch": self.scene_epoch(),
            "updated_at_s": time.time(),
        }
        self.facts[GRIPPER_COMMAND_STATE_KEY] = _memory_fact_entry(state, source=source)
        self.record("gripper_command_state_changed", dict(state))

    def _capture_placement_release(self, action: EnvAction) -> bool:
        command = action.command if isinstance(action.command, dict) else {}
        request = command.get("request")
        if not isinstance(request, dict):
            return False
        name = str(request.get("name") or "")
        parameters = request.get("parameters")
        parameters = parameters if isinstance(parameters, dict) else {}
        if name == "move_to":
            target_pose = parameters.get("target_pose")
            if not isinstance(target_pose, dict):
                return False
            placement_stage = str(target_pose.get("placement_stage") or "")
            call = _successful_tool_call(action, "move_to")
            if call is None:
                return False
            adaptive_release_pose = (
                _near_receptacle_release_pose(call, target_pose=target_pose)
                if placement_stage == "descend"
                else None
            )
            if _motion_call_rejects_candidate(call) and adaptive_release_pose is None:
                return False
            if placement_stage == "retreat":
                release = self.placement_release()
                if (
                    not isinstance(release, dict)
                    or release.get("status") != "released"
                    or str(target_pose.get("source_grasp_id") or "")
                    != str(release.get("candidate_id") or "")
                ):
                    return False
                release.update(
                    {
                        "status": "retreated",
                        "scene_epoch": self.scene_epoch(),
                        "retreat_pose": dict(target_pose),
                        "retreated_at_s": time.time(),
                    }
                )
                self.facts[PLACEMENT_RELEASE_KEY] = _memory_fact_entry(
                    release,
                    source="placement_retreat_completed",
                )
                self.record("placement_retreated", dict(release))
                return True
            if placement_stage not in {"descend", "release"}:
                return False
            release_pose = dict(adaptive_release_pose or target_pose)
            release_pose["placement_stage"] = "release"
            release = {
                "schema_version": "openeta.placement_release.v1",
                "status": "ready",
                "candidate_id": target_pose.get("source_grasp_id"),
                "placement_pose_id": target_pose.get("placement_pose_id"),
                "release_pose": release_pose,
                "scene_epoch": self.scene_epoch(),
                "ready_at_s": time.time(),
                "arrival_stage": (
                    "descend_near_receptacle"
                    if adaptive_release_pose is not None
                    else placement_stage
                ),
            }
            self.facts[PLACEMENT_RELEASE_KEY] = _memory_fact_entry(
                release,
                source="placement_release_pose_reached",
            )
            self.record("placement_release_ready", dict(release))
            return True
        if name != "gripper_control" or _successful_tool_call(action, name) is None:
            return False
        if _binary_gripper_position(parameters.get("position")) != 1:
            return False
        release = self.placement_release()
        host_obligation = _action_host_obligation(action)
        if (
            not isinstance(release, dict)
            and host_obligation.get("stage") == "placement_drop_detected"
        ):
            release = {
                "schema_version": "openeta.placement_release.v1",
                "status": "retreated",
                "candidate_id": host_obligation.get("candidate_id"),
                "placement_pose_id": host_obligation.get("placement_pose_id"),
                "scene_epoch": self.scene_epoch(),
                "release_mode": "detached_over_receptacle",
                "released_at_s": time.time(),
                "retreated_at_s": time.time(),
            }
            self.facts[PLACEMENT_RELEASE_KEY] = _memory_fact_entry(
                release,
                source="placement_drop_detected",
            )
            self._finalize_placement_subgoal(release)
            self.record("placement_drop_completed", dict(release))
            return True
        if not isinstance(release, dict) or release.get("status") != "ready":
            return False
        release.update(
            {
                "status": "released",
                "scene_epoch": self.scene_epoch(),
                "released_at_s": time.time(),
            }
        )
        self.facts[PLACEMENT_RELEASE_KEY] = _memory_fact_entry(
            release,
            source="placement_release_completed",
        )
        self._finalize_placement_subgoal(release)
        self.record("placement_released", dict(release))
        return True

    def _capture_placement_release_denial(self, action: EnvAction) -> bool:
        obligation = _action_host_obligation(action)
        if obligation.get("stage") not in {"release", "placement_drop_detected"}:
            return False
        call = _tool_call(action, "gripper_control")
        if not isinstance(call, dict) or not _call_has_diagnostic(
            call,
            "supervision_denied",
        ):
            return False
        release = self.placement_release()
        if not isinstance(release, dict) or release.get("status") != "ready":
            return False
        result = call.get("result")
        reason = (
            str(result.get("content") or "Independent reviewer denied placement release.")
            if isinstance(result, dict)
            else "Independent reviewer denied placement release."
        )
        release.update(
            {
                "status": "failed",
                "failure_reason": reason,
                "failed_at_s": time.time(),
            }
        )
        self.facts[PLACEMENT_RELEASE_KEY] = _memory_fact_entry(
            release,
            source="placement_release_supervision_denied",
        )
        for key in (
            PENDING_SAM3_SELECTION_KEY,
            SELECTED_SAM3_DETECTION_KEY,
            PENDING_REFERENCE_LOCALIZATION_KEY,
            TARGET_ASSET_REFERENCE_KEY,
            SAM3_NO_DETECTION_KEY,
            GRASP_CANDIDATE_POLICY_KEY,
            LEGACY_ANYGRASP_CANDIDATE_POLICY_KEY,
            GRASP_LIFT_PROBE_KEY,
            ARTICULATED_ATTACHMENT_PROBE_KEY,
            GRASP_EXECUTION_KEY,
            GRASP_ESTIMATION_RECOVERY_KEY,
            ATTACHMENT_GATE_KEY,
        ):
            self.facts.pop(key, None)
        for key in (
            "anyplace_placement_candidates_latest",
            "camera_pose_to_world_world_pose_latest",
        ):
            self.artifacts.pop(key, None)
        self.record(
            "placement_release_failed",
            {
                "candidate_id": release.get("candidate_id"),
                "placement_pose_id": release.get("placement_pose_id"),
                "reason": reason,
            },
        )
        return True

    def _finalize_placement_subgoal(self, release: JsonDict) -> None:
        policy = self.grasp_candidate_policy() or {}
        target = policy.get("target_detection")
        target = target if isinstance(target, dict) else {}
        completed = _memory_fact_value(self.facts.get(COMPLETED_PLACEMENT_SUBGOALS_KEY))
        items = list(completed.get("items") or []) if isinstance(completed, dict) else []
        items.append(
            {
                "candidate_id": release.get("candidate_id"),
                "placement_pose_id": release.get("placement_pose_id"),
                "target_object": target.get("target_prompt") or target.get("label"),
                "release_mode": release.get("release_mode", "explicit_gripper_open"),
                "completed_at_s": time.time(),
            }
        )
        self.facts[COMPLETED_PLACEMENT_SUBGOALS_KEY] = _memory_fact_entry(
            {"items": items[-16:]},
            source="placement_subgoal_completed",
        )
        for key in (
            PENDING_SAM3_SELECTION_KEY,
            SELECTED_SAM3_DETECTION_KEY,
            PENDING_REFERENCE_LOCALIZATION_KEY,
            TARGET_ASSET_REFERENCE_KEY,
            SAM3_NO_DETECTION_KEY,
            GRASP_CANDIDATE_POLICY_KEY,
            LEGACY_ANYGRASP_CANDIDATE_POLICY_KEY,
            GRASP_LIFT_PROBE_KEY,
            ARTICULATED_ATTACHMENT_PROBE_KEY,
            GRASP_EXECUTION_KEY,
            GRASP_ESTIMATION_RECOVERY_KEY,
            ATTACHMENT_GATE_KEY,
        ):
            self.facts.pop(key, None)
        for key in (
            "anyplace_placement_candidates_latest",
            "camera_pose_to_world_world_pose_latest",
        ):
            self.artifacts.pop(key, None)

    def _advance_grasp_execution(self, action: EnvAction) -> bool:
        execution = self.grasp_execution()
        if not isinstance(execution, dict) or execution.get("status") != "required":
            return False
        stage = str(execution.get("stage") or "")
        required = execution.get("required_action")
        if stage == "align":
            return False
        if stage == "probe":
            articulated_probe = self.articulated_attachment_probe()
            if isinstance(articulated_probe, dict):
                if articulated_probe.get("status") == "completed":
                    return self._advance_articulated_probe_to_attachment(execution)
                return False
            return self._advance_probe_to_attachment(execution)
        if stage == "attachment":
            gate = self.attachment_gate() or {}
            verdict = str(gate.get("verdict") or "UNKNOWN").upper()
            if execution.get("attachment_mode") == "articulated_handle":
                actions = execution.get("attachment_actions")
                fail_action = actions.get("fail") if isinstance(actions, dict) else None
                if verdict == "PASS":
                    self._complete_attachment_execution(
                        execution,
                        source="runtime_articulated_attachment_pass",
                    )
                    self.record(
                        "articulated_attachment_confirmed",
                        {
                            "candidate_id": execution.get("candidate_id"),
                            "scene_epoch": self.scene_epoch(),
                        },
                    )
                    return True
                if (
                    verdict == "FAIL"
                    and isinstance(fail_action, dict)
                    and _action_matches(action, fail_action)
                    and _successful_tool_call(action, str(fail_action.get("name") or ""))
                    is not None
                ):
                    return False
                return False
            actions = execution.get("attachment_actions")
            pass_action = actions.get("pass") if isinstance(actions, dict) else None
            successful_pass_call = (
                _successful_tool_call(action, str(pass_action.get("name") or ""))
                if isinstance(pass_action, dict)
                else None
            )
            if (
                isinstance(pass_action, dict)
                and _action_matches(action, pass_action)
                and successful_pass_call is not None
                and not _motion_call_rejects_candidate(successful_pass_call)
            ):
                gate.update(
                    {
                        "pass_action_attempt_count": int(
                            gate.get("pass_action_attempt_count") or 0
                        )
                        + 1,
                        "pass_action_completed": True,
                        "pass_action_completed_at_s": time.time(),
                    }
                )
                self.facts[ATTACHMENT_GATE_KEY] = _memory_fact_entry(
                    gate,
                    source="runtime_attachment_full_lift",
                )
                self.record(
                    "attachment_full_lift_completed",
                    {
                        "candidate_id": execution.get("candidate_id"),
                        "verdict": verdict,
                        "attempt_count": gate["pass_action_attempt_count"],
                    },
                )
            selected = actions.get(verdict.lower()) if isinstance(actions, dict) else None
            if not isinstance(selected, dict) or not _action_matches(action, selected):
                return bool(gate.get("pass_action_completed"))
            if _successful_tool_call(action, str(selected.get("name") or "")) is None:
                return False
            if verdict == "PASS":
                self._complete_attachment_execution(
                    execution,
                    source="runtime_attachment_pass",
                )
                return True
            return False
        if not isinstance(required, dict) or not _grasp_stage_action_matches(
            action,
            required,
            stage=stage,
        ):
            return False
        name = str(required.get("name") or "")
        successful_call = _successful_tool_call(action, name)
        if successful_call is None:
            return False
        if name in {"move_to", "follow_eef_trajectory"} and _motion_call_rejects_candidate(
            successful_call
        ):
            return False

        compiled = execution.get("compiled_grasp")
        compiled = compiled if isinstance(compiled, dict) else {}
        if stage == "open":
            hover = compiled.get("hover_pose")
            if not isinstance(hover, dict):
                return False
            execution.update(
                {
                    "stage": "hover",
                    "scene_epoch": self.scene_epoch(),
                    "required_action": {
                        "name": "move_to",
                        "parameters": {"target_pose": _pose_for_epoch(hover, self.scene_epoch())},
                    },
                }
            )
        elif stage == "hover":
            execution.update(
                {
                    "stage": "align",
                    "scene_epoch": self.scene_epoch(),
                    "required_action": None,
                    "required_tool": "compute_wrist_alignment",
                }
            )
        elif stage == "align_move":
            contact = execution.get("adjusted_contact_pose")
            if not isinstance(contact, dict):
                return False
            precontact = execution.get("adjusted_precontact_pose")
            if isinstance(precontact, dict):
                next_stage = "precontact"
                next_pose = precontact
            else:
                next_stage = "descend"
                next_pose = contact
            execution.update(
                {
                    "stage": next_stage,
                    "scene_epoch": self.scene_epoch(),
                    "required_action": {
                        "name": "move_to",
                        "parameters": {
                            "target_pose": _pose_for_epoch(next_pose, self.scene_epoch())
                        },
                    },
                }
            )
        elif stage == "precontact":
            contact = execution.get("adjusted_contact_pose")
            if not isinstance(contact, dict):
                return False
            execution.update(
                {
                    "stage": "descend",
                    "scene_epoch": self.scene_epoch(),
                    "required_action": {
                        "name": "move_to",
                        "parameters": {
                            "target_pose": _pose_for_epoch(contact, self.scene_epoch())
                        },
                    },
                }
            )
        elif stage == "descend":
            self._mark_active_candidate_accepted(source="compiled_contact_reached")
            execution.update(
                {
                    "stage": "close",
                    "scene_epoch": self.scene_epoch(),
                    "required_action": {
                        "name": "gripper_control",
                        "parameters": {"position": 0},
                    },
                }
            )
        elif stage == "close":
            policy = self.grasp_candidate_policy() or {}
            articulated = policy.get("interaction_family") == "articulated_handle"
            execution.update(
                {
                    "stage": "prepare_probe" if articulated else "probe",
                    "scene_epoch": self.scene_epoch(),
                    "required_action": None,
                    "probe_kind": (
                        "articulated_attachment" if articulated else "grasp_lift"
                    ),
                }
            )
        else:
            return False
        execution.pop("required_tool", None)
        if execution.get("stage") == "align":
            execution["required_tool"] = "compute_wrist_alignment"
        self.facts[GRASP_EXECUTION_KEY] = _memory_fact_entry(
            execution,
            source="runtime_grasp_transition",
        )
        self.record(
            "grasp_execution_transition",
            {
                "candidate_id": execution.get("candidate_id"),
                "completed_stage": stage,
                "next_stage": execution.get("stage"),
                "scene_epoch": self.scene_epoch(),
            },
        )
        return True

    def _advance_probe_to_attachment(self, execution: JsonDict) -> bool:
        probe = self.grasp_lift_probe()
        if not isinstance(probe, dict) or probe.get("status") != "completed":
            return False
        target = ((probe.get("required_parameters") or {}).get("target_pose") or {}).get("xyz")
        if not _finite_xyz(target):
            return False
        full_lift_pose = {
            "frame": "world",
            "xyz": [
                float(target[0]),
                float(target[1]),
                float(target[2]) + GRASP_FULL_LIFT_DISTANCE_M,
            ],
            "source_grasp_id": execution.get("candidate_id"),
            "compiled_grasp_id": execution.get("compiled_grasp_id"),
            "grasp_stage": "full_lift",
            "scene_epoch": self.scene_epoch(),
        }
        execution.update(
            {
                "stage": "attachment",
                "scene_epoch": self.scene_epoch(),
                "required_action": None,
                "attachment_actions": {
                    "pass": {"name": "move_to", "parameters": {"target_pose": full_lift_pose}},
                    "fail": {"name": "gripper_control", "parameters": {"position": 1}},
                },
            }
        )
        self.facts[GRASP_EXECUTION_KEY] = _memory_fact_entry(
            execution,
            source="runtime_attachment_gate",
        )
        self.facts[ATTACHMENT_GATE_KEY] = _memory_fact_entry(
            {
                "status": "pending",
                "verdict": "UNKNOWN",
                "candidate_id": execution.get("candidate_id"),
                "scene_epoch": self.scene_epoch(),
            },
            source="runtime_attachment_gate",
        )
        self.record(
            "attachment_gate_required",
            {
                "candidate_id": execution.get("candidate_id"),
                "scene_epoch": self.scene_epoch(),
            },
        )
        return True

    def _advance_articulated_probe_to_attachment(self, execution: JsonDict) -> bool:
        probe = self.articulated_attachment_probe()
        if (
            not isinstance(probe, dict)
            or probe.get("status") != "completed"
            or str(probe.get("candidate_id") or "")
            != str(execution.get("candidate_id") or "")
        ):
            return False
        recovery_open = {"name": "gripper_control", "parameters": {"position": 1}}
        execution.update(
            {
                "stage": "attachment",
                "scene_epoch": self.scene_epoch(),
                "required_action": None,
                "attachment_mode": "articulated_handle",
                "attachment_actions": {"fail": recovery_open},
            }
        )
        self.facts[GRASP_EXECUTION_KEY] = _memory_fact_entry(
            execution,
            source="runtime_articulated_attachment_gate",
        )
        self.facts[ATTACHMENT_GATE_KEY] = _memory_fact_entry(
            {
                "status": "pending",
                "verdict": "UNKNOWN",
                "candidate_id": execution.get("candidate_id"),
                "scene_epoch": self.scene_epoch(),
                "attachment_mode": "articulated_handle",
                "unknown_observation_count": 0,
            },
            source="runtime_articulated_attachment_gate",
        )
        self.record(
            "attachment_gate_required",
            {
                "candidate_id": execution.get("candidate_id"),
                "scene_epoch": self.scene_epoch(),
                "attachment_mode": "articulated_handle",
            },
        )
        return True

    def _mark_active_candidate_accepted(self, *, source: str) -> bool:
        policy = self.anygrasp_candidate_policy()
        if not isinstance(policy, dict) or policy.get("status") != "active":
            return False
        active = policy.get("active_candidate")
        if not isinstance(active, dict):
            return False
        policy.update(
            {
                "status": "accepted",
                "accepted_candidate": dict(active),
                "accepted_at_s": time.time(),
            }
        )
        self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(policy, source=source)
        self.record(
            "anygrasp_candidate_accepted",
            {
                "result_id": policy.get("result_id"),
                "candidate_id": active.get("id"),
                "rank": policy.get("active_rank"),
                "score": active.get("score"),
                "source": source,
            },
        )
        return True

    def _capture_attachment_verdict(self, action: EnvAction) -> bool:
        execution = self.grasp_execution()
        if not isinstance(execution, dict) or execution.get("stage") != "attachment":
            return False
        if execution.get("attachment_mode") == "articulated_handle":
            return False
        outcome = _action_grasp_outcome(action)
        if outcome not in {"pass", "fail", "unknown"}:
            return False
        candidate_id = str(execution.get("candidate_id") or "")
        existing = self.attachment_gate() or {}
        existing_verdict = str(existing.get("verdict") or "UNKNOWN").upper()
        if (
            outcome == "unknown"
            and existing.get("status") == "resolved"
            and existing_verdict in {"PASS", "FAIL"}
        ):
            return False
        gate = {
            **existing,
            "status": "resolved" if outcome in {"pass", "fail"} else "pending",
            "verdict": outcome.upper(),
            "candidate_id": candidate_id,
            "scene_epoch": self.scene_epoch(),
            "evidence_source": "independent_action_reviewer",
            "updated_at_s": time.time(),
        }
        self.facts[ATTACHMENT_GATE_KEY] = _memory_fact_entry(
            gate,
            source="independent_action_reviewer",
        )
        self.record("attachment_gate_verdict", dict(gate))
        return True

    def _capture_attachment_observation_verdict(
        self,
        observation: EnvObservation,
    ) -> bool:
        execution = self.grasp_execution()
        if (
            not isinstance(execution, dict)
            or execution.get("status") != "required"
            or execution.get("stage") != "attachment"
        ):
            return False
        gate = self.attachment_gate()
        if not isinstance(gate, dict):
            return False
        if execution.get("attachment_mode") == "articulated_handle":
            command = self._latest_action_command()
            request = command.get("request") if isinstance(command, dict) else None
            if (
                str(gate.get("verdict") or "UNKNOWN").upper() != "UNKNOWN"
                or not isinstance(request, dict)
                or request.get("name") != "observe"
                or gate.get("refresh_required") is not True
                or gate.get("unknown_refresh_completed") is True
            ):
                return False
            gate.update(
                {
                    "unknown_refresh_completed": True,
                    "refresh_required": False,
                    "last_observed_scene_epoch": self.scene_epoch(),
                    "last_observed_at_s": time.time(),
                }
            )
            self.facts[ATTACHMENT_GATE_KEY] = _memory_fact_entry(
                gate,
                source="articulated_attachment_observation",
            )
            self.record(
                "articulated_attachment_probe_verdict",
                {
                    "candidate_id": execution.get("candidate_id"),
                    "verdict": "UNKNOWN",
                    "refresh_completed": True,
                    "scene_epoch": self.scene_epoch(),
                },
            )
            return True
        verdict, evidence = _gripper_attachment_evidence(
            observation.robot.gripper_state,
            command_state=self.gripper_command_state(),
        )
        if verdict not in {"PASS", "FAIL"}:
            return False
        gate.update(
            {
                "status": "resolved",
                "verdict": verdict,
                "scene_epoch": self.scene_epoch(),
                "evidence_source": "gripper_observation",
                "evidence": evidence,
                "updated_at_s": time.time(),
            }
        )
        self.facts[ATTACHMENT_GATE_KEY] = _memory_fact_entry(
            gate,
            source="gripper_observation",
        )
        self.record("attachment_gate_verdict", dict(gate))
        if verdict == "PASS" and gate.get("pass_action_completed") is True:
            self._complete_attachment_execution(
                execution,
                source="runtime_attachment_observation",
            )
        return True

    def _capture_articulated_attachment_assessment(self, action: EnvAction) -> bool:
        call = _successful_tool_call(action, "assess_attachment_probe")
        if call is None:
            return False
        outputs = _tool_call_outputs(call)
        if outputs.get("schema_version") != (
            "openeta.articulated_attachment_assessment.v1"
        ):
            return False
        execution = self.grasp_execution()
        gate = self.attachment_gate()
        if (
            not isinstance(execution, dict)
            or execution.get("stage") != "attachment"
            or execution.get("attachment_mode") != "articulated_handle"
            or not isinstance(gate, dict)
        ):
            return False
        if str(outputs.get("candidate_id") or "") != str(execution.get("candidate_id") or ""):
            return False
        verdict = str(outputs.get("verdict") or "").upper()
        if verdict not in {"PASS", "FAIL", "UNKNOWN"}:
            return False
        assessment_count = int(gate.get("assessment_count") or 0) + 1
        gate.update(
            {
                "status": "resolved" if verdict in {"PASS", "FAIL"} else "pending",
                "verdict": verdict,
                "assessment_count": assessment_count,
                "evidence_source": "independent_attachment_reviewer",
                "assessment_reason": outputs.get("reason"),
                "updated_at_s": time.time(),
                "refresh_required": verdict == "UNKNOWN" and assessment_count == 1,
            }
        )
        self.facts[ATTACHMENT_GATE_KEY] = _memory_fact_entry(
            gate,
            source="independent_attachment_reviewer",
        )
        self.record("articulated_attachment_probe_verdict", dict(gate))
        return True

    def _complete_attachment_execution(
        self,
        execution: JsonDict,
        *,
        source: str,
    ) -> None:
        execution.update(
            {
                "status": "completed",
                "stage": "attached",
                "scene_epoch": self.scene_epoch(),
                "required_action": None,
                "completed_at_s": time.time(),
            }
        )
        self.facts[GRASP_EXECUTION_KEY] = _memory_fact_entry(execution, source=source)
        self.record(
            "grasp_attachment_confirmed",
            {"candidate_id": execution.get("candidate_id"), "source": source},
        )

    def _capture_motion_reconciliation(self, action: EnvAction) -> bool:
        command = action.command if isinstance(action.command, dict) else {}
        request = command.get("request")
        if not isinstance(request, dict):
            return False
        name = str(request.get("name") or "")
        if name not in {"move_to", "follow_eef_trajectory", "gripper_control"}:
            return False
        call = _tool_call(action, name)
        outputs = _tool_call_outputs(call) if isinstance(call, dict) else {}
        if outputs.get("motion_outcome") != "unknown":
            return False
        reconciliation = {
            "status": "required",
            "tool": name,
            "intended_parameters": dict(request.get("parameters") or {}),
            "candidate_id": _parameters_grasp_candidate_id(request.get("parameters") or {}),
            "grasp_stage": (self.grasp_execution() or {}).get("stage"),
            "scene_epoch": self.scene_epoch(),
            "created_at_s": time.time(),
        }
        self.facts[MOTION_RECONCILIATION_KEY] = _memory_fact_entry(
            reconciliation,
            source="simulator_action_outcome_unknown",
        )
        self.record("motion_reconciliation_required", dict(reconciliation))
        return True

    def _reconcile_unknown_motion(self, observation: EnvObservation) -> bool:
        reconciliation = self.motion_reconciliation()
        if not isinstance(reconciliation, dict) or reconciliation.get("status") not in {
            "required",
            "unresolved",
        }:
            return False
        parameters = reconciliation.get("intended_parameters")
        parameters = parameters if isinstance(parameters, dict) else {}
        tool_name = str(reconciliation.get("tool") or "")
        target_pose = parameters.get("target_pose")
        target_xyz = target_pose.get("xyz") if isinstance(target_pose, dict) else None
        measured_xyz = observation.robot.end_effector_pose.get("xyz")
        if tool_name == "gripper_control":
            requested_position = parameters.get("position", parameters.get("open"))
            verdict = _reconcile_gripper_position(
                requested_position, observation.robot.gripper_state
            )
            if verdict == "completed":
                self.facts[SCENE_EPOCH_KEY] = _memory_fact_entry(
                    {"epoch": self.scene_epoch() + 1},
                    source="reconciled_world_mutation",
                )
                position = _binary_gripper_position(requested_position)
                if position is not None:
                    self._set_gripper_command_state(
                        position,
                        source="reconciled_gripper_command",
                    )
                self._advance_reconciled_grasp_stage(str(reconciliation.get("grasp_stage") or ""))
            reconciliation.update(
                {
                    "status": verdict,
                    "measured_gripper_state": dict(observation.robot.gripper_state),
                    "resolved_at_s": time.time(),
                }
            )
        elif _finite_xyz(target_xyz) and _finite_xyz(measured_xyz):
            distance = math.sqrt(
                sum(
                    (float(target_xyz[index]) - float(measured_xyz[index])) ** 2
                    for index in range(3)
                )
            )
            tolerance = float(parameters.get("tolerance") or 0.02)
            previous_measured_xyz = reconciliation.get("measured_eef_xyz")
            observation_count = int(reconciliation.get("observation_count") or 0) + 1
            if distance <= max(0.005, tolerance):
                verdict = "completed"
                self.facts[SCENE_EPOCH_KEY] = _memory_fact_entry(
                    {"epoch": self.scene_epoch() + 1},
                    source="reconciled_world_mutation",
                )
                self._advance_reconciled_grasp_stage(str(reconciliation.get("grasp_stage") or ""))
            elif (
                observation_count >= 3
                and _finite_xyz(previous_measured_xyz)
                and math.sqrt(
                    sum(
                        (float(previous_measured_xyz[index]) - float(measured_xyz[index])) ** 2
                        for index in range(3)
                    )
                )
                <= 0.005
            ):
                verdict = "failed"
            elif distance <= 0.20:
                verdict = "required"
            else:
                verdict = "unresolved"
            reconciliation.update(
                {
                    "status": verdict,
                    "observation_count": observation_count,
                    "distance_to_target_m": round(distance, 6),
                    "measured_eef_xyz": [float(value) for value in measured_xyz[:3]],
                    "resolved_at_s": time.time(),
                }
            )
        else:
            reconciliation.update({"status": "unresolved", "resolved_at_s": time.time()})
        self.facts[MOTION_RECONCILIATION_KEY] = _memory_fact_entry(
            reconciliation,
            source="same_handle_observation",
        )
        self.record("motion_reconciliation_result", dict(reconciliation))
        if reconciliation.get("status") == "failed":
            self._reject_candidate_after_failed_motion_reconciliation(reconciliation)
        return True

    def _reject_candidate_after_failed_motion_reconciliation(
        self,
        reconciliation: JsonDict,
    ) -> None:
        policy = self.grasp_candidate_policy()
        if not isinstance(policy, dict) or policy.get("status") != "active":
            return
        active = policy.get("active_candidate")
        if not isinstance(active, dict):
            return
        candidate_id = str(reconciliation.get("candidate_id") or "")
        if not candidate_id or candidate_id != str(active.get("id") or ""):
            return
        advanced = self._apply_anygrasp_candidate_rejection(
            policy=policy,
            active=active,
            rejection={
                "source": "reconciled_candidate_motion_rejected",
                "target_tool": reconciliation.get("tool"),
                "reason": "reconciled_target_not_reached",
            },
        )
        if advanced:
            self._schedule_grasp_recovery(
                xyz=reconciliation.get("measured_eef_xyz"),
                rejection={
                    "source": "reconciled_candidate_motion_rejected",
                    "target_tool": reconciliation.get("tool"),
                    "reason": "reconciled_target_not_reached",
                },
                candidate_id=candidate_id,
            )
            self.facts.pop(GRASP_LIFT_PROBE_KEY, None)
            self.facts.pop(ARTICULATED_ATTACHMENT_PROBE_KEY, None)
            self.facts.pop(GRASP_EXECUTION_KEY, None)
            self.facts.pop(ATTACHMENT_GATE_KEY, None)

    def _advance_reconciled_grasp_stage(self, stage: str) -> None:
        execution = self.grasp_execution()
        if not isinstance(execution, dict) or execution.get("stage") != stage:
            return
        # Reconciliation commits only the already-generated deterministic edge.
        if stage == "hover":
            execution.update(
                {
                    "stage": "align",
                    "required_action": None,
                    "required_tool": "compute_wrist_alignment",
                }
            )
        elif stage == "align_move":
            contact = execution.get("adjusted_contact_pose")
            if isinstance(contact, dict):
                precontact = execution.get("adjusted_precontact_pose")
                next_stage = "precontact" if isinstance(precontact, dict) else "descend"
                next_pose = precontact if isinstance(precontact, dict) else contact
                execution.update(
                    {
                        "stage": next_stage,
                        "required_action": {
                            "name": "move_to",
                            "parameters": {
                                "target_pose": _pose_for_epoch(next_pose, self.scene_epoch())
                            },
                        },
                    }
                )
        elif stage == "precontact":
            contact = execution.get("adjusted_contact_pose")
            if isinstance(contact, dict):
                execution.update(
                    {
                        "stage": "descend",
                        "required_action": {
                            "name": "move_to",
                            "parameters": {
                                "target_pose": _pose_for_epoch(contact, self.scene_epoch())
                            },
                        },
                    }
                )
        elif stage == "descend":
            self._mark_active_candidate_accepted(source="reconciled_contact_reached")
            execution.update(
                {
                    "stage": "close",
                    "required_action": {"name": "gripper_control", "parameters": {"position": 0}},
                }
            )
        elif stage == "open":
            compiled = execution.get("compiled_grasp")
            hover = compiled.get("hover_pose") if isinstance(compiled, dict) else None
            if isinstance(hover, dict):
                execution.update(
                    {
                        "stage": "hover",
                        "required_action": {
                            "name": "move_to",
                            "parameters": {
                                "target_pose": _pose_for_epoch(hover, self.scene_epoch())
                            },
                        },
                    }
                )
        elif stage == "close":
            policy = self.grasp_candidate_policy() or {}
            articulated = policy.get("interaction_family") == "articulated_handle"
            execution.update(
                {
                    "stage": "prepare_probe" if articulated else "probe",
                    "required_action": None,
                    "probe_kind": (
                        "articulated_attachment" if articulated else "grasp_lift"
                    ),
                }
            )
        elif stage == "probe":
            articulated_probe = self.articulated_attachment_probe()
            if isinstance(articulated_probe, dict):
                articulated_probe.update(
                    {
                        "status": "completed",
                        "completed_at_s": time.time(),
                        "last_attempt_status": "reconciled",
                    }
                )
                self.facts[ARTICULATED_ATTACHMENT_PROBE_KEY] = _memory_fact_entry(
                    articulated_probe,
                    source="motion_reconciliation",
                )
                self.record(
                    "articulated_attachment_probe_completed",
                    dict(articulated_probe),
                )
                self._advance_articulated_probe_to_attachment(execution)
                return
            probe = self.grasp_lift_probe()
            if isinstance(probe, dict):
                probe.update(
                    {
                        "status": "completed",
                        "completed_at_s": time.time(),
                        "last_attempt_status": "reconciled",
                    }
                )
                self.facts[GRASP_LIFT_PROBE_KEY] = _memory_fact_entry(
                    probe,
                    source="motion_reconciliation",
                )
                self.record("grasp_lift_probe_reconciled", dict(probe))
                self._advance_probe_to_attachment(execution)
                return
        execution["scene_epoch"] = self.scene_epoch()
        self.facts[GRASP_EXECUTION_KEY] = _memory_fact_entry(
            execution,
            source="motion_reconciliation",
        )

    def _append_transition_ledger(self, action: EnvAction) -> None:
        command = action.command if isinstance(action.command, dict) else {}
        request = command.get("request")
        request = request if isinstance(request, dict) else {}
        name = str(request.get("name") or "")
        call = _tool_call(action, name)
        row = {
            "index": len(self.transition_ledger()),
            "timestamp_s": time.time(),
            "scene_epoch": self.scene_epoch(),
            "tool": name,
            "effect": "world_mutating"
            if name in {"move_to", "gripper_control", "follow_eef_trajectory"}
            else "other",
            "grasp_stage": (self.grasp_execution() or {}).get("stage"),
            "candidate_id": _parameters_grasp_candidate_id(request.get("parameters") or {}),
            "verdict": _transition_call_verdict(call),
        }
        rows = [*self.transition_ledger(), row][-TRANSITION_LEDGER_LIMIT:]
        self.facts[TRANSITION_LEDGER_KEY] = _memory_fact_entry(
            {"rows": rows},
            source="runtime_transition_ledger",
        )

    def record_environment_receipt(
        self,
        *,
        reward: object,
        terminated: bool,
        truncated: bool,
        info: JsonDict | None = None,
    ) -> None:
        try:
            reward_value = float(reward)
        except (TypeError, ValueError):
            reward_value = 0.0
        receipt = {
            "reward": reward_value,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "info": dict(info or {}),
            "scene_epoch": self.scene_epoch(),
            "timestamp_s": time.time(),
        }
        self.facts["latest_environment_receipt"] = _memory_fact_entry(
            receipt,
            source="environment_step",
        )
        self.record("environment_receipt", receipt)
        rows = self.transition_ledger()
        rows.append(
            {
                "index": len(rows),
                "timestamp_s": receipt["timestamp_s"],
                "scene_epoch": self.scene_epoch(),
                "tool": "environment_receipt",
                "reward": reward_value,
                "terminated": bool(terminated),
                "truncated": bool(truncated),
                "verdict": "PASS" if reward_value > 0 else "UNKNOWN",
            }
        )
        self.facts[TRANSITION_LEDGER_KEY] = _memory_fact_entry(
            {"rows": rows[-TRANSITION_LEDGER_LIMIT:]},
            source="runtime_transition_ledger",
        )
        self._save_working_memory()

    def _capture_grasp_lift_probe_obligation(
        self,
        observation: EnvObservation,
    ) -> bool:
        policy = self.anygrasp_candidate_policy()
        if not isinstance(policy, dict) or str(policy.get("status") or "") != "accepted":
            return False
        if policy.get("interaction_family") == "articulated_handle":
            return False
        active = policy.get("active_candidate")
        if not isinstance(active, dict):
            return False
        candidate_id = str(active.get("id") or "")
        if not candidate_id:
            return False
        existing = self.grasp_lift_probe()
        if isinstance(existing, dict) and str(existing.get("candidate_id") or "") == candidate_id:
            return False
        command = self._latest_action_command()
        if not _successful_gripper_close_command(command):
            return False
        pose = observation.robot.end_effector_pose
        xyz = pose.get("xyz") if isinstance(pose, dict) else None
        if not _finite_xyz(xyz):
            return False
        start_xyz = [float(value) for value in xyz[:3]]
        target_xyz = [start_xyz[0], start_xyz[1], start_xyz[2] + GRASP_LIFT_PROBE_DISTANCE_M]
        probe = {
            "status": "required",
            "candidate_id": candidate_id,
            "distance_m": GRASP_LIFT_PROBE_DISTANCE_M,
            "start_eef_xyz": start_xyz,
            "required_parameters": {
                "target_pose": {
                    "frame": "world",
                    "xyz": target_xyz,
                    "probe_type": "grasp_lift",
                    "source_grasp_id": candidate_id,
                }
            },
            "created_at_s": time.time(),
        }
        self.facts[GRASP_LIFT_PROBE_KEY] = _memory_fact_entry(
            probe,
            source="runtime_grasp_lift_probe",
        )
        self.record("grasp_lift_probe_required", dict(probe))
        return True

    def _capture_grasp_lift_probe_result(self, action: EnvAction) -> bool:
        probe = self.grasp_lift_probe()
        if not isinstance(probe, dict) or str(probe.get("status") or "") != "required":
            return False
        command = action.command if isinstance(action.command, dict) else {}
        request = command.get("request")
        if not isinstance(request, dict) or str(request.get("name") or "") != "move_to":
            return False
        parameters = request.get("parameters")
        required = probe.get("required_parameters")
        if not isinstance(parameters, dict) or parameters != required:
            return False
        move_call = next(
            (
                call
                for call in command.get("tool_calls", []) or []
                if isinstance(call, dict) and str(call.get("name") or "") == "move_to"
            ),
            None,
        )
        probe["attempt_count"] = int(probe.get("attempt_count") or 0) + 1
        if (
            not isinstance(move_call, dict)
            or not _call_result_success(move_call)
            or _motion_call_rejects_candidate(move_call)
        ):
            probe["last_attempt_status"] = "failed"
            self.facts[GRASP_LIFT_PROBE_KEY] = _memory_fact_entry(
                probe,
                source="runtime_grasp_lift_probe",
            )
            self.record(
                "grasp_lift_probe_failed",
                {
                    "candidate_id": probe.get("candidate_id"),
                    "attempt_count": probe["attempt_count"],
                },
            )
            return True
        probe.update(
            {
                "status": "completed",
                "completed_at_s": time.time(),
                "last_attempt_status": "executed",
            }
        )
        self.facts[GRASP_LIFT_PROBE_KEY] = _memory_fact_entry(
            probe,
            source="runtime_grasp_lift_probe",
        )
        self.record("grasp_lift_probe_completed", dict(probe))
        return True

    def _latest_action_command(self) -> JsonDict:
        for event in reversed(self.events[:-1]):
            if event.event_type == "action":
                command = event.payload.get("command")
                return dict(command) if isinstance(command, dict) else {}
            if event.event_type in {"observation", "episode_start"}:
                break
        return {}

    def _accept_anygrasp_candidate_after_motion(self, action: EnvAction) -> bool:
        execution = self.grasp_execution()
        if isinstance(execution, dict) and execution.get("status") == "required":
            return False
        policy = self.anygrasp_candidate_policy()
        if policy is None or str(policy.get("status") or "") != "active":
            return False
        active = policy.get("active_candidate")
        if not isinstance(active, dict):
            return False
        if not _candidate_linked_motion_succeeded(
            action,
            active_candidate_id=str(active.get("id") or ""),
            artifacts=self.artifacts,
        ):
            return False
        policy.update(
            {
                "status": "accepted",
                "accepted_candidate": dict(active),
                "accepted_at_s": time.time(),
            }
        )
        source_tool = str(policy.get("source_tool") or "anygrasp")
        self.facts[GRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
            policy,
            source=f"{source_tool}_motion_accepted",
        )
        self.record(
            f"{source_tool}_candidate_accepted",
            {
                "result_id": policy.get("result_id"),
                "candidate_id": active.get("id"),
                "rank": policy.get("active_rank"),
                "score": active.get("score"),
            },
        )
        return True

    def delete_memory(self, key: str, *, namespace: str = "all") -> JsonDict:
        deleted: JsonDict = {}
        if namespace in {"all", "facts"}:
            deleted["facts"] = self.facts.pop(key, None) is not None
        if namespace in {"all", "artifacts"}:
            deleted["artifacts"] = self.artifacts.pop(key, None) is not None
        if namespace in {"all", "skill_notes"}:
            deleted["skill_notes"] = self.skill_notes.pop(key, None) is not None
        self.record("memory_deleted", {"key": key, "namespace": namespace, "deleted": deleted})
        self._save_working_memory()
        return deleted

    def clear_working_memory(self) -> None:
        """Clear persisted working memory without deleting session trace files."""

        self.facts.clear()
        self.artifacts.clear()
        self.skill_notes.clear()
        self.compact_summary = ""
        self.record("working_memory_cleared", {})
        self._save_working_memory()

    def compact(self, *, max_events: int = 8) -> str:
        recent = self.recent_events(max_events)
        conversation_checkpoint = self.conversation.compact()
        self._append_conversation_record(checkpoint_record(conversation_checkpoint))
        parts = [
            f"task={self.task}",
            f"current_user_request={self.current_user_request}",
            f"facts={list(self.facts)}",
            f"artifacts={list(self.artifacts)}",
            f"skill_notes={list(self.skill_notes)}",
            "recent_events=" + ",".join(event.event_type for event in recent),
            "conversation="
            f"{conversation_checkpoint['source_item_count']}->"
            f"{conversation_checkpoint['retained_item_count']}",
        ]
        self.compact_summary = "; ".join(parts)
        self.record("memory_compacted", {"summary": self.compact_summary})
        self._save_working_memory()
        return self.compact_summary

    def recent_events(self, limit: int = 8) -> list[MemoryEvent]:
        if limit <= 0:
            return []
        return self.events[-limit:]

    def latest_human_interaction(self) -> JsonDict | None:
        """Return the latest operator answer from the current episode."""

        for event in reversed(self.events):
            if event.event_type == "episode_start":
                break
            if event.event_type != "human_answer":
                continue
            question = event.payload.get("question")
            answer = event.payload.get("answer")
            if not isinstance(answer, str) or not answer.strip():
                return None
            interaction: JsonDict = {
                "answer": _compact_value(answer.strip()),
                "timestamp_s": event.timestamp_s,
            }
            if isinstance(question, str) and question.strip():
                interaction["question"] = _compact_value(question.strip())
            return interaction
        return None

    def latest_guidance_interaction(self) -> JsonDict | None:
        """Return the latest non-human guidance answer with explicit provenance."""

        for event in reversed(self.events):
            if event.event_type == "episode_start":
                break
            if event.event_type != "guidance_answer":
                continue
            answer = event.payload.get("answer")
            if not isinstance(answer, str) or not answer.strip():
                return None
            interaction: JsonDict = {
                "answer": _compact_value(answer.strip()),
                "source": "guidance_agent",
                "timestamp_s": event.timestamp_s,
            }
            question = event.payload.get("question")
            if isinstance(question, str) and question.strip():
                interaction["question"] = _compact_value(question.strip())
            return interaction
        return None

    def planning_context(self, *, max_events: int = 8) -> JsonDict:
        """Return compact context suitable for a planner prompt or policy."""

        return {
            "session_id": self.session_id,
            "task": self.current_user_request or self.task,
            "session_initial_task": self.task,
            "current_user_request": self.current_user_request,
            "active_environment_task": self.active_environment_task(),
            "conversation": self.conversation.planning_context(max_items=0),
            "metadata": self.metadata,
            "selection_obligation": self.pending_sam3_selection(),
            "selected_sam3_detection": self.selected_sam3_detection(),
            "reference_localization_obligation": self.pending_reference_localization(),
            "target_asset_reference": self.target_asset_reference(),
            "reference_localization_failure": self.reference_localization_failure(),
            "sam3_no_detection": self.sam3_no_detection(),
            "grasp_candidate_policy": self.anygrasp_candidate_policy(),
            "retained_targeted_grasp": self.retained_targeted_grasp(),
            "grasp_lift_probe": self.grasp_lift_probe(),
            "articulated_attachment_probe": self.articulated_attachment_probe(),
            "grasp_execution": self.grasp_execution(),
            "grasp_recovery": self.grasp_recovery(),
            "grasp_estimation_recovery": self.grasp_estimation_recovery(),
            "gripper_command_state": self.gripper_command_state(),
            "attachment_gate": self.attachment_gate(),
            "placement_release": self.placement_release(),
            "motion_reconciliation": self.motion_reconciliation(),
            "scene_epoch": self.scene_epoch(),
            "transition_ledger": self.transition_ledger()[-12:],
            "latest_environment_receipt": self.latest_environment_receipt(),
            "latest_human_interaction": self.latest_human_interaction(),
            "latest_guidance_interaction": self.latest_guidance_interaction(),
            "working_memory": {
                "facts": {
                    key: value
                    for key, value in self.facts.items()
                    if key
                    not in {
                        PENDING_SAM3_SELECTION_KEY,
                        SELECTED_SAM3_DETECTION_KEY,
                        PENDING_REFERENCE_LOCALIZATION_KEY,
                        REFERENCE_LOCALIZATION_FAILURE_KEY,
                        TARGET_ASSET_REFERENCE_KEY,
                        SAM3_NO_DETECTION_KEY,
                        GRASP_CANDIDATE_POLICY_KEY,
                        LEGACY_ANYGRASP_CANDIDATE_POLICY_KEY,
                        GRASP_LIFT_PROBE_KEY,
                        ARTICULATED_ATTACHMENT_PROBE_KEY,
                        GRASP_EXECUTION_KEY,
                        GRASP_RECOVERY_KEY,
                        GRASP_ESTIMATION_RECOVERY_KEY,
                        GRIPPER_COMMAND_STATE_KEY,
                        ATTACHMENT_GATE_KEY,
                        PLACEMENT_RELEASE_KEY,
                        MOTION_RECONCILIATION_KEY,
                        SCENE_EPOCH_KEY,
                        TRANSITION_LEDGER_KEY,
                        ACTIVE_ENVIRONMENT_TASK_KEY,
                    }
                },
                "artifacts": {
                    key: summarize_memory_artifact(value) for key, value in self.artifacts.items()
                },
                "skill_notes": self.skill_notes,
                "compact_summary": self.compact_summary,
            },
            "recent_events": [
                {
                    "type": event.event_type,
                    "timestamp_s": event.timestamp_s,
                    "payload": summarize_event_payload(event.payload),
                }
                for event in self.recent_events(max_events)
            ],
        }

    def _load_working_memory(self) -> None:
        if self.store is None:
            return
        memory = self.store.load_working_memory()
        facts = memory.get("facts", {})
        artifacts = memory.get("artifacts", {})
        skill_notes = memory.get("skill_notes", {})
        if isinstance(facts, dict):
            self.facts = dict(facts)
        if isinstance(artifacts, dict):
            self.artifacts = dict(artifacts)
        if isinstance(skill_notes, dict):
            self.skill_notes = {
                str(skill): list(notes) if isinstance(notes, list) else []
                for skill, notes in skill_notes.items()
            }
        self.compact_summary = str(memory.get("compact_summary", ""))

    def model_conversation_messages(self) -> list[JsonDict]:
        """Return canonical chat messages in provider-compatible form."""

        return self.conversation.model_messages()

    def conversation_checkpoint_summary(self) -> str:
        summary = self.conversation.checkpoint.get("summary")
        return summary if isinstance(summary, str) else ""

    def _append_conversation_record(self, record: JsonDict) -> None:
        if self.store is not None:
            self.store.append_conversation_record(record)

    def _reconstruct_legacy_conversation(self, rows: list[JsonDict]) -> None:
        """Recover user turns from traces created before conversation.jsonl existed."""

        seen: list[str] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            event_type = str(row.get("event_type") or "")
            payload = row.get("payload")
            if not isinstance(payload, dict):
                continue
            if event_type == "user_message":
                text = payload.get("text")
            elif event_type in {"session_start", "episode_start"}:
                text = payload.get("task")
            else:
                continue
            if not isinstance(text, str) or not text.strip():
                continue
            normalized = text.strip()
            if seen and seen[-1] == normalized:
                continue
            item = self.conversation.begin_user_turn(normalized, source="legacy_trace")
            seen.append(item.content)

    def _save_working_memory(self) -> None:
        if self.store is not None:
            self.store.save_working_memory(self)


def _grasp_candidate_sort_key(candidate: JsonDict) -> tuple[float, int]:
    try:
        score = float(candidate.get("score"))
    except (TypeError, ValueError):
        score = float("-inf")
    try:
        rank = int(candidate.get("rank"))
    except (TypeError, ValueError):
        rank = 1_000_000
    return (-score, rank)


def _normalize_grasp_geometry_family(value: object) -> str:
    family = str(value or "").strip().lower()
    return "articulated_handle" if family == "drawer_handle" else family


def _active_task_text(memory: AgentMemory) -> str:
    active = memory.active_environment_task()
    task = active.get("task") if isinstance(active, dict) else None
    if isinstance(task, str) and task.strip():
        return task.strip()
    return str(memory.current_user_request or memory.task or "").strip()


def _articulated_handle_candidate_policy(
    *,
    candidates: list[JsonDict],
    selected_target: JsonDict,
    task: str,
    source_tool: str,
    source_rgb: str,
    source_mask: str,
    camera_frame_id: str,
    scene_epoch: int,
    camera_extrinsics: JsonDict | None,
) -> JsonDict | None:
    """Build host-owned mode queues only for a current targeted handle grasp."""

    if source_tool not in {"grasp_pose_estimate", "anygrasp"} or not candidates:
        return None
    if not camera_frame_id or not isinstance(camera_extrinsics, dict):
        return None
    try:
        selected_epoch = int(selected_target.get("scene_epoch"))
    except (TypeError, ValueError):
        return None
    if selected_epoch != scene_epoch:
        return None
    selected_image = str(selected_target.get("source_image") or "")
    selected_mask = str(selected_target.get("mask_ref") or "")
    if not selected_image or not source_rgb or not _same_local_file(selected_image, source_rgb):
        return None
    if not selected_mask or not source_mask or selected_mask != source_mask:
        return None
    geometry_family = _normalize_grasp_geometry_family(
        selected_target.get("target_geometry_family")
    )
    target_is_handle = _selected_target_is_handle(selected_target)
    if _selected_target_is_non_articulated_handle(selected_target):
        return None
    if geometry_family != "articulated_handle" and not (
        _is_articulated_container_task(task) and target_is_handle
    ):
        return None
    try:
        from agent.tools.grasp_geometry import (
            GraspGeometryError,
            camera_optical_forward_world,
            grasp_candidate_approach_world,
        )

        optical_forward = camera_optical_forward_world(camera_extrinsics)
        grouped: dict[str, list[JsonDict]] = {
            mode: [] for mode in ARTICULATED_HANDLE_APPROACH_MODES
        }
        for candidate in candidates:
            approach_world = grasp_candidate_approach_world(candidate, camera_extrinsics)
            mode = _classify_articulated_handle_approach(
                approach_world,
                optical_forward_world=optical_forward,
            )
            candidate["approach_mode"] = mode
            candidate["native_approach_world_xyz"] = [
                round(float(value), 6) for value in approach_world
            ]
            grouped[mode].append(candidate)
    except (TypeError, ValueError, GraspGeometryError):
        return None
    ordered_modes = [mode for mode in ARTICULATED_HANDLE_APPROACH_MODES if grouped[mode]]
    if not ordered_modes:
        return None
    active_mode = ordered_modes[0]
    active_candidates = grouped[active_mode]
    active = dict(active_candidates[0])
    return {
        "interaction_family": "articulated_handle",
        "normalized_target_geometry_family": "articulated_handle",
        "approach_mode_order": ordered_modes,
        "approach_mode_queues": {
            mode: [str(candidate.get("id")) for candidate in grouped[mode]]
            for mode in ordered_modes
        },
        "active_approach_mode": active_mode,
        "active_mode_rank": 0,
        "active_rank": int(active.get("rank") or 0),
        "active_candidate": active,
        "remaining_candidate_ids": [
            str(candidate.get("id")) for candidate in active_candidates[1:]
        ],
        "candidate_attempt_count": 0,
        "mode_attempt_count": 0,
        "mode_max_candidate_attempts": GRASP_CANDIDATE_MAX_ATTEMPTS,
        "compile_hints": {
            "target_geometry_family": "articulated_handle",
            "approach_mode": active_mode,
            "strategy_id": _articulated_handle_strategy_id(active_mode),
        },
        "approach_classification": {
            "camera_frame_id": camera_frame_id,
            "scene_epoch": scene_epoch,
            "method": "world_approach_with_matching_camera_extrinsics",
        },
    }


def _classify_articulated_handle_approach(
    approach_world: list[float],
    *,
    optical_forward_world: list[float],
) -> str:
    downward_alignment = -float(approach_world[2])
    front_alignment = abs(
        sum(
            float(approach_world[index]) * float(optical_forward_world[index])
            for index in range(3)
        )
    )
    if downward_alignment >= 0.5:
        return "top_down"
    if front_alignment >= 0.5:
        return "front"
    return "side"


def _selected_target_is_handle(selected_target: JsonDict) -> bool:
    values = [
        selected_target.get("target_prompt"),
        selected_target.get("label"),
        selected_target.get("class_name"),
        selected_target.get("name"),
    ]
    return any(
        any(
            token == "handle" or token.endswith("_handle")
            for token in _grasp_word_tokens(str(value or ""))
        )
        for value in values
    )


def _selected_target_is_non_articulated_handle(selected_target: JsonDict) -> bool:
    values = " ".join(
        str(selected_target.get(field) or "")
        for field in ("target_prompt", "label", "class_name", "name")
    )
    tokens = set(_grasp_word_tokens(values))
    portable_handles = {
        "mug",
        "cup",
        "basket",
        "bucket",
        "bag",
        "suitcase",
        "pan",
        "pot",
    }
    non_handle_controls = {"button", "knob", "dial", "lever", "switch"}
    return bool(tokens & portable_handles) or bool(tokens & non_handle_controls)


def _is_articulated_container_task(task: str) -> bool:
    tokens = set(_grasp_word_tokens(task))
    containers = {
        "drawer",
        "drawers",
        "microwave",
        "cabinet",
        "cupboard",
        "fridge",
        "refrigerator",
        "door",
    }
    if not tokens & containers:
        return False
    explicit_interaction = bool(tokens & {"open", "pull"})
    interior_access = bool(tokens & {"in", "inside", "into", "interior"})
    return explicit_interaction or interior_access


def _grasp_word_tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


def _same_local_file(left: str, right: str) -> bool:
    if left == right:
        return True
    try:
        left_path = Path(left)
        right_path = Path(right)
        if not left_path.is_file() or not right_path.is_file():
            return False
        if left_path.stat().st_size != right_path.stat().st_size:
            return False
        return left_path.read_bytes() == right_path.read_bytes()
    except OSError:
        return False


def _articulated_handle_strategy_id(approach_mode: str) -> str:
    return {
        "top_down": "top-down-drawer-handle-panda-p8",
        "front": "native-front-articulated-handle-panda-p8",
        "side": "native-side-articulated-handle-panda-p8",
    }[approach_mode]


def _highest_refinable_candidate(policy: JsonDict) -> JsonDict | None:
    candidates = policy.get("candidates")
    rejected = policy.get("rejected_candidates")
    if not isinstance(candidates, list) or not isinstance(rejected, list):
        return None
    refinable_ids = {
        str(item.get("candidate_id") or "")
        for item in rejected
        if isinstance(item, dict)
        and str(item.get("recovery_class") or "")
        in {"perception_refinable", "uncertain_review"}
    }
    eligible = [
        dict(candidate)
        for candidate in candidates
        if isinstance(candidate, dict) and str(candidate.get("id") or "") in refinable_ids
    ]
    return min(eligible, key=_grasp_candidate_sort_key) if eligible else None


def _policy_grasp_source(policy: JsonDict) -> JsonDict:
    source = policy.get("grasp_source")
    if isinstance(source, dict):
        return dict(source)
    return {
        "mode": "targeted",
        "rgb": policy.get("source_rgb"),
        "depth": policy.get("source_depth"),
        "camera_frame_id": policy.get("camera_frame_id"),
    }


def _append_refinable_fallback_candidate(
    candidates: list[JsonDict],
    entry: JsonDict,
) -> None:
    candidate = entry.get("candidate")
    if not isinstance(candidate, dict):
        return
    identity = (
        str(entry.get("source_result_id") or ""),
        str(candidate.get("id") or ""),
    )
    for item in candidates:
        stored = item.get("candidate")
        stored_identity = (
            str(item.get("source_result_id") or ""),
            str(stored.get("id") or "") if isinstance(stored, dict) else "",
        )
        if stored_identity == identity:
            return
    candidates.append(dict(entry))


def _candidate_fits_gripper(
    candidate: JsonDict,
    *,
    max_gripper_width_m: float,
) -> bool:
    try:
        width = float(candidate.get("width"))
    except (TypeError, ValueError):
        return False
    return math.isfinite(width) and 0.0 < width <= max_gripper_width_m


def _policy_max_gripper_width(policy: JsonDict, *, default: float) -> float:
    try:
        value = float(policy.get("physical_width_limit_m"))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) and 0.0 < value <= 0.2 else default


def _fallback_candidate_capabilities(
    entry: JsonDict,
    *,
    default: JsonDict,
) -> JsonDict:
    capabilities = entry.get("grasp_calibration")
    capabilities = dict(capabilities) if isinstance(capabilities, dict) else {}
    max_width = _policy_max_gripper_width(
        {"physical_width_limit_m": capabilities.get("max_gripper_width_m")},
        default=float(default["max_gripper_width_m"]),
    )
    return {
        "calibration_id": str(
            capabilities.get("calibration_id") or default.get("calibration_id") or ""
        ),
        "profile_path": str(
            capabilities.get("profile_path") or default.get("profile_path") or ""
        ),
        "max_gripper_width_m": max_width,
    }


def _parameters_grasp_candidate_id(parameters: JsonDict) -> str:
    for key in ("source_grasp_id", "grasp_candidate_id"):
        value = parameters.get(key)
        if isinstance(value, str) and value:
            return value
    for key in ("camera_pose", "target_pose", "pose", "eef_pose"):
        pose = parameters.get(key)
        if not isinstance(pose, dict):
            continue
        for id_key in ("id", "source_grasp_id", "grasp_candidate_id"):
            value = pose.get(id_key)
            if isinstance(value, str) and value:
                return value
    target_parameters = parameters.get("target_parameters")
    if isinstance(target_parameters, dict):
        return _parameters_grasp_candidate_id(target_parameters)
    trajectory = parameters.get("trajectory")
    if isinstance(trajectory, list):
        for pose in trajectory:
            if not isinstance(pose, dict):
                continue
            value = pose.get("source_grasp_id") or pose.get("grasp_candidate_id")
            if isinstance(value, str) and value:
                return value
    return ""


def _candidate_linked_rejection(
    action: EnvAction,
    *,
    active_candidate_id: str,
    artifacts: dict[str, JsonDict],
    final_refinable_fallback: bool = False,
) -> JsonDict | None:
    if not active_candidate_id:
        return None
    command = action.command if isinstance(action.command, dict) else {}
    request = command.get("request")
    if not isinstance(request, dict):
        request = {}
    request_name = str(request.get("name") or command.get("request_name") or "")
    parameters = request.get("parameters")
    if not isinstance(parameters, dict):
        parameters = {}
    candidate_id = _parameters_grasp_candidate_id(parameters)
    if not candidate_id:
        candidate_id = _matching_latest_world_pose_candidate(parameters, artifacts)
    if candidate_id != active_candidate_id:
        return None

    failed_safety = next(
        (
            call
            for call in command.get("safety_checks", []) or []
            if isinstance(call, dict) and _safety_call_rejects_candidate(call)
        ),
        None,
    )
    if isinstance(failed_safety, dict):
        return {
            "source": "safety_check_rejected",
            "checker": failed_safety.get("name"),
            "target_tool": request_name,
            "reason": _call_failure_reason(failed_safety),
        }

    if request_name == "compile_grasp_seed":
        failed_compile = next(
            (
                call
                for call in command.get("tool_calls", []) or []
                if isinstance(call, dict)
                and str(call.get("name") or "") == request_name
                and _grasp_seed_compile_rejects_candidate(call)
            ),
            None,
        )
        if isinstance(failed_compile, dict):
            recovery_metadata = _grasp_seed_compile_recovery_metadata(failed_compile)
            return {
                "source": "grasp_seed_geometry_rejected",
                "target_tool": request_name,
                "reason": _call_failure_reason(failed_compile),
                **recovery_metadata,
            }
        if final_refinable_fallback:
            terminal_compile = next(
                (
                    call
                    for call in command.get("tool_calls", []) or []
                    if isinstance(call, dict)
                    and str(call.get("name") or "") == request_name
                    and not _call_result_success(call)
                ),
                None,
            )
            if isinstance(terminal_compile, dict):
                return {
                    "source": "final_fallback_compile_failed",
                    "target_tool": request_name,
                    "reason": _call_failure_reason(terminal_compile),
                }
        return None

    if request_name not in {"move_to", "follow_eef_trajectory"}:
        return None
    failed_review = next(
        (
            call
            for call in command.get("tool_calls", []) or []
            if isinstance(call, dict)
            and str(call.get("name") or "") == request_name
            and _independent_precontact_review_rejects_candidate(call)
        ),
        None,
    )
    if isinstance(failed_review, dict):
        return {
            "source": "independent_precontact_review_rejected",
            "target_tool": request_name,
            "reason": _call_failure_reason(failed_review),
            **_independent_review_recovery_metadata(failed_review),
        }
    failed_tool = next(
        (
            call
            for call in command.get("tool_calls", []) or []
            if isinstance(call, dict)
            and str(call.get("name") or "") == request_name
            and _motion_call_rejects_candidate(call)
        ),
        None,
    )
    if not isinstance(failed_tool, dict):
        return None
    return {
        "source": "candidate_motion_rejected",
        "target_tool": request_name,
        "reason": _call_failure_reason(failed_tool),
    }


def _host_stage_precontact_review_rejection(
    action: EnvAction,
    *,
    active_candidate_id: str,
    execution: JsonDict | None,
) -> JsonDict | None:
    if (
        not active_candidate_id
        or not isinstance(execution, dict)
        or str(execution.get("candidate_id") or "") != active_candidate_id
        or str(execution.get("stage") or "")
        not in {"open", "hover", "align_move", "precontact", "descend", "close"}
    ):
        return None
    required = execution.get("required_action")
    if not isinstance(required, dict):
        return None
    command = action.command if isinstance(action.command, dict) else {}
    request = command.get("request")
    if not isinstance(request, dict):
        return None
    request_name = str(request.get("name") or "")
    parameters = request.get("parameters")
    if (
        request_name != str(required.get("name") or "")
        or not isinstance(parameters, dict)
        or parameters != required.get("parameters")
    ):
        return None
    call = _tool_call(action, request_name)
    if not isinstance(call, dict) or not _independent_precontact_review_rejects_candidate(call):
        return None
    return {
        "source": "independent_host_stage_review_rejected",
        "target_tool": request_name,
        "grasp_stage": execution.get("stage"),
        "reason": _call_failure_reason(call),
        **_independent_review_recovery_metadata(call),
    }


def _wrist_reference_localization_rejection(
    action: EnvAction,
    *,
    active_candidate_id: str,
    execution: JsonDict | None,
) -> JsonDict | None:
    if (
        not active_candidate_id
        or not isinstance(execution, dict)
        or str(execution.get("candidate_id") or "") != active_candidate_id
        or str(execution.get("stage") or "") != "align"
    ):
        return None
    command = action.command if isinstance(action.command, dict) else {}
    request = command.get("request")
    if (
        not isinstance(request, dict)
        or str(request.get("name") or "") != "retrieve_asset_reference"
    ):
        return None
    call = _tool_call(action, "retrieve_asset_reference")
    if not isinstance(call, dict) or _call_result_success(call):
        return None
    result = call.get("result")
    details = result.get("details") if isinstance(result, dict) else None
    outputs = details.get("outputs") if isinstance(details, dict) else None
    diagnostics = details.get("diagnostics") if isinstance(details, dict) else None
    reason = outputs.get("reason") if isinstance(outputs, dict) else None
    diagnostic_codes = {
        str(item.get("code") or "") for item in diagnostics or [] if isinstance(item, dict)
    }
    if (
        str(reason or "") != "object_memory_localization_failed"
        and "object_memory_localization_failed" not in diagnostic_codes
    ):
        return None
    return {
        "source": "wrist_reference_localization_rejected",
        "target_tool": "retrieve_asset_reference",
        "grasp_stage": "align",
        "reason": (
            "Target instance could not be localized in the current wrist view after "
            "the reference localizer exhausted its provider retries."
        ),
    }


def _candidate_linked_grasp_outcome_rejection(
    action: EnvAction,
    *,
    active_candidate_id: str,
    lift_probe: JsonDict | None,
    articulated_probe: JsonDict | None = None,
    attachment_gate: JsonDict | None = None,
    execution: JsonDict | None = None,
) -> JsonDict | None:
    if not active_candidate_id:
        return None
    completed_probe = next(
        (
            probe
            for probe in (lift_probe, articulated_probe)
            if isinstance(probe, dict)
            and str(probe.get("status") or "") == "completed"
            and str(probe.get("candidate_id") or "") == active_candidate_id
        ),
        None,
    )
    if completed_probe is None:
        return None
    command = action.command if isinstance(action.command, dict) else {}
    request = command.get("request")
    if not isinstance(request, dict):
        return None
    request_name = str(request.get("name") or "")
    if request_name not in {"gripper_control", "move_to", "follow_eef_trajectory"}:
        return None
    if (
        completed_probe is articulated_probe
        and isinstance(attachment_gate, dict)
        and str(attachment_gate.get("verdict") or "").upper() == "FAIL"
        and isinstance(execution, dict)
        and execution.get("attachment_mode") == "articulated_handle"
    ):
        actions = execution.get("attachment_actions")
        fail_action = actions.get("fail") if isinstance(actions, dict) else None
        call = _tool_call(action, request_name)
        if (
            isinstance(fail_action, dict)
            and _action_matches(action, fail_action)
            and isinstance(call, dict)
            and _call_result_success(call)
        ):
            return {
                "source": "articulated_attachment_assessment_failed",
                "target_tool": request_name,
                "reason": str(
                    attachment_gate.get("assessment_reason")
                    or "Independent multi-view assessment found no attachment."
                ),
            }
    for call in command.get("tool_calls", []) or []:
        if not isinstance(call, dict) or str(call.get("name") or "") != request_name:
            continue
        supervision = _call_supervision(call)
        review = supervision.get("details") if isinstance(supervision, dict) else None
        if not isinstance(review, dict) or str(review.get("grasp_outcome") or "").lower() not in {
            "fail",
            "failed",
        }:
            continue
        if str(review.get("candidate_id") or "") != active_candidate_id:
            continue
        if request_name == "gripper_control":
            parameters = request.get("parameters")
            if not isinstance(parameters, dict):
                continue
            if _binary_gripper_position(
                parameters.get("position")
            ) != 1 or not _call_result_success(call):
                continue
        elif _call_result_success(call) or supervision.get("allowed") is not False:
            continue
        return {
            "source": "independent_grasp_outcome_rejected",
            "target_tool": request_name,
            "reason": str(supervision.get("reason") or "Visual grasp verification failed."),
        }
    return None


def _call_supervision(call: JsonDict) -> JsonDict | None:
    result = call.get("result")
    details = result.get("details") if isinstance(result, dict) else None
    if not isinstance(details, dict):
        return None
    supervision = details.get("supervision")
    if isinstance(supervision, dict):
        return supervision
    outputs = details.get("outputs")
    if not isinstance(outputs, dict):
        return None
    supervision = outputs.get("supervision")
    return supervision if isinstance(supervision, dict) else None


def _successful_gripper_close_command(command: JsonDict) -> bool:
    request = command.get("request")
    if not isinstance(request, dict) or str(request.get("name") or "") != "gripper_control":
        return False
    parameters = request.get("parameters")
    if not isinstance(parameters, dict):
        return False
    if _binary_gripper_position(parameters.get("position")) != 0:
        return False
    return any(
        isinstance(call, dict)
        and str(call.get("name") or "") == "gripper_control"
        and _call_result_success(call)
        for call in command.get("tool_calls", []) or []
    )


def _finite_xyz(value: object) -> bool:
    return (
        isinstance(value, list | tuple)
        and len(value) >= 3
        and all(
            isinstance(component, int | float) and math.isfinite(float(component))
            for component in value[:3]
        )
    )


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _candidate_linked_motion_succeeded(
    action: EnvAction,
    *,
    active_candidate_id: str,
    artifacts: dict[str, JsonDict],
) -> bool:
    if not active_candidate_id:
        return False
    command = action.command if isinstance(action.command, dict) else {}
    request = command.get("request")
    if not isinstance(request, dict) or str(request.get("name") or "") != "move_to":
        return False
    parameters = request.get("parameters")
    if not isinstance(parameters, dict):
        return False
    candidate_id = _parameters_grasp_candidate_id(parameters)
    if not candidate_id:
        candidate_id = _matching_latest_world_pose_candidate(parameters, artifacts)
    if candidate_id != active_candidate_id:
        return False
    return any(
        isinstance(call, dict)
        and str(call.get("name") or "") == "move_to"
        and str(call.get("status") or "") == "executed"
        and _call_result_success(call)
        and not _motion_call_rejects_candidate(call)
        for call in command.get("tool_calls", []) or []
    )


def _call_result_success(call: JsonDict) -> bool:
    result = call.get("result")
    return isinstance(result, dict) and bool(result.get("success"))


def _call_has_diagnostic(call: JsonDict, code: str) -> bool:
    result = call.get("result")
    details = result.get("details") if isinstance(result, dict) else None
    diagnostics = details.get("diagnostics") if isinstance(details, dict) else None
    return isinstance(diagnostics, list) and any(
        isinstance(diagnostic, dict) and diagnostic.get("code") == code
        for diagnostic in diagnostics
    )


def _tool_call(action: EnvAction, name: str) -> JsonDict | None:
    command = action.command if isinstance(action.command, dict) else {}
    return next(
        (
            call
            for call in command.get("tool_calls", []) or []
            if isinstance(call, dict) and str(call.get("name") or "") == name
        ),
        None,
    )


def _successful_tool_call(action: EnvAction, name: str) -> JsonDict | None:
    call = _tool_call(action, name)
    if not isinstance(call, dict) or not _call_result_success(call):
        return None
    return call


def _tool_call_outputs(call: JsonDict) -> JsonDict:
    result = call.get("result")
    details = result.get("details") if isinstance(result, dict) else None
    if not isinstance(details, dict):
        return {}
    outputs = details.get("outputs")
    return dict(outputs) if isinstance(outputs, dict) else dict(details)


def _assigned_task_from_tool_call(call: JsonDict) -> tuple[str, str] | None:
    result = call.get("result")
    details = result.get("details") if isinstance(result, dict) else None
    if not isinstance(details, dict):
        return None
    outputs = details.get("outputs")
    outputs = outputs if isinstance(outputs, dict) else {}
    state_delta = details.get("state_delta")
    state_delta = state_delta if isinstance(state_delta, dict) else {}
    candidates = (
        ("outputs.assigned_task", outputs.get("assigned_task")),
        (
            "outputs.observation_summary.task",
            _nested_mapping_value(outputs, "observation_summary", "task"),
        ),
        (
            "outputs.initial_observation.observation_summary.task",
            _nested_mapping_value(
                outputs,
                "initial_observation",
                "observation_summary",
                "task",
            ),
        ),
        (
            "outputs.response.observation_summary.task",
            _nested_mapping_value(outputs, "response", "observation_summary", "task"),
        ),
        (
            "state_delta.observation.task",
            _nested_mapping_value(state_delta, "observation", "task"),
        ),
    )
    for source_field, value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip(), source_field
    return None


def _nested_mapping_value(value: object, *path: str) -> object:
    current = value
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _transition_call_verdict(call: JsonDict | None) -> str:
    if not isinstance(call, dict):
        return "UNKNOWN"
    if not _call_result_success(call) or _motion_call_rejects_candidate(call):
        return "FAIL"
    return "PASS"


def _action_matches(action: EnvAction, required: JsonDict) -> bool:
    command = action.command if isinstance(action.command, dict) else {}
    request = command.get("request")
    if not isinstance(request, dict):
        return False
    return str(request.get("name") or "") == str(required.get("name") or "") and request.get(
        "parameters"
    ) == required.get("parameters")


def grasp_reference_action_error(
    *,
    stage: str,
    tool_name: str,
    parameters: JsonDict,
    required_action: JsonDict,
) -> str | None:
    """Validate one staged action against a reference pose safety envelope."""

    required_name = str(required_action.get("name") or "")
    required_parameters = required_action.get("parameters")
    if tool_name != required_name or not isinstance(required_parameters, dict):
        return f"Grasp stage {stage!r} requires tool {required_name!r}."
    if required_name != "move_to":
        if parameters == required_parameters:
            return None
        return (
            f"Grasp stage {stage!r} requires the host-generated {required_name!r} "
            "parameters unchanged."
        )

    reference_pose = required_parameters.get("target_pose")
    target_pose = parameters.get("target_pose")
    if not isinstance(reference_pose, dict) or not isinstance(target_pose, dict):
        return f"Grasp stage {stage!r} requires a concrete move_to target_pose."
    if str(target_pose.get("frame") or "") != "world":
        return "Adjusted grasp poses must remain in the world frame."
    reference_xyz = reference_pose.get("xyz")
    target_xyz = target_pose.get("xyz")
    if not _finite_xyz(reference_xyz) or not _finite_xyz(target_xyz):
        return "Adjusted grasp poses require finite target_pose.xyz values."
    distance = math.sqrt(
        sum((float(target_xyz[index]) - float(reference_xyz[index])) ** 2 for index in range(3))
    )
    if distance > GRASP_REFERENCE_POSITION_TOLERANCE_M:
        return (
            f"Adjusted grasp pose is {distance:.4f} m from the reference, exceeding "
            f"the {GRASP_REFERENCE_POSITION_TOLERANCE_M:.3f} m closed-loop envelope."
        )

    reference_rotation = reference_pose.get("rotation_matrix")
    target_rotation = target_pose.get("rotation_matrix")
    if reference_rotation is not None:
        angle_deg = _rotation_delta_deg(reference_rotation, target_rotation)
        if angle_deg is None:
            return (
                "Adjusted grasp poses must preserve a finite 3x3 rotation_matrix "
                "when the reference pose provides one."
            )
        if angle_deg > GRASP_REFERENCE_ORIENTATION_TOLERANCE_DEG:
            return (
                f"Adjusted grasp orientation is {angle_deg:.2f} deg from the reference, "
                "exceeding the "
                f"{GRASP_REFERENCE_ORIENTATION_TOLERANCE_DEG:.1f} deg envelope."
            )

    reference_candidate = _parameters_grasp_candidate_id(required_parameters)
    target_candidate = _parameters_grasp_candidate_id(parameters)
    if target_candidate and reference_candidate and target_candidate != reference_candidate:
        return (
            f"Adjusted grasp pose references candidate {target_candidate!r}, but stage "
            f"{stage!r} belongs to {reference_candidate!r}."
        )
    return None


def _grasp_stage_action_matches(
    action: EnvAction,
    required: JsonDict,
    *,
    stage: str,
) -> bool:
    command = action.command if isinstance(action.command, dict) else {}
    request = command.get("request")
    if not isinstance(request, dict):
        return False
    parameters = request.get("parameters")
    if not isinstance(parameters, dict):
        return False
    return (
        grasp_reference_action_error(
            stage=stage,
            tool_name=str(request.get("name") or ""),
            parameters=parameters,
            required_action=required,
        )
        is None
    )


def _rotation_delta_deg(reference: object, target: object) -> float | None:
    if not _finite_rotation_matrix(reference) or not _finite_rotation_matrix(target):
        return None
    reference_rows = reference
    target_rows = target
    trace = sum(
        float(reference_rows[row][column]) * float(target_rows[row][column])
        for row in range(3)
        for column in range(3)
    )
    cosine = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    return math.degrees(math.acos(cosine))


def _finite_rotation_matrix(value: object) -> bool:
    return (
        isinstance(value, list | tuple)
        and len(value) == 3
        and all(
            isinstance(row, list | tuple)
            and len(row) == 3
            and all(
                isinstance(component, int | float) and math.isfinite(float(component))
                for component in row
            )
            for row in value
        )
    )


def _pose_for_epoch(pose: JsonDict, epoch: int) -> JsonDict:
    updated = dict(pose)
    updated["scene_epoch"] = epoch
    return updated


def _optional_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _is_anyplace_pose(parameters: JsonDict) -> bool:
    pose = parameters.get("camera_pose")
    if not isinstance(pose, dict):
        return False
    pose_id = str(pose.get("id") or "")
    return pose_id.startswith("place_grasp_") or str(pose.get("source_tool") or "") == "anyplace"


def _reconcile_gripper_position(position: object, state: JsonDict) -> str:
    requested = _binary_gripper_position(position)
    if requested is None:
        return "unresolved"
    is_open = state.get("open")
    openness = state.get("openness")
    if requested == 1:
        if is_open is True or isinstance(openness, (int, float)) and openness >= 0.8:
            return "completed"
    else:
        # A grasped object can stop the fingers well above the empty-close value.
        # Any observed departure from fully open reconciles the binary close
        # transition; attachment is adjudicated only after the lift probe.
        if is_open is False or isinstance(openness, (int, float)) and openness < 0.8:
            return "completed"
    return "unresolved"


def _gripper_attachment_evidence(
    state: JsonDict,
    *,
    command_state: JsonDict | None,
) -> tuple[str, JsonDict]:
    if not isinstance(command_state, dict) or command_state.get("position") != 0:
        return "UNKNOWN", {}
    object_detection = str(state.get("object_detection") or "").strip().lower()
    if object_detection == "object_detected_closing":
        return "PASS", {
            "object_detection": object_detection,
            "position": state.get("position"),
            "position_normalized": state.get("position_normalized"),
            "requested_position": state.get("requested_position"),
        }
    if object_detection == "at_position_no_object":
        return "FAIL", {
            "object_detection": object_detection,
            "position": state.get("position"),
            "position_normalized": state.get("position_normalized"),
            "requested_position": state.get("requested_position"),
        }
    return "UNKNOWN", {}


def _binary_gripper_position(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or numeric not in {0.0, 1.0}:
        return None
    return int(numeric)


def _action_grasp_outcome(action: EnvAction) -> str:
    command = action.command if isinstance(action.command, dict) else {}
    for call in command.get("tool_calls", []) or []:
        if not isinstance(call, dict):
            continue
        supervision = _call_supervision(call)
        review = supervision.get("details") if isinstance(supervision, dict) else None
        if not isinstance(review, dict):
            continue
        outcome = str(review.get("grasp_outcome") or "").strip().lower()
        aliases = {"failed": "fail", "passed": "pass", "uncertain": "unknown"}
        return aliases.get(outcome, outcome)
    return ""


def _action_host_obligation(action: EnvAction) -> JsonDict:
    command = action.command if isinstance(action.command, dict) else {}
    metadata = command.get("metadata")
    planner_metadata = metadata.get("planner_metadata") if isinstance(metadata, dict) else None
    obligation = (
        planner_metadata.get("host_obligation") if isinstance(planner_metadata, dict) else None
    )
    return obligation if isinstance(obligation, dict) else {}


def _safety_call_rejects_candidate(call: JsonDict) -> bool:
    if _call_result_success(call):
        return False
    result = call.get("result")
    if not isinstance(result, dict):
        return False
    details = result.get("details")
    if isinstance(details, dict):
        outputs = details.get("outputs")
        for source in (outputs, details):
            if not isinstance(source, dict):
                continue
            if source.get("feasible") is False or source.get("clear") is False:
                return True
            verdict = str(source.get("verdict") or "").strip().lower()
            if verdict in {"unsafe", "infeasible", "rejected", "collision", "blocked"}:
                return True
    return _failure_text_rejects_candidate(call)


def _independent_precontact_review_rejects_candidate(call: JsonDict) -> bool:
    """Treat any fail-closed pre-contact review as a candidate rejection.

    The independent reviewer uses ``abstain`` when visual evidence is too
    ambiguous to approve a candidate-specific motion.  At this boundary an
    abstention still means that the active grasp candidate cannot be executed,
    so it must consume the same bounded candidate budget as an explicit reject.
    """

    if _call_result_success(call):
        return False
    supervision = _call_supervision(call)
    if not isinstance(supervision, dict):
        return False
    review = supervision.get("details")
    if not isinstance(review, dict):
        return False
    return (
        supervision.get("allowed") is False
        and str(supervision.get("source") or "") == "independent_reviewer"
        and str(review.get("decision") or "").strip().lower() in {"reject", "abstain"}
        and str(review.get("grasp_outcome") or "not_assessed").strip().lower() == "not_assessed"
    )


def _independent_review_recovery_metadata(call: JsonDict) -> JsonDict:
    supervision = _call_supervision(call)
    review = supervision.get("details") if isinstance(supervision, dict) else None
    if not isinstance(review, dict):
        return {}
    recovery_class = str(
        review.get("recovery_class") or review.get("rejection_class") or ""
    ).strip()
    if recovery_class not in {"perception_refinable", "uncertain_review"}:
        return {}
    return {"recovery_class": recovery_class}


def _grasp_seed_compile_rejects_candidate(call: JsonDict) -> bool:
    if _call_result_success(call):
        return False
    result = call.get("result")
    if not isinstance(result, dict):
        return False
    details = result.get("details")
    if not isinstance(details, dict):
        return False
    diagnostics = details.get("diagnostics")
    if not isinstance(diagnostics, list):
        return False
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, dict):
            continue
        if diagnostic.get("candidate_rejection") is True:
            return True
        if str(diagnostic.get("code") or "") != "grasp_seed_compile_failed":
            continue
        message = str(diagnostic.get("message") or "").lower()
        if "candidate width" in message and (
            "restricted bounds" in message or "strategy bounds" in message
        ):
            return True
    return False


def _grasp_seed_compile_recovery_metadata(call: JsonDict) -> JsonDict:
    outputs = _tool_call_outputs(call)
    recovery_class = str(outputs.get("recovery_class") or "")
    rejection_code = str(outputs.get("rejection_code") or "")
    result = call.get("result")
    details = result.get("details") if isinstance(result, dict) else None
    diagnostics = details.get("diagnostics") if isinstance(details, dict) else None
    if isinstance(diagnostics, list):
        for diagnostic in diagnostics:
            if not isinstance(diagnostic, dict):
                continue
            recovery_class = recovery_class or str(diagnostic.get("recovery_class") or "")
            rejection_code = rejection_code or str(diagnostic.get("rejection_code") or "")
    if recovery_class not in {"perception_refinable", "uncertain_review"}:
        return {}
    return {
        "recovery_class": recovery_class,
        "rejection_code": rejection_code,
    }


def _grasp_estimation_recovery_class(rejection: JsonDict) -> str:
    structured = str(rejection.get("recovery_class") or "")
    if structured in {"perception_refinable", "uncertain_review"}:
        return structured
    source = str(rejection.get("source") or "")
    if source in {
        "physical_gripper_width_filter",
        "wrist_reference_localization_rejected",
    }:
        return "perception_refinable"
    if source in {
        "uncertain_review",
        "main_vlm_high_hover_uncertain",
    }:
        return "uncertain_review"
    return "none"


def _is_grasp_estimation_recovery_action(
    tool_name: str,
    parameters: JsonDict,
    *,
    recovery: JsonDict | None,
) -> bool:
    if (
        not isinstance(recovery, dict)
        or recovery.get("status") != "required"
        or tool_name not in {"ik_preview_check", "obstacle_avoidance", "move_to"}
    ):
        return False
    recovery_id = str(recovery.get("recovery_id") or "")
    if not recovery_id:
        return False
    if tool_name == "obstacle_avoidance":
        path = parameters.get("path")
        return isinstance(path, dict) and str(path.get("recovery_id") or "") == recovery_id
    target_pose = parameters.get("target_pose")
    return (
        isinstance(target_pose, dict)
        and target_pose.get("grasp_stage") == "grasp_estimation_refinement_hover"
        and str(target_pose.get("recovery_id") or "") == recovery_id
    )


def _camera_frame_for_source_image(policy: JsonDict, source_image: str) -> str:
    if str(policy.get("source_rgb") or "") == source_image:
        return str(policy.get("camera_frame_id") or "")
    return ""


def _motion_call_rejects_candidate(call: JsonDict) -> bool:
    result = call.get("result")
    if not isinstance(result, dict):
        return False
    details = result.get("details")
    if not isinstance(details, dict):
        return False
    if _structured_motion_reached_target(details) is False:
        if _safe_precontact_pose_is_near_target(details):
            return False
        return True
    if _call_result_success(call):
        return False
    diagnostics = details.get("diagnostics")
    if isinstance(diagnostics, list):
        for diagnostic in diagnostics:
            if not isinstance(diagnostic, dict):
                continue
            if diagnostic.get("candidate_rejection") is True:
                return True
            if str(diagnostic.get("code") or "") in {
                "grasp_candidate_collision",
                "grasp_candidate_infeasible",
                "grasp_candidate_unreachable",
            }:
                return True
    outputs = details.get("outputs")
    if isinstance(outputs, dict):
        for source in (outputs, outputs.get("motion_summary"), outputs.get("response")):
            if not isinstance(source, dict):
                continue
            if source.get("candidate_rejection") is True:
                return True
            failure_class = str(source.get("failure_class") or "").strip().lower()
            if failure_class in {
                "grasp_candidate_collision",
                "grasp_candidate_infeasible",
                "grasp_candidate_unreachable",
            }:
                return True
    return False


def _safe_precontact_pose_is_near_target(details: JsonDict) -> bool:
    parameters = details.get("parameters")
    target_pose = parameters.get("target_pose") if isinstance(parameters, dict) else None
    stage = str(target_pose.get("grasp_stage") or "") if isinstance(target_pose, dict) else ""
    if stage not in {"hover", "align"}:
        return False
    outputs = details.get("outputs")
    response = outputs.get("response") if isinstance(outputs, dict) else None
    motion = response.get("motion_summary") if isinstance(response, dict) else None
    if not isinstance(motion, dict):
        motion = outputs.get("motion_summary") if isinstance(outputs, dict) else None
    if not isinstance(motion, dict):
        return False
    collision = motion.get("collision")
    if isinstance(collision, dict) and collision.get("detected") is True:
        return False
    end = motion.get("end")
    target = motion.get("target")
    end_xyz = end.get("xyz") if isinstance(end, dict) else None
    if isinstance(target, dict):
        target_xyz = target.get("xyz")
        if not _finite_xyz(target_xyz):
            target_xyz = [target.get("x"), target.get("y"), target.get("z")]
    else:
        target_xyz = None
    if not _finite_xyz(end_xyz) or not _finite_xyz(target_xyz):
        return False
    position_error_m = math.sqrt(
        sum((float(end_xyz[index]) - float(target_xyz[index])) ** 2 for index in range(3))
    )
    return position_error_m <= 0.04


def _near_receptacle_release_pose(
    call: JsonDict,
    *,
    target_pose: JsonDict,
) -> JsonDict | None:
    """Accept a collision-free descend that stalls slightly above a receptacle."""

    outputs = _tool_call_outputs(call)
    response = outputs.get("response") if isinstance(outputs, dict) else None
    motion = response.get("motion_summary") if isinstance(response, dict) else None
    if not isinstance(motion, dict):
        motion = outputs.get("motion_summary") if isinstance(outputs, dict) else None
    if not isinstance(motion, dict) or motion.get("reached_target") is not False:
        return None
    collision = motion.get("collision")
    if isinstance(collision, dict) and collision.get("detected") is True:
        return None
    end = motion.get("end")
    end_xyz = end.get("xyz") if isinstance(end, dict) else None
    target_xyz = target_pose.get("xyz")
    if not _finite_xyz(end_xyz) or not _finite_xyz(target_xyz):
        return None
    xy_error_m = math.hypot(
        float(end_xyz[0]) - float(target_xyz[0]),
        float(end_xyz[1]) - float(target_xyz[1]),
    )
    height_error_m = float(end_xyz[2]) - float(target_xyz[2])
    if xy_error_m > 0.04 or not 0.0 <= height_error_m <= 0.06:
        return None
    release_pose = dict(target_pose)
    release_pose["xyz"] = [float(value) for value in end_xyz[:3]]
    release_pose["adaptive_release"] = {
        "reason": "controller_stalled_safely_above_receptacle",
        "xy_error_m": round(xy_error_m, 6),
        "height_above_requested_m": round(height_error_m, 6),
    }
    return release_pose


def _structured_motion_reached_target(details: JsonDict) -> bool | None:
    outputs = details.get("outputs")
    response = outputs.get("response") if isinstance(outputs, dict) else None
    state_delta = details.get("state_delta")
    sources = (
        outputs.get("motion_summary") if isinstance(outputs, dict) else None,
        response.get("motion_summary") if isinstance(response, dict) else None,
        details.get("motion_summary"),
        state_delta.get("motion") if isinstance(state_delta, dict) else None,
    )
    for source in sources:
        if isinstance(source, dict) and isinstance(source.get("reached_target"), bool):
            return source["reached_target"]
    return None


def _failure_text_rejects_candidate(call: JsonDict) -> bool:
    text_parts = [str(call.get("reason") or "")]
    result = call.get("result")
    if isinstance(result, dict):
        text_parts.append(str(result.get("content") or ""))
        details = result.get("details")
        if isinstance(details, dict):
            text_parts.extend(
                str(details.get(key) or "") for key in ("reason", "verdict", "diagnostic")
            )
            outputs = details.get("outputs")
            if isinstance(outputs, dict):
                text_parts.extend(
                    str(outputs.get(key) or "") for key in ("reason", "verdict", "diagnostic")
                )
            diagnostics = details.get("diagnostics")
            if isinstance(diagnostics, list):
                text_parts.extend(str(item) for item in diagnostics if isinstance(item, dict))
    text = " ".join(text_parts).lower()
    non_pose_failures = (
        "timeout",
        "transport",
        "connection",
        "mcp_call_failed",
        "not configured",
        "missing_",
        "invalid_",
        "malformed",
        "schema",
        "operator_denied",
        "user denied",
        "session",
        "backend unavailable",
    )
    if any(marker in text for marker in non_pose_failures):
        return False
    pose_rejections = (
        "unsafe",
        "infeasible",
        "unreachable",
        "outside_workspace",
        "outside workspace",
        "ik failed",
        "ik target",
        "joint limit",
        "path blocked",
        "target not reached",
        "motion rejected",
        "pose rejected",
    )
    return any(marker in text for marker in pose_rejections)


def _call_failure_reason(call: JsonDict) -> str:
    result = call.get("result")
    if isinstance(result, dict):
        details = result.get("details")
        if isinstance(details, dict):
            if _structured_motion_reached_target(details) is False:
                return "Simulator motion summary reports that the target was not reached."
            outputs = details.get("outputs")
            for source in (outputs, details):
                if not isinstance(source, dict):
                    continue
                for key in ("reason", "verdict", "diagnostic"):
                    value = source.get(key)
                    if isinstance(value, str) and value:
                        return value
        content = result.get("content")
        if isinstance(content, str) and content:
            return content
    reason = call.get("reason")
    return str(reason or "candidate-linked tool rejected the grasp pose")


def _matching_latest_world_pose_candidate(
    parameters: JsonDict,
    artifacts: dict[str, JsonDict],
) -> str:
    target_xyz = _parameters_target_xyz(parameters)
    if target_xyz is None:
        target_parameters = parameters.get("target_parameters")
        if isinstance(target_parameters, dict):
            target_xyz = _parameters_target_xyz(target_parameters)
    if target_xyz is None:
        return ""
    latest: tuple[float, JsonDict] | None = None
    for entry in artifacts.values():
        if not isinstance(entry, dict):
            continue
        value = entry.get("value")
        if not isinstance(value, dict) or value.get("type") != "world_pose":
            continue
        source_grasp_id = value.get("source_grasp_id")
        world_xyz = value.get("translation_xyz")
        if not isinstance(source_grasp_id, str) or not _xyz_equal(target_xyz, world_xyz):
            continue
        try:
            timestamp = float(entry.get("timestamp_s") or 0.0)
        except (TypeError, ValueError):
            timestamp = 0.0
        if latest is None or timestamp >= latest[0]:
            latest = (timestamp, value)
    return str(latest[1].get("source_grasp_id") or "") if latest else ""


def _parameters_target_xyz(parameters: JsonDict) -> list[float] | None:
    pose = parameters.get("target_pose") or parameters.get("pose") or parameters.get("eef_pose")
    if not isinstance(pose, dict):
        return None
    xyz = pose.get("xyz") or pose.get("translation_xyz") or pose.get("position")
    if not isinstance(xyz, list | tuple) or len(xyz) < 3:
        return None
    try:
        return [float(xyz[0]), float(xyz[1]), float(xyz[2])]
    except (TypeError, ValueError):
        return None


def _xyz_equal(left: list[float], right: Any, *, tolerance: float = 1e-8) -> bool:
    if not isinstance(right, list | tuple) or len(right) < 3:
        return False
    try:
        return all(abs(left[idx] - float(right[idx])) <= tolerance for idx in range(3))
    except (TypeError, ValueError):
        return False


def _select_memory(items: dict[str, JsonDict], key: str | None) -> dict[str, JsonDict]:
    if key is None:
        return dict(items)
    if key in items:
        return {key: items[key]}
    return {}


def _memory_fact_entry(value: JsonDict, *, source: str) -> JsonDict:
    return {"value": dict(value), "source": source, "timestamp_s": time.time()}


def _memory_fact_value(entry: JsonDict | None) -> JsonDict | None:
    if not isinstance(entry, dict):
        return None
    value = entry.get("value")
    return dict(value) if isinstance(value, dict) else None


def _grasp_fallback_attempts(policy: JsonDict | None) -> list[JsonDict]:
    if not isinstance(policy, dict):
        return []
    value = policy.get("fallback_attempts")
    if not isinstance(value, list):
        return []
    return [dict(attempt) for attempt in value if isinstance(attempt, dict)]


def _append_grasp_fallback_attempt(
    attempts: list[JsonDict],
    attempt: JsonDict,
) -> None:
    identity = (
        str(attempt.get("backend") or ""),
        str(attempt.get("source_rgb") or ""),
        str(attempt.get("outcome") or ""),
    )
    if any(
        (
            str(existing.get("backend") or ""),
            str(existing.get("source_rgb") or ""),
            str(existing.get("outcome") or ""),
        )
        == identity
        for existing in attempts
    ):
        return
    attempts.append(dict(attempt))


def _extract_action_artifacts(action: EnvAction) -> list[JsonDict]:
    artifacts: list[JsonDict] = []
    seen_paths: set[str] = set()
    command = action.command if isinstance(action.command, dict) else {}
    for call in command.get("tool_calls", []) or []:
        if not isinstance(call, dict):
            continue
        result = call.get("result")
        if not isinstance(result, dict):
            continue
        details = result.get("details")
        if not isinstance(details, dict):
            continue
        for artifact in details.get("artifacts", []) or []:
            if not isinstance(artifact, dict):
                continue
            path = artifact.get("path")
            if not isinstance(path, str) or not path:
                continue
            if path in seen_paths:
                continue
            seen_paths.add(path)
            normalized = dict(artifact)
            normalized.setdefault("tool", call.get("name"))
            artifacts.append(normalized)
        artifacts.extend(_extract_camera_packet_artifacts(call, details))
        artifacts.extend(_extract_depth_prior_artifacts(call, details))
        artifacts.extend(_extract_depth_enhancement_artifacts(call, details))
        artifacts.extend(_extract_sensor_safety_check_artifacts(call, details))
        artifacts.extend(_extract_grasp_candidate_artifacts(call, details))
        artifacts.extend(_extract_placement_candidate_artifacts(call, details))
        artifacts.extend(_extract_world_pose_artifacts(call, details))
    return artifacts


def _extract_camera_packet_artifacts(call: JsonDict, details: JsonDict) -> list[JsonDict]:
    if str(call.get("name") or "") != "observe":
        return []
    outputs = details.get("outputs")
    if not isinstance(outputs, dict):
        return []
    response = outputs.get("response")
    if not isinstance(response, dict):
        return []
    response_path = response.get("response_path")
    if not isinstance(response_path, str) or not response_path:
        return []
    try:
        payload = json.loads(Path(response_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    cameras = _find_camera_payloads(payload)
    artifacts: list[JsonDict] = []
    for index, camera in enumerate(cameras):
        packet = _camera_packet_from_payload(
            camera,
            index=index,
            response_path=response_path,
            tool=str(call.get("name") or "observe"),
        )
        if packet is not None:
            artifacts.append(packet)
    return artifacts


def _find_camera_payloads(payload: Any) -> list[JsonDict]:
    if isinstance(payload, dict):
        cameras = payload.get("cameras")
        if isinstance(cameras, list):
            return [camera for camera in cameras if isinstance(camera, dict)]
        for value in payload.values():
            found = _find_camera_payloads(value)
            if found:
                return found
    if isinstance(payload, list):
        for value in payload:
            found = _find_camera_payloads(value)
            if found:
                return found
    return []


def _camera_packet_from_payload(
    camera: JsonDict,
    *,
    index: int,
    response_path: str,
    tool: str,
) -> JsonDict | None:
    frame_id = str(camera.get("frame_id") or camera.get("camera") or f"camera_{index}")
    rgb_path = _string_field(camera, "rgb_path") or _string_field(camera, "image_path")
    depth_path = _string_field(camera, "depth_path")
    intrinsics = camera.get("intrinsics")
    extrinsics = camera.get("extrinsics")
    if not rgb_path and not depth_path:
        return None
    if not isinstance(intrinsics, dict):
        intrinsics = {}
    if not isinstance(extrinsics, dict):
        extrinsics = {}
    depth_scale, depth_scale_source = _camera_depth_scale(camera, intrinsics)
    normalized_intrinsics: JsonDict = dict(intrinsics)
    if depth_scale is not None:
        normalized_intrinsics["scale"] = depth_scale

    packet: JsonDict = {
        "type": "camera_packet",
        "kind": "rgbd_camera",
        "tool": tool,
        "index": frame_id,
        "frame_id": frame_id,
        "response_path": response_path,
        "rgb_path": rgb_path,
        "depth_path": depth_path,
        "intrinsics": normalized_intrinsics,
        "anygrasp_intrinsics": dict(normalized_intrinsics),
        "extrinsics": dict(extrinsics),
    }
    role = camera.get("role")
    if isinstance(role, str) and role:
        packet["role"] = role
    for field_name in (
        "width",
        "height",
        "depth_min",
        "depth_max",
        "depth_encoding",
    ):
        if field_name in camera:
            packet[field_name] = camera[field_name]
    if depth_scale is not None:
        packet["depth_scale"] = depth_scale
        packet["depth_scale_source"] = depth_scale_source
    camera_frame = extrinsics.get("camera_frame")
    if isinstance(camera_frame, str) and camera_frame:
        packet["camera_frame"] = camera_frame
    matrix_layout = extrinsics.get("matrix_layout")
    if isinstance(matrix_layout, str) and matrix_layout:
        packet["matrix_layout"] = matrix_layout
    return packet


def _camera_depth_scale(camera: JsonDict, intrinsics: JsonDict) -> tuple[float | None, str]:
    for key in ("scale", "depth_scale"):
        value = intrinsics.get(key)
        parsed = _positive_float(value)
        if parsed is not None:
            return parsed, f"intrinsics.{key}"
    for key in ("depth_scale", "scale"):
        value = camera.get(key)
        parsed = _positive_float(value)
        if parsed is not None:
            return parsed, f"camera.{key}"
    depth_path = _string_field(camera, "depth_path")
    if depth_path and depth_path.lower().endswith(".png"):
        return 1000.0, "default_png_millimeters"
    return None, "missing"


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _extract_depth_prior_artifacts(call: JsonDict, details: JsonDict) -> list[JsonDict]:
    if str(call.get("name") or "") != "estimate_depth_prior":
        return []
    outputs = details.get("outputs")
    if not isinstance(outputs, dict):
        return []
    prior_depth = _string_field(outputs, "prior_depth")
    if not prior_depth:
        return []
    camera_id = str(outputs.get("camera_id") or "camera")
    return [
        {
            "type": "depth_prior",
            "kind": "metric_depth_prior",
            "tool": "estimate_depth_prior",
            "index": camera_id,
            "camera_id": camera_id,
            "source_rgb": outputs.get("source_rgb"),
            "prior_depth": prior_depth,
            "prior_confidence": outputs.get("prior_confidence"),
            "prior_confidence_semantics": outputs.get(
                "prior_confidence_semantics"
            ),
            "backend": outputs.get("backend"),
            "model": outputs.get("model"),
            "request_ref": outputs.get("request_ref"),
            "raw_output_ref": outputs.get("raw_output_ref"),
            "next_tool_hint": outputs.get("next_tool_hint")
            or (
                "Call enhance_depth with the same rgb/depth/intrinsics and this "
                "prior_depth path."
            ),
        }
    ]


def _extract_depth_enhancement_artifacts(call: JsonDict, details: JsonDict) -> list[JsonDict]:
    if str(call.get("name") or "") != "enhance_depth":
        return []
    outputs = details.get("outputs")
    if not isinstance(outputs, dict):
        return []
    report_path = _string_field(outputs, "report_path")
    fused_depth_npy = _string_field(outputs, "fused_depth_npy")
    fused_depth_png = _string_field(outputs, "fused_depth_png")
    safety_depth_npy = _string_field(outputs, "safety_depth_npy")
    safety_depth_png = _string_field(outputs, "safety_depth_png")
    point_cloud_npz = _string_field(outputs, "point_cloud_npz")
    provenance_mask_png = _string_field(outputs, "provenance_mask_png")
    if not report_path or not fused_depth_npy:
        return []
    camera_id = str(outputs.get("camera_id") or "camera")
    return [
        {
            "type": "depth_enhancement",
            "kind": "rgbd_depth_enhancement",
            "tool": "enhance_depth",
            "index": camera_id,
            "camera_id": camera_id,
            "calibration_profile_id": outputs.get("calibration_profile_id"),
            "enabled": bool(outputs.get("enabled")),
            "reason": outputs.get("reason"),
            "source_rgb": outputs.get("source_rgb"),
            "source_depth": outputs.get("source_depth"),
            "source_sensor_confidence": outputs.get("source_sensor_confidence"),
            "source_rgb_sha256": outputs.get("source_rgb_sha256"),
            "source_depth_sha256": outputs.get("source_depth_sha256"),
            "intrinsics": outputs.get("intrinsics") if isinstance(outputs.get("intrinsics"), dict) else {},
            "candidate_intrinsics": (
                outputs.get("candidate_intrinsics")
                if isinstance(outputs.get("candidate_intrinsics"), dict)
                else {}
            ),
            "scene_epoch": outputs.get("scene_epoch"),
            "rgb_timestamp_s": outputs.get("rgb_timestamp_s"),
            "depth_timestamp_s": outputs.get("depth_timestamp_s"),
            "registration_status": outputs.get("registration_status"),
            "calibration_hash": outputs.get("calibration_hash"),
            "report_path": report_path,
            "fused_depth_npy": fused_depth_npy,
            "fused_depth_png": fused_depth_png,
            "candidate_depth_npy": outputs.get("candidate_depth_npy")
            or fused_depth_npy,
            "candidate_depth_png": outputs.get("candidate_depth_png")
            or fused_depth_png,
            "safety_depth_npy": safety_depth_npy,
            "safety_depth_png": safety_depth_png,
            "point_cloud_npz": point_cloud_npz,
            "candidate_point_cloud_npz": outputs.get(
                "candidate_point_cloud_npz"
            )
            or point_cloud_npz,
            "safety_point_cloud_npz": outputs.get("safety_point_cloud_npz"),
            "provenance_mask_png": provenance_mask_png,
            "alignment": outputs.get("alignment") if isinstance(outputs.get("alignment"), dict) else {},
            "quality": outputs.get("quality") if isinstance(outputs.get("quality"), dict) else {},
            "next_tool_hint": (
                "Use fused_depth_npy or a derived depth PNG for perception/grasp "
                "candidate generation only when quality.use_for_grasp_candidate_generation "
                "is true; never use mono-filled geometry for final collision clearance."
            ),
        }
    ]


def _extract_sensor_safety_check_artifacts(
    call: JsonDict,
    details: JsonDict,
) -> list[JsonDict]:
    if str(call.get("name") or "") != "obstacle_avoidance":
        return []
    parameters = details.get("parameters")
    path = parameters.get("path") if isinstance(parameters, dict) else None
    outputs = details.get("outputs")
    if (
        not isinstance(path, dict)
        or path.get("kind") != "enhanced_grasp_sensor_safety_check"
        or not isinstance(outputs, dict)
        or outputs.get("clear") is not True
    ):
        return []
    candidate_id = str(path.get("candidate_id") or "")
    safety_depth = str(path.get("safety_depth_png") or "")
    safety_cloud = str(path.get("safety_point_cloud_npz") or "")
    report_path = str(path.get("report_path") or "")
    if (
        not candidate_id
        or not Path(safety_depth).is_file()
        or not Path(safety_cloud).is_file()
        or not Path(report_path).is_file()
    ):
        return []
    return [
        {
            "type": "enhanced_grasp_sensor_safety_check",
            "kind": "sensor_safety_check",
            "tool": "obstacle_avoidance",
            "index": candidate_id,
            "candidate_id": candidate_id,
            "scene_epoch": path.get("scene_epoch"),
            "safety_depth_png": safety_depth,
            "safety_point_cloud_npz": safety_cloud,
            "report_path": report_path,
            "clear": True,
        }
    ]


def _extract_grasp_candidate_artifacts(call: JsonDict, details: JsonDict) -> list[JsonDict]:
    tool_name = str(call.get("name") or "")
    if tool_name not in {
        "grasp_pose_estimate",
        "anygrasp",
        "graspgenx",
        "contact_graspnet",
    }:
        return []
    candidates = details.get("grasp_candidates")
    source = details
    if not isinstance(candidates, list):
        outputs = details.get("outputs")
        if isinstance(outputs, dict):
            candidates = outputs.get("grasp_candidates")
            source = outputs
    if not isinstance(candidates, list) or not candidates:
        return []
    compact_candidates = [
        dict(candidate) for candidate in candidates[:5] if isinstance(candidate, dict)
    ]
    if not compact_candidates:
        return []
    grasp_source = source.get("source")
    if not isinstance(grasp_source, dict):
        grasp_source = {}
    return [
        {
            "type": "grasp_candidates",
            "kind": "grasp_candidates",
            "tool": tool_name,
            "index": "latest",
            "candidate_count": len(candidates),
            "best_grasp_candidate": compact_candidates[0],
            "grasp_candidates": compact_candidates,
            "source_rgb": source.get("source_rgb") or grasp_source.get("rgb"),
            "source_depth": source.get("source_depth") or grasp_source.get("depth"),
            "target_mask": source.get("target_mask") or grasp_source.get("object_mask"),
            "selected_grasp_source": source.get("source"),
            "source_tool": grasp_source.get("source_tool") or tool_name,
            "source_backend": grasp_source.get("source_backend")
            or source.get("selected_backend")
            or tool_name,
            "gripper_name": grasp_source.get("gripper_name"),
            "raw_output_ref": source.get("raw_output_ref"),
            "next_tool_hint": (
                "Call compile_grasp_seed with the active normalized candidate, "
                "matching camera extrinsics/frame id, and current scene_epoch; "
                "then follow host-generated grasp_execution stages."
            ),
        }
    ]


def _grasp_backend_label(value: str) -> str:
    return {
        "anygrasp": "AnyGrasp",
        "contact_graspnet": "Contact-GraspNet",
        "graspgenx": "GraspGenX",
        "grasp_pose_estimate": "grasp estimator",
    }.get(value, value or "grasp estimator")


def _extract_placement_candidate_artifacts(
    call: JsonDict,
    details: JsonDict,
) -> list[JsonDict]:
    if str(call.get("name") or "") != "anyplace":
        return []
    candidates = details.get("placement_candidates")
    source = details
    if not isinstance(candidates, list):
        outputs = details.get("outputs")
        if isinstance(outputs, dict):
            candidates = outputs.get("placement_candidates")
            source = outputs
    if not isinstance(candidates, list) or not candidates:
        return []
    compact_candidates = [
        dict(candidate) for candidate in candidates[:5] if isinstance(candidate, dict)
    ]
    if not compact_candidates:
        return []
    return [
        {
            "type": "placement_candidates",
            "kind": "placement_candidates",
            "tool": "anyplace",
            "index": "latest",
            "candidate_count": len(candidates),
            "selected_grasp_id": source.get("selected_grasp_id"),
            "placement_candidates": compact_candidates,
            "source": source.get("source"),
            "raw_output_ref": source.get("raw_output_ref"),
            "next_tool_hint": (
                "After pickup is visually verified, choose one complete "
                "placement_candidates[i].place_grasp_pose and transform it with "
                "camera_pose_to_world using the matching pre-grasp camera extrinsics."
            ),
        }
    ]


def _extract_world_pose_artifacts(call: JsonDict, details: JsonDict) -> list[JsonDict]:
    if str(call.get("name") or "") != "camera_pose_to_world":
        return []
    outputs = details.get("outputs")
    if not isinstance(outputs, dict):
        return []
    world_pose = outputs.get("world_pose")
    if not isinstance(world_pose, dict):
        return []
    translation = world_pose.get("translation_xyz") or outputs.get("translation_xyz")
    if not isinstance(translation, list) or len(translation) != 3:
        return []
    is_placement_reference = str(world_pose.get("id") or "").startswith("place_grasp_") and str(
        world_pose.get("source_grasp_id") or ""
    ).startswith("grasp_")
    next_tool_hint = (
        "Treat world_pose as the low AnyPlace release reference, not a one-step "
        "carry target. Preserve the current EEF orientation, first move to a "
        "pre-place hover at the same world X/Y and at least 0.10 m above its Z, "
        "verify attachment from a fresh image, then descend vertically before release."
        if is_placement_reference
        else (
            "Pass the complete world_pose to move_to.target_pose without changing "
            "its world-frame translation or orientation."
        )
    )
    return [
        {
            "type": "world_pose",
            "kind": "world_pose",
            "tool": "camera_pose_to_world",
            "index": "latest",
            "frame": world_pose.get("frame") or outputs.get("frame") or "world",
            "camera_frame_id": outputs.get("camera_frame_id"),
            "world_pose": dict(world_pose),
            "translation_xyz": list(translation),
            "rotation_matrix": world_pose.get("rotation_matrix") or outputs.get("rotation_matrix"),
            "gripper_tip_position_xyz": world_pose.get("gripper_tip_position_xyz")
            or outputs.get("gripper_tip_position_xyz"),
            "source_grasp_id": world_pose.get("id"),
            "next_tool_hint": next_tool_hint,
        }
    ]


def _string_field(value: JsonDict, key: str) -> str:
    field = value.get(key)
    return field if isinstance(field, str) and field else ""


def _artifact_memory_key(artifact: JsonDict, *, fallback_index: int) -> str:
    tool = str(artifact.get("tool") or "tool")
    artifact_type = str(artifact.get("type") or artifact.get("kind") or "artifact")
    index = str(artifact.get("index") or "")
    if not index:
        path = artifact.get("path")
        index = Path(str(path)).stem if path else str(fallback_index)
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", f"{tool}:{artifact_type}:{index}").strip("._")
    return safe or f"tool_artifact_{fallback_index}"


def summarize_memory_artifact(artifact: JsonDict) -> JsonDict:
    """Return a compact artifact summary for planner context."""

    value = artifact.get("value", {})
    summary: JsonDict = {
        "source": artifact.get("source", ""),
        "timestamp_s": artifact.get("timestamp_s"),
    }
    if isinstance(value, dict):
        summary["keys"] = sorted(str(key) for key in value)
        for field in (
            "id",
            "type",
            "kind",
            "index",
            "tool",
            "content",
            "path",
            "grep_hint",
            "chars",
            "response_path",
            "response_chars",
            "response_omitted",
            "image_root",
            "image_count",
            "latest_image_path",
            "paths",
            "env_id",
            "handle",
            "session_id",
            "mcp_server_url",
            "dashboard_url",
            "frame_id",
            "role",
            "rgb_path",
            "depth_path",
            "width",
            "height",
            "depth_min",
            "depth_max",
            "depth_scale",
            "depth_scale_source",
            "camera_frame_id",
            "camera_frame",
            "matrix_layout",
            "intrinsics",
            "anygrasp_intrinsics",
            "extrinsics",
            "candidate_count",
            "best_grasp_candidate",
            "grasp_candidates",
            "selected_grasp_id",
            "placement_candidates",
            "source_rgb",
            "source_depth",
            "source_sensor_confidence",
            "target_mask",
            "selected_grasp_source",
            "source_tool",
            "gripper_name",
            "raw_output_ref",
            "prior_depth",
            "prior_confidence",
            "backend",
            "model",
            "request_ref",
            "camera_id",
            "calibration_profile_id",
            "enabled",
            "reason",
            "report_path",
            "fused_depth_npy",
            "fused_depth_png",
            "point_cloud_npz",
            "provenance_mask_png",
            "alignment",
            "quality",
            "frame",
            "world_pose",
            "translation_xyz",
            "rotation_matrix",
            "gripper_tip_position_xyz",
            "source_grasp_id",
            "next_tool_hint",
        ):
            if field in value:
                structured_fields = {
                    "best_grasp_candidate",
                    "grasp_candidates",
                    "placement_candidates",
                    "selected_grasp_source",
                    "world_pose",
                    "extrinsics",
                    "rotation_matrix",
                    "intrinsics",
                    "anygrasp_intrinsics",
                    "alignment",
                    "quality",
                }
                max_depth = 4 if field in structured_fields else 2
                max_items = 16 if field in structured_fields else 8
                summary[field] = _compact_value(
                    value[field],
                    max_depth=max_depth,
                    max_items=max_items,
                )
        image_paths = _extract_artifact_image_paths(value)
        if image_paths:
            summary["image_paths"] = image_paths
    else:
        summary["type"] = type(value).__name__
    return summary


def _extract_artifact_image_paths(value: JsonDict, *, limit: int = 20) -> list[str]:
    paths: list[str] = []
    for field_name in ("images", "image_artifacts"):
        images = value.get(field_name)
        if not isinstance(images, list):
            continue
        for image in images:
            if not isinstance(image, dict):
                continue
            for key in ("path", "rgb_path", "depth_path", "image_path"):
                path = image.get(key)
                if isinstance(path, str) and path and path not in paths:
                    paths.append(path)
                if len(paths) >= limit:
                    return paths
    return paths


def summarize_event_payload(payload: JsonDict) -> JsonDict:
    """Return a bounded event payload summary for planner context.

    Session traces may keep rich action and tool metadata for debugging, but the
    planner should not receive complete historical commands. Full payloads can
    contain prior planner contexts, image payloads, or tool outputs, which then
    recursively inflate later prompts.
    """

    if not isinstance(payload, dict):
        return {"type": type(payload).__name__}

    summary: JsonDict = {"keys": sorted(str(key) for key in payload)}
    for key in (
        "task",
        "session_id",
        "source",
        "environment",
        "max_turns",
        "turn_index",
        "question",
        "answer",
    ):
        if key in payload:
            summary[key] = _compact_value(payload[key])

    metadata = payload.get("metadata")
    if isinstance(metadata, dict):
        summary["metadata"] = _compact_metadata(metadata)

    observation = payload.get("observation")
    if isinstance(observation, dict):
        summary["observation"] = _compact_observation_payload(observation)

    step_result = payload.get("step_result")
    if isinstance(step_result, dict):
        summary["step_result"] = _compact_step_result(step_result)

    action = payload.get("action")
    if isinstance(action, dict):
        summary["action"] = _compact_action_payload(action)

    command = payload.get("command")
    if isinstance(command, dict):
        summary["command"] = _compact_command_payload(command)

    request = payload.get("request")
    if isinstance(request, dict):
        summary["request"] = _compact_request_payload(request)

    if "interfaces" in payload and isinstance(payload["interfaces"], list):
        summary["interfaces"] = [
            _compact_named_item(item)
            for item in payload["interfaces"][:8]
            if isinstance(item, dict)
        ]
        summary["interface_count"] = len(payload["interfaces"])
    if "tools" in payload and isinstance(payload["tools"], list):
        summary["tools"] = list(payload["tools"][:32])
        summary["tool_count"] = len(payload["tools"])
    if "skills" in payload and isinstance(payload["skills"], list):
        summary["skills"] = list(payload["skills"][:16])
        summary["skill_count"] = len(payload["skills"])

    return summary


def _compact_step_result(step_result: JsonDict) -> JsonDict:
    return {
        "reward": step_result.get("reward"),
        "terminated": step_result.get("terminated"),
        "truncated": step_result.get("truncated"),
        "observation": _compact_observation_payload(step_result.get("observation"))
        if isinstance(step_result.get("observation"), dict)
        else None,
        "info": _compact_metadata(step_result.get("info") or {})
        if isinstance(step_result.get("info"), dict)
        else {},
    }


def _compact_observation_payload(observation: JsonDict) -> JsonDict:
    robot = observation.get("robot")
    if not isinstance(robot, dict):
        robot = {}
    objects = observation.get("objects")
    if not isinstance(objects, list):
        objects = []
    return {
        "task": _compact_value(observation.get("task")),
        "camera_ids": list(observation.get("camera_ids") or []),
        "num_cameras": observation.get("num_cameras")
        if "num_cameras" in observation
        else len(observation.get("camera_ids") or []),
        "object_count": len(objects),
        "objects": [_compact_value(obj) for obj in objects[:5]],
        "robot": {
            "end_effector_pose": _compact_value(robot.get("end_effector_pose")),
            "gripper_state": _compact_value(robot.get("gripper_state")),
            "base_pose": _compact_value(robot.get("base_pose")),
        },
        "metadata": _compact_metadata(observation.get("metadata") or {})
        if isinstance(observation.get("metadata"), dict)
        else {},
    }


def _compact_action_payload(action: JsonDict) -> JsonDict:
    return {
        "action_type": action.get("action_type"),
        "request_kind": action.get("request_kind"),
        "request_name": action.get("request_name"),
        "status": action.get("status"),
        "tool_calls": [
            _compact_tool_call(call)
            for call in (action.get("tool_calls") or [])[:8]
            if isinstance(call, dict)
        ],
        "metadata": _compact_metadata(action.get("metadata") or {})
        if isinstance(action.get("metadata"), dict)
        else {},
    }


def _compact_command_payload(command: JsonDict) -> JsonDict:
    return {
        "status": command.get("status"),
        "schema_version": command.get("schema_version"),
        "request": _compact_request_payload(command.get("request") or {})
        if isinstance(command.get("request"), dict)
        else {},
        "tool_calls": [
            _compact_tool_call(call)
            for call in (command.get("tool_calls") or [])[:8]
            if isinstance(call, dict)
        ],
        "metadata": _compact_metadata(command.get("metadata") or {})
        if isinstance(command.get("metadata"), dict)
        else {},
    }


def _compact_tool_call(call: JsonDict) -> JsonDict:
    result = call.get("result")
    compact_result: JsonDict | None = None
    if isinstance(result, dict):
        compact_result = {
            "success": result.get("success"),
            "content": _compact_value(result.get("content")),
        }
        details = result.get("details")
        if isinstance(details, dict):
            compact_details = _compact_tool_result_details(details)
            if compact_details:
                compact_result["details"] = compact_details
    return {
        "name": call.get("name"),
        "status": call.get("status"),
        "reason": _compact_value(call.get("reason")),
        "result": compact_result,
    }


def _compact_tool_result_details(details: JsonDict) -> JsonDict:
    compact: JsonDict = {}
    for key in (
        "candidate_count",
        "best_grasp_candidate",
        "grasp_candidates",
        "ranking",
        "result_id",
        "selection_required",
        "selected_detection",
        "selection_bundle",
        "source_rgb",
        "source_depth",
        "target_mask",
        "raw_output_ref",
        "frame",
        "camera_frame_id",
        "world_pose",
        "translation_xyz",
        "rotation_matrix",
        "gripper_tip_position_xyz",
        "observation_summary",
        "motion_summary",
    ):
        if key in details:
            structured = {
                "best_grasp_candidate",
                "grasp_candidates",
                "world_pose",
                "rotation_matrix",
                "selected_detection",
                "selection_bundle",
                "observation_summary",
                "motion_summary",
            }
            max_depth = 5 if key in structured else 2
            max_items = 32 if key in {"observation_summary", "motion_summary"} else 16
            compact[key] = _compact_value(
                details[key],
                max_depth=max_depth,
                max_items=max_items,
            )
    outputs = details.get("outputs")
    if isinstance(outputs, dict):
        useful_outputs: JsonDict = {}
        for key in (
            "result",
            "detection_count",
            "detections",
            "candidate_count",
            "best_grasp_candidate",
            "grasp_candidates",
            "ranking",
            "result_id",
            "selection_required",
            "selected_detection",
            "selection_bundle",
            "source_rgb",
            "source_depth",
            "target_mask",
            "frame",
            "camera_frame_id",
            "world_pose",
            "translation_xyz",
            "rotation_matrix",
            "gripper_tip_position_xyz",
            "observation_summary",
            "motion_summary",
            "response",
            "mcp",
            "schema_version",
            "query",
            "answer",
            "answer_truncated",
            "result_count",
            "results",
            "search_call_count",
            "provider_role",
            "provider",
            "model",
            "url",
            "content_type",
            "title",
            "text",
            "truncated",
            "returned_char_count",
            "source_byte_count",
            "untrusted_external_content",
        ):
            if key in outputs:
                if key in {"answer", "text"}:
                    useful_outputs[key] = _compact_web_text(outputs[key])
                    continue
                structured = {
                    "best_grasp_candidate",
                    "grasp_candidates",
                    "world_pose",
                    "rotation_matrix",
                    "selected_detection",
                    "selection_bundle",
                    "observation_summary",
                    "motion_summary",
                    "results",
                }
                max_depth = 5 if key in structured else 2
                max_items = 32 if key in {"observation_summary", "motion_summary"} else 16
                useful_outputs[key] = _compact_value(
                    outputs[key],
                    max_depth=max_depth,
                    max_items=max_items,
                )
        if useful_outputs:
            compact["outputs"] = useful_outputs
    state_delta = details.get("state_delta")
    if isinstance(state_delta, dict) and state_delta:
        compact["state_delta"] = _compact_value(
            state_delta,
            max_depth=5,
            max_items=32,
        )
    artifacts = details.get("artifacts")
    if isinstance(artifacts, list):
        compact_artifacts = []
        for artifact in artifacts[:8]:
            if not isinstance(artifact, dict):
                continue
            compact_artifacts.append(
                {
                    key: _compact_value(artifact[key], max_depth=1)
                    for key in (
                        "type",
                        "kind",
                        "tool",
                        "index",
                        "label",
                        "path",
                        "mask_ref",
                        "overlay_ref",
                        "crop_ref",
                        "frame_id",
                        "rgb_path",
                        "depth_path",
                    )
                    if key in artifact
                }
            )
        if compact_artifacts:
            compact["artifacts"] = compact_artifacts
    diagnostics = details.get("diagnostics")
    if isinstance(diagnostics, list) and diagnostics:
        compact["diagnostics"] = _compact_value(diagnostics, max_depth=2)
    return compact


def _compact_request_payload(request: JsonDict) -> JsonDict:
    return {
        "kind": request.get("kind"),
        "name": request.get("name"),
        "parameters": _compact_value(request.get("parameters")),
        "reasoning": _compact_value(request.get("reasoning")),
    }


def _compact_metadata(metadata: JsonDict) -> JsonDict:
    compact: JsonDict = {}
    for key, value in metadata.items():
        if key in {"planner_metadata", "tool_context", "raw_backend_payload"}:
            compact[key] = "<omitted>"
        elif key == "previous_action":
            compact[key] = _compact_previous_action(value)
        elif key in {"observation", "raw_payload"}:
            compact[key] = _compact_value(value, max_depth=1)
        else:
            compact[key] = _compact_value(value)
    return compact


def _compact_previous_action(value: Any) -> Any:
    if not isinstance(value, dict):
        return _compact_value(value, max_depth=1)
    if "command" in value and isinstance(value.get("command"), dict):
        return {
            "action_type": value.get("action_type"),
            "command": _compact_command_payload(value["command"]),
            "metadata": _compact_metadata(value.get("metadata") or {})
            if isinstance(value.get("metadata"), dict)
            else {},
        }
    if "tool_calls" in value or "request_name" in value or "request_kind" in value:
        return _compact_action_payload(value)
    return _compact_value(value, max_depth=1)


def _compact_named_item(item: JsonDict) -> JsonDict:
    return {
        "name": item.get("name"),
        "kind": item.get("kind"),
        "implemented": item.get("implemented"),
    }


def _compact_value(value: Any, *, max_depth: int = 2, max_items: int = 8) -> Any:
    if max_depth <= 0:
        if isinstance(value, dict):
            return {"type": "dict", "keys": sorted(str(key) for key in value)[:max_items]}
        if isinstance(value, list):
            return {"type": "list", "count": len(value)}
    if isinstance(value, str):
        return value if len(value) <= 300 else value[:300] + "...[truncated]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, dict):
        compact: JsonDict = {}
        for idx, key in enumerate(sorted(value)):
            if idx >= max_items:
                compact["..."] = f"{len(value) - max_items} more keys"
                break
            if _looks_like_inline_blob_key(str(key)):
                compact[str(key)] = "<omitted>"
            else:
                compact[str(key)] = _compact_value(
                    value[key],
                    max_depth=max_depth - 1,
                    max_items=max_items,
                )
        return compact
    if isinstance(value, (list, tuple)):
        compact_list = [
            _compact_value(item, max_depth=max_depth - 1, max_items=max_items)
            for item in list(value)[:max_items]
        ]
        if len(value) > max_items:
            compact_list.append(f"... {len(value) - max_items} more items")
        return compact_list
    return str(type(value).__name__)


def _compact_web_text(value: Any, *, max_chars: int = 4000) -> Any:
    if not isinstance(value, str):
        return _compact_value(value)
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + "...[truncated]"


def _looks_like_inline_blob_key(key: str) -> bool:
    lowered = key.lower()
    return "base64" in lowered or lowered in {
        "rgb",
        "depth",
        "image",
        "pixels",
        "array",
        "raw_payload",
    }


def summarize_observation(observation: EnvObservation) -> JsonDict:
    """Create a compact, JSON-friendly observation summary for memory."""

    return {
        "task": observation.task,
        "camera_ids": [camera.frame_id for camera in observation.cameras],
        "num_cameras": len(observation.cameras),
        "robot": {
            "joint_positions": observation.robot.joint_positions,
            "joint_velocities": observation.robot.joint_velocities,
            "end_effector_pose": observation.robot.end_effector_pose,
            "gripper_state": observation.robot.gripper_state,
            "base_pose": observation.robot.base_pose,
            "metadata": _compact_metadata(observation.robot.metadata),
        },
        "objects": [_compact_value(obj) for obj in observation.objects[:8]],
        "object_count": len(observation.objects),
        "metadata": _compact_metadata(observation.metadata),
    }
