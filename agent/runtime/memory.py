"""Session memory for lightweight embodied agents."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from adapter.protocol import EnvAction, EnvObservation, JsonDict


PENDING_SAM3_SELECTION_KEY = "pending_sam3_selection"
SELECTED_SAM3_DETECTION_KEY = "selected_sam3_detection"
ANYGRASP_CANDIDATE_POLICY_KEY = "anygrasp_candidate_policy"


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
        self.metadata: JsonDict = {}
        self.events: list[MemoryEvent] = []
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
        self.metadata = dict(metadata or {})
        self.events.clear()
        self.facts = boot_facts
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
        self.record(
            "session_resumed",
            {
                "session_id": session_id,
                "task": self.task,
                "loaded_event_count": len(self.events),
            },
        )

    def record(self, event_type: str, payload: JsonDict | None = None) -> MemoryEvent:
        event = MemoryEvent(event_type=event_type, payload=dict(payload or {}))
        self.events.append(event)
        if self.store is not None:
            self.store.append_event(event)
        return event

    def add_observation(self, observation: EnvObservation) -> None:
        self.record("observation", summarize_observation(observation))

    def add_action(self, action: EnvAction) -> None:
        self._capture_sam3_selection_state(action)
        self._capture_anygrasp_candidate_policy(action)
        captured_artifacts = _extract_action_artifacts(action)
        for artifact in captured_artifacts:
            key = _artifact_memory_key(artifact, fallback_index=len(self.artifacts))
            self.artifacts[key] = {
                "value": artifact,
                "source": "tool_result",
                "timestamp_s": time.time(),
            }
        candidate_advanced = self._advance_anygrasp_candidate_after_rejection(action)
        candidate_accepted = (
            False if candidate_advanced else self._accept_anygrasp_candidate_after_motion(action)
        )
        if captured_artifacts or candidate_advanced or candidate_accepted:
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

    def add_external_event(self, event: JsonDict) -> None:
        event_type = str(event.get("type", "external"))
        self.record(event_type, event)

    def save_fact(self, key: str, value: JsonDict, *, source: str = "") -> None:
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
                self.skill_notes
                if key is None
                else {key: self.skill_notes.get(key, [])}
            ),
            "compact_summary": self.compact_summary,
        }

    def pending_sam3_selection(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(PENDING_SAM3_SELECTION_KEY))

    def selected_sam3_detection(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(SELECTED_SAM3_DETECTION_KEY))

    def anygrasp_candidate_policy(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(ANYGRASP_CANDIDATE_POLICY_KEY))

    def grasp_candidate_gate_error(
        self,
        *,
        tool_name: str,
        parameters: JsonDict,
    ) -> str | None:
        policy = self.anygrasp_candidate_policy()
        if policy is None:
            return None
        if tool_name not in {
            "camera_pose_to_world",
            "move_to",
            "follow_eef_trajectory",
            "ik_preview_check",
            "obstacle_avoidance",
        }:
            return None
        status = str(policy.get("status") or "")
        if status == "accepted":
            return None
        active = policy.get("active_candidate")
        if status == "exhausted" or not isinstance(active, dict):
            return (
                "All AnyGrasp candidates were rejected. Observe and rerun AnyGrasp "
                "instead of reusing an exhausted pose."
            )
        active_id = str(active.get("id") or "")
        supplied_id = _parameters_grasp_candidate_id(parameters)
        if tool_name == "camera_pose_to_world":
            if not supplied_id:
                return (
                    "camera_pose_to_world must receive the active AnyGrasp candidate "
                    f"{active_id!r}, including its id."
                )
            if supplied_id != active_id:
                return (
                    "Greedy AnyGrasp policy requires the current active candidate "
                    f"{active_id!r}; later candidates are available only after a "
                    "candidate-linked safety or motion rejection."
                )
            return None
        if supplied_id and supplied_id != active_id:
            return (
                f"Tool {tool_name!r} references AnyGrasp candidate {supplied_id!r}, "
                f"but the current active candidate is {active_id!r}."
            )
        return None

    def resolve_sam3_selection(
        self,
        *,
        result_id: str,
        detection_id: str,
        selection_source: str,
        confidence: float | None = None,
        reason: str = "",
    ) -> JsonDict:
        pending = self.pending_sam3_selection()
        if pending is None:
            raise ValueError("No SAM3 detection selection is pending.")
        expected_result_id = str(pending.get("result_id") or "")
        if not result_id or result_id != expected_result_id:
            raise ValueError(
                "select_sam3_detection requires the exact pending sam3_result_id."
            )
        candidates = pending.get("candidates")
        if not isinstance(candidates, list):
            candidates = []
        selected = next(
            (
                dict(candidate)
                for candidate in candidates
                if isinstance(candidate, dict)
                and str(candidate.get("id") or "") == detection_id
            ),
            None,
        )
        if selected is None:
            raise ValueError("detection_id does not belong to the pending SAM3 result.")
        selected.update(
            {
                "result_id": result_id,
                "selection_source": selection_source or "main_agent_vlm",
                "selection_confidence": confidence,
                "selection_reason": reason,
                "selected_at_s": time.time(),
            }
        )
        self.facts.pop(PENDING_SAM3_SELECTION_KEY, None)
        self.facts[SELECTED_SAM3_DETECTION_KEY] = _memory_fact_entry(
            selected,
            source="select_sam3_detection",
        )
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

    def detection_selection_gate_error(
        self,
        *,
        tool_name: str,
        parameters: JsonDict,
        world_mutating: bool = False,
    ) -> str | None:
        pending = self.pending_sam3_selection()
        mode = str(parameters.get("mode") or "targeted").strip().lower()
        if pending is not None and tool_name == "anygrasp" and mode != "scene":
            return (
                "Targeted AnyGrasp is blocked until the pending SAM3 candidates are "
                "resolved with select_sam3_detection."
            )
        if pending is not None and world_mutating:
            return (
                "World-mutating tools are blocked while a SAM3 detection selection "
                "obligation is pending."
            )
        if tool_name != "anygrasp":
            return None
        if mode == "scene":
            return None
        selected = self.selected_sam3_detection()
        if selected is None:
            return None
        expected_mask = str(selected.get("mask_ref") or "")
        supplied_mask = str(parameters.get("target_mask") or "")
        if expected_mask and supplied_mask != expected_mask:
            return (
                "Targeted AnyGrasp must use the mask_ref from the recorded "
                "select_sam3_detection result."
            )
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
            base = {
                "result_id": result_id,
                "target_prompt": outputs.get("prompt") or parameters.get("prompt"),
                "source_image": outputs.get("source_image") or parameters.get("image"),
                "ranking": outputs.get("ranking") or "score_descending",
                "candidate_count": len(candidates),
                "candidates": candidates,
                "selection_bundle": dict(selection_bundle),
            }
            self.facts.pop(PENDING_SAM3_SELECTION_KEY, None)
            self.facts.pop(SELECTED_SAM3_DETECTION_KEY, None)
            if candidates:
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
                self.record("sam3_no_detection", {"result_id": result_id})
            self._save_working_memory()

    def _capture_anygrasp_candidate_policy(self, action: EnvAction) -> None:
        command = action.command if isinstance(action.command, dict) else {}
        for call in command.get("tool_calls", []) or []:
            if not isinstance(call, dict) or str(call.get("name") or "") != "anygrasp":
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
            candidates_value = outputs.get("grasp_candidates")
            if not isinstance(candidates_value, list):
                continue
            candidates = [
                dict(candidate)
                for candidate in candidates_value
                if isinstance(candidate, dict) and str(candidate.get("id") or "")
            ]
            if not candidates:
                continue
            candidates.sort(key=_grasp_candidate_sort_key)
            for rank, candidate in enumerate(candidates):
                candidate["rank"] = rank
            result_id = str(outputs.get("result_id") or "")
            if not result_id:
                result_id = f"anygrasp-{int(time.time() * 1000)}"
            policy = {
                "result_id": result_id,
                "ranking": "score_descending",
                "status": "active",
                "candidate_count": len(candidates),
                "active_rank": 0,
                "active_candidate": candidates[0],
                "remaining_candidate_ids": [
                    str(candidate.get("id")) for candidate in candidates[1:]
                ],
                "candidates": candidates,
                "rejected_candidates": [],
                "activated_at_s": time.time(),
            }
            self.facts[ANYGRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
                policy,
                source="anygrasp",
            )
            self.record(
                "anygrasp_candidate_activated",
                {
                    "result_id": result_id,
                    "candidate_id": candidates[0].get("id"),
                    "rank": 0,
                    "score": candidates[0].get("score"),
                    "candidate_count": len(candidates),
                },
            )
            self._save_working_memory()

    def _advance_anygrasp_candidate_after_rejection(self, action: EnvAction) -> bool:
        policy = self.anygrasp_candidate_policy()
        if policy is None or str(policy.get("status") or "") != "active":
            return False
        active = policy.get("active_candidate")
        if not isinstance(active, dict):
            return False
        rejection = _candidate_linked_rejection(
            action,
            active_candidate_id=str(active.get("id") or ""),
            artifacts=self.artifacts,
        )
        if rejection is None:
            return False

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
                "rejected_at_s": time.time(),
            }
        )
        next_rank = current_rank + 1
        next_candidate = (
            dict(candidates[next_rank])
            if next_rank < len(candidates) and isinstance(candidates[next_rank], dict)
            else None
        )
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
                "last_rejection": dict(rejection),
            }
        )
        self.facts[ANYGRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
            policy,
            source="anygrasp_greedy_fallback",
        )
        self.record(
            "anygrasp_candidate_rejected",
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
                "anygrasp_candidate_activated",
                {
                    "result_id": policy.get("result_id"),
                    "candidate_id": next_candidate.get("id"),
                    "rank": next_rank,
                    "score": next_candidate.get("score"),
                    "activation_source": "previous_candidate_rejected",
                },
            )
        return True

    def _accept_anygrasp_candidate_after_motion(self, action: EnvAction) -> bool:
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
        self.facts[ANYGRASP_CANDIDATE_POLICY_KEY] = _memory_fact_entry(
            policy,
            source="anygrasp_motion_accepted",
        )
        self.record(
            "anygrasp_candidate_accepted",
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
        parts = [
            f"task={self.task}",
            f"facts={list(self.facts)}",
            f"artifacts={list(self.artifacts)}",
            f"skill_notes={list(self.skill_notes)}",
            "recent_events=" + ",".join(event.event_type for event in recent),
        ]
        self.compact_summary = "; ".join(parts)
        self.record("memory_compacted", {"summary": self.compact_summary})
        self._save_working_memory()
        return self.compact_summary

    def recent_events(self, limit: int = 8) -> list[MemoryEvent]:
        if limit <= 0:
            return []
        return self.events[-limit:]

    def planning_context(self, *, max_events: int = 8) -> JsonDict:
        """Return compact context suitable for a planner prompt or policy."""

        return {
            "session_id": self.session_id,
            "task": self.task,
            "metadata": self.metadata,
            "selection_obligation": self.pending_sam3_selection(),
            "selected_sam3_detection": self.selected_sam3_detection(),
            "grasp_candidate_policy": self.anygrasp_candidate_policy(),
            "working_memory": {
                "facts": {
                    key: value
                    for key, value in self.facts.items()
                    if key
                    not in {
                        PENDING_SAM3_SELECTION_KEY,
                        SELECTED_SAM3_DETECTION_KEY,
                        ANYGRASP_CANDIDATE_POLICY_KEY,
                    }
                },
                "artifacts": {
                    key: summarize_memory_artifact(value)
                    for key, value in self.artifacts.items()
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
    return ""


def _candidate_linked_rejection(
    action: EnvAction,
    *,
    active_candidate_id: str,
    artifacts: dict[str, JsonDict],
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
            if isinstance(call, dict)
            and _safety_call_rejects_candidate(call)
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

    if request_name not in {"move_to", "follow_eef_trajectory"}:
        return None
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
        for call in command.get("tool_calls", []) or []
    )


def _call_result_success(call: JsonDict) -> bool:
    result = call.get("result")
    return isinstance(result, dict) and bool(result.get("success"))


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


def _motion_call_rejects_candidate(call: JsonDict) -> bool:
    if _call_result_success(call):
        return False
    result = call.get("result")
    if not isinstance(result, dict):
        return False
    details = result.get("details")
    if not isinstance(details, dict):
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
                    str(outputs.get(key) or "")
                    for key in ("reason", "verdict", "diagnostic")
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
    pose = (
        parameters.get("target_pose")
        or parameters.get("pose")
        or parameters.get("eef_pose")
    )
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
        artifacts.extend(_extract_grasp_candidate_artifacts(call, details))
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
    for field in ("width", "height", "depth_min", "depth_max"):
        if field in camera:
            packet[field] = camera[field]
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


def _extract_grasp_candidate_artifacts(call: JsonDict, details: JsonDict) -> list[JsonDict]:
    if str(call.get("name") or "") != "anygrasp":
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
    return [
        {
            "type": "grasp_candidates",
            "kind": "grasp_candidates",
            "tool": "anygrasp",
            "index": "latest",
            "candidate_count": len(candidates),
            "best_grasp_candidate": compact_candidates[0],
            "grasp_candidates": compact_candidates,
            "source_rgb": source.get("source_rgb") or source.get("rgb"),
            "source_depth": source.get("source_depth") or source.get("depth"),
            "target_mask": source.get("target_mask"),
            "raw_output_ref": source.get("raw_output_ref"),
            "next_tool_hint": (
                "Call camera_pose_to_world with best_grasp_candidate as "
                "camera_pose and the matching camera_packet.extrinsics before "
                "calling move_to."
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
            "rotation_matrix": world_pose.get("rotation_matrix")
            or outputs.get("rotation_matrix"),
            "gripper_tip_position_xyz": world_pose.get("gripper_tip_position_xyz")
            or outputs.get("gripper_tip_position_xyz"),
            "source_grasp_id": world_pose.get("id"),
            "next_tool_hint": (
                "Pass the complete world_pose to move_to.target_pose without changing "
                "its world-frame translation or orientation."
            ),
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
            "source_rgb",
            "source_depth",
            "target_mask",
            "raw_output_ref",
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
                    "world_pose",
                    "extrinsics",
                    "rotation_matrix",
                    "intrinsics",
                    "anygrasp_intrinsics",
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
    for field in ("images", "image_artifacts"):
        images = value.get(field)
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
    for key in ("task", "session_id", "source", "environment", "max_turns", "turn_index"):
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
            _compact_named_item(item) for item in payload["interfaces"][:8] if isinstance(item, dict)
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
        ):
            if key in outputs:
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


def _looks_like_inline_blob_key(key: str) -> bool:
    lowered = key.lower()
    return (
        "base64" in lowered
        or lowered in {"rgb", "depth", "image", "pixels", "array", "raw_payload"}
    )


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
