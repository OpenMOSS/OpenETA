"""Session memory for lightweight embodied agents."""

from __future__ import annotations

import hashlib
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
from agent.runtime.structured_artifacts import (
    load_structured_tool_output,
    materialize_structured_tool_output,
)
from agent.runtime.depth_contract import target_depth_cutoff_factor
from agent.runtime.memory_migrations import (
    is_removed_task_policy_key,
    purge_removed_task_policy_facts,
)
from agent.runtime.observation_packets import (
    ObservationPacketResolutionError,
    build_observation_packet_entries,
    find_packet_id_for_path,
    find_packet_reference_for_path,
    packet_integrity_fingerprint,
    resolve_packet_source,
)
from agent.tools.grasp_geometry import (
    GraspGeometryError,
    assess_target_mask_quality,
    rebase_camera_direction,
    rebase_camera_grasp_candidate,
    world_up_direction_camera,
)


PENDING_SAM3_SELECTION_KEY = "pending_sam3_selection"
SELECTED_SAM3_DETECTION_KEY = "selected_sam3_detection"
SELECTED_SAM3_DETECTIONS_KEY = "selected_sam3_detections"
PENDING_REFERENCE_LOCALIZATION_KEY = "pending_reference_localization"
REFERENCE_LOCALIZATION_FAILURE_KEY = "reference_localization_failure"
TARGET_LOCALIZATION_BUDGET_KEY = "target_localization_budget"
TARGET_ASSET_REFERENCE_KEY = "target_asset_reference"
TARGET_IDENTITY_ANCHOR_KEY = "target_identity_anchor"
SAM3_NO_DETECTION_KEY = "sam3_no_detection"
SAM3_NO_DETECTIONS_KEY = "sam3_no_detections"
ARTICULATED_ATTACHMENT_PROBE_KEY = "articulated_attachment_probe"
GRIPPER_COMMAND_STATE_KEY = "gripper_command_state"
ATTACHMENT_EVIDENCE_KEY = "attachment_evidence"
MOTION_RECONCILIATION_KEY = "motion_reconciliation"
IK_PREVIEW_RECEIPTS_KEY = "ik_preview_receipts"
CONTROLLER_CAPABILITIES_KEY = "controller_capabilities"
OBSERVED_OBJECT_SCENE_CHANGE_KEY = "observed_object_scene_change"
SCENE_EPOCH_KEY = "scene_epoch"
OBJECT_SCENE_EPOCH_KEY = "object_scene_epoch"
ROBOT_MOTION_EPOCH_KEY = "robot_motion_epoch"
GRASP_PROVENANCE_KEY = "grasp_provenance"
GRASP_INPUT_BUNDLES_KEY = "grasp_input_bundles"
WRIST_ALIGNMENT_BUNDLES_KEY = "wrist_alignment_bundles"
ANYPLACE_INPUT_BUNDLES_KEY = "anyplace_input_bundles"
PLACEMENT_WORLD_REFERENCE_KEY = "placement_world_reference"
GRASP_ADJUSTMENT_BUDGET_KEY = "grasp_adjustment_budget"
TRANSITION_LEDGER_KEY = "transition_ledger"
TOOL_HEALTH_KEY = "tool_health"
ACTIVE_ENVIRONMENT_TASK_KEY = "active_environment_task"
LATEST_COMPILED_CONTACT_EXECUTION_KEY = "latest_compiled_contact_execution"
LATEST_COMPILED_CLEARANCE_EXECUTION_KEY = "latest_compiled_clearance_execution"
# Closing is a finger-only action and the Agent receives fresh dual-view evidence
# after contact motion.  Keep an independent residual envelope, but do not make
# it tighter than the centimetre-scale execution accuracy of the supported
# controllers; an overly strict 5 mm veto prevented visually valid closes after
# otherwise safe contact reaches.
COMPILED_CONTACT_CLOSE_POSITION_TOLERANCE_M = 0.01
COMPILED_CONTACT_CLOSE_ORIENTATION_TOLERANCE_RAD = 0.30
# A clearance point is a geometric route anchor, not the final contact pose.  A
# controller receipt that misses only inside this envelope can still support a
# subsequent approach-corridor check; larger misses remain explicit failures.
COMPILED_CLEARANCE_POSITION_TOLERANCE_M = 0.01
COMPILED_CLEARANCE_ORIENTATION_TOLERANCE_RAD = 0.30
# A long contact move must enter along the estimator's approach axis.  This is
# a geometric safety/evidence invariant, not a grasp-stage requirement: the
# Agent remains free to choose any number and meaning of preceding waypoints.
COMPILED_CONTACT_NEAR_FIELD_RADIUS_M = 0.04
COMPILED_CONTACT_APPROACH_CORRIDOR_RADIUS_M = 0.025
COMPILED_CONTACT_APPROACH_MIN_COSINE = 0.94
# Entering the object while the wrist is still making a large rotation can let
# an open finger or palm push the target away even when the Cartesian approach
# corridor is correct.  This is a pose-envelope invariant, not a task stage:
# the Agent remains free to align at any safe waypoint or choose another grasp.
COMPILED_CONTACT_ENTRY_ORIENTATION_TOLERANCE_RAD = 0.30
GRASP_REFERENCE_POSITION_TOLERANCE_M = 0.05
GRASP_REFERENCE_ORIENTATION_TOLERANCE_DEG = 20.0
GRASP_ADJUSTMENT_STEP_LIMIT_M = 0.02
GRASP_ADJUSTMENT_CUMULATIVE_LIMIT_M = 0.10
GRASP_ADJUSTMENT_EPSILON_M = 1e-6
GRASP_GEOMETRY_FAMILIES = {
    "upright_can",
    "upright_bottle",
    "lying_bottle",
    "boxed_item",
    "bowl",
    "apple",
    "articulated_handle",
    "drawer_handle",
    "other",
    "unknown",
}
DEFAULT_SAM3_EVIDENCE_ROLE = "target_object"
SAM3_EVIDENCE_ROLES = frozenset({DEFAULT_SAM3_EVIDENCE_ROLE, "placement_region"})
INFRASTRUCTURE_FAILURE_CODES = frozenset(
    {
        "mcp_timeout",
        "mcp_connection_error",
        "mcp_transport_error",
        "mcp_call_failed",
        "backend_timeout",
        "backend_unavailable",
        "connection_error",
        "service_unavailable",
    }
)


def _paths_have_identical_content(left: object, right: object) -> bool:
    """Compare immutable local artifacts without trusting packet-id churn."""

    left_value = str(left or "").strip()
    right_value = str(right or "").strip()
    if not left_value or not right_value:
        return False
    try:
        left_path = Path(left_value).resolve(strict=True)
        right_path = Path(right_value).resolve(strict=True)
        if left_path == right_path:
            return True
        if left_path.stat().st_size != right_path.stat().st_size:
            return False
        return hashlib.sha256(left_path.read_bytes()).digest() == hashlib.sha256(
            right_path.read_bytes()
        ).digest()
    except OSError:
        return False


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

    def __init__(
        self,
        *,
        store: MemoryStore | None = None,
        artifact_root: str | Path | None = None,
    ) -> None:
        self.store = store
        self.artifact_root = Path(artifact_root).resolve() if artifact_root else None
        self.session_id: str | None = None
        self.task: str | None = None
        self.current_user_request: str = ""
        self.metadata: JsonDict = {}
        self.events: list[MemoryEvent] = []
        self.conversation = ConversationHistory()
        self.facts: dict[str, JsonDict] = {}
        self.agent_working_state: dict[str, JsonDict] = {}
        self.artifacts: dict[str, JsonDict] = {}
        self._observation_packets: dict[str, JsonDict] = {}
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
        boot_agent_state = dict(self.agent_working_state) if self.session_id is None else {}
        boot_artifacts = dict(self.artifacts) if self.session_id is None else {}
        self.session_id = session_id or str(uuid4())
        self.task = task
        self.current_user_request = ""
        self.metadata = dict(metadata or {})
        self.events.clear()
        self.conversation.clear()
        self.facts = boot_facts
        purge_removed_task_policy_facts(self.facts)
        self.agent_working_state = boot_agent_state
        purge_removed_task_policy_facts(self.agent_working_state)
        self.facts.setdefault(
            SCENE_EPOCH_KEY,
            _memory_fact_entry({"epoch": 0}, source="runtime"),
        )
        self.facts.setdefault(
            OBJECT_SCENE_EPOCH_KEY,
            _memory_fact_entry({"epoch": self.scene_epoch()}, source="runtime"),
        )
        self.facts.setdefault(
            ROBOT_MOTION_EPOCH_KEY,
            _memory_fact_entry({"epoch": 0}, source="runtime"),
        )
        self.artifacts = boot_artifacts
        self._observation_packets.clear()
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
        max_events: int | None = None,
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
        self.agent_working_state.clear()
        self.artifacts.clear()
        self._observation_packets.clear()
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
                self._index_observation_packets_from_payload(payload)
            records = self.store.load_conversation_records(session_id)
            if records:
                self.conversation.replay(records)
            else:
                self._reconstruct_legacy_conversation(
                    self.store.load_events(session_id, limit=None)
                )
                for item in self.conversation.items:
                    self._append_conversation_record(item_record(item))
        legacy_scene_epoch = self.scene_epoch()
        self.facts.setdefault(
            OBJECT_SCENE_EPOCH_KEY,
            _memory_fact_entry({"epoch": legacy_scene_epoch}, source="session_migration"),
        )
        self.facts.setdefault(
            ROBOT_MOTION_EPOCH_KEY,
            _memory_fact_entry({"epoch": legacy_scene_epoch}, source="session_migration"),
        )
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
        object_scene_updated = self._capture_observed_object_scene_change(observation)
        controller_capabilities_updated = self._capture_controller_capabilities(
            observation
        )
        summary = summarize_observation(observation)
        observation_index = self._next_observation_index()
        summary["observation_index"] = observation_index
        environment_step = observation.metadata.get("step_idx")
        if isinstance(environment_step, int) and not isinstance(environment_step, bool):
            summary["environment_step"] = environment_step
        summary["scene_epoch"] = self.scene_epoch()
        summary["object_scene_epoch"] = self.object_scene_epoch()
        summary["robot_motion_epoch"] = self.robot_motion_epoch()
        summary["visual_signature"] = _observation_visual_signature(observation)
        summary["runtime_camera_calibrations"] = [
            {
                "frame_id": camera.frame_id,
                "role": camera.role,
                "intrinsics": dict(camera.intrinsics),
                "extrinsics": dict(camera.extrinsics),
                "timestamp_s": camera.timestamp_s,
            }
            for camera in observation.cameras
        ]
        runtime_sources: list[JsonDict] = []
        visual_artifacts: list[JsonDict] = []
        seen_sources: set[tuple[str, str]] = set()
        cameras = {camera.frame_id: camera for camera in observation.cameras}
        for artifact in observation.metadata.get("image_artifacts", []):
            if (
                not isinstance(artifact, dict)
                or artifact.get("kind") not in {"rgb", "depth"}
                or not isinstance(artifact.get("path"), str)
            ):
                continue
            frame_id = str(artifact.get("frame_id") or "")
            camera = cameras.get(frame_id)
            visual_artifact: JsonDict = {
                "kind": str(artifact["kind"]),
                "frame_id": frame_id,
                "path": str(artifact["path"]),
            }
            role = str(
                artifact.get("role")
                or getattr(camera, "role", "")
                or ""
            )
            if role:
                visual_artifact["role"] = role
            timestamp_s = getattr(camera, "timestamp_s", None)
            if timestamp_s is not None:
                visual_artifact["timestamp_s"] = timestamp_s
            for field_name in ("packet_id", "width", "height", "format", "index"):
                if artifact.get(field_name) is not None:
                    visual_artifact[field_name] = artifact[field_name]
            visual_artifacts.append(visual_artifact)
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
        summary["visual_artifacts"] = visual_artifacts
        packet_entries = build_observation_packet_entries(
            observation,
            observation_index=observation_index,
            scene_epoch=self.scene_epoch(),
            object_scene_epoch=self.object_scene_epoch(),
            robot_motion_epoch=self.robot_motion_epoch(),
            compact_ids=True,
        )
        if packet_entries:
            packet_aliases = {
                str(entry.get("origin_packet_id") or entry.get("packet_id") or ""):
                str(entry.get("packet_id") or "")
                for entry in packet_entries
            }
            for artifact in visual_artifacts:
                raw_packet_id = str(artifact.get("packet_id") or "")
                if raw_packet_id in packet_aliases:
                    artifact["packet_id"] = packet_aliases[raw_packet_id]
            summary["observation_packets"] = packet_entries
            for entry in packet_entries:
                self._register_observation_packet(entry)
        self.record("observation", summary)
        reconciliation_updated = self._reconcile_unknown_motion(observation)
        if (
            reconciliation_updated
            or object_scene_updated
            or controller_capabilities_updated
        ):
            self._save_working_memory()

    def _capture_controller_capabilities(self, observation: EnvObservation) -> bool:
        create_env = observation.metadata.get("create_env")
        control_spec = (
            create_env.get("control_spec") if isinstance(create_env, dict) else None
        )
        controller = (
            control_spec.get("controller") if isinstance(control_spec, dict) else None
        )
        if not isinstance(controller, dict):
            return False
        collision_scope = str(controller.get("collision_scope") or "")
        value = {
            "schema_version": "openeta.controller_capabilities.v1",
            "controller_id": str(controller.get("controller_id") or ""),
            "goal_executor": str(controller.get("goal_executor") or ""),
            "collision_callback": controller.get("collision_callback") is True,
            "collision_scope": collision_scope,
            "motion_owns_trajectory_world_collision": (
                controller.get("collision_callback") is True
                and "per_step_pre_actuation_and_post_step" in collision_scope
            ),
            "control_spec": dict(control_spec),
        }
        existing = self.controller_capabilities()
        if existing == value:
            return False
        self.facts[CONTROLLER_CAPABILITIES_KEY] = _memory_fact_entry(
            value,
            source="create_env.control_spec",
        )
        self.record("controller_capabilities", dict(value))
        return True

    def _next_observation_index(self) -> int:
        for event in reversed(self.events):
            if event.event_type != "observation":
                continue
            value = event.payload.get("observation_index")
            if isinstance(value, int) and not isinstance(value, bool):
                return value + 1
        if self.store is not None and self.session_id is not None:
            try:
                rows = self.store.load_events(self.session_id, limit=None)
            except (OSError, ValueError, json.JSONDecodeError):
                rows = []
            for row in reversed(rows):
                if not isinstance(row, dict) or row.get("event_type") != "observation":
                    continue
                payload = row.get("payload")
                value = payload.get("observation_index") if isinstance(payload, dict) else None
                if isinstance(value, int) and not isinstance(value, bool):
                    return value + 1
        return sum(event.event_type == "observation" for event in self.events)

    def resolve_observation_packet(
        self,
        source_packet_id: str,
        camera_frame_id: str = "",
        *,
        require_files: bool = True,
    ) -> JsonDict:
        """Resolve a session-owned packet into aligned local camera inputs."""

        packet_id = str(source_packet_id or "").strip()
        if not packet_id:
            raise ObservationPacketResolutionError(
                "missing_source_packet_id",
                "source_packet_id is required.",
                details={"active_agent_session_id": self.session_id or ""},
            )
        entry = self._observation_packets.get(packet_id)
        if entry is None:
            self._load_observation_packet_index_from_store()
            entry = self._observation_packets.get(packet_id)
        if entry is None:
            raise ObservationPacketResolutionError(
                "unknown_source_packet_id",
                "source_packet_id does not exist in the active Agent session.",
                details={
                    "source_packet_id": packet_id,
                    "active_agent_session_id": self.session_id or "",
                    "recent_source_packets": self.recent_observation_packet_refs(),
                },
            )
        return resolve_packet_source(
            entry,
            camera_frame_id=str(camera_frame_id or ""),
            require_files=require_files,
        )

    def resolve_depth_enhancement_input(
        self,
        source_packet_id: str,
        camera_frame_id: str = "",
    ) -> JsonDict:
        """Resolve aligned RGB-D and the newest matching host-owned depth prior."""

        source = self.resolve_observation_packet(
            source_packet_id,
            camera_frame_id,
        )
        depth = str(source.get("depth") or "")
        intrinsics = source.get("intrinsics")
        if not depth:
            raise ObservationPacketResolutionError(
                "source_depth_missing",
                "The selected observation packet camera has no aligned depth artifact.",
                details={
                    "source_packet_id": source.get("packet_id"),
                    "camera_frame_id": source.get("frame_id"),
                },
            )
        if not isinstance(intrinsics, dict) or not intrinsics:
            raise ObservationPacketResolutionError(
                "source_intrinsics_missing",
                "The selected observation packet camera has no calibrated intrinsics.",
                details={
                    "source_packet_id": source.get("packet_id"),
                    "camera_frame_id": source.get("frame_id"),
                },
            )

        source_rgb = str(source.get("rgb") or "")
        matching_prior: JsonDict | None = None
        for entry in reversed(list(self.artifacts.values())):
            value = entry.get("value") if isinstance(entry, dict) else None
            if not isinstance(value, dict) or value.get("type") != "depth_prior":
                continue
            prior_source = str(value.get("source_rgb") or "")
            try:
                same_source = bool(prior_source) and (
                    Path(prior_source).resolve(strict=False)
                    == Path(source_rgb).resolve(strict=False)
                )
            except (OSError, ValueError):
                same_source = prior_source == source_rgb
            if same_source:
                matching_prior = dict(value)
                break

        parameters: JsonDict = {
            "rgb": source_rgb,
            "depth": depth,
            "intrinsics": dict(intrinsics),
            "camera_id": source.get("frame_id"),
            "registration_status": "aligned",
            "rgb_timestamp_s": source.get("timestamp_s"),
            "depth_timestamp_s": source.get("timestamp_s"),
            "scene_epoch": source.get("object_scene_epoch", source.get("scene_epoch")),
            "bundle_id": f"{source.get('packet_id')}:{source.get('frame_id')}",
            "source_packet_id": source.get("packet_id"),
            "camera_frame_id": source.get("frame_id"),
            "_source_observation": dict(source),
        }
        if isinstance(matching_prior, dict):
            parameters.update(
                {
                    "prior_depth": matching_prior.get("prior_depth"),
                    "prior_confidence": matching_prior.get("prior_confidence"),
                    "prior_confidence_semantics": matching_prior.get(
                        "prior_confidence_semantics"
                    ),
                }
            )
        return {
            "schema_version": "openeta.depth_enhancement_input_resolution.v1",
            "source_packet_id": source.get("packet_id"),
            "camera_frame_id": source.get("frame_id"),
            "depth_prior_status": "matched" if matching_prior is not None else "absent",
            "parameters": parameters,
        }

    def observation_packet_id_for_path(self, path: object) -> str:
        """Map one exact session artifact path back to its owning packet."""

        packet_id = find_packet_id_for_path(self._observation_packets.values(), path)
        if packet_id:
            return packet_id
        self._load_observation_packet_index_from_store()
        return find_packet_id_for_path(self._observation_packets.values(), path)

    def observation_packet_reference_for_path(self, path: object) -> JsonDict:
        """Map one exact session artifact path to packet and camera identifiers."""

        reference = find_packet_reference_for_path(self._observation_packets.values(), path)
        if reference:
            return reference
        self._load_observation_packet_index_from_store()
        return find_packet_reference_for_path(self._observation_packets.values(), path)

    def recent_observation_packet_refs(self, *, limit: int = 6) -> list[JsonDict]:
        """Return a bounded repair index without exposing local paths."""

        entries = list(self._observation_packets.values())[-max(0, limit) :]
        refs: list[JsonDict] = []
        for entry in entries:
            frames = sorted(
                {
                    str(item.get("frame_id") or "")
                    for item in entry.get("artifacts", [])
                    if isinstance(item, dict) and item.get("kind") == "rgb"
                }
            )
            refs.append(
                {
                    "source_packet_id": entry.get("packet_id"),
                    "observation_index": entry.get("observation_index"),
                    "camera_frame_ids": frames,
                }
            )
        return refs

    def _register_observation_packet(self, entry: JsonDict) -> None:
        packet_id = str(entry.get("packet_id") or "").strip()
        if not packet_id:
            return
        existing = self._observation_packets.get(packet_id)
        if existing is not None:
            if packet_integrity_fingerprint(existing) != packet_integrity_fingerprint(entry):
                raise ObservationPacketResolutionError(
                    "duplicate_source_packet_id",
                    "One source_packet_id refers to different immutable packet contents.",
                    details={
                        "source_packet_id": packet_id,
                        "active_agent_session_id": self.session_id or "",
                    },
                )
            return
        self._observation_packets[packet_id] = dict(entry)

    def _index_observation_packets_from_payload(self, payload: JsonDict) -> None:
        entries = payload.get("observation_packets")
        if not isinstance(entries, list):
            return
        for entry in entries:
            if isinstance(entry, dict):
                self._register_observation_packet(entry)

    def _load_observation_packet_index_from_store(self) -> None:
        if self.store is None or self.session_id is None:
            return
        try:
            rows = self.store.load_events(self.session_id, limit=None)
        except (OSError, ValueError, json.JSONDecodeError):
            return
        for row in rows:
            if not isinstance(row, dict) or row.get("event_type") != "observation":
                continue
            payload = row.get("payload")
            if isinstance(payload, dict):
                self._index_observation_packets_from_payload(payload)

    def add_action(self, action: EnvAction) -> None:
        # Bind each decision to the exact evidence state it was made from.  This
        # is causal bookkeeping for reflection/debugging, not a task phase: the
        # Agent remains free to choose any next tool.
        input_state_anchor = self._latest_observation_state_anchor()
        environment_task_updated = self._capture_active_environment_task(action)
        tool_health_updated = self._capture_tool_health(action)
        self._capture_reference_localization_state(action)
        self._capture_sam3_selection_state(action)
        self._materialize_complete_structured_outputs(action)
        captured_artifacts = _extract_action_artifacts(action)
        for artifact in captured_artifacts:
            if artifact.get("type") == "grasp_candidates" and artifact.get(
                "scene_epoch"
            ) is None:
                artifact["scene_epoch"] = self.scene_epoch()
            key = _artifact_memory_key(artifact, fallback_index=len(self.artifacts))
            self.artifacts[key] = {
                "value": artifact,
                "source": "tool_result",
                "timestamp_s": time.time(),
            }
        new_grasp_evidence_recorded = self._record_new_targeted_grasp_evidence(action)
        grasp_provenance_updated = self._capture_grasp_provenance(action)
        grasp_input_bundle_updated = self._refresh_grasp_input_bundle()
        wrist_alignment_bundle_updated = self._refresh_wrist_alignment_bundle()
        # Freeze an AnyPlace result against the exact bundle that authorized
        # the completed call before deriving any newer host bundle.  Refreshing
        # first can advance active_bundle_id and make a valid result look
        # unrelated to its own request, leaving the artifact on disk but absent
        # from the session resolver index.
        anyplace_bundle_materialized = self._capture_anyplace_bundle_materialization(action)
        anyplace_bundle_updated = self._refresh_anyplace_input_bundle()
        placement_reference_updated = self._capture_placement_world_reference(action)
        grasp_adjustment_budget_updated = self._capture_grasp_adjustment_budget(action)
        ik_preview_updated = self._capture_ik_preview_receipt(action)
        world_mutated = self._record_world_mutation(action)
        clearance_execution_updated = self._capture_compiled_clearance_execution(action)
        contact_execution_updated = self._capture_compiled_contact_execution(action)
        gripper_state_updated = self._capture_gripper_command_state(action)
        articulated_probe_prepared = self._capture_articulated_attachment_probe(action)
        articulated_probe_updated = self._capture_articulated_attachment_probe_result(action)
        articulated_assessment_updated = self._capture_articulated_attachment_assessment(action)
        reconciliation_updated = self._capture_motion_reconciliation(action)
        self._append_transition_ledger(action)
        if (
            captured_artifacts
            or world_mutated
            or articulated_probe_prepared
            or articulated_probe_updated
            or articulated_assessment_updated
            or reconciliation_updated
            or gripper_state_updated
            or environment_task_updated
            or grasp_provenance_updated
            or new_grasp_evidence_recorded
            or grasp_input_bundle_updated
            or wrist_alignment_bundle_updated
            or anyplace_bundle_updated
            or anyplace_bundle_materialized
            or placement_reference_updated
            or grasp_adjustment_budget_updated
            or ik_preview_updated
            or clearance_execution_updated
            or contact_execution_updated
            or tool_health_updated
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
                "input_state_anchor": input_state_anchor,
            },
        )
        for conversation_item in self.conversation.add_action(action):
            self._append_conversation_record(item_record(conversation_item))

    def add_external_event(self, event: JsonDict) -> None:
        event_type = str(event.get("type", "external"))
        self.record(event_type, event)
        if event_type == "human_answer":
            answer = event.get("answer")
            if isinstance(answer, str) and answer.strip():
                self.begin_user_turn(answer, source="human_answer")

    def save_fact(self, key: str, value: JsonDict, *, source: str = "") -> None:
        if is_removed_task_policy_key(key):
            raise ValueError(
                f"{key!r} is a reserved tombstone for deleted host task policy; "
                "store an Agent-owned plan or hypothesis under a different key"
            )
        entry = {"value": dict(value), "source": source, "timestamp_s": time.time()}
        self.facts[key] = entry
        if source == "save_memory":
            self.agent_working_state[key] = {
                **entry,
                "ownership": "agent",
                "freshness": "agent_managed",
            }
        self.record("memory_fact_saved", {"key": key, "source": source})
        self._save_working_memory()

    def _latest_observation_state_anchor(self) -> JsonDict:
        """Return the newest visual/epoch anchor actually available to a decision."""

        for event in reversed(self.events):
            if event.event_type != "observation":
                continue
            payload = event.payload
            signature = str(payload.get("visual_signature") or "")
            if not signature:
                return {}
            return {
                "observation_index": payload.get("observation_index"),
                "visual_signature": signature,
                "object_scene_epoch": _fact_epoch_value(
                    payload.get("object_scene_epoch")
                ),
                "robot_motion_epoch": _fact_epoch_value(
                    payload.get("robot_motion_epoch")
                ),
            }
        return {}

    def _materialize_complete_structured_outputs(self, action: EnvAction) -> None:
        """Persist complete candidate sets before working memory keeps only previews."""

        if self.artifact_root is None:
            return
        command = action.command if isinstance(action.command, dict) else {}
        for call in command.get("tool_calls", []) or []:
            if not isinstance(call, dict):
                continue
            tool_name = str(call.get("name") or "")
            if tool_name not in {
                "grasp_pose_estimate",
                "anygrasp",
                "graspgenx",
                "contact_graspnet",
                "anyplace",
            }:
                continue
            result = call.get("result")
            if not isinstance(result, dict) or result.get("success") is not True:
                continue
            details = result.get("details")
            if not isinstance(details, dict) or isinstance(
                details.get("structured_artifact"), dict
            ):
                continue
            structured_source = details.get("outputs")
            if not isinstance(structured_source, dict):
                structured_source = details
            candidate_key = (
                "placement_candidates" if tool_name == "anyplace" else "grasp_candidates"
            )
            if not isinstance(structured_source.get(candidate_key), list):
                continue
            try:
                snapshot = json.loads(json.dumps(structured_source, ensure_ascii=False))
                reference = materialize_structured_tool_output(
                    output_root=self.artifact_root,
                    tool=tool_name,
                    outputs=snapshot,
                    result_id=str(structured_source.get("result_id") or tool_name),
                )
            except (OSError, TypeError, ValueError) as exc:
                diagnostics = details.setdefault("diagnostics", [])
                if isinstance(diagnostics, list):
                    diagnostics.append(
                        {
                            "code": "structured_artifact_persistence_failed",
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    )
                continue
            details["structured_artifact"] = reference
            artifacts = details.setdefault("artifacts", [])
            if isinstance(artifacts, list):
                artifacts.append(dict(reference))

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

    def selected_sam3_detection(
        self,
        evidence_role: str = DEFAULT_SAM3_EVIDENCE_ROLE,
    ) -> JsonDict | None:
        role = _normalize_sam3_evidence_role(evidence_role)
        selections = _memory_fact_value(self.facts.get(SELECTED_SAM3_DETECTIONS_KEY))
        if isinstance(selections, dict):
            selected = selections.get(role)
            if isinstance(selected, dict):
                return selected
        if role == DEFAULT_SAM3_EVIDENCE_ROLE:
            return _memory_fact_value(self.facts.get(SELECTED_SAM3_DETECTION_KEY))
        return None

    def selected_sam3_detections(self) -> JsonDict:
        selections = _memory_fact_value(self.facts.get(SELECTED_SAM3_DETECTIONS_KEY))
        result = {
            str(role): dict(selected)
            for role, selected in (selections.items() if isinstance(selections, dict) else [])
            if role in SAM3_EVIDENCE_ROLES and isinstance(selected, dict)
        }
        legacy = _memory_fact_value(self.facts.get(SELECTED_SAM3_DETECTION_KEY))
        if DEFAULT_SAM3_EVIDENCE_ROLE not in result and isinstance(legacy, dict):
            result[DEFAULT_SAM3_EVIDENCE_ROLE] = dict(legacy)
        return result

    def _store_selected_sam3_detection(
        self,
        selected: JsonDict,
        *,
        evidence_role: str,
        source: str,
    ) -> None:
        role = _normalize_sam3_evidence_role(evidence_role)
        selections = self.selected_sam3_detections()
        selections[role] = dict(selected)
        self.facts[SELECTED_SAM3_DETECTIONS_KEY] = _memory_fact_entry(
            selections,
            source=source,
        )
        if role == DEFAULT_SAM3_EVIDENCE_ROLE:
            self.facts[SELECTED_SAM3_DETECTION_KEY] = _memory_fact_entry(
                selected,
                source=source,
            )

    def _remove_selected_sam3_detection(self, *, evidence_role: str) -> None:
        role = _normalize_sam3_evidence_role(evidence_role)
        selections = self.selected_sam3_detections()
        selections.pop(role, None)
        if selections:
            self.facts[SELECTED_SAM3_DETECTIONS_KEY] = _memory_fact_entry(
                selections,
                source="sam3_selection_removed",
            )
        else:
            self.facts.pop(SELECTED_SAM3_DETECTIONS_KEY, None)
        if role == DEFAULT_SAM3_EVIDENCE_ROLE:
            self.facts.pop(SELECTED_SAM3_DETECTION_KEY, None)

    def pending_reference_localization(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(PENDING_REFERENCE_LOCALIZATION_KEY))

    def target_asset_reference(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(TARGET_ASSET_REFERENCE_KEY))

    def target_identity_anchor(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(TARGET_IDENTITY_ANCHOR_KEY))

    def reference_localization_failure(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(REFERENCE_LOCALIZATION_FAILURE_KEY))

    def tool_health(self) -> JsonDict:
        """Return compact session-local backend health inferred from tool receipts."""

        value = _memory_fact_value(self.facts.get(TOOL_HEALTH_KEY))
        return {
            str(name): dict(entry)
            for name, entry in (value.items() if isinstance(value, dict) else [])
            if isinstance(entry, dict)
        }

    def _capture_tool_health(self, action: EnvAction) -> bool:
        """Track infrastructure failures without turning health into a hard gate.

        Semantic failures such as no detection or no grasp candidate say nothing
        about service health and are intentionally ignored. A later successful
        receipt closes the advisory circuit immediately.
        """

        command = action.command if isinstance(action.command, dict) else {}
        calls = command.get("tool_calls")
        calls = calls if isinstance(calls, list) else []
        health = self.tool_health()
        changed = False
        for call in calls:
            if not isinstance(call, dict):
                continue
            name = str(call.get("name") or "").strip()
            status = str(call.get("status") or "").strip().lower()
            result = call.get("result")
            result = result if isinstance(result, dict) else {}
            if not name or status == "skipped":
                continue
            details = result.get("details")
            details = details if isinstance(details, dict) else {}
            diagnostics = details.get("diagnostics")
            diagnostics = diagnostics if isinstance(diagnostics, list) else []
            failure_codes = {
                str(item.get("code") or "").strip().lower()
                for item in diagnostics
                if isinstance(item, dict) and item.get("code")
            }
            reason = str(details.get("reason") or "").strip().lower()
            if reason:
                failure_codes.add(reason)
            infrastructure_codes = sorted(
                code for code in failure_codes if code in INFRASTRUCTURE_FAILURE_CODES
            )
            previous = health.get(name, {})
            previous = previous if isinstance(previous, dict) else {}
            if result.get("success") is True or (
                previous and status in {"executed", "failed"} and not infrastructure_codes
            ):
                if previous:
                    health[name] = {
                        "status": "healthy",
                        "consecutive_infrastructure_failures": 0,
                        "last_success_at_s": (
                            time.time() if result.get("success") is True else None
                        ),
                        "last_backend_response_at_s": time.time(),
                        "last_failure_code": previous.get("last_failure_code"),
                        "policy": "normal_use",
                    }
                    changed = True
                continue
            if status != "failed" or not infrastructure_codes:
                continue
            count = int(previous.get("consecutive_infrastructure_failures") or 0) + 1
            parameters = call.get("parameters")
            if not isinstance(parameters, dict):
                request = command.get("request")
                parameters = request.get("parameters") if isinstance(request, dict) else {}
            parameters = parameters if isinstance(parameters, dict) else {}
            sources = parameters.get("sources")
            first_source = (
                sources[0]
                if isinstance(sources, list)
                and sources
                and isinstance(sources[0], dict)
                else {}
            )
            circuit_open = count >= 2
            health[name] = {
                "status": "circuit_open" if circuit_open else "degraded",
                "consecutive_infrastructure_failures": count,
                "last_failure_code": infrastructure_codes[0],
                "last_failure_at_s": time.time(),
                "last_error": str(result.get("content") or "")[:500],
                "last_request": {
                    key: parameters.get(key, first_source.get(key))
                    for key in ("source_packet_id", "camera_frame_id", "mode")
                    if parameters.get(key, first_source.get(key)) is not None
                },
                "policy": (
                    "do_not_repeat_until_successful_preflight_or_explicit_health_change"
                    if circuit_open
                    else "retry_at_most_once_after_preflight_or_use_alternative"
                ),
                "host_enforcement": "advisory_only",
            }
            changed = True
        if changed:
            self.facts[TOOL_HEALTH_KEY] = _memory_fact_entry(
                health,
                source="runtime_tool_health",
            )
            self.record("tool_health_updated", {"tools": health})
        return changed

    def sam3_no_detection(
        self,
        evidence_role: str = DEFAULT_SAM3_EVIDENCE_ROLE,
    ) -> JsonDict | None:
        role = _normalize_sam3_evidence_role(evidence_role)
        failures = _memory_fact_value(self.facts.get(SAM3_NO_DETECTIONS_KEY))
        if isinstance(failures, dict):
            failure = failures.get(role)
            if isinstance(failure, dict):
                return failure
        if role == DEFAULT_SAM3_EVIDENCE_ROLE:
            return _memory_fact_value(self.facts.get(SAM3_NO_DETECTION_KEY))
        return None

    def sam3_no_detections(self) -> JsonDict:
        failures = _memory_fact_value(self.facts.get(SAM3_NO_DETECTIONS_KEY))
        result = {
            str(role): dict(failure)
            for role, failure in (failures.items() if isinstance(failures, dict) else [])
            if role in SAM3_EVIDENCE_ROLES and isinstance(failure, dict)
        }
        legacy = _memory_fact_value(self.facts.get(SAM3_NO_DETECTION_KEY))
        if DEFAULT_SAM3_EVIDENCE_ROLE not in result and isinstance(legacy, dict):
            result[DEFAULT_SAM3_EVIDENCE_ROLE] = dict(legacy)
        return result

    def same_view_point_grounding_source_error(
        self,
        resolved_sources: list[JsonDict],
    ) -> str | None:
        """Reject a silent packet refresh inside active same-view recovery.

        Packet ids can advance on read-only turns even when the physical camera
        view is unchanged.  A SAM3 no-detection handoff binds immutable pixels,
        not the spelling of one packet id: a later same-camera packet is valid
        when its RGB bytes are identical.  Changed pixels still require the exact
        handoff packet (or an Agent-owned switch to another camera).
        """

        no_detection = self.sam3_no_detection()
        if not isinstance(no_detection, dict):
            return None
        expected_packet = str(no_detection.get("source_packet_id") or "").strip()
        expected_frame = str(no_detection.get("frame_id") or "").strip()
        source_observation = no_detection.get("source_observation")
        source_observation = (
            source_observation if isinstance(source_observation, dict) else {}
        )
        object_epoch = no_detection.get("object_scene_epoch")
        if object_epoch is None:
            object_epoch = no_detection.get("scene_epoch")
        if object_epoch is None:
            object_epoch = source_observation.get("object_scene_epoch")
        robot_epoch = no_detection.get("robot_motion_epoch")
        if robot_epoch is None:
            robot_epoch = source_observation.get("robot_motion_epoch")
        if (
            not expected_packet
            or not expected_frame
            or object_epoch is None
            or robot_epoch is None
            or _fact_epoch_value(object_epoch) != self.object_scene_epoch()
            or _fact_epoch_value(robot_epoch) != self.robot_motion_epoch()
        ):
            return None
        same_frame = [
            source
            for source in resolved_sources
            if str(source.get("frame_id") or "") == expected_frame
        ]
        if not same_frame or any(
            str(source.get("packet_id") or "") == expected_packet
            for source in same_frame
        ):
            return None
        expected_rgb = str(
            source_observation.get("rgb") or no_detection.get("source_image") or ""
        ).strip()
        if expected_rgb and any(
            _paths_have_identical_content(expected_rgb, source.get("rgb"))
            for source in same_frame
        ):
            return None
        supplied = sorted(
            {
                str(source.get("packet_id") or "")
                for source in same_frame
                if source.get("packet_id")
            }
        )
        return (
            "Same-view point grounding must include the exact SAM3 no-detection "
            f"source packet {expected_packet!r} for camera {expected_frame!r}; "
            f"received same-camera packet(s) {supplied!r}. Copy the immutable "
            "handoff id instead of substituting a packet whose pixels changed. "
            "A byte-identical read-only packet refresh is accepted automatically."
        )

    def _store_sam3_no_detection(
        self,
        failure: JsonDict,
        *,
        evidence_role: str,
        source: str,
    ) -> None:
        role = _normalize_sam3_evidence_role(evidence_role)
        failures = self.sam3_no_detections()
        failures[role] = dict(failure)
        self.facts[SAM3_NO_DETECTIONS_KEY] = _memory_fact_entry(
            failures,
            source=source,
        )
        if role == DEFAULT_SAM3_EVIDENCE_ROLE:
            self.facts[SAM3_NO_DETECTION_KEY] = _memory_fact_entry(
                failure,
                source=source,
            )

    def _remove_sam3_no_detection(self, *, evidence_role: str) -> None:
        role = _normalize_sam3_evidence_role(evidence_role)
        failures = self.sam3_no_detections()
        failures.pop(role, None)
        if failures:
            self.facts[SAM3_NO_DETECTIONS_KEY] = _memory_fact_entry(
                failures,
                source="sam3_no_detection_removed",
            )
        else:
            self.facts.pop(SAM3_NO_DETECTIONS_KEY, None)
        if role == DEFAULT_SAM3_EVIDENCE_ROLE:
            self.facts.pop(SAM3_NO_DETECTION_KEY, None)

    def retained_targeted_grasp(self) -> JsonDict | None:
        """Return the latest Agent-selected targeted-grasp evidence and source packet."""

        provenance = _memory_fact_value(self.facts.get(GRASP_PROVENANCE_KEY))
        if (
            isinstance(provenance, dict)
            and _fact_epoch_value(provenance.get("object_scene_epoch"))
            == self.object_scene_epoch()
            and not provenance.get("contact_geometry_invalidated_at_s")
        ):
            candidate = provenance.get("candidate")
            source = provenance.get("source")
            if isinstance(candidate, dict) and isinstance(source, dict):
                return {
                    "schema_version": "openeta.retained_targeted_grasp.v1",
                    "evidence_id": provenance.get("evidence_id"),
                    "artifact_key": provenance.get("artifact_key"),
                    "result_id": provenance.get("result_id"),
                    "compiled_grasp_id": provenance.get("compiled_grasp_id"),
                    "status": "retained",
                    "candidate": dict(candidate),
                    "source": dict(source),
                }

        # Ranked estimator output remains evidence in working-memory artifacts.
        # It becomes a singular retained target only after the Agent explicitly
        # selects and compiles a candidate, which records grasp provenance above.
        return None

    def anyplace_input_bundle(self) -> JsonDict | None:
        """Return the public reference for the active host-resolved AnyPlace bundle."""

        state = _memory_fact_value(self.facts.get(ANYPLACE_INPUT_BUNDLES_KEY))
        if not isinstance(state, dict):
            return None
        public = state.get("public")
        return dict(public) if isinstance(public, dict) else None

    def grasp_input_bundle(self) -> JsonDict | None:
        """Return the public reference for the active targeted-grasp input bundle."""

        state = _memory_fact_value(self.facts.get(GRASP_INPUT_BUNDLES_KEY))
        if not isinstance(state, dict):
            return None
        public = state.get("public")
        if not isinstance(public, dict):
            return None
        projected = dict(public)
        bundle_epoch = _fact_epoch_value(projected.get("object_scene_epoch"))
        if projected.get("status") == "ready" and bundle_epoch != self.object_scene_epoch():
            projected.update(
                {
                    "status": "stale_object_scene",
                    "bundle_id": None,
                    "call_parameters": None,
                    "recovery": "segment the target on a current observation packet",
                }
            )
        return projected

    def wrist_alignment_bundle(self) -> JsonDict | None:
        state = _memory_fact_value(self.facts.get(WRIST_ALIGNMENT_BUNDLES_KEY))
        public = state.get("public") if isinstance(state, dict) else None
        if not isinstance(public, dict):
            return None
        projected = dict(public)
        if projected.get("status") == "ready":
            if (
                _fact_epoch_value(projected.get("object_scene_epoch"))
                != self.object_scene_epoch()
            ):
                projected.update(
                    {
                        "status": "stale_object_scene",
                        "bundle_id": None,
                        "call_parameters": None,
                        "recovery": "segment the target on a current wrist observation packet",
                    }
                )
            elif (
                _fact_epoch_value(projected.get("robot_motion_epoch"))
                != self.robot_motion_epoch()
            ):
                projected.update(
                    {
                        "status": "stale_robot_motion",
                        "bundle_id": None,
                        "call_parameters": None,
                        "recovery": (
                            "observe fresh wrist RGB-D after the latest robot motion, "
                            "then segment and select the same target instance"
                        ),
                    }
                )
        return projected

    def resolve_wrist_alignment_bundle(self, bundle_id: str) -> JsonDict:
        requested = str(bundle_id or "").strip()
        state = _memory_fact_value(self.facts.get(WRIST_ALIGNMENT_BUNDLES_KEY))
        active_id = str(state.get("active_bundle_id") or "") if isinstance(state, dict) else ""
        bundles = state.get("bundles") if isinstance(state, dict) else None
        if not requested or requested != active_id:
            raise ValueError(
                "wrist alignment bundle is not active; inspect "
                "host_resolved_inputs.wrist_alignment and copy its exact bundle_id"
            )
        bundle = bundles.get(requested) if isinstance(bundles, dict) else None
        parameters = bundle.get("parameters") if isinstance(bundle, dict) else None
        if not isinstance(parameters, dict):
            raise ValueError("wrist alignment bundle parameters are unavailable")
        if _fact_epoch_value(bundle.get("object_scene_epoch")) != self.object_scene_epoch():
            raise ValueError("wrist alignment bundle belongs to a stale object-scene epoch")
        if _fact_epoch_value(bundle.get("robot_motion_epoch")) != self.robot_motion_epoch():
            raise ValueError(
                "wrist alignment bundle belongs to a stale robot-motion epoch; "
                "observe fresh wrist RGB-D and reselect the target"
            )
        return {
            "schema_version": "openeta.wrist_alignment_bundle_resolution.v1",
            "bundle_id": requested,
            "parameters": dict(parameters),
            "target_evidence_id": bundle.get("target_evidence_id"),
            "compiled_grasp_id": bundle.get("compiled_grasp_id"),
        }

    def resolve_grasp_input_bundle(self, bundle_id: str) -> JsonDict:
        """Resolve an opaque targeted-grasp bundle to immutable host-owned inputs."""

        requested = str(bundle_id or "").strip()
        state = _memory_fact_value(self.facts.get(GRASP_INPUT_BUNDLES_KEY))
        active_id = str(state.get("active_bundle_id") or "") if isinstance(state, dict) else ""
        bundles = state.get("bundles") if isinstance(state, dict) else None
        if not requested:
            raise ValueError("grasp_pose_estimate requires a non-empty bundle_id")
        if requested != active_id:
            raise ValueError(
                "grasp bundle is not the active host-resolved evidence bundle; inspect "
                "host_resolved_inputs.grasp_pose_estimate and use its exact bundle_id"
            )
        bundle = bundles.get(requested) if isinstance(bundles, dict) else None
        parameters = bundle.get("parameters") if isinstance(bundle, dict) else None
        if not isinstance(parameters, dict):
            raise ValueError("grasp bundle parameters are unavailable")
        bundle_epoch = _fact_epoch_value(bundle.get("object_scene_epoch"))
        if bundle_epoch != self.object_scene_epoch():
            raise ValueError(
                "grasp bundle belongs to a stale object-scene epoch; segment the target "
                "on a current observation packet"
            )
        return {
            "schema_version": "openeta.grasp_input_bundle_resolution.v1",
            "bundle_id": requested,
            "parameters": dict(parameters),
            "target_evidence_id": bundle.get("target_evidence_id"),
        }

    def resolve_wrist_viewpoint_input(
        self,
        *,
        compiled_grasp_id: str,
        source_packet_id: str,
        camera_frame_id: str,
    ) -> JsonDict:
        """Resolve current wrist geometry for read-only target-facing view proposals."""

        compiled_id = str(compiled_grasp_id or "").strip()
        if not compiled_id:
            raise ValueError("propose_wrist_viewpoints requires compiled_grasp_id")
        compiled = self._compiled_grasp_artifact(compiled_id)
        if not isinstance(compiled, dict):
            raise ValueError(
                "compiled_grasp_id is unknown in the active provenance graph; compile "
                "one current target candidate first"
            )
        source = self.resolve_observation_packet(source_packet_id, camera_frame_id)
        frame_id = str(source.get("frame_id") or "")
        role = str(source.get("role") or "")
        if not _is_wrist_camera(frame_id, role):
            raise ValueError(
                "propose_wrist_viewpoints requires a wrist-camera frame from the source packet"
            )
        packet_object_epoch = _fact_epoch_value(source.get("object_scene_epoch"))
        packet_robot_epoch = _fact_epoch_value(source.get("robot_motion_epoch"))
        if packet_object_epoch != self.object_scene_epoch():
            raise ValueError(
                "source packet belongs to a stale object-scene epoch; observe again"
            )
        if packet_robot_epoch != self.robot_motion_epoch():
            raise ValueError(
                "source packet belongs to a stale robot-motion epoch; observe again after motion"
            )
        missing = [
            name
            for name in ("extrinsics", "current_eef_pose")
            if not isinstance(source.get(name), dict) or not source.get(name)
        ]
        if missing:
            raise ValueError(
                "wrist source packet lacks calibrated viewpoint inputs: " + ", ".join(missing)
            )
        return {
            "schema_version": "openeta.wrist_viewpoint_input_resolution.v1",
            "compiled_grasp_id": compiled_id,
            "source_packet_id": source.get("packet_id"),
            "camera_frame_id": frame_id,
            "parameters": {
                "compiled_grasp": compiled,
                "source_packet_id": source.get("packet_id"),
                "camera_frame_id": frame_id,
                "camera_extrinsics": source.get("extrinsics"),
                "current_eef_pose": source.get("current_eef_pose"),
                "object_scene_epoch": self.object_scene_epoch(),
                "robot_motion_epoch": self.robot_motion_epoch(),
            },
        }

    def resolve_grasp_candidate_input(
        self,
        *,
        grasp_result_id: str,
        candidate_id: str,
    ) -> JsonDict:
        """Resolve one estimator candidate and its immutable camera calibration."""

        requested_result = str(grasp_result_id or "").strip()
        requested_candidate = str(candidate_id or "").strip()
        if not requested_result or not requested_candidate:
            raise ValueError(
                "compile_grasp_seed requires exact grasp_result_id and candidate_id"
            )
        recent_results: list[str] = []
        for event in reversed(self.events):
            if event.event_type != "action":
                continue
            command = event.payload.get("command")
            command = command if isinstance(command, dict) else {}
            calls = command.get("tool_calls")
            calls = calls if isinstance(calls, list) else []
            for call in reversed(calls):
                if not isinstance(call, dict) or str(call.get("name") or "") not in {
                    "grasp_pose_estimate",
                    "anygrasp",
                    "graspgenx",
                    "contact_graspnet",
                }:
                    continue
                outputs = _tool_call_outputs(call)
                result_id = str(outputs.get("result_id") or "")
                if result_id and result_id not in recent_results:
                    recent_results.append(result_id)
                if result_id != requested_result:
                    continue
                candidates = outputs.get("grasp_candidates")
                candidates = candidates if isinstance(candidates, list) else []
                candidate = next(
                    (
                        dict(item)
                        for item in candidates
                        if isinstance(item, dict)
                        and str(item.get("id") or item.get("candidate_id") or "")
                        == requested_candidate
                    ),
                    None,
                )
                if candidate is None:
                    raise ValueError(
                        f"candidate_id {requested_candidate!r} is not present in grasp "
                        f"result {requested_result!r}; inspect that result's candidates"
                    )
                source = outputs.get("source")
                source = source if isinstance(source, dict) else {}
                source_packet_id = str(
                    source.get("source_packet_id")
                    or outputs.get("source_packet_id")
                    or candidate.get("source_packet_id")
                    or ""
                )
                camera_frame_id = str(
                    source.get("camera_frame_id")
                    or outputs.get("camera_frame_id")
                    or candidate.get("camera_frame_id")
                    or ""
                )
                resolved_source = self.resolve_observation_packet(
                    source_packet_id,
                    camera_frame_id,
                )
                evidence_epoch = _fact_epoch_value(
                    outputs.get("scene_epoch", source.get("scene_epoch"))
                )
                if evidence_epoch != self.object_scene_epoch():
                    raise ValueError(
                        "grasp_result_id belongs to a stale object-scene epoch; rerun "
                        "targeted grasp estimation"
                    )
                extrinsics = resolved_source.get("extrinsics")
                if not isinstance(extrinsics, dict) or not extrinsics:
                    raise ValueError(
                        "the grasp result source packet lacks camera extrinsics"
                    )
                return {
                    "schema_version": "openeta.grasp_candidate_input_resolution.v1",
                    "grasp_result_id": requested_result,
                    "candidate_id": requested_candidate,
                    "source_packet_id": resolved_source.get("packet_id"),
                    "camera_frame_id": resolved_source.get("frame_id"),
                    "parameters": {
                        "camera_pose": candidate,
                        "camera_extrinsics": dict(extrinsics),
                        "camera_frame_id": resolved_source.get("frame_id"),
                        "scene_epoch": self.object_scene_epoch(),
                        **(
                            {
                                "target_mask_quality": dict(
                                    outputs["target_mask_quality"]
                                )
                            }
                            if isinstance(outputs.get("target_mask_quality"), dict)
                            else {}
                        ),
                        "grasp_candidate_count": _optional_int(
                            outputs.get("candidate_count"), default=0
                        ),
                        **(
                            {
                                "grasp_selection_advice": dict(
                                    outputs["grasp_selection_advice"]
                                )
                            }
                            if isinstance(outputs.get("grasp_selection_advice"), dict)
                            else {}
                        ),
                    },
                }
        raise ValueError(
            f"unknown grasp_result_id {requested_result!r} in the active session; "
            f"recent result ids: {recent_results[:6]}"
        )

    def resolve_wrist_viewpoint_candidate(
        self,
        *,
        proposal_id: str,
        candidate_id: str,
    ) -> JsonDict:
        """Resolve a proposal candidate to the exact full pose returned by the host."""

        requested_proposal = str(proposal_id or "").strip()
        requested_candidate = str(candidate_id or "").strip()
        if not requested_proposal or not requested_candidate:
            raise ValueError(
                "ik_preview_check viewpoint reference requires viewpoint_proposal_id "
                "and candidate_id"
            )
        for event in reversed(self.events):
            if event.event_type != "action":
                continue
            command = event.payload.get("command")
            command = command if isinstance(command, dict) else {}
            calls = command.get("tool_calls")
            calls = calls if isinstance(calls, list) else []
            for call in reversed(calls):
                if not isinstance(call, dict) or call.get("name") != "propose_wrist_viewpoints":
                    continue
                outputs = _tool_call_outputs(call)
                if str(outputs.get("proposal_id") or "") != requested_proposal:
                    continue
                if _fact_epoch_value(outputs.get("object_scene_epoch")) != self.object_scene_epoch():
                    raise ValueError(
                        "viewpoint proposal is stale because object_scene_epoch changed"
                    )
                if _fact_epoch_value(outputs.get("robot_motion_epoch")) != self.robot_motion_epoch():
                    raise ValueError(
                        "viewpoint proposal is stale because robot_motion_epoch changed"
                    )
                candidates = outputs.get("candidates")
                candidates = candidates if isinstance(candidates, list) else []
                candidate = next(
                    (
                        item
                        for item in candidates
                        if isinstance(item, dict)
                        and str(item.get("candidate_id") or "") == requested_candidate
                    ),
                    None,
                )
                if not isinstance(candidate, dict):
                    raise ValueError(
                        f"candidate_id {requested_candidate!r} is not present in "
                        f"viewpoint proposal {requested_proposal!r}"
                    )
                target_pose = candidate.get("target_pose")
                if not isinstance(target_pose, dict):
                    raise ValueError("viewpoint candidate does not contain target_pose")
                return {
                    "schema_version": "openeta.wrist_viewpoint_candidate_resolution.v1",
                    "proposal_id": requested_proposal,
                    "candidate_id": requested_candidate,
                    "parameters": {"target_pose": dict(target_pose)},
                }
        raise ValueError(
            f"unknown viewpoint_proposal_id {requested_proposal!r} in the active session"
        )

    def resolve_anyplace_input_bundle(self, bundle_id: str) -> JsonDict:
        """Resolve an Agent-supplied bundle id to immutable host-owned parameters."""

        requested = str(bundle_id or "").strip()
        state = _memory_fact_value(self.facts.get(ANYPLACE_INPUT_BUNDLES_KEY))
        active_id = str(state.get("active_bundle_id") or "") if isinstance(state, dict) else ""
        bundles = state.get("bundles") if isinstance(state, dict) else None
        if not requested:
            raise ValueError("anyplace requires a non-empty bundle_id")
        if requested != active_id:
            raise ValueError(
                "anyplace bundle is not the active host-resolved evidence bundle; "
                "inspect host_resolved_inputs.anyplace and use its exact bundle_id"
            )
        bundle = bundles.get(requested) if isinstance(bundles, dict) else None
        parameters = bundle.get("parameters") if isinstance(bundle, dict) else None
        if not isinstance(parameters, dict):
            raise ValueError("anyplace bundle parameters are unavailable")
        bundle_epoch = _fact_epoch_value(bundle.get("object_scene_epoch"))
        valid_through_epoch = _fact_epoch_value(
            bundle.get("valid_through_object_scene_epoch")
        )
        materialized = bundle.get("materialized") is True
        current_epoch = self.object_scene_epoch()
        if bundle_epoch != current_epoch and not (
            materialized and valid_through_epoch >= current_epoch
        ):
            raise ValueError(
                "anyplace bundle belongs to a stale object-scene epoch; regenerate grasp "
                "and placement evidence"
            )
        return {
            "schema_version": "openeta.anyplace_bundle_resolution.v1",
            "bundle_id": requested,
            "parameters": {**dict(parameters), "bundle_id": requested},
            "grasp_evidence_id": bundle.get("grasp_evidence_id"),
            "placement_evidence_id": bundle.get("placement_evidence_id"),
        }

    def resolve_placement_candidate_input(
        self,
        *,
        placement_result_id: str,
        candidate_id: str,
    ) -> JsonDict:
        """Resolve one frozen AnyPlace candidate with its original camera calibration."""

        requested_result = str(placement_result_id or "").strip()
        requested_candidate = str(candidate_id or "").strip()
        if not requested_result or not requested_candidate:
            raise ValueError(
                "camera_pose_to_world requires exact placement_result_id and candidate_id"
            )
        state = _memory_fact_value(self.facts.get(ANYPLACE_INPUT_BUNDLES_KEY))
        bundles = state.get("bundles") if isinstance(state, dict) else None
        if not isinstance(bundles, dict):
            raise ValueError("no materialized AnyPlace results exist in the active session")
        recent_result_ids: list[str] = []
        for bundle in reversed(list(bundles.values())):
            if not isinstance(bundle, dict) or bundle.get("materialized") is not True:
                continue
            result_id = str(bundle.get("materialized_result_id") or "")
            if result_id and result_id not in recent_result_ids:
                recent_result_ids.append(result_id)
            if result_id != requested_result:
                continue
            candidates = bundle.get("placement_candidates")
            candidates = candidates if isinstance(candidates, list) else []
            candidate = next(
                (
                    dict(item)
                    for item in candidates
                    if isinstance(item, dict)
                    and str(item.get("id") or "") == requested_candidate
                ),
                None,
            )
            if candidate is None:
                raise ValueError(
                    f"candidate_id {requested_candidate!r} is not present in placement "
                    f"result {requested_result!r}; valid ids: "
                    + str(
                        [
                            item.get("id")
                            for item in candidates
                            if isinstance(item, dict) and item.get("id")
                        ]
                    )
                )
            place_pose = candidate.get("place_grasp_pose")
            if not isinstance(place_pose, dict):
                raise ValueError("placement candidate lacks place_grasp_pose")
            parameters = bundle.get("parameters")
            selected_grasp = (
                parameters.get("selected_grasp") if isinstance(parameters, dict) else None
            )
            source = (
                selected_grasp.get("source")
                if isinstance(selected_grasp, dict)
                else None
            )
            source = source if isinstance(source, dict) else {}
            source_packet_id = str(source.get("source_packet_id") or "")
            camera_frame_id = str(source.get("camera_frame_id") or "")
            if not source_packet_id:
                reference = self.observation_packet_reference_for_path(source.get("rgb"))
                reference = reference if isinstance(reference, dict) else {}
                source_packet_id = str(reference.get("source_packet_id") or "")
                camera_frame_id = camera_frame_id or str(
                    reference.get("camera_frame_id") or ""
                )
            try:
                resolved_source = self.resolve_observation_packet(
                    source_packet_id,
                    camera_frame_id,
                    require_files=False,
                )
            except (ValueError, ObservationPacketResolutionError) as exc:
                raise ValueError(
                    "the AnyPlace result's original observation packet cannot provide "
                    f"camera calibration: {exc}"
                ) from exc
            extrinsics = resolved_source.get("extrinsics")
            if not isinstance(extrinsics, dict) or not extrinsics:
                raise ValueError(
                    "the AnyPlace result's original observation packet lacks camera extrinsics"
                )
            return {
                "schema_version": "openeta.placement_candidate_input_resolution.v1",
                "placement_result_id": requested_result,
                "candidate_id": requested_candidate,
                "source_packet_id": resolved_source.get("packet_id"),
                "camera_frame_id": resolved_source.get("frame_id"),
                "parameters": {
                    "camera_pose": dict(place_pose),
                    "camera_extrinsics": dict(extrinsics),
                    "camera_frame_id": resolved_source.get("frame_id"),
                    "placement_result_id": requested_result,
                    "candidate_id": requested_candidate,
                },
            }
        raise ValueError(
            f"unknown placement_result_id {requested_result!r} in the active session; "
            f"recent result ids: {recent_result_ids[:6]}"
        )

    def provenance_evidence_graph(self) -> JsonDict:
        """Project the current grasp/placement lineage as a read-only evidence graph."""

        nodes: list[JsonDict] = []
        edges: list[JsonDict] = []
        inconsistencies: list[JsonDict] = []
        target = self.selected_sam3_detection()
        target_id = (
            f"sam3:{target.get('result_id')}:{target.get('id')}"
            if isinstance(target, dict)
            and target.get("result_id")
            and target.get("id")
            else ""
        )
        grasp = _memory_fact_value(self.facts.get(GRASP_PROVENANCE_KEY))
        if isinstance(grasp, dict) and isinstance(grasp.get("evidence_id"), str):
            grasp_id = str(grasp["evidence_id"])
            grasp_target_id = str(grasp.get("target_evidence_id") or "")
            grasp_anchor_id = str(grasp.get("target_identity_anchor_id") or "")
            current_anchor_id = (
                str(target.get("identity_anchor_id") or "")
                if isinstance(target, dict)
                else ""
            )
            compiled_artifact = next(
                (
                    entry.get("value")
                    for entry in self.artifacts.values()
                    if isinstance(entry, dict)
                    and isinstance(entry.get("value"), dict)
                    and entry["value"].get("type") == "compiled_grasp"
                    and str(entry["value"].get("compiled_grasp_id") or "")
                    == str(grasp.get("compiled_grasp_id") or "")
                ),
                {},
            )
            target_superseded = bool(
                target_id
                and grasp_target_id
                and target_id != grasp_target_id
                and not (
                    grasp_anchor_id
                    and current_anchor_id
                    and grasp_anchor_id == current_anchor_id
                )
            )
            object_scene_current = (
                _fact_epoch_value(grasp.get("object_scene_epoch"))
                == self.object_scene_epoch()
            )
            contact_geometry_invalidated = bool(
                grasp.get("contact_geometry_invalidated_at_s")
            )
            nodes.append(
                {
                    "evidence_id": grasp_id,
                    "kind": "compiled_targeted_grasp",
                    "target_evidence_id": grasp_target_id or None,
                    "target_identity_anchor_id": grasp_anchor_id or None,
                    "candidate_id": grasp.get("candidate_id"),
                    "compiled_grasp_id": grasp.get("compiled_grasp_id"),
                    "artifact_key": grasp.get("artifact_key"),
                    "object_scene_epoch": grasp.get("object_scene_epoch"),
                    "robot_motion_epoch": grasp.get("robot_motion_epoch"),
                    "freshness": (
                        "invalidated_contact_geometry"
                        if contact_geometry_invalidated
                        else (
                            "superseded_target_evidence"
                            if target_superseded
                            else (
                                "current_object_scene"
                                if object_scene_current
                                else "stale_object_scene"
                            )
                        )
                    ),
                    **(
                        {
                            "hover_pose": compiled_artifact.get("hover_pose"),
                            "contact_pose": compiled_artifact.get("contact_pose"),
                        }
                        if isinstance(compiled_artifact, dict)
                        else {}
                    ),
                }
            )
            if target_superseded:
                inconsistencies.append(
                    {
                        "code": "compiled_grasp_target_superseded",
                        "compiled_grasp_id": grasp.get("compiled_grasp_id"),
                        "grasp_evidence_id": grasp_id,
                        "compiled_target_evidence_id": grasp_target_id,
                        "current_target_evidence_id": target_id,
                        "recovery": (
                            "use the current grasp_pose_estimate bundle, choose a new "
                            "candidate, and compile it before contact or gripper close"
                        ),
                    }
                )
            if contact_geometry_invalidated:
                inconsistencies.append(
                    {
                        "code": "compiled_grasp_contact_geometry_invalidated",
                        "compiled_grasp_id": grasp.get("compiled_grasp_id"),
                        "grasp_evidence_id": grasp_id,
                        "invalidated_by": grasp.get("contact_geometry_invalidated_by"),
                        "recovery": (
                            "use fresh target evidence, choose a current grasp candidate, "
                            "and compile new contact geometry before close or probe"
                        ),
                    }
                )
        else:
            grasp_id = ""
            grasp_target_id = ""
        if isinstance(target, dict) and target_id:
            nodes.append(
                {
                    "evidence_id": target_id,
                    "kind": "selected_target_object",
                    "sam3_result_id": target.get("result_id"),
                    "detection_id": target.get("id"),
                    "source_image": target.get("source_image"),
                    "mask_ref": target.get("mask_ref"),
                    "object_scene_epoch": target.get("scene_epoch"),
                }
            )
            if grasp_id and grasp_target_id == target_id:
                edges.append(
                    {"from": target_id, "to": grasp_id, "relation": "target_input"}
                )
            elif (
                grasp_id
                and grasp_anchor_id
                and str(target.get("identity_anchor_id") or "") == grasp_anchor_id
            ):
                edges.append(
                    {
                        "from": target_id,
                        "to": grasp_id,
                        "relation": "same_instance_visual_refinement",
                    }
                )
            elif grasp_id and grasp_target_id:
                edges.append(
                    {"from": target_id, "to": grasp_id, "relation": "supersedes_target_input"}
                )
        placement = self.selected_sam3_detection("placement_region")
        if isinstance(placement, dict):
            placement_id = _placement_evidence_id(placement)
            nodes.append(
                {
                    "evidence_id": placement_id,
                    "kind": "selected_placement_region",
                    "sam3_result_id": placement.get("result_id"),
                    "detection_id": placement.get("id"),
                    "source_image": placement.get("source_image"),
                    "mask_ref": placement.get("mask_ref"),
                    "object_scene_epoch": placement.get("scene_epoch"),
                }
            )
        else:
            placement_id = ""
        bundle = self.anyplace_input_bundle()
        if isinstance(bundle, dict) and bundle.get("status") == "ready":
            bundle_id = str(bundle.get("bundle_id") or "")
            node_id = f"anyplace_bundle:{bundle_id}"
            nodes.append(
                {
                    "evidence_id": node_id,
                    "kind": "anyplace_input_bundle",
                    "bundle_id": bundle_id,
                    "status": "ready",
                }
            )
            if grasp_id:
                edges.append({"from": grasp_id, "to": node_id, "relation": "grasp_input"})
            if placement_id:
                edges.append(
                    {"from": placement_id, "to": node_id, "relation": "placement_input"}
                )
        return {
            "schema_version": "openeta.provenance_evidence_graph.v1",
            "nodes": nodes,
            "edges": edges,
            "inconsistencies": inconsistencies,
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

    def articulated_attachment_probe(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(ARTICULATED_ATTACHMENT_PROBE_KEY))

    def grasp_adjustment_budget(self) -> JsonDict | None:
        """Return the host-owned residual budget for the active compiled grasp."""

        return _memory_fact_value(self.facts.get(GRASP_ADJUSTMENT_BUDGET_KEY))

    def gripper_command_state(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(GRIPPER_COMMAND_STATE_KEY))

    def attachment_evidence(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(ATTACHMENT_EVIDENCE_KEY))

    def placement_world_reference(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(PLACEMENT_WORLD_REFERENCE_KEY))

    def motion_reconciliation(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(MOTION_RECONCILIATION_KEY))

    def ik_preview_receipts(self) -> JsonDict | None:
        """Return the bounded host index of recent endpoint previews."""

        return _memory_fact_value(self.facts.get(IK_PREVIEW_RECEIPTS_KEY))

    def resolve_compiled_grasp_pose_reference(
        self,
        *,
        compiled_grasp_id: str,
        waypoint_role: str,
    ) -> JsonDict:
        """Resolve an exact compiled waypoint without model-side pose copying.

        The Agent still chooses the compiled grasp and waypoint.  The host only
        expands that explicit reference to the immutable pose already present in
        the evidence graph; it does not choose a task stage or a fallback.
        """

        compiled_id = str(compiled_grasp_id or "").strip()
        requested_role = str(waypoint_role or "").strip()
        if not compiled_id:
            raise ValueError("compiled_grasp_id is required")
        if not requested_role:
            raise ValueError("waypoint_role is required")
        role = _compiled_grasp_pose_role({"waypoint_role": requested_role})
        if not role:
            raise ValueError(
                "waypoint_role must be grasp_clearance, grasp_precontact, "
                "grasp_alignment_reference, or grasp_contact"
            )
        compiled = self._compiled_grasp_artifact(compiled_id)
        if not isinstance(compiled, dict):
            available = [
                str(value.get("compiled_grasp_id") or "")
                for entry in reversed(list(self.artifacts.values()))
                if isinstance(entry, dict)
                and isinstance((value := entry.get("value")), dict)
                and value.get("type") == "compiled_grasp"
                and value.get("compiled_grasp_id")
            ][:4]
            raise ValueError(
                f"compiled grasp {compiled_id!r} does not exist; "
                f"available recent ids={available!r}"
            )
        artifact_epoch = _optional_int(compiled.get("scene_epoch"), default=-1)
        if artifact_epoch != self.object_scene_epoch():
            raise ValueError(
                f"compiled grasp {compiled_id!r} belongs to object_scene_epoch "
                f"{artifact_epoch}, current={self.object_scene_epoch()}"
            )
        pose = _compiled_grasp_reference_pose(compiled, role=role)
        if not isinstance(pose, dict):
            raise ValueError(
                f"compiled grasp {compiled_id!r} has no pose for {requested_role!r}"
            )
        return {
            "schema_version": "openeta.compiled_grasp_pose_resolution.v1",
            "compiled_grasp_id": compiled_id,
            "waypoint_role": requested_role,
            "parameters": {"target_pose": dict(pose)},
        }

    def resolve_compiled_grasp_path_sample_reference(
        self,
        *,
        compiled_grasp_id: str,
        path_fraction: float,
    ) -> JsonDict:
        """Resolve an Agent-chosen sample on a compiled clearance-contact line.

        This is deterministic geometry expansion, not a task phase: the Agent
        decides whether samples are useful, how many to request, and their
        fractions. Every returned pose still requires an independent IK preview.
        """

        compiled_id = str(compiled_grasp_id or "").strip()
        if not compiled_id:
            raise ValueError("compiled_grasp_id is required")
        if isinstance(path_fraction, bool) or not isinstance(path_fraction, int | float):
            raise ValueError("path_fraction must be a finite number strictly between 0 and 1")
        fraction = float(path_fraction)
        if not math.isfinite(fraction) or not 0.0 < fraction < 1.0:
            raise ValueError("path_fraction must be a finite number strictly between 0 and 1")
        compiled = self._compiled_grasp_artifact(compiled_id)
        if not isinstance(compiled, dict):
            raise ValueError(f"compiled grasp {compiled_id!r} does not exist")
        artifact_epoch = _optional_int(compiled.get("scene_epoch"), default=-1)
        if artifact_epoch != self.object_scene_epoch():
            raise ValueError(
                f"compiled grasp {compiled_id!r} belongs to object_scene_epoch "
                f"{artifact_epoch}, current={self.object_scene_epoch()}"
            )
        clearance = _compiled_grasp_reference_pose(compiled, role="clearance")
        contact = _compiled_grasp_reference_pose(compiled, role="contact")
        if not isinstance(clearance, dict) or not isinstance(contact, dict):
            raise ValueError(
                f"compiled grasp {compiled_id!r} lacks clearance/contact endpoints"
            )
        clearance_xyz = clearance.get("xyz")
        contact_xyz = contact.get("xyz")
        if not _finite_xyz(clearance_xyz) or not _finite_xyz(contact_xyz):
            raise ValueError(
                f"compiled grasp {compiled_id!r} has non-finite path endpoints"
            )
        pose = dict(contact)
        pose.update(
            {
                "xyz": [
                    round(
                        float(clearance_xyz[index])
                        + fraction
                        * (float(contact_xyz[index]) - float(clearance_xyz[index])),
                        12,
                    )
                    for index in range(3)
                ],
                "waypoint_role": "grasp_path_sample",
                "path_fraction": fraction,
                "segment_start_role": "grasp_clearance",
                "segment_end_role": "grasp_contact",
            }
        )
        return {
            "schema_version": "openeta.compiled_grasp_path_sample_resolution.v1",
            "compiled_grasp_id": compiled_id,
            "path_fraction": fraction,
            "parameters": {"target_pose": pose},
        }

    def resolve_attachment_probe_waypoint(
        self,
        *,
        probe_id: str,
        waypoint_index: int,
    ) -> JsonDict:
        """Resolve one exact frozen probe waypoint from a short public reference.

        The Agent selects the probe and ordered waypoint.  The host validates the
        current evidence epochs and expands that reference to the immutable pose;
        it does not choose a motion or advance a manipulation stage.
        """

        requested_probe = str(probe_id or "").strip()
        if not requested_probe:
            raise ValueError("probe_id is required")
        if isinstance(waypoint_index, bool) or not isinstance(waypoint_index, int):
            raise ValueError("waypoint_index must be a zero-based integer")
        probe = self.articulated_attachment_probe()
        if not isinstance(probe, dict):
            raise ValueError(
                f"attachment probe {requested_probe!r} does not exist; no prepared "
                "probe is available"
            )
        available_probe = str(probe.get("probe_id") or "")
        if requested_probe != available_probe:
            raise ValueError(
                f"attachment probe {requested_probe!r} does not exist; active "
                f"probe_id={available_probe!r}"
            )
        if probe.get("status") != "prepared":
            raise ValueError(
                f"attachment probe {requested_probe!r} is {probe.get('status')!r}; "
                "only a prepared, unexecuted probe can be previewed"
            )
        probe_object_epoch = _fact_epoch_value(
            probe.get("object_scene_epoch", probe.get("scene_epoch"))
        )
        if probe_object_epoch != self.object_scene_epoch():
            raise ValueError(
                f"attachment probe {requested_probe!r} is stale: probe "
                f"object_scene_epoch={probe_object_epoch}, current="
                f"{self.object_scene_epoch()}"
            )
        probe_robot_epoch = _fact_epoch_value(probe.get("robot_motion_epoch"))
        if probe_robot_epoch != self.robot_motion_epoch():
            raise ValueError(
                f"attachment probe {requested_probe!r} is stale: probe "
                f"robot_motion_epoch={probe_robot_epoch}, current="
                f"{self.robot_motion_epoch()}"
            )
        frozen_path = probe.get("frozen_path")
        if not isinstance(frozen_path, list) or not frozen_path:
            raise ValueError(
                f"attachment probe {requested_probe!r} has no frozen waypoints"
            )
        if not 0 <= waypoint_index < len(frozen_path):
            raise ValueError(
                f"waypoint_index {waypoint_index} is out of range for attachment "
                f"probe {requested_probe!r}; valid indices="
                f"{list(range(len(frozen_path)))!r}"
            )
        pose = frozen_path[waypoint_index]
        if not isinstance(pose, dict) or not _finite_xyz(
            pose.get("xyz", pose.get("translation_xyz"))
        ):
            raise ValueError(
                f"attachment probe {requested_probe!r} waypoint {waypoint_index} "
                "has no executable target_pose"
            )
        marker = str(probe.get("path_sha256") or "")
        if not marker or str(pose.get("probe_path_sha256") or "") != marker:
            raise ValueError(
                f"attachment probe {requested_probe!r} waypoint {waypoint_index} "
                "does not match the frozen path digest"
            )
        return {
            "schema_version": "openeta.attachment_probe_waypoint_resolution.v1",
            "probe_id": requested_probe,
            "waypoint_index": waypoint_index,
            "waypoint_count": len(frozen_path),
            "path_sha256": marker,
            "parameters": {"target_pose": dict(pose)},
        }

    def resolve_ik_motion_reference(self, ik_receipt_id: str) -> JsonDict:
        """Expand one exact IK receipt into move parameters.

        This prevents a planner from having to reproduce xyz/orientation arrays
        between preview and execution.  Freshness and authorization are still
        checked by the ordinary motion gates after expansion.
        """

        requested_id = str(ik_receipt_id or "").strip()
        if not requested_id:
            raise ValueError("ik_receipt_id is required")
        state = self.ik_preview_receipts() or {}
        receipts = [
            item for item in state.get("receipts", []) if isinstance(item, dict)
        ]
        receipt = next(
            (
                item
                for item in reversed(receipts)
                if str(item.get("receipt_id") or "") == requested_id
            ),
            None,
        )
        if not isinstance(receipt, dict):
            available = [
                str(item.get("receipt_id") or "")
                for item in reversed(receipts)
                if item.get("receipt_id")
            ][:6]
            raise ValueError(
                f"IK receipt {requested_id!r} does not exist; "
                f"available recent ids={available!r}"
            )
        receipt_object_epoch = _fact_epoch_value(receipt.get("object_scene_epoch"))
        receipt_robot_epoch = _fact_epoch_value(receipt.get("robot_motion_epoch"))
        if (
            receipt_object_epoch != self.object_scene_epoch()
            or receipt_robot_epoch != self.robot_motion_epoch()
        ):
            raise ValueError(
                f"IK receipt {requested_id!r} is stale: receipt epochs "
                f"object={receipt_object_epoch}, robot={receipt_robot_epoch}; current "
                f"object={self.object_scene_epoch()}, robot={self.robot_motion_epoch()}"
            )
        target_pose = receipt.get("target_pose")
        if not isinstance(target_pose, dict) or not _finite_xyz(
            target_pose.get("xyz", target_pose.get("translation_xyz"))
        ):
            raise ValueError(
                f"IK receipt {requested_id!r} has no executable target_pose"
            )
        parameters: JsonDict = {
            "target_pose": dict(target_pose),
            "ik_receipt_id": requested_id,
        }
        parameters["preserve_current_orientation"] = (
            str(receipt.get("orientation_policy") or "") == "preserve_current"
        )
        tolerances = receipt.get("tolerances")
        tolerances = tolerances if isinstance(tolerances, dict) else {}
        if tolerances.get("position_tolerance_m") is not None:
            parameters["tolerance"] = tolerances["position_tolerance_m"]
        if tolerances.get("orientation_tolerance_rad") is not None:
            parameters["ori_tolerance"] = tolerances["orientation_tolerance_rad"]
        return {
            "schema_version": "openeta.ik_motion_reference_resolution.v1",
            "ik_receipt_id": requested_id,
            "parameters": parameters,
        }

    def resolve_ik_trajectory_reference(
        self,
        ik_receipt_ids: object,
    ) -> JsonDict:
        """Resolve 1-5 checked endpoints into one atomic EEF trajectory."""

        if not isinstance(ik_receipt_ids, list) or not 1 <= len(ik_receipt_ids) <= 5:
            raise ValueError("ik_receipt_ids must contain between one and five ids")
        if any(not isinstance(value, str) or not value.strip() for value in ik_receipt_ids):
            raise ValueError("every ik_receipt_ids entry must be a non-empty string")
        ids = [str(value).strip() for value in ik_receipt_ids]
        if len(set(ids)) != len(ids):
            raise ValueError("ik_receipt_ids must not contain duplicates")
        resolutions = [self.resolve_ik_motion_reference(value) for value in ids]
        trajectory = [
            dict(resolution["parameters"]["target_pose"])
            for resolution in resolutions
        ]
        return {
            "schema_version": "openeta.ik_trajectory_reference_resolution.v1",
            "ik_receipt_ids": ids,
            "parameters": {
                "trajectory": trajectory,
                "ik_receipt_ids": ids,
            },
        }

    def resolve_ik_trajectory_execution_bundle(
        self,
        parameters: object,
    ) -> JsonDict | None:
        """Build a host-private exact-route bundle for condition C.

        The Agent remains responsible only for choosing ordered receipt IDs.
        This bundle preserves each receipt's captured orientation and source
        epochs so the simulator can re-preview every endpoint just in time from
        the actual state reached by the previous segment.  It is not exposed in
        the Agent-facing tool schema and does not claim path-collision coverage.
        """

        if not isinstance(parameters, dict):
            return None
        ids = parameters.get("ik_receipt_ids")
        if not isinstance(ids, list) or not ids:
            return None
        # Reuse the ordinary resolver to enforce existence, freshness, endpoint
        # authorization, duplicate rejection, and the current epoch contract.
        resolution = self.resolve_ik_trajectory_reference(ids)
        resolved = resolution.get("parameters")
        trajectory = resolved.get("trajectory") if isinstance(resolved, dict) else None
        if not isinstance(trajectory, list) or len(trajectory) != len(ids):
            return None

        state = self.ik_preview_receipts() or {}
        by_id = {
            str(receipt.get("receipt_id") or ""): receipt
            for receipt in state.get("receipts", []) or []
            if isinstance(receipt, dict) and receipt.get("receipt_id")
        }
        entries: list[JsonDict] = []
        for index, (receipt_id, resolved_pose) in enumerate(zip(ids, trajectory)):
            receipt = by_id.get(str(receipt_id))
            if not isinstance(receipt, dict) or not isinstance(resolved_pose, dict):
                return None
            pose = dict(resolved_pose)
            reachability = receipt.get("reachability")
            reachability = reachability if isinstance(reachability, dict) else {}
            checked_target = reachability.get("target")
            checked_target = checked_target if isinstance(checked_target, dict) else {}
            captured_quat = checked_target.get("quat_xyzw")
            if (
                str(receipt.get("orientation_policy") or "")
                == "preserve_current"
                and isinstance(captured_quat, list)
                and len(captured_quat) == 4
            ):
                source_captured_quat = [float(value) for value in captured_quat]
                # Preserve-current is an epoch-bound six-DoF receipt, not an
                # unconstrained rotation.  The exact orientation captured by
                # the original per-endpoint preview remains the route target;
                # condition C rechecks that exact target from each actual
                # preceding endpoint before it executes the segment.
                pose["quat_xyzw"] = list(source_captured_quat)
            else:
                source_captured_quat = None
            candidate = receipt.get("best_candidate")
            candidate = candidate if isinstance(candidate, dict) else {}
            entries.append(
                {
                    "index": index,
                    "source_ik_receipt_id": str(receipt_id),
                    "target_pose": pose,
                    "orientation_policy": str(
                        receipt.get("orientation_policy") or ""
                    ),
                    "source_captured_quat_xyzw": source_captured_quat,
                    "tolerances": dict(receipt.get("tolerances") or {}),
                    "source_object_scene_epoch": _fact_epoch_value(
                        receipt.get("object_scene_epoch")
                    ),
                    "source_robot_motion_epoch": _fact_epoch_value(
                        receipt.get("robot_motion_epoch")
                    ),
                    "source_seed_hint": {
                        "joint_positions": list(candidate.get("joint_positions") or []),
                        "joint_margin_min_rad": candidate.get(
                            "joint_margin_min_rad"
                        ),
                        "joint_travel_l2_rad": candidate.get(
                            "joint_travel_l2_rad"
                        ),
                    },
                }
            )
        return {
            "schema_version": "openeta.experimental_route_execution_bundle.v1",
            "condition": "C",
            "authority": "host_memory_exact_receipt_resolution",
            "object_scene_epoch": self.object_scene_epoch(),
            "robot_motion_epoch_at_dispatch": self.robot_motion_epoch(),
            "sequential_preview_policy": "just_in_time_from_actual_segment_end",
            "path_collision_authority": "controller_per_step_only",
            "entries": entries,
        }

    def _resolve_executed_motion_reference_for_probe(
        self,
        *,
        name: str,
        parameters: object,
    ) -> JsonDict | None:
        """Recover already-dispatched probe geometry after motion advanced epochs."""

        if not isinstance(parameters, dict):
            return None
        state = self.ik_preview_receipts() or {}
        receipts = {
            str(item.get("receipt_id") or ""): item
            for item in state.get("receipts", [])
            if isinstance(item, dict) and item.get("receipt_id")
        }
        if name == "move_to":
            receipt = receipts.get(str(parameters.get("ik_receipt_id") or ""))
            pose = receipt.get("target_pose") if isinstance(receipt, dict) else None
            if not isinstance(pose, dict):
                return None
            return {
                "target_pose": dict(pose),
                "enable_collision_check": parameters.get("enable_collision_check"),
            }
        if name == "follow_eef_trajectory":
            ids = parameters.get("ik_receipt_ids")
            if not isinstance(ids, list) or not ids:
                return None
            selected = [receipts.get(str(value or "")) for value in ids]
            if any(not isinstance(item, dict) for item in selected):
                return None
            poses = [item.get("target_pose") for item in selected]
            if any(not isinstance(pose, dict) for pose in poses):
                return None
            return {
                "trajectory": [dict(pose) for pose in poses],
                "enable_collision_check": parameters.get("enable_collision_check"),
            }
        return None

    def controller_capabilities(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get(CONTROLLER_CAPABILITIES_KEY))

    def resolve_ik_execution_seed(self, parameters: object) -> JsonDict | None:
        """Resolve a current feasible pose preview to a private joint seed.

        The seed is deliberately carried through execution metadata rather than
        exposed as an Agent-authored ``move_to`` parameter.  The worker still
        validates the joints against the requested endpoint and tolerances before
        applying them, so a stale or malformed receipt cannot silently redirect a
        motion.
        """

        if not isinstance(parameters, dict):
            return None
        signature = _ik_pose_policy_signature(parameters)
        if not signature:
            return None
        target_pose = parameters.get("target_pose")
        if not isinstance(target_pose, dict):
            return None
        orientation_fields = any(
            target_pose.get(key) is not None
            for key in (
                "rotation_matrix",
                "quat_xyzw",
                "quaternion",
                "rotvec",
                "roll",
                "pitch",
                "yaw",
            )
        )
        preserve_current = parameters.get("preserve_current_orientation")
        if preserve_current is None:
            preserve_current = not orientation_fields
        # A preserve-current preview is still a complete six-DoF constraint: the
        # host captures the then-current wrist orientation in the receipt.  The
        # exact robot-motion epoch check below guarantees that no motion occurred
        # before execution, so carrying its private joint seed is both safe and
        # necessary when the local differential controller is near a redundancy
        # or joint-limit boundary.
        if preserve_current is not True and not orientation_fields:
            return None

        state = self.ik_preview_receipts() or {}
        for receipt in reversed(state.get("receipts", []) or []):
            if not isinstance(receipt, dict):
                continue
            if (
                receipt.get("pose_policy_signature") != signature
                and not ik_pose_policy_numerically_equivalent(receipt, parameters)
            ):
                continue
            if not self._ik_receipt_authorizes_motion(receipt, parameters):
                return None
            if (
                _fact_epoch_value(receipt.get("object_scene_epoch"))
                != self.object_scene_epoch()
                or _fact_epoch_value(receipt.get("robot_motion_epoch"))
                != self.robot_motion_epoch()
            ):
                return None
            candidate = receipt.get("best_candidate")
            if not isinstance(candidate, dict):
                reachability = receipt.get("reachability")
                candidate = (
                    reachability.get("best_candidate")
                    if isinstance(reachability, dict)
                    else None
                )
            joints = candidate.get("joint_positions") if isinstance(candidate, dict) else None
            if (
                not isinstance(joints, list)
                or len(joints) != 7
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int | float)
                    or not math.isfinite(float(value))
                    for value in joints
                )
            ):
                return None
            return {
                "schema_version": "openeta.ik_execution_seed.v1",
                "receipt_id": str(receipt.get("receipt_id") or ""),
                "pose_policy_signature": signature,
                "joint_positions": [float(value) for value in joints],
                "preview_tolerances": dict(receipt.get("tolerances") or {}),
                "object_scene_epoch": self.object_scene_epoch(),
                "robot_motion_epoch": self.robot_motion_epoch(),
            }
        return None

    def scene_epoch(self) -> int:
        """Return the object-scene epoch (legacy ``scene_epoch`` alias)."""

        return max(
            _fact_epoch(self.facts.get(OBJECT_SCENE_EPOCH_KEY)),
            _fact_epoch(self.facts.get(SCENE_EPOCH_KEY)),
        )

    def object_scene_epoch(self) -> int:
        """Return the version used to invalidate object-relative evidence."""

        return self.scene_epoch()

    def robot_motion_epoch(self) -> int:
        """Return the version of robot/camera configuration changes."""

        return _fact_epoch(self.facts.get(ROBOT_MOTION_EPOCH_KEY))

    def transition_ledger(self) -> list[JsonDict]:
        value = _memory_fact_value(self.facts.get(TRANSITION_LEDGER_KEY)) or {}
        rows = value.get("rows")
        return (
            [dict(row) for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []
        )

    def latest_environment_receipt(self) -> JsonDict | None:
        return _memory_fact_value(self.facts.get("latest_environment_receipt"))

    def world_evidence_context(self) -> JsonDict:
        """Derive planner evidence with provenance and explicit freshness.

        This is a projection of the event-backed runtime facts, not another
        mutable task-state store. Values that refer to an older scene epoch are
        exposed as stale rather than silently treated as the current world.
        """

        specs = (
            ("selected_target", SELECTED_SAM3_DETECTION_KEY, "perception_result"),
            ("target_asset_reference", TARGET_ASSET_REFERENCE_KEY, "reference_result"),
            ("target_identity_anchor", TARGET_IDENTITY_ANCHOR_KEY, "identity_anchor"),
            ("gripper_command", GRIPPER_COMMAND_STATE_KEY, "commanded_state"),
            ("attachment_verdict", ATTACHMENT_EVIDENCE_KEY, "verifier_result"),
            (
                "placement_world_reference",
                PLACEMENT_WORLD_REFERENCE_KEY,
                "tool_geometry_reference",
            ),
            ("motion_reconciliation", MOTION_RECONCILIATION_KEY, "safety_constraint"),
            ("ik_preview_receipts", IK_PREVIEW_RECEIPTS_KEY, "safety_evidence"),
            ("environment_receipt", "latest_environment_receipt", "trusted_receipt"),
        )
        current_epoch = self.scene_epoch()
        evidence: JsonDict = {}
        for output_key, fact_key, evidence_kind in specs:
            entry = self.facts.get(fact_key)
            if not isinstance(entry, dict):
                continue
            value = _memory_fact_value(entry)
            if not isinstance(value, dict):
                continue
            value_epoch = value.get("scene_epoch")
            if evidence_kind == "trusted_receipt":
                freshness = "authoritative_receipt"
            elif evidence_kind == "commanded_state":
                freshness = "commanded_not_observed"
            elif isinstance(value_epoch, int) and not isinstance(value_epoch, bool):
                freshness = (
                    "current_scene_epoch" if value_epoch == current_epoch else "stale_scene_epoch"
                )
            else:
                freshness = "session_evidence"
            evidence[output_key] = {
                "value": value,
                "provenance": {
                    "source": str(entry.get("source") or "runtime"),
                    "timestamp_s": entry.get("timestamp_s"),
                    "evidence_kind": evidence_kind,
                },
                "freshness": freshness,
                "scene_epoch": value_epoch,
                "current_scene_epoch": current_epoch,
            }
        placement = self.selected_sam3_detection("placement_region")
        selections_entry = self.facts.get(SELECTED_SAM3_DETECTIONS_KEY)
        if isinstance(placement, dict) and isinstance(selections_entry, dict):
            placement_epoch = placement.get("scene_epoch")
            placement_freshness = (
                "current_scene_epoch"
                if isinstance(placement_epoch, int)
                and not isinstance(placement_epoch, bool)
                and placement_epoch == current_epoch
                else "stale_scene_epoch"
                if isinstance(placement_epoch, int)
                and not isinstance(placement_epoch, bool)
                else "session_evidence"
            )
            evidence["placement_region"] = {
                "value": placement,
                "provenance": {
                    "source": str(selections_entry.get("source") or "runtime"),
                    "timestamp_s": selections_entry.get("timestamp_s"),
                    "evidence_kind": "perception_result",
                },
                "freshness": placement_freshness,
                "scene_epoch": placement_epoch,
                "current_scene_epoch": current_epoch,
            }
        candidate_entry = max(
            (
                entry
                for entry in self.artifacts.values()
                if isinstance(entry, dict)
                and isinstance(entry.get("value"), dict)
                and entry["value"].get("type") == "grasp_candidates"
            ),
            key=lambda entry: float(entry.get("timestamp_s") or 0.0),
            default=None,
        )
        if isinstance(candidate_entry, dict):
            candidate_value = dict(candidate_entry["value"])
            value_epoch = candidate_value.get("scene_epoch")
            evidence["grasp_candidates"] = {
                "value": candidate_value,
                "provenance": {
                    "source": str(
                        candidate_value.get("source_tool")
                        or candidate_entry.get("source")
                        or "tool_result"
                    ),
                    "timestamp_s": candidate_entry.get("timestamp_s"),
                    "evidence_kind": "tool_result",
                },
                "freshness": (
                    "current_scene_epoch"
                    if value_epoch == current_epoch
                    else "stale_scene_epoch"
                ),
                "scene_epoch": value_epoch,
                "current_scene_epoch": current_epoch,
            }
        return evidence

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

    def compiled_grasp_target_gate_error(
        self,
        *,
        tool_name: str,
        parameters: JsonDict,
    ) -> str | None:
        """Protect contact/close provenance without imposing a task phase."""

        release_error = self._failed_attached_motion_release_gate_error(
            tool_name=tool_name,
            parameters=parameters,
        )
        if release_error:
            return release_error

        clearance_error = self._compiled_contact_clearance_gate_error(
            tool_name=tool_name,
            parameters=parameters,
        )
        if clearance_error:
            return clearance_error

        approach_error = self._compiled_contact_approach_gate_error(
            tool_name=tool_name,
            parameters=parameters,
        )
        if approach_error:
            return approach_error

        orientation_error = self._compiled_contact_orientation_gate_error(
            tool_name=tool_name,
            parameters=parameters,
        )
        if orientation_error:
            return orientation_error

        adjustment_error = self._compiled_grasp_adjustment_gate_error(
            tool_name=tool_name,
            parameters=parameters,
        )
        if adjustment_error:
            return adjustment_error

        contact_error = self._compiled_contact_close_gate_error(
            tool_name=tool_name,
            parameters=parameters,
        )
        if contact_error:
            return contact_error

        graph = self.provenance_evidence_graph()
        mismatch = next(
            (
                item
                for item in graph.get("inconsistencies", [])
                if isinstance(item, dict)
                and item.get("code") == "compiled_grasp_target_superseded"
            ),
            None,
        )
        if not isinstance(mismatch, dict):
            return None
        unsafe = False
        if tool_name == "gripper_control":
            try:
                unsafe = int(parameters.get("position")) == 0
            except (TypeError, ValueError):
                unsafe = False
        elif tool_name in {"move_to", "follow_eef_trajectory"}:
            compiled_id = str(mismatch.get("compiled_grasp_id") or "")
            node = next(
                (
                    item
                    for item in graph.get("nodes", [])
                    if isinstance(item, dict)
                    and item.get("kind") == "compiled_targeted_grasp"
                    and str(item.get("compiled_grasp_id") or "") == compiled_id
                ),
                {},
            )
            contact = node.get("contact_pose") if isinstance(node, dict) else None
            contact_xyz = contact.get("xyz") if isinstance(contact, dict) else None
            poses = (
                [parameters.get("target_pose")]
                if tool_name == "move_to"
                else parameters.get("trajectory", [])
            )
            unsafe = isinstance(poses, list) and any(
                _pose_near_xyz(pose, contact_xyz, tolerance_m=0.08) for pose in poses
            )
        if not unsafe:
            return None
        bundle = self.grasp_input_bundle()
        bundle_id = bundle.get("bundle_id") if isinstance(bundle, dict) else None
        recovery = (
            f" Call grasp_pose_estimate with bundle_id={bundle_id!r}, then compile a "
            "current candidate before contact or gripper close."
            if isinstance(bundle_id, str) and bundle_id
            else " Re-segment, estimate, and compile a current grasp before contact."
        )
        return (
            "compiled_grasp_target_superseded: compiled grasp "
            f"{mismatch.get('compiled_grasp_id')!r} is bound to target evidence "
            f"{mismatch.get('compiled_target_evidence_id')!r}, while the current "
            f"selected target is {mismatch.get('current_target_evidence_id')!r}. "
            "The old contact/close action was rejected; safe retreat and clearance "
            "waypoints remain allowed." + recovery
        )

    def latest_compiled_contact_execution(self) -> JsonDict | None:
        return _memory_fact_value(
            self.facts.get(LATEST_COMPILED_CONTACT_EXECUTION_KEY)
        )

    def latest_compiled_clearance_execution(self) -> JsonDict | None:
        return _memory_fact_value(
            self.facts.get(LATEST_COMPILED_CLEARANCE_EXECUTION_KEY)
        )

    def _compiled_contact_clearance_gate_error(
        self,
        *,
        tool_name: str,
        parameters: JsonDict,
    ) -> str | None:
        """Prevent an explicitly failed clearance receipt authorizing contact.

        Clearance remains optional and Agent-chosen. This check only applies when
        the Agent already attempted the clearance waypoint for the same compiled
        grasp and immediately depends on that current-robot-state receipt.
        """

        if tool_name != "move_to":
            return None
        target_pose = parameters.get("target_pose")
        if not isinstance(target_pose, dict) or _compiled_grasp_pose_role(target_pose) != "contact":
            return None
        compiled_id = str(target_pose.get("compiled_grasp_id") or "")
        receipt = self.latest_compiled_clearance_execution()
        if (
            not compiled_id
            or not isinstance(receipt, dict)
            or str(receipt.get("compiled_grasp_id") or "") != compiled_id
        ):
            return None
        if (
            _fact_epoch_value(receipt.get("object_scene_epoch"))
            != self.object_scene_epoch()
            or _fact_epoch_value(receipt.get("robot_motion_epoch"))
            != self.robot_motion_epoch()
        ):
            return None
        if receipt.get("reached_target") is True:
            return None
        position_error = receipt.get("position_error_m")
        orientation_error = receipt.get("orientation_error_rad")
        if (
            _finite_number(position_error)
            and _finite_number(orientation_error)
            and float(position_error) <= COMPILED_CLEARANCE_POSITION_TOLERANCE_M
            and float(orientation_error)
            <= COMPILED_CLEARANCE_ORIENTATION_TOLERANCE_RAD
        ):
            return None
        return (
            "compiled_clearance_not_reached: contact motion was rejected because "
            f"the current clearance execution for {compiled_id!r} explicitly reports "
            f"reached_target=false, stop_reason={receipt.get('stop_reason')!r}, "
            f"steps_executed={receipt.get('steps_executed')!r}, "
            f"actual_eef_xyz={receipt.get('actual_xyz')!r}, "
            f"position_error_m={receipt.get('position_error_m')!r}, "
            f"orientation_error_rad={receipt.get('orientation_error_rad')!r}. "
            "The clearance residual acceptance envelope is "
            f"position<={COMPILED_CLEARANCE_POSITION_TOLERANCE_M:.3f}m and "
            f"orientation<={COMPILED_CLEARANCE_ORIENTATION_TOLERANCE_RAD:.2f}rad. "
            "A failed execution receipt cannot serve as a successful prerequisite. "
            "Inspect the fresh agentview/wrist images and this receipt, then adjust "
            "the waypoint/orientation, choose another candidate, or deliberately "
            "replan from the measured EEF pose. Clearance is not a mandatory task "
            "stage; this rejection only preserves the causal meaning of the "
            "Agent-chosen failed waypoint."
        )

    def _compiled_contact_approach_gate_error(
        self,
        *,
        tool_name: str,
        parameters: JsonDict,
    ) -> str | None:
        """Reject a long, cross-axis sweep into a compiled contact anchor.

        Endpoint IK and collision checks cannot establish that the final motion
        enters an object between the fingers. A controller can reach the exact
        contact xyz while the palm or one finger first pushes the object away.
        For motions already inside the near-field radius, Agent-authored visual
        corrections remain unrestricted by this check. Longer motions only need
        to start inside a tube behind the contact pose along the compiled approach
        vector; no semantic waypoint or task phase is required.
        """

        if tool_name != "move_to":
            return None
        target_pose = parameters.get("target_pose")
        if (
            not isinstance(target_pose, dict)
            or _compiled_grasp_pose_role(target_pose) != "contact"
        ):
            return None
        compiled_id = str(target_pose.get("compiled_grasp_id") or "")
        compiled = self._compiled_grasp_artifact(compiled_id)
        if not compiled_id or not isinstance(compiled, dict):
            return None
        current_xyz = self._latest_observed_eef_xyz()
        target_xyz = target_pose.get("xyz")
        approach = compiled.get("approach_world_xyz")
        if (
            not _finite_xyz(current_xyz)
            or not _finite_xyz(target_xyz)
            or not _finite_xyz(approach)
        ):
            return None

        direction = [float(value) for value in approach[:3]]
        direction_norm = math.sqrt(sum(value * value for value in direction))
        if direction_norm <= 1e-9:
            return None
        direction = [value / direction_norm for value in direction]
        delta = [
            float(target_xyz[index]) - float(current_xyz[index]) for index in range(3)
        ]
        travel = math.sqrt(sum(value * value for value in delta))
        if travel <= COMPILED_CONTACT_NEAR_FIELD_RADIUS_M:
            return None
        axial = sum(delta[index] * direction[index] for index in range(3))
        lateral = math.sqrt(max(0.0, travel * travel - axial * axial))
        cosine = axial / travel if travel > 1e-9 else 1.0
        if (
            axial > 0.0
            and lateral <= COMPILED_CONTACT_APPROACH_CORRIDOR_RADIUS_M
            and cosine >= COMPILED_CONTACT_APPROACH_MIN_COSINE
        ):
            return None

        hover_pose = compiled.get("hover_pose")
        hover_xyz = hover_pose.get("xyz") if isinstance(hover_pose, dict) else None
        return (
            "compiled_contact_approach_misaligned: the long final contact motion was "
            "rejected because endpoint reachability and collision clearance do not "
            "prove safe object-relative entry. "
            f"current_eef_xyz={_rounded_xyz(current_xyz)}, "
            f"contact_xyz={_rounded_xyz(target_xyz)}, "
            f"approach_world_xyz={_rounded_xyz(direction)}, travel_m={travel:.4f}, "
            f"axial_progress_m={axial:.4f}, lateral_offset_m={lateral:.4f} "
            f"(limit={COMPILED_CONTACT_APPROACH_CORRIDOR_RADIUS_M:.3f}), "
            f"direction_cosine={cosine:.4f} "
            f"(minimum={COMPILED_CONTACT_APPROACH_MIN_COSINE:.2f}). "
            "Move first to any IK- and collision-checked waypoint inside the approach "
            "corridor behind the contact pose, observe there, then execute the contact; "
            f"the compiled clearance reference is {_rounded_xyz(hover_xyz)}. This is a "
            "geometric corridor contract, not a required hover/descend task stage. "
            "If the current view invalidates this approach, choose or compile a better "
            "candidate instead of forcing the endpoint."
        )

    def _latest_observed_eef_xyz(self) -> list[float] | None:
        """Read event-derived measured EEF xyz, not host task progress."""

        for event in reversed(self.events):
            if event.event_type != "observation":
                continue
            robot = event.payload.get("robot")
            eef_pose = robot.get("end_effector_pose") if isinstance(robot, dict) else None
            xyz = eef_pose.get("xyz") if isinstance(eef_pose, dict) else None
            if _finite_xyz(xyz):
                return [float(value) for value in xyz[:3]]
        return None

    def _compiled_contact_orientation_gate_error(
        self,
        *,
        tool_name: str,
        parameters: JsonDict,
    ) -> str | None:
        """Reject contact entry while a large wrist rotation is still pending.

        The check is purely geometric.  It does not require a named alignment
        waypoint or remember task progress; it compares the latest measured EEF
        orientation with the exact contact orientation that the Agent chose.
        """

        if tool_name != "move_to":
            return None
        target_pose = parameters.get("target_pose")
        if (
            not isinstance(target_pose, dict)
            or _compiled_grasp_pose_role(target_pose) != "contact"
        ):
            return None
        current_pose: JsonDict | None = None
        for event in reversed(self.events):
            if event.event_type != "observation":
                continue
            robot = event.payload.get("robot")
            eef_pose = (
                robot.get("end_effector_pose") if isinstance(robot, dict) else None
            )
            if isinstance(eef_pose, dict):
                current_pose = eef_pose
                break
        if current_pose is None:
            return None
        current_rotation = _ik_orientation_rotation_matrix(current_pose)
        target_rotation = _ik_orientation_rotation_matrix(target_pose)
        delta_deg = _rotation_delta_deg(current_rotation, target_rotation)
        if delta_deg is None:
            return None
        delta_rad = math.radians(delta_deg)
        if delta_rad <= COMPILED_CONTACT_ENTRY_ORIENTATION_TOLERANCE_RAD:
            return None

        compiled_id = str(target_pose.get("compiled_grasp_id") or "")
        return (
            "compiled_contact_orientation_misaligned: contact motion was rejected "
            "because the gripper would enter the object while a large wrist rotation "
            "is still pending. "
            f"compiled_grasp_id={compiled_id!r}, "
            f"current_eef_quat_xyzw={current_pose.get('quat_xyzw')!r}, "
            f"orientation_delta_rad={delta_rad:.4f} "
            f"(limit={COMPILED_CONTACT_ENTRY_ORIENTATION_TOLERANCE_RAD:.2f}). "
            "At any Agent-chosen collision-clear waypoint outside contact, preview and "
            "execute the compiled orientation (or another compatible full pose), then "
            "re-observe and retry contact. If that orientation is infeasible, select a "
            "different candidate or perform a full wrist-view grasp re-estimate. This "
            "is an object-entry geometry check, not a required alignment stage."
        )

    def _compiled_contact_close_gate_error(
        self,
        *,
        tool_name: str,
        parameters: JsonDict,
    ) -> str | None:
        if tool_name != "gripper_control":
            return None
        try:
            closing = int(parameters.get("position")) == 0
        except (TypeError, ValueError):
            return None
        if not closing:
            return None
        commanded = _memory_fact_value(self.facts.get(GRIPPER_COMMAND_STATE_KEY)) or {}
        if commanded.get("position") == 0:
            return None
        provenance = _memory_fact_value(self.facts.get(GRASP_PROVENANCE_KEY)) or {}
        compiled_id = str(provenance.get("compiled_grasp_id") or "")
        if not compiled_id:
            return None
        if provenance.get("contact_geometry_invalidated_at_s"):
            return (
                "compiled_contact_geometry_invalidated: gripper close was rejected "
                f"because active compiled grasp {compiled_id!r} was invalidated by "
                f"{provenance.get('contact_geometry_invalidated_by')!r}. Re-observe, "
                "select a current grasp candidate, compile new contact geometry, and "
                "execute its contact before closing. Identity lineage remains available, "
                "but released or abandoned contact coordinates are not authorization."
            )
        receipt = self.latest_compiled_contact_execution()
        if not isinstance(receipt, dict):
            return (
                "compiled_contact_receipt_missing: gripper close was rejected because "
                f"active compiled grasp {compiled_id!r} has no executed contact receipt. "
                "Preview and execute a current contact pose first; safe observation, "
                "clearance motion, target correction, or opening the gripper remain allowed."
            )
        receipt_id = str(receipt.get("compiled_grasp_id") or "")
        if receipt_id != compiled_id:
            receipt_target_id = str(receipt.get("target_evidence_id") or "")
            active_target_id = str(provenance.get("target_evidence_id") or "")
            receipt_anchor_id = str(receipt.get("target_identity_anchor_id") or "")
            active_anchor_id = str(provenance.get("target_identity_anchor_id") or "")
            same_target = bool(
                receipt_target_id
                and active_target_id
                and receipt_target_id == active_target_id
            ) or bool(
                receipt_anchor_id
                and active_anchor_id
                and receipt_anchor_id == active_anchor_id
            )
            if not same_target:
                return (
                    "compiled_contact_receipt_mismatch: gripper close was rejected because "
                    f"the latest executed contact belongs to {receipt_id!r} "
                    f"(target={receipt_target_id or receipt_anchor_id or None!r}), while the "
                    f"latest planning provenance is {compiled_id!r} "
                    f"(target={active_target_id or active_anchor_id or None!r}). Close binds "
                    "to the latest physically executed contact branch; re-observe and execute "
                    "a contact for the intended target before closing."
                )
        receipt_object_epoch = _fact_epoch_value(receipt.get("object_scene_epoch"))
        receipt_robot_epoch = _fact_epoch_value(receipt.get("robot_motion_epoch"))
        if (
            receipt_object_epoch != self.object_scene_epoch()
            or receipt_robot_epoch != self.robot_motion_epoch()
        ):
            return (
                "compiled_contact_receipt_stale: gripper close was rejected because the "
                f"contact receipt epochs object={receipt_object_epoch}, robot={receipt_robot_epoch} "
                f"do not match current object={self.object_scene_epoch()}, "
                f"robot={self.robot_motion_epoch()}. Re-observe and execute a current "
                "contact motion; do not treat a prior endpoint as the current robot state."
            )
        position_error = (
            receipt.get("max_axis_position_error_m")
            if _finite_number(receipt.get("max_axis_position_error_m"))
            else receipt.get("position_error_m")
        )
        orientation_error = receipt.get("orientation_error_rad")
        position_within_close_envelope = bool(
            (
                _finite_number(position_error)
                and float(position_error) <= COMPILED_CONTACT_CLOSE_POSITION_TOLERANCE_M
            )
            or (
                receipt.get("reached_target") is True
                and not _finite_number(position_error)
            )
        )
        orientation_within_close_envelope = bool(
            not _finite_number(orientation_error)
            or float(orientation_error)
            <= COMPILED_CONTACT_CLOSE_ORIENTATION_TOLERANCE_RAD
        )
        if position_within_close_envelope and orientation_within_close_envelope:
            # Collision belongs to the preceding arm motion. Closing actuates only
            # the fingers, so that historical motion verdict stays visible but
            # cannot by itself veto gripper_control.
            return None
        actual_xyz = receipt.get("actual_xyz")
        return (
            "compiled_contact_not_reached: gripper close was rejected because the latest "
            f"contact execution for {receipt_id!r} is outside the independent close "
            f"contact envelope: reached_target={str(receipt.get('reached_target')).lower()}, "
            f"stop_reason={receipt.get('stop_reason')!r}, "
            f"steps_executed={receipt.get('steps_executed')!r}, "
            f"actual_eef_xyz={actual_xyz!r}, position_error_m={position_error!r} "
            f"(limit={COMPILED_CONTACT_CLOSE_POSITION_TOLERANCE_M}), "
            f"orientation_error_rad={orientation_error!r} "
            f"(limit={COMPILED_CONTACT_CLOSE_ORIENTATION_TOLERANCE_RAD}). "
            "Collision diagnostics from the preceding arm motion do not block the "
            "finger-only close command. Inspect the fresh dual-view evidence, then adjust "
            "or re-estimate only when the current EEF contact geometry is inadequate."
        )

    def _failed_attached_motion_release_gate_error(
        self,
        *,
        tool_name: str,
        parameters: JsonDict,
    ) -> str | None:
        """Do not treat a failed carrying motion as a reached release pose."""

        if tool_name != "gripper_control":
            return None
        try:
            opening = int(parameters.get("position")) == 1
        except (TypeError, ValueError):
            return None
        if not opening:
            return None
        commanded = self.gripper_command_state() or {}
        attachment = self.attachment_evidence() or {}
        if commanded.get("position") != 0 or attachment.get("verdict") != "PASS":
            return None
        latest_motion: JsonDict | None = None
        for row in reversed(self.transition_ledger()):
            if row.get("tool") not in {"move_to", "follow_eef_trajectory"}:
                continue
            latest_motion = row
            break
        if not isinstance(latest_motion, dict) or latest_motion.get("verdict") != "FAIL":
            return None
        motion_evidence: JsonDict = {}
        for event in reversed(self.events):
            if event.event_type != "action":
                continue
            command = event.payload.get("command")
            request = command.get("request") if isinstance(command, dict) else None
            if not isinstance(request, dict) or request.get("name") not in {
                "move_to",
                "follow_eef_trajectory",
            }:
                continue
            calls = command.get("tool_calls")
            call = calls[0] if isinstance(calls, list) and calls else None
            result = call.get("result") if isinstance(call, dict) else None
            details = result.get("details") if isinstance(result, dict) else None
            outputs = details.get("outputs") if isinstance(details, dict) else None
            motion_evidence = dict(outputs) if isinstance(outputs, dict) else {}
            break
        motion_summary = motion_evidence.get("motion_summary")
        motion_summary = motion_summary if isinstance(motion_summary, dict) else {}
        collision = motion_summary.get("collision")
        collision = collision if isinstance(collision, dict) else {}
        pose_feedback = motion_evidence.get("pose_feedback")
        pose_feedback = pose_feedback if isinstance(pose_feedback, dict) else {}
        actual_xyz = pose_feedback.get("actual_xyz")
        return (
            "attached_release_after_failed_motion: gripper open was rejected because "
            "independent evidence says the object is attached, while the latest carrying "
            "motion explicitly failed and did not reach its requested endpoint. "
            f"motion_tool={latest_motion.get('tool')!r}, "
            f"motion_verdict={latest_motion.get('verdict')!r}, "
            f"actual_eef_xyz={actual_xyz!r}, "
            f"collision_geometry={[collision.get('geom1_name'), collision.get('geom2_name')]!r}. "
            "Plan and execute a distinct collision-free pose from the actual EEF position "
            "before releasing; a successful subsequent carrying motion clears this check. "
            "The Agent remains free to choose the new waypoint and release pose."
        )

    def _compiled_grasp_adjustment_gate_error(
        self,
        *,
        tool_name: str,
        parameters: JsonDict,
    ) -> str | None:
        """Bound Agent-authored residuals around a compiled grasp reference.

        This is an evidence/safety invariant, not a task-stage policy.  The
        Agent supplies the final absolute world pose and preserves the compiled
        pose provenance; the host derives every residual and owns the budget.
        """

        if tool_name != "move_to":
            return None
        target_pose = parameters.get("target_pose")
        if not isinstance(target_pose, dict):
            return None
        compiled_id = str(target_pose.get("compiled_grasp_id") or "")
        role = _compiled_grasp_pose_role(target_pose)
        if not compiled_id or not role:
            return None
        compiled = self._compiled_grasp_artifact(compiled_id)
        if not isinstance(compiled, dict):
            return (
                "compiled_grasp_adjustment_unresolved: move_to references compiled grasp "
                f"{compiled_id!r}, but its host-owned pose evidence is unavailable. "
                "Recompile from current grasp evidence instead of inventing an anchor."
            )
        current_compiled = self._latest_compiled_grasp_artifact()
        current_id = str(current_compiled.get("compiled_grasp_id") or "")
        if current_id and current_id != compiled_id:
            return (
                "compiled_grasp_adjustment_superseded: move_to references compiled grasp "
                f"{compiled_id!r}, but the active compiled anchor is {current_id!r}. "
                "Use the latest compiled pose and preserve its provenance fields."
            )
        compiled_epoch = _optional_int(compiled.get("scene_epoch"), default=-1)
        if compiled_epoch != self.object_scene_epoch():
            return (
                "compiled_grasp_adjustment_stale: compiled grasp "
                f"{compiled_id!r} belongs to object_scene_epoch {compiled_epoch}, while "
                f"the current epoch is {self.object_scene_epoch()}. Re-observe and "
                "recompile before contact refinement."
            )
        reference_pose = _compiled_grasp_reference_pose(compiled, role=role)
        if not isinstance(reference_pose, dict):
            return (
                "compiled_grasp_adjustment_unresolved: compiled grasp "
                f"{compiled_id!r} has no host reference for role {role!r}."
            )
        target_xyz = target_pose.get("xyz")
        reference_xyz = reference_pose.get("xyz")
        if not _finite_xyz(target_xyz) or not _finite_xyz(reference_xyz):
            return (
                "compiled_grasp_adjustment_invalid: adjusted move_to poses require "
                "finite world-frame target_pose.xyz and compiled reference xyz values."
            )
        if str(target_pose.get("frame") or "") != "world":
            return "compiled_grasp_adjustment_invalid: adjusted poses must remain world-frame."

        reference_rotation = reference_pose.get("rotation_matrix")
        if reference_rotation is not None:
            requested_rotation = target_pose.get("rotation_matrix")
            position_only_policy = _target_pose_has_no_orientation(target_pose)
            if position_only_policy:
                if not self._has_current_feasible_ik_pose_policy(parameters):
                    return (
                        "compiled_grasp_adjustment_unverified_orientation_policy: the "
                        "requested compiled-grasp move omits an explicit orientation, "
                        "so move_to would preserve the current wrist orientation. No "
                        "current-epoch feasible IK receipt authorizes this exact xyz + "
                        "preserve_current orientation policy. Call ik_preview_check for "
                        "the same target_pose with preserve_current_orientation=true, "
                        "then retry the position-only move; alternatively provide a "
                        "rotation within the compiled anchor envelope."
                    )
                angle_deg = None
            else:
                angle_deg = _rotation_delta_deg(
                    reference_rotation,
                    requested_rotation,
                )
            if (
                not position_only_policy
                and (
                    angle_deg is None
                    or angle_deg > GRASP_REFERENCE_ORIENTATION_TOLERANCE_DEG
                )
            ):
                rendered = "unknown" if angle_deg is None else f"{angle_deg:.2f} deg"
                return (
                    "compiled_grasp_adjustment_out_of_bounds: orientation residual is "
                    f"{rendered}; preserve a valid rotation within "
                    f"{GRASP_REFERENCE_ORIENTATION_TOLERANCE_DEG:.1f} deg of the anchor."
                )

        residual = [
            float(target_xyz[index]) - float(reference_xyz[index]) for index in range(3)
        ]
        budget = self.grasp_adjustment_budget() or {}
        if str(budget.get("compiled_grasp_id") or "") == compiled_id:
            prior_residual_raw = budget.get("last_residual_xyz_m")
            prior_residual = (
                [float(value) for value in prior_residual_raw]
                if _finite_xyz(prior_residual_raw)
                else [0.0, 0.0, 0.0]
            )
            cumulative = _finite_nonnegative_float(
                budget.get("cumulative_translation_m"),
                default=0.0,
            )
        else:
            prior_residual = [0.0, 0.0, 0.0]
            cumulative = 0.0
        step = math.sqrt(
            sum((residual[index] - prior_residual[index]) ** 2 for index in range(3))
        )
        projected_cumulative = cumulative + step
        if step > GRASP_ADJUSTMENT_STEP_LIMIT_M + GRASP_ADJUSTMENT_EPSILON_M:
            return (
                "compiled_grasp_adjustment_out_of_bounds: requested residual change is "
                f"{step:.4f} m, exceeding the per-call limit of "
                f"{GRASP_ADJUSTMENT_STEP_LIMIT_M:.3f} m. Last accepted residual is "
                f"{_rounded_xyz(prior_residual)} m and requested residual is "
                f"{_rounded_xyz(residual)} m. This is an offset-from-anchor budget, "
                "not a limit on travel distance from the current EEF. To execute this "
                f"compiled waypoint, call move_to with the exact host reference xyz "
                f"{_rounded_xyz(reference_xyz)} m (or an adjustment within the residual "
                "limit); do not split the approach into residual-sized increments. If "
                "you intentionally need a far transit waypoint, omit compiled_grasp_id "
                "and waypoint_role, remain outside the contact safety envelope, and "
                "observe before using the compiled anchor."
            )
        if (
            projected_cumulative
            > GRASP_ADJUSTMENT_CUMULATIVE_LIMIT_M + GRASP_ADJUSTMENT_EPSILON_M
        ):
            remaining = max(0.0, GRASP_ADJUSTMENT_CUMULATIVE_LIMIT_M - cumulative)
            return (
                "compiled_grasp_adjustment_out_of_bounds: this increment would raise "
                f"the cumulative residual path to {projected_cumulative:.4f} m, above "
                f"the {GRASP_ADJUSTMENT_CUMULATIVE_LIMIT_M:.3f} m budget. Remaining "
                f"translation budget is {remaining:.4f} m. Re-observe and compile a "
                "new anchor if more correction is genuinely required."
            )
        return None

    def _has_current_feasible_ik_pose_policy(self, parameters: JsonDict) -> bool:
        """Return whether IK validated this exact pose policy in the current state."""

        signature = _ik_pose_policy_signature(parameters)
        if not signature:
            return False
        state = self.ik_preview_receipts() or {}
        for receipt in reversed(state.get("receipts", []) or []):
            if not isinstance(receipt, dict):
                continue
            if (
                receipt.get("pose_policy_signature") != signature
                and not ik_pose_policy_numerically_equivalent(receipt, parameters)
            ):
                continue
            if not self._ik_receipt_authorizes_motion(receipt, parameters):
                return False
            if _fact_epoch_value(receipt.get("object_scene_epoch")) != self.object_scene_epoch():
                return False
            if _fact_epoch_value(receipt.get("robot_motion_epoch")) != self.robot_motion_epoch():
                return False
            return True
        return False

    def _ik_receipt_authorizes_motion(
        self,
        receipt: JsonDict,
        parameters: JsonDict,
    ) -> bool:
        classification = str(receipt.get("classification") or "")
        if classification == "feasible":
            return True
        if classification != "kinematically_feasible_collision_deferred":
            return False
        capabilities = self.controller_capabilities() or {}
        return (
            parameters.get("enable_collision_check") is True
            and capabilities.get("motion_owns_trajectory_world_collision") is True
            and str(receipt.get("reason_code") or "")
            == "endpoint_collision_check_unavailable"
        )

    def _compiled_grasp_artifact(self, compiled_grasp_id: str) -> JsonDict | None:
        for entry in reversed(list(self.artifacts.values())):
            value = entry.get("value") if isinstance(entry, dict) else None
            if (
                isinstance(value, dict)
                and value.get("type") == "compiled_grasp"
                and str(value.get("compiled_grasp_id") or "") == compiled_grasp_id
            ):
                return value
        return None

    def resolve_compiled_contact_authorization(self, target_pose: object) -> JsonDict | None:
        """Resolve a current compiled contact pose to host-owned safety evidence.

        The returned object is carried privately by the simulator adapter.  It is
        deliberately unavailable as an Agent parameter: preserving a
        ``compiled_grasp_id`` and ``waypoint_role`` only requests resolution; it
        cannot manufacture permission to contact arbitrary scene geometry.
        """

        if not isinstance(target_pose, dict):
            return None
        compiled_id = str(target_pose.get("compiled_grasp_id") or "").strip()
        role = _compiled_grasp_pose_role(target_pose)
        if not compiled_id or role != "contact":
            return None
        compiled = self._compiled_grasp_artifact(compiled_id)
        if not isinstance(compiled, dict):
            raise ValueError(
                f"compiled grasp {compiled_id!r} is unavailable for contact authorization"
            )
        artifact_epoch = _optional_int(compiled.get("scene_epoch"), default=-1)
        if artifact_epoch != self.object_scene_epoch():
            raise ValueError(
                f"compiled grasp {compiled_id!r} belongs to object_scene_epoch "
                f"{artifact_epoch}, current={self.object_scene_epoch()}"
            )
        anchor = compiled.get("target_anchor_world_xyz")
        if not _finite_xyz(anchor):
            raise ValueError(
                f"compiled grasp {compiled_id!r} has no finite target anchor"
            )
        provenance = _memory_fact_value(self.facts.get(GRASP_PROVENANCE_KEY))
        if (
            not isinstance(provenance, dict)
            or str(provenance.get("compiled_grasp_id") or "") != compiled_id
        ):
            raise ValueError(
                f"compiled grasp {compiled_id!r} is not the active grasp provenance"
            )
        if provenance.get("contact_geometry_invalidated_at_s"):
            raise ValueError(
                f"compiled grasp {compiled_id!r} contact geometry was invalidated by "
                f"{provenance.get('contact_geometry_invalidated_by')!r}; compile a new "
                "candidate before contact, close, or attachment probing"
            )
        return {
            "schema_version": "openeta.contact_authorization.v1",
            "compiled_grasp_id": compiled_id,
            "waypoint_role": "grasp_contact",
            "target_anchor_world_xyz": [float(value) for value in anchor[:3]],
            "target_evidence_id": provenance.get("target_evidence_id"),
            "object_scene_epoch": self.object_scene_epoch(),
        }

    def resolve_active_attachment_candidate(self) -> JsonDict | None:
        """Bind close to the latest physically executed contact branch.

        A later compile is planning evidence and does not move the robot.  A
        current contact receipt therefore owns close binding when it names the
        same target as the latest planning provenance.  The simulator may use
        this to build a *tentative* carried-object collision proxy, but it is not
        attachment proof and does not create task progress.
        """

        provenance = _memory_fact_value(self.facts.get(GRASP_PROVENANCE_KEY))
        if not isinstance(provenance, dict):
            return None
        if provenance.get("contact_geometry_invalidated_at_s"):
            return None
        active_compiled_id = str(provenance.get("compiled_grasp_id") or "").strip()
        receipt = self.latest_compiled_contact_execution()
        if isinstance(receipt, dict) and (
            _fact_epoch_value(receipt.get("object_scene_epoch"))
            == self.object_scene_epoch()
            and _fact_epoch_value(receipt.get("robot_motion_epoch"))
            == self.robot_motion_epoch()
        ):
            receipt_compiled_id = str(receipt.get("compiled_grasp_id") or "").strip()
            receipt_target_id = str(receipt.get("target_evidence_id") or "")
            active_target_id = str(provenance.get("target_evidence_id") or "")
            receipt_anchor_id = str(receipt.get("target_identity_anchor_id") or "")
            active_anchor_id = str(provenance.get("target_identity_anchor_id") or "")
            same_target = receipt_compiled_id == active_compiled_id or bool(
                receipt_target_id
                and active_target_id
                and receipt_target_id == active_target_id
            ) or bool(
                receipt_anchor_id
                and active_anchor_id
                and receipt_anchor_id == active_anchor_id
            )
            compiled = self._compiled_grasp_artifact(receipt_compiled_id)
            anchor = compiled.get("target_anchor_world_xyz") if isinstance(compiled, dict) else None
            if (
                same_target
                and receipt_compiled_id
                and isinstance(compiled, dict)
                and _optional_int(compiled.get("scene_epoch"), default=-1)
                == self.object_scene_epoch()
                and _finite_xyz(anchor)
            ):
                return {
                    "schema_version": "openeta.contact_authorization.v1",
                    "compiled_grasp_id": receipt_compiled_id,
                    "waypoint_role": "grasp_contact",
                    "target_anchor_world_xyz": [float(value) for value in anchor[:3]],
                    "target_evidence_id": receipt.get("target_evidence_id"),
                    "object_scene_epoch": self.object_scene_epoch(),
                }

        compiled_id = active_compiled_id
        if not compiled_id:
            return None
        compiled = self._compiled_grasp_artifact(compiled_id)
        if not isinstance(compiled, dict):
            raise ValueError(
                f"active compiled grasp {compiled_id!r} is unavailable for attachment binding"
            )
        contact_pose = compiled.get("contact_pose")
        if not isinstance(contact_pose, dict):
            raise ValueError(
                f"active compiled grasp {compiled_id!r} has no contact pose"
            )
        return self.resolve_compiled_contact_authorization(contact_pose)

    def _latest_compiled_grasp_artifact(self) -> JsonDict:
        for entry in reversed(list(self.artifacts.values())):
            value = entry.get("value") if isinstance(entry, dict) else None
            if isinstance(value, dict) and value.get("type") == "compiled_grasp":
                return value
        return {}

    def motion_reconciliation_gate_error(self, *, tool_name: str) -> str | None:
        """Block new actions until an unknown mutation is observed and reconciled."""

        reconciliation = self.motion_reconciliation()
        if not isinstance(reconciliation, dict) or reconciliation.get("status") not in {
            "required",
            "unresolved",
        }:
            return None
        if tool_name == "observe":
            return None
        return (
            "A simulator action has transport-unknown outcome. Observe the same "
            "environment handle before issuing another action. Do not resend a "
            "partial move because the original controller may still be running."
        )

    def ik_execution_gate_error(
        self, *, tool_name: str, parameters: JsonDict
    ) -> str | None:
        """Require a current feasible IK receipt for the exact move pose policy.

        This binds evidence to one endpoint and orientation policy; it does not
        prescribe a task phase or choose a waypoint for the Agent.  Repairable,
        inconclusive, stale, and hard-infeasible receipts remain useful
        diagnostics, but none authorizes a world mutation.
        """

        if tool_name == "follow_eef_trajectory":
            trajectory = parameters.get("trajectory")
            if not isinstance(trajectory, list) or not trajectory:
                return (
                    "ik_preview_required: follow_eef_trajectory requires one to five "
                    "host-resolved IK receipt targets."
                )
            for index, pose in enumerate(trajectory):
                if not isinstance(pose, dict):
                    return (
                        "ik_preview_required: resolved trajectory waypoint "
                        f"{index} is not a pose object."
                    )
                error = self.ik_execution_gate_error(
                    tool_name="move_to",
                    parameters={
                        "target_pose": pose,
                        "enable_collision_check": parameters.get(
                            "enable_collision_check"
                        ),
                    },
                )
                if error:
                    return f"trajectory_waypoint_{index}: {error}"
            return None
        if tool_name != "move_to":
            return None
        pose_signature = _ik_pose_policy_signature(parameters)
        if not pose_signature:
            return None
        state = self.ik_preview_receipts() or {}
        latest_matching: JsonDict | None = None
        for receipt in reversed(state.get("receipts", []) or []):
            if not isinstance(receipt, dict):
                continue
            if (
                receipt.get("pose_policy_signature") != pose_signature
                and not ik_pose_policy_numerically_equivalent(receipt, parameters)
            ):
                continue
            latest_matching = receipt
            receipt_object_epoch = _fact_epoch_value(receipt.get("object_scene_epoch"))
            receipt_robot_epoch = _fact_epoch_value(receipt.get("robot_motion_epoch"))
            current_epoch = (
                receipt_object_epoch == self.object_scene_epoch()
                and receipt_robot_epoch == self.robot_motion_epoch()
            )
            if not current_epoch:
                break
            classification = str(receipt.get("classification") or "")
            if self._ik_receipt_authorizes_motion(receipt, parameters):
                return None
            if classification == "hard_infeasible":
                return (
                    "ik_target_hard_infeasible: the exact requested endpoint and "
                    "orientation policy were rejected by current-epoch IK evidence "
                    f"{receipt.get('receipt_id')!r}. Reason: "
                    f"{receipt.get('reason_code') or 'hard infeasibility'}. Do not "
                    "execute or replay this pose. Change xyz/orientation policy, use "
                    "the returned repair candidate, or preview a separately proposed "
                    "safe endpoint."
                )
            if classification == "kinematically_feasible_collision_deferred":
                capabilities = self.controller_capabilities() or {}
                return (
                    "ik_collision_delegation_not_authorized: current-epoch IK evidence "
                    f"{receipt.get('receipt_id')!r} found a valid joint solution, but "
                    "endpoint collision proof was unavailable. This receipt authorizes "
                    "move_to only when enable_collision_check=true and the environment "
                    "controller capability explicitly owns per-step pre-actuation and "
                    "post-step trajectory/world collision checking. Current request "
                    f"enable_collision_check={parameters.get('enable_collision_check')!r}; "
                    "controller motion_owns_trajectory_world_collision="
                    f"{capabilities.get('motion_owns_trajectory_world_collision')!r}. "
                    "Keep the exact pose, enable verified motion collision checking, "
                    "or repair/redeploy the collision-capable controller."
                )
            return (
                "ik_preview_not_feasible: current-epoch IK evidence "
                f"{receipt.get('receipt_id')!r} classified this exact endpoint and "
                f"orientation policy as {classification or 'unknown'!r} "
                f"({receipt.get('reason_code') or 'no reason code'}). It is useful "
                "repair evidence but does not authorize move_to. Adjust the proposal "
                "and obtain a feasible IK preview for the exact replacement pose."
            )

        if isinstance(latest_matching, dict):
            stale_note = (
                " The latest matching receipt is stale: receipt epochs "
                f"object={_fact_epoch_value(latest_matching.get('object_scene_epoch'))}, "
                f"robot={_fact_epoch_value(latest_matching.get('robot_motion_epoch'))}; "
                f"current epochs object={self.object_scene_epoch()}, "
                f"robot={self.robot_motion_epoch()}."
            )
        else:
            stale_note = " No matching IK receipt exists."
        return (
            "ik_preview_required: move_to requires a current-epoch feasible IK receipt "
            "for the numerically equivalent target xyz and orientation policy "
            "(explicit orientation or "
            "preserve_current_orientation)."
            + stale_note
            + " Call ik_preview_check on this exact proposed pose, then execute it only "
            "if the returned classification is feasible, or if it is explicitly "
            "kinematically_feasible_collision_deferred and this environment declares "
            "a verified per-step collision-owning motion controller. Endpoint IK does "
            "not prove trajectory/world collision clearance; retain collision checking "
            "on move_to."
        )

    def articulated_probe_action_gate_error(
        self,
        *,
        tool_name: str,
        parameters: JsonDict,
    ) -> str | None:
        """Protect a frozen probe by evidence id, without imposing task order."""

        marker = _articulated_probe_path_sha256(tool_name, parameters)
        if not marker:
            return None
        probe = self.articulated_attachment_probe()
        if not isinstance(probe, dict) or str(probe.get("path_sha256") or "") != marker:
            return (
                "articulated_probe_unknown: the action references an unknown frozen "
                "probe hash; call prepare_attachment_probe with current evidence."
            )
        if probe.get("status") != "prepared":
            return (
                "articulated_probe_replay: this frozen probe is no longer pending; "
                "prepare a new evidence-bound probe instead of replaying it."
            )
        frozen = probe.get("frozen_motion")
        if (
            not isinstance(frozen, dict)
            or str(frozen.get("name") or "") != tool_name
            or not _articulated_probe_motion_equivalent(
                frozen.get("parameters"),
                parameters,
            )
        ):
            return (
                "articulated_probe_tampered: preview every frozen probe waypoint and "
                "execute the resulting receipt id(s) without editing endpoint or path."
            )
        return None

    def resolve_sam3_selection(
        self,
        *,
        result_id: str,
        detection_id: str,
        selection_source: str,
        evidence_role: str = "",
        confidence: float | None = None,
        reason: str = "",
        target_geometry_family: str = "",
        identity_anchor_id: str = "",
        identity_relation: str = "",
    ) -> JsonDict:
        pending = self.pending_sam3_selection()
        if pending is None:
            raise ValueError("No SAM3 detection selection is pending.")
        expected_result_id = str(pending.get("result_id") or "")
        if not result_id or result_id != expected_result_id:
            candidates = pending.get("candidates")
            candidate_ids = (
                [
                    str(candidate.get("id") or "")
                    for candidate in candidates
                    if isinstance(candidate, dict) and candidate.get("id")
                ]
                if isinstance(candidates, list)
                else []
            )
            raise ValueError(
                "select_sam3_detection requires the exact pending sam3_result_id; "
                f"expected={expected_result_id!r}, received={str(result_id or '')!r}. "
                "Retry selection without rerunning SAM3; "
                f"available detection_ids={candidate_ids!r}."
            )
        pending_role = _normalize_sam3_evidence_role(pending.get("evidence_role"))
        requested_role = (
            _normalize_sam3_evidence_role(evidence_role) if evidence_role else pending_role
        )
        if requested_role != pending_role:
            raise ValueError(
                "select_sam3_detection evidence_role must match the pending SAM3 result."
            )
        identity_conflict = pending.get("identity_conflict")
        if pending_role == DEFAULT_SAM3_EVIDENCE_ROLE and isinstance(
            identity_conflict, dict
        ):
            raise ValueError(
                "target_identity_conflict: the host-owned asset/reference verifier "
                f"reported {identity_conflict.get('decision')!r} for the exact points "
                "used by this SAM3 result. Reject these detections or gather a new "
                "view; a generic semantic label cannot override exact-instance evidence."
            )
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
                "source_packet_id": pending.get("source_packet_id"),
                "source_observation": pending.get("source_observation"),
                "target_prompt": pending.get("target_prompt"),
                "segmentation_mode": pending.get("segmentation_mode"),
                "selection_source": selection_source or "main_agent_vlm",
                "selection_confidence": confidence,
                "selection_reason": reason,
                "evidence_role": pending_role,
                "selected_at_s": time.time(),
                "scene_epoch": _optional_int(
                    pending.get("scene_epoch"),
                    default=self.scene_epoch(),
                ),
            }
        )
        if geometry_family and geometry_family != "unknown":
            selected["target_geometry_family"] = geometry_family
        self._capture_target_identity_anchor(
            selected,
            identity_anchor_id=identity_anchor_id,
            identity_relation=identity_relation,
            reason=reason,
        )
        self.facts.pop(PENDING_SAM3_SELECTION_KEY, None)
        self._store_selected_sam3_detection(
            selected,
            evidence_role=pending_role,
            source="select_sam3_detection",
        )
        self._refresh_wrist_alignment_bundle()
        self._remove_sam3_no_detection(evidence_role=pending_role)
        self.facts.pop(TARGET_LOCALIZATION_BUDGET_KEY, None)
        self.record(
            "sam3_detection_selected",
            {
                "result_id": result_id,
                "detection_id": detection_id,
                "selection_source": selected["selection_source"],
                "selection_confidence": confidence,
                "evidence_role": pending_role,
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
            "source_packet_id": pending.get("source_packet_id"),
            "source_observation": pending.get("source_observation"),
            "target_prompt": pending.get("target_prompt"),
            "reason": "semantic_candidates_rejected",
            "segmentation_mode": pending.get("segmentation_mode"),
            "rejected_detection_ids": rejected_ids,
            "rejection_reason": rejection_reason,
            "evidence_role": _normalize_sam3_evidence_role(
                pending.get("evidence_role")
            ),
        }
        self.facts.pop(PENDING_SAM3_SELECTION_KEY, None)
        role = str(no_detection["evidence_role"])
        self._remove_selected_sam3_detection(evidence_role=role)
        self._store_sam3_no_detection(
            no_detection,
            evidence_role=role,
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
    ) -> str | None:
        mode = str(parameters.get("mode") or "targeted").strip().lower()
        if tool_name not in {"anygrasp", "graspgenx"}:
            return None
        if tool_name == "anygrasp" and mode == "scene":
            return None
        selected = self.selected_sam3_detection()
        if selected is None:
            pending = self.pending_sam3_selection()
            if isinstance(pending, dict):
                return (
                    f"{tool_name} cannot consume an unverified segmentation mask: "
                    f"SAM3 result {pending.get('result_id')!r} still has unresolved "
                    "target identity. Inspect the candidate evidence and explicitly "
                    "record the selected detection, or reject all candidates."
                )
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

    def gate_repair_bundle(
        self,
        *,
        code: str,
        reason: str,
        requested_tool: str,
        requested_parameters: JsonDict,
    ) -> JsonDict:
        """Return executable, host-grounded recovery choices for a hard rejection."""

        options: list[JsonDict] = [
            {
                "tool": "observe",
                "parameters": {},
                "reason": "refresh visual and measured robot evidence without mutating the world",
            }
        ]
        if code == "ik_preview_required":
            target_pose = requested_parameters.get("target_pose")
            if isinstance(target_pose, dict):
                preview_parameters: JsonDict = {
                    "target_pose": dict(target_pose),
                    "check_endpoint_collision": True,
                }
                if _target_pose_has_no_orientation(target_pose):
                    preview_parameters["preserve_current_orientation"] = True
                if "tolerance" in requested_parameters:
                    preview_parameters["position_tolerance_m"] = requested_parameters[
                        "tolerance"
                    ]
                if "ori_tolerance" in requested_parameters:
                    preview_parameters["orientation_tolerance_rad"] = requested_parameters[
                        "ori_tolerance"
                    ]
                options.append(
                    {
                        "tool": "ik_preview_check",
                        "parameters": preview_parameters,
                        "reason": (
                            "preview the exact requested endpoint and orientation policy; "
                            "a feasible current-epoch receipt can authorize this move"
                        ),
                    }
                )
        elif code == "ik_collision_delegation_not_authorized":
            capabilities = self.controller_capabilities() or {}
            if capabilities.get("motion_owns_trajectory_world_collision") is True:
                options.append(
                    {
                        "tool": requested_tool,
                        "parameters": {
                            **dict(requested_parameters),
                            "enable_collision_check": True,
                        },
                        "reason": (
                            "reuse the exact current-epoch kinematic solution while "
                            "delegating trajectory/world collision proof to the "
                            "declared per-step collision-owning motion controller"
                        ),
                    }
                )
        elif code in {
            "ik_target_hard_infeasible",
            "ik_preview_not_feasible",
            "compiled_grasp_adjustment_unverified_orientation_policy",
        }:
            target_pose = requested_parameters.get("target_pose")
            if isinstance(target_pose, dict):
                position_only_pose = {
                    key: value
                    for key, value in target_pose.items()
                    if key
                    not in {
                        "rotation_matrix",
                        "quat_xyzw",
                        "quaternion",
                        "rotvec",
                        "roll",
                        "pitch",
                        "yaw",
                    }
                }
                options.append(
                    {
                        "tool": "ik_preview_check",
                        "parameters": {
                            "target_pose": position_only_pose,
                            "preserve_current_orientation": True,
                            **(
                                {
                                    "position_tolerance_m": requested_parameters[
                                        "position_tolerance_m"
                                    ]
                                }
                                if "position_tolerance_m" in requested_parameters
                                else {}
                            ),
                            **(
                                {
                                    "orientation_tolerance_rad": requested_parameters[
                                        "ori_tolerance"
                                    ]
                                }
                                if "ori_tolerance" in requested_parameters
                                else {}
                            ),
                        },
                        "reason": (
                            "preview the same position with current wrist orientation; "
                            "this is a new proposal, not authorization for the rejected rotation"
                        ),
                    }
                )
        localization = self.pending_reference_localization()
        if isinstance(localization, dict):
            required_parameter = str(
                localization.get("required_parameter") or "roi_bbox_xyxy"
            )
            parameters: JsonDict = {
                "source_packet_id": localization.get("source_packet_id"),
                "prompt": localization.get("target_object") or "target object",
            }
            if localization.get("camera_frame_id"):
                parameters["camera_frame_id"] = localization["camera_frame_id"]
            required_value = localization.get(required_parameter)
            if required_value is not None:
                # ``positive_points`` is the evidence-bundle field retained
                # for asset-reference compatibility.  The public SAM3 tool
                # contract calls the same value ``points``.  Emit the public
                # spelling here so repair guidance can be executed verbatim.
                if required_parameter == "positive_points":
                    parameters["mode"] = "points"
                    parameters["points"] = required_value
                    parameters.pop("prompt", None)
                else:
                    parameters[required_parameter] = required_value
            options.append(
                {
                    "tool": "sam3",
                    "parameters": parameters,
                    "reason": "consume the active reference-localization evidence",
                }
            )
        no_detection = self.sam3_no_detection()
        if isinstance(no_detection, dict):
            packet_id = str(no_detection.get("source_packet_id") or "").strip()
            frame_id = str(no_detection.get("frame_id") or "").strip()
            source_observation = no_detection.get("source_observation")
            source_observation = (
                source_observation if isinstance(source_observation, dict) else {}
            )
            object_epoch = no_detection.get("object_scene_epoch")
            if object_epoch is None:
                object_epoch = no_detection.get("scene_epoch")
            if object_epoch is None:
                object_epoch = source_observation.get("object_scene_epoch")
            robot_epoch = no_detection.get("robot_motion_epoch")
            if robot_epoch is None:
                robot_epoch = source_observation.get("robot_motion_epoch")
            if (
                packet_id
                and frame_id
                and object_epoch is not None
                and robot_epoch is not None
                and _fact_epoch_value(object_epoch) == self.object_scene_epoch()
                and _fact_epoch_value(robot_epoch) == self.robot_motion_epoch()
            ):
                target = str(no_detection.get("target_prompt") or "target object")
                if code == "same_view_packet_mismatch" and requested_tool == "sam3":
                    exact_sam3 = dict(requested_parameters)
                    exact_sam3["source_packet_id"] = packet_id
                    exact_sam3["camera_frame_id"] = frame_id
                    options.append(
                        {
                            "tool": "sam3",
                            "parameters": exact_sam3,
                            "reason": (
                                "retry the Agent-authored point prompt against the exact "
                                "immutable SAM3 no-detection view"
                            ),
                        }
                    )
                else:
                    prompt = (
                        str(requested_parameters.get("prompt") or "").strip()
                        if requested_tool == "molmopoint"
                        else ""
                    )
                    options.append(
                        {
                            "tool": "molmopoint",
                            "parameters": {
                                "sources": [
                                    {
                                        "source_packet_id": packet_id,
                                        "camera_frame_id": frame_id,
                                    }
                                ],
                                "prompt": prompt or f"Point to the {target} in Image 1.",
                            },
                            "reason": (
                                "preserve the exact immutable SAM3 no-detection view for "
                                "same-view point grounding"
                            ),
                        }
                    )
        pending = self.pending_sam3_selection()
        if isinstance(pending, dict):
            result_id = str(pending.get("result_id") or "")
            evidence_role = _normalize_sam3_evidence_role(pending.get("evidence_role"))
            candidates = pending.get("candidates")
            for candidate in (
                candidates if isinstance(candidates, list) else []
            )[:8]:
                if not isinstance(candidate, dict) or not candidate.get("id"):
                    continue
                options.append(
                    {
                        "tool": "select_sam3_detection",
                        "parameters": {
                            "sam3_result_id": result_id,
                            "detection_id": candidate.get("id"),
                            "evidence_role": evidence_role,
                            "reason": "select after visual identity verification",
                        },
                        "evidence": {
                            "score": candidate.get("score"),
                            "mask_ref": candidate.get("mask_ref"),
                        },
                    }
                )
            if result_id:
                options.append(
                    {
                        "tool": "reject_sam3_detections",
                        "parameters": {
                            "sam3_result_id": result_id,
                            "reason": "none of the candidates matches the task target",
                        },
                        "reason": "use only after visual rejection of every candidate",
                    }
                )
        for public in (self.grasp_input_bundle(), self.anyplace_input_bundle()):
            if not isinstance(public, dict) or public.get("status") != "ready":
                continue
            call_parameters = public.get("call_parameters")
            if isinstance(call_parameters, dict):
                options.append(
                    {
                        "tool": public.get("tool"),
                        "parameters": dict(call_parameters),
                        "reason": "use the active host-resolved provenance bundle",
                    }
                )
        stale_evidence = [
            node
            for node in self.provenance_evidence_graph().get("nodes", [])
            if isinstance(node, dict)
            and str(node.get("freshness") or "").startswith("stale")
        ]
        return {
            "schema_version": "openeta.gate_repair.v1",
            # Reserved, deliberately schema-free extension namespace. v1 tools
            # must not infer keys or semantics until a concrete repair case is
            # reviewed and assigned a versioned extension contract.
            "extensions": {},
            "code": code,
            "violated_invariant": reason,
            "requested_call": {
                "tool": requested_tool,
                "parameters": dict(requested_parameters),
            },
            "evidence_ids": [
                str(node.get("evidence_id"))
                for node in self.provenance_evidence_graph().get("nodes", [])
                if isinstance(node, dict) and node.get("evidence_id")
            ],
            "allowed_next_calls": options,
            "stale_evidence": stale_evidence,
            "recent_source_packets": self.recent_observation_packet_refs(),
            "latest_ik_preview": (self.ik_preview_receipts() or {}).get("latest"),
            "execution_evidence": {
                "latest_compiled_clearance_execution": (
                    self.latest_compiled_clearance_execution()
                ),
                "latest_compiled_contact_execution": (
                    self.latest_compiled_contact_execution()
                ),
            },
        }

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
                parameters = call.get("parameters")
            if not isinstance(parameters, dict):
                parameters = {}
            evidence_role = _normalize_sam3_evidence_role(
                outputs.get("evidence_role") or parameters.get("evidence_role")
            )
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
                or parameters.get("camera_frame_id")
            )
            base = {
                "result_id": result_id,
                "target_prompt": outputs.get("prompt") or parameters.get("prompt"),
                "source_image": outputs.get("source_image"),
                "source_packet_id": outputs.get("source_packet_id"),
                "source_observation": outputs.get("source_observation"),
                "frame_id": source_frame_id,
                "ranking": outputs.get("ranking") or "score_descending",
                "candidate_count": len(candidates),
                "candidates": candidates,
                "selection_bundle": dict(selection_bundle),
                "segmentation_mode": outputs.get("segmentation_mode"),
                "evidence_role": evidence_role,
                "scene_epoch": self.scene_epoch(),
                "object_scene_epoch": self.object_scene_epoch(),
                "robot_motion_epoch": self.robot_motion_epoch(),
            }
            if source_camera_role:
                base["camera_role"] = source_camera_role
            self.facts.pop(REFERENCE_LOCALIZATION_FAILURE_KEY, None)
            asset_reference = self.target_asset_reference()
            if isinstance(asset_reference, dict):
                verification = asset_reference.get("exact_instance_verification")
                supplied_points = parameters.get("points")
                if supplied_points is None:
                    supplied_points = parameters.get("positive_points")
                if (
                    isinstance(verification, dict)
                    and str(parameters.get("source_packet_id") or "")
                    == str(asset_reference.get("source_packet_id") or "")
                    and (
                        not asset_reference.get("camera_frame_id")
                        or str(parameters.get("camera_frame_id") or "")
                        == str(asset_reference.get("camera_frame_id") or "")
                    )
                    and asset_reference.get("positive_points") is not None
                    and supplied_points == asset_reference.get("positive_points")
                ):
                    decision = str(verification.get("decision") or "").lower()
                    if decision == "match":
                        base["reference_verification"] = dict(verification)
                    elif decision in {"mismatch", "abstain"}:
                        base["identity_conflict"] = dict(verification)
            self.facts.pop(PENDING_SAM3_SELECTION_KEY, None)
            if candidates:
                self._remove_sam3_no_detection(evidence_role=evidence_role)
                self.facts[PENDING_SAM3_SELECTION_KEY] = _memory_fact_entry(
                    base,
                    source="sam3",
                )
                self.record(
                    "sam3_detection_selection_required",
                    {
                        "result_id": result_id,
                        "candidate_count": len(candidates),
                        "evidence_role": evidence_role,
                        "verification_scope": (
                            "single_detection" if len(candidates) == 1 else "multiple_detections"
                        ),
                    },
                )
            else:
                self._remove_selected_sam3_detection(evidence_role=evidence_role)
                self._store_sam3_no_detection(
                    base,
                    evidence_role=evidence_role,
                    source="sam3",
                )
                self.record(
                    "sam3_no_detection",
                    {"result_id": result_id, "evidence_role": evidence_role},
                )
            self._save_working_memory()

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
                source_reference = self.observation_packet_reference_for_path(scene_image)
                if not source_reference:
                    continue
                positive_points = bundle.get("positive_points")
                if not isinstance(positive_points, list):
                    positive_points = outputs.get("positive_points")
                bbox_xyxy = bundle.get("bbox_xyxy")
                if not isinstance(bbox_xyxy, list):
                    bbox_xyxy = outputs.get("bbox_xyxy")
                point_prompt = isinstance(positive_points, list) and bool(positive_points)
                ranked_candidates = bundle.get("ranked_candidates")
                if not isinstance(ranked_candidates, list):
                    ranked_candidates = []
                localizer = outputs.get("localizer")
                if not isinstance(localizer, dict):
                    localizer = {}
                verification = localizer.get("verification")
                exact_instance_verification = (
                    {
                        "decision": str(verification.get("decision") or "").lower(),
                        "confidence": verification.get("confidence"),
                        "reason": verification.get("reason"),
                        "candidate_crop": verification.get("candidate_crop"),
                        "reference_geometry": verification.get("reference_geometry"),
                        "candidate_geometry": verification.get("candidate_geometry"),
                        "grasp_geometry_family": verification.get("grasp_geometry_family"),
                    }
                    if isinstance(verification, dict)
                    and str(verification.get("decision") or "").lower()
                    in {"match", "mismatch", "abstain"}
                    else None
                )
                obligation = {
                    "environment": outputs.get("environment") or bundle.get("environment"),
                    "target_object": outputs.get("target_object") or bundle.get("target_object"),
                    "scene_image": scene_image,
                    **source_reference,
                    "reference_images": reference_images,
                    "marked_scene_image": bundle.get("marked_scene_image_ref")
                    or outputs.get("marked_scene_image"),
                    "positive_points": positive_points if point_prompt else None,
                    "bbox_xyxy": bbox_xyxy,
                    "candidate_policy": bundle.get("candidate_policy"),
                    "ranked_candidates": ranked_candidates,
                    "requires_downstream_confirmation": bool(
                        bundle.get("requires_downstream_confirmation")
                    ),
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
                        **source_reference,
                        "reference_images": reference_images,
                        "positive_points": positive_points if point_prompt else None,
                        "bbox_xyxy": bbox_xyxy,
                        "candidate_policy": bundle.get("candidate_policy"),
                        "ranked_candidates": ranked_candidates,
                        "requires_downstream_confirmation": bool(
                            bundle.get("requires_downstream_confirmation")
                        ),
                        "exact_instance_verification": exact_instance_verification,
                    },
                    source=name,
                )
                self.facts.pop(PENDING_SAM3_SELECTION_KEY, None)
                self._remove_selected_sam3_detection(
                    evidence_role=DEFAULT_SAM3_EVIDENCE_ROLE
                )
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
                source_reference = self.observation_packet_reference_for_path(scene_image)
                if not source_reference:
                    continue
                obligation = {
                    "environment": None,
                    "target_object": no_detection.get("target_prompt"),
                    "scene_image": scene_image,
                    **source_reference,
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
                self._remove_selected_sam3_detection(
                    evidence_role=DEFAULT_SAM3_EVIDENCE_ROLE
                )
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
            supplied_points = parameters.get("points")
            if supplied_points is None:
                supplied_points = parameters.get("positive_points")
            geometry_matches = (
                supplied_points == pending.get("positive_points")
                if required_parameter == "positive_points"
                else parameters.get("roi_bbox_xyxy") is not None
            )
            if (
                str(parameters.get("source_packet_id") or "")
                == str(pending.get("source_packet_id") or "")
                and (
                    not pending.get("camera_frame_id")
                    or str(parameters.get("camera_frame_id") or "")
                    == str(pending.get("camera_frame_id") or "")
                )
                and geometry_matches
            ):
                self.facts.pop(PENDING_REFERENCE_LOCALIZATION_KEY, None)
                self.record(
                    "asset_reference_localization_resolved",
                    {
                        "target_object": pending.get("target_object"),
                        required_parameter: (
                            supplied_points
                            if required_parameter == "positive_points"
                            else parameters.get(required_parameter)
                        ),
                    },
                )
                self._save_working_memory()

    def _capture_articulated_attachment_probe(self, action: EnvAction) -> bool:
        call = _successful_tool_call(action, "prepare_attachment_probe")
        if call is None:
            return False
        outputs = _tool_call_outputs(call)
        if outputs.get("schema_version") != "openeta.articulated_attachment_probe.v1":
            return False
        if _optional_int(outputs.get("scene_epoch"), default=-1) != self.scene_epoch():
            return False
        probe_id = str(outputs.get("probe_id") or "")
        compiled_grasp_id = str(outputs.get("compiled_grasp_id") or "")
        if not probe_id or not compiled_grasp_id:
            return False
        graph = self.provenance_evidence_graph()
        if not any(
            isinstance(node, dict)
            and node.get("kind") == "compiled_targeted_grasp"
            and node.get("freshness") == "current_object_scene"
            and str(node.get("compiled_grasp_id") or "") == compiled_grasp_id
            for node in graph.get("nodes", [])
        ):
            return False
        frozen_motion = outputs.get("frozen_motion")
        if not isinstance(frozen_motion, dict):
            return False
        name = str(frozen_motion.get("name") or "")
        parameters = frozen_motion.get("parameters")
        if name not in {"move_to", "follow_eef_trajectory"} or not isinstance(
            parameters, dict
        ):
            return False
        probe = {
            **dict(outputs),
            "status": "prepared",
            "prepared_at_s": time.time(),
        }
        self.facts[ARTICULATED_ATTACHMENT_PROBE_KEY] = _memory_fact_entry(
            probe,
            source="prepare_attachment_probe",
        )
        self.record("articulated_attachment_probe_prepared", dict(probe))
        return True

    def _capture_articulated_attachment_probe_result(self, action: EnvAction) -> bool:
        probe = self.articulated_attachment_probe()
        if not isinstance(probe, dict) or probe.get("status") != "prepared":
            return False
        frozen_motion = probe.get("frozen_motion")
        if not isinstance(frozen_motion, dict):
            return False
        command = action.command if isinstance(action.command, dict) else {}
        request = command.get("request")
        if not isinstance(request, dict):
            return False
        name = str(frozen_motion.get("name") or "")
        parameters = frozen_motion.get("parameters")
        request_parameters = request.get("parameters")
        resolved_request_parameters = self._resolve_executed_motion_reference_for_probe(
            name=name,
            parameters=request_parameters,
        )
        if (
            str(request.get("name") or "") != name
            or not isinstance(parameters, dict)
            or not _articulated_probe_motion_equivalent(
                parameters,
                resolved_request_parameters,
            )
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
        return True

    def _capture_grasp_provenance(self, action: EnvAction) -> bool:
        """Bind a compiled candidate to its host-captured targeted RGB-D packet."""

        call = _successful_tool_call(action, "compile_grasp_seed")
        if call is None:
            return False
        outputs = _tool_call_outputs(call)
        if outputs.get("schema_version") != "openeta.compiled_grasp_seed.v1":
            return False
        command = action.command if isinstance(action.command, dict) else {}
        request = command.get("request")
        parameters = request.get("parameters") if isinstance(request, dict) else None
        executed_parameters = call.get("parameters")
        candidate_id = str(outputs.get("candidate_id") or "")
        public_parameters = parameters if isinstance(parameters, dict) else {}
        grasp_result_id = str(public_parameters.get("grasp_result_id") or "")
        requested_candidate_id = str(public_parameters.get("candidate_id") or "")
        candidate: object = None
        if grasp_result_id and requested_candidate_id:
            try:
                candidate_resolution = self.resolve_grasp_candidate_input(
                    grasp_result_id=grasp_result_id,
                    candidate_id=requested_candidate_id,
                )
            except ValueError as exc:
                self.record(
                    "grasp_provenance_unresolved",
                    {
                        "candidate_id": requested_candidate_id,
                        "compiled_grasp_id": outputs.get("compiled_grasp_id"),
                        "reason": "short_reference_resolution_failed",
                        "message": str(exc),
                    },
                )
                return False
            resolved_parameters = candidate_resolution.get("parameters")
            if isinstance(resolved_parameters, dict):
                candidate = resolved_parameters.get("camera_pose")
        elif isinstance(executed_parameters, dict):
            # Compatibility for host-created actions that predate planner-facing
            # short references. New Agent actions retain only result/candidate ids.
            candidate = executed_parameters.get(
                "camera_pose", executed_parameters.get("candidate")
            )
        if not isinstance(candidate, dict) or str(candidate.get("id") or "") != candidate_id:
            return False
        matched = self._targeted_grasp_artifact_for_candidate(candidate)
        if matched is None:
            self.record(
                "grasp_provenance_unresolved",
                {
                    "candidate_id": candidate_id,
                    "compiled_grasp_id": outputs.get("compiled_grasp_id"),
                    "reason": "candidate_not_found_in_host_grasp_evidence",
                },
            )
            return False
        artifact_key, artifact = matched
        artifact_epoch = _fact_epoch_value(artifact.get("scene_epoch"))
        if artifact_epoch != self.object_scene_epoch():
            self.record(
                "grasp_provenance_unresolved",
                {
                    "candidate_id": candidate_id,
                    "compiled_grasp_id": outputs.get("compiled_grasp_id"),
                    "reason": "targeted_grasp_evidence_is_stale",
                    "evidence_object_scene_epoch": artifact_epoch,
                    "current_object_scene_epoch": self.object_scene_epoch(),
                },
            )
            return False
        source_value = artifact.get("selected_grasp_source")
        source = dict(source_value) if isinstance(source_value, dict) else {}
        for source_key, artifact_field in (
            ("rgb", "source_rgb"),
            ("depth", "source_depth"),
            ("object_mask", "target_mask"),
        ):
            if not isinstance(source.get(source_key), str) or not source[source_key]:
                value = artifact.get(artifact_field)
                if isinstance(value, str) and value:
                    source[source_key] = value
        if str(source.get("mode") or "") != "targeted" or any(
            not isinstance(source.get(key), str) or not source[key]
            for key in ("rgb", "depth", "object_mask")
        ):
            return False
        source.setdefault("source_tool", artifact.get("source_tool") or artifact.get("tool"))
        source.setdefault("source_backend", artifact.get("source_backend"))
        selected_target = self.selected_sam3_detection()
        target_evidence_id = ""
        if (
            isinstance(selected_target, dict)
            and _same_memory_artifact_path(
                selected_target.get("mask_ref"), source.get("object_mask")
            )
        ):
            target_evidence_id = (
                f"sam3:{selected_target.get('result_id')}:{selected_target.get('id')}"
            )
        evidence_payload = {
            "candidate": candidate,
            "source": source,
            "compiled_grasp_id": outputs.get("compiled_grasp_id"),
            "artifact_key": artifact_key,
            "object_scene_epoch": self.object_scene_epoch(),
            "target_evidence_id": target_evidence_id,
            "target_identity_anchor_id": (
                selected_target.get("identity_anchor_id")
                if isinstance(selected_target, dict)
                else None
            ),
        }
        evidence_id = "grasp:" + hashlib.sha256(
            json.dumps(evidence_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:20]
        provenance = {
            "schema_version": "openeta.grasp_provenance.v1",
            "evidence_id": evidence_id,
            "candidate_id": candidate_id,
            "candidate": dict(candidate),
            "source": source,
            "artifact_key": artifact_key,
            "result_id": artifact.get("result_id"),
            "target_evidence_id": target_evidence_id or None,
            "target_identity_anchor_id": (
                selected_target.get("identity_anchor_id")
                if isinstance(selected_target, dict)
                else None
            ),
            "compiled_grasp_id": outputs.get("compiled_grasp_id"),
            "object_scene_epoch": self.object_scene_epoch(),
            "robot_motion_epoch": self.robot_motion_epoch(),
            "created_at_s": time.time(),
        }
        existing = _memory_fact_value(self.facts.get(GRASP_PROVENANCE_KEY))
        if isinstance(existing, dict) and existing.get("evidence_id") == evidence_id:
            return False
        switched_from = (
            str(existing.get("evidence_id") or "")
            if isinstance(existing, dict)
            else ""
        )
        recovery_branch = bool(
            switched_from
            and self._gripper_close_attempted_since_provenance(switched_from)
        )
        self.facts[GRASP_PROVENANCE_KEY] = _memory_fact_entry(
            provenance,
            source="compile_grasp_seed_evidence_graph",
        )
        self.record(
            "grasp_provenance_bound",
            {
                "evidence_id": evidence_id,
                "candidate_id": candidate_id,
                "compiled_grasp_id": outputs.get("compiled_grasp_id"),
                "artifact_key": artifact_key,
                "previous_evidence_id": switched_from or None,
                "recovery_branch": recovery_branch,
                # Durable resolver evidence.  This remains in the event log,
                # not the bounded Agent projection, and lets a later wrist
                # refinement reuse an earlier fixed-camera source safely.
                "provenance": dict(provenance),
            },
        )
        if switched_from:
            self.record(
                "grasp_provenance_switched",
                {
                    "previous_evidence_id": switched_from,
                    "evidence_id": evidence_id,
                    "compiled_grasp_id": outputs.get("compiled_grasp_id"),
                    "recovery_branch": recovery_branch,
                },
            )
        return True

    def _gripper_close_attempted_since_provenance(self, evidence_id: str) -> bool:
        """Return whether the active grasp branch had an acknowledged close attempt.

        A failed grasp is normally opened before the Agent selects a replacement
        candidate.  Looking only at the current latched gripper state therefore
        loses the recovery signal and prevents the bounded episode turn extension
        from activating.  Scan only events after the matching provenance binding,
        so a close from an older branch cannot grant recovery budget here.
        """

        expected = str(evidence_id or "").strip()
        if not expected:
            return False
        for event in reversed(self.events):
            payload = event.payload if isinstance(event.payload, dict) else {}
            if (
                event.event_type == "grasp_provenance_bound"
                and str(payload.get("evidence_id") or "") == expected
            ):
                return False
            if (
                event.event_type == "gripper_command_state_changed"
                and payload.get("position") == 0
                and payload.get("latched") is True
            ):
                return True
        return False

    def _record_new_targeted_grasp_evidence(self, action: EnvAction) -> bool:
        """Record a new proposal without deactivating the active grasp branch.

        Perception is immutable evidence, not an instruction to switch execution
        branches.  The active provenance changes only when the Agent explicitly
        compiles a candidate from the new result.
        """

        call = None
        for name in ("grasp_pose_estimate", "anygrasp", "graspgenx"):
            call = _successful_tool_call(action, name)
            if call is not None:
                break
        if call is None:
            return False
        outputs = _tool_call_outputs(call)
        source = outputs.get("source")
        if not isinstance(source, dict) or source.get("mode") != "targeted":
            return False
        active = _memory_fact_value(self.facts.get(GRASP_PROVENANCE_KEY)) or {}
        self.record(
            "targeted_grasp_evidence_available",
            {
                "active_evidence_id": active.get("evidence_id"),
                "new_grasp_result_id": outputs.get("result_id"),
                "activation": "compile_grasp_seed",
            },
        )
        return False

    def _targeted_grasp_artifact_for_candidate(
        self,
        candidate: JsonDict,
    ) -> tuple[str, JsonDict] | None:
        for artifact_key, entry in sorted(
            self.artifacts.items(),
            key=lambda item: float(item[1].get("timestamp_s") or 0.0)
            if isinstance(item[1], dict)
            else 0.0,
            reverse=True,
        ):
            artifact = entry.get("value") if isinstance(entry, dict) else None
            if not isinstance(artifact, dict) or artifact.get("type") != "grasp_candidates":
                continue
            source = artifact.get("selected_grasp_source")
            if not isinstance(source, dict) or source.get("mode") != "targeted":
                continue
            candidates = artifact.get("grasp_candidates")
            if not isinstance(candidates, list):
                continue
            if any(
                isinstance(item, dict) and _same_grasp_candidate(item, candidate)
                for item in candidates
            ):
                return artifact_key, artifact
            # The Agent-facing working-memory projection keeps only a small
            # preview.  Advisor-selected or code-selected IDs may legitimately
            # refer to a later candidate, so provenance must resolve against the
            # complete immutable host artifact rather than treating projection
            # truncation as evidence loss.
            complete_reference = artifact.get("complete_outputs_artifact")
            if not isinstance(complete_reference, dict):
                continue
            try:
                complete_outputs = load_structured_tool_output(
                    complete_reference,
                    expected_tool=str(artifact.get("tool") or ""),
                    expected_result_id=str(artifact.get("result_id") or ""),
                    allowed_root=self.artifact_root,
                )
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            complete_candidates = complete_outputs.get("grasp_candidates")
            if isinstance(complete_candidates, list) and any(
                isinstance(item, dict) and _same_grasp_candidate(item, candidate)
                for item in complete_candidates
            ):
                return artifact_key, artifact
        return None

    def _refresh_grasp_input_bundle(self) -> bool:
        """Compile selected mask and aligned RGB-D provenance into one opaque input."""

        selected = self.selected_sam3_detection()
        previous = _memory_fact_value(self.facts.get(GRASP_INPUT_BUNDLES_KEY)) or {}
        bundles = dict(previous.get("bundles") or {}) if isinstance(previous, dict) else {}
        public: JsonDict = {
            "schema_version": "openeta.grasp_input_bundle.v1",
            "status": "awaiting_target_selection",
            "tool": "grasp_pose_estimate",
        }
        active_bundle_id = ""
        if isinstance(selected, dict):
            source = selected.get("source_observation")
            source = source if isinstance(source, dict) else {}
            rgb = source.get("rgb") or selected.get("source_image")
            depth = source.get("depth")
            intrinsics = source.get("intrinsics")
            frame_id = source.get("frame_id") or selected.get("frame_id")
            mask_ref = selected.get("mask_ref")
            selected_epoch = _optional_int(selected.get("scene_epoch"), default=-1)
            public.update(
                {
                    "sam3_result_id": selected.get("result_id"),
                    "detection_id": selected.get("id"),
                    "source_packet_id": source.get("packet_id")
                    or selected.get("source_packet_id"),
                    "object_scene_epoch": selected_epoch,
                }
            )
            if selected_epoch != self.object_scene_epoch():
                public.update(
                    {
                        "status": "stale_object_scene",
                        "recovery": "segment the target on a current observation packet",
                    }
                )
            elif not (
                isinstance(rgb, str)
                and rgb
                and isinstance(depth, str)
                and depth
                and isinstance(mask_ref, str)
                and mask_ref
                and isinstance(frame_id, str)
                and frame_id
                and isinstance(intrinsics, dict)
                and all(key in intrinsics for key in ("fx", "fy", "cx", "cy", "scale"))
                and _same_memory_artifact_path(rgb, selected.get("source_image"))
            ):
                public.update(
                    {
                        "status": "source_packet_incomplete",
                        "recovery": "rerun SAM3 from an observation with aligned RGB-D metadata",
                    }
                )
            else:
                try:
                    mask_quality = assess_target_mask_quality(
                        mask_ref,
                        depth_path=depth,
                    )
                except (OSError, ValueError) as exc:
                    public.update(
                        {
                            "status": "target_mask_quality_unavailable",
                            "recovery": (
                                "segment the target again on a current aligned RGB-D packet"
                            ),
                            "diagnostic": {
                                "code": "target_mask_quality_check_failed",
                                "error_type": type(exc).__name__,
                                "message": str(exc),
                            },
                        }
                    )
                    mask_quality = None
                if isinstance(mask_quality, dict):
                    public["target_mask_quality"] = mask_quality
                if isinstance(mask_quality, dict) and mask_quality.get(
                    "usable_for_targeted_geometry"
                ) is not True:
                    public.update(
                        {
                            "status": "requires_better_view",
                            "semantic_outcome": "requires_better_view",
                            "recovery": (
                                "move to a target-facing view where the complete object "
                                "is visible, then observe and segment the same instance"
                            ),
                            "recovery_options": [
                                {
                                    "action": "propose_wrist_viewpoints",
                                    "reason": (
                                        "use a calibrated target-facing wrist observation "
                                        "pose when a compiled grasp and fresh wrist packet exist"
                                    ),
                                },
                                {
                                    "action": "segment_from_scene_primary",
                                    "reason": (
                                        "return to the latest global agentview when the "
                                        "target is outside the current wrist field of view"
                                    ),
                                },
                            ],
                        }
                    )
                if mask_quality is None or mask_quality.get(
                    "usable_for_targeted_geometry"
                ) is not True:
                    parameters = None
                else:
                    calibration_capabilities = (
                        self._active_grasp_calibration_capabilities()
                    )
                    hints = {
                        "depth_cutoff_factor": target_depth_cutoff_factor(
                            depth_path=depth,
                            mask_path=mask_ref,
                            intrinsics=intrinsics,
                        ),
                        "max_gripper_width_m": float(
                            calibration_capabilities["max_gripper_width_m"]
                        ),
                        "execution_reference_point": str(
                            calibration_capabilities["execution_reference_point"]
                        ),
                    }
                    source_extrinsics = source.get("extrinsics")
                    if isinstance(source_extrinsics, dict) and source_extrinsics:
                        try:
                            hints["up_direction_camera"] = world_up_direction_camera(
                                source_extrinsics
                            )
                        except GraspGeometryError:
                            pass
                    parameters = {
                        "mode": "targeted",
                        "source_packet_id": source.get("packet_id")
                        or selected.get("source_packet_id"),
                        "rgb": rgb,
                        "depth": depth,
                        "object_mask": {
                            "mask_ref": mask_ref,
                            "source_image": rgb,
                            "result_id": selected.get("result_id"),
                            "detection_id": selected.get("id"),
                            "quality": mask_quality,
                        },
                        "intrinsics": dict(intrinsics),
                        "camera_frame_id": frame_id,
                        "scene_epoch": selected_epoch,
                        "hints": hints,
                    }
                    geometry_family = str(
                        selected.get("target_geometry_family") or ""
                    ).strip()
                    if geometry_family:
                        parameters["target_geometry_family"] = geometry_family
                if parameters is not None and selected.get("dense_grasp_retry_required") is True:
                    parameters["hints"]["dense_sampling"] = True
                if parameters is None:
                    state = {
                        "schema_version": "openeta.grasp_input_bundle_store.v1",
                        "active_bundle_id": "",
                        "public": public,
                        "bundles": bundles,
                    }
                    comparable_previous = (
                        json.loads(json.dumps(previous)) if isinstance(previous, dict) else {}
                    )
                    comparable_state = json.loads(json.dumps(state))
                    for candidate_state in (comparable_previous, comparable_state):
                        for value in (candidate_state.get("bundles") or {}).values():
                            if isinstance(value, dict):
                                value.pop("created_at_s", None)
                    if comparable_previous == comparable_state:
                        return False
                    self.facts[GRASP_INPUT_BUNDLES_KEY] = _memory_fact_entry(
                        state,
                        source="host_provenance_bundle_resolver",
                    )
                    self.record(
                        "grasp_input_bundle_updated",
                        {
                            "status": public.get("status"),
                            "bundle_id": None,
                            "target_evidence_id": public.get("target_evidence_id"),
                        },
                    )
                    return True
                target_evidence_id = (
                    f"sam3:{selected.get('result_id')}:{selected.get('id')}"
                )
                identity = {
                    "target_evidence_id": target_evidence_id,
                    "parameters": parameters,
                }
                active_bundle_id = "grasp:" + hashlib.sha256(
                    json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(
                        "utf-8"
                    )
                ).hexdigest()[:20]
                bundles[active_bundle_id] = {
                    "schema_version": "openeta.grasp_input_bundle.v1",
                    "bundle_id": active_bundle_id,
                    "parameters": parameters,
                    "target_evidence_id": target_evidence_id,
                    "object_scene_epoch": self.object_scene_epoch(),
                    "created_at_s": time.time(),
                }
                public.update(
                    {
                        "status": "ready",
                        "bundle_id": active_bundle_id,
                        "call_parameters": {"bundle_id": active_bundle_id},
                        "target_evidence_id": target_evidence_id,
                        "target_geometry_family": selected.get(
                            "target_geometry_family"
                        ),
                    }
                )
        state = {
            "schema_version": "openeta.grasp_input_bundle_store.v1",
            "active_bundle_id": active_bundle_id,
            "public": public,
            "bundles": bundles,
        }
        comparable_previous = json.loads(json.dumps(previous)) if isinstance(previous, dict) else {}
        comparable_state = json.loads(json.dumps(state))
        for candidate_state in (comparable_previous, comparable_state):
            for value in (candidate_state.get("bundles") or {}).values():
                if isinstance(value, dict):
                    value.pop("created_at_s", None)
        if comparable_previous == comparable_state:
            return False
        self.facts[GRASP_INPUT_BUNDLES_KEY] = _memory_fact_entry(
            state,
            source="host_provenance_bundle_resolver",
        )
        self.record(
            "grasp_input_bundle_updated",
            {
                "status": public.get("status"),
                "bundle_id": active_bundle_id or None,
                "target_evidence_id": public.get("target_evidence_id"),
            },
        )
        return True

    def _capture_target_identity_anchor(
        self,
        selected: JsonDict,
        *,
        identity_anchor_id: str = "",
        identity_relation: str = "",
        reason: str = "",
    ) -> bool:
        if str(selected.get("evidence_role") or DEFAULT_SAM3_EVIDENCE_ROLE) != (
            DEFAULT_SAM3_EVIDENCE_ROLE
        ):
            return False
        existing = self.target_identity_anchor()
        if isinstance(existing, dict):
            existing_id = str(existing.get("anchor_id") or "")
            exact_same_evidence = all(
                str(selected.get(key) or "") == str(existing.get(anchor_key) or "")
                for key, anchor_key in (
                    ("source_packet_id", "source_packet_id"),
                    ("result_id", "sam3_result_id"),
                    ("id", "detection_id"),
                )
            )
            if exact_same_evidence and not identity_relation:
                selected["identity_anchor_id"] = existing_id
                selected["identity_continuity"] = "same_detection_evidence"
                return False
            supplied_id = str(identity_anchor_id or "").strip()
            relation = str(identity_relation or "").strip().lower()
            if supplied_id != existing_id:
                raise ValueError(
                    "target_identity_confirmation_required: this target selection "
                    "comes from different detection evidence. Copy the exact active "
                    f"identity_anchor_id={existing_id!r} and set identity_relation to "
                    "'same_instance' after cross-view visual comparison, or to "
                    "'replace_misidentified_anchor' when the earlier selection was "
                    "wrong. The host will not silently bind a new mask to the old instance."
                )
            if relation == "same_instance":
                selected["identity_anchor_id"] = existing_id
                selected["identity_continuity"] = "agent_confirmed_same_instance"
                self.record(
                    "target_identity_continuity_confirmed",
                    {
                        "anchor_id": existing_id,
                        "source_packet_id": selected.get("source_packet_id"),
                        "sam3_result_id": selected.get("result_id"),
                        "detection_id": selected.get("id"),
                        "reason": reason,
                    },
                )
                return False
            if relation != "replace_misidentified_anchor":
                raise ValueError(
                    "target_identity_confirmation_required: identity_relation must be "
                    "'same_instance' or 'replace_misidentified_anchor' for new target "
                    "detection evidence."
                )
            verification = existing.get("exact_instance_verification")
            if isinstance(verification, dict) and str(
                verification.get("decision") or ""
            ).lower() == "match":
                raise ValueError(
                    "target_identity_rebind_forbidden: the active anchor has an exact "
                    "reference match. Gather new exact-instance verifier evidence before "
                    "replacing it; generic semantic appearance is insufficient."
                )
            if len(str(reason or "").strip()) < 8:
                raise ValueError(
                    "target_identity_rebind_requires_reason: explain the visual evidence "
                    "showing why the previous anchor was a misidentification."
                )
            self.facts.pop(TARGET_IDENTITY_ANCHOR_KEY, None)
            created = self._capture_target_identity_anchor(selected)
            selected["identity_continuity"] = "anchor_replaced_after_misidentification"
            selected["replaced_identity_anchor_id"] = existing_id
            self.record(
                "target_identity_anchor_replaced",
                {
                    "previous_anchor_id": existing_id,
                    "new_anchor_id": selected.get("identity_anchor_id"),
                    "reason": reason,
                },
            )
            return created
        asset = self.target_asset_reference() or {}
        identity = {
            "target_prompt": selected.get("target_prompt"),
            "sam3_result_id": selected.get("result_id"),
            "detection_id": selected.get("id"),
            "source_packet_id": selected.get("source_packet_id"),
            "frame_id": selected.get("frame_id"),
            "mask_ref": selected.get("mask_ref"),
            "reference_images": asset.get("reference_images", []),
            "exact_instance_verification": selected.get("reference_verification"),
        }
        anchor_id = "target:" + hashlib.sha256(
            json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()[:20]
        anchor = {
            "schema_version": "openeta.target_identity_anchor.v1",
            "anchor_id": anchor_id,
            **identity,
            "created_at_s": time.time(),
            "policy": (
                "later localizers propose positions; they do not silently replace this instance"
            ),
        }
        selected["identity_anchor_id"] = anchor_id
        selected["identity_continuity"] = "anchor_created"
        self.facts[TARGET_IDENTITY_ANCHOR_KEY] = _memory_fact_entry(
            anchor, source="confirmed_target_selection"
        )
        self.record("target_identity_anchor_created", dict(anchor))
        return True

    def _refresh_wrist_alignment_bundle(self) -> bool:
        selected = self.selected_sam3_detection()
        previous = _memory_fact_value(self.facts.get(WRIST_ALIGNMENT_BUNDLES_KEY)) or {}
        bundles = dict(previous.get("bundles") or {}) if isinstance(previous, dict) else {}
        public: JsonDict = {
            "schema_version": "openeta.wrist_alignment_input_bundle.v1",
            "status": "awaiting_wrist_target_selection",
            "tool": "compute_wrist_alignment",
        }
        active_bundle_id = ""
        if isinstance(selected, dict):
            source = selected.get("source_observation")
            source = source if isinstance(source, dict) else {}
            frame_id = str(source.get("frame_id") or selected.get("frame_id") or "")
            role = str(source.get("role") or selected.get("camera_role") or "")
            source_object_epoch = _optional_int(
                source.get("object_scene_epoch"),
                default=_optional_int(selected.get("scene_epoch"), default=-1),
            )
            source_robot_epoch = _optional_int(
                source.get("robot_motion_epoch"), default=-1
            )
            current_object_epoch = self.object_scene_epoch()
            current_robot_epoch = self.robot_motion_epoch()
            public.update(
                {
                    "source_packet_id": source.get("packet_id")
                    or selected.get("source_packet_id"),
                    "camera_frame_id": frame_id,
                    "camera_role": role or None,
                    "object_scene_epoch": source_object_epoch,
                    "robot_motion_epoch": source_robot_epoch,
                }
            )
            if not _is_wrist_camera(frame_id, role):
                public["status"] = "selected_target_is_not_wrist_view"
            elif source_object_epoch != current_object_epoch:
                public.update(
                    {
                        "status": "stale_object_scene",
                        "recovery": "segment the target on a current wrist observation packet",
                    }
                )
            elif source_robot_epoch != current_robot_epoch:
                public.update(
                    {
                        "status": "stale_robot_motion",
                        "recovery": (
                            "observe fresh wrist RGB-D after the latest robot motion, "
                            "then segment and select the same target instance"
                        ),
                    }
                )
            else:
                provenance = _memory_fact_value(self.facts.get(GRASP_PROVENANCE_KEY)) or {}
                compiled_id = str(provenance.get("compiled_grasp_id") or "")
                compiled = self._compiled_grasp_artifact(compiled_id) if compiled_id else None
                provenance_anchor_id = str(
                    provenance.get("target_identity_anchor_id") or ""
                )
                selected_anchor_id = str(selected.get("identity_anchor_id") or "")
                public.update(
                    {
                        "target_identity_anchor_id": selected_anchor_id or None,
                        "compiled_target_identity_anchor_id": (
                            provenance_anchor_id or None
                        ),
                    }
                )
                if provenance_anchor_id and selected_anchor_id != provenance_anchor_id:
                    public.update(
                        {
                            "status": "target_identity_mismatch",
                            "recovery": (
                                "select the same target instance and explicitly confirm "
                                "cross-view identity continuity before wrist alignment"
                            ),
                        }
                    )
                    parameters = None
                else:
                    budget = self.grasp_adjustment_budget() or {}
                    parameters = {
                        "compiled_grasp": compiled,
                        "target_mask": selected.get("mask_ref"),
                        "depth": source.get("depth"),
                        "intrinsics": source.get("intrinsics"),
                        "camera_extrinsics": source.get("extrinsics"),
                        "current_eef_pose": source.get("current_eef_pose"),
                        "scene_epoch": current_object_epoch,
                        "source_object_scene_epoch": source_object_epoch,
                        "current_object_scene_epoch": current_object_epoch,
                        "source_robot_motion_epoch": source_robot_epoch,
                        "current_robot_motion_epoch": current_robot_epoch,
                        "residual_budget": budget,
                    }
                if parameters is None:
                    missing = []
                else:
                    missing = [
                        key
                        for key, value in parameters.items()
                        if value in (None, "", {}) and key != "residual_budget"
                    ]
                if parameters is None:
                    pass
                elif missing:
                    public.update(
                        {
                            "status": "source_packet_incomplete",
                            "missing_inputs": missing,
                            "recovery": (
                                "observe fresh wrist RGB-D with calibrated camera and measured EEF pose"
                            ),
                        }
                    )
                else:
                    target_evidence_id = (
                        f"sam3:{selected.get('result_id')}:{selected.get('id')}"
                    )
                    identity = {
                        "target_evidence_id": target_evidence_id,
                        "compiled_grasp_id": compiled_id,
                        "source_packet_id": public.get("source_packet_id"),
                        "target_identity_anchor_id": selected_anchor_id or None,
                        "object_scene_epoch": current_object_epoch,
                        "robot_motion_epoch": current_robot_epoch,
                        "parameters": parameters,
                    }
                    active_bundle_id = "wrist_alignment:" + hashlib.sha256(
                        json.dumps(
                            identity,
                            ensure_ascii=False,
                            sort_keys=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    ).hexdigest()[:20]
                    bundles[active_bundle_id] = {
                        "schema_version": "openeta.wrist_alignment_input_bundle.v1",
                        "bundle_id": active_bundle_id,
                        "parameters": parameters,
                        "target_evidence_id": target_evidence_id,
                        "compiled_grasp_id": compiled_id,
                        "object_scene_epoch": current_object_epoch,
                        "robot_motion_epoch": current_robot_epoch,
                        "target_identity_anchor_id": selected_anchor_id or None,
                        "created_at_s": time.time(),
                    }
                    public.update(
                        {
                            "status": "ready",
                            "bundle_id": active_bundle_id,
                            "call_parameters": {"bundle_id": active_bundle_id},
                            "target_evidence_id": target_evidence_id,
                            "compiled_grasp_id": compiled_id,
                            "object_scene_epoch": current_object_epoch,
                            "robot_motion_epoch": current_robot_epoch,
                        }
                    )
        state = {
            "schema_version": "openeta.wrist_alignment_bundle_store.v1",
            "active_bundle_id": active_bundle_id,
            "public": public,
            "bundles": bundles,
        }
        comparable_previous = json.loads(json.dumps(previous)) if isinstance(previous, dict) else {}
        comparable_state = json.loads(json.dumps(state))
        for candidate_state in (comparable_previous, comparable_state):
            for value in (candidate_state.get("bundles") or {}).values():
                if isinstance(value, dict):
                    value.pop("created_at_s", None)
        if comparable_previous == comparable_state:
            return False
        self.facts[WRIST_ALIGNMENT_BUNDLES_KEY] = _memory_fact_entry(
            state, source="host_wrist_alignment_bundle_resolver"
        )
        self.record(
            "wrist_alignment_bundle_updated",
            {"status": public.get("status"), "bundle_id": active_bundle_id or None},
        )
        return True

    def _rebase_anyplace_grasp_to_fixed_camera(
        self,
        bundles: JsonDict,
        *,
        grasp: JsonDict,
        placement_evidence_id: str,
    ) -> JsonDict | None:
        """Re-express a wrist grasp in a prior fixed-camera AnyPlace source.

        AnyPlace requires object and placement masks plus the selected grasp in
        one camera geometry.  Re-running fixed-camera grasp estimation after a
        successful wrist refinement discards the better contact geometry and
        can create a source-repair loop.  A prior same-target bundle already
        owns aligned fixed-camera RGB-D and masks, so calibrated packet
        extrinsics are sufficient to express the wrist grasp in that frame.
        """

        source = grasp.get("source")
        candidate = grasp.get("candidate")
        anchor_id = str(grasp.get("target_identity_anchor_id") or "")
        if (
            not isinstance(source, dict)
            or not isinstance(candidate, dict)
            or not anchor_id
            or not _is_moving_camera_frame(
                _camera_frame_hint(source, source.get("rgb"))
            )
        ):
            return None
        source_reference = self.observation_packet_reference_for_path(source.get("rgb"))
        if not source_reference:
            return None
        try:
            source_packet = self.resolve_observation_packet(
                str(source_reference.get("source_packet_id") or ""),
                str(source_reference.get("camera_frame_id") or ""),
                require_files=False,
            )
        except ObservationPacketResolutionError:
            return None
        source_extrinsics = source_packet.get("extrinsics")
        if not isinstance(source_extrinsics, dict) or not source_extrinsics:
            return None

        current_object_epoch = self.object_scene_epoch()
        fixed_sources: list[tuple[JsonDict, JsonDict]] = []
        # The Agent may choose near-field refinement before it has segmented a
        # receptacle or materialized AnyPlace.  Compile events retain the full
        # host-owned provenance off-context, so the earlier same-target fixed
        # camera can still supply aligned object geometry without prescribing
        # an action order. Prefer this canonical grasp lineage over a derived
        # AnyPlace bundle: using the latest bundle as its own rebase source
        # recursively changed source_bundle_id and therefore the next bundle's
        # content-addressed identity after every unrelated action.
        for event in reversed(self.events):
            if event.event_type != "grasp_provenance_bound":
                continue
            historical = event.payload.get("provenance")
            if (
                not isinstance(historical, dict)
                or historical.get("target_identity_anchor_id") != anchor_id
                or historical.get("object_scene_epoch") != current_object_epoch
            ):
                continue
            target_source = historical.get("source")
            if not isinstance(target_source, dict):
                continue
            fixed_sources.append(
                (
                    target_source,
                    {
                        "source_bundle_id": None,
                        "source_grasp_evidence_id": historical.get("evidence_id"),
                    },
                )
            )

        # Retained bundles remain a fallback when no durable fixed-camera grasp
        # provenance is available (for example, restored legacy sessions).
        for bundle_id, prior in reversed(list(bundles.items())):
            if (
                not isinstance(prior, dict)
                or prior.get("placement_evidence_id") != placement_evidence_id
                or prior.get("target_identity_anchor_id") != anchor_id
                or prior.get("object_scene_epoch") != current_object_epoch
            ):
                continue
            parameters = prior.get("parameters")
            selected_grasp = (
                parameters.get("selected_grasp")
                if isinstance(parameters, dict)
                else None
            )
            target_source = (
                selected_grasp.get("source")
                if isinstance(selected_grasp, dict)
                else None
            )
            if not isinstance(target_source, dict):
                continue
            fixed_sources.append(
                (
                    target_source,
                    {
                        "source_bundle_id": bundle_id,
                        "source_grasp_evidence_id": prior.get("grasp_evidence_id"),
                    },
                )
            )

        seen_target_sources: set[tuple[str, str]] = set()
        for target_source, source_lineage in fixed_sources:
            target_frame = _camera_frame_hint(target_source, target_source.get("rgb"))
            if not target_frame or _is_moving_camera_frame(target_frame):
                continue
            target_key = (target_frame, str(target_source.get("rgb") or ""))
            if target_key in seen_target_sources:
                continue
            seen_target_sources.add(target_key)
            target_reference = self.observation_packet_reference_for_path(
                target_source.get("rgb")
            )
            if not target_reference:
                continue
            try:
                target_packet = self.resolve_observation_packet(
                    str(target_reference.get("source_packet_id") or ""),
                    str(target_reference.get("camera_frame_id") or ""),
                    require_files=False,
                )
            except ObservationPacketResolutionError:
                continue
            target_extrinsics = target_packet.get("extrinsics")
            if not isinstance(target_extrinsics, dict) or not target_extrinsics:
                continue
            try:
                rebased_candidate = rebase_camera_grasp_candidate(
                    candidate,
                    source_camera_extrinsics=source_extrinsics,
                    target_camera_extrinsics=target_extrinsics,
                )
            except GraspGeometryError:
                continue
            rebased_source = dict(target_source)
            for key in (
                "source_tool",
                "source_backend",
                "gripper_name",
            ):
                if source.get(key) is not None:
                    rebased_source[key] = source[key]
            if source.get("up_direction_camera") is not None:
                try:
                    rebased_source["up_direction_camera"] = rebase_camera_direction(
                        source["up_direction_camera"],
                        source_camera_extrinsics=source_extrinsics,
                        target_camera_extrinsics=target_extrinsics,
                    )
                except GraspGeometryError:
                    continue
            rebased_source["mode"] = "targeted"
            provenance = {
                "schema_version": "openeta.cross_camera_grasp_rebase.v1",
                "mode": "calibrated_world_invariant",
                "source_packet_id": source_reference.get("source_packet_id"),
                "source_camera_frame_id": source_reference.get("camera_frame_id"),
                "target_packet_id": target_reference.get("source_packet_id"),
                "target_camera_frame_id": target_reference.get("camera_frame_id"),
                "source_candidate_id": candidate.get("id"),
                **source_lineage,
                "target_identity_anchor_id": anchor_id,
            }
            rebased_source["cross_camera_rebase"] = dict(provenance)
            rebased_candidate["cross_camera_rebase"] = dict(provenance)
            return {
                "candidate": rebased_candidate,
                "source": rebased_source,
                "provenance": provenance,
            }
        return None

    def _refresh_anyplace_input_bundle(self) -> bool:
        """Join compatible grasp and placement evidence without exposing raw parameters."""

        previous = _memory_fact_value(self.facts.get(ANYPLACE_INPUT_BUNDLES_KEY)) or {}
        previous_active_id = (
            str(previous.get("active_bundle_id") or "")
            if isinstance(previous, dict)
            else ""
        )
        previous_bundles = previous.get("bundles") if isinstance(previous, dict) else None
        previous_active = (
            previous_bundles.get(previous_active_id)
            if isinstance(previous_bundles, dict) and previous_active_id
            else None
        )
        grasp = _memory_fact_value(self.facts.get(GRASP_PROVENANCE_KEY))
        if not isinstance(grasp, dict):
            return False
        if (
            isinstance(previous_active, dict)
            and previous_active.get("materialized") is True
            and previous_active.get("grasp_evidence_id") == grasp.get("evidence_id")
        ):
            # A successful AnyPlace result is a frozen placement plan.  Later
            # attachment-verification segmentation must not rewrite its source
            # packet or silently rebind it to a newer object-scene epoch.
            return False
        source = grasp.get("source")
        candidate = grasp.get("candidate")
        if not isinstance(source, dict) or not isinstance(candidate, dict):
            return False
        expected_image = source.get("rgb")
        placement = self.selected_sam3_detection("placement_region")
        bundles = dict(previous.get("bundles") or {}) if isinstance(previous, dict) else {}
        public: JsonDict = {
            "schema_version": "openeta.anyplace_input_bundle.v1",
            "status": "awaiting_placement_region",
            "tool": "anyplace",
            "required_source_image": expected_image,
            "grasp_evidence_id": grasp.get("evidence_id"),
        }
        active_bundle_id = ""
        target = self.selected_sam3_detection()
        if isinstance(target, dict) and not _same_memory_artifact_path(
            target.get("mask_ref"), source.get("object_mask")
        ):
            public.update(
                {
                    "status": "target_grasp_mismatch",
                    "selected_target_mask": target.get("mask_ref"),
                    "recovery": (
                        "run targeted grasp estimation and compile against the selected target"
                    ),
                }
            )
        elif isinstance(placement, dict):
            mask_ref = placement.get("mask_ref")
            source_image = placement.get("source_image")
            placement_evidence_id = _placement_evidence_id(placement)
            public["placement_evidence_id"] = placement_evidence_id
            same_source_path = _same_memory_artifact_path(source_image, expected_image)
            identical_source_content = (
                not same_source_path
                and _same_memory_artifact_content(source_image, expected_image)
            )
            source_matches = same_source_path or identical_source_content
            reusable_placement = (
                None
                if source_matches
                else _reusable_fixed_camera_placement(
                    bundles,
                    placement_evidence_id=placement_evidence_id,
                    expected_source=source,
                )
            )
            grasp_rebase = None
            if not source_matches:
                grasp_rebase = self._rebase_anyplace_grasp_to_fixed_camera(
                    bundles,
                    grasp=grasp,
                    placement_evidence_id=placement_evidence_id,
                )
                if isinstance(grasp_rebase, dict):
                    candidate = grasp_rebase["candidate"]
                    source = grasp_rebase["source"]
                    expected_image = source.get("rgb")
                    public["required_source_image"] = expected_image
                    source_matches = _same_memory_artifact_path(
                        source_image, expected_image
                    )
            if not source_matches and reusable_placement is None:
                expected_frame = _camera_frame_hint(source, expected_image)
                placement_observation = placement.get("source_observation")
                placement_observation = (
                    placement_observation
                    if isinstance(placement_observation, dict)
                    else {}
                )
                placement_frame = _camera_frame_hint(
                    placement_observation,
                    source_image,
                )
                # AnyPlace requires grasp/object/placement inputs to share one
                # camera geometry. A wrist camera commonly cannot see the
                # receptacle at all. If placement evidence already comes from
                # a fixed scene camera, rebase the grasp evidence onto that
                # camera instead of prescribing an impossible wrist SAM call.
                rebase_grasp_to_fixed_camera = bool(
                    _is_moving_camera_frame(expected_frame)
                    and placement_frame
                    and not _is_moving_camera_frame(placement_frame)
                )
                if rebase_grasp_to_fixed_camera:
                    repair_source_image = source_image
                    repair_parameters = {
                        "prompt": (
                            target.get("target_prompt")
                            if isinstance(target, dict)
                            else None
                        )
                        or "target object",
                        "evidence_role": DEFAULT_SAM3_EVIDENCE_ROLE,
                    }
                    recovery = (
                        "segment the target object on the fixed placement camera, "
                        "select it, estimate a grasp from its host bundle, and compile "
                        "that candidate; keep the current placement-region selection"
                    )
                    required_source_image = source_image
                else:
                    repair_source_image = expected_image
                    repair_parameters = {
                        "prompt": placement.get("target_prompt") or "placement region",
                        "evidence_role": "placement_region",
                    }
                    recovery = (
                        "segment the placement region on required_source_image and "
                        "select that SAM3 detection"
                    )
                    required_source_image = expected_image
                repair_reference = self.observation_packet_reference_for_path(
                    repair_source_image
                )
                if repair_reference:
                    repair_parameters.update(repair_reference)
                    public.update(
                        {
                            "status": "placement_source_mismatch",
                            "required_source_image": required_source_image,
                            "provided_source_image": source_image,
                            "recovery": recovery,
                            "repair_call": {
                                "tool": "sam3",
                                "parameters": repair_parameters,
                            },
                        }
                    )
                else:
                    public.update(
                        {
                            "status": "source_packet_unresolvable",
                            "required_source_image": required_source_image,
                            "provided_source_image": source_image,
                            "recovery": (
                                "observe again because the required source artifact is not "
                                "owned by any indexed observation packet"
                            ),
                        }
                    )
            elif not isinstance(source.get("intrinsics"), dict) or any(
                candidate.get(key) is None
                for key in (
                    "id",
                    "translation_xyz",
                    "rotation_matrix",
                    "gripper_tip_position_xyz",
                    "depth",
                    "width",
                    "height",
                )
            ):
                public.update(
                    {
                        "status": "grasp_evidence_incomplete",
                        "recovery": (
                            "rerun targeted grasp estimation and compile a complete candidate"
                        ),
                    }
                )
            elif isinstance(mask_ref, str) and mask_ref:
                direct_frame_reuse = bool(identical_source_content)
                effective_source_image = (
                    expected_image
                    if reusable_placement or direct_frame_reuse
                    else source_image
                )
                parameters = {
                    "rgb": source.get("rgb"),
                    "depth": source.get("depth"),
                    "object_mask": source.get("object_mask"),
                    "placement_region_mask": {
                        "mask_ref": mask_ref,
                        "source_image": effective_source_image,
                    },
                    "intrinsics": source.get("intrinsics"),
                    "selected_grasp": {
                        "candidate": candidate,
                        "source": source,
                    },
                }
                identity = {
                    "grasp_evidence_id": grasp.get("evidence_id"),
                    "placement_evidence_id": placement_evidence_id,
                    "parameters": parameters,
                }
                active_bundle_id = "anyplace:" + hashlib.sha256(
                    json.dumps(identity, sort_keys=True, separators=(",", ":")).encode(
                        "utf-8"
                    )
                ).hexdigest()[:20]
                bundles[active_bundle_id] = {
                    "schema_version": "openeta.anyplace_input_bundle.v1",
                    "bundle_id": active_bundle_id,
                    "parameters": parameters,
                    "grasp_evidence_id": grasp.get("evidence_id"),
                    "placement_evidence_id": placement_evidence_id,
                    "target_identity_anchor_id": grasp.get(
                        "target_identity_anchor_id"
                    ),
                    "grasp_rebase": (
                        grasp_rebase.get("provenance")
                        if isinstance(grasp_rebase, dict)
                        else None
                    ),
                    "placement_evidence_reuse": (
                        {
                            "mode": (
                                "identical_frame_content"
                                if direct_frame_reuse
                                else "fixed_camera_identity"
                            ),
                            "evidence_source_image": source_image,
                            "effective_source_image": expected_image,
                            "source_bundle_id": (
                                reusable_placement.get("bundle_id")
                                if isinstance(reusable_placement, dict)
                                else None
                            ),
                        }
                        if isinstance(reusable_placement, dict) or direct_frame_reuse
                        else None
                    ),
                    "object_scene_epoch": self.object_scene_epoch(),
                    "created_at_s": time.time(),
                }
                public.update(
                    {
                        "status": "ready",
                        "bundle_id": active_bundle_id,
                        "call_parameters": {"bundle_id": active_bundle_id},
                        "object_scene_epoch": self.object_scene_epoch(),
                    }
                )
                if isinstance(reusable_placement, dict) or direct_frame_reuse:
                    public["placement_evidence_reuse"] = {
                        "mode": (
                            "identical_frame_content"
                            if direct_frame_reuse
                            else "fixed_camera_identity"
                        ),
                        "evidence_source_image": source_image,
                        "effective_source_image": expected_image,
                        "reason": (
                            "the selected placement frame is byte-identical to the grasp "
                            "source frame; rebind the mask to the exact grasp RGB-D packet"
                            if direct_frame_reuse
                            else "placement region is unchanged on the same fixed scene "
                            "camera; reuse the selected mask while rebinding aligned RGB-D "
                            "for the new grasp source"
                        ),
                    }
                if isinstance(grasp_rebase, dict):
                    public["grasp_rebase"] = dict(grasp_rebase["provenance"])
        state = {
            "schema_version": "openeta.anyplace_input_bundle_store.v1",
            "active_bundle_id": active_bundle_id,
            "public": public,
            "bundles": bundles,
        }
        comparable_previous = (
            json.loads(json.dumps(previous)) if isinstance(previous, dict) else {}
        )
        for value in (comparable_previous.get("bundles") or {}).values():
            if isinstance(value, dict):
                value.pop("created_at_s", None)
        comparable_state = json.loads(json.dumps(state))
        for value in (comparable_state.get("bundles") or {}).values():
            if isinstance(value, dict):
                value.pop("created_at_s", None)
        if comparable_previous == comparable_state:
            return False
        self.facts[ANYPLACE_INPUT_BUNDLES_KEY] = _memory_fact_entry(
            state,
            source="host_provenance_bundle_resolver",
        )
        self.record(
            "anyplace_input_bundle_updated",
            {
                "status": public.get("status"),
                "bundle_id": active_bundle_id or None,
                "grasp_evidence_id": grasp.get("evidence_id"),
                "placement_evidence_id": public.get("placement_evidence_id"),
            },
        )
        return True

    def _capture_anyplace_bundle_materialization(self, action: EnvAction) -> bool:
        """Freeze a successful AnyPlace result as an attachment-bound placement plan."""

        call = _successful_tool_call(action, "anyplace")
        if call is None:
            return False
        outputs = _tool_call_outputs(call)
        candidates = outputs.get("placement_candidates")
        if not isinstance(candidates, list) or not any(
            isinstance(candidate, dict) for candidate in candidates
        ):
            return False
        command = action.command if isinstance(action.command, dict) else {}
        request = command.get("request")
        request_parameters = (
            request.get("parameters") if isinstance(request, dict) else None
        )
        call_parameters = call.get("parameters")
        bundle_id = ""
        for parameters in (request_parameters, call_parameters):
            if not isinstance(parameters, dict):
                continue
            candidate_id = parameters.get("bundle_id")
            if isinstance(candidate_id, str) and candidate_id.strip():
                bundle_id = candidate_id.strip()
                break
        state = _memory_fact_value(self.facts.get(ANYPLACE_INPUT_BUNDLES_KEY))
        if not isinstance(state, dict):
            return False
        active_id = str(state.get("active_bundle_id") or "")
        if not bundle_id:
            bundle_id = active_id
        if not bundle_id or bundle_id != active_id:
            return False
        bundles = dict(state.get("bundles") or {})
        bundle = bundles.get(bundle_id)
        if not isinstance(bundle, dict):
            return False
        current_epoch = self.object_scene_epoch()
        result_identity = {
            "bundle_id": bundle_id,
            "result_id": outputs.get("result_id"),
            "raw_output_ref": outputs.get("raw_output_ref"),
            "placement_candidates": candidates,
        }
        materialized_result_id = str(outputs.get("result_id") or "").strip()
        if not materialized_result_id:
            materialized_result_id = "anyplace-result:" + hashlib.sha256(
                json.dumps(
                    result_identity,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()[:20]
        updated_bundle = {
            **bundle,
            "materialized": True,
            "materialized_result_id": materialized_result_id,
            "placement_candidates": [
                dict(candidate) for candidate in candidates if isinstance(candidate, dict)
            ],
            "freshness_scope": "attachment_bound_plan",
            "input_object_scene_epoch": _fact_epoch_value(
                bundle.get("object_scene_epoch")
            ),
            "valid_through_object_scene_epoch": current_epoch,
            "materialized_at_s": time.time(),
        }
        if (
            bundle.get("materialized") is True
            and bundle.get("materialized_result_id") == materialized_result_id
        ):
            return False
        bundles[bundle_id] = updated_bundle
        public_value = state.get("public")
        public = dict(public_value) if isinstance(public_value, dict) else {}
        public.update(
            {
                "status": "ready",
                "bundle_id": bundle_id,
                "call_parameters": {"bundle_id": bundle_id},
                "materialized": True,
                "materialized_result_id": materialized_result_id,
                "freshness_scope": "attachment_bound_plan",
                "input_object_scene_epoch": updated_bundle[
                    "input_object_scene_epoch"
                ],
                "valid_through_object_scene_epoch": current_epoch,
            }
        )
        self.facts[ANYPLACE_INPUT_BUNDLES_KEY] = _memory_fact_entry(
            {**state, "public": public, "bundles": bundles},
            source="anyplace_result_materialization",
        )
        self.record(
            "anyplace_input_bundle_materialized",
            {
                "bundle_id": bundle_id,
                "materialized_result_id": materialized_result_id,
                "object_scene_epoch": current_epoch,
            },
        )
        return True

    def _capture_placement_world_reference(self, action: EnvAction) -> bool:
        """Retain the latest transformed placement pose as compact tool evidence."""

        call = _successful_tool_call(action, "camera_pose_to_world")
        if call is None:
            return False
        outputs = _tool_call_outputs(call)
        placement = outputs.get("placement_reference")
        world_pose = outputs.get("world_pose")
        if (
            not isinstance(placement, dict)
            or placement.get("schema_version")
            != "openeta.placement_world_reference.v1"
            or not isinstance(world_pose, dict)
            or not _finite_xyz(world_pose.get("translation_xyz"))
        ):
            return False
        value = {
            "schema_version": "openeta.placement_world_reference.v1",
            "semantic_role": placement.get("semantic_role"),
            "execution_authorized": False,
            "placement_result_id": outputs.get("placement_result_id"),
            "candidate_id": outputs.get("candidate_id"),
            "world_pose": {
                "frame": "world",
                "translation_xyz": [
                    float(component)
                    for component in world_pose["translation_xyz"][:3]
                ],
                **(
                    {"rotation_matrix": world_pose.get("rotation_matrix")}
                    if isinstance(world_pose.get("rotation_matrix"), list)
                    else {}
                ),
            },
            "required_before_motion": placement.get("required_before_motion"),
            "agent_release_options": placement.get("agent_release_options"),
            "object_scene_epoch": self.object_scene_epoch(),
            "updated_at_s": time.time(),
        }
        previous = self.placement_world_reference()
        if isinstance(previous, dict):
            comparable_previous = dict(previous)
            comparable_previous.pop("updated_at_s", None)
            comparable_value = dict(value)
            comparable_value.pop("updated_at_s", None)
            if comparable_previous == comparable_value:
                return False
        self.facts[PLACEMENT_WORLD_REFERENCE_KEY] = _memory_fact_entry(
            value,
            source="camera_pose_to_world",
        )
        self.record("placement_world_reference_updated", dict(value))
        return True

    def _capture_ik_preview_receipt(self, action: EnvAction) -> bool:
        call = _pipeline_call(action, "ik_preview_check")
        if not isinstance(call, dict):
            return False
        result = call.get("result")
        details = result.get("details") if isinstance(result, dict) else None
        if not isinstance(details, dict):
            return False
        outputs = details.get("outputs")
        outputs = outputs if isinstance(outputs, dict) else {}
        parameters = call.get("parameters")
        parameters = parameters if isinstance(parameters, dict) else {}
        receipt = outputs.get("ik_preview_receipt")
        if not isinstance(receipt, dict):
            reachability = outputs.get("reachability")
            if not isinstance(reachability, dict):
                return False
            receipt = {
                "schema_version": "openeta.ik_preview_receipt.v1",
                "receipt_id": hashlib.sha256(
                    json.dumps(
                        {"parameters": parameters, "reachability": reachability},
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest()[:20],
                "target_signature": _ik_target_signature(parameters),
                "classification": _ik_reachability_classification(reachability),
                "reason_code": reachability.get("reason_code"),
                "reachability": dict(reachability),
            }
        receipt = {
            **receipt,
            "object_scene_epoch": self.object_scene_epoch(),
            "robot_motion_epoch": self.robot_motion_epoch(),
            "recorded_at_s": time.time(),
        }
        target_pose = parameters.get("target_pose")
        if not isinstance(receipt.get("target_pose"), dict) and isinstance(
            target_pose, dict
        ):
            receipt["target_pose"] = dict(target_pose)
        if not receipt.get("orientation_policy"):
            receipt["orientation_policy"] = _ik_orientation_policy(parameters)
        if not receipt.get("target_signature"):
            receipt["target_signature"] = _ik_target_signature(
                parameters
            )
        if not receipt.get("pose_policy_signature"):
            receipt["pose_policy_signature"] = _ik_pose_policy_signature(
                parameters
            )
        request_reference: JsonDict = {}
        target_reference = target_pose if isinstance(target_pose, dict) else {}
        for field_name in (
            "compiled_grasp_id",
            "waypoint_role",
            "path_fraction",
            "probe_id",
            "waypoint_index",
            "viewpoint_proposal_id",
            "viewpoint_id",
            "candidate_id",
        ):
            value = parameters.get(field_name, target_reference.get(field_name))
            if value is not None:
                request_reference[field_name] = value
        if request_reference:
            receipt["request_reference"] = request_reference
        state = self.ik_preview_receipts() or {
            "schema_version": "openeta.ik_preview_receipt_index.v1",
            "receipts": [],
        }
        receipts = [
            dict(item)
            for item in state.get("receipts", [])
            if isinstance(item, dict)
        ]
        receipts.append(receipt)
        state.update({"receipts": receipts[-16:], "latest": receipt})
        self.facts[IK_PREVIEW_RECEIPTS_KEY] = _memory_fact_entry(
            state, source="ik_preview_check"
        )
        self.record("ik_preview_receipt", dict(receipt))
        return True

    def _record_world_mutation(self, action: EnvAction) -> bool:
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
        call = _tool_call(action, name)
        if call is None:
            return False
        call_succeeded = _call_result_success(call)
        if name in {"move_to", "follow_eef_trajectory"}:
            # A failed goal is not necessarily a no-op. Controllers commonly move
            # for several iterations before reporting iteration_limit, collision,
            # or another terminal failure. Any trustworthy receipt proving that
            # motion occurred must invalidate robot-state-bound IK/view evidence.
            # Conversely, a blocked/skipped call with no execution receipt must not
            # advance the epoch merely because legacy successful calls default to
            # conservative mutation semantics.
            pose_changed = _motion_call_changed_pose(call)
            if not call_succeeded and not _motion_call_receipt_proves_pose_change(call):
                return False
            if not pose_changed:
                self.record(
                    "world_mutation_noop",
                    {
                        "tool": name,
                        "robot_motion_epoch": self.robot_motion_epoch(),
                        "reason": "motion receipt reports zero executed steps and no pose change",
                    },
                )
                return False
        elif not call_succeeded:
            return False
        self._advance_runtime_epochs(
            tool=name,
            # A command acknowledgement is not visual evidence that an object
            # moved, attached, or was released.  Object freshness advances only
            # when a later observation carries host-owned change evidence.
            object_scene_changed=False,
            source=(
                "successful_world_mutation"
                if call_succeeded
                else "failed_motion_partial_execution"
            ),
        )
        return True

    def _capture_compiled_clearance_execution(self, action: EnvAction) -> bool:
        call = _tool_call(action, "move_to")
        if not isinstance(call, dict):
            return False
        parameters = _executed_tool_parameters(action, "move_to")
        target_pose = parameters.get("target_pose")
        if (
            not isinstance(target_pose, dict)
            or _compiled_grasp_pose_role(target_pose) != "clearance"
        ):
            return False
        compiled_id = str(target_pose.get("compiled_grasp_id") or "")
        if not compiled_id:
            return False
        result = call.get("result")
        details = result.get("details") if isinstance(result, dict) else None
        details = details if isinstance(details, dict) else {}
        motion = _structured_motion_summary(details)
        end = motion.get("end")
        end = end if isinstance(end, dict) else {}
        receipt = {
            "schema_version": "openeta.compiled_clearance_execution.v1",
            "compiled_grasp_id": compiled_id,
            "source_grasp_id": target_pose.get("source_grasp_id"),
            "reached_target": motion.get("reached_target") is True
            and details.get("operational_success") is not False,
            "steps_executed": motion.get("steps_executed"),
            "stop_reason": motion.get("stop_reason"),
            "requested_xyz": _rounded_xyz(target_pose.get("xyz")),
            "actual_xyz": (
                _rounded_xyz(end.get("xyz")) if _finite_xyz(end.get("xyz")) else None
            ),
            "position_error_m": motion.get("position_error_m"),
            "orientation_error_rad": motion.get("orientation_error_rad"),
            "diagnostics": [
                dict(item)
                for item in details.get("diagnostics", [])
                if isinstance(item, dict)
            ],
            "object_scene_epoch": self.object_scene_epoch(),
            "robot_motion_epoch": self.robot_motion_epoch(),
            "recorded_at_s": time.time(),
        }
        self.facts[LATEST_COMPILED_CLEARANCE_EXECUTION_KEY] = _memory_fact_entry(
            receipt,
            source="compiled_clearance_execution_receipt",
        )
        self.record("compiled_clearance_execution", dict(receipt))
        return True

    def _capture_compiled_contact_execution(self, action: EnvAction) -> bool:
        call = _tool_call(action, "move_to")
        if not isinstance(call, dict):
            return False
        parameters = _executed_tool_parameters(action, "move_to")
        target_pose = parameters.get("target_pose")
        if not isinstance(target_pose, dict) or _compiled_grasp_pose_role(target_pose) != "contact":
            return False
        compiled_id = str(target_pose.get("compiled_grasp_id") or "")
        if not compiled_id:
            return False
        result = call.get("result")
        details = result.get("details") if isinstance(result, dict) else None
        details = details if isinstance(details, dict) else {}
        motion = _structured_motion_summary(details)
        collision = motion.get("collision")
        collision = dict(collision) if isinstance(collision, dict) else {}
        end = motion.get("end")
        end = end if isinstance(end, dict) else {}
        provenance = _memory_fact_value(self.facts.get(GRASP_PROVENANCE_KEY)) or {}
        receipt = {
            "schema_version": "openeta.compiled_contact_execution.v1",
            "compiled_grasp_id": compiled_id,
            "source_grasp_id": target_pose.get("source_grasp_id"),
            "target_evidence_id": provenance.get("target_evidence_id"),
            "target_identity_anchor_id": provenance.get("target_identity_anchor_id"),
            "reached_target": motion.get("reached_target") is True
            and details.get("operational_success") is not False,
            "steps_executed": motion.get("steps_executed"),
            "stop_reason": motion.get("stop_reason"),
            "requested_xyz": _rounded_xyz(target_pose.get("xyz")),
            "actual_xyz": (
                _rounded_xyz(end.get("xyz")) if _finite_xyz(end.get("xyz")) else None
            ),
            "position_error_m": motion.get("position_error_m"),
            "max_axis_position_error_m": motion.get("max_axis_position_error_m"),
            "orientation_error_rad": motion.get("orientation_error_rad"),
            "collision": collision,
            "diagnostics": [
                dict(item)
                for item in details.get("diagnostics", [])
                if isinstance(item, dict)
            ],
            "object_scene_epoch": self.object_scene_epoch(),
            "robot_motion_epoch": self.robot_motion_epoch(),
            "recorded_at_s": time.time(),
        }
        self.facts[LATEST_COMPILED_CONTACT_EXECUTION_KEY] = _memory_fact_entry(
            receipt,
            source="compiled_contact_execution_receipt",
        )
        self.record("compiled_contact_execution", dict(receipt))
        return True

    def _capture_grasp_adjustment_budget(self, action: EnvAction) -> bool:
        """Persist host-derived residual travel for compiled-grasp move_to calls."""

        compile_call = _successful_tool_call(action, "compile_grasp_seed")
        if compile_call is not None:
            outputs = _tool_call_outputs(compile_call)
            compiled_id = str(outputs.get("compiled_grasp_id") or "")
            if outputs.get("schema_version") == "openeta.compiled_grasp_seed.v1" and compiled_id:
                self.facts[GRASP_ADJUSTMENT_BUDGET_KEY] = _memory_fact_entry(
                    {
                        "schema_version": "openeta.compiled_grasp_adjustment.v1",
                        "compiled_grasp_id": compiled_id,
                        "last_waypoint_role": None,
                        "last_residual_xyz_m": [0.0, 0.0, 0.0],
                        "last_target_xyz": None,
                        "cumulative_translation_m": 0.0,
                        "remaining_translation_m": GRASP_ADJUSTMENT_CUMULATIVE_LIMIT_M,
                        "per_call_limit_m": GRASP_ADJUSTMENT_STEP_LIMIT_M,
                        "cumulative_limit_m": GRASP_ADJUSTMENT_CUMULATIVE_LIMIT_M,
                        "object_scene_epoch": self.object_scene_epoch(),
                    },
                    source="compile_grasp_seed",
                )
                self.record(
                    "compiled_grasp_adjustment_budget_reset",
                    {
                        "compiled_grasp_id": compiled_id,
                        "per_call_limit_m": GRASP_ADJUSTMENT_STEP_LIMIT_M,
                        "cumulative_limit_m": GRASP_ADJUSTMENT_CUMULATIVE_LIMIT_M,
                    },
                )
                return True

        call = _successful_tool_call(action, "move_to")
        if call is None:
            return False
        parameters = _executed_tool_parameters(action, "move_to")
        target_pose = parameters.get("target_pose")
        if not isinstance(target_pose, dict):
            return False
        compiled_id = str(target_pose.get("compiled_grasp_id") or "")
        role = _compiled_grasp_pose_role(target_pose)
        target_xyz = target_pose.get("xyz")
        if not compiled_id or not role or not _finite_xyz(target_xyz):
            return False
        compiled = self._compiled_grasp_artifact(compiled_id)
        reference = (
            _compiled_grasp_reference_pose(compiled, role=role)
            if isinstance(compiled, dict)
            else None
        )
        reference_xyz = reference.get("xyz") if isinstance(reference, dict) else None
        if not _finite_xyz(reference_xyz):
            return False
        residual = [
            float(target_xyz[index]) - float(reference_xyz[index]) for index in range(3)
        ]
        prior = self.grasp_adjustment_budget() or {}
        if str(prior.get("compiled_grasp_id") or "") == compiled_id:
            prior_residual_raw = prior.get("last_residual_xyz_m")
            prior_residual = (
                [float(value) for value in prior_residual_raw]
                if _finite_xyz(prior_residual_raw)
                else [0.0, 0.0, 0.0]
            )
            cumulative = _finite_nonnegative_float(
                prior.get("cumulative_translation_m"),
                default=0.0,
            )
        else:
            prior_residual = [0.0, 0.0, 0.0]
            cumulative = 0.0
        step = math.sqrt(
            sum((residual[index] - prior_residual[index]) ** 2 for index in range(3))
        )
        cumulative = min(GRASP_ADJUSTMENT_CUMULATIVE_LIMIT_M, cumulative + step)
        budget = {
            "schema_version": "openeta.compiled_grasp_adjustment.v1",
            "compiled_grasp_id": compiled_id,
            "last_waypoint_role": role,
            "last_residual_xyz_m": _rounded_xyz(residual),
            "last_target_xyz": _rounded_xyz([float(value) for value in target_xyz]),
            "last_increment_m": round(step, 6),
            "cumulative_translation_m": round(cumulative, 6),
            "remaining_translation_m": round(
                max(0.0, GRASP_ADJUSTMENT_CUMULATIVE_LIMIT_M - cumulative),
                6,
            ),
            "per_call_limit_m": GRASP_ADJUSTMENT_STEP_LIMIT_M,
            "cumulative_limit_m": GRASP_ADJUSTMENT_CUMULATIVE_LIMIT_M,
            "object_scene_epoch": self.object_scene_epoch(),
        }
        self.facts[GRASP_ADJUSTMENT_BUDGET_KEY] = _memory_fact_entry(
            budget,
            source="move_to_compiled_grasp_residual",
        )
        self.record("compiled_grasp_adjustment_consumed", dict(budget))
        return True

    def _advance_runtime_epochs(
        self,
        *,
        tool: str,
        object_scene_changed: bool,
        source: str,
        preserve_materialized_anyplace: bool = False,
    ) -> None:
        robot_epoch = self.robot_motion_epoch() + 1
        self.facts[ROBOT_MOTION_EPOCH_KEY] = _memory_fact_entry(
            {"epoch": robot_epoch},
            source=source,
        )
        object_epoch = self.object_scene_epoch()
        if object_scene_changed:
            object_epoch += 1
            entry = _memory_fact_entry({"epoch": object_epoch}, source=source)
            self.facts[OBJECT_SCENE_EPOCH_KEY] = entry
            self.facts[SCENE_EPOCH_KEY] = _memory_fact_entry(
                {"epoch": object_epoch},
                source=f"{source}_legacy_alias",
            )
            self._invalidate_anyplace_bundle_for_object_scene_change(
                tool=tool,
                object_scene_epoch=object_epoch,
                source=source,
                preserve_materialized=preserve_materialized_anyplace,
            )
        self.record(
            "world_epochs_advanced",
            {
                "tool": tool,
                "robot_motion_epoch": robot_epoch,
                "object_scene_epoch": object_epoch,
                "object_scene_changed": object_scene_changed,
            },
        )

    def _capture_observed_object_scene_change(
        self, observation: EnvObservation
    ) -> bool:
        """Advance object freshness only from explicit observation evidence.

        The simulator or real-robot observation adapter may publish either the
        compact boolean ``object_scene_changed`` or a structured
        ``object_scene_change`` record.  Repeated change ids are idempotent.
        Robot/gripper commands never synthesize this evidence.
        """

        metadata = observation.metadata if isinstance(observation.metadata, dict) else {}
        raw = metadata.get("object_scene_change")
        if isinstance(raw, dict):
            changed = raw.get("changed") is True
            evidence = dict(raw)
        else:
            changed = metadata.get("object_scene_changed") is True
            evidence = {
                "changed": changed,
                "reason": metadata.get("object_scene_change_reason"),
                "change_id": metadata.get("object_scene_change_id"),
            }
        if not changed:
            return False
        change_id = str(evidence.get("change_id") or "").strip()
        previous = _memory_fact_value(self.facts.get(OBSERVED_OBJECT_SCENE_CHANGE_KEY)) or {}
        if change_id and change_id == str(previous.get("change_id") or ""):
            return False
        object_epoch = self.object_scene_epoch() + 1
        self.facts[OBJECT_SCENE_EPOCH_KEY] = _memory_fact_entry(
            {"epoch": object_epoch}, source="observed_object_scene_change"
        )
        self.facts[SCENE_EPOCH_KEY] = _memory_fact_entry(
            {"epoch": object_epoch}, source="observed_object_scene_change_legacy_alias"
        )
        evidence.update(
            {
                "schema_version": "openeta.observed_object_scene_change.v1",
                "object_scene_epoch": object_epoch,
                "observed_at_s": time.time(),
            }
        )
        self.facts[OBSERVED_OBJECT_SCENE_CHANGE_KEY] = _memory_fact_entry(
            evidence, source="observation_adapter"
        )
        self._invalidate_anyplace_bundle_for_object_scene_change(
            tool="observe",
            object_scene_epoch=object_epoch,
            source="observed_object_scene_change",
        )
        self.record("observed_object_scene_change", evidence)
        return True

    def _invalidate_anyplace_bundle_for_object_scene_change(
        self,
        *,
        tool: str,
        object_scene_epoch: int,
        source: str,
        preserve_materialized: bool = False,
    ) -> None:
        state = _memory_fact_value(self.facts.get(ANYPLACE_INPUT_BUNDLES_KEY))
        if not isinstance(state, dict) or not state.get("active_bundle_id"):
            return
        previous_bundle_id = str(state.get("active_bundle_id") or "")
        bundles = dict(state.get("bundles") or {})
        active_bundle = bundles.get(previous_bundle_id)
        if preserve_materialized and isinstance(active_bundle, dict) and active_bundle.get(
            "materialized"
        ) is True:
            active_bundle = {
                **active_bundle,
                "valid_through_object_scene_epoch": object_scene_epoch,
            }
            bundles[previous_bundle_id] = active_bundle
            public_value = state.get("public")
            public = dict(public_value) if isinstance(public_value, dict) else {}
            public.update(
                {
                    "status": "ready",
                    "bundle_id": previous_bundle_id,
                    "call_parameters": {"bundle_id": previous_bundle_id},
                    "materialized": True,
                    "freshness_scope": "attachment_bound_plan",
                    "valid_through_object_scene_epoch": object_scene_epoch,
                }
            )
            self.facts[ANYPLACE_INPUT_BUNDLES_KEY] = _memory_fact_entry(
                {**state, "public": public, "bundles": bundles},
                source=f"{source}_attachment_bound_plan_carry",
            )
            self.record(
                "anyplace_input_bundle_carried",
                {
                    "bundle_id": previous_bundle_id,
                    "tool": tool,
                    "object_scene_epoch": object_scene_epoch,
                },
            )
            return
        public = {
            "schema_version": "openeta.anyplace_input_bundle.v1",
            "status": "stale_object_scene",
            "tool": "anyplace",
            "previous_bundle_id": previous_bundle_id,
            "object_scene_epoch": object_scene_epoch,
            "recovery": "regenerate targeted grasp and placement evidence",
        }
        self.facts[ANYPLACE_INPUT_BUNDLES_KEY] = _memory_fact_entry(
            {
                **state,
                "active_bundle_id": "",
                "public": public,
            },
            source=f"{source}_object_scene_invalidation",
        )
        self.record(
            "anyplace_input_bundle_invalidated",
            {
                "bundle_id": previous_bundle_id,
                "tool": tool,
                "object_scene_epoch": object_scene_epoch,
            },
        )

    def _capture_gripper_command_state(self, action: EnvAction) -> bool:
        call = _successful_tool_call(action, "gripper_control")
        if call is None:
            return False
        parameters = _executed_tool_parameters(action, "gripper_control")
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
        outputs = _tool_call_outputs(call)
        proxy_value = outputs.get("attachment_proxy_receipt")
        proxy = dict(proxy_value) if isinstance(proxy_value, dict) else None
        previous = self.gripper_command_state() or {}
        if position == 1 and previous.get("position") == 0:
            self._invalidate_active_contact_geometry(
                reason="gripper_reopened_after_close",
            )
            self.facts.pop(ATTACHMENT_EVIDENCE_KEY, None)
        self._set_gripper_command_state(
            position,
            source="acknowledged_gripper_command",
            attachment_proxy_receipt=proxy,
        )
        return True

    def _set_gripper_command_state(
        self,
        position: int,
        *,
        source: str,
        attachment_proxy_receipt: JsonDict | None = None,
    ) -> None:
        state = {
            "schema_version": "openeta.gripper_command_state.v1",
            "position": position,
            "state": "open" if position == 1 else "closed",
            "latched": True,
            "scene_epoch": self.scene_epoch(),
            "updated_at_s": time.time(),
        }
        if position == 0 and isinstance(attachment_proxy_receipt, dict):
            state["attachment_proxy_receipt"] = dict(attachment_proxy_receipt)
        self.facts[GRIPPER_COMMAND_STATE_KEY] = _memory_fact_entry(state, source=source)
        self.record("gripper_command_state_changed", dict(state))

    def _invalidate_active_contact_geometry(self, *, reason: str) -> bool:
        provenance = _memory_fact_value(self.facts.get(GRASP_PROVENANCE_KEY))
        if not isinstance(provenance, dict):
            return False
        if provenance.get("contact_geometry_invalidated_at_s"):
            return False
        invalidated = dict(provenance)
        invalidated.update(
            {
                "contact_geometry_invalidated_at_s": time.time(),
                "contact_geometry_invalidated_by": reason,
            }
        )
        self.facts[GRASP_PROVENANCE_KEY] = _memory_fact_entry(
            invalidated,
            source="contact_geometry_evidence_invalidation",
        )
        self.record(
            "compiled_grasp_contact_geometry_invalidated",
            {
                "compiled_grasp_id": invalidated.get("compiled_grasp_id"),
                "evidence_id": invalidated.get("evidence_id"),
                "reason": reason,
            },
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
        probe = self.articulated_attachment_probe()
        if not isinstance(probe, dict) or probe.get("status") != "completed":
            return False
        if str(outputs.get("probe_id") or "") != str(probe.get("probe_id") or ""):
            return False
        if str(outputs.get("candidate_id") or "") != str(probe.get("candidate_id") or ""):
            return False
        verdict = str(outputs.get("verdict") or "").upper()
        if verdict not in {"PASS", "FAIL", "UNKNOWN"}:
            return False
        gate = {
            "schema_version": "openeta.attachment_evidence.v1",
            "probe_id": probe.get("probe_id"),
            "compiled_grasp_id": probe.get("compiled_grasp_id"),
            "candidate_id": probe.get("candidate_id"),
            "object_scene_epoch": self.object_scene_epoch(),
            "verdict": verdict,
            "evidence_source": "independent_attachment_reviewer",
            "assessment_reason": outputs.get("reason"),
            "updated_at_s": time.time(),
        }
        self.facts[ATTACHMENT_EVIDENCE_KEY] = _memory_fact_entry(
            gate,
            source="independent_attachment_reviewer",
        )
        self.record("articulated_attachment_probe_verdict", dict(gate))
        return True

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
        executed_parameters = _executed_tool_parameters(action, name)
        reconciliation = {
            "status": "required",
            "tool": name,
            "intended_parameters": dict(executed_parameters),
            "candidate_id": _parameters_grasp_candidate_id(executed_parameters),
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
                self._advance_runtime_epochs(
                    tool="gripper_control",
                    object_scene_changed=False,
                    source="reconciled_world_mutation",
                )
                position = _binary_gripper_position(requested_position)
                if position is not None:
                    self._set_gripper_command_state(
                        position,
                        source="reconciled_gripper_command",
                    )
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
                self._advance_runtime_epochs(
                    tool=tool_name or "move_to",
                    object_scene_changed=False,
                    source="reconciled_world_mutation",
                )
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
        return True

    def _append_transition_ledger(self, action: EnvAction) -> None:
        command = action.command if isinstance(action.command, dict) else {}
        request = command.get("request")
        request = request if isinstance(request, dict) else {}
        name = str(request.get("name") or "")
        call = _tool_call(action, name)
        executed_parameters = _executed_tool_parameters(action, name)
        row = {
            "index": len(self.transition_ledger()),
            "timestamp_s": time.time(),
            "scene_epoch": self.scene_epoch(),
            "tool": name,
            "effect": "world_mutating"
            if name in {"move_to", "gripper_control", "follow_eef_trajectory"}
            else "other",
            "candidate_id": _parameters_grasp_candidate_id(executed_parameters),
            "verdict": _transition_call_verdict(call),
        }
        rows = [*self.transition_ledger(), row]
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
            {"rows": rows},
            source="runtime_transition_ledger",
        )
        self._save_working_memory()

    def delete_memory(self, key: str, *, namespace: str = "all") -> JsonDict:
        deleted: JsonDict = {}
        if namespace in {"all", "facts"}:
            deleted["facts"] = self.facts.pop(key, None) is not None
            self.agent_working_state.pop(key, None)
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
        self.agent_working_state.clear()
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
            f"agent_working_state={list(self.agent_working_state)}",
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

    def recent_events(self, limit: int | None = 8) -> list[MemoryEvent]:
        if limit is None:
            return list(self.events)
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

    def planning_context(self, *, max_events: int | None = 8) -> JsonDict:
        """Return compact context suitable for a planner prompt or policy."""

        return {
            "session_id": self.session_id,
            "task": self.current_user_request or self.task,
            "session_initial_task": self.task,
            "current_user_request": self.current_user_request,
            "active_environment_task": self.active_environment_task(),
            "conversation": self.conversation.planning_context(max_items=0),
            "metadata": self.metadata,
            "pending_target_selection": self.pending_sam3_selection(),
            "selected_sam3_detection": self.selected_sam3_detection(),
            "selected_sam3_detections": self.selected_sam3_detections(),
            "pending_reference_localization": self.pending_reference_localization(),
            "target_asset_reference": self.target_asset_reference(),
            "target_identity_anchor": self.target_identity_anchor(),
            "reference_localization_failure": self.reference_localization_failure(),
            "tool_health": self.tool_health(),
            "sam3_no_detection": self.sam3_no_detection(),
            "sam3_no_detections": self.sam3_no_detections(),
            "retained_targeted_grasp": (
                self.retained_targeted_grasp()
            ),
            "provenance_evidence_graph": self.provenance_evidence_graph(),
            "grasp_adjustment_budget": self.grasp_adjustment_budget(),
            "grasp_input_bundle": self.grasp_input_bundle(),
            "wrist_alignment_bundle": self.wrist_alignment_bundle(),
            "anyplace_input_bundle": self.anyplace_input_bundle(),
            "articulated_attachment_probe": self.articulated_attachment_probe(),
            "gripper_command_state": self.gripper_command_state(),
            "attachment_evidence": self.attachment_evidence(),
            "placement_world_reference": self.placement_world_reference(),
            "motion_reconciliation": self.motion_reconciliation(),
            "ik_preview_receipts": self.ik_preview_receipts(),
            "controller_capabilities": self.controller_capabilities(),
            "scene_epoch": self.scene_epoch(),
            "object_scene_epoch": self.object_scene_epoch(),
            "robot_motion_epoch": self.robot_motion_epoch(),
            "transition_ledger": self.transition_ledger(),
            "latest_compiled_clearance_execution": (
                self.latest_compiled_clearance_execution()
            ),
            "latest_compiled_contact_execution": self.latest_compiled_contact_execution(),
            "latest_environment_receipt": self.latest_environment_receipt(),
            "world_evidence": self.world_evidence_context(),
            "latest_human_interaction": self.latest_human_interaction(),
            "latest_guidance_interaction": self.latest_guidance_interaction(),
            "agent_working_state": dict(self.agent_working_state),
            "working_memory": {
                "facts": {
                    key: value
                    for key, value in self.facts.items()
                    if key
                    not in {
                        PENDING_SAM3_SELECTION_KEY,
                        SELECTED_SAM3_DETECTION_KEY,
                        SELECTED_SAM3_DETECTIONS_KEY,
                        PENDING_REFERENCE_LOCALIZATION_KEY,
                        REFERENCE_LOCALIZATION_FAILURE_KEY,
                        TOOL_HEALTH_KEY,
                        TARGET_ASSET_REFERENCE_KEY,
                        TARGET_IDENTITY_ANCHOR_KEY,
                        SAM3_NO_DETECTION_KEY,
                        SAM3_NO_DETECTIONS_KEY,
                        ARTICULATED_ATTACHMENT_PROBE_KEY,
                        GRIPPER_COMMAND_STATE_KEY,
                        ATTACHMENT_EVIDENCE_KEY,
                        PLACEMENT_WORLD_REFERENCE_KEY,
                        MOTION_RECONCILIATION_KEY,
                        IK_PREVIEW_RECEIPTS_KEY,
                        OBSERVED_OBJECT_SCENE_CHANGE_KEY,
                        SCENE_EPOCH_KEY,
                        OBJECT_SCENE_EPOCH_KEY,
                        ROBOT_MOTION_EPOCH_KEY,
                        GRASP_PROVENANCE_KEY,
                        GRASP_INPUT_BUNDLES_KEY,
                        WRIST_ALIGNMENT_BUNDLES_KEY,
                        ANYPLACE_INPUT_BUNDLES_KEY,
                        TRANSITION_LEDGER_KEY,
                        ACTIVE_ENVIRONMENT_TASK_KEY,
                        LATEST_COMPILED_CLEARANCE_EXECUTION_KEY,
                        LATEST_COMPILED_CONTACT_EXECUTION_KEY,
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
        agent_working_state = memory.get("agent_working_state", {})
        artifacts = memory.get("artifacts", {})
        skill_notes = memory.get("skill_notes", {})
        removed_task_policy_entries: list[str] = []
        if isinstance(facts, dict):
            self.facts = dict(facts)
            removed_task_policy_entries.extend(purge_removed_task_policy_facts(self.facts))
        if isinstance(agent_working_state, dict) and agent_working_state:
            self.agent_working_state = dict(agent_working_state)
            removed_task_policy_entries.extend(
                purge_removed_task_policy_facts(self.agent_working_state)
            )
        else:
            self.agent_working_state = {
                str(key): {
                    **dict(entry),
                    "ownership": "agent",
                    "freshness": "agent_managed",
                }
                for key, entry in self.facts.items()
                if isinstance(entry, dict) and entry.get("source") == "save_memory"
            }
        if isinstance(artifacts, dict):
            self.artifacts = dict(artifacts)
        if isinstance(skill_notes, dict):
            self.skill_notes = {
                str(skill): list(notes) if isinstance(notes, list) else []
                for skill, notes in skill_notes.items()
            }
        self.compact_summary = str(memory.get("compact_summary", ""))
        if removed_task_policy_entries:
            self._save_working_memory()

    def model_conversation_messages(
        self,
        *,
        max_message_chars: int | None = None,
        max_action_groups: int | None = None,
    ) -> list[JsonDict]:
        """Return canonical chat messages in provider-compatible form."""

        return self.conversation.model_messages(
            max_message_chars=max_message_chars,
            max_action_groups=max_action_groups,
        )

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


def _articulated_probe_path_sha256(tool_name: str, parameters: JsonDict) -> str:
    poses: list[object]
    if tool_name == "move_to":
        poses = [parameters.get("target_pose")]
    elif tool_name == "follow_eef_trajectory":
        trajectory = parameters.get("trajectory")
        poses = list(trajectory) if isinstance(trajectory, list) else []
    else:
        return ""
    markers = {
        str(pose.get("probe_path_sha256") or "")
        for pose in poses
        if isinstance(pose, dict) and pose.get("probe_path_sha256")
    }
    return next(iter(markers)) if len(markers) == 1 else ""


def _articulated_probe_motion_equivalent(
    frozen_parameters: object,
    resolved_parameters: object,
) -> bool:
    """Compare frozen probe geometry while ignoring typed handoff bookkeeping."""

    if not isinstance(frozen_parameters, dict) or not isinstance(
        resolved_parameters, dict
    ):
        return False
    if resolved_parameters.get("enable_collision_check") is not True:
        return False
    if "target_pose" in frozen_parameters:
        return resolved_parameters.get("target_pose") == frozen_parameters.get(
            "target_pose"
        )
    if "trajectory" in frozen_parameters:
        return resolved_parameters.get("trajectory") == frozen_parameters.get(
            "trajectory"
        )
    return False




def _finite_xyz(value: object) -> bool:
    return (
        isinstance(value, list | tuple)
        and len(value) >= 3
        and all(
            isinstance(component, int | float) and math.isfinite(float(component))
            for component in value[:3]
        )
    )


def _is_wrist_camera(frame_id: str, role: str) -> bool:
    marker = f"{frame_id} {role}".lower()
    return any(name in marker for name in ("wrist", "hand", "gripper", "eye_in_hand"))


def _rounded_xyz(value: list[float]) -> list[float]:
    return [round(float(component), 6) for component in value[:3]]


def _finite_nonnegative_float(value: object, *, default: float) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return default
    parsed = float(value)
    return parsed if math.isfinite(parsed) and parsed >= 0.0 else default


def _compiled_grasp_pose_role(pose: JsonDict) -> str:
    waypoint = str(pose.get("waypoint_role") or "").strip().lower()
    if waypoint == "grasp_contact":
        return "contact"
    if waypoint == "grasp_precontact":
        return "precontact"
    if waypoint in {"grasp_clearance", "grasp_alignment_reference"}:
        return "clearance"
    return ""


def _compiled_grasp_reference_pose(
    compiled: JsonDict,
    *,
    role: str,
) -> JsonDict | None:
    field = {
        "clearance": "hover_pose",
        "precontact": "precontact_pose",
        "contact": "contact_pose",
    }.get(role)
    value = compiled.get(field) if field else None
    return value if isinstance(value, dict) else None


def _pose_near_xyz(
    pose: object,
    reference_xyz: object,
    *,
    tolerance_m: float,
) -> bool:
    if not isinstance(pose, dict):
        return False
    xyz = pose.get("xyz")
    if not (_finite_xyz(xyz) and _finite_xyz(reference_xyz)):
        return False
    distance = math.sqrt(
        sum(
            (float(xyz[index]) - float(reference_xyz[index])) ** 2
            for index in range(3)
        )
    )
    return distance <= tolerance_m


def _finite_number(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _call_result_success(call: JsonDict) -> bool:
    result = call.get("result")
    return isinstance(result, dict) and bool(result.get("success"))


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


def _executed_tool_parameters(action: EnvAction, name: str) -> JsonDict:
    """Return the host-resolved parameters actually dispatched to a tool.

    Typed public requests intentionally contain compact ids rather than copied
    geometry. Post-execution memory first consumes the environment-authority
    execution receipt, then the older trusted pose-feedback representation, and
    finally legacy/synthetic call or request parameters.
    """

    call = _tool_call(action, name)
    result = call.get("result") if isinstance(call, dict) else None
    details = result.get("details") if isinstance(result, dict) else None
    details = details if isinstance(details, dict) else {}
    provenance = details.get("host_provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    trusted_environment_result = (
        provenance.get("schema_version") == "openeta.tool_result_provenance.v1"
        and provenance.get("authority") == "environment"
        and str(provenance.get("tool") or "") == name
    )
    if trusted_environment_result:
        receipt = details.get("host_execution_receipt")
        receipt = receipt if isinstance(receipt, dict) else {}
        receipt_parameters = receipt.get("parameters")
        if (
            receipt.get("schema_version") == "openeta.resolved_tool_execution.v1"
            and str(receipt.get("tool") or "") == name
            and isinstance(receipt_parameters, dict)
        ):
            return dict(receipt_parameters)

    call_parameters = call.get("parameters") if isinstance(call, dict) else None
    resolved = dict(call_parameters) if isinstance(call_parameters, dict) else {}
    if trusted_environment_result and name == "move_to":
        outputs = details.get("outputs")
        outputs = outputs if isinstance(outputs, dict) else {}
        pose_feedback = outputs.get("pose_feedback")
        pose_feedback = pose_feedback if isinstance(pose_feedback, dict) else {}
        requested_pose = pose_feedback.get("requested_eef_pose")
        if isinstance(requested_pose, dict) and requested_pose:
            resolved["target_pose"] = dict(requested_pose)
            return resolved
    if isinstance(call_parameters, dict):
        return resolved
    command = action.command if isinstance(action.command, dict) else {}
    request = command.get("request")
    if (
        isinstance(request, dict)
        and str(request.get("name") or "") == name
        and isinstance(request.get("parameters"), dict)
    ):
        return dict(request["parameters"])
    return {}


def _pipeline_call(action: EnvAction, name: str) -> JsonDict | None:
    """Find a named call in either normal tools or inline safety checks."""

    command = action.command if isinstance(action.command, dict) else {}
    for field_name in ("tool_calls", "safety_checks"):
        for call in command.get(field_name, []) or []:
            if isinstance(call, dict) and str(call.get("name") or "") == name:
                return call
    return None


IK_POSE_EQUIVALENCE_POSITION_ATOL_M = 1e-6
IK_POSE_EQUIVALENCE_ORIENTATION_ATOL = 1e-6


def _ik_orientation_fields(target_pose: JsonDict) -> JsonDict:
    return {
        key: target_pose.get(key)
        for key in (
            "rotation_matrix",
            "quat_xyzw",
            "quaternion",
            "euler_xyz_deg",
            "rotvec",
            "roll",
            "pitch",
            "yaw",
        )
        if target_pose.get(key) is not None
    }


def _ik_orientation_policy(parameters: JsonDict) -> str:
    target_pose = parameters.get("target_pose")
    target_pose = target_pose if isinstance(target_pose, dict) else {}
    orientation_fields = _ik_orientation_fields(target_pose)
    preserve_current = parameters.get("preserve_current_orientation")
    if preserve_current is None:
        preserve_current = not orientation_fields
    return "preserve_current" if preserve_current is True else "explicit_orientation"


def ik_pose_policy_numerically_equivalent(
    receipt: JsonDict,
    parameters: JsonDict,
) -> bool:
    """Accept harmless JSON float round-trips without authorizing a new pose.

    Signatures remain the fast exact index.  This fallback compares the stored
    receipt target numerically at micron/component-level tolerances, far below
    controller execution tolerances.  It therefore removes serialization
    brittleness while still requiring a fresh preview for Agent-authored pose
    adjustments.
    """

    requested = parameters.get("target_pose")
    recorded = receipt.get("target_pose")
    if not isinstance(requested, dict) or not isinstance(recorded, dict):
        return False
    requested_xyz = requested.get("xyz", requested.get("translation_xyz"))
    recorded_xyz = recorded.get("xyz", recorded.get("translation_xyz"))
    if not _finite_xyz(requested_xyz) or not _finite_xyz(recorded_xyz):
        return False
    if any(
        abs(float(requested_xyz[index]) - float(recorded_xyz[index]))
        > IK_POSE_EQUIVALENCE_POSITION_ATOL_M
        for index in range(3)
    ):
        return False

    requested_policy = _ik_orientation_policy(parameters)
    recorded_policy = str(receipt.get("orientation_policy") or "")
    if not recorded_policy:
        recorded_policy = (
            "explicit_orientation"
            if _ik_orientation_fields(recorded)
            else "preserve_current"
        )
    if requested_policy != recorded_policy:
        return False
    if requested_policy == "preserve_current":
        return True

    requested_rotation = _ik_orientation_rotation_matrix(requested)
    recorded_rotation = _ik_orientation_rotation_matrix(recorded)
    if requested_rotation is None or recorded_rotation is None:
        return False
    relative_trace = sum(
        requested_rotation[row][column] * recorded_rotation[row][column]
        for row in range(3)
        for column in range(3)
    )
    angle_rad = math.acos(max(-1.0, min(1.0, (relative_trace - 1.0) / 2.0)))
    return angle_rad <= IK_POSE_EQUIVALENCE_ORIENTATION_ATOL


def _ik_orientation_rotation_matrix(target_pose: JsonDict) -> list[list[float]] | None:
    """Canonicalize supported explicit orientations for semantic IK matching."""

    direct_euler = [target_pose.get(axis) for axis in ("roll", "pitch", "yaw")]
    if any(value is not None for value in direct_euler):
        if not all(_finite_number(value) for value in direct_euler):
            return None
        return _xyz_euler_degrees_to_rotation_matrix(direct_euler)

    euler = target_pose.get("euler_xyz_deg")
    if _finite_numeric_vector(euler, length=3):
        return _xyz_euler_degrees_to_rotation_matrix(euler)

    quaternion = target_pose.get("quat_xyzw", target_pose.get("quaternion"))
    if _finite_numeric_vector(quaternion, length=4):
        qx, qy, qz, qw = [float(value) for value in quaternion]
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm <= 1e-12:
            return None
        qx, qy, qz, qw = [value / norm for value in (qx, qy, qz, qw)]
        return [
            [
                1.0 - 2.0 * (qy * qy + qz * qz),
                2.0 * (qx * qy - qz * qw),
                2.0 * (qx * qz + qy * qw),
            ],
            [
                2.0 * (qx * qy + qz * qw),
                1.0 - 2.0 * (qx * qx + qz * qz),
                2.0 * (qy * qz - qx * qw),
            ],
            [
                2.0 * (qx * qz - qy * qw),
                2.0 * (qy * qz + qx * qw),
                1.0 - 2.0 * (qx * qx + qy * qy),
            ],
        ]

    rotation = target_pose.get("rotation_matrix")
    if (
        isinstance(rotation, list | tuple)
        and len(rotation) == 3
        and all(_finite_numeric_vector(row, length=3) for row in rotation)
    ):
        matrix = [[float(value) for value in row] for row in rotation]
        return matrix if _proper_rotation_matrix(matrix) else None

    rotvec = target_pose.get("rotvec")
    if _finite_numeric_vector(rotvec, length=3):
        vector = [float(value) for value in rotvec]
        angle = math.sqrt(sum(value * value for value in vector))
        if angle <= 1e-12:
            return [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        x, y, z = [value / angle for value in vector]
        cosine = math.cos(angle)
        sine = math.sin(angle)
        one_minus_cosine = 1.0 - cosine
        return [
            [
                cosine + x * x * one_minus_cosine,
                x * y * one_minus_cosine - z * sine,
                x * z * one_minus_cosine + y * sine,
            ],
            [
                y * x * one_minus_cosine + z * sine,
                cosine + y * y * one_minus_cosine,
                y * z * one_minus_cosine - x * sine,
            ],
            [
                z * x * one_minus_cosine - y * sine,
                z * y * one_minus_cosine + x * sine,
                cosine + z * z * one_minus_cosine,
            ],
        ]
    return None


def _finite_numeric_vector(value: object, *, length: int) -> bool:
    return (
        isinstance(value, list | tuple)
        and len(value) == length
        and all(_finite_number(item) for item in value)
    )


def _xyz_euler_degrees_to_rotation_matrix(values: object) -> list[list[float]]:
    if not _finite_numeric_vector(values, length=3):
        raise ValueError("xyz Euler orientation requires three finite values")
    numeric_values = values if isinstance(values, list | tuple) else ()
    roll, pitch, yaw = [
        math.radians(float(value)) for value in numeric_values
    ]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return [
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ]


def _proper_rotation_matrix(matrix: list[list[float]]) -> bool:
    for row in range(3):
        for column in range(3):
            dot = sum(matrix[index][row] * matrix[index][column] for index in range(3))
            expected = 1.0 if row == column else 0.0
            if abs(dot - expected) > 1e-3:
                return False
    determinant = (
        matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    )
    return abs(determinant - 1.0) <= 1e-3


def _ik_target_signature(parameters: JsonDict) -> str:
    target_pose = parameters.get("target_pose")
    if not isinstance(target_pose, dict) or not _finite_xyz(
        target_pose.get("xyz", target_pose.get("translation_xyz"))
    ):
        return ""
    orientation_fields = _ik_orientation_fields(target_pose)
    canonical = {
        "target_xyz": target_pose.get("xyz", target_pose.get("translation_xyz")),
        "orientation_policy": _ik_orientation_policy(parameters),
        "orientation": orientation_fields,
        "position_tolerance_m": parameters.get(
            "position_tolerance_m", parameters.get("tolerance")
        ),
        "orientation_tolerance_rad": parameters.get(
            "orientation_tolerance_rad", parameters.get("ori_tolerance")
        ),
        "check_endpoint_collision": parameters.get("check_endpoint_collision"),
    }
    return hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:24]


def _ik_pose_policy_signature(parameters: JsonDict) -> str:
    """Hash only the endpoint and orientation policy shared by preview and move."""

    target_pose = parameters.get("target_pose")
    if not isinstance(target_pose, dict) or not _finite_xyz(
        target_pose.get("xyz", target_pose.get("translation_xyz"))
    ):
        return ""
    orientation_fields = _ik_orientation_fields(target_pose)
    canonical = {
        "target_xyz": target_pose.get("xyz", target_pose.get("translation_xyz")),
        "orientation_policy": _ik_orientation_policy(parameters),
        "orientation": orientation_fields,
    }
    return hashlib.sha256(
        json.dumps(
            canonical,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:24]


def _target_pose_has_no_orientation(target_pose: JsonDict) -> bool:
    return not any(
        target_pose.get(key) is not None
        for key in (
            "rotation_matrix",
            "quat_xyzw",
            "quaternion",
            "rotvec",
            "roll",
            "pitch",
            "yaw",
        )
    )


def _ik_reachability_classification(reachability: JsonDict) -> str:
    status = str(reachability.get("status") or "unknown").lower()
    if status == "reachable":
        return "feasible"
    if (
        status == "unknown"
        and str(reachability.get("reason_code") or "")
        == "endpoint_collision_check_unavailable"
        and isinstance(reachability.get("best_candidate"), dict)
    ):
        return "kinematically_feasible_collision_deferred"
    if status == "unknown":
        return "inconclusive"
    collision = reachability.get("collision")
    if isinstance(collision, dict) and collision.get("detected") is True:
        return "hard_infeasible"
    reason = str(reachability.get("reason_code") or "").lower()
    hard_markers = (
        "outside_workspace",
        "joint_limit",
        "endpoint_collision",
        "self_collision",
        "invalid_target",
    )
    if any(marker in reason for marker in hard_markers):
        return "hard_infeasible"
    if (
        reachability.get("position_only_reachable") is True
        or reachability.get("orientation_only_reachable") is True
        or isinstance(reachability.get("best_candidate"), dict)
        or bool(reachability.get("suggestions"))
    ):
        return "repairable"
    return "hard_infeasible"


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


def _motion_call_changed_pose(call: JsonDict) -> bool:
    """Return whether a successful motion receipt proves robot/camera motion.

    Missing legacy receipts remain conservatively mutating. A modern receipt with
    zero executed steps and identical start/end poses is an acknowledged no-op and
    must not invalidate visual evidence or IK receipts by advancing the robot epoch.
    """

    result = call.get("result")
    details = result.get("details") if isinstance(result, dict) else None
    if not isinstance(details, dict):
        return True
    outputs = details.get("outputs")
    outputs = outputs if isinstance(outputs, dict) else {}
    response = outputs.get("response")
    response = response if isinstance(response, dict) else {}
    state_delta = details.get("state_delta")
    state_delta = state_delta if isinstance(state_delta, dict) else {}
    candidates = (
        outputs.get("motion_summary"),
        response.get("motion_summary"),
        details.get("motion_summary"),
        state_delta.get("motion"),
    )
    for motion in candidates:
        if not isinstance(motion, dict):
            continue
        steps = motion.get("steps_executed")
        if isinstance(steps, int | float) and not isinstance(steps, bool):
            if float(steps) > 0:
                return True
            start = motion.get("start")
            end = motion.get("end")
            start = start if isinstance(start, dict) else {}
            end = end if isinstance(end, dict) else {}
            if _poses_materially_differ(start, end):
                return True
            return False
    return True


def _motion_call_receipt_proves_pose_change(call: JsonDict) -> bool:
    """Require positive execution evidence before mutating state on failure.

    Successful legacy calls remain conservatively mutating via
    ``_motion_call_changed_pose``. Failed or blocked calls are different: only a
    structured receipt with executed steps or materially different start/end poses
    may advance ``robot_motion_epoch``.
    """

    result = call.get("result")
    details = result.get("details") if isinstance(result, dict) else None
    if not isinstance(details, dict):
        return False
    outputs = details.get("outputs")
    outputs = outputs if isinstance(outputs, dict) else {}
    response = outputs.get("response")
    response = response if isinstance(response, dict) else {}
    state_delta = details.get("state_delta")
    state_delta = state_delta if isinstance(state_delta, dict) else {}
    candidates = (
        outputs.get("motion_summary"),
        response.get("motion_summary"),
        details.get("motion_summary"),
        state_delta.get("motion"),
    )
    for motion in candidates:
        if not isinstance(motion, dict):
            continue
        steps = motion.get("steps_executed")
        if (
            isinstance(steps, int | float)
            and not isinstance(steps, bool)
            and float(steps) > 0
        ):
            return True
        start = motion.get("start")
        end = motion.get("end")
        if isinstance(start, dict) and isinstance(end, dict) and _poses_materially_differ(
            start, end
        ):
            return True
    return False


def _poses_materially_differ(start: JsonDict, end: JsonDict) -> bool:
    start_xyz = start.get("xyz")
    end_xyz = end.get("xyz")
    if _finite_xyz(start_xyz) and _finite_xyz(end_xyz):
        if math.sqrt(
            sum(
                (float(start_xyz[index]) - float(end_xyz[index])) ** 2
                for index in range(3)
            )
        ) > 1e-6:
            return True
    start_quat = start.get("quat_xyzw")
    end_quat = end.get("quat_xyzw")
    if (
        isinstance(start_quat, list | tuple)
        and isinstance(end_quat, list | tuple)
        and len(start_quat) >= 4
        and len(end_quat) >= 4
    ):
        try:
            return min(
                sum((float(start_quat[i]) - float(end_quat[i])) ** 2 for i in range(4)),
                sum((float(start_quat[i]) + float(end_quat[i])) ** 2 for i in range(4)),
            ) ** 0.5 > 1e-6
        except (TypeError, ValueError):
            return False
    return False


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


def _optional_int(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_sam3_evidence_role(value: object) -> str:
    role = str(value or DEFAULT_SAM3_EVIDENCE_ROLE).strip().lower()
    if role not in SAM3_EVIDENCE_ROLES:
        raise ValueError(
            "evidence_role must be one of " + ", ".join(sorted(SAM3_EVIDENCE_ROLES)) + "."
        )
    return role


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


def _binary_gripper_position(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if not isinstance(value, int | float) or isinstance(value, bool):
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or numeric not in {0.0, 1.0}:
        return None
    return int(numeric)


def _motion_call_rejects_candidate(call: JsonDict) -> bool:
    result = call.get("result")
    if not isinstance(result, dict):
        return False
    details = result.get("details")
    if not isinstance(details, dict):
        return False
    if _structured_motion_reached_target(details) is False:
        if _safe_clearance_pose_is_near_target(details):
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


def _safe_clearance_pose_is_near_target(details: JsonDict) -> bool:
    parameters = details.get("parameters")
    target_pose = parameters.get("target_pose") if isinstance(parameters, dict) else None
    waypoint_role = (
        str(target_pose.get("waypoint_role") or "") if isinstance(target_pose, dict) else ""
    )
    if waypoint_role not in {
        "grasp_clearance",
        "grasp_refinement_clearance",
        "grasp_alignment_reference",
    }:
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


def _structured_motion_summary(details: JsonDict) -> JsonDict:
    outputs = details.get("outputs")
    outputs = outputs if isinstance(outputs, dict) else {}
    response = outputs.get("response")
    response = response if isinstance(response, dict) else {}
    state_delta = details.get("state_delta")
    state_delta = state_delta if isinstance(state_delta, dict) else {}
    for source in (
        outputs.get("motion_summary"),
        response.get("motion_summary"),
        details.get("motion_summary"),
        state_delta.get("motion"),
    ):
        if isinstance(source, dict):
            return dict(source)
    return {}








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


def _fact_epoch(entry: JsonDict | None) -> int:
    value = _memory_fact_value(entry) or {}
    try:
        return max(0, int(value.get("epoch") or 0))
    except (TypeError, ValueError):
        return 0


def _fact_epoch_value(value: object) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _same_memory_artifact_path(left: object, right: object) -> bool:
    if not isinstance(left, str) or not left or not isinstance(right, str) or not right:
        return False
    try:
        return Path(left).expanduser().resolve(strict=False) == Path(right).expanduser().resolve(
            strict=False
        )
    except (OSError, RuntimeError):
        return left == right


def _same_memory_artifact_content(left: object, right: object) -> bool:
    """Return true only when two distinct session files are byte-identical."""

    if not isinstance(left, str) or not left or not isinstance(right, str) or not right:
        return False
    try:
        left_path = Path(left).expanduser().resolve(strict=True)
        right_path = Path(right).expanduser().resolve(strict=True)
        if not left_path.is_file() or not right_path.is_file():
            return False
        if left_path.stat().st_size != right_path.stat().st_size:
            return False
        left_digest = hashlib.sha256()
        right_digest = hashlib.sha256()
        with left_path.open("rb") as left_handle, right_path.open("rb") as right_handle:
            while True:
                left_chunk = left_handle.read(1024 * 1024)
                right_chunk = right_handle.read(1024 * 1024)
                if left_chunk != right_chunk:
                    return False
                if not left_chunk:
                    break
                left_digest.update(left_chunk)
                right_digest.update(right_chunk)
        return left_digest.digest() == right_digest.digest()
    except (OSError, RuntimeError, ValueError):
        return False


def _same_grasp_candidate(left: JsonDict, right: JsonDict) -> bool:
    # Candidate payloads make a JSON round-trip through the planner before the
    # Agent explicitly compiles one. Requiring bit-identical floats makes a
    # harmless representation change (observed at ~8e-14 in a rotation-matrix
    # element) look like forged geometry and leaves the previous grasp branch
    # active. Keep identity/frame fields exact and accept only machine-noise
    # numeric drift; meaningful Agent-authored geometry edits still fail.
    if any(
        left.get(field) != right.get(field)
        for field in ("id", "frame", "camera_frame")
    ):
        return False
    return all(
        _same_numeric_candidate_field(left.get(field), right.get(field))
        for field in (
            "translation_xyz",
            "rotation_matrix",
            "gripper_tip_position_xyz",
            "depth",
            "width",
            "height",
        )
    )


def _same_numeric_candidate_field(left: object, right: object) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        left_value = float(left)
        right_value = float(right)
        return math.isfinite(left_value) and math.isfinite(right_value) and math.isclose(
            left_value,
            right_value,
            rel_tol=1e-12,
            abs_tol=1e-12,
        )
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(
            _same_numeric_candidate_field(left_item, right_item)
            for left_item, right_item in zip(left, right)
        )
    return left == right


def _placement_evidence_id(selected: JsonDict) -> str:
    result_id = str(selected.get("result_id") or "unknown")
    detection_id = str(selected.get("id") or "unknown")
    return f"placement:{result_id}:{detection_id}"


def _reusable_fixed_camera_placement(
    bundles: JsonDict,
    *,
    placement_evidence_id: str,
    expected_source: JsonDict,
) -> JsonDict | None:
    """Return prior placement evidence reusable on an unchanged fixed camera.

    AnyPlace still receives RGB-D from the new grasp source.  Only the stable
    receptacle mask is reused, and the decision is restricted to matching
    intrinsics and a non-wrist camera identity.  Wrist/hand cameras can move and
    therefore always require fresh aligned placement evidence.
    """

    expected_rgb = expected_source.get("rgb")
    expected_frame = _camera_frame_hint(expected_source, expected_rgb)
    expected_intrinsics = expected_source.get("intrinsics")
    if (
        not expected_frame
        or _is_moving_camera_frame(expected_frame)
        or not isinstance(expected_intrinsics, dict)
    ):
        return None
    for bundle_id, candidate in reversed(list(bundles.items())):
        if (
            not isinstance(candidate, dict)
            or candidate.get("placement_evidence_id") != placement_evidence_id
        ):
            continue
        parameters = candidate.get("parameters")
        if not isinstance(parameters, dict):
            continue
        prior_rgb = parameters.get("rgb")
        selected_grasp = parameters.get("selected_grasp")
        prior_source = (
            selected_grasp.get("source")
            if isinstance(selected_grasp, dict)
            else {}
        )
        prior_frame = _camera_frame_hint(
            prior_source if isinstance(prior_source, dict) else {},
            prior_rgb,
        )
        if prior_frame != expected_frame:
            continue
        if parameters.get("intrinsics") != expected_intrinsics:
            continue
        return {"bundle_id": bundle_id, "bundle": candidate}
    return None


def _camera_frame_hint(source: JsonDict, rgb_path: object) -> str:
    for key in ("camera_frame_id", "frame_id", "camera_name"):
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    if not isinstance(rgb_path, str):
        return ""
    match = re.search(r"cameras\.\d+\.([A-Za-z0-9_-]+)\.(?:rgb|depth)\b", rgb_path)
    return match.group(1).lower() if match else ""


def _is_moving_camera_frame(frame_id: str) -> bool:
    lowered = frame_id.lower()
    return any(token in lowered for token in ("wrist", "hand", "gripper"))


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
        artifacts.extend(_extract_compiled_grasp_artifacts(call, details))
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
                "Call enhance_depth with the same source_packet_id and camera_frame_id; "
                "the host resolves this matching prior."
            ),
            "source_packet_id": outputs.get("source_packet_id"),
            "camera_frame_id": outputs.get("camera_frame_id"),
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
    structured_artifact = details.get("structured_artifact")
    if not isinstance(structured_artifact, dict):
        structured_artifact = source.get("structured_artifact")
    if not isinstance(structured_artifact, dict):
        structured_artifact = {}
    artifact_path = str(structured_artifact.get("path") or "")
    preview_count = len(compact_candidates)
    truncated = preview_count < len(candidates)
    return [
        {
            "type": "grasp_candidates",
            "kind": "grasp_candidates",
            "tool": tool_name,
            "index": "latest",
            "candidate_count": len(candidates),
            "result_id": source.get("result_id"),
            "preview_count": preview_count,
            "truncated": truncated,
            "best_grasp_candidate": compact_candidates[0],
            "grasp_candidates": compact_candidates,
            "source_rgb": source.get("source_rgb") or grasp_source.get("rgb"),
            "source_depth": source.get("source_depth") or grasp_source.get("depth"),
            "target_mask": source.get("target_mask") or grasp_source.get("object_mask"),
            "selected_grasp_source": source.get("source"),
            "grasp_selection_bundle": (
                source.get("grasp_selection_bundle")
                if isinstance(source.get("grasp_selection_bundle"), dict)
                else None
            ),
            "grasp_selection_advice": (
                source.get("grasp_selection_advice")
                if isinstance(source.get("grasp_selection_advice"), dict)
                else None
            ),
            "source_tool": grasp_source.get("source_tool") or tool_name,
            "source_backend": grasp_source.get("source_backend")
            or source.get("selected_backend")
            or tool_name,
            "scene_epoch": source.get("scene_epoch"),
            "gripper_name": grasp_source.get("gripper_name"),
            "raw_output_ref": source.get("raw_output_ref") or artifact_path or None,
            "complete_outputs_artifact": structured_artifact or None,
            "query_hint": (
                "Use python_exec artifacts.read_json(path) and inspect "
                "payload['outputs']['grasp_candidates']; filter/rank the complete list "
                "in code before asking a human."
                if artifact_path and truncated
                else None
            ),
        }
    ]


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
    structured_artifact = details.get("structured_artifact")
    if not isinstance(structured_artifact, dict):
        structured_artifact = source.get("structured_artifact")
    if not isinstance(structured_artifact, dict):
        structured_artifact = {}
    artifact_path = str(structured_artifact.get("path") or "")
    preview_count = len(compact_candidates)
    truncated = preview_count < len(candidates)
    return [
        {
            "type": "placement_candidates",
            "kind": "placement_candidates",
            "tool": "anyplace",
            "index": "latest",
            "candidate_count": len(candidates),
            "preview_count": preview_count,
            "truncated": truncated,
            "selected_grasp_id": source.get("selected_grasp_id"),
            "placement_candidates": compact_candidates,
            "source": source.get("source"),
            "raw_output_ref": source.get("raw_output_ref") or artifact_path or None,
            "complete_outputs_artifact": structured_artifact or None,
            "query_hint": (
                "Use python_exec artifacts.read_json(path) and inspect "
                "payload['outputs']['placement_candidates']; filter/rank the complete list "
                "in code before asking a human."
                if artifact_path and truncated
                else None
            ),
        }
    ]


def _extract_compiled_grasp_artifacts(
    call: JsonDict,
    details: JsonDict,
) -> list[JsonDict]:
    """Retain compiled pose evidence without creating a task-stage obligation."""

    if str(call.get("name") or "") != "compile_grasp_seed":
        return []
    outputs = details.get("outputs")
    if (
        not isinstance(outputs, dict)
        or outputs.get("schema_version") != "openeta.compiled_grasp_seed.v1"
    ):
        return []
    return [
        {
            **dict(outputs),
            "type": "compiled_grasp",
            "kind": "compiled_grasp",
            "tool": "compile_grasp_seed",
            "index": str(outputs.get("compiled_grasp_id") or "latest"),
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
            "rotation_matrix": world_pose.get("rotation_matrix") or outputs.get("rotation_matrix"),
            "gripper_tip_position_xyz": world_pose.get("gripper_tip_position_xyz")
            or outputs.get("gripper_tip_position_xyz"),
            "source_grasp_id": world_pose.get("id"),
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
            "result_id",
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
            "schema_version",
            "compiled_grasp_id",
            "candidate_id",
            "scene_epoch",
            "hover_pose",
            "contact_pose",
            "retreat_pose",
            "approach_world_xyz",
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
                    "hover_pose",
                    "contact_pose",
                    "retreat_pose",
                    "approach_world_xyz",
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
        "effect",
        "operational_success",
        "semantic_outcome",
        "facts_produced",
        "recovery_options",
        "candidate_count",
        "best_grasp_candidate",
        "grasp_candidates",
        "ranking",
        "result_id",
        "selection_required",
        "selected_detection",
        "selection_bundle",
        "grasp_selection_bundle",
        "grasp_selection_advice",
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
                "grasp_selection_bundle",
                "grasp_selection_advice",
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
            "grasp_selection_bundle",
            "grasp_selection_advice",
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
                    "result",
                    "best_grasp_candidate",
                    "grasp_candidates",
                    "world_pose",
                    "rotation_matrix",
                    "selected_detection",
                    "selection_bundle",
                    "grasp_selection_bundle",
                    "grasp_selection_advice",
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
        # Unknown output fields are often the most important part of a newly
        # introduced tool contract. Keep every remaining field under the same
        # structural bound instead of silently reducing it to an output-key
        # list; otherwise a tool can execute successfully while its consumer
        # never receives the value needed for the next call.
        for key, value in outputs.items():
            if key in useful_outputs:
                continue
            useful_outputs[str(key)] = _compact_value(
                value,
                max_depth=5,
                max_items=16,
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


def _observation_visual_signature(observation: EnvObservation) -> str:
    """Hash current RGB evidence independently of packet ids and file paths.

    Runtime image artifacts are preferred because hashing encoded files avoids
    expanding large pixel lists.  Unit/embedded adapters that do not materialize
    files fall back to the typed camera RGB payload.  Frame identity is included
    so swapping two camera feeds cannot look like an unchanged observation.
    """

    artifact_digests: list[JsonDict] = []
    artifacts = observation.metadata.get("image_artifacts")
    if isinstance(artifacts, list):
        for artifact in artifacts:
            if not isinstance(artifact, dict) or artifact.get("kind") != "rgb":
                continue
            path_value = artifact.get("path")
            if not isinstance(path_value, str) or not path_value:
                continue
            try:
                path = Path(path_value)
                if not path.is_file():
                    continue
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
            except OSError:
                continue
            artifact_digests.append(
                {
                    "frame_id": str(artifact.get("frame_id") or ""),
                    "role": str(artifact.get("role") or ""),
                    "sha256": digest,
                }
            )
    if not artifact_digests:
        for camera in observation.cameras:
            try:
                encoded = json.dumps(
                    camera.rgb,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode("utf-8")
            except (TypeError, ValueError):
                continue
            artifact_digests.append(
                {
                    "frame_id": camera.frame_id,
                    "role": camera.role,
                    "sha256": hashlib.sha256(encoded).hexdigest(),
                }
            )
    if not artifact_digests:
        return ""
    artifact_digests.sort(
        key=lambda item: (str(item.get("frame_id") or ""), str(item.get("role") or ""))
    )
    return hashlib.sha256(
        json.dumps(
            artifact_digests,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:24]


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
