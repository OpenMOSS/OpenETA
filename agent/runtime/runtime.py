"""OpenETA lightweight agent runtime."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image

from adapter.protocol import EnvAction, EnvObservation, JsonDict
from agent.runtime.artifact_paths import artifact_session_id
from agent.runtime.checkers import should_record_recovery_feedback
from agent.runtime.depth_enhancement import (
    DepthEnhancementConfig,
    DepthPriorPrediction,
    enhance_rgbd_depth,
    materialize_depth_enhancement,
)
from agent.runtime.interfaces import ActionInterfaceRegistry, build_default_action_interfaces
from agent.runtime.memory import AgentMemory, MemoryStore
from agent.runtime.pipeline import ActionPipeline
from agent.runtime.planner import BasePlanner, ToolCallingPlanner
from agent.runtime.rollout import RolloutRecorder, build_rollout_provenance
from agent.runtime.self_improvement import SelfImprovementReviewer
from agent.runtime.skills import SkillRegistry, build_default_skill_registry
from agent.runtime.visual_history import VisualHistoryManager
from agent.tools.coding import PythonExecRuntime
from agent.tools.registry import (
    ToolExecutionContext,
    ToolRegistry,
    ToolResult,
    build_default_tool_registry,
    make_tool_result,
)


class RuntimeExecutionCancelled(RuntimeError):
    """Raised when an episode loses ownership while planner/tool work is in flight."""


def _raise_if_execution_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeExecutionCancelled("episode execution was cancelled")


def _assert_tool_contract_runtime_alignment(planner: object, pipeline: object) -> None:
    """Fail before execution if Planner and Gate use different contract truth."""

    planner_catalog = getattr(planner, "tool_contract_catalog", None)
    pipeline_catalog = getattr(pipeline, "tool_contract_catalog", None)
    if planner_catalog is not None and pipeline_catalog is not None:
        if planner_catalog.to_dict() != pipeline_catalog.to_dict():
            raise ValueError(
                "Planner and ActionPipeline ToolContract catalogs differ; refusing "
                "to start a runtime with split interface authority."
            )
    planner_policy = getattr(planner, "tool_contract_policy", None)
    pipeline_policy = getattr(pipeline, "tool_contract_policy", None)
    if planner_policy is not None and pipeline_policy is not None:
        if planner_policy.to_dict() != pipeline_policy.to_dict():
            raise ValueError(
                "Planner and ActionPipeline ToolContract runtime policies differ; "
                "refusing to start a runtime with split authority."
            )


class OpenEtaAgentRuntime:
    """Owns planner state, memory, tool registry, and skill registry."""

    def __init__(
        self,
        *,
        planner: BasePlanner | None = None,
        memory: AgentMemory | None = None,
        memory_store: MemoryStore | None = None,
        tools: ToolRegistry | None = None,
        skills: SkillRegistry | None = None,
        interfaces: ActionInterfaceRegistry | None = None,
        pipeline: ActionPipeline | None = None,
        self_improvement_reviewer: SelfImprovementReviewer | None = None,
        rollout_recorder: RolloutRecorder | None = None,
        rollout_enabled: bool = True,
        default_session_id: str | None = None,
        visual_history: VisualHistoryManager | None = None,
        startup_facts: dict[str, JsonDict] | None = None,
    ) -> None:
        self.planner = planner or ToolCallingPlanner()
        self.memory = memory or AgentMemory(store=memory_store)
        self.tools = tools or build_default_tool_registry()
        self.skills = skills or build_default_skill_registry()
        self.interfaces = interfaces or build_default_action_interfaces()
        self.pipeline = pipeline or ActionPipeline(interfaces=self.interfaces)
        _assert_tool_contract_runtime_alignment(self.planner, self.pipeline)
        self.self_improvement_reviewer = self_improvement_reviewer or SelfImprovementReviewer()
        self.default_session_id = default_session_id
        self.visual_history = visual_history
        self.startup_facts = {
            str(name): dict(payload)
            for name, payload in (startup_facts or {}).items()
        }
        self.rollout_recorder = rollout_recorder
        if self.rollout_recorder is None and rollout_enabled:
            store_root = getattr(self.memory.store, "root", None)
            if store_root is not None:
                self.rollout_recorder = RolloutRecorder(store_root)
        if isinstance(self.planner, ToolCallingPlanner):
            self.planner.set_rollout_recorder(self.rollout_recorder)
        if self.visual_history is not None:
            self.visual_history.set_rollout_recorder(self.rollout_recorder)
        if self.rollout_recorder is not None:
            self.tools.add_listener(self.rollout_recorder.record_tool_event)
        self._act_lock = threading.Lock()
        self._bind_memory_tool_handlers()

    def start_session(
        self,
        *,
        task: str,
        metadata: JsonDict | None = None,
        session_id: str | None = None,
    ) -> None:
        self.memory.start_session(
            task=task,
            metadata=metadata,
            session_id=session_id or self.default_session_id,
        )
        for name, payload in self.startup_facts.items():
            self.memory.save_fact(name, payload, source="runtime_preflight")
        if self.rollout_recorder is not None and self.memory.session_id is not None:
            self.rollout_recorder.start_session(
                session_id=self.memory.session_id,
                task=task,
                metadata=metadata,
                provenance=self._rollout_provenance(metadata),
            )
        self.memory.record(
            "runtime_ready",
            {
                "planner": type(self.planner).__name__,
                "pipeline": type(self.pipeline).__name__,
                "interfaces": [interface.descriptor() for interface in self.interfaces.list()],
                "tools": [tool.name for tool in self.tools.list()],
                "skills": [skill.name for skill in self.skills.list()],
                "visual_history": (
                    self.visual_history.descriptor()
                    if self.visual_history is not None
                    else {"enabled": False}
                ),
            },
        )

    def resume_session(self, session_id: str, *, max_events: int | None = None) -> None:
        self.memory.resume_session(session_id, max_events=max_events)
        if self.rollout_recorder is not None:
            self.rollout_recorder.start_session(
                session_id=session_id,
                task=self.memory.task or "(resumed)",
                metadata=self.memory.metadata,
                provenance=self._rollout_provenance(self.memory.metadata),
                resumed=True,
            )
        self.memory.record(
            "runtime_resumed",
            {
                "planner": type(self.planner).__name__,
                "pipeline": type(self.pipeline).__name__,
                "session_id": session_id,
            },
        )

    def act(
        self,
        observation: EnvObservation,
        *,
        execution_id: str = "",
        cancel_event: threading.Event | None = None,
    ) -> EnvAction:
        with self._act_lock:
            _raise_if_execution_cancelled(cancel_event)
            self.memory.add_observation(observation)
            visual_delta: JsonDict | None = None
            if self.visual_history is not None:
                visual_delta = self.visual_history.observe(observation, memory=self.memory)
            execution_metadata: JsonDict = {
                "execution_id": execution_id,
                "session_id": self.memory.session_id or "",
                "task": self.memory.current_user_request or observation.task,
                "_observation_packet_resolver": self.memory.resolve_observation_packet,
                "_contact_authorization_resolver": (
                    self.memory.resolve_compiled_contact_authorization
                ),
                "_attachment_candidate_resolver": (
                    self.memory.resolve_active_attachment_candidate
                ),
                "_ik_execution_seed_resolver": self.memory.resolve_ik_execution_seed,
                "_ik_trajectory_execution_bundle_resolver": (
                    self.memory.resolve_ik_trajectory_execution_bundle
                ),
                "_controller_capabilities_resolver": self.memory.controller_capabilities,
                "supervision_context": {
                    "memory": self.memory.planning_context(max_events=4),
                },
            }
            if cancel_event is not None:
                execution_metadata["_cancel_event"] = cancel_event
            with self.tools.execution_scope(execution_metadata):
                decision = self.planner.plan(
                    observation,
                    memory=self.memory,
                    tools=self.tools,
                    skills=self.skills,
                )
                if visual_delta is not None:
                    decision.metadata["visual_delta_usage"] = {
                        key: visual_delta.get(key)
                        for key in ("delta_id", "status", "provider", "model", "usage")
                        if visual_delta.get(key) is not None
                    }
                _raise_if_execution_cancelled(cancel_event)
                plan = self.pipeline.compile(
                    decision,
                    observation=observation,
                    tools=self.tools,
                    skills=self.skills,
                    memory=self.memory,
                )
                _raise_if_execution_cancelled(cancel_event)
                command = plan.to_command()
                command.setdefault("metadata", {})["execution_id"] = execution_id
                self.memory.record("pipeline_plan", command)
                if should_record_recovery_feedback(plan.status):
                    self.memory.record(
                        "recovery_feedback",
                        {
                            "source": "action_pipeline",
                            "command": command,
                        },
                    )
                action = plan.to_env_action()
                self.memory.add_action(action)
                return action

    def update_memory(self, event: JsonDict) -> None:
        self.memory.add_external_event(event)

    def _rollout_provenance(self, metadata: JsonDict | None) -> JsonDict:
        return build_rollout_provenance(
            planner=self.planner,
            pipeline=self.pipeline,
            tools=self.tools,
            skills=self.skills,
            metadata=metadata,
        )

    def _bind_memory_tool_handlers(self) -> None:
        handlers = {
            "save_memory": self._save_memory_tool,
            "get_memory": self._get_memory_tool,
            "delete_memory": self._delete_memory_tool,
            "compact_memory": self._compact_memory_tool,
            "enhance_depth": self._enhance_depth_tool,
            "select_sam3_detection": self._select_sam3_detection_tool,
            "reject_sam3_detections": self._reject_sam3_detections_tool,
            "python_exec": PythonExecRuntime().handler,
        }
        for name, handler in handlers.items():
            if not self.tools.can_execute(name):
                self.tools.bind_handler(name, handler)

    def _save_memory_tool(self, context: ToolExecutionContext) -> ToolResult:
        namespace = str(context.parameters.get("namespace", "facts")).strip() or "facts"
        key = str(context.parameters.get("key", "")).strip()
        if not key:
            key = str(context.parameters.get("skill", "")).strip()
        content = context.parameters.get("content")
        if not key:
            return ToolResult(False, content="save_memory requires a key.")
        if content in (None, ""):
            return ToolResult(False, content="save_memory requires non-empty content.")
        payload = content if isinstance(content, dict) else {"content": content}
        if namespace == "artifacts":
            self.memory.save_artifact(key, payload, source=context.name)
        elif namespace == "skill_notes":
            self.memory.save_skill_note(key, payload, source=context.name)
        else:
            self.memory.save_fact(key, payload, source=context.name)
        return ToolResult(
            True,
            content="memory saved",
            details={"namespace": namespace, "key": key},
        )

    def _get_memory_tool(self, context: ToolExecutionContext) -> ToolResult:
        namespace = str(context.parameters.get("namespace", "all")).strip() or "all"
        key = context.parameters.get("key")
        key_str = str(key).strip() if key is not None else None
        return ToolResult(
            True,
            content="memory loaded",
            details=self.memory.get_memory(key_str or None, namespace=namespace),
        )

    def _delete_memory_tool(self, context: ToolExecutionContext) -> ToolResult:
        key = str(context.parameters.get("key", "")).strip()
        if not key:
            return ToolResult(False, content="delete_memory requires a key.")
        namespace = str(context.parameters.get("namespace", "all")).strip() or "all"
        deleted = self.memory.delete_memory(key, namespace=namespace)
        return ToolResult(True, content="memory deleted", details=deleted)

    def _compact_memory_tool(self, context: ToolExecutionContext) -> ToolResult:
        raw_max_events = context.parameters.get("max_events", 8)
        try:
            max_events = int(raw_max_events)
        except (TypeError, ValueError):
            max_events = 8
        summary = self.memory.compact(max_events=max_events)
        return ToolResult(True, content=summary, details={"summary": summary})

    def _enhance_depth_tool(self, context: ToolExecutionContext) -> ToolResult:
        rgb_path = str(context.parameters.get("rgb") or "").strip()
        depth_path = str(context.parameters.get("depth") or "").strip()
        intrinsics = context.parameters.get("intrinsics")
        source_packet_id = str(
            context.parameters.get("source_packet_id") or ""
        ).strip()
        source_frame_id = str(
            context.parameters.get("camera_frame_id")
            or context.parameters.get("camera_id")
            or ""
        ).strip()
        public_config = context.parameters.get("config")

        def finish(tool_result: ToolResult) -> ToolResult:
            outputs = tool_result.details.get("outputs")
            if isinstance(outputs, dict) and source_packet_id:
                outputs["source_packet_id"] = source_packet_id
                outputs["camera_frame_id"] = source_frame_id
                outputs["depth_prior_resolution"] = (
                    "matched"
                    if context.parameters.get("prior_depth")
                    else "absent_sensor_only"
                )
            if source_packet_id:
                context.parameters = {
                    "source_packet_id": source_packet_id,
                    "camera_frame_id": source_frame_id,
                    **({"config": public_config} if isinstance(public_config, dict) else {}),
                }
            return tool_result
        if not rgb_path or not depth_path or not isinstance(intrinsics, dict):
            return finish(
                make_tool_result(
                    context,
                    success=False,
                    content="enhance_depth requires host-resolved RGB-D and intrinsics.",
                    diagnostics=[{"code": "invalid_depth_enhancement_request"}],
                )
            )
        try:
            rgb = _read_rgb_image(rgb_path)
            depth = _read_depth_array(depth_path, scale=_intrinsics_scale(intrinsics))
            prior = _read_depth_prior(context.parameters)
            sensor_confidence_path = str(
                context.parameters.get("sensor_confidence") or ""
            ).strip()
            sensor_confidence = (
                _read_optional_numeric_array(sensor_confidence_path)
                if sensor_confidence_path
                else None
            )
            config = _depth_enhancement_config(context.parameters.get("config"))
            result = enhance_rgbd_depth(
                rgb=rgb,
                sensor_depth_m=depth,
                intrinsics=intrinsics,
                camera_id=str(context.parameters.get("camera_id") or "camera"),
                calibration_profile_id=str(
                    context.parameters.get("calibration_profile_id") or ""
                ),
                prior_prediction=prior,
                sensor_confidence=sensor_confidence,
                registration_status=str(
                    context.parameters.get("registration_status") or ""
                ),
                rgb_timestamp_s=_optional_float_parameter(
                    context.parameters.get("rgb_timestamp_s")
                ),
                depth_timestamp_s=_optional_float_parameter(
                    context.parameters.get("depth_timestamp_s")
                ),
                scene_epoch=_optional_int_parameter(
                    context.parameters.get("scene_epoch")
                ),
                calibration_hash=str(
                    context.parameters.get("calibration_hash") or ""
                ),
                config=config,
            )
            artifacts = materialize_depth_enhancement(
                result,
                sensor_depth_m=depth,
                bundle_id=str(context.parameters.get("bundle_id") or "").strip() or None,
                session_id=artifact_session_id(context.metadata),
                source_rgb_path=rgb_path,
                source_depth_path=depth_path,
                source_sensor_confidence_path=sensor_confidence_path,
            )
        except Exception as exc:  # noqa: BLE001 - user-facing tool result.
            return finish(
                make_tool_result(
                    context,
                    success=False,
                    content=f"enhance_depth failed: {type(exc).__name__}: {exc}",
                    diagnostics=[
                        {
                            "code": "depth_enhancement_failed",
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    ],
                )
            )
        artifact = artifacts.to_dict()
        candidate_intrinsics = dict(intrinsics)
        candidate_intrinsics["scale"] = 1000.0
        return finish(
            make_tool_result(
                context,
                success=True,
                content=(
                    "depth enhancement completed"
                    if result.enabled
                    else f"depth enhancement produced sensor-only outputs: {result.reason}"
                ),
                outputs={
                "enabled": result.enabled,
                "reason": result.reason,
                "camera_id": result.camera_id,
                "calibration_profile_id": result.calibration_profile_id,
                "source_rgb": str(Path(rgb_path).expanduser().resolve()),
                "source_depth": str(Path(depth_path).expanduser().resolve()),
                "source_sensor_confidence": (
                    str(Path(sensor_confidence_path).expanduser().resolve())
                    if sensor_confidence_path
                    else ""
                ),
                "source_rgb_sha256": result.source.get("rgb_sha256"),
                "source_depth_sha256": result.source.get("sensor_depth_sha256"),
                "intrinsics": dict(intrinsics),
                "candidate_intrinsics": candidate_intrinsics,
                "safety_intrinsics": candidate_intrinsics,
                "scene_epoch": result.source.get("scene_epoch"),
                "rgb_timestamp_s": result.source.get("rgb_timestamp_s"),
                "depth_timestamp_s": result.source.get("depth_timestamp_s"),
                "registration_status": result.source.get("registration_status"),
                "calibration_hash": result.source.get("calibration_hash"),
                "alignment": result.alignment,
                "quality": result.quality,
                "report_path": artifacts.report_path,
                "fused_depth_npy": artifacts.fused_depth_npy,
                "fused_depth_png": artifacts.fused_depth_png,
                "candidate_depth_npy": artifacts.fused_depth_npy,
                "candidate_depth_png": artifacts.fused_depth_png,
                "safety_depth_npy": artifacts.safety_depth_npy,
                "safety_depth_png": artifacts.safety_depth_png,
                "depth_scale": 1000.0,
                "depth_units": "millimeters",
                "point_cloud_npz": artifacts.point_cloud_npz,
                "candidate_point_cloud_npz": artifacts.point_cloud_npz,
                "safety_point_cloud_npz": artifacts.safety_point_cloud_npz,
                "provenance_mask_png": artifacts.provenance_mask_png,
                },
                artifacts=[artifact],
                diagnostics=result.diagnostics,
                semantic_outcome=(
                    "depth_enhanced"
                    if result.enabled
                    else "requires_depth_alignment_repair"
                ),
                recovery_options=(
                    []
                    if result.enabled
                    else _depth_enhancement_recovery_options(result.reason)
                ),
            )
        )

    def _select_sam3_detection_tool(self, context: ToolExecutionContext) -> ToolResult:
        result_id = str(context.parameters.get("sam3_result_id") or "").strip()
        detection_id = str(context.parameters.get("detection_id") or "").strip()
        if not result_id or not detection_id:
            return make_tool_result(
                context,
                success=False,
                content=("select_sam3_detection requires sam3_result_id and detection_id."),
                diagnostics=[{"code": "invalid_detection_selection"}],
            )
        raw_confidence = context.parameters.get("selection_confidence")
        confidence: float | None = None
        if raw_confidence is not None:
            try:
                confidence = float(raw_confidence)
            except (TypeError, ValueError):
                return make_tool_result(
                    context,
                    success=False,
                    content="selection_confidence must be a finite number between 0 and 1.",
                    diagnostics=[{"code": "invalid_selection_confidence"}],
                )
            if not 0.0 <= confidence <= 1.0:
                return make_tool_result(
                    context,
                    success=False,
                    content="selection_confidence must be between 0 and 1.",
                    diagnostics=[{"code": "invalid_selection_confidence"}],
                )
        try:
            selected = self.memory.resolve_sam3_selection(
                result_id=result_id,
                detection_id=detection_id,
                selection_source="main_agent_vlm",
                evidence_role=str(context.parameters.get("evidence_role") or ""),
                confidence=confidence,
                reason=str(context.parameters.get("reason") or ""),
                target_geometry_family=str(
                    context.parameters.get("target_geometry_family") or ""
                ),
                identity_anchor_id=str(
                    context.parameters.get("identity_anchor_id") or ""
                ),
                identity_relation=str(
                    context.parameters.get("identity_relation") or ""
                ),
            )
        except ValueError as exc:
            message = str(exc)
            diagnostic_code = message.split(":", 1)[0] or "invalid_detection_selection"
            outputs: JsonDict = {}
            recovery_options: list[JsonDict] = []
            if diagnostic_code.startswith("target_identity_"):
                anchor = self.memory.target_identity_anchor() or {}
                pending = self.memory.pending_sam3_selection() or {}
                outputs = {
                    "active_identity_anchor_id": anchor.get("anchor_id"),
                    "pending_sam3_result_id": pending.get("result_id"),
                    "pending_detection_id": detection_id,
                    "identity_relation_choices": [
                        "same_instance",
                        "replace_misidentified_anchor",
                    ],
                }
                recovery_options = [
                    {
                        "action": "retry_target_selection_with_identity_relation",
                        "tool": "select_sam3_detection",
                        "copy_parameters": {
                            "sam3_result_id": result_id,
                            "detection_id": detection_id,
                            "identity_anchor_id": anchor.get("anchor_id"),
                        },
                        "required_choice": "identity_relation",
                        "reason": (
                            "compare the old and new visual evidence, then declare "
                            "same_instance or an explicit misidentification correction"
                        ),
                    },
                    {
                        "action": "reject_pending_detections",
                        "tool": "reject_sam3_detections",
                        "parameters": {"sam3_result_id": result_id},
                        "reason": "use when none of the pending masks is the intended target",
                    },
                ]
            return make_tool_result(
                context,
                success=False,
                content=message,
                outputs=outputs,
                diagnostics=[{"code": diagnostic_code}],
                semantic_outcome=diagnostic_code,
                recovery_options=recovery_options,
            )
        artifacts = []
        mask_ref = selected.get("mask_ref")
        if isinstance(mask_ref, str) and mask_ref:
            artifacts.append(
                {
                    "type": "selected_segmentation_mask",
                    "kind": "mask",
                    "tool": context.name,
                    "index": detection_id,
                    "path": mask_ref,
                    "mask_ref": mask_ref,
                }
            )
        source_observation = selected.get("source_observation")
        source_observation = (
            source_observation if isinstance(source_observation, dict) else {}
        )
        camera_frame_id = str(
            source_observation.get("frame_id")
            or selected.get("frame_id")
            or ""
        )
        camera_role = str(
            source_observation.get("role")
            or selected.get("camera_role")
            or ""
        )
        is_wrist_target = (
            str(selected.get("evidence_role") or "") == "target_object"
            and (
                "wrist" in camera_frame_id.lower()
                or "eye_in_hand" in camera_frame_id.lower()
                or "wrist" in camera_role.lower()
            )
        )
        downstream_handoff: JsonDict = {}
        content_suffix = ""
        if is_wrist_target:
            downstream_handoff = {
                "schema_version": "openeta.perception_consumer_handoff.v1",
                "status": "materializes_in_next_planner_context",
                "producer": "select_sam3_detection",
                "camera_frame_id": camera_frame_id,
                "source_packet_id": source_observation.get("packet_id")
                or selected.get("source_packet_id"),
                "inspect": "host_resolved_inputs.wrist_alignment",
                "primary_consumer": "compute_wrist_alignment",
                "fallback_consumer": "grasp_pose_estimate",
                "instruction": (
                    "On the next planner turn, inspect the wrist-alignment input. "
                    "If status=ready, copy only its exact bundle_id into "
                    "compute_wrist_alignment. If orientation or axial contact depth "
                    "is uncertain, consume the wrist grasp_pose_estimate bundle instead."
                ),
                "agent_discretion": True,
            }
            content_suffix = (
                " This wrist target selection materializes its downstream input on "
                "the next planner turn: inspect host_resolved_inputs.wrist_alignment; "
                "when ready, copy its bundle_id into compute_wrist_alignment, or use "
                "the wrist grasp_pose_estimate bundle when orientation/depth is "
                "uncertain."
            )
        return make_tool_result(
            context,
            success=True,
            content=(
                f"Selected {detection_id} from SAM3 result {result_id}."
                f"{content_suffix}"
            ),
            outputs={
                "result_id": result_id,
                "selected_detection": selected,
                "mask_ref": mask_ref,
                "selection_source": selected.get("selection_source"),
                "evidence_role": selected.get("evidence_role"),
                "target_geometry_family": selected.get("target_geometry_family"),
                **(
                    {"downstream_consumer_handoff": downstream_handoff}
                    if downstream_handoff
                    else {}
                ),
            },
            artifacts=artifacts,
        )

    def _reject_sam3_detections_tool(self, context: ToolExecutionContext) -> ToolResult:
        result_id = str(context.parameters.get("sam3_result_id") or "").strip()
        reason = str(context.parameters.get("reason") or "").strip()
        try:
            rejected = self.memory.reject_sam3_detections(
                result_id=result_id,
                reason=reason,
            )
        except ValueError as exc:
            return make_tool_result(
                context,
                success=False,
                content=str(exc),
                diagnostics=[{"code": "invalid_detection_rejection"}],
            )
        return make_tool_result(
            context,
            success=True,
            content=f"Rejected all detections from SAM3 result {result_id}.",
            outputs={"rejection": rejected},
        )


def _read_rgb_image(path: str) -> np.ndarray:
    resolved = _existing_file(path)
    image = Image.open(resolved).convert("RGB")
    return np.asarray(image)


def _read_depth_array(path: str, *, scale: float) -> np.ndarray:
    resolved = _existing_file(path)
    if resolved.suffix.lower() == ".npy":
        return np.asarray(np.load(resolved), dtype=np.float32)
    array = np.asarray(Image.open(resolved))
    if array.ndim == 3:
        array = array[..., 0]
    if array.dtype.kind in {"u", "i"}:
        return array.astype(np.float32) / float(scale)
    return array.astype(np.float32)


def _read_optional_numeric_array(path: str) -> np.ndarray:
    resolved = _existing_file(path)
    if resolved.suffix.lower() == ".npy":
        return np.asarray(np.load(resolved), dtype=np.float32)
    array = np.asarray(Image.open(resolved))
    if array.ndim == 3:
        array = array[..., 0]
    return array.astype(np.float32)


def _read_depth_prior(parameters: JsonDict) -> DepthPriorPrediction | None:
    prior_depth_path = str(parameters.get("prior_depth") or "").strip()
    if not prior_depth_path:
        return None
    scale = _float_parameter(parameters.get("prior_depth_scale"), default=1.0)
    prior_depth = _read_depth_array(prior_depth_path, scale=scale)
    confidence_path = str(parameters.get("prior_confidence") or "").strip()
    confidence = _read_optional_numeric_array(confidence_path) if confidence_path else None
    return DepthPriorPrediction(
        depth_m=prior_depth,
        confidence=confidence,
        confidence_semantics=str(
            parameters.get("prior_confidence_semantics") or "higher_is_better"
        ),
        metadata={
            "backend": str(parameters.get("prior_backend") or "artifact"),
            "model": str(parameters.get("prior_model") or "external_depth_prior"),
            "prior_depth_path": prior_depth_path,
        },
    )


def _depth_enhancement_config(value: object) -> DepthEnhancementConfig:
    if not isinstance(value, dict):
        return DepthEnhancementConfig()
    allowed = {
        "min_depth_m",
        "max_depth_m",
        "min_alignment_pixels",
        "alignment_trim_fraction",
        "mono_confidence_drop_quantile",
        "sensor_confidence_threshold",
        "allow_mono_fill_low_confidence_sensor",
        "min_alignment_scale",
        "max_alignment_scale",
        "max_fill_ratio",
        "disagreement_threshold_m",
        "max_large_disagreement_ratio",
        "edge_guard_pixels",
        "depth_edge_threshold_m",
        "rgb_edge_threshold",
        "require_registration",
        "max_timestamp_skew_s",
    }
    kwargs: JsonDict = {}
    for key in allowed:
        if key in value:
            kwargs[key] = value[key]
    return DepthEnhancementConfig(**kwargs)


def _depth_enhancement_recovery_options(reason: str) -> list[JsonDict]:
    """Translate a safe sensor-only fallback into concrete Agent choices."""

    if reason == "no_depth_prior":
        return [
            {
                "action": "estimate_matching_depth_prior_then_retry",
                "reason": (
                    "Call estimate_depth_prior for the same source_packet_id and "
                    "camera_frame_id, then call enhance_depth again."
                ),
            },
            {
                "action": "continue_with_sensor_depth_only",
                "reason": "The materialized safety depth remains usable as raw sensor evidence.",
            },
        ]
    if reason in {"rgb_depth_not_registered", "rgb_depth_timestamp_skew"}:
        return [
            {
                "action": "observe_fresh_aligned_rgbd_then_retry",
                "reason": (
                    "Use one fresh packet whose RGB and depth share the same camera "
                    "registration and timestamp envelope."
                ),
            },
            {
                "action": "continue_with_sensor_depth_only",
                "reason": "Do not use the monocular candidate as metric geometry.",
            },
        ]
    if reason == "alignment_scale_out_of_bounds":
        return [
            {
                "action": "inspect_depth_prior_units_and_source_match",
                "reason": (
                    "The prior and sensor overlap, but their fitted metric scale is "
                    "outside the accepted range; verify prior units/model output and "
                    "that both artifacts came from the same packet and camera."
                ),
            },
            {
                "action": "continue_with_sensor_depth_only",
                "reason": "The candidate prior was rejected; use only safety sensor depth.",
            },
        ]
    if reason == "insufficient_alignment_pixels":
        return [
            {
                "action": "acquire_view_with_more_valid_depth_overlap",
                "reason": (
                    "A fresh view with more mutually valid sensor/prior pixels may make "
                    "metric alignment identifiable."
                ),
            },
            {
                "action": "continue_with_sensor_depth_only",
                "reason": "Do not infer metric depth from an underconstrained alignment.",
            },
        ]
    return [
        {
            "action": "continue_with_sensor_depth_only",
            "reason": (
                "Depth enhancement produced no trustworthy mono-filled geometry; "
                "inspect diagnostics before deciding whether a fresh view is useful."
            ),
        }
    ]


def _intrinsics_scale(intrinsics: Mapping[str, Any]) -> float:
    return _float_parameter(intrinsics.get("scale"), default=1000.0)


def _float_parameter(value: object, *, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(parsed) or parsed <= 0:
        return default
    return parsed


def _optional_float_parameter(value: object) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if np.isfinite(parsed) else None


def _optional_int_parameter(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _existing_file(path: str) -> Path:
    if not path.strip():
        raise ValueError("path must be non-empty")
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(path)
    return resolved
