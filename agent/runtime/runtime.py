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
from agent.runtime.image_artifacts import (
    DEFAULT_MCP_IMAGE_OUTPUT_ROOT,
    materialize_mcp_images,
)
from agent.runtime.interfaces import ActionInterfaceRegistry, build_default_action_interfaces
from agent.runtime.memory import AgentMemory, MemoryStore
from agent.runtime.pipeline import ActionPipeline
from agent.runtime.planner import BasePlanner, ToolCallingPlanner
from agent.runtime.rollout import RolloutRecorder, build_rollout_provenance
from agent.runtime.self_improvement import SelfImprovementReviewer
from agent.runtime.skills import SkillRegistry, build_default_skill_registry
from agent.tools.coding import PythonExecRuntime
from agent.tools.registry import (
    ToolExecutionContext,
    ToolRegistry,
    ToolResult,
    build_default_tool_registry,
    make_tool_result,
    make_tool_result_details,
)


class RuntimeExecutionCancelled(RuntimeError):
    """Raised when an episode loses ownership while planner/tool work is in flight."""


def _raise_if_execution_cancelled(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeExecutionCancelled("episode execution was cancelled")


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
    ) -> None:
        self.planner = planner or ToolCallingPlanner()
        self.memory = memory or AgentMemory(store=memory_store)
        self.tools = tools or build_default_tool_registry()
        self.skills = skills or build_default_skill_registry()
        self.interfaces = interfaces or build_default_action_interfaces()
        self.pipeline = pipeline or ActionPipeline(interfaces=self.interfaces)
        self.self_improvement_reviewer = self_improvement_reviewer or SelfImprovementReviewer()
        self.default_session_id = default_session_id
        self.rollout_recorder = rollout_recorder
        if self.rollout_recorder is None and rollout_enabled:
            store_root = getattr(self.memory.store, "root", None)
            if store_root is not None:
                self.rollout_recorder = RolloutRecorder(store_root)
        if isinstance(self.planner, ToolCallingPlanner):
            self.planner.set_rollout_recorder(self.rollout_recorder)
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
            },
        )

    def resume_session(self, session_id: str, *, max_events: int | None = 64) -> None:
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
            execution_metadata: JsonDict = {
                "execution_id": execution_id,
                "session_id": self.memory.session_id or "",
                "task": self.memory.current_user_request or observation.task,
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
            "materialize_mcp_images": self._materialize_mcp_images_tool,
            "enhance_depth": self._enhance_depth_tool,
            "select_sam3_detection": self._select_sam3_detection_tool,
            "reject_sam3_detections": self._reject_sam3_detections_tool,
            "activate_final_grasp_candidate": self._activate_final_grasp_candidate_tool,
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

    def _materialize_mcp_images_tool(self, context: ToolExecutionContext) -> ToolResult:
        payload = context.parameters.get("payload")
        if payload is None:
            payload = context.parameters.get("mcp_payload")
        if payload is None:
            payload = context.parameters.get("observation")
        if not isinstance(payload, dict):
            return make_tool_result(
                context,
                success=False,
                content="materialize_mcp_images requires a dict payload.",
                diagnostics=[{"code": "invalid_payload"}],
            )

        output_root = context.parameters.get("output_root")
        bundle_id = context.parameters.get("bundle_id")
        bundle = materialize_mcp_images(
            payload,
            output_root=str(output_root) if output_root else DEFAULT_MCP_IMAGE_OUTPUT_ROOT,
            bundle_id=str(bundle_id).strip() if bundle_id else None,
            session_id=artifact_session_id(context.metadata),
        )
        bundle_details = bundle.to_dict()
        return ToolResult(
            True,
            content=f"materialized {len(bundle.images)} MCP image(s)",
            details=make_tool_result_details(
                context.spec,
                {
                    "payload": {
                        "base64_omitted": True,
                        "top_level_keys": sorted(str(key) for key in payload),
                    },
                    "output_root": str(output_root)
                    if output_root
                    else str(DEFAULT_MCP_IMAGE_OUTPUT_ROOT),
                    "bundle_id": bundle_details["bundle_id"],
                },
                success=True,
                outputs={
                    "bundle_id": bundle_details["bundle_id"],
                    "artifact_root": bundle_details["artifact_root"],
                    "payload": bundle_details["payload"],
                },
                artifacts=bundle_details["images"],
            ),
        )

    def _enhance_depth_tool(self, context: ToolExecutionContext) -> ToolResult:
        rgb_path = str(context.parameters.get("rgb") or "").strip()
        depth_path = str(context.parameters.get("depth") or "").strip()
        intrinsics = context.parameters.get("intrinsics")
        if not rgb_path or not depth_path or not isinstance(intrinsics, dict):
            return make_tool_result(
                context,
                success=False,
                content="enhance_depth requires rgb, depth, and intrinsics.",
                diagnostics=[{"code": "invalid_depth_enhancement_request"}],
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
            return make_tool_result(
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
        artifact = artifacts.to_dict()
        candidate_intrinsics = dict(intrinsics)
        candidate_intrinsics["scale"] = 1000.0
        return make_tool_result(
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
                confidence=confidence,
                reason=str(context.parameters.get("reason") or ""),
                target_geometry_family=str(
                    context.parameters.get("target_geometry_family") or ""
                ),
            )
        except ValueError as exc:
            return make_tool_result(
                context,
                success=False,
                content=str(exc),
                diagnostics=[{"code": "invalid_detection_selection"}],
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
        return make_tool_result(
            context,
            success=True,
            content=f"Selected {detection_id} from SAM3 result {result_id}.",
            outputs={
                "result_id": result_id,
                "selected_detection": selected,
                "mask_ref": mask_ref,
                "selection_source": selected.get("selection_source"),
                "target_geometry_family": selected.get("target_geometry_family"),
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

    def _activate_final_grasp_candidate_tool(
        self,
        context: ToolExecutionContext,
    ) -> ToolResult:
        recovery_id = str(context.parameters.get("recovery_id") or "").strip()
        try:
            activated = self.memory.activate_final_grasp_candidate(
                recovery_id=recovery_id,
            )
        except ValueError as exc:
            return make_tool_result(
                context,
                success=False,
                content=str(exc),
                diagnostics=[{"code": "invalid_final_grasp_fallback"}],
            )
        return make_tool_result(
            context,
            success=True,
            content="Activated the final highest-scoring refinable grasp candidate.",
            outputs={"activation": activated},
        )


def _read_rgb_image(path: str) -> np.ndarray:
    resolved = _existing_file(path)
    image = Image.open(resolved).convert("RGB")
    return np.asarray(image)


def _read_depth_array(path: str, *, scale: float) -> np.ndarray:
    resolved = _existing_file(path)
    if resolved.suffix.lower() == ".npy":
        array = np.load(resolved)
        return np.asarray(array, dtype=np.float32)
    image = Image.open(resolved)
    array = np.asarray(image)
    if array.ndim == 3:
        array = array[..., 0]
    if array.dtype.kind in {"u", "i"}:
        return array.astype(np.float32) / float(scale)
    return array.astype(np.float32)


def _read_optional_numeric_array(path: str) -> np.ndarray:
    resolved = _existing_file(path)
    if resolved.suffix.lower() == ".npy":
        return np.asarray(np.load(resolved), dtype=np.float32)
    image = Image.open(resolved)
    array = np.asarray(image)
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
