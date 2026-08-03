"""OpenETA lightweight agent runtime."""

from __future__ import annotations

import threading

from adapter.protocol import EnvAction, EnvObservation, JsonDict
from agent.runtime.checkers import should_record_recovery_feedback
from agent.runtime.image_artifacts import (
    DEFAULT_MCP_IMAGE_OUTPUT_ROOT,
    materialize_mcp_images,
)
from agent.runtime.interfaces import ActionInterfaceRegistry, build_default_action_interfaces
from agent.runtime.memory import AgentMemory, MemoryStore
from agent.runtime.pipeline import ActionPipeline
from agent.runtime.planner import BasePlanner, ToolCallingPlanner
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
    ) -> None:
        self.planner = planner or ToolCallingPlanner()
        self.memory = memory or AgentMemory(store=memory_store)
        self.tools = tools or build_default_tool_registry()
        self.skills = skills or build_default_skill_registry()
        self.interfaces = interfaces or build_default_action_interfaces()
        self.pipeline = pipeline or ActionPipeline(interfaces=self.interfaces)
        self.self_improvement_reviewer = (
            self_improvement_reviewer or SelfImprovementReviewer()
        )
        self._act_lock = threading.Lock()
        self._bind_memory_tool_handlers()

    def start_session(
        self,
        *,
        task: str,
        metadata: JsonDict | None = None,
        session_id: str | None = None,
    ) -> None:
        self.memory.start_session(task=task, metadata=metadata, session_id=session_id)
        self.memory.record(
            "runtime_ready",
            {
                "planner": type(self.planner).__name__,
                "pipeline": type(self.pipeline).__name__,
                "interfaces": [
                    interface.descriptor() for interface in self.interfaces.list()
                ],
                "tools": [tool.name for tool in self.tools.list()],
                "skills": [skill.name for skill in self.skills.list()],
            },
        )

    def resume_session(self, session_id: str, *, max_events: int | None = 64) -> None:
        self.memory.resume_session(session_id, max_events=max_events)
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
        execution_metadata: JsonDict = {"execution_id": execution_id}
        if cancel_event is not None:
            execution_metadata["_cancel_event"] = cancel_event
        with self._act_lock, self.tools.execution_scope(execution_metadata):
            _raise_if_execution_cancelled(cancel_event)
            self.memory.add_observation(observation)
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

    def _bind_memory_tool_handlers(self) -> None:
        handlers = {
            "save_memory": self._save_memory_tool,
            "get_memory": self._get_memory_tool,
            "delete_memory": self._delete_memory_tool,
            "compact_memory": self._compact_memory_tool,
            "materialize_mcp_images": self._materialize_mcp_images_tool,
            "select_sam3_detection": self._select_sam3_detection_tool,
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

    def _select_sam3_detection_tool(self, context: ToolExecutionContext) -> ToolResult:
        result_id = str(context.parameters.get("sam3_result_id") or "").strip()
        detection_id = str(context.parameters.get("detection_id") or "").strip()
        if not result_id or not detection_id:
            return make_tool_result(
                context,
                success=False,
                content=(
                    "select_sam3_detection requires sam3_result_id and detection_id."
                ),
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
            },
            artifacts=artifacts,
        )
