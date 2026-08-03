"""Tool registry for embodied agent capabilities."""

from __future__ import annotations

import queue
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from typing import Callable

from adapter.protocol import EnvObservation, JsonDict

TOOL_RESULT_SCHEMA_VERSION = "openeta.tool_result.v1"


class ToolEffect(str, Enum):
    """Side-effect class used to enforce closed-loop tool execution."""

    READ_ONLY = "read_only"
    BOOKKEEPING = "bookkeeping"
    PLANNING = "planning"
    WORLD_MUTATING = "world_mutating"


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Declarative description of an agent-visible atomic tool."""

    name: str
    description: str
    category: str
    parameters: JsonDict = field(default_factory=dict)
    safe_by_default: bool = False
    effect: ToolEffect | str = ToolEffect.READ_ONLY
    batchable: bool | None = None

    def __post_init__(self) -> None:
        if isinstance(self.effect, str):
            object.__setattr__(self, "effect", ToolEffect(self.effect))

    @property
    def allows_batched_observation(self) -> bool:
        """Whether this tool may be grouped before the next observation."""

        if self.batchable is not None:
            return self.batchable
        return self.effect in {
            ToolEffect.READ_ONLY,
            ToolEffect.BOOKKEEPING,
            ToolEffect.PLANNING,
        }

    @property
    def requires_observation_after_call(self) -> bool:
        """Whether the next planner turn must observe before another actuator call."""

        return self.effect == ToolEffect.WORLD_MUTATING


@dataclass(slots=True)
class ToolResult:
    """Structured result returned by a tool handler."""

    success: bool
    content: str = ""
    details: JsonDict = field(default_factory=dict)


@dataclass(slots=True)
class ToolExecutionContext:
    """Runtime context passed to a registered tool handler."""

    name: str
    spec: ToolSpec
    parameters: JsonDict = field(default_factory=dict)
    observation: EnvObservation | None = None
    metadata: JsonDict = field(default_factory=dict)


ToolHandler = Callable[[ToolExecutionContext], ToolResult | JsonDict | str | None]
ToolEventListener = Callable[[JsonDict], None]


def tool_result_type(spec: ToolSpec) -> str:
    """Return the standard result family for a tool spec."""

    if spec.effect == ToolEffect.WORLD_MUTATING:
        return "world_mutating"
    if spec.effect == ToolEffect.BOOKKEEPING or spec.category == "memory":
        return "bookkeeping"
    if spec.category == "perception":
        return "perception"
    if spec.category == "safety":
        return "safety"
    return "planning"


def make_tool_result_details(
    spec: ToolSpec,
    parameters: JsonDict | None = None,
    *,
    success: bool,
    outputs: JsonDict | None = None,
    artifacts: list[JsonDict] | None = None,
    state_delta: JsonDict | None = None,
    diagnostics: list[JsonDict] | None = None,
) -> JsonDict:
    """Build the standard `ToolResult.details` envelope.

    Category-specific payloads live in `outputs`; durable references such as
    mask ids, trajectories, or saved files can be mirrored in `artifacts`.
    World-mutating tools should report simulator/robot changes through
    `state_delta`.
    """

    return {
        "schema_version": TOOL_RESULT_SCHEMA_VERSION,
        "tool": spec.name,
        "category": spec.category,
        "effect": spec.effect.value,
        "result_type": tool_result_type(spec),
        "success": success,
        "parameters": dict(parameters or {}),
        "outputs": dict(outputs or {}),
        "artifacts": list(artifacts or []),
        "state_delta": dict(state_delta or {}),
        "diagnostics": list(diagnostics or []),
        "requires_observation_after_call": spec.requires_observation_after_call,
    }


def make_tool_result(
    context: ToolExecutionContext,
    *,
    success: bool,
    content: str = "",
    outputs: JsonDict | None = None,
    artifacts: list[JsonDict] | None = None,
    state_delta: JsonDict | None = None,
    diagnostics: list[JsonDict] | None = None,
) -> ToolResult:
    """Create a `ToolResult` that already follows the standard envelope."""

    return ToolResult(
        success=success,
        content=content,
        details=make_tool_result_details(
            context.spec,
            context.parameters,
            success=success,
            outputs=outputs,
            artifacts=artifacts,
            state_delta=state_delta,
            diagnostics=diagnostics,
        ),
    )


class ToolRegistry:
    """Registry of stable perception, planning, control, and memory tools."""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._handlers: dict[str, ToolHandler] = {}
        self._listeners: list[ToolEventListener] = []
        self._execution_local = threading.local()

    @contextmanager
    def execution_scope(self, metadata: JsonDict | None = None):
        """Attach per-thread execution ownership and cancellation metadata."""

        previous = getattr(self._execution_local, "metadata", None)
        self._execution_local.metadata = dict(metadata or {})
        try:
            yield
        finally:
            self._execution_local.metadata = previous

    def register(self, spec: ToolSpec, handler: ToolHandler | None = None) -> None:
        if spec.name in self._specs:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._specs[spec.name] = spec
        if handler is not None:
            self.bind_handler(spec.name, handler)

    def bind_handler(
        self,
        name: str,
        handler: ToolHandler,
        *,
        replace: bool = False,
    ) -> None:
        """Attach an executable handler to an existing tool spec."""

        if name not in self._specs:
            raise KeyError(f"Unknown tool: {name}")
        if not replace and name in self._handlers:
            raise ValueError(f"Tool handler already registered: {name}")
        self._handlers[name] = handler

    def unbind_handler(self, name: str) -> None:
        """Remove a handler while keeping the tool spec visible to planners."""

        if name not in self._specs:
            raise KeyError(f"Unknown tool: {name}")
        self._handlers.pop(name, None)

    def get(self, name: str) -> ToolSpec:
        try:
            return self._specs[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool: {name}") from exc

    def list(self, *, category: str | None = None) -> list[ToolSpec]:
        specs = list(self._specs.values())
        if category is not None:
            specs = [spec for spec in specs if spec.category == category]
        return specs

    def can_execute(self, name: str) -> bool:
        return name in self._handlers

    def add_listener(self, listener: ToolEventListener) -> None:
        """Register a best-effort callback for tool execution events."""

        self._listeners.append(listener)

    def call(
        self,
        name: str,
        parameters: JsonDict | None = None,
        *,
        observation: EnvObservation | None = None,
        metadata: JsonDict | None = None,
    ) -> ToolResult:
        parameters = dict(parameters or {})
        scope_metadata = getattr(self._execution_local, "metadata", None)
        combined_metadata = {
            **(dict(scope_metadata) if isinstance(scope_metadata, dict) else {}),
            **dict(metadata or {}),
        }
        requested_name = name
        if _execution_cancelled(combined_metadata):
            return _cancelled_tool_result(requested_name, parameters)
        if name not in self._specs:
            self._emit_tool_event(
                {
                    "phase": "start",
                    "name": requested_name,
                    "parameters": parameters,
                    "metadata": _public_execution_metadata(combined_metadata),
                }
            )
            result = _tool_error_result(
                requested_name,
                parameters,
                content=f"Unknown tool: {requested_name}",
                diagnostics=[{"code": "unknown_tool"}],
            )
            self._emit_tool_result(
                requested_name,
                parameters,
                result,
                metadata=_public_execution_metadata(combined_metadata),
            )
            return result
        spec = self._specs[name]
        self._emit_tool_event(
            {
                "phase": "start",
                "name": requested_name,
                "category": spec.category,
                "effect": spec.effect.value,
                "parameters": parameters,
                "metadata": _public_execution_metadata(combined_metadata),
            }
        )
        handler = self._handlers.get(name)
        if handler is None:
            result = ToolResult(
                False,
                content=f"Tool is registered but has no handler: {requested_name}",
                details=make_tool_result_details(
                    spec,
                    parameters,
                    success=False,
                    diagnostics=[{"code": "missing_handler"}],
                ),
            )
            self._emit_tool_result(
                requested_name,
                parameters,
                result,
                spec=spec,
                metadata=_public_execution_metadata(combined_metadata),
            )
            return result
        context = ToolExecutionContext(
            name=name,
            spec=spec,
            parameters=parameters,
            observation=observation,
            metadata=combined_metadata,
        )
        try:
            result = _coerce_tool_result(
                _invoke_tool_handler(handler, context, combined_metadata),
                tool=name,
            )
            normalized = _normalize_tool_result(result, spec=spec, parameters=context.parameters)
            if _execution_cancelled(combined_metadata):
                cancelled = _cancelled_tool_result(
                    requested_name,
                    parameters,
                    spec=spec,
                    abandoned=True,
                )
                self._emit_tool_result(
                    requested_name,
                    parameters,
                    cancelled,
                    spec=spec,
                    metadata=_public_execution_metadata(combined_metadata),
                )
                return cancelled
            self._emit_tool_result(
                requested_name,
                parameters,
                normalized,
                spec=spec,
                metadata=_public_execution_metadata(combined_metadata),
            )
            return normalized
        except _ToolHandlerAbandoned:
            result = _cancelled_tool_result(
                requested_name,
                parameters,
                spec=spec,
                abandoned=True,
            )
            self._emit_tool_result(
                requested_name,
                parameters,
                result,
                spec=spec,
                metadata=_public_execution_metadata(combined_metadata),
            )
            return result
        except Exception as exc:  # noqa: BLE001 - tool failures must stay structured.
            result = ToolResult(
                False,
                content=f"Tool handler failed: {requested_name}: {exc}",
                details=make_tool_result_details(
                    spec,
                    context.parameters,
                    success=False,
                    diagnostics=[
                        {
                            "code": "handler_exception",
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    ],
                ),
            )
            if _execution_cancelled(combined_metadata):
                return _cancelled_tool_result(
                    requested_name,
                    parameters,
                    spec=spec,
                    abandoned=True,
                )
            self._emit_tool_result(
                requested_name,
                parameters,
                result,
                spec=spec,
                metadata=_public_execution_metadata(combined_metadata),
            )
            return result

    def handler_names(self) -> list[str]:
        """Return names of tools that currently have executable handlers."""

        return sorted(self._handlers)

    def _emit_tool_result(
        self,
        name: str,
        parameters: JsonDict,
        result: ToolResult,
        *,
        spec: ToolSpec | None = None,
        metadata: JsonDict | None = None,
    ) -> None:
        event: JsonDict = {
            "phase": "end",
            "name": name,
            "parameters": parameters,
            "success": result.success,
            "content": result.content,
            "details": result.details,
            "metadata": dict(metadata or {}),
        }
        if spec is not None:
            event["category"] = spec.category
            event["effect"] = spec.effect.value
        self._emit_tool_event(event)

    def _emit_tool_event(self, event: JsonDict) -> None:
        for listener in list(self._listeners):
            try:
                listener(dict(event))
            except Exception:
                continue


def _coerce_tool_result(value: ToolResult | JsonDict | str | None, *, tool: str) -> ToolResult:
    if isinstance(value, ToolResult):
        return value
    if value is None:
        return ToolResult(True, content="", details={"tool": tool})
    if isinstance(value, str):
        return ToolResult(True, content=value, details={"tool": tool})
    if isinstance(value, dict):
        success_value: Any = value.get("success", True)
        content_value = value.get("content", "")
        details_value = value.get("details", value)
        return ToolResult(
            success=bool(success_value),
            content=str(content_value),
            details=details_value if isinstance(details_value, dict) else {"value": details_value},
        )
    return ToolResult(
        False,
        content=f"Unsupported tool result type from {tool}: {type(value).__name__}",
        details={"tool": tool},
    )


def _normalize_tool_result(
    result: ToolResult,
    *,
    spec: ToolSpec,
    parameters: JsonDict,
) -> ToolResult:
    details = dict(result.details)
    if details.get("schema_version") == TOOL_RESULT_SCHEMA_VERSION:
        details.setdefault("tool", spec.name)
        details.setdefault("category", spec.category)
        details.setdefault("effect", spec.effect.value)
        details.setdefault("result_type", tool_result_type(spec))
        details.setdefault("success", result.success)
        details.setdefault("parameters", dict(parameters))
        details.setdefault("outputs", {})
        details.setdefault("artifacts", [])
        details.setdefault("state_delta", {})
        details.setdefault("diagnostics", [])
        details.setdefault(
            "requires_observation_after_call",
            spec.requires_observation_after_call,
        )
    else:
        artifacts_value = details.get("artifacts")
        artifacts = artifacts_value if isinstance(artifacts_value, list) else []
        details = make_tool_result_details(
            spec,
            parameters,
            success=result.success,
            outputs=details,
            artifacts=[
                artifact for artifact in artifacts if isinstance(artifact, dict)
            ],
        )
    return ToolResult(
        success=result.success,
        content=result.content,
        details=details,
    )


def _tool_error_result(
    name: str,
    parameters: JsonDict | None,
    *,
    content: str,
    diagnostics: list[JsonDict],
) -> ToolResult:
    return ToolResult(
        False,
        content=content,
        details={
            "schema_version": TOOL_RESULT_SCHEMA_VERSION,
            "tool": name,
            "category": "unknown",
            "effect": "unknown",
            "result_type": "unknown",
            "success": False,
            "parameters": dict(parameters or {}),
            "outputs": {},
            "artifacts": [],
            "state_delta": {},
            "diagnostics": diagnostics,
            "requires_observation_after_call": False,
        },
    )


def _execution_cancelled(metadata: JsonDict) -> bool:
    event = metadata.get("_cancel_event")
    return bool(event is not None and callable(getattr(event, "is_set", None)) and event.is_set())


class _ToolHandlerAbandoned(RuntimeError):
    """Internal signal that a cancelled execution no longer owns a tool result."""


def _invoke_tool_handler(
    handler: ToolHandler,
    context: ToolExecutionContext,
    metadata: JsonDict,
) -> ToolResult | JsonDict | str | None:
    """Run blocking handlers behind a cancellation-aware ownership boundary."""

    if "_cancel_event" not in metadata:
        return handler(context)

    result_queue: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=1)

    def invoke() -> None:
        try:
            result_queue.put(("result", handler(context)))
        except BaseException as exc:  # noqa: BLE001 - forwarded to the caller thread.
            result_queue.put(("error", exc))

    worker = threading.Thread(
        target=invoke,
        name=f"openeta-tool-{context.name}",
        daemon=True,
    )
    worker.start()
    while True:
        if _execution_cancelled(metadata):
            raise _ToolHandlerAbandoned
        try:
            kind, payload = result_queue.get(timeout=0.05)
        except queue.Empty:
            continue
        if kind == "error":
            if isinstance(payload, BaseException):
                raise payload
            raise RuntimeError(str(payload))
        return payload


def _public_execution_metadata(metadata: JsonDict) -> JsonDict:
    return {str(key): value for key, value in metadata.items() if not str(key).startswith("_")}


def _cancelled_tool_result(
    name: str,
    parameters: JsonDict,
    *,
    spec: ToolSpec | None = None,
    abandoned: bool = False,
) -> ToolResult:
    if spec is None:
        return _tool_error_result(
            name,
            parameters,
            content="Tool execution cancelled before dispatch.",
            diagnostics=[{"code": "execution_cancelled", "abandoned": abandoned}],
        )
    return ToolResult(
        False,
        content=(
            "Tool result abandoned because its episode was cancelled."
            if abandoned
            else "Tool execution cancelled before dispatch."
        ),
        details=make_tool_result_details(
            spec,
            parameters,
            success=False,
            diagnostics=[{"code": "execution_cancelled", "abandoned": abandoned}],
        ),
    )


def build_default_tool_registry() -> ToolRegistry:
    """Create the initial OpenETA tool catalog from the architecture notes."""

    registry = ToolRegistry()
    for spec in [
        ToolSpec(
            name="observe",
            category="perception",
            description="Request or retrieve the latest environment observation.",
            parameters={"reason": "why a fresh observation is needed"},
            effect=ToolEffect.READ_ONLY,
        ),
        ToolSpec(
            name="materialize_mcp_images",
            category="artifact",
            description=(
                "Write MCP base64 RGB/depth image payloads to local files and "
                "return lightweight image references."
            ),
            parameters={
                "payload": "MCP observation, render, or step payload containing base64 images",
                "output_root": "optional artifact root directory",
                "bundle_id": "optional stable bundle id",
            },
            effect=ToolEffect.BOOKKEEPING,
        ),
        ToolSpec(
            name="create_simulator_env",
            category="environment",
            description=(
                "Create exactly one remote simulator environment and reset it to obtain "
                "the initial observation. This is the only agent-facing environment "
                "creation path; do not call simulator create_env through python_exec."
            ),
            parameters={
                "env_id": "required OpenETA simulator environment id",
                "seed": "optional deterministic reset seed; defaults to 0",
                "task": "optional task text forwarded to the simulator",
                "render_mode": "optional render mode; defaults to rgb_array",
                "image_width": "optional camera width; defaults to 512",
                "image_height": "optional camera height; defaults to 512",
                "session_id": "optional session id for dashboard and trace correlation",
                "include_objects": "optional object-metadata toggle",
            },
            effect=ToolEffect.WORLD_MUTATING,
            batchable=False,
        ),
        ToolSpec(
            name="close_simulator_env",
            category="environment",
            description=(
                "Close the currently active remote simulator environment and clear "
                "its bound handle. This is the only agent-facing environment cleanup "
                "path; do not call simulator close_env through python_exec."
            ),
            parameters={},
            effect=ToolEffect.WORLD_MUTATING,
            batchable=False,
        ),
        ToolSpec(
            name="python_exec",
            category="coding",
            description=(
                "Execute a small restricted Python snippet with OpenETA helper APIs. "
                "Use this for one-off API/MCP orchestration that does not deserve a "
                "dedicated agent tool."
            ),
            parameters={
                "code": "Python code. Set a JSON-serializable `result` variable.",
                "sandbox": (
                    "sandbox | outside_sandbox. outside_sandbox requires per-call user "
                    "approval and runs in a disposable host subprocess"
                ),
                "timeout_s": "optional execution timeout; outside_sandbox is capped at 600s",
            },
            effect=ToolEffect.WORLD_MUTATING,
            batchable=False,
        ),
        ToolSpec(
            name="scene_detector",
            category="perception",
            description=(
                "List candidate object names and coarse scene entities. Current "
                "default handler is a dummy placeholder unless a real detector "
                "backend is bound."
            ),
            parameters={
                "image": "camera frame id or local RGB image path",
                "query": "optional user target phrase, e.g. milk box",
            },
            effect=ToolEffect.READ_ONLY,
        ),
        ToolSpec(
            name="sam3",
            category="perception",
            description=(
                "Segment objects or regions from RGB observations, rank detections "
                "by score, and provide candidate visuals for explicit VLM selection."
            ),
            parameters={
                "image": "camera frame id or local RGB image path",
                "prompt": (
                    "concise visual object phrase, preferably English; translate "
                    "non-English user targets such as 罐子 -> can before calling SAM3"
                ),
            },
            effect=ToolEffect.READ_ONLY,
        ),
        ToolSpec(
            name="select_sam3_detection",
            category="perception",
            description=(
                "Resolve a pending SAM3 semantic-verification obligation by selecting "
                "one stable detection id after visually inspecting the original image "
                "and supplied mask overlays, including single-detection results."
            ),
            parameters={
                "sam3_result_id": "exact result_id from the pending SAM3 selection",
                "detection_id": "stable candidate id such as detection_001",
                "selection_confidence": "optional VLM confidence in the semantic selection",
                "reason": "short visual or task-semantic justification",
            },
            safe_by_default=True,
            effect=ToolEffect.PLANNING,
            batchable=False,
        ),
        ToolSpec(
            name="anygrasp",
            category="manipulation",
            description=(
                "Generate score-descending parallel-jaw grasp candidates from RGBD "
                "observations. Rank 0 is the greedy active candidate; linked safety "
                "or motion rejection activates the next ranked candidate."
            ),
            parameters={
                "mode": "targeted or scene; defaults to targeted",
                "rgb": "local RGB image file path",
                "depth": "local depth image file path",
                "intrinsics": (
                    "pinhole camera intrinsics with fx, fy, cx, cy, scale; copy "
                    "from the same observe/render camera_packet.anygrasp_intrinsics "
                    "as rgb/depth in the same observe/render camera metadata"
                ),
                "target_mask": (
                    "local binary target mask path; required for targeted mode; "
                    "use sam3 details.outputs.selected_detection.mask_ref for a "
                    "single detection, or the explicitly disambiguated "
                    "details.outputs.detections[i].mask_ref for multiple detections"
                ),
                "approach_steering": "optional camera-frame 3D approach direction",
                "approach_thresh": "optional approach direction threshold in radians",
                "collision_detection": "optional bool; defaults to true",
                "dense_grasp": "optional bool; defaults to false",
            },
            effect=ToolEffect.PLANNING,
        ),
        ToolSpec(
            name="contact_graspnet",
            category="manipulation",
            description=(
                "Generate targeted Panda-compatible contact grasp candidates from "
                "aligned depth and a SAM3 object mask. This tool is independent "
                "from AnyGrasp and does not call it as a fallback."
            ),
            parameters={
                "rgb": (
                    "local RGB image path from the same observation; used only for "
                    "mask provenance and never sent to the inference MCP"
                ),
                "depth": "aligned local uint16 raw-depth PNG path",
                "object_mask": (
                    "complete SAM3 segmentation artifact containing mask_ref and "
                    "source_image; bare mask paths are not accepted"
                ),
                "intrinsics": (
                    "pinhole camera intrinsics with finite fx, fy, cx, cy, and "
                    "positive scale; copy the same observation camera_packet.intrinsics"
                ),
            },
            safe_by_default=False,
            effect=ToolEffect.PLANNING,
        ),
        ToolSpec(
            name="anyplace",
            category="manipulation",
            description=(
                "Predict five camera-frame object placement transforms and the "
                "corresponding placed grasp poses from one RGBD observation."
            ),
            parameters={
                "rgb": "local RGB image file path from the selected grasp observation",
                "depth": "aligned local depth image file path from the same observation",
                "object_mask": "local object mask path used by the selected AnyGrasp call",
                "placement_region_mask": (
                    "SAM3 segmentation artifact containing mask_ref and source_image; "
                    "additional SAM3 detection metadata is allowed"
                ),
                "intrinsics": "pinhole intrinsics with finite fx, fy, cx, cy, and scale",
                "selected_grasp": (
                    "object with candidate (one complete AnyGrasp candidate) and source "
                    "(the successful targeted AnyGrasp details.source object)"
                ),
            },
            safe_by_default=False,
            effect=ToolEffect.PLANNING,
        ),
        ToolSpec(
            name="camera_pose_to_world",
            category="geometry",
            description=(
                "Transform a camera-frame pose or grasp candidate into the "
                "world frame using simulator MCP camera calibration."
            ),
            parameters={
                "camera_pose": (
                    "camera-frame pose/candidate with frame='camera', "
                    "camera_frame='opencv' by default, translation_xyz, optional "
                    "rotation_matrix, and optional gripper_tip_position_xyz"
                ),
                "camera_to_world": (
                    "preferred simulator MCP camera-to-world transform. "
                    "Supports row-major 4x4 matrix mappings with "
                    "camera_to_world/pose_mat. Defaults to OpenCV camera frame "
                    "unless camera_frame/camera_to_world_frame says otherwise"
                ),
                "camera_extrinsics": (
                    "legacy/simulator alias for camera_to_world. For MuJoCo "
                    "MetaWorld/LIBERO, {pos, mat} uses camera->world, flattened "
                    "row-major mat, OpenGL camera frame (+X right, +Y up, "
                    "camera looks -Z)"
                ),
                "camera_frame_id": "optional camera frame id for traceability",
                "input_camera_frame": "optional pose camera frame; defaults to opencv",
                "camera_to_world_frame": (
                    "optional matrix camera frame. Defaults to opengl for "
                    "simulator {pos, mat}, opencv for 4x4 matrices"
                ),
                "matrix_convention": (
                    "optional matrix direction convention; defaults to "
                    "camera_to_world_row_major"
                ),
            },
            effect=ToolEffect.READ_ONLY,
        ),
        ToolSpec(
            name="anydexgrasp",
            category="manipulation",
            description="Generate dexterous-hand grasp candidates.",
            parameters={"rgbd": "camera frame id or RGBD payload", "target": "object prompt"},
            effect=ToolEffect.PLANNING,
        ),
        ToolSpec(
            name="slam",
            category="navigation",
            description="Maintain or query a spatial map for navigation.",
            parameters={"query": "map query or update request"},
            effect=ToolEffect.PLANNING,
        ),
        ToolSpec(
            name="move_to",
            category="control",
            description=(
                "Move the end effector to one world-frame target pose through the "
                "controller."
            ),
            parameters={
                "target_pose": (
                    "desired world-frame end-effector pose with xyz and optional "
                    "rotation_matrix, euler_xyz_deg, or roll/pitch/yaw"
                ),
                "num_steps": "optional controller step limit",
                "tolerance": "optional position tolerance in metres",
                "ori_tolerance": "optional orientation tolerance in radians",
                "enable_collision_check": "optional simulator collision-check toggle",
            },
            effect=ToolEffect.WORLD_MUTATING,
            batchable=False,
        ),
        ToolSpec(
            name="follow_eef_trajectory",
            category="control",
            description="Follow a short end-effector trajectory through the controller.",
            parameters={"trajectory": "ordered end-effector poses"},
            effect=ToolEffect.WORLD_MUTATING,
            batchable=False,
        ),
        ToolSpec(
            name="gripper_control",
            category="control",
            description="Set gripper open/close ratio in the normalized range 0..1.",
            parameters={"position": "0 closed, 1 open"},
            effect=ToolEffect.WORLD_MUTATING,
            batchable=False,
        ),
        ToolSpec(
            name="lower_body_control_policy",
            category="control",
            description="Execute or preview lower-body navigation/control commands.",
            parameters={"command": "navigation or locomotion command"},
            effect=ToolEffect.WORLD_MUTATING,
            batchable=False,
        ),
        ToolSpec(
            name="hand_pose_database",
            category="manipulation",
            description="Retrieve reference hand poses for dexterous manipulation.",
            parameters={"object": "object name", "task": "manipulation intent"},
            effect=ToolEffect.READ_ONLY,
        ),
        ToolSpec(
            name="ik_preview_check",
            category="safety",
            description="Preview inverse-kinematics feasibility before execution.",
            parameters={
                "target_pose": (
                    "desired end-effector pose; preserve the AnyGrasp candidate id "
                    "from camera_pose_to_world when checking a grasp pose"
                )
            },
            safe_by_default=True,
            effect=ToolEffect.READ_ONLY,
        ),
        ToolSpec(
            name="obstacle_avoidance",
            category="safety",
            description="Check or plan collision-aware motion around obstacles.",
            parameters={"path": "candidate path or motion plan"},
            safe_by_default=True,
            effect=ToolEffect.PLANNING,
        ),
        ToolSpec(
            name="save_memory",
            category="memory",
            description="Save a concise working-memory note for later planner turns.",
            parameters={
                "namespace": "facts | artifacts | skill_notes",
                "key": "memory key or skill name",
                "content": "memory payload",
                "tags": "optional labels",
            },
            effect=ToolEffect.BOOKKEEPING,
        ),
        ToolSpec(
            name="get_memory",
            category="memory",
            description="Read working-memory facts, artifacts, skill notes, or compact summary.",
            parameters={"namespace": "all | facts | artifacts | skill_notes", "key": "optional"},
            effect=ToolEffect.READ_ONLY,
        ),
        ToolSpec(
            name="delete_memory",
            category="memory",
            description="Delete a working-memory fact, artifact, or skill note entry by key.",
            parameters={"namespace": "all | facts | artifacts | skill_notes", "key": "memory key"},
            effect=ToolEffect.BOOKKEEPING,
        ),
        ToolSpec(
            name="compact_memory",
            category="memory",
            description="Compact recent session events and working memory into a short summary.",
            parameters={"max_events": "number of recent events to summarize"},
            effect=ToolEffect.BOOKKEEPING,
        ),
        ToolSpec(
            name="register_skill",
            category="skill_management",
            description="Register a new text-guidance skill document.",
            parameters={
                "name": "skill name",
                "description": "short description",
                "content": "text guidance",
                "allowed_tools": "optional list of atomic tools",
            },
            effect=ToolEffect.BOOKKEEPING,
        ),
        ToolSpec(
            name="update_skill",
            category="skill_management",
            description="Update an existing editable text-guidance skill document.",
            parameters={
                "name": "skill name",
                "content": "replacement text guidance",
                "description": "optional replacement description",
            },
            effect=ToolEffect.BOOKKEEPING,
        ),
    ]:
        registry.register(spec)
    return registry
