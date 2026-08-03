#!/usr/bin/env python3
"""Clean Operator-facing MCP gateway for the LIBERO embodied harness.

The gateway intentionally exposes semantic inputs and native MCP images.  It
does not expose the OpenETA planner/runtime, raw RGB-D base64, or simulator
handles. Explicit pose refinement is exposed as a deliberate Operator action.
"""

from __future__ import annotations

import argparse
import atexit
import functools
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Annotated, Any, Literal, Mapping

import numpy as np

import anyio.to_thread
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.utilities.types import Image
from mcp.types import TextContent, ToolAnnotations
from pydantic import BaseModel, ConfigDict, Field

from tools.embodied_gateway import EmbodiedGateway, GatewayResult, _rotation_to_euler_degrees
from tools.operator_context_profiles import (
    active_profile,
    finalize_contract,
    project_public_result,
    record_contract_resolution_failure,
    tool_description,
)

_ACTIVE_PROFILE = active_profile()
_SERVER_INSTRUCTION_MODE = _ACTIVE_PROFILE.manifest.get(
    "invariants", {}
).get("operator_server_instructions")


def _operator_server_instructions(mode: object) -> str | None:
    if mode == "none_v1":
        return None
    if mode == "compact_v1":
        return (
            "One LIBERO episode is active. Use returned images and world-frame "
            "measurements as evidence, act closed-loop, and keep recoverable "
            "failures nonterminal."
        )
    return (
        "You are connected to one LIBERO embodied episode. Use visual results "
        "as the primary source of truth. The gateway hides simulator handles "
        "and backend payloads. Explicit pose deltas and registered world-frame "
        "poses remain visible, versioned actions. After every world "
        "changing step, inspect the returned image before continuing. A "
        "recoverable issue does not end the episode: report it, then retry or "
        "use the direct move/nudge escape hatches when pose conversion is suspect."
    )


_COMPACT_SERVER_CONTEXT = _SERVER_INSTRUCTION_MODE in {
    "compact_v1",
    "none_v1",
}
mcp = FastMCP(
    "embodied",
    instructions=_operator_server_instructions(_SERVER_INSTRUCTION_MODE),
    log_level="WARNING",
)
_GATEWAY: EmbodiedGateway | None = None
_CONTEXT_LOCK = threading.Lock()
_CONTEXT_SEQ = 0
_CONTROL_SERVER: ThreadingHTTPServer | None = None


def _profile_tool_description(name: str, legacy: str) -> str:
    """Use a profile-owned description when the profile owns this public tool."""

    profile = _ACTIVE_PROFILE
    if name in profile.public_operator_tools:
        return tool_description(name)
    return legacy


def _without_operator_images(result: GatewayResult) -> GatewayResult:
    """Keep lifecycle images in Replay without adding them to model context."""

    return GatewayResult(
        result.success,
        dict(result.text),
        images=[],
        details=dict(result.details),
    )


def _compact_lifecycle_result(
    name: str,
    result: GatewayResult,
) -> GatewayResult:
    """Project lifecycle tools to the fields needed for the next decision."""

    if not _COMPACT_INPUT_SCHEMA:
        return result
    source = result.text
    if name == "check_task":
        text = (
            {
                "success": True,
                "task_success": bool(source.get("task_success")),
            }
            if result.success
            else {
                "success": False,
                "reason": str(
                    source.get("reason")
                    or source.get("issue_code")
                    or "task_checker_unavailable"
                ),
                "retryable": bool(source.get("retryable", True)),
            }
        )
    elif name == "report_issue":
        text = (
            {"success": True, "recorded": True}
            if result.success
            else {
                "success": False,
                "reason": str(source.get("reason") or "record_failed"),
            }
        )
    elif name == "finish_episode":
        if result.success:
            text = {
                "success": True,
                "outcome": str(source.get("outcome") or ""),
            }
        else:
            text = {
                "success": False,
                "reason": str(
                    source.get("reason")
                    or (
                        "native_task_not_confirmed"
                        if source.get("error")
                        else "finish_rejected"
                    )
                ),
                "retryable": bool(source.get("retryable", True)),
            }
            missing = source.get("missing_or_invalid_fields")
            if isinstance(missing, list) and missing:
                text["required_fields"] = [
                    str(value) for value in missing
                ]
    else:  # pragma: no cover - internal misuse guard.
        raise ValueError(f"unsupported compact lifecycle result: {name}")
    return GatewayResult(
        result.success,
        text,
        images=[],
        details={
            **dict(result.details),
            "pre_compact_lifecycle_result": dict(source),
        },
    )


class MoveToTarget(BaseModel):
    """Typed public contract for the single manipulation action."""

    model_config = ConfigDict(extra="forbid")

    position_point_id: str | None = Field(
        None, description="Solved mark_point ID used as the exact grip-site destination."
    )
    position_xyz_m: list[float] | None = Field(
        None,
        min_length=3,
        max_length=3,
        description="Direct numeric world-frame grip-site destination [x,y,z] in metres.",
    )
    position_delta_mm: list[float] | None = Field(
        None,
        min_length=3,
        max_length=3,
        description="World-frame grip-site displacement [dX,dY,dZ] in millimetres.",
    )
    position_delta_from_point_id: str | None = Field(
        None, description="First solved point in a measured B-A translation."
    )
    position_delta_to_point_id: str | None = Field(
        None, description="Second solved point in a measured B-A translation."
    )
    approach_from_point_id: str | None = Field(
        None, description="Solved direction anchor defining the approach axis."
    )
    jaw_toward_point_id: str | None = Field(
        None, description="Solved direction anchor defining the jaw axis."
    )
    gripper: Literal["open", "close"] | None = Field(
        None,
        description='Optional gripper command. Example: {"gripper":"close"}.',
    )
    observation_id: str | None = Field(
        None, description="Optional assertion for the observation used as a delta base."
    )
    delta_frame: Literal["world"] | None = Field(
        None, description="Only world-frame deltas are supported."
    )
    preview_only: bool = Field(
        False, description="Resolve and render the pose without moving or changing the gripper."
    )
    rotation_matrix: list[list[float]] | None = Field(
        None, description="Low-level 3x3 world-frame grip-site rotation."
    )
    approach_direction_world: list[float] | None = Field(
        None,
        min_length=3,
        max_length=3,
        description="Low-level world-frame approach direction.",
    )
    jaw_direction_world: list[float] | None = Field(
        None,
        min_length=3,
        max_length=3,
        description="Low-level world-frame jaw direction.",
    )
    orientation_resolution_policy: str | None = Field(
        None, description="Optional diagnostic label for orientation resolution."
    )


class ObserveInspectionBox(BaseModel):
    """One pixel crop on a calibrated point-cloud view."""

    model_config = ConfigDict(extra="forbid")

    view: Literal[
        "pointcloud_top",
        "pointcloud_front",
        "pointcloud_side",
    ] = Field(
        ..., description="Calibrated source view containing the crop."
    )
    box_xyxy: list[int] = Field(
        ...,
        min_length=4,
        max_length=4,
        description=(
            "Crop rectangle [x0,y0,x1,y1] in source-view pixels, with "
            "top-left origin and x1/y1 exclusive."
        ),
    )


class ObserveInspection(BaseModel):
    """Optional clean local crops attached to observe."""

    model_config = ConfigDict(extra="forbid")

    boxes: list[ObserveInspectionBox] = Field(
        ...,
        min_length=1,
        max_length=3,
        description=(
            "One to three point-cloud pixel boxes to crop from the same fresh "
            "observation. Returned inspection views can be passed to mark_point."
        ),
    )


class DiagnosticHypothesis(BaseModel):
    """One explicitly uncertain explanation for the observed failure."""

    model_config = ConfigDict(extra="forbid")

    suspected_layer: Literal[
        "operator_strategy",
        "prompt_or_context",
        "tool_contract",
        "visualization",
        "perception_geometry",
        "motion_controller",
        "gripper_or_contact",
        "simulator_or_task",
        "unknown",
    ] = Field(
        description=(
            "A suspected layer, not an established root cause. Use unknown "
            "instead of forcing attribution."
        )
    )
    explanation: str = Field(
        min_length=1,
        description="The falsifiable explanation being proposed.",
    )
    supporting_evidence: str = Field(
        min_length=1,
        description="Concrete evidence that is consistent with this explanation.",
    )
    missing_or_conflicting_evidence: str = Field(
        min_length=1,
        description=(
            "Evidence that is unavailable, ambiguous, or conflicts with the "
            "explanation."
        ),
    )
    confidence: Literal["low", "medium", "high"] = Field(
        description="Confidence in this hypothesis, not in the observed failure."
    )


class SingleVariableIntervention(BaseModel):
    """A proposed A/B test; it is not authorization to adopt the change."""

    model_config = ConfigDict(extra="forbid")

    independent_variable: str = Field(
        min_length=1,
        description="The one context, visualization, tool, or runtime factor to change.",
    )
    control_condition: str = Field(
        min_length=1,
        description="The current baseline condition.",
    )
    treatment_condition: str = Field(
        min_length=1,
        description="The proposed condition differing in only the named variable.",
    )
    held_constant: list[str] = Field(
        min_length=1,
        description=(
            "Important factors to keep fixed, such as model, seed schedule, task, "
            "tool schema, controller, or point-cloud backend."
        ),
    )
    predicted_effect: str = Field(
        min_length=1,
        description="The behavior expected to change if the hypothesis is correct.",
    )
    primary_metric: str = Field(
        min_length=1,
        description="The predeclared causal metric used to compare treatment and control.",
    )
    adoption_criterion: str = Field(
        min_length=1,
        description=(
            "The repeatable A/B result required before adopting the treatment."
        ),
    )
    execution_scope: Literal[
        "current_tools",
        "system_change",
        "external_or_unavailable",
    ] = Field(
        description=(
            "Whether the proposed intervention can be executed now with the "
            "currently exposed Operator tools, requires a system change, or "
            "depends on unavailable external capability or evidence."
        )
    )
    current_tool_plan: list[str] = Field(
        description=(
            "Concrete current-tool calls or actions that would execute the "
            "intervention. Required and non-empty when execution_scope is "
            "current_tools; otherwise use an empty list."
        )
    )
    attempted_in_episode: bool = Field(
        description=(
            "Whether this exact intervention was attempted in the current episode."
        )
    )
    attempt_evidence_refs: list[str] = Field(
        description=(
            "Trace, observation, motion, or checker references showing the exact "
            "intervention was attempted. Required and non-empty when "
            "attempted_in_episode is true."
        )
    )
    planned_current_tool_trials: int = Field(
        ge=0,
        description=(
            "Number of matched current-tool trials declared necessary for this "
            "intervention. Must be positive for current_tools."
        ),
    )
    completed_current_tool_trials: int = Field(
        ge=0,
        description=(
            "Number of those declared matched trials completed in this episode."
        ),
    )
    remaining_current_tool_actions: list[str] = Field(
        description=(
            "Concrete declared current-tool actions still needed. Failure remains "
            "blocked while this list is non-empty."
        )
    )
    contradicting_attempts: list[str] = Field(
        description=(
            "Attempt evidence that conflicts with the proposed explanation or "
            "predicted effect. Use an empty list when none is known."
        )
    )
    operator_claimed_exhaustion_reason: str = Field(
        description=(
            "Why the intervention is exhausted now or cannot be executed in this "
            "episode. This is an Operator claim, not established root-cause truth."
        )
    )


class OperatorFeedback(BaseModel):
    """Compact end-of-episode feedback for tool/context design."""

    model_config = ConfigDict(extra="forbid")

    tool_contract_issues: list[str] = Field(
        default_factory=list,
        description="Specific tool inputs or outputs that were ambiguous, redundant, or misleading.",
    )
    context_issues: list[str] = Field(
        default_factory=list,
        description="Specific prompt/context instructions that were unclear, excessive, or contradictory.",
    )
    redundant_information: list[str] = Field(
        default_factory=list,
        description="Information that did not help the next decision and should be removed or hidden.",
    )
    missing_capabilities: list[str] = Field(
        default_factory=list,
        description="Capabilities or tools that would materially improve the task.",
    )
    helpful_evidence: list[str] = Field(
        default_factory=list,
        description="Returned evidence or tool behavior that was especially useful.",
    )
    blocked_step: str = Field(
        default="",
        description="The concrete step where the interface most limited progress.",
    )
    confidence: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Confidence in this interface feedback, not in the task diagnosis.",
    )


class FailurePostmortem(BaseModel):
    """Evidence-first failure signature with optional diagnosis metadata.

    Episode termination should not depend on the Operator completing a large
    research questionnaire.  The evidence fields are the executable contract;
    hypotheses and interventions are optional trace metadata and are still
    validated when supplied.
    """

    model_config = ConfigDict(extra="forbid")

    progress_stopped_at: Literal[
        "target_identification",
        "point_authoring",
        "pose_or_orientation",
        "approach",
        "grasp_contact",
        "retention",
        "transport",
        "placement",
        "task_verification",
        "recovery",
        "unknown",
    ] = Field(description="The stage where useful progress ultimately stopped.")
    expected_observation: str = Field(
        min_length=1,
        description="What observable result should have followed the action.",
    )
    actual_observation: str = Field(
        min_length=1,
        description="What was actually observed instead, without root-cause claims.",
    )
    evidence_refs: list[str] = Field(
        min_length=1,
        description=(
            "Concrete observation IDs, image views, coordinates, motion results, "
            "gripper state, or checker results supporting the failure signature."
        ),
    )
    recovery_attempts: list[str] = Field(
        min_length=1,
        description="Useful recovery actions already tried, each with its outcome.",
    )
    diagnostic_hypotheses: list[DiagnosticHypothesis] | None = Field(
        None,
        description=(
            "Optional uncertain explanations. Agreement does not establish root "
            "cause; omit when there is no useful hypothesis."
        ),
    )
    proposed_intervention: SingleVariableIntervention | None = Field(
        None,
        description=(
            "Optional single-variable intervention proposal. If execution_scope "
            "is current_tools, the declared actions must be attempted before "
            "failure can be finalized."
        ),
    )


class MoveToTargetV5(BaseModel):
    """v5 contract with natural names for the measured B-A point pair."""

    model_config = ConfigDict(extra="forbid")

    position_point_id: str | None = Field(
        None, description="Solved mark_point ID used as the exact grip-site destination."
    )
    position_xyz_m: list[float] | None = Field(
        None,
        min_length=3,
        max_length=3,
        description="Direct numeric world-frame grip-site destination [x,y,z] in metres.",
    )
    position_delta_mm: list[float] | None = Field(
        None,
        min_length=3,
        max_length=3,
        description="World-frame grip-site displacement [dX,dY,dZ] in millimetres.",
    )
    position_from_point_id: str | None = Field(
        None, description="Measured source point A in a B-A grip-site translation."
    )
    position_to_point_id: str | None = Field(
        None, description="Measured destination point B in a B-A grip-site translation."
    )
    approach_from_point_id: str | None = Field(
        None, description="Solved direction anchor defining the approach axis."
    )
    jaw_toward_point_id: str | None = Field(
        None, description="Solved direction anchor defining the jaw axis."
    )
    gripper: Literal["open", "close"] | None = Field(
        None,
        description='Optional gripper command. Example: {"gripper":"close"}.',
    )
    observation_id: str | None = Field(
        None, description="Optional assertion for the observation used as a delta base."
    )
    delta_frame: Literal["world"] | None = Field(
        None, description="Only world-frame deltas are supported."
    )
    preview_only: bool = Field(
        False, description="Resolve and render the pose without moving or changing the gripper."
    )
    rotation_matrix: list[list[float]] | None = Field(
        None, description="Low-level 3x3 world-frame grip-site rotation."
    )
    approach_direction_world: list[float] | None = Field(
        None,
        min_length=3,
        max_length=3,
        description="Low-level world-frame approach direction.",
    )
    jaw_direction_world: list[float] | None = Field(
        None,
        min_length=3,
        max_length=3,
        description="Low-level world-frame jaw direction.",
    )
    orientation_resolution_policy: str | None = Field(
        None, description="Optional diagnostic label for orientation resolution."
    )


class MoveToTargetV6(MoveToTargetV5):
    """Composed contract that presents direction vectors as normal controls."""

    approach_from_point_id: str | None = Field(
        None,
        description=(
            "Fallback marked anchor A for orientation. With grip-site P0, "
            "approach is normalize(P0-A). Use direct approach_direction_world "
            "when the desired world direction is already known."
        ),
    )
    jaw_toward_point_id: str | None = Field(
        None,
        description=(
            "Fallback marked anchor J for orientation. With grip-site P0, the "
            "jaw hint is normalize(J-P0). Use direct jaw_direction_world when "
            "the desired world direction is already known."
        ),
    )
    approach_direction_world: list[float] | None = Field(
        None,
        min_length=3,
        max_length=3,
        description=(
            "Preferred direct world-frame approach direction for grip-site local "
            "+Z. It is normalized by the resolver. Omit every position field to "
            "rotate in place."
        ),
    )
    jaw_direction_world: list[float] | None = Field(
        None,
        min_length=3,
        max_length=3,
        description=(
            "Preferred direct world-frame jaw direction for grip-site local +X. "
            "It is projected perpendicular to approach and normalized. Supply "
            "it with approach_direction_world for a fully specified orientation."
        ),
    )


class MoveToTargetCompact(BaseModel):
    """Compact world-frame grip-site control."""

    model_config = ConfigDict(extra="forbid")

    position_point_id: str | None = None
    position_xyz_m: list[float] | None = Field(
        None,
        min_length=3,
        max_length=3,
    )
    position_delta_mm: list[float] | None = Field(
        None,
        min_length=3,
        max_length=3,
    )
    position_from_point_id: str | None = None
    position_to_point_id: str | None = None
    approach_from_point_id: str | None = None
    jaw_toward_point_id: str | None = None
    approach_direction_world: list[float] | None = Field(
        None,
        min_length=3,
        max_length=3,
    )
    jaw_direction_world: list[float] | None = Field(
        None,
        min_length=3,
        max_length=3,
    )
    gripper: Literal["open", "close"] | None = None
    preview_only: bool = False


_POINT_PAIR_FIELDS = _ACTIVE_PROFILE.manifest.get("invariants", {}).get(
    "point_pair_translation_fields"
)
_DIRECTION_VECTOR_CONTRACT = _ACTIVE_PROFILE.manifest.get(
    "invariants", {}
).get("direction_vector_contract")
_OBSERVE_INSPECTION_ENABLED = (
    _ACTIVE_PROFILE
    .manifest.get("invariants", {})
    .get("observe_inspection_policy")
    != "disabled"
)
_COMPACT_INPUT_SCHEMA = (
    _ACTIVE_PROFILE
    .manifest.get("invariants", {})
    .get("operator_input_schema")
    == "compact_v1"
)
_FLAT_MOVE_TO_INPUT = (
    _ACTIVE_PROFILE
    .manifest.get("invariants", {})
    .get("move_to_input_shape")
    == "flat_v1"
)
_KISS_MOVE_TO_INPUT = (
    _ACTIVE_PROFILE
    .manifest.get("invariants", {})
    .get("move_to_input_shape")
    == "kiss_flat_v1"
)
_KISS_MOVE_TO_INPUT_V2 = (
    _ACTIVE_PROFILE
    .manifest.get("invariants", {})
    .get("move_to_input_shape")
    == "kiss_flat_v2"
)
_KISS_MOVE_TO_INPUT_V3 = (
    _ACTIVE_PROFILE
    .manifest.get("invariants", {})
    .get("move_to_input_shape")
    == "kiss_flat_v3"
)
_KISS_MOVE_TO_INPUT_V4 = (
    _ACTIVE_PROFILE
    .manifest.get("invariants", {})
    .get("move_to_input_shape")
    == "kiss_flat_v4"
)
_KISS_OPERATOR_INPUT = (
    _KISS_MOVE_TO_INPUT
    or _KISS_MOVE_TO_INPUT_V2
    or _KISS_MOVE_TO_INPUT_V3
    or _KISS_MOVE_TO_INPUT_V4
)
_KISS_FINISH_NO_FEEDBACK = (
    _ACTIVE_PROFILE
    .manifest.get("invariants", {})
    .get("finish_episode_input_shape")
    == "kiss_no_feedback_v1"
)
ActiveMoveToTarget = (
    MoveToTargetCompact
    if _COMPACT_INPUT_SCHEMA
    else MoveToTargetV6
    if (
        _POINT_PAIR_FIELDS
        == [
            "position_from_point_id",
            "position_to_point_id",
        ]
        and _DIRECTION_VECTOR_CONTRACT == "world_frame_unit_vectors"
    )
    else MoveToTargetV5
    if _POINT_PAIR_FIELDS
    == [
        "position_from_point_id",
        "position_to_point_id",
    ]
    else MoveToTarget
)


def _gateway() -> EmbodiedGateway:
    if _GATEWAY is None:
        raise RuntimeError("Embodied gateway is not configured")
    return _GATEWAY


class _ManualControlHandler(BaseHTTPRequestHandler):
    def _json(self, payload: object, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
            gateway = _gateway()
            manual_tool = self.path.removeprefix("/")
            if self.path == "/observe":
                result = gateway.observe(
                    views=payload.get("views"),
                    inspect=payload.get("inspect"),
                )
            elif self.path == "/abort":
                # Host/runtime lifecycle endpoint.  This is intentionally
                # outside the public Operator MCP surface: it records an
                # externally interrupted episode as aborted and flushes the
                # durable writer before the Gateway process is stopped.
                reason = str(
                    payload.get("reason")
                    or "External Operator/runtime stop."
                )
                result = gateway.finish_episode("abort", reason=reason)
                if (
                    not result.success
                    and result.text.get("reason") == "task_already_completed"
                ):
                    result = gateway.finish_episode(
                        "success",
                        reason=(
                            "Native task success was confirmed before the "
                            "Operator runtime stopped."
                        ),
                    )
                    manual_tool = "finalize_success"
            elif self.path == "/check_task":
                result = gateway.check_task()
            elif self.path == "/mark_point":
                # The public Operator contract names pixel coordinates x/y;
                # the gateway library uses u/v. Accept both at the manual
                # Chrome boundary so browser testing exercises the same
                # contract instead of failing on an adapter-only KeyError.
                raw_u = payload.get("u", payload.get("x"))
                raw_v = payload.get("v", payload.get("y"))
                if raw_u is None or raw_v is None:
                    return self._json({
                        "success": False,
                        "reason": "invalid_mark_point",
                        "message": "mark_point requires x/y pixel coordinates.",
                    }, 400)
                result = gateway.mark_point_3d(
                    view=str(payload["view"]),
                    u=int(raw_u),
                    v=int(raw_v),
                    point_id=str(payload.get("point_id") or "P0"),
                    label=str(payload.get("label") or ""),
                )
            elif self.path == "/move_to":
                legacy_target = payload.get("target")
                if isinstance(legacy_target, dict):
                    target = dict(legacy_target)
                    if "preview_only" in payload and "preview_only" not in target:
                        target["preview_only"] = bool(payload["preview_only"])
                else:
                    target = {
                        "position_xyz_m": payload.get("xyz_m"),
                        "position_delta_mm": payload.get("delta_mm"),
                        "approach_direction_world": payload.get("approach_world"),
                        "jaw_direction_world": payload.get("jaw_world"),
                        "gripper": payload.get("gripper"),
                        "preview_only": (
                            True if payload.get("preview") is True else None
                        ),
                    }
                    target = {
                        key: value
                        for key, value in target.items()
                        if value is not None
                    }
                if not target:
                    return self._json({
                        "success": False,
                        "reason": "invalid_target",
                        "message": "move_to requires one control input.",
                    })
                result = gateway.move_to_target(
                    target,
                    reason="manual Chrome Operator harness",
                )
            else:
                return self._json({"success": False, "reason": "unknown_manual_endpoint"}, 404)
            _record_operator_context(
                tool=f"manual_{manual_tool}",
                arguments=payload,
                blocks=_content(
                    result,
                    public_tool=(
                        manual_tool
                        if manual_tool in {"observe", "mark_point", "move_to"}
                        else None
                    ),
                ),
            )
            return self._json({"success": result.success, "text": result.text})
        except Exception as exc:  # pragma: no cover - browser-facing error path
            return self._json({"success": False, "reason": "manual_control_error", "message": str(exc)}, 400)

    def log_message(self, *_args: object) -> None:
        pass


def _content(
    result: GatewayResult,
    *,
    public_tool: str | None = None,
) -> list[Any]:
    if public_tool is not None:
        # A move_to execution failure is still a complete public result when
        # it has the endpoint-status contract.  In particular, stalled and
        # non-converged motions intentionally report:
        #   motion_status, actual_grip_site_xyz_m, remaining_target_delta_mm
        # rather than the generic {reason, message} error shape.  Do not
        # downgrade that structured result merely because it has no `reason`.
        structured_move_failure = (
            public_tool == "move_to"
            and result.text.get("success") is False
            and result.text.get("motion_status")
            in {"not_reached", "stalled", "not_converged", "failed"}
            and result.text.get("actual_grip_site_xyz_m") is not None
        )
        if (
            result.text.get("success") is False
            and "reason" not in result.text
            and not structured_move_failure
        ):
            original = dict(result.text)
            reason = str(
                original.get("issue_code")
                or original.get("error")
                or "tool_failed"
            )
            projected: dict[str, Any] = {
                "kind": str(original.get("kind") or public_tool),
                "success": False,
                "reason": reason,
            }
            for key in (
                "retryable",
                "observation_id",
                "point_id",
                "episode_status",
            ):
                if original.get(key) is not None:
                    projected[key] = original[key]
            source_view = original.get("source_view", original.get("view"))
            if source_view is not None:
                projected["source_view"] = source_view
            message = original.get("message", original.get("error"))
            if message is not None:
                projected["message"] = str(message)
            result.details.setdefault(
                "pre_contract_public_result",
                original,
            )
            result.text = projected
        projected, removed = project_public_result(public_tool, result.text)
        if removed:
            result.details.setdefault(
                "pre_contract_public_result",
                dict(result.text),
            )
            result.details.setdefault(
                "public_result_removed_fields",
                removed,
            )
        result.text = projected
    blocks: list[Any] = [
        TextContent(
            type="text",
            text=json.dumps(result.text, ensure_ascii=False, separators=(",", ":")),
        )
    ]
    for path in result.images:
        if path.is_file():
            blocks.append(Image(path=path))
    return blocks


def _operator_tool(
    *,
    description: str,
    annotations: ToolAnnotations,
    exposed: bool = True,
):
    """Register a non-blocking MCP wrapper around a sync gateway method.

    The gateway intentionally has a synchronous library API, while the
    simulator MCP transport uses async MCP clients internally. FastMCP invokes
    synchronous tools inline on its event loop, so calling the gateway there
    would make ``asyncio.run`` fail (and would block SSE replies). Keep the
    direct sync functions available for host-side tests, but register an
    async thread-offloaded wrapper for the actual Operator MCP server.
    """

    def decorator(function):
        if exposed:
            @mcp.tool(
                name=function.__name__,
                description=description,
                annotations=annotations,
                structured_output=False,
            )
            @functools.wraps(function)
            async def _async_wrapper(**kwargs):
                blocks = await anyio.to_thread.run_sync(
                    functools.partial(function, **kwargs)
                )
                try:
                    await anyio.to_thread.run_sync(
                        functools.partial(
                            _record_operator_context,
                            tool=function.__name__,
                            arguments=kwargs,
                            blocks=blocks,
                        )
                    )
                except Exception:  # noqa: BLE001 - context mirroring must not break tools.
                    pass
                return blocks

        return function

    return decorator


def _record_operator_context(
    *, tool: str, arguments: dict[str, Any], blocks: list[Any]
) -> None:
    """Retain the exact MCP projection delivered to the clean Operator."""

    global _CONTEXT_SEQ
    gateway = _gateway()
    normalized_arguments = _json_compatible(arguments)
    texts: list[str] = []
    images: list[str] = []
    for block in blocks:
        if isinstance(block, TextContent):
            texts.append(block.text)
        elif isinstance(block, Image):
            images.append(str(block.path))
    with _CONTEXT_LOCK:
        _CONTEXT_SEQ += 1
        event = {
            "schema_version": "openeta.operator_context.v1",
            "seq": _CONTEXT_SEQ,
            "timestamp_s": time.time(),
            "tool": tool,
            "arguments": normalized_arguments,
            "response_text_blocks": texts,
            "response_image_paths": images,
        }
        path = gateway.root / "operator_context.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(event, ensure_ascii=False, default=str) + "\n"
            )


def _json_compatible(value: Any) -> Any:
    """Convert nested MCP/Pydantic arguments to durable structured JSON."""

    if isinstance(value, BaseModel):
        return _json_compatible(value.model_dump())
    if isinstance(value, Mapping):
        return {
            str(key): _json_compatible(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _observe_with_inspection(
    views: list[str] | None = None,
    inspect: ObserveInspection | None = None,
    history_point_ids: list[str] | None = None,
) -> list[Any]:
    gateway = _gateway()
    if inspect is None:
        if history_point_ids is None:
            result = gateway.observe(views=views)
        else:
            result = gateway.observe(
                views=views,
                history_point_ids=history_point_ids,
            )
    else:
        kwargs = {
            "views": views,
            "inspect": inspect.model_dump(exclude_none=True),
        }
        if history_point_ids is not None:
            kwargs["history_point_ids"] = history_point_ids
        result = gateway.observe(**kwargs)
    return _content(result, public_tool="observe")


def _observe_source_views_only(
    views: list[str] | None = None,
    history_point_ids: list[str] | None = None,
) -> list[Any]:
    gateway = _gateway()
    if history_point_ids is None:
        result = gateway.observe(views=views)
    else:
        result = gateway.observe(
            views=views,
            history_point_ids=history_point_ids,
        )
    return _content(result, public_tool="observe")


_observe_impl = (
    _observe_with_inspection
    if _OBSERVE_INSPECTION_ENABLED
    else _observe_source_views_only
)
_observe_impl.__name__ = "observe"
observe = _operator_tool(
    description=tool_description("observe"),
    annotations=ToolAnnotations(
        readOnlyHint=True,
        idempotentHint=True,
        destructiveHint=False,
    ),
)(_observe_impl)


@_operator_tool(
    description=(
        "Show a canonical LIBERO simulator asset image for an explicit object "
        "category such as `alphabet soup` when the task name is visually "
        "unclear. This is a read-only visual vocabulary lookup: it returns no "
        "scene coordinates, does not select an instance, and does not alter "
        "SAM3. The first image is a canonical 3D multi-view render when the "
        "asset supports it; later images are source textures. Compare dominant "
        "colors, shape, and visible label fragments with observe before using "
        "Python marking or segment_object yourself."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, destructiveHint=False),
    exposed=False,
)
def inspect_object_reference(target: str) -> list[Any]:
    return _content(_gateway().inspect_object_reference(target))


@_operator_tool(
    description=(
        "Segment a semantic target in the current camera observation. Returns "
        "a labeled SAM3 mask image and stable detection ids. Prefer this "
        "text-only call first when the task gives a distinctive object name; "
        "use python_mark_object only when the returned instances remain ambiguous. If you previously "
        "called python_mark_object with exactly one point or box, that mark is "
        "passed as a positive SAM3 point prompt on the original image (a box "
        "uses its visible center); `target` is "
        "kept as its readable label and is not rewritten. If there are "
        "multiple detections, inspect the image and choose one before asking "
        "for grasps. Do not provide image paths or numeric calibration."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, destructiveHint=False),
    exposed=False,
)
def segment_object(target: str) -> list[Any]:
    return _content(_gateway().segment_object(target))


@_operator_tool(
    description=(
        "Run a small Python snippet against the current agentview image to "
        "disambiguate an instance after text grounding is insufficient. Do not "
        "use a guessed point to replace a distinctive task name without visual "
        "verification. The snippet receives `image`, "
        "`width`, `height`, `mark_box(x0, y0, x1, y1, ...)`, and "
        "`mark_point(x, y, ...)`; tuple forms are also accepted. Inspect "
        "the returned marked image, then call segment_object explicitly. Exactly "
        "The result includes the full marked image and a local zoom for identity "
        "verification. Exactly one point or box is required for SAM3 geometric prompting; a box maps "
        "deterministically to its center point, while multiple marks return a "
        "retryable error rather than being guessed. "
        "This does not move the robot."
    ),
    annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=True, destructiveHint=False),
    exposed=False,
)
def python_mark_object(code: str) -> list[Any]:
    return _content(_gateway().python_mark_object(code))


@_operator_tool(
    description=(
        "Select one visible SAM3 detection from the most recent segmentation "
        "image. This records the Operator's visual choice for the current "
        "observation; it does not move the robot."
    ),
    annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=True, destructiveHint=False),
    exposed=False,
)
def select_detection(detection_id: str, reason: str = "") -> list[Any]:
    return _content(_gateway().select_detection(detection_id, reason=reason))


@_operator_tool(
    description=(
        "Propose AnyGrasp candidates for the selected SAM3 object in the same "
        "current observation. Normally call this after select_detection; an "
        "explicit detection_id may also be provided for recovery. "
        "Returns stable candidate ids and triggers the observation-bound Viser "
        "proposal scene. Static pose overlays are not returned. Wait for "
        "get_grasp_inspector to match the current observation, then use "
        "inspect_grasp or the explicit Viser camera tools before selection."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, destructiveHint=False),
    exposed=False,
)
def propose_grasps(detection_id: str | None = None) -> list[Any]:
    return _content(_gateway().propose_grasps(detection_id=detection_id))


@_operator_tool(
    description=(
        "Inspect exactly one AnyGrasp candidate from the most recent proposal. "
        "This read-only operation maps the candidate to its matching Viser pose, "
        "focuses a jaws-side camera, and returns a native Viser capture. It does "
        "not commit the grasp or move the robot. Retry if the matching proposal "
        "scene has not loaded yet."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, destructiveHint=False),
    exposed=False,
)
def inspect_grasp(grasp_id: str) -> list[Any]:
    return _content(_gateway().inspect_grasp(grasp_id))


@_operator_tool(
    description=(
        "Read the persistent 3D inspector state. Proposal scenes return stable "
        "AG/GX/R ids; post-move execution scenes return exact TARGET/ACTUAL ids "
        "bound to that post-action RGB-D observation. This never moves the robot."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, destructiveHint=False),
    exposed=False,
)
def get_grasp_inspector() -> list[Any]:
    return _content(_gateway().get_grasp_inspector())


@_operator_tool(
    description=(
        "Set the 3D grasp viewer explicitly. pose_scope is default, all, or "
        "focus. camera_preset is keep, scene, top, front, side, pose_approach, "
        "or pose_jaws. orbit_azimuth_deg and orbit_elevation_deg rotate the "
        "current camera around the focused pose; zoom_scale changes distance. "
        "Call capture_grasp_view(camera='current') after each chosen view. This "
        "only changes visualization."
    ),
    annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=True, destructiveHint=False),
    exposed=False,
)
def configure_grasp_view(
    show_anygrasp: bool = True,
    show_graspgenx: bool = True,
    show_refined: bool = True,
    pose_scope: Literal["default", "all", "focus"] = "default",
    focus_pose_id: str | None = None,
    camera_preset: Literal[
        "keep", "scene", "top", "front", "side", "pose_approach", "pose_jaws"
    ] = "keep",
    viewer_id: str | None = None,
    orbit_azimuth_deg: float = 0.0,
    orbit_elevation_deg: float = 0.0,
    zoom_scale: float = 1.0,
) -> list[Any]:
    return _content(
        _gateway().configure_grasp_view(
            show_anygrasp=show_anygrasp,
            show_graspgenx=show_graspgenx,
            show_refined=show_refined,
            pose_scope=pose_scope,
            focus_pose_id=focus_pose_id,
            camera_preset=camera_preset,
            viewer_id=viewer_id,
            orbit_azimuth_deg=orbit_azimuth_deg,
            orbit_elevation_deg=orbit_elevation_deg,
            zoom_scale=zoom_scale,
        )
    )


@_operator_tool(
    description=(
        "Capture a native image from the connected 3D grasp viewer. Use current "
        "for the exact human viewport, or a named deterministic camera. Viewer "
        "and capture failures are explicit; no pose is substituted and the "
        "robot never moves."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=False, destructiveHint=False),
    exposed=False,
)
def capture_grasp_view(
    camera: Literal[
        "current", "scene", "top", "front", "side", "pose_approach", "pose_jaws"
    ] = "current",
    viewer_id: str | None = None,
) -> list[Any]:
    return _content(
        _gateway().capture_grasp_view(camera=camera, viewer_id=viewer_id)
    )


@_operator_tool(
    description=(
        "Create an immutable child pose by applying an explicit translation and "
        "rotation delta to any current AG, GX, or R pose. translation_delta_mm "
        "and rotation_delta_deg are xyz triples. frame is grasp_local or world. "
        "The new green R pose is registered in Viser immediately; the parent "
        "pose is never overwritten and the robot does not move. Inspect the result, refine it "
        "again if needed, then select_grasp with the returned pose_id."
    ),
    annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=False, destructiveHint=False),
    exposed=False,
)
def refine_grasp_pose(
    base_grasp_id: str,
    translation_delta_mm: list[float],
    rotation_delta_deg: list[float],
    frame: Literal["grasp_local", "world"] = "grasp_local",
    reason: str = "",
) -> list[Any]:
    return _content(
        _gateway().refine_grasp_pose(
            base_grasp_id=base_grasp_id,
            translation_delta_mm=translation_delta_mm,
            rotation_delta_deg=rotation_delta_deg,
            frame=frame,
            reason=reason,
        )
    )


@_operator_tool(
    description=(
        "Derive a deterministic top-down green R pose from the selected SAM3 "
        "mask and retained RGB-D. base_grasp_id is retained as proposal "
        "provenance. side is x_min, x_max, y_min, or y_max; insertion_depth_m "
        "moves the Panda grip_site downward from the robust visible top-surface "
        "estimate. A small value keeps the grip_site near the rim/top and is not "
        "automatically safer; inspect the jaws-side 3D view and refine again when "
        "the fingers do not overlap the intended contact region. A deeper value "
        "is not automatically safer either: geometry_diagnostics warns when the "
        "requested grip_site falls below the robust visible mask floor. Quantiles and "
        "depth are explicit and are never silently changed. This registers a "
        "pose but does not move the robot or guarantee contact/clearance. The "
        "result reports visible-top-to-grip-site depth relative to the Panda "
        "97 mm base-to-grip-site reference span and emits non-blocking shallow/"
        "deep engagement warnings."
    ),
    annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=False, destructiveHint=False),
    exposed=False,
)
def refine_mask_side_grasp(
    base_grasp_id: str,
    side: Literal["x_min", "x_max", "y_min", "y_max"],
    insertion_depth_m: float,
    surface_floor_quantile: float = 0.05,
    top_quantile: float = 0.99,
    reason: str = "",
) -> list[Any]:
    return _content(
        _gateway().refine_mask_side_grasp(
            base_grasp_id=base_grasp_id,
            side=side,
            insertion_depth_m=insertion_depth_m,
            surface_floor_quantile=surface_floor_quantile,
            top_quantile=top_quantile,
            reason=reason,
        )
    )


@_operator_tool(
    description=(
        "Register an exact Operator-computed world-frame Panda grip_site 4x4 "
        "transform as an immutable green R pose. The matrix is validated but "
        "never modified or substituted. base_grasp_id, method, reason, current "
        "observation binding, and the exact transform are retained as provenance. "
        "This does not move the robot."
    ),
    annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=False, destructiveHint=False),
    exposed=False,
)
def register_grasp_pose(
    pose_id: str,
    base_grasp_id: str,
    transform_world_from_grip_site: list[list[float]],
    method: str,
    reason: str = "",
) -> list[Any]:
    return _content(
        _gateway().register_grasp_pose(
            pose_id=pose_id,
            base_grasp_id=base_grasp_id,
            transform_world_from_grip_site=transform_world_from_grip_site,
            method=method,
            reason=reason,
        )
    )


@_operator_tool(
    description=(
        "Select one visible AG, GX, or refined R candidate by its stable id. "
        "This records a semantic Operator choice "
        "but does not move the robot."
    ),
    annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=True, destructiveHint=False),
    exposed=False,
)
def select_grasp(grasp_id: str, reason: str = "") -> list[Any]:
    return _content(_gateway().select_grasp(grasp_id, reason=reason))


@_operator_tool(
    description=(
        "Low-level escape hatch: move the Panda grip_site directly to an "
        "absolute world-frame position in metres and XYZ roll/pitch/yaw in "
        "degrees. This bypasses AG/GX/R pose conversion but keeps action logs, "
        "post-move raw images, and an exact TARGET/ACTUAL record for Viser. Use it to "
        "diagnose or recover from a grasp-pose conversion bug. tolerance_mm "
        "defaults to 3 mm; do not set it larger merely to hide a missed target."
    ),
    annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=False, destructiveHint=True),
    exposed=False,
)
def move_to_pose(
    position_xyz_m: list[float],
    rotation_rpy_deg: list[float],
    reason: str = "",
    tolerance_mm: float = 3.0,
    max_steps: int = 320,
) -> list[Any]:
    return _content(
        _gateway().move_to_pose(
            position_xyz_m,
            rotation_rpy_deg,
            reason=reason,
            tolerance_mm=tolerance_mm,
            max_steps=max_steps,
        )
    )


@_operator_tool(
    description=(
        "Low-level visual-servo escape hatch: move incrementally from the "
        "simulator-reported current Panda grip_site. translation_delta_mm and "
        "rotation_delta_deg each contain three XYZ values; frame is world, "
        "eef_local, or wrist_camera. wrist_camera uses OpenCV image axes: "
        "+X moves right in the latest wrist image, +Y moves down, and +Z moves "
        "forward along the wrist camera optical axis. Resolve each correction "
        "from the returned post-action wrist image; do not reuse a stale sign. "
        "Small repeated corrections are allowed, and every move "
        "returns post-action raw images and triggers a TARGET/ACTUAL Viser scene. tolerance_mm "
        "defaults to 2 mm so a small nudge cannot succeed without moving."
    ),
    annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=False, destructiveHint=True),
    exposed=False,
)
def nudge_end_effector(
    translation_delta_mm: list[float],
    rotation_delta_deg: list[float] | None = None,
    frame: Literal["world", "eef_local", "wrist_camera"] = "world",
    reason: str = "",
    tolerance_mm: float = 2.0,
    max_steps: int = 320,
) -> list[Any]:
    return _content(
        _gateway().nudge_end_effector(
            translation_delta_mm,
            rotation_delta_deg,
            frame=frame,
            reason=reason,
            tolerance_mm=tolerance_mm,
            max_steps=max_steps,
        )
    )


@_operator_tool(
    description=(
        "Execute one high-level stage. Allowed stages are "
        "move_to_selected_pregrasp, approach_selected_grasp, "
        "close_gripper, lift_grasp, and open_gripper. Normal selected-grasp "
        "execution must start with move_to_selected_pregrasp: it fully opens the gripper, moves only "
        "to a standoff pose, and returns agentview plus wrist RGB. Inspect whether "
        "the object is centered between the fingers, aperture is sufficient, and "
        "the short path is clear before calling approach_selected_grasp. That "
        "checkpoint authorizes exactly one approach attempt."
    ),
    annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=False, destructiveHint=True),
    exposed=False,
)
def step(
    stage: Literal[
        "move_to_selected_pregrasp",
        "approach_selected_grasp",
        "close_gripper",
        "lift_grasp",
        "open_gripper",
    ]
) -> list[Any]:
    return _content(_gateway().step(stage))


@_operator_tool(
    description=(
        "Verify observation-only grasp contact after lift_grasp. The gateway "
        "re-segments the exact semantic target retained at grasp execution and "
        "compares its metric RGB-D motion with the end-effector lift. Returns "
        "contact_status confirmed, not_grasped, or unknown. Never infer contact "
        "from gripper closure alone, and do not begin transport unless status is "
        "confirmed. If multiple current instances are visible, inspect the mask "
        "and retry with the matching detection_id. This tool never moves the robot."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, destructiveHint=False),
)
def verify_grasp(detection_id: str | None = None) -> list[Any]:
    return _content(_gateway().verify_grasp(detection_id=detection_id))


@_operator_tool(
    description=_profile_tool_description("check_task", (
        "Ask the native LIBERO task checker whether the task is complete. This "
        "returns only checker availability and success, never privileged state."
    )),
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, destructiveHint=False),
)
def check_task() -> list[Any]:
    return _content(
        _compact_lifecycle_result("check_task", _gateway().check_task()),
        public_tool="check_task" if _COMPACT_INPUT_SCHEMA else None,
    )


def _report_issue_legacy(
    component: str,
    code: str,
    message: str,
) -> list[Any]:
    return _content(_gateway().report_issue(component, code, message))


def _report_issue_compact(message: str) -> list[Any]:
    return _content(
        _compact_lifecycle_result(
            "report_issue",
            _gateway().report_issue(
                "operator_interface",
                "operator_report",
                message,
            ),
        ),
        public_tool="report_issue",
    )


_report_issue_impl = (
    _report_issue_compact
    if _COMPACT_INPUT_SCHEMA
    else _report_issue_legacy
)
_report_issue_impl.__name__ = "report_issue"
report_issue = _operator_tool(
    description=_profile_tool_description("report_issue", (
        "Record a concrete, visible attempt or tool issue without ending the "
        "episode. Prefer layer-specific codes: pose_execution_mismatch when "
        "TARGET and ACTUAL differ but Viser matches the visible robot; "
        "viser_scene_mismatch when ACTUAL or the cloud conflicts with raw RGB; "
        "grasp_contact_failure when motion aligns but close/lift does not carry "
        "the object; task_progress_failure when manipulation works but the task "
        "relation remains false. Wrong masks, dropped objects, and other concrete "
        "recoverable observations are also valid. A grasp_contact_failure means "
        "the attempted pose did not retain the object; it is not evidence that "
        "the tool is broken. The compact result only confirms recording and "
        "keeps the episode running. The current frames are retained; change a "
        "measured pose variable, retry, and verify retention before transport."
    )),
    annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=False, destructiveHint=False),
)(_report_issue_impl)


@_operator_tool(
    description=(
        "Compatibility alias for report_issue. It records a visible attempt "
        "failure but does not terminate the episode; use finish_episode only "
        "after recovery attempts are exhausted."
    ),
    annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=False, destructiveHint=False),
    exposed=False,
)
def report_failure(component: str, code: str, message: str) -> list[Any]:
    return _content(_gateway().report_failure(component, code, message))


def _finish_episode_legacy(
    outcome: Literal["success", "failure", "abort"],
    reason: str = "",
    failure_postmortem: FailurePostmortem | None = None,
    operator_feedback: OperatorFeedback | None = None,
) -> list[Any]:
    return _content(
        _gateway().finish_episode(
            outcome,
            reason=reason,
            failure_postmortem=(
                failure_postmortem.model_dump()
                if failure_postmortem is not None
                else None
            ),
            operator_feedback=(
                operator_feedback.model_dump()
                if operator_feedback is not None
                else None
            ),
        )
    )


def _finish_episode_compact(
    outcome: Literal["success", "failure", "abort"],
    reason: str = "",
    evidence: list[str] | None = None,
    attempts: list[str] | None = None,
    feedback: list[str] | None = None,
) -> list[Any]:
    reason = str(reason).strip()
    if outcome == "failure" and not reason:
        result = GatewayResult(
            False,
            {
                "kind": "episode_finish",
                "success": False,
                "retryable": True,
                "reason": "failure_reason_required",
                "missing_or_invalid_fields": ["reason"],
            },
        )
        return _content(
            _compact_lifecycle_result("finish_episode", result),
            public_tool="finish_episode",
        )
    postmortem = None
    if outcome == "failure":
        gateway = _gateway()
        current = gateway.current_record or {}
        trace_evidence, trace_attempts = _failure_evidence_from_trace(gateway)
        evidence_refs = [
            str(value).strip()
            for value in (evidence or [])
            if str(value).strip()
        ]
        if not evidence_refs:
            evidence_refs = trace_evidence
        if not evidence_refs and current.get("observation_id"):
            evidence_refs = [str(current["observation_id"])]
        recovery_attempts = [
            str(value).strip()
            for value in (attempts or [])
            if str(value).strip()
        ]
        if not recovery_attempts:
            recovery_attempts = trace_attempts
        postmortem = {
            "progress_stopped_at": "unknown",
            "expected_observation": gateway.task,
            "actual_observation": reason,
            "evidence_refs": evidence_refs,
            "recovery_attempts": recovery_attempts,
        }
    normalized_feedback = None
    if feedback:
        normalized_feedback = {
            "tool_contract_issues": [
                str(value).strip()
                for value in feedback
                if str(value).strip()
            ],
            "context_issues": [],
            "redundant_information": [],
            "missing_capabilities": [],
            "helpful_evidence": [],
            "blocked_step": "",
            "confidence": "medium",
        }
    result = _gateway().finish_episode(
        outcome,
        reason=reason,
        failure_postmortem=postmortem,
        operator_feedback=normalized_feedback,
    )
    return _content(
        _compact_lifecycle_result("finish_episode", result),
        public_tool="finish_episode",
    )


def _failure_evidence_from_trace(gateway: Any) -> tuple[list[str], list[str]]:
    """Derive compact failure evidence from the durable host-side trace.

    ``finish_episode`` should not make the Operator restate actions that the
    gateway has already recorded.  Keep this evidence internal to the retained
    postmortem; the public result remains the compact success/outcome response.
    """

    current = getattr(gateway, "current_record", None) or {}
    evidence_refs: list[str] = []
    observation_id = str(current.get("observation_id") or "").strip()
    if observation_id:
        evidence_refs.append(observation_id)

    root = getattr(gateway, "root", None)
    events_path = Path(root) / "events.jsonl" if root is not None else None
    if events_path is None or not events_path.is_file():
        return evidence_refs, []

    actions: list[dict[str, Any]] = []
    outcomes: dict[str, bool] = {}
    try:
        for raw_line in events_path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            event = json.loads(raw_line)
            action_id = str(event.get("action_id") or "").strip()
            if event.get("kind") == "action" and action_id:
                request = (event.get("payload") or {}).get("request") or {}
                stage = str(
                    request.get("stage")
                    or request.get("action")
                    or "world_action"
                ).strip()
                actions.append({"action_id": action_id, "stage": stage})
            elif event.get("kind") == "tool_result" and action_id:
                outcomes[action_id] = bool(
                    (event.get("payload") or {}).get("success")
                )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return evidence_refs, []

    # The latest actions best represent the recovery frontier.  Eight entries
    # are enough for diagnosis without duplicating a long episode in metadata.
    retained = actions[-8:]
    for action in retained:
        action_id = action["action_id"]
        evidence_refs.append(action_id)

    attempts = []
    for action in retained:
        action_id = action["action_id"]
        outcome = (
            str(outcomes[action_id]).lower()
            if action_id in outcomes
            else "unknown"
        )
        attempts.append(
            f"{action_id}: {action['stage']} "
            f"(tool_result_success={outcome})"
        )
    return list(dict.fromkeys(evidence_refs)), attempts


def _finish_episode_kiss(
    outcome: Literal["success", "failure", "abort"],
    reason: str = "",
    attempts: list[str] | None = None,
    feedback: list[str] | None = None,
) -> list[Any]:
    return _finish_episode_compact(
        outcome,
        reason=reason,
        attempts=attempts,
        feedback=feedback,
    )


def _finish_episode_kiss_no_feedback(
    outcome: Literal["success", "failure", "abort"],
    reason: str = "",
    attempts: list[str] | None = None,
) -> list[Any]:
    return _finish_episode_compact(
        outcome,
        reason=reason,
        attempts=attempts,
    )


_finish_episode_impl = (
    _finish_episode_kiss_no_feedback
    if _KISS_FINISH_NO_FEEDBACK
    else _finish_episode_kiss
    if _KISS_OPERATOR_INPUT
    else _finish_episode_compact
    if _COMPACT_INPUT_SCHEMA
    else _finish_episode_legacy
)
_finish_episode_impl.__name__ = "finish_episode"
finish_episode = _operator_tool(
    description=_profile_tool_description("finish_episode", (
        "Explicitly finalize the episode. outcome='success' is accepted only "
        "after check_task confirmed native success. For outcome='failure', provide "
        "the small evidence record using the exact fields progress_stopped_at, "
        "expected_observation, actual_observation, evidence_refs, and "
        "recovery_attempts; do not use historical names such as stopping_stage, "
        "expected_result, or actual_result. "
        "evidence references, and useful recoveries already tried. Optional "
        "diagnostic_hypotheses, proposed_intervention, and operator_feedback fields "
        "are trace metadata, not a root-cause verdict. operator_feedback can record "
        "tool contract problems, context problems, redundant information, missing "
        "capabilities, helpful evidence, and the blocked step. If a proposed intervention is executable with "
        "current tools, its declared actions must be attempted and evidenced before "
        "failure can be finalized; otherwise the episode remains running. "
        "outcome='abort' is for an external stop and may use reason without a "
        "postmortem."
    )),
    annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=True, destructiveHint=True),
)(_finish_episode_impl)


@_operator_tool(
    description=_profile_tool_description(
        "close_episode",
        "Close the current embodied episode and flush its durable event log.",
    ),
    annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=True, destructiveHint=True),
)
def close_episode() -> list[Any]:
    gateway = _gateway()
    gateway.close()
    cleanup_error = gateway.close_error
    episode_status = gateway.episode_status
    episode_success = gateway.episode_success
    text = {
        "kind": "episode_end",
        "success": cleanup_error is None and episode_status != "failed",
        "episode_status": episode_status,
        "episode_success": episode_success,
        "message": "Episode closed and event log flushed."
        if cleanup_error is None and episode_status != "failed"
        else "Episode finalized as failed; inspect the failure case and Builder artifacts.",
    }
    if cleanup_error:
        text["cleanup_error"] = cleanup_error
    if gateway.failure_case:
        text["failure_case_id"] = gateway.failure_case.get("failure_case_id")
        text["failure_code"] = gateway.failure_case.get("code")
        text["failure_component"] = gateway.failure_case.get("component")
    return _content(
        GatewayResult(
            cleanup_error is None,
            text,
        )
    )


# KISS Operator surface -------------------------------------------------
# Keep the implementation-rich gateway available to host-side callers,
# while exposing a small, stable vocabulary to Codex.  These wrappers compose
# the old stateful operations so the model does not have to manage viewer
# revisions, grasp checkpoints, or pose-registration state itself.

async def _mark_point_common(
    point_id: str,
    view: str,
    x: int,
    y: int,
    *,
    label: str = "",
) -> list[Any]:
    def run() -> list[Any]:
        return _content(
            _gateway().mark_point_3d(
                view=view,
                u=int(x),
                v=int(y),
                point_id=point_id,
            ),
            public_tool="mark_point",
        )
    blocks = await anyio.to_thread.run_sync(run)
    await anyio.to_thread.run_sync(
        functools.partial(
            _record_operator_context,
            tool="mark_point",
            arguments={
                "point_id": point_id,
                "view": view,
                "x": x,
                "y": y,
                **({"label": label} if label else {}),
            },
            blocks=blocks,
        )
    )
    return blocks


async def _mark_point_legacy(
    point_id: str,
    view: str,
    x: int,
    y: int,
    label: str = "",
) -> list[Any]:
    return await _mark_point_common(
        point_id,
        view,
        x,
        y,
        label=label,
    )


async def _mark_point_compact(
    point_id: str,
    view: str,
    x: int,
    y: int,
) -> list[Any]:
    return await _mark_point_common(point_id, view, x, y)


_mark_point_impl = (
    _mark_point_compact
    if _COMPACT_INPUT_SCHEMA
    else _mark_point_legacy
)
_mark_point_impl.__name__ = "mark_point"
mark_point = mcp.tool(
    name="mark_point",
    description=tool_description("mark_point"),
    annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=True, destructiveHint=False),
    structured_output=False,
)(_mark_point_impl)


@mcp.tool(
    name="mark_vector",
    description=(
        "Draw a directed 2D vector on one calibrated point-cloud view. Call "
        "the same role on a complementary view (top+front, top+side, or "
        "front+side) to complete its world-frame 3D vector. The response "
        "returns the partial axis components, completion status, and marked "
        "view images; no point-cloud geometry or object identity is inferred."
    ),
    annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=True, destructiveHint=False),
    structured_output=False,
)
async def mark_vector(
    start_x: int,
    start_y: int,
    end_x: int,
    end_y: int,
    view: str = "pointcloud_top",
    role: str = "approach",
) -> list[Any]:
    def run() -> list[Any]:
        result = _gateway().mark_vector_3d(
            view=view,
            start_x=int(start_x),
            start_y=int(start_y),
            end_x=int(end_x),
            end_y=int(end_y),
            role=role,
        )
        return _content(result)

    blocks = await anyio.to_thread.run_sync(run)
    await anyio.to_thread.run_sync(
        functools.partial(
            _record_operator_context,
            tool="mark_vector",
            arguments={
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y,
                "view": view,
                "role": role,
            },
            blocks=blocks,
        )
    )
    return blocks


@mcp.tool(
    name="sam3",
    description="Segment the named object in the latest observation using remote SAM3.",
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, destructiveHint=False),
    structured_output=False,
)
async def sam3(target: str = "marked object") -> list[Any]:
    def run() -> list[Any]:
        gateway = _gateway()
        if not gateway._manual_marks:
            return _content(GatewayResult(False, {
                "kind": "segmentation",
                "success": False,
                "reason": "mark_required",
                "message": "Call mark_point on the current agentview before sam3.",
            }))
        # A marked point is the only SAM3 prompt. The readable target is kept
        # only as metadata for artifacts and verification.
        result = gateway.segment_object(target)
        payload = dict(result.text)
        detections = result.details.get("detections", []) if isinstance(result.details, dict) else []
        payload["detection_ids"] = [
            str(item.get("id")) for item in detections
            if isinstance(item, dict) and item.get("id") is not None
        ]
        # A text-only call with one unambiguous detection can advance directly;
        # multiple detections remain an explicit choice for propose_pose.
        if result.success and len(payload["detection_ids"]) == 1:
            selected = _gateway().select_detection(payload["detection_ids"][0], reason="only SAM3 detection")
            payload["auto_selected_detection_id"] = payload["detection_ids"][0] if selected.success else None
        return _content(GatewayResult(result.success, payload, images=result.images, details=result.details))
    blocks = await anyio.to_thread.run_sync(run)
    await anyio.to_thread.run_sync(
        functools.partial(_record_operator_context, tool="sam3", arguments={"target": target}, blocks=blocks)
    )
    return blocks


@mcp.tool(
    name="propose_pose",
    description=(
        "Select the current SAM3 target, generate grasp poses, and automatically "
        "inspect the first candidate in Viser. Returns candidate ids plus a native "
        "jaws-side grasp image."
    ),
    annotations=ToolAnnotations(readOnlyHint=True, idempotentHint=True, destructiveHint=False),
    structured_output=False,
)
async def propose_pose(detection_id: str | None = None) -> list[Any]:
    def run() -> list[Any]:
        gateway = _gateway()
        selected_detection_id = detection_id
        selection_reason = "selected for pose proposal"
        if selected_detection_id is None:
            # Keep detection selection inside the KISS wrapper.  The public
            # Operator should not need a separate state-machine tool just
            # because SAM3 returned multimask output.
            latest = gateway.latest_segmentation
            details = latest.details if latest is not None else {}
            detections = details.get("detections", []) if isinstance(details, dict) else []
            ranked: list[tuple[float, float, int, str]] = []
            for index, item in enumerate(detections if isinstance(detections, list) else []):
                if not isinstance(item, dict) or item.get("id") is None:
                    continue
                try:
                    score = float(item.get("score", 0.0))
                    area = float(item.get("area_px", 0.0))
                except (TypeError, ValueError):
                    continue
                if area <= 0.0:
                    continue
                ranked.append((score, area, -index, str(item["id"])))
            if ranked:
                selected_detection_id = max(ranked)[-1]
                selection_reason = "highest-score non-empty SAM3 detection"
        if selected_detection_id is not None:
            selected = gateway.select_detection(selected_detection_id, reason=selection_reason)
            if not selected.success:
                return _content(selected)
        proposed = gateway.propose_grasps(detection_id=selected_detection_id)
        if not proposed.success:
            return _content(proposed)
        candidate_ids = proposed.text.get("candidate_ids") or []
        if not candidate_ids:
            return _content(proposed)
        # Expose the controller-ready representation alongside the Viser image.
        # AnyGrasp candidates arrive in camera coordinates; the gateway already
        # owns the camera->world and AnyGrasp->Panda grip-site conversions.
        poses: list[dict[str, Any]] = []
        for candidate in proposed.details.get("grasp_candidates", []):
            if not isinstance(candidate, dict) or candidate.get("id") is None:
                continue
            try:
                transform = gateway._candidate_world_from_grip_site(candidate)
                rpy = _rotation_to_euler_degrees(transform[:3, :3].tolist())
            except Exception:
                continue
            if rpy is None:
                continue
            poses.append({
                "id": str(candidate["id"]),
                "frame": "world",
                "eef_frame": "panda_grip_site",
                "x": float(transform[0, 3]),
                "y": float(transform[1, 3]),
                "z": float(transform[2, 3]),
                "roll": float(rpy[0]),
                "pitch": float(rpy[1]),
                "yaw": float(rpy[2]),
                "width_m": candidate.get("width"),
                "score": candidate.get("score"),
                "rank": candidate.get("rank"),
            })
        inspected = gateway.inspect_grasp(str(candidate_ids[0]))
        merged = GatewayResult(
            # Candidate generation is the contract of this tool.  Viser is a
            # useful inspection side effect, but a viewer/service hiccup must
            # not turn a valid controller pose into a failed perception call.
            proposed.success,
            {
                **proposed.text,
                "kind": "pose_proposal",
                "success": proposed.success,
                "selected_detection_id": selected_detection_id,
                "inspected_grasp_id": str(candidate_ids[0]),
                "poses": poses,
                "inspection_success": inspected.success,
                "message": (
                    "Pose candidates generated and first candidate inspected in Viser. "
                    "Use one returned world-frame pose with move_to."
                    if inspected.success else
                    "Pose candidates generated; Viser inspection is unavailable, so use the returned world-frame pose and inspect the Chrome dashboard."
                ),
            },
            images=inspected.images,
            details={"proposal": proposed.details, "inspection": inspected.details},
        )
        pose_artifact = gateway.root / "perception" / "pose_candidates.world.json"
        pose_artifact.parent.mkdir(parents=True, exist_ok=True)
        pose_artifact.write_text(
            json.dumps({
                "observation_id": proposed.text.get("observation_id"),
                "inspected_grasp_id": str(candidate_ids[0]),
                "poses": poses,
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return _content(merged)
    blocks = await anyio.to_thread.run_sync(run)
    await anyio.to_thread.run_sync(
        functools.partial(_record_operator_context, tool="propose_pose", arguments={"detection_id": detection_id}, blocks=blocks)
    )
    return blocks


async def _move_to_common(
    target: BaseModel | Mapping[str, Any],
    *,
    reason: str = "",
    views: list[str] | None = None,
    operator_arguments: Mapping[str, Any] | None = None,
) -> list[Any]:
    target_payload = (
        target.model_dump(exclude_none=True)
        if isinstance(target, BaseModel)
        else {
            str(key): value
            for key, value in target.items()
            if value is not None
        }
    )

    def run() -> list[Any]:
        gateway = _gateway()
        moved = gateway.move_to_target(
            target_payload,
            reason=reason,
            views=views,
        )
        return _content(moved, public_tool="move_to")
    blocks = await anyio.to_thread.run_sync(run)
    await anyio.to_thread.run_sync(
        functools.partial(
            _record_operator_context,
            tool="move_to",
            arguments=(
                dict(operator_arguments)
                if operator_arguments is not None
                else {
                    "target": target_payload,
                    **({"reason": reason} if reason else {}),
                    **({"views": views} if views is not None else {}),
                }
            ),
            blocks=blocks,
        )
    )
    return blocks


async def _move_to_legacy(
    target: ActiveMoveToTarget,
    reason: str = "",
    views: list[str] | None = None,
) -> list[Any]:
    return await _move_to_common(
        target,
        reason=reason,
        views=views,
    )


async def _move_to_compact_nested(
    target: ActiveMoveToTarget,
) -> list[Any]:
    return await _move_to_common(target)


async def _move_to_compact_flat(
    xyz_m: Annotated[
        list[float] | None,
        Field(
            min_length=3,
            max_length=3,
            description="Absolute Panda grip-site [x,y,z] in world metres.",
        ),
    ] = None,
    delta_mm: Annotated[
        list[float] | None,
        Field(
            min_length=3,
            max_length=3,
            description="Panda grip-site [dX,dY,dZ] in world millimetres.",
        ),
    ] = None,
    point_id: str | None = None,
    from_point_id: str | None = None,
    to_point_id: str | None = None,
    approach_from_point_id: str | None = None,
    jaw_toward_point_id: str | None = None,
    approach_world: Annotated[
        list[float] | None,
        Field(
            min_length=3,
            max_length=3,
            description="World direction of grip-site local +Z.",
        ),
    ] = None,
    jaw_world: Annotated[
        list[float] | None,
        Field(
            min_length=3,
            max_length=3,
            description="World direction of grip-site local +X.",
        ),
    ] = None,
    gripper: Literal["open", "close"] | None = None,
    preview: bool = False,
) -> list[Any]:
    operator_arguments = {
        "xyz_m": xyz_m,
        "delta_mm": delta_mm,
        "point_id": point_id,
        "from_point_id": from_point_id,
        "to_point_id": to_point_id,
        "approach_from_point_id": approach_from_point_id,
        "jaw_toward_point_id": jaw_toward_point_id,
        "approach_world": approach_world,
        "jaw_world": jaw_world,
        "gripper": gripper,
        "preview": preview,
    }
    compact_target = {
        "position_xyz_m": xyz_m,
        "position_delta_mm": delta_mm,
        "position_point_id": point_id,
        "position_from_point_id": from_point_id,
        "position_to_point_id": to_point_id,
        "approach_from_point_id": approach_from_point_id,
        "jaw_toward_point_id": jaw_toward_point_id,
        "approach_direction_world": approach_world,
        "jaw_direction_world": jaw_world,
        "gripper": gripper,
        "preview_only": True if preview else None,
    }
    return await _move_to_common(
        compact_target,
        operator_arguments={
            key: value
            for key, value in operator_arguments.items()
            if value is not None and not (key == "preview" and value is False)
        },
    )


async def _move_to_kiss_flat(
    xyz_m: Annotated[
        list[float] | None,
        Field(
            min_length=3,
            max_length=3,
            description="Absolute Panda grip-site [x,y,z] in world metres.",
        ),
    ] = None,
    delta_mm: Annotated[
        list[float] | None,
        Field(
            min_length=3,
            max_length=3,
            description="Panda grip-site [dX,dY,dZ] in world millimetres.",
        ),
    ] = None,
    approach_world: Annotated[
        list[float] | None,
        Field(
            min_length=3,
            max_length=3,
            description="World direction of grip-site local +Z.",
        ),
    ] = None,
    jaw_world: Annotated[
        list[float] | None,
        Field(
            min_length=3,
            max_length=3,
            description="World direction of grip-site local +X.",
        ),
    ] = None,
    gripper: Literal["open", "close"] | None = None,
    preview: bool = False,
) -> list[Any]:
    operator_arguments = {
        "xyz_m": xyz_m,
        "delta_mm": delta_mm,
        "approach_world": approach_world,
        "jaw_world": jaw_world,
        "gripper": gripper,
        "preview": preview,
    }
    compact_target = {
        "position_xyz_m": xyz_m,
        "position_delta_mm": delta_mm,
        "approach_direction_world": approach_world,
        "jaw_direction_world": jaw_world,
        "gripper": gripper,
        "preview_only": True if preview else None,
    }
    return await _move_to_common(
        compact_target,
        operator_arguments={
            key: value
            for key, value in operator_arguments.items()
            if value is not None and not (key == "preview" and value is False)
        },
    )


async def _move_to_kiss_flat_v2(
    xyz_m: Annotated[
        list[float] | None,
        Field(
            min_length=3,
            max_length=3,
            description="Absolute Panda grip-site [x,y,z] in world metres.",
        ),
    ] = None,
    position_point_id: Annotated[
        str | None,
        Field(
            description=(
                "Solved mark_point ID used as the exact Panda grip-site "
                "destination."
            ),
        ),
    ] = None,
    delta_mm: Annotated[
        list[float] | None,
        Field(
            min_length=3,
            max_length=3,
            description="Panda grip-site [dX,dY,dZ] in world millimetres.",
        ),
    ] = None,
    approach_world: Annotated[
        list[float] | None,
        Field(
            min_length=3,
            max_length=3,
            description="World direction of grip-site local +Z.",
        ),
    ] = None,
    jaw_world: Annotated[
        list[float] | None,
        Field(
            min_length=3,
            max_length=3,
            description="World direction of grip-site local +X.",
        ),
    ] = None,
    gripper: Literal["open", "close"] | None = None,
    preview: bool = False,
) -> list[Any]:
    operator_arguments = {
        "xyz_m": xyz_m,
        "position_point_id": position_point_id,
        "delta_mm": delta_mm,
        "approach_world": approach_world,
        "jaw_world": jaw_world,
        "gripper": gripper,
        "preview": preview,
    }
    compact_target = {
        "position_xyz_m": xyz_m,
        "position_point_id": position_point_id,
        "position_delta_mm": delta_mm,
        "approach_direction_world": approach_world,
        "jaw_direction_world": jaw_world,
        "gripper": gripper,
        "preview_only": True if preview else None,
    }
    return await _move_to_common(
        compact_target,
        operator_arguments={
            key: value
            for key, value in operator_arguments.items()
            if value is not None and not (key == "preview" and value is False)
        },
    )


async def _move_to_kiss_flat_v3(
    xyz_m: Annotated[
        list[float] | None,
        Field(
            min_length=3,
            max_length=3,
            description="Absolute Panda grip-site [x,y,z] in world metres.",
        ),
    ] = None,
    delta_mm: Annotated[
        list[float] | None,
        Field(
            min_length=3,
            max_length=3,
            description=(
                "Panda grip-site displacement in millimetres. Components use "
                "delta_frame."
            ),
        ),
    ] = None,
    delta_frame: Annotated[
        Literal["world", "grip_site"],
        Field(
            description=(
                "world means [dX,dY,dZ]. grip_site means "
                "[dJAW,dLAT,dAPP] in the actual grip-site axes at call start."
            ),
        ),
    ] = "world",
    approach_world: Annotated[
        list[float] | None,
        Field(
            min_length=3,
            max_length=3,
            description="World direction of grip-site local +Z.",
        ),
    ] = None,
    jaw_world: Annotated[
        list[float] | None,
        Field(
            min_length=3,
            max_length=3,
            description="World direction of grip-site local +X.",
        ),
    ] = None,
    gripper: Literal["open", "close"] | None = None,
    preview: bool = False,
) -> list[Any]:
    operator_arguments = {
        "xyz_m": xyz_m,
        "delta_mm": delta_mm,
        "delta_frame": delta_frame,
        "approach_world": approach_world,
        "jaw_world": jaw_world,
        "gripper": gripper,
        "preview": preview,
    }
    compact_target = {
        "position_xyz_m": xyz_m,
        "position_delta_mm": delta_mm,
        "delta_frame": delta_frame if delta_mm is not None else None,
        "approach_direction_world": approach_world,
        "jaw_direction_world": jaw_world,
        "gripper": gripper,
        "preview_only": True if preview else None,
    }
    return await _move_to_common(
        compact_target,
        operator_arguments={
            key: value
            for key, value in operator_arguments.items()
            if value is not None
            and not (key == "preview" and value is False)
            and not (
                key == "delta_frame"
                and value == "world"
                and delta_mm is None
            )
        },
    )


async def _move_to_kiss_flat_v4(
    xyz_m: Annotated[
        list[float] | None,
        Field(
            min_length=3,
            max_length=3,
            description="Absolute Panda grip-site [x,y,z] in world metres.",
        ),
    ] = None,
    delta_mm: Annotated[
        list[float] | None,
        Field(
            min_length=3,
            max_length=3,
            description=(
                "Panda grip-site displacement in millimetres. Components use "
                "delta_frame."
            ),
        ),
    ] = None,
    delta_frame: Annotated[
        Literal["world", "grip_site"],
        Field(
            description=(
                "world means [dX,dY,dZ]. grip_site means "
                "[dJAW,dLAT,dAPP] in the actual grip-site axes at call start."
            ),
        ),
    ] = "world",
    approach_world: Annotated[
        list[float] | None,
        Field(
            min_length=3,
            max_length=3,
            description="World direction of grip-site local +Z.",
        ),
    ] = None,
    jaw_world: Annotated[
        list[float] | None,
        Field(
            min_length=3,
            max_length=3,
            description="World direction of grip-site local +X.",
        ),
    ] = None,
    gripper: Literal["open", "close"] | None = None,
    preview: bool = False,
    execute_preview_id: Annotated[
        str | None,
        Field(
            description=(
                "Execute exactly one frozen close preview. Use alone with the "
                "preview_id returned by move_to."
            ),
        ),
    ] = None,
) -> list[Any]:
    operator_arguments = {
        "xyz_m": xyz_m,
        "delta_mm": delta_mm,
        "delta_frame": delta_frame,
        "approach_world": approach_world,
        "jaw_world": jaw_world,
        "gripper": gripper,
        "preview": preview,
        "execute_preview_id": execute_preview_id,
    }
    compact_target = {
        "position_xyz_m": xyz_m,
        "position_delta_mm": delta_mm,
        "delta_frame": delta_frame if delta_mm is not None else None,
        "approach_direction_world": approach_world,
        "jaw_direction_world": jaw_world,
        "gripper": gripper,
        "preview_only": True if preview else None,
        "execute_preview_id": execute_preview_id,
    }
    return await _move_to_common(
        compact_target,
        operator_arguments={
            key: value
            for key, value in operator_arguments.items()
            if value is not None
            and not (key == "preview" and value is False)
            and not (
                key == "delta_frame"
                and value == "world"
                and delta_mm is None
            )
        },
    )


_move_to_impl = (
    _move_to_kiss_flat_v4
    if _KISS_MOVE_TO_INPUT_V4
    else _move_to_kiss_flat_v3
    if _KISS_MOVE_TO_INPUT_V3
    else _move_to_kiss_flat_v2
    if _KISS_MOVE_TO_INPUT_V2
    else _move_to_kiss_flat
    if _KISS_MOVE_TO_INPUT
    else _move_to_compact_flat
    if _FLAT_MOVE_TO_INPUT
    else _move_to_compact_nested
    if _COMPACT_INPUT_SCHEMA
    else _move_to_legacy
)
_move_to_impl.__name__ = "move_to"
move_to = mcp.tool(
    name="move_to",
    description=tool_description("move_to"),
    annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=False, destructiveHint=True),
    structured_output=False,
)(_move_to_impl)


@_operator_tool(
    description=(
        "Open or close the Panda gripper as an independent atomic world "
        "action, decoupled from move_to. Call it before or after a motion as "
        "its own step; it returns the measured aperture verification."
    ),
    annotations=ToolAnnotations(readOnlyHint=False, idempotentHint=False, destructiveHint=True),
)
def set_gripper(action: Literal["open", "close"]) -> list[Any]:
    return _content(
        _gateway().step("open_gripper" if action == "open" else "close_gripper")
    )


# Vectors and an independent gripper verb are no longer part of the public
# Operator vocabulary. Host-side gateway methods remain available for
# historical replay and diagnostics.
for _removed_public_tool in ("mark_vector", "set_gripper"):
    try:
        mcp._tool_manager.remove_tool(_removed_public_tool)
    except Exception:
        pass
if "public_operator_tools" in _ACTIVE_PROFILE.manifest:
    _allowed_operator_tools = set(_ACTIVE_PROFILE.public_operator_tools)
    for _registered_tool in list(mcp._tool_manager.list_tools()):
        if _registered_tool.name not in _allowed_operator_tools:
            try:
                mcp._tool_manager.remove_tool(_registered_tool.name)
            except Exception:
                pass


def _shutdown() -> None:
    global _CONTROL_SERVER
    if _CONTROL_SERVER is not None:
        try:
            _CONTROL_SERVER.shutdown()
            _CONTROL_SERVER.server_close()
        except Exception:
            pass
        _CONTROL_SERVER = None
    if _GATEWAY is not None:
        try:
            _GATEWAY.close()
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    global _GATEWAY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="fresh episode artifact root")
    parser.add_argument("--env-id", default="openeta/libero_libero_spatial_task0-v0")
    parser.add_argument("--task", default="pick up the black bowl between the plate and the ramekin and place it on the plate")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--image-height", type=int, default=256)
    parser.add_argument("--sim-url", default="http://127.0.0.1:8765/sse")
    parser.add_argument("--sam3-url", default="http://127.0.0.1:8773/sse")
    parser.add_argument("--anygrasp-url", default="http://127.0.0.1:8774/sse")
    parser.add_argument("--grasp-inspector-url", default="http://127.0.0.1:8082")
    parser.add_argument(
        "--transport",
        choices=("stdio", "sse", "streamable-http"),
        default="stdio",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8780)
    parser.add_argument("--control-host", default="127.0.0.1")
    parser.add_argument("--control-port", type=int, default=8790)
    args = parser.parse_args(argv)

    if __import__("os").environ.get("OPENETA_POINT_ONLY_OPERATOR") == "1":
        for legacy_name in (
            "mark_vector",
            "sam3",
            "propose_pose",
            "set_gripper",
            "verify_grasp",
        ):
            try:
                mcp._tool_manager.remove_tool(legacy_name)
            except Exception:
                pass
    contract_path = args.root / "operator_context_contract.json"
    point_only_operator = (
        __import__("os").environ.get("OPENETA_POINT_ONLY_OPERATOR") == "1"
    )
    if point_only_operator and not contract_path.is_file():
        raise FileNotFoundError(
            "Point-only Operator startup requires a versioned "
            f"context contract, but none exists at {contract_path}"
        )
    if contract_path.is_file():
        try:
            finalize_contract(contract_path, mcp._tool_manager.list_tools())
        except Exception as exc:
            record_contract_resolution_failure(contract_path, exc)
            raise RuntimeError(
                "Operator context contract resolution failed; startup aborted. "
                f"See {contract_path}"
            ) from exc

    _GATEWAY = EmbodiedGateway(
        root=args.root,
        env_id=args.env_id,
        task=args.task,
        seed=args.seed,
        image_width=args.image_width,
        image_height=args.image_height,
        include_objects=(
            __import__("os").environ.get(
                "OPENETA_BUILDER_OBJECT_DIAGNOSTICS", "0"
            )
            == "1"
        ),
        simulator_url=args.sim_url,
        sam3_url=args.sam3_url,
        anygrasp_url=args.anygrasp_url,
        grasp_inspector_url=args.grasp_inspector_url,
    )
    atexit.register(_shutdown)
    mcp.settings.host = args.host
    mcp.settings.port = args.port
    global _CONTROL_SERVER
    _CONTROL_SERVER = ThreadingHTTPServer((args.control_host, args.control_port), _ManualControlHandler)
    control_thread = threading.Thread(target=_CONTROL_SERVER.serve_forever, daemon=True)
    control_thread.start()
    mcp.run(transport=args.transport)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
