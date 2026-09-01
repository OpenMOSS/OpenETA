"""Safe/tool/skill command pipeline for OpenETA agent decisions."""

from __future__ import annotations

from functools import lru_cache

from adapter.protocol import EnvObservation, JsonDict
from agent.runtime.actions import (
    CommandKind,
    CommandPipelinePlan,
    CommandRequest,
    PipelineCall,
    PipelineStatus,
)
from agent.runtime.checkers import (
    CheckerSubagentConfig,
    build_failure_check_call,
    safety_check_parameters,
)
from agent.runtime.interfaces import ActionInterfaceRegistry, build_default_action_interfaces
from agent.runtime.memory import AgentMemory
from agent.runtime.planner import PlannerDecision
from agent.runtime.skills import SkillRegistry
from agent.tools.runtime_contract_bindings import (
    HostResolutionFailure,
    resolve_host_parameters,
)
from agent.tools.registry import ToolRegistry, ToolResult
from agent.tools.contracts import ToolContractCatalog, ToolContractRuntimePolicy


class ActionPipeline:
    """Compile planner decisions into structured `EnvAction.command` payloads."""

    def __init__(
        self,
        *,
        execute_safe_checks: bool = True,
        checker_subagents: CheckerSubagentConfig | None = None,
        interfaces: ActionInterfaceRegistry | None = None,
        tool_contract_catalog: ToolContractCatalog | None = None,
        tool_contract_policy: ToolContractRuntimePolicy | None = None,
    ) -> None:
        self.execute_safe_checks = execute_safe_checks
        self.checker_subagents = checker_subagents or CheckerSubagentConfig()
        self.interfaces = interfaces or build_default_action_interfaces()
        self.tool_contract_catalog = (
            tool_contract_catalog or _default_gate_contract_catalog()
        )
        self.tool_contract_policy = tool_contract_policy or ToolContractRuntimePolicy()
        self.tool_contract_policy.ensure_valid(self.tool_contract_catalog)

    def _gate_repair_bundle(
        self,
        memory: AgentMemory | None,
        *,
        code: str,
        reason: str,
        request: CommandRequest,
        checker_calls: list[PipelineCall] | None = None,
    ) -> JsonDict:
        return _gate_repair_bundle(
            memory,
            code=code,
            reason=reason,
            request=request,
            checker_calls=checker_calls,
            tool_contract_catalog=self.tool_contract_catalog,
            tool_contract_policy=self.tool_contract_policy,
        )

    def compile(
        self,
        decision: PlannerDecision,
        *,
        observation: EnvObservation,
        tools: ToolRegistry,
        skills: SkillRegistry,
        memory: AgentMemory | None = None,
    ) -> CommandPipelinePlan:
        request = _decision_to_request(decision)

        if request.kind == CommandKind.TOOL_CALL:
            if _is_skill_call_request(request):
                return self._compile_skill_call(
                    request,
                    observation=observation,
                    tools=tools,
                    skills=skills,
                    planner_metadata=decision.metadata,
                )
            if _is_safety_check_request(request):
                safe_name = _named_tool_target(request)
                safe_call = self._compile_safety_check(
                    safe_name,
                    request.parameters,
                    tools=tools,
                    observation=observation,
                    reason="Planner-requested safety check.",
                )
                return CommandPipelinePlan(
                    request=request,
                    status=_aggregate_status([safe_call]),
                    safety_checks=[safe_call],
                    metadata={
                        "interface": self.interfaces.descriptor(request.kind, request.name),
                        "planner_metadata": decision.metadata,
                    },
                )
            if _is_direct_tool_like_request(request):
                interface_descriptor = self.interfaces.descriptor(
                    request.kind,
                    request.name,
                )
                return CommandPipelinePlan(
                    request=request,
                    status=_direct_request_status(
                        request.kind,
                        request.name,
                        interface_descriptor,
                    ),
                    metadata={
                        "interface": interface_descriptor,
                        "observation_step": observation.metadata.get("step_idx"),
                        "planner_metadata": decision.metadata,
                    },
                )
            if _is_tool_batch_request(request):
                return self._compile_tool_batch(
                    request,
                    tools=tools,
                    memory=memory,
                    planner_metadata=decision.metadata,
                )

            resolved_parameters = request.parameters
            bundle_resolution: JsonDict | None = None
            bundle_kind = ""
            host_resolution_receipt: JsonDict | None = None
            # Unknown mutation outcome outranks every parameter/reference check:
            # no pose evidence can be interpreted until the same handle is
            # observed and reconciled.
            execution_gate_error = (
                memory.motion_reconciliation_gate_error(tool_name=request.name)
                if memory is not None
                else None
            )
            if execution_gate_error:
                tool_call = _skipped_tool_call(
                    request.name,
                    request.parameters,
                    reason=execution_gate_error,
                )
                return CommandPipelinePlan(
                    request=request,
                    status=PipelineStatus.BLOCKED,
                    tool_calls=[tool_call],
                    metadata={
                        "interface": self.interfaces.descriptor(
                            request.kind, request.name
                        ),
                        "planner_metadata": decision.metadata,
                        "execution_rule": _tool_execution_rule(tool_call, tools),
                        "motion_reconciliation_gate": {
                            "blocked": True,
                            "reason": execution_gate_error,
                        },
                        "repair_bundle": self._gate_repair_bundle(
                            memory,
                            code="motion_reconciliation_required",
                            reason=execution_gate_error,
                            request=request,
                        ),
                    },
                )
            reference_kind = ""
            if request.name == "ik_preview_check" and isinstance(
                request.parameters.get("probe_id"), str
            ):
                reference_kind = "attachment_probe_waypoint"
                try:
                    if memory is None:
                        raise ValueError("runtime memory is unavailable")
                    conflicting = sorted(
                        key
                        for key in (
                            "target_pose",
                            "compiled_grasp_id",
                            "waypoint_role",
                            "viewpoint_proposal_id",
                            "candidate_id",
                            "preserve_current_orientation",
                        )
                        if key in request.parameters
                    )
                    if conflicting:
                        raise ValueError(
                            "provide probe_id + waypoint_index without other target "
                            f"source or orientation-policy fields; conflicting={conflicting!r}"
                        )
                    resolution = memory.resolve_attachment_probe_waypoint(
                        probe_id=str(request.parameters.get("probe_id") or ""),
                        waypoint_index=request.parameters.get("waypoint_index"),
                    )
                    resolved = resolution.get("parameters")
                    if not isinstance(resolved, dict):
                        raise ValueError(
                            "attachment probe waypoint resolver returned invalid parameters"
                        )
                    resolved_parameters = {
                        **resolved,
                        **{
                            key: request.parameters[key]
                            for key in (
                                "position_tolerance_m",
                                "orientation_tolerance_rad",
                                "check_endpoint_collision",
                            )
                            if key in request.parameters
                        },
                    }
                    bundle_resolution = resolution
                except (TypeError, ValueError) as exc:
                    reason = f"ik_preview_check probe reference resolution failed: {exc}"
                    tool_call = _skipped_tool_call(
                        request.name,
                        request.parameters,
                        reason=reason,
                    )
                    return CommandPipelinePlan(
                        request=request,
                        status=PipelineStatus.BLOCKED,
                        tool_calls=[tool_call],
                        metadata={
                            "interface": self.interfaces.descriptor(
                                request.kind, request.name
                            ),
                            "planner_metadata": decision.metadata,
                            "execution_rule": _tool_execution_rule(tool_call, tools),
                            "reference_resolution_gate": {
                                "blocked": True,
                                "reference_kind": reference_kind,
                                "reason": str(exc),
                            },
                            "repair_bundle": self._gate_repair_bundle(
                                memory,
                                code="invalid_attachment_probe_reference",
                                reason=reason,
                                request=request,
                            ),
                        },
                    )
            elif (
                request.name == "ik_preview_check"
                and isinstance(request.parameters.get("compiled_grasp_id"), str)
                and "path_fraction" in request.parameters
            ):
                reference_kind = "compiled_grasp_path_sample"
                try:
                    if memory is None:
                        raise ValueError("runtime memory is unavailable")
                    if "target_pose" in request.parameters or "waypoint_role" in request.parameters:
                        raise ValueError(
                            "provide compiled_grasp_id + path_fraction without target_pose "
                            "or waypoint_role; the host resolves the path sample"
                        )
                    if request.parameters.get("preserve_current_orientation") is True:
                        raise ValueError(
                            "compiled path samples preserve the grasp's explicit full "
                            "orientation; preserve_current_orientation is not compatible"
                        )
                    resolution = memory.resolve_compiled_grasp_path_sample_reference(
                        compiled_grasp_id=str(
                            request.parameters.get("compiled_grasp_id") or ""
                        ),
                        path_fraction=request.parameters.get("path_fraction"),
                    )
                    resolved = resolution.get("parameters")
                    if not isinstance(resolved, dict):
                        raise ValueError(
                            "compiled grasp path-sample resolver returned invalid parameters"
                        )
                    resolved_parameters = {
                        **resolved,
                        **{
                            key: request.parameters[key]
                            for key in (
                                "position_tolerance_m",
                                "orientation_tolerance_rad",
                                "check_endpoint_collision",
                            )
                            if key in request.parameters
                        },
                    }
                    bundle_resolution = resolution
                except (TypeError, ValueError) as exc:
                    reason = f"ik_preview_check path sample resolution failed: {exc}"
                    tool_call = _skipped_tool_call(
                        request.name,
                        request.parameters,
                        reason=reason,
                    )
                    return CommandPipelinePlan(
                        request=request,
                        status=PipelineStatus.BLOCKED,
                        tool_calls=[tool_call],
                        metadata={
                            "interface": self.interfaces.descriptor(
                                request.kind, request.name
                            ),
                            "planner_metadata": decision.metadata,
                            "execution_rule": _tool_execution_rule(tool_call, tools),
                            "reference_resolution_gate": {
                                "blocked": True,
                                "reference_kind": reference_kind,
                                "reason": str(exc),
                            },
                            "repair_bundle": self._gate_repair_bundle(
                                memory,
                                code="invalid_compiled_grasp_path_sample",
                                reason=reason,
                                request=request,
                            ),
                        },
                    )
            elif request.name == "ik_preview_check" and isinstance(
                request.parameters.get("compiled_grasp_id"), str
            ):
                reference_kind = "compiled_grasp_pose"
                try:
                    if memory is None:
                        raise ValueError("runtime memory is unavailable")
                    if "target_pose" in request.parameters:
                        raise ValueError(
                            "provide compiled_grasp_id + waypoint_role without target_pose; "
                            "the host resolves the immutable pose"
                        )
                    resolution = memory.resolve_compiled_grasp_pose_reference(
                        compiled_grasp_id=str(
                            request.parameters.get("compiled_grasp_id") or ""
                        ),
                        waypoint_role=str(
                            request.parameters.get("waypoint_role") or ""
                        ),
                    )
                    resolved = resolution.get("parameters")
                    if not isinstance(resolved, dict):
                        raise ValueError(
                            "compiled grasp pose resolver returned invalid parameters"
                        )
                    resolved_parameters = {
                        **resolved,
                        **{
                            key: request.parameters[key]
                            for key in (
                                "position_tolerance_m",
                                "orientation_tolerance_rad",
                                "preserve_current_orientation",
                                "check_endpoint_collision",
                            )
                            if key in request.parameters
                        },
                    }
                    if request.parameters.get("preserve_current_orientation") is True:
                        # A compiled waypoint carries its grasp orientation as part of
                        # the immutable anchor.  The Agent may deliberately request a
                        # position-only preview of the same xyz for a non-contact
                        # observation or retreat.  Preserve the anchor provenance, but
                        # do not let its embedded rotation silently override that
                        # explicitly selected orientation policy downstream.
                        target_pose = resolved_parameters.get("target_pose")
                        if isinstance(target_pose, dict):
                            resolved_parameters["target_pose"] = {
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
                                    "euler_xyz_deg",
                                }
                            }
                    bundle_resolution = resolution
                except ValueError as exc:
                    reason = f"ik_preview_check reference resolution failed: {exc}"
                    tool_call = _skipped_tool_call(
                        request.name,
                        request.parameters,
                        reason=reason,
                    )
                    return CommandPipelinePlan(
                        request=request,
                        status=PipelineStatus.BLOCKED,
                        tool_calls=[tool_call],
                        metadata={
                            "interface": self.interfaces.descriptor(
                                request.kind, request.name
                            ),
                            "planner_metadata": decision.metadata,
                            "execution_rule": _tool_execution_rule(tool_call, tools),
                            "reference_resolution_gate": {
                                "blocked": True,
                                "reference_kind": reference_kind,
                                "reason": str(exc),
                            },
                            "repair_bundle": self._gate_repair_bundle(
                                memory,
                                code="invalid_compiled_grasp_reference",
                                reason=reason,
                                request=request,
                            ),
                        },
                    )
            elif (
                request.name == "move_to"
                and not (
                    isinstance(request.parameters.get("target_pose"), dict)
                    and self.execute_safe_checks
                    and self.checker_subagents.pre_safety_checks.get("move_to")
                    == "ik_preview_check"
                )
            ):
                reference_kind = "ik_receipt"
                try:
                    if memory is None:
                        raise ValueError("runtime memory is unavailable")
                    if "target_pose" in request.parameters:
                        raise ValueError(
                            "move_to no longer accepts model-copied target_pose; pass "
                            "the exact ik_receipt_id returned by ik_preview_check"
                        )
                    resolution = memory.resolve_ik_motion_reference(
                        str(request.parameters.get("ik_receipt_id") or "")
                    )
                    resolved = resolution.get("parameters")
                    if not isinstance(resolved, dict):
                        raise ValueError("IK receipt resolver returned invalid parameters")
                    resolved_parameters = {
                        **resolved,
                        **{
                            key: request.parameters[key]
                            for key in (
                                "num_steps",
                                "tolerance",
                                "ori_tolerance",
                                "enable_collision_check",
                            )
                            if key in request.parameters
                        },
                    }
                    bundle_resolution = resolution
                except ValueError as exc:
                    reason = f"move_to IK receipt resolution failed: {exc}"
                    tool_call = _skipped_tool_call(
                        request.name,
                        request.parameters,
                        reason=reason,
                    )
                    return CommandPipelinePlan(
                        request=request,
                        status=PipelineStatus.BLOCKED,
                        tool_calls=[tool_call],
                        metadata={
                            "interface": self.interfaces.descriptor(
                                request.kind, request.name
                            ),
                            "planner_metadata": decision.metadata,
                            "execution_rule": _tool_execution_rule(tool_call, tools),
                            "reference_resolution_gate": {
                                "blocked": True,
                                "reference_kind": reference_kind,
                                "reason": str(exc),
                            },
                            "repair_bundle": self._gate_repair_bundle(
                                memory,
                                code="invalid_ik_receipt_reference",
                                reason=reason,
                                request=request,
                            ),
                        },
                    )
            elif request.name == "follow_eef_trajectory":
                reference_kind = "ik_trajectory_receipts"
                try:
                    if memory is None:
                        raise ValueError("runtime memory is unavailable")
                    if "trajectory" in request.parameters:
                        raise ValueError(
                            "follow_eef_trajectory no longer accepts a model-copied "
                            "trajectory; pass ordered ik_receipt_ids returned by "
                            "ik_preview_check"
                        )
                    resolution = memory.resolve_ik_trajectory_reference(
                        request.parameters.get("ik_receipt_ids")
                    )
                    resolved = resolution.get("parameters")
                    if not isinstance(resolved, dict):
                        raise ValueError(
                            "IK trajectory resolver returned invalid parameters"
                        )
                    resolved_parameters = {
                        **resolved,
                        **{
                            key: request.parameters[key]
                            for key in (
                                "num_steps_per_waypoint",
                                "tolerance",
                                "ori_tolerance",
                                "enable_collision_check",
                            )
                            if key in request.parameters
                        },
                    }
                    bundle_resolution = resolution
                except ValueError as exc:
                    reason = (
                        "follow_eef_trajectory IK receipt resolution failed: "
                        f"{exc}"
                    )
                    tool_call = _skipped_tool_call(
                        request.name,
                        request.parameters,
                        reason=reason,
                    )
                    return CommandPipelinePlan(
                        request=request,
                        status=PipelineStatus.BLOCKED,
                        tool_calls=[tool_call],
                        metadata={
                            "interface": self.interfaces.descriptor(
                                request.kind, request.name
                            ),
                            "planner_metadata": decision.metadata,
                            "execution_rule": _tool_execution_rule(tool_call, tools),
                            "reference_resolution_gate": {
                                "blocked": True,
                                "reference_kind": reference_kind,
                                "reason": str(exc),
                            },
                            "repair_bundle": self._gate_repair_bundle(
                                memory,
                                code="invalid_ik_trajectory_reference",
                                reason=reason,
                                request=request,
                            ),
                        },
                    )
            if request.name in {
                "retrieve_asset_reference",
                "sam3",
                "molmopoint",
                "estimate_depth_prior",
                "enhance_depth",
            }:
                try:
                    resolution_contract = self.tool_contract_catalog.get(
                        request.name
                    ).host_resolution
                    resolution_result = resolve_host_parameters(
                        tool_name=request.name,
                        resolver_id=resolution_contract.resolver,
                        parameters=request.parameters,
                        memory=memory,
                    )
                    resolved_parameters = resolution_result.parameters
                    host_resolution_receipt = {
                        "schema_version": "openeta.host_resolution_receipt.v1",
                        "status": "resolved",
                        "tool": request.name,
                        "resolver_id": resolution_contract.resolver,
                        "implementation": resolution_contract.implementation,
                        "dispatch_authority": "tool_contract",
                        "public_parameter_keys": sorted(request.parameters),
                        "resolved_parameter_keys": sorted(resolved_parameters),
                    }
                except HostResolutionFailure as exc:
                    reason = f"{request.name} source packet resolution failed: {exc}"
                    tool_call = _skipped_tool_call(
                        request.name,
                        request.parameters,
                        reason=reason,
                    )
                    return CommandPipelinePlan(
                        request=request,
                        status=PipelineStatus.BLOCKED,
                        tool_calls=[tool_call],
                        metadata={
                            "interface": self.interfaces.descriptor(
                                request.kind, request.name
                            ),
                            "planner_metadata": decision.metadata,
                            "execution_rule": _tool_execution_rule(tool_call, tools),
                            "source_packet_gate": {
                                "blocked": True,
                                "reason": str(exc),
                            },
                            "host_resolution_receipt": {
                                "schema_version": (
                                    "openeta.host_resolution_receipt.v1"
                                ),
                                "status": "rejected",
                                "tool": request.name,
                                "resolver_id": resolution_contract.resolver,
                                "implementation": (
                                    resolution_contract.implementation
                                ),
                                "dispatch_authority": "tool_contract",
                                "repair_code": exc.repair_code,
                                "public_parameter_keys": sorted(
                                    request.parameters
                                ),
                            },
                            "repair_bundle": self._gate_repair_bundle(
                                memory,
                                code=exc.repair_code,
                                reason=reason,
                                request=request,
                            ),
                        },
                    )
            if (
                request.name
                in {
                    "anyplace",
                    "camera_pose_to_world",
                    "compile_grasp_seed",
                    "compute_wrist_alignment",
                    "propose_wrist_viewpoints",
                }
                and (
                    request.name != "camera_pose_to_world"
                    or isinstance(
                        request.parameters.get("placement_result_id"), str
                    )
                )
            ) or (
                request.name == "grasp_pose_estimate"
                and isinstance(request.parameters.get("bundle_id"), str)
            ) or (
                request.name == "ik_preview_check"
                and isinstance(
                    request.parameters.get("viewpoint_proposal_id"), str
                )
            ):
                bundle_id = str(request.parameters.get("bundle_id") or "").strip()
                bundle_kind = request.name
                try:
                    if bundle_kind == "ik_preview_check":
                        if memory is None:
                            raise ValueError("runtime memory is unavailable")
                        bundle_resolution = memory.resolve_wrist_viewpoint_candidate(
                            proposal_id=str(
                                request.parameters.get("viewpoint_proposal_id") or ""
                            ),
                            candidate_id=str(
                                request.parameters.get("candidate_id") or ""
                            ),
                        )
                        resolved = bundle_resolution.get("parameters")
                        if not isinstance(resolved, dict):
                            raise ValueError(
                                "provenance resolver returned invalid parameters"
                            )
                        resolved_parameters = {
                            **resolved,
                            **{
                                key: request.parameters[key]
                                for key in (
                                    "position_tolerance_m",
                                    "orientation_tolerance_rad",
                                    "check_endpoint_collision",
                                )
                                if key in request.parameters
                            },
                            "viewpoint_proposal_id": request.parameters.get(
                                "viewpoint_proposal_id"
                            ),
                            "candidate_id": request.parameters.get("candidate_id"),
                        }
                    else:
                        resolution_contract = self.tool_contract_catalog.get(
                            request.name
                        ).host_resolution
                        resolution_result = resolve_host_parameters(
                            tool_name=request.name,
                            resolver_id=resolution_contract.resolver,
                            parameters=request.parameters,
                            memory=memory,
                        )
                        resolved_parameters = resolution_result.parameters
                        bundle_resolution = resolution_result.evidence
                        host_resolution_receipt = {
                            "schema_version": "openeta.host_resolution_receipt.v1",
                            "status": "resolved",
                            "tool": request.name,
                            "resolver_id": resolution_contract.resolver,
                            "implementation": resolution_contract.implementation,
                            "dispatch_authority": "tool_contract",
                            "public_parameter_keys": sorted(request.parameters),
                            "resolved_parameter_keys": sorted(resolved_parameters),
                        }
                except (HostResolutionFailure, ValueError) as exc:
                    repair_code = (
                        exc.repair_code
                        if isinstance(exc, HostResolutionFailure)
                        else "invalid_provenance_bundle"
                    )
                    reason = f"{request.name} provenance bundle resolution failed: {exc}"
                    tool_call = _skipped_tool_call(
                        request.name,
                        request.parameters,
                        reason=reason,
                    )
                    return CommandPipelinePlan(
                        request=request,
                        status=PipelineStatus.BLOCKED,
                        tool_calls=[tool_call],
                        metadata={
                            "interface": self.interfaces.descriptor(
                                request.kind, request.name
                            ),
                            "planner_metadata": decision.metadata,
                            "execution_rule": _tool_execution_rule(tool_call, tools),
                            "provenance_bundle_gate": {
                                "blocked": True,
                                "bundle_kind": bundle_kind,
                                "bundle_id": bundle_id or None,
                                "reason": str(exc),
                            },
                            **(
                                {
                                    "host_resolution_receipt": {
                                        "schema_version": (
                                            "openeta.host_resolution_receipt.v1"
                                        ),
                                        "status": "rejected",
                                        "tool": request.name,
                                        "resolver_id": resolution_contract.resolver,
                                        "implementation": (
                                            resolution_contract.implementation
                                        ),
                                        "dispatch_authority": "tool_contract",
                                        "repair_code": repair_code,
                                        "public_parameter_keys": sorted(
                                            request.parameters
                                        ),
                                    }
                                }
                                if isinstance(exc, HostResolutionFailure)
                                else {}
                            ),
                            "repair_bundle": self._gate_repair_bundle(
                                memory,
                                code=repair_code,
                                reason=reason,
                                request=request,
                            ),
                        },
                    )

            selection_gate_error = _detection_selection_gate_error(
                request,
                memory=memory,
            )
            if selection_gate_error:
                tool_call = _skipped_tool_call(
                    request.name,
                    request.parameters,
                    reason=selection_gate_error,
                )
                return CommandPipelinePlan(
                    request=request,
                    status=PipelineStatus.BLOCKED,
                    tool_calls=[tool_call],
                    metadata={
                        "interface": self.interfaces.descriptor(request.kind, request.name),
                        "planner_metadata": decision.metadata,
                        "execution_rule": _tool_execution_rule(tool_call, tools),
                        "selection_gate": {
                            "blocked": True,
                            "reason": selection_gate_error,
                        },
                        "repair_bundle": self._gate_repair_bundle(
                            memory,
                            code="perception_provenance_integrity",
                            reason=selection_gate_error,
                            request=request,
                        ),
                    },
                )

            provenance_gate_error = (
                memory.compiled_grasp_target_gate_error(
                    tool_name=request.name,
                    parameters=resolved_parameters,
                )
                if memory is not None
                else None
            )
            if provenance_gate_error:
                tool_call = _skipped_tool_call(
                    request.name,
                    resolved_parameters,
                    reason=provenance_gate_error,
                )
                return CommandPipelinePlan(
                    request=request,
                    status=PipelineStatus.BLOCKED,
                    tool_calls=[tool_call],
                    metadata={
                        "interface": self.interfaces.descriptor(request.kind, request.name),
                        "planner_metadata": decision.metadata,
                        "execution_rule": _tool_execution_rule(tool_call, tools),
                        "provenance_integrity_gate": {
                            "blocked": True,
                            "reason": provenance_gate_error,
                        },
                        "repair_bundle": self._gate_repair_bundle(
                            memory,
                            code=_compiled_grasp_gate_code(provenance_gate_error),
                            reason=provenance_gate_error,
                            request=request,
                        ),
                    },
                )

            probe_gate_error = (
                memory.articulated_probe_action_gate_error(
                    tool_name=request.name,
                    parameters=resolved_parameters,
                )
                if memory is not None
                else None
            )
            if probe_gate_error:
                tool_call = _skipped_tool_call(
                    request.name,
                    resolved_parameters,
                    reason=probe_gate_error,
                )
                return CommandPipelinePlan(
                    request=request,
                    status=PipelineStatus.BLOCKED,
                    tool_calls=[tool_call],
                    metadata={
                        "interface": self.interfaces.descriptor(request.kind, request.name),
                        "planner_metadata": decision.metadata,
                        "execution_rule": _tool_execution_rule(tool_call, tools),
                        "articulated_probe_gate": {
                            "blocked": True,
                            "reason": probe_gate_error,
                        },
                        "repair_bundle": self._gate_repair_bundle(
                            memory,
                            code="articulated_probe_integrity",
                            reason=probe_gate_error,
                            request=request,
                        ),
                    },
                )

            inline_ik_checker = (
                self.execute_safe_checks
                and self.checker_subagents.pre_safety_checks.get(request.name)
                == "ik_preview_check"
            )
            ik_gate_error = (
                memory.ik_execution_gate_error(
                    tool_name=request.name,
                    parameters=resolved_parameters,
                )
                if memory is not None and not inline_ik_checker
                else None
            )
            if ik_gate_error:
                ik_gate_code = (
                    "ik_target_hard_infeasible"
                    if ik_gate_error.startswith("ik_target_hard_infeasible:")
                    else "ik_collision_delegation_not_authorized"
                    if ik_gate_error.startswith(
                        "ik_collision_delegation_not_authorized:"
                    )
                    else "ik_preview_not_feasible"
                    if ik_gate_error.startswith("ik_preview_not_feasible:")
                    else "ik_preview_required"
                )
                tool_call = _skipped_tool_call(
                    request.name,
                    resolved_parameters,
                    reason=ik_gate_error,
                )
                return CommandPipelinePlan(
                    request=request,
                    status=PipelineStatus.BLOCKED,
                    tool_calls=[tool_call],
                    metadata={
                        "interface": self.interfaces.descriptor(request.kind, request.name),
                        "planner_metadata": decision.metadata,
                        "execution_rule": _tool_execution_rule(tool_call, tools),
                        "ik_execution_gate": {
                            "blocked": True,
                            "code": ik_gate_code,
                            "reason": ik_gate_error,
                        },
                        "repair_bundle": self._gate_repair_bundle(
                            memory,
                            code=ik_gate_code,
                            reason=ik_gate_error,
                            request=request,
                        ),
                    },
                )

            safety_checks = self._compile_pre_safety_checks(
                request.name,
                resolved_parameters,
                tools=tools,
                observation=observation,
            )
            if safety_checks and not _checks_allow_tool_execution(safety_checks):
                checker_reason = _failed_checker_reason(safety_checks)
                rejection_reason = (
                    "Tool call skipped because its pre-tool safety checker did not pass. "
                    + checker_reason
                ).strip()
                tool_call = _skipped_tool_call(
                    request.name,
                    resolved_parameters,
                    reason=rejection_reason,
                )
                return CommandPipelinePlan(
                    request=request,
                    status=PipelineStatus.BLOCKED,
                    safety_checks=safety_checks,
                    tool_calls=[tool_call],
                    metadata={
                        "interface": self.interfaces.descriptor(request.kind, request.name),
                        "planner_metadata": decision.metadata,
                        "execution_rule": _tool_execution_rule(tool_call, tools),
                        "checker_results": {
                            "pre_safety_checks": [call.to_dict() for call in safety_checks],
                            "post_failure_checks": [],
                        },
                        "repair_bundle": self._gate_repair_bundle(
                            memory,
                            code=_checker_gate_code(safety_checks),
                            reason=rejection_reason,
                            request=request,
                            checker_calls=safety_checks,
                        ),
                    },
                )

            tool_call = self._compile_tool_call(
                request.name,
                resolved_parameters,
                tools=tools,
                observation=observation,
                reason="Direct planner-requested tool call.",
            )
            if resolved_parameters is not request.parameters:
                # Host-only paths, matrices, and frozen bundle payloads are execution
                # inputs, not Agent-owned conversation state. Preserve the planner's
                # short public references in the action/transition ledger while the
                # durable tool-event stream retains the exact dispatched parameters.
                tool_call.parameters = dict(request.parameters)
                if isinstance(tool_call.result, dict):
                    result_details = tool_call.result.get("details")
                    if isinstance(result_details, dict):
                        result_details["parameters"] = dict(request.parameters)
            post_failure_checks = self._compile_post_failure_checks(tool_call)
            return CommandPipelinePlan(
                request=request,
                status=_aggregate_status([*safety_checks, tool_call]),
                safety_checks=safety_checks,
                tool_calls=[tool_call],
                metadata={
                    "interface": self.interfaces.descriptor(request.kind, request.name),
                    "planner_metadata": decision.metadata,
                    "execution_rule": _tool_execution_rule(tool_call, tools),
                    "checker_results": {
                        "pre_safety_checks": [call.to_dict() for call in safety_checks],
                        "post_failure_checks": [call.to_dict() for call in post_failure_checks],
                    },
                    **(
                        {"host_resolution_receipt": host_resolution_receipt}
                        if isinstance(host_resolution_receipt, dict)
                        else {}
                    ),
                    **(
                        {
                            "provenance_bundle_resolution": {
                                key: bundle_resolution.get(key)
                                for key in (
                                    "schema_version",
                                    "bundle_id",
                                    "target_evidence_id",
                                    "grasp_evidence_id",
                                    "placement_evidence_id",
                                    "compiled_grasp_id",
                                )
                            }
                        }
                        if isinstance(bundle_resolution, dict)
                        else {}
                    ),
                },
            )

        interface_descriptor = self.interfaces.descriptor(request.kind, request.name)
        return CommandPipelinePlan(
            request=request,
            status=_direct_request_status(request.kind, request.name, interface_descriptor),
            metadata={
                "interface": interface_descriptor,
                "observation_step": observation.metadata.get("step_idx"),
                "planner_metadata": decision.metadata,
            },
        )

    def _compile_skill_call(
        self,
        request: CommandRequest,
        *,
        observation: EnvObservation,
        tools: ToolRegistry,
        skills: SkillRegistry,
        planner_metadata: JsonDict,
    ) -> CommandPipelinePlan:
        skill_name = _skill_call_name(request)
        try:
            skill = skills.get(skill_name)
        except KeyError as exc:
            failed_call = PipelineCall(
                kind=CommandKind.TOOL_CALL,
                name=skill_name,
                parameters=request.parameters,
                status=PipelineStatus.FAILED,
                reason=str(exc),
            )
            return CommandPipelinePlan(
                request=request,
                status=PipelineStatus.FAILED,
                skill_call=failed_call,
            )

        del observation
        declared_allowed_tools = list(skill.allowed_tools)
        available_allowed_tools = [
            name for name in declared_allowed_tools if tools.can_execute(name)
        ]
        unavailable_allowed_tools = [
            name for name in declared_allowed_tools if not tools.can_execute(name)
        ]
        availability_rule = (
            "allowed_tools is the static skill declaration, not proof that an "
            "optional backend is configured. Call only available_allowed_tools. "
            "If a required capability is unavailable, choose an explicitly "
            "documented executable alternative or report the capability gap; "
            "never retry an unbound tool."
        )
        skill_call = PipelineCall(
            kind=CommandKind.TOOL_CALL,
            name=skill.name,
            parameters={
                "requested_parameters": request.parameters,
                "task_patterns": list(skill.task_patterns),
                "allowed_tools": declared_allowed_tools,
                "available_allowed_tools": available_allowed_tools,
                "unavailable_allowed_tools": unavailable_allowed_tools,
                "tool_availability_rule": availability_rule,
            },
            status=PipelineStatus.PLANNED,
            result={
                "success": True,
                "content": skill.content,
                "details": {
                    "description": skill.description,
                    "source": skill.source,
                    "version": skill.version,
                    "editable": skill.editable,
                    "metadata": skill.metadata,
                    "available_allowed_tools": available_allowed_tools,
                    "unavailable_allowed_tools": unavailable_allowed_tools,
                    "tool_availability_rule": availability_rule,
                },
            },
            reason=request.reasoning
            or "Skill guidance selected; planner must choose atomic tools explicitly.",
        )
        return CommandPipelinePlan(
            request=request,
            status=PipelineStatus.PLANNED,
            skill_call=skill_call,
            metadata={
                "interface": self.interfaces.descriptor(request.kind, request.name),
                "skill_description": skill.description,
                "planner_metadata": planner_metadata,
                "execution_rule": {
                    "mode": "skill_guidance_only",
                    "summary": (
                        "Skills are editable text guidance. The runtime does not "
                        "auto-expand them into hidden tool calls; the planner must "
                        "select each atomic tool in the closed-loop process."
                    ),
                },
            },
        )

    def _compile_safety_check(
        self,
        name: str,
        parameters: JsonDict,
        *,
        tools: ToolRegistry,
        observation: EnvObservation | None,
        reason: str,
    ) -> PipelineCall:
        if not self.execute_safe_checks:
            return PipelineCall(
                kind=CommandKind.TOOL_CALL,
                name=name,
                parameters=parameters,
                status=PipelineStatus.PENDING,
                reason=reason,
            )
        return self._compile_tool_call(
            name,
            parameters,
            tools=tools,
            observation=observation,
            kind=CommandKind.TOOL_CALL,
            reason=reason,
        )

    def _compile_pre_safety_checks(
        self,
        target_tool: str,
        target_parameters: JsonDict,
        *,
        tools: ToolRegistry,
        observation: EnvObservation | None,
    ) -> list[PipelineCall]:
        checker_tool = self.checker_subagents.safety_tool_for(target_tool)
        if not checker_tool:
            return []
        return [
            self._compile_safety_check(
                checker_tool,
                safety_check_parameters(
                    checker_tool=checker_tool,
                    target_tool=target_tool,
                    target_parameters=target_parameters,
                ),
                tools=tools,
                observation=observation,
                reason=f"Pre-tool safety checker for `{target_tool}`.",
            )
        ]

    def _compile_post_failure_checks(self, tool_call: PipelineCall) -> list[PipelineCall]:
        if not self.checker_subagents.should_run_failure_check(tool_call.name):
            return []
        if tool_call.status != PipelineStatus.FAILED:
            return []
        return [
            build_failure_check_call(
                checker_name=self.checker_subagents.failure_checker_name,
                target_call=tool_call,
            )
        ]

    def _compile_tool_call(
        self,
        name: str,
        parameters: JsonDict,
        *,
        tools: ToolRegistry,
        observation: EnvObservation | None = None,
        reason: str,
        kind: CommandKind = CommandKind.TOOL_CALL,
    ) -> PipelineCall:
        try:
            tools.get(name)
        except KeyError as exc:
            return PipelineCall(
                kind=kind,
                name=name,
                parameters=parameters,
                status=PipelineStatus.FAILED,
                reason=str(exc),
            )

        if not tools.can_execute(name):
            return PipelineCall(
                kind=kind,
                name=name,
                parameters=parameters,
                status=PipelineStatus.PENDING,
                reason=f"{reason} No handler registered yet.",
            )

        result = tools.call(
            name,
            parameters,
            observation=observation,
            metadata={"pipeline_kind": kind.value, "reason": reason},
        )
        return PipelineCall(
            kind=kind,
            name=name,
            parameters=parameters,
            status=PipelineStatus.EXECUTED if result.success else PipelineStatus.FAILED,
            result=_tool_result_to_dict(result),
            reason=reason,
        )

    def _compile_tool_batch(
        self,
        request: CommandRequest,
        *,
        tools: ToolRegistry,
        memory: AgentMemory | None,
        planner_metadata: JsonDict,
    ) -> CommandPipelinePlan:
        calls = _tool_batch_calls(request)
        compiled_calls: list[PipelineCall] = []
        blocked = False

        for call in calls:
            name = str(call.get("name", ""))
            parameters = call.get("parameters", {})
            if not isinstance(parameters, dict):
                parameters = {"value": parameters}
            reason = "Batched planner-requested tool call."
            try:
                spec = tools.get(name)
            except KeyError as exc:
                compiled_calls.append(
                    PipelineCall(
                        kind=CommandKind.TOOL_CALL,
                        name=name,
                        parameters=parameters,
                        status=PipelineStatus.FAILED,
                        reason=str(exc),
                    )
                )
                blocked = True
                continue

            if name == "anyplace":
                compiled_calls.append(
                    PipelineCall(
                        kind=CommandKind.TOOL_CALL,
                        name=name,
                        parameters=parameters,
                        status=PipelineStatus.BLOCKED,
                        reason=(
                            "AnyPlace must be a direct atomic tool call so the host can "
                            "resolve and audit its provenance bundle."
                        ),
                    )
                )
                blocked = True
                continue

            selection_gate_error = (
                memory.detection_selection_gate_error(
                    tool_name=name,
                    parameters=parameters,
                )
                if memory is not None
                else None
            )
            if selection_gate_error:
                compiled_calls.append(
                    PipelineCall(
                        kind=CommandKind.TOOL_CALL,
                        name=name,
                        parameters=parameters,
                        status=PipelineStatus.BLOCKED,
                        reason=selection_gate_error,
                    )
                )
                blocked = True
                continue

            provenance_gate_error = (
                memory.compiled_grasp_target_gate_error(
                    tool_name=name,
                    parameters=parameters,
                )
                if memory is not None
                else None
            )
            if provenance_gate_error:
                compiled_calls.append(
                    PipelineCall(
                        kind=CommandKind.TOOL_CALL,
                        name=name,
                        parameters=parameters,
                        status=PipelineStatus.BLOCKED,
                        reason=provenance_gate_error,
                    )
                )
                blocked = True
                continue

            probe_gate_error = (
                memory.articulated_probe_action_gate_error(
                    tool_name=name,
                    parameters=parameters,
                )
                if memory is not None
                else None
            )
            if probe_gate_error:
                compiled_calls.append(
                    PipelineCall(
                        kind=CommandKind.TOOL_CALL,
                        name=name,
                        parameters=parameters,
                        status=PipelineStatus.BLOCKED,
                        reason=probe_gate_error,
                    )
                )
                blocked = True
                continue

            execution_gate_error = (
                memory.motion_reconciliation_gate_error(tool_name=name)
                if memory is not None
                else None
            )
            if execution_gate_error:
                compiled_calls.append(
                    PipelineCall(
                        kind=CommandKind.TOOL_CALL,
                        name=name,
                        parameters=parameters,
                        status=PipelineStatus.BLOCKED,
                        reason=execution_gate_error,
                    )
                )
                blocked = True
                continue

            if not spec.allows_batched_observation:
                compiled_calls.append(
                    PipelineCall(
                        kind=CommandKind.TOOL_CALL,
                        name=name,
                        parameters=parameters,
                        status=PipelineStatus.BLOCKED,
                        reason=(
                            f"{reason} Tool effect `{spec.effect.value}` requires "
                            "a fresh observation before another tool is selected."
                        ),
                    )
                )
                blocked = True
                continue

            compiled_calls.append(
                self._compile_tool_call(
                    name,
                    parameters,
                    tools=tools,
                    observation=None,
                    reason=reason,
                )
            )

        status = PipelineStatus.BLOCKED if blocked else _aggregate_status(compiled_calls)
        repair_bundles = [
            _batch_gate_repair(memory, call)
            for call in compiled_calls
            if call.status in {PipelineStatus.BLOCKED, PipelineStatus.SKIPPED}
        ]
        return CommandPipelinePlan(
            request=request,
            status=status,
            tool_calls=compiled_calls,
            metadata={
                "interface": self.interfaces.descriptor(request.kind, request.name),
                "planner_metadata": planner_metadata,
                "execution_rule": {
                    "mode": "batched_read_only_tools",
                    "allowed_effects": ["read_only", "bookkeeping", "planning"],
                    "blocked_effects": ["world_mutating"],
                    "requires_observation_after_batch": False,
                },
                **({"repair_bundles": repair_bundles} if repair_bundles else {}),
            },
        )


def _decision_to_request(decision: PlannerDecision) -> CommandRequest:
    kind = _normalize_command_kind(decision.action_type, decision.skill)
    name = decision.action
    parameters = dict(decision.parameters)
    return CommandRequest(
        kind=kind,
        name=name,
        parameters=parameters,
        reasoning=decision.reasoning,
        code=decision.code,
    )


def _normalize_command_kind(action_type: str, skill: str | None) -> CommandKind:
    del skill
    normalized = action_type.lower().strip()
    if normalized == "response":
        return CommandKind.RESPONSE
    if normalized != "tool_call":
        raise ValueError(f"Unsupported command kind: {action_type!r}")
    return CommandKind.TOOL_CALL


def _is_tool_batch_request(request: CommandRequest) -> bool:
    return request.name in {"batch", "tool_batch"} or isinstance(
        request.parameters.get("calls"), list
    )


def _is_skill_call_request(request: CommandRequest) -> bool:
    return request.name == "skill_call"


def _is_safety_check_request(request: CommandRequest) -> bool:
    return request.name == "safe_check"


def _is_direct_tool_like_request(request: CommandRequest) -> bool:
    return request.name in {"sense", "code_policy"}


def _skill_call_name(request: CommandRequest) -> str:
    for key in ("skill", "name"):
        value = request.parameters.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return request.name


def _named_tool_target(request: CommandRequest) -> str:
    for key in ("tool", "target", "name"):
        value = request.parameters.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return request.name


def _tool_batch_calls(request: CommandRequest) -> list[JsonDict]:
    calls = request.parameters.get("calls", [])
    if not isinstance(calls, list):
        return []
    return [call for call in calls if isinstance(call, dict)]


def _tool_execution_rule(call: PipelineCall, tools: ToolRegistry) -> JsonDict:
    try:
        spec = tools.get(call.name)
    except KeyError:
        return {"mode": "unknown_tool"}
    return {
        "mode": "single_tool_closed_loop",
        "effect": spec.effect.value,
        "batchable": spec.allows_batched_observation,
        "requires_observation_after_call": spec.requires_observation_after_call,
    }


def _tool_result_to_dict(result: ToolResult) -> JsonDict:
    return {
        "success": result.success,
        "content": result.content,
        "details": result.details,
    }


def _checks_allow_tool_execution(calls: list[PipelineCall]) -> bool:
    for call in calls:
        if call.status != PipelineStatus.EXECUTED:
            return False
        if isinstance(call.result, dict) and call.result.get("success") is False:
            return False
    return True


def _detection_selection_gate_error(
    request: CommandRequest,
    *,
    memory: AgentMemory | None,
) -> str | None:
    if memory is None:
        return None
    return memory.detection_selection_gate_error(
        tool_name=request.name,
        parameters=request.parameters,
    )


def _gate_repair_bundle(
    memory: AgentMemory | None,
    *,
    code: str,
    reason: str,
    request: CommandRequest,
    checker_calls: list[PipelineCall] | None = None,
    tool_contract_catalog: ToolContractCatalog | None = None,
    tool_contract_policy: ToolContractRuntimePolicy | None = None,
) -> JsonDict:
    if memory is None:
        bundle = {
            "schema_version": "openeta.gate_repair.v1",
            "extensions": {},
            "code": code,
            "violated_invariant": reason,
            "requested_call": {
                "tool": request.name,
                "parameters": dict(request.parameters),
            },
            "evidence_ids": [],
            "allowed_next_calls": [],
            "stale_evidence": [],
        }
    else:
        bundle = memory.gate_repair_bundle(
            code=code,
            reason=reason,
            requested_tool=request.name,
            requested_parameters=request.parameters,
        )
    if checker_calls:
        bundle["checker_evidence"] = [call.to_dict() for call in checker_calls]
    contract_validation = _gate_contract_shadow_validation(
        request.name,
        bundle,
        tool_contract_catalog=tool_contract_catalog,
        tool_contract_policy=tool_contract_policy,
    )
    bundle["contract_shadow_validation"] = contract_validation
    if (
        contract_validation.get("enforcing") is True
        and contract_validation.get("conformant") is False
    ):
        # The world-mutating call is already blocked. Keep it blocked, expose the
        # host defect, and offer only a fresh observation while the malformed
        # repair envelope is retained for diagnosis.
        bundle["contract_enforcement"] = {
            "status": "repair_envelope_rejected",
            "authority": "tool_contract",
            "violations": list(contract_validation.get("violations") or []),
        }
        bundle["allowed_next_calls"] = [{"tool": "observe", "parameters": {}}]
    return bundle


def _gate_contract_shadow_validation(
    tool_name: str,
    bundle: JsonDict,
    *,
    tool_contract_catalog: ToolContractCatalog | None = None,
    tool_contract_policy: ToolContractRuntimePolicy | None = None,
) -> JsonDict:
    """Audit repair feedback and expose its independent validation authority."""

    from agent.tools.contracts import ContractMaturity, check_gate_repair_conformance

    catalog = tool_contract_catalog or _default_gate_contract_catalog()
    policy = tool_contract_policy or ToolContractRuntimePolicy()
    try:
        contract = catalog.get(tool_name)
    except KeyError:
        return {
            "schema_version": "openeta.gate_contract_shadow_validation.v1",
            "evaluated": False,
            "enforcing": False,
            "tool": tool_name,
            "reason": "no ToolContract is registered",
        }
    if contract.maturity is ContractMaturity.INFERRED:
        return {
            "schema_version": "openeta.gate_contract_shadow_validation.v1",
            "evaluated": False,
            "enforcing": False,
            "tool": tool_name,
            "contract_maturity": contract.maturity.value,
            "reason": "inferred contracts are inventory-only",
        }
    violations = check_gate_repair_conformance(contract, bundle)
    contract_authoritative = policy.gate_repair_is_authoritative(tool_name)
    repair_code = str(bundle.get("code") or "")
    matching_bindings = [
        binding.to_dict()
        for binding in contract.gate.bindings
        if repair_code in binding.repair_codes
    ]
    return {
        "schema_version": "openeta.gate_contract_shadow_validation.v1",
        "evaluated": True,
        "enforcing": contract_authoritative,
        "authoritative_gate": "legacy_runtime",
        "repair_envelope_authority": (
            "tool_contract" if contract_authoritative else "legacy_runtime"
        ),
        "tool": tool_name,
        "contract_maturity": contract.maturity.value,
        "contract_gate_checks": list(contract.gate.checks),
        "contract_gate_binding_count": len(contract.gate.bindings),
        "matched_gate_bindings": matching_bindings,
        "conformant": not violations,
        "violations": [violation.to_dict() for violation in violations],
    }


@lru_cache(maxsize=1)
def _default_gate_contract_catalog():
    from agent.tools.contracts import build_default_tool_contract_catalog
    from agent.tools.registry import build_default_tool_registry

    return build_default_tool_contract_catalog(build_default_tool_registry().list())


def _failed_checker_reason(calls: list[PipelineCall]) -> str:
    for call in calls:
        result = call.result if isinstance(call.result, dict) else {}
        details = result.get("details") if isinstance(result.get("details"), dict) else {}
        outputs = details.get("outputs") if isinstance(details.get("outputs"), dict) else {}
        reachability = outputs.get("reachability")
        if isinstance(reachability, dict):
            return (
                f"{call.name} reported {reachability.get('status')!r}: "
                f"{reachability.get('reason_code') or reachability.get('message') or 'no reason'}"
            )
        content = str(result.get("content") or call.reason or "").strip()
        if content:
            return f"{call.name}: {content}"
    return "The checker returned no usable explanation; this is a checker contract defect."


def _checker_gate_code(calls: list[PipelineCall]) -> str:
    for call in calls:
        result = call.result if isinstance(call.result, dict) else {}
        details = result.get("details") if isinstance(result.get("details"), dict) else {}
        outputs = details.get("outputs") if isinstance(details.get("outputs"), dict) else {}
        receipt = outputs.get("ik_preview_receipt")
        if isinstance(receipt, dict) and receipt.get("classification") == "hard_infeasible":
            return "ik_target_hard_infeasible"
        if call.name == "ik_preview_check":
            return "ik_preview_not_feasible"
    return "pre_safety_check_failed"


def _batch_gate_repair(
    memory: AgentMemory | None, call: PipelineCall
) -> JsonDict:
    reason = call.reason or "Batched tool call was rejected without a reason."
    lowered = reason.lower()
    if call.name == "anyplace":
        code = "anyplace_requires_atomic_call"
    elif "unverified segmentation" in lowered or "mask_ref" in lowered:
        code = "perception_provenance_integrity"
    elif "compiled_grasp" in lowered:
        code = _compiled_grasp_gate_code(reason)
    elif "reconciliation" in lowered or "transport-unknown" in lowered:
        code = "motion_reconciliation_required"
    elif "fresh observation" in lowered or "tool effect" in lowered:
        code = "batch_requires_observation_boundary"
    else:
        code = "batch_gate_rejection"
    request = CommandRequest(
        kind=CommandKind.TOOL_CALL,
        name=call.name,
        parameters=dict(call.parameters),
    )
    return _gate_repair_bundle(
        memory,
        code=code,
        reason=reason,
        request=request,
    )


def _compiled_grasp_gate_code(reason: str) -> str:
    if reason.startswith(
        (
            "compiled_grasp_adjustment_",
            "compiled_clearance_",
            "compiled_contact_",
            "attached_release_",
        )
    ):
        return reason.split(":", 1)[0]
    return "compiled_grasp_target_superseded"


def _skipped_tool_call(name: str, parameters: JsonDict, *, reason: str) -> PipelineCall:
    return PipelineCall(
        kind=CommandKind.TOOL_CALL,
        name=name,
        parameters=parameters,
        status=PipelineStatus.SKIPPED,
        reason=reason,
    )


def _aggregate_status(calls: list[PipelineCall]) -> PipelineStatus:
    if any(call.status == PipelineStatus.FAILED for call in calls):
        return PipelineStatus.FAILED
    if any(call.status == PipelineStatus.BLOCKED for call in calls):
        return PipelineStatus.BLOCKED
    if any(call.status == PipelineStatus.PENDING for call in calls):
        return PipelineStatus.PENDING
    if calls and all(call.status == PipelineStatus.EXECUTED for call in calls):
        return PipelineStatus.EXECUTED
    return PipelineStatus.PLANNED


def _direct_request_status(
    kind: CommandKind,
    name: str,
    interface_descriptor: JsonDict,
) -> PipelineStatus:
    if kind == CommandKind.RESPONSE and name in {"talk", "task_complete"}:
        return PipelineStatus.EXECUTED
    if not interface_descriptor.get("implemented", False):
        return PipelineStatus.PENDING
    return PipelineStatus.PLANNED
