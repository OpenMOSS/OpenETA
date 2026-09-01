"""Explicit phase-1 contracts for OpenETA's core manipulation toolchain.

These declarations describe the live interfaces without activating a new
validator or gate.  They are intentionally kept separate from handler code so
the migration can compare the contract against the existing implementation
before making it authoritative at runtime.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import Any

from adapter.protocol import JsonDict
from agent.tools.contracts import (
    ContractMaturity,
    EvidenceLifetimeContract,
    FactAuthority,
    FactBinding,
    GateContract,
    GateCheckBinding,
    HostResolutionContract,
    OutcomeContract,
    ToolContract,
    ToolSpecLike,
)


OBSERVATION_PACKET = "openeta.observation_packet.v1"
SAM3_DETECTION_SET = "openeta.sam3_detection_set.v1"
SELECTED_TARGET_MASK = "openeta.selected_target_mask.v1"
GRASP_INPUT_BUNDLE = "openeta.grasp_input_bundle.v1"
GRASP_CANDIDATE_SET = "openeta.grasp_candidate_set.v1"
COMPILED_GRASP = "openeta.compiled_grasp_seed.v1"
WRIST_VIEWPOINT_PROPOSAL = "openeta.wrist_viewpoint_proposal.v1"
WRIST_ALIGNMENT_BUNDLE = "openeta.wrist_alignment_bundle.v1"
ALIGNED_GRASP_REFERENCE = "openeta.aligned_grasp_reference.v1"
IK_EXECUTION_AUTHORIZATION = "openeta.ik_execution_authorization.v1"
MOTION_EXECUTION_RECEIPT = "openeta.motion_execution_receipt.v1"
GRIPPER_LATCH_RECEIPT = "openeta.gripper_latch_receipt.v1"
PLACEMENT_INPUT_BUNDLE = "openeta.anyplace_input_bundle.v1"
PLACEMENT_CANDIDATE_SET = "openeta.placement_candidate_set.v1"
WORLD_POSE_REFERENCE = "openeta.world_pose_reference.v1"
ENVIRONMENT_SESSION = "openeta.environment_session.v1"
PYTHON_EXECUTION_RESULT = "openeta.python_execution_result.v1"
MEMORY_ENTRY = "openeta.memory_entry.v1"
MEMORY_SNAPSHOT = "openeta.memory_snapshot.v1"
MEMORY_COMPACTION = "openeta.memory_compaction.v1"
WEB_SEARCH_RESULT = "openeta.web_search.v1"
WEB_PAGE_TEXT = "openeta.web_fetch.v1"
DEPTH_PRIOR = "openeta.depth_prior.v1"
ENHANCED_DEPTH = "openeta.enhanced_depth.v1"
POINT_GROUNDING_SET = "openeta.point_grounding_set.v1"
ASSET_REFERENCE_GROUNDING = "openeta.asset_reference_grounding.v1"
SAM3_REJECTION = "openeta.sam3_rejection.v1"
ATTACHMENT_PROBE_PLAN = "openeta.articulated_attachment_probe.v1"
COMPLETED_ATTACHMENT_PROBE = "openeta.completed_articulated_attachment_probe.v1"
ATTACHMENT_ASSESSMENT = "openeta.articulated_attachment_assessment.v1"
CALIBRATION_PROPOSAL = "openeta.calibration_proposal.v1"
PUBLISHED_CALIBRATION = "openeta.published_calibration_profile.v1"
GRASP_STRATEGY_PROPOSAL = "openeta.grasp_strategy_proposal.v1"
PUBLISHED_GRASP_STRATEGY = "openeta.published_grasp_strategy.v1"
EDITABLE_SKILL = "openeta.editable_skill_spec.v1"


def _string(description: str = "", **keywords: Any) -> JsonDict:
    result: JsonDict = {"type": "string", **keywords}
    if description:
        result["description"] = description
    return result


def _number(description: str = "", **keywords: Any) -> JsonDict:
    result: JsonDict = {"type": "number", **keywords}
    if description:
        result["description"] = description
    return result


def _integer(description: str = "", **keywords: Any) -> JsonDict:
    result: JsonDict = {"type": "integer", **keywords}
    if description:
        result["description"] = description
    return result


def _boolean(description: str = "") -> JsonDict:
    result: JsonDict = {"type": "boolean"}
    if description:
        result["description"] = description
    return result


def _object(
    properties: JsonDict | None = None,
    *,
    required: Iterable[str] = (),
    additional_properties: bool = False,
    description: str = "",
    one_of: list[JsonDict] | None = None,
) -> JsonDict:
    result: JsonDict = {
        "type": "object",
        "properties": dict(properties or {}),
        "additionalProperties": additional_properties,
    }
    required_values = list(required)
    if required_values:
        result["required"] = required_values
    if description:
        result["description"] = description
    if one_of:
        result["oneOf"] = one_of
    return result


def _array(items: JsonDict, description: str = "", **keywords: Any) -> JsonDict:
    result: JsonDict = {"type": "array", "items": items, **keywords}
    if description:
        result["description"] = description
    return result


def _outputs(required: Iterable[str], properties: JsonDict | None = None) -> JsonDict:
    return _object(
        properties,
        required=required,
        # Handler diagnostics and backend-specific audit fields remain allowed.
        additional_properties=True,
    )


def _fact(
    fact_type: str,
    path: str,
    authority: FactAuthority,
    *,
    required: bool = True,
    when: str = "",
    description: str = "",
) -> FactBinding:
    return FactBinding(
        fact_type=fact_type,
        schema_version=fact_type,
        path=path,
        authority=authority,
        required=required,
        when=when,
        description=description,
    )


def _operational_failure() -> OutcomeContract:
    return OutcomeContract(
        semantic_outcome="operational_failure",
        operational_success=False,
        output_schema=_object(additional_properties=True),
        diagnostics_required=True,
        recovery_required=True,
        description="The call did not produce its advertised semantic fact.",
    )


def _explicit_contract(
    spec: ToolSpecLike,
    *,
    request_schema: JsonDict,
    outcomes: tuple[OutcomeContract, ...],
    consumes: tuple[FactBinding, ...] = (),
    host_resolution: HostResolutionContract | None = None,
    evidence_lifetime: EvidenceLifetimeContract | None = None,
    gate: GateContract | None = None,
    source_paths: tuple[str, ...] = (),
    maturity: ContractMaturity = ContractMaturity.VERIFIED,
    coverage_gaps: tuple[str, ...] = (),
) -> ToolContract:
    effect = getattr(spec.effect, "value", spec.effect)
    return ToolContract(
        name=spec.name,
        category=spec.category,
        description=spec.description,
        effect=str(effect),
        safe_by_default=spec.safe_by_default,
        batchable=spec.allows_batched_observation,
        requires_observation_after_call=spec.requires_observation_after_call,
        request_schema=request_schema,
        outcomes=outcomes,
        consumes=consumes,
        host_resolution=host_resolution or HostResolutionContract(),
        evidence_lifetime=evidence_lifetime or EvidenceLifetimeContract(),
        gate=gate or GateContract(),
        maturity=maturity,
        source_paths=("agent/tools/default_contracts.py", *source_paths),
        coverage_gaps=coverage_gaps,
    )


def build_default_contract_overrides(
    tool_specs: Iterable[ToolSpecLike],
) -> dict[str, ToolContract]:
    """Return explicit contracts keyed by tool name for the core toolchain."""

    specs = {spec.name: spec for spec in tool_specs}
    builders = {
        "observe": _observe,
        "create_simulator_env": _create_simulator_env,
        "close_simulator_env": _close_simulator_env,
        "python_exec": _python_exec,
        "web_search": _web_search,
        "web_fetch": _web_fetch,
        "estimate_depth_prior": _estimate_depth_prior,
        "enhance_depth": _enhance_depth,
        "sam3": _sam3,
        "retrieve_asset_reference": _retrieve_asset_reference,
        "molmopoint": _molmopoint,
        "select_sam3_detection": _select_sam3,
        "reject_sam3_detections": _reject_sam3,
        "grasp_pose_estimate": _grasp_pose_estimate,
        "compile_grasp_seed": _compile_grasp_seed,
        "propose_wrist_viewpoints": _propose_wrist_viewpoints,
        "compute_wrist_alignment": _compute_wrist_alignment,
        "ik_preview_check": _ik_preview_check,
        "prepare_attachment_probe": _prepare_attachment_probe,
        "assess_attachment_probe": _assess_attachment_probe,
        "move_to": _move_to,
        "follow_eef_trajectory": _follow_eef_trajectory,
        "gripper_control": _gripper_control,
        "anyplace": _anyplace,
        "camera_pose_to_world": _camera_pose_to_world,
        "propose_calibration_profile": _propose_calibration_profile,
        "promote_calibration_profile": _promote_calibration_profile,
        "propose_grasp_strategy": _propose_grasp_strategy,
        "promote_grasp_strategy": _promote_grasp_strategy,
        "save_memory": _save_memory,
        "get_memory": _get_memory,
        "delete_memory": _delete_memory,
        "compact_memory": _compact_memory,
        "register_skill": _register_skill,
        "update_skill": _update_skill,
    }
    contracts: dict[str, ToolContract] = {}
    from agent.tools.runtime_contract_bindings import bind_host_resolution_contract

    for name, spec in specs.items():
        if name not in builders:
            continue
        contract = builders[name](spec)
        contract = replace(
            contract,
            host_resolution=bind_host_resolution_contract(
                name,
                contract.host_resolution,
            ),
        )
        runtime_bindings = _runtime_gate_bindings(name)
        if runtime_bindings:
            contract = replace(
                contract,
                gate=replace(
                    contract.gate,
                    bindings=(*contract.gate.bindings, *runtime_bindings),
                ),
            )
        contracts[name] = contract
    return contracts


_COMPILED_GRASP_REPAIR_CODES = (
    "attached_release_after_failed_motion",
    "compiled_clearance_not_reached",
    "compiled_contact_approach_misaligned",
    "compiled_contact_orientation_misaligned",
    "compiled_contact_not_reached",
    "compiled_contact_receipt_mismatch",
    "compiled_contact_receipt_missing",
    "compiled_contact_receipt_stale",
    "compiled_grasp_adjustment_invalid",
    "compiled_grasp_adjustment_out_of_bounds",
    "compiled_grasp_adjustment_stale",
    "compiled_grasp_adjustment_superseded",
    "compiled_grasp_adjustment_unresolved",
    "compiled_grasp_adjustment_unverified_orientation_policy",
    "compiled_grasp_target_superseded",
)


def _runtime_gate_bindings(tool_name: str) -> tuple[GateCheckBinding, ...]:
    """Bind live gate branches to stable contract ids without prescribing order."""

    bindings: list[GateCheckBinding] = []
    if tool_name != "observe":
        bindings.append(
            GateCheckBinding(
                check_id="runtime.motion_reconciliation",
                description="Unknown mutation outcomes must be observed and reconciled.",
                implementation="agent/runtime/memory.py:AgentMemory.motion_reconciliation_gate_error",
                repair_codes=("motion_reconciliation_required",),
                applies_when="motion reconciliation status is required or unresolved",
            )
        )
    if tool_name in {
        "retrieve_asset_reference",
        "sam3",
        "molmopoint",
        "estimate_depth_prior",
        "enhance_depth",
    }:
        bindings.append(
            GateCheckBinding(
                check_id="runtime.source_packet_resolution",
                description="Opaque packet and camera ids resolve to one immutable host packet.",
                implementation=(
                    "agent/tools/runtime_contract_bindings.py:resolve_host_parameters"
                ),
                repair_codes=("invalid_source_packet", "same_view_packet_mismatch"),
                applies_when="the public request contains source packet references",
            )
        )
    if tool_name in {
        "anyplace",
        "camera_pose_to_world",
        "compile_grasp_seed",
        "compute_wrist_alignment",
        "propose_wrist_viewpoints",
        "grasp_pose_estimate",
        "ik_preview_check",
    }:
        bindings.append(
            GateCheckBinding(
                check_id="runtime.provenance_bundle_resolution",
                description="Opaque evidence references resolve without Agent-authored overrides.",
                implementation="agent/runtime/pipeline.py:ActionPipeline.compile",
                repair_codes=("invalid_provenance_bundle",),
                applies_when="the request selects a host-resolved bundle/reference branch",
            )
        )
    if tool_name == "ik_preview_check":
        bindings.append(
            GateCheckBinding(
                check_id="runtime.compiled_grasp_reference_resolution",
                description="Compiled grasp id and waypoint role resolve to one immutable pose.",
                implementation="agent/runtime/memory.py:AgentMemory.resolve_compiled_grasp_pose_reference",
                repair_codes=("invalid_compiled_grasp_reference",),
                applies_when="compiled_grasp_id is supplied",
            )
        )
    if tool_name == "move_to":
        bindings.append(
            GateCheckBinding(
                check_id="runtime.ik_receipt_resolution",
                description="A motion request resolves an exact host-issued IK receipt.",
                implementation="agent/runtime/memory.py:AgentMemory.resolve_ik_motion_reference",
                repair_codes=("invalid_ik_receipt_reference",),
            )
        )
    if tool_name == "follow_eef_trajectory":
        bindings.append(
            GateCheckBinding(
                check_id="runtime.ik_trajectory_resolution",
                description="Every trajectory waypoint resolves from ordered host-issued IK receipts.",
                implementation="agent/runtime/memory.py:AgentMemory.resolve_ik_trajectory_reference",
                repair_codes=("invalid_ik_trajectory_reference",),
            )
        )
    if tool_name in {"move_to", "follow_eef_trajectory", "gripper_control"}:
        bindings.append(
            GateCheckBinding(
                check_id="runtime.compiled_grasp_provenance",
                description="Contact and close cannot treat stale or failed grasp evidence as success.",
                implementation="agent/runtime/memory.py:AgentMemory.compiled_grasp_target_gate_error",
                repair_codes=_COMPILED_GRASP_REPAIR_CODES,
                applies_when="the action is geometrically tied to a compiled targeted grasp",
            )
        )
    if tool_name in {"move_to", "follow_eef_trajectory"}:
        bindings.extend(
            (
                GateCheckBinding(
                    check_id="runtime.articulated_probe_integrity",
                    description="A frozen attachment probe cannot be replayed or edited.",
                    implementation="agent/runtime/memory.py:AgentMemory.articulated_probe_action_gate_error",
                    repair_codes=("articulated_probe_integrity",),
                    applies_when="motion carries an articulated probe hash",
                ),
                GateCheckBinding(
                    check_id="runtime.ik_execution_authorization",
                    description="Motion uses current, pose-policy-equivalent feasible IK evidence.",
                    implementation="agent/runtime/memory.py:AgentMemory.ik_execution_gate_error",
                    repair_codes=(
                        "ik_preview_required",
                        "ik_preview_not_feasible",
                        "ik_target_hard_infeasible",
                        "ik_collision_delegation_not_authorized",
                    ),
                ),
                GateCheckBinding(
                    check_id="runtime.pre_safety_checker",
                    description="Configured safety checker results fail closed with evidence.",
                    implementation="agent/runtime/pipeline.py:ActionPipeline._compile_pre_safety_checks",
                    repair_codes=(
                        "pre_safety_check_failed",
                        "ik_preview_not_feasible",
                        "ik_target_hard_infeasible",
                    ),
                ),
            )
        )
    bindings.append(
        GateCheckBinding(
            check_id="runtime.batch_boundary",
            description="Dependent or world-mutating calls cannot bypass observation boundaries.",
            implementation="agent/runtime/pipeline.py:ActionPipeline._compile_tool_batch",
            repair_codes=(
                "anyplace_requires_atomic_call",
                "batch_requires_observation_boundary",
                "batch_gate_rejection",
                "perception_provenance_integrity",
            ),
            applies_when="the tool appears inside tool_batch",
        )
    )
    return tuple(bindings)


def _observe(spec: ToolSpecLike) -> ToolContract:
    return _explicit_contract(
        spec,
        request_schema=_object({"reason": _string()}, additional_properties=False),
        outcomes=(
            OutcomeContract(
                "completed",
                True,
                _outputs(
                    ("camera_ids", "objects", "metadata"),
                    {
                        "camera_ids": _array(_string()),
                        "objects": _array(_object(additional_properties=True)),
                        "metadata": _object(additional_properties=True),
                    },
                ),
                produces=(
                    _fact(
                        OBSERVATION_PACKET,
                        "host.post_call_observation.source_packet_id",
                        FactAuthority.HOST,
                        description=(
                            "The environment adapter materializes an opaque packet after "
                            "the observe call; it is not a handler-authored output field."
                        ),
                    ),
                ),
            ),
            _operational_failure(),
        ),
        evidence_lifetime=EvidenceLifetimeContract(
            scope="session_packet",
            freshness_dimensions=("object_scene_epoch", "robot_motion_epoch"),
            invalidated_by=(),
            reusable_across_packet_refresh=True,
            description=(
                "The packet remains durable and queryable; consumers decide whether its "
                "captured epochs are current enough for their purpose."
            ),
        ),
        host_resolution=HostResolutionContract(
            mode="post_call_observation",
            resolver="ObservationPacketStore",
            resolved_parameters=(
                "fresh environment observation",
                "session-owned camera artifact paths",
                "source_packet_id",
            ),
            freshness_dimensions=("object_scene_epoch", "robot_motion_epoch"),
            description=(
                "The observe handler returns a summary. The episode/environment boundary "
                "then creates the durable packet fact consumed by later perception tools."
            ),
        ),
        source_paths=(
            "agent/tools/handlers.py",
            "agent/runtime/episode.py",
            "agent/runtime/observation_packets.py",
        ),
    )


def _create_simulator_env(spec: ToolSpecLike) -> ToolContract:
    request = _object(
        {
            "env_id": _string(minLength=1),
            "seed": _integer(),
            "task": _string(),
            "render_mode": _string(),
            "image_width": _integer(minimum=1),
            "image_height": _integer(minimum=1),
            "session_id": _string(),
            "include_objects": _boolean(),
        },
        required=("env_id",),
    )
    return _explicit_contract(
        spec,
        request_schema=request,
        outcomes=(
            OutcomeContract(
                "mutation_acknowledged",
                True,
                _outputs(
                    ("environment", "initial_observation"),
                    {
                        "environment": _object(
                            {
                                "env_id": _string(),
                                "handle": _string(),
                                "session_id": _string(),
                            },
                            required=("env_id", "handle"),
                            additional_properties=True,
                        ),
                        "initial_observation": _object(additional_properties=True),
                        "assigned_task": _string(),
                    },
                ),
                produces=(
                    _fact(
                        ENVIRONMENT_SESSION,
                        "outputs.environment.handle",
                        FactAuthority.ENVIRONMENT,
                    ),
                ),
            ),
            _operational_failure(),
        ),
        host_resolution=HostResolutionContract(
            mode="environment_lifecycle",
            resolver="SimulatorEnvironmentCreator",
            agent_parameters=("env_id", "seed", "task", "render options", "session_id"),
            resolved_parameters=("remote create_env request", "remote reset_env request"),
            invalidated_by=("another environment is already active",),
            description="The host stores the returned handle; it is never copied into later Agent calls.",
        ),
        evidence_lifetime=EvidenceLifetimeContract(
            scope="active_environment",
            freshness_dimensions=("environment handle",),
            invalidated_by=("close_simulator_env", "environment reset or service loss"),
            reusable_across_packet_refresh=True,
        ),
        gate=GateContract(
            checks=(
                "no environment handle is already active",
                "creation parameters have bounded valid types",
                "create response contains a handle before host state is committed",
            ),
            fail_closed=True,
            description="Cancellation after remote creation performs best-effort cleanup.",
        ),
        source_paths=("agent/tools/sim_mcp.py", "agent/runtime/runtime_assembly.py"),
    )


def _close_simulator_env(spec: ToolSpecLike) -> ToolContract:
    return _explicit_contract(
        spec,
        request_schema=_object(),
        consumes=(
            _fact(
                ENVIRONMENT_SESSION,
                "host.active_environment_handle",
                FactAuthority.HOST,
                required=False,
                description="No-op close is valid when no handle is active.",
            ),
        ),
        outcomes=(
            OutcomeContract(
                "mutation_acknowledged",
                True,
                _outputs(
                    ("closed",),
                    {
                        "closed": _boolean(),
                        "skipped": _boolean(),
                        "environment": _object(additional_properties=True),
                    },
                ),
            ),
            _operational_failure(),
        ),
        host_resolution=HostResolutionContract(
            mode="host_active_handle",
            resolver="SimulatorEnvironmentCloser",
            resolved_parameters=("active handle", "simulator session id"),
            invalidated_by=(),
        ),
        evidence_lifetime=EvidenceLifetimeContract(
            scope="environment_receipt",
            invalidated_by=(),
            reusable_across_packet_refresh=True,
            description="The close receipt is durable; the old handle cannot authorize later calls.",
        ),
        gate=GateContract(
            checks=("only the host-stored active handle may be closed",),
            fail_closed=True,
        ),
        source_paths=("agent/tools/sim_mcp.py",),
    )


def _python_exec(spec: ToolSpecLike) -> ToolContract:
    request = _object(
        {
            "code": _string(minLength=1),
            "sandbox": _string(enum=["sandbox", "outside_sandbox"]),
            "timeout_s": _number(exclusiveMinimum=0, maximum=600),
        },
        required=("code",),
    )
    return _explicit_contract(
        spec,
        request_schema=request,
        outcomes=(
            OutcomeContract(
                "completed",
                True,
                _outputs(
                    ("result", "stdout", "sandbox"),
                    {
                        "result": {},
                        "result_inline_complete": _boolean(),
                        "result_artifact": _object(additional_properties=True),
                        "stdout": _string(),
                        "stderr": _string(),
                        "sandbox": _string(),
                        "workspace": _object(additional_properties=True),
                        "artifacts": _array(_object(additional_properties=True)),
                    },
                ),
                produces=(
                    _fact(
                        PYTHON_EXECUTION_RESULT,
                        "outputs.result",
                        FactAuthority.TOOL,
                    ),
                ),
                description="Large results may be previews accompanied by result_artifact.path.",
            ),
            _operational_failure(),
        ),
        host_resolution=HostResolutionContract(
            mode="sandbox_capability_injection",
            resolver="PythonExecRuntime",
            agent_parameters=("code", "sandbox", "timeout_s"),
            resolved_parameters=(
                "session artifact API",
                "read-only session roots",
                "writable sandbox root",
                "current observation projection",
            ),
            description="The sandbox has no Simulator MCP or network capability.",
        ),
        evidence_lifetime=EvidenceLifetimeContract(
            scope="durable_artifact_or_inline_result",
            invalidated_by=("artifact deletion",),
            reusable_across_packet_refresh=True,
        ),
        gate=GateContract(
            checks=(
                "code is non-empty",
                "sandbox mode is supported",
                "filesystem access stays inside declared roots",
                "outside_sandbox has explicit per-call user approval",
                "execution timeout stays within host cap",
            ),
            fail_closed=True,
            description="World mutation and external services remain separate stable tools.",
        ),
        source_paths=("agent/tools/coding.py",),
    )


def _web_search(spec: ToolSpecLike) -> ToolContract:
    return _explicit_contract(
        spec,
        request_schema=_object(
            {
                "query": _string(minLength=1, maxLength=512),
                "max_results": _integer(minimum=1, maximum=10),
                "language": _string(maxLength=32),
                "time_range": _string(enum=["", "day", "month", "year"]),
            },
            required=("query",),
        ),
        outcomes=(
            OutcomeContract(
                "completed",
                True,
                _outputs(
                    ("schema_version", "answer", "results", "untrusted_external_content"),
                    {
                        "schema_version": _string(enum=[WEB_SEARCH_RESULT]),
                        "answer": _string(),
                        "results": _array(_object(additional_properties=True)),
                        "result_count": _integer(minimum=0),
                        "untrusted_external_content": {"const": True},
                    },
                ),
                produces=(
                    _fact(WEB_SEARCH_RESULT, "outputs", FactAuthority.TOOL),
                ),
            ),
            _operational_failure(),
        ),
        host_resolution=HostResolutionContract(
            mode="configured_provider_capability",
            resolver="HostedWebSearchClient",
            agent_parameters=("query", "max_results", "language", "time_range"),
            resolved_parameters=("provider endpoint", "model", "host credentials"),
            description="Credentials never appear in the Agent request or result.",
        ),
        evidence_lifetime=EvidenceLifetimeContract(
            scope="external_content_snapshot",
            invalidated_by=("source web content may change independently",),
            reusable_across_packet_refresh=True,
        ),
        gate=GateContract(
            checks=(
                "query and bounds are valid",
                "a configured provider actually executes hosted web search",
                "URLs and citations are normalized and bounded",
                "result is labeled untrusted external content",
            ),
            fail_closed=True,
        ),
        source_paths=("agent/runtime/planner.py", "agent/tools/web_access.py"),
    )


def _web_fetch(spec: ToolSpecLike) -> ToolContract:
    return _explicit_contract(
        spec,
        request_schema=_object(
            {
                "url": _string(minLength=1, maxLength=2048, pattern="^https://"),
                "max_chars": _integer(minimum=1, maximum=40000),
            },
            required=("url",),
        ),
        outcomes=(
            OutcomeContract(
                "completed",
                True,
                _outputs(
                    ("schema_version", "url", "text", "untrusted_external_content"),
                    {
                        "schema_version": _string(enum=[WEB_PAGE_TEXT]),
                        "url": _string(),
                        "title": _string(),
                        "text": _string(),
                        "truncated": _boolean(),
                        "untrusted_external_content": {"const": True},
                    },
                ),
                produces=(
                    _fact(WEB_PAGE_TEXT, "outputs", FactAuthority.TOOL),
                ),
            ),
            _operational_failure(),
        ),
        host_resolution=HostResolutionContract(
            mode="public_https_fetch",
            resolver="build_web_fetch_handler",
            agent_parameters=("url", "max_chars"),
            resolved_parameters=("validated public DNS addresses", "bounded HTTP response"),
        ),
        evidence_lifetime=EvidenceLifetimeContract(
            scope="external_content_snapshot",
            invalidated_by=("source web content may change independently",),
            reusable_across_packet_refresh=True,
        ),
        gate=GateContract(
            checks=(
                "URL is absolute public HTTPS without credentials",
                "all resolved addresses are public",
                "redirects, authentication, non-text and oversized responses are rejected",
                "extracted text is bounded and labeled untrusted external content",
            ),
            fail_closed=True,
        ),
        source_paths=("agent/runtime/planner.py", "agent/tools/web_access.py"),
    )


def _packet_camera_request(*, extra: JsonDict | None = None) -> JsonDict:
    return _object(
        {
            "source_packet_id": _string(minLength=1),
            "camera_frame_id": _string(minLength=1),
            **dict(extra or {}),
        },
        required=("source_packet_id",),
    )


def _estimate_depth_prior(spec: ToolSpecLike) -> ToolContract:
    return _explicit_contract(
        spec,
        request_schema=_packet_camera_request(
            extra={"resolution_level": _integer(minimum=0, maximum=9)}
        ),
        consumes=(
            _fact(OBSERVATION_PACKET, "request.source_packet_id", FactAuthority.HOST),
        ),
        outcomes=(
            OutcomeContract(
                "completed",
                True,
                _outputs(
                    ("prior_depth", "source_packet_id", "camera_frame_id"),
                    {
                        "prior_depth": _string(),
                        "prior_confidence": _string(),
                        "prior_confidence_semantics": _string(),
                        "source_packet_id": _string(),
                        "camera_frame_id": _string(),
                        "request_ref": _string(),
                        "raw_output_ref": _string(),
                    },
                ),
                produces=(
                    _fact(DEPTH_PRIOR, "outputs.prior_depth", FactAuthority.TOOL),
                ),
            ),
            _operational_failure(),
        ),
        host_resolution=HostResolutionContract(
            mode="opaque_observation_packet",
            resolver="ActionPipeline depth packet resolver",
            agent_parameters=("source_packet_id", "camera_frame_id", "resolution_level"),
            resolved_parameters=("rgb path", "intrinsics", "camera metadata", "bundle id"),
            freshness_dimensions=("object_scene_epoch", "robot_motion_epoch"),
            invalidated_by=("unknown packet or camera frame",),
        ),
        evidence_lifetime=EvidenceLifetimeContract(
            scope="source_packet_bound_artifact",
            freshness_dimensions=("source packet id", "camera frame id"),
            invalidated_by=("artifact deletion",),
            reusable_across_packet_refresh=True,
            description="A later packet does not erase the prior; enhance_depth matches exact provenance.",
        ),
        gate=GateContract(
            checks=(
                "source packet and camera frame resolve",
                "resolution level is in [0, 10)",
                "returned metric depth is a numeric HxW artifact",
                "confidence semantics are explicit",
            ),
            fail_closed=True,
        ),
        maturity=ContractMaturity.VERIFIED,
        coverage_gaps=(),
        source_paths=("agent/runtime/planner.py", "agent/runtime/pipeline.py", "agent/tools/handlers.py"),
    )


def _enhance_depth(spec: ToolSpecLike) -> ToolContract:
    output_properties = {
        "enabled": _boolean(),
        "reason": _string(),
        "source_packet_id": _string(),
        "camera_frame_id": _string(),
        "fused_depth_npy": _string(),
        "fused_depth_png": _string(),
        "safety_depth_npy": _string(),
        "safety_depth_png": _string(),
        "point_cloud_npz": _string(),
        "report_path": _string(),
        "quality": _object(additional_properties=True),
        "alignment": _object(additional_properties=True),
    }
    return _explicit_contract(
        spec,
        request_schema=_packet_camera_request(
            extra={"config": _object(additional_properties=True)}
        ),
        consumes=(
            _fact(OBSERVATION_PACKET, "request.source_packet_id", FactAuthority.HOST),
            _fact(
                DEPTH_PRIOR,
                "host.latest_matching_depth_prior",
                FactAuthority.HOST,
                required=False,
                when="a prior exists for the exact packet/camera source",
            ),
        ),
        outcomes=(
            OutcomeContract(
                "depth_enhanced",
                True,
                _outputs(("enabled", "fused_depth_npy", "report_path"), output_properties),
                produces=(
                    _fact(ENHANCED_DEPTH, "outputs.fused_depth_npy", FactAuthority.TOOL),
                ),
            ),
            OutcomeContract(
                "requires_depth_alignment_repair",
                True,
                _outputs(("enabled", "reason", "report_path"), output_properties),
                produces=(
                    _fact(ENHANCED_DEPTH, "outputs.safety_depth_npy", FactAuthority.TOOL),
                ),
                recovery_required=True,
                description="Sensor-only safety artifacts remain available; model fusion is not trusted.",
            ),
            _operational_failure(),
        ),
        host_resolution=HostResolutionContract(
            mode="opaque_observation_packet_plus_optional_prior",
            resolver="ActionPipeline depth enhancement resolver",
            agent_parameters=("source_packet_id", "camera_frame_id", "config"),
            resolved_parameters=(
                "aligned RGB-D paths",
                "intrinsics and calibration",
                "timestamps and epochs",
                "latest exact-source depth prior",
            ),
            freshness_dimensions=("source packet id", "camera frame id"),
            invalidated_by=("RGB-D/prior provenance mismatch",),
        ),
        evidence_lifetime=EvidenceLifetimeContract(
            scope="source_packet_bound_artifact",
            freshness_dimensions=("source packet id", "camera frame id"),
            invalidated_by=("artifact deletion",),
            reusable_across_packet_refresh=True,
        ),
        gate=GateContract(
            checks=(
                "RGB and sensor depth are aligned and share packet provenance",
                "intrinsics and scale are complete",
                "optional prior matches the same source",
                "sensor depth remains the hard safety constraint",
            ),
            fail_closed=True,
        ),
        source_paths=("agent/runtime/runtime.py", "agent/runtime/pipeline.py", "agent/runtime/depth_enhancement.py"),
    )


def _retrieve_asset_reference(spec: ToolSpecLike) -> ToolContract:
    return _explicit_contract(
        spec,
        request_schema=_object(
            {
                "environment": _string(minLength=1),
                "target_object": _string(minLength=1),
                "source_packet_id": _string(minLength=1),
                "camera_frame_id": _string(minLength=1),
            },
            required=("environment", "target_object", "source_packet_id"),
        ),
        consumes=(
            _fact(OBSERVATION_PACKET, "request.source_packet_id", FactAuthority.HOST),
        ),
        outcomes=(
            OutcomeContract(
                "completed",
                True,
                _outputs(
                    ("positive_points", "localization_bundle"),
                    {
                        "positive_points": _array(_object(additional_properties=True), minItems=1),
                        "bbox_xyxy": _array(_number(), minItems=4, maxItems=4),
                        "localization_bundle": _object(additional_properties=True),
                        "reference_images": _array(_string()),
                        "marked_scene_image": _string(),
                    },
                ),
                produces=(
                    _fact(ASSET_REFERENCE_GROUNDING, "outputs.localization_bundle", FactAuthority.HOST),
                ),
            ),
            OutcomeContract(
                "reference_service_unavailable",
                False,
                _object(additional_properties=True),
                diagnostics_required=True,
                recovery_required=True,
            ),
            _operational_failure(),
        ),
        host_resolution=HostResolutionContract(
            mode="object_memory_plus_packet",
            resolver="object-memory reference handler and packet resolver",
            agent_parameters=("environment", "target_object", "source_packet_id", "camera_frame_id"),
            resolved_parameters=("scene image", "ranked canonical references", "localizer inputs"),
            freshness_dimensions=("source packet id",),
            invalidated_by=("unknown packet", "ambiguous/low-confidence memory resolution"),
        ),
        evidence_lifetime=EvidenceLifetimeContract(
            scope="same_view_grounding_seed",
            freshness_dimensions=("source packet id", "camera frame id"),
            invalidated_by=("view change for pixel coordinates",),
            reusable_across_packet_refresh=False,
        ),
        gate=GateContract(
            checks=(
                "target_object contains identity/appearance rather than scene relation",
                "object-memory resolution is sufficiently confident",
                "returned pixel seed is in original-image coordinates",
                "downstream SAM3 confirmation remains required",
            ),
            fail_closed=True,
        ),
        source_paths=("agent/tools/asset_references.py", "agent/tools/object_memory.py", "agent/runtime/pipeline.py"),
    )


def _molmopoint(spec: ToolSpecLike) -> ToolContract:
    source = _object(
        {
            "source_packet_id": _string(minLength=1),
            "camera_frame_id": _string(minLength=1),
        },
        required=("source_packet_id",),
    )
    return _explicit_contract(
        spec,
        request_schema=_object(
            {
                "sources": _array(source, minItems=1, maxItems=4),
                "prompt": _string(minLength=1, maxLength=1024),
            },
            required=("sources", "prompt"),
        ),
        consumes=(
            _fact(OBSERVATION_PACKET, "request.sources[].source_packet_id", FactAuthority.HOST),
        ),
        outcomes=(
            OutcomeContract(
                "completed",
                True,
                _outputs(
                    ("image_count", "point_count", "points", "coordinate_convention"),
                    {
                        "image_count": _integer(minimum=1, maximum=4),
                        "point_count": _integer(minimum=0),
                        "points": _array(_object(additional_properties=True)),
                        "coordinate_convention": _object(additional_properties=True),
                        "source_packet_ids": _array(_string()),
                    },
                ),
                produces=(
                    _fact(POINT_GROUNDING_SET, "outputs.points", FactAuthority.TOOL),
                ),
            ),
            _operational_failure(),
        ),
        host_resolution=HostResolutionContract(
            mode="ordered_packet_references",
            resolver="ActionPipeline MolmoPoint source resolver",
            agent_parameters=("sources", "prompt"),
            resolved_parameters=("ordered local images", "ordered source observation metadata"),
            freshness_dimensions=("each source packet id", "each camera frame id"),
            invalidated_by=("unknown packet/frame",),
        ),
        evidence_lifetime=EvidenceLifetimeContract(
            scope="ordered_view_pixel_grounding",
            freshness_dimensions=("source packet ids", "camera frame ids"),
            invalidated_by=("using points on a different image ordering or view",),
            reusable_across_packet_refresh=False,
        ),
        gate=GateContract(
            checks=(
                "one to four ordered sources resolve",
                "prompt is concrete and bounded",
                "returned image_index uses the supplied zero-based ordering",
                "pixels lie inside their original images",
            ),
            fail_closed=True,
        ),
        source_paths=("agent/runtime/planner.py", "agent/runtime/pipeline.py", "agent/tools/handlers.py"),
    )


def _reject_sam3(spec: ToolSpecLike) -> ToolContract:
    return _explicit_contract(
        spec,
        request_schema=_object(
            {"sam3_result_id": _string(minLength=1), "reason": _string(minLength=1)},
            required=("sam3_result_id", "reason"),
        ),
        consumes=(
            _fact(SAM3_DETECTION_SET, "request.sam3_result_id", FactAuthority.TOOL),
        ),
        outcomes=(
            OutcomeContract(
                "completed",
                True,
                _outputs(("rejection",), {"rejection": _object(additional_properties=True)}),
                produces=(
                    _fact(SAM3_REJECTION, "outputs.rejection", FactAuthority.HOST),
                ),
            ),
            _operational_failure(),
        ),
        host_resolution=HostResolutionContract(
            mode="pending_result_lookup",
            resolver="AgentMemory.reject_sam3_detections",
            agent_parameters=("sam3_result_id", "reason"),
            resolved_parameters=("pending candidate set", "source packet and evidence role"),
            freshness_dimensions=("object_scene_epoch",),
            invalidated_by=("result already resolved", "object scene change"),
        ),
        evidence_lifetime=EvidenceLifetimeContract(
            scope="grounding_failure_evidence",
            freshness_dimensions=("source packet id", "evidence role"),
            invalidated_by=("successful new grounding for the same role",),
            reusable_across_packet_refresh=True,
        ),
        gate=GateContract(
            checks=("result id is the exact pending SAM3 result", "reason is non-empty"),
            fail_closed=True,
        ),
        source_paths=("agent/runtime/runtime.py", "agent/runtime/memory.py"),
    )


def _sam3(spec: ToolSpecLike) -> ToolContract:
    point = _object(
        {"x": _number(minimum=0), "y": _number(minimum=0), "label": _integer(enum=[0, 1])},
        required=("x", "y", "label"),
    )
    request = _object(
        {
            "source_packet_id": _string(minLength=1),
            "camera_frame_id": _string(minLength=1),
            "mode": _string(enum=["text", "points"]),
            "prompt": _string(minLength=1),
            "points": _array(point, minItems=1, maxItems=64),
            "positive_points": _array(point, minItems=1, maxItems=64),
            "roi_bbox_xyxy": _array(_number(), minItems=4, maxItems=4),
            "evidence_role": _string(enum=["target_object", "placement_region"]),
        },
        required=("source_packet_id",),
        one_of=[
            {"required": ["prompt"], "properties": {"mode": {"enum": ["text"]}}},
            {"required": ["points"], "properties": {"mode": {"enum": ["points"]}}},
            {"required": ["positive_points"]},
        ],
    )
    common = {
        "result_id": _string(),
        "source_packet_id": _string(),
        "detections": _array(_object(additional_properties=True)),
        "detection_count": _integer(minimum=0),
    }
    return _explicit_contract(
        spec,
        request_schema=request,
        consumes=(
            _fact(OBSERVATION_PACKET, "request.source_packet_id", FactAuthority.HOST),
            _fact(
                ASSET_REFERENCE_GROUNDING,
                "request.positive_points",
                FactAuthority.HOST,
                required=False,
                when="asset-reference point grounding is used",
            ),
            _fact(
                POINT_GROUNDING_SET,
                "request.points",
                FactAuthority.TOOL,
                required=False,
                when="MolmoPoint grounding is converted to a SAM3 point prompt",
            ),
        ),
        outcomes=(
            OutcomeContract(
                "detections_available",
                True,
                _outputs(("result_id", "detections"), common),
                produces=(
                    _fact(
                        SAM3_DETECTION_SET,
                        "outputs.result_id",
                        FactAuthority.TOOL,
                    ),
                ),
                description="One or more candidates require explicit semantic selection.",
            ),
            OutcomeContract(
                "no_detection",
                True,
                _outputs(
                    ("detections", "detection_count"),
                    {
                        **common,
                        "same_view_recovery_handoff": _object(additional_properties=True),
                    },
                ),
                recovery_required=True,
                description="The tool ran correctly but found no target candidate.",
            ),
            _operational_failure(),
        ),
        host_resolution=HostResolutionContract(
            mode="opaque_reference",
            resolver="ActionPipeline observation packet resolver",
            agent_parameters=("source_packet_id", "camera_frame_id"),
            resolved_parameters=("image", "source_observation", "camera metadata"),
            freshness_dimensions=("object_scene_epoch", "robot_motion_epoch"),
            invalidated_by=("unknown packet id", "camera frame absent from packet"),
            description="Agent supplies IDs only; local image paths remain host-private.",
        ),
        evidence_lifetime=EvidenceLifetimeContract(
            scope="pending_selection",
            freshness_dimensions=("object_scene_epoch", "robot_motion_epoch"),
            invalidated_by=("selection or rejection of this result", "object scene change"),
            reusable_across_packet_refresh=False,
        ),
        gate=GateContract(
            checks=(
                "source packet exists in this session",
                "camera frame belongs to source packet",
                "text, point, and ROI prompts obey mutual-exclusion rules",
            ),
            fail_closed=True,
            description="Rejects ungrounded transport paths and malformed prompts.",
        ),
        source_paths=("agent/runtime/planner.py", "agent/tools/handlers.py", "agent/runtime/memory.py"),
    )


def _select_sam3(spec: ToolSpecLike) -> ToolContract:
    request = _object(
        {
            "sam3_result_id": _string(minLength=1),
            "detection_id": _string(minLength=1),
            "selection_confidence": _number(minimum=0, maximum=1),
            "reason": _string(),
            "identity_anchor_id": _string(),
            "identity_relation": _string(
                enum=["same_instance", "replace_misidentified_anchor"]
            ),
            "evidence_role": _string(enum=["target_object", "placement_region"]),
            "target_geometry_family": _string(),
        },
        required=("sam3_result_id", "detection_id"),
    )
    return _explicit_contract(
        spec,
        request_schema=request,
        consumes=(
            _fact(SAM3_DETECTION_SET, "request.sam3_result_id", FactAuthority.TOOL),
        ),
        outcomes=(
            OutcomeContract(
                "completed",
                True,
                _outputs(
                    ("selected_detection",),
                    {
                        "selected_detection": _object(additional_properties=True),
                        "grasp_pose_estimate_handoff": _object(additional_properties=True),
                    },
                ),
                produces=(
                    _fact(
                        SELECTED_TARGET_MASK,
                        "outputs.selected_detection",
                        FactAuthority.HOST,
                        when="evidence_role is target_object",
                    ),
                    _fact(
                        GRASP_INPUT_BUNDLE,
                        "host_resolved_inputs.grasp_pose_estimate.bundle_id",
                        FactAuthority.HOST,
                        when="selected target has complete aligned RGB-D provenance",
                    ),
                ),
            ),
            _operational_failure(),
        ),
        host_resolution=HostResolutionContract(
            mode="evidence_graph_lookup",
            resolver="AgentMemory.resolve_sam3_selection",
            agent_parameters=("sam3_result_id", "detection_id"),
            resolved_parameters=("pending candidate", "source packet", "identity anchor"),
            freshness_dimensions=("object_scene_epoch",),
            invalidated_by=("result already resolved", "object scene change"),
        ),
        evidence_lifetime=EvidenceLifetimeContract(
            scope="active_semantic_evidence",
            freshness_dimensions=("object_scene_epoch",),
            invalidated_by=("new selection for same evidence role", "object scene change"),
            reusable_across_packet_refresh=True,
        ),
        gate=GateContract(
            checks=(
                "result id is the exact pending SAM3 result",
                "detection id belongs to that result",
                "evidence role matches pending result",
                "target identity relation is host-verifiable",
            ),
            fail_closed=True,
            description="A failed selection returns the pending IDs and an actionable repair.",
        ),
        source_paths=("agent/runtime/memory.py", "agent/runtime/pipeline.py"),
    )


def _grasp_pose_estimate(spec: ToolSpecLike) -> ToolContract:
    request = _object(
        {
            "bundle_id": _string(minLength=1),
            "backend_preference": _array(
                _string(enum=["anygrasp", "graspgenx"]),
                minItems=1,
                uniqueItems=True,
            ),
        },
        required=("bundle_id",),
    )
    return _explicit_contract(
        spec,
        request_schema=request,
        consumes=(
            _fact(GRASP_INPUT_BUNDLE, "request.bundle_id", FactAuthority.HOST),
        ),
        outcomes=(
            OutcomeContract(
                "candidates_available",
                True,
                _outputs(
                    ("schema_version", "result_id", "grasp_candidates"),
                    {
                        "schema_version": _string(enum=["openeta.grasp_pose_estimate.v1"]),
                        "result_id": _string(),
                        "grasp_candidates": _array(_object(additional_properties=True), minItems=1),
                        "selected_backend": _string(),
                        "complete_outputs_artifact": _object(additional_properties=True),
                    },
                ),
                produces=(
                    _fact(GRASP_CANDIDATE_SET, "outputs.result_id", FactAuthority.TOOL),
                ),
            ),
            OutcomeContract(
                "no_executable_grasp_candidates",
                False,
                _object(additional_properties=True),
                diagnostics_required=True,
                recovery_required=True,
            ),
            _operational_failure(),
        ),
        host_resolution=HostResolutionContract(
            mode="opaque_bundle",
            resolver="AgentMemory.resolve_grasp_input_bundle",
            agent_parameters=("bundle_id", "backend_preference"),
            resolved_parameters=(
                "mode",
                "rgb",
                "depth",
                "object_mask",
                "intrinsics",
                "camera_frame_id",
                "scene_epoch",
                "hints",
            ),
            freshness_dimensions=("object_scene_epoch",),
            invalidated_by=("target evidence superseded", "object scene change"),
        ),
        evidence_lifetime=EvidenceLifetimeContract(
            scope="immutable_result",
            freshness_dimensions=("object_scene_epoch",),
            invalidated_by=(),
            reusable_across_packet_refresh=True,
            description="The result remains queryable; execution gates assess current provenance.",
        ),
        gate=GateContract(
            checks=(
                "bundle exists and is ready",
                "bundle target mask and RGB-D share provenance",
                "bundle object-scene epoch is current",
                "backend preference is a unique supported subset",
            ),
            fail_closed=True,
        ),
        source_paths=("agent/runtime/planner.py", "agent/runtime/memory.py", "agent/tools/handlers.py"),
    )


def _compile_grasp_seed(spec: ToolSpecLike) -> ToolContract:
    request = _object(
        {
            "grasp_result_id": _string(minLength=1),
            "candidate_id": _string(minLength=1),
            "target_geometry_family": _string(),
            "target_class": _string(),
            "strategy_id": _string(),
            "articulated_handle_options": _object(additional_properties=False),
            "pregrasp_distance_m": _number(minimum=0.04, maximum=0.16),
        },
        required=("grasp_result_id", "candidate_id"),
    )
    return _explicit_contract(
        spec,
        request_schema=request,
        consumes=(
            _fact(GRASP_CANDIDATE_SET, "request.grasp_result_id", FactAuthority.TOOL),
        ),
        outcomes=(
            OutcomeContract(
                "completed",
                True,
                _outputs(
                    ("schema_version", "compiled_grasp_id", "candidate_id"),
                    {
                        "schema_version": _string(enum=[COMPILED_GRASP]),
                        "compiled_grasp_id": _string(),
                        "candidate_id": _string(),
                        "contact_pose": _object(additional_properties=True),
                        "hover_pose": _object(additional_properties=True),
                    },
                ),
                produces=(
                    _fact(COMPILED_GRASP, "outputs.compiled_grasp_id", FactAuthority.HOST),
                ),
            ),
            _operational_failure(),
        ),
        host_resolution=HostResolutionContract(
            mode="evidence_graph_lookup",
            resolver="AgentMemory.resolve_grasp_candidate_input",
            agent_parameters=("grasp_result_id", "candidate_id"),
            resolved_parameters=("camera_pose", "source observation", "camera calibration"),
            freshness_dimensions=("object_scene_epoch",),
            invalidated_by=("candidate missing from result", "stale object-scene epoch"),
        ),
        evidence_lifetime=EvidenceLifetimeContract(
            scope="provenance_branch",
            freshness_dimensions=("object_scene_epoch",),
            invalidated_by=("target evidence superseded for contact use",),
            reusable_across_packet_refresh=True,
        ),
        gate=GateContract(
            checks=(
                "result and candidate ids resolve together",
                "candidate provenance is session-owned",
                "calibration and source packet are available",
                "requested geometry options stay within calibrated bounds",
            ),
            fail_closed=True,
        ),
        source_paths=("agent/runtime/memory.py", "agent/tools/grasp_geometry.py"),
    )


def _propose_wrist_viewpoints(spec: ToolSpecLike) -> ToolContract:
    request = _object(
        {
            "compiled_grasp_id": _string(minLength=1),
            "source_packet_id": _string(minLength=1),
            "camera_frame_id": _string(minLength=1),
        },
        required=("compiled_grasp_id", "source_packet_id", "camera_frame_id"),
    )
    return _explicit_contract(
        spec,
        request_schema=request,
        consumes=(
            _fact(COMPILED_GRASP, "request.compiled_grasp_id", FactAuthority.HOST),
            _fact(OBSERVATION_PACKET, "request.source_packet_id", FactAuthority.HOST),
        ),
        outcomes=(
            OutcomeContract(
                "completed",
                True,
                _outputs(
                    ("schema_version", "proposal_id", "candidates"),
                    {
                        "schema_version": _string(enum=[WRIST_VIEWPOINT_PROPOSAL]),
                        "proposal_id": _string(),
                        "candidates": _array(_object(additional_properties=True), minItems=1),
                    },
                ),
                produces=(
                    _fact(
                        WRIST_VIEWPOINT_PROPOSAL,
                        "outputs.proposal_id",
                        FactAuthority.HOST,
                    ),
                ),
                executable_reference=False,
            ),
            _operational_failure(),
        ),
        host_resolution=HostResolutionContract(
            mode="multi_reference",
            resolver="AgentMemory.resolve_wrist_viewpoint_input",
            agent_parameters=("compiled_grasp_id", "source_packet_id", "camera_frame_id"),
            resolved_parameters=("compiled geometry", "measured EEF", "camera extrinsics"),
            freshness_dimensions=("robot_motion_epoch", "object_scene_epoch"),
            invalidated_by=("stale packet", "unknown compiled grasp", "non-wrist camera"),
        ),
        evidence_lifetime=EvidenceLifetimeContract(
            scope="robot_pose_bound_proposal",
            freshness_dimensions=("robot_motion_epoch", "object_scene_epoch"),
            invalidated_by=("robot motion", "object scene change"),
            reusable_across_packet_refresh=True,
        ),
        gate=GateContract(
            checks=("compiled grasp exists", "wrist packet is fresh", "camera calibration exists"),
            fail_closed=True,
        ),
        source_paths=("agent/runtime/memory.py", "agent/tools/grasp_geometry.py"),
    )


def _compute_wrist_alignment(spec: ToolSpecLike) -> ToolContract:
    request = _object(
        {"bundle_id": _string(minLength=1), "max_correction_m": _number(minimum=0.005, maximum=0.05)},
        required=("bundle_id",),
    )
    return _explicit_contract(
        spec,
        request_schema=request,
        consumes=(
            _fact(WRIST_ALIGNMENT_BUNDLE, "request.bundle_id", FactAuthority.HOST),
        ),
        outcomes=(
            OutcomeContract(
                "completed",
                True,
                _outputs(
                    ("adjusted_contact_pose",),
                    {
                        "adjusted_contact_pose": _object(additional_properties=True),
                        "aligned_hover_pose": _object(additional_properties=True),
                        "adjusted_precontact_pose": _object(additional_properties=True),
                        "correction_world_xyz": _array(_number(), minItems=3, maxItems=3),
                    },
                ),
                produces=(
                    _fact(
                        ALIGNED_GRASP_REFERENCE,
                        "outputs.adjusted_contact_pose",
                        FactAuthority.HOST,
                    ),
                ),
                executable_reference=True,
            ),
            OutcomeContract(
                "requires_better_view",
                True,
                _outputs(("status",), {"status": _string(enum=["requires_better_view"])}),
                diagnostics_required=True,
                recovery_required=True,
                executable_reference=False,
            ),
            _operational_failure(),
        ),
        host_resolution=HostResolutionContract(
            mode="opaque_bundle",
            resolver="AgentMemory.resolve_wrist_alignment_bundle",
            agent_parameters=("bundle_id", "max_correction_m"),
            resolved_parameters=(
                "wrist mask",
                "aligned depth",
                "camera calibration",
                "measured EEF pose",
                "compiled grasp",
            ),
            freshness_dimensions=("robot_motion_epoch", "object_scene_epoch"),
            invalidated_by=("robot motion", "object scene change", "target selection change"),
        ),
        evidence_lifetime=EvidenceLifetimeContract(
            scope="current_geometry",
            freshness_dimensions=("robot_motion_epoch", "object_scene_epoch"),
            invalidated_by=("robot motion", "object scene change"),
            reusable_across_packet_refresh=False,
        ),
        gate=GateContract(
            checks=(
                "bundle exists and epochs are current",
                "mask is not clipped beyond operating limits",
                "measured EEF is near compiled clearance reference",
                "correction remains inside residual adjustment budget",
            ),
            fail_closed=True,
            description="Out-of-region geometry returns requires_better_view without executable poses.",
        ),
        source_paths=("agent/runtime/memory.py", "agent/tools/grasp_geometry.py"),
    )


def _ik_preview_check(spec: ToolSpecLike) -> ToolContract:
    pose = _object(additional_properties=True)
    request = _object(
        {
            "target_pose": pose,
            "compiled_grasp_id": _string(minLength=1),
            "waypoint_role": _string(
                enum=["grasp_clearance", "grasp_precontact", "grasp_alignment_reference", "grasp_contact"]
            ),
            "path_fraction": _number(exclusiveMinimum=0, exclusiveMaximum=1),
            "viewpoint_proposal_id": _string(minLength=1),
            "candidate_id": _string(minLength=1),
            "probe_id": _string(minLength=1),
            "waypoint_index": _integer(minimum=0, maximum=4),
            "position_tolerance_m": _number(exclusiveMinimum=0),
            "orientation_tolerance_rad": _number(exclusiveMinimum=0),
            "preserve_current_orientation": _boolean(),
            "check_endpoint_collision": _boolean(),
        },
        one_of=[
            {"required": ["target_pose"]},
            {"required": ["compiled_grasp_id", "waypoint_role"]},
            {"required": ["compiled_grasp_id", "path_fraction"]},
            {"required": ["viewpoint_proposal_id", "candidate_id"]},
            {"required": ["probe_id", "waypoint_index"]},
        ],
    )
    executable = _fact(
        IK_EXECUTION_AUTHORIZATION,
        "outputs.motion_execution_ref.ik_receipt_id",
        FactAuthority.HOST,
    )
    return _explicit_contract(
        spec,
        request_schema=request,
        consumes=(
            _fact(
                COMPILED_GRASP,
                "request.compiled_grasp_id",
                FactAuthority.HOST,
                required=False,
                when="compiled-grasp reference branch is used",
            ),
            _fact(
                WRIST_VIEWPOINT_PROPOSAL,
                "request.viewpoint_proposal_id",
                FactAuthority.HOST,
                required=False,
                when="wrist-viewpoint reference branch is used",
            ),
            _fact(
                ALIGNED_GRASP_REFERENCE,
                "request.target_pose",
                FactAuthority.HOST,
                required=False,
                when="an aligned reference is previewed",
            ),
            _fact(
                ATTACHMENT_PROBE_PLAN,
                "request.probe_id",
                FactAuthority.HOST,
                required=False,
                when="an attachment-probe waypoint reference branch is used",
            ),
        ),
        outcomes=(
            OutcomeContract(
                "ik_feasible",
                True,
                _outputs(
                    ("ik_preview_receipt", "ik_receipt_id", "motion_execution_ref"),
                    {
                        "ik_preview_receipt": _object(additional_properties=True),
                        "ik_receipt_id": _string(),
                        "motion_execution_ref": _object(additional_properties=True),
                    },
                ),
                produces=(executable,),
                executable_reference=True,
            ),
            OutcomeContract(
                "ik_kinematically_feasible_collision_deferred",
                True,
                _outputs(("ik_preview_receipt", "ik_receipt_id", "motion_execution_ref"), additional_ik_properties()),
                produces=(executable,),
                executable_reference=True,
                diagnostics_required=True,
            ),
            OutcomeContract(
                "ik_repairable",
                True,
                _outputs(("ik_preview_receipt", "ik_receipt_id"), additional_ik_properties()),
                executable_reference=False,
                diagnostics_required=True,
                recovery_required=True,
            ),
            OutcomeContract(
                "ik_inconclusive",
                True,
                _outputs(("ik_preview_receipt", "ik_receipt_id"), additional_ik_properties()),
                executable_reference=False,
                diagnostics_required=True,
                recovery_required=True,
            ),
            OutcomeContract(
                "ik_hard_infeasible",
                True,
                _outputs(("ik_preview_receipt", "ik_receipt_id"), additional_ik_properties()),
                executable_reference=False,
                diagnostics_required=True,
                recovery_required=True,
            ),
            _operational_failure(),
        ),
        host_resolution=HostResolutionContract(
            mode="exclusive_reference_or_agent_pose",
            resolver="ActionPipeline IK reference resolver",
            agent_parameters=(
                "target_pose",
                "compiled_grasp_id + waypoint_role",
                "compiled_grasp_id + path_fraction",
                "viewpoint_proposal_id + candidate_id",
                "probe_id + waypoint_index",
            ),
            resolved_parameters=("target_pose", "orientation policy", "private IK seed", "provenance"),
            freshness_dimensions=("robot_motion_epoch", "object_scene_epoch"),
            invalidated_by=("unknown reference", "stale reference", "reference branch mismatch"),
        ),
        evidence_lifetime=EvidenceLifetimeContract(
            scope="exact_pose_policy_authorization",
            freshness_dimensions=("robot_motion_epoch", "object_scene_epoch"),
            invalidated_by=("robot motion", "object scene change", "pose/policy/tolerance change"),
            reusable_across_packet_refresh=False,
        ),
        gate=GateContract(
            checks=(
                "exactly one target source branch is supplied",
                "opaque references resolve without model-authored overrides",
                "classification supports execution before issuing motion_execution_ref",
                "receipt binds pose, orientation policy, tolerances, and epochs",
            ),
            fail_closed=True,
            description="Infeasible previews remain useful diagnostics but never authorize that exact pose policy.",
        ),
        source_paths=("agent/runtime/pipeline.py", "agent/runtime/memory.py", "agent/tools/sim_mcp.py"),
    )


def additional_ik_properties() -> JsonDict:
    return {
        "ik_preview_receipt": _object(additional_properties=True),
        "ik_receipt_id": _string(),
        "motion_execution_ref": _object(additional_properties=True),
        "diagnostics": _array(_object(additional_properties=True)),
    }


def _prepare_attachment_probe(spec: ToolSpecLike) -> ToolContract:
    vector = _array(_number(), minItems=3, maxItems=3)
    return _explicit_contract(
        spec,
        request_schema=_object(
            {
                "compiled_grasp_id": _string(minLength=1),
                "motion_type": _string(enum=["linear", "arc"]),
                "direction_world_xyz": vector,
                "waypoint_offsets_world_xyz": _array(vector, minItems=2, maxItems=5),
                "reason": _string(maxLength=1024),
            },
            required=("compiled_grasp_id", "motion_type"),
            one_of=[
                {
                    "required": ["direction_world_xyz"],
                    "properties": {"motion_type": {"enum": ["linear"]}},
                },
                {
                    "required": ["waypoint_offsets_world_xyz"],
                    "properties": {"motion_type": {"enum": ["arc"]}},
                },
            ],
        ),
        consumes=(
            _fact(COMPILED_GRASP, "request.compiled_grasp_id", FactAuthority.HOST),
            _fact(
                OBSERVATION_PACKET,
                "host.current_observation",
                FactAuthority.HOST,
                required=False,
                description="Current dual-view images and measured EEF pose are injected by the host.",
            ),
        ),
        outcomes=(
            OutcomeContract(
                "completed",
                True,
                _outputs(
                    ("schema_version", "probe_id", "ik_preview_requests", "execution_handoff"),
                    {
                        "schema_version": _string(enum=[ATTACHMENT_PROBE_PLAN]),
                        "probe_id": _string(),
                        "path_sha256": _string(),
                        "frozen_path": _array(_object(additional_properties=True)),
                        "ik_preview_requests": _array(_object(additional_properties=True), minItems=1),
                        "execution_handoff": _object(additional_properties=True),
                    },
                ),
                produces=(
                    _fact(ATTACHMENT_PROBE_PLAN, "outputs.probe_id", FactAuthority.HOST),
                ),
                executable_reference=False,
                description="The plan contains preview requests, not direct motion authority.",
            ),
            _operational_failure(),
        ),
        host_resolution=HostResolutionContract(
            mode="compiled_grasp_plus_current_observation",
            resolver="prepare_attachment_probe",
            agent_parameters=(
                "compiled_grasp_id",
                "motion_type",
                "direction_world_xyz or waypoint offsets",
                "reason",
            ),
            resolved_parameters=("compiled provenance", "measured EEF pose", "two current RGB views"),
            freshness_dimensions=("scene_epoch", "compiled target evidence"),
            invalidated_by=("compiled target superseded", "probe replay", "scene change"),
        ),
        evidence_lifetime=EvidenceLifetimeContract(
            scope="single_frozen_probe",
            freshness_dimensions=("scene_epoch", "path sha256"),
            invalidated_by=("probe execution", "probe replacement", "scene change"),
            reusable_across_packet_refresh=False,
        ),
        gate=GateContract(
            checks=(
                "compiled grasp exists and target evidence is current",
                "linear direction is non-zero or arc has 2-5 bounded segments",
                "total probe length is 0.05 m within tolerance",
                "current EEF pose and exactly two required RGB views exist",
                "every frozen endpoint still requires exact IK preview",
            ),
            fail_closed=True,
        ),
        source_paths=("agent/tools/attachment_probe.py", "agent/runtime/memory.py"),
    )


def _assess_attachment_probe(spec: ToolSpecLike) -> ToolContract:
    return _explicit_contract(
        spec,
        request_schema=_object(
            {"probe_id": _string(minLength=1)},
            required=("probe_id",),
        ),
        consumes=(
            _fact(COMPLETED_ATTACHMENT_PROBE, "request.probe_id", FactAuthority.HOST),
        ),
        outcomes=(
            OutcomeContract(
                "completed",
                True,
                _outputs(
                    ("schema_version", "probe_id", "verdict", "reason"),
                    {
                        "schema_version": _string(enum=[ATTACHMENT_ASSESSMENT]),
                        "probe_id": _string(),
                        "verdict": _string(enum=["PASS", "FAIL", "UNKNOWN"]),
                        "reason": _string(),
                        "checked_by": _string(),
                    },
                ),
                produces=(
                    _fact(ATTACHMENT_ASSESSMENT, "outputs", FactAuthority.TOOL),
                ),
            ),
            _operational_failure(),
        ),
        host_resolution=HostResolutionContract(
            mode="completed_probe_lookup",
            resolver="assess_attachment_probe",
            agent_parameters=("probe_id",),
            resolved_parameters=("frozen before views", "fresh after views", "probe metadata"),
            freshness_dimensions=("completed probe id", "current after observation"),
            invalidated_by=("probe not completed", "insufficient or ambiguous views"),
        ),
        evidence_lifetime=EvidenceLifetimeContract(
            scope="independent_visual_assessment",
            invalidated_by=(),
            reusable_across_packet_refresh=True,
        ),
        gate=GateContract(
            checks=(
                "probe id is the exact completed frozen probe",
                "before and after each contain scene-primary and wrist-primary views",
                "reviewer verdict is PASS, FAIL, or UNKNOWN",
                "controller success and gripper closure are not treated as attachment proof",
            ),
            fail_closed=True,
        ),
        source_paths=("agent/tools/attachment_probe.py", "agent/runtime/memory.py"),
    )


def _move_to(spec: ToolSpecLike) -> ToolContract:
    request = _object(
        {
            "ik_receipt_id": _string(minLength=1),
            "num_steps": _integer(minimum=1),
            "tolerance": _number(exclusiveMinimum=0),
            "ori_tolerance": _number(exclusiveMinimum=0),
            "enable_collision_check": _boolean(),
        },
        required=("ik_receipt_id",),
    )
    return _motion_contract(spec, request_schema=request, trajectory=False)


def _follow_eef_trajectory(spec: ToolSpecLike) -> ToolContract:
    request = _object(
        {
            "ik_receipt_ids": _array(_string(minLength=1), minItems=1, maxItems=5),
            "num_steps_per_waypoint": _integer(minimum=1),
            "tolerance": _number(exclusiveMinimum=0),
            "ori_tolerance": _number(exclusiveMinimum=0),
            "enable_collision_check": _boolean(),
        },
        required=("ik_receipt_ids",),
    )
    return _motion_contract(spec, request_schema=request, trajectory=True)


def _motion_contract(
    spec: ToolSpecLike,
    *,
    request_schema: JsonDict,
    trajectory: bool,
) -> ToolContract:
    reference_path = "request.ik_receipt_ids[]" if trajectory else "request.ik_receipt_id"
    return _explicit_contract(
        spec,
        request_schema=request_schema,
        consumes=(
            _fact(IK_EXECUTION_AUTHORIZATION, reference_path, FactAuthority.HOST),
        ),
        outcomes=(
            OutcomeContract(
                "mutation_acknowledged",
                True,
                _outputs(
                    ("motion_summary",),
                    {
                        "motion_summary": _object(
                            {"reached_target": _boolean()},
                            required=("reached_target",),
                            additional_properties=True,
                        ),
                        "environment_receipt": _object(additional_properties=True),
                        "post_motion_evidence_handoff": _object(additional_properties=True),
                    },
                ),
                produces=(
                    _fact(MOTION_EXECUTION_RECEIPT, "outputs.motion_summary", FactAuthority.ENVIRONMENT),
                ),
            ),
            OutcomeContract(
                "target_not_reached",
                False,
                _outputs(("motion_summary",), {"motion_summary": _object(additional_properties=True)}),
                produces=(
                    _fact(MOTION_EXECUTION_RECEIPT, "outputs.motion_summary", FactAuthority.ENVIRONMENT),
                ),
                diagnostics_required=True,
                recovery_required=True,
            ),
            OutcomeContract(
                "target_already_within_tolerance",
                True,
                _outputs(
                    ("motion_summary",),
                    {"motion_summary": _object(additional_properties=True)},
                ),
                produces=(
                    _fact(
                        MOTION_EXECUTION_RECEIPT,
                        "outputs.motion_summary",
                        FactAuthority.ENVIRONMENT,
                    ),
                ),
                recovery_required=True,
                description="No meaningful world displacement occurred; inspect fresh evidence before assuming progress.",
            ),
            OutcomeContract(
                "attachment_contract_unavailable",
                True,
                _outputs(
                    ("attachment_proxy_receipt",),
                    {"attachment_proxy_receipt": _object(additional_properties=True)},
                ),
                diagnostics_required=True,
                recovery_required=True,
                description="Motion may have executed, but carried-object collision state is unknown.",
            ),
            OutcomeContract(
                "requires_attachment_confirmation",
                True,
                _outputs(
                    ("attachment_proxy_receipt",),
                    {"attachment_proxy_receipt": _object(additional_properties=True)},
                ),
                recovery_required=True,
                description="Motion completed while attachment proxy evidence remains tentative.",
            ),
            OutcomeContract(
                "no_attachment_evidence",
                True,
                _outputs(
                    ("attachment_proxy_receipt",),
                    {"attachment_proxy_receipt": _object(additional_properties=True)},
                ),
                recovery_required=True,
                description="Motion completed, but the refreshed receipt does not support a carried-object claim.",
            ),
            _operational_failure(),
        ),
        host_resolution=HostResolutionContract(
            mode="execution_receipt_lookup",
            resolver="AgentMemory.resolve_ik_execution_receipt",
            agent_parameters=(("ik_receipt_ids",) if trajectory else ("ik_receipt_id",)),
            resolved_parameters=("target pose(s)", "orientation policy", "private IK seed(s)", "provenance"),
            freshness_dimensions=("robot_motion_epoch", "object_scene_epoch"),
            invalidated_by=("unknown receipt", "stale receipt", "non-executable IK classification"),
        ),
        evidence_lifetime=EvidenceLifetimeContract(
            scope="environment_receipt",
            freshness_dimensions=("robot_motion_epoch", "object_scene_epoch"),
            invalidated_by=(),
            reusable_across_packet_refresh=True,
            description="Receipts are durable history; they do not authorize replay.",
        ),
        gate=GateContract(
            checks=(
                "receipt exists and is executable",
                "receipt pose policy and tolerances match the frozen preview",
                "receipt epochs are current",
                "compiled residual budget is respected when applicable",
                "collision capability matches deferred-collision authorization",
            ),
            fail_closed=True,
            description=(
                "Rejection returns the violated invariant, exact evidence ids, recent packets, "
                "and allowed repair calls; it never chooses the Agent's next motion."
            ),
        ),
        source_paths=("agent/runtime/pipeline.py", "agent/runtime/memory.py", "agent/tools/sim_mcp.py"),
    )


def _gripper_control(spec: ToolSpecLike) -> ToolContract:
    return _explicit_contract(
        spec,
        request_schema=_object(
            {"position": {"type": ["integer", "boolean"], "enum": [0, 1, False, True]}},
            required=("position",),
        ),
        outcomes=(
            OutcomeContract(
                "mutation_acknowledged",
                True,
                _outputs(
                    ("response",),
                    {
                        "response": _object(additional_properties=True),
                        "attachment_proxy_receipt": _object(additional_properties=True),
                    },
                ),
                produces=(
                    _fact(GRIPPER_LATCH_RECEIPT, "outputs.response", FactAuthority.ENVIRONMENT),
                ),
            ),
            OutcomeContract(
                "requires_attachment_probe",
                True,
                _outputs(
                    ("attachment_proxy_receipt",),
                    {"attachment_proxy_receipt": _object(additional_properties=True)},
                ),
                produces=(
                    _fact(
                        GRIPPER_LATCH_RECEIPT,
                        "outputs.attachment_proxy_receipt",
                        FactAuthority.ENVIRONMENT,
                    ),
                ),
                diagnostics_required=False,
                recovery_required=True,
                description="Close is latched and tentative contact is plausible, but attachment is unproven.",
            ),
            OutcomeContract(
                "no_attachment_evidence",
                True,
                _outputs(
                    ("attachment_proxy_receipt",),
                    {"attachment_proxy_receipt": _object(additional_properties=True)},
                ),
                produces=(
                    _fact(
                        GRIPPER_LATCH_RECEIPT,
                        "outputs.attachment_proxy_receipt",
                        FactAuthority.ENVIRONMENT,
                    ),
                ),
                recovery_required=True,
                description="Binary close is latched, but the receipt does not support a grasp claim.",
            ),
            OutcomeContract(
                "attachment_contract_unavailable",
                True,
                _outputs(
                    ("attachment_proxy_receipt",),
                    {"attachment_proxy_receipt": _object(additional_properties=True)},
                ),
                diagnostics_required=True,
                recovery_required=True,
            ),
            _operational_failure(),
        ),
        evidence_lifetime=EvidenceLifetimeContract(
            scope="latched_command",
            freshness_dimensions=("robot_motion_epoch", "object_scene_epoch"),
            invalidated_by=("opposite gripper command", "environment reset"),
            reusable_across_packet_refresh=True,
        ),
        gate=GateContract(
            checks=(
                "command is exactly binary",
                "contact close does not rely on superseded target provenance",
            ),
            fail_closed=True,
            description="Close receipts expose tentative attachment evidence instead of claiming grasp success.",
        ),
        source_paths=("agent/runtime/planner.py", "agent/runtime/memory.py", "agent/tools/sim_mcp.py"),
    )


def _anyplace(spec: ToolSpecLike) -> ToolContract:
    return _explicit_contract(
        spec,
        request_schema=_object({"bundle_id": _string(minLength=1)}, required=("bundle_id",)),
        consumes=(
            _fact(PLACEMENT_INPUT_BUNDLE, "request.bundle_id", FactAuthority.HOST),
        ),
        outcomes=(
            OutcomeContract(
                "candidates_available",
                True,
                _outputs(
                    ("result_id", "placement_candidates"),
                    {
                        "result_id": _string(),
                        "placement_candidates": _array(_object(additional_properties=True), minItems=5, maxItems=5),
                        "camera_pose_to_world_handoff": _object(additional_properties=True),
                    },
                ),
                produces=(
                    _fact(PLACEMENT_CANDIDATE_SET, "outputs.result_id", FactAuthority.TOOL),
                ),
            ),
            _operational_failure(),
        ),
        host_resolution=HostResolutionContract(
            mode="opaque_bundle",
            resolver="AgentMemory.resolve_anyplace_input_bundle",
            agent_parameters=("bundle_id",),
            resolved_parameters=(
                "RGB-D",
                "object mask",
                "placement region mask",
                "intrinsics",
                "selected grasp provenance",
            ),
            freshness_dimensions=("object_scene_epoch",),
            invalidated_by=("attachment evidence lost", "placement mask superseded", "bundle not ready"),
        ),
        evidence_lifetime=EvidenceLifetimeContract(
            scope="attachment_bound_plan",
            freshness_dimensions=("object_scene_epoch",),
            invalidated_by=("attachment evidence invalidated", "placement region superseded"),
            reusable_across_packet_refresh=True,
        ),
        gate=GateContract(
            checks=(
                "bundle exists and is ready",
                "bundle preserves selected grasp and mask provenance",
                "attachment-bound freshness remains valid",
            ),
            fail_closed=True,
        ),
        source_paths=("agent/runtime/planner.py", "agent/runtime/memory.py", "agent/tools/handlers.py"),
    )


def _camera_pose_to_world(spec: ToolSpecLike) -> ToolContract:
    request = _object(
        {
            "placement_result_id": _string(minLength=1),
            "candidate_id": _string(minLength=1),
            "camera_pose": _object(additional_properties=True),
            "camera_to_world": _object(additional_properties=True),
            "camera_extrinsics": _object(additional_properties=True),
            "camera_frame_id": _string(),
            "input_camera_frame": _string(),
        },
        one_of=[
            {"required": ["placement_result_id", "candidate_id"]},
            {"required": ["camera_pose"]},
        ],
    )
    return _explicit_contract(
        spec,
        request_schema=request,
        consumes=(
            _fact(
                PLACEMENT_CANDIDATE_SET,
                "request.placement_result_id",
                FactAuthority.TOOL,
                required=False,
                when="AnyPlace opaque-reference branch is used",
            ),
        ),
        outcomes=(
            OutcomeContract(
                "completed",
                True,
                _outputs(
                    ("world_pose",),
                    {
                        "world_pose": _object(additional_properties=True),
                        "placement_result_id": _string(),
                        "candidate_id": _string(),
                    },
                ),
                produces=(
                    _fact(WORLD_POSE_REFERENCE, "outputs.world_pose", FactAuthority.HOST),
                ),
            ),
            _operational_failure(),
        ),
        host_resolution=HostResolutionContract(
            mode="exclusive_placement_reference_or_explicit_geometry",
            resolver="AgentMemory.resolve_placement_candidate_input",
            agent_parameters=("placement_result_id + candidate_id", "camera_pose"),
            resolved_parameters=("camera pose", "source packet", "camera extrinsics"),
            freshness_dimensions=("object_scene_epoch",),
            invalidated_by=("unknown result/candidate pair", "missing calibration"),
        ),
        evidence_lifetime=EvidenceLifetimeContract(
            scope="placement_plan_geometry",
            freshness_dimensions=("object_scene_epoch",),
            invalidated_by=("placement plan invalidated",),
            reusable_across_packet_refresh=True,
        ),
        gate=GateContract(
            checks=(
                "opaque placement result and candidate resolve together",
                "opaque branch cannot be overridden with copied pose/calibration",
                "generic branch has a complete camera-frame pose and host-trusted calibration",
            ),
            fail_closed=True,
        ),
        source_paths=("agent/runtime/planner.py", "agent/runtime/pipeline.py", "agent/tools/handlers.py"),
    )


def _memory_namespace_schema(*, include_all: bool) -> JsonDict:
    values = ["facts", "artifacts", "skill_notes"]
    if include_all:
        values.append("all")
    return _string(enum=values)


def _evidence_references_schema() -> JsonDict:
    return _array(
        _object(
            {
                "path": _string(minLength=1),
                "split": _string(enum=["canary", "held_out"]),
            },
            required=("path", "split"),
        ),
        minItems=1,
    )


def _propose_calibration_profile(spec: ToolSpecLike) -> ToolContract:
    return _explicit_contract(
        spec,
        request_schema=_object(
            {
                "profile": _object(additional_properties=True),
                "profile_fingerprint": _object(additional_properties=True),
                "validation_gates": _array(_object(additional_properties=True)),
                "rationale": _string(minLength=1),
                "ledger": _array(_object(additional_properties=True)),
            },
            required=("profile", "profile_fingerprint", "rationale"),
        ),
        outcomes=(
            OutcomeContract(
                "completed",
                True,
                _outputs(
                    ("proposal_id", "status", "proposal_path", "profile_sha256", "review"),
                    {
                        "proposal_id": _string(),
                        "status": _string(enum=["reviewed"]),
                        "proposal_path": _string(),
                        "profile_path": _string(),
                        "profile_sha256": _string(),
                        "review": _object(additional_properties=True),
                        "next_gate": _object(additional_properties=True),
                    },
                ),
                produces=(
                    _fact(CALIBRATION_PROPOSAL, "outputs.proposal_id", FactAuthority.HOST),
                ),
            ),
            _operational_failure(),
        ),
        host_resolution=HostResolutionContract(
            mode="session_proposal_workspace",
            resolver="CalibrationLifecycleManager.propose_handler",
            agent_parameters=("profile", "fingerprint", "validation gates", "rationale", "ledger"),
            resolved_parameters=("session id", "default validation gates", "independent reviewer"),
        ),
        evidence_lifetime=EvidenceLifetimeContract(
            scope="session_local_reviewed_proposal",
            invalidated_by=("proposal workspace deletion",),
            reusable_across_packet_refresh=True,
        ),
        gate=GateContract(
            checks=(
                "profile schema and fingerprint agree",
                "deterministic validation gates pass",
                "independent reviewer approves staging",
                "proposal does not publish to shared calibration roots",
            ),
            fail_closed=True,
        ),
        source_paths=("agent/runtime/calibration.py",),
    )


def _promote_calibration_profile(spec: ToolSpecLike) -> ToolContract:
    return _explicit_contract(
        spec,
        request_schema=_object(
            {
                "proposal_id": _string(minLength=1),
                "target_status": _string(enum=["candidate", "validated"]),
                "evidence": _evidence_references_schema(),
            },
            required=("proposal_id", "target_status", "evidence"),
        ),
        consumes=(
            _fact(CALIBRATION_PROPOSAL, "request.proposal_id", FactAuthority.HOST),
        ),
        outcomes=(
            OutcomeContract(
                "completed",
                True,
                _outputs(
                    ("proposal_id", "target_status", "target_path", "authorization"),
                    {
                        "proposal_id": _string(),
                        "target_status": _string(),
                        "target_path": _string(),
                        "source_profile_sha256": _string(),
                        "published_profile_sha256": _string(),
                        "gate_report": _object(additional_properties=True),
                        "review": _object(additional_properties=True),
                        "authorization": _object(additional_properties=True),
                    },
                ),
                produces=(
                    _fact(PUBLISHED_CALIBRATION, "outputs.target_path", FactAuthority.HOST),
                ),
            ),
            _operational_failure(),
        ),
        host_resolution=HostResolutionContract(
            mode="reviewed_proposal_plus_host_evidence",
            resolver="CalibrationLifecycleManager.promote_handler",
            agent_parameters=("proposal_id", "target_status", "evidence references"),
            resolved_parameters=("proposal/profile hashes", "host-read evidence", "publication policy"),
        ),
        evidence_lifetime=EvidenceLifetimeContract(
            scope="published_profile",
            invalidated_by=("explicit later profile publication or repository removal",),
            reusable_across_packet_refresh=True,
        ),
        gate=GateContract(
            checks=(
                "proposal is reviewed and transition is legal",
                "evidence paths stay within configured roots and match profile hash",
                "canary/held-out coverage and deterministic gates pass",
                "independent review and publication authority approve",
            ),
            fail_closed=True,
        ),
        source_paths=("agent/runtime/calibration.py",),
    )


def _propose_grasp_strategy(spec: ToolSpecLike) -> ToolContract:
    return _explicit_contract(
        spec,
        request_schema=_object(
            {
                "strategy": _object(additional_properties=True),
                "base_strategy_sha256": _string(),
                "rationale": _string(minLength=1),
                "rollout_summary": _object(additional_properties=True),
                "ledger": _array(_object(additional_properties=True)),
            },
            required=("strategy", "rationale"),
        ),
        outcomes=(
            OutcomeContract(
                "completed",
                True,
                _outputs(
                    ("proposal_id", "status", "strategy_sha256", "calibration_profile_sha256"),
                    {
                        "proposal_id": _string(),
                        "status": _string(enum=["reviewed"]),
                        "strategy_id": _string(),
                        "strategy_sha256": _string(),
                        "calibration_profile_sha256": _string(),
                        "proposal_path": _string(),
                        "session_strategy_path": _string(),
                        "review": _object(additional_properties=True),
                    },
                ),
                produces=(
                    _fact(GRASP_STRATEGY_PROPOSAL, "outputs.proposal_id", FactAuthority.HOST),
                ),
            ),
            _operational_failure(),
        ),
        host_resolution=HostResolutionContract(
            mode="session_proposal_workspace",
            resolver="GraspStrategyLifecycleManager.propose_handler",
            agent_parameters=("strategy", "base hash", "rationale", "rollout summary", "ledger"),
            resolved_parameters=("configured calibration profile", "session id", "independent reviewer"),
        ),
        evidence_lifetime=EvidenceLifetimeContract(
            scope="session_local_reviewed_proposal",
            invalidated_by=("compare-and-swap conflict", "proposal workspace deletion"),
            reusable_across_packet_refresh=True,
        ),
        gate=GateContract(
            checks=(
                "strategy schema is valid against configured calibration",
                "replacement base hash matches",
                "independent reviewer approves",
                "current episode/tool contracts remain unchanged",
            ),
            fail_closed=True,
        ),
        source_paths=("agent/runtime/grasp_strategy_lifecycle.py",),
    )


def _promote_grasp_strategy(spec: ToolSpecLike) -> ToolContract:
    return _explicit_contract(
        spec,
        request_schema=_object(
            {
                "proposal_id": _string(minLength=1),
                "target_status": _string(enum=["candidate", "validated"]),
                "evidence": _evidence_references_schema(),
            },
            required=("proposal_id", "target_status", "evidence"),
        ),
        consumes=(
            _fact(GRASP_STRATEGY_PROPOSAL, "request.proposal_id", FactAuthority.HOST),
        ),
        outcomes=(
            OutcomeContract(
                "completed",
                True,
                _outputs(
                    ("proposal_id", "target_status", "target_path", "authorization"),
                    {
                        "proposal_id": _string(),
                        "strategy_id": _string(),
                        "target_status": _string(),
                        "target_path": _string(),
                        "strategy_sha256": _string(),
                        "calibration_profile_sha256": _string(),
                        "gate_report": _object(additional_properties=True),
                        "review": _object(additional_properties=True),
                        "authorization": _object(additional_properties=True),
                    },
                ),
                produces=(
                    _fact(PUBLISHED_GRASP_STRATEGY, "outputs.target_path", FactAuthority.HOST),
                ),
            ),
            _operational_failure(),
        ),
        host_resolution=HostResolutionContract(
            mode="reviewed_proposal_plus_host_evidence",
            resolver="GraspStrategyLifecycleManager.promote_handler",
            agent_parameters=("proposal_id", "target_status", "evidence references"),
            resolved_parameters=("strategy/calibration hashes", "host-read evidence", "publication policy"),
        ),
        evidence_lifetime=EvidenceLifetimeContract(
            scope="published_strategy",
            invalidated_by=("explicit later strategy publication or repository removal",),
            reusable_across_packet_refresh=True,
        ),
        gate=GateContract(
            checks=(
                "proposal is reviewed and promotion transition is legal",
                "evidence matches strategy and calibration hashes",
                "canary/held-out coverage gates pass",
                "independent review and publication authority approve",
            ),
            fail_closed=True,
        ),
        source_paths=("agent/runtime/grasp_strategy_lifecycle.py",),
    )


def _save_memory(spec: ToolSpecLike) -> ToolContract:
    return _explicit_contract(
        spec,
        request_schema=_object(
            {
                "namespace": _memory_namespace_schema(include_all=False),
                "key": _string(minLength=1),
                "content": {},
                "tags": _array(_string()),
            },
            required=("key", "content"),
        ),
        outcomes=(
            OutcomeContract(
                "completed",
                True,
                _outputs(
                    ("namespace", "key"),
                    {"namespace": _string(), "key": _string()},
                ),
                produces=(
                    _fact(MEMORY_ENTRY, "outputs.key", FactAuthority.HOST),
                ),
            ),
            _operational_failure(),
        ),
        evidence_lifetime=EvidenceLifetimeContract(
            scope="durable_session_memory",
            invalidated_by=("delete_memory for the same namespace/key", "session cleanup"),
            reusable_across_packet_refresh=True,
        ),
        gate=GateContract(
            checks=("key is non-empty", "content is non-empty", "namespace is supported"),
            fail_closed=True,
        ),
        source_paths=("agent/runtime/runtime.py", "agent/runtime/memory.py"),
    )


def _get_memory(spec: ToolSpecLike) -> ToolContract:
    return _explicit_contract(
        spec,
        request_schema=_object(
            {
                "namespace": _memory_namespace_schema(include_all=True),
                "key": _string(minLength=1),
            }
        ),
        consumes=(
            _fact(
                MEMORY_ENTRY,
                "request.key",
                FactAuthority.HOST,
                required=False,
                when="a specific key is requested",
            ),
        ),
        outcomes=(
            OutcomeContract(
                "completed",
                True,
                _object(additional_properties=True),
                produces=(
                    _fact(MEMORY_SNAPSHOT, "outputs", FactAuthority.HOST),
                ),
            ),
            _operational_failure(),
        ),
        evidence_lifetime=EvidenceLifetimeContract(
            scope="read_snapshot",
            invalidated_by=("later save/delete/compact may change current memory",),
            reusable_across_packet_refresh=True,
        ),
        gate=GateContract(
            checks=("namespace is supported",),
            fail_closed=False,
            description="Missing keys return an explicit empty/not-found snapshot rather than invented content.",
        ),
        source_paths=("agent/runtime/runtime.py", "agent/runtime/memory.py"),
    )


def _delete_memory(spec: ToolSpecLike) -> ToolContract:
    return _explicit_contract(
        spec,
        request_schema=_object(
            {
                "namespace": _memory_namespace_schema(include_all=True),
                "key": _string(minLength=1),
            },
            required=("key",),
        ),
        consumes=(
            _fact(MEMORY_ENTRY, "request.key", FactAuthority.HOST, required=False),
        ),
        outcomes=(
            OutcomeContract(
                "completed",
                True,
                _object(additional_properties=True),
            ),
            _operational_failure(),
        ),
        evidence_lifetime=EvidenceLifetimeContract(
            scope="deletion_receipt",
            invalidated_by=(),
            reusable_across_packet_refresh=True,
        ),
        gate=GateContract(
            checks=("key is non-empty", "namespace is supported"),
            fail_closed=True,
        ),
        source_paths=("agent/runtime/runtime.py", "agent/runtime/memory.py"),
    )


def _compact_memory(spec: ToolSpecLike) -> ToolContract:
    return _explicit_contract(
        spec,
        request_schema=_object(
            {"max_events": _integer(minimum=1)},
        ),
        outcomes=(
            OutcomeContract(
                "completed",
                True,
                _outputs(("summary",), {"summary": _string()}),
                produces=(
                    _fact(MEMORY_COMPACTION, "outputs.summary", FactAuthority.HOST),
                ),
            ),
            _operational_failure(),
        ),
        evidence_lifetime=EvidenceLifetimeContract(
            scope="durable_compaction_summary",
            invalidated_by=(),
            reusable_across_packet_refresh=True,
            description="Compaction changes the bounded projection, not durable history.",
        ),
        gate=GateContract(
            checks=("max_events is normalized to a positive bounded projection size",),
            fail_closed=False,
        ),
        source_paths=("agent/runtime/runtime.py", "agent/runtime/memory.py"),
    )


def _skill_authoring_outcomes() -> tuple[OutcomeContract, ...]:
    return (
        OutcomeContract(
            "completed",
            True,
            _outputs(
                ("skill", "source", "review"),
                {
                    "skill": _string(),
                    "source": _string(),
                    "authoring": _object(additional_properties=True),
                    "provider": _string(),
                    "model": _string(),
                    "review": _object(additional_properties=True),
                },
            ),
            produces=(
                _fact(EDITABLE_SKILL, "outputs.skill", FactAuthority.HOST),
            ),
        ),
        _operational_failure(),
    )


def _register_skill(spec: ToolSpecLike) -> ToolContract:
    return _explicit_contract(
        spec,
        request_schema=_object(
            {
                "name": _string(minLength=1, pattern="^[a-z0-9]+(?:-[a-z0-9]+)*$"),
                "goal": _string(),
                "description": _string(),
                "requirements": {},
                "examples": {},
                "content": _string(),
                "task_patterns": _array(_string()),
                "allowed_tools": _array(_string()),
            },
            required=("name",),
        ),
        outcomes=_skill_authoring_outcomes(),
        host_resolution=HostResolutionContract(
            mode="isolated_authoring_and_review",
            resolver="_skill_change_handler(register)",
            agent_parameters=("name", "authoring instructions", "optional allowed tool names"),
            resolved_parameters=("executable ToolSpecs", "authoring backend", "supervision policy"),
        ),
        evidence_lifetime=EvidenceLifetimeContract(
            scope="session_skill_registry",
            invalidated_by=("later update", "session end"),
            reusable_across_packet_refresh=True,
        ),
        gate=GateContract(
            checks=(
                "name is new and valid",
                "at least one authoring instruction is supplied",
                "allowed tools already exist and are executable",
                "authored SkillSpec passes validation",
                "supervision policy or independent/human review approves",
                "skill authoring cannot create or modify ToolSpecs or handlers",
            ),
            fail_closed=True,
        ),
        source_paths=("agent/runtime/runtime_assembly.py", "agent/skills/registry.py"),
    )


def _update_skill(spec: ToolSpecLike) -> ToolContract:
    return _explicit_contract(
        spec,
        request_schema=_object(
            {
                "name": _string(minLength=1),
                "requested_changes": _string(),
                "examples": {},
                "requirements": {},
                "content": _string(),
            },
            required=("name",),
        ),
        consumes=(
            _fact(EDITABLE_SKILL, "request.name", FactAuthority.HOST),
        ),
        outcomes=_skill_authoring_outcomes(),
        host_resolution=HostResolutionContract(
            mode="isolated_authoring_and_review",
            resolver="_skill_change_handler(update)",
            agent_parameters=("name", "requested changes and supporting instructions"),
            resolved_parameters=("current editable SkillSpec", "executable ToolSpecs", "supervision policy"),
        ),
        evidence_lifetime=EvidenceLifetimeContract(
            scope="session_skill_registry",
            invalidated_by=("later update", "session end"),
            reusable_across_packet_refresh=True,
        ),
        gate=GateContract(
            checks=(
                "named skill exists and is editable",
                "at least one update instruction is supplied",
                "updated SkillSpec preserves tool-contract boundary",
                "supervision policy or independent/human review approves",
            ),
            fail_closed=True,
        ),
        source_paths=("agent/runtime/runtime_assembly.py", "agent/skills/registry.py"),
    )
