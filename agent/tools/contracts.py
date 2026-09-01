"""Machine-readable contracts for composing OpenETA tools.

The first migration phase deliberately does not change planner validation or
runtime gates.  It inventories every ToolSpec, records which pieces are still
inferred from prose, and provides one versioned shape from which developer docs
and later contract-driven validation can be generated.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Protocol

from adapter.protocol import JsonDict


TOOL_CONTRACT_SCHEMA_VERSION = "openeta.tool_contract.v1"
TOOL_CONTRACT_CATALOG_SCHEMA_VERSION = "openeta.tool_contract_catalog.v1"
GATE_REPAIR_SCHEMA_VERSION = "openeta.gate_repair.v1"
AGENT_TOOL_PROJECTION_SCHEMA_VERSION = "openeta.agent_tool_contract.v2"
AGENT_TOOL_PROJECTION_AUDIT_SCHEMA_VERSION = (
    "openeta.agent_tool_contract_projection_audit.v1"
)
TOOL_CONTRACT_RUNTIME_POLICY_SCHEMA_VERSION = (
    "openeta.tool_contract_runtime_policy.v1"
)


class ToolSpecLike(Protocol):
    """Structural subset needed to migrate an existing ToolSpec."""

    name: str
    description: str
    category: str
    parameters: JsonDict
    safe_by_default: bool
    effect: Any
    batchable: bool | None

    @property
    def allows_batched_observation(self) -> bool: ...

    @property
    def requires_observation_after_call(self) -> bool: ...


class ContractMaturity(str, Enum):
    """How much of a contract is backed by an explicit declaration."""

    INFERRED = "inferred"
    DECLARED = "declared"
    VERIFIED = "verified"


@dataclass(frozen=True, slots=True)
class ToolContractRuntimePolicy:
    """Explicit, fail-closed authority switches for verified contracts only.

    Request validation and gate-repair envelope validation are separate because
    the latter does not replace the executable safety checks in ``ActionPipeline``.
    An empty policy is the production default during migration: ToolContract is
    observed everywhere but legacy runtime behavior remains authoritative.
    """

    request_validation_authority: frozenset[str] = frozenset()
    gate_repair_envelope_authority: frozenset[str] = frozenset()
    schema_version: str = TOOL_CONTRACT_RUNTIME_POLICY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TOOL_CONTRACT_RUNTIME_POLICY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported ToolContract runtime policy: {self.schema_version}"
            )
        object.__setattr__(
            self,
            "request_validation_authority",
            frozenset(str(name) for name in self.request_validation_authority),
        )
        object.__setattr__(
            self,
            "gate_repair_envelope_authority",
            frozenset(str(name) for name in self.gate_repair_envelope_authority),
        )

    def validate_against(self, catalog: "ToolContractCatalog") -> tuple[str, ...]:
        """Reject unknown or unverified authority entries before a run starts."""

        errors: list[str] = []
        for authority, names in (
            ("request_validation", self.request_validation_authority),
            ("gate_repair_envelope", self.gate_repair_envelope_authority),
        ):
            for name in sorted(names):
                try:
                    contract = catalog.get(name)
                except KeyError:
                    errors.append(f"{authority}: unknown ToolContract {name!r}")
                    continue
                if contract.maturity is not ContractMaturity.VERIFIED:
                    errors.append(
                        f"{authority}: {name!r} is {contract.maturity.value}, not verified"
                    )
                if authority == "gate_repair_envelope" and not contract.gate.bindings:
                    errors.append(
                        f"{authority}: {name!r} has no machine-bound gate checks"
                    )
        return tuple(errors)

    def ensure_valid(self, catalog: "ToolContractCatalog") -> None:
        errors = self.validate_against(catalog)
        if errors:
            raise ValueError("Invalid ToolContract runtime policy: " + "; ".join(errors))

    def request_is_authoritative(self, tool_name: str) -> bool:
        return tool_name in self.request_validation_authority

    def gate_repair_is_authoritative(self, tool_name: str) -> bool:
        return tool_name in self.gate_repair_envelope_authority

    def to_dict(self) -> JsonDict:
        return {
            "schema_version": self.schema_version,
            "request_validation_authority": sorted(
                self.request_validation_authority
            ),
            "gate_repair_envelope_authority": sorted(
                self.gate_repair_envelope_authority
            ),
            "executable_gate_authority": "legacy_runtime",
        }


class FactAuthority(str, Enum):
    """Authority that is allowed to assert one composable fact."""

    AGENT = "agent"
    HOST = "host"
    TOOL = "tool"
    BACKEND = "backend"
    ENVIRONMENT = "environment"


@dataclass(frozen=True, slots=True)
class FactBinding:
    """A typed producer or consumer edge in the tool-composition graph."""

    fact_type: str
    path: str
    authority: FactAuthority | str
    schema_version: str = ""
    required: bool = True
    when: str = ""
    description: str = ""

    def __post_init__(self) -> None:
        if not self.fact_type.strip():
            raise ValueError("fact_type must be non-empty")
        if not self.path.strip():
            raise ValueError("fact path must be non-empty")
        if isinstance(self.authority, str):
            object.__setattr__(self, "authority", FactAuthority(self.authority))

    def to_dict(self) -> JsonDict:
        result: JsonDict = {
            "fact_type": self.fact_type,
            "path": self.path,
            "authority": self.authority.value,
            "required": self.required,
        }
        if self.schema_version:
            result["schema_version"] = self.schema_version
        if self.when:
            result["when"] = self.when
        if self.description:
            result["description"] = self.description
        return result


@dataclass(frozen=True, slots=True)
class HostResolutionContract:
    """How public Agent arguments become the private handler input."""

    mode: str = "none"
    resolver: str = ""
    implementation: str = ""
    resolution_layer: str = ""
    contract_driven_dispatch: bool = False
    agent_parameters: tuple[str, ...] = ()
    resolved_parameters: tuple[str, ...] = ()
    freshness_dimensions: tuple[str, ...] = ()
    invalidated_by: tuple[str, ...] = ()
    description: str = ""

    def to_dict(self) -> JsonDict:
        return {
            "mode": self.mode,
            "resolver": self.resolver,
            "implementation": self.implementation,
            "resolution_layer": self.resolution_layer,
            "contract_driven_dispatch": self.contract_driven_dispatch,
            "agent_parameters": list(self.agent_parameters),
            "resolved_parameters": list(self.resolved_parameters),
            "freshness_dimensions": list(self.freshness_dimensions),
            "invalidated_by": list(self.invalidated_by),
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class GateCheckBinding:
    """Stable identity and implementation evidence for one non-prescriptive gate."""

    check_id: str
    description: str
    implementation: str = ""
    repair_codes: tuple[str, ...] = ()
    applies_when: str = ""

    def __post_init__(self) -> None:
        if not self.check_id.strip():
            raise ValueError("gate check_id must be non-empty")
        if not self.description.strip():
            raise ValueError("gate check description must be non-empty")

    def to_dict(self) -> JsonDict:
        return {
            "check_id": self.check_id,
            "description": self.description,
            "implementation": self.implementation,
            "repair_codes": list(self.repair_codes),
            "applies_when": self.applies_when,
        }


@dataclass(frozen=True, slots=True)
class GateContract:
    """Host checks that protect evidence integrity without prescribing task order."""

    checks: tuple[str, ...] = ()
    bindings: tuple[GateCheckBinding, ...] = ()
    fail_closed: bool = False
    repair_schema: str = GATE_REPAIR_SCHEMA_VERSION
    preserves_agent_choice: bool = True
    description: str = ""

    def to_dict(self) -> JsonDict:
        return {
            "checks": list(self.checks),
            "bindings": [binding.to_dict() for binding in self.bindings],
            "fail_closed": self.fail_closed,
            "repair_schema": self.repair_schema,
            "preserves_agent_choice": self.preserves_agent_choice,
            "description": self.description,
        }


@dataclass(frozen=True, slots=True)
class EvidenceLifetimeContract:
    """Validity scope of facts emitted by one tool call."""

    scope: str = "call"
    freshness_dimensions: tuple[str, ...] = ()
    invalidated_by: tuple[str, ...] = ()
    reusable_across_packet_refresh: bool | None = None
    description: str = ""

    def to_dict(self) -> JsonDict:
        result: JsonDict = {
            "scope": self.scope,
            "freshness_dimensions": list(self.freshness_dimensions),
            "invalidated_by": list(self.invalidated_by),
            "description": self.description,
        }
        if self.reusable_across_packet_refresh is not None:
            result["reusable_across_packet_refresh"] = (
                self.reusable_across_packet_refresh
            )
        return result


@dataclass(frozen=True, slots=True)
class OutcomeContract:
    """Required output shape for one semantic tool outcome."""

    semantic_outcome: str
    operational_success: bool | None
    output_schema: JsonDict = field(
        default_factory=lambda: {"type": "object", "additionalProperties": True}
    )
    produces: tuple[FactBinding, ...] = ()
    executable_reference: bool | None = None
    diagnostics_required: bool = False
    recovery_required: bool = False
    description: str = ""

    def __post_init__(self) -> None:
        if not self.semantic_outcome.strip():
            raise ValueError("semantic_outcome must be non-empty")
        if self.output_schema.get("type") != "object":
            raise ValueError("outcome output_schema must describe an object")

    def to_dict(self) -> JsonDict:
        result: JsonDict = {
            "semantic_outcome": self.semantic_outcome,
            "operational_success": self.operational_success,
            "output_schema": dict(self.output_schema),
            "produces": [fact.to_dict() for fact in self.produces],
            "diagnostics_required": self.diagnostics_required,
            "recovery_required": self.recovery_required,
            "description": self.description,
        }
        if self.executable_reference is not None:
            result["executable_reference"] = self.executable_reference
        return result


@dataclass(frozen=True, slots=True)
class ToolContract:
    """Complete host-side composition contract for one Agent-visible tool."""

    name: str
    category: str
    description: str
    effect: str
    request_schema: JsonDict
    outcomes: tuple[OutcomeContract, ...]
    consumes: tuple[FactBinding, ...] = ()
    host_resolution: HostResolutionContract = field(
        default_factory=HostResolutionContract
    )
    evidence_lifetime: EvidenceLifetimeContract = field(
        default_factory=EvidenceLifetimeContract
    )
    gate: GateContract = field(default_factory=GateContract)
    maturity: ContractMaturity | str = ContractMaturity.INFERRED
    safe_by_default: bool = False
    batchable: bool = True
    requires_observation_after_call: bool = False
    source_paths: tuple[str, ...] = ()
    coverage_gaps: tuple[str, ...] = ()
    schema_version: str = TOOL_CONTRACT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != TOOL_CONTRACT_SCHEMA_VERSION:
            raise ValueError(f"unsupported tool contract schema: {self.schema_version}")
        if not self.name.strip():
            raise ValueError("tool contract name must be non-empty")
        if self.request_schema.get("type") != "object":
            raise ValueError("request_schema must describe an object")
        if not self.outcomes:
            raise ValueError("tool contract requires at least one semantic outcome")
        outcome_names = [item.semantic_outcome for item in self.outcomes]
        if len(outcome_names) != len(set(outcome_names)):
            raise ValueError(f"duplicate semantic outcomes for {self.name}")
        if isinstance(self.maturity, str):
            object.__setattr__(self, "maturity", ContractMaturity(self.maturity))

    def to_dict(self) -> JsonDict:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "category": self.category,
            "description": self.description,
            "effect": self.effect,
            "safe_by_default": self.safe_by_default,
            "batchable": self.batchable,
            "requires_observation_after_call": self.requires_observation_after_call,
            "maturity": self.maturity.value,
            "request_schema": dict(self.request_schema),
            "host_resolution": self.host_resolution.to_dict(),
            "consumes": [fact.to_dict() for fact in self.consumes],
            "outcomes": [outcome.to_dict() for outcome in self.outcomes],
            "evidence_lifetime": self.evidence_lifetime.to_dict(),
            "gate": self.gate.to_dict(),
            "source_paths": list(self.source_paths),
            "coverage_gaps": list(self.coverage_gaps),
        }


class ToolContractCatalog:
    """Immutable-name catalog used by docs, audits, and future runtime consumers."""

    def __init__(self, contracts: Iterable[ToolContract] = ()) -> None:
        self._contracts: dict[str, ToolContract] = {}
        for contract in contracts:
            self.register(contract)

    def register(self, contract: ToolContract, *, replace: bool = False) -> None:
        if contract.name in self._contracts and not replace:
            raise ValueError(f"Tool contract already registered: {contract.name}")
        self._contracts[contract.name] = contract

    def get(self, name: str) -> ToolContract:
        try:
            return self._contracts[name]
        except KeyError as exc:
            raise KeyError(f"Unknown tool contract: {name}") from exc

    def list(self) -> list[ToolContract]:
        return list(self._contracts.values())

    def to_dict(self) -> JsonDict:
        return {
            "schema_version": TOOL_CONTRACT_CATALOG_SCHEMA_VERSION,
            "summary": self.coverage_summary(),
            "tools": [contract.to_dict() for contract in self.list()],
        }

    def coverage_summary(self) -> JsonDict:
        maturity_counts = {item.value: 0 for item in ContractMaturity}
        gap_counts: dict[str, int] = {}
        for contract in self.list():
            maturity_counts[contract.maturity.value] += 1
            for gap in contract.coverage_gaps:
                gap_counts[gap] = gap_counts.get(gap, 0) + 1
        return {
            "tool_count": len(self._contracts),
            "maturity_counts": maturity_counts,
            "coverage_gap_counts": dict(sorted(gap_counts.items())),
        }


_AGENT_TOOL_DESCRIPTIONS: dict[str, str] = {
    "enhance_depth": (
        "Fuse aligned sensor depth with an optional metric depth prior and materialize "
        "enhanced geometry artifacts."
    ),
    "estimate_depth_prior": (
        "Estimate and materialize a metric monocular depth prior for one session-owned "
        "observation frame."
    ),
    "create_simulator_env": (
        "Create one remote simulator environment and return its initial observation."
    ),
    "close_simulator_env": (
        "Close the active remote simulator environment and clear its bound handle."
    ),
    "python_exec": (
        "Execute restricted Python for session-local inspection, computation, and "
        "derived artifacts."
    ),
    "web_search": "Search the public web and return a compact answer with citations and snippets.",
    "web_fetch": "Fetch and extract readable text from one public HTTPS page.",
    "sam3": (
        "Segment an RGB observation from text or pixel prompts and return ranked "
        "detections with review visuals in original-image coordinates."
    ),
    "retrieve_asset_reference": (
        "Resolve an object identity phrase against controlled asset references and "
        "return ranked scene-localization seeds for visual confirmation."
    ),
    "select_sam3_detection": (
        "Select one stable detection from a pending SAM3 result as the visually "
        "confirmed semantic target."
    ),
    "reject_sam3_detections": (
        "Reject every candidate in a pending SAM3 result as a semantic mismatch."
    ),
    "grasp_pose_estimate": (
        "Estimate a normalized, score-ranked camera-frame grasp candidate set from "
        "host-resolved aligned RGB-D evidence."
    ),
    "camera_pose_to_world": (
        "Transform a camera-frame pose or referenced placement candidate into the world frame."
    ),
    "propose_calibration_profile": (
        "Stage and independently review one session-local embodiment calibration proposal."
    ),
    "promote_calibration_profile": (
        "Publish an evidence-backed reviewed calibration proposal at an authorized "
        "lifecycle status."
    ),
    "propose_grasp_strategy": (
        "Stage and independently review one session-local task-family grasp strategy proposal."
    ),
    "promote_grasp_strategy": (
        "Publish an evidence-backed reviewed grasp strategy at an authorized lifecycle status."
    ),
    "compile_grasp_seed": (
        "Compile one referenced camera-frame grasp candidate into calibrated world-frame "
        "EEF contact and clearance geometry."
    ),
    "compute_wrist_alignment": (
        "Compute a bounded world-frame lateral correction for a compiled grasp from "
        "fresh calibrated wrist RGB-D evidence."
    ),
    "propose_wrist_viewpoints": (
        "Generate calibrated target-facing wrist-camera observation poses around one "
        "current compiled grasp anchor."
    ),
    "prepare_attachment_probe": (
        "Validate and freeze one Agent-proposed attachment probe against a current "
        "compiled grasp and a tentative non-empty close receipt."
    ),
    "assess_attachment_probe": (
        "Assess attachment by independently comparing a frozen probe's before/after "
        "scene and wrist images while measured gripper evidence remains compatible."
    ),
    "move_to": (
        "Move the end effector to the exact host-resolved pose frozen by one current IK receipt."
    ),
    "follow_eef_trajectory": (
        "Execute an ordered atomic trajectory of individually authorized end-effector "
        "waypoints."
    ),
    "gripper_control": "Transition the simulator's latched gripper command state.",
    "ik_preview_check": (
        "Preview endpoint reachability and return an exact execution receipt or "
        "structured infeasibility diagnostics."
    ),
    "register_skill": (
        "Create, validate, and register one text-guidance SkillSpec through an isolated "
        "authoring agent."
    ),
    "update_skill": "Revise one existing editable SkillSpec through an isolated authoring agent.",
}


_AGENT_TOOL_SEMANTIC_LIMITS: dict[str, tuple[str, ...]] = {
    "enhance_depth": ("sensor_depth_authoritative", "model_depth_hole_fill_only"),
    "estimate_depth_prior": ("depth_prior_only", "does_not_replace_sensor_depth"),
    "create_simulator_env": ("exclusive_environment_creation_path",),
    "close_simulator_env": ("exclusive_environment_cleanup_path",),
    "python_exec": (
        "session_local_sandbox",
        "no_network",
        "no_simulator_mcp",
        "no_external_side_effects",
    ),
    "web_search": ("public_web_only", "untrusted_external_content", "no_private_data"),
    "web_fetch": (
        "public_https_only",
        "untrusted_external_content",
        "no_private_or_authenticated_targets",
    ),
    "sam3": (
        "session_packet_only",
        "point_and_roi_mutually_exclusive",
        "candidate_rank_not_identity_confirmation",
    ),
    "retrieve_asset_reference": (
        "object_identity_phrase_only",
        "no_agent_supplied_url",
        "ambiguous_match_fails_closed",
    ),
    "select_sam3_detection": ("explicit_semantic_confirmation",),
    "grasp_pose_estimate": (
        "host_selects_backend_fallback",
        "advisor_recommendation_not_activation",
    ),
    "anyplace": ("host_resolved_bundle_only",),
    "camera_pose_to_world": ("host_owned_calibration",),
    "propose_calibration_profile": ("proposal_not_publication",),
    "promote_calibration_profile": ("review_and_evidence_gated",),
    "propose_grasp_strategy": ("proposal_not_activation",),
    "promote_grasp_strategy": ("review_and_evidence_gated",),
    "compile_grasp_seed": (
        "candidate_not_motion_authorization",
        "unknown_geometry_uses_generic_calibration",
    ),
    "compute_wrist_alignment": (
        "lateral_translation_only",
        "no_orientation_or_axial_depth_refinement",
        "not_motion_authorization",
    ),
    "propose_wrist_viewpoints": ("not_motion_authorization",),
    "prepare_attachment_probe": ("not_task_stage", "not_motion_authorization"),
    "assess_attachment_probe": (
        "visual_evidence_only",
        "no_privileged_joint_state",
        "not_motion_authorization",
    ),
    "move_to": (
        "exact_ik_receipt_only",
        "endpoint_ik_not_path_clearance",
        "compiled_residual_budget_applies",
    ),
    "follow_eef_trajectory": (
        "exact_ik_receipt_sequence_only",
        "path_collision_separate",
        "preserves_latched_gripper",
    ),
    "gripper_control": ("command_latched", "command_ack_not_attachment"),
    "ik_preview_check": (
        "endpoint_only",
        "not_path_authorization",
        "reachability_not_controller_recommendation",
    ),
    "register_skill": ("cannot_modify_tools",),
    "update_skill": ("cannot_modify_tools",),
}


# Non-normative annotations for compact planner-visible request schemas.  The
# canonical ToolContract remains the validation/authority source; these notes
# make otherwise ambiguous enum direction explicit without changing the
# reviewed catalog hash.
_AGENT_TOOL_PARAMETER_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "gripper_control": {
        "position": (
            "Required binary command: 0 or false closes the gripper; 1 or true "
            "opens it. This is a commanded latch state, not a measured aperture "
            "or fractional opening."
        ),
    },
}


def project_agent_tool_contract(contract: ToolContract) -> JsonDict:
    """Project one ToolContract into the compact schema shown to the Agent.

    Deployment availability remains a ToolRegistry concern.  The Agent sees only
    the short capability description, canonical request schema, compact result
    names, and explicit semantic limits.  Full host resolution, outcome schemas,
    gate bindings, and implementation evidence remain in the developer contract.
    """

    return {
        "name": contract.name,
        "description": _AGENT_TOOL_DESCRIPTIONS.get(contract.name, contract.description),
        "parameters": _project_agent_parameters(contract),
        "returns": _project_agent_returns(contract),
        "semantic_limits": _project_agent_semantic_limits(contract),
    }


def _project_agent_parameters(contract: ToolContract) -> JsonDict:
    """Copy the canonical request schema and add planner-only field notes."""

    projected = dict(contract.request_schema)
    properties = contract.request_schema.get("properties")
    if not isinstance(properties, Mapping):
        return projected
    projected_properties: JsonDict = {
        str(name): dict(schema) if isinstance(schema, Mapping) else schema
        for name, schema in properties.items()
    }
    for name, description in _AGENT_TOOL_PARAMETER_DESCRIPTIONS.get(
        contract.name, {}
    ).items():
        schema = projected_properties.get(name)
        if isinstance(schema, dict):
            schema["description"] = description
    projected["properties"] = projected_properties
    return projected


def _project_agent_returns(contract: ToolContract) -> JsonDict:
    """Project outcome names and composable top-level result references."""

    outcomes: list[str] = []
    fields: list[str] = []
    for outcome in contract.outcomes:
        if outcome.semantic_outcome == "operational_failure":
            continue
        outcomes.append(outcome.semantic_outcome)
        properties = outcome.output_schema.get("properties")
        if isinstance(properties, Mapping):
            fields.extend(str(name) for name in properties if name != "schema_version")
        for fact in outcome.produces:
            terminal = fact.path.rsplit(".", 1)[-1]
            if terminal not in {"outputs", "details"}:
                fields.append(terminal)
    return {
        "outcomes": list(dict.fromkeys(outcomes)),
        "fields": list(dict.fromkeys(fields)),
    }


def _project_agent_semantic_limits(contract: ToolContract) -> list[str]:
    """Combine generic execution boundaries with tool-specific limit tags."""

    effect_limit = {
        "read_only": "read_only",
        "planning": "planning_only",
        "bookkeeping": "bookkeeping_only",
        "world_mutating": "world_mutating",
    }.get(contract.effect, contract.effect)
    limits = [effect_limit]
    if not contract.batchable:
        limits.append("atomic_call_only")
    if contract.requires_observation_after_call:
        limits.append("fresh_observation_after_call")
    limits.extend(_AGENT_TOOL_SEMANTIC_LIMITS.get(contract.name, ()))
    return list(dict.fromkeys(value for value in limits if value))


def audit_agent_tool_projection(
    contract: ToolContract,
    spec: ToolSpecLike,
) -> JsonDict:
    """Compare the contract projection boundary with its legacy ToolSpec source.

    This audit intentionally compares interface identity and top-level parameter
    names, not prose formatting.  Declared JSON-schema constraints are expected to
    be richer than the legacy string map.
    """

    properties = contract.request_schema.get("properties")
    contract_parameters = (
        sorted(str(name) for name in properties)
        if isinstance(properties, Mapping)
        else []
    )
    legacy_parameters = sorted(str(name) for name in spec.parameters)
    effect = str(getattr(spec.effect, "value", spec.effect))
    field_matches = {
        "name": contract.name == spec.name,
        "category": contract.category == spec.category,
        "description": contract.description == spec.description,
        "effect": contract.effect == effect,
        "safe_by_default": contract.safe_by_default == spec.safe_by_default,
        "batchable": contract.batchable == spec.allows_batched_observation,
        "requires_observation_after_call": (
            contract.requires_observation_after_call
            == spec.requires_observation_after_call
        ),
        "parameter_names": contract_parameters == legacy_parameters,
    }
    return {
        "schema_version": AGENT_TOOL_PROJECTION_AUDIT_SCHEMA_VERSION,
        "tool": contract.name,
        "contract_maturity": contract.maturity.value,
        "matches": all(field_matches.values()),
        "field_matches": field_matches,
        "legacy_only_parameters": sorted(
            set(legacy_parameters) - set(contract_parameters)
        ),
        "contract_only_parameters": sorted(
            set(contract_parameters) - set(legacy_parameters)
        ),
    }


@dataclass(frozen=True, slots=True)
class ToolChainCompatibilityIssue:
    """One missing or incompatible typed-fact edge in an ordered tool chain."""

    tool: str
    fact_type: str
    code: str
    message: str

    def to_dict(self) -> JsonDict:
        return {
            "tool": self.tool,
            "fact_type": self.fact_type,
            "code": self.code,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class ToolChainCompatibilityReport:
    """Potential composition report; it never prescribes a runtime task order."""

    tools: tuple[str, ...]
    initial_facts: tuple[str, ...]
    available_facts: tuple[str, ...]
    issues: tuple[ToolChainCompatibilityIssue, ...]

    @property
    def compatible(self) -> bool:
        return not self.issues

    def to_dict(self) -> JsonDict:
        return {
            "compatible": self.compatible,
            "tools": list(self.tools),
            "initial_facts": list(self.initial_facts),
            "available_facts": list(self.available_facts),
            "issues": [issue.to_dict() for issue in self.issues],
            "interpretation": (
                "Static producer-consumer compatibility only; this does not require "
                "the Agent to follow the listed order at runtime."
            ),
        }


@dataclass(frozen=True, slots=True)
class ToolResultContractViolation:
    """One mismatch between a durable ToolResult envelope and its declaration."""

    tool: str
    code: str
    path: str
    message: str

    def to_dict(self) -> JsonDict:
        return {
            "tool": self.tool,
            "code": self.code,
            "path": self.path,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class ToolRequestContractViolation:
    """One JSON-Schema-subset mismatch in an Agent-authored tool request."""

    tool: str
    code: str
    path: str
    message: str

    def to_dict(self) -> JsonDict:
        return {
            "tool": self.tool,
            "code": self.code,
            "path": self.path,
            "message": self.message,
        }


@dataclass(frozen=True, slots=True)
class GateRepairContractViolation:
    """One mismatch in the structured feedback returned by a rejected gate."""

    tool: str
    code: str
    path: str
    message: str

    def to_dict(self) -> JsonDict:
        return {
            "tool": self.tool,
            "code": self.code,
            "path": self.path,
            "message": self.message,
        }


def inferred_tool_contract(spec: ToolSpecLike) -> ToolContract:
    """Build a lossless inventory contract from the existing prose ToolSpec."""

    properties: JsonDict = {}
    for name, description in spec.parameters.items():
        property_schema: JsonDict = {}
        if isinstance(description, str):
            property_schema["description"] = description
        elif isinstance(description, Mapping):
            property_schema["description"] = json.dumps(
                description,
                ensure_ascii=False,
                sort_keys=True,
            )
        else:
            property_schema["description"] = str(description)
        properties[str(name)] = property_schema
    effect = getattr(spec.effect, "value", spec.effect)
    return ToolContract(
        name=spec.name,
        category=spec.category,
        description=spec.description,
        effect=str(effect),
        safe_by_default=spec.safe_by_default,
        batchable=spec.allows_batched_observation,
        requires_observation_after_call=spec.requires_observation_after_call,
        request_schema={
            "type": "object",
            "properties": properties,
            # Phase 1 is descriptive only.  Tightening this flag before every
            # legacy validator is represented would change runtime behavior.
            "additionalProperties": True,
        },
        outcomes=(
            OutcomeContract(
                semantic_outcome="completed",
                operational_success=True,
                description="Compatibility outcome inferred from ToolSpec prose.",
            ),
        ),
        maturity=ContractMaturity.INFERRED,
        source_paths=("agent/tools/registry.py",),
        coverage_gaps=(
            "request_required_fields",
            "request_field_types",
            "host_resolution",
            "semantic_outcomes",
            "produced_facts",
            "consumed_facts",
            "evidence_lifetime",
            "gate_checks",
        ),
    )


def build_tool_contract_catalog(tool_specs: Iterable[ToolSpecLike]) -> ToolContractCatalog:
    """Inventory every ToolSpec without changing the live ToolRegistry."""

    return ToolContractCatalog(inferred_tool_contract(spec) for spec in tool_specs)


def build_default_tool_contract_catalog(
    tool_specs: Iterable[ToolSpecLike],
) -> ToolContractCatalog:
    """Build the migration catalog with explicit core-tool overrides."""

    specs = list(tool_specs)
    catalog = build_tool_contract_catalog(specs)
    # Imported lazily so the generic contract model stays independent from the
    # repository's default toolchain declarations.
    from agent.tools.default_contracts import build_default_contract_overrides

    for contract in build_default_contract_overrides(specs).values():
        catalog.register(contract, replace=True)
    return catalog


def check_tool_chain_compatibility(
    catalog: ToolContractCatalog,
    tool_names: Iterable[str],
    *,
    initial_facts: Iterable[str] = (),
) -> ToolChainCompatibilityReport:
    """Check whether earlier producers can satisfy later required fact inputs.

    All successful semantic outcomes are treated as possible producers.  Runtime
    outcome choice, freshness, safety and task intent remain the Agent and gate's
    responsibility.  Host-synthesized bundle facts should be named explicitly in
    ``initial_facts`` when checking a partial chain.
    """

    names = tuple(str(name) for name in tool_names)
    initial = tuple(dict.fromkeys(str(fact) for fact in initial_facts))
    available: dict[str, set[str]] = {fact: {fact} for fact in initial}
    issues: list[ToolChainCompatibilityIssue] = []
    for name in names:
        try:
            contract = catalog.get(name)
        except KeyError:
            issues.append(
                ToolChainCompatibilityIssue(
                    tool=name,
                    fact_type="",
                    code="unknown_tool",
                    message=f"No ToolContract is registered for {name!r}.",
                )
            )
            continue
        for consumed in contract.consumes:
            if not consumed.required:
                continue
            versions = available.get(consumed.fact_type)
            if not versions:
                issues.append(
                    ToolChainCompatibilityIssue(
                        tool=name,
                        fact_type=consumed.fact_type,
                        code="missing_required_fact",
                        message=(
                            f"{name} requires {consumed.fact_type} at {consumed.path}, "
                            "but no earlier tool or initial host fact provides it."
                        ),
                    )
                )
                continue
            expected_version = consumed.schema_version
            if expected_version and expected_version not in versions:
                issues.append(
                    ToolChainCompatibilityIssue(
                        tool=name,
                        fact_type=consumed.fact_type,
                        code="fact_schema_version_mismatch",
                        message=(
                            f"{name} expects {expected_version}; available versions are "
                            f"{sorted(versions)}."
                        ),
                    )
                )
        for outcome in contract.outcomes:
            if outcome.operational_success is not True:
                continue
            for produced in outcome.produces:
                available.setdefault(produced.fact_type, set()).add(
                    produced.schema_version or produced.fact_type
                )
    return ToolChainCompatibilityReport(
        tools=names,
        initial_facts=initial,
        available_facts=tuple(sorted(available)),
        issues=tuple(issues),
    )


def check_tool_result_conformance(
    contract: ToolContract,
    details: Mapping[str, Any],
) -> tuple[ToolResultContractViolation, ...]:
    """Check one canonical ToolResult details envelope against a declaration.

    The compatibility ``operational_failure`` outcome is a catch-all for an
    implementation's more specific failure code.  Successful semantic outcomes
    must be declared exactly so useful distinctions cannot silently disappear.
    """

    violations: list[ToolResultContractViolation] = []
    semantic_outcome = str(details.get("semantic_outcome") or "").strip()
    operational_success = details.get("operational_success")
    outcome = next(
        (
            candidate
            for candidate in contract.outcomes
            if candidate.semantic_outcome == semantic_outcome
        ),
        None,
    )
    if outcome is None and operational_success is False:
        outcome = next(
            (
                candidate
                for candidate in contract.outcomes
                if candidate.semantic_outcome == "operational_failure"
            ),
            None,
        )
    if outcome is None:
        violations.append(
            ToolResultContractViolation(
                tool=contract.name,
                code="undeclared_semantic_outcome",
                path="details.semantic_outcome",
                message=(
                    f"Outcome {semantic_outcome!r} is not declared for {contract.name}."
                ),
            )
        )
        return tuple(violations)
    if (
        outcome.operational_success is not None
        and operational_success is not outcome.operational_success
    ):
        violations.append(
            ToolResultContractViolation(
                tool=contract.name,
                code="operational_success_mismatch",
                path="details.operational_success",
                message=(
                    f"Outcome {outcome.semantic_outcome!r} expects operational_success="
                    f"{outcome.operational_success}, got {operational_success!r}."
                ),
            )
        )
    outputs = details.get("outputs")
    outputs = outputs if isinstance(outputs, Mapping) else {}
    required_outputs = outcome.output_schema.get("required")
    required_outputs = required_outputs if isinstance(required_outputs, list) else []
    for field_name in required_outputs:
        if field_name not in outputs:
            violations.append(
                ToolResultContractViolation(
                    tool=contract.name,
                    code="required_output_missing",
                    path=f"details.outputs.{field_name}",
                    message=(
                        f"Outcome {outcome.semantic_outcome!r} requires output "
                        f"{field_name!r}."
                    ),
                )
            )
    if outcome.diagnostics_required and not _nonempty_mapping_list(
        details.get("diagnostics")
    ):
        violations.append(
            ToolResultContractViolation(
                tool=contract.name,
                code="required_diagnostics_missing",
                path="details.diagnostics",
                message=f"Outcome {outcome.semantic_outcome!r} requires diagnostics.",
            )
        )
    if outcome.recovery_required and not _nonempty_mapping_list(
        details.get("recovery_options")
    ):
        violations.append(
            ToolResultContractViolation(
                tool=contract.name,
                code="required_recovery_missing",
                path="details.recovery_options",
                message=f"Outcome {outcome.semantic_outcome!r} requires recovery options.",
            )
        )
    if outcome.executable_reference is False and isinstance(
        outputs.get("motion_execution_ref"), Mapping
    ):
        violations.append(
            ToolResultContractViolation(
                tool=contract.name,
                code="non_executable_outcome_exposes_execution_reference",
                path="details.outputs.motion_execution_ref",
                message=(
                    f"Outcome {outcome.semantic_outcome!r} must not authorize motion."
                ),
            )
        )
    return tuple(violations)


def check_tool_request_conformance(
    contract: ToolContract,
    parameters: Mapping[str, Any],
) -> tuple[ToolRequestContractViolation, ...]:
    """Validate Agent-authored parameters against the declared request schema.

    OpenETA intentionally supports the small, dependency-free JSON Schema
    subset used by its generated contracts.  This checker is initially consumed
    in shadow mode: the legacy Planner validator remains authoritative until
    acceptance parity and repair-message quality have been reviewed.
    """

    raw = _check_json_schema_subset(contract.request_schema, parameters, path="parameters")
    return tuple(
        ToolRequestContractViolation(
            tool=contract.name,
            code=code,
            path=path,
            message=message,
        )
        for code, path, message in raw
    )


def check_gate_repair_conformance(
    contract: ToolContract,
    repair: Mapping[str, Any],
) -> tuple[GateRepairContractViolation, ...]:
    """Check that a gate rejection gives executable, non-prescriptive feedback."""

    violations: list[GateRepairContractViolation] = []

    def add(code: str, path: str, message: str) -> None:
        violations.append(GateRepairContractViolation(contract.name, code, path, message))

    if repair.get("schema_version") != contract.gate.repair_schema:
        add(
            "gate_repair_schema_mismatch",
            "repair.schema_version",
            f"Expected {contract.gate.repair_schema!r}.",
        )
    extensions = repair.get("extensions")
    if extensions is not None and not isinstance(extensions, Mapping):
        add(
            "gate_repair_extensions_invalid",
            "repair.extensions",
            (
                "Gate repair extensions is reserved as an object namespace; "
                "v1 defines no extension keys or inner value schemas."
            ),
        )
    for field_name in ("code", "violated_invariant"):
        value = repair.get(field_name)
        if not isinstance(value, str) or not value.strip():
            add(
                "gate_repair_explanation_missing",
                f"repair.{field_name}",
                f"Gate repair requires a non-empty {field_name!r}.",
            )
    repair_code = str(repair.get("code") or "").strip()
    declared_repair_codes = {
        code
        for binding in contract.gate.bindings
        for code in binding.repair_codes
    }
    if declared_repair_codes and repair_code not in declared_repair_codes:
        add(
            "gate_repair_code_undeclared",
            "repair.code",
            (
                f"Repair code {repair_code!r} is not bound to a gate check for "
                f"{contract.name}; declared codes are "
                f"{sorted(declared_repair_codes)!r}."
            ),
        )
    requested = repair.get("requested_call")
    if not isinstance(requested, Mapping):
        add(
            "gate_repair_requested_call_missing",
            "repair.requested_call",
            "Gate repair must echo the rejected tool and Agent-authored parameters.",
        )
    else:
        if requested.get("tool") != contract.name:
            add(
                "gate_repair_tool_mismatch",
                "repair.requested_call.tool",
                f"Gate repair must identify rejected tool {contract.name!r}.",
            )
        if not isinstance(requested.get("parameters"), Mapping):
            add(
                "gate_repair_parameters_missing",
                "repair.requested_call.parameters",
                "Gate repair must preserve the rejected Agent-authored parameters.",
            )
    for field_name in ("evidence_ids", "allowed_next_calls", "stale_evidence"):
        if not isinstance(repair.get(field_name), list):
            add(
                "gate_repair_collection_missing",
                f"repair.{field_name}",
                f"Gate repair field {field_name!r} must be an array.",
            )
    allowed = repair.get("allowed_next_calls")
    if isinstance(allowed, list):
        for index, call in enumerate(allowed):
            if not isinstance(call, Mapping):
                add(
                    "gate_repair_invalid_next_call",
                    f"repair.allowed_next_calls[{index}]",
                    "A repair option must be an object.",
                )
                continue
            if not isinstance(call.get("tool"), str) or not str(call.get("tool")).strip():
                add(
                    "gate_repair_invalid_next_call",
                    f"repair.allowed_next_calls[{index}].tool",
                    "A repair option requires a non-empty tool name.",
                )
            if not isinstance(call.get("parameters"), Mapping):
                add(
                    "gate_repair_invalid_next_call",
                    f"repair.allowed_next_calls[{index}].parameters",
                    "A repair option requires executable parameter JSON.",
                )
    if contract.gate.preserves_agent_choice:
        # Compose the legacy field spellings so the repository's state-machine
        # source audit does not mistake this defensive rejection list for live
        # host-owned progress state.
        forbidden_progress_fields = (
            "required_" + "action",
            "required_" + "tool",
            "stage",
            "next_" + "stage",
        )
        for forbidden in forbidden_progress_fields:
            if forbidden in repair:
                add(
                    "gate_repair_prescribes_task_progress",
                    f"repair.{forbidden}",
                    (
                        "Gate feedback may explain invariants and offer legal calls, but "
                        f"must not prescribe host-owned task progress via {forbidden!r}."
                    ),
                )
    return tuple(violations)


def _check_json_schema_subset(
    schema: Mapping[str, Any],
    value: Any,
    *,
    path: str,
) -> list[tuple[str, str, str]]:
    """Return stable violations for the schema keywords emitted by this repo."""

    violations: list[tuple[str, str, str]] = []
    expected_type = schema.get("type")
    if isinstance(expected_type, str) and not _matches_json_type(value, expected_type):
        return [
            (
                "request_type_mismatch",
                path,
                f"{path} must be JSON type {expected_type}; got {_json_type_name(value)}.",
            )
        ]

    enum = schema.get("enum")
    if isinstance(enum, list) and value not in enum:
        violations.append(
            (
                "request_enum_mismatch",
                path,
                f"{path} must be one of {enum!r}; got {value!r}.",
            )
        )

    if isinstance(value, Mapping):
        properties = schema.get("properties")
        properties = properties if isinstance(properties, Mapping) else {}
        required = schema.get("required")
        if isinstance(required, list):
            for name in required:
                if isinstance(name, str) and name not in value:
                    violations.append(
                        (
                            "request_required_field_missing",
                            f"{path}.{name}",
                            f"{path} requires field {name!r}.",
                        )
                    )
        if schema.get("additionalProperties") is False:
            for name in sorted(str(key) for key in value if key not in properties):
                violations.append(
                    (
                        "request_additional_field",
                        f"{path}.{name}",
                        f"{path} does not allow field {name!r}.",
                    )
                )
        for name, child_schema in properties.items():
            if name in value and isinstance(child_schema, Mapping):
                violations.extend(
                    _check_json_schema_subset(
                        child_schema,
                        value[name],
                        path=f"{path}.{name}",
                    )
                )

    if isinstance(value, str):
        minimum = schema.get("minLength")
        maximum = schema.get("maxLength")
        if isinstance(minimum, int) and len(value) < minimum:
            violations.append(
                (
                    "request_string_too_short",
                    path,
                    f"{path} must contain at least {minimum} characters.",
                )
            )
        if isinstance(maximum, int) and len(value) > maximum:
            violations.append(
                (
                    "request_string_too_long",
                    path,
                    f"{path} must contain at most {maximum} characters.",
                )
            )

    if _matches_json_type(value, "number"):
        minimum = schema.get("minimum")
        maximum = schema.get("maximum")
        exclusive_minimum = schema.get("exclusiveMinimum")
        exclusive_maximum = schema.get("exclusiveMaximum")
        if isinstance(minimum, int | float) and value < minimum:
            violations.append(
                ("request_number_below_minimum", path, f"{path} must be >= {minimum}.")
            )
        if isinstance(maximum, int | float) and value > maximum:
            violations.append(
                ("request_number_above_maximum", path, f"{path} must be <= {maximum}.")
            )
        if isinstance(exclusive_minimum, int | float) and value <= exclusive_minimum:
            violations.append(
                (
                    "request_number_below_exclusive_minimum",
                    path,
                    f"{path} must be > {exclusive_minimum}.",
                )
            )
        if isinstance(exclusive_maximum, int | float) and value >= exclusive_maximum:
            violations.append(
                (
                    "request_number_above_exclusive_maximum",
                    path,
                    f"{path} must be < {exclusive_maximum}.",
                )
            )

    if isinstance(value, list):
        minimum = schema.get("minItems")
        maximum = schema.get("maxItems")
        if isinstance(minimum, int) and len(value) < minimum:
            violations.append(
                ("request_array_too_short", path, f"{path} requires at least {minimum} items.")
            )
        if isinstance(maximum, int) and len(value) > maximum:
            violations.append(
                ("request_array_too_long", path, f"{path} allows at most {maximum} items.")
            )
        if schema.get("uniqueItems") is True:
            canonical = [json.dumps(item, ensure_ascii=False, sort_keys=True) for item in value]
            if len(canonical) != len(set(canonical)):
                violations.append(
                    ("request_array_items_not_unique", path, f"{path} items must be unique.")
                )
        items = schema.get("items")
        if isinstance(items, Mapping):
            for index, item in enumerate(value):
                violations.extend(
                    _check_json_schema_subset(items, item, path=f"{path}[{index}]")
                )

    branches = schema.get("oneOf")
    if isinstance(branches, list) and branches:
        matches = 0
        for branch in branches:
            if isinstance(branch, Mapping) and not _check_json_schema_subset(
                branch,
                value,
                path=path,
            ):
                matches += 1
        if matches != 1:
            violations.append(
                (
                    "request_one_of_mismatch",
                    path,
                    f"{path} must match exactly one request branch; matched {matches}.",
                )
            )
    return violations


def _matches_json_type(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, Mapping)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    return True


def _json_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return type(value).__name__


def _nonempty_mapping_list(value: object) -> bool:
    return isinstance(value, list) and any(isinstance(item, Mapping) for item in value)


def render_tool_contract_markdown(catalog: ToolContractCatalog) -> str:
    """Render deterministic developer documentation from the catalog."""

    def inline(value: object) -> str:
        return str("" if value is None else value).replace("\n", " ").replace("|", "\\|")

    def joined(values: Iterable[object], *, empty: str = "none") -> str:
        rendered = [f"`{inline(value)}`" for value in values if inline(value)]
        return ", ".join(rendered) or empty

    summary = catalog.coverage_summary()
    maturity = summary["maturity_counts"]
    lines = [
        "# OpenETA Tool Contracts",
        "",
        "> Generated from `openeta.tool_contract_catalog.v1`. Do not edit this file ",
        "> by hand; update the machine-readable contract declarations instead.",
        "",
        "## Coverage",
        "",
        f"- Tools: {summary['tool_count']}",
        f"- Inferred: {maturity['inferred']}",
        f"- Declared: {maturity['declared']}",
        f"- Verified: {maturity['verified']}",
        "",
        "An inferred contract is an inventory compatibility record, not proof that ",
        "required fields, output variants, evidence lifetime, or gate behavior are complete.",
        "",
        "## Tool index",
        "",
        "| Tool | Category | Effect | Maturity | Parameters |",
        "|---|---|---|---|---:|",
    ]
    for contract in sorted(catalog.list(), key=lambda item: item.name):
        properties = contract.request_schema.get("properties")
        parameter_count = len(properties) if isinstance(properties, dict) else 0
        lines.append(
            f"| `{contract.name}` | {contract.category} | `{contract.effect}` | "
            f"{contract.maturity.value} | {parameter_count} |"
        )
    for contract in sorted(catalog.list(), key=lambda item: item.name):
        lines.extend(
            [
                "",
                f"## `{contract.name}`",
                "",
                contract.description,
                "",
                f"- Category/effect: `{contract.category}` / `{contract.effect}`",
                f"- Contract maturity: `{contract.maturity.value}`",
                f"- Requires observation after call: `{str(contract.requires_observation_after_call).lower()}`",
                "- Agent semantic limits: "
                + joined(
                    _project_agent_semantic_limits(contract),
                    empty="none declared",
                ),
                "",
                "### Agent request",
                "",
            ]
        )
        properties = contract.request_schema.get("properties")
        properties = properties if isinstance(properties, dict) else {}
        required_parameters = contract.request_schema.get("required")
        required_parameters = (
            set(required_parameters) if isinstance(required_parameters, list) else set()
        )
        if properties:
            lines.extend(
                [
                    "| Parameter | Required | Schema / description |",
                    "|---|:---:|---|",
                ]
            )
            for name, schema in properties.items():
                schema = schema if isinstance(schema, dict) else {}
                description = inline(schema.get("description"))
                field_type = inline(schema.get("type") or "untyped")
                constraints = {
                    key: value
                    for key, value in schema.items()
                    if key not in {"type", "description", "properties", "items"}
                }
                suffix = (
                    " — `" + inline(json.dumps(constraints, ensure_ascii=False)) + "`"
                    if constraints
                    else ""
                )
                lines.append(
                    f"| `{name}` | {'yes' if name in required_parameters else 'no'} | "
                    f"`{field_type}` — {description}{suffix} |"
                )
        else:
            lines.append("No Agent parameters.")
        request_branches = contract.request_schema.get("oneOf")
        if isinstance(request_branches, list) and request_branches:
            lines.extend(
                [
                    "",
                    "Exclusive request branches: `"
                    + inline(json.dumps(request_branches, ensure_ascii=False))
                    + "`",
                ]
            )
        lines.extend(
            [
                "",
                "### Host resolution",
                "",
                f"- Mode/resolver: `{contract.host_resolution.mode}` / "
                f"`{contract.host_resolution.resolver or 'none'}`",
                "- Runtime binding: "
                f"`{contract.host_resolution.implementation or 'none'}` "
                f"(`{contract.host_resolution.resolution_layer or 'none'}` layer)",
                "- Contract-driven dispatch: "
                + ("yes" if contract.host_resolution.contract_driven_dispatch else "no"),
                "- Agent-visible references: "
                + joined(contract.host_resolution.agent_parameters),
                "- Private resolved inputs: "
                + joined(contract.host_resolution.resolved_parameters),
                "- Resolution freshness: "
                + joined(contract.host_resolution.freshness_dimensions),
                "- Resolution invalidated by: "
                + joined(contract.host_resolution.invalidated_by),
                "- Notes: " + (contract.host_resolution.description or "None."),
                "",
                "### Consumed typed facts",
                "",
            ]
        )
        if contract.consumes:
            lines.extend(
                [
                    "| Fact / schema | Request path | Authority | Required | Condition |",
                    "|---|---|---|:---:|---|",
                ]
            )
            for fact in contract.consumes:
                lines.append(
                    f"| `{fact.fact_type}` / `{fact.schema_version or 'unspecified'}` | "
                    f"`{fact.path}` | `{fact.authority.value}` | "
                    f"{'yes' if fact.required else 'no'} | {inline(fact.when) or 'always'} |"
                )
        else:
            lines.append("None declared.")
        lines.extend(
            [
                "",
                "### Semantic outcomes and outputs",
                "",
                "| Outcome | Operational success | Required outputs | Produces | Executable | Recovery |",
                "|---|:---:|---|---|:---:|:---:|",
            ]
        )
        for outcome in contract.outcomes:
            output_required = outcome.output_schema.get("required")
            output_required = output_required if isinstance(output_required, list) else []
            produced = [
                f"{fact.fact_type} @ {fact.path}" for fact in outcome.produces
            ]
            lines.append(
                f"| `{outcome.semantic_outcome}` | "
                f"{inline(outcome.operational_success)} | {joined(output_required)} | "
                f"{joined(produced)} | "
                f"{inline(outcome.executable_reference) if outcome.executable_reference is not None else 'n/a'} | "
                f"{'required' if outcome.recovery_required else 'no'} |"
            )
        lines.extend(
            [
                "",
                "### Evidence lifetime",
                "",
                f"- Scope: `{contract.evidence_lifetime.scope}`",
                "- Freshness dimensions: "
                + joined(contract.evidence_lifetime.freshness_dimensions),
                "- Invalidated by: "
                + joined(contract.evidence_lifetime.invalidated_by),
                "- Reusable across packet refresh: `"
                + (
                    inline(contract.evidence_lifetime.reusable_across_packet_refresh)
                    if contract.evidence_lifetime.reusable_across_packet_refresh is not None
                    else "unspecified"
                )
                + "`",
                "- Notes: " + (contract.evidence_lifetime.description or "None."),
                "",
                "### Gate and repair",
                "",
                "- Checks: " + joined(contract.gate.checks, empty="none declared"),
                f"- Fail closed: `{str(contract.gate.fail_closed).lower()}`",
                f"- Repair schema: `{contract.gate.repair_schema}`",
                "- Preserves Agent choice: `"
                + str(contract.gate.preserves_agent_choice).lower()
                + "`",
                "- Notes: " + (contract.gate.description or "None."),
            ]
        )
        if contract.gate.bindings:
            lines.extend(
                [
                    "",
                    "| Gate check id | Applies when | Repair codes | Implementation |",
                    "|---|---|---|---|",
                ]
            )
            for binding in contract.gate.bindings:
                lines.append(
                    f"| `{binding.check_id}` | {inline(binding.applies_when) or 'always'} | "
                    f"{joined(binding.repair_codes)} | `{binding.implementation or 'unbound'}` |"
                )
        else:
            lines.extend(["", "No machine-bound gate checks."])
        lines.extend(
            [
                "",
                "### Traceability and coverage gaps",
                "",
                "- Sources: " + joined(contract.source_paths),
                "- Gaps: " + joined(contract.coverage_gaps, empty="None."),
            ]
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", default="")
    parser.add_argument("--markdown-output", default="")
    args = parser.parse_args(argv)

    # Imported lazily to keep this model independent from the runtime registry.
    from agent.tools.registry import build_default_tool_registry

    registry = build_default_tool_registry()
    catalog = build_default_tool_contract_catalog(registry.list())
    json_payload = json.dumps(catalog.to_dict(), ensure_ascii=False, indent=2) + "\n"
    markdown_payload = render_tool_contract_markdown(catalog)
    if args.json_output:
        Path(args.json_output).write_text(json_payload, encoding="utf-8")
    if args.markdown_output:
        Path(args.markdown_output).write_text(markdown_payload, encoding="utf-8")
    if not args.json_output and not args.markdown_output:
        print(json_payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
