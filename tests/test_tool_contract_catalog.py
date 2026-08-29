from __future__ import annotations

import json
from dataclasses import replace

import pytest

from agent.tools.contracts import (
    ContractMaturity,
    FactBinding,
    OutcomeContract,
    ToolContract,
    ToolContractCatalog,
    ToolContractRuntimePolicy,
    build_default_tool_contract_catalog,
    build_tool_contract_catalog,
    check_gate_repair_conformance,
    check_tool_chain_compatibility,
    check_tool_request_conformance,
    check_tool_result_conformance,
    audit_agent_tool_projection,
    project_agent_tool_contract,
    render_tool_contract_markdown,
)
from agent.tools.runtime_contract_bindings import (
    HostResolutionFailure,
    audit_host_resolver_bindings,
    resolve_host_parameters,
)
from agent.tools.default_contracts import (
    COMPILED_GRASP,
    PLACEMENT_INPUT_BUNDLE,
)
from agent.tools.coding import PythonExecConfig, PythonExecRuntime
from agent.tools.handlers import bind_dummy_tool_handlers
from agent.tools.registry import build_default_tool_registry
from agent.runtime.planner import (
    PlannerDecision,
    _tool_contract_shadow_validation,
    _validate_planner_decision,
)
from agent.runtime.skills import build_default_skill_registry
from agent.runtime.pipeline import _gate_contract_shadow_validation
from agent.runtime.pipeline import ActionPipeline
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.runtime.planner import ToolCallingPlanner
from agent.evals.tool_contract_readiness import audit_tool_contract_request_readiness


def _schema_fixture(schema: dict) -> object:
    schema_type = schema.get("type")
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return enum[0]
    if schema_type == "string":
        return "x" * max(1, int(schema.get("minLength") or 0))
    if schema_type == "integer":
        value = int(schema.get("minimum") or 0)
        if "exclusiveMinimum" in schema:
            value = int(schema["exclusiveMinimum"]) + 1
        return value
    if schema_type == "number":
        value = float(schema.get("minimum") or 0.0)
        if "exclusiveMinimum" in schema:
            value = float(schema["exclusiveMinimum"]) + 1.0
        return value
    if schema_type == "boolean":
        return False
    if schema_type == "array":
        count = int(schema.get("minItems") or 0)
        item_schema = schema.get("items")
        item_schema = item_schema if isinstance(item_schema, dict) else {}
        return [_schema_fixture(item_schema) for _ in range(count)]
    if schema_type == "object":
        properties = schema.get("properties")
        properties = properties if isinstance(properties, dict) else {}
        required = [
            str(name)
            for name in schema.get("required", [])
            if isinstance(name, str)
        ]
        branches = schema.get("oneOf")
        if isinstance(branches, list) and branches and isinstance(branches[0], dict):
            required.extend(
                str(name)
                for name in branches[0].get("required", [])
                if isinstance(name, str)
            )
        return {
            name: _schema_fixture(
                properties.get(name) if isinstance(properties.get(name), dict) else {}
            )
            for name in dict.fromkeys(required)
        }
    return {"fixture": True}


def test_contract_catalog_inventories_every_default_tool_without_behavior_change() -> None:
    registry = build_default_tool_registry()
    catalog = build_tool_contract_catalog(registry.list())

    assert [item.name for item in catalog.list()] == [
        item.name for item in registry.list()
    ]
    assert catalog.coverage_summary()["tool_count"] == 35
    assert catalog.coverage_summary()["maturity_counts"] == {
        "inferred": 35,
        "declared": 0,
        "verified": 0,
    }
    for spec in registry.list():
        contract = catalog.get(spec.name)
        assert contract.maturity is ContractMaturity.INFERRED
        assert set(contract.request_schema["properties"]) == set(spec.parameters)
        assert contract.request_schema["additionalProperties"] is True
        assert contract.effect == spec.effect.value
        assert contract.requires_observation_after_call == (
            spec.requires_observation_after_call
        )


def test_contract_catalog_is_json_serializable_and_versioned() -> None:
    catalog = build_tool_contract_catalog(build_default_tool_registry().list())

    payload = json.loads(json.dumps(catalog.to_dict()))

    assert payload["schema_version"] == "openeta.tool_contract_catalog.v1"
    assert payload["tools"][0]["schema_version"] == "openeta.tool_contract.v1"
    assert payload["summary"]["coverage_gap_counts"]["gate_checks"] == 35


def test_markdown_generation_is_deterministic_and_covers_all_tools() -> None:
    catalog = build_default_tool_contract_catalog(build_default_tool_registry().list())

    first = render_tool_contract_markdown(catalog)
    second = render_tool_contract_markdown(catalog)

    assert first == second
    assert "Tools: 35" in first
    assert "## `sam3`" in first
    assert "## `move_to`" in first
    assert "Do not edit this file" in first
    assert "`openeta.host_resolver.estimate_depth_prior.v1`" in first
    assert (
        "`agent.tools.runtime_contract_bindings._resolve_estimate_depth_prior_input` "
        "(`pipeline` layer)"
    ) in first
    assert "- Contract-driven dispatch: yes" in first


def test_tool_contract_rejects_duplicate_outcomes() -> None:
    with pytest.raises(ValueError, match="duplicate semantic outcomes"):
        ToolContract(
            name="example",
            category="planning",
            description="example",
            effect="read_only",
            request_schema={"type": "object"},
            outcomes=(
                OutcomeContract("completed", True),
                OutcomeContract("completed", False),
            ),
        )


def test_fact_binding_requires_known_authority() -> None:
    with pytest.raises(ValueError, match="not a valid FactAuthority"):
        FactBinding(
            fact_type="selected_target_mask",
            path="outputs.selected_detection",
            authority="model_guess",
        )


def test_default_catalog_verifies_every_reviewed_public_tool() -> None:
    registry = build_default_tool_registry()

    catalog = build_default_tool_contract_catalog(registry.list())

    assert catalog.coverage_summary()["maturity_counts"] == {
        "inferred": 0,
        "declared": 0,
        "verified": 35,
    }
    explicit = [
        contract for contract in catalog.list() if contract.maturity is not ContractMaturity.INFERRED
    ]
    assert all(contract.gate.preserves_agent_choice for contract in explicit)
    assert catalog.get("estimate_depth_prior").maturity is ContractMaturity.VERIFIED
    assert all(
        contract.maturity is ContractMaturity.VERIFIED for contract in explicit
    )
    assert all(not contract.coverage_gaps for contract in explicit)
    assert catalog.get("move_to").request_schema["required"] == ["ik_receipt_id"]
    assert catalog.get("gripper_control").request_schema["additionalProperties"] is False
    assert {
        "scene_detector",
        "anygrasp",
        "graspgenx",
        "list_graspgenx_grippers",
        "contact_graspnet",
        "anydexgrasp",
        "slam",
        "lower_body_control_policy",
        "hand_pose_database",
        "obstacle_avoidance",
    }.isdisjoint({contract.name for contract in catalog.list()})


def test_registry_level_missing_handler_failure_is_contract_conformant() -> None:
    registry = build_default_tool_registry()
    catalog = build_default_tool_contract_catalog(registry.list())

    result = registry.call(
        "save_memory",
        {"key": "fixture", "content": {"value": 1}},
    )

    assert result.success is False
    assert result.details["semantic_outcome"] == "operational_failure"
    assert result.details["recovery_options"]
    assert check_tool_result_conformance(
        catalog.get("save_memory"),
        result.details,
    ) == ()


def test_environment_and_memory_contracts_expose_non_task_state_facts() -> None:
    catalog = build_default_tool_contract_catalog(build_default_tool_registry().list())

    environment = check_tool_chain_compatibility(
        catalog,
        ("create_simulator_env", "close_simulator_env"),
    )
    memory = check_tool_chain_compatibility(
        catalog,
        ("save_memory", "get_memory", "delete_memory", "compact_memory"),
    )

    assert environment.compatible is True
    assert memory.compatible is True
    assert catalog.get("python_exec").gate.checks[-1] == (
        "execution timeout stays within host cap"
    )
    assert check_tool_chain_compatibility(
        catalog,
        ("register_skill", "update_skill"),
    ).compatible is True


def test_all_declared_host_resolvers_have_stable_runtime_bindings() -> None:
    catalog = build_default_tool_contract_catalog(build_default_tool_registry().list())

    report = audit_host_resolver_bindings(catalog)

    assert report == {
        "schema_version": "openeta.host_resolver_binding_audit.v1",
        "bound_count": 30,
        "contract_driven_dispatch_count": 11,
        "contract_driven_dispatch_tools": [
            "anyplace",
            "camera_pose_to_world",
            "compile_grasp_seed",
            "compute_wrist_alignment",
            "enhance_depth",
            "estimate_depth_prior",
            "grasp_pose_estimate",
            "molmopoint",
            "propose_wrist_viewpoints",
            "retrieve_asset_reference",
            "sam3",
        ],
        "identity_only_count": 19,
        "identity_only_tools": [
            "assess_attachment_probe",
            "close_simulator_env",
            "create_simulator_env",
            "follow_eef_trajectory",
            "ik_preview_check",
            "move_to",
            "observe",
            "prepare_attachment_probe",
            "promote_calibration_profile",
            "promote_grasp_strategy",
            "propose_calibration_profile",
            "propose_grasp_strategy",
            "python_exec",
            "register_skill",
            "reject_sam3_detections",
            "select_sam3_detection",
            "update_skill",
            "web_fetch",
            "web_search",
        ],
        "issue_count": 0,
        "conformant": True,
        "issues": [],
    }
    estimate = catalog.get("estimate_depth_prior").host_resolution
    assert estimate.resolver == "openeta.host_resolver.estimate_depth_prior.v1"
    assert estimate.implementation.endswith("._resolve_estimate_depth_prior_input")
    assert estimate.resolution_layer == "pipeline"
    assert estimate.contract_driven_dispatch is True
    assert catalog.get("sam3").host_resolution.contract_driven_dispatch is True
    assert catalog.get("molmopoint").host_resolution.contract_driven_dispatch is True
    assert catalog.get("move_to").host_resolution.contract_driven_dispatch is False
    assert catalog.get("move_to").host_resolution.implementation.endswith(
        "resolve_ik_motion_reference"
    )
    assert catalog.get("follow_eef_trajectory").host_resolution.implementation.endswith(
        "resolve_ik_trajectory_reference"
    )


def test_host_resolution_dispatch_fails_closed_on_contract_identity_drift() -> None:
    with pytest.raises(HostResolutionFailure) as exc_info:
        resolve_host_parameters(
            tool_name="estimate_depth_prior",
            resolver_id="openeta.host_resolver.wrong.v1",
            parameters={},
            memory=None,
        )

    assert exc_info.value.repair_code == "host_resolver_identity_mismatch"


def test_host_resolver_audit_requires_dispatch_failure_code_gate_binding() -> None:
    contract = build_default_tool_contract_catalog(
        build_default_tool_registry().list()
    ).get("estimate_depth_prior")
    unbound = replace(contract, gate=replace(contract.gate, bindings=()))

    report = audit_host_resolver_bindings(ToolContractCatalog([unbound]))

    assert report["conformant"] is False
    assert report["issues"] == [
        {
            "tool": "estimate_depth_prior",
            "code": "resolver_failure_code_unbound",
            "observed": "invalid_source_packet",
        }
    ]


def test_core_pick_chain_is_statically_composable_by_typed_facts() -> None:
    catalog = build_default_tool_contract_catalog(build_default_tool_registry().list())

    report = check_tool_chain_compatibility(
        catalog,
        (
            "observe",
            "sam3",
            "select_sam3_detection",
            "grasp_pose_estimate",
            "compile_grasp_seed",
            "ik_preview_check",
            "move_to",
            "gripper_control",
        ),
    )

    assert report.compatible is True
    assert report.issues == ()
    assert "Static producer-consumer compatibility only" in report.to_dict()[
        "interpretation"
    ]


def test_partial_host_bundle_chains_require_explicit_initial_fact() -> None:
    catalog = build_default_tool_contract_catalog(build_default_tool_registry().list())

    missing = check_tool_chain_compatibility(catalog, ("anyplace", "camera_pose_to_world"))
    supplied = check_tool_chain_compatibility(
        catalog,
        ("anyplace", "camera_pose_to_world"),
        initial_facts=(PLACEMENT_INPUT_BUNDLE,),
    )
    wrist = check_tool_chain_compatibility(
        catalog,
        ("observe", "propose_wrist_viewpoints", "ik_preview_check", "move_to"),
        initial_facts=(COMPILED_GRASP,),
    )

    assert missing.compatible is False
    assert missing.issues[0].code == "missing_required_fact"
    assert supplied.compatible is True
    assert wrist.compatible is True


def test_result_conformance_checks_success_outcome_and_required_outputs() -> None:
    catalog = build_default_tool_contract_catalog(build_default_tool_registry().list())
    contract = catalog.get("grasp_pose_estimate")

    valid = check_tool_result_conformance(
        contract,
        {
            "semantic_outcome": "candidates_available",
            "operational_success": True,
            "outputs": {
                "schema_version": "openeta.grasp_pose_estimate.v1",
                "result_id": "gpe-1",
                "grasp_candidates": [{"id": "candidate-1"}],
            },
            "diagnostics": [],
            "recovery_options": [],
        },
    )
    invalid = check_tool_result_conformance(
        contract,
        {
            "semantic_outcome": "candidates_available",
            "operational_success": True,
            "outputs": {"grasp_candidates": []},
            "diagnostics": [],
            "recovery_options": [],
        },
    )

    assert valid == ()
    assert {item.path for item in invalid} == {
        "details.outputs.schema_version",
        "details.outputs.result_id",
    }


def test_result_conformance_rejects_execution_ref_on_non_executable_ik() -> None:
    catalog = build_default_tool_contract_catalog(build_default_tool_registry().list())

    violations = check_tool_result_conformance(
        catalog.get("ik_preview_check"),
        {
            "semantic_outcome": "ik_hard_infeasible",
            "operational_success": True,
            "outputs": {
                "ik_preview_receipt": {"classification": "hard_infeasible"},
                "ik_receipt_id": "ik-1",
                "motion_execution_ref": {"ik_receipt_id": "ik-1"},
            },
            "diagnostics": [{"code": "ik_hard_infeasible"}],
            "recovery_options": [{"action": "adjust_pose"}],
        },
    )

    assert [item.code for item in violations] == [
        "non_executable_outcome_exposes_execution_reference"
    ]


def test_request_conformance_checks_nested_types_bounds_and_extra_fields() -> None:
    catalog = build_default_tool_contract_catalog(build_default_tool_registry().list())

    assert check_tool_request_conformance(
        catalog.get("gripper_control"),
        {"position": 0},
    ) == ()
    violations = check_tool_request_conformance(
        catalog.get("gripper_control"),
        {"position": 0.5, "measured_aperture": 0.103},
    )

    assert {item.code for item in violations} == {
        "request_enum_mismatch",
        "request_additional_field",
    }


def test_agent_projection_preserves_gripper_command_direction() -> None:
    catalog = build_default_tool_contract_catalog(build_default_tool_registry().list())

    projected = project_agent_tool_contract(catalog.get("gripper_control"))

    position = projected["parameters"]["properties"]["position"]
    assert "0 or false closes" in position["description"]
    assert "1 or true opens" in position["description"]


def test_request_conformance_checks_exactly_one_request_branch() -> None:
    catalog = build_default_tool_contract_catalog(build_default_tool_registry().list())
    contract = catalog.get("ik_preview_check")

    assert check_tool_request_conformance(
        contract,
        {"compiled_grasp_id": "compiled-1", "waypoint_role": "grasp_contact"},
    ) == ()
    assert check_tool_request_conformance(
        contract,
        {"probe_id": "probe:one", "waypoint_index": 0},
    ) == ()
    violations = check_tool_request_conformance(
        contract,
        {
            "target_pose": {"position": [0, 0, 0], "orientation": [0, 0, 0, 1]},
            "compiled_grasp_id": "compiled-1",
            "waypoint_role": "grasp_contact",
        },
    )

    assert [item.code for item in violations] == ["request_one_of_mismatch"]

    mixed_probe = check_tool_request_conformance(
        contract,
        {
            "target_pose": {"xyz": [0, 0, 0.3]},
            "probe_id": "probe:one",
            "waypoint_index": 0,
        },
    )
    assert [item.code for item in mixed_probe] == ["request_one_of_mismatch"]


def test_default_contract_overrides_support_partial_runtime_registries() -> None:
    registry = build_default_tool_registry()
    partial = [registry.get("sam3"), registry.get("move_to")]

    catalog = build_default_tool_contract_catalog(partial)

    assert [contract.name for contract in catalog.list()] == ["sam3", "move_to"]
    assert all(contract.maturity is ContractMaturity.VERIFIED for contract in catalog.list())


def test_agent_tool_projection_uses_contract_request_schema() -> None:
    registry = build_default_tool_registry()
    contract = build_default_tool_contract_catalog(registry.list()).get(
        "gripper_control"
    )

    projection = project_agent_tool_contract(contract)

    assert set(projection) == {
        "name",
        "description",
        "parameters",
        "returns",
        "semantic_limits",
    }
    parameters_without_annotations = json.loads(json.dumps(projection["parameters"]))
    parameters_without_annotations["properties"]["position"].pop("description")
    assert parameters_without_annotations == contract.request_schema
    assert "description" not in contract.request_schema["properties"]["position"]
    assert projection["parameters"]["required"] == ["position"]
    assert projection["parameters"]["properties"]["position"]["enum"] == [
        0,
        1,
        False,
        True,
    ]
    assert projection["returns"] == {
        "outcomes": [
            "mutation_acknowledged",
            "requires_attachment_probe",
            "no_attachment_evidence",
            "attachment_contract_unavailable",
        ],
        "fields": ["response", "attachment_proxy_receipt"],
    }
    assert "world_mutating" in projection["semantic_limits"]
    assert "command_latched" in projection["semantic_limits"]
    assert "command_ack_not_attachment" in projection["semantic_limits"]


def test_agent_tool_projection_audit_exposes_legacy_host_only_arguments() -> None:
    registry = build_default_tool_registry()
    contract = build_default_tool_contract_catalog(registry.list()).get(
        "grasp_pose_estimate"
    )

    audit = audit_agent_tool_projection(
        contract,
        registry.get("grasp_pose_estimate"),
    )

    assert audit["matches"] is False
    assert audit["contract_only_parameters"] == []
    assert set(audit["legacy_only_parameters"]) == {
        "camera_frame_id",
        "depth",
        "hints",
        "intrinsics",
        "mode",
        "object_mask",
        "rgb",
        "scene_epoch",
    }
    assert audit["field_matches"]["parameter_names"] is False


def test_planner_contract_validation_is_shadow_only_and_records_parity() -> None:
    registry = build_default_tool_registry()
    invalid = PlannerDecision(
        action_type="tool_call",
        action="gripper_control",
        parameters={"position": 0.5},
    )

    report = _tool_contract_shadow_validation(invalid, tools=registry)

    assert report is not None
    assert report["evaluated"] is True
    assert report["enforcing"] is False
    assert report["authoritative_validator"] == "legacy_planner"
    assert report["legacy_accepted"] is False
    assert report["contract_accepted"] is False
    assert report["acceptance_match"] is True


def test_planner_shadow_ignores_removed_non_public_tool() -> None:
    registry = build_default_tool_registry()
    decision = PlannerDecision(
        action_type="tool_call",
        action="scene_detector",
        parameters={},
    )

    report = _tool_contract_shadow_validation(decision, tools=registry)

    assert report is None


def test_runtime_policy_rejects_unverified_or_unknown_authority_entries() -> None:
    reviewed = build_default_tool_contract_catalog(build_default_tool_registry().list())
    catalog = ToolContractCatalog(
        replace(contract, maturity=ContractMaturity.DECLARED)
        if contract.name in {"gripper_control", "move_to"}
        else contract
        for contract in reviewed.list()
    )
    policy = ToolContractRuntimePolicy(
        request_validation_authority=frozenset({"gripper_control", "missing"}),
        gate_repair_envelope_authority=frozenset({"move_to"}),
    )

    errors = policy.validate_against(catalog)

    assert "request_validation: 'gripper_control' is declared, not verified" in errors
    assert "request_validation: unknown ToolContract 'missing'" in errors
    assert "gate_repair_envelope: 'move_to' is declared, not verified" in errors
    with pytest.raises(ValueError, match="not verified"):
        policy.ensure_valid(catalog)


def test_runtime_rejects_split_planner_and_pipeline_contract_authority() -> None:
    base = build_default_tool_contract_catalog(build_default_tool_registry().list())
    contracts = [
        replace(contract, maturity=ContractMaturity.VERIFIED)
        if contract.name == "estimate_depth_prior"
        else contract
        for contract in base.list()
    ]
    catalog = ToolContractCatalog(contracts)
    planner = ToolCallingPlanner(tool_contract_catalog=catalog)
    pipeline = ActionPipeline(
        tool_contract_catalog=catalog,
        tool_contract_policy=ToolContractRuntimePolicy(
            request_validation_authority=frozenset({"estimate_depth_prior"})
        ),
    )

    with pytest.raises(ValueError, match="runtime policies differ"):
        OpenEtaAgentRuntime(
            planner=planner,
            pipeline=pipeline,
            rollout_enabled=False,
        )


def test_verified_request_authority_can_diverge_from_legacy_per_tool() -> None:
    registry = build_default_tool_registry()
    bind_dummy_tool_handlers(registry)
    declared = build_default_tool_contract_catalog(registry.list()).get(
        "gripper_control"
    )
    verified = replace(
        declared,
        maturity=ContractMaturity.VERIFIED,
        request_schema={
            "type": "object",
            "properties": {
                "position": {"type": "number", "minimum": 0.0, "maximum": 1.0}
            },
            "required": ["position"],
            "additionalProperties": False,
        },
    )
    catalog = ToolContractCatalog([verified])
    policy = ToolContractRuntimePolicy(
        request_validation_authority=frozenset({"gripper_control"})
    )
    policy.ensure_valid(catalog)
    decision = PlannerDecision(
        action_type="tool_call",
        action="gripper_control",
        parameters={"position": 0.5},
    )

    errors = _validate_planner_decision(
        decision,
        registry,
        build_default_skill_registry(),
        tool_contract_catalog=catalog,
        tool_contract_policy=policy,
    )
    trace = _tool_contract_shadow_validation(
        decision,
        tools=registry,
        tool_contract_catalog=catalog,
        tool_contract_policy=policy,
    )

    assert errors == []
    assert trace is not None
    assert trace["legacy_accepted"] is False
    assert trace["contract_accepted"] is True
    assert trace["acceptance_match"] is False
    assert trace["enforcing"] is True
    assert trace["authoritative_validator"] == "tool_contract"


def test_verified_gate_repair_policy_never_replaces_executable_gate_authority() -> None:
    declared = build_default_tool_contract_catalog(
        build_default_tool_registry().list()
    ).get("move_to")
    verified = replace(declared, maturity=ContractMaturity.VERIFIED)
    catalog = ToolContractCatalog([verified])
    policy = ToolContractRuntimePolicy(
        gate_repair_envelope_authority=frozenset({"move_to"})
    )
    policy.ensure_valid(catalog)
    bundle = {
        "schema_version": "openeta.gate_repair.v1",
        "code": "invalid_ik_receipt_reference",
        "violated_invariant": "The IK receipt does not exist in this session.",
        "requested_call": {"tool": "move_to", "parameters": {}},
        "evidence_ids": [],
        "allowed_next_calls": [{"tool": "ik_preview_check", "parameters": {}}],
        "stale_evidence": [],
    }

    trace = _gate_contract_shadow_validation(
        "move_to",
        bundle,
        tool_contract_catalog=catalog,
        tool_contract_policy=policy,
    )

    assert trace["conformant"] is True
    assert trace["enforcing"] is True
    assert trace["repair_envelope_authority"] == "tool_contract"
    assert trace["authoritative_gate"] == "legacy_runtime"


def test_request_readiness_matrix_is_evidence_only_and_all_tools_have_parity() -> None:
    report = audit_tool_contract_request_readiness()

    assert report["authority"] == "promotion_evidence_only"
    assert report["declared_tool_count"] == 35
    assert report["ready_for_live_invalid_canary_count"] == 35
    assert len(report["ready_for_live_invalid_canary"]) == 35
    assert all(item["deterministic_request_parity"] is True for item in report["tools"])
    assert all(item["mismatch_cases"] == [] for item in report["tools"])


def test_gate_repair_contract_requires_clear_evidence_and_preserves_agent_choice() -> None:
    contract = build_default_tool_contract_catalog(
        build_default_tool_registry().list()
    ).get("move_to")
    valid = {
        "schema_version": "openeta.gate_repair.v1",
        "extensions": {},
        "code": "invalid_ik_receipt_reference",
        "violated_invariant": "The IK receipt does not exist in this session.",
        "requested_call": {
            "tool": "move_to",
            "parameters": {"ik_receipt_id": "missing"},
        },
        "evidence_ids": [],
        "allowed_next_calls": [
            {
                "tool": "ik_preview_check",
                "parameters": {"target_pose": {"position": [0.4, 0.0, 0.2]}},
            }
        ],
        "stale_evidence": [],
    }

    assert check_gate_repair_conformance(contract, valid) == ()
    violations = check_gate_repair_conformance(
        contract,
        {**valid, "required_action": "ik_preview_check"},
    )

    assert [item.code for item in violations] == [
        "gate_repair_prescribes_task_progress"
    ]

    undeclared = check_gate_repair_conformance(
        contract,
        {**valid, "code": "host_invented_task_stage"},
    )
    assert [item.code for item in undeclared] == [
        "gate_repair_code_undeclared"
    ]
    invalid_extensions = check_gate_repair_conformance(
        contract,
        {**valid, "extensions": ["not-yet-defined"]},
    )
    assert [item.code for item in invalid_extensions] == [
        "gate_repair_extensions_invalid"
    ]


def test_runtime_gate_bindings_have_stable_ids_implementations_and_repair_codes() -> None:
    catalog = build_default_tool_contract_catalog(build_default_tool_registry().list())
    move = catalog.get("move_to")

    bindings = {item.check_id: item for item in move.gate.bindings}

    assert "runtime.motion_reconciliation" in bindings
    assert "runtime.ik_receipt_resolution" in bindings
    assert "runtime.ik_execution_authorization" in bindings
    assert bindings["runtime.ik_receipt_resolution"].repair_codes == (
        "invalid_ik_receipt_reference",
    )
    assert bindings["runtime.ik_receipt_resolution"].implementation.endswith(
        "AgentMemory.resolve_ik_motion_reference"
    )
    assert all(not binding.check_id.startswith("stage.") for binding in bindings.values())


def test_every_declared_contract_has_conformant_structural_request_outcome_and_gate_fixtures() -> None:
    """Keep catalog-wide shapes testable before live rollout verification."""

    catalog = build_default_tool_contract_catalog(build_default_tool_registry().list())
    explicit = [
        contract
        for contract in catalog.list()
        if contract.maturity is not ContractMaturity.INFERRED
    ]

    assert len(explicit) == 35
    for contract in explicit:
        request = _schema_fixture(contract.request_schema)
        assert isinstance(request, dict)
        assert check_tool_request_conformance(contract, request) == (), contract.name

        for outcome in contract.outcomes:
            details = {
                "semantic_outcome": outcome.semantic_outcome,
                "operational_success": outcome.operational_success,
                "outputs": _schema_fixture(outcome.output_schema),
                "diagnostics": ([{"code": "fixture"}] if outcome.diagnostics_required else []),
                "recovery_options": (
                    [{"action": "fixture"}] if outcome.recovery_required else []
                ),
            }
            assert check_tool_result_conformance(contract, details) == (), (
                contract.name,
                outcome.semantic_outcome,
            )

        assert contract.gate.bindings, contract.name
        for binding in contract.gate.bindings:
            assert binding.check_id.startswith("runtime."), contract.name
            assert binding.implementation, (contract.name, binding.check_id)
            for repair_code in binding.repair_codes:
                repair = {
                    "schema_version": "openeta.gate_repair.v1",
                    "extensions": {},
                    "code": repair_code,
                    "violated_invariant": binding.description,
                    "requested_call": {"tool": contract.name, "parameters": request},
                    "evidence_ids": [],
                    "allowed_next_calls": [],
                    "stale_evidence": [],
                }
                assert check_gate_repair_conformance(contract, repair) == (), (
                    contract.name,
                    repair_code,
                )


def test_declared_python_exec_contract_matches_live_handler(tmp_path) -> None:
    registry = build_default_tool_registry()
    registry.bind_handler(
        "python_exec",
        PythonExecRuntime(
            PythonExecConfig(
                image_output_root=str(tmp_path / "images"),
                text_output_root=str(tmp_path / "text"),
                response_output_root=str(tmp_path / "responses"),
                structured_output_root=str(tmp_path / "structured"),
                workspace_root=str(tmp_path / "sandbox"),
            )
        ).handler,
    )

    result = registry.call("python_exec", {"code": "result = {'value': 7}"})
    contract = build_default_tool_contract_catalog(registry.list()).get("python_exec")

    assert result.success is True
    assert check_tool_result_conformance(contract, result.details) == ()


def test_observe_contract_distinguishes_handler_output_from_host_packet_fact() -> None:
    registry = build_default_tool_registry()
    bind_dummy_tool_handlers(registry)
    result = registry.call("observe", {"reason": "refresh evidence"})
    contract = build_default_tool_contract_catalog(registry.list()).get("observe")

    assert result.success is True
    assert "source_packet_id" not in result.details["outputs"]
    assert check_tool_result_conformance(contract, result.details) == ()
    produced = contract.outcomes[0].produces[0]
    assert produced.authority.value == "host"
    assert produced.path == "host.post_call_observation.source_packet_id"
