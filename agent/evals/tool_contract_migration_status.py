"""Audit whether the reviewed ToolContract migration is complete.

The report binds local implementation evidence to the durable three-person
review, narrow authority canary, and shared RFC sync receipts. Passing the audit
never grants authority beyond the policy recorded by those receipts.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from adapter.protocol import JsonDict
from agent.evals.tool_contract_readiness import (
    audit_tool_contract_request_readiness,
)
from agent.evals.tool_contract_promotion_campaign import (
    build_tool_contract_promotion_campaign,
)
from agent.tools.contracts import (
    ContractMaturity,
    build_default_tool_contract_catalog,
    check_tool_chain_compatibility,
    render_tool_contract_markdown,
)
from agent.tools.default_contracts import COMPILED_GRASP, PLACEMENT_INPUT_BUNDLE
from agent.tools.registry import build_default_tool_registry
from agent.tools.runtime_contract_bindings import audit_host_resolver_bindings


SCHEMA_VERSION = "openeta.tool_contract_migration_status.v1"


def audit_tool_contract_migration_status(
    repo_root: str | Path,
    *,
    test_passed: int = 0,
    test_skipped: int = 0,
    test_warnings: int = 0,
    harness_revision: int | None = None,
) -> JsonDict:
    """Build a fail-closed, requirement-by-requirement migration audit."""

    root = Path(repo_root)
    generated = root / "docs" / "generated"
    registry = build_default_tool_registry()
    catalog = build_default_tool_contract_catalog(registry.list())
    catalog_sha256 = hashlib.sha256(
        json.dumps(
            catalog.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    summary = catalog.coverage_summary()
    declared = [
        contract
        for contract in catalog.list()
        if contract.maturity is not ContractMaturity.INFERRED
    ]

    structural_issues: list[JsonDict] = []
    for contract in declared:
        if contract.request_schema.get("type") != "object":
            structural_issues.append(_issue(contract.name, "request_schema_not_object"))
        if not contract.outcomes:
            structural_issues.append(_issue(contract.name, "semantic_outcomes_missing"))
        if not contract.evidence_lifetime.scope.strip():
            structural_issues.append(_issue(contract.name, "evidence_scope_missing"))
        if (
            contract.host_resolution.mode != "none"
            and not contract.host_resolution.resolver.strip()
        ):
            structural_issues.append(_issue(contract.name, "stable_resolver_id_missing"))
        if not contract.gate.preserves_agent_choice:
            structural_issues.append(_issue(contract.name, "gate_prescribes_agent_choice"))

    catalog_projection = _projection_check(
        generated / "tool-contracts.json",
        catalog.to_dict(),
        exact_text=False,
    )
    markdown_projection = _projection_check(
        generated / "tool-contracts.md",
        render_tool_contract_markdown(catalog),
        exact_text=True,
    )

    readiness = audit_tool_contract_request_readiness(catalog)
    readiness_projection = _projection_check(
        generated / "tool-contract-readiness.json",
        readiness,
        exact_text=False,
    )
    resolver_audit = audit_host_resolver_bindings(catalog)
    resolver_projection = _projection_check(
        generated / "host-resolver-binding-audit.json",
        resolver_audit,
        exact_text=False,
    )

    chains = {
        "core_pick": check_tool_chain_compatibility(
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
        ).to_dict(),
        "placement": check_tool_chain_compatibility(
            catalog,
            ("anyplace", "camera_pose_to_world"),
            initial_facts=(PLACEMENT_INPUT_BUNDLE,),
        ).to_dict(),
        "wrist_refinement": check_tool_chain_compatibility(
            catalog,
            ("observe", "propose_wrist_viewpoints", "ik_preview_check", "move_to"),
            initial_facts=(COMPILED_GRASP,),
        ).to_dict(),
    }
    chains_conformant = all(row.get("compatible") is True for row in chains.values())

    authority = _read_object(generated / "tool-contract-authority-shadow-audit.json")
    authority_baseline = _read_object(
        generated / "tool-contract-authority-shadow-baseline.json"
    )
    policy = authority_baseline.get("policy")
    policy = policy if isinstance(policy, Mapping) else {}
    alignment = authority_baseline.get("planner_pipeline_alignment")
    alignment = alignment if isinstance(alignment, Mapping) else {}
    authority_safe = (
        authority.get("conformant") is True
        and authority.get("auditable_manifest_count") == 1
        and authority.get("violation_count") == 0
        and policy.get("request_validation_authority") == []
        and policy.get("gate_repair_envelope_authority") == []
        and policy.get("executable_gate_authority") == "legacy_runtime"
        and authority_baseline.get("catalog_sha256") == catalog_sha256
        and alignment.get("catalog_match") is True
        and alignment.get("pipeline_catalog_sha256") == catalog_sha256
        and alignment.get("policy_match") is True
    )

    review = _read_object(generated / "tool-contract-review-decision.json")
    review_decisions = review.get("decisions")
    review_decisions = review_decisions if isinstance(review_decisions, Mapping) else {}
    review_canary = review_decisions.get("first_authority_canary")
    review_canary = review_canary if isinstance(review_canary, Mapping) else {}
    review_approved = (
        review.get("review_status") == "approved"
        and review.get("review_scope") == "three_person_review"
        and review.get("shared_rfc_sync_authorized") is True
        and review_canary.get("status") == "approved"
        and review_canary.get("tool") == "estimate_depth_prior"
        and review_canary.get("request_validation_authority") is True
        and review_canary.get("gate_repair_envelope_authority") is False
        and review_canary.get("executable_gate_authority") == "legacy_runtime"
    )
    remaining_names = sorted(
        contract.name
        for contract in catalog.list()
        if contract.name != "estimate_depth_prior"
    )
    remaining_review = review_decisions.get("remaining_tool_promotions")
    remaining_review = (
        remaining_review if isinstance(remaining_review, Mapping) else {}
    )
    remaining_review_approved = bool(
        review_approved
        and remaining_review.get("status") == "approved"
        and sorted(str(name) for name in remaining_review.get("approved_tools", []))
        == remaining_names
        and remaining_review.get("request_validation_authority_canary") is True
        and remaining_review.get("gate_repair_envelope_authority") is False
        and remaining_review.get("executable_gate_authority") == "legacy_runtime"
    )
    authority_canary = _read_object(
        generated / "tool-contract-authority-canary.json"
    )
    canary_policy = authority_canary.get("policy")
    canary_policy = canary_policy if isinstance(canary_policy, Mapping) else {}
    canary_audit = authority_canary.get("authority_audit")
    canary_audit = canary_audit if isinstance(canary_audit, Mapping) else {}
    authority_canary_safe = (
        authority_canary.get("passed") is True
        and authority_canary.get("tool") == "estimate_depth_prior"
        and authority_canary.get("catalog_sha256") == catalog_sha256
        and canary_policy.get("request_validation_authority")
        == ["estimate_depth_prior"]
        and canary_policy.get("gate_repair_envelope_authority") == []
        and canary_policy.get("executable_gate_authority") == "legacy_runtime"
        and authority_canary.get("successful_execution_count") == 1
        and canary_audit.get("conformant") is True
        and canary_audit.get("violation_count") == 0
    )
    catalog_authority_canary = _read_object(
        generated / "tool-contract-catalog-authority-canary.json"
    )
    catalog_canary_policy = catalog_authority_canary.get("policy")
    catalog_canary_policy = (
        catalog_canary_policy
        if isinstance(catalog_canary_policy, Mapping)
        else {}
    )
    catalog_canary_audit = catalog_authority_canary.get("authority_audit")
    catalog_canary_audit = (
        catalog_canary_audit
        if isinstance(catalog_canary_audit, Mapping)
        else {}
    )
    catalog_canary_rows = catalog_authority_canary.get("tools")
    catalog_canary_rows = (
        catalog_canary_rows if isinstance(catalog_canary_rows, list) else []
    )
    remaining_authority_canary_safe = bool(
        catalog_authority_canary.get("schema_version")
        == "openeta.tool_contract_catalog_authority_canary.v1"
        and catalog_authority_canary.get("passed") is True
        and catalog_authority_canary.get("catalog_sha256") == catalog_sha256
        and catalog_authority_canary.get("tool_count") == 34
        and catalog_authority_canary.get("passed_tool_count") == 34
        and catalog_authority_canary.get("tool_execution_count") == 0
        and catalog_authority_canary.get("world_mutation_count") == 0
        and sorted(catalog_canary_policy.get("request_validation_authority", []))
        == remaining_names
        and catalog_canary_policy.get("gate_repair_envelope_authority") == []
        and catalog_canary_policy.get("executable_gate_authority")
        == "legacy_runtime"
        and sorted(
            str(row.get("tool") or "")
            for row in catalog_canary_rows
            if isinstance(row, Mapping) and row.get("passed") is True
        )
        == remaining_names
        and catalog_canary_audit.get("conformant") is True
        and catalog_canary_audit.get("violation_count") == 0
    )

    shared_rfc_sync = _read_object(
        generated / "tool-contract-shared-rfc-sync.json"
    )
    shared_rfc_synced = (
        shared_rfc_sync.get("status") == "synced"
        and shared_rfc_sync.get("review_decision")
        == "docs/generated/tool-contract-review-decision.json"
        and shared_rfc_sync.get("section")
        == "C.8.1 其余 34 个公共工具的 verified promotion"
        and isinstance(shared_rfc_sync.get("revision"), int)
        and int(shared_rfc_sync.get("revision", 0)) >= 2139
    )

    dossier = _read_object(
        generated / "estimate-depth-prior-promotion-dossier.json"
    )
    evidence_checks = dossier.get("evidence_checks")
    evidence_checks = evidence_checks if isinstance(evidence_checks, Mapping) else {}
    dossier_ready = (
        dossier.get("tool") == "estimate_depth_prior"
        and dossier.get("eligible_for_review") is True
        and dossier.get("eligible_for_verified_promotion") is True
        and dossier.get("promotion_complete") is True
        and dossier.get("current_maturity") == "verified"
        and len(evidence_checks) == 10
        and all(value is True for value in evidence_checks.values())
        and dossier.get("unresolved_requirements") == []
    )

    fixture_receipts = [
        _read_object(path)
        for path in sorted(generated.glob("*-fixture-receipt.json"))
    ]
    integration_canary = _read_object(
        generated / "tool-contract-integration-canary.json"
    )
    expected_campaign = build_tool_contract_promotion_campaign(
        catalog,
        fixture_receipts=fixture_receipts,
        integration_canary=integration_canary,
        review_decision=review,
        authority_canary=catalog_authority_canary,
    )
    campaign_projection = _projection_check(
        generated / "tool-contract-promotion-campaign.json",
        expected_campaign,
        exact_text=False,
    )
    campaign_rows = {
        str(row.get("tool") or ""): row
        for row in expected_campaign.get("tools", [])
        if isinstance(row, Mapping)
    }
    promotion_dossiers = [
        _read_object(path)
        for path in sorted((generated / "promotion-dossiers").glob("*.json"))
    ]
    dossier_by_tool = {
        str(row.get("tool") or ""): row for row in promotion_dossiers
    }
    remaining_promotions_complete = bool(
        expected_campaign.get("verified_count") == 35
        and expected_campaign.get("declared_count") == 0
        and all(
            campaign_rows.get(name, {}).get("eligible_for_verified_promotion")
            is True
            and campaign_rows.get(name, {}).get("promotion_gaps") == []
            and dossier_by_tool.get(name, {}).get("promotion_complete") is True
            and dossier_by_tool.get(name, {}).get("unresolved_requirements") == []
            for name in remaining_names
        )
        and sorted(dossier_by_tool) == remaining_names
    )
    review_request = _read_object(
        generated / "tool-contract-remaining-review-request.json"
    )
    review_request_fulfilled = bool(
        review_request.get("status") == "fulfilled_by_separate_decision_receipt"
        and review_request.get("decision_receipt")
        == "docs/generated/tool-contract-review-decision.json"
        and sorted(str(name) for name in review_request.get("tools", []))
        == remaining_names
    )

    generated_current = all(
        item.get("current") is True
        for item in (
            catalog_projection,
            markdown_projection,
            readiness_projection,
            resolver_projection,
            campaign_projection,
        )
    )
    local_checks = {
        "catalog_inventory": summary.get("tool_count") == len(registry.list()) == 35,
        "declared_contract_structure": not structural_issues and len(declared) == 35,
        "reviewed_catalog_fully_verified": (
            summary.get("maturity_counts")
            == {"inferred": 0, "declared": 0, "verified": 35}
            and summary.get("coverage_gap_counts") == {}
        ),
        "review_decision_approved": review_approved,
        "generated_projections_current": generated_current,
        "typed_toolchains_compatible": chains_conformant,
        "host_resolver_bindings_conformant": (
            resolver_audit.get("conformant") is True
            and resolver_audit.get("bound_count") == 30
            and resolver_audit.get("contract_driven_dispatch_count") == 11
            and resolver_audit.get("identity_only_count") == 19
            and resolver_audit.get("issue_count") == 0
        ),
        "request_readiness_audited": (
            readiness.get("declared_tool_count") == 35
            and readiness.get("ready_for_live_invalid_canary_count") == 35
        ),
        "empty_policy_authority_baseline_safe": authority_safe,
        "estimate_depth_prior_authority_canary_safe": authority_canary_safe,
        "estimate_depth_prior_promotion_complete": dossier_ready,
        "remaining_tool_review_approved": remaining_review_approved,
        "remaining_tool_authority_canary_safe": remaining_authority_canary_safe,
        "remaining_tool_promotions_complete": remaining_promotions_complete,
        "remaining_review_request_fulfilled": review_request_fulfilled,
        "shared_rfc_synced": shared_rfc_synced,
        "agent_choice_preserved": all(
            contract.gate.preserves_agent_choice for contract in declared
        ),
    }
    internal_conformant = all(
        value
        for name, value in local_checks.items()
        if name != "shared_rfc_synced"
    )
    goal_complete = internal_conformant and shared_rfc_synced

    requirements = [
        _requirement(
            "machine_readable_contract_catalog",
            "complete" if local_checks["catalog_inventory"] else "failed",
            ["docs/generated/tool-contracts.json"],
        ),
        _requirement(
            "generated_interface_documentation",
            "complete" if generated_current else "failed",
            ["docs/generated/tool-contracts.md"],
        ),
        _requirement(
            "typed_fact_toolchain_checks",
            "complete" if chains_conformant else "failed",
            ["core_pick", "placement", "wrist_refinement"],
        ),
        _requirement(
            "contract_shadow_validation_and_gate_repairs",
            "complete" if authority_safe else "failed",
            [
                "docs/generated/tool-contract-authority-shadow-audit.json",
                "docs/generated/tool-contract-authority-shadow-baseline.json",
            ],
        ),
        _requirement(
            "stable_host_resolution_contracts",
            "complete"
            if local_checks["host_resolver_bindings_conformant"]
            else "failed",
            ["docs/generated/host-resolver-binding-audit.json"],
            note=(
                "11 pipeline resolvers dispatch by stable contract id; 19 non-pipeline "
                "boundaries retain audited stable identities."
            ),
        ),
        _requirement(
            "agent_autonomy_without_task_stage_machine",
            "complete" if local_checks["agent_choice_preserved"] else "failed",
            [
                "all declared gate.preserves_agent_choice=true",
                "tests/test_no_host_task_state_machine.py",
            ],
            note=(
                "Typed chains describe possible producer-consumer compatibility only; "
                "they do not impose runtime order."
            ),
        ),
        _requirement(
            "promotion_evidence_package",
            "complete"
            if dossier_ready and remaining_promotions_complete
            else "failed",
            [
                "docs/generated/estimate-depth-prior-promotion-dossier.json",
                "docs/generated/tool-contract-promotion-campaign.json",
                "docs/generated/promotion-dossiers/<tool>.json",
            ],
        ),
        _requirement(
            "three_person_schema_and_promotion_review",
            "complete"
            if review_approved and remaining_review_approved
            else "failed",
            [
                "docs/rfc-tool-contract-v1-proposal.md",
                "docs/generated/tool-contract-review-decision.json",
            ],
        ),
        _requirement(
            "shared_rfc_sync",
            "complete" if shared_rfc_synced else "pending_sync",
            ["https://sii-czxy.feishu.cn/docx/HFB9d1ws8ow1CuxQeiacxMbonXe"],
            gaps=[] if shared_rfc_synced else ["approved decision has not been synced"],
        ),
        _requirement(
            "verified_authority_canary",
            "complete"
            if authority_canary_safe and remaining_authority_canary_safe
            else "failed",
            [
                "estimate_depth_prior",
                "docs/generated/tool-contract-authority-canary.json",
                "docs/generated/tool-contract-catalog-authority-canary.json",
            ],
        ),
    ]
    external_blockers = []
    return {
        "schema_version": SCHEMA_VERSION,
        "internal_conformant": internal_conformant,
        "implementation_ready_for_review": internal_conformant,
        "goal_complete": goal_complete,
        "catalog_summary": summary,
        "catalog_sha256": catalog_sha256,
        "local_checks": local_checks,
        "structural_issues": structural_issues,
        "generated_projections": {
            "catalog_json": catalog_projection,
            "markdown": markdown_projection,
            "readiness": readiness_projection,
            "host_resolvers": resolver_projection,
            "promotion_campaign": campaign_projection,
        },
        "typed_toolchains": chains,
        "requirements": requirements,
        "external_blockers": external_blockers,
        "next_action": (
            "Keep gate-repair-envelope authority empty and executable gates on "
            "legacy_runtime until a separately reviewed authority change is approved."
        ),
        "test_summary": {
            "passed": int(test_passed),
            "skipped": int(test_skipped),
            "warnings": int(test_warnings),
        },
        "harness_wiki": {
            "url": "https://sii-czxy.feishu.cn/wiki/A6M9wTBoCiOYPwkT9rWcwgfon2f",
            "revision": harness_revision,
        },
        "interpretation": (
            "The reviewed migration is complete only when local evidence, the durable "
            "review decision, the narrow enforcing canary, and shared RFC sync all pass. "
            "All 35 public tools are verified. Request-validation canaries are scoped "
            "separately; gate-repair and executable-gate authority remain unchanged."
        ),
    }


def _projection_check(path: Path, expected: Any, *, exact_text: bool) -> JsonDict:
    if not path.is_file():
        return {"path": str(path), "present": False, "current": False}
    observed_text = path.read_text(encoding="utf-8")
    if exact_text:
        current = observed_text == expected
    else:
        try:
            current = json.loads(observed_text) == expected
        except json.JSONDecodeError:
            current = False
    return {"path": str(path), "present": True, "current": current}


def _read_object(path: Path) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _issue(tool: str, code: str, detail: str = "") -> JsonDict:
    return {"tool": tool, "code": code, "detail": detail}


def _requirement(
    requirement_id: str,
    status: str,
    evidence: list[str],
    *,
    gaps: list[str] | None = None,
    note: str = "",
) -> JsonDict:
    result: JsonDict = {
        "id": requirement_id,
        "status": status,
        "evidence": evidence,
        "gaps": list(gaps or []),
    }
    if note:
        result["note"] = note
    return result
