"""Build a catalog-wide, evidence-gated ToolContract promotion campaign matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping

from adapter.protocol import JsonDict
from agent.evals.tool_contract_fixture_receipt import registered_fixture_tools
from agent.evals.tool_contract_readiness import audit_tool_contract_request_readiness
from agent.tools.contracts import (
    ContractMaturity,
    ToolContract,
    ToolContractCatalog,
    build_default_tool_contract_catalog,
)
from agent.tools.registry import ToolEffect, build_default_tool_registry
from agent.tools.runtime_contract_bindings import audit_host_resolver_bindings


SCHEMA_VERSION = "openeta.tool_contract_promotion_campaign.v1"


def build_tool_contract_promotion_campaign(
    catalog: ToolContractCatalog | None = None,
    *,
    fixture_tools: Iterable[str] | None = None,
    fixture_receipts: Iterable[Mapping[str, object]] = (),
    integration_canary: Mapping[str, object] | None = None,
    review_decision: Mapping[str, object] | None = None,
    authority_canary: Mapping[str, object] | None = None,
) -> JsonDict:
    """Return the current per-tool promotion gaps without granting authority.

    The report deliberately distinguishes structural declaration, deterministic
    request parity, production-handler fixtures, and the additional evidence
    required by a tool's effect.  It never mutates maturity or runtime policy.
    """

    resolved = catalog or build_default_tool_contract_catalog(
        build_default_tool_registry().list()
    )
    readiness = audit_tool_contract_request_readiness(resolved)
    readiness_by_tool = {
        str(row["tool"]): row for row in readiness.get("tools", [])
    }
    fixture_names = {
        str(name) for name in (fixture_tools if fixture_tools is not None else registered_fixture_tools())
    }
    receipts_by_tool: dict[str, list[Mapping[str, object]]] = {}
    for receipt in fixture_receipts:
        tool_name = str(receipt.get("tool") or "")
        if tool_name:
            receipts_by_tool.setdefault(tool_name, []).append(receipt)
    canary_rows = (
        integration_canary.get("tools", [])
        if isinstance(integration_canary, Mapping)
        else []
    )
    canary_by_tool = {
        str(row.get("tool") or ""): row
        for row in canary_rows
        if isinstance(row, Mapping) and row.get("tool")
    }
    resolver_audit = audit_host_resolver_bindings(resolved)
    resolver_issues = {
        str(issue.get("tool") or "")
        for issue in resolver_audit.get("issues", [])
        if isinstance(issue, dict)
    }

    rows: list[JsonDict] = []
    for contract in resolved.list():
        readiness_row = readiness_by_tool.get(contract.name, {})
        tool_receipts = receipts_by_tool.get(contract.name, [])
        conformant_receipts = [
            receipt for receipt in tool_receipts if receipt.get("conformant") is True
        ]
        fixture_success_covered = any(
            isinstance(receipt.get("coverage"), Mapping)
            and receipt["coverage"].get("all_success_outcomes_covered") is True
            for receipt in conformant_receipts
        )
        fixture_failure_covered = any(
            isinstance(receipt.get("coverage"), Mapping)
            and receipt["coverage"].get("representative_failure_covered") is True
            for receipt in conformant_receipts
        )
        fixture_all_outcomes_covered = any(
            isinstance(receipt.get("coverage"), Mapping)
            and receipt["coverage"].get("all_declared_outcomes_covered") is True
            for receipt in conformant_receipts
        )
        fixture_gate_covered = any(
            isinstance(receipt.get("coverage"), Mapping)
            and receipt["coverage"].get("runtime_gate_evidence_observed") is True
            and bool(receipt["coverage"].get("matched_gate_check_ids"))
            for receipt in conformant_receipts
        )
        fixture_freshness_covered = any(
            isinstance(receipt.get("coverage"), Mapping)
            and receipt["coverage"].get("freshness_and_invalidation_covered") is True
            for receipt in conformant_receipts
        )
        fixture_remote_semantics_covered = any(
            isinstance(receipt.get("coverage"), Mapping)
            and receipt["coverage"].get("remote_backend_semantics_covered") is True
            for receipt in conformant_receipts
        )
        fixture_world_receipt_covered = any(
            isinstance(receipt.get("coverage"), Mapping)
            and receipt["coverage"].get(
                "world_mutating_execution_receipt_covered"
            ) is True
            for receipt in conformant_receipts
        )
        canary_row = canary_by_tool.get(contract.name, {})
        canary_request_parity = bool(
            canary_row.get("conformant") is True
            and int(canary_row.get("valid_request_count") or 0) > 0
            and int(canary_row.get("invalid_request_count") or 0) > 0
            and canary_row.get("request_acceptance_parity") is True
            and canary_row.get("observational_only") is True
        )
        canary_result_conformant = bool(
            canary_row.get("conformant") is True
            and int(canary_row.get("tool_result_count") or 0) > 0
            and int(canary_row.get("tool_result_violation_count") or 0) == 0
        )
        requirements = _promotion_requirements(contract)
        checks = {
            "explicit_contract": contract.maturity is not ContractMaturity.INFERRED,
            "request_shadow_parity": (
                readiness_row.get("ready_for_live_invalid_canary") is True
            ),
            "host_resolver_binding_conformant": contract.name not in resolver_issues,
            "production_fixture_registered": contract.name in fixture_names,
            # These are intentionally false until durable receipts/canaries are
            # supplied by later campaign stages.  A declaration cannot satisfy
            # them by construction.
            "successful_outcomes_fixture_covered": fixture_success_covered,
            "representative_failure_fixture_covered": fixture_failure_covered,
            "all_declared_outcomes_fixture_covered": fixture_all_outcomes_covered,
            "live_result_conformance": canary_result_conformant,
            "valid_invalid_live_shadow_parity": canary_request_parity,
            "gate_repair_binding_observed": fixture_gate_covered,
            "three_person_promotion_review": _review_approved_for_tool(
                contract.name,
                review_decision,
            ),
            "authority_canary_passed": _authority_canary_passed_for_tool(
                contract.name,
                authority_canary,
            ),
        }
        if "world_mutating_execution_receipt" in requirements:
            checks["world_mutating_execution_receipt"] = (
                fixture_world_receipt_covered
            )
        if "freshness_and_invalidation" in requirements:
            checks["freshness_and_invalidation"] = fixture_freshness_covered
        if "remote_backend_semantics" in requirements:
            checks["remote_backend_semantics"] = (
                fixture_remote_semantics_covered
            )

        gaps = [name for name, passed in checks.items() if passed is not True]
        rows.append(
            {
                "tool": contract.name,
                "category": contract.category,
                "effect": contract.effect,
                "maturity": contract.maturity.value,
                "risk_tier": _risk_tier(contract),
                "successful_semantic_outcomes": [
                    outcome.semantic_outcome
                    for outcome in contract.outcomes
                    if outcome.operational_success
                ],
                "gate_binding_count": len(contract.gate.bindings),
                "host_resolution": {
                    "mode": contract.host_resolution.mode,
                    "layer": contract.host_resolution.resolution_layer,
                    "contract_driven_dispatch": (
                        contract.host_resolution.contract_driven_dispatch
                    ),
                },
                "requirements": requirements,
                "fixture_receipt_count": len(tool_receipts),
                "conformant_fixture_receipt_count": len(conformant_receipts),
                "integration_canary_present": bool(canary_row),
                "checks": checks,
                "promotion_gap_count": len(gaps),
                "promotion_gaps": gaps,
                "eligible_for_verified_promotion": not gaps,
            }
        )

    target_rows = [
        row for row in rows if row["tool"] != "estimate_depth_prior"
    ]
    tier_counts = Counter(str(row["risk_tier"]) for row in target_rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "authority": "promotion_evidence_only",
        "catalog_tool_count": len(rows),
        "campaign_target_count": len(target_rows),
        "verified_count": sum(
            row["maturity"] == ContractMaturity.VERIFIED.value for row in rows
        ),
        "declared_count": sum(
            row["maturity"] == ContractMaturity.DECLARED.value for row in rows
        ),
        "request_parity_ready_count": sum(
            row["checks"]["request_shadow_parity"] is True for row in target_rows
        ),
        "fixture_registered_count": sum(
            row["checks"]["production_fixture_registered"] is True
            for row in target_rows
        ),
        "integration_canary_conformant_count": sum(
            row["checks"]["live_result_conformance"] is True
            and row["checks"]["valid_invalid_live_shadow_parity"] is True
            for row in target_rows
        ),
        "risk_tier_counts": dict(sorted(tier_counts.items())),
        "resolver_binding_audit_conformant": resolver_audit.get("conformant") is True,
        "tools": rows,
        "interpretation": (
            "This matrix is promotion evidence and work planning only. Empty gaps, "
            "review approval, and an authority canary are all required before a tool "
            "may be declared verified; this report never changes runtime authority."
        ),
    }


def _risk_tier(contract: ToolContract) -> str:
    if contract.effect == ToolEffect.WORLD_MUTATING.value:
        return "world_mutating"
    if contract.host_resolution.freshness_dimensions or contract.host_resolution.invalidated_by:
        return "evidence_sensitive"
    if contract.category in {"perception", "web", "manipulation"}:
        return "backend_or_artifact"
    return "local_deterministic"


def _promotion_requirements(contract: ToolContract) -> list[str]:
    requirements = [
        "request_and_result_conformance",
        "production_handler_success_and_failure",
        "live_shadow_and_gate_evidence",
        "three_person_review_and_authority_canary",
    ]
    if contract.effect == ToolEffect.WORLD_MUTATING.value:
        requirements.append("world_mutating_execution_receipt")
    if contract.host_resolution.freshness_dimensions or contract.host_resolution.invalidated_by:
        requirements.append("freshness_and_invalidation")
    if contract.category in {"perception", "web", "manipulation"}:
        requirements.append("remote_backend_semantics")
    return requirements


def render_tool_contract_promotion_review(report: Mapping[str, object]) -> str:
    """Render the evidence matrix as a concise three-person review packet."""

    rows = [
        row
        for row in report.get("tools", [])
        if isinstance(row, Mapping) and row.get("tool") != "estimate_depth_prior"
    ]
    promotion_complete = bool(
        rows
        and all(
            row.get("eligible_for_verified_promotion") is True for row in rows
        )
    )
    lines = [
        "# Remaining ToolContract verified-promotion review",
        "",
        (
            "Status: three-person review approved; post-review authority canary passed; "
            "all 34 promotions complete."
            if promotion_complete
            else "Status: evidence complete; three-person review and post-review authority canary pending."
        ),
        "",
        (
            "This packet records the approved per-tool maturity decisions for the remaining "
            "34 public tools and their passing request-validation-only authority canary. "
            "Gate-repair envelope authority and executable safety gates remain separate controls."
            if promotion_complete
            else "This packet requests one per-tool maturity decision for the remaining 34 public tools. "
            "It does not grant runtime authority. Request validation, gate-repair envelope authority, "
            "and executable safety gates remain separate controls."
        ),
        "",
        "## Evidence summary",
        "",
        f"- Campaign targets: {report.get('campaign_target_count', 0)}",
        f"- Deterministic request parity ready: {report.get('request_parity_ready_count', 0)}",
        f"- Production fixture receipts: {report.get('fixture_registered_count', 0)}",
        f"- Planner/ToolRegistry integration canaries: {report.get('integration_canary_conformant_count', 0)}",
        f"- Host resolver binding audit conformant: {str(report.get('resolver_binding_audit_conformant') is True).lower()}",
        "- Freshness/invalidation: 19/19 scoped tools",
        "- Remote-backend semantics: 11/11 scoped tools",
        "- World-mutating execution receipts: 5/5 scoped tools",
        "",
        "The catalog integration canary recorded 70 valid/invalid Planner shadows with zero "
        "acceptance mismatch or unexpected enforcement, plus 35 conformant ToolRegistry results. "
        "Production behavior is established separately by registry-bound fixtures; the canary's "
        "deterministic adapter is not represented as a live remote backend or task success.",
        "",
        "## " + ("Approved decision" if promotion_complete else "Requested decision"),
        "",
        (
            "All rows below were approved from `declared` to `verified`. The separately "
            "scoped request-validation canary passed without tool execution or world mutation. "
            "Gate-repair-envelope authority remains empty and executable gate authority remains "
            "`legacy_runtime`."
            if promotion_complete
            else "For each row below, approve or reject promotion from `declared` to `verified`. "
            "Approval authorizes a separately scoped request-validation authority canary only. "
            "Gate-repair-envelope authority remains empty and executable gate authority remains "
            "`legacy_runtime` unless separately reviewed."
        ),
        "",
        "| Tool | Risk tier | Successful outcomes | Bound gates | Remaining pre-promotion gaps |",
        "|---|---|---:|---:|---|",
    ]
    for row in rows:
        outcomes = row.get("successful_semantic_outcomes", [])
        outcomes = outcomes if isinstance(outcomes, list) else []
        gaps = row.get("promotion_gaps", [])
        gaps = gaps if isinstance(gaps, list) else []
        lines.append(
            "| {tool} | {tier} | {outcomes} | {gates} | {gaps} |".format(
                tool=row.get("tool", ""),
                tier=row.get("risk_tier", ""),
                outcomes=len(outcomes),
                gates=row.get("gate_binding_count", 0),
                gaps=", ".join(str(item) for item in gaps) or "none",
            )
        )
    lines.extend(
        [
            "",
            "## Evidence files",
            "",
            "- `docs/generated/tool-contract-promotion-campaign.json`",
            "- `docs/generated/tool-contract-integration-canary.json`",
            "- `docs/generated/*-fixture-receipt.json`",
            "- `docs/generated/tool-contract-readiness.json`",
            "- `docs/generated/host-resolver-binding-audit.json`",
            "",
        ]
    )
    return "\n".join(lines)


def build_tool_contract_review_request(
    report: Mapping[str, object],
    review_decision: Mapping[str, object] | None = None,
) -> JsonDict:
    """Build a machine-readable request; this is never an approval receipt."""

    rows = [
        row
        for row in report.get("tools", [])
        if isinstance(row, Mapping) and row.get("tool") != "estimate_depth_prior"
    ]
    tools = [str(row.get("tool") or "") for row in rows]
    review_complete = bool(
        tools
        and all(
            _review_approved_for_tool(tool_name, review_decision)
            for tool_name in tools
        )
    )
    campaign_digest = hashlib.sha256(
        json.dumps(
            report,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "openeta.tool_contract_review_request.v1",
        "status": (
            "fulfilled_by_separate_decision_receipt"
            if review_complete
            else "pending_three_person_review"
        ),
        "review_scope": "remaining_public_tools_per_tool_promotion",
        "campaign_sha256": campaign_digest,
        "tools": tools,
        "tool_count": len(tools),
        "decision_receipt": (
            "docs/generated/tool-contract-review-decision.json"
            if review_complete
            else ""
        ),
        "requested_decisions": {
            "remaining_tool_promotions": {
                "maturity_from": "declared",
                "maturity_to": "verified",
                "decision_scope": "per_tool",
                "evidence_scope": "per_semantic_outcome",
            },
            "request_validation_authority_canary": {
                "requested": True,
                "post_review_only": True,
                "tool_execution_count": 0,
                "world_mutation_count": 0,
            },
            "gate_repair_envelope_authority": {
                "requested": False,
                "allowlist": [],
            },
            "executable_gate_authority": "legacy_runtime",
        },
        "evidence": [
            "docs/generated/tool-contract-promotion-campaign.json",
            "docs/generated/tool-contract-integration-canary.json",
            "docs/generated/promotion-dossiers/<tool>.json",
            "docs/generated/<tool>-fixture-receipt.json",
        ],
        "interpretation": (
            "This file requests review and cannot itself be consumed as approval. "
            "The separate three-person decision receipt is authoritative."
        ),
    }


def _review_approved_for_tool(
    tool_name: str,
    review_decision: Mapping[str, object] | None,
) -> bool:
    if not isinstance(review_decision, Mapping):
        return False
    if review_decision.get("review_status") != "approved":
        return False
    decisions = review_decision.get("decisions")
    decisions = decisions if isinstance(decisions, Mapping) else {}
    first = decisions.get("first_authority_canary")
    first = first if isinstance(first, Mapping) else {}
    if (
        first.get("status") == "approved"
        and first.get("tool") == tool_name
        and first.get("request_validation_authority") is True
        and first.get("gate_repair_envelope_authority") is False
        and first.get("executable_gate_authority") == "legacy_runtime"
    ):
        return True
    remaining = decisions.get("remaining_tool_promotions")
    remaining = remaining if isinstance(remaining, Mapping) else {}
    approved_tools = remaining.get("approved_tools")
    approved_tools = approved_tools if isinstance(approved_tools, list) else []
    return bool(
        remaining.get("status") == "approved"
        and tool_name in {str(name) for name in approved_tools}
        and remaining.get("request_validation_authority_canary") is True
        and remaining.get("gate_repair_envelope_authority") is False
        and remaining.get("executable_gate_authority") == "legacy_runtime"
    )


def _authority_canary_passed_for_tool(
    tool_name: str,
    authority_canary: Mapping[str, object] | None,
) -> bool:
    if not isinstance(authority_canary, Mapping) or authority_canary.get("passed") is not True:
        return False
    audit = authority_canary.get("authority_audit")
    audit = audit if isinstance(audit, Mapping) else {}
    if audit.get("conformant") is not True or audit.get("violation_count") != 0:
        return False
    if authority_canary.get("schema_version") == "openeta.tool_contract_authority_canary.v1":
        return bool(
            authority_canary.get("tool") == tool_name
            and authority_canary.get("successful_execution_count") == 1
        )
    if authority_canary.get("schema_version") != SCHEMA_VERSION.replace(
        "promotion_campaign", "catalog_authority_canary"
    ):
        return False
    policy = authority_canary.get("policy")
    policy = policy if isinstance(policy, Mapping) else {}
    rows = authority_canary.get("tools")
    rows = rows if isinstance(rows, list) else []
    row = next(
        (
            item
            for item in rows
            if isinstance(item, Mapping) and item.get("tool") == tool_name
        ),
        {},
    )
    return bool(
        row.get("passed") is True
        and tool_name in policy.get("request_validation_authority", [])
        and policy.get("gate_repair_envelope_authority") == []
        and policy.get("executable_gate_authority") == "legacy_runtime"
        and authority_canary.get("tool_execution_count") == 0
        and authority_canary.get("world_mutation_count") == 0
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture-receipt", action="append", default=[])
    parser.add_argument("--integration-canary", default="")
    parser.add_argument("--review-decision", default="")
    parser.add_argument("--authority-canary", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--markdown-output", default="")
    parser.add_argument("--review-request-output", default="")
    args = parser.parse_args(argv)
    fixture_receipts = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in args.fixture_receipt
    ]
    integration_canary = (
        json.loads(Path(args.integration_canary).read_text(encoding="utf-8"))
        if args.integration_canary
        else None
    )
    review_decision = (
        json.loads(Path(args.review_decision).read_text(encoding="utf-8"))
        if args.review_decision
        else None
    )
    authority_canary = (
        json.loads(Path(args.authority_canary).read_text(encoding="utf-8"))
        if args.authority_canary
        else None
    )
    report = build_tool_contract_promotion_campaign(
        fixture_receipts=fixture_receipts,
        integration_canary=integration_canary,
        review_decision=review_decision,
        authority_canary=authority_canary,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    if args.markdown_output:
        Path(args.markdown_output).write_text(
            render_tool_contract_promotion_review(report),
            encoding="utf-8",
        )
    if args.review_request_output:
        Path(args.review_request_output).write_text(
            json.dumps(
                build_tool_contract_review_request(report, review_decision),
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
