"""Assemble an evidence-only per-tool ToolContract promotion dossier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adapter.protocol import JsonDict
from agent.evals.tool_contract_conformance import audit_tool_contract_conformance
from agent.evals.tool_contract_promotion_campaign import (
    _authority_canary_passed_for_tool,
    _review_approved_for_tool,
)
from agent.evals.tool_contract_readiness import audit_tool_contract_request_readiness
from agent.evals.tool_contract_shadow import audit_tool_contract_shadow
from agent.tools.contracts import build_default_tool_contract_catalog
from agent.tools.registry import build_default_tool_registry


SCHEMA_VERSION = "openeta.tool_contract_promotion_dossier.v1"


def build_tool_contract_promotion_dossier(
    tool_name: str,
    run_paths: list[str | Path],
    fixture_receipt_paths: list[str | Path] | None = None,
    *,
    integration_canary_path: str | Path | None = None,
    review_decision_path: str | Path | None = None,
    authority_canary_path: str | Path | None = None,
) -> JsonDict:
    catalog = build_default_tool_contract_catalog(build_default_tool_registry().list())
    contract = catalog.get(tool_name)
    readiness = audit_tool_contract_request_readiness(catalog)
    readiness_row = next(
        (row for row in readiness["tools"] if row["tool"] == tool_name),
        {},
    )
    request_records: list[JsonDict] = []
    result_reports: list[JsonDict] = []
    run_records: list[JsonDict] = []
    for raw_path in run_paths:
        path = Path(raw_path)
        shadow = audit_tool_contract_shadow(path)
        selected_requests = [
            record for record in shadow["records"] if record["tool"] == tool_name
        ]
        request_records.extend(selected_requests)
        try:
            conformance = audit_tool_contract_conformance(path)
        except FileNotFoundError:
            conformance = {
                "tool_event_count": 0,
                "declared_event_count": 0,
                "conformant": True,
                "violation_count": 0,
                "tool_counts": {},
                "semantic_counts": {},
                "violations": [],
            }
        tool_result_count = int(conformance.get("tool_counts", {}).get(tool_name, 0))
        selected_violations = [
            violation
            for violation in conformance.get("violations", [])
            if violation.get("tool") == tool_name
        ]
        result_reports.append(
            {
                "run": str(path),
                "tool_result_count": tool_result_count,
                "semantic_counts": {
                    key: value
                    for key, value in conformance.get("semantic_counts", {}).items()
                    if str(key).startswith(tool_name + ":")
                },
                "violation_count": len(selected_violations),
                "violations": selected_violations,
            }
        )
        run_records.append(
            {
                "run": str(path),
                "request_record_count": len(selected_requests),
                "request_acceptance_mismatch_count": sum(
                    record.get("acceptance_match") is not True
                    for record in selected_requests
                ),
                "tool_result_count": tool_result_count,
                "tool_result_violation_count": len(selected_violations),
            }
        )
    invalid_requests = [
        record for record in request_records if record.get("contract_accepted") is False
    ]
    valid_requests = [
        record for record in request_records if record.get("contract_accepted") is True
    ]
    enforcing_requests = [
        record for record in request_records if record.get("enforcing") is True
    ]
    result_count = sum(row["tool_result_count"] for row in result_reports)
    result_violation_count = sum(row["violation_count"] for row in result_reports)
    integration_canary = _read_optional_object(integration_canary_path)
    canary_rows = integration_canary.get("tools")
    canary_rows = canary_rows if isinstance(canary_rows, list) else []
    canary_row = next(
        (
            row
            for row in canary_rows
            if isinstance(row, dict) and row.get("tool") == tool_name
        ),
        {},
    )
    canary_schema_valid = bool(
        integration_canary.get("schema_version")
        == "openeta.tool_contract_integration_canary.v1"
        and integration_canary.get("authority") == "promotion_evidence_only"
        and canary_row.get("conformant") is True
    )
    canary_valid_count = (
        int(canary_row.get("valid_request_count") or 0)
        if canary_schema_valid
        else 0
    )
    canary_invalid_count = (
        int(canary_row.get("invalid_request_count") or 0)
        if canary_schema_valid
        else 0
    )
    canary_result_count = (
        int(canary_row.get("tool_result_count") or 0)
        if canary_schema_valid
        else 0
    )
    canary_result_violation_count = (
        int(canary_row.get("tool_result_violation_count") or 0)
        if canary_schema_valid
        else 0
    )
    combined_valid_count = len(valid_requests) + canary_valid_count
    combined_invalid_count = len(invalid_requests) + canary_invalid_count
    combined_result_count = result_count + canary_result_count
    combined_result_violation_count = (
        result_violation_count + canary_result_violation_count
    )
    request_sources_present = bool(request_records) or canary_schema_valid
    request_parity = bool(
        request_sources_present
        and all(record.get("acceptance_match") is True for record in request_records)
        and (
            not canary_schema_valid
            or canary_row.get("request_acceptance_parity") is True
        )
    )
    shadow_only = bool(
        not enforcing_requests
        and (
            not canary_schema_valid
            or canary_row.get("observational_only") is True
        )
    )
    fixture_receipts = [
        _read_fixture_receipt(path, tool_name=tool_name)
        for path in (fixture_receipt_paths or [])
    ]
    conformant_fixture_receipts = [
        receipt
        for receipt in fixture_receipts
        if receipt.get("conformant") is True
    ]
    handler_fixture_coverage = any(
        isinstance(receipt.get("coverage"), dict)
        and receipt["coverage"].get("all_success_outcomes_covered") is True
        and receipt["coverage"].get("all_declared_outcomes_covered") is True
        and receipt["coverage"].get("representative_failure_covered") is True
        for receipt in conformant_fixture_receipts
    )
    gate_binding_coverage = any(
        isinstance(receipt.get("coverage"), dict)
        and receipt["coverage"].get("runtime_gate_evidence_observed") is True
        and bool(receipt["coverage"].get("matched_gate_check_ids"))
        for receipt in conformant_fixture_receipts
    )
    evidence_checks = {
        "declared_or_verified_contract": contract.maturity.value
        in {"declared", "verified"},
        "deterministic_valid_invalid_parity": bool(
            readiness_row.get("ready_for_live_invalid_canary")
        ),
        "live_valid_request_seen": combined_valid_count > 0,
        "live_invalid_request_seen": combined_invalid_count > 0,
        "live_request_acceptance_parity": request_parity,
        "live_tool_result_seen": combined_result_count > 0,
        "live_tool_result_conformant": combined_result_count > 0
        and combined_result_violation_count == 0,
        "evidence_collected_in_shadow": shadow_only,
        "durable_handler_fixture_receipt_conformant": handler_fixture_coverage,
        "runtime_gate_rejection_bound_to_check_id": gate_binding_coverage,
    }
    unresolved = []
    if not handler_fixture_coverage:
        unresolved.append(
            "a durable production-handler fixture receipt must cover every successful semantic outcome and a representative failure"
        )
    if not gate_binding_coverage:
        unresolved.append(
            "observed runtime gate evidence must resolve to a machine-bound check id"
        )
    review_decision = _read_optional_object(review_decision_path)
    review_approved = _review_approved_for_tool(tool_name, review_decision)
    authority_canary = _read_optional_object(authority_canary_path)
    canary_approved = _authority_canary_passed_for_tool(
        tool_name,
        authority_canary,
    )
    if not review_approved:
        unresolved.append(
            "three-person review must approve the shared schema and per-tool maturity promotion"
        )
    if not canary_approved:
        unresolved.append("a separately scoped authority canary must pass after review")
    evidence_ready = all(evidence_checks.values())
    promotion_ready = evidence_ready and review_approved and canary_approved
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": tool_name,
        "current_maturity": contract.maturity.value,
        "promotion_authority": (
            "three_person_review_approved"
            if review_approved
            else "three_person_review_required"
        ),
        "eligible_for_review": evidence_ready,
        "eligible_for_verified_promotion": promotion_ready,
        "promotion_complete": (
            promotion_ready and contract.maturity.value == "verified"
        ),
        "contract_schema_version": contract.schema_version,
        "coverage_gaps": list(contract.coverage_gaps),
        "evidence_checks": evidence_checks,
        "request_evidence": {
            "record_count": len(request_records),
            "valid_count": combined_valid_count,
            "invalid_count": combined_invalid_count,
            "enforcing_count": len(enforcing_requests),
            "records": request_records,
        },
        "tool_result_evidence": {
            "event_count": combined_result_count,
            "violation_count": combined_result_violation_count,
            "runs": result_reports,
        },
        "integration_canary_evidence": {
            "present": bool(canary_row),
            "conformant": canary_schema_valid,
            "source": str(integration_canary_path or ""),
            "row": canary_row,
        },
        "fixture_evidence": {
            "receipt_count": len(fixture_receipts),
            "conformant_receipt_count": len(conformant_fixture_receipts),
            "receipts": fixture_receipts,
        },
        "review_evidence": {
            "approved": review_approved,
            "source": str(review_decision_path or ""),
        },
        "authority_canary_evidence": {
            "passed": canary_approved,
            "source": str(authority_canary_path or ""),
        },
        "runs": run_records,
        "unresolved_requirements": unresolved,
        "interpretation": (
            "This dossier aggregates evidence and never changes maturity or runtime "
            "authority. verified remains a reviewed declaration, not an audit side effect."
        ),
    }


def _read_fixture_receipt(
    path: str | Path,
    *,
    tool_name: str,
) -> JsonDict:
    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"fixture receipt must be an object: {source}")
    if payload.get("schema_version") != "openeta.tool_contract_fixture_receipt.v1":
        raise ValueError(f"unsupported fixture receipt schema: {source}")
    if payload.get("tool") != tool_name:
        raise ValueError(
            f"fixture receipt tool mismatch: expected {tool_name}, "
            f"observed {payload.get('tool')}: {source}"
        )
    return {"source": str(source), **payload}


def _read_optional_object(path: str | Path | None) -> JsonDict:
    if path is None:
        return {}
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tool", required=True)
    parser.add_argument("--run", action="append", default=[])
    parser.add_argument("--fixture-receipt", action="append", default=[])
    parser.add_argument("--integration-canary", default="")
    parser.add_argument("--review-decision", default="")
    parser.add_argument("--authority-canary", default="")
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)
    report = build_tool_contract_promotion_dossier(
        args.tool,
        args.run,
        args.fixture_receipt,
        integration_canary_path=args.integration_canary or None,
        review_decision_path=args.review_decision or None,
        authority_canary_path=args.authority_canary or None,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0 if report["eligible_for_review"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
