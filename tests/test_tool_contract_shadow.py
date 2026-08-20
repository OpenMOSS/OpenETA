from __future__ import annotations

import json

from agent.evals.tool_contract_shadow import audit_tool_contract_shadow


def _write_rows(path, rows) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _row(seq: int, tool: str, shadow: dict | None) -> dict:
    metadata = {"tool_contract_shadow_validation": shadow} if shadow else {}
    return {
        "seq": seq,
        "parsed_decision": {
            "kind": "tool_call",
            "name": tool,
            "parameters": {},
            "metadata": metadata,
        },
    }


def test_shadow_audit_counts_acceptance_parity_and_inventory_records(tmp_path) -> None:
    path = tmp_path / "model_calls.jsonl"
    _write_rows(
        path,
        [
            _row(
                1,
                "sam3",
                {
                    "schema_version": "openeta.tool_contract_shadow_validation.v1",
                    "tool": "sam3",
                    "contract_maturity": "declared",
                    "evaluated": True,
                    "enforcing": False,
                    "legacy_accepted": True,
                    "contract_accepted": True,
                    "acceptance_match": True,
                    "legacy_errors": [],
                    "contract_violations": [],
                },
            ),
            _row(
                2,
                "scene_detector",
                {
                    "schema_version": "openeta.tool_contract_shadow_validation.v1",
                    "tool": "scene_detector",
                    "contract_maturity": "inferred",
                    "evaluated": False,
                    "enforcing": False,
                },
            ),
            _row(3, "talk", None),
        ],
    )

    report = audit_tool_contract_shadow(path)

    assert report["model_call_count"] == 3
    assert report["shadow_record_count"] == 2
    assert report["evaluated_record_count"] == 1
    assert report["inventory_only_record_count"] == 1
    assert report["parity"] is True
    assert report["observational_only"] is True


def test_shadow_audit_exposes_acceptance_mismatch_and_enforcement(tmp_path) -> None:
    path = tmp_path / "model_calls.jsonl"
    _write_rows(
        path,
        [
            _row(
                1,
                "move_to",
                {
                    "schema_version": "openeta.tool_contract_shadow_validation.v1",
                    "tool": "move_to",
                    "contract_maturity": "declared",
                    "evaluated": True,
                    "enforcing": True,
                    "legacy_accepted": True,
                    "contract_accepted": False,
                    "acceptance_match": False,
                    "legacy_errors": [],
                    "contract_violations": [{"code": "request_required_field_missing"}],
                },
            )
        ],
    )

    report = audit_tool_contract_shadow(path)

    assert report["acceptance_mismatch_count"] == 1
    assert report["unexpected_enforcement_count"] == 1
    assert report["parity"] is False
    assert report["observational_only"] is False
    assert report["mismatch_tool_counts"] == {"move_to": 1}
