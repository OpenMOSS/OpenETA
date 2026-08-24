"""Build deterministic valid/invalid request parity evidence for ToolContract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from adapter.protocol import JsonDict
from agent.runtime.planner import _validate_tool_parameters
from agent.tools.contracts import (
    ContractMaturity,
    ToolContract,
    ToolContractCatalog,
    build_default_tool_contract_catalog,
    check_tool_request_conformance,
)
from agent.tools.registry import build_default_tool_registry


SCHEMA_VERSION = "openeta.tool_contract_readiness.v1"


def audit_tool_contract_request_readiness(
    catalog: ToolContractCatalog | None = None,
) -> JsonDict:
    """Compare legacy and contract acceptance across deterministic input classes.

    This report is promotion evidence, not promotion authority. In particular it
    cannot change ``maturity`` to verified or bypass live rollout and review.
    """

    resolved = catalog or build_default_tool_contract_catalog(
        build_default_tool_registry().list()
    )
    records: list[JsonDict] = []
    tool_reports: list[JsonDict] = []
    for contract in resolved.list():
        if contract.maturity is ContractMaturity.INFERRED:
            continue
        tool_records: list[JsonDict] = []
        for case_name, parameters in _request_cases(contract):
            legacy_errors = _validate_tool_parameters(contract.name, parameters)
            contract_violations = check_tool_request_conformance(contract, parameters)
            record: JsonDict = {
                "tool": contract.name,
                "case": case_name,
                "parameters": parameters,
                "legacy_accepted": not legacy_errors,
                "contract_accepted": not contract_violations,
                "acceptance_match": bool(legacy_errors) == bool(contract_violations),
                "legacy_errors": list(legacy_errors),
                "contract_violations": [item.to_dict() for item in contract_violations],
            }
            tool_records.append(record)
            records.append(record)
        valid = next(
            (item for item in tool_records if item["case"] == "valid_minimal"),
            None,
        )
        mismatches = [item for item in tool_records if not item["acceptance_match"]]
        invalid_records = [
            item for item in tool_records if item["case"] != "valid_minimal"
        ]
        tool_reports.append(
            {
                "tool": contract.name,
                "maturity": contract.maturity.value,
                "case_count": len(tool_records),
                "invalid_class_count": len(invalid_records),
                "valid_fixture_accepted_by_both": bool(
                    valid
                    and valid["legacy_accepted"]
                    and valid["contract_accepted"]
                ),
                "acceptance_mismatch_count": len(mismatches),
                "deterministic_request_parity": not mismatches,
                "ready_for_live_invalid_canary": bool(
                    valid
                    and valid["legacy_accepted"]
                    and valid["contract_accepted"]
                    and invalid_records
                    and not mismatches
                ),
                "mismatch_cases": [item["case"] for item in mismatches],
            }
        )
    ready = [
        item["tool"]
        for item in tool_reports
        if item["ready_for_live_invalid_canary"]
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "authority": "promotion_evidence_only",
        "declared_tool_count": len(tool_reports),
        "ready_for_live_invalid_canary_count": len(ready),
        "ready_for_live_invalid_canary": ready,
        "tools": tool_reports,
        "records": records,
    }


def _request_cases(contract: ToolContract) -> list[tuple[str, JsonDict]]:
    schema = contract.request_schema
    valid = _schema_fixture(schema)
    valid = dict(valid) if isinstance(valid, Mapping) else {}
    cases: list[tuple[str, JsonDict]] = [("valid_minimal", valid)]
    for name in schema.get("required", []):
        if isinstance(name, str) and name in valid:
            cases.append((f"missing_required:{name}", _without(valid, name)))
    properties = schema.get("properties")
    properties = properties if isinstance(properties, Mapping) else {}
    for name, property_schema in properties.items():
        if not isinstance(name, str) or not isinstance(property_schema, Mapping):
            continue
        if name not in valid:
            continue
        wrong = _wrong_type_fixture(property_schema)
        if wrong is not None:
            cases.append((f"wrong_type:{name}", {**valid, name: wrong}))
        enum = property_schema.get("enum")
        if isinstance(enum, list) and enum:
            cases.append((f"invalid_enum:{name}", {**valid, name: "__invalid_enum__"}))
    if schema.get("additionalProperties") is False:
        cases.append(("unexpected_property", {**valid, "__unexpected__": True}))
    return cases


def _schema_fixture(schema: Mapping[str, Any]) -> Any:
    enum = schema.get("enum")
    if isinstance(enum, list) and enum:
        return enum[0]
    schema_type = schema.get("type")
    if schema_type == "string":
        pattern = str(schema.get("pattern") or "")
        if pattern.startswith("^https://"):
            return "https://example.com"
        return "x" * max(1, int(schema.get("minLength") or 0))
    if schema_type == "integer":
        value = int(schema.get("minimum") or 0)
        return value + 1 if "exclusiveMinimum" in schema else value
    if schema_type == "number":
        value = float(schema.get("minimum") or 0.0)
        return value + 1.0 if "exclusiveMinimum" in schema else value
    if schema_type == "boolean":
        return False
    if schema_type == "array":
        item_schema = schema.get("items")
        item_schema = item_schema if isinstance(item_schema, Mapping) else {}
        return [
            _schema_fixture(item_schema)
            for _ in range(int(schema.get("minItems") or 0))
        ]
    if schema_type == "object":
        properties = schema.get("properties")
        properties = properties if isinstance(properties, Mapping) else {}
        required = [
            str(name) for name in schema.get("required", []) if isinstance(name, str)
        ]
        branches = schema.get("oneOf")
        if isinstance(branches, list) and branches and isinstance(branches[0], Mapping):
            required.extend(
                str(name)
                for name in branches[0].get("required", [])
                if isinstance(name, str)
            )
        return {
            name: _schema_fixture(
                properties.get(name)
                if isinstance(properties.get(name), Mapping)
                else {}
            )
            for name in dict.fromkeys(required)
        }
    return {}


def _wrong_type_fixture(schema: Mapping[str, Any]) -> Any | None:
    schema_type = schema.get("type")
    return {
        "string": 7,
        "integer": "not-an-integer",
        "number": "not-a-number",
        "boolean": "not-a-boolean",
        "array": {"not": "an-array"},
        "object": ["not", "an-object"],
    }.get(str(schema_type))


def _without(value: Mapping[str, Any], name: str) -> JsonDict:
    return {key: item for key, item in value.items() if key != name}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)
    report = audit_tool_contract_request_readiness()
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
