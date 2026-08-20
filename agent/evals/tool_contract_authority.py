"""Audit durable ToolContract authority configuration against rollout traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from adapter.protocol import JsonDict


SCHEMA_VERSION = "openeta.tool_contract_authority_audit.v1"


def audit_tool_contract_authority(path: str | Path) -> JsonDict:
    root = Path(path)
    manifests = [
        candidate
        for candidate in _discover(root, "manifest.json")
        if _read_object(candidate).get("schema_version") == "openeta.rollout.v1"
    ]
    violations: list[JsonDict] = []
    run_records: list[JsonDict] = []
    for manifest_path in manifests:
        manifest = _read_object(manifest_path)
        provenance = manifest.get("provenance")
        provenance = provenance if isinstance(provenance, dict) else {}
        runtime = provenance.get("tool_contract_runtime")
        if not isinstance(runtime, dict):
            violations.append(
                _violation(manifest_path, "runtime_provenance_missing", "", "")
            )
            continue
        policy = runtime.get("policy")
        policy = policy if isinstance(policy, dict) else {}
        request_authority = {
            str(name) for name in policy.get("request_validation_authority", [])
        }
        gate_authority = {
            str(name) for name in policy.get("gate_repair_envelope_authority", [])
        }
        alignment = runtime.get("planner_pipeline_alignment")
        alignment = alignment if isinstance(alignment, dict) else {}
        if alignment.get("catalog_match") is not True:
            violations.append(
                _violation(manifest_path, "planner_pipeline_catalog_mismatch", "", "")
            )
        if alignment.get("policy_match") is not True:
            violations.append(
                _violation(manifest_path, "planner_pipeline_policy_mismatch", "", "")
            )
        maturity_by_tool = {
            str(row.get("name") or ""): str(
                (row.get("contract") or {}).get("maturity") or ""
            )
            for row in provenance.get("tools", [])
            if isinstance(row, dict) and isinstance(row.get("contract"), dict)
        }
        for tool in sorted(request_authority | gate_authority):
            if maturity_by_tool.get(tool) != "verified":
                violations.append(
                    _violation(
                        manifest_path,
                        "authority_tool_not_verified",
                        tool,
                        maturity_by_tool.get(tool, "missing"),
                    )
                )
        model_call_paths = sorted(manifest_path.parent.rglob("model_calls.jsonl"))
        traces = 0
        enforcing_traces = 0
        for model_calls in model_call_paths:
            for row in _read_jsonl(model_calls):
                decision = row.get("parsed_decision")
                decision = decision if isinstance(decision, dict) else {}
                metadata = decision.get("metadata")
                metadata = metadata if isinstance(metadata, dict) else {}
                trace = metadata.get("tool_contract_shadow_validation")
                if not isinstance(trace, dict) or trace.get("evaluated") is not True:
                    continue
                traces += 1
                tool = str(trace.get("tool") or decision.get("name") or "")
                enforcing = trace.get("enforcing") is True
                enforcing_traces += int(enforcing)
                expected = tool in request_authority
                if enforcing != expected:
                    violations.append(
                        _violation(
                            model_calls,
                            "request_authority_trace_mismatch",
                            tool,
                            f"policy={expected}, trace={enforcing}",
                        )
                    )
                expected_validator = "tool_contract" if expected else "legacy_planner"
                if trace.get("authoritative_validator") != expected_validator:
                    violations.append(
                        _violation(
                            model_calls,
                            "request_validator_identity_mismatch",
                            tool,
                            str(trace.get("authoritative_validator") or ""),
                        )
                    )
        gate_trace_count = 0
        gate_enforcing_count = 0
        for source in sorted(manifest_path.parent.rglob("*.jsonl")):
            if source.name == "model_calls.jsonl":
                continue
            for row in _read_jsonl(source):
                for trace in _find_named_dicts(row, "contract_shadow_validation"):
                    if trace.get("schema_version") != (
                        "openeta.gate_contract_shadow_validation.v1"
                    ):
                        continue
                    if trace.get("evaluated") is not True:
                        continue
                    gate_trace_count += 1
                    tool = str(trace.get("tool") or "")
                    enforcing = trace.get("enforcing") is True
                    gate_enforcing_count += int(enforcing)
                    expected = tool in gate_authority
                    if enforcing != expected:
                        violations.append(
                            _violation(
                                source,
                                "gate_repair_authority_trace_mismatch",
                                tool,
                                f"policy={expected}, trace={enforcing}",
                            )
                        )
                    if trace.get("authoritative_gate") != "legacy_runtime":
                        violations.append(
                            _violation(
                                source,
                                "executable_gate_authority_changed",
                                tool,
                                str(trace.get("authoritative_gate") or ""),
                            )
                        )
        run_records.append(
            {
                "manifest": str(manifest_path),
                "session_id": manifest.get("session_id"),
                "catalog_sha256": runtime.get("catalog_sha256"),
                "request_validation_authority": sorted(request_authority),
                "gate_repair_envelope_authority": sorted(gate_authority),
                "request_trace_count": traces,
                "request_enforcing_trace_count": enforcing_traces,
                "gate_trace_count": gate_trace_count,
                "gate_enforcing_trace_count": gate_enforcing_count,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "source_root": str(root),
        "manifest_count": len(manifests),
        "auditable_manifest_count": len(run_records),
        "conformant": bool(manifests) and not violations,
        "violation_count": len(violations),
        "violations": violations,
        "runs": run_records,
    }


def _discover(path: Path, name: str) -> list[Path]:
    if path.is_file():
        return [path] if path.name == name else []
    return sorted(candidate for candidate in path.rglob(name) if candidate.is_file())


def _read_object(path: Path) -> JsonDict:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def _read_jsonl(path: Path) -> Iterable[JsonDict]:
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        value = json.loads(line)
        if isinstance(value, dict):
            yield value


def _find_named_dicts(value: object, key: str) -> Iterable[JsonDict]:
    if isinstance(value, dict):
        candidate = value.get(key)
        if isinstance(candidate, dict):
            yield candidate
        for item in value.values():
            yield from _find_named_dicts(item, key)
    elif isinstance(value, list):
        for item in value:
            yield from _find_named_dicts(item, key)


def _violation(path: Path, code: str, tool: str, observed: str) -> JsonDict:
    return {
        "code": code,
        "source": str(path),
        "tool": tool,
        "observed": observed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)
    report = audit_tool_contract_authority(args.run)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0 if report["conformant"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
