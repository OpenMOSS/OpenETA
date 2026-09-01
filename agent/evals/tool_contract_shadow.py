"""Audit Planner request-schema shadow parity from durable rollout records."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from adapter.protocol import JsonDict


SCHEMA_VERSION = "openeta.tool_contract_shadow_report.v1"
SHADOW_SCHEMA_VERSION = "openeta.tool_contract_shadow_validation.v1"


def audit_tool_contract_shadow(path: str | Path) -> JsonDict:
    """Summarize non-enforcing contract/legacy acceptance parity."""

    sources = _discover_sources(Path(path))
    model_call_count = 0
    records: list[JsonDict] = []
    tool_counts: Counter[str] = Counter()
    mismatch_tool_counts: Counter[str] = Counter()
    for source in sources:
        for row in _read_jsonl(source):
            model_call_count += 1
            decision = row.get("parsed_decision")
            decision = decision if isinstance(decision, dict) else {}
            metadata = decision.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            shadow = metadata.get("tool_contract_shadow_validation")
            if not isinstance(shadow, dict):
                continue
            tool = str(shadow.get("tool") or decision.get("name") or "")
            evaluated = shadow.get("evaluated") is True
            match = shadow.get("acceptance_match") is True if evaluated else None
            tool_counts[tool] += 1
            record: JsonDict = {
                "source": str(source),
                "seq": row.get("seq"),
                "tool": tool,
                "schema_version": shadow.get("schema_version"),
                "contract_maturity": shadow.get("contract_maturity"),
                "evaluated": evaluated,
                "enforcing": shadow.get("enforcing") is True,
                "legacy_accepted": shadow.get("legacy_accepted"),
                "contract_accepted": shadow.get("contract_accepted"),
                "acceptance_match": match,
                "legacy_errors": list(shadow.get("legacy_errors") or []),
                "contract_violations": list(shadow.get("contract_violations") or []),
            }
            if evaluated and not match:
                mismatch_tool_counts[tool] += 1
            records.append(record)
    evaluated_records = [record for record in records if record["evaluated"]]
    mismatches = [record for record in evaluated_records if not record["acceptance_match"]]
    enforcing_records = [record for record in records if record["enforcing"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "shadow_schema_version": SHADOW_SCHEMA_VERSION,
        "source_root": str(Path(path)),
        "source_count": len(sources),
        "model_call_count": model_call_count,
        "shadow_record_count": len(records),
        "evaluated_record_count": len(evaluated_records),
        "inventory_only_record_count": len(records) - len(evaluated_records),
        "acceptance_mismatch_count": len(mismatches),
        "unexpected_enforcement_count": len(enforcing_records),
        "parity": bool(evaluated_records) and not mismatches,
        "observational_only": not enforcing_records,
        "tool_counts": dict(sorted(tool_counts.items())),
        "mismatch_tool_counts": dict(sorted(mismatch_tool_counts.items())),
        "mismatches": mismatches,
        "records": records,
    }


def _discover_sources(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(path)
    direct = path / "model_calls.jsonl"
    if direct.is_file():
        return [direct]
    sources = sorted(
        candidate for candidate in path.rglob("model_calls.jsonl") if candidate.is_file()
    )
    if not sources:
        raise FileNotFoundError(f"no model_calls.jsonl found under {path}")
    return sources


def _read_jsonl(path: Path) -> Iterable[JsonDict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(value, dict):
                yield value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-calls", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)
    report = audit_tool_contract_shadow(args.model_calls)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0 if report["parity"] and report["observational_only"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
