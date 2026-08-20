"""Replay durable ToolResult events against the machine-readable contract catalog."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable

from adapter.protocol import JsonDict
from agent.tools.contracts import (
    ContractMaturity,
    ToolContractCatalog,
    build_default_tool_contract_catalog,
    check_tool_result_conformance,
)
from agent.tools.registry import build_default_tool_registry


SCHEMA_VERSION = "openeta.tool_contract_conformance_report.v1"


def audit_tool_contract_conformance(
    path: str | Path,
    *,
    catalog: ToolContractCatalog | None = None,
) -> JsonDict:
    """Audit one event file or every rollout event file below a directory."""

    contract_catalog = catalog or build_default_tool_contract_catalog(
        build_default_tool_registry().list()
    )
    sources = _discover_sources(Path(path))
    violations: list[JsonDict] = []
    tool_counts: Counter[str] = Counter()
    maturity_counts: Counter[str] = Counter()
    semantic_counts: Counter[str] = Counter()
    event_count = 0
    for source in sources:
        for row in _read_jsonl(source):
            event = row.get("event")
            if not isinstance(event, dict) or event.get("phase") != "end":
                continue
            event_count += 1
            name = str(event.get("name") or "")
            tool_counts[name] += 1
            try:
                contract = contract_catalog.get(name)
            except KeyError:
                maturity_counts["unknown"] += 1
                violations.append(
                    {
                        "source": str(source),
                        "seq": row.get("seq"),
                        "tool": name,
                        "code": "unknown_tool_contract",
                        "path": "event.name",
                        "message": f"No ToolContract is registered for {name!r}.",
                    }
                )
                continue
            maturity_counts[contract.maturity.value] += 1
            details = event.get("details")
            details = details if isinstance(details, dict) else {}
            semantic = str(details.get("semantic_outcome") or "")
            semantic_counts[f"{name}:{semantic}"] += 1
            if contract.maturity is ContractMaturity.INFERRED:
                continue
            for violation in check_tool_result_conformance(contract, details):
                violations.append(
                    {
                        "source": str(source),
                        "seq": row.get("seq"),
                        **violation.to_dict(),
                        "semantic_outcome": semantic,
                        "operational_success": details.get("operational_success"),
                    }
                )
    violation_codes = Counter(str(item["code"]) for item in violations)
    violation_tools = Counter(str(item["tool"]) for item in violations)
    return {
        "schema_version": SCHEMA_VERSION,
        "source_root": str(Path(path)),
        "source_count": len(sources),
        "tool_event_count": event_count,
        "declared_event_count": sum(
            count
            for maturity, count in maturity_counts.items()
            if maturity in {"declared", "verified"}
        ),
        "conformant": not violations,
        "violation_count": len(violations),
        "maturity_counts": dict(sorted(maturity_counts.items())),
        "tool_counts": dict(sorted(tool_counts.items())),
        "semantic_counts": dict(sorted(semantic_counts.items())),
        "violation_code_counts": dict(sorted(violation_codes.items())),
        "violation_tool_counts": dict(sorted(violation_tools.items())),
        "violations": violations,
    }


def _discover_sources(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(path)
    direct = path / "tool_calls.jsonl"
    if direct.is_file():
        return [direct]
    sources = sorted(candidate for candidate in path.rglob("tool_calls.jsonl") if candidate.is_file())
    if not sources:
        raise FileNotFoundError(f"no tool_calls.jsonl found under {path}")
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
    parser.add_argument("--tool-events", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)
    report = audit_tool_contract_conformance(args.tool_events)
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0 if report["conformant"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
