from __future__ import annotations

import json
from pathlib import Path

from agent.evals.tool_contract_conformance import audit_tool_contract_conformance
from agent.tools.contracts import build_tool_contract_catalog
from agent.tools.registry import ToolSpec


def _write_event(path: Path, *, name: str, details: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "seq": 1,
                "event": {
                    "phase": "end",
                    "name": name,
                    "success": details.get("operational_success", True),
                    "details": details,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_conformance_audit_accepts_declared_result(tmp_path: Path) -> None:
    source = tmp_path / "rollout" / "tool_calls.jsonl"
    _write_event(
        source,
        name="python_exec",
        details={
            "semantic_outcome": "completed",
            "operational_success": True,
            "outputs": {"result": 3, "stdout": "", "sandbox": "sandbox"},
            "diagnostics": [],
            "recovery_options": [],
        },
    )

    report = audit_tool_contract_conformance(tmp_path)

    assert report["conformant"] is True
    assert report["declared_event_count"] == 1
    assert report["violation_count"] == 0


def test_conformance_audit_reports_success_schema_drift(tmp_path: Path) -> None:
    source = tmp_path / "tool_calls.jsonl"
    _write_event(
        source,
        name="move_to",
        details={
            "semantic_outcome": "mutation_acknowledged",
            "operational_success": True,
            "outputs": {},
            "diagnostics": [],
            "recovery_options": [],
        },
    )

    report = audit_tool_contract_conformance(source)

    assert report["conformant"] is False
    assert report["violation_code_counts"] == {"required_output_missing": 1}
    assert report["violations"][0]["path"] == "details.outputs.motion_summary"


def test_conformance_audit_skips_inferred_shape_but_counts_it(tmp_path: Path) -> None:
    source = tmp_path / "tool_calls.jsonl"
    _write_event(
        source,
        name="scene_detector",
        details={
            "semantic_outcome": "anything",
            "operational_success": True,
            "outputs": {},
        },
    )

    catalog = build_tool_contract_catalog(
        [
            ToolSpec(
                name="scene_detector",
                description="test-only inferred contract",
                category="test",
            )
        ]
    )
    report = audit_tool_contract_conformance(source, catalog=catalog)

    assert report["conformant"] is True
    assert report["maturity_counts"] == {"inferred": 1}
