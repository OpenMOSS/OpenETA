from __future__ import annotations

import ast
from pathlib import Path

from sim.envs.univtac.libuipc_core_probe import (
    classify_core_probe,
    core_run_policy,
    expected_stage_events,
)


ROOT = Path(__file__).resolve().parents[2]


def test_complete_core_stage_sequence_passes() -> None:
    payload = {"success": True, "events": [{"event": name} for name in expected_stage_events()]}
    assert classify_core_probe(payload) == "passed"


def test_timeout_and_native_error_are_distinct() -> None:
    assert classify_core_probe({}, timed_out=True) == "libuipc_core_sm120_timeout"
    assert classify_core_probe({"success": False}) == "libuipc_core_sm120_native_error"


def test_first_failure_is_never_retried_and_first_success_repeats_once() -> None:
    assert core_run_policy([]) == "run1"
    assert core_run_policy([{"classification": "libuipc_core_sm120_native_error"}]) is None
    assert core_run_policy([{"classification": "passed"}]) == "run2"
    assert core_run_policy([{"classification": "passed"}, {"classification": "passed"}]) is None


def test_core_probe_has_no_forbidden_wrapper_imports() -> None:
    tree = ast.parse((ROOT / "scripts/univtac/probe_libuipc_core.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    forbidden = ("isaacsim", "isaaclab", "omni", "carb", "tacex_uipc", "openeta")
    assert not any(name.startswith(forbidden) for name in imported)
