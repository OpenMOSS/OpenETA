from __future__ import annotations

import ast
from pathlib import Path


def test_binary_auditor_does_not_import_simulator_package_tree() -> None:
    source = Path("scripts/univtac/audit_blackwell_binaries.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )
    assert not any(name == "sim" or name.startswith("sim.") for name in imported)
