from __future__ import annotations

import ast
import inspect
from pathlib import Path

import pytest

from scripts.univtac import audit_isaac51_compatibility as audit_script
from sim.envs.univtac.source_parity import PinnedSource, validate_pinned_source


LEGACY_CHECKOUT = Path(
    "/home/ubuntu/wybcode/.worktrees/ftp1-policy/r02-official-89fa681"
)
ISAAC51_CHECKOUT = Path(
    "/home/ubuntu/wybcode/.cache/univtac-audit/isaac51-371fac679173"
)
LEGACY_COMMIT = "89fa681d6c014cce28300946b7526db808e0b1c1"
ISAAC51_COMMIT = "371fac67917307026be8f00869fcc1b61c623a9f"


def test_both_sources_are_exact_commits_and_new_checkout_is_detached() -> None:
    legacy = PinnedSource(
        "legacy", "michaelyuancb/ftp1-policy", LEGACY_COMMIT,
        LEGACY_CHECKOUT, LEGACY_CHECKOUT / "UniVTAC"
    )
    new = PinnedSource(
        "isaac51", "univtac/UniVTAC", ISAAC51_COMMIT,
        ISAAC51_CHECKOUT, ISAAC51_CHECKOUT
    )
    assert validate_pinned_source(legacy, require_detached=False)["actual_commit"] == LEGACY_COMMIT
    result = validate_pinned_source(new, require_detached=True)
    assert result["actual_commit"] == ISAAC51_COMMIT
    assert result["detached"] is True


def test_branch_name_cannot_replace_full_commit() -> None:
    source = PinnedSource(
        "moving", "univtac/UniVTAC", "isaac51",
        ISAAC51_CHECKOUT, ISAAC51_CHECKOUT
    )
    with pytest.raises(ValueError, match="full immutable SHA"):
        validate_pinned_source(source, require_detached=True)


def test_static_audit_does_not_import_runtime_modules() -> None:
    tree = ast.parse(inspect.getsource(audit_script))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported.isdisjoint({"isaacsim", "isaaclab", "omni", "uipc", "tacex_uipc"})


def test_static_audit_has_no_install_build_or_simulator_process_runner() -> None:
    source = inspect.getsource(audit_script)
    assert "subprocess" not in source
    assert "os.system" not in source
    assert "Popen" not in source
    assert "run_managed_process" not in source


def test_vendor_tree_remains_unmodified() -> None:
    import subprocess

    completed = subprocess.run(
        ["git", "diff", "--quiet", "--", "third_party/ftp1-policy/UniVTAC"],
        cwd=Path(__file__).resolve().parents[2],
        check=False,
    )
    assert completed.returncode == 0
