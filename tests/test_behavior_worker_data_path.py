"""The BEHAVIOR worker must be handed the dataset path OmniGibson reads.

``omnigibson.macros.determine_data_path()`` consults exactly one environment
variable, ``OMNIGIBSON_DATA_PATH``, and otherwise falls back to a path relative
to the installed module.  The 37 GB dataset lives outside the conda env, so that
fallback is wrong on every real machine -- and it fails at *worker boot*, inside
the other interpreter, as ``AssertionError: Data path ... does not exist!``.
That reads like a missing download rather than a wrong default, so it is worth
pinning here where it costs milliseconds.

The specific bug: the setup script exported ``OMNIGIBSON_DATASET_PATH``, which
nothing reads, while ``worker_mgr`` filled in ``OMNIGIBSON_DATA_PATH`` from a
venv-relative layout that no longer exists.  Every unit test passed and no
BEHAVIOR env could be created through the server.
"""
from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def test_setup_script_exports_the_name_omnigibson_reads() -> None:
    """Guard the variable name itself, which is the whole bug."""
    src = (REPO / "scripts" / "setup_behavior.sh").read_text()
    assert "export OMNIGIBSON_DATA_PATH=" in src
    # The near-miss name is inert; allow it only in prose, never as an export.
    assert "export OMNIGIBSON_DATASET_PATH=" not in src


def test_activate_parser_reads_plain_exports(tmp_path, monkeypatch) -> None:
    from sim.mcp_server import worker_mgr

    venvs = tmp_path / "venvs"
    venvs.mkdir()
    (venvs / "behavior_activate_extra.sh").write_text(
        "# a comment\n"
        "export OMNI_KIT_ACCEPT_EULA=YES\n"
        'export OMNIGIBSON_DATA_PATH="/data/BEHAVIOR-1K/datasets"\n'
        "export DERIVED=$BEHAVIOR_ROOT/datasets\n"   # needs a shell; must skip
        "not_an_export=1\n"
    )
    monkeypatch.setattr(worker_mgr, "_SIM_DIR", tmp_path)

    got = worker_mgr._behavior_env_from_activate()
    assert got["OMNIGIBSON_DATA_PATH"] == "/data/BEHAVIOR-1K/datasets"
    assert got["OMNI_KIT_ACCEPT_EULA"] == "YES"
    # Unexpanded values would be handed to the worker verbatim as a literal
    # "$BEHAVIOR_ROOT/datasets", which is worse than leaving them unset.
    assert "DERIVED" not in got
    assert "not_an_export" not in got


def test_missing_activate_file_is_not_an_error(tmp_path, monkeypatch) -> None:
    """A machine set up before this file existed must still start a worker."""
    from sim.mcp_server import worker_mgr

    monkeypatch.setattr(worker_mgr, "_SIM_DIR", tmp_path / "nope")
    assert worker_mgr._behavior_env_from_activate() == {}


def test_installed_activate_file_points_at_a_real_dataset() -> None:
    """On a provisioned machine, the recorded path must actually exist.

    Skipped rather than failed where BEHAVIOR is not installed, so the suite
    stays runnable on a laptop.
    """
    from sim.mcp_server import worker_mgr

    got = worker_mgr._behavior_env_from_activate()
    path = got.get("OMNIGIBSON_DATA_PATH")
    if not path:
        pytest.skip("BEHAVIOR not provisioned on this machine")
    assert Path(path).is_dir(), (
        f"OMNIGIBSON_DATA_PATH={path} does not exist; the worker will die at "
        "boot with 'Data path ... does not exist'")
