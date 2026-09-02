from __future__ import annotations

import inspect
from pathlib import Path

import yaml

from scripts.univtac import resume_blackwell_native_bridge as runner


def test_r081_config_pins_every_native_source() -> None:
    config = yaml.safe_load(Path("configs/univtac/tinygltf_source_recovery.yaml").read_text())
    blackwell = yaml.safe_load(Path("configs/univtac/blackwell_native_bridge.yaml").read_text())
    assert config["source"]["univtac_commit"] == "371fac67917307026be8f00869fcc1b61c623a9f"
    assert config["source"]["vcpkg_commit"] == "dd3097e305afa53f7b4312371f62058d2e665320"
    assert config["source"]["tinygltf_commit"] == "26422192e2908a562b641175dde18489824e609e"
    assert blackwell["source"]["curobo_commit"] == "ebb71702f3f70e767f40fd8e050674af0288abe8"


def test_r081_gate_order_is_fail_closed() -> None:
    source = inspect.getsource(runner.main)
    assert source.index('name="u0r"') < source.index('name="source_identity"')
    assert source.index('name="source_identity"') < source.index('manifest["gates"]["B0"]')
    assert source.index('manifest["gates"]["B0"]') < source.index('f"U1 run {index} failed"')
    assert source.index('f"U1 run {index} failed"') < source.index('unexpected preexisting cuRobo checkout')


def test_r081_has_one_overlay_and_one_u0r_only() -> None:
    source = inspect.getsource(runner.main)
    assert source.count('name="u0r"') == 1
    assert 'overlay_root / "tinygltf"' in source
    assert "SHA512 0" not in source


def test_r081_never_starts_simulator_or_modifies_sysctl() -> None:
    source = inspect.getsource(runner)
    assert "AppLauncher" not in source
    assert "Isaac Sim" not in source
    assert '"sudo"' not in source
    assert "sysctl -w" not in source


def test_r081_fingerprints_all_protected_environments() -> None:
    source = inspect.getsource(runner.main)
    for token in ('"legacy"', '"r07"', '"r08"', '"openeta_uv"'):
        assert token in source
