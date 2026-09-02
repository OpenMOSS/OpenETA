from __future__ import annotations

from pathlib import Path

import pytest

from sim.envs.univtac.vcpkg_source_recovery import (
    EXPECTED_SHA512,
    TINYGLTF_COMMIT,
    find_historical_asset,
    generate_overlay_portfile,
    recovery_strategy,
    require_fixed_tag_commit,
)


def test_historical_asset_requires_full_expected_sha512(tmp_path: Path) -> None:
    candidate = tmp_path / "tinygltf.tar.gz"
    candidate.write_bytes(b"current archive")
    result = find_historical_asset([tmp_path])
    assert result["historical_asset_recovered"] is False
    assert result["checked"][0]["sha512"] != EXPECTED_SHA512


def test_overlay_changes_only_source_acquisition() -> None:
    original = '''vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO syoyo/tinygltf
    REF "v${VERSION}"
    SHA512 old
    HEAD_REF master
)
vcpkg_replace_string("${SOURCE_PATH}/tiny_gltf.h" "#include \\"json.hpp\\"" "fixed")
'''
    overlay, diff = generate_overlay_portfile(original)
    assert "vcpkg_from_git(" in overlay
    assert f"REF {TINYGLTF_COMMIT}" in overlay
    assert "SHA512 0" not in overlay
    assert diff["non_source_suffix_byte_identical"] is True


def test_tag_movement_fails_closed() -> None:
    require_fixed_tag_commit(TINYGLTF_COMMIT)
    with pytest.raises(ValueError, match="tag moved"):
        require_fixed_tag_commit("0" * 40)


def test_overlay_is_selected_only_when_historical_asset_is_unavailable() -> None:
    assert recovery_strategy(True) == "historical_vcpkg_asset"
    assert recovery_strategy(False) == "exact_commit_overlay"
