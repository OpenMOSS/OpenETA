from __future__ import annotations

import copy
import io
import tarfile
from pathlib import Path

import pytest
import yaml

from sim.envs.univtac.source_to_wheel_contract import (
    EXPECTED_PACKAGES,
    audit_pure_python_scope,
    audit_tar_archive,
    build_derived_lock,
    compare_builds,
    source_tree_manifest,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/univtac/legacy_sdist_wheel_bridge.yaml"


def config() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def make_tar(path: Path, *, member: str = "demo-1.0/setup.py", data: bytes = b"from setuptools import setup\nsetup()\n", kind: bytes | None = None, link: str = "") -> None:
    with tarfile.open(path, "w:gz") as archive:
        info = tarfile.TarInfo(member)
        info.mode = 0o644
        info.type = kind or tarfile.REGTYPE
        info.linkname = link
        if info.type == tarfile.REGTYPE:
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        else:
            archive.addfile(info)


def test_config_allows_exactly_three_sources_and_two_builds() -> None:
    payload = config()
    validate_config(payload)
    assert set(payload["packages"]) == set(EXPECTED_PACKAGES)
    assert payload["build_count"] == 2
    changed = copy.deepcopy(payload)
    changed["packages"]["fourth"] = copy.deepcopy(changed["packages"]["pyperclip"])
    with pytest.raises(ValueError, match="three authorized"):
        validate_config(changed)


def test_archive_safety_rejects_escape_and_special_node(tmp_path: Path) -> None:
    safe = tmp_path / "safe.tar.gz"
    make_tar(safe)
    assert audit_tar_archive(safe, maximum_bytes=10000, maximum_members=10)["success"] is True
    escape = tmp_path / "escape.tar.gz"
    make_tar(escape, member="../outside")
    with pytest.raises(ValueError, match="unsafe"):
        audit_tar_archive(escape, maximum_bytes=10000, maximum_members=10)
    link = tmp_path / "link.tar.gz"
    make_tar(link, member="demo-1.0/link", kind=tarfile.SYMTYPE, link="../../outside")
    with pytest.raises(ValueError, match="unsafe|escapes"):
        audit_tar_archive(link, maximum_bytes=10000, maximum_members=10)
    fifo = tmp_path / "fifo.tar.gz"
    make_tar(fifo, member="demo-1.0/fifo", kind=tarfile.FIFOTYPE)
    with pytest.raises(ValueError, match="special"):
        audit_tar_archive(fifo, maximum_bytes=10000, maximum_members=10)


def test_tree_manifest_is_deterministic_and_native_scope_fails(tmp_path: Path) -> None:
    root = tmp_path / "source"
    root.mkdir()
    (root / "setup.py").write_text("from setuptools import setup\nsetup()\n")
    (root / "pkg.py").write_text("VALUE = 1\n")
    first = source_tree_manifest(root)
    second = source_tree_manifest(root)
    assert first["normalized_tree_sha256"] == second["normalized_tree_sha256"]
    assert audit_pure_python_scope(root)["success"] is True
    (root / "native.c").write_text("int x;")
    assert audit_pure_python_scope(root)["success"] is False


def test_build_comparison_requires_byte_and_entry_identity() -> None:
    first = {"filename": "x.whl", "size": 2, "sha256": "a", "entries": [{"path": "x", "sha256": "b"}]}
    assert compare_builds(first, copy.deepcopy(first))["reproducible"] is True
    second = copy.deepcopy(first)
    second["sha256"] = "c"
    assert compare_builds(first, second) == {"reproducible": False, "differences": ["sha256"]}


def test_derived_lock_changes_only_three_source_records() -> None:
    records = []
    transforms = []
    for index in range(173):
        name = f"wheel-{index}"
        record = {"index": index, "name": name, "version": "1", "artifact_type": "wheel", "filename": f"{name}.whl", "sha256": str(index), "url": "https://example", "url_scheme": "https", "origin_host": "example", "wheel_tags": ["py3-none-any"], "wheel_tag_compatible": True}
        records.append(record)
    for index, (name, values) in zip((29, 99, 135), EXPECTED_PACKAGES.items(), strict=True):
        version, filename, sha, _ = values
        records[index].update({"name": name, "version": version, "artifact_type": "sdist", "filename": filename, "sha256": sha})
        transforms.append({"canonical_name": name, "version": version, "original_p0a_record_index": index, "final_wheel_filename": f"{name}-derived.whl", "final_wheel_sha256": f"derived-{name}", "final_wheel_uri": f"file:///derived/{name}.whl", "wheel_tags": ["py3-none-any"], "transformation_sha256": f"transform-{name}"})
    result = build_derived_lock({"records": records}, transforms)
    assert result["final_install_record_count"] == 173
    assert result["final_install_sdist_count"] == 0
    assert sum(item["url_scheme"] == "file" for item in result["records"]) == 3
