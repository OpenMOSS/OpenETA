from __future__ import annotations

from sim.envs.univtac.source_tree_manifest import compare_manifests, directory_manifest


def test_normalized_tree_comparison_distinguishes_content_mode_missing_and_extra() -> None:
    base = {"a": {"mode": "100644", "sha256": "x"}}
    assert compare_manifests(base, dict(base))["equal"] is True
    assert compare_manifests(base, {})["missing"] == ["a"]
    assert compare_manifests({}, base)["extra"] == ["a"]
    changed = {"a": {"mode": "100755", "sha256": "x"}}
    assert compare_manifests(base, changed)["changed"][0]["path"] == "a"


def test_symlink_target_is_part_of_manifest_identity() -> None:
    left = {"link": {"mode": "120000", "kind": "symlink", "symlink_target": "a"}}
    right = {"link": {"mode": "120000", "kind": "symlink", "symlink_target": "b"}}
    assert compare_manifests(left, right)["equal"] is False


def test_directory_manifest_uses_only_trusted_tracked_paths(tmp_path) -> None:
    (tmp_path / "tracked").write_text("value", encoding="utf-8")
    (tmp_path / "generated").write_text("ignored", encoding="utf-8")
    manifest = directory_manifest(tmp_path, {"tracked": {}})
    assert list(manifest) == ["tracked"]
