"""Normalized Git and archive source-tree manifests."""

from __future__ import annotations

import hashlib
import os
import subprocess
import tarfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_manifest(checkout: Path) -> dict[str, dict[str, Any]]:
    completed = subprocess.run(
        ["git", "ls-files", "-s", "-z"], cwd=checkout, check=True, capture_output=True
    )
    records = {}
    for raw in completed.stdout.split(b"\0"):
        if not raw:
            continue
        metadata, path_bytes = raw.split(b"\t", 1)
        mode, object_id, _stage = metadata.decode().split()
        relative = path_bytes.decode("utf-8", errors="surrogateescape")
        path = checkout / relative
        if mode == "120000":
            target = os.readlink(os.fsencode(path)).decode("utf-8", errors="surrogateescape")
            records[relative] = {"mode": mode, "kind": "symlink", "symlink_target": target}
        elif mode == "160000":
            records[relative] = {"mode": mode, "kind": "submodule", "object_id": object_id}
        else:
            data = path.read_bytes()
            records[relative] = {"mode": mode, "kind": "file", "size": len(data), "sha256": _sha256(data)}
    return records


def archive_manifest(path: Path) -> dict[str, dict[str, Any]]:
    records = {}
    with tarfile.open(path, "r:*") as archive:
        members = [member for member in archive.getmembers() if member.isfile() or member.issym()]
        tops = {PurePosixPath(member.name).parts[0] for member in members if PurePosixPath(member.name).parts}
        if len(tops) != 1:
            raise ValueError("archive must contain exactly one top-level directory")
        for member in members:
            parts = PurePosixPath(member.name).parts[1:]
            if not parts:
                continue
            relative = PurePosixPath(*parts).as_posix()
            if member.issym():
                records[relative] = {"mode": "120000", "kind": "symlink", "symlink_target": member.linkname}
            else:
                handle = archive.extractfile(member)
                data = handle.read() if handle else b""
                mode = "100755" if member.mode & 0o111 else "100644"
                records[relative] = {"mode": mode, "kind": "file", "size": len(data), "sha256": _sha256(data)}
    return records


def directory_manifest(root: Path, tracked_paths: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Manifest exactly the paths declared by a trusted tracked-tree manifest."""

    records: dict[str, dict[str, Any]] = {}
    for relative in sorted(tracked_paths):
        path = root / relative
        if path.is_symlink():
            records[relative] = {
                "mode": "120000",
                "kind": "symlink",
                "symlink_target": os.readlink(os.fsencode(path)).decode(
                    "utf-8", errors="surrogateescape"
                ),
            }
        elif path.is_file():
            data = path.read_bytes()
            records[relative] = {
                "mode": "100755" if os.access(path, os.X_OK) else "100644",
                "kind": "file",
                "size": len(data),
                "sha256": _sha256(data),
            }
    return records


def compare_manifests(baseline: Mapping[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    baseline_paths = set(baseline)
    candidate_paths = set(candidate)
    changed = [
        {"path": path, "baseline": baseline[path], "candidate": candidate[path]}
        for path in sorted(baseline_paths & candidate_paths)
        if baseline[path] != candidate[path]
    ]
    missing = sorted(baseline_paths - candidate_paths)
    extra = sorted(candidate_paths - baseline_paths)
    return {"equal": not missing and not extra and not changed, "missing": missing, "extra": extra, "changed": changed}
