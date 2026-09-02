"""Read-only helpers for comparing pinned UniVTAC source trees."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


FULL_SHA = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class PinnedSource:
    label: str
    repository: str
    commit: str
    checkout: Path
    task_root: Path

    def to_dict(self) -> dict[str, str]:
        payload = asdict(self)
        payload["checkout"] = str(self.checkout.resolve())
        payload["task_root"] = str(self.task_root.resolve())
        return payload


def _git(checkout: Path, *args: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=checkout,
        check=check,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def validate_pinned_source(source: PinnedSource, *, require_detached: bool) -> dict[str, object]:
    if not FULL_SHA.fullmatch(source.commit):
        raise ValueError(f"{source.label} commit must be a full immutable SHA")
    checkout = source.checkout.expanduser().resolve()
    task_root = source.task_root.expanduser().resolve()
    if not checkout.is_dir() or not task_root.is_dir():
        raise FileNotFoundError(f"missing pinned source tree: {source.label}")
    actual = _git(checkout, "rev-parse", "HEAD")
    if actual != source.commit:
        raise RuntimeError(
            f"{source.label} commit mismatch: expected {source.commit}, got {actual}"
        )
    symbolic = _git(checkout, "symbolic-ref", "-q", "HEAD", check=False) or None
    if require_detached and symbolic is not None:
        raise RuntimeError(f"{source.label} audit checkout must be detached")
    return {
        **source.to_dict(),
        "actual_commit": actual,
        "detached": symbolic is None,
        "symbolic_ref": symbolic,
        "read_only_audit": True,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def line_hits(path: Path, patterns: Iterable[str]) -> dict[str, list[int]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return {
        pattern: [index for index, line in enumerate(lines, 1) if pattern in line]
        for pattern in patterns
    }


def require_text(root: Path, relative: str, snippets: Iterable[str]) -> str:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"required audit source missing: {path}")
    text = path.read_text(encoding="utf-8")
    missing = [snippet for snippet in snippets if snippet not in text]
    if missing:
        raise RuntimeError(f"source drift in {relative}; missing {missing}")
    return text


def build_hash_manifest(
    sources: Iterable[PinnedSource], relative_files: Iterable[str]
) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for source in sources:
        for relative in relative_files:
            path = source.task_root / relative
            records.append(
                {
                    "source": source.label,
                    "relative_path": relative,
                    "absolute_path": str(path.resolve()),
                    "exists": path.is_file(),
                    "size": path.stat().st_size if path.is_file() else None,
                    "sha256": sha256_file(path) if path.is_file() else None,
                }
            )
    return {
        "schema_version": "openeta.univtac.source_hash_manifest.v1",
        "files": records,
    }


def deterministic_json(payload: object) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
