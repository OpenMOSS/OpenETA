#!/usr/bin/env python3
"""Generate the exact one-line cuRobo Warp public-API backport and its audit."""

from __future__ import annotations

import argparse
import difflib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sim.envs.univtac.source_backport_contract import (
    BASE_COMMIT,
    NEW_EXPRESSION,
    OLD_EXPRESSION,
    TARGET_FILE,
    UPSTREAM_REFERENCE_COMMIT,
    file_sha256,
    make_backported_source,
    validate_semantic_backport,
)


def git(checkout: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=checkout, check=True, capture_output=True, text=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-checkout", type=Path, required=True)
    parser.add_argument("--warp-version", required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    base = args.base_checkout.resolve()
    if git(base, "rev-parse", "HEAD") != BASE_COMMIT:
        raise ValueError("cuRobo base commit mismatch")
    if git(base, "status", "--short", "--untracked-files=no"):
        raise ValueError("cuRobo base checkout is not tracked-clean")
    if git(base, "cat-file", "-t", UPSTREAM_REFERENCE_COMMIT) != "commit":
        raise ValueError("fixed upstream reference commit is unavailable")
    reference = git(base, "show", "--format=fuller", "--stat", UPSTREAM_REFERENCE_COMMIT)
    reference_diff = git(base, "show", "--format=", UPSTREAM_REFERENCE_COMMIT)
    if "wp.torch.device_from_torch" not in reference_diff or "wp.device_from_torch" not in reference_diff:
        raise ValueError("upstream reference does not prove the public API migration")
    private_uses = git(
        base,
        "grep",
        "-nE",
        r"wp\.torch\.|warp\.torch|from[[:space:]]+warp\.torch|import[[:space:]]+warp\.torch",
        "HEAD",
    ).splitlines()
    expected_use = f"HEAD:{TARGET_FILE}:67:        self._wp_device = {OLD_EXPRESSION}"
    if private_uses != [expected_use]:
        raise ValueError(f"private Warp API inventory changed: {private_uses}")
    before = (base / TARGET_FILE).read_text(encoding="utf-8")
    after = make_backported_source(before)
    semantic = validate_semantic_backport(before, after)
    patch_text = "".join(
        difflib.unified_diff(
            before.splitlines(keepends=True),
            after.splitlines(keepends=True),
            fromfile=f"a/{TARGET_FILE}",
            tofile=f"b/{TARGET_FILE}",
        )
    )
    patch_text = "".join("\n" if line == " \n" else line for line in patch_text.splitlines(keepends=True))
    args.patch.parent.mkdir(parents=True, exist_ok=True)
    args.patch.write_text(patch_text, encoding="utf-8")
    semantic.update(
        {
            "base_commit": BASE_COMMIT,
            "upstream_reference_commit": UPSTREAM_REFERENCE_COMMIT,
            "private_api_uses": private_uses,
            "reference_message": git(base, "show", "-s", "--format=%s", UPSTREAM_REFERENCE_COMMIT),
            "reference_parent": git(base, "show", "-s", "--format=%P", UPSTREAM_REFERENCE_COMMIT),
            "reference_changed_files": git(base, "show", "--format=", "--name-only", UPSTREAM_REFERENCE_COMMIT).splitlines(),
            "excluded_upstream_changes": [
                "sweep collision SDF overload changes",
                "mesh extractor control-flow changes",
                "voxel collision test changes",
                "changelog",
            ],
            "reference_summary": reference.splitlines()[-8:],
        }
    )
    args.audit.parent.mkdir(parents=True, exist_ok=True)
    args.audit.write_text(json.dumps(semantic, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "base_commit": BASE_COMMIT,
        "upstream_reference_commit": UPSTREAM_REFERENCE_COMMIT,
        "patch_sha256": file_sha256(args.patch),
        "changed_files": semantic["changed_files"],
        "changed_line_count": semantic["changed_line_count"],
        "old_expression": OLD_EXPRESSION,
        "new_expression": NEW_EXPRESSION,
        "semantic_claim": "Only the deprecated private Warp Torch attribute chain is migrated to its top-level public API.",
        "excluded_upstream_changes": semantic["excluded_upstream_changes"],
        "warp_version_tested": args.warp_version,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
