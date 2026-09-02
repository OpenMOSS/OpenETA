"""Contracts for recovering the republished tinygltf 2.9.6 source."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

VCPKG_COMMIT = "dd3097e305afa53f7b4312371f62058d2e665320"
TINYGLTF_COMMIT = "26422192e2908a562b641175dde18489824e609e"
EXPECTED_SHA512 = "89397dc2c8884a54ea0c370251449459a200057b5e470210c4468f43c4623947500630b1a67ff6319e0998e648487367398f134711bc7d2c42ebdbd7097770b3"


def require_fixed_tag_commit(actual: str | None) -> None:
    if actual != TINYGLTF_COMMIT:
        raise ValueError("tinygltf tag moved")


def recovery_strategy(historical_asset_recovered: bool) -> str:
    return "historical_vcpkg_asset" if historical_asset_recovered else "exact_commit_overlay"


def sha512_file(path: Path) -> str:
    digest = hashlib.sha512()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_historical_asset(roots: Iterable[Path]) -> dict[str, Any]:
    checked = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*tinygltf*")):
            if not path.is_file():
                continue
            actual = sha512_file(path)
            checked.append({"path": str(path.resolve()), "sha512": actual, "size": path.stat().st_size})
            if actual == EXPECTED_SHA512:
                return {"historical_asset_recovered": True, "asset": checked[-1], "checked": checked}
    return {"historical_asset_recovered": False, "asset": None, "checked": checked}


def generate_overlay_portfile(original: str) -> tuple[str, dict[str, Any]]:
    pattern = re.compile(r"vcpkg_from_github\(.*?\)\n", re.DOTALL)
    matches = pattern.findall(original)
    if len(matches) != 1:
        raise ValueError("expected exactly one vcpkg_from_github block")
    replacement = (
        "vcpkg_from_git(\n"
        "    OUT_SOURCE_PATH SOURCE_PATH\n"
        "    URL https://github.com/syoyo/tinygltf.git\n"
        f"    REF {TINYGLTF_COMMIT}\n"
        "    HEAD_REF master\n"
        ")\n"
    )
    overlay = pattern.sub(replacement, original, count=1)
    if "SHA512 0" in overlay or "v2.9.6" in overlay:
        raise ValueError("overlay must use only the exact commit source acquisition")
    return overlay, {
        "source_fetch_backend": {"from": "github_archive", "to": "exact_git_commit"},
        "source_identity": {"from": "v2.9.6", "to": TINYGLTF_COMMIT},
        "non_source_suffix_byte_identical": overlay.split(")\n", 1)[1] == original.split(")\n", 1)[1],
    }
