#!/usr/bin/env python3
"""Re-apply the two source patches cuRobo needs to build and run here.

``third_party/`` is gitignored (`.gitignore:62`), so the vendored cuRobo tree is
not version-controlled and these edits do not survive a re-clone or a re-vendor.
Without them the build fails to compile on sm_120 and, if it does compile,
crashes constructing ``RobotWorld``.  Run this after obtaining the tree and
before ``uv pip install -e``.

Idempotent: each patch is skipped when its marker is already present, so running
this twice is safe and it can be wired into setup unconditionally.

    python scripts/patch_vendored_curobo.py [--curobo-dir DIR] [--check]

``--check`` reports status and exits non-zero if any patch is missing, without
writing anything -- suitable for CI or a preflight assertion.

Verified against the tree vendored on 2026-08-17 (self-reports version 0.0.0;
the upstream commit is unrecoverable because the vendored directory's gitlink
pointed at a superproject module dir that no longer exists).  Both patches are
guarded so they stay correct on toolchains that do not need them.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CUROBO_DIR = REPO_ROOT / "third_party" / "curobo"


class Patch:
    """One search-and-replace against a vendored file."""

    def __init__(self, relative_path: str, marker: str, old: str, new: str, why: str) -> None:
        self.relative_path = relative_path
        self.marker = marker  # already-applied sentinel
        self.old = old
        self.new = new
        self.why = why


# nvcc resolves an unqualified ::lerp to std::lerp for compute_80 and above once
# <cmath> is in scope (C++17 added std::lerp), turning cuRobo's scalar overload
# into a redeclaration conflict.  Every .cu that includes helper_math.h fails.
# No cuRobo kernel calls the scalar form and the float2/3/4 overloads take
# distinct types, so skipping it where std::lerp exists costs nothing.
# https://forums.developer.nvidia.com/t/nvcc-on-linux-tries-to-resolve-lerp-as-std-lerp-with-compute-80-or-higher/331089
LERP_PATCH = Patch(
    relative_path="src/curobo/curobolib/cpp/helper_math.h",
    marker="__cpp_lib_interpolate",
    old=(
        "inline __device__ __host__ float lerp(float a, float b, float t)\n"
        "{\n"
        "    return a + t*(b-a);\n"
        "}"
    ),
    new=(
        "// nvcc resolves unqualified ::lerp to std::lerp for compute_80 and above once\n"
        "// <cmath> is in scope (C++17 added std::lerp), so this scalar overload becomes\n"
        "// a redeclaration conflict rather than an overload.  The float2/float3/float4\n"
        "// overloads below take distinct parameter types and are unaffected.  No cuRobo\n"
        "// kernel calls the scalar form, so skipping it where std::lerp already exists\n"
        "// costs nothing.  See:\n"
        "// https://forums.developer.nvidia.com/t/nvcc-on-linux-tries-to-resolve-lerp-as-std-lerp-with-compute-80-or-higher/331089\n"
        "#if !defined(__cpp_lib_interpolate)\n"
        "inline __device__ __host__ float lerp(float a, float b, float t)\n"
        "{\n"
        "    return a + t*(b-a);\n"
        "}\n"
        "#endif"
    ),
    why="std::lerp redeclaration conflict; blocks every .cu including helper_math.h",
)

# warp >= 1.16 moved the torch interop to warp._src.torch and re-exports it at
# the top level; the warp.torch submodule cuRobo reaches through is gone, so
# RobotWorld construction dies with AttributeError.  Sole call site in the tree.
WARP_PATCH = Patch(
    relative_path="src/curobo/geom/sdf/world_mesh.py",
    marker='hasattr(wp, "device_from_torch")',
    old="        self._wp_device = wp.torch.device_from_torch(self.tensor_args.device)",
    new=(
        "        # warp >= 1.16 moved the torch interop into warp._src.torch and re-exports\n"
        "        # it at the top level; the warp.torch submodule it used to live in is gone.\n"
        "        # Prefer the top-level name and fall back for older warp builds.\n"
        '        if hasattr(wp, "device_from_torch"):\n'
        "            self._wp_device = wp.device_from_torch(self.tensor_args.device)\n"
        "        else:\n"
        "            self._wp_device = wp.torch.device_from_torch(self.tensor_args.device)"
    ),
    why="warp 1.16 removed warp.torch; RobotWorld construction raises AttributeError",
)

PATCHES = (LERP_PATCH, WARP_PATCH)


def apply(patch: Patch, curobo_dir: Path, *, check_only: bool) -> str:
    """Return one of ``applied``, ``already``, ``missing-file``, ``no-match``."""
    target = curobo_dir / patch.relative_path
    if not target.is_file():
        return "missing-file"

    text = target.read_text(encoding="utf-8")
    if patch.marker in text:
        return "already"
    if patch.old not in text:
        # Upstream drifted; a blind replace would silently do nothing.
        return "no-match"
    if check_only:
        return "would-apply"

    target.write_text(text.replace(patch.old, patch.new, 1), encoding="utf-8")
    return "applied"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--curobo-dir", type=Path, default=DEFAULT_CUROBO_DIR,
                        help=f"vendored cuRobo root (default: {DEFAULT_CUROBO_DIR})")
    parser.add_argument("--check", action="store_true",
                        help="report status only; exit non-zero if a patch is missing")
    args = parser.parse_args()

    curobo_dir = args.curobo_dir.resolve()
    if not curobo_dir.is_dir():
        print(f"cuRobo dir not found: {curobo_dir}", file=sys.stderr)
        return 2

    print(f"cuRobo tree: {curobo_dir}")
    failed = False
    for patch in PATCHES:
        status = apply(patch, curobo_dir, check_only=args.check)
        print(f"  [{status:<12}] {patch.relative_path}")
        if status in {"missing-file", "no-match"}:
            print(f"               needed for: {patch.why}", file=sys.stderr)
            failed = True
        elif status == "would-apply":
            print(f"               needed for: {patch.why}")
            failed = True  # --check: not yet applied

    if failed and args.check:
        print("\nPatches missing. Run without --check to apply.", file=sys.stderr)
        return 1
    if failed:
        print("\nSome patches could not be applied; inspect the tree.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
