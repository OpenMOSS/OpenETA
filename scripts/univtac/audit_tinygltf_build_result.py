#!/usr/bin/env python3
"""Verify the tinygltf source and installed header used by U0R."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sim.envs.univtac.source_tree_manifest import (
    compare_manifests,
    directory_manifest,
    git_manifest,
)

OLD_INCLUDE = b'#include "json.hpp"'
NEW_INCLUDE = b"#include <nlohmann/json.hpp>"


def _record(data: bytes, *, executable: bool = False) -> dict[str, object]:
    return {
        "mode": "100755" if executable else "100644",
        "kind": "file",
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def audit_build_result(
    exact_checkout: Path,
    buildtree: Path,
    installed_header: Path,
) -> dict[str, object]:
    baseline = git_manifest(exact_checkout)
    source_header = (exact_checkout / "tiny_gltf.h").read_bytes()
    if source_header.count(OLD_INCLUDE) != 1:
        raise ValueError("exact tiny_gltf.h must contain one expected include")
    expected_header = source_header.replace(OLD_INCLUDE, NEW_INCLUDE)
    expected = dict(baseline)
    expected["tiny_gltf.h"] = _record(expected_header)
    actual = directory_manifest(buildtree, baseline)
    comparison = compare_manifests(expected, actual)
    installed = installed_header.read_bytes()
    return {
        "schema_version": "openeta.univtac.tinygltf_build_identity.v1",
        "exact_checkout": str(exact_checkout.resolve()),
        "buildtree": str(buildtree.resolve()),
        "installed_header": str(installed_header.resolve()),
        "tracked_path_count": len(baseline),
        "expected_port_replacement": {
            "from": OLD_INCLUDE.decode(),
            "to": NEW_INCLUDE.decode(),
            "replacement_count": 1,
        },
        "buildtree_matches_exact_commit_plus_port_replacement": comparison["equal"],
        "comparison": comparison,
        "installed_header_matches_expected": installed == expected_header,
        "installed_header_sha256": hashlib.sha256(installed).hexdigest(),
        "expected_header_sha256": hashlib.sha256(expected_header).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exact-checkout", type=Path, required=True)
    parser.add_argument("--buildtree", type=Path, required=True)
    parser.add_argument("--installed-header", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cmake-cache", type=Path)
    parser.add_argument("--cuda-object-root", type=Path)
    parser.add_argument("--compile-evidence-output", type=Path)
    parser.add_argument("--build-result", type=Path)
    parser.add_argument("--vcpkg-root", type=Path)
    parser.add_argument("--overlay-root", type=Path)
    parser.add_argument("--resolution-log", type=Path)
    parser.add_argument("--provenance-output", type=Path)
    args = parser.parse_args()
    payload = audit_build_result(args.exact_checkout, args.buildtree, args.installed_header)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.compile_evidence_output:
        if not args.cmake_cache or not args.cuda_object_root:
            raise ValueError("compile evidence requires CMakeCache and CUDA object root")
        cache_text = args.cmake_cache.read_text(encoding="utf-8", errors="replace")
        cuda_objects = sorted(args.cuda_object_root.rglob("*.cu.o"))
        compile_evidence = {
            "schema_version": "openeta.univtac.native_compile_evidence.v1",
            "cmake_cache": str(args.cmake_cache.resolve()),
            "cmake_cuda_compiler": next(
                (line.split("=", 1)[1] for line in cache_text.splitlines() if line.startswith("CMAKE_CUDA_COMPILER:")),
                None,
            ),
            "cmake_cuda_architectures": next(
                (line.split("=", 1)[1] for line in cache_text.splitlines() if line.startswith("CMAKE_CUDA_ARCHITECTURES:")),
                None,
            ),
            "cuda_object_count": len(cuda_objects),
            "entered_cuda_compilation": bool(cuda_objects),
        }
        args.compile_evidence_output.parent.mkdir(parents=True, exist_ok=True)
        args.compile_evidence_output.write_text(
            json.dumps(compile_evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if args.build_result:
            result = json.loads(args.build_result.read_text(encoding="utf-8"))
            result.update(
                {
                    "entered_cuda_compilation": compile_evidence["entered_cuda_compilation"],
                    "cmake_cuda_architectures": compile_evidence["cmake_cuda_architectures"],
                    "cmake_cuda_compiler": compile_evidence["cmake_cuda_compiler"],
                    "cuda_object_count": compile_evidence["cuda_object_count"],
                    "cuda_compilation_evidence": str(args.compile_evidence_output.resolve()),
                }
            )
            args.build_result.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
    if args.provenance_output:
        if not args.vcpkg_root or not args.overlay_root or not args.resolution_log:
            raise ValueError("provenance requires vcpkg root, overlay root, and resolution log")
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=args.vcpkg_root, check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--short", "--untracked-files=no"],
            cwd=args.vcpkg_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        overlay_ports = (
            sorted(path.name for path in args.overlay_root.iterdir() if path.is_dir())
            if args.overlay_root.is_dir()
            else []
        )
        resolution_lines = [
            line.strip()
            for line in args.resolution_log.read_text(encoding="utf-8", errors="replace").splitlines()
            if line.strip()
        ]
        provenance = {
            "schema_version": "openeta.univtac.vcpkg_resolution.v1",
            "fixed_vcpkg_root": str(args.vcpkg_root.resolve()),
            "fixed_vcpkg_head": head,
            "fixed_vcpkg_tracked_status": status,
            "toolchain_file": str((args.vcpkg_root / "scripts/buildsystems/vcpkg.cmake").resolve()),
            "overlay_mechanism": "VCPKG_OVERLAY_PORTS" if overlay_ports else "none",
            "overlay_root": str(args.overlay_root.resolve()) if overlay_ports else None,
            "overlay_ports": overlay_ports,
            "implicit_overlay_count": 0,
            "moving_registry_used": False,
            "resolution_log": str(args.resolution_log.resolve()),
            "resolution_lines": resolution_lines,
        }
        args.provenance_output.parent.mkdir(parents=True, exist_ok=True)
        args.provenance_output.write_text(
            json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    raise SystemExit(
        0
        if payload["buildtree_matches_exact_commit_plus_port_replacement"]
        and payload["installed_header_matches_expected"]
        else 1
    )


if __name__ == "__main__":
    main()
