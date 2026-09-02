#!/usr/bin/env python3
"""Record Isaac51 package provenance and relevant compiled binaries."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PACKAGES = (
    "torch",
    "torchvision",
    "isaacsim",
    "isaaclab",
    "tacex",
    "tacex-assets",
    "tacex-uipc",
    "pyuipc",
    "nvidia-curobo",
)
MODULES = ("torch", "isaacsim", "isaaclab", "tacex", "tacex_assets", "tacex_uipc", "uipc", "curobo")
BINARY_PACKAGES = {"torch", "tacex-uipc", "pyuipc", "nvidia-curobo"}


def _run(command: list[str], cwd: Path | None = None) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command, cwd=cwd, check=False, capture_output=True, text=True, timeout=120
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"command": command, "returncode": None, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _git_commit(path: Path) -> str | None:
    result = _run(["git", "rev-parse", "HEAD"], cwd=path)
    return result.get("stdout") if result.get("returncode") == 0 else None


def _distribution_record(name: str) -> tuple[dict[str, Any], list[Path]]:
    try:
        dist = importlib.metadata.distribution(name)
    except importlib.metadata.PackageNotFoundError:
        return {"package": name, "installed": False}, []
    files = list(dist.files or [])
    binary_paths = sorted(
        {
            Path(dist.locate_file(item)).resolve()
            for item in files
            if name in BINARY_PACKAGES and str(item).endswith(".so")
        }
    )
    direct_url = None
    direct_url_text = dist.read_text("direct_url.json")
    if direct_url_text:
        try:
            direct_url = json.loads(direct_url_text)
        except json.JSONDecodeError:
            direct_url = {"invalid_json": True}
    return (
        {
            "package": name,
            "installed": True,
            "version": dist.version,
            "root": str(Path(dist.locate_file("")).resolve()),
            "direct_url": direct_url,
        },
        binary_paths,
    )


def _binary_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "size": path.stat().st_size if path.is_file() else None,
    }
    for tool, arguments in (
        ("readelf", ["-d"]),
        ("ldd", []),
        ("cuobjdump", ["--list-elf"]),
    ):
        executable = shutil.which(tool)
        if executable is None:
            record[tool] = {"available": False}
            continue
        result = _run([executable, *arguments, str(path)])
        text = "\n".join(str(result.get(key, "")) for key in ("stdout", "stderr"))
        record[tool] = {
            "available": True,
            "returncode": result.get("returncode"),
            "mentions_cuda_12_6": "12.6" in text or "libcudart.so.12" in text,
            "architecture_tokens": sorted(
                set(part for part in text.replace("\n", " ").split() if "sm_" in part or "compute_" in part)
            )[:40],
            "output_tail": text[-4000:],
        }
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--vcpkg-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    distributions = []
    binaries: set[Path] = set()
    for package in PACKAGES:
        record, package_binaries = _distribution_record(package)
        distributions.append(record)
        binaries.update(package_binaries)
    modules = {}
    for name in MODULES:
        spec = importlib.util.find_spec(name)
        modules[name] = {
            "found": spec is not None,
            "origin": str(Path(spec.origin).resolve()) if spec and spec.origin else None,
            "search_locations": (
                sorted(str(Path(item).resolve()) for item in spec.submodule_search_locations)
                if spec and spec.submodule_search_locations
                else []
            ),
        }
    payload = {
        "schema_version": "openeta.univtac.isaac51_install_summary.v1",
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "source_root": str(args.source_root.resolve()),
        "source_commit": _git_commit(args.source_root),
        "source_status": _run(["git", "status", "--short"], cwd=args.source_root),
        "dependency_commits": {
            "vcpkg": _git_commit(args.vcpkg_root),
            "curobo": _git_commit(args.source_root / "third_party" / "curobo"),
        },
        "distributions": distributions,
        "modules": modules,
        "binaries": [_binary_record(path) for path in sorted(binaries)],
        "openeta_imported": any(name == "openeta" or name.startswith("openeta.") for name in sys.modules),
        "credential_environment_recorded": False,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
