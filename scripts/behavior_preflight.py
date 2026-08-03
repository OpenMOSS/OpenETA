#!/usr/bin/env python3
"""Reproducible BEHAVIOR-1K / OmniGibson installation diagnostics.

This script deliberately avoids importing OmniGibson in the caller process:
importing it launches Isaac Kit.  Package discovery and CUDA checks run in the
isolated BEHAVIOR Python and return a machine-readable report.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
BEHAVIOR_ROOT = REPO_ROOT / "sim" / "venvs" / "behavior"
RUNTIME_PYTHON = BEHAVIOR_ROOT / "runtime" / "bin" / "python"
SOURCE_ROOT = BEHAVIOR_ROOT / "src" / "BEHAVIOR-1K"
DATASET_ROOT = SOURCE_ROOT / "datasets"
MANIFEST = REPO_ROOT / "sim" / "envs" / "behavior" / "benchmark_versions.json"


def _run(argv: list[str], timeout: int = 30) -> dict[str, Any]:
    try:
        result = subprocess.run(argv, text=True, capture_output=True, timeout=timeout, check=False)
        return {
            "ok": result.returncode == 0,
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _memory_gib() -> float:
    pages = os.sysconf("SC_PHYS_PAGES")
    page_size = os.sysconf("SC_PAGE_SIZE")
    return round(pages * page_size / 1024**3, 1)


def _version_tuple(value: str) -> tuple[int, ...]:
    parts: list[int] = []
    for part in value.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            break
    return tuple(parts)


def collect_preflight() -> dict[str, Any]:
    runtime_exists = RUNTIME_PYTHON.is_file()
    source_commit = _run(["git", "-C", str(SOURCE_ROOT), "rev-parse", "HEAD"]) if SOURCE_ROOT.is_dir() else None
    gpu = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,driver_version,memory.total,memory.free",
            "--format=csv,noheader,nounits",
        ]
    )
    packages = None
    cuda = None
    if runtime_exists:
        packages = _run(
            [
                str(RUNTIME_PYTHON),
                "-c",
                (
                    "import importlib.util,json;"
                    "names=['torch','isaacsim','omnigibson','bddl','gymnasium'];"
                    "print(json.dumps({n:bool(importlib.util.find_spec(n)) for n in names}))"
                ),
            ]
        )
        cuda = _run(
            [
                str(RUNTIME_PYTHON),
                "-c",
                (
                    "import json,torch;print(json.dumps({"
                    "'torch':torch.__version__,'cuda_runtime':torch.version.cuda,"
                    "'cuda_available':torch.cuda.is_available(),"
                    "'device_count':torch.cuda.device_count(),"
                    "'device_name':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}))"
                ),
            ]
        )

    assets: dict[str, Any] = {}
    for name in ("omnigibson-robot-assets", "behavior-1k-assets", "2026-challenge-task-instances"):
        path = DATASET_ROOT / name
        size_probe = _run(["du", "-sb", str(path)], timeout=120) if path.exists() else None
        size_bytes = None
        if size_probe and size_probe.get("ok"):
            try:
                size_bytes = int(str(size_probe["stdout"]).split()[0])
            except (ValueError, IndexError):
                pass
        assets[name] = {
            "exists": path.exists(),
            "path": str(path),
            "version": (path / "VERSION").read_text().strip() if (path / "VERSION").is_file() else None,
            "size_gib": round(size_bytes / 1024**3, 2) if size_bytes is not None else None,
        }

    disk = shutil.disk_usage(BEHAVIOR_ROOT)
    blockers: list[str] = []
    warnings: list[str] = []
    if not runtime_exists:
        blockers.append("isolated Python runtime is missing")
    if not source_commit or not source_commit.get("ok"):
        blockers.append("pinned BEHAVIOR-1K source checkout is missing")
    if not gpu.get("ok"):
        blockers.append("nvidia-smi cannot access an NVIDIA GPU in this execution context")
    else:
        try:
            first_gpu = str(gpu["stdout"]).splitlines()[0]
            driver = [part.strip() for part in first_gpu.split(",")][2]
            if _version_tuple(driver) < _version_tuple("580.65.06"):
                warnings.append(
                    f"NVIDIA driver {driver} is below Isaac Sim 5.1's tested Linux version 580.65.06"
                )
        except (KeyError, IndexError):
            blockers.append("could not parse NVIDIA driver version")
    if not packages or not packages.get("ok"):
        blockers.append("runtime package discovery failed")
    else:
        try:
            package_flags = json.loads(packages["stdout"])
            for name, installed in package_flags.items():
                if not installed:
                    blockers.append(f"runtime package is missing: {name}")
        except (KeyError, json.JSONDecodeError):
            blockers.append("runtime package discovery returned invalid JSON")
    if not cuda or not cuda.get("ok"):
        blockers.append("Torch CUDA probe failed")
    elif '"cuda_available": true' not in str(cuda.get("stdout", "")).lower():
        blockers.append("Torch cannot initialize CUDA in this execution context")
    for name, item in assets.items():
        if not item["exists"]:
            blockers.append(f"asset directory is missing: {name}")

    report = {
        "ok": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "requirements": {
            "official_release": "BEHAVIOR-1K v3.9.0",
            "python": "3.11",
            "isaac_sim": "5.1.0",
            "behavior_minimum": {
                "os": "Ubuntu 22.04+",
                "ram_gib": 32,
                "gpu": "NVIDIA RTX 2070",
                "vram_gib": 8,
            },
            "isaac_sim_5_1_tested_minimum": {
                "os": "Ubuntu 22.04/24.04",
                "ram_gib": 32,
                "gpu": "GeForce RTX 4080",
                "vram_gib": 16,
                "linux_driver": "580.65.06",
            },
        },
        "host": {
            "platform": platform.platform(),
            "ram_gib": _memory_gib(),
            "nvidia_device_nodes": {
                path: Path(path).exists()
                for path in ("/dev/nvidiactl", "/dev/nvidia0", "/dev/nvidia-uvm")
            },
            "nvidia_smi": gpu,
            "disk_free_gib": round(disk.free / 1024**3, 1),
        },
        "install": {
            "runtime_python": str(RUNTIME_PYTHON),
            "runtime_exists": runtime_exists,
            "source_root": str(SOURCE_ROOT),
            "source_commit": source_commit,
            "manifest": json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else None,
            "packages": packages,
            "torch_cuda": cuda,
            "assets": assets,
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, help="also write the JSON report here")
    args = parser.parse_args()
    report = collect_preflight()
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
