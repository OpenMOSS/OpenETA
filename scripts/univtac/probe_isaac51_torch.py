#!/usr/bin/env python3
"""Probe Torch/CUDA and optional Isaac51 imports without starting Isaac Sim."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import platform
import sys
import traceback
from pathlib import Path
from typing import Any


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _module_path(module: Any) -> str | None:
    path = getattr(module, "__file__", None)
    return str(Path(path).resolve()) if path else None


def required_architecture(capability: tuple[int, int] | list[int]) -> str:
    return f"sm_{int(capability[0])}{int(capability[1])}"


def run_probe(imports: list[str]) -> dict[str, Any]:
    import torch

    payload: dict[str, Any] = {
        "schema_version": "openeta.univtac.isaac51_torch_probe.v1",
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torchvision_version": _version("torchvision"),
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "imports": {},
        "openeta_imported": any(name == "openeta" or name.startswith("openeta.") for name in sys.modules),
    }
    if payload["cuda_available"]:
        device = torch.device("cuda:0")
        capability = list(torch.cuda.get_device_capability(device))
        supported_architectures = torch.cuda.get_arch_list()
        required_arch = required_architecture(capability)
        payload.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(device),
                "cuda_device_capability": capability,
                "cuda_current_device": torch.cuda.current_device(),
                "torch_supported_cuda_architectures": supported_architectures,
                "required_cuda_architecture": required_arch,
            }
        )
        if required_arch not in supported_architectures:
            payload.update(
                {
                    "classification": "unsupported_gpu_architecture",
                    "cuda_probe_attempted": False,
                    "success": False,
                }
            )
            return payload
        left = torch.arange(16, dtype=torch.float32, device=device).reshape(4, 4)
        right = torch.eye(4, dtype=torch.float32, device=device)
        product = left @ right
        torch.cuda.synchronize(device)
        payload.update(
            {
                "classification": "cuda_probe_passed",
                "cuda_probe_attempted": True,
                "cuda_probe_sum": float(product.sum().item()),
            }
        )
    for name in imports:
        try:
            module = importlib.import_module(name)
        except BaseException as exc:
            payload["imports"][name] = {
                "ok": False,
                "error_class": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        else:
            payload["imports"][name] = {"ok": True, "realpath": _module_path(module)}
    payload["success"] = bool(
        payload["cuda_available"]
        and payload["torch_cuda_version"] == "12.6"
        and all(item["ok"] for item in payload["imports"].values())
        and not payload["openeta_imported"]
    )
    if not payload["success"] and "classification" not in payload:
        payload["classification"] = "runtime_or_import_failure"
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--import", dest="imports", action="append", default=[])
    args = parser.parse_args()
    try:
        payload = run_probe(args.imports)
    except BaseException as exc:
        payload = {
            "schema_version": "openeta.univtac.isaac51_torch_probe.v1",
            "success": False,
            "error_class": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True, ensure_ascii=False))
    raise SystemExit(0 if payload.get("success") else 1)


if __name__ == "__main__":
    main()
