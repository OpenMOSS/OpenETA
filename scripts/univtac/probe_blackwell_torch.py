#!/usr/bin/env python3
"""Run two rounds of real CUDA kernels for the pinned Blackwell bridge."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable


def _timed(operation: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter()
    value = operation()
    return value, time.perf_counter() - started


def _round(torch, index: int) -> dict[str, Any]:
    import torch.nn.functional as functional

    torch.manual_seed(1729 + index)
    torch.cuda.manual_seed_all(1729 + index)
    operations: dict[str, Any] = {}
    started = time.perf_counter()
    tensor, latency = _timed(lambda: torch.empty((1024, 1024), dtype=torch.float32, device="cuda"))
    operations["allocation"] = latency
    values, latency = _timed(lambda: torch.arange(4096, dtype=torch.float32, device="cuda"))
    operations["arange"] = latency
    sorted_values, latency = _timed(lambda: torch.sort(values.flip(0)).values)
    operations["sort"] = latency
    reduced, latency = _timed(lambda: sorted_values.sum())
    operations["reduction"] = latency
    matrix, latency = _timed(lambda: torch.eye(256, device="cuda") @ torch.ones((256, 256), device="cuda"))
    operations["matrix_multiplication"] = latency
    image = torch.ones((1, 1, 32, 32), device="cuda")
    kernel = torch.ones((1, 1, 3, 3), device="cuda")
    convolution, latency = _timed(lambda: functional.conv2d(image, kernel))
    operations["convolution"] = latency
    generator = torch.Generator(device="cuda").manual_seed(2718 + index)
    random_values, latency = _timed(lambda: torch.rand((2048,), generator=generator, device="cuda"))
    operations["random_generator"] = latency
    host_values, latency = _timed(lambda: random_values.cpu())
    operations["device_to_host_copy"] = latency
    _, latency = _timed(torch.cuda.synchronize)
    operations["synchronize"] = latency
    del tensor
    return {
        "round": index,
        "success": True,
        "operations": operations,
        "elapsed_seconds": time.perf_counter() - started,
        "witness": {
            "reduction": float(reduced.item()),
            "matrix_sum": float(matrix.sum().item()),
            "convolution_sum": float(convolution.sum().item()),
            "host_random_count": int(host_values.numel()),
        },
    }


def run_probe() -> dict[str, Any]:
    import torch

    payload: dict[str, Any] = {
        "schema_version": "openeta.univtac.blackwell_torch_probe.v1",
        "runtime_variant": "blackwell_compat_adaptation_v1",
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "device_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "torch_arch_list": torch.cuda.get_arch_list() if torch.cuda.is_available() else [],
        "rounds": [],
    }
    if not payload["cuda_available"] or payload["torch_cuda_version"] != "12.8":
        payload["classification"] = "torch_cu128_sm120_failed"
        payload["success"] = False
        return payload
    for index in (1, 2):
        try:
            payload["rounds"].append(_round(torch, index))
        except BaseException as exc:
            payload["rounds"].append(
                {
                    "round": index,
                    "success": False,
                    "error_class": type(exc).__name__,
                    "error_message": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            break
    payload["success"] = len(payload["rounds"]) == 2 and all(item["success"] for item in payload["rounds"])
    payload["classification"] = "passed" if payload["success"] else "torch_cu128_sm120_failed"
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        payload = run_probe()
    except BaseException as exc:
        payload = {
            "schema_version": "openeta.univtac.blackwell_torch_probe.v1",
            "runtime_variant": "blackwell_compat_adaptation_v1",
            "classification": "torch_cu128_sm120_failed",
            "success": False,
            "error_class": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))
    raise SystemExit(0 if payload.get("success") else 1)


if __name__ == "__main__":
    main()
