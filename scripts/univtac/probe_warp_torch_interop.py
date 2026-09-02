#!/usr/bin/env python3
"""Run a minimal Warp sm120 kernel through the public Torch interop API."""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "sim/envs/univtac/warp_api_compatibility.py"
helper_spec = importlib.util.spec_from_file_location("univtac_warp_api_compatibility", HELPER)
if helper_spec is None or helper_spec.loader is None:
    raise ImportError(f"cannot load Warp API helper: {HELPER}")
helper = importlib.util.module_from_spec(helper_spec)
helper_spec.loader.exec_module(helper)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {"success": False, "used_private_warp_torch": False}
    try:
        import torch
        import warp as wp

        wp.init()

        @wp.kernel
        def add_one(source: wp.array(dtype=wp.float32), target: wp.array(dtype=wp.float32)):
            index = wp.tid()
            target[index] = source[index] + 1.0

        device = wp.device_from_torch(torch.device("cuda:0"))
        stream = wp.stream_from_torch(torch.cuda.current_stream())
        source = torch.arange(16, device="cuda", dtype=torch.float32)
        target = torch.empty_like(source)
        wp_source = wp.from_torch(source)
        wp_target = wp.from_torch(target)
        started = time.monotonic()
        wp.launch(add_one, dim=source.numel(), inputs=[wp_source, wp_target], device=device, stream=stream)
        torch.cuda.synchronize()
        first_latency = time.monotonic() - started
        first = target.cpu().tolist()
        source.add_(1)
        started = time.monotonic()
        wp.launch(add_one, dim=source.numel(), inputs=[wp_source, wp_target], device=device, stream=stream)
        torch.cuda.synchronize()
        second_latency = time.monotonic() - started
        second = target.cpu().tolist()
        payload.update(
            {
                "warp_version": getattr(wp, "__version__", None),
                "warp_device": str(device),
                "warp_stream_device": str(stream.device),
                "kernel_cache_path": str(getattr(wp.config, "kernel_cache_dir", "")),
                "device_capability": list(torch.cuda.get_device_capability()),
                "input_witness": list(range(16)),
                "first_output_witness": first,
                "second_output_witness": second,
                "pointer_identity": {
                    "input": int(wp_source.ptr) == source.data_ptr(),
                    "output": int(wp_target.ptr) == target.data_ptr(),
                },
                "first_jit_latency_seconds": first_latency,
                "second_execution_latency_seconds": second_latency,
            }
        )
        helper.validate_warp_kernel_witness(payload)
        payload["success"] = True
        payload["classification"] = "passed"
    except BaseException as exc:  # noqa: BLE001
        payload.update(
            {
                "classification": "warp_sm120_public_api_kernel_failed",
                "error_class": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raise SystemExit(0 if payload["success"] else 1)


if __name__ == "__main__":
    main()
