#!/usr/bin/env python3
"""Build and run a minimal CUDAExtension through CUDA_INC_PATH only."""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

CPP = r'''#include <torch/extension.h>
#include <cuda_runtime_api.h>
torch::Tensor add_one_cuda(torch::Tensor input);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("add_one", &add_one_cuda); }
'''

CUDA = r'''#include <torch/extension.h>
#include <cuda_runtime.h>
__global__ void add_one_kernel(float* x, int64_t n) { int64_t i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] += 1.0f; }
torch::Tensor add_one_cuda(torch::Tensor input) { auto out = input.clone(); int64_t n = out.numel(); add_one_kernel<<<(n + 255) / 256, 256>>>(out.data_ptr<float>(), n); return out; }
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.workspace.mkdir(parents=True, exist_ok=False)
    cpp = args.workspace / "bridge.cpp"
    cuda = args.workspace / "bridge.cu"
    (args.workspace / "build").mkdir()
    cpp.write_text(CPP, encoding="utf-8")
    cuda.write_text(CUDA, encoding="utf-8")
    payload = {"success": False, "source_files": [str(cpp), str(cuda)], "extra_include_paths": []}
    try:
        import torch
        from torch.utils.cpp_extension import load

        module = load(
            name="univtac_cuda_include_bridge",
            sources=[str(cpp), str(cuda)],
            build_directory=str(args.workspace / "build"),
            verbose=True,
        )
        source = torch.arange(16, device="cuda", dtype=torch.float32)
        result = module.add_one(source)
        torch.cuda.synchronize()
        expected = source + 1
        payload.update(
            {
                "success": bool(torch.equal(result, expected)),
                "witness": result.cpu().tolist(),
                "sum": float(result.sum().item()),
                "module_realpath": str(Path(module.__file__).resolve()),
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "device_capability": list(torch.cuda.get_device_capability()),
            }
        )
    except BaseException as exc:  # noqa: BLE001 - persist native build/runtime failures
        payload.update({"error_class": type(exc).__name__, "error_message": str(exc), "traceback": traceback.format_exc()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raise SystemExit(0 if payload["success"] else 1)


if __name__ == "__main__":
    main()
