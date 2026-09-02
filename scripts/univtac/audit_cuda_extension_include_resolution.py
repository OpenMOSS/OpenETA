#!/usr/bin/env python3
"""Record PyTorch CUDAExtension include and library resolution."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from torch.utils.cpp_extension import CUDA_HOME, include_paths, library_paths

    payload = {
        "cuda_home": CUDA_HOME,
        "cudacxx": os.environ.get("CUDACXX"),
        "cuda_inc_path": os.environ.get("CUDA_INC_PATH"),
        "include_paths": include_paths("cuda"),
        "library_paths": library_paths("cuda"),
        "forbidden_include_environment": {
            key: os.environ.get(key) for key in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH")
        },
        "cpp_extension_realpath": str(Path(__import__("torch.utils.cpp_extension", fromlist=["x"]).__file__).resolve()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
