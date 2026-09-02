#!/usr/bin/env python3
"""Audit the installed Warp public Torch interop API without modifying it."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "sim/envs/univtac/warp_api_compatibility.py"
helper_spec = importlib.util.spec_from_file_location("univtac_warp_api_compatibility", HELPER)
if helper_spec is None or helper_spec.loader is None:
    raise ImportError(f"cannot load Warp API helper: {HELPER}")
helper = importlib.util.module_from_spec(helper_spec)
helper_spec.loader.exec_module(helper)
PUBLIC_APIS = helper.PUBLIC_APIS
classify_warp_api = helper.classify_warp_api


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload: dict[str, object] = {"success": False, "python_executable": str(Path(sys.executable).resolve())}
    try:
        import torch
        import warp as wp

        distribution = importlib.metadata.distribution("warp-lang")
        record = (Path(distribution._path) / "RECORD").resolve()
        native_libraries = []
        for name in ("warp.so", "warp-clang.so"):
            path = Path(wp.__file__).resolve().parent / "bin" / name
            native_libraries.append({"path": str(path), "size": path.stat().st_size, "sha256": sha256(path)})
        matrix = {name: hasattr(wp, name) for name in PUBLIC_APIS}
        matrix["torch"] = hasattr(wp, "torch")
        private_spec = importlib.util.find_spec("warp.torch")
        private_import = subprocess.run(
            [sys.executable, "-c", "import warp.torch"], capture_output=True, text=True, check=False
        )
        wp.init()
        device = wp.device_from_torch(torch.device("cuda:0"))
        stream = wp.stream_from_torch(torch.cuda.current_stream())
        tensor = torch.arange(4, device="cuda", dtype=torch.float32)
        array = wp.from_torch(tensor)
        torch.cuda.synchronize()
        payload.update(
            {
                "distribution": {
                    "name": distribution.metadata["Name"],
                    "version": distribution.version,
                    "metadata_path": str(Path(distribution._path).resolve()),
                    "file_count": len(list(distribution.files or [])),
                    "record_path": str(record),
                    "record_sha256": sha256(record),
                },
                "warp": {
                    "realpath": str(Path(wp.__file__).resolve()),
                    "version": getattr(wp, "__version__", None),
                    "config_version": getattr(wp.config, "version", None),
                    "native_libraries": native_libraries,
                },
                "target_prefix": str(args.target_prefix.resolve()),
                "api_matrix": matrix,
                "api_signatures": {name: str(inspect.signature(getattr(wp, name))) for name in PUBLIC_APIS},
                "private_module": {
                    "find_spec": None if private_spec is None else str(private_spec.origin),
                    "import_returncode": private_import.returncode,
                    "import_error": private_import.stderr.strip(),
                },
                "interop": {
                    "device": str(device),
                    "stream_device": str(stream.device),
                    "array_device": str(array.device),
                    "dtype": str(array.dtype),
                    "pointer_identity": int(array.ptr) == tensor.data_ptr(),
                },
                "interop_success": int(array.ptr) == tensor.data_ptr(),
            }
        )
        payload["classification"] = classify_warp_api(payload)
        payload["success"] = payload["classification"] == "passed"
    except BaseException as exc:  # noqa: BLE001 - persist native/API failures
        payload.update(
            {
                "classification": "warp_version_provenance_unresolved",
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
