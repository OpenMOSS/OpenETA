#!/usr/bin/env python3
"""Observe PyTorch-context loader closure without changing loader behavior."""

from __future__ import annotations

import argparse
import ctypes
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HELPER = ROOT / "sim/envs/univtac/elf_loader_closure.py"
helper_spec = importlib.util.spec_from_file_location("univtac_elf_loader_closure", HELPER)
if helper_spec is None or helper_spec.loader is None:
    raise ImportError(f"cannot load loader helper: {HELPER}")
helper = importlib.util.module_from_spec(helper_spec)
helper_spec.loader.exec_module(helper)
loaded_objects = helper.loaded_objects
legacy_paths = helper.legacy_paths
parse_proc_maps = helper.parse_proc_maps
resolve_needed = helper.resolve_needed
ALLOWED_TORCH_SONAMES = helper.ALLOWED_TORCH_SONAMES


class DlPhdrInfo(ctypes.Structure):
    _fields_ = [("dlpi_addr", ctypes.c_void_p), ("dlpi_name", ctypes.c_char_p), ("dlpi_phdr", ctypes.c_void_p), ("dlpi_phnum", ctypes.c_ushort)]


def phdr_objects() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    callback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(DlPhdrInfo), ctypes.c_size_t, ctypes.c_void_p)

    @callback_type
    def callback(info, _size, _data):
        name = info.contents.dlpi_name.decode(errors="replace") if info.contents.dlpi_name else ""
        records.append({"path": name, "address": int(info.contents.dlpi_addr or 0), "phnum": int(info.contents.dlpi_phnum)})
        return 0

    process = ctypes.CDLL(None)
    process.dl_iterate_phdr.argtypes = [callback_type, ctypes.c_void_p]
    process.dl_iterate_phdr.restype = ctypes.c_int
    process.dl_iterate_phdr(callback, None)
    return records


def snapshot() -> dict[str, object]:
    maps = parse_proc_maps(Path("/proc/self/maps").read_text(encoding="utf-8"))
    return {"maps": maps, "objects": loaded_objects(maps), "phdr": phdr_objects(), "dlopen_flags": sys.getdlopenflags()}


def needed(path: Path) -> list[str]:
    result = subprocess.run(["readelf", "-d", str(path)], check=True, capture_output=True, text=True)
    return [line.split("[", 1)[1].split("]", 1)[0] for line in result.stdout.splitlines() if "(NEEDED)" in line]


def sonames(paths: list[str]) -> dict[str, str]:
    records = {}
    for path in paths:
        result = subprocess.run(["readelf", "-d", path], check=False, capture_output=True, text=True)
        line = next((item for item in result.stdout.splitlines() if "(SONAME)" in item), None)
        if line:
            records[path] = line.split("[", 1)[1].split("]", 1)[0]
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("torch", "extension", "x0"), required=True)
    parser.add_argument("--module")
    parser.add_argument("--binary", type=Path)
    parser.add_argument("--expected-root", type=Path)
    parser.add_argument("--target-prefix", type=Path, required=True)
    parser.add_argument("--forbidden-root", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload: dict[str, object] = {"success": False, "mode": args.mode, "stages": {"initial": snapshot()}, "environment": {key: os.environ.get(key) for key in ("LD_LIBRARY_PATH", "LD_PRELOAD", "TORCH_USE_RTLD_GLOBAL")}}
    try:
        import torch

        torch.arange(4, device="cuda").sum()
        torch.cuda.synchronize()
        payload["stages"]["after_torch"] = snapshot()
        payload["torch"] = {"version": torch.__version__, "cuda": torch.version.cuda, "capability": list(torch.cuda.get_device_capability())}
        torch_objects = payload["stages"]["after_torch"]["objects"]
        torch_lib = args.target_prefix.resolve() / "lib/python3.11/site-packages/torch/lib"
        required_torch = {"libtorch_global_deps.so", *ALLOWED_TORCH_SONAMES}
        present_torch = {Path(path).name for path in torch_objects if str(path).startswith(str(torch_lib) + "/")}
        payload["torch_loader_provenance"] = {"expected_root": str(torch_lib), "required": sorted(required_torch), "present": sorted(present_torch & required_torch), "missing": sorted(required_torch - present_torch), "system_cuda_paths": sorted(path for path in torch_objects if path.startswith("/usr/local/cuda"))}
        if args.mode != "torch":
            if not args.binary:
                raise ValueError("extension modes require --binary")
            started = time.monotonic()
            if args.mode == "x0":
                spec = importlib.util.spec_from_file_location("univtac_cuda_include_bridge", args.binary)
                if spec is None or spec.loader is None:
                    raise ImportError("cannot create X0 extension spec")
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                source = torch.arange(16, device="cuda", dtype=torch.float32)
                result = module.add_one(source)
                torch.cuda.synchronize()
                payload["witness"] = result.cpu().tolist()
                payload["witness_sum"] = float(result.sum().item())
            else:
                import curobo

                payload["curobo_realpath"] = str(Path(curobo.__file__).resolve())
                module = importlib.import_module(args.module)
            module_path = Path(module.__file__).resolve()
            torch.cuda.synchronize()
            payload["module"] = {"name": args.module, "realpath": str(module_path), "latency_seconds": time.monotonic() - started}
            payload["stages"]["after_extension"] = snapshot()
            dependencies = needed(module_path)
            objects = payload["stages"]["after_extension"]["objects"]
            object_sonames = sonames(objects)
            closure = resolve_needed(dependencies, objects, object_sonames)
            payload["needed"] = dependencies
            payload["loaded_sonames"] = object_sonames
            payload["closure"] = closure
            by_soname: dict[str, list[str]] = {}
            for path, soname in object_sonames.items():
                by_soname.setdefault(soname, []).append(path)
            payload["duplicate_sonames"] = {name: sorted(set(paths)) for name, paths in by_soname.items() if len(set(paths)) > 1}
            payload["legacy_paths"] = legacy_paths(objects, args.forbidden_root)
            payload["torch_dependency_provenance_ok"] = all(edge["soname"] not in ALLOWED_TORCH_SONAMES or all(path.startswith(str(torch_lib) + "/") for path in edge["realpaths"]) for edge in closure)
            payload["success"] = bool(all(edge["status"] == "resolved" for edge in closure) and payload["torch_dependency_provenance_ok"] and not payload["legacy_paths"])
            if args.expected_root and not str(module_path).startswith(str(args.expected_root.resolve()) + "/"):
                payload["success"] = False
                payload["provenance_error"] = "module outside expected root"
        else:
            payload["success"] = not payload["torch_loader_provenance"]["missing"] and not payload["torch_loader_provenance"]["system_cuda_paths"]
    except BaseException as exc:  # noqa: BLE001
        payload.update({"error_class": type(exc).__name__, "error_message": str(exc), "traceback": traceback.format_exc()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raise SystemExit(0 if payload["success"] else 1)


if __name__ == "__main__":
    main()
