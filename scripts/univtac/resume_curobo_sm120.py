#!/usr/bin/env python3
"""Validate CUDA_INC_PATH and resume the pinned cuRobo sm120 gates."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.univtac.resume_blackwell_native_bridge import managed_cleanup_summary
from scripts.univtac.run_blackwell_native_bridge import _managed, _process_ok
from scripts.univtac.run_isaac51_runtime_gates import clean_environment, fingerprint_environment
from sim.envs.univtac.cuda_include_bridge import (
    bridge_environment,
    file_record,
    validate_include_resolution,
)
from sim.envs.univtac.curobo_build_validation import (
    classify_build_failure,
    find_extension_binaries,
    validate_extension_names,
)
from sim.envs.univtac.libuipc_core_probe import classify_core_probe
from sim.envs.univtac.native_build_contract import binary_architecture_classification
from sim.envs.univtac.resource_sanitation import utc_now, write_json


def run(command: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=False, capture_output=True, text=True)


def git(path: Path, *args: str) -> str:
    result = run(["git", *args], cwd=path)
    if result.returncode:
        raise RuntimeError(result.stderr)
    return result.stdout.strip()


def layout(prefix: Path) -> dict[str, Any]:
    roots = ["bin/nvcc", "include", "lib", "lib64", "targets/x86_64-linux/include", "targets/x86_64-linux/lib", "targets/x86_64-linux/lib64"]
    files = [
        prefix / "targets/x86_64-linux/include" / name
        for name in ("cuda_runtime_api.h", "cuda_runtime.h", "cuda.h")
    ]
    for name in ("libcudart.so", "libcudart_static.a", "libcuda.so"):
        candidates = sorted((prefix / "targets/x86_64-linux/lib").glob(name + "*"))
        files.append(candidates[0] if candidates else prefix / "targets/x86_64-linux/lib" / name)
    return {
        "prefix": str(prefix),
        "roots": {name: {"exists": (prefix / name).exists(), "realpath": str((prefix / name).resolve()), "is_symlink": (prefix / name).is_symlink()} for name in roots},
        "files": [file_record(path) for path in files],
    }


def finalize_summary(output: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    if manifest.get("classification") == "curobo_sm120_binary_audit_failed":
        manifest["gates"]["C1"] = "curobo_sm120_binary_audit_failed"
        audit_path = output / "curobo_binary_audit/audit.json"
        if audit_path.is_file():
            audit = json.loads(audit_path.read_text(encoding="utf-8"))
            missing = audit.get("missing_dependencies", [])
            if missing and all(any(name in item for name in ("libc10", "libtorch")) for item in missing):
                manifest["c1_failure_detail"] = "raw_ldd_missing_torch_shared_libraries"
            manifest["c1_canonical_audit"] = str(audit_path.resolve())
            manifest["c1_extension_record_count"] = len(audit.get("records", []))
    classification = manifest.get("classification")
    if classification in {"cuda_include_bridge_not_recognized", "minimal_cuda_extension_include_bridge_failed"}:
        kind = "include_bridge"
    elif classification in {"curobo_same_cuda_header_failure", "curobo_different_cuda_header_failure", "curobo_host_compile_failure", "curobo_nvcc_sm120_compile_failure", "curobo_link_library_layout_failure", "curobo_packaging_failure"}:
        kind = "native_build"
    elif classification and classification != "blackwell_native_bridge_viable":
        kind = "runtime"
    else:
        kind = "none"
    manifest["author_bundle_type"] = kind
    if kind != "none":
        bundle = output / "author_bundle_curobo" / kind
        bundle.mkdir(parents=True, exist_ok=True)
        write_json(bundle / "summary.json", manifest)
        (bundle / "README.md").write_text(
            "# cuRobo R0.8.2 reproduction bundle\n\n"
            "Fixed-source gate summary only; no binaries, credentials, or large logs are included.\n",
            encoding="utf-8",
        )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--r081-output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--failed-curobo-source", type=Path, required=True)
    parser.add_argument("--fresh-curobo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--conda-exe", type=Path, required=True)
    parser.add_argument("--resume-from-passed-x0", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    output = args.output_root.resolve()
    source = args.source.resolve()
    fresh = args.fresh_curobo.resolve()
    r081 = args.r081_output.resolve()
    prefix = Path(run([str(args.conda_exe), "info", "--base"], cwd=ROOT).stdout.strip()) / "envs" / cfg["environment"]["conda_name"]
    if args.resume_from_passed_x0:
        if not output.is_dir() or fresh.exists():
            raise FileNotFoundError("resume requires existing R0.8.2 output and absent fresh cuRobo")
        manifest = json.loads((output / "summary.json").read_text())
        manifest.pop("failure", None)
        manifest.update({"status": "running", "classification": None, "resumed_from_passed_x0": True})
    else:
        if output.exists() or fresh.exists():
            raise FileExistsError("R0.8.2 output and fresh cuRobo roots must be absent")
        output.mkdir(parents=True)
        manifest = {"runtime_variant": cfg["runtime_variant"], "status": "running", "classification": None, "gates": {}, "started_at": utc_now(), "isaac_started": False, "sysctl_modified": False}
    write_json(output / "run_manifest.json", manifest)
    try:
        prior = json.loads((r081 / "summary.json").read_text())
        if prior.get("libuipc_core_status") != "passed":
            raise RuntimeError("R0.8.1 libuipc evidence is not passed")
        for name in ("libuipc_core_run1", "libuipc_core_run2"):
            payload = json.loads((r081 / name / "probe.json").read_text())
            if payload.get("success") is not True:
                raise RuntimeError("R0.8.1 pure-core result is invalid")
        if git(source, "rev-parse", "HEAD") != cfg["source"]["univtac_commit"] or git(source, "status", "--short", "--untracked-files=no"):
            raise RuntimeError("UniVTAC source is not the clean fixed checkout")
        if git(args.failed_curobo_source, "rev-parse", "HEAD") != cfg["source"]["curobo_commit"] or git(args.failed_curobo_source, "status", "--short", "--untracked-files=no"):
            raise RuntimeError("R0.8.1 cuRobo source is not clean and fixed")
        write_json(output / "cuda_layout/target_layout.json", layout(prefix))
        base = clean_environment({
            "PATH": f"{prefix / 'bin'}:{os.environ.get('PATH', '')}",
            "LD_LIBRARY_PATH": str(prefix / "lib"),
            "CUDA_HOME": str(prefix),
            "CUDA_PATH": str(prefix),
            "CUDACXX": str(prefix / "bin/nvcc"),
            "TORCH_CUDA_ARCH_LIST": "12.0",
            "CMAKE_CUDA_ARCHITECTURES": "120",
            "CC": str(source / "scripts/toolchains/gcc12-system-ld"),
            "CXX": str(source / "scripts/toolchains/gxx12-system-ld"),
            "CUDAHOSTCXX": str(source / "scripts/toolchains/gxx12-system-ld"),
        })
        for key in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "CUDA_INC_PATH"):
            base.pop(key, None)
        bridge = bridge_environment(base, prefix)
        python = prefix / "bin/python"
        if not args.resume_from_passed_x0:
            for name, environment in (("before", base), ("after", bridge)):
                result = _managed([str(python), str(ROOT / "scripts/univtac/audit_cuda_extension_include_resolution.py"), "--output", str(output / f"cuda_layout/include_paths_{name}.json")], cwd=source, root=output / "cuda_layout", name=name, timeout=300, environment=environment)
                if not _process_ok(result):
                    raise RuntimeError(f"include audit {name} failed")
        before = json.loads((output / "cuda_layout/include_paths_before.json").read_text())
        after = json.loads((output / "cuda_layout/include_paths_after.json").read_text())
        try:
            validate_include_resolution(before, after, prefix)
        except ValueError as exc:
            manifest["classification"] = "cuda_include_bridge_not_recognized"
            raise RuntimeError(str(exc))
        manifest["gates"]["I0"] = "passed"

        if not args.resume_from_passed_x0:
            torch_probe = _managed([str(python), str(ROOT / "scripts/univtac/probe_blackwell_torch.py"), "--output", str(output / "preflight/torch_probe.json")], cwd=source, root=output / "preflight", name="torch", timeout=600, environment=bridge)
            pyuipc = _managed([str(python), "-c", "import uipc; print(uipc.__file__)"], cwd=source, root=output / "preflight", name="pyuipc", timeout=120, environment=bridge)
            if not _process_ok(torch_probe) or not _process_ok(pyuipc):
                manifest["classification"] = "blocked_by_external_resources"
                raise RuntimeError("short preflight failed")
            x0 = _managed([str(python), str(ROOT / "scripts/univtac/probe_minimal_cuda_extension.py"), "--workspace", str(output / "minimal_cuda_extension/workspace"), "--output", str(output / "minimal_cuda_extension/runtime_result.json")], cwd=source, root=output / "minimal_cuda_extension", name="build_and_run", timeout=cfg["timeouts_seconds"]["minimal_extension"], environment=bridge)
        else:
            x0 = json.loads((output / "minimal_cuda_extension/processes/build_and_run.json").read_text())
        xpayload = json.loads((output / "minimal_cuda_extension/runtime_result.json").read_text()) if (output / "minimal_cuda_extension/runtime_result.json").is_file() else {}
        if not _process_ok(x0) or xpayload.get("success") is not True:
            manifest["classification"] = "minimal_cuda_extension_include_bridge_failed"
            raise RuntimeError("X0 failed")
        xbins = [Path(xpayload["module_realpath"])]
        if not args.resume_from_passed_x0:
            _managed([str(python), str(ROOT / "scripts/univtac/audit_blackwell_binaries.py"), "--output", str(output / "minimal_cuda_extension/binary_audit.json"), *map(str, xbins)], cwd=source, root=output / "minimal_cuda_extension", name="audit", timeout=600, environment=bridge)
        xaudit_payload = json.loads((output / "minimal_cuda_extension/binary_audit.json").read_text())
        if not xaudit_payload.get("has_sm120_or_compute120"):
            manifest["classification"] = "minimal_cuda_extension_include_bridge_failed"
            raise RuntimeError("X0 binary architecture audit failed")
        manifest["gates"]["X0"] = "passed"

        subprocess.run(["git", "clone", "--no-local", "--no-checkout", str(args.failed_curobo_source), str(fresh)], check=True)
        subprocess.run(["git", "checkout", "--detach", cfg["source"]["curobo_commit"]], cwd=fresh, check=True)
        preinstall = {"find_spec": str(importlib.util.find_spec("curobo")), "fresh_head": git(fresh, "rev-parse", "HEAD"), "fresh_status": git(fresh, "status", "--short")}
        write_json(output / "curobo_build/preinstall_state.json", preinstall)
        build = _managed(["/usr/bin/time", "-v", str(python), "-m", "pip", "install", "--no-build-isolation", "--verbose", "-e", str(fresh)], cwd=fresh, root=output / "curobo_build", name="build", timeout=cfg["timeouts_seconds"]["curobo_build"], environment=bridge)
        blog = Path(build["log_path"]).read_text(errors="replace")
        write_json(output / "curobo_build/build_result.json", {"returncode": build.get("returncode"), "classification": "passed" if _process_ok(build) else classify_build_failure(blog), "same_header_failure_present": "cuda_runtime_api.h: No such file or directory" in blog, "sm120_gencode_present": "compute_120,code=sm_120" in blog})
        if not _process_ok(build):
            manifest["classification"] = classify_build_failure(blog)
            raise RuntimeError("C0 build failed")
        manifest["gates"]["C0_BUILD"] = "passed"
        bins = find_extension_binaries(fresh)
        validate_extension_names(str(path) for path in bins)
        audit_root = output / "curobo_binary_audit"
        audit = _managed([str(python), str(ROOT / "scripts/univtac/audit_blackwell_binaries.py"), "--output", str(audit_root / "audit.json"), *map(str, bins)], cwd=fresh, root=audit_root, name="audit", timeout=cfg["timeouts_seconds"]["binary_audit"], environment=bridge)
        apayload = json.loads((audit_root / "audit.json").read_text()) if (audit_root / "audit.json").is_file() else {"records": []}
        if not _process_ok(audit) or binary_architecture_classification(apayload["records"]) != "passed" or any(not r["has_sm120_or_compute120"] for r in apayload["records"]):
            manifest["classification"] = "curobo_sm120_binary_audit_failed"
            manifest["gates"]["C1"] = "curobo_sm120_binary_audit_failed"
            manifest["c1_failure_detail"] = "raw_ldd_missing_torch_shared_libraries" if apayload.get("missing_dependencies") else "binary_architecture_or_dependency_audit_failed"
            manifest["c1_canonical_audit"] = str((audit_root / "audit.json").resolve())
            manifest["c1_extension_record_count"] = len(apayload.get("records", []))
            raise RuntimeError("C1 failed")
        manifest["gates"]["C1"] = "passed"
        imports = _managed([str(python), str(ROOT / "scripts/univtac/probe_curobo_imports.py"), "--output", str(output / "curobo_import/result.json")], cwd=fresh, root=output / "curobo_import", name="probe", timeout=cfg["timeouts_seconds"]["import"], environment=bridge)
        if not _process_ok(imports):
            manifest["classification"] = "curobo_sm120_import_failed"
            raise RuntimeError("C2 failed")
        manifest["gates"]["C2"] = "passed"
        smoke = _managed([str(python), str(ROOT / "scripts/univtac/probe_blackwell_curobo.py"), "--source-root", str(fresh), "--output", str(output / "curobo_smoke/result.json")], cwd=fresh, root=output / "curobo_smoke", name="probe", timeout=cfg["timeouts_seconds"]["official_smoke"], environment=bridge)
        if not _process_ok(smoke):
            manifest["classification"] = "curobo_sm120_official_smoke_failed"
            raise RuntimeError("C3 failed")
        manifest["gates"]["C3"] = "passed"
        u2root = output / "libuipc_regression"
        u2 = _managed([str(python), str(ROOT / "scripts/univtac/probe_libuipc_core.py"), "--workspace", str(u2root / "workspace"), "--output", str(u2root / "probe.json"), "--events", str(u2root / "events.jsonl")], cwd=source, root=u2root, name="probe", timeout=cfg["timeouts_seconds"]["libuipc_regression"], environment=bridge)
        upayload = json.loads((u2root / "probe.json").read_text()) if (u2root / "probe.json").is_file() else {}
        if not _process_ok(u2) or classify_core_probe(upayload, timed_out=bool(u2.get("timed_out"))) != "passed":
            manifest["classification"] = "blackwell_native_bridge_partially_viable"
            raise RuntimeError("U2 failed")
        manifest["gates"]["U2"] = "passed"
        manifest["classification"] = "blackwell_native_bridge_viable"
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["classification"] = manifest["classification"] or "blocked_by_external_resources"
        manifest["failure"] = {"class": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        raise
    finally:
        for gate in ("I0", "X0", "C0_BUILD", "C1", "C2", "C3", "U2"):
            manifest["gates"].setdefault(gate, "not_run_due_to_gate")
        try:
            write_json(output / "environment_fingerprints/final.json", {name: fingerprint_environment(args.conda_exe, env) for name, env in (("legacy", "UniVTAC"), ("r07", "UniVTAC-isaac51-r07"), ("r08", cfg["environment"]["conda_name"]))})
        except Exception as exc:  # noqa: BLE001
            manifest["fingerprint_error"] = str(exc)
        write_json(output / "final_resources/source.json", {"source_head": git(source, "rev-parse", "HEAD"), "source_status": git(source, "status", "--short", "--untracked-files=no"), "fresh_curobo_head": git(fresh, "rev-parse", "HEAD") if fresh.exists() else None, "fresh_curobo_status": git(fresh, "status", "--short", "--untracked-files=no") if fresh.exists() else None})
        write_json(output / "final_resources/cleanup.json", managed_cleanup_summary(output))
        finalize_summary(output, manifest)
        manifest["ended_at"] = utc_now()
        write_json(output / "summary.json", manifest)
        write_json(output / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
