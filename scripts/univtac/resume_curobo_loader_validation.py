#!/usr/bin/env python3
"""Run loader-aware cuRobo sm120 validation without rebuilding or relinking."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.univtac.resume_blackwell_native_bridge import managed_cleanup_summary
from scripts.univtac.run_blackwell_native_bridge import _managed, _process_ok
from scripts.univtac.run_isaac51_runtime_gates import (
    clean_environment,
    compare_fingerprints,
    fingerprint_environment,
)
from sim.envs.univtac.cuda_include_bridge import bridge_environment
from sim.envs.univtac.curobo_build_validation import EXTENSIONS, find_extension_binaries
from sim.envs.univtac.curobo_runtime_validation import (
    apply_environment_gate,
    classify_example_failure,
    parse_ik_metrics,
)
from sim.envs.univtac.elf_loader_closure import unexpected_static_missing
from sim.envs.univtac.libuipc_core_probe import classify_core_probe
from sim.envs.univtac.resource_sanitation import utc_now, write_json


def git(path: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=path, check=True, capture_output=True, text=True)
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--r082-output", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--curobo", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--conda-exe", type=Path, required=True)
    parser.add_argument("--resume-from-l0", action="store_true")
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    output, source, curobo = args.output_root.resolve(), args.source.resolve(), args.curobo.resolve()
    if args.resume_from_l0:
        if not output.is_dir():
            raise FileNotFoundError("resume requires existing R0.8.3 output")
    else:
        if output.exists():
            raise FileExistsError("R0.8.3 output must be absent")
        output.mkdir(parents=True)
    manifest = {"runtime_variant": cfg["runtime_variant"], "diagnostic_method": cfg["diagnostic_method"], "status": "running", "classification": None, "gates": {}, "started_at": utc_now(), "environment_modified": False, "isaac_started": False}
    write_json(output / "run_manifest.json", manifest)
    before = {}
    try:
        if args.resume_from_l0:
            manifest.update(json.loads((output / "summary.json").read_text()))
            manifest.pop("failure", None)
            manifest.update({"status": "running", "classification": None, "resumed_from_l0": True})
        conda_base = subprocess.run(
            [str(args.conda_exe), "info", "--base"], check=True, capture_output=True, text=True
        ).stdout.strip()
        prefix = Path(conda_base) / "envs" / cfg["environment"]["conda_name"]
        prior = json.loads((args.r082_output / "summary.json").read_text())
        if prior.get("gates", {}).get("C0_BUILD") != "passed":
            raise RuntimeError("R0.8.2 cuRobo build evidence is not passed")
        if git(curobo, "rev-parse", "HEAD") != cfg["source"]["curobo_commit"] or git(curobo, "status", "--short", "--untracked-files=no"):
            raise RuntimeError("fixed cuRobo checkout is not clean")
        binaries = find_extension_binaries(curobo)
        previous = {Path(item["path"]).name: item for item in json.loads((args.r082_output / "curobo_binary_audit/audit.json").read_text())["records"]}
        if {path.name for path in binaries} != set(previous):
            raise RuntimeError("cuRobo binary identity differs from R0.8.2")
        if args.resume_from_l0:
            before = json.loads((output / "environment_fingerprints/before.json").read_text())
        else:
            before = {name: fingerprint_environment(args.conda_exe, env) for name, env in (("legacy", "UniVTAC"), ("r07", "UniVTAC-isaac51-r07"), ("r08", cfg["environment"]["conda_name"]))}
            write_json(output / "environment_fingerprints/before.json", before)
        environment = bridge_environment(clean_environment({"PATH": f"{prefix / 'bin'}:{os.environ.get('PATH', '')}", "LD_LIBRARY_PATH": str(prefix / "lib"), "CUDA_LAUNCH_BLOCKING": "1"}), prefix)
        for key in ("LD_PRELOAD", "TORCH_USE_RTLD_GLOBAL"):
            environment.pop(key, None)
        python = prefix / "bin/python"
        provenance_args = ["--target-prefix", str(prefix), "--forbidden-root", str(prefix.parent / "UniVTAC"), "--forbidden-root", str(prefix.parent / "UniVTAC-isaac51-r07")]

        if not args.resume_from_l0:
            _managed([str(python), str(ROOT / "scripts/univtac/audit_blackwell_binaries.py"), "--output", str(output / "static_elf/audit.json"), *map(str, binaries)], cwd=curobo, root=output / "static_elf", name="audit", timeout=300, environment=environment)
        static_payload = json.loads((output / "static_elf/audit.json").read_text())
        current = {Path(item["path"]).name: item for item in static_payload["records"]}
        if any(current[name]["sha256"] != previous[name]["sha256"] for name in current):
            manifest["classification"] = "blocked_by_external_resources"
            raise RuntimeError("cuRobo binary hash differs from R0.8.2")
        if unexpected_static_missing(item.split(" =>", 1)[0] for item in static_payload["missing_dependencies"]):
            manifest["classification"] = "unexpected_non_torch_static_dependency_missing"
            raise RuntimeError("L0 found non-Torch missing dependencies")
        manifest["gates"]["L0"] = "standalone_ldd_context_unresolved"

        x0_info = json.loads((args.r082_output / "minimal_cuda_extension/runtime_result.json").read_text())
        x0 = _managed([str(python), str(ROOT / "scripts/univtac/audit_pytorch_extension_loader.py"), "--mode", "x0", "--binary", x0_info["module_realpath"], *provenance_args, "--output", str(output / "x0_loader_control/result.json")], cwd=source, root=output / "x0_loader_control", name="probe", timeout=cfg["timeouts_seconds"]["loader_probe"], environment=environment)
        if not _process_ok(x0):
            manifest["classification"] = "loader_audit_positive_control_failed"
            raise RuntimeError("L1 failed")
        manifest["gates"]["L1"] = "passed"
        baseline = _managed([str(python), str(ROOT / "scripts/univtac/audit_pytorch_extension_loader.py"), "--mode", "torch", *provenance_args, "--output", str(output / "torch_loader_baseline/result.json")], cwd=curobo, root=output / "torch_loader_baseline", name="probe", timeout=cfg["timeouts_seconds"]["loader_probe"], environment=environment)
        if not _process_ok(baseline):
            manifest["classification"] = "torch_runtime_library_provenance_failed"
            raise RuntimeError("L2 failed")
        manifest["gates"]["L2"] = "passed"
        for name, binary in zip(EXTENSIONS, binaries, strict=True):
            root = output / "extension_loader" / name
            probe = _managed([str(python), str(ROOT / "scripts/univtac/audit_pytorch_extension_loader.py"), "--mode", "extension", "--module", f"curobo.curobolib.{name}", "--binary", str(binary), "--expected-root", str(curobo), *provenance_args, "--output", str(root / "result.json")], cwd=curobo, root=root, name="probe", timeout=cfg["timeouts_seconds"]["loader_probe"], environment=environment)
            if not _process_ok(probe):
                manifest["classification"] = "curobo_extension_loader_resolution_failed"
                raise RuntimeError(f"L3 failed for {name}")
        manifest["gates"]["L3"] = "runtime_resolved_after_torch_import"
        combined = _managed(
            [
                str(python),
                str(ROOT / "scripts/univtac/probe_curobo_imports.py"),
                "--expected-root",
                str(curobo),
                "--forbidden-root",
                str(prefix.parent / "UniVTAC"),
                "--forbidden-root",
                str(prefix.parent / "UniVTAC-isaac51-r07"),
                "--output",
                str(output / "combined_import/result.json"),
            ],
            cwd=curobo,
            root=output / "combined_import",
            name="probe",
            timeout=cfg["timeouts_seconds"]["combined_import"],
            environment=environment,
        )
        if not _process_ok(combined):
            manifest["classification"] = "curobo_combined_import_failed"
            raise RuntimeError("combined import failed")
        manifest["gates"]["C2"] = "passed"
        examples = (
            ("C3A", "kinematics", "kinematics_example.py", ("kinematics_fused_cu",)),
            ("C3B", "ik", "ik_example.py", EXTENSIONS),
            ("C3C", "collision", "collision_check_example.py", ("geom_cu",)),
        )
        for gate, name, filename, expected_extensions in examples:
            root = output / f"{name}_smoke"
            proc = _managed(
                [
                    str(python),
                    str(ROOT / "scripts/univtac/run_official_curobo_example.py"),
                    "--example",
                    str(curobo / "examples" / filename),
                    "--expected-root",
                    str(curobo),
                    *[item for extension in expected_extensions for item in ("--expected-extension", extension)],
                    "--output",
                    str(root / "result.json"),
                ],
                cwd=curobo,
                root=root,
                name="probe",
                timeout=cfg["timeouts_seconds"][name],
                environment=environment,
            )
            if not _process_ok(proc):
                result = json.loads((root / "result.json").read_text()) if (root / "result.json").is_file() else {}
                message = str(result.get("error_message", ""))
                manifest["classification"] = classify_example_failure(message) if name == "collision" else f"curobo_{name}_smoke_failed"
                manifest["gates"][gate] = manifest["classification"]
                raise RuntimeError(f"{gate} failed")
            if name == "ik":
                write_json(root / "metrics.json", {"runs": parse_ik_metrics(Path(proc["log_path"]).read_text(errors="replace"))})
            manifest["gates"][gate] = "passed"
        u2root = output / "libuipc_regression"
        u2 = _managed([str(python), str(ROOT / "scripts/univtac/probe_libuipc_core.py"), "--workspace", str(u2root / "workspace"), "--output", str(u2root / "probe.json"), "--events", str(u2root / "events.jsonl")], cwd=source, root=u2root, name="probe", timeout=cfg["timeouts_seconds"]["libuipc_regression"], environment=environment)
        upayload = json.loads((u2root / "probe.json").read_text()) if (u2root / "probe.json").is_file() else {}
        if not _process_ok(u2) or classify_core_probe(upayload, timed_out=bool(u2.get("timed_out"))) != "passed":
            manifest["classification"] = "libuipc_regression_failed"
            raise RuntimeError("U2 failed")
        manifest["gates"]["U2"] = "passed"
        manifest.update({"classification": "blackwell_native_bridge_viable", "status": "completed", "author_contact_recommendation": "not_needed"})
    except BaseException as exc:
        manifest.update({"status": "failed", "classification": manifest["classification"] or "blocked_by_external_resources", "failure": {"class": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}})
        raise
    finally:
        for gate in ("L0", "L1", "L2", "L3", "C2", "C3A", "C3B", "C3C", "U2"):
            manifest["gates"].setdefault(gate, "not_run_due_to_gate")
        if manifest["gates"]["L3"] == "runtime_resolved_after_torch_import":
            manifest["loader_closure_status"] = "runtime_resolved_after_torch_import"
            manifest["r082_runtime_bundle_status"] = "superseded_by_loader_context_validation"
        if manifest.get("classification") == "curobo_official_example_packaging_failure":
            manifest["author_bundle_type"] = "none_example_packaging_failure"
        finalization_errors = []
        comparison = {}
        try:
            after = {name: fingerprint_environment(args.conda_exe, env) for name, env in (("legacy", "UniVTAC"), ("r07", "UniVTAC-isaac51-r07"), ("r08", cfg["environment"]["conda_name"]))}
            write_json(output / "environment_fingerprints/after.json", after)
            comparison = {
                name: compare_fingerprints(before[name], after[name])
                for name in before.keys() & after.keys()
            }
            write_json(output / "environment_fingerprints/comparison.json", comparison)
        except BaseException as exc:  # noqa: BLE001 - retain the primary gate failure
            finalization_errors.append({"stage": "environment_fingerprint", "class": type(exc).__name__, "message": str(exc)})
        apply_environment_gate(manifest, comparison)
        try:
            write_json(output / "final_resources/cleanup.json", managed_cleanup_summary(output))
        except BaseException as exc:  # noqa: BLE001
            finalization_errors.append({"stage": "cleanup", "class": type(exc).__name__, "message": str(exc)})
        try:
            write_json(output / "final_resources/source.json", {"curobo_head": git(curobo, "rev-parse", "HEAD"), "curobo_status": git(curobo, "status", "--short", "--untracked-files=no"), "univtac_status": git(source, "status", "--short", "--untracked-files=no")})
        except BaseException as exc:  # noqa: BLE001
            finalization_errors.append({"stage": "source", "class": type(exc).__name__, "message": str(exc)})
        if finalization_errors:
            manifest["finalization_errors"] = finalization_errors
            if manifest.get("status") == "completed":
                manifest.update({"status": "failed", "classification": "finalization_failed"})
        manifest["ended_at"] = utc_now()
        write_json(output / "summary.json", manifest)
        write_json(output / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
