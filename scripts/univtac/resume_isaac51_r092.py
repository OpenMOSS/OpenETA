#!/usr/bin/env python3
"""Run the R0.9.2 preflight and packaging-23 compatibility gate."""

from __future__ import annotations

import argparse
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.univtac.install_isaac51_blackwell_runtime import preflight
from scripts.univtac.resume_blackwell_native_bridge import managed_cleanup_summary
from scripts.univtac.resume_isaac51_blackwell_install import (
    package_probe,
    protected_binary_state,
)
from scripts.univtac.run_isaac51_runtime_gates import compare_fingerprints, fingerprint_environment
from sim.envs.univtac.isaac51_blackwell_runtime import clean_runtime_environment, load_json, managed
from sim.envs.univtac.packaging_compatibility import COMPATIBILITY_LABEL, validate_config as validate_packaging_config
from sim.envs.univtac.resource_sanitation import (
    attach_gpu_usage,
    build_restore_ready,
    collect_gpu_inventory,
    collect_inotify_inventory,
    collect_process_inventory,
    utc_now,
    write_json,
)
from sim.envs.univtac.simulator_dependency_layers import packaging_layers
from sim.envs.univtac.simulator_install_contract import RUNTIME_VARIANT, validate_config, validate_provenance_marker
from sim.envs.univtac.validated_dependency_baseline import compare_binary_manifests


R092_RUNTIME_VARIANT = RUNTIME_VARIANT + "+isaaclab_packaging23_compatibility_bridge_v1"
STAGES = ("A0", "A1", "A2", "A3", "A4", "A5", "N23", "P0A", "P0B", "I0A", "I0B", "I1", "I2", "N0", "S0", "S1", "G0", "L0", "C0", "H0")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--packaging-config", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--curobo-checkout", type=Path, required=True)
    parser.add_argument("--r091-output", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--conda-exe", type=Path, required=True)
    args = parser.parse_args()
    runtime_config = yaml.safe_load(args.runtime_config.read_text(encoding="utf-8"))
    packaging_config = yaml.safe_load(args.packaging_config.read_text(encoding="utf-8"))
    validate_config(runtime_config)
    validate_packaging_config(packaging_config)
    source = args.source_checkout.resolve()
    curobo = args.curobo_checkout.resolve()
    previous = args.r091_output.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"R0.9.2 output must be fresh: {output}")
    output.mkdir(parents=True)
    conda_base = subprocess.run([str(args.conda_exe), "info", "--base"], check=True, capture_output=True, text=True).stdout.strip()
    source_env = runtime_config["environment"]["clone_from"]
    target_env = runtime_config["environment"]["conda_name"]
    source_prefix = Path(conda_base) / "envs" / source_env
    prefix = Path(conda_base) / "envs" / target_env
    python = prefix / "bin/python"
    environment, _ = clean_runtime_environment(prefix, source, runtime_config["runtime"]["gpu"])
    manifest: dict[str, Any] = {
        "schema_version": "openeta.univtac.r092_run.v1",
        "runtime_variant": R092_RUNTIME_VARIANT,
        "compatibility_label": COMPATIBILITY_LABEL,
        **runtime_config["claims"],
        "started_at": utc_now(), "status": "running", "classification": None,
        "dependency_layers": packaging_layers(packaging_config),
        "environment_recloned": False, "r08_modified": False,
        "packaging_install_invocations": 0, "isaac_actual_install_invocations": 0,
        "isaac_started": False, "agent_started": False,
        "stages": {stage: "not_run_due_to_gate" for stage in STAGES},
    }
    write_json(output / "run_manifest.json", manifest)
    r08_before: dict[str, Any] = {}
    try:
        resource = preflight(runtime_config, (REPO_ROOT, source, curobo, previous, output), output)
        write_json(output / "resume_preflight/resources.json", resource)
        if not resource["passed"]:
            manifest["classification"] = "r092_resume_precondition_failed"
            raise RuntimeError("R0.9.2 resource preflight failed")
        marker = load_json(prefix / ".univtac-r09-provenance.json")
        validate_provenance_marker(marker, source=source_env, target=target_env)
        prior = load_json(previous / "run_manifest.json")
        r08_before = fingerprint_environment(args.conda_exe, source_env)
        r08_reference = load_json(previous / "environment_fingerprints/r08_after.json")
        r08_comparison = compare_fingerprints(r08_reference, r08_before)
        current_packages = package_probe(python=python, source=source, root=output / "resume_preflight", name="r09_packages", environment=environment, timeout=float(runtime_config["timeouts_seconds"]["package_check"]))
        prior_packages = load_json(previous / "resume_preflight/r09_packages.json")
        common = set(prior_packages["distributions"]) & set(current_packages["distributions"])
        current_binaries = protected_binary_state(curobo, prefix)
        prior_binaries = load_json(previous / "package_provenance/protected_binaries_before.json")
        binary_comparison = {
            name: compare_binary_manifests(prior_binaries[name], current_binaries[name])
            for name in ("curobo_extensions", "libuipc")
        }
        conditions = {
            "marker_valid": True,
            "r091_expected_classification": prior.get("classification") == "validated_r08_baseline_incompatible_with_isaac_resolution",
            "r091_actual_isaac_installs_zero": prior.get("isaac_actual_install_invocations") == 0,
            "r091_cleanup_complete": prior.get("managed_cleanup_complete") is True,
            "r08_unchanged": r08_comparison["unchanged"],
            "r09_packaging_is_26_3": current_packages["distributions"].get("packaging") == "26.3",
            "r09_packages_match_r091": all(prior_packages["distributions"][name] == current_packages["distributions"][name] for name in common),
            "r09_curobo_extensions_unchanged": binary_comparison["curobo_extensions"]["success"],
            "r09_libuipc_unchanged": binary_comparison["libuipc"]["success"],
            "curobo_editable_source": current_packages.get("nvidia_curobo_editable_source") == str(curobo),
        }
        for name in ("flatdict", "isaacsim", "isaaclab", "tacex", "tacex-assets"):
            conditions[f"{name}_absent"] = current_packages["distributions"].get(name) is None
        resume = {"passed": all(conditions.values()), "conditions": conditions, "r08_comparison": r08_comparison, "binary_comparison": binary_comparison, "packages": current_packages}
        write_json(output / "resume_preflight/summary.json", resume)
        write_json(output / "environment_fingerprints/r08_before.json", r08_before)
        if not resume["passed"]:
            manifest["classification"] = "r092_resume_precondition_failed"
            raise RuntimeError("R0.9.2 resume preconditions failed")

        audit_output = output / "packaging_usage_audit/packaging_requirements_inventory.json"
        audit_command = [
            str(python), str(REPO_ROOT / "scripts/univtac/audit_packaging_compatibility.py"),
            "--source-root", f"curobo={curobo}",
            "--source-root", f"univtac_isaac51={source}",
            "--source-root", f"warp={prefix / 'lib/python3.11/site-packages/warp'}",
            "--source-root", f"libuipc={prefix / 'lib/python3.11/site-packages/uipc'}",
            "--output", str(audit_output),
        ]
        audit_process = managed(audit_command, cwd=source, output_root=output / "packaging_usage_audit", name="audit", timeout_seconds=float(runtime_config["timeouts_seconds"]["package_check"]), environment=environment)
        audit = load_json(audit_output)
        write_json(output / "packaging_usage_audit/process_summary.json", audit_process)
        if audit.get("blockers"):
            manifest["stages"]["A0"] = "failed"
            manifest["classification"] = "existing_environment_requires_packaging24_or_newer"
            raise RuntimeError("A0 found an active installed requirement incompatible with packaging 23.0")
        if audit_process.get("returncode") != 0 or not audit.get("success"):
            manifest["stages"]["A0"] = "failed"
            manifest["classification"] = "blocked_by_external_resources"
            raise RuntimeError("A0 packaging usage audit failed")
        manifest["stages"]["A0"] = "passed"
        raise RuntimeError("A0 unexpectedly passed; A1-A5/N23 continuation is not implemented in this gate-only runner")
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["classification"] = manifest["classification"] or "blocked_by_external_resources"
        manifest["failure"] = {"class": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        bundle = output / "author_bundle/packaging_dependency_conflict"
        bundle.mkdir(parents=True, exist_ok=True)
        write_json(bundle / "summary.json", {"classification": manifest["classification"], "failure": manifest["failure"]})
        raise
    finally:
        errors = []
        try:
            if r08_before:
                r08_after = fingerprint_environment(args.conda_exe, source_env)
                comparison = compare_fingerprints(r08_before, r08_after)
                write_json(output / "environment_fingerprints/r08_after.json", r08_after)
                write_json(output / "environment_fingerprints/r08_comparison.json", comparison)
                manifest["r08_unchanged"] = comparison["unchanged"]
        except BaseException as exc:
            errors.append({"stage": "r08_fingerprint", "error": f"{type(exc).__name__}: {exc}"})
        try:
            cleanup = managed_cleanup_summary(output)
            write_json(output / "final_resources/managed_cleanup.json", cleanup)
            manifest["managed_cleanup_complete"] = cleanup["all_records_cleanup_complete"] and not cleanup["cleanup_residual"]
            inotify = collect_inotify_inventory()
            gpu = collect_gpu_inventory()
            processes = attach_gpu_usage(collect_process_inventory(related_roots=(REPO_ROOT, source, curobo, output)), gpu)
            restore = build_restore_ready(original_instances=128, original_watches=65536, inotify_inventory=inotify, process_inventory=processes, gpu_inventory=gpu)
            restore["sysctl_restore_performed_by_runner"] = False
            write_json(output / "restore_ready.json", restore)
            write_json(output / "final_resources/inventory.json", {"inotify": inotify, "gpu": gpu, "processes": processes})
        except BaseException as exc:
            errors.append({"stage": "cleanup", "error": f"{type(exc).__name__}: {exc}"})
        manifest["finalization_errors"] = errors
        manifest["ended_at"] = utc_now()
        write_json(output / "run_manifest.json", manifest)
        write_json(output / "summary.json", manifest)


if __name__ == "__main__":
    main()
