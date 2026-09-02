#!/usr/bin/env python3
"""Run the R0.9.5.1 legacy-sdist reproducible-wheel bridge."""

from __future__ import annotations

import argparse
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.univtac.install_isaac51_blackwell_runtime import preflight
from scripts.univtac.resume_blackwell_native_bridge import managed_cleanup_summary
from scripts.univtac.resume_isaac51_blackwell_install import compare_protected_binaries, package_probe, protected_binary_state
from scripts.univtac.run_isaac51_runtime_gates import compare_fingerprints, fingerprint_environment
from sim.envs.univtac.isaac51_blackwell_runtime import clean_runtime_environment, load_json, managed, process_ok
from sim.envs.univtac.resource_sanitation import attach_gpu_usage, build_restore_ready, collect_gpu_inventory, collect_inotify_inventory, collect_process_inventory, utc_now, write_json
from sim.envs.univtac.simulator_install_contract import required_runtime_versions, validate_config as validate_runtime_config, validate_provenance_marker
from sim.envs.univtac.source_to_wheel_contract import EXPECTED_PACKAGES, validate_config as validate_bridge_config


STAGES = ("R0", "S0", "S1", "S2", "B0", "B1", "B2", "B3", "B4", "L0R", "L1", "L2", "D0", "D1", "D2", "O0", "I0B_R1", "I0C", "I1", "I2", "N0", "SIM_S0", "SIM_S1", "G0", "TASK_L0", "C0", "H0")


def git(path: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=path, check=False, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--bridge-config", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--curobo-checkout", type=Path, required=True)
    parser.add_argument("--r095-output", type=Path, required=True)
    parser.add_argument("--wheel-cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--conda-exe", type=Path, required=True)
    args = parser.parse_args()
    runtime = yaml.safe_load(args.runtime_config.read_text(encoding="utf-8"))
    bridge = yaml.safe_load(args.bridge_config.read_text(encoding="utf-8"))
    validate_runtime_config(runtime)
    validate_bridge_config(bridge)
    source = args.source_checkout.resolve()
    curobo = args.curobo_checkout.resolve()
    previous = args.r095_output.resolve()
    cache = args.wheel_cache_root.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, mode=0o750)
    output.chmod(0o750)
    conda_base = Path(subprocess.run([str(args.conda_exe), "info", "--base"], check=True, capture_output=True, text=True).stdout.strip())
    prefix = conda_base / "envs" / bridge["environment"]
    python = prefix / "bin/python"
    environment, _ = clean_runtime_environment(prefix, source, runtime["runtime"]["gpu"])
    manifest: dict[str, Any] = {
        "schema_version": "openeta.univtac.r0951_legacy_sdist_bridge.v1",
        "runtime_variant": runtime["runtime_variant"],
        "installation_method_label": "isaac51_hash_locked_offline_wheelhouse_v1",
        "build_method_label": bridge["build_method_label"],
        **runtime["claims"],
        "runtime_package_versions_changed": False,
        "source_package_versions_changed": False,
        "source_code_patched": False,
        "final_install_all_wheel": False,
        "status": "running",
        "classification": None,
        "started_at": utc_now(),
        "stages": {stage: "not_run_due_to_gate" for stage in STAGES},
        "wheel_cache_root": str(cache),
        "wheel_cache_created": False,
        "transport_attempts": 0,
        "total_isaac_install_invocations": 1,
        "authorized_offline_retry_invocations": 0,
        "third_install_invocation": False,
        "agent_started": False,
        "isaac_started": False,
    }
    write_json(output / "run_manifest.json", manifest)
    r08_before: dict[str, Any] = {}
    try:
        resources = preflight(runtime, (ROOT, source, curobo, previous, output, cache), output)
        write_json(output / "resume_preflight/resources.json", resources)
        prior = load_json(previous / "run_manifest.json")
        source_lock = load_json(previous / "artifact_lock/artifact_lock.private.json")
        public_lock = load_json(previous / "artifact_lock/artifact_lock.json")
        current = package_probe(python=python, source=source, root=output / "resume_preflight", name="packages", environment=environment, timeout=600)
        binaries = protected_binary_state(curobo, prefix)
        prior_binaries = load_json(previous / "resume_preflight/summary.json")["protected_binaries"]
        r08_before = fingerprint_environment(args.conda_exe, runtime["environment"]["clone_from"])
        required = required_runtime_versions(runtime)
        sdist_records = {record["name"]: record for record in source_lock["records"] if record["artifact_type"] == "sdist"}
        sdist_identity = all(
            name in sdist_records and (
                sdist_records[name]["version"], sdist_records[name]["filename"], sdist_records[name]["sha256"]
            ) == (values[0], values[1], values[2])
            for name, values in EXPECTED_PACKAGES.items()
        )
        other_failures = {key: value for key, value in public_lock["failures"].items() if key != "nonwheel_artifacts"}
        conditions = {
            "openeta_head": git(ROOT, "rev-parse", "HEAD") == "fb1464c4fd00d71382fd5a61c01d72b475abec19",
            "r095_classification": prior.get("classification") == "artifact_lock_contains_nonwheel",
            "r095_r0": prior.get("stages", {}).get("R0") == "passed",
            "source_record_count": source_lock.get("record_count") == 173,
            "source_sdist_count": len(sdist_records) == 3,
            "source_sdist_identity": sdist_identity,
            "source_wheel_count": sum(record["artifact_type"] == "wheel" for record in source_lock["records"]) == 170,
            "source_wheels_compatible": all(record.get("wheel_tag_compatible") is True for record in source_lock["records"] if record["artifact_type"] == "wheel"),
            "no_other_lock_failures": not any(other_failures.values()),
            "r095_cache_not_created": prior.get("wheel_cache_created") is False and not cache.exists(),
            "r095_transport_zero": prior.get("transport_attempts") == 0,
            "install_accounting": prior.get("total_isaac_install_invocations") == 1 and prior.get("authorized_offline_retry_invocations") == 0,
            "no_isaac_or_tacex": not any(name.startswith("isaacsim") or name.startswith("isaaclab") for name in current["all_distributions"]) and current["all_distributions"].get("tacex") is None and current["all_distributions"].get("tacex-assets") is None,
            "flatdict": current["all_distributions"].get("flatdict") == "4.0.1",
            "protected_versions": all(current["distributions"].get(name) == version for name, version in required.items()),
            "protected_binaries": compare_protected_binaries(prior_binaries, binaries)["success"],
            "r08_unchanged": prior.get("r08_unchanged") is True,
            "source_clean": not git(source, "status", "--short", "--untracked-files=no"),
            "source_head": git(source, "rev-parse", "HEAD") == runtime["source"]["univtac_commit"],
            "vendor_diff_empty": not git(ROOT, "diff", "--", "third_party/ftp1-policy/UniVTAC"),
            "resources": resources["passed"],
            "prior_cleanup": prior.get("managed_cleanup_complete") is True,
        }
        try:
            validate_provenance_marker(load_json(prefix / ".univtac-r09-provenance.json"), source=runtime["environment"]["clone_from"], target=runtime["environment"]["conda_name"])
            conditions["provenance_marker"] = True
        except Exception:
            conditions["provenance_marker"] = False
        write_json(output / "resume_preflight/summary.json", {"passed": all(conditions.values()), "conditions": conditions, "packages": current, "protected_binaries": binaries})
        if not all(conditions.values()):
            manifest["classification"] = "r0951_resume_precondition_failed"
            raise RuntimeError("R0 failed")
        manifest["stages"]["R0"] = "passed"

        bridge_root = output / "legacy_sdist_bridge"
        process = managed(
            [str(python), str(ROOT / "scripts/univtac/build_reproducible_legacy_wheels.py"), "--config", str(args.bridge_config.resolve()), "--source-lock", str(previous / "artifact_lock/artifact_lock.private.json"), "--p0a-report", str(Path(source_lock["source_report"]["path"])), "--target-python", str(python), "--output-root", str(bridge_root)],
            cwd=ROOT,
            output_root=output / "legacy_sdist_bridge_process",
            name="build",
            timeout_seconds=3600,
            environment=environment,
        )
        bridge_manifest = load_json(bridge_root / "run_manifest.json")
        for stage in ("S0", "S1", "S2", "B0", "B1", "B2", "B3", "B4", "L0R"):
            manifest["stages"][stage] = bridge_manifest["stages"].get(stage, "not_run_due_to_gate")
        if not process_ok(process) or bridge_manifest.get("status") != "completed":
            manifest["classification"] = bridge_manifest.get("classification") or "legacy_sdist_wheel_build_failed"
            raise RuntimeError("legacy sdist bridge failed")
        manifest["final_install_all_wheel"] = True
        manifest["status"] = "legacy_sdist_bridge_validated"
        manifest["classification"] = "legacy_sdist_reproducible_wheel_bridge_validated"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["classification"] = manifest["classification"] or "blocked_by_external_resources"
        manifest["failure"] = {"class": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        raise
    finally:
        errors = []
        try:
            if r08_before:
                comparison = compare_fingerprints(r08_before, fingerprint_environment(args.conda_exe, runtime["environment"]["clone_from"]))
                write_json(output / "environment_fingerprints/r08_comparison.json", comparison)
                manifest["r08_unchanged"] = comparison["unchanged"]
        except BaseException as exc:
            errors.append({"stage": "r08", "error": str(exc)})
        try:
            cleanup = managed_cleanup_summary(output)
            write_json(output / "final_resources/managed_cleanup.json", cleanup)
            manifest["managed_cleanup_complete"] = cleanup["all_records_cleanup_complete"] and not cleanup["cleanup_residual"]
            inotify = collect_inotify_inventory()
            gpu = collect_gpu_inventory()
            processes = attach_gpu_usage(collect_process_inventory(related_roots=(ROOT, source, curobo, output, cache)), gpu)
            restore = build_restore_ready(original_instances=128, original_watches=65536, inotify_inventory=inotify, process_inventory=processes, gpu_inventory=gpu)
            restore["sysctl_restore_performed_by_runner"] = False
            write_json(output / "restore_ready.json", restore)
        except BaseException as exc:
            errors.append({"stage": "cleanup", "error": str(exc)})
        manifest["finalization_errors"] = errors
        manifest["ended_at"] = utc_now()
        write_json(output / "run_manifest.json", manifest)
        write_json(output / "summary.json", manifest)


if __name__ == "__main__":
    main()
