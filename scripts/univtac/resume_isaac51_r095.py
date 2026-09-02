#!/usr/bin/env python3
"""Build the R0.9.5 Isaac artifact lock and stop before unsafe downloads."""

from __future__ import annotations

import argparse
import json
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
from scripts.univtac.resume_isaac51_blackwell_install import (
    compare_protected_binaries,
    package_probe,
    protected_binary_state,
)
from scripts.univtac.run_isaac51_runtime_gates import compare_fingerprints, fingerprint_environment
from sim.envs.univtac.artifact_lock_contract import FLATDICT_SHA256, failure_classification, sha256_file, validate_config as validate_wheelhouse_config
from sim.envs.univtac.isaac51_blackwell_runtime import clean_runtime_environment, load_json, managed
from sim.envs.univtac.resource_sanitation import (
    attach_gpu_usage,
    build_restore_ready,
    collect_gpu_inventory,
    collect_inotify_inventory,
    collect_process_inventory,
    utc_now,
    write_json,
)
from sim.envs.univtac.simulator_install_contract import required_runtime_versions, validate_config as validate_runtime_config, validate_provenance_marker
from sim.envs.univtac.validated_dependency_baseline import package_diff


STAGES = ("R0", "L0", "L1", "L2", "D0", "D1", "D2", "O0", "I0B_R1", "I0C", "I1", "I2", "N0", "S0", "S1", "G0", "TASK_L0", "C0", "H0")


def git(path: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=path, check=False, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--wheelhouse-config", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--curobo-checkout", type=Path, required=True)
    parser.add_argument("--r094-output", type=Path, required=True)
    parser.add_argument("--wheel-cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--conda-exe", type=Path, required=True)
    args = parser.parse_args()
    runtime = yaml.safe_load(args.runtime_config.read_text(encoding="utf-8"))
    wheelhouse = yaml.safe_load(args.wheelhouse_config.read_text(encoding="utf-8"))
    validate_runtime_config(runtime)
    validate_wheelhouse_config(wheelhouse)
    source = args.source_checkout.resolve()
    curobo = args.curobo_checkout.resolve()
    previous = args.r094_output.resolve()
    cache = args.wheel_cache_root.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, mode=0o750)
    output.chmod(0o750)
    conda_base = Path(subprocess.run([str(args.conda_exe), "info", "--base"], check=True, capture_output=True, text=True).stdout.strip())
    prefix = conda_base / "envs" / wheelhouse["environment"]
    python = prefix / "bin/python"
    environment, _ = clean_runtime_environment(prefix, source, runtime["runtime"]["gpu"])
    manifest: dict[str, Any] = {
        "schema_version": "openeta.univtac.r095_offline_wheelhouse.v1",
        "runtime_variant": runtime["runtime_variant"],
        "installation_method_label": wheelhouse["installation_method_label"],
        **runtime["claims"],
        "network_transport_semantics_changed": True,
        "runtime_package_semantics_changed": False,
        "status": "running",
        "classification": None,
        "started_at": utc_now(),
        "stages": {stage: "not_run_due_to_gate" for stage in STAGES},
        "wheel_cache_root": str(cache),
        "wheel_cache_created": False,
        "transport_attempts": 0,
        "total_isaac_install_invocations": 1,
        "network_incomplete_invocations": 1,
        "authorized_offline_retry_invocations": 0,
        "additional_install_retry_allowed": False,
        "third_install_invocation": False,
        "isaac_started": False,
        "agent_started": False,
    }
    write_json(output / "run_manifest.json", manifest)
    r08_before: dict[str, Any] = {}
    try:
        resources = preflight(runtime, (ROOT, source, curobo, previous, output), output)
        write_json(output / "resume_preflight/resources.json", resources)
        prior = load_json(previous / "run_manifest.json")
        failure = load_json(previous / "install_isaac/failure_diagnosis.json")
        f2_exact = load_json(previous / "filelock_wheel_venv_remediation/summary.json")
        f5_exact = load_json(previous / "filelock_postcheck_remediation/api_smoke.json")
        nfl = load_json(previous / "native_regression/nfl/summary.json")
        p0a = load_json(previous / "dependency_plan/isaac_summary.json")
        p0b = load_json(previous / "dependency_plan/tacex_summary.json")
        original_f2 = load_json(previous / "filelock_wheel/summary.json")
        prior_install_process = load_json(previous / "install_isaac/processes/install.json")
        install_log = Path(prior_install_process["log_path"]).read_text(encoding="utf-8", errors="replace")
        current = package_probe(python=python, source=source, root=output / "resume_preflight", name="packages", environment=environment, timeout=600)
        baseline = load_json(previous / "dependency_plan/packages_before.json")
        expected_diff = package_diff(baseline["all_distributions"], current["all_distributions"], allowed_additions={"flatdict": "4.0.1"})
        binaries = protected_binary_state(curobo, prefix)
        recorded_binaries = load_json(previous / "resume_preflight/summary.json")["protected_binaries"]
        r08_before = fingerprint_environment(args.conda_exe, runtime["environment"]["clone_from"])
        required = required_runtime_versions(runtime)
        isaac_names = [name for name in current["all_distributions"] if name.startswith("isaacsim") or name.startswith("isaaclab")]
        flatdict_wheel = Path(p0a["flatdict"]["wheel"])
        conditions = {
            "openeta_head": git(ROOT, "rev-parse", "HEAD") == "8fcb38099731d0a984aea68cbb3447a6c076b2bb",
            "provenance_marker": True,
            "r094_classification": prior.get("classification") == "blocked_by_external_resources",
            "r094_subreason": failure.get("failure_reason") == "isaac51_package_download_incomplete",
            "p0a_passed": p0a.get("passed") is True,
            "p0b_passed": p0b.get("passed") is True,
            "f2_original_preserved": original_f2.get("success") is True and original_f2.get("isolation_method") is None,
            "f2_exact_remediation": f2_exact.get("success") is True and f2_exact.get("isolation_method") == "fresh_venv" and f2_exact.get("wheel", {}).get("sha256") == "57dbda9b35157b05fb3e58ee91448612eb674172fab98ee235ccb0b5bee19a1c",
            "f5_exact_remediation": f5_exact.get("success") is True and f5_exact.get("distribution_version") == "3.13.1",
            "nfl_passed": nfl.get("passed") is True,
            "filelock_mutation_count": prior.get("filelock_mutation_invocations") == 1,
            "prior_isaac_install_count": prior.get("isaac_actual_install_invocations") == 1,
            "prior_install_command_count": "--dry-run" not in prior_install_process.get("command", []) and "isaaclab[isaacsim,all]==2.3.0" in prior_install_process.get("command", []),
            "prior_resolver_completed": p0a.get("passed") is True,
            "prior_download_entered": "Downloading isaacsim_extscache_kit" in install_log,
            "prior_transaction_not_entered": "Installing collected packages" not in install_log and "Successfully installed" not in install_log,
            "no_isaac_distributions": not isaac_names,
            "tacex_absent": current["all_distributions"].get("tacex") is None and current["all_distributions"].get("tacex-assets") is None,
            "flatdict_only_post_i0a_diff": expected_diff["success"],
            "flatdict_version": current["all_distributions"].get("flatdict") == "4.0.1",
            "flatdict_wheel_hash": flatdict_wheel.is_file() and sha256_file(flatdict_wheel) == FLATDICT_SHA256,
            "protected_versions": all(current["distributions"].get(name) == version for name, version in required.items()),
            "protected_binaries": compare_protected_binaries(recorded_binaries, binaries)["success"],
            "r08_unchanged": prior.get("r08_unchanged") is True,
            "source_clean": not git(source, "status", "--short", "--untracked-files=no"),
            "source_head": git(source, "rev-parse", "HEAD") == runtime["source"]["univtac_commit"],
            "vendor_diff_empty": not git(ROOT, "diff", "--", "third_party/ftp1-policy/UniVTAC"),
            "resources": resources["passed"],
            "prior_cleanup": prior.get("managed_cleanup_complete") is True,
            "wheel_cache_absent_before_l0": not cache.exists(),
        }
        try:
            validate_provenance_marker(load_json(prefix / ".univtac-r09-provenance.json"), source=runtime["environment"]["clone_from"], target=runtime["environment"]["conda_name"])
        except Exception:
            conditions["provenance_marker"] = False
        remediation = {
            "F2_original": "nonconforming_test_shape",
            "F2_exact_remediation": "passed_in_fresh_venv",
            "F5_original": "functional_check_incomplete",
            "F5_exact_remediation": "passed_in_actual_r09_environment",
            "combined_status": "accepted_after_exact_remediation",
        }
        write_json(output / "resume_preflight/remediation_acceptance.json", remediation)
        write_json(output / "resume_preflight/summary.json", {"passed": all(conditions.values()), "conditions": conditions, "packages": current, "authorized_diff": expected_diff, "protected_binaries": binaries})
        if not all(conditions.values()):
            manifest["classification"] = "r095_resume_precondition_failed"
            raise RuntimeError("R0 failed")
        manifest["stages"]["R0"] = "passed"

        artifact_root = output / "artifact_lock"
        lock_process = managed(
            [
                str(python), str(ROOT / "scripts/univtac/build_isaac_artifact_lock.py"),
                "--config", str(args.wheelhouse_config.resolve()),
                "--p0a-report", str(previous / wheelhouse["source_report"]["relative_path"]),
                "--p0a-process", str(previous / "dependency_plan/isaac/processes/dry_run.json"),
                "--constraints", str(previous / "dependency_plan/final_protected_constraints.txt"),
                "--output-root", str(artifact_root),
            ],
            cwd=ROOT,
            output_root=output / "artifact_lock_process",
            name="build",
            timeout_seconds=600,
            environment=environment,
        )
        lock = load_json(artifact_root / "artifact_lock.json")
        classification = failure_classification(lock)
        if lock_process.get("returncode") == 0 and not classification:
            manifest["stages"]["L0"] = "passed"
            raise RuntimeError("R0.9.5 downloader is intentionally unavailable until L0 is shown to pass")
        manifest["stages"]["L0"] = "failed"
        manifest["classification"] = classification or "wheelhouse_validation_failed"
        if manifest["classification"] == "artifact_lock_contains_nonwheel":
            raise RuntimeError("P0A artifact lock contains non-wheel distributions")
        raise RuntimeError("artifact lock validation failed")
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
