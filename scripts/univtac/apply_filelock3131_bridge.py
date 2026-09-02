#!/usr/bin/env python3
"""Apply the sole authorized R0.9.4 filelock transition and run NFL."""

from __future__ import annotations

import argparse
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

from scripts.univtac.install_isaac51_blackwell_runtime import preflight, run_native_suite
from scripts.univtac.resume_blackwell_native_bridge import managed_cleanup_summary
from scripts.univtac.resume_isaac51_blackwell_install import (
    compare_protected_binaries,
    package_probe,
    protected_binary_state,
)
from scripts.univtac.run_isaac51_runtime_gates import compare_fingerprints, fingerprint_environment
from sim.envs.univtac.filelock_compatibility import (
    AFTER_VERSION,
    APPLIED_LABEL,
    BEFORE_VERSION,
    exact_filelock_transition,
    private_directory_record,
    validate_config as validate_bridge_config,
)
from sim.envs.univtac.isaac51_blackwell_runtime import (
    clean_runtime_environment,
    load_json,
    managed,
    require_process,
)
from sim.envs.univtac.resource_sanitation import utc_now, write_json


STAGES = (
    "R0", "F0", "F1", "F2", "FSEC", "F3", "F4", "F5", "NFL",
    "P0A", "P0B", "I0A", "I0B", "I1", "I2", "N0", "S0", "S1",
    "G0", "L0", "C0", "H0",
)


def _same_binary_state(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return compare_protected_binaries(left, right)["success"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--bridge-config", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--curobo-checkout", type=Path, required=True)
    parser.add_argument("--r093-output", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--conda-exe", type=Path, required=True)
    args = parser.parse_args()

    runtime = yaml.safe_load(args.runtime_config.read_text(encoding="utf-8"))
    bridge = yaml.safe_load(args.bridge_config.read_text(encoding="utf-8"))
    validate_bridge_config(bridge)
    source = args.source_checkout.resolve()
    curobo = args.curobo_checkout.resolve()
    previous = args.r093_output.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    base = Path(
        subprocess.run(
            [str(args.conda_exe), "info", "--base"], check=True, capture_output=True, text=True
        ).stdout.strip()
    )
    prefix = base / "envs" / bridge["environment"]
    python = prefix / "bin/python"
    environment, _ = clean_runtime_environment(prefix, source, runtime["runtime"]["gpu"])
    manifest: dict[str, Any] = {
        "schema_version": "openeta.univtac.r094_filelock_bridge.v1",
        "runtime_variant": runtime["runtime_variant"],
        "applied_label": APPLIED_LABEL,
        **runtime["claims"],
        "status": "running",
        "classification": None,
        "started_at": utc_now(),
        "stages": {stage: "not_run_due_to_gate" for stage in STAGES},
        "filelock_mutation_invocations": 0,
        "isaac_actual_install_invocations": 0,
        "isaac_started": False,
        "agent_started": False,
        "planner_behavior_modified": False,
        "task_behavior_modified": False,
        "constraint_behavior_modified": False,
    }
    write_json(output / "run_manifest.json", manifest)
    r08_before: dict[str, Any] = {}
    before_binaries: dict[str, Any] = {}
    try:
        resources = preflight(runtime, (ROOT, source, curobo, previous, output), output)
        write_json(output / "resume_preflight/resources.json", resources)
        prior = load_json(previous / "run_manifest.json")
        prior_preflight = load_json(previous / "resume_preflight/summary.json")
        packages = package_probe(
            python=python,
            source=source,
            root=output / "resume_preflight",
            name="packages",
            environment=environment,
            timeout=600,
        )
        r08_before = fingerprint_environment(args.conda_exe, runtime["environment"]["clone_from"])
        before_binaries = protected_binary_state(curobo, prefix)
        required = {
            "setuptools": "75.8.2",
            "setuptools-scm": "8.1.0",
            "wheel": "0.42.0",
            "packaging": "23.0",
            "filelock": BEFORE_VERSION,
            "torch": "2.7.0+cu128",
            "torchvision": "0.22.0+cu128",
            "warp-lang": "1.17.0",
            "pyuipc": "0.9.0",
        }
        conditions = {
            "resources": resources["passed"],
            "r093_classification": prior.get("classification") == "build_tool_aligned_isaac_resolution_conflict",
            "r093_build_tool_mutations_complete": all(
                prior.get("stages", {}).get(stage) == "passed"
                for stage in ("D5_STEP1", "D5_STEP2", "D5_STEP3", "D6", "NBT")
            ),
            "r093_isaac_actual_install_zero": prior.get("isaac_actual_install_invocations") == 0,
            "versions": all(packages["all_distributions"].get(name) == version for name, version in required.items()),
            "vcs_versioning_absent": packages["all_distributions"].get("vcs-versioning") is None,
            "flatdict_absent": packages["distributions"].get("flatdict") is None,
            "isaac_absent": packages["distributions"].get("isaaclab") is None and packages["distributions"].get("isaacsim") is None,
            "tacex_absent": packages["distributions"].get("tacex") is None and packages["distributions"].get("tacex-assets") is None,
            "protected_binaries_unchanged": _same_binary_state(prior_preflight["protected_binaries"], before_binaries),
            "curobo_extension_count": len(before_binaries["curobo_extensions"]["records"]) == 5,
            "libuipc_binary_count": len(before_binaries["libuipc"]["records"]) == 12,
            "source_clean": not subprocess.run(
                ["git", "status", "--short", "--untracked-files=no"],
                cwd=source,
                check=False,
                capture_output=True,
                text=True,
            ).stdout.strip(),
        }
        write_json(
            output / "resume_preflight/summary.json",
            {"passed": all(conditions.values()), "conditions": conditions, "packages": packages, "protected_binaries": before_binaries},
        )
        if not all(conditions.values()):
            manifest["classification"] = "r094_resume_precondition_failed"
            raise RuntimeError("R0 failed")
        manifest["stages"]["R0"] = "passed"

        reverse_path = output / "filelock_reverse_dependencies/audit.json"
        reverse = managed(
            [str(python), str(ROOT / "scripts/univtac/audit_filelock_compatibility.py"), "--output", str(reverse_path)],
            cwd=source,
            output_root=output / "filelock_reverse_dependencies",
            name="audit",
            timeout_seconds=600,
            environment=environment,
        )
        require_process("F0 reverse-dependency audit", reverse)
        if not load_json(reverse_path)["success"]:
            raise RuntimeError("filelock reverse dependency rejects 3.13.1")
        manifest["stages"]["F0"] = "passed"

        wheel_root = output / "filelock_wheel"
        prepared = managed(
            [
                str(python), str(ROOT / "scripts/univtac/prepare_filelock3131_wheel.py"),
                "--config", str(args.bridge_config.resolve()), "--target-python", str(python),
                "--output-root", str(wheel_root),
            ],
            cwd=source,
            output_root=output / "isolated_filelock_smoke",
            name="prepare",
            timeout_seconds=600,
            environment=environment,
        )
        require_process("F1/F2 fixed wheel preparation", prepared)
        wheel_summary = load_json(wheel_root / "summary.json")
        if not wheel_summary["success"] or not wheel_summary["temporary_root_removed"]:
            raise RuntimeError("fixed filelock wheel smoke incomplete")
        manifest["stages"]["F1"] = "passed"
        manifest["stages"]["F2"] = "passed"

        private_root = output / "filelock_private_runtime"
        private_root.mkdir(mode=0o700)
        private_root.chmod(0o700)
        private_record = private_directory_record(private_root)
        security = {
            **bridge["security_exception"],
            "environment_prefix": str(prefix),
            "environment_owned_by_current_user": prefix.stat().st_uid == os.getuid(),
            "source_owned_by_current_user": source.stat().st_uid == os.getuid(),
            "output_owned_by_current_user": output.stat().st_uid == os.getuid(),
            "private_runtime_directory": private_record,
            "exploit_tested": False,
            "symlink_attack_tested": False,
        }
        security["success"] = all(
            (
                security["isolated_research_environment"],
                not security["production"],
                not security["untrusted_code"],
                security["environment_owned_by_current_user"],
                security["source_owned_by_current_user"],
                security["output_owned_by_current_user"],
                private_record["private"],
                private_record["owned_by_current_user"],
            )
        )
        write_json(output / "security_exception/manifest.json", security)
        if not security["success"]:
            raise RuntimeError("FSEC private research environment contract failed")
        manifest["stages"]["FSEC"] = "passed"

        wheel = Path(wheel_summary["wheel_path"])
        command = [str(python), "-m", "pip", "install", "--no-index", "--no-deps", str(wheel)]
        write_json(
            output / "filelock_mutation/plan.json",
            {
                "authorized_distribution_changes": {"filelock": {"before": BEFORE_VERSION, "after": AFTER_VERSION}},
                "command": command,
                "automatic_dependency_changes": False,
            },
        )
        manifest["stages"]["F3"] = "passed"
        manifest["filelock_mutation_invocations"] = 1
        write_json(output / "run_manifest.json", manifest)
        mutation = managed(
            command,
            cwd=source,
            output_root=output / "filelock_mutation",
            name="install",
            timeout_seconds=600,
            environment=environment,
        )
        require_process("F4 filelock exact install", mutation)
        after = package_probe(
            python=python,
            source=source,
            root=output / "filelock_mutation",
            name="packages_after",
            environment=environment,
            timeout=600,
        )
        transition = exact_filelock_transition(packages["all_distributions"], after["all_distributions"])
        write_json(output / "filelock_mutation/diff.json", transition)
        if not transition["success"]:
            manifest["classification"] = "filelock_bridge_unexpected_environment_change"
            raise RuntimeError("F4 changed distributions beyond filelock")
        manifest["stages"]["F4"] = "passed"

        pip_check = managed(
            [str(python), "-m", "pip", "check"],
            cwd=source,
            output_root=output / "filelock_postcheck",
            name="pip_check",
            timeout_seconds=600,
            environment=environment,
        )
        require_process("F5 pip check", pip_check)
        direct_smoke = managed(
            [
                str(python), str(ROOT / "scripts/univtac/smoke_installed_filelock.py"),
                "--work-root", str(output / "filelock_postcheck/private_lock_smoke"),
                "--output", str(output / "filelock_postcheck/api_smoke.json"),
            ],
            cwd=source,
            output_root=output / "filelock_postcheck/direct_smoke",
            name="probe",
            timeout_seconds=60,
            environment=environment,
        )
        require_process("F5 direct filelock API smoke", direct_smoke)
        protected_after = protected_binary_state(curobo, prefix)
        post_conditions = {
            "filelock": after["all_distributions"].get("filelock") == AFTER_VERSION,
            "setuptools": after["all_distributions"].get("setuptools") == "75.8.2",
            "setuptools_scm": after["all_distributions"].get("setuptools-scm") == "8.1.0",
            "vcs_versioning_absent": after["all_distributions"].get("vcs-versioning") is None,
            "packaging": after["all_distributions"].get("packaging") == "23.0",
            "wheel": after["all_distributions"].get("wheel") == "0.42.0",
            "protected_binaries_unchanged": _same_binary_state(before_binaries, protected_after),
        }
        write_json(output / "filelock_postcheck/summary.json", {"success": all(post_conditions.values()), "conditions": post_conditions, "packages": after})
        if not all(post_conditions.values()):
            manifest["classification"] = "filelock_bridge_post_install_check_failed"
            raise RuntimeError("F5 post-install checks failed")
        manifest["stages"]["F5"] = "passed"

        run_native_suite(
            stage="NFL",
            python=python,
            source=source,
            curobo=curobo,
            output_root=output / "native_regression",
            environment=environment,
            timeouts=runtime["timeouts_seconds"],
        )
        manifest["stages"]["NFL"] = "passed"
        marker_path = prefix / ".univtac-r09-provenance.json"
        marker_before = load_json(marker_path)
        expected_prior_variant = runtime["runtime_variant"].removesuffix("+" + APPLIED_LABEL)
        original_variant = expected_prior_variant.removesuffix("+isaaclab_setuptools_scm8_packaging23_bridge_v1")
        if marker_before.get("runtime_variant") not in {expected_prior_variant, original_variant}:
            raise RuntimeError("R0.9 provenance marker does not describe the pre-filelock runtime")
        marker_after = {
            **marker_before,
            "runtime_variant": runtime["runtime_variant"],
            "previous_runtime_variant": marker_before["runtime_variant"],
            "filelock_bridge_applied_at": utc_now(),
        }
        write_json(marker_path, marker_after)
        write_json(
            output / "filelock_postcheck/provenance_marker_transition.json",
            {"before": marker_before, "after": marker_after},
        )
        manifest["status"] = "filelock_bridge_validated"
        manifest["classification"] = "filelock_bridge_native_validated"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["classification"] = manifest["classification"] or "filelock_bridge_failed"
        manifest["failure"] = {"class": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        raise
    finally:
        finalization_errors = []
        try:
            if r08_before:
                comparison = compare_fingerprints(
                    r08_before,
                    fingerprint_environment(args.conda_exe, runtime["environment"]["clone_from"]),
                )
                write_json(output / "environment_fingerprints/r08_comparison.json", comparison)
                manifest["r08_unchanged"] = comparison["unchanged"]
        except BaseException as exc:
            finalization_errors.append({"stage": "r08", "error": str(exc)})
        try:
            cleanup = managed_cleanup_summary(output)
            write_json(output / "final_resources/managed_cleanup.json", cleanup)
            manifest["managed_cleanup_complete"] = cleanup["all_records_cleanup_complete"] and not cleanup["cleanup_residual"]
        except BaseException as exc:
            finalization_errors.append({"stage": "cleanup", "error": str(exc)})
        manifest["finalization_errors"] = finalization_errors
        manifest["ended_at"] = utc_now()
        write_json(output / "run_manifest.json", manifest)
        write_json(output / "summary.json", manifest)


if __name__ == "__main__":
    main()
