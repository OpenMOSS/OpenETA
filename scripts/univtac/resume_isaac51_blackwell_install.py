#!/usr/bin/env python3
"""Resume the provenance-marked R0.9 environment through corrected R0.9.1 gates."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.univtac.install_isaac51_blackwell_runtime import preflight, run_native_suite
from scripts.univtac.resume_blackwell_native_bridge import managed_cleanup_summary
from scripts.univtac.run_isaac51_runtime_gates import compare_fingerprints, fingerprint_environment
from sim.envs.univtac.curobo_build_validation import find_extension_binaries
from sim.envs.univtac.flatdict_build_contract import audit_report_uses_wheel
from sim.envs.univtac.isaac51_blackwell_runtime import clean_runtime_environment, load_json, managed, process_ok, require_process
from sim.envs.univtac.resource_sanitation import (
    attach_gpu_usage,
    build_restore_ready,
    collect_gpu_inventory,
    collect_inotify_inventory,
    collect_process_inventory,
    utc_now,
    write_json,
)
from sim.envs.univtac.simulator_install_contract import (
    INSTALLATION_METHOD_LABEL,
    RUNTIME_VARIANT,
    audit_installed_versions,
    audit_pip_report,
    constraints_text,
    load_pip_report,
    required_runtime_versions,
    validate_config,
    validate_provenance_marker,
)
from sim.envs.univtac.validated_dependency_baseline import (
    audit_changes_against_plan,
    binary_manifest,
    compare_binary_manifests,
    compare_config_to_source,
    package_diff,
    report_install_versions,
)


STAGES = ("F0", "F1", "F2", "F3", "P0A", "P0B", "I0A", "I0B", "I1", "I2", "N0", "S0", "S1", "G0", "L0", "C0", "H0")


def prior_actual_installs(root: Path) -> list[dict[str, Any]]:
    installs = []
    for path in sorted(root.rglob("processes/*.json")):
        payload = load_json(path)
        command = [str(item) for item in payload.get("command", [])]
        if "pip" in command and "install" in command and "--dry-run" not in command:
            installs.append({"path": str(path), "command": command})
    return installs


def package_probe(
    *, python: Path, source: Path, root: Path, name: str,
    environment: Mapping[str, str], timeout: float,
) -> dict[str, Any]:
    result_path = root / f"{name}.json"
    result = managed(
        [str(python), str(REPO_ROOT / "scripts/univtac/probe_isaac51_packages.py"), "--output", str(result_path)],
        cwd=source, output_root=root / name, name="probe", timeout_seconds=timeout, environment=environment,
    )
    require_process(name, result)
    return load_json(result_path)


def relative_binary_hashes(paths: list[Path], root: Path) -> dict[str, str]:
    return {
        str(path.resolve().relative_to(root.resolve())): __import__("hashlib").sha256(path.read_bytes()).hexdigest()
        for path in sorted(paths)
    }


def uipc_binaries(prefix: Path) -> list[Path]:
    return sorted((prefix / "lib/python3.11/site-packages/uipc/modules/Release/bin").glob("*.so"))


def dry_run(
    *, name: str, python: Path, source: Path, root: Path, constraints: Path,
    wheelhouse: Path, report: Path, requirements: list[str], extra_index: str | None,
    environment: Mapping[str, str], timeout: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    command = [
        str(python), "-m", "pip", "install", "--dry-run", "--report", str(report),
        "--constraint", str(constraints), "--find-links", str(wheelhouse), "--only-binary=flatdict",
        *requirements,
    ]
    if extra_index:
        command.extend(["--extra-index-url", extra_index])
    process = managed(command, cwd=source, output_root=root / name, name="dry_run", timeout_seconds=timeout, environment=environment)
    if not process_ok(process) or not report.is_file():
        return process, {"success": False, "report_present": report.is_file()}
    return process, dict(load_pip_report(report))


def protected_binary_state(curobo: Path, prefix: Path) -> dict[str, Any]:
    extensions = find_extension_binaries(curobo)
    return {
        "curobo_extensions": binary_manifest(extensions, root=curobo),
        "libuipc": binary_manifest(uipc_binaries(prefix), root=prefix),
    }


def compare_protected_binaries(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    results = {
        name: compare_binary_manifests(before[name], after[name])
        for name in ("curobo_extensions", "libuipc")
    }
    return {"success": all(item["success"] for item in results.values()), "components": results}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--flatdict-config", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--curobo-checkout", type=Path, required=True)
    parser.add_argument("--previous-output", type=Path, required=True)
    parser.add_argument("--r084-output", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--conda-exe", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    source = args.source_checkout.resolve()
    curobo = args.curobo_checkout.resolve()
    previous = args.previous_output.resolve()
    r084 = args.r084_output.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"R0.9.1 output must be fresh: {output}")
    output.mkdir(parents=True)
    conda_base = subprocess.run([str(args.conda_exe), "info", "--base"], check=True, capture_output=True, text=True).stdout.strip()
    source_env = config["environment"]["clone_from"]
    target_env = config["environment"]["conda_name"]
    source_prefix = Path(conda_base) / "envs" / source_env
    prefix = Path(conda_base) / "envs" / target_env
    python = prefix / "bin/python"
    environment, wrappers = clean_runtime_environment(prefix, source, config["runtime"]["gpu"])
    manifest: dict[str, Any] = {
        "schema_version": "openeta.univtac.r091_resume.v1", "runtime_variant": RUNTIME_VARIANT,
        "installation_method_label": INSTALLATION_METHOD_LABEL, **config["claims"],
        "started_at": utc_now(), "status": "running", "classification": None,
        "previous_r09_classification": "protected_dependency_resolution_conflict",
        "previous_r09_evidence": str(previous),
        "invalid_orchestration_attempt": (
            str(output.with_name(output.name + "-invalid-orchestration-attempt0"))
            if output.with_name(output.name + "-invalid-orchestration-attempt0").is_dir()
            else None
        ),
        "previous_p0_status": "superseded_by_corrected_dependency_plan",
        "environment_recloned": False, "r08_modified": False, "official_install_script_executed": False,
        "global_no_build_isolation": False, "isaac_actual_install_invocations": 0,
        "isaac_started": False, "agent_started": False,
        "stages": {stage: "not_run_due_to_gate" for stage in STAGES},
    }
    write_json(output / "run_manifest.json", manifest)
    source_before: dict[str, Any] = {}
    before_binaries: dict[str, Any] = {}
    try:
        resource = preflight(config, (REPO_ROOT, source, curobo, previous, output), output)
        write_json(output / "resume_preflight/resources.json", resource)
        if not resource["passed"]:
            raise RuntimeError("resource preflight failed")
        prior = load_json(previous / "run_manifest.json")
        marker = load_json(prefix / ".univtac-r09-provenance.json")
        validate_provenance_marker(marker, source=source_env, target=target_env)
        actual_installs = prior_actual_installs(previous)
        conditions = {
            "marker_exists_and_valid": True,
            "prior_classification": prior.get("classification") == "protected_dependency_resolution_conflict",
            "prior_e0_passed": prior.get("stages", {}).get("E0") == "passed",
            "prior_cleanup_complete": prior.get("managed_cleanup_complete") is True,
            "prior_official_installer_not_run": prior.get("official_install_script_executed") is False,
            "prior_isaac_not_started": prior.get("isaac_started") is False,
            "no_prior_actual_pip_install": not actual_installs,
            "source_prefix_exists": source_prefix.is_dir(),
            "target_prefix_exists": prefix.is_dir(),
        }
        source_before = fingerprint_environment(args.conda_exe, source_env)
        marker_comparison = compare_fingerprints(marker["source_fingerprint"], source_before)
        conditions["r08_matches_clone_source_fingerprint"] = marker_comparison["unchanged"]
        source_packages = package_probe(python=source_prefix / "bin/python", source=source, root=output / "resume_preflight", name="r08_packages", environment=environment, timeout=float(config["timeouts_seconds"]["package_check"]))
        target_packages = package_probe(python=python, source=source, root=output / "resume_preflight", name="r09_packages", environment=environment, timeout=float(config["timeouts_seconds"]["package_check"]))
        baseline = compare_config_to_source(config, source_packages["distributions"], target_packages["distributions"])
        conditions["protected_versions_match_r08_and_r09"] = baseline["success"]
        for name in ("isaacsim", "isaaclab", "tacex", "tacex-assets", "flatdict"):
            conditions[f"target_{name}_absent"] = target_packages["distributions"].get(name) is None
        conditions["target_curobo_provenance"] = target_packages.get("nvidia_curobo_editable_source") == str(curobo)
        old_packages = load_json(previous / "p0/baseline/package_probe.json")["distributions"]
        common = set(old_packages) & set(target_packages["distributions"])
        conditions["target_matches_prior_p0_semantics"] = all(old_packages[name] == target_packages["distributions"][name] for name in common)

        before_binaries = protected_binary_state(curobo, prefix)
        r08_uipc = relative_binary_hashes(uipc_binaries(source_prefix), source_prefix)
        r09_uipc = relative_binary_hashes(uipc_binaries(prefix), prefix)
        conditions["r08_r09_libuipc_identical"] = r08_uipc == r09_uipc and bool(r08_uipc)
        r084_records = load_json(r084 / "curobo_binary_audit/audit.json").get("records", [])
        historical_extensions = {Path(item["path"]).name: item.get("sha256") for item in r084_records}
        current_extensions = {Path(item["path"]).name: item["sha256"] for item in before_binaries["curobo_extensions"]["records"]}
        conditions["five_curobo_extensions_match_r084"] = len(current_extensions) == 5 and current_extensions == historical_extensions
        conditions["prior_e0_provenance_valid"] = bool(
            load_json(previous / "e0/summary.json").get("passed")
            and load_json(previous / "e0/imports/probe.json").get("module_provenance_ok")
            and not load_json(previous / "e0/imports/probe.json").get("legacy_paths")
        )
        resume = {
            "passed": all(conditions.values()), "conditions": conditions,
            "prior_actual_installs": actual_installs, "marker": marker,
            "r08_fingerprint": source_before, "r08_marker_comparison": marker_comparison,
            "protected_baseline": baseline, "protected_binaries": before_binaries,
            "compiler_wrappers": wrappers,
        }
        write_json(output / "resume_preflight/summary.json", resume)
        write_json(output / "protected_baseline/validated_versions.json", baseline)
        write_json(output / "environment_fingerprints/r08_before.json", source_before)
        write_json(output / "package_provenance/pre_install.json", target_packages)
        write_json(output / "package_provenance/protected_binaries_before.json", before_binaries)
        if not resume["passed"]:
            manifest["classification"] = "r09_resume_precondition_failed"
            raise RuntimeError("R0.9 resume preconditions failed")

        bridge = managed(
            [str(python), str(REPO_ROOT / "scripts/univtac/prepare_flatdict_wheel.py"), "--config", str(args.flatdict_config.resolve()), "--target-python", str(python), "--output-root", str(output / "flatdict_bridge")],
            cwd=source, output_root=output / "flatdict_source", name="prepare", timeout_seconds=float(config["timeouts_seconds"]["pip_dry_run"]), environment=environment,
        )
        if not process_ok(bridge):
            bridge_manifest = load_json(output / "flatdict_bridge/run_manifest.json") if (output / "flatdict_bridge/run_manifest.json").is_file() else {}
            manifest["classification"] = bridge_manifest.get("classification") or "flatdict_wheel_build_failed"
            raise RuntimeError("flatdict F0-F3 bridge failed")
        bridge_manifest = load_json(output / "flatdict_bridge/run_manifest.json")
        for stage in ("F0", "F1", "F2", "F3"):
            manifest["stages"][stage] = bridge_manifest["gates"][stage]
        wheel = Path(bridge_manifest["wheel"]).resolve()
        wheelhouse = wheel.parent

        constraints = output / "protected_baseline/protected_constraints.txt"
        constraints.write_text(constraints_text(config), encoding="utf-8")
        plans = {
            "P0A": ([config["install"]["isaaclab_requirement"]], config["install"]["isaac_extra_index_url"], "isaac"),
            "P0B": (list(config["install"]["tacex_dependencies"]), None, "tacex_dependencies"),
        }
        protected = list(required_runtime_versions(config))
        plan_reports: dict[str, dict[str, Any]] = {}
        for stage, (requirements, extra_index, name) in plans.items():
            report_path = output / f"dependency_plan/{name}_report.json"
            process, report = dry_run(
                name=name, python=python, source=source, root=output / "dependency_plan",
                constraints=constraints, wheelhouse=wheelhouse, report=report_path,
                requirements=requirements, extra_index=extra_index, environment=environment,
                timeout=float(config["timeouts_seconds"]["pip_dry_run"]),
            )
            if not process_ok(process) or not report_path.is_file():
                manifest["stages"][stage] = "failed"
                manifest["classification"] = "additional_legacy_sdist_resolution_failure" if stage == "P0B" else "validated_r08_baseline_incompatible_with_isaac_resolution"
                raise RuntimeError(f"{stage} dry-run did not produce a usable JSON report")
            protected_audit = audit_pip_report(report, protected).to_dict()
            wheel_audit = audit_report_uses_wheel(report, wheel)
            plan_reports[stage] = report
            summary = {"passed": protected_audit["success"] and wheel_audit["success"], "process": process, "protected": protected_audit, "flatdict": wheel_audit}
            write_json(output / f"dependency_plan/{name}_summary.json", summary)
            if not summary["passed"]:
                manifest["stages"][stage] = "failed"
                manifest["classification"] = "validated_r08_baseline_incompatible_with_isaac_resolution"
                raise RuntimeError(f"{stage} protected dependency plan failed")
            manifest["stages"][stage] = "passed"
        write_json(output / "dependency_plan/summary.json", {"passed": True, "stages": {key: manifest["stages"][key] for key in ("P0A", "P0B")}})

        before_flatdict = target_packages
        i0a = managed(
            [str(python), "-m", "pip", "install", "--no-index", "--no-deps", str(wheel)],
            cwd=source, output_root=output / "flatdict_install", name="install", timeout_seconds=float(config["timeouts_seconds"]["dependency_install"]), environment=environment,
        )
        require_process("I0A flatdict wheel install", i0a)
        after_flatdict = package_probe(python=python, source=source, root=output / "flatdict_install", name="packages_after", environment=environment, timeout=float(config["timeouts_seconds"]["package_check"]))
        flatdict_diff = package_diff(before_flatdict["all_distributions"], after_flatdict["all_distributions"], allowed_additions={"flatdict": "4.0.1"})
        after_flatdict_binaries = protected_binary_state(curobo, prefix)
        flatdict_binary_diff = compare_protected_binaries(before_binaries, after_flatdict_binaries)
        i0a_summary = {"passed": flatdict_diff["success"] and flatdict_binary_diff["success"] and after_flatdict["distributions"]["flatdict"] == "4.0.1", "package_diff": flatdict_diff, "protected_binary_diff": flatdict_binary_diff, "packages": after_flatdict}
        write_json(output / "flatdict_install/summary.json", i0a_summary)
        if not i0a_summary["passed"]:
            manifest["classification"] = "flatdict_install_unexpected_environment_change"
            raise RuntimeError("I0A changed more than flatdict 4.0.1")
        manifest["stages"]["I0A"] = "passed"

        actual_report = output / "install_isaac/actual_install_report.json"
        isaac_command = [
            str(python), "-m", "pip", "install", "--report", str(actual_report),
            "--constraint", str(constraints), "--find-links", str(wheelhouse), "--only-binary=flatdict",
            config["install"]["isaaclab_requirement"], "--extra-index-url", config["install"]["isaac_extra_index_url"],
        ]
        manifest["isaac_actual_install_invocations"] = 1
        write_json(output / "run_manifest.json", manifest)
        i0b = managed(isaac_command, cwd=source, output_root=output / "install_isaac", name="install", timeout_seconds=float(config["timeouts_seconds"]["isaac_install"]), environment=environment)
        if not process_ok(i0b):
            manifest["classification"] = "isaac51_package_install_failed"
            raise RuntimeError("I0B Isaac Sim/Lab install failed")
        if not actual_report.is_file():
            manifest["classification"] = "isaac51_package_install_failed"
            raise RuntimeError("I0B did not produce actual install report")
        actual_protected = audit_pip_report(load_pip_report(actual_report), protected).to_dict()
        after_isaac = package_probe(python=python, source=source, root=output / "install_isaac", name="packages_after", environment=environment, timeout=float(config["timeouts_seconds"]["package_check"]))
        after_audit = audit_installed_versions(after_isaac["distributions"], required_runtime_versions(config))
        after_isaac_binaries = protected_binary_state(curobo, prefix)
        isaac_binary_diff = compare_protected_binaries(before_binaries, after_isaac_binaries)
        versions_ok = after_isaac["distributions"].get("isaaclab") == "2.3.0" and str(after_isaac["distributions"].get("isaacsim", "")).startswith("5.1.")
        i0b_summary = {"passed": actual_protected["success"] and after_audit["success"] and isaac_binary_diff["success"] and versions_ok, "protected_report_audit": actual_protected, "protected_versions": after_audit, "protected_binary_diff": isaac_binary_diff, "versions_ok": versions_ok, "packages": after_isaac}
        write_json(output / "install_isaac/summary.json", i0b_summary)
        if not i0b_summary["passed"]:
            manifest["classification"] = "protected_dependency_changed_during_actual_install"
            raise RuntimeError("I0B protected package or binary identity changed")
        manifest["stages"]["I0B"] = "passed"

        before_i1 = package_probe(python=python, source=source, root=output / "install_tacex", name="packages_before", environment=environment, timeout=float(config["timeouts_seconds"]["package_check"]))
        deps_report = output / "install_tacex/dependencies_report.json"
        dependency_install = managed(
            [str(python), "-m", "pip", "install", "--report", str(deps_report), "--constraint", str(constraints), "--find-links", str(wheelhouse), "--only-binary=flatdict", *config["install"]["tacex_dependencies"]],
            cwd=source, output_root=output / "install_tacex/dependencies", name="install", timeout_seconds=float(config["timeouts_seconds"]["dependency_install"]), environment=environment,
        )
        require_process("I1 TacEx dependencies", dependency_install)
        if not deps_report.is_file() or not audit_pip_report(load_pip_report(deps_report), protected).success:
            manifest["classification"] = "protected_dependency_changed_during_actual_install"
            raise RuntimeError("I1 dependency install report changed a protected package")
        editables = [str(source / item) for item in config["install"]["vendored_editables"]]
        editable_command = [str(python), "-m", "pip", "install", "--no-deps"]
        for item in editables:
            editable_command.extend(["-e", item])
        editables_result = managed(editable_command, cwd=source, output_root=output / "install_tacex/editables", name="install", timeout_seconds=float(config["timeouts_seconds"]["editable_install"]), environment=environment)
        require_process("I1 vendored TacEx core/assets", editables_result)
        after_i1 = package_probe(python=python, source=source, root=output / "install_tacex", name="packages_after", environment=environment, timeout=float(config["timeouts_seconds"]["package_check"]))
        i1_diff = audit_changes_against_plan(
            before_i1["all_distributions"], after_i1["all_distributions"],
            report_install_versions(plan_reports["P0B"]),
            additional_allowed={"tacex": "0.1.0", "tacex-assets": "0.1.0"},
        )
        write_json(output / "install_tacex/package_diff.json", i1_diff)
        if not i1_diff["success"]:
            manifest["classification"] = "protected_dependency_changed_during_actual_install"
            raise RuntimeError("I1 changed packages outside the audited P0B plan")
        manifest["stages"]["I1"] = "passed"

        pip_check = managed([str(python), "-m", "pip", "check"], cwd=source, output_root=output / "package_provenance/pip_check", name="check", timeout_seconds=float(config["timeouts_seconds"]["package_check"]), environment=environment)
        require_process("I2 pip check", pip_check)
        installed = package_probe(python=python, source=source, root=output / "package_provenance", name="installed", environment=environment, timeout=float(config["timeouts_seconds"]["package_check"]))
        origins = installed["module_origins"]
        origin_checks = {
            "tacex": bool(origins.get("tacex") and Path(origins["tacex"]).is_relative_to(source)),
            "tacex_assets": bool(origins.get("tacex_assets") and Path(origins["tacex_assets"]).is_relative_to(source)),
            "tacex_uipc": bool(origins.get("tacex_uipc") and Path(origins["tacex_uipc"]).is_relative_to(source)),
            "uipc": bool(origins.get("uipc") and Path(origins["uipc"]).is_relative_to(prefix)),
            "curobo": bool(origins.get("curobo") and Path(origins["curobo"]).is_relative_to(curobo)),
            "isaacsim": bool(origins.get("isaacsim") and Path(origins["isaacsim"]).is_relative_to(prefix)),
            "isaaclab": bool(origins.get("isaaclab") and Path(origins["isaaclab"]).is_relative_to(prefix)),
        }
        final_version_audit = audit_installed_versions(installed["distributions"], required_runtime_versions(config))
        final_binary_diff = compare_protected_binaries(before_binaries, protected_binary_state(curobo, prefix))
        i2 = {"passed": all(origin_checks.values()) and final_version_audit["success"] and final_binary_diff["success"], "origin_checks": origin_checks, "protected_versions": final_version_audit, "protected_binary_diff": final_binary_diff, "packages": installed}
        write_json(output / "package_provenance/summary.json", i2)
        if not i2["passed"]:
            manifest["classification"] = "isaac51_blackwell_source_package_mix"
            raise RuntimeError("I2 source/package provenance failed")
        manifest["stages"]["I2"] = "passed"

        run_native_suite(stage="N0", python=python, source=source, curobo=curobo, output_root=output / "native_regression", environment=environment, timeouts=config["timeouts_seconds"])
        manifest["stages"]["N0"] = "passed"
        manifest["status"] = "install_validated"
        manifest["classification"] = "native_and_package_layer_ready_for_isaac_startup"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["classification"] = manifest["classification"] or "blocked_by_external_resources"
        manifest["failure"] = {"class": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        bundle = output / "author_bundle/dependency_conflict"
        bundle.mkdir(parents=True, exist_ok=True)
        write_json(bundle / "summary.json", {"classification": manifest["classification"], "failure": manifest["failure"]})
        raise
    finally:
        finalization_errors = []
        try:
            if source_before:
                source_after = fingerprint_environment(args.conda_exe, source_env)
                source_comparison = compare_fingerprints(source_before, source_after)
                write_json(output / "environment_fingerprints/r08_after.json", source_after)
                write_json(output / "environment_fingerprints/r08_comparison.json", source_comparison)
                manifest["r08_unchanged"] = source_comparison["unchanged"]
                if manifest.get("status") == "install_validated" and not source_comparison["unchanged"]:
                    manifest["status"] = "failed"
                    manifest["classification"] = "r09_resume_precondition_failed"
        except BaseException as exc:
            finalization_errors.append({"stage": "r08_fingerprint", "error": f"{type(exc).__name__}: {exc}"})
        try:
            cleanup = managed_cleanup_summary(output)
            write_json(output / "final_resources/managed_cleanup.json", cleanup)
            manifest["managed_cleanup_complete"] = cleanup["all_records_cleanup_complete"] and not cleanup["cleanup_residual"]
            final_inotify = collect_inotify_inventory()
            final_gpu = collect_gpu_inventory()
            final_processes = attach_gpu_usage(collect_process_inventory(related_roots=(REPO_ROOT, source, curobo, output)), final_gpu)
            restore = build_restore_ready(original_instances=128, original_watches=65536, inotify_inventory=final_inotify, process_inventory=final_processes, gpu_inventory=final_gpu)
            restore["sysctl_restore_performed_by_runner"] = False
            write_json(output / "restore_ready.json", restore)
            write_json(output / "final_resources/inventory.json", {"inotify": final_inotify, "gpu": final_gpu, "processes": final_processes})
            if manifest.get("status") == "install_validated" and not manifest["managed_cleanup_complete"]:
                manifest["status"] = "failed"
                manifest["classification"] = "cleanup_incomplete"
        except BaseException as exc:
            finalization_errors.append({"stage": "cleanup", "error": f"{type(exc).__name__}: {exc}"})
        manifest["finalization_errors"] = finalization_errors
        manifest["ended_at"] = utc_now()
        write_json(output / "run_manifest.json", manifest)
        write_json(output / "summary.json", manifest)


if __name__ == "__main__":
    main()
