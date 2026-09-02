#!/usr/bin/env python3
"""Clone R0.8 and install the reviewed Isaac51 simulator layer for R0.9."""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.univtac.resume_blackwell_native_bridge import managed_cleanup_summary
from scripts.univtac.run_isaac51_runtime_gates import compare_fingerprints, fingerprint_environment
from sim.envs.univtac.isaac51_blackwell_runtime import (
    clean_runtime_environment,
    load_json,
    managed,
    process_ok,
    require_process,
    run_capture,
    sha256_file,
)
from sim.envs.univtac.resource_sanitation import (
    attach_gpu_usage,
    collect_gpu_inventory,
    collect_inotify_inventory,
    collect_process_inventory,
    utc_now,
    write_json,
)
from sim.envs.univtac.simulator_install_contract import (
    RUNTIME_VARIANT,
    audit_installed_versions,
    audit_pip_report,
    clone_command,
    constraints_text,
    load_pip_report,
    required_runtime_versions,
    validate_config,
)
from sim.envs.univtac.source_backport_contract import (
    BASE_COMMIT,
    TARGET_FILE,
    UPSTREAM_REFERENCE_COMMIT,
    validate_semantic_backport,
)


def git(path: Path, *arguments: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=path, check=False, capture_output=True, text=True
    )
    if check and completed.returncode != 0:
        raise RuntimeError(f"git {' '.join(arguments)} failed in {path}: {completed.stderr.strip()}")
    return completed.stdout.strip()


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(part) for part in text.split(".")[:3])


def _available_ram_gib() -> float:
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / (1024**2)
    return 0.0


def preflight(config: Mapping[str, Any], roots: tuple[Path, ...], output_root: Path) -> dict[str, Any]:
    inotify = collect_inotify_inventory()
    gpu = collect_gpu_inventory()
    processes = attach_gpu_usage(collect_process_inventory(related_roots=roots), gpu)
    gpus = gpu.get("gpus", [])
    thresholds = config["resource_gates"]
    gpu_free_ratio = (
        gpus[0]["memory_free_mib"] / gpus[0]["memory_total_mib"] if len(gpus) == 1 else None
    )
    driver = gpus[0].get("driver_version") if len(gpus) == 1 else None
    capability_result = run_capture(
        ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"]
    )
    capability_text = str(capability_result.get("stdout", "")).strip().splitlines()
    capability = [int(part) for part in capability_text[0].split(".")] if len(capability_text) == 1 else None
    eligible = [item["pid"] for item in processes.get("processes", []) if item.get("eligible_for_cleanup")]
    active_runtime = [
        item["pid"]
        for item in processes.get("processes", [])
        if item.get("matched_reason", {}).get("runtime_entrypoints")
        and not item.get("is_ancestor_of_current_process")
        and item.get("state") != "Z"
    ]
    checks = {
        "driver": driver is not None and _version_tuple(str(driver)) >= _version_tuple(str(thresholds["minimum_driver"])),
        "single_gpu": len(gpus) == 1,
        "compute_capability": capability == list(thresholds["required_compute_capability"]),
        "gpu_free": gpu_free_ratio is not None and gpu_free_ratio >= float(thresholds["minimum_gpu_free_ratio"]),
        "ram": _available_ram_gib() >= float(thresholds["minimum_available_ram_gib"]),
        "disk": shutil.disk_usage(output_root.parent).free / (1024**3) >= float(thresholds["minimum_disk_free_gib"]),
        "inotify_instances": inotify.get("max_user_instances") == int(thresholds["required_inotify_instances"]),
        "inotify_watches": inotify.get("max_user_watches") == int(thresholds["required_inotify_watches"]),
        "inotify_instance_usage": inotify.get("instance_usage_ratio") is not None and inotify["instance_usage_ratio"] < float(thresholds["maximum_inotify_usage_ratio"]),
        "inotify_watch_usage": inotify.get("watch_count_reliable") is True and inotify.get("watch_usage_ratio") is not None and inotify["watch_usage_ratio"] < float(thresholds["maximum_inotify_usage_ratio"]),
        "no_stale_project_runtime": not eligible,
        "no_parallel_runtime": not active_runtime,
    }
    return {
        "captured_at": utc_now(), "passed": all(checks.values()), "checks": checks,
        "driver_version": driver, "compute_capability": capability,
        "gpu_free_ratio": gpu_free_ratio, "available_ram_gib": _available_ram_gib(),
        "disk_free_gib": shutil.disk_usage(output_root.parent).free / (1024**3),
        "eligible_stale_pids": eligible, "active_runtime_pids": active_runtime,
        "inotify": inotify, "gpu": gpu, "processes": processes,
        "sysctl_modified_by_runner": False,
    }


def source_manifest(source: Path, curobo: Path) -> dict[str, Any]:
    if git(source, "rev-parse", "HEAD") != "371fac67917307026be8f00869fcc1b61c623a9f":
        raise RuntimeError("pinned Isaac51 checkout commit mismatch")
    if git(source, "status", "--short", "--untracked-files=no"):
        raise RuntimeError("pinned Isaac51 checkout has tracked changes")
    if git(curobo, "rev-parse", "HEAD") != BASE_COMMIT:
        raise RuntimeError("derived cuRobo checkout base commit mismatch")
    changed = git(curobo, "diff", "--name-only", "--").splitlines()
    if changed != [str(TARGET_FILE)]:
        raise RuntimeError("derived cuRobo checkout must contain exactly the authorized one-file backport")
    before = git(curobo, "show", f"{BASE_COMMIT}:{TARGET_FILE}") + "\n"
    semantic = validate_semantic_backport(before, (curobo / TARGET_FILE).read_text(encoding="utf-8"))
    if git(curobo, "cat-file", "-t", UPSTREAM_REFERENCE_COMMIT) != "commit":
        raise RuntimeError("cuRobo upstream backport reference is unavailable")
    return {
        "univtac": {"root": str(source), "head": git(source, "rev-parse", "HEAD"), "tracked_clean": True},
        "curobo": {
            "root": str(curobo), "base_commit": git(curobo, "rev-parse", "HEAD"),
            "changed_files": changed, "semantic_backport": semantic,
            "target_sha256": sha256_file(curobo / TARGET_FILE),
        },
    }


def run_package_probe(
    *, python: Path, cwd: Path, root: Path, environment: Mapping[str, str], timeout: float
) -> tuple[dict[str, Any], dict[str, Any]]:
    result_path = root / "package_probe.json"
    process = managed(
        [str(python), str(REPO_ROOT / "scripts/univtac/probe_isaac51_packages.py"), "--output", str(result_path)],
        cwd=cwd, output_root=root, name="package_probe", timeout_seconds=timeout, environment=environment,
    )
    require_process("package probe", process)
    return process, load_json(result_path)


def run_native_suite(
    *, stage: str, python: Path, source: Path, curobo: Path, output_root: Path,
    environment: Mapping[str, str], timeouts: Mapping[str, Any]
) -> dict[str, Any]:
    root = output_root / stage.lower()
    if root.exists():
        raise FileExistsError(f"native gate output already exists: {root}")
    root.mkdir(parents=True)
    commands: list[tuple[str, list[str], float]] = [
        ("torch", [str(python), str(REPO_ROOT / "scripts/univtac/probe_blackwell_torch.py"), "--output", str(root / "torch/probe.json")], float(timeouts["native_torch"])),
        ("warp", [str(python), str(REPO_ROOT / "scripts/univtac/probe_warp_torch_interop.py"), "--output", str(root / "warp/probe.json")], float(timeouts["native_warp"])),
        ("imports", [str(python), str(REPO_ROOT / "scripts/univtac/probe_curobo_imports.py"), "--expected-root", str(curobo), "--output", str(root / "imports/probe.json")], float(timeouts["native_imports"])),
        ("kinematics", [str(python), str(REPO_ROOT / "scripts/univtac/run_official_curobo_example.py"), "--example", str(curobo / "examples/kinematics_example.py"), "--expected-root", str(curobo), "--expected-extension", "kinematics_fused_cu", "--output", str(root / "kinematics/result.json")], float(timeouts["native_kinematics"])),
        ("ik", [str(python), str(REPO_ROOT / "scripts/univtac/run_official_curobo_example.py"), "--example", str(curobo / "examples/ik_example.py"), "--expected-root", str(curobo), "--expected-extension", "lbfgs_step_cu", "--expected-extension", "kinematics_fused_cu", "--expected-extension", "line_search_cu", "--expected-extension", "tensor_step_cu", "--expected-extension", "geom_cu", "--output", str(root / "ik/result.json")], float(timeouts["native_ik"])),
        ("collision_official", [str(python), str(REPO_ROOT / "scripts/univtac/run_official_curobo_example.py"), "--example", str(curobo / "examples/collision_check_example.py"), "--expected-root", str(curobo), "--expected-extension", "geom_cu", "--output", str(root / "collision_official/result.json")], float(timeouts["native_collision"])),
        ("collision", [str(python), str(REPO_ROOT / "scripts/univtac/probe_curobo_collision_finite.py"), "--expected-root", str(curobo), "--output", str(root / "collision/probe.json")], float(timeouts["native_collision"])),
        ("libuipc", [str(python), str(REPO_ROOT / "scripts/univtac/probe_libuipc_core.py"), "--workspace", str(root / "libuipc/workspace"), "--output", str(root / "libuipc/probe.json"), "--events", str(root / "libuipc/events.jsonl")], float(timeouts["native_libuipc"])),
    ]
    results: dict[str, Any] = {}
    for name, command, timeout in commands:
        gate_root = root / name
        process = managed(command, cwd=source if name == "libuipc" else curobo, output_root=gate_root, name="probe", timeout_seconds=timeout, environment=environment)
        results[name] = {"passed": process_ok(process), "process": process}
        write_json(root / f"{name}_result.json", results[name])
        require_process(f"{stage} {name}", process)
    summary = {"stage": stage, "passed": all(item["passed"] for item in results.values()), "gates": results}
    write_json(root / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--curobo-checkout", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--conda-exe", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    source = args.source_checkout.resolve()
    curobo = args.curobo_checkout.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / "run_manifest.json").exists():
        raise FileExistsError("R0.9 install manifest already exists; refusing an ambiguous rerun")
    base = subprocess.run([str(args.conda_exe), "info", "--base"], check=True, capture_output=True, text=True).stdout.strip()
    source_env = config["environment"]["clone_from"]
    target_env = config["environment"]["conda_name"]
    source_prefix = Path(base) / "envs" / source_env
    prefix = Path(base) / "envs" / target_env
    marker_path = prefix / ".univtac-r09-provenance.json"
    manifest: dict[str, Any] = {
        "schema_version": "openeta.univtac.isaac51_blackwell_install_run.v1",
        "runtime_variant": RUNTIME_VARIANT,
        **config["claims"],
        "started_at": utc_now(), "status": "running", "classification": None,
        "source_environment": source_env, "target_environment": target_env,
        "official_install_script_executed": False, "planner_modified": False,
        "task_modified": False, "constraint_modified": False, "sysctl_modified": False,
        "isaac_started": False, "stages": {},
    }
    write_json(output / "run_manifest.json", manifest)
    source_before: dict[str, Any] = {}
    try:
        roots = (REPO_ROOT, source, curobo, output)
        resource = preflight(config, roots, output)
        write_json(output / "resource_preflight.json", resource)
        if not resource["passed"]:
            manifest["classification"] = "driver_gate_not_met"
            raise RuntimeError("R0.9 resource/driver preflight failed")
        write_json(output / "source_manifest.json", source_manifest(source, curobo))
        if not source_prefix.is_dir():
            raise FileNotFoundError(f"source R0.8 environment is absent: {source_prefix}")
        if prefix.exists():
            classification = "target_environment_preexists_unknown"
            if marker_path.is_file():
                classification = "target_environment_preexists_from_prior_run"
            manifest["classification"] = classification
            raise FileExistsError(f"target environment already exists: {prefix}")
        source_before = fingerprint_environment(args.conda_exe, source_env)
        write_json(output / "environment/source_before.json", source_before)
        if not source_before.get("success"):
            raise RuntimeError("could not fingerprint protected R0.8 environment")

        clone = managed(
            clone_command(args.conda_exe, source_env, target_env), cwd=source,
            output_root=output / "environment_clone", name="clone",
            timeout_seconds=float(config["timeouts_seconds"]["environment_clone"]),
            environment={**os.environ, "PYTHONNOUSERSITE": "1"},
        )
        if not process_ok(clone):
            manifest["classification"] = "environment_clone_failed"
        require_process("R0.9 environment clone", clone)
        marker = {
            "schema_version": "openeta.univtac.r09_environment_provenance.v1",
            "created_at": utc_now(), "clone_from": source_env,
            "target_environment": target_env, "runtime_variant": RUNTIME_VARIANT,
            "source_fingerprint": source_before,
        }
        write_json(marker_path, marker)
        write_json(output / "environment/provenance_marker.json", marker)
        python = prefix / "bin/python"
        environment, wrappers = clean_runtime_environment(prefix, source, config["runtime"]["gpu"])
        write_json(output / "environment/compiler_wrappers.json", wrappers)
        e0 = run_native_suite(stage="E0", python=python, source=source, curobo=curobo, output_root=output, environment=environment, timeouts=config["timeouts_seconds"])
        manifest["stages"]["E0"] = "passed"
        write_json(output / "run_manifest.json", manifest)

        _, package_before = run_package_probe(
            python=python, cwd=source, root=output / "p0/baseline",
            environment=environment, timeout=float(config["timeouts_seconds"]["package_check"]),
        )
        version_audit = audit_installed_versions(
            package_before["distributions"], required_runtime_versions(config)
        )
        curobo_source_ok = package_before.get("nvidia_curobo_editable_source") == str(curobo)
        version_audit["curobo_editable_source_required"] = str(curobo)
        version_audit["curobo_editable_source_observed"] = package_before.get("nvidia_curobo_editable_source")
        version_audit["curobo_editable_source_ok"] = curobo_source_ok
        write_json(output / "p0/protected_baseline_audit.json", version_audit)

        constraints = output / "install/protected_constraints.txt"
        constraints.parent.mkdir(parents=True, exist_ok=True)
        constraints.write_text(constraints_text(config), encoding="utf-8")
        reports = {
            "isaac": output / "p0/isaac_dry_run_report.json",
            "tacex_dependencies": output / "p0/tacex_dependencies_dry_run_report.json",
        }
        dry_commands = {
            "isaac": [str(python), "-m", "pip", "install", "--dry-run", "--report", str(reports["isaac"]), "--constraint", str(constraints), config["install"]["isaaclab_requirement"], "--extra-index-url", config["install"]["isaac_extra_index_url"]],
            "tacex_dependencies": [str(python), "-m", "pip", "install", "--dry-run", "--report", str(reports["tacex_dependencies"]), "--constraint", str(constraints), *config["install"]["tacex_dependencies"]],
        }
        resolution_audits: dict[str, Any] = {}
        protected_names = list(required_runtime_versions(config))
        for name, command in dry_commands.items():
            result = managed(command, cwd=source, output_root=output / f"p0/{name}", name="dry_run", timeout_seconds=float(config["timeouts_seconds"]["pip_dry_run"]), environment=environment)
            if not process_ok(result) or not reports[name].is_file():
                resolution_audits[name] = {"success": False, "process": result, "report_present": reports[name].is_file()}
            else:
                resolution_audits[name] = {**audit_pip_report(load_pip_report(reports[name]), protected_names).to_dict(), "process": result, "report_present": True}
        p0_passed = bool(version_audit["success"] and curobo_source_ok and all(item.get("success") for item in resolution_audits.values()))
        p0 = {"passed": p0_passed, "baseline": version_audit, "resolutions": resolution_audits}
        write_json(output / "p0/summary.json", p0)
        if not p0_passed:
            manifest["classification"] = "protected_dependency_resolution_conflict"
            raise RuntimeError("P0 protected dependency resolution conflict; install was not started")
        manifest["stages"]["P0"] = "passed"
        write_json(output / "run_manifest.json", manifest)

        install_report = output / "i0/install_report.json"
        i0 = managed(
            [str(python), "-m", "pip", "install", "--report", str(install_report), "--constraint", str(constraints), config["install"]["isaaclab_requirement"], "--extra-index-url", config["install"]["isaac_extra_index_url"]],
            cwd=source, output_root=output / "i0", name="install", timeout_seconds=float(config["timeouts_seconds"]["isaac_install"]), environment=environment,
        )
        if not process_ok(i0):
            manifest["classification"] = "isaac51_package_install_failed"
        require_process("I0 Isaac51 package install", i0)
        manifest["stages"]["I0"] = "passed"

        deps = managed(
            [str(python), "-m", "pip", "install", "--constraint", str(constraints), *config["install"]["tacex_dependencies"]],
            cwd=source, output_root=output / "i1/dependencies", name="install", timeout_seconds=float(config["timeouts_seconds"]["dependency_install"]), environment=environment,
        )
        require_process("I1 reviewed TacEx dependencies", deps)
        editables = [str(source / item) for item in config["install"]["vendored_editables"]]
        editable_command = [str(python), "-m", "pip", "install", "--no-deps", "--no-build-isolation"]
        for item in editables:
            editable_command.extend(["-e", item])
        editable = managed(
            editable_command, cwd=source, output_root=output / "i1/editables", name="install",
            timeout_seconds=float(config["timeouts_seconds"]["editable_install"]), environment=environment,
        )
        require_process("I1 vendored TacEx editables", editable)
        manifest["stages"]["I1"] = "passed"

        pip_check = managed(
            [str(python), "-m", "pip", "check"], cwd=source, output_root=output / "i2/pip_check", name="check",
            timeout_seconds=float(config["timeouts_seconds"]["package_check"]), environment=environment,
        )
        require_process("I2 pip check", pip_check)
        _, package_after = run_package_probe(
            python=python, cwd=source, root=output / "i2/packages", environment=environment,
            timeout=float(config["timeouts_seconds"]["package_check"]),
        )
        after_audit = audit_installed_versions(package_after["distributions"], required_runtime_versions(config))
        origins = package_after["module_origins"]
        prefix_origins = {
            name: bool(origins.get(name) and Path(str(origins[name])).is_relative_to(prefix))
            for name in ("torch", "torchvision", "warp", "uipc", "isaacsim", "isaaclab")
        }
        source_origins = {
            name: bool(origins.get(name) and Path(str(origins[name])).is_relative_to(source))
            for name in ("tacex", "tacex_assets")
        }
        curobo_origin_ok = bool(
            origins.get("curobo") and Path(str(origins["curobo"])).is_relative_to(curobo)
        )
        source_mix = not (
            package_after.get("nvidia_curobo_editable_source") == str(curobo)
            and all(prefix_origins.values()) and all(source_origins.values()) and curobo_origin_ok
        )
        i2_summary = {
            "passed": after_audit["success"] and not source_mix,
            "version_audit": after_audit, "source_package_mix": source_mix,
            "prefix_module_origins": prefix_origins, "source_module_origins": source_origins,
            "curobo_module_origin_ok": curobo_origin_ok, "packages": package_after,
        }
        write_json(output / "i2/summary.json", i2_summary)
        if not i2_summary["passed"]:
            manifest["classification"] = "isaac51_blackwell_source_package_mix" if source_mix else "protected_dependency_resolution_conflict"
            raise RuntimeError("I2 package integrity failed")
        manifest["stages"]["I2"] = "passed"

        run_native_suite(stage="N0", python=python, source=source, curobo=curobo, output_root=output, environment=environment, timeouts=config["timeouts_seconds"])
        manifest["stages"]["N0"] = "passed"
        manifest["status"] = "install_validated"
        manifest["classification"] = "native_and_package_layer_ready_for_isaac_startup"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["classification"] = manifest["classification"] or "blocked_by_external_resources"
        manifest["failure"] = {"class": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        write_json(output / "failure.json", manifest["failure"])
        raise
    finally:
        finalization_errors: list[dict[str, str]] = []
        if source_before:
            try:
                source_after = fingerprint_environment(args.conda_exe, source_env)
                write_json(output / "environment/source_after.json", source_after)
                source_comparison = compare_fingerprints(source_before, source_after)
                write_json(output / "environment/source_comparison.json", source_comparison)
                manifest["source_environment_unchanged"] = source_comparison["unchanged"]
                if manifest.get("status") == "install_validated" and not source_comparison["unchanged"]:
                    manifest["status"] = "failed"
                    manifest["classification"] = "protected_source_environment_changed"
            except BaseException as exc:
                finalization_errors.append({"stage": "source_environment", "error": f"{type(exc).__name__}: {exc}"})
        try:
            cleanup = managed_cleanup_summary(output)
            write_json(output / "final_cleanup.json", cleanup)
            manifest["managed_cleanup_complete"] = cleanup["all_records_cleanup_complete"] and not cleanup["cleanup_residual"]
            if manifest.get("status") == "install_validated" and not manifest["managed_cleanup_complete"]:
                manifest["status"] = "failed"
                manifest["classification"] = "cleanup_incomplete"
        except BaseException as exc:
            finalization_errors.append({"stage": "managed_cleanup", "error": f"{type(exc).__name__}: {exc}"})
        manifest["finalization_errors"] = finalization_errors
        manifest["ended_at"] = utc_now()
        write_json(output / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
