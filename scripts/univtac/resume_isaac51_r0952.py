#!/usr/bin/env python3
"""Run R0.9.5.2 through the all-wheel offline-resolution stop point."""

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
from scripts.univtac.resume_isaac51_blackwell_install import (
    compare_protected_binaries,
    package_probe,
    protected_binary_state,
)
from scripts.univtac.run_isaac51_runtime_gates import compare_fingerprints, fingerprint_environment
from sim.envs.univtac.isaac51_blackwell_runtime import (
    clean_runtime_environment,
    load_json,
    managed,
    process_ok,
)
from sim.envs.univtac.offline_wheelhouse_contract import (
    EXPECTED_DERIVED_LOCK_SHA256,
    EXPECTED_HEAD,
    EXPECTED_SOURCE_LOCK_SHA256,
    EXPECTED_SOURCE_REPORT_SHA256,
    EXPECTED_TRANSFORM_LOCK_SHA256,
    classify_records,
    compare_offline_resolution,
    offline_dry_run_command,
    sha256_file,
    validate_config,
    validate_existing_cache,
    validate_offline_command,
    validate_system_curl_provenance,
)
from sim.envs.univtac.resumable_download_contract import validate_lock_pair, validate_wheel
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
    required_runtime_versions,
    validate_config as validate_runtime_config,
    validate_provenance_marker,
)


STAGES = ("R0", "T0", "T1", "C0", "M0", "M1", "D0", "D1", "D2", "V0", "O0")


def git(path: Path, *arguments: str) -> str:
    result = subprocess.run(["git", *arguments], cwd=path, check=False, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip())
    return result.stdout.strip()


def remote_head() -> str:
    result = subprocess.run(
        ["git", "ls-remote", "private", "refs/heads/tactile-agent-for-univtac"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode or not result.stdout.strip():
        raise RuntimeError(f"remote HEAD unavailable: {result.stderr.strip()}")
    return result.stdout.split()[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-config", type=Path, required=True)
    parser.add_argument("--wheelhouse-config", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--curobo-checkout", type=Path, required=True)
    parser.add_argument("--r095-output", type=Path, required=True)
    parser.add_argument("--r0951-output", type=Path, required=True)
    parser.add_argument("--r0952-output", type=Path, required=True)
    parser.add_argument("--wheel-cache-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--conda-exe", type=Path, required=True)
    args = parser.parse_args()

    runtime = yaml.safe_load(args.runtime_config.read_text(encoding="utf-8"))
    wheelhouse_config = yaml.safe_load(args.wheelhouse_config.read_text(encoding="utf-8"))
    validate_runtime_config(runtime)
    validate_config(wheelhouse_config)
    source = args.source_checkout.resolve()
    curobo = args.curobo_checkout.resolve()
    r095 = args.r095_output.resolve()
    r0951 = args.r0951_output.resolve()
    r0952 = args.r0952_output.resolve()
    cache = args.wheel_cache_root.resolve()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, mode=0o750)
    output.chmod(0o750)

    conda_base = Path(
        subprocess.run(
            [str(args.conda_exe), "info", "--base"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    prefix = conda_base / "envs" / wheelhouse_config["environment"]
    python = prefix / "bin/python"
    environment, _ = clean_runtime_environment(prefix, source, runtime["runtime"]["gpu"])
    manifest: dict[str, Any] = {
        "schema_version": "openeta.univtac.r0952r1_offline_wheelhouse.v1",
        "runtime_variant": runtime["runtime_variant"],
        "installation_method_label": wheelhouse_config["installation_method_label"],
        **runtime["claims"],
        "status": "running",
        "classification": None,
        "started_at": utc_now(),
        "stages": {stage: "not_run_due_to_gate" for stage in STAGES},
        "wheel_cache_root": str(cache),
        "total_isaac_install_invocations": 1,
        "authorized_offline_retry_invocations": 0,
        "third_install_invocation": False,
        "actual_install_executed": False,
        "isaac_started": False,
        "task_started": False,
        "agent_started": False,
        "transport_remediation": "system_curl_with_child_ld_library_path_unset",
        "runtime_environment_changed": False,
        "artifact_selection_changed": False,
    }
    write_json(output / "run_manifest.json", manifest)
    r08_before: dict[str, Any] = {}
    r09_before: dict[str, Any] = {}
    try:
        resources = preflight(runtime, (ROOT, source, curobo, r095, r0951, r0952, cache, output), output)
        write_json(output / "resume_preflight/resources.json", resources)
        local_head = git(ROOT, "rev-parse", "HEAD")
        tracking_head = git(ROOT, "rev-parse", "private/tactile-agent-for-univtac")
        remote_error = None
        try:
            observed_remote = remote_head()
        except RuntimeError as exc:
            observed_remote = None
            remote_error = str(exc)
        delivery = {
            "local_head": local_head,
            "tracking_head": tracking_head,
            "remote_head": observed_remote,
            "remote_error": remote_error,
            "expected_head": EXPECTED_HEAD,
            "fast_forward_delivery_confirmed": local_head == tracking_head == observed_remote == EXPECTED_HEAD,
            "force_push_used": False,
        }
        write_json(output / "git_delivery/summary.json", delivery)

        prior = load_json(r095 / "run_manifest.json")
        prior_bridge = load_json(r0951 / "legacy_sdist_bridge/summary.json")
        prior_r0951 = load_json(r0951 / "run_manifest.json")
        prior_r0952 = load_json(r0952 / "run_manifest.json")
        source_lock_path = r095 / "artifact_lock/artifact_lock.private.json"
        source_lock = load_json(source_lock_path)
        derived_lock_path = r0951 / "legacy_sdist_bridge/derived_install_artifact_lock/derived_install_artifact_lock.private.json"
        derived_lock = load_json(derived_lock_path)
        transform_path = r0951 / "legacy_sdist_bridge/source_to_wheel_transformations/source_to_wheel_transformations.json"
        transformations = load_json(transform_path)
        source_report_path = Path(source_lock["source_report"]["path"])
        constraints_path = Path(source_lock["source_report"]["constraints_path"])
        source_report = load_json(source_report_path)
        validate_lock_pair(source_lock, derived_lock)
        groups = classify_records(derived_lock["records"])

        current = package_probe(
            python=python,
            source=source,
            root=output / "resume_preflight",
            name="packages",
            environment=environment,
            timeout=600,
        )
        binaries = protected_binary_state(curobo, prefix)
        prior_binaries = load_json(r0951 / "resume_preflight/summary.json")["protected_binaries"]
        required = required_runtime_versions(runtime)
        r08_before = fingerprint_environment(args.conda_exe, runtime["environment"]["clone_from"])
        r09_before = fingerprint_environment(args.conda_exe, runtime["environment"]["conda_name"])
        downloader_source = (ROOT / "scripts/univtac/download_isaac_wheelhouse.py").read_text(encoding="utf-8")
        conditions = {
            "git_delivery": delivery["fast_forward_delivery_confirmed"],
            "r095_classification": prior.get("classification") == "artifact_lock_contains_nonwheel",
            "r0951_bridge_classification": prior_bridge.get("classification") == "legacy_sdist_reproducible_wheel_bridge_validated",
            "r0952_classification": prior_r0952.get("classification") == "network_transport_preflight_failed",
            "system_curl_fix_source": 'SYSTEM_CURL = Path("/usr/bin/curl")' in downloader_source and 'environment.pop("LD_LIBRARY_PATH", None)' in downloader_source,
            "r0951_bridge_stages": all(prior_bridge.get("stages", {}).get(stage) == "passed" for stage in ("S0", "S1", "S2", "B0", "B1", "B2", "B3", "B4", "L0R")),
            "exactly_two_builds": set(prior_bridge.get("build_invocations", {}).values()) == {2},
            "no_third_build": all(value == 2 for value in prior_bridge.get("build_invocations", {}).values()),
            "source_report_sha256": sha256_file(source_report_path) == EXPECTED_SOURCE_REPORT_SHA256,
            "source_lock_sha256": source_lock.get("artifact_lock_sha256") == EXPECTED_SOURCE_LOCK_SHA256,
            "transform_lock_sha256": transformations.get("transformation_lock_sha256") == EXPECTED_TRANSFORM_LOCK_SHA256,
            "derived_lock_sha256": derived_lock.get("derived_install_artifact_lock_sha256") == EXPECTED_DERIVED_LOCK_SHA256,
            "derived_closure": len(groups["remote"]) == 169 and len(groups["derived"]) == 3 and len(groups["verified_local"]) == 1,
            "constraints_sha256": sha256_file(constraints_path) == wheelhouse_config["constraints_sha256"],
            "no_isaac_or_tacex": not any(name.startswith(("isaacsim", "isaaclab")) for name in current["all_distributions"]) and current["all_distributions"].get("tacex") is None and current["all_distributions"].get("tacex-assets") is None,
            "flatdict": current["all_distributions"].get("flatdict") == "4.0.1",
            "protected_versions": all(current["distributions"].get(name) == version for name, version in required.items()),
            "protected_binaries": compare_protected_binaries(prior_binaries, binaries)["success"],
            "install_accounting": prior_r0951.get("total_isaac_install_invocations") == 1 and prior_r0951.get("authorized_offline_retry_invocations") == 0 and prior_r0951.get("third_install_invocation") is False,
            "r08_unchanged": prior_r0951.get("r08_unchanged") is True,
            "source_clean": not git(source, "status", "--short", "--untracked-files=no"),
            "source_head": git(source, "rev-parse", "HEAD") == runtime["source"]["univtac_commit"],
            "vendor_diff_empty": not git(ROOT, "diff", "--", "third_party/ftp1-policy/UniVTAC"),
            "resources": resources["passed"],
            "prior_cleanup": prior_r0951.get("managed_cleanup_complete") is True,
        }
        try:
            validate_provenance_marker(
                load_json(prefix / ".univtac-r09-provenance.json"),
                source=runtime["environment"]["clone_from"],
                target=runtime["environment"]["conda_name"],
            )
            conditions["provenance_marker"] = True
        except Exception:
            conditions["provenance_marker"] = False
        cache_result = validate_existing_cache(cache, source_lock, derived_lock, wheelhouse_config)
        write_json(output / "cache_validation/summary.json", cache_result)
        write_json(
            output / "lock_validation/summary.json",
            {
                "source_report_sha256": sha256_file(source_report_path),
                "source_artifact_lock_sha256": source_lock["artifact_lock_sha256"],
                "transformation_lock_sha256": transformations["transformation_lock_sha256"],
                "derived_install_artifact_lock_sha256": derived_lock["derived_install_artifact_lock_sha256"],
                "remote_records": len(groups["remote"]),
                "derived_records": len(groups["derived"]),
                "verified_local_records": len(groups["verified_local"]),
            },
        )
        write_json(output / "resume_preflight/summary.json", {"passed": all(conditions.values()), "conditions": conditions, "packages": current, "protected_binaries": binaries})
        if not all(conditions.values()):
            manifest["stages"]["R0"] = "failed"
            manifest["classification"] = "r0952r1_resume_precondition_failed"
            raise RuntimeError("R0 failed")
        manifest["stages"]["R0"] = "passed"

        child_environment = dict(environment)
        parent_ld_library_path = child_environment.pop("LD_LIBRARY_PATH", None)
        curl_version = subprocess.run(
            ["/usr/bin/curl", "--version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=child_environment,
        )
        curl_ldd = subprocess.run(
            ["/usr/bin/ldd", "/usr/bin/curl"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
            env=child_environment,
        )
        curl_ca_path = child_environment.get("SSL_CERT_FILE", "/etc/ssl/certs/ca-certificates.crt")
        if curl_version.returncode or curl_ldd.returncode or not Path(curl_ca_path).is_file():
            manifest["classification"] = "system_curl_transport_isolation_failed"
            raise RuntimeError("system curl provenance command failed")
        curl_provenance = validate_system_curl_provenance(
            executable=Path("/usr/bin/curl"),
            version_output=curl_version.stdout + curl_version.stderr,
            ldd_output=curl_ldd.stdout + curl_ldd.stderr,
            conda_prefixes=(prefix, conda_base / "envs" / runtime["environment"]["clone_from"]),
        )
        curl_provenance.update(
            {
                "parent_ld_library_path_present": parent_ld_library_path is not None,
                "curl_child_ld_library_path": None,
                "curl_ca_path": curl_ca_path,
                "parent_environment_unchanged": environment.get("LD_LIBRARY_PATH") == parent_ld_library_path,
                "protocols": next((line.split(":", 1)[1].strip().split() for line in curl_version.stdout.splitlines() if line.startswith("Protocols:")), []),
                "features": next((line.split(":", 1)[1].strip().split() for line in curl_version.stdout.splitlines() if line.startswith("Features:")), []),
            }
        )
        write_json(output / "transport_remediation/system_curl.json", curl_provenance)
        manifest["stages"]["T0"] = "passed"

        isaaclab_record = next(record for record in derived_lock["records"] if record["name"] == "isaaclab")
        part = cache / "partial" / f"{isaaclab_record['filename']}.part"
        sidecar_path = cache / "partial" / f"{isaaclab_record['filename']}.part.json"
        sidecar = load_json(sidecar_path)
        if any(sidecar.get(key) != isaaclab_record.get(key) for key in ("url", "filename", "sha256")):
            manifest["classification"] = "existing_complete_part_validation_failed"
            raise RuntimeError("isaaclab partial sidecar identity changed")
        expected_size = sidecar.get("expected_size")
        if not isinstance(expected_size, int) or part.name != f"{isaaclab_record['filename']}.part":
            manifest["classification"] = "existing_complete_part_validation_failed"
            raise RuntimeError("isaaclab partial name or size contract changed")
        try:
            part_validation = validate_wheel(part, isaaclab_record, expected_size, check_filename=False)
        except (OSError, ValueError) as exc:
            manifest["classification"] = "existing_complete_part_validation_failed"
            raise RuntimeError(f"isaaclab complete partial validation failed: {exc}") from exc
        historical_attempts = len(sidecar.get("attempts", []))
        if historical_attempts < 1:
            manifest["classification"] = "existing_complete_part_validation_failed"
            raise RuntimeError("isaaclab partial has no historical transport attempt")
        write_json(
            output / "complete_part_validation/isaaclab.json",
            {
                **part_validation,
                "partial_filename": part.name,
                "expected_final_filename": isaaclab_record["filename"],
                "historical_transport_attempts": historical_attempts,
                "new_transport_attempts_for_this_artifact": 0,
                "recovered_complete_part": True,
                "content_check_filename": False,
                "promotion_deferred_until_canary": True,
                "partial_mode": oct(part.stat().st_mode & 0o777),
                "partial_state": "mutable_until_atomic_promotion",
            },
        )
        manifest["stages"]["T1"] = "passed"

        wheelhouse_root = output / "wheelhouse_pipeline"
        download_process = managed(
            [
                str(python),
                str(ROOT / "scripts/univtac/download_isaac_wheelhouse.py"),
                "--config", str(args.wheelhouse_config.resolve()),
                "--derived-lock", str(derived_lock_path),
                "--source-lock", str(source_lock_path),
                "--legacy-bridge-root", str(r0951 / "legacy_sdist_bridge"),
                "--environment-root", str(prefix),
                "--wheel-cache-root", str(cache),
                "--output-root", str(wheelhouse_root),
            ],
            cwd=ROOT,
            output_root=output / "wheelhouse_process",
            name="materialize",
            timeout_seconds=172800,
            environment=environment,
        )
        if not (wheelhouse_root / "run_manifest.json").is_file():
            manifest["classification"] = "wheelhouse_validation_failed"
            raise RuntimeError(
                "wheelhouse process exited before creating its manifest; "
                f"see {download_process.get('log_path')}"
            )
        wheelhouse_manifest = load_json(wheelhouse_root / "run_manifest.json")
        for stage in ("C0", "M0", "M1", "D0", "D1", "D2", "V0"):
            manifest["stages"][stage] = wheelhouse_manifest["stages"].get(stage, "not_run_due_to_gate")
        manifest["transport_attempts"] = wheelhouse_manifest.get("transport_attempts", 0)
        if not process_ok(download_process) or wheelhouse_manifest.get("status") != "completed":
            manifest["classification"] = wheelhouse_manifest.get("classification") or "blocked_by_external_resources"
            raise RuntimeError("wheelhouse materialization failed")

        manifest["stages"]["O0"] = "running"
        offline_report_path = output / "offline_resolution/offline_report.json"
        command = offline_dry_run_command(
            python,
            cache / "artifacts",
            constraints_path,
            offline_report_path,
            wheelhouse_config["requirement"],
        )
        validate_offline_command(command)
        offline_environment = dict(environment)
        offline_environment.update(
            {
                "PIP_NO_INDEX": "1",
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "PYTHONNOUSERSITE": "1",
            }
        )
        for name in ("PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
            offline_environment.pop(name, None)
        dry_run = managed(
            command,
            cwd=ROOT,
            output_root=output / "offline_resolution",
            name="dry_run",
            timeout_seconds=float(runtime["timeouts_seconds"]["pip_dry_run"]),
            environment=offline_environment,
        )
        if not process_ok(dry_run) or not offline_report_path.is_file():
            manifest["classification"] = "offline_wheelhouse_resolution_mismatch"
            raise RuntimeError("offline dry-run failed")
        offline_report = load_json(offline_report_path)
        parity = compare_offline_resolution(source_report, derived_lock, offline_report)
        write_json(output / "offline_resolution/parity.json", parity)
        if not parity["success"]:
            manifest["classification"] = "offline_wheelhouse_resolution_mismatch"
            raise RuntimeError("offline resolution parity failed")
        manifest["stages"]["O0"] = "passed"
        manifest["status"] = "completed"
        manifest["classification"] = "isaac51_offline_wheelhouse_ready"
        manifest["wheelhouse_manifest_sha256"] = wheelhouse_manifest["wheelhouse_manifest_sha256"]
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["classification"] = manifest["classification"] or "blocked_by_external_resources"
        if manifest["stages"].get("O0") == "running":
            manifest["stages"]["O0"] = "failed"
        manifest["failure"] = {"class": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        raise
    finally:
        errors = []
        try:
            if r08_before:
                comparison = compare_fingerprints(r08_before, fingerprint_environment(args.conda_exe, runtime["environment"]["clone_from"]))
                write_json(output / "environment_fingerprints/r08_comparison.json", comparison)
                manifest["r08_unchanged"] = comparison["unchanged"]
            if r09_before:
                comparison = compare_fingerprints(r09_before, fingerprint_environment(args.conda_exe, runtime["environment"]["conda_name"]))
                write_json(output / "environment_fingerprints/r09_comparison.json", comparison)
                manifest["r09_unchanged"] = comparison["unchanged"]
        except BaseException as exc:
            errors.append({"stage": "environment_fingerprints", "error": str(exc)})
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
