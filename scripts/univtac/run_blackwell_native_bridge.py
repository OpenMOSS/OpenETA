#!/usr/bin/env python3
"""Build and validate the pinned R0.8 Blackwell native bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from scripts.univtac.run_isaac51_runtime_gates import (
    clean_environment,
    compare_fingerprints,
    fingerprint_environment,
)
from sim.envs.univtac.blackwell_adaptation import (
    RUNTIME_VARIANT,
    canonical_sha256,
    classify_torch_probe,
    derive_libuipc_environment,
    load_and_validate_config,
    validate_source_baseline,
)
from sim.envs.univtac.libuipc_core_probe import classify_core_probe
from sim.envs.univtac.native_build_contract import (
    binary_architecture_classification,
    build_log_classification,
    source_tracked_clean,
    validate_native_environment,
)
from sim.envs.univtac.resource_sanitation import (
    attach_gpu_usage,
    collect_gpu_inventory,
    collect_inotify_inventory,
    collect_process_inventory,
    run_managed_process,
    utc_now,
    write_json,
)


def _run(command: list[str], *, cwd: Path | None = None, environment: Mapping[str, str] | None = None, timeout: int = 300) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=dict(environment) if environment is not None else None,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"command": command, "returncode": None, "error": f"{type(exc).__name__}: {exc}"}
    return {"command": command, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}


def _git(path: Path, *arguments: str, check: bool = True) -> str:
    result = _run(["git", *arguments], cwd=path)
    if check and result.get("returncode") != 0:
        raise RuntimeError(f"git {' '.join(arguments)} failed in {path}")
    return str(result.get("stdout", "")).strip()


def _version_tuple(text: str) -> tuple[int, ...]:
    return tuple(int(item) for item in re.findall(r"\d+", text)[:3])


def _managed(command: list[str], *, cwd: Path, root: Path, name: str, timeout: float, environment: Mapping[str, str]) -> dict[str, Any]:
    result = run_managed_process(
        command,
        cwd=cwd,
        log_path=root / "logs" / f"{name}.log",
        timeout_seconds=timeout,
        environment=environment,
    )
    write_json(root / "processes" / f"{name}.json", result)
    return result


def _process_ok(result: Mapping[str, Any]) -> bool:
    return result.get("returncode") == 0 and not result.get("timed_out") and result.get("cleanup_complete") is True


def _require_process(name: str, result: Mapping[str, Any]) -> None:
    if not _process_ok(result):
        raise RuntimeError(f"{name} failed; see {result.get('log_path')}")


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def uv_fingerprint() -> dict[str, Any]:
    result = _run([sys.executable, "-m", "pip", "freeze"], environment=clean_environment())
    content = str(result.get("stdout", ""))
    return {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_version": platform.python_version(),
        "pip_freeze_returncode": result.get("returncode"),
        "pip_freeze_sha256": _hash_text(content) if result.get("returncode") == 0 else None,
        "pip_freeze_line_count": len(content.splitlines()),
        "content_recorded": False,
    }


def native_environment(prefix: Path, config: Mapping[str, Any], vcpkg_root: Path) -> dict[str, str]:
    source = config["environment"]
    conda_gcc = prefix / "bin" / "x86_64-conda-linux-gnu-gcc"
    conda_gxx = prefix / "bin" / "x86_64-conda-linux-gnu-c++"
    environment = clean_environment(
        {
            "PATH": f"{prefix / 'bin'}:{os.environ.get('PATH', '')}",
            "LD_LIBRARY_PATH": str(prefix / "lib"),
            "CUDA_HOME": str(prefix),
            "CUDA_PATH": str(prefix),
            "CUDACXX": str(prefix / "bin" / "nvcc"),
            "CMAKE_CUDA_ARCHITECTURES": source["cuda_arch"],
            "TORCH_CUDA_ARCH_LIST": source["torch_cuda_arch_list"],
            "CMAKE_BUILD_PARALLEL_LEVEL": str(source["build_jobs"]),
            "MAX_JOBS": str(source["build_jobs"]),
            "CMAKE_TOOLCHAIN_FILE": str(vcpkg_root / "scripts/buildsystems/vcpkg.cmake"),
            "UNIVTAC_GCC12": str(conda_gcc),
            "UNIVTAC_GXX12": str(conda_gxx),
            "CC": str(config["paths"]["cc_wrapper"]),
            "CXX": str(config["paths"]["cxx_wrapper"]),
            "CUDAHOSTCXX": str(config["paths"]["cxx_wrapper"]),
        }
    )
    validate_native_environment(environment, prefix)
    return environment


def _binary_candidates(roots: list[Path], pattern: str) -> list[Path]:
    regex = re.compile(pattern, re.IGNORECASE)
    candidates = set()
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.so"):
            if regex.search(path.name):
                candidates.add(path.resolve())
    return sorted(candidates)


def _write_author_bundle(root: Path, summary: Mapping[str, Any]) -> None:
    bundle = root / "author_bundle_blackwell_native"
    bundle.mkdir(parents=True, exist_ok=True)
    write_json(bundle / "summary.json", summary)
    (bundle / "README.md").write_text(
        "# Blackwell native bridge\n\n"
        "This is a Blackwell compatibility adaptation, not the exact public UniVTAC Isaac51 installation recipe.\n\n"
        f"Runtime variant: `{RUNTIME_VARIANT}`. See `summary.json` for gate outcomes and fixed source revisions.\n",
        encoding="utf-8",
    )


def conda_meta_fingerprint(prefix: Path) -> dict[str, Any]:
    meta = prefix / "conda-meta"
    records = []
    for path in sorted(meta.glob("*.json")):
        records.append(
            {
                "name": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    history = meta / "history"
    return {
        "prefix": str(prefix.resolve()),
        "package_records": records,
        "history_sha256": hashlib.sha256(history.read_bytes()).hexdigest() if history.is_file() else None,
        "semantic_sha256": canonical_sha256(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--conda-exe", type=Path, required=True)
    parser.add_argument("--legacy-env", default="UniVTAC")
    parser.add_argument("--r07-env", default="UniVTAC-isaac51-r07")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_and_validate_config(args.config.resolve())
    source = args.source_checkout.resolve()
    output = args.output_root.resolve()
    env_name = config["environment"]["conda_name"]
    base_result = _run([str(args.conda_exe), "info", "--base"], environment=clean_environment())
    if base_result.get("returncode") != 0:
        raise RuntimeError("cannot resolve Conda base")
    prefix = Path(str(base_result["stdout"]).strip()) / "envs" / env_name
    vcpkg_root = source / ".cache" / "toolchains" / "vcpkg-sm120-r08"
    config["paths"] = {
        "cc_wrapper": str(source / "scripts/toolchains/gcc12-system-ld"),
        "cxx_wrapper": str(source / "scripts/toolchains/gxx12-system-ld"),
    }
    if args.dry_run:
        print(json.dumps({
            "runtime_variant": RUNTIME_VARIANT,
            "official_recipe_exact": False,
            "benchmark_reproduction": False,
            "source": str(source),
            "target_environment": env_name,
            "target_prefix": str(prefix),
            "gate_order": config["gate_order"],
            "isaac_sim_started": False,
            "sysctl_modified": False,
        }, indent=2, sort_keys=True))
        return
    if output.exists():
        raise FileExistsError(f"R0.8 output root must be absent: {output}")
    if prefix.exists():
        raise FileExistsError(f"R0.8 target environment already exists: {prefix}")
    output.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "schema_version": "openeta.univtac.blackwell_native_run.v1",
        "runtime_variant": RUNTIME_VARIANT,
        "official_recipe_exact": False,
        "benchmark_reproduction": False,
        "started_at": utc_now(),
        "status": "running",
        "classification": None,
        "gates": {},
        "isaac51_simulator_driver_gate": "pending_manual_upgrade_or_author_confirmation",
        "isaac_sim_started": False,
        "sysctl_modified": False,
        "inotify_restore_skipped_by_user_authorization": True,
    }
    write_json(output / "run_manifest.json", manifest)
    fingerprints_before: dict[str, Any] = {}
    try:
        head = _git(source, "rev-parse", "HEAD")
        status = _git(source, "status", "--short")
        baseline_file = source / "third_party/TacEx/UNIVTAC_BASELINE.md"
        baseline = validate_source_baseline(baseline_file.read_text(encoding="utf-8"), config)
        gpu = collect_gpu_inventory()
        compute_cap_result = _run(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader,nounits"]
        )
        compute_capability = str(compute_cap_result.get("stdout", "")).strip()
        inotify = collect_inotify_inventory()
        processes = attach_gpu_usage(
            collect_process_inventory(related_roots=(REPO_ROOT, source, output)), gpu
        )
        memory_available = int(Path("/proc/meminfo").read_text().split("MemAvailable:", 1)[1].split()[0]) * 1024
        disk_free = shutil.disk_usage(output.parent).free
        gpu_record = gpu["gpus"][0] if len(gpu.get("gpus", [])) == 1 else {}
        driver = str(gpu_record.get("driver_version", "0"))
        p0_checks = {
            "source_commit": head == config["source"]["commit"],
            "source_tracked_clean": source_tracked_clean(status),
            "source_baseline_pins": baseline["valid"],
            "gpu_sm120": compute_capability == "12.0",
            "driver_cuda128_minimum": _version_tuple(driver) >= _version_tuple(config["resource_gates"]["minimum_driver"]),
            "disk_free": disk_free >= float(config["resource_gates"]["minimum_disk_free_gib"]) * 1024**3,
            "memory_available": memory_available >= float(config["resource_gates"]["minimum_available_ram_gib"]) * 1024**3,
            "gpu_free": gpu_record and gpu_record["memory_free_mib"] / gpu_record["memory_total_mib"] >= float(config["resource_gates"]["minimum_gpu_free_ratio"]),
            "no_stale_project_process": not any(item.get("eligible_for_cleanup") for item in processes.get("processes", [])),
        }
        preflight = {
            "checks": p0_checks,
            "passed": all(p0_checks.values()),
            "source_head": head,
            "source_status": status,
            "source_baseline": baseline,
            "gpu": gpu,
            "compute_capability": compute_capability,
            "inotify": inotify,
            "memory_available_bytes": memory_available,
            "disk_free_bytes": disk_free,
            "processes": processes,
            "parent_cuda_environment": {key: os.environ.get(key) for key in ("CUDA_HOME", "CUDA_PATH", "LD_LIBRARY_PATH")},
            "inotify_values_retained_by_user_override": True,
        }
        write_json(output / "preflight" / "manifest.json", preflight)
        if not preflight["passed"]:
            manifest["classification"] = "blocked_by_external_resources"
            raise RuntimeError("P0 failed")
        fingerprints_before = {
            "legacy": fingerprint_environment(args.conda_exe, args.legacy_env),
            "r07": fingerprint_environment(args.conda_exe, args.r07_env),
            "legacy_conda_meta": conda_meta_fingerprint(prefix.parent / args.legacy_env),
            "r07_conda_meta": conda_meta_fingerprint(prefix.parent / args.r07_env),
            "openeta_uv": uv_fingerprint(),
        }
        write_json(output / "legacy_env_fingerprints" / "before.json", fingerprints_before)
        adaptation = {
            "runtime_variant": RUNTIME_VARIANT,
            "official_recipe_exact": False,
            "benchmark_reproduction": False,
            "authorized_adaptations": config["authorized_adaptations"],
            "source": config["source"],
            "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
            "config_semantic_sha256": canonical_sha256({key: value for key, value in config.items() if key != "paths"}),
        }
        write_json(output / "adaptation_manifest.json", adaptation)
        manifest["gates"]["P0"] = "passed"
        write_json(output / "run_manifest.json", manifest)

        base_env = clean_environment()
        create = _managed(
            [str(args.conda_exe), "create", "--name", env_name, "--yes", "--override-channels",
             "--channel", "https://repo.anaconda.com/pkgs/main", "--channel", "conda-forge",
             "python=3.11", "pip", "cuda-toolkit=12.8"],
            cwd=source, root=output / "torch_cu128", name="conda_create", timeout=config["timeouts_seconds"]["environment_update"], environment=base_env,
        )
        _require_process("T0 Conda create", create)
        python = prefix / "bin/python"
        tools = _managed(
            [str(python), "-m", "pip", "install", "setuptools==75.8.2", "wheel==0.42.0"],
            cwd=source, root=output / "torch_cu128", name="build_tools", timeout=600, environment=base_env,
        )
        _require_process("T0 build tools", tools)
        torch_install = _managed(
            [str(python), "-m", "pip", "install", "torch==2.7.0", "torchvision==0.22.0", "--index-url", config["environment"]["torch_index_url"]],
            cwd=source, root=output / "torch_cu128", name="torch_install", timeout=config["timeouts_seconds"]["torch"], environment=base_env,
        )
        _require_process("T0 torch install", torch_install)
        t0 = _managed(
            [str(python), str(REPO_ROOT / "scripts/univtac/probe_blackwell_torch.py"), "--output", str(output / "torch_cu128" / "probe.json")],
            cwd=source, root=output / "torch_cu128", name="torch_probe", timeout=600, environment=base_env,
        )
        torch_payload = json.loads((output / "torch_cu128" / "probe.json").read_text())
        t0_class = classify_torch_probe(torch_payload)
        manifest["gates"]["T0"] = t0_class
        if not _process_ok(t0) or t0_class != "passed":
            manifest["classification"] = "torch_cu128_sm120_failed"
            raise RuntimeError("T0 failed")

        official_yaml = source / "third_party/TacEx/source/tacex_uipc/libuipc/conda/env.yaml"
        official_payload = yaml.safe_load(official_yaml.read_text(encoding="utf-8"))
        derived, semantic_diff = derive_libuipc_environment(official_payload)
        derived_path = output / "derived_environment" / "derived_libuipc_sm120_env.yaml"
        derived_path.parent.mkdir(parents=True, exist_ok=True)
        derived_path.write_text(yaml.safe_dump(derived, sort_keys=False), encoding="utf-8")
        write_json(output / "derived_environment" / "semantic_diff.json", {
            **semantic_diff,
            "official_yaml": str(official_yaml.resolve()),
            "official_sha256": hashlib.sha256(official_yaml.read_bytes()).hexdigest(),
            "derived_sha256": hashlib.sha256(derived_path.read_bytes()).hexdigest(),
        })
        update = _managed(
            [str(args.conda_exe), "env", "update", "--name", env_name, "--file", str(derived_path)],
            cwd=source, root=output / "derived_environment", name="conda_update", timeout=config["timeouts_seconds"]["environment_update"], environment=base_env,
        )
        _require_process("E0 derived environment", update)
        nvcc = _run([str(prefix / "bin/nvcc"), "--version"], environment=base_env)
        write_json(output / "derived_environment" / "toolchain.json", nvcc)
        if nvcc.get("returncode") != 0 or "release 12.8" not in str(nvcc.get("stdout", "")):
            manifest["classification"] = "derived_cuda128_environment_failed"
            raise RuntimeError("target environment nvcc is not CUDA 12.8")
        manifest["gates"]["E0"] = "passed"

        deps = _managed(
            [str(python), "-m", "pip", "install", "toml", "wildmeshing>=0.4.1", "pybind11", "mypy", "transforms3d", "tetgen", "polyscope>=2.5,<3"],
            cwd=source, root=output / "libuipc_build", name="python_build_dependencies", timeout=1200, environment=base_env,
        )
        _require_process("libuipc build dependencies", deps)
        vcpkg_root.parent.mkdir(parents=True, exist_ok=True)
        vcpkg_clone = _managed(
            ["git", "clone", "https://github.com/microsoft/vcpkg.git", str(vcpkg_root)],
            cwd=source, root=output / "libuipc_build", name="vcpkg_clone", timeout=config["timeouts_seconds"]["vcpkg"], environment=base_env,
        )
        _require_process("vcpkg clone", vcpkg_clone)
        checkout = _run(["git", "checkout", "--detach", config["source"]["vcpkg_commit"]], cwd=vcpkg_root, environment=base_env)
        if checkout.get("returncode") != 0:
            raise RuntimeError("vcpkg pinned checkout failed")
        bootstrap = _managed(
            [str(vcpkg_root / "bootstrap-vcpkg.sh"), "-disableMetrics"],
            cwd=vcpkg_root, root=output / "libuipc_build", name="vcpkg_bootstrap", timeout=config["timeouts_seconds"]["vcpkg"], environment=base_env,
        )
        _require_process("vcpkg bootstrap", bootstrap)
        native_env = native_environment(prefix, config, vcpkg_root)
        uipc_root = source / "third_party/TacEx/source/tacex_uipc"
        build_dir = uipc_root / "build"
        if build_dir.exists():
            raise FileExistsError(f"unexpected preexisting libuipc build directory: {build_dir}")
        build = _managed(
            ["/usr/bin/time", "-v", str(python), "-m", "pip", "install", "--no-build-isolation", "-e", str(uipc_root)],
            cwd=uipc_root, root=output / "libuipc_build", name="libuipc_build", timeout=config["timeouts_seconds"]["libuipc_build"], environment=native_env,
        )
        build_log = Path(build["log_path"]).read_text(encoding="utf-8", errors="replace")
        build_returncode = build.get("returncode")
        build_class = build_log_classification(
            build_log,
            returncode=int(build_returncode) if isinstance(build_returncode, int) else 1,
        )
        build_detail = {
            "classification": build_class,
            "returncode": build.get("returncode"),
            "elapsed_seconds": build.get("elapsed_seconds"),
            "log_path": build.get("log_path"),
            "vcpkg_source_hash_mismatch": "unexpected hash" in build_log,
            "requested_cuda_architecture": "120",
            "target_environment_nvcc": str(prefix / "bin/nvcc"),
        }
        write_json(output / "libuipc_build" / "result.json", build_detail)
        manifest["gates"]["U0"] = build_class
        if not _process_ok(build):
            manifest["classification"] = build_class
            raise RuntimeError("U0 libuipc build failed")

        uipc_binaries = _binary_candidates([build_dir, prefix], r"uipc|pyuipc|muda|backend.*cuda")
        if not uipc_binaries:
            manifest["classification"] = "libuipc_binary_missing_sm120_code"
            raise RuntimeError("no libuipc binaries found")
        audit = _managed(
            [str(python), str(REPO_ROOT / "scripts/univtac/audit_blackwell_binaries.py"), "--output", str(output / "libuipc_binary_audit" / "audit.json"), *map(str, uipc_binaries)],
            cwd=source, root=output / "libuipc_binary_audit", name="audit", timeout=1200, environment=native_env,
        )
        audit_payload = json.loads((output / "libuipc_binary_audit" / "audit.json").read_text())
        audit_class = binary_architecture_classification(audit_payload["records"])
        manifest["gates"]["B0"] = audit_class
        if not _process_ok(audit) or audit_class != "passed":
            manifest["classification"] = "libuipc_binary_missing_sm120_code"
            raise RuntimeError("B0 libuipc binary audit failed")

        core_results = []
        for index, timeout in ((1, config["timeouts_seconds"]["libuipc_core_first"]), (2, config["timeouts_seconds"]["libuipc_core_second"])):
            core_root = output / f"libuipc_core_run{index}"
            process = _managed(
                [str(python), str(REPO_ROOT / "scripts/univtac/probe_libuipc_core.py"), "--workspace", str(core_root / "workspace"), "--output", str(core_root / "probe.json"), "--events", str(core_root / "events.jsonl")],
                cwd=source, root=core_root, name="probe", timeout=timeout, environment=native_env,
            )
            payload = json.loads((core_root / "probe.json").read_text()) if (core_root / "probe.json").is_file() else {"success": False}
            classification = classify_core_probe(payload, timed_out=bool(process.get("timed_out")))
            payload["classification"] = classification
            core_results.append(payload)
            manifest["gates"][f"U1_RUN{index}"] = classification
            if classification != "passed" or not _process_ok(process):
                manifest["classification"] = classification
                raise RuntimeError(f"U1 run {index} failed")

        curobo_root = source / "third_party/curobo"
        if curobo_root.exists():
            raise FileExistsError(f"unexpected preexisting cuRobo checkout: {curobo_root}")
        clone = _managed(
            ["git", "clone", "https://github.com/NVlabs/curobo.git", str(curobo_root)],
            cwd=source, root=output / "curobo_build", name="clone", timeout=1200, environment=base_env,
        )
        _require_process("cuRobo clone", clone)
        checkout = _run(["git", "checkout", "--detach", config["source"]["curobo_commit"]], cwd=curobo_root, environment=base_env)
        if checkout.get("returncode") != 0:
            raise RuntimeError("cuRobo pinned checkout failed")
        curobo_build = _managed(
            ["/usr/bin/time", "-v", str(python), "-m", "pip", "install", "--no-build-isolation", "-e", str(curobo_root)],
            cwd=curobo_root, root=output / "curobo_build", name="build", timeout=config["timeouts_seconds"]["curobo_build"], environment=native_env,
        )
        manifest["gates"]["C0_BUILD"] = "passed" if _process_ok(curobo_build) else "curobo_sm120_build_failed"
        if not _process_ok(curobo_build):
            manifest["classification"] = "curobo_sm120_build_failed"
            raise RuntimeError("cuRobo build failed")
        curobo_binaries = _binary_candidates([curobo_root, prefix], r"_cu")
        curobo_audit = _managed(
            [str(python), str(REPO_ROOT / "scripts/univtac/audit_blackwell_binaries.py"), "--output", str(output / "curobo_binary_audit" / "audit.json"), *map(str, curobo_binaries)],
            cwd=curobo_root, root=output / "curobo_binary_audit", name="audit", timeout=1200, environment=native_env,
        )
        manifest["gates"]["C0_AUDIT"] = "passed" if _process_ok(curobo_audit) else "curobo_sm120_build_failed"
        if not _process_ok(curobo_audit):
            manifest["classification"] = "curobo_sm120_build_failed"
            raise RuntimeError("cuRobo binary audit failed")
        curobo_probe = _managed(
            [str(python), str(REPO_ROOT / "scripts/univtac/probe_blackwell_curobo.py"), "--source-root", str(curobo_root), "--output", str(output / "curobo_smoke" / "probe.json")],
            cwd=curobo_root, root=output / "curobo_smoke", name="probe", timeout=config["timeouts_seconds"]["curobo_smoke"], environment=native_env,
        )
        manifest["gates"]["C0_SMOKE"] = "passed" if _process_ok(curobo_probe) else "blackwell_native_bridge_partially_viable"
        manifest["classification"] = "blackwell_native_bridge_viable" if _process_ok(curobo_probe) else "blackwell_native_bridge_partially_viable"
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = "failed"
        if manifest.get("classification") is None:
            manifest["classification"] = "blocked_by_external_resources"
        manifest["failure"] = {"class": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        write_json(output / "failure.json", manifest["failure"])
        raise
    finally:
        final_errors = []
        try:
            if fingerprints_before:
                after = {
                    "legacy": fingerprint_environment(args.conda_exe, args.legacy_env),
                    "r07": fingerprint_environment(args.conda_exe, args.r07_env),
                    "legacy_conda_meta": conda_meta_fingerprint(prefix.parent / args.legacy_env),
                    "r07_conda_meta": conda_meta_fingerprint(prefix.parent / args.r07_env),
                    "openeta_uv": uv_fingerprint(),
                }
                write_json(output / "legacy_env_fingerprints" / "after.json", after)
                comparison = {
                    "legacy": compare_fingerprints(fingerprints_before["legacy"], after["legacy"]),
                    "r07": compare_fingerprints(fingerprints_before["r07"], after["r07"]),
                    "legacy_conda_meta_unchanged": fingerprints_before["legacy_conda_meta"] == after["legacy_conda_meta"],
                    "r07_conda_meta_unchanged": fingerprints_before["r07_conda_meta"] == after["r07_conda_meta"],
                    "openeta_uv_unchanged": fingerprints_before["openeta_uv"]["pip_freeze_sha256"] == after["openeta_uv"]["pip_freeze_sha256"],
                }
                write_json(output / "legacy_env_fingerprints" / "comparison.json", comparison)
                manifest["protected_environments_unchanged"] = all(
                    (
                        comparison["legacy_conda_meta_unchanged"],
                        comparison["r07_conda_meta_unchanged"],
                        comparison["openeta_uv_unchanged"],
                    )
                )
                if not manifest["protected_environments_unchanged"]:
                    manifest["status"] = "failed"
                    if manifest.get("classification") == "blackwell_native_bridge_viable":
                        manifest["classification"] = "blackwell_native_bridge_partially_viable"
        except BaseException as exc:
            final_errors.append(f"fingerprint: {type(exc).__name__}: {exc}")
        try:
            final_status = _git(source, "status", "--short")
            final_state = {"head": _git(source, "rev-parse", "HEAD"), "status": final_status, "tracked_clean": source_tracked_clean(final_status), "tracked_diff": _git(source, "diff", "--")}
            write_json(output / "final_resources" / "source.json", final_state)
            manifest["source_tracked_clean"] = final_state["tracked_clean"]
            if not final_state["tracked_clean"]:
                manifest["status"] = "failed"
                if manifest.get("classification") == "blackwell_native_bridge_viable":
                    manifest["classification"] = "blackwell_native_bridge_partially_viable"
        except BaseException as exc:
            final_errors.append(f"source: {type(exc).__name__}: {exc}")
        try:
            final_gpu = collect_gpu_inventory()
            final_inotify = collect_inotify_inventory()
            final_processes = attach_gpu_usage(collect_process_inventory(related_roots=(REPO_ROOT, source, output)), final_gpu)
            write_json(output / "final_resources" / "resources.json", {"gpu": final_gpu, "inotify": final_inotify, "processes": final_processes})
            residual = [item["pid"] for item in final_processes.get("processes", []) if item.get("eligible_for_cleanup")]
            manifest["cleanup_residual_pids"] = residual
            if residual:
                manifest["classification"] = "cleanup_incomplete"
        except BaseException as exc:
            final_errors.append(f"resources: {type(exc).__name__}: {exc}")
        manifest["finalization_errors"] = final_errors
        for gate in config["gate_order"]:
            manifest["gates"].setdefault(gate, "not_run_due_to_gate")
        manifest["ended_at"] = utc_now()
        write_json(output / "summary.json", manifest)
        write_json(output / "run_manifest.json", manifest)
        _write_author_bundle(output, manifest)


if __name__ == "__main__":
    main()
