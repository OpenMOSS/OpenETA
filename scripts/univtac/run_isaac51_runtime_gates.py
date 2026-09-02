#!/usr/bin/env python3
"""Run the isolated R0.7 Isaac51 environment and ordered runtime gates.

The runner never changes sysctls, reuses the legacy environment, or modifies the
OpenETA vendored UniVTAC tree. Machine-specific paths are supplied explicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import traceback
import re
from pathlib import Path
from typing import Any, Mapping

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from sim.envs.univtac.isaac51_runtime_validation import (
    CollectionGate,
    build_collect_command,
    deterministic_json,
    environment_spec_sha256,
    gate_sequence_decision,
    inspect_fresh_output_root,
    summarize_collection_gate,
    validate_collect_command,
    validate_config,
)
from sim.envs.univtac.resource_sanitation import (
    attach_gpu_usage,
    build_restore_ready,
    collect_gpu_inventory,
    collect_inotify_inventory,
    collect_process_inventory,
    run_managed_process,
    utc_now,
    write_json,
)


STOP_STAGES = ("preflight", "t0", "install", "t1", "s0", "s1", "gates")
IMPORTS = ("torch", "torchvision", "isaacsim", "isaaclab", "uipc", "curobo", "tacex", "tacex_assets", "tacex_uipc")
PRESERVED_ENVIRONMENT = (
    "HOME",
    "USER",
    "LOGNAME",
    "LANG",
    "LC_ALL",
    "TERM",
    "PATH",
    "TMPDIR",
    "SSL_CERT_FILE",
    "REQUESTS_CA_BUNDLE",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "NO_PROXY",
    "http_proxy",
    "https_proxy",
    "no_proxy",
)


def load_config(path: Path) -> tuple[dict[str, Any], tuple[CollectionGate, ...]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("runtime validation config must be a mapping")
    return payload, validate_config(payload)


def clean_environment(extra: Mapping[str, str] | None = None) -> dict[str, str]:
    environment = {key: os.environ[key] for key in PRESERVED_ENVIRONMENT if key in os.environ}
    environment["PYTHONNOUSERSITE"] = "1"
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = "0"
    environment["CUDA_LAUNCH_BLOCKING"] = "1"
    environment["OMNI_KIT_ACCEPT_EULA"] = "YES"
    if extra:
        environment.update({str(key): str(value) for key, value in extra.items()})
    return environment


def _run_capture(command: list[str], *, cwd: Path | None = None, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            env=dict(environment) if environment is not None else None,
            check=False,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {"command": command, "returncode": None, "error": f"{type(exc).__name__}: {exc}"}
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_fingerprint_output(kind: str, text: str) -> str:
    text = re.sub(r"\x1b\[[0-9;?]*[ -/]*[@-~]", "", text).replace("\b", "")
    lines = text.splitlines()
    if kind == "conda_explicit":
        try:
            lines = lines[lines.index("@EXPLICIT") :]
        except ValueError:
            pass
    return "\n".join(line.rstrip() for line in lines if "Retrieving notices" not in line).strip() + "\n"


def fingerprint_environment(conda: Path, name: str) -> dict[str, Any]:
    commands = {
        "conda_explicit": [str(conda), "list", "--name", name, "--explicit"],
        "conda_export": [str(conda), "env", "export", "--name", name, "--no-builds"],
        "pip_freeze": [str(conda), "run", "--name", name, "python", "-m", "pip", "freeze"],
    }
    payload: dict[str, Any] = {"environment": name, "captured_at": utc_now(), "commands": {}}
    for key, command in commands.items():
        result = _run_capture(command, environment=clean_environment())
        stdout = str(result.get("stdout", ""))
        normalized = _normalize_fingerprint_output(key, stdout)
        payload["commands"][key] = {
            "returncode": result.get("returncode"),
            "sha256": _sha256_text(normalized) if result.get("returncode") == 0 else None,
            "line_count": len(normalized.splitlines()),
            "content_recorded": False,
        }
    payload["success"] = all(item["returncode"] == 0 for item in payload["commands"].values())
    return payload


def compare_fingerprints(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    keys = sorted(set(before.get("commands", {})) | set(after.get("commands", {})))
    comparisons = {}
    for key in keys:
        before_item = before.get("commands", {}).get(key, {})
        after_item = after.get("commands", {}).get(key, {})
        before_hash = before_item.get("sha256")
        after_hash = after_item.get("sha256")
        comparisons[key] = bool(
            before_item.get("returncode") == 0
            and after_item.get("returncode") == 0
            and before_hash
            and after_hash
            and before_hash == after_hash
        )
    required = ("conda_explicit", "pip_freeze")
    verification_complete = all(
        key in comparisons
        and before.get("commands", {}).get(key, {}).get("returncode") == 0
        and after.get("commands", {}).get(key, {}).get("returncode") == 0
        for key in required
    )
    unchanged = verification_complete and all(comparisons[key] for key in required)
    return {
        "required_immutability_fingerprints": list(required),
        "supplemental_fingerprints": [key for key in keys if key not in required],
        "verification_complete": verification_complete,
        "unchanged": unchanged,
        "classification": (
            "unchanged" if unchanged else ("changed" if verification_complete else "verification_failed")
        ),
        "comparisons": comparisons,
    }


def _git_output(checkout: Path, *arguments: str, check: bool = True) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=checkout, check=check, capture_output=True, text=True
    )
    return completed.stdout.strip()


def create_pinned_checkout(source_cache: Path, target: Path, commit: str) -> dict[str, Any]:
    if target.exists():
        actual = _git_output(target, "rev-parse", "HEAD")
        status = _git_output(target, "status", "--short")
        detached = not bool(_git_output(target, "symbolic-ref", "-q", "HEAD", check=False))
        if actual != commit or status or not detached:
            raise RuntimeError(f"existing target checkout is not the clean pinned source: {target}")
        return {
            "source_cache": str(source_cache.resolve()),
            "target": str(target.resolve()),
            "commit": actual,
            "detached": True,
            "tracked_clean": True,
            "reused_existing": True,
        }
    if _git_output(source_cache, "rev-parse", "HEAD") != commit:
        raise RuntimeError("source cache is not at the pinned Isaac51 commit")
    if _git_output(source_cache, "status", "--short"):
        raise RuntimeError("source cache is dirty")
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", "--no-local", "--no-checkout", str(source_cache), str(target)],
        check=True,
    )
    subprocess.run(["git", "checkout", "--detach", commit], cwd=target, check=True)
    actual = _git_output(target, "rev-parse", "HEAD")
    if actual != commit or _git_output(target, "status", "--short"):
        raise RuntimeError("new Isaac51 checkout failed immutable clean-source validation")
    return {
        "source_cache": str(source_cache.resolve()),
        "target": str(target.resolve()),
        "commit": actual,
        "detached": not bool(_git_output(target, "symbolic-ref", "-q", "HEAD", check=False)),
        "tracked_clean": True,
        "reused_existing": False,
    }


def _disk_free_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024**3)


def resource_preflight(
    *, config: Mapping[str, Any], source_cache: Path, source_checkout: Path, output_root: Path
) -> dict[str, Any]:
    inotify = collect_inotify_inventory()
    gpu = collect_gpu_inventory()
    processes = attach_gpu_usage(
        collect_process_inventory(
            related_roots=(REPO_ROOT, source_cache, source_checkout, output_root)
        ),
        gpu,
    )
    thresholds = config["resource_gates"]
    gpus = gpu.get("gpus", [])
    gpu_ratio = (
        gpus[0]["memory_free_mib"] / gpus[0]["memory_total_mib"] if len(gpus) == 1 else None
    )
    eligible = [item["pid"] for item in processes.get("processes", []) if item.get("eligible_for_cleanup")]
    checks = {
        "disk_free": _disk_free_gib(output_root.parent) >= float(thresholds["minimum_disk_free_gib"]),
        "single_gpu_visible": len(gpus) == 1,
        "gpu_free": gpu_ratio is not None and gpu_ratio >= float(thresholds["minimum_gpu_free_ratio"]),
        "inotify_instances_raised": inotify.get("max_user_instances") == int(thresholds["required_inotify_instances"]),
        "inotify_watches_raised": inotify.get("max_user_watches") == int(thresholds["required_inotify_watches"]),
        "inotify_instances_below_75_percent": (
            inotify.get("instance_usage_ratio") is not None
            and inotify["instance_usage_ratio"] < float(thresholds["maximum_inotify_usage_ratio"])
        ),
        "inotify_watches_below_75_percent": (
            inotify.get("watch_count_reliable") is True
            and inotify.get("watch_usage_ratio") is not None
            and inotify["watch_usage_ratio"] < float(thresholds["maximum_inotify_usage_ratio"])
        ),
        "no_stale_project_runtime": not eligible,
    }
    return {
        "captured_at": utc_now(),
        "checks": checks,
        "passed": all(checks.values()),
        "disk_free_gib": _disk_free_gib(output_root.parent),
        "gpu_free_ratio": gpu_ratio,
        "inotify": inotify,
        "gpu": gpu,
        "processes": processes,
        "eligible_stale_pids": eligible,
        "sysctl_modified_by_runner": False,
    }


def _managed(
    *, command: list[str], cwd: Path, output_root: Path, name: str, timeout: float, environment: Mapping[str, str]
) -> dict[str, Any]:
    result = run_managed_process(
        command,
        cwd=cwd,
        log_path=output_root / "logs" / f"{name}.log",
        timeout_seconds=timeout,
        environment=environment,
    )
    write_json(output_root / "processes" / f"{name}.json", result)
    return result


def _require_process_ok(name: str, result: Mapping[str, Any]) -> None:
    if result.get("returncode") != 0 or result.get("timed_out") or not result.get("cleanup_complete"):
        raise RuntimeError(f"{name} failed; see {result.get('log_path')}")


def _probe_command(python: Path, output: Path, imports: tuple[str, ...] = ()) -> list[str]:
    command = [str(python), str(REPO_ROOT / "scripts/univtac/probe_isaac51_torch.py"), "--output", str(output)]
    for module in imports:
        command.extend(["--import", module])
    return command


def _collection_tactile_valid(smoke_results: Mapping[str, Any], c0: Mapping[str, Any]) -> bool:
    return bool(
        smoke_results.get("S0", {}).get("passed")
        and c0.get("reset_returned")
        and c0.get("expert_trajectory_completed")
        and c0.get("cleanup_success")
    )


def print_plan(config: Mapping[str, Any], gates: tuple[CollectionGate, ...], source_checkout: Path, output_root: Path, conda: Path) -> None:
    prefix = Path(_run_capture([str(conda), "info", "--base"])["stdout"].strip()) / "envs" / config["environment"]["conda_name"]
    payload = {
        "source_commit": config["source"]["commit"],
        "source_checkout": str(source_checkout.resolve()),
        "target_environment_prefix": str(prefix),
        "output_root": str(output_root.resolve()),
        "ordered_gates": [gate.to_dict() for gate in gates],
        "seed_flags": {gate.gate_id: ["--start_seed", gate.seed, "--max_seed", gate.seed] for gate in gates},
        "mutates_sysctl": False,
        "modifies_legacy_environment": False,
    }
    print(deterministic_json(payload), end="")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--conda-exe", type=Path, required=True)
    parser.add_argument("--legacy-env", default="UniVTAC")
    parser.add_argument("--stop-after", choices=STOP_STAGES, default="gates")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config, gates = load_config(args.config.resolve())
    if args.dry_run:
        print_plan(config, gates, args.source_checkout, args.output_root, args.conda_exe)
        return
    if args.output_root.exists():
        raise FileExistsError(f"R0.7 output root must be absent: {args.output_root}")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "schema_version": "openeta.univtac.isaac51_runtime_run.v1",
        "started_at": utc_now(),
        "status": "running",
        "config_path": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
        "environment_spec_sha256": environment_spec_sha256(config),
        "source_commit": config["source"]["commit"],
        "legacy_environment": args.legacy_env,
        "target_environment": config["environment"]["conda_name"],
        "planner_task_constraint_modified": False,
        "sysctl_modified": False,
        "gates": {},
    }
    write_json(output_root / "run_manifest.json", manifest)
    prefix: Path | None = None
    source_checkout = args.source_checkout.resolve()
    vcpkg_root = source_checkout / ".cache" / "toolchains" / "vcpkg"
    initial_source_status = None
    legacy_before: dict[str, Any] = {}
    try:
        preflight = resource_preflight(
            config=config,
            source_cache=args.source_cache.resolve(),
            source_checkout=source_checkout,
            output_root=output_root,
        )
        write_json(output_root / "resource_preflight.json", preflight)
        if not preflight["passed"]:
            raise RuntimeError("resource preflight failed")
        checkout_record = create_pinned_checkout(
            args.source_cache.resolve(), source_checkout, config["source"]["commit"]
        )
        write_json(output_root / "source_checkout.json", checkout_record)
        initial_source_status = _git_output(source_checkout, "status", "--short")
        base_result = _run_capture([str(args.conda_exe), "info", "--base"], environment=clean_environment())
        if base_result.get("returncode") != 0:
            raise RuntimeError("could not resolve Conda base")
        prefix = Path(str(base_result["stdout"]).strip()) / "envs" / config["environment"]["conda_name"]
        if prefix.exists():
            raise FileExistsError(f"target environment already exists without this run's provenance: {prefix}")
        legacy_before = fingerprint_environment(args.conda_exe, args.legacy_env)
        write_json(output_root / "legacy_environment_before.json", legacy_before)
        if not legacy_before.get("success"):
            raise RuntimeError("could not fingerprint protected legacy environment")
        ledger = {
            "schema_version": "openeta.univtac.environment_ledger.v1",
            "created_at": utc_now(),
            "environment_spec_sha256": manifest["environment_spec_sha256"],
            "source_commit": config["source"]["commit"],
            "source_checkout": str(source_checkout),
            "target_environment": config["environment"]["conda_name"],
            "target_prefix": str(prefix),
            "installer": str((source_checkout / "scripts/install.sh").resolve()),
            "legacy_environment_read_only": True,
        }
        write_json(output_root / "environment_ledger.json", ledger)
        manifest["stages"] = {"preflight": "passed"}
        write_json(output_root / "run_manifest.json", manifest)
        if args.stop_after == "preflight":
            return

        environment = clean_environment()
        t0_create = _managed(
            command=[
                str(args.conda_exe), "create", "--name", config["environment"]["conda_name"], "--yes",
                "--override-channels", "--channel", "https://repo.anaconda.com/pkgs/main", "python=3.11", "pip",
            ],
            cwd=source_checkout,
            output_root=output_root,
            name="t0_conda_create",
            timeout=float(config["timeouts_seconds"]["t0"]),
            environment=environment,
        )
        _require_process_ok("T0 Conda create", t0_create)
        python = prefix / "bin/python"
        t0_build_tools = _managed(
            command=[
                str(python), "-m", "pip", "install", "setuptools==75.8.2", "wheel==0.42.0",
            ],
            cwd=source_checkout,
            output_root=output_root,
            name="t0_build_tools",
            timeout=float(config["timeouts_seconds"]["t0"]),
            environment=environment,
        )
        _require_process_ok("T0 build tools", t0_build_tools)
        t0_torch = _managed(
            command=[
                str(python), "-m", "pip", "install", f"torch=={config['environment']['torch']}",
                f"torchvision=={config['environment']['torchvision']}", "--index-url", config["environment"]["torch_index_url"],
            ],
            cwd=source_checkout,
            output_root=output_root,
            name="t0_torch_install",
            timeout=float(config["timeouts_seconds"]["t0"]),
            environment=environment,
        )
        _require_process_ok("T0 Torch install", t0_torch)
        marker = {**ledger, "t0_completed_at": utc_now()}
        write_json(prefix / ".univtac-r07-provenance.json", marker)
        t0_probe = _managed(
            command=_probe_command(python, output_root / "t0_torch_probe.json"),
            cwd=source_checkout,
            output_root=output_root,
            name="t0_torch_probe",
            timeout=float(config["timeouts_seconds"]["runtime_import"]),
            environment=environment,
        )
        _require_process_ok("T0 Torch/CUDA probe", t0_probe)
        manifest["stages"]["t0"] = "passed"
        write_json(output_root / "run_manifest.json", manifest)
        if args.stop_after == "t0":
            return

        install_environment = clean_environment(
            {
                "UNIVTAC_CONDA_ENV": config["environment"]["conda_name"],
                "UNIVTAC_CUDA_ARCH": config["environment"]["cuda_arch"],
                "UNIVTAC_BUILD_JOBS": str(config["environment"]["build_jobs"]),
                "UNIVTAC_VCPKG_ROOT": str(vcpkg_root),
            }
        )
        install = _managed(
            command=["/usr/bin/time", "-v", "bash", str(source_checkout / "scripts/install.sh")],
            cwd=source_checkout,
            output_root=output_root,
            name="official_install",
            timeout=float(config["timeouts_seconds"]["install"]),
            environment=install_environment,
        )
        _require_process_ok("official install", install)
        check = _managed(
            command=["bash", str(source_checkout / "scripts/install.sh"), "--check"],
            cwd=source_checkout,
            output_root=output_root,
            name="official_install_check",
            timeout=float(config["timeouts_seconds"]["install_check"]),
            environment=install_environment,
        )
        _require_process_ok("official install check", check)
        summary = _managed(
            command=[
                str(python), str(REPO_ROOT / "scripts/univtac/summarize_isaac51_install.py"),
                "--source-root", str(source_checkout), "--vcpkg-root", str(vcpkg_root),
                "--output", str(output_root / "install_summary.json"),
            ],
            cwd=source_checkout,
            output_root=output_root,
            name="install_summary",
            timeout=float(config["timeouts_seconds"]["install_check"]),
            environment=environment,
        )
        _require_process_ok("install summary", summary)
        manifest["stages"]["install"] = "passed"
        write_json(output_root / "run_manifest.json", manifest)
        if args.stop_after == "install":
            return

        t1_probe = _managed(
            command=_probe_command(python, output_root / "t1_runtime_probe.json", IMPORTS),
            cwd=source_checkout,
            output_root=output_root,
            name="t1_runtime_probe",
            timeout=float(config["timeouts_seconds"]["runtime_import"]),
            environment=environment,
        )
        _require_process_ok("T1 runtime imports", t1_probe)
        manifest["stages"]["t1"] = "passed"
        write_json(output_root / "run_manifest.json", manifest)
        if args.stop_after == "t1":
            return

        smoke_results: dict[str, Any] = {}
        for gate_id, backend in (("S0", "taxim"), ("S1", "pix2pix")):
            smoke_dir = output_root / gate_id.lower()
            if smoke_dir.exists():
                raise FileExistsError(f"smoke output must be fresh: {smoke_dir}")
            process = _managed(
                command=[
                    str(python), str(source_checkout / "scripts/smoke_isaac51.py"),
                    "--backend", backend, "--output-dir", str(smoke_dir), "--headless",
                ],
                cwd=source_checkout,
                output_root=output_root,
                name=f"{gate_id.lower()}_{backend}",
                timeout=float(config["timeouts_seconds"]["smoke"]),
                environment=environment,
            )
            log_text = Path(process["log_path"]).read_text(encoding="utf-8", errors="replace")
            result = {
                "gate_id": gate_id,
                "backend": backend,
                "passed": process.get("returncode") == 0 and "PASS backend=" in log_text and process.get("cleanup_complete"),
                "process": process,
            }
            smoke_results[gate_id] = result
            write_json(output_root / f"{gate_id.lower()}_result.json", result)
            if not result["passed"]:
                raise RuntimeError(f"{gate_id} {backend} smoke failed")
            manifest["stages"][gate_id.lower()] = "passed"
            write_json(output_root / "run_manifest.json", manifest)
            if args.stop_after == gate_id.lower():
                return

        completed: dict[str, dict[str, Any]] = {}
        while True:
            c0_tactile_valid = _collection_tactile_valid(smoke_results, completed.get("C0", {}))
            next_gate = gate_sequence_decision(completed, c0_tactile_valid=c0_tactile_valid)
            if next_gate is None:
                break
            gate = next(item for item in gates if item.gate_id == next_gate)
            gate_root = output_root / gate.output_dir
            collection_root = gate_root / "collection"
            freshness = inspect_fresh_output_root(collection_root)
            if not freshness["fresh_output_root"]:
                raise RuntimeError(f"{gate.gate_id} output root is not fresh")
            gate_root.mkdir(parents=True, exist_ok=False)
            command = build_collect_command(
                python=python,
                source_root=source_checkout,
                gate=gate,
                collection_root=collection_root,
                gpu=config["runtime"]["gpu"],
            )
            validate_collect_command(command, gate)
            process = _managed(
                command=command,
                cwd=source_checkout,
                output_root=gate_root,
                name=f"{gate.gate_id.lower()}_collect",
                timeout=float(config["timeouts_seconds"]["collection"]),
                environment=environment,
            )
            log_text = Path(process["log_path"]).read_text(encoding="utf-8", errors="replace")
            result = summarize_collection_gate(
                gate=gate,
                collection_root=collection_root,
                log_text=log_text,
                process_result=process,
                freshness=freshness,
            )
            if gate.gate_id == "C0":
                result["tactile_observation_legal"] = _collection_tactile_valid(smoke_results, result)
            write_json(gate_root / "gate_result.json", result)
            completed[gate.gate_id] = result
            manifest["gates"][gate.gate_id] = result
            write_json(output_root / "run_manifest.json", manifest)
            if result.get("seed_contract_violation"):
                break
        manifest["stages"]["gates"] = "completed"
        manifest["status"] = "completed"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["failure"] = {
            "class": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(output_root / "failure.json", manifest["failure"])
        raise
    finally:
        finalization_errors: list[dict[str, str]] = []
        try:
            if legacy_before:
                legacy_after = fingerprint_environment(args.conda_exe, args.legacy_env)
                write_json(output_root / "legacy_environment_after.json", legacy_after)
                comparison = compare_fingerprints(legacy_before, legacy_after)
                write_json(output_root / "legacy_environment_comparison.json", comparison)
                manifest["legacy_environment_unchanged"] = comparison["unchanged"]
        except BaseException as exc:
            finalization_errors.append({"stage": "legacy_fingerprint", "error": f"{type(exc).__name__}: {exc}"})
        try:
            if source_checkout.exists():
                final_status = _git_output(source_checkout, "status", "--short")
                write_json(
                    output_root / "source_final_state.json",
                    {
                        "initial_status": initial_source_status,
                        "final_status": final_status,
                        "tracked_diff": _git_output(source_checkout, "diff", "--"),
                        "head": _git_output(source_checkout, "rev-parse", "HEAD"),
                    },
                )
        except BaseException as exc:
            finalization_errors.append({"stage": "source_final_state", "error": f"{type(exc).__name__}: {exc}"})
        try:
            final_inotify = collect_inotify_inventory()
            final_gpu = collect_gpu_inventory()
            final_processes = attach_gpu_usage(
                collect_process_inventory(related_roots=(REPO_ROOT, source_checkout, output_root)), final_gpu
            )
            restore = build_restore_ready(
                original_instances=128,
                original_watches=65536,
                inotify_inventory=final_inotify,
                process_inventory=final_processes,
                gpu_inventory=final_gpu,
            )
            restore["sysctl_restore_performed_by_runner"] = False
            write_json(output_root / "restore_ready.json", restore)
        except BaseException as exc:
            finalization_errors.append({"stage": "restore_ready", "error": f"{type(exc).__name__}: {exc}"})
        manifest["finalization_errors"] = finalization_errors
        manifest["ended_at"] = utc_now()
        write_json(output_root / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
