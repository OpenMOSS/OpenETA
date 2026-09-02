#!/usr/bin/env python3
"""Recover tinygltf by exact source identity and resume the R0.8 native gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.univtac.run_blackwell_native_bridge import (
    _binary_candidates,
    _managed,
    _process_ok,
    _require_process,
    native_environment,
    uv_fingerprint,
)
from scripts.univtac.run_isaac51_runtime_gates import (
    clean_environment,
    compare_fingerprints,
    fingerprint_environment,
)
from sim.envs.univtac.blackwell_adaptation import classify_torch_probe
from sim.envs.univtac.libuipc_core_probe import classify_core_probe
from sim.envs.univtac.native_build_contract import binary_architecture_classification
from sim.envs.univtac.resource_sanitation import utc_now, write_json
from sim.envs.univtac.source_tree_manifest import archive_manifest, compare_manifests, git_manifest
from sim.envs.univtac.vcpkg_source_recovery import (
    EXPECTED_SHA512,
    TINYGLTF_COMMIT,
    VCPKG_COMMIT,
    find_historical_asset,
    generate_overlay_portfile,
    recovery_strategy,
    require_fixed_tag_commit,
)


def _run(command: list[str], *, cwd: Path | None = None, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, cwd=cwd, check=False, capture_output=True, text=True, timeout=timeout)


def _git(path: Path, *args: str) -> str:
    result = _run(["git", *args], cwd=path)
    if result.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr}")
    return result.stdout.strip()


def _download(url: str, path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = path.with_suffix(path.suffix + ".headers")
    result = _run(
        ["curl", "--fail", "--location", "--silent", "--show-error", "--dump-header", str(headers), "--output", str(path), "--write-out", "%{url_effective}", url],
        timeout=1800,
    )
    if result.returncode:
        raise RuntimeError(f"download failed: {url}: {result.stderr}")
    data = path.read_bytes()
    return {
        "requested_url": url,
        "final_url": result.stdout,
        "size": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "sha512": hashlib.sha512(data).hexdigest(),
        "headers_path": str(headers.resolve()),
    }


def curobo_failure_summary(log_path: Path, prefix: Path) -> dict[str, Any]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    missing = sorted(set(re.findall(r"fatal error: ([^:]+): No such file or directory", text)))
    arch_lines = {line.strip() for line in text.splitlines() if "gencode" in line and "compute_120" in line}
    return {
        "schema_version": "openeta.univtac.curobo_build_failure.v1",
        "stage": "host_cxx_compile" if missing else "native_extension_build",
        "missing_headers": missing,
        "header_candidates_in_target_environment": {
            name: [str(path.resolve()) for path in prefix.rglob(name)] for name in missing
        },
        "sm120_nvcc_command_observed": bool(arch_lines),
        "sm120_nvcc_command_count": len(arch_lines),
        "source_modified": False,
        "retry_performed": False,
    }


def managed_cleanup_summary(output: Path) -> dict[str, Any]:
    records = []
    residual = []
    for path in sorted(output.glob("**/processes/*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        pid = payload.get("child_root_pid")
        records.append({
            "path": str(path.relative_to(output)),
            "cleanup_complete": payload.get("cleanup_complete"),
            "child_root_pid": pid,
        })
        proc = Path(f"/proc/{pid}/cmdline") if isinstance(pid, int) else None
        if proc and proc.is_file():
            current = proc.read_bytes().replace(b"\0", b" ").decode(errors="replace")
            expected = " ".join(str(item) for item in payload.get("command", []))
            if expected and expected in current:
                residual.append({"pid": pid, "command": current})
    return {
        "schema_version": "openeta.univtac.managed_cleanup.v1",
        "process_record_count": len(records),
        "all_records_cleanup_complete": bool(records) and all(
            item["cleanup_complete"] is True for item in records
        ),
        "matching_residual_processes": residual,
        "cleanup_residual": bool(residual),
        "records": records,
    }


def write_failure_bundle(output: Path, manifest: dict[str, Any]) -> str:
    classification = manifest.get("classification")
    if classification == "tinygltf_overlay_fetch_failed":
        name = "author_bundle_tinygltf_packaging"
    elif classification == "libuipc_sm120_build_failed_after_packaging_recovery":
        name = "author_bundle_blackwell_native"
    else:
        return "none_not_triggered"
    bundle = output / name
    bundle.mkdir(parents=True, exist_ok=True)
    write_json(bundle / "summary.json", manifest)
    (bundle / "README.md").write_text(
        "# R0.8.1 failure bundle\n\n"
        "This bundle records a fixed-source Blackwell native gate failure. "
        "It does not contain binaries, archives, credentials, or simulator artifacts.\n",
        encoding="utf-8",
    )
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--blackwell-config", type=Path, required=True)
    parser.add_argument("--r08-source", type=Path, required=True)
    parser.add_argument("--source-checkout", type=Path, required=True)
    parser.add_argument("--vcpkg-source", type=Path, required=True)
    parser.add_argument("--vcpkg-checkout", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--conda-exe", type=Path, required=True)
    parser.add_argument("--historical-root", type=Path, action="append", default=[])
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    blackwell = yaml.safe_load(args.blackwell_config.read_text())
    output = args.output_root.resolve()
    source = args.source_checkout.resolve()
    vcpkg = args.vcpkg_checkout.resolve()
    prefix = Path(_run([str(args.conda_exe), "info", "--base"]).stdout.strip()) / "envs" / cfg["environment"]["conda_name"]
    if output.exists() or source.exists() or vcpkg.exists():
        raise FileExistsError("R0.8.1 output/source/vcpkg roots must all be absent")
    output.mkdir(parents=True)
    manifest = {
        "schema_version": "openeta.univtac.r081.v1",
        "runtime_variant": cfg["runtime_variant"],
        "started_at": utc_now(),
        "status": "running",
        "classification": None,
        "gates": {},
        "isaac_sim_started": False,
        "sysctl_modified": False,
    }
    write_json(output / "run_manifest.json", manifest)
    before = {}
    try:
        before = {
            "legacy": fingerprint_environment(args.conda_exe, "UniVTAC"),
            "r07": fingerprint_environment(args.conda_exe, "UniVTAC-isaac51-r07"),
            "r08": fingerprint_environment(args.conda_exe, cfg["environment"]["conda_name"]),
            "openeta_uv": uv_fingerprint(),
        }
        write_json(output / "environment_fingerprints" / "before.json", before)
        subprocess.run(["git", "clone", "--no-local", "--no-checkout", str(args.r08_source), str(source)], check=True)
        subprocess.run(["git", "checkout", "--detach", cfg["source"]["univtac_commit"]], cwd=source, check=True)
        if _git(source, "status", "--short"):
            raise RuntimeError("fresh source checkout is dirty")
        subprocess.run(["git", "clone", "--no-local", "--no-checkout", str(args.vcpkg_source), str(vcpkg)], check=True)
        subprocess.run(["git", "checkout", "--detach", VCPKG_COMMIT], cwd=vcpkg, check=True)
        original_port = vcpkg / "ports/tinygltf/portfile.cmake"
        original_json = vcpkg / "ports/tinygltf/vcpkg.json"
        pinned = {
            "vcpkg_commit": _git(vcpkg, "rev-parse", "HEAD"),
            "portfile_sha256": hashlib.sha256(original_port.read_bytes()).hexdigest(),
            "vcpkg_json_sha256": hashlib.sha256(original_json.read_bytes()).hexdigest(),
            "vcpkg_json": json.loads(original_json.read_text()),
            "expected_sha512": EXPECTED_SHA512,
        }
        write_json(output / "archive_drift" / "pinned_port.json", pinned)

        tag = cfg["source"]["tinygltf_tag"]
        ls_remote = _run(["git", "ls-remote", cfg["source"]["tinygltf_url"], f"refs/tags/{tag}", f"refs/tags/{tag}^{{}}"])
        refs = {line.split()[1]: line.split()[0] for line in ls_remote.stdout.splitlines() if len(line.split()) == 2}
        tag_commit = refs.get(f"refs/tags/{tag}^{{}}") or refs.get(f"refs/tags/{tag}")
        write_json(output / "archive_drift" / "tag_ref.json", {"tag": tag, "refs": refs, "resolved_commit": tag_commit})
        try:
            require_fixed_tag_commit(tag_commit)
        except ValueError:
            manifest["classification"] = "tinygltf_tag_moved"
            raise RuntimeError("tinygltf tag moved")

        tiny_checkout = output / "archive_drift" / "tinygltf_exact_checkout"
        subprocess.run(["git", "clone", cfg["source"]["tinygltf_url"], str(tiny_checkout)], check=True)
        subprocess.run(["git", "checkout", "--detach", TINYGLTF_COMMIT], cwd=tiny_checkout, check=True)
        tag_archive = output / "archive_drift" / "tinygltf_tag.tar.gz"
        commit_archive = output / "archive_drift" / "tinygltf_commit.tar.gz"
        archives = {
            "tag": _download(f"https://github.com/syoyo/tinygltf/archive/refs/tags/{tag}.tar.gz", tag_archive),
            "commit": _download(f"https://github.com/syoyo/tinygltf/archive/{TINYGLTF_COMMIT}.tar.gz", commit_archive),
        }
        write_json(output / "archive_drift" / "archive_hashes.json", archives)
        baseline = git_manifest(tiny_checkout)
        tag_tree = archive_manifest(tag_archive)
        commit_tree = archive_manifest(commit_archive)
        tree_comparison = {
            "tag_vs_checkout": compare_manifests(baseline, tag_tree),
            "commit_vs_checkout": compare_manifests(baseline, commit_tree),
            "tag_vs_commit": compare_manifests(tag_tree, commit_tree),
            "focus_files": {name: baseline.get(name) for name in ("tiny_gltf.h", "LICENSE", "CMakeLists.txt", "json.hpp", "stb_image.h", "stb_image_write.h")},
        }
        write_json(output / "archive_drift" / "normalized_tree_comparison.json", tree_comparison)
        if not all(value["equal"] for key, value in tree_comparison.items() if key.endswith("checkout") or key == "tag_vs_commit"):
            manifest["classification"] = "tinygltf_source_tree_mismatch"
            raise RuntimeError("tinygltf normalized source trees differ")

        historical = find_historical_asset(args.historical_root)
        write_json(output / "historical_asset_search" / "result.json", historical)
        overlay_root = output / "generated_overlay_ports"
        recovery = recovery_strategy(bool(historical["historical_asset_recovered"]))
        downloads = output / "vcpkg_resolution" / "downloads"
        downloads.mkdir(parents=True)
        if historical["historical_asset_recovered"]:
            shutil.copy2(historical["asset"]["path"], downloads / "syoyo-tinygltf-v2.9.6.tar.gz")
            overlay_path = None
        else:
            tiny_overlay = overlay_root / "tinygltf"
            tiny_overlay.mkdir(parents=True)
            shutil.copy2(original_json, tiny_overlay / "vcpkg.json")
            overlay_text, diff = generate_overlay_portfile(original_port.read_text())
            (tiny_overlay / "portfile.cmake").write_text(overlay_text)
            diff.update({"overlay_count": 1, "vcpkg_json_byte_identical": (tiny_overlay / "vcpkg.json").read_bytes() == original_json.read_bytes()})
            write_json(overlay_root / "overlay_semantic_diff.json", diff)
            overlay_path = overlay_root
        manifest["packaging_recovery_status"] = recovery
        manifest["gates"]["A0_A2"] = "passed"

        short_probe = _managed(
            [str(prefix / "bin/python"), str(ROOT / "scripts/univtac/probe_blackwell_torch.py"), "--output", str(output / "preflight" / "torch_probe.json")],
            cwd=source, root=output / "preflight", name="torch_probe", timeout=600, environment=clean_environment(),
        )
        if not _process_ok(short_probe) or classify_torch_probe(json.loads((output / "preflight/torch_probe.json").read_text())) != "passed":
            manifest["classification"] = "blocked_by_external_resources"
            raise RuntimeError("R0.8 environment witness failed")
        blackwell["paths"] = {"cc_wrapper": str(source / "scripts/toolchains/gcc12-system-ld"), "cxx_wrapper": str(source / "scripts/toolchains/gxx12-system-ld")}
        native_env = native_environment(prefix, blackwell, vcpkg)
        native_env["VCPKG_DOWNLOADS"] = str(downloads)
        if overlay_path:
            native_env["VCPKG_OVERLAY_PORTS"] = str(overlay_path)
        bootstrap = _managed([str(vcpkg / "bootstrap-vcpkg.sh"), "-disableMetrics"], cwd=vcpkg, root=output / "vcpkg_resolution", name="bootstrap", timeout=3600, environment=native_env)
        if not _process_ok(bootstrap):
            manifest["classification"] = "blocked_by_external_resources"
            raise RuntimeError("vcpkg bootstrap failed")
        uipc = source / "third_party/TacEx/source/tacex_uipc"
        build = _managed(
            ["/usr/bin/time", "-v", str(prefix / "bin/python"), "-m", "pip", "install", "--no-build-isolation", "-e", str(uipc)],
            cwd=uipc, root=output / "libuipc_build", name="u0r", timeout=cfg["limits"]["libuipc_build_timeout_seconds"], environment=native_env,
        )
        build_text = Path(build["log_path"]).read_text(errors="replace")
        cmake_cache = uipc / "build/CMakeCache.txt"
        cmake_text = cmake_cache.read_text(errors="replace") if cmake_cache.is_file() else ""
        cuda_object_count = len(list((uipc / "build").rglob("*.cu.o")))
        write_json(output / "libuipc_build" / "result.json", {
            "returncode": build.get("returncode"),
            "elapsed_seconds": build.get("elapsed_seconds"),
            "entered_cuda_compilation": "CMAKE_CUDA_COMPILER" in cmake_text and cuda_object_count > 0,
            "cmake_cuda_architectures": "120" if "CMAKE_CUDA_ARCHITECTURES:UNINITIALIZED=120" in cmake_text else None,
            "cuda_object_count": cuda_object_count,
            "additional_archive_drift": "unexpected hash" in build_text,
        })
        if not _process_ok(build):
            if "unexpected hash" in build_text:
                manifest["classification"] = "additional_vcpkg_archive_drift"
            elif "tinygltf" in build_text and any(
                token in build_text.lower() for token in ("failed to fetch", "failed to clone", "error: downloading")
            ):
                manifest["classification"] = "tinygltf_overlay_fetch_failed"
            else:
                manifest["classification"] = "libuipc_sm120_build_failed_after_packaging_recovery"
            raise RuntimeError("U0R failed")
        manifest["gates"]["U0R"] = "passed"

        tiny_sources = sorted((vcpkg / "buildtrees/tinygltf/src").glob("*.clean"))
        if len(tiny_sources) != 1:
            raise RuntimeError("expected one resolved tinygltf buildtree")
        identity = _managed(
            [
                sys.executable,
                str(ROOT / "scripts/univtac/audit_tinygltf_build_result.py"),
                "--exact-checkout",
                str(tiny_checkout),
                "--buildtree",
                str(tiny_sources[0]),
                "--installed-header",
                str(uipc / "build/vcpkg_installed/x64-linux/include/tiny_gltf.h"),
                "--output",
                str(output / "vcpkg_resolution/tinygltf_source_identity.json"),
                "--cmake-cache",
                str(uipc / "build/CMakeCache.txt"),
                "--cuda-object-root",
                str(uipc / "build"),
                "--compile-evidence-output",
                str(output / "libuipc_build/compile_evidence.json"),
                "--build-result",
                str(output / "libuipc_build/result.json"),
                "--vcpkg-root",
                str(vcpkg),
                "--overlay-root",
                str(overlay_root),
                "--resolution-log",
                str(vcpkg / "buildtrees/tinygltf/stdout-x64-linux.log"),
                "--provenance-output",
                str(output / "vcpkg_resolution/provenance.json"),
            ],
            cwd=source,
            root=output / "vcpkg_resolution",
            name="source_identity",
            timeout=300,
            environment=clean_environment(),
        )
        _require_process("tinygltf source identity", identity)

        binaries = _binary_candidates([uipc / "build", prefix], r"uipc|pyuipc|muda|backend.*cuda")
        audit = _managed([str(prefix / "bin/python"), str(ROOT / "scripts/univtac/audit_blackwell_binaries.py"), "--output", str(output / "libuipc_binary_audit/audit.json"), *map(str, binaries)], cwd=source, root=output / "libuipc_binary_audit", name="audit", timeout=1200, environment=native_env)
        audit_payload = json.loads((output / "libuipc_binary_audit/audit.json").read_text()) if (output / "libuipc_binary_audit/audit.json").is_file() else {"records": []}
        audit_class = binary_architecture_classification(audit_payload.get("records", []))
        manifest["gates"]["B0"] = audit_class
        if not _process_ok(audit) or audit_class != "passed":
            manifest["classification"] = "libuipc_binary_missing_sm120_code"
            raise RuntimeError("B0 failed")
        for index, timeout in ((1, cfg["limits"]["core_run1_timeout_seconds"]), (2, cfg["limits"]["core_run2_timeout_seconds"])):
            core = output / f"libuipc_core_run{index}"
            process = _managed([str(prefix / "bin/python"), str(ROOT / "scripts/univtac/probe_libuipc_core.py"), "--workspace", str(core / "workspace"), "--output", str(core / "probe.json"), "--events", str(core / "events.jsonl")], cwd=source, root=core, name="probe", timeout=timeout, environment=native_env)
            payload = json.loads((core / "probe.json").read_text()) if (core / "probe.json").exists() else {}
            classification = classify_core_probe(payload, timed_out=bool(process.get("timed_out")))
            if classification != "passed" or not _process_ok(process):
                manifest["classification"] = classification
                raise RuntimeError(f"U1 run {index} failed")
        manifest["libuipc_build_status"] = "passed"
        manifest["libuipc_core_status"] = "passed"

        curobo = source / "third_party/curobo"
        if curobo.exists():
            raise FileExistsError(f"unexpected preexisting cuRobo checkout: {curobo}")
        clone = _managed(
            ["git", "clone", "https://github.com/NVlabs/curobo.git", str(curobo)],
            cwd=source, root=output / "curobo_build", name="clone", timeout=1200, environment=clean_environment(),
        )
        _require_process("cuRobo clone", clone)
        checkout = _run(["git", "checkout", "--detach", blackwell["source"]["curobo_commit"]], cwd=curobo)
        if checkout.returncode:
            raise RuntimeError("cuRobo pinned checkout failed")
        curobo_build = _managed(
            ["/usr/bin/time", "-v", str(prefix / "bin/python"), "-m", "pip", "install", "--no-build-isolation", "-e", str(curobo)],
            cwd=curobo, root=output / "curobo_build", name="build", timeout=cfg["limits"]["curobo_build_timeout_seconds"], environment=native_env,
        )
        manifest["gates"]["C0_BUILD"] = "passed" if _process_ok(curobo_build) else "curobo_sm120_build_failed"
        if not _process_ok(curobo_build):
            manifest["classification"] = "curobo_sm120_build_failed"
            manifest["curobo_status"] = "failed"
            write_json(
                output / "curobo_build/failure_summary.json",
                curobo_failure_summary(Path(curobo_build["log_path"]), prefix),
            )
            raise RuntimeError("cuRobo build failed")
        curobo_binaries = _binary_candidates([curobo, prefix], r"_cu")
        curobo_audit_root = output / "curobo_binary_audit"
        curobo_audit = _managed(
            [str(prefix / "bin/python"), str(ROOT / "scripts/univtac/audit_blackwell_binaries.py"), "--output", str(curobo_audit_root / "audit.json"), *map(str, curobo_binaries)],
            cwd=curobo, root=curobo_audit_root, name="audit", timeout=1200, environment=native_env,
        )
        curobo_payload = json.loads((curobo_audit_root / "audit.json").read_text()) if (curobo_audit_root / "audit.json").is_file() else {"records": []}
        curobo_class = binary_architecture_classification(curobo_payload.get("records", []))
        manifest["gates"]["C0_AUDIT"] = curobo_class
        if not _process_ok(curobo_audit) or curobo_class != "passed":
            manifest["classification"] = "curobo_sm120_build_failed"
            raise RuntimeError("cuRobo binary audit failed")
        curobo_smoke = _managed(
            [str(prefix / "bin/python"), str(ROOT / "scripts/univtac/probe_blackwell_curobo.py"), "--source-root", str(curobo), "--output", str(output / "curobo_smoke/probe.json")],
            cwd=curobo, root=output / "curobo_smoke", name="probe", timeout=blackwell["timeouts_seconds"]["curobo_smoke"], environment=native_env,
        )
        manifest["gates"]["C0_SMOKE"] = "passed" if _process_ok(curobo_smoke) else "blackwell_native_bridge_partially_viable"
        manifest["classification"] = "blackwell_native_bridge_viable" if _process_ok(curobo_smoke) else "blackwell_native_bridge_partially_viable"
        manifest["status"] = "completed"
        manifest["curobo_status"] = "passed" if _process_ok(curobo_smoke) else "failed"
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest.setdefault("libuipc_build_status", "failed_or_not_run")
        manifest.setdefault("libuipc_core_status", "not_run_due_to_gate")
        manifest.setdefault("curobo_status", "not_run_due_to_gate")
        manifest["failure"] = {"class": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        raise
    finally:
        for gate in ("C0_AUDIT", "C0_SMOKE"):
            manifest["gates"].setdefault(gate, "not_run_due_to_gate")
        if manifest["gates"].get("C0_BUILD") == "curobo_sm120_build_failed":
            manifest["curobo_status"] = "failed"
        try:
            after = {
                "legacy": fingerprint_environment(args.conda_exe, "UniVTAC"),
                "r07": fingerprint_environment(args.conda_exe, "UniVTAC-isaac51-r07"),
                "r08": fingerprint_environment(args.conda_exe, cfg["environment"]["conda_name"]),
                "openeta_uv": uv_fingerprint(),
            }
            write_json(output / "environment_fingerprints" / "after.json", after)
            comparison = {
                key: compare_fingerprints(before[key], after[key])
                for key in ("legacy", "r07", "r08")
                if key in before
            }
            comparison["openeta_uv_same"] = before.get("openeta_uv") == after.get("openeta_uv")
            write_json(output / "environment_fingerprints" / "comparison.json", comparison)
        except Exception as exc:  # noqa: BLE001 - preserve the primary gate failure
            manifest["fingerprint_error"] = str(exc)
        if source.exists():
            write_json(output / "final_resources/source.json", {"head": _git(source, "rev-parse", "HEAD"), "status": _git(source, "status", "--short"), "tracked_diff": _git(source, "diff", "--")})
        if vcpkg.exists():
            write_json(output / "final_resources/vcpkg.json", {"head": _git(vcpkg, "rev-parse", "HEAD"), "status": _git(vcpkg, "status", "--short"), "portfile_sha256": hashlib.sha256((vcpkg / "ports/tinygltf/portfile.cmake").read_bytes()).hexdigest()})
        write_json(output / "final_resources/cleanup.json", managed_cleanup_summary(output))
        manifest["author_bundle_type"] = write_failure_bundle(output, manifest)
        manifest["ended_at"] = utc_now()
        write_json(output / "summary.json", manifest)
        write_json(output / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
