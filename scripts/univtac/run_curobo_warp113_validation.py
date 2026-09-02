#!/usr/bin/env python3
"""Run the fixed-source cuRobo Warp public-API backport validation gates."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.univtac.resume_blackwell_native_bridge import managed_cleanup_summary
from scripts.univtac.run_blackwell_native_bridge import _managed, _process_ok, uv_fingerprint
from scripts.univtac.run_isaac51_runtime_gates import (
    clean_environment,
    compare_fingerprints,
    fingerprint_environment,
)
from sim.envs.univtac.cuda_include_bridge import bridge_environment
from sim.envs.univtac.curobo_build_validation import EXTENSIONS, find_extension_binaries
from sim.envs.univtac.curobo_runtime_validation import (
    parse_ik_metrics,
    static_audit_covers_binaries,
    static_audit_process_usable,
)
from sim.envs.univtac.elf_loader_closure import unexpected_static_missing
from sim.envs.univtac.libuipc_core_probe import classify_core_probe
from sim.envs.univtac.resource_sanitation import utc_now, write_json
from sim.envs.univtac.source_backport_contract import (
    BASE_COMMIT,
    TARGET_FILE,
    UPSTREAM_REFERENCE_COMMIT,
    compiler_wrapper_paths,
    corrective_retry_accounting,
    file_sha256,
    validate_compiler_wrappers,
    validate_semantic_backport,
)


def git(path: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=path, check=True, capture_output=True, text=True
    ).stdout.strip()


def runtime_environment(prefix: Path, pinned_source: Path) -> tuple[dict[str, str], dict]:
    wrappers = validate_compiler_wrappers(pinned_source)
    paths = compiler_wrapper_paths(pinned_source)
    environment = bridge_environment(
        clean_environment(
            {
                "PATH": f"{prefix / 'bin'}:{os.environ.get('PATH', '')}",
                "LD_LIBRARY_PATH": str(prefix / "lib"),
                "CUDA_LAUNCH_BLOCKING": "1",
                "CC": str(paths["CC"]),
                "CXX": str(paths["CXX"]),
                "CUDAHOSTCXX": str(paths["CUDAHOSTCXX"]),
            }
        ),
        prefix,
    )
    for key in ("CPATH", "C_INCLUDE_PATH", "CPLUS_INCLUDE_PATH", "LD_PRELOAD", "TORCH_USE_RTLD_GLOBAL"):
        environment.pop(key, None)
    return environment, wrappers


def editable_source(python: Path) -> str | None:
    result = subprocess.run(
        [str(python), "-m", "pip", "show", "nvidia-curobo"],
        check=True,
        capture_output=True,
        text=True,
    )
    prefix = "Editable project location:"
    return next(
        (line.split(":", 1)[1].strip() for line in result.stdout.splitlines() if line.startswith(prefix)),
        None,
    )


def pip_dependency_lines(python: Path) -> dict[str, list[str]]:
    result = subprocess.run(
        [str(python), "-m", "pip", "freeze"], check=True, capture_output=True, text=True
    )
    lines = sorted(line.strip() for line in result.stdout.splitlines() if line.strip())
    curobo = [line for line in lines if "#egg=nvidia_curobo" in line or line.lower().startswith("nvidia-curobo")]
    return {"curobo": curobo, "dependencies": [line for line in lines if line not in curobo]}


def run_example(
    *, python: Path, example: Path, expected_root: Path, expected_extensions: tuple[str, ...],
    output: Path, environment: dict[str, str], timeout: int
) -> dict:
    command = [
        str(python),
        str(ROOT / "scripts/univtac/run_official_curobo_example.py"),
        "--example",
        str(example),
        "--expected-root",
        str(expected_root),
    ]
    for name in expected_extensions:
        command.extend(("--expected-extension", name))
    command.extend(("--output", str(output / "result.json")))
    return _managed(
        command, cwd=expected_root, root=output, name="probe", timeout=timeout, environment=environment
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--pinned-isaac51-source", type=Path, required=True)
    parser.add_argument("--base-curobo", type=Path, required=True)
    parser.add_argument("--derived-curobo", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--patch-manifest", type=Path, required=True)
    parser.add_argument("--invalid-invocation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--conda-exe", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    output = args.output_root.resolve()
    source = args.pinned_isaac51_source.resolve()
    base = args.base_curobo.resolve()
    derived = args.derived_curobo.resolve()
    patch = args.patch.resolve()
    invalid_root = args.invalid_invocation_root.resolve()
    invalid_process = invalid_root / "processes/build.json"
    invalid_log = invalid_root / "logs/build.log"
    if not invalid_process.is_file() or not invalid_log.is_file():
        raise FileNotFoundError("invalid orchestration invocation evidence is incomplete")
    invalid_payload = json.loads(invalid_process.read_text(encoding="utf-8"))
    invalid_text = invalid_log.read_text(encoding="utf-8", errors="replace")
    if (
        invalid_payload.get("returncode") != 1
        or invalid_payload.get("timed_out") is not False
        or invalid_payload.get("cleanup_complete") is not True
        or "No such file or directory" not in invalid_text
    ):
        raise ValueError("invalid invocation evidence does not describe the wrapper-path failure")
    if output.exists() or derived.exists():
        raise FileExistsError("output and derived cuRobo roots must be fresh")
    output.mkdir(parents=True)
    manifest = {
        "runtime_variant": cfg["runtime_variant"],
        "status": "running",
        "classification": None,
        "gates": {},
        "official_recipe_exact": False,
        "benchmark_reproduction": False,
        "isaac_started": False,
        "agent_started": False,
        "started_at": utc_now(),
        **corrective_retry_accounting(),
    }
    write_json(output / "run_manifest.json", manifest)
    invalid_output = output / "curobo_build/invalid_orchestration_invocation"
    (invalid_output / "processes").mkdir(parents=True)
    (invalid_output / "logs").mkdir(parents=True)
    shutil.copy2(invalid_process, invalid_output / "processes/build.json")
    shutil.copy2(invalid_log, invalid_output / "logs/build.log")
    before = {}
    before_uv = {}
    before_editable = None
    before_pip = {}
    runtime_gates_completed = False
    try:
        if git(source, "rev-parse", "HEAD") != cfg["source"]["univtac_commit"] or git(source, "status", "--short", "--untracked-files=no"):
            raise RuntimeError("pinned Isaac51 source is not clean and fixed")
        if git(base, "rev-parse", "HEAD") != BASE_COMMIT or git(base, "status", "--short", "--untracked-files=no"):
            raise RuntimeError("base cuRobo checkout is not clean and fixed")
        if git(base, "cat-file", "-t", UPSTREAM_REFERENCE_COMMIT) != "commit":
            raise RuntimeError("fixed upstream Warp reference commit is unavailable")
        conda_base = subprocess.run(
            [str(args.conda_exe), "info", "--base"], check=True, capture_output=True, text=True
        ).stdout.strip()
        prefix = Path(conda_base) / "envs" / cfg["environment"]["conda_name"]
        python = prefix / "bin/python"
        environment, wrappers = runtime_environment(prefix, source)
        version_probes = {}
        for variable, record in wrappers.items():
            probe = subprocess.run(
                [record["realpath"], "--version"],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            version_probes[variable] = {
                "returncode": probe.returncode,
                "first_line": (probe.stdout or probe.stderr).splitlines()[0]
                if (probe.stdout or probe.stderr).splitlines()
                else "",
            }
        if any(item["returncode"] != 0 for item in version_probes.values()):
            manifest["classification"] = "compiler_wrapper_precondition_failed"
            raise RuntimeError("compiler wrapper version probe failed before pip invocation")
        write_json(
            output / "preflight/compiler_wrappers.json",
            {"wrappers": wrappers, "version_probes": version_probes},
        )
        before = {
            name: fingerprint_environment(args.conda_exe, env)
            for name, env in (("legacy", "UniVTAC"), ("r07", "UniVTAC-isaac51-r07"), ("r08", cfg["environment"]["conda_name"]))
        }
        before_uv = uv_fingerprint()
        before_editable = editable_source(python)
        before_pip = pip_dependency_lines(python)
        write_json(output / "environment_fingerprints/before.json", before)

        w0 = _managed(
            [str(python), str(ROOT / "scripts/univtac/audit_warp_api_compatibility.py"), "--target-prefix", str(prefix), "--output", str(output / "warp_provenance/audit.json")],
            cwd=ROOT, root=output / "warp_provenance", name="probe", timeout=cfg["timeouts_seconds"]["warp_api"], environment=environment,
        )
        if not _process_ok(w0):
            manifest["classification"] = "warp_public_api_unavailable"
            raise RuntimeError("W0 failed")
        manifest["gates"]["W0"] = "passed"

        patch_manifest = json.loads(args.patch_manifest.read_text(encoding="utf-8"))
        if patch_manifest["base_commit"] != BASE_COMMIT or patch_manifest["upstream_reference_commit"] != UPSTREAM_REFERENCE_COMMIT or patch_manifest["patch_sha256"] != file_sha256(patch):
            raise RuntimeError("fixed backport manifest mismatch")
        subprocess.run(["git", "worktree", "add", "--detach", str(derived), BASE_COMMIT], cwd=base, check=True)
        subprocess.run(["git", "apply", "--check", str(patch)], cwd=derived, check=True)
        subprocess.run(["git", "apply", str(patch)], cwd=derived, check=True)
        before_source = git(base, "show", f"{BASE_COMMIT}:{TARGET_FILE}") + "\n"
        after_source = (derived / TARGET_FILE).read_text(encoding="utf-8")
        write_json(output / "backport/patch_semantic_audit.json", validate_semantic_backport(before_source, after_source))
        manifest["gates"]["W1"] = "passed"

        w2 = _managed(
            [str(python), str(ROOT / "scripts/univtac/probe_warp_torch_interop.py"), "--output", str(output / "warp_public_api_probe/probe.json")],
            cwd=ROOT, root=output / "warp_public_api_probe", name="probe", timeout=cfg["timeouts_seconds"]["warp_kernel"], environment=environment,
        )
        if not _process_ok(w2):
            manifest["classification"] = "warp_sm120_public_api_kernel_failed"
            raise RuntimeError("W2 failed")
        manifest["gates"]["W2"] = "passed"

        build = _managed(
            ["/usr/bin/time", "-v", str(python), "-m", "pip", "install", "--no-deps", "--no-build-isolation", "--verbose", "-e", str(derived)],
            cwd=derived, root=output / "curobo_build/c0r1_authorized_retry", name="build", timeout=cfg["timeouts_seconds"]["curobo_build"], environment=environment,
        )
        if not _process_ok(build):
            manifest["classification"] = "curobo_warp113_rebuild_failed"
            raise RuntimeError("C0 failed")
        manifest["gates"]["C0"] = "passed"

        binaries = find_extension_binaries(derived)
        binary_audit = _managed(
            [str(python), str(ROOT / "scripts/univtac/audit_blackwell_binaries.py"), "--output", str(output / "curobo_binary_audit/audit.json"), *map(str, binaries)],
            cwd=derived, root=output / "curobo_binary_audit", name="audit", timeout=1200, environment=environment,
        )
        static = json.loads((output / "curobo_binary_audit/audit.json").read_text(encoding="utf-8"))
        if not static_audit_process_usable(binary_audit, static) or not static_audit_covers_binaries(static, binaries) or unexpected_static_missing(item.split(" =>", 1)[0] for item in static["missing_dependencies"]) or any(not item["has_sm120_or_compute120"] or item["has_forbidden_fallback"] for item in static["records"]):
            manifest["classification"] = "curobo_post_backport_import_failed"
            raise RuntimeError("C1 static audit failed")
        for name, binary in zip(EXTENSIONS, binaries, strict=True):
            probe_root = output / "curobo_import" / name
            probe = _managed(
                [str(python), str(ROOT / "scripts/univtac/audit_pytorch_extension_loader.py"), "--mode", "extension", "--module", f"curobo.curobolib.{name}", "--binary", str(binary), "--expected-root", str(derived), "--target-prefix", str(prefix), "--forbidden-root", str(prefix.parent / "UniVTAC"), "--forbidden-root", str(prefix.parent / "UniVTAC-isaac51-r07"), "--output", str(probe_root / "result.json")],
                cwd=derived, root=probe_root, name="probe", timeout=cfg["timeouts_seconds"]["loader_probe"], environment=environment,
            )
            if not _process_ok(probe):
                manifest["classification"] = "curobo_post_backport_import_failed"
                raise RuntimeError(f"C1 import failed for {name}")
        manifest["gates"]["C1"] = "passed"

        kin = run_example(python=python, example=derived / "examples/kinematics_example.py", expected_root=derived, expected_extensions=("kinematics_fused_cu",), output=output / "kinematics_regression", environment=environment, timeout=cfg["timeouts_seconds"]["kinematics"])
        if not _process_ok(kin):
            manifest["classification"] = "curobo_kinematics_regression"
            raise RuntimeError("C2 kinematics failed")
        ik_root = output / "ik_regression"
        ik = run_example(python=python, example=derived / "examples/ik_example.py", expected_root=derived, expected_extensions=EXTENSIONS, output=ik_root, environment=environment, timeout=cfg["timeouts_seconds"]["ik"])
        if not _process_ok(ik):
            manifest["classification"] = "curobo_ik_regression"
            raise RuntimeError("C2 IK failed")
        write_json(ik_root / "metrics.json", {"runs": parse_ik_metrics(Path(ik["log_path"]).read_text(errors="replace"))})
        manifest["gates"]["C2"] = "passed"

        collision = run_example(python=python, example=derived / "examples/collision_check_example.py", expected_root=derived, expected_extensions=("geom_cu",), output=output / "collision_official", environment=environment, timeout=cfg["timeouts_seconds"]["collision"])
        if not _process_ok(collision):
            result = json.loads((output / "collision_official/result.json").read_text(encoding="utf-8"))
            manifest["classification"] = "curobo_warp113_backport_incomplete" if "warp' has no attribute 'torch" in str(result.get("error_message")) else "curobo_collision_kernel_failed"
            raise RuntimeError("C3 failed")
        manifest["gates"]["C3"] = "passed"
        finite = _managed(
            [str(python), str(ROOT / "scripts/univtac/probe_curobo_collision_finite.py"), "--expected-root", str(derived), "--output", str(output / "collision_finite/probe.json")],
            cwd=derived, root=output / "collision_finite", name="probe", timeout=cfg["timeouts_seconds"]["collision_finite"], environment=environment,
        )
        if not _process_ok(finite):
            manifest["classification"] = "curobo_collision_output_invalid"
            raise RuntimeError("C4 failed")
        manifest["gates"]["C4"] = "passed"

        u2root = output / "libuipc_regression"
        u2 = _managed(
            [str(python), str(ROOT / "scripts/univtac/probe_libuipc_core.py"), "--workspace", str(u2root / "workspace"), "--output", str(u2root / "probe.json"), "--events", str(u2root / "events.jsonl")],
            cwd=source, root=u2root, name="probe", timeout=cfg["timeouts_seconds"]["libuipc_regression"], environment=environment,
        )
        upayload = json.loads((u2root / "probe.json").read_text(encoding="utf-8"))
        if not _process_ok(u2) or classify_core_probe(upayload, timed_out=bool(u2.get("timed_out"))) != "passed":
            manifest["classification"] = "libuipc_regression_failed"
            raise RuntimeError("U2 failed")
        manifest["gates"]["U2"] = "passed"
        runtime_gates_completed = True
    except BaseException as exc:
        manifest.update({"status": "failed", "classification": manifest["classification"] or "blocked_by_external_resources", "failure": {"class": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}})
        raise
    finally:
        for gate in ("W0", "W1", "W2", "C0", "C1", "C2", "C3", "C4", "U2"):
            manifest["gates"].setdefault(gate, "not_run_due_to_gate")
        environment_valid = False
        cleanup_valid = False
        try:
            after = {name: fingerprint_environment(args.conda_exe, env) for name, env in (("legacy", "UniVTAC"), ("r07", "UniVTAC-isaac51-r07"), ("r08", cfg["environment"]["conda_name"]))}
            after_uv = uv_fingerprint()
            after_editable = editable_source(python)
            after_pip = pip_dependency_lines(python)
            write_json(output / "environment_fingerprints/after.json", after)
            semantic_diff = {
                "allowed_change": "nvidia-curobo editable source only",
                "legacy_unchanged": compare_fingerprints(before.get("legacy", {}), after["legacy"])["unchanged"],
                "r07_unchanged": compare_fingerprints(before.get("r07", {}), after["r07"])["unchanged"],
                "target_conda_explicit_unchanged": before.get("r08", {}).get("commands", {}).get("conda_explicit", {}).get("sha256") == after["r08"]["commands"]["conda_explicit"]["sha256"],
                "target_pip_change_expected": before.get("r08", {}).get("commands", {}).get("pip_freeze", {}).get("sha256") != after["r08"]["commands"]["pip_freeze"]["sha256"],
                "editable_before": before_editable,
                "editable_after": after_editable,
                "openeta_uv_unchanged": before_uv == after_uv,
                "pip_dependencies_unchanged": before_pip.get("dependencies")
                == after_pip["dependencies"],
                "curobo_freeze_before": before_pip.get("curobo"),
                "curobo_freeze_after": after_pip["curobo"],
            }
            environment_valid = bool(
                semantic_diff["legacy_unchanged"]
                and semantic_diff["r07_unchanged"]
                and semantic_diff["target_conda_explicit_unchanged"]
                and semantic_diff["target_pip_change_expected"]
                and before_editable == str(base)
                and after_editable == str(derived)
                and semantic_diff["openeta_uv_unchanged"]
                and semantic_diff["pip_dependencies_unchanged"]
            )
            semantic_diff["valid"] = environment_valid
            write_json(output / "environment_semantic_diff.json", semantic_diff)
        except BaseException as exc:  # noqa: BLE001
            manifest["environment_fingerprint_error"] = str(exc)
        try:
            cleanup = managed_cleanup_summary(output)
            cleanup_valid = bool(cleanup["all_records_cleanup_complete"] and not cleanup["cleanup_residual"])
            write_json(output / "final_resources/cleanup.json", cleanup)
        except BaseException as exc:  # noqa: BLE001
            manifest["cleanup_error"] = str(exc)
        if runtime_gates_completed and environment_valid and cleanup_valid:
            manifest.update(
                {
                    "status": "completed",
                    "classification": "blackwell_native_bridge_viable",
                    "author_contact_recommendation": "not_needed",
                }
            )
        elif runtime_gates_completed:
            manifest.update(
                {
                    "status": "failed",
                    "classification": "cleanup_incomplete"
                    if environment_valid
                    else "unexpected_environment_dependency_change",
                }
            )
        manifest["ended_at"] = utc_now()
        write_json(output / "summary.json", manifest)
        write_json(output / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
