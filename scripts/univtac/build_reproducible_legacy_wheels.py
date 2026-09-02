#!/usr/bin/env python3
"""Build byte-reproducible wheels from the three authorized legacy sdists."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import yaml
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

ROOT = Path(__file__).resolve().parents[2]
HELPER_PATH = ROOT / "sim/envs/univtac/source_to_wheel_contract.py"
SPEC = importlib.util.spec_from_file_location("univtac_source_to_wheel_contract", HELPER_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(HELPER_PATH)
HELPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(HELPER)


class BridgeError(RuntimeError):
    def __init__(self, classification: str, message: str):
        super().__init__(message)
        self.classification = classification


def fetch_json(url: str) -> dict:
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "pypi.org":
        raise BridgeError("legacy_sdist_source_identity_failed", f"unapproved metadata source: {url}")
    request = urllib.request.Request(url, headers={"User-Agent": "OpenETA-R0.9.5.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        effective = urlparse(response.geturl())
        if effective.scheme != "https" or effective.hostname != "pypi.org":
            raise BridgeError(
                "legacy_sdist_source_identity_failed",
                f"metadata request redirected to unapproved source: {response.geturl()}",
            )
        return json.load(response)


def download(url: str, path: Path, *, attempts: int) -> list[dict]:
    records = []
    for index in range(attempts):
        started = time.monotonic()
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "OpenETA-R0.9.5.1"})
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read()
                effective = response.geturl()
                headers = dict(response.headers.items())
            parsed = urlparse(effective)
            if parsed.scheme != "https" or parsed.hostname != "files.pythonhosted.org":
                raise ValueError(f"download redirected to unapproved source: {effective}")
            path.write_bytes(data)
            records.append({"attempt": index + 1, "success": True, "size": len(data), "effective_url": effective, "effective_host": urlparse(effective).hostname, "content_length": headers.get("Content-Length"), "elapsed_seconds": time.monotonic() - started})
            return records
        except Exception as exc:
            records.append({"attempt": index + 1, "success": False, "error": f"{type(exc).__name__}: {exc}", "elapsed_seconds": time.monotonic() - started})
    raise BridgeError("legacy_sdist_source_identity_failed", f"download attempts exhausted: {path.name}")


def p0a_metadata(record: dict) -> dict:
    metadata = record["metadata"]
    return {
        "name": canonicalize_name(str(metadata.get("name", ""))),
        "version": str(metadata.get("version", "")),
        "requires_python": metadata.get("requires_python"),
        "requires_dist": sorted(str(Requirement(item)) for item in metadata.get("requires_dist", []) or []),
        "provides_extra": sorted(metadata.get("provides_extra", []) or []),
    }


def run(command: list[str], *, cwd: Path, environment: dict[str, str], log: Path, timeout: int) -> dict:
    log.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(command, cwd=cwd, env=environment, check=False, capture_output=True, text=True, timeout=timeout, preexec_fn=lambda: os.umask(0o022))
    log.write_text("COMMAND: " + " ".join(command) + "\n" + completed.stdout + completed.stderr, encoding="utf-8")
    return {"command": command, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr, "log_path": str(log.resolve())}


def builder_environment(venv: Path, temporary: Path, fixed: dict[str, str]) -> dict[str, str]:
    keep = {key: os.environ[key] for key in ("HOME", "USER", "LOGNAME", "PATH") if key in os.environ}
    keep.update(fixed)
    keep["SOURCE_DATE_EPOCH"] = "315532800"
    keep["PATH"] = f"{venv / 'bin'}:{keep.get('PATH', '')}"
    keep["TMPDIR"] = str(temporary)
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy", "PIP_INDEX_URL", "PIP_EXTRA_INDEX_URL", "PYTHONPATH"):
        keep.pop(key, None)
    return keep


def import_smoke(python: Path, wheel: Path, package: dict, root: Path) -> dict:
    target = root / package["canonical_name"]
    target.mkdir(parents=True, mode=0o700)
    install = subprocess.run([str(python), "-m", "pip", "install", "--no-index", "--no-deps", "--target", str(target), str(wheel)], check=False, capture_output=True, text=True, env={**os.environ, "PYTHONNOUSERSITE": "1"}, timeout=120)
    if install.returncode:
        raise BridgeError("derived_wheel_smoke_failed", install.stderr)
    if package["canonical_name"] == "idna-ssl":
        code = "import idna_ssl,json,pathlib; print(json.dumps({'path':idna_ssl.__file__,'callable':callable(idna_ssl.patch_match_hostname)}))"
    elif package["canonical_name"] == "pyperclip":
        code = "import pyperclip,json; print(json.dumps({'path':pyperclip.__file__,'copy_callable':callable(pyperclip.copy),'paste_callable':callable(pyperclip.paste)}))"
    else:
        code = "import antlr4,json; from antlr4 import InputStream; x=InputStream('tactile'); print(json.dumps({'path':antlr4.__file__,'size':x.size,'text':str(x)}))"
    probe = subprocess.run([str(python), "-c", code], check=False, capture_output=True, text=True, env={**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONPATH": str(target)}, timeout=60)
    if probe.returncode:
        raise BridgeError("derived_wheel_smoke_failed", probe.stderr)
    payload = json.loads(probe.stdout)
    if not Path(payload["path"]).resolve().is_relative_to(target.resolve()):
        raise BridgeError("derived_wheel_smoke_failed", "import did not originate from private target")
    shutil.rmtree(target)
    return {"success": True, "witness": payload, "target_removed": not target.exists()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-lock", type=Path, required=True)
    parser.add_argument("--p0a-report", type=Path, required=True)
    parser.add_argument("--target-python", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    HELPER.validate_config(config)
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, mode=0o750)
    output.chmod(0o750)
    state = {"status": "running", "classification": None, "stages": {stage: "not_run_due_to_gate" for stage in ("S0", "S1", "S2", "B0", "B1", "B2", "B3", "B4", "L0R")}, "build_invocations": {name: 0 for name in config["packages"]}}
    (output / "run_manifest.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    try:
        source_lock = json.loads(args.source_lock.read_text(encoding="utf-8"))
        p0a_report = json.loads(args.p0a_report.read_text(encoding="utf-8"))
        source_records = {record["name"]: record for record in source_lock["records"] if record["artifact_type"] == "sdist"}
        if set(source_records) != set(config["packages"]):
            raise BridgeError("r0951_resume_precondition_failed", "source lock does not contain exactly the three authorized sdists")
        sources = {}
        for name, package in config["packages"].items():
            record = source_records[name]
            if (record["version"], record["filename"], record["sha256"]) != (str(package["version"]), package["source_filename"], package["source_sha256"]):
                raise BridgeError("legacy_sdist_source_identity_failed", f"source lock identity changed: {name}")
            if record["url_scheme"] != "https" or record["origin_host"] != config["allowed_host"]:
                raise BridgeError("legacy_sdist_source_identity_failed", f"source URL is not approved: {name}")
            release = fetch_json(f"https://pypi.org/pypi/{name}/{package['version']}/json")
            candidates = [item for item in release["urls"] if item["filename"] == package["source_filename"] and item["packagetype"] == "sdist"]
            if len(candidates) != 1 or candidates[0].get("yanked") or candidates[0]["digests"]["sha256"] != package["source_sha256"]:
                raise BridgeError("legacy_sdist_source_identity_failed", f"official release identity changed: {name}")
            path = output / "legacy_sdists" / name / package["source_filename"]
            path.parent.mkdir(parents=True, exist_ok=True)
            attempts = download(record["url"], path, attempts=int(config["source_download_attempts"]))
            if HELPER.sha256_file(path) != package["source_sha256"]:
                raise BridgeError("legacy_sdist_source_identity_failed", f"source SHA mismatch: {name}")
            sources[name] = {"record": record, "path": path, "size": path.stat().st_size, "sha256": package["source_sha256"], "download_attempts": attempts, "pypi": {key: candidates[0].get(key) for key in ("filename", "packagetype", "size", "digests", "requires_python", "yanked", "url")}}
        state["stages"]["S0"] = "passed"

        audited = {}
        for name, source in sources.items():
            archive = HELPER.audit_tar_archive(source["path"], maximum_bytes=int(config["maximum_source_bytes"]), maximum_members=int(config["maximum_archive_members"]))
            extracted = HELPER.extract_validated_archive(source["path"], output / "source_audit" / name / "extracted", archive)
            tree = HELPER.source_tree_manifest(extracted)
            scope = HELPER.audit_pure_python_scope(extracted)
            if not scope["success"]:
                raise BridgeError("legacy_sdist_build_scope_not_pure_python", f"native build scope detected: {name}")
            metadata, pkg_info = HELPER.read_source_metadata(extracted)
            p0a_record = p0a_report["install"][int(source["record"]["source_report_item_index"])]
            resolved_metadata = p0a_metadata(p0a_record)
            if (metadata["name"], metadata["version"]) != (resolved_metadata["name"], resolved_metadata["version"]):
                raise BridgeError("derived_wheel_metadata_mismatch", f"sdist Name/Version differs from P0A: {name}")
            audited[name] = {"archive": archive, "tree": tree, "scope": scope, "source_pkg_info_metadata": metadata, "resolved_p0a_metadata": resolved_metadata, "metadata_differences": [key for key in metadata if metadata[key] != resolved_metadata[key]], "pkg_info": pkg_info}
            (output / "source_audit" / name / "summary.json").write_text(json.dumps(audited[name], indent=2, sort_keys=True) + "\n")
        state["stages"]["S1"] = "passed"
        state["stages"]["S2"] = "passed"

        tool_root = output / "builder_toolchain/wheelhouse"
        tool_root.mkdir(parents=True)
        tool_records = {}
        for name, tool in config["builder_tools"].items():
            release = fetch_json(f"https://pypi.org/pypi/{name}/{tool['version']}/json")
            candidates = [item for item in release["urls"] if item["filename"] == tool["filename"] and item["packagetype"] == "bdist_wheel"]
            if len(candidates) != 1 or urlparse(candidates[0]["url"]).hostname != config["allowed_host"] or candidates[0]["size"] != tool["size"] or candidates[0]["digests"]["sha256"] != tool["sha256"]:
                raise BridgeError("legacy_builder_toolchain_lock_failed", f"builder tool release identity changed: {name}")
            path = tool_root / tool["filename"]
            attempts = download(candidates[0]["url"], path, attempts=3)
            if path.stat().st_size != int(tool["size"]) or HELPER.sha256_file(path) != tool["sha256"]:
                raise BridgeError("legacy_builder_toolchain_lock_failed", f"builder tool artifact mismatch: {name}")
            tool_records[name] = {"path": str(path.resolve()), "version": str(tool["version"]), "filename": path.name, "size": path.stat().st_size, "sha256": tool["sha256"], "download_attempts": attempts}
        tool_manifest_sha = HELPER.deterministic_sha256([{key: item[key] for key in ("filename", "size", "sha256", "version")} for item in tool_records.values()])
        (output / "builder_toolchain/manifest.json").write_text(json.dumps({"tools": tool_records, "manifest_sha256": tool_manifest_sha}, indent=2, sort_keys=True) + "\n")
        state["stages"]["B0"] = "passed"

        builders = []
        build_results: dict[str, list[dict]] = {name: [] for name in config["packages"]}
        temp_parent = output / "builder_temporary"
        temp_parent.mkdir(mode=0o700)
        for run_index in (1, 2):
            with tempfile.TemporaryDirectory(prefix=f"builder-run{run_index}-", dir=temp_parent) as temporary_text:
                temporary = Path(temporary_text)
                venv = temporary / "venv"
                created = subprocess.run([str(args.target_python), "-m", "venv", str(venv)], check=False, capture_output=True, text=True, timeout=180)
                if created.returncode:
                    raise BridgeError("legacy_builder_toolchain_lock_failed", created.stderr)
                python = venv / "bin/python"
                environment = builder_environment(venv, temporary / "tmp", {key: str(value) for key, value in config["builder_environment"].items()})
                Path(environment["TMPDIR"]).mkdir(mode=0o700)
                requirements = [f"{name}=={tool['version']}" for name, tool in config["builder_tools"].items()]
                install = run([str(python), "-m", "pip", "install", "--no-index", "--no-deps", "--force-reinstall", "--find-links", str(tool_root), *requirements], cwd=output, environment=environment, log=output / f"build_run{run_index}/builder_install.log", timeout=600)
                if install["returncode"]:
                    raise BridgeError("legacy_builder_toolchain_lock_failed", f"builder tool install failed: run {run_index}")
                probe_code = "import importlib.metadata as m,json,sys; print(json.dumps({'python':sys.version,'executable':sys.executable,'versions':{n:m.version(n) for n in ['pip','setuptools','wheel','packaging']}}))"
                probe = subprocess.run([str(python), "-c", probe_code], check=False, capture_output=True, text=True, env=environment, timeout=60)
                fingerprint = json.loads(probe.stdout) if probe.returncode == 0 else {}
                expected_versions = {name: str(tool["version"]) for name, tool in config["builder_tools"].items()}
                if fingerprint.get("versions") != expected_versions or not fingerprint.get("executable", "").startswith(str(venv)):
                    raise BridgeError("legacy_builder_toolchain_lock_failed", f"builder fingerprint mismatch: run {run_index}")
                freeze = subprocess.run([str(python), "-m", "pip", "freeze", "--all"], check=True, capture_output=True, text=True, env=environment, timeout=60).stdout.splitlines()
                builder_record = {"run": run_index, "fingerprint": fingerprint, "pip_freeze": sorted(freeze), "tool_manifest_sha256": tool_manifest_sha, "venv_root": str(venv.resolve())}
                builders.append(builder_record)
                for name, source in sources.items():
                    wheel_dir = output / f"build_run{run_index}" / name / "wheel"
                    wheel_dir.mkdir(parents=True)
                    command = [str(python), "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "--no-cache-dir", "--wheel-dir", str(wheel_dir), str(source["path"])]
                    state["build_invocations"][name] += 1
                    process = run(command, cwd=output, environment=environment, log=output / f"build_run{run_index}/{name}/build.log", timeout=600)
                    if process["returncode"]:
                        raise BridgeError("legacy_sdist_wheel_build_failed", f"wheel build failed: {name} run {run_index}")
                    wheels = list(wheel_dir.glob("*.whl"))
                    if len(wheels) != 1:
                        raise BridgeError("legacy_sdist_wheel_build_failed", f"build did not produce exactly one wheel: {name} run {run_index}")
                    manifest = HELPER.wheel_manifest(wheels[0])
                    build_results[name].append({"run": run_index, "command": command, "environment": environment, "wheel_path": str(wheels[0].resolve()), "wheel": manifest})
        state["stages"]["B1"] = "passed"
        if builders[0]["fingerprint"]["versions"] != builders[1]["fingerprint"]["versions"] or builders[0]["pip_freeze"] != builders[1]["pip_freeze"]:
            raise BridgeError("legacy_builder_toolchain_lock_failed", "builder venv fingerprints differ")

        transformations = []
        import_root = output / "import_smokes/private_targets"
        import_root.mkdir(parents=True, mode=0o700)
        for name, package in config["packages"].items():
            first, second = build_results[name]
            comparison = HELPER.compare_builds(first["wheel"], second["wheel"])
            if not comparison["reproducible"]:
                (output / f"wheel_validation/{name}_nondeterminism.json").parent.mkdir(parents=True, exist_ok=True)
                (output / f"wheel_validation/{name}_nondeterminism.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
                raise BridgeError("legacy_sdist_wheel_nondeterministic", f"wheel builds differ: {name}")
            validated = HELPER.validate_derived_wheel(Path(first["wheel_path"]), name=name, version=str(package["version"]), expected_metadata=audited[name]["resolved_p0a_metadata"])
            final = output / "derived_wheels" / Path(first["wheel_path"]).name
            final.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(first["wheel_path"], final)
            final.chmod(0o444)
            smoke = import_smoke(args.target_python, final, package, import_root)
            build_recipe = {"command": first["command"], "environment": first["environment"], "source_date_epoch": config["source_date_epoch"], "umask": "0022", "build_count": 2}
            transformation = {
                "canonical_name": name,
                "version": str(package["version"]),
                "original_p0a_record_index": sources[name]["record"]["index"],
                "source_url": sources[name]["record"]["url"],
                "source_filename": package["source_filename"],
                "source_size": sources[name]["size"],
                "source_sha256": package["source_sha256"],
                "normalized_source_tree_sha256": audited[name]["tree"]["normalized_tree_sha256"],
                "pkg_info_sha256": audited[name]["pkg_info"]["sha256"],
                "source_patched": False,
                "builder_python": builders[0]["fingerprint"]["python"],
                "builder_tool_versions": builders[0]["fingerprint"]["versions"],
                "builder_tool_artifact_hashes": {key: value["sha256"] for key, value in tool_records.items()},
                "build_environment": first["environment"],
                "build_command": first["command"],
                "build_recipe_sha256": HELPER.deterministic_sha256(build_recipe),
                "run1_wheel_sha256": first["wheel"]["sha256"],
                "run2_wheel_sha256": second["wheel"]["sha256"],
                "reproducible": True,
                "final_wheel_filename": final.name,
                "final_wheel_size": final.stat().st_size,
                "final_wheel_sha256": HELPER.sha256_file(final),
                "final_wheel_uri": final.resolve().as_uri(),
                "wheel_tags": validated["tags"],
                "metadata_sha256": validated["metadata_sha256"],
                "wheel_sha256": validated["wheel_sha256"],
                "record_sha256": validated["record_sha256"],
                "import_smoke": smoke,
            }
            transformation["transformation_sha256"] = HELPER.deterministic_sha256(transformation)
            transformations.append(transformation)
            (output / f"wheel_validation/{name}.json").parent.mkdir(parents=True, exist_ok=True)
            (output / f"wheel_validation/{name}.json").write_text(json.dumps(validated, indent=2, sort_keys=True) + "\n")
        state["stages"]["B2"] = "passed"
        state["stages"]["B3"] = "passed"
        state["stages"]["B4"] = "passed"

        transformation_lock = {"schema_version": "openeta.univtac.source_to_wheel_transformations.v1", "build_method_label": config["build_method_label"], "transformation_count": 3, "records": transformations}
        transformation_lock["transformation_lock_sha256"] = HELPER.deterministic_sha256(transformation_lock)
        transform_path = output / "source_to_wheel_transformations/source_to_wheel_transformations.json"
        transform_path.parent.mkdir(parents=True)
        transform_path.write_text(json.dumps(transformation_lock, indent=2, sort_keys=True) + "\n")
        derived = HELPER.build_derived_lock(source_lock, transformations)
        derived["original_artifact_lock_sha256"] = source_lock["artifact_lock_sha256"]
        derived["transformation_lock_sha256"] = transformation_lock["transformation_lock_sha256"]
        derived["derived_install_artifact_lock_sha256"] = HELPER.deterministic_sha256(derived)
        private = output / "derived_install_artifact_lock/derived_install_artifact_lock.private.json"
        private.parent.mkdir(parents=True)
        descriptor = os.open(private, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(derived, indent=2, sort_keys=True) + "\n")
        public = {key: value for key, value in derived.items() if key != "records"}
        public["records"] = [{key: record.get(key) for key in ("index", "name", "version", "filename", "artifact_type", "origin_host", "sha256", "wheel_tags", "source_transformation_sha256")} for record in derived["records"]]
        (private.parent / "derived_install_artifact_lock.json").write_text(json.dumps(public, indent=2, sort_keys=True) + "\n")
        state["stages"]["L0R"] = "passed"
        state["status"] = "completed"
        state["classification"] = "legacy_sdist_reproducible_wheel_bridge_validated"
        state["builders"] = builders
        state["source_to_wheel_transformations"] = str(transform_path.resolve())
        state["derived_install_artifact_lock"] = str(private.resolve())
    except BridgeError as exc:
        state["status"] = "failed"
        state["classification"] = exc.classification
        state["failure"] = {"class": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        raise
    except BaseException as exc:
        state["status"] = "failed"
        state["classification"] = "legacy_sdist_wheel_build_failed"
        state["failure"] = {"class": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        raise
    finally:
        state["builder_temporary_removed"] = not (output / "builder_temporary").exists() or not any((output / "builder_temporary").iterdir())
        (output / "run_manifest.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        (output / "summary.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
