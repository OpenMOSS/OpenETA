#!/usr/bin/env python3
"""Build and validate one flatdict 4.0.1 wheel from its fixed PyPI sdist."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import traceback
import urllib.request
import venv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

HELPER_PATH = REPO_ROOT / "sim/envs/univtac/flatdict_build_contract.py"
HELPER_SPEC = importlib.util.spec_from_file_location("univtac_flatdict_build_contract", HELPER_PATH)
if HELPER_SPEC is None or HELPER_SPEC.loader is None:
    raise ImportError(f"cannot load flatdict build contract: {HELPER_PATH}")
helper = importlib.util.module_from_spec(HELPER_SPEC)
HELPER_SPEC.loader.exec_module(helper)
METHOD_LABEL = helper.METHOD_LABEL
select_pypi_sdist = helper.select_pypi_sdist
validate_config = helper.validate_config
validate_flatdict_smoke = helper.validate_flatdict_smoke
validate_sdist = helper.validate_sdist
validate_wheel = helper.validate_wheel


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fetch_json(url: str) -> dict[str, Any]:
    request = urllib.request.Request(url, headers={"User-Agent": "OpenETA-UniVTAC-R0.9.1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise TypeError("PyPI JSON response must be an object")
    return payload


def download(url: str, target: Path) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "OpenETA-UniVTAC-R0.9.1"})
    with urllib.request.urlopen(request, timeout=120) as response, target.open("wb") as handle:
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)


def run(command: list[str], *, environment: dict[str, str] | None = None) -> dict[str, Any]:
    completed = subprocess.run(
        command, check=False, capture_output=True, text=True,
        env=environment, timeout=600,
    )
    return {"command": command, "returncode": completed.returncode, "stdout": completed.stdout, "stderr": completed.stderr}


def require_ok(name: str, result: dict[str, Any]) -> None:
    if result["returncode"] != 0:
        raise RuntimeError(f"{name} failed: {result['stderr'][-2000:]}")


def f0_probe(python: Path) -> dict[str, Any]:
    code = r'''
import hashlib, importlib, importlib.metadata, json, pathlib, platform, warnings
records = {}
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    for module_name, distribution in (("setuptools", "setuptools"), ("wheel", "wheel"), ("pkg_resources", "setuptools")):
        try:
            module = importlib.import_module(module_name)
            path = pathlib.Path(module.__file__).resolve()
            records[module_name] = {"import_success": True, "module_realpath": str(path), "module_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "distribution": distribution, "distribution_version": importlib.metadata.version(distribution)}
        except BaseException as exc:
            records[module_name] = {"import_success": False, "error_class": type(exc).__name__, "error_message": str(exc)}
    warning_records = [{"category": item.category.__name__, "message": str(item.message)} for item in caught]
print(json.dumps({"python_version": platform.python_version(), "modules": records, "warnings": warning_records}, sort_keys=True))
'''
    result = run([str(python), "-c", code], environment={**os.environ, "PYTHONNOUSERSITE": "1"})
    require_ok("F0 host build probe", result)
    payload = json.loads(result["stdout"])
    modules = payload["modules"]
    payload["success"] = bool(
        payload["python_version"].startswith("3.11.")
        and modules["setuptools"].get("distribution_version") == "75.8.2"
        and modules["wheel"].get("distribution_version") == "0.42.0"
        and all(item.get("import_success") for item in modules.values())
    )
    payload["process"] = result
    return payload


def isolated_smoke(python: Path, wheel: Path) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="univtac-flatdict-wheel-") as temporary:
        env_root = Path(temporary) / "venv"
        builder = venv.EnvBuilder(with_pip=True, clear=False, symlinks=True)
        builder.create(env_root)
        venv_python = env_root / "bin/python"
        environment = {**os.environ, "PYTHONNOUSERSITE": "1", "PIP_NO_INDEX": "1"}
        install = run([str(venv_python), "-m", "pip", "install", "--no-index", "--no-deps", str(wheel)], environment=environment)
        require_ok("F3 temporary wheel install", install)
        code = (
            "import importlib.metadata,json; from flatdict import FlatDict; "
            "value=FlatDict({'tactile': {'left': 1, 'right': 2}}, delimiter='.'); "
            "print(json.dumps({'version': importlib.metadata.version('flatdict'), 'flattened': dict(value)}, sort_keys=True))"
        )
        smoke = run([str(venv_python), "-c", code], environment=environment)
        require_ok("F3 isolated import smoke", smoke)
        payload = json.loads(smoke["stdout"])
        validate_flatdict_smoke(payload)
        return {
            "success": True, "temporary_environment_removed": True,
            "install": install, "smoke": {**smoke, "payload": payload},
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target-python", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(f"flatdict bridge output must be fresh: {root}")
    root.mkdir(parents=True)
    manifest: dict[str, Any] = {
        "schema_version": "openeta.univtac.flatdict_wheel_run.v1",
        "installation_method_label": METHOD_LABEL, "started_at": utc_now(),
        "status": "running", "classification": None, "gates": {},
        "source_patched": False, "global_build_isolation_disabled": False,
    }
    write_json(root / "run_manifest.json", manifest)
    try:
        f0 = f0_probe(args.target_python.resolve())
        write_json(root / "f0_host_build_environment.json", f0)
        if not f0["success"]:
            manifest["classification"] = "flatdict_host_build_environment_invalid"
            raise RuntimeError("F0 flatdict host build environment is invalid")
        manifest["gates"]["F0"] = "passed"

        pypi_payload = fetch_json(config["pypi_json_url"])
        selected = select_pypi_sdist(pypi_payload, config)
        write_json(root / "source/pypi_release.json", selected)
        sdist = root / "source" / config["filename"]
        sdist.parent.mkdir(parents=True, exist_ok=True)
        download(selected["url"], sdist)
        try:
            source_validation = validate_sdist(sdist, config)
        except ValueError:
            manifest["classification"] = "flatdict_sdist_hash_mismatch"
            raise
        write_json(root / "source/validation.json", source_validation)
        manifest["gates"]["F1"] = "passed"

        wheelhouse = root / "wheelhouse"
        wheelhouse.mkdir()
        build = run([
            str(args.target_python.resolve()), "-m", "pip", "wheel", "--no-deps",
            "--no-build-isolation", "--wheel-dir", str(wheelhouse), str(sdist),
        ], environment={**os.environ, "PYTHONNOUSERSITE": "1"})
        write_json(root / "wheel/build_process.json", build)
        if build["returncode"] != 0:
            manifest["classification"] = "flatdict_wheel_build_failed"
            raise RuntimeError("F2 flatdict wheel build failed")
        wheels = list(wheelhouse.glob("*.whl"))
        if len(wheels) != 1:
            manifest["classification"] = "flatdict_wheel_validation_failed"
            raise RuntimeError("F2 must produce exactly one wheel")
        try:
            wheel_validation = validate_wheel(wheels[0])
        except ValueError:
            manifest["classification"] = "flatdict_wheel_validation_failed"
            raise
        write_json(root / "wheel/validation.json", wheel_validation)
        manifest["gates"]["F2"] = "passed"

        smoke = isolated_smoke(args.target_python.resolve(), wheels[0])
        write_json(root / "f3_isolated_smoke.json", smoke)
        manifest["gates"]["F3"] = "passed"
        manifest["status"] = "completed"
        manifest["classification"] = "flatdict_wheel_bridge_validated"
        manifest["wheel"] = str(wheels[0].resolve())
        manifest["wheel_sha256"] = wheel_validation["sha256"]
    except BaseException as exc:
        manifest["status"] = "failed"
        manifest["classification"] = manifest["classification"] or "blocked_by_external_resources"
        manifest["failure"] = {"class": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        write_json(root / "failure.json", manifest["failure"])
        raise
    finally:
        manifest["ended_at"] = utc_now()
        write_json(root / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
