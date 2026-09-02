#!/usr/bin/env python3
"""Fetch, verify, and smoke the two fixed official build-tool wheels."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import os
import subprocess
import tempfile
import urllib.request
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

import yaml
from packaging.requirements import Requirement


def fetch(url: str) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "OpenETA-R0.9.3"}), timeout=60) as response:
        return json.load(response)


def download(url: str, path: Path) -> None:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "OpenETA-R0.9.3"}), timeout=120) as response:
        path.write_bytes(response.read())


def verify_wheel(path: Path, *, distribution: str, version: str, size: int, sha256: str) -> dict:
    if path.stat().st_size != size or hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
        raise ValueError(f"{distribution} wheel identity mismatch")
    dist_info = ("setuptools_scm" if distribution == "setuptools-scm" else distribution) + f"-{version}.dist-info"
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("duplicate wheel member")
        for name in names:
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts or "\\" in name or name.lower().endswith((".so", ".pyd", ".dll", ".dylib")):
                raise ValueError(f"unsafe wheel member: {name}")
        metadata_name, wheel_name, record_name = (f"{dist_info}/METADATA", f"{dist_info}/WHEEL", f"{dist_info}/RECORD")
        if not {metadata_name, wheel_name, record_name}.issubset(names):
            raise ValueError("wheel metadata incomplete")
        metadata = BytesParser().parsebytes(archive.read(metadata_name))
        wheel = BytesParser().parsebytes(archive.read(wheel_name))
        if metadata.get("Name").lower().replace("_", "-") != distribution or metadata.get("Version") != version:
            raise ValueError("wheel name/version mismatch")
        if wheel.get("Root-Is-Purelib", "").lower() != "true" or "py3-none-any" not in (wheel.get_all("Tag") or []):
            raise ValueError("wheel is not py3-none-any purelib")
        rows = {row[0]: row[1:] for row in csv.reader(io.StringIO(archive.read(record_name).decode()))}
        if set(rows) != set(names):
            raise ValueError("wheel RECORD incomplete")
        for name in names:
            if name == record_name:
                if rows[name] != ["", ""]:
                    raise ValueError("RECORD self entry mismatch")
                continue
            data = archive.read(name)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
            if rows[name] != [f"sha256={digest}", str(len(data))]:
                raise ValueError(f"RECORD mismatch: {name}")
        requires = metadata.get_all("Requires-Dist") or []
    return {"name": distribution, "version": version, "filename": path.name, "size": size, "sha256": sha256, "requires_dist": requires, "purelib": True, "tag": "py3-none-any", "record_complete": True}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--target-python", type=Path, required=True)
    parser.add_argument("--curobo", type=Path, required=True)
    parser.add_argument("--univtac", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(root)
    root.mkdir(parents=True)
    records = {}
    wheels = []
    for name in ("setuptools-scm", "packaging"):
        spec = cfg["wheels"][name]
        payload = fetch(spec["pypi_json_url"])
        candidates = [item for item in payload["urls"] if item["filename"] == spec["filename"] and item["packagetype"] == "bdist_wheel"]
        if len(candidates) != 1:
            raise ValueError(f"expected one official {name} wheel")
        item = candidates[0]
        if urlparse(item["url"]).hostname != cfg["allowed_host"] or item.get("yanked") or item["size"] != spec["size"] or item["digests"]["sha256"] != spec["sha256"]:
            raise ValueError(f"official {name} release identity mismatch")
        path = root / "wheelhouse" / spec["filename"]
        path.parent.mkdir(parents=True, exist_ok=True)
        download(item["url"], path)
        validation = verify_wheel(path, distribution=name, version=str(spec["version"]), size=int(spec["size"]), sha256=spec["sha256"])
        records[name] = {"pypi": {key: item.get(key) for key in ("filename", "url", "size", "digests", "upload_time_iso_8601", "requires_python", "yanked")}, "wheel": validation, "path": str(path)}
        wheels.append(path)
    scm_requires = records["setuptools-scm"]["wheel"]["requires_dist"]
    active = [Requirement(item) for item in scm_requires if Requirement(item).marker is None]
    active_contract = {(item.name.lower().replace("_", "-"), str(item.specifier)) for item in active}
    if active_contract != {("packaging", ">=20"), ("setuptools", "")} or any("vcs" in item.lower() for item in scm_requires):
        raise ValueError("setuptools-scm 8.1.0 dependency contract mismatch")
    if records["packaging"]["wheel"]["requires_dist"]:
        raise ValueError("packaging 23.0 must have no dependencies")
    with tempfile.TemporaryDirectory(prefix="univtac-build-tools-") as temporary:
        target = Path(temporary) / "target"
        command = [str(args.target_python), "-m", "pip", "install", "--no-index", "--no-deps", "--target", str(target), *map(str, wheels)]
        install = subprocess.run(command, check=False, capture_output=True, text=True, env={**os.environ, "PYTHONNOUSERSITE": "1"})
        if install.returncode:
            raise RuntimeError(install.stderr)
        code = """
import json, pathlib, sys
from packaging.version import Version
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
import packaging, setuptools_scm
roots = sys.argv[1:]
versions = [setuptools_scm.get_version(root=root) for root in roots]
print(json.dumps({'packaging': str(pathlib.Path(packaging.__file__).resolve()), 'setuptools_scm': str(pathlib.Path(setuptools_scm.__file__).resolve()), 'checks': [str(Version('1.17.0')), str(Requirement('packaging<24')), SpecifierSet('>=20').contains('23.0')], 'versions': versions, 'vcs_imported': 'vcs_versioning' in sys.modules}))
"""
        env = {**os.environ, "PYTHONNOUSERSITE": "1", "PYTHONPATH": str(target)}
        smoke = subprocess.run([str(args.target_python), "-c", code, str(args.curobo), str(args.univtac)], check=False, capture_output=True, text=True, env=env)
        if smoke.returncode:
            raise RuntimeError(smoke.stderr)
        witness = json.loads(smoke.stdout)
        if not witness["checks"][2] or witness["vcs_imported"] or not all(witness["versions"]):
            raise ValueError("isolated build-tool smoke failed")
    payload = {"success": True, "records": records, "isolated_smoke": witness, "temporary_target_removed": True}
    (root / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
