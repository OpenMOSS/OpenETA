"""Pure-Python requirement and official-wheel checks for the packaging 23 bridge."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import stat
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

from packaging.markers import default_environment
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name


SCHEMA_VERSION = "openeta.univtac.isaaclab_packaging23_bridge.v1"
COMPATIBILITY_LABEL = "isaaclab_packaging23_compatibility_bridge_v1"
PACKAGE = "packaging"
NATIVE_VERSION = "26.3"
SIMULATOR_VERSION = "23.0"
WHEEL_FILENAME = "packaging-23.0-py3-none-any.whl"
WHEEL_SIZE = 42678
WHEEL_SHA256 = "714ac14496c3e68c99c29b00845f7a2b85f3bb6f1078fd9f72fd20f0570002b2"


def validate_config(config: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": SCHEMA_VERSION,
        "compatibility_label": COMPATIBILITY_LABEL,
        "package": PACKAGE,
        "native_reference_version": NATIVE_VERSION,
        "simulator_integration_version": SIMULATOR_VERSION,
        "baseline_source": "isaaclab_2_3_metadata_and_pinned_univtac_isaac51_install_recipe",
        "pypi_json_url": "https://pypi.org/pypi/packaging/23.0/json",
        "allowed_file_host": "files.pythonhosted.org",
        "filename": WHEEL_FILENAME,
        "packagetype": "bdist_wheel",
        "expected_size": WHEEL_SIZE,
        "expected_sha256": WHEEL_SHA256,
        "expected_tag": "py3-none-any",
        "expected_purelib": True,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"packaging bridge {key} must be {value!r}")


def audit_requirements(
    distributions: Iterable[Mapping[str, Any]], *, target_version: str = SIMULATOR_VERSION,
    current_version: str = NATIVE_VERSION, environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    marker_environment = dict(default_environment())
    marker_environment["extra"] = ""
    if environment:
        marker_environment.update(environment)
    records: list[dict[str, Any]] = []
    for distribution in distributions:
        for raw in distribution.get("requires", []) or []:
            try:
                requirement = Requirement(str(raw))
            except InvalidRequirement:
                continue
            if canonicalize_name(requirement.name) != PACKAGE:
                continue
            marker = str(requirement.marker) if requirement.marker else None
            active = requirement.marker is None or requirement.marker.evaluate(marker_environment)
            target_ok = requirement.specifier.contains(target_version, prereleases=True)
            current_ok = requirement.specifier.contains(current_version, prereleases=True)
            records.append(
                {
                    "distribution": distribution.get("name"),
                    "installed_version": distribution.get("version"),
                    "requirement_string": str(raw),
                    "specifier": str(requirement.specifier),
                    "environment_marker": marker,
                    "marker_active": active,
                    "packaging_23_satisfies": target_ok,
                    "packaging_26_3_satisfies": current_ok,
                }
            )
    records.sort(key=lambda item: (str(item["distribution"]).lower(), item["requirement_string"]))
    blockers = [item for item in records if item["marker_active"] and not item["packaging_23_satisfies"]]
    return {
        "target_version": target_version,
        "current_version": current_version,
        "requirements": records,
        "blockers": blockers,
        "success": not blockers,
    }


def select_official_wheel(payload: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    info = payload.get("info", {})
    if canonicalize_name(str(info.get("name", ""))) != PACKAGE or str(info.get("version")) != SIMULATOR_VERSION:
        raise ValueError("PyPI packaging package/version mismatch")
    matches = [item for item in payload.get("urls", []) if item.get("filename") == WHEEL_FILENAME and item.get("packagetype") == "bdist_wheel"]
    if len(matches) != 1:
        raise ValueError("PyPI JSON must contain exactly one fixed packaging wheel")
    item = matches[0]
    parsed = urlparse(str(item.get("url", "")))
    if parsed.scheme != "https" or parsed.hostname != config["allowed_file_host"]:
        raise ValueError("packaging wheel must come from files.pythonhosted.org")
    if item.get("yanked") is True or int(item.get("size", -1)) != WHEEL_SIZE or item.get("digests", {}).get("sha256") != WHEEL_SHA256:
        raise ValueError("packaging wheel release identity mismatch")
    return {key: item.get(key) for key in ("filename", "url", "packagetype", "size", "digests", "upload_time_iso_8601", "requires_python", "yanked")}


def _record_digest(data: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode("ascii").rstrip("=")


def validate_official_wheel(path: Path) -> dict[str, Any]:
    if path.name != WHEEL_FILENAME or path.stat().st_size != WHEEL_SIZE or hashlib.sha256(path.read_bytes()).hexdigest() != WHEEL_SHA256:
        raise ValueError("packaging wheel local identity mismatch")
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        if len(names) != len(set(names)):
            raise ValueError("packaging wheel contains duplicate members")
        for item in infos:
            pure = PurePosixPath(item.filename)
            if "\\" in item.filename or pure.is_absolute() or ".." in pure.parts or stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError(f"unsafe packaging wheel member: {item.filename}")
            if item.filename.lower().endswith((".so", ".pyd", ".dll", ".dylib")):
                raise ValueError("packaging wheel contains native binary")
            if not (item.filename.startswith("packaging/") or item.filename.startswith("packaging-23.0.dist-info/")):
                raise ValueError(f"unexpected packaging wheel member: {item.filename}")
        dist_info = "packaging-23.0.dist-info"
        required = {f"{dist_info}/METADATA", f"{dist_info}/WHEEL", f"{dist_info}/RECORD"}
        if not required.issubset(names):
            raise ValueError("packaging wheel dist-info incomplete")
        metadata = BytesParser().parsebytes(archive.read(f"{dist_info}/METADATA"))
        wheel = BytesParser().parsebytes(archive.read(f"{dist_info}/WHEEL"))
        if metadata.get("Name") != PACKAGE or metadata.get("Version") != SIMULATOR_VERSION or metadata.get_all("Requires-Dist"):
            raise ValueError("packaging wheel METADATA mismatch")
        if wheel.get("Root-Is-Purelib", "").lower() != "true" or wheel.get_all("Tag") != ["py3-none-any"]:
            raise ValueError("packaging wheel is not py3-none-any purelib")
        rows = list(csv.reader(io.StringIO(archive.read(f"{dist_info}/RECORD").decode("utf-8"))))
        record = {row[0]: row[1:] for row in rows}
        if set(record) != set(names):
            raise ValueError("packaging wheel RECORD paths incomplete")
        tree = []
        for name in names:
            fields = record[name]
            if name == f"{dist_info}/RECORD":
                if fields != ["", ""]:
                    raise ValueError("packaging RECORD self-entry mismatch")
                continue
            data = archive.read(name)
            if fields != [f"sha256={_record_digest(data)}", str(len(data))]:
                raise ValueError(f"packaging RECORD mismatch: {name}")
            tree.append({"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    return {"success": True, "name": PACKAGE, "version": SIMULATOR_VERSION, "filename": path.name, "size": path.stat().st_size, "sha256": WHEEL_SHA256, "purelib": True, "tag": "py3-none-any", "record_complete": True, "tree": sorted(tree, key=lambda item: item["path"])}
