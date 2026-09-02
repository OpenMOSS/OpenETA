"""Deterministic source and wheel validation for the flatdict 4.0.1 bridge."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import stat
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from urllib.parse import unquote, urlparse


SCHEMA_VERSION = "openeta.univtac.flatdict_sdist_bridge.v1"
METHOD_LABEL = "flatdict_4_0_1_verified_sdist_wheel_bridge_v1"
PACKAGE = "flatdict"
VERSION = "4.0.1"
SDIST_FILENAME = "flatdict-4.0.1.tar.gz"
SDIST_SHA256 = "cd32f08fd31ed21eb09ebc76f06b6bd12046a24f77beb1fd0281917e47f26742"
NATIVE_SUFFIXES = (".so", ".pyd", ".dll", ".dylib")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_config(config: Mapping[str, Any]) -> None:
    expected = {
        "schema_version": SCHEMA_VERSION,
        "installation_method_label": METHOD_LABEL,
        "package": PACKAGE,
        "version": VERSION,
        "pypi_json_url": "https://pypi.org/pypi/flatdict/4.0.1/json",
        "allowed_file_host": "files.pythonhosted.org",
        "filename": SDIST_FILENAME,
        "packagetype": "sdist",
        "expected_size": 8341,
        "expected_sha256": SDIST_SHA256,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"flatdict bridge {key} must be {value!r}")
    build = config.get("wheel_build", {})
    if build != {
        "no_deps": True,
        "no_build_isolation": True,
        "expected_purelib": True,
        "expected_tag_suffix": "none-any",
    }:
        raise ValueError("flatdict wheel build contract changed")


def select_pypi_sdist(payload: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    info = payload.get("info", {})
    if str(info.get("name", "")).lower() != PACKAGE or str(info.get("version")) != VERSION:
        raise ValueError("PyPI JSON package/version mismatch")
    matches = [
        item for item in payload.get("urls", [])
        if item.get("filename") == SDIST_FILENAME and item.get("packagetype") == "sdist"
    ]
    if len(matches) != 1:
        raise ValueError("PyPI JSON must contain exactly one fixed flatdict sdist")
    item = matches[0]
    parsed = urlparse(str(item.get("url", "")))
    if parsed.scheme != "https" or parsed.hostname != config["allowed_file_host"]:
        raise ValueError("flatdict sdist must come from the fixed official PyPI file host")
    if item.get("yanked") is True:
        raise ValueError("flatdict sdist is yanked")
    if int(item.get("size", -1)) != int(config["expected_size"]):
        raise ValueError("flatdict sdist size mismatch")
    digests = item.get("digests", {})
    if digests.get("sha256") != SDIST_SHA256:
        raise ValueError("flatdict sdist PyPI SHA256 mismatch")
    return {
        "package": PACKAGE,
        "version": VERSION,
        "filename": item["filename"],
        "url": item["url"],
        "packagetype": item["packagetype"],
        "size": item["size"],
        "digests": dict(digests),
        "upload_time_iso_8601": item.get("upload_time_iso_8601"),
        "python_version": item.get("python_version"),
        "requires_python": item.get("requires_python"),
        "yanked": bool(item.get("yanked")),
    }


def validate_sdist(path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    validate_config(config)
    if path.name != SDIST_FILENAME:
        raise ValueError("flatdict sdist filename mismatch")
    size = path.stat().st_size
    digest = sha256_file(path)
    if size != int(config["expected_size"]) or digest != SDIST_SHA256:
        raise ValueError("flatdict sdist hash or size mismatch")
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        for member in members:
            pure = PurePosixPath(member.name)
            if pure.is_absolute() or ".." in pure.parts or not pure.parts or pure.parts[0] != "flatdict-4.0.1":
                raise ValueError(f"unsafe flatdict sdist member: {member.name}")
            if member.issym() or member.islnk() or member.isdev():
                raise ValueError(f"unsupported flatdict sdist link/device: {member.name}")
    return {"filename": path.name, "size": size, "sha256": digest, "member_count": len(names), "members": sorted(names)}


def _record_digest(data: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode("ascii").rstrip("=")


def validate_wheel(path: Path) -> dict[str, Any]:
    if not path.name.startswith("flatdict-4.0.1-") or not path.name.endswith(".whl"):
        raise ValueError("flatdict wheel filename/version mismatch")
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        for info in infos:
            pure = PurePosixPath(info.filename)
            mode = info.external_attr >> 16
            if pure.is_absolute() or ".." in pure.parts:
                raise ValueError(f"unsafe wheel path: {info.filename}")
            if stat.S_ISLNK(mode):
                raise ValueError(f"wheel contains symlink: {info.filename}")
            if info.filename.lower().endswith(NATIVE_SUFFIXES):
                raise ValueError(f"flatdict wheel unexpectedly contains native binary: {info.filename}")
            if not (info.filename == "flatdict.py" or info.filename.startswith("flatdict-4.0.1.dist-info/")):
                raise ValueError(f"flatdict wheel contains unexpected bundled file: {info.filename}")
        dist_info = "flatdict-4.0.1.dist-info"
        required = {f"{dist_info}/METADATA", f"{dist_info}/WHEEL", f"{dist_info}/RECORD"}
        if not required.issubset(names):
            raise ValueError("flatdict wheel is missing required dist-info records")
        metadata = BytesParser().parsebytes(archive.read(f"{dist_info}/METADATA"))
        if metadata.get("Name", "").lower() != PACKAGE or metadata.get("Version") != VERSION:
            raise ValueError("flatdict wheel METADATA name/version mismatch")
        if metadata.get_all("Requires-Dist"):
            raise ValueError("flatdict wheel unexpectedly bundles dependency metadata")
        wheel_metadata = BytesParser().parsebytes(archive.read(f"{dist_info}/WHEEL"))
        tags = wheel_metadata.get_all("Tag") or []
        if wheel_metadata.get("Root-Is-Purelib", "").lower() != "true" or not any(
            tag.endswith("-none-any")
            and any(part.startswith("py3") for part in tag.split("-", 1)[0].split("."))
            for tag in tags
        ):
            raise ValueError("flatdict wheel is not a compatible pure Python 3 wheel")
        record_rows = list(csv.reader(io.StringIO(archive.read(f"{dist_info}/RECORD").decode("utf-8"))))
        recorded = {row[0]: row[1:] for row in record_rows}
        if set(recorded) != set(names):
            raise ValueError("flatdict wheel RECORD paths are incomplete")
        for name in names:
            fields = recorded[name]
            if name == f"{dist_info}/RECORD":
                if fields != ["", ""]:
                    raise ValueError("wheel RECORD self-entry must have empty hash and size")
                continue
            data = archive.read(name)
            if fields != [f"sha256={_record_digest(data)}", str(len(data))]:
                raise ValueError(f"wheel RECORD mismatch: {name}")
    return {
        "filename": path.name, "sha256": sha256_file(path), "size": path.stat().st_size,
        "name": PACKAGE, "version": VERSION, "root_is_purelib": True,
        "tags": tags, "record_complete": True, "contains_native_binary": False,
        "archive_member_count": len(names), "archive_members": sorted(names),
    }


def validate_flatdict_smoke(payload: Mapping[str, Any]) -> None:
    if payload.get("version") != VERSION or payload.get("flattened") != {"tactile.left": 1, "tactile.right": 2}:
        raise ValueError("flatdict isolated wheel smoke witness mismatch")


def audit_report_uses_wheel(report: Mapping[str, Any], wheel: Path) -> dict[str, Any]:
    flatdict = [
        item for item in report.get("install", [])
        if str(item.get("metadata", {}).get("name", "")).lower() == PACKAGE
    ]
    if len(flatdict) != 1:
        raise ValueError("pip report must contain exactly one flatdict install")
    record = flatdict[0]
    if record.get("metadata", {}).get("version") != VERSION:
        raise ValueError("pip report resolved an unexpected flatdict version")
    url = str(record.get("download_info", {}).get("url", ""))
    resolved = Path(unquote(urlparse(url).path)).resolve() if urlparse(url).scheme == "file" else None
    if resolved != wheel.resolve():
        raise ValueError("pip report did not resolve flatdict from the verified local wheel")
    return {"success": True, "flatdict_version": VERSION, "wheel": str(wheel.resolve()), "download_url": url}
