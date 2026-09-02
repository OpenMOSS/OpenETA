"""Pure helpers for the locked, resumable R0.9.5 Isaac wheelhouse."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import re
import stat
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from urllib.parse import unquote, urlparse

from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename


ALLOWED_REMOTE_HOSTS = {"files.pythonhosted.org", "pypi.nvidia.com"}
AUTHORIZED_TRANSFORMS = {"idna-ssl", "pyperclip", "antlr4-python3-runtime"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def validate_lock_pair(source: Mapping[str, Any], derived: Mapping[str, Any]) -> None:
    if source.get("schema_version") != "openeta.univtac.isaac_artifact_lock.v1" or source.get("record_count") != 173:
        raise ValueError("source artifact lock identity changed")
    source_copy = dict(source)
    source_digest = source_copy.pop("artifact_lock_sha256", None)
    if source_digest != deterministic_sha256(source_copy):
        raise ValueError("source artifact lock self-hash mismatch")
    if derived.get("schema_version") != "openeta.univtac.derived_install_artifact_lock.v1":
        raise ValueError("derived artifact lock schema changed")
    expected_counts = {
        "record_count": 173, "source_record_count": 173, "source_sdist_count": 3,
        "source_wheel_count": 170, "final_install_record_count": 173,
        "final_install_sdist_count": 0, "final_install_wheel_count": 173,
        "transformation_count": 3,
    }
    if any(derived.get(key) != value for key, value in expected_counts.items()):
        raise ValueError("derived artifact closure counts changed")
    derived_copy = dict(derived)
    derived_digest = derived_copy.pop("derived_install_artifact_lock_sha256", None)
    if derived_digest != deterministic_sha256(derived_copy):
        raise ValueError("derived artifact lock self-hash mismatch")
    if derived.get("original_artifact_lock_sha256") != source_digest:
        raise ValueError("derived lock parent identity changed")
    source_record_list = source.get("records", [])
    derived_record_list = derived.get("records", [])
    source_records = {int(record["index"]): record for record in source_record_list}
    derived_records = {int(record["index"]): record for record in derived_record_list}
    if (
        len(source_record_list) != 173
        or len(derived_record_list) != 173
        or len(source_records) != 173
        or len(derived_records) != 173
        or set(source_records) != set(derived_records)
    ):
        raise ValueError("source and derived record indices differ")
    converted = set()
    for index, original in source_records.items():
        final = derived_records[index]
        if (original["name"], original["version"]) != (final["name"], final["version"]):
            raise ValueError("derived package name/version changed")
        if original["artifact_type"] == "sdist":
            converted.add(original["name"])
            if final["artifact_type"] != "wheel" or final.get("url_scheme") != "file" or not final.get("source_transformation_sha256"):
                raise ValueError("authorized source was not converted to a local wheel")
        elif original != final:
            raise ValueError("non-transformed source record changed")
    if converted != AUTHORIZED_TRANSFORMS:
        raise ValueError("derived lock transformed an unauthorized source set")


def parse_headers(text: str) -> dict[str, Any]:
    blocks = [block for block in re.split(r"\r?\n\r?\n", text.strip()) if block.startswith("HTTP/")]
    if not blocks:
        raise ValueError("curl response headers contain no HTTP status")
    block = blocks[-1]
    lines = block.splitlines()
    status = int(lines[0].split()[1])
    headers: dict[str, str] = {}
    for line in lines[1:]:
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip().lower()] = value.strip()
    content_range = headers.get("content-range")
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", content_range or "")
    return {
        "http_status": status,
        "headers": headers,
        "accept_ranges": headers.get("accept-ranges"),
        "etag": headers.get("etag"),
        "last_modified": headers.get("last-modified"),
        "content_length": int(headers["content-length"]) if headers.get("content-length", "").isdigit() else None,
        "content_range": content_range,
        "range_start": int(match.group(1)) if match else None,
        "range_end": int(match.group(2)) if match else None,
        "content_range_total": int(match.group(3)) if match else None,
    }


def expected_total(metadata: Mapping[str, Any]) -> int:
    total = metadata.get("content_range_total") or metadata.get("content_length")
    if not isinstance(total, int) or total <= 0:
        raise ValueError("remote object size is unavailable")
    return total


def sidecar_matches(sidecar: Mapping[str, Any], record: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool:
    expected = {
        "url": record["url"],
        "filename": record["filename"],
        "sha256": record["sha256"],
        "expected_size": expected_total(metadata),
        "etag": metadata.get("etag"),
    }
    return all(sidecar.get(key) == value for key, value in expected.items())


def curl_download_command(*, url: str, output: Path, headers: Path, offset: int, config: Mapping[str, Any]) -> list[str]:
    command = [
        "curl", "--location", "--fail", "--show-error", "--http1.1",
        "--proto", "=https", "--proto-redir", "=https",
        "--connect-timeout", str(config["connect_timeout_seconds"]),
        "--max-time", str(config["max_time_seconds"]),
        "--speed-time", str(config["speed_time_seconds"]),
        "--speed-limit", str(config["speed_limit_bytes_per_second"]),
        "--dump-header", str(headers), "--output", str(output),
        "--write-out", "%{http_code}\n%{url_effective}\n%{size_download}\n%{time_total}\n",
    ]
    if offset:
        command.extend(["--continue-at", "-"])
    command.append(url)
    return command


def disk_space_gate(*, total_bytes: int, remaining_bytes: int, largest_bytes: int, cache_root: Path, environment_root: Path) -> dict[str, Any]:
    cache_free = os.statvfs(cache_root.parent)
    env_free = os.statvfs(environment_root)
    cache_available = cache_free.f_bavail * cache_free.f_frsize
    env_available = env_free.f_bavail * env_free.f_frsize
    same_filesystem = cache_root.parent.stat().st_dev == environment_root.stat().st_dev
    gib = 1024**3
    if same_filesystem:
        required = max(100 * gib, int(2.5 * total_bytes + 20 * gib))
        passed = cache_available >= required
        requirements = {"shared_required_bytes": required}
    else:
        cache_required = remaining_bytes + largest_bytes + 10 * gib
        env_required = int(1.5 * total_bytes + 30 * gib)
        passed = cache_available >= cache_required and env_available >= env_required
        requirements = {"cache_required_bytes": cache_required, "environment_required_bytes": env_required}
    return {"passed": passed, "same_filesystem": same_filesystem, "total_planned_wheel_bytes": total_bytes, "total_remaining_download_bytes": remaining_bytes, "largest_artifact_bytes": largest_bytes, "cache_available_bytes": cache_available, "environment_available_bytes": env_available, **requirements}


def validate_wheel(path: Path, record: Mapping[str, Any], expected_size: int) -> dict[str, Any]:
    if path.name != record["filename"] or path.stat().st_size != expected_size or sha256_file(path) != record["sha256"]:
        raise ValueError("wheel file identity mismatch")
    wheel_name, wheel_version, _, filename_tags = parse_wheel_filename(path.name)
    compatible = set(cpython_tags(python_version=(3, 11))) | set(compatible_tags(python_version=(3, 11), interpreter="cp311"))
    if canonicalize_name(str(wheel_name)) != record["name"] or str(wheel_version) != record["version"] or not compatible.intersection(filename_tags):
        raise ValueError("wheel filename metadata or tag mismatch")
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("wheel ZIP CRC failed")
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("wheel contains duplicate members")
        for name in names:
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts or "\\" in name:
                raise ValueError("wheel contains unsafe path")
        metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
        wheel_names = [name for name in names if name.endswith(".dist-info/WHEEL")]
        record_names = [name for name in names if name.endswith(".dist-info/RECORD")]
        if not (len(metadata_names) == len(wheel_names) == len(record_names) == 1):
            raise ValueError("wheel dist-info files are not unique")
        metadata = BytesParser().parsebytes(archive.read(metadata_names[0]))
        if canonicalize_name(str(metadata.get("Name", ""))) != record["name"] or str(metadata.get("Version", "")) != record["version"]:
            raise ValueError("wheel METADATA identity mismatch")
        wheel_message = BytesParser().parsebytes(archive.read(wheel_names[0]))
        metadata_tags = set()
        for raw in wheel_message.get_all("Tag") or []:
            from packaging.tags import parse_tag
            metadata_tags.update(parse_tag(raw))
        if not metadata_tags or not compatible.intersection(metadata_tags):
            raise ValueError("wheel WHEEL tags are incompatible")
        rows = list(csv.reader(io.StringIO(archive.read(record_names[0]).decode("utf-8"))))
        if len(rows) != len(names) or len({row[0] for row in rows}) != len(rows):
            raise ValueError("wheel RECORD coverage mismatch")
        row_map = {row[0]: row[1:] for row in rows}
        for name in names:
            if name == record_names[0]:
                if row_map.get(name) != ["", ""]:
                    raise ValueError("wheel RECORD self entry mismatch")
                continue
            data = archive.read(name)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
            if row_map.get(name) != [f"sha256={digest}", str(len(data))]:
                raise ValueError(f"wheel RECORD mismatch: {name}")
    return {"filename": path.name, "size": expected_size, "sha256": record["sha256"], "name": record["name"], "version": record["version"], "filename_tags": sorted(map(str, filename_tags)), "metadata_tags": sorted(map(str, metadata_tags)), "record_complete": True, "zip_valid": True}


def closed_wheelhouse(records: Sequence[Mapping[str, Any]], artifacts: Path, sizes: Mapping[str, int]) -> dict[str, Any]:
    expected = {record["filename"]: record for record in records}
    actual = {path.name: path for path in artifacts.iterdir() if path.is_file()}
    extra = sorted(set(actual) - set(expected))
    missing = sorted(set(expected) - set(actual))
    mismatches = []
    manifest = []
    for filename in sorted(set(expected) & set(actual)):
        path = actual[filename]
        record = expected[filename]
        digest = sha256_file(path)
        mode = stat.S_IMODE(path.stat().st_mode)
        if path.stat().st_size != sizes[filename] or digest != record["sha256"] or mode != 0o444:
            mismatches.append(filename)
        manifest.append({"filename": filename, "size": path.stat().st_size, "sha256": digest, "mode": f"{mode:04o}"})
    parts = sorted(path.name for path in artifacts.parent.joinpath("partial").glob("*") if path.is_file()) if artifacts.parent.joinpath("partial").exists() else []
    success = not extra and not missing and not mismatches and not parts and len(expected) == len(records) == 173 and all(name.endswith(".whl") for name in expected)
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    return {"success": success, "record_count": len(records), "artifact_count": len(actual), "extra": extra, "missing": missing, "mismatches": mismatches, "partial_files": parts, "manifest": manifest, "wheelhouse_manifest_sha256": hashlib.sha256(encoded).hexdigest()}


def normalized_install_plan(report: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for item in report.get("install", []):
        metadata = item.get("metadata", {})
        name = canonicalize_name(str(metadata.get("name", "")))
        url = str(item.get("download_info", {}).get("url", ""))
        result[name] = {
            "version": str(metadata.get("version", "")),
            "filename": Path(unquote(urlparse(url).path)).name,
            "sha256": item.get("download_info", {}).get("archive_info", {}).get("hashes", {}).get("sha256"),
            "requested": bool(item.get("requested")),
            "requested_extras": ["all", "isaacsim"] if name == "isaaclab" and item.get("requested") else [],
            "url_scheme": urlparse(url).scheme,
        }
    return result
