"""Contracts for reproducibly converting three locked legacy sdists to wheels."""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import os
import re
import stat
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

from packaging.requirements import Requirement
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename


SCHEMA_VERSION = "openeta.univtac.legacy_sdist_wheel_bridge.v1"
BUILD_METHOD_LABEL = "legacy_sdist_reproducible_wheel_bridge_v1"
EXPECTED_PACKAGES = {
    "idna-ssl": ("1.1.0", "idna-ssl-1.1.0.tar.gz", "a933e3bb13da54383f9e8f35dc4f9cb9eb9b3b78c6b36f311254d6d0d92c6c7c", "idna_ssl"),
    "pyperclip": ("1.8.0", "pyperclip-1.8.0.tar.gz", "b75b975160428d84608c26edba2dec146e7799566aea42c1fe1b32e72b6028f2", "pyperclip"),
    "antlr4-python3-runtime": ("4.9.3", "antlr4-python3-runtime-4.9.3.tar.gz", "f224469b4168294902bb1efa80a8bf7855f24c99aef99cbefc1bcd3cce77881b", "antlr4"),
}
EXPECTED_TOOLS = {
    "pip": ("26.2.1", "pip-26.2.1-py3-none-any.whl", 1816632, "71138adf1f4ca900cdb7d289c21b7494329f2332b6d85f0e1c42108c0384ed3e"),
    "setuptools": ("75.8.2", "setuptools-75.8.2-py3-none-any.whl", 1229385, "558e47c15f1811c1fa7adbd0096669bf76c1d3f433f58324df69f3f5ecac4e8f"),
    "wheel": ("0.42.0", "wheel-0.42.0-py3-none-any.whl", 65375, "177f9c9b0d45c47873b619f5b650346d632cdc35fb5e4d25058e09c9e581433d"),
    "packaging": ("23.0", "packaging-23.0-py3-none-any.whl", 42678, "714ac14496c3e68c99c29b00845f7a2b85f3bb6f1078fd9f72fd20f0570002b2"),
}
NATIVE_SUFFIXES = (".c", ".cc", ".cpp", ".cxx", ".cu", ".rs", ".so", ".pyd", ".dll", ".dylib")
NATIVE_TOKENS = ("ext_modules", "Extension(", "Cython", "maturin", "meson", "cmake")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_sha256(payload: Any) -> str:
    return sha256_bytes(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != SCHEMA_VERSION or config.get("build_method_label") != BUILD_METHOD_LABEL:
        raise ValueError("legacy sdist bridge schema or label changed")
    if config.get("environment") != "UniVTAC-isaac51-sm120-r09" or config.get("build_count") != 2:
        raise ValueError("legacy builder target or build count changed")
    packages = config.get("packages", {})
    if set(packages) != set(EXPECTED_PACKAGES):
        raise ValueError("only the three authorized legacy sdists may be transformed")
    for name, expected in EXPECTED_PACKAGES.items():
        record = packages[name]
        actual = (str(record["version"]), record["source_filename"], record["source_sha256"], record["import_name"])
        if record.get("canonical_name") != name or actual != expected:
            raise ValueError(f"fixed source identity changed: {name}")
    if set(config.get("builder_tools", {})) != set(EXPECTED_TOOLS):
        raise ValueError("builder tool set changed")
    for name, expected in EXPECTED_TOOLS.items():
        record = config["builder_tools"][name]
        actual = (str(record["version"]), record["filename"], int(record["size"]), record["sha256"])
        if actual != expected:
            raise ValueError(f"builder tool identity changed: {name}")
    if config.get("allowed_host") != "files.pythonhosted.org" or config.get("source_date_epoch") != 315532800:
        raise ValueError("builder source host or epoch changed")


def _safe_parts(value: str) -> tuple[str, ...]:
    if "\x00" in value or "\\" in value:
        raise ValueError(f"unsafe archive path: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"unsafe archive path: {value!r}")
    return path.parts


def audit_tar_archive(path: Path, *, maximum_bytes: int, maximum_members: int) -> dict[str, Any]:
    records = []
    top_levels = set()
    total = 0
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        if len(members) > maximum_members:
            raise ValueError("archive member count exceeds limit")
        names = {member.name for member in members}
        for member in members:
            parts = _safe_parts(member.name)
            top_levels.add(parts[0])
            total += max(member.size, 0)
            if member.isdev() or member.isfifo():
                raise ValueError(f"archive contains special node: {member.name}")
            if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                raise ValueError(f"archive contains unsupported member type: {member.name}")
            if member.issym():
                target_parts = _safe_parts(str(PurePosixPath(member.name).parent / member.linkname))
                if target_parts[0] != parts[0]:
                    raise ValueError(f"symlink escapes source root: {member.name}")
            if member.islnk():
                target_parts = _safe_parts(member.linkname)
                if target_parts[0] != parts[0] or member.linkname not in names:
                    raise ValueError(f"hardlink escapes source root: {member.name}")
            records.append({"path": member.name, "type": member.type.decode("ascii", errors="replace"), "mode": f"{member.mode:04o}", "size": member.size, "linkname": member.linkname or None})
    if len(top_levels) != 1 or total > maximum_bytes:
        raise ValueError("archive root or expanded size is outside the allowed scope")
    return {"member_count": len(records), "total_file_bytes": total, "top_level_root": next(iter(top_levels)), "members": records, "success": True}


def extract_validated_archive(path: Path, destination: Path, audit: Mapping[str, Any]) -> Path:
    if not audit.get("success") or destination.exists():
        raise ValueError("archive must be validated and extraction root fresh")
    destination.mkdir(parents=True, mode=0o700)
    destination.chmod(0o700)
    with tarfile.open(path, "r:gz") as archive:
        archive.extractall(destination)
    root = destination / str(audit["top_level_root"])
    if not root.is_dir():
        raise ValueError("validated source root was not extracted")
    return root


def source_tree_manifest(root: Path) -> dict[str, Any]:
    records = []
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = path.relative_to(root).as_posix()
        info = path.lstat()
        if path.is_symlink():
            record = {"path": relative, "type": "symlink", "mode": f"{stat.S_IMODE(info.st_mode):04o}", "size": info.st_size, "sha256": None, "symlink_target": os.readlink(path)}
        elif path.is_file():
            record = {"path": relative, "type": "file", "mode": f"{stat.S_IMODE(info.st_mode):04o}", "size": info.st_size, "sha256": sha256_file(path), "symlink_target": None}
        elif path.is_dir():
            record = {"path": relative, "type": "directory", "mode": f"{stat.S_IMODE(info.st_mode):04o}", "size": 0, "sha256": None, "symlink_target": None}
        else:
            raise ValueError(f"unexpected extracted file type: {relative}")
        records.append(record)
    return {"root": str(root.resolve()), "record_count": len(records), "records": records, "normalized_tree_sha256": deterministic_sha256(records)}


def audit_pure_python_scope(root: Path) -> dict[str, Any]:
    native_files = [path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path.name.lower().endswith(NATIVE_SUFFIXES)]
    token_hits = []
    for name in ("setup.py", "setup.cfg", "pyproject.toml", "MANIFEST.in"):
        path = root / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for token in NATIVE_TOKENS:
            if re.search(re.escape(token), text, flags=re.IGNORECASE):
                token_hits.append({"path": name, "token": token})
    return {"success": not native_files and not token_hits, "native_files": sorted(native_files), "build_token_hits": token_hits}


def normalized_metadata(message: Mapping[str, Any]) -> dict[str, Any]:
    def requirements() -> list[str]:
        return sorted(str(Requirement(item)) for item in (message.get_all("Requires-Dist") or []))
    return {
        "name": canonicalize_name(str(message.get("Name", ""))),
        "version": str(message.get("Version", "")),
        "requires_python": message.get("Requires-Python"),
        "requires_dist": requirements(),
        "provides_extra": sorted(message.get_all("Provides-Extra") or []),
    }


def read_source_metadata(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    candidates = sorted(root.glob("PKG-INFO"))
    if len(candidates) != 1:
        raise ValueError("sdist must contain one top-level PKG-INFO")
    data = candidates[0].read_bytes()
    message = BytesParser().parsebytes(data)
    return normalized_metadata(message), {"path": str(candidates[0]), "sha256": sha256_bytes(data)}


def wheel_manifest(path: Path) -> dict[str, Any]:
    entries = []
    with zipfile.ZipFile(path) as archive:
        if archive.testzip() is not None:
            raise ValueError("wheel ZIP CRC failed")
        for info in archive.infolist():
            pure = PurePosixPath(info.filename)
            if pure.is_absolute() or ".." in pure.parts or "\\" in info.filename:
                raise ValueError("wheel contains unsafe path")
            data = archive.read(info.filename)
            entries.append({"path": info.filename, "size": len(data), "sha256": sha256_bytes(data), "mode": f"{(info.external_attr >> 16) & 0o7777:04o}"})
    return {"filename": path.name, "size": path.stat().st_size, "sha256": sha256_file(path), "entries": entries}


def validate_derived_wheel(path: Path, *, name: str, version: str, expected_metadata: Mapping[str, Any]) -> dict[str, Any]:
    wheel_name, wheel_version, _, tags = parse_wheel_filename(path.name)
    compatible = set(cpython_tags(python_version=(3, 11))) | set(compatible_tags(python_version=(3, 11), interpreter="cp311"))
    if canonicalize_name(str(wheel_name)) != name or str(wheel_version) != version or not compatible.intersection(tags):
        raise ValueError("derived wheel filename identity or tag mismatch")
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or any(item.lower().endswith((".so", ".pyd", ".dll", ".dylib")) for item in names):
            raise ValueError("derived wheel has duplicate or native members")
        metadata_names = [item for item in names if item.endswith(".dist-info/METADATA")]
        wheel_names = [item for item in names if item.endswith(".dist-info/WHEEL")]
        record_names = [item for item in names if item.endswith(".dist-info/RECORD")]
        if not (len(metadata_names) == len(wheel_names) == len(record_names) == 1):
            raise ValueError("derived wheel dist-info is incomplete")
        metadata_data = archive.read(metadata_names[0])
        wheel_data = archive.read(wheel_names[0])
        record_data = archive.read(record_names[0])
        metadata = normalized_metadata(BytesParser().parsebytes(metadata_data))
        wheel_message = BytesParser().parsebytes(wheel_data)
        if metadata != expected_metadata:
            raise ValueError("derived wheel metadata differs from locked source metadata")
        if wheel_message.get("Root-Is-Purelib", "").lower() != "true":
            raise ValueError("derived wheel is not purelib")
        rows = list(csv.reader(io.StringIO(record_data.decode("utf-8"))))
        if len(rows) != len(names) or len({row[0] for row in rows}) != len(rows):
            raise ValueError("derived wheel RECORD does not cover each member exactly once")
        row_map = {row[0]: row[1:] for row in rows}
        for member in names:
            if member == record_names[0]:
                if row_map[member] != ["", ""]:
                    raise ValueError("RECORD self entry mismatch")
                continue
            data = archive.read(member)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")
            if row_map.get(member) != [f"sha256={digest}", str(len(data))]:
                raise ValueError(f"derived wheel RECORD mismatch: {member}")
    manifest = wheel_manifest(path)
    return {
        **manifest,
        "name": name,
        "version": version,
        "tags": sorted(map(str, tags)),
        "metadata": metadata,
        "metadata_sha256": sha256_bytes(metadata_data),
        "wheel_sha256": sha256_bytes(wheel_data),
        "record_sha256": sha256_bytes(record_data),
        "purelib": True,
        "record_complete": True,
    }


def compare_builds(first: Mapping[str, Any], second: Mapping[str, Any]) -> dict[str, Any]:
    fields = ("filename", "size", "sha256", "entries")
    differences = [field for field in fields if first.get(field) != second.get(field)]
    return {"reproducible": not differences, "differences": differences}


def build_derived_lock(source_lock: Mapping[str, Any], transformations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_index = {int(item["original_p0a_record_index"]): item for item in transformations}
    if len(by_index) != 3 or {item["canonical_name"] for item in transformations} != set(EXPECTED_PACKAGES):
        raise ValueError("source-to-wheel transformation set must be exactly the authorized three")
    records = []
    for record in source_lock["records"]:
        index = int(record["index"])
        if index not in by_index:
            records.append(dict(record))
            continue
        transformation = by_index[index]
        if record["name"] != transformation["canonical_name"] or record["version"] != transformation["version"]:
            raise ValueError("transformation does not match original source record")
        derived = dict(record)
        derived.update({
            "artifact_type": "wheel",
            "filename": transformation["final_wheel_filename"],
            "sha256": transformation["final_wheel_sha256"],
            "url": transformation["final_wheel_uri"],
            "url_scheme": "file",
            "origin_host": None,
            "wheel_tags": transformation["wheel_tags"],
            "wheel_tag_compatible": True,
            "source_transformation_sha256": transformation["transformation_sha256"],
            "original_source": {"filename": record["filename"], "sha256": record["sha256"], "url": record["url"]},
        })
        records.append(derived)
    if len(records) != 173 or any(item["artifact_type"] != "wheel" for item in records):
        raise ValueError("derived install closure is not exactly 173 wheels")
    return {"schema_version": "openeta.univtac.derived_install_artifact_lock.v1", "record_count": len(records), "records": records, "source_record_count": 173, "source_sdist_count": 3, "source_wheel_count": 170, "final_install_record_count": 173, "final_install_sdist_count": 0, "final_install_wheel_count": 173, "transformation_count": 3}
