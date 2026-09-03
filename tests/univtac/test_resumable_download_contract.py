from __future__ import annotations

import copy
import base64
import csv
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import pytest

from scripts.univtac import build_reproducible_legacy_wheels as builder
from scripts.univtac import download_isaac_wheelhouse as downloader
from sim.envs.univtac.resumable_download_contract import (
    curl_download_command,
    deterministic_sha256,
    expected_total,
    parse_headers,
    sha256_file,
    sidecar_matches,
    validate_wheel,
    validate_lock_pair,
)


CONFIG = {"connect_timeout_seconds": 60, "max_time_seconds": 3600, "speed_time_seconds": 180, "speed_limit_bytes_per_second": 1024}


def _record_digest(data: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")


def _legal_wheel(tmp_path: Path) -> tuple[bytes, dict]:
    wheel_path = tmp_path / "demo_pkg-1.0-py3-none-any.whl"
    files = {
        "demo_pkg/__init__.py": b"",
        "demo_pkg-1.0.dist-info/METADATA": (
            b"Metadata-Version: 2.1\nName: demo-pkg\nVersion: 1.0\n\n"
        ),
        "demo_pkg-1.0.dist-info/WHEEL": (
            b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n\n"
        ),
    }
    record_path = "demo_pkg-1.0.dist-info/RECORD"
    rows = [
        [name, f"sha256={_record_digest(data)}", str(len(data))]
        for name, data in files.items()
    ]
    rows.append([record_path, "", ""])
    output = io.StringIO()
    csv.writer(output, lineterminator="\n").writerows(rows)
    files[record_path] = output.getvalue().encode()
    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    payload = wheel_path.read_bytes()
    record = {
        "index": 0,
        "name": "demo-pkg",
        "version": "1.0",
        "filename": wheel_path.name,
        "url": "https://files.pythonhosted.org/demo_pkg-1.0-py3-none-any.whl",
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    return payload, record


def _official_like_wheel(
    tmp_path: Path,
    *,
    omit_from_primary_record: set[str] | None = None,
    empty_hash_entries: set[str] | None = None,
    primary_record_self: tuple[str, str] = ("", ""),
) -> tuple[Path, dict]:
    filename = "isaacsim_kernel-5.1.0.0-cp311-none-manylinux_2_35_x86_64.whl"
    primary_dir = "isaacsim_kernel-5.1.0.0.dist-info"
    nested_root = "isaacsim/kit/extscore/registry/pip_requests"
    files = {
        "isaacsim_kernel/__init__.py": b"",
        f"{primary_dir}/METADATA": (
            b"Metadata-Version: 2.1\nName: isaacsim-kernel\nVersion: 5.1.0.0\n\n"
        ),
        f"{primary_dir}/WHEEL": (
            b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\n"
            b"Tag: cp311-none-manylinux_2_35_x86_64\n\n"
        ),
        f"{nested_root}/requests-2.0.dist-info/METADATA": b"Name: requests\nVersion: 2.0\n",
        f"{nested_root}/requests-2.0.dist-info/WHEEL": b"vendored requests wheel metadata\n",
        f"{nested_root}/requests-2.0.dist-info/RECORD": b"vendored requests record\n",
        f"{nested_root}/certifi-1.0.dist-info/METADATA": b"Name: certifi\nVersion: 1.0\n",
        f"{nested_root}/certifi-1.0.dist-info/WHEEL": b"vendored certifi wheel metadata\n",
        f"{nested_root}/certifi-1.0.dist-info/RECORD": b"vendored certifi record\n",
    }
    primary_record_path = f"{primary_dir}/RECORD"
    omitted = omit_from_primary_record or set()
    empty = empty_hash_entries or set()
    rows = []
    for name, data in files.items():
        if name in omitted:
            continue
        if name in empty:
            rows.append([name, "", ""])
        else:
            rows.append([name, f"sha256={_record_digest(data)}", str(len(data))])
    rows.append([primary_record_path, *primary_record_self])
    output = io.StringIO()
    csv.writer(output, lineterminator="\n").writerows(rows)
    files[primary_record_path] = output.getvalue().encode()
    wheel_path = tmp_path / filename
    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    record = {
        "index": 9,
        "name": "isaacsim-kernel",
        "version": "5.1.0.0",
        "filename": filename,
        "url": f"https://pypi.nvidia.com/{filename}",
        "sha256": sha256_file(wheel_path),
    }
    return wheel_path, record


def _rewrite_wheel(path: Path, transform) -> None:
    with zipfile.ZipFile(path) as archive:
        members = [(name, archive.read(name)) for name in archive.namelist()]
    rewritten = transform(members)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in rewritten:
            archive.writestr(name, data)


def test_parse_range_identity_and_total() -> None:
    text = "HTTP/1.1 200 Connection established\r\n\r\nHTTP/1.1 206 Partial Content\r\nETag: abc\r\nContent-Range: bytes 0-0/3021300000\r\nContent-Length: 1\r\n\r\n"
    result = parse_headers(text)
    assert result["http_status"] == 206
    assert result["range_start"] == 0
    assert expected_total(result) == 3021300000


def test_resume_command_only_adds_continue_at_for_existing_partial(tmp_path: Path) -> None:
    first = curl_download_command(url="https://files.pythonhosted.org/x.whl", output=tmp_path / "x.part", headers=tmp_path / "h", offset=0, config=CONFIG)
    resumed = curl_download_command(url="https://files.pythonhosted.org/x.whl", output=tmp_path / "x.part", headers=tmp_path / "h", offset=42, config=CONFIG)
    assert "--continue-at" not in first
    assert resumed[resumed.index("--continue-at") + 1] == "-"
    assert "--retry" not in resumed and "--insecure" not in resumed and "--parallel" not in resumed


def test_partial_sidecar_binds_url_hash_size_and_etag() -> None:
    record = {"url": "https://pypi.nvidia.com/x.whl", "filename": "x.whl", "sha256": "abc"}
    metadata = {"content_length": 10, "etag": "tag"}
    sidecar = {"url": record["url"], "filename": "x.whl", "sha256": "abc", "expected_size": 10, "etag": "tag"}
    assert sidecar_matches(sidecar, record, metadata) is True
    sidecar["etag"] = "changed"
    assert sidecar_matches(sidecar, record, metadata) is False


def _lock_pair() -> tuple[dict, dict]:
    transformed = {"idna-ssl", "pyperclip", "antlr4-python3-runtime"}
    records = []
    for index in range(173):
        name = sorted(transformed)[index] if index < 3 else f"package-{index}"
        records.append(
            {
                "index": index,
                "name": name,
                "version": "1.0",
                "artifact_type": "sdist" if name in transformed else "wheel",
            }
        )
    source = {
        "schema_version": "openeta.univtac.isaac_artifact_lock.v1",
        "record_count": 173,
        "records": records,
    }
    source["artifact_lock_sha256"] = deterministic_sha256(source)
    derived_records = copy.deepcopy(records)
    for record in derived_records[:3]:
        record.update(
            artifact_type="wheel",
            url_scheme="file",
            source_transformation_sha256="transform",
        )
    derived = {
        "schema_version": "openeta.univtac.derived_install_artifact_lock.v1",
        "record_count": 173,
        "source_record_count": 173,
        "source_sdist_count": 3,
        "source_wheel_count": 170,
        "final_install_record_count": 173,
        "final_install_sdist_count": 0,
        "final_install_wheel_count": 173,
        "transformation_count": 3,
        "original_artifact_lock_sha256": source["artifact_lock_sha256"],
        "records": derived_records,
    }
    derived["derived_install_artifact_lock_sha256"] = deterministic_sha256(derived)
    return source, derived


def test_lock_pair_requires_exact_unique_indices() -> None:
    source, derived = _lock_pair()
    validate_lock_pair(source, derived)
    duplicate = copy.deepcopy(source)
    duplicate["records"][-1]["index"] = duplicate["records"][-2]["index"]
    duplicate["artifact_lock_sha256"] = deterministic_sha256(
        {key: value for key, value in duplicate.items() if key != "artifact_lock_sha256"}
    )
    duplicate_derived = copy.deepcopy(derived)
    duplicate_derived["original_artifact_lock_sha256"] = duplicate["artifact_lock_sha256"]
    duplicate_derived["derived_install_artifact_lock_sha256"] = deterministic_sha256(
        {
            key: value
            for key, value in duplicate_derived.items()
            if key != "derived_install_artifact_lock_sha256"
        }
    )
    with pytest.raises(ValueError, match="indices differ"):
        validate_lock_pair(duplicate, duplicate_derived)


def test_missing_source_input_is_classified(tmp_path: Path) -> None:
    record = {"filename": "legacy.tar.gz", "sha256": "0" * 64}
    with pytest.raises(downloader.WheelhouseError) as caught:
        downloader.copy_source_input(record, tmp_path / "missing", tmp_path / "target")
    assert caught.value.classification == "wheel_artifact_hash_mismatch"


def test_invalid_existing_cached_wheel_is_classified(tmp_path: Path) -> None:
    cache = tmp_path / "cache"
    (cache / "artifacts").mkdir(parents=True)
    (cache / "partial").mkdir()
    (cache / "quarantined").mkdir()
    final = cache / "artifacts" / "broken-1.0-py3-none-any.whl"
    final.write_bytes(b"not a wheel")
    record = {
        "filename": final.name,
        "name": "broken",
        "version": "1.0",
        "sha256": "0" * 64,
    }
    with pytest.raises(downloader.WheelhouseError) as caught:
        downloader.download_remote(record, {"content_length": final.stat().st_size}, cache, {}, tmp_path / "completed")
    assert caught.value.classification == "wheel_artifact_hash_mismatch"


def test_complete_part_uses_locked_filename_for_identity(tmp_path: Path) -> None:
    payload, record = _legal_wheel(tmp_path)
    part = tmp_path / f"{record['filename']}.part"
    part.write_bytes(payload)
    result = validate_wheel(part, record, len(payload), check_filename=False)
    assert result["name"] == "demo-pkg"
    assert result["record_complete"] is True

    with pytest.raises(ValueError, match="wheel file identity mismatch"):
        validate_wheel(part, record, len(payload), check_filename=True)


def test_complete_part_still_validates_locked_filename_identity(tmp_path: Path) -> None:
    payload, record = _legal_wheel(tmp_path)
    part = tmp_path / f"{record['filename']}.part"
    part.write_bytes(payload)

    wrong_name = {**record, "filename": "other_pkg-1.0-py3-none-any.whl"}
    with pytest.raises(ValueError, match="filename metadata or tag mismatch"):
        validate_wheel(part, wrong_name, len(payload), check_filename=False)

    invalid_name = {**record, "filename": "not-a-wheel.part"}
    with pytest.raises(ValueError):
        validate_wheel(part, invalid_name, len(payload), check_filename=False)


def test_complete_part_promotion_is_offline_and_strictly_revalidated(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload, record = _legal_wheel(tmp_path)
    cache = tmp_path / "cache"
    for name in ("artifacts", "partial", "quarantined"):
        (cache / name).mkdir(parents=True, exist_ok=True)
    part = cache / "partial" / f"{record['filename']}.part"
    part.write_bytes(payload)
    sidecar = {
        "schema_version": "openeta.univtac.wheel_partial.v1",
        "url": record["url"],
        "filename": record["filename"],
        "sha256": record["sha256"],
        "expected_size": len(payload),
        "etag": "locked-etag",
        "attempts": [{"attempt_index": 1}],
    }
    sidecar_path = cache / "partial" / f"{record['filename']}.part.json"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    header = cache / "partial" / f"{record['filename']}.attempt-1.headers"
    header.write_text("preserved historical header", encoding="utf-8")
    curl_calls = []

    def unexpected_curl(*_args, **_kwargs):
        curl_calls.append(True)
        raise AssertionError("complete part must not access the network")

    monkeypatch.setattr(downloader, "run_curl", unexpected_curl)
    result, attempts = downloader.download_remote(
        record,
        {"content_length": len(payload), "etag": "locked-etag"},
        cache,
        {"max_transport_attempts_per_artifact": 20},
        tmp_path / "completed.jsonl",
    )
    final = cache / "artifacts" / record["filename"]
    assert curl_calls == [] and attempts == []
    assert final.is_file() and not part.exists()
    assert not sidecar_path.exists() and not header.exists()
    assert result["disposition"] == "recovered_complete_part"
    assert result["historical_transport_attempts"] == 1
    assert result["new_transport_attempts"] == 0
    assert result["cumulative_transport_attempts"] == 1
    assert sha256_file(final) == record["sha256"]
    assert validate_wheel(final, record, len(payload))["zip_valid"] is True
    assert (final.stat().st_mode & 0o777) == 0o444

    renamed = tmp_path / "renamed-1.0-py3-none-any.whl"
    os.link(final, renamed)
    with pytest.raises(ValueError, match="wheel file identity mismatch"):
        validate_wheel(renamed, record, len(payload), check_filename=True)


def test_nested_vendored_dist_info_uses_only_root_primary_metadata(tmp_path: Path) -> None:
    wheel, record = _official_like_wheel(tmp_path)
    result = validate_wheel(wheel, record, wheel.stat().st_size)
    assert result["name"] == "isaacsim-kernel"
    assert result["record_complete"] is True


def test_nested_dist_info_does_not_contribute_to_primary_count(tmp_path: Path) -> None:
    wheel, record = _official_like_wheel(tmp_path)
    with zipfile.ZipFile(wheel) as archive:
        dist_info_dirs = {
            name.split(".dist-info/", 1)[0] + ".dist-info"
            for name in archive.namelist()
            if ".dist-info/" in name
        }
    assert len(dist_info_dirs) == 3
    assert validate_wheel(wheel, record, wheel.stat().st_size)["zip_valid"] is True


def test_official_like_complete_part_accepts_nested_dist_info(tmp_path: Path) -> None:
    wheel, record = _official_like_wheel(tmp_path)
    part = tmp_path / f"{record['filename']}.part"
    part.write_bytes(wheel.read_bytes())
    assert validate_wheel(part, record, part.stat().st_size, check_filename=False)[
        "zip_valid"
    ] is True


def test_second_top_level_dist_info_is_rejected(tmp_path: Path) -> None:
    wheel, record = _official_like_wheel(tmp_path)
    _rewrite_wheel(
        wheel,
        lambda members: members + [("other_pkg-1.0.dist-info/METADATA", b"Name: other-pkg\n")],
    )
    record["sha256"] = sha256_file(wheel)
    with pytest.raises(ValueError, match="primary dist-info directory mismatch"):
        validate_wheel(wheel, record, wheel.stat().st_size)


def test_nested_dist_info_cannot_replace_missing_root_primary(tmp_path: Path) -> None:
    wheel, record = _official_like_wheel(tmp_path)
    primary = "isaacsim_kernel-5.1.0.0.dist-info/"
    _rewrite_wheel(wheel, lambda members: [item for item in members if not item[0].startswith(primary)])
    record["sha256"] = sha256_file(wheel)
    with pytest.raises(ValueError, match="primary dist-info directory mismatch"):
        validate_wheel(wheel, record, wheel.stat().st_size)


def test_wrong_root_primary_dir_fails_even_when_metadata_matches(tmp_path: Path) -> None:
    wheel, record = _official_like_wheel(tmp_path)
    primary = "isaacsim_kernel-5.1.0.0.dist-info/"
    _rewrite_wheel(
        wheel,
        lambda members: [
            (name.replace(primary, "wrong_name-5.1.0.0.dist-info/", 1), data)
            if name.startswith(primary)
            else (name, data)
            for name, data in members
        ],
    )
    record["sha256"] = sha256_file(wheel)
    with pytest.raises(ValueError, match="primary dist-info directory mismatch"):
        validate_wheel(wheel, record, wheel.stat().st_size)


def test_primary_record_must_cover_nested_member(tmp_path: Path) -> None:
    missing = {
        "isaacsim/kit/extscore/registry/pip_requests/requests-2.0.dist-info/METADATA"
    }
    wheel, record = _official_like_wheel(tmp_path, omit_from_primary_record=missing)
    with pytest.raises(ValueError, match="RECORD coverage mismatch"):
        validate_wheel(wheel, record, wheel.stat().st_size)


def test_nested_record_is_hashed_as_regular_payload(tmp_path: Path) -> None:
    nested_record = {
        "isaacsim/kit/extscore/registry/pip_requests/requests-2.0.dist-info/RECORD"
    }
    wheel, record = _official_like_wheel(tmp_path, empty_hash_entries=nested_record)
    with pytest.raises(ValueError, match="RECORD mismatch"):
        validate_wheel(wheel, record, wheel.stat().st_size)


def test_only_primary_record_self_entry_can_have_empty_hash(tmp_path: Path) -> None:
    wheel, record = _official_like_wheel(
        tmp_path, primary_record_self=("sha256=not-the-record", "1")
    )
    with pytest.raises(ValueError, match="RECORD self entry mismatch"):
        validate_wheel(wheel, record, wheel.stat().st_size)


def test_official_like_complete_part_promotion_uses_no_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    wheel, record = _official_like_wheel(tmp_path)
    payload = wheel.read_bytes()
    cache = tmp_path / "cache"
    for name in ("artifacts", "partial", "quarantined"):
        (cache / name).mkdir(parents=True, exist_ok=True)
    part = cache / "partial" / f"{record['filename']}.part"
    part.write_bytes(payload)
    sidecar = {
        "schema_version": "openeta.univtac.wheel_partial.v1",
        "url": record["url"],
        "filename": record["filename"],
        "sha256": record["sha256"],
        "expected_size": len(payload),
        "etag": "locked-etag",
        "attempts": [{"attempt_index": 1}],
    }
    sidecar_path = cache / "partial" / f"{record['filename']}.part.json"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")
    header = cache / "partial" / f"{record['filename']}.attempt-1.headers"
    header.write_text("historical header", encoding="utf-8")
    curl_calls = []

    def unexpected_curl(*_args, **_kwargs):
        curl_calls.append(True)
        raise AssertionError("complete part must not access the network")

    monkeypatch.setattr(downloader, "run_curl", unexpected_curl)
    result, attempts = downloader.download_remote(
        record,
        {"content_length": len(payload), "etag": "locked-etag"},
        cache,
        {"max_transport_attempts_per_artifact": 20},
        tmp_path / "completed.jsonl",
    )
    final = cache / "artifacts" / record["filename"]
    assert curl_calls == [] and attempts == []
    assert final.is_file() and not part.exists()
    assert not sidecar_path.exists() and not header.exists()
    assert result["disposition"] == "recovered_complete_part"
    assert result["historical_transport_attempts"] == 1
    assert result["new_transport_attempts"] == 0
    assert result["cumulative_transport_attempts"] == 1
    assert validate_wheel(final, record, len(payload))["zip_valid"] is True
    assert (final.stat().st_mode & 0o777) == 0o444


class _RedirectedJsonResponse(io.BytesIO):
    headers = {}

    def __enter__(self) -> "_RedirectedJsonResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def geturl(self) -> str:
        return "http://example.invalid/project.json"


def test_metadata_fetch_rejects_unapproved_redirect(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(builder.urllib.request, "urlopen", lambda *_args, **_kwargs: _RedirectedJsonResponse(json.dumps({}).encode()))
    with pytest.raises(builder.BridgeError, match="redirected to unapproved"):
        builder.fetch_json("https://pypi.org/pypi/example/1/json")
