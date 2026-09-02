from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest
import yaml

from sim.envs.univtac.flatdict_build_contract import (
    SDIST_SHA256,
    audit_report_uses_wheel,
    select_pypi_sdist,
    validate_config,
    validate_flatdict_smoke,
    validate_wheel,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/univtac/flatdict_legacy_sdist_bridge.yaml"


def config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def pypi_payload(sha256: str = SDIST_SHA256):
    return {
        "info": {"name": "flatdict", "version": "4.0.1"},
        "urls": [{
            "filename": "flatdict-4.0.1.tar.gz", "packagetype": "sdist", "size": 8341,
            "url": "https://files.pythonhosted.org/packages/fixed/flatdict-4.0.1.tar.gz",
            "digests": {"sha256": sha256, "blake2b_256": "abc"}, "yanked": False,
        }],
    }


def wheel_digest(data: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")


def make_wheel(path: Path, *, version: str = "4.0.1", native: bool = False) -> None:
    dist = "flatdict-4.0.1.dist-info"
    members = {
        "flatdict.py": b"class FlatDict: pass\n",
        f"{dist}/METADATA": f"Metadata-Version: 2.1\nName: flatdict\nVersion: {version}\n\n".encode(),
        f"{dist}/WHEEL": b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n\n",
    }
    if native:
        members["bad.so"] = b"native"
    record = io.StringIO()
    writer = csv.writer(record, lineterminator="\n")
    for name, data in members.items():
        writer.writerow([name, f"sha256={wheel_digest(data)}", str(len(data))])
    writer.writerow([f"{dist}/RECORD", "", ""])
    members[f"{dist}/RECORD"] = record.getvalue().encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in members.items():
            archive.writestr(name, data)


def test_bridge_config_and_fixed_sdist_identity() -> None:
    validate_config(config())
    record = select_pypi_sdist(pypi_payload(), config())
    assert record["filename"] == "flatdict-4.0.1.tar.gz"
    assert record["digests"]["sha256"] == SDIST_SHA256


def test_hash_or_host_drift_is_rejected() -> None:
    with pytest.raises(ValueError, match="SHA256"):
        select_pypi_sdist(pypi_payload("bad"), config())
    payload = pypi_payload()
    payload["urls"][0]["url"] = "https://example.com/flatdict-4.0.1.tar.gz"
    with pytest.raises(ValueError, match="official PyPI"):
        select_pypi_sdist(payload, config())


def test_wheel_metadata_and_record_are_verified(tmp_path: Path) -> None:
    wheel = tmp_path / "flatdict-4.0.1-py3-none-any.whl"
    make_wheel(wheel)
    result = validate_wheel(wheel)
    assert result["record_complete"] is True
    assert result["contains_native_binary"] is False


def test_wrong_wheel_version_or_native_bundle_is_rejected(tmp_path: Path) -> None:
    wheel = tmp_path / "flatdict-4.0.1-py3-none-any.whl"
    make_wheel(wheel, version="4.1.0")
    with pytest.raises(ValueError, match="METADATA"):
        validate_wheel(wheel)
    make_wheel(wheel, native=True)
    with pytest.raises(ValueError, match="native binary|unexpected bundled"):
        validate_wheel(wheel)


def test_report_must_use_exact_local_wheel(tmp_path: Path) -> None:
    wheel = (tmp_path / "flatdict-4.0.1-py3-none-any.whl").resolve()
    report = {"install": [{"metadata": {"name": "flatdict", "version": "4.0.1"}, "download_info": {"url": wheel.as_uri()}}]}
    assert audit_report_uses_wheel(report, wheel)["success"] is True
    report["install"][0]["metadata"]["version"] = "4.1.0"
    with pytest.raises(ValueError, match="unexpected"):
        audit_report_uses_wheel(report, wheel)


def test_flatdict_smoke_witness_is_exact() -> None:
    validate_flatdict_smoke({"version": "4.0.1", "flattened": {"tactile.left": 1, "tactile.right": 2}})
    with pytest.raises(ValueError, match="witness"):
        validate_flatdict_smoke({"version": "4.1.0", "flattened": {}})
