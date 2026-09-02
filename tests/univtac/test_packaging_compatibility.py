from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from sim.envs.univtac.packaging_compatibility import (
    WHEEL_SHA256,
    audit_requirements,
    select_official_wheel,
    validate_config,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/univtac/isaaclab_packaging23_bridge.yaml"


def config():
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_packaging_bridge_is_fixed_to_official_23_wheel() -> None:
    validate_config(config())
    assert config()["simulator_integration_version"] == "23.0"
    assert config()["expected_sha256"] == WHEEL_SHA256
    for forbidden in ("23.1", "23.2", "24.0"):
        changed = copy.deepcopy(config())
        changed["simulator_integration_version"] = forbidden
        with pytest.raises(ValueError, match="simulator_integration_version"):
            validate_config(changed)


def test_active_requirement_incompatible_with_23_blocks() -> None:
    result = audit_requirements([
        {"name": "vcs-versioning", "version": "2.3.2", "requires": ["packaging>=26.2"]},
        {"name": "setuptools-scm", "version": "10.2.2", "requires": ["packaging>=20"]},
    ])
    assert result["success"] is False
    assert [item["distribution"] for item in result["blockers"]] == ["vcs-versioning"]


def test_inactive_extra_requirement_does_not_block() -> None:
    result = audit_requirements([
        {"name": "setuptools", "version": "75.8.2", "requires": ["packaging>=24.2; extra == 'test'"]},
    ])
    assert result["success"] is True
    assert result["requirements"][0]["marker_active"] is False


def test_official_wheel_selection_rejects_wrong_host_or_hash() -> None:
    payload = {"info": {"name": "packaging", "version": "23.0"}, "urls": [{
        "filename": "packaging-23.0-py3-none-any.whl", "packagetype": "bdist_wheel",
        "size": 42678, "digests": {"sha256": WHEEL_SHA256}, "yanked": False,
        "url": "https://files.pythonhosted.org/packages/fixed/packaging-23.0-py3-none-any.whl",
    }]}
    assert select_official_wheel(payload, config())["filename"].startswith("packaging-23.0")
    payload["urls"][0]["url"] = "https://example.com/package.whl"
    with pytest.raises(ValueError, match="files.pythonhosted"):
        select_official_wheel(payload, config())
