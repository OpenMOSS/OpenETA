from __future__ import annotations

import inspect

from scripts.univtac import resume_blackwell_native_bridge as runner
from sim.envs.univtac.vcpkg_source_recovery import TINYGLTF_COMMIT, VCPKG_COMMIT


def test_source_recovery_pins_are_immutable() -> None:
    assert VCPKG_COMMIT == "dd3097e305afa53f7b4312371f62058d2e665320"
    assert TINYGLTF_COMMIT == "26422192e2908a562b641175dde18489824e609e"


def test_resume_runner_has_no_isaac_or_sysctl_actions() -> None:
    source = inspect.getsource(runner)
    assert "AppLauncher" not in source
    assert "smoke_isaac51" not in source
    assert '"sudo"' not in source
    assert "sysctl -w" not in source


def test_only_one_u0r_build_invocation_exists() -> None:
    source = inspect.getsource(runner)
    assert source.count('name="u0r"') == 1
