from __future__ import annotations

from pathlib import Path

import pytest

from sim.envs.univtac.cuda_include_bridge import bridge_environment, validate_include_resolution


def test_bridge_uses_only_target_cuda_inc_path(tmp_path: Path) -> None:
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin/nvcc").write_text("", encoding="utf-8")
    (tmp_path / "targets/x86_64-linux/include").mkdir(parents=True)
    env = bridge_environment({"CPATH": "bad", "CPLUS_INCLUDE_PATH": "bad"}, tmp_path)
    assert env["CUDA_HOME"] == str(tmp_path.resolve())
    assert env["CUDA_INC_PATH"] == str((tmp_path / "targets/x86_64-linux/include").resolve())
    assert "CPATH" not in env and "CPLUS_INCLUDE_PATH" not in env
    assert "/usr/local/cuda" not in repr(env)


def test_include_resolution_requires_before_after_difference(tmp_path: Path) -> None:
    target = str((tmp_path / "targets/x86_64-linux/include").resolve())
    before = {"cuda_home": str(tmp_path.resolve()), "include_paths": []}
    after = {"cuda_home": str(tmp_path.resolve()), "include_paths": [target], "cuda_inc_path": target}
    validate_include_resolution(before, after, tmp_path)
    with pytest.raises(ValueError):
        validate_include_resolution(after, after, tmp_path)
