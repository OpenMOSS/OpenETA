from __future__ import annotations

import inspect

from scripts.univtac import resume_isaac51_blackwell_install as runner


def test_isaac_install_uses_local_flatdict_without_global_build_override() -> None:
    source = inspect.getsource(runner.main)
    assert '"--find-links", str(wheelhouse), "--only-binary=flatdict"' in source
    assert 'config["install"]["isaaclab_requirement"]' in source
    assert "--upgrade" not in source
    assert "--no-build-isolation" not in source


def test_resume_requires_prior_marker_e0_and_no_actual_install() -> None:
    source = inspect.getsource(runner.main)
    for token in (
        'validate_provenance_marker(marker',
        '"prior_e0_passed"',
        '"no_prior_actual_pip_install"',
        '"target_matches_prior_p0_semantics"',
        '"five_curobo_extensions_match_r084"',
        '"r08_r09_libuipc_identical"',
    ):
        assert token in source


def test_r08_is_fingerprinted_before_and_after() -> None:
    source = inspect.getsource(runner.main)
    assert source.index("source_before = fingerprint_environment") < source.index("source_after = fingerprint_environment")
    assert 'manifest["r08_unchanged"]' in source


def test_outputs_and_unrun_stages_are_explicit() -> None:
    source = inspect.getsource(runner.main)
    assert 'raise FileExistsError(f"R0.9.1 output must be fresh' in source
    assert '{stage: "not_run_due_to_gate" for stage in STAGES}' in source
    assert 'write_json(output / "summary.json", manifest)' in source
    assert 'write_json(output / "restore_ready.json", restore)' in source
    assert 'manifest["stages"][stage] = "failed"' in source
