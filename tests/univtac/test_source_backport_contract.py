from __future__ import annotations

import pytest

from sim.envs.univtac.source_backport_contract import (
    NEW_EXPRESSION,
    OLD_EXPRESSION,
    compiler_wrapper_paths,
    corrective_retry_accounting,
    make_backported_source,
    validate_compiler_wrappers,
    validate_semantic_backport,
)


def test_only_authorized_attribute_chain_changes() -> None:
    before = f"import warp as wp\n\ndef f(self):\n    return {OLD_EXPRESSION}\n"
    after = make_backported_source(before)
    assert NEW_EXPRESSION in after
    audit = validate_semantic_backport(before, after)
    assert audit["control_flow_unchanged"] is True
    assert audit["call_arguments_unchanged"] is True


def test_control_flow_or_constant_change_is_rejected() -> None:
    before = f"import warp as wp\n\ndef f(self):\n    return {OLD_EXPRESSION}\n"
    after = make_backported_source(before).replace("return ", "if True:\n        return ")
    with pytest.raises(ValueError, match="more than"):
        validate_semantic_backport(before, after)


def test_compiler_wrappers_are_derived_from_pinned_source(tmp_path) -> None:
    paths = compiler_wrapper_paths(tmp_path)
    assert paths["CC"] == tmp_path / "scripts/toolchains/gcc12-system-ld"
    assert paths["CXX"] == tmp_path / "scripts/toolchains/gxx12-system-ld"
    with pytest.raises(ValueError, match="not an executable"):
        validate_compiler_wrappers(tmp_path)


def test_authorized_retry_accounting_is_bounded() -> None:
    accounting = corrective_retry_accounting()
    assert accounting["total_c0_invocations"] == 2
    assert accounting["invalid_orchestration_invocations"] == 1
    assert accounting["valid_curobo_build_attempts"] == 1
    assert accounting["authorized_corrective_retries"] == 1
    assert accounting["additional_retry_allowed"] is False
    assert accounting["c0_invocations"][1]["correction_scope"] == ["CC", "CXX", "CUDAHOSTCXX"]
    assert accounting["c0_invocations"][1]["runtime_scope_expanded"] is False
