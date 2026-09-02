from __future__ import annotations

import pytest

from sim.envs.univtac.warp_api_compatibility import (
    PUBLIC_APIS,
    classify_warp_api,
    validate_warp_kernel_witness,
)


def test_public_api_classification() -> None:
    payload = {"api_matrix": {name: True for name in PUBLIC_APIS}, "interop_success": True}
    assert classify_warp_api(payload) == "passed"
    payload["api_matrix"]["from_torch"] = False
    assert classify_warp_api(payload) == "warp_public_api_unavailable"


def test_warp_kernel_witness_rejects_private_api_or_nonfinite_contract() -> None:
    payload = {
        "used_private_warp_torch": False,
        "input_witness": list(range(16)),
        "first_output_witness": list(range(1, 17)),
        "second_output_witness": list(range(2, 18)),
        "pointer_identity": {"input": True, "output": True},
    }
    validate_warp_kernel_witness(payload)
    payload["used_private_warp_torch"] = True
    with pytest.raises(ValueError, match="private"):
        validate_warp_kernel_witness(payload)
