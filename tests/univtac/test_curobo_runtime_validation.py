from sim.envs.univtac.curobo_runtime_validation import (
    apply_environment_gate,
    classify_example_failure,
    parse_ik_metrics,
)


def test_parse_official_ik_metrics() -> None:
    line = "Success, Solve Time(s), hz  1.0 0.0064 1533.7 tensor(4.2e-06, device='cuda:0') tensor(4.1e-06, device='cuda:0')"
    assert parse_ik_metrics(line) == [{"success_ratio": 1.0, "solve_time_seconds": 0.0064, "hz": 1533.7, "position_error": 4.2e-06, "rotation_error": 4.1e-06}]


def test_warp_api_error_is_packaging_failure() -> None:
    assert classify_example_failure("module 'warp' has no attribute 'torch'") == "curobo_official_example_packaging_failure"


def test_environment_change_blocks_viable_result() -> None:
    manifest = {
        "status": "completed",
        "classification": "blackwell_native_bridge_viable",
        "author_contact_recommendation": "not_needed",
    }
    apply_environment_gate(manifest, {"r08": {"unchanged": False}})
    assert manifest["status"] == "failed"
    assert manifest["classification"] == "environment_fingerprint_changed"


def test_environment_gate_preserves_existing_failure_classification() -> None:
    manifest = {"status": "failed", "classification": "curobo_official_example_packaging_failure"}
    apply_environment_gate(manifest, {})
    assert manifest["classification"] == "curobo_official_example_packaging_failure"
