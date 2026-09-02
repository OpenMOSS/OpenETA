from sim.envs.univtac.curobo_runtime_validation import (
    apply_environment_gate,
    classify_example_failure,
    parse_ik_metrics,
    static_audit_covers_binaries,
    static_audit_process_usable,
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


def test_static_audit_warning_exit_is_usable_but_timeout_is_not() -> None:
    process = {"returncode": 1, "timed_out": False, "cleanup_complete": True}
    assert static_audit_process_usable(process, {"records": [{"path": "extension.so"}]})
    process["timed_out"] = True
    assert not static_audit_process_usable(process, {"records": [{"path": "extension.so"}]})


def test_static_audit_must_cover_exact_binary_set(tmp_path) -> None:
    binaries = [tmp_path / "a.so", tmp_path / "b.so"]
    payload = {"records": [{"path": str(path)} for path in binaries]}
    assert static_audit_covers_binaries(payload, binaries)
    payload["records"].pop()
    assert not static_audit_covers_binaries(payload, binaries)
