from sim.envs.univtac.exact_dependency_transition import audit_exact_transition, classify_install_failure


def test_exact_transition_rejects_unplanned_distribution_change() -> None:
    before = {"A_pkg": "1", "B": "2"}
    after = {"a-pkg": "3", "B": "2"}
    assert audit_exact_transition(before, after, expected={"a.pkg": ("1", "3")})["success"] is True
    after["B"] = "4"
    result = audit_exact_transition(before, after, expected={"a.pkg": ("1", "3")})
    assert result["success"] is False
    assert [item["name"] for item in result["changes"]] == ["a-pkg", "b"]


def test_incomplete_tls_download_is_not_a_dependency_conflict() -> None:
    log = "WARNING: Connection interrupted while downloading\nSSLEOFError\nerror: incomplete-download"
    assert classify_install_failure(log) == "blocked_by_external_resources"
    assert classify_install_failure("ERROR: resolution impossible") == "isaac51_package_install_failed"
