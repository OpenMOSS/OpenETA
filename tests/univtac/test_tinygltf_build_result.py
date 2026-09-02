from __future__ import annotations

from scripts.univtac.audit_tinygltf_build_result import NEW_INCLUDE, OLD_INCLUDE


def test_expected_port_replacement_is_narrow() -> None:
    source = b"before\n" + OLD_INCLUDE + b"\nafter\n"
    replaced = source.replace(OLD_INCLUDE, NEW_INCLUDE)
    assert replaced == b"before\n#include <nlohmann/json.hpp>\nafter\n"
    assert replaced.count(NEW_INCLUDE) == 1


def test_build_result_update_is_exposed_by_cli() -> None:
    source = __import__("inspect").getsource(
        __import__(
            "scripts.univtac.audit_tinygltf_build_result", fromlist=["main"]
        ).main
    )
    assert '"--build-result"' in source
    assert '"cuda_compilation_evidence"' in source
