"""Contract for the fixed R0.9 build-tool graph transition."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from packaging.markers import default_environment
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import Version


SCHEMA_VERSION = "openeta.univtac.isaaclab_build_tool_bridge.v1"
APPLIED_LABEL = "isaaclab_setuptools_scm8_packaging23_bridge_v1"
EXPECTED_ORDER = (
    "install_setuptools_scm_8_1_0",
    "uninstall_vcs_versioning_2_3_2",
    "install_packaging_23_0",
)


def validate_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != SCHEMA_VERSION or config.get("applied_label") != APPLIED_LABEL:
        raise ValueError("build-tool bridge schema or label mismatch")
    if config.get("environment") != "UniVTAC-isaac51-sm120-r09":
        raise ValueError("build-tool bridge may only target R0.9")
    if config.get("before") != {"setuptools-scm": "10.2.2", "vcs-versioning": "2.3.2", "packaging": "26.3"}:
        raise ValueError("build-tool before graph mismatch")
    if config.get("after") != {"setuptools": "75.8.2", "setuptools-scm": "8.1.0", "vcs-versioning": None, "packaging": "23.0", "wheel": "0.42.0", "filelock": "3.32.3"}:
        raise ValueError("build-tool after graph mismatch")
    if tuple(config.get("mutation_order", ())) != EXPECTED_ORDER:
        raise ValueError("build-tool mutation order changed")
    expected = {
        "setuptools-scm": ("8.1.0", "setuptools_scm-8.1.0-py3-none-any.whl", 43666, "897a3226a6fd4a6eb2f068745e49733261a21f70b1bb28fce0339feb978d9af3"),
        "packaging": ("23.0", "packaging-23.0-py3-none-any.whl", 42678, "714ac14496c3e68c99c29b00845f7a2b85f3bb6f1078fd9f72fd20f0570002b2"),
    }
    for name, values in expected.items():
        item = config["wheels"][name]
        if (str(item["version"]), item["filename"], int(item["size"]), item["sha256"]) != values:
            raise ValueError(f"fixed {name} wheel identity changed")


def dependency_graph(distributions: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    env = default_environment()
    env["extra"] = ""
    nodes = {}
    reverse: dict[str, list[dict[str, str]]] = {}
    for item in distributions:
        name = canonicalize_name(str(item["name"]))
        active = []
        for raw in item.get("requires", []) or []:
            req = Requirement(str(raw))
            if req.marker is None or req.marker.evaluate(env):
                active.append(str(req))
                reverse.setdefault(canonicalize_name(req.name), []).append({"distribution": name, "requirement": str(req)})
        nodes[name] = {"version": str(item["version"]), "requires_dist": list(item.get("requires", []) or []), "active_requirements": sorted(active)}
    return {"nodes": nodes, "reverse_dependencies": {name: sorted(items, key=lambda value: value["distribution"]) for name, items in sorted(reverse.items())}}


def validate_proposed_graph(graph: Mapping[str, Any]) -> dict[str, Any]:
    reverse = graph.get("reverse_dependencies", {}).get("vcs-versioning", [])
    other = [item for item in reverse if item["distribution"] != "setuptools-scm"]
    proposed = {name: item["version"] for name, item in graph.get("nodes", {}).items()}
    proposed.update({"setuptools-scm": "8.1.0", "packaging": "23.0"})
    proposed.pop("vcs-versioning", None)
    remaining_conflicts = []
    replacement_requirements = ("packaging>=20", "setuptools")
    for raw in replacement_requirements:
        req = Requirement(raw)
        version = proposed.get(canonicalize_name(req.name))
        if version is None or not req.specifier.contains(version, prereleases=True):
            remaining_conflicts.append({"distribution": "setuptools-scm", "requirement": raw, "observed": version})
    for name, item in graph.get("nodes", {}).items():
        if name in {"setuptools-scm", "vcs-versioning"}:
            continue
        for raw in item.get("active_requirements", []):
            req = Requirement(raw)
            dependency = canonicalize_name(req.name)
            if dependency == "packaging" and not req.specifier.contains("23.0", prereleases=True):
                remaining_conflicts.append({"distribution": name, "requirement": raw, "observed": "23.0"})
    curobo_req = Requirement("setuptools-scm>=6.2")
    if not curobo_req.specifier.contains(Version("8.1.0")):
        remaining_conflicts.append({"distribution": "nvidia-curobo", "requirement": str(curobo_req), "observed": "8.1.0"})
    return {"success": not other and not remaining_conflicts, "vcs_versioning_reverse_dependencies": reverse, "other_vcs_versioning_reverse_dependencies": other, "remaining_conflicts": remaining_conflicts, "proposed_versions": proposed}
