from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from agent.runtime.memory import AgentMemory, GRASP_GEOMETRY_FAMILIES as MEMORY_FAMILIES
from agent.tools.contracts import (
    build_default_tool_contract_catalog,
    check_tool_request_conformance,
    project_agent_tool_contract,
    render_tool_contract_markdown,
)
from agent.tools.grasp_types import GRASP_GEOMETRY_FAMILIES
from agent.tools.registry import build_default_tool_registry


def _pending_memory() -> AgentMemory:
    memory = AgentMemory()
    memory.start_session(task="inspect a target")
    memory.save_fact(
        "pending_sam3_selection",
        {
            "result_id": "sam-geometry",
            "evidence_role": "target_object",
            "source_packet_id": "packet-1",
            "candidates": [{"id": "detection_000", "mask_ref": "target.png"}],
        },
        source="test",
    )
    return memory


def _select(memory: AgentMemory, **kwargs):
    return memory.resolve_sam3_selection(
        result_id="sam-geometry",
        detection_id="detection_000",
        selection_source="main_agent_vlm",
        **kwargs,
    )


def _contract(name="select_sam3_detection"):
    return build_default_tool_contract_catalog(build_default_tool_registry().list()).get(name)


@pytest.mark.parametrize("family", ["", *sorted(GRASP_GEOMETRY_FAMILIES)])
def test_selection_schema_and_runtime_accept_the_same_canonical_vocabulary(family):
    contract = _contract()
    assert MEMORY_FAMILIES is GRASP_GEOMETRY_FAMILIES
    assert contract.request_schema["properties"]["target_geometry_family"]["enum"] == [
        "", *sorted(GRASP_GEOMETRY_FAMILIES)
    ]
    assert not check_tool_request_conformance(contract, {
        "sam3_result_id": "sam-geometry",
        "detection_id": "detection_000",
        "target_geometry_family": family,
    })
    selected = _select(_pending_memory(), target_geometry_family=family)
    if family in {"", "unknown"}:
        assert "target_geometry_family" not in selected
    else:
        assert selected["target_geometry_family"] == family


@pytest.mark.parametrize("family", ["mug", "mug_handle", "custom:tool", None, 4, [], {}])
def test_invalid_selection_hint_is_rejected_without_consuming_evidence(family):
    memory = _pending_memory()
    pending = deepcopy(memory.pending_sam3_selection())
    assert check_tool_request_conformance(_contract(), {
        "sam3_result_id": "sam-geometry",
        "detection_id": "detection_000",
        "target_geometry_family": family,
    })
    with pytest.raises(ValueError, match="target_geometry_family"):
        _select(memory, target_geometry_family=family)
    assert memory.pending_sam3_selection() == pending
    assert memory.selected_sam3_detection() is None


def test_omitted_and_legacy_noncanonical_hints_remain_compatible():
    assert "target_geometry_family" not in _select(_pending_memory())
    selected = _select(_pending_memory(), target_geometry_family="  BoWL  ")
    assert selected["target_geometry_family"] == "bowl"
    # Planner-authored requests use canonical spelling, while old direct callers
    # keep the existing trim/lower compatibility path in the memory API.
    assert check_tool_request_conformance(_contract(), {
        "sam3_result_id": "sam-geometry",
        "detection_id": "detection_000",
        "target_geometry_family": "  BoWL  ",
    })


@pytest.mark.parametrize("family", ["mug", "mug_handle", "custom:tool", ""])
def test_compiler_contract_preserves_extension_strings(family):
    contract = _contract("compile_grasp_seed")
    schema = contract.request_schema["properties"]["target_geometry_family"]
    assert "enum" not in schema
    assert schema["examples"] == sorted(GRASP_GEOMETRY_FAMILIES)
    assert not check_tool_request_conformance(contract, {
        "grasp_result_id": "grasp-1",
        "candidate_id": "candidate-1",
        "target_geometry_family": family,
    })


def test_agent_projection_retains_selection_enum():
    projection = project_agent_tool_contract(_contract())
    assert projection["parameters"]["properties"]["target_geometry_family"]["enum"] == [
        "", *sorted(GRASP_GEOMETRY_FAMILIES)
    ]


def test_generated_contract_documents_match_current_catalog():
    generated = Path(__file__).resolve().parents[1] / "docs" / "generated"
    catalog = build_default_tool_contract_catalog(build_default_tool_registry().list())
    assert json.loads((generated / "tool-contracts.json").read_text()) == catalog.to_dict()
    assert (generated / "tool-contracts.md").read_text() == render_tool_contract_markdown(catalog)
