from __future__ import annotations

import pytest

from agent.runtime.planner import _host_obligation_decision, _validate_tool_parameters
from agent.tools.handlers import bind_dummy_tool_handlers
from agent.tools.registry import build_default_tool_registry


def _registry_without_molmopoint():
    tools = bind_dummy_tool_handlers(build_default_tool_registry())
    tools.unbind_handler("molmopoint")
    assert not tools.can_execute("molmopoint")
    return tools


_POINT_FALLBACK_REQUIRED = {
    "schema_version": "openeta.molmopoint_fallback_obligation.v1",
    "status": "required",
    "attempt": 1,
    "required_parameters": {
        "images": ["/tmp/scene.png"],
        "prompt": "Point to the can.",
    },
}


def test_molmopoint_planner_accepts_ordered_paths_and_complete_prompt() -> None:
    assert _validate_tool_parameters(
        "molmopoint",
        {
            "images": ["/tmp/reference.png", "/tmp/scene.jpg"],
            "prompt": "Look at Image 1. In Image 2, point to the same object.",
        },
    ) == []


@pytest.mark.parametrize(
    "parameters",
    [
        {"images": [], "prompt": "Point to a cup."},
        {"images": ["<image>"], "prompt": "Point to a cup."},
        {"images": ["/tmp/image.png"], "prompt": "<prompt>"},
        {"images": ["/tmp/image.png"], "prompt": "x" * 1025},
    ],
)
def test_molmopoint_planner_rejects_invalid_structure_and_placeholders(parameters) -> None:
    assert _validate_tool_parameters("molmopoint", parameters)


def test_disabled_molmopoint_fallback_escalates_instead_of_stalling() -> None:
    # When the pointing backend is not configured, a still-`required` fallback must
    # escalate to ask_human rather than fall through and loop until tool_call_limit.
    tools = _registry_without_molmopoint()
    decision = _host_obligation_decision(
        {"molmopoint_fallback_obligation": dict(_POINT_FALLBACK_REQUIRED)},
        tools=tools,
    )
    assert decision is not None
    assert decision.action_type == "response"
    assert decision.action == "ask_human"
    assert decision.parameters["failure_code"] == "target_localization_exhausted"
    assert decision.metadata["host_obligation"]["status"] == "molmopoint_unavailable"


def test_exhausted_molmopoint_fallback_still_escalates_when_disabled() -> None:
    tools = _registry_without_molmopoint()
    decision = _host_obligation_decision(
        {
            "molmopoint_fallback_obligation": {
                "schema_version": "openeta.molmopoint_fallback_obligation.v1",
                "status": "exhausted",
            }
        },
        tools=tools,
    )
    assert decision is not None
    assert decision.action == "ask_human"
    assert decision.metadata["host_obligation"]["status"] == "exhausted"
