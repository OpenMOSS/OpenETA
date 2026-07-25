from __future__ import annotations

import pytest

from agent.runtime.planner import _validate_tool_parameters


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
