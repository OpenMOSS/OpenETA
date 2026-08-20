from __future__ import annotations

import pytest

from agent.runtime.planner import _validate_tool_parameters


def test_molmopoint_planner_accepts_ordered_packet_sources_and_complete_prompt() -> None:
    assert _validate_tool_parameters(
        "molmopoint",
        {
            "sources": [
                {"source_packet_id": "obs-0000", "camera_frame_id": "agentview"},
                {"source_packet_id": "obs-0001", "camera_frame_id": "wrist"},
            ],
            "prompt": "Look at Image 1. In Image 2, point to the same object.",
        },
    ) == []


@pytest.mark.parametrize(
    "parameters",
    [
        {"sources": [], "prompt": "Point to a cup."},
        {"images": ["/tmp/image.png"], "prompt": "Point to a cup."},
        {"sources": [{"source_packet_id": ""}], "prompt": "Point to a cup."},
        {
            "sources": [{"source_packet_id": "obs-0000", "image": "/tmp/image.png"}],
            "prompt": "Point to a cup.",
        },
        {"sources": [{"source_packet_id": "obs-0000"}], "prompt": "<prompt>"},
        {"sources": [{"source_packet_id": "obs-0000"}], "prompt": "x" * 1025},
    ],
)
def test_molmopoint_planner_rejects_invalid_structure_and_placeholders(parameters) -> None:
    assert _validate_tool_parameters("molmopoint", parameters)
