"""Tests for the XML planner-decision parser.

The main embodied planner returns its decision as an XML ``<decision>``
element. These tests lock the parser contract that ``_parse_backend_payload``
now enforces: the XML->dict output must match the dict shape the previous JSON
parser produced, with real Python types, and multi-line code must survive
CDATA with literal newlines (the bug that motivated the JSON->XML switch).
"""

from __future__ import annotations

from agent.runtime.planner import (
    _build_planner_decision,
    _parse_backend_payload,
    _parse_xml_decision,
)


def test_response_talk_roundtrip() -> None:
    xml = (
        "<decision><kind>response</kind><name>talk</name>"
        "<reasoning>Nothing to do.</reasoning>"
        "<parameters><message>hello</message></parameters></decision>"
    )
    payload, errors = _parse_backend_payload(xml)
    assert errors == []
    assert payload["kind"] == "response"
    assert payload["name"] == "talk"
    assert payload["reasoning"] == "Nothing to do."
    assert payload["parameters"] == {"message": "hello"}


def test_empty_parameters_becomes_empty_dict() -> None:
    xml = (
        "<decision><kind>response</kind><name>task_complete</name>"
        "<reasoning>reward</reasoning><parameters/></decision>"
    )
    payload, errors = _parse_backend_payload(xml)
    assert errors == []
    assert payload["parameters"] == {}
    decision, build_errors = _build_planner_decision(payload)
    assert build_errors == []
    assert decision.parameters == {}


def test_dict_input_branch_preserved() -> None:
    # Backend fallback payloads are dicts and must still pass through unchanged.
    payload, errors = _parse_backend_payload(
        {"kind": "response", "name": "ask_human", "parameters": {"message": "x"}}
    )
    assert errors == []
    assert payload["name"] == "ask_human"


def test_multiline_code_cdata_preserves_real_newlines() -> None:
    xml = (
        "<decision><kind>tool_call</kind><name>python_exec</name>"
        "<reasoning>Inspect a file.</reasoning>"
        "<parameters><code><![CDATA[\n"
        "import os\n"
        "result = {\"exists\": os.path.exists(\"/tmp/x.json\")}\n"
        "]]></code></parameters></decision>"
    )
    payload, errors = _parse_backend_payload(xml)
    assert errors == []
    code = payload["parameters"]["code"]
    assert isinstance(code, str)
    # Real newline byte present, literal backslash-n absent -> the escaping bug
    # this whole change targets is gone.
    assert "\n" in code
    assert "\\n" not in code
    assert "import os" in code
    assert code.count("\n") >= 2


def test_code_policy_top_level_code() -> None:
    xml = (
        "<decision><kind>tool_call</kind><name>code_policy</name>"
        "<reasoning>bounded snippet</reasoning>"
        "<code><![CDATA[return 1 + 1]]></code><parameters/></decision>"
    )
    payload, errors = _parse_backend_payload(xml)
    assert errors == []
    decision, build_errors = _build_planner_decision(payload)
    assert build_errors == []
    assert decision.code == "return 1 + 1"


def test_sam3_points_typing_int_float_and_not_bool() -> None:
    xml = (
        "<decision><kind>tool_call</kind><name>sam3</name>"
        "<reasoning>point at target</reasoning><parameters>"
        "<source_packet_id>obs:3:agentview</source_packet_id>"
        "<mode>points</mode><points>"
        '<item><x type="integer">272</x><y type="integer">152</y>'
        '<label type="integer">1</label></item>'
        '<item><x type="number">305.5</x><y type="integer">200</y>'
        '<label type="integer">0</label></item>'
        "</points></parameters></decision>"
    )
    payload, errors = _parse_backend_payload(xml)
    assert errors == []
    points = payload["parameters"]["points"]
    assert isinstance(points, list) and len(points) == 2
    first = points[0]
    assert set(first) == {"x", "y", "label"}
    assert type(first["x"]) is int
    assert type(first["label"]) is int
    # Must be a real int, never a bool, or the sam3 validator rejects it.
    assert not isinstance(first["label"], bool)
    assert type(points[1]["x"]) is float
    # A numeric-looking id stays a string via the reserved-name rule.
    assert payload["parameters"]["source_packet_id"] == "obs:3:agentview"


def test_boolean_and_null_require_explicit_type() -> None:
    xml = (
        "<decision><kind>tool_call</kind><name>demo_tool</name>"
        "<reasoning>types</reasoning><parameters>"
        '<flag type="boolean">true</flag>'
        '<missing type="null"/>'
        "<label>true</label>"  # no type -> stays the string "true"
        "</parameters></decision>"
    )
    payload, errors = _parse_backend_payload(xml)
    assert errors == []
    params = payload["parameters"]
    assert params["flag"] is True
    assert params["missing"] is None
    assert params["label"] == "true"


def test_tool_batch_calls_list() -> None:
    xml = (
        "<decision><kind>tool_call</kind><name>tool_batch</name>"
        "<reasoning>two reads</reasoning><calls>"
        "<call><name>get_observation</name><parameters/></call>"
        "<call><name>list_graspgenx_grippers</name><parameters/></call>"
        "</calls></decision>"
    )
    payload, errors = _parse_backend_payload(xml)
    assert errors == []
    decision, build_errors = _build_planner_decision(payload)
    assert build_errors == []
    calls = decision.parameters["calls"]
    assert [call["name"] for call in calls] == [
        "get_observation",
        "list_graspgenx_grippers",
    ]


def test_move_to_nested_xyz() -> None:
    xml = (
        "<decision><kind>tool_call</kind><name>move_to</name>"
        "<reasoning>approach</reasoning><parameters><target_pose>"
        "<xyz><item>0.4</item><item>0.1</item><item>0.25</item></xyz>"
        "</target_pose></parameters></decision>"
    )
    payload, errors = _parse_backend_payload(xml)
    assert errors == []
    xyz = payload["parameters"]["target_pose"]["xyz"]
    assert xyz == [0.4, 0.1, 0.25]
    assert all(isinstance(value, float) for value in xyz)


def test_fenced_xml_is_tolerated() -> None:
    xml = (
        "```xml\n"
        "<decision><kind>response</kind><name>talk</name>"
        "<reasoning>hi</reasoning>"
        "<parameters><message>fenced</message></parameters></decision>\n"
        "```"
    )
    payload, errors = _parse_backend_payload(xml)
    assert errors == []
    assert payload["parameters"]["message"] == "fenced"


def test_decision_span_salvaged_from_surrounding_prose() -> None:
    xml = (
        "Sure, here is my decision:\n"
        "<decision><kind>response</kind><name>talk</name>"
        "<reasoning>hi</reasoning><parameters/></decision>\n"
        "Let me know if that works."
    )
    payload, errors = _parse_backend_payload(xml)
    assert errors == []
    assert payload["name"] == "talk"


def test_malformed_xml_returns_parse_error() -> None:
    # Mismatched tags -> a parse error that flows into the validation-retry loop.
    payload, errors = _parse_xml_decision(
        "<decision><kind>tool_call</kind><name>sam3</reasoning></decision>"
    )
    assert payload == {}
    assert errors and "invalid XML" in errors[0]


def test_missing_decision_element_returns_error() -> None:
    payload, errors = _parse_backend_payload("I think we should pick the can.")
    assert payload == {}
    assert errors and "<decision>" in errors[0]
