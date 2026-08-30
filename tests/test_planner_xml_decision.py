"""Contract tests for the main planner XML decision wire format."""

from __future__ import annotations

from agent.runtime.planner import (
    _build_planner_decision,
    _parse_backend_payload,
    _parse_xml_decision,
)


def test_response_talk_roundtrip() -> None:
    payload, errors = _parse_backend_payload(
        "<decision><kind>response</kind><name>talk</name>"
        "<reasoning>Nothing to do.</reasoning>"
        "<parameters><message>hello</message></parameters></decision>"
    )

    assert errors == []
    assert payload == {
        "kind": "response",
        "name": "talk",
        "reasoning": "Nothing to do.",
        "parameters": {"message": "hello"},
    }


def test_empty_parameters_becomes_empty_dict() -> None:
    payload, errors = _parse_backend_payload(
        "<decision><kind>response</kind><name>task_complete</name>"
        "<reasoning>reward</reasoning><parameters/></decision>"
    )

    assert errors == []
    assert payload["parameters"] == {}
    decision, build_errors = _build_planner_decision(payload)
    assert build_errors == []
    assert decision.parameters == {}


def test_dict_host_fallback_branch_is_preserved() -> None:
    payload, errors = _parse_backend_payload(
        {"kind": "response", "name": "ask_human", "parameters": {"message": "x"}}
    )

    assert errors == []
    assert payload["name"] == "ask_human"


def test_main_planner_string_is_strictly_xml_not_legacy_json() -> None:
    payload, errors = _parse_backend_payload(
        '{"kind":"response","name":"talk","parameters":{"message":"legacy"}}'
    )

    assert payload == {}
    assert errors and "<decision>" in errors[0]


def test_multiline_python_code_cdata_preserves_real_newlines() -> None:
    payload, errors = _parse_backend_payload(
        "<decision><kind>tool_call</kind><name>python_exec</name>"
        "<reasoning>Inspect a file.</reasoning><parameters><code><![CDATA[\n"
        "import os\n"
        'result = {"exists": os.path.exists("/tmp/x.json")}\n'
        "]]></code></parameters></decision>"
    )

    assert errors == []
    code = payload["parameters"]["code"]
    assert isinstance(code, str)
    assert "\n" in code
    assert "\\n" not in code
    assert code.count("\n") >= 2


def test_code_policy_top_level_code_is_preserved() -> None:
    payload, errors = _parse_backend_payload(
        "<decision><kind>tool_call</kind><name>code_policy</name>"
        "<reasoning>bounded snippet</reasoning>"
        "<code><![CDATA[return 1 + 1]]></code><parameters/></decision>"
    )

    assert errors == []
    decision, build_errors = _build_planner_decision(payload)
    assert build_errors == []
    assert decision.code == "return 1 + 1"


def test_sam3_point_array_has_numeric_types_and_string_packet_id() -> None:
    payload, errors = _parse_backend_payload(
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

    assert errors == []
    points = payload["parameters"]["points"]
    assert isinstance(points, list) and len(points) == 2
    assert type(points[0]["x"]) is int
    assert type(points[0]["label"]) is int
    assert type(points[1]["x"]) is float
    assert payload["parameters"]["source_packet_id"] == "obs:3:agentview"


def test_boolean_and_null_require_explicit_types() -> None:
    payload, errors = _parse_backend_payload(
        "<decision><kind>tool_call</kind><name>demo_tool</name>"
        "<reasoning>types</reasoning><parameters>"
        '<flag type="boolean">true</flag><missing type="null"/>'
        "<label>true</label></parameters></decision>"
    )

    assert errors == []
    assert payload["parameters"] == {
        "flag": True,
        "missing": None,
        "label": "true",
    }


def test_tool_batch_calls_decode_as_list() -> None:
    payload, errors = _parse_backend_payload(
        "<decision><kind>tool_call</kind><name>tool_batch</name>"
        "<reasoning>two reads</reasoning><calls>"
        "<call><name>get_observation</name><parameters/></call>"
        "<call><name>list_graspgenx_grippers</name><parameters/></call>"
        "</calls></decision>"
    )

    assert errors == []
    decision, build_errors = _build_planner_decision(payload)
    assert build_errors == []
    assert [call["name"] for call in decision.parameters["calls"]] == [
        "get_observation",
        "list_graspgenx_grippers",
    ]


def test_nested_pose_array_decodes_to_floats() -> None:
    payload, errors = _parse_backend_payload(
        "<decision><kind>tool_call</kind><name>move_to</name>"
        "<reasoning>approach</reasoning><parameters><target_pose>"
        "<xyz><item>0.4</item><item>0.1</item><item>0.25</item></xyz>"
        "</target_pose></parameters></decision>"
    )

    assert errors == []
    assert payload["parameters"]["target_pose"]["xyz"] == [0.4, 0.1, 0.25]


def test_fenced_xml_and_surrounding_prose_are_tolerated() -> None:
    payload, errors = _parse_backend_payload(
        "```xml\nHere is the result:\n"
        "<decision><kind>response</kind><name>talk</name>"
        "<reasoning>hi</reasoning><parameters/></decision>\n```"
    )

    assert errors == []
    assert payload["name"] == "talk"


def test_malformed_or_missing_xml_returns_parse_feedback() -> None:
    payload, errors = _parse_xml_decision(
        "<decision><kind>tool_call</kind><name>sam3</reasoning></decision>"
    )
    assert payload == {}
    assert errors and "invalid XML" in errors[0]

    payload, errors = _parse_backend_payload("I think we should pick the can.")
    assert payload == {}
    assert errors and "<decision>" in errors[0]
