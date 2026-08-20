from agent.runtime.mcp_catalog import compact_mcp_tool_catalog


def _tool(name: str, *parameters: str) -> dict:
    return {
        "name": name,
        "input_schema": {
            "type": "object",
            "properties": {parameter: {"type": "object"} for parameter in parameters},
        },
    }


def test_catalog_reports_old_attachment_and_execution_contracts() -> None:
    catalog = compact_mcp_tool_catalog(
        {
            "tools": [
                _tool("gripper_close", "handle", "session_id"),
                _tool("move_to", "handle", "x", "y", "z"),
            ]
        },
        response_path="/tmp/catalog.json",
        grep_hint="grep catalog",
        url="http://sim/sse",
    )

    assert catalog["contract_compatible"] is False
    assert {item["code"] for item in catalog["contract_diagnostics"]} == {
        "attachment_proxy_contract_outdated",
        "motion_contact_authorization_unsupported",
        "motion_ik_seed_unsupported",
    }


def test_catalog_accepts_current_private_simulator_contracts() -> None:
    catalog = compact_mcp_tool_catalog(
        {
            "tools": [
                _tool("gripper_close", "handle", "contact_authorization"),
                _tool(
                    "move_to",
                    "handle",
                    "x",
                    "y",
                    "z",
                    "contact_authorization",
                    "ik_execution_seed",
                ),
            ]
        },
        response_path="/tmp/catalog.json",
        grep_hint="grep catalog",
        url="http://sim/sse",
    )

    assert catalog["contract_compatible"] is True
    assert catalog["contract_diagnostics"] == []
