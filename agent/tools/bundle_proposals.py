"""Experimental read-only proposal authoring; no simulator or actuator dispatch."""
from copy import deepcopy

from agent.tools.contracts import ToolContract, OutcomeContract, check_tool_request_conformance
from agent.tools.registry import ToolSpec, ToolResult


def proposal_contract(name):
    from agent.tools.contracts import build_default_tool_contract_catalog
    from agent.tools.registry import build_default_tool_registry
    if name == "propose_motion_target":
        base = build_default_tool_contract_catalog(build_default_tool_registry().list()).get("ik_preview_check")
        schema = deepcopy(base.request_schema)
        schema["properties"].pop("bundle_id")
        schema["oneOf"] = schema["oneOf"][1:]
        description = (
            "Author one target_pose bundle without moving or running IK. Supply a world target_pose "
            "or an existing compiled-grasp role/path_fraction, viewpoint candidate, or frozen probe waypoint. "
            "Use this for new geometry; existing producer bundles can go directly to ik_preview_check. "
            "Preserving current orientation is full-pose IK, never position-only execution."
        )
    elif name == "compose_ik_trajectory":
        from agent.runtime.tool_bundles import BUNDLE_ID_PATTERN
        schema = {"type": "object", "properties": {"bundle_ids": {"type": "array",
            "items": {"type": "string", "minLength": 1, "pattern": f"^{BUNDLE_ID_PATTERN}$"}, "minItems": 1, "maxItems": 5}},
            "required": ["bundle_ids"], "additionalProperties": False}
        description = (
            "Group 1-5 ordered, current ik_result bundle_ids into one ik_trajectory bundle. "
            "Does not move, approve the path or refresh receipt lifetimes. Agent chooses the order; "
            "follow_eef_trajectory still checks every exact receipt and the combined path."
        )
    else:
        raise ValueError("unknown proposal tool")
    return ToolContract(name=name, category="planning", description=description,
        effect="planning", request_schema=schema, safe_by_default=True, batchable=False,
        outcomes=(OutcomeContract("proposal_registered", True, {"type": "object", "additionalProperties": True}),),
        source_paths=("agent/tools/bundle_proposals.py",),
        coverage_gaps=("experimental_profile_requires_collaborator_review",))


def bind_profile_proposals(tools, profile, memory_provider):
    from agent.runtime.interface_profiles import BUNDLE_STAGE2, BUNDLE_STAGE3
    if profile not in {BUNDLE_STAGE2, BUNDLE_STAGE3}:
        return
    names = ["propose_motion_target"]
    if profile == BUNDLE_STAGE3:
        names.append("compose_ik_trajectory")
    for name in names:
        contract = proposal_contract(name)
        tools.register(ToolSpec(name=name, description=contract.description, category="planning",
            parameters=contract.request_schema["properties"], effect="planning", safe_by_default=True, batchable=False),
            _handler(name, memory_provider))


def _handler(name, memory_provider):
    def handler(context):
        from agent.runtime.planner import _validate_tool_parameters
        contract = proposal_contract(name)
        violations = check_tool_request_conformance(contract, context.parameters)
        try:
            if violations:
                raise ValueError("; ".join(v.message for v in violations))
            memory = memory_provider()
            if name == "propose_motion_target":
                errors = _validate_tool_parameters("ik_preview_check", context.parameters)
                if errors:
                    raise ValueError("; ".join(errors))
                proposal = {"kind": "target_pose", "reference_parameters": deepcopy(context.parameters),
                    "summary": {"consumer_tool": "ik_preview_check", "request_reference": deepcopy(context.parameters),
                                "purpose": "agent_authored_geometry", "motion_authorized": False}}
            else:
                ids = context.parameters["bundle_ids"]
                receipts = [memory.resolve_tool_bundle("move_to", {"bundle_id": value})["ik_receipt_id"] for value in ids]
                if len(set(receipts)) != len(receipts):
                    raise ValueError("Duplicate IK receipts are not a trajectory")
                proposal = {"kind": "ik_trajectory", "reference_parameters": {"ik_receipt_ids": receipts}, "parents": list(ids),
                    "summary": {"consumer_tool": "follow_eef_trajectory", "waypoint_count": len(receipts),
                                "receipt_ids": receipts, "motion_authorized": False}}
        except (TypeError, ValueError, OSError) as exc:
            return ToolResult(False, f"Proposal was not registered: {exc}",
                              {"diagnostics": [{"code": "invalid_bundle_proposal", "message": str(exc)}]})
        # Registration occurs only in the runtime's owned memory commit after
        # this read-only call returns. A late/cancelled handler cannot write.
        return ToolResult(True, "Proposal prepared for Host registration; it does not authorize execution.",
            {"outputs": {"bundle_proposal": proposal, "motion_authorized": False}})
    return handler
