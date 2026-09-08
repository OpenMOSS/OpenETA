import json

import pytest

from agent.backends.planner import CallablePlannerBackend, PlannerBackendResult
from agent.backends.request_lineage import isolated_request_lineage
from agent.runtime.supervision import BackendActionReviewer, BackendGuidanceResolver
from agent.tools.registry import ToolExecutionContext, build_default_tool_registry
from tools.manual_vlm_openeta import OpenETAProtocolAdapter, classify_request
from tools.manual_vlm_proxy import RequestStore


def wire(context):
    return {"messages": [{"role": "user", "content": json.dumps({
        "instruction": "Return the requested JSON.", "tool_context": context,
    })}], "response_format": {"type": "json_object"}}


@pytest.mark.parametrize("parent", [None, "", "bad\nparent", "bad\x7fparent", "x" * 257, {}, 7])
def test_lineage_never_invents_invalid_parent(parent):
    assert isolated_request_lineage(parent) == {}


@pytest.mark.parametrize("schema,role,label", [
    ("openeta.supervision.v1", "guidance_agent", "guidance_agent"),
    ("openeta.supervision.v1", "independent_action_reviewer", "action_reviewer"),
    ("openeta.visual_delta_request.v1", "visual_differencing", "visual_differencing"),
])
def test_isolated_roles_are_grouped_without_merging_parent_history(schema, role, label):
    store = RequestStore(adapter=OpenETAProtocolAdapter())
    parent = store.add({"messages": []}, session_hint="parent")
    contexts = [{"schema_version": schema, "role": role,
                 **isolated_request_lineage("parent"),
                 "history": {"session_id": "wrong-historical-session"}} for _ in range(2)]
    children = [store.add(wire(context)) for context in contexts]
    assert len({parent.session_id, *(child.session_id for child in children)}) == 3
    for child in children:
        detail = store.public_detail(child.request_id)
        assert detail["parent_session_id"] == "parent"
        assert detail["request_type"] == label
        assert detail["wait_reason"].startswith("等待人工")
        assert child.session_turn == 1
    store.cancel(children[0].request_id, "cancel only this invocation")
    assert store.public_detail(children[0].request_id)["wait_reason"] == ""
    assert parent.status == children[1].status == "pending"


@pytest.mark.parametrize("role", ["main_planner", "unknown", {}, []])
def test_unrecognized_role_cannot_claim_lineage(role):
    body = wire({"schema_version": "openeta.supervision.v1", "role": role,
                 **isolated_request_lineage("parent")})
    assert "parent_session_id" not in classify_request(body)


def test_guidance_emits_only_explicit_host_parent_and_unique_invocations():
    requests = []

    def decide(request):
        requests.append(request)
        return PlannerBackendResult(payload={"decision": "abstain", "reason": "uncertain"})

    resolver = BackendGuidanceResolver(CallablePlannerBackend(decide))
    for parent in ["host-session", "host-session", None]:
        context = {"memory": {"session_id": "historical-session"}}
        if parent:
            context["_host_parent_session_id"] = parent
        resolver.resolve(question="which target?", context=context)
    first, second, legacy = [request.tool_context for request in requests]
    assert first["request_lineage"]["parent_session_id"] == "host-session"
    assert first["request_lineage"]["child_session_id"] != second["request_lineage"]["child_session_id"]
    assert "_host_parent_session_id" not in first["session_context"]
    assert "request_lineage" not in legacy
    assert classify_request(wire(first))["type"] == "guidance_agent"


def test_action_reviewer_uses_runtime_metadata_not_nested_history():
    requests = []

    def decide(request):
        requests.append(request)
        return PlannerBackendResult(payload={"decision": "approve", "reason": "fixture"})

    context = ToolExecutionContext(name="move_to", spec=build_default_tool_registry().get("move_to"),
        parameters={"target_pose": {"frame": "world", "xyz": [0, 0, 0.3]}},
        metadata={"session_id": "host-session", "supervision_context": {
            "memory": {"session_id": "wrong-history"},
        }})
    BackendActionReviewer(CallablePlannerBackend(decide)).review(context)
    payload = requests[0].tool_context
    assert payload["request_lineage"]["parent_session_id"] == "host-session"
    assert classify_request(wire(payload))["type"] == "action_reviewer"
