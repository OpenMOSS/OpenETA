"""Public planner feedback must not ask the model to assemble Host RGB-D inputs."""
from copy import deepcopy

import pytest

from adapter.protocol import EnvObservation, RobotState
from agent.backends.planner import StaticPlannerBackend
from agent.runtime.memory import AgentMemory
from agent.runtime.planner import ToolCallingPlanner, _validate_tool_parameters
from agent.runtime.skills import build_default_skill_registry
from agent.tools.registry import build_default_tool_registry


@pytest.mark.parametrize("parameters", [
    {}, {"object_description": "alphabet soup", "evidence_ids": ["observation:0:agentview"]},
    *[{"bundle_id": value} for value in (None, "", "  ", 1, True, [], {})],
    {"rgb": "/fixture/rgb.png", "depth": "/fixture/depth.npy", "mode": "scene",
     "intrinsics": {"fx": 100, "fy": 100, "cx": 50, "cy": 50, "scale": 1000},
     "camera_frame_id": "agentview", "scene_epoch": 0},
])
def test_missing_or_invalid_bundle_requires_public_reference_not_raw_inputs(parameters):
    original = deepcopy(parameters)
    errors = _validate_tool_parameters("grasp_pose_estimate", parameters)
    assert errors
    assert any("parameters.bundle_id" in error for error in errors)
    assert not any("as a concrete local path" in error or "must contain fx" in error
                   or "requires a concrete camera_frame_id" in error for error in errors)
    assert parameters == original


def test_valid_bundle_remains_syntactic_only_and_cannot_override_host_inputs():
    # Existence/freshness belongs to the runtime resolver, not this syntactic check.
    assert not _validate_tool_parameters("grasp_pose_estimate", {"bundle_id": "grasp:opaque"})
    assert _validate_tool_parameters("grasp_pose_estimate", {"bundle_id": "grasp:opaque", "rgb": "/other.png"})
    assert _validate_tool_parameters("grasp_pose_estimate", {"bundle_id": "grasp:opaque", "backend_preference": ["invented"]})


def test_actual_planner_retry_feedback_preserves_published_bundle_contract():
    class Backend(StaticPlannerBackend):
        def __init__(self):
            super().__init__([
                {"kind": "tool_call", "name": "grasp_pose_estimate",
                 "parameters": {"object_description": "alphabet soup", "evidence_ids": ["observation:0:agentview"]}},
                {"kind": "response", "name": "talk", "parameters": {"message": "No ready grasp bundle; inspect perception evidence."}},
            ])
            self.requests = []

        def decide(self, request):
            self.requests.append(deepcopy(request))
            return super().decide(request)

    backend = Backend()
    memory = AgentMemory()
    memory.start_session(task="pick the soup can")
    tools = build_default_tool_registry()
    tools.bind_handler("grasp_pose_estimate", lambda _context: pytest.fail("Invalid request executed"))
    decision = ToolCallingPlanner(backend, max_validation_retries=1).plan(
        EnvObservation(task="pick the soup can", cameras=[], robot=RobotState()), memory=memory,
        tools=tools, skills=build_default_skill_registry(),
    )
    assert decision.action == "talk"
    assert len(backend.requests) == 2
    assert any("parameters.bundle_id" in error for error in backend.requests[1].validation_errors)
    assert not any("parameters.rgb" in error or "parameters.depth" in error
                   for error in backend.requests[1].validation_errors)
