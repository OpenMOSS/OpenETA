from copy import deepcopy

import pytest

from adapter.protocol import EnvAction, EnvObservation, RobotState
from agent.runtime.memory import AgentMemory, _reconcile_gripper_position
from agent.tools.gripper_evidence import (
    gripper_actuation_receipt_error,
    measured_gripper_open,
)


@pytest.mark.parametrize(("state", "expected"), [
    ({"open": True, "openness": 0.4}, False),
    ({"open": False, "openness": 0.95}, True),
    ({"openness": 0.8}, True),
    ({"openness": 0.799}, False),
    ({"openness": 0.0}, False),
    ({"openness": 1.0}, True),
    ({"open": True}, True),
    ({"open": False}, False),
    ({"openness": None, "open": False}, False),
    ({}, None),
    ({"open": "false"}, None),
    *[({"open": False, "openness": value}, None)
      for value in [True, False, float("nan"), float("inf"), -0.1, 1.1, "0.4", {}, 10**400]],
])
def test_aperture_precedence_and_invalid_measurements(state, expected):
    assert measured_gripper_open(state) is expected
    for position in (0, 1):
        assert _reconcile_gripper_position(position, state) == (
            "completed" if expected is not None and expected == bool(position)
            else "unresolved"
        )


def _receipt(command="close"):
    return {
        "schema_version": "openeta.gripper_actuation_receipt.v1",
        "command": command,
        "command_latched": True,
        "steps_executed": 60,
        "measured_open_fraction": 0.4,
    }


def _action(name="gripper_control", *, position=0, outputs=None, success=True):
    parameters = {"position": position} if name == "gripper_control" else {}
    return EnvAction(action_type="tool_call", command={
        "request": {"kind": "tool_call", "name": name, "parameters": parameters},
        "status": "executed" if success else "failed",
        "tool_calls": [{
            "name": name, "status": "executed" if success else "failed",
            "parameters": parameters,
            "result": {"success": success, "details": {"outputs": outputs or {}}},
        }],
    })


def _closed_memory():
    memory = AgentMemory()
    memory.start_session(task="inspect attachment evidence")
    memory.add_action(_action(outputs={
        "gripper_actuation_receipt": _receipt(),
        "attachment_proxy_receipt": {"status": "tentative"},
    }))
    memory.save_fact("grasp_provenance", {
        "compiled_grasp_id": "compiled-1", "candidate_id": "candidate-1",
    }, source="test")
    memory.save_fact("attachment_evidence", {
        "compiled_grasp_id": "compiled-1", "verdict": "PASS",
    }, source="test")
    return memory


def test_acknowledged_receipt_and_attachment_survive_unrelated_arm_motion():
    memory = _closed_memory()
    before = deepcopy(memory.gripper_command_state())
    attachment = deepcopy(memory.attachment_evidence())
    memory.add_action(_action("move_to"))
    assert memory.gripper_command_state() == before
    assert before["gripper_actuation_receipt"] == _receipt()
    assert memory.attachment_evidence() == attachment


@pytest.mark.parametrize("patch", [
    {"command": "open"}, {"command_latched": False}, {"command_latched": 1},
    {"schema_version": "other"}, {"steps_executed": 0},
    {"steps_executed": True}, {"steps_executed": -1}, {"steps_executed": 60.0},
])
def test_inconsistent_modern_receipt_cannot_inherit_attachment(patch):
    memory = _closed_memory()
    receipt = {**_receipt(), **patch}
    assert gripper_actuation_receipt_error(receipt, position=0)
    memory.add_action(_action(outputs={
        "gripper_actuation_receipt": receipt,
        "attachment_proxy_receipt": {"status": "tentative"},
    }))
    state = memory.gripper_command_state()
    assert state["latched"] is False
    assert state["receipt_error"]
    assert "attachment_proxy_receipt" not in state
    assert memory.attachment_evidence() is None


@pytest.mark.parametrize("receipt", [None, "invalid", []])
def test_present_malformed_receipt_is_not_treated_as_legacy_absence(receipt):
    memory = _closed_memory()
    memory.add_action(_action(outputs={"gripper_actuation_receipt": receipt}))
    assert memory.gripper_command_state()["latched"] is False
    assert memory.attachment_evidence() is None


def test_reconciled_open_retires_attachment_and_contact_without_inventing_latch():
    memory = _closed_memory()
    memory.add_action(_action(position=1, outputs={"motion_outcome": "unknown"}, success=False))
    memory.add_observation(EnvObservation(task="inspect attachment evidence", cameras=[], robot=RobotState(
        gripper_state={"open": True, "openness": 0.4},
    )))
    assert memory.motion_reconciliation()["status"] == "unresolved"
    assert memory.motion_reconciliation_gate_error(tool_name="prepare_attachment_probe")
    memory.add_observation(EnvObservation(task="inspect attachment evidence", cameras=[], robot=RobotState(
        gripper_state={"open": False, "openness": 0.95},
    )))
    assert memory.motion_reconciliation()["position_reconciliation_status"] == "completed"
    assert memory.motion_reconciliation()["status"] == "unresolved"
    assert memory.motion_reconciliation_gate_error(tool_name="move_to")
    assert memory.gripper_command_state()["position"] == 1
    assert memory.gripper_command_state()["latched"] is False
    assert memory.attachment_evidence() is None
    assert memory.facts["grasp_provenance"]["value"]["contact_geometry_invalidated_by"] == (
        "gripper_reopened_after_close"
    )


def test_reconciled_close_does_not_inherit_old_proxy_or_attachment():
    memory = _closed_memory()
    memory.add_action(_action(outputs={"motion_outcome": "unknown"}, success=False))
    memory.add_observation(EnvObservation(task="inspect attachment evidence", cameras=[], robot=RobotState(
        gripper_state={"open": True, "openness": 0.4},
    )))
    state = memory.gripper_command_state()
    assert memory.motion_reconciliation()["position_reconciliation_status"] == "completed"
    assert memory.motion_reconciliation()["status"] == "unresolved"
    assert memory.motion_reconciliation_gate_error(tool_name="move_to")
    assert state["position"] == 0
    assert state["latched"] is False
    assert "gripper_actuation_receipt" not in state
    assert "attachment_proxy_receipt" not in state
    assert memory.attachment_evidence() is None
    # Lack of latch acknowledgement alone does not assert that EEF contact moved.
    assert "contact_geometry_invalidated_by" not in memory.facts["grasp_provenance"]["value"]


@pytest.mark.parametrize("state", [{"openness": 0.95, "open": False}, {"open": True}])
def test_fresh_open_telemetry_revokes_pass_without_overwriting_command_history(state):
    memory = _closed_memory()
    command = deepcopy(memory.gripper_command_state())
    memory.add_observation(EnvObservation(
        task="inspect attachment evidence", cameras=[], robot=RobotState(gripper_state=state),
    ))
    attachment = memory.attachment_evidence()
    assert attachment["verdict"] == "UNKNOWN"
    assert attachment["invalidation_reason"] == "measured_gripper_open"
    assert memory.gripper_command_state() == command


def test_partial_aperture_does_not_revoke_pass_from_coarse_open_flag():
    memory = _closed_memory()
    memory.add_observation(EnvObservation(
        task="inspect attachment evidence", cameras=[],
        robot=RobotState(gripper_state={"open": True, "openness": 0.4}),
    ))
    assert memory.attachment_evidence()["verdict"] == "PASS"
