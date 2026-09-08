"""One deterministic Agent receipt dispatch inside the owned controller canary.

No model, perception service, environment creation, or retry occurs here. The
caller supplies the actual preview and remains responsible for environment close.
"""
from __future__ import annotations

from copy import deepcopy

from adapter.protocol import EnvAction, EnvObservation, RobotState
from agent.backends.planner import StaticPlannerBackend
from agent.runtime.planner import ToolCallingPlanner
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.tools.registry import build_default_tool_registry
from agent.tools.call_budget import ToolCallBudget
from agent.tools.sim_mcp import (
    SimulatorMcpToolProxyConfig,
    _ik_preview_receipt,
    bind_simulator_mcp_tool_handlers,
)


def runtime_seed_move(*, transport, preview, target, handle, session_id,
                      tolerance_m, max_steps, timeout_s):
    preview_parameters = {
        "target_pose": {"frame": "world", "xyz": list(target)},
        "preserve_current_orientation": True,
        "position_tolerance_m": tolerance_m,
        "orientation_tolerance_rad": 0.05,
        "check_endpoint_collision": True,
        "include_scene_objects": True,
    }
    # Same production compiler used by the simulator proxy. Input is the actual
    # just-completed preview, not a synthetic feasible fixture or repaired result.
    receipt = _ik_preview_receipt(preview_parameters, deepcopy(preview))
    if receipt.get("classification") != "feasible":
        raise RuntimeError("Runtime probe requires an actual feasible preview")
    request = {"ik_receipt_id": receipt["receipt_id"], "num_steps": max_steps,
               "enable_collision_check": True}
    dispatched = []

    class SingleMoveTransport:
        def call_tool(self, name, arguments, *, timeout_s=None):
            expected = {
                "handle": handle, "session_id": session_id,
                "x": target[0], "y": target[1], "z": target[2],
                "tolerance": tolerance_m, "ori_tolerance": 0.05,
                "num_steps": max_steps, "enable_collision_check": True,
            }
            seed = arguments.get("ik_execution_seed")
            if (dispatched or name != "move_to"
                    or any(arguments.get(key) != value for key, value in expected.items())
                    or set(arguments) != {*expected, "ik_execution_seed"}
                    or not isinstance(seed, dict)
                    or seed.get("receipt_id") != receipt["receipt_id"]
                    or seed.get("joint_positions") != preview["best_candidate"]["joint_positions"]):
                raise RuntimeError("Runtime probe refused unexpected or duplicate motion dispatch")
            # Reserve before RPC: timeout must never allow a second attempt.
            dispatched.append({"seed": deepcopy(seed)})
            response = transport.call_tool(name, arguments, timeout_s=timeout_s)
            dispatched[0]["response"] = deepcopy(response)
            return response

    registry = bind_simulator_mcp_tool_handlers(
        build_default_tool_registry(), transport=SingleMoveTransport(),
        config=SimulatorMcpToolProxyConfig(handle=handle, session_id=session_id,
                                           timeout_s=timeout_s),
        tool_names=("move_to",),
    )
    runtime = OpenEtaAgentRuntime(
        tools=registry, rollout_enabled=False,
        planner=ToolCallingPlanner(StaticPlannerBackend({
            "kind": "tool_call", "name": "move_to", "parameters": request,
        })),
    )
    task = "Bounded controller receipt diagnostic; no benchmark task success claimed"
    runtime.start_session(task=task)
    runtime.memory.add_action(EnvAction(action_type="tool_call", command={
        "request": {"kind": "tool_call", "name": "ik_preview_check",
                    "parameters": preview_parameters},
        "status": "executed",
        "tool_calls": [{"name": "ik_preview_check", "status": "executed",
                        "parameters": preview_parameters,
                        "result": {"success": True, "details": {
                            "outputs": {"ik_preview_receipt": receipt},
                        }}}],
    }))
    budget = ToolCallBudget(1)
    action = runtime.act(EnvObservation(task=task, cameras=[], robot=RobotState()),
                         tool_call_budget=budget)
    if not dispatched or "response" not in dispatched[0]:
        raise RuntimeError(f"Runtime did not finish its one move: {action.command.get('status')}")
    return dispatched[0]["response"], dispatched[0]["seed"], {
        "planner": "StaticPlannerBackend",
        "scope": "host_preview_ingestion_then_production_agent_move_dispatch",
        "agent_session_id": runtime.memory.session_id,
        "request": request,
        "pipeline_status": action.command.get("status"),
        "remote_motion_calls": len(dispatched),
        "tool_admission": budget.snapshot(),
        "model_or_perception_requests": 0,
    }
