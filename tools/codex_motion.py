"""Single-request motion preflight using ordinary, budgeted Host tool steps.

This experimental ingress composition does not change shared tool contracts.
It never calls a simulator directly or manufactures an IK authorization.
"""
from copy import deepcopy

from mcp.types import Tool
from tools.codex_feedback import motion_feedback


def motion_schema(original):
    schema = deepcopy(original.inputSchema)
    schema.pop("required", None)
    schema["oneOf"] = [{"required": ["bundle_id"]}, {"required": ["target_pose"]}]
    schema["properties"]["target_pose"] = {
        "type": "object", "description": "World-frame EEF target; checked by the Host before motion.",
        "properties": {"frame": {"const": "world"},
                       "xyz": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                       "quat_xyzw": {"type": "array", "items": {"type": "number"}, "minItems": 4, "maxItems": 4}},
        "required": ["frame", "xyz", "quat_xyzw"], "additionalProperties": False,
    }
    return Tool(name="move_to", inputSchema=schema, description=(
        "Move to a world target_pose or current target_pose/ik_result bundle_id. "
        "Host automatically runs fresh IK for the exact target and execution tolerances, "
        "then executes only an authorized result through existing motion/collision gates. "
        "No separate propose_motion_target or ik_preview_check call is needed. "
        "Preflight failure executes no motion. IK success does not guarantee controller convergence. "
        "One native request consumes up to three ordinary Host turns/tool calls."))


def _outputs(command, tool):
    calls = [c for c in command.get("tool_calls", []) if c.get("name") == tool]
    if len(calls) != 1 or calls[0].get("status") != "executed":
        raise ValueError(f"{tool} did not execute successfully")
    result = calls[0].get("result") or {}
    if result.get("success") is not True:
        raise ValueError(f"{tool} returned an unsuccessful result")
    return (result.get("details") or {}).get("outputs") or {}


def run_motion_hook(host, arguments, *, preflight_only=False):
    """Return (last command, error, hook receipt); caller owns the Host lock."""
    requested = {"kind": "tool_call", "name": "move_to", "parameters": deepcopy(arguments)}
    hook = {"schema_version": "openeta.codex_motion_hook.v1", "stages": [], "motion_dispatched": False}
    last = {}

    def stage(name, parameters):
        nonlocal last
        payload = {"kind": "tool_call", "name": name, "parameters": parameters}
        if name == "move_to":
            # A cancelled or lost response cannot prove that no motion occurred.
            hook["motion_dispatched"] = None
        last, error = host._execute(payload, parent_request=requested)
        if name == "move_to" and last:
            hook["motion_dispatched"] = any(
                c.get("name") == name and c.get("status") in {"executed", "failed"}
                for c in last.get("tool_calls", []))
        if name == "move_to":
            # Failed commands still carry authoritative zero/partial execution
            # receipts. Project them BEFORE raising on the command status.
            hook.update(motion_feedback(last))
        hook["stages"].append({"tool": name, "status": last.get("status"), "error": error})
        if error:
            raise ValueError(error.get("message") or f"{name}: {error.get('code', 'tool_execution_failed')}")
        if last.get("request", {}).get("name") != name:
            raise ValueError("Host invariant substituted another command; motion sequence stopped")
        return _outputs(last, name)

    def bundle(kind, predicate):
        matches = [b for b in host.runtime.memory.tool_handoffs(limit=32)
                   if b.get("kind") == kind and b.get("current_epoch") and predicate(b)]
        if len(matches) != 1:
            raise ValueError(f"Expected one current {kind} handoff from this preflight")
        return matches[0]["bundle_id"]

    try:
        memory = host.runtime.memory
        reference = None
        target_bundle = None
        inherited = {}
        if "target_pose" in arguments:
            reference = {"target_pose": deepcopy(arguments["target_pose"])}
        else:
            # Each resolver validates session, file integrity, type and epochs.
            # A target bundle is not interpreted as an execution authorization.
            try:
                native = memory.resolve_tool_bundle("move_to", {"bundle_id": arguments["bundle_id"]})
            except ValueError:
                native = memory.resolve_tool_bundle("ik_preview_check", {"bundle_id": arguments["bundle_id"]})
                inherited = native
                target_bundle = arguments["bundle_id"]
            else:
                resolved = memory.resolve_ik_motion_reference(native["ik_receipt_id"])["parameters"]
                inherited = {"position_tolerance_m": resolved.get("tolerance", .002),
                             "orientation_tolerance_rad": resolved.get("ori_tolerance", .05)}
                reference = {"target_pose": resolved["target_pose"],
                             "preserve_current_orientation": resolved.get("preserve_current_orientation", False)}
        # Do not check one tolerance and silently execute a stricter default.
        pos_tol = arguments.get("tolerance", inherited.get("position_tolerance_m", .002))
        ori_tol = arguments.get("ori_tolerance", inherited.get("orientation_tolerance_rad", .05))
        controls = {"position_tolerance_m": pos_tol, "orientation_tolerance_rad": ori_tol}
        if reference is not None:
            before = {b["bundle_id"] for b in memory.tool_handoffs(limit=32)}
            stage("propose_motion_target", {**reference, **controls})
            target_bundle = bundle("target_pose", lambda b: b["bundle_id"] not in before)
        before_preview = {b["bundle_id"] for b in memory.tool_handoffs(limit=32)}
        preview = stage("ik_preview_check", {"bundle_id": target_bundle, **controls})
        hook["ik_preview_receipt"] = preview.get("ik_preview_receipt")
        hook["execution_authorization"] = preview.get("execution_authorization")
        authorization = preview.get("execution_authorization") or {}
        if authorization.get("authorized_for_move_to") is not True:
            receipt = preview.get("ik_preview_receipt") or {}
            hook["reason_code"] = receipt.get("reason_code") or preview.get("reason_code") or "ik_preflight_not_authorized"
            raise ValueError("Fresh IK did not authorize execution; no motion was dispatched")
        rid = preview.get("ik_receipt_id")
        if not rid:
            raise ValueError("IK returned no receipt ID")
        execution_bundle = bundle("ik_result", lambda b: b["bundle_id"] not in before_preview
                                  and b.get("summary", {}).get("receipt_id") == rid)
        motion_args = {"bundle_id": execution_bundle, "tolerance": pos_tol, "ori_tolerance": ori_tol,
                       "enable_collision_check": arguments.get("enable_collision_check", True)}
        if "num_steps" in arguments:
            motion_args["num_steps"] = arguments["num_steps"]
        if preflight_only:
            # Private continuation: execute this exact checked bundle, never
            # re-solve IK after choosing between symmetric endpoints.
            hook["_prepared_move"] = motion_args
            return last, None, hook
        motion = stage("move_to", motion_args)
        hook["motion_summary"] = motion.get("motion_summary")
        hook["collision_coverage"] = motion.get("collision_coverage")
        hook["reason_code"] = (motion.get("motion_summary") or {}).get("stop_reason", "motion_returned")
        return last, None, hook
    except (ValueError, TypeError, KeyError, OSError) as exc:
        hook["failure_stage"] = hook["stages"][-1]["tool"] if hook["stages"] else "resolve_target"
        return last, {"code": hook.get("reason_code", "motion_hook_stopped"), "message": str(exc)}, hook


def run_symmetric_motion_hook(host, poses, current_rotation, *, bind_selected):
    """Two ordinary budgeted preflights, then at most one exact-ID dispatch.

    Endpoint-only experiment; per-step collision checks remain with Mink.
    Caller holds the Host lock and verifies empty/symmetric gripper semantics.
    """
    from tools.codex_atomic_geometry import quat_matrix
    from tools.codex_orientation import endpoint_score, rotation_distance

    if len(poses) != 2:
        raise ValueError("Symmetry selection requires exactly two targets")
    memory = host.runtime.memory
    epoch = (memory.object_scene_epoch(), memory.robot_motion_epoch())
    selection = {"mode": "parallel_jaw_symmetric", "path_check": "not_run",
                 "selection_status": "endpoint_only", "candidates": []}
    hook = {"stages": [], "motion_dispatched": False, "physics_executed": False,
            "orientation_selection": selection}
    viable = []
    last = {}
    for index, pose in enumerate(poses):
        last, error, checked = run_motion_hook(host, {"target_pose": pose}, preflight_only=True)
        hook["stages"].extend(checked["stages"])
        receipt = checked.get("ik_preview_receipt") or {}
        item = {"index": index, "ik_authorized": checked.get("execution_authorization", {}).get("authorized_for_move_to") is True,
                "reason_code": checked.get("reason_code") or receipt.get("reason_code")}
        selection["candidates"].append(item)
        if error:
            # Only an ordinary, completed negative IK result permits checking
            # the other target. Transport, invariants and exhausted budgets stop.
            if (checked.get("failure_stage") == "ik_preview_check" and receipt
                    and "_prepared_move" not in checked and not item["ik_authorized"]):
                continue
            hook.update({"failure_stage": checked.get("failure_stage"), "reason_code": error["code"]})
            return last, error, hook
        try:
            item.update(endpoint_score(receipt.get("best_candidate") or {},
                rotation_distance(current_rotation, quat_matrix(pose["quat_xyzw"]))))
        except ValueError as exc:
            item["reason_code"] = "orientation_metrics_unavailable"
            item["message"] = str(exc)
            continue
        viable.append((item["score"], index, checked))
    if not viable:
        hook.update(reason_code="no_authorized_orientation_candidate", failure_stage="orientation_selection")
        return last, {"code": hook["reason_code"], "message": "No candidate had both IK authorization and valid ranking metrics; no motion dispatched"}, hook
    if epoch != (memory.object_scene_epoch(), memory.robot_motion_epoch()):
        hook.update(reason_code="stale_orientation_preflight", failure_stage="orientation_selection")
        return last, {"code": hook["reason_code"], "message": "Scene or robot changed during orientation preflight; no motion dispatched"}, hook
    _, index, checked = min(viable, key=lambda value: (value[0], value[1]))
    selection.update(selected_index=index, symmetry_applied=bool(index),
                     selected_target=deepcopy(poses[index]), selection_reason="lowest_endpoint_joint_and_rotation_cost")
    hook["ik_preview_receipt"] = checked["ik_preview_receipt"]
    hook["execution_authorization"] = checked["execution_authorization"]
    bind_selected(poses[index])
    payload = {"kind": "tool_call", "name": "move_to", "parameters": checked["_prepared_move"]}
    hook["motion_dispatched"] = None
    hook["physics_executed"] = None
    last, error = host._execute(payload, parent_request={"kind": "tool_call", "name": "move_to",
        "parameters": {"target_pose": poses[0], "orientation_mode": "parallel_jaw_symmetric"}})
    if not last and error and error.get("code") in {"episode_not_active", "host_validation"}:
        # These Host verdicts return before runner.step(); unlike a transport
        # or execution exception they prove that no motion was dispatched.
        hook["motion_dispatched"] = False
        hook["physics_executed"] = False
    if last:
        hook["motion_dispatched"] = any(c.get("name") == "move_to" and c.get("status") in {"executed", "failed"}
                                         for c in last.get("tool_calls", []))
        if hook["motion_dispatched"] is False:
            hook["physics_executed"] = False
    hook.update(motion_feedback(last))
    hook["stages"].append({"tool": "move_to", "status": last.get("status"), "error": error})
    try:
        if error:
            raise ValueError(error.get("message") or error.get("code"))
        if last.get("request", {}).get("name") != "move_to":
            raise ValueError("Host invariant substituted another command; motion sequence stopped")
        motion = _outputs(last, "move_to")
        hook["motion_summary"] = motion.get("motion_summary")
        hook["reason_code"] = (motion.get("motion_summary") or {}).get("stop_reason", "motion_returned")
        return last, None, hook
    except (ValueError, TypeError, KeyError) as exc:
        hook["failure_stage"] = "move_to"
        return last, {"code": hook.get("reason_code", "motion_hook_stopped"), "message": str(exc)}, hook
