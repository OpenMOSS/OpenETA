"""Non-invasive wrappers for native UniVTAC reset and planner diagnostics."""

from __future__ import annotations

import copy
import json
import math
import time
import types
from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np


PLANNER_DIAGNOSTIC_SCHEMA_VERSION = "openeta.univtac.planner_diagnostic.v1"
KNOWN_RESULT_FIELDS = (
    "success",
    "status",
    "valid_query",
    "attempts",
    "ik_time",
    "graph_time",
    "trajopt_time",
    "solve_time",
)


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def safe_diagnostic_value(value: Any, *, small_limit: int = 32) -> Any:
    """Convert simulator values to strict JSON without assuming concrete types."""

    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, Mapping):
        return {
            str(key): safe_diagnostic_value(child, small_limit=small_limit)
            for key, child in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [safe_diagnostic_value(child, small_limit=small_limit) for child in value]

    detached = value
    if hasattr(detached, "detach"):
        try:
            detached = detached.detach()
        except Exception:
            pass
    if hasattr(detached, "cpu"):
        try:
            detached = detached.cpu()
        except Exception:
            pass
    if hasattr(detached, "numpy"):
        try:
            detached = detached.numpy()
        except Exception:
            pass
    try:
        array = np.asarray(detached)
    except Exception:
        array = None
    if array is not None and array.dtype != object:
        descriptor: dict[str, Any] = {
            "shape": [int(size) for size in array.shape],
            "dtype": str(array.dtype),
        }
        if array.size == 1:
            scalar = array.reshape(-1)[0].item()
            finite = _finite_float(scalar)
            descriptor["value"] = finite if finite is not None else repr(scalar)
        elif array.size <= small_limit:
            values = array.tolist()
            if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
                descriptor["values"] = safe_diagnostic_value(values)
            else:
                descriptor["values"] = values
        elif array.size and np.issubdtype(array.dtype, np.number):
            finite = array[np.isfinite(array)]
            if finite.size:
                descriptor["finite_range"] = [float(finite.min()), float(finite.max())]
        return descriptor

    if hasattr(value, "p") and hasattr(value, "q"):
        return {
            "position": safe_diagnostic_value(getattr(value, "p")),
            "quaternion": safe_diagnostic_value(getattr(value, "q")),
        }
    try:
        return {"repr": repr(value), "type": type(value).__name__}
    except Exception:
        return {"repr": "<unavailable>", "type": type(value).__name__}


def serialize_motion_gen_result(result: Any) -> dict[str, Any]:
    """Serialize known and optional MotionGenResult fields without requiring them."""

    payload: dict[str, Any] = {"result_type": type(result).__name__}
    captured: set[str] = set()
    for field_name in KNOWN_RESULT_FIELDS:
        if not hasattr(result, field_name):
            continue
        captured.add(field_name)
        try:
            payload[field_name] = safe_diagnostic_value(getattr(result, field_name))
        except Exception as exc:
            payload[field_name] = {
                "error_type": type(exc).__name__,
                "repr": "<unavailable>",
            }

    try:
        optional_values = vars(result)
    except TypeError:
        optional_values = {}
    other_fields: dict[str, Any] = {}
    for field_name, value in sorted(optional_values.items()):
        if field_name.startswith("_") or field_name in captured or callable(value):
            continue
        if isinstance(value, (bool, int, float, str, type(None))):
            other_fields[field_name] = safe_diagnostic_value(value)
    if other_fields:
        payload["other_scalar_fields"] = other_fields
    json.dumps(payload, sort_keys=True, allow_nan=False)
    return payload


def _pose_payload(pose: Any) -> dict[str, Any]:
    if pose is None:
        return {"position": None, "quaternion": None}
    if hasattr(pose, "p") and hasattr(pose, "q"):
        position = safe_diagnostic_value(getattr(pose, "p"))
        quaternion = safe_diagnostic_value(getattr(pose, "q"))
        return {"position": position, "quaternion": quaternion}
    serialized = safe_diagnostic_value(pose)
    return {"serialized": serialized}


def _action_payload(action: Any, action_index: int) -> dict[str, Any]:
    target_pose = getattr(action, "target_pose", None)
    return {
        "action_index": int(action_index),
        "action_repr": str(action),
        "action_type": str(getattr(action, "action", type(action).__name__)),
        "target_ee_pose": _pose_payload(target_pose),
        "args": safe_diagnostic_value(copy.deepcopy(getattr(action, "args", {}))),
    }


def motion_gen_success(result_payload: Mapping[str, Any]) -> bool | None:
    success = result_payload.get("success")
    if isinstance(success, bool):
        return success
    if isinstance(success, Mapping) and "value" in success:
        return bool(success["value"])
    return None


def motion_gen_status(result_payload: Mapping[str, Any]) -> str | None:
    status = result_payload.get("status")
    if isinstance(status, str):
        return status
    if isinstance(status, Mapping):
        if "value" in status:
            return str(status["value"])
        if "repr" in status:
            return str(status["repr"])
    return None


@dataclass
class PlannerDiagnosticRecorder:
    """Instance-scoped wrappers that preserve native planner behavior."""

    task: Any
    planning_calls: list[dict[str, Any]] = field(default_factory=list)
    move_calls: list[dict[str, Any]] = field(default_factory=list)
    plan_success_transitions: list[dict[str, Any]] = field(default_factory=list)
    current_seed: int | None = None
    _move_stack: list[dict[str, Any]] = field(default_factory=list, init=False)
    _installed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self._original_move = self.task.move
        self._planner = self.task._robot_manager.planner
        self._original_plan_path = self._planner.plan_path

    def install(self) -> None:
        if self._installed:
            return
        recorder = self

        def move_wrapper(task_self, actions, *args, **kwargs):
            actions_list = list(actions) if actions is not None else []
            move_index = len(recorder.move_calls) + 1
            move_record = {
                "move_call_index": move_index,
                "actions": [
                    _action_payload(action, index)
                    for index, action in enumerate(actions_list)
                ],
                "call_args": safe_diagnostic_value(args),
                "call_kwargs": safe_diagnostic_value(copy.deepcopy(kwargs)),
                "task_state_before": recorder._task_state(),
                "plan_success_before": bool(getattr(task_self, "plan_success", False)),
                "planning_call_indices": [],
            }
            recorder.move_calls.append(move_record)
            context = {
                "move_call_index": move_index,
                "actions": move_record["actions"],
                "next_action_cursor": 0,
                "planning_call_indices": move_record["planning_call_indices"],
            }
            recorder._move_stack.append(context)
            started = time.perf_counter()
            try:
                returned = recorder._original_move(actions_list, *args, **kwargs)
                move_record["returned"] = safe_diagnostic_value(returned)
                return returned
            except BaseException as exc:
                move_record["exception"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                raise
            finally:
                move_record["elapsed_seconds"] = time.perf_counter() - started
                move_record["plan_success_after"] = bool(
                    getattr(task_self, "plan_success", False)
                )
                move_record["task_state_after"] = recorder._task_state()
                if (
                    move_record["plan_success_before"] is True
                    and move_record["plan_success_after"] is False
                ):
                    transition = {
                        "location": "BaseTask.move",
                        "move_call_index": move_index,
                        "planning_call_index": (
                            move_record["planning_call_indices"][-1]
                            if move_record["planning_call_indices"]
                            else None
                        ),
                        "task_state": recorder._task_state(),
                    }
                    recorder.plan_success_transitions.append(transition)
                    if move_record["planning_call_indices"]:
                        plan_record = recorder.planning_calls[
                            move_record["planning_call_indices"][-1] - 1
                        ]
                        plan_record["task_plan_success_after_call"] = False
                        plan_record["plan_success_transition_location"] = "BaseTask.move"
                recorder._move_stack.pop()

        def plan_path_wrapper(
            planner_self,
            curr_joint_pos,
            curr_joint_vel,
            target_ee_pose,
            real_robot_pose,
            pre_dis=None,
            constraint_pose=None,
            time_dilation_factor=None,
        ):
            plan_index = len(recorder.planning_calls) + 1
            context = recorder._move_stack[-1] if recorder._move_stack else None
            action_payload = recorder._next_action(context)
            record = {
                "schema_version": PLANNER_DIAGNOSTIC_SCHEMA_VERSION,
                "seed": recorder.current_seed,
                "planning_call_index": plan_index,
                "enclosing_move_call_index": (
                    context["move_call_index"] if context is not None else None
                ),
                "action": action_payload,
                "action_repr": action_payload.get("action_repr") if action_payload else None,
                "action_type": action_payload.get("action_type") if action_payload else None,
                "target_ee_pose": _pose_payload(target_ee_pose),
                "current_ee_pose": recorder._current_ee_pose(),
                "current_arm_joint_position": safe_diagnostic_value(curr_joint_pos),
                "current_arm_joint_velocity": safe_diagnostic_value(curr_joint_vel),
                "pre_dis": safe_diagnostic_value(pre_dis),
                "constraint_pose": safe_diagnostic_value(constraint_pose),
                "time_dilation_factor": safe_diagnostic_value(time_dilation_factor),
                "task_step_count": int(getattr(recorder.task, "step_count", 0)),
                "task_atom_id": int(getattr(recorder.task, "atom_id", 0)),
                "task_atom_tag": str(getattr(recorder.task, "atom_tag", "")),
                "task_plan_success_before_call": bool(
                    getattr(recorder.task, "plan_success", False)
                ),
            }
            recorder.planning_calls.append(record)
            if context is not None:
                context["planning_call_indices"].append(plan_index)
            started = time.perf_counter()
            try:
                result = recorder._original_plan_path(
                    curr_joint_pos,
                    curr_joint_vel,
                    target_ee_pose,
                    real_robot_pose,
                    pre_dis=pre_dis,
                    constraint_pose=constraint_pose,
                    time_dilation_factor=time_dilation_factor,
                )
                result_payload = serialize_motion_gen_result(result)
                record["motion_gen_result"] = result_payload
                record["motion_gen_success"] = motion_gen_success(result_payload)
                record["motion_gen_status"] = motion_gen_status(result_payload)
                return result
            except BaseException as exc:
                record["exception"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
                raise
            finally:
                record["planning_elapsed_seconds"] = time.perf_counter() - started
                record["task_plan_success_after_raw_result"] = bool(
                    getattr(recorder.task, "plan_success", False)
                )
                record.setdefault(
                    "task_plan_success_after_call",
                    record["task_plan_success_after_raw_result"],
                )
                json.dumps(record, sort_keys=True, allow_nan=False)

        self.task.move = types.MethodType(move_wrapper, self.task)
        self._planner.plan_path = types.MethodType(plan_path_wrapper, self._planner)
        self._installed = True

    def start_seed(self, seed: int) -> None:
        self.current_seed = int(seed)
        self.planning_calls.clear()
        self.move_calls.clear()
        self.plan_success_transitions.clear()
        self._move_stack.clear()

    def _task_state(self) -> dict[str, Any]:
        return {
            "step_count": int(getattr(self.task, "step_count", 0)),
            "atom_id": int(getattr(self.task, "atom_id", 0)),
            "atom_tag": str(getattr(self.task, "atom_tag", "")),
            "take_action_count": int(getattr(self.task, "take_action_cnt", 0)),
            "plan_success": bool(getattr(self.task, "plan_success", False)),
        }

    def _current_ee_pose(self) -> dict[str, Any]:
        try:
            pose = self.task._robot_manager.get_ee_pose()
        except Exception as exc:
            return {"error_type": type(exc).__name__}
        return _pose_payload(pose)

    @staticmethod
    def _next_action(context: dict[str, Any] | None) -> dict[str, Any] | None:
        if context is None:
            return None
        actions = context["actions"]
        cursor = context["next_action_cursor"]
        while cursor < len(actions):
            action = actions[cursor]
            cursor += 1
            context["next_action_cursor"] = cursor
            if action.get("action_type") in {"move", "all"}:
                return action
        return None

    def planner_failure(self) -> dict[str, Any] | None:
        failed = [
            record
            for record in self.planning_calls
            if record.get("motion_gen_success") is False
            or record.get("task_plan_success_after_call") is False
            or "exception" in record
        ]
        if not failed:
            return None
        record = copy.deepcopy(failed[-1])
        return {
            "schema_version": PLANNER_DIAGNOSTIC_SCHEMA_VERSION,
            "seed": self.current_seed,
            "failed_move_call_index": record.get("enclosing_move_call_index"),
            "failed_planning_call_index": record.get("planning_call_index"),
            "failed_action": record.get("action_repr"),
            "failed_action_type": record.get("action_type"),
            "failed_target_ee_pose": record.get("target_ee_pose"),
            "robot_state_before_failure": {
                "current_ee_pose": record.get("current_ee_pose"),
                "current_arm_joint_position": record.get(
                    "current_arm_joint_position"
                ),
                "current_arm_joint_velocity": record.get(
                    "current_arm_joint_velocity"
                ),
                "task_step_count": record.get("task_step_count"),
                "task_atom_id": record.get("task_atom_id"),
                "task_atom_tag": record.get("task_atom_tag"),
            },
            "native_result_status": record.get("motion_gen_status"),
            "native_result": record.get("motion_gen_result"),
            "plan_success_before": record.get("task_plan_success_before_call"),
            "plan_success_after": record.get("task_plan_success_after_call"),
            "plan_success_transition_location": record.get(
                "plan_success_transition_location"
            ),
        }

    def to_json(self) -> str:
        payload = {
            "schema_version": PLANNER_DIAGNOSTIC_SCHEMA_VERSION,
            "seed": self.current_seed,
            "move_calls": self.move_calls,
            "planning_calls": self.planning_calls,
            "plan_success_transitions": self.plan_success_transitions,
        }
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
