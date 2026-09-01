"""Post-hoc tool collaboration contract audit for durable OpenETA rollouts."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable

from adapter.protocol import JsonDict
from agent.runtime.memory import ik_pose_policy_numerically_equivalent


SOURCE_BOUND_PERCEPTION_TOOLS = {
    "sam3",
    "molmopoint",
    "estimate_depth_prior",
    "enhance_depth",
    "grasp_pose_estimate",
    "anygrasp",
    "graspgenx",
}
MOTION_TOOLS = {"move_to", "follow_eef_trajectory"}
MAX_PORTABLE_ATTACHMENT_PROBE_DISPLACEMENT_M = 0.07
NONTERMINAL_SEMANTIC_PREFIXES = (
    "no_",
    "requires_",
    "target_not_",
    "reference_service_",
    "ik_repairable",
    "ik_inconclusive",
)
GENERIC_RECOVERY_ACTIONS = {
    "",
    "inspect_diagnostics",
    "inspect_recovery_options",
    "retry_same_call",
}
REFERENCE_RESOLUTION_CODES = {
    "unknown_source_packet_id",
    "unknown_grasp_result_id",
    "unknown_candidate_id",
    "unknown_bundle_id",
    "invalid_bundle_reference",
    "artifact_reference_not_found",
}


def audit_tool_event_file(path: str | Path) -> JsonDict:
    """Audit one rollout file or a directory containing exactly one rollout."""

    source = _resolve_tool_event_source(Path(path))
    issues: list[JsonDict] = []
    calls: list[JsonDict] = []
    last_world_mutation_seq = -1
    seen_perception: dict[str, tuple[int, int]] = {}
    previous_read_only_signature = ""
    previous_read_only_seq = -1
    pending_viewpoint_proposal: JsonDict | None = None
    feasible_ik_pose_policies: dict[str, JsonDict] = {}
    deferred_collision_ik_pose_policies: dict[str, JsonDict] = {}
    executable_ik_receipts: dict[str, JsonDict] = {}
    inconclusive_ik_gaps: dict[str, JsonDict] = {}
    previous_zero_step_motion: JsonDict | None = None
    failed_motion_attractors: dict[str, JsonDict] = {}
    pending_close_probe: JsonDict | None = None
    successful_contact_executions: dict[str, list[JsonDict]] = {}
    for row in _read_jsonl(source):
        event = row.get("event")
        if not isinstance(event, dict) or event.get("phase") != "end":
            continue
        seq = int(row.get("seq") or 0)
        name = str(event.get("name") or "")
        effect = str(event.get("effect") or "")
        details = event.get("details")
        details = details if isinstance(details, dict) else {}
        outputs = details.get("outputs")
        outputs = outputs if isinstance(outputs, dict) else {}
        if isinstance(outputs.get("outputs"), dict):
            issues.append(
                _issue(
                    seq,
                    name,
                    "double_nested_tool_outputs",
                    "error",
                    evidence={
                        "note": (
                            "canonical ToolResult outputs were wrapped a second time; "
                            "adjacent consumers will not find producer fields at "
                            "details.outputs.<field>"
                        )
                    },
                )
            )
        parameters = event.get("parameters")
        parameters = parameters if isinstance(parameters, dict) else {}
        diagnostics = details.get("diagnostics")
        diagnostics = diagnostics if isinstance(diagnostics, list) else []
        output_diagnostics = outputs.get("diagnostics")
        output_diagnostics = (
            output_diagnostics if isinstance(output_diagnostics, list) else []
        )
        recovery = _combined_recovery_options(details, outputs)
        operational = details.get("operational_success")
        semantic = str(details.get("semantic_outcome") or "")
        actual_mutation = effect == "world_mutating" and _event_changed_world_state(
            name,
            outputs,
        )
        call = {
            "seq": seq,
            "name": name,
            "effect": effect,
            "success": event.get("success"),
            "operational_success": operational,
            "semantic_outcome": semantic,
        }
        calls.append(call)

        if (
            name == "sam3"
            and parameters.get("roi_bbox_xyxy") is not None
            and (
                str(parameters.get("mode") or "").lower() == "points"
                or parameters.get("points") is not None
                or parameters.get("positive_points") is not None
            )
        ):
            issues.append(
                _issue(
                    seq,
                    name,
                    "mutually_exclusive_perception_prompts",
                    "warning",
                    evidence={
                        "mode": parameters.get("mode"),
                        "has_points": parameters.get("points") is not None,
                        "has_positive_points": parameters.get("positive_points")
                        is not None,
                        "has_roi_bbox_xyxy": True,
                        "note": (
                            "SAM3 point mode consumes original-image coordinates and "
                            "must not be combined with ROI attention"
                        ),
                    },
                )
            )

        if name == "move_to" and event.get("success") is True:
            target_pose = parameters.get("target_pose")
            waypoint_role = (
                str(target_pose.get("waypoint_role") or "")
                if isinstance(target_pose, dict)
                else ""
            )
            motion = outputs.get("motion_summary")
            motion = motion if isinstance(motion, dict) else {}
            compiled_grasp_id = (
                str(target_pose.get("compiled_grasp_id") or "")
                if isinstance(target_pose, dict)
                else ""
            )
            if (
                waypoint_role == "grasp_contact"
                and compiled_grasp_id
                and motion.get("reached_target") is True
                and operational is not False
            ):
                successful_contact_executions.setdefault(
                    compiled_grasp_id, []
                ).append(
                    {
                        "seq": seq,
                        "timestamp_s": _finite_number(row.get("timestamp_s")),
                        "compiled_grasp_id": compiled_grasp_id,
                    }
                )
            if waypoint_role == "grasp_contact" and motion.get("reached_target") is True:
                max_axis_error = _finite_number(motion.get("max_axis_position_error_m"))
                orientation_error = _finite_number(motion.get("orientation_error_rad"))
                if (
                    max_axis_error is not None
                    and max_axis_error > 0.005 + 1e-9
                ) or (
                    orientation_error is not None
                    and orientation_error > 0.10 + 1e-9
                ):
                    issues.append(
                        _issue(
                            seq,
                            name,
                            "contact_execution_residual_exceeds_guidance",
                            "warning",
                            evidence={
                                "max_axis_position_error_m": max_axis_error,
                                "euclidean_position_error_m": motion.get(
                                    "position_error_m"
                                ),
                                "orientation_error_rad": orientation_error,
                                "recommended_max_axis_error_m": 0.005,
                                "recommended_orientation_error_rad": 0.10,
                                "note": (
                                    "controller reached_target uses its declared metric; "
                                    "object-relative contact still needs fresh wrist review"
                                ),
                            },
                        )
                    )

        if name == "gripper_control" and event.get("success") is True:
            position = _binary_gripper_position(parameters.get("position"))
            if position == 0:
                proxy_receipt = outputs.get("attachment_proxy_receipt")
                if not isinstance(proxy_receipt, dict):
                    issues.append(
                        _issue(
                            seq,
                            name,
                            "gripper_close_without_attachment_proxy_receipt",
                            "error",
                            evidence={
                                "note": (
                                    "the Agent cannot distinguish a host-bound tentative "
                                    "collision proxy from an empty close"
                                )
                            },
                        )
                    )
                    proxy_receipt = {}
                elif (
                    proxy_receipt.get("status") == "tentative"
                    and proxy_receipt.get("binding_source")
                    != "host_compiled_target_provenance"
                ):
                    issues.append(
                        _issue(
                            seq,
                            name,
                            "attachment_proxy_not_bound_to_compiled_target",
                            "error",
                            evidence={
                                "binding_source": proxy_receipt.get("binding_source"),
                                "target_object_name": proxy_receipt.get(
                                    "target_object_name"
                                ),
                            },
                        )
                    )
                pending_close_probe = {
                    "seq": seq,
                    "open_fraction": _measured_gripper_open_fraction(outputs),
                    "target_object_name": proxy_receipt.get("target_object_name"),
                    "proxy_status": proxy_receipt.get("status"),
                    "proxy_reason": proxy_receipt.get("reason"),
                }
            elif position == 1:
                pending_close_probe = None
        elif name in MOTION_TOOLS and isinstance(pending_close_probe, dict):
            motion = outputs.get("motion_summary")
            motion = motion if isinstance(motion, dict) else {}
            start = motion.get("start")
            end = motion.get("end")
            displacement = _xyz_distance(
                start.get("xyz") if isinstance(start, dict) else None,
                end.get("xyz") if isinstance(end, dict) else None,
            )
            post_open_fraction = _measured_gripper_open_fraction(outputs)
            close_open_fraction = _finite_number(
                pending_close_probe.get("open_fraction")
            )
            attachment_refresh = outputs.get("attachment_proxy_receipt")
            meaningful_probe = displacement != float("inf") and displacement >= 0.015
            if (
                meaningful_probe
                and displacement > MAX_PORTABLE_ATTACHMENT_PROBE_DISPLACEMENT_M
                and pending_close_probe.get("proxy_status") == "tentative"
            ):
                target_pose = parameters.get("target_pose")
                target_pose = target_pose if isinstance(target_pose, dict) else {}
                issues.append(
                    _issue(
                        seq,
                        name,
                        "oversized_attachment_probe_after_close",
                        "warning",
                        evidence={
                            "gripper_close_seq": pending_close_probe.get("seq"),
                            "target_object_name": pending_close_probe.get(
                                "target_object_name"
                            ),
                            "eef_displacement_m": displacement,
                            "recommended_probe_distance_range_m": [0.02, 0.05],
                            "waypoint_role": target_pose.get("waypoint_role"),
                            "interpretation": (
                                "the first motion after a tentative close was too large "
                                "to isolate attachment; use a fresh short probe from the "
                                "measured current EEF pose instead of an old approach anchor"
                            ),
                        },
                    )
                )
            if meaningful_probe and (
                pending_close_probe.get("proxy_status") in {"not_armed", "retired"}
                or (
                    close_open_fraction is not None
                    and close_open_fraction < 0.08
                )
            ):
                issues.append(
                    _issue(
                        seq,
                        name,
                        "lift_after_explicit_empty_close",
                        "warning",
                        evidence={
                            "gripper_close_seq": pending_close_probe.get("seq"),
                            "proxy_status": pending_close_probe.get("proxy_status"),
                            "proxy_reason": pending_close_probe.get("proxy_reason"),
                            "post_close_open_fraction": close_open_fraction,
                            "eef_displacement_m": displacement,
                            "note": (
                                "the close receipt already said no carried-object "
                                "proxy was active; reopen and repair contact before lift"
                            ),
                        },
                    )
                )
            expects_attachment_refresh = pending_close_probe.get(
                "proxy_status"
            ) not in {"not_armed", "retired"}
            if (
                meaningful_probe
                and expects_attachment_refresh
                and not isinstance(attachment_refresh, dict)
            ):
                issues.append(
                    _issue(
                        seq,
                        name,
                        "lift_probe_without_attachment_refresh_receipt",
                        "error",
                        evidence={
                            "gripper_close_seq": pending_close_probe.get("seq"),
                            "target_object_name": pending_close_probe.get(
                                "target_object_name"
                            ),
                            "eef_displacement_m": displacement,
                            "note": (
                                "the Agent cannot tell whether the conservative "
                                "carried-object proxy remains active or was retired"
                            ),
                        },
                    )
                )
            elif meaningful_probe and expects_attachment_refresh:
                refresh_target = str(
                    attachment_refresh.get("target_object_name") or ""
                )
                close_target = str(
                    pending_close_probe.get("target_object_name") or ""
                )
                if close_target and refresh_target != close_target:
                    issues.append(
                        _issue(
                            seq,
                            name,
                            "attachment_refresh_target_mismatch",
                            "error",
                            evidence={
                                "gripper_close_seq": pending_close_probe.get("seq"),
                                "close_target_object_name": close_target,
                                "refresh_target_object_name": refresh_target,
                                "eef_displacement_m": displacement,
                            },
                        )
                    )
                receipt_aperture = _finite_number(
                    attachment_refresh.get("measured_open_fraction")
                )
                if post_open_fraction is None:
                    post_open_fraction = receipt_aperture
            if (
                meaningful_probe
                and close_open_fraction is not None
                and close_open_fraction >= 0.08
                and post_open_fraction is not None
                and post_open_fraction < 0.08
            ):
                issues.append(
                    _issue(
                        seq,
                        name,
                        "probable_grasp_slip_after_lift_probe",
                        "warning",
                        evidence={
                            "gripper_close_seq": pending_close_probe.get("seq"),
                            "target_object_name": pending_close_probe.get(
                                "target_object_name"
                            ),
                            "eef_displacement_m": displacement,
                            "post_close_open_fraction": close_open_fraction,
                            "post_motion_open_fraction": post_open_fraction,
                            "interpretation": (
                                "the aperture collapsed toward empty-close during a "
                                "meaningful EEF motion; correlate with dual-view "
                                "co-motion/source-vacancy evidence"
                            ),
                        },
                    )
                )
            if meaningful_probe:
                pending_close_probe = None

        if event.get("success") is False and not diagnostics and not output_diagnostics:
            issues.append(_issue(seq, name, "failed_without_diagnostics", "error"))
        all_diagnostics = [*diagnostics, *output_diagnostics]
        if (
            (event.get("success") is False or operational is False)
            and not _is_cancelled_failure(all_diagnostics)
            and not _has_actionable_recovery(recovery)
        ):
            issues.append(
                _issue(
                    seq,
                    name,
                    "failure_without_actionable_recovery",
                    "error",
                    evidence={
                        "diagnostic_codes": _diagnostic_codes(all_diagnostics),
                        "offered_actions": _recovery_actions(recovery),
                    },
                )
            )
        unresolved_reference = _unresolved_reference_evidence(
            parameters,
            all_diagnostics,
            outputs,
        )
        if unresolved_reference:
            issues.append(
                _issue(
                    seq,
                    name,
                    "agent_visible_reference_unresolvable",
                    "error",
                    evidence=unresolved_reference,
                )
            )
        host_reference_failure = _host_reference_resolution_evidence(all_diagnostics)
        if host_reference_failure:
            issues.append(
                _issue(
                    seq,
                    name,
                    "host_reference_resolution_failure",
                    "error",
                    evidence=host_reference_failure,
                )
            )
        opaque_controller = _opaque_controller_failure_evidence(
            name,
            event,
            outputs,
            all_diagnostics,
        )
        if opaque_controller:
            issues.append(
                _issue(
                    seq,
                    name,
                    "opaque_controller_failure",
                    "error",
                    evidence=opaque_controller,
                )
            )
        if operational is None:
            issues.append(_issue(seq, name, "missing_operational_success", "error"))
        if semantic == "target_not_reached" and operational is not False:
            issues.append(
                _issue(
                    seq,
                    name,
                    "target_not_reached_marked_operational_success",
                    "error",
                )
            )
        if semantic.startswith(NONTERMINAL_SEMANTIC_PREFIXES) and not recovery:
            issues.append(_issue(seq, name, "semantic_failure_without_recovery", "error"))
        if effect == "world_mutating":
            if (
                name in MOTION_TOOLS
                and event.get("success") is True
                and operational is True
                and not actual_mutation
                and semantic != "target_already_within_tolerance"
            ):
                issues.append(
                    _issue(
                        seq,
                        name,
                        "motion_acknowledged_without_state_change",
                        "warning",
                        evidence=_motion_noop_evidence(outputs),
                    )
                )
        if actual_mutation:
            last_world_mutation_seq = seq
            inconclusive_ik_gaps.clear()
            previous_read_only_signature = ""
            previous_read_only_seq = -1
        if name in MOTION_TOOLS:
            referenced_receipts = [
                executable_ik_receipts.get(receipt_id)
                for receipt_id in _motion_ik_receipt_ids(name, parameters)
            ]
            motion = outputs.get("motion_summary")
            motion = motion if isinstance(motion, dict) else {}
            motion_failed = (
                event.get("success") is False
                or operational is False
                or motion.get("reached_target") is False
            )
            if motion_failed:
                for referenced in referenced_receipts:
                    risk = _execution_fragile_ik_evidence(referenced)
                    if not risk:
                        continue
                    issues.append(
                        _issue(
                            seq,
                            name,
                            "failed_motion_after_execution_fragile_ik",
                            "warning",
                            evidence=risk,
                        )
                    )
        if effect == "world_mutating":
            if event.get("success") is True and not isinstance(
                details.get("environment_receipt"), dict
            ):
                issues.append(_issue(seq, name, "mutation_without_environment_receipt", "error"))
        if name in MOTION_TOOLS and event.get("success") is True:
            typed_handoff_error = _typed_motion_handoff_error(
                name,
                parameters,
                executable_ik_receipts,
            )
            if typed_handoff_error:
                issues.append(
                    _issue(
                        seq,
                        name,
                        "motion_without_exact_typed_ik_handoff",
                        "error",
                        evidence=typed_handoff_error,
                    )
                )
            if not isinstance(outputs.get("pose_feedback"), dict):
                issues.append(_issue(seq, name, "motion_without_pose_feedback", "error"))
            controller_receipt = _controller_execution_receipt(outputs)
            if controller_receipt is None:
                issues.append(
                    _issue(
                        seq,
                        name,
                        "motion_without_controller_execution_receipt",
                        "error",
                    )
                )
            else:
                controller_error = _controller_receipt_error(
                    controller_receipt,
                    outputs,
                )
                if controller_error:
                    issues.append(
                        _issue(
                            seq,
                            name,
                            "controller_execution_receipt_inconsistent",
                            "error",
                            evidence=controller_error,
                        )
                    )
            coverage = outputs.get("collision_coverage")
            if not isinstance(coverage, dict):
                issues.append(_issue(seq, name, "motion_without_collision_coverage", "error"))
            elif (
                coverage.get("coverage_complete") is not True
                and coverage.get("collision_detected") is not True
            ):
                issues.append(
                    _issue(
                        seq,
                        name,
                        "collision_coverage_incomplete",
                        "warning",
                        evidence={
                            "coverage_status": coverage.get("coverage_status"),
                            "trajectory_checked": coverage.get("trajectory_checked"),
                            "world_checked": coverage.get("world_checked"),
                        },
                    )
                )
            if name == "move_to":
                pose_policy = _ik_pose_policy_signature(parameters)
                deferred = deferred_collision_ik_pose_policies.get(pose_policy)
                if not isinstance(deferred, dict):
                    deferred = _matching_ik_pose_policy(
                        deferred_collision_ik_pose_policies.values(),
                        parameters,
                    )
                feasible = feasible_ik_pose_policies.get(pose_policy)
                if not isinstance(feasible, dict):
                    feasible = _matching_ik_pose_policy(
                        feasible_ik_pose_policies.values(),
                        parameters,
                    )
                delegated_collision_is_proven = (
                    isinstance(deferred, dict)
                    and parameters.get("enable_collision_check") is True
                    and isinstance(coverage, dict)
                    and coverage.get("coverage_complete") is True
                )
                if (
                    pose_policy
                    and not isinstance(feasible, dict)
                    and not delegated_collision_is_proven
                ):
                    issues.append(
                        _issue(
                            seq,
                            name,
                            "motion_without_matching_feasible_ik_receipt",
                            "error",
                            evidence={
                                "pose_policy_signature": pose_policy,
                                "note": (
                                    "a move requires either a feasible IK preview, or "
                                    "a kinematically-feasible/collision-deferred receipt "
                                    "consumed with enable_collision_check=true and complete "
                                    "trajectory/world collision coverage, for the exact xyz "
                                    "and orientation policy after the latest mutation"
                                ),
                                **(
                                    {
                                        "deferred_ik_seq": deferred.get("seq"),
                                        "enable_collision_check": parameters.get(
                                            "enable_collision_check"
                                        ),
                                        "motion_collision_coverage_complete": (
                                            coverage.get("coverage_complete")
                                            if isinstance(coverage, dict)
                                            else None
                                        ),
                                    }
                                    if isinstance(deferred, dict)
                                    else {}
                                ),
                            },
                        )
                    )
                deadlock = _zero_step_motion_evidence(outputs)
                if deadlock is not None:
                    if (
                        isinstance(previous_zero_step_motion, dict)
                        and previous_zero_step_motion.get("actual_xyz")
                        == deadlock.get("actual_xyz")
                    ):
                        issues.append(
                            _issue(
                                seq,
                                name,
                                "repeated_zero_step_motion_deadlock",
                                "error",
                                evidence={
                                    "previous_seq": previous_zero_step_motion.get("seq"),
                                    **deadlock,
                                },
                            )
                        )
                    previous_zero_step_motion = {"seq": seq, **deadlock}
                else:
                    previous_zero_step_motion = None
                failed_motion = _failed_motion_attractor_evidence(outputs)
                if pose_policy and failed_motion is not None:
                    previous = failed_motion_attractors.get(pose_policy)
                    if (
                        isinstance(previous, dict)
                        and _xyz_distance(
                            previous.get("actual_xyz"), failed_motion.get("actual_xyz")
                        )
                        <= 0.005
                    ):
                        issues.append(
                            _issue(
                                seq,
                                name,
                                "repeated_failed_motion_attractor",
                                "error",
                                evidence={
                                    "previous_seq": previous.get("seq"),
                                    "pose_policy_signature": pose_policy,
                                    **failed_motion,
                                },
                            )
                        )
                    failed_motion_attractors[pose_policy] = {
                        "seq": seq,
                        **failed_motion,
                    }
            if actual_mutation:
                feasible_ik_pose_policies.clear()
                deferred_collision_ik_pose_policies.clear()
                executable_ik_receipts.clear()
        elif actual_mutation:
            # Gripper and other mutations advance the robot-state epoch just as
            # motion does, so an earlier endpoint preview cannot authorize a
            # later move through that state change.
            feasible_ik_pose_policies.clear()
            deferred_collision_ik_pose_policies.clear()
            executable_ik_receipts.clear()
        if name == "ik_preview_check" and not isinstance(
            outputs.get("collision_coverage"), dict
        ):
            issues.append(_issue(seq, name, "ik_without_collision_coverage", "warning"))
        if name == "ik_preview_check":
            receipt = outputs.get("ik_preview_receipt")
            classification = (
                str(receipt.get("classification") or "")
                if isinstance(receipt, dict)
                else ""
            )
            reason_code = (
                str(receipt.get("reason_code") or "")
                if isinstance(receipt, dict)
                else ""
            )
            if (
                classification == "inconclusive"
                and reason_code == "endpoint_collision_check_unavailable"
                and isinstance(receipt.get("best_candidate"), dict)
            ):
                # Normalize traces produced before the dedicated classification
                # existed.  The old receipt already carried the same two facts:
                # valid joint solution plus unavailable endpoint collision proof.
                classification = "kinematically_feasible_collision_deferred"
            if not classification and semantic == "ik_feasible":
                classification = "feasible"
            response = outputs.get("response")
            response = response if isinstance(response, dict) else {}
            execution_authorization = outputs.get("execution_authorization")
            if not isinstance(execution_authorization, dict):
                execution_authorization = response.get("execution_authorization")
            motion_execution_ref = outputs.get("motion_execution_ref")
            if not isinstance(motion_execution_ref, dict):
                motion_execution_ref = response.get("motion_execution_ref")
            event_content = str(event.get("content") or "")
            delegation = (
                receipt.get("motion_collision_delegation")
                if isinstance(receipt, dict)
                else None
            )
            delegated_collision_authority = bool(
                classification == "kinematically_feasible_collision_deferred"
                and isinstance(delegation, dict)
                and delegation.get("available_for_matching_move") is True
            )
            executable_classification = (
                classification == "feasible" or delegated_collision_authority
            )
            lower_content = event_content.lower()
            if not executable_classification and (
                isinstance(motion_execution_ref, dict)
                or (
                    isinstance(execution_authorization, dict)
                    and execution_authorization.get("authorized_for_move_to") is True
                )
                or "execution reference:" in lower_content
                or "pass only this id to move_to" in lower_content
            ):
                issues.append(
                    _issue(
                        seq,
                        name,
                        "nonexecutable_ik_exposes_motion_reference",
                        "error",
                        evidence={
                            "classification": classification,
                            "reason_code": reason_code,
                            "has_motion_execution_ref": isinstance(
                                motion_execution_ref, dict
                            ),
                            "authorization_value": (
                                execution_authorization.get(
                                    "authorized_for_move_to"
                                )
                                if isinstance(execution_authorization, dict)
                                else None
                            ),
                            "content_exposes_execution_reference": (
                                "execution reference:" in lower_content
                                or "pass only this id to move_to" in lower_content
                            ),
                            "verified_motion_collision_delegation": (
                                delegated_collision_authority
                            ),
                            "note": (
                                "a rejected IK receipt is durable evidence but cannot be "
                                "presented as a move_to authorization for the same pose"
                            ),
                        },
                    )
                )
            pose_policy = _ik_pose_policy_signature(parameters)
            if classification == "feasible" and pose_policy:
                deferred = deferred_collision_ik_pose_policies.get(pose_policy)
                if (
                    isinstance(deferred, dict)
                    and parameters.get("check_endpoint_collision") is False
                ):
                    issues.append(
                        _issue(
                            seq,
                            name,
                            "redundant_kinematics_only_ik_after_deferred_solution",
                            "warning",
                            evidence={
                                "previous_seq": deferred.get("seq"),
                                "pose_policy_signature": pose_policy,
                                "reason_code": deferred.get("reason_code"),
                                "note": (
                                    "the earlier IK already proved kinematic feasibility; "
                                    "the exact no-collision repeat adds no evidence when a "
                                    "verified collision-owning motion controller is available"
                                ),
                            },
                        )
                    )
                feasible_ik_pose_policies[pose_policy] = {
                    "seq": seq,
                    "receipt": dict(receipt) if isinstance(receipt, dict) else {},
                }
            if (
                classification == "kinematically_feasible_collision_deferred"
                and pose_policy
            ):
                deferred_collision_ik_pose_policies[pose_policy] = {
                    "seq": seq,
                    "reason_code": (
                        receipt.get("reason_code")
                        if isinstance(receipt, dict)
                        else None
                    ),
                    "receipt": dict(receipt) if isinstance(receipt, dict) else {},
                }
            receipt_id = (
                str(receipt.get("receipt_id") or "")
                if isinstance(receipt, dict)
                else ""
            )
            if receipt_id and classification in {
                "feasible",
                "kinematically_feasible_collision_deferred",
            }:
                executable_ik_receipts[receipt_id] = {
                    **(dict(receipt) if isinstance(receipt, dict) else {}),
                    "target_pose": (
                        dict(receipt["target_pose"])
                        if isinstance(receipt, dict)
                        and isinstance(receipt.get("target_pose"), dict)
                        else dict(parameters.get("target_pose") or {})
                    ),
                    "orientation_policy": (
                        receipt.get("orientation_policy")
                        if isinstance(receipt, dict)
                        and receipt.get("orientation_policy")
                        else (
                            "explicit_orientation"
                            if isinstance(parameters.get("target_pose"), dict)
                            and any(
                                parameters["target_pose"].get(key) is not None
                                for key in (
                                    "rotation_matrix",
                                    "quat_xyzw",
                                    "quaternion",
                                    "euler_xyz_deg",
                                    "rotvec",
                                )
                            )
                            else "preserve_current"
                        )
                    ),
                    "seq": seq,
                }
            if classification == "inconclusive" and pose_policy:
                gap_key = f"{pose_policy}:{reason_code}"
                previous_gap = inconclusive_ik_gaps.get(gap_key)
                if isinstance(previous_gap, dict):
                    issues.append(
                        _issue(
                            seq,
                            name,
                            "repeated_unresolved_ik_capability_gap",
                            "error",
                            evidence={
                                "previous_seq": previous_gap.get("seq"),
                                "pose_policy_signature": pose_policy,
                                "reason_code": reason_code,
                                "note": (
                                    "read-only calls and fresh packet ids cannot repair "
                                    "an unchanged backend capability gap"
                                ),
                            },
                        )
                    )
                inconclusive_ik_gaps[gap_key] = {"seq": seq}
        if name == "compile_grasp_seed" and not parameters.get("grasp_result_id"):
            issues.append(
                _issue(
                    seq,
                    name,
                    "compile_grasp_missing_short_id_resolution",
                    "error",
                    evidence={
                        "note": (
                            "planner-visible camera candidate/calibration copying should be "
                            "replaced by grasp_result_id + candidate_id host resolution"
                        )
                    },
                )
            )
        if name == "anyplace" and event.get("success") is True:
            result_id = str(outputs.get("result_id") or "")
            handoff = outputs.get("camera_pose_to_world_handoff")
            source_packet_id = str(outputs.get("source_packet_id") or "")
            camera_frame_id = str(outputs.get("camera_frame_id") or "")
            handoff_valid = (
                isinstance(handoff, dict)
                and handoff.get("tool") == "camera_pose_to_world"
                and handoff.get("placement_result_id") == result_id
                and handoff.get("required_parameters")
                == ["placement_result_id", "candidate_id"]
            )
            if not result_id or not source_packet_id or not camera_frame_id or not handoff_valid:
                issues.append(
                    _issue(
                        seq,
                        name,
                        "anyplace_missing_executable_placement_handoff",
                        "error",
                        evidence={
                            "has_result_id": bool(result_id),
                            "has_source_packet_id": bool(source_packet_id),
                            "has_camera_frame_id": bool(camera_frame_id),
                            "handoff_valid": handoff_valid,
                            "note": (
                                "AnyPlace must expose short placement identities and "
                                "host-resolvable source calibration; the Agent should not "
                                "parse artifacts or copy camera matrices"
                            ),
                        },
                    )
                )
        if name == "camera_pose_to_world":
            camera_pose = parameters.get("camera_pose")
            looks_like_anyplace_pose = isinstance(camera_pose, dict) and (
                str(camera_pose.get("id") or "").startswith("place_grasp_")
                or bool(camera_pose.get("source_grasp_id"))
            )
            placement_result_id = str(parameters.get("placement_result_id") or "")
            candidate_id = str(parameters.get("candidate_id") or "")
            if looks_like_anyplace_pose and (not placement_result_id or not candidate_id):
                issues.append(
                    _issue(
                        seq,
                        name,
                        "anyplace_pose_transformed_without_short_id_resolution",
                        "error",
                        evidence={
                            "camera_pose_id": camera_pose.get("id"),
                            "source_grasp_id": camera_pose.get("source_grasp_id"),
                            "note": (
                                "use placement_result_id + candidate_id so the host binds "
                                "the exact source observation calibration"
                            ),
                        },
                    )
                )
            if placement_result_id and candidate_id and event.get("success") is True:
                if (
                    outputs.get("placement_result_id") != placement_result_id
                    or outputs.get("candidate_id") != candidate_id
                ):
                    issues.append(
                        _issue(
                            seq,
                            name,
                            "placement_transform_identity_not_preserved",
                            "error",
                            evidence={
                                "requested_placement_result_id": placement_result_id,
                                "requested_candidate_id": candidate_id,
                                "returned_placement_result_id": outputs.get(
                                    "placement_result_id"
                                ),
                                "returned_candidate_id": outputs.get("candidate_id"),
                            },
                        )
                    )
                placement_reference = outputs.get("placement_reference")
                if not (
                    isinstance(placement_reference, dict)
                    and placement_reference.get("schema_version")
                    == "openeta.placement_world_reference.v1"
                    and placement_reference.get("execution_authorized") is False
                    and placement_reference.get("placement_result_id")
                    == placement_result_id
                    and placement_reference.get("candidate_id") == candidate_id
                ):
                    issues.append(
                        _issue(
                            seq,
                            name,
                            "placement_transform_missing_execution_semantics",
                            "error",
                            evidence={
                                "placement_result_id": placement_result_id,
                                "candidate_id": candidate_id,
                                "note": (
                                    "a transformed AnyPlace pose must be labelled as a "
                                    "non-authorizing low release reference so it is not "
                                    "mistaken for a one-step carry command"
                                ),
                            },
                        )
                    )
        if name == "propose_wrist_viewpoints" and event.get("success") is True:
            pending_viewpoint_proposal = outputs
        elif name == "ik_preview_check" and pending_viewpoint_proposal is not None:
            if not _ik_consumes_exact_viewpoint(parameters, pending_viewpoint_proposal):
                issues.append(
                    _issue(
                        seq,
                        name,
                        "viewpoint_ik_without_exact_full_pose_binding",
                        "error",
                        evidence={
                            "proposal_id": pending_viewpoint_proposal.get("proposal_id"),
                            "note": (
                                "use viewpoint_proposal_id + candidate_id host resolution, "
                                "or pass the exact returned xyz and orientation"
                            ),
                        },
                    )
                )
            pending_viewpoint_proposal = None
        elif pending_viewpoint_proposal is not None:
            # Attribute a viewpoint proposal only to the immediately following
            # IK call.  Carrying it across unrelated perception or grasp
            # compilation makes a later, unrelated IK check look malformed.
            pending_viewpoint_proposal = None
        elif actual_mutation:
            pending_viewpoint_proposal = None
        if effect == "read_only" and event.get("success") is True:
            request_signature = _semantic_request_signature(name, event)
            if request_signature == previous_read_only_signature:
                issues.append(
                    _issue(
                        seq,
                        name,
                        "repeated_equivalent_read_only_call",
                        "warning",
                        evidence={
                            "previous_seq": previous_read_only_seq,
                            "semantic_signature": request_signature,
                            "note": "source packet provenance is ignored in this signature",
                        },
                    )
                )
            previous_read_only_signature = request_signature
            previous_read_only_seq = seq
        elif actual_mutation:
            previous_read_only_signature = ""
            previous_read_only_seq = -1
        if name in SOURCE_BOUND_PERCEPTION_TOOLS and event.get("success") is True:
            source_packet = outputs.get("source_packet_id")
            source_observation = outputs.get("source_observation")
            source_packets = outputs.get("source_packet_ids")
            source_observations = outputs.get("source_observations")
            if (
                not source_packet
                and not isinstance(source_observation, dict)
                and not (isinstance(source_packets, list) and source_packets)
                and not (isinstance(source_observations, list) and source_observations)
            ):
                issues.append(_issue(seq, name, "perception_without_source_packet", "error"))
            signature = _perception_signature(name, event, outputs)
            if signature:
                previous = seen_perception.get(signature)
                if previous is not None and previous[1] == last_world_mutation_seq:
                    issues.append(
                        _issue(
                            seq,
                            name,
                            "repeated_perception_on_unchanged_view",
                            "warning",
                            evidence={"previous_seq": previous[0], "signature": signature},
                        )
                    )
                seen_perception[signature] = (seq, last_world_mutation_seq)

    trace_source = _resolve_sibling_trace_source(source)
    trace_audit = _audit_pipeline_trace(
        trace_source,
        successful_contact_executions=successful_contact_executions,
    )
    issues.extend(trace_audit["issues"])
    counts = Counter(str(issue["code"]) for issue in issues)
    severities = Counter(str(issue["severity"]) for issue in issues)
    error_count = int(severities.get("error", 0))
    quality_checks = {
        "failures_have_diagnostics": counts.get("failed_without_diagnostics", 0) == 0,
        "failures_have_actionable_recovery": (
            counts.get("failure_without_actionable_recovery", 0) == 0
        ),
        "agent_references_resolve": (
            counts.get("agent_visible_reference_unresolvable", 0) == 0
            and counts.get("host_reference_resolution_failure", 0) == 0
        ),
        "mutations_have_environment_receipts": (
            counts.get("mutation_without_environment_receipt", 0) == 0
        ),
        "motions_have_pose_and_controller_receipts": (
            counts.get("motion_without_pose_feedback", 0) == 0
            and counts.get("motion_without_controller_execution_receipt", 0) == 0
        ),
        "motion_collision_coverage_is_declared": (
            counts.get("motion_without_collision_coverage", 0) == 0
        ),
        "motions_use_exact_typed_ik_handoffs": (
            counts.get("motion_without_exact_typed_ik_handoff", 0) == 0
        ),
        "failed_motions_avoid_execution_fragile_ik": (
            counts.get("failed_motion_after_execution_fragile_ik", 0) == 0
        ),
        "ik_receipts_expose_consistent_execution_authority": (
            counts.get("nonexecutable_ik_exposes_motion_reference", 0) == 0
        ),
        "source_bound_perception_has_packet_provenance": (
            counts.get("perception_without_source_packet", 0) == 0
        ),
        "tool_outputs_use_canonical_single_envelope": (
            counts.get("double_nested_tool_outputs", 0) == 0
        ),
        "placement_uses_executable_short_id_handoff": (
            counts.get("anyplace_missing_executable_placement_handoff", 0) == 0
            and counts.get("anyplace_pose_transformed_without_short_id_resolution", 0)
            == 0
            and counts.get("placement_transform_identity_not_preserved", 0) == 0
        ),
        "pipeline_gate_blocks_are_causally_consistent": (
            counts.get("blocked_pipeline_without_repair_code", 0) == 0
            and counts.get("blocked_pipeline_without_actionable_repair", 0) == 0
            and counts.get("gate_rejected_close_after_successful_contact", 0) == 0
        ),
    }
    return {
        "schema_version": "openeta.tool_contract_audit.v1",
        "source": str(source),
        "trace_source": str(trace_source) if trace_source is not None else None,
        "tool_call_count": len(calls),
        "pipeline_blocked_count": trace_audit["blocked_count"],
        "issue_count": len(issues),
        "contract_pass": error_count == 0,
        "error_count": error_count,
        "severity_counts": dict(sorted(severities.items())),
        "issue_code_counts": dict(sorted(counts.items())),
        "quality_checks": quality_checks,
        "issues": issues,
        "calls": calls,
    }


def _perception_signature(name: str, event: JsonDict, outputs: JsonDict) -> str:
    """Describe perception intent independently of minted packet handles.

    ``last_world_mutation_seq`` already scopes the comparison to one physical
    world state.  Including ``source_packet_id`` here made a no-op observation
    refresh look like new visual evidence and hid exactly the packet-churn loops
    this audit is meant to expose.
    """

    parameters = event.get("parameters")
    parameters = parameters if isinstance(parameters, dict) else {}
    packet = parameters.get("source_packet_id") or outputs.get("source_packet_id")
    if not packet and isinstance(parameters.get("sources"), list):
        packet = parameters.get("sources")
    if not packet and isinstance(outputs.get("source_packet_ids"), list):
        packet = outputs.get("source_packet_ids")
    if not packet:
        return ""
    stable = {
        "name": name,
        "camera_frame_id": parameters.get("camera_frame_id"),
        "prompt": parameters.get("prompt"),
        "mode": parameters.get("mode") or ("text" if name == "sam3" else None),
        "bundle_id": parameters.get("bundle_id"),
    }
    return json.dumps(stable, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _combined_recovery_options(details: JsonDict, outputs: JsonDict) -> list[JsonDict]:
    combined: list[JsonDict] = []
    for container in (details, outputs):
        value = container.get("recovery_options")
        if not isinstance(value, list):
            continue
        combined.extend(item for item in value if isinstance(item, dict))
    return combined


def _recovery_actions(recovery: list[JsonDict]) -> list[str]:
    return [str(item.get("action") or "") for item in recovery]


def _has_actionable_recovery(recovery: list[JsonDict]) -> bool:
    return any(
        str(item.get("action") or "").strip() not in GENERIC_RECOVERY_ACTIONS
        for item in recovery
    )


def _diagnostic_codes(diagnostics: list[Any]) -> list[str]:
    codes: list[str] = []
    for item in diagnostics:
        code = item.get("code") if isinstance(item, dict) else ""
        if code:
            codes.append(str(code))
    return codes


def _is_cancelled_failure(diagnostics: list[Any]) -> bool:
    return "execution_cancelled" in _diagnostic_codes(diagnostics)


def _unresolved_reference_evidence(
    parameters: JsonDict,
    diagnostics: list[Any],
    outputs: JsonDict,
) -> JsonDict:
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, dict):
            continue
        code = str(diagnostic.get("code") or "")
        if code not in REFERENCE_RESOLUTION_CODES:
            continue
        requested = (
            diagnostic.get("source_packet_id")
            or diagnostic.get("reference_id")
            or parameters.get("source_packet_id")
            or parameters.get("grasp_result_id")
            or parameters.get("bundle_id")
        )
        recent = diagnostic.get("recent_source_packets")
        valid_ids = []
        if isinstance(recent, list):
            valid_ids = [
                str(item.get("source_packet_id"))
                for item in recent
                if isinstance(item, dict) and item.get("source_packet_id")
            ][-5:]
        return {
            "diagnostic_code": code,
            "requested_reference": requested,
            **({"recent_valid_references": valid_ids} if valid_ids else {}),
            "reason": outputs.get("reason") or diagnostic.get("message"),
        }
    return {}


def _host_reference_resolution_evidence(diagnostics: list[Any]) -> JsonDict:
    reference_phrases = (
        "active grasp provenance",
        "unknown reference",
        "reference not found",
        "does not exist",
        "cannot resolve",
        "failed to resolve",
        "expired reference",
    )
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, dict):
            continue
        code = str(diagnostic.get("code") or "")
        message = str(diagnostic.get("message") or "")
        lowered = message.lower()
        if code == "simulator_mcp_argument_error" and any(
            phrase in lowered for phrase in reference_phrases
        ):
            return {"diagnostic_code": code, "message": message}
    return {}


def _opaque_controller_failure_evidence(
    name: str,
    event: JsonDict,
    outputs: JsonDict,
    diagnostics: list[Any],
) -> JsonDict:
    if name not in MOTION_TOOLS:
        return {}
    motion = outputs.get("motion_summary")
    motion = motion if isinstance(motion, dict) else {}
    content = str(event.get("content") or "")
    blank_assertion = "assertionerror" in content.lower() and not content.split(
        "AssertionError:", 1
    )[-1].strip()
    controller_error = motion.get("stop_reason") == "controller_error"
    named_qp_failure = "mink_qp_no_solution" in content.lower()
    structured_failure = motion.get("controller_failure")
    if isinstance(structured_failure, dict) and structured_failure.get("code"):
        return {}
    if not (blank_assertion or controller_error or named_qp_failure):
        return {}
    concrete = []
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, dict):
            continue
        message = str(diagnostic.get("message") or "").strip()
        error_type = str(diagnostic.get("error_type") or "").strip()
        if error_type or (message and message != "Simulator motion did not reach the requested target."):
            concrete.append(
                {
                    "code": diagnostic.get("code"),
                    "error_type": error_type,
                    "message": message,
                }
            )
    if concrete and not blank_assertion:
        return {}
    return {
        "stop_reason": motion.get("stop_reason"),
        "steps_executed": motion.get("steps_executed"),
        "content": content[:300],
        "diagnostic_codes": _diagnostic_codes(diagnostics),
    }


def _event_changed_world_state(name: str, outputs: JsonDict) -> bool:
    if name not in MOTION_TOOLS:
        return True
    motion = outputs.get("motion_summary")
    if not isinstance(motion, dict):
        response = outputs.get("response")
        motion = response.get("motion_summary") if isinstance(response, dict) else None
    if not isinstance(motion, dict):
        return True
    steps = motion.get("steps_executed")
    if not isinstance(steps, int | float) or isinstance(steps, bool):
        return True
    if float(steps) > 0:
        return True
    start = motion.get("start")
    end = motion.get("end")
    start_xyz = start.get("xyz") if isinstance(start, dict) else None
    end_xyz = end.get("xyz") if isinstance(end, dict) else None
    return _xyz_distance(start_xyz, end_xyz) > 1e-6 and _xyz_distance(
        start_xyz, end_xyz
    ) != float("inf")


def _motion_noop_evidence(outputs: JsonDict) -> JsonDict:
    motion = outputs.get("motion_summary")
    motion = motion if isinstance(motion, dict) else {}
    return {
        "steps_executed": motion.get("steps_executed"),
        "stop_reason": motion.get("stop_reason"),
        "position_error_m": motion.get("position_error_m"),
        "note": (
            "a successful zero-step motion must not advance robot freshness; avoid "
            "reissuing it unless another receipt is genuinely required"
        ),
    }


def _semantic_request_signature(name: str, event: JsonDict) -> str:
    parameters = event.get("parameters")
    parameters = parameters if isinstance(parameters, dict) else {}
    stable = {
        "name": name,
        "parameters": _drop_packet_provenance(parameters),
    }
    canonical = json.dumps(
        stable,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(canonical.encode("utf-8")).hexdigest()[:20]


def _drop_packet_provenance(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _drop_packet_provenance(item)
            for key, item in value.items()
            if str(key)
            not in {
                "source_packet_id",
                "source_packet_ids",
                "source_observation",
                "source_observations",
            }
        }
    if isinstance(value, list):
        return [_drop_packet_provenance(item) for item in value]
    return value


def _ik_pose_policy_signature(parameters: JsonDict) -> str:
    target = parameters.get("target_pose")
    if not isinstance(target, dict):
        return ""
    xyz = target.get("xyz", target.get("translation_xyz"))
    if not (
        isinstance(xyz, list | tuple)
        and len(xyz) >= 3
        and all(isinstance(item, int | float) and not isinstance(item, bool) for item in xyz[:3])
    ):
        return ""
    orientation = {
        key: target.get(key)
        for key in (
            "rotation_matrix",
            "quat_xyzw",
            "quaternion",
            "rotvec",
            "roll",
            "pitch",
            "yaw",
        )
        if target.get(key) is not None
    }
    preserve_current = parameters.get("preserve_current_orientation")
    if preserve_current is None:
        preserve_current = not orientation
    canonical = {
        "target_xyz": list(xyz[:3]),
        "orientation_policy": (
            "preserve_current" if preserve_current is True else "explicit_orientation"
        ),
        "orientation": orientation,
    }
    return sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]


def _matching_ik_pose_policy(
    entries: Iterable[JsonDict],
    parameters: JsonDict,
) -> JsonDict | None:
    """Find a semantically identical IK endpoint after harmless JSON drift."""

    for entry in reversed(list(entries)):
        receipt = entry.get("receipt") if isinstance(entry, dict) else None
        if isinstance(receipt, dict) and ik_pose_policy_numerically_equivalent(
            receipt,
            parameters,
        ):
            return entry
    return None


def _zero_step_motion_evidence(outputs: JsonDict) -> JsonDict | None:
    motion = outputs.get("motion_summary")
    motion = motion if isinstance(motion, dict) else {}
    if motion.get("reached_target") is not False or motion.get("steps_executed") != 0:
        return None
    end = motion.get("end")
    end = end if isinstance(end, dict) else {}
    actual_xyz = end.get("xyz")
    if not isinstance(actual_xyz, list | tuple) or len(actual_xyz) < 3:
        pose_feedback = outputs.get("pose_feedback")
        pose_feedback = pose_feedback if isinstance(pose_feedback, dict) else {}
        actual_xyz = pose_feedback.get("actual_xyz")
    if not isinstance(actual_xyz, list | tuple) or len(actual_xyz) < 3:
        return None
    collision = motion.get("collision")
    collision = collision if isinstance(collision, dict) else {}
    return {
        "actual_xyz": list(actual_xyz[:3]),
        "stop_reason": motion.get("stop_reason"),
        "collision_type": collision.get("collision_type"),
        "obstacle": collision.get("obstacle"),
    }


def _failed_motion_attractor_evidence(outputs: JsonDict) -> JsonDict | None:
    motion = outputs.get("motion_summary")
    motion = motion if isinstance(motion, dict) else {}
    if motion.get("reached_target") is not False:
        return None
    end = motion.get("end")
    end = end if isinstance(end, dict) else {}
    actual_xyz = end.get("xyz")
    if not isinstance(actual_xyz, list | tuple) or len(actual_xyz) < 3:
        return None
    return {
        "actual_xyz": list(actual_xyz[:3]),
        "position_error_m": motion.get("position_error_m"),
        "steps_executed": motion.get("steps_executed"),
        "stop_reason": motion.get("stop_reason"),
    }


def _controller_execution_receipt(outputs: JsonDict) -> JsonDict | None:
    motion = outputs.get("motion_summary")
    receipt = motion.get("controller_receipt") if isinstance(motion, dict) else None
    if isinstance(receipt, dict):
        return receipt
    response = outputs.get("response")
    response_motion = (
        response.get("motion_summary") if isinstance(response, dict) else None
    )
    receipt = (
        response_motion.get("controller_receipt")
        if isinstance(response_motion, dict)
        else None
    )
    return receipt if isinstance(receipt, dict) else None


def _controller_receipt_error(receipt: JsonDict, outputs: JsonDict) -> JsonDict:
    missing = [
        key
        for key in (
            "controller_id",
            "command_interface",
            "goal_executor",
            "execution_location",
            "orientation_policy",
            "iteration_budget",
            "steps_executed",
            "stop_reason",
            "reached_target",
        )
        if receipt.get(key) in (None, "")
    ]
    mismatches: JsonDict = {}
    motion = outputs.get("motion_summary")
    motion = motion if isinstance(motion, dict) else {}
    pose_feedback = outputs.get("pose_feedback")
    pose_feedback = pose_feedback if isinstance(pose_feedback, dict) else {}
    for key, observed in (
        ("steps_executed", motion.get("steps_executed")),
        ("stop_reason", motion.get("stop_reason")),
        ("reached_target", pose_feedback.get("reached_target")),
    ):
        if observed is not None and receipt.get(key) != observed:
            mismatches[key] = {
                "receipt": receipt.get(key),
                "motion": observed,
            }
    if receipt.get("schema_version") != "openeta.controller_execution_receipt.v1":
        mismatches["schema_version"] = {
            "receipt": receipt.get("schema_version"),
            "expected": "openeta.controller_execution_receipt.v1",
        }
    return {
        **({"missing_fields": missing} if missing else {}),
        **({"mismatches": mismatches} if mismatches else {}),
    }


def _xyz_distance(left: object, right: object) -> float:
    if not (
        isinstance(left, list | tuple)
        and isinstance(right, list | tuple)
        and len(left) >= 3
        and len(right) >= 3
    ):
        return float("inf")
    try:
        return sum((float(left[i]) - float(right[i])) ** 2 for i in range(3)) ** 0.5
    except (TypeError, ValueError):
        return float("inf")


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _binary_gripper_position(value: object) -> int | None:
    number = _finite_number(value)
    if number not in {0.0, 1.0}:
        return None
    return int(number)


def _measured_gripper_open_fraction(outputs: JsonDict) -> float | None:
    response = outputs.get("response")
    sources = (
        outputs.get("observation_summary"),
        response.get("observation_summary") if isinstance(response, dict) else None,
    )
    for source in sources:
        if not isinstance(source, dict):
            continue
        robot = source.get("robot")
        gripper = robot.get("gripper_state") if isinstance(robot, dict) else None
        if isinstance(gripper, dict):
            openness = _finite_number(gripper.get("openness"))
            if openness is not None:
                return openness
        evidence = source.get("gripper_evidence")
        measured = (
            evidence.get("measured_aperture")
            if isinstance(evidence, dict)
            else None
        )
        if isinstance(measured, dict):
            openness = _finite_number(measured.get("open_fraction"))
            if openness is not None:
                return openness
    return None


def _ik_consumes_exact_viewpoint(
    parameters: JsonDict,
    proposal: JsonDict,
) -> bool:
    proposal_id = str(proposal.get("proposal_id") or "")
    requested_proposal = str(parameters.get("viewpoint_proposal_id") or "")
    requested_candidate = str(parameters.get("candidate_id") or "")
    if proposal_id and requested_proposal == proposal_id and requested_candidate:
        return True
    target = parameters.get("target_pose")
    if not isinstance(target, dict):
        return False
    target_has_orientation = isinstance(target.get("rotation_matrix"), list) or isinstance(
        target.get("quat_xyzw"), list
    )
    if not target_has_orientation:
        return False
    candidates = proposal.get("candidates")
    candidates = candidates if isinstance(candidates, list) else []
    return any(
        isinstance(candidate, dict)
        and isinstance(candidate.get("target_pose"), dict)
        and candidate["target_pose"] == target
        for candidate in candidates
    )


def _motion_ik_receipt_ids(tool_name: str, parameters: JsonDict) -> list[str]:
    if tool_name == "move_to":
        receipt_id = str(parameters.get("ik_receipt_id") or "")
        return [receipt_id] if receipt_id else []
    values = parameters.get("ik_receipt_ids")
    return (
        [str(value) for value in values if str(value or "")]
        if isinstance(values, list)
        else []
    )


def _execution_fragile_ik_evidence(receipt: object) -> JsonDict:
    if not isinstance(receipt, dict):
        return {}
    reachability = receipt.get("reachability")
    reachability = reachability if isinstance(reachability, dict) else {}
    quality = reachability.get("execution_seed_quality")
    quality = quality if isinstance(quality, dict) else {}
    candidate = receipt.get("best_candidate")
    candidate = candidate if isinstance(candidate, dict) else {}
    margin = _finite_number(
        quality.get("selected_joint_margin_rad", candidate.get("joint_margin_min_rad"))
    )
    risk_level = str(quality.get("risk_level") or "")
    if not risk_level and margin is not None and margin < 0.10:
        risk_level = "critical" if margin < 0.05 else "elevated"
    if risk_level not in {"elevated", "critical"}:
        return {}
    return {
        "ik_preview_seq": receipt.get("seq"),
        "ik_receipt_id": receipt.get("receipt_id"),
        "risk_level": risk_level,
        "selected_joint_margin_rad": margin,
        "robust_margin_threshold_rad": quality.get(
            "robust_margin_threshold_rad", 0.10
        ),
        "distant_robust_solution_count": quality.get(
            "distant_robust_solution_count"
        ),
        "note": (
            "the failed motion consumed a kinematically feasible but explicitly "
            "execution-fragile IK branch; compare another waypoint, orientation, or "
            "grasp candidate before replaying"
        ),
    }


def _typed_motion_handoff_error(
    tool_name: str,
    parameters: JsonDict,
    executable_receipts: dict[str, JsonDict],
) -> JsonDict | None:
    """Verify that dispatched motion geometry came from exact typed IK receipts."""

    if tool_name == "move_to":
        receipt_id = str(parameters.get("ik_receipt_id") or "")
        if not receipt_id:
            return {
                "reason": "missing_ik_receipt_id",
                "note": "move_to must carry the host receipt that resolved its pose",
            }
        receipt = executable_receipts.get(receipt_id)
        if not isinstance(receipt, dict):
            return {
                "reason": "unknown_or_non_executable_ik_receipt_id",
                "ik_receipt_id": receipt_id,
            }
        if not _typed_receipt_matches_motion(receipt, parameters):
            return {
                "reason": "receipt_pose_mismatch",
                "ik_receipt_id": receipt_id,
                "ik_preview_seq": receipt.get("seq"),
            }
        return None

    ids = parameters.get("ik_receipt_ids")
    trajectory = parameters.get("trajectory")
    if not isinstance(ids, list) or not 1 <= len(ids) <= 5:
        return {
            "reason": "missing_or_invalid_ik_receipt_ids",
            "note": "follow_eef_trajectory requires one to five ordered receipt ids",
        }
    if not isinstance(trajectory, list) or len(trajectory) != len(ids):
        return {
            "reason": "receipt_trajectory_cardinality_mismatch",
            "receipt_count": len(ids),
            "trajectory_count": len(trajectory) if isinstance(trajectory, list) else None,
        }
    for index, (receipt_id_value, pose) in enumerate(zip(ids, trajectory)):
        receipt_id = str(receipt_id_value or "")
        receipt = executable_receipts.get(receipt_id)
        if not receipt_id or not isinstance(receipt, dict):
            return {
                "reason": "unknown_or_non_executable_trajectory_receipt",
                "waypoint_index": index,
                "ik_receipt_id": receipt_id,
            }
        if not isinstance(pose, dict) or not _typed_receipt_matches_motion(
            receipt, {"target_pose": pose}
        ):
            return {
                "reason": "trajectory_receipt_pose_mismatch",
                "waypoint_index": index,
                "ik_receipt_id": receipt_id,
                "ik_preview_seq": receipt.get("seq"),
            }
    return None


def _typed_receipt_matches_motion(receipt: JsonDict, parameters: JsonDict) -> bool:
    """Match exact serialized poses first, then allow harmless numeric drift."""

    recorded_pose = receipt.get("target_pose")
    if isinstance(recorded_pose, dict):
        recorded_parameters = {
            "target_pose": recorded_pose,
            "preserve_current_orientation": (
                str(receipt.get("orientation_policy") or "") == "preserve_current"
            ),
        }
        if _ik_pose_policy_signature(recorded_parameters) == _ik_pose_policy_signature(
            parameters
        ):
            return True
    return ik_pose_policy_numerically_equivalent(receipt, parameters)


def _issue(
    seq: int,
    tool: str,
    code: str,
    severity: str,
    *,
    evidence: JsonDict | None = None,
) -> JsonDict:
    return {
        "seq": seq,
        "tool": tool,
        "code": code,
        "severity": severity,
        **({"evidence": evidence} if evidence else {}),
    }


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if isinstance(row, dict):
                yield row


def _resolve_sibling_trace_source(tool_events: Path) -> Path | None:
    candidates = (
        tool_events.parent / "trace.jsonl",
        tool_events.parent.parent / "trace.jsonl",
    )
    return next((candidate for candidate in candidates if candidate.is_file()), None)


def _audit_pipeline_trace(
    trace_source: Path | None,
    *,
    successful_contact_executions: dict[str, list[JsonDict]],
) -> JsonDict:
    """Audit plans rejected before a tool event could be emitted.

    Tool-event logs necessarily omit blocked calls.  The durable runtime trace is
    therefore the authoritative companion source for gate feedback.  This audit
    observes host decisions only; it does not add or enforce a task-stage policy.
    """

    if trace_source is None:
        return {"blocked_count": 0, "issues": []}
    issues: list[JsonDict] = []
    blocked_count = 0
    repeated: Counter[tuple[str, str]] = Counter()
    for line_number, row in enumerate(_read_jsonl(trace_source), start=1):
        if row.get("event_type") != "pipeline_plan":
            continue
        payload = row.get("payload")
        payload = payload if isinstance(payload, dict) else {}
        if payload.get("status") != "blocked":
            continue
        blocked_count += 1
        request = payload.get("request")
        request = request if isinstance(request, dict) else {}
        tool = str(request.get("name") or "")
        metadata = payload.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        repair = metadata.get("repair_bundle")
        repair = repair if isinstance(repair, dict) else {}
        code = str(repair.get("code") or "")
        if not code:
            issues.append(
                _issue(
                    line_number,
                    tool,
                    "blocked_pipeline_without_repair_code",
                    "error",
                )
            )
            continue
        allowed = repair.get("allowed_next_calls")
        actionable = [item for item in allowed if isinstance(item, dict)] if isinstance(
            allowed, list
        ) else []
        if not actionable:
            issues.append(
                _issue(
                    line_number,
                    tool,
                    "blocked_pipeline_without_actionable_repair",
                    "error",
                    evidence={"gate_code": code},
                )
            )
        repeated[(tool, code)] += 1
        if repeated[(tool, code)] == 2:
            issues.append(
                _issue(
                    line_number,
                    tool,
                    "repeated_pipeline_gate_block",
                    "warning",
                    evidence={"gate_code": code, "occurrences": 2},
                )
            )
        if code != "compiled_contact_receipt_missing":
            continue
        latest_ik = repair.get("latest_ik_preview")
        latest_ik = latest_ik if isinstance(latest_ik, dict) else {}
        target_pose = latest_ik.get("target_pose")
        target_pose = target_pose if isinstance(target_pose, dict) else {}
        compiled_id = str(target_pose.get("compiled_grasp_id") or "")
        trace_timestamp = _finite_number(row.get("timestamp_s"))
        prior_contacts = [
            item
            for item in successful_contact_executions.get(compiled_id, [])
            if trace_timestamp is None
            or item.get("timestamp_s") is None
            or float(item["timestamp_s"]) <= trace_timestamp
        ]
        if compiled_id and prior_contacts:
            latest_contact = prior_contacts[-1]
            issues.append(
                _issue(
                    line_number,
                    tool,
                    "gate_rejected_close_after_successful_contact",
                    "error",
                    evidence={
                        "gate_code": code,
                        "compiled_grasp_id": compiled_id,
                        "successful_contact_tool_seq": latest_contact.get("seq"),
                        "successful_contact_timestamp_s": latest_contact.get(
                            "timestamp_s"
                        ),
                        "note": (
                            "the simulator already reported reached_target=true for "
                            "this compiled contact, but the host-derived contact receipt "
                            "was absent when close was requested"
                        ),
                    },
                )
            )
    return {"blocked_count": blocked_count, "issues": issues}


def _resolve_tool_event_source(path: Path) -> Path:
    """Resolve a file, rollout directory, session directory, or attempt directory.

    Refuse ambiguous run/job directories instead of silently auditing an arbitrary
    attempt. This keeps the CLI convenient without weakening experiment identity.
    """

    if path.is_file():
        return path
    if not path.exists():
        raise FileNotFoundError(
            f"tool event source does not exist: {path}; pass tool_calls.jsonl or "
            "a directory containing exactly one such file"
        )
    if not path.is_dir():
        raise ValueError(f"tool event source is neither a file nor directory: {path}")

    direct_candidates = (
        path / "tool_calls.jsonl",
        path / "rollout" / "tool_calls.jsonl",
    )
    for candidate in direct_candidates:
        if candidate.is_file():
            return candidate

    matches = sorted(
        candidate
        for candidate in path.rglob("tool_calls.jsonl")
        if candidate.is_file()
    )
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"no tool_calls.jsonl found beneath directory: {path}")
    preview = ", ".join(str(candidate) for candidate in matches[:5])
    suffix = " ..." if len(matches) > 5 else ""
    raise ValueError(
        f"ambiguous tool event directory {path}: found {len(matches)} rollouts "
        f"({preview}{suffix}); pass one session, rollout directory, or file"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tool-events",
        required=True,
        help=(
            "tool_calls.jsonl, or a rollout/session/attempt directory containing "
            "exactly one tool_calls.jsonl"
        ),
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)
    report = audit_tool_event_file(args.tool_events)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
