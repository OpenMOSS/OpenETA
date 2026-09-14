"""Small allowlisted motion receipts for the experimental Codex operator.

Never forward simulator error strings, scene names, internal paths or raw
recovery dictionaries across this boundary.
"""
import math

REASONS = {
    'contact_authorization_unresolved': ('contact_binding_failed', 'Contact could not be authorized for the measured surface. Inspect the contact point; repeated identical requests will not repair missing or unsupported geometry.'),
    'collision_detected': ('replan_collision', 'A collision check stopped the motion. Inspect the returned views and actual pose; choose a different checked waypoint that increases clearance. Keep collision checks enabled.'),
    'control_step_failed': ('replan_controller', 'The controller could not find a safe next step. Use the actual pose and choose a materially different checked waypoint or orientation.'),
    'iteration_limit': ('replan_convergence', 'The controller stopped at its step limit before reaching the target. Inspect position/orientation errors and replan from the actual pose.'),
    'local_convergence_stalled': ('replan_convergence', 'Motion stalled before reaching the target. Choose a different waypoint or orientation from the measured pose.'),
    'target_reached': ('inspect_result', 'The robot reached its target; inspect object-relative evidence before judging task success.'),
    'gripper_horizon_completed': ('inspect_grasp', 'The checked gripper horizon completed. Inspect aperture and images; completion does not prove a grasp.'),
    'episode_terminated': ('check_episode', 'The episode ended; consult official task success evidence.'),
    'ik_search_no_solution': ('replan_ik', 'IK did not find a solution. Change target position or orientation; no execution is authorized.'),
    'ik_search_timeout': ('replan_ik', 'IK search timed out without authorization. Change the target or retry after inspecting the current pose.'),
    'tool_execution_failed': ('inspect_result', 'The tool failed without a recognized motion verdict. Inspect the measured state; motion may be unknown.'),
}
METRICS = ('steps_executed', 'position_error_m', 'max_axis_position_error_m',
           'orientation_error_rad', 'orientation_error_deg')


def tool_outputs(command, name):
    calls = [c for c in command.get('tool_calls', []) if c.get('name') == name]
    if len(calls) != 1:
        return {}
    return ((calls[0].get('result') or {}).get('details') or {}).get('outputs') or {}


def motion_feedback(command, name='move_to'):
    outputs = tool_outputs(command, name)
    raw = outputs.get('motion_summary') or (outputs.get('response') or {}).get('motion_summary') or {}
    if not raw:
        return {}
    reason = raw.get('stop_reason')
    reason = reason if reason in REASONS else 'tool_execution_failed'
    summary = {'stop_reason': reason}
    for key in METRICS:
        value = raw.get(key)
        if isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0:
            summary[key] = value
    if type(raw.get('reached_target')) is bool:
        summary['reached_target'] = raw['reached_target']
    collision = raw.get('collision') or {}
    safe_collision = {k: collision[k] for k in ('detected','endpoint_checked','trajectory_checked','world_checked','self_checked') if type(collision.get(k)) is bool}
    if collision.get('collision_class') in ('none','attached_object_world','self_collision','world_collision','robot_world','robot_self','endpoint','unspecified_contact'):
        safe_collision['collision_class'] = collision['collision_class']
    for key, choices in {'check_mode':('pre_actuation_configuration','post_step_configuration','per_step_pre_actuation_and_post_step_configuration','pre_actuation_attached_object_configuration','post_step_attached_object_configuration'),
                         'robot_part':('arm','gripper','unknown'), 'checked_during':('gripper_actuation',),
                         'contact_scope': ('local_articulated_patch','single_fixture_geom','target_object','none'),
                         'contact_scope_violation': ('outside_local_patch','fixture_penetration','contact_geometry_uncertain'),
                         'gripper_part':('finger','base_or_palm'),
                         'obstacle_relation':('robot_self','unbound_world','authorized_target','outside_contact_target'),
                         'check_stage':('target_endpoint',),
                         'placement_constraint':('outside_receptacle_corridor','receptacle_corridor_too_narrow')}.items():
        if collision.get(key) in choices:
            safe_collision[key] = collision[key]
    for key in ('contact_binding_active','prediction_checked'):
        if type(collision.get(key)) is bool:
            safe_collision[key] = collision[key]
    if safe_collision:
        summary['collision'] = safe_collision
    failure = raw.get('controller_failure') or {}
    if failure.get('code') in ('constraint_escape_preview_rejected','mink_qp_no_solution','joint_limit_violation',
                             'cartesian_segment_blocked','cartesian_path_deviation','cartesian_progress_stalled','local_convergence_stalled'):
        summary['controller_failure'] = {'code': failure['code']}
    constraints = failure.get('constraints')
    if 'controller_failure' in summary and isinstance(constraints, list):
        summary['controller_failure']['constraints'] = sorted({v for v in constraints if isinstance(v, str)
            and v in {'joint_limit', 'robot_collision', 'attached_collision', 'path_tracking', 'fixture_contact_scope'}})
    from sim.controllers.collision_feedback import public_candidate_obstacles
    candidates = public_candidate_obstacles(failure.get("candidate_obstacles"))
    if candidates and "controller_failure" in summary:
        summary["controller_failure"]["candidate_obstacles"] = candidates
    from sim.controllers.collision_feedback import public_tracking_constraints
    tracking_constraints = public_tracking_constraints(failure.get("tracking_constraints"))
    if tracking_constraints and "controller_failure" in summary:
        summary["controller_failure"]["tracking_constraints"] = tracking_constraints
    from sim.controllers.stall_feedback import public_stall_context
    stall = public_stall_context(failure.get('stall_context'))
    if stall and 'controller_failure' in summary:
        summary['controller_failure']['stall_context'] = stall
    from sim.controllers.candidate_feedback import public_candidate_trace
    trace = public_candidate_trace(failure.get('candidate_trace'))
    if trace and 'controller_failure' in summary:
        summary['controller_failure']['candidate_trace'] = trace
    steps = summary.get('steps_executed')
    controller = raw.get('controller_receipt') or {}
    pose = {k:controller[k] for k in ('ik_seed_validated','position_within_tolerance','orientation_within_tolerance') if type(controller.get(k)) is bool}
    if controller.get('full_pose_outcome') in ('reached','execution_failed_after_validated_ik','local_execution_failed'):
        pose['full_pose_outcome'] = controller['full_pose_outcome']
    for key in ('arm_joint_margin_min_rad','nearest_limit_joint_index','control_tick_peak_orientation_error_rad'):
        value = controller.get(key)
        if type(value) in (float,int) and math.isfinite(value):pose[key] = value
    if pose:summary['pose_diagnostics'] = pose
    if controller.get('cartesian_tracking_enabled') is True:
        tracking = {'enabled': True}
        for key in ('path_peak_cross_track_m', 'path_peak_rotation_deviation_rad', 'path_backtracked_steps'):
            value = controller.get(key)
            if type(value) in (float, int) and math.isfinite(value) and value >= 0:
                tracking[key] = value
        summary['path_tracking'] = tracking
    if type(controller.get('fixture_grip_stabilization_active')) is bool:
        summary['fixture_grip_stabilization_active'] = controller['fixture_grip_stabilization_active']
    if type(controller.get('grip_stabilization_active')) is bool:
        summary['grip_stabilization_active'] = controller['grip_stabilization_active']
    if controller.get('grip_stabilization_kind') in ('none', 'fixture', 'attached_object'):
        summary['grip_stabilization_kind'] = controller['grip_stabilization_kind']
    adjustments = {}
    for key in ('joint_velocity_limit_rad_s', 'iteration_budget', 'position_tolerance_m',
                'joint_velocity_projection_steps', 'path_backtracked_steps', 'path_minimum_step_scale',
                'path_cross_track_limit_m', 'path_rotation_deviation_limit_rad', 'orientation_tolerance_rad', 'recovery_resolve_steps', 'verified_qp_fallback_steps'):
        value = controller.get(key)
        if type(value) in (int, float) and math.isfinite(value) and value >= 0:
            adjustments[key] = value
    if controller.get('transport_profile') in ('nominal', 'fixture_grip_stabilized', 'attached_object_stabilized', 'attached_object_gentle'):
        adjustments['transport_profile'] = controller['transport_profile']
    if adjustments:
        summary['control_adjustments'] = adjustments
    if controller.get('motion_mode') in ('strict', 'recovery'):
        summary.setdefault('control_adjustments', {})['motion_mode'] = controller['motion_mode']
    contact = raw.get('gripper_contact') or {}
    safe_contact = {k: contact[k] for k in ('available', 'left_finger_contact',
        'right_finger_contact', 'left_fingerpad_contact', 'right_fingerpad_contact')
        if type(contact.get(k)) is bool}
    if contact.get('contact_pattern') in ('bilateral_pads', 'single_pad', 'finger_body_only', 'no_contact'):
        safe_contact['contact_pattern'] = contact['contact_pattern']
    if safe_contact:
        safe_contact['retention_proven'] = False
        summary['gripper_contact'] = safe_contact
    action, message = REASONS[reason]
    if reason == 'collision_detected' or (
            reason == 'control_step_failed' and safe_collision.get('detected') is True):
        action, message = REASONS['collision_detected']
        stage = safe_collision.get('check_mode')
        message = ('The next actuation was rejected before execution. ' if stage in ('pre_actuation_configuration', 'pre_actuation_attached_object_configuration') else
                   'A protected contact was detected after a physics step. ' if stage in ('post_step_configuration', 'post_step_attached_object_configuration') else '') + message
        if safe_collision.get('check_stage') == 'target_endpoint':
            message = ('The requested endpoint was rejected before execution; this does not establish '
                       'a collision at the current pose. Inspect fresh views and change the proposed '
                       'placement or waypoint while keeping collision checks enabled.')
        constraint = safe_collision.get('placement_constraint')
        if constraint == 'outside_receptacle_corridor':
            action = 'realign_carried_object_before_descent'
            message += (' The carried object would extend outside the receptacle interior corridor. '
                        'At a checked clear height, visually align the whole object inside the opening '
                        'before descending. Account for its offset from the gripper; raising alone '
                        'at the same XY will not correct the proposed placement.')
        elif constraint == 'receptacle_corridor_too_narrow':
            action = 'reconsider_carried_object_orientation'
            message += (' The conservative carried-object footprint cannot fit the opening at this '
                        'orientation. Inspect fresh views and choose a checked orientation or placement '
                        'alternative; do not repeatedly descend at the same pose.')
        if safe_collision.get('gripper_part') == 'base_or_palm':
            message += ' The interfering robot part is the gripper base/palm, not just a fingertip.'
        if safe_collision.get('obstacle_relation') == 'outside_contact_target':
            message += ' Contact with your marked target is already authorized; the obstruction is outside that target. Re-marking the same target will not clear this obstruction. Change the approach side or clearance waypoint using fresh views.'
        if name == 'gripper_control':
            message += ' Gripper actuation was interrupted; inspect the actual aperture and use a checked release before retrying contact.'
    if name == 'gripper_control' and safe_contact.get('available'):
        message += ' Finger contact is with external geometry, not confirmed target identity or retention. '
        message += ('Both pads contact geometry; confirm retention with a small checked motion and fresh views.'
                    if safe_contact.get('contact_pattern') == 'bilateral_pads' else
                    'Both pads are not in contact. After closing, inspect/reposition instead of assuming a secure grasp and making a long pull.')
    if failure.get('code') == 'cartesian_segment_blocked':
        action = 'replan_cartesian_segment'
        message = ('No checked next step could follow the requested straight position/rotation segment. '
                   'This does not prove the current pose is colliding or the endpoint is unreachable. '
                   'Inspect the actual pose and choose a different clearance waypoint; keep safety checks enabled.')
    elif failure.get('code') == 'cartesian_path_deviation':
        action = 'inspect_path_deviation'
        message = ('Actual motion departed from the requested Cartesian segment and was stopped. '
                   'Use the measured end pose and fresh views before planning another move.')
    elif failure.get('code') == 'cartesian_progress_stalled':
        action = 'replan_stalled_segment'
        message = ('Neither position nor orientation made meaningful progress in the bounded tracking window. '
                   'Inspect the actual pose, joint margin and views; change the waypoint or orientation instead of repeating the same target.')
    if 'fixture_contact_scope' in summary.get('controller_failure', {}).get('constraints', []):
        message += (' Candidate steps could not satisfy the local fixture contact scope. '
                    'Inspect the marked surface and approach; the grant does not cover remote surfaces or deep penetration.')
    scope = safe_collision.get('contact_scope_violation')
    if scope:
        action = 'replan_local_contact'
        message = ('Contact left the local patch around your marked surface. Refresh a visible contact point or change the approach.'
                   if scope == 'outside_local_patch' else
                   'The local contact geometry could not be verified. Inspect fresh views and choose a different checked approach.'
                   if scope == 'contact_geometry_uncertain' else
                   'Contact exceeded the fixture penetration limit. Inspect actual state and choose a checked retreat; do not keep pushing.')
    if failure.get('code') == 'cartesian_segment_blocked':
        if any(row.get('robot_part') == 'arm' for row in candidates):
            message += ' Rejected candidates were constrained by arm clearance; include the arm in your clearance plan.'
        if any(row.get('gripper_part') == 'base_or_palm' for row in candidates):
            message += ' Rejected candidates were constrained by gripper base/palm clearance; inspect that volume when changing approach.'
        if any(row.get('gripper_part') == 'finger' for row in candidates):
            message += ' Rejected candidates were constrained by finger clearance.'
        if any(row.get('obstacle_relation') == 'outside_contact_target' for row in candidates):
            message += ' The constraining geometry is outside the authorized contact target. Re-marking that same target does not change this constraint; choose another approach or clearance waypoint.'
        if any(row.get('constraint_boundary') == 'clearance_recovery' for row in candidates):
            message += ' A candidate failed to improve the active clearance boundary; this is not a report of newly executed penetration.'
    if failure.get('code') == 'cartesian_segment_blocked':
        if 'rotation_corridor' in tracking_constraints:
            message += ' Candidate orientations could not stay within the planned rotation corridor. Choose a different rotation axis or split reorientation and clearance translation into separately checked moves.'
        if 'position_corridor' in tracking_constraints:
            message += ' Candidate positions could not stay within the planned straight segment. Replan a different segment from the actual pose instead of appending collinear points.'
    if stall:
        active = stall.get('active_clearance_constraints', [])
        measured = stall.get('measured_robot_contacts', [])
        if active:
            message += ' The last QP had binding robot-clearance constraints; these are associated evidence, not a proven cause of the stall.'
        if any(row.get('gripper_part') == 'base_or_palm' for row in active):
            message += ' Gripper base/palm clearance was constrained.'
        if any(row.get('gripper_part') == 'base_or_palm' for row in measured):
            message += ' Measured end-state contact includes the gripper base/palm; finger-only contact feedback does not cover it.'
        if any(row.get('obstacle_relation') == 'outside_contact_target' for row in active + measured):
            message += ' This evidence includes geometry outside the authorized contact target. Retreat and change approach before continuing; re-marking the same target does not authorize this obstruction.'
        if active or measured:
            message += ' collision.detected=false means no hard collision stop was triggered, not absence of contact or avoidance constraints.'
        if stall.get('qp_evidence') == 'unavailable' or stall.get('contact_evidence') == 'unavailable':
            message += ' Some stall diagnostics are unavailable; do not infer free space from missing evidence.'
    if pose.get('ik_seed_validated') and reason in ('iteration_limit','local_convergence_stalled','control_step_failed'):
        message += ' The endpoint IK seed passed validation; local execution failed. Check joint margin and separate clearance motion from large orientation changes.'
    if name == 'move_to' and reason in ('iteration_limit', 'local_convergence_stalled', 'control_step_failed', 'collision_detected'):
        message += ' If this was a grasp approach, do not close at the assumed target. Inspect the actual grip-site and fresh views, then retreat/reorient or remeasure and correct the approach before closing.'
    return {'reason_code': reason, 'motion_summary': summary,
            'physics_executed': bool(steps > 0) if steps is not None else None,
            'recovery': {'action': action, 'message': message}}


def execution_error(command):
    if command.get('status') not in ('failed', 'blocked'):
        return None
    receipt = motion_feedback(command, (command.get('request') or {}).get('name', 'move_to'))
    reason = receipt.get('reason_code', 'tool_execution_failed')
    return {'code': reason, 'status': command['status'],
            'message': (receipt.get('recovery') or {}).get('message', REASONS[reason][1]), **({'motion': receipt} if receipt else {})}
