"""Request-scoped explanations of changes between native intent and execution."""
from copy import deepcopy


def event(source, effect, code, message, *, execution_state=None, details=None):
    row = {'source':source, 'effect':effect, 'reason_code':code, 'message':message}
    if execution_state is not None:row['execution_state'] = execution_state
    if details:row['details'] = deepcopy(details)
    return row


def motion_interventions(motion):
    if not isinstance(motion, dict):return []
    rows=[];summary=motion.get('motion_summary') or {};reason=motion.get('reason_code')
    if reason and reason not in ('target_reached','gripper_horizon_completed'):
        rows.append(event('motion_pipeline','stopped',reason,
            (motion.get('recovery') or {}).get('message') or
            'The requested motion did not complete. Use the returned actual state; execution may be unknown.',
            execution_state=motion.get('execution_state','unknown'),
            details={'failure_stage':motion.get('failure_stage','execution'),
                     'diagnostics':summary}))
    adjustments=summary.get('control_adjustments')
    if adjustments:
        message = 'Control used the reported speed, horizon and tracking settings.'
        if adjustments.get('path_backtracked_steps', 0):
            message += ' Some proposed joint-velocity steps were reduced to pass Cartesian and geometry checks.'
        if adjustments.get('joint_velocity_projection_steps', 0):
            message += ' Joint velocities were projected to respect joint limits during checked boundary recovery.'
        if adjustments.get('recovery_resolve_steps', 0):
            message += ' Alternative velocity QPs were selected after tracking/geometry checks; the requested endpoint is unchanged.'
        if adjustments.get('verified_qp_fallback_steps', 0):
            message += ' The primary collision QP had no solution on some ticks; fallback candidates passed the existing geometric/joint checks and actual post-step checks.'
        if adjustments.get('transport_profile') in ('fixture_grip_stabilized', 'attached_object_stabilized', 'attached_object_gentle'):
            message += ' Contact/load state selected a slower stabilized transport profile.'
        changed = (adjustments.get('verified_qp_fallback_steps', 0) or adjustments.get('recovery_resolve_steps', 0) or adjustments.get('path_backtracked_steps', 0) or
                   adjustments.get('joint_velocity_projection_steps', 0) or
                   adjustments.get('transport_profile') in ('fixture_grip_stabilized', 'attached_object_stabilized', 'attached_object_gentle'))
        rows.append(event('controller','adjusted' if changed else 'configured','bounded_control',
            message,
            execution_state=motion.get('execution_state','unknown'),details=adjustments))
    return rows
