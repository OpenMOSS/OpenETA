"""Read-only stalled-motion evidence; never changes a QP or authorizes motion."""
import math
from sim.controllers.collision_feedback import classify_collision, public_candidate_obstacles


def _classes(value):
    return [{k:v for k,v in row.items() if k != 'constraint_boundary'}
            for row in public_candidate_obstacles(value)]


def public_stall_context(value):
    if not isinstance(value, dict):
        return {}
    result = {}
    for key in ('qp_evidence', 'contact_evidence'):
        if value.get(key) in ('available', 'unavailable'):
            result[key] = value[key]
    for key in ('active_clearance_constraints', 'measured_robot_contacts'):
        if key in value:
            result[key] = _classes(value[key])
    for key in ('requested_joint_speed_max_rad_s', 'commanded_joint_speed_max_rad_s'):
        number = value.get(key)
        if type(number) in (int, float) and math.isfinite(number) and number >= 0:
            result[key] = number
    if result:
        result['qp_scope'] = 'last_pre_actuation_configuration'
        result['contact_scope'] = 'measured_end_configuration'
        result['causal_attribution'] = 'not_established'
    return result


def stalled_motion_context(configuration, policy, requested_velocity, commanded_velocity,
                           arm_indices, dt, actual_data, *, collision_qp_used):
    """Report binding robot-clearance rows and measured external robot contact.

    A binding row is evidence, not proof it caused the stall. No counterfactual
    solve is performed. A diagnostic failure must not mask the motion outcome.
    """
    import numpy as np
    result = {'qp_evidence': 'unavailable', 'contact_evidence': 'unavailable'}
    try:
        for name, velocity in [('requested_joint_speed_max_rad_s', requested_velocity),
                               ('commanded_joint_speed_max_rad_s', commanded_velocity)]:
            result[name] = float(np.max(np.abs(velocity[arm_indices])))
        if collision_qp_used:
            limit = policy['limit']
            constraint = limit.compute_qp_inequalities(configuration, dt)
            slack = constraint.h - constraint.G @ (requested_velocity * dt)
            rows = []
            active = np.flatnonzero(np.isfinite(slack) & (np.abs(slack) <= 1e-7)
                                    & (np.linalg.norm(constraint.G[:, arm_indices], axis=1) > 1e-10))
            for index in active:
                a,b = limit.geom_id_pairs[index]
                rows.append(classify_collision({'detected': True, 'geom1_id':a, 'geom2_id':b}, policy))
            result.update(qp_evidence='available', active_clearance_constraints=_classes(rows))
    except Exception:
        # Keep already established execution/stop evidence if diagnostics fail.
        result['qp_evidence'] = 'unavailable'
    try:
        rows = []
        robot_ids = set(policy['robot_geom_ids'])
        for contact in actual_data.contact:
            distance = float(contact.dist)
            if not math.isfinite(distance) or distance > 0:
                continue  # A positive-distance margin record is not touching.
            a,b = int(contact.geom1), int(contact.geom2)
            if len({a,b} & robot_ids) != 1:
                continue  # Closed fingertips touching each other are not external contact.
            rows.append(classify_collision({'detected': True, 'geom1_id':a, 'geom2_id':b}, policy))
        result.update(contact_evidence='available', measured_robot_contacts=_classes(rows))
    except Exception:
        result['contact_evidence'] = 'unavailable'
    return public_stall_context(result)
