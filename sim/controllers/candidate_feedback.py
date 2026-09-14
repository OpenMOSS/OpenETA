"""Bounded per-candidate diagnostics, without simulator geometry or QP data."""
import math
from sim.controllers.collision_feedback import public_candidate_obstacles, public_tracking_constraints

GEOMETRY_REASONS = {'joint_limit', 'robot_collision', 'attached_collision', 'fixture_contact_scope'}
VARIANTS = {'primary', 'track_constrained', 'hold_posture', 'seed_relaxed', 'posture_positive', 'posture_negative'}


def public_candidate_trace(value):
    if not isinstance(value, list):
        return []
    rows = []
    for item in value[:36]:
        if not isinstance(item, dict) or not isinstance(item.get('variant'), str) or item['variant'] not in VARIANTS:
            continue
        row = {'variant': item['variant']}
        for key in ('scale', 'position_step_m', 'rotation_step_rad'):
            n = item.get(key)
            if type(n) in (float, int) and math.isfinite(n) and n >= 0:
                row[key] = n
        for key in ('tracking_passed', 'checks_passed'):
            if type(item.get(key)) is bool:
                row[key] = item[key]
        if item.get('geometry_status') in ('not_checked', 'passed', 'rejected'):
            row['geometry_status'] = item['geometry_status']
        row['tracking_rejections'] = public_tracking_constraints(item.get('tracking_rejections'))
        reasons = item.get('geometry_rejections')
        row['geometry_rejections'] = sorted({v for v in reasons if isinstance(v, str) and v in GEOMETRY_REASONS}) if isinstance(reasons, list) else []
        row['obstacles'] = public_candidate_obstacles(item.get('obstacles'))
        rows.append(row)
    return rows
