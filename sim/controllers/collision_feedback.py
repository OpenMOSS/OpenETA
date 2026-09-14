"""Classify private collision geometry into bounded robot-side feedback."""


def classify_collision(collision, policy):
    result = dict(collision)
    if not result.get('detected'):
        return result
    ids = {result.get('geom1_id'), result.get('geom2_id')}
    robot = ids.intersection(policy['robot_geom_ids'])
    grip = ids.intersection(policy['gripper_geom_ids'])
    result['collision_type'] = result.get('collision_type') or (
        'self_collision' if len(robot) == 2 else 'robot_world' if robot else 'unspecified_contact')
    result['robot_part'] = 'gripper' if grip and robot <= grip else 'arm' if robot else 'unknown'
    result['contact_binding_active'] = bool(policy.get('authorized_target_geom_count'))
    if grip and 'finger_geom_ids' in policy:
        result['gripper_part'] = ('finger' if grip <= set(policy['finger_geom_ids']) else 'base_or_palm')
    if len(robot) == 2:
        result['obstacle_relation'] = 'robot_self'
    elif len(robot) == 1:
        world = ids - robot
        if not result['contact_binding_active']:
            result['obstacle_relation'] = 'unbound_world'
        elif 'authorized_target_geom_ids' in policy:
            result['obstacle_relation'] = ('authorized_target' if world <= set(policy['authorized_target_geom_ids']) else 'outside_contact_target')
    return result


CANDIDATE_FIELDS = {
    'robot_part': ('arm', 'gripper', 'unknown'),
    'gripper_part': ('finger', 'base_or_palm'),
    'obstacle_relation': ('robot_self', 'unbound_world', 'authorized_target', 'outside_contact_target'),
    'constraint_boundary': ('hard_distance', 'clearance_recovery'),
}


def public_candidate_obstacles(value):
    """Bounded candidate classifications, never an actual-contact assertion."""
    rows = []
    if not isinstance(value, list):
        return rows
    for item in value[:48]:
        if not isinstance(item, dict):
            continue
        row = {k: item[k] for k, choices in CANDIDATE_FIELDS.items()
               if isinstance(item.get(k), str) and item[k] in choices}
        if row.get('robot_part') and row not in rows:
            rows.append(row)
        if len(rows) == 8:
            break
    return rows


def rejected_candidate_obstacles(current, candidate, policy, *, recovering):
    """Classify near pairs of a geometrically rejected velocity candidate.

    These are proximity constraints on a proposed next configuration, not
    collisions asserted at the measured current configuration.
    """
    boundary = float(policy['hard_stop_distance_m'])
    if recovering:
        boundary = max(boundary, float(policy['minimum_distance_from_collisions_m']))
    selected = {pair for pair, distance in candidate.items() if distance < boundary}
    if recovering:
        active = {pair: distance for pair, distance in current.items() if distance < boundary}
        if not active or set(current) != set(candidate):
            return []
        # Mirror the authoritative escape predicate's public default epsilons
        # only to explain its rejection; this helper never authorizes motion.
        direct = {pair for pair, distance in candidate.items()
                  if (pair not in active and distance < policy['hard_stop_distance_m'])
                  or (pair in active and distance < active[pair]-1e-6)}
        selected = direct or {pair for pair in active
                              if candidate[pair] <= min(active.values())+1e-5}
    rows = []
    for pair in sorted(selected):
        classified = classify_collision({'detected': True, 'geom1_id': pair[0],
                                          'geom2_id': pair[1]}, policy)
        classified['constraint_boundary'] = 'clearance_recovery' if recovering else 'hard_distance'
        rows.append(classified)
    return public_candidate_obstacles(rows)


TRACKING_CONSTRAINTS = {'position_step_limit', 'rotation_step_limit', 'position_corridor', 'rotation_corridor'}


def public_tracking_constraints(value):
    return sorted({v for v in value if isinstance(v, str) and v in TRACKING_CONSTRAINTS}) if isinstance(value, list) else []
