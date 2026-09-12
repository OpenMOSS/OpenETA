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
