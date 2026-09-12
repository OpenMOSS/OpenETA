"""Robot-side binary contact telemetry, with no scene identity or target oracle.

Contacts are reconstructed at the final configuration in separate MjData. No
physics steps, contact forces, object poses, or collision exemptions are added.
"""


def contact_summary(model, qpos, finger_groups, robot_geom_ids):
    import mujoco
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    touched = set()
    robot = set(robot_geom_ids)
    for contact in data.contact:
        if contact.dist > 0:
            continue
        a, b = int(contact.geom1), int(contact.geom2)
        if a in robot and b not in robot:
            touched.add(a)
        if b in robot and a not in robot:
            touched.add(b)
    result = {'available': True, 'measurement': 'final_configuration_contact',
              'retention_proven': False}
    for group, ids in finger_groups.items():
        result[group + '_contact'] = bool(touched.intersection(ids))
    pads = result['left_fingerpad_contact'], result['right_fingerpad_contact']
    fingers = result['left_finger_contact'], result['right_finger_contact']
    result['contact_pattern'] = ('bilateral_pads' if all(pads) else
        'single_pad' if any(pads) else 'finger_body_only' if any(fingers) else 'no_contact')
    return result


def measure_gripper_contact(env):
    # A telemetry failure must not erase the outcome of completed actuation.
    try:
        return _measure_gripper_contact(env)
    except Exception:
        import logging
        logging.getLogger(__name__).exception('Final gripper contact telemetry unavailable')
        return {'available': False, 'retention_proven': False}


def _measure_gripper_contact(env):
    import mujoco
    from sim.controllers.mink_goal import _libero_runtime
    raw, robot = _libero_runtime(env)
    gripper = robot.gripper
    if isinstance(gripper, dict):
        if len(gripper) != 1:
            return {'available': False, 'retention_proven': False}
        gripper = next(iter(gripper.values()))
    names = gripper.important_geoms
    groups = ('left_finger', 'right_finger', 'left_fingerpad', 'right_fingerpad')
    if any(not names.get(group) for group in groups):
        return {'available': False, 'retention_proven': False}
    model = raw.sim.model._model
    ids = {group: [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
                   for name in names[group]] for group in groups}
    if any(i < 0 for values in ids.values() for i in values):
        return {'available': False, 'retention_proven': False}
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot.robot_model.root_body)
    if root < 0:
        return {'available': False, 'retention_proven': False}
    bodies = {root}
    for i in range(root + 1, model.nbody):
        if int(model.body_parentid[i]) in bodies:
            bodies.add(i)
    robot_geoms = [i for i in range(model.ngeom) if int(model.geom_bodyid[i]) in bodies]
    return contact_summary(model, raw.sim.data.qpos, ids, robot_geoms)
