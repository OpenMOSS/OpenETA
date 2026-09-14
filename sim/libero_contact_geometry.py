"""Host-private, live geometry for LIBERO articulated fixture contact.

Each candidate identifies an articulated body through one visible collision geom.
Its bounded contact patch follows that body; fixed panels remain obstacles.
"""
import numpy as np


def descendant(model, body_id, root_id):
    while body_id > 0 and body_id != root_id:
        body_id = int(model.body_parentid[body_id])
    return body_id == root_id


def articulated(model, body_id, root_id):
    while body_id > 0:
        first, count = int(model.body_jntadr[body_id]), int(model.body_jntnum[body_id])
        if any(int(model.jnt_type[j]) in (2, 3) for j in range(first, first + count)):
            return True  # MuJoCo slide/hinge; never a free-body carry grant.
        if body_id == root_id:
            break
        body_id = int(model.body_parentid[body_id])
    return False


def fixture_geometry(model, data, fixtures, bounds_fn):
    records = []
    for obj in fixtures:
        root = model.body_name2id(obj.root_body)
        for gid in range(model.ngeom):
            bid = int(model.geom_bodyid[gid])
            if not descendant(model, bid, root):
                continue
            if not (int(model.geom_contype[gid]) or int(model.geom_conaffinity[gid])):
                continue
            bounds = bounds_fn(model, data, gid)
            if bounds is None:
                continue
            lo, hi = [np.asarray(v, dtype=float) for v in bounds]
            name = model.geom_id2name(gid)
            if not name:
                continue
            movable = articulated(model, bid, root)
            native_data = getattr(data, '_data', data)
            body_positions = getattr(native_data, 'xpos', getattr(data, 'body_xpos', None))
            body_rotations = getattr(native_data, 'xmat', getattr(data, 'body_xmat', None))
            records.append({
                "name": f"{obj.name}::geom::{name}",
                "category": getattr(obj, "category_name", "fixture"),
                "geometry_kind": "fixture_contact" if movable else "fixture_static",
                "position": ((lo + hi) / 2).tolist(),
                "aabb_min": lo.tolist(), "aabb_max": hi.tolist(), "dims": (hi-lo).tolist(),
                "contact_object_name": obj.name, "contact_body_name": model.body_id2name(bid),
                "contact_geom_name": name,
                **({'contact_body_position': np.asarray(body_positions[bid]).tolist(),
                    'contact_body_rotation': np.asarray(body_rotations[bid]).reshape(3, 3).tolist()}
                   if body_positions is not None and body_rotations is not None else {}),
            })
    return records


def validate_fixture_geom(model, fixture, authorization):
    """Revalidate the narrow Host grant against the live worker model."""
    root = model.body_name2id(fixture.root_body)
    gid = model.geom_name2id(authorization.get("target_geom_name", ""))
    if root < 0 or gid < 0:
        raise ValueError("Unknown fixture body or collision geom")
    bid = int(model.geom_bodyid[gid])
    if (not descendant(model, bid, root) or not articulated(model, bid, root)
            or model.body_id2name(bid) != authorization.get("target_body_name")
            or not (int(model.geom_contype[gid]) or int(model.geom_conaffinity[gid]))):
        raise ValueError("Fixture contact grant does not match a live articulated collision geom")
    return gid


PATCH_RADIUS_M = .06


def local_patch_fields(record, anchor):
    """Resolve the measured world anchor once; never rebase a persisted grant."""
    position, rotation = record.get('contact_body_position'), record.get('contact_body_rotation')
    if position is None or rotation is None:
        return {}  # Older adapters keep their existing single-geom scope.
    local = np.asarray(rotation).T @ (np.asarray(anchor) - np.asarray(position))
    return {'contact_scope': 'local_articulated_patch',
            'anchor_body_xyz': local.tolist(), 'patch_radius_m': PATCH_RADIUS_M}


def fixture_patch(model, fixture, authorization):
    gid = validate_fixture_geom(model, fixture, authorization)
    if authorization.get('contact_scope') != 'local_articulated_patch':
        return {gid}, None
    local = np.asarray(authorization.get('anchor_body_xyz'), dtype=float)
    if local.shape != (3,) or not np.isfinite(local).all() or authorization.get('patch_radius_m') != PATCH_RADIUS_M:
        raise ValueError('Invalid Host-local fixture contact patch')
    body = int(model.geom_bodyid[gid])
    # Exact same body only: no other articulated joint or fixed frame is exempt.
    geoms = {i for i in range(model.ngeom) if int(model.geom_bodyid[i]) == body
             and (int(model.geom_contype[i]) or int(model.geom_conaffinity[i]))}
    # Keep remote pieces in the ordinary QP avoidance set. Bounding spheres
    # only select candidates; actual contact witnesses enforce the exact patch.
    nearby = {i for i in geoms if np.linalg.norm(np.asarray(model.geom_pos[i])-local)
              <= PATCH_RADIUS_M + float(model.geom_rbound[i])}
    return nearby, {'body_id': body, 'anchor_body_xyz': local,
                    'radius_m': PATCH_RADIUS_M, 'geom_ids': geoms}


def check_fixture_patch(configuration, policy):
    """Check every nearby target/gripper witness and contact, including palm.

    The QP may approach the local surface, while exact pre/post-step checks
    enforce the patch and the unchanged 1 mm hard penetration bound.
    """
    patch = policy.get('fixture_patch')
    if patch is None:
        return {'detected': False}
    import mujoco
    model, data = configuration.model, configuration.data
    mujoco.mj_collision(model, data)
    body = patch['body_id']
    anchor = data.xpos[body] + data.xmat[body].reshape(3, 3) @ patch['anchor_body_xyz']
    target = patch['geom_ids']
    gripper = set(policy['gripper_geom_ids'])
    samples = []
    contact_pairs = {frozenset((int(data.contact[i].geom1), int(data.contact[i].geom2))) for i in range(data.ncon)}
    for g in gripper:
        for t in target:
            witness = np.zeros(6)
            distance = float(mujoco.mj_geomDistance(model, data, g, t, .003, witness))
            # Explicit manifold contacts give reliable witnesses when MuJoCo's
            # orthogonal-box distance degenerates to zero with an empty witness.
            if distance < .003 and (distance < policy['hard_stop_distance_m'] or
                                     frozenset((g, t)) not in contact_pairs):
                if distance == 0 and not np.any(witness):
                    # Bounds can exclude a nearby contact, but cannot certify
                    # its depth or supply a witness for local authorization.
                    overlap, _ = _bounded_patch_overlap(model, data, g, t, patch)
                    if not overlap:
                        continue
                    from sim.convex_distance import certified_convex_separation
                    proof = certified_convex_separation(model, data, g, t,
                        cache=policy.setdefault('_convex_distance_cache', {}))
                    if proof is None:
                        return {'detected': True, 'geom1_id': g, 'geom2_id': t,
                                'contact_scope_violation': 'contact_geometry_uncertain',
                                'collision_type': 'robot_world'}
                    distance, witness = proof
                    if distance < .003:
                        samples.append((g, t, distance, witness[3:]))
                else:
                    samples.append((g, t, distance, witness[3:]))
    for i in range(data.ncon):
        c = data.contact[i]
        g, t = int(c.geom1), int(c.geom2)
        if t in gripper and g in target:
            g, t = t, g
        if g in gripper and t in target:
            samples.append((g, t, float(c.dist), np.asarray(c.pos)))
    for g, t, distance, point in samples:
        violation = ('fixture_penetration' if distance < policy['hard_stop_distance_m'] else
                     'outside_local_patch' if np.linalg.norm(point-anchor) > patch['radius_m'] else None)
        if violation:
            return {'detected': True, 'geom1_id': g, 'geom2_id': t,
                    'minimum_distance_m': distance, 'contact_scope_violation': violation,
                    'collision_type': 'robot_world'}
    return {'detected': False}


def _bounded_patch_overlap(model, data, first, second, patch):
    body = patch['body_id']
    inverse = data.xmat[body].reshape(3, 3).T
    bounds = []
    for geom in (first, second):
        center = inverse @ (data.geom_xpos[geom]-data.xpos[body])
        rotation = inverse @ data.geom_xmat[geom].reshape(3, 3)
        if int(model.geom_type[geom]) == 7:
            mesh = int(model.geom_dataid[geom])
            if mesh >= 0:
                start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
                if count:
                    # Same compiled-vertex transform as UnifiedEnv's scene AABB.
                    vertices = model.mesh_vert[start:start+count] @ rotation.T + center
                    bounds.append((vertices.min(axis=0), vertices.max(axis=0)))
                    continue
        # Box bounds are exact projections; rbound is a conservative fallback
        # for other shapes. No estimated point is supplied to the Agent.
        extent = (np.abs(rotation) @ model.geom_size[geom]
                  if int(model.geom_type[geom]) == 6 else np.full(3, model.geom_rbound[geom]))
        bounds.append((center-extent, center+extent))
    lo = np.maximum(bounds[0][0]-.003, bounds[1][0])
    hi = np.minimum(bounds[0][1]+.003, bounds[1][1])
    if np.any(lo > hi):
        return False, False
    farthest = np.maximum(np.abs(lo-patch['anchor_body_xyz']), np.abs(hi-patch['anchor_body_xyz']))
    return True, bool(np.linalg.norm(farthest) <= patch['radius_m'])


def same_local_patch(first, second):
    if any(first.get(k) != second.get(k) for k in ('target_object_name', 'target_body_name')):
        return False
    if any(grant.get('contact_scope') != 'local_articulated_patch' for grant in (first, second)):
        return False
    a, b = (np.asarray(grant.get('anchor_body_xyz'), dtype=float) for grant in (first, second))
    return bool(a.shape == b.shape == (3,) and np.isfinite(a).all() and np.isfinite(b).all()
                and np.linalg.norm(a-b) <= PATCH_RADIUS_M)
