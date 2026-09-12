"""Host-private, live geometry for LIBERO articulated fixture contact.

Each contact candidate is one collision geom on a jointed fixture body.
Fixed cabinet panels remain obstacles, never whole-cabinet exemptions.
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
            records.append({
                "name": f"{obj.name}::geom::{name}",
                "category": getattr(obj, "category_name", "fixture"),
                "geometry_kind": "fixture_contact" if movable else "fixture_static",
                "position": ((lo + hi) / 2).tolist(),
                "aabb_min": lo.tolist(), "aabb_max": hi.tolist(), "dims": (hi-lo).tolist(),
                "contact_object_name": obj.name, "contact_body_name": model.body_id2name(bid),
                "contact_geom_name": name,
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
