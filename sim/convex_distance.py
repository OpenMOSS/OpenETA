"""Certify separation when a native convex distance query has no witness.

Worker-private geometry only. A regularized closest-point QP proposes points;
membership and a separating support plane independently certify the result.
Unsupported, intersecting or numerically uncertain geometry returns None.
"""
import itertools
import numpy as np

CERTIFICATE_EPSILON_M = 1e-7
MINIMUM_CERTIFIED_GAP_M = 1e-9
MAX_MESH_VERTICES = 512
MAX_EDGE_AXIS_PAIRS = 4096


def _mesh(model, geom, cache):
    from scipy.spatial import ConvexHull, QhullError

    mesh = int(model.geom_dataid[geom])
    if mesh < 0:
        return None
    start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
    if not 4 <= count <= MAX_MESH_VERTICES:
        return None
    vertices = model.mesh_vert[start:start + count].astype(float)
    if not np.isfinite(vertices).all():
        return None
    entry = cache.get(mesh)
    # Restoring a private snapshot can replace model arrays in place.
    if entry is None or not np.array_equal(entry['vertices'], vertices):
        try:
            hull = ConvexHull(vertices)
        except QhullError:
            return None
        indices = sorted({tuple(sorted((face[i], face[j])))
                          for face in hull.simplices for i, j in ((0, 1), (1, 2), (2, 0))})
        entry = {'vertices': vertices, 'faces': hull.equations,
                 'edges': np.asarray([vertices[i] - vertices[j] for i, j in indices])}
        cache[mesh] = entry
    return entry


def _polytope(model, data, geom, reference, cache):
    kind = int(model.geom_type[geom])
    if kind == 6:  # box
        size = np.asarray(model.geom_size[geom], dtype=float)
        if not np.isfinite(size).all() or np.any(size <= 0):
            return None
        vertices = np.asarray(list(itertools.product((-1, 1), repeat=3))) * size
        normals = np.vstack((np.eye(3), -np.eye(3)))
        offsets = -np.tile(size, 2)
        edges = np.eye(3)
    elif kind == 7:  # MuJoCo's compiled convex collision mesh
        mesh = _mesh(model, geom, cache)
        if mesh is None:
            return None
        vertices, edges = mesh['vertices'], mesh['edges']
        normals, offsets = mesh['faces'][:, :3], mesh['faces'][:, 3]
    else:
        return None
    rotation = data.geom_xmat[geom].reshape(3, 3)
    center = data.geom_xpos[geom] - reference
    if not np.isfinite(rotation).all() or not np.isfinite(center).all():
        return None
    matrix = normals @ rotation.T
    return {'vertices': vertices @ rotation.T + center,
            'matrix': matrix, 'offset': offsets - matrix @ center,
            'edges': edges @ rotation.T}


def _support_gap(first, second, axes):
    norms = np.linalg.norm(axes, axis=1)
    axes = axes[norms > 1e-10] / norms[norms > 1e-10, None]
    if not len(axes):
        return -np.inf
    a = first['vertices'] @ axes.T
    b = second['vertices'] @ axes.T
    # Both signs of every tested axis are valid separating-plane candidates.
    return float(np.maximum(b.min(axis=0) - a.max(axis=0),
                            a.min(axis=0) - b.max(axis=0)).max())


def _certified(first, second, points):
    epsilon = CERTIFICATE_EPSILON_M
    if np.asarray(points).shape != (6,) or not np.isfinite(points).all():
        return False
    for shape, point in ((first, points[:3]), (second, points[3:])):
        if np.any(shape['matrix'] @ point + shape['offset'] > epsilon):
            return False
    delta = points[3:] - points[:3]
    distance = float(np.linalg.norm(delta))
    if distance <= 1e-12:
        return False
    axes = np.vstack((delta.reshape(1, 3) / distance, first['matrix'], second['matrix']))
    gap = _support_gap(first, second, axes)
    if gap > MINIMUM_CERTIFIED_GAP_M and abs(distance - gap) <= epsilon:
        return True
    # Nearly touching edges need an edge-cross-edge axis; the cap bounds work.
    if len(first['edges']) * len(second['edges']) > MAX_EDGE_AXIS_PAIRS:
        return False
    axes = np.cross(first['edges'][:, None, :], second['edges'][None, :, :]).reshape(-1, 3)
    gap = _support_gap(first, second, axes)
    return bool(gap > MINIMUM_CERTIFIED_GAP_M and abs(distance - gap) <= epsilon)


def certified_convex_separation(model, data, first, second, *, cache=None):
    """Return (positive distance, verified world witnesses), or no certificate.

    No model flags, physics state, contact grants or motion targets are changed.
    Solver success alone never authorizes clearance or a local contact patch.
    """
    import quadprog

    cache = {} if cache is None else cache
    reference = np.asarray(data.geom_xpos[second]).copy()
    a = _polytope(model, data, first, reference, cache)
    b = _polytope(model, data, second, reference, cache)
    if a is None or b is None:
        return None
    count = len(a['matrix'])
    matrix = np.zeros((count + len(b['matrix']), 6))
    matrix[:count, :3], matrix[count:, 3:] = a['matrix'], b['matrix']
    offsets = np.concatenate((a['offset'], b['offset']))
    regularization = 1e-8
    hessian = np.block([[np.eye(3), -np.eye(3)], [-np.eye(3), np.eye(3)]])
    hessian += np.eye(6) * regularization
    initial = np.concatenate((data.geom_xpos[first] - reference, np.zeros(3)))
    try:
        points = quadprog.solve_qp(hessian, initial * regularization, -matrix.T, offsets)[0]
    except ValueError:
        return None
    if not _certified(a, b, points):
        return None
    witness = points + np.tile(reference, 2)
    return float(np.linalg.norm(points[3:] - points[:3])), witness
