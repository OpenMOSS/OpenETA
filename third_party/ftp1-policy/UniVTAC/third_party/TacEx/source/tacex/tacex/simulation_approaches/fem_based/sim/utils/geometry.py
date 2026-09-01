import math
import numpy as np
import scipy


def estimate_rigid_transform(P, Q):
    assert P.shape == Q.shape
    n, dim = P.shape
    centeredP = P - P.mean(axis=0)
    centeredQ = Q - Q.mean(axis=0)
    C = np.dot(np.transpose(centeredP), centeredQ) / n

    try:
        V, S, W = np.linalg.svd(C)
        d = np.linalg.det(V) * np.linalg.det(W)
        D = np.eye(3)
        D[2, 2] = d
        R = np.dot(np.dot(V, D), W)
    except Exception as e:
        print(e)
        try:
            V, S, W = scipy.linalg.svd(C, lapack_driver="gesvd")
            d = np.linalg.det(V) * np.linalg.det(W)
            D = np.eye(3)
            D[2, 2] = d
            R = np.dot(np.dot(V, D), W)
        except Exception as e2:
            print(e2)
            R = np.eye(3)

    t = Q.mean(axis=0) - P.mean(axis=0).dot(R)

    return R, t


def quat_product(q1, q2):
    r1 = q1[0]
    r2 = q2[0]
    v1 = np.array([q1[1], q1[2], q1[3]])
    v2 = np.array([q2[1], q2[2], q2[3]])
    r = r1 * r2 - np.dot(v1, v2)
    v = r1 * v2 + r2 * v1 + np.cross(v1, v2)
    q = np.array([r, v[0], v[1], v[2]])

    return q


def transform_mesh(vertices, pos, rot: np.ndarray = np.array([1.0, 0.0, 0.0, 0.0])):
    pos = np.array(pos)
    rot_mat = quat2R_np(rot)
    vertices = vertices @ rot_mat.transpose() + pos
    return vertices


def transform_pts(pts, RT):
    n = pts.shape[0]
    pts = np.concatenate([pts, np.ones((n, 1))], axis=1)
    pts = RT @ pts.T
    pts = pts.T[:, :3]
    return pts


def quat2R_np(q):
    """quaternion to rotation matrix"""
    w = q[0]
    x = q[1]
    y = q[2]
    z = q[3]
    R = np.array([
        [2 * (w * w + x * x) - 1, 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 2 * (w * w + y * y) - 1, 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 2 * (w * w + z * z) - 1],
    ])
    return R


def dist2np(pts_0, pts_1):
    """compute MxN point distance"""
    square_sum0 = np.sum(pts_0**2, axis=1, keepdims=True)
    square_sum1 = np.sum(pts_1**2, axis=1, keepdims=True)
    square_sum = square_sum0 + square_sum1.T
    square_sum -= 2 * pts_0 @ pts_1.T
    return np.sqrt(square_sum + 1e-7)


def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
    coordinates of `M` points in `K` dimensions for which Delaunay triangulation
    will be computed
    """
    from scipy.spatial import Delaunay

    if not isinstance(hull, Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p) >= 0


def vertical_ray_intersects_segment(ray_point, segment_start, segment_end):
    if segment_start[0] == segment_end[0]:
        return False
    segment_slope = (segment_end[1] - segment_start[1]) / (segment_end[0] - segment_start[0])
    segment_b = segment_end[1] - segment_end[0] * segment_slope

    intersect_point = (ray_point[0], ray_point[0] * segment_slope + segment_b)
    if intersect_point[1] >= ray_point[1]:
        intersect_ratio = (ray_point[0] - segment_start[0]) / (segment_end[0] - segment_start[0])
        return 0 <= intersect_ratio < 1
    else:
        return False


def point_in_polygon(point, polygon_points):
    polygon_point_num = len(polygon_points)
    intersect_num = 0
    for i in range(polygon_point_num):
        seg_start = polygon_points[i]
        seg_end = polygon_points[(i + 1) % polygon_point_num]
        if vertical_ray_intersects_segment(point, seg_start, seg_end):
            intersect_num += 1

    return intersect_num % 2 != 0


def generate_rectangle(center, size, theta, rotation_first=False):
    center_x, center_y = center
    x, y = size
    v = np.array([[-x / 2, -y / 2], [x / 2, -y / 2], [x / 2, y / 2], [-x / 2, y / 2]])
    rot = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    if not rotation_first:
        v_rotated = (rot @ v.T).T + np.array([center_x, center_y])
    else:
        v_rotated = (rot @ (v + np.array([center_x, center_y])).T).T
    return v_rotated.tolist()
