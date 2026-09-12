"""Sensor-only geometry for the experimental atomic Codex profile.

Design reference: OpenETA-Light ddf900c, pointcloud_pose_marking.py. No learned
perception, simulator object poses, segmentation or object meshes are read.
"""
from __future__ import annotations

import base64
import io

import numpy as np
from PIL import Image, ImageDraw
from agent.tools.grasp_geometry import _opencv_camera_to_world, _eef_rotation


# LIBERO Panda's observable/IK/controller quaternion describes right_hand,
# while grip_site (and the physical jaw axis) is rotated -90 deg about local Z.
# This fixed robot calibration is verified against the loaded MuJoCo model;
# it is not a scene/object pose. Keep the shared controller body-frame contract.
PANDA_BODY_TO_GRIP_SITE = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])


def body_to_grip_site(rotation):
    return np.asarray(rotation) @ PANDA_BODY_TO_GRIP_SITE


def grip_site_to_body(rotation):
    return np.asarray(rotation) @ PANDA_BODY_TO_GRIP_SITE.T


def vector(value, size=3):
    result = np.asarray(value, dtype=float)
    if result.shape != (size,) or not np.isfinite(result).all():
        raise ValueError(f"Expected {size} finite numbers")
    return result


def unit(value):
    result = vector(value)
    length = np.linalg.norm(result)
    if length < 1e-6:
        raise ValueError("Direction is zero or parallel to the other axis")
    return result / length


def orientation(current_quat, approach=None, jaw=None):
    """Local +Z points along approach, +X along jaws, +Y completes the frame."""
    current = quat_matrix(current_quat)
    if approach is None and jaw is None:
        return current
    z = unit(current[:, 2] if approach is None else approach)
    x = unit(current[:, 0] if jaw is None else jaw)
    if approach is None and jaw is not None:
        z = unit(z - x * np.dot(z, x))
    else:
        x = unit(x - z * np.dot(x, z))
    return np.column_stack((x, np.cross(z, x), z))


def quat_matrix(quat):
    return np.asarray(_eef_rotation({"quat_xyzw": vector(quat, 4).tolist()})[0])


def matrix_quat(matrix):
    # Choose the largest quaternion component to remain stable near 180 deg.
    r = np.asarray(matrix)
    candidates = np.array([1+r[0,0]-r[1,1]-r[2,2], 1-r[0,0]+r[1,1]-r[2,2],
                           1-r[0,0]-r[1,1]+r[2,2], 1+np.trace(r)])
    i = int(np.argmax(candidates))
    q = np.zeros(4)
    q[i] = np.sqrt(max(0., candidates[i])) / 2
    denom = 4*q[i]
    if i == 3:
        q[:3] = [r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1]]
        q[:3] /= denom
    else:
        j, k = (i+1)%3, (i+2)%3
        q[j] = (r[j,i]+r[i,j])/denom
        q[k] = (r[k,i]+r[i,k])/denom
        q[3] = (r[k,j]-r[j,k])/denom
    return (q/np.linalg.norm(q)).tolist()


def calibration(source):
    intrinsics = source["intrinsics"]
    values = [float(intrinsics[k]) for k in ("fx", "fy", "cx", "cy")]
    if not np.isfinite(values).all() or min(values[:2]) <= 0:
        raise ValueError("Invalid camera intrinsics")
    rotation, translation = _opencv_camera_to_world(source["extrinsics"])
    return values, np.asarray(rotation), np.asarray(translation)


def surface_point(source, x, y):
    image = Image.open(source["rgb"])
    depth = np.asarray(Image.open(source["depth"]), dtype=float)
    if depth.shape != (image.height, image.width):
        raise ValueError("RGB and depth are not aligned")
    if not (0 <= x < image.width and 0 <= y < image.height):
        raise ValueError("Pixel is outside the source image")
    # Retained simulator depth PNGs are encoded in millimetres unless the
    # source explicitly declares another units-per-metre scale.
    scale = float(source["intrinsics"].get("scale", 1000))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Invalid depth scale")
    z = float(depth[y, x] / scale)
    if not np.isfinite(z) or not 0 < z < 10:
        raise ValueError("No valid visible surface depth at this pixel")
    patch = depth[max(0, y-1):y+2, max(0, x-1):x+2] / scale
    valid = patch[np.isfinite(patch) & (patch > 0)]
    spread = float(np.ptp(valid)) if len(valid) else 0.
    (fx, fy, cx, cy), rotation, translation = calibration(source)
    xyz = rotation @ np.array([(x-cx)*z/fx, (y-cy)*z/fy, z]) + translation
    return xyz.tolist(), {"depth_m": z, "local_depth_spread_m": spread,
                        "depth_edge": spread > .02, "image_size": list(image.size)}


def project(source, xyz):
    (fx, fy, cx, cy), rotation, translation = calibration(source)
    p = rotation.T @ (vector(xyz) - translation)
    if p[2] <= 1e-6:
        return None
    return (float(fx*p[0]/p[2]+cx), float(fy*p[1]/p[2]+cy))


def render(source, *, mark=None, actual=None, target=None, rotation=None, crop=None):
    """Calibrated overlays; their pixels still refer to the original depth."""
    im = Image.open(source["rgb"]).convert("RGB")
    draw = ImageDraw.Draw(im)
    def dot(xyz, color, radius=3):
        xy = project(source, xyz)
        if xy is not None:
            x, y = xy
            draw.ellipse((x-radius,y-radius,x+radius,y+radius), outline=color, width=1)
            draw.point((x,y), fill=color)
        return xy
    if actual is not None:
        dot(actual, "magenta")
    if mark is not None:
        dot(mark, "cyan")
    if target is not None:
        center = dot(target, "yellow")
        if center is not None and rotation is not None:
            # Jaw span illustrates the robot's 80 mm maximum opening; it is
            # a projection, not a collision or grasp-quality prediction.
            ends = [project(source, vector(target)+rotation[:,0]*s) for s in (-.04,.04)]
            if all(v is not None for v in ends):
                draw.line((*ends[0],*ends[1]), fill="orange", width=2)
            approach = project(source, vector(target)-rotation[:,2]*.05)
            if approach is not None:
                draw.line((*approach,*center), fill="lime", width=2)
    if crop is not None:
        x0,y0,x1,y1 = crop
        if not (0 <= x0 < x1 <= im.width and 0 <= y0 < y1 <= im.height):
            raise ValueError("Crop is outside source image")
        im = im.crop(crop)
        im = im.resize((im.width*2, im.height*2))
    stream = io.BytesIO()
    im.save(stream, format="PNG")
    return base64.b64encode(stream.getvalue()).decode()
