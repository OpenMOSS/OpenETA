"""Per-geom-type world AABBs must not inherit the bounding-sphere inflation.

``geom_rbound`` is a bounding *sphere* radius, so an axis-aligned cube of side
``s`` reported ``s * sqrt(3)`` per axis — a 1.73x over-estimate that propagated
into the carried-object collision box and the receptacle placement corridor.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")

from sim.unified_env import UnifiedEnv

_IDENTITY = np.eye(3, dtype=np.float64)


def _half(
    geom_type,
    size,
    rot=_IDENTITY,
    rbound=1.0,
    *,
    mesh_vertices=None,
):
    vertices = np.asarray(mesh_vertices or [], dtype=np.float64).reshape(-1, 3)
    mesh_data_id = 0 if vertices.size else -1
    model = SimpleNamespace(
        geom_type=np.asarray([int(geom_type)], dtype=np.int32),
        geom_size=np.asarray([size], dtype=np.float64),
        geom_dataid=np.asarray([mesh_data_id], dtype=np.int32),
        geom_rbound=np.asarray([rbound], dtype=np.float64),
        mesh_vertadr=np.asarray([0], dtype=np.int32),
        mesh_vertnum=np.asarray([len(vertices)], dtype=np.int32),
        mesh_vert=vertices,
    )
    data = SimpleNamespace(
        geom_xpos=np.zeros((1, 3), dtype=np.float64),
        geom_xmat=np.asarray([rot], dtype=np.float64),
    )
    bounds = UnifiedEnv._mujoco_geom_world_aabb(model, data, 0)
    if bounds is None:
        return None
    minimum, maximum = bounds
    return (maximum - minimum) / 2.0


def test_axis_aligned_box_is_exact_not_sphere_inflated() -> None:
    half_side = 0.03  # 6 cm cube
    rbound = half_side * math.sqrt(3)
    half = _half(mujoco.mjtGeom.mjGEOM_BOX, [half_side] * 3, rbound=rbound)
    assert np.allclose(half, [half_side] * 3)
    # The old behaviour would have produced the circumscribed-sphere radius.
    assert half[0] < rbound


def test_sphere_uses_first_size_entry_and_is_rotation_invariant() -> None:
    rot = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    half = _half(mujoco.mjtGeom.mjGEOM_SPHERE, [0.04, 0.0, 0.0], rot=rot)
    assert np.allclose(half, [0.04] * 3)


def test_box_rotated_90deg_swaps_extents() -> None:
    rot = np.array(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    half = _half(mujoco.mjtGeom.mjGEOM_BOX, [0.02, 0.05, 0.10], rot=rot)
    assert np.allclose(half, [0.05, 0.02, 0.10])


def test_box_rotated_45deg_grows_but_stays_under_the_sphere_bound() -> None:
    angle = math.pi / 4
    cos_a, sin_a = math.cos(angle), math.sin(angle)
    rot = np.array(
        [[cos_a, -sin_a, 0.0], [sin_a, cos_a, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    size = [0.03, 0.03, 0.05]
    half = _half(mujoco.mjtGeom.mjGEOM_BOX, size, rot=rot)
    expected_xy = (0.03 + 0.03) * cos_a
    assert half[0] == pytest.approx(expected_xy, abs=1e-9)
    assert half[2] == pytest.approx(0.05, abs=1e-9)
    # Still tighter than the circumscribed sphere it replaces.
    assert half[0] < math.sqrt(sum(v * v for v in size))


def test_capsule_adds_end_caps_along_its_axis() -> None:
    radius, half_len = 0.02, 0.06
    half = _half(mujoco.mjtGeom.mjGEOM_CAPSULE, [radius, half_len, 0.0])
    assert np.allclose(half, [radius, radius, half_len + radius])


def test_cylinder_has_flat_faces_without_cap_padding() -> None:
    radius, half_len = 0.02, 0.06
    half = _half(mujoco.mjtGeom.mjGEOM_CYLINDER, [radius, half_len, 0.0])
    assert np.allclose(half, [radius, radius, half_len])


def test_mesh_uses_compiled_vertices_instead_of_bounding_sphere() -> None:
    vertices = [
        [x, y, z]
        for x in (-0.07, 0.07)
        for y in (-0.03, 0.03)
        for z in (-0.02, 0.02)
    ]
    half = _half(
        mujoco.mjtGeom.mjGEOM_MESH,
        [0.0, 0.0, 0.0],
        rbound=0.08,
        mesh_vertices=vertices,
    )
    assert np.allclose(half, [0.07, 0.03, 0.02])


def test_degenerate_geom_is_skipped() -> None:
    assert _half(mujoco.mjtGeom.mjGEOM_MESH, [0.0, 0.0, 0.0], rbound=0.0) is None
    assert _half(mujoco.mjtGeom.mjGEOM_PLANE, [0.0, 0.0, 0.0], rbound=0.0) is None


def test_receptacle_corridor_widens_with_tight_bounds() -> None:
    """A 8 cm cube in a 14 cm bowl: feasible tight, blocked when inflated."""
    side, bowl_span, margin = 0.08, 0.14, 0.005
    tight = bowl_span - side - 2 * margin
    inflated = bowl_span - side * math.sqrt(3) - 2 * margin
    assert tight > 0
    assert inflated < 0
