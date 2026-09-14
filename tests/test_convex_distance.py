import numpy as np
import pytest

from sim.convex_distance import certified_convex_separation


@pytest.fixture
def shapes():
    import mujoco
    model = mujoco.MjModel.from_xml_string('''<mujoco><asset>
      <mesh name="cube" vertex="-.01 -.01 -.01  -.01 -.01 .01  -.01 .01 -.01  -.01 .01 .01
                                .01 -.01 -.01   .01 -.01 .01   .01 .01 -.01   .01 .01 .01"/>
      </asset><worldbody>
        <geom name="a" type="mesh" mesh="cube"/>
        <body pos=".03 0 0"><freejoint/><geom name="b" type="box" size=".01 .01 .01"/></body>
      </worldbody></mujoco>''')
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return mujoco, model, data


def test_separated_hulls_have_verified_witnesses_without_mutating_physics(shapes):
    _, model, data = shapes
    before = (data.qpos.copy(), data.qvel.copy(), data.ctrl.copy(), int(model.opt.disableflags))
    distance, witness = certified_convex_separation(model, data, 0, 1)
    assert distance == pytest.approx(.01, abs=1e-7)
    assert witness[0] == pytest.approx(.01, abs=1e-7)
    assert witness[3] == pytest.approx(.02, abs=1e-7)
    for old, current in zip(before[:3], (data.qpos, data.qvel, data.ctrl)):
        np.testing.assert_array_equal(old, current)
    assert int(model.opt.disableflags) == before[3]


@pytest.mark.parametrize('center', [.015, .02])
def test_overlap_or_exact_touch_never_receives_positive_clearance(shapes, center):
    mj, model, data = shapes
    data.qpos[0] = center
    mj.mj_forward(model, data)
    assert certified_convex_separation(model, data, 0, 1) is None


def test_submicrometre_separation_is_independently_certified(shapes):
    mj, model, data = shapes
    data.qpos[0] = .0200002
    mj.mj_forward(model, data)
    proof = certified_convex_separation(model, data, 0, 1)
    assert proof is not None and 1e-7 < proof[0] < 3e-7


@pytest.mark.parametrize('points', [[100, 0, 0, 101, 0, 0], [-.02, -.01, 0, -.01, .01, 0]])
def test_optimizer_claim_cannot_bypass_membership_or_distance_certificate(shapes, monkeypatch, points):
    import quadprog
    monkeypatch.setattr(quadprog, 'solve_qp', lambda *args: [np.asarray(points)])
    _, model, data = shapes
    assert certified_convex_separation(model, data, 0, 1) is None


def test_mesh_cache_rechecks_vertices_after_model_array_replacement(shapes):
    _, model, data = shapes
    cache = {}
    first = certified_convex_separation(model, data, 0, 1, cache=cache)[0]
    model.mesh_vert[:] *= .5
    second = certified_convex_separation(model, data, 0, 1, cache=cache)[0]
    assert second - first == pytest.approx(.005, abs=1e-7)


def test_solver_failure_stays_uncertified(shapes, monkeypatch):
    import quadprog
    def failure(*args):
        raise ValueError('infeasible')
    monkeypatch.setattr(quadprog, 'solve_qp', failure)
    _, model, data = shapes
    assert certified_convex_separation(model, data, 0, 1) is None


def test_signed_guard_repairs_only_certified_uncorroborated_negative_distance(shapes, monkeypatch):
    from sim.controllers.mink_goal import _signed_geom_pair_distance
    mj, model, data = shapes
    monkeypatch.setattr(mj, 'mj_geomDistance', lambda *args: -.02)
    assert _signed_geom_pair_distance(model, data, 0, 1, contact_distances={}) == pytest.approx(.01, abs=1e-7)
    data.qpos[0] = .015
    mj.mj_forward(model, data)
    # A failed certificate never replaces an intersecting pair's native verdict.
    assert _signed_geom_pair_distance(model, data, 0, 1, contact_distances={}) == -.02


def test_actual_contact_record_is_never_overridden_by_distance_recheck(shapes, monkeypatch):
    from sim.controllers.mink_goal import _signed_geom_pair_distance
    import sim.convex_distance as geometry
    mj, model, data = shapes
    monkeypatch.setattr(mj, 'mj_geomDistance', lambda *args: -.02)
    def forbidden(*args, **kwargs):
        raise AssertionError('Actual contacts must retain the conservative guard')
    monkeypatch.setattr(geometry, 'certified_convex_separation', forbidden)
    assert _signed_geom_pair_distance(model, data, 0, 1,
                                      contact_distances={(0, 1): -.004}) == -.02
