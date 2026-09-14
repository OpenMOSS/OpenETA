"""Real MuJoCo witnesses: local surface permissions never exempt whole fixtures."""
from types import SimpleNamespace as NS
import numpy as np
import pytest
from sim.libero_contact_geometry import (fixture_patch, check_fixture_patch,
                                        fixture_geometry, local_patch_fields)


@pytest.fixture
def scene():
    mj = pytest.importorskip('mujoco')
    model = mj.MjModel.from_xml_string('''<mujoco><worldbody>
      <body name="cabinet"><geom name="fixed" type="box" pos="0 0 -.2" size=".2 .1 .02"/>
        <body name="drawer"><joint type="slide" axis="0 1 0"/>
          <geom name="front_left" type="box" pos="-.04 0 0" size=".04 .01 .04"/>
          <geom name="front_right" type="box" pos=".04 0 0" size=".04 .01 .04"/>
          <geom name="far_panel" type="box" pos=".3 0 0" size=".02 .01 .04"/>
        </body>
        <body name="other_drawer" pos="0 0 .2"><joint type="slide" axis="0 1 0"/>
          <geom name="other" type="box" size=".08 .01 .04"/>
        </body>
      </body>
      <body name="gripper" pos=".02 .0198 0"><freejoint/>
        <geom name="finger" type="box" size=".01 .01 .01"/>
      </body>
      </worldbody></mujoco>''')
    data = mj.MjData(model)
    mj.mj_forward(model, data)
    wrapper = NS(ngeom=model.ngeom, geom_bodyid=model.geom_bodyid,
        geom_pos=model.geom_pos, geom_rbound=model.geom_rbound,
        body_parentid=model.body_parentid, body_jntadr=model.body_jntadr,
        body_jntnum=model.body_jntnum, jnt_type=model.jnt_type,
        geom_contype=model.geom_contype, geom_conaffinity=model.geom_conaffinity,
        body_name2id=lambda n: model.body(n).id, body_id2name=lambda i: model.body(i).name,
        geom_name2id=lambda n: model.geom(n).id)
    fixture = NS(root_body='cabinet')
    grant = {'target_geom_name': 'front_left', 'target_body_name': 'drawer',
             **local_patch_fields({'contact_body_position': [0, 0, 0],
                                   'contact_body_rotation': np.eye(3).tolist()}, [0, .01, 0])}
    geoms, patch = fixture_patch(wrapper, fixture, grant)
    policy = {'fixture_patch': patch, 'gripper_geom_ids': [model.geom('finger').id],
              'hard_stop_distance_m': -.001}
    return mj, model, data, wrapper, fixture, grant, geoms, policy


def check(scene, x=.02, y=.0198, drawer_y=0):
    mj, model, data, *_, policy = scene
    data.qpos[0] = drawer_y
    data.qpos[2:5] = [x, y, 0]
    mj.mj_forward(model, data)
    return check_fixture_patch(NS(model=model, data=data), policy)


def test_same_body_adjacent_piece_allowed_but_static_and_other_drawer_not_exempt(scene):
    _, model, _, _, _, _, geoms, _ = scene
    assert model.geom('front_right').id in geoms
    assert model.geom('far_panel').id not in geoms
    assert model.geom('fixed').id not in geoms
    assert model.geom('other').id not in geoms
    assert check(scene)['detected'] is False


def test_far_contact_on_same_body_is_not_authorized(scene):
    r = check(scene, x=.3)
    assert r['detected'] and r['contact_scope_violation'] == 'outside_local_patch'


def test_deep_penetration_in_authorized_patch_is_blocked(scene):
    r = check(scene, y=.015)
    assert r['detected'] and r['contact_scope_violation'] == 'fixture_penetration'


def test_patch_moves_with_articulated_body_without_rebasing(scene):
    assert not check(scene, y=.2198, drawer_y=.2)['detected']
    assert check(scene, x=.3, y=.2198, drawer_y=.2)['contact_scope_violation'] == 'outside_local_patch'


def test_changed_scope_radius_or_body_is_rejected(scene):
    _, _, _, wrapper, fixture, grant, _, _ = scene
    for extra in ({'patch_radius_m': 1.}, {'target_body_name': 'other_drawer'},
                  {'anchor_body_xyz': [float('nan'), 0, 0]}):
        with pytest.raises(ValueError):
            fixture_patch(wrapper, fixture, {**grant, **extra})


def test_zeroed_distance_witness_is_not_mistaken_for_world_origin_contact(scene, monkeypatch):
    mj = scene[0]
    actual_distance = mj.mj_geomDistance
    native_flag = int(mj.mjtDisableBit.mjDSBL_NATIVECCD)
    original_flags = int(scene[1].opt.disableflags)
    def degenerate(model, data, g, t, limit, witness):
        if int(model.opt.disableflags) & native_flag:
            return actual_distance(model, data, g, t, limit, witness)
        witness[:] = 0
        return 0.
    monkeypatch.setattr(mj, 'mj_geomDistance', degenerate)
    # No physical contact yet; all potential local contacts are inside the patch.
    assert not check(scene, y=.0205)['detected']
    # A large separation is proved by nonoverlapping conservative bounds.
    assert not check(scene, y=.2)['detected']
    # Uncertain geometry near a remote surface never obtains authorization.
    assert check(scene, x=.3, y=.0205)['contact_scope_violation'] == 'outside_local_patch'
    assert int(scene[1].opt.disableflags) == original_flags
    assert '_fixture_distance_model' not in scene[-1]


def test_uncertified_convex_solver_does_not_authorize_unknown_local_depth(scene, monkeypatch):
    mj = scene[0]
    def empty(model, data, g, t, limit, witness):
        witness[:] = 0
        return 0.
    monkeypatch.setattr(mj, 'mj_geomDistance', empty)
    import sim.convex_distance as geometry
    monkeypatch.setattr(geometry, 'certified_convex_separation', lambda *args, **kwargs: None)
    # A local bounding-box overlap says nothing about penetration depth.
    assert check(scene, y=.0205)['contact_scope_violation'] == 'contact_geometry_uncertain'
    assert check(scene, y=.2)['detected'] is False
    assert check(scene, y=.015)['contact_scope_violation'] == 'fixture_penetration'
