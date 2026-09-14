from types import SimpleNamespace as NS
import numpy as np
import pytest
from sim.libero_contact_geometry import fixture_geometry, validate_fixture_geom
from sim.unified_env import UnifiedEnv
from sim.mcp_server.collision import resolve_contact_authorization


def fixture_scene():
    bodies=['world','cabinet','drawer','other'];geoms=['panel','handle','visual','unrelated']
    m=NS(ngeom=4,body_parentid=[0,0,1,0],body_jntadr=[-1,-1,0,-1],body_jntnum=[0,0,1,0],jnt_type=[2],
        geom_bodyid=[1,2,2,3],geom_contype=[1,1,0,1],geom_conaffinity=[1,1,0,1],
        geom_type=[6]*4,geom_size=np.array([[.1,.01,.2],[.04,.008,.008],[1,1,1],[.01,.01,.01]]),
        body_name2id=bodies.index,body_id2name=lambda i:bodies[i],geom_id2name=lambda i:geoms[i],geom_name2id=geoms.index)
    d=NS(geom_xpos=np.array([[0,0,1],[0,.1,1],[0,.1,1],[2,2,2.]]),geom_xmat=np.array([np.eye(3)]*4))
    f=NS(root_body='cabinet',name='cabinet_1',category_name='wooden_cabinet')
    return m,d,f


def grant(anchor):
    return {'schema_version':'openeta.model_point_contact.v1','source_kind':'model_rgbd_point','session_id':'s',
            'source_packet_id':'p','camera_frame_id':'wrist','point_id':'mark',
            'waypoint_role':'grasp_contact','target_anchor_world_xyz':anchor}


def test_fixture_extraction_uses_live_articulated_geoms_and_tracks_motion():
    m,d,f=fixture_scene()
    env=object.__new__(UnifiedEnv)
    sim=NS(model=m,data=d)
    d.qpos=np.array([]) # A fixed fixture has no free-body qpos to slice.
    env._env=NS(_env=NS(env=NS(objects=[],fixtures=[f]),sim=sim))
    records=env._extract_libero_objects()
    assert len(records)==2
    chosen,receipt=resolve_contact_authorization(grant([0,.1,1]),records)
    assert chosen['contact_geom_name']=='handle'
    assert receipt['target_object_name']=='cabinet_1'
    assert receipt['target_geom_name']=='handle' and receipt['target_body_name']=='drawer'
    assert validate_fixture_geom(m,f,receipt)==1
    d.geom_xpos[1,1]+=.2
    moved=env._extract_libero_objects()
    assert resolve_contact_authorization(grant([0,.1,1]),moved)[0] is None
    assert resolve_contact_authorization(grant([0,.3,1]),moved)[0] is not None


def test_static_cabinet_panel_and_visual_geom_are_not_contact_targets():
    m,d,f=fixture_scene();records=fixture_geometry(m,d,[f],UnifiedEnv._mujoco_geom_world_aabb)
    assert {r['contact_geom_name'] for r in records}=={'panel','handle'}
    assert resolve_contact_authorization(grant([0,0,1]),records)[0] is None
    for name,body in [('panel','cabinet'),('visual','drawer'),('unrelated','other'),('handle','cabinet')]:
        with pytest.raises(ValueError):validate_fixture_geom(m,f,{'target_geom_name':name,'target_body_name':body})


def test_articulated_contact_not_accepted_as_legacy_whole_object_grant():
    m,d,f=fixture_scene();records=fixture_geometry(m,d,[f],UnifiedEnv._mujoco_geom_world_aabb)
    g={**grant([0,.1,1]),'schema_version':'openeta.contact_authorization.v1'}
    assert resolve_contact_authorization(g,records)[0] is None


def test_gripper_fixture_binding_persists_for_pull_and_clears_on_open(monkeypatch):
    from sim.mcp_server import server as s
    m,d,f=fixture_scene()
    objects=fixture_geometry(m,d,[f],UnifiedEnv._mujoco_geom_world_aabb)
    meta={'backend':'libero','_sid':'fixture','remote_handle':'h','worker_url':'test','_collision_objects':objects,
          'control_spec':{'controller':{'goal_executor':'openeta.worker_mink_goal.v1','supports_position':True,'supports_orientation':True}}}
    monkeypatch.setattr(s,'_session_envs',{'fixture':{'h':meta}})
    monkeypatch.setattr(s,'_touch_session',lambda *_:None)
    monkeypatch.setattr(s,'make_gripper_action',lambda *a,**k:[0])
    monkeypatch.setattr(s,'_step_gripper_with_final_observation',lambda *a,**k:{'observation':{'robot':{'gripper_state':{'openness':.3}}}})
    closed=s.gripper_close.__wrapped__(handle='h',session_id='fixture',contact_authorization=grant([0,.1,1]))
    assert closed['attachment_proxy_receipt']['reason']=='articulated_fixture_contact'
    assert '_attachment_proxy' not in meta
    assert meta['_fixture_contact']['target_geom_name']=='handle'
    s.gripper_close.__wrapped__(handle='h',session_id='fixture')
    assert meta['_fixture_contact']['target_geom_name']=='handle'
    monkeypatch.setattr(s,'require_controller_capability',lambda *a,**k:meta['control_spec']['controller'])
    monkeypatch.setattr(s,'cartesian_scales',lambda *a,**k:(.05,.5))
    monkeypatch.setattr(s,'cartesian_command_frame',lambda *a,**k:'world')
    sent=[]
    monkeypatch.setattr(s,'_proxy_controller_goal',lambda meta,body:sent.append(body) or {'ok':True,'steps_executed':1})
    # The controller body receives the narrow contact even though a pull move
    # supplies no new pixel. The worker separately revalidates this geom.
    s.move_to.__wrapped__(handle='h',session_id='fixture',x=0,y=.2,z=1)
    assert sent[-1]['contact_authorization']['target_geom_name']=='handle'
    assert 'attachment_proxy' not in sent[-1]
    s.gripper_open.__wrapped__(handle='h',session_id='fixture')
    assert '_fixture_contact' not in meta
    s.move_to.__wrapped__(handle='h',session_id='fixture',x=0,y=.2,z=1)
    assert 'contact_authorization' not in sent[-1]
    s.gripper_close.__wrapped__(handle='h',session_id='fixture',contact_authorization=grant([0,.1,1]))
    monkeypatch.setattr(s,'_proxy_reset',lambda *a,**k:{})
    monkeypatch.setattr(s,'_settle_env',lambda *a,**k:{})
    s.reset_env.__wrapped__(handle='h',session_id='fixture',seed=0)
    assert '_fixture_contact' not in meta and meta['_gripper_cmd']==-1.0


def test_failed_close_cannot_arm_fixture_contact(monkeypatch):
    from sim.mcp_server import server as s
    m,d,f=fixture_scene()
    meta={'backend':'libero','_collision_objects':fixture_geometry(m,d,[f],UnifiedEnv._mujoco_geom_world_aabb)}
    monkeypatch.setattr(s,'_session_envs',{'fixture':{'h':meta}})
    monkeypatch.setattr(s,'_touch_session',lambda *_:None)
    monkeypatch.setattr(s,'make_gripper_action',lambda *a,**k:[0])
    monkeypatch.setattr(s,'_step_gripper_with_final_observation',lambda *a,**k:{'error':'actuator failed'})
    result=s.gripper_close.__wrapped__(handle='h',session_id='fixture',contact_authorization=grant([0,.1,1]))
    assert result['error']=='actuator failed' and '_fixture_contact' not in meta


def test_adjacent_fixture_pieces_are_one_association_but_different_bodies_stay_ambiguous(monkeypatch):
    monkeypatch.setenv('OPENETA_LIBERO_FIXTURE_CONTACT_PATCH', '1')
    from copy import deepcopy
    m,d,f=fixture_scene()
    records=fixture_geometry(m,d,[f],UnifiedEnv._mujoco_geom_world_aabb)
    first=next(r for r in records if r['geometry_kind']=='fixture_contact')
    adjacent={**deepcopy(first), 'name':'adjacent','contact_geom_name':'adjacent'}
    assert resolve_contact_authorization(grant([0,.1,1]),[first,adjacent])[0] is not None
    other={**adjacent,'contact_body_name':'other_drawer'}
    assert resolve_contact_authorization(grant([0,.1,1]),[first,other])[1]['code']=='contact_target_geometry_ambiguous'


def test_same_patch_requires_same_object_and_body_and_bounded_anchor():
    from sim.libero_contact_geometry import same_local_patch
    first={'target_object_name':'cab','target_body_name':'drawer','contact_scope':'local_articulated_patch',
           'anchor_body_xyz':[0,0,0],'target_geom_name':'piece1'}
    assert same_local_patch(first,{**first,'target_geom_name':'piece2','anchor_body_xyz':[.01,0,0]})
    for changed in ({'target_body_name':'other'}, {'target_object_name':'other'}, {'anchor_body_xyz':[.3,0,0]}):
        assert not same_local_patch(first,{**first,**changed})


@pytest.mark.parametrize('enabled', [False, True])
def test_resolver_only_emits_local_patch_in_opt_in_mode(monkeypatch, enabled):
    monkeypatch.setenv('OPENETA_LIBERO_FIXTURE_CONTACT_PATCH', '1' if enabled else '0')
    m,d,f=fixture_scene()
    records=fixture_geometry(m,d,[f],UnifiedEnv._mujoco_geom_world_aabb)
    for r in records:
        r.update(contact_body_position=[0,0,0],contact_body_rotation=np.eye(3).tolist())
    _, receipt=resolve_contact_authorization(grant([0,.1,1]),records)
    assert ('anchor_body_xyz' in receipt) is enabled
    assert (receipt.get('contact_scope')=='local_articulated_patch') is enabled
