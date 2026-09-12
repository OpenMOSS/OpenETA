from types import SimpleNamespace as NS
import numpy as np
import pytest
from sim.controllers import gripper_guard as guard


def rig(monkeypatch, start=0.):
    mj=pytest.importorskip('mujoco');pytest.importorskip('mink')
    m=mj.MjModel.from_xml_string('''<mujoco><worldbody>
      <body><joint type="slide" axis="1 0 0"/><geom name="finger" type="sphere" size="0.01"/></body>
      <geom name="obstacle" type="sphere" pos="0.03 0 0" size="0.01"/>
      </worldbody></mujoco>''')
    d=mj.MjData(m);d.qpos[0]=start;mj.mj_forward(m,d)
    finger=m.geom('finger').id;obstacle=m.geom('obstacle').id
    raw=NS(sim=NS(model=NS(_model=m),data=d))
    monkeypatch.setattr(guard,'_libero_runtime',lambda _: (raw,None))
    def policy(*_,contact_authorization,**kwargs):
        pairs=[] if contact_authorization else [(finger,obstacle)]
        return {'protected_pairs':pairs,'hard_stop_distance_m':-.001,
                'robot_geom_ids':[finger],'gripper_geom_ids':[finger],
                'world_geom_count':1,'world_object_count':1,'protected_pair_count':len(pairs),
                'authorized_target_geom_count':1 if contact_authorization else 0}
    monkeypatch.setattr(guard,'_libero_collision_policy',policy)
    calls=[]
    def step(action,render):
        calls.append(render);d.qpos[0]+=.004*action[-1];mj.mj_forward(m,d)
        return {'reward':0.,'terminated':False,'observation':{'robot':{}}}
    return step,calls


def test_close_stops_on_actual_protected_contact_without_rendering(monkeypatch):
    step,calls=rig(monkeypatch)
    r=guard.execute_checked_gripper(None,action=[0]*7+[1],max_steps=60,
        contact_authorization=None,step_callback=step)
    assert r['stop_reason']=='collision_detected' and r['steps_executed']==3
    assert r['collision']['collision_type']=='robot_world'
    assert r['collision']['robot_part']=='gripper'
    assert r['collision']['prediction_checked'] is False
    assert calls==[False]*3


def test_existing_contact_blocks_close_but_allows_checked_release(monkeypatch):
    step,calls=rig(monkeypatch,start=.015)
    r=guard.execute_checked_gripper(None,action=[0]*7+[1],max_steps=60,
        contact_authorization=None,step_callback=step)
    assert r['steps_executed']==0 and not calls
    r=guard.execute_checked_gripper(None,action=[0]*7+[-1],max_steps=4,
        contact_authorization=None,step_callback=step)
    assert r['gripper_horizon_completed'] and r['steps_executed']==4


def test_authorized_contact_does_not_become_false_obstacle(monkeypatch):
    step,calls=rig(monkeypatch)
    r=guard.execute_checked_gripper(None,action=[0]*7+[1],max_steps=4,
        contact_authorization={'target':'bound'},step_callback=step)
    assert r['gripper_horizon_completed'] and len(calls)==4


def test_camera_observables_are_restored_even_on_failure():
    class Observable:
        def __init__(self,enabled):self.enabled=enabled
        def is_enabled(self):return self.enabled
        def set_enabled(self,enabled):self.enabled=enabled
    image,depth,joints=Observable(True),Observable(False),Observable(True)
    raw=NS(env=NS(_observables={'wrist_image':image,'wrist_depth':depth,'joint_pos':joints}))
    with pytest.raises(RuntimeError):
        with guard.suspend_camera_observables(raw):
            assert not image.enabled and not depth.enabled and joints.enabled
            raise RuntimeError('physics failed')
    assert image.enabled and not depth.enabled and joints.enabled
