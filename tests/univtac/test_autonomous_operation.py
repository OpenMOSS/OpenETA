from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sim.envs.univtac.autonomous_operation import (
    NativeController,
    damped_joint_delta,
    resolve_target,
    rotvec_matrix,
)
from sim.envs.univtac.autonomous_session import AutonomousSession


def test_position_units_and_tcp_rotation():
    current = np.eye(4)
    current[:3,3] = [0.4,0.2,0.3]
    current[:3,:3] = rotvec_matrix([0,0,np.pi/2])
    target = resolve_target(current,{'delta_mm':[10,0,0],'delta_frame':'grip_site'})
    np.testing.assert_allclose(target[:3,3],[0.4,0.21,0.3])
    rotated = resolve_target(current,{'approach_world':[0,1,0],'jaw_world':[1,0,0]})
    np.testing.assert_allclose(rotated[:3,3],current[:3,3])
    np.testing.assert_allclose(rotated[:3,2],[0,1,0])
    assert np.linalg.det(rotated[:3,:3]) == pytest.approx(1)
    np.testing.assert_allclose(resolve_target(current,{'gripper':'open'}),current)


@pytest.mark.parametrize('command',[
    {'xyz_m':[1,2,3],'delta_mm':[1,2,3]}, {'delta_mm':[1,2]},
    {'approach_world':[0,0,0]}, {'approach_world':[0,0,1],'jaw_world':[0,0,2]},
    {'gripper':'half'}, {'delta_mm':[1,2,3],'delta_frame':'object'}, {'seed':1000003},
])
def test_rejects_unsupported_commands(command):
    with pytest.raises(ValueError):
        resolve_target(np.eye(4),command)


def test_ik_delta_bounded_and_reduces_error():
    jac = np.c_[np.eye(6),np.zeros(6)]
    error = np.array([0.01,0,0,0,0,0.1])
    delta = damped_joint_delta(jac,error,0.025)
    assert np.abs(delta).max() <= 0.025
    assert np.linalg.norm(error-jac@delta) < np.linalg.norm(error)


def test_native_success_not_gated_by_planner_flag(tmp_path):
    task = SimpleNamespace(_robot_manager=None,step_count=12,_physics_step_count=12,take_action_cnt=0,
                           cfg=SimpleNamespace(step_lim=300,sim=SimpleNamespace(dt=1/120)),
                           eval_success=False,plan_success=False,check_success=lambda: True)
    controller = NativeController(task,{},tmp_path)
    assert controller.check()=={'available':True,'success':True}
    assert controller.terminal()=='native_success'
    task.check_success=None
    controller.success_available=False
    assert controller.check()=={'available':False,'success':None}


def test_session_rejected_request_counts_and_preview_does_not(tmp_path):
    task=SimpleNamespace(_robot_manager=None,step_count=0,_physics_step_count=0,take_action_cnt=0,
                         cfg=SimpleNamespace(step_lim=300,sim=SimpleNamespace(dt=1/120)),eval_success=False)
    session=AutonomousSession(task,{'max_tool_calls':100,'max_move_requests':30},tmp_path,1000003)
    session.controller.tcp=lambda:np.eye(4)
    session.latest={'observation_id':'initial'}
    session.capture=lambda:{'observation_id':'fresh','images':[]}
    preview=session.call('move_to',{'delta_mm':[10,0,0],'preview':True})
    assert preview['ok'] and session.move_requests==0
    bad=session.call('move_to',{'xyz_m':[1,2,3],'delta_mm':[1,2,3]})
    assert not bad['ok'] and session.move_requests==1
    assert bad['text']['observation']['observation_id']=='fresh'
    assert session.previews=={}


def test_backend_tool_surface():
    from tools.embodied_mcp_server import build_live_backend_server
    server=build_live_backend_server(root=Path('/tmp'),worker_url='http://127.0.0.1:1')
    assert set(server._tool_manager._tools)=={'observe','mark_point','move_to','report_issue','check_task','finish_episode'}
    schema=server._tool_manager._tools['move_to'].parameters
    assert {'xyz_m','delta_mm','approach_world','jaw_world','gripper','preview'} <= set(schema['properties'])


def test_control_path_uses_native_steps_without_expert():
    import ast
    path=Path('sim/envs/univtac/autonomous_operation.py')
    tree=ast.parse(path.read_text())
    attrs={n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute)}
    assert not attrs & {'play_once','expert_check','plan_arm','set_dof_positions','set_pose','prism','slot','target_pose'}
    calls=[n for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='take_action']
    assert len(calls)==1
    values={k.arg:ast.literal_eval(k.value) for k in calls[0].keywords}
    assert values['force'] is False and values['action_type']=='qpos'


def test_native_control_budget_counts_each_waypoint(tmp_path, monkeypatch):
    import sys

    class Array(np.ndarray):
        def detach(self):
            return self
        def cpu(self):
            return self
        def numpy(self):
            return np.asarray(self)
    def array(x):
        return np.asarray(x).view(Array)
    monkeypatch.setitem(sys.modules,'torch',SimpleNamespace(tensor=lambda x,**_:array(x),float32='float32'))
    data=SimpleNamespace(joint_pos=array(np.zeros((1,7))),
        soft_joint_pos_limits=array(np.tile([-3,3],(1,7,1))),body_link_pos_w=array(np.zeros((1,1,3))))
    physx=SimpleNamespace(get_jacobians=lambda:array(np.c_[np.eye(6),np.zeros(6)][None,None]))
    robot=SimpleNamespace(robot=SimpleNamespace(data=data,root_physx_view=physx),_arm_ids=np.arange(7),
        _jacobi_body_idx=0,_body_idx=0,get_gripper_qpos=lambda:0)
    task=SimpleNamespace(_robot_manager=robot,step_count=20,_physics_step_count=20,take_action_cnt=0,
        cfg=SimpleNamespace(step_lim=2,sim=SimpleNamespace(dt=1/120)),eval_success=False,device='cpu',
        check_success=lambda:False,check_early_stop=lambda:False)
    calls=[]
    def take_action(action,**kwargs):
        calls.append(kwargs)
        data.joint_pos[0]=action[:7]
        data.body_link_pos_w[0,0]=action[:3]
        task.take_action_cnt+=1
        task.step_count+=1
        task._physics_step_count+=2
    task.take_action=take_action
    c=NativeController(task,{'max_control_steps_per_move':80,'position_tolerance_m':.003,
        'orientation_tolerance_rad':.052,'max_joint_delta_rad':.025},tmp_path)
    def tcp():
        m=np.eye(4);m[:3,3]=data.joint_pos[0,:3];return m
    c.tcp=tcp
    c.state=lambda:{'xyz_m':tcp()[:3,3].tolist()}
    target=np.eye(4);target[0,3]=.5
    result=c.execute(target,None)
    assert len(calls)==2 and all(k['force'] is False for k in calls)
    assert c.control_steps==task.take_action_cnt==2
    assert c.counts()['physics_steps']==4
    assert result['error']=='native_step_limit' and not result['reached']


def test_camera_pose_refreshed_without_physics(tmp_path, monkeypatch):
    import sim.envs.univtac.autonomous_session as module
    task=SimpleNamespace(_robot_manager=None,step_count=1,_physics_step_count=1,take_action_cnt=0,
        cfg=SimpleNamespace(step_lim=300,sim=SimpleNamespace(dt=1/120)),eval_success=False)
    cameras={}
    refresh=[]
    for name in ['head','wrist']:
        data=SimpleNamespace(output={'depth':np.ones((1,2,2))},intrinsic_matrices=np.array([[[2,0,1],[0,2,1],[0,0,1]]]),
            quat_w_ros=np.array([[1,0,0,0]]),pos_w=np.zeros((1,3)))
        cam=SimpleNamespace(data=data,_ALL_INDICES=[0])
        def update(ids,cam=cam,name=name):
            refresh.append(name);cam.data.pos_w[0,0]+=1
        cam._update_poses=update
        cameras[name]=cam
    task._camera_manager=SimpleNamespace(cameras=cameras)
    task._get_observations=lambda:{'observation':{n:{'rgb':np.zeros((2,2,3))} for n in cameras}}
    visible={'cameras':{n:{'rgb':{'path':n+'.png'}} for n in cameras},
             'tactile':{n:{'rgb_marker':{'path':n+'.png'}} for n in ['left_tactile','right_tactile']}}
    monkeypatch.setattr(module,'capture_snapshot',lambda *a,**k:SimpleNamespace(snapshot=SimpleNamespace(operator_visible=visible,to_dict=dict)))
    s=AutonomousSession(task,{'max_move_requests':30,'max_tool_calls':100},tmp_path,1000003)
    s.controller.state=dict
    s.capture();s.capture()
    assert refresh==['head','head']
    assert s.frames[0]['metadata']['extrinsics']['pos']==[2,0,0]
    assert not s.latest['mark_point']['wrist']['available']
    assert task._physics_step_count==1


def test_r14_dashboard_keeps_debug_separate(tmp_path):
    import json

    from scripts.univtac.autonomous_dashboard import load_autonomous_runs
    p=tmp_path/'univtac-isaac51-r14-debug';p.mkdir()
    (p/'run_manifest.json').write_text(json.dumps({'round':'R1.4','mode':'debug'}))
    e=p/'seed_1000003';e.mkdir()
    (e/'episode.json').write_text(json.dumps({'scored':False,'seed':1000003}))
    d=load_autonomous_runs(tmp_path)
    assert d['batches'][0]['manifest']['mode']=='debug'
    assert d['batches'][0]['episodes'][0]['context']==[]


def test_native_terminal_allows_process_to_finish_and_report(tmp_path):
    import os
    import sys

    from scripts.univtac.run_codex_readonly_observation import _run_to_files
    stop=tmp_path/'stop.json';stop.write_text('{}')
    result=_run_to_files([sys.executable,'-c','print("final usage returned")'],cwd=tmp_path,
        environment=dict(os.environ),stdout_path=tmp_path/'out',stderr_path=tmp_path/'err',
        timeout_seconds=5,stop_path=stop)
    assert result['returncode']==0 and not result['stopped_by_worker']
    assert 'final usage returned' in (tmp_path/'out').read_text()
