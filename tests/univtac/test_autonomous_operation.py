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
