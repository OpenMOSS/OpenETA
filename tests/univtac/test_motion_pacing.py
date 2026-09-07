import json
import sys
from types import SimpleNamespace

import numpy as np

from scripts.univtac.run_motion_pacing_comparison import freeze_commands
from sim.envs.univtac.autonomous_operation import (
    NativeController,
    paced_reference,
    pose_error,
    rotvec_matrix,
)


def test_pacing_bounds_translation_and_geodesic_rotation():
    start = np.eye(4)
    target = np.eye(4)
    target[:3, 3] = [.03, .04, 0]
    target[:3, :3] = rotvec_matrix([0, 0, .4])
    previous = start
    for step in range(1, 76):
        ref = paced_reference(start, target, step/60, .04, .4)
        dp, dr = pose_error(previous, ref)
        assert np.linalg.norm(dp) <= .04/60 + 1e-10
        assert np.linalg.norm(dr) <= .4/60 + 1e-10
        previous = ref
    np.testing.assert_allclose(previous, target, atol=1e-10)
    assert np.linalg.norm(pose_error(paced_reference(start,target,1/60,.04,.4),target)[0]) > .003


def test_one_mm_request_already_inside_unchanged_tolerance(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace())
    class Array(np.ndarray):
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return np.asarray(self)
    fingers = np.array([.005, .006]).view(Array)
    c = NativeController.__new__(NativeController)
    c.root = tmp_path
    c.config = {'max_control_steps_per_move':80, 'position_tolerance_m':.003, 'orientation_tolerance_rad':.052}
    c.robot = SimpleNamespace(get_gripper_qpos_all=lambda:fingers, _arm_ids=[])
    c.tcp = lambda:np.eye(4)
    c.state = dict
    c.counts = lambda:{'control_steps':c.control_steps, 'physics_steps':0}
    c.terminal = lambda:None
    c.grasp = SimpleNamespace(select=lambda *a:None, command='inherited_hold', targets=np.array([.005,.006]))
    c.control_steps = c.actual_motion_requests = 0
    target = np.eye(4); target[0,3] = .001
    result = c.execute(target, None)
    assert result['arm_reached'] and result['control_steps'] == 0
    assert not result['physical_motion'] and result['counts']['physics_steps'] == 0


def test_freeze_resolved_preview_excludes_rejected_requests(tmp_path):
    pose={'xyz_m':[.5,0,.3], 'approach_world':[0,0,1], 'jaw_world':[1,0,0]}
    rows=[{'tool':'move_to','arguments':{'preview':True},'result':{'text':{'resolved_target':pose}}},
          {'tool':'move_to','arguments':{'execute_preview_id':'old'},'result':{'text':{
              'execution':{'requested_target':pose,'requested_gripper':'close'}}}},
          {'tool':'move_to','arguments':{'delta_mm':[0,0,15]},'result':{'text':{'error':'native_early_stop'}}}]
    (tmp_path/'tool_trace.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    commands=freeze_commands(tmp_path)
    assert len(commands)==1 and commands[0]['source_seq']==2
    assert commands[0]['request']==dict(pose,gripper='close')
    assert 'execute_preview_id' not in commands[0]['request']
