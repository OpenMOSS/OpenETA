import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from PIL import Image

from sim.envs.univtac.autonomous_operation import GripperTargets
from sim.envs.univtac.tactile_history import (
    VIEWS,
    TactileRecorder,
    export_review_video,
    motion_features,
    select_times,
)


class Array(np.ndarray):
    @property
    def device(self):
        return 'cpu'

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.asarray(self)


def arr(x):
    return np.asarray(x).view(Array)


def cfg():
    return yaml.safe_load(Path('configs/univtac/autonomous_insert_hole.yaml').read_text())['tactile_history']


def test_pair_is_last_submitted_command_and_survives_scalar_dispatch(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, 'torch', SimpleNamespace(tensor=lambda x,**k:arr(x)))
    data = SimpleNamespace(joint_pos_target=arr([[0,.011,.013]]), joint_stiffness=arr([[0,2000,2000]]),
                           joint_damping=arr([[0,100,100]]), joint_effort_limits=arr([[0,200,200]]))
    robot = SimpleNamespace(_gripper_ids=[1,2], _gripper_joint_names=['left','right'],gripper_max_qpos=.039,
        robot=SimpleNamespace(data=data, _joint_pos_target_sim=arr([[0,.011,.013]])),
        get_gripper_qpos_all=lambda:arr([.018,.021]))
    writes=[]
    def write(target, **kw):
        writes.append((target.tolist(),kw))
        data.joint_pos_target[0,1:]=target
    robot.set_gripper=write
    pair=GripperTargets(robot,tmp_path)
    np.testing.assert_allclose(pair.targets,[.011,.013])
    calls=[]
    def native(action,**kw):
        calls.append(kw)
        robot.set_gripper(action[-1],force=kw['force'])
    for command,expected in [(None,[.011,.013]),('close',[0,0]),(None,[0,0]),('open',[.039,.039]),(None,[.039,.039])]:
        pair.select(command,{})
        pair.dispatch(arr([0,123]),native)
        np.testing.assert_allclose(writes[-1][0],expected)
        assert writes[-1][1]['force'] is False
    assert len(calls)==5 and robot.set_gripper is write


def test_observed_patch_motion_and_low_quality_fallback():
    rng=np.random.default_rng(11)
    a=rng.integers(0,255,(120,160,3),dtype=np.uint8)
    b=np.roll(a,2,axis=1)
    f=motion_features(a,b,cfg())
    assert f['tracking_reliable'] and f['motion_change_px']==pytest.approx(2)
    flat=motion_features(np.zeros_like(a),np.full_like(a,20),cfg())
    assert not flat['tracking_reliable'] and flat['selection_source']=='image_difference_fallback'
    assert flat['image_difference']>0


def test_segment_peak_is_causal_sorted_and_shared():
    rows=[{'sample_id':i,'features':{'left':{'change_strength':s},'right':{'change_strength':0}}}
          for i,s in enumerate([0,.2,4,.1,.2])]
    assert [r['sample_id'] for r in select_times(rows)]==[0,1,2,4]
    assert [r['sample_id'] for r in select_times(rows[:2])]==[0,1]
    assert select_times(rows[:1])==rows[:1]


def test_sampling_copies_buffers_without_physics_and_builds_exact_strips(tmp_path,monkeypatch):
    pixels=np.zeros((60,80,3),dtype=np.uint8)
    sensors={n:SimpleNamespace(frame=np.array([1]),_timestamp_last_update=np.array([0.])) for n in VIEWS}
    task=SimpleNamespace(
        _camera_manager=SimpleNamespace(cameras={n:sensors[n] for n in VIEWS[:2]},
            get_observations=lambda _: {n:{'rgb':pixels} for n in VIEWS[:2]}),
        _tactile_manager=SimpleNamespace(tactiles={n:SimpleNamespace(sensor=sensors[n]) for n in VIEWS[2:]},
            get_observations=lambda _:{n:{'rgb_marker':pixels} for n in VIEWS[2:]}))
    count={'simulation_time_seconds':0.,'physics_steps':0}
    state={'xyz_m':[0,0,0],'gripper_command':'inherited_hold','gripper_target_positions_m':[.01,.012],
           'gripper_finger_positions_m':[.02,.02]}
    controller=SimpleNamespace(counts=lambda:dict(count),state=lambda:state,terminal=lambda:'native_success')
    recorder=TactileRecorder(task,controller,tmp_path,cfg())
    recorder.sample(); recorder.begin('action_001',{'delta_mm':[1,0,0]})
    pixels[:]=100
    for s in sensors.values():s.frame+=1;s._timestamp_last_update+=.02
    count.update(simulation_time_seconds=.02,physics_steps=2)
    recorder.sample(); recorder.sample()
    assert len(recorder.rows)==2 and count['physics_steps']==2
    from PIL import ImageDraw
    drawn=[]
    original_text=ImageDraw.ImageDraw.text
    def record_text(self,xy,text,*args,**kwargs):
        drawn.append(text)
        return original_text(self,xy,text,*args,**kwargs)
    monkeypatch.setattr(ImageDraw.ImageDraw, "text", record_text)
    images, selection=recorder.history()
    assert drawn and all("native_success" not in t and "terminal" not in t for t in drawn)
    assert recorder.rows[-1]["native_terminal"] == "native_success"
    assert selection['sample_ids']==[0,1]
    assert len(images)==2
    first=np.array(Image.open(tmp_path/recorder.rows[0]['images']['left_tactile']))
    strip=np.array(Image.open(tmp_path/images[0]['path']))
    np.testing.assert_array_equal(strip[30:,:80],first)
    assert first.max()==0
    recorder.close()
    export_review_video(tmp_path)
    assert (tmp_path/'review.mp4').stat().st_size>0
    assert json.loads((tmp_path/'video.json').read_text())['frame_count']==2


@pytest.mark.parametrize('deadline,grace,expected',[(4,.05,'terminal_grace_expired'),(.05,4,'overall_deadline')])
def test_terminal_grace_respects_overall_deadline(tmp_path,deadline,grace,expected):
    from scripts.univtac.run_codex_readonly_observation import _run_to_files
    stop=tmp_path/'stop';stop.write_text('{}')
    result=_run_to_files([sys.executable,'-c','import time; time.sleep(5)'],cwd=tmp_path,
                        environment=dict(os.environ),stdout_path=tmp_path/'out',stderr_path=tmp_path/'err',
                        timeout_seconds=deadline,stop_path=stop,terminal_grace_seconds=grace)
    assert result['exit_mode']==expected


def test_live_mcp_returns_actual_images_and_records_paths(tmp_path, monkeypatch):
    import asyncio

    import tools.embodied_gateway as module
    from tools.embodied_mcp_server import build_live_backend_server
    paths=[]
    for n in VIEWS:
        path=tmp_path/(n+'.png')
        Image.new('RGB',(30,20),'red').save(path)
        paths.append(path)
    monkeypatch.setattr(module.LiveBackendGateway, 'call', lambda *a,**k:module.GatewayResult(
        True, {'observation':{'tactile_history':{'sample_ids':[0,2,3]}}}, images=paths))
    server=build_live_backend_server(root=tmp_path,worker_url='http://127.0.0.1:1')
    result=asyncio.run(server.call_tool('move_to',{'delta_mm':[1,0,0]}))
    assert sum(getattr(b,'type',None)=='image' for b in result)==4
    row=json.loads((tmp_path/'operator_context.jsonl').read_text())
    assert row['response_image_paths']==[p.name for p in paths]
    assert 'sample_ids' in row['response_text_blocks'][0]


def test_video_range_and_nested_r15_dashboard(tmp_path):
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer

    from scripts.univtac.serve_experiment_dashboard import make_handler
    batch=tmp_path/'univtac-isaac51-r15'/'debug'
    root=batch/'seed_1000003';root.mkdir(parents=True)
    (batch/'run_manifest.json').write_text(json.dumps({'round':'R1.5','mode':'debug'}))
    (root/'episode.json').write_text(json.dumps({'seed':1000003,'scored':False}))
    (root/'review.mp4').write_bytes(b'0123456789')
    server=ThreadingHTTPServer(('127.0.0.1',0),make_handler(tmp_path))
    thread=threading.Thread(target=server.serve_forever);thread.start()
    try:
        base=f'http://127.0.0.1:{server.server_port}'
        with urllib.request.urlopen(base+'/api/r15-autonomous') as response:
            assert len(json.load(response)['batches'])==1
        request=urllib.request.Request(base+'/artifact?run=univtac-isaac51-r15/debug/seed_1000003&path=review.mp4',headers={'Range':'bytes=2-5'})
        with urllib.request.urlopen(request) as response:
            assert response.status==206 and response.read()==b'2345'
    finally:
        server.shutdown();thread.join();server.server_close()
