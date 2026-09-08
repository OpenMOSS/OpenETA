import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from scripts.univtac.prepare_official_demonstrations import tactile_without_vision
from scripts.univtac.run_eight_task_coverage import retryable_initialization, run_cell
from scripts.univtac.run_fourway_capacity import Coordinator


def test_d_only_removes_historical_vision(tmp_path):
    c={'text':{'task_goal':'task','examples':[{'outcome':'success','segments':[{
        'vision_labels':['head historical','wrist historical'],'before':{'ee':[1]},
        'after':{'ee':[2]},'measured_motion':{'translation':[1]}}]}],
        'historical_touch':[{'image_labels':['touch']} ]},
        'images':[{'label':n,'path':n+'.png'} for n in ('head historical','wrist historical','touch')]}
    for im in c['images']:Image.new('RGB',(4,4),'red').save(tmp_path/im['path'])
    d=tactile_without_vision(c)
    assert d['images']==[c['images'][-1]]
    assert 'vision_labels' not in d['text']['examples'][0]['segments'][0]
    assert d['text']['historical_touch']==c['text']['historical_touch']
    assert d['text']['examples'][0]['outcome']=='success'
    assert np.array_equal(np.array(Image.open(tmp_path/d['images'][0]['path'])),np.array(Image.open(tmp_path/c['images'][-1]['path'])))
    assert 'vision_labels' in c['text']['examples'][0]['segments'][0]


def test_distinct_same_seed_cells(tmp_path):
    keys=[('insert_hole',1000040,'D'),('grasp_classify',1000040,'C')]
    c=Coordinator(tmp_path,keys,protocol_smoke=True)
    c.ready[keys[0]]={}
    c.barrier(keys[0])
    c.processes[keys[1],'worker']=SimpleNamespace(poll=lambda:0)
    c.abort('native init failure',keys[1])
    assert not c.lane_cancel[keys[0]].is_set() and c.lane_cancel[keys[1]].is_set()
    assert c.events[-1]['task']=='grasp_classify'


def test_resume_ready_episode_is_never_replaced(tmp_path,monkeypatch):
    key=('insert_hole',1000040,'D');c=Coordinator(tmp_path,[key],protocol_smoke=True)
    cell={'task':key[0],'seed':key[1],'condition':key[2],'cell_key':list(key)}
    folder=tmp_path/'cells'/key[0]/str(key[1])/key[2]/'attempt_1';folder.mkdir(parents=True)
    (folder/'ready.json').write_text('{}')
    (folder/'episode.json').write_text(json.dumps({'status':'completed','task_success':False}))
    (folder/'worker_lifecycle.json').write_text(json.dumps({'cleanup_complete':True}))
    monkeypatch.setattr('scripts.univtac.run_eight_task_coverage.run_episode',lambda *_:(_ for _ in ()).throw(AssertionError('must not rerun')))
    result=run_cell(SimpleNamespace(output_root=tmp_path),cell,c)
    assert result['reused_on_resume'] and result['episode']['task_success'] is False


def test_persisted_attempts_exhausted_and_oom_not_retryable(tmp_path):
    key=('insert_hole',1000040,'D');c=Coordinator(tmp_path,[key],protocol_smoke=True)
    cell={'task':key[0],'seed':key[1],'condition':key[2],'cell_key':list(key)}
    for i in range(1,4):
        folder=tmp_path/'cells'/key[0]/str(key[1])/key[2]/f'attempt_{i}';folder.mkdir(parents=True)
        (folder/'worker_lifecycle.json').write_text(json.dumps({'cleanup_complete':True}))
        (folder/'worker_error.json').write_text(json.dumps({'traceback':'task.reset(seed=args.seed)'}))
    result=run_cell(SimpleNamespace(output_root=tmp_path),cell,c)
    assert result['status']=='initialization_unavailable'
    (folder/'launcher').mkdir();(folder/'launcher/stdout_stderr.log').write_text('Out of GPU memory')
    assert not retryable_initialization(folder)


def test_all_native_atom_boundaries_are_retained():
    from scripts.univtac.prepare_official_demonstrations import segment_bounds
    ids=np.array([1,1,2,2,3,3,4])
    tags=np.array([b'try_forward',b'try_forward',b'rotate',b'rotate',b'delay',b'delay',b'move'])
    assert segment_bounds(ids,tags,all_atoms=True)==[(0,1),(1,3),(3,5),(5,6)]


def test_task_specific_prompt_limits():
    import yaml

    from scripts.univtac.run_autonomous_insert_hole import operator_prompt
    from sim.envs.univtac.feedback_protocol import new_run_config
    base=yaml.safe_load(Path('configs/univtac/current_tactile_ablation.yaml').read_text())
    config=yaml.safe_load(Path('configs/univtac/eight_task_coverage.yaml').read_text())
    for spec in config['tasks'].values():
        c=new_run_config({**base,'native_control_step_limit':spec['native_control_step_limit'],'task_information':'R'})
        assert f"limited to {spec['native_control_step_limit']} steps" in operator_prompt(c)
        assert 'check_task' not in operator_prompt(c)


def test_queue_has_two_slots_including_cleanup_and_continues_native_failure(tmp_path,monkeypatch):
    import threading
    import time

    from scripts.univtac.run_eight_task_coverage import run_queue
    tasks=['a','b','c','d','e','f','g','h']
    cells=[{'task':t,'seed':1000040,'condition':'C','cell_key':[t,1000040,'C'],'status':'prepared'} for t in tasks]
    (tmp_path/'prepared_cells.json').write_text(json.dumps({'cells':cells,'existing_insert_hole_C':'historical'}))
    lock=threading.Lock();active=0;peak=0;started=[]
    def fake(args,cell,coordinator):
        nonlocal active,peak
        with lock:
            active+=1;peak=max(peak,active);started.append(cell['task'])
        time.sleep(.02)  # includes cleanup before returning the slot
        with lock:active-=1
        return {**cell,'status':'completed','episode':{'task_success':False}}
    monkeypatch.setattr('scripts.univtac.run_eight_task_coverage.run_cell',fake)
    monkeypatch.setattr('scripts.univtac.run_eight_task_coverage.resources',lambda *_:{'MemAvailable':2**30})
    run_queue(SimpleNamespace(output_root=tmp_path))
    assert peak==2 and started==tasks
    assert len(json.loads((tmp_path/'coverage_results.json').read_text())['cells'])==8


def crash_fixture(folder):
    folder.mkdir(parents=True,exist_ok=True)
    (folder/'launcher').mkdir(exist_ok=True)
    (folder/'launcher/stdout_stderr.log').write_text('[Fatal] libomniclient.so!omniClientFreeContent')
    (folder/'episode.json').write_text(json.dumps({
        'codex_process_count':0,'reset_valid':False,'status':'infrastructure_issue'}))
    (folder/'worker_lifecycle.json').write_text(json.dumps({
        'root_pid':123,'cleanup_complete':True,'final_process_group_members':[],
        'returncode':-11,'sigterm_sent':False,'sigkill_sent':False}))


def test_narrow_crash_and_authorized_cancel_classification(tmp_path):
    from scripts.univtac.run_eight_task_coverage import native_startup_crash
    crash_fixture(tmp_path)
    assert native_startup_crash(tmp_path) and retryable_initialization(tmp_path)
    log=tmp_path/'launcher/stdout_stderr.log'
    original=log.read_text()
    for text in ('[Fatal] some_other_library',original+' Out of memory'):
        log.write_text(text)
        assert not retryable_initialization(tmp_path)
    log.write_text(original)
    (tmp_path/'ready.json').write_text('{}')
    assert not retryable_initialization(tmp_path)
    (tmp_path/'ready.json').unlink()
    episode={'codex_process_count':0,'reset_valid':False,'status':'cancelled',
             'infrastructure_error':'RuntimeError: batch cancelled: original crash'}
    (tmp_path/'episode.json').write_text(json.dumps(episode))
    life=json.loads((tmp_path/'worker_lifecycle.json').read_text())
    life.update(returncode=-15,sigterm_sent=True)
    (tmp_path/'worker_lifecycle.json').write_text(json.dumps(life))
    assert not retryable_initialization(tmp_path)
    (tmp_path/'authorized_pre_ready_cancelled.json').write_text(json.dumps({
        'pro_message_id':'8070380f-699e-45aa-b208-6012acbdf5a6',
        'original_error':episode['infrastructure_error']}))
    assert retryable_initialization(tmp_path) and not native_startup_crash(tmp_path)
    life['cleanup_complete']=False
    (tmp_path/'worker_lifecycle.json').write_text(json.dumps(life))
    assert not retryable_initialization(tmp_path)


def test_crash_streak_uses_persistent_ready_order(tmp_path):
    from scripts.univtac.run_eight_task_coverage import startup_crashes_since_ready
    rows=[]
    for i in range(3):
        f=tmp_path/f'attempt_{i+1}';crash_fixture(f)
        rows.extend([{'event':'attempt_started','timestamp_s':i*2,'cell':[i],'output_root':str(f)},
                     {'event':'cleanup','timestamp_s':i*2+1,'cell':[i]}])
    (tmp_path/'events.jsonl').write_text('\n'.join(map(json.dumps,rows))+'\n')
    assert startup_crashes_since_ready(tmp_path)==3
    with (tmp_path/'events.jsonl').open('a') as f:
        f.write(json.dumps({'event':'ready_resident','timestamp_s':6})+'\n')
    assert startup_crashes_since_ready(tmp_path)==0


def test_local_crash_retry_preserves_peer_and_attempt_number(tmp_path,monkeypatch):
    from scripts.univtac import run_eight_task_coverage as module
    from scripts.univtac.run_shot_scaling import recover_cell
    key=('lift_can',1000000,'C_2shot');peer=('insert_tube',1000000,'C_2shot')
    coordinator=Coordinator(tmp_path,[key,peer],protocol_smoke=True)
    cell={'task':key[0],'seed':key[1],'condition':key[2],'cell_key':list(key),
          'config':str(tmp_path/'config.json')}
    Path(cell['config']).write_text('{}')
    folder=tmp_path/'cells'/key[0]/str(key[1])/key[2]/'attempt_1'
    crash_fixture(folder)
    assert recover_cell(tmp_path,cell) is None
    seen=[]
    def episode(args,config,seed,root):
        seen.append(root.name);crash_fixture(root)
        if root.name=='attempt_2':
            coordinator.processes[key,'worker']=SimpleNamespace(poll=lambda:-11)
            coordinator.abort('worker exited before ready',key)
            assert not coordinator.cancel.is_set() and not coordinator.lane_cancel[peer].is_set()
            return json.loads((root/'episode.json').read_text())
        (root/'ready.json').write_text('{}')
        return {'status':'completed','codex_process_count':1,'task_success':False}
    monkeypatch.setattr(module,'run_episode',episode)
    result=run_cell(SimpleNamespace(output_root=tmp_path),cell,coordinator)
    assert seen==['attempt_2','attempt_3'] and result['status']=='completed'
    assert not coordinator.cancel.is_set()


def test_third_startup_crash_pauses_before_another_attempt(tmp_path,monkeypatch):
    from scripts.univtac import run_eight_task_coverage as module
    key=('lift_can',1000000,'C_2shot')
    coordinator=Coordinator(tmp_path,[key],protocol_smoke=True)
    cell={'task':key[0],'seed':key[1],'condition':key[2],'cell_key':list(key),
          'config':str(tmp_path/'config.json')}
    Path(cell['config']).write_text('{}')
    calls=[]
    def episode(args,config,seed,folder):
        calls.append(folder.name);crash_fixture(folder)
        return json.loads((folder/'episode.json').read_text())
    monkeypatch.setattr(module,'run_episode',episode)
    monkeypatch.setattr(module,'startup_crashes_since_ready',lambda root:3)
    result=run_cell(SimpleNamespace(output_root=tmp_path),cell,coordinator)
    assert calls==['attempt_1'] and result['status']=='infrastructure_issue'
    assert coordinator.cancel.is_set()


def test_single_initialization_failure_does_not_retry(tmp_path,monkeypatch):
    from scripts.univtac import run_eight_task_coverage as module
    key=('lift_can',1000000,'C_1shot')
    coordinator=Coordinator(tmp_path,[key],protocol_smoke=True)
    cell={'task':key[0],'seed':key[1],'condition':key[2],'cell_key':list(key),
          'config':str(tmp_path/'config.json')}
    Path(cell['config']).write_text('{}')
    seen=[]
    def episode(args,config,seed,folder):
        seen.append(folder.name);folder.mkdir(parents=True)
        (folder/'worker_lifecycle.json').write_text(json.dumps({'cleanup_complete':True}))
        (folder/'worker_error.json').write_text(json.dumps({'traceback':'task.reset(seed=args.seed)'}))
        return {'status':'infrastructure_issue','codex_process_count':0}
    monkeypatch.setattr(module,'run_episode',episode)
    result=run_cell(SimpleNamespace(output_root=tmp_path,max_initialization_attempts=1),cell,coordinator)
    assert seen==['attempt_1'] and result['status']=='initialization_unavailable'
    assert result['attempts']==1 and not coordinator.cancel.is_set()
