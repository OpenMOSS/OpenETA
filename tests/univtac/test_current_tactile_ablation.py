"""Delivery-only tactile ablation, fixed scheduling and separate replay labels."""
import asyncio
import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml
from PIL import Image

from scripts.univtac.autonomous_dashboard import R110_HTML, load_autonomous_runs
from scripts.univtac.official_icl_review import CURRENT_TOUCH_NAMES, pair_specs, summarize
from scripts.univtac.run_autonomous_insert_hole import operator_prompt
from scripts.univtac.run_official_tactile_icl import condition_config, experiment_plan
from sim.envs.univtac.autonomous_session import AutonomousSession, project_current_observation
from tools.embodied_mcp_server import build_live_backend_server

REPO=Path(__file__).resolve().parents[2]
BASE=yaml.safe_load((REPO/'configs/univtac/current_tactile_ablation.yaml').read_text())


def test_fixed_plan_prompt_and_execution_settings():
    conditions,order=experiment_plan(BASE)
    patterns=[('B_live','C_live','C_no_live','B_no_live'),('C_live','B_no_live','B_live','C_no_live'),
              ('B_no_live','C_no_live','C_live','B_live'),('C_no_live','B_live','B_no_live','C_live')]*2
    assert order==[(s,c) for s,cs in zip(range(1000018,1000026),patterns,strict=True) for c in cs]
    assert len(set(order))==32
    configs=[condition_config(BASE,c) for c in conditions]
    assert len({operator_prompt(c) for c in configs})==1
    prompt=operator_prompt(configs[0])
    assert 'when provided' in prompt and 'Missing tactile\ninput is not a tool failure.' in prompt
    assert BASE['public_task_rules'] in prompt
    old=yaml.safe_load((REPO/'configs/univtac/new_seed_tactile_icl.yaml').read_text())
    excluded={'round','seeds','episode_order','current_tactile_ablation'}
    assert {k:v for k,v in BASE.items() if k not in excluded}=={k:v for k,v in old.items() if k not in excluded}
    for c in configs:
        assert c['task_information']=='R'
        assert c['current_tactile']==(c['condition'] in ('B_live','C_live'))
        assert c['demonstration_condition']==('visual_action_icl' if c['condition'][0]=='B' else 'tactile_action_icl')


def session_fixture(root, live):
    task=SimpleNamespace(_robot_manager=None,step_count=0,_physics_step_count=0,take_action_cnt=0,
        cfg=SimpleNamespace(step_lim=300,sim=SimpleNamespace(dt=1/120)),eval_success=False)
    session=AutonomousSession(task,{'current_tactile':live,'max_tool_calls':100,'max_move_requests':30},root,1000018)
    descriptors=[]
    for name,label in [('head','head RGB'),('wrist','wrist RGB'),('left','left_tactile rgb_marker'),('right','right_tactile rgb_marker')]:
        Image.new('RGB',(40,30),'red').save(root/f'{name}.png')
        descriptors.append({'label':label,'path':f'{name}.png'})
    def capture():
        session.observation_index+=1
        session.latest={'observation_id':f'obs_{session.observation_index}', 'images':copy.deepcopy(descriptors),
                        'robot':{'gripper_target_positions_m':[.01,.012]},'counts':{'control_steps':0},'terminal':None}
        return session.latest
    session.capture=capture
    session.controller.tcp=lambda:np.eye(4)
    session.controller.execute=lambda target,gripper:{'requested_target':{'xyz_m':target[:3,3].tolist()},'arm_reached':True}
    session.recorder=SimpleNamespace(begin=lambda *a:None,history=lambda:(descriptors[2:],{
        'action_id':'action_001','selection':{'tracking_quality':.9},'sample_ids':[0,2]}))
    return session,task


@pytest.mark.parametrize('live',[True,False])
def test_session_delivery_all_observation_paths_keep_host_data(tmp_path,live):
    session,task=session_fixture(tmp_path,live)
    results=[session.call('observe',{}),session.call('move_to',{'delta_mm':[5,0,0]})]
    # Capture immutable host evidence before another observe overwrites latest.
    assert 'tactile_history' in session.latest and len(session.latest['images'])==4
    results += [session.call('move_to',{'delta_mm':[1,0,0],'xyz_m':[1,2,3]}),session.call('observe',{})]
    assert not results[2]['ok'] and results[2]['text']['recoverable']
    for payload in results:
        assert len(payload['images'])==(4 if live else 2)
        obs=payload['text']['observation']
        assert obs['robot']['gripper_target_positions_m']==[.01,.012]
        if not live:
            assert 'tactile' not in json.dumps(payload)
    assert task.step_count==task.take_action_cnt==0  # mocked execution never advances physics
    assert len(session.latest['images'])==4
    preview=session.call('move_to',{'delta_mm':[1,0,0],'preview':True})
    assert preview['images']==[] and preview['text']['physics_stepped'] is False


def test_c_no_live_historical_touch_and_current_native_mcp_images(tmp_path,monkeypatch):
    session,_=session_fixture(tmp_path,False)
    session.config['demonstrations']=True
    folder=tmp_path/'demonstrations';folder.mkdir()
    images=[{'label':f'historical touch {i}','path':'left.png'} for i in range(24)]
    (folder/'projection.json').write_text(json.dumps({'text':{'examples':['historical']},'images':images}))
    def urlopen(request,**kwargs):
        request=json.loads(request.data)
        return io.BytesIO(json.dumps(session.call(request['tool'],request['arguments'])).encode())
    monkeypatch.setattr('urllib.request.urlopen',urlopen)
    server=build_live_backend_server(root=tmp_path,worker_url='http://unused',demonstrations=True)
    for tool,args,count in [('review_demonstrations',{},24),('observe',{},2),('move_to',{'delta_mm':[5,0,0]},2),
                            ('move_to',{'delta_mm':[1,0,0],'xyz_m':[1,2,3]},2),('review_demonstrations',{},0)]:
        blocks=asyncio.run(server.call_tool(tool,args))
        assert sum(b.type=='image' for b in blocks)==count
        assert all(b.data for b in blocks if b.type=='image')
    rows=[json.loads(l) for l in (tmp_path/'operator_context.jsonl').read_text().splitlines()]
    assert len(rows[0]['response_image_paths'])==24
    for row in rows[1:4]:
        assert row['response_image_paths']==['head.png','wrist.png']
        assert 'tactile' not in ''.join(row['response_text_blocks'])


def test_projection_does_not_mutate_saved_observation():
    source={'ok':True,'text':{'observation':{'images':[{'label':'head RGB','path':'head.png'},{'label':'left_tactile','path':'touch.png'}],
            'tactile_history':{'selection':{'tracking_quality':.2}},'robot':{'q':[1]}}},'images':[]}
    before=copy.deepcopy(source)
    assert project_current_observation(source,current_tactile=False)['text']['observation']=={'images':[{'label':'head RGB','path':'head.png'}],'robot':{'q':[1]}}
    assert source==before and project_current_observation(source) is source


def test_four_groups_pairs_and_difference_in_differences(tmp_path):
    root=tmp_path/'univtac-isaac51-r110';batch=root/'batch';batch.mkdir(parents=True)
    _,order=experiment_plan(BASE)
    (batch/'run_manifest.json').write_text(json.dumps({'round':'R1.10','order':order}))
    wins={'B_live':4,'C_live':6,'B_no_live':3,'C_no_live':4}
    for seed,c in order:
        folder=batch/f'seed_{seed}'/c;folder.mkdir(parents=True)
        (folder/'episode.json').write_text(json.dumps({'seed':seed,'condition':c,'evaluable':True,'task_success':seed-1000018<wins[c]}))
    summarize(root);d=json.loads((root/'results.json').read_text())
    assert len(d['cells'])==32 and all(g['planned']==8 for g in d['groups'].values())
    assert {c:g['successes'] for c,g in d['groups'].items()}==wins
    assert d['contrasts_percentage_points']['historical_touch_difference_in_differences']==12.5
    assert len(d['paired_outcomes'])==4
    assert d['paired_outcomes']['C_live-B_live']['method_only_success']==2
    assert pair_specs(root)==[('B_live','B_no_live'),('C_live','C_no_live')]
    assert len(load_autonomous_runs(tmp_path,'R1.10')['batches'])==4
    assert 'currentTouchTable(d)' in R110_HTML and 'function newSeedTable' not in R110_HTML
    assert '仅供用户审阅，本episode未送给Agent' in R110_HTML
    assert 'Host-only 前后记录' in R110_HTML
    assert set(CURRENT_TOUCH_NAMES)==set(wins)


def test_no_live_video_marks_touch_without_changing_raw_frames(tmp_path,monkeypatch):
    from PIL import ImageDraw

    from sim.envs.univtac.tactile_history import REVIEW_ONLY_TOUCH, VIEWS, export_review_video
    state={'xyz_m':[0,0,0],'gripper_command':'inherited_hold','gripper_target_positions_m':[.01,.012],
           'gripper_finger_positions_m':[.013,.014]}
    paths={}
    for n in VIEWS:
        path=tmp_path/f'{n}.png';Image.new('RGB',(40,30),'blue').save(path);paths[n]=path.name
    raw={p:(tmp_path/p).read_bytes() for p in paths.values()}
    rows=[{'sample_id':i,'simulation_time_seconds':i/60,'action_id':'takeover' if i==0 else 'action_001',
           'robot':state,'request':{},'native_terminal':None,'images':paths} for i in range(2)]
    (tmp_path/'samples.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (tmp_path/'episode.json').write_text(json.dumps({'current_tactile':False}))
    text=[];original=ImageDraw.ImageDraw.text
    def draw(self,xy,value,*args,**kwargs):
        text.append(value);return original(self,xy,value,*args,**kwargs)
    monkeypatch.setattr(ImageDraw.ImageDraw,'text',draw)
    export_review_video(tmp_path)
    assert sum(REVIEW_ONLY_TOUCH in t for t in text)==4
    assert (tmp_path/'review.mp4').stat().st_size>0
    assert raw=={p:(tmp_path/p).read_bytes() for p in paths.values()}


def test_two_replay_pairs_keep_distinct_files(tmp_path):
    from scripts.univtac.official_icl_review import review_pairs
    root=tmp_path/'univtac-isaac51-r110';batch=root/'batch';batch.mkdir(parents=True)
    (batch/'run_manifest.json').write_text(json.dumps({'round':'R1.10','order':[[1000018,c] for c in CURRENT_TOUCH_NAMES]}))
    for c in CURRENT_TOUCH_NAMES:
        folder=batch/'seed_1000018'/c;(folder/'review_labelled').mkdir(parents=True)
        Image.new('RGB',(960,900),'white').save(folder/'review_labelled/000000.png')
        (folder/'samples.jsonl').write_text(json.dumps({'sample_id':0,'simulation_time_seconds':0})+'\n')
        (folder/'episode.json').write_text(json.dumps({'task_success':True}))
        (folder/'review_video.json').write_text('{}')
    review_pairs(root)
    movies=list((batch/'seed_1000018').glob('paired_*.mp4'))
    assert len(movies)==4 and all(p.stat().st_size>0 for p in movies)
    mappings=[json.loads(p.with_suffix('.json').read_text()) for p in movies]
    assert {tuple(m['conditions']) for m in mappings}=={('B_live','B_no_live'),('C_live','C_no_live')}
    assert all(m['synthetic_observations']==0 and m['final_display_hold_seconds']==1 for m in mappings)
