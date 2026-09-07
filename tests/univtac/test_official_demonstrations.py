import asyncio
import json
from types import SimpleNamespace

from PIL import Image

from scripts.univtac.prepare_official_demonstrations import segment_bounds
from scripts.univtac.run_autonomous_insert_hole import operator_prompt
from scripts.univtac.run_official_tactile_icl import CONDITIONS, ORDER
from sim.envs.univtac.autonomous_session import AutonomousSession
from tools.embodied_mcp_server import build_live_backend_server


def test_boundaries_use_motion_and_keep_continuous_transitions():
    assert segment_bounds([1,1,2,3,3,4], [b'move',b'move',b'delay',b'move',b'move',b'delay']) == [(0,2),(2,5)]


def test_same_prompt_and_balanced_order():
    prompts = {operator_prompt({'demonstrations':True,'condition':c}) for c in CONDITIONS}
    assert len(prompts) == 1
    assert 'No operation demonstrations are provided.' not in next(iter(prompts))
    assert len(set(ORDER)) == 9


def test_dashboard_keeps_condition_denominators_separate(tmp_path):
    from scripts.univtac.autonomous_dashboard import load_autonomous_runs

    root=tmp_path/'univtac-isaac51-r17/batch';root.mkdir(parents=True)
    (root/'run_manifest.json').write_text(json.dumps({'round':'R1.7'}))
    for seed,condition in ORDER:
        folder=root/f'seed_{seed}'/condition;folder.mkdir(parents=True)
        (folder/'episode.json').write_text(json.dumps({'seed':seed,'evaluable':True,
            'task_success':condition=='visual_action_icl' and seed!=1000005}))
    groups=load_autonomous_runs(tmp_path,'R1.7')['batches']
    assert [len(g['episodes']) for g in groups]==[3,3,3]
    assert [sum(e['episode']['task_success'] for e in g['episodes']) for g in groups]==[0,2,0]


def test_review_native_images_and_actual_context(tmp_path, monkeypatch):
    import tools.embodied_gateway as gateway
    from tools.embodied_gateway import GatewayResult

    image = tmp_path/'touch.png'
    Image.new('RGB',(32,24),'red').save(image)
    monkeypatch.setattr(gateway.LiveBackendGateway,'call',lambda self,tool,args:
        GatewayResult(True,{'demonstrations':{'examples':['historical motion']},'image_labels':['touch']}, images=[image]))
    server = build_live_backend_server(root=tmp_path,worker_url='http://unused',demonstrations=True)
    assert len(server._tool_manager._tools) == 7
    blocks = asyncio.run(server.call_tool('review_demonstrations',{}))
    assert [b.type for b in blocks] == ['text','image']
    assert blocks[1].mimeType == 'image/png' and blocks[1].data
    context = json.loads((tmp_path/'operator_context.jsonl').read_text())
    assert context['tool'] == 'review_demonstrations'
    assert 'touch.png' in json.dumps(context)


def test_review_once_no_physics_and_reject_motion_before_review(tmp_path):
    task=SimpleNamespace(_robot_manager=None,step_count=0,_physics_step_count=0,take_action_cnt=0,
        cfg=SimpleNamespace(step_lim=300,sim=SimpleNamespace(dt=1/120)),eval_success=False)
    session=AutonomousSession(task,{'demonstrations':True,'max_tool_calls':100,'max_move_requests':30},tmp_path,1000003)
    folder=tmp_path/'demonstrations';folder.mkdir()
    (folder/'projection.json').write_text(json.dumps({'text':{'examples':[]},'images':[]}))
    session.capture=lambda: (_ for _ in ()).throw(AssertionError('No observation/physics during review'))
    assert not session.call('move_to',{'delta_mm':[1,0,0]})['ok']
    first=session.call('review_demonstrations',{})
    assert first['ok'] and first['text']['demonstrations']=={'examples':[]}
    assert session.call('review_demonstrations',{})['images']==[]
    assert session.tool_count==3 and session.move_requests==0 and task.take_action_cnt==0
