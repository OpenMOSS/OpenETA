"""Offline query feedback isolation through the actual native MCP route."""
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

from scripts.univtac.run_autonomous_insert_hole import codex_command, operator_prompt
from sim.envs.univtac.autonomous_operation import NativeController
from sim.envs.univtac.autonomous_session import AutonomousSession
from sim.envs.univtac.feedback_protocol import ENDED, PROTOCOL, new_run_config, project_query
from tools.embodied_mcp_server import build_live_backend_server


def session_at(root):
    task = SimpleNamespace(_robot_manager=None, step_count=0, _physics_step_count=0, take_action_cnt=0,
        cfg=SimpleNamespace(step_lim=300, sim=SimpleNamespace(dt=1/120)), eval_success=False,
        plan_success=True, metadata={}, check_success=lambda: False)
    config = new_run_config({'task':'insert_hole','max_tool_calls':100,'max_move_requests':30})
    session = AutonomousSession(task, config, root, 1000040)
    images = []
    for name,label in [('head','head RGB'),('wrist','wrist RGB'),('left','left_tactile rgb_marker'),('right','right_tactile rgb_marker')]:
        Image.new('RGB',(10,10),'red').save(root/f'{name}.png')
        images.append({'label':label,'path':f'{name}.png'})
    def capture():
        session.observation_index += 1
        session.latest = {'observation_id':f'obs_{session.observation_index}',
            'robot':{'xyz_m':[0,0,0],'gripper_target_positions_m':[.005,.006]},
            'counts':session.controller.counts(), 'images':copy.deepcopy(images),
            'terminal':session.termination(), 'execution_feedback':session.feedback}
        return session.latest
    session.capture = capture
    session.controller.tcp = lambda: np.eye(4)
    def execute(*_):
        task.take_action_cnt += 1
        task._physics_step_count += 2
        return {'arm_reached':True, 'gripper_command':'inherited_hold',
                'segment_end_reason':session.termination() or 'arm_reached', 'error':None}
    session.controller.execute = execute
    return session, task


@pytest.mark.parametrize('native', ['native_success','native_early_stop','native_step_limit'])
def test_terminal_projection_host_retained_and_no_more_motion(tmp_path,native):
    session,task=session_at(tmp_path)
    session.call('observe',{})
    if native=='native_success':task.eval_success=True
    elif native=='native_early_stop':session.controller.early_stop=True
    else:task.take_action_cnt=300
    session.feedback={'arm_reached':True,'segment_end_reason':native,'error':native,
                      'actual':{'xyz_m':[0,0,0],'task_success':True},'reward':1}
    for tool,args in [('observe',{}),('move_to',{'delta_mm':[1,0,0]}),
                      ('move_to',{'delta_mm':[1,0,0],'preview':True}),
                      ('move_to',{'execute_preview_id':'old'}),('mark_point',{'view':'head','u':0,'v':0}),
                      ('report_issue',{'message':'sensor view unclear'}),('finish_episode',{}),('observe',{})]:
        payload=session.call(tool,args)
        text=json.dumps(payload)
        assert (': '+json.dumps(native)) not in text and 'task_success' not in text and '"success"' not in text
        assert payload['text']['terminal']==ENDED
        assert task._physics_step_count==0
    final=json.loads((tmp_path/'final_result.json').read_text())
    assert final['termination']==native and final['feedback_protocol']==PROTOCOL
    assert native in (tmp_path/'host_tool_trace.jsonl').read_text()
    assert (': '+json.dumps(native)) not in ''.join(json.dumps(json.loads(l)['result']) for l in (tmp_path/'tool_trace.jsonl').read_text().splitlines())


def test_false_continues_recoverable_errors_and_voluntary_finish(tmp_path):
    session,task=session_at(tmp_path)
    session.call('observe',{})
    assert session.controller.check()['success'] is False
    bad=session.call('move_to',{'xyz_m':[0,0,0],'delta_mm':[1,0,0]})
    assert bad['text']['recoverable'] and bad['text']['terminal'] is None
    good=session.call('move_to',{'delta_mm':[10,0,0]})
    assert good['ok'] and good['text']['execution']['arm_reached']
    assert good['text']['execution']['gripper_command']=='inherited_hold'
    ended=session.call('finish_episode',{})
    assert ended['text']=={'finished':True,'terminal':ENDED}
    session.call('move_to',{'delta_mm':[10,0,0]})
    assert task._physics_step_count==2
    final=session.finalize()
    assert final['task_success'] is False and final['infrastructure_error'] is None
    assert final['termination']=='agent_finish'


def test_success_priority_and_step_limit_unchanged(tmp_path):
    session,task=session_at(tmp_path)
    c=session.controller
    assert isinstance(c,NativeController) and c.terminal() is None
    c.early_stop=True;task.eval_success=True;task.take_action_cnt=300
    assert c.terminal()=='native_success'
    task.eval_success=False
    assert c.terminal()=='native_early_stop'
    c.early_stop=False
    assert c.terminal()=='native_step_limit'


def test_prompt_tools_version_and_legacy_config_unchanged(tmp_path):
    old=yaml.safe_load(Path('configs/univtac/current_tactile_ablation.yaml').read_text())
    old['task_information']='R';snapshot=copy.deepcopy(old)
    new=new_run_config(old)
    prompt=operator_prompt(new)
    assert 'check_task' not in prompt and '0.04' in prompt and '0.99' in prompt
    command=codex_command(tmp_path,'http://unused',new)
    assert 'check_task' not in ' '.join(command)
    assert PROTOCOL in ' '.join(command) and 'model_reasoning_effort="low"' in command
    assert old==snapshot and 'check_task' in operator_prompt(old)
    new_server=build_live_backend_server(root=tmp_path,worker_url='http://unused',feedback_protocol=PROTOCOL)
    old_server=build_live_backend_server(root=tmp_path,worker_url='http://unused')
    assert 'check_task' not in new_server._tool_manager._tools
    assert 'check_task' in old_server._tool_manager._tools


def test_real_mcp_blocks_context_history_and_rejection(tmp_path,monkeypatch):
    session,task=session_at(tmp_path)
    session.config['demonstrations']=True
    demo=tmp_path/'demonstrations';demo.mkdir()
    (demo/'projection.json').write_text(json.dumps({'text':{'examples':[{'outcome':'success'}]},
        'images':[{'label':'historical expert success','path':'left.png'}]}))
    def urlopen(request,**kwargs):
        x=json.loads(request.data)
        return io.BytesIO(json.dumps(session.call(x['tool'],x['arguments'])).encode())
    monkeypatch.setattr('urllib.request.urlopen',urlopen)
    server=build_live_backend_server(root=tmp_path,worker_url='http://unused',demonstrations=True,feedback_protocol=PROTOCOL)
    listed=asyncio.run(server.list_tools())
    assert 'check_task' not in [t.name for t in listed]
    blocks=asyncio.run(server.call_tool('review_demonstrations',{}))
    assert '"outcome": "success"' in blocks[0].text and blocks[1].type=='image'
    assert all(t.outputSchema is None for t in listed)
    asyncio.run(server.call_tool('observe',{}))
    task.eval_success=True
    blocks=asyncio.run(server.call_tool('move_to',{'delta_mm':[1,0,0]}))
    assert sum(b.type=='image' for b in blocks)==4
    assert 'native_success' not in blocks[0].text and ENDED in blocks[0].text
    context=[json.loads(l) for l in (tmp_path/'operator_context.jsonl').read_text().splitlines()]
    assert context[-1]['response_text_blocks']==[blocks[0].text]
    assert len(context[-1]['response_image_paths'])==4
    with pytest.raises(Exception,match='Unknown tool'):
        asyncio.run(server.call_tool('check_task',{}))


def test_explicit_nested_projection_drops_unknown_scores():
    payload={'ok':True,'text':{'success':True,'reward':1,'error':{'native_success':True},
        'execution':{'arm_reached':False,'segment_end_reason':'native_early_stop','error':'native_early_stop',
                     'evaluation':{'score':1},'actual':{'xyz_m':[1,2,3],'success':True}},
        'observation':{'images':[{'label':'native_success','path':'review.png'}],
                       'execution_feedback':{'segment_end_reason':'native_success','error':'native_success'},'reward':4}},
        'images':[],'structuredContent':{'success':True}}
    result=project_query(payload,tool='move_to',ended=True)
    text=json.dumps(result)
    assert all(x not in text for x in ['native_success','native_early_stop','reward','evaluation','structuredContent','review.png'])
    assert result['ok'] and result['text']['execution']['arm_reached'] is False
    assert result['text']['execution']['actual']['xyz_m']==[1,2,3]
    assert payload['text']['success'] is True


def test_dashboard_and_summary_keep_protocols_separate(tmp_path):
    from scripts.univtac.autonomous_dashboard import load_autonomous_runs
    from scripts.univtac.official_icl_review import summarize
    for name,version in [('old',None),('new',PROTOCOL)]:
        root=tmp_path/'univtac-isaac51-r110'/name
        folder=root/'seed_1000018'/'C_live';folder.mkdir(parents=True)
        manifest={'round':'R1.10'}
        state={'seed':1000018,'task_success':True}
        if version:
            manifest['feedback_protocol']=version
            state['feedback_protocol']=version
        (root/'run_manifest.json').write_text(json.dumps(manifest))
        (folder/'episode.json').write_text(json.dumps(state))
        (folder/'host_evaluator.json').write_text(json.dumps({'eval_success':True}))
    old=load_autonomous_runs(tmp_path,'R1.10')
    new=load_autonomous_runs(tmp_path,'R1.10',feedback_protocol=PROTOCOL)
    assert all(b['name'].startswith('old') for b in old['batches'])
    assert all(b['name'].startswith('new') for b in new['batches'])
    assert next(e for b in new['batches'] for e in b['episodes'])['host_evaluator']['eval_success']
    batch=tmp_path/'mixed'/'batch'
    folder=batch/'seed_1000018'/'B_live';folder.mkdir(parents=True)
    base=yaml.safe_load(Path('configs/univtac/current_tactile_ablation.yaml').read_text())
    (batch/'run_manifest.json').write_text(json.dumps({'round':'R1.10','config':base,'order':[[1000018,'B_live']]}))
    (folder/'episode.json').write_text(json.dumps({'feedback_protocol':PROTOCOL}))
    with pytest.raises(ValueError,match='different feedback protocols'):
        summarize(tmp_path/'mixed')
