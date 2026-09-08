import copy
import json
from pathlib import Path
from types import SimpleNamespace

from scripts.univtac.run_shot_scaling import (
    atomic_json,
    matched_subset,
    ordered_cells,
    recover_cell,
)

TASKS=['insert_tube','lift_can','lift_bottle','pull_out_key']


def test_fixed_matrix_and_nested_assignments():
    seeds=json.loads(Path('configs/univtac/main_query_seeds.json').read_text())
    assert seeds==list(range(1000000,1000100))
    cells=list(ordered_cells(TASKS,seeds))
    assert len(cells)==2400
    assert len({(c['task'],c['seed'],c['condition']) for c in cells})==2400
    for task in TASKS:
        for shot in (1,2,4):
            b=[c for c in cells if c['task']==task and c['shot']==shot and c['comparison_condition']=='B']
            c=[c for c in cells if c['task']==task and c['shot']==shot and c['comparison_condition']=='C']
            assert sorted(x['seed'] for x in b)==seeds
            assert {x['seed']:x['expert_ids'] for x in b}=={x['seed']:x['expert_ids'] for x in c}
            if shot==1:
                assert sum(x['expert_ids']==[0] for x in b)==50
                assert sum(x['expert_ids']==[1] for x in b)==50
            else:
                assert all(x['expert_ids']==list(range(shot)) for x in b+c)
    assert {c['condition'] for c in cells[:24]}=={'B_1shot','C_1shot','B_2shot','C_2shot','B_4shot','C_4shot'}


def test_subset_preserves_episode_and_shared_images(tmp_path):
    examples=[{'example_id':f'official_episode_{i}','segments':[{'vision_labels':[f'v{i}'],'measured_motion':{'dx':i}}],'outcome':'success'} for i in range(4)]
    history=[{'example_id':e['example_id'],'image_labels':[f't{i}']} for i,e in enumerate(examples)]
    full={'text':{'examples':examples,'historical_touch':history,'interpretation':'measured'},
          'images':[{'label':f'{view}{i}','path':f'{view}{i}.png'} for view in ('v','t') for i in range(4)]}
    unchanged=copy.deepcopy(full)
    b,c=matched_subset([(tmp_path,full)],[0,1],tmp_path/'two')
    assert full==unchanged and b['text']['examples']==examples[:2]
    assert c['text']['historical_touch']==history[:2]
    assert c['images'][:2]==b['images']
    assert all(Path(x['path']).is_absolute() for x in c['images'])
    assert not list((tmp_path/'two').glob('*.png'))


def make_cell(tmp_path,condition='B_1shot'):
    c={'task':'lift_can','seed':1000000,'condition':condition}
    f=tmp_path/'cells'/c['task']/str(c['seed'])/condition/'attempt_1'
    f.mkdir(parents=True);atomic_json(f/'ready.json',{})
    return c,f


def test_resume_failed_task_skips_without_touching_other_shot(tmp_path):
    c,f=make_cell(tmp_path)
    atomic_json(f/'episode.json',{'status':'completed','task_success':False,'termination':'agent_finish'})
    atomic_json(f/'worker_lifecycle.json',{'cleanup_complete':True})
    assert recover_cell(tmp_path,c)['episode']['task_success'] is False
    assert recover_cell(tmp_path,{**c,'condition':'B_2shot'}) is None


def test_resume_recovers_interruption_after_cleanup_without_rerun(tmp_path):
    c,f=make_cell(tmp_path)
    atomic_json(f/'episode.json',{'status':'running','codex_process_count':1})
    atomic_json(f/'worker_lifecycle.json',{'cleanup_complete':True,'returncode':0})
    atomic_json(f/'codex_lifecycle.json',{'exit_mode':'natural_exit'})
    atomic_json(f/'final_result.json',{'reset_valid':True,'native_success_available':True,'task_success':False})
    result=recover_cell(tmp_path,c)
    assert result['status']=='completed' and result['episode']['recovered_after_cleanup']
    assert len(list(f.parent.glob('attempt_*')))==1


def test_unresolved_accepted_never_retried(tmp_path):
    c,_=make_cell(tmp_path)
    assert recover_cell(tmp_path,c)['status']=='unresolved_previous_attempt'


def test_cleanup_slots_do_not_wait_for_media_and_resume_retains_old_results(tmp_path,monkeypatch):
    import threading
    import time

    from scripts.univtac import run_shot_scaling as module
    cells=[]
    for i in range(4):
        task=f'task{i}';cells.append({'task':task,'seed':1000000,'condition':'B_1shot','comparison_condition':'B','cell_key':[task,1000000,'B_1shot'],'status':'not_run'})
    for cell in cells:
        config=tmp_path/(cell['task']+'.json');atomic_json(config,{})
        cell.update(config=str(config),effective_config={})
    settings={'media_workers':1,'video_playback_rate':.05,'minimum_disk_free_bytes':0}
    atomic_json(tmp_path/'manifest.json',{'settings':settings,'cells':cells})
    finished=[];rendering=threading.Event();released=threading.Event()
    def fake_run(args,cell,coordinator):
        if len(finished)>=2:
            assert rendering.wait(2)
            released.set()
        folder=tmp_path/cell['task'];folder.mkdir();finished.append(cell['task'])
        time.sleep(.01)
        return {**cell,'status':'completed','episode_path':str(folder),'episode':{'task_success':False}}
    def fake_media(*_):
        rendering.set();assert released.wait(2)
    monkeypatch.setattr(module,'run_cell',fake_run)
    monkeypatch.setattr(module,'render_media',fake_media)
    monkeypatch.setattr(module,'verify_delivery',lambda *_:None)
    monkeypatch.setattr(module,'resources',lambda *_:{'MemAvailable':2**30})
    monkeypatch.setattr('scripts.univtac.summarize_shot_scaling.report',lambda *_:None)
    module.run(SimpleNamespace(output_root=tmp_path),settings)
    assert len(finished)==4
    assert all(c['status']=='completed' for c in json.loads((tmp_path/'results.json').read_text())['cells'])


def test_missing_is_not_native_failure_and_pairing_uses_same_seed():
    from scripts.univtac.summarize_shot_scaling import paired, statistics
    cells=[{'task':'t','seed':1,'comparison_condition':'B','condition':'B_1shot','shot':1,'status':'completed','episode':{'evaluable':True,'task_success':False}},
           {'task':'t','seed':2,'comparison_condition':'B','condition':'B_1shot','shot':1,'status':'initialization_unavailable'}]
    group=statistics(cells)['groups'][0]
    assert group['planned']==2 and group['evaluable']==1 and group['unavailable']==1
    assert group['success']==0 and group['wilson_95_evaluable'][1]>0
    result=paired({1:False,2:True},{1:True,3:False})
    assert result['valid_pairs']==1 and result['difference_pp']==100


def test_resume_does_not_erase_delivery_failure(tmp_path):
    c,f=make_cell(tmp_path)
    atomic_json(f/'episode.json',{'status':'completed','task_success':True,'evaluable':True})
    atomic_json(f/'worker_lifecycle.json',{'cleanup_complete':True})
    atomic_json(f/'protocol_delivery_error.json',{'error':'missing fourth expert'})
    result=recover_cell(tmp_path,c)
    assert result['delivery_or_runner_error']=='missing fourth expert'
    from scripts.univtac.summarize_shot_scaling import statistics
    result.update(comparison_condition='B',shot=1)
    assert statistics([result])['groups'][0]['evaluable']==0


def test_shared_demonstration_image_is_delivered_and_recorded(tmp_path):
    from PIL import Image

    from scripts.univtac.run_shot_scaling import check_mcp
    shared=tmp_path/'shared.png';Image.new('RGB',(12,12),'red').save(shared)
    package=tmp_path/'package';package.mkdir()
    b={'text':{'examples':[{'example_id':'official_episode_0','segments':[]}]},'images':[{'label':'head','path':str(shared)}]}
    c=copy.deepcopy(b);c['text']['historical_touch']=[]
    atomic_json(package/'visual_action_icl.json',b);atomic_json(package/'tactile_action_icl.json',c)
    result=check_mcp(package)
    assert result['B']['images']==1 and result['C']['native_mcp_pixels_match']
    rows=[json.loads(x) for x in (package/'offline_mcp_context.jsonl').read_text().splitlines()]
    assert rows[0]['response_image_paths']==[str(shared)]


def test_c_only_preserves_order_assignments_and_history():
    from scripts.univtac.run_shot_scaling import scoped_cells
    from scripts.univtac.summarize_shot_scaling import statistics
    cells=[{**c, 'status':'not_run'} for c in ordered_cells(TASKS,list(range(1000000,1000100)))]
    before=copy.deepcopy(cells)
    b=next(c for c in cells if c['comparison_condition']=='B')
    b.update(status='completed',episode={'evaluable':True,'task_success':False})
    revised=scoped_cells(cells,['C'])
    active=[c for c in revised if c['in_current_scope']]
    assert len(active)==1200
    assert [(c['task'],c['seed'],c['shot'],c['expert_ids']) for c in active]==[
        (c['task'],c['seed'],c['shot'],c['expert_ids']) for c in before if c['comparison_condition']=='C']
    historical=[c for c in revised if not c['in_current_scope']]
    assert sum(c['scope_disposition']=='executed_before_scope_revision' for c in historical)==1
    assert sum(c['scope_disposition']=='deferred_to_main_table' for c in historical)==1199
    assert b['episode']['task_success'] is False
    stats=statistics(revised,['C'])
    assert len(stats['groups'])==12 and sum(g['planned'] for g in stats['groups'])==1200
    assert not any('C-B' in c['comparison'] for c in stats['paired'])


def test_c_dispatch_skips_completed_grace_and_exhausted_initialization(tmp_path,monkeypatch):
    from scripts.univtac import run_shot_scaling as module
    cells=[]
    for i,condition in enumerate(['B_1shot','C_1shot','C_2shot','C_4shot']):
        c={'task':f'task{i}','seed':1000000,'condition':condition,
           'comparison_condition':condition[0],'cell_key':[i],'status':'not_run',
           'config':str(tmp_path/f'{i}.json'),'effective_config':{}}
        atomic_json(Path(c['config']),{})
        cells.append(c)
    settings={'media_workers':1,'video_playback_rate':.05,'minimum_disk_free_bytes':0}
    atomic_json(tmp_path/'manifest.json',{'settings':settings,'cells':cells})
    def recover(root,c):
        if c['condition']=='C_1shot':
            return {**c,'status':'initialization_unavailable','attempts':3}
        if c['condition']=='C_2shot':
            folder=tmp_path/'ended';folder.mkdir(exist_ok=True)
            atomic_json(folder/'protocol_delivery_check.json',{'passed':True})
            return {**c,'status':'completed','episode_path':str(folder),
                    'episode':{'evaluable':True,'task_success':True,'model_exit_mode':'terminal_grace_expired'}}
    dispatched=[]
    def execute(args,c,coordinator):
        dispatched.append(c['condition'])
        return {**c,'status':'initialization_unavailable','attempts':3}
    monkeypatch.setattr(module,'recover_cell',recover)
    monkeypatch.setattr(module,'run_cell',execute)
    monkeypatch.setattr(module,'render_media',lambda *_:None)
    monkeypatch.setattr(module,'resources',lambda *_:{})
    monkeypatch.setattr('scripts.univtac.summarize_shot_scaling.report',lambda *_:None)
    module.run(SimpleNamespace(output_root=tmp_path),{**settings,'dispatch_conditions':['C']})
    assert dispatched==['C_4shot']
    saved=json.loads((tmp_path/'results.json').read_text())
    assert saved['planned']==3 and len(saved['cells'])==4
    assert saved['cells'][0]['scope_disposition']=='deferred_to_main_table'
    assert saved['cells'][1]['attempts']==3
    assert saved['cells'][2]['status']=='completed'
