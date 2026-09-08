"""Task dispatch and evidence boundaries for the second-task pilot."""
import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from scripts.univtac.autonomous_dashboard import R111_HTML, load_autonomous_runs
from scripts.univtac.official_icl_review import pair_specs, summarize
from scripts.univtac.prepare_official_demonstrations import segment_bounds
from scripts.univtac.run_autonomous_insert_hole import codex_command, operator_prompt
from scripts.univtac.run_official_tactile_icl import condition_config, experiment_plan
from sim.envs.univtac.autonomous_session import AutonomousSession

REPO=Path(__file__).resolve().parents[2]
BASE=yaml.safe_load((REPO/'configs/univtac/grasp_classify_tactile_icl.yaml').read_text())


def test_shared_rules_model_and_original_controls(tmp_path):
    conditions,order=experiment_plan(BASE)
    assert conditions==('A','B','C') and len(order)==len(set(order))==18
    assert [c for s,c in order if s==1000029]==['A','C','B']
    prompts=[]
    for c in conditions:
        cfg=condition_config(BASE,c)
        prompts.append(operator_prompt(cfg))
        command=codex_command(tmp_path,'http://unused',cfg)
        assert command[command.index('-m')+1]=='gpt-6-astra'
        assert 'model_reasoning_effort="low"' in command
        assert cfg['demonstration_condition']=={'A':'no_demo','B':'visual_action_icl','C':'tactile_action_icl'}[c]
    assert len(set(prompts))==1
    p=prompts[0]
    assert BASE['task_instruction'] in p and 'rough object -> orange' in p
    assert 'plain/smooth object -> green' in p and '0.965' in p
    assert 'Insert Hole' not in p and 'hole.' not in p and '0.04 m' not in p
    old=yaml.safe_load((REPO/'configs/univtac/new_seed_tactile_icl.yaml').read_text())
    for key in ('max_control_steps_per_move','max_joint_delta_rad','position_tolerance_m','orientation_tolerance_rad','tactile_history','max_move_requests','terminal_grace_seconds'):
        assert BASE[key]==old[key]
    assert not BASE.get('motion_pacing')


def test_one_native_motion_segment_not_three():
    assert segment_bounds([1]*47+[2],[b'move']*47+[b'delay'])==[(0,47)]


def test_task_capture_excludes_current_class(tmp_path, monkeypatch):
    import sim.envs.univtac.autonomous_session as module
    visible={'cameras':{v:{'rgb':{'path':v+'.png'}} for v in ('head','wrist')},
             'tactile':{v:{'rgb_marker':{'path':v+'.png'}} for v in ('left_tactile','right_tactile')}}
    seen=[]
    def capture(obs,**kw):
        seen.append(kw)
        return SimpleNamespace(snapshot=SimpleNamespace(operator_visible=visible,to_dict=lambda:visible))
    monkeypatch.setattr(module,'capture_snapshot',capture)
    s=object.__new__(AutonomousSession)
    s.config=BASE;s.root=tmp_path;s.seed=1000026;s.observation_index=s.move_requests=s.tool_count=0;s.feedback=None
    s.task=SimpleNamespace(_get_observations=dict,choice='rough',step_count=0,take_action_cnt=0,cfg=SimpleNamespace(step_lim=300))
    s.controller=SimpleNamespace(state=dict,counts=dict,terminal=lambda:None)
    result=s.capture()
    assert seen[0]['task_name']=='grasp_classify' and seen[0]['task_metadata']=={}
    assert result['task_instruction']==BASE['task_instruction']
    assert len(result['images'])==4 and 'rough' not in json.dumps(result)


def test_observe_only_finalization_does_not_check_success(tmp_path):
    s=object.__new__(AutonomousSession)
    s.config={**BASE,'observe_only':True};s.root=tmp_path;s.seed=999999
    s.task=SimpleNamespace(metadata={},plan_success=True,eval_success=False,choice='plain')
    def forbidden():
        raise AssertionError('observe-only must not invoke checker')
    s.controller=SimpleNamespace(check=forbidden,early_stop=False,counts=dict,grasp=None,terminal=lambda:None)
    s.recorder=None;s.infrastructure_error=s.finish_reason=None;s.tool_count=1;s.move_requests=0;s.started=0
    result=s.finalize('unscored_observation_check')
    assert result['task_success'] is None and not result['native_success_available']
    assert json.loads((tmp_path/'host_evaluator.json').read_text())['object_class']=='plain'
    assert 'object_class' not in result


def test_six_denominators_class_subgroups_and_bc_pairs(tmp_path):
    root=tmp_path/'univtac-isaac51-r111';batch=root/'batch';batch.mkdir(parents=True)
    _,order=experiment_plan(BASE)
    (batch/'run_manifest.json').write_text(json.dumps({'round':'R1.11','order':order}))
    for seed,c in order:
        folder=batch/f'seed_{seed}'/c;folder.mkdir(parents=True)
        (folder/'episode.json').write_text(json.dumps({'seed':seed,'condition':c,'evaluable':True,'task_success':c=='C'}))
        (folder/'host_evaluator.json').write_text(json.dumps({'object_class':'rough' if seed%2 else 'plain'}))
    summarize(root)
    result=json.loads((root/'results.json').read_text())
    assert result['groups']['C']['successes']==6 and result['groups']['A']['planned']==6
    assert result['contrasts_percentage_points']['C-B']==100
    assert result['class_subgroups_host_only']['C']['rough']=={'episodes':3,'successes':3}
    assert pair_specs(root)==[('B','C')]
    debug=root/'debug';debug.mkdir()
    (debug/'run_manifest.json').write_text(json.dumps({'round':'R1.11','mode':'observe_only'}))
    groups=load_autonomous_runs(tmp_path,'R1.11')['batches']
    assert [len(g['episodes']) for g in groups]==[6,6,6]
    assert '/r111-pairs' in R111_HTML and 'Insert Hole' not in R111_HTML.split('<script>')[0]


def test_completed_six_not_labelled_pending():
    assert 'valid(c)<12' not in R111_HTML
    assert 'valid(c)<6' in R111_HTML


def test_recovery_keeps_failed_attempt_and_stops_on_next_infrastructure(tmp_path, monkeypatch):
    import scripts.univtac.run_official_tactile_icl as runner
    root=tmp_path/'batch';a=root/'seed_1000026/A';b=root/'seed_1000026/B'
    a.mkdir(parents=True);b.mkdir()
    (a/'episode.json').write_text(json.dumps({'evaluable':True}))
    original={'evaluable':False,'reset_valid':False,'codex_process_count':0,'task_success':None}
    (b/'episode.json').write_text(json.dumps(original))
    _,order=experiment_plan(BASE)
    (root/'run_manifest.json').write_text(json.dumps({'round':'R1.11','order':order}))
    for c in 'ABC':(root/f'{c}.yaml').write_text(yaml.safe_dump(condition_config(BASE,c)))
    calls=[]
    def fake(args,config,seed,folder):
        calls.append((seed,config['condition'],folder))
        folder.mkdir(parents=True)
        return {'evaluable':False,'task_success':None,'infrastructure_error':'reset timeout','codex_process_count':0}
    monkeypatch.setattr(runner,'run_episode',fake)
    runner.resume_r111(SimpleNamespace(output_root=root))
    assert calls==[(1000026,'B',b/'attempt_2')]
    assert json.loads((b/'episode.json').read_text())==original
    assert json.loads((root/'recovery_manifest.json').read_text())['status']=='infrastructure_issue'
    import pytest
    with pytest.raises(FileExistsError):runner.resume_r111(SimpleNamespace(output_root=root))


def test_recovery_failure_is_not_replaced_by_original_and_pairs_exclude_missing(tmp_path):
    from scripts.univtac.official_icl_review import effective_episode_folder
    root=tmp_path/'univtac-isaac51-r111';batch=root/'batch';batch.mkdir(parents=True)
    _,order=experiment_plan(BASE)
    (batch/'run_manifest.json').write_text(json.dumps({'round':'R1.11','order':order}))
    for seed,c in order:
        folder=batch/f'seed_{seed}'/c;folder.mkdir(parents=True)
        (folder/'episode.json').write_text(json.dumps({'seed':seed,'evaluable':True,'task_success':True}))
    b=batch/'seed_1000026/B';second=b/'attempt_2';second.mkdir()
    (second/'episode.json').write_text(json.dumps({'seed':1000026,'evaluable':False,'task_success':None,'infrastructure_error':'reset'}))
    assert effective_episode_folder(b)==second
    summarize(root)
    d=json.loads((root/'results.json').read_text())
    assert d['groups']['B']['evaluable']==5
    assert d['paired_comparisons']['C-B']=={'evaluable_pairs':5,'difference_percentage_points':0}
    assert d['contrasts_percentage_points']['C-B'] is None
    groups=load_autonomous_runs(tmp_path,'R1.11')['batches']
    bgroup=next(g for g in groups if g['name'].endswith('/ B'))
    assert len(bgroup['episodes'])==6
    assert bgroup['episodes'][0]['run'].endswith('/B/attempt_2')


def test_timed_recovery_only_runs_fixed_ten_and_accepts_native_failure(tmp_path,monkeypatch):
    import scripts.univtac.run_official_tactile_icl as runner
    root=tmp_path/'batch';(root/'seed_1000026/A').mkdir(parents=True);(root/'seed_1000028/B').mkdir(parents=True)
    (root/'seed_1000026/A/episode.json').write_text(json.dumps({'evaluable':True}))
    (root/'seed_1000028/B/episode.json').write_text(json.dumps({'reset_valid':False,'codex_process_count':0}))
    _,order=experiment_plan(BASE)
    (root/'run_manifest.json').write_text(json.dumps({'round':'R1.11','order':order}))
    for c in 'ABC':(root/f'{c}.yaml').write_text(yaml.safe_dump(condition_config(BASE,c)))
    calls=[]
    def fake(args,config,seed,folder):
        assert args.record_reset_timing
        assert operator_prompt(config)==operator_prompt(condition_config(BASE,'A'))
        calls.append((seed,config['condition'],folder));folder.mkdir(parents=True)
        return {'evaluable':True,'task_success':False,'infrastructure_error':None}
    monkeypatch.setattr(runner,'run_episode',fake)
    runner.resume_r111(SimpleNamespace(output_root=root,resume_r111_timed=True))
    assert [(s,c) for s,c,_ in calls]==order[8:] and len(calls)==10
    assert calls[0][2]==root/'seed_1000028/B/attempt_2'
    record=json.loads((root/'timed_recovery_manifest.json').read_text())
    assert record['status']=='completed' and record['total_invocation_limit']==22



def test_extended_recovery_changes_only_native_wait_and_preserves_two_failures(tmp_path,monkeypatch):
    import scripts.univtac.run_official_tactile_icl as runner
    from scripts.univtac.official_icl_review import (
        effective_episode_folder,
        previous_episode_attempts,
    )
    root=tmp_path/'batch';(root/'seed_1000026/A').mkdir(parents=True)
    b=root/'seed_1000028/B';(b/'attempt_2').mkdir(parents=True)
    (root/'seed_1000026/A/episode.json').write_text(json.dumps({'evaluable':True}))
    failed={'reset_valid':False,'codex_process_count':0}
    for folder in (b,b/'attempt_2'):(folder/'episode.json').write_text(json.dumps(failed))
    _,order=experiment_plan(BASE)
    (root/'run_manifest.json').write_text(json.dumps({'round':'R1.11','order':order}))
    for c in 'ABC':(root/f'{c}.yaml').write_text(yaml.safe_dump(condition_config(BASE,c)))
    calls=[]
    def fake(args,config,seed,folder):
        assert args.record_reset_timing and config['native_reset_time_limit_seconds']==600
        old=condition_config(BASE,config['condition'])
        assert {k:v for k,v in config.items() if k!='native_reset_time_limit_seconds'}==old
        assert operator_prompt(config)==operator_prompt(old)
        assert args.config.parent==root/'reset_limit_recovery_configs'
        calls.append((seed,config['condition'],folder));folder.mkdir(parents=True)
        return {'evaluable':True,'task_success':False,'infrastructure_error':None}
    monkeypatch.setattr(runner,'run_episode',fake)
    runner.resume_r111(SimpleNamespace(output_root=root,resume_r111_reset_limit=True))
    assert len(calls)==10 and calls[0][2]==b/'attempt_3'
    assert effective_episode_folder(b)==b/'attempt_3'
    assert previous_episode_attempts(b)==[failed,failed]
    record=json.loads((root/'reset_limit_recovery_manifest.json').read_text())
    assert record['total_invocation_limit']==23 and record['previous_invocations']==13
    assert 'native_reset_time_limit_seconds' not in yaml.safe_load((root/'B.yaml').read_text())
