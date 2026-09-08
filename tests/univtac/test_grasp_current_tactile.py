"""R1.12 task wiring and cost accounting, without physical/model execution."""
import json
from pathlib import Path

import yaml

from scripts.univtac.autonomous_dashboard import R112_HTML, load_autonomous_runs
from scripts.univtac.official_icl_review import pair_specs, round_page, summarize
from scripts.univtac.run_autonomous_insert_hole import codex_command, operator_prompt
from scripts.univtac.run_official_tactile_icl import condition_config, experiment_plan

REPO = Path(__file__).resolve().parents[2]
BASE = yaml.safe_load((REPO/'configs/univtac/grasp_classify_current_tactile_ablation.yaml').read_text())


def test_frozen_grasp_task_four_inputs_and_launch(tmp_path):
    old = yaml.safe_load((REPO/'configs/univtac/grasp_classify_tactile_icl.yaml').read_text())
    changed = {'round','seeds','episode_order','current_tactile_ablation',
               'native_reset_time_limit_seconds','record_reset_timing','missing_modality_notice'}
    assert {k:v for k,v in BASE.items() if k not in changed} == {k:v for k,v in old.items() if k not in changed}
    conditions, order = experiment_plan(BASE)
    patterns = [('B_live','C_live','C_no_live','B_no_live'),('C_live','B_no_live','B_live','C_no_live'),
                ('B_no_live','C_no_live','C_live','B_live'),('C_no_live','B_live','B_no_live','C_live')]*2
    assert order == [(s,c) for s,cs in zip(range(1000032,1000040),patterns,strict=True) for c in cs]
    configs = [condition_config(BASE,c) for c in conditions]
    assert len({operator_prompt(c) for c in configs}) == 1
    for c in configs:
        prompt = operator_prompt(c)
        assert BASE['missing_modality_notice'] in prompt
        assert BASE['task_instruction'] in prompt and BASE['public_task_rules'] in prompt
        assert 'when provided' in prompt and 'Insert Hole' not in prompt
        assert c['current_tactile'] == (c['condition'] in ('B_live','C_live'))
        assert c['demonstration_condition'] == ('visual_action_icl' if c['condition'][0]=='B' else 'tactile_action_icl')
        cmd = codex_command(tmp_path,'http://unused',c)
        assert cmd[cmd.index('-m')+1] == 'gpt-6-astra'
        assert 'model_reasoning_effort="low"' in cmd
        assert c['native_reset_time_limit_seconds'] == 600 and c['record_reset_timing']


def test_results_preserve_failures_missing_costs_and_joint_success(tmp_path):
    root=tmp_path/'univtac-isaac51-r112';batch=root/'batch';batch.mkdir(parents=True)
    _,order=experiment_plan(BASE)
    (batch/'run_manifest.json').write_text(json.dumps({'round':'R1.12','order':order,'config':BASE}))
    for seed,c in order[:8]:
        folder=batch/f'seed_{seed}'/c;folder.mkdir(parents=True)
        good=not (seed==1000033 and c=='B_no_live')
        (folder/'episode.json').write_text(json.dumps({'seed':seed,'condition':c,'evaluable':True,
            'task_success':good,'control_steps':10 if good else 300,'move_request_count':1 if good else 8}))
        (folder/'codex_trace_summary.json').write_text(json.dumps({'usage':{'input_tokens':100,'cached_input_tokens':80,'output_tokens':10}}))
    summarize(root);d=json.loads((root/'results.json').read_text())
    key='B_live-B_no_live'
    assert d['groups']['B_no_live']['native_failures']==1
    assert d['groups']['B_no_live']['planned']==8
    assert d['paired_comparisons'][key]=={'evaluable_pairs':2,'difference_percentage_points':50.0}
    assert d['paired_outcomes'][key]['unavailable']==6
    assert d['contrasts_percentage_points'][key] is None
    costs=d['costs'];assert costs['all_evaluable']['B_no_live']['control_steps']==310
    joint=costs['jointly_successful_pairs'][key]
    assert joint['paired_seeds']==[1000032] and joint['B_no_live']['control_steps']==10
    assert costs['all_evaluable']['B_live']['usage']['input_tokens']==200
    assert costs['all_evaluable']['B_live']['usage']['reasoning_output_tokens'] is None
    assert costs['all_evaluable']['B_live']['codex_wall_seconds'] is None
    assert len(load_autonomous_runs(tmp_path,'R1.12')['batches'])==4
    assert 'r112-autonomous' in round_page(root)
    assert pair_specs(root)==[('B_live','B_no_live'),('C_live','C_no_live')]
    assert 'Grasp & Classify' in R112_HTML and 'Insert Hole' not in R112_HTML
    assert '仅供用户审阅，本episode未送给Agent' in R112_HTML
