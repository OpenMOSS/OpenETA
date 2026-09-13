#!/usr/bin/env python3
"""Prepare and summarize the authorized A0/B2 campaign using the existing runner."""
from __future__ import annotations
import argparse
import copy
import html
import json
import math
import os
from pathlib import Path
import random
import shlex
import subprocess
import sys

import numpy as np
import yaml
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.univtac.run_autonomous_insert_hole import operator_prompt, codex_command
from scripts.univtac.run_shot_scaling import atomic_json, check_mcp, load
from scripts.univtac.summarize_shot_scaling import artifact, paired
from sim.envs.univtac.feedback_protocol import PROTOCOL

TASKS = ['insert_tube', 'lift_can', 'lift_bottle', 'pull_out_key']
PROJECTIONS = {'A': 'no_demo', 'B': 'visual_action_icl', 'C': 'tactile_action_icl'}
EXPECTED_C = dict(zip(TASKS, [32, 2, 36, 26]))


def fixed_cells(seeds):
    assert seeds == list(range(1000000, 1000100))
    pairs = [(task, seed) for task in TASKS for seed in seeds]
    random.Random(20260913).shuffle(pairs)
    cells = []
    for task, seed in pairs:
        t, i = TASKS.index(task), seed - 1000000
        host = 'local' if (i+t)%2 == 0 else 'hzz-server'
        for condition in (['A', 'B_2shot'] if (i//2+t)%2 == 0 else ['B_2shot', 'A']):
            a = condition == 'A'
            row = dict(task=task, seed=seed, condition=condition, shot=0 if a else 2,
                       expert_ids=[] if a else [0, 1], demonstration_set_id='none' if a else 'official_0_1',
                       comparison_condition='A' if a else 'B', query_index=i, host=host,
                       feedback_protocol=PROTOCOL, phase='main_AB', status='not_run')
            row['cell_key'] = [task, seed, condition, row['shot'], row['demonstration_set_id'], PROTOCOL]
            cells.append(row)
    return cells


def reference_inputs(rows):
    """Read the selected C identities on their owning host, never the placeholder host."""
    code = '''import json,sys
from pathlib import Path
result=[]
for path in json.load(sys.stdin):
 p=Path(path)
 def read(n):
  q=p/n
  return json.loads(q.read_text()) if q.exists() else None
 result.append(dict(path=path,prompt=(p/'prompt.txt').read_text(),projection=read('demonstrations/projection.json'),episode=read('episode.json'),final=read('final_result.json'),worker=read('worker_lifecycle.json'),model=read('codex_lifecycle.json'),trace_summary=read('codex_trace_summary.json'),initialization=read('native_reset_limit.json')))
print(json.dumps(result))'''
    result = {}
    for host in ['local', 'hzz-server']:
        paths = [r['episode_path'] for r in rows if r['host'] == host]
        cmd = [sys.executable, '-c', code] if host == 'local' else ['ssh', '-o', 'BatchMode=yes', host, 'python3 -c '+shlex.quote(code)]
        data = json.loads(subprocess.check_output(cmd, input=json.dumps(paths), text=True))
        result.update({d['path']: d for d in data})
    return result


def normalized_projection(value):
    value = copy.deepcopy(value)
    for im in value['images']:
        # Host roots differ; the shared artifact's repository-relative path is stable.
        im['path'] = str(im['path']).split('/outputs/', 1)[-1]
    return value


def validate(manifest):
    cells, refs = manifest['cells'], manifest['references']
    assert manifest['phase'] == 'main_AB' and manifest['historical_AB_reuse'] is False
    expected = fixed_cells(manifest['seeds'])
    assert [(c['cell_key'], c['host']) for c in cells] == [(c['cell_key'], c['host']) for c in expected]
    assert len(refs) == 400 and len({tuple(c['cell_key']) for c in cells+refs}) == 1200
    for task in TASKS:
        cr = [r for r in refs if r['task'] == task]
        assert len(cr) == 100 and sum(r['task_success'] is True for r in cr) == EXPECTED_C[task]
        assert sorted(r['seed'] for r in cr) == manifest['seeds']
        assert all(r['condition']=='C_2shot' and r['shot']==2 and r['evaluable'] for r in cr)
    for c in cells:
        cfg = c['original_config']
        expected_cfg = dict(model='gpt-6-astra', reasoning_effort='low', feedback_protocol=PROTOCOL,
            current_tactile=True, demonstrations=True, disable_initialization_timeout=True,
            max_move_requests=30, max_tool_calls=100, max_control_steps_per_move=80,
            codex_timeout_seconds=3600, terminal_grace_seconds=300, shutdown_timeout_seconds=300,
            defer_review_video=True, shared_demonstration_media=True,
            native_control_step_limit=500 if c['task']=='lift_bottle' else 300)
        assert all(cfg.get(k)==v for k,v in expected_cfg.items()), c['cell_key']
        assert not cfg.get('motion_pacing') and cfg['cell_key']==c['cell_key']
        assert cfg['expert_ids']==c['expert_ids']
        assert operator_prompt(cfg)==manifest['frozen_prompts'][c['task']]
        assert 'check_task' not in ' '.join(codex_command(REPO/'outputs/offline-unused', 'http://unused', cfg))
    return dict(AB=800, C_references=400, C_success=96, historical_AB_reused=0,
                disjoint_hosts=True, frozen_prompt_budget_match=True)


def prepare(root):
    assert not (root/'manifest.json').exists(), 'Never overwrite a frozen campaign'
    seeds = load(REPO/'configs/univtac/main_query_seeds.json')
    refs = [r for r in load(REPO/'outputs/univtac-shot-scaling-analysis/selected_results.json') if r['shot']==2]
    inputs = reference_inputs(refs)
    prompts, checks, configs = {}, {}, {}
    for task in TASKS:
        cfg = yaml.safe_load((REPO/f'outputs/univtac-shot-scaling/configs/{task}/1000000/C_2shot.yaml').read_text())
        source = REPO/f'outputs/univtac-shot-scaling/demonstrations/{task}/official_0_1'
        c, b = [load(source/f'{name}.json') for name in ['tactile_action_icl', 'visual_action_icl']]
        common = copy.deepcopy(c['text']); common.pop('historical_touch')
        assert common == b['text'] and c['images'][:len(b['images'])] == b['images']
        assert [e['example_id'] for e in common['examples']] == ['official_episode_0', 'official_episode_1']
        delivered = copy.deepcopy(c); delivered['text']['task_goal'] = cfg['task_instruction']
        for row in [r for r in refs if r['task']==task]:
            data = inputs[row['episode_path']]
            assert data['prompt']==operator_prompt(cfg)
            assert normalized_projection(data['projection']) == normalized_projection(delivered), row['cell_key']
            assert data['episode']['task_success'] is row['task_success'] and data['worker']['cleanup_complete']
            row.update(episode=data['episode'], native_initialization=data['initialization'],
                       costs=dict(usage=(data['trace_summary'] or {}).get('usage'),
                                  codex_seconds=(data['model'] or {}).get('elapsed_seconds'),
                                  model_exit_mode=(data['model'] or {}).get('exit_mode')),
                       readonly_reference=True, comparison_condition='C')
        prompts[task] = operator_prompt(cfg)
        package = root/'demonstrations'/task
        package.mkdir(parents=True, exist_ok=True)
        for name, projection in [('visual_action_icl', b), ('tactile_action_icl', c)]:
            for image in projection['images']:
                image['path'] = os.path.relpath(source/image['path'], package)
            atomic_json(package/f'{name}.json', projection)
        atomic_json(package/'no_demo.json', {'text':{'examples':[], 'message':'No historical demonstrations are provided.'}, 'images':[]})
        checks[task] = check_mcp(package, PROJECTIONS)
        cfg.update(demonstration_package=str(package.relative_to(REPO)), disable_initialization_timeout=True,
                   native_reset_time_limit_seconds=None, round='Four-task main A0/B2; historical C2 references')
        configs[task] = cfg
    cells = fixed_cells(seeds)
    for cell in cells:
        cfg = {**configs[cell['task']], **cell, 'seeds':[cell['seed']],
               'demonstration_condition':PROJECTIONS[cell['comparison_condition']],
               'expert_source_seeds':[] if cell['shot']==0 else [0,1]}
        cell.update(original_config=cfg, mcp_expectation=checks[cell['task']][cell['comparison_condition']])
    manifest = dict(phase='main_AB', historical_AB_reuse=False, seeds=seeds, cells=cells,
                    references=refs, frozen_prompts=prompts, initialization_policy='startup_deadlines_disabled',
                    model='gpt-6-astra', reasoning_effort='low', pair_shuffle_seed=20260913,
                    slots_per_host=2, media_workers=1, local_weekly_stop_remaining_percent=2)
    result = validate(manifest)
    atomic_json(root/'manifest.json', manifest)
    atomic_json(root/'reused_results.json', {'historical_AB':[], 'C2':refs})
    atomic_json(root/'remaining_AB.json', {'cells':cells})
    atomic_json(root/'offline_validation.json', {**result, 'demonstrations':checks, 'all_400_saved_C2_prompts_and_projections_match':True})
    for host in ['local', 'hzz-server']:
        atomic_json(root/f'queue_{host}.json', [c['cell_key'] for c in cells if c['host']==host])
    report(root)
    print(json.dumps(result), flush=True)


def exact_pair(left, right):
    result = paired(left, right)
    n = result.get('left_only_success',0)+result.get('right_only_success',0)
    k = min(result.get('left_only_success',0),result.get('right_only_success',0))
    result['p_exact'] = min(1., 2*sum(math.comb(n,i) for i in range(k+1))/2**n) if n else 1.
    result['unpaired'] = 100-result['valid_pairs']
    return result


def campaign_status(cells, host_states, media_passed):
    if any(h.get('paused') for h in host_states.values()):
        return 'paused'
    if any(c.get('delivery_or_runner_error') for c in cells):
        return 'review_required'
    if all(c.get('episode',{}).get('evaluable') for c in cells):
        return 'completed' if media_passed == len(cells) else 'media_pending'
    return 'running' if host_states else 'prepared'


def report(root):
    manifest = load(root/'manifest.json')
    lookup = {tuple(c['cell_key']):dict(c) for c in manifest['cells']}
    host_states = {}
    for host in ['local', 'hzz-server']:
        path = root/'hosts'/host/'results.json'
        if path.exists():
            data=load(path); host_states[host]={k:v for k,v in data.items() if k!='cells'}
            for c in data['cells']:
                key=tuple(c['cell_key']); assert lookup[key]['host']==host
                lookup[key]=c
    cells=list(lookup.values())
    rows=cells+manifest['references']
    groups=[]; comparisons=[]
    for task in TASKS:
        values={}
        for condition in ['A','B_2shot','C_2shot']:
            rr=[r for r in rows if r['task']==task and r['condition']==condition]
            native=[r for r in rr if r.get('episode',{}).get('evaluable') or r.get('readonly_reference')]
            valid=[r for r in native if not r.get('delivery_or_runner_error')]
            outcomes={r['seed']:bool(r.get('episode',r).get('task_success')) for r in valid}
            values[condition]=outcomes
            groups.append(dict(task=task,condition=condition,planned=100,evaluable=len(valid),success=sum(outcomes.values()),
                missing=100-len(native), native_evaluable=len(native), delivery_review=len(native)-len(valid),
                usage_available=sum(r.get('costs',{}).get('usage') is not None for r in valid)))
        for left,right in [('B_2shot','C_2shot'),('A','B_2shot'),('A','C_2shot')]:
            comparisons.append(dict(task=task,comparison=right+'-'+left,**exact_pair(values[left],values[right])))
    primary=[c for c in comparisons if c['comparison']=='C_2shot-B_2shot']
    previous=0
    for rank,c in enumerate(sorted(primary,key=lambda x:x['p_exact'])):
        previous=max(previous,(4-rank)*c['p_exact']);c['p_holm_four_tasks']=min(1.,previous)
    stats=dict(groups=groups,paired=comparisons,hosts=host_states,
        note='Four configuration-selection tasks; historical C noncontemporaneous; current table retains all valid outcomes. Bootstrap uses existing paired routine, seed20260908/2000 resamples. Four-test primary Holm family; incomplete results provisional.')
    atomic_json(root/'statistics.json', stats)
    media_passed = sum(c.get('media_status') == 'completed' for c in cells)
    atomic_json(root/'run_manifest.json', {'phase':'main_AB', 'planned_AB':800,
        'readonly_C2':400, 'historical_AB_reused':0,
        'status':campaign_status(cells, host_states, media_passed), 'AB_media_passed':media_passed,
        'AB_valid_results':sum(bool(c.get('episode',{}).get('evaluable')) for c in cells),
        'hosts':host_states})
    atomic_json(root/'main_table_index.json', {'cells':rows})
    atomic_json(root/'remaining_AB.json', {'cells':[c for c in cells if not c.get('episode',{}).get('evaluable')],
        'locked_for_delivery_review':[c for c in cells if c.get('delivery_or_runner_error')]})
    page='<meta charset="utf-8"><meta http-equiv="refresh" content="60"><h1>四任务 ABC：A0 / B2 / 历史 C2</h1><p>旧A/B不复用；C2仅只读引用。按原生判据，配置选择任务及非同期限制保留。</p><table border="1"><tr><th>任务</th><th>条件</th><th>成功/可评价/计划</th><th>缺结果</th><th>usage覆盖</th></tr>'
    for g in groups:
        page+=f'<tr><td>{g["task"]}</td><td>{g["condition"]}</td><td>{g["success"]}/{g["evaluable"]}/100</td><td>{g["missing"]}</td><td>{g["usage_available"]}/{g["evaluable"]}</td></tr>'
    page+='</table><pre>'+html.escape(json.dumps(host_states,ensure_ascii=False,indent=2))+'</pre><table border="1">'
    for c in cells:
        link=f'<a href="{artifact(Path(c["episode_path"])/"review.html")}">记录/慢放</a>' if c.get('episode_path') else ''
        page+=f'<tr><td>{c["task"]}/{c["seed"]}/{c["condition"]}</td><td>{c["host"]}</td><td>{c.get("status")}</td><td>{link}</td></tr>'
    (root/'report.html').write_text(page+'</table>')
    return stats


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--output-root',type=Path,required=True);p.add_argument('--phase',choices=['prepare','validate','report'],required=True);args=p.parse_args();root=args.output_root.resolve();root.mkdir(parents=True,exist_ok=True)
    if args.phase=='prepare':prepare(root)
    elif args.phase=='validate':print(json.dumps(validate(load(root/'manifest.json'))))
    else:report(root)
