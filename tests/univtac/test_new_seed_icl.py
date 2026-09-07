"""New-seed scheduling, frozen inputs, paired outcomes and playback alignment."""
import json
from pathlib import Path

import yaml

from scripts.univtac.autonomous_dashboard import R19_HTML, load_autonomous_runs
from scripts.univtac.official_icl_review import paired_outcomes, paired_timeline, summarize
from scripts.univtac.run_autonomous_insert_hole import operator_prompt
from scripts.univtac.run_official_tactile_icl import condition_config, experiment_plan

REPO=Path(__file__).resolve().parents[2]
BASE=yaml.safe_load((REPO/'configs/univtac/new_seed_tactile_icl.yaml').read_text())


def test_fixed_new_seeds_and_unchanged_r_prompt():
    conditions,order=experiment_plan(BASE)
    assert conditions==('RA','RB','RC') and len(order)==len(set(order))==36
    assert sorted({s for s,_ in order})==list(range(1000006,1000018))
    expected=[('RA','RB','RC'),('RB','RC','RA'),('RC','RA','RB'),('RA','RC','RB'),('RC','RB','RA'),('RB','RA','RC')]*2
    assert order==[(s,c) for s,cs in zip(BASE['seeds'],expected,strict=True) for c in cs]
    old=yaml.safe_load((REPO/'configs/univtac/task_rule_tactile_icl.yaml').read_text())
    for c in conditions:
        assert operator_prompt(condition_config(BASE,c))==operator_prompt(condition_config(old,c))
    assert {k:v for k,v in BASE.items() if k not in ('round','seeds','episode_order')}=={k:v for k,v in old.items() if k not in ('round','seeds')}


def test_pair_categories_include_unavailable():
    cells=[]
    for seed,(a,b) in enumerate([(True,False),(False,True),(True,True),(False,False),(None,False)]):
        cells.extend([{'seed':seed,'condition':c,'evaluable':v is not None,'task_success':v} for c,v in [('RC',a),('RB',b)]])
    assert paired_outcomes(cells,'RC','RB')=={'method_only_success': 1, 'control_only_success': 1, 'both_success': 1, 'both_failure': 1, 'unavailable': 1}


def test_same_simulation_clock_holds_short_side_without_future_frames():
    streams=[[{'sample_id':i,'simulation_time_seconds':t} for i,t in enumerate(ts)] for ts in ([0,.1],[0,.1,.2])]
    mapping=paired_timeline(streams)
    assert [r['simulation_time_seconds'] for r in mapping]==[0,.1,.2]
    assert mapping[-1]['sample_ids']==[1,2] and mapping[-1]['ended']==[True,True]
    assert mapping[1]['ended']==[True,False]


def test_twelve_denominators_do_not_mix_r18(tmp_path):
    root=tmp_path/'univtac-isaac51-r19';batch=root/'batch';batch.mkdir(parents=True)
    _,order=experiment_plan(BASE)
    (batch/'run_manifest.json').write_text(json.dumps({'round':'R1.9','order':order}))
    for seed,c in order:
        folder=batch/f'seed_{seed}'/c;folder.mkdir(parents=True)
        (folder/'episode.json').write_text(json.dumps({'seed':seed,'condition':c,'evaluable':True,'task_success':c=='RC'}))
    summarize(root)
    d=json.loads((root/'results.json').read_text())
    assert len(d['cells'])==36 and d['groups']['RC']['successes']==12
    assert all(g['planned']==12 for g in d['groups'].values())
    assert d['paired_outcomes']['RC-RB']['method_only_success']==12
    assert d['contrasts_percentage_points']['RC-RB']==100
    groups=load_autonomous_runs(tmp_path,'R1.9')['batches']
    assert len(groups)==3 and all(len(g['episodes'])==12 for g in groups)
    assert 'function newSeedTable' in R19_HTML and 'function ruleTable' not in R19_HTML
    assert 'newSeedTable(d)' in R19_HTML and "valid.length+'/12" in R19_HTML
    assert '/r19-pairs' in R19_HTML
