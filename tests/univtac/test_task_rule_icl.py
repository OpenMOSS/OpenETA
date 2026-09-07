"""Offline checks of the information intervention and its actual launch route."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import yaml

from scripts.univtac.autonomous_dashboard import R18_HTML, load_autonomous_runs
from scripts.univtac.official_icl_review import RULE_NAMES, summarize
from scripts.univtac.run_autonomous_insert_hole import (
    codex_command,
    demonstration_projection,
    operator_prompt,
)
from scripts.univtac.run_official_tactile_icl import RULE_CONDITIONS, RULE_ORDER, condition_config
from sim.envs.univtac.autonomous_session import AutonomousSession

REPO = Path(__file__).resolve().parents[2]
BASE = yaml.safe_load((REPO/'configs/univtac/task_rule_tactile_icl.yaml').read_text())


def test_exact_order_and_model_launch_parameters(tmp_path):
    assert len(RULE_ORDER) == len(set(RULE_ORDER)) == 18
    assert [c for seed,c in RULE_ORDER if seed==1000004] == ['RB','UB','RC','UC','RA','UA']
    for c in RULE_CONDITIONS:
        config = condition_config(BASE, c)
        cmd = codex_command(tmp_path,'http://localhost:1234',config)
        assert cmd[cmd.index('-m')+1] == 'gpt-6-astra'
        assert 'model_reasoning_effort="low"' in cmd
        assert '--ignore-user-config' in cmd
        assert cmd[-1] == operator_prompt(config)
        assert 'review_demonstrations' in ' '.join(cmd)


def test_only_r_gets_static_rules_and_common_prompt_is_identical():
    prompts = {c:operator_prompt(condition_config(BASE,c)) for c in RULE_CONDITIONS}
    assert prompts['UA']==prompts['UB']==prompts['UC']
    assert prompts['RA']==prompts['RB']==prompts['RC']
    assert prompts['RA'] == prompts['UA']+'\n'+BASE['public_task_rules']+'\n'
    for c,prompt in prompts.items():
        assert BASE['task_instruction'] in prompt
        assert 'Insert the held object into the hole.' not in prompt
        assert '300 steps' in prompt and '80 control steps' in prompt
        assert '30 admitted non-preview' in prompt
        if c.startswith('U'):
            assert 'public_task_rules' not in condition_config(BASE,c)
            assert '0.04' not in prompt and '0.99' not in prompt and '0.01 m' not in prompt
    assert 'At exactly 0.04 m' in prompts['RA']


def test_projection_keeps_bc_common_data_and_does_not_insert_rules(tmp_path):
    common={'task_goal':'Insert the held object into the hole.','examples':[{'movement':[1,2,3]}]}
    vision={'text':common,'images':[{'path':'vision.png','label':'vision'}]}
    touch=copy.deepcopy(vision);touch['images'].append({'path':'touch.png','label':'touch'})
    for name,p in [('visual_action_icl',vision),('tactile_action_icl',touch)]:
        (tmp_path/f'{name}.json').write_text(json.dumps(p))
    base={**BASE,'demonstration_package':str(tmp_path)}
    b=demonstration_projection(condition_config(base,'RB'))
    c=demonstration_projection(condition_config(base,'RC'))
    assert b['text']==c['text']=={**common,'task_goal':BASE['task_instruction']}
    assert c['images'][:len(b['images'])]==b['images']
    assert '0.04' not in json.dumps(c)
    assert json.loads((tmp_path/'visual_action_icl.json').read_text())==vision


def test_observe_and_snapshot_deliver_same_official_instruction(tmp_path, monkeypatch):
    import sim.envs.univtac.autonomous_session as module
    visible={'cameras':{v:{'rgb':{'path':v+'.png'}} for v in ('head','wrist')},
             'tactile':{v:{'rgb_marker':{'path':v+'.png'}} for v in ('left_tactile','right_tactile')}}
    seen=[]
    def capture(obs, **kwargs):
        seen.append(kwargs['task_instruction'])
        return SimpleNamespace(snapshot=SimpleNamespace(operator_visible=visible,to_dict=lambda:visible))
    monkeypatch.setattr(module,'capture_snapshot',capture)
    for c in ('UA','RA'):
        session=object.__new__(AutonomousSession)
        session.config=condition_config(BASE,c);session.root=tmp_path/c;session.seed=1000003
        session.observation_index=0;session.move_requests=session.tool_count=0;session.feedback=None
        session.task=SimpleNamespace(_get_observations=dict,step_count=0,take_action_cnt=0,cfg=SimpleNamespace(step_lim=300))
        session.controller=SimpleNamespace(state=dict,counts=dict,terminal=lambda:None)
        result=session.capture()
        assert result['task_instruction']==BASE['task_instruction']
        assert '0.04' not in json.dumps(result)
    assert seen==[BASE['task_instruction']]*2


def test_six_groups_and_fixed_percentage_point_contrasts(tmp_path):
    root=tmp_path/'univtac-isaac51-r18';batch=root/'batch';batch.mkdir(parents=True)
    (batch/'run_manifest.json').write_text(json.dumps({'round':'R1.8'}))
    for seed,c in RULE_ORDER:
        folder=batch/f'seed_{seed}'/c;folder.mkdir(parents=True)
        (folder/'episode.json').write_text(json.dumps({'seed':seed,'condition':c,'evaluable':True,
            'task_success':c=='RC' and seed==1000003,'model':'gpt-6-astra','reasoning_effort':'low'}))
    data=load_autonomous_runs(tmp_path,'R1.8')
    assert len(data['batches'])==6
    assert all(len(b['episodes'])==3 for b in data['batches'])
    summarize(root)
    result=json.loads((root/'results.json').read_text())
    assert len(result['cells'])==18 and result['groups']['RC']['successes']==1
    assert result['contrasts_percentage_points']['RC-RB']==100/3
    assert 'ruleTable(d)' in R18_HTML and 'function ruleTable' in R18_HTML
    assert 'function primaryTable' not in R18_HTML
    assert 'e.prompt' in R18_HTML and 'gpt' not in RULE_NAMES['UA'].lower()
    assert '/r18-videos' in R18_HTML
