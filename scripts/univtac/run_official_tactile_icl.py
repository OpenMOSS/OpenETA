"""Run the fixed official-demonstration pilot through the existing autonomous live runner."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path

import yaml

from scripts.univtac.run_autonomous_insert_hole import REPO, operator_prompt, run_episode
from sim.envs.univtac.trace import write_json

CONDITIONS = ('no_demo', 'visual_action_icl', 'tactile_action_icl')
ORDER = [(1000003, c) for c in CONDITIONS] + [
    (1000004, c) for c in CONDITIONS[1:] + CONDITIONS[:1]
] + [(1000005, c) for c in CONDITIONS[2:] + CONDITIONS[:2]]


RULE_CONDITIONS = ('UA', 'UB', 'UC', 'RA', 'RB', 'RC')
RULE_ORDER = [(1000003, c) for c in ('UA', 'RA', 'UB', 'RB', 'UC', 'RC')] + [
    (1000004, c) for c in ('RB', 'UB', 'RC', 'UC', 'RA', 'UA')
] + [(1000005, c) for c in ('UC', 'RC', 'UA', 'RA', 'UB', 'RB')]


def experiment_plan(base):
    if base.get('episode_order'):
        order = [tuple(row) for row in base['episode_order']]
        return tuple(sorted({c for _,c in order})), order
    return (RULE_CONDITIONS, RULE_ORDER) if base.get('task_rule_ablation') else (CONDITIONS, ORDER)


def condition_config(base, condition):
    config = copy.deepcopy(base)
    config['condition'] = condition
    if base.get('current_tactile_ablation'):
        config['task_information'] = 'R'
        config['demonstration_condition'] = CONDITIONS['ABC'.index(condition[0])]
        config['current_tactile'] = condition in ('B_live', 'C_live')
    elif base.get('task_information') == 'R' and condition in ('A', 'B', 'C'):
        config['demonstration_condition'] = CONDITIONS['ABC'.index(condition)]
    elif base.get('task_rule_ablation'):
        config['task_information'] = condition[0]
        config['demonstration_condition'] = CONDITIONS['ABC'.index(condition[1])]
        # U workers do not need the R-only text, even in their host configuration.
        if condition[0] == 'U':
            config.pop('public_task_rules')
    return config


def resume_r111(args):
    """One explicitly authorized initialization recovery, then the untouched tail."""
    root = args.output_root.resolve()
    extended = getattr(args, 'resume_r111_reset_limit', False)
    timed = extended or getattr(args, 'resume_r111_timed', False)
    recovery_seed = 1000028 if timed else 1000026
    start_index = 8 if timed else 1
    record_path = root/('reset_limit_recovery_manifest.json' if extended else 'timed_recovery_manifest.json' if timed else 'recovery_manifest.json')
    manifest = json.loads((root/'run_manifest.json').read_text())
    first = json.loads((root/'seed_1000026/A/episode.json').read_text())
    failed_path = root/f'seed_{recovery_seed}'/'B'
    if extended:
        failed_path = failed_path/'attempt_2'
    failed = json.loads((failed_path/'episode.json').read_text())
    if manifest['round'] != 'R1.11' or not first['evaluable'] or failed['codex_process_count'] != 0 or failed['reset_valid']:
        raise ValueError('This recovery is only for the recorded R1.11 B initialization failure')
    record = {'authorization':('Pro 32151410-590d-4469-9d8c-c5e734294536' if extended else 'Pro ea3a4ea7-4666-4444-99e0-eccce5c4a742' if timed else 'Pro b42baf5b-0e18-467a-88e7-162a7a2149bf'),
              'total_invocation_limit':23 if extended else 22 if timed else 20, 'previous_invocations':13 if extended else 12 if timed else 3,
              'maximum_new_invocations':10 if timed else 17, 'record_reset_timing':timed,
              'native_reset_time_limit_seconds':600 if extended else 120, 'status':'running', 'attempts':[],
              'repo_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()}
    with record_path.open('x') as f:
        json.dump(record, f, indent=2)
    manifest['status'] = 'recovering'
    write_json(root/'run_manifest.json', manifest)
    config_root = root
    if extended:
        config_root = root/'reset_limit_recovery_configs'
        config_root.mkdir()
        for condition in ('A','B','C'):
            config = yaml.safe_load((root/f'{condition}.yaml').read_text())
            config['native_reset_time_limit_seconds'] = 600.0
            (config_root/f'{condition}.yaml').write_text(yaml.safe_dump(config,sort_keys=False))
    args.mode = 'batch'
    args.record_reset_timing = timed
    for seed, condition in manifest['order'][start_index:]:
        args.config = config_root/f'{condition}.yaml'
        config = yaml.safe_load(args.config.read_text())
        folder = root/f'seed_{seed}'/condition
        if (seed, condition) == (recovery_seed, 'B'):
            folder = folder/('attempt_3' if extended else 'attempt_2')
        print(json.dumps({'starting':seed,'condition':condition,'attempt_root':str(folder)}),flush=True)
        episode = run_episode(args,config,seed,folder)
        episode['condition'] = condition
        write_json(folder/'episode.json',episode)
        record['attempts'].append({'seed':seed,'condition':condition,'path':str(folder.relative_to(root)),
                                   'evaluable':episode['evaluable'],'infrastructure_error':episode.get('infrastructure_error')})
        write_json(record_path,record)
        print(json.dumps({'finished':seed,'condition':condition,'success':episode.get('task_success'),
                          'evaluable':episode['evaluable'],'error':episode.get('infrastructure_error')}),flush=True)
        if episode.get('infrastructure_error'):
            record['status'] = 'infrastructure_issue'
            break
    else:
        record['status'] = 'completed'
    manifest['status'] = record['status']
    write_json(record_path, record)
    write_json(root/'run_manifest.json', manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=REPO/'configs/univtac/new_seed_tactile_icl.yaml')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--demonstrations', type=Path, required=True)
    parser.add_argument('--resume-r111', action='store_true', help='Only the Pro-authorized one-time R1.11 recovery')
    parser.add_argument('--resume-r111-timed', action='store_true', help='Authorized B28 recovery and final nine cells with reset timing')
    parser.add_argument('--resume-r111-reset-limit', action='store_true', help='Authorized B28 attempt_3 and tail with native reset limit 600 seconds')
    parser.add_argument('--runtime-python', type=Path, default=Path('/home/ubuntu/anaconda3/envs/UniVTAC-isaac51-sm120-r09/bin/python3.11'))
    parser.add_argument('--source-root', type=Path, default=Path('/home/ubuntu/wybcode/.worktrees/univtac-isaac51-r081'))
    args = parser.parse_args()
    if args.resume_r111 or args.resume_r111_timed or args.resume_r111_reset_limit:
        return resume_r111(args)
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    base = yaml.safe_load(args.config.read_text())
    base['demonstration_package'] = str(args.demonstrations.resolve())
    args.mode = 'batch'
    source = subprocess.check_output(['git', '-C', str(args.source_root), 'rev-parse', 'HEAD'], text=True).strip()
    if source != '371fac67917307026be8f00869fcc1b61c623a9f':
        raise ValueError('pinned Isaac51 source mismatch')
    if base.get('motion_pacing'):
        raise ValueError('original controller required')
    conditions, order = experiment_plan(base)
    configs = {}
    for condition in conditions:
        config = condition_config(base, condition)
        path = root/f'{condition}.yaml'
        path.write_text(yaml.safe_dump(config, sort_keys=False))
        configs[condition] = (config, path)
        (root/f'{condition}_prompt.txt').write_text(operator_prompt(config))
    manifest = {'round':base['round'], 'status':'running', 'order':order, 'source_head':source,
                'repo_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                'config':base, 'prompts':{c:operator_prompt(v[0]) for c,v in configs.items()},
                'codex_cli_version':subprocess.check_output(['codex','--version'],text=True).strip(),
                'controller':'original'}
    write_json(root/'run_manifest.json', manifest)
    (root/'git_status.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=REPO,text=True))
    (root/'git_diff.patch').write_text(subprocess.check_output(['git','diff'],cwd=REPO,text=True))
    episodes = []
    for seed, condition in order:
        config, args.config = configs[condition]
        print(json.dumps({'starting':seed, 'condition':condition}), flush=True)
        episode = run_episode(args,config,seed,root/f'seed_{seed}'/condition)
        episode['condition'] = condition
        write_json(root/f'seed_{seed}'/condition/'episode.json',episode)
        episodes.append(episode)
        write_json(root/'summary.json', {'episodes':episodes, 'planned':len(order)})
        print(json.dumps(episode), flush=True)
        if episode.get('infrastructure_error'):
            manifest['status'] = 'infrastructure_issue'
            break
    else:
        manifest['status'] = 'completed'
    write_json(root/'run_manifest.json',manifest)


if __name__ == '__main__':
    main()
