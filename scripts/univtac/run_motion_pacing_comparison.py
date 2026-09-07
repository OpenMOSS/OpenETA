#!/usr/bin/env python3
"""Six fixed-target replays through the existing live worker, without Codex."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import yaml

from scripts.univtac.run_autonomous_insert_hole import REPO, run_episode
from sim.envs.univtac.codex_readonly import read_jsonl
from sim.envs.univtac.trace import write_json


def freeze_commands(source: Path) -> list[dict]:
    commands = []
    for seq, row in enumerate(read_jsonl(source/'tool_trace.jsonl'), 1):
        execution = row.get('result', {}).get('text', {}).get('execution')
        if row['tool'] != 'move_to' or row['arguments'].get('preview') or execution is None:
            continue
        pose = execution['requested_target']
        request = {k: pose[k] for k in ('xyz_m', 'approach_world', 'jaw_world')}
        if execution['requested_gripper'] is not None:
            request['gripper'] = execution['requested_gripper']
        commands.append({'source_seq': seq, 'request': request, 'resolved_target': pose,
                         'source_arguments': row['arguments']})
    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=REPO/'configs/univtac/motion_pacing_comparison.yaml')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--execute-prepared', action='store_true')
    args = parser.parse_args()
    root = args.output_root.resolve()
    plan = yaml.safe_load(args.config.read_text())
    if not args.execute_prepared:
        root.mkdir(parents=True, exist_ok=False)
        base = yaml.safe_load((REPO/plan['base_config']).read_text())
        commands = {str(seed): freeze_commands(REPO/plan['source_runs']/f'seed_{seed}')
                    for seed in (1000003, 1000004, 1000005)}
        write_json(root/'commands.json', commands)
        for variant in ('original', 'paced_candidate'):
            cfg = copy.deepcopy(base)
            cfg.update(round='R1.6', replay_diagnostics=True)
            if variant == 'paced_candidate':
                cfg['motion_pacing'] = plan['candidate']
            (root/f'{variant}.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
        write_json(root/'comparison_plan.json', plan)
    if args.prepare_only:
        return 0
    plan = json.loads((root/'comparison_plan.json').read_text())
    commands = json.loads((root/'commands.json').read_text())
    source = Path('/home/ubuntu/wybcode/.worktrees/univtac-isaac51-r081')
    source_head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if source_head != '371fac67917307026be8f00869fcc1b61c623a9f':
        raise ValueError('Pinned source mismatch')
    manifest = {'round': 'R1.6', 'kind': 'fixed_target_controller_comparison', 'plan': plan,
                'repo_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip(),
                'source_head': source_head, 'codex_process_count': 0, 'episodes': []}
    (root/'git_status.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=REPO,text=True))
    (root/'git_diff.patch').write_text(subprocess.check_output(['git','diff'],cwd=REPO,text=True))
    write_json(root/'run_manifest.json', manifest)
    for seed, variant in plan['order']:
        config_path = root/f'{variant}.yaml'
        config = yaml.safe_load(config_path.read_text())
        run_args = SimpleNamespace(mode='replay', source_root=source, config=config_path,
            runtime_python=Path('/home/ubuntu/anaconda3/envs/UniVTAC-isaac51-sm120-r09/bin/python3.11'),
            replay_commands=commands[str(seed)])
        result = run_episode(run_args, config, seed, root/f'seed_{seed}'/variant)
        result.update(variant=variant, kind='fixed_target_replay', codex_process_count=0)
        write_json(root/f'seed_{seed}'/variant/'episode.json', result)
        manifest['episodes'].append(result)
        write_json(root/'run_manifest.json', manifest)
        if result.get('infrastructure_error'):
            return 1
    manifest['status'] = 'completed'
    write_json(root/'run_manifest.json', manifest)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
