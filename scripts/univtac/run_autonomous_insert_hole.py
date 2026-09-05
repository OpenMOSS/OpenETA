#!/usr/bin/env python3
"""Run unscored control debugging or the fixed three-episode R1.4 batch."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.univtac.run_codex_readonly_observation import _run_to_files
from scripts.univtac.run_pull_out_key_live_codex import _post, _wait_for_worker
from sim.envs.univtac.autonomous_operation import PROMPT, TOOLS, quaternion_matrix, rotvec_matrix
from sim.envs.univtac.codex_readonly import read_jsonl, summarize_codex_exec
from sim.envs.univtac.scoped_isaac51_launcher import (
    ScopedIsaac51LaunchSpec,
    run_scoped_isaac51_command,
)
from sim.envs.univtac.trace import write_json
from tools.embodied_gateway import LiveBackendGateway


def codex_command(root: Path, url: str, config: dict) -> list[str]:
    mcp_args = [f'PYTHONPATH={REPO}',str(REPO/'.venv/bin/python'),'-m','tools.embodied_mcp_server',
                '--root',str(root),'--live-worker-url',url]
    options = ['features.memories=false','memories.use_memories=false','memories.generate_memories=false',
               'features.enable_request_compression=false','history.persistence="none"',
               'features.shell_tool=false','features.view_image=false','features.multi_agent=false','features.image_generation=false',
               f'model_reasoning_effort="{config["reasoning_effort"]}"',
               'mcp_servers.univtac.command="env"',f'mcp_servers.univtac.args={json.dumps(mcp_args)}',
               'mcp_servers.univtac.required=true',f'mcp_servers.univtac.enabled_tools={json.dumps(list(TOOLS))}',
               'mcp_servers.univtac.default_tools_approval_mode="approve"','mcp_servers.univtac.tool_timeout_sec=600']
    command = [shutil.which('codex') or 'codex','-m',config['model'],'exec','-C',str(root/'operator-workspace'),
               '-s','read-only','--ephemeral','--ignore-user-config','--ignore-rules','--skip-git-repo-check',
               '--json','--output-last-message',str(root/'agent_final.md')]
    for option in options:
        command += ['-c',option]
    return command + [PROMPT]


def debug_controls(gateway, root):
    """Unscored robot-relative commands, never task-conditioned instructions."""
    rows = []
    def call(tool, args):
        result = gateway.call(tool,args)
        rows.append({'tool':tool,'arguments':args,'ok':result.success,'result':result.text})
        write_json(root/'debug_controls.json',{'scored':False,'rows':rows})
        if not result.success:
            raise RuntimeError(f'debug {tool}: {result.text}')
        return result.text
    first = call('observe',{})['observation']
    call('mark_point',{'view':'head','u':220,'v':170})
    state = first['robot']
    initial_rotation = quaternion_matrix(state['quat_xyzw'])
    call('move_to',{'delta_mm':[5,0,0],'preview':True})
    call('move_to',{'delta_mm':[5,0,0]})
    call('move_to',{'xyz_m':state['xyz_m']})
    rotation = rotvec_matrix([0,0,0.08]) @ initial_rotation
    call('move_to',{'approach_world':rotation[:,2].tolist(),'jaw_world':rotation[:,0].tolist()})
    call('move_to',{'approach_world':initial_rotation[:,2].tolist(),'jaw_world':initial_rotation[:,0].tolist()})
    # Gripper tests happen only after motion checks in this disposable episode.
    call('move_to',{'gripper':'close'})
    call('move_to',{'gripper':'open'})
    call('finish_episode',{'reason':'unscored control debug complete'})
    return rows


def run_episode(args, config, seed, root):
    root.mkdir(parents=True, exist_ok=False)
    episode = {'round':'R1.4','task':'insert_hole','seed':seed,'scored':args.mode=='batch',
               'model':config['model'] if args.mode=='batch' else None,'status':'starting','codex_process_count':0}
    write_json(root/'episode.json',episode)
    command = [str(REPO/'scripts/univtac/serve_autonomous_worker.py'),'--repo-root',str(REPO),
               '--source-root',str(args.source_root),'--output-root',str(root), '--config',str(args.config),
               '--seed',str(seed),'--headless']
    timeout = config['startup_timeout_seconds'] + config['codex_timeout_seconds'] + config['shutdown_timeout_seconds']
    spec = ScopedIsaac51LaunchSpec(python_executable=args.runtime_python,command=tuple(command),
            cwd=args.source_root,output_root=root,timeout_seconds=timeout)
    url = None
    codex_home = root/'runtime/codex-home'
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run_scoped_isaac51_command,spec)
        try:
            ready = _wait_for_worker(root/'ready.json',future,config['startup_timeout_seconds'])
            url = ready['worker_url']
            episode['status'] = 'running'
            write_json(root/'episode.json',episode)
            if args.mode == 'debug':
                debug_controls(LiveBackendGateway(root=root,worker_url=url),root)
            else:
                auth = Path.home()/'.codex/auth.json'
                if not auth.is_file():
                    raise FileNotFoundError('Codex authentication unavailable')
                workspace = root/'operator-workspace'
                workspace.mkdir()
                codex_home.mkdir(parents=True)
                (codex_home/'auth.json').symlink_to(auth)
                cmd = codex_command(root,url,config)
                write_json(root/'codex_command.json',{'command':cmd})
                (root/'prompt.txt').write_text(PROMPT)
                episode['codex_process_count'] = 1
                write_json(root/'episode.json',episode)
                life = _run_to_files(cmd,cwd=workspace,environment={**os.environ,'CODEX_HOME':str(codex_home)},
                         stdout_path=root/'codex_exec.jsonl',stderr_path=root/'codex_stderr.log',
                         timeout_seconds=config['codex_timeout_seconds'],stop_path=root/'stop.json')
                write_json(root/'codex_lifecycle.json',life)
                write_json(root/'codex_trace_summary.json',summarize_codex_exec(read_jsonl(root/'codex_exec.jsonl')))
                if life['returncode'] and not life['stopped_by_worker'] and not life['timed_out']:
                    episode['infrastructure_error'] = 'codex_process_failure'
                if life['timed_out']:
                    episode['codex_time_limit'] = True
        except Exception as exc:  # noqa: BLE001 -- retain runtime failure evidence
            episode['infrastructure_error'] = f'{type(exc).__name__}: {exc}'
        finally:
            if url and not future.done():
                try:
                    _post(url,'/host_finalize')
                except OSError:
                    pass
            try:
                lifecycle = future.result(timeout=config['shutdown_timeout_seconds']).to_dict()
                write_json(root/'worker_lifecycle.json',lifecycle)
                if lifecycle['returncode'] != 0 or lifecycle['timed_out'] or not lifecycle['cleanup_complete']:
                    episode['infrastructure_error'] = episode.get('infrastructure_error') or 'worker_failure'
            except TimeoutError:
                episode['infrastructure_error'] = 'worker_shutdown_timeout'
            if codex_home.exists():
                shutil.rmtree(codex_home)
    final = json.loads((root/'final_result.json').read_text()) if (root/'final_result.json').exists() else {'reset_valid':False,'native_success_available':False,'task_success':None}
    runner_error = episode.get('infrastructure_error')
    episode.update(final)
    if runner_error:
        episode['infrastructure_error'] = runner_error
    if episode.get('infrastructure_error'):
        episode['evaluable'] = False
    else:
        episode['evaluable'] = bool(final['reset_valid'] and final['native_success_available'])
    episode['status'] = 'completed' if episode['evaluable'] else 'infrastructure_issue'
    write_json(root/'episode.json',episode)
    return episode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=REPO/'configs/univtac/autonomous_insert_hole.yaml')
    parser.add_argument('--runtime-python',type=Path,default=Path('/home/ubuntu/anaconda3/envs/UniVTAC-isaac51-sm120-r09/bin/python3.11'))
    parser.add_argument('--source-root',type=Path,default=Path('/home/ubuntu/wybcode/.worktrees/univtac-isaac51-r081'))
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--mode',choices=['debug','batch'],required=True)
    args = parser.parse_args(argv)
    args.config = args.config.resolve()
    args.output_root = args.output_root.resolve()
    if args.output_root.exists():
        raise FileExistsError('Use a fresh output root; no automatic episode retries')
    args.output_root.mkdir(parents=True)
    config = yaml.safe_load(args.config.read_text())
    if config['task']!='insert_hole' or config['seeds']!=[1000003,1000004,1000005] or config['mode']!='eval':
        raise ValueError('R1.4 task/seed/mode contract mismatch')
    source = subprocess.check_output(['git','-C',str(args.source_root),'rev-parse','HEAD'],text=True).strip()
    if source != '371fac67917307026be8f00869fcc1b61c623a9f':
        raise ValueError('pinned Isaac51 source mismatch')
    manifest = {'round':'R1.4','mode':args.mode,'status':'running','config':config,
                'repo_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                'source_head':source,'seeds':config['seeds'] if args.mode=='batch' else [config['seeds'][0]]}
    write_json(args.output_root/'run_manifest.json',manifest)
    shutil.copyfile(args.config,args.output_root/'config.snapshot.yaml')
    (args.output_root/'git_status.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=REPO,text=True))
    (args.output_root/'git_diff.patch').write_text(subprocess.check_output(['git','diff'],cwd=REPO,text=True))
    episodes = []
    for seed in manifest['seeds']:
        episodes.append(run_episode(args,config,seed,args.output_root/f'seed_{seed}'))
        write_json(args.output_root/'summary.json',{'mode':args.mode,'episodes':episodes,
                  'evaluable_count':sum(e['evaluable'] for e in episodes),
                  'native_success_count':sum(e['evaluable'] and e['task_success'] for e in episodes)})
        if episodes[-1].get('infrastructure_error'):
            break
    manifest['status']='completed' if len(episodes)==len(manifest['seeds']) else 'infrastructure_issue'
    write_json(args.output_root/'run_manifest.json',manifest)
    return 0 if all(e['evaluable'] for e in episodes) else 1


if __name__ == '__main__':
    raise SystemExit(main())
