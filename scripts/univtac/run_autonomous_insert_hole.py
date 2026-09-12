#!/usr/bin/env python3
"""Run unscored controls or the fixed three-episode autonomous development batch."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
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
from sim.envs.univtac.feedback_protocol import (
    PROTOCOL,
    new_run_config,
    offline_feedback,
    protocol,
    public_tools,
)
from sim.envs.univtac.scoped_isaac51_launcher import (
    ScopedIsaac51LaunchSpec,
    run_scoped_isaac51_command,
)
from sim.envs.univtac.tactile_history import export_review_video
from sim.envs.univtac.trace import write_json
from tools.embodied_gateway import LiveBackendGateway


def operator_prompt(config):
    if offline_feedback(config):
        config = new_run_config(config)
    prompt = PROMPT.replace('UniVTAC Insert Hole', 'UniVTAC ' + config.get('task_display_name', 'Insert Hole'))
    if config.get('demonstrations'):
        prompt = prompt.replace('Start by calling observe.',
            'Start by calling review_demonstrations once, then observe the current episode.').replace(
            'No operation demonstrations are provided.',
            'The demonstration tool may provide historical expert experience or no examples. '
            'Use any examples as references for reasoning about the current observations, '
            'not as absolute-coordinate scripts. Recorded measured movement is not necessarily '
            'a recorded tool command. Historical success does not guarantee current success.')
    if config.get('task_instruction'):
        prompt = prompt.replace('Insert the held object into the hole.', config['task_instruction'])
    if config.get('task_rule_ablation'):
        prompt += (
            '\nShared episode budgets and termination semantics:\n'
            f"At most {config['max_move_requests']} admitted non-preview move_to requests, "
            f"{config['max_tool_calls']} MCP calls and {config['codex_timeout_seconds']} seconds "
            'of total Codex wall time are available. Planning rejection of an admitted '
            'motion request consumes its request budget.\n'
            f"Native control is limited to {config.get('native_control_step_limit', 300)} steps, with at most {config['max_control_steps_per_move']} "
            'control steps per motion segment. Each control step advances two physics steps '
            'at 120 Hz; physics pauses while you reason. A segment limit need not end the '
            'episode: inspect the actual feedback. Native terminal or an episode budget '
            'ends physical interaction. After native terminal, last-observation and completion '
            f"tools remain available for up to {config['terminal_grace_seconds']} seconds, "
            'within the total Codex wall-time limit. Finish naturally when done.\n'
        )
        if config.get('task_information') == 'R':
            prompt += '\n' + config['public_task_rules'] + '\n'
    if config.get('current_tactile_ablation'):
        prompt = prompt.replace('bilateral tactile images, robot state',
            'bilateral tactile images when provided, robot state').replace(
            'show real segment-end tactile history',
            'show, when provided, real segment-end tactile history')
        prompt += ('\n' + config['missing_modality_notice'] + '\n') if config.get('missing_modality_notice') else ('\nSome observation modalities may be absent. Use only the\n'
                   'images and measurements actually returned. Missing tactile\n'
                   'input is not a tool failure.\n')
    if offline_feedback(config):
        prompt = prompt.replace("Use check_task to verify completion and finish_episode to end.",
            "Use finish_episode to end voluntarily or acknowledge the neutral episode-ended notification. "
            "Voluntary ending is final. Native evaluation and automatic stopping run in the background; "
            "current task scores and the specific termination reason are not provided.")
    return prompt


def demonstration_projection(config):
    package = Path(config['demonstration_package'])
    condition = config.get('demonstration_condition', config['condition'])
    projection = json.loads((package/f'{condition}.json').read_text())
    if config.get('task_instruction') and 'task_goal' in projection['text']:
        projection['text']['task_goal'] = config['task_instruction']
    return projection


def codex_command(root: Path, url: str, config: dict) -> list[str]:
    mcp_args = [f'PYTHONPATH={REPO}',str(REPO/'.venv/bin/python'),'-m','tools.embodied_mcp_server',
                '--root',str(root),'--live-worker-url',url]
    if config.get('demonstrations'):
        mcp_args.append('--demonstrations')
    if offline_feedback(config):
        mcp_args += ['--univtac-feedback-protocol', PROTOCOL]
    enabled_tools = list(public_tools(config, TOOLS)) + (['review_demonstrations'] if config.get('demonstrations') else [])
    options = ['features.memories=false','memories.use_memories=false','memories.generate_memories=false',
               'features.enable_request_compression=false','history.persistence="none"',
               'features.shell_tool=false','features.view_image=false','features.multi_agent=false','features.image_generation=false',
               f'model_reasoning_effort="{config["reasoning_effort"]}"',
               'mcp_servers.univtac.command="env"',f'mcp_servers.univtac.args={json.dumps(mcp_args)}',
               'mcp_servers.univtac.required=true',f'mcp_servers.univtac.enabled_tools={json.dumps(enabled_tools)}',
               'mcp_servers.univtac.default_tools_approval_mode="approve"','mcp_servers.univtac.tool_timeout_sec=600']
    command = [shutil.which('codex') or 'codex','-m',config['model'],'exec','-C',str(root/'operator-workspace'),
               '-s','read-only','--ephemeral','--ignore-user-config','--ignore-rules','--skip-git-repo-check',
               '--json','--output-last-message',str(root/'agent_final.md')]
    for option in options:
        command += ['-c',option]
    return command + [operator_prompt(config)]


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
    call('observe',{})
    call('move_to',{'delta_mm':[4,0,0]})
    call('move_to',{'gripper':'open'})
    call('finish_episode',{'reason':'unscored control debug complete'})
    return rows


def replay_controls(gateway, root, commands):
    """Replay already-resolved targets; no model or online relative accumulation."""
    gateway.call('observe', {})
    rows = []
    reason = 'sequence_complete'
    for command in commands:
        result = gateway.call('move_to', command['request'])
        execution = result.text.get('execution')
        rows.append({'source_seq': command['source_seq'], 'request': command['request'],
                     'ok': result.success, 'result': result.text})
        write_json(root/'replay_results.json', {'origin': 'fixed_R1.5_targets', 'rows': rows})
        if not result.success or execution is None:
            raise RuntimeError(f'Replay interface failure: {result.text}')
        if result.text.get('terminal'):
            break
        if not execution['arm_reached']:
            reason = 'segment_unfinished'
            break
    gateway.call('finish_episode', {'reason': reason})
    return rows


def run_episode(args, config, seed, root):
    root.mkdir(parents=True, exist_ok=False)
    if config.get('demonstrations'):
        package = Path(config['demonstration_package'])
        projection = demonstration_projection(config)
        for image in projection['images']:
            source = package/image['path']
            if config.get('shared_demonstration_media'):
                image['path'] = str(source.resolve())
                continue
            destination = root/'demonstrations'/image['path']
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            image['path'] = str(destination.relative_to(root))
        write_json(root/'demonstrations/projection.json', projection)
    episode = {'feedback_protocol':protocol(config), 'round':config['round'],'task':config['task'],'seed':seed,'scored':config.get('scored', args.mode=='batch'),
               'model':config['model'] if args.mode=='batch' else None,
               'reasoning_effort':config['reasoning_effort'] if args.mode=='batch' else None,
               'task_information':config.get('task_information'),
               'current_tactile':config.get('current_tactile', True),
               'demonstration_condition':config.get('demonstration_condition',config.get('condition')),'status':'starting','codex_process_count':0}
    if getattr(args, 'execution_head', None):
        episode['execution_head'] = args.execution_head
    episode.update({k:config[k] for k in ('condition','shot','expert_ids','expert_source_seeds','demonstration_set_id','query_index','cell_key') if k in config})
    write_json(root/'episode.json',episode)
    command = [str(REPO/'scripts/univtac/serve_autonomous_worker.py'),'--repo-root',str(REPO),
               '--source-root',str(args.source_root),'--output-root',str(root), '--config',str(args.config),
               '--seed',str(seed),'--headless']
    if getattr(args, 'record_reset_timing', False) or config.get('record_reset_timing', False):
        command.append('--reset-timing')
    timeout = config['startup_timeout_seconds'] + config['codex_timeout_seconds'] + config['shutdown_timeout_seconds']
    spec = ScopedIsaac51LaunchSpec(python_executable=args.runtime_python,command=tuple(command),
            cwd=args.source_root,output_root=root,timeout_seconds=timeout)
    coordinator = getattr(args, 'coordinator', None)
    coord_key = tuple(config['cell_key']) if config.get('cell_key') else seed
    hooks = coordinator.hooks(coord_key, 'worker') if coordinator else {}
    url = None
    codex_home = root/'runtime/codex-home'
    worker_start = time.monotonic()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(run_scoped_isaac51_command,spec, **hooks)
        try:
            ready = (coordinator.wait_ready(coord_key, root, future, config['startup_timeout_seconds']) if coordinator else
                     _wait_for_worker(root/'ready.json',future,config['startup_timeout_seconds']))
            episode['initialization_wall_seconds'] = time.monotonic()-worker_start
            url = ready['worker_url']
            if coordinator:
                coordinator.barrier(coord_key)
                _post(url, '/host_release')
            episode['status'] = 'running'
            write_json(root/'episode.json',episode)
            if args.mode == 'observe_only':
                result = LiveBackendGateway(root=root,worker_url=url).call('observe', {})
                write_json(root/'observe_only.json', {'scored':False, 'ok':result.success, 'text':result.text})
                if not result.success:
                    raise RuntimeError(f'observe-only check failed: {result.text}')
            elif args.mode == 'debug':
                debug_controls(LiveBackendGateway(root=root,worker_url=url),root)
            elif args.mode == 'replay':
                replay_controls(LiveBackendGateway(root=root,worker_url=url),root,args.replay_commands)
            else:
                auth = Path.home()/'.codex/auth.json'
                if not auth.is_file():
                    raise FileNotFoundError('Codex authentication unavailable')
                workspace = root/'operator-workspace'
                workspace.mkdir()
                codex_home.mkdir(parents=True)
                (codex_home/'auth.json').symlink_to(auth)
                cmd = codex_command(root,url,config)
                cli_version = subprocess.check_output([cmd[0], '--version'], text=True).strip()
                episode['codex_cli_version'] = cli_version
                write_json(root/'codex_command.json',{'command':cmd, 'model':config['model'],
                    'reasoning_effort':config['reasoning_effort'], 'cli_version':cli_version})
                (root/'prompt.txt').write_text(operator_prompt(config))
                episode['codex_process_count'] = 1
                write_json(root/'episode.json',episode)
                life = _run_to_files(cmd,cwd=workspace,environment={**os.environ,'CODEX_HOME':str(codex_home)},
                         stdout_path=root/'codex_exec.jsonl',stderr_path=root/'codex_stderr.log',
                         timeout_seconds=config['codex_timeout_seconds'],stop_path=root/'stop.json',
                         terminal_grace_seconds=config['terminal_grace_seconds'],
                         **(coordinator.hooks(coord_key, 'codex') if coordinator else {}))
                write_json(root/'codex_lifecycle.json',life)
                write_json(root/'codex_trace_summary.json',summarize_codex_exec(read_jsonl(root/'codex_exec.jsonl')))
                if life['returncode'] and not life['stopped_by_worker'] and not life['timed_out']:
                    episode['infrastructure_error'] = 'codex_process_failure'
                    if any(e.get('type') == 'turn.failed' and e.get('error', {}).get('message') == 'Selected model is at capacity. Please try a different model.' for e in read_jsonl(root/'codex_exec.jsonl')):
                        episode['infrastructure_error'] = 'provider_model_capacity'
                if life['timed_out']:
                    episode['codex_time_limit'] = True
        except Exception as exc:  # noqa: BLE001 -- retain runtime failure evidence
            episode['infrastructure_error'] = f'{type(exc).__name__}: {exc}'
            if episode['status']=='starting':
                episode['startup_failure_wall_seconds'] = time.monotonic()-worker_start
        finally:
            if coordinator:
                if episode.get('infrastructure_error'):
                    (coordinator.pause_dispatch if episode['infrastructure_error'] == 'provider_model_capacity' else coordinator.abort)(episode['infrastructure_error'], coord_key)
                coordinator.phase(coord_key, 'cleanup')
            if url and not future.done():
                try:
                    _post(url,'/host_finalize')
                except OSError:
                    pass
            try:
                lifecycle = future.result(timeout=config['shutdown_timeout_seconds']).to_dict()
                write_json(root/'worker_lifecycle.json',lifecycle)
                if lifecycle['returncode'] != 0 or lifecycle['timed_out'] or not lifecycle['cleanup_complete']:
                    episode['infrastructure_error'] = 'worker_failure'
            except TimeoutError:
                episode['infrastructure_error'] = 'worker_shutdown_timeout'
            if codex_home.exists():
                shutil.rmtree(codex_home)
    if (root/'samples.jsonl').exists() and not config.get('defer_review_video'):
        try:
            export_review_video(root)
        except Exception as exc:  # noqa: BLE001 -- raw recording is independently retained
            write_json(root/'video_error.json', {'error': str(exc), 'rebuild': 'export_review_video(episode_root)'})
            episode['video_error'] = str(exc)
    final = json.loads((root/'final_result.json').read_text()) if (root/'final_result.json').exists() else {'reset_valid':False,'native_success_available':False,'task_success':None}
    runner_error = episode.get('infrastructure_error')
    episode.update(final)
    if runner_error:
        episode['infrastructure_error'] = runner_error
    if episode.get('infrastructure_error'):
        episode['evaluable'] = False
    else:
        episode['evaluable'] = bool(final['reset_valid'] and final['native_success_available'])
    episode['status'] = 'completed' if episode['evaluable'] or (args.mode == 'observe_only' and not episode.get('infrastructure_error')) else 'infrastructure_issue'
    if coordinator and coordinator.cancel.is_set() and coordinator.abort_seed != coord_key and episode.get('infrastructure_error'):
        episode['status'] = 'cancelled'
        episode['cancelled_due_to_seed'] = coordinator.abort_seed
    write_json(root/'episode.json',episode)
    if coordinator:
        coordinator.phase(coord_key, episode['status'])
        if episode.get('infrastructure_error'):
            (coordinator.pause_dispatch if episode['infrastructure_error'] == 'provider_model_capacity' else coordinator.abort)(episode['infrastructure_error'], coord_key)
    return episode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=REPO/'configs/univtac/autonomous_insert_hole.yaml')
    parser.add_argument('--runtime-python',type=Path,default=Path('/home/ubuntu/anaconda3/envs/UniVTAC-isaac51-sm120-r09/bin/python3.11'))
    parser.add_argument('--source-root',type=Path,default=Path('/home/ubuntu/wybcode/.worktrees/univtac-isaac51-r081'))
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--mode',choices=['debug','batch','observe_only'],required=True)
    args = parser.parse_args(argv)
    args.config = args.config.resolve()
    args.output_root = args.output_root.resolve()
    if args.output_root.exists():
        raise FileExistsError('Use a fresh output root; no automatic episode retries')
    args.output_root.mkdir(parents=True)
    config = new_run_config(yaml.safe_load(args.config.read_text()))
    args.config = args.output_root/'run_config.yaml'
    args.config.write_text(yaml.safe_dump(config, sort_keys=False))
    if args.mode == 'observe_only':
        config['demonstrations'] = False
        config['observe_only'] = True
        args.config = args.output_root/'observe_only.yaml'
        args.config.write_text(yaml.safe_dump(config, sort_keys=False))
    if config['mode']!='eval' or (args.mode != 'observe_only' and (config['task']!='insert_hole' or config['seeds']!=[1000003,1000004,1000005])):
        raise ValueError('Autonomous task/seed/mode contract mismatch')
    source = subprocess.check_output(['git','-C',str(args.source_root),'rev-parse','HEAD'],text=True).strip()
    if source != '371fac67917307026be8f00869fcc1b61c623a9f':
        raise ValueError('pinned Isaac51 source mismatch')
    manifest = {'feedback_protocol':protocol(config), 'round':config['round'],'mode':args.mode,'status':'running','config':config,
                'repo_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                'source_head':source,'seeds':([999999] if args.mode=='observe_only' else config['seeds'] if args.mode=='batch' else [config['seeds'][0]])}
    write_json(args.output_root/'run_manifest.json',manifest)
    shutil.copyfile(args.config,args.output_root/'config.snapshot.yaml')
    (args.output_root/'git_status.txt').write_text(subprocess.check_output(['git','status','--short'],cwd=REPO,text=True))
    (args.output_root/'git_diff.patch').write_text(subprocess.check_output(['git','diff'],cwd=REPO,text=True))
    episodes = []
    for seed in manifest['seeds']:
        episodes.append(run_episode(args,config,seed,args.output_root/f'seed_{seed}'))
        write_json(args.output_root/'summary.json',{'feedback_protocol':protocol(config),'mode':args.mode,'episodes':episodes,
                  'evaluable_count':sum(e['evaluable'] for e in episodes),
                  'native_success_count':sum(e['evaluable'] and e['task_success'] for e in episodes)})
        if episodes[-1].get('infrastructure_error'):
            break
    manifest['status']='completed' if len(episodes)==len(manifest['seeds']) else 'infrastructure_issue'
    write_json(args.output_root/'run_manifest.json',manifest)
    return 0 if all(e['status']=='completed' for e in episodes) else 1


if __name__ == '__main__':
    raise SystemExit(main())
