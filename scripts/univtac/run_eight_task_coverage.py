#!/usr/bin/env python3
"""Prepare fixed expert projections and run the bounded eight-cell coverage queue."""
from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import io
import json
import shutil
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.univtac.prepare_official_demonstrations import tactile_without_vision
from scripts.univtac.run_autonomous_insert_hole import codex_command, operator_prompt, run_episode
from scripts.univtac.run_fourway_capacity import Coordinator, resources
from sim.envs.univtac.feedback_protocol import PROTOCOL, new_run_config
from sim.envs.univtac.trace import write_json
from tools.embodied_mcp_server import build_live_backend_server

PROJECTIONS = {'A':'no_demo', 'B':'visual_action_icl', 'C':'tactile_action_icl', 'D':'tactile_action_no_vision_icl'}


def load(path):
    return json.loads(path.read_text())


def copy_package(source, dest):
    dest.mkdir(parents=True, exist_ok=False)
    for name in ('no_demo', 'visual_action_icl', 'tactile_action_icl', 'provenance'):
        shutil.copyfile(source/f'{name}.json', dest/f'{name}.json')
    full = load(dest/'tactile_action_icl.json')
    for im in full['images']:
        target = dest/im['path']; target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source/im['path'], target)
    write_json(dest/'tactile_action_no_vision_icl.json', tactile_without_vision(full))
    write_json(dest/'package_origin.json', {'source_package': str(source), 'resegmented': False})


def validate_package(package):
    a,b,c,d = [load(package/f'{PROJECTIONS[k]}.json') for k in 'ABCD']
    assert a['text']['examples'] == [] and a['images'] == []
    shared = copy.deepcopy(c['text']); shared.pop('historical_touch')
    assert shared == b['text'] and c['images'][:len(b['images'])] == b['images']
    assert tactile_without_vision(c) == d
    c_images = {im['label']: im for im in c['images']}
    for im in d['images']:
        assert np.array_equal(np.asarray(Image.open(package/im['path'])),
                              np.asarray(Image.open(package/c_images[im['label']]['path'])))
    return {'ABCD_matched': True, 'image_counts': {k:len(v['images']) for k,v in zip('ABCD',(a,b,c,d))},
            'segment_counts': [len(e['segments']) for e in c['text']['examples']]}



def validate_mcp_images(package):
    """Offline transport test with real package images and a mocked worker reply."""
    from unittest.mock import patch
    results={}
    for condition,name in PROJECTIONS.items():
        projection=load(package/f'{name}.json')
        payload={'ok':True,'text':{'demonstrations':projection['text'],'terminal':None,
            'image_labels':[im['label'] for im in projection['images']]},'images':projection['images']}
        server=build_live_backend_server(root=package,worker_url='http://unused',demonstrations=True,feedback_protocol=PROTOCOL)
        with patch('urllib.request.urlopen',side_effect=lambda *_args, data=payload, **_kwargs: io.BytesIO(json.dumps(data).encode())):
            blocks=asyncio.run(server.call_tool('review_demonstrations',{}))
        images=[b for b in blocks if b.type=='image']
        assert len(images)==len(projection['images'])
        for block,descriptor in zip(images,projection['images']):
            assert np.array_equal(np.asarray(Image.open(io.BytesIO(base64.b64decode(block.data)))),
                                  np.asarray(Image.open(package/descriptor['path'])))
        results[condition]={'native_mcp_images':len(images),'decoded_pixels_match':True}
    context=package/'operator_context.jsonl'
    if context.exists():context.replace(package/'offline_mcp_context.jsonl')
    write_json(package/'offline_mcp_validation.json',results)
    return results

def prepare_queue(args):
    root = args.output_root
    settings = yaml.safe_load(args.config.read_text())
    base = new_run_config(yaml.safe_load((REPO/settings['base_config']).read_text()))
    cells=[]
    for task,condition in settings['queue']:
        spec = settings['tasks'][task]
        config = copy.deepcopy(base)
        config.update(task=task, task_display_name=task.replace('_',' '), seeds=[settings['seed']],
            round='Eight-task new-protocol coverage', condition=condition,
            demonstration_condition=PROJECTIONS[condition], cell_key=[task,settings['seed'],condition],
            current_tactile=True, scored=False, wait_for_operator_release=True,
            native_control_step_limit=spec['native_control_step_limit'])
        config.pop('episode_order',None)
        if 'native_reset_time_limit_seconds' in spec:
            config['native_reset_time_limit_seconds']=spec['native_reset_time_limit_seconds']
        instruction=load(args.source_root/'instructions'/f'{task}.json')['seen'][0]
        config['task_instruction']=instruction
        if 'public_task_rules' in spec:
            config['public_task_rules']=spec['public_task_rules']+' These are static requirements, not current measurements. Only neutral episode-ended feedback is returned.'
        if not spec.get('package'):
            config['demonstration_segment_policy']='all_atoms'
        package=root/'demonstrations'/task
        config['demonstration_package']=str(package)
        config_path=root/'configs'/f'{task}_{condition}.yaml'
        config_path.parent.mkdir(parents=True,exist_ok=True)
        config_path.write_text(yaml.safe_dump(config,sort_keys=False))
        cell={'task':task,'seed':settings['seed'],'condition':condition,'config':str(config_path),
              'cell_key':config['cell_key'],'status':'prepared'}
        try:
            if not package.exists():
                if spec.get('package'):
                    copy_package(REPO/spec['package'],package)
                else:
                    for name in ('metadata.json','0.hdf5','1.hdf5'):
                        if not (root/'data'/task/name).is_file():
                            raise FileNotFoundError(root/'data'/task/name)
                    subprocess.run([str(args.runtime_python), '-m', 'scripts.univtac.prepare_official_demonstrations',
                        '--raw', str(root/'data'/task), '--output', str(package), '--config', str(config_path)],
                        cwd=REPO, check=True)
            cell['package_check']=validate_package(package)
            cell['mcp_check']=validate_mcp_images(package)
            # Actual server declarations and prompts, without contacting a worker/model.
            check_root=root/'offline_checks'/task;check_root.mkdir(parents=True,exist_ok=True)
            server=build_live_backend_server(root=check_root,worker_url='http://127.0.0.1:0',demonstrations=True,feedback_protocol=PROTOCOL)
            assert 'check_task' not in server._tool_manager._tools
            prompts=[]
            for variant in 'ABCD':
                cfg={**config,'condition':variant,'demonstration_condition':PROJECTIONS[variant]}
                prompts.append(operator_prompt(cfg))
                assert 'check_task' not in ' '.join(codex_command(check_root,'http://127.0.0.1:0',cfg))
            assert len(set(prompts))==1
            (check_root/'prompt.txt').write_text(prompts[0])
        except (OSError,ValueError,KeyError,AssertionError,subprocess.CalledProcessError) as exc:
            cell.update(status='data_blocked',error=f'{type(exc).__name__}: {exc}')
        cells.append(cell)
        print(task,cell['status'],cell.get('package_check',cell.get('error')),flush=True)
        write_json(root/'prepared_cells.json',{'feedback_protocol':PROTOCOL,'cells':cells,
            'existing_insert_hole_C':str(REPO/settings['existing_insert_hole_C']),
            'max_simulator_starts':24,'max_codex_starts':8,'concurrency':2})


def confirmed_pre_ready_exit(folder):
    """Use host lifecycle evidence, not absence of ready alone."""
    if any((folder/name).exists() for name in (
            'ready.json', 'codex_command.json', 'codex_lifecycle.json',
            'tool_trace.jsonl', 'operator_context.jsonl', 'native_reset_limit.json')):
        return False
    if not (folder/'episode.json').exists() or not (folder/'worker_lifecycle.json').exists():
        return False
    episode=load(folder/'episode.json'); life=load(folder/'worker_lifecycle.json')
    return (episode.get('codex_process_count') == 0 and episode.get('reset_valid') is False
            and life.get('root_pid') is not None and life.get('cleanup_complete') is True
            and life.get('final_process_group_members') == []
            and all(episode.get(k,0) == 0 for k in ('control_steps','physics_steps','actual_motion_requests')))


def native_startup_crash(folder):
    """Only the observed pre-reset omniClientFreeContent SIGSEGV is eligible."""
    if not confirmed_pre_ready_exit(folder):
        return False
    life=load(folder/'worker_lifecycle.json')
    if life.get('returncode') != -11 or life.get('sigterm_sent') or life.get('sigkill_sent'):
        return False
    log=folder/'launcher/stdout_stderr.log'
    text=log.read_text(errors='replace') if log.exists() else ''
    return ('libomniclient.so!omniClientFreeContent' in text and '[Fatal]' in text
            and not any(x in text.lower() for x in
                        ('out of gpu memory','out of memory','failed to allocate memory'))
            and not (folder/'worker_error.json').exists())


def authorized_cancelled_initialization(folder):
    marker=folder/'authorized_pre_ready_cancelled.json'
    if not marker.exists() or not confirmed_pre_ready_exit(folder):
        return False
    authorization=load(marker); episode=load(folder/'episode.json')
    life=load(folder/'worker_lifecycle.json')
    return (authorization.get('pro_message_id') == '8070380f-699e-45aa-b208-6012acbdf5a6'
            and authorization.get('original_error') == episode.get('infrastructure_error')
            and episode.get('status') == 'cancelled'
            and 'batch cancelled:' in episode.get('infrastructure_error','')
            and life.get('returncode') == -15 and life.get('sigterm_sent') is True)


def startup_crashes_since_ready(root):
    """Persist the streak through the existing chronological event ledger."""
    events=root/'events.jsonl'
    if not events.exists():
        return 0
    attempts={}; completed=[]; last_ready=0
    for line in events.read_text().splitlines():
        row=json.loads(line); key=tuple(row.get('cell',[]))
        if row['event']=='attempt_started':
            attempts[key]=Path(row['output_root'])
        elif row['event']=='ready_resident':
            last_ready=row['timestamp_s']
        elif row['event']=='cleanup' and key in attempts:
            completed.append((row['timestamp_s'],attempts[key]))
    return sum(native_startup_crash(folder) for timestamp,folder in completed if timestamp>last_ready)


def retryable_initialization(folder):
    """Confirmed reset failures plus narrowly identified cleaned pre-ready exits."""
    error_path=folder/'worker_error.json'
    raw=error_path.read_text() if error_path.exists() else ''
    log=folder/'launcher/stdout_stderr.log'
    text=log.read_text(errors='replace') if log.exists() else ''
    if any(x in text.lower() for x in ('out of gpu memory','out of memory','failed to allocate memory')):
        return False
    if native_startup_crash(folder) or authorized_cancelled_initialization(folder):
        return True
    if 'task.reset(seed=args.seed)' in raw or 'official reset/pre_move failed' in raw:
        return True
    episode=load(folder/'episode.json') if (folder/'episode.json').exists() else {}
    return ('ready deadline' in episode.get('infrastructure_error','') and
            (folder/'native_reset_limit.json').exists())


def reviewed_initialization_issue(folder, cell):
    """Skip only explicitly reviewed, cleaned, pre-ready issues; never retry."""
    marker=folder/'reviewed_initialization_issue.json'
    if not marker.exists():
        return None
    review=load(marker)
    if review.get('pro_message_id')!='79a54d66-91fb-4a0f-a49d-dd2826296502':
        return None
    if review.get('cell_key')!=cell['cell_key'] or review.get('attempt')!=folder.name:
        return None
    if any((folder/n).exists() for n in ('ready.json','codex_command.json',
            'codex_lifecycle.json','operator_context.jsonl','tool_trace.jsonl',
            'protocol_delivery_error.json')):
        return None
    episode=load(folder/'episode.json'); life=load(folder/'worker_lifecycle.json')
    if (episode.get('codex_process_count')!=0 or episode.get('reset_valid') is not False
            or not life.get('cleanup_complete') or life.get('final_process_group_members')!=[]
            or review.get('original_error')!=episode.get('infrastructure_error')
            or review.get('returncode')!=life.get('returncode')):
        return None
    return {**cell,'status':'infrastructure_issue','episode':episode,'episode_path':str(folder),
            'attempts':int(folder.name.split('_')[-1]),'reviewed_skip':True,
            'infrastructure_subtype':review['subtype'],'root_cause':'unknown'}


def run_cell(args,cell,coordinator):
    key=tuple(cell['cell_key']);root=args.output_root/'cells'/cell['task']/str(cell['seed'])/cell['condition']
    root.mkdir(parents=True,exist_ok=True)
    # Existing attempts are the durable budget; neither a restart nor failed result resets it.
    for number in range(1,4):
        folder=root/f'attempt_{number}'
        if not folder.exists():break
        reviewed=reviewed_initialization_issue(folder,cell)
        if reviewed:
            return reviewed
        if (folder/'ready.json').exists() or (folder/'codex_command.json').exists():
            episode=load(folder/'episode.json') if (folder/'episode.json').exists() else {}
            life=load(folder/'worker_lifecycle.json') if (folder/'worker_lifecycle.json').exists() else {}
            if episode.get('status')=='completed' and life.get('cleanup_complete'):
                return {**cell,'status':'completed','episode_path':str(folder),'episode':episode,'reused_on_resume':True}
            coordinator.abort('Previously accepted or unresolved episode must not be replaced')
            return {**cell,'status':'unresolved_previous_attempt','episode_path':str(folder)}
        life=load(folder/'worker_lifecycle.json') if (folder/'worker_lifecycle.json').exists() else {}
        if not life.get('cleanup_complete') or not retryable_initialization(folder):
            coordinator.abort('Prior attempt is not a cleaned native initialization failure')
            return {**cell,'status':'unresolved_previous_attempt','episode_path':str(folder)}
    else:
        return {**cell,'status':'initialization_unavailable','attempts':3}
    attempt_limit=getattr(args,'max_initialization_attempts',3)
    if number>attempt_limit:
        return {**cell,'status':'initialization_unavailable','attempts':number-1,
                'episode_path':str(root/f'attempt_{number-1}')}
    config=yaml.safe_load(Path(cell['config']).read_text())
    lane_args=SimpleNamespace(**vars(args));lane_args.config=Path(cell['config']);lane_args.mode='batch';lane_args.coordinator=coordinator
    for attempt in range(number,attempt_limit+1):
        if coordinator.cancel.is_set():return {**cell,'status':'cancelled'}
        folder=root/f'attempt_{attempt}'
        with coordinator.lock:
            coordinator.active_roots[key]=folder;coordinator.initialization_failures.discard(key)
            coordinator.lane_cancel[key].clear();coordinator.phase(key,'submitted')
            coordinator.event(key,'attempt_started',attempt=attempt,output_root=str(folder))
        episode=run_episode(lane_args,config,cell['seed'],folder)
        if (folder/'ready.json').exists() or episode.get('codex_process_count'):
            if episode.get('infrastructure_error'):
                coordinator.abort('Accepted episode infrastructure failure: '+str(folder))
            return {**cell,'status':episode['status'],'episode_path':str(folder),'episode':episode,'attempts':attempt}
        life=load(folder/'worker_lifecycle.json') if (folder/'worker_lifecycle.json').exists() else {}
        if not life.get('cleanup_complete') or not retryable_initialization(folder):
            coordinator.abort('Non-retryable initialization/interface failure: '+str(folder))
            return {**cell,'status':'infrastructure_issue','episode_path':str(folder),'episode':episode,'attempts':attempt}
        if native_startup_crash(folder) and startup_crashes_since_ready(args.output_root)>=3:
            coordinator.abort('Three omniClient startup crashes without an intervening ready')
            return {**cell,'status':'infrastructure_issue','episode_path':str(folder),'episode':episode,'attempts':attempt}
    return {**cell,'status':'initialization_unavailable','episode_path':str(folder),'episode':episode,'attempts':attempt_limit}


def run_queue(args):
    root=args.output_root;prepared=load(root/'prepared_cells.json');cells=prepared['cells']
    coordinator=Coordinator(root,[tuple(c['cell_key']) for c in cells],protocol_smoke=True)
    results=[c for c in cells if c['status']=='data_blocked']
    previous={};done=threading.Event()
    initial_resources=resources(coordinator,previous)
    if not (root/'resources_before.json').exists():
        write_json(root/'resources_before.json',initial_resources)
    def sample():
        while not done.is_set():
            start=time.monotonic()
            try:
                row=resources(coordinator,previous)
                with (root/'resources.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                if row['MemAvailable']<256*1024**2:coordinator.abort('Critical live system memory exhaustion')
                with coordinator.lock:
                    for key,folder in coordinator.active_roots.items():
                        error=folder/'worker_error.json'
                        if error.exists() and not coordinator.cancel.is_set():coordinator.abort(error.read_text(),key)
                        final=folder/'final_result.json'
                        if final.exists() and load(final).get('infrastructure_error'):
                            coordinator.abort('Live interface failure: '+str(folder))
                        log=folder/'launcher/stdout_stderr.log'
                        if log.exists() and any(word in log.read_text(errors='replace')[-65536:].lower() for word in
                                ('out of gpu memory','out of memory','failed to allocate memory')):
                            coordinator.abort('Native OOM log: '+str(folder))
            except Exception as exc:  # noqa: BLE001 -- preserve monitoring failure
                coordinator.event(None,'resource_sampling_error',error=str(exc))
            done.wait(max(0,1-(time.monotonic()-start)))
    monitor=threading.Thread(target=sample);monitor.start()
    session_start={'started_s':time.time(),'repo_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()}
    manifest=load(root/'run_manifest.json') if (root/'run_manifest.json').exists() else {**prepared,**session_start}
    manifest['status']='running'
    manifest.setdefault('run_sessions',[]).append(session_start)
    write_json(root/'run_manifest.json',manifest)
    queue=iter(c for c in cells if c['status']=='prepared')
    def save():
        write_json(root/'coverage_results.json',{'feedback_protocol':PROTOCOL,'cells':results,
            'existing_insert_hole_C':prepared['existing_insert_hole_C'],'abort_reason':coordinator.abort_reason})
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            active={}
            while True:
                while len(active)<2 and not coordinator.cancel.is_set():
                    cell=next(queue,None)
                    if cell is None:break
                    active[pool.submit(run_cell,args,cell,coordinator)]=cell
                if not active:break
                finished,_=wait(active,return_when=FIRST_COMPLETED)
                for future in finished:
                    cell=active.pop(future)
                    try:results.append(future.result())
                    except Exception as exc:  # noqa: BLE001 -- stop new dispatch on real interface errors
                        coordinator.abort(str(exc));results.append({**cell,'status':'infrastructure_issue','error':str(exc)})
                    save()
    finally:
        done.set();monitor.join();write_json(root/'resources_after.json',resources(coordinator,previous))
        seen={tuple(c['cell_key']) for c in results}
        results.extend({**c,'status':'not_run'} for c in cells if tuple(c['cell_key']) not in seen)
        save();manifest.update(status='paused' if coordinator.cancel.is_set() else 'completed',ended_s=time.time())
        write_json(root/'run_manifest.json',manifest)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=('prepare','run'),required=True)
    parser.add_argument('--config',type=Path,default=REPO/'configs/univtac/eight_task_coverage.yaml')
    parser.add_argument('--output-root',type=Path,default=REPO/'outputs/univtac-eight-task-coverage')
    parser.add_argument('--source-root',type=Path,default=Path('/home/ubuntu/wybcode/.worktrees/univtac-isaac51-r081'))
    parser.add_argument('--runtime-python',type=Path,default=Path('/home/ubuntu/anaconda3/envs/UniVTAC-isaac51-sm120-r09/bin/python3.11'))
    args=parser.parse_args();args.output_root=args.output_root.resolve()
    args.output_root.mkdir(parents=True,exist_ok=True)
    if args.phase=='prepare':prepare_queue(args)
    else:run_queue(args)


if __name__=='__main__':main()
