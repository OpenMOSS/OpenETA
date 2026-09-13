#!/usr/bin/env python3
"""Separate, user-authorized missing-result campaign; never edits first-pass records."""
import argparse
from collections import deque
import fcntl
import signal
import time
import selectors
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from types import SimpleNamespace

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from scripts.univtac.run_autonomous_insert_hole import run_episode
from scripts.univtac.run_eight_task_coverage import retryable_initialization
from scripts.univtac.run_fourway_capacity import Coordinator, resources
from scripts.univtac.run_shot_scaling import atomic_json, costs, render_media, verify_delivery


def supplement_config(original):
    config = dict(original)
    # Infinity disables native reset, outer ready and total worker startup deadlines.
    # Operator timeout, terminal grace and cleanup timeout remain unchanged.
    config['disable_initialization_timeout'] = True
    return config


def classify(folder):
    """Lock valid outcomes, including failures; never guess an unknown attempt valid."""
    episode_path, life_path = folder/'episode.json', folder/'worker_lifecycle.json'
    if not episode_path.exists() or not life_path.exists():
        return 'blocked', {'evaluable': False, 'infrastructure_error': 'incomplete_attempt_evidence'}
    episode = json.loads(episode_path.read_text())
    life = json.loads(life_path.read_text())
    if not life.get('cleanup_complete'):
        return 'blocked', episode
    final_path = folder/'final_result.json'
    final = json.loads(final_path.read_text()) if final_path.exists() else {}
    native_terminal = (final.get('reset_valid') and final.get('native_success_available')
                       and final.get('termination') not in (None, 'codex_exit')
                       and not final.get('infrastructure_error'))
    if episode.get('evaluable') or native_terminal:
        return 'completed', {**episode, **final, 'evaluable': True}
    if not (folder/'ready.json').exists() and not (folder/'codex_command.json').exists() and retryable_initialization(folder):
        return 'retry', episode
    if episode.get('infrastructure_error') in ('provider_model_capacity', 'provider_service_error'):
        return 'retry', episode  # Dispatch remains durably paused until explicit release.
    return 'blocked', episode


def weekly_remaining():
    """Read the local account quota without a model turn or credential output."""
    result = {'read_at_s': time.time(), 'remaining_percent': None}
    try:
        proc = subprocess.Popen([shutil.which('codex') or 'codex', 'app-server'],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, bufsize=1)
    except OSError as exc:
        return {**result, 'error':type(exc).__name__}
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)
    try:
        proc.stdin.write(json.dumps({'id':1, 'method':'initialize', 'params':{'clientInfo':{'name':'univtac-quota-read','version':'1'}, 'capabilities':{}}})+'\n');proc.stdin.flush()
        deadline = time.monotonic()+20
        while time.monotonic()<deadline:
            if not selector.select(timeout=1):
                continue
            line=proc.stdout.readline()
            if not line:break
            row=json.loads(line)
            if row.get('id')==1:
                proc.stdin.write(json.dumps({'method':'initialized'})+'\n'+json.dumps({'id':2,'method':'account/rateLimits/read'})+'\n');proc.stdin.flush()
            elif row.get('id')==2:
                if row.get('error'):
                    result['error']='rate_limits_read_failed';break
                data=row.get('result',{})
                limits=(data.get('rateLimitsByLimitId') or {}).get('codex') or data.get('rateLimits') or {}
                windows=[limits.get(k) or {} for k in ('primary','secondary')]
                weekly=next((w for w in windows if w.get('windowDurationMins')==10080),None)
                if weekly is not None:result['remaining_percent']=100-weekly['usedPercent']
                break
        return result
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return {**result, 'error':type(exc).__name__}
    finally:
        selector.close()
        proc.terminate()
        try:proc.wait(timeout=5)
        except subprocess.TimeoutExpired:proc.kill();proc.wait()


def round_pending(cells, state, round_number):
    return [c for c in cells if state[tuple(c['cell_key'])]['status'] not in ('completed','blocked')
            and (state[tuple(c['cell_key'])].get('last_round',0)<round_number
                 or state[tuple(c['cell_key'])]['status']=='not_run')]


def media_complete(folder):
    path = Path(folder)/'media_check.json'
    return path.exists() and json.loads(path.read_text()).get('passed') is True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--media-only', action='store_true')
    parser.add_argument('--host', choices=['local', 'hzz-server'], required=True)
    parser.add_argument('--runtime-python', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    campaign = args.output_root.resolve()
    manifest = json.loads((campaign/'manifest.json').read_text())
    assert manifest['phase'] in ('supplement','main_AB'), 'Explicit campaign manifest required'
    if manifest['phase']=='main_AB':
        from scripts.univtac.main_four_task import validate
        validate(manifest)
    root = campaign/'hosts'/args.host if manifest['phase']=='main_AB' else campaign
    root.mkdir(parents=True,exist_ok=True)
    cells = [c for c in manifest['cells'] if c['host'] == args.host]
    co = Coordinator(root, [tuple(c['cell_key']) for c in cells], protocol_smoke=True)
    pause_path = root/'dispatch_pause.json'
    lock = (root/('media.lock' if args.media_only else 'queue.lock')).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if not args.media_only and pause_path.exists():
        raise RuntimeError('Persistent dispatch pause requires explicit user release: '+str(pause_path))
    previous = json.loads((root/'results.json').read_text()) if (root/'results.json').exists() else {}
    old_state = {tuple(c['cell_key']):c for c in previous.get('cells',[])}
    state = {tuple(c['cell_key']):{**c, **old_state.get(tuple(c['cell_key']),{}), 'status':'not_run'} for c in cells}
    round_number = previous.get('round',1)
    done = threading.Event()
    last_report = 0.

    def persist_pause():
        with co.lock:
            if not pause_path.exists():
                atomic_json(pause_path, {'reason':co.abort_reason, 'requested_s':time.time(), 'requires_user_release':True})

    def pause(reason, seed=None):
        co.pause_dispatch(reason, seed)
        persist_pause()

    def media_job(folder, cell):
        media_lock = (root/'media.lock').open('a')
        try:
            if not args.media_only:fcntl.flock(media_lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
            render_media(folder, {**cell['original_config'], **cell,
                         'comparison_condition':cell.get('comparison_condition','C')}, 0.05)
            return 'completed'
        except BlockingIOError:
            return 'pending'
        except Exception as exc:
            atomic_json(folder/'media_error.json', {'error':str(exc)})
            return 'error'
        finally:media_lock.close()

    # Recover by reading only this campaign's attempts; the first valid outcome wins.
    for cell in cells:
        key=tuple(cell['cell_key']);base=root/'cells'/cell['task']/str(cell['seed'])/cell['condition']
        for folder in sorted(base.glob('attempt_*'),key=lambda p:int(p.name.split('_')[-1])):
            status,episode=classify(folder)
            state[key].update(status=status,episode=episode,episode_path=str(folder),
                attempts=int(folder.name.split('_')[-1]),costs=costs(folder))
            if status in ('completed','blocked'):break
        if args.media_only and state[key]['status']=='completed':
            media_job(Path(state[key]['episode_path']),cell)
    if args.media_only:return
    for key,row in state.items():
        if row['status']=='blocked':pause('Unresolved previous attempt: '+row['episode_path'],key)

    def save():
        nonlocal last_report
        if co.cancel.is_set() or co.dispatch_paused.is_set():
            persist_pause()
        with co.lock:phases={str(k):v for k,v in co.phases.items() if k in co.active_roots}
        atomic_json(root/'results.json', {'phase':manifest['phase'], 'host':args.host,
            'initialization_limit':'disabled','round':round_number,'cells':list(state.values()),
            'abort_reason':co.abort_reason,'paused':co.dispatch_paused.is_set() or co.cancel.is_set(),
            'active_phases':phases,'updated_s':time.time()})
        atomic_json(root/'media_pending.json', [r['episode_path'] for r in state.values()
            if r['status']=='completed' and not media_complete(r['episode_path'])])
        if manifest['phase']=='main_AB' and (time.monotonic()-last_report>=30 or done.is_set()):
            from scripts.univtac.main_four_task import report
            report(campaign)
            last_report=time.monotonic()

    def check_quota():
        quota=weekly_remaining();atomic_json(root/'weekly_quota.json',quota)
        if quota['remaining_percent'] is not None and quota['remaining_percent']<=2:
            pause('Local weekly quota remaining <=2%')

    def monitor():
        samples={};last_quota=time.monotonic()
        while not done.wait(2):
            try:
                row=resources(co,samples)
                with (root/'resources.jsonl').open('a') as out:out.write(json.dumps(row)+'\n')
                if row['MemAvailable']<256*1024**2 or shutil.disk_usage(root).free<5*1024**3:co.abort('Critical memory or disk exhaustion')
                with co.lock:folders=list(co.active_roots.values())
                for folder in folders:
                    log=folder/'launcher/stdout_stderr.log'
                    if log.exists():
                        with log.open('rb') as f:f.seek(max(0,log.stat().st_size-65536));tail=f.read().decode(errors='replace').lower()
                        if any(w in tail for w in ('out of gpu memory','out of memory','failed to allocate memory')):co.abort('Native OOM')
                if args.host=='local' and time.monotonic()-last_quota>=1800:
                    check_quota();last_quota=time.monotonic()
                if (co.dispatch_paused.is_set() or co.cancel.is_set()) and not pause_path.exists():
                    persist_pause()
            except Exception as exc:co.event(None,'resource_sampling_error',error=str(exc))

    def attempt(cell):
        key=tuple(cell['cell_key']);base=root/'cells'/cell['task']/str(cell['seed'])/cell['condition']
        number=max([int(p.name.split('_')[-1]) for p in base.glob('attempt_*')]+[0])+1
        folder=base/f'attempt_{number}'
        config=supplement_config(cell['original_config'])
        config_path=root/'configs'/cell['task']/str(cell['seed'])/(cell['condition']+'.yaml')
        config_path.parent.mkdir(parents=True,exist_ok=True);config_path.write_text(yaml.safe_dump(config,sort_keys=False))
        lane=SimpleNamespace(**vars(args),mode='batch',config=config_path,coordinator=co,
            execution_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
        with co.lock:co.active_roots[key]=folder;co.initialization_failures.discard(key);co.lane_cancel[key].clear()
        run_episode(lane,config,cell['seed'],folder)
        status,episode=classify(folder)
        if status=='blocked':
            co.abort('Infrastructure/state issue requires review: '+str(folder))
            persist_pause()
        return {**cell,'status':status,'episode':episode,'attempts':number,'last_round':round_number,
                'episode_path':str(folder),'costs':costs(folder)}

    signal.signal(signal.SIGINT,lambda *_:pause('User requested safe drain'))
    if args.host=='local':check_quota()
    save()
    thread=threading.Thread(target=monitor);thread.start()
    try:
        with ThreadPoolExecutor(max_workers=1) as media, ThreadPoolExecutor(max_workers=2) as pool:
            active={};media_future=None;media_key=None;media_attempted=set()
            while True:
                queue=deque(round_pending(cells,state,round_number))
                while queue or active:
                    while queue and len(active)<2 and not co.cancel.is_set() and not co.dispatch_paused.is_set():
                        cell=queue.popleft();key=tuple(cell['cell_key'])
                        state[key].update(status='in_progress',last_round=round_number)
                        save()  # Dispatch intent persists before launching this attempt.
                        active[pool.submit(attempt,cell)]=cell
                    if media_future is not None and media_future.done():
                        state[media_key]['media_status']=media_future.result();media_future=None
                    if media_future is None:
                        todo=next((r for k,r in state.items() if r['status']=='completed' and k not in media_attempted
                            and not media_complete(r['episode_path'])),None)
                        if todo:
                            media_key=tuple(todo['cell_key']);media_attempted.add(media_key)
                            media_future=media.submit(media_job,Path(todo['episode_path']),todo)
                    save()
                    if not active:break
                    finished,_=wait(active,timeout=2,return_when=FIRST_COMPLETED)
                    for future in finished:
                        cell=active.pop(future);key=tuple(cell['cell_key'])
                        try:
                            state[key]=future.result()
                            if state[key]['status']=='completed':verify_delivery(Path(state[key]['episode_path']),cell)
                        except Exception as exc:
                            # Preserve a valid result even when a separate delivery audit blocks dispatch.
                            co.abort(str(exc));state[key]['error']=str(exc)
                            if state[key].get('episode',{}).get('evaluable'):
                                state[key]['delivery_or_runner_error']=str(exc)
                                atomic_json(Path(state[key]['episode_path'])/'protocol_delivery_error.json',{'error':str(exc)})
                            else:state[key]['status']='blocked'
                        with co.lock:
                            co.active_roots.pop(key,None);co.ready.pop(key,None)
                            for kind in ('worker','codex'):co.processes.pop((key,kind),None)
                        save()
                if co.cancel.is_set() or co.dispatch_paused.is_set():break
                if all(r['status']=='completed' for r in state.values()):break
                if any(r['status']=='blocked' for r in state.values()):pause('Blocked attempt requires review');break
                round_number+=1;save()  # All slots cleaned before the next missing-result round.
            if media_future is not None:state[media_key]['media_status']=media_future.result()
            # Bounded one-at-a-time drain of remaining media after dispatch ends.
            for key,row in state.items():
                if row['status']=='completed' and key not in media_attempted:
                    row['media_status']=media_job(Path(row['episode_path']),row)
    finally:
        done.set();thread.join();save();lock.close()


if __name__ == '__main__':
    main()
