#!/usr/bin/env python3
"""Separate, user-authorized missing-result campaign; never edits first-pass records."""
import argparse
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
    episode = json.loads((folder/'episode.json').read_text())
    life = json.loads((folder/'worker_lifecycle.json').read_text())
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
    return 'blocked', episode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--media-only', action='store_true')
    parser.add_argument('--host', choices=['local', 'hzz-server'], required=True)
    parser.add_argument('--runtime-python', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    manifest = json.loads((root/'manifest.json').read_text())
    assert manifest['phase'] == 'supplement', 'Separate supplement manifest required'
    cells = [c for c in manifest['cells'] if c['host'] == args.host]
    if args.media_only:
        for cell in cells:
            base = root/'cells'/cell['task']/str(cell['seed'])/cell['condition']
            for folder in base.glob('attempt_*'):
                if (folder/'worker_lifecycle.json').exists() and classify(folder)[0] == 'completed':
                    render_media(folder, {**cell['original_config'], **cell, 'comparison_condition': 'C'}, 0.05)
        return
    co = Coordinator(root, [tuple(c['cell_key']) for c in cells], protocol_smoke=True)
    state = {tuple(c['cell_key']): {**c, 'status': 'not_run'} for c in cells}
    done = threading.Event()

    def save():
        atomic_json(root/'results.json', {'phase': 'supplement', 'host': args.host,
                    'initialization_limit': 'disabled', 'cells': list(state.values()),
                    'abort_reason': co.abort_reason})

    def monitor():
        previous = {}
        while not done.wait(2):
            try:
                row = resources(co, previous)
                with (root/'resources.jsonl').open('a') as out:
                    out.write(json.dumps(row)+'\n')
                if row['MemAvailable'] < 256*1024**2 or shutil.disk_usage(root).free < 5*1024**3:
                    co.abort('Critical memory or disk exhaustion')
                with co.lock:
                    folders = list(co.active_roots.values())
                for folder in folders:
                    log = folder/'launcher/stdout_stderr.log'
                    if log.exists():
                        with log.open('rb') as f:
                            f.seek(max(0, log.stat().st_size-65536))
                            tail = f.read().decode(errors='replace').lower()
                        if any(w in tail for w in ('out of gpu memory', 'out of memory', 'failed to allocate memory')):
                            co.abort('Native OOM')
            except Exception as exc:
                co.event(None, 'resource_sampling_error', error=str(exc))

    def attempt(cell):
        key = tuple(cell['cell_key'])
        base = root/'cells'/cell['task']/str(cell['seed'])/cell['condition']
        old = sorted(base.glob('attempt_*'), key=lambda p: int(p.name.split('_')[-1]))
        for folder in old:
            status, episode = classify(folder)
            if status == 'completed':
                return {**cell, 'status': status, 'episode': episode, 'episode_path': str(folder)}
            if status == 'blocked':
                raise RuntimeError('Unresolved previous supplement attempt: '+str(folder))
        number = len(old)+1
        folder = base/f'attempt_{number}'
        config = supplement_config(cell['original_config'])
        config_path = root/'configs'/cell['task']/str(cell['seed'])/(cell['condition']+'.yaml')
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(yaml.safe_dump(config, sort_keys=False))
        lane = SimpleNamespace(**vars(args), mode='batch', config=config_path,
                               coordinator=co, execution_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip())
        with co.lock:
            co.active_roots[key] = folder
            co.initialization_failures.discard(key)
            co.lane_cancel[key].clear()
        run_episode(lane, config, cell['seed'], folder)
        status, episode = classify(folder)
        if status == 'blocked':
            co.pause_dispatch('Supplement infrastructure failure requires review', key)
        return {**cell, 'status': status, 'episode': episode, 'attempts': number,
                'episode_path': str(folder), 'costs': costs(folder)}

    def media_job(folder, cell):
        try:
            render_media(folder, {**cell['original_config'], **cell, 'comparison_condition': 'C'}, 0.05)
        except Exception as exc:
            atomic_json(folder/'media_error.json', {'error': str(exc)})

    thread = threading.Thread(target=monitor)
    thread.start()
    try:
        with ThreadPoolExecutor(max_workers=1) as media, ThreadPoolExecutor(max_workers=2) as pool:
            queue = list(cells)
            active = {}
            while queue or active:
                while queue and len(active)<2 and not co.cancel.is_set() and not co.dispatch_paused.is_set():
                    cell = queue.pop(0)
                    state[tuple(cell['cell_key'])]['status'] = 'in_progress'
                    active[pool.submit(attempt, cell)] = cell
                save()
                if not active:
                    break
                finished, _ = wait(active, return_when=FIRST_COMPLETED)
                for future in finished:
                    cell = active.pop(future)
                    key = tuple(cell['cell_key'])
                    try:
                        result = future.result()
                        state[key] = result
                        if result['status'] == 'completed':
                            verify_delivery(Path(result['episode_path']), cell)
                            media.submit(media_job, Path(result['episode_path']), cell)
                        elif result['status'] == 'retry':
                            queue.append(cell)
                    except Exception as exc:
                        co.abort(str(exc))
                        state[key]['status'] = 'blocked'
                        state[key]['error'] = str(exc)
                    with co.lock:
                        co.active_roots.pop(key, None)
                        co.ready.pop(key, None)
                        for kind in ('worker', 'codex'):
                            co.processes.pop((key, kind), None)
                    save()
    finally:
        done.set()
        thread.join()
        save()


if __name__ == '__main__':
    main()
