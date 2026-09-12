#!/usr/bin/env python3
"""One unscored two- or four-worker batch, using the existing autonomous runner."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.univtac.run_autonomous_insert_hole import codex_command, operator_prompt, run_episode
from sim.envs.univtac.feedback_protocol import PROTOCOL, new_run_config, protocol
from sim.envs.univtac.trace import write_json

SEEDS = (1000040, 1000041, 1000042, 1000043)


class Coordinator:
    def __init__(self, root, seeds=SEEDS, protocol_smoke=False):
        self.root = root
        self.seeds = tuple(seeds)
        self.protocol_smoke = protocol_smoke
        self.lane_cancel = {s: threading.Event() for s in seeds}
        self.active_roots = {}
        self.initialization_failures = set()
        self.cancel = threading.Event()
        self.dispatch_paused = threading.Event()
        self.lock = threading.RLock()
        self.processes = {}
        self.phases = dict.fromkeys(self.seeds, 'submitted')
        self.ready = {}
        self.release = threading.Event()
        self.abort_reason = None
        self.abort_seed = None
        self.events = []

    def event(self, seed, name, **extra):
        with self.lock:
            row = {'timestamp_s': time.time(), 'seed': seed, 'event': name, **extra}
            if isinstance(seed, tuple):
                row.update(cell=list(seed), task=seed[0], seed=seed[1], condition=seed[2])
            self.events.append(row)
            with (self.root/'events.jsonl').open('a') as f:
                f.write(json.dumps(row)+'\n')
            return row

    def phase(self, seed, phase):
        with self.lock:
            self.phases[seed] = phase
            self.event(seed, phase)

    def hooks(self, seed, kind):
        def started(process):
            with self.lock:
                self.processes[seed, kind] = process
                self.phases[seed] = 'initializing' if kind == 'worker' else 'operating'
                self.event(seed, kind+'_started', pid=process.pid)
        return {'cancel_event': self.lane_cancel[seed] if self.protocol_smoke else self.cancel, 'on_started': started}

    def pause_dispatch(self, reason, seed=None):
        """Drain existing lanes without cancelling their original budgets."""
        with self.lock:
            if not self.dispatch_paused.is_set():
                self.abort_reason = str(reason)
                self.event(seed, "dispatch_paused", reason=str(reason))
                self.dispatch_paused.set()

    def abort(self, reason, seed=None):
        with self.lock:
            if self.protocol_smoke and seed is not None and seed not in self.ready and (seed, 'worker') in self.processes:
                if seed not in self.initialization_failures:
                    self.event(seed, 'initialization_attempt_failed', reason=str(reason))
                self.initialization_failures.add(seed)
                self.lane_cancel[seed].set()
                return
            if not self.cancel.is_set():
                self.abort_reason = str(reason)
                self.abort_seed = seed
                self.event(seed, 'batch_abort', reason=str(reason))
                self.cancel.set()
                for event in self.lane_cancel.values():
                    event.set()

    def wait_ready(self, seed, root, future, timeout):
        deadline = time.monotonic()+timeout
        while time.monotonic() < deadline:
            if self.cancel.is_set():
                raise RuntimeError('batch cancelled: '+str(self.abort_reason))
            if future.done():
                raise RuntimeError(f'{seed}: worker exited before ready: {future.result()}')
            if (root/'worker_error.json').exists():
                raise RuntimeError(f'{seed}: '+(root/'worker_error.json').read_text())
            if (root/'ready.json').exists():
                ready = json.loads((root/'ready.json').read_text())
                with self.lock:
                    process = self.processes.get((seed, 'worker'))
                    if process is None or process.poll() is not None:
                        raise RuntimeError(f'{seed}: ready worker not alive')
                    self.ready[seed] = ready
                    self.phase(seed, 'ready_resident')
                return ready
            time.sleep(0.2)
        raise TimeoutError(f'{seed}: ready deadline {timeout}s')

    def barrier(self, seed):
        if self.protocol_smoke:
            with self.lock:
                if self.cancel.is_set():
                    raise RuntimeError('batch cancelled before operator release')
                if len(self.ready) == len(self.seeds) and not self.release.is_set() and all(
                        self.processes[s, 'worker'].poll() is None for s in self.seeds):
                    self.event(None, 'all_ready_resident')
                    self.release.set()
                self.event(seed, 'operator_released')
            return
        while not self.release.wait(0.1):
            with self.lock:
                if self.cancel.is_set():
                    raise RuntimeError('batch cancelled before release')
                if len(self.ready) == len(self.seeds) and not self.release.is_set():
                    if any(self.processes[s, 'worker'].poll() is not None for s in self.seeds):
                        self.abort('worker exited at all-ready barrier')
                        raise RuntimeError(self.abort_reason)
                    self.event(None, 'all_ready_resident')
                    self.release.set()
        if self.cancel.is_set():
            raise RuntimeError('batch cancelled at release')
        self.event(seed, 'operator_released')


def resources(coordinator, previous):
    now = time.time()
    mem = {line.split(':')[0]: int(line.split()[1])*1024
           for line in Path('/proc/meminfo').read_text().splitlines() if len(line.split()) >= 2}
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu',
                          '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5, check=False)
    processes = {}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            text = path.read_text(); fields = text[text.rfind(')')+2:].split()
            processes[int(path.parent.name)] = {'ppid': int(fields[1]), 'pgid': int(fields[2]),
                'ticks': int(fields[11])+int(fields[12]), 'rss': int(fields[21])*os.sysconf('SC_PAGE_SIZE')}
        except (OSError, ValueError, IndexError):
            continue
    with coordinator.lock:
        owned = list(coordinator.processes.items())
        phases = coordinator.phases.copy()
    trees = []
    for (seed, kind), process in owned:
        members = {pid for pid, row in processes.items() if row['pgid'] == process.pid}
        while True:
            extended = members | {pid for pid, row in processes.items() if row['ppid'] in members}
            if extended == members:
                break
            members = extended
        ticks = sum(processes[p]['ticks'] for p in members)
        key = (seed, kind)
        old = previous.get(key)
        cpu = max(0, ticks-old[1])/os.sysconf('SC_CLK_TCK')/(now-old[0])*100 if old and now>old[0] else None
        previous[key] = (now, ticks)
        trees.append({'seed': seed, 'kind': kind, 'root_pid': process.pid, 'pids': sorted(members),
                      'rss_bytes_sum': sum(processes[p]['rss'] for p in members), 'cpu_percent': cpu})
        if kind == 'worker' and process.poll() is not None and phases[seed] in ('initializing', 'ready_resident', 'operating'):
            coordinator.abort(f'{seed}: worker exited during {phases[seed]}, code={process.returncode}', seed)
    return {'timestamp_s': now, 'gpu_csv': gpu.stdout.strip(), 'gpu_error': gpu.stderr.strip(),
            'MemAvailable': mem['MemAvailable'], 'MemTotal': mem['MemTotal'],
            'swap_used_bytes': mem['SwapTotal']-mem['SwapFree'], 'trees': trees,
            'phases': {str(k): v for k,v in phases.items()}}



def run_lane(args, config, seed, root):
    """Only pre-takeover failures may retry; a ready episode is never replaced."""
    coordinator = args.coordinator
    attempts = 3 if coordinator.protocol_smoke else 1
    for number in range(1, attempts + 1):
        if coordinator.cancel.is_set():
            return {'seed': seed, 'status': 'cancelled', 'evaluable': False}
        folder = root/f'attempt_{number}' if coordinator.protocol_smoke else root
        with coordinator.lock:
            coordinator.active_roots[seed] = folder
            coordinator.initialization_failures.discard(seed)
            coordinator.lane_cancel[seed].clear()
            coordinator.phase(seed, 'submitted')
            coordinator.event(seed, 'attempt_started', attempt=number, output_root=str(folder))
        episode = run_episode(args, config, seed, folder)
        if not coordinator.protocol_smoke or seed in coordinator.ready or episode.get('codex_process_count'):
            return episode
        lifecycle_path = folder/'worker_lifecycle.json'
        if not lifecycle_path.exists() or not json.loads(lifecycle_path.read_text()).get('cleanup_complete'):
            coordinator.abort('Initialization attempt did not finish owned-process cleanup')
            return episode
        coordinator.event(seed, 'initialization_attempt_closed', attempt=number)
    return episode


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol-smoke', action='store_true', help='Two independent new-protocol lanes, at most three pre-ready attempts each')
    parser.add_argument('--concurrency', type=int, choices=(2, 4), default=4)
    parser.add_argument('--config', type=Path, default=REPO/'outputs/univtac-isaac51-r110/batch/C_live.yaml')
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--runtime-python', type=Path, default=Path('/home/ubuntu/anaconda3/envs/UniVTAC-isaac51-sm120-r09/bin/python3.11'))
    parser.add_argument('--source-root', type=Path, default=Path('/home/ubuntu/wybcode/.worktrees/univtac-isaac51-r081'))
    args = parser.parse_args()
    if args.protocol_smoke and args.concurrency != 2:
        parser.error('--protocol-smoke requires --concurrency 2')
    seeds = SEEDS[:args.concurrency]
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    config = new_run_config(yaml.safe_load(args.config.read_text()))
    original_prompt = operator_prompt(config)
    config.update(round=('Two-way capacity' if args.concurrency == 2 else 'Four-way capacity'), seeds=list(seeds), scored=False, wait_for_operator_release=True)
    if args.protocol_smoke:
        config['round'] = 'No-online-feedback live smoke'
    config.pop('episode_order', None)
    if (config['task'], config['model'], config['reasoning_effort'], config['condition']) != ('insert_hole', 'gpt-6-astra', 'low', 'C_live'):
        raise ValueError('Expected successful Insert Hole C_live configuration')
    assert operator_prompt(config) == original_prompt
    args.config = root/'config.yaml'
    args.config.write_text(yaml.safe_dump(config, sort_keys=False))
    (root/'prompt.txt').write_text(original_prompt)
    if args.protocol_smoke:
        from tools.embodied_mcp_server import build_live_backend_server
        server = build_live_backend_server(root=root, worker_url='http://127.0.0.1:0',
            demonstrations=True, feedback_protocol=protocol(config))
        command = codex_command(root, 'http://127.0.0.1:0', config)
        if protocol(config) != PROTOCOL or 'check_task' in server._tool_manager._tools or 'check_task' in ' '.join(command):
            raise ValueError('New protocol preflight mismatch')
        write_json(root/'protocol_preflight.json', {'feedback_protocol':protocol(config),
            'tools':list(server._tool_manager._tools), 'command_not_executed':command, 'passed':True})
    args.mode = 'batch'
    args.coordinator = coordinator = Coordinator(root, seeds, args.protocol_smoke)
    write_json(root/'run_manifest.json', {'feedback_protocol':protocol(config), 'scored': False, 'seeds': seeds, 'max_simulator_starts': args.concurrency * (3 if args.protocol_smoke else 1),
        'max_codex_starts': args.concurrency, 'config': config, 'started_s': time.time(),
        'repo_head': subprocess.check_output(['git','rev-parse','HEAD'], cwd=REPO, text=True).strip(),
        'source_head': subprocess.check_output(['git','rev-parse','HEAD'], cwd=args.source_root, text=True).strip()})
    (root/'git_status.txt').write_text(subprocess.check_output(['git','status','--short'], cwd=REPO,text=True))
    (root/'git_diff.patch').write_text(subprocess.check_output(['git','diff'],cwd=REPO,text=True))
    previous = {}
    write_json(root/'resources_before.json', resources(coordinator, previous))
    (root/'hardware.txt').write_text(subprocess.check_output(['nvidia-smi'], text=True)+'\n'+subprocess.check_output(['lscpu'],text=True))
    done = threading.Event()
    def sample():
        while not done.is_set():
            start = time.monotonic()
            try:
                row = resources(coordinator, previous)
                with (root/'resources.jsonl').open('a') as f:
                    f.write(json.dumps(row)+'\n')
                # Actual critical memory exhaustion, not an estimated concurrency gate.
                if row['MemAvailable'] < 256*1024**2:
                    coordinator.abort('MemAvailable below 256 MiB during live batch')
                for seed in seeds:
                    with coordinator.lock:
                        error = coordinator.active_roots.get(seed, root/f'seed_{seed}')/'worker_error.json'
                        if error.exists() and not coordinator.cancel.is_set():
                            coordinator.abort(f'{seed}: '+error.read_text(), seed)
            except Exception as exc:  # noqa: BLE001 -- retain batch/resource failure evidence
                coordinator.event(None, 'resource_sampling_error', error=str(exc))
            done.wait(max(0, 1-(time.monotonic()-start)))
    monitor = threading.Thread(target=sample)
    monitor.start()
    episodes = []
    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = [pool.submit(run_lane, args, config, seed, root/f'seed_{seed}') for seed in seeds]
            for future in futures:
                try:
                    episodes.append(future.result())
                except Exception as exc:  # noqa: BLE001 -- retain batch/resource failure evidence
                    coordinator.abort(str(exc))
                    episodes.append({'infrastructure_error': str(exc), 'evaluable': False})
    finally:
        done.set(); monitor.join()
        write_json(root/'resources_after.json', resources(coordinator, previous))
    write_json(root/'summary.json', {'feedback_protocol':protocol(config), 'scored': False, 'ended_s': time.time(), 'episodes': episodes,
        'all_ready': coordinator.release.is_set(), 'abort_reason': coordinator.abort_reason, 'abort_seed': coordinator.abort_seed,
        'completed_count': sum(e.get('evaluable', False) for e in episodes), 'events': coordinator.events})
    return int(coordinator.cancel.is_set() or any(e.get('infrastructure_error') for e in episodes))


if __name__ == '__main__':
    raise SystemExit(main())
