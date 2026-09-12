"""Frozen 40-task, two-independent-trial Codex/LIBERO evaluation."""
from __future__ import annotations
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

from scripts import codex_campaign as runtime

ROOT = Path(__file__).resolve().parents[1]
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10')


def tasks_from_preflight(report):
    if report.get('passed') is not True or report.get('cleanup') is not True:
        raise ValueError('Successful independent initial-state preflight required')
    rows = report['all_state_files']
    expected = {(suite, index) for suite in SUITES for index in range(10)}
    if len(rows) != 40 or {(x['suite'], x['task_index']) for x in rows} != expected:
        raise ValueError('Exactly the four official ten-task suites are required')
    return [{**x, 'key': f"{x['suite']}:{x['task_index']}",
             'env_id': f"openeta/libero_{x['suite']}_task{x['task_index']}-v0"} for x in rows]


def schedule(tasks):
    # Execute both rounds, including second trials of first-round successes.
    return [(task, repetition) for repetition in (1, 2) for task in tasks]


def scored_success(entry):
    return (runtime.successful_attempt(entry)
            and entry.get('initial_state_verified') is True
            and entry.get('host', {}).get('official_task_success') is True)


def metrics(tasks, trials):
    finished = [x for x in trials.values() if x.get('finalized') is True]
    first = sum(scored_success(trials.get(f"{t['key']}/1", {})) for t in tasks)
    at_two = sum(any(scored_success(trials.get(f"{t['key']}/{n}", {})) for n in (1, 2)) for t in tasks)
    successes = sum(scored_success(x) for x in finished)
    return {'completed_trials': len(finished), 'planned_trials': 80,
            'complete': len(finished) == 80,
            'pass_at_1_count': first, 'pass_at_2_count': at_two, 'task_denominator': 40,
            'pass_at_1': first / 40, 'pass_at_2': at_two / 40,
            'rates_are_lower_bounds_until_complete': len(finished) != 80,
            'successful_trials': successes,
            'attempt_success_rate': successes / len(finished) if finished else None,
            'integration_failures': sum(x.get('integration_passed') is not True for x in finished),
            'stream_error_events': sum(x.get('stream_errors', 0) for x in finished)}


def write_results(base, state):
    report = metrics(state['tasks'], state['trials'])
    runtime.save(base / 'results.json', {'metrics': report, 'trials': state['trials']})
    lines = ['# LIBERO 40 × 2 — Astra high', '',
             f"Completed trials: {report['completed_trials']}/80. Batch complete: {report['complete']}.",
             f"pass@1: {report['pass_at_1_count']}/40; pass@2: {report['pass_at_2_count']}/40; successful trials: {report['successful_trials']}/{report['completed_trials']}.",
             'Task rates are lower bounds until all 80 trials finish. Infrastructure failures are retained; no replacement trials.', '',
             '| Task | Trial 1 | Trial 2 |', '|---|---|---|']
    for task in state['tasks']:
        cells = []
        for n in (1, 2):
            entry = state['trials'].get(f"{task['key']}/{n}", {})
            if entry.get('finalized') is not True:
                cells.append('pending')
            else:
                label = 'success' if scored_success(entry) else 'integration failure' if not entry.get('integration_passed') else 'task failure' if not entry.get('task_success') else 'audit failure'
                result = Path(entry['output']).relative_to(base) / 'result.json'
                cells.append(f'[{label}]({result})')
        lines.append(f"| {task['key']} | {' | '.join(cells)} |")
    lines += ['', f"Integration failures: {report['integration_failures']}; logged stream errors: {report['stream_error_events']}.",
              '', 'Same official initial-state index 0 and independent contexts; one fixed state per task, not the full official initial-state benchmark. Official success may occur before gripper release.',
              '', '[Frozen protocol](protocol.json) · [Machine-readable results](results.json)']
    temp = base / 'results.md.tmp'
    temp.write_text('\n'.join(lines) + '\n')
    temp.replace(base / 'results.md')


def file_snapshot(libero_root, config_dir, catalog):
    files = runtime.source_snapshot()
    files['@marketplace'] = runtime.digest(ROOT / '.agents/plugins/marketplace.json')
    files['@catalog'] = runtime.digest(catalog)
    files['@libero_config'] = runtime.digest(config_dir / 'config.yaml')
    files['@dependency_versions'] = hashlib.sha256(json.dumps(dependency_versions(), sort_keys=True).encode()).hexdigest()
    tracked = subprocess.check_output(['git', '-C', str(libero_root), 'ls-files', '-z']).decode().split('\0')
    for name in tracked:
        if name and (libero_root / name).is_file():
            files['@libero/' + name] = runtime.digest(libero_root / name)
    for path in sorted((ROOT / 'tmp/codex-mink-deps').rglob('*.py')):
        files['@mink_overlay/' + str(path.relative_to(ROOT / 'tmp/codex-mink-deps'))] = runtime.digest(path)
    codex = Path(shutil.which('codex')).resolve()
    files['@codex_entrypoint'] = runtime.digest(codex)
    package = codex.parents[1]
    for path in package.rglob('codex'):
        if path.is_file():
            files['@codex_binary/' + str(path.relative_to(package))] = runtime.digest(path)
    return files


def dependency_versions():
    command = 'import importlib.metadata as m,json; print(json.dumps(dict(sorted((x.metadata["Name"],x.version) for x in m.distributions()))))'
    return {name: json.loads(subprocess.check_output([str(ROOT / executable), '-c', command]))
            for name, executable in [('host', '.venv/bin/python'), ('simulator', 'sim/venvs/libero/bin/python')]}


def verify_initialization(setup, task, prior):
    path = setup / 'private-state/initialization.jsonl'
    if not path.exists():
        return False, None
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not rows or any(r['initial_state_index'] != 0 or r['reset_seed'] != 0
                       or r['task'] != task['task'] or r['suite'] != task['suite']
                       or r['initial_states_file_sha256'] != task['sha256'] for r in rows):
        return False, None
    actual = {r['actual_initial_state_sha256'] for r in rows}
    if len(actual) != 1:
        return False, None
    value = next(iter(actual))
    return prior is None or value == prior, value


def run_trial(base, task, repetition, state, frozen, snapshot, env):
    key = f"{task['key']}/{repetition}"
    destination = base / task['suite'] / f"task-{task['task_index']:02d}" / f'trial-{repetition:02d}'
    destination.mkdir(parents=True, exist_ok=False)
    setup, out = destination / 'setup', destination / 'run'
    setup.mkdir()
    network = destination / 'network'
    network.mkdir()
    shutil.copy2(ROOT / 'tmp/codex-goal3-symmetry-20260910/network/monitor.py', network / 'monitor.py')
    entry = {'task': task['key'], 'repetition': repetition, 'output': str(destination),
             'started': runtime.now(), 'status': 'starting', 'finalized': False,
             'model': 'gpt-6-astra', 'reasoning_effort': 'high'}
    state['trials'][key] = entry
    state['current_trial'] = key
    runtime.state()
    runtime.save(destination / 'source-snapshot.json', frozen)
    local_env = dict(env, OPENETA_WORKER_LOG_DIR=str(setup / 'worker-logs'))
    timeline = runtime.Timeline(out)
    processes, owned = [], {}
    try:
        if not runtime.port_free():
            raise RuntimeError('Dedicated port occupied; its owner is not touched')
        with (setup / 'sim.log').open('w') as sim_log, (setup / 'launcher.log').open('w') as launch_log, (network / 'monitor.log').open('w') as net_log:
            monitor = subprocess.Popen(['/usr/bin/python3', str(network / 'monitor.py')], env=local_env, stdout=net_log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(monitor)
            server = subprocess.Popen([str(ROOT / '.venv/bin/python'), '-u', '-m', 'scripts.codex_sim_server',
                                       '--host', '127.0.0.1', '--port', str(runtime.PORT),
                                       '--private-state-dir', str(setup / 'private-state')], cwd=ROOT, env=local_env, stdout=sim_log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(server)
            entry['server_pid'] = server.pid
            deadline = time.monotonic() + 30
            while runtime.port_free():
                if runtime.STOP or server.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('Simulator initialization failed or was interrupted')
                time.sleep(.2)
            command = [str(ROOT / '.venv/bin/python'), '-u', '-m', 'scripts.codex_plugin_smoke',
                       '--sim-url', f'http://127.0.0.1:{runtime.PORT}/sse', '--env-id', task['env_id'],
                       '--task', task['language'] + '.', '--seed', '0', '--output', str(out),
                       '--model', 'gpt-6-astra', '--effort', 'high', '--timeout', '2400',
                       '--max-requests', '160', '--tool-profile', 'atomic',
                       '--model-catalog-json', str(base / 'model-catalog.json')]
            runtime.save(setup / 'command.json', command)
            launcher = subprocess.Popen(command, cwd=ROOT, env=local_env, stdout=launch_log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(launcher)
            entry.update(launcher_pid=launcher.pid, status='running')
            deadline = time.monotonic() + 2850
            last = 0
            while launcher.poll() is None and not runtime.STOP and time.monotonic() < deadline:
                runtime.track([p.pid for p in processes], owned)
                timeline.poll()
                if time.monotonic() - last >= 5:
                    entry.update(runtime.final_observation({'host': runtime.readjson(out / 'host/host-status.json')}, timeline))
                    state['metrics'] = metrics(state['tasks'], state['trials'])
                    runtime.state()
                    last = time.monotonic()
                time.sleep(1)
            entry.update(launcher_returncode=launcher.poll(), status='interrupted' if runtime.STOP else 'watchdog_timeout' if launcher.poll() is None else 'finished')
    except Exception as exc:
        entry.update(status='setup_or_supervisor_error', error=str(exc))
    finally:
        entry['remaining_owned_pids'] = runtime.cleanup(processes, owned)
        (out / 'codex-home/auth.json').unlink(missing_ok=True)
        timeline.poll()
        summary = runtime.analyze(out, timeline)['summary'] if out.exists() else {}
        entry.update(runtime.final_observation(summary, timeline))
        current = snapshot()
        entry.update(task_success=summary.get('task_success') is True,
                     integration_passed=summary.get('integration_passed') is True,
                     elapsed_s=summary.get('elapsed_s'), ended=runtime.now(),
                     port_released=runtime.port_free(), private_auth_removed=not (out / 'codex-home/auth.json').exists(),
                     source_changed=[p for p in sorted(set(frozen) | set(current)) if frozen.get(p) != current.get(p)])
        prior = state['trials'].get(f"{task['key']}/1", {}) if repetition == 2 else {}
        entry['initial_state_verified'], entry['actual_initial_state_sha256'] = verify_initialization(setup, task, prior.get('actual_initial_state_sha256'))
        entry['finalized'] = True
        runtime.save(destination / 'result.json', entry)
        state['metrics'] = metrics(state['tasks'], state['trials'])
        runtime.state()
        write_results(base, state)
        print(json.dumps({'trial': key, 'task_success': entry['task_success'], 'metrics': state['metrics']}, ensure_ascii=False), flush=True)
    return entry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--libero-root', type=Path, required=True)
    parser.add_argument('--model-catalog-json', type=Path, required=True)
    parser.add_argument('--port', type=int, default=18787)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    base, libero_root = args.output.resolve(), args.libero_root.resolve()
    base.mkdir(parents=True, exist_ok=True)
    lock = (base / 'campaign.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    tasks = tasks_from_preflight(runtime.readjson(base / 'preflight.json'))
    catalog = base / 'model-catalog.json'
    if not catalog.exists():
        shutil.copy2(args.model_catalog_json, catalog)
    if runtime.digest(catalog) != runtime.digest(args.model_catalog_json):
        raise RuntimeError('Requested catalog differs from frozen catalog')
    config = base / 'libero-config'
    protocol = {'schema': 'openeta.libero_pass2.v1', 'model': 'gpt-6-astra', 'effort': 'high',
                'controller': 'mink_joint_velocity', 'tool_profile': 'atomic', 'seed': 0,
                'initial_state_index': 0, 'initial_state_policy': 'same_official_state_per_task',
                'settle_steps': 5, 'repetitions': 2, 'episode_timeout_s': 2400, 'max_requests': 160,
                'memory_between_trials': False, 'tasks': tasks, 'libero_root': str(libero_root),
                'libero_commit': subprocess.check_output(['git', '-C', str(libero_root), 'rev-parse', 'HEAD']).decode().strip(),
                'codex_version': subprocess.check_output(['codex', '--version']).decode().strip(),
                'dependency_versions': dependency_versions(),
                'model_catalog_sha256': runtime.digest(catalog),
                'failed_trial_policy': 'retain_and_count; integration failures pause queue; no replacement trials'}
    runtime.BASE, runtime.PORT = base, args.port
    snapshot = lambda: file_snapshot(libero_root, config, catalog)
    frozen = snapshot()
    protocol['source_snapshot_sha256'] = hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()
    path = base / 'protocol.json'
    if path.exists() and runtime.readjson(path) != protocol:
        raise RuntimeError('Frozen protocol/source differs; use a new experiment, not a mixed-version resume')
    if not path.exists():
        runtime.save(path, protocol)
        runtime.save(base / 'frozen-source.json', frozen)
    if args.prepare_only:
        print(json.dumps({'prepared': True, 'trials': 80, 'protocol': str(path)}))
        return
    state = runtime.readjson(base / 'batch-status.json') or {'created': runtime.now(), 'tasks': tasks, 'trials': {}}
    runtime.STATE = state
    state.update(status='running', pid=os.getpid(), updated=runtime.now())
    state.pop('error', None)
    write_results(base, state)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, runtime.stop)
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=str(ROOT), LIBERO_DIR=str(libero_root),
               LIBERO_CONFIG_PATH=str(config), OPENETA_LIBERO_INIT_STATE_INDEX='0',
               OPENETA_WORKER_GPUS='0', OPENETA_WORKER_POOL_MAX='1')
    try:
        for task, repetition in schedule(tasks):
            key = f"{task['key']}/{repetition}"
            previous = state['trials'].get(key)
            if previous:
                if previous.get('finalized') is not True:
                    raise RuntimeError('An unfinalized previous trial requires ownership/cleanup audit')
                continue
            if runtime.STOP or (base / 'pause-after-current').exists():
                break
            if snapshot() != frozen:
                raise RuntimeError('Frozen source changed before next trial')
            entry = run_trial(base, task, repetition, state, frozen, snapshot, env)
            issue = runtime.attempt_boundary_error(entry)
            if issue or not entry['initial_state_verified']:
                raise RuntimeError(issue or 'Official initial-state identity check failed')
        state['metrics'] = metrics(tasks, state['trials'])
        state['status'] = 'complete' if state['metrics']['complete'] else 'paused'
    except Exception as exc:
        state.update(status='error', error=str(exc))
        raise
    finally:
        state['ended'] = runtime.now()
        runtime.state()
        write_results(base, state)
        lock.close()


if __name__ == '__main__':
    main()
