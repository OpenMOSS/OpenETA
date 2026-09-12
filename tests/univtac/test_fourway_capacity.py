import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from scripts.univtac.run_codex_readonly_observation import _run_to_files
from scripts.univtac.run_fourway_capacity import SEEDS, Coordinator


def test_barrier_requires_four_live_workers(tmp_path):
    coordinator = Coordinator(tmp_path)
    coordinator.processes = {(s, 'worker'): SimpleNamespace(poll=lambda: None) for s in SEEDS}
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = []
        for seed in SEEDS[:3]:
            coordinator.ready[seed] = {}
            futures.append(pool.submit(coordinator.barrier, seed))
        time.sleep(0.2)
        assert not coordinator.release.is_set()
        coordinator.ready[SEEDS[3]] = {}
        futures.append(pool.submit(coordinator.barrier, SEEDS[3]))
        for future in futures:
            future.result(timeout=2)
    assert coordinator.release.is_set()
    assert sum(e['event'] == 'all_ready_resident' for e in coordinator.events) == 1


def test_failed_lane_cancels_before_release(tmp_path):
    coordinator = Coordinator(tmp_path)
    coordinator.abort('one initialization failed')
    with pytest.raises(RuntimeError, match='cancelled'):
        coordinator.barrier(SEEDS[0])
    assert not coordinator.release.is_set()


def test_operator_cancellation_cleans_owned_process(tmp_path):
    cancel = threading.Event()
    def started(process):
        (tmp_path/'pid.json').write_text(json.dumps(process.pid))
        cancel.set()
    result = _run_to_files([sys.executable, '-c', 'import time; time.sleep(60)'],
        cwd=tmp_path, environment=os.environ.copy(), stdout_path=tmp_path/'out',
        stderr_path=tmp_path/'err', timeout_seconds=60, cancel_event=cancel, on_started=started)
    assert result['cancelled'] and not result['timed_out']
    assert result['exit_mode'] == 'batch_cancelled'
    with pytest.raises(ProcessLookupError):
        os.kill(result['pid'], 0)


def test_two_way_barrier_and_abort_source(tmp_path):
    seeds = SEEDS[:2]
    coordinator = Coordinator(tmp_path, seeds)
    coordinator.processes = {(s, 'worker'): SimpleNamespace(poll=lambda: None) for s in seeds}
    with ThreadPoolExecutor(max_workers=2) as pool:
        coordinator.ready[seeds[0]] = {}
        first = pool.submit(coordinator.barrier, seeds[0])
        time.sleep(0.2)
        assert not coordinator.release.is_set()
        coordinator.ready[seeds[1]] = {}
        second = pool.submit(coordinator.barrier, seeds[1])
        first.result(timeout=2)
        second.result(timeout=2)
    assert len(coordinator.phases) == 2
    coordinator.abort('initialization failure', seeds[0])
    coordinator.abort('collateral cancellation', seeds[1])
    assert coordinator.abort_seed == seeds[0]


def test_smoke_retry_is_pre_ready_only(tmp_path, monkeypatch):
    from scripts.univtac.run_fourway_capacity import run_lane
    coordinator = Coordinator(tmp_path, SEEDS[:2], protocol_smoke=True)
    args=SimpleNamespace(coordinator=coordinator)
    calls=[]
    def episode(args,config,seed,folder):
        calls.append(folder)
        folder.mkdir(parents=True)
        (folder/'worker_lifecycle.json').write_text(json.dumps({'cleanup_complete':True}))
        coordinator.processes[seed,'worker']=SimpleNamespace(poll=lambda:0)
        if len(calls)==1:
            coordinator.abort('native reset failed',seed)
            assert not coordinator.cancel.is_set()
            return {'seed':seed,'infrastructure_error':'native reset failed','codex_process_count':0}
        coordinator.ready[seed]={}
        return {'seed':seed,'codex_process_count':1,'evaluable':True,'task_success':False}
    monkeypatch.setattr('scripts.univtac.run_fourway_capacity.run_episode',episode)
    result=run_lane(args,{},SEEDS[0],tmp_path/'lane')
    assert len(calls)==2 and result['task_success'] is False
    assert [p.name for p in calls]==['attempt_1','attempt_2']


def test_smoke_accepts_first_ready_without_waiting_or_replacement(tmp_path):
    coordinator=Coordinator(tmp_path,SEEDS[:2],protocol_smoke=True)
    coordinator.ready[SEEDS[0]]={}
    coordinator.barrier(SEEDS[0])
    coordinator.processes[SEEDS[1],'worker']=SimpleNamespace(poll=lambda:0)
    coordinator.abort('other initialization failed',SEEDS[1])
    assert not coordinator.cancel.is_set()
    assert not coordinator.lane_cancel[SEEDS[0]].is_set()
    assert coordinator.lane_cancel[SEEDS[1]].is_set()
    assert any(e['event']=='operator_released' for e in coordinator.events)


def test_capacity_pause_drains_but_hard_abort_still_cancels(tmp_path):
    from scripts.univtac.run_fourway_capacity import Coordinator
    c = Coordinator(tmp_path, seeds=(1, 2), protocol_smoke=True)
    c.pause_dispatch("provider_model_capacity", 1)
    assert c.dispatch_paused.is_set()
    assert not c.cancel.is_set()
    assert not any(e.is_set() for e in c.lane_cancel.values())
    c.abort("worker_shutdown_timeout")
    assert c.cancel.is_set()
    assert all(e.is_set() for e in c.lane_cancel.values())
