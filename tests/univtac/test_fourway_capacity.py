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
