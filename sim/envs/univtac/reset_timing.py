"""Opt-in timing of existing initialization calls; never advances simulation."""
from __future__ import annotations

import faulthandler
import functools
import json
import os
import resource
import time
from contextlib import contextmanager


class ResetTiming:
    def __init__(self, root):
        self.log = (root/'reset_timing.jsonl').open('x')
        self.stacks = (root/'reset_stacks.txt').open('x')
        self.started = time.perf_counter()
        self.active_step = None
        self.restores = []
        self.write(event='diagnostic_enter', pid=os.getpid())

    def write(self, **row):
        row.update(monotonic_s=time.perf_counter(), utc_timestamp_s=time.time())
        self.log.write(json.dumps(row)+'\n')
        self.log.flush()

    @contextmanager
    def span(self, name):
        start = time.perf_counter()
        cpu = resource.getrusage(resource.RUSAGE_SELF)
        self.write(event='enter', name=name, reset_test_step=self.active_step)
        error = None
        try:
            yield
        except BaseException as exc:
            error = f'{type(exc).__name__}: {exc}'
            raise
        finally:
            end_cpu = resource.getrusage(resource.RUSAGE_SELF)
            self.write(event='exit', name=name, reset_test_step=self.active_step,
                       elapsed_s=time.perf_counter()-start,
                       process_user_cpu_s=end_cpu.ru_utime-cpu.ru_utime,
                       process_system_cpu_s=end_cpu.ru_stime-cpu.ru_stime,
                       error=error)

    def wrap(self, obj, method, name):
        original = getattr(obj, method)
        @functools.wraps(original)
        def timed(*args, **kwargs):
            if self.active_step is None:
                return original(*args, **kwargs)
            with self.span(name):
                return original(*args, **kwargs)
        setattr(obj, method, timed)
        self.restores.append((obj, method, original))

    def install_uipc_callbacks(self, cls):
        # Patch before Task construction registers bound callbacks. No callback
        # is removed, reordered, or invoked by the diagnostic itself.
        self.wrap(cls, 'step', 'UipcSim.step[advance+retrieve]')
        self.wrap(cls, 'update_render_meshes', 'UipcSim.update_render_meshes')

    def install_task(self, task):
        for obj, method, label in (
            (task.scene, 'write_data_to_sim', 'scene.write_data_to_sim'),
            (task.sim, 'step', 'sim.step'),
            (task.sim, 'render', 'sim.render'),
            (task.scene, 'update', 'scene.update'),
            (task._actor_manager, 'update', 'actor_manager.update'),
            (task._tactile_manager, 'update', 'tactile_manager.update'),
            (task, '_update_render', 'task._update_render'),
        ):
            self.wrap(obj, method, label)
        original = task._step
        @functools.wraps(original)
        def step(*args, **kwargs):
            index = task.step_count+1
            if task.first_frame is not None or index > 5:
                return original(*args, **kwargs)
            self.active_step = index
            faulthandler.dump_traceback_later(30, repeat=False, file=self.stacks)
            try:
                with self.span('task._step'):
                    return original(*args, **kwargs)
            finally:
                faulthandler.cancel_dump_traceback_later()
                self.active_step = None
        task._step = step
        self.restores.append((task, '_step', original))

    def close(self):
        faulthandler.cancel_dump_traceback_later()
        for obj, method, original in reversed(self.restores):
            setattr(obj, method, original)
        self.write(event='diagnostic_close', elapsed_s=time.perf_counter()-self.started)
        self.log.close()
        self.stacks.close()
