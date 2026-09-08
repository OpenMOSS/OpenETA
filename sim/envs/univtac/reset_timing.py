"""Opt-in timing of existing initialization calls; never advances simulation."""
from __future__ import annotations

import faulthandler
import functools
import json
import os
import resource
import sys
import time
from contextlib import contextmanager
from pathlib import Path


class ResetTiming:
    def __init__(self, root):
        self.root = root
        self.native_report = None
        self.report_index = 0
        self.intervals = {}
        self.active_interval = None
        self.log = (root/'reset_timing.jsonl').open('x')
        self.stacks = (root/'reset_stacks.txt').open('x')
        self.started = time.perf_counter()
        self.active_step = None
        self.restores = []
        self.write(event='diagnostic_enter', pid=os.getpid())

    def write(self, **row):
        row.setdefault('test_interval', self.active_interval)
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

    def enable_native_reports(self, cls):
        # UipcSim already enables Timer. Export drains its existing window;
        # never toggle Timer or add device synchronization here.
        self.native_report = cls.get_sim_time_report
        (self.root/'uipc_timer').mkdir(exist_ok=True)

    def export_native_report(self, boundary, native_step=None):
        if self.native_report is None:
            return
        self.report_index += 1
        started = time.perf_counter()
        row = {'boundary':boundary, 'native_step':native_step,
               'test_interval':self.active_interval, 'window':'since_previous_export; export clears native timer'}
        try:
            report = self.native_report(as_json=True)
            row['export_call_seconds'] = time.perf_counter()-started
            row['tree'] = report
        except Exception as exc:  # noqa: BLE001 -- retain the original reset failure even if report export fails
            row['export_call_seconds'] = time.perf_counter()-started
            row['unavailable'] = f'{type(exc).__name__}: {exc}'
        path = self.root/'uipc_timer'/f'{self.report_index:04d}_{boundary}.json'
        path.write_text(json.dumps(row, indent=2))
        self.write(event='native_timer_export', path=str(path.relative_to(self.root)),
                   export_call_seconds=row['export_call_seconds'],
                   export_and_write_seconds=time.perf_counter()-started,
                   native_step=native_step, unavailable=row.get('unavailable'))

    def record_uipc_configuration(self, task):
        import importlib.metadata

        import uipc
        sim = task.uipc_sim
        data = {'python_module':uipc.__file__,
                'versions':{name:importlib.metadata.version(name) for name in ('pyuipc','tacex_uipc')},
                'loaded_uipc_binaries':sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                                               if '.so' in line and 'uipc' in line}),
                'workspace':sim.cfg.workspace, 'logger_level':sim.cfg.logger_level,
                'effective_scene_config':sim.scene.config().to_json(),
                'configured_uipc':sim.cfg.to_dict(),
                'timer_semantics':'Existing Timer enabled by UipcSim; export drains. Nested inclusive seconds/count per path. No added sync.'}
        (self.root/'uipc_configuration.json').write_text(json.dumps(data,indent=2,default=str))

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
            caller = sys._getframe(1)
            native_clock = caller.f_locals.get('reset_test_start') if caller.f_code.co_name == 'reset' else None
            if caller.f_code.co_name == '_stabilize_and_calibrate_marker_references':
                native_clock = task.start_time
            if task.step_count == 0 and caller.f_code.co_name == 'reset':
                self.write(event='reset_pretest_guard', native_elapsed_s=caller.f_locals.get('total_cost'))
            def invoke():
                value = original(*args, **kwargs)
                if native_clock is not None:
                    # Read the native clock, never replace it. The native guard
                    # runs just after this wrapper returns, including logging overhead.
                    self.write(event='reset_interval_progress', source_line=caller.f_lineno,
                               source_function=caller.f_code.co_name,
                               interval_start_monotonic_s=native_clock,
                               native_interval_elapsed_s=time.perf_counter()-native_clock,
                               native_step=task.step_count)
                return value
            index = task.step_count+1
            testing = caller.f_code.co_name == 'reset' and native_clock is not None
            if self.native_report and testing:
                if native_clock not in self.intervals:
                    self.intervals[native_clock] = ('initial_5', 'post_actor_20', 'final_5')[len(self.intervals)]
                self.active_interval = self.intervals[native_clock]
                self.export_native_report('before_test_step', task.step_count)
            elif task.first_frame is not None or index > 5:
                return invoke()
            self.active_step = index
            faulthandler.dump_traceback_later(30, repeat=False, file=self.stacks)
            try:
                with self.span('task._step'):
                    return invoke()
            finally:
                faulthandler.cancel_dump_traceback_later()
                if self.native_report and testing:
                    self.export_native_report('after_test_step', task.step_count)
                self.active_step = None
                self.active_interval = None
        task._step = step
        self.restores.append((task, '_step', original))

    def close(self):
        faulthandler.cancel_dump_traceback_later()
        for obj, method, original in reversed(self.restores):
            setattr(obj, method, original)
        self.write(event='diagnostic_close', elapsed_s=time.perf_counter()-self.started)
        self.log.close()
        self.stacks.close()
