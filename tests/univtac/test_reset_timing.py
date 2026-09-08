"""The opt-in probe observes existing calls and preserves return/exception behavior."""
import json
from types import SimpleNamespace

import pytest

from sim.envs.univtac.reset_timing import ResetTiming


def test_existing_calls_once_order_and_restore(tmp_path):
    events=[]
    def action(name):
        def f(*args,**kwargs):
            events.append((name,args,kwargs));return name
        return f
    task=SimpleNamespace(step_count=0,first_frame=None,
        scene=SimpleNamespace(write_data_to_sim=action('write'),update=action('scene')),
        sim=SimpleNamespace(step=action('physics'),render=action('render')),
        _actor_manager=SimpleNamespace(update=action('actor')),
        _tactile_manager=SimpleNamespace(update=action('touch')),
        _update_render=action('update_render'))
    def step(is_save=True):
        task.step_count+=1
        task.scene.write_data_to_sim();task.sim.step(render=False)
        return ('original',is_save)
    task._step=step
    probe=ResetTiming(tmp_path);probe.install_task(task)
    for _ in range(7):assert task._step(is_save=False)==('original',False)
    assert [x[0] for x in events]==['write','physics']*7
    assert events[1][2]=={'render':False}
    probe.close()
    assert task._step is step
    rows=[json.loads(l) for l in (tmp_path/'reset_timing.jsonl').read_text().splitlines()]
    steps=[r for r in rows if r.get('name')=='task._step' and r['event']=='exit']
    assert [r['reset_test_step'] for r in steps]==[1,2,3,4,5]


def test_callback_patch_before_registration_and_original_exception(tmp_path):
    class Callback:
        def step(self,dt=0):
            raise ValueError('native error')
        def update_render_meshes(self,dt=0):return dt
    original=Callback.step
    probe=ResetTiming(tmp_path);probe.install_uipc_callbacks(Callback)
    registered=Callback().step
    probe.active_step=1
    with pytest.raises(ValueError,match='native error'):registered(.01)
    probe.close()
    assert Callback.step is original
    rows=[json.loads(l) for l in (tmp_path/'reset_timing.jsonl').read_text().splitlines()]
    assert any(r.get('error')=='ValueError: native error' for r in rows)


@pytest.mark.parametrize('only',[False,True])
@pytest.mark.parametrize('override',[None,600.0])
def test_worker_timing_returns_same_reset_task_to_live_session(tmp_path,monkeypatch,only,override):
    import sys

    import yaml

    import sim.envs.univtac.autonomous_session as sessions
    import sim.envs.univtac.reset_timing as probes
    from scripts.univtac.serve_autonomous_worker import main

    calls=[];instances=[]
    cfg=SimpleNamespace(step_lim=300,reset_time_limit=120,sim=SimpleNamespace(dt=1/120),decimation=2)
    class Task:
        def __init__(self,cfg,mode):
            self.cfg=cfg;self.mode=mode;self.plan_success=True;self.step_count=240;self._physics_step_count=240
            instances.append(self);calls.append('construct')
        def reset(self,seed):calls.append(('reset',seed))
        def close(self):calls.append('task_close')
    class Probe:
        def __init__(self,root):calls.append('probe')
        def install_uipc_callbacks(self,cls):pass
        def install_task(self,task):pass
        def span(self,name):
            from contextlib import nullcontext
            return nullcontext()
        def close(self):calls.append('probe_close')
    class AppLauncher:
        @staticmethod
        def add_app_launcher_args(parser):pass
        def __init__(self,args):self.app=SimpleNamespace(close=lambda:calls.append('app_close'))
    class Session:
        def __init__(self,task,config,root,seed):
            assert task is instances[0] and calls[-1]=='probe_close'
            calls.append('session');self.host_closed=True;self.recorder=None
        def finalize(self,reason):calls.append('finalize')
    monkeypatch.setitem(sys.modules,'isaaclab.app',SimpleNamespace(AppLauncher=AppLauncher))
    monkeypatch.setitem(sys.modules,'envs.utils.env_parser',SimpleNamespace(
        load_task_config=lambda p:({},None),
        build_task_env_cfg=lambda *a,**k:(SimpleNamespace(Task=Task),cfg,SimpleNamespace(),None)))
    monkeypatch.setitem(sys.modules,'tacex_uipc.sim.uipc_sim',SimpleNamespace(UipcSim=object))
    monkeypatch.setattr(probes,'ResetTiming',Probe)
    monkeypatch.setattr(sessions,'AutonomousSession',Session)
    config=tmp_path/'config.yaml';config.write_text(yaml.safe_dump({'task':'grasp_classify','task_config':'demo','codex_timeout_seconds':3600,'shutdown_timeout_seconds':300,'native_reset_time_limit_seconds':override}))
    assert main(['--repo-root',str(tmp_path),'--source-root',str(tmp_path),'--output-root',str(tmp_path/'out'),
                 '--config',str(config),'--seed','1000028','--reset-timing-only' if only else '--reset-timing'])==0
    assert calls.count('construct')==1 and calls.count(('reset',1000028))==1
    assert calls.count('probe_close')==1
    assert ('session' in calls) is not only
    assert (tmp_path/'out/ready.json').exists() is not only

    recorded=json.loads((tmp_path/'out/native_reset_limit.json').read_text())
    assert recorded=={'native_default_seconds':120.0,'override_seconds':override,'configured_seconds':override or 120.0,'actual_task_seconds':override or 120.0}


def test_native_interval_clock_is_read_not_replaced(tmp_path):
    import time
    noop=lambda *a,**k:None
    task=SimpleNamespace(step_count=0,first_frame=None,scene=SimpleNamespace(write_data_to_sim=noop,update=noop),
        sim=SimpleNamespace(step=noop,render=noop),_actor_manager=SimpleNamespace(update=noop),
        _tactile_manager=SimpleNamespace(update=noop),_update_render=noop)
    def step():task.step_count+=1
    task._step=step
    probe=ResetTiming(tmp_path);probe.install_task(task)
    def reset():
        reset_test_start=time.perf_counter();total_cost=.1
        task._step();task._step()
        assert total_cost==.1
        return reset_test_start
    origin=reset();probe.close()
    rows=[json.loads(l) for l in (tmp_path/'reset_timing.jsonl').read_text().splitlines()]
    measured=[r for r in rows if r['event']=='reset_interval_progress']
    assert len(measured)==2 and all(r['interval_start_monotonic_s']==origin for r in measured)
    assert measured[1]['native_interval_elapsed_s']>=measured[0]['native_interval_elapsed_s']>=0
    assert next(r for r in rows if r['event']=='reset_pretest_guard')['native_elapsed_s']==.1


def test_native_reports_cover_three_intervals_without_extra_steps(tmp_path):
    import time
    events=[]
    noop=lambda *a,**k:None
    task=SimpleNamespace(step_count=0,first_frame=None,scene=SimpleNamespace(write_data_to_sim=noop,update=noop),
        sim=SimpleNamespace(step=noop,render=noop),_actor_manager=SimpleNamespace(update=noop),
        _tactile_manager=SimpleNamespace(update=noop),_update_render=noop)
    def step(is_save=False):
        events.append(('step',is_save));task.step_count+=1
        return 'original'
    task._step=step
    class Native:
        @staticmethod
        def get_sim_time_report(as_json):
            assert as_json is True
            events.append(('report',task.step_count))
            return {'name':'GlobalTimer','duration':0,'count':1,'children':[]}
    probe=ResetTiming(tmp_path);probe.enable_native_reports(Native);probe.install_task(task)
    probe.export_native_report('after_construct',0)
    def reset():
        for count in (5,20,5):
            reset_test_start=time.perf_counter()
            for _ in range(count):assert task._step(is_save=False)=='original'
            task.first_frame=7
            assert reset_test_start>0
    reset();probe.export_native_report('reset_final',task.step_count);probe.close()
    rows=[json.loads(l) for l in (tmp_path/'reset_timing.jsonl').read_text().splitlines()]
    steps=[r for r in rows if r.get('name')=='task._step' and r['event']=='exit']
    assert [r['reset_test_step'] for r in steps]==list(range(1,31))
    assert [r['test_interval'] for r in steps]==['initial_5']*5+['post_actor_20']*20+['final_5']*5
    assert len([r for r in events if r[0]=='step'])==30
    assert len(list((tmp_path/'uipc_timer').glob('*.json')))==62
    assert task._step is step


def test_native_report_failure_does_not_replace_native_exception(tmp_path):
    class Native:
        @staticmethod
        def get_sim_time_report(as_json):raise RuntimeError('report unavailable')
    probe=ResetTiming(tmp_path);probe.enable_native_reports(Native)
    probe.export_native_report('reset_final',15);probe.close()
    row=json.loads(next((tmp_path/'uipc_timer').glob('*.json')).read_text())
    assert row['unavailable']=='RuntimeError: report unavailable'
    assert row['export_call_seconds']>=0
