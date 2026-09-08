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
