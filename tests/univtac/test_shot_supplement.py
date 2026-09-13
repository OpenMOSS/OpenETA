import json
from scripts.univtac.run_shot_supplement import supplement_config, classify


def test_only_initialization_limits_change():
    old = {'startup_timeout_seconds': 900, 'codex_timeout_seconds': 3600,
           'shutdown_timeout_seconds': 300, 'max_tool_calls': 100}
    new = supplement_config(old)
    assert new.pop('disable_initialization_timeout') is True
    assert new == old
    assert old['startup_timeout_seconds'] == 900


def test_native_failure_is_a_result_but_unclean_is_blocked(tmp_path):
    (tmp_path/'episode.json').write_text(json.dumps({'evaluable': True, 'task_success': False}))
    (tmp_path/'worker_lifecycle.json').write_text(json.dumps({'cleanup_complete': True}))
    assert classify(tmp_path)[0] == 'completed'
    (tmp_path/'worker_lifecycle.json').write_text(json.dumps({'cleanup_complete': False}))
    assert classify(tmp_path)[0] == 'blocked'


def test_clean_reset_failure_can_retry_without_operator(tmp_path):
    (tmp_path/'episode.json').write_text(json.dumps({'evaluable': False}))
    (tmp_path/'worker_lifecycle.json').write_text(json.dumps({'cleanup_complete': True}))
    (tmp_path/'worker_error.json').write_text('official reset/pre_move failed')
    assert classify(tmp_path)[0] == 'retry'
    (tmp_path/'ready.json').write_text('{}')
    assert classify(tmp_path)[0] == 'blocked'


def test_deadline_and_missing_usage_are_locked(tmp_path):
    for termination in ['native_success','native_early_stop','native_step_limit','agent_finish','codex_exit']:
        (tmp_path/'episode.json').write_text(json.dumps({'evaluable': True, 'task_success': termination=='native_success', 'termination':termination}))
        (tmp_path/'worker_lifecycle.json').write_text(json.dumps({'cleanup_complete': True}))
        assert classify(tmp_path)[0] == 'completed'
    assert not (tmp_path/'agent_final.md').exists()
    assert not (tmp_path/'codex_trace_summary.json').exists()


def test_provider_missing_result_is_retry_but_unknown_is_blocked(tmp_path):
    (tmp_path/'worker_lifecycle.json').write_text(json.dumps({'cleanup_complete': True}))
    (tmp_path/'ready.json').write_text('{}')
    for error, expected in [('provider_model_capacity','retry'),('provider_service_error','retry'),('adapter_unknown','blocked')]:
        (tmp_path/'episode.json').write_text(json.dumps({'evaluable':False,'infrastructure_error':error}))
        assert classify(tmp_path)[0] == expected


def test_round_keeps_clean_retry_out_until_next_pass():
    from scripts.univtac.run_shot_supplement import round_pending
    cells=[{'cell_key':['t',i]} for i in range(4)]
    state={('t',0):{'status':'retry','last_round':1},('t',1):{'status':'completed','last_round':1},
           ('t',2):{'status':'not_run'},('t',3):{'status':'blocked','last_round':1}}
    assert round_pending(cells,state,1)==[cells[2]]
    assert round_pending(cells,state,2)==[cells[0],cells[2]]


def test_queue_rounds_resume_and_capacity_pause_are_persistent(tmp_path, monkeypatch):
    import threading
    import sys
    from scripts.univtac import run_shot_supplement as runner
    cells=[dict(task='task',seed=i,condition='A',host='hzz-server',cell_key=['task',i,'A'],
                original_config={},mcp_expectation={'expert_ids':[],'images':0}) for i in range(4)]
    (tmp_path/'manifest.json').write_text(json.dumps({'phase':'supplement','cells':cells}))
    calls=[];lock=threading.Lock()
    def fake_run(args,config,seed,folder):
        folder.mkdir(parents=True)
        with lock:calls.append(seed)
        valid=not (seed==0 and folder.name=='attempt_1')
        (folder/'episode.json').write_text(json.dumps({'evaluable':valid,'task_success':False}))
        (folder/'worker_lifecycle.json').write_text(json.dumps({'cleanup_complete':True}))
        if not valid:(folder/'worker_error.json').write_text('official reset/pre_move failed')
    monkeypatch.setattr(runner,'run_episode',fake_run)
    monkeypatch.setattr(runner,'verify_delivery',lambda *_:None)
    monkeypatch.setattr(runner,'render_media',lambda folder,*_:(folder/'media_check.json').write_text('{"passed":true}'))
    monkeypatch.setattr(sys,'argv',['runner','--output-root',str(tmp_path),'--host','hzz-server','--runtime-python',sys.executable,'--source-root',str(tmp_path)])
    runner.main()
    assert len(calls)==5 and calls[-1]==0 and sorted(calls[:4])==[0,1,2,3]
    runner.main()
    assert len(calls)==5  # Valid failures with no usage/final never start again.
    (tmp_path/'dispatch_pause.json').write_text('{"reason":"quota or capacity","requires_user_release":true}')
    import pytest
    with pytest.raises(RuntimeError,match='Persistent dispatch pause'):
        runner.main()
    assert len(calls)==5


def test_unknown_pre_ready_failure_hard_stops_sibling(tmp_path, monkeypatch):
    import sys
    from scripts.univtac import run_shot_supplement as runner
    cells=[dict(task='task',seed=i,condition='A',host='hzz-server',cell_key=['task',i,'A'],original_config={}) for i in range(2)]
    (tmp_path/'manifest.json').write_text(json.dumps({'phase':'supplement','cells':cells}))
    observed=[]
    def fake_run(args,config,seed,folder):
        folder.mkdir(parents=True)
        if seed==1:
            observed.append(args.coordinator.cancel.wait(2))
        (folder/'episode.json').write_text(json.dumps({'evaluable':False,'infrastructure_error':'unknown_adapter'}))
        (folder/'worker_lifecycle.json').write_text(json.dumps({'cleanup_complete':True}))
    monkeypatch.setattr(runner,'run_episode',fake_run)
    monkeypatch.setattr(sys,'argv',['runner','--output-root',str(tmp_path),'--host','hzz-server','--runtime-python',sys.executable,'--source-root',str(tmp_path)])
    runner.main()
    assert (tmp_path/'dispatch_pause.json').exists()
    assert all(observed)  # A sibling already dispatched is globally cancelled.
