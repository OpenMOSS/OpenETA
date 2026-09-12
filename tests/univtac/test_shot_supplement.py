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
