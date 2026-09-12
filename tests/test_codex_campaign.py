import pytest
from scripts.codex_campaign import successful_attempt, attempt_boundary_error


def test_final_observation_uses_success_after_last_live_poll(tmp_path):
    import json
    from scripts.codex_campaign import Timeline, final_observation
    timeline = Timeline(tmp_path)
    path = tmp_path / 'codex-events.jsonl'
    event = {'type': 'item.completed', 'item': {'type': 'mcp_tool_call',
             'id': 'last-open', 'tool': 'gripper_control', 'status': 'completed'}}
    path.write_text(json.dumps(event) + '\n')
    timeline.poll()
    summary = {'host': {'requests': 26, 'tool_calls': 44, 'closed': True,
                       'official_task_success': True}}
    entry = {'native_completed_calls': 0,
             'host': {'requests': 25, 'tool_calls': 43, 'closed': False,
                      'official_task_success': False}}
    entry.update(final_observation(summary, timeline))
    assert entry['native_completed_calls'] == 1
    assert entry['host']['requests'] == 26 and entry['host']['tool_calls'] == 44
    assert entry['host']['closed'] is True and entry['host']['official_task_success'] is True
    assert entry['latest_native']['id'] == 'last-open'
    assert final_observation({}, timeline)['host']['official_task_success'] is None


def test_campaign_continues_robot_failure_but_stops_integration_failure():
    entry = {'status': 'finished', 'task_success': False, 'integration_passed': True,
             'port_released': True, 'private_auth_removed': True,
             'remaining_owned_pids': [], 'source_changed': []}
    assert attempt_boundary_error(entry) is None
    entry['integration_passed'] = False
    assert 'integration failed' in attempt_boundary_error(entry)
    entry['status'] = 'interrupted'
    assert attempt_boundary_error(entry) is None
    entry['private_auth_removed'] = False
    assert 'Cleanup incomplete' in attempt_boundary_error(entry)


def test_importing_campaign_does_not_replace_callers_signal_handlers():
    import subprocess
    import sys
    result = subprocess.run([sys.executable, '-c',
        'import signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); '
        'import scripts.codex_campaign; '
        'assert signal.getsignal(signal.SIGTERM) == signal.SIG_IGN'],
        capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('field,value', [('task_success', False), ('task_success', 'true'),
    ('integration_passed', False), ('port_released', False), ('private_auth_removed', False),
    ('remaining_owned_pids', [123]), ('source_changed', ['sim/bench_worker.py'])])
def test_campaign_never_counts_incomplete_or_contaminated_attempt(field, value):
    entry = {'task_success': True, 'integration_passed': True, 'port_released': True,
             'private_auth_removed': True, 'remaining_owned_pids': [], 'source_changed': []}
    assert successful_attempt(entry)
    entry[field] = value
    assert not successful_attempt(entry)
    entry.pop(field)
    assert not successful_attempt(entry)
