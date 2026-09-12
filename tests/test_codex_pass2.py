import json
import pytest

from scripts.codex_pass2 import SUITES, tasks_from_preflight, schedule, metrics, verify_initialization


def task_rows():
    return tasks_from_preflight({'passed': True, 'cleanup': True, 'all_state_files': [
        {'suite': suite, 'task_index': i, 'task': f'name-{i}', 'sha256': 'file-hash'}
        for suite in SUITES for i in range(10)]})


def passed():
    return dict(finalized=True, task_success=True, integration_passed=True,
                port_released=True, private_auth_removed=True, remaining_owned_pids=[],
                source_changed=[], initial_state_verified=True,
                host={'official_task_success': True})


def test_two_complete_rounds_include_first_round_successes():
    tasks = task_rows()
    jobs = schedule(tasks)
    assert len(jobs) == 80
    assert len({(t['key'], n) for t, n in jobs}) == 80
    assert [n for _, n in jobs] == [1]*40 + [2]*40


def test_pass2_uses_task_union_and_keeps_failed_trials_in_denominator():
    tasks = task_rows()
    trials = {f"{t['key']}/{n}": passed() | {'task_success': False}
              for t, n in schedule(tasks)}
    trials[f"{tasks[0]['key']}/1"] = passed()
    trials[f"{tasks[0]['key']}/2"] = passed()
    trials[f"{tasks[1]['key']}/2"] = passed()
    trials[f"{tasks[2]['key']}/2"] = passed() | {'integration_passed': False}
    report = metrics(tasks, trials)
    assert report['complete'] and report['completed_trials'] == 80
    assert report['pass_at_1'] == 1/40 and report['pass_at_2'] == 2/40
    assert report['attempt_success_rate'] == 3/80
    assert report['integration_failures'] == 1
    del trials[f"{tasks[-1]['key']}/2"]
    assert metrics(tasks, trials)['rates_are_lower_bounds_until_complete']


@pytest.mark.parametrize('change', [
    {'initial_state_verified': False}, {'source_changed': ['runtime.py']},
    {'host': {'official_task_success': False}}, {'remaining_owned_pids': [999]},
])
def test_contaminated_or_unverified_success_is_not_scored(change):
    tasks = task_rows()
    report = metrics(tasks, {f"{tasks[0]['key']}/1": passed() | change})
    assert report['pass_at_2_count'] == 0


def test_repetitions_require_identical_actual_initial_state(tmp_path):
    task = task_rows()[0]
    p = tmp_path/'private-state'
    p.mkdir()
    receipt = {'initial_state_index': 0, 'reset_seed': 0,
               'task': task['task'], 'suite': task['suite'],
               'initial_states_file_sha256': task['sha256'],
               'actual_initial_state_sha256': 'exact-state'}
    (p/'initialization.jsonl').write_text(json.dumps(receipt)+'\n')
    assert verify_initialization(tmp_path, task, None) == (True, 'exact-state')
    assert verify_initialization(tmp_path, task, 'exact-state')[0]
    assert not verify_initialization(tmp_path, task, 'different-state')[0]
    receipt['initial_state_index'] = 1
    (p/'initialization.jsonl').write_text(json.dumps(receipt)+'\n')
    assert not verify_initialization(tmp_path, task, None)[0]


def test_invalid_or_incomplete_task_manifest_is_rejected():
    with pytest.raises(ValueError):
        tasks_from_preflight({'passed': True, 'cleanup': True, 'all_state_files': []})
    with pytest.raises(ValueError):
        tasks_from_preflight({'passed': False, 'cleanup': True})
