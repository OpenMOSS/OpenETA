from copy import deepcopy
from types import SimpleNamespace

import pytest

from agent.runtime.memory import AgentMemory
from tools.codex_host import CodexHost


@pytest.mark.parametrize('case,expected', [
    ('valid', True), ('shaped', False), ('nonterminal', False),
    ('wrong_execution', False), ('wrong_session', False), ('untrusted', False),
    ('truncated', False), ('missing', False),
])
def test_direct_success_read_keeps_official_receipt_gates_without_context(case, expected):
    memory = AgentMemory()
    memory.start_session(task='fixture', session_id='session-1', metadata={
        'env_id': 'openeta/libero_libero_object_task0-v0', 'execution_id': 'execution-1',
        'require_official_reward': True})
    reward, terminated, truncated = 1.0, True, False
    if case == 'shaped':reward = .5
    if case == 'nonterminal':terminated = False
    if case == 'truncated':truncated = True
    receipt = {'schema_version': 'openeta.environment_receipt.v1', 'receipt_id': 'receipt-1',
        'agent_session_id': 'session-1', 'execution_id': 'execution-1',
        'reward_present': True, 'reward': reward, 'terminated': terminated, 'truncated': truncated}
    if case == 'wrong_execution':receipt['execution_id'] = 'other-execution'
    if case == 'wrong_session':receipt['agent_session_id'] = 'other-session'
    if case != 'missing':
        memory.record_environment_receipt(reward=reward, terminated=terminated, truncated=truncated,
            info={'environment_receipt_trusted': case != 'untrusted',
                  'official_reward': True, 'environment_receipt': receipt})
    host = CodexHost.__new__(CodexHost)
    host.runtime = SimpleNamespace(memory=memory)
    def forbidden():raise AssertionError('Official success must not rebuild planner/image context')
    host.context = forbidden
    before = deepcopy(memory.latest_environment_receipt())
    assert bool(host.success_evidence()) is expected
    assert memory.latest_environment_receipt() == before
    if expected:
        memory.record_environment_receipt(reward=0., terminated=False, truncated=False, info={})
        assert not host.success_evidence(), 'An old success must not override a newer non-success receipt'
