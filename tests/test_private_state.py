import pytest
from sim import private_state as state


def test_recording_disabled_does_not_touch_non_simulator_object(monkeypatch):
    monkeypatch.delenv('OPENETA_PRIVATE_STATE_DIR',raising=False)
    assert state.capture(object(),'test') is None


def test_post_snapshot_failure_cannot_erase_executed_motion(monkeypatch):
    captures=[]
    def capture(*args):
        captures.append(args[1])
        if len(captures)==2:raise OSError('disk unavailable')
        return None
    monkeypatch.setattr(state,'capture',capture)
    result=state.recorded(object(),'move',{},lambda:{'steps_executed':7,'stop_reason':'collision_detected'})
    assert result['steps_executed']==7 and result['stop_reason']=='collision_detected'
    assert result['operator_recording_status']=='post_state_failed'


def test_initial_snapshot_failure_prevents_unrecorded_actuation(monkeypatch):
    called=[]
    def capture(*args):raise OSError('disk unavailable')
    monkeypatch.setattr(state,'capture',capture)
    with pytest.raises(OSError):state.recorded(object(),'move',{},lambda:called.append(True))
    assert not called
