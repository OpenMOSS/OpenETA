import pytest

from sim.mcp_server import server as s


@pytest.fixture
def setup(monkeypatch):
    meta = {"_sid": "gripper-final-test", "worker_url": "fixture", "remote_handle": "h"}
    step_result = {"observation": {"robot": {"joint_positions": [0.5]}, "cameras": []},
                   "reward": 0.25, "terminated": False, "truncated": False,
                   "info": {"source": "physics"}}
    final = {"robot": {"joint_positions": [0.5]},
             "cameras": [{"frame_id": "final-image", "rgb_base64": "fixture"}]}
    calls = []
    def step(meta, action, *, num_steps, render):
        calls.append(("step", num_steps, render))
        return step_result
    def render(meta):
        calls.append(("render",))
        return final
    monkeypatch.setattr(s, "_proxy_step", step)
    monkeypatch.setattr(s, "_proxy_render", render)
    monkeypatch.setattr(s, "_session_last_obs", {})
    return meta, step_result, final, calls


@pytest.mark.parametrize("steps", [40, 60])
def test_gripper_batch_renders_once_after_physics_and_preserves_receipt(setup, steps):
    meta, _, final, calls = setup
    result = s._step_gripper_with_final_observation(meta, [1.0], num_steps=steps)
    assert calls == [("step", steps, False), ("render",)]
    assert result["observation"] == final
    assert result["reward"] == 0.25 and result["info"] == {"source": "physics"}
    assert not result["terminated"] and not result["truncated"]
    assert s._session_last_obs[meta["_sid"]][s._obs_key(meta)] == final


@pytest.mark.parametrize("flag", ["terminated", "truncated", "error"])
def test_terminal_or_failed_step_is_not_overwritten_by_render(setup, flag):
    meta, step_result, _, calls = setup
    step_result[flag] = "step failed" if flag == "error" else True
    step_result["observation"]["cameras"] = [{"frame_id": "exact-terminal-image"}]
    result = s._step_gripper_with_final_observation(meta, [1.0], num_steps=60)
    assert calls == [("step", 60, False)]
    assert result is step_result
    assert result["observation"]["cameras"][0]["frame_id"] == "exact-terminal-image"
    assert result["reward"] == 0.25


@pytest.mark.parametrize("raises", [False, True])
def test_final_render_failure_keeps_actuation_receipt_and_reports_error(setup, monkeypatch, raises):
    meta, _, _, calls = setup
    def failed(meta):
        calls.append(("render",))
        if raises:
            raise TimeoutError("fixture render timeout")
        return {"error": "fixture render failure"}
    monkeypatch.setattr(s, "_proxy_render", failed)
    result = s._step_gripper_with_final_observation(meta, [1.0], num_steps=60)
    assert calls == [("step", 60, False), ("render",)]
    assert result["reward"] == 0.25
    assert result["error"].startswith("Post-actuation render failed:")
    assert not s._session_last_obs


def test_mink_gripper_uses_checked_worker_and_reports_actual_interrupted_steps(setup, monkeypatch):
    meta,_,_,calls=setup
    meta['control_spec']={'controller':{'goal_executor':'openeta.worker_mink_goal.v1'}}
    grant={'contact_kind':'articulated_fixture','target_geom_name':'handle'}
    sent=[]
    def worker(meta,body):
        sent.append(body)
        return {'steps_executed':2,'stop_reason':'collision_detected',
                'gripper_horizon_completed':False,'observation':{'cameras':[{'frame_id':'fresh'}]}}
    monkeypatch.setattr(s,'_proxy_gripper_goal',worker)
    r=s._step_gripper_with_final_observation(meta,[0]*7+[1],num_steps=60,contact_authorization=grant)
    assert not calls and r['ok'] is False
    assert sent[0]['contact_authorization']==grant
    receipt=s._gripper_actuation_receipt(r,command='close',steps_executed=60)
    assert receipt['steps_executed']==2 and receipt['horizon_completed'] is False
