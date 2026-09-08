from __future__ import annotations

from copy import deepcopy

import pytest

from adapter.protocol import EnvAction, EnvObservation, RobotState, StepResult
from agent.evals.visual_history_rollout import _objective_success
from agent.evals.runner import _has_objective_success
from agent.runtime.episode import EpisodeResult, EpisodeStep
from agent.runtime.experiments import objective_success_evidence
from agent.runtime.parallel import classify_episode_result
from agent.runtime.planner import PlannerDecision, _validate_official_reward_completion
from agent.runtime.success_evidence import (
    episode_success_evidence,
    objective_reward_values,
    trusted_reward_receipt,
)


LIBERO = "openeta/libero_libero_spatial_task0-v0"
ROBOCASA = "openeta/robocasa_target_PnPCounterToCab-v0"
GENERIC = "openeta/shaped-reward-v0"


def _episode(env_id=LIBERO, *, reward=1.0, terminated=True):
    return {
        "session_id": "session-1",
        "metadata": {"execution_id": "execution-1", "env_id": env_id},
        "terminated": terminated,
        "truncated": False,
        "steps": [{
            "action": {"command": {}},
            "step_result": {
                "reward": reward, "terminated": terminated, "truncated": False,
                "info": {
                    "environment_receipt_trusted": True,
                    "official_reward": True,
                    "environment_receipt": {
                        "schema_version": "openeta.environment_receipt.v1",
                        "receipt_id": "receipt-1",
                        "agent_session_id": "session-1", "execution_id": "execution-1",
                        "reward_present": True, "reward": reward,
                        "terminated": terminated, "truncated": False,
                    },
                },
            },
        }],
    }


def _live(payload):
    observation = EnvObservation(task="test", robot=RobotState(), cameras=[])
    return EpisodeResult(
        task="test", session_id=payload["session_id"], metadata=payload["metadata"],
        terminated=payload["terminated"], truncated=payload["truncated"],
        steps=[EpisodeStep(
            turn_index=index, observation=observation,
            action=EnvAction(action_type="tool_call", command=step.get("action", {}).get("command", {})),
            step_result=StepResult(observation=observation, **step["step_result"]),
        ) for index, step in enumerate(payload["steps"])],
    )


def _assert_consumers(payload, expected):
    assert bool(episode_success_evidence(payload)) is expected
    assert _objective_success(payload) is expected
    assert bool(objective_success_evidence({"status": "success", "episode": payload})) is expected
    assert _has_objective_success({"status": "success", "episode": payload}) is expected
    assert (classify_episode_result(_live(payload)) == "success") is expected
    assert bool(episode_success_evidence(_live(payload).to_dict())) is expected


@pytest.mark.parametrize("env_id", [LIBERO, ROBOCASA])
def test_known_binary_adapters_have_consistent_success_consumers(env_id):
    episode = _episode(env_id)
    _assert_consumers(episode, True)
    assert objective_reward_values(episode) == [1.0]


@pytest.mark.parametrize("reward", [0.1, 1.0, 20.0, True, "1", float("nan"), float("inf")])
@pytest.mark.parametrize("official", [False, True])
def test_positive_or_malformed_reward_alone_never_proves_generic_success(reward, official):
    episode = _episode(GENERIC, reward=reward)
    if not official:
        episode["steps"][0]["step_result"]["info"] = {}
    _assert_consumers(episode, False)
    assert objective_reward_values(episode) == []


@pytest.mark.parametrize("reward", [0, 0.5, 2, True, "1", float("inf"), float("nan"), 10 ** 1000])
def test_binary_success_requires_finite_exact_numeric_one(reward):
    _assert_consumers(_episode(reward=reward), False)


@pytest.mark.parametrize("env_id", [LIBERO, ROBOCASA])
def test_binary_success_also_requires_terminal_receipt(env_id):
    _assert_consumers(_episode(env_id, terminated=False), False)


@pytest.mark.parametrize("key,value", [
    ("execution_id", "previous-run"), ("agent_session_id", "other-session"),
    ("schema_version", "unknown"), ("reward_present", False), ("reward", 0.5),
    ("terminated", False), ("terminated", 1), ("truncated", True),
])
def test_receipt_binding_and_projection_must_match(key, value):
    episode = _episode()
    episode["steps"][0]["step_result"]["info"]["environment_receipt"][key] = value
    _assert_consumers(episode, False)


@pytest.mark.parametrize("key", ["environment_receipt_trusted", "official_reward"])
def test_missing_or_rejected_trust_cannot_be_replaced_by_info_success(key):
    episode = _episode()
    info = episode["steps"][0]["step_result"]["info"]
    info[key] = False
    info["task_success"] = True
    _assert_consumers(episode, False)


def test_receipts_need_expected_execution_and_session_not_just_self_consistency():
    result = _episode()["steps"][0]["step_result"]
    assert trusted_reward_receipt(result, execution_id="", session_id="session-1") is None
    assert trusted_reward_receipt(result, execution_id="execution-1") is None


@pytest.mark.parametrize("failure", ["budget", "truncation", "step_truncation", "failed_completion", "human_wait"])
def test_terminal_failure_or_wait_overrides_prior_success(failure):
    episode = _episode()
    if failure == "budget":
        episode["metadata"]["failure_reason"] = {"code": "max_tool_calls"}
    elif failure == "truncation":
        episode["truncated"] = True
    elif failure == "step_truncation":
        episode["steps"][0]["step_result"]["truncated"] = True
    elif failure == "failed_completion":
        episode["steps"][0]["action"]["command"]["request"] = {
            "kind": "response", "name": "task_complete", "parameters": {"success": False},
        }
    else:
        episode["metadata"]["waiting_for_human"] = True
    _assert_consumers(episode, False)
    assert objective_reward_values(episode) == []


def test_latest_explicit_outcome_wins_but_false_intermediate_state_can_recover():
    episode = _episode(GENERIC, reward=0.0, terminated=False)
    step = episode["steps"][0]
    step["step_result"]["info"] = {"task_success": False}
    later = deepcopy(step)
    later["step_result"]["info"] = {"task_success": True}
    episode["steps"].append(later)
    _assert_consumers(episode, True)
    episode["steps"].append(deepcopy(step))
    _assert_consumers(episode, False)


def test_conflicting_flags_do_not_accept_terminal_reward():
    episode = _episode()
    episode["steps"][0]["step_result"]["info"].update(task_success=True, checker_success=False)
    _assert_consumers(episode, False)


@pytest.mark.parametrize("official_required", [False, True])
def test_generic_checker_success_is_not_conditioned_on_positive_reward(official_required):
    episode = _episode(GENERIC, reward=0.0)
    episode["metadata"]["require_official_reward"] = official_required
    episode["steps"][0]["step_result"]["info"]["checker_success"] = True
    _assert_consumers(episode, True)
    # Current playbook schema still requires official positive reward evidence.
    assert objective_reward_values(episode) == []


def test_runtime_completion_fallback_is_not_objective_success():
    episode = _episode(GENERIC)
    episode["steps"][0]["step_result"]["info"] = {}
    episode["metadata"]["stop_reason"] = "task_complete"
    assert classify_episode_result(_live(episode)) == "success"
    assert not episode_success_evidence(episode)
    assert not _objective_success(episode)
    assert not objective_reward_values(episode)
    episode["metadata"]["require_official_reward"] = True
    assert classify_episode_result(_live(episode)) == "fail"


@pytest.mark.parametrize("env_id", [LIBERO, ROBOCASA, GENERIC])
def test_planner_completion_gate_uses_same_execution_bound_verdict(env_id):
    episode = _episode(env_id)
    episode["metadata"].update(source="ParallelEpisodeHarness", require_official_reward=True)
    receipt = episode["steps"][0]["step_result"]
    if env_id == GENERIC:
        receipt["info"]["checker_success"] = True
    context = {"memory": {"session_id": episode["session_id"], "metadata": episode["metadata"]},
               "latest_environment_receipt": receipt}
    decision = PlannerDecision(action_type="response", action="task_complete")
    assert not _validate_official_reward_completion(decision, tool_context=context)
    context["memory"]["metadata"]["execution_id"] = "new-execution-same-session"
    assert _validate_official_reward_completion(decision, tool_context=context)


def test_failed_batch_outcome_cannot_seed_experiment_candidates():
    assert not objective_success_evidence({"status": "fail", "episode": _episode()})


@pytest.mark.parametrize("verified,status,expected", [
    (True, "success", 1), (False, "success", 0), (True, "fail", 0),
])
def test_calibration_batch_success_rates_use_objective_evidence(verified, status, expected):
    from agent.runtime.calibration import _collect_parallel_batch_evidence

    episode = _episode()
    episode["metadata"]["calibration_profile_sha256"] = "profile-hash"
    if not verified:
        episode["steps"][0]["step_result"]["info"] = {}
    counts = {"canary": {"attempts": 0, "successes": 0, "assisted": 0}}
    excluded = []
    _collect_parallel_batch_evidence(
        {"outcomes": [{"status": status, "episode": episode}]},
        split="canary", expected_profile_sha256="profile-hash",
        split_counts=counts, excluded=excluded,
    )
    assert counts["canary"] == {"attempts": 1, "successes": expected, "assisted": 0}
    assert not excluded


@pytest.mark.parametrize("verified,flag,verdict", [
    (False, None, "UNKNOWN"), (False, False, "FAIL"), (True, None, "PASS"),
])
def test_memory_ledger_does_not_present_positive_reward_as_task_pass(verified, flag, verdict):
    from agent.runtime.memory import AgentMemory

    episode = _episode(LIBERO if verified else GENERIC)
    result = episode["steps"][0]["step_result"]
    if not verified:
        result["info"] = {} if flag is None else {"task_success": flag}
    memory = AgentMemory()
    memory.start_session(task="test", session_id=episode["session_id"], metadata=episode["metadata"])
    memory.record_environment_receipt(**result)
    assert memory.transition_ledger()[-1]["verdict"] == verdict


def test_reused_session_binds_planner_to_current_execution():
    from agent.runtime.episode import DummyEpisodeEnvironment, OpenEtaEpisodeRunner
    from agent.runtime.planner import BasePlanner
    from agent.runtime.runtime import OpenEtaAgentRuntime

    seen = []

    class CapturePlanner(BasePlanner):
        def plan(self, observation, *, memory, tools, skills):
            seen.append(dict(memory.metadata))
            return PlannerDecision(action_type="response", action="talk")

    runtime = OpenEtaAgentRuntime(planner=CapturePlanner(), rollout_enabled=False)
    runtime.start_session(task="old task", metadata={"execution_id": "old-run"})
    runner = OpenEtaEpisodeRunner(runtime=runtime, environment=DummyEpisodeEnvironment(max_steps=1))
    first = runner.run(task="first", max_turns=1, metadata={"env_id": GENERIC})
    second = runner.run(task="second", max_turns=1, metadata={"env_id": ROBOCASA,
                                                            "require_official_reward": True})
    assert first.session_id == second.session_id
    assert seen[0]["execution_id"] == first.metadata["execution_id"]
    assert seen[1]["execution_id"] == second.metadata["execution_id"]
    assert seen[0]["execution_id"] != seen[1]["execution_id"]
    assert seen[1]["env_id"] == ROBOCASA
    assert second.metadata["require_official_reward"] is True


def test_official_completion_gate_allows_explicit_failure_report():
    decision = PlannerDecision(action_type="response", action="task_complete", parameters={"success": False})
    assert not _validate_official_reward_completion(decision, tool_context={
        "memory": {"metadata": {"source": "ParallelEpisodeHarness", "env_id": LIBERO}},
    })


def test_incomplete_binary_reward_is_not_a_failed_ledger_transition():
    from agent.runtime.memory import AgentMemory

    episode = _episode(reward=0, terminated=False)
    memory = AgentMemory()
    memory.start_session(task="test", session_id=episode["session_id"], metadata=episode["metadata"])
    memory.record_environment_receipt(**episode["steps"][0]["step_result"])
    assert memory.transition_ledger()[-1]["verdict"] == "UNKNOWN"
