"""Shared objective-success reduction for live and serialized episodes.

StepResult.info belongs to the host environment/checker boundary, not arbitrary
tool outputs. A reward is numeric feedback, not a generic task-success verdict.
"""

from __future__ import annotations

import math
import re

SUCCESS_FLAGS = ("task_success", "environment_success", "checker_success", "benchmark_success")


def _object(value) -> dict:
    return value if isinstance(value, dict) else {}


def finite_number(value) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def episode_evidence_payload(episode) -> dict:
    """Project live EpisodeResult without serializing camera arrays/artifacts."""
    if isinstance(episode, dict):
        return episode
    return {"session_id": episode.session_id, "metadata": episode.metadata,
            "terminated": episode.terminated, "truncated": episode.truncated,
            "steps": [{"action": {"command": {"request": step.action.command.get("request")}},
                       "observation": {"metadata": step.observation.metadata},
                       "step_result": {"reward": step.step_result.reward,
                                       "terminated": step.step_result.terminated,
                                       "truncated": step.step_result.truncated,
                                       "info": step.step_result.info}} for step in episode.steps]}


def episode_environment_id(episode: dict, env_id: str = "") -> str:
    if env_id:
        return env_id
    metadata = _object(episode.get("metadata"))
    if metadata.get("env_id"):
        return str(metadata["env_id"])
    for step in episode.get("steps") or []:
        observation = _object(_object(step).get("observation"))
        value = _object(observation.get("metadata")).get("env_id")
        if value:
            return str(value)
    return ""


def is_libero_environment(env_id: str) -> bool:
    return re.search(r"(?:^|[/_-])libero(?:$|[/_-])", env_id.lower()) is not None


def has_binary_terminal_reward(env_id: str) -> bool:
    # These are the concrete OpenETA adapters, not arbitrary similarly named
    # benchmarks. RoboCasaDirectEnv.step explicitly replaces shaped reward with
    # 1/0 from _check_success(); the training vector wrapper is NOT this API.
    return is_libero_environment(env_id) or re.fullmatch(
        r"openeta/robocasa_(?:pretrain|target)_[A-Za-z0-9_]+-v0", env_id
    ) is not None


def requires_official_reward(*, env_id: str, explicit=None) -> bool:
    return explicit if isinstance(explicit, bool) else has_binary_terminal_reward(env_id)


def episode_failed(episode: dict) -> bool:
    metadata = _object(episode.get("metadata"))
    if metadata.get("failure_reason") or episode.get("truncated") is True:
        return True
    steps = episode.get("steps") or []
    if any(_object(_object(step).get("step_result")).get("truncated") is True for step in steps):
        return True
    if steps:
        action = _object(_object(steps[-1]).get("action"))
        request = _object(_object(action.get("command")).get("request"))
        name = request.get("name", action.get("request_name"))
        parameters = _object(request.get("parameters", action.get("request_parameters")))
        if name == "task_complete" and parameters.get("success") is False:
            return True
    return False


def trusted_reward_receipt(result: dict, *, execution_id: str, session_id: str = "") -> dict | None:
    info = _object(result.get("info"))
    receipt = _object(info.get("environment_receipt"))
    reward = result.get("reward")
    if (not execution_id or not session_id or info.get("environment_receipt_trusted") is not True
            or info.get("official_reward") is not True
            or receipt.get("schema_version") != "openeta.environment_receipt.v1"
            or receipt.get("execution_id") != execution_id
            or receipt.get("agent_session_id") != session_id
            or receipt.get("reward_present") is not True
            or not finite_number(reward) or not finite_number(receipt.get("reward"))
            or receipt["reward"] != reward):
        return None
    for key in ("terminated", "truncated"):
        if (type(receipt.get(key)) is not bool or type(result.get(key, False)) is not bool
                or receipt[key] != result.get(key, False)):
            return None
    return receipt


def assess_episode_success(episode: dict, *, env_id: str = "", require_official_reward=None) -> dict:
    """Latest explicit outcome wins; intermediate false flags may precede success.

    A conflicting packet is not positive evidence. Resource failure, truncation,
    human wait, or an explicit failed completion cannot be erased by earlier reward.
    """
    metadata = _object(episode.get("metadata"))
    if episode_failed(episode):
        return {"evidence": [], "explicit_failure": True}
    if metadata.get("waiting_for_human") is True:
        return {"evidence": [], "explicit_failure": False}
    env_id = episode_environment_id(episode, env_id)
    required = requires_official_reward(
        env_id=env_id, explicit=(require_official_reward if require_official_reward is not None
                                 else metadata.get("require_official_reward")),
    )
    execution_id = str(metadata.get("execution_id") or "")
    session_id = str(episode.get("session_id") or "")
    evidence = []
    explicit_failure = False
    for index, step in enumerate(episode.get("steps") or []):
        result = _object(_object(step).get("step_result"))
        info = _object(result.get("info"))
        flags = {key: info[key] for key in SUCCESS_FLAGS if type(info.get(key)) is bool}
        receipt = trusted_reward_receipt(result, execution_id=execution_id, session_id=session_id)
        # Never treat rejected receipt projections as native adapter evidence.
        native_info = "environment_receipt_trusted" not in info and "environment_receipt" not in info
        if flags and (native_info or receipt is not None):
            if any(value is False for value in flags.values()):
                evidence = []
                explicit_failure = True
                continue
            if not has_binary_terminal_reward(env_id) and (not required or receipt is not None):
                evidence = [{"kind": key, "step": index, "value": True,
                             "source": "environment_receipt" if receipt else "step_result.info"}
                            for key in flags]
                explicit_failure = False
                continue
        # Binary terminal success is a known adapter contract. Neither
        # reward presence nor an arbitrary positive shaped reward establishes it.
        if has_binary_terminal_reward(env_id) and receipt is not None:
            if result["reward"] == 1 and receipt["terminated"] and not receipt["truncated"]:
                evidence = [{"kind": "official_terminal_reward", "step": index,
                             "value": 1.0, "source": "environment_receipt",
                             "receipt_id": receipt.get("receipt_id"), "execution_id": execution_id}]
                explicit_failure = False
            else:
                evidence = []
                explicit_failure = True
    return {"evidence": evidence, "explicit_failure": explicit_failure}


def episode_success_evidence(episode: dict, *, env_id: str = "", require_official_reward=None) -> list[dict]:
    return assess_episode_success(episode, env_id=env_id,
                                  require_official_reward=require_official_reward)["evidence"]


def objective_reward_values(episode: dict, *, env_id: str = "") -> list[float]:
    """Official positive rewards supporting a proven objective outcome only."""
    metadata = _object(episode.get("metadata"))
    rewards = []
    for index in sorted({item["step"] for item in episode_success_evidence(episode, env_id=env_id)}):
        result = episode["steps"][index]["step_result"]
        receipt = trusted_reward_receipt(result, execution_id=str(metadata.get("execution_id") or ""),
                                         session_id=str(episode.get("session_id") or ""))
        if receipt is not None and receipt["reward"] > 0:
            rewards.append(float(receipt["reward"]))
    return rewards
