"""Pure R1.2 contracts for tactile-conditioned branching qualification."""

from __future__ import annotations

import copy
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from sim.envs.univtac.contract import UniVTACContractError

TASKS = ("lift_bottle", "insert_hole")
SEEDS = (1_000_000, 1_000_001, 1_000_002)
MAX_SIMULATOR_EPISODES = 8
TASK_INSTRUCTIONS = {
    "lift_bottle": "Grasp the bottle and lift it vertically, keeping its final base near the wall.",
    "insert_hole": "Insert the peg into the hole.",
}
LIFT_CLASSES = ("direct_release", "correct_pose_then_release")
INSERT_CLASSES = (
    "negative_x_correction",
    "near_zero_x_correction",
    "positive_x_correction",
)


def validate_branching_config(config: Mapping[str, Any]) -> dict[str, Any]:
    required = {
        "tasks",
        "seeds",
        "task_config",
        "mode",
        "device",
        "task_instructions",
        "save_host_only",
        "strict_two_tactile_sensors",
        "timeout_seconds",
        "max_simulator_episodes",
        "source_commit",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise UniVTACContractError(f"branching config is missing: {missing}")
    expected = {
        "tasks": list(TASKS),
        "seeds": list(SEEDS),
        "task_config": "demo",
        "mode": "collect",
        "device": "cuda:0",
        "task_instructions": TASK_INSTRUCTIONS,
        "save_host_only": True,
        "strict_two_tactile_sensors": True,
        "max_simulator_episodes": MAX_SIMULATOR_EPISODES,
        "source_commit": "371fac67917307026be8f00869fcc1b61c623a9f",
    }
    for key, value in expected.items():
        if config[key] != value:
            raise UniVTACContractError(
                f"branching protocol requires {key}={value!r}, got {config[key]!r}"
            )
    timeout = float(config["timeout_seconds"])
    if timeout <= 0:
        raise UniVTACContractError("timeout_seconds must be positive")
    validated = copy.deepcopy(dict(config))
    validated["timeout_seconds"] = timeout
    return validated


def classify_insert_correction(x_move: float) -> str:
    if x_move < -0.001:
        return "negative_x_correction"
    if x_move > 0.001:
        return "positive_x_correction"
    return "near_zero_x_correction"


def classify_lift_branch(check_mid_success: bool) -> str:
    return "direct_release" if check_mid_success else "correct_pose_then_release"


def expert_segment_name(
    *, task: str, move_index: int, lift_mid_success: bool | None = None
) -> str:
    if move_index < 1:
        raise UniVTACContractError("move index is one-based")
    if task == "insert_hole":
        names = ("first_downward", "corrective_xz", "final_insert")
        return names[move_index - 1] if move_index <= len(names) else f"unexpected_move_{move_index}"
    if task != "lift_bottle":
        raise UniVTACContractError(f"unsupported branching task: {task}")
    if move_index == 1:
        return "close_gripper"
    if 2 <= move_index <= 5:
        return f"gripper_rotate_{move_index - 1}"
    if lift_mid_success is True:
        return "open_gripper" if move_index == 6 else f"unexpected_move_{move_index}"
    if lift_mid_success is False:
        names = {6: "corrective_tilt", 7: "corrective_translation", 8: "open_gripper"}
        return names.get(move_index, f"unexpected_move_{move_index}")
    return f"post_rotate_move_{move_index}"


def qualify_task(task: str, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if task not in TASKS:
        raise UniVTACContractError(f"unsupported branching task: {task}")
    if [int(row["seed"]) for row in rows] != list(SEEDS):
        raise UniVTACContractError("task results must contain the three fixed seeds in order")
    successes = sum(bool(row.get("expert_episode_success")) for row in rows)
    classes = [str(row["decision_class"]) for row in rows if row.get("decision_class")]
    distribution = dict(sorted(Counter(classes).items()))
    evidence_complete = all(
        bool(row.get("decision_snapshot_complete"))
        and bool(row.get("action_trace_complete"))
        for row in rows
    )
    candidate = successes >= 2 and len(distribution) >= 2 and evidence_complete
    return {
        "task": task,
        "native_expert_success_count": successes,
        "native_expert_success_rate": successes / len(SEEDS),
        "decision_class_count": len(distribution),
        "decision_class_distribution": distribution,
        "decision_evidence_complete": evidence_complete,
        "branching_candidate": candidate,
        "results": [copy.deepcopy(dict(row)) for row in rows],
    }


def select_counterfactual(
    task_summaries: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any] | None:
    insert = task_summaries.get("insert_hole", {})
    if insert.get("branching_candidate"):
        for row in insert.get("results", []):
            if abs(float(row.get("native_x_move", 0.0))) > 0.001:
                return {"task": "insert_hole", "seed": int(row["seed"])}
        raise UniVTACContractError("qualified Insert Hole has no non-zero correction seed")
    lift = task_summaries.get("lift_bottle", {})
    if lift.get("branching_candidate"):
        for row in lift.get("results", []):
            if row.get("check_mid_success") is False:
                return {"task": "lift_bottle", "seed": int(row["seed"])}
        raise UniVTACContractError("qualified Lift Bottle has no corrective branch seed")
    return None


def validate_counterfactual_pair(
    correct: Mapping[str, Any], wrong: Mapping[str, Any]
) -> None:
    if correct.get("condition") != "correct" or wrong.get("condition") != "wrong":
        raise UniVTACContractError("counterfactual pair must be correct then wrong")
    for key in ("task", "seed"):
        if correct.get(key) != wrong.get(key):
            raise UniVTACContractError(f"counterfactual pair differs in {key}")
    task = str(correct.get("task"))
    if task == "insert_hole":
        cx = float(correct["applied_x_move"])
        wx = float(wrong["applied_x_move"])
        cnx = float(correct["native_x_move"])
        wnx = float(wrong["native_x_move"])
        if cx != cnx or wx != -wnx:
            raise UniVTACContractError(
                "Insert Hole wrong continuation must only flip its native x sign"
            )
        if float(correct["applied_z_move"]) != float(correct["native_z_move"]):
            raise UniVTACContractError("Insert Hole correct continuation changed native z")
        if float(wrong["applied_z_move"]) != float(wrong["native_z_move"]):
            raise UniVTACContractError("Insert Hole wrong continuation changed native z")
        for key in ("final_insert_z", "corrective_time_dilation", "final_time_dilation"):
            if correct.get(key) != wrong.get(key):
                raise UniVTACContractError(f"Insert Hole counterfactual changed {key}")
    elif task == "lift_bottle":
        if correct.get("corrective_moves_executed") is not True:
            raise UniVTACContractError("Lift Bottle correct continuation must correct pose")
        if wrong.get("corrective_moves_executed") is not False:
            raise UniVTACContractError("Lift Bottle wrong continuation must skip correction")
    else:
        raise UniVTACContractError(f"unsupported counterfactual task: {task}")


def select_tactile_icl_task(
    candidate: Mapping[str, Any] | None,
    correct: Mapping[str, Any] | None,
    wrong: Mapping[str, Any] | None,
) -> tuple[str, str]:
    if candidate is None:
        return "none", "no_branching_candidate"
    if correct is None or wrong is None:
        raise UniVTACContractError("selected candidate requires a counterfactual pair")
    validate_counterfactual_pair(correct, wrong)
    if bool(correct.get("expert_episode_success")) and not bool(
        wrong.get("expert_episode_success")
    ):
        return str(candidate["task"]), "correct_succeeds_wrong_fails"
    return "none", "action_choice_not_consequential_under_current_intervention"


def invoke_passthrough(
    original: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
    *,
    before: Callable[[tuple[Any, ...], dict[str, Any]], None],
    after: Callable[[Any], None],
) -> Any:
    """Observe one call while preserving its arguments, return value, and exceptions."""

    copied_kwargs = dict(kwargs)
    before(args, copied_kwargs)
    returned = original(*args, **copied_kwargs)
    after(returned)
    return returned
