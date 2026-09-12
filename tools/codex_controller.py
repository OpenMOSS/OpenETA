"""Controller selection for the isolated Codex experiment entry points."""

DEFAULT_CONTROLLER = "mink_joint_velocity"
CONTROLLER_IDS = {
    "mink_joint_velocity": "mink.robosuite_joint_velocity",
    "osc_pose": "robosuite.osc_pose",
}


def require_controller(capabilities, expected):
    if expected is None:
        return
    actual = (capabilities or {}).get("controller_id")
    if actual != CONTROLLER_IDS[expected]:
        raise RuntimeError(
            f"Expected controller {expected} ({CONTROLLER_IDS[expected]}), "
            f"but the simulator reported {actual or 'no controller identity'}. "
            "Start a dedicated simulator with the matching --controller option."
        )
