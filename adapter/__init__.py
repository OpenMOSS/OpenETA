"""OpenETA simulator-agent adapter layer."""

# ── Protocol data classes ───────────────────────────────────────────
from adapter.agent import AgentAdapter
from adapter.bridge import AgentSimBridge, EpisodeRunResult
from adapter.protocol import CameraFrame, EnvAction, EnvObservation, RobotState, StepResult
from adapter.protocol import JsonDict, from_json, to_json
from adapter.sim import SimulatorAdapter

# ── Trigger gym registration when simulator extras are installed ─────
try:
    import sim.env_registry  # noqa: F401 — side-effect: populates gym registry
except ModuleNotFoundError as exc:
    if exc.name != "gymnasium":
        raise

__all__ = [
    "AgentAdapter",
    "AgentSimBridge",
    "CameraFrame",
    "EnvAction",
    "EnvObservation",
    "EpisodeRunResult",
    "from_json",
    "JsonDict",
    "RobotState",
    "SimulatorAdapter",
    "StepResult",
    "to_json",
]
