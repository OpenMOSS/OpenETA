"""Bridge orchestration between simulator and agent adapters."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adapter.agent import AgentAdapter
from adapter.protocol import EnvObservation, StepResult
from adapter.sim import SimulatorAdapter


@dataclass(slots=True)
class EpisodeRunResult:
    """Compact outcome from :meth:`AgentSimBridge.run_episode`."""

    task: str
    status: str
    success: bool | None
    turns: int
    simulator_steps: int
    final_observation: EnvObservation
    final_step_result: StepResult | None = None
    episode_dir: Path | None = None


class AgentSimBridge:
    """Runs a simple synchronous simulator-agent loop."""

    def __init__(self, *, simulator: SimulatorAdapter, agent: AgentAdapter) -> None:
        self.simulator = simulator
        self.agent = agent

    def run_once(self, *, task: str, seed: int | None = None) -> StepResult:
        observation = self.simulator.reset(task=task, seed=seed)
        self.agent.start_session(task=task, metadata={"seed": seed})
        action = self.agent.act(observation)
        result = self.simulator.step(action)
        self.agent.update_memory(
            {
                "type": "step_result",
                "reward": result.reward,
                "terminated": result.terminated,
                "truncated": result.truncated,
                "info": result.info,
            }
        )
        return result

    @staticmethod
    def _response_name(action: Any) -> str:
        if getattr(action, "action_type", "") != "response":
            return ""
        command = getattr(action, "command", {})
        request = command.get("request", {}) if isinstance(command, dict) else {}
        name = str(request.get("name", "")) if isinstance(request, dict) else ""
        return name or "response"

    @staticmethod
    def _log_fields(action: Any, result: StepResult | None) -> dict[str, dict]:
        command = getattr(action, "command", {})
        if not isinstance(command, dict):
            command = {}
        request = command.get("request", {})
        safety_checks = command.get("safety_checks", [])
        metadata = command.get("metadata", {})
        checker_results = (
            metadata.get("checker_results", {}) if isinstance(metadata, dict) else {}
        )
        failure = checker_results
        if result is not None and isinstance(result.info, dict):
            failure = result.info.get("failure_verdict", failure)
        return {
            "plan": request if isinstance(request, dict) else {},
            "safety_verdict": {
                "checks": safety_checks if isinstance(safety_checks, list) else [],
                "status": command.get("status", ""),
            },
            "failure_verdict": failure if isinstance(failure, dict) else {},
        }

    def run_episode(
        self,
        *,
        task: str,
        seed: int | None = None,
        max_steps: int = 100,
        logger: Any | None = None,
        environment: str = "",
        episode_id: str | None = None,
        metadata: dict | None = None,
    ) -> EpisodeRunResult:
        """Run a synchronous closed loop until response, termination, or limit.

        ``response`` actions end the bridge turn and are never forwarded to a
        simulator Box action space.  Non-response actions execute exactly one
        simulator step and the fresh observation is used for the next turn.
        """
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")

        observation = self.simulator.reset(task=task, seed=seed)
        self.agent.start_session(task=task, metadata={"seed": seed, **(metadata or {})})
        episode_dir: Path | None = None
        if logger is not None:
            episode_dir = logger.start_episode(
                task=task,
                environment=environment or type(self.simulator).__name__,
                seed=seed,
                episode_id=episode_id,
                metadata=metadata,
            )

        status = "max_steps"
        success: bool | None = False
        final_result: StepResult | None = None
        simulator_steps = 0
        turns = 0
        try:
            for turn_index in range(max_steps):
                turns = turn_index + 1
                plan_started = time.perf_counter()
                action = self.agent.act(observation)
                plan_ms = (time.perf_counter() - plan_started) * 1000.0
                response_name = self._response_name(action)

                step_result: StepResult | None = None
                exec_ms = 0.0
                if not response_name:
                    exec_started = time.perf_counter()
                    step_result = self.simulator.step(action)
                    exec_ms = (time.perf_counter() - exec_started) * 1000.0
                    simulator_steps += 1
                    final_result = step_result
                    self.agent.update_memory(
                        {
                            "type": "step_result",
                            "reward": step_result.reward,
                            "terminated": step_result.terminated,
                            "truncated": step_result.truncated,
                            "info": step_result.info,
                        }
                    )

                if logger is not None:
                    fields = self._log_fields(action, step_result)
                    logger.log_step(
                        step_index=turn_index,
                        observation=observation,
                        action=action,
                        result=step_result,
                        latency_ms={"plan": plan_ms, "exec": exec_ms},
                        **fields,
                    )

                if response_name:
                    status = {
                        "task_complete": "success",
                        "ask_human": "need_human",
                    }.get(response_name, "response")
                    success = True if response_name == "task_complete" else None
                    break

                assert step_result is not None
                observation = step_result.observation
                if step_result.terminated or step_result.truncated:
                    status = "terminated" if step_result.terminated else "truncated"
                    if isinstance(step_result.info, dict) and "success" in step_result.info:
                        success = bool(step_result.info["success"])
                    else:
                        success = None
                    break

            if logger is not None:
                episode_dir = logger.finish_episode(
                    status=status,
                    success=success,
                    final_observation=observation,
                    metadata={
                        "turns": turns,
                        "simulator_steps": simulator_steps,
                        "completion_source": (
                            "agent_response" if status in {"success", "need_human", "response"}
                            else "simulator"
                        ),
                    },
                )
        except BaseException as exc:
            if logger is not None and getattr(logger, "active", False):
                logger.abort_episode(exc)
            raise

        return EpisodeRunResult(
            task=task,
            status=status,
            success=success,
            turns=turns,
            simulator_steps=simulator_steps,
            final_observation=observation,
            final_step_result=final_result,
            episode_dir=episode_dir,
        )

    def close(self) -> None:
        """Release both adapter endpoints, even if one close operation fails."""
        try:
            self.agent.close()
        finally:
            self.simulator.close()
