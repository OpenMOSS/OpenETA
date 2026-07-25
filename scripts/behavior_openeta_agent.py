#!/usr/bin/env python3
"""Run one bounded OpenETA agent attempt in BEHAVIOR and always save an MP4.

This is an experiment runner, not a task-specific oracle.  When no configured
LLM/VLM provider is available it uses OpenETA's deterministic planner backend
to exercise the same closed-loop runtime, MCP tools, memory, and simulator
episode interfaces as the interactive agent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from agent.backends.planner import StaticPlannerBackend  # noqa: E402
from agent.runtime.episode import OpenEtaEpisodeRunner  # noqa: E402
from agent.runtime.memory import AgentMemory  # noqa: E402
from agent.runtime.memory_store import JsonMemoryStore  # noqa: E402
from agent.runtime.planner import ToolCallingPlanner  # noqa: E402
from agent.runtime.runtime import OpenEtaAgentRuntime  # noqa: E402
from agent.tools.registry import (  # noqa: E402
    ToolExecutionContext,
    ToolResult,
    build_default_tool_registry,
    make_tool_result_details,
)
from agent.tools.sim_mcp import (  # noqa: E402
    SseSimulatorMcpTransport,
    SimulatorMcpEpisodeConfig,
    SimulatorMcpEpisodeEnvironment,
    SimulatorMcpToolProxyConfig,
    bind_simulator_mcp_tool_handlers,
)


def _sse_endpoint(url: str) -> str:
    """Use the simulator transport already exposed by OpenETA's agent API."""

    endpoint = str(url or "").strip().rstrip("/")
    if endpoint.endswith("/mcp"):
        return f"{endpoint[:-len('/mcp')]}/sse"
    if endpoint.endswith("/sse"):
        return endpoint
    return f"{endpoint}/sse"


class _RecordingSimulatorEnvironment(SimulatorMcpEpisodeEnvironment):
    """Script-local video recorder that leaves agent core untouched."""

    def __init__(
        self,
        *,
        video_path: Path,
        video_fps: int,
        video_frame_id: str,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.video_path = video_path
        self.video_fps = max(1, int(video_fps))
        self.video_frame_id = video_frame_id
        self.video_frames_dir = video_path.parent / f".{video_path.stem}_frames"
        self.video_frame_paths: list[Path] = []

    def _observation_from_payload(self, payload, *, metadata=None):
        observation = super()._observation_from_payload(payload, metadata=metadata)
        images = observation.metadata.get("image_artifacts", [])
        rgb_images = [image for image in images if image.get("kind") == "rgb"]
        selected = next(
            (image for image in rgb_images if image.get("frame_id") == self.video_frame_id),
            rgb_images[0] if rgb_images else None,
        )
        if selected is not None:
            source = Path(str(selected.get("path", "")))
            if source.is_file():
                self.video_frames_dir.mkdir(parents=True, exist_ok=True)
                destination = self.video_frames_dir / f"{len(self.video_frame_paths):06d}{source.suffix}"
                shutil.copy2(source, destination)
                self.video_frame_paths.append(destination)
        return observation

    def write_video(self) -> dict[str, Any]:
        if not self.video_frame_paths:
            return {
                "ok": False,
                "path": str(self.video_path.resolve()),
                "frame_count": 0,
                "error": "No RGB frames were recorded.",
            }
        import imageio.v2 as imageio

        self.video_path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(
            self.video_path,
            fps=self.video_fps,
            codec="libx264",
            macro_block_size=None,
        ) as writer:
            for frame_path in self.video_frame_paths:
                writer.append_data(imageio.imread(frame_path))
        return {
            "ok": True,
            "path": str(self.video_path.resolve()),
            "frame_count": len(self.video_frame_paths),
            "fps": self.video_fps,
            "frame_id": self.video_frame_id,
        }


def _scripted_decisions() -> list[dict[str, Any]]:
    """A conservative exploration attempt for the default R1Pro task."""
    return [
        {
            "kind": "tool_call",
            "name": "observe",
            "parameters": {"reason": "Inspect the initial scene before moving."},
            "reasoning": "Begin from a fresh visual observation.",
        },
        {
            "kind": "tool_call",
            "name": "gripper_control",
            "parameters": {"position": 1.0, "num_steps": 8},
            "reasoning": "Open the right gripper before searching for trash.",
        },
        {
            "kind": "tool_call",
            "name": "lower_body_control_policy",
            "parameters": {"forward": 0.35, "lateral": 0.0, "yaw": 0.0, "num_steps": 24},
            "reasoning": "Move forward to explore the visible corridor.",
        },
        {
            "kind": "tool_call",
            "name": "observe",
            "parameters": {"reason": "Re-observe after forward motion."},
            "reasoning": "World-changing actions require fresh visual feedback.",
        },
        {
            "kind": "tool_call",
            "name": "lower_body_control_policy",
            "parameters": {"forward": 0.0, "lateral": 0.0, "yaw": 0.30, "num_steps": 18},
            "reasoning": "Turn left to widen the search area.",
        },
        {
            "kind": "tool_call",
            "name": "observe",
            "parameters": {"reason": "Inspect the scene after turning."},
            "reasoning": "Check the new camera view before another motion.",
        },
        {
            "kind": "tool_call",
            "name": "lower_body_control_policy",
            "parameters": {"forward": 0.25, "lateral": 0.0, "yaw": 0.0, "num_steps": 18},
            "reasoning": "Approach the newly visible area cautiously.",
        },
        {
            "kind": "tool_call",
            "name": "observe",
            "parameters": {"reason": "Inspect the final approach view."},
            "reasoning": "Use a fresh frame before attempting a grasp action.",
        },
        {
            "kind": "tool_call",
            "name": "gripper_control",
            "parameters": {"position": 0.0, "num_steps": 10},
            "reasoning": "Close the right gripper as a bounded pickup attempt.",
        },
        {
            "kind": "tool_call",
            "name": "observe",
            "parameters": {"reason": "Verify whether the pickup changed task state."},
            "reasoning": "Inspect the outcome and native BEHAVIOR reward.",
        },
        {
            "kind": "response",
            "name": "talk",
            "parameters": {
                "message": "The bounded BEHAVIOR pickup attempt is complete; report the native checker result."
            },
            "reasoning": "Stop after the bounded exploration and grasp attempt.",
        },
    ]


def _response_summary(raw: dict[str, Any]) -> dict[str, Any]:
    observation = raw.get("observation", raw)
    return {
        "reward": float(raw.get("reward", 0.0) or 0.0),
        "terminated": bool(raw.get("terminated", False)),
        "truncated": bool(raw.get("truncated", False)),
        "info": raw.get("info", {}),
        "observation_keys": sorted(observation) if isinstance(observation, dict) else [],
    }


def _behavior_control_handler(transport, proxy_config: SimulatorMcpToolProxyConfig, *, gripper: bool):
    def handler(context: ToolExecutionContext) -> ToolResult:
        if not proxy_config.handle:
            raise RuntimeError("BEHAVIOR control called before environment reset")
        action = [0.0] * 23
        parameters = context.parameters
        if gripper:
            # Official R1Pro v3.9 action order: right_gripper is index 22.
            action[22] = -1.0 if float(parameters.get("position", 0.0)) >= 0.5 else 1.0
        else:
            # Official R1Pro v3.9 action order: base velocity is indices 0:3.
            action[0] = float(parameters.get("forward", 0.0))
            action[1] = float(parameters.get("lateral", 0.0))
            action[2] = float(parameters.get("yaw", 0.0))
        arguments = {
            "handle": proxy_config.handle,
            "session_id": proxy_config.session_id,
            "action": action,
            "num_steps": max(1, int(parameters.get("num_steps", 1))),
        }
        raw = transport.call_tool("step_env", arguments, timeout_s=proxy_config.timeout_s)
        success = isinstance(raw, dict) and "error" not in raw and raw.get("success") is not False
        summary = _response_summary(raw) if isinstance(raw, dict) else {"raw_type": type(raw).__name__}
        diagnostics = [] if success else [{"code": "behavior_step_failed", "message": str(raw)}]
        return ToolResult(
            success=success,
            content=(
                f"BEHAVIOR {'right gripper' if gripper else 'base'} control executed: "
                f"reward={summary.get('reward', 0.0)} terminated={summary.get('terminated', False)}"
            ),
            details=make_tool_result_details(
                context.spec,
                parameters,
                success=success,
                outputs={"response": summary, "raw_action": action},
                state_delta={
                    "reward": summary.get("reward", 0.0),
                    "terminated": summary.get("terminated", False),
                    "truncated": summary.get("truncated", False),
                },
                diagnostics=diagnostics,
            ),
        )

    return handler


def _write_diagnostic_video(path: Path, message: str, *, width: int, height: int, fps: int) -> None:
    """Guarantee an MP4 artifact even if simulator initialization fails."""
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[..., 0] = 64
    # Encode a visible diagnostic stripe without depending on a font package.
    frame[max(0, height // 2 - 4) : min(height, height // 2 + 4), :, :] = [220, 60, 60]
    with imageio.get_writer(path, fps=max(1, fps), codec="libx264", macro_block_size=None) as writer:
        for _ in range(max(2, fps)):
            writer.append_data(frame)
    (path.with_suffix(".diagnostic.txt")).write_text(message + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mcp-url", default="http://127.0.0.1:8765/mcp")
    parser.add_argument("--env-id", default="openeta/behavior_picking_up_trash-v0")
    parser.add_argument("--task", default="Pick up the trash in the scene.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--fps", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("outputs/behavior/agent_attempt"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    video_path = args.output / "openeta_agent_attempt.mp4"
    result_path = args.output / "result.json"
    started = time.monotonic()
    transport = None
    environment = None
    result: dict[str, Any] = {
        "ok": False,
        "task_success": False,
        "planner": {
            "provider": "static",
            "model": "fixture",
            "fallback_reason": "OpenETA LLM provider model/api_base/api_key are not configured.",
        },
        "env_id": args.env_id,
        "task": args.task,
        "seed": args.seed,
        "video": str(video_path.resolve()),
    }
    exit_code = 1
    try:
        transport = SseSimulatorMcpTransport(_sse_endpoint(args.mcp_url))
        proxy_config = SimulatorMcpToolProxyConfig(timeout_s=180.0)
        environment = _RecordingSimulatorEnvironment(
            transport=transport,
            config=SimulatorMcpEpisodeConfig(
                env_id=args.env_id,
                seed=args.seed,
                image_width=args.image_size,
                image_height=args.image_size,
                timeout_s=180.0,
            ),
            tool_proxy_config=proxy_config,
            video_path=video_path,
            video_fps=args.fps,
            video_frame_id="zed_head",
        )
        tools = build_default_tool_registry()
        bind_simulator_mcp_tool_handlers(
            tools,
            transport=transport,
            config=proxy_config,
            tool_names=("observe",),
            replace=True,
        )
        tools.bind_handler(
            "lower_body_control_policy",
            _behavior_control_handler(transport, proxy_config, gripper=False),
            replace=True,
        )
        tools.bind_handler(
            "gripper_control",
            _behavior_control_handler(transport, proxy_config, gripper=True),
            replace=True,
        )
        runtime = OpenEtaAgentRuntime(
            planner=ToolCallingPlanner(StaticPlannerBackend(_scripted_decisions())),
            tools=tools,
            memory=AgentMemory(store=JsonMemoryStore()),
        )
        runner = OpenEtaEpisodeRunner(runtime=runtime, environment=environment)
        episode = runner.run(
            task=args.task,
            max_turns=len(_scripted_decisions()),
            metadata={
                "source": "behavior_openeta_agent",
                "environment_mode": "simulator_mcp",
                "env_id": args.env_id,
                "planner_fallback": True,
            },
        )
        episode_dict = episode.to_dict()
        rewards = [float(step["step_result"].get("reward", 0.0)) for step in episode_dict["steps"]]
        native_terminated = any(
            bool(step["step_result"].get("terminated"))
            and step["step_result"].get("info", {}).get("termination_source") != "agent"
            for step in episode_dict["steps"]
        )
        result.update(
            {
                "ok": True,
                "task_success": bool(native_terminated or any(reward > 0.0 for reward in rewards)),
                "episode": episode_dict,
                "max_reward": max(rewards, default=0.0),
                "native_terminated": native_terminated,
                "session_id": episode.session_id,
            }
        )
        exit_code = 0
    except BaseException as exc:
        result["error_type"] = type(exc).__name__
        result["error"] = str(exc)
    finally:
        close_result: dict[str, Any] = {"ok": True, "skipped": True}
        if environment is not None:
            try:
                close_result = environment.close()
            except BaseException as exc:
                close_result = {"ok": False, "error_type": type(exc).__name__, "error": str(exc)}
            try:
                close_result["video"] = environment.write_video()
            except BaseException as exc:
                close_result["video"] = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        result["close"] = close_result
        if transport is not None and callable(getattr(transport, "close", None)):
            try:
                transport.close()
            except BaseException as exc:
                result["transport_close_error"] = f"{type(exc).__name__}: {exc}"
        if not video_path.is_file() or video_path.stat().st_size == 0:
            _write_diagnostic_video(
                video_path,
                result.get("error", "No simulator RGB frames were recorded."),
                width=args.image_size,
                height=args.image_size,
                fps=args.fps,
            )
            result["video_fallback"] = True
        result["elapsed_s"] = round(time.monotonic() - started, 3)
        result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
