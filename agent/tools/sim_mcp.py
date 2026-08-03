"""MCP-only simulator tool proxy for OpenETA runtime."""

from __future__ import annotations

import asyncio
import json
import math
import os
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from adapter.protocol import EnvAction, EnvObservation, JsonDict, RobotState, StepResult
from agent.runtime.image_artifacts import (
    DEFAULT_MCP_IMAGE_OUTPUT_ROOT,
    materialize_mcp_images,
)
from agent.runtime.response_artifacts import (
    DEFAULT_RESPONSE_ARTIFACT_OUTPUT_ROOT,
    build_motion_summary,
    build_observation_summary,
    build_response_reference,
    materialize_json_response,
)
from agent.runtime.text_artifacts import (
    DEFAULT_MAX_INLINE_TEXT_CHARS,
    DEFAULT_TEXT_ARTIFACT_OUTPUT_ROOT,
)
from agent.tools.registry import (
    ToolExecutionContext,
    ToolHandler,
    ToolRegistry,
    ToolResult,
    make_tool_result,
    make_tool_result_details,
)


DEFAULT_SIMULATOR_MCP_TOOL_NAMES = (
    "create_simulator_env",
    "close_simulator_env",
    "observe",
    "move_to",
    "follow_eef_trajectory",
    "gripper_control",
)

DEFAULT_SIMULATOR_IMAGE_WIDTH = 512
DEFAULT_SIMULATOR_IMAGE_HEIGHT = 512

SIMULATOR_CONTROL_MCP_TOOL_NAMES = (
    "move_to",
    "follow_eef_trajectory",
    "gripper_control",
)

DEFAULT_SIMULATOR_MCP_TOOL_MAP = {
    "create_simulator_env": "create_env",
    "close_simulator_env": "close_env",
    "observe": "render_env",
    "move_to": "move_to",
}

def mcp_server_url_from_endpoint(url: str) -> str:
    """Return the browser/API base URL for an MCP SSE endpoint."""

    endpoint = str(url or "").strip().rstrip("/")
    if endpoint.endswith("/sse"):
        return endpoint[: -len("/sse")]
    return endpoint


def mcp_server_url_from_transport(transport: object) -> str:
    """Best-effort server URL extraction from a configured MCP transport."""

    url = getattr(transport, "url", "")
    if not isinstance(url, str):
        return ""
    return mcp_server_url_from_endpoint(url)


def mcp_dashboard_url(server_url: str, session_id: object) -> str:
    """Return the simulator dashboard URL for a session when enough data exists."""

    session = str(session_id or "").strip()
    base = str(server_url or "").strip().rstrip("/")
    if not base or not session:
        return ""
    return f"{base}/session/{session}"


class SimulatorMcpTransport(Protocol):
    """Synchronous MCP tool transport used by simulator tool proxies."""

    def list_tools(self, *, timeout_s: float | None = None) -> JsonDict:
        """List simulator MCP tools and return compact JSON metadata."""
        ...

    def call_tool(
        self,
        name: str,
        arguments: JsonDict,
        *,
        timeout_s: float | None = None,
    ) -> JsonDict:
        """Call one simulator MCP tool and return its JSON payload."""
        ...


SimulatorMcpResponseCallback = Callable[[str, JsonDict, JsonDict], None]


@dataclass(slots=True)
class SimulatorMcpToolProxyConfig:
    """Configuration shared by simulator MCP tool proxy handlers."""

    session_id: str = ""
    handle: str = ""
    timeout_s: float = 120.0
    tool_name_map: Mapping[str, str] = field(default_factory=dict)
    materialize_images: bool = True
    image_output_root: str | Path = DEFAULT_MCP_IMAGE_OUTPUT_ROOT
    image_bundle_id: str = ""
    materialize_text: bool = True
    text_output_root: str | Path = DEFAULT_TEXT_ARTIFACT_OUTPUT_ROOT
    response_output_root: str | Path = DEFAULT_RESPONSE_ARTIFACT_OUTPUT_ROOT
    max_inline_text_chars: int = DEFAULT_MAX_INLINE_TEXT_CHARS
    lifecycle_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


@dataclass(slots=True)
class SimulatorMcpEpisodeConfig:
    """Configuration for one MCP-backed simulator episode."""

    env_id: str
    render_mode: str = "rgb_array"
    seed: int = 0
    image_width: int | None = DEFAULT_SIMULATOR_IMAGE_WIDTH
    image_height: int | None = DEFAULT_SIMULATOR_IMAGE_HEIGHT
    session_id: str = ""
    handle: str = ""
    timeout_s: float = 120.0
    image_output_root: str | Path = DEFAULT_MCP_IMAGE_OUTPUT_ROOT
    startup_attempts: int = 2
    startup_retry_delay_s: float = 0.5


class SimulatorMcpEpisodeEnvironment:
    """EpisodeEnvironment backed by a remote simulator MCP server.

    Control tools are executed by ``SimulatorMcpToolProxy`` during
    ``OpenEtaAgentRuntime.act()``. The episode environment owns env lifecycle
    and turns post-tool feedback into the next ``EnvObservation``.
    """

    def __init__(
        self,
        *,
        transport: SimulatorMcpTransport,
        config: SimulatorMcpEpisodeConfig,
        tool_proxy_config: SimulatorMcpToolProxyConfig | None = None,
    ) -> None:
        self.transport = transport
        self.config = config
        self.tool_proxy_config = tool_proxy_config or SimulatorMcpToolProxyConfig(
            session_id=config.session_id,
            handle=config.handle,
            timeout_s=config.timeout_s,
            image_output_root=config.image_output_root,
        )
        self.task = ""
        self.create_result: JsonDict = {}
        self.last_payload: JsonDict = {}
        self.startup_attempt_count = 0
        self._close_lock = threading.Lock()

    def reset(self, *, task: str, metadata: JsonDict | None = None) -> EnvObservation:
        self.task = task
        owns_environment = not self.config.handle
        attempts = max(1, self.config.startup_attempts if owns_environment else 1)
        for attempt in range(1, attempts + 1):
            self.startup_attempt_count = attempt
            try:
                if not self.config.handle:
                    self._create_env(task)
                payload = self._reset_env()
                break
            except RuntimeError as exc:
                if attempt >= attempts or not _is_transient_startup_error(exc):
                    raise
                if self.config.handle:
                    close_simulator_mcp_env(
                        self.transport,
                        handle=self.config.handle,
                        session_id=self.config.session_id,
                        timeout_s=min(self.config.timeout_s, 30.0),
                    )
                self.config.handle = ""
                self.tool_proxy_config.handle = ""
                if self.config.startup_retry_delay_s > 0:
                    time.sleep(self.config.startup_retry_delay_s)
        else:  # pragma: no cover - loop either returns payload or raises.
            raise RuntimeError("simulator MCP startup attempts exhausted")
        return self._observation_from_payload(payload, metadata=metadata)

    def _create_env(self, task: str) -> None:
        create_args: JsonDict = {
            "env_id": self.config.env_id,
            "render_mode": self.config.render_mode,
            "seed": self.config.seed,
            "task": task,
        }
        if self.config.image_width is not None:
            create_args["image_width"] = self.config.image_width
        if self.config.image_height is not None:
            create_args["image_height"] = self.config.image_height
        if self.config.session_id:
            create_args["session_id"] = self.config.session_id
        self.create_result = self.transport.call_tool(
            "create_env", create_args, timeout_s=self.config.timeout_s
        )
        _raise_if_mcp_error(self.create_result, tool_name="create_env")
        self.config.session_id = str(
            self.create_result.get("session_id") or self.config.session_id
        )
        self.config.handle = str(self.create_result.get("handle") or "")
        if not self.config.handle:
            raise RuntimeError("create_env did not return a simulator handle")
        self._sync_tool_proxy_config()

    def _reset_env(self) -> JsonDict:
        reset_args: JsonDict = {"handle": self.config.handle, "seed": self.config.seed}
        if self.config.session_id:
            reset_args["session_id"] = self.config.session_id
        payload = self.transport.call_tool(
            "reset_env", reset_args, timeout_s=self.config.timeout_s
        )
        _raise_if_mcp_error(payload, tool_name="reset_env")
        return payload

    def step(self, action: EnvAction) -> StepResult:
        render_args: JsonDict = {"handle": self.config.handle}
        if self.config.session_id:
            render_args["session_id"] = self.config.session_id
        payload = self.transport.call_tool(
            "render_env",
            render_args,
            timeout_s=self.config.timeout_s,
        )
        _raise_if_mcp_error(payload, tool_name="render_env")
        observation = self._observation_from_payload(
            payload,
            metadata={
                "previous_action": _summarize_mcp_action(action),
                "source": type(self).__name__,
            },
        )
        return StepResult(
            observation=observation,
            reward=_latest_action_reward(action, payload),
            terminated=_latest_action_flag(action, payload, "terminated"),
            truncated=_latest_action_flag(action, payload, "truncated"),
            info={
                "environment": type(self).__name__,
                "env_id": self.config.env_id,
                "session_id": self.config.session_id,
                "handle": self.config.handle,
                "previous_action": _summarize_mcp_action(action),
            },
        )

    def close(self) -> JsonDict:
        with self._close_lock:
            handle = self.config.handle
            session_id = self.config.session_id
            if not handle:
                return {"ok": True, "skipped": True}
            self.config.handle = ""
            self.tool_proxy_config.handle = ""
        return close_simulator_mcp_env(
            self.transport,
            handle=handle,
            session_id=session_id,
            timeout_s=min(self.config.timeout_s, 30.0),
        )

    def _sync_tool_proxy_config(self) -> None:
        self.tool_proxy_config.session_id = self.config.session_id
        self.tool_proxy_config.handle = self.config.handle
        self.tool_proxy_config.timeout_s = self.config.timeout_s
        self.tool_proxy_config.image_output_root = self.config.image_output_root
        if not self.tool_proxy_config.image_bundle_id:
            self.tool_proxy_config.image_bundle_id = (
                self.config.session_id or self.config.handle or self.config.env_id
            )

    def _observation_from_payload(
        self,
        payload: JsonDict,
        *,
        metadata: JsonDict | None = None,
    ) -> EnvObservation:
        bundle = materialize_mcp_images(
            payload,
            output_root=self.config.image_output_root,
            bundle_id=self.config.session_id or self.config.handle or None,
        )
        scrubbed = bundle.payload
        self.last_payload = scrubbed
        observation_payload = _extract_observation_payload(scrubbed)
        observation = EnvObservation.from_dict(observation_payload, task=self.task)
        merged_metadata: JsonDict = {
            **observation.metadata,
            "source": type(self).__name__,
            "env_id": self.config.env_id,
            "session_id": self.config.session_id,
            "handle": self.config.handle,
            "create_env": self.create_result,
            "startup_attempt_count": self.startup_attempt_count,
        }
        if bundle.images:
            merged_metadata["image_artifacts"] = [image.to_dict() for image in bundle.images]
        merged_metadata.update(dict(metadata or {}))
        observation.metadata = merged_metadata
        return observation


class SimulatorMcpToolProxy:
    """Tool handler that forwards OpenETA AgentTools to simulator MCP tools."""

    def __init__(
        self,
        *,
        transport: SimulatorMcpTransport,
        config: SimulatorMcpToolProxyConfig | None = None,
    ) -> None:
        self.transport = transport
        self.config = config or SimulatorMcpToolProxyConfig()
        self._artifact_sequence = 0

    def handler_for(self, tool_name: str) -> ToolHandler:
        def handler(context: ToolExecutionContext) -> ToolResult:
            return self.call(context, tool_name=tool_name)

        return handler

    def call(self, context: ToolExecutionContext, *, tool_name: str | None = None) -> ToolResult:
        agent_tool = tool_name or context.name
        try:
            mcp_tool, arguments = self._mcp_call(context, agent_tool=agent_tool)
        except Exception as exc:  # noqa: BLE001 - validation must stay structured.
            return ToolResult(
                False,
                content=f"Simulator MCP proxy could not build arguments for {agent_tool}: {exc}",
                details=make_tool_result_details(
                    context.spec,
                    context.parameters,
                    success=False,
                    diagnostics=[
                        {
                            "code": "simulator_mcp_argument_error",
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    ],
                ),
            )

        try:
            raw_response = self.transport.call_tool(
                mcp_tool,
                arguments,
                timeout_s=self.config.timeout_s,
            )
        except Exception as exc:  # noqa: BLE001 - tool failures must stay structured.
            return ToolResult(
                False,
                content=f"Simulator MCP tool failed: {mcp_tool}: {exc}",
                details=make_tool_result_details(
                    context.spec,
                    context.parameters,
                    success=False,
                    outputs={
                        "mcp": {
                            "tool": mcp_tool,
                            "agent_tool": agent_tool,
                            "session_id": arguments.get("session_id", ""),
                            "handle": arguments.get("handle", ""),
                        }
                    },
                    diagnostics=[
                        {
                            "code": "simulator_mcp_call_failed",
                            "error_type": type(exc).__name__,
                            "message": str(exc),
                        }
                    ],
                ),
            )

        success = _response_success(raw_response)
        normalized = self._normalize_response(raw_response, agent_tool=agent_tool, mcp_tool=mcp_tool)
        diagnostics = _response_diagnostics(raw_response) if not success else []
        return ToolResult(
            success,
            content=_response_content(
                normalized["outputs"]["response"],
                mcp_tool=mcp_tool,
                success=success,
            ),
            details=make_tool_result_details(
                context.spec,
                context.parameters,
                success=success,
                outputs=normalized["outputs"],
                artifacts=normalized["artifacts"],
                state_delta=normalized["state_delta"],
                diagnostics=diagnostics,
            ),
        )

    def _mcp_call(
        self,
        context: ToolExecutionContext,
        *,
        agent_tool: str,
    ) -> tuple[str, JsonDict]:
        if agent_tool in self.config.tool_name_map:
            return self.config.tool_name_map[agent_tool], self._with_session(
                dict(context.parameters)
            )
        if agent_tool == "observe":
            return self._mcp_tool_name(agent_tool), self._with_session({})
        if agent_tool == "move_to":
            return self._mcp_tool_name(agent_tool), self._move_to_arguments(context.parameters)
        if agent_tool == "follow_eef_trajectory":
            return self._mcp_tool_name(agent_tool), self._with_session(dict(context.parameters))
        if agent_tool == "gripper_control":
            return self._gripper_tool_name(context.parameters), self._with_session({})
        return self._mcp_tool_name(agent_tool), self._with_session(dict(context.parameters))

    def _mcp_tool_name(self, agent_tool: str) -> str:
        if agent_tool in self.config.tool_name_map:
            return self.config.tool_name_map[agent_tool]
        return DEFAULT_SIMULATOR_MCP_TOOL_MAP.get(agent_tool, agent_tool)

    def _with_session(self, arguments: JsonDict) -> JsonDict:
        if self.config.handle:
            arguments.setdefault("handle", self.config.handle)
        if self.config.session_id:
            arguments.setdefault("session_id", self.config.session_id)
        if not arguments.get("handle"):
            raise ValueError(
                "No active simulator MCP environment handle is bound. "
                "Create/reset a simulator environment before calling control tools."
            )
        return arguments

    def _move_to_arguments(self, parameters: JsonDict) -> JsonDict:
        x, y, z = _extract_xyz(parameters, tool_name="move_to")
        arguments: JsonDict = {"x": x, "y": y, "z": z}
        if "speed" in parameters:
            raise ValueError(
                "move_to `speed` is unsupported by the simulator MCP; "
                "use num_steps/tolerance or omit speed."
            )
        arguments.update(_extract_orientation_arguments(parameters, tool_name="move_to"))
        for key in ("handle", "session_id"):
            if key in parameters:
                arguments[key] = parameters[key]
        for key in ("num_steps", "tolerance", "ori_tolerance", "enable_collision_check"):
            if key in parameters:
                arguments[key] = parameters[key]
        return self._with_session(arguments)

    def _gripper_tool_name(self, parameters: JsonDict) -> str:
        position = parameters.get("position")
        if position is None:
            position = parameters.get("open")
        if position is None:
            raise ValueError("gripper_control requires `position` or `open`.")
        return "gripper_open" if float(position) >= 0.5 else "gripper_close"

    def _normalize_response(
        self,
        response: JsonDict,
        *,
        agent_tool: str,
        mcp_tool: str,
    ) -> JsonDict:
        payload = dict(response)
        bundle_id = self._next_artifact_bundle_id(mcp_tool)
        artifacts: list[JsonDict] = []
        if self.config.materialize_images:
            bundle = materialize_mcp_images(
                payload,
                output_root=self.config.image_output_root,
                bundle_id=bundle_id,
            )
            payload = bundle.payload
            artifacts = [image.to_dict() for image in bundle.images]
        payload = _with_anygrasp_camera_intrinsics(payload)
        response_artifact = materialize_json_response(
            payload,
            output_root=self.config.response_output_root,
            bundle_id=bundle_id,
            name=f"{mcp_tool}-response",
        )
        response_ref = build_response_reference(
            payload,
            response_artifact,
            image_artifacts=artifacts,
        )
        artifacts.append(response_artifact.to_dict())

        outputs: JsonDict = {
            "mcp": {
                "tool": mcp_tool,
                "agent_tool": agent_tool,
                "session_id": self.config.session_id,
                "handle": self.config.handle,
            },
            "response": response_ref,
        }
        for key in ("observation_summary", "motion_summary"):
            summary = response_ref.get(key)
            if isinstance(summary, dict):
                outputs[key] = summary
        return {
            "outputs": outputs,
            "artifacts": artifacts,
            "state_delta": _state_delta_from_response(payload),
        }

    def _next_artifact_bundle_id(self, mcp_tool: str) -> str:
        self._artifact_sequence += 1
        base = self.config.image_bundle_id or self.config.session_id or "simulator-mcp"
        return f"{base}-{self._artifact_sequence:04d}-{mcp_tool}"


class SimulatorEnvironmentCreator:
    """Create and reset one simulator environment through a stable AgentTool."""

    def __init__(
        self,
        *,
        transport: SimulatorMcpTransport,
        config: SimulatorMcpToolProxyConfig | None = None,
        response_callback: SimulatorMcpResponseCallback | None = None,
    ) -> None:
        self.transport = transport
        self.config = config or SimulatorMcpToolProxyConfig()
        self.response_callback = response_callback
        self.proxy = SimulatorMcpToolProxy(transport=transport, config=self.config)

    def handler(self, context: ToolExecutionContext) -> ToolResult:
        env_id = str(context.parameters.get("env_id") or "").strip()
        if not env_id:
            return self._failure(
                context,
                content="create_simulator_env requires a non-empty env_id.",
                diagnostics=[{"code": "missing_env_id"}],
            )
        with self.config.lifecycle_lock:
            active_handle = self.config.handle
        if active_handle:
            return self._failure(
                context,
                content=(
                    "A simulator environment is already active. Call "
                    "close_simulator_env before creating another one."
                ),
                diagnostics=[
                    {
                        "code": "simulator_environment_already_active",
                        "handle": active_handle,
                    }
                ],
            )

        try:
            create_args = self._create_arguments(context, env_id=env_id)
        except (TypeError, ValueError) as exc:
            return self._failure(
                context,
                content=f"create_simulator_env parameters are invalid: {exc}",
                diagnostics=[
                    {
                        "code": "simulator_mcp_argument_error",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                ],
            )
        try:
            create_response = self.transport.call_tool(
                "create_env",
                create_args,
                timeout_s=self.config.timeout_s,
            )
        except Exception as exc:  # noqa: BLE001 - transport errors stay structured.
            return self._transport_failure(context, "create_env", exc)

        if _context_execution_cancelled(context):
            abandoned_handle = str(create_response.get("handle") or "").strip()
            abandoned_session_id = str(
                create_response.get("session_id") or create_args.get("session_id") or ""
            ).strip()
            if abandoned_handle:
                self._close_abandoned_environment(
                    handle=abandoned_handle,
                    session_id=abandoned_session_id,
                )
            return self._failure(
                context,
                content="Simulator environment creation was cancelled and cleaned up.",
                diagnostics=[{"code": "execution_cancelled", "abandoned": True}],
            )

        create_normalized = self.proxy._normalize_response(  # noqa: SLF001
            create_response,
            agent_tool="create_simulator_env",
            mcp_tool="create_env",
        )
        create_ref = create_normalized["outputs"]["response"]
        self._notify("create_env", create_args, create_ref)
        if not _response_success(create_response):
            return self._failure(
                context,
                content=_response_content(create_ref, mcp_tool="create_env", success=False),
                outputs={
                    "mcp": create_normalized["outputs"]["mcp"],
                    "create_response": create_ref,
                },
                artifacts=create_normalized["artifacts"],
                diagnostics=_response_diagnostics(create_response),
            )

        handle = str(create_response.get("handle") or "").strip()
        session_id = str(
            create_response.get("session_id") or create_args.get("session_id") or ""
        ).strip()
        if not handle:
            return self._failure(
                context,
                content="Simulator create_env succeeded without returning a handle.",
                outputs={"create_response": create_ref},
                artifacts=create_normalized["artifacts"],
                diagnostics=[{"code": "create_env_missing_handle"}],
            )
        if _context_execution_cancelled(context):
            self._close_abandoned_environment(handle=handle, session_id=session_id)
            return self._failure(
                context,
                content="Simulator environment creation was cancelled and cleaned up.",
                diagnostics=[{"code": "execution_cancelled", "abandoned": True}],
            )

        with self.config.lifecycle_lock:
            self.config.handle = handle
            self.config.session_id = session_id
            self.config.image_bundle_id = session_id or handle
        reset_args: JsonDict = {"handle": handle, "seed": create_args["seed"]}
        if session_id:
            reset_args["session_id"] = session_id
        try:
            reset_response = self.transport.call_tool(
                "reset_env",
                reset_args,
                timeout_s=self.config.timeout_s,
            )
        except Exception as exc:  # noqa: BLE001 - transport errors stay structured.
            return self._transport_failure(
                context,
                "reset_env",
                exc,
                outputs={"create_response": create_ref},
                artifacts=create_normalized["artifacts"],
            )
        if _context_execution_cancelled(context):
            with self.config.lifecycle_lock:
                owns_handle = self.config.handle == handle
                if owns_handle:
                    self.config.handle = ""
            if owns_handle:
                self._close_abandoned_environment(handle=handle, session_id=session_id)
            return self._failure(
                context,
                content="Simulator environment reset was cancelled and cleaned up.",
                diagnostics=[{"code": "execution_cancelled", "abandoned": True}],
            )

        reset_normalized = self.proxy._normalize_response(  # noqa: SLF001
            reset_response,
            agent_tool="create_simulator_env",
            mcp_tool="reset_env",
        )
        reset_ref = reset_normalized["outputs"]["response"]
        self._notify("reset_env", reset_args, reset_ref)
        success = _response_success(reset_response)
        server_url = mcp_server_url_from_transport(self.transport)
        dashboard_url = mcp_dashboard_url(server_url, session_id)
        environment: JsonDict = {
            "env_id": env_id,
            "handle": handle,
            "session_id": session_id,
        }
        if server_url:
            environment["mcp_server_url"] = server_url
        if dashboard_url:
            environment["dashboard_url"] = dashboard_url
        outputs: JsonDict = {
            "mcp": {
                "tool": "create_env",
                "auto_reset_tool": "reset_env",
                "handle": handle,
                "session_id": session_id,
            },
            "environment": environment,
            "create_response": create_ref,
            "initial_observation": reset_ref,
        }
        for key in ("observation_summary",):
            summary = reset_normalized["outputs"].get(key)
            if isinstance(summary, dict):
                outputs[key] = summary
        return ToolResult(
            success,
            content=(
                "Simulator environment created and reset."
                if success
                else _response_content(reset_ref, mcp_tool="reset_env", success=False)
            ),
            details=make_tool_result_details(
                context.spec,
                context.parameters,
                success=success,
                outputs=outputs,
                artifacts=[
                    *create_normalized["artifacts"],
                    *reset_normalized["artifacts"],
                ],
                state_delta={
                    **reset_normalized["state_delta"],
                    "simulator_environment": environment,
                },
                diagnostics=[] if success else _response_diagnostics(reset_response),
            ),
        )

    def _create_arguments(
        self,
        context: ToolExecutionContext,
        *,
        env_id: str,
    ) -> JsonDict:
        parameters = context.parameters
        args: JsonDict = {
            "env_id": env_id,
            "render_mode": str(parameters.get("render_mode") or "rgb_array"),
            "seed": _required_integer(parameters.get("seed", 0), name="seed"),
            "image_width": _positive_integer(
                parameters.get("image_width") or DEFAULT_SIMULATOR_IMAGE_WIDTH,
                name="image_width",
            ),
            "image_height": _positive_integer(
                parameters.get("image_height") or DEFAULT_SIMULATOR_IMAGE_HEIGHT,
                name="image_height",
            ),
        }
        task = parameters.get("task")
        if not task and context.observation is not None:
            task = context.observation.task
        if isinstance(task, str) and task:
            args["task"] = task
        session_id = parameters.get("session_id") or self.config.session_id
        if isinstance(session_id, str) and session_id:
            args["session_id"] = session_id
        if "include_objects" in parameters:
            include_objects = parameters["include_objects"]
            if not isinstance(include_objects, bool):
                raise TypeError("include_objects must be a boolean")
            args["include_objects"] = include_objects
        return args

    def _notify(self, name: str, arguments: JsonDict, response: JsonDict) -> None:
        if self.response_callback is not None:
            self.response_callback(name, arguments, response)

    def _close_abandoned_environment(self, *, handle: str, session_id: str) -> None:
        result = close_simulator_mcp_env(
            self.transport,
            handle=handle,
            session_id=session_id,
            timeout_s=min(self.config.timeout_s, 30.0),
        )
        if _response_success(result):
            return
        with self.config.lifecycle_lock:
            if not self.config.handle:
                self.config.handle = handle
                self.config.session_id = session_id

    def _transport_failure(
        self,
        context: ToolExecutionContext,
        mcp_tool: str,
        exc: Exception,
        *,
        outputs: JsonDict | None = None,
        artifacts: list[JsonDict] | None = None,
    ) -> ToolResult:
        return self._failure(
            context,
            content=f"Simulator MCP tool failed: {mcp_tool}: {exc}",
            outputs=outputs,
            artifacts=artifacts,
            diagnostics=[
                {
                    "code": "simulator_mcp_call_failed",
                    "tool": mcp_tool,
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            ],
        )

    @staticmethod
    def _failure(
        context: ToolExecutionContext,
        *,
        content: str,
        outputs: JsonDict | None = None,
        artifacts: list[JsonDict] | None = None,
        diagnostics: list[JsonDict] | None = None,
    ) -> ToolResult:
        return ToolResult(
            False,
            content=content,
            details=make_tool_result_details(
                context.spec,
                context.parameters,
                success=False,
                outputs=outputs,
                artifacts=artifacts,
                diagnostics=diagnostics,
            ),
        )


class SimulatorEnvironmentCloser:
    """Close the one active simulator environment through a stable AgentTool."""

    def __init__(
        self,
        *,
        transport: SimulatorMcpTransport,
        config: SimulatorMcpToolProxyConfig,
        response_callback: SimulatorMcpResponseCallback | None = None,
    ) -> None:
        self.transport = transport
        self.config = config
        self.response_callback = response_callback

    def handler(self, context: ToolExecutionContext) -> ToolResult:
        with self.config.lifecycle_lock:
            handle = self.config.handle
            session_id = self.config.session_id
            if handle:
                self.config.handle = ""
        if not handle:
            return make_tool_result(
                context,
                success=True,
                content="No active simulator environment to close.",
                outputs={"closed": False, "skipped": True},
            )
        arguments: JsonDict = {"handle": handle}
        if session_id:
            arguments["session_id"] = session_id
        try:
            response = self.transport.call_tool(
                "close_env",
                arguments,
                timeout_s=min(self.config.timeout_s, 30.0),
            )
        except Exception as exc:  # noqa: BLE001 - lifecycle failures stay structured.
            with self.config.lifecycle_lock:
                if not self.config.handle:
                    self.config.handle = handle
            return make_tool_result(
                context,
                success=False,
                content=f"Simulator MCP tool failed: close_env: {exc}",
                outputs={"handle": handle, "session_id": session_id},
                diagnostics=[
                    {
                        "code": "simulator_mcp_call_failed",
                        "tool": "close_env",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                ],
            )
        success = _response_success(response)
        if not success:
            with self.config.lifecycle_lock:
                if not self.config.handle:
                    self.config.handle = handle
        elif self.response_callback is not None:
            self.response_callback("close_env", arguments, response)
        return make_tool_result(
            context,
            success=success,
            content=(
                "Simulator environment closed."
                if success
                else _response_content(response, mcp_tool="close_env", success=False)
            ),
            outputs={
                "closed": success,
                "environment": {"handle": handle, "session_id": session_id},
                "response": response,
            },
            state_delta={
                "simulator_environment": {
                    "handle": handle,
                    "session_id": session_id,
                    "status": "closed" if success else "close_failed",
                }
            },
            diagnostics=[] if success else _response_diagnostics(response),
        )

def _with_anygrasp_camera_intrinsics(payload: JsonDict) -> JsonDict:
    """Add metric depth scale to generic and legacy camera intrinsics."""

    enriched = json.loads(json.dumps(payload))
    _enrich_anygrasp_camera_intrinsics(enriched)
    return enriched if isinstance(enriched, dict) else dict(payload)


def _enrich_anygrasp_camera_intrinsics(value: Any) -> None:
    if isinstance(value, dict):
        _enrich_camera_dict(value)
        for item in value.values():
            _enrich_anygrasp_camera_intrinsics(item)
    elif isinstance(value, list):
        for item in value:
            _enrich_anygrasp_camera_intrinsics(item)


def _enrich_camera_dict(camera: JsonDict) -> None:
    intrinsics = camera.get("intrinsics")
    if not isinstance(intrinsics, dict):
        return
    has_camera_payload = any(
        isinstance(camera.get(key), str) and camera.get(key)
        for key in ("rgb_path", "depth_path", "image_path", "rgb_ref", "depth_ref")
    )
    if not has_camera_payload:
        return
    normalized_intrinsics = dict(intrinsics)
    scale = _camera_depth_scale(camera, intrinsics)
    if scale is not None:
        normalized_intrinsics["scale"] = scale
    camera["intrinsics"] = normalized_intrinsics
    camera.setdefault("anygrasp_intrinsics", dict(normalized_intrinsics))


def _camera_depth_scale(camera: JsonDict, intrinsics: JsonDict) -> float | None:
    for key in ("scale", "depth_scale"):
        parsed = _positive_float(intrinsics.get(key))
        if parsed is not None:
            return parsed
    for key in ("depth_scale", "scale"):
        parsed = _positive_float(camera.get(key))
        if parsed is not None:
            return parsed
    depth_path = camera.get("depth_path")
    if isinstance(depth_path, str) and depth_path.lower().endswith(".png"):
        return 1000.0
    return None


def _positive_float(value: Any) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _required_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _positive_integer(value: Any, *, name: str) -> int:
    parsed = _required_integer(value, name=name)
    if parsed <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return parsed


@dataclass(slots=True)
class StdioSimulatorMcpTransport:
    """Synchronous stdio MCP transport for local simulator-server launches."""

    command: str
    args: Sequence[str] = ()
    cwd: str | Path | None = None

    def list_tools(self, *, timeout_s: float | None = None) -> JsonDict:
        return asyncio.run(
            _list_stdio_mcp_tools(
                command=self.command,
                args=list(self.args),
                cwd=str(self.cwd) if self.cwd is not None else None,
                timeout_s=timeout_s,
            )
        )

    def call_tool(
        self,
        name: str,
        arguments: JsonDict,
        *,
        timeout_s: float | None = None,
    ) -> JsonDict:
        return asyncio.run(
            _call_stdio_mcp_tool(
                command=self.command,
                args=list(self.args),
                cwd=str(self.cwd) if self.cwd is not None else None,
                tool_name=name,
                arguments=arguments,
                timeout_s=timeout_s,
            )
        )


@dataclass(slots=True)
class SseSimulatorMcpTransport:
    """Synchronous SSE MCP transport for an already-running simulator server."""

    url: str = "http://localhost:8765/sse"

    def list_tools(self, *, timeout_s: float | None = None) -> JsonDict:
        with _temporary_no_proxy_for_url(self.url):
            return asyncio.run(
                _with_optional_timeout(
                    _list_sse_mcp_tools(
                        url=self.url,
                        timeout_s=timeout_s,
                    ),
                    timeout_s=timeout_s,
                )
            )

    def call_tool(
        self,
        name: str,
        arguments: JsonDict,
        *,
        timeout_s: float | None = None,
    ) -> JsonDict:
        with _temporary_no_proxy_for_url(self.url):
            return asyncio.run(
                _with_optional_timeout(
                    _call_sse_mcp_tool(
                        url=self.url,
                        tool_name=name,
                        arguments=arguments,
                        timeout_s=timeout_s,
                    ),
                    timeout_s=timeout_s,
                )
            )


def bind_simulator_mcp_tool_handlers(
    tools: ToolRegistry,
    *,
    transport: SimulatorMcpTransport,
    config: SimulatorMcpToolProxyConfig | None = None,
    tool_names: Sequence[str] = DEFAULT_SIMULATOR_MCP_TOOL_NAMES,
    response_callback: SimulatorMcpResponseCallback | None = None,
    replace: bool = False,
) -> ToolRegistry:
    """Bind simulator-owned AgentTools to MCP proxy handlers."""

    shared_config = config or SimulatorMcpToolProxyConfig()
    proxy = SimulatorMcpToolProxy(transport=transport, config=shared_config)
    creator = SimulatorEnvironmentCreator(
        transport=transport,
        config=shared_config,
        response_callback=response_callback,
    )
    closer = SimulatorEnvironmentCloser(
        transport=transport,
        config=shared_config,
        response_callback=response_callback,
    )
    for name in tool_names:
        tools.get(name)
        if tools.can_execute(name) and not replace:
            continue
        if name == "create_simulator_env":
            handler = creator.handler
        elif name == "close_simulator_env":
            handler = closer.handler
        else:
            handler = proxy.handler_for(name)
        tools.bind_handler(name, handler, replace=replace)
    return tools


def close_simulator_mcp_env(
    transport: SimulatorMcpTransport,
    *,
    handle: str,
    session_id: str = "",
    timeout_s: float | None = 30.0,
) -> JsonDict:
    """Best-effort cleanup for a remote simulator environment.

    Any code path that creates an MCP env for tests or smoke runs must call
    ``close_env`` in a ``finally`` block. This helper keeps cleanup failures
    structured so the original test failure is not masked by a secondary close
    exception.
    """

    arguments: JsonDict = {"handle": handle}
    if session_id:
        arguments["session_id"] = session_id
    try:
        result = transport.call_tool("close_env", arguments, timeout_s=timeout_s)
    except Exception as exc:  # noqa: BLE001 - cleanup must be best-effort.
        return {
            "ok": False,
            "error": str(exc),
            "error_type": type(exc).__name__,
            "handle": handle,
            "session_id": session_id,
        }
    if not isinstance(result, dict):
        return {
            "ok": False,
            "error": f"close_env returned {type(result).__name__}",
            "handle": handle,
            "session_id": session_id,
        }
    return result


async def _call_stdio_mcp_tool(
    *,
    command: str,
    args: list[str],
    cwd: str | None,
    tool_name: str,
    arguments: JsonDict,
    timeout_s: float | None,
) -> JsonDict:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=command, args=args, cwd=cwd)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                tool_name,
                arguments,
                read_timeout_seconds=_timeout_delta(timeout_s),
            )
    payload = _parse_mcp_tool_result(result)
    return payload


async def _list_stdio_mcp_tools(
    *,
    command: str,
    args: list[str],
    cwd: str | None,
    timeout_s: float | None,
) -> JsonDict:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(command=command, args=args, cwd=cwd)
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
    return _parse_mcp_tools_result(result)


async def _call_sse_mcp_tool(
    *,
    url: str,
    tool_name: str,
    arguments: JsonDict,
    timeout_s: float | None,
) -> JsonDict:
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                tool_name,
                arguments,
                read_timeout_seconds=_timeout_delta(timeout_s),
            )
    payload = _parse_mcp_tool_result(result)
    return payload


async def _list_sse_mcp_tools(
    *,
    url: str,
    timeout_s: float | None,
) -> JsonDict:
    from mcp import ClientSession
    from mcp.client.sse import sse_client

    async with sse_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
    return _parse_mcp_tools_result(result)


async def _with_optional_timeout(coro: Any, *, timeout_s: float | None) -> Any:
    if timeout_s is None or timeout_s <= 0:
        return await coro
    return await asyncio.wait_for(coro, timeout=timeout_s)


@contextmanager
def _temporary_no_proxy_for_url(url: str):
    """Bypass configured HTTP/SOCKS proxies for one MCP call.

    ``httpx`` still constructs a SOCKS proxy transport when ``ALL_PROXY`` is
    present, even if the target is listed in ``NO_PROXY``.  On hosts without
    the optional ``socksio`` dependency that fails before the request is
    attempted.  MCP simulator/perception services are explicitly selected by
    URL, so direct transport is the least surprising behavior here.
    """

    entries = _no_proxy_entries_for_url(url)
    if not entries:
        yield
        return
    proxy_keys = (
        "ALL_PROXY", "all_proxy",
        "HTTP_PROXY", "http_proxy",
        "HTTPS_PROXY", "https_proxy",
    )
    old_values = {
        key: os.environ.get(key)
        for key in ("NO_PROXY", "no_proxy", *proxy_keys)
    }
    try:
        merged = _merge_no_proxy_entries(old_values["NO_PROXY"], entries)
        os.environ["NO_PROXY"] = merged
        os.environ["no_proxy"] = merged
        for key in proxy_keys:
            os.environ.pop(key, None)
        yield
    finally:
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _no_proxy_entries_for_url(url: str) -> list[str]:
    parsed = urlparse(str(url or ""))
    host = parsed.hostname
    if not host:
        return []
    entries = [host]
    if parsed.port is not None:
        entries.append(f"{host}:{parsed.port}")
    return entries


def _merge_no_proxy_entries(existing: str | None, entries: Sequence[str]) -> str:
    merged: list[str] = []
    seen: set[str] = set()
    for value in [*(existing or "").split(","), *entries]:
        item = value.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        merged.append(item)
    return ",".join(merged)


def _parse_mcp_tools_result(result: Any) -> JsonDict:
    if isinstance(result, Mapping):
        raw_tools = result.get("tools", [])
    elif hasattr(result, "model_dump"):
        dumped = result.model_dump()
        raw_tools = dumped.get("tools", []) if isinstance(dumped, Mapping) else []
    else:
        raw_tools = getattr(result, "tools", [])
    if not isinstance(raw_tools, (list, tuple)):
        raw_tools = []
    tools = [_mcp_tool_to_dict(tool) for tool in raw_tools]
    return {"tools": tools, "tool_count": len(tools)}


def _mcp_tool_to_dict(tool: Any) -> JsonDict:
    if isinstance(tool, Mapping):
        payload = dict(tool)
    elif hasattr(tool, "model_dump"):
        dumped = tool.model_dump()
        payload = dict(dumped) if isinstance(dumped, Mapping) else {}
    else:
        payload = {
            "name": getattr(tool, "name", ""),
            "description": getattr(tool, "description", ""),
        }
        for attr in ("inputSchema", "input_schema"):
            value = getattr(tool, attr, None)
            if isinstance(value, Mapping):
                payload[attr] = dict(value)
    input_schema = payload.get("inputSchema")
    if input_schema is None:
        input_schema = payload.get("input_schema")
    normalized: JsonDict = {
        "name": str(payload.get("name") or ""),
        "description": str(payload.get("description") or ""),
    }
    if isinstance(input_schema, Mapping):
        normalized["input_schema"] = dict(input_schema)
    return normalized


def _parse_mcp_tool_result(result: Any) -> JsonDict:
    is_error = _mcp_result_is_error(result)
    content_items: Any = []
    payload: JsonDict | None = None
    if isinstance(result, Mapping):
        content_items = result.get("content", [])
        if any(key in result for key in ("isError", "is_error", "structuredContent", "structured_content")):
            for key in ("structuredContent", "structured_content"):
                structured = result.get(key)
                if isinstance(structured, Mapping):
                    payload = dict(structured)
                    break
            if payload is None:
                payload = _parse_mcp_content_items(content_items)
        else:
            payload = dict(result)

    if payload is None:
        for attr in ("structuredContent", "structured_content"):
            structured = getattr(result, attr, None)
            if isinstance(structured, Mapping):
                payload = dict(structured)
                break

    if payload is None and hasattr(result, "model_dump"):
        dumped = result.model_dump()
        if isinstance(dumped, Mapping):
            content_items = dumped.get("content", [])
            for key in ("structuredContent", "structured_content"):
                structured = dumped.get(key)
                if isinstance(structured, Mapping):
                    payload = dict(structured)
                    break
            if payload is None:
                payload = _parse_mcp_content_items(content_items)

    if payload is None:
        content_items = getattr(result, "content", []) or content_items
        payload = _parse_mcp_content_items(content_items)

    text_content = "\n".join(_mcp_content_texts(content_items)).strip()
    if payload is None:
        text = str(result)
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, dict):
            payload = decoded

    if payload is None:
        message = text_content or "Simulator MCP tool returned an invalid response."
        payload = {
            "success": False,
            "error": message,
            "content": message,
            "failure_class": _mcp_error_failure_class(message, is_error=is_error),
            "candidate_rejection": False,
            "details": {"raw_result_type": type(result).__name__},
        }

    if is_error:
        payload["success"] = False
        error_message = str(payload.get("error") or text_content or "").strip()
        if error_message:
            payload.setdefault("error", error_message)
            payload.setdefault("content", error_message)
        payload.setdefault(
            "failure_class",
            _mcp_error_failure_class(error_message, is_error=True),
        )
        payload.setdefault("candidate_rejection", False)
        details = payload.get("details")
        normalized_details = dict(details) if isinstance(details, Mapping) else {}
        normalized_details.setdefault("raw_result_type", type(result).__name__)
        normalized_details["mcp_is_error"] = True
        payload["details"] = normalized_details
    return payload


def _parse_mcp_content_items(items: Any) -> JsonDict | None:
    if not isinstance(items, (list, tuple)):
        return None
    for item in items:
        if isinstance(item, Mapping):
            if isinstance(item.get("json"), Mapping):
                return dict(item["json"])
            if isinstance(item.get("data"), Mapping):
                return dict(item["data"])
            text = item.get("text", "")
        else:
            if isinstance(getattr(item, "json", None), Mapping):
                return dict(getattr(item, "json"))
            if isinstance(getattr(item, "data", None), Mapping):
                return dict(getattr(item, "data"))
            text = getattr(item, "text", "")
        if isinstance(text, Mapping):
            return dict(text)
        if not isinstance(text, str) or not text.strip():
            continue
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _mcp_content_texts(items: Any) -> list[str]:
    if not isinstance(items, (list, tuple)):
        return []
    texts: list[str] = []
    for item in items:
        text = (
            item.get("text", "")
            if isinstance(item, Mapping)
            else getattr(item, "text", "")
        )
        if isinstance(text, str) and text.strip():
            texts.append(text.strip())
    return texts


def _mcp_result_is_error(result: Any) -> bool:
    if isinstance(result, Mapping):
        return result.get("isError") is True or result.get("is_error") is True
    if getattr(result, "isError", False) or getattr(result, "is_error", False):
        return True
    if hasattr(result, "model_dump"):
        dumped = result.model_dump()
        if isinstance(dumped, Mapping):
            return dumped.get("isError") is True or dumped.get("is_error") is True
    return False


def _mcp_error_failure_class(message: str, *, is_error: bool) -> str:
    normalized = message.lower()
    missing_patterns = (
        "unknown tool",
        "tool not found",
        "no tool named",
        "method not found",
    )
    if any(pattern in normalized for pattern in missing_patterns) or (
        "tool" in normalized and "not found" in normalized
    ):
        return "remote_capability_missing"
    return "mcp_tool_error" if is_error else "invalid_mcp_response"


def _response_success(response: JsonDict) -> bool:
    if response.get("success") is False:
        return False
    if response.get("ok") is False:
        return False
    return "error" not in response


def _context_execution_cancelled(context: ToolExecutionContext) -> bool:
    event = context.metadata.get("_cancel_event")
    return bool(
        event is not None
        and callable(getattr(event, "is_set", None))
        and event.is_set()
    )


def _response_content(response: JsonDict, *, mcp_tool: str, success: bool) -> str:
    content = response.get("content")
    if isinstance(content, str) and content.strip():
        return content if len(content) <= 500 else content[:500].rstrip()
    if not success:
        return str(response.get("error") or f"Simulator MCP tool failed: {mcp_tool}")
    response_path = response.get("response_path")
    if isinstance(response_path, str) and response_path:
        return f"Simulator MCP tool executed: {mcp_tool}; response saved to {response_path}"
    return f"Simulator MCP tool executed: {mcp_tool}"


def _brief_response_error(response: JsonDict) -> str:
    collision = response.get("collision")
    if _collision_rejects_motion(collision):
        return str(collision.get("message") or "Simulator motion collided before reaching target.")
    motion = build_motion_summary(response)
    if motion.get("reached_target") is False:
        return "Simulator motion did not reach the requested target."
    message = str(response.get("error") or response.get("content") or "")
    if len(message) > 500:
        return message[:500].rstrip()
    return message


def _response_diagnostics(response: JsonDict) -> list[JsonDict]:
    failure_class = str(response.get("failure_class") or "").strip()
    candidate_rejection = response.get("candidate_rejection") is True
    collision = response.get("collision")
    if _collision_rejects_motion(collision):
        return [
            {
                "code": failure_class or "simulator_mcp_collision",
                "message": _brief_response_error(response),
                "collision": dict(collision),
                "candidate_rejection": candidate_rejection,
                "failure_class": failure_class,
            }
        ]
    motion = build_motion_summary(response)
    if motion.get("reached_target") is False:
        return [
            {
                "code": failure_class or "simulator_mcp_target_not_reached",
                "message": _brief_response_error(response),
                "motion_summary": motion,
                "candidate_rejection": candidate_rejection,
                "failure_class": failure_class,
            }
        ]
    return [
        {
            "code": failure_class or "simulator_mcp_error",
            "message": _brief_response_error(response),
            "candidate_rejection": candidate_rejection,
            "failure_class": failure_class,
        }
    ]


def _collision_rejects_motion(collision: object) -> bool:
    if not isinstance(collision, dict) or collision.get("detected") is not True:
        return False
    return collision.get("new_or_worsened") is not False


def _state_delta_from_response(response: JsonDict) -> JsonDict:
    delta: JsonDict = {}
    if "reward" in response:
        delta["reward"] = response.get("reward")
    if "terminated" in response:
        delta["terminated"] = response.get("terminated")
    if "truncated" in response:
        delta["truncated"] = response.get("truncated")
    observation = build_observation_summary(response)
    if observation:
        delta["observation"] = observation
    motion = build_motion_summary(response)
    if motion:
        delta["motion"] = motion
    return delta


def _extract_orientation_arguments(
    parameters: JsonDict,
    *,
    tool_name: str,
) -> JsonDict:
    pose = parameters.get("target_pose") or parameters.get("pose") or parameters.get("eef_pose")
    if not isinstance(pose, dict):
        return {}

    direct = [pose.get(axis) for axis in ("roll", "pitch", "yaw")]
    if any(value is not None for value in direct):
        if not all(isinstance(value, int | float) for value in direct):
            raise ValueError("move_to orientation requires roll, pitch, and yaw together.")
        return {
            "roll": float(direct[0]),
            "pitch": float(direct[1]),
            "yaw": float(direct[2]),
        }

    euler = pose.get("euler_xyz_deg")
    if euler is not None:
        if not _finite_numeric_sequence(euler, length=3):
            raise ValueError(
                f"{tool_name} target_pose.euler_xyz_deg must contain 3 finite numbers."
            )
        return {axis: float(euler[idx]) for idx, axis in enumerate(("roll", "pitch", "yaw"))}

    rotation = pose.get("rotation_matrix")
    if rotation is None:
        return {}
    matrix = _finite_rotation_matrix(rotation)
    if matrix is None:
        raise ValueError(
            f"{tool_name} target_pose.rotation_matrix must be a finite 3x3 matrix."
        )
    roll, pitch, yaw = _rotation_matrix_to_xyz_intrinsic_degrees(matrix)
    return {"roll": roll, "pitch": pitch, "yaw": yaw}


def _rotation_matrix_to_xyz_intrinsic_degrees(
    matrix: list[list[float]],
) -> tuple[float, float, float]:
    pitch = math.asin(max(-1.0, min(1.0, -matrix[2][0])))
    cosine_pitch = math.cos(pitch)
    if abs(cosine_pitch) > 1e-8:
        roll = math.atan2(matrix[2][1], matrix[2][2])
        yaw = math.atan2(matrix[1][0], matrix[0][0])
    else:
        roll = math.atan2(-matrix[1][2], matrix[1][1])
        yaw = 0.0
    return tuple(math.degrees(value) for value in (roll, pitch, yaw))


def _finite_rotation_matrix(value: Any) -> list[list[float]] | None:
    if not isinstance(value, list | tuple) or len(value) != 3:
        return None
    rows: list[list[float]] = []
    for row in value:
        if not _finite_numeric_sequence(row, length=3):
            return None
        rows.append([float(item) for item in row])
    return rows


def _finite_numeric_sequence(value: Any, *, length: int) -> bool:
    return (
        isinstance(value, list | tuple)
        and len(value) == length
        and all(isinstance(item, int | float) and math.isfinite(float(item)) for item in value)
    )


def _extract_observation_payload(payload: JsonDict) -> JsonDict:
    observation = payload.get("observation")
    if isinstance(observation, dict):
        return observation
    if any(key in payload for key in ("cameras", "robot", "proprio", "objects")):
        return payload
    if any(key in payload for key in ("rgb_path", "rgb_ref", "image_path", "image_ref")):
        return {"cameras": [{"frame_id": "render", **payload}]}
    return {"cameras": [], "robot": RobotState().to_dict(), "metadata": {"raw_payload": payload}}


def _raise_if_mcp_error(payload: JsonDict, *, tool_name: str) -> None:
    if not isinstance(payload, dict):
        raise RuntimeError(f"{tool_name} returned {type(payload).__name__}")
    if payload.get("success") is False or payload.get("ok") is False or "error" in payload:
        raise RuntimeError(str(payload.get("error") or f"{tool_name} failed"))


def _is_transient_startup_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in ("unknown handle", "handle not found", "connection refused")
    )


def _summarize_mcp_action(action: EnvAction) -> JsonDict:
    request = action.command.get("request", {})
    return {
        "action_type": action.action_type,
        "request_kind": request.get("kind"),
        "request_name": request.get("name"),
        "status": action.command.get("status"),
        "tool_calls": [
            {
                "name": call.get("name"),
                "status": call.get("status"),
                "result_content": _truncate_action_text((call.get("result") or {}).get("content"))
                if isinstance(call.get("result"), dict)
                else None,
            }
            for call in action.command.get("tool_calls", [])
            if isinstance(call, dict)
        ],
    }


def _truncate_action_text(value: object, *, max_chars: int = 300) -> object:
    if not isinstance(value, str):
        return value
    return value if len(value) <= max_chars else value[:max_chars] + "...[truncated]"


def _latest_action_reward(action: EnvAction, payload: JsonDict) -> float:
    if "reward" in payload:
        try:
            return float(payload["reward"])
        except (TypeError, ValueError):
            return 0.0
    for call in reversed(action.command.get("tool_calls", [])):
        if not isinstance(call, dict):
            continue
        result = call.get("result")
        if not isinstance(result, dict):
            continue
        state_delta = result.get("details", {}).get("state_delta", {})
        if isinstance(state_delta, dict) and "reward" in state_delta:
            try:
                return float(state_delta["reward"])
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _latest_action_flag(action: EnvAction, payload: JsonDict, key: str) -> bool:
    if key in payload:
        return bool(payload.get(key))
    for call in reversed(action.command.get("tool_calls", [])):
        if not isinstance(call, dict):
            continue
        result = call.get("result")
        if not isinstance(result, dict):
            continue
        state_delta = result.get("details", {}).get("state_delta", {})
        if isinstance(state_delta, dict) and key in state_delta:
            return bool(state_delta[key])
    return False


def _extract_xyz(
    parameters: JsonDict,
    *,
    tool_name: str,
) -> tuple[float, float, float]:
    pose = (
        parameters.get("target_pose")
        or parameters.get("pose")
        or parameters.get("eef_pose")
        or parameters
    )
    if isinstance(pose, dict):
        frame = str(pose.get("frame") or "").strip().lower()
        if frame and frame != "world":
            raise ValueError(f"{tool_name} target_pose.frame must be 'world'.")
        xyz = pose.get("xyz") or pose.get("position")
        if xyz is None:
            xyz = pose.get("translation_xyz")
        if xyz is None and all(axis in pose for axis in ("x", "y", "z")):
            xyz = [pose["x"], pose["y"], pose["z"]]
    else:
        xyz = pose
    if not isinstance(xyz, (list, tuple)) or len(xyz) < 3:
        raise ValueError(f"{tool_name} requires target_pose.xyz or x/y/z.")
    return float(xyz[0]), float(xyz[1]), float(xyz[2])


def _timeout_delta(timeout_s: float | None) -> timedelta | None:
    if timeout_s is None:
        return None
    return timedelta(seconds=timeout_s)
