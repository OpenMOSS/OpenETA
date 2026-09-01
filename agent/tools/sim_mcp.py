"""MCP-only simulator tool proxy for OpenETA runtime."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import threading
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse
from uuid import uuid4

from adapter.motion_profiles import motion_control_profile
from adapter.protocol import EnvAction, EnvObservation, JsonDict, RobotState, StepResult
from agent.runtime.artifact_paths import artifact_session_id
from agent.runtime.image_artifacts import (
    DEFAULT_MCP_IMAGE_OUTPUT_ROOT,
    materialize_mcp_images,
)
from agent.runtime.response_artifacts import (
    DEFAULT_RESPONSE_ARTIFACT_OUTPUT_ROOT,
    build_motion_summary,
    build_observation_snapshot,
    build_observation_summary,
    build_reachability_summary,
    build_response_reference,
    materialize_json_response,
)
from agent.runtime.text_artifacts import (
    DEFAULT_MAX_INLINE_TEXT_CHARS,
    DEFAULT_TEXT_ARTIFACT_OUTPUT_ROOT,
)
from agent.tools.registry import (
    ENVIRONMENT_AUTHORITY,
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
    "ik_preview_check",
    "move_to",
    "follow_eef_trajectory",
    "gripper_control",
)

DEFAULT_SIMULATOR_IMAGE_WIDTH = 512
DEFAULT_SIMULATOR_IMAGE_HEIGHT = 512
DEFAULT_MCP_SSE_READ_TIMEOUT_S = 300.0
MCP_SSE_TIMEOUT_GRACE_S = 5.0
ENVIRONMENT_RECEIPT_SCHEMA_VERSION = "openeta.environment_receipt.v1"
RESOLVED_TOOL_EXECUTION_SCHEMA_VERSION = "openeta.resolved_tool_execution.v1"

SIMULATOR_CONTROL_MCP_TOOL_NAMES = (
    "move_to",
    "follow_eef_trajectory",
    "gripper_control",
)

DEFAULT_SIMULATOR_MCP_TOOL_MAP = {
    "create_simulator_env": "create_env",
    "close_simulator_env": "close_env",
    "observe": "render_env",
    "ik_preview_check": "ik_preview_check",
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


class SimulatorMcpTransportError(RuntimeError):
    """Typed transport failure that preserves a concrete nested SDK error."""

    def __init__(self, operation: str, cause: BaseException) -> None:
        primary = _primary_transport_exception(cause)
        if _is_transport_timeout(cause):
            code = "simulator_mcp_transport_timeout"
        elif _is_transient_mcp_transport_error(cause):
            code = "simulator_mcp_transport_connection_lost"
        else:
            code = "simulator_mcp_call_failed"
        message = str(primary).strip()
        detail = type(primary).__name__
        if message and message != detail:
            detail = f"{detail}: {message}"
        super().__init__(f"{operation} failed: {detail}")
        self.code = code
        self.operation = operation
        self.cause_type = type(primary).__name__


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
    forward_grasp_candidate_orientation: bool = False
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
    artifact_session_id: str = ""
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
        self.execution_id = ""
        self.agent_session_id = ""
        self._close_lock = threading.Lock()
        self._artifact_sequence = 0
        self._artifact_instance_id = uuid4().hex[:10]

    def reset(self, *, task: str, metadata: JsonDict | None = None) -> EnvObservation:
        self.task = task
        if isinstance(metadata, dict):
            self.execution_id = str(metadata.get("execution_id") or "")
            self.agent_session_id = str(metadata.get("agent_session_id") or "")
            self.config.artifact_session_id = str(
                metadata.get("agent_session_id") or self.config.artifact_session_id or ""
            ).strip()
        owns_environment = not self.config.handle
        attempts = max(1, self.config.startup_attempts if owns_environment else 1)
        for attempt in range(1, attempts + 1):
            self.startup_attempt_count = attempt
            try:
                if not self.config.handle:
                    self._create_env(task)
                payload = self._reset_env()
                break
            except Exception as exc:  # noqa: BLE001 - transient MCP failures may be grouped.
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
        self.config.session_id = str(self.create_result.get("session_id") or self.config.session_id)
        self.config.handle = str(self.create_result.get("handle") or "")
        if not self.config.handle:
            raise RuntimeError("create_env did not return a simulator handle")
        self._sync_tool_proxy_config()

    def _reset_env(self) -> JsonDict:
        reset_args: JsonDict = {"handle": self.config.handle, "seed": self.config.seed}
        if self.config.session_id:
            reset_args["session_id"] = self.config.session_id
        payload = self.transport.call_tool("reset_env", reset_args, timeout_s=self.config.timeout_s)
        _raise_if_mcp_error(payload, tool_name="reset_env")
        return payload

    def step(self, action: EnvAction) -> StepResult:
        render_args: JsonDict = {"handle": self.config.handle}
        if self.config.session_id:
            render_args["session_id"] = self.config.session_id
        attempts = max(1, self.config.startup_attempts)
        for attempt in range(1, attempts + 1):
            try:
                payload = self.transport.call_tool(
                    "render_env",
                    render_args,
                    timeout_s=self.config.timeout_s,
                )
                _raise_if_mcp_error(payload, tool_name="render_env")
                break
            except Exception as exc:  # noqa: BLE001 - transient MCP failures may be grouped.
                if attempt >= attempts or not _is_transient_startup_error(exc):
                    raise
                if self.config.startup_retry_delay_s > 0:
                    time.sleep(self.config.startup_retry_delay_s)
        observation = self._observation_from_payload(
            payload,
            metadata={
                "previous_action": _summarize_mcp_action(action),
                "source": type(self).__name__,
            },
        )
        remote_termination_reason = _latest_action_termination_reason(action)
        info = {
            "environment": type(self).__name__,
            "env_id": self.config.env_id,
            "session_id": self.config.session_id,
            "handle": self.config.handle,
            "previous_action": _summarize_mcp_action(action),
        }
        if remote_termination_reason:
            info.update(
                {
                    "termination_source": "simulator_mcp",
                    "termination_reason": remote_termination_reason,
                }
            )
        reward = _latest_action_reward(action, payload)
        terminated = _latest_action_flag(action, payload, "terminated") or bool(
            remote_termination_reason
        )
        truncated = _latest_action_flag(action, payload, "truncated")
        receipt = {
            "schema_version": ENVIRONMENT_RECEIPT_SCHEMA_VERSION,
            "receipt_id": uuid4().hex,
            "backend": "simulator_mcp_episode_environment",
            "agent_tool": "environment_step",
            "remote_tool": "render_env",
            "execution_id": self.execution_id,
            "agent_session_id": self.agent_session_id,
            "simulator_session_id": self.config.session_id,
            "handle": self.config.handle,
            "timestamp_s": time.time(),
            "reward_present": ("reward" in payload or _latest_action_receipt_has_reward(action)),
            "reward": reward,
            "terminated": terminated,
            "truncated": truncated,
            "observation_fresh": True,
        }
        info.update(
            {
                "environment_receipt_trusted": True,
                "official_reward": receipt["reward_present"],
                "environment_receipt": receipt,
            }
        )
        return StepResult(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
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
            bundle_id=self._next_observation_bundle_id(),
            session_id=self.config.artifact_session_id,
        )
        scrubbed = _with_anygrasp_camera_intrinsics(bundle.payload)
        self.last_payload = scrubbed
        observation_payload = _extract_observation_payload(scrubbed)
        assigned_task = observation_payload.get("task") or observation_payload.get(
            "task_description"
        )
        if isinstance(assigned_task, str) and assigned_task.strip():
            self.task = assigned_task.strip()
        else:
            assigned_task = ""
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
        if assigned_task:
            merged_metadata["assigned_task"] = self.task
            merged_metadata["assigned_task_source"] = "simulator_observation"
        if bundle.images:
            merged_metadata["image_artifacts"] = [image.to_dict() for image in bundle.images]
        merged_metadata.update(dict(metadata or {}))
        observation.metadata = merged_metadata
        return observation

    def _next_observation_bundle_id(self) -> str:
        self._artifact_sequence += 1
        base = self.config.session_id or self.config.handle or self.config.env_id
        return f"{base}-{self._artifact_instance_id}-{self._artifact_sequence:04d}-observation"


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
        self._artifact_instance_id = uuid4().hex[:10]
        # Capability observation, not task progress: once a backend accepts a
        # host contact authorization but omits the required attachment receipt,
        # keep later lift feedback honest instead of silently pretending that
        # the carried-object proxy is active.
        self._attachment_proxy_contract_missing = False

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
            transport_timeout = _is_transport_timeout(exc)
            transport_connection_lost = _is_transient_mcp_transport_error(exc)
            transport_unknown = context.spec.effect.value == "world_mutating" and (
                transport_timeout or transport_connection_lost
            )
            details = make_tool_result_details(
                context.spec,
                context.parameters,
                success=False,
                outputs={
                    "mcp": {
                        "tool": mcp_tool,
                        "agent_tool": agent_tool,
                        "session_id": arguments.get("session_id", ""),
                        "handle": arguments.get("handle", ""),
                    },
                    "motion_outcome": "unknown" if transport_unknown else "failed",
                    "reconciliation_required": transport_unknown,
                },
                diagnostics=[
                    {
                        "code": (
                            "simulator_mcp_transport_timeout"
                            if transport_timeout
                            else "simulator_mcp_transport_connection_lost"
                            if transport_unknown
                            else "simulator_mcp_call_failed"
                        ),
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
                ],
            )
            execution_receipt = _resolved_tool_execution_receipt(
                agent_tool,
                context.parameters,
                dispatch_status=(
                    "outcome_unknown" if transport_unknown else "transport_failed"
                ),
            )
            if execution_receipt:
                details["host_execution_receipt"] = execution_receipt
            return ToolResult(
                False,
                content=f"Simulator MCP tool failed: {mcp_tool}: {exc}",
                details=details,
            )

        success = _response_success(raw_response)
        incomplete_motion_receipt = (
            success
            and agent_tool in {"move_to", "follow_eef_trajectory"}
            and _move_response_lacks_completion_receipt(raw_response)
        )
        if incomplete_motion_receipt:
            success = False
        agent_feedback = _agent_visible_simulator_response(raw_response)
        normalized = self._normalize_response(
            raw_response,
            agent_tool=agent_tool,
            mcp_tool=mcp_tool,
            artifact_session_id=artifact_session_id(context.metadata),
            execution_metadata=context.metadata,
        )
        if agent_tool == "observe":
            response = normalized["outputs"].get("response")
            response = response if isinstance(response, dict) else {}
            cameras = response.get("cameras")
            cameras = cameras if isinstance(cameras, list) else []
            normalized["outputs"].update(
                {
                    "camera_ids": [
                        str(camera.get("frame_id") or "")
                        for camera in cameras
                        if isinstance(camera, dict) and camera.get("frame_id")
                    ],
                    "objects": (
                        response.get("objects")
                        if isinstance(response.get("objects"), list)
                        else []
                    ),
                    "metadata": (
                        response.get("metadata")
                        if isinstance(response.get("metadata"), dict)
                        else {}
                    ),
                }
            )
        attachment_contract_missing = False
        if mcp_tool == "gripper_open":
            # Opening retires any candidate attachment.  The backend capability
            # observation intentionally remains sticky for this proxy/session.
            pass
        elif mcp_tool == "gripper_close" and not isinstance(
            raw_response.get("attachment_proxy_receipt"), dict
        ):
            authorization = arguments.get("contact_authorization")
            if isinstance(authorization, dict):
                self._attachment_proxy_contract_missing = True
                attachment_contract_missing = True
                receipt = _missing_attachment_proxy_receipt(
                    authorization=authorization,
                    source_tool=mcp_tool,
                )
            else:
                receipt = _unarmed_attachment_proxy_receipt(source_tool=mcp_tool)
            normalized["outputs"]["attachment_proxy_receipt"] = receipt
            normalized["outputs"]["response"]["attachment_proxy_receipt"] = receipt
        elif (
            agent_tool in {"move_to", "follow_eef_trajectory"}
            and self._attachment_proxy_contract_missing
            and isinstance(arguments.get("contact_authorization"), dict)
            and not isinstance(raw_response.get("attachment_proxy_receipt"), dict)
        ):
            attachment_contract_missing = True
            receipt = _missing_attachment_proxy_receipt(
                authorization=arguments["contact_authorization"],
                source_tool=mcp_tool,
                refresh=True,
            )
            normalized["outputs"]["attachment_proxy_receipt"] = receipt
            normalized["outputs"]["response"]["attachment_proxy_receipt"] = receipt
        collision_coverage = _collision_coverage_receipt(
            raw_response,
            agent_tool=agent_tool,
            requested_collision_check=context.parameters.get(
                "enable_collision_check",
                context.parameters.get("check_endpoint_collision"),
            ),
        )
        if collision_coverage:
            normalized["outputs"]["collision_coverage"] = collision_coverage
            normalized["outputs"]["response"]["collision_coverage"] = collision_coverage
        if agent_tool in {"move_to", "follow_eef_trajectory"}:
            pose_feedback = _pose_feedback(context.parameters, agent_feedback)
            if pose_feedback:
                normalized["outputs"]["pose_feedback"] = pose_feedback
        if agent_tool == "move_to":
            evidence_handoff = _post_motion_evidence_handoff(
                context.parameters,
                agent_feedback,
            )
            if evidence_handoff:
                normalized["outputs"]["post_motion_evidence_handoff"] = (
                    evidence_handoff
                )
                normalized["outputs"]["response"][
                    "post_motion_evidence_handoff"
                ] = evidence_handoff
        if agent_tool == "ik_preview_check":
            # Execution authorization is host-owned. A backend may return a
            # legacy or stale execution reference, but it cannot authorize a
            # world-mutating call in the Agent-visible projection. Preserve the
            # complete backend response in the durable raw artifact and rebuild
            # only the host-verified fields below.
            for payload in (
                normalized["outputs"],
                normalized["outputs"].get("response"),
                normalized["outputs"].get("mcp"),
            ):
                if not isinstance(payload, dict):
                    continue
                for field_name in (
                    "motion_execution_ref",
                    "execution_authorization",
                    "ik_receipt_id",
                ):
                    payload.pop(field_name, None)
            reachability = normalized["outputs"].get("reachability")
            if isinstance(reachability, dict):
                ik_receipt = _ik_preview_receipt(
                    context.parameters,
                    reachability,
                )
                capability_resolver = context.metadata.get(
                    "_controller_capabilities_resolver"
                )
                controller_capabilities = (
                    capability_resolver() if callable(capability_resolver) else None
                )
                collision_delegation = _ik_motion_collision_delegation(
                    ik_receipt,
                    controller_capabilities=(
                        controller_capabilities
                        if isinstance(controller_capabilities, dict)
                        else {}
                    ),
                )
                ik_receipt["motion_collision_delegation"] = collision_delegation
                normalized["outputs"]["ik_preview_receipt"] = ik_receipt
                normalized["outputs"]["ik_receipt_id"] = ik_receipt.get("receipt_id")
                execution_authorization = _ik_execution_authorization(ik_receipt)
                normalized["outputs"]["execution_authorization"] = (
                    execution_authorization
                )
                normalized["outputs"]["response"]["ik_receipt_id"] = (
                    ik_receipt.get("receipt_id")
                )
                normalized["outputs"]["response"]["execution_authorization"] = (
                    execution_authorization
                )
                read_only_effect = {
                    "schema_version": "openeta.read_only_preflight_effect.v1",
                    "world_mutated": False,
                    "eef_pose_unchanged": True,
                    "robot_motion_epoch_unchanged": True,
                    "object_scene_epoch_unchanged": True,
                    "interpretation": (
                        "IK preview only checked geometry. It did not move the robot "
                        "or create a new camera viewpoint. Execute the returned receipt "
                        "with move_to, or combine 1-5 current-epoch receipt ids with "
                        "follow_eef_trajectory, before claiming the target was reached."
                    ),
                }
                normalized["outputs"]["world_effect"] = read_only_effect
                normalized["outputs"]["response"]["world_effect"] = read_only_effect
                if execution_authorization["authorized_for_move_to"] is True:
                    execution_ref = {
                        "schema_version": "openeta.ik_motion_execution_ref.v1",
                        "tool": "move_to",
                        "ik_receipt_id": ik_receipt.get("receipt_id"),
                        "instruction": (
                            "This preview did not move the robot. Pass this ik_receipt_id "
                            "to move_to to physically reach the checked endpoint, or "
                            "include it in ordered ik_receipt_ids for "
                            "follow_eef_trajectory; do not copy target_pose."
                        ),
                    }
                    normalized["outputs"]["motion_execution_ref"] = execution_ref
                    normalized["outputs"]["response"]["motion_execution_ref"] = (
                        execution_ref
                    )
                normalized["outputs"]["motion_collision_delegation"] = (
                    collision_delegation
                )
                normalized["outputs"]["response"]["motion_collision_delegation"] = (
                    collision_delegation
                )
        if agent_tool == "move_to" and _is_anyplace_pose(context.parameters):
            normalized["outputs"]["mcp"]["target_orientation_mode"] = "preserve_current"
        elif agent_tool == "move_to" and _is_ranked_grasp_candidate_pose(context.parameters):
            normalized["outputs"]["mcp"]["target_orientation_mode"] = (
                "graspnet_to_panda_eef"
                if self.config.forward_grasp_candidate_orientation
                else "preserve_current"
            )
        response_unknown = incomplete_motion_receipt or (
            not success
            and context.spec.effect.value == "world_mutating"
            and _response_lost_action_receipt(raw_response)
        )
        motion_target_not_reached = (
            agent_tool in {"move_to", "follow_eef_trajectory"}
            and not response_unknown
            and build_motion_summary(raw_response).get("reached_target") is False
        )
        motion_already_within_tolerance = (
            agent_tool in {"move_to", "follow_eef_trajectory"}
            and not response_unknown
            and _motion_already_within_tolerance(raw_response)
        )
        if motion_already_within_tolerance:
            normalized["outputs"]["motion_outcome"] = "no_state_change"
        if response_unknown:
            normalized["outputs"].update(
                {
                    "motion_outcome": "unknown",
                    "reconciliation_required": True,
                }
            )
            diagnostics = [
                {
                    "code": (
                        "simulator_mcp_motion_receipt_incomplete"
                        if incomplete_motion_receipt
                        else "simulator_mcp_action_receipt_unavailable"
                    ),
                    "message": _brief_response_error(agent_feedback),
                    "candidate_rejection": False,
                    "failure_class": "action_outcome_unknown",
                }
            ]
        else:
            diagnostics = (
                _response_diagnostics(agent_feedback)
                if not success or motion_target_not_reached
                else []
            )
        if (
            collision_coverage
            and collision_coverage.get("coverage_complete") is not True
            and collision_coverage.get("collision_detected") is not True
        ):
            diagnostics.append(
                {
                    "code": "collision_coverage_incomplete",
                    "severity": "warning",
                    "message": collision_coverage["interpretation"],
                    "coverage_status": collision_coverage["coverage_status"],
                    "trajectory_checked": collision_coverage["trajectory_checked"],
                    "world_checked": collision_coverage["world_checked"],
                }
            )
        if attachment_contract_missing:
            diagnostics.append(
                {
                    "code": "attachment_proxy_backend_contract_missing",
                    "severity": "warning",
                    "message": (
                        "The simulator accepted host contact authorization but did "
                        "not return an attachment-proxy receipt. Physical attachment "
                        "and carried-object collision coverage remain unknown."
                    ),
                    "backend_tool": mcp_tool,
                }
            )
        ik_receipt = normalized["outputs"].get("ik_preview_receipt")
        ik_classification = (
            str(ik_receipt.get("classification") or "")
            if isinstance(ik_receipt, dict)
            else ""
        )
        semantic_outcome = (
            "attachment_contract_unavailable"
            if attachment_contract_missing
            else "target_not_reached"
            if motion_target_not_reached
            else "target_already_within_tolerance"
            if motion_already_within_tolerance
            else (f"ik_{ik_classification}" if ik_classification else None)
        )
        recovery_options = (
            _attachment_contract_recovery_options()
            if attachment_contract_missing
            else _motion_target_miss_recovery_options(agent_feedback)
            if motion_target_not_reached
            else _motion_noop_recovery_options(agent_feedback)
            if motion_already_within_tolerance
            else (
                _ik_recovery_options(ik_receipt)
                if isinstance(ik_receipt, dict)
                and ik_classification
                in {
                    "repairable",
                    "inconclusive",
                    "kinematically_feasible_collision_deferred",
                    "hard_infeasible",
                }
                else None
            )
        )
        details = make_tool_result_details(
            context.spec,
            context.parameters,
            success=success,
            outputs=normalized["outputs"],
            artifacts=normalized["artifacts"],
            state_delta=normalized["state_delta"],
            environment_receipt=normalized["environment_receipt"],
            diagnostics=diagnostics,
            semantic_outcome=semantic_outcome,
            recovery_options=recovery_options,
            operational_success=(
                True
                if agent_tool == "ik_preview_check" and ik_classification
                else (success and not motion_target_not_reached)
            ),
        )
        execution_receipt = _resolved_tool_execution_receipt(
            agent_tool,
            context.parameters,
            dispatch_status="response_received",
        )
        if execution_receipt:
            details["host_execution_receipt"] = execution_receipt
        result_content = _response_content(
            normalized["outputs"]["response"],
            mcp_tool=mcp_tool,
            success=success,
        )
        # Recovery options are structured in details for auditing, but the
        # planner's compact tool-result projection is content-first. Surface
        # the primary executable repair inline so a large response artifact is
        # not required just to learn how to leave a collision boundary.
        if motion_target_not_reached and recovery_options:
            primary_recovery = recovery_options[0]
            action = str(primary_recovery.get("action") or "").strip()
            reason = str(primary_recovery.get("reason") or "").strip()
            parameters = primary_recovery.get("parameters")
            parameter_note = (
                "; suggested_parameters="
                + json.dumps(parameters, ensure_ascii=False, separators=(",", ":"))
                if isinstance(parameters, Mapping) and parameters
                else ""
            )
            result_content = (
                f"{result_content} Recommended recovery: {action}{parameter_note}. "
                f"{reason}"
            ).strip()
        return ToolResult(
            success,
            content=result_content,
            details=details,
        )

    def _mcp_call(
        self,
        context: ToolExecutionContext,
        *,
        agent_tool: str,
    ) -> tuple[str, JsonDict]:
        if agent_tool == "gripper_control":
            binary_position = self._binary_gripper_position(context.parameters)
            arguments: JsonDict = {}
            if binary_position == 0:
                resolver = context.metadata.get("_attachment_candidate_resolver")
                if callable(resolver):
                    authorization = resolver()
                    if isinstance(authorization, dict):
                        arguments["contact_authorization"] = authorization
            if agent_tool in self.config.tool_name_map:
                return self.config.tool_name_map[agent_tool], self._with_session(
                    {"position": binary_position, **arguments}
                )
            return self._gripper_tool_name(binary_position), self._with_session(arguments)
        if agent_tool in self.config.tool_name_map:
            return self.config.tool_name_map[agent_tool], self._with_session(
                dict(context.parameters)
            )
        if agent_tool == "observe":
            return self._mcp_tool_name(agent_tool), self._with_session({})
        if agent_tool == "ik_preview_check":
            return self._mcp_tool_name(agent_tool), self._ik_preview_arguments(
                context.parameters
            )
        if agent_tool == "move_to":
            return self._mcp_tool_name(agent_tool), self._move_to_arguments(
                context.parameters,
                metadata=context.metadata,
            )
        if agent_tool == "follow_eef_trajectory":
            arguments = dict(context.parameters)
            profile = motion_control_profile()
            if profile.sequential_route_preview_enabled:
                resolver = context.metadata.get(
                    "_ik_trajectory_execution_bundle_resolver"
                )
                if not callable(resolver):
                    raise ValueError(
                        "condition C requires the host-private trajectory bundle resolver"
                    )
                bundle = resolver(context.parameters)
                if not isinstance(bundle, dict):
                    raise ValueError(
                        "condition C could not resolve a current sequential route bundle"
                    )
                arguments["route_execution_bundle"] = bundle
            # Receipt ids are a host-side authorization/reference mechanism.  The
            # simulator owns only the resolved path and must not need to understand
            # OpenETA memory identifiers.
            arguments.pop("ik_receipt_ids", None)
            return self._mcp_tool_name(agent_tool), self._with_session(arguments)
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

    def _move_to_arguments(
        self,
        parameters: JsonDict,
        *,
        metadata: JsonDict | None = None,
    ) -> JsonDict:
        x, y, z = _extract_xyz(parameters, tool_name="move_to")
        arguments: JsonDict = {"x": x, "y": y, "z": z}
        if "speed" in parameters:
            raise ValueError(
                "move_to `speed` is unsupported by the simulator MCP; "
                "use num_steps/tolerance or omit speed."
            )
        is_anyplace_pose = _is_anyplace_pose(parameters)
        is_grasp_candidate = _is_ranked_grasp_candidate_pose(parameters)
        if is_anyplace_pose:
            pass
        elif is_grasp_candidate and self.config.forward_grasp_candidate_orientation:
            arguments.update(
                _extract_graspnet_panda_orientation_arguments(
                    parameters,
                    tool_name="move_to",
                )
            )
        elif not is_grasp_candidate:
            arguments.update(_extract_orientation_arguments(parameters, tool_name="move_to"))
        for key in ("handle", "session_id"):
            if key in parameters:
                arguments[key] = parameters[key]
        for key in ("num_steps", "tolerance", "ori_tolerance", "enable_collision_check"):
            if key in parameters:
                arguments[key] = parameters[key]
        target_pose = parameters.get("target_pose")
        resolver = (metadata or {}).get("_contact_authorization_resolver")
        if callable(resolver) and isinstance(target_pose, dict):
            authorization = resolver(target_pose)
            if isinstance(authorization, dict):
                arguments["contact_authorization"] = authorization
        seed_resolver = (metadata or {}).get("_ik_execution_seed_resolver")
        if callable(seed_resolver):
            execution_seed = seed_resolver(parameters)
            if isinstance(execution_seed, dict):
                arguments["ik_execution_seed"] = execution_seed
        return self._with_session(arguments)

    def _ik_preview_arguments(self, parameters: JsonDict) -> JsonDict:
        x, y, z = _extract_xyz(parameters, tool_name="ik_preview_check")
        arguments: JsonDict = {"x": x, "y": y, "z": z}
        is_anyplace_pose = _is_anyplace_pose(parameters)
        is_grasp_candidate = _is_ranked_grasp_candidate_pose(parameters)
        if is_anyplace_pose:
            pass
        elif is_grasp_candidate and self.config.forward_grasp_candidate_orientation:
            arguments.update(
                _extract_graspnet_panda_orientation_arguments(
                    parameters,
                    tool_name="ik_preview_check",
                )
            )
        elif not is_grasp_candidate:
            arguments.update(
                _extract_orientation_arguments(parameters, tool_name="ik_preview_check")
            )
        for key in (
            "position_tolerance_m",
            "orientation_tolerance_rad",
            "max_attempts",
            "max_nfev_per_attempt",
            "timeout_s",
            "preserve_current_orientation",
            "check_endpoint_collision",
            "include_scene_objects",
            "handle",
            "session_id",
        ):
            if key in parameters:
                arguments[key] = parameters[key]
        return self._with_session(arguments)

    def _binary_gripper_position(self, parameters: JsonDict) -> int:
        position = parameters.get("position")
        if position is None:
            position = parameters.get("open")
        if position is None:
            raise ValueError("gripper_control requires `position` or `open`.")
        if isinstance(position, bool):
            binary_position = int(position)
        elif isinstance(position, int | float) and not isinstance(position, bool):
            binary_position = float(position)
            if not math.isfinite(binary_position) or binary_position not in {0.0, 1.0}:
                raise ValueError("gripper_control position must be exactly 0 or 1.")
            binary_position = int(binary_position)
        else:
            raise ValueError("gripper_control position must be exactly 0 or 1.")
        return binary_position

    def _gripper_tool_name(self, binary_position: int) -> str:
        return "gripper_open" if binary_position == 1 else "gripper_close"

    def _normalize_response(
        self,
        response: JsonDict,
        *,
        agent_tool: str,
        mcp_tool: str,
        artifact_session_id: str = "",
        execution_metadata: JsonDict | None = None,
    ) -> JsonDict:
        payload = dict(response)
        bundle_id = self._next_artifact_bundle_id(mcp_tool)
        artifacts: list[JsonDict] = []
        if self.config.materialize_images:
            bundle = materialize_mcp_images(
                payload,
                output_root=self.config.image_output_root,
                bundle_id=bundle_id,
                session_id=artifact_session_id,
            )
            payload = bundle.payload
            artifacts = [image.to_dict() for image in bundle.images]
        payload = _with_anygrasp_camera_intrinsics(payload)
        # The transport response is trusted host input.  From this point on the
        # payload is Agent-owned: it enters ToolResult, memory/context, and a JSON
        # artifact readable through python_exec.  Project simulator-only safety
        # geometry once at this boundary so no later representation can revive it.
        payload = _agent_visible_simulator_response(payload)
        observation_snapshot = build_observation_snapshot(
            payload,
            image_artifacts=artifacts,
        )
        environment_receipt = _build_environment_receipt(
            payload,
            observation_snapshot=observation_snapshot,
            agent_tool=agent_tool,
            mcp_tool=mcp_tool,
            simulator_session_id=self.config.session_id,
            handle=self.config.handle,
            execution_metadata=execution_metadata,
        )
        response_artifact = materialize_json_response(
            payload,
            output_root=self.config.response_output_root,
            bundle_id=bundle_id,
            name=f"{mcp_tool}-response",
            session_id=artifact_session_id,
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
        for key in (
            "attachment_proxy_receipt",
            "gripper_actuation_receipt",
            "contact_authorization",
        ):
            value = payload.get(key)
            if isinstance(value, dict):
                outputs[key] = dict(value)
                response_ref[key] = dict(value)
        reachability_summary = build_reachability_summary(payload)
        if reachability_summary:
            outputs["reachability"] = reachability_summary
            outputs["feasible"] = reachability_summary.get("feasible")
            outputs["status"] = reachability_summary.get("status")
            outputs["reason_code"] = reachability_summary.get("reason_code")
        return {
            "outputs": outputs,
            "artifacts": artifacts,
            "state_delta": _state_delta_from_response(payload),
            "environment_receipt": environment_receipt,
        }

    def _next_artifact_bundle_id(self, mcp_tool: str) -> str:
        self._artifact_sequence += 1
        base = self.config.image_bundle_id or self.config.session_id or "simulator-mcp"
        return f"{base}-{self._artifact_instance_id}-{self._artifact_sequence:04d}-{mcp_tool}"


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
            artifact_session_id=artifact_session_id(context.metadata),
            execution_metadata=context.metadata,
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
            artifact_session_id=artifact_session_id(context.metadata),
            execution_metadata=context.metadata,
        )
        reset_ref = reset_normalized["outputs"]["response"]
        self._notify("reset_env", reset_args, reset_ref)
        success = _response_success(reset_response)
        assigned_task = _response_assigned_task(reset_ref)
        server_url = mcp_server_url_from_transport(self.transport)
        dashboard_url = mcp_dashboard_url(server_url, session_id)
        environment: JsonDict = {
            "env_id": env_id,
            "handle": handle,
            "session_id": session_id,
        }
        if assigned_task:
            environment["assigned_task"] = assigned_task
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
        if assigned_task:
            outputs["assigned_task"] = assigned_task
        for key in ("observation_summary",):
            summary = reset_normalized["outputs"].get(key)
            if isinstance(summary, dict):
                outputs[key] = summary
        return ToolResult(
            success,
            content=(
                (
                    (f"Simulator environment created and reset. Assigned task: {assigned_task}")
                    if assigned_task
                    else "Simulator environment created and reset."
                )
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
                environment_receipt={
                    **reset_normalized["environment_receipt"],
                    "simulator_session_id": session_id,
                    "handle": handle,
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
                environment_receipt={
                    "schema_version": ENVIRONMENT_RECEIPT_SCHEMA_VERSION,
                    "receipt_id": uuid4().hex,
                    "backend": "simulator_mcp",
                    "agent_tool": "close_simulator_env",
                    "remote_tool": "close_env",
                    "simulator_session_id": session_id,
                    "handle": "",
                    "timestamp_s": time.time(),
                    "reward_present": False,
                    "observation_fresh": False,
                    "environment_closed": True,
                },
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
            environment_receipt={
                "schema_version": ENVIRONMENT_RECEIPT_SCHEMA_VERSION,
                "receipt_id": uuid4().hex,
                "backend": "simulator_mcp",
                "agent_tool": "close_simulator_env",
                "remote_tool": "close_env",
                "simulator_session_id": session_id,
                "handle": handle,
                "timestamp_s": time.time(),
                "reward_present": False,
                "observation_fresh": False,
                "environment_closed": success,
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
        try:
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
        except SimulatorMcpTransportError:
            raise
        except Exception as exc:
            raise SimulatorMcpTransportError("list_tools", exc) from exc

    def call_tool(
        self,
        name: str,
        arguments: JsonDict,
        *,
        timeout_s: float | None = None,
    ) -> JsonDict:
        try:
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
        except SimulatorMcpTransportError:
            raise
        except Exception as exc:
            raise SimulatorMcpTransportError(f"call_tool:{name}", exc) from exc


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
        tools.bind_handler(
            name,
            handler,
            replace=replace,
            authority=ENVIRONMENT_AUTHORITY,
        )
    return tools


def close_environment_mcp_env(
    transport: SimulatorMcpTransport,
    *,
    handle: str,
    session_id: str = "",
    timeout_s: float | None = 30.0,
) -> JsonDict:
    """Best-effort cleanup for a remote MCP-managed environment.

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


def close_simulator_mcp_env(
    transport: SimulatorMcpTransport,
    *,
    handle: str,
    session_id: str = "",
    timeout_s: float | None = 30.0,
) -> JsonDict:
    """Backward-compatible simulator-specific name for MCP environment cleanup."""

    return close_environment_mcp_env(
        transport,
        handle=handle,
        session_id=session_id,
        timeout_s=timeout_s,
    )


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

    async with sse_client(
        url,
        sse_read_timeout=_sse_read_timeout_s(timeout_s),
    ) as (read, write):
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

    async with sse_client(
        url,
        sse_read_timeout=_sse_read_timeout_s(timeout_s),
    ) as (read, write):
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
    """Bypass local HTTP proxies for the target MCP host during one call."""

    entries = _no_proxy_entries_for_url(url)
    if not entries:
        yield
        return
    old_values = {key: os.environ.get(key) for key in ("NO_PROXY", "no_proxy")}
    try:
        merged = _merge_no_proxy_entries(old_values["NO_PROXY"], entries)
        os.environ["NO_PROXY"] = merged
        os.environ["no_proxy"] = merged
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
        if any(
            key in result
            for key in ("isError", "is_error", "structuredContent", "structured_content")
        ):
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
        text = item.get("text", "") if isinstance(item, Mapping) else getattr(item, "text", "")
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


_SIMULATOR_PRIVATE_SAFETY_KEYS = frozenset(
    {
        "attached_object",
        "binding_source",
        "contact_authorization",
        "constraint_boundary_recovery",
        "current_minimum_distance_m",
        "dims",
        "eef_to_target_distance_m",
        "geom1_name",
        "geom2_name",
        "geometry_names",
        "minimum_distance_m",
        "object_name",
        "obstacle",
        "pairs",
        "predicted_minimum_distance_m",
        "receptacle_aabb",
        "relative_xyz",
        "target_anchor_world_xyz",
        "target_object_name",
        "valid_center_xy",
        "world_object_count",
    }
)


def _agent_visible_simulator_response(response: JsonDict) -> JsonDict:
    """Project a trusted simulator response into non-privileged Agent evidence.

    The simulator may use object poses, collision geometry names, signed distances,
    or object extents internally.  Those values have no real-robot analogue and must
    not enter ToolResult, memory, planner context, or Agent-readable artifacts.
    Robot proprioception and requested/actual EEF poses remain visible.
    """

    def project(value: object, *, key: str = "") -> object:
        if isinstance(value, list):
            if key == "objects":
                return []
            return [project(item) for item in value]
        if not isinstance(value, dict):
            return value
        if key == "collision":
            return _agent_visible_collision_receipt(value)
        if key == "attachment_proxy_receipt":
            return _agent_visible_attachment_proxy_receipt(value)
        if key == "controller_failure":
            return _agent_visible_controller_failure(value)
        public: JsonDict = {}
        for raw_key, item in value.items():
            field = str(raw_key)
            if field in _SIMULATOR_PRIVATE_SAFETY_KEYS:
                continue
            if field == "objects":
                public[field] = []
                continue
            public[field] = project(item, key=field)
        return public

    projected = project(response)
    public = projected if isinstance(projected, dict) else {}
    collision = response.get("collision")
    controller_failure = response.get("controller_failure")
    attachment = response.get("attachment_proxy_receipt")
    if isinstance(collision, dict) and collision.get("detected") is True:
        public.pop("content", None)
        public["message"] = (
            "Motion was stopped by a host-private safety check. Inspect the fresh "
            "agentview/wrist evidence and replan from the reported actual EEF pose."
        )
    elif isinstance(controller_failure, dict):
        public.pop("content", None)
        public["message"] = (
            "The controller could not safely reach the requested pose. Inspect the "
            "fresh visual evidence and choose a materially different checked waypoint."
        )
    elif isinstance(attachment, dict):
        # Backend prose around the proxy commonly embeds the simulator instance name.
        public.pop("content", None)
        if isinstance(public.get("message"), str):
            public.pop("message", None)
    public["safety_feedback_projection"] = {
        "schema_version": "openeta.agent_safe_safety_feedback.v1",
        "privileged_geometry_exposed": False,
        "retained_evidence": (
            "safety verdict, checked scope, controller outcome, and robot proprioception"
        ),
    }
    return public


def _agent_visible_collision_receipt(collision: JsonDict) -> JsonDict:
    public: JsonDict = {}
    for field in (
        "available",
        "checked",
        "detected",
        "new_or_worsened",
        "self_checked",
        "self_collision",
        "world_collision",
        "endpoint_checked",
        "check_endpoint_collision",
        "trajectory_checked",
        "path_checked",
        "world_checked",
        "scene_checked",
    ):
        if isinstance(collision.get(field), bool):
            public[field] = collision[field]
    collision_type = str(collision.get("collision_type") or "").strip()
    public["collision_class"] = (
        collision_type
        if collision_type
        in {"self_collision", "robot_world", "attached_object_world", "endpoint"}
        else "unspecified_contact"
        if public.get("detected") is True
        else "none"
    )
    public["feedback_scope"] = "verdict_and_recovery_class_only"
    return public


def _agent_visible_attachment_proxy_receipt(receipt: JsonDict) -> JsonDict:
    public: JsonDict = {}
    for field in (
        "schema_version",
        "status",
        "attachment_proven",
        "collision_proxy_active",
        "contact_authorization_forwarded",
        "source_tool",
    ):
        value = receipt.get(field)
        if isinstance(value, str | bool) or value is None:
            public[field] = value
    for field in (
        "measured_open_fraction",
        "eef_displacement_m",
        "eef_displacement_since_close_m",
    ):
        value = receipt.get(field)
        if isinstance(value, int | float) and not isinstance(value, bool):
            public[field] = value
    reason = str(receipt.get("reason") or "").strip()
    allowed_reasons = {
        "awaiting_independent_co_motion_evidence",
        "aperture_collapsed_to_empty_close",
        "empty_close_or_no_measurable_contact",
        "eef_pose_unavailable",
        "no_active_contact_authorization",
        "remote_attachment_proxy_receipt_missing",
        "remote_attachment_proxy_refresh_receipt_missing",
        "visual_attachment_not_proven",
    }
    public_reason_aliases = {
        "authorized_target_outside_contact_envelope": (
            "close_not_supported_by_host_contact_envelope"
        ),
        "close_near_bound_target_pending_visual_confirmation": (
            "non_empty_close_with_tentative_safety_proxy"
        ),
        "non_empty_close_near_bound_target": "non_empty_close_with_tentative_safety_proxy",
    }
    public["reason"] = (
        reason
        if reason in allowed_reasons
        else public_reason_aliases.get(reason, "host_safety_proxy_update")
    )
    public["feedback_scope"] = "proxy_status_without_simulator_object_geometry"
    return public


def _agent_visible_controller_failure(failure: JsonDict) -> JsonDict:
    public: JsonDict = {}
    for field in ("schema_version", "code", "failure_class"):
        value = failure.get(field)
        if isinstance(value, str):
            public[field] = value
    public["recovery"] = (
        "Use the actual EEF pose and fresh visual evidence to choose a materially "
        "different checked waypoint or orientation; do not replay the rejected target."
    )
    public["feedback_scope"] = "controller_failure_class_without_private_geometry"
    return public


def _response_success(response: JsonDict) -> bool:
    if response.get("success") is False:
        return False
    if response.get("ok") is False:
        return False
    return "error" not in response


def _collision_coverage_receipt(
    response: JsonDict,
    *,
    agent_tool: str,
    requested_collision_check: object,
) -> JsonDict:
    """Describe what collision evidence a remote result actually covers."""

    if agent_tool not in {"ik_preview_check", "move_to", "follow_eef_trajectory"}:
        return {}
    collision = response.get("collision")
    collision = collision if isinstance(collision, dict) else {}

    def explicit_bool(*keys: str) -> bool:
        for key in keys:
            value = collision.get(key)
            if isinstance(value, bool):
                return value
        return False

    # IK services commonly expose a configuration-level collision receipt as
    # ``collision.checked`` rather than the motion-oriented
    # ``endpoint_checked`` spelling.  For IK that configuration is the
    # requested endpoint, so preserve the evidence instead of reporting the
    # scope as wholly unavailable.  Motion tools must still name endpoint
    # coverage explicitly; a generic ``checked`` bit is too ambiguous there.
    endpoint_checked = explicit_bool("endpoint_checked", "check_endpoint_collision")
    if agent_tool == "ik_preview_check" and not endpoint_checked:
        endpoint_checked = explicit_bool("checked")
    trajectory_checked = explicit_bool("trajectory_checked", "path_checked")
    world_checked = explicit_bool("world_checked", "scene_checked")
    detected = collision.get("detected") if isinstance(collision.get("detected"), bool) else None
    receipt_failed = (
        response.get("success") is False
        or response.get("ok") is False
        or response.get("available") is False
        or collision.get("available") is False
        or bool(response.get("error"))
        or bool(collision.get("error"))
    )
    world_update_failed = bool(
        response.get("world_update_error") or collision.get("world_update_error")
    )
    if receipt_failed:
        endpoint_checked = False
        trajectory_checked = False
        world_checked = False
    elif world_update_failed:
        world_checked = False
    if trajectory_checked and world_checked:
        status = "trajectory_and_world"
    elif trajectory_checked:
        status = "trajectory_without_world"
    elif endpoint_checked and world_checked:
        status = "endpoint_and_world"
    elif endpoint_checked:
        status = "endpoint_only"
    elif collision:
        status = "remote_collision_result_without_coverage"
    else:
        status = "unavailable"
    coverage_complete = (
        trajectory_checked and world_checked
        if agent_tool in {"move_to", "follow_eef_trajectory"}
        else endpoint_checked and world_checked
    )
    return {
        "schema_version": "openeta.collision_coverage_receipt.v1",
        "agent_tool": agent_tool,
        "requested_collision_check": (
            requested_collision_check
            if isinstance(requested_collision_check, bool)
            else None
        ),
        "coverage_status": status,
        "coverage_complete": coverage_complete,
        "endpoint_checked": endpoint_checked,
        "trajectory_checked": trajectory_checked,
        "world_checked": world_checked,
        "collision_detected": detected,
        "interpretation": (
            "No collision was reported, but the remote receipt does not prove full "
            "trajectory-and-world collision coverage. Treat it as unknown outside the "
            "explicit checked scope; inspect fresh visual evidence and use conservative "
            "clearance."
            if not coverage_complete and detected is not True
            else "Collision coverage is explicit for this request."
            if coverage_complete
            else (
                "The host-private safety checker reported a collision. Inspect fresh "
                "visual evidence and replan from the actual robot pose; simulator "
                "geometry names and distances are intentionally withheld."
            )
        ),
    }


def _context_execution_cancelled(context: ToolExecutionContext) -> bool:
    event = context.metadata.get("_cancel_event")
    return bool(event is not None and callable(getattr(event, "is_set", None)) and event.is_set())


def _motion_target_miss_recovery_options(response: JsonDict) -> list[JsonDict]:
    """Turn a failed motion receipt into executable recovery evidence.

    A zero-step collision is materially different from a controller that moved
    and stopped later.  In the former case the current configuration is already
    on or beyond a collision boundary; re-segmenting the same object cannot
    change that robot configuration.  Tell the Agent to escape from the actual
    endpoint first while leaving the retreat direction to its visual reasoning.
    """

    motion = build_motion_summary(response)
    collision = motion.get("collision")
    collision = collision if isinstance(collision, dict) else {}
    steps = motion.get("steps_executed")
    end = motion.get("end")
    actual_xyz = _motion_xyz(end) if isinstance(end, dict) else None
    controller_failure = motion.get("controller_failure")
    if isinstance(controller_failure, dict):
        return [
            {
                "action": "exit_reported_controller_boundary",
                "parameters": {
                    "actual_eef_xyz": actual_xyz,
                    "preserve_current_orientation": True,
                    "enable_collision_check": True,
                },
                "evidence": dict(controller_failure),
                "reason": str(
                    controller_failure.get("recovery")
                    or "Choose a materially different waypoint from the actual EEF pose."
                ),
            },
            {
                "action": "change_wrist_orientation_or_candidate",
                "reason": (
                    "If a short monotonic-clearance waypoint is unavailable, reject "
                    "this pose candidate rather than replaying the same QP attractor."
                ),
            },
        ]
    if steps == 0 and collision.get("detected") is True:
        return [
            {
                "action": "escape_current_collision_boundary",
                "parameters": {
                    "actual_eef_xyz": actual_xyz,
                    "preserve_current_orientation": True,
                    "enable_collision_check": True,
                },
                "evidence": {
                    "collision_detected": True,
                    "collision_class": collision.get("collision_class"),
                    "steps_executed": 0,
                    "feedback_scope": "host_private_geometry_withheld",
                },
                "reason": (
                    "No controller step executed because the current configuration is "
                    "already on or beyond a safety boundary. Inspect the "
                    "returned agentview/wrist images, choose a short retreat from "
                    "actual_eef_xyz that increases separation, preview it, and execute "
                    "it with the current orientation and collision checking. The "
                    "controller permits only a monotonic boundary exit; do not rotate "
                    "toward a new candidate until the current contact is cleared."
                ),
            },
            {
                "action": "consume_returned_motion_evidence",
                "reason": (
                    "The tool already returned a fresh observation, the unchanged actual "
                    "EEF pose, and a collision verdict. Re-segmenting the same object "
                    "does not move the robot or clear this boundary unless the image shows "
                    "that the object itself moved."
                ),
            },
            {
                "action": "replan_from_actual_pose",
                "parameters": {"actual_eef_xyz": actual_xyz},
                "reason": (
                    "Do not treat the rejected target pose as the robot's current pose."
                ),
            },
        ]
    if collision.get("detected") is True:
        target = motion.get("target")
        requested_xyz = _motion_xyz(target) if isinstance(target, dict) else None
        return [
            {
                "action": "classify_collision_visually_before_replanning",
                "evidence": {
                    "collision_detected": True,
                    "collision_class": collision.get("collision_class"),
                    "actual_eef_xyz": actual_xyz,
                    "requested_target_xyz": requested_xyz,
                    "steps_executed": steps,
                    "feedback_scope": "host_private_geometry_withheld",
                },
                "reason": (
                    "Use the returned agentview/wrist images to "
                    "distinguish transit clutter from intended target contact. A raised "
                    "transit detour does not repair a bad near-contact corridor or a "
                    "target that moved."
                ),
            },
            {
                "action": "plan_ik_checked_raised_or_lateral_detour",
                "parameters": {
                    "start_from_actual_eef_xyz": actual_xyz,
                    "preserve_current_orientation_for_clearance": True,
                    "preview_each_waypoint_separately": True,
                    "execute_with": "follow_eef_trajectory",
                    "enable_collision_check": True,
                },
                "reason": (
                    "For unrelated transit clutter, choose a visually free lift and/or "
                    "lateral point outside the contact envelope, IK-preview every point, "
                    "then execute the ordered receipts with collision checking. Do not "
                    "use the arithmetic midpoint of the failed segment because it remains "
                    "on the same swept path."
                ),
            },
            {
                "action": "replan_from_actual_pose",
                "parameters": {"actual_eef_xyz": actual_xyz},
                "reason": (
                    "The controller moved before stopping; the requested target is not "
                    "the current robot pose."
                ),
            },
        ]
    return [
        {
            "action": "inspect_fresh_observation",
            "reason": (
                "the controller executed but did not reach the requested pose; use the "
                "reported end pose and fresh images before deciding whether to retry, "
                "replan, or continue"
            ),
        },
        {
            "action": "replan_from_actual_pose",
            "reason": "do not treat the requested target pose as the robot's current pose",
        },
    ]


def _missing_attachment_proxy_receipt(
    *,
    authorization: JsonDict,
    source_tool: str,
    refresh: bool = False,
) -> JsonDict:
    """Describe an old/incomplete simulator contract without inventing state."""

    return {
        "schema_version": "openeta.attachment_proxy_receipt.v1",
        "status": "backend_contract_missing",
        "reason": (
            "remote_attachment_proxy_refresh_receipt_missing"
            if refresh
            else "remote_attachment_proxy_receipt_missing"
        ),
        "source_tool": source_tool,
        "target_object_name": str(authorization.get("target_object_name") or ""),
        "compiled_grasp_id": str(authorization.get("compiled_grasp_id") or ""),
        "contact_authorization_forwarded": True,
        "attachment_proven": False,
        "collision_proxy_active": None,
    }


def _unarmed_attachment_proxy_receipt(*, source_tool: str) -> JsonDict:
    """Make a close-without-host-target explicit to the Agent and auditor."""

    return {
        "schema_version": "openeta.attachment_proxy_receipt.v1",
        "status": "not_armed",
        "reason": "no_active_contact_authorization",
        "source_tool": source_tool,
        "contact_authorization_forwarded": False,
        "attachment_proven": False,
        "collision_proxy_active": False,
    }


def _attachment_contract_recovery_options() -> list[JsonDict]:
    return [
        {
            "action": "inspect_fresh_dual_view",
            "reason": (
                "the backend did not establish whether the target is attached; "
                "use source vacancy and object/gripper co-location evidence"
            ),
        },
        {
            "action": "small_guarded_lift_probe",
            "reason": (
                "if visual evidence is plausible, use only a small checked lift and "
                "verify co-motion before transport"
            ),
        },
        {
            "action": "upgrade_or_restart_simulator_service",
            "reason": (
                "the running service must return attachment_proxy_receipt for "
                "host-authorized close and carried-object proxy refresh"
            ),
        },
    ]


def _response_content(response: JsonDict, *, mcp_tool: str, success: bool) -> str:
    if _response_reports_remote_episode_terminated(response):
        return (
            "Simulator MCP reports that the remote episode is already terminated. "
            "No controller action was executed; do not retry or replan another "
            "world-mutating action in this environment."
        )
    content = response.get("content")
    if isinstance(content, str) and content.strip():
        # Preserve enough semantic feedback for the Agent to diagnose and
        # reflect.  Complete bulky responses still live in response_path;
        # planner-level total token projection, not a 500-character slice,
        # governs how much historical feedback remains model-visible.
        return content if len(content) <= 4_000 else content[:4_000].rstrip()
    reachability = response.get("reachability_summary")
    if isinstance(reachability, dict):
        status = str(reachability.get("status") or "unknown")
        reason = str(reachability.get("reason_code") or "unspecified")
        message = str(reachability.get("message") or "").strip()
        coverage = response.get("collision_coverage")
        coverage_note = ""
        if isinstance(coverage, dict) and coverage.get("coverage_complete") is not True:
            coverage_note = (
                " Endpoint kinematics do not prove path/world clearance: "
                + str(coverage.get("interpretation") or "collision coverage is incomplete")
            )
        delegation = response.get("motion_collision_delegation")
        delegation_note = ""
        if isinstance(delegation, dict) and delegation.get("applicable") is True:
            if delegation.get("available_for_matching_move") is True:
                delegation_note = (
                    " The current environment controller explicitly owns per-step "
                    "pre-actuation and post-step trajectory/world collision checking. "
                    "This exact pose may be passed to move_to with "
                    "enable_collision_check=true; the motion receipt, not this endpoint "
                    "preview, will provide path/world coverage."
                )
            else:
                delegation_note = (
                    " The current controller does not declare the required per-step "
                    "trajectory/world collision ownership, so this deferred preview "
                    "does not authorize motion."
                )
        receipt_id = str(response.get("ik_receipt_id") or "").strip()
        execution_ref = response.get("motion_execution_ref")
        authorization = response.get("execution_authorization")
        if receipt_id and isinstance(execution_ref, dict):
            receipt_note = (
                f" Execution reference: ik_receipt_id={receipt_id}; pass only this id "
                "to move_to and do not copy target_pose."
            )
        elif receipt_id and isinstance(authorization, dict):
            receipt_note = " " + str(
                authorization.get("instruction")
                or "This IK receipt does not authorize motion."
            )
        else:
            receipt_note = ""
        world_effect = response.get("world_effect")
        effect_note = ""
        if (
            isinstance(world_effect, dict)
            and world_effect.get("world_mutated") is False
        ):
            effect_note = (
                " This was a read-only preview and did not move the robot or "
                "create a new camera viewpoint."
            )
        return (
            f"IK preview {status} ({reason}). {message}{coverage_note}"
            f"{delegation_note}{effect_note}{receipt_note}"
        ).strip()
    has_motion_evidence = isinstance(response.get("motion_summary"), dict) or any(
        key in response for key in ("start", "end", "target", "controller_failure")
    )
    if not success and not has_motion_evidence:
        return str(
            response.get("message")
            or response.get("error")
            or f"Simulator MCP tool failed: {mcp_tool}"
        )
    response_path = response.get("response_path")
    attachment_receipt = response.get("attachment_proxy_receipt")
    if mcp_tool == "gripper_close" and isinstance(attachment_receipt, dict):
        actuation_receipt = response.get("gripper_actuation_receipt")
        actuation_receipt = (
            actuation_receipt if isinstance(actuation_receipt, dict) else {}
        )
        status = str(attachment_receipt.get("status") or "unknown")
        reason = str(attachment_receipt.get("reason") or "unspecified")
        aperture = attachment_receipt.get("measured_open_fraction")
        facts = [
            "binary close command is latched",
            (
                f"stationary_settle_steps={int(actuation_receipt['steps_executed'])}"
                if isinstance(actuation_receipt.get("steps_executed"), int)
                and not isinstance(actuation_receipt.get("steps_executed"), bool)
                else ""
            ),
            f"attachment_proxy_status={status}",
            f"reason={reason}",
            (
                f"measured_open_fraction={float(aperture):.4f}"
                if isinstance(aperture, int | float) and not isinstance(aperture, bool)
                else ""
            ),
            "attachment_proven=false",
        ]
        suffix = f" Full response saved to {response_path}" if response_path else ""
        prefix = (
            "Simulator MCP gripper close acknowledged: "
            + "; ".join(item for item in facts if item)
        )
        if status == "tentative":
            guidance = (
                ". Use the fresh dual-view observation and an Agent-chosen 2-5 cm "
                "lift probe from the measured current EEF pose for co-motion/source-"
                "vacancy evidence before transport. Exact-IK-check the new probe pose; "
                "do not reuse a prior grasp_clearance or grasp_precontact waypoint, "
                "which is usually too long or lateral for attachment verification."
            )
        elif status == "backend_contract_missing":
            guidance = (
                ". The running simulator did not report whether its carried-object "
                "collision proxy was armed. Physical attachment is unknown: inspect "
                "fresh dual-view evidence and, only if plausible, use a small guarded "
                "lift to verify co-motion. Upgrade or restart the simulator service "
                "before relying on attachment-aware collision coverage."
            )
        else:
            guidance = (
                ". No carried-object proxy is active. Do NOT treat a lift as an "
                "attachment probe; inspect the fresh dual-view observation, reopen "
                "the gripper, and repair or reacquire contact before lifting."
            )
        return prefix + guidance + suffix
    attachment_note = ""
    if isinstance(attachment_receipt, dict):
        attachment_status = str(attachment_receipt.get("status") or "unknown")
        attachment_reason = str(attachment_receipt.get("reason") or "unspecified")
        attachment_note = (
            " Carried-object proxy feedback: "
            f"status={attachment_status}; reason={attachment_reason}; "
            + "attachment_proven=false. Use fresh dual-view evidence for the "
            "attachment verdict."
        )
    compact_motion = response.get("motion_summary")
    collision = response.get("collision")
    if not isinstance(collision, dict) and isinstance(compact_motion, dict):
        collision = compact_motion.get("collision")
    if _collision_rejects_motion(collision):
        steps = compact_motion.get("steps_executed") if isinstance(compact_motion, dict) else None
        stop_note = (
            " No controller step executed; choose a checked waypoint that reduces or "
            "escapes the visually observed contact instead of replaying the motion."
            if steps == 0
            else ""
        )
        suffix = f" Full response saved to {response_path}" if response_path else ""
        return (
            "Simulator MCP tool stopped for collision. The host-private checker "
            "withholds simulator object names, geometry, and exact clearance values; "
            f"use the fresh dual-view images and actual EEF pose for recovery.{stop_note}"
            f"{attachment_note}{suffix}"
        )
    motion = (
        dict(compact_motion)
        if isinstance(compact_motion, dict)
        else build_motion_summary(response)
    )
    evidence_handoff = response.get("post_motion_evidence_handoff")
    evidence_handoff = (
        evidence_handoff if isinstance(evidence_handoff, dict) else {}
    )
    handoff_note = ""
    if evidence_handoff.get("status") == "fresh_wrist_packet_expected":
        camera_frame_id = str(evidence_handoff.get("camera_frame_id") or "wrist")
        handoff_note = (
            " Wrist observation viewpoint reached. This gathered evidence; it did "
            "not refine the older contact pose. In the next planner context copy "
            "current_observation.source_packet_id, segment the same target with "
            f"camera_frame_id={camera_frame_id}, confirm its identity, then consume "
            "the ready wrist-alignment bundle or run a full wrist grasp estimate "
            "before reusing the scene-view contact reference."
        )
    if motion.get("reached_target") is False:
        target = motion.get("target") if isinstance(motion.get("target"), dict) else {}
        end = motion.get("end") if isinstance(motion.get("end"), dict) else {}
        target_xyz = _motion_xyz(target)
        end_xyz = _motion_xyz(end)
        error_m = _position_error_m(target_xyz, end_xyz)
        controller_receipt = motion.get("controller_receipt")
        controller_id = (
            str(controller_receipt.get("controller_id") or "")
            if isinstance(controller_receipt, dict)
            else ""
        )
        controller_failure = motion.get("controller_failure")
        controller_failure = (
            controller_failure if isinstance(controller_failure, dict) else {}
        )
        facts = [
            f"requested_target_xyz={target_xyz}" if target_xyz else "",
            f"actual_end_xyz={end_xyz}" if end_xyz else "",
            f"position_error_m={error_m:.4f}" if error_m is not None else "",
            f"controller_id={controller_id}" if controller_id else "",
            (
                f"controller_failure={controller_failure.get('code')}"
                if controller_failure.get("code")
                else ""
            ),
        ]
        summary = "; ".join(item for item in facts if item)
        suffix = f" Full response saved to {response_path}" if response_path else ""
        return (
            f"Simulator MCP tool executed: {mcp_tool}, but the requested target was NOT "
            f"reached. {summary}. Do not assume the requested pose was achieved; inspect "
            f"the fresh observation and replan from the actual end pose. "
            f"{controller_failure.get('recovery') or ''}"
            f"{attachment_note}{suffix}"
        )
    if motion:
        target = motion.get("target") if isinstance(motion.get("target"), dict) else {}
        end = motion.get("end") if isinstance(motion.get("end"), dict) else {}
        target_xyz = _motion_xyz(target)
        end_xyz = _motion_xyz(end)
        error_m = _position_error_m(target_xyz, end_xyz)
        facts = [
            f"requested_target_xyz={target_xyz}" if target_xyz else "",
            f"actual_end_xyz={end_xyz}" if end_xyz else "",
            f"position_error_m={error_m:.4f}" if error_m is not None else "",
        ]
        summary = "; ".join(item for item in facts if item)
        suffix = f" Full response saved to {response_path}" if response_path else ""
        if _motion_already_within_tolerance({"motion_summary": motion}):
            return (
                "Simulator MCP tool executed zero controller steps because the current "
                f"EEF pose was already inside the requested tolerance; {summary}. The "
                "robot and physical camera viewpoint did NOT change. A newly captured "
                "packet is not a materially new view. If a different view is required, "
                "propose a checked pose outside the current tolerance envelope or use "
                "a justified tighter tolerance/orientation, then preview that exact pose."
                f"{attachment_note}{suffix}"
            )
        if summary:
            return (
                f"Simulator MCP tool executed: {mcp_tool}; {summary}."
                f"{handoff_note}{attachment_note}{suffix}"
            )
    if isinstance(response_path, str) and response_path:
        return f"Simulator MCP tool executed: {mcp_tool}; response saved to {response_path}"
    return f"Simulator MCP tool executed: {mcp_tool}"


def _motion_already_within_tolerance(response: JsonDict) -> bool:
    nested = response.get("motion_summary")
    motion = dict(nested) if isinstance(nested, dict) else build_motion_summary(response)
    if motion.get("reached_target") is not True or motion.get("steps_executed") != 0:
        return False
    start = motion.get("start")
    end = motion.get("end")
    start_xyz = _motion_xyz(start if isinstance(start, dict) else None)
    end_xyz = _motion_xyz(end if isinstance(end, dict) else None)
    if not start_xyz or not end_xyz:
        return False
    distance = _position_error_m(start_xyz, end_xyz)
    return distance is not None and distance <= 1e-9


def _motion_noop_recovery_options(response: JsonDict) -> list[JsonDict]:
    motion = build_motion_summary(response)
    return [
        {
            "action": "consume_existing_visual_evidence",
            "reason": (
                "zero controller steps means the physical viewpoint did not change; "
                "do not repeat perception merely because a new packet id was minted"
            ),
        },
        {
            "action": "propose_materially_distinct_checked_endpoint",
            "evidence": {
                "actual_eef_pose": motion.get("end"),
                "requested_target": motion.get("target"),
            },
            "reason": (
                "if the task needs a different camera view or contact geometry, choose "
                "a pose outside the current tolerance envelope and preview it exactly"
            ),
        },
    ]


def _pose_feedback(parameters: JsonDict, response: JsonDict) -> JsonDict:
    """Return compact requested/actual EEF feedback without inferring task success."""

    requested = parameters.get("target_pose")
    if not isinstance(requested, dict):
        requested = None
    motion = build_motion_summary(response)
    actual = motion.get("end") if isinstance(motion.get("end"), dict) else None
    remote_target = motion.get("target") if isinstance(motion.get("target"), dict) else None
    requested_xyz = _motion_xyz(requested or remote_target)
    actual_xyz = _motion_xyz(actual)
    if requested is None and remote_target is None and actual is None:
        return {}
    return {
        "schema_version": "openeta.eef_pose_feedback.v1",
        "requested_eef_pose": dict(requested or remote_target or {}),
        "actual_eef_pose": dict(actual or {}),
        "requested_xyz": requested_xyz or None,
        "actual_xyz": actual_xyz or None,
        "position_error_m": _position_error_m(requested_xyz, actual_xyz),
        "reached_target": motion.get("reached_target"),
        "interpretation": (
            "EEF kinematic feedback only; use fresh visual evidence to judge "
            "object-relative contact and task progress."
        ),
    }


def _resolved_tool_execution_receipt(
    agent_tool: str,
    parameters: JsonDict,
    *,
    dispatch_status: str,
) -> JsonDict:
    """Persist exact host-resolved motion inputs outside Agent-owned parameters.

    The public planner contract intentionally carries short IK receipt ids.  Memory
    still needs the exact geometry that the trusted simulator proxy consumed in
    order to derive contact/clearance receipts and reconcile unknown outcomes.
    This receipt remains a top-level ToolResult detail, so bounded conversation
    projections do not replay the full pose or trajectory to the Agent.
    """

    if agent_tool not in {"move_to", "follow_eef_trajectory"}:
        return {}
    geometry_key = "target_pose" if agent_tool == "move_to" else "trajectory"
    geometry = parameters.get(geometry_key)
    if not isinstance(geometry, dict if geometry_key == "target_pose" else list):
        return {}
    resolved_parameters = deepcopy(parameters)
    reference_kind = (
        "ik_receipt"
        if agent_tool == "move_to" and parameters.get("ik_receipt_id")
        else "ik_trajectory_receipts"
        if agent_tool == "follow_eef_trajectory" and parameters.get("ik_receipt_ids")
        else "host_resolved_geometry"
    )
    return {
        "schema_version": RESOLVED_TOOL_EXECUTION_SCHEMA_VERSION,
        "receipt_id": f"resolved-execution:{uuid4().hex}",
        "tool": agent_tool,
        "reference_kind": reference_kind,
        "dispatch_status": dispatch_status,
        "parameters": resolved_parameters,
    }


def _post_motion_evidence_handoff(
    parameters: JsonDict,
    response: JsonDict,
) -> JsonDict:
    """Expose how an observation waypoint should be consumed after motion.

    This is receipt-derived workflow information, not a task stage or a motion
    authorization.  It prevents a successful camera move from being mistaken for
    an update to the older grasp contact geometry.
    """

    target_pose = parameters.get("target_pose")
    if not isinstance(target_pose, dict):
        return {}
    if str(target_pose.get("waypoint_role") or "") != "wrist_observation_viewpoint":
        return {}
    motion = build_motion_summary(response)
    reached_target = motion.get("reached_target") is True
    steps_executed = motion.get("steps_executed")
    materially_new_view = bool(
        reached_target
        and isinstance(steps_executed, int)
        and not isinstance(steps_executed, bool)
        and steps_executed > 0
    )
    status = (
        "fresh_wrist_packet_expected"
        if materially_new_view
        else "no_new_physical_view"
        if reached_target
        else "viewpoint_not_reached"
    )
    camera_frame_id = str(target_pose.get("camera_frame_id") or "wrist")
    return {
        "schema_version": "openeta.post_motion_evidence_handoff.v1",
        "waypoint_role": "wrist_observation_viewpoint",
        "status": status,
        "reached_target": reached_target,
        "materially_new_view": materially_new_view,
        "camera_frame_id": camera_frame_id,
        "compiled_grasp_id": str(target_pose.get("compiled_grasp_id") or ""),
        "viewpoint_candidate_id": str(
            target_pose.get("viewpoint_candidate_id") or ""
        ),
        "fresh_packet_source": (
            "current_observation.source_packet_id in the next planner context"
            if materially_new_view
            else None
        ),
        "agent_discretion": True,
        "recommended_next_actions": (
            [
                {
                    "tool": "sam3",
                    "parameters_from_next_context": {
                        "source_packet_id": "current_observation.source_packet_id",
                        "camera_frame_id": camera_frame_id,
                    },
                    "purpose": "segment the same target on the fresh wrist view",
                },
                {
                    "tool": "select_sam3_detection",
                    "purpose": "confirm cross-view target identity",
                },
                {
                    "tool": "compute_wrist_alignment_or_grasp_pose_estimate",
                    "purpose": (
                        "refine lateral contact from the ready host bundle or replace "
                        "uncertain orientation/depth using a full wrist estimate"
                    ),
                },
            ]
            if materially_new_view
            else []
        ),
        "interpretation": (
            "The observation viewpoint was reached, but the older scene-view contact "
            "pose was not thereby refined. Consume the fresh wrist evidence before "
            "reusing that contact reference."
            if materially_new_view
            else (
                "The requested observation viewpoint did not produce a new physical "
                "camera view; do not claim fresh near-field evidence."
            )
        ),
    }


def _ik_preview_receipt(parameters: JsonDict, reachability: JsonDict) -> JsonDict:
    target_pose = parameters.get("target_pose")
    target_pose = dict(target_pose) if isinstance(target_pose, dict) else {}
    orientation = {
        key: target_pose.get(key)
        for key in (
            "rotation_matrix",
            "quat_xyzw",
            "quaternion",
            "rotvec",
            "roll",
            "pitch",
            "yaw",
        )
        if target_pose.get(key) is not None
    }
    preserve_current = parameters.get("preserve_current_orientation")
    if preserve_current is None:
        preserve_current = not orientation
    canonical = {
        "target_xyz": target_pose.get("xyz", target_pose.get("translation_xyz")),
        "orientation_policy": (
            "preserve_current" if preserve_current is True else "explicit_orientation"
        ),
        "orientation": orientation,
        "position_tolerance_m": parameters.get(
            "position_tolerance_m", parameters.get("tolerance")
        ),
        "orientation_tolerance_rad": parameters.get(
            "orientation_tolerance_rad", parameters.get("ori_tolerance")
        ),
        "check_endpoint_collision": parameters.get("check_endpoint_collision"),
    }
    target_signature = hashlib.sha256(
        json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:24]
    pose_policy_signature = hashlib.sha256(
        json.dumps(
            {
                "target_xyz": canonical["target_xyz"],
                "orientation_policy": canonical["orientation_policy"],
                "orientation": canonical["orientation"],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:24]
    classification = _ik_reachability_classification(reachability)
    receipt_id = hashlib.sha256(
        json.dumps(
            {"target": canonical, "reachability": reachability},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:20]
    return {
        "schema_version": "openeta.ik_preview_receipt.v1",
        "receipt_id": receipt_id,
        "target_signature": target_signature,
        "pose_policy_signature": pose_policy_signature,
        "classification": classification,
        "target_pose": target_pose,
        "orientation_policy": canonical["orientation_policy"],
        "tolerances": {
            "position_tolerance_m": canonical["position_tolerance_m"],
            "orientation_tolerance_rad": canonical["orientation_tolerance_rad"],
        },
        "reason_code": reachability.get("reason_code"),
        "message": reachability.get("message"),
        "best_candidate": reachability.get("best_candidate"),
        "suggestions": reachability.get("suggestions", []),
        "reachability": dict(reachability),
    }


def _ik_reachability_classification(reachability: JsonDict) -> str:
    status = str(reachability.get("status") or "unknown").lower()
    if status == "reachable":
        return "feasible"
    if (
        status == "unknown"
        and str(reachability.get("reason_code") or "")
        == "endpoint_collision_check_unavailable"
        and isinstance(reachability.get("best_candidate"), dict)
    ):
        return "kinematically_feasible_collision_deferred"
    if status == "unknown":
        return "inconclusive"
    collision = reachability.get("collision")
    if isinstance(collision, dict) and collision.get("detected") is True:
        return "hard_infeasible"
    reason = str(reachability.get("reason_code") or "").lower()
    if any(
        marker in reason
        for marker in (
            "outside_workspace",
            "joint_limit",
            "endpoint_collision",
            "self_collision",
            "invalid_target",
        )
    ):
        return "hard_infeasible"
    if (
        reachability.get("position_only_reachable") is True
        or reachability.get("orientation_only_reachable") is True
        or isinstance(reachability.get("best_candidate"), dict)
        or bool(reachability.get("suggestions"))
    ):
        return "repairable"
    return "hard_infeasible"


def _ik_motion_collision_delegation(
    receipt: JsonDict,
    *,
    controller_capabilities: JsonDict,
) -> JsonDict:
    applicable = (
        str(receipt.get("classification") or "")
        == "kinematically_feasible_collision_deferred"
        and str(receipt.get("reason_code") or "")
        == "endpoint_collision_check_unavailable"
    )
    available = bool(
        applicable
        and controller_capabilities.get("motion_owns_trajectory_world_collision")
        is True
    )
    return {
        "schema_version": "openeta.ik_motion_collision_delegation.v1",
        "applicable": applicable,
        "available_for_matching_move": available,
        "controller_id": str(controller_capabilities.get("controller_id") or ""),
        "goal_executor": str(controller_capabilities.get("goal_executor") or ""),
        "collision_scope": str(controller_capabilities.get("collision_scope") or ""),
        "required_move_parameters": {"enable_collision_check": True},
        "pose_requirement": "same_numerically_equivalent_target_and_orientation_policy",
        "interpretation": (
            "Endpoint IK proved kinematics; the declared motion controller will own "
            "trajectory/world collision checking for the matching move."
            if available
            else (
                "Endpoint collision was deferred and no verified collision-owning "
                "motion controller is declared."
                if applicable
                else "Collision delegation is not needed for this IK classification."
            )
        ),
    }


def _ik_execution_authorization(receipt: JsonDict) -> JsonDict:
    """Separate durable IK evidence from permission to execute that exact pose."""

    classification = str(receipt.get("classification") or "")
    delegation = receipt.get("motion_collision_delegation")
    delegated = bool(
        classification == "kinematically_feasible_collision_deferred"
        and isinstance(delegation, dict)
        and delegation.get("available_for_matching_move") is True
    )
    authorized = classification == "feasible" or delegated
    receipt_id = str(receipt.get("receipt_id") or "")
    reason_code = str(receipt.get("reason_code") or "unspecified")
    if authorized:
        instruction = (
            f"Pass ik_receipt_id={receipt_id} to move_to and do not copy target_pose. "
            "The authorization is valid only for the numerically equivalent target "
            "and orientation policy recorded by this receipt."
        )
    else:
        instruction = (
            f"Do not pass ik_receipt_id={receipt_id} to move_to: this receipt is "
            f"non-executable ({reason_code}). Change the target pose, orientation "
            "policy, or grasp candidate and run ik_preview_check again. Repeating the "
            "same xyz and explicit orientation while merely omitting a tolerance is "
            "not a recovery."
        )
    return {
        "schema_version": "openeta.ik_execution_authorization.v1",
        "ik_receipt_id": receipt_id,
        "authorized_for_move_to": authorized,
        "authorization_basis": (
            "endpoint_feasible"
            if classification == "feasible"
            else (
                "kinematics_plus_verified_motion_collision_delegation"
                if delegated
                else f"rejected_{classification or 'unknown'}"
            )
        ),
        "exact_pose_and_orientation_policy_only": authorized,
        "same_pose_retry_disposition": (
            "execution_reference_available"
            if authorized
            else "requires_materially_changed_pose_or_policy"
        ),
        "instruction": instruction,
    }


def _ik_recovery_options(receipt: JsonDict) -> list[JsonDict]:
    reason_code = str(receipt.get("reason_code") or "").strip().lower()
    if reason_code == "endpoint_collision_check_unavailable":
        delegation = receipt.get("motion_collision_delegation")
        delegation = delegation if isinstance(delegation, dict) else {}
        if delegation.get("available_for_matching_move") is True:
            return [
                {
                    "action": "execute_exact_pose_with_verified_motion_collision",
                    "parameters": {"enable_collision_check": True},
                    "reason": (
                        "The current controller capability directly confirms per-step "
                        "pre/post trajectory-and-world collision ownership. Execute "
                        "only this numerically equivalent pose and inspect the motion "
                        "receipt."
                    ),
                },
                {
                    "action": "choose_non_execution_recovery",
                    "reason": (
                        "The Agent may still choose another candidate, viewpoint, or "
                        "observation when visual evidence does not support the move."
                    ),
                },
            ]
        return [
            {
                "action": "delegate_collision_to_verified_motion_controller",
                "parameters": {"enable_collision_check": True},
                "reason": (
                    "IK already produced a valid joint solution. Execute this exact "
                    "pose only if controller_capabilities says motion owns per-step "
                    "pre/post trajectory-and-world collision checking; keep "
                    "enable_collision_check=true. Repeating the same IK with collision "
                    "disabled adds no evidence."
                ),
            },
            {
                "action": "restore_endpoint_collision_backend",
                "reason": (
                    "If endpoint collision proof is mandatory, install or repair the "
                    "reported collision backend before retrying this check."
                ),
            },
        ]
    options: list[JsonDict] = [
        {
            "action": "inspect_fresh_observation",
            "reason": "observation is read-only and can be refreshed without executing the rejected pose",
        }
    ]
    best = receipt.get("best_candidate")
    if isinstance(best, dict):
        options.append(
            {
                "action": "review_nearest_reachable_candidate",
                "candidate": dict(best),
                "reason": "use it as evidence for an Agent-authored adjusted endpoint, not a silent host substitution",
            }
        )
    for suggestion in receipt.get("suggestions", []) or []:
        if isinstance(suggestion, str):
            options.append({"action": suggestion, "reason": "IK backend repair suggestion"})
    options.append(
        {
            "action": "preview_modified_pose",
            "reason": "change xyz or orientation policy, then run a new endpoint preview",
        }
    )
    return options[:10]


def _motion_xyz(value: object) -> list[float]:
    if not isinstance(value, dict):
        return []
    xyz = value.get("xyz")
    if isinstance(xyz, list | tuple) and len(xyz) >= 3:
        try:
            parsed = [float(component) for component in xyz[:3]]
        except (TypeError, ValueError):
            return []
        if all(math.isfinite(component) for component in parsed):
            return parsed
    keys = ("x", "y", "z")
    try:
        parsed = [float(value[key]) for key in keys]
    except (KeyError, TypeError, ValueError):
        return []
    return parsed if all(math.isfinite(component) for component in parsed) else []


def _position_error_m(target_xyz: list[float], end_xyz: list[float]) -> float | None:
    if len(target_xyz) != 3 or len(end_xyz) != 3:
        return None
    return math.sqrt(sum((target - end) ** 2 for target, end in zip(target_xyz, end_xyz)))


def _response_assigned_task(response: JsonDict) -> str:
    observation = response.get("observation_summary")
    if not isinstance(observation, dict):
        return ""
    task = observation.get("task")
    return task.strip() if isinstance(task, str) else ""


def _brief_response_error(response: JsonDict) -> str:
    if _response_reports_remote_episode_terminated(response):
        return str(
            response.get("error")
            or response.get("message")
            or response.get("content")
            or "Remote simulator episode is terminated."
        )
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


def _iter_exception_chain(exc: BaseException) -> Iterable[BaseException]:
    """Yield nested exceptions, including Python 3.11 exception groups."""

    pending = [exc]
    seen: set[int] = set()
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        yield current
        grouped = getattr(current, "exceptions", ())
        if isinstance(grouped, tuple):
            pending.extend(item for item in grouped if isinstance(item, BaseException))
        if isinstance(current.__cause__, BaseException):
            pending.append(current.__cause__)
        if isinstance(current.__context__, BaseException):
            pending.append(current.__context__)


def _primary_transport_exception(exc: BaseException) -> BaseException:
    chain = list(_iter_exception_chain(exc))
    leaves = [item for item in chain if not getattr(item, "exceptions", ())]
    for item in leaves:
        if _is_transport_timeout(item) or _is_transient_mcp_transport_error(item):
            return item
    return leaves[0] if leaves else exc


def _is_transport_timeout(exc: BaseException) -> bool:
    return any(
        "timeout" in type(item).__name__.lower()
        or "timed out" in str(item).lower()
        or "read timeout" in str(item).lower()
        for item in _iter_exception_chain(exc)
    )


def _is_transient_mcp_transport_error(exc: BaseException) -> bool:
    type_markers = (
        "brokenresourceerror",
        "connecterror",
        "connectionreseterror",
        "endofstream",
        "readerror",
        "remoteprotocolerror",
    )
    message_markers = (
        "connection refused",
        "connection reset",
        "incomplete chunked read",
        "peer closed connection",
        "server disconnected",
        "unexpected eof",
    )
    for item in _iter_exception_chain(exc):
        name = type(item).__name__.lower()
        message = str(item).lower()
        if "unsupportedprotocol" in name or "missing protocol" in message:
            continue
        if any(marker in name for marker in type_markers):
            return True
        if any(marker in message for marker in message_markers):
            return True
    return False


def _response_lost_action_receipt(response: JsonDict) -> bool:
    """Return true when a mutating call may have run but its receipt was lost."""

    message = str(response.get("error") or response.get("content") or "").lower()
    return "out of range float values are not json compliant" in message


def _move_response_lacks_completion_receipt(response: JsonDict) -> bool:
    """Detect a controller receipt that ran steps but omitted its terminal pose."""

    motion = build_motion_summary(response)
    if motion.get("reached_target") in {True, False}:
        return False
    try:
        steps = int(motion.get("steps_executed") or response.get("steps_executed") or 0)
    except (TypeError, ValueError):
        return False
    if steps <= 0:
        return False
    end = motion.get("end")
    end_xyz = end.get("xyz") if isinstance(end, dict) else None
    return not (
        isinstance(end_xyz, list | tuple)
        and len(end_xyz) >= 3
        and all(
            isinstance(value, int | float) and math.isfinite(float(value)) for value in end_xyz[:3]
        )
    )


def _response_diagnostics(response: JsonDict) -> list[JsonDict]:
    if _response_reports_remote_episode_terminated(response):
        return [
            {
                "code": "remote_episode_terminated",
                "message": _brief_response_error(response),
                "candidate_rejection": False,
                "failure_class": "environment_terminal",
            }
        ]
    reachability = build_reachability_summary(response)
    if reachability.get("status") == "unreachable":
        return [
            {
                "code": str(reachability.get("reason_code") or "ik_target_unreachable"),
                "message": str(
                    reachability.get("message")
                    or "The requested endpoint pose is unreachable."
                ),
                "reachability": reachability,
                "candidate_rejection": True,
                "failure_class": "ik_target_unreachable",
            }
        ]
    failure_class = str(response.get("failure_class") or "").strip()
    candidate_rejection = response.get("candidate_rejection") is True
    collision = response.get("collision")
    if _collision_rejects_motion(collision):
        motion = build_motion_summary(response)
        target = motion.get("target") if isinstance(motion.get("target"), dict) else {}
        end = motion.get("end") if isinstance(motion.get("end"), dict) else {}
        return [
            {
                "code": failure_class or "simulator_mcp_collision",
                "message": _brief_response_error(response),
                "collision": dict(collision),
                "motion_summary": motion,
                "position_error_m": _position_error_m(
                    _motion_xyz(target), _motion_xyz(end)
                ),
                "candidate_rejection": candidate_rejection,
                "failure_class": failure_class,
            }
        ]
    motion = build_motion_summary(response)
    if motion.get("reached_target") is False:
        target = motion.get("target") if isinstance(motion.get("target"), dict) else {}
        end = motion.get("end") if isinstance(motion.get("end"), dict) else {}
        position_error_m = _position_error_m(_motion_xyz(target), _motion_xyz(end))
        return [
            {
                "code": failure_class or "simulator_mcp_target_not_reached",
                "message": _brief_response_error(response),
                "motion_summary": motion,
                "position_error_m": position_error_m,
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
    elif _response_reports_remote_episode_terminated(response):
        delta["terminated"] = True
    if "truncated" in response:
        delta["truncated"] = response.get("truncated")
    observation = build_observation_summary(response)
    if observation:
        delta["observation"] = observation
    motion = build_motion_summary(response)
    if motion:
        delta["motion"] = motion
    return delta


def _build_environment_receipt(
    response: JsonDict,
    *,
    observation_snapshot: JsonDict,
    agent_tool: str,
    mcp_tool: str,
    simulator_session_id: str,
    handle: str,
    execution_metadata: JsonDict | None,
) -> JsonDict:
    metadata = dict(execution_metadata or {})
    receipt: JsonDict = {
        "schema_version": ENVIRONMENT_RECEIPT_SCHEMA_VERSION,
        "receipt_id": uuid4().hex,
        "backend": "simulator_mcp",
        "agent_tool": agent_tool,
        "remote_tool": mcp_tool,
        "execution_id": str(metadata.get("execution_id") or ""),
        "agent_session_id": str(metadata.get("session_id") or ""),
        "simulator_session_id": simulator_session_id,
        "handle": handle,
        "timestamp_s": time.time(),
        "reward_present": "reward" in response,
        "observation_fresh": bool(observation_snapshot),
    }
    for key in ("reward", "terminated", "truncated", "scene_epoch"):
        if key in response:
            receipt[key] = response.get(key)
    if _response_reports_remote_episode_terminated(response):
        # A worker can discover the horizon boundary only when the next action
        # is attempted.  The explicit remote error is authoritative evidence
        # that this environment can no longer accept world-mutating actions,
        # even when the failing response omitted a boolean terminal field.
        receipt["terminated"] = True
    motion = build_motion_summary(response)
    if motion:
        receipt["motion"] = motion
    if observation_snapshot:
        receipt["observation_snapshot"] = observation_snapshot
    return receipt


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

    quaternion = pose.get("quat_xyzw")
    if quaternion is not None:
        if not _finite_numeric_sequence(quaternion, length=4):
            raise ValueError(
                f"{tool_name} target_pose.quat_xyzw must contain 4 finite numbers."
            )
        qx, qy, qz, qw = [float(value) for value in quaternion]
        norm = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
        if norm <= 1e-9:
            raise ValueError(f"{tool_name} target_pose.quat_xyzw must be non-zero.")
        qx, qy, qz, qw = [value / norm for value in (qx, qy, qz, qw)]
        sin_roll_cos_pitch = 2.0 * (qw * qx + qy * qz)
        cos_roll_cos_pitch = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sin_roll_cos_pitch, cos_roll_cos_pitch)
        sin_pitch = 2.0 * (qw * qy - qz * qx)
        pitch = math.copysign(math.pi / 2.0, sin_pitch) if abs(sin_pitch) >= 1.0 else math.asin(sin_pitch)
        sin_yaw_cos_pitch = 2.0 * (qw * qz + qx * qy)
        cos_yaw_cos_pitch = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(sin_yaw_cos_pitch, cos_yaw_cos_pitch)
        return {
            "roll": math.degrees(roll),
            "pitch": math.degrees(pitch),
            "yaw": math.degrees(yaw),
        }

    rotation = pose.get("rotation_matrix")
    if rotation is None:
        return {}
    matrix = _finite_rotation_matrix(rotation)
    if matrix is None:
        raise ValueError(f"{tool_name} target_pose.rotation_matrix must be a finite 3x3 matrix.")
    roll, pitch, yaw = _rotation_matrix_to_xyz_intrinsic_degrees(matrix)
    return {"roll": roll, "pitch": pitch, "yaw": yaw}


def _is_ranked_grasp_candidate_pose(parameters: JsonDict) -> bool:
    """Return whether target_pose carries normalized grasp-candidate provenance.

    GraspNet-family rotation matrices describe a grasp frame, not the simulator
    controller's EEF frame. The compatibility default preserves the current EEF
    orientation until a deployment explicitly enables its calibrated mapping.
    """

    pose = parameters.get("target_pose") or parameters.get("pose")
    if not isinstance(pose, dict):
        return False
    candidate_id = str(pose.get("id") or pose.get("candidate_id") or "").strip()
    source_model = str(pose.get("source_model") or "").strip().lower()
    if source_model in {"anygrasp", "contact_graspnet"}:
        return True
    if candidate_id.startswith("place_grasp_") and str(
        pose.get("source_grasp_id") or ""
    ).startswith("grasp_"):
        return True
    return candidate_id.startswith("grasp_") and any(
        key in pose for key in ("rank", "backend_index", "score", "gripper_tip_position_xyz")
    )


def _is_anyplace_pose(parameters: JsonDict) -> bool:
    pose = parameters.get("target_pose") or parameters.get("pose")
    if not isinstance(pose, dict):
        return False
    candidate_id = str(pose.get("id") or pose.get("candidate_id") or "").strip()
    source_grasp_id = str(pose.get("source_grasp_id") or "").strip()
    return candidate_id.startswith("place_grasp_") and source_grasp_id.startswith("grasp_")


def _extract_graspnet_panda_orientation_arguments(
    parameters: JsonDict,
    *,
    tool_name: str,
) -> JsonDict:
    """Map a world GraspNet grasp frame to robosuite Panda EEF axes."""

    pose = parameters.get("target_pose") or parameters.get("pose")
    if not isinstance(pose, dict):
        return {}
    rotation = pose.get("rotation_matrix")
    if rotation is None:
        return {}
    grasp_matrix = _finite_rotation_matrix(rotation)
    if grasp_matrix is None:
        raise ValueError(f"{tool_name} target_pose.rotation_matrix must be a finite 3x3 matrix.")

    # GraspNet: x=approach, y=closing, z=binormal. Panda EEF:
    # x=closing, y=binormal, z=approach.
    eef_matrix = [[row[1], row[2], row[0]] for row in grasp_matrix]
    roll, pitch, yaw = _rotation_matrix_to_xyz_intrinsic_degrees(eef_matrix)
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


def _is_transient_startup_error(exc: BaseException) -> bool:
    if _is_transport_timeout(exc) or _is_transient_mcp_transport_error(exc):
        return True
    return any(
        marker in str(item).lower()
        for item in _iter_exception_chain(exc)
        for marker in ("unknown handle", "handle not found")
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
    receipt = _latest_trusted_action_environment_receipt(action)
    receipt_reward = receipt.get("reward")
    if (
        (receipt.get("terminated") is True or receipt.get("truncated") is True)
        and receipt.get("reward_present") is True
        and isinstance(receipt_reward, int | float)
        and not isinstance(receipt_reward, bool)
        and math.isfinite(float(receipt_reward))
    ):
        # ``payload`` is the read-only render performed after the Agent action.
        # A legacy backend may echo reward=0 there; it cannot overwrite the
        # action's trusted terminal receipt.
        return float(receipt_reward)
    if "reward" in payload:
        reward = payload.get("reward")
        if (
            isinstance(reward, int | float)
            and not isinstance(reward, bool)
            and math.isfinite(float(reward))
        ):
            return float(reward)
        return 0.0
    reward = receipt_reward
    if (
        receipt.get("reward_present") is True
        and isinstance(reward, int | float)
        and not isinstance(reward, bool)
        and math.isfinite(float(reward))
    ):
        return float(reward)
    return 0.0


def _latest_action_flag(action: EnvAction, payload: JsonDict, key: str) -> bool:
    receipt = _latest_trusted_action_environment_receipt(action)
    receipt_value = receipt.get(key)
    if receipt_value is True and key in {"terminated", "truncated"}:
        return True
    if key in payload:
        value = payload.get(key)
        return value if isinstance(value, bool) else False
    return receipt_value if isinstance(receipt_value, bool) else False


def _latest_action_receipt_has_reward(action: EnvAction) -> bool:
    return _latest_trusted_action_environment_receipt(action).get("reward_present") is True


def _latest_trusted_action_environment_receipt(action: EnvAction) -> JsonDict:
    calls = action.command.get("tool_calls")
    if not isinstance(calls, list):
        return {}
    latest: JsonDict = {}
    for call in calls:
        if not isinstance(call, dict):
            continue
        result = call.get("result")
        details = result.get("details") if isinstance(result, dict) else None
        if not isinstance(details, dict):
            continue
        provenance = details.get("host_provenance")
        receipt = details.get("environment_receipt")
        if (
            isinstance(provenance, dict)
            and provenance.get("authority") == ENVIRONMENT_AUTHORITY
            and isinstance(receipt, dict)
            and receipt.get("schema_version") == ENVIRONMENT_RECEIPT_SCHEMA_VERSION
        ):
            latest = receipt
            if receipt.get("terminated") is True or receipt.get("truncated") is True:
                # Terminal evidence is absorbing within the action.  The
                # episode runner cannot stop between receipts produced inside
                # one runtime.act(), so a later synthetic/replayed response
                # must not overwrite the official terminal reward.
                return receipt
    return latest


def _latest_action_termination_reason(action: EnvAction) -> str:
    """Recognize an explicit remote episode-terminal error in the last tool result."""

    markers = (
        "executing action in terminated episode",
        "episode is terminated",
        "episode already terminated",
    )
    for call in reversed(action.command.get("tool_calls", [])):
        if not isinstance(call, dict):
            continue
        result = call.get("result")
        if not isinstance(result, dict) or result.get("success") is not False:
            continue
        messages = [result.get("content")]
        details = result.get("details")
        diagnostics = details.get("diagnostics") if isinstance(details, dict) else None
        messages.extend(item.get("message") for item in diagnostics or [] if isinstance(item, dict))
        if any(marker in str(message or "").lower() for marker in markers for message in messages):
            return "remote_episode_terminated"
    return ""


def _response_reports_remote_episode_terminated(response: Mapping[str, object]) -> bool:
    """Recognize a worker response that explicitly reports a terminal episode."""

    markers = (
        "executing action in terminated episode",
        "episode is terminated",
        "episode already terminated",
    )
    messages = (
        response.get("error"),
        response.get("message"),
        response.get("content"),
    )
    return any(
        marker in str(message or "").lower()
        for marker in markers
        for message in messages
    )


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


def _sse_read_timeout_s(timeout_s: float | None) -> float:
    """Keep the transport stream alive through the caller-owned deadline."""

    if timeout_s is None or timeout_s <= 0:
        return DEFAULT_MCP_SSE_READ_TIMEOUT_S
    return max(DEFAULT_MCP_SSE_READ_TIMEOUT_S, float(timeout_s) + MCP_SSE_TIMEOUT_GRACE_S)
