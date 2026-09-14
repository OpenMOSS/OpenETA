"""Experimental external-decision ingress for the existing OpenETA host.

No model API client is constructed here. Codex supplies one typed command; the
ordinary planner, pipeline, registry and episode runner remain authoritative.
"""
from __future__ import annotations

import argparse
import json
import threading
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

import jsonschema
from mcp.types import CallToolResult, ImageContent, TextContent, Tool

from agent.backends.planner import (
    PlannerBackend, PlannerBackendRequest, PlannerBackendResult,
    OpenAICompatiblePlannerBackendConfig, _planner_user_content,
    _planner_visible_main_agent_context,
)
from agent.backends.provider_config import PlannerProviderConfig
from agent.runtime.episode import OpenEtaEpisodeRunner
from agent.runtime.planner import build_tool_context, _decision_from_backend_result
from agent.runtime.runtime_assembly import (
    RuntimeAssemblyConfig, RuntimeMcpEndpoints, assemble_runtime,
)
from agent.runtime.session_workspace import SessionWorkspace
from agent.runtime.supervision import SupervisionPolicy
from agent.runtime.visual_history import VisualHistoryConfig
from agent.runtime.interface_profiles import project_tool_reference
from agent.runtime.success_evidence import episode_success_evidence
from agent.tools.contracts import project_agent_tool_contract
from agent.tools.sim_mcp import (
    SimulatorMcpEpisodeConfig, SimulatorMcpEpisodeEnvironment,
    SimulatorMcpToolProxyConfig, SseSimulatorMcpTransport,
)
from agent.tools.web_access import WebAccessConfig
from tools.codex_feedback import execution_error
from tools.codex_motion import motion_schema, run_motion_hook
from tools.codex_evidence import observation_references, repair_feedback, image_label
from tools.codex_controller import CONTROLLER_IDS, require_controller

# This is an admission subset, not another source of parameter schemas.
PICK_TOOLS = frozenset({
    "observe", "sam3", "inspect_evidence", "select_sam3_detection",
    "reject_sam3_detections", "grasp_pose_estimate", "compile_grasp_seed",
    "compute_wrist_alignment", "propose_wrist_viewpoints", "propose_motion_target",
    "ik_preview_check", "compose_ik_trajectory", "move_to",
    "follow_eef_trajectory", "gripper_control",
})


class DisabledModelBackend(PlannerBackend):
    def decide(self, request):
        raise RuntimeError("Auxiliary model inference is disabled in Codex plugin smoke mode")


class SubmittedBackend(PlannerBackend):
    def __init__(self):
        self.pending = None

    def decide(self, request):
        if self.pending is None:
            raise RuntimeError("No external decision pending")
        payload, self.pending = self.pending, None
        return PlannerBackendResult(payload=payload, provider="codex-mcp", model="external",
                                    details={"usage_source": "external_unreported"})


class CodexHost:
    def __init__(self, runner, backend, *, output: Path, max_requests=80, tool_profile="pick"):
        self.runner, self.backend = runner, backend
        self.runtime = runner.runtime
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.max_requests = max_requests
        self.requests = 0
        self.lock = threading.RLock()
        self.stop_requested = threading.Event()
        self.closed = False
        self.cleanup = None
        self.official_task_success = False
        self.completion_claim = None
        self.timer = None
        self.schemas = {}
        self.last_command = None
        self.interventions = []
        self.request_physics = []
        self.request_tool = None
        self.call_sequence = 0
        self.previous_response = None
        self.execution_accounting = {'preflight_calls':0, 'physical_dispatches':0,
                                     'physical_calls_with_steps':0, 'physical_unknown_outcomes':0}
        self.tool_profile = tool_profile
        self.atomic = None
        self._published_observation = self.runner.current_observation
        for spec in self.runtime.tools.list():
            if spec.name not in PICK_TOOLS or not self.runtime.tools.can_execute(spec.name):
                continue
            ref = project_tool_reference(
                project_agent_tool_contract(self.runtime.pipeline.tool_contract_catalog.get(spec.name)),
                self.runtime.pipeline.agent_interface_profile,
            )
            self.schemas[spec.name] = Tool(name=spec.name, description=ref["description"],
                                           inputSchema=ref["parameters"])
        # Compose existing contracts only at this experimental ingress. The
        # internal planner/pipeline keep their original receipt-only move API.
        self.motion_hook_enabled = (
            self.runtime.pipeline.agent_interface_profile == "bundle_stage3"
            and all(name in self.schemas for name in ("move_to", "propose_motion_target", "ik_preview_check"))
        )
        if self.motion_hook_enabled:
            self.schemas["move_to"] = motion_schema(self.schemas["move_to"])
        empty = {"type": "object", "properties": {}, "additionalProperties": False}
        self.schemas["episode_status"] = Tool(name="episode_status", inputSchema=empty,
            description="Read current episode state, evidence and images. Does not refresh the simulator.")
        self.schemas["finish_episode"] = Tool(name="finish_episode", inputSchema={
            "type": "object", "properties": {"success": {"type": "boolean"}, "reason": {"type": "string", "minLength": 1}},
            "required": ["success", "reason"], "additionalProperties": False},
            description="Submit task completion to existing Host gates. Success needs current official evidence. Use success=false to stop an unsuccessful attempt.")
        if tool_profile == "atomic":
            from tools.codex_atomic import AtomicTools
            if not self.motion_hook_enabled:
                raise ValueError("Atomic tools require the bundle_stage3 motion hook")
            self.atomic = AtomicTools(self)
            self.schemas = self.atomic.schemas
        self._write_status()

    def start_watchdog(self):
        # Inference time between MCP requests belongs to the episode deadline.
        self.timer = threading.Timer(max(0.001, self.runner.timeout_s - self.runner.elapsed_s), self._expire)
        self.timer.daemon = True
        self.timer.start()

    def _expire(self):
        self.runner.interrupt(code="episode_timeout", limit=self.runner.timeout_s,
                              observed=self.runner.elapsed_s, unit="seconds")
        self.close("episode_timeout")

    def status(self):
        remaining_internal = max(0, min(self.runner.max_turns-self.runner.turn_index,
                                        self.runner.max_tool_calls-self.runner.tool_call_count))
        return {
            "session_id": self.runtime.memory.session_id,
            "tool_profile": self.tool_profile,
            "execution_id": self.runner.execution_id,
            "controller_id": (self.runtime.memory.controller_capabilities() or {}).get("controller_id"),
            "terminated": self.runner.terminated, "truncated": self.runner.truncated,
            "waiting_for_human": self.runner.waiting_for_human,
            "failure_reason": self.runner.failure_reason,
            "turns": self.runner.turn_index, "tool_calls": self.runner.tool_call_count,
            "requests": self.requests, "elapsed_s": self.runner.elapsed_s,
            "limits": {"timeout_s": self.runner.timeout_s, "max_requests": self.max_requests,
                       "max_turns": self.runner.max_turns, "max_tool_calls": self.runner.max_tool_calls},
            "closed": self.closed, "cleanup": self.cleanup,
            "official_task_success": self.official_task_success,
            "completion_claim": self.completion_claim,
            "auxiliary_model_inference": "disabled",
            "model_usage": "external; see Codex events, not host token estimates",
            'execution_accounting': dict(self.execution_accounting),
            'remaining_budget': {'native_requests':max(0,self.max_requests-self.requests),
                'internal_stages':remaining_internal,
                'estimated_strict_moves':remaining_internal//3,
                'estimated_symmetric_moves':remaining_internal//5,
                'seconds':max(0.,self.runner.timeout_s-self.runner.elapsed_s),
                'estimate_scope':'uninterrupted full preflight+move sequences; routes cost per segment; time and other calls can reduce capacity'},
        }

    def _write_status(self):
        p = self.output / "host-status.json"
        temp = p.with_suffix(".tmp")
        temp.write_text(json.dumps(self.status(), ensure_ascii=False, indent=2) + "\n")
        temp.replace(p)

    def context(self):
        return build_tool_context(observation=self.runner.current_observation,
            memory=self.runtime.memory, tools=self.runtime.tools, skills=self.runtime.skills,
            config=self.runtime.planner.context_config)

    def success_evidence(self):
        # The planner context copies this same trusted receipt. Rebuilding image
        # and conversation context here adds work without changing the verdict.
        return episode_success_evidence({
            "session_id": self.runtime.memory.session_id,
            "metadata": self.runtime.memory.metadata,
            "steps": [{"step_result": self.runtime.memory.latest_environment_receipt() or {}}],
        }, env_id=str(self.runtime.memory.metadata.get("env_id") or ""))

    def publish_observation(self):
        # The ordinary runtime indexes an observation at the next act(). Codex
        # must already have its packet references while choosing that next act.
        # Publish the new result observation through the normal memory path;
        # repeated status calls must not manufacture additional observations.
        observation = self.runner.current_observation
        if observation is not self._published_observation:
            self.runtime.memory.add_observation(observation)
            self._published_observation = observation

    def response_receipt(self, result):
        body = json.loads(result.content[0].text)
        body['current_request'] = {'sequence':self.call_sequence, 'tool':self.request_tool}
        if self.request_tool == 'episode_status':
            body['previous_request'] = self.previous_response
        else:
            self.previous_response = {key:body[key] for key in (
                'current_request', 'feedback', 'interventions', 'error', 'robot', 'motion_hook', 'repair') if key in body}
        try:
            with (self.output / 'native-feedback.jsonl').open('a') as stream:
                stream.write(json.dumps({'current_request':body['current_request'],
                    'feedback':body.get('feedback'), 'interventions':body.get('interventions', []),
                    'error':body.get('error'), 'repair':body.get('repair')}, ensure_ascii=False)+'\n')
        except OSError:
            body['feedback_audit_status'] = 'write_failed'
        result.content[0].text = json.dumps(body, ensure_ascii=False)
        return result

    def record_intervention(self, source, effect, code, message, **kwargs):
        from tools.codex_interventions import event
        self.interventions.append(event(source, effect, code, message, **kwargs))

    def request_execution_state(self):
        if any(value is None for value in self.request_physics):
            return 'unknown'
        return 'partial' if any(self.request_physics) else 'not_started'

    def result(self, *, error=None, requested=None, executed=None, motion_hook=None):
        self.publish_observation()
        if error:
            messages = {
                'episode_not_active': 'The episode is inactive or cancellation was requested. This request was not dispatched; inspect episode state and limits.',
                'codex_request_limit': 'The native request budget was exhausted before this request could execute.',
                'invalid_arguments': 'The request failed input validation before dispatch. Correct the reported arguments.',
                'official_success_not_established': 'The success claim was rejected because current official evidence does not establish task success.',
                'host_execution_error': 'The Host encountered an execution error; completion or zero physical motion cannot be assumed.',
            }
            code = error.get('code', 'request_failed')
            self.record_intervention('host', 'request_failed', code,
                messages.get(code) or error.get('message') or 'The request was interrupted; inspect its current motion receipt and episode state.',
                execution_state=self.request_execution_state())
        if self.atomic is not None:
            return self.response_receipt(self.atomic.result(error=error))
        ctx = self.context()
        agent_ctx = dict(ctx.get("agent_context") or ctx)
        # Tool schemas are already delivered by MCP. Do not advertise unavailable
        # internal tools or duplicate skill bodies in every observation.
        for key in ("available_tools", "tool_references", "relevant_skills", "skill_usage"):
            agent_ctx.pop(key, None)
        visible = _planner_visible_main_agent_context(agent_ctx)
        body = {"episode": self.status(), "context": visible, "interventions": list(self.interventions)}
        body["observation_references"] = observation_references(self.runtime.memory)
        repair = repair_feedback(self.last_command, self.schemas)
        if repair is not None: body["repair"] = repair
        if error: body["error"] = error
        if requested: body["requested"] = requested
        if executed: body["executed"] = executed
        if motion_hook is not None: body["motion_hook"] = motion_hook
        parts, attachments = _planner_user_content(PlannerBackendRequest(tool_context=agent_ctx),
            OpenAICompatiblePlannerBackendConfig(max_vision_images=8))
        content = [TextContent(type="text", text=json.dumps(body, ensure_ascii=False))]
        attached = iter(a for a in attachments if a.get("attached") is True)
        image_index = 0
        for part in parts if isinstance(parts, list) else []:
            if part.get("type") != "image_url": continue
            url = part["image_url"]["url"]
            if url.startswith("data:image/") and ";base64," in url:
                header, data = url.split(",", 1)
                image_index += 1
                content.append(TextContent(type="text", text=image_label(
                    self.runtime.memory, next(attached), image_index)))
                content.append(ImageContent(type="image", mimeType=header[5:].split(";")[0], data=data))
        return self.response_receipt(CallToolResult(content=content, isError=bool(error)))

    def call(self, name, arguments):
        with self.lock:
            self.interventions, self.request_physics = [], []
            self.request_tool = name
            self.call_sequence += 1
            self.last_command = None
            if self.atomic is not None:
                self.atomic.feedback, self.atomic.overlay = None, {}
                self.atomic.crop, self.atomic.selected_views = None, None
            if self.stop_requested.is_set() or self.closed or self.runner.terminated or self.runner.truncated or self.runner.waiting_for_human:
                return self.result(error={"code": "episode_not_active"})
            self.requests += 1
            self.last_command = None
            if self.requests > self.max_requests:
                self.runner.interrupt(code="codex_request_limit", limit=self.max_requests,
                                      observed=self.requests, unit="requests")
                self.close("codex_request_limit")
                return self.result(error={"code": "codex_request_limit"})
            try:
                if name not in self.schemas: raise ValueError(f"Tool not exposed: {name}")
                jsonschema.validate(arguments, self.schemas[name].inputSchema)
            except (ValueError, jsonschema.ValidationError) as exc:
                self._write_status()
                return self.result(error={"code": "invalid_arguments", "message": str(exc)[:2500]})
            if name == "episode_status":
                self._write_status()
                return self.result()
            if self.atomic is not None and name in {"observe", "mark_point", "move_to", "gripper_control"}:
                return self.atomic.call(name, arguments)
            payload = {"kind": "tool_call", "name": name, "parameters": arguments}
            if name == "move_to" and self.motion_hook_enabled:
                command, error, hook = run_motion_hook(self, arguments)
                self._write_status()
                return self.result(error=error, requested=payload,
                                   executed=command.get("request"), motion_hook=hook)
            if name == "finish_episode":
                if arguments["success"] and not self.success_evidence():
                    return self.result(error={"code": "official_success_not_established"})
                payload = {"kind": "response", "name": "task_complete",
                           "parameters": {"success": arguments["success"], "message": arguments["reason"]}}
            command, error = self._execute(payload)
            return self.result(error=error, requested=payload, executed=command.get("request"))

    def _execute(self, payload, *, parent_request=None):
        """One ordinary budgeted runner step; internal hook stages are logged too."""
        with self.lock:
            if self.stop_requested.is_set() or self.closed or self.runner.terminated or self.runner.truncated or self.runner.waiting_for_human:
                return {}, {"code": "episode_not_active"}
            stage_context = self.context()
            _, errors = _decision_from_backend_result(PlannerBackendResult(payload=payload),
                tools=self.runtime.tools, skills=self.runtime.skills,
                tool_contract_catalog=self.runtime.planner.tool_contract_catalog,
                tool_contract_policy=self.runtime.planner.tool_contract_policy,
                tool_context=stage_context, agent_interface_profile=self.runtime.pipeline.agent_interface_profile)
            if errors:
                self.record_intervention('host', 'rejected', 'host_validation',
                    'The requested stage failed Host contract validation before dispatch.',
                    execution_state=self.request_execution_state(),
                    details={'stage':payload.get('name'), 'validation_messages':[str(e)[:500] for e in errors[:8]]})
                self._write_status()
                return {}, {"code": "host_validation", "messages": errors}
            self.backend.pending = payload
            physical = payload.get('name') in ('move_to', 'gripper_control')
            if physical:
                self.request_physics.append(None)
            try:
                step = self.runner.step()
            except Exception as exc:
                self.close("host_execution_error")
                return {}, {"code": "host_execution_error", "message": str(exc)[:1500]}
            finally:
                # A host invariant may execute a refresh/stop instead of consuming
                # this request. Never replay that stale decision on the next call.
                self.backend.pending = None
            command = step.action.command
            self.last_command = command
            for call in command.get('tool_calls', []):
                if call.get('status') not in ('executed', 'failed'):
                    continue
                tool_name = call.get('name')
                if tool_name in ('propose_motion_target', 'ik_preview_check'):
                    self.execution_accounting['preflight_calls'] += 1
                elif tool_name in ('move_to', 'gripper_control'):
                    from tools.codex_feedback import motion_feedback
                    self.execution_accounting['physical_dispatches'] += 1
                    executed_physics = motion_feedback(command, tool_name).get('physics_executed')
                    if executed_physics is True:
                        self.execution_accounting['physical_calls_with_steps'] += 1
                    elif executed_physics is None:
                        self.execution_accounting['physical_unknown_outcomes'] += 1
            executed = command.get("request", {})
            if physical:
                from tools.codex_feedback import motion_feedback
                calls = [c for c in command.get('tool_calls', [])
                         if c.get('name') == payload.get('name') and c.get('status') in ('executed', 'failed')]
                self.request_physics[-1] = (motion_feedback(command, payload['name']).get('physics_executed')
                                           if calls else False)
            if executed.get('name') != payload.get('name') or executed.get('kind') != payload.get('kind'):
                code = 'host_invariant_substitution'
                message = 'A Host gate substituted another operation for the requested stage; its specific reason is unavailable. Inspect the episode state before replanning.'
                if isinstance(stage_context.get('motion_reconciliation'), dict):
                    code = 'transport_outcome_unknown'
                    message = ('A previous simulator action has unknown remote completion. The Host stopped this episode instead of dispatching the requested stage. '
                               'Do not resend that action or infer completion from position. Host-confirmed environment retirement and a new session are required.')
                elif (stage_context.get('fresh_observation_obligation') or {}).get('required') is True:
                    code = 'fresh_observation_required'
                    message = ('The previous world mutation returned no fresh observation. The Host substituted a same-environment observation refresh before permitting further control. '
                               'Inspect the new observation and submit a new plan; the requested stage was not replayed automatically.')
                self.record_intervention('host', 'substituted', code,
                    message + ' The remaining motion sequence stops; this is not target arrival.',
                    execution_state=self.request_execution_state(),
                    details={'requested_stage': payload.get('name'),
                             'executed_stage': executed.get('name') if executed.get('name') in {*self.schemas, 'task_complete', 'propose_motion_target', 'ik_preview_check'} else 'internal_operation'})
            metadata = command.get('metadata') or {}
            for gate, message in (
                ('interface_profile_gate', 'The request uses parameters unavailable in this interface profile. Use the exposed tool schema.'),
                ('bundle_resolution_gate', 'The supplied evidence bundle could not be resolved for this operation. Refresh the required evidence before replanning.'),
            ):
                if (metadata.get(gate) or {}).get('blocked') is True:
                    self.record_intervention('host', 'rejected', gate, message,
                        execution_state=self.request_execution_state())
            self.official_task_success = bool(self.success_evidence())
            if executed.get("name") == "task_complete":
                self.completion_claim = executed.get("parameters", {}).get("success")
            with (self.output / "host-commands.jsonl").open("a") as f:
                f.write(json.dumps({"requested": payload, "command": command,
                    **({"parent_request": parent_request} if parent_request is not None else {}),
                    "reward": step.step_result.reward, "info": step.step_result.info}, ensure_ascii=False) + "\n")
            if self.runner.turn_index >= self.runner.max_turns and not self.runner.terminated:
                self.runner.interrupt(code="codex_turn_limit", limit=self.runner.max_turns,
                                      observed=self.runner.turn_index, unit="turns")
            if self.runner.terminated or self.runner.truncated or self.runner.waiting_for_human:
                self.close("episode_terminal")
            self._write_status()
            error = execution_error(command)
            return command, error

    def request_stop(self):
        # Does not wait for the execution lock. A cancelled MCP request must stop
        # the next route segment even while its current worker call is finishing.
        self.stop_requested.set()

    def close(self, reason="transport_closed"):
        self.request_stop()
        with self.lock:
            if self.closed: return
            self.closed = True
            if self.timer: self.timer.cancel()
            if not (self.runner.terminated or self.runner.truncated):
                self.runner.interrupt(code=reason)
            close = getattr(self.runner.environment, "close", None)
            try:
                self.cleanup = close() if close else {"ok": True, "mode": "no_external_environment"}
            except Exception as exc:
                self.cleanup = {"ok": False, "error": str(exc)}
            self._write_status()


def build_host(args):
    output = args.output.resolve()
    session = uuid4().hex
    workspace = SessionWorkspace.create(session, root=output / "workspace", environment_id=args.env_id,
        source_task_playbooks=output / "empty-playbooks", source_grasp_strategies=output / "empty-strategies")
    transport = SseSimulatorMcpTransport(args.sim_url)
    proxy = SimulatorMcpToolProxyConfig(timeout_s=120,
        image_output_root=workspace.artifacts_dir / "images",
        text_output_root=workspace.artifacts_dir / "text", response_output_root=workspace.artifacts_dir / "responses")
    environment = SimulatorMcpEpisodeEnvironment(transport=transport,
        config=SimulatorMcpEpisodeConfig(env_id=args.env_id, seed=args.seed,
            image_output_root=workspace.artifacts_dir / "images", timeout_s=120), tool_proxy_config=proxy)
    assembly = assemble_runtime(RuntimeAssemblyConfig(workspace=workspace,
        provider=PlannerProviderConfig(provider="codex-mcp", model="external"),
        backend_factory=lambda **kwargs: DisabledModelBackend(),
        supervision_policy=SupervisionPolicy.for_profile("standard"),
        simulator_transport=transport, simulator_proxy_config=proxy,
        endpoints=RuntimeMcpEndpoints(sam3_url=args.sam3_url if args.tool_profile == "pick" else "",
                                     anygrasp_url=args.anygrasp_url if args.tool_profile == "pick" else ""),
        web_access_config=WebAccessConfig(), visual_history=VisualHistoryConfig(enabled=False),
        grasp_pose_advisor_enabled=False, max_validation_retries=0,
        agent_interface_profile="bundle_stage3"))
    runtime = assembly.runtime
    backend = SubmittedBackend()
    runtime.planner.backend = backend
    runtime.planner.context_config = replace(runtime.planner.context_config, auto_compact_enabled=False)
    runner = OpenEtaEpisodeRunner(runtime=runtime, environment=environment)
    try:
        runner.start(task=args.task, max_turns=args.max_requests, max_tool_calls=args.max_requests,
                     timeout_s=args.timeout, metadata={"env_id": args.env_id, "require_official_reward": True})
        runtime.memory.add_observation(runner.current_observation)
        require_controller(runtime.memory.controller_capabilities(), args.expected_controller)
        host = CodexHost(runner, backend, output=output, max_requests=args.max_requests,
                         tool_profile=args.tool_profile)
        host.start_watchdog()
        return host
    except BaseException:
        environment.close()
        raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sim-url", required=True)
    p.add_argument("--tool-profile", choices=("pick", "atomic"), default="pick")
    p.add_argument("--expected-controller", choices=CONTROLLER_IDS,
                   help="Verify the connected simulator's actual controller before serving tools")
    p.add_argument("--env-id", default="openeta/libero_libero_spatial_task0-v0")
    p.add_argument("--task", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sam3-url", default="")
    p.add_argument("--anygrasp-url", default="")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--timeout", type=float, default=1200)
    p.add_argument("--max-requests", type=int, default=80)
    return p
