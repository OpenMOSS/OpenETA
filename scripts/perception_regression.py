"""Opt-in LIBERO regression with the configured real LLM; no motion by default.

Run as a module. Requires an independently owned simulator on port 18766.
Only newly generated task data is used; no historical memory/skills are imported.
Object 0 supports explicit bounded task motion; success requires official evidence.
"""
from __future__ import annotations

import argparse
import json
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from agent.backends.planner import OpenAICompatiblePlannerBackend, OpenAICompatiblePlannerBackendConfig
from agent.backends.provider_config import load_planner_provider_config
from agent.runtime.episode import OpenEtaEpisodeRunner, ToolFeedbackEpisodeEnvironment
from agent.runtime.parallel import classify_episode_result
from adapter.environment_lifecycle import close_response_error
from agent.runtime.runtime_assembly import RuntimeAssemblyConfig, RuntimeMcpEndpoints, assemble_runtime
from agent.runtime.self_improvement import SelfImprovementConfig, SelfImprovementReviewer
from agent.runtime.session_workspace import SessionWorkspace
from agent.runtime.supervision import SupervisionPolicy
from agent.runtime.visual_history import VisualHistoryConfig
from agent.tools.sim_mcp import SimulatorMcpToolProxyConfig, SseSimulatorMcpTransport, close_simulator_mcp_env
from agent.tools.web_access import WebAccessConfig

TASKS = {
    "object0": ("openeta/libero_libero_object_task0-v0", "pick up the alphabet soup and place it in the basket"),
    "spatial0": ("openeta/libero_libero_spatial_task0-v0", "pick up the black bowl between the plate and the ramekin and place it on the plate"),
    "long9": ("openeta/libero_libero_10_task9-v0", "put the yellow and white mug in the microwave and close it"),
}
PERCEPTION_TOOLS = frozenset({
    "create_simulator_env", "observe", "sam3", "select_sam3_detection",
    "reject_sam3_detections", "grasp_pose_estimate", "compile_grasp_seed", "camera_pose_to_world",
    "compute_wrist_alignment", "propose_wrist_viewpoints", "ik_preview_check",
})
TASK_TOOLS = PERCEPTION_TOOLS | {
    "move_to", "follow_eef_trajectory", "gripper_control", "anyplace",
    "prepare_attachment_probe", "assess_attachment_probe", "save_memory", "get_memory",
}


class DiagnosticTransport:
    def __init__(self, *, allow_motion=False):
        self.delegate = SseSimulatorMcpTransport("http://127.0.0.1:18766/sse")
        self.calls = []
        self.created = False
        self.allow_motion = allow_motion
        self.reserved_controller_steps = 0

    def call_tool(self, name, arguments, *, timeout_s=None):
        motion_tools = {"move_to", "follow_eef_trajectory", "gripper_open", "gripper_close"}
        allowed = {"create_env", "reset_env", "render_env", "observe_env", "close_env", "ik_preview_check"}
        if self.allow_motion:
            allowed |= motion_tools
        if name not in allowed:
            raise RuntimeError("No-motion diagnostic refuses simulator operation: " + name)
        steps = 0
        if name in {"move_to", "follow_eef_trajectory"}:
            if arguments.get("enable_collision_check", True) is not True:
                raise RuntimeError("Task regression requires collision checks")
            if name == "move_to":
                steps = arguments.get("num_steps", 150)
                if type(steps) is not int or not 1 <= steps <= 200:
                    raise RuntimeError("Task regression move budget is 1..200 steps")
            else:
                per_waypoint = arguments.get("num_steps_per_waypoint", 60)
                route = arguments.get("trajectory")
                if (type(per_waypoint) is not int or not 1 <= per_waypoint <= 100
                        or not isinstance(route, list) or not 1 <= len(route) <= 5):
                    raise RuntimeError("Task regression route budget is 1..5 waypoints at 1..100 steps each")
                steps = per_waypoint * len(route)
        elif name in {"gripper_open", "gripper_close"}:
            steps = 40 if name == "gripper_open" else 60
        if self.reserved_controller_steps + steps > 6000:
            raise RuntimeError("Task regression total controller admission budget exhausted")
        self.reserved_controller_steps += steps
        if name == "create_env":
            if self.created:
                raise RuntimeError("Diagnostic permits only one create attempt")
            self.created = True
        self.calls.append({"name": name, "arguments": arguments})
        return self.delegate.call_tool(name, arguments, timeout_s=timeout_s)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--allow-authorized-test-data", action="store_true")
    parser.add_argument("--episode-seconds", type=int, default=300)
    parser.add_argument("--allow-task-motion", action="store_true", help="Object 0 only; explicitly enable bounded full-task control")
    args = parser.parse_args()
    if not args.allow_authorized_test_data:
        parser.error("Explicit authorization for the configured LLM and specified perception services is required")
    if args.allow_task_motion and args.task != "object0":
        parser.error("Task motion is currently supported only for Object 0 regression")
    if not 30 <= args.episode_seconds <= (1800 if args.allow_task_motion else 600):
        parser.error("episode-seconds must be 30..600 (30..1800 in explicit Object 0 task mode)")
    turns, tool_limit, provider_limit = (80, 80, 128) if args.allow_task_motion else (8, 7, 12)
    provider = load_planner_provider_config()
    provider.fallback = None
    provider.timeout_s = 180 if args.allow_task_motion else 60
    provider.max_attempts = 1
    provider.retry_backoff_s = 0
    if provider.missing_fields():
        parser.error("Primary provider configuration is incomplete")
    root = Path(tempfile.mkdtemp(prefix="perception-regression-", dir="tmp")).resolve()
    empty = root / "empty-inputs"
    empty.mkdir()
    env_id, instruction = TASKS[args.task]
    session_id = str(uuid4())
    workspace = SessionWorkspace.create(
        session_id, root=root / "memory", environment_id=env_id,
        source_skills=empty, source_task_playbooks=empty,
    )
    proxy = SimulatorMcpToolProxyConfig(
        timeout_s=90 if args.allow_task_motion else 45, image_output_root=workspace.artifacts_dir / "images",
        text_output_root=workspace.artifacts_dir / "text", response_output_root=workspace.artifacts_dir / "responses",
    )
    transport = DiagnosticTransport(allow_motion=args.allow_task_motion)
    started = time.monotonic()
    request_lock = threading.Lock()
    provider_calls = 0

    class BudgetedBackend(OpenAICompatiblePlannerBackend):
        def decide(self, request):
            nonlocal provider_calls
            with request_lock:
                if provider_calls >= provider_limit or time.monotonic() - started >= args.episode_seconds:
                    raise RuntimeError("Diagnostic provider admission budget exhausted")
                provider_calls += 1
                print(json.dumps({"event": "provider_admitted", "number": provider_calls}), flush=True)
            return super().decide(request)

    def backend_factory(**kwargs):
        config = OpenAICompatiblePlannerBackendConfig.from_provider_config(provider)
        for key, value in kwargs.items():
            if value is not None:
                setattr(config, key, value)
        config.max_tokens = min(config.max_tokens, 8192)
        return BudgetedBackend(config)

    report = {
        "experiment": "real_llm_task" if args.allow_task_motion else "real_llm_perception_only", "task": args.task,
        "env_id": env_id, "agent_session_id": session_id, "task_success_claimed": False,
        "provider": {"model": provider.model, "host": urlsplit(provider.api_base).hostname, "fallback": False},
        "limits": {"turns": turns, "tool_admissions": tool_limit, "provider_calls": provider_limit, "episode_seconds": args.episode_seconds,
                   "provider_timeout_seconds": provider.timeout_s, "provider_output_tokens_per_call": 8192,
                   "controller_admission_steps": 6000 if args.allow_task_motion else 0},
    }
    print(json.dumps({"output": str(root), "session_id": session_id}), flush=True)
    try:
        assembly = assemble_runtime(RuntimeAssemblyConfig(
            workspace=workspace, provider=provider, backend_factory=backend_factory,
            supervision_policy=SupervisionPolicy.for_profile("standard"),
            endpoints=RuntimeMcpEndpoints(sam3_url="http://10.11.39.173:8773/sse", anygrasp_url="http://10.11.39.173:8774/sse",
                anyplace_url="http://10.11.39.173:8775/sse" if args.allow_task_motion else ""),
            simulator_transport=transport, simulator_proxy_config=proxy,
            web_access_config=WebAccessConfig(), visual_history=VisualHistoryConfig(enabled=False),
            perception_tool_timeout_s=120 if args.allow_task_motion else 60, max_validation_retries=2,
        ))
        runtime = assembly.runtime
        runtime.self_improvement_reviewer = SelfImprovementReviewer(config=SelfImprovementConfig(enabled=False))
        for spec in runtime.tools.list():
            if spec.name not in (TASK_TOOLS if args.allow_task_motion else PERCEPTION_TOOLS):
                runtime.tools.unbind_handler(spec.name)
        report["perception_capabilities"] = assembly.perception_capabilities
        runner = OpenEtaEpisodeRunner(runtime=runtime, environment=ToolFeedbackEpisodeEnvironment())
        task = (f"完成完整 LIBERO Object 0 pick-place 回归：新建 {env_id}，seed 0。任务：{instruction}。"
                "通过真实感知和注册工具自主完成抓取、搬运、放置；只能依据本次环境官方成功证据 task_complete。"
                "先做 IK/碰撞检查，保留闭环观察；仅创建一次环境。未知运动结果必须停止，不重复执行。"
                "单次 move 最多200步，路径最多5点且每点100步；无需总用满预算。"
                "若必要能力不可用或任务失败，请用带非空 message 的 talk 明确说明。"
                if args.allow_task_motion else
                (f"有界感知诊断：新建 {env_id}，seed 0。语义基准：{instruction}。"
                  "本轮仅验证观察、SAM3、必要的 grasp_pose_estimate 与抓取几何。"
                  "运动、Python、外部资料、技能修改均禁用；不要创建第二个环境。"
                  "在固定 agentview 首次消歧；遮挡或候选不足时报告证据并 talk 停止，不声称完成操作任务。"))
        result = runner.run(
            task=task, session_id=session_id, max_turns=turns, max_tool_calls=tool_limit, timeout_s=args.episode_seconds,
            max_total_tokens=1500000 if args.allow_task_motion else 300000,
            metadata={"env_id": env_id, "require_official_reward": True, "experiment": report["experiment"]},
        )
        report["episode"] = result.to_dict()
        report["task_outcome"] = classify_episode_result(result, env_id=env_id, require_official_reward=True)
    except Exception as exc:
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        report["remote_identity"] = {"handle": proxy.handle, "session_id": proxy.session_id}
        try:
            report["cleanup"] = (close_simulator_mcp_env(transport, handle=proxy.handle, session_id=proxy.session_id, timeout_s=30)
                                 if proxy.handle else (
                                     {"ok": False, "reason": "create_attempt_without_returned_handle"}
                                     if transport.created else {"ok": True, "skipped": True, "reason": "no_create_attempt"}
                                 ))
        except Exception as exc:
            report["cleanup"] = {"ok": False, "error_type": type(exc).__name__, "message": str(exc)}
        report["remote_calls"] = transport.calls
        report["provider_calls_admitted"] = provider_calls
        report["controller_steps_reserved"] = transport.reserved_controller_steps
        report["task_success_claimed"] = bool(args.allow_task_motion and report.get("task_outcome") == "success"
                                               and close_response_error(report["cleanup"]) is None)
        report["elapsed_s"] = time.monotonic() - started
        (root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(json.dumps({"report": str(root / "report.json"), "cleanup": report["cleanup"]}), flush=True)


if __name__ == "__main__":
    main()
