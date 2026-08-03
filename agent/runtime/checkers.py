"""Checker hook primitives for optional OpenETA sub-agent integration."""

from __future__ import annotations

from dataclasses import dataclass, field

from adapter.protocol import JsonDict
from agent.runtime.actions import CommandKind, PipelineCall, PipelineStatus


CHECKER_RESULT_SCHEMA_VERSION = "openeta.checker_result.v1"


@dataclass(frozen=True, slots=True)
class CheckerSubagentConfig:
    """Configuration for lightweight checker hooks around selected tool calls.

    The current implementation keeps checker outputs inside pipeline metadata
    and existing `PipelineCall` records. This reserves the sub-agent boundary
    without locking a final SafetyVerdict/FailureVerdict schema.
    """

    pre_safety_checks: dict[str, str] = field(default_factory=dict)
    post_failure_checks: tuple[str, ...] = ()
    failure_checker_name: str = "failure_check"

    def safety_tool_for(self, target_tool: str) -> str | None:
        return self.pre_safety_checks.get(target_tool)

    def should_run_failure_check(self, target_tool: str) -> bool:
        return target_tool in self.post_failure_checks


def safety_check_parameters(
    *,
    checker_tool: str,
    target_tool: str,
    target_parameters: JsonDict,
) -> JsonDict:
    """Build parameters passed to a pre-tool safety checker."""

    return {
        **dict(target_parameters),
        "checker_tool": checker_tool,
        "target_tool": target_tool,
        "target_parameters": dict(target_parameters),
    }


def build_failure_check_call(
    *,
    checker_name: str,
    target_call: PipelineCall,
) -> PipelineCall:
    """Build a post-tool failure-check record from the completed tool call."""

    result = target_call.result if isinstance(target_call.result, dict) else {}
    success = bool(result.get("success", target_call.status == PipelineStatus.EXECUTED))
    verdict = "ok" if success and target_call.status == PipelineStatus.EXECUTED else "failed"
    diagnostics: list[JsonDict] = []
    if verdict == "failed":
        diagnostics.append(
            {
                "code": "tool_call_failed",
                "target_tool": target_call.name,
                "tool_status": target_call.status.value,
            }
        )
    return PipelineCall(
        kind=CommandKind.TOOL_CALL,
        name=checker_name,
        parameters={
            "target_tool": target_call.name,
            "target_status": target_call.status.value,
        },
        status=PipelineStatus.EXECUTED,
        result={
            "success": verdict == "ok",
            "content": (
                "Failure checker found no issue."
                if verdict == "ok"
                else "Failure checker detected a failed tool call."
            ),
            "details": {
                "schema_version": CHECKER_RESULT_SCHEMA_VERSION,
                "checker": checker_name,
                "checker_type": "failure",
                "target_tool": target_call.name,
                "target_status": target_call.status.value,
                "verdict": verdict,
                "tool_result": result,
                "diagnostics": diagnostics,
            },
        },
        reason="Post-tool failure checker hook.",
    )


def should_record_recovery_feedback(status: PipelineStatus) -> bool:
    """Return whether a pipeline outcome should be highlighted for replanning."""

    return status in {PipelineStatus.BLOCKED, PipelineStatus.FAILED}
