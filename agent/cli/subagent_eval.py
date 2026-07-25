"""Explicit live-provider evaluation for isolated OpenETA sub-agent roles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adapter.protocol import JsonDict
from agent.backends.planner import (
    OpenAICompatiblePlannerBackend,
    OpenAICompatiblePlannerBackendConfig,
)
from agent.backends.provider_config import load_planner_provider_config
from agent.evals.subagents import (
    SUBAGENT_ROLES,
    default_subagent_eval_cases,
    run_subagent_evaluation,
)
from agent.runtime.skill_authoring import SKILL_AUTHORING_MAX_OUTPUT_TOKENS


def main() -> None:
    cases = default_subagent_eval_cases()
    case_ids = {case.case_id for case in cases}
    parser = argparse.ArgumentParser(
        description="Run fixed cases through OpenETA's production sub-agent adapters."
    )
    parser.add_argument("--role", action="append", choices=SUBAGENT_ROLES)
    parser.add_argument("--case", action="append", dest="case_ids")
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--model", default="", help="Override OPENETA_LLM_MODEL.")
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--output", default="", help="Optional relative JSON output path.")
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero when any case fails or a critical error is observed.",
    )
    args = parser.parse_args()

    requested_case_ids = set(args.case_ids or ())
    unknown_case_ids = requested_case_ids - case_ids
    if unknown_case_ids:
        raise SystemExit("Unknown sub-agent eval cases: " + ", ".join(sorted(unknown_case_ids)))
    selected_roles = tuple(args.role or SUBAGENT_ROLES)
    selected_cases = [
        case
        for case in cases
        if case.role in selected_roles
        and (not requested_case_ids or case.case_id in requested_case_ids)
    ]
    if args.list_cases:
        print(json.dumps([case.to_dict() for case in selected_cases], indent=2))
        return

    provider = load_planner_provider_config()
    if args.model:
        provider.model = args.model
    if args.timeout_s is not None:
        provider.timeout_s = args.timeout_s
    missing = provider.missing_fields()
    if missing:
        raise SystemExit("Missing provider fields: " + ", ".join(missing))

    def backend_factory(role: str) -> OpenAICompatiblePlannerBackend:
        config = OpenAICompatiblePlannerBackendConfig.from_provider_config(provider)
        config.max_tokens = (
            SKILL_AUTHORING_MAX_OUTPUT_TOKENS if role == "skill_author" else 512
        )
        return OpenAICompatiblePlannerBackend(config)

    report = run_subagent_evaluation(
        backend_factory,
        cases=selected_cases,
        repeats=args.repeat,
    )
    report["provider"] = provider.redacted()
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        _write_output(args.output, report)
    metrics = report["metrics"]
    if args.strict and (
        metrics["failed"]
        or metrics["critical_false_approval_count"]
        or metrics["unsupported_guidance_answer_count"]
    ):
        raise SystemExit(1)


def _write_output(path: str, report: JsonDict) -> None:
    output = Path(path)
    if output.is_absolute() or ".." in output.parts:
        raise ValueError("--output must be a relative path inside the repository")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
