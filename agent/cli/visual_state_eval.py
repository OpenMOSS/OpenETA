"""Run explicit live-provider evaluations of visual state understanding."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from adapter.protocol import JsonDict
from agent.backends.planner import (
    REASONING_SUBAGENT_MAX_OUTPUT_TOKENS,
    OpenAICompatiblePlannerBackend,
    OpenAICompatiblePlannerBackendConfig,
)
from agent.backends.provider_config import load_planner_provider_config
from agent.evals.visual_state import (
    VISUAL_STATE_PROBE_SCHEMA_VERSION,
    load_visual_state_eval_cases,
    run_visual_state_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure whether planner state assessments change with visual evidence."
    )
    parser.add_argument("--manifest", required=True, help="Relative JSON case manifest path.")
    parser.add_argument("--model", default="", help="Override OPENETA_LLM_MODEL.")
    parser.add_argument("--timeout-s", type=float, default=None)
    parser.add_argument("--output", default="", help="Optional relative JSON output path.")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--list-cases", action="store_true")
    args = parser.parse_args()

    manifest_path = _repository_path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cases = load_visual_state_eval_cases(manifest)
    if args.list_cases:
        print(json.dumps([case.to_dict() for case in cases], ensure_ascii=False, indent=2))
        return

    provider = load_planner_provider_config()
    if args.model:
        provider.model = args.model
    if args.timeout_s is not None:
        provider.timeout_s = args.timeout_s
    missing = provider.missing_fields()
    if missing:
        raise SystemExit("Missing provider fields: " + ", ".join(missing))
    config = OpenAICompatiblePlannerBackendConfig.from_provider_config(provider)
    config.max_tokens = REASONING_SUBAGENT_MAX_OUTPUT_TOKENS
    backend = OpenAICompatiblePlannerBackend(config)
    report = run_visual_state_evaluation(backend, cases=cases)
    report["provider"] = _provider_descriptor(provider.redacted())
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        _write_output(args.output, report)
    metrics = report["metrics"]
    if args.strict and (
        metrics["failed"]
        or metrics["correct_counterfactual_flip_count"]
        != metrics["expected_flip_group_count"]
    ):
        raise SystemExit(1)


def _repository_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("paths must be relative to the repository")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _provider_descriptor(value: JsonDict) -> JsonDict:
    """Keep useful provider provenance without persisting secret fragments."""

    descriptor = dict(value)
    if "api_key" in descriptor:
        descriptor["api_key"] = "<redacted>"
    fallback = descriptor.get("fallback")
    if isinstance(fallback, dict):
        descriptor["fallback"] = _provider_descriptor(fallback)
    return descriptor


def _write_output(path: str, report: JsonDict) -> None:
    output = Path(path)
    if output.is_absolute() or ".." in output.parts:
        raise ValueError("--output must be a relative path inside the repository")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
