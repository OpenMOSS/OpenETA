"""Extract ordered toolchain coverage from durable OpenETA rollouts.

This is deliberately an experiment-side extractor rather than part of the
universal eval reducer.  A sequence spec describes observable milestones in
``tool_calls.jsonl``; the extractor reports which rollouts exercised the chain
and where incomplete rollouts stopped without changing Agent prompts or policy.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from adapter.protocol import JsonDict


SCHEMA_VERSION = "openeta.toolchain_coverage.v1"
SPEC_SCHEMA_VERSION = "openeta.toolchain_coverage_spec.v1"


def extract_toolchain_coverage(
    path: str | Path,
    sequences: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Evaluate ordered tool milestones across one or many durable rollouts."""

    normalized_sequences = _normalize_sequences(sequences)
    sources = _discover_tool_event_sources(Path(path))
    rollout_reports: list[JsonDict] = []
    aggregate_tool_counts: Counter[str] = Counter()

    for source in sources:
        calls = list(_read_end_calls(source))
        tool_counts = Counter(str(call["name"]) for call in calls)
        aggregate_tool_counts.update(tool_counts)
        sequence_reports = [
            _match_sequence(calls, sequence) for sequence in normalized_sequences
        ]
        rollout_reports.append(
            {
                "source": str(source),
                "session_id": _session_id_from_source(source),
                "tool_call_count": len(calls),
                "tool_counts": dict(sorted(tool_counts.items())),
                "sequences": sequence_reports,
            }
        )

    aggregate_sequences: list[JsonDict] = []
    for sequence_index, sequence in enumerate(normalized_sequences):
        reports = [
            rollout["sequences"][sequence_index] for rollout in rollout_reports
        ]
        complete_sources = [
            rollout_reports[index]["source"]
            for index, report in enumerate(reports)
            if report["complete"]
        ]
        frontier_counts = Counter(
            str(report["missing_next_milestone_id"])
            for report in reports
            if not report["complete"] and report["missing_next_milestone_id"]
        )
        best_index = max(
            range(len(reports)),
            key=lambda index: (
                int(reports[index]["matched_count"]),
                -int(reports[index].get("span_call_count") or 0),
            ),
            default=None,
        )
        aggregate_sequences.append(
            {
                "id": sequence["id"],
                "milestone_count": len(sequence["milestones"]),
                "complete_rollout_count": len(complete_sources),
                "completion_rate": (
                    len(complete_sources) / len(rollout_reports)
                    if rollout_reports
                    else 0.0
                ),
                "complete_sources": complete_sources,
                "frontier_counts": dict(sorted(frontier_counts.items())),
                "best_partial": (
                    {
                        "source": rollout_reports[best_index]["source"],
                        "session_id": rollout_reports[best_index]["session_id"],
                        "matched_count": reports[best_index]["matched_count"],
                        "matched_milestone_ids": reports[best_index][
                            "matched_milestone_ids"
                        ],
                        "missing_next_milestone_id": reports[best_index][
                            "missing_next_milestone_id"
                        ],
                    }
                    if best_index is not None
                    else None
                ),
            }
        )

    return {
        "schema_version": SCHEMA_VERSION,
        "source_root": str(Path(path)),
        "rollout_count": len(rollout_reports),
        "tool_call_count": sum(
            int(rollout["tool_call_count"]) for rollout in rollout_reports
        ),
        "tool_counts": dict(sorted(aggregate_tool_counts.items())),
        "sequence_summaries": aggregate_sequences,
        "rollouts": rollout_reports,
    }


def load_toolchain_spec(path: str | Path) -> tuple[JsonDict, ...]:
    """Load and validate an ``openeta.toolchain_coverage_spec.v1`` file."""

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"toolchain spec must be a JSON object: {source}")
    schema_version = payload.get("schema_version")
    if schema_version != SPEC_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported toolchain spec schema_version {schema_version!r}; "
            f"expected {SPEC_SCHEMA_VERSION!r}"
        )
    sequences = payload.get("sequences")
    if not isinstance(sequences, list):
        raise ValueError("toolchain spec sequences must be a list")
    return _normalize_sequences(sequences)


def _normalize_sequences(
    sequences: Sequence[Mapping[str, Any]],
) -> tuple[JsonDict, ...]:
    if not sequences:
        raise ValueError("at least one toolchain sequence is required")
    normalized: list[JsonDict] = []
    seen_ids: set[str] = set()
    for sequence_index, raw_sequence in enumerate(sequences):
        if not isinstance(raw_sequence, Mapping):
            raise ValueError(f"sequence {sequence_index} must be an object")
        sequence_id = str(raw_sequence.get("id") or "").strip()
        if not sequence_id:
            raise ValueError(f"sequence {sequence_index} requires a non-empty id")
        if sequence_id in seen_ids:
            raise ValueError(f"duplicate toolchain sequence id: {sequence_id}")
        seen_ids.add(sequence_id)
        raw_milestones = raw_sequence.get("milestones")
        if not isinstance(raw_milestones, (list, tuple)) or not raw_milestones:
            raise ValueError(f"sequence {sequence_id!r} requires milestones")
        milestones = tuple(
            _normalize_milestone(sequence_id, index, milestone)
            for index, milestone in enumerate(raw_milestones)
        )
        max_gap = raw_sequence.get("max_gap")
        if max_gap is not None and (
            not isinstance(max_gap, int) or isinstance(max_gap, bool) or max_gap < 0
        ):
            raise ValueError(
                f"sequence {sequence_id!r} max_gap must be a non-negative integer"
            )
        normalized.append(
            {
                "id": sequence_id,
                "description": str(raw_sequence.get("description") or ""),
                "max_gap": max_gap,
                "milestones": milestones,
            }
        )
    return tuple(normalized)


def _normalize_milestone(
    sequence_id: str,
    milestone_index: int,
    raw_milestone: object,
) -> JsonDict:
    if isinstance(raw_milestone, str):
        raw: Mapping[str, Any] = {"tool": raw_milestone}
    elif isinstance(raw_milestone, Mapping):
        raw = raw_milestone
    else:
        raise ValueError(
            f"sequence {sequence_id!r} milestone {milestone_index} must be a "
            "tool name or object"
        )
    raw_tools = raw.get("tools", raw.get("tool"))
    if isinstance(raw_tools, str):
        tools = (raw_tools.strip(),)
    elif isinstance(raw_tools, (list, tuple)):
        tools = tuple(str(tool).strip() for tool in raw_tools)
    else:
        tools = ()
    if not tools or any(not tool for tool in tools):
        raise ValueError(
            f"sequence {sequence_id!r} milestone {milestone_index} requires tool/tools"
        )
    where = raw.get("where", {})
    if not isinstance(where, Mapping):
        raise ValueError(
            f"sequence {sequence_id!r} milestone {milestone_index} where must be an object"
        )
    any_of = raw.get("any_of", [])
    if not isinstance(any_of, list) or any(
        not isinstance(branch, Mapping) for branch in any_of
    ):
        raise ValueError(
            f"sequence {sequence_id!r} milestone {milestone_index} any_of must be "
            "a list of condition objects"
        )
    capture = raw.get("capture", {})
    if not isinstance(capture, Mapping) or any(
        not str(binding).strip() or not isinstance(path, str) or not path.strip()
        for binding, path in capture.items()
    ):
        raise ValueError(
            f"sequence {sequence_id!r} milestone {milestone_index} capture must be "
            "an object mapping binding names to field paths"
        )
    milestone_id = str(raw.get("id") or tools[0]).strip()
    return {
        "id": milestone_id,
        "tools": tools,
        "where": dict(where),
        "any_of": [dict(branch) for branch in any_of],
        "capture": {str(binding): path for binding, path in capture.items()},
    }


def _read_end_calls(path: Path) -> Iterable[JsonDict]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                continue
            event = row.get("event")
            if not isinstance(event, dict) or event.get("phase") != "end":
                continue
            details = event.get("details")
            details = details if isinstance(details, dict) else {}
            outputs = details.get("outputs")
            outputs = outputs if isinstance(outputs, dict) else {}
            parameters = event.get("parameters")
            parameters = parameters if isinstance(parameters, dict) else {}
            yield {
                "seq": int(row.get("seq") or 0),
                "name": str(event.get("name") or ""),
                "success": event.get("success"),
                "effect": str(event.get("effect") or ""),
                "parameters": parameters,
                "details": details,
                "outputs": outputs,
                "operational_success": details.get("operational_success"),
                "semantic_outcome": str(details.get("semantic_outcome") or ""),
            }


def _match_sequence(calls: Sequence[JsonDict], sequence: JsonDict) -> JsonDict:
    milestones = sequence["milestones"]
    attempts: list[tuple[list[int], JsonDict]] = []
    first = milestones[0]
    for call_index, call in enumerate(calls):
        bindings = _match_and_capture(call, first, {})
        if bindings is None:
            continue
        matched = [call_index]
        previous_index = call_index
        for milestone in milestones[1:]:
            next_match = _find_next_match(
                calls,
                milestone,
                after_index=previous_index,
                max_gap=sequence["max_gap"],
                bindings=bindings,
            )
            if next_match is None:
                break
            next_index, bindings = next_match
            matched.append(next_index)
            previous_index = next_index
        attempts.append((matched, bindings))

    best, best_bindings = max(
        attempts,
        key=lambda attempt: (
            len(attempt[0]),
            -(attempt[0][-1] - attempt[0][0] if attempt[0] else 0),
            -attempt[0][0] if attempt[0] else 0,
        ),
        default=([], {}),
    )
    matched_calls = [calls[index] for index in best]
    matched_count = len(best)
    complete = matched_count == len(milestones)
    return {
        "id": sequence["id"],
        "complete": complete,
        "coverage": matched_count / len(milestones),
        "matched_count": matched_count,
        "milestone_count": len(milestones),
        "matched_milestone_ids": [
            milestones[index]["id"] for index in range(matched_count)
        ],
        "matched_calls": [
            {
                "milestone_id": milestones[index]["id"],
                "seq": call["seq"],
                "tool": call["name"],
                "success": call["success"],
                "semantic_outcome": call["semantic_outcome"],
            }
            for index, call in enumerate(matched_calls)
        ],
        "bindings": best_bindings,
        "missing_next_milestone_id": (
            None if complete else milestones[matched_count]["id"]
        ),
        "last_matched_seq": matched_calls[-1]["seq"] if matched_calls else None,
        "span_call_count": best[-1] - best[0] + 1 if best else 0,
    }


def _find_next_match(
    calls: Sequence[JsonDict],
    milestone: JsonDict,
    *,
    after_index: int,
    max_gap: int | None,
    bindings: JsonDict,
) -> tuple[int, JsonDict] | None:
    end = len(calls)
    if max_gap is not None:
        end = min(end, after_index + max_gap + 2)
    for index in range(after_index + 1, end):
        next_bindings = _match_and_capture(calls[index], milestone, bindings)
        if next_bindings is not None:
            return index, next_bindings
    return None


def _match_and_capture(
    call: JsonDict,
    milestone: JsonDict,
    bindings: Mapping[str, Any],
) -> JsonDict | None:
    if call["name"] not in milestone["tools"]:
        return None
    if not _conditions_match(call, milestone["where"], bindings):
        return None
    any_of = milestone["any_of"]
    if any_of and not any(
        _conditions_match(call, branch, bindings) for branch in any_of
    ):
        return None
    captured = dict(bindings)
    for binding, path in milestone["capture"].items():
        found, value = _lookup_path(call, path)
        if not found:
            return None
        if binding in captured and captured[binding] != value:
            return None
        captured[binding] = value
    return captured


def _conditions_match(
    call: JsonDict,
    conditions: Mapping[str, Any],
    bindings: Mapping[str, Any],
) -> bool:
    for path, expected in conditions.items():
        found, actual = _lookup_path(call, str(path))
        if isinstance(expected, Mapping) and any(
            str(key).startswith("$") for key in expected
        ):
            if not _operator_match(found, actual, expected, bindings):
                return False
        elif not found or actual != expected:
            return False
    return True


def _operator_match(
    found: bool,
    actual: object,
    expression: Mapping[str, Any],
    bindings: Mapping[str, Any],
) -> bool:
    supported = {"$exists", "$in", "$ne", "$ref"}
    unknown = set(expression) - supported
    if unknown:
        raise ValueError(f"unsupported toolchain condition operator(s): {sorted(unknown)}")
    if "$exists" in expression and found is bool(expression["$exists"]):
        pass
    elif "$exists" in expression:
        return False
    if "$in" in expression:
        choices = expression["$in"]
        if not isinstance(choices, list):
            raise ValueError("toolchain $in condition requires a list")
        if not found or actual not in choices:
            return False
    if "$ne" in expression and found and actual == expression["$ne"]:
        return False
    if "$ref" in expression:
        binding = expression["$ref"]
        if not isinstance(binding, str) or not binding:
            raise ValueError("toolchain $ref condition requires a binding name")
        if binding not in bindings or not found or actual != bindings[binding]:
            return False
    return True


def _lookup_path(root: object, dotted_path: str) -> tuple[bool, object]:
    current = root
    for part in dotted_path.split("."):
        if not part or not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _discover_tool_event_sources(path: Path) -> tuple[Path, ...]:
    if path.is_file():
        return (path,)
    if not path.exists():
        raise FileNotFoundError(f"tool event source does not exist: {path}")
    if not path.is_dir():
        raise ValueError(f"tool event source is neither a file nor directory: {path}")
    direct = (path / "tool_calls.jsonl", path / "rollout" / "tool_calls.jsonl")
    for candidate in direct:
        if candidate.is_file():
            return (candidate,)
    matches = tuple(
        sorted(candidate for candidate in path.rglob("tool_calls.jsonl") if candidate.is_file())
    )
    if not matches:
        raise FileNotFoundError(f"no tool_calls.jsonl found beneath directory: {path}")
    return matches


def _session_id_from_source(source: Path) -> str:
    if source.parent.name == "rollout":
        return source.parent.parent.name
    return ""


def _inline_sequence(value: str, index: int) -> JsonDict:
    sequence_id = f"sequence_{index + 1}"
    body = value
    if "=" in value:
        raw_id, body = value.split("=", 1)
        sequence_id = raw_id.strip()
    tools = [tool.strip() for tool in body.split(",") if tool.strip()]
    return {"id": sequence_id, "milestones": tools}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tool-events",
        required=True,
        help="One tool_calls.jsonl, a session/rollout directory, or a run root",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--spec", help=f"JSON {SPEC_SCHEMA_VERSION} file")
    source.add_argument(
        "--sequence",
        action="append",
        help="Ordered tool names as '[id=]tool_a,tool_b,...'; repeatable",
    )
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)

    sequences = (
        load_toolchain_spec(args.spec)
        if args.spec
        else tuple(
            _inline_sequence(value, index)
            for index, value in enumerate(args.sequence or [])
        )
    )
    report = extract_toolchain_coverage(args.tool_events, sequences)
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
