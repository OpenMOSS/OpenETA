from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.univtac.record_project_progress import append_entry, build_entry


def test_progress_entry_is_append_only_human_summary(tmp_path: Path) -> None:
    args = argparse.Namespace(
        round="R0.9.14",
        status="in_progress",
        pro_instruction_summary="做三 seed 触觉因果 pilot。",
        codex_work_summary="已完成代码和真实调用。",
        result_summary="正在核对解析错误。",
        experiment_root="univtac-isaac51-r0914",
        commit="",
        push="",
    )
    entry = build_entry(args)
    path = tmp_path / "progress.jsonl"
    append_entry(path, entry)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["round"] == "R0.9.14"
    assert "auth" not in json.dumps(saved).lower()
    assert saved["experiment_root"] == "univtac-isaac51-r0914"
    assert saved["pro_question"] == args.pro_instruction_summary
    assert saved["method_summary"] == args.codex_work_summary
    assert saved["observed_summary"] == args.result_summary
    assert saved["one_line_conclusion"] == args.result_summary
    assert saved["evidence"]["experiment_root"] == "univtac-isaac51-r0914"


def test_progress_entry_preserves_plain_language_narrative_and_evidence() -> None:
    args = argparse.Namespace(
        round="R0.9.15",
        status="completed",
        pro_instruction_summary="测试 difference map。",
        codex_work_summary="完成九次只读调用。",
        result_summary="mixed signal。",
        pro_question="显式差分是否更容易读懂？",
        method_summary="三个 seed，每个看三种输入。",
        observed_summary="交换差分图后，四次判断随之交换。",
        implication_summary="模型会读局部变化，但差分没有提高区域准确率。",
        one_line_conclusion="差分有方向敏感性，但暂未带来准确率增益。",
        experiment_root="univtac-isaac51-r0915",
        commit="582e728",
        push="pushed",
        tests=["392 passed", "Ruff passed"],
        artifact_paths=["summary.json", "dashboard/pilot.png"],
    )
    entry = build_entry(args)
    assert entry["pro_question"] == args.pro_question
    assert entry["implication_summary"] == args.implication_summary
    assert entry["one_line_conclusion"] == args.one_line_conclusion
    assert entry["evidence"]["tests"] == args.tests
    assert entry["evidence"]["artifact_paths"] == args.artifact_paths
