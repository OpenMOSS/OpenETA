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
