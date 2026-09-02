#!/usr/bin/env python3
"""Audit the pinned UniVTAC Isaac 5.1 compatibility boundary without imports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sim.envs.univtac.benchmark_compatibility import (
    CONTACT_RICH_TASKS,
    FTP1_TASKS,
    action_contract_matrix,
    benchmark_compatibility,
    cuda_build_matrix,
    dependency_matrix,
    official_validation_scope,
    tactile_contract_matrix,
    task_semantic_matrix,
    validate_expected_source,
)
from sim.envs.univtac.source_parity import (
    PinnedSource,
    build_hash_manifest,
    deterministic_json,
    validate_pinned_source,
)


DEFAULT_CONFIG = REPO_ROOT / "configs/univtac/isaac51_static_audit.yaml"
DEFAULT_OUTPUT = REPO_ROOT / "outputs/univtac-isaac51-r06"
HASH_FILES = (
    "README.md",
    "docs/Installation.md",
    "docs/Collection.md",
    "docs/Deploy.md",
    "docs/isaacsim_5_1_migration.md",
    "scripts/install.sh",
    "scripts/smoke_isaac51.py",
    "envs/_base_task.py",
    "envs/utils/atom.py",
    "envs/sensors/tactile.py",
    "task_config/contact.yml",
    *(f"envs/{task}.py" for task in FTP1_TASKS),
    "third_party/TacEx/UNIVTAC_BASELINE.md",
    "third_party/TacEx/source/tacex/setup.py",
    "third_party/TacEx/source/tacex_uipc/setup.py",
    "third_party/TacEx/source/tacex_uipc/libuipc/conda/env.yaml",
)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(deterministic_json(payload), encoding="utf-8")


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("static audit config must be a mapping")
    return payload


def _source(config: dict[str, Any], name: str) -> PinnedSource:
    payload = config["sources"][name]
    checkout = Path(payload["checkout"]).expanduser().resolve()
    task_root = checkout / payload.get("task_root", ".")
    return PinnedSource(
        label=name,
        repository=str(payload["repository"]),
        commit=str(payload["commit"]),
        checkout=checkout,
        task_root=task_root.resolve(),
    )


def migration_risks() -> dict[str, Any]:
    return {
        "schema_version": "openeta.univtac.migration_risk_register.v1",
        "risks": [
            {"risk": "RTX5090 native build recipe is claimed but not fully documented", "severity": "blocker_for_install", "mitigation": "confirm UNIVTAC_CUDA_ARCH and PTX/SASS policy with authors"},
            {"risk": "Isaac4.5 and Isaac5.1 datasets are declared non-cross-compatible", "severity": "high", "mitigation": "label new results UniVTAC-Isaac51 and keep legacy parity as a separate track"},
            {"risk": "static/kinematic actor semantics and initial heights changed", "severity": "high", "mitigation": "freeze an Isaac51 benchmark version and compare E0-E3 only within it"},
            {"risk": "adaptive grasp moved from camera distance to press depth", "severity": "high", "mitigation": "migrate observation/runtime adapter and audit thresholds before rollout"},
            {"risk": "six FTP-1 tasks lack public Isaac51 validation evidence", "severity": "high", "mitigation": "run isolated official smoke then reset-only task gates in R0.7 after authorization"},
            {"risk": "README branch and dataset wording is internally ambiguous", "severity": "medium", "mitigation": "bind all claims to commit 371fac6 and ask authors for intended release scope"},
        ],
    }


def two_track_plan() -> dict[str, Any]:
    return {
        "schema_version": "openeta.univtac.two_track_plan.v1",
        "track_a": {
            "name": "current_research_platform",
            "platform": "UniVTAC-Isaac51 pinned environment",
            "purpose": "Run OpenETA tactile-context E0/E1/E2/E3 on RTX5090 with all variants sharing one version.",
            "can_answer": "within-version causal comparison for the research claim",
            "cannot_claim": "direct reproduction of FTP-1 Isaac4.5 paper numbers",
        },
        "track_b": {
            "name": "legacy_paper_parity",
            "platform": "FTP-1 frozen Isaac4.5 stack on an author-confirmed compatible GPU/runtime",
            "purpose": "Reproduce old checkpoint and benchmark numbers.",
            "blocking_track_a": False,
        },
        "decision": "Keep both tracks; do not use Track A success as evidence that Track B parity is achieved.",
    }


def author_questions() -> dict[str, Any]:
    questions = [
        "What UNIVTAC_CUDA_ARCH should be used for RTX 5090?",
        "Was commit 371fac6 validated on an RTX 5090 specifically, rather than only an RTX 40-series GPU?",
        "Does the RTX 50-series path use sm_89/PTX forward compatibility or another native build configuration?",
        "Which Isaac51 tasks have completed official smoke, reset, expert collection and policy evaluation?",
        "Do all six FTP-1 tasks pass reset, collection and evaluation on Isaac51?",
        "Can the old FTP-1 checkpoint consume the Isaac51 observation/action contract without retraining?",
        "Does the project consider Isaac4.5 and Isaac5.1 success rates directly comparable?",
        "What is the recommended path for the frozen Isaac4.5 benchmark on RTX 50-series GPUs?",
        "Should legacy parity be retained on a 30/40-series GPU even if new research moves to Isaac51?",
    ]
    return {
        "schema_version": "openeta.univtac.author_questions.v1",
        "questions": [{"id": index + 1, "question": text} for index, text in enumerate(questions)],
    }


def author_message() -> str:
    return """各位老师好，我们已经看到官方 isaac51 分支对 RTX 40/50 系的适配公告。

当前复现故意使用 FTP-1 论文对应的 frozen Isaac Sim 4.5 stack，以保持旧 checkpoint、旧数据和旧 benchmark protocol 的 parity。由于官方明确说明 4.5 与 5.1 数据不互通，我们尚未把旧复现直接迁移到 5.1，也不声称发现了官方 bug。

我们在 RTX 5090 上清理运行时并提高 inotify 后确认：基础 CUDA/Warp 正常，D0/D1 映射同一 GPU，旧的 cudaErrorInvalidDevice 未复现，但官方 UIPC hello path 的首个 sentinel step 在两种模式下均未在 300 秒内完成。我们准备将新研究与旧论文 parity 分成两条轨道。

希望确认：RTX 5090 推荐的 UNIVTAC_CUDA_ARCH 与 PTX/SASS 策略、isaac51 实际验证的任务范围、六个 FTP-1 任务的 reset/collection/eval 状态、旧 checkpoint 是否可迁移，以及新旧成功率是否被视为可直接比较。需要时我们可以提供不含凭据和大型文件的 clean-state bundle。
"""


def render_summary(
    *, compatibility: dict[str, Any], cuda: dict[str, Any], tasks: dict[str, Any],
    validation: dict[str, Any]
) -> str:
    rows = [
        "# UniVTAC Isaac51 compatibility boundary",
        "",
        "This is a static, commit-pinned audit. No installer, native build, simulator, task, policy, or system mutation was run.",
        "",
        "## Decision",
        "",
        f"- RTX 5090 recipe: `{cuda['rtx5090_build_recipe_status']}`.",
        f"- Within-version validity: `{compatibility['within_version_experimental_validity']}`.",
        f"- Cross-version numeric comparability: `{compatibility['cross_version_numeric_comparability']}`.",
        f"- Adapter migration: `{compatibility['openeta_adapter_migration_scope']}`.",
        f"- Six-task validation: `{compatibility['six_task_validation_status']}`.",
        f"- Next: `{compatibility['recommended_next_step']}`.",
        "",
        "## Six-task summary",
        "",
        "| Task | Static compatibility |",
        "|---|---|",
    ]
    rows.extend(
        f"| `{item['task']}` | `{item['task_semantic_compatibility']}` |"
        for item in tasks["tasks"]
    )
    rows.extend(
        [
            "",
            "## Public validation boundary",
            "",
            f"- Six FTP-1 tasks: `{validation['phase_one_explicit_evidence']['six_ftp1_tasks']}`.",
            "- Public phase-one evidence covers the Isaac51 smoke and one `grasp_classify` collection command, not six-task evaluation parity.",
            "- Track A may support the new research after isolated runtime validation; Track B remains necessary for old paper-number parity.",
        ]
    )
    return "\n".join(rows) + "\n"


def audit(config_path: Path, output_root: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    legacy = _source(config, "legacy")
    isaac51 = _source(config, "isaac51")
    pinned = {
        "schema_version": "openeta.univtac.pinned_sources.v1",
        "sources": [
            validate_pinned_source(legacy, require_detached=False),
            validate_pinned_source(isaac51, require_detached=True),
        ],
        "moving_branch_head_used": False,
        "runtime_imported": False,
        "installer_executed": False,
        "native_build_executed": False,
        "simulator_executed": False,
    }
    validate_expected_source(legacy.task_root, isaac51.task_root)
    dependencies = dependency_matrix()
    cuda = cuda_build_matrix()
    tasks = task_semantic_matrix()
    actions = action_contract_matrix()
    tactile = tactile_contract_matrix()
    validation = official_validation_scope()
    compatibility = benchmark_compatibility(
        legacy_commit=legacy.commit, isaac51_commit=isaac51.commit
    )
    artifacts: dict[str, Any] = {
        "pinned_sources.json": pinned,
        "source_hash_manifest.json": build_hash_manifest((legacy, isaac51), HASH_FILES),
        "dependency_matrix.json": dependencies,
        "cuda_build_matrix.json": cuda,
        "task_semantic_matrix.json": tasks,
        "action_contract_matrix.json": actions,
        "tactile_contract_matrix.json": tactile,
        "benchmark_compatibility.json": compatibility,
        "official_validation_scope.json": validation,
        "migration_risk_register.json": migration_risks(),
        "recommended_two_track_plan.json": two_track_plan(),
        "author_questions.json": author_questions(),
    }
    for name, payload in artifacts.items():
        write_json(output_root / name, payload)
    (output_root / "author_message.md").write_text(author_message(), encoding="utf-8")
    (output_root / "audit_summary.md").write_text(
        render_summary(
            compatibility=compatibility,
            cuda=cuda,
            tasks=tasks,
            validation=validation,
        ),
        encoding="utf-8",
    )
    return compatibility


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    result = audit(
        Path(args.config).expanduser().resolve(),
        Path(args.output_root).expanduser().resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
