#!/usr/bin/env python3
"""Statically audit the public FTP-1 UniVTAC evaluation protocol."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable


CANONICAL_COMMIT = "89fa681d6c014cce28300946b7526db808e0b1c1"
DEFAULT_CHECKOUT = Path("/home/ubuntu/wybcode/.worktrees/ftp1-policy/r02-official-89fa681")
DEFAULT_OUTPUT = Path("outputs/ftp1-protocol-audit")
AUDIT_FILES = (
    "README.md",
    "UniVTAC/README.md",
    "UniVTAC/scripts/shell/eval_ftp1.sh",
    "UniVTAC/scripts/shell/eval_ftp1_batch.sh",
    "UniVTAC/scripts/eval_ftp1.py",
    "UniVTAC/scripts/eval_ftp1_until_success.py",
    "UniVTAC/scripts/eval_policy.py",
    "UniVTAC/parallel_eval.sh",
    "UniVTAC/scripts/parallel_eval_policy.py",
    "UniVTAC/policy/task_settings.json",
    "UniVTAC/task_config/contact.yml",
    "UniVTAC/policy/FTP1/deploy_policy.py",
    "UniVTAC/policy/FTP1/deploy.yml",
    "data_processing/parse_data_module/task_settings.json",
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(checkout: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _line_numbers(path: Path, patterns: Iterable[str]) -> dict[str, list[int]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return {
        pattern: [index for index, line in enumerate(lines, 1) if pattern in line]
        for pattern in patterns
    }


def _shell_array(text: str, name: str) -> list[str]:
    match = re.search(rf"(?ms)^\s*{re.escape(name)}=\(\s*(.*?)^\s*\)", text)
    if not match:
        return []
    return re.findall(r'["\']([^"\']+)["\']', match.group(1))


def _shell_default_int(text: str, name: str) -> int | None:
    match = re.search(rf"(?m)^\s*{re.escape(name)}=\$\{{{re.escape(name)}:-([0-9]+)\}}", text)
    return int(match.group(1)) if match else None


def _argparse_default(text: str, flag: str) -> int | None:
    pattern = rf'add_argument\(\s*["\']{re.escape(flag)}["\'][^\)]*?default\s*=\s*([0-9]+)'
    match = re.search(pattern, text, flags=re.S)
    return int(match.group(1)) if match else None


def discover_univtac_tasks(univtac_root: Path) -> list[str]:
    tasks: list[str] = []
    for path in sorted((univtac_root / "envs").glob("*.py")):
        if path.name.startswith("_") or path.stem == "collect":
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        class_names = {
            node.name for node in tree.body if isinstance(node, ast.ClassDef)
        }
        if {"Task", "TaskCfg"}.issubset(class_names):
            tasks.append(path.stem)
    return tasks


def discover_instruction_tasks(path: Path, supported: Iterable[str]) -> list[str]:
    supported_set = set(supported)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        for key in node.keys:
            if isinstance(key, ast.Constant) and isinstance(key.value, str):
                if key.value in supported_set:
                    found.add(key.value)
    return sorted(found)


def classify_task_sets(
    *,
    environment_tasks: Iterable[str],
    instruction_tasks: Iterable[str],
    batch_tasks: Iterable[str],
    checkpoint_tasks: Iterable[str],
    single_task_defaults: Iterable[str],
) -> dict[str, list[str]]:
    environment = sorted(set(environment_tasks))
    parseable = sorted(set(environment).intersection(instruction_tasks))
    return {
        "environment_supported_tasks": environment,
        "ftp1_code_supported_tasks": parseable,
        "batch_evaluated_tasks": list(dict.fromkeys(batch_tasks)),
        "checkpoint_covered_tasks": list(dict.fromkeys(checkpoint_tasks)),
        "single_task_default_tasks": list(dict.fromkeys(single_task_defaults)),
    }


def _protocol_paths(
    *, batch_tasks: list[str], total_num: int, start_seed: int, until_seed: int
) -> list[dict[str, Any]]:
    shared = {
        "task_config": "contact.yml",
        "whether_model_is_loaded_before_task_reset": True,
        "whether_task_reset_occurs_before_first_policy_action": True,
    }
    return [
        {
            "entry_point": "UniVTAC/scripts/shell/eval_ftp1_batch.sh",
            "called_script": "scripts/shell/eval_ftp1.sh -> scripts/eval_ftp1.py",
            "task_list": batch_tasks,
            "default_start_seed": start_seed,
            "default_episode_count": 20,
            "batch_override_episode_count": total_num,
            "seed_iteration_rule": "range(start_seed, start_seed + total_num); worker_id adds 100000",
            "whether_expert_check_is_used": False,
            "whether_seed_manifest_is_used": False,
            "whether_reset_failure_is_skipped": False,
            "whether_reset_failure_is_counted": (
                "plan_success is not checked after reset; normal loop completion increments done; "
                "a reset exception aborts before the episode is counted"
            ),
            "whether_reset_exception_aborts_worker": True,
            "whether_policy_failure_is_retried": False,
            "whether_until_success_semantics_is_used": False,
            **shared,
        },
        {
            "entry_point": "UniVTAC/scripts/shell/eval_ftp1.sh",
            "called_script": "scripts/eval_ftp1.py",
            "task_list": ["insert_HDMI"],
            "default_start_seed": start_seed,
            "default_episode_count": 20,
            "batch_override_episode_count": total_num,
            "seed_iteration_rule": "fixed consecutive seed range",
            "whether_expert_check_is_used": False,
            "whether_seed_manifest_is_used": False,
            "whether_reset_failure_is_skipped": False,
            "whether_reset_failure_is_counted": "same as eval_ftp1.py",
            "whether_reset_exception_aborts_worker": True,
            "whether_policy_failure_is_retried": False,
            "whether_until_success_semantics_is_used": False,
            **shared,
        },
        {
            "entry_point": "UniVTAC/scripts/eval_ftp1_until_success.py",
            "called_script": "self",
            "task_list": "required --task_name",
            "task_config": "contact.yml",
            "default_start_seed": until_seed,
            "default_episode_count": "target_successes=5",
            "batch_override_episode_count": None,
            "seed_iteration_rule": "increment seeds until 5 successes or max_episodes",
            "whether_expert_check_is_used": False,
            "whether_seed_manifest_is_used": "success_manifest.json is runtime output, not a frozen seed manifest",
            "whether_reset_failure_is_skipped": False,
            "whether_reset_failure_is_counted": (
                "reset exceptions abort before episodes_run increments; normal failed policy "
                "episodes increment episodes_run"
            ),
            "whether_reset_exception_aborts_worker": True,
            "whether_policy_failure_is_retried": False,
            "whether_until_success_semantics_is_used": True,
            "whether_model_is_loaded_before_task_reset": True,
            "whether_task_reset_occurs_before_first_policy_action": True,
        },
        {
            "entry_point": "UniVTAC/scripts/eval_policy.py",
            "called_script": "self (generic policy evaluator, not FTP-1 batch)",
            "task_list": "required positional task",
            "task_config": "required positional config",
            "default_start_seed": "1000000 * (1 + deploy_config.seed) when CLI=-1",
            "default_episode_count": "deploy config",
            "batch_override_episode_count": None,
            "seed_iteration_rule": "increments seed; expert-invalid seeds can be skipped",
            "whether_expert_check_is_used": "optional --expert_check",
            "whether_seed_manifest_is_used": "runtime metadata/seeds.json",
            "whether_reset_failure_is_skipped": "only under generic expert_check semantics",
            "whether_reset_failure_is_counted": "exceptions decrement test_num",
            "whether_reset_exception_aborts_worker": False,
            "whether_policy_failure_is_retried": False,
            "whether_until_success_semantics_is_used": False,
            "whether_model_is_loaded_before_task_reset": True,
            "whether_task_reset_occurs_before_first_policy_action": True,
        },
        {
            "entry_point": "UniVTAC/parallel_eval.sh",
            "called_script": "scripts/parallel_eval_policy.py (generic, not FTP-1)",
            "task_list": "required positional task",
            "task_config": "required positional config",
            "default_start_seed": "delegated",
            "default_episode_count": 100,
            "batch_override_episode_count": 100,
            "seed_iteration_rule": "delegated to generic parallel evaluator",
            "whether_expert_check_is_used": "unresolved_from_public_code at shell layer",
            "whether_seed_manifest_is_used": "unresolved_from_public_code at shell layer",
            "whether_reset_failure_is_skipped": "unresolved_from_public_code at shell layer",
            "whether_reset_failure_is_counted": "unresolved_from_public_code at shell layer",
            "whether_reset_exception_aborts_worker": "unresolved_from_public_code at shell layer",
            "whether_policy_failure_is_retried": "unresolved_from_public_code at shell layer",
            "whether_until_success_semantics_is_used": False,
            "whether_model_is_loaded_before_task_reset": "unresolved_from_public_code at shell layer",
            "whether_task_reset_occurs_before_first_policy_action": "unresolved_from_public_code at shell layer",
        },
    ]


def audit_protocol(checkout: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    checkout = checkout.expanduser().resolve()
    commit = _git_head(checkout)
    if commit != CANONICAL_COMMIT:
        raise RuntimeError(f"canonical FTP-1 commit mismatch: {commit}")
    univtac = checkout / "UniVTAC"
    batch_text = (univtac / "scripts/shell/eval_ftp1_batch.sh").read_text(encoding="utf-8")
    single_text = (univtac / "scripts/shell/eval_ftp1.sh").read_text(encoding="utf-8")
    eval_text = (univtac / "scripts/eval_ftp1.py").read_text(encoding="utf-8")
    until_text = (univtac / "scripts/eval_ftp1_until_success.py").read_text(encoding="utf-8")

    environment_tasks = discover_univtac_tasks(univtac)
    instruction_tasks = discover_instruction_tasks(
        univtac / "scripts/eval_ftp1.py", environment_tasks
    )
    batch_tasks = _shell_array(batch_text, "TASKS")
    checkpoint_dirs = _shell_array(batch_text, "CKPT_DIRS")
    checkpoint_tasks = [
        task for task, checkpoint in zip(batch_tasks, checkpoint_dirs) if checkpoint
    ]
    default_match = re.search(
        r'TASK_LIST=\$\{TASK_LIST:-\$\{1:-["\']([^"\']+)["\']\}\}',
        single_text,
    )
    default_task = default_match.group(1) if default_match else "unresolved_from_public_code"
    task_sets = classify_task_sets(
        environment_tasks=environment_tasks,
        instruction_tasks=instruction_tasks,
        batch_tasks=batch_tasks,
        checkpoint_tasks=checkpoint_tasks,
        single_task_defaults=[default_task],
    )
    total_num = _shell_default_int(batch_text, "TOTAL_NUM") or 100
    start_seed = _argparse_default(eval_text, "--start_seed") or 1000000
    until_seed = _argparse_default(until_text, "--start_seed") or 1000000

    audit = {
        "schema_version": "openeta.ftp1_protocol_audit.v1",
        "canonical_repository": "https://github.com/michaelyuancb/ftp1-policy.git",
        "canonical_commit": commit,
        "task_sets": task_sets,
        "checkpoint_coverage_basis": "public batch configuration paths; checkpoint contents not loaded",
        "public_checkpoint_count_claim": 6,
        "default_start_seed": start_seed,
        "batch_episode_count": total_num,
        "total_num_100_meaning": (
            "100 fixed consecutive seed attempts per task for workers=1; not 100 reset-valid "
            "episodes and not 100 successes"
        ),
        "protocol_paths": _protocol_paths(
            batch_tasks=batch_tasks,
            total_num=total_num,
            start_seed=start_seed,
            until_seed=until_seed,
        ),
        "expert_check_used_by_ftp1_path": False,
        "frozen_valid_seed_manifest_present": False,
        "reset_exception_behavior": (
            "task.reset is outside the inner cleanup handler; an exception propagates, "
            "aborts the worker, and is not counted"
        ),
        "policy_failure_retry": False,
        "paper_protocol_resolution": "unresolved_from_public_code",
        "unresolved_from_public_code": [
            "Public code does not prove that the paper used expert-filtered or reset-valid seeds.",
            "Public checkpoint contents and train_config action representation were not loaded.",
            "README dataset episodes do not uniquely establish evaluation episode eligibility semantics.",
            "Multi-worker seed offsets change the global seed set when worker allocation changes.",
        ],
    }

    line_patterns = {
        "README.md": ["ftp1_univtac_finetune", "eval_ftp1_batch.sh"],
        "UniVTAC/README.md": ["100 episodes per task"],
        "UniVTAC/scripts/shell/eval_ftp1.sh": ["TASK_LIST=", "TOTAL_NUM=", "eval_ftp1.py"],
        "UniVTAC/scripts/shell/eval_ftp1_batch.sh": ["TOTAL_NUM=", "TASKS=(", "CKPT_DIRS=("],
        "UniVTAC/scripts/eval_ftp1.py": ["start_seed", "seeds = list(range", "task.reset("],
        "UniVTAC/scripts/eval_ftp1_until_success.py": ["target_successes", "start_seed", "task.reset("],
        "UniVTAC/scripts/eval_policy.py": ["expert_check", "seeds.json", "task.reset("],
        "UniVTAC/parallel_eval.sh": ["TOTAL_NUM", "parallel_eval_policy.py"],
        "UniVTAC/scripts/parallel_eval_policy.py": ["task.reset(", "expert_check", "seeds.json"],
        "UniVTAC/policy/task_settings.json": ["insert_hole", "insert_tube", "lift_can"],
        "UniVTAC/task_config/contact.yml": ["render_frequency", "episode_num", "observations"],
        "UniVTAC/policy/FTP1/deploy_policy.py": ["train_config", "action_rep"],
        "UniVTAC/policy/FTP1/deploy.yml": ["policy_class"],
        "data_processing/parse_data_module/task_settings.json": ["insert_hole"],
    }
    source_files = []
    for relative in AUDIT_FILES:
        path = (checkout / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"required audit source is missing: {path}")
        source_files.append(
            {
                "relative_path": relative,
                "absolute_path": str(path),
                "sha256": _sha256(path),
                "evidence_lines": _line_numbers(path, line_patterns.get(relative, [])),
                "from_canonical_checkout": path.is_relative_to(checkout),
            }
        )
    source_manifest = {
        "schema_version": "openeta.ftp1_protocol_source_manifest.v1",
        "repository": "https://github.com/michaelyuancb/ftp1-policy.git",
        "commit": commit,
        "canonical_checkout": str(checkout),
        "files": source_files,
    }
    return audit, source_manifest


def render_markdown(audit: dict[str, Any]) -> str:
    sets = audit["task_sets"]
    lines = [
        "# FTP-1 UniVTAC public protocol audit",
        "",
        f"Canonical commit: `{audit['canonical_commit']}`",
        "",
        "## Task scopes",
        "",
        "| Scope | Tasks |",
        "|---|---|",
    ]
    for key in (
        "environment_supported_tasks",
        "ftp1_code_supported_tasks",
        "batch_evaluated_tasks",
        "checkpoint_covered_tasks",
        "single_task_default_tasks",
    ):
        lines.append(f"| {key} | {', '.join(sets[key])} |")
    lines.extend(
        [
            "",
            "## Seed and episode protocol",
            "",
            f"- Default FTP-1 start seed: `{audit['default_start_seed']}`.",
            f"- Batch episode count: `{audit['batch_episode_count']}`.",
            f"- `TOTAL_NUM=100`: {audit['total_num_100_meaning']}.",
            "- FTP-1 expert check: no.",
            "- Frozen valid-seed manifest: no.",
            f"- Reset exception: {audit['reset_exception_behavior']}.",
            "- Until-success is a separate, non-batch entrypoint.",
            "",
            "## Public-code boundary",
            "",
            "`unresolved_from_public_code`: the repository does not uniquely establish the paper's episode-eligibility protocol.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, default=DEFAULT_CHECKOUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError(f"output directory is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    audit, source_manifest = audit_protocol(args.checkout)
    _write_json(output_root / "ftp1_protocol_audit.json", audit)
    _write_json(output_root / "source_manifest.json", source_manifest)
    (output_root / "ftp1_protocol_audit.md").write_text(
        render_markdown(audit), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
