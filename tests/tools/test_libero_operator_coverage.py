import json
import subprocess
from argparse import Namespace
from pathlib import Path

from scripts.embodied import libero_operator_coverage as coverage
from scripts.embodied.episode_summary import summarize_episode


def test_stop_episode_services_is_idempotent(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = []

    def fake_run(*args, **kwargs):
        calls.append((args, kwargs))
        return subprocess.CompletedProcess(args[0], 0)

    monkeypatch.setattr(coverage.subprocess, "run", fake_run)

    coverage._stop_episode_services(str(tmp_path))
    coverage._stop_episode_services(str(tmp_path))

    assert calls == [
        (
            (
                [
                    str(
                        coverage.REPO_ROOT
                        / "scripts/embodied/stop_episode_services.sh"
                    ),
                    str(tmp_path),
                ],
            ),
            {
                "cwd": coverage.REPO_ROOT,
                "stdout": subprocess.DEVNULL,
                "stderr": subprocess.DEVNULL,
                "check": False,
            },
        )
    ]
    assert (tmp_path / ".batch-replay-services-stopped").is_file()


def test_episode_summary_falls_back_to_last_tool_terminal_fields(
    tmp_path: Path,
) -> None:
    (tmp_path / "operator_context.jsonl").write_text(
        json.dumps(
            {
                "tool": "finish_episode",
                "response_text_blocks": [
                    json.dumps(
                        {
                            "episode_status": "running",
                            "episode_success": False,
                        }
                    )
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_episode(tmp_path)

    assert summary["episode_status"] == "running"
    assert summary["episode_success"] is False


def test_plan_matrix_writes_each_task_seed_once(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tasks = [
        {
            "env_id": f"openeta/libero_libero_spatial_task{index}-v0",
            "suite": "libero_spatial",
            "task": f"spatial task {index}",
        }
        for index in range(10)
    ]
    tasks.append(
        {
            "env_id": "openeta/libero_libero_object_task0-v0",
            "suite": "libero_object",
            "task": "object task",
        }
    )
    monkeypatch.setattr(coverage, "_authoritative_tasks", lambda *_: tasks)
    args = Namespace(
        output=tmp_path,
        libero_python=Path("python"),
        libero_dir=Path("LIBERO"),
        suite="libero_spatial",
        seeds=list(range(10)),
        profile="profile",
        model="model",
        reasoning_effort="medium",
        model_provider="example",
        pointcloud_mode="live-multiview-consensus",
        image_width=512,
        image_height=512,
        force=False,
    )

    assert coverage.plan_matrix(args) == 0

    manifest = json.loads(
        (tmp_path / "coverage_manifest.json").read_text(encoding="utf-8")
    )
    planned = manifest["tasks"]
    assert manifest["suite_task_count"] == 10
    assert manifest["seeds"] == list(range(10))
    assert manifest["reasoning_effort"] == "medium"
    assert len(planned) == 100
    assert len({item["run_id"] for item in planned}) == 100
    assert [item["ordinal"] for item in planned] == list(range(100))
    assert {
        (item["env_id"], item["seed"]) for item in planned
    } == {
        (task["env_id"], seed)
        for task in tasks[:10]
        for seed in range(10)
    }


def test_run_key_keeps_repeated_environment_seeds_distinct() -> None:
    assert coverage._run_key({"run_id": "task0-seed-0", "ordinal": 0}) != (
        coverage._run_key({"run_id": "task0-seed-1", "ordinal": 10})
    )
    assert coverage._run_key({"ordinal": 7}) == "ordinal-007"


def test_plan_first_success_covers_all_suites_and_freezes_bootstrap(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tasks = [
        {
            "env_id": f"openeta/libero_{suite}_task{index}-v0",
            "suite": suite,
            "task": f"{suite} task {index}",
        }
        for suite, count in (
            ("libero_10", 10),
            ("libero_goal", 10),
            ("libero_object", 10),
            ("libero_spatial", 10),
            ("libero_90", 90),
        )
        for index in range(count)
    ]
    bootstrap = [
        {
            "episode_root": "/artifact/success",
            "env_id": tasks[0]["env_id"],
            "seed": 0,
            "episode_success": True,
            "infrastructure_valid": True,
            "finish_episode_count": 1,
        }
    ]
    monkeypatch.setattr(coverage, "_authoritative_tasks", lambda *_: tasks)
    monkeypatch.setattr(coverage, "_matching_attempts", lambda *_args, **_kwargs: bootstrap)
    args = Namespace(
        output=tmp_path,
        artifacts_root=tmp_path / "artifacts",
        libero_python=Path("python"),
        libero_dir=Path("LIBERO"),
        seeds=[0, 1, 2],
        profile="profile",
        model="model",
        reasoning_effort="medium",
        model_provider="example",
        pointcloud_mode="live-multiview-consensus",
        image_width=512,
        image_height=512,
        force=False,
    )

    assert coverage.plan_first_success(args) == 0

    manifest = json.loads(
        (tmp_path / "first_success_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["authoritative_task_count"] == 130
    assert manifest["suite_task_counts"] == {
        "libero_10": 10,
        "libero_90": 90,
        "libero_goal": 10,
        "libero_object": 10,
        "libero_spatial": 10,
    }
    assert manifest["seed_schedule"] == [0, 1, 2]
    assert manifest["reasoning_effort"] == "medium"
    assert manifest["bootstrap_attempts"] == bootstrap
    assert manifest["bootstrap_successful_task_count"] == 1
    assert len(manifest["tasks"]) == 130


def test_first_success_task_rows_stop_after_any_valid_success() -> None:
    manifest = {
        "tasks": [
            {
                "env_id": "openeta/libero_libero_spatial_task0-v0",
                "suite": "libero_spatial",
                "task": "task",
                "ordinal": 0,
                "slug": "spatial-task0",
            }
        ]
    }
    attempts = [
        {
            "env_id": manifest["tasks"][0]["env_id"],
            "seed": 0,
            "episode_status": "failed",
            "episode_success": False,
            "infrastructure_valid": True,
            "finish_episode_count": 1,
            "episode_root": "/failure",
        },
        {
            "env_id": manifest["tasks"][0]["env_id"],
            "seed": 1,
            "episode_status": "completed",
            "episode_success": True,
            "infrastructure_valid": False,
            "finish_episode_count": 1,
            "model_provider_match": True,
            "episode_root": "/success",
        },
    ]

    rows = coverage._first_success_task_rows(manifest, attempts)

    assert rows[0]["success"] is True
    assert rows[0]["successful_seed"] == 1
    assert rows[0]["successful_attempt_infrastructure_valid"] is False
    assert rows[0]["valid_attempt_count"] == 1
    assert rows[0]["valid_failed_seed_count"] == 1
