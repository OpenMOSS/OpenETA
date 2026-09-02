from __future__ import annotations

import inspect
from pathlib import Path

from scripts.univtac import rerun_clean_reset_gates as gates
from sim.envs.univtac.cuda_device_diagnostics import should_continue_after_gate


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_gate_failure_does_not_mark_followups_as_run() -> None:
    summary = gates._new_gate_summary()
    assert summary["tasks"]["lift_can"]["1000000"]["status"] == "not_run_due_to_gate"
    for task in ("pull_out_key", "insert_hole", "insert_tube"):
        assert set(summary["tasks"][task]) == {"1000000", "1000001", "1000002"}
        assert all(
            value["status"] == "not_run_due_to_gate"
            for value in summary["tasks"][task].values()
        )


def test_external_inotify_precondition_checks_both_limits_and_usage() -> None:
    config = {
        "external_inotify_minimums": {
            "max_user_instances": 1024,
            "max_user_watches": 524288,
        }
    }
    inventory = {
        "max_user_instances": 1024,
        "max_user_watches": 524288,
        "instance_usage_ratio": 0.2,
        "watch_usage_ratio": 0.3,
    }
    assert gates._validate_external_inotify(inventory, config)["satisfied"] is True
    failed = gates._validate_external_inotify(
        {**inventory, "watch_usage_ratio": 0.75}, config
    )
    assert failed["satisfied"] is False
    assert "inotify_watch_usage_at_or_above_75_percent" in failed["failures"]


def test_runtime_abort_never_passes_a_gate() -> None:
    runtime_failure = {
        "failure_stage": "reset_runtime_error",
        "reset_valid": False,
        "reset_returned": False,
        "planning_call_count": 0,
    }
    assert should_continue_after_gate(runtime_failure, require_reset_valid=False) is False
    assert should_continue_after_gate(runtime_failure, require_reset_valid=True) is False


def test_lift_control_requires_reset_valid_not_task_success() -> None:
    reset_valid = {
        "failure_stage": None,
        "reset_valid": True,
        "reset_returned": True,
        "planning_call_count": 4,
        "check_success": False,
        "policy_success": False,
    }
    assert should_continue_after_gate(reset_valid, require_reset_valid=True) is True


def test_gate_source_does_not_load_models_or_modify_planner_contract() -> None:
    source = inspect.getsource(gates)
    assert "FTP1InferenceWrapper" not in source
    assert "FTP1ChunkPolicy" not in source
    assert "check_success" not in source
    assert "plan_success=True" not in source.replace(" ", "")
    assert "constraint_pose=" not in source


def test_vendor_tree_remains_unmodified() -> None:
    import subprocess

    completed = subprocess.run(
        ["git", "diff", "--quiet", "--", "third_party/ftp1-policy/UniVTAC"],
        cwd=REPO_ROOT,
        check=False,
    )
    assert completed.returncode == 0


def test_author_bundle_has_fixed_small_text_file_set() -> None:
    source = inspect.getsource(gates._create_author_bundle)
    required = {
        "README.md",
        "protocol_summary.md",
        "clean_runtime_summary.md",
        "process_cleanup_summary.json",
        "inotify_summary.json",
        "cuda_device_matrix.json",
        "uipc_sentinel_results.json",
        "reset_gate_results.json",
        "runtime_manifest_clean.json",
        "minimal_commands.sh",
        "relevant_log_tail.txt",
        "system_restore_verification.json",
        "suggested_author_message.md",
    }
    assert all(name in source for name in required)
    assert ".pt" not in source
    assert ".mp4" not in source
    assert '"owners"' not in source
    assert 'public_runtime["final_gpu"] = final_gpu' not in source
