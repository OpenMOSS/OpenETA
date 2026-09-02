from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from sim.envs.univtac.resource_sanitation import (
    CUDA_ENV_ALLOWLIST,
    _cmdline,
    _proc_stat,
    _read_link,
    apply_cleanup_plan,
    build_cleanup_plan,
    build_restore_ready,
    cleanup_action_for_record,
    decide_inotify_change,
    run_managed_process,
    run_noninteractive_sysctl,
    safe_device_environment,
)


def _record(**overrides):
    record = {
        "pid": 123,
        "uid": os.getuid(),
        "start_ticks": 9,
        "eligible_for_cleanup": True,
        "is_ancestor_of_current_process": False,
        "open_tty": False,
        "cleanup_exclusion_reason": [],
        "cmdline": "python diagnose_reset.py",
        "cwd": "/repo",
        "exe": sys.executable,
        "process_group_id": 123,
    }
    record.update(overrides)
    return record


def test_cleanup_defaults_to_dry_run_plan() -> None:
    plan = build_cleanup_plan({"processes": [_record()]})
    assert plan["default_mode"] == "dry_run"
    assert plan["entries"][0]["action"] == "terminate"


def test_only_current_uid_eligible_process_can_terminate() -> None:
    assert cleanup_action_for_record(_record()) == "terminate"
    assert cleanup_action_for_record(_record(uid=os.getuid() + 1)) == "keep"
    assert cleanup_action_for_record(_record(eligible_for_cleanup=False)) == "uncertain"


@pytest.mark.parametrize(
    "overrides",
    [
        {"is_ancestor_of_current_process": True},
        {"open_tty": True},
        {"cleanup_exclusion_reason": ["excluded_interactive_or_agent_process"]},
    ],
)
def test_protected_interactive_or_agent_processes_are_kept(overrides) -> None:
    assert cleanup_action_for_record(_record(**overrides)) == "keep"


def test_uncertain_process_is_never_selected_for_termination() -> None:
    record = _record(eligible_for_cleanup=False, cleanup_exclusion_reason=[])
    plan = build_cleanup_plan({"processes": [record]})
    assert plan["entries"][0]["action"] == "uncertain"
    assert plan["counts"]["terminate"] == 0


def test_environment_snapshot_uses_only_allowlist() -> None:
    environment = {
        "CUDA_VISIBLE_DEVICES": "0",
        "RANK": "1",
        "WANDB_API_KEY": "must-not-appear",
    }
    snapshot = safe_device_environment(environment)
    assert set(snapshot) == set(CUDA_ENV_ALLOWLIST)
    assert "must-not-appear" not in json.dumps(snapshot)


def test_inotify_change_threshold() -> None:
    assert decide_inotify_change({"instance_usage_ratio": 0.49})[
        "instances_change_required"
    ] is False
    assert decide_inotify_change({"instance_usage_ratio": 0.75})[
        "instances_change_required"
    ] is True


def test_restore_ready_requires_clean_resources_and_low_reliable_counts() -> None:
    ready = build_restore_ready(
        original_instances=128,
        original_watches=65536,
        inotify_inventory={
            "max_user_instances": 1024,
            "max_user_watches": 524288,
            "instance_count": 120,
            "watch_count": 64000,
            "watch_count_reliable": True,
        },
        process_inventory={"processes": []},
        gpu_inventory={"compute_processes": []},
    )
    assert ready["safe_for_manual_restore"] is True
    assert ready["not_ready_reasons"] == []


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"instance_count": 128}, "instance_count_not_below_original_limit"),
        ({"watch_count": 65536}, "watch_count_not_below_original_limit"),
        ({"watch_count_reliable": False}, "watch_count_unreliable"),
    ],
)
def test_restore_ready_rejects_unsafe_inotify_state(override, reason) -> None:
    inventory = {
        "max_user_instances": 1024,
        "max_user_watches": 524288,
        "instance_count": 120,
        "watch_count": 64000,
        "watch_count_reliable": True,
        **override,
    }
    ready = build_restore_ready(
        original_instances=128,
        original_watches=65536,
        inotify_inventory=inventory,
        process_inventory={"processes": []},
        gpu_inventory={"compute_processes": []},
    )
    assert ready["safe_for_manual_restore"] is False
    assert reason in ready["not_ready_reasons"]


def test_managed_process_uses_process_group_and_reaps_timeout(tmp_path: Path) -> None:
    run = run_managed_process(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        cwd=tmp_path,
        log_path=tmp_path / "timeout.log",
        timeout_seconds=0.1,
        cleanup_wait_seconds=0.5,
    )
    assert run["timed_out"] is True
    assert run["sigterm_sent"] is True
    assert run["final_process_group_members"] == []
    assert run["cleanup_complete"] is True


def test_managed_process_records_clean_normal_exit(tmp_path: Path) -> None:
    run = run_managed_process(
        [sys.executable, "-c", "print(123)"],
        cwd=tmp_path,
        log_path=tmp_path / "normal.log",
        timeout_seconds=5,
    )
    assert run["returncode"] == 0
    assert run["timed_out"] is False
    assert run["final_process_group_members"] == []


def test_sigterm_precedes_sigkill_for_confirmed_stubborn_pid(tmp_path: Path) -> None:
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(30)",
        ],
        cwd=tmp_path,
    )
    try:
        time.sleep(0.1)
        stat = _proc_stat(process.pid)
        assert stat is not None
        plan = {
            "entries": [
                {
                    "pid": process.pid,
                    "start_ticks": stat["start_ticks"],
                    "action": "terminate",
                    "identity": {
                        "uid": os.getuid(),
                        "cmdline": _cmdline(process.pid),
                        "cwd": _read_link(Path("/proc") / str(process.pid) / "cwd"),
                        "exe": _read_link(Path("/proc") / str(process.pid) / "exe"),
                    },
                }
            ]
        }
        actions = apply_cleanup_plan(plan, term_wait_seconds=0.1)
        assert [action["signal"] for action in actions if action["signal"]] == [
            "SIGTERM",
            "SIGKILL",
        ]
    finally:
        process.kill()
        process.wait(timeout=5)


def test_sysctl_helper_is_noninteractive_and_never_writes_permanent_files(monkeypatch) -> None:
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    result = run_noninteractive_sysctl("fs.inotify.max_user_instances", 1024)
    assert result["success"] is True
    assert captured["command"] == [
        "sudo",
        "-n",
        "sysctl",
        "-w",
        "fs.inotify.max_user_instances=1024",
    ]
    assert not any("/etc/sysctl" in token for token in captured["command"])


def test_sysctl_helper_rejects_unapproved_keys() -> None:
    with pytest.raises(ValueError, match="not approved"):
        run_noninteractive_sysctl("kernel.hostname", 1)
