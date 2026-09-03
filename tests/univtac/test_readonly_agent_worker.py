from __future__ import annotations

from pathlib import Path

import pytest

from sim.envs.univtac import read_only_agent_worker as worker

REPO_ROOT = Path(__file__).resolve().parents[2]


def _config() -> dict:
    return {
        "simulator_gate_config": "configs/univtac/pull_out_key_seed1000000_gate.yaml",
        "task": "pull_out_key",
        "seed": 1_000_000,
        "agent_mode": "read_only",
        "backend_mode": "localhost_capture",
        "expected_image_count": 4,
        "simulator_invocation_limit": 1,
        "agent_request_limit": 1,
        "allow_agent_action": False,
        "allow_tools": False,
        "allow_external_network": False,
    }


def test_worker_config_fixes_limits_and_disables_actions() -> None:
    assert worker.validate_readonly_gate_config(_config())["seed"] == 1_000_000
    with pytest.raises(worker.ReadOnlyWorkerError):
        worker.validate_readonly_gate_config({**_config(), "allow_agent_action": True})


def test_simulator_failure_stops_before_agent_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def fail_process(*args, **kwargs):
        calls.append((args, kwargs))
        return ({"returncode": 1, "cleanup_complete": True}, "failed")

    monkeypatch.setattr(worker, "_run_logged_process", fail_process)
    with pytest.raises(worker.ReadOnlyWorkerError, match="simulator_gate_failed"):
        worker.run_read_only_agent_worker(
            config=_config(),
            runtime_python=Path("/runtime/python"),
            source_root=Path("/source"),
            output_root=tmp_path / "run",
            repo_root=REPO_ROOT,
            headless=True,
        )
    assert len(calls) == 1
    assert not (tmp_path / "run/agent_input").exists()
    assert not (tmp_path / "run/agent_backend/endpoint.json").exists()


def test_worker_source_has_one_simulator_call_and_zero_action_paths() -> None:
    source = (REPO_ROOT / "sim/envs/univtac/read_only_agent_worker.py").read_text(encoding="utf-8")
    assert source.count("_run_logged_process(") == 3  # definition + simulator + agent
    assert source.count("simulator_lifecycle, _ = _run_logged_process(") == 1
    assert "OpenEtaAgentRuntime" not in source
    assert "AgentSimBridge" not in source
    assert "ToolCallingPlanner" not in source
    assert "no_parallel_runtime" not in source
    assert "hashlib" not in source and "sha256" not in source.lower()
    assert "pkill" not in source and "killall" not in source
    agent_script = (REPO_ROOT / "scripts/univtac/run_readonly_observation_agent.py").read_text(
        encoding="utf-8"
    )
    assert "--snapshot-pre" not in agent_script
    assert "--simulator-output-root" not in agent_script
    assert "--host-only-root" not in agent_script
    assert "def act(" not in agent_script


def test_capture_rejects_absolute_path_outside_common_temp_roots() -> None:
    payload = {
        "model": "local-readonly-capture",
        "temperature": 0,
        "messages": [
            {"role": "system", "content": worker.SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            '{"task_instruction":"/etc/private/context.json",'
                            '"step_identifiers":{},"proprio":{},"visual_descriptors":[]}'
                        ),
                    },
                    *[{"type": "image_url", "image_url": {"url": "data:image/png;base64,"}}] * 4,
                ],
            },
        ],
    }
    with pytest.raises(worker.ReadOnlyWorkerError, match="local path"):
        worker._validate_capture_payload(payload)


def test_success_contract_requires_exactly_one_reset_and_observation() -> None:
    source = (REPO_ROOT / "sim/envs/univtac/read_only_agent_worker.py").read_text(encoding="utf-8")
    assert '"simulator_reset_count": 1' in source
    assert '"native_observation_count": 1' in source
    assert "counters == expected_counters" in source
