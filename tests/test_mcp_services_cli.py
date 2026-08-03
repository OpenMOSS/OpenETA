from __future__ import annotations

import json
import signal
from pathlib import Path

import pytest

from scripts import openeta_mcp_services as cli


def test_status_without_pid_reports_not_running(tmp_path: Path, capsys) -> None:
    assert cli.main(["status", "sam3", "--state-dir", str(tmp_path), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["sam3"]["running"] is False
    assert payload["sam3"]["pid"] is None


def test_health_unreachable_reports_failed(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "_http_health_check", lambda _url, expected_server=None: False)

    assert cli.main(["health", "sam3", "--state-dir", str(tmp_path), "--json"]) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["sam3"]["health"] == "failed"


def test_start_sam3_dry_run_prints_command(tmp_path: Path, capsys) -> None:
    assert (
        cli.main(
            [
                "start",
                "sam3",
                "--state-dir",
                str(tmp_path),
                "--sam3-python",
                "/path/to/sam3-env/bin/python",
                "--dry-run",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "/path/to/sam3-env/bin/python" in output
    assert "tools/sam3_mcp_server.py" in output
    assert "--transport sse" in output
    assert "--port 8773" in output
    assert not (tmp_path / "sam3.pid").exists()


def test_start_anygrasp_requires_sdk_and_checkpoint(tmp_path: Path, capsys) -> None:
    assert (
        cli.main(
            [
                "start",
                "anygrasp",
                "--state-dir",
                str(tmp_path),
                "--dry-run",
            ]
        )
        == 2
    )

    assert "anygrasp sdk root" in capsys.readouterr().err.lower()


def test_start_anyplace_requires_root_and_config(tmp_path: Path, capsys) -> None:
    assert (
        cli.main(
            ["start", "anyplace", "--state-dir", str(tmp_path), "--dry-run"]
        )
        == 2
    )

    assert "anyplace root" in capsys.readouterr().err.lower()


def test_start_contact_graspnet_requires_root_and_checkpoint(tmp_path: Path, capsys) -> None:
    assert (
        cli.main(
            ["start", "contact_graspnet", "--state-dir", str(tmp_path), "--dry-run"]
        )
        == 2
    )

    assert "contact-graspnet root" in capsys.readouterr().err.lower()

    assert (
        cli.main(
            [
                "start",
                "contact_graspnet",
                "--state-dir",
                str(tmp_path),
                "--contact-graspnet-root",
                "/path/to/contact-graspnet",
                "--dry-run",
            ]
        )
        == 2
    )
    assert "checkpoint directory" in capsys.readouterr().err.lower()


def test_start_all_dry_run_includes_four_services(tmp_path: Path, capsys) -> None:
    assert (
        cli.main(
            [
                "start",
                "all",
                "--state-dir",
                str(tmp_path),
                "--sam3-python",
                "/path/to/sam3-env/bin/python",
                "--anygrasp-python",
                "/path/to/anygrasp-env/bin/python",
                "--anygrasp-sdk-root",
                "/path/to/anygrasp_sdk",
                "--anygrasp-checkpoint-path",
                "/path/to/checkpoint_detection.tar",
                "--anyplace-python",
                "/path/to/anyplace-env/bin/python",
                "--anyplace-root",
                "/path/to/anyplace",
                "--anyplace-config-path",
                "/path/to/anyplace-config.yaml",
                "--contact-graspnet-python",
                "/path/to/contact-env/bin/python",
                "--contact-graspnet-root",
                "/path/to/contact-graspnet",
                "--contact-graspnet-checkpoint-dir",
                "/path/to/contact-checkpoint",
                "--dry-run",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "tools/sam3_mcp_server.py" in output
    assert "tools/anygrasp_mcp_server.py" in output
    assert "tools/anyplace_mcp_server.py" in output
    assert "tools/contact_graspnet_mcp_server.py" in output
    assert "--sdk-root /path/to/anygrasp_sdk" in output
    assert "--checkpoint-path /path/to/checkpoint_detection.tar" in output
    assert "--anyplace-root /path/to/anyplace" in output
    assert "--config-path /path/to/anyplace-config.yaml" in output
    assert "--contact-graspnet-root /path/to/contact-graspnet" in output
    assert "--checkpoint-dir /path/to/contact-checkpoint" in output


def test_contact_graspnet_config_reads_environment(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENETA_CONTACT_GRASPNET_PYTHON", "/env/contact/bin/python")
    monkeypatch.setenv("OPENETA_CONTACT_GRASPNET_ROOT", "/env/contact/root")
    monkeypatch.setenv(
        "OPENETA_CONTACT_GRASPNET_CHECKPOINT_DIR",
        "/env/contact/checkpoint",
    )
    args = cli.build_parser().parse_args(
        ["start", "contact_graspnet", "--state-dir", str(tmp_path), "--dry-run"]
    )

    config = cli._build_configs(args)[0]

    assert config.python == "/env/contact/bin/python"
    assert config.port == 8776
    assert "--contact-graspnet-root" in config.command
    assert "/env/contact/root" in config.command
    assert "--checkpoint-dir" in config.command
    assert "/env/contact/checkpoint" in config.command
    assert "--seed" not in config.command
    assert "--max-candidates" not in config.command


def test_stop_uses_sigterm_by_default(tmp_path: Path, monkeypatch) -> None:
    pid_file = tmp_path / "sam3.pid"
    pid_file.write_text("1234\n", encoding="utf-8")
    sent = []
    monkeypatch.setattr(cli, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(cli, "_pid_matches_command", lambda _pid, _config: True)
    monkeypatch.setattr(cli.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)
    monkeypatch.setattr(cli, "STOP_TIMEOUT_S", 0.0)

    assert cli.main(["stop", "sam3", "--state-dir", str(tmp_path)]) == 1
    assert sent
    assert sent[0] == (1234, signal.SIGTERM)


def test_stop_force_uses_sigkill(tmp_path: Path, monkeypatch) -> None:
    pid_file = tmp_path / "sam3.pid"
    pid_file.write_text("1234\n", encoding="utf-8")
    sent = []
    monkeypatch.setattr(cli, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(cli, "_pid_matches_command", lambda _pid, _config: True)
    monkeypatch.setattr(cli.os, "kill", lambda pid, sig: sent.append((pid, sig)))
    monkeypatch.setattr(cli.time, "sleep", lambda _seconds: None)

    assert cli.main(["stop", "sam3", "--state-dir", str(tmp_path), "--force"]) == 0
    assert sent == [(1234, signal.SIGKILL)]


def test_smoke_uses_mcp_list_tools_without_real_service(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "_mcp_list_tools", lambda _url: ["segment"])

    assert cli.main(["smoke", "sam3", "--state-dir", str(tmp_path), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["sam3"]["smoke"] == "ok"
    assert payload["sam3"]["tools"] == ["segment"]


def test_anyplace_smoke_requires_predict_placement(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "_mcp_list_tools", lambda _url: ["predict_placement"])

    assert cli.main(["smoke", "anyplace", "--state-dir", str(tmp_path), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["anyplace"]["smoke"] == "ok"
    assert payload["anyplace"]["tools"] == ["predict_placement"]


def test_anyplace_smoke_fails_when_tool_is_missing(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "_mcp_list_tools", lambda _url: [])

    assert cli.main(["smoke", "anyplace", "--state-dir", str(tmp_path), "--json"]) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["anyplace"]["reason"] == "missing_tool:predict_placement"


def test_contact_graspnet_smoke_requires_predict_grasps(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(cli, "_mcp_list_tools", lambda _url: ["predict_grasps"])

    assert (
        cli.main(["smoke", "contact_graspnet", "--state-dir", str(tmp_path), "--json"])
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload["contact_graspnet"]["smoke"] == "ok"
    assert payload["contact_graspnet"]["tools"] == ["predict_grasps"]


def test_unknown_target_is_rejected() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["status", "other"])


def test_restart_dry_run_does_not_stop_running_service(tmp_path: Path, monkeypatch, capsys) -> None:
    pid_file = tmp_path / "sam3.pid"
    pid_file.write_text("1234\n", encoding="utf-8")
    sent = []
    monkeypatch.setattr(cli, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(cli, "_pid_matches_command", lambda _pid, _config: True)
    monkeypatch.setattr(cli.os, "kill", lambda pid, sig: sent.append((pid, sig)))

    assert (
        cli.main(
            [
                "restart",
                "sam3",
                "--state-dir",
                str(tmp_path),
                "--sam3-python",
                "/path/to/sam3-env/bin/python",
                "--dry-run",
            ]
        )
        == 0
    )

    assert sent == []
    assert "dry-run" in capsys.readouterr().out


def test_invalid_pid_is_not_killed(tmp_path: Path, monkeypatch) -> None:
    pid_file = tmp_path / "sam3.pid"
    pid_file.write_text("0\n", encoding="utf-8")
    sent = []
    monkeypatch.setattr(cli.os, "kill", lambda pid, sig: sent.append((pid, sig)))

    assert cli.main(["stop", "sam3", "--state-dir", str(tmp_path)]) == 1
    assert sent == []


def test_start_with_mismatched_live_pid_fails_without_already_running(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    pid_file = tmp_path / "sam3.pid"
    pid_file.write_text("1234\n", encoding="utf-8")
    monkeypatch.setattr(cli, "_pid_alive", lambda _pid: True)
    monkeypatch.setattr(cli, "_pid_matches_command", lambda _pid, _config: False)

    assert (
        cli.main(
            [
                "start",
                "sam3",
                "--state-dir",
                str(tmp_path),
                "--sam3-python",
                "/path/to/sam3-env/bin/python",
                "--dry-run",
            ]
        )
        == 1
    )

    output = capsys.readouterr().out
    assert "pid_mismatch" in output
    assert "already_running" not in output


def test_health_passes_expected_server_to_checker(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_health(url: str, expected_server: str | None = None) -> bool:
        calls.append((url, expected_server))
        return True

    monkeypatch.setattr(cli, "_http_health_check", fake_health)

    assert cli.main(["health", "sam3", "--state-dir", str(tmp_path)]) == 0
    assert calls == [("http://127.0.0.1:8773/", "sam3")]


def test_anyplace_health_uses_default_port_and_server_name(tmp_path: Path, monkeypatch) -> None:
    calls = []

    def fake_health(url: str, expected_server: str | None = None) -> bool:
        calls.append((url, expected_server))
        return True

    monkeypatch.setattr(cli, "_http_health_check", fake_health)

    assert cli.main(["health", "anyplace", "--state-dir", str(tmp_path)]) == 0
    assert calls == [("http://127.0.0.1:8775/", "anyplace")]


def test_contact_graspnet_health_uses_default_port_and_server_name(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls = []

    def fake_health(url: str, expected_server: str | None = None) -> bool:
        calls.append((url, expected_server))
        return True

    monkeypatch.setattr(cli, "_http_health_check", fake_health)

    assert cli.main(["health", "contact_graspnet", "--state-dir", str(tmp_path)]) == 0
    assert calls == [("http://127.0.0.1:8776/", "contact_graspnet")]


def test_default_state_dir_is_repo_relative() -> None:
    args = cli.build_parser().parse_args(["status", "sam3"])
    config = cli._build_configs(args)[0]

    assert config.state_dir == cli.REPO_ROOT / "outputs/mcp_services"
