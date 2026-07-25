from __future__ import annotations

import sys

import pytest

import real.mcp.observation_server as observation_server


@pytest.mark.parametrize("run_error", [None, KeyboardInterrupt()])
def test_stdio_server_closes_manager_when_transport_stops(monkeypatch, run_error) -> None:
    events: list[str] = []

    class FakeManager:
        def close(self) -> None:
            events.append("close")

    manager = FakeManager()
    monkeypatch.setattr(
        observation_server,
        "RealEnvManager",
        lambda *_args, **_kwargs: manager,
    )
    monkeypatch.setattr(observation_server, "_build_env_factory", lambda _args: object())

    def run(*, transport):
        events.append(f"run:{transport}")
        if run_error is not None:
            raise run_error

    monkeypatch.setattr(observation_server.mcp, "run", run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["observation_server", "--transport", "stdio"],
    )

    if run_error is None:
        assert observation_server.main() == 0
    else:
        with pytest.raises(KeyboardInterrupt):
            observation_server.main()

    assert events == ["run:stdio", "close"]
