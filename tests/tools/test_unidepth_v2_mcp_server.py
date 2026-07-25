from __future__ import annotations

from tools import unidepth_v2_mcp_server as server


def test_estimate_depth_reports_unconfigured_backend(monkeypatch) -> None:
    monkeypatch.setattr(server, "_BACKEND", None)

    result = server.estimate_depth(
        {"format": "png", "base64": "unused"},
        {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
    )

    assert result["success"] is False
    assert result["details"]["reason"] == "model_load_failed"


def test_estimate_depth_forwards_agent_contract(monkeypatch) -> None:
    calls = []

    class Backend:
        def estimate_depth(self, **kwargs):
            calls.append(kwargs)
            return {"success": True, "details": {"confidence_semantics": "higher_is_better"}}

    monkeypatch.setattr(server, "_BACKEND", Backend())
    result = server.estimate_depth(
        {"format": "png", "base64": "abc"},
        {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0},
        camera_id="wrist",
        resolution_level=6,
    )

    assert result["success"] is True
    assert calls[0]["camera_id"] == "wrist"
    assert calls[0]["resolution_level"] == 6
