"""Explicit acknowledgements for remote environment cleanup."""

from __future__ import annotations


def close_response_error(response: object) -> str | None:
    """No transport exception is not the same as confirmed resource retirement."""
    if not isinstance(response, dict):
        return f"close returned {type(response).__name__}, expected an acknowledgement object"
    if (response.get("error") or response.get("isError") is True
            or response.get("cleanup_errors") or response.get("ok") is False
            or response.get("success") is False or response.get("pending") is True):
        return str(response.get("error") or response.get("cleanup_errors") or "close not confirmed")
    if response.get("ok") is True or response.get("already_closed") is True:
        return None
    return "close response lacks an explicit ok/already_closed acknowledgement"
