"""Host-owned, process-local admission quota for executable tool attempts."""

import threading


class ToolCallBudget:
    """Reserve before authorization/dispatch; failures do not refund attempts.

    Carried usage is conservatively charged from the legacy episode attempt
    counter. It is not newly measured dispatch evidence from a previous run.
    """

    def __init__(self, limit: int, *, carried_usage: int = 0) -> None:
        if type(limit) is not int or limit <= 0:
            raise ValueError("Tool-call budget limit must be a positive integer")
        if type(carried_usage) is not int or carried_usage < 0:
            raise ValueError("Carried tool usage must be a non-negative integer")
        self._limit = limit
        self._carried_usage = carried_usage
        self._admitted = 0
        self._denied = 0
        self._lock = threading.Lock()

    def reserve(self) -> bool:
        with self._lock:
            if self._carried_usage + self._admitted >= self._limit:
                self._denied += 1
                return False
            self._admitted += 1
            return True

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "limit": self._limit,
                "carried_usage": self._carried_usage,
                "admitted_this_run": self._admitted,
                "denied_this_run": self._denied,
                "remaining": max(0, self._limit - self._carried_usage - self._admitted),
            }
