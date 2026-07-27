from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from agent.backends.planner import (
    PlannerBackend,
    PlannerBackendRequest,
    PlannerBackendResult,
    ProviderConcurrencyLimiter,
    ProviderQueueTimeoutError,
)


class _TrackingBackend(PlannerBackend):
    def __init__(self, *, delay_s: float = 0.03) -> None:
        self.delay_s = delay_s
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def decide(self, request: PlannerBackendRequest) -> PlannerBackendResult:
        del request
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            time.sleep(self.delay_s)
            return PlannerBackendResult(payload={"kind": "response", "name": "talk"})
        finally:
            with self.lock:
                self.active -= 1


class _BlockingBackend(PlannerBackend):
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def decide(self, request: PlannerBackendRequest) -> PlannerBackendResult:
        del request
        self.entered.set()
        self.release.wait(timeout=1.0)
        return PlannerBackendResult(payload={"kind": "response", "name": "talk"})


def test_provider_limiter_bounds_shared_backend_concurrency() -> None:
    backend = _TrackingBackend()
    limiter = ProviderConcurrencyLimiter(2, queue_timeout_s=1.0)
    wrapped = limiter.wrap(backend)
    request = PlannerBackendRequest(tool_context={})

    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(lambda _: wrapped.decide(request), range(6)))

    metrics = limiter.snapshot()
    assert backend.max_active == 2
    assert metrics["max_active"] == 2
    assert metrics["active"] == 0
    assert metrics["request_count"] == 6
    assert metrics["queue_timeout_count"] == 0
    assert all(result.details["provider_concurrency"]["limit"] == 2 for result in results)


def test_provider_limiter_times_out_before_calling_queued_backend() -> None:
    backend = _BlockingBackend()
    limiter = ProviderConcurrencyLimiter(1, queue_timeout_s=0.02)
    wrapped = limiter.wrap(backend)
    request = PlannerBackendRequest(tool_context={})

    with ThreadPoolExecutor(max_workers=2) as executor:
        running = executor.submit(wrapped.decide, request)
        assert backend.entered.wait(timeout=1.0)
        with pytest.raises(ProviderQueueTimeoutError) as raised:
            wrapped.decide(request)
        backend.release.set()
        running.result(timeout=1.0)

    assert raised.value.code == "provider_queue_timeout"
    metrics = limiter.snapshot()
    assert metrics["request_count"] == 2
    assert metrics["queue_timeout_count"] == 1
    assert metrics["max_active"] == 1
