"""Observation-only OpenETA runtime with no action surface."""

from __future__ import annotations

from agent.runtime.observation_interfaces import (
    ReadOnlyObservationBackend,
    ReadOnlyObservationContext,
    ReadOnlyObservationReport,
    validate_readonly_context,
    validate_readonly_report,
)


class OpenEtaReadOnlyObservationRuntime:
    def __init__(self, backend: ReadOnlyObservationBackend) -> None:
        self._backend = backend
        self.observe_call_count = 0

    def observe(self, context: ReadOnlyObservationContext) -> ReadOnlyObservationReport:
        validate_readonly_context(context)
        self.observe_call_count += 1
        if self.observe_call_count != 1:
            raise RuntimeError("read-only runtime permits exactly one observe call")
        report = self._backend.observe(context)
        if not isinstance(report, ReadOnlyObservationReport):
            raise TypeError("read-only backend must return ReadOnlyObservationReport")
        validate_readonly_report(report)
        return report
