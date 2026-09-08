"""Ownership rules for Agent-authored notes, separate from host evidence.

These helpers do not infer trust from an entry's value. Only the host-created
outer provenance determines whether a legacy entry belongs to the Agent.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


AGENT_MEMORY_SOURCE = "save_memory"


def is_agent_entry(entry: Any) -> bool:
    return isinstance(entry, dict) and entry.get("source") == AGENT_MEMORY_SOURCE


def agent_entry(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        **deepcopy(entry),
        "source": AGENT_MEMORY_SOURCE,
        "ownership": "agent",
        "freshness": "agent_managed",
    }


def separate_legacy_agent_entries(
    evidence: dict[str, Any], notes: dict[str, Any]
) -> list[str]:
    """Move old note mirrors out of a host store without upgrading their trust.

    An explicit notes snapshot wins over its older compatibility mirror. A
    overwritten host fact cannot be reconstructed from a note; callers must
    reacquire that evidence rather than promote the note back into authority.
    """

    migrated = []
    for key, entry in list(evidence.items()):
        if is_agent_entry(entry):
            notes.setdefault(key, agent_entry(entry))
            del evidence[key]
            migrated.append(key)
    return migrated
