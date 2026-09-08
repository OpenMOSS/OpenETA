"""Host-supplied display correlation for isolated provider invocations.

Not authorization, cancellation ownership, or a cross-request lifecycle lease.
Never derive a parent from nested evidence/history fields.
"""
from uuid import uuid4


def isolated_request_lineage(parent_session_id: object) -> dict:
    if (
        not isinstance(parent_session_id, str)
        or not parent_session_id.strip()
        or len(parent_session_id) > 256
        or any(ord(char) < 32 or ord(char) == 127 for char in parent_session_id)
    ):
        return {}
    return {"request_lineage": {
        "parent_session_id": parent_session_id.strip(),
        "child_session_id": f"isolated-{uuid4().hex}",
    }}
