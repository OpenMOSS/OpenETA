# Attachment probe short-ID IK handoff

Status: three-person review approved on 2026-08-24; ready for live validation.

## Problem

`prepare_attachment_probe` previously returned each frozen IK target as a nested
`target_pose`. Large-tool-result projection intentionally omitted that deep pose,
while the handoff correctly instructed the Agent not to copy frozen geometry.
The Agent therefore knew that a probe existed but could not issue the required
IK preview safely.

## Agent-facing request

The prepared probe now returns ordered requests in this form:

```json
{
  "tool": "ik_preview_check",
  "parameters": {
    "probe_id": "probe:<sha256>",
    "waypoint_index": 0,
    "position_tolerance_m": 0.01,
    "orientation_tolerance_rad": 0.1,
    "check_endpoint_collision": true
  }
}
```

`waypoint_index` is zero-based. Linear probes expose index `0`; arc probes expose
ordered indices `0..N-1`. The Agent selects and submits the returned references
in order. It never copies `target_pose`.

## Host resolution

`AgentMemory.resolve_attachment_probe_waypoint` resolves the public pair to the
immutable pose in the active probe. It fails closed when:

- `probe_id` is missing, unknown, completed, or superseded;
- `waypoint_index` is not an integer or is outside the returned range;
- object-scene or robot-motion epoch differs from probe preparation;
- the frozen path or its SHA-256 marker is missing or inconsistent.

Successful resolution injects only the exact target pose into the simulator IK
handler. The action/conversation ledger retains the short public request.
`move_to` and `follow_eef_trajectory` remain receipt-only.

## Context projection

Both the recent conversation projection and the latest-tool-result projection
preserve `probe_id`, ordered `ik_preview_requests`, and `execution_handoff`, while
omitting the large `frozen_path`. Thus the handoff remains executable without
duplicating pose matrices in planner context.

## Contract status

This is an additive fourth request branch for `ik_preview_check`, alongside:

1. Agent-authored `target_pose`;
2. `compiled_grasp_id + waypoint_role`;
3. `viewpoint_proposal_id + candidate_id`;
4. `probe_id + waypoint_index`.

The branch changes the verified ToolContract catalog hash. Three-person review
accepted the additive schema branch on 2026-08-24. The authority baseline and
request-validation canaries must therefore be regenerated against the reviewed
catalog hash before live validation.

## Validation

- attachment-probe, resolver, contract, planner, conversation, and decision
  context tests: 251 passed;
- broader runtime/fixture/proxy tests: 183 passed, with only the expected stale
  reviewed-catalog audit failing;
- `python -m compileall agent` and `git diff --check` are part of final local
  validation.
