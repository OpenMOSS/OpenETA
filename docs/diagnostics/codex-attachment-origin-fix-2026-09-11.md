# Carry proxy geometry origin correction — 2026-09-11

## Evidence

Goal 2 attempt 001 ultimately succeeded, but its first bottle lift failed before actuation despite bilateral finger-pad contacts. The close response reported `close_not_supported_by_host_contact_envelope`. The free-joint `position` was z=0.899310 m, near the bottle base, while the collision AABB spanned z=0.900315–1.057396 m. The grip site was z=1.028295 m. Distance to the asset origin was 128.986 mm; distance to the actual AABB centre was only 49.450 mm.

`_arm_attachment_proxy` had treated `position` as the centre both for the 120 mm proximity gate and for the box stored in the carry proxy. This missed the neck grasp and would also shift a successfully armed asymmetric asset's box away from its true geometry. The absent proxy disabled held-object stabilization and attached-object/world trajectory checks. The native model worked around the contact problem by repeatedly marking the bottle again. Official success does not establish that the missing checks were harmless.

## Change

`sim/mcp_server/server.py` now uses the canonical world AABB centre and extents consistently for proximity, grasp-relative offset and carry dimensions. Existing position/dimensions fallback remains for legacy geometry. The 120 mm gate, target provenance, fixture/receptacle exclusions, independent retention semantics and collision tolerances are unchanged. This does not solve every possible long-object grasp beyond the existing envelope. Named LIBERO bowls use their asset category (e.g. `akita_black_bowl`), so the generic `bowl` exclusion was not changed without evidence.

No public schema field or tool was added. `eef_to_target_distance_m` now measures the collision geometry centre when bounds exist. This private geometry correction should be reviewed with the simulator owner before any main-branch merge. No main branch or external document was modified.

## Validation

164 targeted tests passed, including real recorded geometry, AABB reconstruction, geometry origin invariance, remote geometry rejection, collision coverage, gripper latch, attachment lifecycle, fixture authorization and projection. The new regression also checks that an upper-bottle collision previously missed by an origin-centred box is detected.

A matched real-physics replay used the same saved first-lift state, target, IK seed, 150-step budget and collision checks. Original absent proxy reproduced `control_step_failed` at 0 steps. The host-generated corrected proxy reached the target in 19 steps with 1.467 mm Euclidean position error and 0.1515 degree orientation error, retained bilateral pad contact, and activated attached-object stabilization. Robot/world and carried-object/world predicted/actual checks remained enabled; the carried geometry's minimum recorded clearance was 5.729 mm. The real server endpoint predicate separately passed against the restored live scene. This is a controlled diagnostic with zero model calls, not an additional task success.

Artifacts: `tmp/codex-attachment-envelope-20260911/` (`wine-geometry.json`, `fixed-proxy.json`, `scene.json`, `endpoint-check.json`, `replay-report.json`, `replay.log`). Both diagnostic environments closed. Pass 9 completed with 12/20 successes and paused; next pass retries Goal 1 and continues Goals 3–9 with a fresh private plugin installation per attempt.
