# Held-object orientation in endpoint collision checks

Object 4 attempt 003 completed with official task success false after 75 native requests / 157 internal tool calls, 1679.486 seconds, and zero stream errors. The model called finish_episode; this was not a network or integration failure. All owned processes exited, the private authentication copy was removed, port 18786 was released, and source files remained unchanged during the attempt. Audited campaign success remains Object 0–3 (4/20).

## Evidence and cause

The host attachment proxy captured a world-frame relative offset and world AABB dimensions at gripper closure, but carried no orientation anchor. Mink endpoint checks translated the original box even after the held bottle rotated. Saved-state MuJoCo FK (no stepping, model calls or live-task modification) found the old proxy centre 53.683 mm from the actual collision-bound centre, and its X width remained 150.393 mm while the actual width was 61.900 mm.

Evidence: `tmp/codex-held-rotation-diagnosis/geometry.py`, `geometry.json`, and the three named saved-state references in that JSON. This is operator diagnosis; simulator identities and geometry are not given to the robot agent.

## Repair

On close, retain the measured EEF body quaternion as private `anchor_eef_quat_xyzw`. For an endpoint with body rotation R and grasp anchor R0, rotate the offset by D = R R0^T, and take dimensions abs(D) * original_dimensions. This equals the AABB of all eight rigidly transformed original box corners. Evaluate the baseline overlap using the separately measured current rotation. Position-only endpoints use current orientation; explicit targets use their target orientation.

Only the Mink endpoint path supplies the new quaternion arguments. Legacy callers without an anchor keep their translation-only behavior; the OSC sweep has not been migrated. Exact per-step MuJoCo collision checks, contact authorization, thresholds, official success checking, and physics are unchanged. The tentative attachment remains tentative: pad contact or a proxy does not prove retention.

Historical replay approximates the missing anchor with the first saved post-close body quaternion. The rotated proxy centre error becomes 6.181 mm (previously 53.683 mm). Its dimensions remain conservative (X 127.017 mm versus actual 61.900 mm); rotating a world AABB cannot recover exact mesh geometry or account for slip. This fix must not be described as exact tracking of a rigidly held object.

## Controller observation

Requests 54 and 55 triggered `mink_controller_diverged` after 8 steps each: Cartesian errors 101.178 mm and 113.221 mm. This is the >100 mm drift guard, not a reported QP no-solution error. The first target IK seed touches joint 6's upper bound, so endpoint margin remains a possible contributor. Later subdivided angular moves converged, e.g. requests 57–59 reached with 0.085/0.091/0.053 degree final errors. No controller law was changed in this repair.

## Validation and deployment

`PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest -q tests/test_attachment_rotation.py tests/test_attached_object_collision.py tests/test_codex_placement_feedback.py tests/test_codex_failure_feedback.py tests/test_simulator_mcp_proxy.py`: **124 passed in 2.02 s**. Tests cover arbitrary non-identity anchors via independent Rodrigues/corner calculations, quaternion scale/sign equivalence, corrected corridor fit, newly introduced collisions, orientation-aware baseline egress, legacy behavior, measured anchor capture and server target/current orientation routing. `git diff --check` and plugin validation passed.

Plugin cachebuster: `0.1.0+codex.20260910163753`. Campaign launcher reinstalls into a fresh private Codex home and starts a new exec thread per attempt. No global Codex configuration or marketplace changes.

Private contract addition example: `attachment_proxy.anchor_eef_quat_xyzw = [0, 0, 0, 1]` (EEF body quaternion, xyzw). Backward compatible; mark for collaborator three-person schema review before merging into main. Work remains solely in the authorized private branch; no main merge or external document publication.
