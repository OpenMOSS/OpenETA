# Opposing-finger contact and recovery feedback — 2026-09-11

## Failure evidence

Goal 5 attempt 001 eventually passed, but requests 24 and 26 stopped with `constraint_escape_preview_rejected`. Saved-state replays reproduced 3-step and 1-step stops. Request 24's next candidate brought the palm into a non-target bowl (-1.108 mm signed distance; hard limit -1 mm), which was correctly unsafe. Request 26's candidate moved the palm away from the bowl, but normal contact between the two closed finger pads remained at about -0.58 micrometres. The recovery predicate required the minimum distance over boundary pairs to increase, which arm motion could not accomplish for this intentional internal contact. The strict QP also included those opposing-finger pairs; this diagnosis does not claim they require positive separation in every Mink inequality.

The first failure also exposed a feedback bug: the rejected candidate's actual hard collision was discarded when reporting the QP failure, producing `collision.detected=false` and generic controller advice.

## Changes

`robot_self_collision_groups` in `sim/controllers/collision_recovery.py` excludes only pairs between the left and right finger/fingerpad groups supplied by robot metadata. `mink_goal.py` uses these groups when constructing its self-collision limit. Missing or overlapping side definitions retain coverage. Every world pair and every other robot self pair remains protected. Gripper actuation physics, measurements, target provenance, thresholds and joint limits are unchanged. Normal opposing-finger closure is not external grasp evidence.

When a fallback candidate is rejected by geometric escape checks and actually crosses the hard collision threshold, `mink_goal.py` now preserves that preview's collision report alongside the controller error. `tools/codex_feedback.py` keeps the original `control_step_failed` reason and executed-step count, while selecting existing collision-specific recovery text when the collision flag is true. Joint-only or positive-distance boundary failures are not fabricated into hard collisions. Private object identities and geometry distances remain redacted.

No public tools, fields or enums were added. This is a checker-policy correction and requires three-person contract review before main-branch merge. The user authorized the private experiment fixes; no main changes, commits, pushes or external messages were made.

## Validation

The first focused suite passed 38 tests with 11 dependency skips in the lightweight environment. Simulator-environment runs covered the skipped MuJoCo/Mink cases (joint limits: 7 passed; gripper contact/fixture suite: 8 passed; checked gripper: 4 passed). A further feedback/motion-hook/placement/grouping suite passed 32 tests. Plugin validation and diff whitespace checks passed.

The final production replay compared collision-pair multisets with the old grouping and verified that exactly the intended three unique opposing-finger pairs (six entries, because the old self grouping includes both orders) were removed. No other pair was added or removed; world coverage was identical.

- Actual palm/bowl obstruction: still stops at 3 steps, but now reports pre-actuation palm collision outside the authorized target and preserves the controller-failure code.
- Previously rejected escape: reaches in 5 steps, 1.448 mm position error, 1.174 degree angle error. Joint and world checks remain active.

Real failed receipt → agent projection → motion summary → atomic feedback was also checked: collision-specific recovery survives without leaking private bowl identity or minimum distance. Both real diagnostic environments closed. These replays use zero model calls and do not count as new task successes.

Artifacts: `tmp/codex-goal5-contact-diagnosis-20260911/` (original), `tmp/codex-opposing-finger-diagnosis-20260911/` (candidate), `tmp/codex-opposing-finger-production-20260911/` (final production and redacted `agent-feedback.json`). Pre-feedback receipts are retained separately. Production plugin version: `0.1.0+codex.20260910190140`.
