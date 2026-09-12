# Empty-gripper orientation hold — 2026-09-11

Goal 3 attempt 004 ended task-false after 1126.02 s, 77 native / 141 internal calls and no stream errors. Integration, source and cleanup checks passed. The drawer was opened but the bowl was not grasped. The native model ended the task before either budget expired.

## Cause and bounded change

For a validated IK-seeded Cartesian goal, the original position cost was 500 and orientation cost 20. Empty-gripper translations near the cabinet could trade excessive angular deviation for positional progress despite an almost unchanged requested orientation. This is a trajectory tracking issue, not proof the endpoint IK is unreachable.

`sim/controllers/orientation_tracking.py` now selects the existing 25x hold weight when the initial requested rotation is finite and within 0.05 rad, including empty-gripper translations. Deliberate rotations outside that band and unseeded execution keep the baseline weight. `mink_goal.py` applies it independently of grip evidence. Empty-arm speed, arrival criteria, target pose, IK seed, collision/contact authorization and iteration budget remain unchanged. Existing grip telemetry still reports no grip; separate controller receipt fields identify the orientation weight and hold activation.

## Matched real-physics evidence

All cases restore the same Goal 3 attempt 004 snapshot and use its unchanged target, seed, 150-step budget, 0.5 rad/s velocity and collision checks. No model calls or private operator hints.

| Saved request | Before | Actual fixed controller |
| --- | --- | --- |
| 43: bowl preapproach | collision at step 4, 16.313 degree angle error | reached in 27 steps, 1.604 mm / 0.339 degree |
| 48: another bowl approach | collision at step 13, 10.411 degree | still collision at step 15, 1.458 degree |
| 41: deliberate large rotation | reached in 40 steps, 2.485 degree | identical 40 steps, 2.485 degree; multiplier 1 |

Request 48 remains obstructed by real cabinet geometry. Increasing orientation weight does not remove that obstacle or grant contact. Ablation factors 1/5/25 and complete receipts are preserved under `tmp/codex-empty-orientation-diagnosis-20260911/`. Actual production replay with negative control is under `tmp/codex-empty-orientation-production-20260911/`; complete and cleanup flags true.

72 unit/regression tests passed, covering orientation selection, fixture stabilization, body/site frame conversion, motion hook, collision feedback, placement feedback and opposing-finger collision coverage. Plugin validation and `git diff --check` passed. Plugin cachebuster: `0.1.0+codex.20260910194628`; next native attempt uses a fresh private installation.

This changes controller behavior and requires the project three-person policy review before main merge. It is a private development change, not a new public tool or a frozen-version benchmark claim.
