# Bounded recovery implementation and validation — 2026-09-13

Implemented in the isolated `OpenETA-codex-plugin` worktree on
`dev/huaizezheng/codex-plugin-smoke-2026-09-08`. Existing uncommitted Host changes
were preserved. Pre-change runtime source is archived in
`tmp/codex-recovery-implementation-20260913/baseline-source.tar.gz`.

## Changes

1. Fixed a disagreement between two controller checks. When the collision QP
   needed a fallback, the segment checker demanded monotonic escape even if
   current authoritative geometry had no active recovery-boundary violation.
   The later checker already handled this case with ordinary hard-distance
   checks. Both now agree: escape requires an actual active boundary; otherwise
   the unchanged hard-distance, attachment, joint and post-step checks apply.
   In the diagnosed saved state, the current minimum was 3.151 mm, above the
   3 mm recovery boundary. No safety threshold or contact grant was widened.
2. Added bounded per-candidate diagnostics. Tracking and geometry are recorded
   separately; geometry skipped after tracking rejection is explicitly untested.
   Public projection includes candidate variant, scale, predicted robot step,
   bounded reason/obstacle categories and check outcomes. It excludes geometry
   identities, positions and QP matrices. `checks_passed` does not prove execution.
3. Added explicit atomic `move_to(motion_mode="recovery")`. It accepts one target
   within 150 mm, with no new contact point, route or symmetric orientation mode.
   It allows a 20 mm position corridor and 12 degree rotation-path corridor
   (18 mm / 10 degree candidate bounds); per-step limits and final pose
   tolerances are unchanged. Existing carried-object/contact checks remain.
   Ordinary moves default to strict with the previous 10 mm / 6 degree bounds.
4. Recovery can re-solve velocities with linearized world-frame step/position
   corridor bounds, then replace the endpoint posture attraction with current
   or bounded opposite posture biases, or a frame-only task. At most five
   alternatives are considered per tick, and each still passes nonlinear checks.
   This is local control, not a task planner or an automatically executed detour.
   Ordinary strict execution does not perform these alternate direction searches.
5. Host binds recovery to the exact requested pose through a private resolver,
   retires the binding at request completion and reports the selected bounds.
   Worker/controller reject unsupported modes or recovery without collision
   checking/tracking. Receipts identify verified QP fallback and re-solve counts.
6. Host separates preflight calls, physical dispatches, known physical steps and
   unknown outcomes. Remaining budget reports both stage limits and approximate
   strict/symmetric move capacity. The existing total budgets remain unchanged.
   Recent motion history reports measured displacement/rotation and outcomes;
   missing state remains unavailable and cannot mask a known motion result.

The plugin skill documents choosing recovery with visible clearance and checking
the actual endpoint. It contains no task coordinates, hidden object identities,
goal predicates or preset manipulation sequence. Plugin cachebuster:
`0.1.0+codex.20260913114937`. Interface additions require three-person review
before integration into main; main and global configuration were not modified.

## Validation

Artifacts: `tmp/codex-recovery-implementation-20260913/`.

- Host regression group: 162 passed, with the real stdio case selected separately.
- Simulator regression group: 60 passed. Groups overlap and are not additive
  unique counts. Final additive feedback tests are in `final-feedback-tests.log`.
- Real MCP stdio/disconnect test: 1 passed.
- Live native Host → simulator → worker recovery call: a 20 mm upward move
  reached the target and reported recovery mode; cleanup passed, no model called.
- Plugin/skill validation and `git diff --check` passed.

Eleven saved-state cases were paired against archived pre-change source. Four
previously successful cases retained identical joint traces. Nine additional
historical successful cases also retained identical traces: **13/13 preserved**.

| Local case | Before | New strict | Explicit recovery |
|---|---|---|---|
| Goal 5 request 40 | Rejected at 0 steps | Reached in 23 steps | Reached in 23 steps |
| Goal 5 request 42 | Rejected at 0 steps | Reached in 24 steps | Reached in 24 steps |
| Long 3 request 48 | Segment blocked | Segment blocked | Reached in 43 steps |

Long 3 request 48 had 10.61 mm peak cross-track displacement, within its explicit
20 mm recovery corridor. Other stalled cases remained failures. Goal 5 request
30 used alternate recovery solves but still failed, with 18.12 mm peak deviation.
Thus recovery is bounded and useful for some states, not a universal escape.

A prototype that also re-solved strict moves caused an additional path deviation
in one failed case; alternate searches were restricted to explicit recovery and
the final paired tests rerun. The ordinary successful trajectories remain intact.
Snapshots have the existing restore-scope limits; paired local results do not
claim bitwise reproduction of every historical failed rollout or benchmark gain.

Fresh full-task retests are now complete: Goal 5 passed and Long 3 failed at the
160-stage budget boundary; both integrations passed. See the separate
[full-task report](codex-recovery-retest-2026-09-13.md) and artifacts under
`tmp/codex-recovery-retest-20260913/`. These selected attempts do not establish
a new benchmark success rate or isolate the effects of individual changes.
