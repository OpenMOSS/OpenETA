# Recovery design findings after the two-task retest

This is a diagnosis and design proposal, not an implemented controller change.
See [the retest](codex-feedback-retest-2026-09-13.md). Main was untouched; no new
Agent trial or model request was launched for this analysis.

## Evidence

Goal 5 failed well before its budget limit. Long 3 used three native waypoint
routes, eventually recovered from the rack, and later stopped at 159/160 internal
calls. Neither outcome is explained simply by absence of waypoint support or
network interruption. Feedback was delivered, but selecting a useful recovery
motion remained difficult.

Native summaries report position-step restrictions on seven Goal 5 calls and
position-corridor restrictions on two. Long 3 has seven position-corridor cases.
These are nonexclusive aggregated reasons, not independent causal diagnoses.

The current controller solves a velocity QP and then tries six scalar multiples
of that same velocity, checking tracking and geometry in order. It already has
collision/joint limits and an IK-seed posture task. The remaining issue to study
is agreement between those objectives/constraints and the nonlinear checks,
including recovery near existing constraints. Merely adding another posture
task or increasing the horizon would not address all these cases.

An operator-only headless reconstruction of Goal 5's final zero-step rejection
(`1789297168178565913-dbe1189e-move-start`) refused any physics callback. The
unscaled candidate moved the predicted EEF 12.09 mm and failed the 8 mm step
bound. The five smaller production candidates passed tracking, but failed
geometry at 6.02, 3.01, 1.50, 0.75 and 0.375 mm. No action was executed, and the
environment was cleaned up. Artifacts: `tmp/codex-recovery-design-20260913/`.

The reconstructed failure includes `robot_collision` with no candidate obstacle
classification, whereas the original native summary emphasized tracking. This
is a diagnostic discrepancy to resolve using per-candidate evidence and state
parity checks; the reconstruction is not proof of the original rejection's
precise geometric cause. It does disprove the assumption that the step bound
alone explains every rejected candidate in this reconstruction.

## Proposed order

1. Capture per-candidate rejection stages and current-to-proposed changes, privately
   retaining full solver data. Expose only bounded categories and meaningful
   magnitudes. Diagnose the mismatch between QP feasibility, actual tracking and
   nonlinear geometry checks before changing safety thresholds.
2. Test controller recovery that can change the candidate velocity direction or
   null-space posture, rather than only scaling one direction. Incorporate local
   tracking bounds into the solve where practical, retaining the final geometry
   check. Keep strict ordinary moves; make any wider recovery corridor explicit,
   bounded and Agent-selected. Previously reached robot states can inform retreat
   proposals, but require fresh checks because the scene/load may have changed.
3. Separate native decisions, IK/preflight work and physical execution budgets.
   Report remaining estimated move capacity and retain total time/internal caps.
   Add compact progress history so the Agent can recognize repeated negligible
   motion and change approach, without a hidden automatic task policy.

Validate on saved blocked approaches/retreats and prior successful local moves
first: count recoverable cases, actual safety violations, cross-track/angular
excursions, final error and internal cost. Then run fresh independent tasks with
the protocol frozen. Do not relax collision authorization, expose hidden geometry,
or infer benchmark improvement from this selected pair. Any interface additions
need contract review before main integration.
