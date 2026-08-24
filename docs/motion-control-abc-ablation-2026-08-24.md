# Motion control ABC ablation — 2026-08-24

Status: implementation and canary evidence complete; default remains condition
A pending three-person review of the shared response semantics and condition-C
execution bundle.

## Question

Does the waypoint mechanism fail because waypoints are unhelpful, or because
the current executor treats independently previewed endpoints as if they formed
a dynamically consistent route?

The comparison isolates three host-selected conditions. The Agent cannot choose
or modify the condition during a session.

| Condition | Execution behavior |
| --- | --- |
| A | Existing Mink behavior. Each waypoint uses the previously compiled receipt without a stable-arrival hold or route-local seed chain. |
| B | A plus lower joint-velocity limits, three consecutive in-tolerance/low-velocity arrival steps, and aligned-progress/cross-track stall diagnostics. |
| C | B plus just-in-time preview of each waypoint from the actual preceding endpoint and a host-private per-waypoint IK seed chain. |

The configuration is selected with
`OPENETA_MOTION_EXPERIMENT_CONDITION=A|B|C`. Invalid values fail closed. A is
still the default, so the experiment does not silently change production
behavior.

## Implementation boundaries

- The Agent still chooses the number and geometry of waypoints. No
  hover/align/contact/place phase or semantic waypoint type was introduced.
- Condition C resolves the Agent's short receipt ids into an exact host-private
  route bundle. It checks target pose, captured preserve-current orientation,
  tolerance, robot epoch, and object-scene epoch.
- Immediately before segment *i*, C previews that segment from the simulator's
  actual endpoint of segment *i-1*. A failed preview stops the route before that
  segment is executed.
- Joint seeds are execution hints, not Agent-visible privileged state. They are
  never added to Agent-authored tool parameters.
- Collision safety remains worker-local and is checked before and after every
  controller step. Sequential endpoint preview is not claimed to be a global
  collision-free path planner.
- `follow_eef_trajectory` now preserves an authoritative worker
  `reached_target=false`; the wrapper no longer overwrites it using a final
  position-only approximation.
- The compact motion result now reports requested/completed waypoint counts and
  per-waypoint attainment, stop reason, residual, collision, and controller
  receipt. Large raw responses remain durable artifacts.

## Deterministic simulator canary

Environment: `openeta/libero_libero_object_task2-v0`, seed 2, local LIBERO Mink
service. Each condition started from a newly created environment. Complete
responses are in:

- `tmp/motion-control-abc-A-final-20260824.json`
- `tmp/motion-control-abc-B-final-20260824.json`
- `tmp/motion-control-abc-C-final-20260824.json`

| Scenario | A | B | C |
| --- | --- | --- | --- |
| 2 cm short upward move | success, 2 steps | success, 5 steps, 3 stable steps | success, 5 steps, 3 stable steps |
| Same clear three-point route | **1/3**, iteration limit, 27.29 cm final Euclidean error | **1/3**, iteration limit, 27.20 cm error; first point stably attained | **3/3**, 120 steps, 0.99 cm final Euclidean error; 3/3 stable and 3/3 JIT-authorized |
| Deliberately unsafe crossing route | stopped at 2/3 by collision check | stopped at 2/3 by controller/safety infeasibility | stopped at 2/3 by collision check |

The object-displacement probe showed the same approximately 18.2 mm reset-time
settling of ketchup/salad-dressing in all three conditions. It is useful only as
a relative check in this canary; there was no condition-specific additional
object displacement.

Interpretation:

1. B verifies that stable arrival and controlled velocity work, but those alone
   do not repair a route whose independent IK solutions occupy inconsistent
   redundancy basins.
2. C repairs the clear route by solving each segment against the state that
   actually resulted from the prior segment and keeping execution in that
   solution basin.
3. C does not make an unsafe user-selected polyline safe. The existing
   per-step worker collision contract continues to stop it. A future route
   planner or Agent-selected detour is still needed for obstacle avoidance.

## Provider-driven fixed-route comparison

The evaluation harness sent the same fixed three-point route, task, seed,
budgets, and instructions to `gpt-5.6-luna` under each condition. The provider
was authorized for this experiment. Experiment-local extraction reads durable
rollout artifacts rather than adding ABC-specific fields to the universal eval
schema.

| Condition | Controller result | Agent terminal result | Planner calls to motion | Tokens to motion |
| --- | --- | --- | ---: | ---: |
| A | 1/3, iteration limit, 27.21 cm max-axis final error | fail | 6 | 132,955 |
| B | 1/3, iteration limit, 27.11 cm max-axis final error; 1 stable waypoint | fail | 6 | 132,324 |
| C | 3/3, 0.91 cm max-axis final error; 3 stable/JIT-authorized waypoints | success | 6 | 132,957 |

Run ids:

- `motion-control-abc-fixed-A-20260824`
- `motion-control-abc-fixed-B-20260824`
- `motion-control-abc-fixed-C-feedback-r2-20260824`

The first C execution already completed 3/3, but the compact tool result omitted
the intermediate waypoint receipts. The Agent conservatively returned failure
because it could only verify the final endpoint. After the additive compact
feedback fix, the repeated C run returned `task_complete` and explicitly cited
`waypoints_requested=3`, `waypoints_completed=3`, collision-free execution, and
the current views. This separates a harness-observability defect from a control
defect.

The official LIBERO task reward remains zero because this is a motion-only
canary, not an attempt to complete the pick-and-place objective. The relevant
success criterion is the exact requested route plus the controller and
collision receipts.

## Review decision requested

Before changing the default from A, three-person review should cover:

1. the host-private route-bundle and per-waypoint seed authority boundary;
2. preserving authoritative worker failure instead of wrapper position-only
   success reconstruction;
3. the additive per-waypoint compact response exposed to the Agent; and
4. whether C should become the default for `follow_eef_trajectory`, while
   retaining worker-local collision checking and making no global path-safety
   claim.

The evidence supports C for multi-waypoint execution. It does not yet establish
that a waypoint route is better than direct motion for every grasp task; that
requires a paired end-to-end task suite with obstacle-density and carried-object
strata.
