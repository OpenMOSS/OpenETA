# Controller capability contract and staged Mink integration

Status: phase 0 implementation candidate; controller replacement requires
three-person RFC review.

## Decision summary

The Agent-facing `move_to` contract remains controller-neutral. The controller
profile is selected by the host when an environment is created and cannot be
changed by the Agent during a session. Every environment and motion result must
truthfully identify the configured low-level controller, command interface,
goal executor, and verified safety coverage.

Mink must execute inside the simulator bench worker. The current outer MCP
server only has HTTP `step`/`observe` access; Mink requires the live MuJoCo
model/data, Panda joint indices, EEF site, controller output limits, and control
frequency. Reimplementing those through the outer step loop would duplicate
privileged simulator state and make collision/contact receipts unreliable.

## Current boundary audit

### 1. Trajectory and collision

The current `move_to` is an outer-server OSC loop. It predicts one small
Cartesian batch for the attached-object AABB before sending actions, then checks
the resulting joint configuration with cuRobo after a batch. The public receipt
correctly does not claim full trajectory/world coverage.

A production Mink executor needs a worker-side pre-actuation callback for every
joint-velocity step. It must distinguish joint/configuration limits,
self-collision, robot/world collision, and attached-object/world collision. A
missing checker remains `UNKNOWN`; it must never be represented as collision
free. The outer server may still enforce host evidence and attachment policy,
but must not claim it checked a worker-local path it never observed.

### 2. Manipulation contact

The controller remains fixed for the environment. We should not hot-swap OSC
and JOINT_VELOCITY controllers at hover/contact boundaries: that would restore
an implicit phase machine and robosuite does not guarantee equivalent controller
state across a live swap.

Mink contact validation therefore needs its own canary using the same
JOINT_VELOCITY environment from free-space approach through close and lift. The
final contact segment uses a bounded velocity profile and the existing binary
gripper latch. Contact success is still judged from actual EEF feedback and
fresh dual-view attachment evidence, not from a controller success flag.

### 3. Dependency and runtime

The endpoint canary used Mink 0.0.6, qpsolvers 4.13.0, and quadprog 0.1.13 with
the existing MuJoCo 3.3.0 runtime. This is a compatibility profile, not a global
planner dependency. The LIBERO worker owns these packages. For isolated
canaries, the manager continues to strip generic `PYTHONPATH` and the worker
accepts only the host-owned `OPENETA_LIBERO_MINK_DEPENDENCY_PATH`; this avoids
changing package resolution for other benches or accepting a dependency path
from an Agent tool argument.

If a requested controller profile is unavailable or its version fingerprint is
wrong, environment creation fails with an actionable capability error. It must
not silently fall back to OSC, because that would invalidate A/B identity and
make controller receipts false.

### 4. Simulator/real-robot boundary

`move_to` shares request and outcome semantics across backends, not an
implementation. A real robot may use a vendor Cartesian or joint controller;
it must publish its own controller id, command interface, feedback source,
tolerances, and safety coverage. Simulator-only joint state or privileged FK is
not added to the Agent contract. Requested/actual EEF pose and camera feedback
remain the portable recovery evidence.

## Capability and execution receipts

`openeta.sim_control.v1` is extended additively with a `controller` block:

```json
{
  "schema_version": "openeta.sim_control.v1",
  "controller": {
    "controller_id": "robosuite.osc_pose",
    "configured_name": "OSC_POSE",
    "command_interface": "normalized_cartesian_delta_pose",
    "goal_executor": "openeta.outer_closed_loop_cartesian.v1",
    "execution_location": "mcp_server",
    "supports_position": true,
    "supports_orientation": true
  },
  "cartesian_delta": {
    "supported": true,
    "position_indices": [0, 1, 2],
    "rotation_indices": [3, 4, 5],
    "command_frame": "world",
    "position_scale_m": 0.05,
    "rotation_scale_rad": 0.5
  },
  "gripper": {
    "supported": true,
    "indices": [6],
    "open_value": -1.0,
    "close_value": 1.0
  }
}
```

Each motion returns `openeta.controller_execution_receipt.v1` with the exact
controller identity, executor, requested orientation policy, iteration budget,
steps executed, stop reason, and final attainment. This receipt says which
controller ran; it does not expand collision coverage.

For LIBERO phase 0, missing or non-OSC control capability fails before
`move_to`; an old worker can no longer be silently driven using hard-coded
action assumptions. This is intentionally a deployment compatibility break for
the not-yet-released Stage 2 harness.

## Staged validation

1. **Phase 0 — capability truthfulness:** expose the actual OSC control spec,
   fail closed on missing/mismatched capability, retain controller receipts in
   context and durable rollout, and add auditor coverage.
2. **Phase 1 — worker-local Mink free-space:** add a host-configured LIBERO Mink
   profile and worker goal endpoint; rerun paired endpoint and unreachable-pose
   canaries. No contact claim.
3. **Phase 2 — collision/contact:** add stepwise collision callbacks and paired
   approach/contact/close/lift cases, including attached-object egress. Do not
   proceed if collision scope is ambiguous.
4. **Phase 3 — Agent A/B:** run paired full grasp episodes with identical tasks,
   seeds, provider/model, perception artifacts, and turn/tool budgets. Compare
   endpoint attainment, collision stops, stable attachment, recovery turns,
   task success, and runtime.
5. **Production decision:** only after phase 2 and phase 3 evidence, propose a
   default-controller change to the shared RFC. Keep controller selection
   host-owned and immutable within a session.

## Phase 0 live evidence — 2026-08-19

`scripts/controller_capability_canary.py` ran against the local SSE service and
created `openeta/libero_libero_object_task2-v0` at seed 2. The complete result is
stored at `tmp/controller-capability-canary-20260819.json`.

- worker creation returned `robosuite.osc_pose`,
  `normalized_cartesian_delta_pose`, and
  `openeta.outer_closed_loop_cartesian.v1`;
- a preserve-current-orientation target 2 cm above the initial EEF was
  independently reported reachable by `ik_preview_check`;
- OSC reached the target in 6/40 iterations with 2.995 mm Euclidean position
  error at the requested 3 mm per-axis tolerance;
- `openeta.controller_execution_receipt.v1` exactly matched the returned
  controller id, step count, stop reason, and attainment flag;
- cleanup succeeded and released the simulator environment.

Full repository regression after phase 0: `1180 passed, 12 skipped`.

## Phase 1 worker-local evidence — 2026-08-19

The experimental `mink_joint_velocity` profile is host-selected with
`OPENETA_LIBERO_CONTROLLER_PROFILE`; it is not an Agent tool argument and does
not change the default OSC deployment. A separate local service on port 8767
used the isolated dependency directory from the endpoint canary.

The first environment-create attempt failed closed because the worker manager
intentionally strips generic `PYTHONPATH`; the response named all three missing
packages and explicitly stated that no OSC fallback was attempted. The worker
bootstrap was then limited to the dedicated host-owned
`OPENETA_LIBERO_MINK_DEPENDENCY_PATH` rather than weakening general import
isolation.

Two live canaries then passed:

- **free-space execution:** the worker declared
  `mink.robosuite_joint_velocity` and `openeta.worker_mink_goal.v1`; the same
  2 cm upward target was independently IK reachable and reached in 3/40 steps
  with 0.388 mm position error. The worker-local controller receipt matched the
  returned steps, stop reason, and attainment exactly. Result:
  `tmp/controller-capability-mink-phase1-20260819.json`.
- **safety refusal:** with normal collision checking requested, the worker
  returned `mink_collision_callback_unavailable`, zero executed steps, and
  `safety_capability_unavailable`. No motion or OSC fallback occurred. Result:
  `tmp/controller-capability-mink-safety-rejection-20260819.json`.

The current Mink route also refuses a confirmed attached-object motion before
execution. It therefore remains a free-space experiment and cannot yet be used
for an Agent grasp episode. Phase 2 must define the worker-local collision
callback and intentional target-contact policy before relaxing either gate.

Full repository regression after phase 1: `1182 passed, 12 skipped`.

## Phase 2 collision/contact evidence — 2026-08-19

The worker-local Mink profile now builds collision pairs from actual MuJoCo
body ancestry. Panda arm geoms always avoid the world; only the gripper subtree
may contact the one object resolved from a current host-owned compiled-grasp
authorization. The Agent cannot manufacture that authorization. Pre-actuation
predictions and actual post-step configurations are checked on every controller
step. Free-joint object velocities are fixed in the QP so the preview cannot
make an obstacle move hypothetically to satisfy a constraint.

The deterministic contact canary at
`tmp/mink-contact-canary-r6-20260819.json` passed clearance, precontact,
unauthorized-contact refusal, authorized contact, binary close, and 12 cm lift.
The object rose 11.76 cm and remained 4.21 cm from the EEF. Robot trajectory
coverage is complete for that path; the receipt still declares carried-object
coverage as endpoint AABB only.

An Agent run then exposed a controller recovery boundary: after physics left an
open finger at the active floor margin, the strict collision QP returned no
velocity and the worker collapsed the cause into an empty `AssertionError`.
The controller now reports QP failure explicitly and may use a fallback velocity
only after a one-step preview proves all existing collision/joint boundary
violations decrease monotonically, no new hard collision appears, and joint
limits are preserved or repaired. This is controller safety recovery, not a
grasp-task stage or host-selected waypoint.

`scripts/mink_penetration_escape_canary.py` deliberately seeds that adverse
state with collision checking disabled as test setup, then reenables full
checking for the retreat. The authoritative result at
`tmp/mink-penetration-escape-canary-r7-20260819.json` used verified constraint
recovery steps, reached the 20 cm retreat target, increased EEF height by more
than 5 cm, and ended without collision. The original contact canary passed
again afterward, establishing non-regression.

The follow-up replaces endpoint-only carried-object coverage. Worker-local
Mink now checks the target object's actual MuJoCo collision geoms against every
other world collision geom at each predicted pre-actuation configuration and
each realized post-step configuration. Prediction translates the target free
body by the candidate EEF delta. The receipt reports object/world geom counts,
predicted/actual step coverage, minimum signed distance, and any monotonic
boundary-egress use.

Object bounds used for the initial conservative proxy are shape-aware. Mesh
bounds come from transformed compiled vertices and analytic geom types use
their oriented extents; collidable geoms are preferred over large visual-only
meshes. This removed the false cube-sized bounds previously produced by using
`geom_rbound` as three independent axis extents.

After a compiled contact, gripper close can receive private host-resolved
contact authorization. The tentative carried-object proxy is then bound to the
same target object rather than nearest-neighbour geometry and returns
`openeta.attachment_proxy_receipt.v1`; it explicitly does not claim attachment.
The deterministic result
`tmp/mink-contact-attachment-binding-mink-v2-20260819.json` passed all 15
contact, lift, target-binding, shape, and per-step trajectory checks.

The attachment lifecycle no longer compares motion against the reset-time
`_collision_objects` catalogue. That catalogue is geometry, not a live
object-state stream, so it cannot establish co-motion or drop. A host-bound
proxy remains tentative while the latched close reports non-empty aperture;
each motion returns `openeta.attachment_proxy_receipt.v1`. It retires only when
measured aperture collapses into the empty-close range, while independent
dual-view evidence owns the attachment verdict. The follow-up result
`tmp/mink-contact-attachment-refresh-mink-v3-20260819.json` passed all 16
checks, including persistence and receipt propagation after the 11.76 cm lift.

Controller tolerance semantics are now explicit: `position_tolerance_m` is a
maximum per-axis absolute residual. The receipt includes both the value and
metric; Euclidean `position_error_m` may therefore be slightly larger while
`reached_target=true` remains internally consistent.

Gripper close has a separate contact-admissibility envelope. It does not inherit
the arm controller's aggregate `reached_target` verdict or collision stop as an
automatic veto because `gripper_control` actuates only the fingers. A current
compiled-contact receipt authorizes close when maximum per-axis position error
is at most 5 mm and any reported orientation error is at most 0.30 rad. Motion
collision diagnostics remain visible to the Agent for recovery but do not block
the finger-only command. Close binds to the latest physically executed contact
branch; a later compiled planning candidate for the same target does not replace
that branch, while a cross-target change or epoch mismatch still fails closed.

Full repository regression after the attachment-refresh and semantic-feedback
patches: `1225 passed, 12 skipped`.

## Phase 2.1 joint-boundary and carried-object transport closure — 2026-08-19

A natural Agent run reached a wrist-view target with Panda joint 6 only 0.0175
rad below its upper bound. Mink's strict configuration-limit QP then became
infeasible. The unconstrained emergency solution crossed the limit in one
control step, so the safety preview correctly rejected it; however, rejecting
the entire velocity vector stranded every later Cartesian command even though
the robot was collision-free and not yet outside a hard joint limit.

The emergency path now projects only outward joint-velocity components onto the
hard joint box before re-running the existing predicted collision, attached
object, and joint-limit checks. Projection never bypasses a geometric preview.
The receipt reports how many projected steps were used and which joints were
affected; a remaining failure reports per-joint current/predicted margins.

IK execution binding now also covers preserve-current orientation. The preview
captures the current orientation and joint solution; the host forwards that
seed only when the exact pose policy, tolerance, object epoch, and robot-motion
epoch still match. A stronger seed posture cost steers local Mink toward the
global preview's redundancy basin while Cartesian error remains the primary
task. Near-limit previews add the nearest joint/boundary and margin as a warning
rather than falsely changing geometric reachability.

Finally, an active tentative or confirmed attachment proxy selects an immutable
controller-side `attached_object_gentle` transport profile with a 0.2 rad/s
joint-velocity limit. Nominal free-space motion remains at 0.5 rad/s. This is a
physical-evidence policy, not a pick/place stage or Agent-authored safety flag.

Live evidence:

- `tmp/mink-joint-boundary-recovery-r18-fix-20260819.json`: the exact r18 wrist
  target reached in 70 steps and a seeded preserve-current 5 cm retreat reached
  in 3 steps without the former deadlock;
- `tmp/mink-contact-gentle-transport-r18-fix-20260819.json`: deterministic
  contact/close/12 cm lift passed all checks, used the 0.2 rad/s profile, lifted
  the object 11.42 cm, and retained complete attached-object trajectory
  coverage.

The worker dependency overlay must remain minimal. A directory that also
contained MuJoCo 3.11 shadowed LIBERO's compatible MuJoCo 3.3 and broke old
robosuite at `MjData.qM`; `/tmp/openeta-mink-canary-min` contains only Mink,
qpsolvers, and quadprog.

## Full-pose convergence budget — 2026-08-19

Live r22 showed two collision-free, IK-seeded full-pose goals reach 5--8 mm
position error while retaining 0.27/0.39 rad orientation error at 100 steps.
Exact replay improved to 0.197 rad at 150 steps but was unchanged at 300 steps,
including when collision checking was disabled. This is a local control/joint-
margin plateau, not merely insufficient iteration budget, so the default stays
at 100.

The worker now stops once position is within tolerance but explicit orientation
has failed to improve for 30 control steps. It returns
`local_convergence_stalled` with the best/final orientation residual, distance
from the exact IK seed, nearby joint limits, and an explicit instruction to
change orientation or waypoint rather than replay with a larger budget. The IK
preview warning threshold is 0.05 rad: r22's exact solution had only 0.0329 rad
margin even though it was outside the former 0.02 rad warning threshold.
