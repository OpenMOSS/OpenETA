# Native observation references and OSC convergence — 2026-09-09

## Outcome and scope

The native Codex observation-reference feedback is repaired and validated with a
real simulator/SAM3 round trip. The IK-approved Astra approach is independently
reproducible as OSC nonconvergence. No additional model episode was run, no
AnyPlace tool was added, and no production controller/physics setting changed.
All implementation work is in the isolated `OpenETA-codex-plugin` checkout on
`dev/huaizezheng/codex-plugin-smoke-2026-09-08`.

This follows [the Astra automatic-IK retest](codex-plugin-astra-hook-retest-2026-09-09.md).
The original checkout was not edited. Its 53 inherited dirty/untracked files in
this clone remain at their recorded snapshot hashes. No commit, push or shared
document update was performed.

## Three causes of missing native reference feedback

1. The internal rejected command contained `metadata.repair_bundle`, including
   usable short packet IDs, but the native response exposed only the slim error
   and projected planner context. The useful repair index was lost.
2. The bridge forwarded the planner's images but dropped their text labels.
   This removed the adjacent camera/history labels from the native wire format.
3. The ordinary runtime indexes observations at the beginning of `act()`. After
   a runner step, the new image was already available to the native response,
   but was not indexed until the *next* decision. Adding labels alone therefore
   still could not bind those newly returned images to a registered packet.

`tools/codex_evidence.py` now exposes a bounded public reference index and repair
fields. `allowed_next_calls` is filtered to exposed tools and valid native
schemas; the complete private repair payload is not forwarded. Each image has
an adjacent JSON label derived from its exact registered RGB path, with
`source_packet_id`, `camera_frame_id` and `is_current_observation` when resolvable.
Derived images without a raw RGB reference are explicitly not point sources.
Local transport paths are not included in these labels/indexes.

`CodexHost.result()` registers each newly returned observation through the normal
memory API before projecting its response. Repeated status calls on the same
observation object do not add records. The core still registers the observation
at its next `act()`; this can produce another alias for the same immutable
image. Previously published aliases remain valid, and no extra simulation step,
Host turn or tool call is introduced. This is an ingress-only change; the shared
runtime and packet index implementation remain untouched.

## Validation

- 39 tests passed across `test_codex_host.py`, `test_codex_motion_hook.py` and
  `test_codex_plugin_launcher.py`, including the real MCP stdio integration.
- Tests cover actionable reference repair, private-field/hint filtering,
  current versus historical/derived image labels, publication before the next
  decision, repeat-status stability, and continued validity of a published ID
  after the core registers the same image again.
- `tmp/codex-reference-live-03/report.json`: `passed=true`. A fresh IK-hook move
  raised the EEF by 25 mm with a 5 mm axis tolerance (18 steps, target reached).
  A deliberately invalid SAM3 source ID produced structured repair feedback.
  The next SAM3 request using the returned current ID succeeded. Returned
  agentview and wrist image labels both identified `obs-0006` as current.
  This checks transport/recovery, not segmentation quality or manipulation success.
- Earlier `codex-reference-live` and `-02` runs recovered the text SAM3 request
  but failed the image-label assertion. Those failures exposed the publication
  timing bug; only `-03` validates the final implementation.

## Deterministic controller replay

The recorded candidate-12 target from Astra02 was replayed from the same LIBERO
spatial-task-0 initial state, seed 0, 512-pixel images and OSC profile. The replay
used the same full-pose target, 2 mm maximum-axis position tolerance and 0.05 rad
orientation tolerance. It issued 150 steps, then continued the same target for
450 additional steps. Each tool call retained the 120-second transport limit.

| Configuration | Total steps | Position norm | Maximum-axis error | Orientation error | Stop |
| --- | ---: | ---: | ---: | ---: | --- |
| Original physics | 150 | 25.126 mm | 21.363 mm | 0.089525 rad | iteration limit |
| Original physics | 600 | 4.161 mm | 3.445 mm | 0.016945 rad | iteration limit |
| Arm frictionloss set to zero, diagnostic only | 150 | 7.338 mm | 6.291 mm | 0.026589 rad | iteration limit |
| Arm frictionloss set to zero, diagnostic only | 600 | 4.359 mm | 3.669 mm | 0.017524 rad | iteration limit |

The original-physics 150-step result matches Astra02's 25.126 mm residual to
numerical precision. Its move took 20.007 seconds; the 450-step continuation took
55.342 seconds. Neither exhausted the simulator call's wall-clock limit.
The original-physics final 60 samples range only from 4.160351 to 4.160641 mm;
maximum final arm speed is 0.0000242 rad/s. This is a stable residual at the
measured timescale, so simply increasing the iteration ceiling has not attained
the existing precision requirement.

Independent FK of the preview's joint solution reproduces the target with
0.232 mm position error, 0.0321 rad orientation error and all arm joints within
limits (minimum margin 0.397 rad). Thus this particular failed execution has a
verified reachable endpoint. It does not show that every other failed approach
is reachable, nor that a collision-free path exists.

Per-step private MuJoCo telemetry recorded no robot/gripper contacts in the
original replay. An early elbow excursion briefly crossed its upper joint limit
by about 0.00115 rad; this is an observed transient, not a proven explanation for
the final residual. The production receipt still reports incomplete collision
coverage; sampled contact telemetry is not a full swept-path clearance check.

The friction ablation changed only arm-DOF `frictionloss` from 0.1 to zero after
reset in a throwaway diagnostic worker. It improved the 150-step residual, but
had not reached tolerance by 600 steps. Its final 60 errors ranged from 4.359 to
5.313 mm and final joint speed reached 0.0661 rad/s: unlike the baseline, it was
still moving. This does **not** establish friction-independent steady-state
bias or justify changing benchmark physics. It only shows that removing friction
did not solve this bounded replay.

Source inspection shows full-pose `move_to` refreshes its outer-loop error every
three steps and applies interpolated position/rotation deltas. OSC does not use
the preview's joint solution as its execution seed; the separate Mink executor
has an explicit seed path. Endpoint IK therefore cannot guarantee which joint
branch OSC follows or its dynamic convergence. No controller fix is claimed.

## Artifacts, incomplete diagnostics and next discriminating check

- `tmp/codex-convergence-03/`: original replay script, server/worker probes,
  target/source, report, private per-step telemetry and `comparison.json`.
- `tmp/codex-convergence-friction/`: clearly marked physics-ablation counterpart.
- The first two replay setups stopped before target motion because the diagnostic
  script expected nested reset data; the endpoint returns robot data at the root.
  These were script response-shape failures, not simulator reset failures.
- `tmp/codex-convergence-controller/` reproduced the motion again but failed to
  serialize NumPy arrays in the extra internal-state telemetry. It does not
  provide valid internal-controller evidence.
- `tmp/codex-convergence-controller-02/` fixes that serializer, but the execution
  request and its one retry both timed out in automatic permission review. The
  script was **not launched**. The next useful measurement is the internal OSC
  setpoint versus reported EEF pose, Jacobian conditioning and force balance at
  the plateau, before selecting a controller change.
- `tmp/codex-reference-live-03/`: final live native feedback/recovery validation,
  exact image labels, test log and simulator/Host logs.

All launched dedicated servers were closed and their reports confirm port 18778
was released. These diagnostics made no model calls and exposed no private
controller/contact telemetry to a tested Agent. They are not new task-success
samples. Skill-file reading through the restricted Codex profile remains a
separate previously observed limitation; adding guidance to the plugin skill
here is not proof that the model can read that file.
