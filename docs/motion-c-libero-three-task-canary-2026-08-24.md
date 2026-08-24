# Motion-C LIBERO three-task canary — 2026-08-24

Status: complete. Condition C is not yet an end-to-end stable solution.

## Question

Can the reviewed motion condition C stably complete three full LIBERO object
pick-and-place tasks when the Agent chooses its own recovery actions and
waypoints?

Condition C means condition B's lower-velocity stable-arrival behavior plus
just-in-time preview and a route-local IK seed chain for an exact multi-waypoint
`follow_eef_trajectory` bundle. It is not a global path planner, and the Agent
still owns waypoint count and geometry.

## Setup

- Run: `motion-c-libero-three-task-luna-20260824-r1`
- Plan: `motion-c-libero-three-task-canary-v1`
- Plan SHA-256: `b0984af103b730aa89c15c5b172416073334b816147d3b5b154907331f8fb4f5`
- Repository revision at launch: `82eed2c1802de36f831cfc4586cbe6fc25c922a5` (dirty worktree)
- Model: `gpt-5.6-luna`, primary provider only; no retry or failover was needed
- Controller: local LIBERO `mink_joint_velocity`, motion condition C
- Grasp backend: AnyGrasp, deployment width 0.08 m; preflight matched the
  0.08 m Panda calibration width
- Memory Bank: healthy, namespace `libero`, 40 objects
- Execution: serial, one environment at a time
- Budget: 120 base turns, optional 16-turn recovery extension, 240 tool calls,
  10,800 s per episode
- Visual memory: current + initial + bounded recent turns with wrist and VDM

Tasks:

1. LIBERO Object task 2, seed 2: pick salad dressing and place it in the basket.
2. LIBERO Object task 4, seed 4: pick ketchup and place it in the basket.
3. LIBERO Object task 7, seed 7: pick milk and place it in the basket.

Alphabet soup was intentionally excluded because the target is too small in
agentview for this harness-level motion canary.

## Result

| Task | Turns | Duration | Furthest verified stage | Final result |
| --- | ---: | ---: | --- | --- |
| Salad dressing | 120 | 40.2 min | Several clearance/contact approaches; no executed close | Fail, truncated |
| Ketchup | 136 | 59.4 min | Contact reached and close/lift probed twice; attachment disproven | Fail, truncated |
| Milk | 136 | 55.0 min | Contact, close, and lift probes; attachment not proven and object remained/dropped to floor | Fail, truncated |

Aggregate evaluator result: 0/3 objective successes, three
`agent_failure / episode_failed` outcomes, 9,276.78 s wall time, and no human
assistance.
AnyPlace succeeded once in every task, but no task reached a valid placement or
release, so this run does not evaluate placement-path or release reliability.

## Motion evidence

The Agent issued 61 `move_to` calls and zero `follow_eef_trajectory` calls.

| Motion result | Salad dressing | Ketchup | Milk | Total |
| --- | ---: | ---: | ---: | ---: |
| Target reached | 5 | 15 | 10 | 30 |
| `control_step_failed` | 8 | 2 | 7 | 17 |
| `local_convergence_stalled` | 2 | 0 | 4 | 6 |
| Collision stopped | 4 | 3 | 1 | 8 |

This distinction matters: the full-task run selected one waypoint at a time,
so it did not exercise condition C's most distinctive exact-route behavior:
per-segment JIT preview from the actual previous endpoint and the route-local IK
seed chain used by `follow_eef_trajectory`. The run did exercise C/B stable
arrival and worker-local Mink safety for individual motions.

Collision receipts were useful and safe:

- Salad dressing: three gripper/target collisions and one link-6/ketchup
  collision were rejected.
- Ketchup: one link-5/basket collision, one link-7/BBQ-sauce collision, and one
  gripper/ketchup collision were rejected. The BBQ-sauce prediction reported
  approximately -8.8 mm signed distance before actuation, so the worker stopped
  the motion rather than knocking over the object.
- Milk: the carried milk/floor collision was detected during the attempted
  raised transport. The attachment proxy was retired and the Agent recovered
  instead of continuing a false transport.

The Agent also demonstrated useful un-scripted behavior: it selected a raised
left-side waypoint after the basket collision, used short vertical and lateral
escape points, switched to wrist alignment, and combined MolmoPoint with
point-prompted SAM3. These behaviors show that the harness exposes enough
evidence for recovery decisions. They did not yield reliable task completion.

## Grasp and attachment evidence

Salad dressing never produced an executed `gripper_control`: three planner
close attempts were rejected before tool execution because contact evidence was
not valid.

Milk produced one close with measured open fraction 0.496, but
`attachment_proven=false`. Short lift probes reached their motion targets, yet
fresh images did not prove source vacancy/co-motion. The subsequent raised
transport collided with the floor and the milk was observed on the floor, so
the Agent reopened and retried.

Ketchup produced two close attempts:

- first measured open fraction 0.032 and was visually an empty close;
- second measured open fraction 0.418 and looked tentatively obstructed, but a
  successful 2.9 cm lift probe showed the bottle still on the floor.

One ketchup contact pose was reached with about 0.1 mm positional residual, yet
the wrist image still placed the bottle outside the jaws. This cleanly
separates controller endpoint accuracy from grasp/contact-pose quality: a
precisely reached bad pose is still a failed grasp.

## Harness behavior and cost

The run used 449 planner decision calls and 474 provider requests in total. The
449 recorded planner calls consumed 25,969,445 prompt tokens and 144,078 output
tokens (26,113,523 total); auxiliary provider calls are not included in that
token sum. Mean planner prompt size was 56k–61k tokens per task and the maximum
was 82,325, so the bounded-context refactor avoided the earlier 200k–300k
per-request growth. Aggregate cost remains very high because recovery chains
were long.

The most frequent tools were 81 SAM3 calls, 51 selection calls, 30 grasp-pose
estimates, 31 grasp compilations, and 89 IK previews. A repeated read-only
`sam3 -> select -> sam3 -> select` freshness loop consumed four pairs in the
milk recovery. New observation packet ids were treated as making just-selected
evidence old even though no object-scene or robot-motion mutation occurred.

Three `python_exec` artifact-inspection attempts failed with
`outside_sandbox_requires_approval`. The requested operations were local
artifact reads and should have used the normal sandbox; the failure was clearly
returned to the Agent, but the tool guidance did not lead it to the usable
mode.

## Diagnosis

The evidence does **not** support the conclusion that waypoints are useless.
It supports a narrower conclusion: condition C makes selected free-space
segments more observable and often reachable, and its collision checks prevent
unsafe motion, but the present Agent/tool chain neither invokes C's multi-point
route primitive nor reliably reaches and grasps the object.

Priority bottlenecks are:

1. **C route under-use.** The Agent expresses every detour as independent
   `move_to` calls, so the route-local JIT seed chain is never activated.
2. **IK/execution inconsistency.** Many endpoints pass `ik_preview_check` and
   then fail with `control_step_failed` or `local_convergence_stalled` under the
   actual Mink executor.
3. **Clearance corridor quality.** Some compiled clearance paths intersect the
   target, basket, or another object even when the endpoint itself is reachable.
   Endpoint-only preview is not a global or full-body path proof.
4. **Grasp/contact quality.** Reaching the compiled contact pose can still leave
   the target outside the jaws. Wrist alignment and MolmoPoint help perception
   but do not yet consistently correct the final contact anchor.
5. **Attachment proof and carried geometry.** Aperture obstruction alone is not
   attachment proof. The current visual probe correctly rejects false grasps,
   but the carried-object proxy/trajectory chain is not robust enough to reach
   transport reliably.
6. **Read-only freshness churn.** Evidence should remain usable across packet
   refreshes when both object-scene and robot-motion epochs are unchanged.
7. **Artifact-query ergonomics.** Read-only local artifact inspection should
   naturally select the sandboxed `python_exec` path.

## Recommended next experiment boundary

Do not rerun another three full 120-turn suite yet. First use short targeted
canaries to establish:

1. the Agent actually calls `follow_eef_trajectory` for a self-chosen 2–4 point
   free-space route, and C returns every JIT segment receipt;
2. a preview-passed single segment has a high Mink execution success rate from
   the same actual state, with failures retaining solver/constraint evidence;
3. compiled clearance is checked as a swept full-robot corridor, while
   intentional target contact is authorized only for the contact segment;
4. a wrist-refined grasp can repeatedly pass close + lift + source-vacancy
   attachment proof on one easy object; and
5. read-only SAM3/select evidence survives packet refreshes without a motion or
   object-scene epoch change.

Only after these pass should the same three-task suite be repeated. A stronger
planner model is unlikely to repair the observed controller, corridor, and
contact-anchor inconsistencies by itself; this run already showed appropriate
recovery choices from the available evidence.

## Artifacts

- Evaluation report: `.openeta_eval/runs/motion-c-libero-three-task-luna-20260824-r1/report.json`
- Compiled plan: `.openeta_eval/runs/motion-c-libero-three-task-luna-20260824-r1/compiled_plan.json`
- Per-job final records and complete rollout artifacts are under
  `.openeta_eval/runs/motion-c-libero-three-task-luna-20260824-r1/jobs/`.
