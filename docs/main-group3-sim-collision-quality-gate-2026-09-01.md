# Main feature integration: Group 3 simulator/collision quality gate

Date: 2026-09-01

## Decision

Group 3 passes the fixed-seed LIBERO Object-10 quality gate with **4/10
(40%)** objective success. This is above both the frozen 2/10 baseline and the
manually promoted Group 2 sample of 1/10. All ten jobs produced valid terminal
outcomes; there were no provider, scheduler, simulator-startup, or preflight
failures and no retries.

The acceptance result supports merging the BEHAVIOR/R1Pro/cuRobo simulator
surface while preserving LIBERO/Mink. It does **not** establish that the
success-rate increase was caused by collision handling: the four successful
task identities are disjoint from the successes in the frozen baseline, Group
1, and Group 2 samples. Repeated seeds or paired controller variants are still
needed for a causal comparison.

## Evaluation provenance

| Field | Value |
| --- | --- |
| Run | `libero-object-10-main-group3-sim-collision-20260831-r01` |
| Plan | `libero-object-10-pass1-main-group3-sim-collision-20260831` |
| Plan digest | `57669d6f2fbeb1ecb7bfca6c42544cbd7255ac599c09d817c27e24e6f6f14086` |
| Model | `gpt-5.6-luna` |
| Benchmark | LIBERO Object, tasks 0–9, seed 0, pass@1 |
| Execution | serial, provider concurrency 1, 160-turn hard limit |
| Feature head | `f7a0a87` |
| Launch revision | `ce2669a4b28bffda6f541fa9505478d047869af0` |
| Objective authority | official trusted environment reward |
| Result root | `.openeta_eval/runs/libero-object-10-main-group3-sim-collision-20260831-r01` |

Preflight verified the local simulator contract, the Object Memory service,
and compatible AnyGrasp geometry. The run made 1,481 provider requests with
zero queue timeouts. Total wall time was 26,926.426 seconds (7.48 hours), and
mean episode time was 2,692.625 seconds.

The launch revision was reported dirty only because two pre-existing,
user-owned deletions (`.env.example` and `.mcp.example.json`) were present.
Those files were not part of Group 3 and are intentionally excluded from the
integration commit.

## Per-task result

| Task | Object | Result | Turns | Duration | Session |
| ---: | --- | --- | ---: | ---: | --- |
| 0 | alphabet soup | fail: remote episode terminated | 130 | 56.7 min | `0a448f8a-499e-4c54-809b-260eeccb2627` |
| 1 | cream cheese | fail: turn limit | 160 | 48.2 min | `90f28d5c-1b7f-47c4-a0da-30015f02d917` |
| 2 | salad dressing | fail: turn limit | 160 | 62.6 min | `165d0551-c9b9-45b4-aaad-361c4909549d` |
| 3 | BBQ sauce | success | 75 | 28.4 min | — |
| 4 | ketchup | success | 131 | 50.8 min | — |
| 5 | tomato sauce | success | 49 | 18.7 min | — |
| 6 | butter | fail: turn limit | 160 | 47.2 min | `318faaa6-f372-44cc-a1e1-17ba51cb396c` |
| 7 | milk | fail: turn limit | 160 | 60.4 min | `c799419a-272f-474f-854c-0a56a85a4eef` |
| 8 | chocolate pudding | fail: turn limit | 160 | 47.6 min | `72fffb7e-9cd5-42af-919c-e6d9acd8e93f` |
| 9 | orange juice | success | 73 | 28.3 min | — |

## Six-failure diagnosis

### Task 0: alphabet soup — horizon exhaustion after expensive recovery

The Agent repeatedly rebuilt perception, grasp selection, and IK evidence for a
small target. Its seven `move_to` calls included one 150-iteration failure at
29.1 mm residual and one 300-iteration timeout. Two closes were attempted. The
second reported openness `0.5765` and the wrist image showed plausible contact,
but the Agent reopened without running the prepared attachment probe. The next
motion reached its requested point; the final tool call failed because the
remote episode had already terminated.

This is not primarily a collision-verifier rejection. Long controller and
gripper-settle budgets, small-target visual ambiguity, repeated candidate
reconstruction, and abandoning a plausible grasp consumed the environment
horizon.

### Task 1: cream cheese — endpoint IK did not predict executable local motion

All 21 IK previews reported reachable, yet all 15 `move_to` calls failed. The
first calls executed 62, 22, and 3 steps before stopping; later calls executed
zero steps with `constraint_escape_preview_rejected`. The EEF remained near
`[0.0854, -0.0892, 0.1653]`, while one repeated contact target was about 215 mm
away. No gripper close was attempted.

The failure exposes a contract boundary: endpoint kinematic reachability is not
current-state route feasibility or proof that the local Mink QP can escape its
joint/constraint basin. Repeating new grasp candidates did not change that
basin.

### Task 2: salad dressing — failed attachment followed by target knockdown

The first tentative close was reopened before assessment. The second close
reported openness `0.6656`; a 3 cm attachment probe ran, but visual assessment
returned `FAIL`. Subsequent recovery contained three correctly reported
collision stops. By the final observations the salad-dressing object had been
toppled near the gripper, and a 530 mm recovery target was rejected by the
constraint-escape preview.

The dominant cause is poor contact/attachment quality, followed by recovery
motion that disturbed the target and left the robot in a hard local
configuration. The collision path was active and prevented further risky
motions, but did not supply a feasible escape route.

### Task 6: butter — oversized escape waypoints and perception churn

The first two moves reached their references within 5.2 mm and 8.9 mm. Four
later waypoint requests were 445–468 mm from the current EEF and were all
rejected before execution by the constraint-escape preview. No close was ever
attempted. The remaining episode was dominated by 79 SAM3 calls and repeated
candidate searches, including all-colliding GraspGenX results.

The failure is a planner/controller recovery mismatch: the Agent proposed
global-sized escape waypoints to a local controller, then lacked an actionable
bounded recovery after rejection. Small, flat target geometry amplified the
perception loop.

### Task 7: milk — successful pick and carry, incorrect basket-interior release

After two failed close branches, the third close reported openness `0.5460`.
The 3 cm probe and attachment assessment passed, and the milk co-moved. AnyPlace
and camera-to-world conversion succeeded. The Agent then carried the milk to
the basket through checked motions. Unsafe placement descents were stopped by
collision checks, so the Agent raised the object and eventually opened the
gripper while the milk was resting on or just behind the rear basket rim. The
official reward remained zero.

This task demonstrates real Group 3 coverage: attachment and carry succeeded,
and collision receipts constrained the placement path. The remaining failure
is placement geometry and route generation—not pick quality. Avoiding a risky
descent by moving higher did not produce a reference demonstrably inside the
basket volume.

### Task 8: chocolate pudding — execution-fragile IK accepted as reachable

The first motion executed 65 steps but stopped with 12.7 mm residual. A repeat
executed zero steps, and the third executed 13 steps before rejection despite a
3.97 mm endpoint error. The corresponding IK seeds had only 0.001234 rad and
then 0.000000 rad joint margin. The Agent never closed the gripper and spent the
remaining turns in perception/selection churn.

The endpoint was mathematically reachable but execution-fragile. The current
IK receipt exposes joint margin, yet `reachable=true` alone is too weak a
planning signal for local motion near a hard joint boundary.

## Cross-task failure taxonomy

| Dominant boundary | Tasks | Count |
| --- | --- | ---: |
| Local controller/escape failure before first close | 1, 6, 8 | 3 |
| Contact failure, target disturbance, then recovery trap | 2 | 1 |
| Placement/reference failure after verified pick and carry | 7 | 1 |
| Environment horizon exhausted by recovery/perception churn | 0 | 1 |

The most common remaining defect is the gap between endpoint IK and executable
current-state motion. `ik_preview_check` is useful evidence but must not be
described as trajectory or local-convergence authorization. Likewise, a tool
call whose transport status is `executed` is not evidence that motion reached
its target; consumers must use `reached_target`, residuals, stop reason, and the
collision receipt.

## Group 3 capability coverage

- LIBERO remained functional while the BEHAVIOR/R1Pro surface was present.
- Motion receipts reported world/trajectory checking, and collision stops were
  exercised in both failed recovery and late placement paths.
- The new collision capability prevented several unsafe motions, but it is not
  a route planner and does not manufacture a safe escape waypoint.
- R1Pro joint-name mapping is additive simulator evidence. A missing
  `joint_names` field remains valid for legacy robots whose ordered joint
  vector already matches the configured model.
- cuRobo's host-side discrete configuration check and Mink's worker-local
  per-step collision scope are separate capabilities and must remain separate
  in receipts and documentation.

## Comparison and promotion rationale

| Gate | Objective result | Decision |
| --- | ---: | --- |
| Frozen seed-0 Object-10 baseline | 2/10 | reference floor |
| Group 1 | 3/10 | passed |
| Group 2 | 1/10 | manually promoted after protocol remediation |
| Group 3 | 4/10 | passed numerically |

Group 3 clears the agreed no-regression floor and has no infrastructure-invalid
jobs, so the integration branch may be pushed. Because success identities move
substantially between single-seed samples, the 4/10 result should be treated as
a promotion gate, not a statistically stable estimate of capability.

## Recommended next evidence

1. Add a route-feasibility/fragility summary distinct from endpoint IK,
   including current-state joint margin and a bounded escape recommendation.
2. Make motion rejection propose a small, executable recovery envelope rather
   than encouraging global 40–50 cm waypoint jumps.
3. For container placement, verify that the release reference lies inside an
   estimated interior volume and that a collision-checked approach corridor
   exists.
4. Penalize reopening immediately after a tentative close when a valid
   attachment probe is already prepared; this is planner guidance, not a host
   stage transition.
5. Run paired repeated seeds before attributing the numerical improvement to
   Group 3 collision changes.

## Verification before the live gate

- full sandbox suite: 1,580 passed and 15 skipped;
- four manual VLM proxy cases failed only because the sandbox denied local
  sockets;
- unrestricted rerun of `tests/test_manual_vlm_proxy.py`: 20 passed;
- effective aggregate: **1,584 passed, 15 skipped**.
