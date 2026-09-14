# Bounded recovery: two complete task retests — 2026-09-13

**Goal 5 passed; Long 3 did not. Both integrations completed normally.**
These are one fresh diagnostic session per selected historical failure, not a
new benchmark pass@2 estimate. Historical evaluation results remain unchanged.

Implementation and local validation:
[bounded recovery implementation](codex-recovery-implementation-2026-09-13.md).
Previous full-task comparison:
[intervention-feedback retest](codex-feedback-retest-2026-09-13.md).
Artifacts: `tmp/codex-recovery-retest-20260913/`, including `protocol.json`,
`runtime-source.tar.gz`, `frozen-source.json`, `results.json`, and the reproducible
operator analysis `analyze.py` / `recovery-analysis.json`.

## Protocol and results

Astra/high through Codex subscription authentication; fresh private Codex home
and independent session per task. Atomic tools, Mink, Cartesian tracking and
local fixture contact patches enabled. Official initial-state index 0, seed 0,
five settling steps. Episode limits: 2400 seconds, 160 native requests and 160
internal stages/turns. Recovery does not increase these budgets.

Both installed plugin manifests were verified as
`0.1.0+codex.20260913114937`. Initial-state hashes matched the historical trials;
source hashes stayed frozen across both runs.

| Task | Official result | Host time | Host-admitted native / internal calls | Termination |
|---|---|---:|---:|---|
| Goal 5: push plate to front of stove | **Success** | 1531.33 s | 78 / 145 | Official success during final push |
| Long 3 (`libero_10:3`): bowl into bottom drawer and close | Failure | 1606.95 s | 76 / 160 | `codex_turn_limit`, 160/160 |

Goal 5's final motion stopped early with `episode_terminated` because the official
checker had succeeded; it need not reach the requested endpoint. A subsequent
`finish_episode` returned `episode_not_active` after automatic closure. This
redundant completion request did not reverse success or fail integration.

Both sessions issued one final `finish_episode` after automatic closure. Counting
these rejected requests, Codex completed 79 and 77 native calls respectively;
the public journals contain matching contiguous sequences. The table counts
Host-admitted requests, consistent with the experiment's `host.requests` field.

Long 3 completed its final recovery lift, then hit the internal-stage boundary.
This is a real Host budget termination, unlike the preceding retest, where the
Agent voluntarily finished at 159/160. Neither new run reached the wall timeout.

## Actual use of the changes

| Native move receipt | Goal 5 | Long 3 |
|---|---:|---:|
| `target_reached` | 34 | 26 |
| `control_step_failed` | 6 | 7 |
| `local_convergence_stalled` | 3 | 10 |
| `iteration_limit` | 0 | 1 |
| `collision_detected` | 0 | 5 |
| `ik_search_no_solution` | 3 | 0 |
| `no_authorized_orientation_candidate` | 1 | 0 |
| `episode_terminated` with official success | 1 | 0 |
| Explicit recovery requests | 8 | 15 |
| Recovery requests reaching target | **8** | **10** |

Counts summarize top-level native receipts. Long 3 used one waypoint route,
which can contain multiple internal segments; the table is not a segment count.
Goal 5 used successive individual moves.

Goal 5 repeatedly withdrew, remeasured the plate rim, corrected pushing height,
and changed approach orientation. The Agent reported moving the plate forward,
then sideways, then toward the stove. Request 78 triggered official success.
Its eight recovery moves all reached their targets, with maximum recorded
cross-track displacement 9.78 mm. None selected an alternative recovery solve;
seven used verified collision-QP fallback steps. These observations are
consistent with useful recovery and the guard fix, but they do **not** isolate
which change caused task success. In particular, this task does not establish a
benefit from the new alternative solver or the wider actual 20 mm corridor.

Long 3's recovery outcomes were ten arrivals, four local stalls and one segment
failure. Alternate solves were selected in requests 9, 11 and 72 (9, 22 and 2
control ticks respectively); the first two still stalled, while 72 reached.
Successful recovery request 64 had 15.84 mm peak cross-track displacement, inside
the explicit 20 mm limit. Recovery request 37 stopped at a candidate corridor
limit after reaching 18.00 mm actual peak displacement. Recovery remained
bounded and did not bypass collision checks.

## Remaining Long 3 failure

The Agent reported a successful test lift, released the bowl in the drawer, and
observed several inward drawer movements. At the end it reported that the bowl
remained inside but the drawer was partly open. Those intermediate scene claims
are Agent observations, not privileged predicate evidence. The official task
checker stayed false.

The records distinguish several different limitations:

- The lowering request 22 exhausted 150 controller steps, with 103 mm position
  error but only 0.86 degree orientation error. IK seed validation had passed;
  149 steps were reduced by tracking/geometry checks. This was not an endpoint
  angle-IK rejection.
- Closing requests 57, 59 and 63 failed the straight position corridor. Their
  final candidates were rejected by tracking before geometry was tested. It
  would be incorrect to label these candidate failures as observed collisions.
- Request 70 stalled with measured finger-body contact on the authorized target,
  no active QP clearance rows in the last recorded configuration, and velocity
  reduced from 0.157 to 0.00984 rad/s. That supports investigating contact-induced
  motion and step reduction, not assuming an unrelated obstacle was blocking it.
- Request 73 exposed both an active arm-clearance row and measured arm contact.
  The final push, request 75, exposed an active arm constraint outside the contact
  target but **no measured end-state external contact**. Constraint activity and
  physical collision are different observations; the telemetry does not prove
  a complete causal account of either stall.

Recovery helped the Agent leave several difficult poses. It did not supply a
stable contact geometry or a task-level route for closing the drawer. The remaining
work should therefore investigate the contact/whole-arm execution cases from
saved states rather than simply increase the budget or widen collision grants.
A useful next controlled comparison would constrain candidate direction changes
inside the existing strict corridor for requests 57/59/63, and independently
check gripper/handle geometry and joint posture for requests 70/73/75. No such
additional controller change or full-task retry was made during this retest.

## Network, cleanup and interpretation

Both Codex sessions recorded zero stream-error events. Goal 5 had 52 probe
rounds; all 260 HTTPS probes completed successfully at the transport level.
Long 3 had 54 complete HTTPS rounds (270 probes), all successful. Its final
round at 20:45:38 +08:00 contains five HTTPS subprocess exits of `-15`, after
model completion and during process-group cleanup. The raw monitor counts these
as failures; they are recorded separately as SIGTERM-interrupted probes rather
than evidence of a network outage. Raw samples were preserved.

Local direct DNS resolution for `chatgpt.com` still timed out: 52 Goal 5 samples
and 54 Long 3 samples. Long 3's last DNS sample ended during cleanup without an
address. Proxy-mediated HTTPS and the model sessions operated despite this;
these tests do not diagnose the resolver issue or prove universally healthy
network access.

Both trials: `integration_passed=true`, `initial_state_verified=true`,
`historical_initial_state_match=true`, `source_changed=[]`,
`remaining_owned_pids=[]`, `port_released=true`, `private_auth_removed=true`.
The main worktree and global configuration were untouched. Code remains in the
isolated plugin worktree with the existing uncommitted changes preserved.

The outcome is **1/2 official task successes and 2/2 healthy integrations**.
It is encouraging compared with the immediately preceding 0/2 diagnostic
retest, but one selected attempt per task cannot establish a new success rate
or attribute the improvement to a single component.
