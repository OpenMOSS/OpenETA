# Codex Host recovery fixes and selected retests — 2026-09-12

Worktree: `OpenETA-codex-plugin`, branch
`dev/huaizezheng/codex-plugin-smoke-2026-09-08`. Pre-Host checkpoint
`bb66378f3b5eef99c117ceb215180a886abef581` remains available. Main worktree was
not modified. No push, global plugin/configuration change or shared-document
edit was performed. Additive interfaces need three-person review before main
integration.

## First stage: four tasks with default Host changes

Artifacts: `tmp/codex-four-host-default-20260912/`. Each task used one independent
Astra/high session, official initial-state index 0, seed 0, five settling steps,
2400 s episode budget, 160 native/internal limits, Mink with Cartesian tracking.
Waypoint support and clearer Host receipts were enabled; fixture patches were
off. All four initial-state hashes matched the historical evaluation; source
remained frozen through completion, and owned processes, port and copied auth
were cleaned up.

| Task | Official result | Native / internal calls | Host time | Stream errors | Integration |
|---|---|---:|---:|---:|---|
| Object 1 | Success | 13 / 19 | 209.9 s | 0 | Passed |
| Goal 5 | Failure | 81 / 155 | 2400 s | 4 | Episode and launcher timeout |
| Long 3 | Failure | 76 / 159 | 1690.3 s | 0 | Passed |
| Long 5 | Success | 54 / 94 | 1000.8 s | 0 | Passed |

Goal 5 recovered from WebSocket broken-pipe errors through HTTPS and continued
using tools, but later exhausted its episode budget; the launcher needed its
2610 s outer timeout. This result remains an integration failure. A recorded
boundary audit permitted only the two unstarted tasks to continue; Goal 5 was
not silently retried or relabeled. Its failures also include palm/world
clearance and position/rotation corridor constraints, so network interruption
alone does not explain the failed manipulation.

Long 3 placed the bowl but failed to finish closing the drawer. A separate
operator-only final-state audit confirmed that distinction; no predicate
breakdown or simulator coordinates were provided to the acting Agent. Many
subsequent retreat/rotation attempts were rejected or stalled.

Long 5 initially selected the wrong compartment, checked the official negative
result, regrasped the book and completed placement in another compartment.
It used three route calls. Object 1 used one completed route. Goal 5 submitted
no waypoint arrays, although it did issue separate clearance/rotation moves;
Long 3 completed both of its two route calls. Route availability alone therefore
does not ensure an effective task-level path.

## Repairs applied after the four-task stage completed

1. Rejected Cartesian candidates now report bounded obstacle categories and
   tracking constraints through all three feedback filters. The message can
   identify palm versus finger clearance, an obstacle outside the contact target,
   or position/rotation corridor limits. Candidate rejection is not relabeled as
   an actual current collision. Path-specific messages no longer overwrite
   validated-IK context or failed-approach instructions.
2. The local fixture patch retains its 60 mm body-local scope and existing 1 mm
   penetration bound. A new private convex-distance helper handles independently
   certifiable separation for compiled meshes/boxes when the native query has
   an empty witness or reports deep penetration without an actual contact
   record. Closest-point QP output is accepted only with convex-membership and
   separating-plane certificates. Actual contacts take precedence; unsupported,
   intersecting or uncertified cases retain conservative behavior. No simulator
   collision flags or safety thresholds changed, and Mink's QP collision limit
   was not replaced.
3. General skill guidance covers clearance and intermediate-orientation choices,
   checking object motion after short pushes, fresh geometry for recovery grasps,
   and reconsidering compartment identity before forcing a placement deeper.
   There are no benchmark coordinates, hidden goal-region hints, scripted
   trajectories, new manipulation tools or external perception models.

Plugin snapshot: `0.1.0+codex.20260912134949`, installed by the existing launcher
into each trial's private Codex home. Source changes are recorded in
`tmp/codex-host-recovery-staging-20260912/validation-summary.json`.

The native-query diagnosis corrects an earlier interpretation of two saved
states as centimetre-scale arm/cabinet penetration. Those states had no matching
contact manifold; independent separation certificates disproved that reading.
An alternative legacy-query prototype also produced unreliable witness points
and was discarded. The production fix does not switch physics collision engines.

## Validation

Actual production source passed **167 tests, zero skips**: 125 Host/unit tests,
41 simulator tests, and one real MCP stdio/disconnect cleanup test. Plugin and
skill validators and `git diff --check` passed. Tests cover forged solver points,
overlap/touch rejection, tiny certified gaps, mesh-cache invalidation, actual
contact precedence, unchanged physics arrays, and public feedback redaction.

Thirteen paired controller regression replays retained identical motion traces
and all nine prior successful local moves. Three fixture replays give the
following narrower evidence; they are not Agent task-performance samples:

| Saved-state prefix | Checkpoint | Repaired local patch |
|---|---|---|
| 1789135548384649191 | Collision report, 29 steps, 4.67 mm error | Local convergence stall, 103 steps, 3.86 mm error |
| 1789135613591706014 | Stall, 45 steps, 3.20 mm error | Reached, 20 steps, 1.33 mm error |
| 1789135922912644628 | Reached, 12 steps, 0.97 mm error | Reached, 44 steps, 1.63 mm error |

The third replay had regressed to an uncertainty stop before this repair; that
regression is resolved. The first still does not reach its target. Saved-state
replay is useful for paired diagnosis but does not establish exact reproduction
of every original live trajectory or aggregate safety/success.

## Targeted repair retest

Artifacts: `tmp/codex-host-recovery-retest-20260912/`. Goal 5 and Long 3 each get
one new independent session with the same model, effort, official initial state
and budgets. The server explicitly enables `--fixture-contact-patch`. Object 1
and Long 5 are not rerun in this stage. Runtime snapshot SHA-256:
`d225d0b5c3e3ade191542656755eb71e0f75168397a69d795e416c2c2069c05b`.

Goal 5 reached the 2400 s episode timeout without official success, using 55
native and 100 internal calls. Four stream reconnect errors preceded HTTPS
fallback. The gap between calls 47 and 48 was 540.45 s; that observed idle gap
includes transport/retry and response time and is not a pure network latency
measurement. Later response gaps were also substantial. The controller still
encountered local path constraints: its final approach executed nine steps before
stopping at the position corridor/step limit, with about 30.4 mm endpoint error.
Thus this attempt is affected by both transport delay and incomplete manipulation.

Both attempts are complete:

| Task | Official result | Native / internal calls | Host time | Stream errors | Integration |
|---|---|---:|---:|---:|---|
| Goal 5 | Failure | 55 / 100 | 2400.2 s | 4 | Launcher timeout, 2610.4 s |
| Long 3 | Failure | 40 / 72 | 2384.4 s | 5 | Passed; Agent finished |

Both retained the frozen source and matching official initial states. No owned
processes remained, the dedicated port was released, and copied authentication
was removed. Goal 5's integration failure was retained; an explicit operator
boundary audit resumed only unstarted Long 3. No full-task retries were hidden.

Long 3's final-state audit confirms bowl-in-drawer true and drawer-closed false.
Three candidate lowering endpoints were rejected before execution for
attached-object/world collision; the Agent released the bowl from clearance.
Several subsequent retreat/rotation segments stalled or hit tracking corridors,
then an alternative retreat and reorientation succeeded. The task used one
completed waypoint route.

Four executed handle-contact moves reported `local_articulated_patch`: two
reached their target and the last two stalled. None reported local-scope,
uncertain-witness or actual collision rejection. A separate 140 mm push request
was rejected by the existing 100 mm mark-to-target bound before actuation; the
Agent shortened it to 90 mm. The last two executed moves stalled after 38 and
26 steps, with 60.7 and 81.9 mm position error, respectively. Both passed endpoint
IK, retained about 1 rad joint-limit margin, and ended with orientation error
below 0.064 degrees. Their end receipts reported no finger contact. These data
do not establish whether QP clearance constraints, contact mechanics, loss of
effective pushing contact or another controller issue caused the stalled
progress. They do rule out treating these two failures as another local-patch
authorization rejection.

Long 3 also had five request-timeout/reconnect events and fell back to HTTPS.
The fallback-spanning inter-call gap was 318.59 s. Both tasks showed much longer
post-fallback response gaps, but the traces do not separate model reasoning from
transport latency. Network probes and full error events remain in each trial's
`network/` and `run/` directories. The final private predicate audit is
`operator-goal-state-audit.json`; it was never injected into either Agent session.

## Remaining work and rollout limit

The planned three stages are complete, but these repairs have not solved the
two remaining tasks. Local query correctness and feedback improved; selected
replays and actual contact calls validate those narrower claims. Fixture patches
remain explicitly opt-in. The next useful controller diagnosis should preserve
the stalled states and record proposed/accepted joint velocity, active clearance
constraints, actual progress and contact evidence through the stalled window.
This can distinguish a constrained QP from unsuccessful pushing contact before
changing tracking limits or contact scope. The network/response-delay issue also
needs resolution before interpreting another fixed-wall-clock task comparison.

No additional full task campaign was launched after these two attempts. Code is
reviewable in the isolated branch; the pre-Host checkpoint remains unchanged.

Follow-up diagnosis on 2026-09-13 identifies palm contact with the middle-drawer
handle as the dominant QP restriction in the last two Long 3 moves. See
[instrumented saved-state diagnosis](codex-long3-stall-2026-09-13.md).

These are selected diagnostic retests across source versions. They do not change
the historical frozen pass@1 34/40 or pass@2 36/40, and cannot establish a new
40-task success rate. A new comparable estimate requires a separately frozen
campaign.
