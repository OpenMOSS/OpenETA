# Codex Host routes, contact scope and feedback — 2026-09-12

Baseline checkpoint: `bb66378` on
`dev/huaizezheng/codex-plugin-smoke-2026-09-08`. This commit preserves the complete
current experimental source before these Host changes. Main worktree untouched;
no push or shared-document edits. Current changes are relative to this checkpoint.

The historical frozen evaluation remains pass@1 34/40 and pass@2 36/40. That
predates the Cartesian controller update. Neither the checkpoint nor these Host
changes have a new full-task success estimate.

## Rollout decision

Routes, explicit feedback, and cancellation handling are enabled. **Local fixture
patches remain opt-in**, because one local replay regressed. The dedicated server
requires `--fixture-contact-patch` with Mink; absent that flag it overrides ambient
settings and retains the original single-geom association/authorization behavior.
The internal switch is `OPENETA_LIBERO_FIXTURE_CONTACT_PATCH=1`. Older grants are
still accepted. Feedback reports the active contact scope without geometry IDs.
Do not enable this switch for a claimed comparable evaluation without recording it.

## Changes

- `tools/codex_route.py` composes at most five strict pose segments (four optional
  intermediate waypoints and the final target). All route arguments/references
  are validated before any actuation. IK is checked immediately before EACH
  segment using the existing normal budgeted motion hook. No simulator bypass,
  inferred IK authorization or new task-level manipulation primitive.
- No deltas, contact grants or symmetry selection inside routes. Orientation and
  position omissions inherit the preceding requested pose. Single-call behavior
  remains backward compatible. Relative manipulation and contact use separate
  native calls, and retention/scene checks can still be inserted between moves.
- Intermediate observations follow normal memory publication. Rendering happens
  once at return. Geometry-only preview draws a numbered route, not a collision
  or feasibility prediction. Failure returns completed/stopped/unexecuted indices,
  actual state and known zero/partial/unknown execution. A start/result journal
  retains partial route history independently of the final native response.
- Cancellation and stdio teardown set a thread-safe stop event before acquiring
  the execution lock; subsequent IK/motion dispatches are blocked while an
  already-running worker call finishes. Cleanup then closes the episode. This
  does not promise instantaneous preemption of a running physics call.
- The launcher aligns native MCP tool timeout with Host episode timeout plus
  30 seconds (at least the existing 180 s), avoiding a fixed 180 s transport cut
  during a legal route. Host episode, request, turn and tool budgets still apply.
- Fixture association treats adjacent collision pieces of the same body as one
  association candidate; different bodies/objects remain ambiguous. A new grant
  stores the measured anchor in that body's local frame with 60 mm radius.
  Candidate pieces are culled by conservative bounds; exact contact/witness
  checks bound scope and penetration before and after physics and during gripper
  actuation. Fixed cabinet panels, other drawers, arm and carried-object checks
  remain protected. Re-marking within a held patch preserves its original anchor;
  attempting to rebase the patch requires release.
- MuJoCo may return distance zero and an empty closest-point witness. This is not
  interpreted as contact at world origin. Conservative body-frame box/compiled
  mesh bounds can prove separation or containment; unresolved cases return
  `contact_geometry_uncertain`. No guessed contact point is supplied to the Agent.
- Public feedback preserves specific constraint classes across BOTH simulator
  redaction and motion-summary compaction. Only enums survive: joint limit,
  robot collision, carried collision, path tracking, fixture contact scope.
  Carried-object pre/post-step stages and local scope failure enums also survive.
- `contact_state` reports mark generation, close-binding and symmetric-orientation
  availability without claiming attachment. Stale and depth-edge marks get
  separate error codes. The skill describes proactive clearance/turning choices,
  while leaving short direct moves and Agent-selected routes available.

Shared/additive interfaces: **needs three-person review before main integration**.
No benchmark/task-specific trajectory, hidden goal region, SAM3, AnyGrasp or
AnyPlace was added. The local plugin cachebuster was updated with the plugin
helper; the existing per-run launcher installs the updated snapshot into each
new private Codex home.

## Tests

- 122 Host, atomic route, motion hook, feedback, fixture association and launcher
  tests passed under `.venv/bin/python` (excluding the separate stdio test).
- 1 real MCP stdio connection/disconnect cleanup test passed in the normal local
  process environment. The sandboxed subprocess variant hung and its specific
  process group was terminated; the normal run uses a 90 s outer timeout.
- 28 simulator tests passed using `sim/venvs/libero/bin/python` plus
  `tmp/codex-mink-deps`: fixture patch, gripper guard, Cartesian segment,
  joint-limit and opposing-finger checks. No skips in this run.
- Total: **151 tests passed**. Plugin manifest and atomic skill validation passed.
  The pre-change checkpoint had separately passed its 108 targeted tests.

Route regressions cover invalid later poses rejected before the first move,
per-segment fresh IK/exact orientation/budget accounting, actual-state receipts,
geometry-only preview, missing arrival, zero/partial failures, budget expiry
between IK and motion, transport stop between segments and incremental journaling.
Fixture negatives cover far surfaces on the same body, fixed structures, other
bodies, excessive penetration, invalid grants, moving body anchors and degenerate
closest-point witnesses. Feedback tests pass data through both real redaction and
compaction layers and assert that private names/coordinates are absent.

## Three paired local fixture replays

Artifacts: `tmp/codex-host-20260912/`. `replay.py` restores trusted operator
snapshots from Long 3 trial 1; it runs no model and creates no performance samples.
Both sides use Cartesian tracking. Baseline controller source is extracted from
`bb66378`; only candidate authorization is upgraded from the old measured-point
receipt into the new local body-frame patch at the restored starting state.
These snapshots are not persisted-held-grant lifecycle tests; that lifecycle has
separate unit tests. Earlier development outputs are retained under iteration
subdirectories. All replay environments were closed.

| Snapshot prefix | Checkpoint | New Host scope | Interpretation |
|---|---|---|---|
| 1789135548384649191 | collision after 29 steps; 4.67 mm endpoint error | collision after 84 steps; 3.56 mm error | Native distance checks report arm/cabinet collision; the follow-up below disproves the reported deep penetration. |
| 1789135613591706014 | stalled after 45 steps; 3.20 mm error | reached after 20 steps; 1.33 mm error | Removing the neighboring-piece obstacle improves this local move. |
| 1789135922912644628 | reached after 12 steps; 0.97 mm error | stopped after 20 steps; 18.26 mm error | New guard cannot prove a degenerate finger/fixture witness is within the patch. This is a regression in local completion and remains unresolved. |

**Correction from the subsequent saved-state investigation:** the first case's
native distance queries report about -22.84 mm and -19.97 mm after their last
steps, but these values are not reliable evidence of physical penetration.
Those exact configurations have no corresponding contact manifold. Independent
convex membership and separating-plane certificates instead establish positive
separation, with computed witness distances of approximately 0.233 and 0.084
micrometres (certificate agreement tolerance: 0.1 micrometre). The earlier interpretation as actual centimetre-scale penetration
was incorrect. Evidence is retained in
`tmp/codex-contact-witness-20260912/arm-collision-probes.json` and
`sat-arm-probes.json`; the subsequent certified-query replays are separate from
the unchanged results in the table above. Actual contact records and the -1 mm
guard remain relevant; neither a native query alone nor these selected replays
establish aggregate safety or task success.

The third case is explicitly reported as geometry uncertainty, not proof of a
current collision or endpoint IK failure. The conservative scope must not be
relaxed solely to reproduce a successful endpoint. The opt-in rollout decision prevents this regression from changing the default contact policy. A follow-up should examine
reliable contact witnesses and short approach alternatives before a full-task
comparison. Baseline `bb66378` remains available for comparison or rollback.

At the point recorded above, no full LIBERO task had been rerun, and neither original frozen results nor official
success predicates were modified. A new campaign must freeze the new source and
use a new output directory before reporting pass@1/pass@2.

Follow-up: [recovery fixes and retests](codex-host-recovery-2026-09-12.md)
records the subsequent four-task trial, certified-query repair, and targeted
retests. The table above is the original pre-repair replay result.
