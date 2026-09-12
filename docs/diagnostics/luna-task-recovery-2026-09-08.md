# Luna real-task recovery with existing object memory

User authorized local failure-output regressions, then a few fresh standard Luna
batch episodes: correct localization / selection / grasp input first, followed
by approach / grasp / lift, complete Object 0 and only then Spatial 0 / Long 9.
Reuse `retrieve_asset_reference`; do not inject simulator oracle labels.

## Baseline and boundaries

Current repository was clean on `experimental` at `723bed1` (snapshot of prior
work). New personal branch: `dev/huaizezheng/luna-task-recovery-2026-09-08`.
No old worktree restoration, commit or push. OpenETA collaboration skill used;
shared RFC access was denied in the prior discussion and is not retried here.
Interface/output changes remain pending collaborator review; no approval hashes
or default profile changed. Goal mechanism remains paused; current direct user
authorization is used, not a replacement goal.

## Local changes

- Repair feedback repeats the rejected tool's actual projected schema close to
  the rejected request, retains valid choices, and explains optional omission.
  Full tool availability and all validation/authorization gates remain unchanged.
- Object-memory handler preserves successfully materialized catalog images if
  the optional visual localizer fails. The operation remains failed; no pixel
  seed, identity confirmation or motion authorization is manufactured.
- Main-Agent review includes these catalog images, clearly marked as appearance
  references rather than live observations. Successful references remain visible
  through downstream selection; current-scene images retain priority under caps.
- Experimental profile explicitly advertises the existing lookup for unfamiliar
  names / conflicting visual interpretations. Agent chooses whether to query and
  what to select; no mandatory task order or object-specific answer is added.
- Local fixtures reproduce wrong selection field, bad geometry enum, null identity
  fields, and preserve the existing malformed-XML repair tests. No historical
  provider trace is sent externally.

Focused tests: 18 passed. Full suite: 2419 passed, 25 skipped, one pre-existing
failure in `test_reviewed_tool_contract_migration_is_complete_and_narrowly_authoritative`.
No approval hash or reviewed-authority gate was bypassed. Results:
`tmp/luna-task-recovery-tests.xml` and `tmp/luna-task-recovery-tests.log`.
`git diff --check` passed.

## Preflight and experiment

Configured object bank `http://10.11.18.197:8080`: health ok, 40 objects.
Named lookup `alphabet soup` resolved `libero/alphabet_soup`, returning three
reference images (671490 bytes total). Name-only query, no scene upload to bank.
References are catalog assets, not evidence of which scene instance matches.

Run root: `tmp/luna-task-recovery-im3f7H/`. One fresh standard batch Object 0 /
seed 0, original task text, `bundle_stage3`, main/fallback Luna only.
15M known cumulative tokens / 160 turns / 320 tools / 10800 s. Use milestones
only for diagnostic assessment; only official reward/termination determines task
success. No paid Spatial/Long expansion before earlier milestones pass.

Session: `cb9cb3af-5bed-4236-be70-1eb8d78ad1ea`. Dedicated simulator port 18766,
Mink joint-velocity controller; server PID 2189960, batch PID 2192084 at launch.
Stopped at failed semantic milestone: Agent ignored optional reference guidance,
segmented and selected the foreground tomato-sauce can, then authored a valid
grasp-input bundle call. Five recorded main requests / 138355 known main tokens,
one malformed XML (missing closing reasoning tag) successfully repaired by the
next request. SAM3 selection and grasp-input parameters passed strict validation.
Only three tools completed (SAM3 twice, selection once), no motion. Cancellation
may leave an in-flight provider request unaccounted; these are known recorded
tokens, not a provider billing total. `episode_interrupt.cleanup` reports closed,
ok, no cleanup errors; batch exited 130, server exited 0, port 18766 verified free.
The existing interrupted-batch finalization issue remains; no official result
file/task success is inferred from this diagnostic interruption.

## Fresh guidance follow-up

Strengthen general name-to-appearance guidance for named catalog objects and
similar packages. Explain that successful point segmentation establishes a
region, not a semantic identity; empty SAM3 results now suggest the existing
reference tool as another Agent-selected recovery option. No target-specific
coordinates, colors or simulator oracle labels added; no mandatory tool order
or automatic movement. Follow-up focused tests: 74 passed.

Second fresh run: `tmp/luna-reference-guidance-KCnUai/`, same standard entry,
task text, seed, budgets and Luna-only configuration.

Session `f3858565-7ac7-499c-8909-8aee1313294d`: the first action queried the
reference bank correctly. The isolated localizer rejected the foreground can,
but then classified a different small package as a match at (218,240). Its
"verified_match" is a model claim, not ground truth. Agent copied those agentview
obs-0000 pixels into SAM3 wrist / obs-0001; the existing source resolver accepted
that misuse. Stop before motion: three recorded main requests / 98257 known main
tokens, zero schema rejects, only retrieval and SAM3 completed. The third authored
selection also confused a catalog key with an identity-anchor ID; it was not
dispatched before cancellation. Localizer accounting retains only selected-call
usage, so main tokens are not the total experiment cost. No provider overload is
established by the available evidence.

Cleanup: episode interrupt closed/ok/no errors, client exited 130 (background
request unwind delayed process exit), server exited 0, port verified free.

## Local repairs after cross-view failure (pending shared contract review)

- Reused pending reference points/ROI now require their original packet and
  camera before SAM3 dispatch. Independent newly authored points remain allowed;
  this is a known-seed provenance guard, not proof of arbitrary point provenance.
- Existing `retrieve_asset_reference` accepts optional `localize` (boolean,
  default true). `false` returns `reference_images_available`, with
  `localization_status=not_requested` and `identity_confirmed=false`, no pixel seed
  or localization obligation, and no isolated localizer call. Default behavior
  is preserved. Example: environment=`openeta/libero_libero_object_task0-v0`,
  target_object=`alphabet soup`, source_packet_id=`obs-0000`,
  camera_frame_id=`agentview`, localize=false.
- The latest catalog's three appearance views remain visible during subsequent
  selection, including successful pure retrieval and failed-localizer cases.
  They never become world observations or identity authority. The experimental
  profile recommends direct lookup for Agent-owned comparison.
- Public catalog regenerated; approval hashes untouched. Request/outcome and
  expanded source-gate meaning require collaborator review. Added local runtime
  fixture coverage for both lookup and legacy localization outcomes.

Initial restricted full-suite run could not bind local HTTP test ports; rerun
with loopback permission and all remote integration flags disabled had 2425 pass,
25 skip, the pre-existing authority failure and a newly uncovered fixture-coverage
gap. The latter was fixed by exercising the new successful outcome explicitly;
final full rerun: **2426 passed / 25 skipped / one existing authority failure**,
`tmp/luna-reference-final-tests.xml` and `.log`; `git diff --check` passed.

Third fresh standard run: `tmp/luna-reference-only-UPGDVB/`, session
`d43fcb8f-9a33-4ef2-8ca1-5e2e9d827c50`. Same task, seed, budgets, controller and
Luna-only models. Agent explicitly requested `localize=false`; retrieval succeeded,
and the next provider request attached both current scenes plus all three catalog
views. Text SAM3 returned no detections.

Third run stopped before motion after another semantic/view mismatch: Agent
claimed a foreground-center blue/orange can and submitted newly authored points
on wrist / obs-0002 despite referring to agentview in its reasoning. Unlike the
second run, these were not copied localizer pixels, so the known-seed provenance
guard does not infer they are invalid. Six recorded main requests / 186680 known
main tokens, one XML syntax failure repaired next attempt, four tools completed
(lookup, SAM3 twice, selection). The next grasp-input request was authored but
not completed. Cleanup closed/ok/no errors; client exited 130 and server exited 0.

### Confirmed context-projection omission

The real `semantic_request.tool_context` contains
`open_questions.target_selection.selection_bundle`, including source/candidate
contact-sheet paths. Backend image assembly still queried the obsolete top-level
`pending_target_selection`. Actual requests therefore attached current scenes,
catalog images and history but **no SAM3 selection review images**. This explains
missing visual evidence during selection; it does not establish the cause of
every initial semantic error or prove that corrected images suffice for success.

Backend now reads canonical `open_questions` for target selection and reference
localization, keeping old caller shapes compatible. A pending localization no
longer suppresses a simultaneous selection review. With catalog evidence, image
ordering preserves current scenes, selection overview/source and catalog views
before individual crops/history. Synthetic canonical-context regression: 16
focused tests passed. Fourth fresh standard run prepared in
`tmp/luna-visual-projection-c6MrB1/`; same task/seed/budgets, no injected answer.

Fourth session `74c275b3-1794-4925-a0b8-d3dc1b072a31`: direct reference lookup,
SAM3 text query, correct left-side alphabet-soup can detection (bbox
[109,219,148,267]), explicit bundle selection, then a correct grasp-input bundle.
The actual selection request now attaches `target_selection_review`, its source,
all three catalog views and current scenes. AnyGrasp produced 20 candidates.
This establishes one live localization/selection/grasp-input milestone, not
robustness across episodes or task completion. Physical outcomes follow below.
Post-projection full suite: **2427 passed / 25 skipped / one existing authority
failure**, `tmp/luna-visual-projection-tests.xml` / `.log`.

### Fourth run: physical execution and safe rejection

- Advisor retained 7 feasible-geometry candidates; Agent explicitly compiled
  candidate `gpe-f15e2872f9104e71-000` with `top-down-vertical-panda-p8`.
- Clearance IK and execution succeeded: 13 controller steps, 2.04 mm Euclidean
  position error, 0.0845 rad orientation error, recorded matching seed ID and
  trajectory/world collision checks with no detected collision.
- Contact preview found a seed with 6.3 micrometre position residual, 0.00121 rad
  orientation residual and 0.852 rad joint margin. Actual contact ran 150 steps,
  stopped at `iteration_limit`, 22.42 mm Euclidean / 20.62 mm max-axis error and
  0.282 rad orientation error. This is seed-forwarding evidence, not proof of
  successful local convergence. Large final joint deviation from the seed remains
  unexplained; collision constraints versus controller dynamics are hypotheses.
- One malformed XML (duplicated partial decision prefix) was repaired next try.
  Fresh SAM3 reselection used the correct Host anchor and `same_instance`.
  A subsequent close request was BLOCKED by `compiled_contact_not_reached`
  (1 cm independent close envelope); no gripper close was dispatched.
- Fresh grasp advisor abstained on four rim/cap contacts. Agent requested a wrist
  viewpoint, previewed its exact target bundle, and attempted checked motion.
  That move failed with `constraint_escape_preview_rejected`, roughly 22 mm
  residual. No successful grasp, lift, attachment PASS or task reward claimed.
- Diagnostic stop after 17 episode steps: 19 recorded main requests / 979723
  known main tokens; 22 model records including VDM, all recorded provider models
  Luna. Advisor and interrupted/in-flight usage is not fully captured by the main
  token total. Budgets were not exhausted. `episode_interrupt.cleanup` closed/ok,
  no errors; client exited 130, server exited 0 and port verified free.

## Local controller isolation (no provider)

`scripts/replay_recorded_mink_approach.py` reuses the first recorded clearance and
contact poses, exact IK joints, tolerances, iteration budgets and compiled anchor
in a fresh loopback-only simulator. It makes no perception/LLM call, performs no
close/release, retains collision checks and stops after failed arrival. This
bypasses Agent planning for isolation only and cannot count as Agent acceptance.
Four fixture tests cover preservation, unsafe-check rejection, bounded stopping,
timeout cleanup and external URL rejection. Run root:
`tmp/mink-recorded-replay-kKh0fj/`.

- `replay-A.json`: essentially reproduces the paid run numerically: clearance
  13 steps / 2.037 mm; contact 150 steps / 22.421 mm / 0.282 rad, not reached.
  Target contact authorization correctly resolves to alphabet soup. This failure
  does not require planner/provider involvement to reproduce.
- `replay-C.json`: existing host-only C configuration, unchanged poses/seeds and
  tolerances: clearance 21 steps / 4.737 mm, stable arrival; contact 150 steps /
  25.801 mm / 0.191 rad, progress stall. C changes several settings, so this is
  not a seed-weight-only ablation and is not promoted to the default.
- Both environments closed/ok/no cleanup errors; dedicated servers stopped.
  The collision receipt's 20 mm minimum is distance-query saturation, not proof
  that contact has not occurred: authorized target contact is excluded from its
  protected pairs. Boundary recovery also includes QP fallback/joint recovery.

### Confirmed controlled-subspace constraint conflict

`scripts/diagnose_recorded_mink_approach.py` executes the worker controller directly
with a telemetry wrapper around the unchanged Mink solver. It records private
joint/solver/physical-contact data locally, never in planner context. It is not a
standard batch or task-performance sample. The direct-worker baseline is close
but not identical to MCP replay (22.872 mm vs 22.421 mm), so comparisons below
use paired direct-worker runs rather than combining their exact metrics.

`telemetry-A.json`: contact has 224 failed QP attempts (strict and fallback) in
150 executed steps. The first failure is at step 12: uncontrolled
`gripper0_finger_joint1` is -1.61 mm outside its lower limit. Full-scene
`ConfigurationLimit` requires inward finger motion, while
`_FixedVelocitySubspaceLimit` requires that same velocity to be zero. Arm-only
motion cannot satisfy both. This is a confirmed modeling conflict, not evidence
that the Cartesian pose is globally unreachable.

Fix: `_arm_configuration_limit` retains Mink's exact upper/lower inequalities
for every controlled arm DoF and fails closed for missing/duplicate limits.
Non-arm predicted velocities remain fixed; protected collision pairs, carried
object checks and physical MuJoCo limits are unchanged. No task-stage rule or
Agent-authored safety override was introduced.

`telemetry-arm-limits-A.json`: 150 strict QPs solve, zero QP failures; endpoint
still 22.872 mm. The conflict is fixed, but it was not sufficient to fix the
physical contact failure. Seven numerical tests pass in the LIBERO Python with
the validated Mink dependency overlay, including out-of-range fingers and
retained arm limits; the ordinary agent environment skips these optional-dep tests.

`telemetry-open-A.json`: opening during the same approach also fails at 22.407 mm.
Private physical contacts show the finger body/pad hitting the authorized target
from contact step 9 (about 5 mm initial penetration, about 6 mm at the end).
This distinguishes target-entry obstruction from protected-world collision.
`telemetry-open-first-A.json` first executes the existing 40-step opening horizon:
measured openness is 0.9921, but contact still fails at 22.413 mm / 0.281 rad,
with finger/target contact from step 9. Missing opening alone therefore does not
explain this fixture. Candidate geometry and strategy orientation changes need
review: the advisor assessed the native candidate, while the chosen compile
strategy clamps it to a fixed top-down pose. Native grasp quality must not be
assumed to validate a changed execution geometry.

Advisor prompt now explicitly limits its recommendation to the rendered native
candidate, and the compiler's existing warning reports this loss of applicability
when orientation is clamped. This is truthful evidence-scoping guidance, **not**
the still-needed compiled-geometry revalidation or a physical recovery fix.

Latest full suite after controlled-subspace fix: **2431 passed / 32 skipped /
one existing reviewed-authority failure**, `tmp/luna-task-recovery-controller-tests.xml`
and `.log`. Seven new optional-Mink tests are skipped by the ordinary environment
but all seven pass in the LIBERO worker Python with its dependency overlay.
After the final advisor/compiler wording change, 62 focused geometry/advisor/
recovery tests passed. `git diff --check` passed. All local diagnostic environments
report cleanup ok; batch/server processes are stopped and port 18766 is free.

## Handoff: acceptance remains incomplete

Stages 1 and one Stage 2 live chain passed. Stage 3 clearance passed, but no
successful contact/close/lift; Stage 4 Object 0 completion, Spatial 0 and Long 9
remain unattempted after that failure. No new paid run follows the deterministic
physical failure. Next implementation priority is final compiled-geometry support
and entry-clearance assessment using observed RGB-D/masks, scoped to the exact
orientation/contact point, plus useful distinction between target obstruction,
protected-world collision and QP infeasibility. Do not use private simulator
contact labels as oracle planner input, loosen close tolerance, or promote C on
the strength of these failed local replays. Shared contract review is still pending.
