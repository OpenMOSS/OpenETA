# UniVTAC tactile-agent research plan

This is the canonical research-plan and status entry point for
`tactile-agent-for-univtac`, updated on 2026-09-08 against the current checkout
and retained run evidence. The design below is the next research direction;
it is not a claim that all planned capabilities already work.

[Related Work](related-work.md) and its [BibTeX](related-work.bib) collect the
primary references, reading depth, and design implications. R1.5 implements
segment-end tactile-change clip selection, with local image difference as a
control. This is neither slip recognition nor an online motion
interrupt. Event detection supports observations and example organization
within the tactile ICL question, without changing the A/B/C comparison below.

## Research question

Can a frozen embodied Agent, without fine-tuning, use a few successful
tactile–action–outcome examples in context to make better use of current touch,
operate autonomously, improve UniVTAC native task success, and reduce trial and
error?

Vision and proprioception remain normal operating inputs. Tactile ICL does not
require touch-only control. Visual–action ICL is a control for whether any gain
comes from historical touch rather than simply seeing successful operations.
Prompt ordering, left/right image-patch questions, and opaque button mappings
are not the paper's main problem.

## Target operating loop

Use OpenMOSS/OpenETA's `openeta-for-codex` branch as the design reference and
reuse the existing OpenETA tools, MCP/Gateway/worker separation, and replay
facilities. The target is direct operation:

```text
current vision + bilateral touch + proprioception + operation history
    -> Codex decides the next operation
    -> observe / mark_point / move_to and ordinary gripper commands
    -> executor performs the specified command
    -> new observations and execution feedback
    -> Codex continues, adjusts, retreats, or recovers
    -> UniVTAC native checker evaluates the episode
```

The Agent chooses motion targets, direction, magnitude, orientation, and gripper
operation. IK, trajectory interpolation, and ordinary low-level control may
remain in the executor. They implement the requested motion, not a task solution.
Retain the initialization already included in official evaluation reset/`pre_move`;
from the official policy handoff point, the Agent decides the task body.

The online executor must not call the task expert, use hidden object/hole poses
to compute a correct correction, or automatically perform an expert prefix,
correction formula, or final insertion. Historical `ember`/`slate` expert buttons
are not the main experiment interface. A reusable skill may teach the Agent how
to operate, but must not conceal an online expert solution or take over control.

Allow observation, probing, and recovery within the same episode and shared
budget. These are distinct from restarting failed episodes or selecting the
best attempt. Fresh runs use `native_eval_no_online_task_feedback_v1`.
Native success latching, early failure and budget termination remain active in
the background. Agent-visible tools expose no current-query task judgement,
reward, evaluator components, hidden target errors or specific native termination
reason. All termination reasons use `episode_ended`; further physical actions
are prohibited. `check_task` is unavailable, and `finish_episode` confirms
voluntary or completed ending without returning a score. Ordinary motion,
robot and gripper feedback, static public rules and historical expert outcomes
remain available. The R1.4 general-tool backend completed two unscored control-debug episodes
and three fresh no-demo Codex episodes: all three were natively evaluable,
with autonomous development success 0/3.

## What a demonstration contains

Demonstrations must come from actual successful execution of the current
UniVTAC-Isaac51 native expert. Human operations, Agent successes and fixed-target
diagnostic replays do not substitute for this expert bank. Preserve the actual
commands when recorded, units, coordinate frame and timing. If only robot states
are available, describe measured expert motion; do not invent original commands
or execution feedback, and do not rename native actions as OpenETA calls:

```text
task goal + pre-action vision / bilateral tactile short history / proprioception
    -> recorded expert action, or explicitly labelled measured expert motion
    -> recorded execution feedback (or explicitly unavailable)
    -> post-action vision / touch / proprioception
    -> subsequent operations and final native outcome
```

In question–answer terms, Q is the current task, state, recent action, and tactile
change; A is the operation actually performed. The following observations and
outcome show its consequences. Failed attempts and recovery inside a successful
trajectory may remain; do not present every step as optimal.

Deliver actual images or short sequences to the Agent, not just file paths,
array shapes, or a caption. Align bilateral tactile history with each action.
State action units and coordinate frames explicitly, preserving the actual call
and its scene context. New targets must be grounded in the current observation;
examples must not encourage copying world coordinates from a different scene.

R1.7 uses official published `isaac51/insert_hole` episodes 0 and 1, selected
in numeric ID order. Metadata records success and seeds/source seeds 0 and 1,
separate from query development seeds `1000003`–`1000005`. A task-specific
train/test split and the data-producing commit are not recorded in the inspected
release metadata. R1.6 local support data remain historical evidence and are not
mixed into this package. The offline expert may use
its native ground-truth algorithm. Agent-visible examples contain historical
observations, actual actions, feedback and success outcomes, without executable
truth-based correction formulas, hidden-pose queries or future query answers.
Visual-action and tactile-action exports share the same trajectories and
non-tactile content; the latter adds only aligned historical touch. Historical
EE poses and motions are not current-scene OpenETA TCP commands to copy blindly.

## Core comparison: change only historical examples

All three conditions receive the same current vision, touch, proprioception,
and current operation history. They share the frozen Agent, normal tool
instructions, tools, executor, task starting point, and action/time/context
budget policy. A must understand tool functions without guessing button meanings.

| Condition | Historical context | Current input |
| --- | --- | --- |
| A — no examples | No historical operation demonstrations | Vision + touch + proprioception + operation history |
| B — visual–action control | Historical vision, proprioception, actual actions, execution feedback, and outcomes; no historical touch or touch-derived descriptions/features | Same as A |
| C — tactile–action ICL (main method) | Exactly B's trajectories and non-tactile content, plus action-aligned bilateral tactile history and post-action touch | Same as A |

B/C use identical example sources, counts, actions, outcomes, and non-tactile
content. C versus A tests whether complete successful examples help operation.
C versus B tests whether historical touch adds value. Do not remove B's current
touch: that would conflate the value of current sensing with historical tactile
examples. Actual context and inference costs are recorded, including C's extra
images, under the common budget policy.

A system with no current touch may be a later supplementary baseline; it does
not replace A/B/C. Removing tactile input does not disable contact physics or
change the controller. The older vision-only pilot arms are not automatically
this new B condition.

D is an auxiliary historical-touch control, not a replacement for A/B/C. It
uses C's same demonstrations, measured actions, proprioception, feedback,
outcomes and bilateral tactile history, removing only historical external
vision images and their references. Current vision and touch stay unchanged.

## Current paper design and shot-selection experiment

The planned main table is eight tasks × A/B/C × 100 fixed query seeds (2400
planned cells). A has no historical examples; B has expert visual observations,
measured motion, proprioception, time and outcomes; C adds historical touch to
exactly B. All current inputs retain vision, bilateral tactile history, robot
state and execution feedback. The number of expert episodes in B/C is not yet
fixed at two.

Before that main table, the authorized experiment is Insert Tube, Lift Can,
Lift Bottle and Pull Out Key × C × 1/2/4-shot × 100 seeds: 1200 planned
fresh autonomous episodes. This supersedes the original B/C dispatch scope;
the original 2400-cell manifest and all executed B/C records remain intact. It contains no A/0-shot. A shot is a complete official
successful expert episode, retaining its existing segmentation and image format.
Each task reuses its unchanged official episodes 0/1 and adds the next two
metadata-success episodes in ID order. Four-shot nests the unchanged first two
examples. For one-shot, even query-list indices use example 0 and odd indices
use example 1 (50 queries each). The original assignment is unchanged.
C retains historical vision as well as bilateral touch; it is not touch-only D.

The user subsequently fixed 1000000–1000099, replacing the initial fallback
1000100–1000199 before any simulator or operator Codex launch. The complete list
is stored in `configs/univtac/main_query_seeds.json`; the future main table must
reference that same file. This range overlaps retained development seeds, which
is disclosed rather than described as an independent held-out test. Query seeds are
planned slots, not a quota of successful resets or successful tasks. Failed
initializations, accepted task failures and voluntary endings remain separate.
Completed accepted cells must be skipped after a runner interruption, including
completed native failures; neither their attempts nor their sample count resets.

The user will review the complete curves and select one common K for all main-
table tasks and both B/C. No per-task or per-condition best-K selection is
allowed. The matching four-task C_K results (400 planned cells) will be
referenced directly in the main table, including failures and unavailable cells,
without rerunning or counting them twice. Already executed B cells matching K
can also be reused; remaining main-table B cells are measured later. This round
does not require full B curves or C−B comparisons. Reuse requires identical expert IDs
and one-shot assignment, prompt, model/effort, protocol, controller and budgets;
matching query seeds alone is insufficient. Retain all shot curves and disclose
that these four tasks participated in selecting K: the reused data are not an
independent test outside configuration selection.

D (historical touch without external vision), E (expert motion records only),
current-touch ablations and sol/luna/terra model comparisons are deferred until
after the main table, with scope to be decided later. No tactile mismatch
experiment is authorized here. LEMMo-Plan remains a closest related-work
comparison; tactile ICL itself is not claimed as a first proposal. This design
update does not open a new literature survey.

The shot experiment keeps gpt-6-astra/low, the no-online-task-feedback protocol,
original control, pinned Isaac51 and each task's accepted public rules/budgets.
Use two simulator slots including cleanup and at most three pre-ready attempts
per cell, with attempts retained across the scope revision. The old
2400 Codex / 7200 simulator ceilings are historical upper bounds, not an
authorization to dispatch B or replenish any exhausted cell. Model-started episodes are
not initialization retries. Fixed expert media are shared read-only. Only slow
review videos are required by the latest user amendment; bounded offline media
processing must not hold up simulator/operator dispatch. Preserve every raw
recording; media can be rebuilt without replaying physics. The full main table
and all other ablations remain unstarted and unauthorized by this round.

## Evidence and implementation status

The existing Isaac 5.1 harness provides reusable simulation startup, native
reset/`pre_move`, head/wrist and bilateral tactile capture, proprioception,
Codex MCP communication, actual action traces, and dashboard replay. Exact
Agent inputs are retained in `operator_context.jsonl`; host-only diagnostics
must not be substituted for what the Agent saw.

R1.4 adds a general-tool MCP/Gateway path with a synchronous UniVTAC worker,
robot-target control, four-image observation projection, native checking, and
control/physics counters. The original task-specific and expert-assisted paths
remain historical implementations. Two separate unscored control debug runs
completed: small translation/rotation and close reached; opening remained partial
and was reported as not reached. The focused suite passed 22 tests. Head-camera
marking is supported; wrist geometry is unavailable due to stale camera poses,
while all four RGB/tactile images remain available. The three fresh Terra medium
episodes all ended through native early stop, with native success 0/3. This is
a no-demo autonomous development result, not evidence of tactile ICL benefit. See [Architecture](../architecture.md) for implementation pointers.

### R1.4 autonomous no-demo development result

The two independent control-debug episodes are excluded from this table.
All three formal development episodes had valid reset, available native
checking, and no worker infrastructure error. Each made five non-preview
requests, all of which physically moved the robot. None reached the tool,
control-step, or wall-time budget; the native early-stop condition ended them.

| Seed | Native success | Tools | Control steps | Physics steps | Simulation seconds | Codex wall seconds |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1000003 | false | 22 | 16 | 32 | 0.267 | 143.034 |
| 1000004 | false | 20 | 7 | 14 | 0.117 | 146.050 |
| 1000005 | false | 16 | 6 | 12 | 0.100 | 111.029 |

These counts start after official initialization. The Agent repeatedly chose
translations along world -Z (1000003/5) or its current TCP +Z (1000004), without
an orientation or gripper change in the formal episodes. The host-only native
checker recorded `inhand_bias` values of 45.77/46.10/46.93 mm, exceeding the
40 mm early-stop threshold. This native indicator is not a measured cumulative
slip path or proof that the object fully fell; failure attribution remains open. Motion-target arrival did not imply insertion success.
See the retained [summary](../../outputs/univtac-isaac51-r14/summary.json) and
[delivery metrics](../../outputs/univtac-isaac51-r14/delivery_metrics.json).

Workers saved and closed normally. All three Codex processes were stopped
after the frozen launcher's 15-second native-terminal grace period; complete
token usage is unavailable, not zero. Seed 1000004 did not call `finish_episode`;
the host finalized its native outcome. The post-batch launcher extends final
reporting grace to 300 seconds within the unchanged 3600-second Codex deadline.
This fix is not a rerun or recovered cost measurement; natural Agent finalization
and complete cost reporting were subsequently verified in R1.5 below; R1.4
usage remains unavailable.

GPT-6 Pro accepted R1.4 delivery and the unchanged 0/3 result, with limits on
failure attribution. In the frozen R1.4 implementation, when `gripper` was omitted, the controller used
the measured finger opening as its next target, rather than retaining the
previous commanded closing target. This is confirmed implementation behavior;
its effect on loaded grasp retention and these failures is unverified. Likewise,
pose arrival does not establish velocity settling or contact stability. The
formal trajectories contain no Agent-requested gripper changes, but that does
not exclude a gripper-control contribution to failure.

In R1.4, touch was observed between tool calls. Control-step logs contain robot targets
and state, not a complete within-action tactile sequence. No real-time tactile
controller or fully reliable contact-control capability is claimed. These
limits describe R1.4. R1.5 changes command retention and observation delivery as
recorded below; the old batch has not been rerun or relabelled. If the controller or sensing interface changes, future B/C must be
compared with A under that same version, not directly with this historical A.

The reusable command uses the existing r09 runtime and pinned Isaac51 source:

```bash
uv run --frozen --extra dev python scripts/univtac/run_autonomous_insert_hole.py \
  --config configs/univtac/autonomous_insert_hole.yaml \
  --mode batch --output-root outputs/univtac-isaac51-r15-new
```

Use a fresh output root. `--mode debug` runs separate unscored controls. These
commands are usage documentation, not authorization to repeat the completed
batch. Replay is at `http://127.0.0.1:9400/r14-autonomous`, served by the local
`univtac-r14-dashboard.service`; raw outputs remain local and are not in Git.

### R1.5 grasp commands and segment-end tactile history

R1.5 reuses the general tools, synchronous worker, native dynamic controller,
and r09 runtime. `GripperTargets` reads the last submitted two-finger command
from `Articulation._joint_pos_target_sim` after official reset returns. A narrow
adapter replaces only the scalar gripper dispatch inside the counted native
qpos action. It preserves the two targets independently; omitted gripper
commands never replace them with measured opening. Explicit close/open sets
and retains the native targets. No extra closing motion happens at takeover.

`arm_reached` is independent of gripper closure. A new gripper command gets a
finite wait: target error at most 1 mm, or both positions vary by at most 0.2 mm
in 0.10 simulation seconds, with a 40-control-step maximum wait. These settings
were retained after unscored debugging, not selected from formal outcomes.
Waiting ends without claiming stable grasp; subsequent moves retain the target.

One fresh no-Codex debug episode completed in
`outputs/univtac-isaac51-r15/debug/seed_1000003`: seven physical requests,
28 native control steps, 56 physics steps, and 29 four-view samples including
takeover. Inherited targets were 0.005721318535506725 and
0.005722052417695522 m in native joint1/joint2 order. Translation/rotation
preserved them. Close used six steps, and its zero targets persisted through
an ensuing translation; open used 16 steps in the disposable debug state.
Loaded stiffness/damping/effort limits were read as 2000/100/200 for both fingers
and were not changed. This debug is not part of task success evaluation.

Sampling reads the existing refreshed buffers after each native qpos control
step, copies arrays before buffer reuse, and writes raw frames independently
of offline video encoding. Debug intervals were 1/60 simulation second, with
matching four-sensor timestamps and consecutive frame counters. This is not
60 wall-clock frames/second or proof of zero renderer latency. No additional
physics was used for recording. The four-panel H.264 video and raw frame index
are retained for review; model thinking occurs while physics is paused.

The implemented detector is small fixed-ROI integer patch matching plus a
same-frame image-difference control, not a reproduction of a full slip method.
Each pad has independent quality and scores; low quality is explicitly marked
as image-difference fallback. Debug had one such left-pad transition among 28.
The initial thresholds (2 original-image pixels, difference 0.015, quality 0.35)
were retained after inspecting debug data. At segment end, up to four shared
times select before/preceding-peak/peak/latest frames, sorted and deduplicated.
Two labelled tactile strips plus current head/wrist use native MCP images;
`operator_context.jsonl` records the actual delivered payload. No demonstrations,
slip labels, automatic correction or tactile interrupts are added.

The frozen formal controller/observation version is commit `66afcc6`; one fresh
no-demo episode per seed completed under `outputs/univtac-isaac51-r15/batch`.
All three had valid reset, available native checking, and no infrastructure
error: **autonomous development success is 0/3**. The single debug above is
excluded. No formal episode was retried or used to tune the detector/controller.

| Seed | Native success / ending | MCP calls | Non-preview requests received / admitted / physical moves | Control / physics steps | Simulation seconds | Four-view samples | Codex wall seconds |
| --- | --- | ---: | --- | --- | ---: | ---: | ---: |
| 1000003 | false / native early stop | 15 | 7 / 7 / 7 | 27 / 54 | 0.450 | 28 | 147.784 |
| 1000004 | false / Agent finish | 17 | 9 / 9 / 9 | 72 / 144 | 1.200 | 73 | 166.735 |
| 1000005 | false / native early stop | 14 | 6 / 5 / 5 | 6 / 12 | 0.100 | 7 | 120.584 |

Counts start after official reset. Samples include takeover. Seed 1000005's
sixth non-preview request arrived after native termination and was rejected
before the session's admission counter; it caused no additional physical step.
No episode exhausted a development or native step budget.

The worker remained available for final reading and `finish_episode` until host
cleanup after Codex exit. All three models exited naturally with return code 0
and real usage. Seeds 1000003/5 continued for 20.767/32.568 wall seconds after
the launcher observed native terminal, within the 300-second grace and original
3600-second total limit. Seed 1000004 finished voluntarily; it is not a native
terminal-grace witness. Live runs did not exercise full grace/total-time expiry;
focused tests cover those cleanup branches. Results and recording indices were
saved before `SimulationApp.close`.

| Seed | Input tokens | Cached input tokens (included in input) | Output tokens | Reported reasoning tokens |
| --- | ---: | ---: | ---: | ---: |
| 1000003 | 581,584 | 522,880 | 2,947 | 1,870 |
| 1000004 | 660,481 | 592,256 | 4,410 | 3,034 |
| 1000005 | 482,666 | 433,408 | 2,195 | 1,149 |

These are reported usage fields, not additive independent cost categories.
Total input/output were 1,724,731/9,552 tokens; no monetary cost is inferred.

**What the recordings show:**

- **1000003:** the Agent requested four downward world-Z translations, opened
  the gripper, moved down again, then requested a 50 mm retreat. The retreat
  executed one control step before native early stop and did not reach its
  target. Host-only `inhand_bias` was 41.61 mm, above the unchanged 40 mm
  threshold. The final review image still shows the rod near the fixture;
  neither complete dropping nor a unique failure cause is established.
- **1000004:** downward probes were interleaved with open/close commands,
  followed by a 35 mm upward request with open, another close, and voluntary
  finish. Arm targets were reached, but native success remained false. The
  final head image shows the rod leaning diagonally out of the fixture below
  the gripper. Closing fingers to about 0.762 mm did not establish a retained
  object or successful insertion.
- **1000005:** three preview-resolved motions and two further downward moves
  preceded native early stop (`inhand_bias` 46.29 mm, host-only). The later
  15 mm retreat request was rejected without physics. The Agent never changed
  the inherited gripper command. The tactile images change substantially, but
  this is not a verified slip label or a correct-action diagnosis.

The native joint1/joint2 inherited target pairs, in mm, were respectively
(5.721921, 5.722739), (5.721828, 5.722760), and (5.721528, 5.722320). Command
changes were open; open/close/open/close; and none. Final target pairs were
(39, 39), (0, 0), and the retained initial pair; measured final openings were
(38.653348, 38.662773), (0.762617, 0.762174), and (5.722758, 5.721204) mm.
These names follow native finger-joint order; anatomical left/right mapping
has not been independently checked. Holding a command does not prove grasp
stability. All formal motion requests were translations; rotation was exercised
in the separate debug, not claimed as formal Agent behavior.

The formal recordings contain 108 four-view sample times and 105 control-step
transitions. Recorded sensor IDs were consecutive, with no unrefreshed sample,
no missing frame, matching four-sensor timestamps, and approximately 1/60-second
simulation intervals. This metadata agreement does not prove zero renderer lag.
Of 210 pad transitions, one used labelled image-difference fallback (the left
pad in seed 1000003); none did so in the other seeds. Tracking quality is not
semantic detection accuracy. The Agent received 7/9/5 segment histories with
17/24/11 selected shared time points: 104 tactile image cells in total, including
reused references. All delivered cells were checked against their raw PNGs and
matched pixel-for-pixel. Video decode produced 28/73/7 recorded frames, without
invented observations. Waiting for Codex does not advance simulation time.

Replay is at `http://127.0.0.1:9401/r15-autonomous`, served by the local
`univtac-r15-dashboard.service`. It separates full videos/raw frames and
host-only curves/diagnostics from actual Agent images, requests and responses;
it provides playback speed, frame stepping and both detectors' candidate windows.
The retained videos are [1000003](../../outputs/univtac-isaac51-r15/batch/seed_1000003/review.mp4),
[1000004](../../outputs/univtac-isaac51-r15/batch/seed_1000004/review.mp4),
[1000005](../../outputs/univtac-isaac51-r15/batch/seed_1000005/review.mp4), and
[unscored debug](../../outputs/univtac-isaac51-r15/debug/seed_1000003/review.mp4).
[Delivery metrics](../../outputs/univtac-isaac51-r15/delivery_metrics.json)
retain per-action feedback, command history, timing, sampling, usage and lifecycle
fields. These artifacts are local and excluded from Git. Post-batch replay
edits only clarify request counts, preview target annotations and display; they
do not change the frozen controller, selector, prompt or retained raw runs.

The 45 focused tests, scoped Ruff/compile checks, HTTP/range responses and video
decoding passed. No browser was available for an interactive playback acceptance
check; server/template and raw-frame checks do not replace that check.

This completes the R1.5 development scope, with no successful autonomous
trajectory to promote into a demonstration. No B/C runs or automatic example
selection were performed. Future A/B/C must share this controller and current
observation mechanism; a change relative to R1.4 cannot establish ICL benefit
or isolate the value of change selection. Integer patch tracking, contact
interpretation, reliable manipulation and success-example collection remain
limitations or future work, not reasons to retune this completed batch.

### R1.6 fixed-target pacing comparison and expert curation

R1.6 started at `4161567`; the six replay runs used controller commit `96c85b2`.
Subsequent changes concern offline videos, expert exports and documentation.
There were exactly six fresh fixed-target episodes, zero new Codex decision
experiments and zero expert recaptures. This is a controller diagnostic, not an
Agent success batch or the ICL A/B/C experiment. All six resets/outcomes were
valid; no infrastructure retry or speed/configuration search occurred.

The [comparison configuration](../../configs/univtac/motion_pacing_comparison.yaml)
freezes the 7/9/5 admitted R1.5 commands, including resolved world TCP position,
orientation and explicit gripper changes. Preview IDs are resolved once from
recorded execution; the rejected post-terminal request is excluded. Both variants
receive identical absolute requests, without accumulating new relative targets.
Execution order was original/candidate for 1000003, candidate/original for
1000004, and original/candidate for 1000005. Prepared commands and variant
configs were saved before the first launch.

`original` retains R1.5 execution semantics and remains the default. The optional
`paced_candidate` advances an intermediate Cartesian reference using control
simulation time, then uses the same IK and dynamic actuators. Translation is
bounded by a 0.04 m/s reference and shortest-path rotation by 0.4 rad/s; both
share a duration so they arrive together. At 1/60-second control dt, these are
at most 0.667 mm and 0.006667 rad per reference step. A 50 mm request requires
75 reference steps, leaving only five of the unchanged 80 segment steps for
tracking. Reference velocity is not an actual-velocity guarantee.

Final-target arrival, 3 mm / 3 degree tolerances, IK correction limits, gripper
commands, physics dt/decimation, actuator settings, native thresholds and the
300-control-step budget are unchanged. A no-simulator test documents that an
otherwise unchanged 1 mm request can return reached with zero physical steps
under the existing tolerance; the tolerance was not fixed in this comparison.
Native terminal stops immediately. A non-reached segment budget would stop the
replay without hidden steps; none of these six reached that budget branch.

| Seed | Variant | Native success / ending | Arrived / attempted / planned targets | Control / physics steps | Simulation seconds | Sample times | After-reset wall seconds |
| --- | --- | --- | --- | --- | ---: | ---: | ---: |
| 1000003 | original | false / early stop | 6 / 6 / 7 | 24 / 48 | 0.400 | 25 | 7.752 |
| 1000003 | paced_candidate | true / success | 6 / 7 / 7 | 127 / 254 | 2.117 | 128 | 25.731 |
| 1000004 | original | true / success | 9 / 9 / 9 | 72 / 144 | 1.200 | 73 | 19.119 |
| 1000004 | paced_candidate | true / success | 7 / 8 / 9 | 104 / 208 | 1.733 | 105 | 24.396 |
| 1000005 | original | false / early stop | 5 / 5 / 5 | 6 / 12 | 0.100 | 7 | 4.969 |
| 1000005 | paced_candidate | false / early stop | 4 / 5 / 5 | 89 / 178 | 1.483 | 90 | 19.734 |

The after-reset wall clock includes synchronous tool handling, not initialization,
video export or model thinking. Full worker wall times in seed order,
original/candidate, were 88.078/101.005, 108.491/112.291 and 85.659/109.106 seconds.
Every sampled transition corresponds to one native control step and two physics
steps; each episode has takeover plus one sample per control, without missing
or unrefreshed samples in the retained records. The native success latch can end
a request before its arm target is reached: this happened during retreat in
1000003 candidate and during the upward/open request in 1000004 candidate.
Those partial targets are not counted as arrived. No replay success enters the
expert demonstration bank.

**Measured comparison and limitations.** Peak sampled-interval TCP speeds were
0.616/0.045, 0.962/0.042 and 0.785/0.042 m/s (original/candidate); these are norms
of net displacement per control interval, not instantaneous physics-step peaks.
The candidate changed actual timing rather than just a reference plot. Maximum
along-request positional overshoot was approximately 2.059/0.017,
8.946/0.008 and 2.293/0 mm; this is a directional projection, not every possible
contact instability. Candidate arrived endpoints commonly stop near the 3 mm
tolerance boundary, so smaller image changes alone are not comparable progress.

Before the first explicit gripper change, both variants completed the same first
four targets for 1000003 and first three for 1000004. Maximum host-only native
in-hand Z change over those prefixes was 32.58/27.44 and 15.41/10.89 mm.
Later explicit opens are retained, not attributed entirely to motion speed.
1000005 never changed gripper command: both variants stopped early. Its candidate
ended 7.07 mm from the final target versus original's 2.15 mm, so the smaller
40.36 versus 46.32 mm in-hand change does not establish better stability at
matched completion. This field is the native relative-pose indicator, not a
measured cumulative slip path.

Initial states were not bit-identical. Candidate-minus-original initial TCP
position differences (mm) were (0.031, 0.024, -0.295),
(-0.046, -0.001, -0.001), and (-0.087, -0.023, -0.320). Initial prism-relative-to-
gripper X differed by about +1.76, -0.97 and -2.48 mm; orientations, robot joints,
finger targets and measured openings are retained in the offline comparison.
These contact differences limit causal attribution from three single pairs.
1000004 succeeded in both variants; 1000005 failed in both, with more candidate
steps and less final completion. The observed native-success count 1/3 versus
2/3 is a development result for these commands, not evidence that speed is the
unique cause of historical failures or an ICL gain. Keep the candidate for user
review; do not replace the default on this evidence alone.

**How to watch.** The [local R1.6 page](http://127.0.0.1:9401/r16-motion-pacing)
provides Chinese explanations, separate four-view videos, side-by-side 1× and
0.05× simulation-time versions, and one-second-per-frame inspection copies.
Both sides share the takeover-relative clock and playback multiplier. A shorter
side explicitly holds its last frame; no independent length normalization or
synthetic intermediate observations are used. Display holds, including the final
one-second hold, are playback metadata rather than additional physical time.

- **1000003:** first compare the four-target prefix, then action 5 open and the
  retreat. Original stops during action 6; candidate reaches native success
  during action 7 before completing that retreat target. The final external
  views show a released rod near the fixture; native success is reported from
  the checker, not inferred from the image.
- **1000004:** both succeed. Compare the first three targets and action 8's
  upward/open operation. Candidate terminates earlier in that command; original
  completes the final close too. Do not present different termination points
  as equal completion of the full nine-command list.
- **1000005:** the 0.1-second original is especially short; use its frame-hold
  version, then the common-rate side-by-side video. Both show tactile-pattern
  changes, and both stop early. Candidate moves more slowly but has not finished
  the final target; the pattern difference is not a slip label.

**Expert bank.** Support seeds 1000000/1000001 reuse actual R1.2 native
`task.play_once()` successes under the pinned Isaac51 collect configuration:
120 Hz control/physics, decimation 1, native planner/dense moves (default
`force=True`). These are not the dynamic OpenETA replay controller and not
proof that the backends are equivalent. Each has three aligned before/after
vision, bilateral touch, robot-state and native-action pairs. Source steps were
395→560→646→724 (checker at 744) and 446→671→811→934 (checker at 954).
The timestamps derive from these collect step differences / 120; no dense
sensor frame IDs or instantaneous velocity is fabricated. Native save/video
frequency had been disabled in the R1.2 harness, so these are not official
full HDF5 collections. Existing data suffice for sparse boundary examples;
no expert recapture was required.

Each export preserves native EE/base position in metres, quaternion wxyz,
actual target, realized displacement, time-dilation parameter and outcome;
it does not claim the expert called OpenETA tools. Dense tactile motion,
per-frame sensor IDs and gripper target-buffer history remain unavailable.
The visual-action and tactile-action JSONs differ only by aligned touch, with
real PNGs copied from the same original trajectory. Provenance manifests remain
host-only. The separately labelled expert videos dwell on six boundary images,
not a continuous physical process. No human/Agent/fixed-replay success was
substituted and no ICL inference was run.

Artifacts live in `outputs/univtac-isaac51-r16/`: `commands.json`, per-variant
control/sample/contact logs, `comparison_summary.json`, three pair folders,
and `expert/seed_1000000` / `seed_1000001`. They are excluded from Git.
Reproduction uses a fresh output root; the second command is offline only:

```bash
uv run --frozen --extra dev python -m scripts.univtac.run_motion_pacing_comparison \
  --output-root outputs/univtac-isaac51-r16-new
uv run --frozen --extra dev python -m scripts.univtac.motion_pacing_review \
  --root outputs/univtac-isaac51-r16-new
```

The 47 focused tests, scoped Ruff/compile checks and Markdown links passed.
Video presentation timestamps were checked against the simulation-time mapping;
actual browser loading, playback and seeking were exercised, and key frames
were visually inspected. This is not a claim of an independent frame-by-frame
review of every video. Scoped neat reconciled the current expert-source rule
and retained the historical results.

This completed batch is not authorization to repeat it. The replay page's fixed
route points at the retained R1.6 root; another root's generated `review.html`
can be served through the existing artifact route. No new autonomous or ICL
batch starts until the user has reviewed the videos and decided.

### Historical evidence, with its original scope

These are development results, not one autonomous-success leaderboard. Raw
results and unsuccessful episodes remain unchanged in their original output
roots; the links below refer to local retained artifacts, not a portable dataset.

- **R1.0, native expert:** Pull Out Key succeeded on 3/3 development seeds,
  with nine recorded action transitions
  ([expert summary](../../outputs/univtac-isaac51-r10/expert_summary.json)).
  **Separately, a restricted-skill Codex smoke** selected three native skills
  in one live episode and reached native success
  ([episode](../../outputs/univtac-isaac51-r10/agent/attempt_02/seed_1000000/episode.json)).
  This demonstrates real model-selected actions and feedback, not autonomous
  choice of motion parameters.
- **R1.1, restricted-skill ICL:** Pull Out Key continuation success counts were
  3/3 without examples, 3/3 with correct multimodal examples, 2/3 in the old
  visual-only condition, and 1/3 with swapped action labels. The recorded
  interpretation is `no_detectable_icl_signal`
  ([summary](../../outputs/univtac-isaac51-r11/summary.json)). Native outcomes
  and the report's selection-based gains are different metrics; neither
  establishes a positive tactile ICL result under the new A/B/C design.
- **R1.2, expert-assisted task qualification:** native experts succeeded 3/3
  on both Lift Bottle and Insert Hole. Insert Hole showed two correction
  classes; a one-seed correct/wrong continuation comparison succeeded/failed
  respectively ([summary](../../outputs/univtac-isaac51-r12/summary.json)).
  This motivates a development task, not an autonomous policy claim.
- **R1.3, expert-assisted Insert Hole continuation:** three query experts
  succeeded; 12 Codex choices and 12 continuations brought the total to 15
  simulator episodes. No-demo, correct multimodal, and old vision-only arms
  each succeeded 2/3; swapped labels succeeded 1/3
  ([summary](../../outputs/univtac-isaac51-r13/summary.json),
  [manifest](../../outputs/univtac-isaac51-r13/run_manifest.json)). The Agent
  chose between two opaque procedures; the harness supplied the initial
  downward move, native correction magnitude/z, and final insertion. The
  result does not establish action-ICL or tactile-specific benefit, and is not
  a general-tool autonomous Agent success rate.

R0.9.13–R0.9.18 remain observation/representation diagnostics. By project decision,
R0.9.19–R0.9.21 are historical exploration only: preserve their records, do not
use them to choose the next method, and do not expand that branch. R0.9.21's
capture-only record must not be relabelled as a completed transfer experiment.

## R1.7 autonomous tactile ICL pilot and next work

R1.7 completed nine fresh Insert Hole episodes with `gpt-5.6-terra` / medium,
the original controller, and identical current tactile-history mechanisms.
The task began at `b60aa97`; frozen experiment code was `51e6a0e`. There were
no added expert/debug episodes, task-failure retries, or controller changes.

Official HDF5 episodes 0/1 contain 189/166 recorded frames and three movement/
delay segments each. B received 12 before/after vision strips; C received the
same non-tactile content and vision strips plus 12 bilateral tactile-history
strips. Common boundaries and visual times use movement metadata, independently
of touch. HDF5 contains JPEG image streams, `embodiment/ee`, `embodiment/joint`,
`step`, and `atom/id` / `atom/tag`, but no independent raw action commands.
Measured EE is base-relative panda_hand xyz + wxyz, not current world TCP;
nominal seconds use recorded step differences / 120, without an embedded clock.
The documentation revision is not the data-producing commit and does not
upgrade our pinned runtime. The task-specific train/test split remains unknown.
Each real Codex context reviewed its fixed demonstration/empty result once.
A received no demonstration images, B received 12 native MCP images, and C
received 24, followed in all conditions by live observations and autonomous
commands. Actual delivery is recorded in each `operator_context.jsonl`.

A (`no_demo`), B (`visual_action_icl`), and C (`tactile_action_icl`) each achieved
**0/3 native task successes**. All nine episodes were evaluable, ended with
`native_early_stop`, and had natural Codex exits with recorded usage; no
infrastructure errors were reported. C−A, C−B and B−A were all zero percentage
points. This three-seed development pilot found no success-rate benefit; it
does not establish that tactile ICL is generally ineffective.

| Seed | Condition | Native success | Actual motions | Control steps | Simulation s | Codex wall s |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| 1000003 | A | false | 4 | 11 | 0.183 | 348.21 |
| 1000003 | B | false | 4 | 7 | 0.117 | 97.18 |
| 1000003 | C | false | 3 | 11 | 0.183 | 144.83 |
| 1000004 | A | false | 12 | 78 | 1.300 | 253.90 |
| 1000004 | B | false | 3 | 15 | 0.250 | 124.53 |
| 1000004 | C | false | 3 | 6 | 0.100 | 94.53 |
| 1000005 | A | false | 5 | 24 | 0.400 | 170.19 |
| 1000005 | B | false | 4 | 11 | 0.183 | 129.03 |
| 1000005 | C | false | 2 | 6 | 0.100 | 110.93 |

Physics steps are twice the control steps in this batch. The
[results](../../outputs/univtac-isaac51-r17/results.json) retain per-episode
tool counts, worker time, raw input/cached/output/reasoning token usage, video
paths and errors; the [manifest](../../outputs/univtac-isaac51-r17/batch/run_manifest.json)
records execution. Earlier termination or fewer actions is not evidence of
greater task efficiency when the task failed.

Most B/C runs mainly attempted downward motions. Seed 1000004 A made twelve
motions including vertical probing and lateral adjustment; seed 1000005 A
opened the gripper and subsequently lifted before early termination. Inspect
these different trajectories individually rather than attributing all failures
to a single claim that the Agent did not understand touch.

The [episode dashboard](http://127.0.0.1:9401/r17-autonomous) shows actual
operator input, requested commands, execution feedback, outcomes and cost.
[Videos](http://127.0.0.1:9401/r17-videos) include all nine 1× recordings and
slow versions: 0.1× for seed 1000004 A, 0.05× for the other eight. Existing raw
frame views support frame-by-frame inspection. Slow playback repeats recorded
frames, without synthesizing observations.
[Demonstration previews](http://127.0.0.1:9401/r17-demonstrations) keep the
historical official experts separate from autonomous episodes.

Read the downloaded official files with the existing r09 Python (fresh output):

```bash
/home/ubuntu/anaconda3/envs/UniVTAC-isaac51-sm120-r09/bin/python3.11 \
  -m scripts.univtac.prepare_official_demonstrations \
  --raw outputs/univtac-isaac51-r17/data/official \
  --output outputs/official-demonstrations-new
```

Use the [fixed configuration](../../configs/univtac/official_tactile_icl.yaml)
and a fresh batch output directory:

```bash
uv run --no-sync python -m scripts.univtac.run_official_tactile_icl \
  --config configs/univtac/official_tactile_icl.yaml \
  --demonstrations outputs/univtac-isaac51-r17/demonstrations \
  --output-root outputs/univtac-isaac51-r17-new-batch
```

These commands describe reuse, not authorization to repeat this pilot. The
completed round stops for review of the actual inputs and videos. Additional
seeds, model/controller changes, or another round require a new decision.

The first representation is real tactile images and short action-aligned
history. Later candidates include baseline/current or difference images,
image-derived structured measurements, Octopi-1.5, VT-MUSE-inspired temporal
representations, and reusable operation skills. These enhance the same tactile
ICL question; none is a prerequisite for direct operation or the first ICL test.
Octopi has not been validated on our simulated touch, and VT-MUSE is not an
integrated, effective component. Image change alone is not verified force,
slip, or stable grasp.

The final target is main results on at least three UniVTAC-Isaac51 manipulation
tasks; candidates are Insert Hole, Pull Out Key, and Lift Bottle. Use a fixed
formal test batch independent of development and example selection. The primary
metric is native task success rate. Record action count, recovery behavior,
completion time, context usage, and inference cost as secondary metrics. Report
expert success, expert-assisted continuation success, and autonomous Agent
success separately, not in one main-result table.

All conditions use the same Isaac51 task implementation. These results cannot
be presented as direct reproductions of FTP-1/Isaac 4.5 numbers. See
[Isaac 5.1 compatibility boundary](isaac51_compatibility_boundary.md) and
[Vendor Notes](../vendor-notes.md). This documentation update does not launch or
authorize execution of the next experiments.


## R1.8: public rules × tactile ICL with GPT-6 low

R1.8 completed all 18 fresh development episodes using execution commit
`902178e`, starting from `3ae61ee`. All 18 were natively evaluable, with no
infrastructure failures, natural Codex exits and recorded usage. New project operators and execution/review/neat agents use
`gpt-6-astra` with `model_reasoning_effort="low"` explicitly passed to Codex.
Historical R1.7 and earlier model labels remain unchanged.

U uses the pinned Isaac51 seen instruction: “Explore the inclined hole through
contact to determine its orientation, then insert the test tube into the hole.”
R adds static public success/early-stop rules, without current hidden state or
motion advice. R is the preselected full-information setting; U removes those
extra rules. The exact rule text and unchanged control budgets are in
[the R1.8 config](../../configs/univtac/task_rule_tactile_icl.yaml). Each frozen
condition prompt is saved before the first episode; initial prompt and observe
share the official instruction. U does not receive the additional numeric rules.

A has no demonstrations; B reuses R1.7 official Isaac51 `0.hdf5`/`1.hdf5`
visual–measured-motion examples; C adds their existing aligned bilateral touch.
The historical package stays intact. Only its generic task-goal sentence is
replaced at delivery with the same official sentence for B/C; examples,
segmentation, selected images and measured movements are unchanged. Every
condition retains current vision, tactile history, robot state and feedback.

The fixed order is seed 1000003: UA/RA/UB/RB/UC/RC; seed 1000004:
RB/UB/RC/UC/RA/UA; seed 1000005: UC/RC/UA/RA/UB/RB. Each cell is a fresh live
Codex episode under the original controller, native 300-control-step budget,
80-step segment budget, 30 admitted motion requests, 100 tools, 3600 seconds
Codex and the existing terminal grace. No extra simulator debug, expert replay
or model capability run is part of this round.

```bash
uv run --no-sync python scripts/univtac/run_official_tactile_icl.py \
  --config configs/univtac/task_rule_tactile_icl.yaml \
  --demonstrations outputs/univtac-isaac51-r17/demonstrations \
  --output-root outputs/univtac-isaac51-r18/batch
```

Report six native successes/3 and evaluability separately. Fixed percentage-point
contrasts are RA−UA, RB−UB and RC−UC for rules; RB−RA and RC−RA for demonstrations;
RC−RB and UC−UB for historical touch. R1.7 differs in model and task wording and
cannot isolate either effect. Three development seeds do not establish held-out
generalization. Failures and incomplete cells remain visible.

The existing dashboard exposes `/r18-autonomous`, `/r18-videos` and
`/r18-demonstrations` on port 9401. It shows actual initial prompts and native MCP
inputs separately from host-only review videos. All 18 four-view recordings and their slow versions were generated from retained
frames. Browser playback reached the end for all 36 files; representative frames
were also visually inspected. This is playback verification, not a claim of
human inspection of every video frame.


### R1.8 observed results

| Task information | A: no demo | B: visual–motion | C: added historical touch |
|---|---:|---:|---:|
| U: official instruction | 1/3 | 1/3 | 3/3 |
| R: instruction + public rules | 1/3 | 0/3 | 2/3 |

Every cell is evaluable. The eight successes ended with `native_success`; the
other ten ended with `native_early_stop`. No native/control/tool/wall-clock budget
was extended and no failed cell was rerun. R remains the preselected full-input
setting, despite U's higher C score.

Rule contrasts RA−UA, RB−UB and RC−UC are respectively 0.0, −33.3 and −33.3
percentage points. Within R, RB−RA is −33.3 pp, RC−RA is +33.3 pp, and RC−RB
is +66.7 pp. UC−UB is also +66.7 pp. Thus this pilot observed a positive
historical-touch difference against matched visual examples in both rows, but
no success-rate improvement from supplying the additional rules. These are
three-seed descriptive differences, not proof of a general benefit or of rules
being harmful. Fresh resets showed initial-state differences; historical
examples were fixed, but current physical states were not pixel-identical.
R1.7 is not a same-model control.

| Seed | Cell | Native result | Actual motions | Control / physics steps | Sim s | Codex / worker wall s |
|---|---|---|---:|---:|---:|---:|
| 1000003 | UA | early stop | 15 | 77 / 154 | 1.283 | 191.4 / 311.1 |
| 1000003 | RA | early stop | 10 | 40 / 80 | 0.667 | 129.1 / 272.9 |
| 1000003 | UB | early stop | 4 | 14 / 28 | 0.233 | 69.3 / 195.0 |
| 1000003 | RB | early stop | 7 | 17 / 34 | 0.283 | 92.8 / 217.5 |
| 1000003 | UC | success | 5 | 18 / 36 | 0.300 | 77.6 / 206.3 |
| 1000003 | RC | success | 6 | 18 / 36 | 0.300 | 310.0 / 434.7 |
| 1000004 | RB | early stop | 5 | 14 / 28 | 0.233 | 182.2 / 315.9 |
| 1000004 | UB | early stop | 5 | 13 / 26 | 0.217 | 74.0 / 200.6 |
| 1000004 | RC | early stop | 6 | 17 / 34 | 0.283 | 96.5 / 230.1 |
| 1000004 | UC | success | 6 | 19 / 38 | 0.317 | 86.3 / 216.1 |
| 1000004 | RA | early stop | 13 | 64 / 128 | 1.067 | 209.1 / 337.7 |
| 1000004 | UA | early stop | 9 | 28 / 56 | 0.467 | 126.8 / 253.6 |
| 1000005 | UC | success | 6 | 18 / 36 | 0.300 | 102.7 / 230.4 |
| 1000005 | RC | success | 7 | 20 / 40 | 0.333 | 299.3 / 429.0 |
| 1000005 | UA | success | 8 | 41 / 82 | 0.683 | 152.1 / 278.9 |
| 1000005 | RA | success | 11 | 72 / 144 | 1.200 | 182.4 / 308.1 |
| 1000005 | UB | success | 5 | 18 / 36 | 0.300 | 70.0 / 193.8 |
| 1000005 | RB | early stop | 6 | 15 / 30 | 0.250 | 75.9 / 199.7 |

Worker wall time includes initialization and shutdown; it excludes offline
review encoding. Codex wall time includes model/tool waits. Two RC episodes
had long waits after demonstration delivery and eventually completed normally;
model-catalog refresh timeout messages in stderr did not become episode failures.
Do not interpret shorter failed episodes as higher efficiency.

| Cell | Input tokens | Cached input (subset) | Output tokens | Reasoning output (subset) |
|---|---:|---:|---:|---:|
| UA | 1,807,564 | 1,666,304 | 5,353 | 880 |
| UB | 1,041,113 | 915,200 | 2,427 | 218 |
| UC | 1,374,565 | 1,218,048 | 2,671 | 200 |
| RA | 2,173,043 | 1,994,624 | 6,189 | 1,566 |
| RB | 1,296,784 | 1,158,784 | 2,692 | 223 |
| RC | 1,476,663 | 1,299,328 | 2,959 | 259 |

These are raw CLI cumulative fields summed over three episodes per condition;
cached input is not added to input, nor reasoning output to output. All actual
commands record `gpt-6-astra`, `model_reasoning_effort="low"` and CLI 0.153.4.
A/B/C delivered 0/12/24 historical images per context, once each. The ordinary
model/effort, prompt, image-count and budget readback is retained in
`outputs/univtac-isaac51-r18/delivery_validation.json`.

**How to watch.** On [the video page](http://127.0.0.1:9401/r18-videos), start
with seed 1000003 UB versus UC: UB explored positive x before returning downward
and failed, while UC explored negative x before positive-x/downward moves and
succeeded. Then compare seed 1000004 RC with UC, where the exploration directions
also differed and only UC succeeded. Seed 1000005 UA/RA show successful explicit
orientation adjustment without examples. All six A episodes explicitly requested
orientation changes; B/C episodes requested translations without explicit orientation changes. Seed 1000005 UA and RA explicitly commanded close and later open with retreat;
native success arrived during that final action. The other sixteen episodes
retained inherited gripper commands. Seed 1000005 UA also recovered from one
`mark_point` rejection for a pixel without valid observed depth by choosing
other pixels; this was a recoverable tool response, not an infrastructure failure. These request patterns describe behavior, not evidence that a
particular tactile feature caused a choice. Actual robot orientation can change
even when no new orientation was requested.

Videos use 0.1× for seed 1000003 UA/RA, seed 1000004 RA, and seed 1000005 UA/RA;
the other thirteen use 0.05×. Each also has a 1× version. The
[dashboard](http://127.0.0.1:9401/r18-autonomous) provides original-frame stepping,
exact prompts, actual MCP images and responses, action feedback and separate
host-only diagnostics. The [demonstration preview](http://127.0.0.1:9401/r18-demonstrations)
shows the two fixed official experts separately from current Agent episodes.

Validation: 51 focused tests, scoped Ruff, compileall and diff checks passed.
No simulator/model capability pretest or extra physical attempt was added.
Pro subsequently accepted this result and assigned the frozen new-seed
validation below; no prompt or model comparison was added.


## R1.9: frozen method on twelve new query seeds

Pro accepted R1.8 as a positive development signal for historical touch, while
recognizing that extra public rules did not improve success. R1.9 completed
all 36 fresh episodes on seeds 1000006–1000017 with the preselected R input.
RA/RB/RC achieved **3/12, 10/12 and 11/12** native successes. The main
comparison RC−RB is **+8.3 percentage points**, only one discordant seed in
RC’s favor; RC−RA and RB−RA are +66.7 and +58.3 points. This is a small
positive new-seed signal for historical touch, much smaller than R1.8’s
three-seed difference. It does not establish a robust or universal tactile
advantage. R1.8 stays separate; its old seeds are not pooled into this table.

The scoped search of this checkout's 505 episode/manifest/summary JSON files
and documentation found no execution records for these seeds; only the R1.9
plan named them. Other historical worktrees and remotes were not searched.
This is project new-seed validation, not an official published test split.
No expert screening, model pretest or controller debug is part of this round.

[The fixed R1.9 config](../../configs/univtac/new_seed_tactile_icl.yaml) records
all 36 cells before launch, with six condition permutations repeated twice.
Generated prompts were compared directly with R1.8's saved R prompt; all three
are identical. Demonstration text/image references were also compared with
R1.8's actual delivered projections. The same R1.7 source package and unchanged
delivery transformation reproduce those projections; no data is reselected or
resegmented. Model, current observation/history, tools, original controller,
physical parameters and every budget remain unchanged.

```bash
uv run --no-sync python scripts/univtac/run_official_tactile_icl.py \
  --config configs/univtac/new_seed_tactile_icl.yaml \
  --demonstrations outputs/univtac-isaac51-r17/demonstrations \
  --output-root outputs/univtac-isaac51-r19/batch
```

All 36 planned cells were evaluable: 24 native successes and 12 native early
stops, with no infrastructure failures, replacements, retries or extra debug
episodes. No cell ended on a budget limit. All Codex processes exited naturally
and provided usage. The execution version was `c22dade`, starting from `e6ae93a`;
no controller, prompt, demonstration, sampling or budget changed during the
batch. The unrelated untracked `heldout_grounding_skill.py` was preserved.

The existing dashboard adds `/r19-autonomous`, `/r19-videos`,
`/r19-demonstrations` and `/r19-pairs`. RB/RC paired videos reuse the recorded-frame
alignment approach: common simulation time and playback rate, with the shorter
side explicitly holding its final frame. Commands may differ because the Agent
chooses them. They do not launch or advance physics. Every episode has a 1× video and a
slow version (0.05× below 0.5 seconds, otherwise 0.1×); each RB/RC pair has
1× and 0.05× versions. Full playback validation is recorded separately below.


### R1.9 outcomes and costs

| Condition | Planned / evaluable | Native success | Native failure | Infrastructure |
|---|---:|---:|---:|---:|
| RA: no examples | 12 / 12 | 3/12 (25.0%) | 9 | 0 |
| RB: visual–measured-motion examples | 12 / 12 | 10/12 (83.3%) | 2 | 0 |
| RC: same examples plus touch | 12 / 12 | 11/12 (91.7%) | 1 | 0 |

Each entry below is **native outcome; actual motion requests / control steps**.
All failures are native early stops. Physics steps equal twice control steps;
post-takeover simulation time is control steps / 60 seconds. Initialization is
recorded separately and is not folded into these task-body counts.

| Query seed | RA | RB | RC |
|---|---|---|---|
| 1000006 | early stop; 9 / 24 | success; 7 / 20 | success; 7 / 19 |
| 1000007 | early stop; 26 / 202 | early stop; 6 / 18 | success; 7 / 19 |
| 1000008 | success; 14 / 91 | success; 6 / 19 | success; 7 / 19 |
| 1000009 | early stop; 13 / 52 | success; 6 / 23 | success; 11 / 97 |
| 1000010 | early stop; 13 / 58 | success; 6 / 18 | success; 7 / 19 |
| 1000011 | early stop; 16 / 81 | success; 7 / 17 | success; 5 / 18 |
| 1000012 | success; 13 / 74 | success; 6 / 19 | success; 6 / 19 |
| 1000013 | early stop; 18 / 92 | success; 6 / 16 | success; 6 / 13 |
| 1000014 | early stop; 17 / 93 | success; 7 / 19 | success; 6 / 19 |
| 1000015 | success; 20 / 96 | success; 6 / 18 | success; 7 / 20 |
| 1000016 | early stop; 17 / 69 | early stop; 6 / 15 | early stop; 6 / 18 |
| 1000017 | early stop; 14 / 66 | success; 5 / 17 | success; 7 / 20 |

| Pair (RC versus control) | RC only succeeds | Control only succeeds | Both succeed | Both fail | Unavailable |
|---|---:|---:|---:|---:|---:|
| RC / RB | 1 | 0 | 10 | 1 | 0 |
| RC / RA | 8 | 0 | 3 | 1 | 0 |

The sole RC/RB discordant seed is 1000007. Both fail on 1000016. Thus the
new batch supports a limited positive replication signal, not the large effect
size seen on the old development seeds. The larger RB−RA difference belongs
to visual–motion examples; it must not be called a tactile benefit. Current
bilateral touch remains available in every condition. One attempt per cell
cannot separate model variability from fresh-reset variability, and this remains
one task on project-selected new seeds rather than a multi-task benchmark.

The following totals cover twelve episodes per condition. Worker wall time
includes initialization and shutdown, excludes offline review encoding; Codex
wall time includes model/tool waiting. Shorter failure is not efficiency gain.

| Condition | Motion requests | MCP calls | Control / physics steps | Sim seconds | Codex / worker wall seconds |
|---|---:|---:|---:|---:|---:|
| RA | 190 | 283 | 998 / 1996 | 16.633 | 3018.3 / 4682.4 |
| RB | 74 | 123 | 219 / 438 | 3.650 | 1167.5 / 2784.2 |
| RC | 82 | 140 | 300 / 600 | 5.000 | 1318.4 / 2889.0 |

| Condition | Input tokens | Cached input (subset) | Output tokens | Reasoning output (subset) |
|---|---:|---:|---:|---:|
| RA | 12,283,829 | 11,498,496 | 35,175 | 9,666 |
| RB | 5,109,416 | 4,546,944 | 12,201 | 1,508 |
| RC | 6,690,215 | 6,096,128 | 13,649 | 1,797 |

These are actual cumulative CLI fields, not estimates; subset fields are not
added again. All 36 commands explicitly use `gpt-6-astra`,
`model_reasoning_effort="low"`, CLI 0.153.4. Saved prompts equal the R1.8 R
prompt, and each context delivers demonstrations once: RA/RB/RC 0/12/24 images.
Readback of actual model arguments, prompt, demonstration text/image references,
image existence and budget/step counters passed for all 36. Evidence:
`outputs/univtac-isaac51-r19/delivery_validation.json`; full per-cell costs and
outcomes are in `results.json` beside it. There are 1,553 synchronized recorded
sample sets, including the takeover frame in each episode; no recording step
was added to physics.

### What to watch

Start at [RB/RC paired replay](http://127.0.0.1:9401/r19-pairs), then use
[all individual videos](http://127.0.0.1:9401/r19-videos) and
[actual Agent inputs and actions](http://127.0.0.1:9401/r19-autonomous).
The [fixed historical expert preview](http://127.0.0.1:9401/r19-demonstrations)
is separate from these current Agent results.

- **1000007 RB/RC:** the only RC-only success. Compare the actual requested
  directions and returned observations; do not infer that a particular touch
  feature caused the different decisions merely from the outcome.
- **1000016 RB/RC:** both fail. RB explores positive x before reversing;
  RC first explores negative x, then positive x/downward. Both retain inherited
  gripper targets and request no explicit orientation change. Different
  exploration directions did not ensure success. Watch the early-stop frame
  and the labelled end-of-side hold at equal playback speed.
- **1000009 RC:** a longer 97-step success, including translation, explicit
  close/open and orientation requests. It is not the same short translation-only
  pattern as most C episodes.
- **1000012 RA and 1000015 RA:** successful no-example recovery attempts after
  different orientation probes. Seed 1000012 succeeds during an open plus
  pose-change request, before reaching that request’s final target. Seed
  1000015 closes, tilts toward negative x and makes repeated diagonal moves,
  succeeding with the close target retained. These are observed sequences,
  not proof that opening or closing alone caused success.
- **1000007 RA:** 26 actual motions and 202 control steps end in native early
  stop, not budget exhaustion. Keep this long failed attempt alongside the
  shorter successes.

There was one recoverable `mark_point` invalid-depth-pixel response on
1000012/RA; the Agent selected a different pixel and continued. This is not an
infrastructure failure. A tool execution field containing `native_success`
marks task termination even when the requested arm target was not reached;
command arrival and native success remain distinct.

Prelaunch validation passed 55 focused tests covering new ordering, frozen
inputs/model, paired outcomes/timeline and previous interfaces, plus scoped
Ruff, compileall and diff checks. No extra simulator/model test was launched.
Final GPT-6 low full-history scoped neat found no result/cost inconsistency.
All 96 files (36 individual episodes × two rates, 12 pairs × two rates) reached
browser `ended`; receipts are in
`outputs/univtac-isaac51-r19/browser_playback_check.json`. This verifies native
playback, not manual viewing of every frame. Representative paired frames were
also visually inspected. Two offline playback-check interruptions were resolved
without regenerating observations: one long browser wait, and one navigation
collision between page inspection and playback. Playback now uses its own browser
session. Neither affected the simulator, Agent input or episode results.
This round adds no mechanism ablation or task; subsequent experimental action
requires the next concrete Pro instruction under the continuing project goal.


## R1.10: historical versus current tactile input

Pro accepted R1.9 while tightening the claim: expert demonstrations helped on
this batch, but the single extra RC success does not establish a reliable
historical-touch benefit. RB still had current touch; its 10/12 is not evidence
that vision alone suffices. R1.10 is a supplementary modality ablation, not a
change to the original A/B/C definitions or the complete multimodal method.

The [fixed configuration](../../configs/univtac/current_tactile_ablation.yaml)
completed eight new seeds 1000018–1000025, four conditions each (32 fresh episodes):
B_live / C_live retain current bilateral tactile history; B_no_live / C_no_live
omit it from Agent-visible outputs. B has the same twelve historical visual
images and measured-motion text as R1.9; C adds the same twelve historical
bilateral-touch images. C_no_live retains those historical tactile images.
There is no no-example A condition or reuse of R1.9 outcomes in the new table.

`project_current_observation` filters current touch after all normal and
recoverable-error observation paths in the existing session. It leaves the
host observation/recorder unchanged, removes current tactile image descriptors
and the complete tactile-history selection metadata, and does not affect
historical review. All four groups retain current head/wrist, robot/gripper
state, execution feedback, budgets and minimal native checks. Sensors, sampling,
reset/pre_move adaptive grasp and contact physics remain active in no_live;
it is not a claim of a system that never used tactile sensing.

All four prompts are identical. The R1.9 rule, coordinate and budget text is
unchanged; tactile format statements gain “when provided” and a common missing-
modality explanation. Model remains `gpt-6-astra` / low, with original control
and unchanged 30 motion / 100 MCP / 3600 Codex-second / 300 native-step /
80 segment-step limits and terminal grace. No debugging simulator episode,
model pretest, expert screening/collection or parameter change is added.

```bash
uv run --no-sync python scripts/univtac/run_official_tactile_icl.py \
  --config configs/univtac/current_tactile_ablation.yaml \
  --demonstrations outputs/univtac-isaac51-r17/demonstrations \
  --output-root outputs/univtac-isaac51-r110/batch
```

Prelaunch offline checks used retained R1.9 observations: removing current touch
preserved other fields and the host payload; all four historical projections
matched actual R1.9 delivery and the four prompts were equal. An explicitly
labelled two-frame offline video test reused R1.9 data; it is not a new physical
episode. The no_live touch region reads “仅供用户审阅，本episode未送给Agent”.
Raw frame stepping is host-only; Agent Saw uses actual operator_context images.
The existing dashboard adds `/r110-autonomous`, `/r110-videos`,
`/r110-demonstrations` and `/r110-pairs`; paired views compare live/no_live within
B and within C on common simulation time, holding an ended side's last frame.

The 2×2 main table reports successes/8, evaluability, native failures and
infrastructure issues. Fixed contrasts are C_live−B_live, C_no_live−B_no_live,
B_live−B_no_live and C_live−C_no_live, each with paired outcomes. The difference
between the first two is descriptive, not proof of model internals. Input length
also changes when touch is omitted. If scores do not decrease, report that this
batch did not detect a performance drop, not that the model ignored touch.

Prelaunch validation passed 63 focused tests, scoped Ruff, compileall and diff
checks. The MCP file retains 19 pre-existing Ruff diagnostics; comparison with
the baseline found no new diagnostic from its changed tool description.
GPT-6 low full-history scoped neat found no documentation inconsistency.
The 32-cell batch completed on execution commit `601f02c`, starting from
`7757bea`. There were no replacements, retries, added debugging episodes or
mid-batch implementation changes. All 32 were evaluable, 26 succeeded and six
ended in native early stop. No episode exhausted its budget. All model processes
exited naturally with usage; no recoverable tool error or infrastructure failure
was recorded. Previous rounds remain separate.


### R1.10 results: historical-touch gain appears only under no_live in this batch

| Historical examples | Current touch available | Current touch omitted |
|---|---:|---:|
| B: visual–measured-motion | 7/8 (87.5%) | 5/8 (62.5%) |
| C: same plus historical touch | 7/8 (87.5%) | 7/8 (87.5%) |

Every cell has planned=8, evaluable=8 and infrastructure issues=0. Native
failures for B_live / C_live / B_no_live / C_no_live are 1 / 1 / 3 / 1.

| Fixed comparison | Difference (percentage points) | Only first succeeds | Only second succeeds | Both succeed | Both fail | Unavailable |
|---|---:|---:|---:|---:|---:|---:|
| B_live-B_no_live | +25.0 | 2 | 0 | 5 | 1 | 0 |
| C_live-B_live | +0.0 | 1 | 1 | 6 | 0 | 0 |
| C_live-C_no_live | +0.0 | 1 | 1 | 6 | 0 | 0 |
| C_no_live-B_no_live | +25.0 | 2 | 0 | 5 | 1 | 0 |

The descriptive difference-in-differences is **−25.0 percentage points**:
(C_live−B_live)−(C_no_live−B_no_live). This batch does not support the simple
claim that historical touch mainly improves the use of current touch. With B,
current touch adds two successes; with C, its aggregate difference is zero,
but one seed improves and another worsens. Historical touch adds no net success
under live input and two under no_live. These are limited input-condition
comparisons, not evidence that the model ignored a modality or learned a
particular internal mechanism. Removing touch also changes context length;
fresh resets and one model attempt per cell leave substantial uncertainty.
No new default or controller change follows from these results alone.

Each entry is **native outcome; actual motion requests / control steps**.
Physics steps are twice control steps, and task-body simulation seconds are
control steps / 60. Initialization counts remain separate.

| Seed | B_live | C_live | B_no_live | C_no_live |
|---|---|---|---|---|
| 1000018 | success; 6 / 17 | early stop; 6 / 18 | early stop; 6 / 18 | success; 6 / 16 |
| 1000019 | success; 6 / 17 | success; 6 / 17 | success; 6 / 19 | success; 7 / 17 |
| 1000020 | success; 7 / 17 | success; 7 / 17 | success; 6 / 15 | success; 7 / 19 |
| 1000021 | success; 6 / 18 | success; 6 / 18 | success; 6 / 15 | success; 6 / 18 |
| 1000022 | success; 7 / 16 | success; 6 / 14 | success; 7 / 19 | success; 5 / 14 |
| 1000023 | success; 6 / 14 | success; 6 / 19 | success; 6 / 16 | success; 5 / 18 |
| 1000024 | success; 8 / 20 | success; 5 / 17 | early stop; 6 / 18 | early stop; 6 / 17 |
| 1000025 | early stop; 6 / 17 | success; 6 / 19 | early stop; 5 / 14 | success; 6 / 19 |

The following are totals over eight episodes per condition. Worker wall time
includes startup/shutdown; offline review encoding is excluded. Codex wall time
includes model/tool waiting, not just physics. In particular, 1000022/C_live
waited after demonstrations before continuing and exited normally; its Codex
wall time was 348.8 seconds. Shorter failure is not an efficiency gain.

| Condition | Motion requests | MCP calls | Control / physics steps | Sim seconds | Codex / worker wall seconds |
|---|---:|---:|---:|---:|---:|
| B_live | 52 | 86 | 136 / 272 | 2.267 | 822.1 / 1887.1 |
| C_live | 48 | 82 | 139 / 278 | 2.317 | 1081.2 / 2146.1 |
| B_no_live | 48 | 82 | 134 / 268 | 2.233 | 855.9 / 1906.7 |
| C_no_live | 48 | 81 | 138 / 276 | 2.300 | 806.5 / 1835.5 |

| Condition | Input tokens | Cached input (subset) | Output tokens | Reasoning output (subset) |
|---|---:|---:|---:|---:|
| B_live | 3,625,932 | 3,162,368 | 8,189 | 915 |
| C_live | 3,871,966 | 3,463,296 | 7,677 | 711 |
| B_no_live | 3,195,812 | 2,840,832 | 7,981 | 833 |
| C_no_live | 3,595,349 | 3,247,488 | 7,475 | 643 |

These are actual CLI cumulative fields; cached and reasoning subsets are not
added again. All 32 command records explicitly use GPT-6 `gpt-6-astra` / low,
CLI 0.153.4, and the common frozen prompt. Demonstrations are delivered once
per context: B=12 and C=24 historical images. All current observation returns
contain four images under live and two head/wrist images under no_live;
no_live omits tactile-history selection metadata. Current touch remains in
host recordings, not Agent Saw. Readback passed for all 32 in
`outputs/univtac-isaac51-r110/delivery_validation.json`; full per-cell costs and
paired counts are in `results.json` beside it. Raw recording retains 579
synchronized sample sets, including each takeover frame, without added physics.

### R1.10 viewing guide

[All 32 videos](http://127.0.0.1:9401/r110-videos),
[live/no_live paired views](http://127.0.0.1:9401/r110-pairs),
[actual inputs/actions](http://127.0.0.1:9401/r110-autonomous), and
[historical expert examples](http://127.0.0.1:9401/r110-demonstrations) use the
existing dashboard. All current trajectories explicitly request only world
translations and preserve inherited gripper commands; this does not mean actual
orientation is perfectly constant.

- **1000018:** B_live succeeds and B_no_live fails; C_live fails and C_no_live
  succeeds. The two successful request sequences first explore negative x and
  then move positive-x/downward; failed sequences explore positive x first.
  Watch both B and C pairs, not just one favorable comparison.
- **1000024:** both live conditions succeed and both no_live conditions fail.
  Successful requests again explore negative x first; failed requests explore
  positive x first. These are action descriptions, not a tactile interpretation.
- **1000025:** both B conditions fail and both C conditions succeed, irrespective
  of current touch. It provides a counterpoint to a story requiring current
  touch for historical examples to help.

The same seed does not create identical initial robot states. For example,
1000024's observed initial approach-axis x component was approximately −0.0093
and −0.0102 under B_live/C_live, versus +0.0332 and +0.0348 under the two no_live
runs. These ordinary proprioceptive differences limit single-pair attribution;
no object or robot state was moved to force matching. Requested translation is
not measured translation, and native success is not target arrival. Some final
requests report arm arrival and native early stop together; others reach native
success before the requested target. Both fields remain available in host records and user review; fresh
no-online-feedback runs expose ordinary arm-arrival feedback and neutral
episode termination to the Agent.

Every episode has 1× and 0.05× playback; all recorded task-body durations are
below 0.5 seconds. Sixteen same-seed B/C live/no_live pairs also have 1× and
0.05× versions, on shared simulation time with explicit ended-side frame holds.
No interpolated sensor observations are generated. The no_live tactile panels
are visibly labelled “仅供用户审阅，本episode未送给Agent”.
All 96 playback files reached browser `ended`; receipts are in
`outputs/univtac-isaac51-r110/browser_playback_check.json`. Representative paired
frames were visually inspected, including the no_live warning. This is not a
claim of manually reviewing every frame. Final GPT-6 low full-history scoped neat completed a read-only result/cost and
claim-boundary check; no numerical inconsistency was found.

## R1.11: autonomous Grasp & Classify — completed development pilot

The 18 planned task cells are now natively evaluable: **A 4/6, B 6/6, C 6/6**.
The main comparison **C−B is 0 pp**, with all six pairs successful in both
conditions. B−A and C−A are each +33.3 pp (two method-only successes, four
shared successes). This supports a demonstration benefit in these six
development seeds, not an additional historical-touch success-rate benefit.
It is not a full benchmark or held-out generalization claim. Insert Hole
results remain separate; stable historical-touch benefit and positive
historical/current-touch interaction remain unestablished.

### Fixed method, task information and official examples

Task: pinned Isaac51 `grasp_classify`; seeds 1000026–1000031. A has no examples,
B has official expert visual–measured-motion–outcome examples, C adds only
historical bilateral touch to the same examples. All current inputs retain
head/wrist, bilateral tactile history, proprioception and actual feedback.
Model: explicit `gpt-6-astra` / `model_reasoning_effort="low"`, real Codex CLI
0.153.4, isolated context/memory/history and scoped MCP. Original controller,
retained gripper targets, 30 admitted move requests, 100 MCP calls, 3600 Codex
seconds, native 300 control steps, 80-step segments and terminal grace remain.

The official instruction is: “Touch the cylinders to perceive their surface
texture, classify each object, and place it at the goal region for its class.”
All groups also know one object is active, rough maps to orange, plain/smooth
to green, and static native success tolerances: object origin in its correct
goal frame has strict |x|/|y| < 0.02 m and |z| < 0.01 m; positive-axis dot
product > 0.965. No current class, target transform or target error is supplied.
The task has no special wrong-goal early stop. Its native initialization uses
`use_adaptive_grasp=False`; native friction and grasp setup were preserved.
Class is written only in host finalization for subgroup analysis. Head marking
uses current depth/calibration; wrist geometry remains explicitly unavailable.

Only official `isaac51/grasp_classify` HDF5 0/1 were downloaded, in ascending
ID order. Metadata records success and seed/source_seed 0/1. Class is absent
from metadata; historical active-actor/final-goal data establish 0 as the first
plain and 1 as the first rough success, solely for host selection. No hidden
class/asset-name answer label enters examples. Each has 48 aligned rows,
steps 222–316 at gap 2 and one actual motion segment; all 384 four-view images
decoded. Metadata ends at step 337: the expert's final unsaved delay is not
fabricated as video. Original commands are absent, so examples describe
measured EE/joint motion, not invented OpenETA calls. B has four visual strips;
C adds four tactile strips. The non-tactile JSON and image correspondence match.
The data producer commit and a task-specific formal split remain unrecorded.
The existing dataset entry in [Related Work](related-work.md) covers this reuse.

### Native outcomes and actual operations

Each table entry is native outcome; admitted requests / control steps.
All requests below also caused physical stepping. Success is the native checker,
not classification text, goal naming, command arrival or model self-report.

| Seed | A: no examples | B: visual–motion | C: plus historical touch |
|---|---|---|---|
| 1000026 | success; 3 / 24 | success; 1 / 11 | success; 1 / 9 |
| 1000027 | success; 5 / 39 | success; 3 / 22 | success; 2 / 11 |
| 1000028 | success; 6 / 58 | success; 2 / 12 | success; 1 / 10 |
| 1000029 | step-limit failure; 8 / 300 | success; 3 / 20 | success; 2 / 11 |
| 1000030 | success; 4 / 39 | success; 2 / 17 | success; 2 / 12 |
| 1000031 | step-limit failure; 7 / 300 | success; 2 / 12 | success; 2 / 11 |

Every group has six evaluable results and zero unresolved infrastructure cells.
This does **not** mean no infrastructure attempts failed: three earlier startup
failures are retained below. All 18 Codex processes exited naturally with real
usage. Actual operator-context readback passed: demonstrations delivered once
(0/4/8 images for A/B/C), four current images per observation return, common
frozen prompt, explicit model/effort, and separate control/physics counts.
There were no task-stage tool errors. The recordings contain 936 four-view
sample sets (including takeover), without added physics.

Actual classes were rough on 1000026/28/31 and plain on 1000027/29/30. A succeeded
2/3 within each class; B/C succeeded 3/3 within each. No query class screening or
replacement occurred. Class correctness alone was never scored as task success.
A29 and A31 first requested +65 mm lift with close, then lateral movement with
open. Inspected frames show the cylinder outside the fingers/on the table;
subsequent regrasp requests consumed the native 300-step budget. This is a
specific control sequence and failed recovery, not evidence that the Agent
simply “does not understand touch.” B/C succeeded on both seeds; their shared
success does not isolate touch from the rest of the examples.

### Costs (six evaluable episodes per condition)

Worker wall time includes initialization/cleanup and excludes offline review
encoding. Initialization failures and non-scored launches are listed separately.
Cached input is an input subset and reasoning is an output subset; do not add
them twice. A's larger cost includes two full-budget failures and is not a
matched-success efficiency comparison. C's lower control count does not establish
an extra success-rate gain or a general efficiency mechanism.

| Condition | Requests / MCP | Control / physics | Sim s | Codex / worker wall s | Input / cached subset | Output / reasoning subset |
|---|---:|---:|---:|---:|---:|---:|
| A | 33 / 89 | 760 / 1520 | 12.667 | 915.673 / 1380.357 | 2,347,941 / 2,085,888 | 7,945 / 1,678 |
| B | 13 / 52 | 94 / 188 | 1.567 | 503.452 / 1005.123 | 1,671,581 / 1,455,360 | 4,292 / 715 |
| C | 10 / 52 | 64 / 128 | 1.067 | 515.988 / 1386.798 | 1,853,972 / 1,654,144 | 4,175 / 602 |

### Initialization history and explicit mitigation

There were **23 simulator starts = 18 valid autonomous episodes + three failed
initializations + two non-scored starts**. The non-scored runs were debug999999
(reset/observe/close, zero post-reset steps) and one reset-only timing diagnostic
seed1000028 (240 native initialization physics steps, no Agent/task-body actions).
Neither was an expert collection or formal task result. Debug worker wall time
was 184.981 s; reset-only diagnostic worker wall time was 70.017 s.

| Startup record | Outcome / evidence | Worker wall s |
|---|---|---:|
| B26 original | reset five-step interval 166.903 s > 120; no Codex | 198.161 |
| B26 attempt_2 | unique valid task result, success; original error retained | included in B cost |
| B28 original | reset interval 133.951 s > 120; no Codex | 164.059 |
| reset-only diagnostic B28 | reset returned in 39.703 s; not reproduced, not fixed | 70.017 |
| B28 attempt_2 with timing | reset interval 120.783 s > 120; no Codex | 150.342 |
| B28 attempt_3 with 600 s | unique valid task result, success; both earlier errors retained | included in B cost |

Thus two cells ultimately recovered successfully, through three recovery starts
(one failed). No already-valid Agent episode was retried. Failures were recognized
from `worker_error`/not-ready even where launcher returncode was zero. All three
failed starts cleaned up. Every stop/resume was separately authorized by Pro;
retained instructions and partial reports remain under the R1.11 output root.

Timed B28 failure localized its two long `_step`s to UIPC: whole steps
66.516/54.267 s, UIPC advance+retrieve 66.442/54.196 s, render updates about
0.064/0.065 s. Both 30-second main-thread stacks were at `world.advance()`.
Interval process CPU was 76.204/62.344 s across all threads; a contemporaneous
GPU snapshot showed 95% utilization. This locates the slow call interval and
shows computational activity, but does not identify solver/compilation/contact/
driver internals. Nested spans overlap. The earlier normal diagnostic's
6.223-second UIPC first step was not used as proof of the failed-run cause.

Pro explicitly authorized **only the remaining ten starts** to use native
`cfg.reset_time_limit=600.0`, with the existing 900-second ready deadline.
The first eight valid episodes used the original 120-second native default.
This mid-batch startup configuration difference is disclosed; the batch is not
claimed to have an identical startup configuration throughout. Task body, input,
controller, physics and native outcome rules were unchanged. No pinned source,
global default, cache, driver, solver, sensor or thread configuration was changed.

New copies in `batch/reset_limit_recovery_configs` add only the explicit override;
`native_reset_limit.json` records default/requested/configured/actual Task values.
Timing closes before the same Task enters ready/Codex; no second reset occurs.
The native clock is read after original steps, never replaced. Small logging
wall-time overhead remains. Clock observations distinguish separate five/20/five
reset loops and the marker-calibration clock; whole reset is not a substitute.

**C29 is the actual mitigation witness:** first five-step interval 430.807 s
(over 120, under 600), post-actor 20-step interval 53.719 s, final five-step
interval 0.398 s; marker-calibration global clock 485.604 s; complete reset
507.953 s, then normal ready and native task success. These overlapping clocks
must not be added together. The extra window enabled this initialization to
complete; it did not make UIPC faster or establish a root-cause repair. Other
new recorded intervals remained below 120 seconds, so their normal recovery
alone is not evidence that the override helped them.

### Reproduction and review

Configuration: `configs/univtac/grasp_classify_tactile_icl.yaml`. Main source/data
artifacts: `outputs/univtac-isaac51-r111/{demonstrations,results.json,final_analysis.json}`.
Original order remains in `batch/run_manifest.json`; original partial summary
and failed attempts remain intact. The first recovery, timed recovery and
600-second recovery have separate manifests. Effective cells choose the latest
**authorized** attempt regardless of outcome, never whichever succeeded best.
Execution revisions and their backfill provenance are in `execution_versions.json`.

The original fresh batch used:

```bash
uv run --no-sync python -m scripts.univtac.run_official_tactile_icl \
  --config configs/univtac/grasp_classify_tactile_icl.yaml \
  --demonstrations outputs/univtac-isaac51-r111/demonstrations \
  --output-root outputs/univtac-isaac51-r111/batch
```

The authorized final recovery used the same command with
`--resume-r111-reset-limit`. Its manifest is one-time and already consumed;
these commands document execution, not permission to repeat this completed pilot.
The runner reuses the existing generic worker/MCP and original controller.

[Actual Agent inputs and requests](http://127.0.0.1:9401/r111-autonomous),
[all individual videos](http://127.0.0.1:9401/r111-videos),
[B/C paired playback](http://127.0.0.1:9401/r111-pairs), and
[historical expert/B–C input previews](http://127.0.0.1:9401/r111-demonstrations)
use the existing dashboard. Start with A29/A31 to see release/regrasp failures,
then B/C pairs for those same seeds: both example conditions succeed. The
separate [initialization diagnostic](http://127.0.0.1:9401/artifact?run=univtac-isaac51-r111/reset-diagnostic&path=report.html)
is host-only and excluded from task success.

Validation: 65 focused tests passed; scoped Ruff, compileall and git diff checks passed. All 48 query video files (36 individual 1×/slow videos and 12 paired 1×/0.05× videos) reached browser playback end. Individual slow versions use 0.1× or 0.05× according to duration. This verifies playable delivery, not a human watching every frame; representative failure and success frames were inspected separately. No generated intermediate observations were used. Scoped GPT-6 low neat confirmed the result and cost statements.

### R1.12 Grasp & Classify current/history touch ablation — infrastructure pause

Pro accepted R1.11, including the descriptive secondary result that C used fewer
control steps than B in all six jointly successful pairs (94 versus 64 total).
This does not establish overall lower time/cost: C had more input tokens and
slightly more Codex wall time. The 23-start R1.11 ledger is closed.

R1.12 uses eight new development seeds 1000032–1000039 and four conditions:
B_live/B_no_live share the frozen R1.11 B package; C_live/C_no_live share C,
including C_no_live's historical touch. Only live receives current bilateral
tactile history. All still record four host views with unchanged sensors,
original controller and native task physics (`use_adaptive_grasp=False`).
No_live is omitted Agent input, not a no-contact measurement.

Configuration: `configs/univtac/grasp_classify_current_tactile_ablation.yaml`.
All prompts are identical, with R1.11 rules plus the explicit missing-modality
notice and “when provided” format wording. The official 0/1 demonstrations,
segmentation, image selection and measured-motion interpretation remain frozen.
No task hints, new examples, controller changes or classification quiz are added.

The primary comparison is B_live−B_no_live. Also report C_live−B_live,
C_live−C_no_live and C_no_live−B_no_live, evaluable paired n/outcomes, and the
historical-touch difference-in-differences descriptively. Native success remains
the main outcome. Report all-evaluable costs and jointly-successful paired costs;
missing usage remains unavailable and subsets are not added twice. Final success
does not imply correct first classification, since feedback-driven recovery is
allowed. Initialization/failed-start costs stay separate.

All 32 starts use native reset limit 600 s, outer ready 900 s, cleanup 300 s,
Codex 3600 s including terminal grace 300 s, native 300 control steps,
80 per segment, 30 admitted motion requests and 100 MCP calls. Exactly one
fresh Task reset and Codex context per cell. No debug, warmup, expert or retry
reserve. Any infrastructure error pauses the remaining batch; native failure
continues the frozen order. No UIPC performance diagnosis is added.

Historical prelaunch checkpoint: the following command had not yet run when frozen; execution and pause are recorded below.

```bash
uv run --no-sync python -m scripts.univtac.run_official_tactile_icl \
  --config configs/univtac/grasp_classify_current_tactile_ablation.yaml \
  --demonstrations outputs/univtac-isaac51-r111/demonstrations \
  --output-root outputs/univtac-isaac51-r112/batch
```

Existing replay routes: [R1.12 inputs/results](http://127.0.0.1:9401/r112-autonomous),
[videos](http://127.0.0.1:9401/r112-videos),
[live/no_live pairs](http://127.0.0.1:9401/r112-pairs).
Complete delivery is 64 individual and 32 paired video files. No_live touch is
labelled host-only; missing/unready episodes do not receive fabricated videos.
Implementation/configuration and offline checks precede all physical execution.

Historical prelaunch evidence records 32 passing offline tests, equal B/C non-tactile
content, and 4/8 historical images. Simulator and operator starts are both zero
at this checkpoint; runtime delivery remains unverified.


#### R1.12 infrastructure pause (2026-09-08)

Frozen implementation `7dd93c0` ran ten simulator starts and nine Codex contexts.
The tenth start, seed 1000034 / C_no_live, did not reach ready. The runner recorded
its outer 900-second ready timeout; the worker subsequently preserved the native
reset error at `_base_task.py:478`: second initialization interval 670.778 s >
600 s. The first five-step interval was 250.640 s; complete failed reset took
921.815 s and worker lifecycle 950.301 s. These overlapping durations are not
additive. Native checks occur after a step returns, so the interval can exceed
600 before the exception. Cleanup completed with no SIGTERM/SIGKILL; no Codex or
query action started for this cell. This is not a native task failure.

The remaining 22 cells were not started. No retry or new diagnostic was launched.
All nine evaluable trajectories succeeded and naturally returned model usage:

| Seed | B_live | B_no_live | C_live | C_no_live |
| --- | --- | --- | --- | --- |
| 1000032 | success, 15 steps | success, 10 | success, 11 | success, 9 |
| 1000033 | success, 21 steps | success, 11 | success, 14 | success, 10 |
| 1000034 | not run | success, 10 | not run | initialization failure, unavailable |
| 1000035–1000039 | not run | not run | not run | not run |

Planned denominators stay eight per condition. Evaluable/success counts are
B_live 2/2, B_no_live 3/3, C_live 2/2 and C_no_live 2/2; these are **partial counts,
not completed eight-seed success rates**. Each fixed contrast has only two valid
pairs, both jointly successful, paired difference 0 pp. Full-batch contrasts and
difference-in-differences remain unavailable. No modality conclusion is supported.

Nine actual operator contexts passed model/prompt/once-only demonstration,
current-image and metadata checks. Live delivered four current images; no_live
delivered two and no tactile selection metadata; C_no_live kept eight historical
images. All nine preserved four-view host recording and native control/physics
counts. `results.json` contains all-evaluable costs and joint-success paired
costs with missing values left unavailable. Failed initialization has no model
usage because no model was started. Old R1.11 results remain unchanged.

Delivery at this pause is 18 individual videos and 8 paired videos for the two
complete seeds. No task video is fabricated for the unready cell. Detailed
attempts, timing and actions are in `outputs/univtac-isaac51-r112/partial_analysis.json`.
All 26 delivered video files reached browser playback end (18 individual and
8 paired); this verifies playable delivery, not human inspection of every frame.
Receipts are in `browser_playback_check.json`.

#### Authorized UIPC internal-timer diagnostic (not formal recovery)

After the R1.12 pause, one reset-only grasp_classify/1000034 start is authorized,
with the failed C_no_live configuration and unchanged native600/ready900/cleanup300
limits. It is unscored, has no operator, task-body actions, or additional success-checker calls, and cannot be
reused as a formal attempt. The 9 valid/1 failed/22 unstarted records stay frozen.
Existing outputs contain no native Timer tree. The installed Timer export clears
its window; inclusive nested durations must not be added. UipcSim already enables
Timer (including its existing CUDA timing synchronization); this diagnostic adds
no Timer enable or explicit sync. It exports construction/before/after-step/final
windows and export overhead, covers the native 5/20/5 intervals, and changes only
the existing diagnostic process's UIPC logger from Error to Info. No solver fix
or additional retry is authorized. At the prelaunch checkpoint, runtime internal evidence was pending.


The sole diagnostic completed (implementation `e8999b5`): reset 155.887 s,
243 native initialization/physics steps, zero operator/task-body/additional checker
calls. Native test intervals were 134.529/1.662/0.333 s. Newton scope counts were
1024/1024/640 for the first three steps, 6/6 for steps4/5, then 2–8. Sustained slow
steps were not reproduced; normal completion does not prove a fix. First-step
Newton45.931 s includes line search31.953 s, including trajectory candidates16.918 s
and CCD13.690 s. Both repeated iterations and per-scope cost increased. These
inclusive times are not additive. Info logs confirm max-iteration exits for the
first two steps; native strict_mode0 still returns without proving convergence.

All 62 draining Timer windows exported successfully. Export calls totaled
0.007956 s; total export-and-write overhead was 0.025283 s. Existing Timer CUDA synchronization was unchanged;
Info overhead is unmeasured. Actual r09 binary paths, pyuipc0.9.0/tacex_uipc0.1.0,
effective config and workspace are retained. No specific collision pair or
validated repair was established; no solver/geometry/initialization fix applied.
No extra startup or formal recovery is authorized. Related R1.12 starts total11,
with formal9 valid/1 failed/22 unstarted unchanged. See the
[short diagnostic report](http://127.0.0.1:9401/artifact?run=univtac-isaac51-r112/uipc-diagnostic&path=report.html)
and `outputs/univtac-isaac51-r112/uipc-diagnostic/report.json` for raw references.

#### Goal-pad scene diagnostic (O/K/Z, unscored): completed

The next authorized diagnostic holds grasp_classify/1000034 fixed for nine fresh
processes: O/K/Z, K/Z/O, Z/O/K. O keeps both pads dynamic at z0.002; K changes only
the two pads to native kinematic; Z changes only their initial z to0.010. The
formal default, pinned checkout and other actors remain untouched. These are
scene interventions, not benchmark-equivalent performance fixes. No formal
recovery or candidate adoption is authorized.

Offline USD tetrahedral-boundary inspection finds metre units, identity authored
transforms and collision-surface minimum gap at the source-specified initial pose
(offline, before runtime verification) about+1mm to the UIPC
z0.001 plane (+9mm for Z). This is actual boundary-vertex geometry against a
plane, not just actor origins or AABB overlap. The visible/PhysX table surface is
a different surface; it must not be conflated with the UIPC implicit ground.
Runtime pad transforms, native fixed/contact attributes and surface gaps will
be recorded before reset and after native steps. Existing render-buffer sampling
will record initialization only, with no extra physics/render/sync or Agent.

The common native600/outer900/cleanup300 limits, max_iter1024 and prior Timer/Info
instrumentation remain. Expected initialization timeouts are comparison outcomes:
continue the next predeclared cell only after cleanup. Any unrelated startup,
configuration, CUDA or cleanup error pauses. No replacement attempts. Maximum9
new starts brings related R1.12 total to20; formal9 valid/1 failed/22not_run and
R1.11 remain unchanged. Runtime results were pending at that historical prelaunch checkpoint; final results follow.


All nine fixed starts completed reset; no task-body/checker/model runs occurred.
Related R1.12 starts are now20, while the formal9 valid/1 failed/22not_run ledger
is unchanged. No candidate was adopted and no formal recovery was started.

| Repeat | O reset seconds / first Newton count | K | Z |
| --- | --- | --- | --- |
| 1 | 48.684 / 140 | 47.777 / 87 | 45.712 / 60 |
| 2 | 46.692 / 118 | 45.828 / 59 | 46.955 / 69 |
| 3 | 46.547 / 152 | 120.370 / 1024 | 43.660 / 71 |

K3's second step also required844 scopes; native Info records one max-iteration
exit. K therefore did not reliably remove slow startup computation. Z reduced
first-step iterations in all three paired repetitions, but full-reset time was
not always lower and all O runs were normal. This does not establish a repair
for historical persistent slow steps or stability across seeds. Complete30-step
counts, nested timing and common-step differences remain in the diagnostic CSV.

Native fixed flags were1 only for K; contact element and collision surfaces
remained present. Recorded pad-to-UIPC-ground surface distances stayed positive:
K about1mm, O/Z settling near0.48mm. Z fell after creation; K stayed fixed and its
native target reference z became0.017000m versus about0.016486m for O/Z, with no
compensation. Thus these interventions change scene semantics. After construction,
other prism translations differed by up to0.681mm across conditions despite
unchanged inputs; robot joint arrays matched. Do not claim identical initial state.
Native initialization counters239–242 exclude two earlier construction UIPC
frames, which are separately visible in native logs (total241–244). Neither is
zero motion. Early construction camera frames were not fabricated.

The recorded render buffers are replayed at1× and0.1× simulation time; solver
wall time is shown separately in curves. Raw same-time refreshes are retained,
while video takes the last real image per timestamp. Recording/Info overhead and
resource variation limit precise wall-time attribution; some offline encoding
overlapped the later matrix cells. Twelve scoped tests passed; runtime configs
match across all nine, and all processes cleaned up. See
[goal-pad diagnostic table, curves and initialization replay](http://127.0.0.1:9401/artifact?run=univtac-isaac51-r112/pad-scene-diagnostic&path=report.html)
and `outputs/univtac-isaac51-r112/pad-scene-diagnostic/report.md` for full values.
All18 final videos with unavailable-camera warnings reached browser playback
end; this verifies playback only, not successful capture of dynamic camera views.

Post-matrix media inspection found head/wrist frame IDs stayed1 and sensor time0
throughout all nine runs: the passive `_data` reads did not trigger camera lazy
buffer updates. Only bilateral tactile frames refreshed. The raw camera caches
remain evidence but are not current visual trajectories; delivery masks those
panels as unavailable and preserves actual touch/state/geometry. Complete dynamic
four-view recording was not achieved. This was found after all nine had ended;
no physical restart or retrospective control change is authorized.


### Four-way Insert Hole capacity test — initialization failed

The unscored fixed batch used seeds 1000040–1000043 and the existing successful
C_live input: official instruction/public rules, two official visual/tactile
expert measured-motion demonstrations, and current four-view observations.
The original controller, benchmark, physics and r09 runtime stayed unchanged.
Native reset remained 120 s and outer ready 900 s; the Grasp & Classify 600-second
override was not imported. Four workers started within 0.037 s, but none reached
ready: zero Codex contexts and zero autonomous rollouts completed. This is an
infrastructure capacity failure, not Agent success 0/4.

Across 121 approximately one-second samples, GPU usage peaked at 32,061 MiB with
only 40 MiB free; system MemAvailable reached 22.96 GiB and swap grew from 3 MiB
to about 2 GiB. Seed 1000040 failed a 256 MiB PhysX GPU allocation and could not
create its physics scene; 1000040/1000041 logs explicitly report GPU OOM.
1000042 had entered native pre_move before cancellation. The batch aborted and
all four worker groups were cleaned; batch wall time was 120.53 s. No four-ready
residency interval, autonomous action overlap or evaluable task video exists.
No retry or reduced-concurrency fallback ran within that four-way batch.

The bounded entry point is `uv run --no-sync python scripts/univtac/run_fourway_capacity.py
--output-root outputs/<fresh-dir>`; it defaults to the retained successful C_live
configuration. The all-ready barrier and operator release are capacity-only;
normal single-worker control/observation semantics remain unchanged.
[Capacity report and raw evidence](http://127.0.0.1:9401/artifact?run=univtac-fourway-capacity&path=report.html)
record this attempt. R1.12 remains 9 valid/1 failed initialization/22 not_run;
the proposed D condition and eight-task rollout remain unstarted.


### Two-way Insert Hole capacity test — completed, unscored

A separate fixed batch reused `run_fourway_capacity.py --concurrency 2`
for seeds 1000040/1000041 with C_live and gpt-6-astra/low. Both workers
initialized concurrently, passed the all-ready barrier, and completed autonomous
episodes with native success and natural Codex exit. No infrastructure error
was recorded; both lanes cleaned up. Batch wall time was 385.17 s. Worker
lifetimes were 381.44/369.29 s and Codex lifetimes 100.08/87.98 s.
Each episode used 5 motion requests, 9 tool calls, 17 control steps and
34 physics steps (0.2833 s of task simulation, excluding initialization).

Across 385 resource samples, GPU memory peaked at 20,726 MiB with a minimum
11,376 MiB free; minimum system MemAvailable was 53.33 GiB. Swap started
at 2,147,438,592 bytes (about 2 GiB of pre-existing occupancy), with zero
sampled peak increase, and ended at 2,146,717,696 bytes. Conservative simultaneous
ready residency was 88.028 s, Codex lifetime overlap 87.975 s, and three
pairs of motion execution intervals overlapped for 4.325 s in total.
Each episode recorded 18 samples after existing control/render steps.
Both 1× and 0.05× labelled videos reached browser playback end without error.
Cleanup left GPU usage at 987 MiB and system MemAvailable at 68.96 GiB.

[Two-way capacity report, videos and actual inputs](http://127.0.0.1:9401/artifact?run=univtac-two-way-capacity&path=report.html)
remain separate from the retained four-way OOM evidence. This is an unscored
capacity test, excluded from paper success rates. It establishes only this
Insert Hole two-way run; without a serial control it establishes neither 2×
speedup nor long-term concurrency stability across tasks. D, the eight-task
rollout and the full batch remain unstarted.


### Query evaluation feedback removal — implemented; live acceptance completed

The completed two-way results above used the legacy online-feedback protocol;
they are not acceptance evidence for `native_eval_no_online_task_feedback_v1`.
The fresh-run autonomous, official-ICL and capacity entry points stamp the new
protocol into generated configuration and results. Existing configuration files
are unchanged; explicit historical recovery branches retain their old protocol.
The new protocol removes `check_task` from live UniVTAC registration and Codex
`enabled_tools`; other backends retain their own tool. A query-field allowlist
projects actual MCP returns, including nested execution/observation feedback
and errors. Historical demonstrations retain their recorded success outcome.
Native evaluation/control stopping remains unchanged; voluntary finish is final
and is not an infrastructure failure. Host raw responses are preserved in
`host_tool_trace.jsonl`, evaluator results remain in the usual host artifacts,
and `operator_context.jsonl` records the actual delivered text/images.

53 focused offline tests passed, including native stop timing/priority, no
post-terminal motion, live MCP output/context, tactile strip annotations and
protocol-separated result loading. The existing shared MCP file's unrelated
lint debt was not expanded. `outputs/univtac-no-online-feedback-offline/`
contains read-only projection checks over 18 historical tool responses;
original records and expert outcomes were retained. That implementation/offline
round started no simulator or operator Codex. The separate live acceptance below
subsequently exercised the new protocol. D, the eight-task rollout and the full
batch remain unstarted; R1.12 remains paused under its existing protocol.


### No-online-feedback live acceptance — completed, unscored

A fresh C_live Insert Hole batch used seeds 1000040/1000041, gpt-6-astra/low
and `native_eval_no_online_task_feedback_v1`, reusing
`run_fourway_capacity.py --protocol-smoke --concurrency 2`. This option allows
at most three pre-ready attempts per lane, retains each attempt separately,
and never replaces an accepted ready episode. Both lanes accepted attempt 1:
two simulator starts, two independent Codex contexts, no retries. Each ready
lane proceeded independently; this was protocol acceptance, not another
capacity scan. Batch wall time was 379.09 s.

| Seed | Motions / tools | Control / physics steps | Task simulation s | Codex / worker s | Host native result |
|---|---:|---:|---:|---:|---|
| 1000040 | 7 / 10 | 18 / 36 | 0.3000 | 86.276 / 375.231 | success |
| 1000041 | 4 / 7 | 17 / 34 | 0.2833 | 76.675 / 363.736 | success |

Both actual MCP lists contain only `review_demonstrations`, `observe`,
`mark_point`, `move_to`, `report_issue` and `finish_episode`; `check_task` is
absent. All 17 returns matched host response projection and actual
`operator_context` delivery. Historical demonstrations retained their outcomes
and delivered 24 images each; query observations and motion returns delivered
four images each. Ordinary execution feedback remained available. Both
automatic endings delivered neutral feedback, and both Codex finals explicitly
acknowledged that tools had not supplied task success.

Protocol delivery and the autonomous chain passed independently of host task
outcomes. The only actual post-terminal tool was `finish_episode`, without
additional physics. Post-terminal motion rejection and unobserved failure/error
branches remain offline-test coverage. The focused suite passed 55 tests.
Finals, usage and host results were saved; worker and Codex process trees were
empty after cleanup. The two 1× and two 0.05× videos all reached browser playback
end without error, from 19/18 real samples; final score labels are user-only.

Lightweight sampling recorded GPU peak use 20,951 MiB, minimum GPU free memory
11,151 MiB and minimum MemAvailable 55.26 GiB. Pre-existing swap was
2,143,350,784 bytes with zero sampled peak increase. Cleanup left GPU use at
988 MiB. No benchmark, controller motion, physics, demonstration or budget
change was made.

[New-protocol report, videos and actual inputs](http://127.0.0.1:9401/artifact?run=univtac-no-online-feedback-live-smoke&path=report.html)
remain separate from legacy two-way/four-way evidence and the paused R1.12
ledger. These two episodes do not enter paper success rates and establish
neither old/new protocol equivalence nor absence of performance loss. D,
eight-task rollout and the full batch remain unstarted.


### Eight-task new-protocol coverage — completed

`configs/univtac/eight_task_coverage.yaml` fixes eight new cells at seed
1000040: Insert Hole D and one C cell each for Grasp & Classify, Insert Tube,
Insert HDMI, Pull Out Key, Lift Bottle, Lift Can and Put Bottle in Shelf.
The completed new-protocol Insert Hole C episode is referenced without rerunning
it. All cells retain `native_eval_no_online_task_feedback_v1` and
`gpt-6-astra` / low.

Offline preparation completed for all eight cells. The 32 A/B/C/D demonstration
returns passed actual MCP image decoding and pixel comparisons. Existing Insert
Hole and Grasp & Classify packages were copied unchanged, with D added separately.
The six new task packages preserve recorded action-atom segments rather than
forcing three segments per example. Official episodes 0/1 are used throughout;
Shelf episode 1 records source seed 2, which is retained in provenance.

The runner reuses the existing Coordinator and episode runner, with two slots
including cleanup, at most three pre-ready initialization attempts per cell,
and limits of eight operator Codex starts and 24 simulator starts. Native task
control limits remain 600 for HDMI, 500 for Lift Bottle and 300 for the others.
The preparation suite passed 55 focused offline tests. The implementation was
committed and pushed as `415f56d` before the fixed queue started. R1.12 remains
paused; no full batch was resumed.

Prepare with `uv run --no-sync python scripts/univtac/run_eight_task_coverage.py
--phase prepare`, then run the frozen queue with the same command and
`--phase run`. Artifacts use `outputs/univtac-eight-task-coverage/`; persistent
attempt directories retain the initialization budget across invocations.

The frozen queue completed all eight new autonomous cells with eight operator
Codex processes and nine simulator starts in 1616.16 wall seconds. Lift Can
required one preserved pre-ready initialization retry: native reset timed out
after 153.77 s against its unchanged 120 s limit. Attempt 2 was accepted. No
accepted episode was rerun. All eight accepted episodes were evaluable; all
nine worker lifecycles report complete cleanup.

| New cell (seed 1000040) | Native outcome / ending | Motion requests / physical motions | Tools | Control / physics steps | Simulation s | Codex wall s | Input / output tokens |
|---|---|---:|---:|---:|---:|---:|---:|
| Insert Hole D | success | 5 / 5 | 8 | 14 / 28 | 0.2333 | 117.18 | 345379 / 754 |
| Grasp & Classify C | success | 2 / 2 | 7 | 14 / 28 | 0.2333 | 121.73 | 269383 / 621 |
| Insert Tube C | early stop, failure | 6 / 6 | 9 | 12 / 24 | 0.2000 | 97.23 | 509914 / 976 |
| Insert HDMI C | success | 8 / 7 | 16 | 13 / 26 | 0.2167 | 150.18 | 870747 / 1634 |
| Pull Out Key C | success | 5 / 5 | 8 | 62 / 124 | 1.0333 | 99.98 | 413336 / 898 |
| Lift Bottle C | early stop, failure | 24 / 24 | 37 | 377 / 754 | 6.2833 | 876.82 | 3613599 / 4240 |
| Lift Can C | early stop, failure | 2 / 2 | 6 | 22 / 44 | 0.3667 | 130.63 | 460397 / 593 |
| Put Bottle in Shelf C | voluntary finish, failure | 9 / 9 | 16 | 299 / 598 | 4.9833 | 389.17 | 1137735 / 2326 |

All eight Codex processes exited naturally with final text and actual usage.
Cached input and reasoning output remain subsets in raw usage; they are not
added again to the input/output totals above. The 107 actual tool responses
matched host projection and recorded operator context. No `check_task` was
registered. Historical expert outcomes remained visible, current-query scoring
remained host-only, and all actual post-terminal calls preserved physics counts.
Unobserved terminal/error branches remain covered by offline tests.

Sampled GPU usage peaked at 20807 MiB; minimum MemAvailable was 53.28 GiB,
with no sampled swap increase. GPU use after cleanup was 908 MiB. The failed
initialization is retained separately from four unsuccessful task outcomes.
The prior Insert Hole C episode is a separate coverage reference, not a ninth
new result. This one-seed coverage check establishes neither ICL gains nor
multi-seed reliability.

[Eight-task report, real inputs and 1×/0.05× videos](http://127.0.0.1:9401/artifact?run=univtac-eight-task-coverage&path=report.html)
uses real recorded frames; slow playback adds display time, not observations.
Host outcome labels are user-only. No benchmark, motion algorithm, physical
parameter, solver setting or success threshold changed.


### Four-task C: 1/2/4-shot — current dispatch and historical preparation

The current C-only overlays are `dispatch_conditions: [C]` and
`dispatch_max_initialization_attempts: 1` in
`configs/univtac/shot_scaling.yaml`. Apply it offline with
`uv run --no-sync python scripts/univtac/run_shot_scaling.py --phase revise-scope`.
Do not rerun prepare or clear the old output. The original manifest and effective
episode configurations remain unchanged; `scope_revision.json` selects C in
its original relative order. The pre-revision result/runtime/page snapshots are
retained under `pre_c_only_scope/`. This operation does not release the current
service-error pause or launch any simulator/model.

The user has explicitly authorized recovery. The existing `--phase run` command
dispatches only pending C cells. Completed C episodes, including incomplete
model final/usage, and Lift Can C_1shot's exhausted three attempts are skipped.
For new dispatch, each cell has one total initialization start, not one retry.
A cleaned ordinary initialization failure becomes unavailable without another
start. Historical attempt_2/3 results and exhausted attempts remain unchanged;
recovery scans them before applying the new start limit. The old active queue
is drained before restarting with this policy; in-flight cells keep their
original execution version. Unexpected infrastructure-error pause rules remain.
B results remain queryable; unrun B cells carry `deferred_to_main_table` scope
metadata while their original not_run state is preserved. The C report contains
one curve per task, equal-task means, same-seed 2−1/4−2 pairs, costs and missing
states. Historical B is separate, not a completed B/C comparison.

The verified revision ledger is 1200 C planned = 4 evaluable + 1 initialization
unavailable + 1195 not_run; B has 6 completed and 1194 deferred cells. No new
starts occurred for this revision. WebSocket, automatic HTTPS fallback/reconnect,
service-error pause policy, CLI, model, prompt, control and budgets are unchanged.
The user subsequently released the pause. Resume only pending C cells and
monitor/report every 30 minutes after launch; the service-error policy remains
unchanged. The scope-edit check itself consumed no simulator or model starts.

The following records describe the original preparation, before scope reduction.

The design-only update was pushed as `c99e39e` before implementation. The user
then changed the shared query list to 1000000–1000099 before any new simulator
or operator started. `configs/univtac/shot_scaling.yaml` references the full
`configs/univtac/main_query_seeds.json` list. Four-task coverage records at seed
1000040 overlap this range and remain disclosed separately, not reused as this
shot experiment's cells. Expert source seeds do not overlap the query range.

Use `uv run --no-sync python scripts/univtac/run_shot_scaling.py --phase download`,
then `--phase prepare`. The immutable `manifest.json` under
`outputs/univtac-shot-scaling/` contains all 2400 cell identities, ordered expert
IDs/source seeds, effective configurations and one-shot assignment. Preparation
refuses to overwrite an existing frozen manifest. The original 0/1 exports are
shared unchanged; only official episodes 2/3 are newly processed. Lift Bottle
episode 3 has source seed 4.

After the implementation is committed and pushed, use the same command with
`--phase run`; this is also the resumption command. It reconstructs all accepted
completed states before dispatch, including unsuccessful tasks, and never
restarts them. A crash after native finalization and complete worker/model
cleanup can be reconciled from those persisted records without a new rollout.
Unresolved accepted attempts and persisted delivery failures pause dispatch.
At original preparation, attempt directories retained a three-attempt budget;
the current one-attempt dispatch overlay supersedes that policy for new starts.
An interrupted run cannot erase a delivery error or turn it into an evaluable
result. Each accepted episode records its actual execution commit.

Simulator slots include cleanup but exclude offline review encoding. One
independent media worker produces 0.05× review videos with task, seed, condition
and shot labels and checks full decoding/frame counts. Raw frames are retained;
media failures can be rebuilt without rerunning completed tasks. No 1× videos
are required. Shared expert images use absolute source references in historical
MCP context; current query image path rules remain unchanged. A 5 GiB free-disk
reserve stops new dispatch and defers media rather than deleting evidence.

`results.json` always includes every planned cell. The existing dashboard's
artifact route serves `report.html`, cell review pages, actual operator context,
host results and slow videos. `summarize_shot_scaling.py` reports planned,
evaluable, successes and unavailable separately; Wilson intervals condition on
evaluability. The original B/C report supported C−B; the current C-only report retains
same-seed 2−1/4−2 comparisons and effective pair
counts and paired empirical bootstrap intervals, including their possible
degeneracy. Use the existing r09 Python with `summarize_shot_scaling.py
outputs/univtac-shot-scaling --plots` to render the four task curves; no new
plotting dependency is installed. Cached input/reasoning remain subset fields.

This entry describes the prepared execution path, not completed experiments.
No simulator or operation Codex has started at this preparation checkpoint.
Future Pro reports use https://chatgpt.com/c/6aa00ac2-bef8-83ee-9a7d-82186e056c59.

Preparation completed with all 2400 frozen cells, 100 shared seeds and 32 actual
MCP B/C image-return checks (one-shot checks both balanced expert assignments).
All four old 0/1 text projections are unchanged; eight new HDF5 files were
downloaded. Four-shot C delivers 92/144/192/64 images for Tube/Can/Bottle/Key
respectively, without dropping segments or reducing resolution. The focused
suite passed 63 tests; Ruff and compile checks passed. The native/model path
for these larger inputs still requires the planned formal episodes, not an
extra model pretest. The preparation checkpoint consumed zero simulator and
zero operator Codex starts.


### Shot scaling pause checkpoint — 2026-09-09

Execution used the pushed `9bac8b5` baseline. The fixed 2400-cell matrix is
**not complete**: 10 accepted episodes are native-evaluable, one cell exhausted
its three pre-ready attempts, and 2389 cells remain not_run. All touched cells
use query seed 1000000. There were 13 simulator starts and 10 operator Codex
processes, with no accepted episode rerun.

| Task | Condition | Native result / termination | Control steps | Model ending |
|---|---|---|---:|---|
| Insert Tube | B_1shot | failure / early stop | 18 | natural |
| Insert Tube | C_1shot | success | 22 | natural, after reconnection |
| Insert Tube | B_2shot | failure / early stop | 13 | terminal grace expired |
| Lift Can | C_1shot | initialization unavailable after 3 attempts | — | no Codex |
| Lift Can | B_2shot | success | 46 | natural |
| Lift Bottle | B_2shot | failure / voluntary finish | 472 | natural |
| Lift Bottle | C_2shot | failure / early stop | 301 | natural |
| Lift Bottle | B_4shot | success | 73 | terminal grace expired |
| Pull Out Key | C_2shot | failure / early stop | 50 | natural |
| Pull Out Key | B_4shot | failure / early stop | 58 | natural |
| Pull Out Key | C_4shot | success | 58 | terminal grace expired |

The pause was triggered by recurring Codex WebSocket disconnections across
independent contexts: logs include broken-pipe errors, retries through 5/5 and
automatic fallback to HTTPS. The CLI did recover and continue operation, so
these messages do not establish permanent service unavailability, a 4-shot
capacity failure or a native task-failure cause. Dispatch was paused under the
user's recurring-service-error rule; accepted episodes kept their original
budgets and finalized. Seven models exited naturally; three reached the
existing terminal-grace limit. Their missing final/usage is not replaced with
zero. Native outcomes and model/transport status remain separate.

The first dispatch pause was followed by an actual resumption after connection
recovery: seven completed identities were skipped and only new identities
started. The second pause is the current state; the older
`pause_resolution.json` describes only that first recovery. Do not interpret it
as authorization to resume now. No execution configuration changed during
either session. All 13 worker lifecycles report complete cleanup.

All ten actual deliveries passed host/context projection and expert-count
checks, including four-shot B for Bottle (96 images) and four-shot C for Key
(64 images). This does not establish live delivery of the unrun 192-image
Bottle C_4shot package. All ten slow videos passed complete decoding/frame-count
checks; representative browser checks are indexed separately. Native terminal
feedback remained neutral and no `check_task` was exposed.

Sampled GPU use peaked at 20847 MiB, minimum GPU free was 11254 MiB, minimum
MemAvailable was 48.00 GiB, and swap did not increase. Minimum sampled disk free
was 640.00 GiB. The old resource log is retained (about 1 GiB); a recording-only
follow-up limits future per-second phase entries to active lanes instead of
repeating all 2400 pending cells. It does not alter control, inputs or budgets.

[Paused matrix, partial curves, raw results and slow videos](http://127.0.0.1:9401/artifact?run=univtac-shot-scaling&path=report.html)
include every planned state. There are too few evaluated seeds or matched
conditions to select K or infer shot/tactile gains. The incomplete curves are
not a completed 100-seed result. Preserve all completed and unavailable cells
when the authorized queue is resumed; do not regrant attempts or rerun them.
No A, D/E, main-table remainder, other-model comparison or old R1.12 recovery
was started.

### C-only authorized recovery — native startup failure

After the user's explicit pause release, the C-only service started two new
workers at seed 1000000. Lift Can C_2shot attempt_1 exited before ready with
SIGSEGV (returncode -11, 22.04 s); the native stack includes libomniclient.so.
Insert Tube C_2shot attempt_1 was cancelled because of that batch failure
(SIGTERM, returncode -15, 26.27 s), not a second independent crash. Neither
reached ready or launched Codex. Both process groups completed cleanup.

The updated ledger is C 1200 = 4 completed/evaluable + 1 initialization
unavailable + 2 infrastructure_issue + 1193 not_run. B retains six completed
results and 1194 deferred pending cells. Total starts are 15 simulator and
10 operator Codex, with all 15 worker cleanups complete. Each new failed or
cancelled attempt remains counted; no accepted episode or exhausted cell was
rerun. The library named in the stack is not an established root cause.

The existing non-retryable initialization/interface-error branch stopped
dispatch. No retry-policy, connection, CLI, benchmark or runtime modification
was applied. A single evidence report was sent to the current Pro conversation
for advice; it must be read back before any ambiguous resend. The user-level
service `univtac-shot-c-only.service` is inactive at this checkpoint.
Provider job `job_db8ebae6e3de4d20` checks and reports every 1800 seconds;
it preserves scope and attempts and does not silently bypass this error.

Pro message `8070380f-699e-45aa-b208-6012acbdf5a6` subsequently authorized
remaining-attempt recovery without investigating or changing the native library.
The host now recognizes only a cleaned, pre-ready/no-Codex native SIGSEGV with
the observed fatal omniClientFreeContent stack. The peer cancellation requires
the explicit saved authorization for this attempt and retains its original
cancellation reason. Both current cells continue at attempt_2, with attempt_3
as their final possible initialization try. The same classifier is used by
live handling and recovery readback. An eligible local failure does not cancel
a normal peer. Three such native crashes without any intervening worker ready
pause dispatch; the existing chronological event ledger preserves this count
across restarts. Cancelled attempts do not count as native crashes.

The focused offline suite passed 22 tests, including third-crash stopping,
peer isolation, accepted/exhausted-cell preservation and C-only order.
Readback of the two actual failed attempts confirmed the narrow classifications
and a current crash streak of one. No additional simulator/model was used for
these checks. Existing runtime, WebSocket/HTTPS behavior, model, task inputs and
budgets remain unchanged. Restart the same service once after commit/push,
with Restart=no; do not create another queue or rerun prepare.

### Seed 1000005 startup pause — 2026-09-09

Insert Tube C_4shot attempt_1 exited before ready with SIGABRT (returncode -6,
18.01 s); Lift Bottle C_1shot attempt_1 was cancelled by the batch (host SIGTERM,
returncode 0, 28.37 s). Neither launched Codex. All 75 worker starts have complete
cleanup. C remains 52 evaluable (12 successes), eight initialization unavailable,
two infrastructure issues and 1138 not_run; historical B remains six completed.
All 52 C videos passed decoding checks. No failed identity will be replaced.

Pro message `79a54d66-91fb-4a0f-a49d-dd2826296502` is recorded in per-attempt
reviewed markers. Both recovery paths preserve the original error and skip these
reviewed, cleaned, pre-ready issues without creating another attempt. They remain
`infrastructure_issue`, not completed or native failures. Unknown handoff state,
incomplete cleanup and delivery errors still block recovery. The focused suite
passed 25 tests; Ruff and diff checks passed.

The single local read-only check recorded an inotify instance limit of 128 and
165 inotify FD references for UID 1000. Shared/inherited descriptors can duplicate
instances; this is not 165 distinct instances. There were 3376 watch references
against a 65536 watch limit, six read races/permission failures, and service soft
and hard LimitNOFILE of 1048576. System file-nr was 59008; the system maximum was
9223372036854775807. Failure logs place inotify warnings before the kvdb warning
and the final std::system_error, but provide neither specific errno nor a kvdb
lock path/holder. The historical SIGABRT root cause remains unknown.

Available inotify instance capacity remains unconfirmed, so dispatch stays paused
under Pro's resource-check condition. No sysctl, cache, lock, runtime, connection,
model or budget was changed; the check started zero simulator/operator processes.
The earlier three-attempt recovery is historical. New dispatch remains limited
to one total initialization. The 30-minute monitor must not treat the old drain
handover instructions as permission to release this new resource-related pause.
