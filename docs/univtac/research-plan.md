# UniVTAC tactile-agent research plan

This is the canonical research-plan and status entry point for
`tactile-agent-for-univtac`, updated on 2026-09-13 against the current checkout
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

## Paper design and completed shot-selection experiment

The planned main table is eight tasks × A/B/C × 100 fixed query seeds (2400
planned cells). A has no historical examples; B has expert visual observations,
measured motion, proprioception, time and outcomes; C adds historical touch to
exactly B. All current inputs retain vision, bilateral tactile history, robot
state and execution feedback. B/C now use exactly two fixed official successful expert episodes0/1;
A has no demonstrations and shot=0.

Before that main table, the completed C-only campaign covers Insert Tube, Lift Can,
Lift Bottle and Pull Out Key × C × 1/2/4-shot × 100 seeds: 1200 planned
cell identities, completed through the first pass and a separate missing-result supplement. This supersedes the original B/C dispatch scope;
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

The user selected uniform K=2 across tasks and B/C. Preserve all shot curves
and disclose configuration selection and development-seed overlap. The latest
four-task main-table authorization requires800 **new** A/B identities, with no
historical A/B reuse, and400 read-only selected C2 references including failures.
Historical C2 first-pass/supplement provenance remains explicit. The detailed
[current four-task A/B contract](main-four-task-2shot.md) supersedes earlier
reuse, initialization and launch restrictions for this bounded campaign only.

D (historical touch without external vision), E (expert motion records only),
current-touch ablations and sol/luna/terra model comparisons are deferred until
after the main table, with scope to be decided later. No tactile mismatch
experiment is authorized here. LEMMo-Plan remains a closest related-work
comparison; tactile ICL itself is not claimed as a first proposal. This design
update does not open a new literature survey.

The shot experiment keeps gpt-6-astra/low, the no-online-task-feedback protocol,
original control, pinned Isaac51 and each task's accepted public rules/budgets.
Each host used two simulator slots including cleanup. The first-pass dispatch
allowed one total initialization attempt per new cell; historical attempts remain
preserved. The separately authorized supplement disabled native reset, ready and
worker startup deadlines and allowed new attempts only for identities lacking
a valid result. It retained operation/cleanup budgets and kept all attempts in
a separate output root. Valid native failures were not rerun to seek success. The old
2400 Codex / 7200 simulator ceilings are historical upper bounds, not an
authorization to dispatch B or replenish any exhausted cell. Model-started episodes are
not initialization retries. Fixed expert media are shared read-only. Only slow
review videos are required by the latest user amendment; bounded offline media
processing must not hold up simulator/operator dispatch. Preserve every raw
recording; media can be rebuilt without replaying physics. The full main table
and all other ablations remain unstarted and unauthorized by this round.

## Completed C-only checkpoint — 2026-09-13

The four-task C-only campaign is complete: **1200 unique evaluable identities,
262 native successes (21.83%)**. The frozen first pass contributes 1086 evaluable
results and 252 successes; the independent supplement contributes 114 and 10.
The combined table is not a 1200-cell single-attempt experiment under one
initialization policy. All selected media checks and worker cleanups passed;
media checks establish decoding/frame agreement, not human inspection of every video.
The old C-only 30-minute monitor is disabled. At this C-only completion
checkpoint, no main-table experiment had started; the later authorized A/B
campaign is recorded in the active-campaign section below.
On2026-09-13, the user selected uniform **K=2** after the offline analysis.
This configuration decision authorizes no new rollout by itself.

The offline 2-shot/4-shot analysis now covers all800 selected episodes, plus
Can100 C1 episodes,8747 execution events,60 fixed-case keyframe reviews and16
expert content overviews. Bottle/Can C4 frequently starts by repositioning with
an open gripper, unlike C2's predominantly closing-first behavior. Key retains
similar high-level plans but uses small first pulls more often; Tube shows no
single correction rule separating outcomes. These are observed associations,
not isolated causal effects of K or historical touch. The image review is
unblinded and model-assisted, not human or full-video inspection.

| Record | Purpose |
|---|---|
| [Shot-scaling results and analysis](shot-scaling-results.md) | Final table, merge population, sampled messages, evidence locations, open questions and proposed follow-up |
| [Shot-scaling history](shot-scaling-history.md) | Retained preparation, pauses and recovery checkpoints; historical instructions only |
| [Development history R1.4–R1.9](development-history-r14-r19.md) | General-tool development, expert curation and early ICL pilots |
| [Development history R1.10 onward](development-history-r110-onward.md) | Modality pilots, initialization/capacity diagnostics and new-protocol acceptance |

Current decisions and next designs belong here; update the relevant campaign
record after every experimental or analytical step. Keep observed results,
interpretations and unexecuted proposals distinct. Raw attempts and media stay
in their original output roots, referenced from the campaign record.

## Uniform K decision and next design

The two-round 2026-09-13 Pro consultation is complete. Local replication confirms
Bottle4−2 = −34pp and Key4−2 = +22pp, including phase/termination sensitivities
and post-hoc paired tests. All57 `codex_exit` cases are operator overall-deadline
endings. Pro reviewed eight Can event cases and found the native predicate's
semantics explain the apparent disagreement with stable-lift self-reports;
original scores remain unchanged. See [the campaign analysis](shot-scaling-results.md#pro-analysis-consultation--2026-09-13)
for methods, complete replies, local verification, costs and evidence limits.

The proposed offline800/Can300/60-case/16-expert inspection is completed in
[the full behavior analysis](shot-scaling-results.md#full-offline-behavior-analysis--2026-09-13),
with full-key features, per-case annotations, original frames and source traces
under `outputs/univtac-shot-scaling-analysis/behavior/`. Added expert examples
preserve the major routes; they do not demonstrate the opening-first Bottle/Can
route. Example identity/content and count change together, so context length,
example content/order and downstream feedback are still not causally separated.
P0 missing exact timestamps/per-step object poses remain unknown and need no
historical rerun. The next experiment choice should address those distinctions,
not rerun valid failures until success. The user's subsequent instruction,
“发给pro，并且支持2—shot”, selects uniform **2-shot** for subsequent comparisons
across tasks and B/C, without choosing a best K separately for each task.
This is an experimental tradeoff: observed macro success is24.0% versus22.5%
for1-shot and19.0% for4-shot, with heterogeneous task effects. The2−1 paired
interval includes zero and1-shot has lower observed cost;2-shot is not a proven
statistical optimum. At that consultation checkpoint, main-table initialization/missing-result policy
and reuse of prior supplement data still required a design decision. Those
policies are now fixed for the bounded campaign linked below. No main table, new
rollout or change of native score/model/controller/budget is authorized by the
K decision. The conditional Bottle24-episode proposal remains unapproved.

The completed third Pro reply recommends returning to the core C2−B2 comparison:
match expert0/1 and common inputs, retain current touch in both, and vary only
historical tactile content. It proposes filling matching B2 for the existing
four-task400 C2 results and interleaving B/C on remaining tasks. This proposal
is not an execution authorization or approval of historical-result reuse;
those policies were undecided at that checkpoint and are now fixed by the
bounded campaign authorization below.
See the campaign record for the full reply and locally checked wording fixes.

## Active authorized campaign: four-task A/B completion

The2026-09-13 user instruction authorizes preparing, offline-testing, committing/
pushing and launching only800 new A/B identities for Tube/Can/Bottle/Key, then
limited missing-result rounds. Its final amendment excludes all old A/B reuse.
A is shot0; B2 and read-only C2 use official experts0/1. The400 C2 references
remain96/400 successes. New A/B initialization disables startup limits while
retaining task and cleanup budgets; first valid outcomes lock, including failures.
The full identity, pair ordering, pause/quota, input, media and analysis contract
is [main-four-task-2shot.md](main-four-task-2shot.md). That record owns current
implementation/launch state; no C or other-task run is authorized. Previous
Pro proposals and historical statements about unspecified policies are superseded
only by this explicit bounded authorization.
