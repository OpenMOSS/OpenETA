# Four-task C-only shot scaling: results and analysis

Updated 2026-09-13. This is the campaign's durable result and analysis record.
The [research plan](research-plan.md) owns the research question and next design;
[campaign history](shot-scaling-history.md) preserves preparation and recovery
checkpoints. Historical commands are not permission to resume completed work.

## Completed population and result table

The population is four UniVTAC-Isaac51 tasks × C × 1/2/4-shot × query seeds
1000000–1000099, with 1200 unique original `cell_key` identities. C contains
historical vision and tactile examples; current vision, bilateral touch and
proprioception remain available. The model is GPT-6 low, with the
`native_eval_no_online_task_feedback_v1` protocol. Native evaluation remains
host-owned; Agent prose is not the score.

The frozen first pass ended with 1086 evaluable results (252 successes),
104 initialization-unavailable cells, six infrastructure issues and four
cancellations. A separate authorized supplement supplied the remaining
114 evaluable results (10 successes). It removed native-reset, ready and total
worker-startup deadlines; operation and cleanup budgets stayed unchanged.
Only missing-result identities were eligible for repeated attempts. An existing
valid native failure was not rerun to obtain success, and missing Agent final
text/usage alone did not make a valid native result eligible for supplementation.

**Combined: 1200 evaluable identities, 262 native successes (21.83%).** This is
a completed-identity population across two initialization policies, not a
1200-cell single-attempt experiment under the original initialization limit.

| Task | 1-shot | 2-shot | 4-shot | All shots |
|---|---:|---:|---:|---:|
| Insert Tube | 22/100 (22%) | 32/100 (32%) | 23/100 (23%) | 77/300 (25.67%) |
| Lift Can | 2/100 (2%) | 2/100 (2%) | 3/100 (3%) | 7/300 (2.33%) |
| Lift Bottle | 31/100 (31%) | 36/100 (36%) | 2/100 (2%) | 69/300 (23.00%) |
| Pull Out Key | 35/100 (35%) | 26/100 (26%) | 48/100 (48%) | 109/300 (36.33%) |
| All tasks | 90/400 (22.50%) | 96/400 (24.00%) | 76/400 (19.00%) | 262/1200 (21.83%) |

All denominators above are evaluable identities. The number contributed by the
supplement, in 1/2/4-shot order, is Tube 7/6/6, Can 26/31/18, Bottle 5/3/1,
and Key 2/4/5. Report these selection differences with any comparison.
The first-pass success fraction among evaluable results is 252/1086 (23.20%);
252/1200 (21.00%) uses all planned first-pass identities, including missing
results. Neither is the combined 262/1200 statistic.

All 1200 selected episode media records passed decoding/frame-count checks,
and all selected worker lifecycle records report complete cleanup. This is
not human inspection of every video. Offline supplement `--media-only`
regenerated missing slow videos/pages without running physics or restarting
rollout; old `media_error.json` files remain historical evidence. A later
passing `media_check.json` is the media acceptance record.
The 30-minute provider monitor is disabled (rechecked 2026-09-13).
No main-table run was launched and no uniform K has been selected.

## Execution changes retained with the result

The first continuation assigned 720 pending cells equally across the two hosts,
leaving 480 historical terminal cells on local. A later explicit authorization
moved 115 never-started local cells to hzz, producing the final 245/475
ownership; started/terminal cells were not moved. Old drain/resume snapshots
are historical and must not trigger another allocation or restart.

The user waived the earlier undelivered Pro consultation after local network
recovery and authorized sealing Tube1000056/C_1shot and
Bottle1000056/C_2shot in the first pass. Later supplement authorization applied
only after all first-pass identities were terminal. The remote inotify setting
was separately raised with user authorization to 1024 instances / 524288 watches;
old 128/65536 warning snapshots are not its current-setting evidence and do not
authorize another sysctl change. No such setting was changed in this analysis.
The local weekly-remaining <=2% safe-drain rule remains part of the recorded
execution policy; an old recovery authorization does not release a quota pause.

## Evidence locations and reconstruction

| Host | Repository |
|---|---|
| local | `/home/ubuntu/wybcode/TACTILE-for agent/worktrees/openeta-univtac-main` |
| hzz-server (SSH) | `/media/user/B29202FA9202C2B91/univtac-experiments/OpenETA-UniVTAC` |

Both repositories retain first-pass `outputs/univtac-shot-scaling/` and
supplement `outputs/univtac-shot-scaling-supplement/`. Raw ledgers, failed
attempts and recordings were not overwritten. Final first-pass ownership is
local 245 and hzz 475, plus 480 historical local identities. The earlier
360/360 assignment is superseded; its snapshot is
`two_host_assignment.before_second_split.json`. The supplement manifest fixes
57 identities per host in its separate phase.

Reconstruction uses the latest first-pass `two_host_assignment.json`:

1. For each assigned identity, take the row from its owning host's first-pass
   `results.json`; take identities outside the 720-cell assignment from local.
2. Select the 1086 rows whose `episode.evaluable` is true. Add the 114 disjoint
   evaluable supplement identities, preserving host, phase and attempt provenance.
3. Count `episode.task_success`, grouping by task and shot. Never count remote
   unowned `not_run` rows as globally pending, and never merge by seed alone.
4. Check selected `media_check.json` and `worker_lifecycle.json` records. Keep
   `codex_exit` and other termination values as recorded; do not rewrite them
   as native early stops or infer zero cost when usage is absent.

The derived local analysis directory is
[`outputs/univtac-shot-scaling-analysis/`](../../outputs/univtac-shot-scaling-analysis/):

- [`selected_results.json`](../../outputs/univtac-shot-scaling-analysis/selected_results.json)
  contains the 1200 selected keys, phase, host, attempt counts, exact episode
  paths, native outcome/termination, counters and acceptance flags;
- [`summary.json`](../../outputs/univtac-shot-scaling-analysis/summary.json)
  records all task/shot counts, first-pass status, supplement contribution,
  action means and Bottle/Key paired outcomes, regenerated on 2026-09-13;
- [`sample_agent_messages.json`](../../outputs/univtac-shot-scaling-analysis/sample_agent_messages.json)
  preserves the 48 selected episodes' Agent messages with original JSONL line
  numbers, native labels, phases and host paths.

The retained offline reconstruction command, from the local repository, is:

```bash
python3 outputs/univtac-shot-scaling-analysis/rebuild.py
```

It reads both hosts over SSH and writes only the derived `selected_results.json`
and `summary.json`. It verifies disjoint ownership, identity coverage and saved
media/cleanup acceptance; it starts no simulator or model. The message sample
is a separate retained extraction described below. These derived files and the
local reconstruction helper are durable local artifacts, excluded from Git;
the tracked tables, merge rules and evidence paths in this document remain
the portable summary. A clone alone does not contain the raw experiment data.

Each `episode_path` points to the selected attempt directory. It contains
`episode.json`, `final_result.json`, `host_evaluator.json`, lifecycle records,
`codex_exec.jsonl`, `operator_context.jsonl`, `host_tool_trace.jsonl`,
`tool_trace.jsonl`, `media_check.json`, `review.mp4` and `review.html` as
applicable. Actual delivered input, Agent messages, tool actions, host scores
and user-only video overlays are distinct sources.

## 2-shot versus 4-shot: quantitative observations

The following census covers all 400 selected Bottle/Key episodes, 100 per row.
Means use all outcomes, not only successes. `move_request_count` counts requests;
it is not interchangeable with `actual_motion_requests` or executed distance.

| Task / shot | Native success | Early stop | Agent finish | Codex exit | Step limit | Mean move requests | Mean control steps | Mean tool calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Bottle / 2 | 36 | 18 | 33 | 12 | 1 | 11.31 | 258.94 | 17.41 |
| Bottle / 4 | 2 | 34 | 46 | 2 | 16 | 15.62 | 369.68 | 29.58 |
| Key / 2 | 26 | 72 | 0 | 2 | 0 | 5.59 | 54.33 | 8.54 |
| Key / 4 | 48 | 50 | 0 | 2 | 0 | 5.86 | 55.66 | 8.78 |

| Same-seed pair | 2 succeeds / 4 fails | 2 fails / 4 succeeds | Both succeed | Both fail |
|---|---:|---:|---:|---:|
| Bottle | 35 | 1 | 1 | 63 |
| Key | 10 | 32 | 16 | 42 |

Bottle's 4-shot runs have lower observed success, more control steps and more
step-limit endings. Key's 4-shot runs have higher observed success with similar
mean action/step counts. These are task-dependent observed associations. One
model attempt per accepted episode, fresh-reset variability and the separate
supplement policy limit causal interpretation. No significance test was
performed in this analysis. The C-only campaign does not isolate historical
touch from other demonstration content and does not establish a tactile gain.

## Agent messages and actions: exploratory sample

The 2026-09-13 analysis read 48 episodes, each seed paired across 2/4-shot:

| Task | Selected seeds | Episodes / selection |
|---|---|---|
| Tube | 1000000, 1000001, 1000002, 1000003 | 8; first four seeds |
| Bottle | 1000000, 1000001, 1000004, 1000006, 1000082, 1000091 | 12; first two seeds per paired outcome class, all when fewer |
| Key | 1000000, 1000001, 1000002, 1000003, 1000004, 1000005, 1000017, 1000027 | 16; first two seeds per paired outcome class |
| Can | 1000002, 1000030, 1000045, 1000057, 1000071, 1000078 | 12; all five 2/4-shot success identities, their same-seed counterparts, plus one both-failure pair |

This deterministic sample is not random, exhaustive, or a frequency estimate
for language patterns. No videos/images were visually inspected in this text
analysis. In the references below, L/H identify the repositories above; R is
`outputs/univtac-shot-scaling/cells`, S is
`outputs/univtac-shot-scaling-supplement/cells`. A reference ending `:N` names
line N of `attempt_1/codex_exec.jsonl` under the specified cell directory.
The retained sample JSON resolves all paths and line numbers.

**Bottle.** Sampled 4-shot failures repeatedly describe empty closure,
repositioning and slip during rotation. L/R/lift_bottle/1000004/C_4shot:25
says “The fingers closed almost fully without visible tactile contact”; :72
says “The bottle slipped back onto the plane during the turn.” That episode
ends at 500 control steps. Its 2-shot counterpart reports sustained contact
at :17 and succeeds in 95 steps. For seed1000006, 4-shot describes missed
grasp (:37) and slip during rotation (:58), then ends voluntarily; 2-shot
succeeds. However, regrasp language is not unique to failure:
H/R/lift_bottle/1000091/C_4shot:46 plans a higher body regrasp and ultimately
succeeds, and seed1000082 succeeds in both shot groups after extended recovery.
The aggregate data support longer 4-shot operation on this task; the sample
does not establish that additional demonstrations caused a specific mistake.

**Key.** Both shot groups often describe tactile deformation or resistance,
then easing rotation and pulling vertically. L/R/pull_out_key/1000002/C_2shot:15
and C_4shot:19 describe this same sequence, but only 4-shot succeeds (46 versus
47 control steps). Similar wording accompanies both outcomes in other pairs.
No stable verbal strategy branch separating success from failure was identified.
Seed1000027/C_4shot instead ends on `codex_exit` near its 3600-second wall
budget after only three move requests; it should not be assigned the same
touch-strategy explanation as an ordinary early stop.

**Tube.** Both groups mention tactile changes and corrective operations.
L/R/insert_tube/1000001/C_2shot:17 describes deformation and orientation
correction; :34 says lateral adjustment reduced distortion. The trace includes
retreat and lateral correction before success. Its 4-shot counterpart also
reacts to increased deformation and retreats, but fails. Seed1000003 succeeds
in both groups after corrections, while seeds1000000/2 fail despite contact
descriptions and adjustment attempts. This sample does not support saying
4-shot ignored touch. Timing and correction magnitude remain candidate
explanations requiring broader action-level analysis.

**Can: native labels and model conclusions disagree.** All five selected
native-success identities in the complete 2/4-shot success set have negative
Agent conclusions about establishing a stable lift:

| Native-success identity | Agent-message reference | Agent conclusion excerpt |
|---|---|---|
| 1000057 / 2-shot | H/R/lift_can/1000057/C_2shot:63 | “a stable lift without slippage was not established” |
| 1000078 / 2-shot | H/S/lift_can/1000078/C_2shot:32 | “did not establish a stable grasp or vertical lift” |
| 1000030 / 4-shot | L/R/lift_can/1000030/C_4shot:20 | “A stable lift without slippage was not achieved” |
| 1000045 / 4-shot | H/R/lift_can/1000045/C_4shot:67 | “A stable lift without slippage was not established” |
| 1000071 / 4-shot | H/R/lift_can/1000071/C_4shot:34 | “rotated and lost contact” |

For seed1000030/C_4shot, `episode.json` records `task_success=true` and
`termination=native_success`; `host_evaluator.json` also records positive
native evaluation. The score is retained. Under no-online-feedback, Agent
final text is not a readback of that score. The discrepancy warrants checking
native event timing and video; it does not yet establish an evaluator bug,
stable-grasp success, or a model perception error. The opposite discrepancy
also occurs: Can1000045/C_2shot claims visible lift in its message but ends
as an Agent-finish failure. Do not substitute prose for native labels.

## Conclusions and proposed next investigations

Confirmed: shot-count differences vary by task; Bottle's 4-shot failures are
longer on average, while Key improves without a distinct sampled verbal
strategy. Sampled messages alone cannot explain why these outcomes differ.
The Can mismatch is a specific evidence discrepancy, not an established cause.

The following are **proposed, not executed** investigations, with no new
physical experiments launched or authorized by this document:

| Question / hypothesis | Comparison and retained evidence | Decision criterion / limitation |
|---|---|---|
| Does Bottle 4-shot alter grasp/rotation order before failure? | All same-seed 2/4 pairs; code grasp, first lift, rotation, release and regrasp from actual tool traces, with raw-video spot checks; keep native outcome and phase separate | Determine whether the candidate sequence recurs across the full population; message wording alone is insufficient and association is not causality |
| Does Key differ in rotation/backoff angle or pull timing despite similar prose? | Extract requested and executed rotation, reversal and lift timing from the same 100 pairs; retain initial-state differences | Check whether those action features distinguish discordant pairs without treating identical seed as identical reset |
| Why do Can success labels disagree with final messages? | Inspect all five success pairs' native latch time, evaluator rules, actual gripper/object trajectory and recorded video | Separate a legitimate native threshold event from model misdescription or a reproducible scoring issue; do not relabel without evidence |
| Is Tube recovery timing different across shot groups? | Extend beyond four sampled pairs; align tactile observations with retreat, orientation and lateral correction requests | Test an explicit timing/amplitude pattern; do not infer tactile causation from a mention of deformation |

These are offline evidence analyses, not a frozen next experiment matrix.
Before any new experiment, record its hypothesis, controls, fixed seed/source
population, model/protocol, action/time budgets, initialization policy, primary
metric and execution authorization in the research plan. Uniform K and
main-table reuse of supplement results remain undecided. Preserve C-only
scope and the distinction between first-pass and supplement evidence.
