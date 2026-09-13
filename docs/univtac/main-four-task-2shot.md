# Four-task ABC main-table first part

## Authorized design — 2026-09-13

Status: first-pass A/B queue running on both hosts (launch2026-09-13 23:05–23:06 +08:00). The latest
user amendment **excludes all historical A/B reuse**, overriding the earlier
reuse paragraph. This campaign starts800 new A/B identities and references400
selected historical C2 identities. Valid results produced within this new
campaign are locked and reused on recovery, including failures. Old campaigns
are never restarted. No C, other four tasks, D/E, model comparisons, shot sweep,
R1.12 or expert replay is authorized here.

Uniform K=2 is selected for B/C: fixed official successful expert episodes0/1,
in that order. A has no examples, condition=A, shot=0, expert_ids=[] (never A2).
This is an overall performance/cost/comparability choice, not statistical proof
that2-shot is optimal. Preserve all1/2/4-shot results and do not select K by task.
The four tasks participated in configuration selection and the query seed range
overlaps development records; this is not an independent held-out test.

| Task | A planned | B2 planned | C2 historical success/evaluable |
|---|---:|---:|---:|
| insert_tube |100|100|32/100|
| lift_can |100|100|2/100|
| lift_bottle |100|100|36/100|
| pull_out_key |100|100|26/100|
| Total |400|400|96/400|

Use the unchanged `configs/univtac/main_query_seeds.json`,1000000–1000099.
A/B target count800 is neither an attempt cap nor a quota of successes.
C2 references come from `outputs/univtac-shot-scaling-analysis/selected_results.json`,
retaining full cell_key, host, phase, attempts/path, native outcome, termination
and costs. No C1/C4 enters the ABC table. C2 comprises356 first-pass and44
supplement results; preserve their actual initialization policies. Old C and
new A/B are non-contemporaneous, and same seed does not guarantee identical
physical initial state. Do not rerun C to hide this limitation.

## Frozen inputs and execution

Each task's A/B prompt is identical to its saved C2 prompt, including public
rules, tools, coordinates and budgets. A uses the same review_demonstrations
tool returning no_demo with empty examples/images, without expert summaries.
B uses the existing expert0/1 visual–measured-motion projection; C's shared
non-tactile content, segmentation, head/wrist frames/times, robot states and
historical outcomes are unchanged. Remove only historical touch/derived content
from B. Native MCP Image blocks deliver actual pixels once per context.
Current head/wrist, bilateral segment-end touch history, robot/gripper state and
ordinary execution feedback remain available in all conditions. No findings
such as close-first or small-pull enter prompts/controllers.

Use gpt-6-astra/low, existing CLI version/connection configuration, original
controller, dual-finger hold, TCP semantics, r09 and pinned Isaac51. No dependency,
benchmark, physics, solver, scoring or system-parameter changes. Protocol remains
native_eval_no_online_task_feedback_v1: no check_task, current score/reward,
hidden target error or specific native terminal reason. Neutral episode_ended,
ordinary tracking feedback, budgets, static public rules and historical outcomes
remain. Native latching/early-stop/priorities are unchanged; no physics after end.

Task body: Tube/Can/Key300 control steps, Bottle500;80 steps per segment,
30 admitted non-preview requests,100 MCP calls,3600 seconds Codex total wall
(including at most300 terminal grace),300 seconds cleanup. Physics120Hz,
control60Hz,decimation2; model waiting never advances physics. Reset/pre_move
remain official; initialization steps/costs are separate.

Every new A/B attempt uses the already implemented disable_initialization_timeout
host setting: native reset, outer ready and worker startup deadlines disabled.
Keep native default/requested/effective limit records, and retain task/cleanup
limits. Initialization does not consume the Codex3600 seconds. No early120/600s
initialization restriction is reintroduced. Context/workspace/CODEX_HOME and
access isolation remain per attempt.

## Identity, ordering and missing results

Campaign root: `outputs/univtac-main-four-task-2shot/` on each host. The manifest
has800 executable A/B cells; the main-table index separately includes400 C2
references. `reused_results.json` records C2 only and zero old A/B reuse;
`remaining_AB.json` records new A/B identities. Full keys include task, seed,
condition, shot, demo set and protocol, not seed alone.

Construct400 pairs in task order Tube/Can/Bottle/Key, seeds ascending, then
shuffle once with random.Random(20260913). For task_index0..3 and i=seed−1000000,
assign local when(i+task_index) is even, otherwise hzz-server. Within a pair,
A then B when(floor(i/2)+task_index) is even, otherwise B then A. Filter shuffled
pairs by host and expand; save lists and never reshuffle on recovery. Because
old A/B reuse is disallowed, historical-host preference does not apply.
Each machine has at most2 slots including initialization, operation and cleanup;
no all-ready barrier or capacity pretrial. Ports/output/UIPC are isolated,
expert media shared read-only. Merge only the assigned host's A/B record.

First pass attempts each missing identity once. Known pre-takeover reset failure
is preserved and fully cleaned, then queued for a later missing-result round,
not immediate repeated occupation of a slot. After the pass drains, retry only
identities with no valid native result, in frozen order, using fresh attempt
directories and unchanged config. First valid result permanently locks identity.
Native success/early-stop/step-limit, agent_finish failure and evaluable operation
budget/overall-deadline failure all lock. Reconnect, missing final/usage or media
failure do not invalidate such a result. Explicit post-takeover infrastructure
loss is eligible only if native result is actually unavailable; preserve the
interruption. Unknown adapter/state/scoring issues pause for review, never retry
to conceal them. Do not infer validity solely from exit code or model prose.

Capacity or explicit terminal provider errors stop dispatch on the affected host,
allow the other live slot its original budget and cleanup, and persist paused.
User authorization is required to resume; never switch models or probe/restart
in a loop. OOM, leakage, mixed data and failed cleanup retain hard-stop behavior.
Local weekly remaining<=2% triggers safe drain and persistent quota pause;
unknown quota is reported unknown. Read quota at launch and each30-minute check.
No old recovery authorization clears a quota pause. Services use Restart=no.

## Evidence, media, monitoring and completion

Retain per-attempt configuration, prompt, actual CLI command/version, delivered
operator_context, host/native records, requests/zero-step/physical events, four
raw image streams, lifecycles, usage availability and initialization/model/cleanup
costs. Unknown times are null; do not derive event times from file mtime.
Encode only0.05x slow review from original frames, with bounded independent
media work and persistent pending items. Media-only never starts physics/model;
repair existing frames without rerunning valid episodes. Decode/frame counts
cover all selected media; browser/keyframe spot checks are separately labelled.

Progress uses existing report/episode-page conventions and records A/B pending,
initializing, operating, cleaning, missing-result, paused and completed states;
C2 remains separately labelled historical. A new scoped30-minute status job may
read progress/resources/weekly quota and repair offline media, but never restore
paused services. Old C-only monitoring remains disabled.

Before launch: verify800 unique new A/B plus400 C2, fixed C counts, exact shared
prompts/projections/native image delivery, current-touch/query-score boundary,
initialization versus operation/cleanup limits, valid-failure recovery, no
capacity-induced sibling cancellation, disjoint hosts and cleanup-owned slots.
Use existing offline tests/artifacts only; scoped neat, focused commit and normal
push precede launch. Record both execution versions. Preserve unrelated user
untracked `heldout_grounding_skill.py`.

Primary analysis is C2−B2; B2−A and C2−A are secondary. Report per task
success/evaluable/planned, missing identities and infrastructure attempts, four
paired outcomes and unpaired count. Reuse paired bootstrap/exact McNemar; Holm
family has four task-level C−B tests for this stage. Future eight-task comparisons
will use their full family, without selecting significant tasks. Costs include
usage coverage without double-counting cached/reasoning subsets. Keep all valid
results in the main table; phase/policy/exit-type sensitivities do not replace it.
Can retains its native predicate, not a stable-suspended-lift interpretation.

End when all targets have valid results and required media, or a clear condition
requires user handling. Save exact remaining identities and attempt/start/cleanup
ledgers; focused commit/push and report status. No next campaign starts automatically.

## Preparation evidence

Preparation produced800 unique new A/B cells (400/host) and400 C2 references,
with unchanged C2 counts32/2/36/26. All400 saved C2 prompts and delivered
projections were checked on their owning hosts. A/B prompts match exactly.
Offline MCP image decoding matched original pixels: B/C image counts are
Tube22/44, Can36/72, Bottle48/96, Key16/32; A has0 images. These are per-task
existing projections, not a universal image count.

Implementation extends `run_shot_supplement.py` for `phase=main_AB`, using the
same two-slot Coordinator and episode runner; it adds durable host pause,
frozen-order missing-result rounds, and one active media encoding with pending
paths retained on disk. Per-host results live in `hosts/<host>/results.json`;
`main_four_task.py` prepares/validates/reports the matrix, not a second execution
framework. SIGINT requests safe drain; a persisted dispatch_pause.json prevents
restart until explicitly released by the user.

Local read-only quota checks at22:41 and22:53 +08:00 report67% weekly remaining.
Both old rollout campaigns were inactive during preparation. Resource snapshots
are in the campaign working notes. No new simulator/model has started at this
preparation checkpoint. Focused offline checks cover identity/order, recovered
valid failures, missing usage, known reset retries versus unknown errors,
round ordering, persistent pause and capacity drain; live state is recorded
separately after the authorized deployment.

Prelaunch review tightened unknown pre-ready/unclean failure handling to global
hard-stop (provider pause still drains), and marks delivery-audit issues for
review without rerunning locked native outcomes. Native evaluability is retained
separately; such issues withhold paired-analysis acceptance until reviewed.

Validation:40 focused queue/manifest/delivery/capacity/protocol tests and35
existing autonomous-operation/tactile-ICL tests passed across the scoped runs.
The final new queue tests include an unknown pre-ready hard-stop case. No
simulator debug, model canary or expert replay was used.

## Launch and continuation

After focused commit/push and identical execution-code verification, hzz started
at23:05 and local at23:06 +08:00 on2026-09-13, service
`univtac-main-four-task-ab`, Restart=no, two slots each. Remote synchronization
used a new Git bundle because its origin points to an older local bundle; no
branch rollback or force update occurred. Both operator CLIs are0.153.4.
The local first service-creation command failed before spawning any experiment
because of an environment-argument format; corrected arguments created exactly
one new local queue. Launch quota at23:06 was66% remaining.

The first remote Key1000070 A/B2 pair produced two valid native early-stop
failures, both cleaned with usage and passing slow media. The queue proceeded
to Key1000082 without retrying either failure. Local Bottle1000036 A/B2 reached
actual GPT-6 low operation. Real native-reset records show default120seconds,
requested override=null, disabled=true and effective/actual limit=null; the
operation and cleanup limits remain separately recorded. Current counts are in
`launch_acceptance.json` and host results, not frozen in this narrative.

New provider monitor `job_0e7e212ad2594ec1` checks every1800seconds; its first
scheduled check is23:38:45 +08:00. The old C-only job stays disabled. The monitor
never resumes a paused service. The existing9399 read-only dashboard service
was refreshed to load current campaign-path support; the campaign report now
passes an HTTP content check. This is not a browser/video-watching claim.
[Progress page](http://127.0.0.1:9399/artifact?run=univtac-main-four-task-2shot&path=report.html)
uses the existing dashboard. Per-host JSON remains authoritative; remote mirrors
are snapshots taken at their recorded update times. The full800-result and final
ABC/media acceptance remains in progress.

## Latest monitoring checkpoint — 2026-09-13 23:47 +08:00

Owner-host results merged by full cell key contain30/800 completed evaluable
A/B identities and4 native successes: local11/400 with2 successes, hzz19/400
with2 successes. There are770 identities without a locked result:4 in progress
(local1 initializing +1 operating; hzz2 operating),766 never started, and0
finished missing-result/blocked identities. Both services are running with
Restart=no and neither host is paused. Historical C2 remains96/400, read-only.
These are incomplete progress counts, not a final treatment-effect conclusion.

The independent process/lifecycle inspection at23:42 verified31 simulator starts,
30 Codex starts and27 completed cleanups across the hosts; all27 completed
attempt media checks passed. The later23:47 report contains29 passing media
statuses and1 pending, which remains with the existing bounded encoder. No
media-only job or rollout restart was needed. Browser/keyframe inspection was
not performed during this check. Do not confuse these two observation times.

At23:42:11 the live account-only weekly quota request returned no weekly value:
**current quota unknown**; the previous65% value was not substituted. No quota
pause was inferred. The23:42 resource samples show local25.03GiB available RAM,
20739/32607MiB GPU memory used and755GiB free disk; hzz95.32GiB available RAM,
20327/32607MiB GPU memory used and1596GiB free disk. No new service, cleanup
or infrastructure anomaly was found in the inspected evidence.

Evidence: campaign-root `latest_monitor_snapshot.json`, `monitor_quota.json`,
`remaining_AB.json`, `statistics.json`, and each owner host's `results.json`,
`events.jsonl`, `resources.jsonl` and attempt lifecycle/media checks. The remote
owner ledger was mirrored only into `hosts/hzz-server`; local and historical C
records were not overwritten. The new1800-second monitor remains enabled,
next scheduled00:08:45 +08:00 on2026-09-14; the old C-only monitor remains disabled.
