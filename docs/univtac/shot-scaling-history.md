# Shot-scaling preparation and recovery history

These are historical checkpoints, not current execution state or authorization to resume.
See the [research plan](research-plan.md) for the current scope and
[shot-scaling results](shot-scaling-results.md) for the completed campaign.
Old pause/recovery instructions below apply only to their recorded checkpoints.

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

### Fixed two-host continuation

The remaining 720 C cells are assigned once: 360 local and 360 on hzz-server,
30 per task/shot per host. `--assignment` and `--host` filter dispatch only,
preserving order and all previously accepted results. Remote paths are relocated
without changing demonstration content or experimental settings. Host reports are
local ledgers, not global totals; merge disjoint results by cell identity.
Assignment does not establish launch; actual services and records do.

### Reviewed hzz provider-capacity pause

hzz-server seed1000057 Tube C_1shot stopped on provider model capacity;
Bottle C_2shot was peer-cancelled. Pro reviewed both records. Cleaned accepted
operator issues use a per-attempt reviewed marker to skip without retry,
preserving original errors and costs. These remain infrastructure_issue and
cancelled respectively, not native task failures.

### Provider-capacity drain handling

Exact terminal model-capacity errors now pause new dispatch without cancelling
the existing peer. Existing lanes retain their original budgets and cleanup.
Other infrastructure and cleanup failures still hard-abort. Twenty focused
offline tests passed. Historical results are not reclassified or retried.

Pro reviewed one local recovery after a 30-minute cooldown and four cleaned
issue markers. Local markers are saved; remote marker confirmation is blocked
by SSH connectivity, so local recovery has not yet been launched. The original
local assignment retains 235 pending cells. hzz has 360 terminal cells and must
not restart. Another capacity failure drains and pauses without automatic retry.
The independent weekly <=2% pause remains active.

### Independent missing-result supplement

The user authorized a separate campaign after the first pass reached 1200
terminal cells: 1086 evaluable (252 successes), 104 initialization unavailable,
six infrastructure issues and four cancellations. All original records remain
unchanged. The supplement assigns the 114 missing-result identities 57 per host,
with new attempt directories under univtac-shot-scaling-supplement. Native reset,
ready and worker startup total deadlines are disabled only by the supplement
configuration. Operator and cleanup budgets remain unchanged. Clean ordinary
initialization failures can repeat; a valid native failure is a result and is
not retried. Hard infrastructure failures pause for review. Provider capacity
drains peers without automatic service restart. Media remains asynchronous.
The 23 focused offline checks passed before launch; live acceptance is pending.
