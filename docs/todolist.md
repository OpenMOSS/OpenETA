# OpenETA TODO List

This list tracks the near-term OpenETA agent work derived from the RFC.

## Approved bundle / IK / perception direction — 2026-09-07

User approved implementation of typed bundle handoffs, stronger IK search with
accumulated failure evidence, and separate grasp-refinement/active-perception
contracts. See [implementation and remaining scope](typed-bundle-and-ik-recovery-2026-09-07.md).

- [x] First local bundle slice: immutable registered manifests, Python inspection,
  candidate-to-compile / target-to-IK / IK-to-move alternatives, existing live
  provenance and safety gates retained; wrist candidate handoffs reuse this path.
- [x] Shared IK defaults 64 starts / 1000 evaluations per start / 30 seconds;
  completed search without a solution is unknown, not proved unreachable;
  add repeated-search evidence scoped by state, target and solver configuration.
- [x] Explicitly distinguish existing alignment and viewpoint quality semantics;
  neither quality assessment grants motion authority.
- [ ] Three-person review of new public alternatives, outputs and checker meaning;
  historical authority/catalog approval hashes remain unchanged.
- [ ] Finish the perception/selection side of the unified manifest lifecycle,
  including existing native grasp/placement bundles; do not claim every ID now
  maps to the new JSON manifest format.
- [ ] Implement compiled-grasp-free active perception, weak localization and
  post-motion projection; keep object identity distinct from graspable parts.
- [ ] Benchmark known-reachable LIBERO poses and actual candidate geometries;
  larger search budgets alone do not establish adequate convergence coverage.
- [ ] Fresh standard Luna batch performance validation. Restore historical
  **per-task** budgets: 15M cumulative tokens / 160 turns / 320 tools / 3 hours.
  Prepared `evaluations/manifests/libero_object_task0_bundle_formal_seed0.json`;
  do not compare 500k-token diagnostics directly with historical success rates.

Local evidence for this slice: last full suite 2379 passed / 25 skipped /
one historical authority/catalog failure; final bundle checks 19 passed;
LIBERO-Python synthetic numerical checks 5 passed. No paid performance run or
new simulator was launched in that initial slice. These do not satisfy the open live acceptance items.

Staged follow-up (user-approved): [bundle-only Stage 1 experiment](diagnostics/bundle-stage1-luna-2026-09-07.md).
Host-selected `bundle_stage1` makes compile and single-step move bundle-only;
IK/placement/probe/trajectory migration remains partial. Default stable profile
is unchanged; no silent native-reference fallback. New local suite: 2385 passed,
25 skipped, one existing authority failure. Standard Luna batch with restored
historical budget was deliberately interrupted after reproducing contradictory
IK feedback (29 main requests, 18 tool calls, no motion); it is not a performance
sample. Environment and owned processes were closed and verified.
- [x] Locally correct IK failed-search feedback: a best numerical iterate near a
  joint limit no longer acquires prose claiming a feasible endpoint/executable seed.
  Fresh live validation of the correction remains pending.
  Post-repair full suite: 2387 passed / 25 skipped / one existing authority failure.
- [ ] Finalize interrupted standard batches durably: environment cancellation
  closes correctly, but CLI currently omits result.json and leaves rollout status
  active. Preserve partial usage/termination/cleanup records without claiming task success.
- [ ] Finish staged migration after real batch evidence; do not expose dual
  reference branches as the final simplified interface.

Stages 2/3 follow-up (user authorized continuous implementation and experiments):
[migration scope](bundle-interface-stages23-2026-09-07.md) and
[standard Luna batch record](diagnostics/bundle-stages23-luna-2026-09-07.md).
- [x] Opt-in `bundle_stage2`: bundle-only IK, explicit target-authoring tool
  preserving all five target sources, and target manifests for alignment,
  camera-to-world placement poses and attachment-probe waypoints.
- [x] Opt-in `bundle_stage3`: ordered IK trajectory manifests, SAM3 result
  selection/rejection manifests and wrappers around ready native grasp/placement
  inputs. Existing live provenance, identity and motion checks remain authoritative.
- [x] Project actionable context/history hints to the selected interface;
  validate the same public schema before native Host resolution, without a
  silent legacy fallback. Default stable profile remains unchanged.
- [x] Execute the two fresh sequential standard Luna/Object 0 experiments;
  inspect actual provider contracts, bundle usage, IK feedback and trusted reward.
  Initial run `tmp/batch-luna-stages23-MJYJuG/` was interrupted in stage 2
  after 36 requests / 1,444,644 known main tokens / no motion: recency-only
  handoffs hid the still-current candidate manifest. Stage 3 had not started.
  Owned environment/processes closed. Fresh post-repair manifest:
  `tmp/batch-luna-stages23-discovery-tv2f3f/manifest.json`, same historical budgets.
  [Post-repair experiment record](diagnostics/bundle-stages23-discovery-luna-2026-09-07.md).
  Both fresh episodes have now ended with `planner_validation_failed`, no motion,
  confirmed cleanup, 250722 known main tokens combined. Neither exhausted its
  task budget. Execution of the experiments is complete; live acceptance of
  the migrated chains remains open (they stopped before selection/grasp/IK).
- [x] Keep current typed sources discoverable across long pose/IK histories;
  preserve target association in IK summaries and reject native/whitespace IDs
  at every migrated schema boundary. Focused checks 58 passed; live recheck pending.
  Post-repair full suite: 2414 passed / 25 skipped / one existing authority failure.
- [x] Clarify experimental repair feedback: required versus optional fields,
  omit unsupported nulls; list legal response names. No admission relaxation or
  automatic response repair. Post-run focused regression 182 passed; paid efficacy
  untested.
  Final full suite: **2415 passed / 25 skipped / one existing authority failure**,
  `tmp/bundle-stages23-handoff-tests.xml` / `.log`; `git diff --check` passed.
- [ ] Investigate initial semantic misidentification independently of IK:
  current runs confuse red/green tomato sauce with blue alphabet soup despite
  different MolmoPoint evidence. Do not inject simulator oracle labels into Agent
  context or treat identity continuity as proof of initial semantic correctness.
- [ ] Retain bounded diagnostic payload/status/usage for invalid grasp-advisor
  outputs; current generic error cannot distinguish the invalid decision value
  or faithfully account for failed-advisor token usage.
- [ ] Decide stable promotion / legacy removal after live evidence and required
  review. These slices do not migrate every perception parameter or implement
  compiled-grasp-free active perception.

Stages 2/3 local suite: **2404 passed / 25 skipped / one existing reviewed-authority
failure**. Catalog audit confirms generated projections current and no structural
issues, but authority canaries still fail; neither full suite nor audit is green.

User cost constraint (2026-09-07): subsequent paid Agent regressions use
`gpt-5.6-luna` only. Do not launch sol/astra comparisons or automatically
upgrade models on failure without renewed user approval. This supersedes
earlier cross-model comparison suggestions. The already-started sol run
ended and its owned simulator was cleaned up; see the Object 0 diagnostic.

User-confirmed regression entry: use the standard `agent.cli.batch_eval`,
not the restricted perception diagnostic as a substitute. The first fresh
standard Luna/Object 0 run completed observe/SAM3/selection/AnyGrasp/compile,
then exhausted motion-preparation validation with zero dispatched motion.
See [standard batch evidence](diagnostics/standard-batch-luna-object0-2026-09-07.md).

User implementation authority: directly fix clear defects while pursuing the
full goal; pause for material human design decisions. The target remains a
robust, usable refactored harness, not merely passing local unit tests.

- [ ] Align public planner validation feedback with the published grasp bundle
  contract. Standard batch exposed missing bundle_id falling into legacy
  rgb/depth/mask/intrinsics requirements, despite the Agent schema only exposing
  bundle_id/backend_preference. Preserve Host internal provenance checks;
  do not repair by instructing the Agent to synthesize raw paths. The branch
  exists in current HEAD too, so this is not proven to be a new refactor regression.
  - Local repair implemented: public validation now requires the published bundle
    reference and rejects raw-input overrides; Host resolution/gates are unchanged.
    Malformed main-planner responses also receive bounded untrusted raw-text repair
    context, without automatic syntax repair or execution. Targeted 210 tests pass;
    merged full regression: 2359 passed, 21 skipped, one existing authority failure.
    A fresh standard batch reached five infeasible IK results and stopped at
    its token threshold (529225 known main tokens vs 500000 limit), with no
    motion and confirmed cleanup. Neither new feedback branch was exercised
    in that run; full task success and live efficacy remain unproven. See
    [repair boundaries](diagnostics/planner-feedback-repairs-2026-09-07.md).
- [ ] Audit SAM3 point/text mutual exclusions in the published schema against
  planner/runtime gates. Standard Luna batch supplied prompt with point mode;
  do not assume a field inventory fully communicates cross-field constraints.
- [ ] Validate isolated grasp advisor candidate-ID fidelity and clarify
  bundle/result/compiled/IK reference handoffs in the actual model request.
  Standard batch produced an unknown advisor candidate and repeated parameter
  aliases; strict rejection prevented motion. Single-run evidence does not
  establish root cause or performance improvement.

Primary execution path: closed-loop `tool_call`.
Optional path: bounded `code_policy` as an atomic-tool backend only.

Refactoring is active on `dev/huaizezheng/harness-refactor-2026-09-05`.
See the [implementation and validation log](harness-refactor-progress-2026-09-05.md)
for the dirty-worktree baseline, batch plan, partial fixes, and unverified scope.

Current delivery scope (user-confirmed 2026-09-06):
[resume real task experiments](experiment-ready-milestone-2026-09-06.md).
This milestone takes precedence over the historical batch order below; it does
not require stable end-to-end Spatial 0 / Long 9 success or close the full backlog.

Live evidence (2026-09-07): independent current-code RAG/cuRobo + OSC short
canary and Codex-operated Human VLM observation smoke completed with explicit
cleanup. See [receipts and limits](diagnostics/experiment-ready-live-2026-09-07.md).
This is not an Object 0 pick-place regression or whole-path collision proof;
the existing-success regression and Spatial 0 / Long 9 diagnostics remain open.
Follow-up: CLI and Human VLM launchers now explicitly pin the selected simulator
URL, reject arbitrary MCP fallback and retain the original connection while
cleanup or a tracked worker is outstanding. See [endpoint selection](simulator-endpoint-selection.md).
The earlier perception-data authorization block was explicitly cleared on
2026-09-07, including permission to send framework test information to the
configured LLM provider. A fresh real-LLM Spatial 0 run reached SAM3 and stopped
at ambiguous target selection, with confirmed cleanup; it did not manipulate
the scene. See [authorized-run evidence and limits](diagnostics/authorized-agent-perception-2026-09-07.md).
- [ ] Improve live structured-action reliability without accepting invented
  fields/references. Fresh pure-Agent runs exposed wrong command kinds,
  malformed XML and incorrect tool parameter names. Exact existing contract
  reminders were added; bounded validation repair can work but still costs
  extra requests. Measure on further runs rather than claiming prompt efficacy
  from one successful startup.
  - R0 28: Object 0 full-task mode reached environment creation but exhausted
    SAM3 validation while alternating wrong/missing fields, with confirmed
    cleanup and zero motion. Added a bounded, detached echo of the previous
    parsed candidate to main-planner repair feedback, without repairing or
    authorizing it on the Host. The fresh same-model rerun still exhausted
    validation (including malformed XML), with cleanup and zero motion; no
    task improvement is claimed. Compare configured model behavior next; see
    [Object 0 evidence and entry limitations](diagnostics/object0-task-regression-2026-09-07.md).
  - R0 29: process-only `gpt-6-astra` comparison failed on its first request
    with upstream HTTP 500 / `server_is_overloaded`, before environment
    creation. Provider failure classification and unknown-usage accounting
    behaved as intended, without human assistance; owned server cleanup was
    confirmed. No model action was obtained, so this is neither a structured
    output comparison result nor a completed task regression.
- [ ] Review the candidate planner-failure reporting correction: exhausted Host
  validation is now `planner_validation_failed` with retained errors/attempts,
  not an ordinary successful `talk`/`status_report`; model parameters alone
  cannot forge that failure. Episode termination presentation needs collaborator
  review. Also resolve empty `talk` messages whose explanation appears only in
  reasoning; this latter issue is not yet fixed.
- [ ] Distinguish provider infrastructure failure, requesting human help and
  actually receiving assistance. Fresh Long 9 reached SAM3 selection, then a
  60 s provider timeout became `ask_human` with empty `failure_reason`; a 0.019 s
  waiting interval marked `human_assisted=true` without any operator response.
  Retain unknown usage for the timed-out request. See the authorized-run report;
  this is not covered by the planner-validation failure correction.
  - R0 27 candidate: Host backend failure now truncates as
    `planner_provider_failed`, without human pause/guidance. Received nonempty
    current-episode answer evidence replaces elapsed-wait inference; pending,
    blank and zero-time cases are tested. Failed calls preserve unknown usage
    sources. See [scope and review requirement](manual-provider-accounting.md).
    Fresh LIBERO/provider regression and cross-role/resume attribution remain open.
- [ ] Resolve current cuRobo MESH / Warp compatibility before claiming mesh
  collision support or selecting that backend. Local GPU tests found that the
  installed Warp 1.16.0 lacks the `warp.torch` API used by the cuRobo mesh checker.
  Production explicitly uses PRIMITIVE, whose checks pass. The test fixture now
  covers both backends and preserves all three mesh initialization errors; no
  dependency shim/downgrade or third-party edits were applied. See the
  [validation matrix and remaining gaps](diagnostics/local-validation-matrix-2026-09-07.md).
- [ ] Separate benchmark-assigned task presentation from diagnostic execution
  constraints in reusable experiment entry points. The live observation smoke
  confirmed that simulator observations supply the console's task title; its
  no-motion boundary was enforced by a transport allowlist, not task prose.
  Preserve the benchmark instruction while making run constraints explicit;
  do not silently change the main task or introduce a Host task-stage machine.

## Harness Correctness And Human VLM Follow-up — 2026-09-05

The completed milestones below describe earlier implementation, not proof that
all runtime edge cases or LIBERO families are covered. Track the current repair
work in [the code review and experiment backlog](harness-code-review-2026-09-05.md).
The report distinguishes reproduced defects, static findings, and reported
Human VLM observations; preserve those evidence boundaries when closing items.

- [ ] Address the review's ten correctness/reliability items: host-memory ownership,
  Python isolation, late writes after cancellation, success classification,
  concurrent stdout, environment cleanup, complete budgets, proxy configuration,
  crash recovery, and contract-validation evidence.
  - Partial: Agent notes/artifact references are separated from host evidence;
    local memory tools check session generation and cancellation at commit.
    Default `python_exec` now uses a disposable Linux Landlock/seccomp worker,
    bounded streams/resources, per-process stdout, and cooperative cancellation
    that reaps before releasing turn ownership. See the
    [sandbox boundary and remaining limitations](python-exec-isolation.md).
    Approved outside-sandbox execution, remote workers, aggregate disk quotas,
    and the remaining review items are still open.
  - Partial: memory checkpoints now publish one checksummed generation; trace
    and conversation share strict sequence/tail recovery, quarantined torn
    bytes, and checkpoint-cursor checks. Restoring incomplete history revokes
    old epoch-bound evidence. See [recovery boundaries](memory-store-recovery.md);
    this is not a transaction across an entire Agent turn or remote motion.
  - Partial: live/offline success consumers now share explicit adapter/checker
    evidence rules; generic positive reward no longer certifies completion.
    Execution/session-bound receipts survive episode serialization, and failure
    cannot be erased by earlier reward. See [success semantics](success-evidence.md)
    for known backend policies, stricter playbook requirements, and legacy-record
    limitations. Real backend regression and general policy registration remain open.
  - Partial: client/server/worker cleanup now retains failed-close identities,
    checks explicit acknowledgements, and retries completed phases without duplicate
    worker release. TTL shares the retirement path; BEHAVIOR release checks process
    exit. See [cleanup ownership and limits](environment-cleanup.md). Durable orphan
    reconciliation, whole-worker shutdown, execution leases, and live validation
    remain open.
  - Follow-up (2026-09-07): MCP observation now shares the complete per-handle
    control lock. Position-only reconciliation (including legacy completed/failed
    snapshots) no longer releases unknown-operation gates; the planner stops the
    episode instead of polling indefinitely. See [timeout safety](motion-outcome-safety.md).
    Host operation IDs/terminal queries/expired-request fencing await design
    confirmation and review; in-place unlock and live timeout validation remain open.
  - Follow-up (2026-09-07): shared local startup admission now prevents duplicate
    create/reset, premature successful close, proxy reuse and endpoint replacement
    while a create/reset callback is outstanding. CLI close retries failures;
    returned handles survive post-processing failures. See [cleanup boundaries](environment-cleanup.md).
    Pending close still requires caller retry; unknown remote creation after a
    transport exception, durable orphan cleanup and live race validation remain open.
  - Partial (2026-09-07): guidance now shares the remaining episode deadline;
    detached worker results cannot write back after timeout, and tracked busy
    runners cannot be reused. Normal guidance usage, including abstention, is
    charged; post-review failures retain the episode result, and budget/interrupt
    stops do not start new reviews. See [budget boundaries](episode-budget-boundaries.md).
    Provider cancellation, missing usage reconciliation and
    independent normal-review budgets remain open; this is not a hard global budget.
  - Follow-up (2026-09-07): post-review now receives a detached episode result
    and returns a detached report; callbacks cannot rewrite task steps/outcome
    through those aliases, even on failure or after return. Review event publication
    checks session generation/memory/execution ownership, including same-ID reopen.
    See [postprocessing isolation](episode-budget-boundaries.md). Independent review
    budgets, provider cancellation and compute/commit separation remain open;
    existing proposal/auto-apply side effects are not rolled back.
  - Partial (2026-09-07): default review and built-in auto-apply now split
    preparation from commit. Runner preparation waits have an independent
    120 s configurable default, detached inputs and tracked late workers; no
    framework proposal/skill commit follows timeout or session replacement.
    Timely commits check config and current skill snapshots. See
    [supported review deadline](episode-budget-boundaries.md). Legacy custom
    callbacks remain synchronous; commit I/O, token budgets, remote cancellation
    and partial-write recovery are still open. This is not a hard whole-review budget.
  - Follow-up (2026-09-07): proposal creation now validates canonical IDs and
    atomically refuses to overwrite existing pending/resolved records. Random
    exclusive temporary files replace the symlink-prone fixed temp name; reads
    check regular-file and embedded identity boundaries. Replayed prepared commits
    stop before repeated auto-apply. See [publication limits](episode-budget-boundaries.md).
    Approval/application CAS, multi-file crash recovery and partial-commit rollback
    remain open; no historical record was rewritten.
  - Follow-up (2026-09-07): a Host-owned atomic quota now admits executable tool
    attempts before authorization/dispatch, including batches and scoped nested
    calls. Exhausted attempts do not reach handlers; pause/resume conservatively
    carries nested admissions. Legacy attempt counts remain distinct from the new
    `tool_admission` ledger. Direct/unscoped clients and per-controller-step limits
    remain outside this quota; see [budget boundaries](episode-budget-boundaries.md).
  - Review item 8 fixed locally (2026-09-07): SSE MCP calls now use a scoped
    HTTP client with direct routing only for the target host, instead of
    mutating process-wide NO_PROXY values. Other-host proxy routes and TLS CA
    environment remain effective. Concurrent/timeout fixtures and a real SDK
    loopback roundtrip passed; see [proxy isolation](mcp-proxy-isolation.md).
    This does not complete the remaining review items or external-service
    regression, and does not authorize pending perception data transmission.
  - Follow-up (2026-09-07): CLI/batch simulator RPC timeouts no longer grow with
    manual/model provider waiting. Independent `--simulator-timeout-s` defaults
    to 300 s and is forwarded by the Human VLM launchers, experiment execution
    and explicit batch resume. Create/reset/tools share it; cleanup remains
    capped at 30 s. Non-default values must be re-supplied on resume. See
    [timeout boundaries](episode-budget-boundaries.md); this is not unified
    remaining-deadline propagation or remote cancellation.
- [ ] Address issue #12's remaining resource-contract, GraspGenX SE(3), compact
  feedback, visual evidence, and placement-bundle gaps using the report's current
  branch comparison; do not reclassify already implemented mechanisms as absent.
- [ ] [HV-01 — Align IK preview with controller execution](harness-code-review-2026-09-05.md#hv-01).
  - Partial (2026-09-06): feasible preview feedback now prominently distinguishes
    endpoint IK from unverified local execution. Declared OSC does not consume
    its joint seed; declared Mink has a validated-seed path but preview alone
    does not prove consumption/convergence. Gates and classification enums remain
    unchanged; controller-local validation and hard-task regression remain open.
  - Live follow-up (2026-09-07): the bounded Object 0 seed-0 Mink canary now
    explicitly forwards the preview seed and checks its receipt ID/policy. The
    2 cm upward probe reached its target in 2/40 steps (1.91 mm Euclidean error),
    with configuration collision checks and confirmed cleanup. See
    [evidence and limits](diagnostics/mink-seed-canary-2026-09-07.md).
    This direct Host probe does not validate the main Agent receipt resolver,
    difficult IK branches, contact or full tasks; HV-01 remains open.
  - Exact-reference fix (2026-09-07): seed resolution now honors the selected
    receipt ID instead of borrowing the latest same-pose seed. The selected
    receipt must itself authorize motion; a newer rejection still vetoes an old
    feasible reference. Trajectory gates retain each waypoint's selected ID.
    See [binding and test scope](ik-seed-binding.md). Main runtime-to-transport
    fixture coverage is not live controller/task regression; review remains open.
  - Runtime live follow-up (2026-09-07): `--through-agent-runtime` dispatched one
    deterministic receipt-ID move through production runtime/gate/proxy into an
    owned Mink simulator; selected seed ID matched the controller receipt,
    2/40 steps reached the 2 cm target, cleanup confirmed. See
    [runtime probe scope](diagnostics/mink-seed-canary-2026-09-07.md).
    Host-ingested actual preview + static planner is not autonomous proposal,
    perception, complex branch or task validation; HV-01 remains open.
- [ ] [HV-02 — Expose tolerance-induced zero-step actions](harness-code-review-2026-09-05.md#hv-02).
  - Partial (2026-09-06): zero-step hints now require a non-boolean integer zero,
    a target-reached stop reason, and no conflicting controller receipt. Removed
    the unsupported `no_state_change` inference and absolute camera/world-state
    claims; receipt-driven zero steps do not prove the whole world is unchanged.
    The proposed structured execution/world-state fields remain open.
- [ ] [HV-03 — Enable pre-grasp active perception and diverse wrist viewpoints](harness-code-review-2026-09-05.md#hv-03).
  - Partial (2026-09-06): existing compiled-grasp entry now samples four azimuths
    and two elevations with an eight-candidate cap. Full rotation/candidate geometry
    is bound into proposal IDs; hints prefer the existing exact short-ID IK path.
    See [sampling limits](wrist-viewpoint-sampling.md). No workspace prefilter,
    actual occlusion-quality validation, pre-SAM3 entry or new-packet projection is
    claimed. Weak-localization versus identity ownership has been raised for decision.
  - Follow-up (2026-09-07): candidate resolution now requires successful results,
    strict epochs and complete source/mount bindings; rechecks active target
    evidence and original/latest wrist calibration, rejects duplicate IDs, and
    returns independent pose copies. Legacy incomplete proposals need regeneration;
    this does not implement pre-SAM3 identity or environment-operation lifetimes.
  - [ ] Review [design options](active-perception-design-options-2026-09-05.md):
    resolve pre-SAM3 anchor bootstrapping, cross-view identity evidence, and
    fresh-packet projection ownership before settling the public contract.
    Active perception is required; the proposed identity-anchor input and
    projection-handoff interface are not yet approved implementation choices.
- [ ] [HV-04 — Separate object identity from graspable-part selection](harness-code-review-2026-09-05.md#hv-04).
- [ ] [HV-05 — Share geometry types across schema and runtime gates](harness-code-review-2026-09-05.md#hv-05).
  - Partial (2026-09-06): selection schema, runtime gate and registry guidance now
    share one immutable vocabulary, including the existing empty/omitted hint
    behavior. Compiler extensions deliberately remain open strings and are tested
    with generic fallback. See [geometry hint boundaries](grasp-geometry-types.md).
    Object/part taxonomy, selection extensions, review and live regression remain
    open; no mug/handle identity capability is inferred from a geometry label.
- [ ] [HV-06 — Surface nested Human VLM advisor requests](harness-code-review-2026-09-05.md#hv-06).
  - Partial (2026-09-06): host-parent correlation, independent child identity,
    parent-grouped queue and explicit human-advisor waiting labels implemented
    for grasp advice. Real local HTTP test verifies discovery and response-unblock;
    this is a fixture, not a human task experiment. See [console contract and
    limits](manual-vlm-harness-debugger.md#isolated-grasp-advisor-correlation-2026-09-06-candidate).
    Running-service rollout, other advisor roles, cancellation propagation and
    latency attribution remain open; additive fields need collaboration review.
  - Role coverage follow-up (2026-09-07): guidance, independent action review
    and visual differencing now receive explicit Host parent correlation and
    fresh child identities; the console recognizes their exact schema/role
    pairs and labels the pending wait. Producer/queue tests plus a fixed-text
    loopback guidance HTTP test pass. See [additional roles](manual-vlm-harness-debugger.md#additional-isolated-roles-2026-09-07-candidate).
    Skill author/review roles, live rollout, cancellation ownership and nested
    wait accounting remain open. Parent session is not a provider request lease.
  - Live follow-up (2026-09-07): a private console served two actual main-planner
    requests with real images; observe feedback was visible and both requests
    were answered. Nested-advisor live coverage remains open. The observation
    smoke also reproduced `human_wait_s=0` / `human_assisted=false` despite
    operator waits of about 44 s and 57 s: these fields currently describe
    runner interaction handling, not all manual-provider activity. Add explicit
    provider-mode/wait attribution before using these metrics for autonomous
    success or latency comparisons; do not rewrite historical records.
  - Partial (2026-09-07): the console now reports monotonic response wait and
    explicit manual-channel mode in its completion envelope. Backend/planner
    validation retries preserve this evidence; recorded main-planner actions
    aggregate it without duplicate request IDs and flag manual assistance.
    See [accounting scope](manual-provider-accounting.md). Existing runner wait
    and budget clocks are unchanged; other roles, incomplete requests, carry
    across resume and fresh live validation remain open. New fields/flag semantics
    require collaboration review; absence is not proof of autonomy.
  - Follow-up (2026-09-07): the existing console's structured `human_cancelled`
    503 now ends the current backend invocation without automatic retry or
    fallback. Main-planner/episode and real loopback cancellation/console-timeout
    fixtures verify that no second request is enqueued. See [cancellation limits](manual-provider-accounting.md).
    This does not implement parent-child cancellation propagation, remote request
    cancellation on client timeout, or cancelled-request usage accounting.
- [ ] [HV-07 — Validate enclosed-container insertion, release, and door clearance](harness-code-review-2026-09-05.md#hv-07).
- [ ] [HV-08 — Reconcile gripper commands with current attachment evidence](harness-code-review-2026-09-05.md#hv-08).
  - Partial (2026-09-06): shared aperture precedence, retained/checked modern
    actuation receipts, no inferred latch from aperture-only reconciliation,
    consistent reopen invalidation, and fresh-open PASS revocation. Ordinary arm
    motion preserves valid acknowledged evidence. See [gripper evidence](gripper-evidence.md).
    Legacy receipt compatibility, full freshness/consumer audit, remote operation
    completion and live recovery validation remain open; this is not a blanket
    removal of the close-evidence gate.
- [ ] [HV-09 — Measure and reduce avoidable episode cost](harness-code-review-2026-09-05.md#hv-09).
  - Partial accounting (2026-09-07): timely guidance responses retain provider
    usage and charge both answers and abstentions. Timeout/malformed-response costs
    remain unknown; no same-task cost-reduction experiment or complete all-role
    usage accounting has been claimed.
- [ ] Retest Spatial 0 and Long 9 after their prerequisites are fixed; retain
  Goal 0, Object 0, and Long 0 as regression cases. Record controller, seed,
  assistance, trusted success evidence, and actual placement-estimator use.

Prioritize HV-01, HV-03, and HV-04 before resuming long diagnostic runs, alongside
the review's execution-boundary fixes. Each item needs a fix revision and relevant
validation; recording a proposal here does not mark it implemented or authorize
an unreviewed shared-contract change.

## Issue #27 — Tool Results, Context, And Evidence Access — 2026-09-05

Source: [issue #27](https://github.com/OpenETA7/Stage2/issues/27), opened by
`No-518`; read on 2026-09-05, OPEN, last updated at 13:08:35 UTC, no comments
at retrieval. The issue reports manual-agent observations and explicitly treats
its implementation ideas as suggestions. The notes below distinguish local
static checks from reported behavior; no simulator or checkpoint-loading
experiment was run for this update. Shared RFC access remains network-blocked;
public contract changes require collaborator review before implementation.

These items refine the existing issue #12 / HV backlog rather than creating a
second implementation track. I27 identifiers map to the issue's nine sections;
priorities below are local triage, not priorities assigned by the author.
Issues #3 / #6 and PRs #9 / #10 are reported there as having resolved the `set`,
terminal-reward, and collision problems; do not reopen them without regression
evidence. I27-09 is the separately reported remaining weight-loading problem.

- [ ] **I27-01 / P2 — Avoid redundant observation after initialization.**
  Merge with HV-09's avoidable-call accounting. `create_simulator_env` already
  returns `initial_observation`, while the generic world-mutation feedback in
  [registry.py](../agent/tools/registry.py) recommends `observe`.
  - Acceptance: when a valid initial packet/images/state are registered and
    available to the next planner turn, do not recommend a duplicate observation
    solely because initialization mutated the world. Missing/invalid observation
    and transport-unknown outcomes must retain their observation requirements;
    do not remove fresh-evidence gates globally.

- [ ] **I27-02 / P2 — Define useful, non-duplicated default tool results.**
  Refine issue #12's compact-feedback work and HV-09. Audit the full path through
  conversation history, current observation, recent transitions, and memory;
  [planner.py](../agent/runtime/planner.py) and
  [memory.py](../agent/runtime/memory.py) still contain `<omitted>` projections.
  - Acceptance: outcome-specific projections preserve actionable IDs, results,
    safety/uncertainty evidence, and failure reasons. Remove useless placeholder
    shells and duplicate payloads; retain an accessible complete artifact.
    Measure the final model input, not only the original tool return.

- [ ] **I27-03 / P1 — Make downstream context truncation truthful and resumable.**
  - Partial (2026-09-06): bounded dictionary projections now disclose omission
    paths/counts and model-visible incompleteness separately from source flags;
    scalar leaves and fixed-size text cursors survive depth limits. A real Python
    worker → final provider body → cursor readback regression is covered. Other
    compression paths and list/scalar-root envelopes remain open. See
    [artifact reading boundaries](artifact-text-reading.md).
  `_bounded_decision_value` in [planner.py](../agent/runtime/planner.py) slices
  lists/dicts and replaces deep values without updating a retained tool-level
  `truncated` flag. The issue's 100-to-8 grep example is reported, not replayed.
  - Acceptance: distinguish original-result completeness from model-visible
    completeness; expose what was retained/omitted, continuation or a full-result
    reference, and unknown counts when totals are not known. Preserve useful
    items and required IDs instead of retaining only their keys. Exercise long
    text, deep nesting, and long hit lists through final prompt assembly; a
    tool's `truncated=false` must not imply that a cropped projection is complete.

- [ ] **I27-04 / P2 — Make stored evidence directly browsable and long lines resumable.**
  - Partial (2026-09-06): existing Python artifact API gained `read_text_page`
    with Unicode/intra-line cursors, path/content-version binding, EOF lookahead
    and unchanged session-root checks. Typed text/JSON paths survive final
    provider projection. Native non-Python tools and image-opening flow remain
    unimplemented; new return schemas/cursors need collaborator review.
  [coding.py](../agent/tools/coding.py) already provides session-scoped
  `artifacts.list_files/list_images/read_json/read_text/grep_text` via Python;
  this is an ergonomics and continuation gap, not absence of artifact storage.
  Its `read_text` currently returns a bounded prefix without a continuation.
  The issue also reports a line-based reader; that exact interface was not
  identified in this local check.
  - Acceptance: expose simple list/read/search/image-view capabilities without
    requiring handwritten Python; names such as `ls/read/grep` remain proposals.
    Returned references must work across these tools within the same session's
    allowed roots, without broadening filesystem access.
  - Acceptance: provide line numbers, honest truncation, and a next cursor;
    support intra-line offsets or structured JSON-field reads so an oversized
    JSONL record can be read without repeating its prefix or skipping its tail.
    Define offset units and file-change behavior. Explicitly opened images must
    reach the next model input, not merely return a path or success message.
    End-to-end reading tests must include I27-03's downstream projection.

- [ ] **I27-05 / P2 — Return grep snippets around actual matches.**
  - Partial (2026-09-06): matching-span snippets, independent snippet/match clipping,
    exact N versus N+1 semantics, query/file-bound pagination and per-hit text
    read cursors implemented/tested, including final model-input readback.
    General regex/long-line resource and native-tool integration limits are
    documented in [artifact reading boundaries](artifact-text-reading.md).
  [text_artifacts.py](../agent/runtime/text_artifacts.py) `grep_text_artifact`
  currently returns `line[:500]` and sets `truncated` when retained matches reach
  the limit; both mechanisms match the issue's reported risks.
  - Acceptance: include the matching span and useful surrounding text, source
    line and offset for follow-up reads; distinguish snippet clipping from
    omitted matches. Exactly N matches at limit N is complete; N+1 must indicate
    more results. Also cover a hit beyond character 500, no matches, Unicode,
    and very long lines. Integrate continuation with I27-04.

- [ ] **I27-06 / P2 — Expose the asset-query environment namespace explicitly.**
  [asset_references.py](../agent/tools/asset_references.py) resolves catalog
  environment names/aliases; initialization's public contract exposes `env_id`
  but does not require an explicit asset-query namespace. A simulator identifier
  such as `openeta/libero_libero_spatial_task2-v0` is not itself a documented
  source for the query value `libero`.
  - Acceptance: expose both the simulator ID and an authoritative catalog/object
    memory namespace, with clear parameter documentation. The Agent still fills
    `retrieve_asset_reference.environment` using that field; it need not guess
    by parsing an ID. Test multiple environments, aliases, and unavailable
    mappings; do not silently default all tasks to LIBERO.

- [ ] **I27-07 / P2 — Separate outstanding questions from established grounding evidence.**
  Refine I27-02 and HV-03/HV-04's identity presentation. In
  [planner.py](../agent/runtime/planner.py), `unresolved_obligations` starts as
  `dict(open_questions)` and both fields enter the decision context. The full
  extent of duplication with `world_evidence` still needs prompt-level auditing.
  - Acceptance: define one compact authoritative presentation for identity,
    localization, packet provenance, and verification rationale; other sections
    reference it rather than repeat the whole record. Only unresolved matters
    appear as obligations. Preserve explicit uncertainty and distinct evidence
    authorities; check the complete model input before/after selection and
    cross-view identity confirmation, not just one compacting function.

- [ ] **I27-08 / P2 — Make grasp visualizations usable for candidate selection.**
  Extend issue #12's existing visualization work rather than add a duplicate
  renderer. [handlers.py](../agent/tools/handlers.py) already renders GraspGenX
  native-pose/sweep-volume top-1 and top-N overlays with candidate references.
  Rendering existence does not establish correct projection, useful framing,
  or delivery to the model.
  - Acceptance: validate calibration and the full pose transform against the
    final candidate IDs/geometry used for selection and execution; show the
    object together with both fingers and enough scene context. Do not let a
    crop hide the gripper, and report out-of-frame geometry instead of implying
    a complete view. Preserve issue #12's SE(3) correction as a dependency.
  - Evaluate reuse of upstream AnyGrasp/GraspGenX visualization and an inline
    top-three image budget, with remaining candidates accessible via I27-04.
    These are suggested implementations, not a fixed contract. Test actual
    next-turn image input, ordering/ID consistency, and bounded visual cost;
    visually plausible candidates still require normal safety checks.

- [ ] **I27-09 / P1 — Remove Contact-GraspNet's NumPy import-order sensitivity.**
  - Progress: leaf-module resolution and restricted-loader unit regressions are
    implemented on the refactor branch. Four fresh-process import-order tests
    are present but skipped here because torch/mink are unavailable; real-weight
    integration remains open. See the [batch log](harness-refactor-progress-2026-09-05.md).
  In [contact_graspnet_core.py](../tools/contact_graspnet_core.py),
  `_register_checkpoint_safe_globals` checks whether `np._core` exists but then
  dereferences `numpy_core.multiarray.scalar` unconditionally. The failure after
  importing Mink is reported in issue #27 / PR #10; only this unsafe attribute
  assumption was statically confirmed here.
  - Acceptance: resolve the actually available `multiarray.scalar` entry and
    register the required checkpoint type aliases across supported NumPy
    layouts. Keep restricted checkpoint loading; do not use an unrestricted
    pickle loader as a workaround. Add a small missing-attribute unit test and
    isolated-process integration tests for Mink-first and reverse import order
    using a trusted checkpoint. Record dependency versions and explicit skips
    when the relevant environment/weights are unavailable.

Suggested order: address I27-03 before relying on compacted evidence tools, and
I27-09 before a Contact-GraspNet/Mink rerun; implement I27-04/05 together. Merge
I27-01/02/07/08 into the existing cost, context, and visualization work. Link
fix revisions and relevant tests before checking any item complete; documenting
an issue is not evidence that its runtime behavior has been fixed.

## Agent Brain

- [x] Implement a multi-step closed-loop episode runner.
  - Run `observe -> plan -> tool_call -> tool result -> memory update -> observe`.
  - Use independent runner-owned turn/tool-call/time/token budgets; let the
    agent or env/checker terminate the episode before resource exhaustion.
  - Expose the loop through the `uv run openeta` CLI and `/run`.

- [x] Define the agent-side tool handler adapter contract.
  - Standardize `ToolResult.details` shapes for perception, planning, safety,
    bookkeeping, and world-mutating tools.
  - Provide dummy handlers for `scene_detector`, `sam3`, `anygrasp`,
    `ik_preview_check`, `move_to`, and `gripper_control`.
  - Ensure tool results can be recorded into session trace and working memory.

- [x] Add minimal checker hooks without locking final sub-agent schema.
  - Add protocol-style placeholders for safety and failure checker backends.
  - Keep `safe_check` as a named `tool_call` capability.
  - Record checker outputs through `metadata` or `ToolResult.details` until the
    shared schema is reviewed.

- [x] Improve planner context assembly.
  - Include relevant markdown skill guidance in planner context.
  - Keep context bounded with skill metadata and memory summaries.
  - Preserve `skill_call` as guidance-only, not hidden execution.

## Skills

- [x] Add `place.md` as text guidance under `agent/skills/`.
  - Use atomic tools such as `observe`, `scene_detector`, `ik_preview_check`,
    `obstacle_avoidance`, `move_to`, and `gripper_control`.

- [x] Add `push.md`, `pull.md`, and `stack.md` skeleton guidance.
  - Keep them as editable markdown skills with frontmatter.
  - Do not introduce macro execution.

- [x] Add a skill selection smoke test.
  - Verify markdown frontmatter loading.
  - Verify selected skill content appears in planner-facing context.

## Memory

- [x] Add CLI visibility for local memory.
  - Add a slash command or tool trace view for facts, artifacts, skill notes,
    compact summary, and session path.

- [x] Implement a compact policy for long sessions.
  - Keep explicit `compact_memory`.
  - Add automatic compaction when planner context reaches the configured
    context-window threshold.
  - Resolve model context windows from provider metadata when available; keep
    manual `OPENETA_LLM_CONTEXT_WINDOW_TOKENS` as the reliable fallback.

- [x] Add an explicit promoted-memory workflow.
  - Keep `.openeta_memory/` as gitignored runtime state.
  - Write to `agent/memory/` only through an explicit reviewed action.
  - Added `memory_extract` skill for agent-driven working-memory extraction;
    reviewed promotion into `agent/memory/` is handled by explicit CLI command.
  - Add `/promote-memory` as the reviewed CLI action for writing promoted
    markdown entries under `agent/memory/`.

## CLI And Runtime UX

- [x] Make CLI `/run` show multi-step tool-call traces.
  - Show planner request, tool parameters, result, memory update, and next turn.
  - Require permission for world-mutating dummy commands.

- [x] Add a dry-run example for model-backed planning.
  - Use existing OpenAI-compatible backend config.
  - Avoid requiring a real simulator.
  - Add `examples/model_backed_planner_dry_run.py` for planner-only backend
    calls without simulator step or tool execution.

- [x] Add a resume or session-id display path.
  - Show current `session_id`.
  - Show JSONL trace path when `JsonMemoryStore` is attached.
  - Add `/session` and print session trace path with episode output.

## Simulator And Tool Integration Boundary

- [x] Keep real simulator handler integration behind the tool registry.
  - Do not let planner call raw simulator APIs directly.
  - Use simulator MCP tools through `SimulatorMcpToolProxy` as the narrow
    boundary.
  - Verified remote MCP tools include `move_to`, `gripper_open`, and
    `gripper_close`.

- [x] Remove low-level simulator state fields from the agent-side observation
  requirement.
  - IK, safety checks, controller expansion, action clipping, EE pose
    convention, and backend action details are simulator-side responsibilities.
  - Agent-facing MCP results should carry only planner/perception-relevant
    information, tool outcomes, and materialized image refs.
  - Full camera intrinsics/extrinsics, mandatory object lists, step index, and
    full-state `observe_env` are no longer required for control execution.
  - Current remote smoke uses `render_env` for fresh camera refresh.

- [ ] Coordinate with perception/control owners on real handlers.
  - [x] `sam3` via remote MCP `segment` and `segment_points`
  - [x] `anygrasp` via remote MCP `detect_grasps`
  - `ik_preview_check`
  - `obstacle_avoidance`
  - [x] `move_to` via MCP `move_to`
  - [x] `gripper_control` via MCP `gripper_open` / `gripper_close`

- [x] Run a real MCP simulator smoke episode.
  - Completed against remote metaworld env
    `openeta/metaworld_50_assembly-v3-v0`.
  - Executed `create_env -> reset -> render_env -> move_to -> gripper_open ->
    render_env -> close_env`.
  - Confirmed MCP base64 images are materialized to local refs before planner
    context.
  - Any future MCP smoke/integration test that calls `create_env` must call
    `close_env` in `finally`; use `close_simulator_mcp_env()` for best-effort
    cleanup.
  - Note: remote MCP does not currently expose `observe_env`; `render_env` is
    used as the observe substitute.

## RFC And Review

- [x] Review section 5 schema with collaborators.
  - Confirmed `tool_call` and `response` as the sufficient top-level command
    classification; `noop` is not part of the planner-facing command surface.
  - Confirmed `skill_call` guidance-only semantics.
  - Confirmed `SkillSpec` markdown/frontmatter fields.
  - Confirmed checker hook direction before adding hard schemas.

Durable contract changes should continue to be synchronized to the RFC with
the branch, commit hash, changed files, validation commands, and any open
contract questions. This is an ongoing collaboration rule rather than a
one-time implementation TODO.

## Demo Acceptance Path

- [x] Add bounded parallel simulator evaluation.
  - `uv run openeta-batch --manifest ... --concurrency 10` runs independent
    model-planned episodes in a thread pool.
  - Keep each episode's world-mutating tool loop serial; isolate sessions,
    traces, artifacts, failures, and cleanup.
  - Keep a hard concurrency limit of 32 and default to 10.
  - Use `success / need_human / fail` lifecycle labels and report autonomous
    versus assisted success separately.
  - Persist `need_human` session/interaction ids, close the expiring simulator
    handle, and restart the same env/task/seed with preserved Agent memory after
    a human answer.
  - Define executable `fail` budgets: 50 concrete tool calls, 600 seconds per
    MCP environment episode, and 5,000,000 cumulative model tokens; report a
    structured failure reason and keep the planner-turn guard separate.
  - Actively interrupt deadline-exceeded turns, request thread-safe environment
    cleanup, reject late step commits, and estimate missing provider token usage
    through the shared TUI token-counting module.

- [x] Run the real agent workflow through `uv run openeta` with multi-step
  closed-loop decisions.
- [ ] Run a real RLinf-backed episode once sim/tool handlers are ready.
  - [x] Validate the remote LIBERO RGBD -> SAM3 -> AnyGrasp -> control/cleanup
    chain against `openeta/libero_libero_10_task0-v0`.
  - [ ] Complete a model-planned long-horizon LIBERO task.
- [ ] Demonstrate one safety block and successful replan.
- [ ] Demonstrate one failure detection and recovery.
- [x] Persist complete episode trace data.
  - `JsonMemoryStore` writes the session event stream to
    `.openeta_memory/sessions/<session_id>/trace.jsonl` and keeps session-local
    working state alongside it.
- [ ] Implement offline episode reconstruction/demo replay from persisted
  traces.
- [ ] Validate no schema fields are missing across a full episode.
