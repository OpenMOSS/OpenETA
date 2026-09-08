# Stages 2/3: standard Luna batch experiments

User authorized continuous stages 2 and 3 migration and provider testing.
The OpenETA collaboration skill was used to separate experimental contracts
and per-session evidence from shared stable-schema authority. RFC collaboration
constraints read at revision 2158; no shared writes, review hash changes,
commits or pushes. Design/scope: [stage migration notes](../bundle-interface-stages23-2026-09-07.md).

## Configuration

Run root: `tmp/batch-luna-stages23-MJYJuG/`. One manifest, two fresh Object 0 /
seed 0 episodes, in order `bundle_stage2`, `bundle_stage3`. Separate workspace
roots `stage2/` and `stage3/`, original task text, no imported historical traces.
Standard batch entry, default skills/vision history/supervision/backend bindings,
concurrency 1, provider concurrency 1. Main/fallback both Luna at the authorized
provider; no higher-cost model comparison or silent fallback of interfaces.

Each task: 15,000,000 known cumulative tokens, 160 planner turns, 320 tool calls,
10,800 seconds. Dedicated simulator 127.0.0.1:18766, one GPU 0 worker, Mink
joint-velocity profile with existing `/tmp/openeta-mink-canary-min` overlay.
Outer simulator lifetime 22,500 seconds covers two sequential task budgets and
cleanup. Owned simulator PID 1680762. Runtime code frozen during the batch.

```bash
env OPENETA_LLM_MODEL=gpt-5.6-luna OPENETA_LLM_FALLBACK_MODEL=gpt-5.6-luna \
  .venv/bin/python -u -m agent.cli.batch_eval \
  --manifest tmp/batch-luna-stages23-MJYJuG/manifest.json \
  --concurrency 1 --provider-concurrency 1 --model gpt-5.6-luna \
  --sim-url http://127.0.0.1:18766/sse \
  --batch-id standard-luna-stages23-20260907 \
  --output tmp/batch-luna-stages23-MJYJuG/result.json
```

## Preflight

- Final local suite: 2404 passed / 25 skipped / one existing reviewed-authority
  failure, 36 warnings, 65.63 s. `tmp/bundle-stages23-final-tests.xml` / `.log`.
  External model integrations disabled; local loopback fixtures permitted.
- New migration tests cover all five IK sources, auto-produced targets,
  identity choices, native evidence-gate retention, ordered trajectory references,
  no registration from late/uncommitted handlers, durable resume staleness,
  request projections and common runtime assembly.
- Schema pattern checking exposed a pre-existing structural test fixture that
  used a non-URL for a declared HTTPS URL; the fixture now satisfies the schema.
  Superseded-grasp guidance now points to the profile-visible current input
  instead of printing a native grasp ID; its assertion was updated accordingly.
- Manifest validate-only and `git diff --check` passed. Per-stage results are
  interface regression evidence, not a success-rate estimate or causal study.

## Stage 2

Agent session `ed1a1a8c-54d3-4249-82f4-2a2be0c145cd`.
First recorded provider request verified Luna / 16384 output cap / 36 tools:
compile, IK and move have strict bundle-only schemas, while
`propose_motion_target` preserves all five native target sources as an authoring
tool. Stage scope and safety notes reach the actual provider request.

Interim evidence (not final outcome): three early planner rejections involved
SAM3 `evidence_id`, point-mode shape/prompt mixing, and the still-native stage-2
selection `result_id` alias. A compile call with surrounding spaces in its
manifest ID passed schema admission but was blocked by exact-ID resolution;
the Agent then copied the exact ID and compiled successfully. This is a schema
diagnostic consistency improvement candidate, not a reason to loosen resolution.

First IK used a target manifest successfully. Its multistart search completed
without a full-pose solution (`unknown`, `feasible=null`, no motion authorization);
the response did not claim an executable best iterate. Agent next compiled a
different candidate. One provider attempt returned HTTP 500 with explicit
`server_is_overloaded`; same-model Luna retry succeeded (151.4 s failed attempt,
74.3 s successful attempt, 240.7 s total request including backoff).

Final disposition: operator-interrupted after reproducing a Host discovery bug;
not a completed performance sample. 36 main requests / 37 HTTP attempts,
5 planner rejections, 1,444,644 known main tokens, 30 episode steps and 25
dispatched tool calls. These usage counts exclude unrecorded failed-advisor usage.
Tool counts: SAM3 2, selection 1, grasp estimation 2, compile 10, IK 7,
observe 1, Python 2. No move, trajectory or gripper calls; no task success.

Post-run semantic correction: the selected mask [253,302,299,369] covers the
foreground red/green tomato-sauce can, not the blue alphabet-soup can requested
by the task. Confirmed by scene and local LIBERO asset texture inspection during
the fresh rerun; see [semantic evidence](bundle-stages23-discovery-luna-2026-09-07.md).
References above to the target describe Agent intent, not verified object identity.
Do not treat these previews as intended-target IK coverage.

Additional interim context finding: request 14's reasoning calls candidate 002
an already-failed IK target, although dispatched IK covered only 000 and 001.
The actual provider context correctly contains the new candidate-002 target
manifests and the two older receipt IDs; there is no evidence of Host relabeling
002 as failed. However, the compact `ik_result` handoff summary omits the target
bundle / candidate / waypoint association and emphasizes numerical diagnostics.
Preserving that association in the summary is a useful follow-up; it is not
proof that this omission alone caused the Agent's mistaken recollection.

By request 28, two grasp-generation calls returned matching early geometries
under different result/candidate IDs. The existing geometric search fingerprints
correctly joined the repeated IK searches: `attempt_count=2`, previous receipt
present and `same_search_previously_failed=true` in the actual provider context.
Thus cross-result accumulation is working for these exact repeated poses, but
the Agent still continued candidate recompilation. Both advisor calls failed
decision validation; current diagnostics do not preserve the rejected advisor
decision, so the precise bad value/shape is unknown. An AnyPlace raw-image call
was rejected at planner admission; a later `grasp:` ID passed string admission
but was rejected by AnyPlace's native provenance resolver before remote dispatch.

## Stage 3

Not started: the parent batch was interrupted before this queued episode.
It must be tested in the fresh post-repair run, not described as a failure.

## Confirmed discovery defect and post-run repair

At request 33 the current provider context's recency-only 12 handoffs contained
only target and IK results. The still-current grasp-candidate manifest
`bnd-990fda64f1eb44549b8a5961d48e2895` was absent, while the next compile still
required it. The Agent repeatedly substituted visible native grasp IDs.
Earlier requests had successfully used the correct ID. This is concrete Host
discoverability loss, though it does not explain every earlier model mistake.

After complete cleanup, changed the bounded handoff window to reserve the latest
current bundle of each kind, then fill remaining slots by recency. No epoch or
native gate is relaxed. IK summaries now retain target pose/signature and parent
target bundle. All migrated consumers and trajectory composition share the
registered ID pattern at schema admission, rejecting spaces/native IDs early.
Focused checks: 58 passed. A fresh manifest with the same task budgets is at
`tmp/batch-luna-stages23-discovery-tv2f3f/manifest.json`; results are separate.

## Cleanup

Client PID 1683050 exited 130 after SIGINT. Trace `episode_interrupt` records
`parallel_batch_interrupted` and confirmed remote `close_state=closed`, `ok=true`,
no cleanup errors. Then owned simulator PID 1680762 exited 0. Exact client,
server, worker 1683211 and port 18766 were verified gone. Unrelated services
were not touched. Existing CLI interruption-finalization defect remains: no
normal result.json was emitted. Counts above are reconstructed from raw records,
not a fabricated successful batch result.
