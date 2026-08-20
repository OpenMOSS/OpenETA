# OpenETA tool contract system

Status: reviewed ToolContract v1 integration with contract-driven Agent
documentation, fail-closed per-tool authority policy, and the first narrowly
enforcing request-validation canary.
The Agent-visible `available_tools` projection now comes from ToolContract,
while deployment availability still comes from ToolRegistry handler binding.
The production default authority allowlists remain empty. The reviewed r41
canary explicitly enables request validation only for the verified
`estimate_depth_prior` contract; runtime gates remain legacy-authoritative.
ToolContract cannot become authoritative for an inferred or declared tool.

## Why this exists

OpenETA tools must compose through explicit evidence rather than shared
assumptions hidden in Planner validation, memory resolvers, or handler-specific
branches. A tool contract therefore describes four different surfaces:

1. the small request the Agent is allowed to author;
2. private inputs that the host resolves from opaque ids and evidence bundles;
3. outputs for each semantic outcome, including whether an output authorizes
   execution or only supplies diagnostics;
4. typed facts, authority, freshness, invalidation, gate checks, and repair
   feedback used between producers and consumers.

Typed-fact compatibility is not a task state machine. A producer-consumer edge
states that two interfaces can connect. It never requires the Agent to call the
tools in that order or makes the host select a candidate, fallback, waypoint,
or recovery action.

## Single source of truth

The versioned model and catalog live in `agent/tools/contracts.py`. Explicit
core-tool declarations live in `agent/tools/default_contracts.py`; every other
registered ToolSpec receives an `inferred` inventory contract until its request,
outcomes, evidence lifetime, and gates are reviewed.

Generated projections are:

- `docs/generated/tool-contracts.json` for machines, tests, and future runtime
  consumers;
- `docs/generated/tool-contracts.md` for human interface review.
- `docs/generated/tool-contract-readiness.json` for deterministic valid/invalid
  Planner acceptance parity evidence.
- `docs/generated/estimate-depth-prior-promotion-dossier.json` for the first
  candidate's review evidence and explicitly unresolved promotion conditions.
- `docs/generated/estimate-depth-prior-fixture-receipt.json` for durable
  production-handler and host-resolution gate evidence for that candidate.
- `docs/generated/tool-contract-authority-shadow-audit.json` for a current
  no-action shadow provenance canary.
- `docs/generated/tool-contract-authority-shadow-baseline.json` for the
  reproducible command output that created that manifest.
- `docs/generated/host-resolver-binding-audit.json` for resolver identity,
  dispatch coverage, and resolver-failure-to-gate-code coverage.
- `docs/generated/tool-contract-review-decision.json` for the approved three-person
  authority and promotion decisions.
- `docs/generated/tool-contract-authority-canary.json` for the first narrow
  enforcing request-validation canary.
- `docs/generated/tool-contract-shared-rfc-sync.json` for the verified shared RFC
  section and revision.
- `docs/generated/tool-contract-migration-status.json` for the reproducible
  end-to-end migration completion audit.

Do not edit generated files by hand. Regenerate them from their corresponding
scripts after a declaration or evidence change.

```bash
python scripts/generate_tool_contract_docs.py \
  --json-output docs/generated/tool-contracts.json \
  --markdown-output docs/generated/tool-contracts.md
```

At runtime, `available_tools_schema_version=openeta.agent_tool_contract.v1`
identifies the compact Agent projection. Its `parameters` field is the canonical
request schema, rather than another prose-only parameter map. A host-only
`tool_contract_projection_audit` compares identity and top-level parameter names
against legacy ToolSpec. The current two intentional mismatches remove obsolete
host-resolved geometry/calibration fields from `grasp_pose_estimate` and
`camera_pose_to_world`; they are evidence for deleting those ToolSpec duplicates,
not hidden compatibility fallbacks. The entire tool block remains in the
cache-stable prefix.

The same contract projection is supplied to the independent action reviewer,
skill lint reads public parameter names from the contract, and rollout manifests
persist the full versioned contract alongside the deployment-time `executable`
flag. ToolRegistry remains the binding/dispatch authority; those consumers no
longer create additional public interface descriptions.

The catalog maturity levels are:

- `inferred`: lossless inventory derived from ToolSpec prose; not enforceable;
- `declared`: request, semantic outcomes, typed facts, lifetime and gate intent
  have been made explicit, but live rollout conformance is not yet proven;
- `verified`: declaration is checked against handler/gate behavior and rollout
  fixtures before it can become authoritative.

Promotion to `verified` is deliberately evidence-gated and per tool, not a
catalog-wide switch. A tool is eligible only when all of the following hold:

1. its structural request, every declared semantic outcome, and every bound gate
   repair code pass catalog-wide conformance fixtures;
2. production handler fixtures cover every operationally successful outcome and
   at least one representative operational failure without undeclared fields or
   outcomes;
3. live rollout events contain no ToolResult conformance violations;
4. Planner shadow samples cover both a valid request and relevant invalid request
   classes with 100% legacy/contract acceptance parity;
5. every observed gate rejection resolves to a declared `check_id`, returns a
   conformant repair bundle, and never prescribes a task stage;
6. host resolution, freshness and invalidation have focused stale/unknown-id
   tests; world-mutating tools additionally prove execution-reference and
   environment-receipt safety;
7. the three-person review approves the shared schema/authority change and a
   separately scoped canary enables contract authority for that tool.

The catalog-wide structural fixtures cover all 35 explicit contracts, but that
alone does not satisfy live handler/rollout evidence. `estimate_depth_prior` is
the only contract promoted to `verified`; the other 34 remain `declared`.

`openeta.tool_contract_runtime_policy.v1` is the only authority switch. It has
separate per-tool allowlists for request validation and gate-repair-envelope
validation, defaults both to empty, and rejects startup if any configured tool
is unknown, is not `verified`, or lacks gate bindings. Planner and ActionPipeline
receive the same policy and catalog from runtime assembly. Executable gate
predicates remain `legacy_runtime`: repair-envelope authority never substitutes
for motion reconciliation, provenance, IK, freshness, or safety checks.

Every new rollout now persists
`openeta.tool_contract_runtime_provenance.v1`: the exact catalog SHA-256 and
coverage summary, the actual policy allowlists, and independent Planner/Pipeline
Planner and ActionPipeline disagree. This prevents a future authority canary
from silently recording the default catalog while executing a promoted custom
catalog.

Create a reproducible no-action baseline without calling a provider or tool:

```bash
python scripts/build_tool_contract_authority_baseline.py \
  --root .openeta_eval/audits/toolcontract-authority-shadow-20260820-r40 \
  --session-id authority-shadow-r40 \
  --output docs/generated/tool-contract-authority-shadow-baseline.json
```

Audit that durable configuration and every observed request/gate trace with:

```bash
python scripts/check_tool_contract_authority.py --run .openeta_eval/runs/<run-id>
```

The audit rejects non-`verified` allowlist entries, Planner/Pipeline identity
drift, enforcing traces absent from policy, policy entries absent from traces,
and any attempt to transfer executable gate authority away from
`legacy_runtime`. Historical runs created before this provenance schema remain
valid experiment evidence, but correctly fail the new authority-reproducibility
audit rather than being retroactively rewritten.

The deterministic r40 provenance canary created one current manifest without a
model or tool call. Its auditor found one auditable manifest, matching
Planner/Pipeline catalog hash, matching empty policies, and zero violations.
This is the empty-policy baseline compared with the reviewed r41 enforcing
canary without conflating behavior with configuration drift.

Host resolution no longer uses free-form prose as its runtime identity. The 30
explicit tools with host-resolved inputs now expose stable
`openeta.host_resolver.<tool>.v1` ids plus a qualified implementation and
resolution layer (`pipeline`, `memory`, `handler`, or `post_environment`).
`audit_host_resolver_bindings` checks every non-`none` resolution contract
against the runtime binding inventory. This caught and removes an important
ambiguity: `move_to` binds `resolve_ik_motion_reference`, while
`follow_eef_trajectory` binds `resolve_ik_trajectory_reference`; both had
previously shared the same vague resolver label.

Eleven resolvers now go one step beyond identity audit. The first five are
source-packet resolvers:
`sam3`, `retrieve_asset_reference`, `molmopoint`, `estimate_depth_prior`, and
`enhance_depth`. The next six are provenance-bundle resolvers:
`grasp_pose_estimate`, `anyplace`, `camera_pose_to_world`,
`compile_grasp_seed`, `compute_wrist_alignment`, and
`propose_wrist_viewpoints`. They dispatch through the stable resolver id stored
in their ToolContract. Success and rejection both emit
`openeta.host_resolution_receipt.v1`, including the selected id,
implementation, public/resolved parameter-key boundary, and
`dispatch_authority=tool_contract`. Resolver failure still flows through the
existing fail-closed runtime gate and `openeta.gate_repair.v1`; executable gate
authority remains `legacy_runtime`. The binding audit reports 11 as
contract-driven and 19 as identity-only bindings, so migration progress is
measurable without falsely labeling handler- or environment-owned resolution as
a Pipeline branch.

Regenerate the resolver audit with:

```bash
python scripts/check_host_resolver_bindings.py \
  --output docs/generated/host-resolver-binding-audit.json
```

The audit also requires every dispatch resolver's default failure repair code to
be declared by a machine-bound gate check. A resolver cannot silently invent a
new rejection code that the ToolContract repair surface does not describe.

Generate the migration completion audit after the full regression suite with:

```bash
python scripts/check_tool_contract_migration_status.py \
  --repo-root . \
  --test-passed 1353 \
  --test-skipped 12 \
  --test-warnings 36 \
  --harness-revision 267 \
  --output docs/generated/tool-contract-migration-status.json \
  --require-complete
```

The command exits successfully only after local evidence, review, shared RFC
sync, and the narrow authority canary all conform. The current report records
`implementation_ready_for_review=true` and `goal_complete=true`; this completion
does not grant runtime authority beyond the explicitly reviewed canary policy.

The default catalog now contains only the 35 reviewed public interfaces, all at
explicit maturity: 34 are `declared` and `estimate_depth_prior` is the first
review-approved `verified` contract. Six unimplemented architecture placeholders were removed
from default registration. AnyGrasp and GraspGenX are host-internal handlers in
the facade's backend map rather than ToolSpecs; GraspGenX capability discovery
is host-only. Contact-GraspNet is no longer loaded by the runtime or accepted by
the facade. Its isolated adapter/service code remains only for legacy deployment
compatibility and cannot enter Agent `available_tools`.

The 2026-08-20 three-person review approved the five v1 fact-authority
categories, per-tool maturity with per-outcome evidence, and the narrow
`estimate_depth_prior` request-validation authority canary. `openeta.gate_repair.v1`
now emits a reserved empty `extensions` object; v1 assigns no keys, semantics, or
inner schema to it. A concrete future need must define a separately reviewed,
versioned extension rather than retroactively guessing one here. Gate-repair
authority remains empty and executable gate authority remains `legacy_runtime`.

## Checking a proposed chain

Use the static checker to detect missing producer facts before implementing a
new chain:

```bash
python scripts/check_tool_chain_contracts.py \
  observe,sam3,select_sam3_detection,grasp_pose_estimate,compile_grasp_seed,ik_preview_check,move_to
```

Partial chains that begin with a host-synthesized bundle name that fact
explicitly:

```bash
python scripts/check_tool_chain_contracts.py \
  anyplace,camera_pose_to_world \
  --initial-fact openeta.anyplace_input_bundle.v1
```

The checker considers all operationally successful outcomes as possible
producers. Runtime semantic outcome, freshness, safety, and task suitability
still need live evidence and gate checks.

## Auditing live results

Audit one rollout or a run tree containing multiple `tool_calls.jsonl` files:

```bash
python -m agent.evals.tool_contract_conformance \
  --tool-events .openeta_eval/runs/<run-id>
```

The audit checks declared semantic outcomes, required outputs, diagnostics,
recovery options, and the rule that non-executable results cannot expose a
motion authorization. It skips inferred contracts rather than treating their
inventory shape as authoritative.

Planner request validation now records
`openeta.tool_contract_shadow_validation.v1` in each parsed decision. The record
contains legacy acceptance, contract acceptance, and their parity. Gate repair
bundles similarly include `openeta.gate_contract_shadow_validation.v1`, which
checks that a rejection identifies the violated invariant, echoes the rejected
call, cites evidence, and provides executable repair options without prescribing
a host-owned task stage. Gate declarations also bind stable `check_id` values to
their implementation locations and allowed repair codes. Shadow validation
records every matching check id and treats an unregistered repair code as a
contract discrepancy. Both integrations are observational only.

Summarize request-validator parity from a rollout tree with:

```bash
python -m agent.evals.tool_contract_shadow \
  --model-calls .openeta_eval/runs/<run-id>
```

The command fails when any request has different legacy/contract acceptance or
when a supposedly shadow-only record unexpectedly reports enforcement.

Generate deterministic valid/invalid request classes with:

```bash
python scripts/check_tool_contract_readiness.py \
  --output docs/generated/tool-contract-readiness.json
```

The current matrix covers all 35 explicit tools and finds six with complete
parity over its generated classes: `enhance_depth`, `estimate_depth_prior`,
`grasp_pose_estimate`, `anyplace`, `camera_pose_to_world`, and
`gripper_control`. This is promotion evidence only. `estimate_depth_prior` is
the preferred first read-only authority canary because r28 supplies a live
successful request/result and r31 supplies the relevant invalid-request repair.
It is now the only `verified` contract after review and the separately scoped
authority canary completed. The matrix also makes legacy drift explicit instead
of hiding it behind a catalog-wide switch.

First build the durable runtime fixture receipt, then build the review dossier;
neither command changes maturity or authority:

```bash
python scripts/build_tool_contract_fixture_receipt.py \
  --tool estimate_depth_prior \
  --artifact-root .openeta_eval/audits/toolcontract-fixture-estimate-depth-prior-r38 \
  --output docs/generated/estimate-depth-prior-fixture-receipt.json
```

```bash
python scripts/build_tool_contract_promotion_dossier.py \
  --tool estimate_depth_prior \
  --run .openeta_eval/runs/systematic-harness-toolcontract-projection-20260820-r28 \
  --run .openeta_eval/runs/systematic-harness-toolcontract-invalid-request-20260820-r31 \
  --fixture-receipt docs/generated/estimate-depth-prior-fixture-receipt.json \
  --review-decision docs/generated/tool-contract-review-decision.json \
  --authority-canary docs/generated/tool-contract-authority-canary.json \
  --output docs/generated/estimate-depth-prior-promotion-dossier.json
```

The current dossier is `eligible_for_review=true` and
`eligible_for_verified_promotion=true`, with `promotion_complete=true`. It proves deterministic and live
valid/invalid request parity, a conformant live result, and shadow-only evidence.
The r38 receipt additionally executes the registry-bound production handler for
both `completed` and structured MCP-timeout `operational_failure`, then blocks an
unknown packet before handler execution and resolves `invalid_source_packet` to
`runtime.source_packet_resolution`. Its host-resolution receipt proves that the
stable resolver id selected the new contract-driven dispatch path; its repair
shadow is conformant and keeps `authoritative_gate=legacy_runtime`. The approved
review receipt and passing r41 canary close the prior two unresolved conditions.

The r28 no-motion packet canary provides the first live evidence for the
contract-driven projection: three Agent-authored calls copied only
`obs-0000`/`agentview`, achieved 3/3 Planner acceptance parity, and produced
three conformant ToolResults. MolmoPoint timed out operationally, after which the
Agent followed the canary contract and did not retry; depth-prior succeeded and
depth enhancement returned the declared diagnostic outcome
`requires_depth_alignment_repair`. The generic evaluator labels the final
`talk` status report as `agent_failure` because this canary explicitly does not
claim physical task completion; that benchmark label is not a ToolContract
failure. A separate deterministic invalid-packet canary bound the rejection to
`runtime.source_packet_resolution` with conformant `observe` repair guidance.

The r29-r31 invalid-request canaries exercise the Planner retry boundary for
`estimate_depth_prior`. r29 proved that merely serializing `validation_errors`
was insufficient: the model repeated the same empty parameters on all three
attempts. The retry prompt now carries an explicit
`previous_attempt_rejected` envelope, says the first attempt is complete, and
requires a changed candidate. r30 then rejected the empty request and accepted
the corrected `source_packet_id=obs-0000` request, but the following turn could
not see that same-decision rejection. The canonical host-result conversation now
includes `openeta.planner_validation_receipt.v1`, a compact receipt that lists
rejected candidates, validation errors, the accepted attempt, and the fact that
only the accepted candidate was executed. r31 passed the complete chain:

- invalid and corrected request shadows both matched legacy acceptance;
- the model corrected the request on attempt 2 using exact visible evidence;
- exactly one corrected `estimate_depth_prior` call executed and its completed
  ToolResult had zero conformance violations;
- the next Planner turn accurately reported the rejection, repair, and preserved
  `obs-0000` / `agentview` provenance from the receipt;
- no robot or gripper action occurred.

The generic evaluator still labels this run `agent_failure` because the canary
ends with `talk` and intentionally does not claim physical task completion. The
contract-specific objective passed.

## Migration rules

- First declare and test current behavior; do not silently redesign an
  interface while documenting it.
- Agent-visible requests should prefer short session-owned ids. Host resolvers
  recover paths, calibration, provenance, epochs, and private controller seeds.
- Facts created after an environment boundary must retain host authority. For
  example, `observe` returns a camera/object summary; the environment adapter
  subsequently creates `host.post_call_observation.source_packet_id`. The tool
  must not claim that packet id as a handler-authored output.
- A non-executable outcome must never expose a motion execution reference.
- Gate failures use `openeta.gate_repair.v1` and include the violated invariant,
  relevant evidence ids, current/stale evidence, and legal repair options.
- A runtime rejection code must be listed by at least one machine-bound gate
  check for that tool. Binding a code documents an evidence/safety invariant; it
  must never encode a host-selected task stage or required next action.
- Freshness dimensions are evidence properties, not task stages. Packet-id-only
  refresh does not invalidate geometry whose declared epochs are unchanged.
- Large complete outputs remain durable artifacts. The ToolResult projection
  exposes stable ids, semantic facts, diagnostics, and explicit artifact paths.
- Planner request validation and gate-repair envelopes can enforce only
  explicitly allowlisted `verified` contracts. Executable gate predicates remain
  host runtime checks. Non-allowlisted tools retain the existing validator/gate
  authority and the contract runs only as a comparison oracle.

Any shared schema or authority change is an RFC proposal and requires the
project's three-person review before the shared RFC is edited. Canary details
and rollout-specific findings belong in the Harness experiment Wiki rather than
the universal contract.

## Current validation baseline

The 2026-08-20 reviewed migration baseline passes the full regression suite.
The generated catalog contains 34 declared plus one verified public tool and no inferred
placeholder/backend entries. All 30
non-`none` host-resolution declarations have stable runtime bindings, and the
r40 durable authority audit reports one auditable empty-policy manifest with zero
violations. The r41 canary enables only `estimate_depth_prior` request validation,
executes one repaired valid request after one contract rejection, and also audits
with zero violations. No executable gate predicate is transferred.

The machine-readable migration audit requires the local projections, review
decision receipt, r41 authority canary, promotion dossier, and shared RFC sync
receipt together. Passing it grants no authority beyond the one reviewed tool
and one request-validation surface.
