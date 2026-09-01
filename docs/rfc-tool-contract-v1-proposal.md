# ToolContract v1 RFC proposal

Status: approved by three-person review on 2026-08-20 and synchronized to shared
OpenETA RFC section C.8 at revision 2133.

Review requirement satisfied: the durable decision receipt is
`docs/generated/tool-contract-review-decision.json`.

## Decision requested

Adopt `openeta.tool_contract.v1` and
`openeta.tool_contract_catalog.v1` as the single machine-readable source for
Agent-visible tool requests, host resolution, semantic outcomes, evidence
lifetime, gates, repair feedback, and producer-consumer typed facts.

This proposal does not authorize a task-stage state machine. Typed facts state
that interfaces can compose; they never require a particular call order or let
the host select a candidate, fallback, waypoint, or recovery action.

## Contract surfaces

Each tool declaration contains:

1. `request_schema`: parameters authored by the Agent;
2. `host_resolution`: short references and the private paths, calibration,
   epochs, evidence, or controller inputs recovered by the host;
3. `outcomes`: semantic outcome, operational success, required output shape,
   diagnostics/recovery requirements, executable-reference status, and produced
   typed facts;
4. `consumes`: required or optional typed facts with schema version and
   authority;
5. `evidence_lifetime`: scope, freshness dimensions, and invalidation causes;
6. `gate`: deterministic invariants, fail-closed intent, repair schema, stable
   check ids, implementation bindings, allowed repair codes, and the guarantee
   that the Agent retains recovery choice.

Each nontrivial `host_resolution` also carries a stable
`openeta.host_resolver.<tool>.v1` id, qualified runtime implementation, and
resolution layer, plus an explicit `contract_driven_dispatch` flag. A catalog
audit currently binds all 30 declared host-resolved tools with zero identity
gaps; free-form resolver prose is no longer treated as machine identity. Eleven
packet- or provenance-bundle-based tools already select their resolver
implementation through this id and emit `openeta.host_resolution_receipt.v1`;
the other 19 bindings remain explicitly identity-only for incremental migration
at their correct handler, memory, or environment boundary.

`openeta.gate_repair.v1` remains the rejection feedback envelope. It reserves an
empty `extensions` object but defines no extension keys or inner value schema
until a concrete case receives a separately reviewed versioned contract. It must name
the violated invariant, echo the rejected call, expose relevant current/stale
evidence, and offer legal calls with complete parameters. Fields such as
`required_action`, `stage`, or `next_stage` are forbidden when
`preserves_agent_choice=true`.

## Authority boundary

- Agent authority covers only the public request and its intent parameters.
- Host authority covers short-id resolution, session ownership, evidence graph
  lookup, freshness epochs, bundle materialization, and post-environment
  observation packet creation.
- Tool authority covers normalized semantic outcomes and declared result facts.
- Backend authority covers service capabilities, inference metadata, and raw
  backend provenance that is not directly Agent-authored.
- Environment authority covers observations and actual mutation/execution receipts.
- A model may select among valid facts, but may not manufacture host receipts,
  provenance, freshness, or execution authorization.

The `observe` boundary is the canonical example: its handler returns an
observation summary; `source_packet_id` is generated later by the host from the
fresh environment observation and is represented as
`host.post_call_observation.source_packet_id`.

## Migration and compatibility policy

- `inferred`: inventory derived from current ToolSpec prose; never enforceable.
- `declared`: explicitly reviewed shape, still shadow-only.
- `verified`: handler fixtures, rollout conformance, and legacy-validator/gate
  parity have passed; eligible for a separately reviewed enforcement switch.

The existing Planner validator and runtime gates remain authoritative during
shadow migration. Rollouts record both decisions so disagreements can be
inspected without changing behavior. Non-executable tools cannot be promoted
from inferred maturity.

The implementation adds `openeta.tool_contract_runtime_policy.v1` as a single,
fail-closed authority switch shared by Planner and ActionPipeline. Request
validation and gate-repair-envelope authority are separately allowlisted per
tool; both are empty by default and startup rejects unknown or non-`verified`
entries. This policy does not transfer executable safety-gate authority: motion,
provenance, freshness, IK, and receipt predicates continue to run in the host.

Rollout manifests persist `openeta.tool_contract_runtime_provenance.v1` with the
actual catalog hash, actual policy, and Planner/Pipeline alignment. Runtime
construction rejects split catalog or policy authority. An offline auditor
cross-checks those allowlists against every recorded enforcing/shadow request
and gate-repair trace and rejects any executable gate authority other than
`legacy_runtime`.

The Agent-visible `available_tools` block is the first low-risk runtime consumer:
its public request schemas now project from ToolContract, while ToolRegistry
continues to own handler availability and all execution behavior. A host-only
parity record exposes two intentional legacy-only parameter sets currently left
in ToolSpec (`grasp_pose_estimate` geometry inputs and `camera_pose_to_world`
calibration convention inputs). Those duplicates should be deleted after review;
they are not accepted into the new public projection.

No compatibility fallback to local image paths is proposed. Public perception
chains use session-owned short ids so defects in the new resolver path are
visible before product launch.

## Generated projections and tests

- `docs/generated/tool-contracts.json`: machine projection;
- `docs/generated/tool-contracts.md`: human interface reference;
- static typed-fact chain checker for producer-consumer compatibility;
- request-schema shadow parity in Planner rollouts;
- gate-repair shadow conformance in blocked pipeline plans;
- ToolResult semantic conformance over fixtures and live rollout JSONL.
- deterministic valid/invalid request-class readiness matrix, explicitly marked
  as promotion evidence rather than promotion authority.
- per-tool promotion dossier that aggregates deterministic and live evidence,
  lists unresolved criteria, and cannot mutate maturity as a side effect;
- durable authority auditor over rollout manifests and request/gate traces.

`verified` promotion is proposed per tool and requires: catalog-wide structural
request/outcome/gate fixtures; handler fixtures for successful outcomes and a
representative failure; live ToolResult conformance; valid and invalid Planner
shadow parity; observed repair-code-to-check-id coverage; focused host
resolution/freshness tests; extra receipt/authorization tests for world mutation;
and the existing three-person review plus a separately scoped authority canary.
Structural coverage alone is explicitly insufficient for promotion.

## Current implementation evidence

- 35 public facade/runtime interfaces registered and verified;
- all 35 explicit tools pass generated structural request, semantic-outcome, and
  bound gate-repair fixtures;
- no inferred placeholder or direct-backend ToolSpecs remain in the default
  Agent registry. AnyGrasp and GraspGenX are host-internal facade backends;
  Contact-GraspNet has been removed from the facade and runtime assembly;
- core pick, wrist refinement, AnyPlace, memory, lifecycle, and skill-edit chains
  pass static typed-fact compatibility tests;
- current r25 rollout: 22 declared tool-result events, zero conformance
  violations after adding the observed `move_to:no_attachment_evidence` outcome;
- r28 contract-projection canary: the Agent authored `molmopoint`,
  `estimate_depth_prior`, and `enhance_depth` requests using only the same
  `source_packet_id=obs-0000` / `camera_frame_id=agentview`; all three Planner
  shadows matched legacy acceptance and all three ToolResults conformed. The
  Agent did not retry MolmoPoint after its structured timeout, and correctly
  interpreted depth enhancement as `requires_depth_alignment_repair`;
- a deterministic rejected unknown-packet canary returned
  `invalid_source_packet`, matched `runtime.source_packet_resolution`, offered
  `observe`, and had zero gate-repair conformance violations;
- the deterministic request matrix covers all 35 explicit tools. The first
  read-only authority canary promoted `estimate_depth_prior`; the later exact
  34-tool review and post-review canary completed the remainder;
- r29-r31 tested live invalid-request repair for `estimate_depth_prior`. An
  explicit retry rejection envelope stopped unchanged invalid repeats, and the
  new `openeta.planner_validation_receipt.v1` preserves rejected attempts into
  later Agent turns without pretending they executed. r31 rejected the empty
  request, accepted the exact `obs-0000` correction on attempt 2, executed one
  conformant read-only result, and let the next turn accurately report the
  causal chain. This satisfies the candidate's relevant live invalid-request
  evidence, but does not by itself authorize `verified` promotion;
- production registry-bound fixtures now prove the candidate's successful
  `completed` result and structured MCP-timeout `operational_failure` both
  conform. A durable r38 fixture receipt also blocks an unknown source packet
  before tool execution and binds `invalid_source_packet` to
  `runtime.source_packet_resolution`, with a conformant repair shadow and
  `authoritative_gate=legacy_runtime`. The generated dossier now records the
  approved review, passing canary, and completed promotion with no unresolved items;
- deterministic r40 records the current 35-tool empty-policy authority provenance without any model or
  tool call: one auditable manifest, matching Planner/Pipeline catalog and empty
  policies, and zero auditor violations. This is the durable shadow baseline
  for comparison with the enforcing canary;
- r41 records two Planner attempts: one contract-authoritative invalid-request
  rejection and one corrected successful execution. Only
  `estimate_depth_prior` appears in request-validation authority; gate-repair
  authority is empty, executable gates remain legacy, and the authority audit
  reports zero violations;
- the catalog campaign covers the remaining 34 tools with production fixtures,
  per-tool dossiers, 70 integration shadows, and an exact three-person approval.
  Its post-review request-validation canary passed 34/34 tools over 68 Planner
  traces with zero tool execution, zero world mutation, and zero authority-audit
  violations. Gate-repair authority stayed empty and executable gates stayed
  `legacy_runtime`;
- an older pre-receipt rollout correctly reports historical IK and AnyPlace
  schema drift and is not accepted as current behavior.

## Review decision (approved 2026-08-20)

1. Agent/host/tool/backend/environment are the approved v1 fact-authority
   categories. A human category is deferred until a concrete contract requires
   machine-verifiable human confirmation.
2. `verified` maturity is promoted per tool. Evidence remains tracked per
   semantic outcome, while request-validation and gate-repair runtime authority
   remain separate per-tool allowlists. Contract-field maturity is not used.
3. `openeta.gate_repair.v1` reserves an `extensions` object. It is emitted as an
   empty object and has no defined keys or inner value schema. The first concrete
   need must receive a separately reviewed, versioned extension contract.
4. `estimate_depth_prior` is approved as the first `verified` tool and the first
   request-validation authority canary. Gate-repair authority stays empty and
   executable gate authority stays `legacy_runtime`.

The machine-readable decision receipt is
`docs/generated/tool-contract-review-decision.json`. Shared RFC synchronization
is authorized by this review.
