# Typed bundle handoffs and IK recovery

User approved the direction on 2026-09-07. Implementation is incremental on
`dev/huaizezheng/harness-refactor-2026-09-05`; no commit or push. RFC revision
2158 was read. New request alternatives, output fields and IK verdict meaning
require three-person contract review. User design approval is not that review.
Historical authority hashes are not rewritten to claim approval.

## Implemented first slice

- `grasp_pose_estimate` publishes an inspectable `grasp_candidates` bundle;
  `compile_grasp_seed` accepts its `bundle_id` plus the Agent's `candidate_id`.
- Compilation publishes distinct `target_pose` bundles for clearance,
  precontact and contact when those poses exist. IK accepts one bundle ID.
- Existing wrist-viewpoint results also publish one target-pose bundle per
  candidate (at most eight). Selecting a view still belongs to the Agent.
- IK publishes an `ik_result` bundle, including failed diagnostic results.
  `move_to(bundle_id=...)` resolves its exact receipt and runs the old live
  authorization, freshness and execution-seed gates. Merely having an IK
  result bundle never authorizes movement.
- Legacy reference parameters remain supported as mutually exclusive input
  alternatives. Bundle and partial/full legacy reference branches cannot mix.
  Runtime translates references, not unknown parameter aliases or guessed IDs.
- `artifacts.read_bundle(bundle_id)` supplements existing list, grep, paged
  text and JSON inspection inside the Python sandbox. It is read-only evidence
  access, not an authority grant or a simulator API.
- Public SAM3 schema now rejects the observed nonempty text/point mixtures,
  matching the existing planner gate for these cases. This is not a claim of
  exhaustive parity for every legacy empty/null/alias combination.

Example shape (IDs are placeholders, not executable references):

```json
{"tool":"compile_grasp_seed","parameters":{"bundle_id":"<candidate-bundle>","candidate_id":"<chosen-candidate>"}}
```

```json
{"tool":"ik_preview_check","parameters":{"bundle_id":"<chosen-waypoint-bundle>"}}
```

```json
{"tool":"move_to","parameters":{"bundle_id":"<authorized-current-ik-bundle>"}}
```

There is no mandatory tool order. The examples explain reference consumption,
not a Host-selected task state machine.

## Storage and trust

One ID maps to one immutable JSON manifest under the session artifact root's
`bundles/` directory. A new manifest gets a new ID; files are never overwritten.
Each contains schema/type, session, producer, parent bundle IDs when available,
object/robot epochs, typed reference bindings and a decision summary. Large
candidate assets remain referenced rather than duplicated through the chain.
This slice wraps existing provenance resources; it does not eliminate them or
make a manifest self-sufficient without the session's evidence graph.

Only Host-registered entries in working memory resolve for tool consumption.
The resolver checks registration, session/type, exact path, file hash and epochs,
then invokes the existing reference path and all its calibration/provenance gates.
Agent `save_memory` writes a separate namespace. A JSON file written by Python,
including a copied/edited manifest, is not registered proof. Standard assembly
grants session reads but only sandbox-directory writes to generated Python.

Registration is session scoped and persisted with action memory. The planner
gets a bounded recent handoff projection (12 by default), not the entire index;
older files remain inspectable. Current-epoch status is a cheap projection, not
a substitute for the live ownership/calibration/receipt gate. In-memory-only
runtimes without an artifact root retain legacy references and publish no
pretend durable bundle. Persistence failures leave explicit diagnostics and
do not retroactively alter the physical result of the tool.
Session resume preserves registration and files but intentionally advances
evidence epochs; old bundles remain inspectable, not automatically executable.
Nested bundle-to-bundle resolution is rejected rather than recursively expanded.

## IK search and failure evidence

Direct solver, worker endpoint and MCP server now share defaults of 64 starts,
1000 function evaluations per start and 30 seconds total. Existing hard caps
remain 64 / 2000 / 30 seconds. The total wall budget also covers the diagnostic
position-only and orientation-only searches. Easy valid targets can finish early.
These are Host search settings, not new Agent-authored numerical parameters.

Completed numerical search without a solution now returns `status=unknown`,
`feasible=null`, `reason_code=ik_search_no_solution`. `constraint_diagnosis`
retains the component diagnosis; it is not an infeasibility proof. Timeout has
its separate `ik_search_timeout` reason. Search budget, residuals and suggestions
are preserved. Older explicit rejection receipts remain fail-closed.
Timeouts retain completed attempts, their solver-reported function evaluations,
actual residual-call counts and the best completed full-pose candidate. The
partially interrupted attempt has no fabricated solver-reported evaluation count.

The memory receipt includes bounded-history retry evidence keyed by pose/policy,
tolerances, effective solver budget, robot/object epochs and controller
capabilities. An identical failed search is shown explicitly; repetition is not
automatically prohibited and Host does not choose the next candidate. A changed
budget or epoch starts a separate search context. Current fingerprint uses the
existing pose-policy signature, not a new complete numerical-equivalence
canonicalizer: alternate equivalent rotation encodings may not deduplicate.

## Grasp refinement versus active perception

Existing `compute_wrist_alignment` and `propose_wrist_viewpoints` remain separate
tools. Their outputs now declare `geometry_intent` with distinct quality
criteria and explicit `quality_authorizes_motion=false`. Geometry, storage,
inspection and exact-motion gates can be shared without equating visual quality
with contact validity. A reached viewpoint never proves a refined grasp.

Not implemented in this slice: pre-SAM/compiled-grasp-free entry, weak point/ROI
localization bundles, measured post-motion target projection, object/part
identity separation, or a new orientation-changing contact-refinement tool.
The existing viewpoint input still requires a compiled grasp. See the
[active-perception design](active-perception-design-options-2026-09-05.md).

## Regression policy and validation

Formal standard batch uses Luna only, with historical per-task budgets:
15,000,000 cumulative tokens, 160 planner turns, 320 tool calls and 10,800 seconds.
The new Object 0 formal manifest records these values; small diagnostic runs
remain separately labeled. Provider usage across roles and outer simulator
lifetime must be checked independently; this budget is not a provider billing
cap. No paid model or new simulator experiment was launched in this slice.

Validation:

- Last full local suite: **2379 passed, 25 skipped, 1 existing authority/catalog
  failure, 36 warnings**, 66.41 seconds. JUnit:
  `tmp/bundle-ik-delivery-tests.xml`; log: `tmp/bundle-ik-delivery-tests.log`.
  Real external model integration was disabled explicitly. Local HTTP/SSE
  fixtures ran with permission to bind loopback ports.
- The only failure remains
  `test_reviewed_tool_contract_migration_is_complete_and_narrowly_authoritative`.
  New catalog changes do not inherit the historical approval hash. No approval
  hash, completed-review assertion or external sign-off was forged.
- Five generated projections are current, structural issues are empty and the
  three existing typed chains are compatible. These static checks do not certify
  the new bundle branch as physically successful. Audit:
  `tmp/bundle-ik-delivery-audit.json`.
- After the full suite, added a defensive nested-bundle rejection and its test;
  the final bundle suite is **19 passed**. It covers canonical model-context
  visibility, exact legacy resolver reuse, session resume/staleness, wrong type,
  tampering, Agent namespace forgery, Python read-only access, failure receipt
  rejection, and schema/planner parity for the new reference shapes.
- LIBERO Python numerical suite: **5 passed** (`tests/test_ik_search_budget.py`).
  Four numerical cases skip in the main Python environment, which lacks SciPy;
  they ran in the existing LIBERO environment. These use synthetic FK, not a
  newly started LIBERO simulation. No new dependency was installed.
- Existing candidate/calibration compilation test now runs both legacy and
  bundle entry paths; existing alignment/viewpoint tests check separate intents.
- Formal manifest validate-only, compileall and `git diff --check` passed.

Synthetic FK tests validate solver semantics, not actual LIBERO pose coverage
or end-to-end task success. Representative known-reachable LIBERO poses and
fresh standard batch remain required acceptance work.
