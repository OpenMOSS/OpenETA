# Grasp Strategy Lifecycle

OpenETA separates embodiment calibration from task-family grasp strategy.

An embodiment calibration answers stable hardware questions: which robot,
gripper, controller, camera calibration, and grasp backend are compatible, and
how a grasp-frame pose maps to the robot EEF. It is selected by host-owned
environment metadata and remains read-only during an episode.

A grasp strategy contains revisable task evidence: geometry families for
automatic activation, width bounds, approach policy, orientation policy, and
validation scope. A strategy cannot change the underlying calibration transform
or exceed the calibration's physical gripper limit.

## Runtime Selection

```text
environment fingerprint
  -> one embodiment calibration
truthful target geometry family + optional explicit strategy_id
  -> compatible validated strategy
  -> compatible candidate strategy
  -> generic calibrated fallback
```

No strategy match is an expected state, not a grasp rejection. Generic fallback
preserves the normalized estimator orientation and approach while retaining
camera/world provenance and hardware width checks. An explicit incompatible or
unknown strategy ID fails closed.

Parallel workers copy repository strategies into:

```text
<session-workspace>/strategies/grasp/
  candidate/*.json
  validated/*.json
```

These copies are session-owned and writable so reviewed-autonomy experiments do
not race on shared files. The repository copies are only shared baselines.

## Automated Review And Promotion

`GraspStrategyLifecycleManager` provides two bookkeeping tools:

- `propose_grasp_strategy` validates against the host-staged calibration,
  invokes an independent clean-context reviewer, and writes only to the session
  proposal area. It never changes the current episode.
- `promote_grasp_strategy` reads host-generated paired evidence and publishes
  through a file-locked compare-and-swap boundary.

Every proposal and evidence file binds both `strategy_sha256` and
`calibration_profile_sha256`. Candidate publication requires paired canary
non-regression. Validated publication additionally defaults to at least 20
held-out attempts, 95% objective success, two held-out tasks, no regressed
episode IDs, no human intervention, and zero safety or contract violations.
Model approval cannot bypass these gates.

| Profile | Session proposal | Shared publication |
|---|---|---|
| `standard` | Independent review may stage it | Denied |
| `human_gated` | Independent review may stage it | Explicit human approval |
| `reviewed_autonomy` | Independent review may stage it | Objective gates plus independent reviewer |

`openeta --command iterate --approvement reviewed_autonomy` validates strategy
and skill changes in separate lanes. A clean strategy author proposes at most
one revision. The harness compares baseline and candidate with identical tasks
and seeds, then performs held-out validation. Only an accepted strategy tree
becomes the next generation baseline. Skill validation runs afterward on that
fixed tree, keeping attribution separate.

Parallel workers stage proposals under their own session memory. Shared
publication is performed only by the lifecycle publisher under
`.grasp-strategy-publish.lock`; workers never overwrite repository strategies
directly. Remote Git publication remains a separate branch or PR operation.
