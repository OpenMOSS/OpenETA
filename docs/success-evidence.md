# Objective success evidence

Local refactor candidate, 2026-09-06. Shared checker/result semantics and the
serialization changes below require collaborator review before integration.
This is not a report of new simulator successes or a historical-score rewrite.

## One reduction, multiple consumers

`agent/runtime/success_evidence.py` reduces host environment/checker evidence for
live episodes and serialized episode records. Parallel classification, experiment
candidate collection, evaluation reports, visual-history analysis, calibration
batch statistics, exact-task playbook extraction/review, and RoboCasa batch
conversion use this reduction. The planner's official-completion gate uses the
same rules for the latest receipt. The memory ledger no longer labels arbitrary
positive reward `PASS`; incomplete, nonterminal reward remains `UNKNOWN`.

| Evidence | Objective success |
| --- | --- |
| Generic positive reward, including a trusted official shaped reward | No |
| Generic host adapter/checker boolean success flag | Yes, unless contradicted or superseded by failure |
| Generic flag with `require_official_reward=true` | Requires a valid same-execution/session reward receipt as well; reward need not be positive |
| LIBERO or registered `openeta/robocasa_{pretrain,target}_…-v0` | Valid receipt with numeric reward exactly `1`, `terminated=true`, `truncated=false` |
| Resource failure, episode/step truncation, explicit failed completion, or human wait | No; earlier success cannot erase these conditions |
| Agent `task_complete` declaration | Not objective evidence |

The known binary policies come from the local LIBERO adapter and
`sim/envs/robocasa/direct_env.py`, whose `step` replaces raw reward with `1/0`
from `_check_success()`. They do not cover RoboCasa's vector training wrapper,
arbitrary external RoboCasa environments, or all reward-bearing backends.
Unknown reward semantics fail closed; new policies need explicit adapter evidence
and tests, not an Agent-supplied success threshold.

Accepted flags are the existing `task_success`, `environment_success`,
`checker_success`, and `benchmark_success` booleans in `StepResult.info`.
Contradictory flags in one packet are not positive evidence. The latest explicit
outcome wins; an intermediate false flag may precede a later true flag. Arbitrary
tool `success`, motion `reached_target`, Agent notes, and model checker prose are
not read by this reducer. Rejected receipt projections cannot masquerade as native
adapter info.

## Receipt and execution binding

Reward evidence requires the host trust marker, `official_reward=true`, receipt
schema `openeta.environment_receipt.v1`, nonempty matching execution/session IDs,
`reward_present=true`, identical finite numeric rewards, and matching boolean
terminal flags. Strings, booleans used as numbers, NaN, infinity, and mismatched
or prior-execution receipts are not accepted. `official_reward` establishes reward
origin/presence; it is not itself a task-success assertion.

The runtime binds host memory metadata to the current execution before planning.
Restarting a run in the same session does not authorize that session's previous
completion receipt. Episode results retain `env_id` and `require_official_reward`
metadata for offline consumers.

`EpisodeStep.to_dict()` retains all eight success-critical receipt fields:

```json
{
  "schema_version": "openeta.environment_receipt.v1",
  "receipt_id": "receipt-example",
  "execution_id": "execution-example",
  "agent_session_id": "session-example",
  "reward_present": true,
  "reward": 1.0,
  "terminated": true,
  "truncated": false
}
```

These fields and the surrounding success/trust flags bypass generic eight-key
metadata compaction. Camera snapshots and large tool responses are not copied
into this projection. Tests compare actual host-receipt live classification with
the result after `EpisodeResult.to_dict()`.

## Compatibility and limits

- The batch v2 status enum remains unchanged. A non-official run can still report
  runtime completion through `task_complete`; objective metrics and training
  candidate selection do not count that declaration as proof. Explicit failure
  reports remain permitted by the planner gate and classify as failure.
- Exact-task playbook schema still requires positive official reward evidence
  *supporting proven success*. A generic checker-only or zero-reward success can
  count in objective metrics but does not satisfy that stricter playbook schema.
- Old compacted records missing required fields are not silently grandfathered
  into trusted success. RoboCasa conversion rejects them; recover full original
  host evidence or report unverifiable evidence. No historical records were edited
  and no historical success rates were recalculated by this batch.
- This validates host-owned evidence and consistent serialized projections, not
  cryptographic authenticity of user-edited JSON. Native adapter/checker
  `StepResult.info` remains a trusted integration boundary. General per-backend
  checker policy registration and tamper-evident external result import remain
  outside this implementation.
- Positive reward may still trigger a self-improvement *review* or appear as
  numeric progress information. It cannot by itself approve a successful task
  playbook, calibration rollout, or experiment candidate. Specialized legacy
  canary scripts are not all migrated; inspect their backend-specific semantics
  before using them as cross-backend success metrics.

Regression entry points: `tests/test_success_evidence.py`,
`tests/test_tool_feedback_episode_environment.py`, `tests/test_robocasa_benchmark.py`,
and the experiment/playbook/self-improvement tests. Full validation counts and
remaining failures are recorded in the [refactor log](harness-refactor-progress-2026-09-05.md).
