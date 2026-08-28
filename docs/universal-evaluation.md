# Universal Evaluation Entry

`openeta-eval` is the durable evaluation control plane for OpenETA. It is
experiment-agnostic: it reliably produces isolated, reproducible rollout
bundles and generic execution metrics. Research-specific interpretation lives
in separate extractors that consume those bundles.

## Four layers

1. **Evaluation plan** — declares episodes, paired runtime variants, repeats,
   deterministic schedule seed, and execution budgets. Compilation expands it
   into stable jobs and hashes the fully resolved inputs.
2. **Durable scheduler** — stores every attempt before execution, atomically
   commits each settled outcome, retries only structured infrastructure
   failures, and resumes attempts left incomplete by a dead process.
3. **Episode runtime** — reuses `ParallelEpisodeHarness`,
   `OpenEtaEpisodeRunner`, shared runtime assembly, isolated session workspaces,
   provider admission control, and mandatory cleanup.
4. **Generic reducer** — reports runtime/objective success, failure class and
   stage, retry count, latency, throughput, and provider concurrency. It does
   not contain visual-history or benchmark-specific scoring rules.

The evaluator does not modify `CommandRequest`, `CommandPipelinePlan`, or the
existing `openeta.parallel_episode_batch.v2` outcome. It wraps each batch-v2
outcome in a durable attempt record.

## Commands

```bash
uv run openeta-eval validate \
  --plan evaluations/visual_history_abc.json

uv run openeta-eval preflight \
  --plan evaluations/visual_history_abc.json

uv run openeta-eval run \
  --plan evaluations/visual_history_abc.json \
  --run-id visual-history-canary-001

uv run openeta-eval inspect --run-id visual-history-canary-001
uv run openeta-eval resume --run-id visual-history-canary-001
uv run openeta-eval report --run-id visual-history-canary-001
```

`validate` performs no provider or simulator calls. `preflight` additionally
checks provider configuration and the simulator MCP tool catalog without
creating an environment; `run` performs the same preflight before allocating a
run. `run` refuses to reuse an existing run id. `resume` preserves already terminal jobs, marks orphaned
`running` attempts as abandoned, and restarts only unfinished or retryable
infrastructure attempts.

## Evaluation plan

Plans use `schema_version=openeta.evaluation_plan.v1`:

```json
{
  "schema_version": "openeta.evaluation_plan.v1",
  "plan_id": "example-v1",
  "episode_manifest": "manifests/tasks.json",
  "variants": [
    {
      "variant_id": "baseline",
      "runtime": {}
    }
  ],
  "repeats": 3,
  "schedule": {"shuffle_seed": 20260813},
  "execution": {
    "concurrency": 4,
    "provider_concurrency": 2,
    "provider_queue_timeout_s": 180,
    "supervision_profile": "standard",
    "max_attempts": 2
  }
}
```

`episode_manifest` is resolved relative to the plan and may be replaced by an
inline `episodes` list. Every `(episode, repeat)` defines a `pair_id`; all
variants share that pair identity and the same task and seed. Jobs are shuffled
with the declared schedule seed so provider/server drift is not confounded with
variant ordering.

`runtime` is transported by the generic plan compiler and interpreted by the
runtime assembly boundary. Unsupported runtime sections fail validation rather
than being silently ignored. Registered sections include `visual_history` and
the evaluation-only `grasp_strategies` projection. The latter can exclude a
strategy or strip its task-specific `canary_evidence` from the isolated session
snapshot; it never mutates the repository strategy tree.

## Durable layout

```text
.openeta_eval/runs/<run_id>/
  run.json
  compiled_plan.json
  journal.jsonl
  report.json
  jobs/<job_id>/
    final.json
    attempts/001/
      state.json
      result.json
      sessions/<session_id>/
        rollout/
          manifest.json
          model_calls.jsonl
          tool_calls.jsonl
          transitions.jsonl
          episodes.jsonl
          artifacts/
```

`compiled_plan.json` is immutable and includes the resolved episode list and
plan SHA-256. `run.json` records git/provider/prompt provenance without API
credentials. A job receives `final.json` only when successful, non-retryable,
or out of attempts. The session rollout remains the lossless evidence source.

## Failure and retry contract

The generic failure dimensions are:

- `infrastructure/provider`
- `infrastructure/environment`
- `infrastructure/cleanup`
- `resource_limit/episode`
- `task_failure/verdict`
- `agent_failure/episode`
- `human_intervention/agent`

Only structured infrastructure failures are retried automatically. Task,
Agent, resource-budget, and human-intervention outcomes remain evidence and are
not erased by automatic reruns.

## Visual-history ABC evaluation

[`evaluations/visual_history_abc.json`](../evaluations/visual_history_abc.json)
defines paired variants:

- **A** — current fixed-main-camera image only;
- **B** — initial and latest three main-view observations plus current wrist,
  without VDM;
- **C** — the same bounded raw window plus main-view VDM.

Run-specific visual metrics are intentionally separate:

```bash
uv run openeta-visual-history-rollout-extract \
  --run-id visual-history-canary-001
```

The extractor writes:

```text
extractors/visual_history/metrics.json
extractors/visual_history/state_probe_cases.jsonl
```

It derives false completion, repeated consecutive actions, raw visual evidence
size, compressed-delta coverage, VDM reliability/cost, and an offline state
probe dataset from recorded model calls and transitions. These fields do not
belong to the universal evaluator and can evolve with the research question.
