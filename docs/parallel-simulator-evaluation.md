# Parallel Simulator Evaluation

`openeta-batch` runs independent model-planned simulator episodes concurrently.
It is an evaluation harness around the existing single-episode closed loop; it
does not batch world-mutating tool calls inside an episode and does not change
the `CommandRequest` or `CommandPipelinePlan` contract.

## Usage

```bash
uv run openeta-batch \
  --manifest examples/parallel_libero_eval.json \
  --concurrency 10 \
  --approvement reviewed_autonomy \
  --output outputs/parallel-libero.json
```

Validate configuration without model or MCP network calls:

```bash
uv run openeta-batch \
  --manifest examples/parallel_libero_eval.json \
  --concurrency 10 \
  --validate-only
```

Before spending model calls or moving robots, the same manifest can exercise
only concurrent simulator lifecycle capacity:

```bash
uv run python examples/parallel_simulator_capacity_smoke.py \
  --manifest examples/parallel_libero_eval.json \
  --concurrency 10
```

This creates and resets every environment, emits an immediate static
`task_complete`, and closes every handle. It does not execute control actions.

`--validate-only` validates the local manifest without model or MCP network
calls. Normal runs resolve and call the configured MCP tools when each episode
worker starts; they do not require a grasp-specific extension to the remote
`move_to` schema.

Concurrency defaults to 10 and is bounded to 1–32. The effective worker count
is also capped by the number of manifest entries.

## Unattended Command Workflow

`openeta --command` reuses the same episode worker and cleanup boundary without
starting the interactive console:

```bash
uv run openeta --command preflight \
  --manifest examples/parallel_libero_eval.json

uv run openeta --command run \
  --manifest examples/parallel_libero_eval.json \
  --experiment-id libero-grasp-v1 \
  --concurrency 4 \
  --provider-concurrency 2 \
  --episode-id libero-object-0 \
  --episode-id libero-object-4 \
  --approvement reviewed_autonomy \
  --on-need-human fail

uv run openeta --command iterate \
  --train-manifest examples/parallel_libero_train.json \
  --validation-manifest examples/parallel_libero_holdout.json \
  --experiment-id libero-grasp-v1 \
  --rounds 3 \
  --concurrency 10 \
  --approvement reviewed_autonomy \
  --on-need-human fail

uv run openeta --command inspect --experiment-id libero-grasp-v1
```

`preflight` validates the manifest, provider configuration, configured MCP
endpoints, simulator `list_tools` catalog, skill baseline, and main Planner
prompt hash without creating an environment. Add `--require-perception` when
SAM3 and AnyGrasp must both be configured. `run` executes one training
generation and records candidate diffs. `iterate` requires
`reviewed_autonomy`; each generation performs a training batch and independently
authors/reviews at most one grasp-strategy revision. Strategy canary and held-out
comparisons run before skill validation so attribution stays separate. Skill
diffs are still collected only from episodes with positive reward or explicit
checker success. Promotion requires objective success, no paired episode
regression, no lower runtime success count, and no increase in failures or
human requests.

Shared strategy candidate publication is enabled by default for `iterate`; use
`--no-publish-grasp-strategies` for experiment-local evolution only. Validated
defaults are configurable with `--strategy-min-canary-attempts`,
`--strategy-min-held-out-attempts`,
`--strategy-min-held-out-success-rate`, and
`--strategy-min-held-out-tasks`.

The lineage layout is:

```text
.openeta_memory/experiments/<experiment_id>/generations/000/
  baseline_skills/
  baseline_grasp_strategies/
  train/sessions/<agent_session_id>/
  train/result.json
  candidates/
  grasp_strategy_candidates/
  grasp_strategy_lifecycle/
  proposed_grasp_strategies/
  proposed_skills/
  validation-baseline/
  validation-candidate/
  result.json
```

Every session starts from the generation's immutable baseline. Session-local
changes never write shared `agent/skills/*.md`; a successful promotion only
seeds `baseline_skills/` for the next generation. By default an unresolved
`ask_human` in command mode becomes
`need_human_in_unattended_run`, cleanup still runs, and no resumable pause
record is created.

Environment concurrency (`--concurrency`) is separate from provider concurrency
(`--provider-concurrency`, default 2). All model clients created for one batch,
including the main Planner, action reviewer, guidance resolver, skill author,
and skill reviewer, share one provider semaphore. A slot remains held across
the provider backend's retry sequence. `--provider-queue-timeout-s` defaults to
180 seconds; expiry produces `provider_queue_timeout` before a late API request
can start. Batch results include the limit, request and timeout counts, total
and maximum queue wait, and observed maximum active calls.

Repeated `--episode-id` arguments select a stable ordered subset for canary
runs and fail if an id is absent from the manifest. `run` prints metrics,
failures, provider concurrency, and `result_path` by default. The complete batch
is always written beneath the experiment generation; `--print-full-result`
restores the previous full stdout payload.

## Current Observation Images

Each planner turn derives `current_camera_artifacts` exclusively from the
current `EnvObservation.metadata.image_artifacts`. Exact RGB and depth entries
are exposed, ordered by `agentview`, `render`, `wrist`, then other frames, with
RGB before depth within a frame. Only the primary exact RGB path is placed in
`vision_image_paths`, so depth is available as a structured AnyGrasp parameter
without being attached as a VLM image. Reference localization or SAM3 selection
may add controlled images for their dedicated model request. Object-memory
localization uses an isolated four-image request (scene plus three reference
views); after it returns a validated point, those references are not injected
into the main planner turn.

SAM3 should receive the exact local RGB path. For compatibility, its handler
accepts a frame id only when that id resolves to an RGB artifact in the same
current observation. It never searches another session or a global temporary
directory. The canonical resolved path is retained as `source_image`; AnyGrasp
continues to consume the original full-frame RGB, depth, and same-size mask.
When `positive_points` is present, the SAM3 handler sends the original RGB to
the deployed `segment_points` tool and preserves all three ranked full-frame
masks for the normal explicit selection step.

The agent-side AnyGrasp candidate queue consumes the existing normalized motion
summary without changing the remote Sim MCP schema. An explicit
`motion_summary.reached_target=false` rejects the active pose and activates the
next score-ranked candidate, even when a legacy remote response has no error and
the compatibility ToolResult remains successful. Transport errors and motion
failures without this structured evidence do not advance the queue.

## Manifest

The file may be a JSON list or an object containing `episodes`:

```json
{
  "episodes": [
    {
      "episode_id": "libero-object-0",
      "env_id": "openeta/libero_libero_object_task0-v0",
      "task": "pick up the alphabet soup and place it in the basket",
      "seed": 0,
      "max_turns": 100,
      "max_tool_calls": 200,
      "timeout_s": 3600,
      "max_total_tokens": 5000000,
      "metadata": {"suite": "libero_object"}
    }
  ]
}
```

`task` and `env_id` are required. `episode_id` defaults to an ordered generated
id and must be unique. `seed` defaults to 0. Resource defaults are 100 planner
turns, 200 concrete tool calls, 3,600 seconds of wall-clock episode time, and
5,000,000 cumulative model tokens. Validation retries count toward token usage.
Tool-call and token usage stay cumulative across `need_human` environment
restarts for the same Agent session; the 3,600-second clock restarts with each
new simulator environment.

## Isolation and Cleanup

Each worker owns:

- one planner backend and `OpenEtaAgentRuntime`;
- one simulator MCP session and environment handle;
- one `SimulatorMcpEpisodeEnvironment`;
- one `.openeta_memory/sessions/<session_id>/` root;
- one private skill snapshot under `sessions/<session_id>/skills/`;
- one trace, conversation, working-memory, and rollout bundle directly under
  that session root;
- one artifact tree under `sessions/<session_id>/artifacts/`;
- one writable Python sandbox root under `sessions/<session_id>/sandbox/`.

The Python sandbox's exposed file APIs resolve relative paths under the session
workspace and reject path escape. `outside_sandbox` remains a separate host
executor and is disabled for batch workers; reviewed autonomy does not grant it.

The thread pool isolates worker exceptions and waits for every submitted job.
Each worker calls `close_env` from `finally`; cleanup failure is reported as
`cleanup_failed` without hiding the episode result or primary exception.

## Result Semantics

The aggregate uses `schema_version=openeta.parallel_episode_batch.v2` and keeps
outcomes in manifest order. Lifecycle status has exactly three values:

- `success`: terminal success;
- `need_human`: resumable non-terminal pause caused by `ask_human`;
- `fail`: terminal failure, including execution/cleanup failure or a precise
  resource failure: `tool_call_limit_exceeded`, `episode_timeout`, or
  `token_limit_exceeded`.

The 100-turn guard is deliberately separate from the 200-tool-call budget:
skill inspection and response turns do not consume tool calls. An episode
fails once observed concrete tool attempts exceed 200. Timeout includes environment
creation/reset and all model/MCP work. Each turn runs in a daemon worker for the
remaining deadline; timeout immediately marks the episode failed, requests
`close_env`, and refuses to commit a late `EpisodeStep`. Simulator close is
handle-claiming and thread-safe, so interrupt cleanup and the worker's `finally`
cannot issue duplicate `close_env` calls.

Provider `usage.total_tokens` is authoritative when positive. If only prompt
and completion fields exist, OpenETA derives the total. If usage is absent or
invalid, it reuses `agent/runtime/token_counting.py`: `tiktoken` when available,
otherwise the conservative character-ratio estimator. Traces label accounting
as `provider`, `provider_derived`, or `estimated`, and validation retries remain
part of cumulative session usage.

Every outcome also includes `assistance.assisted`,
`assistance.guidance_intervention_count`, and
`assistance.human_intervention_count`. Aggregate metrics report autonomous,
agent-assisted, and human-assisted successes separately as well as total
success, pending-human, and failure counts. A
`success` label currently follows the agent/runtime terminal
signal; benchmark-grade task success should additionally use backend reward,
a task checker, or recorded visual/state evidence.

## Supervision and Resume

`--approvement` selects one host-owned profile:

| Profile | World-mutating tools | Skill changes | `ask_human` |
|---|---|---|---|
| `human_gated` | Actual human approval | Actual human approval | Actual human answer |
| `standard` | Deterministic runtime/checker gates | Session skill/strategy proposals only | Human answer |
| `reviewed_autonomy` | Independent clean-context reviewer plus deterministic gates | Clean author, second reviewer, paired evidence, controlled publication | Guidance client, then human on abstention |

The profile is supplied by the host CLI and is not an agent tool. A higher
autonomy profile does not disable IK, collision, backend, or failure checks.
The action reviewer receives the current scene attachment plus selection and
active AnyGrasp candidate provenance. Because some simulator adapters leave
`observation.objects` empty, that empty list alone is not negative evidence;
the reviewer still rejects contradictory images, missing provenance/frame, or
planner-edited poses. A runtime-activated fallback candidate with an unchanged
transformed pose remains reviewable after structured target-not-reached feedback.

In reviewed autonomy, a guidance answer is recorded as
`latest_guidance_interaction` with `source=guidance_agent`. It is not written to
`latest_human_interaction`. Resolution occurs inline while the environment is
active. A repeated question, exhausted guidance budget, reviewer error, or
explicit abstention falls through to the normal pause contract below.

The reviewed-autonomy Skill Author uses a 4096-token output budget in both the
interactive and parallel paths. The independently created reviewer client keeps
the bounded 512-token response budget because it returns only a decision and
reason. Before relying on a provider/model combination for unattended runs, use
`openeta-subagent-eval --strict`; the live suite calls the same production
sub-agent adapters and reports dangerous false approvals separately.

`need_human` returns a `session_id`, unique `interaction_id`, and question. A
paused record is written atomically under
`.openeta_memory/parallel_interactions/`. The record contains the task identity
and Agent memory location, not a simulator handle. The worker immediately calls
`close_env`, because the MCP backend's 10-minute connection lifetime is shorter
than a realistic human response cycle.

```bash
uv run openeta-batch \
  --resume-session SESSION_ID \
  --interaction-id INTERACTION_ID \
  --answer "Pick the red cube"
```

Resume rejects stale interaction ids and writes the answer to the original
Agent session workspace and memory. It then creates and resets a new environment with the
same `env_id`, task, and seed and retries from the task's initial state with a
fresh `max_turns` budget. Physical simulator state is not continued. If the
agent asks again, the restarted environment is also closed and the same Agent
session receives a new interaction id. Terminal success/failure closes the new
environment and deletes the paused record.

The batch harness is single-process and thread-based. It enables concurrent
remote simulator/model I/O and bounded text-skill iteration, but it is not
distributed rollout infrastructure or model-weight training.
