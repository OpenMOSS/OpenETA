# Rollout Data Contract

OpenETA persists four session-local data layers. They are intentionally
separate because they have different consumers and retention semantics.

| Layer | Purpose | May compact or summarize |
| --- | --- | --- |
| `trace.jsonl` | Runtime debugging and behavioral audit | Yes |
| `conversation.jsonl` | Canonical model-visible history and resume | Yes |
| `working/*.json` | Mutable facts, artifacts, and skill notes | Yes |
| `rollout/` | Immutable evidence for evaluation and training export | No |

The rollout recorder is enabled automatically when the runtime uses a
filesystem-backed `JsonMemoryStore`. Recording is best-effort and must never
interrupt robot, simulator, planner, or tool execution.

## Bundle Layout

```text
.openeta_memory/sessions/<session-id>/rollout/
  manifest.json
  model_calls.jsonl
  tool_calls.jsonl
  transitions.jsonl
  episodes.jsonl
  artifacts.jsonl
  artifacts/<sha256-prefix>/<sha256>.<ext>
```

All JSONL rows carry a schema version and a stream-local monotonic `seq`.
Artifacts are content-addressed by SHA256 and referenced by relative bundle
path. Repeated media is stored once.

## Recorded Evidence

`manifest.json` records:

- session task and metadata;
- git commit, dirty-state hashes, Python, and platform;
- planner and prompt metadata;
- full skill snapshots and content hashes;
- complete visible tool contracts and binding status.

`model_calls.jsonl` records every validation attempt, including rejected
attempts:

- complete semantic planner request;
- exact provider request body and response envelope when exposed by the
  OpenAI-compatible backend;
- provider fallback attempts, timing, model, and endpoint role;
- raw completion, parsed decision, full parameters/reasoning, and validation
  errors;
- ordered image references after externalizing inline data URLs.

`tool_calls.jsonl` records the exact start/end tool boundary emitted by the
tool registry. This includes parameters, structured result, supervision
metadata, diagnostics, state deltas, and artifact references.

`transitions.jsonl` records one
`observation -> action -> next_observation -> reward` row per runner turn:

- all objects and uncompressed robot state;
- all camera intrinsics, extrinsics, frame IDs, and timestamps;
- RGB as lossless PNG;
- depth as NPY with dtype, shape, unit, and scale metadata;
- full action, environment info, reward, terminal flags, and timing.

`episodes.jsonl` records the initial observation and episode result so reset
state and zero-step/interrupted episodes remain observable.

## Security

API keys, authorization headers, cookies, passwords, and common bearer/key
patterns are redacted before persistence. Inline image data is decoded into
content-addressed artifacts instead of being duplicated in JSONL. Recorder-only
raw provider exchange fields are removed before planner metadata enters the
normal trace and conversation layers.

Rollout bundles still contain user instructions, scene images, robot state, and
model outputs. They must be treated as training data, not ordinary logs, and
should follow the deployment's data retention and access policy.

## Training Export

Training code should consume an explicit exporter rather than training directly
against these raw files. The exporter is responsible for:

- joining model calls, tool outcomes, transitions, and episode reward;
- selecting accepted successful outputs for SFT;
- constructing rejected/accepted pairs from validation retries for preference
  training;
- filtering interrupted, assisted, unsafe, or schema-incompatible samples;
- resolving content-addressed media and verifying every SHA256;
- emitting a versioned model-specific dataset schema.

The raw rollout bundle remains model-agnostic evidence so future exporters can
derive planner VLM, grounding VLM, or action-policy datasets without changing
runtime recording.

Before export, call
`agent.runtime.rollout.validate_rollout_bundle(<session>/rollout)`. It checks
the manifest schema, every JSONL sequence, artifact presence, and artifact
SHA256 rather than silently accepting a partial or corrupted bundle.
