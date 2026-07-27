---
name: memory_extract
description: Guidance for extracting useful working memory during long closed-loop episodes.
version: v1
editable: true
task_patterns:
  - remember useful context
  - extract memory
  - compact memory
  - summarize episode context
allowed_tools:
  - get_memory
  - save_memory
  - compact_memory
---
# Memory Extract

Use this skill as text guidance only. It is not executable code and must not be
expanded into hidden tool calls.

Use this skill when an episode is getting long, repeated failures reveal a useful
lesson, or a tool result produces information that later planner turns should
reuse.

## What To Save

- Save stable task facts under `facts`, such as the current target object,
  target receptacle, user constraint, or selected scene frame.
- Save reusable tool outputs under `artifacts`, such as segmentation ids,
  grasp candidate ids, validated poses, or checker diagnostics.
- Save task-level lessons under `skill_notes`, keyed by the relevant skill name,
  such as `pick`, `place`, `push`, `pull`, or `stack`.

## Recommended Tool Sequence

1. Call `get_memory` if you need to inspect existing working memory before
   writing.
2. Call `save_memory` with a concise key and JSON payload. Prefer small,
   durable facts over raw logs.
3. Call `compact_memory` when the recent episode trace is long or when the
   planner context budget is close to its automatic compaction threshold.

## Boundaries

- Do not write directly to `agent/memory/`. That directory is reserved for
  reviewed, curated project memory.
- Do not store full images, full point clouds, raw traces, or large tool outputs
  in working memory. Store ids, summaries, and paths instead.
- Do not use memory extraction as a substitute for fresh observation after
  world-mutating tools.
