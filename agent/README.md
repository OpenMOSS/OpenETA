# Agent Module

`agent/` owns code-agent runtime integrations.

The general OpenETA runtime is the lightweight Python package under
`agent/runtime/`. It keeps the agent loop explicit: memory, tools, skills,
planner, runtime, and adapter conversion to the sim/env `EnvAction` payload.

Naming convention:

- `Skill` / `TaskSkill`: task-level markdown guidance such as `pick`, `place`,
  `push`, `pull`, or `stack`.
- `Tool` / `AgentTool`: agent-callable capability registered as `ToolSpec`.
- `AtomAction`: embodied physical primitive, represented as the world-mutating
  control-tool subset of `ToolSpec`.
- `AgentCommand`: agent runtime decision. The schema uses `CommandKind` with
  top-level `tool_call` and `response`; historical top-level action kinds are
  not accepted.
- `SlashCommand`: CLI user command such as `/provider`, `/run`, or `/step`.
- `EnvAction`: structured sim/env action payload sent to the simulator adapter.

Current runtime pieces:

- `agent.runtime.memory.AgentMemory`: session event log plus working facts,
  artifacts, skill notes, and compact summary for planner context.
- `agent.runtime.memory_store.JsonMemoryStore`: optional local persistence.
  Each local session owns `.openeta_memory/sessions/<session_id>/trace.jsonl`
  plus `.openeta_memory/sessions/<session_id>/working/*.json` for facts,
  artifacts, skill notes, and compact summary. Session-local working memory is
  not shared across sessions; reviewed long-term project memory belongs under
  `agent/memory/`. The CLI uses this store by default, and `.openeta_memory/`
  is ignored by git. Pre-session-scoped traces from the old
  `.openeta_memory/sessions/<session_id>.jsonl` layout are migrated into the
  new per-session directory on startup. The old global `.openeta_memory/working/`
  directory is archived under `.openeta_memory/legacy/working/` because its
  session ownership is ambiguous.
- `agent.tools.registry.ToolRegistry`: perception, manipulation, navigation,
  control, memory, and skill-management tool metadata. Tools are stable atomic
  capabilities; `pick` and `place` are not tools. World-mutating control tools
  are the `AtomAction` subset. Each tool declares its side-effect class so the
  pipeline can decide whether it may be batched before the next observation.
- `agent.tools.registry.ToolResult`: normalized handler result envelope. Every
  executable handler result is normalized to
  `schema_version=openeta.tool_result.v1` with `result_type`, `outputs`,
  `artifacts`, `state_delta`, and `diagnostics`.
- `agent.tools.handlers.bind_dummy_tool_handlers`: deterministic dummy
  handlers for common perception, planning, safety, and control tools. CLI and
  tests use these until real simulator/perception/control handlers are wired.
- `agent.runtime.checkers.CheckerSubagentConfig`: optional pre-tool safety and
  post-tool failure checker hooks. Agent-requested `safe_check` remains an
  explicit planning/preview `tool_call`; checker hooks are runtime execution
  gates around configured tools. CLI pre-check gates are opt-in through
  `OPENETA_PRE_SAFETY_CHECKS`; failed tools emit compact recovery feedback for
  the next planner turn. Checker outputs stay in pipeline metadata as
  sub-agent placeholders until the final checker schema is reviewed.
- `agent.runtime.self_improvement.SelfImprovementReviewer`: post-episode
  review hook for stable skill learning. It delegates to a restricted
  `SkillReviewSubagent` after useful signals such as many tool calls, failures,
  truncation, or positive reward. The first implementation writes pending
  proposal JSON under `.openeta_memory/skill_reviews/pending/`. Skill markdown
  edits require an explicit human approval step through `/skill-reviews`,
  `/skill-review <id>`, `/approve-skill-update <id>`, or
  `/reject-skill-update <id>`.
- `agent.runtime.skills.SkillRegistry`: editable text-guidance documents such
  as pick, place, push, pull, and stack. A skill can recommend a tool sequence,
  but the runtime never auto-expands it into hidden tool calls. Built-in skill
  markdown files live under `agent/skills/*.md` and are loaded into the registry
  at runtime.
- `agent.runtime.planner.ToolCallingPlanner`: default planner bridge for the
  closed-loop pattern `observe -> tool(parameter) -> result -> observe`. It can
  call a `PlannerBackend`, validate the returned JSON command request, and
  retry once with validation feedback before falling back to
  `response::ask_human`. Planner context uses `PlannerContextConfig` to keep
  memory and skill guidance bounded: `skill_references` is a metadata index,
  while `selected_skill_guidance` contains the matched markdown bodies.
- `agent.backends.planner.PlannerBackend`: LLM/VLM backend boundary for
  closed-loop tool selection. The current package includes a placeholder backend
  and a deterministic `StaticPlannerBackend` for tests and local smoke runs.
  `CallablePlannerBackend` adapts provider SDK/API wrappers that return JSON
  decision payloads.
- `agent.backends.provider_config.PlannerProviderConfig`: API provider settings
  model for CLI and future GUI configuration. It can load `.env` or a local
  `apikey.md`, validate missing fields, redact secrets for display, and write a
  `.env` file.
- `agent.backends.planner.OpenAICompatiblePlannerBackend`: real
  `/v1/chat/completions` backend for OpenAI-compatible providers. When a SAM3
  selection obligation is pending, it attaches the original image and candidate
  contact sheet as bounded multimodal image parts. Provider timeouts, connection
  failures, HTTP 408/429, and selected HTTP 5xx responses use bounded exponential
  backoff before falling back to `response::ask_human`; request and schema errors
  are not retried.
- SAM3 multi-candidate selection is explicit: runtime memory persists a
  `selection_obligation`, the main VLM calls `select_sam3_detection`, and the
  pipeline blocks targeted AnyGrasp or world-mutating tools until the selected
  mask is recorded.
- AnyGrasp multi-candidate handling is greedy and stateful: memory exposes rank
  0 as `grasp_candidate_policy.active_candidate`; candidate-linked safety or
  motion rejection advances to the next score-ranked pose, while successful
  `move_to` accepts the queue and releases its downstream gate.
- `agent.runtime.episode.OpenEtaEpisodeRunner`: multi-step closed-loop runner
  for `observe -> plan -> tool_call -> tool result -> memory update -> observe`
  episodes. `ToolFeedbackEpisodeEnvironment` feeds bound-tool summaries into
  the next CLI planner turn when no simulator-owned episode environment is
  active; `DummyEpisodeEnvironment` remains a test compatibility subclass.
  The runner owns separate resource budgets: 50 concrete tool calls, a
  600-second wall-clock deadline, and 5,000,000 cumulative model tokens, plus a
  compatibility `max_turns=100` guardrail. The agent can end an episode with
  `response::task_complete` or explicit completion parameters, while
  env/checker feedback can force `terminated`/`truncated`. Each turn runs in a
  daemon worker behind the remaining episode deadline; timeout abandons the
  turn, prevents late step commit, and requests environment cleanup.
- `agent.runtime.parallel.ParallelEpisodeHarness`: bounded thread-pool harness
  for independent simulator episodes. It defaults to 10 concurrent workers,
  preserves a serial closed loop inside each worker, isolates failures, keeps
  manifest ordering, and always invokes worker cleanup.
- `agent.cli.batch_eval`: non-interactive `openeta-batch` entry point. It builds
  a separate planner/runtime/MCP environment and trace/artifact root per
  manifest entry so parallel runs do not share mutable session state.
- `agent.tools.registry.ToolExecutionContext`: context passed to executable tool
  handlers. Handlers can inspect parameters, tool metadata, the current
  observation, and pipeline metadata, then return a structured `ToolResult`.
- Runtime-owned memory tools: `save_memory`, `get_memory`, `delete_memory`, and
  `compact_memory` are bound by `OpenEtaAgentRuntime` and update the current
  `AgentMemory`. If a `JsonMemoryStore` is attached, these changes are also
  persisted to local working-memory JSON.
- `agent.runtime.planner.CodePolicyPlanner`: optional planner bridge for
  bounded Code-as-Policy snippets when an atomic tool backend needs generated
  code.
- `agent.backends.code_policy.CodePolicyBackend`: generation boundary for commercial
  API or local-model backends used by optional code-policy execution.
- `agent.runtime.env_facade.RlinfEnvFacade`: narrow OpenETA-facing control
  surface over constructed RLinf envs.
- `agent.runtime.sandbox.RlinfCodePolicySandbox`: simulator-side boundary for
  executing or dry-running generated code against RLinf-backed envs under
  `sim/`. RLinf-derived env classes are resolved through
  `sim.envs.get_env_cls`; recording instrumentation is adapter-owned because
  the current repository does not expose a shared wrappers package.
- `agent.runtime.planner.RuleBasedPlanner`: deterministic bootstrap/fallback
  planner for smoke tests only.
- `agent.runtime.runtime.OpenEtaAgentRuntime`: runtime owner used by
  `adapter.openeta_agent.OpenEtaAgentAdapter`.
- `agent.runtime.actions` and `agent.runtime.pipeline`: structured
  agent-command schema and safe/tool/skill compilation pipeline. The primary
  class names are `CommandKind`, `CommandRequest`, and `CommandPipelinePlan`.
- `agent.runtime.interfaces`: reserved execution interfaces for command
  subtypes. `skill_call`, `safe_check`, `code_policy`, and `sense` are named
  `tool_call` capabilities, while `ask_human`, `talk`, and `task_complete` are
  `response` subtypes.

`agent/memory/` is reserved for curated project memory that should be reviewed
and committed. Automatic traces and local working state must stay in the
gitignored `.openeta_memory/` directory.

## Provider Smoke Test

Create `.env` from `.env.example`, or place a local ignored `apikey.md` with a
newapi channel JSON object. Then run:

```bash
uv run python examples/openai_compatible_planner_smoke.py --list-models
uv run python examples/openai_compatible_planner_smoke.py --model gpt-5.4-mini
```

The smoke test asks the model for one closed-loop action and executes registered
dummy handlers for read-only tools such as `sam3`.

## Local Provider GUI

OpenETA has a minimal local GUI for the same provider configuration path:

```bash
uv run python -m agent.gui.provider_config_app
```

It serves a local-only page with provider/API base/API key/model fields, model
listing, `.env` saving, and a planner smoke test. The server only returns
redacted secrets to the browser.

## Terminal Agent Console

The preferred developer-facing interface is a terminal REPL:

```bash
uv run openeta
```

Type `/` at the `›` prompt to open the slash-command popup, then use the
keyboard to select commands such as `/provider`, `/model`, `/models`, `/tools`,
`/sessions`, `/resume`, `/new`, `/run`, `/step`, and `/quit`. Normal task text
runs one closed-loop agent turn in the current session; `/new` starts a fresh
session. `/resume` opens the local session picker, while
`/resume <session_id>` or `/resume --last` restores a local session-scoped
trace and working memory directly. Each turn prints planner usage,
request, reasoning, compact parameters, tool calls, and a display-only result
summary. Tool results are projected to key diagnostics, semantic outputs, and
artifact paths, then bounded to five terminal rows. The complete structured
result remains in the session trace, working memory, and planner context.
`response::ask_human` prompts in the terminal and automatically resumes the same
episode runner after recording the answer. The dummy world-mutating control
handler asks for explicit approval before returning a result. Skill
self-improvement proposals remain staged in `.openeta_memory/skill_reviews/pending/`
until the user inspects and approves them with the skill-review slash commands.
