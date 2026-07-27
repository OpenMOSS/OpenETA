# Agent Module

`agent/` owns code-agent runtime integrations.

The primary OpenETA runtime is now the lightweight Python package under
`agent/runtime/`. It is RAS-inspired and keeps the first agent loop explicit:
memory, tools, skills, planner, runtime, and adapter conversion to the sim/env
`EnvAction` payload.

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

- `agent.runtime.memory.AgentMemory`: session event log, canonical conversation
  history, and working facts/artifacts/skill notes. Exact user messages are
  recorded automatically; `save_memory` is not required for same-session
  instruction continuity.
- `agent.runtime.memory_store.JsonMemoryStore`: optional local persistence.
  Each local session owns `.openeta_memory/sessions/<session_id>/trace.jsonl`
  as the append-only audit log,
  `.openeta_memory/sessions/<session_id>/conversation.jsonl` as the model-visible
  message/action/result projection with compaction checkpoints, and
  `.openeta_memory/sessions/<session_id>/working/*.json` for structured state.
  Persistent runtimes also create
  `.openeta_memory/sessions/<session_id>/rollout/` as a fourth, training-oriented
  evidence layer. It stores exact redacted model exchanges, tool boundaries,
  lossless environment transitions, run provenance, and content-addressed
  media without injecting any of that data back into planner context. See
  `docs/rollout-data-contract.md`.
  Session-local working memory is not shared across sessions; reviewed long-term
  project memory belongs under `agent/memory/`. The CLI uses this store by
  default, and `.openeta_memory/` is ignored by git. Pre-session-scoped traces from the old
  `.openeta_memory/sessions/<session_id>.jsonl` layout are migrated into the
  new per-session directory on startup. The old global `.openeta_memory/working/`
  directory is archived under `.openeta_memory/legacy/working/` because its
  session ownership is ambiguous.
- `agent.tools.registry.ToolRegistry`: perception, manipulation, navigation,
  control, memory, and skill-management tool metadata. Tools are stable atomic
  capabilities; `pick` and `place` are not tools. World-mutating control tools
  are the `AtomAction` subset. Each tool declares its side-effect class so the
  pipeline can decide whether it may be batched before the next observation.
  Tool contracts are host-owned and immutable from the Agent: skill-management
  tools cannot create, update, rename, or remove `ToolSpec` entries or handlers.
  A host-installed execution gate runs before every `WORLD_MUTATING` handler,
  including Simulator MCP proxies; handlers cannot bypass the selected
  supervision profile.
- `agent.runtime.supervision`: host-owned `human_gated`, `standard`, and
  `reviewed_autonomy` policies. Human-gated mode requires actual operator input
  for embodied actions and skill changes. Standard mode preserves deterministic
  runtime safety checks and human interaction pauses. Reviewed autonomy uses
  clean-context action/skill reviewer clients and a separate guidance client;
  it never disables IK, collision, backend, or checker gates.
- `agent.tools.registry.ToolResult`: normalized handler result envelope. Every
  executable handler result is normalized to
  `schema_version=openeta.tool_result.v1` with `result_type`, `outputs`,
  `artifacts`, `state_delta`, and `diagnostics`. Generic `state_delta` is
  planner feedback, not an authoritative environment step. Only handlers bound
  with the host-owned environment authority can publish an
  `openeta.environment_receipt.v1`; the registry overwrites its provenance
  before official reward or terminal fields may enter `StepResult`.
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
  In a reviewed-autonomy batch workspace, a clean authoring client must produce
  a validated `SkillSpec` and a second client must approve it before the
  session-local markdown is replaced. Shared `agent/skills/*.md` is never
  modified by this automatic path.
- `agent.runtime.skill_authoring.BackendSkillAuthoringSubagent`: isolated
  provider client used by agent-facing `register_skill` and `update_skill`.
  Every call starts with a clean context containing only the requested change,
  current skill when updating, executable atomic tool references, and the
  OpenETA skill-creator contract. Its strict `SkillSpec` output cannot contain
  tool mutations or unavailable tools. These tools update only the active
  runtime registry; persistent edits to built-in markdown continue to require
  the explicit self-improvement approval path above.
- `agent.runtime.calibration.CalibrationLifecycleManager`: session-owned
  embodiment profile proposal and publication boundary. It performs
  deterministic schema/numeric checks, invokes an independent clean-context
  calibration reviewer, reads profile-hash-linked canary and held-out evidence,
  and enforces supervision policy before atomically publishing to
  `agent/calibrations/candidate/` or `agent/calibrations/validated/`. Standard
  mode cannot publish shared profiles.
- `agent.runtime.calibration_registry`: deterministically selects one read-only
  embodiment calibration from environment and robot identity. Calibration v2
  contains transforms and physical compatibility, never task object allowlists.
- `agent.tools.grasp_strategies`: validates and selects session-local
  task-family grasp policies. A missing match uses the generic calibrated pose;
  an explicit unknown or incompatible strategy fails closed.
- `agent.runtime.grasp_strategy_lifecycle.GraspStrategyLifecycleManager`:
  clean-context review, strategy/calibration hash-linked paired evidence,
  session staging, and file-locked candidate/validated publication according
  to the host supervision profile.
- `agent.evals.subagents`: fixed, production-path cases for the action reviewer,
  guidance agent, SkillSpec author, and SkillSpec reviewer. It checks expected
  labels and skill invariants, and reports critical false approvals and
  unsupported guidance answers separately from ordinary mismatches.
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
- `agent.backends.provider_config.PlannerProviderConfig`: primary and optional
  fallback API provider settings for CLI and future GUI configuration. It can
  load `.env` or a local `apikey.md`, validate missing fields, redact secrets
  for display, and write a `.env` file.
- `agent.backends.planner.OpenAICompatiblePlannerBackend`: real
  `/v1/chat/completions` backend for OpenAI-compatible providers. When a SAM3
  selection obligation is pending, it attaches the original image and candidate
  contact sheet as bounded multimodal image parts. Provider timeouts, connection
  failures, HTTP 408/429, and selected HTTP 5xx responses use bounded exponential
  backoff before falling back to `response::ask_human`; request and schema errors
  are not retried. When configured, authentication/key rejection, rate limiting,
  connection failure, or timeout switches the next attempt to the other endpoint;
  consecutive switch-eligible failures alternate primary and fallback.
- SAM3 multi-candidate selection is explicit: runtime memory persists a
  `selection_obligation`, the main VLM calls `select_sam3_detection`, and the
  pipeline blocks targeted `grasp_pose_estimate` or world-mutating tools until
  the selected mask is recorded.
- Grasp estimation is exposed as one normalized façade over AnyGrasp,
  Contact-GraspNet, and GraspGenX. Compatible backend failures fall through in
  host-owned order; backend-local scores are never compared across estimators.
  Multi-candidate handling is greedy and stateful: memory
  exposes rank 0 as `grasp_candidate_policy.active_candidate`; a later grasp
  inference replaces the active policy, candidate-linked safety or motion
  rejection advances to the next score-ranked pose, and successful `move_to`
  accepts the queue and releases its downstream gate.
- Public web access is exposed through host-owned `web_search` and `web_fetch`
  tools, never through `python_exec`. `web_search` reuses the configured planner
  provider's `/v1/responses` hosted `web_search` capability, tries the configured
  fallback once after a structured primary failure, and returns a bounded answer
  plus URL citations. Provider keys remain in host-owned Authorization headers
  and never enter tool parameters or results. `web_fetch` accepts only public
  HTTPS text pages and uses prevalidated, IP-pinned TLS connections; it rejects
  redirects, credentials, local/private/non-routable or mixed-DNS destinations,
  nonstandard ports, oversized bodies, compression, and unsupported media.
  Both tools mark results as untrusted external content. They default on when
  planner provider configuration is complete and can be disabled independently
  with `OPENETA_WEB_SEARCH_ENABLED=false` or
  `OPENETA_WEB_FETCH_ENABLED=false`.
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
- `agent.runtime.session_workspace.SessionWorkspace`: one filesystem ownership
  root per parallel Agent session, with private `skills`, `memory`, `artifacts`,
  `sandbox`, and grasp `strategies` directories plus a read-only staged
  calibration. Python sandbox writes are permitted only below the session
  `sandbox` directory; the separately approved host subprocess remains outside
  this automatic path.
- `agent.runtime.planner_prompts`: composes the main Planner's base prompt with
  the host-owned embodied closed-loop contract and records a reproducible
  SHA-256 descriptor. The contract is not injected into role-specific
  author/reviewer/guidance sub-agents.
- `agent.runtime.experiments.ExperimentWorkspace`: owns immutable generation
  skill and grasp-strategy baselines, phase-specific session workspaces,
  objective-evidence candidate collection, paired canary/holdout metrics, and
  promotion lineage under `.openeta_memory/experiments/`.
- `agent.cli.batch_eval`: non-interactive `openeta-batch` entry point. It builds
  a separate planner/runtime/MCP environment and trace/artifact root per
  manifest entry so parallel runs do not share mutable session state.
- `agent.cli.experiment`: dispatcher behind `openeta --command
  preflight|run|iterate|inspect`. `iterate` requires `reviewed_autonomy`, fails
  unattended `ask_human` requests without creating pause records, independently
  reviews skill and strategy candidates in separate lanes, and promotes only
  after paired objective validation.
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

## Sub-agent Provider Evaluation

The sub-agent evaluation is explicit because it makes real provider calls:

```bash
uv run openeta-subagent-eval --list-cases
uv run openeta-subagent-eval --role action_reviewer --role guidance_agent
uv run openeta-subagent-eval --case skill-review-abstain-underspecified-request
uv run openeta-subagent-eval --repeat 3 --strict \
  --output outputs/subagent-eval.json
```

The default suite covers every decision label and two skill-authoring
operations. `--strict` exits non-zero on any mismatch, critical false approval,
or unsupported guidance answer. Reports use `openeta.subagent_eval.v1`, include
provider usage when available, and contain only redacted provider configuration.
Skill Author receives a 4096-token output budget in both the TUI and parallel
harness; bounded decision reviewers retain the 512-token default.

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
keyboard to select commands such as `/provider`, `/model`, `/models`, `/approvement`, `/tools`,
`/sessions`, `/resume`, `/new`, `/run`, `/step`, and `/quit`. Normal task text
runs one closed-loop agent turn in the current session; `/new` starts a fresh
session. `/resume` opens the local session picker, while
`/resume <session_id>` or `/resume --last` restores the canonical conversation
from its latest checkpoint and suffix, along with local trace and working state.
Each turn prints planner usage,
request, reasoning, compact parameters, tool calls, and a display-only result
summary. Tool results are projected to key diagnostics, semantic outputs, and
artifact paths, then bounded to five terminal rows. Complete structured results
remain in the session trace and artifact store; the planner receives bounded
action/result envelopes and artifact references.
`response::ask_human` prompts in the terminal and automatically resumes the same
episode runner after recording the answer unless reviewed autonomy resolves it
through the guidance client. `/approvement` displays the three profiles and
changes the current host gate without rebuilding the active session. Skill
self-improvement proposals remain staged in `.openeta_memory/skill_reviews/pending/`
until the user inspects and approves them with the skill-review slash commands.
