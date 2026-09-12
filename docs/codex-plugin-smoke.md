# Codex plugin smoke prototype

This experimental entry point runs Codex with ChatGPT subscription authentication
against the current OpenETA host. It is a Codex/native-MCP evaluation, not a test
of the XML model-output protocol. The existing API-backed entry points are intact.

## Isolated checkout

Branch: `dev/huaizezheng/codex-plugin-smoke-2026-09-08`.

The checkout is `OpenETA-codex-plugin`, beside the original `OpenETA` checkout.
It has its own Git object database, index, branch and Python environment. Base
commit: `723bed12721910e6bcc8db783cb9c1c16cfddaa1`. The 53 then-uncommitted source
files were copied as a stable read-only snapshot, without committing or changing
the original checkout. Their hashes and the inherited patch are retained in
`tmp/codex-plugin-baseline/`. Existing inherited changes are not this prototype's
implementation diff.

## Boundaries

```text
Codex exec (ChatGPT authentication, one thread)
    -> installed OpenETA plugin (pick skill + native MCP tool schemas)
    -> CodexHost / SubmittedBackend
    -> ToolCallingPlanner -> OpenEtaAgentRuntime -> ActionPipeline / ToolRegistry
    -> OpenEtaEpisodeRunner / SimulatorMcpEpisodeEnvironment
    -> separately owned simulation and configured perception services
```

The bridge normally queues one externally supplied decision. It retains planner
validation, host invariant dispatch, reference resolution, tool admission,
execution scope, observation feedback, memory updates and environment receipts.
An invariant refresh/stop may replace the requested command; the MCP response
reports both requested and executed commands, and discards unconsumed decisions.

Most MCP input schemas are generated from the live ToolContract catalog and
`bundle_stage3` projection. The `move_to` ingress additionally composes target
authoring, IK and receipt-based motion behind one native request, as described below.
The ingress validates the advertised JSON
Schema before execution. A schema rejection is returned for repair without
executing a tool or terminating the episode. This extra admission check is part
of the experimental ingress, not a promotion of the shared contract policy.

The published subset covers observe, SAM3, candidate inspection/selection, grasp
estimation/compilation, waypoint authoring, IK, trajectory composition, motion
and gripper control. `episode_status` reads state; `finish_episode` submits a
completion through existing host processing. Successful completion additionally
requires the shared official-success reducer to accept evidence bound to this
execution and session. A model's `success=true` is never a checker receipt.

Images use native MCP ImageContent, with the existing planner image-selection
and public context projection. Private visual transport paths are filtered.
The copied pick guidance is explicitly a snapshot; live tool schemas control
availability and parameters. It is not an executable pick macro.

## Automatic IK before native move_to

The experimental Codex Host accepts a world EEF pose directly:

```json
{"target_pose":{"frame":"world","xyz":[0.1,0.2,1.0],"quat_xyzw":[1,0,0,0]},"tolerance":0.005,"ori_tolerance":0.05}
```

`move_to` also accepts a current `target_pose` bundle from an existing producer,
or an existing `ik_result` bundle. Grasp compilation and provenance checks still
apply. The Host runs fresh IK with exactly the requested execution tolerances,
requires explicit execution authorization, and passes the newly registered IK
result to the original motion pipeline. Unknown, stale or failed references do
not become executable. Search failure/timeout stops the sequence before motion;
a successful search still does not guarantee controller convergence or collision
clearance. The independent preview tool remains available for diagnosis.

This is an ingress preflight hook implemented in `tools/codex_motion.py`, using
ordinary runner steps; it does not enable or rewrite the shared pipeline's optional
`pre_safety_checks` mapping. Its internal operations use the existing contracts
and record a `parent_request` in `host-commands.jsonl`. A raw pose or prior IK
result consumes up to three Host turns/tool calls; a target bundle consumes up to
two, while either is one native request/model round trip. Original turn, tool-call
and time budgets apply between stages. Host invariant substitution stops the
sequence. Native responses include `motion_hook` with the IK authorization and
motion stop reason. `motion_dispatched` describes Host dispatch, not physical
arrival; a lost/cancelled motion result can leave it unknown.

This native schema extension is local to the prototype. Shared main-harness
contracts and the API/XML entry point are unchanged; adopting the extension there
requires collaborator review. See the
[IK hook validation](diagnostics/codex-motion-ik-hook-2026-09-09.md).
The [Astra task retest with this hook](diagnostics/codex-plugin-astra-hook-retest-2026-09-09.md)
used automatic preflight for all three move requests. It ended unsuccessfully
after a motion convergence failure and source-reference repair problems, before
grasping; internal repair packet IDs were absent from the native responses.

## Subscription and auxiliary models

The launcher accepts saved file-based ChatGPT authentication only, forces the
built-in OpenAI provider, and uses an isolated Codex home/cache/config/workspace.
It copies authentication into a mode-0700 directory with a mode-0600 file and
deletes that copy in its finalizer. Normal user plugin installation and settings
are untouched. Keyring-only authentication is not implemented by this prototype.

All internal model backends fail explicitly without network inference. Grasp
advisor and visual differencing are disabled; the standard deterministic
supervision profile remains enabled. There is no paid provider fallback.
SAM3/AnyGrasp are separately configured services, not subscription LLM calls.

Shell tools, multi-agent delegation, web search and Codex memories are disabled
for this dedicated test configuration. Native OpenETA MCP calls are approved in
that isolated plugin configuration, while host execution checks remain active.
The launcher audits emitted shell/web/file-change events separately.

## Run

Requires the repository dependencies (`uv sync --extra dev`), an installed Codex
CLI, and `codex login` using ChatGPT. Validated with Codex CLI 0.153.2. Install the
benchmark environment separately following `sim/README.md`. Start a **dedicated**
simulator from this checkout; do not stop/reconfigure other agents' services:

```bash
PYTHONDONTWRITEBYTECODE=1 LIBERO_DIR=/path/to/LIBERO \
  .venv/bin/python -m scripts.codex_sim_server --host 127.0.0.1 --port 18778
```

The dedicated entry point defaults to Mink and uses the local minimal overlay
`tmp/codex-mink-deps` (override with `--mink-dependency-path`). Missing dependencies
fail setup; there is no automatic OSC fallback. The experiment launcher defaults
to expecting Mink and checks the actual simulator identity before serving tools.
Host status records `controller_id`; the launcher summary records
`expected_controller`.

First run the bounded observation smoke (no robot motion):

```bash
.venv/bin/python -m scripts.codex_plugin_smoke \
  --sim-url http://127.0.0.1:18778/sse \
  --task 'Pick up the black bowl between the plate and the ramekin and place it on the plate.' \
  --model gpt-5.6-sol --effort medium \
  --timeout 180 --max-requests 8 --observe-only \
  --output tmp/my-codex-observe
```

For one manipulation attempt, omit `--observe-only`, use `--timeout 1200
--max-requests 80` (the defaults), and explicitly pass your `--sam3-url` and `--anygrasp-url`.
Use a fresh output directory for every run. The launcher stages and installs a
private copy of the repo marketplace/plugin, rendering its MCP command with the
checkout's interpreter and per-run arguments. The source MCP file is a template;
the launcher is the supported installation path for this prototype.

Review `summary.json`, `codex-events.jsonl`, `final.txt`, `host/host-status.json`,
`host/host-commands.jsonl` and the host session workspace. Raw event files include
image payloads and are kept in the ignored run directory. `integration_passed`
requires real host tool execution, confirmed cleanup, a completed Codex run and
no observed non-MCP actions. It is separate from `task_success`.

Model token usage is reported by Codex events. Host token estimates are not a
subscription billing total. This prototype bounds elapsed time and requests;
it does not enforce a streaming Codex token ceiling. The host deadline includes
time spent waiting for Codex between MCP calls. Invalid requests consume the
ingress request budget. A watchdog expires abandoned episodes, and normal MCP
disconnect/SIGTERM performs host cleanup. Missing cleanup acknowledgement stays
unknown; it is never converted to success.

As of 2026-09-09 the episode budget is 1,200 seconds and the request, turn and
tool-call ceilings are each 80. This doubles the previous 600-second/40-request
defaults and exceeds twice the 30-request ceiling used by the first live pick
experiment. The per-call simulator timeout remains 120 seconds. Configured
episode limits are recorded in both launcher summaries and Host status.

## Validation

```bash
.venv/bin/python -m pytest -q \
  tests/test_codex_host.py tests/test_codex_plugin_launcher.py tests/test_runtime_assembly.py \
  tests/test_episode_resource_budgets.py tests/test_tool_feedback_episode_environment.py
```

The stdio integration test needs a subprocess environment with working pipe I/O.
In this Codex tool sandbox it stalled at MCP initialization; the same bounded
test passed outside that sandbox. Other local tests passed inside it. The live
Codex plugin also completed the actual stdio/image/host chain.

The source plugin and skill pass the installed `plugin-creator` and
`skill-creator` validators. The repository's minimum MCP dependency already
requires `jsonschema>=4.20.0`, used by the ingress.

## Scope and next work

No shared schema fields, main planner implementation or contract approval hashes
were changed. A subsequent simulator fix makes `render=false` omit intermediate
camera conversion/encoding while preserving raw cached images and terminal
receipts; it applies to callers of that worker in this checkout. The new external ingress and MCP
projection remain an experimental proposal pending collaborator review before
shared integration. No RFC/experiment document was edited during this isolated
prototype; local experiment evidence is recorded separately.

See [the 2026-09-08 experiment record](diagnostics/codex-plugin-smoke-2026-09-08.md)
for the successful observation smoke and the unsuccessful manipulation attempt.
The [2026-09-09 motion diagnosis](diagnostics/codex-motion-timeout-2026-09-09.md)
records the timeout reproduction, camera-processing fix and increased budgets.
The [full-task retest](diagnostics/codex-plugin-pick-retest-2026-09-09.md) reached
the release request but exhausted that increased episode time budget; it also
records the gripper rendering and placement-tool limitations exposed by the run.
The [gripper follow-up](diagnostics/codex-gripper-final-render-2026-09-09.md)
replaces per-step gripper image work with one final render; the placement-tool
coverage limitation remains.
The [Astra subscription experiment](diagnostics/codex-plugin-astra-2026-09-09.md)
uses that optimized gripper path with the same tool subset. It completed the
integration but timed out after two grasp attempts, without reaching placement;
argument/reference repairs, OSC convergence and collision recovery remained relevant.

The adapter currently calls a few private planner projection/validation helpers;
a maintained integration should extract public interfaces once the experiment
justifies them. Further work includes streaming usage/cancellation through the
SDK/app-server, independently supervised gateway lifetime, explicit auxiliary
model strategy, generated skill synchronization, richer artifact paging and
batch evaluation. These are not silently provided by installing the plugin.

Official references: [Codex plugins](https://learn.chatgpt.com/docs/plugins),
[plugin packaging](https://developers.openai.com/plugins/build/plugins),
[non-interactive mode](https://learn.chatgpt.com/docs/non-interactive-mode),
[authentication](https://learn.chatgpt.com/docs/auth).

### Current observation references

Native responses expose `observation_references` and, when applicable, a bounded
`repair` object. Every attached image has an adjacent JSON label with its exact
registered raw RGB reference and current/historical status. The ingress registers
new result observations before returning them, so Codex can ground the next
perception call without waiting for another Host turn. See the
[reference repair and OSC diagnostic](diagnostics/codex-reference-and-convergence-2026-09-09.md)
for validation, publication timing and the remaining controller limitation.

### Default controller and OSC comparisons

Mink is the default for both `scripts.codex_sim_server` and
`scripts.codex_plugin_smoke`. The server wrapper selects it explicitly even if
an inherited `OPENETA_LIBERO_CONTROLLER_PROFILE` names OSC. Selection takes place
when a fresh dedicated simulator/environment is started; the launcher cannot
reconfigure an already running service. It passes `--expected-controller` to
the Host, which rejects missing or mismatched simulator identities and closes
the created environment. This protects experiment attribution; shared simulator
code and other agents' services keep their existing configuration.

For an intentional OSC comparison, use `--controller osc_pose` on **both** the
server wrapper and the experiment launcher. The OSC wrapper does not require the
Mink overlay. Direct low-level `tools.codex_host` diagnostics can optionally
pass `--expected-controller`; that path remains available for other backends.

The default local dependency overlay must exist; the experiment copied the
minimal Mink/qpsolvers/quadprog overlay without replacing LIBERO's
MuJoCo/NumPy/SciPy. Keep the native move's automatic IK and collision checking
enabled. Verify its receipt names `mink.robosuite_joint_velocity` and its
execution seed matches the fresh IK receipt. See the
[Mink replay record](diagnostics/codex-mink-replay-2026-09-09.md) for the outcome
and limits of the single-target comparison.

The first [full Astra attempt with default Mink](diagnostics/codex-plugin-astra-mink-2026-09-09.md)
reached the episode deadline before any motion command. It recovered from model
stream disconnects and a main-view grasp-width rejection, obtained wrist-view
grasp candidates, then timed out. It supplies no new controller-performance result.

Atomic alternative: pass `--tool-profile atomic`; see [atomic tools](codex-atomic-tools.md) and the [Astra experiment](diagnostics/codex-atomic-astra-2026-09-09.md).
