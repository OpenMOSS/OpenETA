# Manual VLM Console

The manual VLM console is a standalone OpenAI-compatible service that lets a
person answer model requests while inspecting the exact provider-facing input.
Its transport and UI core are protocol-neutral. Project-specific semantics are
optional plugins loaded through the `ProtocolAdapter` boundary.

## Architecture

The module is split into four independent pieces:

```text
OpenAI-compatible client
          │
          ▼
manual_vlm_proxy.py        transport, queue, records, images, wire audit
          │
          ▼
manual_vlm_protocol.py     small adapter interface + generic fallback
          │
          ├── generic      raw assistant content; no project dependency
          └── adapter      request projection + structured composer + encoder
```

The core does not import a planner, agent runtime, tool registry, simulator, or
any project protocol adapter. Adapters are discovered only when explicitly
selected with `--adapter module:attribute`.

The browser is also data-driven. It understands generic presentation sections
(`text`, `json`, and `images`) and generic composer intents (`tool_call`,
`action`, or raw `content`). It does not understand an adapter's output syntax.
The selected adapter translates those intents into the client protocol.

## Generic mode

Generic mode has no project-specific dependency. Select it explicitly in this
repository because the project configuration defaults to the OpenETA adapter:

```bash
python -m tools.manual_vlm_proxy --adapter generic --port 8099 --open \
  --record-dir tmp/manual-vlm-traces
```

Point any client that uses `POST /v1/chat/completions` at:

```text
http://127.0.0.1:8099/v1
```

The console shows messages, ordered content parts, images, non-message request
options, and the original request JSON. Enter the exact assistant content in the
raw response box. The core wraps it in a standard chat-completion response.

Generic session grouping uses `X-Session-ID`, a top-level `session_id` or
`conversation_id`, or best-effort conversation lineage.

## OpenETA adapter

The optional adapter in `tools.manual_vlm_openeta` owns all OpenETA-specific
behavior:

- recognizing `openeta.*` schemas and planner roles;
- reading serialized `tool_context`, objectives, observations, attempts, and
  validation errors;
- extracting `available_tools` and normalizing parameter descriptions;
- exposing `talk`, `ask_human`, and `task_complete` actions;
- converting the console's generic structured intent into the response format
  declared by the provider request: compact JSON for the current planner or
  typed `<decision>` XML for an XML planner.

This repository selects the adapter by default through
`tools/manual_vlm_config.json`, so the normal command is:

```bash
python -m tools.manual_vlm_proxy --port 8099 --open \
  --record-dir tmp/manual-vlm-traces
```

The equivalent explicit invocation is:

```bash
python -m tools.manual_vlm_proxy \
  --adapter tools.manual_vlm_openeta:OpenETAProtocolAdapter \
  --port 8099 --open
```

Then point the planner provider at the proxy:

```bash
OPENETA_LLM_PROVIDER=manual-vlm \
OPENETA_LLM_MODEL=human-vlm \
OPENETA_LLM_API_BASE=http://127.0.0.1:8099/v1 \
OPENETA_LLM_API_KEY=local-placeholder \
OPENETA_LLM_TIMEOUT_S=86400 \
OPENETA_LLM_MAX_ATTEMPTS=1 \
uv run openeta --once "inspect the current scene" --max-turns 8
```

In the structured composer, select a tool and choose how to handle each
parameter:

- **填写值** enters a custom scalar or JSON value;
- **省略（不发送）** leaves an optional parameter out entirely;
- **使用默认值** sends the default documented by the tool descriptor.

The browser submits a protocol-neutral intent. It never asks the operator to
write JSON or XML and does not generate either encoding itself. The OpenETA
adapter follows the provider request's response contract on the server. Main
planner requests retain the structured composer in both formats; isolated JSON
planners use the raw response composer because their output contracts are
role-specific.

## Views

### Isolated grasp advisor correlation (2026-09-06 candidate)

The host now passes the owning Agent session ID privately into the grasp
advisor call. The published `grasp_selection_bundle.v1` is unchanged. The
isolated provider prompt may include this display-only context:

```json
{
  "schema_version": "openeta.grasp_selection_advice.v1",
  "role": "read_only_grasp_pose_advisor",
  "request_lineage": {
    "parent_session_id": "main-agent-session",
    "child_session_id": "advisor-unique-id",
    "parent_tool": "grasp_pose_estimate"
  }
}
```

This links a child to a **parent session**, not to an exact provider request or
action ID. Each advisor invocation gets a separate child identity; retries of
the same provider request retain it. No main-planner conversation history,
tool permission, or candidate activation authority is inherited. The lineage
is a UI correlation hint, not trusted execution evidence or authorization.

The OpenETA adapter recognizes the current advice schema, shows “等待人工 advisor
响应” while pending, and retains the raw JSON composer. `/api/requests` and
request details expose additive `parent_session_id` and `wait_reason` fields.
The queue groups linked children under their parent session and shows a global
pending count. When querying for a main session, match `session_id` **or**
`parent_session_id`; matching only the former still excludes isolated children.
Responded/cancelled requests have an empty wait reason. Answering or cancelling
a child does not answer/cancel another request.

Legacy advisor requests without lineage remain independently visible with a
waiting label. The console does not guess their parent from task text, images,
candidate metadata, or timing. Additional linked roles are listed below. Generic
adapters do not acquire OpenETA-specific parsing dependencies.

These are personal-branch implementation candidates; the additive prompt/queue
fields and compatibility behavior require three-person review before shared
main integration. The running console service must load the new Python code
and the browser must reload the JS to show the new behavior. Do not restart a
service with pending requests without coordinating their completion/cancellation.
This patch does not implement nested cancellation propagation or split total
latency into inference versus human waiting time.

### Additional isolated roles (2026-09-07 candidate)

Guidance, independent action review and visual differencing now use the same
display-only parent/child convention. Only recognized schema/role pairs can
supply lineage to the OpenETA console adapter:

| Schema | Role | Pending label |
| --- | --- | --- |
| `openeta.supervision.v1` | `guidance_agent` | 等待人工 guidance 响应 |
| `openeta.supervision.v1` | `independent_action_reviewer` | 等待人工 action reviewer 响应 |
| `openeta.visual_delta_request.v1` | `visual_differencing` | 等待人工 visual-delta 响应 |

For example, the isolated guidance context adds:

```json
{"schema_version":"openeta.supervision.v1","role":"guidance_agent","request_lineage":{"parent_session_id":"main-agent-session","child_session_id":"isolated-unique-id"}}
```

The episode runner supplies its current memory session ID through private
guidance context; the action reviewer uses runtime execution metadata; visual
differencing uses the owning memory session directly. These producers do not
search nested history for a parent. Missing/invalid Host parent IDs omit lineage
rather than inventing a parent; older no-lineage console identity fallback is
unchanged. Each isolated invocation receives a fresh child ID, while retries
within the same backend request retain it. Explicit session headers continue
to take precedence in the console's session identity logic.

The existing grouped queue supports these roles without new UI controls.
Local tests cover producer inputs, distinct child histories and clearing the
wait label at completion; a real loopback guidance/backend/console fixture
confirms discovery and response-unblock using fixed text only. This is not a
live task experiment or a deployment to the existing service.

These additive contexts/labels remain pending three-person review. Skill
author/reviewer and other isolated roles are not all covered. Parent session
correlation does not identify an owning active request or execution lifetime:
the planner's request may already be answered when its tool starts an advisor.
Do not infer cascade cancellation from that answered status. Parent/child
cancellation ownership and non-main-provider wait accounting remain open;
this patch neither cancels siblings nor changes their permissions or budgets.

### 操作台

The selected adapter provides a bounded projection made of generic sections.
With the OpenETA adapter this includes current visual evidence, observation
state, latest action, validation errors, unresolved obligations, relevant
skills, and the required output contract.

The separate **Tool Call 审计** ledger pairs each serialized `openeta_action`
with its `openeta_host_result` by `action_id`. It shows the exact projected
arguments, execution status, success verdict, result content, structured
outputs, and artifact references. Records are deduplicated and accumulated by
session while the console is running, so a later prompt compaction does not
remove results the console has already observed. A result appears when the next
provider request carries the host execution feedback; it cannot appear before
the tool has executed.

An adapter projection never replaces or mutates the original request. Missing
fields stay absent rather than being invented by the console.

### 模型输入审计

The audit view is entirely core-owned and always available, including in generic
mode. It preserves message order, roles, content-part types and sizes, complete
text, images, request options, and a normalized SHA-256 digest.

Inline data URLs are replaced in the polling response with small byte-preserving
image endpoints. **Raw JSON** exposes the original parsed body, including the
original data URLs.

## Adapter contract

An adapter implements `tools.manual_vlm_protocol.ProtocolAdapter`:

```python
class ProtocolAdapter(Protocol):
    adapter_id: str
    label: str

    def classify_request(self, body): ...
    def session_identity(self, body, headers): ...
    def attempt(self, body): ...
    def presentation(self, body, *, request_id): ...
    def encode_response(self, body, submission, *, request_id): ...
```

`presentation` returns a generic operator-view and composer schema.
`encode_response` returns an `EncodedResponse` containing a standard assistant
message and finish reason. A new client protocol can therefore be supported
without changing the proxy, request store, HTTP handler, or browser code.

## Useful controls and safety

- Click an image to inspect it at intrinsic resolution.
- Switch views without losing the selected request.
- Press Ctrl/Cmd+Enter to submit.
- Cancel a request to exercise provider retry or failure handling.
- `--record-dir` stores exact request, response, and cancellation JSON.
- `--decision-timeout 0` waits indefinitely; a positive value cancels an
  unanswered request after that many seconds.
- Multiple workers are kept as separate pending requests and grouped by exact
  session identity or inferred conversation lineage.

The service binds to loopback by default and has no authentication. Do not bind
it to a shared network when prompts or images may contain sensitive data.
