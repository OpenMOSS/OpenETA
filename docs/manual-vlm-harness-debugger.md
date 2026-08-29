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
