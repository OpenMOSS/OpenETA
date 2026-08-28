# Manual VLM harness debugger

`tools.manual_vlm_proxy` lets a person take the place of the VLM while the real
OpenETA harness continues to build prompts, validate decisions, execute tools,
enforce gates, refresh observations, and record rollouts. Its default **决策台**
view is a bounded human-operator projection; **模型输入审计** preserves the exact
provider-facing request for turn-by-turn verification.

It is a standalone OpenAI-compatible HTTP service. It does not import the agent
runtime and therefore does not need to track internal harness classes. The GUI
shows the final wire request: every system/history/user message, ordered content
part, inline scene image, mask overlay, contact sheet, grasp preview, and request
option. The raw JSON remains available for exact inspection.

## Two views

### 决策台 (operator view)

Use this view to make the next decision. It promotes only the information a
human normally needs:

- current objective and request type (main planner or an isolated VLM role);
- validation errors and retry attempt;
- labelled current visual evidence at inspectable resolution;
- current robot/environment state and the latest executed action result;
- unresolved obligations, special output contracts, and relevant skills;
- a searchable tool picker backed by a structured parameter form.

Select a tool and fill only its parameters; the operator does not need to write
XML. Each parameter exposes its key, description, inferred value type, and one
of three explicit handling modes when allowed:

- **填写值** — enter a custom scalar, or JSON for an object/array;
- **省略（不发送）** — omit an optional or unspecified parameter entirely;
- **使用默认值** — send the default documented by the tool contract.

Explicitly required parameters cannot be omitted. Parameters with documented
defaults initially select the default; optional parameters without defaults
initially remain omitted. Because the current tool registry uses descriptive
parameter strings rather than strict JSON Schema, required/optional/default
metadata is normalized from conventions such as `required`, `optional`, and
`defaults to`. Native `{type, required, default, enum}` descriptors are also
supported for future tools.

The browser submits a structured decision object. The proxy serializes it to
typed nested XML before returning it to the harness, so booleans, numbers,
objects, arrays, empty objects, and escaped text retain their intended types.
The generated XML remains available as a read-only preview for verification.
Use **其他响应** for `talk`, `ask_human`, `task_complete`, or an advanced raw
response. Isolated JSON sub-planners retain a raw JSON editor because their
output contracts are role-specific.

The operator projection is derived from the wire request and never mutates or
replaces it. Missing fields remain visibly unknown rather than being inferred by
the proxy.

### 模型输入审计 (wire audit)

Use this view to verify exactly what one model call received. It shows message
order, roles, content-part types and sizes, the complete text of every message,
every image attachment, and every non-message request option. Long text is paged
for navigation but is not summarized or truncated.

The browser-facing audit endpoint replaces inline image data URLs with small
byte-preserving image endpoints so polling does not repeatedly transfer base64.
Open **Raw JSON** to inspect the original parsed request body, including the
original data URLs. Requests are grouped by session and retain separate turns,
including validation retries. Isolated roles such as visual differencing are
labelled separately from the main planner.

## Start

In terminal 1:

```bash
python -m tools.manual_vlm_proxy --port 8099 --open \
  --record-dir tmp/manual-vlm-traces
```

In terminal 2, point only the planner provider at the proxy. Do not put these
values in `.env` unless you intentionally want them to persist:

```bash
OPENETA_LLM_PROVIDER=manual-vlm \
OPENETA_LLM_MODEL=human-vlm \
OPENETA_LLM_API_BASE=http://127.0.0.1:8099/v1 \
OPENETA_LLM_API_KEY=local-placeholder \
OPENETA_LLM_TIMEOUT_S=86400 \
OPENETA_LLM_MAX_ATTEMPTS=1 \
uv run openeta --once "inspect the current scene" --max-turns 8
```

Open `http://127.0.0.1:8099/` if the browser was not opened automatically.
Each planner call appears in the queue. Review the decision-focused state and
images, optionally cross-check the full wire audit, enter the exact XML decision,
then choose **Send to harness**. The agent advances through its normal pipeline
and the next planner request appears with fresh tool feedback and observations.

The structured response form is pinned to the bottom of the viewport. In the audit view, long
prompts are split into bounded text pages and long conversations into message
pages, so neither requires scrolling to the end before responding. In the
operator view, search the tool picker and choose **选择** to expose that tool's
parameters. Choose whether each optional value is omitted, customized, or set
to its documented default, then send the generated decision.

For an isolated JSON-returning sub-planner, inspect `response_format` under
**Request options** and return the JSON object requested by its system prompt.

## Useful controls

- Click an image to inspect it at its intrinsic resolution. Images are served by
  index so the browser view does not duplicate multi-megabyte base64 in polling.
- Switch between **决策台** and **模型输入审计** without losing the selected turn;
  the preferred view is remembered locally.
- Open **raw JSON** to inspect the exact request body received from the harness.
- Press Ctrl/Cmd+Enter to submit a response.
- Use **Cancel request** to make the provider call fail and exercise retry/failure
  handling.
- `--record-dir` stores `request.json`, `response.json`, or `cancel.json` under a
  request-id directory for bug reports and diffing.
- Multiple simultaneous evaluation workers are supported and appear as separate
  pending requests. The left sidebar groups requests by session. It uses an exact
  ID from `X-OpenETA-Session-ID`, `X-Session-ID`, or serialized session metadata
  when present; otherwise it labels a best-effort conversation-lineage group as
  `inferred`.
- Drag the reply editor vertically to make it as large as needed. Its height is
  stored in browser-local settings and restored on the next launch.

The server binds to loopback by default and has no authentication. Avoid binding
it to a shared network: prompts and images can contain sensitive environment data.

## Why this stays decoupled

The only contract is OpenAI-compatible `POST /v1/chat/completions` (plus
`GET /v1/models`). The proxy observes the serialized payload after OpenETA has
already selected and encoded visual attachments, so it neither reconstructs
planner context nor reaches into memory, tool, simulator, or GUI internals. A
harness update only affects this debugger if it removes or incompatibly changes
the external OpenAI-compatible provider contract.
