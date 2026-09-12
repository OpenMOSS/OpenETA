# Codex subscription/plugin verification — 2026-09-08

Follow-up: [the 2026-09-09 motion diagnosis](codex-motion-timeout-2026-09-09.md)
reproduced the timeout and fixed redundant intermediate image processing.
The historical run evidence below is unchanged.

## Finding

ChatGPT-authenticated Codex CLI 0.153.2 running `gpt-5.6-sol` at medium effort
successfully used an isolated installed OpenETA plugin to read native MCP images
and execute the current Host tools. The observation smoke completed and received
a successful environment-close receipt. One bounded manipulation attempt reached
SAM3, AnyGrasp, grasp compilation, IK and motion dispatch, but did **not** establish
task success. Motion and subsequent environment close timed out.

This verifies a subscription-backed Codex/native-MCP entry point. It does not
validate the original XML parser or demonstrate a successful manipulation policy.
No internal LLM API backend was enabled; no paid-provider fallback was configured.
SAM3 and AnyGrasp used the existing explicitly configured perception services.

## Isolation and implementation

- Checkout: `/media/user/B29202FA9202C2B91/Stage2-OpenETA/OpenETA-codex-plugin`.
- Branch: `dev/huaizezheng/codex-plugin-smoke-2026-09-08`.
- Base: `723bed12721910e6bcc8db783cb9c1c16cfddaa1` plus 53 captured uncommitted
  files from the source checkout. This is an independent clone, with its own
  Git database and Python environments. All inherited file hashes remained intact.
- Baseline manifest/patch: `tmp/codex-plugin-baseline/`; inherited patch SHA-256:
  `4ecb71b076848abadfb599f400e6233a0eaef18b417e5d5850f98ec9a9e2e095`.
- New code: `tools/codex_host.py`, `tools/codex_mcp_server.py`,
  `scripts/codex_plugin_smoke.py`; packaging in `plugins/openeta` and
  `.agents/plugins/marketplace.json`. No existing runtime or simulator file edited.
- Each run installed a private plugin copy into a private Codex home, with
  ChatGPT authentication, shell/web/delegation disabled, and an empty working
  directory. No normal user plugin cache or configuration changed.
- Simulator used only the dedicated `127.0.0.1:18778` service from this checkout.

See [the usage guide](../codex-plugin-smoke.md) for architecture and reproduction.
This local prototype has not been committed, pushed, or integrated into the shared
RFC. Existing inherited changes are excluded from the prototype review patch.

## Live runs

All runs used LIBERO spatial task 0, seed 0: pick up the black bowl between the
plate and ramekin and place it on the plate. Observation runs explicitly prohibited
motion and requested `finish_episode(success=false)`.

| Run directory under `tmp/` | Elapsed | Host requests / tool calls | Result |
| --- | ---: | ---: | --- |
| `codex-plugin-observe-01` | 76.058 s | 0 / 0 | Plugin MCP approval blocked calls; connection smoke failed. No close acknowledgement. |
| `codex-plugin-observe-02` | 74.014 s | 3 / 1 | Status → observe → unsuccessful finish; native images received; cleanup `ok=true`. Connection smoke passed. |
| `codex-plugin-pick-01` | 507.562 s | 20 / 15 | Motion outcome unknown after timeout; no official success; cleanup `ok=false`. Full attempt failed. |

The first run exposed the need to configure MCP tool approval explicitly for
`codex exec`. The fix sets approval only for the isolated OpenETA plugin instance.
The stdio server also gained a SIGTERM cleanup handler. The second run exercised
both changes successfully. Codex completed with exit code 0 in all three runs;
this alone does not mean the episode passed.

The historical run summaries predate the final `integration_passed`,
`task_success`, `official_task_success`, and `completion_claim` report fields.
Raw summaries were preserved, not rewritten to look like newer runs. The current
launcher requires confirmed cleanup, real Host tool execution, a completed Codex
run and no observed non-MCP actions for `integration_passed`. It reports official
task success separately. These final reporting changes have local regression
coverage but were not used by another subscription run.

### Manipulation evidence

Host session: `537ae3ce990d480c88ebaafc62c02763`.
Execution: `68409d08-c802-44d2-b00e-c3e6719f7fd4`.

1. An initial SAM3 request omitted required image references. The ingress returned
   a schema error; Sol supplied the references on retry without terminating the
   episode. This is evidence of repair, not elimination of all parameter errors.
2. SAM3 returned three bowl detections. Sol selected `detection_000`; AnyGrasp
   returned 17 candidates. Bundle references resolved through the existing Host.
3. Several candidates failed bounded six-degree-of-freedom IK searches. Sol
   compiled a different candidate and obtained a kinematically feasible receipt.
4. The first `move_to` was blocked by
   `ik_collision_delegation_not_authorized`. The local service reported cuRobo
   unavailable. A subsequent preview explicitly used
   `check_endpoint_collision=false`; motion retained
   `enable_collision_check=true`. Acceptance by the existing Host does not prove
   complete trajectory/world collision coverage.
5. The next motion request failed after the 120-second transport timeout.
   Its execution receipt says `dispatch_status=outcome_unknown` and
   `reconciliation_required=true`. The actual remote motion outcome is unknown.
6. The following requested IK call was replaced by the Host invariant's `talk`
   stop. It was **not** dispatched as a new IK call. There was no gripper command,
   no established grasp/place result, and all 16 recorded step rewards were 0.
7. `close_env` also timed out. `closed=true` indicates the local Host stopped;
   the separate cleanup receipt remained `ok=false`. The prototype does not solve
   simulator worker retirement after a transport-unknown action.

The exact root cause of motion/close timeout was not established by this smoke.
It should be diagnosed at the simulator/worker boundary before batch evaluation.
No retry loop or fabricated success receipt was used to pass the test.

### Usage and remaining cost concerns

| Run | Input tokens | Cached input tokens | Output tokens | Reasoning output tokens |
| --- | ---: | ---: | ---: | ---: |
| observe-01 | 156,246 | 136,576 | 1,508 | 739 |
| observe-02 | 215,653 | 159,744 | 1,048 | 378 |
| pick-01 | 2,844,490 | 2,684,288 | 4,849 | 1,659 |

These are aggregated counters emitted by Codex, not a conversion into subscription
credits or API charges. The run still consumes subscription allowance. The large
input total shows that repeated context/image delivery needs optimization before
large-scale evaluation. No token-level hard cap is implemented; wall time and
request count are bounded. There were no recorded shell, web or file-change actions
from the tested Codex process in any run.

## Cleanup and local validation

After the final run, the dedicated simulator process group `2679081` was sent
SIGTERM. Its shell, server PID `2679084` and worker PID `2680371` were confirmed
absent; port 18778 was confirmed unbound. This retires the dedicated service after
the failed close. It does not retroactively turn either failed close receipt into
a successful one. All three temporary `codex-home/auth.json` copies were confirmed
deleted. Other agents' services were not stopped.

Focused validation: 15 Host tests, 6 launcher tests, and 29 existing runtime,
resource-budget and tool-feedback tests passed (50 total). These cover typed
argument repair, unavailable tools, reference admission, official-success gating,
single-use decisions, invariant overrides, budgets, watchdog/close behavior,
actual MCP stdio disconnect, image pixels and credential/config isolation.
The stdio test required execution outside the Codex tool sandbox; inside it MCP
initialization stalled. Plugin and skill validators passed, as did compilation
and whitespace checks. No full repository suite was claimed.

The next useful changes are simulator timeout/retirement diagnosis and smaller
MCP context/artifact projections. Broader SDK/app-server orchestration, batch
evaluation, automatic skill synchronization and public Host adapter interfaces
remain separate work.
