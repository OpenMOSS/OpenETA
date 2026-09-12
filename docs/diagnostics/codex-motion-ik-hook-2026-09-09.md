# Automatic IK before native move_to — 2026-09-09

## Problem and implemented boundary

The user requested that `move_to(pose)` trigger an IK preview hook, avoiding a
separate model-managed check/execute sequence. The Codex ingress now implements
this in the isolated branch `dev/huaizezheng/codex-plugin-smoke-2026-09-08`.
The original checkout and all 53 inherited source files remain unchanged.

`tools/codex_motion.py` composes the existing authoring, preview and execution
contracts. `CodexHost._execute` supplies one ordinary planner/runner step to each
stage. This is not a new solver or a direct simulator call. The shared pipeline
already supports an optional `pre_safety_checks` mapping; this prototype does not
enable that separate mechanism or change the shared receipt-only move contract.
It implements the requested single-request behavior at the external Host boundary.

Native `move_to` accepts either:

- `target_pose` with `frame=world`, `xyz`, and explicit `quat_xyzw`;
- a current `target_pose` bundle produced by compilation, viewpoint/geometry
  tools or target authoring;
- an existing current `ik_result` bundle, which is resolved and freshly checked.

Position and orientation tolerances are shared by preview and execution. Explicit
request values take precedence; otherwise stored target/receipt tolerances apply,
with simulator defaults 0.002 m and 0.05 rad as fallback. The hook preserves
resolved target provenance and orientation policy. Explicit motion controls remain
available and collision checking defaults to true. It never retries by disabling
checks, changing the target, loosening tolerances or raising step counts.

Direct poses and old IK results use `propose_motion_target -> ik_preview_check
-> move_to`; existing target bundles use `ik_preview_check -> move_to`. The final
move uses the new immutable IK handoff. It still passes reference freshness,
execution authorization, provenance and collision gates. Separate preview remains
available for diagnosis and candidate comparison.

Each internal operation is an ordinary budgeted Host turn/tool call. The external
call counts as one request, and avoids up to two intervening model round trips.
This implementation does **not** reduce the number of underlying tools required.
If the deadline, turn limit, tool budget or a Host invariant interrupts a stage,
the sequence does not proceed to later motion. Logs retain `parent_request` on
each internal step for attribution. No subscription/model call was needed to
validate this change.

## Result semantics

The MCP response adds `motion_hook`, containing stages, IK receipt, explicit
execution authorization, motion summary and stop reason. An unsuccessful or
inconclusive preview does not dispatch motion. In particular:

- `ik_search_no_solution`: the search did not find a solution; this is not a
  formal proof that none exists.
- `ik_search_timeout`: the numerical search budget expired before certification.
- `iteration_limit`: IK authorized execution but the controller did not arrive
  within its internal step budget.
- `collision_detected`: a motion safety stop, separately visible from IK.

Successful tool transport does not establish arrival. `motion_dispatched` reports
dispatch to the motion handler, not steps executed, attachment or task success.
It can remain unknown if a motion result is lost/cancelled. Failure responses
include the stage at which the composition stopped.

The earlier Astra run already performed successful IK previews before its two
150-step convergence failures (60.8 and 77.8 mm residuals). This change prevents
omitted/separated preflight and makes its evidence visible with motion, but does
not prove that those two failures were caused by omitted IK. Diagnosing them
still requires checking the recorded pose, IK/FK conventions and local controller
execution. A feasible IK branch is not proof that OSC can enter that branch.
No probability estimate for the competing causes was established here.

## Validation

Focused tests exercise the real planner, memory bundle registration, runner and
gates with synthetic IK/motion handlers. They cover:

- one native request, ordered internal stages and matching execution tolerances;
- target bundles, prior IK bundles, repeated identical IK receipt identities,
  and stricter execution tolerances;
- unknown/stale/mixed references, no-solution and timeout results, and transport
  failure without motion fallback;
- turn/tool-budget exhaustion and Host invariant substitution;
- controller iteration limits and collision stops remaining distinct from IK.

The Host/launcher regression checks and real MCP stdio connection/cleanup test
were also run: 34 focused tests passed, plus one separately run stdio test.
The stdio test used the unrestricted process environment because
the tool sandbox's pipe behavior had previously blocked MCP initialization.

An independent LIBERO OSC simulation used the real Host without any model:

| Case | Native-call time | IK result | Motion |
| --- | ---: | --- | --- |
| Target at world `[9, 0, 1]` | 34.415 s | Search timed out at 30.0001 s; inconclusive | No move dispatched; cached joint state unchanged |
| Current EEF +2.5 cm vertically | 11.514 s | Feasible; solver 0.039 s | Reached in 18 steps; 4.41 mm position error with 5 mm tolerance |

Two native requests consumed five internal turns/tool calls. This validates
preflight blocking and a reachable motion, not a complete manipulation task or
an improvement in Astra's success rate. The far target is an obvious diagnostic
stress case, but the observed solver receipt is still `inconclusive`, not a
certified unreachable verdict.

Artifacts: `tmp/codex-motion-hook-02/report.json`, `run.py`, `run.log`, `sim.log`,
and `host/host-commands.jsonl` plus the session rollout. Host session
`86b71088a1e34ffc854964aa2ed122f3`, execution
`1924b8f4-2aac-4733-a133-f2857b989ce9`. The episode was explicitly closed after
validation; it did not exhaust its 240-second budget or claim task success.
Dedicated simulator PID 3363556 exited and port 18778 was released.

The first diagnostic script (`tmp/codex-motion-hook-01`) incorrectly accessed
the EEF pose dictionary as an object. It stopped before any motion and cleaned
up its dedicated process. The corrected second script produced the results above.

## Files and adoption

Implementation: `tools/codex_host.py`, new `tools/codex_motion.py`.
Tests: new `tests/test_codex_motion_hook.py`. Plugin skill guidance and the local
usage guide describe the native call and its budget accounting. The existing
source-snapshot experiment records remain historical evidence.

The new native ingress schema and `motion_hook` receipt are experimental
extensions. Shared/API harness schemas, contract promotion hashes and the original
checkout are unchanged. Shared adoption should be reviewed by the collaborators;
no commit, push or shared-document update was performed.
