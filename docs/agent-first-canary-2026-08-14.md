# Agent-first canary report — 2026-08-14

## Scope

These canaries test whether the runtime gives the main Agent enough current
evidence, tool output, artifact access, and repair information to own the task
loop. They are contract diagnostics, not benchmark success estimates. Runs were
stopped once they had answered the diagnostic question; interrupted runs are
not relabelled as failures or successes.

The main task was LIBERO object task 2, `pick up the salad dressing and place it
in the basket`, seed 2. This task replaced the milk canary after repeated grasp
backend failures on a very small target made it difficult to distinguish memory
retrieval from downstream perception and grasp quality.

## Runtime changes under test

- The durable transcript remains complete, while the model receives real user
  turns plus the last eight action groups. Large host results are compacted but
  exact artifact paths remain visible. A replay of the 106-item milk transcript
  reduced the model-facing conversation from about 485k to 61k characters.
- The decision-state artifact index is bounded to 24 entries / 30k characters.
  Complete artifacts remain available to `python_exec` by exact path.
- Current camera intrinsics/extrinsics are included directly in the current
  observation packet, removing calibration-file discovery from the normal task
  path.
- Target selection is linked to the compiled grasp in the host evidence graph.
  A newer target selection blocks only stale contact/close actions; clearance,
  retreat, observation, and read-only recovery remain available.
- `move_to` now separates command execution from target attainment. A controller
  receipt with `reached_target=false` remains `operational_success=true`, but is
  exposed as `semantic_outcome=target_not_reached` with requested target, actual
  endpoint, position error, diagnostic, exact response path, and recovery options.
- Pick guidance now recommends preserving failed grasp geometry and selecting a
  materially different safe strategy, rather than treating a regenerated rank-0
  candidate as new evidence.

## Recorded runs

### Luna baseline before motion-result remediation

`agent-first-salad-dressing-luna-20260814-r1` completed 37 Agent actions and 36
tool calls before manual interruption. It performed three complete grasp cycles,
including re-segmentation and re-estimation after each failed attachment, but
continued to take the highest-ranked candidate. Model prompt sizes stayed bounded
between 21,232 and 68,662 tokens (about 54,937 average); total provider-reported
tokens were 2,045,678. This verified the context-window fix and target/grasp
freshness repair, while exposing weak strategy variation.

### Invalid nominal Sol run

`agent-first-salad-dressing-sol-20260814-r1` is **not a Sol comparison**. The
primary provider rejected its credential with HTTP 401 and runtime failover sent
all 14 Agent calls to the configured `gpt-5.6-luna` fallback. Its behavior must
not be attributed to Sol. The run was manually interrupted during a MolmoPoint
recovery call; that interruption is not evidence of a provider timeout.

### Luna after explicit motion feedback

`agent-first-salad-dressing-luna-20260814-r2-motion-feedback` completed 30 Agent
actions and 30 tool calls before manual interruption. Four motion calls reported
`semantic_outcome=target_not_reached`. On the first contact descent, the inline
feedback reported a 3.02 cm error. Luna cited the actual endpoint and current
images, did not blindly replay the nominal pose, and chose to close because the
fingers visibly surrounded the bottle. It subsequently ran a lift probe, rejected
attachment when the bottle remained at the source, and rebuilt current target and
grasp evidence. On a later descent it used one short retry from the reported actual
pose, then closed. Two `python_exec` calls completed and their results were
available in later context. Prompt sizes remained 21,527–71,263 tokens (about
55,703 average); total provider-reported tokens were 1,681,905.

### Provenance-verified Sol comparison

Before rerunning, a minimal provider probe confirmed that the usable fallback
endpoint supports `gpt-5.6-sol`. The run
`agent-first-salad-dressing-sol-20260814-r2-verified` set the fallback model
explicitly and verified every completed Agent call as
`result.model=gpt-5.6-sol`, `provider_role=fallback`.

The run completed 17 Agent actions and 17 tool calls before manual interruption.
Sol correctly distinguished two visually similar SAM candidates, consumed the
explicit 4.09 cm contact error, closed only after checking both cameras, treated
post-close openness as weak rather than conclusive evidence, and diagnosed
attachment failure after a lift from target/EEF non-co-motion and absent source
vacancy. It then opened, re-segmented, re-estimated, and recompiled. Its second
cycle still chose the newly estimated highest-ranked candidate rather than using
a materially different stored strategy. Prompt sizes remained 21,279–66,156
tokens (about 50,436 average); total provider-reported tokens were 863,447.

## Conclusions

1. The harness is active: current images, structured tool results, exact artifact
   paths, `python_exec` results, camera calibration, and evidence freshness all
   influenced later Agent decisions.
2. The host no longer needs a grasp-phase state machine to make the Agent recover.
   Both models could infer failure, invalidate old evidence, and restart from a
   current observation using skill guidance.
3. Clear semantic motion feedback materially reduced ambiguous replay. It should
   remain an additive ToolResult contract: command acknowledgement and physical
   goal attainment are different facts.
4. The remaining dominant failure is below the high-level planner: candidate
   geometry and controller contact behavior often leave the gripper 3–4 cm from
   the requested pose or produce ineffective rank-0 grasps. Sol improved the
   precision of visual reasoning but did not by itself create grasp-strategy
   diversity.
5. Object-memory retrieval should not be judged solely by downstream grasp
   success. Retrieval/localization, SAM selection, grasp estimation, controller
   attainment, attachment, and placement require separate rollout metrics.
6. Do not add a host retry script for this failure. Preserve the full candidate
   set and outcome artifacts, make failure history queryable, and keep strategy
   variation as editable skill guidance. Deterministic gates should continue to
   enforce only safety, evidence identity/freshness, and authority.

## Follow-up measurements

- Extract per-attempt candidate geometry and compare it with prior failed
  candidates, independent of regenerated ids.
- Measure requested-versus-actual EEF error by waypoint role and controller
  settings.
- Add attachment evidence scoring based on target/EEF co-motion and source
  vacancy, but keep it advisory unless used for a necessary safety invariant.
- Run benchmark completion trials only after contact attainment and candidate
  diversity improve; current interrupted canaries do not establish task success.
