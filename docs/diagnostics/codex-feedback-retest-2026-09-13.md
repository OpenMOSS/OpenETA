# Two-task intervention-feedback retest — 2026-09-13

Both previously unfinished tasks were rerun once with independent Astra/high
sessions. Neither passed the official checker. Both integrations completed
normally, with zero recorded Codex stream errors.

Artifacts: `tmp/codex-feedback-retest-20260913/`. Protocol and source archive are
in `protocol.json`, `frozen-source.json` and `runtime-source.tar.gz`; results are
in `results.json`, and public feedback statistics in `feedback-analysis.json`.

## Fixed settings

Atomic profile, Mink joint velocity, Cartesian tracking and local fixture contact
patches enabled; official initial-state index 0, seed 0, five settling steps.
Each episode allowed 2400 seconds and 160 native/internal calls. Plugin version
`0.1.0+codex.20260913085939` was installed into a fresh private Codex home for
each trial, using subscription authentication. Runtime source hashes matched the
preceding feedback validation and stayed unchanged throughout both runs.

| Task | Official result | Host time | Native / internal calls | End condition |
|---|---|---:|---:|---|
| Goal 5: push plate to front of stove | Failure | 810.0 s | 43 / 76 | Agent submitted failure after blocked approaches and withdrawals |
| Long 3: bowl into bottom drawer and close | Failure | 1450.3 s | 75 / 159 | Agent submitted failure with insufficient remaining internal budget for another ordinary motion sequence |

Neither ended at the wall-clock timeout. Long 3's final message describes an
exhausted action budget, but the Host recorded **159/160**, no budget exception,
and a normal `finish_episode(success=false)`. This is an Agent decision near
the limit, not a Host forced timeout or an integration failure.

## Observed manipulation and feedback

Goal 5's Agent reported that the plate did not move. Native motion receipts
contain 11 target arrivals, 9 Cartesian-segment failures, 3 local stalls and
2 failed IK searches, plus one completed gripper action. Arrivals alone did
not establish plate displacement. Stall receipts distinguish binding arm/palm
clearance constraints and actual palm contact from a hard collision stop.
Later recovery segments also failed position-corridor checks. The Agent changed
approach direction repeatedly, then explicitly reported inability to complete.

Long 3's Agent reported a retained bowl after a short lift, then placement inside
the bottom drawer. Four lowering requests were rejected by collision checking;
it adjusted placement and released the bowl. It later reported that a short push
moved the drawer and bowl inward, but the drawer remained partly open. These
intermediate scene interpretations come from the acting Agent, not a new private
predicate audit. Official task success stayed false.

Long 3's native receipts include 22 target arrivals, 8 Cartesian-segment failures,
11 local stalls, 4 collision rejections, one iteration-limit result and 6 completed
gripper actions. Route calls may contain multiple segments; these counts summarize
the top-level native motion receipt, not every internal segment. During recovery,
stall feedback exposed arm contact and active arm clearance constraints. The last
stall (request 68) reported an active arm constraint outside the contact target,
with **no measured end-state external contact**. That distinction prevents
equating all stalls with the previous experiment's palm/other-handle contact.

All 118 native requests have contiguous sequence-correlated public journal
receipts. New intervention messages, control settings, contact-binding retirement,
orientation selection and stall context were present in real responses. Agent
messages acknowledge obstructions and changes of approach. This establishes use
and delivery of feedback, not that feedback alone improves task success or proves
the causal origin of every controller stall.

## Network and cleanup

Codex recorded zero stream-error events in both trials; neither needed a recorded
WebSocket-to-HTTPS fallback. This is narrower than a claim of perfect network
health. Unauthenticated 30-second probes recorded:

- Goal 5: 28 rounds. One round at 18:47:08 +08:00 had timeouts on all four proxied
  HTTPS targets; direct domestic HTTPS and the local proxy TCP listener succeeded.
- Long 3: 49 rounds, zero HTTPS probe failures.
- The separate local resolver probe (`getent ahostsv4 chatgpt.com`, 5-second limit)
  timed out in all 77 rounds. Proxy HTTPS and model sessions still operated.
  The exact resolver cause was not diagnosed in this retest; direct resolution
  and proxy-mediated requests exercise different paths.

Both initial-state hashes matched the historical evaluation. Source changes were
empty, owned processes were cleaned up, the dedicated port was released, and
private authentication copies were deleted. Main worktree and global settings
were untouched. No automatic replacement trial was launched.

These selected diagnostic attempts remain separate from the frozen 40-task
pass@1/pass@2 evaluation. The outcome is **0/2 task successes, 2/2 healthy
integrations**; this small sample does not estimate a new benchmark success rate.
