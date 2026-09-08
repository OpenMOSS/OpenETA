# Stages 2/3 Luna rerun after handoff discovery repair

Preceding interrupted run and root cause:
[initial staged experiment](bundle-stages23-luna-2026-09-07.md).
No historical trace was imported into the new Agent contexts. Runtime code is
frozen during this run. Default stable profile and reviewed authority hashes
remain unchanged; these are explicit experimental profiles, not promotion.

## Repair validation

- Bounded handoff discovery preserves the latest current source of each kind,
  then fills remaining slots by recency. No new execution authority.
- IK summary carries its target pose/signature and parent target bundle.
- All migrated consumers and trajectory composition validate the same registered
  manifest ID pattern. No trimming, alias repair or native fallback.
- Focused checks: 58 passed. Full suite: **2414 passed / 25 skipped / one existing
  reviewed-authority failure**, 36 warnings, 70.67 s. Remote integrations disabled.
  Local evidence `tmp/bundle-stages23-discovery-tests.xml` and `.log`.
- Manifest validate-only passed. No claim of full task or review acceptance.

## Configuration and ownership

Root `tmp/batch-luna-stages23-discovery-tv2f3f/`, manifest `manifest.json`.
Standard `agent.cli.batch_eval`, batch ID
`standard-luna-stages23-discovery-20260907`, concurrency 1, provider concurrency 1.
Two sequential fresh LIBERO Object 0 / seed 0 episodes with original task text:
stage 2 then stage 3. Main/fallback Luna only; authorized provider and configured
small-model services only. Each episode: 15M known cumulative tokens, 160 planner
turns, 320 tools, 10800 seconds. These are staged interface checks, not a
success-rate estimate or controlled causal comparison.

Dedicated simulator at 127.0.0.1:18766, Mink joint-velocity profile, one GPU 0
worker, existing `/tmp/openeta-mink-canary-min` dependency overlay. Simulator
timeout 22500 seconds covers both episode budgets. Owned server PID 1718468,
batch client 1718901, initial worker 1719063.

Runtime fingerprint after cleanup/repair and before rerun (SHA-256):

| File | SHA-256 |
| --- | --- |
| agent/runtime/interface_profiles.py | d2c23c564e5826ad559c9da19b35c74e0cbb66af368206c3f46a18d60cd788fc |
| agent/runtime/memory.py | ec2f7af294e686f3e828b097860d43927afe53d6b5db5de00235b784b91b6856 |
| agent/tools/bundle_proposals.py | 4ae32528f3e1e1812a2827053bc1141860592c306ce548dc701518c8e1f6a12e |

## Results

Stage 2 Agent session `d591ac9b-9fd7-4264-815a-d56a7e567ce7` ended naturally
with `planner_validation_failed`, `terminated=false`, `truncated=true` after
3 episode steps, 6 model requests and 2 dispatched MolmoPoint calls. Known main
tokens 130450, elapsed 256.695 s, four planner rejections across the run.
The final action exhausted three validation attempts: grasp estimation with raw
pixel/packet fields, SAM3 with MolmoPoint-style `sources`, then mismatched XML
tags. This is not token exhaustion, motion failure or a performance success.
No candidate bundle was generated, so this episode did not exercise the repaired
long-history discovery path in a live Agent run.

Stage 3 Agent session `06761774-fd19-4242-bfe0-3f79322d6a59` also ended with
`planner_validation_failed`, `terminated=false`, `truncated=true`, 3 episode
steps, 7 model requests / 7 HTTP attempts, 120272 known main tokens and two
dispatched SAM3 calls (text: no detections; points: detections). Five planner
rejections overall. Final selection attempts were:

1. Leading space in the correct manifest ID plus `candidate_id` instead of
   `detection_id`: both rejected by the declared stage-3 schema.
2. Exact manifest ID and correct detection field, but geometry family `can`
   instead of allowed `upright_can`: rejected.
3. Correct geometry family, but newly added null identity fields where optional
   strings are required if present: rejected. Omitting those fields is valid.

Actual provider requests contained all migrated bundle-only schemas, including
the common ID pattern, and the new trajectory composition tool. SAM3 results
were durably registered as manifests and the Agent copied the right ID after
feedback. **Selection was never dispatched**: this run does not demonstrate
the native grasp/placement wrappers or trajectory execution in a real episode.

Final standard batch `result.json`: two failures, zero successes, 493.016 s.
Stage-2 outcome 257.620 s; stage-3 outcome 235.396 s. Thirteen main requests /
thirteen HTTP attempts, 250722 known main tokens; no provider retry/error in
this rerun. Neither task exhausted its token/turn/tool/wall-clock budget.
There were no motion or gripper calls. The 0/2 outcome is not a success-rate
estimate. First interrupted run plus this rerun used **1,695,366 known main
tokens**; this excludes unrecorded failed-advisor usage, not total billable cost.

## Post-run feedback repair (not part of recorded batch)

After cleanup, experimental validation feedback now separately lists required
fields and says optional fields may be omitted, not filled with null unless the
schema permits it. Unsupported response-name feedback now lists ask_human,
talk and task_complete. Admission, three-attempt limit and task budgets remain
unchanged. No automatic XML/JSON repair, alias mapping, null stripping or
execution. Focused regression: 182 passed. Final full suite after this change:
**2415 passed / 25 skipped / one existing reviewed-authority failure**, 36 warnings,
69.87 s (`tmp/bundle-stages23-handoff-tests.xml` / `.log`). `git diff --check`
passed. This small feedback change has not
been tested with another paid run; do not attribute the recorded results to it.

## Handoff / remaining acceptance

- Stage 2/3 implementation and local coverage exist, but neither end-to-end
  stage has passed a fresh autonomous task. Do not promote or remove legacy yet.
- Discovery-loss regression is locally covered across 40 pose outputs; the
  post-repair paid episodes stopped too early to re-exercise that long path.
- Remaining perception entry contracts still differ (MolmoPoint `sources`,
  SAM3 packet/point fields). Optional fields and XML syntax remain failure modes
  despite actual schemas being present. More retries alone is not proven to fix
  these; no retry/budget change was silently introduced.
- Initial semantic identity can be wrong with high claimed confidence. Identity
  continuity and correctly typed bundles cannot repair a wrong initial target.
- Failed advisor calls need better recorded diagnostics/usage; no evidence here
  supports calling those failures deliberate geometric abstention.
- Review authority and interrupted-batch durable finalization remain pending.

## Semantic target confound

MolmoPoint pointed at the left blue can around (128, 243), while Luna's rejected
grasp request specified the foreground red/green can around (274, 337).
Local image inspection and the installed LIBERO asset textures confirm blue
`alphabet_soup` versus red/green `tomato_sauce`:
`/tmp/LIBERO/libero/libero/assets/stable_hope_objects/{alphabet_soup,tomato_sauce}/texture_map.png`.
The earlier interrupted stage-2 run selected the red/green can's mask at
[253,302,299,369], not the requested alphabet-soup instance. Its IK failures
therefore do not establish intended-target convergence coverage. The discovery
defect remains independently reproduced. Asset inspection was diagnostic only;
no oracle identity, asset textures or corrective instructions were sent to either
Agent session.

## Cleanup

Both outcomes include `cleanup.ok=true`, `close_state=closed`, remote `ok=true`
and empty errors. Client PID 1718901 exited 1 (batch failures); then the owned
server PID 1718468 was stopped with SIGINT and exited 0. Exact client/server/
worker 1719063 PIDs and port 18766 were verified absent. No unrelated service
was touched. No commit, push, shared-document write, default-profile switch or
legacy-implementation removal was performed.
