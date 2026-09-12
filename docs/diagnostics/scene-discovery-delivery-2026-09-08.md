# Scene discovery delivery — implementation and acceptance boundaries

Branch: `dev/huaizezheng/luna-task-recovery-2026-09-08`. Existing uncommitted work
preserved; no commit/push or shared-document writes. The OpenETA collaboration
skill's separation of implementation and experiment evidence is followed below.
Shared RFC access remains unavailable; new public contracts need collaborator review.
See the [current delivery and remaining backlog](../todolist.md).

## Implementation

- `pick.md` and bundle-profile identity guidance now recommend full fixed-agentview
  SAM3 `prompt="object"` discovery for ambiguous names/packages. The Agent selects
  with task semantics and optional object-bank references. Named text/point prompts
  remain alternatives; neither complete segmentation recall nor identity is guaranteed.
- SAM3 candidate crops now use original RGB rather than tinted overlays. Overlays
  still show mask geometry. Initial sheets retain the existing eight-candidate cap.
- New read-only `inspect_evidence` supports exactly one input branch: a registered
  SAM3 `bundle_id` with optional nonnegative `offset`, or an exact `image_ref` under
  the active session artifact root. Bundle pages contain four candidates, stable
  original detection IDs, `next_offset`, source packet/camera and scene-epoch metadata.
  Inspection attaches a sheet/image to the next planner input and neither selects
  identity, runs inference, renews evidence nor authorizes movement. Session/hash/type,
  path/symlink, image-byte/pixel and page bounds remain checked. Failed inspection
  clears the previous requested view instead of silently displaying it again.
- All profiles publish durable SAM3 result manifests for inspection. This does not
  promote the opt-in bundle-only execution interfaces into the stable default.
- Initial catalog-like identity IDs plus `same_instance` receive explicit nonblocking
  normalization feedback: Host creates its own anchor and does not claim reuse.
  Explicit correction of a prior anchor is allowed even if the old reference verifier
  said `match`. Exact previous identity and a reason remain required; the old/new IDs
  and reason are recorded. Existing reference integrity/IK/collision checks remain.
- Canonical `open_questions` are referenced, not copied, in `unresolved_obligations`.
  This removes one duplication source, not every history/world-evidence duplication.
- Invalid grasp-advisor replies retain response-validation phase, provider/model,
  latency and whitelisted known usage. No rejected response is turned into a valid
  recommendation. Missing provider usage remains unknown. Full all-role aggregate
  accounting and richer bounded rejected-field diagnostics remain open.
- Ctrl-C in standard batch writes an atomic partial report and returns 130. Rollout
  records interruption and unknown remote state. Per-runner committed usage is a
  snapshot, not a provider billing total. A live-discovered race is fixed locally:
  a worker returning failure without EpisodeResult during cleanup must not erase the
  captured session/usage or confirmed interrupt cleanup. Worker-close and interrupt
  cleanup outcomes remain separately visible. Late worker completion cannot rewrite
  the frozen report. SIGKILL/crash/distributed cancellation are not solved here.

Public viewer maturity is **declared**, not verified. Generated catalog and local
parity evidence are distinct from unchanged reviewed-authority approval hashes.

## Local verification

Tests cover original-color pages beyond the initial visual limit, unchanged IDs and
identity/epochs, invalid input and symlink rejection, failed-view clearing, image
delivery through actual provider-body construction under a one-image cap, initial
identity feedback and correction despite old verifier match, invalid-advisor known
usage, Ctrl-C report persistence and cleanup/result races, and rollout interruption.

Full suite before the final live-race follow-up: **2458 passed / 32 skipped / one
reviewed-authority migration failure**, 36 Pillow deprecation warnings; logs in
`tmp/scene-discovery-tests.log` and `.xml`. After the race correction, 39 focused
viewer/batch/rollout tests passed. The historical review evidence is stale and the
new viewer is unreviewed; the overall migration is not green or approved.

Final full suite including that race: **2459 passed / 32 skipped / one reviewed-
authority migration failure**, 36 warnings, 68.80 seconds. Final logs:
`tmp/scene-discovery-final-tests.log` and `.xml`. All 36 tools have deterministic
request parity; generated catalog/Markdown/readiness/resolver projections are
current and structural issues are empty. Historical promotion/approval evidence
remains stale. First-selection field descriptions were also clarified after the
live run; this wording change is not part of its paid experimental evidence.
The final wording follow-up passed 68 focused viewer/contract/prompt/recovery
tests; `git diff --check` passed. No approval or shared RFC record was modified.

## Fresh standard Luna experiment (not performance acceptance)

Run: `tmp/luna-scene-discovery-oI3zN9/`.
Session: `555f7523-b1ae-424a-9855-ccb0777fe245`.
Original Object 0 instruction, seed 0, standard `agent.cli.batch_eval`,
`bundle_stage3`, Mink joint-velocity controller. Main/advisor/fallback configured
Luna only. Per-task limits unchanged: 15M known tokens / 160 turns / 320 tools /
10800 seconds. No oracle target labels/coordinates were inserted into Agent context.

Observed executed sequence:

1. SAM3 text `object` on agentview: seven scene proposals.
2. `inspect_evidence` at offset 0: successful first four-candidate page.
3. Select `detection_001`: **wrong foreground red/green can**, not the blue alphabet
   soup can partially behind milk (`detection_004` in this run's initial sheet).

The Agent did not query the object bank or request the second page. At selection,
nine recorded image attachments include the requested page first, the source scene,
the original seven-candidate sheet and individual candidate overlays/crops. Thus
the requested-page delivery worked; this sample does not establish correct semantic
comparison or that all seven individual crops were examined. Main call four proposed
grasp estimation, but diagnostic interruption prevented recorded grasp execution.
No motion, close, attachment/lift or official task success was accepted.

Four recorded main calls, all `gpt-5.6-luna`, **142130 known main tokens**; durations
15.9 / 11.3 / 15.8 / 11.9 seconds. Three committed actions accounted for 104474
tokens at interruption; the later recorded fourth model response explains the
difference. No observed provider overload, format-repair failure or budget exhaustion.
These totals are not complete external billing.

Ctrl-C produced `result.json` and rollout status `interrupted`. The first live
partial report exposed the cleanup race described above: it retained a failed worker
outcome but lost the top-level session/usage because EpisodeResult was absent. The
rollout retained the session and usage and confirmed `close_state=closed`, `ok=true`;
the worker's separate close attempt reported already-in-progress. The fix has local
deterministic coverage, **not another paid live acceptance**. Original experiment
artifacts were not rewritten to pretend the fix was already present.

Batch PID 2612612 exited 130. Dedicated simulator PID 2611953 was closed; both
processes absent and port 18766 verified free. No second paid repeat was launched.
Only new simulation/test data used authorized service routes; no source, secrets or
historical trace upload. Object bank was configured but not queried in this sample.

## Remaining acceptance

Scene discovery and explicit viewing now exist in the real batch path, but the
correct-object milestone remains **open**. Do not continue to physical acceptance on
this wrong selection or report this as a failed grasp. Next diagnosis should address
the Agent's comparison/uncertainty handling across all relevant candidates and catalog
references without adding a Host semantic veto. Approach/close/lift, full Object 0,
Spatial 0 and Long 9 remain open alongside the implementation backlog in TODO.
