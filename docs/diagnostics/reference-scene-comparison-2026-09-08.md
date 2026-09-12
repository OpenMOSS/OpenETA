# Reference-assisted scene candidate comparison

## Implementation

User requested reference images in scene segmentation/selection and a Luna experiment.
Branch `dev/huaizezheng/luna-task-recovery-2026-09-08`; existing changes preserved,
no commit/push/shared-document writes. OpenETA collaboration skill used; shared RFC
access remains unavailable. Additive output/contract wording needs collaborator review.

This is a scene-discovery/Agent-comparison change, **not** reference-image prompting
inside the SAM3 model. Existing `retrieve_asset_reference(localize=false)` explicitly
retrieves appearance references; full-agentview SAM3 `prompt="object"` supplies scene
proposals. `inspect_evidence` then:

- Uses the latest explicit lookup's local reference images (up to three), showing
  their object name and namespace. It performs no hidden network/model lookup.
- Builds one untinted reference/candidate comparison sheet covering up to eight
  scene candidates, so comparison is not limited to the first four mask-detail tiles.
  Further candidates have an explicit continuation offset. Existing four-candidate
  mask-detail pages remain available separately.
- Prioritizes the comparison sheet in the next actual model request, including under
  a one-image cap. Original candidate IDs and masks are unchanged.
- Labels references as appearance only, never scene pixels or confirmed instance
  identity. Agent choice, exact provenance, IK and collision checks are unchanged.
- Does not fall back to an older object's references after a newer unsuccessful
  lookup. Unavailable/invalid reference images leave ordinary candidate viewing usable.

Guidance now recommends reference retrieval before comparing ambiguous named objects
and checking the reference name/all plausible alternatives. It does not enforce that
tool order or automatically select the best-looking candidate.

Local targeted tests: 148 passed, including original RGB, comparison coverage and
pagination, no identity creation, no old-reference fallback, image delivery through
the real provider-body builder and prompt-length/tool-guidance checks.

## Fresh experiment

Run root: `tmp/luna-reference-scene-TRhWOP/`.
Session: `74ad399a-a2d5-4a26-a5f1-3907d73bf07a`.
Standard `agent.cli.batch_eval`, unchanged Object 0 task/seed 0, `bundle_stage3`,
Mink joint-velocity controller; main/advisor/fallback Luna only.
Per-task budgets remain 15M known tokens / 160 turns / 320 tools / 10800 seconds.
Only fresh test data uses authorized service routes. Bank lookup is name/namespace
GET plus reference downloads, with no scene upload. No oracle coordinates/identities,
source, secrets or historical trace uploads.

### Outcome: correct selection and accepted grasp input

Executed lookup → SAM3 `object` → reference comparison → select `detection_004`
→ AnyGrasp, which returned **10 host-filtered candidates**. The selected blue/orange
alphabet-soup can is partially behind the milk carton, unlike the wrong foreground
red/green can selected in previous session `555f7523…` without retrieved references.
The main Agent's selection reason explicitly compared the blue body and orange
alphabet artwork with the reference appearance. No Host semantic selection was added.

The comparison output contained **three reference views and all seven candidates**.
At selection, the actual request had nine image attachments, with
`reference_comparison.png` first, then detail/scene/reference views. Inspect:

- Comparison: `tmp/luna-reference-scene-TRhWOP/workspace/sessions/74ad399a-a2d5-4a26-a5f1-3907d73bf07a/artifacts/evidence_views/78f7bee559a54d41b68f4dd2bd3d2756/reference_comparison.png`
- SAM3 sheet: `tmp/luna-reference-scene-TRhWOP/workspace/sessions/74ad399a-a2d5-4a26-a5f1-3907d73bf07a/artifacts/sam3_images/74ad399a-a2d5-4a26-a5f1-3907d73bf07a/20260908T132050829779Z-79736969/selection.contact_sheet.png`

Advisor completed on Luna and recommended a candidate. A compile and IK preview
also completed before cancellation reached the batch. No `move_to`, finger close,
attachment/lift or official task success was accepted. Stop is a deliberate bounded
perception diagnosis, not autonomous episode completion.

Ten recorded main calls used **423493 known main tokens**; one grasp-advisor reply
reported **10304 tokens**. These are known recorded usages, not complete provider
billing. All recorded model calls used `gpt-5.6-luna`. Main call 7 had a mismatched
XML tag and was repaired; no human answer or model upgrade was used. No observed
provider overload or budget exhaustion.

Two `inspect_evidence` attempts used `observation:…:agentview` labels as file paths
and failed (one before reference lookup and another after IK). Added explicit local
feedback explaining that observation labels are not filenames and pointing to real
image/crop references or SAM3 bundles. This follow-up does not reinterpret labels,
weaken path ownership checks, or change the recorded experiment retroactively.

Later same-day correction: a full-task attempt established that these exact image
IDs are published by the Host itself. The viewer now resolves registered IDs
without guessing or weakening file ownership. The historical failure above remains
unchanged, but attributing it solely to caller misuse was incomplete. See the
[continuation record](object0-image-reference-recovery-2026-09-08.md).

### Validation and cleanup

Full suite: **2461 passed / 32 skipped / one pending reviewed-authority migration
failure**, 36 warnings, 72.60 seconds; `tmp/reference-scene-tests.log` and `.xml`.
After the label-feedback follow-up, 59 focused viewer/contract/prompt tests passed.
Catalog/readiness regenerated; approval hashes and stable interface profile unchanged.

Ctrl-C wrote `result.json` (exit 130) and rollout status `interrupted`. The partial
report preserved session ID and 369937 committed main tokens; unknown remote state
and incomplete usage remained explicit. Its cleanup receipt says **pending**, not
confirmed closed; do not convert that into a completed remote-cleanup claim.
Dedicated batch PID 2630987 and simulator PID 2630206 subsequently exited (130 / 0),
both verified absent and port 18766 free. No second paid run was launched this turn.

This is a positive single-sample comparison, **not** a success-rate estimate.
Reference-first guidance and the new comparison sheet changed together, so the
experiment cannot isolate either cause. Reference retrieval and coverage remain
Agent choices; repeated correct selection and physical-task acceptance remain open.
