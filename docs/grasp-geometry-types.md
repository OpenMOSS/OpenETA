# Geometry hints: selection vocabulary and compiler extensions

2026-09-06, HV-05 partial implementation on the personal refactor branch.
The schema/projection correction is a **candidate requiring three-person review**;
it does not broaden the production authority allowlist or renew historical canary
evidence. The catalog's inherited maturity labels are not approval of this delta.

## Current contract

`agent/tools/grasp_types.py` owns the immutable gross-geometry vocabulary. The
SAM3 selection gate, request schema, and registry hint derive from that source:

```text
apple, articulated_handle, bowl, boxed_item, drawer_handle, lying_bottle,
other, unknown, upright_bottle, upright_can
```

`select_sam3_detection.target_geometry_family` is optional. Its schema accepts
these canonical values and the empty string. An omitted/empty hint is unspecified;
`unknown` likewise does not add a geometry hint to the selected detection. `other`
is retained as an explicit generic category. No new vocabulary entries were added.
The direct memory API preserves its old trim/lower compatibility behavior, so
`"  BoWL  "` still normalizes to `"bowl"`; newly authored schema-validated requests
must use canonical spelling. Non-string values and unknown labels are rejected
without consuming pending selection evidence.

Example selection, using actual session-owned IDs:

```json
{
  "sam3_result_id": "sam-geometry",
  "detection_id": "detection_000",
  "target_geometry_family": "other"
}
```

Existing identity confirmation rules still apply when selecting new evidence for
an established target. This abbreviated example does not waive those gates.

`compile_grasp_seed.target_geometry_family` deliberately remains an open string,
as does its legacy `target_class` alias. Its schema lists the shared vocabulary as
examples, **not** an enum. A compiler extension such as `"mug"` or `"custom:tool"`
can use the existing generic calibrated transform when no strategy matches. This
is not proof that a part is graspable, the strategy is validated for that object,
or the motion is authorized. Existing calibration, geometry, freshness, strategy,
IK, and collision checks remain in force.

## Scope and remaining decisions

The original mismatch was the selection schema advertising any string while its
runtime gate rejected unknown values. Making the compiler equally restrictive
would remove an intentional compatibility/extension path, so this patch does not
do that. Tests cover every canonical selection value, the final Agent projection,
malformed hints with unchanged pending evidence, generated document parity, and
actual compilation of unfamiliar labels using generic fallback.

`mug` and `mug_handle` are still not accepted as selection vocabulary entries.
When uncertain, omit the optional hint; use `other` only when that description is
truthful. This does not force an incorrect can/bottle classification, but also
does not solve the whole-object-mask versus handle-mask problem.

HV-05 remains open for a reviewed object/part taxonomy or selection extension
contract, coordinated with HV-04's object identity versus graspable-part evidence.
Adding a word alone must not silently establish cross-view identity, replace
whole-object collision geometry, or grant an articulated-object strategy to a
cup handle. No live LIBERO regression or deployed-service schema rollout is
claimed by these unit/contract tests.
