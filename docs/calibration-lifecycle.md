# Embodiment Calibration Lifecycle

OpenETA treats robot, gripper, controller, camera, and grasp-to-EEF
calibrations as reviewed profiles. The main Planner must not edit
`agent/calibrations` with `python_exec`.

## Lifecycle

```text
explicit embodiment_explore session
  -> propose_calibration_profile
  -> session-local profile and proposal
  -> deterministic schema/numeric checks
  -> independent calibration review
  -> repeated canary and held-out runs with the staged profile SHA-256
  -> promote_calibration_profile(target_status=candidate)
  -> additional evidence until every validation gate passes
  -> promote_calibration_profile(target_status=validated)
```

`propose_calibration_profile` accepts
`libero.grasp_to_eef_calibration.v2` and legacy v1. The proposal must have
`status=candidate`, a complete robot/controller/environment/camera fingerprint,
a rationale, and machine-readable validation gates. Known GraspNet-to-Panda
profiles default to:

- finger-center P95 <= 5 mm;
- axis P95 <= 3 degrees;
- held-out objective success rate >= 95%;
- held-out attempts >= 20.

The tool checks JSON finiteness, calibration ID/path safety, rigid-rotation
orthonormality and determinant, translation bounds, gripper width, v2
compatibility metadata (or legacy v1 restricted geometry), fingerprint
completeness, and gate syntax before invoking the clean reviewer client. A
blocked proposal remains in the session for audit.

Version 2 profiles contain only embodiment calibration and hard physical
compatibility. Object-family pose policy, task-specific width bounds, and
exploration heuristics belong in `agent/strategies/grasp`, not in calibration.
This prevents a robot transform from becoming a global task allowlist.

## Session Ownership

Proposals and generated profiles are stored beneath the current session
calibration root. Parallel workers use:

```text
<session-workspace>/calibrations/<session-id>/
  profiles/<calibration-id>.json
  proposals/<proposal-id>.json
```

This profile is selected deterministically from the environment/robot
fingerprint, or explicitly by a canary manifest using
`metadata.calibration_profile_path`. The worker copies that file to its
read-only `tools/grasp_profile.json` and records the staged semantic
SHA-256 in every episode result as `calibration_profile_sha256`. The hash
excludes publication receipts and normalizes lifecycle status, so unchanged
calibration parameters retain one identity across proposal, candidate, and
validated stages.

## Evidence

Promotion accepts only local `[{path, split}]` references under configured
evidence roots. `split` is `canary` or `held_out`. The host reads evidence and
computes attempts and objective success; Agent-supplied percentages are not
accepted as promotion results.

Supported evidence sources are:

- `openeta.parallel_episode_batch.v2` result files whose episode metadata
  contains the exact staged profile SHA-256;
- `openeta.calibration_evidence.v1` metric artifacts with the same profile
  SHA-256, split, attempt/success counts, and measured calibration metrics.

Batch outcomes without an episode or without matching profile provenance are
excluded. Infrastructure failures do not become physical calibration failures.
Duplicate paths or duplicate file content are rejected, and custom metric
artifacts cannot override host-derived attempt counts or objective success
rates.
Candidate publication requires at least two canary attempts and one held-out
attempt. Candidate publication records failed gates without pretending the
profile is validated. Validated publication requires a previously published
candidate and every metric gate to pass.

## Publication Policy

| Supervision profile | Shared calibration publication |
|---|---|
| `standard` | Denied; session-local proposals only |
| `human_gated` | Independent review plus explicit human approval |
| `reviewed_autonomy` | Deterministic gates plus independent reviewer |

Candidate and validated files are written atomically under
`agent/calibrations/candidate/` and `agent/calibrations/validated/`. An existing
different file with the same calibration ID is a conflict, never an overwrite.
Repeating an already completed promotion returns its existing receipt without
rewriting the profile.

Calibration tools are Planner-gated to an explicitly selected
`embodiment_explore` skill. Ordinary benchmark failure, one rejected grasp,
provider timeout, or model OOM cannot open this lifecycle.
