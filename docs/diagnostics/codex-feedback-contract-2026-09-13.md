# Cross-layer intervention feedback — 2026-09-13

Worktree: `OpenETA-codex-plugin`; branch
`dev/huaizezheng/codex-plugin-smoke-2026-09-08`. Main worktree remains untouched.
This change implements the user's requirement that behavior-changing Host
operations explain themselves to the Agent across the current plugin path.
See the [contract and audit](../codex-intervention-feedback.md).

## Changes

- Native replies carry request-scoped interventions and sequence/tool correlation.
  State resets before early validation/budget/inactive gates, preventing feedback
  from an earlier action being presented as the current result.
- Direction normalization, symmetric target selection, extended loaded-motion
  horizon, speed profiles, joint projection and bounded step reduction are
  explained with the resolved settings. Unchanged baseline settings are labeled
  separately. Route events identify their segment and preserve unattempted indices.
- Host substitutions explain missing fresh observation or unknown remote action
  completion. Profile/bundle gates have bounded guidance; atomic replies retain
  filtered repair hints previously exposed only in the pick profile.
- Contact-mark and close-binding retirement are explained, including retirement
  after free motion. Out-of-envelope contact errors give the requested distance
  and existing maximum rather than only a generic validation error.
- Stalled Mink execution adds read-only binding-clearance and measured-contact
  evidence. Positive-distance margin records, robot self-contact and non-robot QP
  rows are excluded from the external-contact interpretation. Diagnostic failure
  reports unavailable; no diagnostic changes the applied action.
- The three public filters preserve bounded stall evidence and control settings;
  specific recovery messages survive the common execution-error path. Hidden
  geometry identifiers, world coordinates and constraint matrices stay private.
- `episode_status` exposes an explicitly historical previous request receipt;
  every native response is journaled. This helps diagnose/recover a lost reply
  while the Host is reachable; it does not establish remote operation completion
  or authorize replay after unknown execution.

## Validation

Artifacts: `tmp/codex-feedback-contract-20260913/`.

- Host regression group: **137 passed**, with the real stdio test selected
  separately (`host-final-tests.log`).
- Simulator regression group: **44 passed** (`simulator-tests.log`). The stall
  helper tests overlap with the Host group; these are not additive unique counts.
- Real MCP stdio call and disconnect cleanup: **1 passed**, bounded by 90 s
  (`stdio-tests.log`). No model API request was made.
- Final focused tests after route correlation/contact-feedback and contact-distance
  filtering refinements: **49 passed** (`final-focused-tests.log`); the isolated
  stall helper check also passed all 3 cases (`stall-final-tests.log`). These
  checks overlap with the regression groups above.
- Plugin and skill validators, and `git diff --check`, passed. The skill validator
  uses the simulator Python environment with PyYAML; simulator tests requiring
  SciPy run there, not in the lightweight Host environment.

Production code was replayed from two operator-only saved states from the last
Long 3 attempt, with the same Mink settings and local fixture patch. No Agent or
counterfactual action was involved:

| Saved-state prefix | Steps | Stop | Maximum per-step qpos difference from previous diagnosis |
|---|---:|---|---:|
| `1789226058684663654` | 38 | Local convergence stalled | 0.0 |
| `1789226276138276052` | 26 | Local convergence stalled | 0.0 |

Both public receipts distinguish binding palm-clearance constraints from measured
end-state palm contact outside the authorized target. `collision.detected` remains
false because neither reached the hard-stop threshold. Causal attribution remains
explicitly unestablished in the public receipt. This agrees with the separate
[operator diagnosis](codex-long3-stall-2026-09-13.md) without exposing its private
geometry identities or counterfactual solver analysis.

`replay-results.json` retains private operator results/traces; the three-filter
projection is in `public-replay-feedback.json`. Environment cleanup completed.
Plugin version: `0.1.0+codex.20260913085939`. The existing launcher installs the
source plugin into a fresh private Codex home on the next independent trial;
global configuration and the user's current plugin installation were not changed.

No full task or campaign was rerun. Historical pass@1/pass@2 results remain frozen;
these tests establish feedback behavior and unchanged local motion, not improved
task success. Changes remain uncommitted on the experimental branch. Additive
interfaces require three-person contract review before main integration.
