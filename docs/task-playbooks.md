# Task Playbooks

Task playbooks preserve successful exact-task experience without turning a
benchmark trajectory into a general robot skill. They complement, rather than
replace, these layers:

- `agent/calibrations/`: embodiment and frame calibration.
- `agent/strategies/grasp/`: geometry-family grasp constraints.
- `agent/skills/`: reusable task workflows and safety rules.
- `agent/task_playbooks/`: exact environment, suite, task index, and normalized
  task-text experience.

## Runtime Contract

Each session receives an isolated copy under `workspace/task_playbooks/`.
Planner context includes at most one playbook only when all exact scope fields
and any declared calibration ID match. Guidance is a prior: the agent must
re-observe the current scene and retain segmentation, safety, attachment,
placement, and official-reward gates. Stored world poses, `move_to` parameters,
and candidate ranks are rejected by schema validation.

## Learning Contract

A positive official reward can produce a session-local candidate from the
rollout tool-call ledger. A host-side reviewer verifies schema, exact task
scope, source session, objective reward, and non-executable guidance. Parallel
experiments collect only candidates from objectively successful outcomes.

For the next experiment generation, a unique best-supported candidate is
copied into the generation baseline. An exact-scope candidate merges with an
existing curated candidate, preserving curated guidance while adding new
queries, signatures, stage evidence, and source episode evidence. Conflicting
variants with equal support fail closed.

This propagation does not publish to the shared registry and does not change
`status` from `candidate`. Shared publication requires repeated canaries and
held-out validation; only then may a reviewed record move to `validated/`.

## Bowl Grasp Candidate

The Panda bowl strategy remains a candidate. It filters estimator seeds whose
native approach is not sufficiently downward, preserves useful yaw while
compiling a top-down EEF orientation, aligns against a shallow visible rim
surface instead of the cavity centroid, and inserts a host-generated
precontact waypoint before contact. These constraints require objective lift
and held-out task evidence before validation.
