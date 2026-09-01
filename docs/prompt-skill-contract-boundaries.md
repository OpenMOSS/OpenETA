# Planner Instruction Ownership

OpenETA keeps planner instructions in four layers with one semantic owner for
each kind of information. This prevents a rule from being copied into the
system prompt, multiple skills, and every tool description.

## System prompt

The system prompt owns only task-agnostic invariants:

- planner response shape and one-atomic-action decision boundary;
- Agent ownership of decomposition and recovery;
- current evidence precedence, uncertainty, and closed-loop observation;
- task-completion authority;
- infrastructure failure versus task failure;
- obedience to host safety, freshness, permission, and supervision checks; and
- the precedence of tool contracts, skills, playbooks, and memory.

It must not describe a particular perception backend, grasp estimator,
placement method, object family, task, benchmark episode, parameter name,
receipt, or bundle.

## Skills

Skills own reusable task-domain decisions: how to recognize a good grasp, when a
wrist view is useful, how to structure a carry, or how drawer pulling differs
from pick-and-place. They may name tools as capabilities, but they do not restate
request fields, output schemas, opaque identifiers, validity epochs, numeric
interface defaults, or gate repair payloads.

Skills are not trajectories and do not contain evidence from one named task,
object instance, seed, run, or benchmark canary.

## Tool contracts

The live AgentTool contract is the exclusive model-facing authority for:

- exact request parameters and mutually exclusive branches;
- structured outcomes and output fields;
- opaque references, receipts, bundles, and host-resolved private inputs;
- evidence lifetime and invalidation;
- deterministic gates and executable repair payloads; and
- controller defaults or numeric limits that affect interface behavior.

Generated contract documentation is derived from the runtime contract catalog;
skills should link decisions to capabilities instead of copying this material.

The Planner receives `openeta.agent_tool_contract.v2`, a five-field projection:

- `name`: the callable capability name;
- `description`: one or two sentences describing only the capability;
- `parameters`: the canonical JSON request schema;
- `returns`: compact semantic outcome and top-level output/reference names; and
- `semantic_limits`: short tags such as `read_only`, `endpoint_only`,
  `not_motion_authorization`, or `fresh_observation_after_call`.

The complete developer contract remains host-only. It retains detailed outcome
schemas, fact bindings, evidence lifetime, gate implementations, repair codes,
source paths, and verification evidence; neither the generated Markdown nor the
full catalog JSON is inserted into the Planner prompt.

## Playbooks and strategies

Task playbooks own experience that is safe only under an exact environment,
suite, task index, normalized task text, and optional calibration match. The
Planner receives at most one exact match as a cache-stable prior. Cross-task or
language-similarity fallback is forbidden.

Grasp strategies own reusable geometry- and calibration-scoped pose policies.
Candidate strategies require explicit Agent selection; only validated compatible
strategies may activate automatically. Task/run evidence remains in strategy
provenance and is not projected into the Agent-facing option summary.

## Context layout

Tool contracts, selected reusable skills, skill-usage metadata, and the matched
task playbook are placed in the cache-stable prefix ahead of growing conversation
history. Current observation, transitions, evidence, obligations, and working
memory remain in the dynamic turn projection.
