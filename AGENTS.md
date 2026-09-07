# OpenETA-UniVTAC agent instructions

The current research direction and evidence boundaries are maintained in
[the UniVTAC research plan](docs/univtac/research-plan.md).

When a paper is actually cited or adopted, update
[Related Work](docs/univtac/related-work.md) and its BibTeX with the verified
source/version, reading depth, and concrete effect on the design. Keep one
record per contribution; no separate document per paper is required.

## Current execution model

New experiment operators and execution/review/neat subagents use `gpt-6-astra`
with reasoning effort `low`, unless the user changes this decision. Pass both
explicitly to Codex (`-m gpt-6-astra -c model_reasoning_effort="low"`); never
rely on global defaults or substitute another model. Keep historical run/model
labels intact. The current official-demonstration runner defaults to
`configs/univtac/new_seed_tactile_icl.yaml`; older configs are historical.
R1.10 is invoked explicitly with `configs/univtac/current_tactile_ablation.yaml`;
its no_live projection is an ablation, not a new default for the complete method.
This project rule does not authorize changing global Codex configuration.

## Neat after each experimental step

After every substantive experimental step, spawn one subagent that forks the
current main conversation history and performs one scoped neat pass. Use GPT-6
with low reasoning effort (`gpt-6-astra`, `reasoning_effort="low"`); inherit the full conversation, not just
a handoff summary. This is an explicit exception to the ordinary scout rules
requiring the default model, `fork_turns="none"`, and read-only exploration.

The neat agent synchronizes existing project documentation with that step's
confirmed decisions and actual evidence, separating completed work, plans, and
unverified claims. Prefer local doc-neat, otherwise the available neat skill or
direct Markdown edits. Keep the pass within the step's scope; preserve user
changes, do not run experiments, and do not create duplicate per-round reports.
The main agent pauses edits to the same documents, reviews the returned changes,
and includes them in the step's focused commit and push. Do not silently replace
the requested model or history fork if the current tools cannot provide them.
