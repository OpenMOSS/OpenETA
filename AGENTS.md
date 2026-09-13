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
R1.12 uses `configs/univtac/grasp_classify_current_tactile_ablation.yaml`
explicitly for Grasp & Classify; current-touch omission is an ablation, not the
complete-method default.
This project rule does not authorize changing global Codex configuration.

## Query feedback protocol

Fresh UniVTAC runs use `native_eval_no_online_task_feedback_v1`: native
evaluation and automatic termination remain host-owned. Agent tools expose no
current-query task score or specific native termination reason. `check_task`
is absent; terminal feedback is `episode_ended`, and `finish_episode` confirms
voluntary/final ending without a score. Keep old configurations, traces and
results under their original protocol; never mix versions in a results table
or switch an unfinished batch mid-run.

## Mandatory neat after experiments, analyses, conclusions and designs

After every substantive experiment (including failed or paused runs), result
analysis, new or revised conclusion, and new or revised experiment design,
perform a scoped `neat-freak` pass before reporting the step as complete.
This is standing authorization to update the relevant project records; do not
leave results or decisions only in chat or merely offer to document them later.

Use the existing canonical research plan and campaign/result record. Record:

- experiments: date, scope/configuration, actual state, results, attempts and evidence paths;
- analyses: question, population/denominator, selection method, method/command,
  numerical tables and trace references sufficient to reproduce the finding;
- conclusions: verified observations, interpretations, uncertainty and counterexamples;
- designs: hypothesis, comparison, controlled variables, metrics, budgets,
  execution/approval status and the evidence needed to decide the next step.

Replace stale current status; retain historical evidence under explicit historical
headings. Link detailed records from the research entrypoint instead of copying
results into AGENTS.md or creating a report per conversation. Recording a proposed
design does not authorize execution. Preserve raw ledgers, attempts and media;
separate phases when protocols or budgets differ. State missing evidence as unknown.

Spawn one GPT-6 low scoped neat subagent with the full conversation history, an
explicit exception to ordinary clean-context read-only scouts. Pause edits to
its owned documents; review its changes before the focused commit and push.
If tool/role restrictions prevent full-history low reasoning or document edits,
state the restriction, use any permitted read-only check, and complete the edits
in the main agent without blocking this documentation requirement. Never silently
substitute a different model or history policy. Do not run new experiments in neat.
The final reply links the updated records and identifies anything still unverified.

Use [research-plan.md](docs/univtac/research-plan.md) for current intent/design
and [shot-scaling-results.md](docs/univtac/shot-scaling-results.md) for this campaign.
