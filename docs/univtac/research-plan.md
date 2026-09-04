# UniVTAC tactile-agent research plan

This page is the canonical research-status entry point for the
`tactile-agent-for-univtac` branch. It separates working infrastructure from
exploratory observations and from evidence that still has to be produced.

## Research question

The project is not trying to show only that a general multimodal model can
describe color changes in a tactile image. The target question is:

> Can a frozen embodied Agent use tactile–action–outcome demonstrations from
> its context to select an appropriate manipulation skill for the current
> tactile state, and thereby improve UniVTAC native task success?

Every core method must therefore end in executable manipulation and evaluation
with the task's native success checker. Read-only perception studies are useful
diagnostics, but they are not the final result.

## Terms and roles

| Term | Role in this project | Current status |
| --- | --- | --- |
| Raw tactile input | `rgb`, `rgb_marker`, tactile keyframes, or a short tactile history | Bilateral `rgb_marker` capture works in the current direct harness. |
| Tactile interpreter | Describes contact state and how it changes | Current image/difference summaries are exploratory. Octopi-1.5 is a future candidate, not an integrated dependency or validated result. |
| Temporal tactile representation | Represents strengthening, weakening, slip, sticking, or contact loss | Baseline/current pairs can be saved. A VT-MUSE-inspired representation remains future work and is not itself ICL. |
| Manipulation skill | An executable robot procedure such as align, pull, tighten grasp, pause-and-reobserve, or lower-and-regrasp | Not yet exposed to Codex for UniVTAC. The current `tactile_grounding_image_first` candidate is a read-only evidence-ordering procedure, not a manipulation skill. |
| Tactile ICL | Context examples of `tactile before -> selected skill/action -> tactile after -> outcome`, followed by a new state requiring a skill choice | No demonstration bank or tactile-ICL experiment exists yet. |

## System boundary

The current UniVTAC path is a project-scoped research harness rather than a
registered backend in the generic OpenETA simulator registry:

```text
pinned UniVTAC Isaac 5.1 runtime
        -> native task reset / pre_move
        -> snapshot and tactile-pair capture
        -> operator_visible projection
        -> read-only UniVTAC MCP images and text
        -> Codex observation
        -> trace and dashboard replay
```

The model-visible projection includes the task instruction, step identifiers,
proprioception, head/wrist RGB, and bilateral tactile `rgb_marker` artifacts.
Privileged actor state, tactile pose/depth, planner state, and native success
remain host-only. The current MCP path intentionally exposes no UniVTAC action
tool.

This direct harness does not yet provide the generic OpenETA backend lifecycle,
UniVTAC action translation, or an Agent-controlled native evaluation loop.

## Evidence status

### Demonstrated on the current development machine

- The isolated Isaac 5.1 stack can launch and shut down on the RTX 5090 using
  the project-scoped runtime compatibility path.
- The official Taxim smoke and the `grasp_classify` phase-one collection gate
  have completed in that runtime.
- Pull Out Key reset, native `pre_move`, head/wrist capture, bilateral tactile
  capture, proprioception, and trace serialization work for the tested seeds.
- Codex can receive the four native MCP images from a fresh pre-action snapshot
  and produce a retained read-only answer.
- The experiment dashboard can replay the exact MCP context, runtime events,
  model-visible answer, and separately labelled host-only diagnostics.
- Baseline/current tactile pairs can be retained around `pre_move` for later
  demonstration construction.

These facts establish an observation and evidence pipeline. They do not
establish Pull Out Key task success or an autonomous Agent baseline.

### Exploratory read-only studies

- R0.9.13–R0.9.18 established native image handoff and explored whether Codex
  could describe raw, difference, structured, and staged tactile evidence.
- R0.9.19–R0.9.20 tested prompt wording, evidence ordering, and conflict
  handling. Their useful engineering observation is that showing raw tactile
  images before derived summaries reduced blind reliance on a deliberately
  swapped summary in three constructed examples.
- R0.9.21 froze that reading procedure and captured fresh pairs for seeds
  `1000003`, `1000004`, and `1000005`. The planned semantic transfer trials
  were not run; the round is capture-only and has no transfer result.

These studies may remain as engineering diagnostics or appendix material.
They are not manipulation skills, tactile ICL, task-success evidence, or paper
main results, and they do not determine the research method.

### Not yet demonstrated

- Pull Out Key native expert success on the fixed evaluation seeds.
- A Codex-selected or Codex-executed UniVTAC manipulation action.
- An action followed by a fresh tactile observation in one closed loop.
- A demonstration bank containing tactile-before, executed action or skill,
  tactile-after, and native outcome.
- Tactile ICL, Octopi effectiveness on UniVTAC, or a VT-MUSE-inspired temporal
  representation.
- A multi-task UniVTAC benchmark comparing native success rates.

## R1.0: return to executable manipulation

The next stage starts with Pull Out Key and fixed seeds `1000000`, `1000001`,
and `1000002`.

### Native expert baseline

Run the complete native path for each seed:

```text
reset -> pre_move -> expert play_once / task body -> native check_success
```

Report the three individual outcomes and the aggregate expert success count.
Retain, at minimum:

- tactile state before the relevant expert action;
- the actual native action or action segment;
- tactile state after the action;
- the native outcome and lifecycle evidence.

This baseline comes before Agent action development. It confirms that the
task, action path, and evaluator work end to end in the same Isaac 5.1
implementation used by later comparisons.

### First Codex closed-loop episode

After the expert baseline is established, derive a small, explicit manipulation
skill library from the native expert action path. Then run one genuine loop:

```text
observe
-> select a permitted manipulation skill
-> execute that skill through the reviewed UniVTAC action boundary
-> observe the new tactile state
-> select the next skill
-> native check_success
```

The first episode is a vertical-slice validation, not a success-rate claim. It
must retain the exact observation, selected skill, executed action, resulting
tactile state, and native checker result.

## Evaluation ladder

1. Use the three fixed seeds for development and the first causal comparison.
2. Expand to ten fixed seeds after the direction is technically and
   scientifically credible.
3. Use a fixed, complete seed batch per task and condition for paper results.
4. Compare at least three UniVTAC tasks with native success rate as the primary
   outcome.

Tactile ICL begins only after real action transitions exist. Its first controls
should compare no demonstration, correct demonstration, irrelevant
demonstration, and action-swapped demonstration while keeping the frozen Agent,
task implementation, action boundary, and evaluator fixed.

## Version and benchmark interpretation

Current experiments belong to the `UniVTAC-Isaac51` implementation track.
Isaac 5.1 changes runtime, physics, initialization, control, rendering, and
tactile semantics relative to the FTP-1-era Isaac 4.5 stack. Within-version
comparisons are valid when all conditions share the same implementation, but
their success rates are not direct reproductions of FTP-1 Isaac 4.5 numbers.

See [Isaac 5.1 compatibility boundary](isaac51_compatibility_boundary.md) for
the source and interpretation boundary, [Architecture](../architecture.md) for
the system decomposition, and [Vendor Notes](../vendor-notes.md) for the
vendored legacy source provenance.
