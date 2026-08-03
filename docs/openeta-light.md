# OpenETA-Light interface

OpenETA-Light exposes geometry and control without prescribing a manipulation
strategy. The Operator may use any returned image and compose the tools in any
order allowed by their schemas.

## Versioned context

The release profile is
[`configs/embodied/operator-context/openeta-light`](../configs/embodied/operator-context/openeta-light).
Its manifest pins every prompt, tool-description, result-contract, and renderer
component by SHA-256. At startup, OpenETA resolves the ordered components and
verifies both the component composition and manifest hashes.

Run the following command to inspect the exact resolved surface:

~~~bash
PYTHONPATH=. uv run python scripts/embodied/inspect_operator_contract.py \
  --include-content
~~~

The startup prompt contains only cross-tool semantics. Field-level details stay
with the corresponding tool description and input schema.

## Tools

### `observe`

Returns fresh Agentview, wrist, or calibrated orthographic point-cloud views.
The default is Agentview plus wrist. A solved mark can be redrawn only when its
ID is explicitly supplied in `history_point_ids`.

### `mark_point`

Marks a world-space point from any image returned by the tools:

- Agentview and wrist RGB-D clicks solve the first visible surface when aligned
  depth is valid.
- Two complementary orthographic point-cloud clicks can solve a surface or
  free-space point.
- Candidate-local front and side views can solve a point in the candidate
  `JAW/LAT/APP` volume.

The result returns an image showing only the current mark. Older points are not
accumulated in that feedback.

### `move_to`

Controls the Panda grip site with a world position, a world or current-grip-site
delta, orientation directions, and an optional gripper command. A preview uses
the same target resolver as execution. A close preview returns an immutable
`preview_id`; execution commits that exact preview or submits a corrected pose.

If motion does not reach the target, the result returns the actual grip-site
position and remaining delta. LIBERO feedback may add tiny cyan markers for
sampled final-step robot contacts; these markers are evidence, not a classified
failure reason.

### Lifecycle tools

`report_issue` records nonterminal evidence. `check_task` queries the native
LIBERO checker. `finish_episode` accepts success only after the native checker
returns true; a failed episode requires a concise reason and exhausted attempts.

## Visual feedback

Markers are intentionally small:

- current grip site: tiny magenta ring with an exact one-pixel center;
- current solved point: a micro marker on only the contributing current views;
- rejected point: a current-click micro marker;
- final-step robot contact after `not_reached`: tiny cyan dots;
- point-cloud axes: decluttered metric edge labels.

Images remain in Replay artifacts even when they are omitted from live model
context.

## Stability contract

The release regression locks:

- startup prompt content;
- public tool set and descriptions;
- public input schemas;
- resolved result-contract and rendering invariants;
- renderer-mode selection;
- content-addressed component composition.

Changing any of these surfaces requires a new reviewed profile revision and a
fresh evaluation.
