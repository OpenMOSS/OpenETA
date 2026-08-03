# OpenETA architecture

OpenETA separates the model-visible Operator interface from simulator control,
optional perception services, and retained evaluation artifacts.

## Operator boundary

A fresh Operator process receives the task-specific startup prompt and the six
tools declared by the active context profile. It runs in an isolated workspace
and Codex home without repository files, earlier sessions, memories, or
unrelated MCP servers.

The released OpenETA-Light profile is content-addressed. It resolves the
startup prompt, tool descriptions, result contracts, and renderer modes before
the Operator MCP server registers its public tools. Profile integrity failure
stops launch rather than falling back to another context.

## Gateway

`tools/embodied_mcp_server.py` exposes the public MCP tools.
`tools/embodied_gateway.py` owns one episode and translates each tool call into
simulator operations, geometry authoring, compact model-visible results, and
visual feedback.

The gateway keeps model-visible results separate from retained diagnostic
details. Privileged simulator state and full controller telemetry are written
to episode artifacts but are not injected into the Operator prompt or tool
results.

## Simulator boundary

`sim/mcp_server/` owns simulator sessions and routes each environment to an
isolated benchmark worker. Each evaluation attempt has its own environment
handle, seed, simulator ports, Gateway ports, and artifact root. Calls inside
one attempt remain serial; independent attempts may run concurrently.

Task success comes only from the simulator-native checker. The Gateway latches
terminal success so a later action cannot invalidate a completed task before
`finish_episode` records it.

## Geometry and visual feedback

`tools/pointcloud_pose_marking.py` back-projects calibrated RGB-D observations,
renders orthographic views, solves multi-view point constraints, and creates
current-only mark and pose-preview images. Agentview and wrist clicks use their
aligned RGB-D surface directly when valid.

The same target resolver is used for pose preview and execution. Close previews
are immutable and require an explicit preview ID to commit. Motion feedback
reports measured endpoints; LIBERO may additionally render small sampled robot
contact markers after `not_reached` without assigning a collision reason.

## Artifacts and evaluation

`logger/` retains append-only events, images, contracts, and episode
projections. Replay tools consume those artifacts without changing the live
episode.

`scripts/embodied/libero_operator_coverage.py` creates deterministic task and
seed manifests, launches independent Operators, validates provider and
reasoning-effort identity, and reports native-checker success separately from
infrastructure validity.

See [OpenETA-Light interface](openeta-light.md) and
[LIBERO evaluation](libero-evaluation.md) for the public contracts.
