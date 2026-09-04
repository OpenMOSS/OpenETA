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

### UniVTAC development boundary

The `tactile-agent-for-univtac` branch vendors the official UniVTAC source at
`third_party/ftp1-policy/UniVTAC` for legacy provenance. Current experiments use
a separately installed, pinned Isaac 5.1 source and a project-scoped direct
harness under `sim/envs/univtac/` and `scripts/univtac/`; they do not import the
vendored tree as a generic OpenETA backend.

The direct harness currently supports:

```text
native reset / pre_move
        -> snapshot and tactile-pair capture
        -> operator_visible projection
        -> read-only UniVTAC MCP
        -> Codex observation
        -> trace and dashboard replay
```

The public projection carries task text, step identifiers, proprioception,
head/wrist RGB, and bilateral tactile `rgb_marker` artifacts. Privileged actor
state, tactile pose/depth, planner state, and native success remain host-only.
The UniVTAC MCP path exposes observation but no manipulation tool.

UniVTAC is still not registered in the generic OpenETA simulator registry, and
there is no Agent-controlled action translation or native-evaluation loop. The
next integration boundary is therefore executable manipulation: first validate
the complete native expert task path, then expose a small reviewed skill/action
surface and close the loop through a fresh observation and the native checker.

The research objective and evidence ladder are maintained in the
[UniVTAC research plan](univtac/research-plan.md).

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
