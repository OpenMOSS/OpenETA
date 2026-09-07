# OpenETA-UniVTAC

This is the development branch for
[`No-518/OpenETA-UniVTAC`](https://github.com/No-518/OpenETA-UniVTAC). It builds
on OpenETA-Light and vendors the official UniVTAC benchmark under
`third_party/ftp1-policy/UniVTAC`.

The research question is whether a frozen embodied Agent can use a few
successful tactile–action–outcome examples in context to operate autonomously,
improve UniVTAC native task success, and reduce trial and error. Vision and
proprioception remain normal inputs; visual–action examples are a control for
the added value of historical touch.

Current status:

- the isolated Isaac 5.1 harness supports tactile capture, Codex MCP
  communication, real action traces, and dashboard replay;
- R1.0–R1.3 include native expert successes, model-selected restricted skills,
  and expert-assisted continuations. They establish real execution, but not
  general-tool autonomous success rates or positive tactile ICL benefit;
- R1.4 connected the general tools to a synchronous UniVTAC worker and ran
  three fresh Insert Hole no-demo episodes: native success 0/3, all evaluable.
  Two control-debug episodes are excluded. The A/B/C comparison remains unrun;
  R1.6 uses native expert successes as its demonstration source, without
  requiring human or Agent successes first;
- R1.5 preserves native gripper commands and delivers recorded segment-end
  tactile history. One unscored debug and three fresh no-demo episodes completed:
  native success 0/3, all evaluable, all Codex processes exited naturally with
  usage. Review videos and exact Agent inputs are separate in the
  [local R1.5 dashboard](http://127.0.0.1:9401/r15-autonomous). This is control and
  observation validation, not an ICL gain;
- R1.6 completed six fixed-target controller replays: original native success
  1/3, paced candidate 2/3, with measured timing and completion differences.
  Original remains the default. Two sparse R1.2 native expert successes were
  curated without recapture; these are the demonstration source, distinct from
  replay success. See the [Chinese video comparison](http://127.0.0.1:9401/r16-motion-pacing);
- A has no examples; B has visual–action examples; C adds historical bilateral
  touch to exactly B's trajectories. All three retain current vision, touch,
  proprioception, and operation history;
- R0.9.19–R0.9.21 remain historical exploration, with no further expansion or
  role in selecting the next method.

Read the [UniVTAC research plan](docs/univtac/research-plan.md) first. See
[Related Work and bibliography](docs/univtac/related-work.md) for literature
and its effect on the design,
[Architecture](docs/architecture.md) for the system boundary,
[Isaac 5.1 compatibility](docs/univtac/isaac51_compatibility_boundary.md) for
benchmark interpretation, and [Vendor Notes](docs/vendor-notes.md) for source
provenance.

## OpenETA-Light interface

**Run Codex as a visual robot operator in LIBERO.**

OpenETA-Light connects an ordinary Codex TUI to a robot simulator through one
MCP server. Codex receives a task, six typed tools, and fresh visual and
structured feedback throughout the loop. It observes the scene, marks 3D
points, moves the Panda gripper, checks the native LIBERO success condition,
and recovers from failed attempts in the same interactive loop you use for
coding tasks.

~~~text
task + versioned context
          |
      Codex TUI
          |
  OpenETA-Light MCP
          |
 observe -> mark_point -> move_to -> check_task
          |
   LIBERO images and task truth
~~~

OpenETA-Light does not provide a learned robot policy, demonstrations, or
privileged object poses to Codex. The interface exposes geometry and control;
the model decides how to use them.

## The six tools

| Tool | What Codex can do |
| --- | --- |
| `observe` | Request fresh Agentview, wrist, or calibrated point-cloud views. |
| `mark_point` | Turn clicks in any returned image into an immutable world-space point. |
| `move_to` | Preview or execute a Panda grip-site pose and control the gripper. |
| `report_issue` | Retain nonterminal failure evidence without ending the episode. |
| `check_task` | Query the native LIBERO success checker. |
| `finish_episode` | End the episode after success or after attempts are exhausted. |

Codex also receives a compact startup prompt defining the cross-tool semantics,
plus the tool descriptions, input schemas, compact results, and images. These
surfaces are versioned together under
[`configs/embodied/operator-context/openeta-light`](configs/embodied/operator-context/openeta-light)
and fail closed if a pinned component changes.

## Try the inherited interface live in the Codex TUI

### 1. Install OpenETA-Light and LIBERO

Requirements: Python 3.10+, [uv](https://docs.astral.sh/uv/), the Codex CLI,
and a Codex login.

~~~bash
git clone --branch tactile-agent-for-univtac https://github.com/No-518/OpenETA-UniVTAC.git
cd OpenETA-UniVTAC

uv sync --extra dev
export LIBERO_DIR="$PWD/third_party/LIBERO"
scripts/setup_envs.sh libero
codex login
~~~

### 2. Start one interactive task

~~~bash
OPENETA_OPERATOR_TASK="pick up the black bowl between the plate and the ramekin and place it on the plate" \
OPENETA_LIBERO_ENV_ID="openeta/libero_libero_spatial_task0-v0" \
OPENETA_LIBERO_SEED=0 \
OPENETA_OPERATOR_MODEL="gpt-5.6-terra" \
scripts/embodied/launch_openeta_light_tui.sh
~~~

The terminal becomes the normal Codex TUI. Watch its `observe`, `mark_point`,
`move_to`, and `check_task` calls as they happen. The launcher also prints a
local dashboard URL for live simulator images and retained replay artifacts.
Exit the TUI to stop world-changing services; the episode trace remains on
disk.

The lifecycle wrapper starts LIBERO and the replay services, creates a fresh
empty workspace and Codex home, then runs the equivalent of:

~~~bash
codex -C <fresh-empty-workspace> \
  --no-alt-screen \
  -m <model> \
  -c 'mcp_servers.operator.command="env"' \
  -c 'mcp_servers.operator.args=[...OpenETA-Light Gateway...]' \
  '<task plus the released OpenETA-Light prompt>'
~~~

The isolated workspace prevents repository files, previous sessions, memories,
and unrelated MCP servers from changing the demonstration. Authentication is
reused from the user's Codex login. For a custom configured provider, set
`OPENETA_OPERATOR_MODEL_PROVIDER` and `OPENETA_OPERATOR_PROVIDER_CONFIG`.

## Inspect exactly what Codex receives

~~~bash
PYTHONPATH=. uv run python scripts/embodied/inspect_operator_contract.py
~~~

Add `--include-content` to print the complete startup prompt, descriptions,
schemas, and resolved result/rendering invariants. The evaluated release is
`openeta-light@1`, with composition SHA-256:

~~~text
bc1749ac21fdfa3871b87aed77e1a41571a09fd30c4625be45a93f7d9b898399
~~~

See [OpenETA-Light interface](docs/openeta-light.md) for the image and action
contracts.

## Run LIBERO evaluation

Create a deterministic task-by-seed manifest:

~~~bash
PYTHONPATH=. uv run python scripts/embodied/libero_operator_coverage.py \
  plan-matrix \
  --output outputs/libero-spatial \
  --suite libero_spatial \
  --seeds 0 1 2 3 4 \
  --model MODEL \
  --reasoning-effort medium \
  --model-provider openai
~~~

Run independent episodes in parallel:

~~~bash
PYTHONPATH=. uv run python scripts/embodied/libero_operator_coverage.py \
  run --output outputs/libero-spatial --jobs 8 --base-port 14000
~~~

Task success comes only from the native LIBERO checker. See
[LIBERO evaluation](docs/libero-evaluation.md) for Pass@k, early-stop, and
infrastructure-validity semantics.

## Repository layout

~~~text
adapter/   Protocol and bridge types
agent/     General OpenETA agent runtime
configs/   Released Operator context and environment configuration
logger/    Episode logging and replay
scripts/   Interactive launchers and evaluation entry points
sim/       Simulator registry and backend wrappers
tools/     Operator MCP gateway and perception/control adapters
tests/     Contract and behavior regression tests
third_party/  Pinned upstream source snapshots, including UniVTAC
~~~

Simulator and optional perception service configuration is documented in
[MCP services](docs/mcp-services.md).
