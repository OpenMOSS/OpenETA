# OpenETA-Light

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

## Try it live in the Codex TUI

### 1. Install OpenETA-Light and LIBERO

Requirements: Python 3.10+, [uv](https://docs.astral.sh/uv/), the Codex CLI,
and a Codex login.

~~~bash
git clone --branch openeta-light https://github.com/OpenMOSS/OpenETA.git
cd OpenETA

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
~~~

Simulator and optional perception service configuration is documented in
[MCP services](docs/mcp-services.md).
