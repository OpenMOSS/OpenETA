# OpenETA

<p align="center">
  <img src="docs/assets/openeta-header-eta-clear.png" alt="OpenETA — Embodied Task Agent" width="100%">
</p>

[English](README.md) · [简体中文](README.zh-CN.md)

[📄 Paper](https://arxiv.org/abs/2608.03924) ·
[🌐 Project Page](https://openmoss.ai/OpenETA/) ·
[💻 GitHub](https://github.com/OpenMOSS/OpenETA)

> **A new agentic paradigm for moving intelligence from the digital world into the physical world.**

Physical intelligence is fundamentally about closing a causal loop with the world: understanding the
current state, acting to change it, observing the real consequence, updating the underlying belief,
and accumulating experience through continued interaction. Frontier agentic models have already
begun to exhibit this capability in the digital domain, where they reason over goals, invoke tools,
inspect execution results, and complete long-horizon tasks through feedback. What they still lack is
a reliable way to extend that loop into the physical world.

OpenETA explores this transition from Digital Agent to Physical Agent as a new paradigm. Instead of
reducing embodied intelligence to a single observation-to-action model, it places the Agent inside
the world's causal chain so that perception, action, verification, and learning become one continuous
process. Intelligence moves beyond processing digital information and begins to understand and
change reality.

[Architecture](docs/architecture.md) ·
[Action Pipeline](docs/agent-action-pipeline.md) ·
[Parallel Evaluation](docs/parallel-simulator-evaluation.md) ·
[Real-robot Deployment](real/README.md)

## What's New

- **2026-08-03** — **OpenETA for Codex:** We released the
  [`openeta-light`](https://github.com/OpenMOSS/OpenETA/tree/openeta-light)
  branch, which connects the Codex TUI to LIBERO through six typed tools and a
  versioned Operator context. With `gpt-5.6-sol` at medium reasoning effort,
  Codex + OpenETA reaches **70.8% Pass@1** and **90.0% Pass@5** across
  all 130 LIBERO tasks.
- **2026-07-27** — **Real-Robot Deployment:** We released the
  [`dev/real-robot-deployment`](https://github.com/OpenMOSS/OpenETA/tree/dev/real-robot-deployment)
  branch with the real-robot deployment stack, hardware interfaces, and physical-agent tooling.
- **2026-07-25** — **Physical Hello World:** OpenETA's first public release, extending the agentic
  loop from the digital domain into the physical world.

## Core Features

| Capability | OpenETA system boundary |
| --- | --- |
| Causal closed loop | Every world-changing action creates a fresh-observation obligation before reasoning, recovery, or completion can continue. |
| Replaceable Planner | Different LLMs and VLMs connect through an OpenAI-compatible interface with primary and fallback endpoints. |
| Composable physical capabilities | Perception, grasping, placement, motion, and future native policies compose through stable Tool and MCP interfaces. |
| Tool / Skill separation | Tools are host-owned atomic capabilities; Skills are readable and reviewable experience, never hidden execution. |
| Host-owned supervision | Schemas, provenance, safety gates, approval modes, and reviewers remain outside the Agent's authority. |
| Auditable Memory and Rollouts | Session working memory and immutable evidence are stored separately so success, failure, and uncertainty can be replayed. |
| Unified sim-to-real boundary | Simulators and robots share observation, action, result, and MCP lifecycle contracts. |
| Verifiable experience evolution | Experience begins as a session-local candidate and advances only after review, canary, and holdout validation. |

The core loop is:

```text
observe -> reason -> act -> verify world change -> observe again -> adapt
```

## Architecture

<p align="center">
  <img src="docs/assets/openeta-framework-black.png" alt="OpenETA framework: agentic intelligence crossing the host interface into simulators and robots, with world feedback closing the causal loop" width="100%">
</p>

Three boundaries constrain the system:

- **Cognition / execution boundary**: the Agent proposes intent, the Host decides whether to execute
  it, and the Simulator or Robot retains authority over environment truth.
- **Action boundary**: read-only capabilities may run in parallel, while world-mutating AtomActions
  execute one at a time and must satisfy the fresh-observation obligation before the next action.
- **Evidence boundary**: Tool completion, world-state change, and task success are distinct claims.
  Only trusted environment receipts, rewards, termination signals, or checkers can establish task
  completion.

## Feature Matrix

<table>
  <thead>
    <tr>
      <th>Agentic Planner</th>
      <th>Tools</th>
      <th>Simulator</th>
      <th>Real World</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td valign="top">
        <ul>
          <li>OpenAI-compatible LLM / VLM ✅</li>
          <li>Custom Planner ✅</li>
          <li>Primary / fallback endpoint ✅</li>
        </ul>
      </td>
      <td valign="top">
        <ul>
          <li><strong>Semantic Tools</strong>
            <ul>
              <li><strong>Perception</strong>
                <ul>
                  <li>SAM 3 ✅</li>
                  <li>UniDepth V2 ✅</li>
                  <li>MolmoPoint-8B ✅</li>
                </ul>
              </li>
              <li><strong>Grasp &amp; Placement</strong>
                <ul>
                  <li>AnyGrasp ✅</li>
                  <li>GraspGenX ✅</li>
                  <li>AnyPlace ✅</li>
                  <li>Contact-GraspNet</li>
                </ul>
              </li>
            </ul>
          </li>
          <li><strong>Control Tools</strong>
            <ul>
              <li><strong>Policy Adapters</strong>
                <ul>
                  <li>OpenVLA</li>
                  <li>OpenVLA-OFT</li>
                  <li>openpi</li>
                  <li>GR00T N1.6 / N1.7</li>
                </ul>
              </li>
              <li><strong>Controllers &amp; Safety</strong>
                <ul>
                  <li>robosuite <code>OSC_POSE</code> ✅</li>
                  <li>cuRobo ✅</li>
                </ul>
              </li>
            </ul>
          </li>
        </ul>
      </td>
      <td valign="top">
        <ul>
          <li><strong>Manipulation Environments</strong>
            <ul>
              <li>MetaWorld / Sawyer ✅</li>
              <li>ManiSkill / multiple robots ✅</li>
              <li>LIBERO / Franka Panda ✅</li>
              <li>RoboCasa365 / PandaOmron ✅</li>
              <li>BEHAVIOR-1K / R1Pro ✅</li>
              <li>Genesis / Franka ✅</li>
            </ul>
          </li>
        </ul>
      </td>
      <td valign="top">
        <ul>
          <li><strong>Observation</strong>
            <ul>
              <li>RealSense D400 / L515 ✅</li>
              <li>Webcam / RTSP ✅</li>
              <li>UR5e state ✅</li>
            </ul>
          </li>
          <li><strong>Motion</strong>
            <ul>
              <li>UR5e ✅</li>
              <li>Franka</li>
            </ul>
          </li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

✅ Adapted. Unmarked items are not yet adapted. Simulator backends use isolated venvs; see
[Simulation Layer](sim/README.md) for installation, environment counts, and action spaces.

## Quick Start

### Install and Run

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra dev
uv run openeta
```

Use `/provider` in the TUI to configure an OpenAI-compatible Planner, then enter a task directly.
Common commands include `/tools`, `/memory`, `/approvement`, `/run`, `/sessions`, and `/resume`.

For a bounded single-task run:

```bash
uv run openeta --once "inspect the current scene" --max-turns 8
```

### Interfaces

| Entry point | Purpose |
| --- | --- |
| `uv run openeta` | Interactive Agent TUI for provider setup, Tool inspection, execution, and session recovery |
| `uv run openeta --once "..."` | Single-task execution with a bounded turn count |
| `uv run openeta-batch --manifest ...` | Parallel evaluation with isolated episodes and human recovery |
| `uv run openeta --command preflight/run/iterate/inspect ...` | Unattended experiments and experience iteration |
| `uv run openeta-replay EPISODE_DIR --output replay.mp4` | Reconstruct a timeline and video from episode logs |
| `uv run python -m sim.mcp_server` | Simulator MCP, REST API, and Web Dashboard |
| `uv run openeta-robocasa-benchmark ...` | RoboCasa365 manifests, task suites, and benchmark utilities |

See `uv run openeta --help`, `uv run openeta-batch --help`, and each subcommand's `--help` for the
complete CLI reference.

### Simulator and Dashboard

Install one simulator backend, then start the MCP service and live Dashboard:

```bash
bash scripts/setup_envs.sh libero
uv run python -m sim.mcp_server
# Open http://localhost:8765/
```

### Evaluation

Validate a parallel-evaluation manifest without connecting to a model or MCP service:

```bash
uv run openeta-batch \
  --manifest examples/parallel_libero_eval.json \
  --validate-only
```

The repository includes LIBERO canary, train, holdout, and full-evaluation manifests. After
connecting a Planner and the required MCP services, run isolated episodes in parallel:

```bash
uv run openeta-batch \
  --manifest examples/parallel_libero_eval.json \
  --concurrency 10 \
  --approvement reviewed_autonomy \
  --output outputs/parallel-libero.json
```

Unattended experiments follow four stages: `preflight -> run -> iterate -> inspect`.

```bash
uv run openeta --command preflight \
  --manifest examples/parallel_libero_eval.json

uv run openeta --command iterate \
  --train-manifest examples/parallel_libero_six_train.json \
  --validation-manifest examples/parallel_libero_six_holdout.json \
  --experiment-id libero-skill-v1 \
  --rounds 3 \
  --approvement reviewed_autonomy \
  --on-need-human fail

uv run openeta --command inspect \
  --experiment-id libero-skill-v1
```

Process completion does not imply task success. Final outcomes are established by trusted
environment receipts, rewards, checkers, and retained rollout evidence.

## Roadmap



## Documentation

| Document | Scope |
| --- | --- |
| [Architecture](docs/architecture.md) | Boundaries across the Agent, Adapter, Simulator, and MCP layers |
| [Agent command pipeline](docs/agent-action-pipeline.md) | Command schemas, Tool contracts, AtomActions, and safety gates |
| [Simulation layer](sim/README.md) | Installation, environment registry, MCP, REST, and Dashboard |
| [Parallel simulator evaluation](docs/parallel-simulator-evaluation.md) | Concurrency, budgets, human recovery, and experience promotion |
| [Rollout data contract](docs/rollout-data-contract.md) | Session data layers and immutable evidence |
| [Code policy runtime](docs/code-policy-runtime.md) | Bounded code backends and sandbox boundaries |
| [Calibration lifecycle](docs/calibration-lifecycle.md) | Calibration proposal, review, and publication |
| [Real-robot deployment](real/README.md) | Cameras, robot arms, real-environment MCP, and safety boundaries |

## Contributing

OpenETA remains under active research and development. New Planners, Tools, simulators, robots, and
Skills should preserve the existing observation, command, and result contracts and include tests for
side effects, failure semantics, and resource cleanup. Changes to schemas, Tool contracts, task
success semantics, or safety gates must update the corresponding design documents.

```bash
uv sync --extra dev
uv run pytest
```

## Acknowledgements

OpenETA's initial simulator integrations were migrated and adapted from
[RLinf](https://github.com/RLinf/RLinf). The project builds on the open-source robotics ecosystems
around LIBERO, MetaWorld, ManiSkill, RoboCasa, BEHAVIOR, MuJoCo, SAPIEN, and OmniGibson. Its
perception and manipulation capabilities benefit from SAM 3, UniDepth, AnyGrasp, AnyPlace,
Contact-GraspNet, GraspGenX, and Molmo.

## Citation and License

Formal citation metadata will be added before the first public technical report. OpenETA is licensed
under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). Third-party components
and migrated code remain subject to their respective upstream licenses and retained source-file
notices.
