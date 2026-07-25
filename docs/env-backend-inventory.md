# RLinf Environment Backend Inventory

This document summarizes the simulator/runtime backends used by the
RLinf-derived environment wrappers under `sim/envs`.

## Summary

The environments that are clearly based on the robosuite / MuJoCo ecosystem are:

1. `robocasa`: directly creates environments with `robosuite.make(...)`.
2. `libero`: creates `libero.*.envs.OffScreenRenderEnv`; LIBERO is part of the
   robosuite / MuJoCo manipulation ecosystem, although this wrapper does not
   directly import `robosuite`.

Most other environment wrappers already expose a Gym-like top-level interface,
but their underlying engines, reset/step signatures, vectorization model, tensor
device placement, and process lifecycle differ significantly.

## Backend Table

| RLinf env type | Backend / engine | Evidence in local code | Notes |
|---|---|---|---|
| `robocasa` | robosuite / MuJoCo | `sim/envs/robocasa/robocasa_env.py` imports `robosuite` and calls `robosuite.make(...)`. | Uses subprocess isolation to avoid OpenGL context sharing issues. |
| `libero` | LIBERO `OffScreenRenderEnv`, robosuite / MuJoCo ecosystem | `sim/envs/libero/libero_env.py` and `sim/envs/libero/venv.py` construct `libero`, `liberopro`, or `liberoplus` `OffScreenRenderEnv`. | Supports standard, Pro, and Plus variants through dynamic import routing. |
| `metaworld` | MetaWorld / MuJoCo | `sim/envs/metaworld/metaworld_env.py` imports `metaworld` and creates `gym.make("Meta-World/MT1", ...)`. | Uses custom subprocess vector env and task reconfiguration. |
| `frankasim` | FrankaSim / MuJoCo | `sim/envs/frankasim/frankasim_env.py` imports the external FrankaSim environment. | Wrapper manually manages a list of Gym envs. |
| `d4rl` | D4RL / Gym, often MuJoCo-based depending on task | `sim/envs/d4rl/d4rl_env.py` imports `d4rl` and creates tasks with `gym.make(task_name)`. | Also covers non-MuJoCo D4RL task families depending on `task_name`. |
| `maniskill` | ManiSkill / SAPIEN | `sim/envs/maniskill/maniskill_env.py` creates the env through `gym.make(...)`. | Supports batched GPU envs and tensor observations. |
| `isaaclab` | Isaac Lab / Isaac Sim | `sim/envs/isaaclab/tasks/stack_cube.py` launches `isaaclab.app.AppLauncher`. | Runs the Isaac Lab env in a subprocess; observations are CUDA tensors. |
| `polaris` | PolaRiS + Isaac Lab / Isaac Sim | `sim/envs/polaris/polaris_env.py` inherits `IsaaclabBaseEnv` and imports `polaris.environments`. | Currently restricted to `num_envs=1` per worker because the renderer is not vectorized. |
| `behavior` | OmniGibson / Isaac Sim | `sim/envs/behavior/behavior_env.py` creates the OmniGibson environment. | Has Isaac Sim / OmniGibson lifecycle constraints. |
| `habitat` | Habitat-Lab / Habitat-Sim | `sim/envs/habitat/habitat_env.py` imports Habitat components. | Primarily navigation-style environments. |
| `calvin` | CALVIN / `calvin_env` | `sim/envs/calvin/calvin_gym_env.py` constructs the external CALVIN env. | Uses subprocess vector env and CALVIN task-oracle utilities. |
| `genesis` | Genesis | `sim/envs/genesis/genesis_env.py` imports `genesis as gs` and builds a `gs.Scene`. | Uses Genesis batched scene execution. |
| `robotwin` | External RobotWin vector env | `sim/envs/robotwin/robotwin_env.py` imports RobotWin. | The physics backend is hidden behind the external package. |
| `roboverse` | MetaSim abstraction | `sim/envs/roboverse/roboverse_env.py` delegates to MetaSim. | Backend is selected by config. |
| `embodichain` | External EmbodiChain env | `sim/envs/embodichain/embodichain_env.py` resolves an EmbodiChain config. | The concrete simulator backend is external. |

## Gym Unification Assessment

It is feasible to expose all of these environments through a unified Gym-like
adapter, but it should not be done by forcing every backend to behave like a
strict Gymnasium environment internally.

RLinf already wraps most top-level environments as `gym.Env` or `gymnasium.Env`.
However, the wrappers are not fully uniform:

1. Some underlying envs return old Gym 4-tuples:
   `obs, reward, done, info`.
2. Some return Gymnasium 5-tuples:
   `obs, reward, terminated, truncated, info`.
3. Some observations and rewards are NumPy arrays; others are CUDA
   `torch.Tensor`s.
4. Some environments are single-env wrappers; others are batched or vectorized
   by construction.
5. Several embodied wrappers expose `chunk_step`, which is important for VLA and
   Code-as-Policy workflows but is not part of the Gym API.
6. Isaac Sim, OmniGibson, Habitat, robosuite, and MuJoCo backends have different
   process, EGL/OpenGL, GPU, and close/reset lifecycle requirements.
7. `realworld` environments should not be treated as normal simulators because
   they control physical hardware.
8. `opensora_wm` and `wan_wm` are world-model environments, not physics
   simulators.

## Recommended OpenETA Direction

OpenETA keeps the RLinf-derived wrappers isolated under `sim/envs` and exposes
them through a separate adapter layer instead of forcing every backend into a
strict Gymnasium implementation internally.

Recommended layering:

1. RLinf native env wrapper: preserve existing backend-specific logic.
2. Gym compatibility shim: normalize 4-tuple vs 5-tuple returns, tensor vs NumPy
   placement, single vs vectorized envs, and optional `chunk_step`.
3. OpenETA `SimulatorAdapter`: expose the stable OpenETA methods
   `reset`, `observe`, `step`, and `close`.
4. OpenETA embodied protocol: convert simulator observations into
   `EnvObservation` and simulator results into `StepResult`.

For the first real simulator integration, `libero` and `robocasa` are the best
starting points because they are both close to the robosuite / MuJoCo
manipulation domain and are likely to expose the largest practical issues around
offscreen rendering, process isolation, reset state management, action
conversion, and observation normalization.
