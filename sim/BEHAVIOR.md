# BEHAVIOR-1K integration

OpenETA pins the official BEHAVIOR-1K `v3.9.0` release (commit
`6559858f7c814143f08be27830d24fac16a12058`). That release installs
OmniGibson 3.9.0 on Isaac Sim 5.1.0 with Python 3.11 and PyTorch 2.7/cu128.
The complete machine-readable pin is in
`sim/envs/behavior/benchmark_versions.json`.

## Host requirements

The BEHAVIOR page lists Ubuntu 22.04+, 32 GiB RAM, an NVIDIA RTX 2070 or
newer, and 8 GiB VRAM. Its pinned Isaac Sim 5.1 backend has stricter current
tested requirements: RTX 4080, 16 GiB VRAM, and Linux driver 580.65.06. Isaac
Sim must be able to access `/dev/nvidia*`; a
working `nvidia-smi` in the host shell does not prove that a sandboxed worker
has GPU access. The installer accepts NVIDIA's Isaac Sim EULA and the
non-commercial BEHAVIOR dataset license when it is run non-interactively.

On the current integration host, driver 570.86.10 is below NVIDIA's current
tested 580.65.06 entry, but the real Isaac Sim 5.1 headless Vulkan/PhysX app
has been started and shut down successfully on the RTX 5090. Preflight keeps
the version difference as a warning rather than claiming a launch failure.

## Install and diagnose

```bash
bash scripts/setup_envs.sh behavior
python scripts/behavior_preflight.py --output outputs/behavior/preflight.json
sim/venvs/behavior/runtime/bin/python scripts/behavior_isaac_smoke.py
```

The isolated conda prefix is `sim/venvs/behavior/runtime`; the fixed source
checkout and downloaded datasets live under `sim/venvs/behavior/src/BEHAVIOR-1K`.
The latter are ignored by git.

The preflight checks the source commit, package discovery, Torch CUDA access,
NVIDIA device nodes, disk/RAM, and all three official dataset directories:

- `omnigibson-robot-assets`
- `behavior-1k-assets`
- `2026-challenge-task-instances`

## OpenETA smoke test

```bash
source sim/venvs/behavior/activate_extra.sh
sim/venvs/behavior/runtime/bin/python scripts/behavior_smoke.py \
  --task picking_up_trash --seed 0
```

This exercises the real chain `gym.make -> OmniGibson Environment -> reset ->
zero-action step -> native BDDL checker -> render`. It writes a result JSON
and short MP4 under `outputs/behavior/smoke`.

The task specs are registered statically, so environment discovery works
without importing Isaac Sim:

```python
from sim.env_registry import list_envs
assert len(list_envs(env_type="behavior")) == 1016
```

The pinned BDDL 3.7 catalog contains 1,016 activity definitions (the official
release retains the BEHAVIOR-1K name). The installed 2026 challenge bundle
provides canonical offline scene instances for 100 benchmark tasks; the direct
adapter reads its `available_tasks.yaml` to select the correct scene and R1Pro
start pose. OpenETA also retains RLinf's older 50-task manifest with curated
natural-language instructions. Catalog registration does not claim that all
1,016 definitions have cached scene instances: definitions outside the
challenge bundle require online object sampling. The default controller is the
official R1Pro configuration and exposes OmniGibson's flattened continuous
action vector.
Task success and episode termination remain OmniGibson/BDDL-native; OpenETA
does not substitute a synthetic checker.

## Worker architecture

`BenchWorkerManager` starts BEHAVIOR with
`sim/venvs/behavior/runtime/bin/python`; the main Agent process never imports
Isaac Sim. `BehaviorDirectEnv` is intentionally separate from the vendored
RLinf `BehaviorEnv`: the latter requires Ray process-group metadata and is
suited to distributed training, while the direct adapter is a normal single
environment Gymnasium path for interactive evaluation.

Isaac Kit is process-global and main-thread-affine. The BEHAVIOR worker runs
uvicorn in a service thread and marshals create/reset/step/render calls onto a
synchronous process-main-thread executor; `render_all` is serialized through
the same path. Each worker is single-environment and single-use: HTTP close
acknowledges first, then `BenchWorkerManager` retires the process. Standalone
direct environments call the official `og.shutdown()` lifecycle hook.
