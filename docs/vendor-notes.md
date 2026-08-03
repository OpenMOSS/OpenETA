# Vendor Notes

## RLinf

- Source: `https://github.com/RLinf/RLinf.git`
- Local source commit used for the initial migration:
  `f0a6429147e4b829c0b89e00455c7c8c27d9b809`
- License: Apache-2.0. RLinf-derived source files under `sim/envs/` retain the
  upstream copyright and license headers.
- The initial environment subset was migrated into OpenETA-owned modules under
  `sim/envs/`; it is no longer maintained as a full mirror under `sim/rlinf/`.

Current layout:

- `sim/envs/`: RLinf-derived environment wrappers adapted to OpenETA imports.
- `sim/rlinf/rlinf/envs/venv/`: limited compatibility copy retained for older
  code paths; new OpenETA code should import `sim.envs`.
- Training, scheduler, data-pipeline, real-world, and world-model packages from
  upstream RLinf are not vendored here.
