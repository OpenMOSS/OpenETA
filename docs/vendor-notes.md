# Vendor Notes

## UniVTAC

- OpenETA import commit: `596492ec4c8b42133523e630cb5803ac0c608773`.
- Source: `michaelyuancb/ftp1-policy@89fa681d6c014cce28300946b7526db808e0b1c1`.
- Imported UniVTAC tree: `7e2ae9fa0d9237735fa293939bede4e0984233fc`.
- Local path: `third_party/ftp1-policy/UniVTAC`.
- License: MIT; retain the vendored `LICENSE` and upstream notices.

The tree is an upstream snapshot, not a registered OpenETA backend at the
import commit. Keep OpenETA-specific adapters and runtime wiring outside the
vendor tree unless deliberately refreshing the pinned upstream snapshot.

Current `UniVTAC-Isaac51` experiments use the separately pinned official source
described in the [Isaac 5.1 compatibility boundary](univtac/isaac51_compatibility_boundary.md).
Do not treat the vendored FTP-1-era tree and the Isaac 5.1 runtime source as the
same benchmark implementation.

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
