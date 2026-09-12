# Default Mink for Codex experiments — 2026-09-09

The user selected Mink as the default after the successful recorded-target
replay. This change applies to the isolated Codex experiment entry points;
shared simulator defaults and the original checkout are untouched.

- New `scripts.codex_sim_server` starts a dedicated simulator with
  `mink_joint_velocity` by default, overriding an ambient OSC profile. It uses
  the clone's `tmp/codex-mink-deps` overlay unless explicitly configured otherwise.
  Missing or incompatible overlays fail setup rather than selecting OSC.
- `scripts.codex_plugin_smoke` defaults to expecting Mink, forwards
  `--expected-controller` to the Host, and records the expectation in its summary.
- After environment creation/reset, the Host checks the actual controller ID
  before exposing tools; mismatch or missing identity raises and closes the
  created environment. Host status includes the measured `controller_id`.
- Intentional OSC comparisons require `--controller osc_pose` on both the server
  wrapper and experiment launcher. Direct Host diagnostics can omit the optional
  expectation check for other backends. No already running service is reconfigured.

Validation: 45 tests passed across the controller-default, Host, motion-hook and
launcher tests, including the real MCP stdio integration. Tests verify default
agreement, propagation to Host arguments, rejection of missing/wrong runtime
identity, overlay failures, and explicit OSC selection. Both script-path and
module CLI entry points were checked.

`tmp/codex-mink-default-01/report.json` records the live startup check. The parent
environment deliberately specified `OPENETA_LIBERO_CONTROLLER_PROFILE=osc_pose`;
the new service wrapper received no controller flag, and the Host reported
`mink.robosuite_joint_velocity`. Only `episode_status` was requested: zero motion
calls, zero Host turns/tools and no model inference. The dedicated server exited
and port 18778 was released. This is configuration validation, not another task
or controller-performance sample.

The 53 inherited snapshot files remain unchanged. Implementation, setup guide,
tests and this record are confined to branch
`dev/huaizezheng/codex-plugin-smoke-2026-09-08` in `OpenETA-codex-plugin`.
No commit, push or shared-document update was made. See the
[setup guide](../codex-plugin-smoke.md) and
[preceding Mink replay](codex-mink-replay-2026-09-09.md).
