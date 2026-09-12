# Experimental atomic Codex tools

The isolated Codex plugin accepts `--tool-profile atomic` on its launcher and
Host. The existing pick profile remains available. Atomic mode exposes only
observe, mark_point, move_to, gripper_control, episode_status and finish_episode.
No SAM3 / AnyGrasp endpoints are configured and no AnyPlace tool is exposed.
The private installed snapshot contains only the atomic skill; its complete
short body is embedded in the operator prompt because shell/file tools are off.

Reference: OpenETA-Light, OpenMOSS/OpenETA `openeta-for-codex`, commit
`ddf900c03ee478f334c19def49cdb9e5042c7827`. Reused design concepts are first-surface
RGB-D marks, explicit grip-site direction axes, geometric previews and compact
measured feedback. This is an adaptation over the current Host, not an import
of the reference gateway or controller.

## Geometry and public contract

All coordinates and deltas are metres. Pixels are zero-based original-image
coordinates. Cropped views carry an explicit original-pixel mapping (2x display
scale). Marking requires a current packet and exact camera ID. The point is
immutable world geometry, not an object identity, centre, tracked point or grasp.
Depth discontinuities are reported and cannot supply contact authorization.

Example calls (point IDs below are placeholders for actual returned IDs):

```json
{"tool":"mark_point","arguments":{"source_packet_id":"obs-0000","camera_frame_id":"agentview","x":320,"y":250}}
{"tool":"move_to","arguments":{"point_id":"point-returned-id","offset_m":[0,0,0.08],"approach_world":[0,0,-1],"jaw_world":[1,0,0],"preview":true}}
{"tool":"move_to","arguments":{"delta_m":[0,0,0.01],"delta_frame":"world"}}
{"tool":"gripper_control","arguments":{"action":"close","contact_point_id":"point-fresh-id"}}
```

Approach specifies Panda grip-site +Z, jaw direction +X. The Host normalizes
directions, orthogonalizes the jaw hint, and rejects degenerate frames. Position
forms are absolute XYZ, point+world offset, or a world/current-grip-site delta.
Independent moves implement waypoints with feedback between segments. This first
implementation supplies RGB-D and local crops; free-space targets use offsets or
numeric positions. Orthographic/candidate-local multiview marking is not ported.

Preview renders the proposed frame without motion or IK. Executing always uses
the existing fresh-IK hook, exact motion receipt, Mink and collision gates.
The preview is not contact/retention prediction. The renderer illustrates an
80 mm maximum jaw span, not a physical mesh collision simulation.

## Contact provenance: additive experimental contract, needs collaborator review

The simulator now also accepts the host-private schema
`openeta.model_point_contact.v1` with source_kind=model_rgbd_point, Agent session,
point/packet/camera IDs, target_anchor_world_xyz and waypoint_role=grasp_contact.
The public model cannot provide this block or an object name. The Host creates
it only from a retained mark; target grip-site must be within 100 mm. Old contact
measurements expire after contact attempts or gripper actions. One immediately
preceding contact move may bind close; opening/closing retires that binding.
The private resolver compares the exact executing pose (including rotation,
allowing numerical quaternion sign equivalence) against the pending request.

Simulator association retains its existing non-receptacle geometry and ambiguity
rules, with a 20 mm anchor-to-bounds limit and 150 mm centre envelope for this
new provenance. As in the current Mink pipeline, the gripper subtree alone may
contact the resolved target; arm, other world geometry and carried-object checks
remain active. MuJoCo geometry is privileged collision infrastructure and is not
returned as a model grasp answer. This association is not a segmentation method
or proof of attachment. The simulator trusts its Host-private transport, as it
does for compiled-grasp authorization; the Agent and simulator session IDs are
distinct identifiers.

The existing compiled-grasp contract is unchanged. No shared RFC document has
been edited. The additive simulator contract needs collaborator review before
integration into the shared main harness.

## Validation and accounting

`tests/test_codex_atomic.py` covers calibrated depth projection, arbitrary
orientations, quaternion half-turns, stale point/image handling, exposed tools,
private-state exclusion, exact scoped contact grants and the normal IK path.
Existing collision, attachment, Host and launcher checks accompany it.

Native requests include geometry, preview and invalid calls. Physical moves
consume up to three ordinary Host turns/tool calls for propose, IK and execute.
`host/atomic-commands.jsonl` records point provenance, inputs, feedback and
episode-relative start/end times; normal Host command/rollout records remain.

No-model simulation canary: `tmp/codex-atomic-canary-01/report.json`. Sensor mark,
preview, 10 mm lift in free space, open and crop succeeded. Mink reached in two
steps, position error 0.946 mm, with explicit collision coverage and no detected
collision. This is not a grasp/task-success sample. The dedicated server exited
and the environment returned cleanup success. The initial canary returned the
duplicate render camera; the final atomic view filter keeps agentview and wrist.

### Fixture contact and failure receipts (2026-09-10)

The Mink experiment now resolves model surface points against live articulated
fixture collision parts, including LIBERO drawer handles. A contact exemption
covers the selected collision geom and gripper only. Closing retains that
binding for subsequent pull waypoints; opening/reset clears it. A fixture is
not a carried free object. The public input schema is unchanged.

Motion feedback includes `physics_executed` (true/false/null), actual executed
steps, a specific `reason_code`, and an allowlisted `recovery` action/message.
`motion_dispatched` records request dispatch, not proof of physical execution.
Zero-step failures preserve contact marks; unknown/positive execution remains
conservative. Private geometry names and raw error strings stay inside the Host.
The new experimental output fields/private fixture binding require interface
review before merging into the shared harness. Validation and limits are in
`docs/diagnostics/codex-fixture-feedback-fix-2026-09-10.md`.

### State records and checked gripper execution (2026-09-10)

The dedicated server enables operator-only snapshots by default under
`tmp/codex-private-state/<server-start>/`; `--private-state-dir` selects another
operator directory outside the model workspace. Initial reset and each Mink
arm/gripper operation's start/end save a full MuJoCo binary model (deduplicated),
INTEGRATION state, PID/buffer state, reset RNG state, source hashes and raw private
outcomes. Paths and simulator geometry are not sent to the atomic model.
`sim.private_state.restore` is a local diagnostic helper for compatible worker
environments, not a public tool. Snapshots precede the Host's reset settling;
the first operation's start snapshot includes the settled state.

LIBERO creation/reset now honor the supplied seed. Explicit equal seeds reproduce
the scene, differing seeds change it, and reset without a seed continues that
environment's isolated RNG stream. This is seeded procedural reset, not selection
of LIBERO's published benchmark initial-state files.

Mink gripper control checks the current state and every actual physics control
tick with the same collision/contact policy. Opening may make a checked monotonic
escape from existing penetration. The checker does not forecast contact dynamics;
`prediction_checked=false` makes this limit explicit. Camera observables are
suspended during the loop, then refreshed once at the end. An interrupted horizon
returns its actual step count and is not reported as a completed close/open.
Zero-step rejection preserves measured contact points; unknown/positive execution
retires them. The existing single-geom fixture authorization scope is unchanged.

Collision feedback now preserves `collision_class`, `check_mode`, `robot_part`,
`contact_binding_active`, `checked_during`, and `prediction_checked` when available.
An active contact binding does not authorize the colliding pair. Robot pose feedback
also separates `ik_seed_validated` from local execution success and includes
position/orientation tolerance verdicts and measured joint-limit margin.
These additive verdict fields require three-person interface review before a main
merge. See `docs/diagnostics/codex-state-gripper-pose-fix-2026-09-10.md` for validation.
# 对称夹爪姿态选择实验（2026-09-10）

`move_to` 新增可选 `orientation_mode: strict | parallel_jaw_symmetric`，默认 strict。
只对给出 approach/jaw 至少一个方向的请求选择候选；纯平移继续保持当前姿态。

```json
{
  "xyz_m": [0.1, 0.0, 0.95],
  "approach_world": [0.0, -0.8, -0.6],
  "jaw_world": [0.0, -0.6, 0.8],
  "orientation_mode": "parallel_jaw_symmetric"
}
```

坐标仅为参数示例，不能直接作为任务中的可执行目标。
该模式明确允许交换左右夹指；要求实验 Panda 夹爪已知空手、测量 openness >= 0.95，
没有先前接触绑定或闭合/接触后的未知负载状态。定向物体操作、把手操作和所需腕部视角使用 strict。
完成打开且收到 checked gripper_horizon_completed、测量充分打开，才清除闭合/接触后的锁定。

Host 分别为 R 与 R @ diag(-1,-1,1) 生成目标和新鲜 IK receipt；采用合法候选中成本最低者，
以其精确 receipt/bundle 执行一次，不在选择后重新求解另一 seed。
成本使用现有 IK 的关节 L2 位移 / sqrt(7)、最小关节余量惩罚，以及末端转角；
这版是可获取指标的终点评分，不是设计文档中的完整路径评分。
通常消耗 5 个 Host 步骤（两个 proposal、两个 IK、一次 move），而非原先 3 个。
预算、传输或状态过期失败不会绕过原 gate；完整负面 IK 结果允许继续检查另一个候选。

反馈包含 requested_target、实际 target 和 orientation_selection；后者明确
`selection_status=endpoint_only`、`path_check=not_run`。实际执行仍开启 Mink 碰撞保护。
`preview=true` 保持只渲染、不执行 IK 的语义，反馈两个四元数几何候选，不宣称已选择可执行目标。

当前尚未实现在线隔离 rollout、多 IK 解收集、跨途径点滞回或自动撤退重试。
完整设计见 [姿态选择设计](codex-gripper-orientation-selection-design.md)。
参数为独立实验入口的兼容性扩展，合入 main 前需要三人评审。
