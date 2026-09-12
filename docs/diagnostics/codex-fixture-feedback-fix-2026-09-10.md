# 柜子部件授权与 Codex 失败反馈修复

日期：2026-09-10。分支：`dev/huaizezheng/codex-plugin-smoke-2026-09-08`。
用户授权修复两项：柜子几何缺少接触授权、Host 对 agent 的失败反馈不明确。mihomo 日志由用户处理，本次没有修改代理、登录配置或 main 工作目录，也没有发起模型调用。

## 已实现的行为

### 私有几何与局部接触

- `UnifiedEnv._extract_libero_objects()` 保留现有自由物体提取，并增加 fixtures 的实时碰撞几何目录。
- 新模块 `sim/libero_contact_geometry.py` 逐一使用 MuJoCo geom 的实时世界变换计算 AABB。不会把固定/关节 fixture 当成 7 维自由关节读取 qpos。
- 关节 fixture 的 slide/hinge 子部件碰撞几何可作为模型打点的接触候选。固定柜板仅为障碍物，visual-only 几何不参与授权。
- 匹配仍使用现有 2 cm 表面距离及歧义检查；不放宽阈值。授权包含父 fixture、具体碰撞 geom 及 body 的私有引用。
- Mink worker 按实时模型重新检查 geom 是否属于指定 fixture、对应指定 body、位于关节部件且是碰撞几何。只豁免夹爪与该一个 geom 的碰撞对；机械臂与该 geom、夹爪与其他柜板/场景物体仍受保护。
- 闭爪成功后在仿真端保留该局部接触绑定，后续无新打点的拉动可继续使用。重复闭爪保留绑定；松爪、重置、闭爪失败清除绑定。切换已绑定的运动接触 geom 前需松爪。
- fixture 不建立自由物体携带代理，避免“柜子没有 free joint”及把整个柜子当作被搬运物体的问题。此路径针对默认 Mink 的原子工具实验验证。

### 失败回执

新增 `tools/codex_feedback.py`，通过白名单投影失败回执；`CodexHost` 不再把已知运动失败一律压成 host_command_rejected。

`run_motion_hook()` 在处理 failed 状态前提取回执，atomic 返回：

- `reason_code`：例如 contact_authorization_unresolved、collision_detected、control_step_failed；
- `motion_summary`：实际执行步数、是否到达、位置/姿态误差、允许公开的碰撞/控制器失败类别；
- `physics_executed`：true / false / null，分别表示回执证明执行过控制步、证明零步、无法确定；
- `recovery`：相应的恢复类别与简短说明；
- `robot`：沿用实际观测的 EEF 与夹爪状态，不能把目标位置冒充实际位置。

`motion_dispatched` 继续表示请求是否送出，不等同于物理执行。明确零步的失败不再触发旧点失效；实际执行或状态未知时仍保守处理。

不透传私有对象名字、碰撞几何坐标、响应文件路径或原始错误文本。未知原因退化为 tool_execution_failed，不猜测成功或零步。

示例（Goal 4 历史失败重投影）：

```json
{
  "reason_code": "collision_detected",
  "physics_executed": false,
  "motion_summary": {
    "steps_executed": 0,
    "stop_reason": "collision_detected",
    "reached_target": false,
    "collision": {
      "detected": true,
      "endpoint_checked": true,
      "trajectory_checked": false,
      "world_checked": true,
      "collision_class": "attached_object_world"
    }
  },
  "recovery": {
    "action": "replan_collision",
    "message": "A collision check stopped the motion. Inspect the returned views and actual pose; choose a different checked waypoint that increases clearance. Keep collision checks enabled."
  }
}
```

## 验证

1. 主测试环境：121 passed，7 skipped（该环境缺 MuJoCo）；含真实 MCP stdio、本次新回归、Host/atomic/motion hook、几何/碰撞及夹爪相关测试。
2. 将上述 7 项 Mink 关节限制测试在真实 LIBERO Python + Mink overlay 环境补跑：7 passed。合计 128 项不同测试通过。
3. 增加重置生命周期断言后单独复验 fixture 测试：5 passed（计入以上 128 项，不重复计数）。
4. 真实 LIBERO Goal 0、3，seed=0，加载官方场景、提取几何、重放原实验 7 次接触请求的点（包含重复点），全部匹配成功；Mink 每个授权的 target geom count 均为 1，夹爪该局部接触对豁免，arm-target 保护仍存在。两环境已关闭。
5. 上轮全部 11 条 failed Host command 重投影：6 条接触授权失败、3 条携带物碰撞失败、2 条控制器失败；均保留执行步数，未泄漏私有物体名或路径。

真实场景匹配：

- Goal 0 → middle drawer，实际匹配到 `wooden_cabinet_1_g29`；历史点到表面距离约 0–0.53 mm。
- Goal 3 → top drawer，实际匹配到 `wooden_cabinet_1_g18`；历史点到表面距离约 0.02–3.40 mm。

这些名字仅用于本地诊断，不传给模型。

## 产物与范围

产物在 `tmp/codex-fixture-feedback-fix/`：

- `before/`、`before-hashes.json`：此次修改前的文件快照。
- `tests.log`、`tests.xml`、`mink-tests.xml`：测试结果。
- `check_real_geometry.py`、`real-geometry-report.json`：无模型真实场景验证与局部碰撞策略检查。
- `replayed-failures.json`：历史失败的 agent 可见投影。
- `fix.patch`：仅本轮修复相对修改前快照的补丁，避免混入继承的未提交工作。

未重跑模型任务，未执行物理开抽屉成功实验；验证结果不计入 agent 任务成功率。携带物 AABB 建模与网络问题不在本次修复范围。

接口说明：public 工具入参保持不变，实验输出增加 physics_executed/recovery 等字段；Host-private 几何/授权增加 fixture 子部件引用。合入共享 harness 前需三人接口复核。此次独立分支实现和验证按用户授权完成；未提交、推送或更新外部协作文档。

原先继承的 53 个 dirty/untracked 文件中，仅 `sim/controllers/mink_goal.py` 因此次授权的 fixture 碰撞策略适配发生变化；修改基于保留原内容的局部补丁，其余继承文件哈希不变。
