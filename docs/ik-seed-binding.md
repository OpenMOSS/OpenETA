# IK receipt 与执行 seed 的绑定

2026-09-07，个人重构分支候选；修复既有 exact-reference 行为，不新增 Agent 参数或 simulator schema。授权语义修正需三人 review，未更新共享 RFC。

## 已复现的问题

主 Agent 的 `move_to(ik_receipt_id)` 原本按指定 ID 还原目标、姿态策略和容差；私有 seed resolver 却按最新的等价目标查询。例如同一 epoch、同一末端目标先后得到 `first`（关节解 A）与 `second`（关节解 B），选择 `first` 会发送 `second` 的 seed。两份 seed 的末端都可能符合目标，worker 的 FK 校验不能发现这是选错了引用。

另一个复现是：选中 hard-infeasible receipt 后，如果同目标又有 feasible receipt，motion gate 会借用后一份证据放行前者。原有逐项单测未覆盖多份同目标证据的组合。

## 当前行为

- 指定 ID 必须存在、属于当前 object/robot epoch、匹配请求目标及姿态策略，并且自身满足运动授权条件。
- Seed 从指定 receipt 提取，不能因其缺少/包含非法 joints 而退回另一份同目标 seed。未提供 ID 的内部旧调用保留原来的等价目标查询。
- 选中旧 feasible receipt 不会绕过最新等价目标的失败证据；原有较新 hard-infeasible/inconclusive veto 保留。
- 轨迹 gate 按顺序将每个 `ik_receipt_id` 与对应 waypoint 一起校验，不再丢掉 ID 后仅按几何匹配。显式 ID 列表长度必须与 waypoint 数量一致。
- 原有 collision-deferred 授权仍要求显式碰撞检查及对应控制器能力；exact ID 不产生新的安全授权。

例如 Agent 只提交：

```json
{"ik_receipt_id": "first", "num_steps": 40}
```

Host 还原 `first` 的目标和容差，私有 MCP 请求中的 `ik_execution_seed.receipt_id` 也是 `first`。Agent 不填写 joint seed；原始 request/audit 参数不注入私有 seed。仍允许既有契约中的显式容差覆盖，worker 继续按实际执行容差校验 seed FK。

## 验证与范围

新增 11 项测试实例：两份同目标成功 seed 的选择、选中失败证据不能借用新成功证据、较新失败 veto、无效 ID、runtime/planner/pipeline/registry/proxy 绑定链路及轨迹逐项授权。两个缺陷在修复前分别复现失败。

链路测试使用 StaticPlannerBackend 和本地模拟 transport，没有新 simulator、HTTP 或感知调用。它证明主 Host 链路的参数行为，不证明物理收敛。此前 [live Mink 短程证据](diagnostics/mink-seed-canary-2026-09-07.md) 是另一次直接 Host 运行，不是本次新实现的 live 回归。

HV-01 继续开放：需要在实际主 Agent + simulator 上验证选中 receipt 的 seed 回执、复杂转腕/支路，以及真实任务诊断。不能由这些测试推断 OSC 已消费 seed，也没有切换默认控制器。

后续 R0 第二十三批已取得 [实际 runtime → Mink 短程回执](diagnostics/mink-seed-canary-2026-09-07.md)：Host 摄入实际 preview 后，确定性 planner 的 receipt-ID move 经生产 runtime/gate/proxy 执行，所选 ID 与执行回执匹配。该局部链路不再只有模拟 transport 证据；但 preview/proposal 选择、真实视觉决策、复杂转腕和完整任务仍未验证。
