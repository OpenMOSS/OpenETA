# 姿态传递修复、接触反馈与 20 任务 campaign

日期：2026-09-10；独立分支 `dev/huaizezheng/codex-plugin-smoke-2026-09-08`。
后续补充：此前的数值传递检查只验证到 body 姿态；另发现 atomic 将 body 与实际 grip_site
坐标系混用，已单独修复并直接验证真实手指方向，见 [夹持坐标系修复](codex-grip-frame-fix-2026-09-10.md)。
依据 [Goal 3 根因复核](codex-goal3-root-cause-2026-09-10.md)。
用户授权修复明确 bug、持续重试，目标是通过 Codex/OpenETA 插件完成
LIBERO Object 0–9 与 Goal 0–9 共 20 个任务；实质性方案问题再讨论。

## 修复

`agent/tools/sim_mcp.py` 将 quaternion 路径统一到带奇异点处理的矩阵→Euler 转换。
pitch 用 `atan2(-R20, hypot(R00,R10))`，避免 `asin` 在接近极点时提前舍入到 ±90°；
极点固定 yaw 并保留耦合 roll。接口仍使用原来的 Euler 参数，没有改控制目标语义。

新增 `sim/controllers/gripper_contact.py`：从最终 qpos 在独立 MjData 中重建接触，不推进物理步、
不更改运行中的状态。move 和 gripper 两条 worker 路径均返回外部几何与两指/指垫接触布尔值。
不提供目标身份、对象真值位置、几何名字、距离或接触力；没有添加任何碰撞豁免。
机器人自身接触不计入外部接触。遥测异常返回 unavailable，不能抹去已执行动作的结果。

公开 `motion_summary.gripper_contact` 示例：

```json
{
  "available": true,
  "left_finger_contact": true,
  "right_finger_contact": false,
  "left_fingerpad_contact": false,
  "right_fingerpad_contact": false,
  "contact_pattern": "finger_body_only",
  "retention_proven": false
}
```

contact_pattern 可为 bilateral_pads / single_pad / finger_body_only / no_contact。
这是接触诊断，不是力闭合或目标抓取证明；双侧可能接触不同几何。
`_agent_visible_simulator_response` 和 atomic feedback 都对白名单字段投影，禁止携带私有扩展。

失败接近的 recovery 提醒 agent 不要把目标当作实际到达位置，不要立即闭合；
先检查误差和新图像、重新打点、修正接近或撤离换姿态。
atomic skill 增加小幅试拉/试抬、检查目标是否跟随、失败后松开重新定位的策略。
没有自动替模型操作夹爪，也没有把指垫双侧接触作为绝对运动门槛。

这些是向后兼容的实验输出新增项，合入共享 main 前仍需三人接口复核。
本次没有提交、推送、修改 main 或更新外部文档。

## 验证证据

- 真实 adapter 两条参数路径 → MCP server 使用的 quaternion 编码，覆盖正负奇异点、邻近角、
  非零耦合 roll/yaw、q 和 -q，以及历史 `[.5,.5,-.5,.5]`：57 项通过。
- 姿态、motion hook、对称算法组合：92 项通过。
- 扩展 atomic/失败反馈/接触验证组合：116 passed、1 skipped；skip 是主 Python 缺 MuJoCo。
- LIBERO Python 中接触几何、机器人自接触排除、受检查夹爪：7 项通过，补足 MuJoCo 覆盖。
- 后续 campaign 判定与反馈边界定向检查：74 passed、1 skipped（相同 MuJoCo 项，不重复计为额外覆盖）。
- 插件 manifest 与 atomic skill 验证通过；插件版本更新为 `0.1.0+codex.20260910124202`。
  每次试验创建新的私有 Codex home 和插件安装目录，普通用户 Codex 配置不变。

历史完整任务 22 个选中姿态经修复后的转换与独立 SciPy 重建比较，
最大旋转偏差约 `3.08e-14°`。新的接触模块重建历史三次闭合快照，均返回 finger_body_only，
分别为双指体、左指体、右指体接触，与此前独立接触审计一致。

真实 Host → MCP → Mink 无模型 canary：seed=0 新环境，先 open/close/open，随后发送历史 #7 的
选中水平目标，位置 `[0.05840754,-0.01149189,1.09776329]`，approach=-Y，jaw=-Z。

- worker 目标旋转与原目标相同；执行 seed receipt 与 IK preview 相同。
- 66 步 target_reached；位置误差 0.619 mm、姿态误差 2.153°。
- 最小关节余量 0.48652 rad；没有碰撞停止。
- 三次夹爪和运动均返回 available=true 的接触反馈。
- 运行结束 Host cleanup 成功，自有 server 已退出。

这是修复后真实链路的局部验证，不是历史完整动态状态的严格 A/B 回放，也不是自主任务成功。
此前对称候选失败的控制目标确实被改变过；此结果支持先修正传递再评价控制器。
证据目录：`tmp/codex-pose-contact-fix-20260910/`，含 `live_host.py`、`live-host-report.json`、
`saved-state-validation.json`、私有前后状态和 Host 收据。

## 持续实验规则

`scripts/codex_campaign.py` 建立可恢复的 20 任务清单。每次尝试隔离服务端口、私有状态、
登录文件副本、插件安装、网络采样、Codex 原始事件和 Host 动作日志。
保留所有失败尝试；只有官方成功为 true 且集成、清理、源码稳定检查通过才标记任务完成。
不因 shell exit=0 或工具返回成功而计任务成功。

默认配置：gpt-6-astra / medium / ChatGPT 登录 / atomic 六工具 / Mink；
seed=0 程序化 reset，沿用当前诊断基准，不称为官方 init-state benchmark 成功率。
单次 2400 秒、160 native requests/Host turns/tool calls；预算与上一轮一致。
用户授权的长期 goal 没有设总 token 限额。每个任务允许基于诊断修复后重试；
不隐去失败、不换 seed 挑选容易初始状态。

清单及状态：`tmp/codex-libero20-20260910/batch-status.json`。
第一项是修复后的完整 Goal 3；之后覆盖其余 19 项。首轮失败后可定向重跑尚未完成的任务。
当全部 20 项取得官方成功证据后，再将长期 goal 标为完成。
