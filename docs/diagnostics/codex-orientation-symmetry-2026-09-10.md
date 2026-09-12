# 对称夹爪姿态选择：实现与实验

日期：2026-09-10。分支：`dev/huaizezheng/codex-plugin-smoke-2026-09-08`。
工作仅在 `OpenETA-codex-plugin` 独立副本；未修改 main，未提交/推送或更新外部协作文档。
本轮模型调用为 0；没有新增自主任务成功率样本。

## 结果

针对 Goal 3 响应 0009 的大角度动作，保持目标位置和接近方向、交换左右夹指后，
所需旋转由 **179.47°** 降到 **50.56°**。
相同保存状态上的 Mink 执行由 150 步后仍有 24.09° 误差，改善为 **22 步到位、1.12° 误差**。
真实 Host → MCP → worker 工具链另一次短程验证也通过，所选 IK receipt 与 worker 实际使用 seed 的 ID 一致。

## 同状态控制器对照

运行入口：`scripts/experiment_gripper_symmetry.py`。
源记录：`tmp/codex-goal-0-3-retest-20260910/goal-3/host`。
数据：`tmp/codex-orientation-symmetry-20260910/report.json`，私有完整快照在同目录 `private/`。

使用原响应 0009 的目标与响应 0007 末尾的机器人关节状态，在新 seed=0 场景中设初速度为零。
创建一次完整快照；三组运行均恢复该快照，检查起点 qpos/qvel 完全相等。
原历史世界状态和速度没有保存，因此这是同状态配对实验，**不是原任务的精确历史回放**。

共同条件：默认 `mink_joint_velocity`、150 步上限、位置每轴容差 2 mm、姿态容差 0.05 rad，
开启同一碰撞检查，无接触授权、无携物代理，保持打开夹爪。
全部组暂停相机 observables，避免渲染占据执行时间；没有修改控制器任务权重或放宽容差。

| 条件 | 执行步数 | 停止原因 | 位置 L2 误差 | 姿态误差 | 终点最小关节余量 |
| --- | ---: | --- | ---: | ---: | ---: |
| 原姿态，原记录 IK seed | 150 | iteration_limit | 5.768 mm | 24.088° | -0.0000524 rad |
| 原姿态，新鲜 IK seed | 150 | iteration_limit | 5.768 mm | 24.088° | -0.0000524 rad |
| 等价姿态，新鲜 IK seed | 22 | target_reached | 0.312 mm | 1.119° | 1.14405 rad |

三组均未触发碰撞停止，所有 seed 均通过执行端完整姿态 FK 校验。
原姿态两组结果一致；主要失败表现仍为第 7 关节处在上限边界。
表中余量是实际终点的最小余量，不能解释为整条路径的最小余量。

候选 IK 的终点评分：

| 候选 | 所需末端转角 | IK 最小关节余量 | 关节 RMS 位移 | 分数（越低越好） | IK 用时 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原姿态 | 179.469° | 0.13398 rad | 1.38718 rad | 1.58659 | 0.574 s |
| 等价姿态 | 50.559° | 1.10999 rad | 0.35033 rad | 0.40650 | 0.033 s |

IK 预检前后，实际环境 qpos/qvel 未改变。评分在实际执行两组之前即选择等价姿态，
不是先看到执行结果再事后挑选候选。三组执行耗时约 4.81、4.76、0.70 秒，仅代表本地无渲染诊断。

## 已接入的最小版本

- `tools/codex_orientation.py`：两个局部滚转等价候选、SO(3) 转角，以及现有 IK 指标评分。
- `tools/codex_motion.py`：将既有 hook 支持私有 preflight-only continuation，
  分别执行两个普通 proposal/IK，最后通过选中的精确 bundle/receipt 执行一次。
- `tools/codex_atomic.py`：新增可选 orientation_mode、夹爪负载/接触约束检查和实际目标反馈。
- `tests/test_codex_orientation.py`：几何、评分、精确引用、状态过期、预算和工具失败回归。

默认 strict。`parallel_jaw_symmetric` 是显式允许交换左右夹指；只有给出姿态方向时才产生两个候选，
仅平移仍保持当前姿态。测量未充分打开（openness < 0.95）、未知负载或已有接触约束时拒绝自动翻转。
完成带检查的打开动作并测得充分打开才解除闭合/接触后的锁定；不能只凭 aperture 认定闭合抓空。

有限位关节的位移沿用 IK 的实际 q 差，不做角度 wrap。
初版评分是 `travel_L2/sqrt(7) + 2*max(0,1-min_margin/0.10)^2 + 0.2*rotation/pi`；
用已有 receipt 的最小余量替代设计中的逐关节余量，并使用 RMS 弧度位移。
这是明确记录的第一版近似，不是已经完成设计文档的整条路径评分。
缺失/非法评分指标的候选不会被选中，也不会凭较短转角绕过 IK 授权。

两候选一般使用 5 个 Host 工具步骤，原 strict 路径仍为 3 步。
完整负面 IK 结果允许检查另一候选；传输故障、预算/验证失败和状态过期不触发运动。
补充了预算恰在预检后耗尽的零步反馈：明确的 `episode_not_active`/`host_validation`
发生在 runner.step() 前，报告 physics_executed=false；未知执行异常仍保留不确定性。

公开反馈包含 requested_target、实际 target、两个候选的标量评分与选中原因；
不暴露关节数组、场景几何和私有文件路径。
`preview=true` 仍然只做几何显示，不调用 IK。

## 真实 Host 工具链验证

本地入口与记录：

- `tmp/codex-orientation-integration-20260910/live_host.py`
- `tmp/codex-orientation-integration-20260910/live-host-report.json`
- 同目录 `host/host-commands.jsonl`、`host/atomic-commands.jsonl`、`private/`。

使用独立 loopback MCP 服务、新建 Goal 3 seed=0 环境，先完成 40 步空中打开。
随后请求向上移动 3 cm 并倾斜 20°，人为给出相反 jaw 符号，使原表示需要 180° 转动。
Host 自主比较两个新鲜 IK 结果后选择 **20°** 等价目标：

- 5 个 Host 步骤，实际 Mink **20 步 target_reached**。
- 位置 L2 误差 **0.515 mm**，姿态误差 **0.174°**，无检测到的碰撞。
- 实际终点最小关节余量 **0.41803 rad**。
- 执行端 `ik_execution_seed_receipt_id` 与被选中候选的 receipt ID 完全相同。
- 打开动作后的负载锁解除，所选 target 和 requested_target 同时出现在原子反馈中。

Host 已关闭，独立服务已退出。Host interrupt 会先关闭环境；随后重复 close 返回
`ok=true, skipped=true`（环境 handle 已清空），不是跳过未关闭的环境。
worker 使用现有 parent-death SIGTERM 机制随自有服务结束。

## 验证和局限

最终回归 **74 passed in 7.98s**：

```text
tests/test_codex_orientation.py
tests/test_codex_motion_hook.py
tests/test_codex_atomic.py
tests/test_codex_failure_feedback.py
tests/test_gripper_final_observation.py
tests/test_gripper_latch.py
```

最终日志在 `tmp/codex-orientation-integration-20260910/tests.log`。
上述用例包括真实 Host planner/gate/receipt 的模拟适配器测试；真实物理证据来自前述两类实验，
不把模拟测试通过数量混称为任务成功样本。
运行前旧文件保存在同目录 `before/`，本轮文件指纹和差异保存在该目录的 manifest/patch 中。

**在线版本仍是 endpoint_only，反馈 path_check=not_run。**
本轮独立配对执行不是已经接入每次 move_to 的隔离路径预演。
多 IK 解候选、跨途径点滞回、在线隔离 rollout 和自动恢复均未实现。
单个大角度动作与短程工具链改善，不能推导 Goal 3 整任务或 goal 0–4 成功率。

新增参数和反馈为兼容性实验扩展；合入 main 前需按协作约定进行三人接口评审。
