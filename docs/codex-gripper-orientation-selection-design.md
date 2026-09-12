# 对称平行夹爪的姿态选择算法（设计草案）

日期：2026-09-10。独立分支：`dev/huaizezheng/codex-plugin-smoke-2026-09-08`。
本次交付是算法设计，没有接入运行时、改变工具行为或运行新的机器人任务。
新增接口语义在合入 main 前需三人评审；本次只保存本地设计，不更新外部协作文档。

## 1. 目标与已知证据

在不改变目标位置、接近方向、平行夹爪闭合轴线的前提下，选取更容易执行的等价末端姿态。
优化对象是“离散夹爪姿态 + 连续关节解 + 当前控制器的可执行路径”，而不只是欧拉角差。

Goal 3 响应 0009 的记录显示：原姿态相对起点旋转 179.46947°；
把夹指方向取反后，保持接近方向不变，旋转量降为 50.55914°。
接近方向本身相差 50.55595°。这只证明存在更短的等价姿态，不证明新姿态 IK 或路径可行。
此前 15/15 记录 seed 通过 FK；同快照 150/300 步均卡在关节边界的结果见
[诊断记录](diagnostics/codex-state-gripper-pose-fix-2026-09-10.md)。

## 2. 何时允许对称等价

提出可选参数 `orientation_mode: strict | parallel_jaw_symmetric`。
既有调用默认 strict，保持原先定向向量与省略姿态时保持当前姿态的契约。
实验配置/技能可让普通空手预抓取显式选用 symmetric；不要求模型自己计算翻转后的向量。

| 情形 | 策略 |
| --- | --- |
| 已知对称平行夹爪、空手、任务允许交换左右夹指 | 可选择两个候选 |
| 夹持物体、接触夹紧柜子/把手、闭合后负载不确定 | strict；不能凭空判断等价 |
| 要求物体定向、特定手指接触、腕部相机视角 | strict |
| 两个方向均省略，只请求平移 | 保持当前姿态，不主动翻腕 |
| 只有一个方向 | 第一版按现有 orientation() 补齐为完整姿态，再选择离散等价候选；不把缺省解释为任意自由滚转 |

夹爪对称性来自工具型号配置，不通过视觉猜测。闭合后不能仅凭 aperture 判定空载；
负载/接触不确定则拒绝 symmetric 请求并返回原因，不静默接受可能改变物体姿态的目标。
明确完成打开、解除操作约束并处于空手状态后才重新允许选择。
工具的对称不代表整条机械臂、腕部相机或所携物体对称；所有候选仍检查完整机器人几何。

## 3. 几何候选

沿用 `tools/codex_atomic_geometry.py`：局部 +Z 为 approach，+X 为 jaw。
输入经过有限值、非零和非平行检查，并使用现有正交化规则构成目标旋转 R。
记位置为 p，当前姿态为 R_now：

```text
S = diag(-1, -1, 1)             # 绕夹爪自身 Z 轴转 π
T0 = (p, R)
T1 = (p, R @ S)                # 右乘；不是绕世界 Z 轴旋转
theta(Tk) = acos(clip((trace(R_now.T @ Rk) - 1) / 2, -1, 1))
```

仅在 symmetric 且允许的情形产生 T1。两候选的 approach 完全相同，jaw 互为相反数。
不加入 ±90°、反转 approach 或放宽姿态容差，它们一般会改变抓取语义。
四元数 q 和 -q 表示同一姿态，不是这两个物理姿态；转换/去重使用旋转矩阵或 SO(3) 距离。

## 4. 对每个候选搜索关节解

从同一个状态版本 s0 读取当前关节 q_now、速度、限位、工具/负载状态、接触授权及场景版本。
在私有 IK 层中对每个 Tk 搜索最多 K 个不同关节解，初版建议 K=4（待性能验证）：

1. 优先从 q_now 热启动，保留当前冗余构型附近的解。
2. 再尝试兼容的上次选中关节解，以及模型定义的中性构型/有界确定性重启。
3. 缓存只能作初值，每个解必须重新 FK 校验完整 pose 与执行容差、限位及适用的几何约束。
4. 没有通过检查的解不进入排名；超时/搜索耗尽称为 no_candidate_found 或 inconclusive，
   不能宣称整个目标数学上不可达。

当前公开 IK 预检只返回选中的 receipt/seed，不假设已有多解接口。
最小集成可先各取一个现有 IK 解（K=1），多解版本需新增私有候选收集逻辑。
不能只改 receipt 的 joints：每个被选中的解都必须有绑定自身目标、关节解和状态版本的证据。

有限位转动关节使用实际差 q_i - q_now_i，禁止统一 wrap 到 [-π,π] 后当成可走的短路径。
仅机器人模型声明连续旋转的关节才允许按其拓扑计算距离。

## 5. 硬约束优先、评分辅助

任何碰撞/限位/姿态硬约束失败的候选都不能靠低转角分数获选。
安全间距、容差和允许接触集合沿用当前 worker 策略，新增评分不扩大接触授权。
endpoint_collision_check 若被现有契约延后，明确保留 deferred，不能写成已检查无碰撞。

对终点关节解先计算粗排分数；对实际 rollout 路径再用路径量重新排名：

```text
d_i(q) = min(q_i - lower_i, upper_i - q_i)       # rad
P_limit(q) = mean_i(max(0, 1 - d_i(q)/margin_i)^2)
P_path = max_t P_limit(q(t))
L_joint = mean_i(sum_t abs(q_i(t+1)-q_i(t)) / (upper_i-lower_i))
J = L_joint + 2 * P_path + 0.2 * theta/pi
```

没有路径时，粗排使用终点 P_limit 与起点到终点的归一化关节位移，并标记 endpoint_only。
建议第一轮 margin_i=0.10 rad；它是软偏好，不是新的硬限位。
系数 1/2/0.2 和软裕度是拟议初始值，没有实验校准；记录分项便于后续消融。
对非转动关节必须使用模型对应单位和参考尺度，不能套用 rad 参数。

预设机器人硬限位仍是硬约束；初始状态已经在边界/微小越界时沿用现有、经过检查的退出规则，
不因“初始裕度不足”使退出动作永远无法执行。
路径最小关节裕度、碰撞最小距离独立记录；不会把有限路径采样宣称为连续无碰撞证明。

## 6. 检查当前控制器是否能走过去

这是解决本次“合法 IK seed + 局部控制器碰限位”所需的第二层。
第一版使用隔离仿真副本，从同一个完整 s0 执行当前 Mink 控制器，目标 Tk 与 seed qk 均固定：

- 副本恢复物理状态、PID 历史、RNG、模型和接触授权，关闭相机渲染与评估副作用。
- 使用与真实动作相同的预算、容差、控制参数、碰撞检查和停止条件。
- 检查的是实际 Mink 控制器生成的运动，不能用简单关节直线插值成功替代。
- 记录完整 pose 是否到位、停止原因、实际步数、最小限位裕度、关节总运动量与碰撞阶段。
- IK 成功但 rollout 卡限位/碰撞/超预算：标记 local_execution_failed，不能进入通过候选组。
- rollout 通过仍不保证真实执行成功；执行阶段继续逐步检查，场景变化会使预检失效。

以单独私有 worker/copy 实现；不要在实际环境中“执行后回滚”冒充无运动预检。
当前 snapshot restore 仅验证兼容环境中的回放，隔离副本的构建和完整一致性仍需实现验证。
不具备隔离副本时只能提供 endpoint_only 优化，必须反馈 path_check=not_run，不能声称已解决路径收敛。

预算建议：两姿态各粗排最佳解首先各试一次，最多 4 次完整 rollout。
若尚有候选且前两次失败，可在余下预算内尝试其他关节解；无通过项则零步返回诊断。
各次 rollout 上限等于本次请求的执行步数上限，而非另一个更宽松预算。
总计算时间设配置上限，必须为最终执行预留时间；耗尽时返回未完成，不能关闭检查后强行执行。
这些仿真搜索消耗本地算力；需计入内部计算/Host 审计预算，不能隐形绕过工具预算。

## 7. 稳定选择、绑定与执行

在同等级已通过路径检查的候选中选择 J 最小者。若上次实际采用的对称分支仍可行，
只有新候选改善超过滞回阈值才切换；建议阈值为 max(0.02, 0.1*abs(J_previous_branch))。
这里的 previous_branch 分数必须基于当前状态重新计算，不能复用旧分数。
用旋转距离匹配上一目标的物理朝向，不按候选 0/1 编号跟踪；模型把 jaw 向量取反时编号会互换。
上次分支不可行则不受滞回约束。一次运动开始后锁定所选姿态和 seed，不能每个控制 tick 切换等价分支。

```text
select_and_move(request, state):
    validate_orientation_mode_and_manipulation_state()
    targets = enumerate_equivalent_targets(request, state)
    solutions = bounded_ik_search(targets, state)
    candidates = hard_filter_and_endpoint_rank(solutions)
    checked = bounded_isolated_mink_rollouts(candidates, state)
    chosen = choose_passed_candidate_with_hysteresis(checked)
    if chosen is None:
        return no_motion_with_candidate_diagnostics()
    if current_state_version != state.version:
        return stale_preflight_no_motion()
    bind_exact_target_seed_tolerances_and_authorization(chosen)
    return guarded_mink_execute(chosen)
```

选择必须发生在最终执行目标/seed receipt 绑定之前。候选预检可生成各自真实 receipt，
选中后沿用其精确 ID；如需再验证同一 seed，须检查没有返回另一个构型，改变就要重新检查路径。
已有显式 bundle/receipt-ID 请求坚持原目标，不在解析后偷偷改变朝向。
状态、接触绑定或负载变化会使选择证据过期；通过 Host 锁和 epoch/state token 保证顺序。

如果执行中停止，反馈实际状态；本次调用不自动再翻转 180°、撤退或重试剩余候选。
因为已经发生的接触/运动可能改变世界，重试必须重新观测和预检。

## 8. 接入位置与反馈

- `tools/codex_atomic_geometry.py`：纯几何候选生成，保留现有 orientation() 的严格行为。
- `tools/codex_atomic.py`：解析显式 orientation_mode、状态适用性及请求/实际目标反馈。
- `tools/codex_motion.py`：将当前 preflight+execute 拆成可复用阶段；对候选分别预检，最终只执行一次。
- 私有 IK/worker 层：多解指标、隔离 controller rollout、状态版本校验；不向模型暴露原始场景几何。
- `sim/controllers/mink_goal.py`：第一轮保持现有控制目标权重、容差、限位与碰撞保护，避免混入控制器消融。

现有 `preview=true` 是纯渲染、无 IK。保持该语义：可画两个请求候选，但只能标记 geometric_candidates，
不能提前显示“已通过检查的选中目标”；执行调用才产生经过预检的 selected_target。
若以后新增计算型预览，应另行明确命名和预算契约。

拟议 agent 可见的最小反馈如下（数字仅为几何历史核验，IK/路径结果不捏造）：

```json
{
  "orientation_selection": {
    "mode": "parallel_jaw_symmetric",
    "requested_rotation_deg": 179.46947,
    "equivalent_rotation_deg": 50.55914,
    "selection_status": "geometry_only",
    "path_check": "not_run"
  }
}
```

执行时增加 selected_target、symmetry_applied、selection_reason、
endpoint_check/path_check 以及现有实际执行反馈。公开结果保留所需的标量诊断；
多候选关节数组、几何对、完整轨迹留在私有记录。
只有两个候选都检查过时才说“比较后选择”；预算只允许一个时明确说明覆盖范围。

## 9. 验证与分阶段验收

第一阶段：几何候选 + 两个现有 IK 预检 + 限位/位移评分 + 精确 receipt 绑定。
它减少不必要翻腕；路径尚未预检时返回 endpoint_only，不能以该阶段结果承诺控制器收敛。
两个目标通常最多两组 propose/IK + 一次执行，即 5 个 Host 工具步骤（现有单目标为 3）；
多解/rollout 需另计私有计算预算，模型仍只发一个 move_to。

第二阶段：隔离完整 Mink rollout + 同状态可执行性选择。建议先完成这一步的固定快照对照，
再判断是否需要改控制器 seed 权重或新增撤退/路径工具。

必要验证：

1. 两候选正交且 det=+1，approach 相同、jaw 相反；四元数符号不改变候选集合。
2. near-180°、方向退化、无姿态参数、只给单轴、strict 和夹持/接触状态覆盖。
3. 末端最短角候选碰限位/碰撞时选另一可行项；两者失败零物理步，不谎报不可达。
4. 有限位关节不通过角度 wrap 越界；同姿态多 IK seed 的 receipt 与执行保持精确一致。
5. jaw 符号连续抖动不引起翻转振荡；执行中不切换，状态过期不使用旧选择。
6. 隔离预检不修改实际环境、episode 步数、RNG、控制器历史或任务完成状态。
7. 在同一保存状态上比较 strict、仅几何最短角、IK评分、IK+rollout 四组；
   记录姿态误差、关节裕度、碰撞、执行步数、预检时间和 Host 开销。
8. Goal 3 的旧响应 0009 可作目标/关节起点回归，但旧记录无完整历史世界状态，不能宣称精确历史回放。
   新任务使用保存快照进行配对，再测 goal 0–4；本设计本身不增加任务成功率样本。

## 参考与范围

[Mink 官方说明](https://github.com/kevinzakka/mink)说明其采用微分 IK，并提供关节位置/速度限制与几何对碰撞约束。
以上离散对称候选、评分、滞回及隔离 rollout 选择是为本仓库提出的设计，并非声称 Mink 已自带该算法。
本次查阅了本地姿态、运动 hook、IK receipt 绑定和诊断记录；没有更改共享 RFC 或运行时。
