# Goal 3 碰撞诊断与 main 迁移评估

日期：2026-09-10。分支：`dev/huaizezheng/codex-plugin-smoke-2026-09-08`。
本轮只新增本地诊断脚本和本文，没有修改运行逻辑、main、代理配置或调用模型。

## 结论与证据边界

**不需要重新迁移 main 的整套碰撞检测。插件已经包含相同的 Mink 检测、约束恢复和携带物检测核心。**
当前优先项是接触授权粒度、夹爪动作的检查覆盖、可操作的反馈，以及可复现的私有仿真记录。

原实验确定发生了四次 `collision_detected` 安全停止，均在抽屉操作阶段；
不是四次都已发生物理碰撞。其中一次零步停止，明确没有执行该次运动。
其余停止可能来自实际步后接触或下一步运动预测，原始公开回执未保留检查阶段，无法逐次定性。
**原场景及未脱敏碰撞回执没有完整保存，因此不能把下述辅助回放的具体碰撞对认定为原实验四次停止的确切碰撞对。**

## 原实验

来源：`tmp/codex-goal-0-3-retest-20260910/goal-3/`。
Agent session：`763df4688f0a4ec390946571aa04bf63`。
Sim session：`9b08ebc5-5de0-406d-875f-b0df0ea2a9ce`。
任务：open the top drawer and put the bowl inside；Astra medium、Mink、原子工具。
官方任务结果 false；1065.107 秒；33 次原生调用、53 次 Host 工具调用；无模型断流。

表中请求号来自 `host/atomic-commands.jsonl`，响应序号来自保存的 simulator response 目录。

| 原子请求 / 响应 | 动作 | 原始停止证据 | 能确定的原因范围 |
| --- | --- | --- | --- |
| 7 / 0005 | 从把手上方下降，目标在腕部打点下方 7 mm，带接触授权 | collision_detected，6 步；位置残差 22.44 mm | 接近/接触阶段的保护触发；尚未执行 close |
| 15 / 0016 | 首次 close 后沿世界 +Y 拉 90 mm | collision_detected，0 步 | 在动作执行前被拒绝；不能描述为此次拉动造成了碰撞 |
| 17 / 0018 | 刷新打点并沿 +Y 拉 60 mm | collision_detected，1 步 | 重新请求接触仍未绕过保护，不是缺少调用预算 |
| 31 / 0034 | 大姿态调整失败后再次接近把手 | collision_detected，1 步；位置残差 70.04 mm | 在不利关节构型下的接近运动触发保护；原回执不足以判定具体碰撞体 |

另有五次 150 步 iteration_limit、一次 68 步 local_convergence_stalled、一次
116 步 control_step_failed，以及两次未执行运动的 ik_search_no_solution。
这些不能全算成碰撞，也不是网络超时。

部分失败主要剩下姿态误差：请求 9、11、20 分别约 24.40°、23.58°、23.43°，
位置残差仅约 3–4 mm；请求 29 位置误差 1.14 mm、姿态误差 24.93°，
错误明确为备用速度未通过碰撞/关节约束检查，未执行该备用步。
请求 31 的 IK 候选距 joint 6 上限仅约 4.08e-7 rad，已有 execution-fragile 提示。
因此增加步数或补一套碰撞检测不能单独解决这条轨迹。

第一次 close 后测得开度 0.8423，第二次 close 后 0.03884；二者都不能证明抓住把手。
第一次 close 期间 EEF 位移约 1.39 mm、转角约 0.22°，没有证据支持“大幅漂移导致所有碰撞”的说法。
原记录最后一次抬升成功，但任务官方结果仍为失败。

## 接触粒度：已确认机制，尚未确认原实验逐次因果

当前 `sim/controllers/mink_goal.py::_libero_collision_policy` 对 fixture 只豁免一个 geom，
机械臂与所有世界几何体的保护保持启用，夹爪与其他几何体的保护也保持启用。

LIBERO 的 `assets/articulated_objects/wooden_cabinet.xml` 中，上层把手由三个碰撞 box 构成：

- `wooden_cabinet_1_g18`：把手横杆；本次打点在辅助场景中解析为这个 geom。
- `wooden_cabinet_1_g19`、`wooden_cabinet_1_g20`：横杆后方两侧连接部位。
- 三者都属于 `wooden_cabinet_1_cabinet_top`，但该 body 还包含抽屉壁和底板。

第二次辅助回放中，第一段 59 步运动与原记录末端位置完全相同；接触段则在 3 步后，
下一步预测的 `gripper0_finger1_collision` 对 `wooden_cabinet_1_g20` 距离为 -6.177 mm，
低于 -1 mm 硬停止阈值，因而在执行该步前停止。授权覆盖 g18，未覆盖 g20。

这验证了“同一个视觉把手的邻接碰撞部件仍会挡住夹爪”这一机制。
但它不是原实验 6 步停止的精确复现。也不能直接把整个 drawer body 或整个柜子都豁免：
那会同时允许手指穿入抽屉板和柜体。

若后续扩展授权，应限定于由新鲜打点绑定的、经过几何验证的局部可接触部件集合，
保留机械臂、夹爪掌部与非目标障碍的约束，并用同一初始状态对照验证。
“是否豁免连接柱”应结合接触用途和穿透深度决定，不能因为控制器停住就直接放宽阈值。

## 夹爪检查与反馈缺口

`gripper_close` 通过 `_step_gripper_with_final_observation` 连续执行 60 步，open 执行 40 步。
这里没有调用 Mink move_to 中的逐步预测/步后碰撞保护。
零关节速度指令不等价于物理场景不会改变，手指本身在运动，接触也可能推动抽屉。
下一次 move_to 才处理已进入的接触/约束状态，是需要补齐的覆盖缺口。
原实验缺少 close 期间的接触日志，尚不能证明它就是请求 15 停止的唯一原因。

应保持用户要求的“最终只渲染一次”，独立增加物理过程中的碰撞检查；
不必为每个物理步渲染图像。

反馈存在两层确定的信息损失：

1. Mink 普通机器人碰撞分支记录了 geom 对、距离、`check_mode`，但没有设置
   `collision_type`。`agent/tools/sim_mcp.py::_agent_visible_collision_receipt`
   因此输出 `unspecified_contact`，并去除具体几何体和检查阶段。
2. `tools/codex_feedback.py` 的 collision_class 白名单又不包含 unspecified_contact，
   原子工具最终连这个分类也没有保留，只剩 detected 和 checked 布尔值、通用恢复建议。

保密和可诊断可以同时满足：私有 operator trace 保存碰撞对、距离、授权范围和预测/实际阶段；
agent 获得安全的类别、机器人受影响部分、停止阶段、是否执行过物理步、实际末端位姿。
不向 agent 提供模拟器实例名字、隐藏对象位姿或自动规划好的答案。
接触未授权、实际碰撞、下一步预测碰撞、关节边界拒绝需要有不同的反馈。

## 为什么不能精确回放

`sim/env_registry.py::_LibEnvWrapper.reset(seed=...)` 当前只调用 `self._env.reset()`，
忽略传入 seed；direct LIBERO 创建路径也没有应用官方初始状态。
该包装类与本机 main 的版本相同。

静态验证在同一进程连续两次 reset(seed=0)，g18 中心从
`[0.04260516, -0.14342444, 1.08893]` 变为
`[0.02529844, -0.13223004, 1.08893]`，相差约 20.6 mm。
机器人初始姿态一致，不能据此推断场景几何一致；这解释了无接触第一段能相同、接触段出现分歧。

两次辅助回放均在首次分歧停止，没有继续将新场景轨迹冒充原始记录。
一次接触段达到 150 步未收敛，另一次 3 步后预测碰撞；这两次不是配对消融试验。

使用原记录的关节角，另做了四次 collision 停止及一次 controller failure 末端构型的静态检查。
未检测到这些构型的受保护 arm-vs-arm 穿透；这仅排除了已记录末端构型的这部分自碰撞，
不能排除尚未执行的预测构型，也不能排除 arm-vs-gripper 接触。
公共开度只是均值，无法还原左右指的独立关节状态；新场景 world 检查不作为历史事实使用。

后续需要同时保存模型/资源版本、初始模型与完整状态、控制器状态、随机数状态，
并在每次失败保留私有当前状态和候选动作。只记录 seed 和 EEF 位姿不足以重放接触动力学。
历史任务的官方成功/失败仍是实际观察结果，但标记 seed=0 不足以证明场景配对一致。

## 与本机 main 的逐项比较

以 AST 比较函数/类实现，并记录文件 SHA256；诊断结束再次核对，所读文件都未变化。

| 部分 | 对比结果 | 建议 |
| --- | --- | --- |
| Mink 实时几何、碰撞 QP、执行前/步后保护 | 相同；仅 `_libero_collision_policy` 有插件 fixture 授权扩展 | 保留，避免重复迁移 |
| `collision_recovery.py` | 文件逐字相同 | 保留现有受约束退出 |
| `mcp_server/collision.py` | 仅接触授权解析不同；cuRobo 和携带 AABB 检测相同 | 不覆盖现有 fixture 修复 |
| `server.py` 的 IK endpoint checker、OSC 碰撞循环、携带扫掠 | 对应函数/逻辑已在插件中；Mink 路径直接返回 worker 结果 | 无需复制另一份 |
| agent 碰撞脱敏函数 | 与 main 相同 | 在公共类别与私有日志间补齐反馈 |

main 另一条 cuRobo 路径以 Franka 机器人近似几何和物体 cuboid 检测，
用于 OSC 批次后检查，也可用于可选 IK endpoint 检查。
Mink 已在 worker 使用当前 MuJoCo 模型、实际指状态与世界几何，QP 期望间距 3 mm，
硬停止阈值 -1 mm，并结合 mj_geomDistance 与真实/预测 contact 距离。
cuRobo 世界穿透阈值是 5 mm、指关节锁定的机器人模型；它不是同等粒度的抽屉接触替代方案。
两者直接叠加还需协调授权和不同距离语义，可能增加误拒绝，不能默认视为改善。

IK 可达也不等于路线可执行。若要加强预检查，应优先使用同一份 live MuJoCo 几何与接触政策，
检查候选端点以及有界的局部运动预测，明确标注检查覆盖；endpoint pass 不应宣称整条路径可行。

## 建议的后续最小范围（本轮未实施）

1. 先补私有初始状态/失败 trace、碰撞类型和停止阶段反馈，修复 seed/官方初始状态的应用。
2. 给夹爪物理动作复用同一接触政策和检查器，保留最终一次渲染。
3. 固定同一初始状态后验证局部把手授权集合，并区分连接件接触与抽屉板/柜体碰撞。
4. 姿态收敛问题独立处理：保留关节裕度告警，先退出受限构型，再调整姿态；不要把它混成网络或步数问题。

不需要因此添加 SAM3、AnyGrasp、AnyPlace 或新的高层任务工具。
若上述后续修改新增 checker verdict 字段/枚举，按协作约定标记为需三人接口评审后再合入 main。

## 本轮产物与验证

`tmp/codex-goal3-collision-diagnosis/`：

- `main-comparison.json`：函数比较、SHA256 与结束时一致性核对。
- `replay.py`、`incoming-quaternions.json`：无模型、有碰撞保护的记录回放。
- `replay-initial-divergence.json`、`replay-result.json`：两次回放结果及原记录差异。
- `static_check.py`、`static-result.json`：重复 seed 验证、模型几何说明及历史关节构型检查。
- `replay.log`、`static.log`：执行日志。

两次回放均按分歧条件提前停止并正常关闭环境；静态检查正常关闭环境，未执行物理动作。
三个诊断进程均退出 0；没有发起新的自主任务，也没有新增任务成功率样本。
诊断脚本通过语法检查；运行代码未变，未重复运行无关测试。
