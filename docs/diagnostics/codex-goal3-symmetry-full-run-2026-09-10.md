# Goal 3 完整重跑：对称姿态最小版本

日期：2026-09-10。独立分支：`dev/huaizezheng/codex-plugin-smoke-2026-09-08`。
用户授权完整重跑一次 Goal 3；本轮只运行这一次任务，没有自动重开任务或额外调用其他模型。

> 后续复核勘误（2026-09-10）：下面对请求 #7/#20 的解释需要修正。
> 四元数转欧拉角存在奇异点错误，这两次 worker 收到的目标与选中目标相差 90°；
> 不能将其视为原目标正确传递后 Mink 仍失败的证据。
> 三次闭合拉动各自带动抽屉不足 0.002 mm，最终 12.2 mm 主要出现在最后撤离动作。
> 详细复现、接触证据与更新后的修复优先级见 [根因复核](codex-goal3-root-cause-2026-09-10.md)。

## 结果

**官方任务成功为 false。** Astra 在尝试多种把手接近方式后，完成撤离并主动调用
`finish_episode(success=false)`。碗仍在桌上；这次失败不是用尽工具调用或 episode 时间预算。

- 任务：`open the top drawer and put the bowl inside`。
- 环境：`openeta/libero_libero_goal_task3-v0`，seed=0。
- 模型：`gpt-6-astra`，reasoning effort=medium，ChatGPT 登录，经 Codex exec；无 API-key 模型调用。
- 工具：atomic 六工具，默认 `mink_joint_velocity`，无 SAM3、AnyGrasp、AnyPlace。
- 预算：2400 秒；native requests / Host turns / tool calls 上限均为 160。
- 时间：18:20:51–18:53:04 CST；launcher 用时 1930.618 秒，Host episode 用时 1903.356 秒。
- 实际：43 次 native 请求、84 个 Host turn、83 次内部工具调用。
- Codex exit=0、Host 正常关闭、integration_passed=true；这不代表任务成功。
- Agent session：`d2062af2162b4b57985e9e480336a3e4`。
- Execution：`fbbd5117-1aa1-4e8e-94da-826982089d3b`。

seed 使用当前修复后的程序化 reset，而非选择官方 LIBERO init-state 文件。
此前完整任务运行的 seed 接收尚未修复，且本次增加了状态保存/反馈/夹爪检查和 skill 说明、扩大总预算，
因此不能把新旧整任务表现视为只改变对称算法的严格 A/B 实验。

## 本次准备与代码范围

只修改 `plugins/openeta/skills/openeta-atomic/SKILL.md`：告知模型在空手预抓取时可优先选择
`orientation_mode=parallel_jaw_symmetric`，保持定向/携物/fixture 操作的 strict 语义，
并说明 3/5 次内部调用开销和 endpoint-only 局限。
通过新私有 Codex home 安装当次插件副本，未修改用户正常配置或缓存。
skill-creator 验证在 LIBERO Python 环境通过（主 `.venv` 缺少 YAML 依赖，未安装新依赖）。

运行期间没有改变目标、碰撞授权、控制器参数或向模型注入人工建议。
operator 对私有日志/快照的读取不回传给运行中的模型。`source_changed=[]`。
未修改 main、代理配置，未提交/推送代码，未更新外部协作文档。

## 运动统计

24 次 move_to 请求中，2 次因对称模式的夹爪/接触前置条件不满足而零步拒绝；22 次实际执行：

| 结果 | 次数 |
| --- | ---: |
| target_reached | 13 |
| collision_detected | 4 |
| iteration_limit | 3 |
| local_convergence_stalled | 1 |
| control_step_failed | 1 |

另外有 10 次打点、7 次夹爪命令（4 次打开、3 次闭合），均未使用外部感知模型。
7 次夹爪命令均完成各自受检查的 horizon；这不证明成功抓住把手。
动作工具总耗时约 927.016 秒；其余时间包含模型规划、图像传输、网络等待和结束流程，
不能全部归为模型推理或网络故障。

## 对称姿态实际使用

共 7 次请求携带 symmetric；2 次被前置状态检查拒绝，5 次实际比较候选并执行。

| Native 请求 | 选中转角 | 选中 IK 最小关节余量 | 结果 | 说明 |
| --- | ---: | ---: | --- | --- |
| 7 | 118.12° | 0.09488 rad | control_step_failed，114 步 | 原方向未找到解，选等价方向；实际第 6 关节逼近限位，姿态误差 23.40° |
| 8 | 102.16° | 0.81739 rad | target_reached，116 步 | 两候选可行，选择关节运动更小的一项；恢复到上方位置 |
| 20 | 46.79° | 0.00740 rad | local_convergence_stalled，95 步 | 另一候选未找到解；实际第 1 关节处在边界，姿态误差 13.16° |
| 30 | 28.66° | 0.13254 rad | target_reached，10 步 | 另一候选约 179.55°，选择成本低的原方向 |
| 36 | 114.07° | 0.64888 rad | target_reached，69 步 | 另一候选角度略小（109.88°），但关节位移更大，未获选 |

5 次执行中 3 次到位、2 次仍在关节边界失败。只有请求 7 实际切换了输入姿态的 jaw 正负号，
且该次未到位；其他 4 次选择原方向。这组任务数据证明候选比较及其反馈被模型实际使用，
不能单凭 3/5 就声称自动翻转提高了整任务成功率。

请求 7/20 都只有一个候选通过 IK 搜索。当前评分只在候选间排序，没有执行路径预演，
也没有因唯一候选过于脆弱而返回重新规划。这是最小版本的重要限制。
此前 179.47° → 50.56°、22 步到位的固定状态对照仍然有效，见
[局部实验](codex-orientation-symmetry-2026-09-10.md)，但不能推广为所有接近方向都可收敛。

## 碰撞与抽屉状态

本轮保存了 **59 个完整状态快照**：1 个初始 reset、22 次 move 和 7 次 gripper 的前后状态。
私有模型、状态与 outcome 在 `goal-3-setup/private-state/`。

4 次碰撞停止都发生在 `pre_actuation_configuration` 检查：此前已执行若干步，
被拒绝的是预测会违反距离约束的下一步。下表距离来自预测碰撞报告，不能当作实际已经发生的穿透深度。

| Native 请求 | 已执行步数 | 预测碰撞几何 | 距离 |
| --- | ---: | --- | ---: |
| 11 | 6 | gripper0_finger1_collision ↔ wooden_cabinet_1_g20 | -1.854 mm |
| 23 | 3 | gripper0_finger1_collision ↔ wooden_cabinet_1_g20 | -5.029 mm |
| 38 | 3 | gripper0_finger1_collision ↔ wooden_cabinet_1_g9 | -12.384 mm |
| 41 | 5 | robot0_link4_collision ↔ wine_rack_1_g1 | -2.181 mm |

前两次都已成功把打点绑定到 `wooden_cabinet_1_g18`，接触授权数量为 1；
`g20` 仍是受保护几何。这给“横杆授权与邻接把手连接几何不一致”提供了本轮直接证据。
后两次涉及柜子其他几何和机械臂与酒架，不能通过扩大把手授权一并解决。

只读解析保存模型的 cabinet slide joints 与快照 qpos：

- 上层抽屉初始 qpos=0，最终 qpos=-0.01221492 m，约 **12.2 mm** 位移。
- 已保存动作边界状态中的最大绝对位移也是 12.2 mm；这不是每个物理积分步的全程峰值。
- 中层和下层抽屉在这些状态中保持 0。
- 几次夹爪 6–9 cm 拉动没有形成稳定、充分的抽屉打开；未完成碗放置。

结果记录为 `physics-analysis.json`；只读分析程序不连接或推进运行中的仿真。

## 网络

- Codex 记录 **1 条断流/重连事件**，18:28:29：
  `Reconnecting... 2/5 (stream disconnected before completion: websocket closed by server before response.completed)`。
- 后续自动恢复并继续任务。该提示不是 2 条已记录断流，不能按 2/5 推算故障次数。
- 最长相邻动作空档 **216.781 秒**，位于该事件附近；其中各部分耗时不能进一步精确归因。
- stderr 有 **8 条模型目录刷新失败**记录，单独统计，不与任务流断开次数混为一谈。
- 65 轮 ChatGPT HEAD 探测均完成 TLS/HTTP 返回，状态码为 405；P50 1.069 s、P95 4.038 s、最大 7.383 s。
- 本地代理 TCP 65/65 成功。短连接探测不能证明已认证模型 WebSocket 的稳定性。
- 已读取本次窗口的 mihomo journald：610 条原始记录，377 条相关记录，经脱敏保存。
  选出的相关日志没有匹配到 warn/error/timeout/reset/broken pipe 等问题关键字。
  尚不能由此定位断流根因，也不能直接归因于本地代理或 OpenAI 服务。

## 清理、额度记录与下一步

监督器确认 `remaining_owned_pids=[]`、`port_released=true`、`private_auth_removed=true`。
Host cleanup 成功，所有当前任务自有服务/子进程已清理，普通用户 Codex 登录文件保持原样。

Codex 原始 usage 字段：input_tokens=4,871,602，其中 cached_input_tokens=4,689,024；
output_tokens=6,642，reasoning_output_tokens=1,222。不据此换算订阅扣费或 API 价格。

本次未扩大授权或改动路径控制器。后续优先项是：

1. 对只有脆弱 IK 解的候选增加实际控制器路径检查/明确的重规划结果。
2. 核查并细化把手组件的接触授权，使用已保存状态比较 g18/g20 接触行为。
3. 对柜子和酒架等障碍，结合全臂碰撞反馈改善接近路径；不能只调整末端转角。
4. 验证夹持保持与抽屉位移反馈，使“拉动到位”和“抽屉确实打开”容易区分。

全部原始记录在 `tmp/codex-goal3-symmetry-20260910/`：
`batch-status.json`、`source-snapshot.json`、`goal-3/summary.json`、`goal-3/analysis.json`、
`motion-analysis.json`、`physics-analysis.json`、`network/`、`mihomo-events.jsonl`。
本轮只扩展 skill 说明，没有修改共享 schema；此前新增工具参数的 main 合入仍需三人接口评审。
