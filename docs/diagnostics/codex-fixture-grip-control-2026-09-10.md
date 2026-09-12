# 活动机构夹持运动与抓碗碰撞反馈修复

日期：2026-09-10，独立 `OpenETA-codex-plugin` 工作树。20 项长期 goal 仍在进行。
未修改 main，未提交、推送、外发消息或调用外部感知模型。

## 完整 Goal 3 attempt-002 的结果

21:05:14 开始，40 分钟 Host 时限耗尽，官方 task_success=false；
CLI 正常退出、integration_passed=true，不能据此算任务成功。
53 个 native completed calls、90 个内部 tool calls，0 个记录到的 stream errors。
自有进程全部退出、端口释放、私有 auth 副本删除，source_changed=[]。
原始证据：`tmp/codex-libero20-20260910/goal-3/attempt-002/{result.json,run/summary.json}`。

修复坐标系后首次抓取产生双侧指垫接触，首次 40 mm 拉动使抽屉打开 38.088 mm。
随后一次 180 mm 请求发生脱手，只再打开 39.201 mm；后来模型通过重新抓取和短拉动，
在抓碗阶段前已将抽屉打开到约 159.920 mm（关节下限 -160 mm）。
因此最终未成功的主要剩余障碍是抓碗接近，不是抽屉没有打开。

三次抓碗失败接近（native #41、#43、#48）的完整状态回放分别精确复现 12、1、8 步后停止。
均为 `gripper0_hand_collision` 对 `wooden_cabinet_1_g11` 的下一步预测碰撞：
预测有符号距离分别为 -1.286、-1.510、-5.337 mm，不能称为真实穿透深度。
#43、#48 的碗接触授权已生效（40 个目标几何体），仍被柜体阻挡。
证据：`tmp/codex-goal3-bowl-collisions-20260910/report.json`。
最初该诊断脚本未处理可选授权字段，发生 KeyError 并关闭环境；修正脚本后重放通过，日志保留。

## 成对控制回放

均在独立仿真实例恢复完整保存状态，使用相同物理参数；无模型调用，不向执行任务的模型提供真值。
历史长拉动状态：`1789045978847351903-e5464399-move-start`。
原速回放精确复现原始 17 步停止与最终 drawer qpos。

| 18 cm 请求对照 | 抽屉打开增量 | 末态接触 | 采样到的峰值姿态误差 |
| --- | ---: | --- | ---: |
| 原始 0.5 rad/s | 39.201 mm | no_contact | 17.677° |
| 仅降至 0.2 rad/s | 40.209 mm | no_contact | 20.301° |
| 仅降至 0.1 rad/s | 77.584 mm，局部停滞 | no_contact | 20.314° |
| 0.5 rad/s，姿态权重乘 25 | 111.107 mm | no_contact | 2.577° |

原始 18 cm 命令超过剩余约 12.2 cm 的抽屉行程，因此最终失去接触不是纯粹的姿态控制比较。
不能据此说“仅降速有效”或“增强姿态权重已经解决夹持”。
另做行程内 8 cm 目标，目标与 IK seed 在全部条件之间相同：
原控制打开 40.237 mm 后失去接触；姿态权重乘 25 后打开 75.878 mm，但末态仅单侧指垫接触。
增加稳定到达后，0.5 rad/s 和 0.2 rad/s 分别打开 75.055、76.545 mm，末态均双侧指垫接触。
这是控制步采样的接触和姿态信息，不是连续物理子步最大值，也不证明任意后续运动都不会脱手。

再用另外四处历史短拉动的原目标、原 IK seed 验证最终集成实现：

| Native 请求 | 原控制增量 / 接触 | 新控制增量 / 接触 |
| --- | --- | --- |
| #12 | 38.088 mm / bilateral_pads | 40.951 mm / bilateral_pads |
| #18 | 25.204 mm / no_contact | 31.275 mm / bilateral_pads |
| #35 | 22.838 mm / bilateral_pads | 22.406 mm / bilateral_pads |
| #36 | 32.901 mm / bilateral_pads | 32.582 mm / bilateral_pads |

#36 已接近行程端点，不能用它估计无限制拉动的跟踪精度。
四次新控制都达到姿态容差，并有 3 个稳定控制步、最终关节速度 <0.05 rad/s。
三次张开夹爪抓碗接近的新旧完整控制结果逐项相同，稳定控制均未激活。
证据：`tmp/codex-fixture-grip-integrated-20260910/report.json`，passed=true、cleanup=true。

其余回放目录：`tmp/codex-fixture-pull-{speed,orientation,validation}-20260910/`，
`tmp/codex-fixture-pull8cm-{orientation,settle}-20260910/`。全部环境已关闭。
早期记录的特定横杆原生 contact 列表与重建接触不一致；本结论不使用该列表推断接触力或脱手时刻。

## 最终改动及适用边界

- `sim/controllers/fixture_grip.py`：仅在 Host 开启、有效活动机构接触授权、闭合命令、
  双侧指垫接触、有效 IK seed、目标与当前姿态差 ≤0.05 rad 时启用。
  上限取原值与 0.2 rad/s 的较小值，姿态权重乘 25，稳定到达至少 3 步。
  不选择目标，不推断行程，不增加接触权限；主动大角度旋转、张开接近及自由物体搬运保持原控制。
- `scripts/codex_sim_server.py` 为独立 Codex Mink 实验显式开启该选项；
  `--no-fixture-grip-stabilization` 可做成对基线。普通 worker 未设置环境变量时不启用。
  回执记录实际参数、`fixture_grip_stabilization_active`、`transport_profile=fixture_grip_stabilized`；
  原 A/B/C 条件标签保留，启用后的实际行为必须描述为该条件加 fixture override，不能当作原始 A。
- 记录 `control_tick_peak_orientation_error_rad`，避免只看终点误差遗漏中途转动。
- 碰撞反馈新增 `gripper_part`（finger/base_or_palm）与 `obstacle_relation`
  （robot_self/unbound_world/authorized_target/outside_contact_target）。
  不公开模拟器名称、ID、几何位置、距离或运动建议向量。
  目标已授权但障碍在目标之外时，明确提醒同处重打点不能消除障碍，需从图像调整接近方向或途径点。
- 监督器支持创建 `batch-root/pause-after-current` 后在任务边界暂停，避免维护时启动额外任务；
  源码审计同时检查新增、删除和修改文件。清除该标记后可重新运行原命令。

共享回执的兼容新增字段应在合入 main 前由三方审查；本文记录本地接口变化，未代替共同批准。
当前实验设置没有增加工具，也没有调节环境摩擦、抽屉阻尼或夹爪物理参数。

## 验证与后续任务

- Python 3.13：姿态、夹爪、接触反馈、控制门限等 104 passed，1 skipped（该环境缺 MuJoCo）。
- LIBERO Python：实际 MuJoCo 接触、闭合检查、控制门限、关节边界 29 passed。
- 监督器测试：7 passed；修改文件语法检查及 tracked diff whitespace 检查通过。
- 插件使用 cachebuster helper 更新为 `0.1.0+codex.20260910135605`；LIBERO Python manifest 校验通过。
  最初默认 Python 校验因缺 PyYAML 失败，切换已有 LIBERO Python 后通过，未安装新依赖。
- task boundary 维护暂停与 Goal 3 自然结束有微小竞争，Object 0 attempt-001 只启动服务约 0.1 秒，
  未启动模型，完整清理后保留为 setup_or_supervisor_error；不是一次自主任务失败。
- 之后开始 pass3：Object 0–9，再 Goal 0–9；每次新建 Codex exec 会话并在私有目录重装新插件。
  仍使用 Astra medium、ChatGPT 订阅认证、Mink、seed=0 程序化 reset、2400 s/160 次限制。
  当前官方成功仍为 0/20，诊断回放不计任务成绩。
