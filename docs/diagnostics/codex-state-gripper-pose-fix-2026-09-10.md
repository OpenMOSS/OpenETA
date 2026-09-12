# Codex 实验：状态保存、seed、夹爪检查与姿态误差

日期：2026-09-10。分支：`dev/huaizezheng/codex-plugin-smoke-2026-09-08`。
本轮仅修改独立副本，没有修改 main、网络配置或提交/推送代码。没有调用模型，也没有新增自主任务成功率样本。

## 已完成的改动

### 初始状态和私有诊断记录

`scripts/codex_sim_server.py` 默认启用 `OPENETA_PRIVATE_STATE_DIR`，目录为
`tmp/codex-private-state/<启动时间>/`，可用 `--private-state-dir` 指定其他 operator 目录。
状态文件不放进模型工作区，不作为工具返回字段。

Worker 自动记录初始 reset，以及每次 Mink move/gripper 动作之前和之后的状态：

- 完整 MuJoCo 二进制模型，按 SHA256 去重，包含实际场景几何。
- `mjSTATE_INTEGRATION`，包括关节状态、执行器和求解器 warmstart；另存 qpos/qvel/ctrl 便于检查。
- Robosuite 环境、机器人、夹爪和速度控制器的数值状态，包括 PID 的导数历史缓冲区。
- 环境隔离 RNG 和进程随机状态、MuJoCo 版本、源文件哈希、私有请求及原始安全结果。
- handle/任务/控制器信息用于关联记录。初始 reset 快照位于 Host 的 5 步 settling 之前；首个动作的 start 快照包含 settling 后的状态。

`sim.private_state.restore` 为本地诊断辅助函数，只用于兼容的已有 LIBERO 环境；
它校验版本、模型校验和和基础拓扑，恢复模型数组、物理积分状态和控制器状态。
未支持的对象字段在 metadata 中列出，不声称任意跨版本或所有后端都能精确回放。

真实验证发现，仅保存物理状态和 PID 标量仍会产生约 1e-7 量级差异；
补存 `derr_buf` 等历史缓冲区后，同一快照重放 5 步的 qpos/qvel 最大差均为 **0**。
保存动作前快照失败时不执行动作；动作后落盘失败保留已执行步数和原安全回执，
不会把已经发生的运动错误报告为零步。

二进制模型在该 LIBERO 场景中约 211 MiB；同一模型只保存一次，单次状态目录约 70–84 KiB。
本轮三个独立验证目录分别保留了模型副本。

### LIBERO seed

`make_env → _make_libero_direct_worker → _make_libero_direct → _LibEnvWrapper`
完整传递 seed。构造和 reset 都使用隔离的 NumPy/Python RNG，显式 reset seed
也会传给底层 LIBERO。避免重置一个环境时改变其他环境的随机序列。

- 同一 seed 再次 reset，实际 qpos 和模型 body_pos 逐项完全相同。
- 不同 seed 改变场景布局。
- reset(seed=None) 延续该环境自己的随机流；创建时的 seed 决定初始随机流。
- 非整数、布尔值、负数和越界 seed 在执行 reset 前被拒绝。

这是修复当前 **程序化场景 reset** 的可重复性，没有改成选择官方 LIBERO init-state 文件。
做正式基准对齐时仍应单独指定官方初始状态协议，不能混称为同一评估设置。

### 夹爪检查与渲染

新增 worker 私有 `/gripper-goal` 路径，Mink 夹爪动作复用同一 live MuJoCo 碰撞/授权策略：

- close 前发现已有受保护穿透，零步拒绝继续挤压。
- 每个实际物理控制 tick 后检查受保护几何，遇到新碰撞即停止剩余 horizon。
- open 可以对已有穿透进行检查过的单调退出；没有进展或产生新的碰撞则停止。
- 单 geom 的 fixture 接触授权范围保持原样；没有放开整个柜子，也没有迁移第二套 cuRobo 判定。
- 返回实际执行步数、horizon_completed 和碰撞阶段；不再把中断的 close/open 报为完成 60/40 步。

这是当前状态和 **每个控制步后的检查**，不预测接触动力学、不保证控制 tick 内所有积分子步都零穿透。
回执明确 `prediction_checked=false`。检测到步后碰撞时，该步已经发生。

原来的 `render=False` 仅关闭额外渲染/图像编码，Robosuite 的 camera observables
仍可能在 env.step 内渲染。新夹爪路径同时暂停 camera observables，末尾重新启用并刷新
agentview、腕部和深度，检查和渲染彼此独立；异常退出也恢复 camera observables。
真实接口验证：60 个 close 控制步，HTTP 200，约 **1.47 秒**，实际只调用两次相机渲染
（agentview、robot0_eye_in_hand），均在动作末尾。最终返回图像和私有快照齐全。

该新检查适用于默认 Mink 实验。显式选择 OSC 的旧夹爪路径没有伪装成已经具备此检查。

### 明确的反馈

Mink 根据实际几何归属生成机器人/世界或自碰撞类别；agent 安全投影和原子工具继续保留：

```json
{
  "reason_code": "collision_detected",
  "physics_executed": true,
  "motion_summary": {
    "steps_executed": 3,
    "collision": {
      "collision_class": "robot_world",
      "robot_part": "gripper",
      "check_mode": "post_step_configuration",
      "checked_during": "gripper_actuation",
      "contact_binding_active": true,
      "prediction_checked": false
    }
  }
}
```

`contact_binding_active` 只表示当前存在接触绑定，不表示碰撞对已获授权。
原始 geom 名字、距离和场景状态留在私有 outcome 中。
零步拒绝会保留新鲜打点；实际执行过或执行状态未知时，仍保守地使旧接触测量过期。

运动反馈另外提供 `ik_seed_validated`、位置/姿态是否满足容差、实际关节限位裕度和
`execution_failed_after_validated_ik`，避免把“局部执行失败”误说成“目标 IK 不可达”。
新增字段为兼容性扩展；按协作约定，合入 main 前需要三人接口评审。本轮未更新外部协作文档。

## 20° 以上姿态误差的调查结果

原记录：`tmp/codex-goal-0-3-retest-20260910/goal-3/`。
逐一取出 15 次实际执行使用的 IK joint seed，在同一 Panda 模型上独立做正运动学；
**15/15 都满足原目标位置和姿态容差。**
这是对已记录关节解的静态核验，不依赖原场景随机布局，能够证明这些目标存在运动学解。

| 原始响应序号 | 最终姿态误差 | IK 解最小关节裕度 | 实际终点状态 |
| --- | ---: | ---: | --- |
| 0009 | 24.396° | 0.13398 rad | 第 7 关节距上限约 0.000172 rad |
| 0011 | 23.579° | 0.97155 rad | 第 4 关节超出软限位约 8.19e-6 rad |
| 0022 | 23.430° | 1.18241 rad | 第 4 关节超出软限位约 6.02e-6 rad |
| 0024 | 15.270° | 1.27226 rad | 第 4 关节超出软限位约 4.43e-6 rad |
| 0032 | 24.933° | 0.00807 rad | 第 2 关节超出软限位约 1.48e-5 rad；备用步被拒绝 |

表中关节序号是人类习惯的 1–7；回执 `nearest_limit_joint_index` 使用 0–6。
不是每个失败都能归因于关节限位：响应 0026 姿态误差 14.398°、位置误差约 20 mm，
实际关节仍有较大裕度，该次接触动力学/约束原因需完整状态复盘，不能混入同一个结论。

响应 0009 请求的姿态变化约 **179.47°**，是一次接近半圈的大姿态调整。
当前局部笛卡尔 QP 仍可能停在与全局 IK 解不同的冗余构型/约束边界；
传入一个可行 seed 不等于已经有通往它的可执行路径。
这一解释有实际终点关节数据和下述对照支持；尚未通过消融证明某一个权重参数是唯一根因。

## 同一快照的 150/300 步对照

用响应 0009 的原始目标和 IK seed、上一段记录的机器人关节姿态，
将初速度设为零，在新 seed=0 场景创建快照。两次运行都恢复同一完整快照，
保持 Mink 和碰撞保护，仅改变 iteration budget：

| 最大控制步数 | 停止原因 | 位置残差 | 姿态残差 | 最近限位 |
| --- | --- | ---: | ---: | --- |
| 150 | iteration_limit | 5.768 mm | 24.088° | 第 7 关节，约 -5.24e-5 rad |
| 300 | iteration_limit | 3.374 mm | 24.291° | 第 7 关节，约 0.000109 rad |

该对照不是原任务的精确回放，原始世界状态和速度此前未保存；
但两种 budget 之间是同状态对照，支持“单纯翻倍步数不能解决这个大姿态调整”的结论。

本轮没有放宽姿态容差，也没有把未到位当作成功；控制器轨迹生成的根本改进尚未实施。
建议后续验证：在有空间余量处先退出限位构型，再分段调整姿态；对称夹爪的 jaw 与 -jaw
可作为显式候选方向比较 IK 裕度和所需转角，避免不必要的近 180° 翻转。
若修改 seed 的冗余构型引导或加入局部路径预检，应在固定快照上做配对对照，
不能静默替换 agent 已选的目标姿态。

## 验证与产物

- Python 主环境相关回归：**122 passed**，覆盖 seed、私有记录失败语义、Host/原子反馈、碰撞、夹爪、MCP stdio 等。
- MuJoCo/Mink 环境：**11 passed**，覆盖实际几何导致的夹爪早停、受保护 close、检查过的 release、授权范围和关节约束。
- 共 **133 个通过的测试**；局部重复测试不重复计数。
- 真实仿真：相同 seed 完全复现、不同 seed 改变布局、快照回放 qpos/qvel 差为 0、60/40 步空中 close/open 通过。
- 真实 worker 夹爪接口：200、实际 60 步、末尾两路相机渲染、三个阶段快照与私有 outcome 齐全。
- 历史 IK 正运动学核验：15/15；150/300 步对照均保持碰撞保护。
- 沙箱内完整 pytest 在 stdio 环节等待退出；终止精确核验的本轮测试进程后，同一测试集在允许本地子进程的环境 **6.84 秒全部通过**。未更改产品代码来绕过测试。
- 所有本轮仿真环境已关闭，测试和仿真进程已退出；没有启动付费 API 或订阅模型任务。

本轮目录：`tmp/codex-safety-state-fix/`。
`real-validation.json` 保存真实验证和历史姿态核验；`pose-isolation.json` 保存同快照步数对照；
`worker-validation.json` 保存接口、渲染与快照检查；`tests-unsandboxed.log` 和
`mujoco-tests-final.log` 保存最终测试结果；`before/` 保留本轮修改前的已有运行文件。
