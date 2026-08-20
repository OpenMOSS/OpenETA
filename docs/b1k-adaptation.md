# BEHAVIOR-1K 专项适配分支

本仓库工作树用于隔离 BEHAVIOR-1K（B1K）仿真适配与主干 Agent 开发。它从
Agent 基线提交 `9c2ce70` 创建，开发分支为
`dev/huaizezheng/b1k-sim-adapter`。

## 仓库与远程约定

- 主干 Agent 工作树：`Stage2-OpenETA/OpenETA`
- B1K 专项工作树：`Stage2-OpenETA/OpenETA-B1K`
- `origin`：`https://github.com/OpenETA7/Stage2.git`
- `upstream`：`https://github.com/OpenMOSS/OpenETA.git`
- B1K 分支定期从确认过的 Agent 集成基线同步；不要直接把主干工作树的未提交文件复制进来。

## 职责边界

共享 RFC 第 4、5、9.2 节是本分支的接口依据：

- Agent 只调用稳定的 Agent tool，不感知 18/23 维 raw action、OmniGibson controller slot 或 Isaac Sim 生命周期。
- B1K 专项代码负责环境启动、观测归一化、低层控制编码、worker 生命周期和环境真实回执。
- `EnvObservation`、`ToolResult`、可信 environment receipt 及 Agent 可见 tool schema 是共享硬契约。
- 字段重命名、枚举变化、成功语义变化或新增 Agent 可见控制工具，必须先进入 RFC 协作评审；不得只在 B1K 分支形成私有 Agent 协议。
- 环境特有的兼容字段应采用向后兼容追加，并保留唯一规范字段。例如夹爪连续开度规范为 `openness`，旧 `open_fraction` 仅作为兼容 alias。

## 代码归属

现有 `sim/envs/behavior/` 已经是 B1K agent-facing DirectEnv 的唯一实现位置，
不再新建第二套环境 wrapper：

- `sim/envs/behavior/direct_env.py`：OmniGibson 单环境封装、RGB-D/机器人状态与控制布局。
- `sim/mcp_server/action_codecs.py`：稳定 MCP 控制指令到 B1K action slot 的编码。
- `sim/mcp_server/`、`sim/bench_worker.py`：worker 部署、会话与 MCP 工具。
- `scripts/behavior_*.py`：安装、预检、真实环境 smoke 与实验入口。
- `tests/b1k/`：不依赖 Isaac Sim 的 Agent–MCP 边界门禁。
- `tests/test_behavior_*.py`：DirectEnv、RLinf batched wrapper 与完整 B1K 单元契约。

除非共享 Agent contract 已经完成评审并进入上游基线，本分支不修改
`agent/tools/registry.py` 中的稳定 tool schema。

## Agent–MCP 适配状态

| Agent tool | MCP server 操作 | B1K 状态 | 专项要求 |
|---|---|---|---|
| `create_simulator_env` | `create_env` + `reset_env` | 已接入 | R1Pro、action dim 和 `control_spec` 必须来自真实创建回执 |
| `close_simulator_env` | `close_env` | 已接入 | 幂等关闭并回收单用途 Isaac worker |
| `observe` | `render_env` | 已接入 | 三相机 RGB + optical-Z depth + 对应 OpenCV 外参 |
| `move_to` | `move_to` | 部分接入 | 已有显式右臂 delta-pose slot；仍需 B1K IK/碰撞能力与真实 GPU 验证 |
| `follow_eef_trajectory` | `follow_eef_trajectory` | 部分接入 | 复用 `move_to`，受相同 IK/碰撞限制 |
| `gripper_control` | `gripper_open` / `gripper_close` | 已接入 | 使用创建回执声明的 gripper slot；观测输出 `openness` |
| `ik_preview_check` | `ik_preview_check` | 未专项适配 | 当前应明确返回 `unknown/backend_unsupported`，不能伪造可达 |

### 当前显式缺口

1. `scripts/behavior_openeta_agent.py` 仍为实验 runner，其中底盘和夹爪 handler 直接构造固定 23 维 action。它不能作为通用 Agent 部署入口。
2. `lower_body_control_policy` 尚未进入稳定 Simulator MCP binding。新增 B1K 底盘控制前，需要评审 Agent 参数语义和 server-side 工具名；评审前不能把 `step_env` 暴露给 Agent。
3. B1K 尚未填充可供 attachment/collision 推理使用的规范 `objects` 列表。
4. B1K 尚无真实 `ik_preview_check`、路径碰撞检查和 R1Pro 抓取标定证据。
5. CPU 契约测试不能替代 Isaac Sim 5.1 + R1Pro 的 GPU smoke、任务 reward 和 termination 验证。

## 实施顺序

### P0：锁定边界

- 维护 `tests/b1k/test_agent_mcp_boundary.py`，防止 raw `step_env` 或环境 action layout 泄漏到 Agent tool contract。
- 归一化 `openness`、末端 7D `xyz + quat_xyzw`、相机标定、`step_index` 和真实环境回执。
- 保留现有 LIBERO 行为作为对照，所有 B1K 差异必须位于 simulator-side adapter。

### P1：补齐运动与场景状态

- 对 B1K `move_to` 做真实 R1Pro 闭环验证并增加 IK preview。
- 提议并评审底盘控制的稳定 tool 参数；评审通过后再实现 MCP handler，替换实验 runner 的 raw action。
- 提取规范 scene objects / attached-object evidence，接入碰撞和抓取后验证。

### P2：部署验收

- 运行 `behavior_preflight.py`、Isaac smoke、OpenETA smoke 和 Agent MCP smoke。
- 记录 create/reset/observe/move/gripper/close 全链路延迟、reward、termination、视频与失败诊断。
- 只有真实 BDDL checker 的 reward/termination 可作为任务成功证据。

## 本地验证

不安装 Isaac Sim 时先运行：

```bash
python -m pytest -q tests/b1k tests/test_behavior_integration.py tests/test_sim_control_codecs.py
```

安装 B1K runtime 后再运行：

```bash
python scripts/behavior_preflight.py --output outputs/behavior/preflight.json
sim/venvs/behavior/runtime/bin/python scripts/behavior_isaac_smoke.py
sim/venvs/behavior/runtime/bin/python scripts/behavior_smoke.py --task picking_up_trash --seed 0
```
