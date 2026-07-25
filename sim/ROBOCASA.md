# RoboCasa365 benchmark 接入

OpenETA 的 RoboCasa 接入以官方 RoboCasa 1.0.1 benchmark 为准，不把旧的
RLinf 训练向量环境当作评测环境。默认机器人是移动式 `PandaOmron`；也可用
`robot="Panda"` 运行固定底盘的 7D OSC 原子任务。固定 Panda 是接入与策略
验证配置，不替代官方 RoboCasa365 默认 PandaOmron benchmark 口径。

## 环境与资产

```bash
./scripts/setup_envs.sh robocasa
source sim/venvs/robocasa/bin/activate
source sim/venvs/robocasa/activate_extra.sh
```

安装脚本会完成以下工作：

- 用 uv 创建 `sim/venvs/robocasa`（Python 3.11）；
- checkout `benchmark_versions.json` 中固定的 RoboCasa 和 robosuite commit；
- 在隔离环境里安装 RoboCasa 的 `gymnasium 0.29.x` 与 OpenETA worker 运行依赖；
- 运行两个项目的 macro setup；
- 下载 `--type all` 的官方厨房资产，并写入幂等完成标记。

解压后的资产当前约 23 GB。源码、venv 和资产都位于被 gitignore 的
`sim/venvs/robocasa/`，不会进入提交。

## 环境契约

每个官方任务注册两个环境：

```text
openeta/robocasa_pretrain_<TaskClass>-v0
openeta/robocasa_target_<TaskClass>-v0
```

当前官方注册表有 65 atomic + 252 composite = 317 tasks，共 634 个 env
ID。reset seed 决定 scene/object scenario；horizon 和 `_check_success()` 直接
使用 RoboCasa 官方实现。

OpenETA 协议输出保持统一：

- `EnvObservation.cameras`：PandaOmron 为 left、right、wrist 三路 RGB-D；
  固定 Panda 为 agentview、wrist 两路 RGB-D；depth 均为米；
- `RobotState`：关节、世界系 EEF pose、gripper、PandaOmron base pose；
- metadata：task、split、scenario seed、horizon、elapsed steps；
- `StepResult`：二值 reward、success termination、horizon truncation。

官方 12D action 顺序为：

```text
0:3   EEF position delta (PandaOmron base frame)
3:6   EEF axis-angle delta (PandaOmron base frame)
6     gripper, -1=open / +1=close
7:10  mobile base forward / lateral-left / yaw-CCW velocity
10    torso
11    hybrid mode, -1=arm / +1=base
```

固定 Panda 使用标准 7D action：

```text
0:3   EEF position delta (fixed Panda base frame)
3:6   EEF axis-angle delta (fixed Panda base frame)
6     gripper, -1=open / +1=close
```

RoboCasa 1.0.1 的厨房 reset 原本无条件写入 PandaOmron 移动关节；OpenETA 的
direct wrapper 仅对无移动关节的 Panda 安装兼容放置逻辑，并把静态根节点安装
到任务采样锚点。上游源码和官方 PandaOmron 路径不被修改。

`move_to` 对外仍接收世界坐标，MCP server 会用 observation 中的 base
quaternion 转到控制器 base frame。`lower_body_control_policy` 映射到
`base_control`，可用结构化 `forward/lateral/yaw/torso/num_steps`，也可用
`forward/backward/left/right/turn_left/turn_right/stop` 快捷命令。

配置好 planner provider 并启动 OpenETA MCP server 后，CLI 可直接运行真实
RoboCasa episode，同时保存 agent 实际接收的相机反馈：

```bash
openeta --once "pick the sweet potato and place it in the cabinet" \
  --env-id openeta/robocasa_target_PickPlaceCounterToCabinet-v0 \
  --sim-mcp-url http://127.0.0.1:8765/mcp \
  --seed 43 --image-width 256 --image-height 256 \
  --video outputs/robocasa/pick_place_seed43.mp4 --video-fps 10
```

项目自身的 legacy `/sse` URL 也可以传入；CLI 会升级到同源、持久的
Streamable HTTP `/mcp` 会话，从环境创建一直复用到 `close_env`。

固定 Panda 的可复现实验（官方 checker + 双视角视频）可直接运行：

```bash
MUJOCO_GL=egl PYTHONPATH=. sim/venvs/robocasa/bin/python \
  scripts/robocasa_fixed_panda_demo.py
```

默认任务是 target split、seed 0 的 `StartCoffeeMachine`。脚本使用特权按钮
位姿做接入校准，不代表通用视觉策略；任务成功仍只由 RoboCasa 原生
`_check_success()` 判定。

## 官方评测清单与恢复

官方协议对选定 task set/split 的每个任务评测 50 个随机 scenario，并对
二值 success 求平均。先把随机性固化为清单：

```bash
python -m scripts.robocasa_benchmark manifest \
  --task-set all_tasks \
  --split target \
  --scenarios-per-task 50 \
  --seed 0 \
  --output runs/robocasa/all_tasks_target.json
```

`all_tasks` 会生成 15,850 个 rollout。manifest 保存 task、split、seed、
horizon、RoboCasa commit 和 SHA-256；任务顺序变化不会改变已有任务的 seeds。

评测 runner 是 `python.module:callable`。callable 接收一个不可变的
`RoboCasaScenario`，完成整条 episode，并返回 `RoboCasaRolloutResult` 或同字段
mapping：

```python
def rollout(scenario):
    # scenario.env_id / seed / horizon 可直接交给 OpenETA agent 或 vector policy
    return RoboCasaRolloutResult(
        scenario_id=scenario.scenario_id,
        task=scenario.task,
        split=scenario.split,
        seed=scenario.seed,
        success=success,
        steps=steps,
    )
```

```bash
python -m scripts.robocasa_benchmark evaluate \
  runs/robocasa/all_tasks_target.json \
  --runner my_policy.robocasa:rollout \
  --output runs/robocasa/results.json
```

结果在每个 rollout 后原子写盘；同一命令会依据 manifest SHA-256 跳过已经
完成的 scenario。`--max-rollouts 1 --runner noop` 可用于物理/MCP smoke test，
但 noop 不是有意义的 benchmark policy。

## 版本边界

RoboCasa/robosuite 的注册任务、controller layout 或 horizon 改变时，先更新
`sim/envs/robocasa/benchmark_versions.json`，再跑真实 12D PandaOmron 和 7D
固定 Panda smoke test 与全量单元测试。若 action spec 与所选机器人不匹配，
环境会直接拒绝运行，避免产生看似成功但协议错误的 benchmark 结果。
