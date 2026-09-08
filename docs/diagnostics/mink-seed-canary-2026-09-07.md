# Mink preview seed 短程执行诊断 — 2026-09-07

结论：独立 LIBERO Object 0、seed 0 环境中，preview seed 经直接 Host 诊断入口送入 Mink，回执确认匹配的 seed ID 与 seeded execution policy；上移 2 cm 到达目标，清理确认成功。不是完整 pick-place、主 Agent receipt resolver 或复杂 IK 支路验收。

## 环境与证据

- 个人分支 `dev/huaizezheng/harness-refactor-2026-09-05`，未提交候选。
- 新建私有服务 `127.0.0.1:18766`，单 worker、GPU 0，未替换既有 8766 服务。
- 解释器 `sim/venvs/libero/bin/python3.10`，隔离依赖 `/tmp/openeta-mink-canary-min`；Mink 0.0.6、qpsolvers 4.13.0、quadprog 0.1.13、MuJoCo 3.3.0。未安装或替换依赖。
- Simulator session `578b56a0-44cb-4378-8a5e-6bb134920a95`，handle `5dca7f20-ebd`。
- 原始报告：`tmp/experiment-ready-r0-21.VjhGgR/mink-seed-canary.json`。该 tmp 文件为本机证据，不保证随仓库分发。
- 只运行本地模拟控制器，没有向 SAM3、AnyGrasp、AnyPlace 或模型发送仿真数据。

| 检查 | 本次结果 |
| --- | --- |
| 控制器 | `mink.robosuite_joint_velocity` / `joint_velocity` / `openeta.worker_mink_goal.v1` |
| Preview | reachable；world/self 与场景物体检查完成，7 个障碍物，未报告碰撞 |
| 起点 / 目标 z | 0.2612794757 / 0.2812794757 m；x/y 不变 |
| 实际终点 xyz | `[-0.1478677946, 0.0010050537, 0.2797681073]` |
| 误差 / 步数 | 欧氏误差 0.0019106586 m；2 / 40 controller steps |
| Seed 回执 | `controller-canary-92309f657a864eef9ec42d802b9d6439` 匹配；`ik_preview_seeded_cartesian_goal` |
| 运动碰撞回执 | world/self checked；逐步 pre-actuation/post-step configuration 检查，未报告碰撞 |
| 停止 / 清理 | target_reached；close_state=closed，remote.ok=true，cleanup_errors 为空 |

先确认环境 close acknowledgement，随后停止本次自建服务；服务进程退出码 0。没有关闭其他 session 或处理已有控制台请求。

## 入口改动与复现

`scripts/controller_capability_canary.py` 新增可选 `--require-ik-seed`，仅允许 Mink 正向非零运动诊断。必须从 reachable preview 取得七个有限、非布尔关节值，绑定新的诊断 receipt ID，并要求运动回执的 ID 与 execution policy 匹配。预览/运动姿态容差均显式为 0.05 rad，保持当前位置姿态。原有碰撞、终点、步数和关闭检查不放宽。

以下会创建新仿真；先确认端口空闲、依赖覆盖目录有效，再使用新输出文件名。每次 RPC 超时不是整个脚本的总期限。

```bash
timeout --signal=INT --kill-after=10s 480s env \
  OPENETA_WORKER_POOL_MAX=1 OPENETA_WORKER_GPUS=0 \
  OPENETA_LIBERO_CONTROLLER_PROFILE=mink_joint_velocity \
  OPENETA_LIBERO_MINK_DEPENDENCY_PATH=/tmp/openeta-mink-canary-min \
  sim/venvs/libero/bin/python3.10 -u -m sim.mcp_server \
  --host 127.0.0.1 --port 18766
```

另一终端：

```bash
.venv/bin/python -m scripts.controller_capability_canary \
  --url http://127.0.0.1:18766/sse \
  --env-id openeta/libero_libero_object_task0-v0 --seed 0 \
  --expected-controller-id mink.robosuite_joint_velocity \
  --require-ik-seed --max-steps 40 --timeout-s 45 \
  --output tmp/mink-seed-canary-NEW-RUN.json
```

## 不应外推的结论

Host 诊断直接将 preview seed 送入 simulator，报告显式标记 `source=direct_preview_not_agent_receipt_resolution`。尚未验证主 Agent 的 proposal → IK receipt → move 引用解析全链路；也未验证复杂转腕、关节极限、多支路选择或 seed 对成功率的因果改善。

本次 motion 回执报告逐步配置检查；preview 明确 path.checked=false。离散检查不等同任意连续 swept-path 安全证明，也没有接触、夹持或携带物体覆盖。本次控制器自报的历史 contact_validated 元数据不是新接触实验。

HV-01 仍开放；至少一条既有成功路径回归及 Spatial 0 / Long 9 有界诊断仍待完成。未据此切换生产默认控制器或批准共享接口。

## 后续：实际 runtime receipt 派发（R0 第二十三批）

在第二十二批 exact receipt 修复后新增 `--through-agent-runtime` 并实际运行。同样是 Object 0/seed 0、独立 18766 服务、已有隔离 Mink 依赖，上移 2 cm；不是重用前一次环境/报告。

- 原始报告：`tmp/experiment-ready-r0-23.VtoZ9a/mink-runtime-canary.json`，passed=true。
- Simulator session `59929ff6-c7b8-4fc0-a42e-ade3e7c12a71` / handle `a1a5ddc5-c07`。
- Agent session `f8aca038-636c-4db2-a818-7bb5003041f7`。
- Agent 所选 ID 与实际 seed/controller receipt ID 均为 `4357a45e1ccb03add8b9`；policy=`ik_preview_seeded_cartesian_goal`。
- 实际终点 `[-0.14786779464165775, 0.0010050536943984494, 0.2797681073438621]`，欧氏误差 0.0019106585915797847 m，2/40 controller steps，target_reached。
- Preview 的 world/self/场景检查，以及 motion 的逐步配置检查完成，未报告碰撞；preview.path.checked=false，仍无连续 swept-path 保证。
- Runtime pipeline_status=executed，tool admission=1/1，remote_motion_calls=1。Close confirmed、remote.ok=true、cleanup_errors=[]；随后停止自建服务，退出码 0。没有操作已有服务/环境或发送外部感知数据。

复现使用上述启动命令，客户端增加 `--through-agent-runtime` 并选择新输出路径：

```bash
.venv/bin/python -m scripts.controller_capability_canary \
  --url http://127.0.0.1:18766/sse \
  --env-id openeta/libero_libero_object_task0-v0 --seed 0 \
  --expected-controller-id mink.robosuite_joint_velocity \
  --require-ik-seed --through-agent-runtime --max-steps 40 --timeout-s 45 \
  --output tmp/mink-runtime-canary-NEW-RUN.json
```

新增 helper `scripts/controller_runtime_probe.py` 用实际 preview 经生产 receipt compiler 建立 Host 证据，随后由 StaticPlannerBackend 给出唯一的 receipt-ID move。实际执行 ToolCallingPlanner → runtime → pipeline/gate → registry → proxy → 本地 simulator；私有 seed 由 memory resolver 产生，不由 planner 填写。额外 transport guard 核对目标、容差、步数、环境、碰撞标记及 seed，RPC 前预留唯一派发机会，超时不重试。调用方仍负责关闭。

范围限制：preview 由 Host 发起后摄入 memory，不是 planner 自主选择或通过 compiled-grasp/viewpoint proposal 发起。传入 runtime 的观察不含图像/实测 robot state，控制诊断依赖刚取得且期间无运动的 preview；没有视觉判断、外部模型或 Human VLM 参与。不能把本次结果称为完整模型闭环、复杂支路、抓放或完整任务验收。

运行时候选源码 SHA-256（不是运行时字节证明）：

```text
scripts/controller_capability_canary.py a9ef12e0d259599d298d9f3fdadba0c1aee07e26beb6af20cb6e1961cf7601c4
scripts/controller_runtime_probe.py 1a1c893f7b3dd22c7849070cb6b1b2dd33b2d18af8dfa6e59aab2ff4dcf19e05
agent/runtime/memory.py 5d60808a41a1d9a023656d979d9e58aa6116d7685141595b56bc2bd8b5ff285b
```
