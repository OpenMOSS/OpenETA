# 实验恢复：独立服务 live 验证 — 2026-09-07

个人分支 `dev/huaizezheng/harness-refactor-2026-09-05` 的未提交候选。
本轮取得短程控制和观察链路证据，未完成 LIBERO 物体任务；不升级历史 authority/canary 或替代协作者评审。

## 环境与隔离

- 原服务 `8766`（PID 824969）和 Human VLM `8099`（PID 850459）均启动于 9 月 4 日，不能当作当前候选已部署的证据。本轮未重启、占用旧 handle 或回答旧请求；前后 `8099 /health` 均 pending=2。
- 独立候选模拟器使用 `18766`，最多一个 GPU 0 worker、600 s 外层寿命。独立控制台使用 `18099`，单请求等待上限 120 s、服务寿命 600 s。
- 服务解释器 `sim/venvs/libero/bin/python3.10` 指向 RAG 环境，Python 3.10.19，具备 torch/cuRobo/MCP；客户端与控制台使用 `.venv/bin/python`。实际 collision receipt 验证了检查器可用，而不只检查包能导入。
- 每个新环境都有明确 close acknowledgement；随后停止本轮独立服务，最终端口检查中 `18766/18099` 均已释放，原 `8766/8099` 仍在。没有全主机孤儿进程审计或 durable lease 证明。

本轮所有原始证据保留于 `tmp/experiment-ready-r0-11.xOsga4/`，这是本地实验目录，不是长期归档或已提交产物。

## 发现并修复 canary 的假通过

第一次用 `.venv` 启动候选模拟器时，短程控制成功、关闭确认，但服务输出 `cuRobo not installed. Collision checking disabled.`。旧版 canary 只检查请求是否启用碰撞检测，没有检查真实覆盖，报告 `passed=true`。

保留 `controller-canary.json` 原样；它仅证明控制器执行与清理，不作安全验收。之前采用同样弱检查的报告也不能据此被提升为碰撞覆盖证据。

修改 `scripts/controller_capability_canary.py`：

- preview 明确请求 endpoint collision 与 scene objects；必须有 checked、scene_objects_included、world_checked、self_checked 的严格 true，且 detected=false，才派发正常运动。
- 运动后保留并检查实际 collision receipt；缺失、不可用、覆盖不全或检测到碰撞不能整体通过。
- 显式关闭 collision 的诊断仍可运行，但不作为 acceptance pass。
- `path.checked=false` 和 `trajectory_checked=false` 原样保留。检查的是端点及执行后配置，不是扫掠体或连续路径安全；服务端 OSC 的 best-effort 检测并未因此变成 fail-closed 执行器。

## 严格短程 canary

报告：`controller-canary-strict-rag.json`。

| 项目 | 实测结果 |
| --- | --- |
| 环境 / seed | `openeta/libero_libero_object_task0-v0` / 0 |
| 新 simulator session / handle | `1ae1fd91-1f92-4294-ada9-354ad33c1754` / `e06c6750-936` |
| Controller / interface | `robosuite.osc_pose` / `normalized_cartesian_delta_pose` |
| 目标 / 预算 | 当前 EEF 上方 0.02 m；最多 40 controller steps；每调用 45 s |
| Preview | reachable；scene/world/self 检查完成，7 个障碍物，无碰撞 |
| 执行 | 6 steps，stop_reason=target_reached，receipt 一致 |
| 实际终点 xyz | `[-0.1491383091, 0.000000000125, 0.2783615726]` |
| 终点欧氏误差 | 0.0029946554 m |
| Motion collision | world/self checked=true，detected=false；trajectory_checked=false |
| 清理 / 结果 | close_state=closed，remote.ok=true，无 cleanup_errors；严格 canary passed=true |

这不证明 OSC 消费 IK joint seed，不证明复杂关节支路局部可执行，也不证明抓取、携带或放置成功。

## 主 Agent + Human VLM 模式观察 smoke

操作者为 Codex，经真实图像查看后逐次回答私有控制台，不是 StaticPlannerBackend 或预填 mock 响应；不是另一个真人参与实验。

- Agent session：`5466d563-585c-4adc-ac79-fdf60988964e`。
- Simulator session / handle：`a873c9f4-232f-4eca-b8c4-c2b8c8dd2d9d` / `7318f385-822`。
- 生产 `OpenAICompatiblePlannerBackend → ToolCallingPlanner → OpenEtaAgentRuntime → OpenEtaEpisodeRunner`；Host 新建/重置环境。只绑定 observe，实验 transport 只允许 create/reset/render/close，拒绝运动。最多 2 turns、1 个 executable tool admission、240 s，关闭自动 review。
- 第一请求 `1f8293d495c84ab38ee4d4401595e6bc`：2 张真实 agentview/wrist 图像；查看后提交 observe。
- 第二请求 `b62453d7d98146659355a18d31501445`：2 张图像，observe 审计 status=executed、success=true、diagnostics=[]；检查刷新图像后提交 talk，明确未完成 Object 0。
- 远端调用顺序：create_env → reset_env → render_env（observe）→ render_env（environment feedback）→ close_env。两个 render 是当前 EpisodeEnvironment 的实际行为，不伪装为仅一次远端调用；新图像可以内容相同，不据此推断世界未变化。
- 2 turns，1 次准入，episode elapsed=112.759 s，stop_reason=status_report。报告中的 terminated=true 是 episode 停止，不是 LIBERO 成功。清理 remote.ok=true、close_state=closed；私有控制台 pending=0 后退出。
- 报告 `observation-smoke/report.json`；完整记忆/图像在该目录，wire 请求/响应在 `manual-traces/<request_id>/`；实验驱动保留为 `observation_smoke.py`，通过 exist_ok=false 防止覆盖该次运行。

本轮还确认两个记录边界，后续需处理：

1. 控制台的实际等待约 43.845 s 和 57.166 s，而 episode `human_wait_s=0`、`human_assisted=false`。这些字段未覆盖 manual provider 模式，不得据此报告为无人协助或纯模型推理耗时；33385 tokens 也是两次 estimated，不是 provider 账单。需要显式模式/等待归因，接口候选需评审。
2. SimulatorMcpEpisodeEnvironment 使用仿真的 assigned task 替换 observation.task，控制台标题因此显示 Object 0 指令。实验的只读约束由 transport allowlist 强制实施，不能只依赖传入 task 中的诊断文字。应在复用通用诊断入口前明确运行约束与 benchmark task 的呈现边界。

未测试：SAM3/AnyGrasp/AnyPlace、嵌套 advisor、动作 receipt 引用解析、失败后恢复、已成功任务全程回归、Spatial 0 / Long 9。这些仍是待办，不能用本次 observe smoke 替代。

## 可复核启动命令

仅在端口空闲时运行；启动前核实所用解释器和 GPU。不要复用本次输出文件名。

```bash
timeout --signal=INT --kill-after=10s 600s env \
  OPENETA_WORKER_POOL_MAX=1 OPENETA_WORKER_GPUS=0 \
  OPENETA_LIBERO_CONTROLLER_PROFILE=osc_pose \
  sim/venvs/libero/bin/python3.10 -u -m sim.mcp_server \
  --host 127.0.0.1 --port 18766
```

另一终端：

```bash
.venv/bin/python -m scripts.controller_capability_canary \
  --url http://127.0.0.1:18766/sse \
  --env-id openeta/libero_libero_object_task0-v0 --seed 0 \
  --expected-controller-id robosuite.osc_pose \
  --max-steps 40 --timeout-s 45 \
  --output tmp/controller-canary-NEW-RUN.json
```

先检查 close acknowledgement，再停止自己启动的服务；600 s 外层 timeout 是兜底，不替代正常清理。

## 候选源码指纹

未提交工作树，单个 commit SHA 不足以标识候选。以下为本轮启动路径的文件 SHA-256，不宣称是运行时字节证明：

```text
scripts/controller_capability_canary.py 27cf8e5f888061350f610ddb73f089f785bc6646d1159c0d241167090263c107
tools/manual_vlm_proxy.py 57845ebea49a0970a32bffcdb3dff7e3e04f7af8428ebd8132b8af75514c6799
agent/backends/planner.py 7bb43d0ff1b0c9dcca75d7f06979473ad4da0899476e7213c806d576c9765629
agent/runtime/episode.py a69c6bbb61023caec7f5dfdc940e0645002aa9954d9e1ac1e1bcb5bbb6ee77db
sim/mcp_server/server.py 3300ecf16ee4b48e0fe8925eb76926c308a5cf2964b9630a254d6b82cf3ec366
sim/mcp_server/worker_mgr.py da0892b27e361fddb62c26a602bc8f5fb3d6d4b35d06791b74fdcbd83dd88a41
sim/bench_worker.py d9a53a6828ff7907867275d2007d18d16e2457438f318e54bc50a16241c24735
```
