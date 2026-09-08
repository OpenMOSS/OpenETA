# 本地验收矩阵 — 2026-09-07，R0 第二十五批

结论：当前候选完成扩大本地回归与失败归因，但不是全绿或完整任务验收。既有 authority 证据尚未绑定当前 catalog；当前依赖组合的 cuRobo MESH backend 初始化失败。生产 PRIMITIVE backend 的本地 GPU 检查通过。

个人分支 `dev/huaizezheng/harness-refactor-2026-09-05`，未提交工作树；按协作技能只读核对 RFC 第 5 节 revision 2158，没有更新共享评审或历史 hash。

## 结果

| 范围 / 解释器 | 结果 | 证据与限制 |
| --- | --- | --- |
| `tests/tools` / `.venv` | 587 passed，36 warnings，3.08 s | 本地模型/transport 夹具，不调用真实感知 |
| 合并 `tests` / `.venv` | 2313 passed，21 skipped，1 failed，36 warnings，63.46 s | 真实模型 integration 显式关闭；本机 HTTP 和 MCP SDK loopback 实际执行 |
| MuJoCo bounds、sim codecs、CGN synthetic import-order / LIBERO Python | 50 passed，2 deselected，4.67 s | 已有最小 Mink overlay；两项 trusted-local checkpoint 主动不选，没有加载部署权重 |
| cuRobo 原测试 / LIBERO Python + GPU | 8 passed，3 setup errors，3.31 s | 默认 MESH fixture 因 Warp API 缺失失败 |
| cuRobo 显式 backend 矩阵 / LIBERO Python + GPU | 11 passed，3 setup errors，3.36 s | 新增 PRIMITIVE 三项通过；保留 MESH 三项错误，不 skip/xfail |

各组重叠，不能相加计算总通过数。合并运行后只修改 cuRobo 测试 fixture，新增显式 backend 参数化；没有改生产代码或第三方依赖。compileall 与 git diff --check 通过。

本机原始 JUnit 报告位于 `tmp/experiment-ready-r0-25.Ef1XFp/`：`combined-tests.xml`、`libero-cpu-contracts.xml`、`curobo-world-contracts.xml`、`curobo-backend-matrix.xml`。这些 tmp 文件不保证随仓库分发。

## 失败归因

### 历史 authority 绑定未更新

唯一合并测试失败是 `tests/test_tool_contract_migration_status.py::test_reviewed_tool_contract_migration_is_complete_and_narrowly_authoritative`，在 `internal_conformant is True` 断言处失败。

当前 catalog SHA-256：`1316d67698201d4c670ba8955c3647145ad416cf2da0d5a0407362dbf340a9c3`。

以下三份历史证据仍绑定 `d957c3a33e5e41866d880468c3d7225c4cd1e261ec8a63ddb782e78ec0b797b4`：

- `docs/generated/tool-contract-authority-shadow-baseline.json`
- `docs/generated/tool-contract-authority-canary.json`
- `docs/generated/tool-contract-catalog-authority-canary.json`

对应 `empty_policy_authority_baseline_safe`、`estimate_depth_prior_authority_canary_safe`、`remaining_tool_authority_canary_safe` 为 false；structural_issues 为空，五份生成投影均 current。旧测试还固定了 revision 271 和历史 catalog hash，不能将其解释为当前候选已完成三人评审。

这是实际的验证/评审缺口，不是可直接删除的“坏测试”。需要针对当前候选重新形成适用的验证与评审证据，不能只改 JSON hash、更新期望值或伪造批准。

### cuRobo 默认 MESH 与 Warp 不兼容

三项错误共享 `robot_world` fixture：`test_curobo_accepts_an_empty_world_config_as_a_clear[mesh]`、`test_primitive_flag_stays_true_after_a_clear[mesh]`、`test_clear_then_repopulate_is_reusable[mesh]`。

当前 cuRobo 的 `WorldMeshCollision` 调用 `wp.torch.device_from_torch`；已安装 `warp-lang==1.16.0` 不提供 `warp.torch` 属性，子模块查找也不存在，初始化报 AttributeError。没有修改外部 cuRobo 源码、给 Warp 注入属性或安装/降级包。

生产 `sim/mcp_server/collision.py` 显式选择 `CollisionCheckerType.PRIMITIVE`。原测试直接调用 RobotWorldConfig 时未选 backend，落到 MESH 默认值，因而未独立验证生产组合。本轮将 fixture 参数化为 PRIMITIVE/MESH：既保留原 MESH 覆盖和失败，又新增生产 PRIMITIVE 的三项实际 GPU API 检查；原有 CollisionChecker 八项也通过。

该问题不否定本次 PRIMITIVE 验证，但在依赖兼容性修复并重测前，不能宣称 MESH backend 可用或据此切换生产 backend。也不把 box/primitive 检查外推为精确 mesh 或连续 swept-path 安全证明。

## 跳过项与告警

合并运行的 21 个 skip：

- 10 个真实模型/感知 integration：SAM3、AnyGrasp、AnyPlace、Contact-GraspNet、GraspGenX、MolmoPoint 的服务/工具测试，显式关闭 opt-in 开关。未发送数据或启动这些模型服务。
- 4 个 CGN/Mink import-order：base `.venv` 缺 torch；随后已有 LIBERO Python + 最小 Mink overlay 补测两个 synthetic 用例，两个部署权重用例未选。
- 7 个可选运行环境项：BEHAVIOR batched-env、cuRobo、MuJoCo bounds、RoboCasa legacy 四个模块因缺 torch/mujoco 跳过；BEHAVIOR worker 因未部署跳过；sim codecs 两项缺 MuJoCo。随后补测 MuJoCo/codec 和 cuRobo 的结果见矩阵；未宣称 BEHAVIOR/RoboCasa training 路径验收完成。

36 个告警来自 SAM3 point mask 校验中的两次 `Image.getdata()` 调用，属于 Pillow 弃用告警，不是本次分割结果失败；未用 warning filter 隐藏。

## 复现

合并测试需要允许其自建 loopback fixture。以下显式禁用真实推理测试，不修改进程外的环境配置；仍保留历史 authority 失败。新的 JUnit 输出应使用独立路径，避免覆盖旧证据。

```bash
env OPENETA_RUN_ANYGRASP_INTEGRATION=0 OPENETA_RUN_ANYPLACE_INTEGRATION=0 \
  OPENETA_RUN_CONTACT_GRASPNET_INTEGRATION=0 OPENETA_RUN_GRASPGENX_INTEGRATION=0 \
  OPENETA_RUN_GRASPGENX_TOOL_INTEGRATION=0 OPENETA_RUN_MOLMOPOINT_INTEGRATION=0 \
  OPENETA_RUN_MOLMOPOINT_TOOL_INTEGRATION=0 OPENETA_RUN_SAM3_INTEGRATION=0 \
  OPENETA_RUN_SAM3_TOOL_INTEGRATION=0 \
  .venv/bin/pytest -q tests --continue-on-collection-errors -ra

env PYTHONPATH=/tmp/openeta-mink-canary-min \
  sim/venvs/libero/bin/python3.10 -m pytest -q \
  tests/test_mujoco_geom_bounds.py tests/test_sim_control_codecs.py \
  tests/integration/test_contact_graspnet_import_order.py -k 'not trusted_local' -ra

sim/venvs/libero/bin/python3.10 -m pytest -q tests/test_curobo_world_integration.py -ra
```

Overlay 和解释器是本机已核实路径，不是便携安装步骤。所有测试进程已退出，自建 HTTP fixture 已关闭，GPU 测试只操作自己的内存模型；未操作现有实验环境。既有成功任务回归、Spatial 0/Long 9 诊断及感知发送授权仍开放。
