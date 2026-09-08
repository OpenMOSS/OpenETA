# Harness 重构执行记录 — 2026-09-05

目标来源：[todolist](todolist.md)、[审查与 Human VLM 待办](harness-code-review-2026-09-05.md)、[主动感知候选设计](active-perception-design-options-2026-09-05.md)。目标进行中，不以文档登记或单测通过代替整个任务完成。

## 分支与基线

- 专用分支：`dev/huaizezheng/harness-refactor-2026-09-05`。
- 起点：`dev/huaizezheng/human-vlm-stability` / `fa56247` 的工作树。
- 创建分支时已有 26 个 tracked 文件差异（995 行新增、96 行删除），以及未跟踪的审查文档、诊断脚本、family canary 文档和四个 Human VLM 启动脚本。`.env.example` / `.mcp.example.json` 的删除也是既有改动。全部保留；不能将整个分支差异归为本轮重构成果。
- 未自动提交或推送；实现按小批次验证，待提交时须区分既有改动与本轮修改。
- 通过允许的只读网络访问取得共享 RFC revision 2158，已核对第 1–4 节及 5.4.1 的相关约束。旧审查记录中“当时网络不可用”仍是历史事实。本轮不替协作者批准新接口，不修改共享 main。

## 批次与验收边界

2026-09-06 范围调整：用户确认优先交付“能恢复进行真实任务实验”的版本。
后续顺序以 [实验恢复里程碑](experiment-ready-milestone-2026-09-06.md) 为准；下列五批保留为历史计划，不再要求完成全部基础设施工作后才恢复实验验证。

1. 宿主证据/Agent 笔记隔离、执行代次与取消边界；随后落实 Python 可终止 worker、文件/网络隔离与独立 stdout。独立进程本身不等于安全沙箱。
2. 成功判定、环境关闭/重试、代理并发、预算及崩溃恢复；成功语义与公共回执变化需明确契约评审。
3. I27 的可靠证据读取与上下文投影：二次截断、续读/grep、重复内容、环境 namespace、可视化、后端导入兼容性。
4. HV-01/03/04/05 与 issue #12 的几何/身份/资源工作：先解决启动与生命周期设计，再落公共接口；不机械照搬尚未定案的字段提案。
5. 人工子请求、封闭容器放置和恢复；分层实验后重测 Spatial 0 / Long 9，保留成功路径回归并统计成本。

任务状态由实际完成的不变量决定。大项只部分实现时保留未勾选；每批记录修改范围、验证命令、失败与未测范围。

## 批次 1A：笔记所有权与本地记忆提交

当前状态：已实现第一部分并通过本地回归；未完成整个审查第 1–3 项。

- `AgentMemory.facts` / `artifacts` 只保存宿主写入；`save_memory` 的 facts 写入既有 `agent_working_state`，artifact 引用写入独立 `agent_artifacts`。仅模型可见的只读查询/索引可合并展示，宿主同名项优先，几何和执行解析器不消费合并视图。
- Agent 删除操作只删除自身笔记/引用/skill note；不删除宿主事实、工具 artifact 或宿主 skill note。可信宿主的直接管理 API 保留原有能力。
- 新增内部持久化 `agent_artifacts.json`。恢复时迁移旧的 `source=save_memory` 镜像，不将 Agent 内容提升为宿主事实；已有快照若显示三个 epoch 之一被 Agent 覆盖，拒绝恢复并保留原数据，避免重置为零后重授权旧 receipt。普通历史笔记正常迁移。
- 本地记忆工具在实际读写时校验取消标记、session ID 和宿主生成的 session generation；与会话切换共享提交锁。同一 session ID 重开也使旧代次失效。覆盖 save/get/delete/compact 与 SAM3 select/reject。
- 局限：这不终止后台 handler、不撤回已发出的远端运动，也未解决 Python 原生模块的文件访问或跨进程隔离。完整执行所有权仍需后续 worker/transport 工作。
- 内部存储/所有权行为已改变，公开工具名称和请求 schema 未改。共享契约合入前应评审笔记查询兼容视图、旧快照迁移与宿主可信删除边界。

验证：

- `pytest -q tests/test_memory_ownership.py tests/test_agent_decision_context.py tests/test_planner_tool_registry.py tests/test_agent_contract_remediation.py tests/test_tool_contract_fixture_receipt.py --maxfail=4`：262 passed。
- 新测试覆盖真实 Planner → Pipeline → save_memory 的 epoch 覆写、伪造 artifact 隔离、四种 namespace 删除、只读查询别名隔离、持久化/旧快照迁移、受污染 epoch 的 fail-closed 恢复、同 ID 不同代次、取消后延迟 handler 向新会话写入。
- 尚未进行真实机器人/仿真任务、模型请求或性能对比。历史诊断脚本仍用于重现旧缺陷，其断言不应直接作为修复后验收；新增行为测试才是本批回归入口。

## 批次 1B：Contact-GraspNet 导入顺序兼容

I27-09 的实现与单测已补齐，真实依赖集成验证尚未完成，因此 TODO 保留未勾选。

- `_register_checkpoint_safe_globals` 显式解析 NumPy `multiarray` 叶模块，不再假设 `_core` 父模块必有 `multiarray` 属性；兼容新旧模块布局，并为两种 checkpoint 类型路径注册同一受限 scalar 类型。
- 缺失布局可回退；模块内部的依赖错误不被吞掉。始终使用 `torch.load(weights_only=True)`，未放开 pickle 加载。
- 新单测覆盖不完整父模块、新布局缺失或缺 scalar、双布局不可用和内部依赖异常。
- 新增独立解释器的 Mink-first / loader-first 集成测试，分别支持测试自建 checkpoint 与显式配置的可信本地 checkpoint；不自动下载权重、启动服务或跑推理。
- `pytest -q tests/tools/test_contact_graspnet_core.py tests/integration/test_contact_graspnet_import_order.py -rs`：28 passed、4 skipped。当前 `.venv` 为 NumPy 2.5.1，未安装 torch / mink；四项跳过明确由缺失 torch 触发，不能作为真实权重加载成功证据。

## 扩展回归

- `pytest -q tests --ignore=tests/integration --ignore=tests/tools --maxfail=8`：1104 passed、7 skipped、5 failed；四项为沙箱 socket 禁止，另一项为起点已存在的 `test_tool_contract_migration_status` 验证证据过期。未修改历史 canary hash 或放松断言。
- 允许本机临时端口后单独重跑 `tests/test_manual_vlm_proxy.py`：22 passed，确认四项 socket 失败不是本批回归。
- `pytest -q tests/tools`：535 passed、36 个已有 Pillow 弃用告警。
- 最后补充“resume 失败后禁止继续 act/记忆写入”，再执行 memory ownership、decision context、planner registry、contract remediation、fixture receipt、episode resource budgets、runtime assembly 七组测试：280 passed。
- 上述测试存在包含关系，不直接相加；先前全量数字不冒充最后一项改动之后的全量重跑。
- Python compileall 与 `git diff --check` 通过。

## 批次 1C：默认 Python worker 与取消回收

当前状态：默认 `python_exec` 从进程内 `exec` / 全局 stdout 重定向切换为一次性受限进程；已通过原生模块越界和 episode 级取消测试。未宣称修完审查第 2/3 项涉及的所有执行路径。

- 新增 `python_sandbox.py`（宿主有界流监督）、`python_sandbox_worker.py`（可信启动）、`python_sandbox_policy.py`（Linux Landlock/seccomp 和资源限额）。generated code 在策略安装成功后才执行，缺少内核能力或 libseccomp 时 fail closed。
- 允许解释器/包库读取、当前 session 读取与 sandbox 内写入；NumPy 等原生模块也受内核路径策略约束。禁止网络、fork/exec、新线程和向其他进程发送信号；不继承宿主凭据环境或无关 FD。
- CPU/地址空间/单文件/FD/输出流/JSON 大小有界；默认 worker wall budget 120 秒，请求只能缩短它。超时、取消和异常均 kill/reap；取消前置拒绝启动。JSON-only 输入不携带宿主对象引用。
- 默认 sandbox handler 通过宿主专用 cooperative-cancellation 标记在拥有执行权的 turn 线程中运行，避免 registry 提前放弃 daemon handler。新增测试在 worker 已回收、handler 清理仍受阻时确认 `wait_for_idle=false`，释放清理后才 idle。
- stdout/stderr 使用独立子进程 FD；并行 Python 与原生 `os.write` 不争用宿主 `sys.stdout`。宿主拒收子进程 artifact manifest，只扫描 workspace 正常文件，跳过 symlink/越界路径，并增量计算哈希。
- 兼容变更：无显式 roots 不再读取 cwd / 任意 `/tmp`；`extra_globals` 限 JSON；包不能创建线程/子进程。保持 per-call outside-sandbox 审批，不把审批路径冒充受限执行。
- 完整边界、错误示例、系统要求与未完成项见 [Python isolation](python-exec-isolation.md)。共享工具请求字段未改名；部署要求、timeout 行为与新增 diagnostic/output 信息在共享契约合入前仍需协作者评审。按 collaboration skill 保留个人分支和本地记录，未代替协作者批准或改写共享 RFC。

验证记录（各组有重叠，不直接相加）：

- 初步执行/取消/registry 四组：165 passed。
- 首次在 Codex 外层文件/网络限制之外运行原生隔离测试：15 passed，确认不是外层沙箱替 OpenETA 拦截；随后补充请求 timeout 校验测试。
- 最后收紧文件锁 syscall，防止通过只读文件加锁阻塞宿主；定向回归 `test_python_sandbox_isolation`、`test_python_exec_runtime`、`test_tool_contract_fixture_receipt`、`test_tool_contract_catalog`、`test_episode_resource_budgets`、`test_planner_tool_registry`：244 passed。
- 最终版本在 Codex 外层限制之外复测 `test_python_sandbox_isolation`：24 passed，含文件锁和请求 timeout 校验。
- 中间核心回归（请求 timeout 最后补丁前）：1120 passed、7 skipped、5 failed；失败仍为四项 socket 限制与既有契约迁移证据过期。允许本机端口重跑 `test_manual_vlm_proxy`：22 passed。
- 请求 timeout 补丁后、最后文件锁收紧前，再跑核心全量：1128 passed、7 skipped、相同 5 failed；没有修改历史迁移证据 hash 或将其失败忽略成通过。
- `tests/tools`：535 passed、36 个 Pillow 弃用告警。`compileall` 与 `git diff --check` 通过。
- 测试宿主为非 root Linux 6.8 x86_64、Landlock ABI 4；未验证 aarch64、其他部署内核、真实仿真/机器人或模型实验。

开放项：outside-sandbox 取消/有界捕获、总磁盘/inode 配额及宿主 artifact 后处理预算；其他 handler 与远端 worker 所有权；存储恢复与整体预算；I27-03/04/05 证据访问链路。主动感知与身份/部件接口仍待按候选设计验证后定案。目标保持进行中。

## 批次 2A：记忆快照与日志崩溃恢复

当前状态：实现审查第 9 项的快照原子发布、统一尾部恢复和安全证据失效处理；不将其扩展为“整个 Agent turn/远端执行事务已完成”。

- 本轮按 collaboration skill 只读复核共享 RFC revision 2158 的 memory 三层边界、不可变 durable history 与原子 working snapshot 约束；未修改共享文档。新增内部存储格式和恢复语义是个人分支候选，合入前需协作者复核。
- `memory_snapshot.py` 将五类 memory map 与 compact summary 放入一个带 session ID、generation、journal cursors 和 SHA-256 的 `working/snapshot.json`。唯一临时文件 → fsync → replace → fsync directory；旧或新代次完整可见。旧 namespace JSON 仅作兼容导出，混合导出绝不参与新格式恢复。
- 首次迁移写 `.snapshot-managed`；快照损坏/丢失不回退旧文件。兼容导出失败保留新快照并明确报告。per-session thread/flock 与 generation 比较防止另一 store 的旧快照覆盖新代次；不是同一 session 多个活跃 runtime 的执行租约。
- `memory_journal.py` 统一 trace/conversation：新记录有序号及 checksum，旧无序号 prefix 保留；流式校验、按 limit 保留尾部。只允许未换行的末尾坏 JSON/UTF-8 作为 torn tail，完整坏行、中间损坏、序号/校验和错误一律 fail closed。
- 查询只报告，不改写 journal；恢复/后续追加先将坏尾部精确 bytes、offset/hash 保存到 recovery 目录，再发布持久恢复标记，最后截断坏后缀。有效历史原字节不变。新目录、文件和 index replacement 均执行相应同步；session index 损坏不再被静默重建为空。
- snapshot cursor 能发现完整记录后缀丢失；日志落后于 checkpoint 时禁止恢复/覆盖，领先时报告未 checkpoint 的历史。尾部修复标记跨重启保留，避免修复后、epoch 保存前再崩溃导致失效处理丢失。
- 恢复历史不完整或存在未 checkpoint 记录时，推进 robot/object evidence epoch 并记录 `session_journal_recovery`；不是合成“物体实测发生变化”的环境事件。保留历史证据，但旧 IK seed 不再能授权动作；Agent 需重新取得新鲜环境证据。
- 全部故障注入只操作临时夹具，没有修剪或迁移既有 Human VLM 实验 trace。详细格式、兼容性与操作边界见 [memory store recovery](memory-store-recovery.md)。

验证记录（有包含关系，不相加）：

- 初步既有 memory ownership / planner registry / conversation：161 passed。
- 新恢复测试包含实际 subprocess 在 snapshot replace 前/后 `os._exit`、namespace export 半写失败、双 store 冲突、无锁读者、并行 journal writer、legacy 迁移、UTF-8 半尾、完整坏行、sequence/checksum/cursor 损坏、持久恢复标记，以及真实 `ik_execution_gate_error` / `resolve_ik_execution_seed` 恢复前后的授权变化。
- 在最后补充“已坏 recovery fence 不得被新修复覆盖”之前，memory recovery / ownership / conversation / contract remediation / planner registry / CLI 六组：298 passed。
- 同期核心全量：1159 passed、7 skipped、5 failed；仍为四项本机 socket 限制和基线已有契约迁移证据过期，未修改 hash 或忽略该断言。允许临时本机端口后，manual VLM proxy：22 passed。
- 最后补充恢复标记保护、区分证据失效与实测物体变化后，再跑核心全量：1160 passed、7 skipped、相同 5 failed。四项端口测试已单独放开复测通过，唯一已确认的代码/证据失败仍是原有契约迁移验收项。
- 最终独立复测 `tests/test_memory_store_recovery.py`：31 passed。
- `tests/tools`：535 passed、36 个已有 Pillow 弃用告警。compileall / diff whitespace 检查通过。

未验证断电/设备损坏、非 POSIX 存储、真实仿真任务和性能 canary；未实现跨 journal/snapshot/index/remote action 的事务或完整事件重放，也不能为旧布局补造原本不存在的一致性证明。后续继续成功判定、生命周期清理与预算，以及 I27/HV 几何与证据链路。Goal 保持 active。

## 批次 2B：统一成功证据与离线结果

当前状态：审查第 4 项的通用正奖励误判已修复；不宣称所有后端 checker 策略和真实任务已验证。

- `success_evidence.py` 为并行分类、experiment/eval/visual-history、calibration batch、task playbook 和 RoboCasa batch 转换提供共同判据。通用正奖励不再算 objective success；已知 LIBERO / 注册的 RoboCasa direct adapter 使用严格的二值终止回执，其他环境需要明确宿主成功证据。
- 核对本地 RoboCasa direct 与 vector training wrapper 的不同语义；只对前者注册的 env ID 应用二值规则，没有将全部 RoboCasa 或所有 `official_reward` 泛化为二值奖励。
- 校验 execution/session、schema、有限奖励及 terminal 字段一致性；旧 execution 的回执不能因 session 相同重新授权。资源失败、截断和明确失败完成覆盖此前成功；中间 false 证据可被后续成功更新。
- planner 完成检查与 memory transition ledger 使用共同语义；显式报告失败无需先取得成功奖励，未完成的普通环境反馈保留 UNKNOWN。非 official 的 runtime `task_complete` 兼容保留，但不作为实验/训练 objective evidence。
- 真实 host receipt 测试发现 episode 序列化按八键压缩会丢失 schema/reward/terminal 字段；改为保留成功判定必需字段和标记，验证 live / serialized 结果一致。RoboCasa 汇总不再宽松接受缺关键字段的旧投影；不改写历史实验或补造其证据。
- 原来只用 `reward=1` 的成功测试夹具补上真正的 host checker 或 execution-bound receipt；新增 reward-only、NaN/inf/bool/string、跨会话/跨执行、矛盾 flags、失败覆盖、旧压缩回执和候选拒绝反例，没有放松成功断言。
- 共享 RFC revision 2158 的 batch v2 / assistance / resource-failure 示例已只读复核。按 collaboration skill 保留个人分支、本地记录与评审边界；成功语义和序列化兼容变化需协作者评审，未修改共享 RFC。

验证记录（各组有包含关系，不相加）：

- 最后补充显式失败报告、未完成 ledger UNKNOWN 后，success evidence / tool feedback / parallel / experiments / task playbooks / planner / self-improvement / RoboCasa / calibration / evaluation / decision context：309 passed。
- 此前核心全量：1220 passed、7 skipped、5 failed；仍为四项沙箱 socket 限制与基线契约迁移证据过期。允许本机临时端口后 manual VLM proxy：22 passed。
- `tests/tools`：535 passed、36 个已有 Pillow 弃用告警。compileall 与 diff whitespace 检查通过。
- 未跑真实仿真、机器人、模型或重算历史成功率。通用后端 policy 注册、旧证据恢复/重算、专用旧 canary 脚本的跨后端语义审查仍开放。详见 [success evidence](success-evidence.md)。

下一批继续环境关闭失败与重试所有权；其后还有取消、预算、I27 证据访问和 HV 几何/主动感知需求，重构尚未完成。

## 批次 2C：环境关闭确认与重试所有权

当前状态：审查第 6 项的客户端提前丢句柄、服务端忽略业务失败，以及 worker 吞掉 close 异常已修复；持久化 orphan 清理与完整远端执行租约仍未完成。

- 新增 `adapter/environment_lifecycle.py` 的明确关闭 ACK 检查；正向 ACK 与错误/pending 矛盾时拒绝。`official_reward` 等成功语义与资源关闭语义保持分离。
- 客户端 episode / Agent close tool 保留句柄直到确认；并发关闭返回 pending 而非伪成功，close_failed 状态禁止复用。启动 retry 不能在旧环境关闭失败后丢弃身份、创建替代环境。
- 服务端 `env_lifecycle.py` 统一显式 close 与 TTL 清理；先远端确认，再释放 worker，再清理 cache/checker，最后删除句柄。记录已完成阶段，失败重试不重复 remote DELETE 或已完成的引用释放。资源列表增加 lifecycle_state。
- 服务端和 worker 的 per-env lock 使用 weak ownership；有等待者时不能删除锁后生成第二把锁绕过串行约束。TTL 清理保留失败句柄、缓存与后续 sweep 登记，阻塞工作不占用 asyncio 主循环。
- worker 的 env.close 异常不再吞掉；成功或明确不存在才清除环境/缓存。异步 close 将整个锁定关闭段交给已有 simulator executor。BEHAVIOR 的 worker-retire 回执保留，但 manager 必须确认进程退出才退休引用。
- 协作技能要求的个人分支与评审边界保持不变；新增 ACK/absence/lifecycle 字段与错误语义是待评审候选，未更新共享 RFC、未提交或推送。

验证记录（有包含关系，不相加）：

- 最后补充 BEHAVIOR 退出确认和启动失败回归后，environment lifecycle / simulator proxy / control codecs / terminal result / tool feedback：151 passed、3 skipped。
- 在外层沙箱之外执行完整 `tests/test_environment_lifecycle.py`：23 passed。仅假环境/假进程，无实际 worker 终止。
- 原生异步测试首次在外层沙箱中阻塞，已中断诊断进程；确认 socketpair 创建允许、send 返回 EPERM，导致跨线程 self-pipe 无法唤醒 asyncio。增加能力检查和明确 skip，并用外层沙箱之外的实测作为并发验证证据，没有删除或弱化核心并发断言。
- BEHAVIOR 最后补丁前核心全量：1242 passed、8 skipped、5 failed；失败仍为四项本机端口限制与既有契约迁移证据过期。新增的第八项 skip 是上述 self-pipe 能力限制。
- 最终核心全量：1244 passed、8 skipped、相同 5 failed。四项本机端口测试已在外层沙箱之外单独验证；唯一已确认的代码/验收证据失败仍为既有契约迁移证据过期，未修改旧 canary hash 冒充当前验收。
- 最终 `tests/tools`：535 passed、36 个已有 Pillow 弃用告警。compileall 与 diff whitespace 检查通过。
- 收尾再次在允许本机临时端口的环境复测 `tests/test_manual_vlm_proxy.py`：22 passed。

详细兼容变化和未覆盖边界见 [environment cleanup](environment-cleanup.md)。本轮继续请求已推进两批代码，未将整个 TODO 重构标记完成；真实仿真、持久恢复/租约和 I27/HV 主线仍有后续工作。

## 实验恢复 R0 第一批 — 2026-09-06

用户确认按“能恢复进行真实任务实验”交付；新增 [范围、验收标准与短程命令](experiment-ready-milestone-2026-09-06.md)，不再以清空 TODO 或 Spatial 0 / Long 9 稳定全程成功作为本轮完成条件。按 OpenETA 协作技能只读复核 RFC revision 2158 的 batch 输出约束，保留个人分支/已有修改，未改共享 RFC、提交或推送。

代码：

- `agent/tools/sim_mcp.py` 的零步容差提示现在要求整数零（不接受 bool/float）、`target_reached` 停止原因和不矛盾的 controller receipt；`response_artifacts.py` 不再把 bool/负数当作控制步数。删除从末端 xyz 不变推导出的 `motion_outcome=no_state_change` 与“机器人/视角绝对未变”文案，仅表述 receipt 报告无控制器驱动运动；重新使用感知证据仍须满足有效性和新鲜度要求。
- 这是 HV-02 的局部反馈修复，不是完整的 `reached_before_execution` / `controller_steps` / `world_state_changed` 接口。未新增这些字段，也未改变状态 epoch、碰撞或 IK 授权规则；Agent 可见反馈兼容性仍须合入评审。
- `scripts/controller_capability_canary.py` 在任何创建前校验有效预算和非空操作目标；控制器不匹配则在 reset/move 前退出，preview 非 reachable 则不 move。检查 receipt controller/interface/executor、整数步数和实际终点容差，清理未确认不再给出整体通过。异常生成失败报告，已知 handle 始终尝试关闭。
- 新增 27 项 canary 夹具测试和 9 项零步错误/冲突 receipt 测试。两个旧契约夹具缺少停止原因，已补为生产结果形状；没有放宽覆盖断言、替换旧 canary hash 或重写历史 session。

真实服务诊断（非模型/人工任务回归）：

- 外层沙箱内 localhost 网络访问受限；获准后只读确认 Human VLM `8099/health` HTTP 200，仿真服务 `8766/health` HTTP 404（该服务无此健康路由，不能据此判离线）。随后实际 MCP 调用成功。
- 先按旧成功记录中的 Mink 预期探测，当前服务却声明 `robosuite.osc_pose` / `OSC_POSE`。在任何 reset/motion 前拒绝，确认关闭新环境 `0620f1e9-a09`，session `4c64bc13-f45d-4436-9601-827c1cdfa8e7`。证据：[controller mismatch](../tmp/experiment-ready-controller-canary-20260906.json)。没有自动切换控制器或重启服务。
- 改为**验证当前 OSC 配置本身**，独立新建 Object 0、seed 0，2 cm 向上目标、3 mm 每轴容差、40 步上限、碰撞检查开启。初次诊断及补完独立终点校验后的最终复测均通过。
- 最终 session `ce00b837-779f-4180-914c-040192698426` / handle `9d3d1db8-e3c`：6 步，`stop_reason=target_reached`；目标 xyz `[-0.148464661, 0, 0.281279476]`，实测 `[-0.149138309, 0, 0.278361573]`，欧氏位置误差 `0.0029946553654484967 m`。控制器回执一致、独立每轴残差检查通过、`cleanup.ok=true`。证据：[final OSC canary](../tmp/experiment-ready-osc-final-canary-20260906.json)；初次记录为 [initial OSC canary](../tmp/experiment-ready-osc-canary-20260906.json)。三次新建环境均关闭确认，没有操作已有实验 handle。
- 这些报告验证的是**当前运行服务**，未核实服务载入的代码 revision，不能冒充部署了工作树全部修改；更不能证明复杂转腕 IK/OSC 对齐、Mink 路径、Agent receipt gate、感知服务或完整任务已经通过。

回归：

- 扩展定向八组（最后增加 5 项终点错误测试前）：280 passed、3 skipped。
- 最终核心全量 `pytest -q tests --ignore=tests/integration --ignore=tests/tools --maxfail=8`：1280 passed、8 skipped、5 failed，49.16 秒。四项本机 socket 限制，一项既有契约迁移验证证据过期。
- 获准本机临时端口后复测 `tests/test_manual_vlm_proxy.py`：22 passed，2.42 秒。未把此结果冒充真实人工 advisor 流程验证。
- 修改文件 compileall 与 `git diff --check` 通过。本批未再次执行 `tests/tools`；上一批记录不作为本批全量结果。

交付状态：R0 第一批局部改动和短程仿真诊断完成，实验恢复里程碑仍未验收。下一批优先审查 HV-01 的当前控制器执行契约、HV-06 人工子请求可见性和 I27 关键证据投影，再进入有限人工闭环 smoke 与成功任务回归；主动感知启动/身份/部件公共接口仍待讨论，不因本轮缩小交付范围而自动定案。

## 实验恢复 R0 第二批 — 2026-09-06

用户请求恢复 goal 持续推进。本轮读取到 goal 状态仍为 `blocked`；可用 goal 工具不支持将其改回运行态，未将未完成目标标记完成或创建替代目标。已告知需由客户端恢复自动续跑，本轮继续完成了以下局部实现。该运行态问题不等于代码工作已无安全下一步。

### HV-06：人工抓取 advisor 可发现性

- `handlers.py` 从宿主工具上下文取父 Agent session，仅传入 advisor 的私有 bundle 深拷贝；不写入公开 bundle/几何 artifact，也避免 advisor 原地修改嵌套候选影响已发布 bundle。
- `BackendGraspPoseAdvisor` 为每次调用生成独立 child session，在隔离 provider context 中附加 `request_lineage`（父 session、子 session、来源 tool）。不继承主 planner 对话、memory 或工具权限，不激活推荐候选；既有自定义 advisor 的方法签名不变。
- OpenETA console adapter 识别实际使用的 `grasp_selection_advice.v1` schema。队列 API 附加 parent session 和当前等待原因，前端将子请求归入父会话、标注“等待人工 advisor 响应”和全局待响应数；终止等待后清除标签。仍保留独立 child identity/turn/audit，不把父子历史合并。
- 没有 lineage 的旧请求仍显示等待，不依据任务文本、候选数据或时间猜测其父会话；其他隔离 advisor 尚未接入。只有父会话关联，不声称已精确关联某个父 provider request/action。关联字段只用于显示/追踪，不授予执行权。
- 测试覆盖父/子/另一会话的独立取消、畸形/错角色 lineage、旧请求兼容、真实 Node.js 队列渲染和 HTML escaping；新增真实本机 HTTP 集成，从生产 advisor/provider 发出请求，经队列发现，再用测试回答解除阻塞。它是可重复的人工响应夹具，**不是**真实 Human VLM 任务实验。

### HV-01：区分端点可行与执行已验证

- 确认本地已有 Host seed resolver → 私有 move metadata → Mink worker 校验/消费路径；OSC outer loop 没有消费该 seed。没有把整套 seed 功能当作尚未实现，也没有擅自切换控制器。
- 对可行/碰撞延后 IK 预览，新增既有 diagnostics 容器内的 `ik_controller_execution_unverified` warning，并放到简短反馈前部：端点 IK 不等于局部执行能力已验证；分别说明声明 OSC、声明 Mink、未知或不一致 executor 的限制。Mink 支持该路径不等于本次 seed 已消费或实际运动必收敛。
- 不修改 IK classification 枚举、epoch/receipt 授权、碰撞门禁或执行策略；execution ref 文案从必然到达改为尝试到达。warning 非 candidate rejection，保留可执行引用与实际回执/残差检查要求。
- 尚未实现控制器局部可执行性验证/双结论新契约，没有用这次反馈修复宣称复杂转腕与 Spatial 0 / Long 9 已解决。

最终验证（组间重叠，不相加）：

- `pytest -q tests/tools tests/test_manual_vlm_console.py tests/test_simulator_mcp_proxy.py tests/test_tool_contract_fixture_receipt.py tests/test_agent_decision_context.py`：702 passed，36 个已有 Pillow 告警。
- 获准使用测试自己的临时 localhost 端口后，`pytest -q tests/test_manual_vlm_proxy.py`：31 passed，3.20 秒，包含上述 HTTP 等待/响应闭环。
- 核心全量 `pytest -q tests --ignore=tests/integration --ignore=tests/tools --maxfail=8`：1294 passed、8 skipped、6 failed，49.49 秒。五项是外层 socket 限制（包含新增的 HTTP 测试），已由上述 31 项实际 HTTP 回归验证；另一项仍是既有契约迁移证据过期。未改旧验证 hash 或放宽断言。
- Python compileall、Node `--check` 与 `git diff --check` 通过。

按 OpenETA 协作技能只读核对共享 RFC revision 2158 的 advisor 只读/隔离和协作评审约束。新增 lineage、队列字段和反馈兼容性已在 [控制台文档](manual-vlm-harness-debugger.md#isolated-grasp-advisor-correlation-2026-09-06-candidate) 标记需三人评审；未更新共享 RFC、提交或推送。未重启现有 console/simulator，也未回答任何已有人工请求；新功能尚未在正在运行的服务上部署验收。后续仍需可协调的服务更新、实际人工闭环 smoke、感知恢复设计与任务回归。

## 实验恢复 R0 第三批 — 2026-09-06

本轮启动时 goal 已为 `active`，上一轮属于代码与验证均有产出的有效进展；本轮继续推进 I27-03/04/05，没有修改目标完成标准或将完整 TODO 标记完成。

实现与新证据：

- `grep_text_artifact` 改为匹配行分页，围绕真正的 regex span 输出最多 500 字符片段；分别报告片段/匹配裁剪、位置、匹配行数和下一游标。采用 N+1 实际命中判断截断，首次扫描全量完成时才给出已知总数；游标绑定文件及 pattern/ignore_case。
- 新增 `read_text_artifact` 与现有 Python facade 的 `artifacts.read_text_page`，支持 Unicode/归一化换行后的字符偏移、行号、列号、EOF lookahead 和超长单行续读；grep 命中携带可直接传给文本页的游标。保持 session-owned roots，旧 `read_text` 返回值不变。
- 真实临时文件测试发现 metadata-only 版本检查不足：同长度快速改写可以保留相同时间戳。版本指纹因此包含全文 SHA-256，并在读前/读后校验；普通文件 nonblocking 打开后检查 FD，拒绝 FIFO。每页两次内容扫描与前缀定位的 I/O 成本如实记录，不宣称已完成大文件读取性能优化。
- `_bounded_decision_value` 保留标量叶节点，字典投影附加 `__context_projection__`，记录源完整性标志、模型可见不完整、结构/字符串省略位置和计数；不修改源 `truncated` 或 episode 终止语义。固定大小文本游标作为原子引用保留，避免更深压缩只剩游标字段名。最后收紧 `source_truncated` 类型，只允许 bool/None，防止把源中的非法大对象绕过投影预算重新注入。
- 端到端测试发现 provider 的视觉路径清理会递归删除所有 `path`，连文本证据位置也一起删除。现在保留明确类型的 JSON/text 以及新文本页/搜索 schema 的路径；仍隐藏普通 camera transport path。省略记录使用 `json_pointer`，不混用文件路径字段。
- 测试链路：真实受限 Python worker 搜索 100 个长行命中 → memory action → 主 planner → 生产 provider prompt formatter → 捕获最终请求 body → 使用其中的游标回读命中。使用假 provider transport，没有发真实模型请求。测试过程中纠正了一个“请求 8 字符却期待 9 字符”的夹具断言；没有放松内容/游标校验。

验证（组间重叠，不相加）：

- 四组文本/Python/decision context/planner registry：200 passed。
- `tests/tools` 加文本/Python/decision context：614 passed，36 个已有 Pillow 告警。
- 核心全量：1317 passed、8 skipped、6 failed，50.87 秒。五项是外层 socket 限制，另一项仍是既有契约迁移验证证据过期。
- 允许测试自己的临时本机端口后，Human VLM proxy：31 passed，3.06 秒。
- 最后 `source_truncated` 类型限缩之后，文本/Python/decision context 定向复测：77 passed，7.99 秒；以上全量数字不是该最后小补丁后的全量重跑。
- 修改文件 compileall 与 `git diff --check` 通过。未更改旧 canary hash、历史 trace 或正在运行的实验服务。

按 OpenETA 协作技能只读核对 RFC revision 2158 的 ToolResult 与未知输出字段/完整 artifact 约束；新增 schema/游标/投影字段标记为个人分支候选、需三人评审。详细语义、示例、资源边界和未完成项见 [artifact text reading](artifact-text-reading.md)。仍需原生非 Python 证据工具、图片显式打开到模型的链路、其他压缩路径/列表根完整性、全部关键 ID 优先级与真实任务验收；goal 保持 active。

## 实验恢复 R0 第四批 — 2026-09-06

Goal 确认为 `active`；继续按“恢复真实任务实验”的近期范围推进 HV-05，未清空完整 TODO 或调整完成标准。按 OpenETA 协作技能只读核对 RFC revision 2158 的三人契约评审约束；精准查询 `target_geometry_family` 未命中，不能据此声称共享 RFC 已批准此次变更。

实现与边界：

- 新增轻量 `agent/tools/grasp_types.py`，把既有 10 个几何类型收为不可变共享来源。SAM3 选择的 runtime gate、canonical request enum 和 registry guidance 共同引用；保留 `memory.GRASP_GEOMETRY_FAMILIES` 的导入兼容性。
- schema 明示空字符串/省略表示未指定；direct memory API 保留历史 trim/lower 行为。非法字符串及非字符串报错前不消耗 pending detection，避免重新分割才能恢复。
- 代码核对发现抓取编译器原本有意允许扩展字符串，因此不将其改成封闭枚举。其 schema 明示扩展语义并从共享来源生成 examples；实际几何测试覆盖 `apple`、`mug`、`mug_handle`、`custom:tool` 的通用 fallback，既有策略与安全检查不变。
- 没有新增 mug/handle 到选择枚举，也没有将几何提示升级为物体身份或部件证据。HV-04/05 的部件类型/身份及选择扩展设计仍待评审。接口示例和限制见 [geometry hints](grasp-geometry-types.md)。
- 使用仓库生成器更新两份工具契约文档，并增加生成结果与当前 catalog 精确一致的测试；未改历史 review/canary hash、生产 authority allowlist 或共享文档。catalog 继承的 maturity 标签不代表当前候选已批准。

验证：

- `pytest -q tests/test_grasp_geometry_contract.py tests/test_tool_contract_catalog.py tests/test_agent_contract_remediation.py tests/test_agent_decision_context.py tests/test_planner_tool_registry.py tests/tools`：809 passed，36 个已有 Pillow 告警，5.84 秒。
- 新增 25 项类型契约测试；原 compiler fallback 用例扩展 3 个实例。第一次定向运行发现测试误用了投影字段名 `request_schema`，已改为实际 Agent 投影的 `parameters`，再次及扩大回归通过。
- `tests/test_tool_contract_migration_status.py`：1 failed，仍未满足历史 authority baseline/canary。只读审计确认 `empty_policy_authority_baseline_safe`、`estimate_depth_prior_authority_canary_safe`、`remaining_tool_authority_canary_safe` 为 false；catalog JSON、Markdown、readiness、host resolver、promotion campaign 五份生成投影均 current。未伪造新验收证据以消除此失败。
- 修改文件 compileall 与 `git diff --check` 通过。本批未重跑完整核心测试，没有新增 live 仿真或 Human VLM 成功证据，未重启现有服务、提交或推送。

HV-05 仅标记 partial。后续继续处理执行/夹持证据等可验证缺陷；主动感知与部件接口选择、真实闭环 smoke 和任务回归仍是近期交付的开放项。

## 实验恢复 R0 第五批 — 2026-09-06

上一轮有实际代码与回归产出，属于有效进展。本轮继续 HV-08，没有修改 goal 的完整范围。按 OpenETA 协作技能只读核对 RFC revision 2158 已批准的 attachment probe 短 ID handoff 与匹配 IK receipt 边界；未外推为本轮新增字段/语义已批准。

事实核对：

- 当前 `_capture_gripper_command_state` 只处理 gripper 调用，普通 move 不覆盖关闭命令。检查提供的五条代表 trace 的 prepare-probe 调用，没有定位到所述“最近命令不是 close”的同一拒绝；Goal 0 有 carried proxy 不适用的旧拒绝，Long 9 两次 prepare 成功。未将报告泛化为所有 session 已复现或已修复。
- 确认两处代码缺陷：旧对账逻辑将 coarse `open` 与 aperture 用 OR 合并，会把部分张开/布尔标志矛盾误判为已打开；观测对账后的打开绕过正常 reopen 的 contact/attachment 失效路径。同时仅凭 aperture 对账会设置 `latched=true`，并不能由该观测证明。

修改：

- 新增 `agent/tools/gripper_evidence.py`，统一 aperture 优先级及 modern actuation receipt 校验。有限且归一化的 aperture 优先于 coarse bool，非法值不回退为更强证据；保留缺失 aperture 时的旧 bool-only 兼容。
- Memory 保留 modern actuation receipt，检查 schema、命令一致性、明确 latch 和正整数 settling steps；存在但畸形/矛盾的 receipt 不当作缺失，设置 unconfirmed latch/原因，撤销旧 proxy/attachment 及矛盾 contact evidence。
- 观测对账仅说明位置条件，不再制造 latch 或沿用旧 proxy/PASS。统一 setter 的 reopen 失效路径；仅缺失 latch 不凭空断言 EEF contact 已移动。普通 arm move 保留未被反证的关闭证据。
- Fresh fully-open telemetry 将旧 attachment PASS 降为 UNKNOWN，保留真实命令历史，不猜物体去向；既有 aperture-collapse 失效仍保留。
- Probe 检查 supplied receipt 和显式 latch 状态，拒绝 malformed aperture；原 tentative proxy / articulated reached-contact fallback、目标绑定、短 ID、IK 与碰撞门禁未绕过。无 modern receipt 的旧确认路径仍兼容，不能将其称为现代回执验证。
- 新增命令 → memory → unknown action → 新 observation 的状态序列测试，以及真实 probe 准备、冲突 receipt、malformed aperture、普通运动证据保留等回归。初次新增观测夹具遗漏 `task/cameras` 必填字段，已修正并复测。

最终验证（组间有重叠，不相加）：

- 定向扩展组（gripper/probe、contract remediation、planner registry、decision context、memory recovery、sim proxy、catalog、全部 tools）：974 passed，36 个已有 Pillow 告警，9.64 秒。
- 核心 `pytest -q tests --ignore=tests/integration --ignore=tests/tools --ignore=tests/test_manual_vlm_proxy.py --ignore=tests/test_tool_contract_migration_status.py`：1364 passed、8 skipped，50.18 秒。它不是无排除的全仓测试。
- 单独获准使用测试临时本机端口：`tests/test_manual_vlm_proxy.py` 31 passed，3.06 秒。
- 单独 `tests/test_tool_contract_migration_status.py` 仍 1 failed：`internal_conformant` 不满足历史迁移证据。未改旧 hash、authority 或历史 trace。
- 修改文件 compileall、`git diff --check` 通过。

字段示例、兼容边界与未完成项见 [gripper evidence](gripper-evidence.md)，新增保留字段、`latched` 语义、invalidation reason 与 gate 兼容变化均标记个人分支候选、待三人评审。没有 live 仿真/中断恢复验收，没有重启用户服务、写共享 RFC、提交或推送。

HV-08 保持 partial。特别是原有 reconciliation 的 `completed` 并不证明远端控制器已退出：本轮没有实现 operation completion/lease，也不能授权超时后并发重发动作。还需完整 freshness/consumer 审计、真实恢复回归，以及 HV-03/04 设计选择和真实任务实验验收；goal 保持 active。

## 实验恢复 R0 第六批 — 2026-09-06

上一轮为有效进展。本轮推进 HV-03 的多方位几何部分，并通过异步问题明确提出前期入口的设计需求：建议 point/ROI 弱定位与已确认身份锚点分离，可靠分割后再显式关联。尚未收到用户决定，未新增 localization resource 或改变 identity-anchor 语义。Goal 完整范围保持不变。

修改与证据：

- 现有 `propose_wrist_viewpoints` 从固定世界 X 偏移扩展为目标周围四方位、40°/65° 两俯仰，共八个独立候选。Host 选择数量，公开参数不增加 count/standoff。保留旧 standoff 的世界 Z 高度含义，新增完整 XYZ offset 和真实径向距离，不把字段静默改成另一种量。
- 采样基准随当前目标到相机的方位变化，垂直/重合退化时使用 camera X/world X；保留完整 EEF↔camera 安装变换。候选提供几何角度差，目标可见性、遮挡和成像尺寸仍 unverified/unknown/null，没有伪造遮挡改善证据。
- 没有工作域初筛，输出明确标记；每个候选仍要求 workspace/IK/collision checks。结果不是安全授权，不规定扫描顺序。输入仍要求 compiled grasp，所以没有把这一批宣称为 pre-SAM3 恢复能力。
- 修复 proposal hash 漏掉 EEF/mount 旋转和实际候选几何的问题，防止旋转改变后出现同 ID 不同 pose；仅 packet 刷新且几何不变仍复用 ID。Hash 只标识内容，不代替全部目标/标定 freshness 审计。
- 输出契约声明八候选上限；handoff 改为推荐已存在的 `viewpoint_proposal_id + candidate_id` IK 分支，不再要求复制矩阵。之后仍须匹配 IK receipt 才能运动。
- 新增 17 个测试实例：方位/俯仰/距离覆盖、刚体安装还原、正交旋转/光轴、世界 yaw 等变性、rotation-only hash 变化、packet-only 稳定、退化视点、非法高度、Agent 无权填数量、实际生成候选经 memory 短 ID 解析及真实 action 记录推进 epoch 后的失效。测试初期纠正了 resolver envelope 层次与 epoch fact 夹具形状，最终使用实际 move action 推进 epoch，而非篡改无效形状的 fact。

验证：

- `pytest -q tests/tools tests/test_agent_contract_remediation.py tests/test_tool_contract_catalog.py tests/test_grasp_geometry_contract.py tests/test_planner_tool_registry.py tests/test_agent_decision_context.py`：826 passed，36 个已有 Pillow 告警，5.87 秒。
- 两份 generated 工具契约由仓库生成器更新，生成结果一致性测试通过；未更改旧 review/canary hash 或生产 authority。
- 修改文件 compileall 与 `git diff --check` 通过。本批没有无排除核心全量、没有 live 视点运动或杯把手可见性实验，不复用前轮数字冒充本轮验收。

边界与字段示例见 [wrist sampling](wrist-viewpoint-sampling.md)；[主动感知设计记录](active-perception-design-options-2026-09-05.md) 更新了局部实现和未决问题。按协作技能只读核对 RFC revision 2158 的评审边界，新增字段、hash、候选 cap 和 handoff 提示均为待三人评审的个人分支候选；未写共享文档、重启现有服务、提交或推送。

下一步需要弱定位/身份决策，再实现前期恢复入口与实际新 packet 投影；工作域过滤、质量评估、完整 freshness 和真实 Spatial 0 / Long 9 回归仍未完成。HV-03 保持 partial，goal 保持 active。

## 实验恢复 R0 第七批 — 2026-09-07（跨日续行）

上一轮有实际代码与验证产出。本轮继续检查视点引用有效期；用户尚未确认弱定位与身份锚点的设计边界，自动 goal 续行与日期更新不视为方案批准。没有创建新的前期感知接口。

缺陷与修改：

- `resolve_wrist_viewpoint_candidate` 原先未检查生产调用成功，使用宽松 epoch 转换（缺失、畸形或 bool 可退为 0），且仅浅拷贝嵌套 pose。现要求明确成功结果、非失败状态、严格整数 epoch，拒绝重复 candidate ID，并深拷贝解析位姿，避免消费者修改历史几何。
- proposal 必须带完整 schema、compiled target、source packet、camera 和 mount 绑定。旧精简记录不能凭当前信息补造其原始来源，应重新生成；输出契约同步声明生产代码已有的这些必需字段。
- 生成与解析时检查 compiled artifact 的当前 object epoch；有活动 grasp provenance 时，拒绝不同 compiled ID、目标 superseded、contact invalidation。解析再比较冻结目标锚点与当前 compiled geometry；同身份新 mask 不因 detection ID 刷新而自动失效。
- 基于原始和最新同相机 packet 的实测 EEF/相机外参，进行有界纯几何计算并比较相机安装变换（绝对误差 `1e-6`）。同 robot epoch 下的安装平移/旋转变化或标定缺失也会拒绝旧引用；等价 packet-only refresh 继续复用。
- 该检查不创建新工具调用、注册新 proposal、替换选中 pose 或授权运动；仍由既有 exact IK 和 receipt gate 处理执行。缺少 provenance graph 不被提升为 Host 独立身份验证，环境 incarnation/operation lifetime 仍未解决。

测试与审计：

- 新增 30 个视点引用测试实例，涵盖非法/缺失 epochs、失败 producer、嵌套可变别名、重复 ID、mount 变化/缺失、目标切换/几何变更、同身份新 mask、等价 packet 刷新和缺少来源。旧的正向 pipeline 夹具补为真实图像 packet + compiled artifact + 实际生成 proposal，保留 exact short-ID→IK handler 验证，没有改成负向测试来掩盖回归。
- 最终扩大组 `tests/tools`、contract remediation、catalog、fixture receipt、geometry contract、planner registry、decision context：893 passed，36 个已有 Pillow 告警，7.22 秒。
- 核心测试（排除 integration、tools、manual proxy 和迁移状态测试，与上一批相同划分）：1364 passed、8 skipped，50.28 秒。本批未再次运行独立 HTTP proxy 组，不借用前轮数字冒充当前验证。
- 最后四个新增用例加入后，视点与生成文档一致性定向组：72 passed，0.52 秒；之后上述扩大组再次通过。
- 只读迁移审计仍 `internal_conformant=false`，失败项仍为历史 empty-policy authority baseline、depth-prior canary 和 remaining-tool canary；五份生成投影均 current。未修改历史 review/canary hash。
- 修改文件 compileall 与 `git diff --check` 通过。

按 OpenETA 协作技能只读核对 RFC revision 2158 的契约边界，mandatory result bindings 与 gate/兼容性变化标记为待三人评审的个人分支候选；未写共享文档、重启服务、提交或推送。详见 [视点引用有效期](wrist-viewpoint-sampling.md)。

本批没有 live motion、pre-SAM3 恢复或 Spatial 0 / Long 9 验收。完整 TODO 和近期真实实验验收均保持开放，goal 保持 active。

## 实验恢复 R0 第八批 — 2026-09-07

本轮继续执行与恢复安全检查，未缩减完整 goal。按 OpenETA 协作技能只读核对 RFC revision 2158 的评审边界；弱定位/身份资源的选择尚未确认。本轮另提出 Host 私有 operation ID、终态查询和过期请求拒绝的单进程 MCP 最小协议建议，尚未收到设计决定，未实现或视为获批。

复现与修复：

- 执行真实 AgentMemory 代码的无仿真诊断：记录 transport-unknown move，再输入目标 XYZ 完全匹配的 observation；旧实现由 blocked=true 变为 status=completed、blocked=false，输入没有任何远端操作完成证据。三次稳定 miss 与 gripper aperture agreement 也存在将位置结论作为解锁依据的问题。
- MCP `observe_env` 先前只使用后台线程包装，没有持有完整 control 锁；worker 的单步 observation 锁无法阻止 OSC 外层循环在观测后继续运动。现与 control/cleanup 共用 session/handle 锁，拒绝 retiring handle，其他 handle 保持独立。该锁不能阻止尚未到达服务端的请求，不是 operation completion 协议。
- Memory 将位置结论保留为 `position_reconciliation_status`，操作 `status` 始终 unresolved、`remote_completion_verified=false`；保留任何 reconciliation fact 都阻止除 observe 外的工具。旧 completed/failed 快照不再豁免，首次匹配只推进一次 epoch；继续保留 gripper 的无推断 latch/旧 proxy/PASS 规则。
- Planner 不再对未知操作反复强制 observe，改用既有 terminal response/talk 停止 episode，优先于 fresh-observation obligation。停止不证明任务成功，也不通过人工/模型口头回答解锁。未知操作记录始终保留在 unresolved obligations 中；手动只读 observe 仍可经过 pipeline。
- 当前没有原地恢复入口。需由 Host 确认旧环境已清理后再开始新环境/session，不能靠删 fact、reset 或换 session ID 证明旧物理操作停止；planner 响应本身不重启服务或清理用户 handle。
- 工具 gate 描述和两份生成契约同步实际行为；新增 [超时安全边界](motion-outcome-safety.md)，更新 cleanup、gripper、近期验收文档与 TODO。新 memory 字段、legacy gate 兼容和 planner 停止语义均标记个人分支候选、待三人评审，未更新生产 authority。

最终验证（组间重叠，不相加）：

- 新增 `tests/test_motion_reconciliation.py` 16 项：目标匹配仍 fenced、连续稳定 miss、重复观测只推进一次 epoch、五类持久化旧快照、四种状态的终止决策、实际 pipeline 禁止后续 gripper 但允许 observe、无未知操作时保留正常 refresh、无单一终点的 trajectory 保持 unresolved。生命周期新增 4 项测试实例覆盖完整 control 间隙观测阻塞、不同 handle 独立和三类 retiring 状态。
- 定向扩大组（motion、gripper、decision context、lifecycle、catalog、contract remediation、planner registry、episode feedback、sim proxy/control/collision）：475 passed、3 skipped，4.94 秒。
- 核心组 `pytest -q tests --ignore=tests/integration --ignore=tests/tools --ignore=tests/test_manual_vlm_proxy.py --ignore=tests/test_tool_contract_migration_status.py`：1384 passed、8 skipped，51.53 秒；不是无排除全仓运行。
- 获准在沙箱外单独运行 lifecycle，实际覆盖 asyncio 唤醒：27 passed，0.77 秒；只使用内存/并发夹具，不连接现有仿真服务。
- tools + motion + catalog + geometry contract + fixture receipt：698 passed，36 个已有 Pillow 告警，5.56 秒。
- 初次新增 normal-refresh 用例没有绑定 observe handler，因 registry correctly reports unexecutable 而失败；补齐正向测试夹具后重跑以上各组通过，没有更改该生产逻辑来迎合测试。
- 只读迁移审计仍 `internal_conformant=false`，失败项仍为 empty-policy authority baseline、depth-prior canary、remaining-tool canary；五份生成投影均 current，未改旧 hash。本轮未重复运行独立 manual HTTP proxy 组或 integration，不引用历史数字充当本轮验证。
- 修改文件 compileall 和 `git diff --check` 通过。

这批修复的是“超时后不能误续跑”，不是“超时后已能透明恢复”。没有 live 超时、迟到动作、Human VLM 或 Spatial 0 / Long 9 验收；没有服务重启、共享文档写入、提交或推送。最小 operation 协议的设计确认、前期感知入口以及真实实验验收仍开放，goal 保持 active。

## 实验恢复 R0 第九批 — 2026-09-07

上一轮有实际代码和回归产出。本轮检查近期实验入口的预算路径，按当前代码确认审查第 7 项的 guidance 仍位于受计时保护的 planner/action worker 之外；这会影响 Human VLM/模型辅助实验的有限时间预算。检查启动脚本后没有直接运行其中长时间默认任务，也没有改写用户的脚本。Goal 范围不变。

修改：

- Guidance 使用剩余 episode 时间，而非在主线程无界等待。输入 memory/observation 深拷贝，只有 runner 提交回答；deadline、cancel 和 session generation 在提交边界复核。修正 guidance context 使用下一轮索引的问题，现记录实际刚完成的 turn。
- 超时返回已有的 planner/action step，保留 episode_timeout/truncated，不误转入 human-wait 而冻结过期预算。显式 interrupt 也能结束等待；迟到 worker 的结果/异常不写回 memory。
- `wait_for_idle` 保留对仍存活 guidance worker 的跟踪。复用同一 runner 前要求原 tracked worker 已退出，不能以停止等待冒充请求已终止。同步 provider 本身未被强制取消。
- Guidance 保留 backend usage/source 的独立副本，正常回答和 abstain 都计入 episode tokens，并用 `guidance:<source>` 标记来源调用数。严格使用非负整数计数，缺 total 时可由两个有效组件求和；超时、非法响应或缺失 usage 的成本仍不完整，不伪造为免费调用。
- Action 已耗尽预算时不再派发 guidance；预算/interrupt 停止后也不启动新的自动 self-improvement review。正常 review 的异常或畸形报告作为结果附属错误返回，保留已产生的 episode steps/outcome，并明确 `partial_effects_possible=true`，不声称回滚已写提案或已批准应用。
- 正常 review 尚未有独立预算；其方法可能保存/应用变更，本轮没有把它简单放入可遗弃线程。Provider cancellation、统一调用上下文、dispatch 前工具配额和完整 usage reconciliation 仍未实现。

新增 `tests/test_guidance_budget.py` 21 项测试实例，覆盖阻塞真实线程的截止/显式中断、迟到写入与输入别名、busy runner 拒绝复用、实际 episode step 保留、session 替换、context 准备耗尽预算、普通失败转人工、正常/abstain usage、非法计数及嵌套副本、review 异常/畸形输出保留结果。测试中的后台线程都有 finally 释放和 join，不遗留等待任务。

验证：

- 最终定向扩大组（guidance、episode budgets、supervision、self-improvement、parallel harness/human interaction、episode feedback、success evidence）：140 passed，3.36 秒。
- 最终核心组 `pytest -q tests --ignore=tests/integration --ignore=tests/tools --ignore=tests/test_manual_vlm_proxy.py --ignore=tests/test_tool_contract_migration_status.py`：1405 passed、8 skipped，52.27 秒。本批没有重跑真实 HTTP、integration 或 live 仿真，不复用历史结果充当当前验收。
- 只读迁移审计仍 internal_conformant=false，失败项仍是历史 empty-policy authority baseline、depth-prior canary、remaining-tool canary；五份生成投影均 current。本批未修改工具生成契约、旧 authority/canary hash。
- 修改文件 compileall 与 `git diff --check` 通过。

按 OpenETA 协作技能只读核对 RFC revision 2158 的评审边界；usage 字段/计账标签、timeout event、review failure/skip metadata 均记录为个人分支候选、待三人评审。新增 [预算边界](episode-budget-boundaries.md)，同步 TODO、审查第 7 项和近期交付文档；未写共享 RFC/Wiki、提交、推送或重启服务。

这是 guidance 等待与迟到提交的局部闭环，不是全调用硬预算。没有真实模型取消、Human VLM 或 LIBERO 回归验收；前期感知资源/operation 协议设计决定也仍待确认。完整 goal 与近期验收均保持开放。

## 实验恢复 R0 第十批 — 2026-09-07

上一轮有实际代码和回归产出。本轮继续审查第 7 项，确认 `max_tool_calls` 仍在动作返回后用 `>` 检查，顺序请求可进入第 N+1 个 handler，batch 可一次越界多个。保留完整 goal 和近期真实实验交付范围，不将计账字段语义混同为物理任务成功。

修改：

- 新增 `agent/tools/call_budget.py`：Host 私有、线程安全的进程内 admission ledger。Runner 以配置上限和已携带 attempt usage 创建配额，通过 runtime execution scope 传入注册表；Agent 不填写预算对象或预留参数。
- 注册表在可执行工具的 authorization/handler 前原子预留一次。额度不足返回 `tool_call_budget_exhausted` / dispatched=false，不进入 handler 或昂贵的 world-action reviewer。失败、授权拒绝或预留后的取消不返还次数；调用开始时已取消则不预留。
- 嵌套 scope 和 caller metadata 不能覆盖 Host quota；注册表后台 handler 线程继续携带同一配额。包装层保留 cooperative-cancellation marker，避免破坏 Python sandbox 的清理/执行权边界；authorization 期间取消后也不会进入 handler。
- Read-only batch 逐项准入，返回已执行结果和明确的拒绝项，不声称事务回滚。额度刚好用尽仍可 response/pause；下一次拒绝会触发已有 tool_call_limit_exceeded。嵌套调用被拒绝时，即使 outer handler 返回 success，episode 仍停止。
- 保留 `tool_call_count` 的历史请求/失败/阻断尝试计数，并增加已携带次数 + 实际准入次数的保守下限，避免嵌套调用在 pause/resume 时丢失。新 `usage.tool_admission` 和 failure.admission 分别呈现 limit、carried_usage、admitted_this_run、denied_this_run、remaining。它们不是 controller step 或成功任务计数。
- 工具契约增加 `runtime.tool_admission` gate binding。同步 catalog JSON/Markdown；只读审计发现 promotion campaign 的 gate_binding_count 投影过期，使用现有生成器和原 fixture/review/canary 输入重新生成。差异为 gate 数量增加，没有改旧 approval 或 canary hash。

测试与验证（组间有重叠，不相加）：

- 新增 `tests/test_tool_call_budget.py` 23 项实例，覆盖 batch 部分执行、exact-limit response、carried usage、并发竞争、scope/metadata 覆盖、两种 handler 线程模式下的嵌套调用、授权前拒绝、失败不返还、cooperative marker、私有对象不进入公共事件、取消、pause/resume 以及 outer success 不能掩盖 nested denial。
- 原“预算 2 次，第三次尝试失败”用例保留历史 attempt count，同时新增真实 handler 调用记录，验证只进入两次 handler，并检查 admission 分解。没有只改报告数字来冒充少执行了一次。
- 定向组（quota、episode budgets、Python sandbox isolation、catalog、geometry contract、parallel human interaction）：120 passed，10.41 秒。初次两个新测试误用了已有 handler 的绑定，补齐显式 replace=True 后复测通过；没有放宽注册表重复绑定规则。
- 核心组 `pytest -q tests --ignore=tests/integration --ignore=tests/tools --ignore=tests/test_manual_vlm_proxy.py --ignore=tests/test_tool_contract_migration_status.py`：1428 passed、8 skipped，52.18 秒。
- Tools + quota + catalog + fixture receipt + geometry contract + promotion campaign：711 passed，36 个已有 Pillow 告警，4.88 秒。
- 最终只读迁移审计仍 internal_conformant=false，失败项回到历史 empty-policy authority baseline、depth-prior canary、remaining-tool canary 三项；五份生成投影均 current。未将重建派生投影视为协作者批准。
- 修改文件 compileall 与 `git diff --check` 通过。

按 OpenETA 协作技能只读核对 RFC revision 2158，新增私有 runtime 参数、诊断/gate、计账字段及保守下限语义均是待三人评审的个人分支候选。[预算边界](episode-budget-boundaries.md)、审查第 7 项、TODO 与近期验收说明已同步；未写共享文档、重启服务、提交或推送。

该配额只覆盖绑定 execution scope 的可执行工具尝试。Direct simulator 调用、环境 reset/cleanup、handler 内部模型调用、未携带 scope 的其他 registry/raw thread，以及单次动作内部 controller steps 仍有各自边界。它不是分布式 lease 或全系统硬预算；没有 live Human VLM/LIBERO 回归验收。完整 goal 保持 active。

## 实验恢复 R0 第十一批 — 2026-09-07

本轮推进实际服务验证，不再仅积累单测。沿用个人重构分支和完整 goal，按 OpenETA 协作技能只读核对 RFC revision 2158；新增 canary 诊断字段/验收语义仍是待评审的本地候选。未提交、推送或写共享 RFC/Wiki。

实际发现与修改：

- 既有 8766/8099 服务启动于 9 月 4 日，8099 有 2 个待答请求。本轮另启独立 18766 模拟器和 18099 控制台，不操作已有环境或请求。
- 首次候选服务使用 `.venv`，canary 完成 6 steps、约 3 mm 终点误差并确认关闭，但日志指出没有 cuRobo；旧 canary 没有核验实际 collision coverage，仍报告 passed。保留原报告并明确降为控制器/清理证据，不当作安全通过。
- `controller_capability_canary.py` 新增真实覆盖核验：运动前要求含场景物体的 endpoint world/self 检查，运动后检查实际 collision receipt。缺失/不可用/覆盖不全/检测到碰撞不能整体通过；关闭检查的 debug 运行也不能 acceptance pass。保留 path 与 collision 原始摘要，不声称扫掠路径安全。
- 使用已有 RAG Python 3.10/cuRobo 依赖重新启动独立候选，严格 canary 实际通过：Object 0 seed 0、OSC、6/40 steps、误差 0.0029946554 m、world/self 覆盖、明确关闭。OSC seed/local execution 和连续路径安全仍未因此解决。
- 运行生产 planner/runtime/runner 的实际 Human VLM 模式观察 smoke，Codex 查看真实图像后逐次回答两个私有请求。observe 执行、结果及刷新图像进入下一轮、talk 停止；2 turns、1 tool admission、112.759 s，Host finally 确认关闭。无运动、抓取或物体任务成功，不覆盖嵌套 advisor。
- Live 报告暴露等待归因缺口：控制台分别等待约 44/57 s，而 episode human_wait_s=0、human_assisted=false。另确认模拟器 assigned task 成为 observation/控制台任务标题；诊断限制不能只靠 task prose。本轮将两项补入 TODO，没有回写历史统计或未经设计确认改公共字段。
- 独立控制台 pending=0 后停止；独立端口均已释放。原 8766/8099 PID 保持不变、8099 pending 仍为 2。没有全主机孤儿扫描或 crash/lease 验收。

验证：

- Canary 新增 14 个覆盖检查测试实例；canary/control codecs/collision safety 定向组 91 passed、2 skipped，0.79 s。
- 核心组 `pytest -q tests --ignore=tests/integration --ignore=tests/tools --ignore=tests/test_manual_vlm_proxy.py --ignore=tests/test_tool_contract_migration_status.py`：1442 passed、8 skipped，53.06 s。不是无排除全仓运行。
- 沙箱外 HTTP/console 夹具组 `tests/test_manual_vlm_proxy.py tests/test_manual_vlm_console.py`：34 passed，3.13 s；与上述实际两轮观察实验分开记录，不把夹具中的嵌套请求当作 live advisor 验收。
- 只读迁移审计仍 internal_conformant=false：empty-policy authority baseline、depth-prior canary、remaining-tool canary 三项历史证据未满足；五份生成投影均 current。未改旧 authority/canary hash。
- 修改代码 compileall 与 `git diff --check` 通过。

会话 ID、实际回执、启动命令、源码指纹及本地原始证据路径见 [本轮 live 记录](diagnostics/experiment-ready-live-2026-09-07.md)。近期里程碑只勾选短程 canary 与观察 smoke；既有成功路径全程回归、Spatial 0 / Long 9 有界诊断、受控超时验证与剩余设计/实现仍开放。完整 goal 保持 active。

## 实验恢复 R0 第十二批 — 2026-09-07

前一轮取得实际控制/观察证据，本轮继续准备难任务诊断。沿用原分支、保留所有既有修改；按 OpenETA 协作技能只读复核 RFC revision 2158。没有将 goal 缩小为感知 smoke，也未关闭完整回归要求。

实际检查与权限阻断：

- 只读提取 `.mcp.json` 的已配置端点；SAM3/AnyGrasp/AnyPlace 的根路径健康接口均返回 ok。最初 `/health` 为 404，经本地服务源码定位正确路由；没有误报后端宕机。AnyGrasp capability 查询明确返回 0.08 m / 0.03 m 几何、最多 20 candidates。
- 已准备基于共享 runtime assembly 的私有 Spatial 0/Long 9 感知驱动，预算为每任务 8 turns、7 tool attempts、600 s，并禁用运动与夹爪 handler。但客户端启动被权限审查拒绝：新仿真数据发送至具体内网模型端点尚缺明确授权。未执行该驱动、未创建新环境、未调用感知推理，未改通道或地址绕过拒绝。
- 启动过的独立 18766/18099 服务均正常停止。最终只读端口检查中私有端口已释放，原 8766/8099 PID 不变、原控制台 pending=2。没有新的难任务结果，相关验收保持开放；等待用户确认 RGB/深度/掩码/标定数据向指定感知服务发送的授权。

受控入口修复：

- 发现五个 Human VLM 启动脚本的 `OPENETA_LOCAL_SIM_PORT` 仅影响端口检查，没有传给 CLI。CLI 仍读 `.mcp.json`，因而可能检查 18766 却实际连接 8766；缺少命名 simulator 时还会选第一个任意 MCP 服务。
- 新增 CLI Host 参数 `--simulator-mcp-url`，在 runtime 构建前验证并保存 run override。五个脚本只改必要的 argv，把检查过的 simulator 地址显式传入；没有重写用户的任务/预算或磁盘配置。未提供 override 时仅接受 `openeta-sim` 或 legacy `openeta`，不再回退到感知服务。
- 验证 URL scheme、host、端口及敏感/畸形形式；拒绝 credentials/query/fragment/控制字符，错误不回显原始敏感值。初次实现用空字符串作 argparse default 导致四个旧 CLI 用例失败，修为 None default 并加入专用错误包装后复测通过；没有删除旧用例。
- 保留有绑定 handle、closing/close_failed、close_in_progress 或 tracked worker 尚未退出时的原 transport，拒绝换地址/移除配置。新 transport 构造失败不改旧地址与 catalog；空闲清理确认后才允许重新选择。该保护不是远端 operation lease，未知创建/未跟踪线程等问题仍开放。
- 运行 registry 投影显示 effective simulator 地址及配置来源，其他感知服务保持不变；不在 Agent 参数或工具 schema 中增加 endpoint 字段。Host 配置兼容、registry 来源和生命周期行为变更仍是待协作者评审的个人分支候选。

验证（组间重叠，不相加）：

- 新增 `tests/test_simulator_endpoint_selection.py` 共 45 项实例。涵盖端点选择、旧连接保留与原服务 cleanup、构造失败、创建前 worker 阻断、非法 URL 不回显、参数进入 constructor；使用实际 Bash 执行五个脚本在两个端口下的 argv，curl/nc/openeta 全为隔离本地替身，不发网络请求。
- 最后增加的正向用例使用真实 shared runtime assembly，验证 registry 的 observe 和 CLI cleanup 均进入 override transport。最终 endpoint/CLI/runtime assembly/lifecycle 组：134 passed、1 skipped，6.43 s。
- 核心组（同前轮排除 integration、tools、manual proxy 和历史 migration-status 测试）：1486 passed、8 skipped，53.09 s；此组在最后一个正向用例加入前运行，后续扩大的定向组已覆盖该用例，不虚增核心测试数。
- 只读 migration audit 仍 internal_conformant=false，历史 empty-policy/depth-prior/remaining-tool authority 三项未满足；五份生成投影均 current。没有改旧审核/hash 或生成新 authority。
- compileall、CLI `--help` 与 `git diff --check` 通过。此批未重跑其他真实 HTTP fixture，也没有新增 live motion。

新增 [simulator 地址选择与授权边界](simulator-endpoint-selection.md)，同步 TODO 和近期交付文档。未提交、推送、写共享文档或重启已有用户服务。完整 goal 保持 active；感知发送授权是本轮首次遇到的外部阻断，不满足将整个 goal 标记 blocked 的条件。

## 实验恢复 R0 第十三批 — 2026-09-07

上一轮完成入口修复，本轮尚未收到感知数据发送授权，不把自动 goal continuation 当作批准，也没有重试被拒绝的客户端。继续处理审查第 8 项的可独立修复问题；完整 goal 和真实任务回归范围保持不变。按 OpenETA 协作技能只读核对 RFC revision 2158。

复现与实现：

- 在不发网络请求的独立进程复现 NO_PROXY context 交错：A 进入、B 进入、A 退出后，B 仍活动但 bypass 消失；B 退出后 A 的 host 与原本不存在的小写变量残留。原 finally 恢复无法解决并发的非栈式退出。
- 检查 `.venv` 和 RAG 实际 MCP 1.28.1 SDK 源码/签名后，SSE list/call 改用其 `httpx_client_factory`，每次连接创建目标 host 的独立 direct mount，不再修改全进程 NO_PROXY。没有用全局网络锁序列化不同请求。
- 保留其他 host 的环境代理、环境 CA/TLS verify、SDK headers/auth/timeout 与 redirect 行为；只针对准确 host，不将其扩展为子域名授权。SDK context 负责 client 关闭，错误与超时分类保持不变；不宣称同步超时已取消远端模型推理。
- 去掉已无用途的环境改写辅助函数/import。项目依赖下限提高到已验证的 `mcp>=1.28.1,<2`，锁文件原本已是 1.28.1，仅同步 requirement metadata，没有安装/升级依赖。
- 原测试不再要求出现临时全局 NO_PROXY 写入，改为明确检查不变；另用真实 HTTPX 路由与真实 SDK loopback 验证 direct 行为，未以删除旧断言代替功能证明。

验证（组间重叠，不相加）：

- 新增 `tests/test_mcp_proxy_isolation.py` 16 项实例：IPv4/IPv6/IDNA/大小写、跨端口及其他 host、两 MCP client 与普通 client 共存、synthetic redirect、headers/auth/timeout、CA 校验、list/call 的正常/失败/超时清理、两线程交错及外部新环境值不被旧调用恢复覆盖。
- 最终 proxy/transport/endpoint/catalog/runtime assembly 定向组：162 passed，3.48 s。初次命令误把 AnyGrasp capability 测试写在 tests 根目录而未找到文件，修正路径后重新执行，未将该命令当作通过。
- Core 组（同前轮排除 integration、tools、manual proxy、historical migration-status）：1503 passed、8 skipped，54.53 s。
- AnyGrasp capability / SAM3 handler / grasp estimate handler 夹具：85 passed、36 个已有 Pillow 告警，1.26 s；没有实际感知推理。
- 获准运行 `tests/integration/test_mcp_proxy_loopback.py`：1 passed，1.14 s。实际 FastMCP + SDK 在本机随机端口完成 list_tools 与两个并发 call_tool；只发送 fixture-a/fixture-b 固定字符串。故障代理指向本机不可用地址，目标仍直连；环境不变，测试线程/socket 已退出。不是 Human VLM 或 LIBERO 验收。
- `uv lock --offline --check --no-cache` 通过（60 packages，147 ms）。默认沙箱中 snap uv 因 capability 限制未启动，获准后执行相同离线检查，没有在线下载或更换渠道绕过数据发送限制。
- 只读 migration audit 仍 internal_conformant=false，失败项仍为历史 empty-policy/depth-prior/remaining-tool authority 三项；五份生成投影均 current，未改旧 review/hash。compileall 和 `git diff --check` 通过。

新增 [MCP 代理隔离](mcp-proxy-isolation.md)，同步审查第 8 项、TODO 与近期交付边界。未新增 Agent 工具 schema，网络/依赖兼容行为保留个人分支待部署审查；没有写共享 RFC/Wiki、提交、推送或重启现有用户服务。真实感知授权、成功路径回归与 Spatial 0/Long 9 诊断仍待推进；本轮有实质代码进展，不满足整个 goal 的 blocked 条件。

## 实验恢复 R0 第十四批 — 2026-09-07

前轮代理隔离有代码和验证产出；本轮感知数据发送授权仍未收到，没有重试被拒绝的实验。继续审查近期实验入口的预算路径，保留完整 goal。按 OpenETA 协作技能只读核对 RFC revision 2158；新增 Host 参数与行为边界保留个人分支待集成评审，未改 Agent tool schema。

确认与修改：

- CLI runtime 重建和 batch factory 都把 simulator timeout 设置成 `max(300, provider.timeout_s)`。Human VLM 脚本设 provider timeout=86400 时，create/reset/工具 RPC 的等待期限也会升至 86400；这与已有 episode 等待/未知结果防护并不是同一个预算层。
- 新增共享 `agent/tools/mcp_timeouts.py` 的默认值与正有限数校验。CLI constructor/state 与 batch worker factory 采用独立 `simulator_timeout_s`（默认仍 300 s），不再读 provider timeout 来抬高它。拒绝 bool、非数、NaN/inf、零/负值；可以明确选择低于旧 300 s floor 的值。
- CLI、batch 和 experiment 的 runtime parser 支持 `--simulator-timeout-s`；新 batch、experiment run/iterate 和显式 batch resume 均把值传到 factory。Preflight 的 `--mcp-timeout-s` 仍用于其独立 discovery，不借新 flag 改其语义。
- Batch environment 的 create/reset/render 与工具 proxy 共享所选值；CLI 工具使用相同 Host 配置，切换 provider/重建 runtime 不改它。Cleanup 仍为 `min(selected, 30)`，catalog discovery 保留短预算。
- 五个 Human VLM 脚本增加 `OPENETA_SIMULATOR_TIMEOUT_S`（默认 300），显式传入 CLI；其余用户任务、provider/episode/token 默认值保持原样。没有修改 `.env` 或 `.mcp.json`。
- 非默认 simulator timeout 尚未写入 paused record schema；恢复时需显式重传，否则用独立 300 s 默认值。文档明确此边界，不冒充原 deadline 自动恢复或远端操作已终止。完整剩余 deadline 传播、provider cancellation、未知创建和原地恢复仍待解决。

验证（组间重叠，不相加）：

- 新增 `tests/test_simulator_timeout_config.py` 35 项实例。除正/非法输入与 parser/API 转发外，使用真实 CLI/shared runtime 与 batch environment 装配，断言 provider timeout=86400、重建后=172800 都不改变所选 simulator 值，create/reset/observe/close 的实际 transport 调用参数一致。
- 已有真实 Bash launcher 替身测试增加 timeout argv 断言，全部五个脚本在两种 simulator 端口下均传入显式 45 s，没有连接服务。
- Endpoint/timeout/CLI/parallel/human-resume/experiment 定向组：172 passed，7.92 s。
- Core 组（同前轮排除 integration、tools、manual proxy、historical migration-status）：1538 passed、8 skipped，54.91 s；不是全仓无排除运行。
- 只读 migration audit 仍 internal_conformant=false，历史 empty-policy/depth-prior/remaining-tool authority 三项未满足；五份生成投影 current，没有更新旧 review/hash。
- 修改模块 compileall、batch CLI help 和 `git diff --check` 通过。没有 live provider、simulator 或感知调用，也没有重复引用前轮 loopback 数字作本轮验证。

同步 [预算边界](episode-budget-boundaries.md)、[入口使用说明](simulator-endpoint-selection.md)、TODO 与近期交付文档。未提交、推送、写共享文档或修改现有服务。该修复缩短的是独立 RPC 等待配置，不证明远端硬取消；真实任务验收与授权请求仍开放，goal 保持 active。

## 实验恢复 R0 第十五批 — 2026-09-07

继续收敛实验入口的本地资源生命周期。按 OpenETA 协作技能只读核对 RFC 第 5 节约束，revision 2158；共享契约审查仍开放。感知数据发送授权尚未收到，没有重试此前被拒绝的诊断，也未启动/关闭任何真实服务或环境。

发现与实现：

- 原 tool creator 只在发出 create 前短暂检查 handle；RPC 尚未返回时，关闭可以报告成功 skip，另一创建也能进入。Batch reset 有同类窗口。新增共享 Host 私有 `startup_in_progress` 标记，在同一 lifecycle lock 下预留，覆盖创建、重置和后处理，finally 释放，不跨网络持有互斥锁。
- 启动期间三个关闭入口返回 failure/pending；拒绝重复创建/重置、proxy 使用及 CLI 端点替换。Batch 自己的有界启动重试仍可内部清理，不允许其他公开 close 越过标记；另一个共享配置中已有的不同 handle 不可被 batch reset 覆盖。
- CLI close 纳入本地互斥，失败结果不再永久缓存；旧的成功关闭结果不遮蔽后来新建的环境。保留 handle 却没有 transport 时不再返回成功 skip。
- Tool creator 在 artifact normalization/callback 前保存成功创建的 identity，后处理异常也有已知 handle 可清理。取消后的 cleanup 仅在确认成功时清除匹配 identity；失败保留 handle 和 close_failed。取消反馈改成“已尝试清理”，不再未经确认宣称“已清理”。
- 新增取消用例一度发现提前保存 handle 后，第二个取消检查点的失败 cleanup 仍留下 active 状态；修正为对同一 identity 同样设置 close_failed，未删除失败断言。

验证（组间重叠，不相加）：

- 新增 `tests/test_environment_startup_admission.py` 20 项实例，使用 event-controlled 本地线程与假 transport。覆盖 create/reset 两阶段、两个启动入口、三个关闭入口、重复启动、端点拒绝、代理拒绝、失败/非法参数释放、callback 失败保留身份、取消后 cleanup 成功/失败、CLI 缓存与重试。
- 最终 startup/lifecycle/proxy/endpoint/CLI/timeout 定向组：274 passed、1 skipped。既有 transient startup retry 测试仍通过。
- Core 组（同前轮排除 integration、tools、manual proxy、historical migration-status）：1558 passed、8 skipped，54.45 s。不是全仓无排除通过，也不替代 live 并发/任务验收。
- 只读 migration audit 仍 internal_conformant=false，历史 empty-policy/depth-prior/remaining-tool authority 三项未满足；五份生成投影 current，未改历史 review/hash。修改模块 compileall 与 git diff --check 通过。

同步 [清理边界](environment-cleanup.md)、TODO 与近期交付文档。该标记不是 remote operation ID/lease：transport 异常且无 handle 的未知创建仍未解决，历史 transient create retry 未在此批重设计；pending close 不会自动延期执行，调用方须等待 tracked startup 退出后重试。新增诊断/行为保留个人分支待协作 review；未提交、推送或写共享文档。完整 goal 保持 active，真实成功路径回归和 Spatial 0/Long 9 诊断仍等待数据发送授权。

## 实验恢复 R0 第十六批 — 2026-09-07

前轮有实质代码/验证进展；本轮沿 live smoke 已确认的问题，推进 Human VLM 主 planner 统计归因。按 OpenETA 协作技能只读核对 RFC 第 5 节，revision 2158；不将尚未批准的数据发送当作已授权，没有重试真实感知诊断。

实现与边界：

- 确认原 console completion 没有模式/时长元数据，backend 只传 token usage，episode 的 human_wait_s 仅覆盖 runner 交互暂停。这解释了实际约 44 s/57 s 操作者等待未被识别的 smoke 记录；未改写该历史记录。
- 独立控制台在接受回答前计算 monotonic queue-to-response 时长，在 completion envelope 和 response trace 中写入 `manual_vlm.provider_interaction.v1`。字段由服务生成，不来自 adapter/operator message；不修改 assistant 内容或伪造 billed token usage。
- Backend 严格校验已知版本/mode、非空有界 request ID、正或零有限数时长（拒绝 bool/字符串/NaN/inf/溢出）。Planner 对所有 action-validation 尝试保留有效元数据，最终 action 携带这些记录。
- Episode 对已记录主 planner action 中的 request ID 去重，新增 usage.manual_provider 和 assistance.manual_provider_assisted；human_assisted 同时识别该手动通道，包括零等待响应。该标记不证明操作者是自然人：Codex 操作 Human VLM 仍算 manual-provider assistance。
- 旧 human_wait_s 和预算时钟完全保留，未把 provider 等待从 episode timeout 中额外扣除。覆盖范围明确为本次运行已记录主 planner 响应；nested advisor、guidance、visual delta、review、未完成/取消/未记录 action 和跨 resume 汇总仍开放。字段缺失/false 不能证明完全自主。

验证（组间重叠，不相加）：

- 新增 `tests/test_manual_provider_accounting.py` 20 项实例，包括真实 backend/planner/runtime/episode 装配（假 provider transport）、校验重试累计、去重、新 run 清账、零等待、非法 metadata、monotonic 计时与忽略 operator 自填 metadata。
- 初版测试用 JSON 而非实际 XML wire，触发校验重试后假响应耗尽；修正夹具协议再验证，未把 provider failure 当作正常响应证据。HTTP 测试另发现通用 console 不能含项目专属命名，将 envelope schema 改为独立 manual_vlm 命名空间，保留原 generic-core 隔离断言。
- 最终 accounting/resource budgets/planner/parallel/guidance 定向组：190 passed，3.28 s。
- 获准执行本机随机端口 HTTP/console 夹具：34 passed，3.27 s；只用固定测试数据，未连接现有用户控制台或感知服务，测试服务已关闭。这不等同 fresh live smoke。
- 最终 Core 组（同前轮排除 integration、tools、manual proxy、historical migration-status）：1578 passed、8 skipped，54.43 s。不是全仓无排除通过。
- 只读 migration audit 仍 internal_conformant=false，历史 empty-policy/depth-prior/remaining-tool authority 三项未满足；五份生成投影 current。compileall/git diff --check 通过，未更新旧 review/hash。

新增 [人工 provider 计账与限制](manual-provider-accounting.md)，同步 TODO 和近期交付文档。Envelope、新 metadata 和 human_assisted 扩展语义是个人分支候选，需要三人 review；未新增 Agent 输入工具或修改任务成功判据。未提交、推送、写共享文档或重启已有服务。完整 goal 仍 active，真实成功路径回归、Spatial 0/Long 9 诊断和感知发送授权仍待推进。

## 实验恢复 R0 第十七批 — 2026-09-07

前轮主 planner 统计已有代码与验证产出，本轮继续检查 HV-06 的真实使用阻断：操作者取消请求后是否会被自动重试。按 OpenETA 协作技能只读核对 RFC 第 5 节，revision 2158；没有重试被拒绝的感知实验，也未操作任何已有控制台请求或 simulator。

确认与实现：

- 控制台对显式取消和 decision timeout 都返回带 human_cancelled code/type/request_id 的 HTTP 503；原 ProviderHttpError 只保存 status_code，因而会自动 retry，配置 fallback 时还可能切换 provider。这会把已经终止的手动请求重新提交。
- ProviderHttpError 现在识别该已有结构化 envelope；transient/failover 两个判定均拒绝重试，终态返回 manual_provider_cancelled / retryable=false。错误消息含 quota/overload 不覆盖明确取消证据；普通 503、连接故障与既有账号 failover 行为保持。
- 识别仅限 HTTP 503、匹配 code/type、非空有界 request ID 和最多 16,384 字符的可解析 JSON；不靠错误字符串包含某词判断。没有改 console HTTP status/schema，能够理解已有服务的取消响应，不需要为本次验证重启用户服务。
- 保留 main planner 的 structured ask_human 错误路径，验证层/episode 不因此再向 provider 请求。停止范围仅当前 backend invocation，后续显式重试仍可由操作者决定；不冒充父子请求级联、client timeout 远端取消或已取消请求成本计账。

验证（组间重叠，不相加）：

- 新增 `tests/test_manual_provider_cancellation.py` 20 项实例：取消/console timeout、fallback 有无、错误消息干扰、普通/畸形/超限 503 兼容、实际 urllib HTTPError 转换、backend → planner → episode 无重复调用。与既有 fallback 组共 34 passed，0.62 s。
- 新增两项真实 loopback backend/console 夹具，分别手动取消和 console deadline；配置备用 provider 也只指向自建本机 fixture，断言总共一个 cancelled 请求、一次 provider attempt、无 failover。最终 HTTP/console 组 36 passed，4.18 s。只发送固定测试文字，测试 server 已关闭，未使用仿真图像/现有服务。
- Core 组（同前轮排除 integration、tools、manual proxy、historical migration-status）：1598 passed、8 skipped，53.60 s。未宣称无排除全仓或真实任务验收完成。
- 只读 migration audit 仍 internal_conformant=false，历史 empty-policy/depth-prior/remaining-tool authority 三项未满足；五份生成投影 current。compileall/git diff --check 通过，没有改历史 review/hash。

同步 [人工 provider 取消与限制](manual-provider-accounting.md)、TODO 和近期交付文档。新增错误码/重试语义保留个人分支待三人 review；未提交、推送、写共享文档或重启用户服务。感知发送授权、实际成功路径回归及 Spatial 0/Long 9 诊断仍开放，完整 goal 保持 active。

## 实验恢复 R0 第十八批 — 2026-09-07

前轮明确取消的重试修复有实质产出；本轮回到普通 post-review 的预算/迟到写入待办。按 OpenETA 协作技能只读核对 RFC 第 5 节，revision 2158。没有收到新的感知数据发送授权，也未重试被拒绝的实验。

在着手独立预算前确认了一个前置隔离缺陷：runner 把原始 EpisodeResult 直接交给 reviewer，捕获异常并不能阻止其清空 steps、改写 task/outcome/metadata；返回 report 同样保留 reviewer 可继续修改的引用。先新增四项复现，原实现全部失败，分别覆盖成功、抛错、畸形返回以及返回后修改保留引用。

修改：

- review 接收完整结果的独立 deepcopy，成功报告也在发布前 deepcopy。保持原结果内容与 review 接口，不让 review 通过这两个引用拥有任务步骤、结果或已发布 metadata。
- 记录 review 前保存 runtime session generation、memory identity 和 execution ID；发布 review memory event 时在 commit lock 下复核。session/episode 被替换（含相同 session ID 重新打开）则不向新 session 插入旧事件；旧返回结果仍附带报告和 memory_publication.recorded=false/reason。
- 没有把会写 proposal、应用已批准 skill 的整个 reviewer 丢到超时 daemon 中。独立 review deadline/token budget、计算/提交分离以及文件写入回滚仍未实现；本次复制不是时间/内存预算或任意 Python 沙箱，持有其他 runtime 引用的扩展仍有其独立副作用边界。

验证（组间重叠，不相加）：

- 新增 `tests/test_review_result_isolation.py` 最终 8 项实例；四项原问题复现修复通过，另四项覆盖成功/失败 review 遇到不同 ID 或同 ID session 重建时不污染新 trace。
- 最终 isolation/guidance/self-improvement 定向组：38 passed。原有 proposal 写入、正常 review 行为、预算停止跳过 review 和异常保留结果的测试继续通过。
- Core 组（同前轮排除 integration、tools、manual proxy、historical migration-status）：1606 passed、8 skipped，54.70 s。没有本轮 live/HTTP/provider 调用，未引用之前 fixture 数字作本轮新验收。
- 只读 migration audit 仍 internal_conformant=false，历史 empty-policy/depth-prior/remaining-tool authority 三项未满足；五份生成投影 current，未改 review/hash。compileall/git diff --check 通过。

同步 [后处理隔离和预算边界](episode-budget-boundaries.md)、TODO 与近期交付文档。memory publication 字段是个人分支候选，待三人 review；未提交、推送、写共享文档或操作用户服务。完整 goal 保持 active；此批没有以引用隔离替代独立 review 预算，也未标记真实任务验收完成。

## 实验恢复 R0 第十九批 — 2026-09-07

在上一轮引用隔离的基础上，继续实现默认 post-review 的计算/写入分离及独立准备期限。按 OpenETA 协作技能只读核对 RFC 第 5 节，revision 2158；阶段接口/config/metadata 保留个人分支待 review。没有收到新的感知发送授权，未重试此前被拒绝的实验。

实现：

- SelfImprovementReviewer 拆为 prepare_review / commit_review；BackendReviewedSkillAutoApplier 拆为 prepare / commit。原 maybe_review / apply 保留同步兼容入口。准备阶段计算提案、调用 author 与独立 reviewer，但不写框架 proposal/skill 文件，也不更新 live skill registry。
- 准备使用独立 episode/context/proposal/skill 数据；多个 skill 更新在虚拟 registry 上按顺序准备，主线程再按同一顺序提交。提交前比较 live registry skill 与准备快照、editability、候选名字；review config/applier/store 被替换也拒绝旧计划。不是磁盘外部编辑的内容哈希/CAS 协议。
- SelfImprovementConfig 新增 preparation_timeout_s，默认 120 s，严格正有限数。默认 reviewer 加 built-in backend applier/no applier 时，runner 用 tracked worker 等待准备，不给 worker 提交回调。期限或 session 失效在子阶段间复核，提交前在 memory commit lock 下复核。
- 超时保留原任务结果，review 返回 ReviewPreparationTimeout 和 preparation_budget（completed_in_time=false、worker_pending、commit_started=false）。未退出线程继续阻止 runner 复用；迟到结果不更新已返回报告，不写 proposal/skill，过期 author 返回后不再调用下一阶段独立 reviewer。
- 自定义旧 reviewer/subclass 或 opaque custom auto-applier 保持同步路径，不谎称已受上述期限限制。提交 I/O、task-playbook 提取、token 总额和 provider 内部请求/重试未变成硬取消或全局预算；部分提交失败不回滚。该边界也不是任意 Python 扩展的安全沙箱。

验证（组间重叠，不相加）：

- 新增 `tests/test_review_preparation_budget.py` 最终 20 项实例：非法配置、proposal/author/reviewer 阶段超时、无文件/registry 迟到写入、tracked busy 拒绝复用、及时正常提交、session/config 替换、skill 过期/变为不可编辑、legacy 扩展兼容、多个准备更新的虚拟及实际顺序。
- 最终 preparation/isolation/guidance/self-improvement 定向组：58 passed，1.33 s；既有实际 author/reviewer 类配 StaticBackend 的正常应用测试继续通过。
- Core 组（同前轮排除 integration、tools、manual proxy、historical migration-status）：1625 passed、8 skipped，54.83 s。最后一项多提案顺序用例在核心组之后加入并定向通过，未回填核心计数；实现代码在核心运行后未变更。
- 只读 migration audit 仍 internal_conformant=false，历史 empty-policy/depth-prior/remaining-tool authority 三项未满足；五份生成投影 current，未改历史 review/hash。compileall/git diff --check 通过。
- 本轮没有 live model、HTTP fixture、simulator 或感知推理；测试使用本地假服务对象和 tmp_path，所有测试阻塞线程已释放并通过 idle 确认退出。

同步 [准备期限及剩余边界](episode-budget-boundaries.md)、TODO 和近期交付文档，修正了当前文档中“默认 review 完全没有独立期限”的旧描述，同时保留全路径预算未完成的事实。未提交、推送、写共享文档或操作现有用户服务；真实任务验收和感知发送授权仍开放，完整 goal 保持 active。

## 实验恢复 R0 第二十批 — 2026-09-07

前轮默认 review 两阶段准备有代码与验证进展；本轮检查提交阶段的文件和重复执行边界。按 OpenETA 协作技能只读核对 RFC 第 5 节，revision 2158。感知发送授权仍未收到，没有重试被拒绝的请求或操作任何已有服务/session。

复现与修复：

- Store.save 直接拼接 proposal_id，未复用 load 的路径校验；同 ID save 会覆盖任何 pending/approved/rejected 记录。新增五项选定复现全部失败：三种状态覆盖、固定临时文件 symlink 改写 fixture 文件、读取接受内部 ID 与文件名不匹配。
- Save 在创建目录前验证规范 ID；路径分隔符、空/点名称、控制字符、超长及首尾空白/.json 后缀的非规范新 ID 被拒绝。Load 保留正常 filename.json 别名，但要求 JSON 内部 proposal_id 与实际文件名匹配，显示 path 由 Host 重建。
- 新建提案使用独占随机临时文件，完整写入并 flush/fsync 后通过 hard link 原子发布；目标已存在则 FileExistsError，不覆盖旧内容/决议。Status 更新也换用随机临时文件和 replace，去掉可跟随旧 .json.tmp symlink 的写法。发布失败只清理本次拥有的临时文件。
- Load 先检查普通非 symlink 文件，再用打开后的 fd 身份/类型校验防止替换后读入另一个文件；不信任存储 JSON 的 path 字段。没有自动改名/修复历史畸形记录。
- 重复提交准备计划会在 save 阶段失败，不能再次调用 auto-applier 或把 approved 记录重置为 pending。仍不宣称多提案事务、审批/application CAS 或部分提交可自动回放。

验证（组间重叠，不相加）：

- 新增 `tests/test_review_store_boundaries.py` 19 项实例，含非法 ID、三种既有状态保护、临时 symlink、内部身份/显示路径、两个 store 并发新建、重复 prepared commit 和发布故障清理。
- 在新建且自有的仓库内 `tmp/review-store-fs.ulZxok` 作为 basetemp 执行 store/self-improvement/preparation/isolation 组：56 passed，1.07 s，实际覆盖仓库文件系统的 hard-link 原子发布。只操作测试 fixture，未处理任何真实提案。
- Core 组（同前轮排除 integration、tools、manual proxy、historical migration-status）：1645 passed、8 skipped，55.88 s。
- 只读 migration audit 仍 internal_conformant=false，历史 empty-policy/depth-prior/remaining-tool authority 三项未满足；五份生成投影 current，未改历史 review/hash。compileall/git diff --check 通过。

同步 [提案发布边界](episode-budget-boundaries.md)、TODO 与近期交付文档。存储拒绝覆盖/严格身份行为待协作 review；hard-link 不支持时失败而非降级覆盖，配置目录祖先仍受信任。未实现 parent-directory fsync 的完整崩溃持久性，也未实现跨进程审批与 skill 应用事务；这些待办继续开放。未提交、推送、写共享文档或运行真实感知/仿真，完整 goal 保持 active。

## 实验恢复 R0 第二十一批 — 2026-09-07

本轮回到 HV-01，补充 preview seed 的明确传递验证与实际 Mink 短程诊断。按 OpenETA 协作技能只读核对 RFC 第 5 节，revision 2158；没有收到感知数据发送的新授权，也未重试被拒绝的感知实验。

- `controller_capability_canary.py` 新增可选 `--require-ik-seed`：仅限 Mink 非零运动诊断，从 reachable preview 提取七个有限关节值，构造带唯一 receipt ID 的执行 seed，要求运动回执 ID/policy 匹配。姿态容差显式对齐为 0.05 rad；既有碰撞覆盖、终点、步数和清理要求保留。
- 新增 10 项用例，覆盖 seed 原样传递、畸形/缺失关节在运动前拒绝、错误回执 ID/policy、OSC 模式误用；canary 定向组 51 passed。
- 使用已有隔离 Mink 依赖目录启动独立限时服务 18766，只新建 Object 0/seed 0。预览、seed 执行与回执检查通过：上移 2 cm，2/40 controller steps，终点欧氏误差约 1.91 mm；预览 world/self/场景及运动逐步配置检查未报告碰撞。新环境 close 确认后停止本次服务，退出码 0。
- 证据、命令、环境/session 与局限见 [Mink seed 诊断](diagnostics/mink-seed-canary-2026-09-07.md)。直接 Host 探针不等同主 Agent receipt resolver 验收；未验证复杂 IK 支路、接触、抓放或连续 swept-path 安全，也未切换默认控制器。
- Core 组（同前轮排除 integration、tools、manual proxy、historical migration-status）：1655 passed、8 skipped，54.96 s。只读 migration audit 仍 internal_conformant=false，历史 authority 三项未满足，五份生成投影 current。未修改历史 review/hash。

同步 TODO 和近期交付文档；HV-01 与完整任务回归继续开放。未提交、推送、写共享文档、操作已有用户服务或向外部感知/模型发送数据；完整 goal 保持 active。

## 实验恢复 R0 第二十二批 — 2026-09-07

上一轮实际 Mink seed canary 是有效进展；本轮检查主 Agent 的 receipt 引用链路。按 OpenETA 协作技能只读核对 RFC 第 5 节，revision 2158，保护已有工作树。未获得新的感知数据发送授权，未重试感知请求或启动 simulator。

复现与修复：

- 同 epoch、同目标的两份 feasible receipt 有不同 joint seed 时，选择第一份却会解析第二份 seed。另一个复现：选择 hard-infeasible receipt 后，同目标较新 feasible receipt 会被 gate 借用来放行。新增初始三项用例中两项失败，分别证明这两个缺陷。
- Seed resolver 现在绑定显式选中的 receipt ID，并先复核运动 gate；不再静默换用另一份等价目标的 seed。未指定 ID 的内部旧调用保留等价目标查询。
- Gate 要求所选 receipt 自身满足当前 epoch、目标/姿态和授权条件；不因后来的成功证据给失败引用补授权。较新的失败等价目标仍 veto 旧 feasible 引用，不能靠选旧 ID 回避新风险。
- 轨迹 gate 保留每个 waypoint 的 ID 绑定，检查显式 ID 列表长度；原有 collision-deferred 条件和错误码保持。Agent/MCP schema 不新增字段，授权修正作为个人候选待三人 review。

验证（组间重叠，不相加）：

- 最终新增 11 项实例，覆盖同目标 seed 选择、失败引用借用、较新失败 veto、非法 ID、逐 waypoint 授权，以及 StaticPlannerBackend → ToolCallingPlanner → runtime → pipeline → registry → simulator proxy 的正/负链路。成功链路核对选定 seed ID/joints、xyz、容差、步数、handle，以及 Agent 参数不混入私有 seed；失败链路未调用模拟 transport。
- 定向 remediation/memory-recovery/proxy：185 passed，4.37 s。
- Core 组（同前轮排除 integration、tools、manual proxy、historical migration-status）：1666 passed、8 skipped，55.15 s。compileall/git diff --check 通过。
- 只读 migration audit 仍 internal_conformant=false，三个历史 authority 检查未满足；五份生成投影 current，未修改历史 review/hash。

新增 [IK seed 绑定说明](ik-seed-binding.md)，同步 TODO 和近期交付文档。当前链路测试使用本地模拟 transport，不是新的 live 主 Agent/Mink 回归；前次直接 Host canary 不能替代本次代码的 live 验收。复杂支路和真实任务诊断仍待推进；未提交、推送、写共享文档或操作现有用户服务，完整 goal 保持 active。

## 实验恢复 R0 第二十三批 — 2026-09-07

上一轮两个 exact-reference 缺陷已修复，本轮推进该修复后的实际 simulator 派发证据。按 OpenETA 协作技能只读核对 RFC 第 5 节，revision 2158。仅运行自建本地 simulator，感知发送授权仍未收到，未重试相关推理。

- Canary 新增 `--through-agent-runtime`（要求 seed 模式且碰撞开启）；新 helper 使用实际 Host preview 经生产 compiler/记忆摄入生成 receipt，再由 StaticPlannerBackend 仅给出 receipt-ID move，执行生产 ToolCallingPlanner/runtime/pipeline/gate/registry/proxy 链路。
- 限一次工具准入；额外 transport guard 检查目标、容差、步数、环境、碰撞和 seed。RPC 前预留唯一派发机会，异常不重试；外层原有关闭、实际终点/步数/碰撞/seed 回执验证保留，另要求 pipeline executed。
- 独立 Object 0/seed 0 live probe 通过：2 cm 上移、2/40 步、误差约 1.91 mm，Agent 所选 receipt 与 Mink seed 回执一致，仅一次远端运动。新环境 close 确认后停止自建服务，退出码 0。完整 ID、命令、报告与源码指纹见 [后续 runtime 诊断](diagnostics/mink-seed-canary-2026-09-07.md)。
- 实验没有图像或实测 robot observation 输入 planner；preview 仍由 Host 发起并摄入，planner 为确定性输入，不是视觉决策或 proposal 选择验收。未验证复杂转腕/支路、抓放或完整任务，不切换默认控制器。

验证（组间重叠，不相加）：最终 canary 组 58 passed，0.54 s；新增 7 项覆盖实际 runtime 装配、必需选项、目标/步数/碰撞被改写时派发前拒绝，以及运动超时无重试且继续关闭。Core 组（同前轮排除 integration、tools、manual proxy、historical migration-status）1669 passed、8 skipped，54.55 s；最后四项 guard/timeout 测试在 core 启动后加入并定向通过，未回填核心计数，实现代码在 live/core 后未变更。compileall/git diff --check 通过；只读 migration audit 仍三个历史 authority 检查未满足、五份生成投影 current。

同步 TODO、近期交付与 seed 绑定文档，HV-01/整体验收保持开放。未提交、推送、写共享文档、操作已有用户服务或发送外部感知/模型请求；完整 goal 保持 active。

## 实验恢复 R0 第二十四批 — 2026-09-07

前轮取得实际 runtime/Mink 派发证据，本轮回到 HV-06 请求编排。按 OpenETA 协作技能只读核对 RFC 第 5 节，revision 2158。检查确认现有 lineage 仅表示父 session，不能安全推导父请求结束后的级联取消；主 planner 请求可能已经回答，随后工具才发起 advisor。因此没有把 answered 状态当作取消授权。

本轮实现可独立验证的角色覆盖缺口：

- Guidance、独立 action reviewer、visual-delta 现在从各自 Host 入口取得父 session，生成独立 child ID。新 helper 拒绝空/非字符串/超长/控制字符父 ID，不扫描嵌套 history 推断身份。Guidance 的私有父字段不原样复制到公开 session_context，只投影 request_lineage。
- OpenETA 控制台 adapter 限定新增 schema/role 组合，显示对应角色及人工等待提示；复用现有 parent 分组，完成后清空 wait label，取消某个 child 不影响 parent/sibling。未知/畸形角色不取得 lineage，缺少 Host 父 ID 时保留旧 fallback，不声称已修复所有旧请求的身份推断。
- 每次独立调用有新 child ID，同一次 backend request 的重试复用原 context/ID。该信息仅用于呈现，不是权限、执行生命周期、成本计账或取消租约；skill author/reviewer 等其他角色继续开放。

验证（组间重叠，不相加）：

- 新增 `tests/test_isolated_request_lineage.py` 16 项实例，覆盖非法父 ID、三角色队列分组与独立历史、拒绝未知角色、Host 父来源与新调用身份；补充 episode 到 guidance 的父 ID 传递测试，以及现有 visual-history 测试中的每次调用身份断言。定向 lineage/supervision/guidance/visual-history：60 passed，2.89 s。
- 新增真实 loopback guidance/backend/console 夹具，固定测试文本：父组中发现 pending child、等待提示正确、响应后恢复且清空提示。HTTP/console 组 37 passed，4.77 s；fixture server/thread 已清理，未查询或处理现有控制台请求。
- Core 组（同前轮排除 integration、tools、manual proxy、historical migration-status）：1690 passed、8 skipped，55.55 s。compileall/git diff --check 通过；只读 migration audit 仍三个历史 authority 检查未满足、五份生成投影 current，未修改历史 review/hash。

同步 [新增角色与限制](manual-vlm-harness-debugger.md#additional-isolated-roles-2026-09-07-candidate)、TODO、近期交付文档。新增 isolated prompt 字段/角色标签为个人分支待三人 review；未提交、推送、写共享文档或重启已有服务。没有实际任务/感知推理；感知发送授权、既有成功回归、Spatial 0/Long 9 诊断、级联取消和子请求计账仍开放，完整 goal 保持 active。

## 实验恢复 R0 第二十五批 — 2026-09-07

上一轮角色呈现有实际代码与 HTTP 证据，本轮收敛近期交付的扩大验证缺口。按 OpenETA 协作技能只读核对 RFC 第 5 节 revision 2158；先检查工具/integration 测试的服务调用条件，显式关闭九个真实模型 opt-in 开关，没有重试感知数据发送。

- `tests/tools`：587 passed、36 Pillow warnings，3.08 s。
- 合并 `tests`（包括此前分开的 HTTP/本机 SDK loopback，真实模型 integration 开关关闭）：2313 passed、21 skipped、1 failed、36 warnings，63.46 s。唯一失败为历史 authority/catalog 绑定；只读审计确认三个 authority 检查 false、五投影 current、structural_issues 为空。没有改历史 hash/断言或伪造 review。
- 已有 LIBERO Python + 最小 Mink overlay 补跑 MuJoCo bounds、sim codecs 和 CGN synthetic import-order：50 passed、2 个 trusted-local checkpoint 用例未选，4.67 s。未加载部署权重、调用感知推理或安装依赖。
- 实际 GPU cuRobo 测试首次 8 passed、3 setup errors，发现当前 Warp 1.16.0 没有旧 cuRobo MESH 调用的 warp.torch。生产 CollisionChecker 显式使用 PRIMITIVE，原直接 API fixture 却使用默认 MESH；不能将前者成功外推后者。
- 修改 `tests/test_curobo_world_integration.py`，将 fixture 显式参数化 PRIMITIVE/MESH，保留原覆盖及错误，新增生产组合的三项 GPU API 验证。复跑 11 passed、3 个 MESH setup errors，3.36 s；不 skip/xfail、不改第三方源码或依赖。

完整失败、skip 分组、命令和四份 JUnit 报告路径见 [本地验证矩阵](diagnostics/local-validation-matrix-2026-09-07.md)。门槛保持开放，尚非全绿；同步近期交付和 TODO，新增 mesh 兼容待办。compileall/git diff --check 通过。所有测试进程终止、自建 HTTP fixture 清理，没有操作现有服务或仿真任务；未提交、推送、写共享文档，完整 goal 保持 active。

## 实验恢复 R0 第二十六批 — 2026-09-07

用户明确补充了新仿真数据至指定 SAM3/AnyGrasp/AnyPlace 服务及框架测试信息至 LLM provider 的授权。此前 goal 因授权问题已标为 blocked；本轮授权阻塞解除并实际继续工作，但目标服务仍返回 blocked，不能通过只支持 complete/blocked 的工具伪造恢复状态。未缩小或重建完整目标。

- 新增可复用 `scripts/perception_regression.py`：生产 runtime + 真实配置 LLM，独立目录/环境，显式服务与工具白名单，跨角色请求上限、无 HTTP 重试/fallback、有限 episode 时间，报告清理异常。修正初版白名单遗漏及未创建/未知创建的清理报告区分。
- 保留两份启动失败基线；主请求重申既有命令 kind/参数契约，不放宽解析/安全 gate。真实 Spatial 0 创建→SAM3→歧义停止通过工具链，3 steps/6 LLM calls，221.713 s，未运动；真实 Long 9 创建→SAM3→选择成功，4 steps/6 LLM admissions，184.595 s，下一次 provider 60 s 超时，未进入 AnyGrasp/运动。两个环境均明确关闭，临时 simulator 以 SIGINT 正常退出（exit 0）。
- 先复现后修复 planner 校验耗尽被普通 talk 隐藏：Host 校验错误与次数进入 `planner_validation_failed`，episode/step truncated，批量分类 `EpisodePlannerFailure`。普通 talk、伪造 response 参数及既有成功证据规则保持测试覆盖。该终止呈现候选需三人 review，未改共享契约/旧 trace。
- 新增待办：空 talk 只在 reasoning 留解释；provider 超时转 ask_human 掩盖基础设施失败；未收到人工回答也因短暂等待标为 human_assisted。后两项在 Long 9 原始报告中得到实际证据，本轮尚未修复。
- 本地全量回归 2327 passed、21 skipped、1 个既有 authority/catalog 失败、36 warnings，63.31 s；真实 integration 显式关闭，独立 live episode 另计。没有改旧 authority hash 或审批。

完整 session/handle、预算、原始报告、代码边界和复用命令见 [授权后的纯 Agent 诊断](diagnostics/authorized-agent-perception-2026-09-07.md)。这不是完整成功路径回归；AnyGrasp/AnyPlace、运动恢复、Spatial/Long9 深层故障及整体验收仍开放。未提交、推送或更新共享文档。

## 实验恢复 R0 第二十七批 — 2026-09-07

Goal 已由产品恢复 active，完整目标不变。上一轮仅确认扩大的数据授权，没有新的代码/证据，按 no-progress 重新核对当前工作树后推进已知 live 缺陷。按协作技能只读核对 RFC revision 2158；不操作既有服务或历史 trace。

- 先复现 Long 9 暴露的错误：新增测试 6 failed、1 passed；LLM timeout 被算 ask_human，未回答等待 5 s 被算协助，零时长真实回答反而不算。
- Host backend failed 状态现在生成 `planner_provider_failed`，保留 provider 类型/错误码/重试性/尝试数，episode 与最后一步 truncated，不触发人工等待或 guidance。批量分类为 `EpisodePlannerFailure`。只保留模型原 fallback 命令作为 trace，不把其 ask_human 字面值作为 runner 状态。
- 人工协助依据本 episode 的非空 Host human_answer 或已收到的 manual-provider 响应，不依据等待时间/裸 resume；后续空回答不擦除已有协助，旧 episode 和 guidance_answer 不计作本轮人类回答。原时间预算不变；跨角色/跨重建归因仍开放。
- 失败 backend 缺 usage 时保留 unknown 来源计数，已知 total_tokens 不是完整账单。扩大定向测试发现最初实现误计确定性 backend 的 unknown，收窄到 failed backend 后修复；没有修改既有 guidance 断言迁就行为。
- 本机真实 HTTP read-timeout 夹具经生产 backend/planner/runner 验证；不调用外部 LLM、不诱发真实服务故障，不将夹具作为新 LIBERO 实验。
- 全量本地回归 2336 passed、21 skipped、1 个既有 authority/catalog 失败、36 warnings，64.43 s。真实 integration 显式关闭，JUnit `tmp/provider-failure-r0-27.oOJW3Z/local-tests.xml`；compileall 与 git diff --check 通过。所有测试进程终止，HTTP 夹具清理。

修改 `agent/runtime/episode.py`、`agent/runtime/parallel.py`，新增 `tests/test_provider_failure_episode.py`；同步 TODO、近期交付与 [统计边界文档](manual-provider-accounting.md)。本地结果分类候选需三人 review；未提交、推送或写共享文档。用户授权已扩展至 10.11.39.173 各类小模型和指定外部 provider，仍仅发送有界测试所需数据。下一步继续既有成功路径的真实回归，不能以本轮计账修复代替整体验收。

## 实验恢复 R0 第二十八批 — 2026-09-07

上一轮有代码与验证进展，本轮转向既有成功任务 Object 0 的实际完整尝试。扩展现有回归入口，显式任务模式只对 Object 0 开启运动/AnyPlace；保留默认无运动边界和原安全 gate，新增单次/累计 controller 额度检查、官方证据分类与清理条件。仍为不导入旧技能/历史/不启用 Python 的受限基线，不能冒充完整 CLI 配置。

两次真实同模型尝试均创建成功但在 SAM3 结构化输出校验耗尽，未发送感知推理或执行运动；分别 189.746 s/38137 tokens 与 91.708 s/37872 tokens，各 4 次 LLM 请求。新失败分类准确输出 planner_validation_failed、无人工等待，两个环境关闭确认。只读 provider model list 返回 8 个 ID，本轮没有换模型。

按第一轮出现的“修正字段又丢掉合法引用”补上有界、分离副本、明确不可信的上一份已解析候选回显；解析失败/超大内容不伪造回显，不自动修参/授权，不改变 isolated JSON 协议。相关 160 项测试通过，但同模型 live 复测仍失败，因此不宣称修复带来任务成功或服从性提升。下一步做可用模型对照，避免同条件反复消耗。

全量本地测试 2345 passed、21 skipped、1 个既有 authority/catalog 失败、36 warnings，61.78 s；真实 integration 关闭。详见 [两轮报告和复现边界](diagnostics/object0-task-regression-2026-09-07.md)。两个客户端、自建服务及测试进程均正常结束，清理自有资源，未重启用户服务。compileall/git diff --check 通过；未提交、推送或修改共享文档。完整任务回归仍未成功，goal 保持 active。

## 实验恢复 R0 第二十九批 — 2026-09-07

按协作技能只读核对 RFC revision 2158，本轮不改代码/用户配置，只对独立 Object 0 任务进程覆盖模型为 provider 已列出的 `gpt-6-astra`，保持上一轮任务/工具/预算和无 fallback 条件。首次请求在约 75.181 s 后收到 HTTP 500 / `server_is_overloaded`，没有创建环境或得到模型动作输出；不能比较模型任务能力。

原始报告 `tmp/perception-regression-nxiy19q0/report.json`：1 provider admission、0 tool admissions、0 controller 预留，task_outcome=fail。R0 27 分类在真实外部错误下输出 planner_provider_failed，未进入人工等待/协助；usage 缺失记 unknown=1，不将已知 tokens=0 当作免费。cleanup 明确 no_create_attempt，而非伪称环境关闭。

客户端 exit 0 表示写出报告；自建 simulator PID 1256996 正常 SIGINT exit 0，18766 端口与准确 PID 均核实已释放。未操作旧服务、发送源码/历史 trace，未提交、推送或写共享文档。更新 Object 0 实证、TODO、近期交付说明；本轮没有代码改动或新全量测试，沿用上一轮测试证据。完整任务成功路径仍未完成，goal 保持 active；后续可做另一可用模型的有界对照，不在同条件下无限重试上游过载。
