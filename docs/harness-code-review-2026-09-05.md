# OpenETA 代码审查 — 2026-09-05

审查对象是 `OpenETA/` 的当前工作树，包含已有未提交改动。分支 `dev/huaizezheng/human-vlm-stability`，HEAD 为 `fa56247`。没有修改项目源代码、测试或生成契约文件，没有调用真实模型或运行机器人。复现脚本保存在 `docs/diagnostics/harness_review_2026_09_05.py`；运行产物写入独立临时目录。共享 RFC 在前一轮因代理访问限制未能读取，本报告以本地实现为依据。

重点检查：Runtime / EpisodeRunner、Planner / Pipeline、Memory、ToolRegistry、PythonExec、MCP 生命周期、并行评测和契约迁移。没有逐行审计全部代码，也没有验证各仿真器的真实物理行为。

整体判断：项目已有明确的模块边界和大量测试，但有几处“文档和字段表达了隔离/可信/取消语义，实际写入或执行边界没有完整保证”的问题。优先修复这些行为问题，再拆分大模块。

优先级：P1 表示应优先修复的执行、数据或评测正确性问题；P2 表示随后处理的可靠性与维护问题。以下保留发现时的未修复状态；当前实现与验证进展另见[重构执行记录](harness-refactor-progress-2026-09-05.md)，避免把历史复现结果当作当前状态。

## 验证结果

- 第一组针对性测试：109 passed，属于随后广泛测试的子集。
- `pytest -q tests --ignore=tests/integration --ignore=tests/tools --maxfail=8`：1086 passed、7 skipped、5 failed。四项失败是沙箱禁止本机 socket，一项是工具契约迁移检查失败。
- 放开本机 socket 后单独执行 `tests/test_manual_vlm_proxy.py`：22 passed，确认上述四项不是代码缺陷。
- `pytest -q tests/tools --maxfail=5`：530 passed、36 warnings。警告来自 Pillow `getdata()` 弃用。
- 对上述测试按用例去重并采用端口测试重跑结果：1620 passed、7 skipped、1 failed；这是多次执行的合并结论，不是一次完整测试命令的输出。
- 另外执行了审查诊断脚本，确认下列边界行为。脚本中的断言用于确认当前缺陷存在，脚本成功不表示实现正确。

## 1. P1：Agent 可通过普通记忆工具修改宿主权威状态

位置：[memory.py:845](../agent/runtime/memory.py)、[runtime.py:278](../agent/runtime/runtime.py)、[memory.py:7289](../agent/runtime/memory.py)。

`save_memory` 最终调用 `save_fact`，后者首先写入 `self.facts[key]`，随后才把来源为 `save_memory` 的条目复制到 `agent_working_state`。只拦截已废弃任务状态机的键，没有限制现行宿主状态键。`delete_memory` 同样可以删除 `self.facts` 中的宿主条目。

复现走过真实的 `ToolCallingPlanner → ActionPipeline → ToolRegistry → handler`：先由宿主设置 `robot_motion_epoch=7`，随后模型调用 `save_memory(namespace='facts', key='robot_motion_epoch', content={'epoch': 0})`，返回 `pipeline_status='executed'`，实际 epoch 变成 0，来源变成 `save_memory`。

影响：IK 和观测证据依靠 epoch 判断有效期。模型笔记能改变用于判断的权威状态，使这些检查失去可信前提。已确认能修改计数器，未在真实机器人上尝试借此执行过期动作。

建议：物理分离 `HostEvidenceStore` 和 Agent 笔记存储；模型工具只写后者。解析器只读取宿主证据。对修改、删除、恢复会话、artifact 引用都执行同一所有权规则，不能只在提示词里要求模型遵守。

## 2. P1：Python 沙箱没有实现所声明的文件边界

位置：[coding.py:184](../agent/tools/coding.py)、[coding.py:598](../agent/tools/coding.py)。

代码限制了内置 `open` 和部分 `Path`，但 `_safe_import` 直接返回完整 NumPy 等模块。这些模块自己的文件 API 不经过包装后的 `open`。

复现：在 `sandbox` 模式、`allow_outside_sandbox=False`、配置了独立 session/sandbox 路径的条件下，执行 `numpy.savetxt` 和 `numpy.loadtxt`，成功在 session 目录外创建并读取数值 314159。目标仅为审查专用临时文件，没有访问凭据或用户文件。

同一执行路径没有使用 `default_timeout_s`。将它设为 0.01 秒，运行一个有界 0.08 秒循环，仍返回成功。若由 episode 线程取消，只是停止等待，不能终止正在执行的 Python。

建议：将生成代码放入可终止的独立 worker，并由操作系统落实文件、网络和资源约束。子进程本身不等于文件沙箱；不要仅继续补 import 黑名单。把结果、stdout、artifact 清单作为显式返回数据。

## 3. P1：取消执行后，旧 handler 仍可污染新会话

位置：[registry.py:1188](../agent/tools/registry.py)、[episode.py:743](../agent/runtime/episode.py)、[openeta_cli.py:1237](../agent/cli/openeta_cli.py)。

有取消标记时，工具 handler 会在额外的 daemon 线程运行。取消只使调用方抛出 `_ToolHandlerAbandoned`；handler 自身还在运行。`wait_for_idle()` 跟踪的是外层 episode worker，不包含该工具线程。

复现：用事件阻塞真实 `save_memory` handler 的执行，发出 interrupt；外层结束且 `wait_for_idle()` 返回 true；启动新会话，再释放旧 handler。旧笔记实际进入新会话。这里注入的是可控调度延迟，最终执行的是项目原有记忆写入逻辑。

建议：所有提交携带不可变的 session/execution generation，提交时检查所有权；工具生成结果或待提交变更，由拥有当前执行权的线程提交。对无法取消的工作保留任务登记并隔离其可变引用；Python 等可进程化任务应真正终止 worker。仅丢弃返回值不足以撤销副作用。

## 4. P1：正奖励被泛化为任务成功，可能污染评测结论

位置：[parallel.py:451](../agent/runtime/parallel.py)、[parallel.py:479](../agent/runtime/parallel.py)。

对不要求 official reward 的环境，只要任一步 `reward > 0`，分类器就返回 success；没有要求对应任务成功，也没有让明确的失败信号覆盖该推断。默认仅通过 env_id 是否包含 libero 来选择更严格规则。

复现输入：`reward=0.1`、`task_success=false`、`terminated=false`，episode 已因 timeout 截断。以 MetaWorld 形式的 env_id 调用分类器，输出仍为 success。

影响：只要接入具有正向中间奖励的环境，就可能把取得进展的失败轨迹计为成功。没有据此断言历史 LIBERO 结果错误，也没有重算历史成功率。

建议：每种后端声明成功语义，生产统一的、带来源的 success verdict；区分 `reward`、`task_success`、`terminated`、`truncated`。二值奖励可作为成功信号的前提应由明确契约给出，不能对通用奖励隐式套用。训练候选筛选与评测汇总共用这套判定。

## 5. P1：并行 Python 调用会把输出写入其他会话

位置：[coding.py:184](../agent/tools/coding.py)。

`redirect_stdout` 修改进程级 `sys.stdout`。并行 harness 是线程池，每个工具又可能使用线程，所以两个 session 的 Python 调用会争用同一 stdout。

复现：A 开始打印；B 进入 Python 执行；A 再打印。A 的 `A-during-B` 出现在 B 的 ToolResult stdout，而 A 自己的结果缺失该行。使用事件严格控制交错，完成后恢复原 stdout。

影响：模型可能看到另一 episode 的计算输出，trace 和训练样本也会串会话。

建议：进程级执行隔离能够同时解决输出、取消和部分资源问题。短期可以提供局部 print sink，但它无法自动捕获第三方库对全局 stdout 的写入，因此不应当作完整修复。

## 6. P1：环境关闭失败被丢弃或误报成功

位置：[sim_mcp.py:369](../agent/tools/sim_mcp.py)、[server.py:2742](../sim/mcp_server/server.py)、[worker_mgr.py:342](../sim/mcp_server/worker_mgr.py)。

客户端 `SimulatorMcpEpisodeEnvironment.close()` 在发送请求前清空句柄。注入一次关闭超时后，第一次 close 返回失败，第二次返回 `ok=true, skipped=true`，远端关闭调用只有一次。

服务端 `close_env` 只把抛出的异常计入 `cleanup_errors`。但 `BenchWorkerHandle.proxy` 会把 HTTP/网络异常转换成 `{'error': ...}`。注入这样的 worker 返回值后，顶层仍返回 `ok=true`，并删除本地映射、释放 worker 计数。

影响：远端资源可能仍存在，客户端却失去正常重试路径；环境数量和 worker 的引用计数也可能偏离事实。

建议：显式维护 active/closing/close_failed/closed 的资源生命周期；失败保留清理记录，只有确认关闭或确认不存在才退休句柄。worker transport 统一返回类型，调用方必须检查业务失败；保留关闭请求的幂等身份和有界重试。

## 7. P2：episode 的时间预算没有覆盖全部模型工作

位置：[episode.py:377](../agent/runtime/episode.py)、[episode.py:654](../agent/runtime/episode.py)、[supervision.py:347](../agent/runtime/supervision.py)、[self_improvement.py:348](../agent/runtime/self_improvement.py)。

受超时保护的 episode worker 返回后，主线程同步执行 guidance resolver。复现把 episode timeout 设为 0.04 秒，resolver 睡眠 0.12 秒，整个调用约 0.128 秒后才报告超时。真实 resolver 会调用模型，且其返回详情没有保留 usage。

结束后的 self-improvement review 同样同步执行；静态检查可见 review 的异常没有在 `continue_run` 中隔离，可能让已产生 episode result 的调用整体失败。这一后处理路径未做本次动态故障复现。

建议：把总 deadline、cancel token、token usage 聚合下沉至统一调用上下文，并给后处理独立预算。后处理失败应作为结果附属信息保留，不能吞掉已完成的任务结果。工具调用数目前在执行后用 `>` 检查，允许越过上限；如该字段旨在表示硬预算，应在 dispatch 前预留调用配额。

2026-09-07 partial：guidance 等待纳入剩余 episode 时间，隔离输入/迟到提交并保留正常返回的 usage；review 异常不再吞掉结果，预算或 interrupt 停止时不启动新 review。后续第十批为 scoped executable tools 增加原子 admission，超额调用在 authorization/handler 前拒绝，覆盖 batch/嵌套及保守续跑计账；历史 attempt count 与实际 admission 分别呈现。详见 [当前预算边界](episode-budget-boundaries.md)。统一 provider cancellation、未绑定配额的直接调用、正常 review 独立预算及真实运行验收仍未完成，不以局部修复关闭第 7 项。

## 8. P2：并行 MCP 请求竞争进程级代理环境变量

位置：[sim_mcp.py:1957](../agent/tools/sim_mcp.py)。

每个 SSE 请求临时改写 `NO_PROXY/no_proxy`，结束时恢复旧值。两次请求重叠时，恢复操作没有嵌套关系保证。

可控交错复现：A 进入、B 进入、A 退出；此时 B 仍在执行，但 B 的 host 已从 NO_PROXY 中消失。随后 B 退出，环境变量却残留 A 的 host。复现只改动独立测试进程的环境，没有发网络请求。

建议：将是否使用代理作为 HTTP client/transport 实例配置；避免请求运行期间修改 `os.environ`。

2026-09-07 修复候选：SSE list/call 现通过 SDK client factory 使用准确目标 host 的 direct mount，不再写 `NO_PROXY/no_proxy`；其他 host 的环境代理和环境 CA 校验保留。交错线程、真实 HTTPX 路由、超时清理及本机实际 MCP roundtrip 已验证，详见 [代理隔离边界](mcp-proxy-isolation.md)。这不代表其他 HTTP 后端或所有真实感知部署已完成回归。

## 9. P2：trace 尾部损坏会阻断整个会话恢复

位置：[memory_store.py:162](../agent/runtime/memory_store.py)、[memory_store.py:113](../agent/runtime/memory_store.py)。

`load_events` 对每一行直接 `json.loads`。在正常 trace 末尾追加一个模拟进程中断的半行 JSON 后，真实 `resume_session` 抛出 `JSONDecodeError`，无法恢复前面的有效记录。其他某些读取路径虽然会跳过坏行，但这里没有统一恢复策略。

此外，working memory 分别替换 facts、agent state、artifacts 等五个 JSON 文件；每个文件的替换是原子的，整个快照不是一个事务。中途崩溃可能留下混合代次，这是静态发现，未在本轮动态模拟。

建议：带序列号的追加日志 + 明确的快照 generation。尾部半记录可隔离并报告恢复信息；中间损坏应明确失败，不能静默跳过后仍宣称完整。快照可写入新 generation 目录后原子切换 manifest，也可评估 SQLite 事务。

## 10. P2：当前契约改动未完成验证证据更新

位置：[test_tool_contract_migration_status.py:22](../tests/test_tool_contract_migration_status.py)、[tool_contract_migration_status.py:189](../agent/evals/tool_contract_migration_status.py)。

这是现有测试中唯一确认的实际失败。当前 catalog hash 为 `c9437fe6…`，两组 canary 仍绑定 `d957c3a3…`，authority baseline 也未通过一致性检查。审计返回 `internal_conformant=false`，失败项为：

- empty_policy_authority_baseline_safe
- estimate_depth_prior_authority_canary_safe
- remaining_tool_authority_canary_safe

与此同时，catalog JSON、Markdown、readiness 等生成投影都被判为 current，结构检查没有错误。这说明当前工作树的契约声明已更新，但相应验证证据还没有闭合；该结论包含用户已有未提交改动，不应当归咎于之前的已提交版本。

建议：按变更范围重新执行生成器和真实验证，保留历史审阅证据与版本绑定；不要手动替换历史 canary 的 hash，或只改测试断言使其通过。把目录声明、请求投影、验证产物是否对应同一版本作为自动质量门禁。

## 结构与性能改进方向

这些是基于当前代码组织的设计建议，不是已测量的性能结论。

1. 拆分 Memory 的责任。`memory.py` 有 9899 行，同时承担会话、观测包、抓取、放置、IK、状态有效期和持久化。优先抽出宿主证据存储与 Agent 笔记边界，再拆出 ObservationPacketIndex、GraspEvidence、PlacementEvidence 和会话持久化。保留统一的事件提交入口，避免拆文件后写入更分散。
2. 收敛工具接口的重复定义。`planner.py` 5385 行，仍有 `_STATIC_TOOL_PARAMETER_RULES` 和 legacy validators；ToolSpec、ToolContract、handler 又各自携带部分接口知识。当前渐进迁移和保留 legacy gate 有明确理由，应逐工具建立行为等价验证，再迁移到共同结构，不能直接把全部 authority 开关打开。
3. 让处理顺序可检验。`AgentMemory.add_action` 串联多种捕获、刷新和失效逻辑；其中已有注释说明 AnyPlace materialization 与 refresh 的顺序会改变行为。建议显式的 typed event/reducer 输入和依赖关系，增加跨工具序列测试，减少依靠函数排列维护不变量。
4. 度量后优化存储。每次 `append_event` 都加锁读取、重写整个 session_index；每次工作记忆保存写五个文件。并行 session 增长时会形成全局串行 I/O。应先测每回合本地开销、文件写入量和索引锁等待，再做索引更新合并、增量持久化或事务存储。
5. 将后台清理移出事件循环。`session.py:208` 的 async sweeper 直接调用同步 `_cleanup_session`，后者发阻塞 worker HTTP 请求，默认超时可达 600 秒。清理应受有界后台任务控制，并与同一环境的控制操作共享生命周期协调；不能只把函数放进线程而忽略共享字典和锁。该点本次仅做静态检查。
6. 增加行为边界测试。现有 1600 余项通过用例并未覆盖上述输出串会话、晚写、正奖励假成功、关闭业务错误等情况。应补确定性交错、故障注入、崩溃恢复和多工具序列测试。字符串扫描与固定 hash 测试可保留作补充，行为保证应依赖真实调用边界的验证。

## 建议实施顺序

第一批：宿主记忆所有权、Python 执行隔离、取消后的提交所有权。它们共同决定模型和旧执行能否越过宿主边界。

第二批：后端成功判定、环境关闭的失败与重试语义、并发 stdout 与代理状态。修复后再使用成功率比较模型或筛选训练轨迹；对可能受影响的历史样本另行重算，不直接推定全部失效。

第三批：恢复事务、完整预算、Memory/ToolContract 的渐进重构和存储 profiling。每批围绕可验证的不变量提交小改动，保留当前功能基线。

## Issue #12 对照：操作接口、几何语义和证据生命周期

来源：[OpenETA7/Stage2 issue #12](https://github.com/OpenETA7/Stage2/issues/12)，标题为「[Epic] Make the perception → grasp → place contract compact, backend-neutral, and drivable by both humans and VLMs」。2026-09-05 通过 GitHub CLI 读取正文及两条评论；读取时 issue 为 OPEN，作者 SII-ZhangYiFei。评论包含轨迹 ZIP 附件及协作者提醒，本次未下载附件或独立重算轨迹统计。

版本边界：issue 记录的是 `e0e886f` / `stage2/feat/general-manipulation-resources` 加较大未提交原型；本轮审查的是 `fa56247` / `dev/huaizezheng/human-vlm-stability` 加当前未提交改动。不能把另一工作树的每项观察直接视为本地当前状态。

### issue 提供的实验依据

据 issue 正文，四条人工 VLM LIBERO 成功轨迹共调用工具 144 次，其中 `move_to` 44 次、`observe` 45 次，22 次运动未到达目标。四条轨迹均收到正奖励和终止信号，但仍依赖人工选择、修正抓取姿态及手工放置坐标；唯一一次尝试的 `estimate_placements` 在进入 AnyPlace 前就被 provenance gate 拦截。

正文还报告：100 次实际人工规划决策估算共消耗约 700 万输入 token，平均每次约 7 万；审计导出约 4 MB、近 10 万行。这些数字是 issue 作者的测量结果，本轮未独立验证。

这些证据与本报告第 4 项关注的层次不同：一条任务确实成功的轨迹，仍不代表自动抓取/放置工具链完整可用。后续应分别记录任务成功、人工协助、手写目标使用和放置估计器实际执行情况。

### 当前代码对照

| issue 主题 | 当前分支检查结果 | 后续处理 |
|---|---|---|
| 默认反馈过大，模型承担路径和底层参数拼装 | 已有 ToolContract 公共投影、artifact 引用及上下文预算，不能描述为完全没有精简；尚未测量当前分支真实请求体积 | 在 provider 请求边界记录各字段 token、图像数和重复引用，再统一紧凑决策反馈与完整审计数据 |
| 没有新图像仍触发视觉差分 | `VisualHistoryManager.observe()` 已计算图像 SHA-256，相同内容直接返回 `no_visible_change`；已有针对性测试 | 保留现有修复。失败/不可用结果目前按 observation index 对生成的 delta_id 去重，仍需验证跨 observation 的同一证据对能否去重 |
| GraspGenX 原点转换不完整 | `normalise_grasp_candidates()` 的 4×4 变换仍只设置旋转；fingertip 单独计算，并未写入标准化位姿平移 | 列为 P1 几何契约核查项；依据具体夹爪定义和参考坐标建立完整 SE(3) 测试。issue 的约 93.4 mm 是其测试夹爪的观察，不应直接作为通用常数 |
| 抓取接触点需要局部修正 | 当前有 compiled grasp residual 和预算机制；注册表未发现 `move_along_approach` 公共工具 | 评估按 grasp/world approach vector 表达有界相对运动；保留实际位移、碰撞和残差反馈，不把 candidate.depth 当通用插入修正量 |
| SAM3 selection 过早绑定消费角色 | 当前 `select_sam3_detection` 仍暴露 `evidence_role`，AnyPlace 仍读取 `placement_region` selection | 将 detection 作为可复用资源，消费动作再指定用途；同一结果允许保留物体与目的地等多个 selection |
| AnyPlace 抓取前证据包在物体移动后不可用 | 已有 bundle、固定相机证据复用和对 materialized placement plan 的部分保留；未满足条件时仍清空 active bundle | 区分「输入证据不可变」「估计结果已生成」「结果当前是否可执行」。补先冻结输入、后抓取、再首次调用放置估计器的端到端验证 |
| 简单资源 ID、统一执行坐标、按需展开审计 | 已有观测包、短 ID、宿主解析器等基础设施；issue 所列 `estimate_grasps` / `estimate_placements` / `select_detection` façade 不在当前公共注册表 | 逐工具推进同一套资源接口，让简单 ID 路径和专家覆盖共享宿主验证；执行位姿显式标注 world/task/base frame |
| 全场景分割及多视角抓取预览 | issue 提供了具体交互设计，当前分支完整覆盖程度未在本轮验证 | 根据后端能力提供候选总览和多选；用带标定来源的指爪/接触通道多视角渲染辅助选择，避免把该建议直接当作所有后端现成功能 |

代码入口：[visual_history.py](../agent/runtime/visual_history.py)、[graspgenx_core.py](../tools/graspgenx_core.py)、[memory.py](../agent/runtime/memory.py)、[registry.py](../agent/tools/registry.py)、[planner.py](../agent/runtime/planner.py)。

本次对照额外运行 `tests/test_visual_history.py::test_identical_main_view_skips_vdm_but_preserves_delta_coverage` 和 `tests/test_placement_evidence_reuse.py`：10 passed。它们验证当前已有行为，不代表 issue 的全部验收条件已完成。保存到 docs 后的诊断脚本也重新运行，11 个复现检查均输出预期缺陷表现，没有 `*_probe_error`。

### 对实施顺序的补充

原报告的执行所有权、隔离和评测可靠性问题继续优先处理；同时应把 GraspGenX 完整坐标转换加入最早的几何正确性核查。资源接口与反馈精简可作为另一条实现工作线，但不能只增加一层新工具名而保留多个相互不一致的底层契约。

AnyPlace 的输入一致性要求应由后端能力声明，并由宿主维护可复用的不可变证据；运动前的安全有效性仍需重新检查。保留历史输入，不等于允许历史运动授权永久生效。提前暴露缺失视角或证据的条件，也不等于让宿主自动编排整个抓放任务。

后续验收除了奖励和终止，还应检查：无重复图像差分调用、默认反馈体积、无需人工复制底层参数、完整坐标转换、冻结输入跨抓取使用，以及至少一条实际调用放置估计器完成的轨迹。上述实验验收本轮尚未执行。

<a id="human-vlm-backlog"></a>

## Human VLM 实验补充：待改动项 HV-01 至 HV-09

来源：2026-09-05 用户提供的测试 Agent 总结。本次仅整理待办并做局部代码核查、trace 路径存在性检查，没有重新执行任务、逐事件审计或重算轮次。下表的结果、耗时和步骤数均按该总结记录；不能把这批有人协助的结果视为自主成功率或 LIBERO 完整覆盖。

### 任务与证据基线

| 任务 | 代表 session / 本地 trace | 报告结果 | 验证内容与成本 |
|---|---|---|---|
| Goal 0：打开中间抽屉 | [9edd6d5b-9fa4-4322-b943-c50e835e5785](../.openeta_memory/sessions/9edd6d5b-9fa4-4322-b943-c50e835e5785/trace.jsonl) | 成功，reward=1、terminated=true | 把手定位、分割、抓取编译、接触、attachment probe、拉动；43 episode steps |
| Object 0：alphabet soup → basket | [e4e91a6a-4670-40e4-918f-6a713c47b6a5](../.openeta_memory/sessions/e4e91a6a-4670-40e4-918f-6a713c47b6a5/trace.jsonl) | 成功，reward=1、terminated=true | 感知、抓取、携带、AnyPlace、路径、释放；48 episode steps |
| Long 0：两个物体 → basket | [3e3580c6-934e-40a9-9071-f2b80ab63d1e](../.openeta_memory/sessions/3e3580c6-934e-40a9-9071-f2b80ab63d1e/trace.jsonl) | 成功，reward=1、terminated=true | 两套顺序 pick-place；99 episode steps，报告有 19 次 IK preview、19 次 move、19 次 observe |
| Spatial 0：关系指定的黑碗 → 盘子 | [f273b9d5-f065-49c2-b919-80c51afea997](../.openeta_memory/sessions/f273b9d5-f065-49c2-b919-80c51afea997/trace.jsonl)、[d26365f1-812e-4b36-8379-dce85bb25c5d](../.openeta_memory/sessions/d26365f1-812e-4b36-8379-dce85bb25c5d/trace.jsonl) | 未成功 | 关系指代、遮挡、视角和运动执行共同阻塞，不能在未重放前归结为单一原因 |
| Long 9：杯子 → 微波炉并关门 | [4ae4d4e5-6129-44d9-8c34-17f7d9bad016](../.openeta_memory/sessions/4ae4d4e5-6129-44d9-8c34-17f7d9bad016/trace.jsonl) | 未成功，主动暂停 | 曾抓起并运到微波炉附近，释放失败后进入恢复；203 episode steps、约 160 个主 planner turns |

以上六个完整 session 路径均在记录时确认存在；`.openeta_memory/` 是 gitignored 本地运行数据，这些链接不会随文档自动发布，其他机器需另行取得 trace。后续复现应从 trace/manifest 提取 seed、实际控制器、运动配置、模型/人工模式和版本，不能仅按任务编号假设环境相同。

已有 [family canary 记录](human-vlm-libero-family-canary-2026-09-04.md) 提供 Object 0、Long 0 和部分故障上下文。该记录保留当时状态；本节补充其后用户报告的 Goal 0 / Long 9 结果。issue #12 的“放置估计器零次成功”只属于其四条原始轨迹，不能泛化到本批已成功运行 AnyPlace 的任务。

### 待办总览

状态均为未完成：已存在部分机制不等于整个问题已经修复。P1 项优先于进一步长时诊断实验；P2 项按依赖跟进。字段和枚举名称均为接口提案，落地时应更新共享契约并标记需协作者三方评审。

| ID | 优先级 | 待改动主题 | 主要关联 |
|---|---|---|---|
| HV-01 | P1 | IK preview 与实际控制器执行语义对齐 | 控制器能力、IK receipt、执行 seed；Spatial 0 / Long 9 |
| HV-02 | P2 | 显式表达容差内零步动作 | 审查第 4 项成功语义、第 7 项预算；issue #12 紧凑运动反馈 |
| HV-03 | P1 | 分割/抓取候选前的主动感知与多方位 wrist viewpoint | issue #12 多视角几何、观察证据；Spatial 0 / Long 9 |
| HV-04 | P1 | 分离 object identity 与 graspable part | issue #12 消费者无关 selection、AnyPlace 证据 |
| HV-05 | P1 | 几何类型的 schema/gate 共用定义 | 审查第 10 项契约一致性；依赖 HV-04 类型设计 |
| HV-06 | P2 | Human VLM 父子请求关联与等待状态 | 审查第 7 项时间预算；人工工作台 |
| HV-07 | P1 | 封闭腔体放置与释放后验证 | issue #12 放置生命周期；Long 9 |
| HV-08 | P1 | attachment probe 采用一致的当前证据 | 审查第 1 项宿主证据所有权；恢复和夹爪状态 |
| HV-09 | P2 | 分析并降低可避免的轮次成本 | HV-01/02/03/06、issue #12 反馈/视觉差分、证据复用 |

<a id="hv-01"></a>

### HV-01：IK preview 与控制器执行契约

实验观察：preview 可找到可达且关节余量良好的解，但 OSC 未进入对应支路；Spatial 0 目标与实际终点偏差大，Long 9 通过多次小步转腕避开关节限制。

当前代码核查：[sim_mcp.py](../agent/tools/sim_mcp.py) 已调用 `_ik_execution_seed_resolver`，而 [server.py](../sim/mcp_server/server.py) 在 `openeta.worker_mink_goal.v1` 分支中传递 seed。应调查失败 session 实际使用的 executor、seed 是否被消费及路径约束，不能描述为所有 `move_to` 路径都忽略 IK seed。仅静态发现传递也不能证明能实现指定支路。

拟改动：分别表达数学可达性、当前控制器局部可执行性与实际执行结果；声明 seed 消费能力、预览模型/控制器及容差版本。无法保证局部执行的预览应返回 unknown 或明确限制。若执行器消费 seed，须验证它与目标、起始状态和预览证据一致。

- [ ] 验收：构造全局 IK 有解、局部 OSC 不收敛的用例，preview 不再被解释为运动必达；Mink/OSC 能力与 receipt 可区分。
- [ ] 验收：记录 preview/execute 的目标、关节支路或 seed 摘要、实际终点、残差及 stop reason，覆盖关节极限、过期 seed 和局部不可执行情况。
- [ ] 重测：优先 Spatial 0 的偏差动作、Long 9 的转腕恢复；逐步升级到整任务，不通过无限增加 controller steps 隐藏不一致。

<a id="hv-02"></a>

### HV-02：容差内零步动作的显式语义

实验观察：Long 9 请求约 0.09 rad 转腕，姿态容差为 0.15 rad，返回已经在容差内并执行 0 步。这可以符合绝对目标 API 的定义，但无法完成调用者期待的调整。

当前代码已有 `_motion_already_within_tolerance`、`motion_outcome='no_state_change'`。待办是统一并完善反馈，保留已有语义，避免重复建设。

拟改动：在共享 motion receipt 中明确表达 `reached_before_execution`、`controller_steps`、`world_state_changed` 或等价字段；区分目标已满足、实际执行到达、执行未达和结果未知。目标差小于容差时给出收紧容差或明确相对动作的建议，不自动修改调用者目标。

- [ ] 验收：覆盖 0.09 rad / 0.15 rad 的零步情况及收紧容差后的实际转动，调用者可从紧凑反馈区分。
- [ ] 验收：已确认零步且状态未变时，不计为有效运动进展或无端推进 robot epoch；缺少完整回执时仍为 unknown，不能直接判定未变。

<a id="hv-03"></a>

### HV-03：前置主动感知与多方位 wrist viewpoint

实验观察：Long 9 的中心、+x、-x 视角均遮挡杯把手，切回不同 y 方位的历史位姿后把手可见。Spatial 0 的遮挡也可能受益，但因果关系仍待重测。

当前代码确认：[grasp_geometry.py](../agent/tools/grasp_geometry.py) 的候选使用 `lateral_offsets=[0,-0.04,0.04]`，位置为 `[target_x+lateral,target_y,target_z+standoff]`，未覆盖目标周围不同 y 方位。公开契约和 `resolve_wrist_viewpoint_input` 还要求 `compiled_grasp_id`，因此缺口不仅是视角数量，也包括首次分割或抓取候选失败时无法启动。

明确需求：支持抓取编译前的主动感知，保持固定 agentview 的任务语义参照、跨视角身份连续性和 Agent 决策权。用户提出复用 `propose_wrist_viewpoints`、增加 `identity_anchor_id` 输入及投影 handoff，但明确尚未确定实现方式；不能将这些字段直接视为已批准契约。详见[主动感知需求与候选设计](active-perception-design-options-2026-09-05.md)。

候选方向：按显式参考系采样方位角、俯仰角和距离，结合相机标定得到 EEF 位姿。Host 生成有界且几何多样的集合并做粗筛，Agent 自主选择；排序信息不能替代精确 IK、碰撞与执行 gate，也不触发自动扫描。优先评估扩展现有工具，但须先解决“首次 SAM3 失败时身份锚点从何而来”和“实际新观测产生后如何绑定投影”两个设计问题。身份确认与运动后观察 handoff 已有部分实现，应复用而非重复建设。

- [ ] 设计验收：分别覆盖有 mask 无 compiled grasp、首次 SAM3 无检测、深度不足；明确身份锚点/临时定位证据的创建和升级规则，不以成功分割作为所有恢复路径的前提。
- [ ] 验收：候选覆盖多个 x/y 方位和俯仰，坐标旋转后仍有相同几何意义，均可正确重投影。
- [ ] 验收：新视角投影绑定实际新 packet、实测相机位姿与原身份来源；相似物体、遮挡物深度和物体移动不导致静默换目标，`same_instance` 不被当作自动几何证明。
- [ ] 验收：目标只在侧后方可见的场景可产生候选；所有候选不可用时报告具体限制。保持可复制位姿或可执行的稳定引用，避免候选数值在反馈中被省略。
- [ ] 重测：Long 9 把手遮挡、Spatial 0 关系指代与遮挡。

<a id="hv-04"></a>

### HV-04：物体身份与可抓部件分离

实验观察：整杯 mask 产生的 20 个 AnyGrasp 候选均被可见闭合跨度检查拒绝，最小约 9.74 cm，夹爪上限 8 cm；advisor 正确 abstain。把手可见后改分割黄色把手，才得到可能可抓的区域。这不是放宽夹爪宽度检查的理由。

拟改动：object instance ID 保持同一只杯子的身份；graspable-part selection 表示杯把手/根部等局部候选，并显式关联所属物体、来源观测和掩码。抓取宽度检查消费局部几何，携带碰撞、目标跟踪和放置消费整物体几何；同一个 mask 不能同时默认承担这两种角色。

- [ ] 验收：整物体超宽时可选择可抓部件继续规划，身份保持不变，证据有明确 part→object 关联。
- [ ] 验收：只看到把手时，不把把手尺寸当成携带物整体碰撞包络；杯体/锅体/工具整体的放置和障碍检查仍有效。
- [ ] 重测：Long 9 杯把手；增加锅柄或工具握持部位的可控测试，验证不是仅对 mug 名称生效的特例。

<a id="hv-05"></a>

### HV-05：几何类型统一来源

实验观察：传入 `mug` 被运行时拒绝，只能回退 `other`。当前 [default_contracts.py](../agent/tools/default_contracts.py) 的 `target_geometry_family` 使用普通 string，而 [memory.py](../agent/runtime/memory.py) 校验 `GRASP_GEOMETRY_FAMILIES` 固定集合，确认接口约束不一致。

拟改动：schema、Planner 校验、运行时 gate、文档投影共享同一类型定义。结合 HV-04 确定 object 类别和部件几何是否应为不同字段；补齐 mug/mug_handle 或定义明确 extension 机制，避免无约束字符串暗含可执行几何承诺。

- [ ] 验收：目录声明允许的值在各校验层一致接受；未知值一致拒绝并返回允许值/扩展方式，错误在生成动作前可见。
- [ ] 验收：重新生成契约及相关验证证据，执行本报告第 10 项的版本一致性检查；不手改生成 JSON 或历史 canary hash。

<a id="hv-06"></a>

### HV-06：Human VLM 嵌套请求工作台

实验观察：一次 `grasp_pose_estimate` 后端候选生成仅数秒，整体等待却为 8 分 52 秒；实际等待隔离 advisor session `inferred-047-e6a2bc`，主 session 过滤未显示该请求。回答 advisor 后主调用恢复。该因果说明来自测试 Agent 总结，本轮未重算耗时。

拟改动：请求携带父 Agent session、父工具调用、子请求 ID 和角色，工作台按父调用嵌套展示，显式区分后端计算、排队、等待人工、取消与完成。保留 advisor 的隔离模型上下文，UI 关联不应混入主会话对话。可评估人工模式显式关闭建议型 advisor、由主操作者选候选的选项；这不能隐式绕过宿主安全检查。

- [ ] 验收：主 session 页面即可发现和回答子请求；推断 session 名变化、重试或多个并发父调用不导致串联。
- [ ] 验收：分别记录 compute/queue/human-wait 时间；父调用取消后子请求不会继续提交结果或悄悄变为新会话请求。
- [ ] 重测：Long 9 的 advisor 等待场景；保留主、子请求各自的输入和回答来源审计。

<a id="hv-07"></a>

### HV-07：封闭腔体 placement

实验观察：Object 0、Long 0 的开放 basket 放置成功；Long 9 运杯到微波炉附近后释放，杯子落在外侧。失败涉及入口、门板、腔体边界、整体杯子姿态和释放位置，不能只按 EEF 点位到达判成功。

拟改动：表达放置对象整体几何、容器入口/内部区域、门状态、进入路径和释放后稳定性。AnyPlace 证据与规划能力应明确是否支持该布局；不支持时返回可理解的限制和所需观察。到达、物体整体进入、释放后留在容器、可关闭门应分别记录为待验证条件，由 Agent 根据反馈选择后续动作。

- [ ] 验收：进入路径同时检查夹爪与携带物包络、门板和腔体碰撞；把手 mask 不能代替杯体几何。
- [ ] 验收：释放后观察确认物体留在腔体，门关闭空间可用；物体掉到外侧不能仅凭释放命令成功判为完成。
- [ ] 重测：Long 9 先验证插入/释放片段，再跑放入并关门整任务，以可信环境成功证据判定任务完成。

<a id="hv-08"></a>

### HV-08：attachment probe 的命令与实测证据协调

实验观察：某些准备 probe 的调用因最近确认的夹爪命令不是 close 而拒绝；需要区分当前实测状态、确认命令和恢复操作造成的历史变化。

当前代码确认：[attachment_probe.py](../agent/tools/attachment_probe.py) 先检查 `gripper_command_state` 为 closed，然后还检查 proxy/contact receipt 和实测状态。因此不能描述为只检查最后一条命令；要定位有效 close 证据在哪一步丢失或被错误覆盖。

拟改动：联合当前可信 gripper receipt、对应 compiled-contact receipt 与可复用 attachment evidence，显式定义证据身份、顺序和失效条件。无关运动不应仅凭事件顺序清除有效 close 证据；真正打开、掉落/开度塌缩、环境重建或明确矛盾应失效。实测闭合本身也不能证明夹住目标。

- [ ] 验收：有效 close→无关移动→prepare probe 仍可依据一致证据判定；有效 close→open、句柄更换或实测矛盾不得复用旧授权。
- [ ] 验收：拒绝时准确指出缺失/矛盾的 receipt 和恢复所需证据；保留关节式把手与刚性携带物的区别，不把抽屉把手当可携带物。
- [ ] 重测：Goal 0 作关节式操作回归，Long 9 作恢复夹持验证。

<a id="hv-09"></a>

### HV-09：轮次成本与证据复用

基线按用户总结保留：Goal 0 / Object 0 / Long 0 为 43 / 48 / 99 episode steps；Long 9 达 203 steps、约 160 主 planner turns 仍未完成。episode step、planner turn、工具调用和控制器步数是不同量，后续不得直接相加或混作成功率分母。

拟改动：逐次归因必要闭环、重复候选编译、零步动作、epoch 导致 proposal 重建、人工子请求等待、IK/OSC 不一致的小步试探和 AnyPlace 同源证据重建。分别统计主/子模型 token 与耗时、实际物理动作、空操作和缓存命中；证据复用遵守数据不可变性与当前运动授权有效期，不能通过删除必要观察降低轮次。

- [ ] 验收：同任务、seed、控制器、预算及协助策略前后比较，列出成本降低来自哪些已修复原因；同时报告失败/超时样本，避免仅筛选成功短轨迹。
- [ ] 验收：在不降低碰撞和任务判定标准的前提下减少可避免重复；具体成本目标先依据 trace 分析制定，本轮不随意承诺固定下降比例。

### 合并后的实施与回归顺序

1. 基础可靠性继续按本报告第 1–10 项推进；完整 GraspGenX 坐标转换保持 P1 核查。它们与实验缺口一起构成待修复集合。
2. 恢复长任务测试前，优先 HV-01（控制器契约）、HV-03（前置主动感知与多方位观察）、HV-04（物体/部件分离），并联动 HV-05 保证新类型能通过共同契约。HV-03 先完成候选设计评审，不能只增加采样方向而保留分割失败时的启动循环依赖。HV-02 和 HV-08 是这些路径的回执与恢复配套。
3. 先重测 Spatial 0 的关系指代/遮挡和运动问题，再重测 Long 9 的部件抓取、狭窄入口和恢复；Long 9 的完整验收同时依赖 HV-07。HV-06 应在人工重测时提供子请求可见性。
4. Goal 0 / Object 0 / Long 0 作为成功路径回归保留，覆盖关节式操作、开放容器与顺序任务；不能只重复这些成功案例来代替 Spatial 0 / Long 9 的缺口验证。
5. 最后按 HV-09 汇总成本、任务成功、人工协助和放置估计器实际执行情况。每个 HV 项只有在附上修复提交、对应测试及所需实验结果后才能勾选完成。

本节定义待改动范围与验收条件，不直接授权运行新的机器人任务或修改共享接口。后续改字段、枚举和回执时应同步共享契约并标记需要协作者评审。

## 复现入口

[harness_review_2026_09_05.py](diagnostics/harness_review_2026_09_05.py) 使用项目当前实现及受控的 dummy/fake transport，不访问真实模型、仿真服务器或机器人。

```bash
# 从 OpenETA 仓库根目录运行
.venv/bin/python docs/diagnostics/harness_review_2026_09_05.py
```

运行会创建独立的系统临时目录，并在其中生成少量测试文件。它会输出各项复现结果；`*_probe_error` 表示复现脚本本身未成功，应先检查脚本错误再解读该项。
