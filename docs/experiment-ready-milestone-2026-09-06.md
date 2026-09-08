# 本轮交付：恢复真实任务实验

2026-09-06 用户确认：先交付能恢复进行真实任务实验的版本。这里的“真实”指实际运行模型/人工 VLM、感知服务与 LIBERO 仿真任务，不是仅使用 mock；不包含实体机器人部署。

此范围覆盖此前批次计划的近期执行顺序，但不删除 [完整 TODO](todolist.md)。当前仍是开发中候选，**已取得短程控制器与观察闭环证据，尚未完成真实任务实验恢复的整体验收**。

授权后的新增证据（2026-09-07）：用户允许新仿真数据发送至指定三项感知服务，以及框架测试信息发送至 LLM provider。纯 Agent Spatial 0 已完成创建、真实 SAM3 分割和歧义停止，环境关闭确认；不是完整操作任务回归。详见 [本轮报告](diagnostics/authorized-agent-perception-2026-09-07.md)。本轮全量本地回归 2327 passed、21 skipped、1 个既有 authority 失败，门槛仍开放。

Long 9 同轮已完成创建、真实 SAM3 与目标选择，之后 LLM 60 s 超时，环境关闭确认；未进入 AnyGrasp/运动。发现 provider 失败分类和“等待即已人工协助”的统计缺口，已入 TODO。Spatial 0/Long 9 目前只有前置感知诊断，不覆盖原先运动/部件/封闭容器问题；完整成功路径回归和整体验收仍未完成。

R0 27 补充：用户进一步允许测试数据发送至 `10.11.39.173` 的各类小模型及 `https://open.xiaojingai.com` provider；不再局限三项感知服务，但不将源码、密钥或历史 trace 作为测试内容发送。授权不等同自动开启所有服务、重启已有服务或无限预算。本轮 goal 查询已恢复 active。provider 超时/人工协助分类已作本地候选修复，见 [证据与限制](manual-provider-accounting.md)；不是新的完整任务回归证据。

R0 28：两次 Object 0 完整任务尝试都停在创建后的 SAM3 输出校验，未执行运动且清理确认；新增有界纠错候选回显，同模型复测未改善结局。见 [Object 0 实证](diagnostics/object0-task-regression-2026-09-07.md)。全量本地回归 2345 passed、21 skipped、1 个既有 authority 失败；仍不满足成功路径验收。

R0 29 对照补充：仅覆盖进程模型为 `gpt-6-astra`，首次请求即收到上游 HTTP 500 / `server_is_overloaded`，没有创建环境或执行动作。真实失败被正确归类为 `planner_provider_failed`，无人工等待/协助，usage 未知；自建服务清理确认。未取得可比较的模型动作输出，验收状态不变，详见上述 Object 0 实证。

## 优先级与范围

标准入口补充（2026-09-07）：用户要求从 `agent.cli.batch_eval` 测试。首轮 Luna/Object 0 保留默认技能/视觉/工具配置，完成观察、SAM3、选择、AnyGrasp、编译后在运动准备校验耗尽，未运动；530.124 s，主 planner 295328 tokens（非完整跨角色账单），环境和自建服务清理确认。发现 grasp bundle 公开契约与旧原始输入纠错反馈冲突；此分支在 HEAD 已存在。详见 [标准 batch 实证](diagnostics/standard-batch-luna-object0-2026-09-07.md)。不满足完整成功路径验收，后续不以独立诊断入口替代标准回归。

实验成本约束（2026-09-07 用户最新指示）：后续统一使用 `gpt-5.6-luna`，不再启动 sol 或其他高成本模型对照，不因失败自动升级模型。此前已发出的 sol 运行结束并清理，细节见 Object 0 实证；旧文中的跨模型比较建议不再是当前计划。

后续修复与复测：公开 grasp 缺参反馈现对齐 bundle 契约，XML 解析失败可带有界不可信原文供模型纠错；Host 不自动修补或放行。合并本地 2359 passed、21 skipped、1 个既有 authority 失败。标准 batch/Luna 新运行进入五次 IK 工具调用但均不可行，最后 token_limit_exceeded，未运动、清理确认；本轮没有触发两项新反馈分支，不能宣称其 live 效果。详见 [修复与复测边界](diagnostics/planner-feedback-repairs-2026-09-07.md)。用户允许直接修明显缺陷，设计取舍仍需讨论，整体验收保持开放。

1. **可信、受控的实验入口。** 收敛已有修改，保留失败基线；核对实际控制器、服务与依赖，使用有限步数/轮次/时间预算。启动、成功判定、超时与关闭失败都要留有可诊断记录。实验路径上的过期授权或超时后迟到运动仍是阻断项，不能因推迟完整分布式租约而忽略。
2. **执行与反馈。** 优先 HV-01、HV-02、HV-06，以及 I27 中会丢失关键结果/证据的部分。区分 IK 的数学可达性和当前控制器执行能力；如实表达目标未到达、零步、未知结果及人工等待。不以工具调用成功替代动作达成，不以位姿局部不变证明整个世界未变化。
3. **必要的感知/抓取恢复。** 推进 HV-03/04/05 的最小通用能力。主动感知是确定需求，但尚未批准具体的 anchor 启动、跨视角投影和部件接口；沿用 [设计讨论边界](active-perception-design-options-2026-09-05.md)，需要新选择时单独讨论，不自动引入 Host 任务状态机。
4. **诊断与针对性回归。** Spatial 0 / Long 9 用于定位遮挡、部件抓取和封闭容器失败；Goal 0 / Object 0 / Long 0 保留成功路径回归。HV-07/08 中阻断受控测试的缺陷纳入；狭窄腔体稳定放置、关门的完整泛化能力不是当前完成条件。

## 验收门槛

- [ ] 定向测试通过；全量失败逐项解释，不修改旧 canary hash 来制造验证通过。
  - 2026-09-07 已完成 [扩大测试与失败归因](diagnostics/local-validation-matrix-2026-09-07.md)：合并本地测试 2313 passed、21 skipped、1 个历史 authority 失败；LIBERO CPU 合同 50 passed；cuRobo backend 矩阵 11 passed、3 个 MESH 初始化错误。生产 PRIMITIVE 检查通过，但总体非全绿，门槛仍开放。
- [x] 在实际选用的服务/依赖环境上完成有界 canary：记录环境 ID、seed、controller ID/interface、preview、实际终点/残差/步数、停止原因和关闭确认。2026-09-07 独立候选 RAG/cuRobo + OSC 服务通过短程检查；仅端点/执行后配置覆盖，不是 swept-path 安全证明。见 [本轮证据](diagnostics/experiment-ready-live-2026-09-07.md)。
- [x] 完成一次实际 Human VLM 模式闭环 smoke（Codex 担任操作者，新 session、有限预算）：真实图像 → 手动 observe 决策 → 结果及新图像可见 → talk 停止 → Host 关闭。控制台 pending 可发现；episode 的人工等待/协助统计仍未识别该模式，不能把它当作自主模型耗时。未覆盖嵌套 advisor。见 [本轮证据](diagnostics/experiment-ready-live-2026-09-07.md)。
- [ ] 完成至少一条既有成功路径回归，并对 Spatial 0 / Long 9 各做一轮有界诊断；记录未成功原因和复现证据。难任务不要求本轮全程成功。
- [ ] 交付可复用启动/诊断命令、已测组合和已知限制，明确哪些问题允许实验继续、哪些应停止当前 session。

默认先进行只读服务检查，再运行独立新建的有限仿真环境；不占用已有实验 handle、不改写历史 trace、不自动启动数小时的完整任务。服务不可达时区分运行环境限制与产品缺陷，单测不能替代 live 验收。

2026-09-07 入口修复：CLI 新增显式 `--simulator-mcp-url`，五个 Human VLM 脚本把检查过的 host/port 传给实际 transport；不再把第一个任意 MCP 当作 simulator。旧环境或 tracked worker 未释放时拒绝换地址，见 [地址选择边界](simulator-endpoint-selection.md)。本轮感知客户端因新仿真数据发往配置内网服务的授权未明确，被权限审查在启动前拒绝；没有取得新的 Spatial 0 / Long 9 结果。

2026-09-07 超时安全补充：位置匹配或连续观测不动不能证明远端操作结束。当前候选会保留未知操作 gate，并明确停止 episode，不再反复强制 observe。原地恢复尚未实现；应由 Host 确认旧环境清理后再创建新环境/session，不能靠删 gate 或换 session ID 绕过。详见 [超时停止与清理边界](motion-outcome-safety.md)。Host 私有 operation ID、终态查询、过期请求拒绝的最小协议仍待设计确认与评审；本批单测不消除 live 超时/迟到运动验收项。

预算补充：guidance 使用剩余 episode 时间并隔离迟到回答；预算停止不再启动新 review，正常 review 失败作为附属错误返回。默认 review 的准备阶段已有独立期限，但同步 provider 请求仍可能在后台等待，提交 I/O 与 token 尚未统一受限；不要把 episode timeout 宣称为所有工作都已硬终止。已返回的 guidance usage 纳入计账，未知成本不伪造为零。详见 [预算及后处理边界](episode-budget-boundaries.md)。

入口超时补充：`--simulator-timeout-s` 独立限制 simulator execution RPC（默认 300 s）；不再用 planner/provider 的超时抬高 simulator 期限。全部 Human VLM 启动脚本通过 `OPENETA_SIMULATOR_TIMEOUT_S` 显式传值。该项只有本地装配/派发回归，不构成 live 超时恢复验收；未知运动仍停止，不凭本地 timeout 宣称远端已取消。

工具配额补充：scoped executable tool 在 authorization/handler 前原子预留额度，batch 超额项不会执行；观察 `usage.tool_admission`，不要把包含失败/拒绝尝试的 `tool_call_count` 当成成功执行次数。该机制不限制单次 move 的 controller steps，也不覆盖绕过注册表的直接 simulator 调用；短程 canary 仍需自己的步数/时间约束。

连接补充：SSE MCP 现使用 client-scoped target-host proxy bypass，不再临时改写全进程 `NO_PROXY`；保留非目标主机代理与 TLS CA 配置，见 [代理隔离](mcp-proxy-isolation.md)。实际 SDK loopback fixture 已验证，但未替代真实感知/任务回归或数据发送授权。

## 暂缓但不关闭

人工子请求补充（2026-09-07）：guidance、独立 action reviewer、visual-delta 的 Host 父 session/独立 child ID 已接入现有分组和等待提示，见 [角色覆盖](manual-vlm-harness-debugger.md#additional-isolated-roles-2026-09-07-candidate)。只有本地 producer/queue 与固定文本 HTTP 夹具验证，未重启已有控制台。父 session 关联不等同父请求生命周期，级联取消、其他角色及子请求等待计账仍开放。

Review 提案提交补充（2026-09-07）：修复 save 缺失 ID 校验、同 ID 覆盖已决议记录及固定临时文件跟随 symlink 的问题；新建改为不可覆盖原子发布，重复提交在再次 auto-apply 前失败。已在仓库所在文件系统运行夹具，见 [提交边界](episode-budget-boundaries.md)。这不是审批/应用的跨进程事务或多文件回滚，未改历史记录。

Review 准备期限补充（2026-09-07）：默认 reviewer/built-in auto-applier 已拆为准备和提交两阶段。准备使用独立 120 s 默认期限，超时计划不写 proposal/skill，未退出线程阻止 runner 复用；及时结果经 ownership/skill 检查再提交。详见 [支持范围](episode-budget-boundaries.md)。自定义旧回调仍同步；磁盘提交、token 总额和远端硬取消不在此期限保证内，仅完成本地夹具验证。

Review 隔离补充（2026-09-07）：修复下游 review 可通过共享引用清空步骤/改写任务结果的问题，并拒绝将旧 review 事件写入重建后的 session。输入及报告复制不等同执行沙箱；后续已为默认路径拆开准备/提交，但全路径硬预算仍未完成，见 [后处理边界](episode-budget-boundaries.md)。本项仅本地夹具验证，不提高真实任务验收状态。

人工取消补充（2026-09-07）：backend 不再把控制台明确的 human_cancelled 503 当作可重试故障，也不自动转向 fallback。真实 loopback 固定文本夹具验证取消/控制台超时后没有第二次排队；参见 [取消边界](manual-provider-accounting.md)。尚未实现主请求取消后的子请求级联，也没有改变 simulator 超时语义或完成实际任务回归。

人工统计补充（2026-09-07）：控制台新增显式 manual-provider 响应元数据，主 planner 的校验重试可累计等待并标记协助；旧 `human_wait_s` 仍只表示 runner pause，预算扣时不变。详见 [覆盖范围](manual-provider-accounting.md)。当前仅本地 backend/episode 与 HTTP 夹具验证，未重新运行真实 smoke；其他角色、未完成响应和跨 resume 计账尚未覆盖，不可用 false 标记声称全自主实验。

生命周期补充（2026-09-07）：同一 Host 配置中的 create/reset 现在先占用启动标记；handle 尚未返回时，close 也返回 pending 而不是“空环境已关闭”，端点切换和重复启动同样被拒绝。CLI 关闭失败允许重试；创建后处理抛错保留已返回 handle。见 [清理边界](environment-cleanup.md)。这只经过本地并发夹具验证，调用方仍须重试 pending close；远端 create 超时无 handle 的孤儿问题尚未解决，不提升真实任务验收状态。

完整分布式租约/孤儿环境重建、全事件回放、跨平台沙箱、所有后端认证、全仓模块拆分、纯上下文美化和全面成本优化。若这些问题在选定实验路径上造成安全或正确性阻断，再按证据提升优先级。

专用分支保持 `dev/huaizezheng/harness-refactor-2026-09-05`。按 OpenETA 协作技能只读核对共享 RFC（本轮 revision 2158），所有本地接口候选仍须协作者评审；未授权提交、推送或修改共享协作文档。

## 短程控制器诊断入口

Runtime live 补充（R0 第二十三批，2026-09-07）：新增 `--through-agent-runtime`，实际 Host preview → receipt-ID planner 输入 → runtime/gate/proxy → Mink 链路通过，seed ID 匹配、2/40 步到达、清理确认。见 [后续运行](diagnostics/mink-seed-canary-2026-09-07.md)。确定性 planner、无图像观察，不替代视觉决策、proposal 选择或完整任务验收。

Exact receipt 修复（2026-09-07）：主 Host 链路不再把指定 receipt 的目标/容差与另一份同目标预览的 seed 混用；选中的失败证据不能借用后来的成功证据放行，较新的失败 veto 仍保留。已加入 runtime 到模拟 transport 的回归，见 [绑定规则](ik-seed-binding.md)。没有本次 live 主 Agent 仿真验收，不能把前次直接 Host Mink canary 当作此链路的新证据。

2026-09-07 Mink seed 补充：新增 `--require-ik-seed`，独立 Object 0/seed 0 上移 2 cm 诊断通过，2/40 步、终点误差约 1.91 mm，seed ID/policy 匹配且清理确认。见 [Mink 证据与复现](diagnostics/mink-seed-canary-2026-09-07.md)。仅直接 Host 短程路径，不替代主 Agent 引用解析、复杂支路或完整任务回归；未切换默认控制器。

先确定服务实际配置，不根据旧实验成功记录猜测控制器。以下命令新建独立仿真环境，不使用已有任务 handle；默认开启碰撞检查，目标为当前末端上方 2 cm：

```bash
.venv/bin/python -m scripts.controller_capability_canary \
  --url http://127.0.0.1:8766/sse \
  --env-id openeta/libero_libero_object_task0-v0 --seed 0 \
  --expected-controller-id robosuite.osc_pose \
  --max-steps 40 --timeout-s 30 \
  --output tmp/controller-experiment-ready-UNIQUE-RUN.json
```

每次选择新的 output 文件名；`--timeout-s` 是每次调用预算，不是整个脚本的总 wall time。`--expected-controller-id` 必须匹配本次实验方案：如果实验要求 Mink，使用 `mink.robosuite_joint_velocity`，不通过改写期望值来冒充 Mink 验收。

诊断要求控制器匹配后才 reset/preview，preview reachable 且明确完成含场景物体的 world/self 端点检查后才请求运动；运动步数须在预算内且非零，receipt 须与实际控制器/执行结果一致，实测终点的每轴误差须在容差内。还须检查实际 motion collision receipt 的 world/self 覆盖，不能只看请求中开启了检查。`--disable-collision-check` 仅作调试，整体不通过。上述配置检查不证明连续路径安全，也未改变服务端 best-effort 碰撞检查语义。异常保留失败报告并尝试关闭，关闭未确认则整体不通过。若远端创建已发生但响应丢失，脚本无法凭空恢复 handle，仍需人工对账；这不是执行租约实现。

默认模式直接调用 simulator MCP，不覆盖主 Agent 的 receipt 引用解析。新增 `--require-ik-seed --through-agent-runtime` 模式则将实际 Host preview 摄入 memory，经确定性 planner、runtime/gate/proxy 解析 receipt 并派发一次运动；该局部链路已有实际 Mink 短程证据。两种模式都不覆盖自主 proposal 选择、感知/人工 advisor、视觉模型决策或完整任务成功路径，不能用短程通过替代这些验收项。
