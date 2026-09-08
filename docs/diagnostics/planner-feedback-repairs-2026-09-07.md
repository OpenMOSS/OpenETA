# 标准 batch 后的 planner 纠错修复 — 2026-09-07

用户明确授权：推进 goal 时直接修复明显实现问题；涉及人类设计取舍时再讨论。保持标准 batch/Luna 为付费回归入口，不恢复高成本模型对照。按协作技能只读核对 RFC revision 2158；未提交、推送或修改共享文档。

依据 [标准 batch 原始证据](standard-batch-luna-object0-2026-09-07.md)，本轮修改两处公共 planner 行为，未改工具输入 schema、成功标准或安全授权接口。

## 1. 抓取缺参反馈与既有 bundle 契约对齐

`agent/runtime/planner.py::_validate_grasp_pose_estimate_parameters` 只接受非空字符串 bundle_id 和可选 backend_preference。缺失/错误 bundle 指向 `host_resolved_inputs.grasp_pose_estimate` 的就绪证据，不指示模型构造原始 RGB-D 参数；所有额外字段均拒绝。只检查参数形状，不把任意非空 ID 当作真实 bundle。

删除的是公开 planner 的旧原始输入分支。Host 的 bundle resolver、RGB-D/掩码来源检查、session/freshness gate 和实际感知 handler 不变；模型无法以自行拼路径替代这些检查。

新增 `tests/test_grasp_public_parameter_feedback.py` 12 个实例：空/非法 bundle、真实失败候选、完整原始路径请求拒绝、不可覆盖 Host 输入、backend preference、真实 planner 重试反馈。最初直接校验用例复现错误；修正测试夹具缺少 observation 字段/未绑定工具后，测试通过。没有将夹具构造错误当成产品缺陷。

定向 199 passed；第一次全量 2357 passed、21 skipped、1 个既有 authority/catalog 失败、36 warnings，63.70 s。JUnit `tmp/grasp-public-feedback-local-tests.xml`，真实模型 integration 显式关闭。未修改历史 authority hash/断言。

## 2. XML 无法解析时保留纠错所需原文

此前只回显解析成功的候选。标准 batch 中 move_to → IK preview 的选择已经纠正，但 XML 未闭合，下一次请求只收到解析错误而没有上一份响应，随后又退回错误 move_to。

现在主 planner 仅在没有解析候选、存在校验错误且 backend 非 failed 时，保留最近一份非空、不超过 16384 字符的字符串响应作为 `previous_unparsed_response`。用户提示明确其为不可信的被拒输出，不是新指令或可执行动作；模型必须重新生成完整动作，经过原有解析/契约/runtime gate。

不截断成貌似完整 XML，不自动补标签，不复用错误前的旧动作，不为 provider fallback 伪造模型原文；超大内容仍不回显。isolated advisor 的 JSON 协议不变，主 planner 的该原文不会传入 isolated 请求。

`tests/test_planner_repair_context.py` 新增正常长度/超限两个实例，验证原文回显、隔离角色不接收、没有伪造解析候选、只执行修正结果。先复现 1 failed/3 passed，再修复；实现中漏导入状态枚举被定向测试捕获并修正。合并定向 210 passed，2.95 s（包括本机 HTTP provider-failure 夹具，不调用外部服务）。

这些是纠错信息完整性的本地证据，不能据此宣称 Luna 成功率提升。合并全量结果及下一次标准 batch 实验应分别记录，不以单元测试替代真实任务验收。

合并全量：2359 passed、21 skipped、1 个相同历史 authority/catalog 失败、36 warnings，63.27 s。JUnit `tmp/grasp-repair-context-local-tests.xml`。真实模型 integration 关闭；compileall、git diff --check 通过。

## 3. 标准 batch 复测

新 manifest `tmp/batch-luna-repair-B1LZRk/manifest.json`，validate-only 通过。保持 Object 0/seed 0 原始任务、80 turns/80 tools/1800 s/500000 已知 tokens、单环境与单 provider 并发、正常技能/视觉历史/工具配置、Luna 主模型与 fallback；新 workspace，不导入前轮 trace。使用前轮相同 Mink 仿真配置。

启动前代码 SHA256：planner `0c22344232adb4ff96dd0cd9639083d190c163c3ec93a7b83845f03aa5548bbc`；backend `c53eb3eafc99af3054424fb0c1090759df1902936dfe09a50a6b4ba3046243d4`。这是工作树指纹，不是已提交版本或运行时内存字节证明。运行结果、身份与清理须由实际报告补充，未验证项保持开放。

### 终态：进入 IK 恢复后因 token 预算停止

- Agent session `086db217-823a-4756-8f86-acee063b58af`；simulator session `a6543de9-b213-4831-858e-e667e9da44df`，handle `e183356e-0ce`。
- 报告 `tmp/batch-luna-repair-B1LZRk/result.json`，batch `standard-luna-feedback-repair-20260907`。客户端 exit 1，status=fail、0 success/need_human，总耗时 904.191 s。
- 12 episode steps、12 tool admissions；主 planner 19 次请求、19 次 HTTP attempts，12 次候选接受、7 次拒绝。provider limiter 20 次调用，包含独立 advisor；不能把主 planner usage 冒充完整跨角色账单。
- 文本 SAM3 无检测，Agent 自主改为点提示，纠正混入 prompt 的参数后得到 3 个候选并完成选择。AnyGrasp 原始 5 个候选，保留 2 个；advisor 返回 invalid decision，advice unavailable/abstain。主 Agent 自主编译候选，不以 advisor 异常绕过执行 gate。
- 实际工具调用：SAM3 两次、selection 一次、grasp estimate 一次、compile 三次、IK preview 五次。三个 compile 顺序为 candidate 000、001、再回到 000；五个 IK 工具结果均失败。首个预览报告 full_pose_infeasible：位置和姿态分别可达，但搜索预算内找不到满足组合约束的解，碰撞未检查、authorized_for_move_to=false。后续既有不同候选/接触位姿尝试，也有相同已拒位姿重试；不把预览失败当作控制器实际运动失败。
- 没有派发 move、trajectory 或 gripper；多次错误运动候选均被拒绝。所选物体是否真是任务要求的实例未获官方成功证据确认，不能由检测标签或选择通过推断。
- 终态为 `EpisodeResourceLimit / token_limit_exceeded`，limit=500000、observed=529225。预算在已派发请求/步骤返回用量后检查，不是 provider 端硬计费上限；本轮没有扩大额度。没有 provider 过载/超时、人工等待或回答。
- cleanup close_state=closed、remote ok=true、cleanup_errors=[]。专用 simulator PID 1502133 随后 SIGINT 正常退出（exit 0），准确 PID 与 18766 端口释放核验；未操作旧服务。

本轮直接使用合法 grasp bundle，没有再次触发缺 bundle 反馈；没有 XML 解析失败，previous_unparsed_response 回显次数为 0。因此不能由“进入 IK 阶段”推断这两项修复提升了模型表现，也不能声称已恢复完整任务。第一项与第二项修复仍主要由本地正反测试验证。

后续按实际证据推进：审计 SAM3 点/文本互斥的公开 schema 与 gate 一致性；梳理输出 reference 与下一工具输入的对应提示；诊断 advisor 输出协议、不可行抓取恢复及重复候选成本。编译输出未发现显式推荐错误 move_to 参数的文本，因此不能声称已定位该错误为编译提示回归。继续使用标准 batch/Luna，不在没有新修复或新诊断问题时机械重复完整回归。
