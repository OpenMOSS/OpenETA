需求：根据本项目code base/docs中的相关设计文档/rfc文档 撰写一篇8-10k字的中文博客

对标的同类工作：https://mp.weixin.qq.com/s/jt5KZ0pYDzUn7i4u6Nn3qg

初稿：（请你在下面完成初稿的组织；图片部分放占位符，并且描述需要一张什么样的图片）

---

# η0：我们如何把 VLM 变成一个会观察、会验证、会复盘的具身任务 Agent

> 副标题：从一次性 Code-as-Policy，到面向机器人操作的闭环 Agent Harness
>
> 本文基于 OpenETA 截至 2026 年 7 月 22 日的当前代码、设计文档、协作 RFC 与 LIBERO 实验记录整理。

让大模型控制机器人，看起来似乎只需要三步：给模型一张图，让它生成动作，再把动作发给机械臂。但真正开始做长程操作后，问题很快从“模型会不会规划”变成了一串更难的工程问题：它看到的是不是当前画面？分割出的是不是任务指定的那一个物体？抓取模型给出的相机坐标如何变成机器人世界坐标？控制器返回成功，是否就代表物体真的被抓起？网络超时后动作究竟执行了还是没执行？如果一次轨迹成功，Agent 能否从中学到什么，又如何避免把偶然性当成通用规律？

η0 是我们对这些问题的一次系统回答。它不是一个新的机械臂控制算法，也不是把某个多模态大模型包成聊天机器人。我们更关心的是一个可复用的“具身任务 Agent 系统边界”：让模型能够看见环境、调用原子能力、接收真实回执、在证据不足时拒绝冒进，并把每一次成功与失败留下来，用于评测、复盘与后续训练。

> 【图片占位 1｜文章封面】
>
> 建议画面：16:9 横图。中央是一台 Panda 机械臂在 LIBERO 桌面场景中将物体放入篮子；左侧是 RGB-D 观测与物体分割蒙版，右侧是 Planner、Tool、Memory、Verifier 组成的闭环。不要画成科幻脑机界面，重点突出“每次动作后都回到观测”。标题文字可放“η0 / OpenETA”。

## 一、从 Code-as-Policy 出发，但不把代码当成唯一答案

η0 的起点与 Code-as-Policy 密切相关。这条路线的核心吸引力是：语言模型可以生成程序，再用程序组合感知、几何、运动与控制原语，从而用更少的专用训练完成新任务。相比一次输出动作的端到端策略，程序有结构、可编辑、可调试，还能显式表达分支、循环和重试。

但是，当整个任务被生成为一段长程序时，具身环境的非确定性会迅速放大。视觉分割的偏差、深度噪声、抓取位姿的误差、控制器的未到位、物体的滑落，都会让后续步骤继续在过时状态上运行。程序语法正确，并不等于世界按程序假设发展。

我们因此做了一个很明确的选择：OpenETA 的主执行路径不是“生成一段解决整个任务的代码”，而是闭环 `tool_call`：

```text
observe
  -> planner 选择一个下一步命令
  -> host/runtime 验证参数、权限与前置证据
  -> 执行一个 tool
  -> 接收 ToolResult 与环境回执
  -> 重新观测
  -> 再规划下一步
```

`code_policy` 仍然被保留，但它被降到一个有边界的工具后端：只适合短时域、可局部验证、能在内部做 checkpoint 的小段代码。它不能绕过 simulator MCP，不能拿到原始环境对象，也不能取代外层的观测—执行—反馈循环。这不是放弃 Code-as-Policy，而是把它放回最合适的层级。

这个选择也改变了我们对 benchmark 对象的理解。真正应该被评测的，不只是一个 MLLM 的裸模型能力，也不是它使用一套人工精心编写技能库后的分数，而是“模型 + harness”这个组合：模型如何选择工具，harness 如何表达证据、限制动作、存储经验、处理失败，以及两者如何在一次次 rollout 中一起改进。

## 二、先把边界分清：Simulator、Agent 与 Adapter

η0 并没有把所有功能堆在一个“万能 Agent”里。仓库从一开始就被分成三个顶层边界：

- `sim/` 负责仿真环境、后端封装、环境注册、MCP 服务与运行时状态；
- `agent/` 负责轻量 Python Agent runtime、VLM Planner、Tool/Skill Registry、Memory、审查器与实验编排；
- `adapter/` 维护 OpenETA 自己的观测、动作与单步 bridge 契约，隔离仿真器实现和 Agent 决策。

仿真侧与 Agent 侧的长期通信边界是 MCP，而不是某个临时 REST 接口或直接 Python 对象调用。Agent 只选择稳定的高层工具与参数；IK、碰撞检查、控制器展开、action clipping 和后端 action vector 由仿真机器拥有。这一点很重要：大模型不需要生成 MuJoCo 或某个控制器特有的低层向量，更不应该通过推测隐式调用它不理解的后端细节。

三类最核心的数据包贯穿这个边界：

1. `EnvObservation` 包含任务文本、多相机 RGB-D、内外参、机器人 proprioception、物体摘要与环境 metadata；
2. `AgentCommand` 是每个规划回合的结构化决策，顶层只有 `tool_call` 和 `response` 两类；
3. `ToolResult` 用统一 envelope 返回 `outputs`、`artifacts`、`state_delta`、`diagnostics` 以及是否需要新观测。

为什么 `CommandKind` 故意只保留两类？因为“抓取”、“查询技能”、“做安全检查”和“向人提问”本来就不在同一个抽象层。OpenETA 将这些概念明确分开：

- **Tool** 是 Agent 可调用的原子能力，例如 `sam3`、`grasp_pose_estimate`、`move_to`、`gripper_control`；
- **Skill** 是 Markdown 文本指导，例如 pick、place、pull、stack，它可以推荐工具顺序，但不会在背后展开成隐藏动作；
- **AtomAction** 是会改变世界的物理原语，在当前实现中是 `ToolSpec(effect="world_mutating")` 的子集；
- **Response** 用于 `talk`、`ask_human` 或 `task_complete`，它结束当前决策回合，而不是一个机器人动作。

这些命名看似繁琐，实际上它们是多人协作和长期演进的硬边界。只有先区分“模型做了什么决策”、“host 调用了什么能力”和“仿真世界真正发生了什么”，我们才能对失败负责。

### 模型输出先被验证和编译，而不是直接执行

在这套边界里，Planner Backend 的权力其实很小：它接收一份有预算的多模态上下文，返回一个 JSON 决策，但它既不执行工具，也不拥有环境。`ToolCallingPlanner` 会先检查顶层命令类型、工具名称、参数 schema、当前工具是否可见，以及动作是否满足待处理的感知和状态义务；随后 `ActionPipeline` 才把有效决策编译成 `CommandPipelinePlan`，记录安全检查、工具调用、状态和失败原因。一个不存在的工具、一组尺寸不匹配的 RGB-D、缺少来源的 mask，或试图批量夹带物理动作的请求，都不会因为“模型很确信”而被放行。

模型调用失败与模型决策无效也被拆成两层处理。网络超时、限流、连接错误等瞬态 provider 故障进入有界传输重试，并可在主、备 endpoint 之间切换；日志只保存 provider role、实际模型、切换次数和重试诊断，不保存明文密钥。模型已经成功返回、但 JSON 或动作语义不合法时，才进入 validation retry，下一次请求会收到紧凑的校验反馈。对于能够由 host 确定修复的门禁，例如必须先读取完整 skill guidance，runtime 不会浪费一次模型重试，而是把请求显式重定向为可追踪的 `skill_call`。校验预算耗尽则形成结构化 planner failure，而不是把内部错误伪装成一个需要人类回答的问题。

更重要的是，下一步不总由 VLM 自由选择。Memory 中存在一组 host-owned obligations：新观测缺失时必须先 `observe`，SAM3 结果未确认时必须先选择或拒绝 mask，抓取 candidate 尚未绑定标定时必须先编译，接触前后必须完成腕部对齐、夹爪状态、lift probe 与 attachment gate，放置阶段还要处理 pose transform、分段 carry 和 release obligation。它们像一组显式状态机护栏，约束模型只能在当前证据允许的边上前进。护栏可以自动安排只读刷新或确定性的下一项检查，却不会在 Skill 背后偷偷执行一串物理动作；所有世界变化仍以独立 tool call 出现在 trace 中。

这种“模型提出意图、host 编译并持有不变量”的分工，让替换 VLM provider 不再等于重写机器人运行时。更强的模型可以减少无效回合、改善视觉判断和恢复策略，但不能改变同一套工具契约、证据门禁与环境真值来源。对 benchmark 来说，这也使模型能力和 harness 能力能够被分别分析，而不是混成一个无法解释的总分。

> 【图片占位 2｜OpenETA 三层架构图】
>
> 建议画面：一张扁平化架构图，从左到右为 `sim/`、`adapter/`、`agent/`，下方单独放 Perception/Control MCP services。用实线标出 `EnvObservation -> AgentCommand -> ToolResult/StepResult`，用虚线标出 logger 和 rollout recorder。图中应特别标注：“Simulator 拥有 IK/碰撞/控制器细节”，“Agent 只选择稳定原子工具”。

## 三、一次只改变世界一次

具身 Agent 与普通聊天 Agent 最大的区别，是它的错误会立即改变世界。一个错误的数据库查询可以撤销，一次错误的抓取可能已经把物体撞倒、推走，甚至让后续观测完全失效。因此，OpenETA 对 world-mutating tool 设置了一条简单但强硬的规则：**一次只执行一个，执行后必须获得 fresh observation，才能做下一个依赖当前状态的动作。**

只读感知和纯规划工具可以在安全条件下批处理，但 `move_to`、`follow_eef_trajectory`、`gripper_control` 不能被模型打包成一串不可观测的动作。如果一个物理工具返回了可信的新状态快照，runtime 直接用它构建下一轮 `EnvObservation`；如果工具成功或结果未知，却没有返回新快照，host 会生成 `fresh_observation_obligation`，在下一次 VLM 决策前自动调用同一环境 handle 的 `observe`。连续三次仍无法获得规范快照时，episode 会以结构化原因终止，而不是无限重试。

这里有一个很容易被忽略的细节：历史图像仍然可以保留在 trace 和 memory 里，但它们不再被标记为“当前画面”。当一次物理动作发生后，runtime 会移除动作前的 RGB-D、相机标定和机器人状态的“当前”资格。它们是可用于复盘的历史证据，不是可用于下一次动作的新鲜证据。

同样的原则也用于工具结果。`ToolResult.success=true` 只能说明这次工具调用在自己的边界内完成了，不能说明任务已经成功。例如：

- SAM3 成功返回蒙版，不代表蒙版就是目标实例；
- 抓取位姿模型成功返回 candidate，不代表它可达或适合当前夹爪；
- `move_to` 返回没有传输错误，不代表末端已到达目标；
- 夹爪闭合，不代表物体已附着；
- 画面看起来放进了篮子，不代表 benchmark 的 reward 或 checker 已经判定任务完成。

因此我们在系统里同时保留三类语义：工具调用是否成功、世界状态是否给出了支持性证据、任务检查器或官方 reward 是否认可。这三者不能相互替代。

> 【图片占位 3｜单回合闭环时序图】
>
> 建议画面：横向泳道图，泳道分为 Planner、Host Runtime、Tool/MCP、Simulator、Memory 五行。用编号标出 Observation、Decision、Validation、World Mutation、Environment Receipt、Fresh Observation 六步。在 World Mutation 后用红色门标出“禁止直接执行第二个物理动作”。

## 四、一次抓放任务，实际上是一条证据链

要理解 η0，最好的方式不是继续罗列模块，而是跟着一次 pick-and-place 走完全程。

### 1. 先解决“任务说的到底是哪个东西”

对“red cube”这类属性清晰的目标，Agent 可以在当前 RGB 上用简洁的英文 prompt 调用 SAM3。但 LIBERO 中有很多具有特定包装和纹理的仿真资产，例如 `alphabet soup`、`akita black bowl`。把 `alphabet soup` 粗暴改写成 `soup can` 可能会分割出同场景中另一个类别相同的罐子。这不是“分割失败”，而是更危险的“实例身份错误”。

为此，OpenETA 引入了受控的 object memory bank。它不是一个让 Planner 随意填 URL 的搜图工具，而是按环境 namespace 管理的资产目录。当前客户端先做排名搜索，再获取精确 bundle；除了 exact-key 命中外，自动选择要求 Top1 分数至少为 0.75，且与 Top2 的差值至少为 0.10。低置信度和候选歧义都会结构化地 fail closed，而不是默认选第一个。

得到目标资产的正面、侧面和顶视参考图后，一个隔离上下文的小型视觉定位器会同时查看原始场景和三张参考图，在原图坐标系中返回一个前景点与紧致包围框。它不只比较大致类别，还要比较包装几何、主色、文字、标签图案、瓶盖或把手等属性。另一个独立验证器再对候选 crop 做 exact-instance review；几何不匹配或显著外观冲突时，它会拒绝当前候选并要求定位器寻找别的物体，最多进行有界尝试。

### 2. 分割分数是排序，不是身份证明

无论 SAM3 返回一个还是多个蒙版，runtime 都会生成显式的 `selection_obligation`。下一轮 VLM 会同时收到原图和候选 contact sheet，然后必须通过 `select_sam3_detection` 选择稳定的 detection id。在这个义务未解除前，定向抓取估计和物理控制都被阻止。

这一步看似多了一次模型调用，却解决了一个关键语义缺口：感知模型优化的是 mask 质量和分数，任务需要的却是“这是不是用户指定的那一个”。分数可以帮助排序，不应该被偷换成身份判定。

### 3. 抓取位姿是带来源的数字参考

目标蒙版确认后，`grasp_pose_estimate` 会使用同一帧的 RGB、深度、相机内参和完整 mask artifact。当前 host 负责选择具体抓取后端并整合候选，Planner 不直接在 AnyGrasp、Contact-GraspNet 或 GraspGenX 之间随意重试。返回的 candidate 按各后端内部分数排序，但不同后端的分数不可直接比较。

一个 candidate 从相机坐标转到世界坐标后，依然携带 candidate id、原始 rank、预测器来源、相机 frame、scene epoch 与标定哈希。`compile_grasp_seed` 由 host 从 memory 绑定相机位姿与外参，防止模型手抄浮点数时丢失指数、使用过时标定，或在转换后偷偷修改位姿。

候选回退也不是“失败了就换下一个”。只有与当前 candidate id 绑定的安全拒绝或动作/附着失败，才会消耗这个候选并激活下一个 rank。参数错误、标定缺失、网络超时、GPU OOM 或无关工具失败，都不是候选质量的证据，因此不会跳过它。

### 4. 夹爪闭合不是抓取成功

运动阶段按 hover、腕部视觉对齐、contact、close、lift probe 逐步展开。每一条物理边都是单独的原子工具调用，并在下一条边前重新观测。夹爪闭合的确认与观测到的 openness 只能证明夹爪状态，不能证明物体状态。

当前 pick guidance 要求一个固定的小幅抬升探针，然后由独立证据检查目标是否与末端共同运动、原位置是否空出。只有两类证据一致支持时，attachment gate 才进入 PASS；如果物体留在原地，才可以标记 FAIL、张开夹爪并回退到下一 candidate；证据不清楚时是 UNKNOWN，系统应该继续观测，而不是把不确定性当作失败或成功。

### 5. 放置不是把抓取位姿倒过来用

对组合 pick-and-place 任务，放置规划需要在第一次抓取动作前保留同一帧对齐的 RGB-D 证据。Agent 分割放置区域，然后将原物体蒙版、放置区蒙版、相机标定与已选抓取一起交给 AnyPlace。AnyPlace 的输出仍然只是一个低位几何参考，不是可直接执行的长轨迹。

真正的携带过程会先上升到安全高度，再使用 host 生成的有界水平 waypoints 移动，每段后检查附着是否仍然成立，最后才浅降、释放和退出。早先的抬升探针 PASS 不会成为永久许可：物体可能在携带中滑落，因此每一段运动都需要新证据。最终任务是否成功，仍由仿真环境官方 reward、termination 或任务 checker 决定。

> 【图片占位 4｜一次抓放的证据链】
>
> 建议画面：一张两行多阶段流程图。上行为感知：任务文本→资产参考图→点定位→SAM3 候选→显式 mask 选择。下行为动作：抓取候选→标定编译→hover→wrist align→contact→close→lift probe→attachment verdict→carry waypoints→release→reward。在每个阶段下写明关键 provenance id。

> 【图片占位 5｜实验关键帧组图】
>
> 建议画面：从一个实际成功 episode 中导出 6–8 张帧，包含原始 agentview、参考定位标记点、SAM3 mask/contact sheet、抓取 candidate overlay、contact、lift probe、篮子上方携带、放置完成。每张图下方标记 turn 和 tool name，避免只放最终成功截图。

## 五、把安全和失败恢复放在模型意图之上

一个自主 Agent 不应该拥有修改自己安全级别的权限。OpenETA 把 supervision profile 放在 host 配置面，而不是注册成一个 Agent tool。当前支持三种运行方式：`human_gated` 中物理动作和技能修改需要真人批准；`standard` 依靠确定性 runtime/checker gate；`reviewed_autonomy` 会在确定性门禁之上加入隔离上下文的 action reviewer、guidance resolver、skill author 和 reviewer。无论选哪一档，Agent 都不能自行升档，也不能关闭 IK、碰撞、后端或证据门禁。

当前项目必须区分“已经实现”和“目标架构”。独立进程形态的 Safety Checker 和 Failure Checker 仍是 RFC 中的目标，最终 verdict schema 尚未完全冻结；当前生产路径已有 `ActionPipeline` 的内联 `safe_check`、世界动作的中央 host gate、可选 checker hook、reviewed-autonomy action reviewer、仿真后端真实运动检查，以及围绕抓取/放置的显式证据义务。我们不希望用一个“智能检查器”的名称掩盖尚未完成的部分。

失败分类同样强调证据。一次传输超时的结果是 **UNKNOWN**，不是动作失败；Agent 必须先在同一 handle 上观测并对账，才能决定重试还是继续。GPU OOM、provider timeout、MCP 断连和仿真服务不可达属于基础设施失败，不能计为抓取几何失败，也不能被当成学习某个参数的负样本。反过来，只有来自当前执行、当前环境 handle 并通过 host provenance 校验的 environment receipt，才能提供 benchmark reward 和 terminal 字段。

人类介入也被设计成可审计状态，而不是一个模糊的“人帮了一下”标签。并行评测的 outcome 只有 `success`、`need_human`、`fail` 三种。如果 `ask_human` 无法在当前 supervision profile 下解决，harness 会保存 Agent session 与唯一 interaction id，立即关闭将过期的 simulator handle。人类回答后，系统使用相同 env id、任务和 seed 重建环境，保留 Agent memory，却不伪装物理世界跨越了长时等待。最终统计会分开 autonomous、guidance-agent-assisted 和 human-assisted success。

## 六、Memory 不是聊天记录，而是当前世界的工作台

闭环 Agent 需要记住什么？最直觉的答案是“历史对话”，但对具身操作来说远远不够。它还需要记住哪个 SAM3 result 正在等待选择、哪个抓取 candidate 当前激活、它属于哪一帧和 scene epoch、夹爪命令是张开还是闭合、lift probe 是否完成、附着证据是 PASS/FAIL/UNKNOWN、放置任务已经完成了哪个子目标，以及上一次运动超时是否还在等待状态对账。

OpenETA 因此将会话历史、工作记忆和证据记录分开。每个 session 下有追加式 `trace.jsonl`、模型可见的 `conversation.jsonl`、可变的 `working/*.json`，以及不允许压缩的 `rollout/`。前三者为了运行和上下文管理可以紧凑化，rollout bundle 则是评测和训练的不可变证据：它保留完整模型调用、被 schema validation 拒绝的尝试、工具边界、观测—动作—新观测—奖励 transition，以及内容寻址的媒体 artifact。

这套分层还解决了两个实际问题。第一，MCP 返回的 RGB/depth base64 不会被直接塞进 Planner 上下文。它们先落盘成 session-local artifact，Planner 只获得精确路径、校验后的标定和紧凑摘要。第二，并行 episode 不会共享可写状态。每个 worker 都拥有独立的 `skills/`、`memory/`、`artifacts/`、`sandbox/`、`calibrations/`、`strategies/` 与 `task_playbooks/` 根目录。一个 Agent 在自己会话中修改的内容，不会污染其他评测 worker，更不会直接覆盖仓库中的共享默认值。

主 Planner 每次看到的也不是整个技能库和全量轨迹。`build_tool_context` 会组装有界上下文：当前观测、少量最近事件、工具摘要、技能 metadata 索引、按任务匹配的少数 skill guidance，以及必须完成的 host obligations。上下文压缩的目标不是让模型“忘记”，而是把真正会影响下一次动作的状态提到最高优先级。固定尺寸的相机矩阵和标定数据必须完整保留，长文本和历史工具结果则只保留后续决策需要的部分。

这套记录方式也为后续训练预留了比“保存一段对话”更完整的接口。`model_calls.jsonl` 会记录每一次 schema validation 尝试，包括被拒绝的原始输出和校验原因；`tool_calls.jsonl` 保存精确参数、监督信息、结构化结果和 artifact 引用；`transitions.jsonl` 保存 observation、action、next observation、reward 与终止状态；`episodes.jsonl` 则保证零步失败和中断 episode 也不会从数据集中消失。RGB 以无损 PNG、depth 以带单位和尺度信息的数组保存，重复媒体按照内容哈希去重。API key、authorization header、cookie 和常见 bearer 模式在持久化前会被脱敏。

但原始 rollout 不会被直接拿去训练。一个显式 exporter 必须先把模型调用、工具结果、状态 transition 与最终 reward 对齐，过滤中断、接受人类协助、不安全或 schema 不兼容的样本，再选择成功输出构造 SFT 数据，或利用 validation retry 中的拒绝/接受结果构造 preference pair。导出前还要验证每个 JSONL 序号、artifact 是否存在以及 SHA-256 是否匹配。这样保留下来的原始 bundle 是模型无关的证据层：未来既可以导出主 Planner VLM 数据，也可以导出视觉定位器、动作 reviewer 或小型控制策略的数据，而不需要为了某一种训练格式重新设计在线 runtime。

可复现性同样不是一个 commit hash 就够了。一次结果还与工作树中的未提交改动、主 Planner prompt、可见工具 schema、handler 是否真实绑定、skill snapshot、provider/model、随机种子和标定 profile 相关。OpenETA 将这些信息放入 rollout manifest，并为 prompt、skill tree、strategy tree 与 calibration 计算稳定哈希。这样，当两次相同任务结果不同，我们能够判断变化来自模型、提示词、工具后端、环境、知识基线还是物理参数，而不是只能重复观看两段看起来相似的视频。

> 【图片占位 6｜Session Workspace 与数据分层图】
>
> 建议画面：左侧是一个 session workspace 目录树，展开 skills、memory、artifacts、sandbox、calibrations、strategies、task_playbooks；右侧是 trace、conversation、working memory、rollout 四层表格，标明“可压缩/不可压缩”与主要消费者。用锁形图标表示 session 隔离。

## 七、自我改进的难点，是不把偶然成功当成真理

如果 Agent 只会在一次 episode 里试错，那么它并不会真正改进。但如果允许它在每次失败后立即修改共享 skill、标定或工具契约，它又很容易过拟合单个场景，把一个偶然成功的 XYZ 坐标或某次候选 rank 写成全局规则。η0 将“学到经验”拆成几个适用范围完全不同的层次。

**Embodiment calibration** 回答的是稳定硬件问题：机器人、夹爪、控制器、相机与抓取坐标如何兼容，抓取 frame 如何映射到末端 frame。它由 host 根据环境 fingerprint 选择，正常任务 episode 只读，不能因为一次抓取失败就重新标定。

**Grasp strategy** 回答的是可修订的任务族经验：哪些几何类别可以激活某种抓取方式，宽度上限、接近方向、姿态策略和验证范围是什么。它可以是 candidate 或 validated，但不能修改底层标定变换，也不能突破物理夹爪宽度。

**Skill** 是可复用的任务流程与安全指导。它用文本描述何时观测、如何选 mask、何时做附着验证、什么失败能换 candidate，却不包含隐藏的复合执行流。主 Planner 不直接生成最终 replacement content；独立 clean-context author 根据变更目标生成严格 `SkillSpec`，另一个 reviewer 检查工具 allowlist、不可变边界与是否夹带隐藏工作流。

**Task playbook** 则只保存精确任务经验。它同时绑定 environment id、suite、task index、规范化任务文本哈希与可选 calibration id，只有所有 scope 完全命中才会进入 Planner 上下文。它可以记录曾经有效的物体查询、抓取几何签名和阶段顺序，但 schema 明确禁止世界绝对坐标、`move_to` 参数和 candidate rank。换句话说，playbook 是先验，不是轨迹回放。

实验编排采用 generation-based 流程。每一代都有不可变的 baseline skills、grasp strategies 与 task playbooks；训练 episode 在私有 workspace 中提出候选修改；只有正 reward 或显式环境 checker 成功才能贡献 candidate；候选经独立 reviewer 后，还要用相同 holdout manifest、episode id 和 seed 对 baseline/candidate 做成对比较。新版本必须不丢失 baseline 已经成功的 episode，不降低客观成功和 runtime success，不增加 fail 或 `need_human`，才能成为下一代 baseline。即使如此，它也不会自动改写共享 `agent/skills/*.md`。

对 calibration 和 strategy，门槛更高。每个提案和证据都同时绑定自身 SHA-256 和标定 profile SHA-256。validated strategy 的默认门槛包括至少 20 次 held-out 尝试、95% 客观成功率、覆盖两个 held-out 任务、无回归 episode、无人类介入且零安全/契约违规。模型的“我认为这个更好”无法绕过这些 host-owned gate。

> 【图片占位 7｜自我改进的四层知识与晋升流程】
>
> 建议画面：左边画四层金字塔，从下到上为 Calibration（机型级）、Grasp Strategy（几何族）、Skill（通用任务流程）、Task Playbook（精确任务），在每层旁写适用范围与禁止内容。右边画 Session Candidate→Independent Review→Paired Canary/Holdout→Next Generation Baseline→Explicit Shared Publication，用红叉标出“单次成功直接写全局”。

## 八、评测不只要跑起来，还要知道它为什么停下

为了从单次 Demo 走向可比较实验，OpenETA 实现了独立 episode 的并行评测 harness。跨环境可以并发，但每个 episode 内的 world-mutating 闭环仍然严格串行。环境并发与 provider 并发分开限流：主 Planner、action reviewer、guidance resolver、skill author 与 reviewer 共享一个有界 provider semaphore，避免“开了十个环境”等价于“同时冲击十次模型 API”。

每个 episode 都有独立的 turn、tool-call、wall-clock 和 token 预算。预算不是为了简化统计，而是让“没做成”具有可操作的原因：`tool_call_limit_exceeded`、`episode_timeout`、`token_limit_exceeded`、`target_localization_exhausted`、`provider_queue_timeout` 和 simulator transport failure 对应完全不同的修复方向。在超时时，Runner 拥有 deadline 与取消权，会立即请求 `close_env` 并拒绝迟到的 turn 结果回写 episode。远程环境 handle 是稀缺资源，所有创建路径都必须在 `finally` 中尽力清理。

日志则作为整个系统的事实来源。episode、step 和 tool event 三个层级串起了“看到什么”、“模型决定什么”、“host 允许了什么”、“仿真器做了什么”、“什么证据让系统继续或停止”。完整 rollout 还保存 git commit、dirty-state hash、prompt hash、可见工具契约、skill snapshot 和 provider fallback 详情。没有这些信息，一次“成功视频”几乎无法被复现，也无法转化为高质量训练数据。

## 九、我们在 LIBERO 中真正跑出了什么

一个具身系统最容易陷入的陷阱，是把“展示过一次”写成“已经解决”。我们希望用实验数据保持克制。

截至当前，项目留存了两次同一 seed-0 LIBERO Object 任务的独立 reward-1 记录。第一次任务是“拿起 alphabet soup 并放入篮子”：官方 reward 为 1，环境 termination 为 true，共 49 次 tool call，耗时 1308.616 秒，计入 835,520 个模型 token，人类与 guidance-agent 介入都为 0。奖励直到释放并退出后才出现，之前的抓取、夹爪闭合、附着与视觉完成都只被记为中间证据。

后续的 placement-contract 复现同样达到了 reward 1，共 57 次 tool call，耗时 1521.165 秒，无人类或 guidance 介入。前两个抓取候选在附着检查中失败，每个结构化拒绝只触发一次 0.12 米的世界 Z 方向退回；后续 candidate 通过抬升和附着检查，放置阶段使用了最大 0.08 米的 host-generated carry waypoints。这次 reward 在携带过程中已经出现，系统因此没有再执行一次多余释放。

2026 年 7 月 22 日的 ranked object-memory 六任务 canary 给出了更接近真实当前能力的结果：**1/6，成功率 16.7%**。成功 episode 是 LIBERO-10 task 1，Agent 在第 60 轮将 cream-cheese box 和 butter 都放入篮子，收到官方 `reward=1.0`，同样没有人类或 guidance 介入。六个远程环境都完成了清理。

| 实验 | 客观结果 | 关键意义 |
| --- | --- | --- |
| alphabet-soup 单任务 | reward 1，49 calls，无介入 | 验证了完整闭环可以跑通，但代价和时延很高 |
| placement-contract 复现 | reward 1，57 calls，无介入 | 验证了结构化 candidate 回退与有界 carry |
| 7 月 22 日六任务 | 1/6，官方 reward 判定 | 证明参考检索路径在真实 rollout 中生效，但远未解决通用长程操作 |

这次六任务运行也让我们看到 reference path 的边界。`black bowl` 能够解析到 `libero/akita_black_bowl`，五次 reference-localizer 调用成功，一次因外观冲突主动 abstain；该 episode 至少一次确认了附着，但最终没有在 100 轮内完成所需放置关系。`butter` 在成功 episode 中完成了检索与定位，而早先的 `cream cheese` 查询因 Top1 分数低于 0.75 被安全拒绝。两次 `alphabet soup` 定位因场景罐体图案与参考图冲突而 abstain，有界 Molmo fallback 也耗尽后，episode 以 `target_localization_exhausted` 停止，没有对未定位目标执行抓取。这种“没成功，但没有乱动”本身就是我们想保留的系统属性。

整个 batch 共记录 355 次 provider 请求，provider concurrency 为 2，没有 queue timeout；累计排队 691.1 秒，最长单次排队 22.7 秒，batch 总耗时 3177 秒。长间隔不是 semaphore 饥饿，而是下游模型、验证和仿真调用的综合延迟。这些数字提醒我们：当前系统的闭环是真的，但它还不便宜、不迅速，也远不够稳定。

> 【图片占位 8｜LIBERO 实验结果图】
>
> 建议画面：一张信息图而不是只有成功率柱状图。左侧画两个 reward-1 单任务时间线，标出 49/57 calls 与无介入；右侧画六任务 1/6 的分布，把其他任务按“定位耗尽”、“放置/长时域未完成”、“抽屉策略未验证”分类，不要把基础设施失败混成策略失败。

## 十、这些实验改变了我们的哪些设计

η0 的许多设计并不是坐在白板前想出来的，而是在一次次失败 trace 中被逼出来的。

首先，**目标身份比类别更重要**。把奇特资产名宽化成普通类别可以提高“有 mask”的概率，却可能降低“mask 正确”的概率。这导致了 object-memory ranked search、参考视图、隔离定位器与显式 mask selection 这条证据路径。

第二，**传递数值必须同时传递来源**。RGB、depth、mask、intrinsics、extrinsics、frame id、scene epoch 和 candidate id 只要有一个脱节，后续几何就可能在数学上合法、在物理上错误。因此这些数据不再被当作可随意复制的 JSON 片段，而是一条不能断开的 provenance chain。

第三，**失败恢复必须有边界**。候选回退、超时重试、定位 fallback、抬升探针、失败后退回都有次数和距离限制。没有边界的“自我修复”往往只是在消耗更多 token 和仿真时间，并可能把环境推得更远。

第四，**并发数不等于有效吞吐**。早期六任务运行曾经暴露 simulator SSE 端点不稳定、环境 handle 丢失和 AnyGrasp GPU OOM。某次 concurrency 3 已经高于当时部署被验证的稳定容量。因此环境容量、provider 并发、感知 GPU 容量和单 tool deadline 需要分开测量，而不是用一个“并发数”概括。

第五，**一条成功轨迹是候选经验，不是全局策略**。例如顶视 bowl 抓取、drawer-handle approach 和 upright-bottle 策略都可以从局部 canary 得到可达性证据，但在反复 attachment、拉出方向和 held-out 成功尚未完成前，它们只能留在 candidate。

## 十一、η0 现在能做什么，还不能做什么

当前 η0 已经不再是一个 dummy loop。它有可交互的 OpenETA console、多 provider 与故障切换、多模态单步 Planner、稳定 Tool Registry、可编辑文本 Skill、session memory、仿真/感知 MCP proxy、并行 episode harness、人类恢复、reviewed autonomy、不可变 rollout recorder，以及初步的 generation-based skill/strategy/playbook 实验编排。它已经在 LIBERO 中凭借完整闭环和官方 reward 跑出长程抓放成功。

但它远不是一个通用机器人基础模型。16.7% 的六任务 canary 比单个成功视频更能代表当前状态。多物体任务仍缺少更好的子目标进度与剩余 horizon 管理；放置关系、释放与携带中的附着仍是主要瓶颈；水平抽屉把手的附着保持与有界拉动还没有客观验证；视觉参考路径能提高特殊资产定位的可控性，却不会自动解决抓取、放置和长程完成；真实部署的网络、GPU、MCP liveness 和模型延迟仍会显著影响实验。

真机也不是当前文章可以宣称已解决的部分。仓库中已经把 calibration、supervision、证据、人类审批与可恢复 session 设计为仿真和真机共用的抽象，但真正上机仍需要更严格的硬件安全包络、控制频率、急停、传感器标定、负载限制和独立验收。仿真闭环跑通是必要条件，不是充分条件。

## 十二、下一步：从会完成任务，走向会生产可验证经验

η0 的近期主线仍然是把单机单臂的 pick-and-place 与铰链类任务做得更可靠：稳定 simulator 容量与取消语义，完成独立 failure verdict 契约，扩大客观 checker 覆盖，改进长时域子目标预算，并用更多重复 canary 和 held-out 证据验证 candidate strategy。这一阶段我们更希望得到一条可信的成功率曲线，而不是更多精选 Demo。

再往后，OpenETA 会将仿真任务、rollout 证据、数据导出和小模型训练连成一条自动研究流程：在多个带 verifiable reward 的具身任务中批量收集闭环 tool-calling trajectory，用明确 exporter 将模型调用、工具结果、transition 和 reward 连接成 SFT 或 preference 数据，再研究小参数 agent model 能否从这些轨迹中学到具身工具使用与短时域 code policy。这就是我们对 Cap-Agent1 式 auto research workflow 的工程化理解：自动化不是让 Agent 随意改自己，而是让每次变更都有来源、有对照、有回归检查、有可撤销边界。

中期路线还包括双臂协作、Agent 间通信、高频专用 control policy、底盘与双臂的移动操作。但这些方向不应该通过往现有 Planner 里继续堆 prompt 来实现。η0 的基础价值正在于，它已经把仿真环境、原子工具、任务指导、状态记忆、证据判定和实验生命周期分成了可替换的边界。未来无论换成更强的 VLM、更快的感知模型、真机控制器还是学习得到的 policy，都不需要重新定义“什么叫一次可信的具身决策”。

> 【图片占位 9｜η0 到后续版本的路线图】
>
> 建议画面：横向三阶段 roadmap。阶段一是单臂 PnP/铰链任务、客观成功率与 checker；阶段二是多任务 RLVR、rollout exporter、小模型训练和 auto research loop；阶段三是真机、双臂与移动操作。每个阶段下方分别标注“客观验收条件”，避免只画功能清单。

## 结语：模型很重要，但可信的闭环更像一个系统问题

η0 当前最有价值的结论，不是某个 VLM 已经学会了抓放，而是我们逐渐看清了一个具身 Agent 真正需要的“脚手架”是什么：稳定的原子工具，不会隐藏执行的文本技能，每次物理动作后的新观测，贯穿图像、标定和位姿的来源链，把成功与不确定分开的证据门禁，以及一套不会被单次偶然结果污染的学习生命周期。

从这个角度看，具身智能不只是让模型“想出一个动作”，而是让整个系统能够回答五个问题：你刚才看到了什么？你准备改变什么？谁允许你这样做？世界实际上发生了什么？这次经验在什么条件下还值得被相信？

当这五个问题都能留下结构化答案时，我们才真正拥有了一个可以评测、可以改进、也有机会走向真实世界的 Embodied Task Agent。
