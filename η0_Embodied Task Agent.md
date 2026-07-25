Baseline工作的题目：A Framework for Benchmarking and Improving Coding Agents for Robot Manipulation

Baseline工作的摘要：
“Code-as-Policy” considers how executable code can complement data-intensive Vision-LanguageAction (VLA) methods, yet their effectiveness as autonomous controllers for embodied manipulation remains underexplored. We present CaPX, an open-access framework for systematically studying Code-as-Policy agents in robot manipulation. At its core is CaP-Gym, an interactive environment in which agents control robots by synthesizing and executing programs that compose perception and control primitives. Building on this foundation, CaP-Bench evaluates frontier language and vision-language models across varying levels of abstraction, interaction, and perceptual grounding. Across 12 models, CaP-Bench reveals a consistent trend: performance improves with human-crafted abstractions but degrades as these priors are removed, exposing a dependence on designer scaffolding. At the same time, we observe that this gap can be mitigated through scaling agentic test-time computation–through multiturn interaction, structured execution feedback, visual differencing, automatic skill synthesis, and ensembled reasoning–substantially improves robustness even when agents operate over low-level primitives. These findings allow us to derive CaPAgent0, a training-free framework that recovers human-level reliability on several manipulation tasks in simulation and on real embodiments. We further introduce CaP-RL, showing reinforcement learning with verifiable rewards improves success rates and transfers from simulation to real robot with minimal gap. Together, CaP-X provides a principled, open-access platform for advancing embodied coding agents.


Our contributions:
1. 将 cap-gym 拓展到人型机器人+灵巧手操作/导航任务和更多仿真环境中 （适配更多机型+更多任务）
2. 将 cap-bench 变得更公平，不强迫受测试的MLLM使用人类预设的skills lab，我们只提供一套用于参考的skill模板（适当减少agent的探索空间，加快探索速度），允许agent自己探索和构建最适合自己的harness，将MLLM+自行构建的harness作为受测试对象，返回得分
3. 在现有的cap-agent0 扩展到真机场景，验证这套方法控制真机的可能性（包含机械臂桌面操作任务和人型机器人移动操作任务）
4. 目前 Cap-RL 中仅开源了一个可用任务的 RLVR环境，我们制作了更多场景中的RLVR环境，并尝试将多种任务中rollout 得到的轨迹训练到一个code agent model中，观察code agent model是否可以用小参数量模型完成泛化到Code as policy任务
5. 结合Cap-RL 部分的改动，在cap-agent0 workflow之上新增了一层 cap-agent1 auto research workflow，实现自动化的轨迹收集/数据格式转换/新的 code agent Model （例如 qwen2.5 coder 7B）训练


Title: η0: Embodied Task Agent
Abstract:
Code-as-Policy offers a promising paradigm for controlling embodied agents through executable programs, yet existing studies remain largely centered on robot manipulation tasks, human-designed skill libraries, and limited reinforcement learning environments. In this work, we extend the CaP-X framework toward more general embodied robotics by broadening its task scope, evaluation protocol, real-world validation, and learning pipeline. First, we expand CaP-Gym beyond tabletop manipulation to include humanoid robot manipulation and navigation tasks across diverse simulation environments, enabling systematic evaluation of coding agents in richer embodied settings. Second, we redesign Cap-Bench to provide a fairer assessment of model capability: instead of requiring tested MLLMs to use a fixed human-crafted skill library, we provide only reference skill templates and evaluate the combined system of the MLLM and the harness it autonomously explores, adapts, and constructs. This protocol better captures an agent’s ability to build task-relevant abstractions rather than merely exploit predefined scaffolding.
Building on this benchmark, we extend Cap-Agent0 to real-world scenarios and investigate the feasibility of deploying Code-as-Policy workflows on physical robots, including both tabletop manipulation with robotic arms and mobile manipulation with humanoid robots. We further develop a broader suite of reinforcement learning with verifiable reward environments, addressing the limitation that the original Cap-RL release includes only a single usable task. By collecting rollouts across multiple RLVR tasks and training compact code-agent models, such as Qwen2.5-Coder-7B, we study whether small-parameter models can generalize to Code-as-Policy tasks through trajectory-based learning. Finally, we introduce Cap-Agent1, an automated research workflow built on top of Cap-Agent0 that performs trajectory collection, data conversion, and code-agent model training with minimal human intervention. Together, our work moves Code-as-Policy agents from benchmarked manipulation toward scalable, self-improving, and real-world deployable embodied intelligence.


Current Engineering Roadmap:

OpenETA 当前不再直接沿用原始 CaP-X codebase，而是重新构建一个更清晰、可维护、可扩展的 embodied task-agent 系统边界。新的工程路线是以 RLinf 作为仿真和具身环境基础，以轻量 Python agent runtime 作为 closed-loop tool-calling agent runtime，以 OpenETA 自有 adapter 层定义 simulator-agent 之间的协议和桥接逻辑。Code-as-Policy 不再作为整任务主执行循环，而是保留为 tool/skill backend 中可选的短 horizon、可验证代码片段能力。

1. Simulator substrate: 使用 RLinf 中更完整的 embodied simulation、real-world env、数据与调度相关能力，作为后续 CaP-Gym 扩展、RLVR 环境构建、轨迹采集和 sim-to-real 验证的基础。OpenETA 不在早期复制或重写 RLinf 的环境体系，而是优先实现稳定的 RLinf-backed simulator adapter。

2. Agent substrate: 使用 RAS planner agent 作为第一参考，构建 OpenETA 自有轻量 Python agent runtime；借鉴 Pi agent harness 中 typed tools、session/event log、skills 和 agent lifecycle 的设计，但不直接引入 TypeScript runtime。Codex 仅保留为 legacy/reference，不再作为主 agent substrate。

3. OpenETA adapter contract: 在 `adapter/` 中维护 OpenETA 自有的 simulator-agent 协议，将具身环境中的 task text、RGBD observation、robot proprioception、object state、reward、termination 和 env metadata 转换为 agent 可使用的上下文；同时将 agent 的 closed-loop tool call、结构化 command、skill 调用、可选 code_policy snippet 和执行意图转换为 simulator 可执行的动作。

4. Minimal executable loop: 先保持一个同步、单 episode、单步或少步的最小闭环，用 dummy simulator 和 dummy agent 固化 `EnvObservation -> EnvAction -> StepResult` contract；随后替换为第一个真实 RLinf env 和第一个 OpenETA lightweight agent session。

5. Benchmark harness: 在稳定 adapter contract 之后，实现任务列表、episode runner、success/reward logging、failure trace、视觉/状态差分反馈和 trajectory 保存。Cap-Bench 的新评测对象不只是 MLLM 本身，而是 MLLM + 其自行探索、构建和维护的 harness / tools / skills / optional policy snippets。

6. Tool, skill and memory workflow: 主执行方式采用 `observe -> tool(parameter) -> result -> observe` 的具身闭环控制。只提供 reference tool/skill templates 来缩小早期探索空间，不强制 agent 使用固定的人类 skill library。agent 可以基于执行反馈、环境观测和历史轨迹合成、修改、筛选并复用自己的 tools/skills，使 benchmark 更公平地衡量模型构建任务相关抽象的能力。

7. RLVR and training pipeline: 在多个 RLinf-backed embodied tasks 中构建带 verifiable reward 的环境，批量收集 closed-loop tool-calling trajectory，并转换为可训练数据，用于研究较小 agent model 是否可以通过轨迹学习泛化到具身工具调用和可选 Code-as-Policy snippet 任务。

8. Cap-Agent1 auto research workflow: 在 Cap-Agent0 的多轮交互、执行反馈、视觉差分和 skill synthesis 基础上，增加自动化研究流程，覆盖任务运行、轨迹收集、数据格式转换、模型训练、评测回归和失败案例分析。
