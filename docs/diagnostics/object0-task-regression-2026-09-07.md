# Object 0 完整任务回归 — 2026-09-07，R0 28–30

目标是完整 `pick up the alphabet soup and place it in the basket`，不是把感知通过定义成任务成功。当前尚未成功。个人重构分支保留原有改动；共享 RFC revision 2158 只读核对，没有提交、推送或更新共享文档。

## 入口和边界

复用 `scripts/perception_regression.py`，新增 **Object 0 专用显式任务模式**；其他任务及默认模式继续拒绝运动：

```bash
.venv/bin/python -m scripts.perception_regression object0 \
  --allow-authorized-test-data --allow-task-motion --episode-seconds 1800
```

用户已允许测试数据发往配置的小模型/外部 provider；本次只绑定 SAM3/8773、AnyGrasp/8774、AnyPlace/8775，以及配置的 `gpt-5.6-luna`。没有导入旧 memory/skills/playbooks；保留工具契约驱动的受限实验基线，不等同于带全部默认技能、Python artifact reader 的 CLI 配置。其他外部资料、代码执行、技能修改仍不可执行；不能将由这些限制产生的失败直接归因于生产能力缺失。

- 独立 `127.0.0.1:18766` simulator，LIBERO Python + 最小 Mink overlay，单 worker，服务硬上限 40 分钟。
- episode 1800 s，80 turns/80 tool admissions，128 次跨角色 LLM admission，1500000 已知 tokens；单 provider 请求 180 s、8192 output tokens，无 HTTP retry/fallback。
- 模拟器 RPC 90 s、感知 120 s；运动保留原 runtime/gate/receipt/未知结果停止机制。额外拒绝关闭碰撞检查，单 move 1–200 步，路径 1–5 点、每点 1–100 步。
- 累计最多预留 6000 controller steps；即使远端超时，已派发的完整额度也不退还。预留不等于实测执行步数。夹爪按当前服务固定 open/40、close/60 步计入；此计数依赖已检查的当前服务实现，不是任意未来 backend 的通用证明。
- 成功标记须经官方证据分类并有清理确认；退出码本身不是成功判定。预算/工具拒绝不授权换环境重试未知操作。

## 首轮：创建后校验耗尽，无运动

- Agent session `cd5893ee-da05-41a8-bce9-eadcfc5469b1`。
- Simulator session `de0605c3-cd0c-4d67-a5d7-1455d885b243`；handle `34bacc06-24a`。
- 原始报告：`tmp/perception-regression-yd681btx/report.json`。
- 实际创建成功，随后 SAM3 三次候选分别错用 `evidence_id`、错用 `text_prompt`、纠正 prompt 后丢失原先正确的 `source_packet_id`；三次均被 Host 拒绝，没有调用 SAM3 推理或执行运动。
- 2 steps，4 次 LLM 请求，38137 provider tokens，总耗时 189.746 s，controller 预留 0。
- `planner_validation_failed`、validation_attempts=3、truncated=true；未进入人工等待，human_assisted=false。这提供了 R0 26 失败分类在实际 provider 运行中的证据。
- close_state=closed、remote ok=true、cleanup_errors=[]；自建服务 PID 1237211 正常 SIGINT 退出（exit 0），未操作用户旧服务。

## 按证据修复纠错上下文

检查发送给模型的 SAM3 契约，确实已包含 `source_packet_id`、`prompt` 和 `additionalProperties=false`。不能以“模型看不到参数”为由放宽校验。但重试只提供错误列表，没有上一份候选，实测出现修改一个字段又丢失正确字段的往复。

主 planner 现回显最近一份已解析候选的 kind/name/parameters/code，并要求保留未被错误指出的字段及合法引用。最多 16384 字符，JSON 分离副本；超大内容或无法解析的 XML 不回显，不把 Host fallback 冒充模型候选。回显明确标记为不可信，仅供模型纠错，不自动补值/选动作/授权。isolated advisor 原有 JSON 协议不变。

新增 `tests/test_planner_repair_context.py` 验证候选继承、只执行修正结果、隔离角色不接收回显、大小限制/复制/解析失败边界；相关 160 项测试通过。任务入口及控制预算相关 135 项测试通过。这些夹具不证明任务性能提升，需继续实际复测。

另只读查询指定 provider `/v1/models`，返回 8 个模型 ID（含当前 luna、sol、terra、astra 等）；本次没有更换模型，不能仅由列表推断实际质量或价格。

## 第二轮：仍校验耗尽，无运动

- Agent session `88e612a8-38b2-4b89-b02a-d634af552871`。
- 新目录 `tmp/perception-regression-za1ym7ja/`；终态由该目录 `report.json` 与具体 RPC/执行进程共同核实，目录存在不等于进程还在运行。
- 同模型、任务、seed、预算，加入上述纠错回显；使用新自建服务，前一环境已确认关闭。不能把第一轮 trace 作为新 session 的任务上下文。
- Simulator session `7d0c6bea-faa9-4ac3-9683-ca08360aa994`，handle `1a375b73-a70`。
- 2 steps、4 次 LLM 请求、37872 provider tokens、总耗时 91.708 s，controller 预留 0。SAM3 先错用 frame_id，纠错后引用正确但 XML 未闭合，最后又错用 image_ref/text_prompt；全部被拒绝，没有感知推理或运动。
- `planner_validation_failed`、3 次校验、没有人工等待/协助；close_state=closed、remote ok=true、cleanup_errors=[]。
- 实际第二次候选带了回显；XML 解析失败后的下一次请求不带解析候选（符合已声明边界）。这次复测没有改善最终结果，不能声称提示修复有效，更不能标记成功路径完成。后续应比较已部署可用模型或 Human VLM，区分模型输出服从性和接口表达问题，不对同一错误无限重试。

## 验证与收尾

全量本地回归：2345 passed、21 skipped、1 个既有 authority/catalog 失败、36 warnings，61.78 s；真实模型 integration 显式关闭。JUnit `tmp/perception-regression-yd681btx/repair-local-tests.xml`。compileall/git diff --check 通过，未改旧 hash/审批。

两轮客户端均退出并收到环境关闭确认；第二个自建服务 PID 1248289 以 SIGINT 正常退出（exit 0），自有 worker 随服务清理。未操作用户既有服务或请求。下一轮新建任务前仍须核对端口/进程，不能依赖本文推断外部状态。

## R0 29：astra 对照在首次 provider 请求失败

只对本次进程设置 `OPENETA_LLM_MODEL=gpt-6-astra`，其余入口、任务、预算和运行代码同第二轮，没有修改用户配置。独立 simulator 已 ready；没有将历史 trace 作为输入发送。

- Agent session `b4270661-b867-418d-bc9b-8e5a5e188672`，原始报告 `tmp/perception-regression-nxiy19q0/report.json`。
- 首次 provider 请求约 75.181 s 后返回 HTTP 500，响应体包含 `server_is_overloaded`；不是本地 180 s 超时，也不是模型输出格式校验失败。没有 retry/fallback。
- 总耗时 76.537 s，1 次 provider admission、1 episode step、0 tool admissions、0 controller 预留。没有 create 尝试，simulator session/handle 为空，未生成或发送仿真 RGB/深度/掩码，没有 SAM3/AnyGrasp/AnyPlace 推理。
- `task_outcome=fail`、`task_success_claimed=false`；Host failure 为 `planner_provider_failed` / `transient_provider_failure` / `ProviderHttpError`，attempts=1、retryable=true。retryable 是错误属性，不表示本轮进行了重试。
- episode 与最后一步 truncated，`waiting_for_human=false`、`human_assisted=false`、`human_wait_s=0`。原始 fallback action 的 `ask_human` 仍留在 trace；模型日志的 validation accepted 仅说明该 fallback 可解析，不表示 provider 或任务成功。这是 R0 27 失败分类的真实外部 HTTP 错误证据。
- provider 未返回 usage，`token_usage_sources.unknown=1`；已知 token 总数 0 **不证明未计费**。
- cleanup 为 `ok=true, skipped=true, reason=no_create_attempt`；不是“关闭了一个环境”的证明。客户端 exit 0 仅表示报告正常写出。自建 simulator PID 1256996 在核实无 create 后正常 SIGINT 退出（exit 0），端口释放；未触碰旧服务。

本轮没有取得可比较的模型动作输出，不能从上游过载判定 astra 的任务能力或接口服从性。也不能将 model list 中存在该 ID 当作当前推理可用证明。本轮只新增实验记录，没有代码改动；沿用 R0 28 测试结果，不冒充新一轮全量验证。后续可在同一受限入口做另一个已配置模型的有界运行；若切换到完整 CLI 技能/工件读取配置，则须明确列为另一实验条件。

## R0 30：用户收紧成本边界，后续仅用 luna

用户明确要求不再启动 sol 对照，后续使用 `gpt-5.6-luna`；此前“另一个模型对照”的计划被此约束取代。不得因 luna 输出失败自动升级到 sol/astra 或其他高成本模型，未经新授权不做付费跨模型比较。保留已有默认配置，不修改为 sol。

收到指令时已经启动的 sol 运行被立即检查并尝试 SIGINT；信号发送时客户端已自行正常结束（No such process），因此不能声称成功取消了已发送的 provider 请求。

- Agent session `d82dccca-1139-4aab-b5df-84c938cebea5`，报告 `tmp/perception-regression-e0wqcmyb/report.json`。
- Simulator session `3a15bea7-f9d7-4cb6-a7ff-4db4ab8b3267`，handle `ea85d54e-39f`。
- 4 次 provider admissions，报告 37958 tokens，总耗时 121.031 s；tokens 不是货币账单。创建成功后 SAM3 校验耗尽，最终错误包括 unsupported `evidence_id` 和缺少 `source_packet_id`。task_outcome=fail，planner_validation_failed，无人工协助。
- 1 次工具 admission（创建），controller 预留 0；环境 close_state=closed、remote ok=true、cleanup_errors=[]。客户端 exit 0；专用 simulator PID 1270492 随后 SIGINT 正常退出（exit 0），端口释放。未操作用户旧服务。

本轮不启动任何替代实验，先落实成本约束；没有新增代码或全量测试。后续围绕 luna 的真实错误分析和必要的有界回归推进，不将付费模型更换作为恢复实验的前提。
