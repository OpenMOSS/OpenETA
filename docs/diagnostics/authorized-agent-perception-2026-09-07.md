# 授权后的纯 Agent 感知诊断 — 2026-09-07

本轮用户明确允许：新建 LIBERO 的 RGB、深度、掩码及标定可发送至 `10.11.39.173` 的 SAM3/8773、AnyGrasp/8774、AnyPlace/8775；框架测试信息可发送至配置的 LLM API provider。不发送源码、密钥内容或历史 trace。

使用个人分支 `dev/huaizezheng/harness-refactor-2026-09-05` 的未提交候选，只读核对共享 RFC revision 2158。没有修改共享文档、历史 authority hash 或既有服务。此前的数据发送授权阻塞已解除；目标服务查询仍显示旧 `blocked` 状态，本轮没有将其冒充为已完成或重新定义目标。

## 实验边界

新入口 `scripts/perception_regression.py`：复用生产 runtime/planner/gate/registry/MCP 链路，真实配置模型 `gpt-5.6-luna`（`open.xiaojingai.com`）。没有人工代答、固定模型响应或自动 fallback；不导入旧 memory、skills、task playbooks。

- 私有 simulator `127.0.0.1:18766`，现有 LIBERO Python + 最小 Mink overlay，单 worker；不占用已有 handle。
- 只绑定 SAM3 和 AnyGrasp，AnyPlace 本轮未绑定/调用；其他感知服务不启用。
- 工具白名单保留环境创建、观察、分割/选择及抓取几何；禁止运动、夹爪、Python、外部资料与技能修改。底层 transport 再次拒绝运动，创建尝试最多一次，即使超时也不重试创建。
- 每轮最多 8 turns、7 tool admissions、12 次跨角色 provider admission，每次最多 8192 output tokens、60 s，禁用 HTTP 重试及 fallback；最终入口默认 episode 300 s，允许显式 30–600 s。
- episode 时间预算不是远端取消保证；已派发请求仍可能在单请求期限内结束。报告总耗时包含清理，不能据此声称全路径硬实时。
- 新建目录记录 trace、模型调用和报告；`finally` 尝试关闭具体 handle，关闭异常也写入报告。未尝试创建与“创建后没收到 handle”分开记录；后者不伪造关闭确认。

## Spatial 0

前两次启动基线均未派发 simulator RPC：

| Agent session | 输出目录 | 结果 |
| --- | --- | --- |
| `09e250dd-0752-4fe4-a21a-ac2e5c82add3` | `tmp/perception-regression-iwshl09y/` | 1 次 provider 请求；输出 `kind=tool`，校验拒绝。诊断白名单也遗漏创建/选择工具，随后修正 |
| `0189f746-e27c-4c44-b11c-0c3486c27b5a` | `tmp/perception-regression-pn7mf84w/` | 2 次 provider 请求；先错误 kind，再错误 `observe.camera_ids`；无工具执行 |

旧报告中的 `no_returned_handle_to_confirm` 不表示已经泄漏环境：对应 `remote_calls=[]`，根本没有创建尝试。后续入口修正这一报告区分，旧文件保留原样。

第三轮完成真实分割诊断，未完成操作任务：

- Agent session：`a292909b-15cc-49c2-861b-d564011ccd03`。
- Simulator session：`92c40cc8-cc74-40dc-a554-8d0aa8b02c24`；handle `6b2f1f07-d69`。
- 环境 `openeta/libero_libero_spatial_task0-v0`、seed 0；controller `mink.robosuite_joint_velocity`，没有运动调用。
- 报告：`tmp/perception-regression-me8oj4cp/report.json`，同目录 `memory/sessions/<Agent session>/rollout/model_calls.jsonl` 和 `tool_calls.jsonl` 可定位格式修复及分割证据。
- 3 个 episode step、6 次 LLM 请求、2 个工具执行额度、provider 报告 81436 tokens；总耗时 221.713 s。本次仍使用旧默认 episode 600 s。
- 实际工具链：`create_simulator_env → sam3 → talk`。SAM3 prompt 为 `black bowl between the plate and the ramekin`，source packet `obs-0001`，3 个候选分数为 0.72265625、0.72265625、0.5078125，mask 面积 2926、2376、1291 px。
- 主 Agent 只读查看候选 contact sheet 作证据判读（没有把判断反馈给运行中的测试 Agent）：两个相邻黑碗和后方部分遮挡的容器均被分割出来。关系式 prompt 不能直接充当实例身份确认。
- Agent 未选择候选，在 reasoning 中以身份歧义为由停止；未执行 AnyGrasp、抓取或运动。最终 `talk` 参数为空，说明“停止原因只留在 reasoning，用户消息缺失”的可观测性缺口仍存在。
- 校验纠错期间出现 XML 缺闭合标签、虚构 `evidence_id` 和 `parameters` 非对象；正确引用后 SAM3 才执行。不能将工具成功等同于语义目标确认。
- 关闭确认：`close_state=closed`、remote `ok=true`、`cleanup_errors=[]`。

这证明已恢复一条**真实 LLM → 仿真观察 → SAM3 → 结果判断**链路，不证明 Spatial 0 pick-place 成功或多视角主动感知已完成。

## Long 9

- Agent session：`18d52a8c-16e3-4b22-ae7f-20debb4cb8a6`；报告 `tmp/perception-regression-gsn56ztk/report.json`。
- Simulator session：`93b50fef-d5ed-4a3d-b134-9d9ef2c603f1`，handle `61f5b96c-9e7`；环境 `openeta/libero_libero_10_task9-v0`、seed 0，仍为独立 Mink 服务。
- 实际工具链：`create_simulator_env → sam3 → select_sam3_detection`，三项均 operational success。目标 `yellow and white mug` 返回唯一候选 `detection_000`，score 0.90625、面积 5155 px，并成功选择。曾将 SAM3 参数写成 `image_ref/text_prompt`、将选择参数写成 `result_id`，经现有校验反馈纠正。
- 4 个 episode step、6 次 LLM admission、3 个工具执行额度、总耗时 184.595 s（episode 上限 300 s）。5 次已收到响应的 provider usage 合计 66035 tokens；第六次超时请求的实际服务端耗费未知，不能记成零。
- 第六次 provider 请求在 60.145 s 后 read timeout，backend 返回 `ask_human`，没有启动 fallback/重试；未调用 AnyGrasp、AnyPlace、抓取或任何运动。未覆盖杯柄抓取、腔体放置或关门约束。
- 关闭确认：`close_state=closed`、remote `ok=true`、`cleanup_errors=[]`。
- 新暴露的统计/分类缺口：`failure_reason={}`、`stop_reason=ask_human`、`waiting_for_human=true`，约 0.019 s runner 等待使 `human_assisted=true`，尽管本轮没有任何人工回答或指导。这是原始统计，不将它改写成 false；应将请求人工、实际收到协助和 provider 基础设施失败分开。此项**尚未修复**，与本轮已修的校验耗尽分类不是同一个问题。

两轮纯 Agent 感知阶段都有实际证据，但已有成功路径完整回归、AnyGrasp 恢复和 Long 9 深层故障仍开放；不能将本轮结果称作“已完成真实任务实验整体验收”。

## 本轮代码修复与待评审边界

1. 每次主 planner 的请求说明重复现有 `tool_call/response` 类型和现有工具参数约束；isolated advisor JSON 契约不变。没有为错误类型增加别名或放宽 gate。第三轮启动通过不能单凭一次样本归因于提示修改。
2. 单测先复现 planner 校验耗尽被摘要成普通 `status_report`、`failure_reason={}`。现将 Host 生成的校验失败记录为 `planner_validation_failed`，episode 和最后一步均 truncated，不再普通终止；并保留错误及尝试数。批量错误类型为 `EpisodePlannerFailure`，不再错误称作资源预算耗尽。必须同时有 Host planner 的失败校验元数据，模型仅提交同名 response 参数不能伪造该状态。
3. 普通 `talk` 保持原行为。格式恢复、非空用户消息、其他 provider 失败分类及降低轮次成本仍需后续验证。

第 2 项改变 episode 终止/失败呈现，作为本地候选注明需要三人 review；没有修改官方 reward/task success 授权，也没有改写旧实验结果。
该分类修复目前是单测证据：Spatial 0 客户端在修复前已启动，Long 9 未触发校验耗尽分支，不以这两次 live 运行冒充新失败分类的实际验收。

## 本地验证

- 先复现：新的失败分类测试 1 failed、2 passed；修复后通过。
- 扩大定向测试：138 passed（随后再加入 3 个非法时间预算用例，包含在下述全量中）。
- 全量 `tests`，真实模型 integration 开关显式关闭、允许自建 loopback：2327 passed、21 skipped、1 failed、36 warnings，63.31 s。
- 唯一失败仍为 `test_reviewed_tool_contract_migration_is_complete_and_narrowly_authoritative` 的历史 authority/catalog 绑定，原因沿用 [上轮矩阵](local-validation-matrix-2026-09-07.md)。未重写旧 hash/审批；可选依赖跳过与 Pillow warnings 未隐藏。
- JUnit：`tmp/perception-regression-me8oj4cp/post-fix-local-tests.xml`。这些单测与 live episode 分开统计，不能互相替代。

## 复用命令

先检查 18766 空闲，并启动自有、有限生命周期的当前代码 simulator；不要指向他人实例。已有最小 Mink overlay 只是本机路径，不是可移植安装说明。

```bash
.venv/bin/python -m scripts.perception_regression spatial0 \
  --allow-authorized-test-data --episode-seconds 300
.venv/bin/python -m scripts.perception_regression long9 \
  --allow-authorized-test-data --episode-seconds 300
```

两条命令顺序执行且确认前一环境已关闭。授权标记只是显式 opt-in，不会替代真实用户的数据发送许可；报告文件位于新建 `tmp/perception-regression-*`，不保证随仓库分发。

本轮结束时两个环境均关闭确认，测试客户端及本地测试进程已退出；自建 simulator PID 974791 收到 SIGINT 后正常退出（exit 0）。未操作已有 8766/8099 服务及其旧请求。
