# Codex 原子工具不可见：实验接入修正

LIBERO20 campaign 当前仍为 4/20。该文件记录接入失败和只观察 canary；不能把连接验证计为机器人任务成功。

## 失败记录

2026-09-10 23:47 后恢复 pass5，Object 4 attempt-002、Object 5 attempt-001、Object 6 attempt-001
均在没有任何原生工具调用的情况下结束，模型称看不到六个 OpenETA 工具。Host 已初始化，
随后因 CLI 退出而关闭；`transport_closed` 是此处的关闭结果，不能直接解释工具为何不可见。
三次 integration_passed=false、官方失败，全部自有进程清理，端口释放、私有 auth 删除、source_changed=[]。
队列已在 Object 6 后暂停，未继续把所有任务逐项跑空。

CLI 为 0.153.4，二进制文件时间未在本轮改变。失败会话的模型目录中 Astra 声明
`tool_mode=code_mode_only`；原提示词严格限制六个直接原生工具名，与包装层访问存在不兼容风险。
仅增加允许 `functions.exec` 和 tools/ALL_TOOLS 包装访问的说明未稳定解决：

| 只观察诊断 | 实际方法 | 集成结果 |
| --- | --- | --- |
| observe-wrapper-001 | 临时追加包装层说明，未固定目录 | true，43.150 s |
| observe-production-001 | 实际启动器加入包装层说明，未固定目录 | false，20.249 s，0 次工具调用 |
| observe-native-catalog-001 | 保留实际提示词，仅在私有模型目录将 Astra tool_mode 设为 null | true，106.282 s |
| observe-pinned-production-001 | 实际启动器的新目录参数 | true，176.121 s，3 次预期调用 |

前述成功 canary 均为 episode_status → observe → finish_episode(false)，未执行 move_to 或 gripper_control。
全部输出在 `tmp/codex-tool-visibility-diagnosis/`；各自 cleanup.json 记录自有服务清理。
不能仅凭这些结果断言排除了所有 CLI 启动时序问题；后续长任务仍需观察，接入失败会立即停队列。

## 实现

- 启动器只允许六个 OpenETA 机器人操作，但允许 Codex 使用自己的函数包装层调用它们。
  不开放 shell、文件读写、外部网络、其他模型或其他应用工具；原有 shell_tool=false 保持。
- 增加可选 `--model-catalog-json`，将目录复制到每次运行的私有 Codex home，并在私有配置引用。
  保留生效提示词 `run/prompt.txt` 和私有目录副本哈希。
- campaign 保存目录路径，纳入每轮 source snapshot；恢复时继续使用同一目录。
- 目录来自本次实际获取的远程缓存，不改模型名称、权重、推理级别或模型指令，
  只将 Astra 的 `tool_mode` 从 `code_mode_only` 改为 null，使用当前 CLI 的原生默认工具模式。
  来源和字段差异记录在 `tmp/codex-libero20-20260910/native-model-catalog-provenance.json`。
  该目录不含 auth；用户的全局 Codex 配置和代理均未修改。
- 修正监督器：接入失败先停住排查，正常集成下的机器人任务失败仍可继续队列。
  清理检查包含私有登录副本；正常人工中断不误报为接入错误。恢复时清除上次错误状态。

启动器/监督器相关 16 项检查通过。实际目录参数的只观察验证结束后，才恢复完整队列。

最终实际启动器 canary 已结束，integration_passed=true、task_success=false、non_mcp_actions=[]；
remaining_owned_pids=[]、port_released=true、private_auth_removed=true。原目录固定与实际参数两次
canary 均成功。准备以 `--model-catalog-json tmp/codex-libero20-20260910/native-model-catalog.json`
恢复 Object 4–9、Goal 0–9；所有之前的零动作接入失败保留。

## Follow-up: tool-mode pin was not a complete fix

Object 5 attempt 002 again completed with zero native calls / zero internal tools and an unavailable-tools final response despite the same pinned native catalog that completed Object 4. Integration failed and the supervisor correctly stopped the queue; no robot motion occurred, and cleanup/source audits passed. Elapsed 20.060 s. This disproves any claim that the catalog override alone reliably fixed startup.

The installed Codex 0.153.4 binary contains `mcp_optional_startup_grace_ms`. Current official configuration reference documents a default 1000 ms wait for optional MCP servers in the initial tool catalog, and value 0 waits for each server's own bounded startup timeout. OpenETA's MCP process creates/resets the simulation before serving MCP initialization, so this default introduces a real startup ordering gap.

Source: https://learn.chatgpt.com/docs/config-file/config-reference (fetched Sep 11). Production launcher now writes `mcp_optional_startup_grace_ms = 0` only in its private Codex home; the plugin server's 180 s startup timeout remains unchanged. No global config was modified. Launcher/campaign tests: 16 passed in 0.14 s; diff check passed.

Delayed-start canary `tmp/codex-tool-visibility-diagnosis/observe-startup-wait-001`: an 8 s wrapper delay was injected only into that private installed plugin snapshot. The actual production launcher then completed episode_status -> observe -> finish_episode(false), 110.150 s, integration true, no non-MCP actions, no movement, full cleanup. This validates waiting under deliberately delayed startup; it is not an official task success. A second canary with the unmodified remote model catalog is being evaluated before choosing the next campaign configuration.

Hash clarification: `4c5724ac25e69f7f0ac691f0baedf0aab0d28246ade8bce2064ea7bd14ea12e7` is the native-override catalog file hash. The original preserved remote cache copied to `remote-model-catalog.json` is `f52b7bc5b87881d7d1cd39f395ca954d3d0511e7b06876bcbfa88e2f472e85cf` with Astra tool_mode=code_mode_only. The launcher reserializes its private copy and records that separate hash. Neither catalog changes the fetched model base instructions.

Second delayed-start canary `observe-startup-wait-remote-001` also passed: original remote catalog with Astra tool_mode=code_mode_only, same artificial 8 s delay and startup grace 0, 116.001 s, episode_status -> observe -> finish_episode(false), no non-MCP actions or motion, and complete cleanup. Therefore the next campaign uses the original remote catalog unchanged. The prior null tool-mode override is retained only as historical evidence, not used for subsequent runs. The successful paired canaries support the startup wait repair; a finite check does not prove absence of all future integration failures, so the queue's first-integration-failure stop remains.
