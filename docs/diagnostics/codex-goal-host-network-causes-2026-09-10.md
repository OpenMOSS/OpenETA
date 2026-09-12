# Goal 0–4：运动拒绝与 Broken pipe 原因调查

调查日期：2026-09-10。只读分析既有实验、实现、LIBERO 定义及当前代理配置；没有重新调用模型、没有更改 harness 或网络设置。实验目录：`tmp/codex-goal-0-4-20260909`。

## 结论与证据强度

| 问题 | 已证实的直接原因 | 尚未证明的部分 |
|---|---|---|
| Goal 0/3 抽屉接触拒绝 | 各 3 次 `contact_target_geometry_not_found`，0 控制步；LIBERO 柜子是 fixture，几何提取只遍历 objects，授权目录缺柜子/把手 | 仅补全目录后是否足以完成任务，仍需进一步验证关节部件授权与运动 |
| Goal 4 抬升拒绝 | 3 次 `attached_object_endpoint_collision`，0 控制步；携带碗的保守包围盒与酒瓶包围盒相交 | 是否真实接触还是保守/陈旧几何误报，既有公开回执不足以重建数值几何 |
| 模型只得到泛化拒绝信息 | Host 将失败统一为 `host_command_rejected`，motion hook 在提取失败回执前抛出，atomic 输出不含具体 stop_reason/恢复建议 | 修复后模型恢复能力提升幅度 |
| Goal 1 Broken pipe | Codex 在发送模型 WebSocket 请求时出现 EPIPE，发生在两个已完成/随后开始的工具之间；该连接已不能继续写入 | 是本地代理、上游节点、远端还是超时后的连接复用先触发关闭，现有日志无法确定 |

## 1. 抽屉：授权几何目录缺失，不是收敛太慢

原始回执明确是 `contact_target_geometry_not_found` / `contact_authorization_unresolved`，`steps_executed=0`。Goal 0 的 Host 命令日志第 21、24、30 行；Goal 3 第 8、14、20 行。

调用链：

1. `sim/unified_env.py:985` 的 `_extract_libero_objects()` 只遍历 `inner_env.objects`（第 1006 行附近）。
2. LIBERO `bddl_base_domain.py:383` 将 `objects_dict` 和 `fixtures_dict` 分别保存到 `objects` 和 `fixtures`。
3. `libero_goal/open_the_middle_drawer_of_the_cabinet.bddl` 将 `wooden_cabinet_1` 明确列入 `:fixtures`，普通 `:objects` 只有碗、奶酪、酒瓶、盘子。Goal 3 的柜子同样属于 fixtures。
4. `sim/mcp_server/server.py:1321` 附近拿 `_collision_objects` 调用 `resolve_contact_authorization()`。
5. `sim/mcp_server/collision.py:43` 对模型 RGB-D 点要求与候选几何在 2 cm 内匹配；候选目录根本没有柜子/把手，因此在该场景返回 geometry_not_found。

这不是“模型重复打点不够准”即可解决的问题，也不应该直接放宽 2 cm 阈值去把把手错误绑定到其他物体。

建议：Host 私有几何目录支持 fixtures / articulation 子部件，并让接触授权绑定到正确把手及其关节部件；不能简单把整个柜体当作可任意接触的目标。fixture 位姿应取实时 body/geom 变换，不能直接照搬自由物体的 7 维 qpos 切片。

## 2. Goal 4：携带物包围盒预检触发

Host 命令日志第 25、28、42 行的底层错误一致：携带 `akita_black_bowl_1` 预计与 `wine_bottle_1` 相交；code 为 `attached_object_endpoint_collision`。

Mink 分支在 `sim/mcp_server/server.py:1298` 附近先运行 `check_attached_object_collision()`，以目标 EEF 位置计算携带物 AABB，命中后直接返回，未进入 Mink 控制循环。因此本次三次拒绝不是 IK 不可达或 150 步不收敛。

`sim/mcp_server/collision.py:245` 附近的实现：携带物包围盒加 5 mm margin；对已有包围盒交叠，只有目标位置严格减小交叠体积时才允许退出，否则继续拦截。

仍需验证的建模风险：

- `_collision_objects` 是 reset 时的几何目录；`server.py` 自身在 `_refresh_attachment_proxy()` 注释中明确不是实时物体状态流。
- `_arm_attachment_proxy()` 用对象的 root position 构造 EEF 相对偏移；包围盒尺寸来自几何范围，root position 不必等于 AABB 中心。
- 携带物的 `relative_xyz` 与 dims 在该检查中只随 EEF 平移，不随 EEF 旋转更新。
- 代理状态是 tentative，不能把“启用了携带物碰撞代理”当成抓取已成功的证据。

这些是代码上可见的限制，但没有留存本次完整私有 attachment/obstacle 数值，不能据此宣布本次碰撞必定误报。应先记录私有实时几何、baseline/predicted overlap 与姿态关系，再决定是否需要修正；不应关闭碰撞检查绕过。

## 3. 失败信息在插件适配层丢失

`tools/codex_host.py:289` 将所有 failed/blocked command 返回为 `{code: host_command_rejected, status: failed}`。

`tools/codex_motion.py:stage()` 拿到这个 error 后立即抛出。后续设置 `motion_summary`、`collision_coverage`、`reason_code` 的语句没有运行。`tools/codex_atomic.py:196` 又只透传 hook 中已有的几个字段。因此模型实际收到的是：

- `motion_dispatched=true`：只表示请求已经送到远端，不能理解成物理运动已经发生；本次很多回执实际为 0 步。
- `failure_stage=move_to`。
- `motion_hook_stopped` / `host_command_rejected`。

底层已经保存了具体 code、0 步、碰撞类别和恢复提示，但模型没有得到这些信息。这也解释了模型多次换点/抬升后只能报告“Host 拒绝”。

建议优先修正：从失败 command 中也提取允许公开的结构化回执，保留 stop_reason、steps_executed、actual EEF、collision_class、controller_failure 和恢复类别。不要直接把含私有物体名字/几何的原始 response 整段返回。区分请求 dispatch 与物理 steps。

补充：Goal 2 虽然最终成功，也有 2 次底层失败 `mink_qp_no_solution` / `constraint_escape_preview_rejected`，分别执行 4 和 2 步后停止。说明泛化的 Host 拒绝实际混合了几何授权、碰撞和控制器失败等不同原因。

## 4. Broken pipe：模型网络连接写失败，具体关闭者未确定

Goal 1 本地事件观察时间（UTC+8，约 1 秒采样误差）：

| 时间 | 事件 |
|---|---|
| 23:22:30.321 | `move_to` 已完成 |
| 23:25:08.293 | Reconnecting 2/5，request timed out |
| 23:26:45.077 | Reconnecting 3/5，failed to send websocket request，Broken pipe / os error 32 |
| 23:26:59.341 | 恢复，开始 `gripper_control(open)` |
| 23:27:07.492 | 夹爪工具完成 |

两次工具之间约 269 秒间隔，包含模型请求、超时与重试等开销，不能全当成 TCP 断网时长。此时不存在一个持续 269 秒执行的机器人工具；错误文本明确指出模型 WebSocket 发送失败，不能误判为 MCP stdio 断管或控制器超时。

同一错误前后 30 秒内，10 个 HTTPS 探测有 4 个失败，涉及 ChatGPT、OpenAI、GitHub、国内代理目标；支持共同网络/代理路径异常的怀疑，但不是因果证明。整个 Goal 1 期间代理端口 30/30 可连接、ChatGPT 短请求失败 5/30、OpenAI 失败 6/30。短请求可用也不保证已建立的 WebSocket 可持续写入。

当前实际代理是 `/home/user/Workspace/mihomo/mihomo`，监听 7897，配置目录 `/home/user/Workspace/mihomo/config`。进程 stdout/stderr 都连到 `/dev/pts/1`，没有在该目录找到持久化运行日志。旧 Clash Verge 日志截止 2026-02-02，与此次实验无关，不能用于归因。没有读取终端输入或认证内容。

当前配置及备份显示 2026-09-10 11:26 附近发生过修改，不能把今天的规则/当前节点当作昨晚确切生效状态。此次调查没有更改它们。

可确定的机制是连接写失败；可能涉及连接被关闭后继续发送、代理/上游中断或超时恢复路径。缺少当时代理日志、TCP FIN/RST 时序与 CLI 更细传输日志，不能区分究竟哪一端先关闭，也不能声称一定是 WebSocket 不支持或某个固定 idle timeout。

下一轮可控诊断：保留带时间戳的代理日志与 Codex 传输事件，必要时仅记录 TCP 连接生命周期；固定路由节点，并与同一任务的流式 HTTP 路径作独立对照。先验证所用 Codex 版本支持的配置，再修改传输选择。

OpenAI Docs 的[配置参考](https://learn.chatgpt.com/docs/config-file/config-reference)区分 `supports_websockets` 与 SSE 的 `stream_idle_timeout_ms` / `stream_max_retries`。文档里的 SSE 默认 300000 ms 不能直接拿来解释本次 WebSocket 的 269 秒间隔，也没有凭此改配置。

## 产物

- `tmp/codex-goal-0-4-20260909/rejection-details.json`：11 次 Host failed command 的精简底层回执索引。
- `goal-N/host/host-commands.jsonl`：详细 Host 命令/工具结果与原始响应文件路径。
- `goal-N/event-timeline.jsonl`：模型流错误与工具事件时间。
- `network/samples.jsonl`：同步探测原始样本。

尚未改代码、未做新模型实验、未提交或推送，main 未写入。
