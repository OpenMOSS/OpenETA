# MCP client 的代理隔离

2026-09-07 个人重构分支候选；修复代码审查第 8 项的进程环境竞争。没有新增 Agent 工具参数、动作/观察 schema，也没有修改主机代理配置。

## 原问题与当前实现

原 `SseSimulatorMcpTransport` 在每次 list/call 外临时添加 `NO_PROXY/no_proxy`，退出时恢复旧值。用两个交错 context 的独立进程复现：A 进入、B 进入、A 退出，会移除仍在执行的 B 的 host；随后 B 退出又留下 A 的 host 和原本不存在的小写变量。即使每个请求的 finally 都执行，也不能按栈恢复并发写入。

当前 list/call 通过 MCP SDK 的 `httpx_client_factory` 创建每次连接独立的 HTTP client：

- 仅把本次 URL 的准确 host 配置为 direct mount，覆盖该 host 的各端口；支持 IPv4、IPv6、大小写和 IDNA 域名。不是对子域名的泛化授权。
- 不读改恢复 `NO_PROXY` 来包裹请求，不用全局锁把不同环境的网络调用串行化；某个调用退出不会覆盖其他代码新设置的环境值。
- client 仍保留 `trust_env=True`，其他 host 使用其既有环境代理/NO_PROXY 规则；跨主机 HTTP redirect 也使用相应目标的规则。没有改写 SDK 的 redirect 或 SSE endpoint origin 校验策略。
- 保留 SDK 传入的 headers、auth 和 timeout，TLS verify 保持开启并继续读取环境 CA 配置。无效 `SSL_CERT_FILE` 仍使构造失败，而不是被静默忽略。
- client 由 SDK 的 async context 关闭；正常、失败及超时路径都有关闭回归。顶层超时/错误分类保持原行为，不把停止等待升级为远端推理已取消。

这修复的是本代码不再写共享代理环境。其他代码若继续改写 `os.environ`，仍可能影响之后创建的 client；不宣称整个进程的所有代理配置已不可变。显式 proxy 配置、全系统连接池以及其他 HTTP 后端不在本批范围内。

## 依赖边界

已检查 `.venv` 与 RAG 的实际 SDK：均为 MCP 1.28.1，支持 `httpx_client_factory`。将项目依赖下限提高至本次验证过的 `mcp>=1.28.1,<2`；锁文件本来已锁定 1.28.1，本次只同步 requirement 下限，没有升级已锁定包或安装软件。

`uv lock --offline --check --no-cache` 通过。默认沙箱内 snap 版 uv 因启动 capability 限制未能运行，获准后在沙箱外执行相同离线检查；没有在线解析或下载。

## 验证与未验证项

- 本地路由测试检查真实 HTTPX client 的目标与非目标 transport，覆盖两个 MCP client 和普通 HTTP client 共存；合成 redirect 经公开 request 方法验证仍走其他 host 的代理。
- SDK 边界夹具覆盖 list/call、正常/失败/超时，并确认创建的 client 已关闭；两个真实线程复现交错退出，验证不再改动/恢复过期环境值。
- `tests/integration/test_mcp_proxy_loopback.py` 使用实际 MCP SDK、FastMCP 与本机随机端口 HTTP server，只发送固定字符串 `fixture-a/fixture-b`。在 HTTP/HTTPS/ALL_PROXY 指向不可用本地地址时，list_tools 与并发 call_tool 成功；环境变量不变，线程和监听 socket 清理完成。此夹具 1 passed，不是 Human VLM 或物体任务成功。
- 没有连接既有仿真/控制台服务，没有调用 SAM3/AnyGrasp/AnyPlace 推理，也没有获得新的感知数据发送授权。此前真实任务回归仍待授权和实施；此修复不能替代那些验收。

无需调整用户的代理环境即可采用新代码。依赖/路由兼容性仍应在部署审查中确认；按 OpenETA 协作技能保持个人分支候选，没有提交、推送或写入共享 RFC/Wiki。
