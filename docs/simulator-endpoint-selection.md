# 实验入口的 simulator 地址选择

2026-09-07 个人分支候选，待协作者评审。只改变 CLI 的 Host 配置/连接生命周期，不新增 Agent 工具参数或修改 `.mcp.json` 文件。

## 修复的错误连接路径

五个 `run_human_vlm_libero_*.sh` 脚本读取 `OPENETA_LOCAL_SIM_PORT`，但原先只用它做 `nc` 检查，执行 `openeta` 时没有传 simulator 地址。CLI 则独立读取 `.mcp.json`。因此选择 `18766` 可以通过新服务的检查，实际创建环境却仍可能走文件中的 `8766`。

此外，CLI 缺少 `openeta-sim` / 旧名 `openeta` 时，会把第一个任意 MCP 服务当作 simulator；配置重读改变 URL 时，也会直接替换 transport，即使旧 handle 仍然绑定。后续 cleanup 因而可能使用错误服务。

## 当前候选行为

- `openeta --simulator-mcp-url http://127.0.0.1:18766/sse ...` 显式固定本次 CLI simulator 地址，优先于文件；未提供时只接受 `.mcp.json` 的 `openeta-sim`，其次旧名 `openeta`。不再回退到感知服务。
- 五个 Human VLM 脚本均把刚刚检查的 host/port 作为该参数传入。其他用户脚本内容、任务和长时间默认预算没有被重写；仍应显式选择适合诊断的预算。
- URL 要求 HTTP(S)、host 和合法端口，不接受内嵌用户名/密码、query、fragment、空白或控制字符。无 override 仍按原命名配置读取；提供畸形值会失败，不静默退回旧地址。错误消息不回显潜在凭据。
- 有 active handle、closing/close_failed 状态、close_in_progress 或尚未退出的 runner worker 时，拒绝更换地址/删除连接。保留原 transport、catalog 和 handle，供原服务上的清理。创建请求尚未返回时，tracked worker 也会阻止更换。
- 空闲且已确认清理后可重新选择地址；先构造新 transport，再更新状态，构造异常不会把原 transport 的地址元数据改掉。移除空闲配置会清除旧 catalog。
- 运行中的 registry 投影显示实际 override 地址，保留其他感知服务，来源标记 `.mcp.json + --simulator-mcp-url`；不会改写磁盘配置或声称共享 registry 已变更。

这是单个 CLI 对象内的保护，不是分布式执行租约。未知远端创建、响应丢失后 handle 对账、脱离 runner 跟踪的请求、其他进程、旧 runtime 引用和已遗弃线程仍需独立治理；不能据此声称可安全热切换所有活动系统。

## 有界使用示例

先确认独立模拟器 `18766` 与控制台 `18099` 已按实验方案启动，且感知服务的数据发送已获授权。再在仓库根目录运行，例如：

```bash
OPENETA_LOCAL_SIM_PORT=18766 OPENETA_MANUAL_VLM_PORT=18099 OPENETA_SIMULATOR_TIMEOUT_S=45 \
OPENETA_HUMAN_MAX_TURNS=8 OPENETA_HUMAN_MAX_TOOL_CALLS=7 \
OPENETA_HUMAN_EPISODE_TIMEOUT_S=600 OPENETA_HUMAN_MAX_TOTAL_TOKENS=500000 \
bash scripts/run_human_vlm_libero_long9.sh
```

该脚本能够请求运动，不是只读感知驱动。Provider 的历史默认超时仍为 86400 s；2026-09-07 后续修复已让 simulator RPC 独立使用 `OPENETA_SIMULATOR_TIMEOUT_S`（默认 300 s，本例 45 s），不再被 provider 等待时间拉长。Runner 超时停止等待不等于 provider/远端操作硬取消，见 [预算边界](episode-budget-boundaries.md)。上面的显式工具配额变量由 Long 9 脚本消费；不要推断其余脚本也支持相同变量。Simulator timeout 变量已由全部五个脚本显式传入。

本轮没有执行此命令。端口检查和 argv 一致性通过实际 Bash + 本地替身 curl/nc/openeta 测试；无网络夹具验证了 override 传给 transport、registry 反映实际地址、旧环境只能在原 transport 上清理。它们不是 Spatial 0 / Long 9 live 验收。

## 本轮真实感知诊断的权限边界

已只读验证配置的 `10.11.39.173:8773/8774/8775` 根路径健康接口均 ok；`/health` 为 404 是路由差异，不是服务不可达。AnyGrasp `get_capabilities` 声明宽度 0.08 m、高度 0.03 m、深度截断 1.0 m、最多 20 个候选；没有调用推理接口。

准备好的独立诊断客户端在执行前被权限审查拒绝：将新仿真 RGB/深度/掩码等数据发到上述内网模型服务尚需明确授权。没有换通道、改地址或绕过拒绝；Spatial 0/Long 9 客户端未启动，没有新环境 handle、SAM3 推理或任务结果。仅启动的私有 `18766/18099` 服务已退出，原 `8766/8099` PID 与 pending=2 保持不变。

待用户确认具体数据/目的地授权后再运行有界感知和任务回归。候选驱动仅保留于本地 `tmp/experiment-ready-r0-12.FKtp6Q/perception_diagnostic.py`，尚未执行或验收，不当作可发布入口。
