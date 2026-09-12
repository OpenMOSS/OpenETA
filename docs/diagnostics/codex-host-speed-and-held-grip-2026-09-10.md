# Host 提速及持物姿态稳定控制：已合入独立实验分支

2026-09-10 22:37 CST。长期目标仍为 Object 0–9 与 Goal 0–9 全部官方成功。
当前完成 1/20：Object 0 attempt-002 在抓取错误番茄酱罐后自主纠正，最终把字母汤罐放入篮子。
其官方 reward/termination 证据通过 Host 判定；task_success=true、integration_passed=true，
43 个 native completed calls、77 个内部 tool calls，0 个 stream errors。
自有进程无残留、端口释放、私有 auth 副本删除、source_changed=[]。
证据：`tmp/codex-libero20-20260910/object-0/attempt-002/result.json` 和 `run/summary.json`。
该成功使用本次维护前的控制与 Host；不得归因于下面新实现。

## 性能修改保持判定与路径归属语义

`CodexHost.success_evidence()` 从 `memory.latest_environment_receipt()` 直接读取现有可信回执，
继续使用原 `episode_success_evidence()`，不再为了这个字段构建整个规划上下文。
成功所需的执行 ID、session ID、官方/可信标志、二元 reward、terminated/truncated 条件保持原样。

相机证据路径查询通过 `find_packet_references_for_paths()` 在单次批量调用内合并规范化工作，
`AgentMemory` 至多重新加载一次持久 packet index，planner 一次查询当前图像的路径归属。
路径缓存不跨调用保存；最新唯一所有者、跨 frame 冲突、同索引冲突、缺失路径和 symlink 别名语义保持。
没有删除历史、减少模型上下文，或者放宽路径/packet 归属校验。

用相同已结束任务的离线 memory 副本，比较冻结的旧路径算法和实际新实现：

| 完整上下文 | 旧算法（无 profiler） | 新实现 | 上下文与官方判定 |
| --- | ---: | ---: | --- |
| Goal 3 失败会话 | 7.802 s | 1.201 s | 逐项一致，false |
| Object 0 成功会话 | 6.057 s | 1.050 s | 逐项一致，true |

新成功回执读取每次约 2.4–3.0 微秒。这里使用相同 dummy registry/default skills，
仅说明这两份离线重建的性能，不是整轮任务耗时的实测提速比例。
证据：`tmp/codex-host-speed-integrated-20260910/report.json`（passed=true）。

## 持物控制修改

先前的活动机构稳定控制扩展为：Host 有效 fixture 授权或自由物体 attachment_proxy
（tentative/confirmed）、闭合命令、双侧指垫接触、已验证 IK seed、有效目标朝向时可启用。
fixture 仍要求目标与当前朝向差 ≤0.05 rad；自由物体允许明确的姿态修正。
attachment_proxy 及双侧接触均不单独构成 retention_proven，不增加成功证据或接触权限。

实际变化仍为姿态权重乘 25、关节速度上限不超过原值与 0.2 rad/s 中的较小值，
并在位置/姿态和速度容差内至少稳定 3 个控制步。原本更慢或等待更多步的配置继续保留。
继续检查 robot/world、self 及持物碰撞；不可达/碰撞目标不会因启用该控制而获得授权。
自由物体的大角度搬运旋转未在这些回放中单独评估，不能将小角度结果外推为任意旋转保证。

- 新环境选项：`OPENETA_LIBERO_GRIP_STABILIZATION=1`，仅独立 Codex 服务显式开启。
  worker 接受旧 fixture 环境变量作为缺省兼容；服务显式覆盖并移除旧变量。
- `--no-grip-stabilization` 可关闭；旧 CLI 拼写 `--no-fixture-grip-stabilization` 保留为别名。
- 回执新增 `grip_stabilization_active`、`grip_stabilization_kind`（none/fixture/attached_object）、
  `grip_orientation_cost_multiplier`；旧 fixture 字段只描述 fixture 情况，未冒充自由物体控制。
  `transport_profile` 区分 fixture_grip_stabilized、attached_object_stabilized 与原始模式。
- 控制步峰值角度误差现在也纳入最终读取的姿态，覆盖终止分支最后一步的反馈。

真正调用最终 worker 代码，未 monkeypatch 控制器参数：
14 组 Goal 3 fixture/张开接近条件，加 16 组番茄酱与字母汤持物条件全部通过。
原始无接触张开接近结果逐项一致；稳定持物成功条件保留双侧指垫接触，
两个原本姿态停滞的搬运变为到达；篮筐真实碰撞保持停止。
字母汤长搬运需要 146/150 步，仍需在之后任务观察迭代余量。
证据：`tmp/codex-grip-final-{fixture,carry}-20260910/report.json`，两者 passed=true、cleanup=true。

上述回执新增字段是兼容扩展，合入 main 前仍需要三方审查；没有外发共同架构文档。
没有新工具、场景真值提示、摩擦/夹持力调参或模型 API 调用。

## 测试与恢复实验

- 针对性回归 130 passed；包括路径差分/别名冲突、可信成功回执正负样例、Host、MCP stdio、
  controller gating 和 campaign 计分条件。
- 初次在沙箱运行时停在 MCP stdio 子进程退出检查；相同完整套件在非沙箱环境 3.43 秒通过。
  旧测试进程已确认终止并回收，未修改测试成功条件。
- 清理过程中发现 `scripts.codex_campaign` 在导入时覆盖进程信号处理器，导致旧 pytest 忽略 SIGTERM。
  将注册移至 main，新增导入不覆盖调用者 SIGTERM 的测试；campaign 8 passed。
- 语法及 tracked diff whitespace 检查通过；插件校验通过。
- 插件新版本 `0.1.0+codex.20260910143608`，通过标准 cachebuster helper 更新。
- 自己创建的 `pause-after-current` 维护标记已删除；pass4 已启动，Object 1–9，再 Goal 0–9。
  每项仍有独立订阅登录 Codex exec、私有插件安装、源码审计与自有进程清理。
  Mink、Astra medium、seed=0 程序化 reset、2400 s/160 次限制保持。

所有更改在当前 Object 0 结束与下一任务启动之间进行。main 未改动，没有提交、推送或对外消息。
