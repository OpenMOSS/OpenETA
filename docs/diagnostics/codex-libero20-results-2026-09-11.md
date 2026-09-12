# Codex + OpenETA：LIBERO 20 任务开发通关记录

当前经审计的官方成功：**20/20**。已结束 33 个任务尝试，其中 13 个未计为成功；所有失败证据保留。

使用 gpt-6-astra、Codex 订阅登录、OpenETA 原子工具和 Mink；前期使用 medium，Goal 3 第六轮起试用 high，具体配置保存在各轮 summary.json。每轮是全新模型上下文，未加入跨轮失败记忆，也未向模型提供仿真内部身份、坐标或人工动作提示。

**统计口径：**这是允许修复和重试的开发通关验证，运行版本随修复变化；不是冻结版本单次 20/20 成功率。使用 procedural reset、seed=0，不是 LIBERO 官方 init-state 集合的基准评测。每项成功必须同时通过官方任务判定、集成检查、进程／端口清理、私有认证删除和运行源码未变检查。

| 任务 | 成功轮次 | 推理强度 | 成功轮耗时（分钟） | 原生／内部调用 | 流错误 | 证据 |
|---|---:|---|---:|---:|---:|---|
| object:0 | 2 | medium | 25.58 | 43/77 | 0 | [result](../../tmp/codex-libero20-20260910/object-0/attempt-002/result.json) |
| object:1 | 1 | medium | 10.91 | 18/26 | 0 | [result](../../tmp/codex-libero20-20260910/object-1/attempt-001/result.json) |
| object:2 | 1 | medium | 6.14 | 16/22 | 0 | [result](../../tmp/codex-libero20-20260910/object-2/attempt-001/result.json) |
| object:3 | 1 | medium | 11.67 | 27/41 | 3 | [result](../../tmp/codex-libero20-20260910/object-3/attempt-001/result.json) |
| object:4 | 4 | medium | 16.60 | 45/85 | 0 | [result](../../tmp/codex-libero20-20260910/object-4/attempt-004/result.json) |
| object:5 | 3 | medium | 5.89 | 13/19 | 0 | [result](../../tmp/codex-libero20-20260910/object-5/attempt-003/result.json) |
| object:6 | 2 | medium | 3.05 | 14/22 | 0 | [result](../../tmp/codex-libero20-20260910/object-6/attempt-002/result.json) |
| object:7 | 1 | medium | 17.88 | 37/63 | 4 | [result](../../tmp/codex-libero20-20260910/object-7/attempt-001/result.json) |
| object:8 | 1 | medium | 6.19 | 18/27 | 0 | [result](../../tmp/codex-libero20-20260910/object-8/attempt-001/result.json) |
| object:9 | 1 | medium | 3.42 | 13/19 | 0 | [result](../../tmp/codex-libero20-20260910/object-9/attempt-001/result.json) |
| goal:0 | 1 | medium | 28.05 | 58/115 | 4 | [result](../../tmp/codex-libero20-20260910/goal-0/attempt-001/result.json) |
| goal:1 | 2 | medium | 4.24 | 20/31 | 0 | [result](../../tmp/codex-libero20-20260910/goal-1/attempt-002/result.json) |
| goal:2 | 1 | medium | 4.42 | 26/31 | 0 | [result](../../tmp/codex-libero20-20260910/goal-2/attempt-001/result.json) |
| goal:3 | 6 | high | 10.95 | 54/79 | 0 | [result](../../tmp/codex-libero20-20260910/goal-3/attempt-006/result.json) |
| goal:4 | 1 | medium | 5.43 | 21/38 | 0 | [result](../../tmp/codex-libero20-20260910/goal-4/attempt-001/result.json) |
| goal:5 | 1 | medium | 11.57 | 49/84 | 0 | [result](../../tmp/codex-libero20-20260910/goal-5/attempt-001/result.json) |
| goal:6 | 1 | medium | 5.69 | 26/44 | 0 | [result](../../tmp/codex-libero20-20260910/goal-6/attempt-001/result.json) |
| goal:7 | 1 | medium | 6.47 | 34/44 | 0 | [result](../../tmp/codex-libero20-20260910/goal-7/attempt-001/result.json) |
| goal:8 | 1 | medium | 4.47 | 20/35 | 0 | [result](../../tmp/codex-libero20-20260910/goal-8/attempt-001/result.json) |
| goal:9 | 1 | medium | 4.43 | 24/32 | 0 | [result](../../tmp/codex-libero20-20260910/goal-9/attempt-001/result.json) |

调用数从完整 Codex 事件与最终 Host summary 重新汇总；早期逐轮日志中的监控快照可能少计最后一次调用，原始记录保留。

官方成功有时在下降或松爪过程中立即终止仿真。部分任务成功时尚未完全松开夹爪，因此不能统一表述为“已释放并静置稳定”。具体终止动作见逐轮记录。

## 保留的未成功尝试

| 任务 | 轮次 | 分类 | 原生调用 | 证据 |
|---|---:|---|---:|---|
| object:0 | 1 | 集成检查未通过或缺失 | 0 | [result](../../tmp/codex-libero20-20260910/object-0/attempt-001/result.json) |
| object:4 | 1 | 任务未成功 | 53 | [result](../../tmp/codex-libero20-20260910/object-4/attempt-001/result.json) |
| object:4 | 2 | 集成检查未通过或缺失 | 0 | [result](../../tmp/codex-libero20-20260910/object-4/attempt-002/result.json) |
| object:4 | 3 | 任务未成功 | 75 | [result](../../tmp/codex-libero20-20260910/object-4/attempt-003/result.json) |
| object:5 | 1 | 集成检查未通过或缺失 | 0 | [result](../../tmp/codex-libero20-20260910/object-5/attempt-001/result.json) |
| object:5 | 2 | 集成检查未通过或缺失 | 0 | [result](../../tmp/codex-libero20-20260910/object-5/attempt-002/result.json) |
| object:6 | 1 | 集成检查未通过或缺失 | 0 | [result](../../tmp/codex-libero20-20260910/object-6/attempt-001/result.json) |
| goal:1 | 1 | 任务未成功 | 8 | [result](../../tmp/codex-libero20-20260910/goal-1/attempt-001/result.json) |
| goal:3 | 1 | 集成检查未通过或缺失 | 28 | [result](../../tmp/codex-libero20-20260910/goal-3/attempt-001/result.json) |
| goal:3 | 2 | 任务未成功 | 53 | [result](../../tmp/codex-libero20-20260910/goal-3/attempt-002/result.json) |
| goal:3 | 3 | 任务未成功 | 22 | [result](../../tmp/codex-libero20-20260910/goal-3/attempt-003/result.json) |
| goal:3 | 4 | 任务未成功 | 77 | [result](../../tmp/codex-libero20-20260910/goal-3/attempt-004/result.json) |
| goal:3 | 5 | 任务未成功 | 19 | [result](../../tmp/codex-libero20-20260910/goal-3/attempt-005/result.json) |

已结束尝试累计记录 16 个流连接错误事件。零错误不代表没有响应延迟，短 HTTPS 探测也不等于模型长连接正常。网络事件与机器人失败分开保存。

## 主要修复与验证

- [姿态传输与夹爪接触反馈](codex-pose-contact-fix-2026-09-10.md)、[body / grip-site 帧一致性](codex-grip-frame-fix-2026-09-10.md)。
- [携带物体旋转后的碰撞几何](codex-held-geometry-rotation-2026-09-11.md)、[姿态保持与主动旋转的控制权重](codex-held-rotation-control-2026-09-11.md)。
- [MCP 启动等待与工具可见性](codex-native-tool-visibility-2026-09-11.md)。
- [模型原点与碰撞体中心混用](codex-attachment-origin-fix-2026-09-11.md)。
- [夹爪内部预期接触与失败碰撞反馈](codex-opposing-finger-recovery-fix-2026-09-11.md)。
- [空夹爪平移的姿态保持](codex-empty-orientation-hold-2026-09-11.md)。
- [Goal 3 最终成功过程与判定范围](codex-goal3-completion-2026-09-11.md)。

每项修复都保留针对性测试或真实保存状态对照；物理重放不计作模型任务成功。主 OpenETA 工作目录未修改，未提交或推送。私有分支中的接口／检查策略调整仍需在合入 main 前按协作约定评审。

详细时间线：[逐轮实验记录](codex-libero20-progress-2026-09-10.md)。可机读汇总：[CSV](../../tmp/codex-libero20-analysis/results.csv)、[JSON](../../tmp/codex-libero20-analysis/results-summary.json)。
