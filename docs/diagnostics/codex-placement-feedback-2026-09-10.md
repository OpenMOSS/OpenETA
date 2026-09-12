# Object 4 放置约束反馈修正

状态：Object 4 attempt-001 于 23:44 CST 主动报告失败并自然结束，完整清理且 source_changed=[]。
在维护边界已应用生产修改并通过验证，插件版本 `0.1.0+codex.20260910154521`。

## 可复现问题

Object 4 将 BBQ sauce 误认为 ketchup 放入篮筐；官方 checker 拒绝成功后，模型自行纠正。
ketchup 首次放置靠在篮筐边缘，重新抓取后 #43 抬高、#44 横移成功，#45 下降预检拒绝。
本次实际原因不是运动不收敛：server 的 `attached_object_endpoint_collision` 分支在派发 worker 前
用请求的目标位置检查持物 AABB，发现其超出篮筐保守内部 XY 通道。

私有原始回复要求约 +3.7 mm 世界 X 方向修正；该数值只用于操作员诊断，未送给执行模型。
原子公开日志只提供 `attached_object_world` 和通用 `replan_collision`，丢失了具体的放置通道约束。
此外，agent/tools/sim_mcp.py 的内部恢复逻辑把所有 0 步碰撞都解释为当前位置已在安全边界上；
这一推断不成立，目标预检也可在 0 步拒绝。**内层的这段错误文字未原样传给本轮 Astra**，
不能将它说成模型直接阅读了“当前位置已碰撞”并据此行动，也不能据此解释网络等待。

证据：
- `tmp/codex-libero20-20260910/object-4/attempt-001/run/host/atomic-commands.jsonl`，#45。
- 同目录 host-commands.jsonl 中最后一条对应的 move_to 回复及内部 recovery_options。
- 私有响应路径后缀 `7633c183c2-0058-move_to/move_to-response.json`。
- #44 对应快照 `1789053788235771351-881c4e5b-move-start`，包含持物 proxy 与真实状态。

## 最小修改

保持几何判定、保守边界、IK 和执行门限不变；不新增操作工具。

1. checker 在已有结果中增加有限枚举 `placement_constraint`：
   `outside_receptacle_corridor` 或 `receptacle_corridor_too_narrow`。
2. server 仅在该目标预检分支增加 `check_stage=target_endpoint`。
3. adapter 与 Codex 两层投影仅允许上述枚举，仍不公开名称、几何、距离、修正向量。
4. 目标预检反馈要求用图像将**整个持物**对齐开口、考虑持物相对夹爪的偏移，
   再执行受检下降；明确同一 XY 上反复抬高不能修复拟议放置。
5. 未知阶段的 0 步拒绝不再断言当前位置已碰撞；仍要求检查现场、保持碰撞检查。

schema 为向后兼容字段新增，若合入 main 前需要三人 review。
公开示例：
```json
{"collision_class":"attached_object_world","check_stage":"target_endpoint",
 "placement_constraint":"outside_receptacle_corridor","detected":true}
```

## 临时版本验证

`tmp/codex-placement-feedback-fix/prepare.py` 只生成临时副本和 change.diff，不修改运行源码。
`verify.py` 通过临时包 overlay 导入这四个修改模块，其他模块来自原仓库。
新增 tests/test_codex_placement_feedback.py 覆盖真实 AABB 判定、server 不派发 worker、
双层反馈、未知枚举拒绝和不泄露私有字段；合计 113 项相关测试通过。
首次新测试的字符串断言误把合法枚举后缀当作私有字典键，修正为完整 JSON 键后通过；
未修改 checker 来迎合断言。
真正安装生产源码后仍需执行同组测试并更新插件 cachebuster，由下一次全新 Codex exec 安装。
此处不声称反馈修改已改善自主任务成功率。

## 实际源码验证与任务结果

生产包中同组 113 项检查通过（1.98 s）；确认四份实际源码与临时验证副本逐字一致。
插件 manifest 验证和 git diff --check 通过。下一 attempt 由监督器创建独立插件安装和全新 Codex exec。

Object 4 本轮不是超时终止：模型在约 2315 s 的 Host elapsed 主动 finish_episode(success=false)，
总运行 2332.288 s，53 native calls / 103 内部工具调用 / 5 次重连错误，integration_passed=true、
official success=false。14 次启用持物稳定控制并到达目标的移动全部保持 bilateral_pads，
最终角误差 0.072–0.188°；另有一次持物运动被碰撞保护停止，没有 local_convergence_stalled
或 iteration_limit。共 6 次碰撞拒绝/停止，最后 #51 接近误放 BBQ sauce 时掌部碰撞，
#52 撤离也被停止。原生工具间大于 60 s 的等待合计 1249.317 s，约 20.8 分钟；
其中仅部分伴随明确网络错误，不能将全部时间归因于网络。

阶段诊断是识别错误增加场景阻碍、放置未满足整体几何条件和反馈不足，网络/模型等待
显著消耗墙钟时间。没有证据据此否定当前 Mink 持物姿态收敛；不因本轮失败改控制器参数。
详见 `tmp/codex-libero20-analysis/object4-native-waits.json` 和 campaign-audit.json。
