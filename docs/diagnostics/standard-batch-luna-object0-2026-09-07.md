# 标准 batch / Luna / Object 0 — 2026-09-07

用户要求以标准 batch 入口测试改动后的 harness，不再将受限的 `scripts/perception_regression.py` 作为标准入口替代品。按协作技能只读核对 RFC revision 2158；分支 `dev/huaizezheng/harness-refactor-2026-09-05`，未改运行代码或用户默认配置。

## 配置与复现

本次唯一 manifest：`tmp/batch-luna-object0-KisX8i/manifest.json`；单个 `openeta/libero_libero_object_task0-v0`、seed 0，原始任务文本 `pick up the alphabet soup and place it in the basket`。最多 80 turns、80 tool calls、1800 s、500000 已知 tokens。manifest `--validate-only` 通过。

```bash
env OPENETA_LLM_MODEL=gpt-5.6-luna OPENETA_LLM_FALLBACK_MODEL=gpt-5.6-luna \
  .venv/bin/python -u -m agent.cli.batch_eval \
  --manifest tmp/batch-luna-object0-KisX8i/manifest.json \
  --concurrency 1 --provider-concurrency 1 --model gpt-5.6-luna \
  --sim-url http://127.0.0.1:18766/sse \
  --batch-id standard-luna-object0-20260907 \
  --output tmp/batch-luna-object0-KisX8i/result.json
```

专用 simulator 使用 LIBERO Python、`mink_joint_velocity`、`/tmp/openeta-mink-canary-min`，GPU 0、单 worker，外层 2400 s 硬上限。未切换或重启用户旧 simulator。保留标准 batch RPC 超时、默认工具绑定、技能加载、视觉历史、模型请求重试和监督配置；没有手工删减工具或替换 system prompt。主模型与 fallback 均核实为 Luna、指定 provider，默认单请求 180 s、最多 3 attempts。episode 预算不是外部服务硬取消或完整计费证明。

使用新 workspace，没有设置 legacy memory/artifact roots。模型可见默认工具/技能指导属于标准运行配置，不导入历史 trace，不将源码/密钥作为测试内容发送。共享文档只读。

## 本轮身份与当前证据

- Agent session `ba549011-dd99-4e9c-8264-ec000f041c3d`。
- Simulator session `e64e9f39-eb3f-447c-ac03-e552bf0e8091`，handle `f6859d7a-481`。
- rollout 根：`tmp/batch-luna-object0-KisX8i/memory/sessions/ba549011-dd99-4e9c-8264-ec000f041c3d/rollout/`；客户端日志 `tmp/batch-luna-object0-KisX8i/client.log`。
- 实际首个请求：Luna、35 个工具、`place`/`pick` 技能、无匹配 playbook、16384 输出上限；第一轮之前 batch 环境适配器已完成创建并提供真实观察。
- 首次 `grasp_pose_estimate` 候选字段无效；模型改为 `observe`，实际执行成功。随后 SAM3 错用 rgb/text_prompt，经纠正为 source_packet_id/prompt 后真实分割成功。接下来选择工具错用 result_id，进入正常纠错。

## 最终结果与清理

标准 batch 客户端 exit 1，结果写入 `tmp/batch-luna-object0-KisX8i/result.json`：1 fail、0 success、0 need_human。总耗时 530.124 s；6 episode steps，5 次工具 admission，未耗尽轮次/工具/时间/token 预算，而是 `planner_validation_failed` / `EpisodePlannerFailure`。

实际成功执行顺序：`observe → sam3 → select_sam3_detection → grasp_pose_estimate → compile_grasp_seed`。环境由 batch 适配器在首次 planner 请求前创建，不计入这五个 Agent 工具调用。没有实际执行 IK preview、move、trajectory 或 gripper 操作；不能将错误候选中出现的动作名称当作已派发动作。

- SAM3 返回单个 detection，score 0.765625、area 2581；模型自行采用了“red and green alphabet soup can”的描述并选择目标。本轮没有验证所选物体的任务身份正确性；SAM3 与 selection 成功不等于选对任务实例。主审查者只读查看本轮 contact sheet，没有把人工判断回送给运行中 Agent。
- AnyGrasp 实际返回 20 个原始候选，14 个因物理夹爪宽度过滤，保留 6 个。独立 advisor 推荐未知候选 ID，被拒绝，advice 为 abstain/status=unavailable；主 Agent 继续自主选择并成功编译候选 `gpe-25e4c778ed6b4b6c-000`，compiled ID `f59b5643ac16b874d9f1`。
- 最后一轮三次候选：① move_to 错用 compiled_grasp_id/waypoint_role；② 改为 ik_preview_check，但 XML reasoning 标签未闭合；③ 又用 target_pose 调 move_to。全部拒绝，最终要求合法 `ik_receipt_id`。
- 最后请求仍完整提供 move_to 的必需 ik_receipt_id 契约。第二次尝试带上一份已解析候选回显；XML 无法解析后的第三次不带候选回显。这与已知纠错边界一致，不证明提示改动有效。
- 主 planner 共 12 次请求（每次 1 HTTP attempt），5 次候选接受、7 次拒绝；episode 报告 295328 tokens、provider 来源计数 12。provider limiter 共记录 13 次调用，包含独立 advisor；不将主 planner 统计冒充完整跨角色计费，也不从 token 数推断金额。
- 无 provider 过载/超时、无人工回答、human_assisted=false、waiting_for_human=false；episode 与最后一步 truncated。
- cleanup 为 close_state=closed、remote ok=true、cleanup_errors=[]。专用 simulator PID 1291268 随后 SIGINT 正常退出（exit 0），准确 PID 与 18766 端口均核实已释放。未操作用户旧服务或请求。

本轮没有修改运行代码、用户配置或重新跑全量单元测试；manifest 校验与 git diff --check 通过。实际 live 结果是未成功，不能用感知链路通过替代完整任务验收。

## 已定位的契约反馈缺口

首个请求公开 `grasp_pose_estimate.parameters` 仅允许 `bundle_id` 和可选 `backend_preference`，必需 bundle_id；模型却给出 object_description/evidence_ids。Host 反馈没有引导回 bundle 契约，而要求 rgb、depth、object_mask、intrinsics、camera_frame_id、scene_epoch。

`agent/runtime/planner.py::_validate_grasp_pose_estimate_parameters` 在缺少有效 bundle_id 时进入旧原始输入分支，导致公开契约与纠错要求不一致。当前 HEAD 中已经存在同一分支，不能归为本轮未提交重构新增。该错误不证明模型首次错误由 Host 导致，但会使后续纠错偏离公开接口。应审计 planner 公共输入验证与 Host 内部解析验证的边界；不能通过让模型拼原始路径绕过 bundle/provenance gate 来修复。

标准入口已比先前受限入口走得更远，但仍有字段混用和 XML 错误。单次运行、入口配置不同，不能据此宣称标准入口消除了格式问题或建立因果关系。后续优先处理公开契约与纠错反馈一致性、合法引用的就近呈现、解析失败后的纠错上下文；继续使用标准 batch/Luna 验证，不恢复付费跨模型对照。最初原始输入验证分支在当前 HEAD 中也存在，不能将此次发现本身视为重构新增回归。
