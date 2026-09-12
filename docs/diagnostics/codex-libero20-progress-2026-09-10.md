# LIBERO 20 任务进行中记录

目标：Object 0–9 和 Goal 0–9 均取得官方成功证据；保留所有失败和重试。
完整状态以 `tmp/codex-libero20-20260910/batch-status.json` 及每次运行的官方结果为准。
本文件为阶段记录，不代表完成目标。

## Goal 3 attempt-002：首次有效拉动抽屉

采用 quaternion 奇异点修复、body/grip_site 坐标系修复、指垫接触反馈及恢复策略。
本轮从 2026-09-10 21:05 CST 开始，后面排队的是 Object 0–9 和其余 9 项 Goal。
本节保留运行中的阶段观察；最终结果见下方新增记录。运行期间未修改源码。

从原生动作日志与保存的完整状态对应得到：

| Native 请求 | 拉动命令 | 抽屉打开增量 | 动作结束时接触 |
| --- | ---: | ---: | --- |
| #12 | +Y 40 mm | 38.088 mm | bilateral_pads |
| #13 | +Y 180 mm | 39.201 mm | no_contact |
| #18 | +Y 30 mm | 25.204 mm | no_contact |

#11、#17 两次闭合均检测到双侧指垫接触。#12 是本 campaign 首次在拉动后仍保持双侧接触，
并由关节状态确认抽屉实际跟随的动作。#13 的长距离拉动发生脱手，不能将它说成第二次稳定的 40 mm 拉动。

另一个值得后续核查的现象：#16 重新接近时，抽屉 qpos 从约 -0.077433 变为 -0.052351 m，
即接近过程中先将抽屉推回约 25.1 mm；#18 又打开约 25.2 mm。
因此不能只看拉动动作的位移判断重新抓取产生了净进展。
仅这组状态不足以断言摩擦系数、速度、夹持力或某一接触点是脱手原因；后续动态对照见新增记录。

之后模型改为上方接近，#23 发生预测夹爪碰撞，#24 撤离成功，再次调整方向。
截至该阶段没有记录到模型断流。仍需完成抽屉打开与放碗，不能计作任务成功。

证据：`tmp/codex-libero20-analysis/goal3-attempt002-pulls.json`，包含请求号、原始参数、
motion_summary、起止快照 ID 和 drawer qpos；原始日志与快照在 attempt-002 内。

## 运行与完成边界

- attempt-001：发现 body/grip_site 坐标系错误后正常中止，保留全部证据，不计成功。
- attempt-002：40 分钟超时，官方失败；抽屉已开到约 159.9 mm，抓碗被柜体碰撞阻挡。
- 当前没有依据将任何任务或长期 goal 标为完成。
- 当前自有监督器以完整 20 项队列运行；运行结束后继续依据失败诊断修复、定向重试未完成项。

## 21:56 后续批次

Goal 3 attempt-002 已自然结束并完整清理，53 个 native completed calls、90 个内部 tool calls、
0 个 stream errors；integration_passed=true，task_success=false。
已完成成对控制回放及夹持稳定/碰撞反馈修复，详见
[codex-fixture-grip-control-2026-09-10.md](codex-fixture-grip-control-2026-09-10.md)。

pass3 从 Object 0–9 开始，之后 Goal 0–9；监督器负责每次安装新插件、保存源码和完整状态、
清理自有服务。维护边界启动过但未进入模型的 Object 0 attempt-001 保留为 setup 中断；
实际自主运行从 attempt-002 开始。尚未取得官方成功；后续以 batch-status.json 为准。

## 22:37：Object 0 成功，恢复剩余 19 项

Object 0 attempt-002 已通过官方成功与完整清理审计，当前 1/20。
43 native completed calls、77 内部 tool calls、0 stream errors。初次抓错番茄酱罐后模型自主纠正目标；
未把错误物体放置或操作员回放算作成功。

任务边界已合入 Host 路径批量查询、官方回执直读和持物姿态稳定控制，并完成回归与真实回放。
pass4 正在继续 Object 1–9、Goal 0–9。详细证据见
[codex-host-speed-and-held-grip-2026-09-10.md](codex-host-speed-and-held-grip-2026-09-10.md)。

## 22:48：Object 1 成功，当前 2/20

Object 1 attempt-001（cream cheese → basket）官方成功，integration_passed=true，
18 native completed calls、26 内部 tool calls、0 stream errors。全部自有进程清理，端口释放、
私有 auth 副本删除，source_changed=[]。启动至结束约 654.6 s；不同于 Object 0 的任务与操作序列，
不能据两轮时长差直接估计控制或 Host 修改的因果提速。

第一次垂直试提 #10 被掌部对非授权目标障碍的预测碰撞挡住，未执行物理步；
模型改为斜向试提 #11 成功。之后 #13、#14、#17 完成抬高、搬运与放置前下降。
这四次成功的持物移动实际启用了 attached_object 稳定控制，末态角度误差 0.074–0.149°，
均为 bilateral_pads，无 iteration_limit 或 local_convergence_stalled。#18 张开夹爪时环境官方终止成功。

证据：`tmp/codex-libero20-20260910/object-1/attempt-001/result.json`、
`tmp/codex-libero20-analysis/object1-success.json`。Object 2 已由同一监督器自动启动。

## 22:54：Object 2 成功，当前 3/20

Object 2 attempt-001（salad dressing → basket）官方成功，integration_passed=true，
16 native completed calls、22 内部 tool calls、0 stream errors，总运行 368.493 s。
remaining_owned_pids=[]，port_released=true，private_auth_removed=true，source_changed=[]。

#12 试提、#13 抬高、#14 搬运均启用 attached_object 稳定控制，分别 14/43/62 步，
末态角误差约 0.177°/0.085°/0.098°，指垫反馈均为 bilateral_pads。
#16 下降本身触发了官方成功并终止环境；没有在终止后补做张开夹爪。
按本 campaign 的官方 checker 成功口径计入，但不表述为已完成松爪后稳定放置验证。

证据：`tmp/codex-libero20-20260910/object-2/attempt-001/result.json`、
该 attempt 的 `run/host/atomic-commands.jsonl` 与 `run/final.txt`。

## Object 3 启动网络波动（运行中）

22:55:07、22:55:23、22:55:38 出现三次 Codex 请求超时重连，22:56:00 已恢复原生工具调用。
22:54:39、22:55:09 两轮探针中 chatgpt/openai/github/domestic 四条代理 HTTPS 同时在 TLS 建连阶段超时，
而本机代理 TCP 和国内直连正常；22:55:39 起代理 HTTPS 恢复。
这支持当时代理出站链路有短暂异常，尚不能仅凭这些探针定位具体节点或服务端原因。
探针为无鉴权 HEAD（HTTP 405/401 仅代表链路响应），不等同于真实推理流健康检测。
本地 getent DNS 探针持续 5 s 超时，但经 HTTP 代理的 HTTPS 可以成功，不能据此认定它直接造成推理超时。
保留本轮继续执行，未改动代理配置或重启任务。

补充：独立 `journalctl --namespace=mihomo` 中，22:55:39 前后的请求仍走日本节点，
22:55:45 起出现 `Tokyo-Hysteria2-8443`，22:56:00 Codex 原生调用恢复。
节点切换与恢复相邻，但切换前 22:55:39 的 HEAD 已恢复，不能据此断言切换是唯一恢复原因。
普通 `journalctl -u mihomo` 仅有启动日志，应读取独立 namespace。

## 23:06：Object 3 成功，当前 4/20

Object 3 attempt-001（BBQ sauce → basket）官方成功，integration_passed=true，
27 native completed calls、41 内部 tool calls、3 stream errors，总运行 700.366 s。
全部自有进程清理、端口释放、私有 auth 删除、source_changed=[]。
启动三次网络超时后恢复，不重启任务；首次接近瓶颈被相邻瓶体挡住，后续接触将目标碰倒。
模型自行撤离、重新观察打点、改抓倒下的瓶身。#21 闭合后双侧指垫接触，
#22 试提 7 cm、#25 抬高 19 cm、#26 搬运均达到目标且保持 bilateral_pads，
末态姿态误差分别约 0.109°、0.100°、0.097°。
#27 下降过程中触发官方成功，未到达原下降终点、也未松爪；不能将该步计为 target_reached。

证据：`tmp/codex-libero20-20260910/object-3/attempt-001/result.json`、
原生动作日志及 `network/samples.jsonl`。聚合审计由
`tmp/codex-libero20-analysis/audit.py` 生成 `campaign-audit.json`，包含全部失败和成功 attempt。
Object 4 已在同一冻结源码版本下启动。

## Object 4 运行中：识别纠正与 WebSocket 恢复

首次将 BBQ sauce 误认为 ketchup 搬入篮筐，#17 finish_episode(success=true) 被
`official_success_not_established` 拒绝。模型随后自行从标签纠正目标；未向模型注入操作员答案。
第二次持物长途 #25 成功，但用了 148/150 步，后续观察迭代余量。
下降 #27 触发 attached_object_world，随后抬高和重新测量；#31 预检 0 步拒绝，
短下降 #32 成功。#33 松爪后 ketchup 靠在篮筐边缘，官方未成功，模型决定调整。

23:18:08、23:18:25、23:19:08 为 WebSocket 发送 Broken pipe，
23:19:14 为 server before response.completed，23:19:38 恢复 move_to。
这些是原 CLI 中的四条重连错误，不是四次新任务。23:17:53 OpenAI/GitHub/国内代理
HTTPS 在 TLS 建连超时，但 chatgpt HEAD 正常；代理日志仍走 Tokyo-Hysteria2-8443。
无鉴权短连接探针不能证明推理 WebSocket 健康。未改网络配置，未重启模型或仿真。
本节只是运行中记录，尚未将 Object 4 计入成功。

## 23:46：Object 4 失败归档与反馈修正

Object 4 attempt-001 主动 finish_episode(success=false)，不是 40 分钟超时终止。
53 native calls / 103 内部工具调用 / 5 次重连，总运行 2332.288 s；官方失败、集成成功。
全部自有进程清理，端口释放、私有 auth 删除、source_changed=[]。当前仍为 4/20。

目标识别错误造成先误放 BBQ sauce；ketchup 后续横靠篮筐边缘，清除误放物体时掌部碰撞
并撤离受阻。持物到达目标的 14 次运动均为双侧接触，角误差 0.072–0.188°；没有收敛停滞
或步数耗尽。超过 60 s 的模型/CLI/网络等待间隔合计约 20.8 分钟，不能全归因于网络。

维护边界修复了“目标位置预检”与“当前碰撞”混淆、持物放置通道信息被原子反馈丢弃的问题。
不修改碰撞判定或控制器参数。实际源码 113 项检查通过，插件版本更新为
`0.1.0+codex.20260910154521`，细节见
[codex-placement-feedback-2026-09-10.md](codex-placement-feedback-2026-09-10.md)。
计划恢复 Object 4–9、Goal 0–9；Object 4 会创建 attempt-002，保留本轮失败。

## 2026-09-11 00:05：接入兼容修正完成，准备 pass6

pass5 的 Object 4 attempt-002、Object 5/6 attempt-001 均为 0 工具调用的接入失败，
并非机器人执行失败；已在第三次后停队列，三次都完整清理，当前仍为 4/20。
Astra 目录声明 code_mode_only，单独允许包装层调用未稳定解决。将原目录中 Astra 的
tool_mode 固定为 null（不改模型指令和模型本身）后，私有配置与实际启动器两次
只观察 canary 成功。完整记录见
[codex-native-tool-visibility-2026-09-11.md](codex-native-tool-visibility-2026-09-11.md)。

新增目录私有复制、哈希记录、输入提示保存，并让监督器在第一次接入失败后停止队列。
启动器/监督器 16 项检查通过。准备 pass6：Object 4–9、Goal 0–9；开始 Object 4 attempt-003。

## Sep 11: Object 4 attempt 003 and rotating proxy correction

Object 4 attempt 003 ended official false, integration true, 75 native requests / 157 internal calls, 1679.486 s, zero stream errors. Cleanup and source audit passed. Total remains 4/20. The queue paused at its requested maintenance boundary. A saved-state geometry diagnosis found that the held-object endpoint box failed to rotate with the EEF; see `codex-held-geometry-rotation-2026-09-11.md` for the fix, quantitative replay, limitations and 124 passing tests. The next pass resumes incomplete Object 4–9 then Goal 0–9 using the pinned native model catalog.

## Sep 11 00:55: Object 4 official success (5/20)

Object 4 attempt 004 passed the official checker with integration and cleanup audits true: 996.087 s, 45 native requests, 85 internal tools, zero stream errors, owned processes empty, port released, private auth removed, source_changed empty. The new private grasp quaternion anchor was present in saved move requests.

The model initially selected BBQ again, then autonomously recognized the mistake, retrieved the BBQ from the basket and returned it to the table before approaching ketchup. Two palm/outside-target collision stops during retrieval were followed by retreat and revised approaches. Ketchup first closed with single-pad contact; a short lift was followed by a raise that ended with bilateral pads. Subsequent stabilized carry reached in 103 steps with 0.118 degree final orientation error. Request 45 descent triggered official success / episode termination after 20 steps, before the commanded pose endpoint and before opening the gripper. This is official benchmark success, not independently verified released-and-settled placement. The legacy pose_diagnostics full_pose_outcome says execution_failed_after_validated_ik for such early termination; this must not be counted as controller failure when official success ended the episode.

The result follows the rotating-proxy fix but is not a matched causal comparison: the agent also changed its recovery strategy. No operator target identities/coordinates or previous-attempt memory were supplied. Current pipeline proceeded to Object 5 attempt 002 (attempt 001 had zero robot calls due the earlier native tool visibility integration failure).

Initial saved-state ray diagnosis additionally confirms ketchup centre visibility in agentview; complete robot occlusion does not explain the repeated misidentification. See `tmp/codex-held-rotation-diagnosis/initial-visibility.json`; centre-ray visibility does not prove label readability. The user has been asked whether future retries may use the previous model-authored failure summary. That preference is pending; no retry-memory behavior is implemented.

Object 5 attempt 002 subsequently failed integration before any robot call (20.060 s, zero stream errors, zero tools). The queue stopped automatically and cleanup/source audits passed. The initial-tool-catalog startup grace is now under delayed-start validation; see the native-tool-visibility diagnostic follow-up. Official count remains 5/20.

Pass 8 resumes Object 5–9 then Goal 0–9 after two delayed-start observe-only canaries passed. Private launch configuration waits for MCP startup (`mcp_optional_startup_grace_ms=0`); model catalog is the unmodified preserved remote cache, not the prior null tool-mode override. No retry memory or other policy changes.

## Sep 11 01:11: Object 5 official success (6/20)

Object 5 attempt 003 passed official success and full integration/cleanup/source audits: 353.103 s, 13 native requests, 19 internal tools, zero stream errors. Attempts 001 and 002 were zero-motion integration failures, so this was the first robot execution for Object 5. The agent selected tomato sauce correctly, maintained bilateral pad contact through lift/raise/carry (final angles 0.150/0.109/0.098 degrees), and official success terminated descent request 13 after 22 steps before gripper release. No collision stops or controller errors occurred; one initial symmetric orientation request was rejected for the empty/open precondition and replaced by strict mode.

The startup-wait fix with the unmodified remote model catalog now has a full-task success, and the next fresh Object 6 instance successfully called its tools. Queue continues without changing runtime source.

## Sep 11 01:14: Object 6 official success (7/20)

Object 6 attempt 002 passed full official/integration/cleanup/source audits. Attempt 001 was the earlier zero-motion integration failure; this was its first robot execution. The current startup-wait and original-model-catalog configuration remains unchanged. Official success occurred during the final downward move (2 steps), before release.

```json
{
  "task": "object:6",
  "attempt": 2,
  "official_success": true,
  "elapsed_s": 182.896,
  "native_calls": 14,
  "internal_tools": 22,
  "stream_errors": 0,
  "motion_reasons": {
    "target_reached": 6,
    "gripper_horizon_completed": 1,
    "episode_terminated": 1
  },
  "stabilized_reached_angles_deg": [
    0.14250623801628862,
    0.09584006130625226,
    0.09865728547016908,
    0.07970384658060671
  ]
}
```

## Sep 11 01:32: Object 7 official success (8/20)

Object 7 attempt 001 passed official/integration/cleanup/source audits: 1072.587 s, 37 native requests, 63 internal tools, four stream reconnect errors. The first approach tilted the carton (143 steps to the pose); closing produced finger-body-only contact. The agent opened, remeasured and changed jaw direction; the second contact approach reached in 16 steps and closing/short lift produced bilateral pads. A 130-step carry reached with 0.070 degree residual.

Two intended held-object rotations failed with `mink_controller_diverged` after 10 and 15 steps, at Cartesian errors 90.302 and 89.968 mm (the repeated-regression branch of the drift guard, not necessarily the >100 mm branch). An intermediate rotation reached in 42 steps. Multiple descent endpoints were rejected with zero physics steps / outside_receptacle_corridor. The agent adjusted at height and ultimately opened; official success triggered during opening request 37 after 9 steps. This demonstrates task recovery despite controller/placement limitations, not clean one-shot control.

Network timeline: 01:29:17 Broken pipe, 01:29:23/26/29 websocket closed before response.completed, retry notices 2/5 through 5/5. Subsequent native actions resumed and the task passed. No proxy configuration was changed. Full cleanup succeeded and no runtime source changed. The owned pause-after-current marker stopped the queue for matched saved-state weight ablations (zero model calls) before resuming Object 8–9 and Goal 0–9.

## Sep 11: held-rotation repair and pass 9

Nine matched weight ablations reproduced three recorded drift failures and isolated the excessive orientation cost. Six actual-controller integration runs then verified conditional weighting, unchanged carry/fixture behavior and a 300-step ketchup probe that reached in 203 steps. Final targeted tests: 94 plus 2 real-frame tests passed. See `codex-held-rotation-control-2026-09-11.md` for all metrics, limitations, source files and the narrowly extended potentially-loaded rotation budget. Pass 9 resumes Object 8–9 and Goal 0–9; official count remains 8/20.

## Sep 11 01:48: Object 8 official success (9/20)

First attempt passed full audits. Initial finger-body-only closure was opened and corrected; second closure/short lift were bilateral. A 142-step carry reached with 0.076 degree final error. Official success was triggered during descent before release. No new source changes were made during the task.

```json
{
  "task_success": true,
  "integration_passed": true,
  "elapsed_s": 371.351,
  "native_completed_calls": 18,
  "stream_errors": 0,
  "remaining_owned_pids": [],
  "private_auth_removed": true,
  "port_released": true,
  "source_changed": [],
  "internal_tools": 27,
  "motion_reasons": {
    "target_reached": 7,
    "gripper_horizon_completed": 3,
    "episode_terminated": 1
  },
  "last_action": {
    "request_index": 18,
    "tool": "move_to",
    "arguments": {
      "xyz_m": [
        0.005,
        0.271,
        0.065
      ]
    }
  },
  "last_steps": 3
}
```

## Sep 11 01:52: Object 9 official success; Object suite complete (10/20)

Object 9 first attempt passed full audits. All Object 0–9 now have retained official successes; earlier failed attempts remain in the campaign. Goal 0–9 are still pending.

```json
{
  "task_success": true,
  "integration_passed": true,
  "elapsed_s": 204.905,
  "native_completed_calls": 13,
  "stream_errors": 0,
  "remaining_owned_pids": [],
  "private_auth_removed": true,
  "port_released": true,
  "source_changed": [],
  "internal_tools": 19,
  "motion_reasons": {
    "target_reached": 5,
    "gripper_horizon_completed": 1,
    "episode_terminated": 1
  },
  "last_action": {
    "request_index": 13,
    "tool": "move_to",
    "arguments": {
      "point_id": "point-60efa27a4ce1",
      "offset_m": [
        0,
        0,
        0.14
      ]
    }
  },
  "last_steps": 26
}
```

### Goal 0 attempt 001 — official success, 2026-09-11 02:19

Pass 9 now has 11/20 audited successes (Object 0–9 and Goal 0). Goal 0 took 1682.729 s, 58 native calls / 115 internal tool calls, with 4 recovered stream errors. Cleanup, released port, private-auth removal and unchanged runtime source all passed.

Middle-drawer grasping was the difficult phase: repeated palm collisions near the upper handle, then finger-body contact and a missed shallow regrasp. Native model eventually selected a different diagonal jaw/horizontal approach, closed, and pulled in increments. Request 54/56/57 reached targets with 0.096/0.817/1.789 degree final orientation error; request 58 triggered official success after one simulation step, before its full requested endpoint. Grip feedback was finger_body_only, then single_pad, so this task did not activate the bilateral-pad stabilization branch. The result establishes successful drawer opening with this grasp; it does not establish bilateral-pad retention.

Read-only saved-state geometry/contact notes: `tmp/codex-goal0-contact-diagnosis-20260911/findings.md`. No internal identities, coordinates, or diagnostic conclusions were fed to the running model. No runtime fixes were made during this attempt. A roughly 5.5-minute native-call gap recovered without a new logged stream error; proxied short HTTPS probes worked during the gap and mihomo recorded no application warning. This does not identify the long-stream delay's cause.

Evidence: `tmp/codex-libero20-20260910/goal-0/attempt-001/result.json`. Goal 1 started automatically.

### Goal 1 failure and Goal 2 success — 2026-09-11 02:26

Goal 1 attempt 001: task false / integration true, 96.329 s, 8 native / 9 internal calls, no stream errors. Native model ended after two different blocked paths; matched saved-state replays identify wine-rack obstruction on approach and cabinet obstruction on retreat. Diagnosis: `tmp/codex-goal1-recovery-diagnosis-20260911/findings.md`. Goal 1 remains pending.

Goal 2 attempt 001: official success and all cleanup/source audits passed, 265.136 s, 25 native / 31 internal calls, no stream errors. Success occurred during request 26 descent after 3 steps, not at a fully reached placement endpoint. Total audited completion 12/20.

Pass 9 paused at the requested maintenance boundary, process reaped. Goal 2 exposed a missed tentative carry proxy for a wine-bottle neck grasp despite bilateral pad contacts. `_arm_attachment_proxy` uses a 12 cm center-distance envelope, so the proxy and carry stabilization were absent; native model repeatedly re-marked the bottle to keep contact authorized. Investigating and fixing this before the next pass.

### Carry proxy asset-origin correction

The Goal 2 attachment failure was specifically an asset-origin/AABB-centre mismatch, rather than a need to enlarge the 12 cm gate. `_arm_attachment_proxy` now consistently uses collision AABB centre and extents. 164 tests passed; matched actual first-lift replay changed 0-step failure to 19-step arrival, 1.467 mm / 0.1515 degree error, bilateral pads, with carried-object checks active. Details: `codex-attachment-origin-fix-2026-09-11.md`.

### Goal 1 attempt 002 — official success, 2026-09-11 02:36

Pass 10, plugin 0.1.0+codex.20260910183210. Goal 1 independently retried and succeeded with all cleanup/source audits passed: 254.567 s, 19 native calls / 30 internal calls, 0 stream errors. The first bowl approach and upward retreat were blocked, but this fresh model context continued, selected a different rim point and jaw direction, grasped bilaterally and lifted. The 6 cm and 11 cm lifts reached in 14/22 steps with 0.0847/0.0713 degree orientation error and attached-object stabilization active.

13/20 tasks now have audited official success. Goal 3 attempt 003 started automatically. Previous failures are preserved; no prior-attempt memory or private operator hints were supplied.

Goal 1 success detail: request 20 descent triggered official termination after 26 steps with 0.09436 degree error and bilateral pads; the gripper had not opened, so this is not a verified released/settled placement.

### Goal 3 attempt 003 — model-ended failure, 2026-09-11 02:41

Task false / integration true, 258.968 s, 22 native / 31 internal calls, no stream errors. Cleanup and source audit passed. The model opened the top drawer, released the handle, then the bowl-pregrasp transition was blocked after 13 steps and two retreat choices were rejected at 0 steps. It ended before grasping the bowl. The recorded 12 degree error at the blocked transition is an interrupted-motion error, not proof of unreachable IK or failed convergence.

The queue proceeds to Goal 4; Goal 3 remains pending. Saved-state velocity ablations are being run separately to distinguish local discretization from path geometry without changing the live runtime.

### Goal 4 attempt 001 — official success, 2026-09-11 02:46

Task success and all integration/cleanup/source checks passed. 325.536 s, 21 native / 38 internal calls, no stream errors. Model grasped the bowl, lifted, recovered an attached-object endpoint rejection by raising before traversing, and changed position after two endpoint IK-search failures. All 6 stabilized target arrivals retained bilateral pads, with final orientation errors [0.09299522822386551, 0.10218305294260908, 0.0825058892454825, 0.08532972059516035, 0.08118662323185467, 0.0992562745404879].

14/20 audited successes. Goal 5 started automatically; Goal 3 remains pending.

Goal 4 success detail: request 21 final 2.2 cm descent triggered official termination after 3 steps with 0.2115 degree error; no release had yet occurred.

### Goal 5 attempt 001 — official success, 2026-09-11 02:58

694.158 s, 49 native / 84 internal calls, zero stream errors. Task success and integration/cleanup/source audits passed. Native model used a closed empty gripper to push the plate, recovered several approach collisions using a tilted approach, and repositioned for later pushes. Final request 49 triggered official termination after 3 steps.

15/20 audited successes. Pass 10 paused after the task to correct intentional opposing-finger contact classification: normal internal pad contact can veto otherwise valid arm boundary recovery. Separate production replay confirms one formerly rejected motion now reaches in 5 steps while a real palm/bowl obstruction still stops at 3 steps. Details forthcoming in the dedicated diagnostic. Remaining tasks: Goal 3, 6, 7, 8, 9.

Opposing-finger classification and missing fallback-collision feedback are fixed and validated; see `codex-opposing-finger-recovery-fix-2026-09-11.md`. Final real replay retains the true 3-step obstruction and allows the previously rejected escape to reach in 5 steps. Pass 11 will continue Goal 6–9, then independently retry Goal 3. No retry memory is enabled.

### Goal 6 attempt 001 — official success, 2026-09-11 03:10

341.59 s, 25 native / 43 internal calls, zero stream errors; all integration/cleanup/source audits passed. After two blocked cream-cheese approaches, the model repositioned, achieved bilateral-pad grasp, lifted and carried above the bowl. Stabilized target arrivals: 4 with angles [0.13415812761920284, 0.058180134417407374, 0.08030329024576578, 0.08826728634393369].

16/20 audited successes; Goal 7 started automatically.

Goal 6 success detail: final descent to z=0.983 m reached in 27 steps with 0.08827 degree error; request 26 open triggered official success after 7 simulation steps.

### Goal 7 attempt 001 — official success, 2026-09-11 03:17

388.344 s, 34 native / 44 internal calls, zero stream errors; all integration/cleanup/source audits passed. The model grasped the stove control and attempted incremental turns. Two early turns were rejected because a fresh mark changed the existing closed fixture contact binding; a later turn was stopped by an actual finger/world collision, now correctly reported alongside the controller error. The model released, regrasped and continued turning until the official stove-on predicate succeeded. Saved joint state confirmed actual knob rotation during the attempt, not just EEF motion.

17/20 audited successes. Goal 8 started automatically. No fixture contact scope was broadened for this task.

Goal 7 success detail: a turn to the 30-degree jaw direction reached in 14 steps (1.2055 degree endpoint error); the subsequent 60-degree target was rejected at 0 steps. Request 34 open triggered official success after 22 steps. Do not describe this as a successful full 60-degree commanded turn.

### Goal 8 attempt 001 — official success, 2026-09-11 03:21

268.332 s, 20 native / 35 internal calls, zero stream errors; all integration/cleanup/source audits passed. A combined rotation/approach was blocked, then the model retreated upward, rotated in clearance, remeasured a rim, grasped the bowl and moved it onto the plate. Stabilized target-arrival angles: [0.10749698439968829, 0.10162775262372643, 0.16173327814955324]; contacts at those arrivals were bilateral pads.

18/20 audited successes. Goal 9 started automatically; Goal 3 remains pending.

Goal 8 success detail: a 4 cm descent was rejected by endpoint collision checking; a 2.5 cm descent reached in 7 steps with 0.1617 degree error. Request 20 open triggered official success after 3 simulation steps.

### Goal 9 attempt 001 — official success, 2026-09-11 03:26

265.538 s, 23 native / 31 internal calls, zero stream errors; all integration/cleanup/source audits passed. The native model grasped the wine bottle neck, verified a lift, tilted the bottle to the rack slope, raised and routed it above the support, then lowered it. The corrected geometry-centred carry proxy was active from the first lift, which reached in 19 steps with 0.1468 degree error and bilateral pads.

19/20 audited successes. Only Goal 3 remains; its fourth independent attempt started automatically with the same Astra medium configuration.

Goal 9 success detail: final 2.5 cm descent was rejected at the endpoint without execution. Request 24 open from the existing position triggered official success after 5 simulation steps.


### Goal 3 attempt 004 and empty-orientation fix — 2026-09-11 03:46

Attempt 004 ended task-false / integration-true after 1126.02 s, 77 native / 141 internal calls, zero stream errors. Cleanup and source checks passed. The model opened the drawer, repeatedly approached the bowl, then ended without grasping it. This was not a timeout or exhausted tool budget.

Matched real replay isolated unnecessary angular drift during some empty-arm translations. A conditional orientation hold is now applied to validated poses within the existing 0.05 rad band without claiming a grip or slowing the empty arm. Actual replay: request 43 changes collision-at-4 to reached-at-27 (1.604 mm / 0.339 degree); request 48 still collides with the cabinet; intentional rotation request 41 remains identical (40 steps). 72 tests passed. See `codex-empty-orientation-hold-2026-09-11.md`.

19/20 audited successes remain. Pass 12 will independently retry Goal 3 with plugin 0.1.0+codex.20260910194628, Astra medium, same task budgets and no retry memory.


### Goal 3 attempt 005 — controller fix retained, planning remains unresolved

273.553 s, 19 native / 33 internal calls, zero stream errors. Task false / integration true; all cleanup/source audits passed. The new hold weight kept the blocked bowl approach at 1.419 degree error after 14 steps, but it still encountered cabinet geometry. Two retreat choices stopped at 0 steps and the model ended voluntarily. Visual evidence from attempt 004 also shows the opened top drawer covering much of the bowl. This is not proof that all feasible routes are absent.

Next pass 13 changes only native Astra reasoning effort from medium to high, preserving the current simulator/plugin, 160 internal calls, 2400 seconds, procedural seed 0 and fresh context without retry memory or operator hints. Per-attempt configuration is explicitly recorded.

An independent reporting bug was also corrected at the idle boundary: campaign result counts used the last live poll rather than drained Codex events and final Host summary. Raw historical result files are preserved; the operator aggregate now re-derives final counts and independently requires final Host official success. Historical timeline counts may be lower by the last call; this does not affect the 19/20 success count.


### Goal 3 attempt 006 — official success; campaign complete, 2026-09-11 04:05

Astra high, 657.16 s, 54 completed native / 79 internal calls, zero stream errors. Same controller/plugin as attempt 005; fresh independent context. The model recovered repeated approach collisions, changed the approach and jaw direction, grasped the bowl rim, pulled it outward before a larger lift, and routed it above the drawer. Request 54 descent triggered official success after 7 steps while bilateral contact remained; no release occurred. See `codex-goal3-completion-2026-09-11.md`.

All Object 0–9 and Goal 0–9 now have independently audited final Host official success, integration success, empty owned-process lists, released dedicated port, deleted private auth and unchanged runtime source during execution. Total 33 finished attempts, 20 counted successes and 13 preserved unsuccessful attempts. All campaign private auth copies are absent. This is an iterative development completion at procedural seed 0, not a frozen-version benchmark or proof of released/settled placement on every task.

Final summary: `codex-libero20-results-2026-09-11.md`. Final controller fix validation: 72 tests plus three matched real physics cases. Campaign/launcher final-evidence fix: 17 tests. No commit/push or main worktree change was performed.
