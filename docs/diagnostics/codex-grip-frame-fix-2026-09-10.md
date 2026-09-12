# Atomic 夹持中心与手腕 body 坐标系混用修复

日期：2026-09-10；独立插件分支。20 任务 goal 继续，尚未计入任何任务成功。

## 新证据

在 campaign 的 Goal 3 attempt-001 中，四元数→Euler 修复后仍然多次空抓。
前两次闭合均为 no_contact，两指垫到把手横杆的间隙约 5.4–6.4 mm。
模型已根据新反馈松开重试，没有执行此前那样的空抓长距离拉动。

继续检查一次 iteration_limit 的保存状态发现：模型给出的 jaw_world=-Z，
实际 grip_site 的 +X 却接近世界 -X。源代码和真实模型共同确认另一项错误：

- robosuite robot0_eef_pos 来自 grip_site；robot0_eef_quat 却来自 robot_model.eef_name，即 right_hand body。
- Mink/IK 原有契约一致使用该 body 姿态，并在内部正确转换到 Mink site FrameTask。
- Atomic 错误地把这个 body quaternion 解释为物理 grip_site quaternion，公开 jaw_world 和局部位移也跟着错误。
- Panda 的 grip_site 相对 right_hand 固定绕局部 Z 旋转 -90°。
- 同次任务 33 个完整状态中，`R_body.T @ R_site` 与该固定变换的最大矩阵误差为 `3.55e-15`。

这也更正了 [上一份修复记录](codex-pose-contact-fix-2026-09-10.md) 的验证边界：
此前证实的是 agent quaternion → worker body quaternion 数值传递正确，
尚未证实该 quaternion 的物理坐标系与 atomic 对用户承诺的 grip_site 一致。
两个 90° 错误相互独立：一个是 Euler 奇异点丢失旋转，另一个是固定坐标系解释错误。

## 修复范围

共享 sim、IK、controller 的 body 姿态契约保持原样，仅修正 atomic 边界：

```text
R_world_site = R_world_body @ R_body_site
R_world_body_target = R_world_site_target @ R_body_site.T
R_body_site = [[0,1,0],[-1,0,0],[0,0,1]]
```

`tools/codex_atomic_geometry.py` 保存经过真实 Panda 模型验证的固定机器人标定。
`tools/codex_atomic.py` 覆盖实际状态的方向、grip_site 局部 delta、显式目标、保持姿态、
预览与对称候选选中反馈；接触授权继续绑定转换后的准确 body pose，不更换 IK receipt。
旋转距离在固定坐标变换下不变，候选的关节成本仍使用实际传入 IK 的 body pose。

未新增工具、场景真值或碰撞权限。标定针对当前 LIBERO Panda；其他机器人不能直接沿用。
main 未修改；没有提交、推送或外部消息。

## 验证与运行处理

- 原子工具/对称/反馈相关测试：45 项通过。首次新测试遗漏 pytest rig 导入导致 2 项 setup error，修正后通过。
- 新测试明确验证：body=identity 时物理 jaw=-Y；grip_site +X 位移映射到世界 -Y；
  两条实际 adapter 与 MCP quaternion 编码后的 body 姿态产生正确物理 jaw/approach；
  对称反馈使用 site 姿态，执行授权使用 body 姿态。
- 真实无模型 canary：同一个水平接近目标，67 步 target_reached；控制回执位置误差 0.728 mm、
  角度误差 2.340°，最小关节余量 0.49959 rad，无碰撞停止。
- **独立重建最终 MuJoCo 状态后，直接读取 site_xmat 与两指垫中心连线**：
  实际夹爪张开轴与请求 -Z 相差 **0.653°**，完整 site 姿态误差 **2.303°**；
  指垫连线与 site +X 一致。这次验证覆盖真实手指方向，不只比较收据或传输数值。
- canary 自有 Host 与服务已清理；没有调用模型。

Goal 3 attempt-001 在发现此错误后由操作员正常停止，保留为 interrupted，不能当作完成任务或自然失败终局。
当时记录到 28 个 native completed calls、42 个内部 tool calls、0 条断流；
清理确认无自有残留 PID、端口释放、私有 auth 副本删除、运行期间 source_changed=[]。
修复在停止后进行。随后以新源码快照重跑 attempt-002。

证据：`tmp/codex-libero20-analysis/body-site-calibration.json`、`goal3-attempt001-closes.json`，
以及 `tmp/codex-grip-frame-fix-20260910/live_host.py`、`live-host-report.json`、
`physical-frame-validation.json` 和真实私有前后状态。
