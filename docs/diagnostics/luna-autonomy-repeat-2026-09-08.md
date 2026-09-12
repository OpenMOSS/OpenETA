# Fresh Luna repeat after gate simplification

User requested another Luna experiment. No code/prompt changes in this turn;
standard `agent.cli.batch_eval`, same original Object 0 task and seed 0,
`bundle_stage3`, Mink joint-velocity controller. Main/advisor/fallback: Luna only.
Budgets: 15M cumulative known tokens / 160 turns / 320 tools / 10800 s. No oracle
target identity/coordinates, no old trace upload, no cross-model comparison.

Branch: `dev/huaizezheng/luna-task-recovery-2026-09-08`, uncommitted code preserved.
Pre-experiment tracked diff SHA256:
`b04d5ee3edc64738874868fbfbce8b7ed4277bfa32542b8a7512bc7e283b9aeb`
(not a complete source snapshot; untracked files are not included).
OpenETA collaboration skill used; shared RFC access remains unavailable and was
not worked around. This local experiment record makes no schema approval claim.

Run root: `tmp/luna-autonomy-retry-Mbnfvu/`; manifest validated locally.
Session: `637c0314-57be-4b9b-8db2-478dc821ec42`.
Dedicated simulator: `127.0.0.1:18766`. New simulation test data may go to the
already authorized services on `10.11.39.173` and `open.xiaojingai.com`.
The existing object bank at `10.11.18.197:8080` receives name/namespace GETs only
and returns appearance references; its client receives no scene images.

## Outcome

**Stopped for repeated wrong-object selection before motion.** This is a bounded
diagnostic interruption, not a completed autonomous performance sample. No grasp,
lift or official task success; all recorded rewards remain zero.

Six recorded main calls, all `gpt-5.6-luna`, **185548 known main tokens**. Call
durations: 27.9, 17.9, 25.0, 30.0, 30.2, 53.9 seconds. These are recorded main
usage, not a provider billing total; interrupted in-flight work may be unaccounted.
No budget exhaustion or observed provider overload. Two invalid XML responses
(`mismatched tag` and `no element found`); the first was repaired and execution
continued, while the second was recorded before diagnostic cancellation.

Completed tools: reference-only `retrieve_asset_reference`, text SAM3 with no
detections, point SAM3, and `select_sam3_detection`. No motion or gripper command
was issued. Luna again called the foreground red/green can alphabet soup and
selected its full-body mask (`detection_002`). Local inspection of
`artifacts/sam3_images/637c0314-57be-4b9b-8db2-478dc821ec42/20260908T105435253566Z-4dbd8fa4/selection.contact_sheet.png`
confirms the mismatch. At selection, recorded provider attachments include both
current views, the real contact sheet and source image, three catalog appearance
references, candidate overlay and crop. Thus omission of those images does not
explain this sample; their presence alone does not guarantee correct selection.

## Additional interface observation (not changed in this experiment)

First selection supplied `identity_anchor_id="alphabet_soup"` and
`identity_relation="same_instance"`, treating the catalog object name as an
existing live-instance anchor. The actual tool received those fields. Because
no anchor existed, `AgentMemory._capture_target_identity_anchor` took its initial
creation branch and silently ignored both fields. It created
`target:0ad7b5f389b761ecd413`, with `identity_continuity="anchor_created"`.

This did **not** install a forged ID or confirm cross-view continuity. The gap is
feedback/contract clarity for first-use identity parameters, distinct from the
visual wrong-object choice. Follow-up should make initial anchor creation vs
reuse explicit (e.g. report normalization or repair invalid reuse), without
adding a model-opinion veto or making a catalog image an instance authority.

## Cleanup and scope

Batch PID 2512552 exited 130; `episode_interrupt.cleanup` reports `ok=true`,
`close_state=closed`, no cleanup errors. Dedicated simulator PID 2512002 exited
0 after closure. Port 18766 verified free. No code/prompt changes, no commit/push,
no shared RFC writes, and no second paid repeat in this turn. The experiment
again failed to reach the contact-recovery milestone, so it cannot establish
whether simplified gates improve autonomous grasp recovery.
