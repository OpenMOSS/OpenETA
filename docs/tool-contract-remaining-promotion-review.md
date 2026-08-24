# Remaining ToolContract verified-promotion review

Status: three-person review approved; post-review authority canary passed; all 34 promotions complete.

This packet records the approved per-tool maturity decisions for the remaining 34 public tools and their passing request-validation-only authority canary. Gate-repair envelope authority and executable safety gates remain separate controls.

## Evidence summary

- Campaign targets: 34
- Deterministic request parity ready: 34
- Production fixture receipts: 34
- Planner/ToolRegistry integration canaries: 34
- Host resolver binding audit conformant: true
- Freshness/invalidation: 19/19 scoped tools
- Remote-backend semantics: 11/11 scoped tools
- World-mutating execution receipts: 5/5 scoped tools

The catalog integration canary recorded 70 valid/invalid Planner shadows with zero acceptance mismatch or unexpected enforcement, plus 35 conformant ToolRegistry results. Production behavior is established separately by registry-bound fixtures; the canary's deterministic adapter is not represented as a live remote backend or task success.

## Approved decision

All rows below were approved from `declared` to `verified`. The separately scoped request-validation canary passed without tool execution or world mutation. Gate-repair-envelope authority remains empty and executable gate authority remains `legacy_runtime`.

| Tool | Risk tier | Successful outcomes | Bound gates | Remaining pre-promotion gaps |
|---|---|---:|---:|---|
| observe | evidence_sensitive | 1 | 1 | none |
| enhance_depth | evidence_sensitive | 2 | 3 | none |
| create_simulator_env | world_mutating | 1 | 2 | none |
| close_simulator_env | world_mutating | 1 | 2 | none |
| python_exec | local_deterministic | 1 | 2 | none |
| web_search | backend_or_artifact | 1 | 2 | none |
| web_fetch | backend_or_artifact | 1 | 2 | none |
| sam3 | evidence_sensitive | 2 | 3 | none |
| retrieve_asset_reference | evidence_sensitive | 1 | 3 | none |
| molmopoint | evidence_sensitive | 1 | 3 | none |
| select_sam3_detection | evidence_sensitive | 1 | 2 | none |
| reject_sam3_detections | evidence_sensitive | 1 | 2 | none |
| grasp_pose_estimate | evidence_sensitive | 1 | 3 | none |
| anyplace | evidence_sensitive | 1 | 3 | none |
| camera_pose_to_world | evidence_sensitive | 1 | 3 | none |
| propose_calibration_profile | local_deterministic | 1 | 2 | none |
| promote_calibration_profile | local_deterministic | 1 | 2 | none |
| propose_grasp_strategy | local_deterministic | 1 | 2 | none |
| promote_grasp_strategy | local_deterministic | 1 | 2 | none |
| compile_grasp_seed | evidence_sensitive | 1 | 3 | none |
| compute_wrist_alignment | evidence_sensitive | 2 | 3 | none |
| propose_wrist_viewpoints | evidence_sensitive | 1 | 3 | none |
| prepare_attachment_probe | evidence_sensitive | 1 | 2 | none |
| assess_attachment_probe | evidence_sensitive | 1 | 2 | none |
| move_to | world_mutating | 5 | 7 | none |
| follow_eef_trajectory | world_mutating | 5 | 7 | none |
| gripper_control | world_mutating | 4 | 3 | none |
| ik_preview_check | evidence_sensitive | 5 | 4 | none |
| save_memory | local_deterministic | 1 | 2 | none |
| get_memory | local_deterministic | 1 | 2 | none |
| delete_memory | local_deterministic | 1 | 2 | none |
| compact_memory | local_deterministic | 1 | 2 | none |
| register_skill | local_deterministic | 1 | 2 | none |
| update_skill | local_deterministic | 1 | 2 | none |

## Evidence files

- `docs/generated/tool-contract-promotion-campaign.json`
- `docs/generated/tool-contract-integration-canary.json`
- `docs/generated/*-fixture-receipt.json`
- `docs/generated/tool-contract-readiness.json`
- `docs/generated/host-resolver-binding-audit.json`
