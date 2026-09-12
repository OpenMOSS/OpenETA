"""Host-selected experimental interfaces, separate from reviewed default contracts.

The projection and admission check share the same schema. This is not a runtime
fallback: native references exist only behind bundle resolution for migrated tools.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from functools import lru_cache

from agent.tools.contracts import build_default_tool_contract_catalog, check_tool_request_conformance
from agent.runtime.tool_bundles import BUNDLE_ID_PATTERN

DEFAULT_INTERFACE_PROFILE = "legacy_compatible"
BUNDLE_STAGE1 = "bundle_stage1"
BUNDLE_STAGE2 = "bundle_stage2"
BUNDLE_STAGE3 = "bundle_stage3"
MIGRATED_TOOLS = frozenset({"compile_grasp_seed", "move_to"})
PROPOSAL_TOOLS = frozenset({"propose_motion_target", "compose_ik_trajectory"})


def migrated_tools(profile):
    validate_interface_profile(profile)
    if profile == DEFAULT_INTERFACE_PROFILE:
        return frozenset()
    result = MIGRATED_TOOLS
    if profile in {BUNDLE_STAGE2, BUNDLE_STAGE3}:
        result = result | {"ik_preview_check"}
    if profile == BUNDLE_STAGE3:
        result = result | {"follow_eef_trajectory", "select_sam3_detection",
                           "reject_sam3_detections", "grasp_pose_estimate", "anyplace"}
    return result


def validate_interface_profile(profile: str) -> str:
    if profile not in {DEFAULT_INTERFACE_PROFILE, BUNDLE_STAGE1, BUNDLE_STAGE2, BUNDLE_STAGE3}:
        raise ValueError(f"Unknown host agent_interface_profile: {profile!r}")
    return profile


@lru_cache(maxsize=16)
def _stage_contract(name):
    from agent.tools.registry import build_default_tool_registry
    if name in PROPOSAL_TOOLS:
        from agent.tools.bundle_proposals import proposal_contract
        return proposal_contract(name)
    contract = build_default_tool_contract_catalog(build_default_tool_registry().list()).get(name)
    schema = deepcopy(contract.request_schema)
    old_fields = {
        "compile_grasp_seed": {"grasp_result_id"}, "move_to": {"ik_receipt_id"},
        "ik_preview_check": {"target_pose", "compiled_grasp_id", "waypoint_role", "path_fraction",
                             "viewpoint_proposal_id", "candidate_id", "probe_id", "waypoint_index"},
        "follow_eef_trajectory": {"ik_receipt_ids"},
        "select_sam3_detection": {"sam3_result_id"},
        "reject_sam3_detections": {"sam3_result_id"},
    }.get(name, set())
    for key in old_fields:
        schema["properties"].pop(key, None)
    schema["properties"]["bundle_id"] = {"type": "string", "minLength": 1,
        "pattern": f"^{BUNDLE_ID_PATTERN}$",
        "description": "Exact Host-registered typed handoff ID from tool_handoffs."}
    schema.pop("oneOf", None)
    schema["required"] = list(dict.fromkeys([*(k for k in schema.get("required", []) if k not in old_fields), "bundle_id"]))
    schema["additionalProperties"] = False
    return replace(contract, request_schema=schema)


def profile_parameter_errors(profile, name, parameters):
    validate_interface_profile(profile)
    if name not in migrated_tools(profile) and not (
        name == "propose_motion_target" and profile in {BUNDLE_STAGE2, BUNDLE_STAGE3}
        or name == "compose_ik_trajectory" and profile == BUNDLE_STAGE3
    ):
        return []
    violations = check_tool_request_conformance(_stage_contract(name), parameters)
    if not violations:
        return []
    schema = _stage_contract(name).request_schema
    allowed = ", ".join(schema["properties"])
    required = ", ".join(schema.get("required", []))
    field_guidance = (f"Required fields: {required}. " if required else
                      "Follow the required fields of the selected schema branch. ")
    field_guidance += (
        "Optional fields may be omitted; do not fill them with null unless their schema accepts null. "
        "Preserve valid supplied fields while repairing the reported errors."
    )
    hint = ("This authoring tool requires the declared geometry/reference fields; it does not execute motion."
            if name in PROPOSAL_TOOLS else
            "Use the exact bundle_id in tool_handoffs; no native-reference fallback.")
    return [f"{name} [{profile}]: {v.message} Allowed fields: {allowed}. {field_guidance} {hint}"
            for v in violations]


def project_tool_reference(reference, profile):
    validate_interface_profile(profile)
    if reference.get("name") not in migrated_tools(profile) | PROPOSAL_TOOLS:
        return reference
    result = deepcopy(reference)
    name = result["name"]
    result["parameters"] = deepcopy(_stage_contract(name).request_schema)
    descriptions = {
        "compile_grasp_seed": (
        "Compile a selected grasp: pass a grasp_candidates bundle_id and candidate_id. "
        "Optional geometry/strategy fields are listed in this schema. Returns world-frame "
        "geometry and target_pose handoff bundles for IK preview; does not move the robot."
        ), "move_to": (
        "Execute an ik_result bundle_id returned after IK preview. Host rechecks the exact "
        "pose, orientation, freshness, controller and collision evidence; a failed preview "
        "bundle is inspectable but never authorizes motion. Optional execution controls "
        "are listed in this schema."
        ),
        "ik_preview_check": "Preview one target_pose bundle_id. Inspect residuals/search history, not just tool success. "
            "If creating a new pose, use propose_motion_target first. Failed search is not proof of infeasibility. "
            "Returns ik_result bundle for checked execution; no movement occurs here.",
        "follow_eef_trajectory": "Execute one ik_trajectory bundle_id made by compose_ik_trajectory from ordered IK result bundles. "
            "Every original receipt and the whole trajectory must pass existing live gates.",
        "select_sam3_detection": "Choose detection_id from a sam3_detections bundle_id after visual comparison. "
            "Identity and evidence-role fields remain explicit Agent choices. Selection is not grasp or motion authority.",
        "reject_sam3_detections": "Reject a sam3_detections bundle_id with a reason; obtain different perception evidence as needed.",
        "grasp_pose_estimate": "Estimate grasps from a grasp_input manifest bundle_id in host_resolved_inputs or tool_handoffs. "
            "If none is ready, segment and confirm the intended target. Do not supply raw RGB-D or invent a bundle.",
        "anyplace": "Estimate placement candidates from a placement_input manifest bundle_id in host_resolved_inputs or tool_handoffs. "
            "Host revalidates both object and destination evidence; ranking is not motion authority.",
    }
    result["description"] = descriptions.get(name, _stage_contract(name).description)
    if name in PROPOSAL_TOOLS:
        result["returns"] = {"outcomes": ["proposal_registered"], "fields": ["handoff_bundles", "motion_authorized"]}
    return result


def project_profile_evidence(value, profile, handoffs):
    """Project callable guidance only; recorded native evidence is not rewritten.

    This is display-time guidance, never an executable alias/fallback. Missing
    or stale references remain non-callable and the Agent must choose new evidence.
    """
    if profile not in {BUNDLE_STAGE2, BUNDLE_STAGE3}:
        return value
    refs = [r for r in handoffs if r.get("current_epoch")]

    def matching(kind, key, value):
        return next((r["bundle_id"] for r in reversed(refs) if r.get("kind") == kind
                     and r.get("summary", {}).get(key) == value), None)

    def call_hint(call):
        if not isinstance(call, dict) or not isinstance(call.get("parameters"), dict):
            return call
        tool, parameters = call.get("tool"), call["parameters"]
        if tool not in migrated_tools(profile) or str(parameters.get("bundle_id", "")).startswith("bnd-"):
            return call
        result = deepcopy(call)
        if tool == "ik_preview_check":
            controls = {"position_tolerance_m", "orientation_tolerance_rad", "preserve_current_orientation", "check_endpoint_collision"}
            reference = {k: v for k, v in parameters.items() if k not in controls}
            found = next((r for r in reversed(refs) if r.get("kind") == "target_pose"
                          and {k: v for k, v in (r.get("summary", {}).get("request_reference") or {}).items()
                               if k not in controls} == reference), None)
            if found:
                result["parameters"] = {"bundle_id": found["bundle_id"],
                                        **{k: v for k, v in parameters.items() if k in controls}}
                return result
            result["tool"] = "propose_motion_target"
            result["instruction"] = "This authors a new target bundle; it does not run IK or authorize motion."
            return result
        kind, key, old = {
            "move_to": ("ik_result", "receipt_id", "ik_receipt_id"),
            "select_sam3_detection": ("sam3_detections", "result_id", "sam3_result_id"),
            "reject_sam3_detections": ("sam3_detections", "result_id", "sam3_result_id"),
            "grasp_pose_estimate": ("grasp_input", "native_reference", "bundle_id"),
            "anyplace": ("placement_input", "native_reference", "bundle_id"),
        }.get(tool, (None, None, None))
        if tool == "follow_eef_trajectory" and isinstance(parameters.get("ik_receipt_ids"), list):
            bundles = [matching("ik_result", "receipt_id", rid) for rid in parameters["ik_receipt_ids"]]
            if bundles and all(bundles):
                return {"tool": "compose_ik_trajectory", "parameters": {"bundle_ids": bundles},
                        "instruction": "Group these exact receipts; the returned bundle still needs trajectory gates."}
        bundle = matching(kind, key, parameters.get(old)) if kind else None
        if bundle:
            result["parameters"] = {k: v for k, v in parameters.items() if k != old}
            result["parameters"]["bundle_id"] = bundle
        else:
            result.pop("parameters", None)
            result["callable"] = False
            result["instruction"] = "Inspect current tool_handoffs; this native hint has no current manifest reference."
        return result

    def visit(node):
        if isinstance(node, list):
            return [visit(x) for x in node]
        if not isinstance(node, dict):
            return node
        result = {}
        for key, child in node.items():
            if key in {"allowed_next_calls", "ik_preview_requests"} and isinstance(child, list):
                result[key] = [call_hint(x) for x in child]
            elif key in {"execution_reference", "execution_handoff", "recommended_call", "next_call"}:
                result[key] = call_hint(child)
            else:
                result[key] = visit(child)
        return result
    result = visit(value)
    # Native input bundles already have a public ready/not-ready envelope.
    # Keep status/recovery evidence but present the manifest invocation form.
    def inputs(node):
        if isinstance(node, list):
            return [inputs(x) for x in node]
        if not isinstance(node, dict):
            return node
        out = {k: inputs(v) for k, v in node.items()}
        old = out.get("bundle_id")
        if (profile == BUNDLE_STAGE3 and isinstance(old, str) and old.startswith(("grasp:", "anyplace:"))
                and ("status" in out or "call_parameters" in out)):
            kind = "grasp_input" if old.startswith("grasp:") else "placement_input"
            bundle = matching(kind, "native_reference", old)
            if bundle:
                out["bundle_id"] = bundle
                if isinstance(out.get("call_parameters"), dict):
                    out["call_parameters"]["bundle_id"] = bundle
            elif "status" in out or "call_parameters" in out:
                out.update(bundle_id=None, call_parameters=None, status="manifest_not_current")
        return out
    return inputs(result)


def profile_guidance(profile):
    validate_interface_profile(profile)
    if profile == DEFAULT_INTERFACE_PROFILE:
        return {}
    return {
        "name": profile,
        "object_identity_guidance": (
            "For unfamiliar names or similar packages, first retrieve_asset_reference(localize=false) "
            "for the task object when the bank is available. Then use a full-agentview "
            "SAM3 text call with prompt='object' to propose scene instances, then compare original-color "
            "candidate crops against the task and catalog references. inspect_evidence automatically "
            "shows the latest retrieved references beside up to eight original-color candidate crops; "
            "check its reference_object label and compare every plausible candidate before selecting. Use it "
            "to page the detection bundle or open a saved image. Backend proposals may miss objects; "
            "do not assume a complete inventory or equate point-prompt masks with different instances. "
            "This is a recommended discovery strategy, not a required tool order. "
            "Establish name-to-appearance evidence before committing a named object as the task target. "
            "For catalog objects, especially branded packages or several similar containers, prefer "
            "retrieve_asset_reference early to compare the existing object memory bank's reference images "
            "against the scene, unless prior verified appearance evidence already resolves the identity. "
            "Prefer localize=false for a direct reference-image lookup, then make your own visual "
            "comparison; the optional isolated localizer can also misidentify an object. "
            "A plausible location, a can-shaped mask, or successful point segmentation does not establish "
            "which named object it is. A failed text prompt is a reason to reconsider semantic identity, "
            "not just to point at the same guessed object. Use an object-only "
            "target_object phrase and the current environment/source packet; compare references with the "
            "actual scene before choosing a mask. Retrieval/localizer output is not proof of scene identity. "
            "You choose whether to query; no task step or tool order is mandated."
        ),
        "bundle_only_tools": sorted(migrated_tools(profile)),
        "scope": ("Partial migration: IK, placement, probes and trajectories retain their listed contracts."
                  if profile == BUNDLE_STAGE1 else
                  "All IK reference branches have bundle coverage. SAM3 prompting, identity decisions, camera transforms "
                  "and geometric authoring retain explicit fields; proposals never authorize execution."),
        "rules": (
            "For compile_grasp_seed use a grasp_candidates bundle_id plus candidate_id; "
            "for move_to use an ik_result bundle_id. Inspect tool_handoffs or use "
            "artifacts.read_bundle in python_exec. Native fields in recorded evidence are "
            "data, not alternative invocation forms. No automatic fallback or Agent-selected "
            "profile switch exists. A bundle never bypasses live safety gates."
            + (" IK consumes target_pose bundles from compile, wrist, probe, transform or propose_motion_target. "
               "Use propose_motion_target for a materially changed pose/reference, not to replay failed geometry."
               if profile in {BUNDLE_STAGE2, BUNDLE_STAGE3} else "")
            + (" compose_ik_trajectory groups 1-5 ordered IK result bundles for follow_eef_trajectory. "
               "Selection/rejection uses sam3_detections bundles; grasp/placement inputs use registered manifests."
               if profile == BUNDLE_STAGE3 else "")
        ),
    }
