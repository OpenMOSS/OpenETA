"""Separate proposal quality from shared motion authorization infrastructure."""


def geometry_intent(kind: str) -> dict:
    criteria = {
        "active_perception": ["view_diversity", "target_projection", "visibility_evidence", "identity_continuity"],
        "grasp_refinement": ["bounded_contact_residual", "grasp_orientation_preserved", "current_local_depth"],
    }
    if kind not in criteria:
        raise ValueError("unsupported geometry proposal intent")
    return {
        "schema_version": "openeta.geometry_intent.v1", "kind": kind,
        "quality_criteria": criteria[kind], "quality_authorizes_motion": False,
        "requires_exact_ik_and_motion_gates": True,
        "viewpoint_reached_proves_grasp_alignment": False,
    }
