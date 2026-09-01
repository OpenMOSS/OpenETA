"""Host-internal grasp backend metadata.

These specs support backend handler tests and diagnostics.  They are not
registered by ``build_default_tool_registry`` and must never be projected into
the Agent's ``available_tools``.  The only public grasp-estimation interface is
``grasp_pose_estimate``.
"""

from __future__ import annotations

from agent.tools.registry import ToolEffect, ToolSpec


def build_internal_grasp_backend_specs() -> dict[str, ToolSpec]:
    """Return implementation metadata for the two supported facade backends."""

    return {
        "anygrasp": ToolSpec(
            name="anygrasp",
            category="internal_grasp_backend",
            description="Host-internal AnyGrasp backend for grasp_pose_estimate.",
            parameters={
                "mode": "targeted or scene",
                "rgb": "host-resolved local RGB image path",
                "depth": "host-resolved local depth image path",
                "intrinsics": "host-resolved aligned camera intrinsics",
                "target_mask": "host-resolved target mask for targeted mode",
                "approach_steering": "optional camera-frame approach direction",
                "approach_thresh": "optional approach threshold in radians",
                "collision_detection": "optional backend collision toggle",
                "dense_grasp": "optional dense sampling toggle",
                "depth_cutoff_factor": "optional fixed-cutoff compatibility factor",
            },
            effect=ToolEffect.PLANNING,
        ),
        "graspgenx": ToolSpec(
            name="graspgenx",
            category="internal_grasp_backend",
            description="Host-internal GraspGenX backend for grasp_pose_estimate.",
            parameters={
                "rgb": "host-resolved RGB path used for provenance and overlays",
                "depth": "host-resolved aligned depth path",
                "object_mask": "host-resolved SAM3 mask artifact",
                "intrinsics": "host-resolved aligned camera intrinsics",
                "gripper_name": "host-configured GraspGenX gripper name",
                "up_direction_camera": "host-configured camera-frame up direction",
                "depth_cutoff_factor": "optional service depth-cutoff factor",
            },
            effect=ToolEffect.PLANNING,
        ),
        "list_graspgenx_grippers": ToolSpec(
            name="list_graspgenx_grippers",
            category="internal_grasp_backend",
            description="Host-only GraspGenX capability discovery call.",
            parameters={},
            safe_by_default=True,
            effect=ToolEffect.READ_ONLY,
        ),
    }


def build_retired_contact_graspnet_spec() -> ToolSpec:
    """Metadata for isolated legacy handler tests, never runtime registration."""

    return ToolSpec(
        name="contact_graspnet",
        category="retired_grasp_backend",
        description=(
            "Retired Panda-specific Contact-GraspNet adapter retained only for "
            "legacy service compatibility tests."
        ),
        parameters={
            "rgb": "host-resolved RGB path",
            "depth": "host-resolved aligned depth path",
            "object_mask": "host-resolved SAM3 mask artifact",
            "intrinsics": "host-resolved aligned camera intrinsics",
        },
        effect=ToolEffect.PLANNING,
    )

