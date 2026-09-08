"""Shared vocabulary for gross geometry hints, not object/part identity.

SAM3 selection accepts only this vocabulary (or an omitted/empty hint).
The grasp compiler deliberately also accepts extension strings; an extension
does not establish a validated strategy, graspable part, or execution authority.
"""

GRASP_GEOMETRY_FAMILIES = frozenset({
    "upright_can",
    "upright_bottle",
    "lying_bottle",
    "boxed_item",
    "bowl",
    "apple",
    "articulated_handle",
    "drawer_handle",
    "other",
    "unknown",
})

GRASP_GEOMETRY_FAMILY_HINT = (
    "optional truthful gross-geometry hint: "
    + ", ".join(sorted(GRASP_GEOMETRY_FAMILIES))
    + "; omit or use an empty string when unspecified. Use canonical lowercase "
    "values; this hint is not object identity or graspable-part evidence."
)


def normalize_selection_geometry_family(value: str) -> str:
    """Validate a selection hint, preserving legacy whitespace/case tolerance."""
    if not isinstance(value, str):
        raise ValueError("target_geometry_family must be a string.")
    family = value.strip().lower()
    if family and family not in GRASP_GEOMETRY_FAMILIES:
        raise ValueError(
            "target_geometry_family must be one of "
            + ", ".join(sorted(GRASP_GEOMETRY_FAMILIES))
            + ". Omit the hint when uncertain; extension strings are accepted "
            "by compile_grasp_seed, not by select_sam3_detection."
        )
    return family
