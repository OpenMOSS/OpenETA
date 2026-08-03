from __future__ import annotations

from scripts.embodied.inspect_operator_contract import build_surface
from tools.operator_context_profiles import load_profile


EXPECTED_TOOL_SCHEMAS = {
    "check_task": "5ff129ba1b72ea350b3a7c97b190e1c0f0126bb020fc2021e3bb70ad2881f8d2",
    "finish_episode": "390c53c0dc10058cc50a38a48c6984cdde9bd448a2e37d604bc4aa728f196a58",
    "mark_point": "ff341169bbfd955540b3dc3c3b6e716fca9a53bd987341aa078792965c62d8cc",
    "move_to": "cf947f7cb87601090c14aaffc71425208e2b51b467b17be4ab9cdb0cc7d40c0b",
    "observe": "eb1642b557ec678b4a82b1da0e8b2bd7533b2b6b2657b2e5fb839403c07e93ac",
    "report_issue": "c4fdd84d2339edd3461220f6d3911e77c0363f3e3807feb7c2a4c349a3ed4685",
}


def test_release_surface_matches_evaluated_operator_contract() -> None:
    surface = build_surface()

    assert surface["profile"] == "openeta-light@1"
    assert surface["status"] == "release"
    assert (
        surface["composition_sha256"]
        == "bc1749ac21fdfa3871b87aed77e1a41571a09fd30c4625be45a93f7d9b898399"
    )
    assert (
        surface["prompt_sha256"]
        == "40ffdda3a234df2949883498b9d66e8378927a809543098cf4f3b1327ed55d38"
    )
    assert (
        surface["tool_descriptions_sha256"]
        == "c45a85c9bad2dd4c4d2aabf1ad147470929cd01e8863c478641e119ee2c69386"
    )
    assert (
        surface["resolved_invariants_sha256"]
        == "151d44d1010435f3a993d0c17d7e6cf4834f1074a5e8bb9db1c80c269d81663f"
    )
    assert {
        name: tool["input_schema_sha256"]
        for name, tool in surface["tools"].items()
    } == EXPECTED_TOOL_SCHEMAS


def test_release_profile_exposes_only_the_six_operator_tools() -> None:
    profile = load_profile()

    assert profile.public_operator_tools == [
        "observe",
        "mark_point",
        "move_to",
        "report_issue",
        "check_task",
        "finish_episode",
    ]
    assert set(profile.tool_descriptions) == set(profile.public_operator_tools)


def test_release_renderer_and_feedback_modes_are_pinned() -> None:
    invariants = load_profile().manifest["invariants"]

    assert invariants["pointcloud_metric_tick_band_style"] == "readable_v3"
    assert invariants["current_grip_site_image_feedback"] == "micro_marker_v1"
    assert invariants["solved_mark_image_feedback"] == "micro_marker_v1"
    assert (
        invariants["move_to_not_reached_contact_feedback"]
        == "current_robot_mujoco_contact_micro_marker_v1"
    )
    assert invariants["pose_preview_view_namespace"] == "candidate_v1"
    assert invariants["mark_point_image_contract"] == "all_returned_views_v1"
    assert invariants["agentview_first_surface_shortcut"] is True


def test_release_prompt_is_task_parameterized_without_changing_template() -> None:
    profile = load_profile()
    rendered = profile.render_prompt("put the bowl on the plate")

    assert "put the bowl on the plate" in rendered
    assert "{{TASK}}" not in rendered
