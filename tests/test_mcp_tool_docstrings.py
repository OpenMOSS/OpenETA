from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _function_docstring(path: str, name: str) -> str:
    tree = ast.parse((REPO_ROOT / path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_docstring(node) or ""
    raise AssertionError(f"{name} not found in {path}")


def _flat(text: str) -> str:
    return " ".join(text.split())


def test_sam3_mcp_docstring_describes_input_semantics() -> None:
    docstring = _function_docstring("tools/sam3_mcp_server.py", "segment")
    flat_docstring = _flat(docstring)

    assert "image_base64" in docstring
    assert "<base64-encoded png bytes>" in docstring
    assert "prompt" in docstring
    assert "confidence_threshold" in docstring
    assert "not local file paths" in flat_docstring
    assert "materialize" in docstring
    assert "local temporary files" in flat_docstring


def test_anygrasp_mcp_docstring_describes_input_semantics() -> None:
    docstring = _function_docstring("tools/anygrasp_mcp_server.py", "detect_grasps")
    flat_docstring = _flat(docstring)

    assert "rgb" in docstring
    assert "depth" in docstring
    assert "intrinsics" in docstring
    assert "target_mask" in docstring
    assert '"scale": 1000.0' in docstring
    assert "<base64-encoded binary mask png>" in docstring
    assert "raw_depth / intrinsics" in docstring
    assert "For uint16 millimeter depth, use scale=1000" in docstring
    assert "valid point count" in docstring
    assert "structured input failures" in flat_docstring
    assert "not local file paths" in flat_docstring
    assert "materialize" in docstring
    assert "local temporary files" in flat_docstring


def test_anyplace_mcp_docstring_describes_input_semantics() -> None:
    docstring = _function_docstring("tools/anyplace_mcp_server.py", "predict_placement")
    flat_docstring = _flat(docstring)

    assert "rgb" in docstring
    assert "depth" in docstring
    assert "object_mask" in docstring
    assert "placement_region_mask" in docstring
    assert "intrinsics" in docstring
    assert "selected_grasp" in docstring
    assert "raw_depth / intrinsics" in docstring
    assert "not local file paths" in flat_docstring
    assert "frame=camera" in docstring
    assert "camera_frame=opencv" in docstring
    assert "exactly five" in flat_docstring
    assert "transform_matrix" in docstring
    assert "truncated at 1 metre by default" in flat_docstring
    assert "does not return point clouds" in flat_docstring


def test_contact_graspnet_mcp_docstring_describes_input_semantics() -> None:
    docstring = _function_docstring(
        "tools/contact_graspnet_mcp_server.py",
        "predict_grasps",
    )
    flat_docstring = _flat(docstring)

    assert "depth" in docstring
    assert "object_mask" in docstring
    assert "intrinsics" in docstring
    assert "RGB is deliberately absent" in flat_docstring
    assert "base64-encoded PNG bytes" in flat_docstring
    assert "scale=1000" in docstring
    assert "0.2-1.8 meters" in docstring
    assert "Panda-compatible" in docstring
    assert "does not return RGB" in flat_docstring
