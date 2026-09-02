"""Exact-source contract for the minimal cuRobo Warp public-API backport."""

from __future__ import annotations

import ast
import hashlib
import os
from pathlib import Path
from typing import Any

BASE_COMMIT = "ebb71702f3f70e767f40fd8e050674af0288abe8"
UPSTREAM_REFERENCE_COMMIT = "0c1de0fc90de59d95a2627ec8ed3955d2d96c07c"
TARGET_FILE = Path("src/curobo/geom/sdf/world_mesh.py")
OLD_EXPRESSION = "wp.torch.device_from_torch(self.tensor_args.device)"
NEW_EXPRESSION = "wp.device_from_torch(self.tensor_args.device)"


def make_backported_source(source: str) -> str:
    if source.count(OLD_EXPRESSION) != 1 or NEW_EXPRESSION in source:
        raise ValueError("pinned source does not contain exactly one authorized old expression")
    return source.replace(OLD_EXPRESSION, NEW_EXPRESSION)


class _RestorePrivateApi(ast.NodeTransformer):
    def visit_Call(self, node: ast.Call) -> Any:
        self.generic_visit(node)
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "device_from_torch"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "wp"
        ):
            node.func.value = ast.Attribute(value=ast.Name(id="wp", ctx=ast.Load()), attr="torch", ctx=ast.Load())
        return node


def validate_semantic_backport(before: str, after: str) -> dict[str, Any]:
    expected = make_backported_source(before)
    if after != expected:
        raise ValueError("backport changes more than the authorized attribute chain")
    before_tree = ast.parse(before)
    after_tree = _RestorePrivateApi().visit(ast.parse(after))
    ast.fix_missing_locations(after_tree)
    if ast.dump(before_tree, include_attributes=False) != ast.dump(after_tree, include_attributes=False):
        raise ValueError("AST differs after normalizing the authorized attribute chain")
    return {
        "changed_files": [str(TARGET_FILE)],
        "changed_line_count": 1,
        "old_expression": OLD_EXPRESSION,
        "new_expression": NEW_EXPRESSION,
        "control_flow_unchanged": True,
        "call_arguments_unchanged": True,
        "imports_unchanged": True,
        "numeric_constants_unchanged": True,
        "tensor_shapes_unchanged": True,
    }


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compiler_wrapper_paths(pinned_isaac51_source: Path) -> dict[str, Path]:
    root = pinned_isaac51_source.resolve() / "scripts/toolchains"
    gcc = root / "gcc12-system-ld"
    gxx = root / "gxx12-system-ld"
    return {"CC": gcc, "CXX": gxx, "CUDAHOSTCXX": gxx}


def validate_compiler_wrappers(pinned_isaac51_source: Path) -> dict[str, dict[str, Any]]:
    source = pinned_isaac51_source.resolve()
    records = {}
    for variable, path in compiler_wrapper_paths(source).items():
        resolved = path.resolve()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError(f"{variable} compiler wrapper is not an executable regular file: {path}")
        if not resolved.is_relative_to(source):
            raise ValueError(f"{variable} compiler wrapper escapes pinned Isaac51 source")
        records[variable] = {
            "path": str(path),
            "realpath": str(resolved),
            "sha256": file_sha256(path),
        }
    return records


def corrective_retry_accounting() -> dict[str, Any]:
    return {
        "c0_invocations": [
            {
                "index": 0,
                "valid_build_attempt": False,
                "classification": "invalid_orchestration_invocation",
                "reason": "compiler_wrapper_path_not_found",
                "entered_meaningful_compilation": False,
                "generated_derived_extensions": False,
            },
            {
                "index": 1,
                "valid_build_attempt": True,
                "authorized_by_user": True,
                "correction_scope": ["CC", "CXX", "CUDAHOSTCXX"],
                "runtime_scope_expanded": False,
            },
        ],
        "total_c0_invocations": 2,
        "invalid_orchestration_invocations": 1,
        "valid_curobo_build_attempts": 1,
        "authorized_corrective_retries": 1,
        "maximum_valid_curobo_build_attempts": 1,
        "maximum_corrective_retries": 1,
        "additional_retry_allowed": False,
    }
