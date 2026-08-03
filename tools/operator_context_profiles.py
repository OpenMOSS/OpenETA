"""Versioned Operator context profiles and immutable episode snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import re
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Mapping

from jsonschema import Draft202012Validator


DEFAULT_PROFILE_ID = "openeta-light"
_PROFILE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_COMPONENT_ORDER_FIELDS = (
    "contract_order",
    "result_order",
    "visual_order",
    "renderer_order",
    "prompt_order",
    "tool_description_order",
)
_INTEGRITY_SCHEMA = "openeta.operator_context_integrity.v1"
_COMPOSITION_ALGORITHM = "sha256-ordered-components-v1"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def profile_root() -> Path:
    configured = os.environ.get("OPENETA_OPERATOR_CONTEXT_PROFILE_ROOT")
    return Path(configured).resolve() if configured else (
        _repo_root() / "configs/embodied/operator-context"
    )


def _component_spec(value: Any, *, source: str) -> tuple[str, str | None]:
    if isinstance(value, str) and value:
        return value, None
    if isinstance(value, dict):
        ref = value.get("ref")
        expected_sha256 = value.get("sha256")
        if (
            isinstance(ref, str)
            and ref
            and isinstance(expected_sha256, str)
            and _SHA256_RE.fullmatch(expected_sha256)
        ):
            return ref, expected_sha256
    raise ValueError(
        f"{source} must be a component ref string or an object containing "
        "non-empty ref and lowercase SHA-256"
    )


def _ordered_component_slots(manifest: Mapping[str, Any]) -> list[str]:
    """Return the manifest's canonical component order and validate coverage.

    Older v2 profiles only use contract/prompt/tool orders.  The optional
    result/visual/renderer orders let newer profiles version dynamic result
    semantics and preview rendering independently without changing the public
    tool set.
    """

    components = manifest.get("components")
    if not isinstance(components, Mapping):
        raise ValueError("Operator context manifest must declare components")
    ordered: list[str] = []
    for field in _COMPONENT_ORDER_FIELDS:
        values = manifest.get(field, [])
        if not isinstance(values, list) or not all(
            isinstance(value, str) and value for value in values
        ):
            raise ValueError(f"{field} must be a list of non-empty slot names")
        ordered.extend(values)
    duplicates = sorted(
        slot for slot in set(ordered) if ordered.count(slot) > 1
    )
    if duplicates:
        raise ValueError(
            "Operator context component slots appear more than once: "
            + ", ".join(duplicates)
        )
    missing = sorted(set(components) - set(ordered))
    unknown = sorted(set(ordered) - set(components))
    if missing or unknown:
        parts = []
        if missing:
            parts.append("unlisted=" + ",".join(missing))
        if unknown:
            parts.append("unknown=" + ",".join(unknown))
        raise ValueError(
            "Operator context component order does not cover components: "
            + "; ".join(parts)
        )
    return ordered


def _component_descriptor(
    manifest: Mapping[str, Any],
    components: Mapping[str, Mapping[str, str]],
) -> list[dict[str, str]]:
    """Build the ordered, content-addressed recipe descriptor."""

    descriptor: list[dict[str, str]] = []
    for slot in _ordered_component_slots(manifest):
        spec = components.get(slot)
        if not isinstance(spec, Mapping):
            raise ValueError(f"missing resolved component spec for {slot!r}")
        ref, sha256 = _component_spec(spec, source=f"component {slot!r}")
        if sha256 is None:
            raise ValueError(
                f"resolved component {slot!r} must include a pinned sha256"
            )
        descriptor.append({"slot": slot, "ref": ref, "sha256": sha256})
    return descriptor


def build_manifest_integrity(
    manifest: Mapping[str, Any],
    *,
    resolved_components: Mapping[str, Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    """Generate the non-recursive integrity block for an authored manifest.

    ``manifest_sha256`` covers the manifest with the integrity block removed.
    This makes the generated value stable and avoids a self-referential hash.
    ``composition_sha256`` deliberately hashes an ordered list; changing layer
    order is a real context change even when the same component files are used.
    """

    components = (
        resolved_components
        if resolved_components is not None
        else manifest.get("components")
    )
    if not isinstance(components, Mapping):
        raise ValueError("manifest components must be an object")
    descriptor = _component_descriptor(manifest, components)
    composition_sha256 = canonical_sha256(
        {
            "algorithm": _COMPOSITION_ALGORITHM,
            "components": descriptor,
        }
    )
    unsigned = copy.deepcopy(dict(manifest))
    unsigned.pop("integrity", None)
    return {
        "schema": _INTEGRITY_SCHEMA,
        "composition_algorithm": _COMPOSITION_ALGORITHM,
        "component_order": [item["slot"] for item in descriptor],
        "composition_sha256": composition_sha256,
        "manifest_sha256": canonical_sha256(unsigned),
    }


def validate_manifest_integrity(manifest: Mapping[str, Any]) -> None:
    """Fail closed when an authored profile manifest was edited in place."""

    integrity = manifest.get("integrity")
    if integrity is None:
        return
    if not isinstance(integrity, Mapping):
        raise RuntimeError("Operator context manifest integrity must be an object")
    if integrity.get("schema") != _INTEGRITY_SCHEMA:
        raise RuntimeError(
            "unsupported Operator context manifest integrity schema: "
            f"{integrity.get('schema')!r}"
        )
    try:
        expected = build_manifest_integrity(manifest)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "Operator context manifest integrity mismatch: malformed "
            "component order or component specs"
        ) from exc
    for key in (
        "composition_algorithm",
        "component_order",
        "composition_sha256",
        "manifest_sha256",
    ):
        if integrity.get(key) != expected[key]:
            raise RuntimeError(
                "Operator context manifest integrity mismatch for "
                f"{key!r}: expected={expected[key]!r}, "
                f"actual={integrity.get(key)!r}"
            )


def _merge_contract_invariants(
    target: dict[str, Any],
    additions: Mapping[str, Any],
) -> None:
    """Merge layered contract invariants without replacing sibling schemas.

    Context profiles are intentionally composable.  In particular, a small
    contract layer may refine one public result schema while inheriting the
    other tool schemas from its base layer.  A shallow ``dict.update`` made
    that impossible and encouraged copying large, stale contract files.
    """

    for key, value in additions.items():
        if (
            isinstance(target.get(key), dict)
            and isinstance(value, Mapping)
        ):
            _merge_contract_invariants(target[key], value)
        else:
            target[key] = value


@dataclass(frozen=True)
class OperatorContextProfile:
    profile_id: str
    revision: int
    status: str
    manifest: dict[str, Any]
    prompt_template: str
    tool_descriptions: dict[str, str]
    directory: Path
    components: dict[str, dict[str, str]]
    composition_sha256: str
    manifest_sha256: str

    @property
    def label(self) -> str:
        return f"{self.profile_id}@{self.revision}"

    def render_prompt(self, task: str) -> str:
        return self.prompt_template.replace("{{TASK}}", task).strip()

    @property
    def public_operator_tools(self) -> list[str]:
        declared = self.manifest.get("public_operator_tools")
        if declared is None:
            declared = self.manifest.get("public_manipulation_tools", [])
        if not isinstance(declared, list) or not all(
            isinstance(name, str) and name for name in declared
        ):
            raise RuntimeError(
                f"Operator context profile {self.label!r} has invalid "
                "public_operator_tools"
            )
        return list(declared)


def load_profile(profile_id: str | None = None) -> OperatorContextProfile:
    selected = profile_id or os.environ.get(
        "OPENETA_OPERATOR_CONTEXT_PROFILE", DEFAULT_PROFILE_ID
    )
    if not _PROFILE_ID_RE.fullmatch(selected):
        raise ValueError(f"invalid Operator context profile id: {selected!r}")
    directory = profile_root() / selected
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Operator context profile does not exist: {manifest_path}"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    authored_manifest = copy.deepcopy(manifest)
    if manifest.get("profile_id") != selected:
        raise ValueError(
            f"profile_id mismatch in {manifest_path}: {manifest.get('profile_id')!r}"
        )
    revision = manifest.get("revision")
    if not isinstance(revision, int) or revision < 1:
        raise ValueError(f"invalid revision in {manifest_path}")
    validate_manifest_integrity(manifest)
    components: dict[str, dict[str, str]] = {}
    if manifest.get("schema_version") == "openeta.operator_context_profile.v2":
        overrides_raw = os.environ.get(
            "OPENETA_OPERATOR_CONTEXT_COMPONENT_OVERRIDES", ""
        ).strip()
        overrides = json.loads(overrides_raw) if overrides_raw else {}
        if not isinstance(overrides, dict) or not all(
            isinstance(key, str) for key in overrides
        ):
            raise ValueError(
                "OPENETA_OPERATOR_CONTEXT_COMPONENT_OVERRIDES must be a "
                "JSON object mapping component slots to pinned component specs"
            )
        declared = manifest.get("components")
        if not isinstance(declared, dict):
            raise ValueError(f"invalid components in {manifest_path}")
        unknown_overrides = sorted(set(overrides) - set(declared))
        if unknown_overrides:
            raise ValueError(
                f"unknown Operator context component override slots: "
                f"{unknown_overrides}"
            )
        resolved_values: dict[str, Any] = {}
        component_root = profile_root() / "components"
        for slot, default_spec in declared.items():
            selected_spec = overrides.get(slot, default_spec)
            ref, expected_sha256 = _component_spec(
                selected_spec,
                source=f"component {slot!r}",
            )
            component_path = (component_root / ref).resolve()
            try:
                component_path.relative_to(component_root.resolve())
            except ValueError as exc:
                raise ValueError(
                    f"component {slot!r} escapes component root: {ref!r}"
                ) from exc
            if not component_path.is_file():
                raise FileNotFoundError(
                    f"Operator context component does not exist: {component_path}"
                )
            raw = component_path.read_bytes()
            actual_sha256 = hashlib.sha256(raw).hexdigest()
            if (
                expected_sha256 is not None
                and expected_sha256 != actual_sha256
            ):
                raise RuntimeError(
                    f"Operator context component digest mismatch for {slot!r}: "
                    f"ref={ref!r}, expected={expected_sha256}, "
                    f"actual={actual_sha256}"
                )
            components[slot] = {
                "ref": ref,
                "sha256": actual_sha256,
            }
            resolved_values[slot] = (
                json.loads(raw.decode("utf-8"))
                if component_path.suffix == ".json"
                else raw.decode("utf-8")
            )
        contract: dict[str, Any] = {
            "public_manipulation_tools": [],
            "public_operator_tools": None,
            "invariants": {},
        }
        contract_order = [
            *manifest.get("contract_order", ["contract"]),
            *manifest.get("result_order", []),
            *manifest.get("visual_order", []),
            *manifest.get("renderer_order", []),
        ]
        for slot in contract_order:
            layer = resolved_values.get(str(slot))
            if not isinstance(layer, dict):
                raise ValueError(
                    f"contract component slot {slot!r} must resolve to JSON"
                )
            if "public_manipulation_tools" in layer:
                tools = layer["public_manipulation_tools"]
                if not isinstance(tools, list):
                    raise ValueError(
                        f"contract component {slot!r} has invalid public tools"
                    )
                contract["public_manipulation_tools"] = list(tools)
            if "public_operator_tools" in layer:
                tools = layer["public_operator_tools"]
                if not isinstance(tools, list):
                    raise ValueError(
                        f"contract component {slot!r} has invalid public "
                        "Operator tools"
                    )
                contract["public_operator_tools"] = list(tools)
            invariants = layer.get("invariants", {})
            if not isinstance(invariants, dict):
                raise ValueError(
                    f"contract component {slot!r} has invalid invariants"
                )
            _merge_contract_invariants(contract["invariants"], invariants)
        manifest = {
            **manifest,
            "public_manipulation_tools": contract.get(
                "public_manipulation_tools", []
            ),
            **(
                {
                    "public_operator_tools": contract[
                        "public_operator_tools"
                    ]
                }
                if contract.get("public_operator_tools") is not None
                else {}
            ),
            "invariants": contract.get("invariants", {}),
        }
        prompt_order = manifest.get("prompt_order", [])
        tool_order = manifest.get("tool_description_order", [])
        prompt_parts = []
        for slot in prompt_order:
            value = resolved_values.get(str(slot))
            if not isinstance(value, str):
                raise ValueError(
                    f"prompt component slot {slot!r} must resolve to text"
                )
            prompt_parts.append(value.strip())
        prompt_template = "\n\n".join(part for part in prompt_parts if part)
        descriptions: dict[str, str] = {}
        for slot in tool_order:
            layer = resolved_values.get(str(slot))
            if not isinstance(layer, dict):
                raise ValueError(
                    f"tool-description component slot {slot!r} must resolve to JSON"
                )
            replace = layer.get("replace", {})
            append = layer.get("append", {})
            if not isinstance(replace, dict) or not isinstance(append, dict):
                raise ValueError(
                    f"tool-description layer {slot!r} has invalid operations"
                )
            for name, value in replace.items():
                descriptions[str(name)] = str(value).strip()
            for name, value in append.items():
                name = str(name)
                if name not in descriptions:
                    raise ValueError(
                        f"tool-description layer {slot!r} appends unknown tool {name!r}"
                    )
                descriptions[name] = (
                    descriptions[name].rstrip() + " " + str(value).strip()
                )
    else:
        prompt_path = (directory / str(manifest["startup_prompt"])).resolve()
        descriptions_path = (
            directory / str(manifest["tool_descriptions"])
        ).resolve()
        prompt_template = prompt_path.read_text(encoding="utf-8")
        descriptions = json.loads(descriptions_path.read_text(encoding="utf-8"))
        for slot, path in (
            ("prompt.monolith", prompt_path),
            ("tools.monolith", descriptions_path),
        ):
            components[slot] = {
                "ref": str(path.relative_to(profile_root())),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
    expected = set(
        manifest.get(
            "public_operator_tools",
            manifest.get("public_manipulation_tools", []),
        )
    )
    description_source = (
        f"composed profile {manifest_path}"
        if manifest.get("schema_version") == "openeta.operator_context_profile.v2"
        else str(descriptions_path)
    )
    if set(descriptions) != expected:
        raise ValueError(
            f"{description_source} tools {sorted(descriptions)} do not match "
            f"manifest tools {sorted(expected)}"
        )
    if not all(isinstance(value, str) and value.strip() for value in descriptions.values()):
        raise ValueError(
            f"empty or non-string tool description in {description_source}"
        )
    if manifest.get("schema_version") == "openeta.operator_context_profile.v2":
        # Integrity covers the authored recipe, not the runtime-expanded
        # contract/invariant view assembled from its layers.  This keeps the
        # manifest hash stable while still giving component overrides their
        # own resolved composition identity.
        integrity = build_manifest_integrity(
            authored_manifest,
            resolved_components=components,
        )
        manifest["integrity"] = integrity
        composition_sha256 = str(integrity["composition_sha256"])
        manifest_sha256 = str(integrity["manifest_sha256"])
    else:
        composition_sha256 = canonical_sha256(components)
        manifest_sha256 = canonical_sha256(manifest)
    return OperatorContextProfile(
        profile_id=selected,
        revision=revision,
        status=str(manifest.get("status", "draft")),
        manifest=manifest,
        prompt_template=prompt_template,
        tool_descriptions=descriptions,
        directory=directory.resolve(),
        components=components,
        composition_sha256=composition_sha256,
        manifest_sha256=manifest_sha256,
    )


def active_profile() -> OperatorContextProfile:
    return load_profile()


def tool_description(name: str) -> str:
    """Return one profile-owned public tool description or fail closed."""
    profile = active_profile()
    try:
        return profile.tool_descriptions[name]
    except KeyError as exc:
        raise RuntimeError(
            "Operator context profile "
            f"{profile.label!r} does not define public tool {name!r}; "
            f"refusing to use an unversioned fallback description "
            f"(profile directory: {profile.directory})"
        ) from exc


def public_result_schema(name: str) -> dict[str, Any] | None:
    """Return the active profile's Operator-facing result schema, if enabled."""

    invariants = active_profile().manifest.get("invariants", {})
    schemas = invariants.get("public_result_schemas")
    if schemas is None:
        return None
    if not isinstance(schemas, dict):
        raise RuntimeError("public_result_schemas must be an object")
    schema = schemas.get(name)
    if not isinstance(schema, dict):
        raise RuntimeError(
            f"active Operator context requires a public result schema for {name!r}"
        )
    Draft202012Validator.check_schema(schema)
    return schema


def validate_public_result(name: str, payload: Mapping[str, Any]) -> None:
    """Fail closed when a runtime result drifts from the versioned contract."""

    schema = public_result_schema(name)
    if schema is None:
        return
    errors = sorted(
        Draft202012Validator(schema).iter_errors(dict(payload)),
        key=lambda error: list(error.absolute_path),
    )
    if not errors:
        return
    summaries = []
    for error in errors[:5]:
        path = ".".join(str(part) for part in error.absolute_path) or "<root>"
        summaries.append(f"{path}: {error.message}")
    raise RuntimeError(
        f"Operator public result contract violation for {name!r}: "
        + "; ".join(summaries)
    )


def _schema_property_names(schema: Mapping[str, Any]) -> set[str]:
    """Collect top-level object properties from a public-result schema.

    Public result schemas are intentionally expressed as ``oneOf`` success and
    error objects.  The runtime needs one stable allow-list for the public
    boundary, while ``details`` remains the lossless diagnostic channel.
    """

    names: set[str] = set()
    defs = schema.get("$defs", {})

    def visit(node: Any) -> None:
        if not isinstance(node, Mapping):
            return
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            target = defs.get(ref.rsplit("/", 1)[-1])
            if target is not None:
                visit(target)
        properties = node.get("properties")
        if isinstance(properties, Mapping):
            names.update(str(key) for key in properties)
        for key in ("oneOf", "anyOf", "allOf"):
            branches = node.get(key)
            if isinstance(branches, list):
                for branch in branches:
                    visit(branch)

    visit(schema)
    return names


def project_public_result(
    name: str,
    payload: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Project a runtime result onto the active versioned public contract.

    This is deliberately performed at the final boundary rather than in every
    internal branch.  Rich branch-specific data is retained by the caller in
    ``GatewayResult.details``; the Operator receives only the schema-owned
    surface.  The returned payload is still validated, so projection cannot
    turn an otherwise malformed result into a contract-valid one.
    """

    schema = public_result_schema(name)
    if schema is None:
        return dict(payload), []
    allowed = _schema_property_names(schema)
    source = dict(payload)
    aliases: dict[str, str] = {}
    if name == "observe":
        aliases = {
            "returned_views": "views",
            "actual_grip_site_xyz_m": "grip_xyz_m",
            "gripper_aperture_mm": "aperture_mm",
            "gripper_reference_mm": "aperture_limits_mm",
        }
    elif name == "move_to":
        aliases = {
            "motion_status": "motion",
            "actual_grip_site_xyz_m": "grip_xyz_m",
            "returned_views": "views",
            "remaining_target_delta_mm": "remaining_delta_mm",
            "gripper_aperture_mm": "aperture_mm",
            "gripper_reference_mm": "aperture_limits_mm",
        }
    for internal_name, public_name in aliases.items():
        if public_name in allowed and public_name not in source:
            if source.get(internal_name) is not None:
                source[public_name] = source[internal_name]
    if name == "move_to":
        directions = payload.get("actual_directions_world")
        if isinstance(directions, Mapping):
            if "approach_world" in allowed and directions.get("approach") is not None:
                source["approach_world"] = directions["approach"]
            if "jaw_world" in allowed and directions.get("jaw") is not None:
                source["jaw_world"] = directions["jaw"]
    projected = {key: value for key, value in source.items() if key in allowed}
    # Older internal branches used ``view`` for errors.  The public contract
    # intentionally calls this ``source_view`` so the same field has one
    # meaning across pending, solved, and error results.
    if (
        name == "mark_point"
        and "source_view" in allowed
        and "source_view" not in projected
        and payload.get("view") is not None
    ):
        projected["source_view"] = payload["view"]
    if (
        name == "move_to"
        and "gripper_aperture_mm" in allowed
        and "gripper_aperture_mm" not in projected
    ):
        gripper = payload.get("gripper")
        if isinstance(gripper, Mapping) and isinstance(
            gripper.get("aperture_mm"), (int, float)
        ):
            projected["gripper_aperture_mm"] = gripper["aperture_mm"]
    removed = sorted(set(payload) - set(projected))
    validate_public_result(name, projected)
    return projected, removed


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_initial_contract(
    path: Path,
    *,
    profile: OperatorContextProfile,
    task: str,
    prompt: str,
    model: str,
    reasoning_effort: str | None = None,
    operator_root: str,
    yolo: bool,
) -> dict[str, Any]:
    payload = {
        "schema_version": "openeta.operator_context_contract.v2",
        "context_profile": {
            "profile_id": profile.profile_id,
            "revision": profile.revision,
            "label": profile.label,
            "status": profile.status,
            "manifest": profile.manifest,
            "components": profile.components,
            "composition_sha256": profile.composition_sha256,
            "manifest_sha256": profile.manifest_sha256,
        },
        "task": task,
        "prompt": prompt,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "operator_root": operator_root,
        "clean_context": {
            "fresh_codex_home": True,
            "memories": False,
            "history_persistence": "none",
            "builder_session_history": False,
            "enabled_mcp_servers": ["operator"],
            "local_shell_and_python": True,
            "yolo": yolo,
        },
        "resolved": False,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def finalize_contract(path: Path, tools: list[Any]) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    profile = active_profile()
    recorded_profile = payload.get("context_profile")
    recorded_profile = (
        recorded_profile if isinstance(recorded_profile, dict) else {}
    )
    if recorded_profile.get("label") != profile.label:
        raise RuntimeError(
            "Operator context contract/profile mismatch: "
            f"contract={recorded_profile.get('label')!r}, "
            f"runtime={profile.label!r}"
        )
    if (
        recorded_profile.get("composition_sha256")
        != profile.composition_sha256
    ):
        raise RuntimeError(
            "Operator context contract/component composition mismatch: "
            f"contract={recorded_profile.get('composition_sha256')!r}, "
            f"runtime={profile.composition_sha256!r}"
        )
    if (
        recorded_profile.get("manifest_sha256")
        != profile.manifest_sha256
    ):
        raise RuntimeError(
            "Operator context contract/profile manifest mismatch: "
            f"contract={recorded_profile.get('manifest_sha256')!r}, "
            f"runtime={profile.manifest_sha256!r}"
        )

    tools_by_name = {tool.name: tool for tool in tools}
    expected_names = set(profile.public_operator_tools)
    missing_names = sorted(expected_names - set(tools_by_name))
    if missing_names:
        raise RuntimeError(
            f"Operator context profile {profile.label!r} requires missing "
            f"public tools: {missing_names}"
        )
    if "public_operator_tools" in profile.manifest:
        unexpected_names = sorted(set(tools_by_name) - expected_names)
        if unexpected_names:
            raise RuntimeError(
                f"Operator context profile {profile.label!r} does not allow "
                f"unexpected public tools: {unexpected_names}"
            )
    mismatched_descriptions = sorted(
        name
        for name in expected_names
        if tools_by_name[name].description != profile.tool_descriptions[name]
    )
    if mismatched_descriptions:
        raise RuntimeError(
            f"Operator context profile {profile.label!r} does not match the "
            "resolved MCP tool descriptions for: "
            f"{mismatched_descriptions}"
        )

    expected_input_schema_sha256 = profile.manifest.get(
        "invariants", {}
    ).get("public_input_schema_sha256")
    if expected_input_schema_sha256 is not None:
        if not isinstance(expected_input_schema_sha256, Mapping):
            raise RuntimeError(
                "public_input_schema_sha256 must be an object"
            )
        if set(expected_input_schema_sha256) != expected_names:
            raise RuntimeError(
                "public_input_schema_sha256 tools do not match "
                f"public_operator_tools: hashes="
                f"{sorted(expected_input_schema_sha256)}, "
                f"tools={sorted(expected_names)}"
            )
        drifted_input_schemas = sorted(
            name
            for name in expected_names
            if expected_input_schema_sha256.get(name)
            != canonical_sha256(tools_by_name[name].parameters)
        )
        if drifted_input_schemas:
            raise RuntimeError(
                f"Operator context profile {profile.label!r} input schema "
                f"digest mismatch for: {drifted_input_schemas}"
            )

    public_tools = []
    for tool in sorted(tools, key=lambda item: item.name):
        public_tools.append(
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
        )
    resolved_context = {
        "profile": payload["context_profile"],
        "startup_prompt": payload["prompt"],
        "public_tools": public_tools,
    }
    payload["public_tools"] = public_tools
    payload["resolved_context_sha256"] = canonical_sha256(resolved_context)
    payload["resolved"] = True
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return payload


def record_contract_resolution_failure(path: Path, error: BaseException) -> None:
    """Persist a startup failure when an initial episode contract exists."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            payload = {}
    except Exception:
        payload = {}
    payload["resolved"] = False
    payload["resolution_error"] = {
        "type": type(error).__name__,
        "message": str(error),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
