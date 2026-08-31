"""Optional OpenETA protocol adapter for :mod:`tools.manual_vlm_proxy`.

This is the only manual-console module that understands OpenETA schemas,
operator semantics, tool descriptors, or typed decision XML.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from typing import Any, Mapping

from tools.manual_vlm_protocol import EncodedResponse, JsonObject


_DEFAULT_VALUE_PATTERN = re.compile(r"\bdefaults?\s+to\s+([^.;]+)", re.IGNORECASE)


def _message_texts(body: JsonObject) -> list[str]:
    texts: list[str] = []
    messages = body.get("messages")
    if not isinstance(messages, list):
        return texts
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if isinstance(content, str):
            texts.append(content)
        elif isinstance(content, list):
            texts.extend(
                str(part.get("text"))
                for part in content
                if isinstance(part, dict)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
            )
    return texts


def _json_values_from_text(text: str) -> list[Any]:
    values: list[Any] = []
    decoder = json.JSONDecoder()
    stripped = text.strip()
    try:
        return [json.loads(stripped)]
    except json.JSONDecodeError:
        pass
    positions = [0]
    positions.extend(index + 1 for index, char in enumerate(text) if char == "\n")
    for position in positions:
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text) or text[position] not in "{[":
            continue
        try:
            value, _end = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            continue
        values.append(value)
    return values


def _wire_json_values(body: JsonObject) -> list[Any]:
    values: list[Any] = [body]
    for text in _message_texts(body):
        values.extend(_json_values_from_text(text))
    return values


def _walk_json(value: Any) -> list[tuple[str, Any]]:
    found: list[tuple[str, Any]] = []
    if isinstance(value, dict):
        for key, nested in value.items():
            found.append((str(key), nested))
            found.extend(_walk_json(nested))
    elif isinstance(value, list):
        for nested in value:
            found.extend(_walk_json(nested))
    return found


def _planner_wire_payload(body: JsonObject) -> JsonObject:
    candidates = [
        value
        for value in _wire_json_values(body)
        if isinstance(value, dict)
        and isinstance(value.get("tool_context"), dict)
        and isinstance(value.get("instruction"), str)
    ]
    return candidates[-1] if candidates else {}


def _stable_wire_context(body: JsonObject) -> JsonObject:
    candidates = [
        value
        for value in _wire_json_values(body)
        if isinstance(value, dict)
        and value.get("schema_version") == "openeta.planner_static_context.v1"
    ]
    return candidates[-1] if candidates else {}


def extract_tool_catalog(body: JsonObject) -> list[JsonObject]:
    """Extract OpenETA's serialized ``available_tools`` descriptors."""

    tools: dict[str, JsonObject] = {}
    for value in _wire_json_values(body):
        for key, nested in _walk_json(value):
            if key != "available_tools" or not isinstance(nested, list):
                continue
            for item in nested:
                if isinstance(item, str) and item:
                    candidate: JsonObject = {"name": item}
                elif isinstance(item, dict) and isinstance(item.get("name"), str):
                    candidate = dict(item)
                else:
                    continue
                name = str(candidate["name"])
                previous = tools.get(name)
                if previous is None or len(json.dumps(candidate)) > len(json.dumps(previous)):
                    tools[name] = candidate
    return [tools[name] for name in sorted(tools)]


def _parameter_value_type(name: str, description: str, declared: str = "") -> str:
    normalized = declared.strip().lower()
    if normalized in {"string", "integer", "number", "boolean", "json"}:
        return normalized
    if normalized in {"array", "object"}:
        return "json"
    normalized_name = name.strip().lower()
    if (
        normalized_name.endswith(("_id", "_path", "_ref"))
        or normalized_name
        in {"code", "depth", "image", "mode", "prompt", "rgb", "url"}
    ):
        return "string"
    if normalized_name in {"hints", "intrinsics", "object_mask"}:
        return "json"
    text = f"{name} {description}".lower()
    if any(marker in text for marker in ("boolean", "toggle", "true or false")):
        return "boolean"
    json_patterns = (
        r"\bjson\b",
        r"\blist(?:\s+of)?\b",
        r"\barray\b",
        r"\bdict\b",
        r"\bintrinsics\b",
        r"\bmatrix\b",
        r"\bbbox\b",
        r"\bpose\b",
        r"\btrajectory\b",
        r"\bconfig(?:uration)?\b",
        r"\{\s*x\s*,",
        r"\[\s*left\s*,",
    )
    if re.search(r"(?:^|_)points?$", name.lower()) or any(
        re.search(pattern, text) for pattern in json_patterns
    ):
        return "json"
    if any(
        marker in text
        for marker in ("integer", " count", "_count", "width", "height", "seed", "limit", "max_")
    ):
        return "integer"
    if any(
        marker in text
        for marker in ("number", "float", "scale", "timestamp", "resolution", "timeout", "position")
    ):
        return "number"
    return "string"


def _coerce_documented_default(value: Any, value_type: str) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip().strip("`\"")
    lowered = text.lower()
    if value_type == "boolean":
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    if value_type == "integer":
        try:
            return int(text)
        except ValueError:
            return text
    if value_type == "number":
        try:
            return float(text)
        except ValueError:
            return text
    if value_type == "json":
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return text
    if lowered in {"none", "null"}:
        return None
    return text


def _parameter_choices(description: str) -> list[str]:
    first_clause = description.split(";", 1)[0].strip()
    if "|" not in first_clause:
        return []
    values = [value.strip().strip("`\"") for value in first_clause.split("|")]
    if not 2 <= len(values) <= 8 or any(not value or len(value) > 64 for value in values):
        return []
    return values


def _normalize_parameter_field(name: str, spec: Any) -> JsonObject:
    if isinstance(spec, dict):
        description = str(spec.get("description") or spec.get("help") or "")
        declared_type = str(spec.get("type") or "")
        explicit_required = spec.get("required")
        has_default = "default" in spec
        default = spec.get("default")
        raw_choices = spec.get("enum") or spec.get("choices") or []
        choices = [str(value) for value in raw_choices] if isinstance(raw_choices, list) else []
        exclusive = spec.get("exclusive") is True
    else:
        description = str(spec or "")
        declared_type = ""
        explicit_required = None
        default_match = _DEFAULT_VALUE_PATTERN.search(description)
        has_default = default_match is not None
        default = default_match.group(1).strip() if default_match else None
        choices = _parameter_choices(description)
        lowered_description = description.lower()
        exclusive = (
            "when supplied" in lowered_description
            and re.search(r"\bhost\s+(?:atomically\s+)?resolves?\b", lowered_description)
            is not None
        )
    lowered = description.lower()
    if isinstance(explicit_required, bool):
        required: bool | None = explicit_required
    elif re.search(r"\brequired\s+(?:for|when|if|in)\b", lowered):
        required = False
    elif re.search(r"\brequired\b", lowered):
        required = True
    elif re.search(r"\boptional\b|\bomitt?\b", lowered) or has_default:
        required = False
    else:
        required = None
    value_type = _parameter_value_type(name, description, declared_type)
    return {
        "name": name,
        "description": description,
        "value_type": value_type,
        "required": required,
        "has_default": has_default,
        "default": _coerce_documented_default(default, value_type) if has_default else None,
        "choices": choices,
        "exclusive": exclusive,
    }


def _move_to_profile(context: JsonObject) -> str:
    """Choose the operator-facing move form from current environment evidence."""

    objective = context.get("objective")
    objective = objective if isinstance(objective, dict) else {}
    active = objective.get("active_environment_task")
    active = active if isinstance(active, dict) else {}
    env_id = str(active.get("env_id") or "").lower()
    if "behavior" in env_id or "b1k" in env_id:
        return "bimanual"

    current = context.get("current_observation")
    current = current if isinstance(current, dict) else {}
    summary = current.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    robot = summary.get("robot")
    robot = robot if isinstance(robot, dict) else {}
    metadata = robot.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    arms = metadata.get("arms")
    if isinstance(arms, dict) and {"left", "right"}.issubset(arms):
        return "bimanual"
    return "single"


def _move_to_fields(profile: str) -> list[JsonObject]:
    def field(
        name: str,
        description: str,
        *,
        required: bool = False,
        default: Any = ...,
        value_type: str = "number",
    ) -> JsonObject:
        spec: JsonObject = {
            "type": value_type,
            "description": description,
            "required": required,
        }
        if default is not ...:
            spec["default"] = default
        return _normalize_parameter_field(name, spec)

    if profile == "bimanual":
        fields = [
            field(f"{arm}_{axis}", f"Required {arm} EEF world-frame {axis.upper()} in metres.", required=True)
            for arm in ("left", "right")
            for axis in ("x", "y", "z")
        ]
        fields.extend(
            field(
                f"{arm}_{axis}",
                f"Optional {arm} EEF {axis} in degrees; provide roll, pitch, and yaw together.",
            )
            for arm in ("left", "right")
            for axis in ("roll", "pitch", "yaw")
        )
        fields.extend(
            [
                field("tolerance", "Position tolerance in metres.", default=0.01),
                field(
                    "orientation_tolerance",
                    "Orientation tolerance in radians.",
                    default=0.05,
                ),
                field(
                    "max_ticks",
                    "Maximum coordinated playback ticks.",
                    default=800,
                    value_type="integer",
                ),
            ]
        )
        return fields

    return [
        field(axis, f"Required EEF world-frame {axis.upper()} in metres.", required=True)
        for axis in ("x", "y", "z")
    ] + [
        field(axis, f"Optional EEF {axis} in degrees; provide roll, pitch, and yaw together.")
        for axis in ("roll", "pitch", "yaw")
    ] + [
        field("num_steps", "Maximum closed-loop controller iterations.", value_type="integer"),
        field("tolerance", "Position tolerance in metres."),
        field("ori_tolerance", "Orientation tolerance in radians."),
        field(
            "enable_collision_check",
            "Whether to enable simulator collision checking.",
            value_type="boolean",
        ),
    ]


def _normalize_move_to_submission(body: JsonObject, parameters: JsonObject) -> JsonObject:
    """Turn the single-arm primitive form back into OpenETA's stable pose contract."""

    payload = _planner_wire_payload(body)
    context = payload.get("tool_context")
    context = context if isinstance(context, dict) else {}
    if _move_to_profile(context) == "bimanual" or "target_pose" in parameters:
        return parameters
    if not all(axis in parameters for axis in ("x", "y", "z")):
        return parameters

    resolved = context.get("host_resolved_inputs")
    resolved = resolved if isinstance(resolved, dict) else {}
    move_input = resolved.get("move_to")
    move_input = move_input if isinstance(move_input, dict) else {}
    host_parameters = move_input.get("call_parameters")
    host_parameters = host_parameters if isinstance(host_parameters, dict) else {}
    host_pose = host_parameters.get("target_pose")
    pose = dict(host_pose) if isinstance(host_pose, dict) else {"frame": "world"}
    pose["xyz"] = [parameters[axis] for axis in ("x", "y", "z")]

    orientation = [parameters.get(axis) for axis in ("roll", "pitch", "yaw")]
    if any(value is not None for value in orientation):
        if not all(value is not None for value in orientation):
            raise ValueError("move_to roll, pitch, and yaw must be provided together")
        pose["euler_xyz_deg"] = orientation

    normalized = {
        key: value
        for key, value in parameters.items()
        if key not in {"x", "y", "z", "roll", "pitch", "yaw"}
    }
    normalized["target_pose"] = pose
    return normalized


def build_tool_form_catalog(body: JsonObject) -> list[JsonObject]:
    payload = _planner_wire_payload(body)
    context = payload.get("tool_context")
    context = context if isinstance(context, dict) else {}
    output: list[JsonObject] = []
    catalog = extract_tool_catalog(body)
    if not any(str(tool.get("name") or "") == "move_to" for tool in catalog):
        backend_move = next(
            (tool for tool in catalog if str(tool.get("name") or "") == "move_eefs"),
            None,
        )
        if backend_move is not None:
            alias = dict(backend_move)
            alias["name"] = "move_to"
            catalog.append(alias)
    for tool in catalog:
        raw_parameters = tool.get("parameters")
        raw_parameters = raw_parameters if isinstance(raw_parameters, dict) else {}
        name = str(tool.get("name") or "")
        if name == "move_eefs":
            continue
        profile = _move_to_profile(context) if name == "move_to" else ""
        fields = (
            _move_to_fields(profile)
            if name == "move_to"
            else [
                _normalize_parameter_field(str(parameter_name), spec)
                for parameter_name, spec in raw_parameters.items()
            ]
        )
        form: JsonObject = {
            "name": name,
            "description": tool.get("description") or tool.get("category") or "",
            "fields": fields,
        }
        if profile:
            form["control_profile"] = profile
        exclusive_fields = [
            str(field["name"]) for field in fields if field.get("exclusive") is True
        ]
        if len(exclusive_fields) == 1:
            form["exclusive_parameter"] = exclusive_fields[0]
        output.append(form)
    return output


def detect_response_mode(body: JsonObject) -> str:
    response_format = body.get("response_format")
    if isinstance(response_format, dict) and response_format.get("type") == "json_object":
        return "json"
    for text in _message_texts(body):
        if "return only the exact json object" in text.lower():
            return "json"
    return "decision"


def _request_label_from_schema(schema: str, *, response_mode: str) -> tuple[str, str]:
    labels = {
        "openeta.agent_context.v2": ("main_planner", "Main planner"),
        "openeta.visual_delta_request.v1": ("visual_differencing", "Visual differencing"),
        "openeta.reference_localization_request.v1": (
            "reference_localization",
            "Reference localization",
        ),
        "openeta.grasp_pose_advisor_request.v1": ("grasp_advisor", "Grasp advisor"),
    }
    if schema in labels:
        return labels[schema]
    if schema:
        readable = schema.removeprefix("openeta.").removesuffix(".v1")
        return readable.replace(".", "_"), readable.replace("_", " ").title()
    if response_mode == "json":
        return "isolated_json", "Isolated JSON planner"
    return "planner", "Planner"


def classify_request(body: JsonObject) -> JsonObject:
    payload = _planner_wire_payload(body)
    context = payload.get("tool_context")
    context = context if isinstance(context, dict) else {}
    stable = _stable_wire_context(body)
    schema = str(context.get("schema_version") or stable.get("agent_context_schema_version") or "")
    response_mode = detect_response_mode(body)
    request_type, label = _request_label_from_schema(schema, response_mode=response_mode)
    objective = context.get("objective")
    objective = objective if isinstance(objective, dict) else {}
    errors = payload.get("validation_errors")
    errors = errors if isinstance(errors, list) else []
    attempt = payload.get("attempt")
    if not isinstance(attempt, int) or isinstance(attempt, bool):
        attempt = 1
    return {
        "type": request_type,
        "label": label,
        "schema_version": schema,
        "response_mode": response_mode,
        "attempt": attempt,
        "validation_error_count": len(errors),
        "task": str(objective.get("task") or context.get("task") or ""),
    }


def extract_tool_audit_records(body: JsonObject) -> list[JsonObject]:
    """Pair serialized planner actions with host results using ``action_id``."""

    messages = body.get("messages")
    if not isinstance(messages, list):
        return []
    actions: dict[str, JsonObject] = {}
    results: list[tuple[int, JsonObject]] = []
    for message_index, message in enumerate(messages):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        texts = [content] if isinstance(content, str) else []
        if isinstance(content, list):
            texts.extend(
                str(part.get("text"))
                for part in content
                if isinstance(part, dict)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
            )
        for text in texts:
            for value in _json_values_from_text(text):
                if not isinstance(value, dict):
                    continue
                action = value.get("openeta_action")
                if isinstance(action, dict):
                    action_id = action.get("action_id")
                    request = action.get("request")
                    if isinstance(action_id, str) and isinstance(request, dict):
                        actions[action_id] = {
                            "request": dict(request),
                            "message_index": message_index,
                        }
                host_result = value.get("openeta_host_result")
                if isinstance(host_result, dict):
                    results.append((message_index, host_result))

    records: list[JsonObject] = []
    for message_index, host_result in results:
        action_id = str(host_result.get("action_id") or "")
        action = actions.get(action_id, {})
        request = action.get("request")
        request = request if isinstance(request, dict) else {}
        calls = host_result.get("tool_calls")
        calls = calls if isinstance(calls, list) else []
        for call_index, call in enumerate(calls):
            if not isinstance(call, dict):
                continue
            result = call.get("result")
            result = dict(result) if isinstance(result, dict) else {}
            success = result.get("success")
            if not isinstance(success, bool):
                operational = result.get("operational_success")
                success = operational if isinstance(operational, bool) else None
            name = str(call.get("name") or request.get("name") or "tool")
            parameters = request.get("parameters")
            parameters = dict(parameters) if isinstance(parameters, dict) else {}
            record_id = (
                f"{action_id}:{call_index}"
                if action_id
                else f"message-{message_index}:{call_index}:{name}"
            )
            records.append(
                {
                    "id": record_id,
                    "kind": "tool_call",
                    "title": name,
                    "status": str(call.get("status") or host_result.get("status") or ""),
                    "success": success,
                    "arguments": parameters,
                    "result": result,
                    "action_id": action_id,
                    "action_message_index": action.get("message_index"),
                    "result_message_index": message_index,
                    "source": "wire_conversation",
                }
            )
    return records


def _nested_field(value: Any, *names: str) -> Any:
    """Return the first shallowest non-empty field from a nested tool result."""

    queue = [value]
    while queue:
        current = queue.pop(0)
        if isinstance(current, dict):
            for name in names:
                candidate = current.get(name)
                if candidate is not None and candidate != "":
                    return candidate
            queue.extend(current.values())
        elif isinstance(current, list):
            queue.extend(current)
    return None


def _latest_tool_record(
    audit_records: list[JsonObject],
    name: str,
) -> JsonObject | None:
    matches = [
        (index, record)
        for index, record in enumerate(audit_records)
        if isinstance(record, dict) and record.get("title") == name
    ]
    if not matches:
        return None

    def order(item: tuple[int, JsonObject]) -> tuple[int, int, int]:
        index, record = item
        turn = record.get("last_seen_turn")
        message = record.get("result_message_index")
        return (
            turn if isinstance(turn, int) else -1,
            message if isinstance(message, int) else -1,
            index,
        )

    return max(matches, key=order)[1]


def _compact_tool_state(
    audit_records: list[JsonObject],
    name: str,
) -> JsonObject:
    record = _latest_tool_record(audit_records, name)
    if record is None:
        return {"status": "not_run"}
    result = record.get("result")
    result = result if isinstance(result, dict) else {}
    outputs = result.get("outputs")
    outputs = outputs if isinstance(outputs, dict) else result
    arguments = record.get("arguments")
    arguments = arguments if isinstance(arguments, dict) else {}
    state: JsonObject = {
        "status": record.get("status") or "unknown",
        "success": record.get("success"),
    }
    turn = record.get("last_seen_turn")
    if isinstance(turn, int):
        state["turn"] = turn

    if name == "observe":
        reason = arguments.get("reason")
        if reason:
            state["reason"] = reason
        return state

    if name == "sam3":
        fields = {
            "result_id": _nested_field(outputs, "result_id"),
            "source_packet_id": arguments.get("source_packet_id")
            or _nested_field(outputs, "source_packet_id"),
            "camera_frame_id": arguments.get("camera_frame_id")
            or _nested_field(outputs, "camera_frame_id"),
            "mode": arguments.get("mode") or _nested_field(outputs, "mode"),
            "prompt": arguments.get("prompt") or _nested_field(outputs, "prompt"),
            "evidence_role": arguments.get("evidence_role")
            or _nested_field(outputs, "evidence_role"),
            "detection_count": _nested_field(outputs, "detection_count"),
        }
    else:
        fields = {
            "result_id": _nested_field(outputs, "result_id"),
            "input_bundle_id": arguments.get("bundle_id"),
            "source_packet_id": _nested_field(outputs, "source_packet_id", "packet_id"),
            "camera_frame_id": _nested_field(outputs, "camera_frame_id", "frame_id"),
            "source_backend": _nested_field(outputs, "source_backend", "backend"),
            "candidate_count": _nested_field(outputs, "candidate_count"),
        }
    state.update(
        {key: value for key, value in fields.items() if value is not None and value != ""}
    )
    return state


def _latest_context_artifact(context: JsonObject, tool: str) -> JsonObject | None:
    artifacts = context.get("artifacts")
    artifacts = artifacts if isinstance(artifacts, dict) else {}
    candidates = [
        value
        for value in artifacts.values()
        if isinstance(value, dict) and value.get("tool") == tool
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda value: (
            float(value.get("timestamp_s"))
            if isinstance(value.get("timestamp_s"), (int, float))
            else -1.0
        ),
    )


def _fallback_tool_state(
    context: JsonObject,
    name: str,
    state: JsonObject,
) -> JsonObject:
    decision_state = context.get("decision_state")
    decision_state = decision_state if isinstance(decision_state, dict) else {}
    effect = decision_state.get("last_action_effect")
    effect = effect if isinstance(effect, dict) else {}
    if effect.get("tool") == name:
        synthetic = {
            "title": name,
            "status": effect.get("status"),
            "success": effect.get("operational_success"),
            "result": {"outputs": effect.get("outputs") or {}},
        }
        # The wire audit projection intentionally keeps tool results compact. The
        # current decision state may carry useful fields such as result_id that were
        # omitted there, so fill holes without replacing session/turn provenance.
        state = {**_compact_tool_state([synthetic], name), **state}
    if state.get("status") != "not_run":
        return state

    artifact = _latest_context_artifact(context, name)
    if artifact is None:
        return state
    projected: JsonObject = {"status": "available_from_context"}
    for key in (
        "result_id",
        "candidate_count",
        "detection_count",
        "source_backend",
        "source_packet_id",
        "camera_frame_id",
    ):
        value = _nested_field(artifact, key)
        if value is not None and value != "":
            projected[key] = value
    return projected


def build_latest_state(
    body: JsonObject,
    *,
    audit_records: list[JsonObject] | None = None,
) -> JsonObject:
    """Project persistent session state without conflating live and frozen packets."""

    payload = _planner_wire_payload(body)
    context = payload.get("tool_context")
    context = context if isinstance(context, dict) else {}
    decision_state = context.get("decision_state")
    decision_state = decision_state if isinstance(decision_state, dict) else {}
    packet = decision_state.get("current_observation_packet")
    packet = packet if isinstance(packet, dict) else {}
    current = context.get("current_observation")
    current = current if isinstance(current, dict) else {}
    summary = current.get("summary")
    summary = summary if isinstance(summary, dict) else {}
    metadata = summary.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}

    packet_ids = packet.get("packet_ids")
    packet_ids = packet_ids if isinstance(packet_ids, list) else []
    camera_artifacts = packet.get("camera_artifacts")
    camera_artifacts = camera_artifacts if isinstance(camera_artifacts, list) else []
    camera_frame_ids = list(
        dict.fromkeys(
            str(item.get("frame_id"))
            for item in camera_artifacts
            if isinstance(item, dict) and item.get("frame_id")
        )
    )
    if not camera_frame_ids:
        raw_camera_ids = summary.get("camera_ids")
        if isinstance(raw_camera_ids, list):
            camera_frame_ids = [str(value) for value in raw_camera_ids]

    records = audit_records if audit_records is not None else extract_tool_audit_records(body)
    resolved = context.get("host_resolved_inputs")
    resolved = resolved if isinstance(resolved, dict) else {}
    grasp_input = resolved.get("grasp_pose_estimate")
    grasp_input = grasp_input if isinstance(grasp_input, dict) else {}
    grasp_input_resolution = {
        "status": grasp_input.get("status") or "unavailable",
        "bundle_id": grasp_input.get("bundle_id"),
        "source_packet_id": grasp_input.get("source_packet_id"),
        "sam3_result_id": grasp_input.get("sam3_result_id"),
        "detection_id": grasp_input.get("detection_id"),
    }
    grasp_input_resolution = {
        key: value
        for key, value in grasp_input_resolution.items()
        if value is not None and value != ""
    }
    anyplace_input = resolved.get("anyplace")
    anyplace_input = anyplace_input if isinstance(anyplace_input, dict) else {}
    anyplace_resolution = {
        key: anyplace_input.get(key)
        for key in (
            "status",
            "bundle_id",
            "materialized_result_id",
            "grasp_evidence_id",
            "placement_evidence_id",
            "repair_call",
            "recovery",
        )
        if anyplace_input.get(key) is not None
        and anyplace_input.get(key) != ""
    }
    resource_catalog = context.get("resource_catalog")
    resource_catalog = (
        resource_catalog if isinstance(resource_catalog, dict) else {}
    )
    latest_observe = _fallback_tool_state(
        context, "observe", _compact_tool_state(records, "observe")
    )
    latest_sam3 = _fallback_tool_state(
        context, "sam3", _compact_tool_state(records, "sam3")
    )
    latest_grasp = _fallback_tool_state(
        context,
        "grasp_pose_estimate",
        _compact_tool_state(records, "grasp_pose_estimate"),
    )
    return {
        "current_observation_packet": {
            "step": packet.get("step_idx", metadata.get("step_idx")),
            "packet_ids": packet_ids,
            "camera_frame_ids": camera_frame_ids,
        },
        "latest_observe": latest_observe,
        "latest_sam3": latest_sam3,
        "latest_grasp_pose_estimate": latest_grasp,
        "grasp_input_resolution": grasp_input_resolution,
        "anyplace_resolution": anyplace_resolution,
        "active_selections": resource_catalog.get("selections") or {},
        "retained_results": {
            key: resource_catalog.get(key) or []
            for key in (
                "detection_results",
                "grasp_results",
                "placement_results",
            )
        },
    }


def _compact_latest_action(context: JsonObject, observation: JsonObject) -> JsonObject:
    decision_state = context.get("decision_state")
    decision_state = decision_state if isinstance(decision_state, dict) else {}
    latest = decision_state.get("last_action_effect")
    if isinstance(latest, dict):
        return {
            key: latest.get(key)
            for key in (
                "tool",
                "status",
                "content",
                "operational_success",
                "semantic_outcome",
                "effect",
            )
            if latest.get(key) is not None
        }
    metadata = observation.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    previous = metadata.get("previous_action")
    if not isinstance(previous, dict):
        return {}
    calls = previous.get("tool_calls")
    call = calls[-1] if isinstance(calls, list) and calls and isinstance(calls[-1], dict) else {}
    result = call.get("result")
    result = result if isinstance(result, dict) else {}
    return {
        "tool": call.get("name") or previous.get("request_name"),
        "status": call.get("status") or previous.get("status"),
        "content": result.get("content"),
        "operational_success": result.get("success"),
    }


def _image_details(body: JsonObject) -> list[str]:
    details: list[str] = []
    messages = body.get("messages")
    if not isinstance(messages, list):
        return details
    for message in messages:
        content = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):
            for part in content:
                if not isinstance(part, dict) or part.get("type") != "image_url":
                    continue
                image_url = part.get("image_url")
                detail = image_url.get("detail") if isinstance(image_url, dict) else ""
                details.append(str(detail or ""))
    return details


def build_operator_summary(body: JsonObject, *, request_id: str) -> JsonObject:
    payload = _planner_wire_payload(body)
    context = payload.get("tool_context")
    context = context if isinstance(context, dict) else {}
    stable = _stable_wire_context(body)
    classification = classify_request(body)
    objective = context.get("objective")
    objective = objective if isinstance(objective, dict) else {}
    active_task = objective.get("active_environment_task")
    active_task = active_task if isinstance(active_task, dict) else {}
    current = context.get("current_observation")
    current = current if isinstance(current, dict) else {}
    observation = current.get("summary")
    observation = observation if isinstance(observation, dict) else {}
    metadata = observation.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    robot = observation.get("robot")
    robot = robot if isinstance(robot, dict) else {}
    decision_state = context.get("decision_state")
    decision_state = decision_state if isinstance(decision_state, dict) else {}
    skills = stable.get("relevant_skills")
    skills = skills if isinstance(skills, list) else []
    errors = payload.get("validation_errors")
    errors = [str(value) for value in errors] if isinstance(errors, list) else []
    evidence = context.get("vision_evidence")
    evidence = evidence if isinstance(evidence, list) else []
    paths = context.get("vision_image_paths")
    paths = paths if isinstance(paths, list) else []
    evidence_by_path = {
        str(item.get("path")): item
        for item in evidence
        if isinstance(item, dict) and item.get("path")
    }
    images: list[JsonObject] = []
    for index, detail in enumerate(_image_details(body)):
        path = str(paths[index]) if index < len(paths) else ""
        descriptor = evidence_by_path.get(path, {})
        images.append(
            {
                "index": index,
                "url": f"/api/requests/{request_id}/images/{index}",
                "label": str(
                    descriptor.get("camera_role")
                    or descriptor.get("role")
                    or descriptor.get("evidence_id")
                    or f"Image {index + 1}"
                ),
                "role": str(descriptor.get("role") or "visual evidence"),
                "freshness": str(descriptor.get("freshness") or ""),
                "detail": detail,
            }
        )
    return {
        "classification": classification,
        "instruction": str(payload.get("instruction") or ""),
        "task": str(objective.get("task") or active_task.get("task") or context.get("task") or ""),
        "environment": {"env_id": active_task.get("env_id"), "handle": active_task.get("handle")},
        "attempt": classification["attempt"],
        "validation_errors": errors,
        "observation": {
            "status": current.get("status"),
            "step": metadata.get("step_idx"),
            "fresh": metadata.get("observation_fresh"),
            "camera_ids": observation.get("camera_ids") or [],
            "object_count": observation.get("object_count"),
            "objects": observation.get("objects") or [],
            "end_effector_pose": robot.get("end_effector_pose"),
            "gripper_state": robot.get("gripper_state"),
        },
        "latest_action": _compact_latest_action(context, observation),
        "unresolved_obligations": decision_state.get("unresolved_obligations") or {},
        "open_questions": context.get("open_questions") or {},
        "host_resolved_inputs": context.get("host_resolved_inputs") or {},
        "skills": [
            {"name": item.get("name"), "description": item.get("description")}
            for item in skills
            if isinstance(item, dict) and item.get("name")
        ],
        "required_output": context.get("required_output"),
        "images": images,
    }


def _safe_xml_tag(name: object) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]", "_", str(name or "value"))
    return value if value and re.match(r"[A-Za-z_]", value) else f"field_{value}"


def _append_xml_value(parent: ET.Element, name: object, value: Any) -> ET.Element:
    element = ET.SubElement(parent, _safe_xml_tag(name))
    if value is None:
        element.set("type", "null")
    elif isinstance(value, bool):
        element.set("type", "boolean")
        element.text = "true" if value else "false"
    elif isinstance(value, int):
        element.set("type", "integer")
        element.text = str(value)
    elif isinstance(value, float):
        element.set("type", "number")
        element.text = str(value)
    elif isinstance(value, list):
        element.set("type", "array")
        for item in value:
            _append_xml_value(element, "item", item)
    elif isinstance(value, dict):
        element.set("type", "object")
        for key, nested in value.items():
            _append_xml_value(element, key, nested)
    else:
        element.set("type", "string")
        element.text = str(value)
    return element


def serialize_decision_xml(decision: JsonObject) -> str:
    kind = decision.get("kind")
    name = decision.get("name")
    parameters = decision.get("parameters", {})
    reasoning = decision.get("reasoning", "")
    if not isinstance(kind, str) or not kind.strip():
        raise ValueError("decision.kind must be a non-empty string")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("decision.name must be a non-empty string")
    if not isinstance(parameters, dict):
        raise ValueError("decision.parameters must be an object")
    root = ET.Element("decision")
    _append_xml_value(root, "kind", kind)
    _append_xml_value(root, "name", name)
    _append_xml_value(root, "parameters", parameters)
    _append_xml_value(root, "reasoning", reasoning)
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode", short_empty_elements=True)


class OpenETAProtocolAdapter:
    adapter_id = "openeta"
    label = "OpenETA"

    def classify_request(self, body: JsonObject) -> JsonObject:
        return classify_request(body)

    def session_identity(
        self,
        body: JsonObject,
        headers: Mapping[str, str],
    ) -> tuple[str, str]:
        for header in ("x-openeta-session-id", "x-session-id"):
            value = headers.get(header, "").strip()
            if value:
                return value, f"header:{header}"
        priority = ("active_agent_session_id", "agent_session_id", "session_id")
        for field in priority:
            for value in _wire_json_values(body):
                for key, nested in _walk_json(value):
                    if key == field and isinstance(nested, str) and nested.strip():
                        return nested.strip(), f"wire:{field}"
        return "", ""

    def attempt(self, body: JsonObject) -> int:
        for value in _wire_json_values(body):
            if isinstance(value, dict):
                attempt = value.get("attempt")
                if isinstance(attempt, int) and not isinstance(attempt, bool):
                    return attempt
        return 1

    def presentation(
        self,
        body: JsonObject,
        *,
        request_id: str,
        audit_records: list[JsonObject] | None = None,
    ) -> JsonObject:
        operator = build_operator_summary(body, request_id=request_id)
        classification = operator["classification"]
        sections: list[JsonObject] = []
        if operator["images"]:
            sections.append(
                {
                    "type": "images",
                    "title": "Visual evidence",
                    "items": operator["images"],
                    "column": "main",
                }
            )
        sections.extend(
            [
                {
                    "type": "json",
                    "title": "Latest state",
                    "value": build_latest_state(
                        body,
                        audit_records=audit_records,
                    ),
                    "column": "side",
                },
                {
                    "type": "json",
                    "title": "Current observation",
                    "value": operator["observation"],
                    "column": "side",
                },
                {
                    "type": "text",
                    "title": "Current instruction",
                    "value": operator["instruction"],
                    "column": "side",
                },
                {
                    "type": "json",
                    "title": "Latest action",
                    "value": operator["latest_action"],
                    "column": "side",
                },
                {
                    "type": "json",
                    "title": "Unresolved obligations",
                    "value": operator["unresolved_obligations"],
                    "column": "side",
                },
                {
                    "type": "json",
                    "title": "Questions / output contract",
                    "value": operator["open_questions"] or operator["required_output"] or {},
                    "column": "side",
                },
                {
                    "type": "json",
                    "title": "Host-resolved inputs",
                    "value": operator["host_resolved_inputs"],
                    "column": "side",
                },
                {
                    "type": "json",
                    "title": "Relevant skills",
                    "value": operator["skills"],
                    "column": "side",
                },
            ]
        )
        response_mode = classification["response_mode"]
        if response_mode == "json":
            composer: JsonObject = {
                "kind": "raw",
                "label": "JSON response",
                "placeholder": "Enter the complete JSON object required by this request.",
            }
        else:
            composer = {
                "kind": "tool_form",
                "label": "Structured response",
                "tools": build_tool_form_catalog(body),
                "reasoning": True,
                "actions": [
                    {"name": "talk", "label": "Talk", "message": True},
                    {"name": "ask_human", "label": "Ask human", "message": True},
                    {"name": "task_complete", "label": "Task complete", "message": False},
                ],
                "allow_raw": True,
            }
        return {
            "view": {
                "eyebrow": classification["label"],
                "title": operator["task"] or classification["label"],
                "badges": [
                    f"attempt {operator['attempt']}",
                    str(operator["environment"].get("env_id") or ""),
                ],
                "alerts": operator["validation_errors"],
                "sections": sections,
            },
            "composer": composer,
        }

    def audit_records(self, body: JsonObject) -> list[JsonObject]:
        return extract_tool_audit_records(body)

    def encode_response(
        self,
        body: JsonObject,
        submission: JsonObject,
        *,
        request_id: str,
    ) -> EncodedResponse:
        raw = submission.get("content")
        if isinstance(raw, str):
            if not raw.strip():
                raise ValueError("content must be a non-empty string")
            return EncodedResponse(message={"role": "assistant", "content": raw})
        decision = submission.get("decision")
        if not isinstance(decision, dict):
            intent = submission.get("intent")
            if not isinstance(intent, dict):
                raise ValueError("content or intent must be provided")
            intent_type = intent.get("type")
            name = intent.get("name")
            arguments = intent.get("arguments", {})
            reasoning = intent.get("reasoning", "")
            if intent_type == "tool_call":
                decision = {
                    "kind": "tool_call",
                    "name": name,
                    "parameters": arguments,
                    "reasoning": reasoning,
                }
            elif intent_type == "action":
                decision = {
                    "kind": "response",
                    "name": name,
                    "parameters": arguments,
                    "reasoning": reasoning,
                }
            else:
                raise ValueError("intent.type must be tool_call or action")
        if decision.get("kind") == "tool_call" and decision.get("name") == "move_to":
            parameters = decision.get("parameters")
            if isinstance(parameters, dict):
                decision = dict(decision)
                decision["parameters"] = _normalize_move_to_submission(body, parameters)
        content = serialize_decision_xml(decision)
        return EncodedResponse(message={"role": "assistant", "content": content})
