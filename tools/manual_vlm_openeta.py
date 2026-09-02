"""Optional OpenETA protocol adapter for :mod:`tools.manual_vlm_proxy`.

This is the only manual-console module that understands OpenETA schemas,
operator semantics, tool descriptors, or typed planner decisions. It follows
the request's declared response mode so the console works during JSON-to-XML
planner migrations without coupling the generic proxy to either encoding.
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


def _normalize_parameter_field(
    name: str,
    spec: Any,
    *,
    required_override: bool | None = None,
) -> JsonObject:
    if isinstance(spec, dict):
        description = str(spec.get("description") or spec.get("help") or "")
        declared_type = str(spec.get("type") or "")
        explicit_required = spec.get("required")
        has_default = "default" in spec
        default = spec.get("default")
        raw_choices = spec.get("enum") or spec.get("choices") or []
        choices = [str(value) for value in raw_choices] if isinstance(raw_choices, list) else []
    else:
        description = str(spec or "")
        declared_type = ""
        explicit_required = None
        default_match = _DEFAULT_VALUE_PATTERN.search(description)
        has_default = default_match is not None
        default = default_match.group(1).strip() if default_match else None
        choices = _parameter_choices(description)
    lowered = description.lower()
    if isinstance(required_override, bool):
        required: bool | None = required_override
    elif isinstance(explicit_required, bool):
        required: bool | None = explicit_required
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
    }


def build_tool_form_catalog(body: JsonObject) -> list[JsonObject]:
    output: list[JsonObject] = []
    for tool in extract_tool_catalog(body):
        raw_parameters = tool.get("parameters")
        raw_parameters = raw_parameters if isinstance(raw_parameters, dict) else {}
        schema_properties = raw_parameters.get("properties")
        if isinstance(schema_properties, dict):
            field_specs = schema_properties
            raw_required = raw_parameters.get("required")
            required_names = (
                {str(name) for name in raw_required if isinstance(name, str)}
                if isinstance(raw_required, list)
                else set()
            )
            schema_required = True
        else:
            field_specs = raw_parameters
            required_names = set()
            schema_required = False
        output.append(
            {
                "name": tool.get("name"),
                "description": tool.get("description") or tool.get("category") or "",
                "fields": [
                    _normalize_parameter_field(
                        str(name),
                        spec,
                        required_override=(str(name) in required_names)
                        if schema_required
                        else None,
                    )
                    for name, spec in field_specs.items()
                ],
            }
        )
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

    def presentation(self, body: JsonObject, *, request_id: str) -> JsonObject:
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
                    "title": "Relevant skills",
                    "value": operator["skills"],
                    "column": "side",
                },
            ]
        )
        response_mode = classification["response_mode"]
        if response_mode == "json" and classification["type"] != "main_planner":
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
        if detect_response_mode(body) == "json":
            content = json.dumps(decision, ensure_ascii=False, separators=(",", ":"))
        else:
            content = serialize_decision_xml(decision)
        return EncodedResponse(message={"role": "assistant", "content": content})
