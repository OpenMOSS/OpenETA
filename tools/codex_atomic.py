"""Experimental atomic tool surface over the existing budgeted Host runtime."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from uuid import uuid4

import numpy as np
from mcp.types import CallToolResult, ImageContent, TextContent, Tool

from agent.tools.registry import ENVIRONMENT_AUTHORITY
from tools.codex_atomic_geometry import (orientation, render, surface_point, vector,
    quat_matrix, matrix_quat, body_to_grip_site, grip_site_to_body)
from tools.codex_evidence import observation_references
from tools.codex_motion import run_motion_hook, run_symmetric_motion_hook
from tools.codex_orientation import equivalent_rotations, rotation_distance
from tools.codex_feedback import motion_feedback


def schema(properties, required=()):
    return {"type": "object", "properties": properties, "required": list(required),
            "additionalProperties": False}


VEC = {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}
STR = {"type": "string", "minLength": 1}


class AtomicTools:
    def __init__(self, host):
        self.host = host
        self.points = {}
        self.generation = 0
        self.pending = None
        self.contact = None
        self.symmetry_load_uncertain = False
        self.feedback = None
        self.overlay = {}
        self.selected_views = None
        self.crop = None
        self.audit = host.output / "atomic-commands.jsonl"
        self.schemas = {k: host.schemas[k] for k in ("episode_status", "finish_episode")}
        self.schemas.update({
            "observe": Tool(name="observe", description="Refresh RGB-D views and measured robot state. Optional crop is in original image pixels; cropped-image clicks must be mapped back using returned origin/scale.", inputSchema=schema({
                "camera": {"enum": ["agentview", "robot0_eye_in_hand", "wrist"]},
                "crop_xyxy": {"type": "array", "items": {"type": "integer"}, "minItems": 4, "maxItems": 4}})),
            "mark_point": Tool(name="mark_point", description="Measure the first visible surface at a pixel of a CURRENT RGB image. Uses aligned depth and calibration, no learned perception. Returns immutable world point and confirmation image. A surface point is not automatically a grip-site or grasp. Depth edges are reported; prefer an interior pixel. Use point + offset for free-space waypoints.", inputSchema=schema({
                "source_packet_id": STR, "camera_frame_id": STR,
                "x": {"type": "integer", "minimum": 0}, "y": {"type": "integer", "minimum": 0}},
                ("source_packet_id", "camera_frame_id", "x", "y"))),
            "move_to": Tool(name="move_to", description="Control the Panda grip-site. Choose xyz_m, point_id + optional offset_m, or delta_m in world/grip_site frame; omit position to rotate. approach_world specifies local +Z and jaw_world local +X; supply both for a full pose or omit both to preserve orientation. preview=true renders the proposed pose without motion. Execution automatically checks fresh IK then uses Mink collision checks. The default controller tracks a straight position segment with gradual shortest-angle rotation; a safe endpoint does not guarantee this path is executable. If path tracking is blocked, choose a different waypoint or rotate at a clear waypoint. contact_point_id explicitly requests gripper contact with that measured surface's object; supply only for near-contact moves. Execute consecutive waypoints as separate calls and inspect actual state. No arrival or preview proves grasp/task success.", inputSchema=schema({
                "xyz_m": VEC, "point_id": STR, "offset_m": VEC, "delta_m": VEC,
                "delta_frame": {"enum": ["world", "grip_site"]},
                "approach_world": VEC, "jaw_world": VEC,
                "orientation_mode": {"enum": ["strict", "parallel_jaw_symmetric"],
                    "description": "Default strict. Symmetric explicitly permits swapping fingers of an empty, open Panda gripper. Two IK candidates are ranked by joint travel, limit margin and rotation; controller path is not previewed. Do not use for held objects, fixture manipulation or required wrist view."},
                "contact_point_id": STR, "preview": {"type": "boolean"}})),
            "gripper_control": Tool(name="gripper_control", description="Open or close the gripper and return measured state plus final images. Close can use contact_point_id from a fresh surface mark, or the preceding contact move's binding. Opening clears attachment. Aperture alone does not establish object retention.", inputSchema=schema({
                "action": {"enum": ["open", "close"]}, "contact_point_id": STR}, ("action",))),
        })
        # Preserve the environment authority and original proxy handler. Only
        # these two experimental bindings replace private provenance resolvers.
        for name in ("move_to", "gripper_control"):
            original = host.runtime.tools._handlers[name]
            def wrapped(ctx, original=original):
                metadata = dict(ctx.metadata)
                metadata["_contact_authorization_resolver"] = self.motion_authorization
                metadata["_attachment_candidate_resolver"] = self.close_authorization
                return original(replace(ctx, metadata=metadata))
            host.runtime.tools.bind_handler(name, wrapped, replace=True, authority=ENVIRONMENT_AUTHORITY)

    def state(self):
        robot = self.host.runner.current_observation.robot
        pose = robot.end_effector_pose
        if "xyz" not in pose or "quat_xyzw" not in pose:
            return {"available": False, "reason": "Measured grip-site pose is unavailable; observe before motion"}
        rotation = body_to_grip_site(quat_matrix(pose["quat_xyzw"]))
        grip = robot.gripper_state
        return {"grip_xyz_m": pose["xyz"], "approach_world": rotation[:,2].tolist(),
                "jaw_world": rotation[:,0].tolist(),
                "gripper": {k: grip[k] for k in ("openness", "aperture_m", "position", "open", "is_open") if k in grip}}

    def sources(self):
        memory = self.host.runtime.memory
        result = []
        for ref in observation_references(memory)["current_packets"]:
            for camera in ref["camera_frame_ids"]:
                if camera not in {"agentview", "wrist", "robot0_eye_in_hand"}:
                    continue
                source = memory.resolve_observation_packet(ref["source_packet_id"], camera)
                result.append((ref["source_packet_id"], camera, source))
        return result

    def point(self, point_id, *, contact=False):
        point = self.points.get(point_id)
        if point is None:
            raise ValueError("Unknown point_id in this episode")
        if contact and point["generation"] != self.generation:
            raise ValueError("Contact point predates a gripper/contact action; mark a fresh visible surface")
        if contact and point["measurement"]["depth_edge"]:
            raise ValueError("Contact point lies on a depth discontinuity; mark an interior surface pixel")
        return point

    def authorization(self, point_id, xyz):
        point = self.point(point_id, contact=True)
        if np.linalg.norm(vector(xyz)-vector(point["xyz_m"])) > .10:
            raise ValueError("Contact authorization requires grip-site within 0.10 m of the measured point")
        return {"schema_version": "openeta.model_point_contact.v1",
                "source_kind": "model_rgbd_point", "session_id": self.host.runtime.memory.session_id,
                "point_id": point_id, "source_packet_id": point["source_packet_id"],
                "camera_frame_id": point["camera_frame_id"], "waypoint_role": "grasp_contact",
                "target_anchor_world_xyz": point["xyz_m"],
                "target_evidence_id": point_id, "object_scene_epoch": self.host.runtime.memory.object_scene_epoch()}

    def motion_authorization(self, pose):
        if self.pending is None:
            return None
        expected, grant = self.pending
        if not (np.allclose(vector(pose.get("xyz")), vector(expected["xyz"]), atol=1e-9, rtol=0)
                and np.allclose(quat_matrix(pose.get("quat_xyzw")), quat_matrix(expected["quat_xyzw"]), atol=1e-9, rtol=0)):
            raise ValueError("Atomic contact binding does not match the executing pose")
        return deepcopy(grant)

    def close_authorization(self):
        if self.contact is None:
            return None
        if np.linalg.norm(vector(self.state()["grip_xyz_m"])-vector(self.contact["target_anchor_world_xyz"])) > .10:
            raise ValueError("Previous contact point is outside the current grip-site contact envelope")
        return deepcopy(self.contact)

    def call(self, name, args):
        self.feedback, self.overlay, self.crop, self.selected_views = None, {}, None, None
        record = {"tool": name, "arguments": deepcopy(args), "generation": self.generation,
                  "started_episode_s": self.host.runner.elapsed_s, "request_index": self.host.requests}
        try:
            result = self._call(name, args)
            record["feedback"] = self.feedback
            record["is_error"] = result.isError
            return result
        except (ValueError, KeyError, OSError) as exc:
            record["error"] = str(exc)
            self.crop, self.selected_views = None, None
            return self.host.result(error={"code": "atomic_argument_or_geometry_error", "message": str(exc)})
        finally:
            self.pending = None
            record["ended_episode_s"] = self.host.runner.elapsed_s
            self.host._write_status()
            with self.audit.open("a") as stream:
                stream.write(json.dumps(record, ensure_ascii=False)+"\n")

    def _call(self, name, args):
        host = self.host
        if name == "mark_point":
            source = next((s for p,c,s in self.sources() if p == args["source_packet_id"] and c == args["camera_frame_id"]), None)
            if source is None:
                raise ValueError("Use the exact source_packet_id/camera_frame_id of a current image")
            xyz, measurement = surface_point(source, args["x"], args["y"])
            point_id = "point-" + uuid4().hex[:12]
            point = {"point_id": point_id, "xyz_m": xyz, "measurement": measurement,
                     "source_packet_id": args["source_packet_id"], "camera_frame_id": args["camera_frame_id"],
                     "pixel_xy": [args["x"], args["y"]], "generation": self.generation}
            self.points[point_id] = point
            self.feedback = {"point": point, "meaning": "first_visible_surface; does not track objects"}
            self.overlay = {"mark": xyz}
            return host.result()
        if name == "observe":
            self.selected_views = args.get("camera")
            self.crop = args.get("crop_xyxy")
            if self.crop and not self.selected_views:
                raise ValueError("Choose camera when requesting a crop")
            _, error = host._execute({"kind": "tool_call", "name": "observe", "parameters": {}})
            return host.result(error=error)
        if name == "move_to":
            current = host.runner.current_observation.robot.end_effector_pose
            current_site = body_to_grip_site(quat_matrix(current["quat_xyzw"]))
            rotation = orientation(matrix_quat(current_site), args.get("approach_world"), args.get("jaw_world"))
            choices = [k for k in ("xyz_m", "point_id", "delta_m") if k in args]
            if len(choices) > 1 or ("offset_m" in args and "point_id" not in args):
                raise ValueError("Choose one position form; offset_m requires point_id")
            if "delta_frame" in args and "delta_m" not in args:
                raise ValueError("delta_frame requires delta_m")
            xyz = vector(current["xyz"])
            if "xyz_m" in args:
                xyz = vector(args["xyz_m"])
            elif "point_id" in args:
                xyz = vector(self.point(args["point_id"])["xyz_m"]) + vector(args.get("offset_m", [0,0,0]))
            elif "delta_m" in args:
                delta = vector(args["delta_m"])
                if args.get("delta_frame", "world") == "grip_site":
                    delta = current_site @ delta
                xyz += delta
            pose = {"frame": "world", "xyz": xyz.tolist(), "quat_xyzw": matrix_quat(grip_site_to_body(rotation))}
            symmetric = (args.get("orientation_mode") == "parallel_jaw_symmetric"
                         and any(k in args for k in ("approach_world", "jaw_world")))
            if symmetric:
                openness = self.state().get("gripper", {}).get("openness")
                if (self.symmetry_load_uncertain or self.contact is not None
                        or isinstance(openness, bool) or not isinstance(openness, (int, float))
                        or not np.isfinite(openness) or openness < .95):
                    raise ValueError("Symmetric orientation requires a known empty, open gripper without an active contact constraint; use strict or complete release first")
            grant = self.authorization(args["contact_point_id"], xyz) if "contact_point_id" in args else None
            self.feedback = {"target": {"xyz_m": xyz.tolist(), "approach_world": rotation[:,2].tolist(), "jaw_world": rotation[:,0].tolist()},
                             "contact_requested": grant is not None}
            if args.get("preview", False):
                self.overlay = {"target": xyz.tolist(), "rotation": rotation}
                self.feedback["motion"] = "preview_only; no IK or physics executed"
                if symmetric:
                    self.feedback["orientation_selection"] = {"mode": "parallel_jaw_symmetric",
                        "selection_status": "geometry_only", "path_check": "not_run",
                        "candidate_quat_xyzw": [matrix_quat(r) for r in equivalent_rotations(rotation)]}
                return host.result()
            self.pending = (pose, grant) if grant else None
            if symmetric:
                poses = [{**pose, "quat_xyzw": matrix_quat(grip_site_to_body(r))} for r in equivalent_rotations(rotation)]
                def bind_selected(selected):
                    self.pending = (selected, grant) if grant else None
                    self.feedback["requested_target"] = deepcopy(self.feedback["target"])
                    r = body_to_grip_site(quat_matrix(selected["quat_xyzw"]))
                    self.feedback["target"] = {"xyz_m": selected["xyz"], "approach_world": r[:,2].tolist(), "jaw_world": r[:,0].tolist()}
                command, error, hook = run_symmetric_motion_hook(host, poses, quat_matrix(current["quat_xyzw"]), bind_selected=bind_selected)
                selection = deepcopy(hook["orientation_selection"])
                selected = selection.get("selected_target")
                if selected:
                    selected["quat_xyzw"] = matrix_quat(body_to_grip_site(quat_matrix(selected["quat_xyzw"])))
                self.feedback["orientation_selection"] = selection
            else:
                motion_args = {"target_pose": pose}
                # A possibly loaded reorientation keeps the slow carry speed.
                # Its 150-step default can expire while still converging; allow
                # a bounded 300-step horizon without altering IK or tolerances.
                if (self.symmetry_load_uncertain
                        and rotation_distance(current_site, rotation) > .05):
                    motion_args["num_steps"] = 300
                command, error, hook = run_motion_hook(host, motion_args)
            hook.update(motion_feedback(command))
            self.feedback["motion"] = {k: hook[k] for k in ("reason_code", "motion_dispatched", "physics_executed", "motion_summary", "recovery", "failure_stage") if k in hook}
            # A physical contact attempt can move scene geometry even if the
            # endpoint was missed. Retire old contact measurements conservatively.
            if hook.get("motion_dispatched") is not False and hook.get("physics_executed") is not False:
                self.contact = grant
                if grant:
                    self.generation += 1
                    self.symmetry_load_uncertain = True
            return host.result(error=error)
        if name == "gripper_control":
            if args["action"] == "open" and "contact_point_id" in args:
                raise ValueError("Opening does not accept a contact point")
            if "contact_point_id" in args:
                self.contact = self.authorization(args["contact_point_id"], self.state()["grip_xyz_m"])
            command, error = host._execute({"kind": "tool_call", "name": "gripper_control", "parameters": {"position": 1 if args["action"] == "open" else 0}})
            feedback = motion_feedback(command, 'gripper_control')
            if feedback.get('physics_executed') is not False:
                self.generation += 1
                self.contact = None
                # Aperture alone never clears a possible held load after close.
                # Only a successful release plus measured open state clears it.
                self.symmetry_load_uncertain = True
                openness = self.state().get("gripper", {}).get("openness")
                summary = feedback.get('motion_summary') or {}
                if (args['action'] == 'open' and not error
                        and summary.get('stop_reason') == 'gripper_horizon_completed'
                        and isinstance(openness, (int, float)) and not isinstance(openness, bool)
                        and np.isfinite(openness) and openness >= .95):
                    self.symmetry_load_uncertain = False
            self.feedback = {"gripper_command": args["action"], "attachment_proven": False}
            self.feedback['motion'] = feedback
            return host.result(error=error)
        raise ValueError("Unsupported atomic operation")

    def result(self, error=None):
        body = {"episode": self.host.status(), "robot": self.state(),
                "feedback": self.feedback, "views": [], "error": error,
                "point_semantics": "immutable world coordinates; remeasure moved geometry; all positions/offsets in metres"}
        images = []
        for packet, camera, source in self.sources():
            if self.selected_views and camera != self.selected_views and not (self.selected_views == "wrist" and "eye_in_hand" in camera):
                continue
            view = {"source_packet_id": packet, "camera_frame_id": camera,
                    "pixel_origin": "top_left", "is_current_observation": True}
            overlay = dict(self.overlay, actual=body["robot"].get("grip_xyz_m"))
            data = render(source, **overlay, crop=self.crop)
            if self.crop:
                view["crop_to_source"] = {"origin_xy": self.crop[:2], "display_scale": 2,
                    "instruction": "source_xy = crop_pixel_xy / 2 + origin_xy; mark_point uses source pixels"}
            body["views"].append(view)
            images.extend([TextContent(type="text", text=json.dumps(view)), ImageContent(type="image", mimeType="image/png", data=data)])
        # Intentionally project measured proprioception and action outcomes,
        # never context object poses, candidate geometry or internal filenames.
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(body, ensure_ascii=False)), *images], isError=bool(error))
