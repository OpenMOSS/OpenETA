"""Live observation/command session; all task truth stays behind evaluation."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sim.envs.univtac.autonomous_operation import (
    TOOLS,
    NativeController,
    append_row,
    pose_dict,
    quaternion_matrix,
    resolve_target,
)
from sim.envs.univtac.observation import capture_snapshot, to_numpy
from sim.envs.univtac.planner_diagnostics import safe_diagnostic_value
from sim.envs.univtac.tactile_history import TactileRecorder
from sim.envs.univtac.trace import write_json
from tools.pointcloud_pose_marking import camera_ray_from_image_click


def project_current_observation(payload, *, current_tactile=True):
    """Remove only current touch at delivery; preserve host capture and demos."""
    if current_tactile or 'observation' not in payload['text']:
        return payload
    observation = dict(payload['text']['observation'])
    observation.pop('tactile_history', None)
    observation['images'] = [im for im in observation['images']
                             if im['label'] in ('head RGB', 'wrist RGB')]
    return {**payload, 'text': {**payload['text'], 'observation': observation},
            'images': observation['images']}


class AutonomousSession:
    def __init__(self, task, config, root: Path, seed: int):
        self.task, self.config, self.root, self.seed = task, config, root, seed
        self.controller = NativeController(task, config, root)
        self.tool_count = self.move_requests = self.observation_index = 0
        self.finished = False
        self.finish_reason = self.infrastructure_error = None
        self.demonstrations_reviewed = False
        self.latest = None
        self.frames = []
        self.feedback = None
        self.previews = {}
        self.started = time.monotonic()
        self.recorder = None
        if 'tactile_history' in config:
            self.recorder = TactileRecorder(task, self.controller, root, config['tactile_history'])
            self.recorder.sample()
            self.controller.after_step = self.recorder.sample
        if config.get('replay_diagnostics'):
            self.record_replay_diagnostics()
            def after_step():
                if self.recorder:
                    self.recorder.sample()
                self.record_replay_diagnostics()
            self.controller.after_step = after_step
        self.terminal_saved = False

    def record_replay_diagnostics(self):
        # Read-only host evidence; never feeds target resolution or IK.
        prism = self.task.prism.get_pose()
        relative = prism.rebase(self.task._robot_manager.get_gripper_center_pose())
        append_row(self.root/'replay_host_contact.jsonl', {
            'counts': self.controller.counts(), 'robot': self.controller.state(),
            'prism_pose_native': safe_diagnostic_value(prism.tolist()),
            'prism_in_gripper_native': safe_diagnostic_value(relative.tolist()),
            'inhand_z_change_m': float(abs(self.task.origin_inhand_pose[2] - relative[2])),
            'native_terminal': self.controller.terminal()})

    def termination(self):
        return self.controller.terminal() or ("move_request_limit" if self.move_requests >= self.config['max_move_requests'] else None) or ("tool_call_limit" if self.tool_count >= self.config['max_tool_calls'] else None)

    def capture(self):
        self.observation_index += 1
        folder = self.root/'observations'/f'{self.observation_index:04d}'
        obs = self.task._get_observations()
        snap = capture_snapshot(obs, output_root=self.root, seed_dir=folder, task_name=self.config.get('task', 'insert_hole'), seed=self.seed,
            phase='pre_action', action_id=f'obs_{self.observation_index}', simulator_step=int(self.task.step_count),
            take_action_count=int(self.task.take_action_cnt), task_instruction=self.config.get('task_instruction', 'Insert the held object into the hole.'),
            task_metadata={}, native_check_success=None, save_host_only=True,
            strict_two_tactile_sensors=True, fail_on_missing_rgb_marker=True)
        write_json(folder/'snapshot.json',snap.snapshot.to_dict())
        visible = snap.snapshot.operator_visible
        images = []
        for name in ('head','wrist'):
            images.append({'label':name+' RGB', 'path':visible['cameras'][name]['rgb']['path']})
        for name in ('left_tactile','right_tactile'):
            images.append({'label':name+' rgb_marker', 'path':visible['tactile'][name]['rgb_marker']['path']})
        frames, availability = [], {}
        for name in ('head','wrist'):
            try:
                if name == 'wrist':
                    raise ValueError('live wrist extrinsics unavailable: sensor pose cache does not track articulation')
                cam = self.task._camera_manager.cameras[name]
                # TiledCamera defaults to a cached initialization pose. Refresh
                # sensor extrinsics without rendering or advancing simulation.
                cam._update_poses(cam._ALL_INDICES)
                depth = to_numpy(cam.data.output['depth'][0]).squeeze()
                k = to_numpy(cam.data.intrinsic_matrices[0])
                q = to_numpy(cam.data.quat_w_ros[0])
                pos = to_numpy(cam.data.pos_w[0])
                rgb = to_numpy(obs['observation'][name]['rgb'])
                if depth.shape != rgb.shape[:2] or not np.isfinite(k).all() or k[0,0] <= 0 or k[1,1] <= 0:
                    raise ValueError('depth/calibration does not align with RGB')
                valid = np.isfinite(depth) & (depth > 0) & (depth < 65)
                if not valid.any():
                    raise ValueError('no valid sensor depth')
                depth_path = folder/f'{name}_depth.png'
                Image.fromarray(np.where(valid,np.clip(depth*1000,0,65535),0).astype(np.uint16)).save(depth_path)
                extrinsics = {'pos':pos.tolist(), 'mat':quaternion_matrix(q[[1,2,3,0]]).reshape(-1).tolist(), 'camera_frame':'opencv'}
                frame = {'camera_id':name, 'rgb_path':images[0 if name=='head' else 1]['path'], 'depth_path':str(depth_path.relative_to(self.root)),
                         'metadata':{'intrinsics':{'fx':float(k[0,0]),'fy':float(k[1,1]),'cx':float(k[0,2]),'cy':float(k[1,2]),
                         'width':depth.shape[1],'height':depth.shape[0],'scale':1000}, 'extrinsics':extrinsics}}
                frames.append(frame)
                availability[name] = {'available':True, 'depth_unit':'metre', 'depth_type':'camera_z', 'valid_pixels':int(valid.sum())}
            except (KeyError, AttributeError, ValueError) as exc:
                availability[name] = {'available':False,'reason':str(exc)}
        self.frames = frames
        write_json(folder/'geometry.json',{'frames':frames,'availability':availability})
        self.latest = {'observation_id':f'obs_{self.observation_index}', 'task_instruction':self.config.get('task_instruction', 'Insert the held object into the hole.'),
                       'robot':self.controller.state(), 'counts':self.controller.counts(),
                       'remaining_budget':{'move_to':max(0,self.config['max_move_requests']-self.move_requests),
                        'tools':max(0,self.config['max_tool_calls']-self.tool_count),
                        'native_control_steps':max(0,self.task.cfg.step_lim-self.task.take_action_cnt)},
                       'mark_point':availability, 'execution_feedback':self.feedback, 'terminal':self.termination(), 'images':images}
        write_json(folder/'operator_observation.json',self.latest)
        return self.latest

    def call(self, tool: str, args: dict[str,Any]):
        if self.tool_count >= self.config['max_tool_calls']:
            return {'ok':False,'text':{'error':'tool_call_limit','terminal':True},'images':[]}
        args = {k:v for k,v in args.items() if v is not None}
        self.tool_count += 1
        before = self.latest
        result = {}
        try:
            allowed = (*TOOLS, 'review_demonstrations') if self.config.get('demonstrations') else TOOLS
            if tool not in allowed:
                raise ValueError('unsupported tool')
            if self.config.get('demonstrations') and not self.demonstrations_reviewed and tool != 'review_demonstrations':
                raise ValueError('call review_demonstrations first')
            if tool == 'review_demonstrations':
                if self.demonstrations_reviewed:
                    result = {'demonstrations': 'already delivered; refer to previous tool result', 'demonstration_images': []}
                else:
                    projection = json.loads((self.root/'demonstrations/projection.json').read_text())
                    result = {'demonstrations': projection['text'], 'demonstration_images': projection['images']}
                    self.demonstrations_reviewed = True
            elif tool == 'observe':
                result = {'observation':self.capture()}
            elif self.latest is None:
                raise ValueError('call observe first')
            elif tool == 'move_to':
                preview = bool(args.get('preview',False))
                if not preview:
                    if self.termination() or self.finished:
                        raise ValueError(self.termination() or 'episode_finished')
                    self.move_requests += 1
                preview_id = args.get('execute_preview_id')
                if preview_id:
                    if any(v is not None and v is not False for k,v in args.items() if k not in ('execute_preview_id','delta_frame')):
                        raise ValueError('execute_preview_id must be used alone')
                    if preview_id not in self.previews:
                        raise ValueError('preview is unavailable or expired')
                    target, gripper = self.previews[preview_id]
                else:
                    target = resolve_target(self.controller.tcp(), args)
                    gripper = args.get('gripper')
                if preview:
                    preview_id = f'preview_{self.tool_count}'
                    self.previews[preview_id] = (target, gripper)
                    result = {'preview_id':preview_id,'resolved_target':pose_dict(target),'gripper':gripper,'physics_stepped':False}
                else:
                    self.previews.clear()
                    if self.recorder:
                        self.recorder.begin(f'action_{self.move_requests:03d}', args)
                    self.feedback = self.controller.execute(target, gripper)
                    observation = self.capture()
                    if self.recorder:
                        images, history = self.recorder.history()
                        observation['images'] = observation['images'][:2] + images
                        observation['tactile_history'] = history
                        write_json(self.root/'observations'/f'{self.observation_index:04d}'/'operator_observation.json', observation)
                    result = {'execution':self.feedback,'observation':observation}
            elif tool == 'mark_point':
                name = args['view']
                frame = next((f for f in self.frames if f['camera_id']==name),None)
                if frame is None:
                    raise ValueError('mark_point unavailable: no aligned sensor depth/calibration')
                intr = frame['metadata']['intrinsics']
                if not 0 <= args['u'] < intr['width'] or not 0 <= args['v'] < intr['height']:
                    raise ValueError('pixel outside the current image')
                ray = camera_ray_from_image_click({'frames':self.frames},camera_id=name,u=args['u'],v=args['v'],artifact_root=self.root)
                xyz = ray.get('visible_surface', {}).get('xyz_m')
                if xyz is None:
                    raise ValueError('clicked pixel has no valid observed depth')
                result = {'point_id':args.get('point_id','P0'),'observation_id':self.latest['observation_id'],'xyz_m':xyz,'frame':'world'}
            elif tool == 'check_task':
                result = self.controller.check()
            elif tool == 'report_issue':
                result = {'recorded':True, 'message':str(args.get('message',''))}
            elif tool == 'finish_episode':
                result = self.controller.check()
                self.finished = True
                self.finish_reason = ('fixed_replay_' + str(args.get('reason', 'finish'))) if self.config.get('replay_diagnostics') else 'agent_finish'
                result['finished'] = True
            result['terminal'] = self.termination()
            image_descriptors = result.pop('demonstration_images', result.get('observation',{}).get('images',[]))
            if tool == 'review_demonstrations':
                result['image_labels'] = [x['label'] for x in image_descriptors]
            payload = {'ok':True,'text':result,'images':image_descriptors}
        except ValueError as exc:
            result = {'error':str(exc),'recoverable':not bool(self.termination()),'terminal':self.termination()}
            if tool=='move_to' and not args.get('preview',False) and (not self.config.get('demonstrations') or self.demonstrations_reviewed):
                self.previews.clear()
                self.feedback = {'error':str(exc),'reached':False,'physical_motion':False}
                result['observation'] = self.capture()
            payload = {'ok':False,'text':result,'images':result.get('observation',{}).get('images',[])}
        except Exception as exc:  # noqa: BLE001 -- retain runtime failure evidence
            self.infrastructure_error = f'{type(exc).__name__}: {exc}'
            self.finish_reason = 'infrastructure_error'
            payload = {'ok':False,'text':{'error':self.infrastructure_error,'terminal':'infrastructure_error'},'images':[]}
        payload = project_current_observation(payload, current_tactile=self.config.get('current_tactile', True))
        append_row(self.root/'tool_trace.jsonl',{'tool':tool,'arguments':args,'before':before,'result':payload,'counts':self.controller.counts(),'timestamp_s':time.time()})
        if self.termination() or self.infrastructure_error:
            write_json(self.root/'stop.json',{'reason':self.termination() or self.infrastructure_error,'timestamp_s':time.time()})
            if not self.terminal_saved:
                self.finalize()
                self.terminal_saved = True
        return payload

    def finalize(self, reason=None):
        checker = ({'available':False, 'success':None} if self.config.get('observe_only') else self.controller.check())
        result = {'round':self.config.get('round','R1.4'),'seed':self.seed,'reset_valid':True,
                  'native_success_available':checker['available'], 'task_success':checker['success'],
                  'native_early_stop':self.controller.early_stop,'termination':reason or self.termination() or self.finish_reason or ('unscored_observation_check' if self.config.get('observe_only') else 'codex_exit'),
                  'infrastructure_error':self.infrastructure_error, 'tool_call_count':self.tool_count,
                  'move_request_count':self.move_requests,'elapsed_seconds':time.monotonic()-self.started, **self.controller.counts()}
        if self.recorder:
            result['recording'] = self.recorder.save_index()
        if self.controller.grasp:
            result['inherited_gripper'] = self.controller.grasp.inherited
            result['final_robot'] = self.controller.state()
        write_json(self.root/'final_result.json',result)
        host = {'metadata':safe_diagnostic_value(self.task.metadata),'plan_success':bool(self.task.plan_success),'eval_success':bool(self.task.eval_success)}
        if self.config.get('task') == 'grasp_classify':
            # Post-run subgroup evidence only; never used in observations or targets.
            host['object_class'] = str(self.task.choice)
        write_json(self.root/'host_evaluator.json', host)
        return result
