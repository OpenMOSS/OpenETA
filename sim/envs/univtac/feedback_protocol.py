"""UniVTAC query projection; native evaluation remains host-owned."""
from __future__ import annotations

import copy
import re

PROTOCOL = 'native_eval_no_online_task_feedback_v1'
LEGACY = 'native_eval_online_task_feedback_legacy'
ENDED = 'episode_ended'


def protocol(config):
    return config.get('feedback_protocol', LEGACY)


def offline_feedback(config):
    return protocol(config) == PROTOCOL


def new_run_config(config):
    """Stamp a fresh run only. Existing configuration files are never rewritten."""
    result = copy.deepcopy(config)
    result['feedback_protocol'] = PROTOCOL
    if 'public_task_rules' in result:
        result['public_task_rules'] = re.sub(
            r'A false check_task result alone.*?interaction can continue\.',
            'Use the neutral episode-ended notification to determine whether physical interaction can continue.',
            result['public_task_rules'], flags=re.DOTALL)
    return result


def public_tools(config, tools):
    return tuple(t for t in tools if t != 'check_task' or not offline_feedback(config))


def pick(value, names):
    return {k: copy.deepcopy(value[k]) for k in names.split() if k in value}


POSE = 'xyz_m quat_xyzw approach_world jaw_world'
ROBOT = POSE + ' tcp_frame world_axes position_unit quaternion_order joint_positions_rad gripper_finger_positions_m gripper_target_positions_m gripper_command gripper_joint_names gripper_open_fraction gripper_max_finger_qpos_m tcp_offset_from_hand_m'
COUNTS = 'actual_motion_requests control_steps native_action_count physics_steps simulation_time_seconds simulator_step native_step_limit initialization_control_steps initialization_physics_steps'
EXECUTION = 'requested_gripper reached arm_reached gripper_command gripper_target_positions_m gripper_measured_positions_m gripper_wait_finished segment_finished remaining_position_delta_m remaining_rotation_rad control_steps physical_motion elapsed_seconds'
REASONS = {'control_segment_limit', 'arm_reached', 'arm_reached_and_gripper_wait_finished'}
ERRORS = {'control_segment_not_reached', 'unsupported tool', 'call observe first',
          'call review_demonstrations first', 'execute_preview_id must be used alone',
          'preview is unavailable or expired', 'mark_point unavailable: no aligned sensor depth/calibration',
          'pixel outside the current image', 'clicked pixel has no valid observed depth',
          'gripper must be open or close', 'delta_frame must be world or grip_site',
          'use xyz_m or delta_mm, not both', 'specify a position, orientation, or gripper command',
          'approach_world must be nonzero', 'jaw_world must not be parallel to approach_world'}
ERRORS |= {f'{name} must contain three finite numbers' for name in ('xyz_m', 'delta_mm', 'approach_world', 'jaw_world')}


def error_text(value, ended):
    if value is None:
        return None
    if ended:
        return ENDED
    return value if isinstance(value, str) and value in ERRORS else 'tool_execution_error'


def execution(value, ended):
    if value is None:
        return None
    result = pick(value, EXECUTION)
    for key, fields in [('requested_target', POSE), ('actual', ROBOT), ('counts', COUNTS)]:
        if key in value:
            result[key] = pick(value[key], fields)
    if 'segment_end_reason' in value:
        reason = value['segment_end_reason']
        result['segment_end_reason'] = ENDED if ended else (reason if reason in REASONS else 'segment_finished')
    if 'error' in value:
        result['error'] = error_text(value['error'], ended)
    return result


def image_descriptors(images):
    # Current sensor labels/paths are constructed by capture/history, never evaluator text.
    allowed = {'head RGB', 'wrist RGB', 'left_tactile rgb_marker', 'right_tactile rgb_marker',
               'left_tactile segment-end tactile history', 'right_tactile segment-end tactile history'}
    return [pick(im, 'label path') for im in images if im.get('label') in allowed]


def observation(value, ended):
    result = pick(value, 'observation_id task_instruction')
    for key, fields in [('robot', ROBOT), ('counts', COUNTS),
                        ('remaining_budget', 'move_to tools native_control_steps')]:
        if key in value:
            result[key] = pick(value[key], fields)
    result['terminal'] = ENDED if ended else None
    result['images'] = image_descriptors(value.get('images', []))
    if 'execution_feedback' in value:
        result['execution_feedback'] = execution(value['execution_feedback'], ended)
    if 'mark_point' in value:
        result['mark_point'] = {name: pick(data, 'available depth_unit depth_type valid_pixels')
            for name, data in value['mark_point'].items() if name in ('head', 'wrist')}
        for data in result['mark_point'].values():
            if not data.get('available'):
                data['reason'] = 'aligned sensor depth/calibration unavailable'
    if 'tactile_history' in value:
        history = value['tactile_history']
        result['tactile_history'] = pick(history, 'kind action_id sample_ids simulation_times_s shared_bilateral_timeline online_interrupt')
        if 'request' in history:
            result['tactile_history']['request'] = pick(history['request'], 'xyz_m delta_mm delta_frame approach_world jaw_world gripper preview execute_preview_id')
        if 'selection' in history:
            result['tactile_history']['selection'] = [
                {n: pick(v, 'selection_source tracking_quality tracking_reliable') for n,v in row.items()
                 if n in ('left_tactile', 'right_tactile')} for row in history['selection']]
    return result


def project_query(payload, *, tool, ended):
    """Allowlist query fields; historical demonstration outcome remains untouched."""
    source = payload['text']
    text = pick(source, 'preview_id gripper physics_stepped point_id observation_id xyz_m frame recorded message finished recoverable')
    if 'recoverable' in text and ended:
        text['recoverable'] = False
    text['terminal'] = ENDED if ended else None
    if 'error' in source:
        text['error'] = error_text(source['error'], ended)
    if 'resolved_target' in source:
        text['resolved_target'] = pick(source['resolved_target'], POSE)
    if 'execution' in source:
        text['execution'] = execution(source['execution'], ended)
    if 'observation' in source:
        text['observation'] = observation(source['observation'], ended)
    if tool == 'review_demonstrations':
        text.update(pick(source, 'demonstrations image_labels'))
        images = copy.deepcopy(payload.get('images', []))
    else:
        images = image_descriptors(payload.get('images', []))
    return {'ok': bool(payload['ok']), 'text': text, 'images': images}
