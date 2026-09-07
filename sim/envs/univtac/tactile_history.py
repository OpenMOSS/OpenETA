"""Observed-image segment history and offline review video; no physics calls."""
from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from sim.envs.univtac.autonomous_operation import append_row
from sim.envs.univtac.observation import to_numpy
from sim.envs.univtac.trace import write_json

VIEWS = ('head', 'wrist', 'left_tactile', 'right_tactile')


def rgb_array(value):
    a = to_numpy(value).copy()[..., :3]
    if a.dtype != np.uint8:
        a = np.clip(a * (255 if a.max() <= 1.01 else 1), 0, 255).astype(np.uint8)
    return a


def motion_features(previous, current, cfg):
    """Small integer patch search; all features use pixels, never native flow."""
    width = cfg['tracking_width']
    height = round(width * current.shape[0] / current.shape[1])
    a, b = [np.asarray(Image.fromarray(x).convert('L').resize((width, height)), dtype=float)
            for x in (previous, current)]
    x0, y0, x1, y1 = cfg['roi_fraction']
    x0, x1, y0, y1 = int(x0*width), int(x1*width), int(y0*height), int(y1*height)
    diff = float(np.mean(np.abs(b[y0:y1, x0:x1]-a[y0:y1, x0:x1]))/255)
    radius, search = cfg['patch_radius'], cfg['search_radius']
    margin = radius + search
    vectors, locations = [], []
    for y in np.linspace(max(y0, margin), min(y1, height-margin-1), 3).astype(int):
        for x in np.linspace(max(x0, margin), min(x1, width-margin-1), 4).astype(int):
            patch = a[y-radius:y+radius+1, x-radius:x+radius+1]
            if patch.std() < cfg['minimum_patch_std']:
                continue
            costs = []
            for dy in range(-search, search+1):
                for dx in range(-search, search+1):
                    q = b[y+dy-radius:y+dy+radius+1, x+dx-radius:x+dx+radius+1]
                    costs.append((float(np.mean((patch-q)**2)), dx, dy))
            costs.sort()
            best, second = costs[:2]
            if best[0] > cfg['maximum_patch_mse'] or second[0]-best[0] < cfg['minimum_match_margin']:
                continue
            scale = current.shape[1]/width
            vectors.append([best[1]*scale, best[2]*scale])
            locations.append([x*scale, y*scale])
    quality = len(vectors)/12
    motion = float(np.percentile(np.linalg.norm(vectors, axis=1), 75)) if vectors else 0.0
    reliable = quality >= cfg['minimum_tracking_quality']
    return {'local_displacement_px': vectors, 'patch_centers_px': locations,
            'motion_change_px': motion, 'image_difference': diff,
            'tracking_quality': quality, 'tracking_reliable': reliable,
            'selection_source': 'patch_motion' if reliable else 'image_difference_fallback',
            'change_strength': motion/cfg['motion_threshold_px'] if reliable else diff/cfg['difference_threshold']}


def select_times(rows, mode='motion'):
    """Completed-segment peak only: before, preceding peak, peak, latest."""
    if not rows:
        return []
    def score(row):
        fields = row.get('features', {}).values()
        key = 'change_strength' if mode == 'motion' else 'difference_strength'
        return max((x.get(key, 0) for x in fields), default=0)
    peak = max(range(1, len(rows)), key=lambda i: score(rows[i]), default=0)
    selected = [0, len(rows)-1]
    if peak and score(rows[peak]) >= 1:
        selected += [peak-1, peak]
    return [rows[i] for i in sorted(set(selected))]


class TactileRecorder:
    """Main-thread sensor copies; a single writer only encodes owned arrays."""
    def __init__(self, task, controller, root, config):
        self.task, self.controller, self.root, self.cfg = task, controller, root, config
        self.rows, self.pending = [], []
        self.pool = ThreadPoolExecutor(max_workers=1)
        self.previous = None
        self.action_id, self.request = 'takeover', {}
        self.segment_start = 0
        self.closed = False

    def sample(self):
        cameras = self.task._camera_manager.get_observations(['rgb'])
        tactile = self.task._tactile_manager.get_observations(['rgb_marker'])
        raw_touch = {n: to_numpy(tactile[n]['rgb_marker']).copy() for n in VIEWS[2:]}
        pixels = {n: rgb_array(cameras[n]['rgb']) for n in VIEWS[:2]}
        pixels.update({n: rgb_array(tactile[n]['rgb_marker']) for n in VIEWS[2:]})
        sensors = {n: self.task._camera_manager.cameras[n] for n in VIEWS[:2]}
        sensors.update({n: self.task._tactile_manager.tactiles[n].sensor for n in VIEWS[2:]})
        stamps = {n: {'frame_id': int(to_numpy(s.frame).reshape(-1)[0]),
                      'sensor_time_seconds': float(to_numpy(s._timestamp_last_update).reshape(-1)[0])}
                  for n, s in sensors.items()}
        counts = self.controller.counts()
        row = {'sample_id': len(self.rows), 'action_id': self.action_id, 'request': self.request,
               'simulation_time_seconds': counts['simulation_time_seconds'], 'counts': counts,
               'robot': self.controller.state(), 'sensors': stamps, 'features': {},
               'native_terminal': self.controller.terminal(), 'images': {}, 'raw_touch': {},
               'render_control_step': getattr(self.task, 'last_render', None),
               'render_physics_step': getattr(self.task, '_last_render_physics_step', None)}
        for n in VIEWS:
            old = self.rows[-1]['sensors'][n] if self.rows else None
            stamps[n]['refreshed'] = old is None or stamps[n]['frame_id'] != old['frame_id']
        if self.rows and all(not s['refreshed'] for s in stamps.values()):
            append_row(self.root/'sampling_gaps.jsonl', {'counts': counts, 'reason': 'no_sensor_refresh'})
            return
        for n in VIEWS[2:]:
            if self.previous is not None and stamps[n]['refreshed']:
                f = motion_features(self.previous[n], pixels[n], self.cfg)
                f['difference_strength'] = f['image_difference']/self.cfg['difference_threshold']
                row['features'][n] = f
        for n, a in pixels.items():
            path = self.root/'recording'/f'{len(self.rows):06d}'/f'{n}.png'
            path.parent.mkdir(parents=True, exist_ok=True)
            row['images'][n] = str(path.relative_to(self.root))
            self.pending.append(self.pool.submit(Image.fromarray(a).save, path))
            if n in raw_touch:
                source = raw_touch[n]
                row['raw_touch'][n] = {'dtype':str(source.dtype), 'shape':list(source.shape)}
                if source.dtype != np.uint8:
                    raw_path = path.with_suffix('.npy')
                    self.pending.append(self.pool.submit(np.save,raw_path,source,allow_pickle=False))
                    row['raw_touch'][n]['path'] = str(raw_path.relative_to(self.root))
                else:
                    row['raw_touch'][n]['path'] = str(path.relative_to(self.root))
        self.previous = pixels
        self.rows.append(row)
        append_row(self.root/'samples.jsonl', row)
        if len(self.pending) >= 16:
            self.flush()

    def flush(self):
        for f in self.pending:
            f.result()
        self.pending.clear()

    def begin(self, action_id, request):
        self.action_id, self.request = action_id, dict(request)
        self.segment_start = max(0, len(self.rows)-1)

    def history(self):
        self.flush()
        rows = self.rows[self.segment_start:]
        chosen = select_times(rows)
        comparison = select_times(rows, 'difference')
        output = self.root/'history'/self.action_id
        output.mkdir(parents=True, exist_ok=True)
        images = []
        for n in VIEWS[2:]:
            frames = [Image.open(self.root/r['images'][n]).convert('RGB') for r in chosen]
            w, h = frames[0].size
            strip = Image.new('RGB', (w*len(frames), h+30), 'white')
            draw = ImageDraw.Draw(strip)
            for i, (r, frame) in enumerate(zip(chosen, frames, strict=True)):
                strip.paste(frame, (i*w,30))
                draw.text((i*w+5,5), f"{n} | t={r['simulation_time_seconds']:.4f}s | frame {r['sample_id']}", fill='black')
            p = output/f'{n}.png'
            strip.save(p)
            images.append({'label': n+' segment-end tactile history', 'path': str(p.relative_to(self.root))})
        metadata = {'kind': 'segment-end tactile image/motion change', 'action_id': self.action_id,
                    'request': self.request, 'sample_ids': [r['sample_id'] for r in chosen],
                    'simulation_times_s': [r['simulation_time_seconds'] for r in chosen],
                    'selection': [{n: {k: f[k] for k in ('selection_source', 'tracking_quality', 'tracking_reliable')}
                                   for n, f in r['features'].items()} for r in chosen],
                    'difference_candidate_sample_ids': [r['sample_id'] for r in comparison],
                    'shared_bilateral_timeline': True, 'online_interrupt': False}
        write_json(output/'selection.json', metadata)
        return images, {k:v for k,v in metadata.items() if k != 'difference_candidate_sample_ids'}

    def save_index(self):
        self.flush()
        times = np.array([r['simulation_time_seconds'] for r in self.rows])
        dt = np.diff(times)
        index = {'sample_count': len(self.rows), 'intervals_seconds': dt.tolist(),
                 'sample_index': 'samples.jsonl', 'phase': 'official_reset_return_to_episode_end',
                 'sampling': 'after each existing native qpos control/render; no added physics',
                 'wall_clock_waits': 'physics pauses while Codex thinks; video uses simulation time',
                 'raw_frames': 'recording/', 'terminal': self.controller.terminal()}
        write_json(self.root/'recording.json', index)
        return index

    def close(self):
        if not self.closed:
            self.save_index()
            self.pool.shutdown(wait=True)
            self.closed = True


def encode_recorded_frames(files, durations, video: Path):
    """Display existing frames for specified durations; never synthesize observations."""
    import imageio_ffmpeg
    manifest = video.with_suffix('.ffconcat')
    entries = ['ffconcat version 1.0']
    for file, duration in zip(files, durations, strict=True):
        escaped = str(Path(file).resolve()).replace("'", "'\\''")
        entries += [f"file '{escaped}'", 'option framerate 1000', f'duration {duration:.9f}']
    entries += [f"file '{escaped}'", 'option framerate 1000']
    manifest.write_text('\n'.join(entries)+'\n')
    write_json(video.with_suffix('.playback.json'), {'source_frame_count':len(files),
        'display_durations_seconds':durations, 'final_frame_anchor_repeat':1,
        'synthetic_sensor_observations':0})
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error',
        '-f', 'concat', '-safe', '0', '-i', str(manifest), '-fps_mode', 'vfr',
        '-c:v', 'libx264', '-bf', '0', '-threads', '2', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)],
        check=True, capture_output=True)
    return video


def export_review_video(root: Path):
    """Offline VFR H.264 export; failures leave raw images and an error record."""
    import imageio_ffmpeg
    rows = [json.loads(s) for s in (root/'samples.jsonl').read_text().splitlines() if s.strip()]
    resolved = {}
    if (root/'tool_trace.jsonl').exists():
        for line in (root/'tool_trace.jsonl').read_text().splitlines():
            record = json.loads(line)['result']['text']
            history = record.get('observation', {}).get('tactile_history')
            if history and 'execution' in record:
                resolved[history['action_id']] = record['execution']['requested_target']
    folder = root/'review_frames'
    folder.mkdir(exist_ok=True)
    files = []
    for row in rows:
        canvas = Image.new('RGB', (960,900), '#f2f5f8')
        draw = ImageDraw.Draw(canvas)
        state = row['robot']
        first_xyz = np.asarray(rows[0]['robot']['xyz_m'])
        delta = (np.asarray(state['xyz_m'])-first_xyz)*1000
        request = dict(row['request'])
        if request.get('execute_preview_id') and row['action_id'] in resolved:
            request['resolved_xyz_m'] = [round(v,5) for v in resolved[row['action_id']]['xyz_m']]
        lines = [f"sample {row['sample_id']}  sim {row['simulation_time_seconds']:.4f}s  action {row['action_id']}",
                 'request '+json.dumps(request),
                 f"TCP delta from takeover (mm): {delta.round(3).tolist()}",
                 f"gripper {state['gripper_command']} target(m): {state['gripper_target_positions_m']}",
                 f"measured(m): {state['gripper_finger_positions_m']}  terminal: {row['native_terminal']}",
                 'Physics pauses during model thinking. No interpolated observations.']
        for i, text in enumerate(lines):
            draw.text((10,8+20*i), text, fill='black')
        for i,n in enumerate(VIEWS):
            x,y = (i%2)*480, 140+(i//2)*380
            draw.text((x+8,y+2), n, fill='black')
            with Image.open(root/row['images'][n]) as im:
                canvas.paste(im.convert('RGB').resize((480,360)), (x,y+20))
        file = folder/f"{row['sample_id']:06d}.png"
        canvas.save(file)
        files.append(file)
    manifest = root/'review.ffconcat'
    entries = ['ffconcat version 1.0']
    durations = []
    for i, file in enumerate(files):
        duration = rows[i+1]['simulation_time_seconds']-rows[i]['simulation_time_seconds'] if i+1<len(rows) else 1/60
        duration = max(duration, .001)
        durations.append(duration)
        entries += [f"file '{file.relative_to(root)}'", 'option framerate 1000', f'duration {duration:.9f}']
    manifest.write_text('\n'.join(entries)+'\n')
    video = root/'review.mp4'
    proc = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-loglevel', 'error', '-f', 'concat', '-safe', '0',
                           '-i',str(manifest), '-fps_mode', 'vfr', '-c:v','libx264','-pix_fmt','yuv420p',
                           '-movflags','+faststart',str(video)], capture_output=True, text=True, check=False)
    if proc.returncode:
        raise RuntimeError(proc.stderr)
    write_json(root/'video.json', {'path':'review.mp4','frame_count':len(rows),'durations_seconds':durations,
                                 'clock':'simulation time; millisecond video time base', 'interpolation':False})
    return video
