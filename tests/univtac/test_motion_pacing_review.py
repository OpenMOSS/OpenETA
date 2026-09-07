import json

import imageio_ffmpeg
from PIL import Image

from scripts.univtac.motion_pacing_review import panel
from sim.envs.univtac.tactile_history import VIEWS, encode_recorded_frames


def test_slow_playback_retains_source_frames_and_duration(tmp_path):
    files=[]
    for i,color in enumerate(['red','green','blue']):
        p=tmp_path/f'{i}.png';Image.new('RGB',(32,32),color).save(p);files.append(p)
    video=encode_recorded_frames(files,[.2,.4,.5],tmp_path/'slow.mp4')
    reader=imageio_ffmpeg.read_frames(str(video),pix_fmt='rgb24',output_params=['-vsync','0']);meta=next(reader)
    frames=list(reader)
    assert len(frames)==4  # three originals and a final display anchor
    assert 1.09 <= meta['duration'] <= 1.13
    record=json.loads(video.with_suffix('.playback.json').read_text())
    assert record['synthetic_sensor_observations']==0 and record['source_frame_count']==3


def test_chinese_panel_keeps_all_four_views_below_annotations(tmp_path):
    for n in VIEWS:
        Image.new('RGB',(320,240),'#d04020').save(tmp_path/f'{n}.png')
    row={'robot':{'xyz_m':[.5,0,.3],'gripper_command':'inherited_hold',
                 'gripper_target_positions_m':[.005,.006],'gripper_finger_positions_m':[.005,.006]},
         'counts':{'control_steps':1},'simulation_time_seconds':1/60,'action_id':'action_001',
         'native_terminal':'native_early_stop','images':{n:f'{n}.png' for n in VIEWS}}
    image=panel(tmp_path,row,'原版',.05,True)
    assert image.size==(960,1040)
    for x,y in [(240,500),(720,500),(240,900),(720,900)]:
        assert image.getpixel((x,y))==(208,64,32)


def test_existing_artifact_route_accepts_comparison_batch_root(tmp_path):
    from scripts.univtac.serve_experiment_dashboard import _safe_artifact, _safe_run_root
    root=tmp_path/'univtac-isaac51-r16';root.mkdir()
    (root/'run_manifest.json').write_text('{"round":"R1.6"}')
    pair=root/'seed_1000003';pair.mkdir()
    video=pair/'comparison_1x.mp4';video.write_bytes(b'video')
    resolved=_safe_run_root(tmp_path,root.name)
    assert _safe_artifact(resolved,'seed_1000003/comparison_1x.mp4')==video
