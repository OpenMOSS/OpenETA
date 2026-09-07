"""Read two official Isaac51 files and build matched visual/tactile projections."""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

from sim.envs.univtac.autonomous_operation import matrix_rotvec, quaternion_matrix
from sim.envs.univtac.tactile_history import VIEWS, motion_features, select_times
from sim.envs.univtac.trace import write_json

REPO=Path(__file__).resolve().parents[2]


def segment_bounds(atom_ids, tags):
    starts=[i for i,t in enumerate(tags) if t==b'move' and (i==0 or atom_ids[i]!=atom_ids[i-1])]
    ends=[i-1 for i in starts[1:]]+[len(tags)-1]
    return [(0 if k==0 else ends[k-1],end) for k,end in enumerate(ends)]


def strip(frames, selected, steps, path, label):
    w,h=frames[0].size
    canvas=Image.new('RGB',(w*len(selected),h+30),'white');draw=ImageDraw.Draw(canvas)
    for i,index in enumerate(selected):
        canvas.paste(frames[index],(i*w,30))
        draw.text((i*w+5,8),f'{label} | row {index} | step {int(steps[index])}',fill='black')
    path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)


def prepare(raw: Path, out: Path):
    import h5py
    out.mkdir(parents=True,exist_ok=False)
    spec=importlib.util.spec_from_file_location('native_hdf_reader',
        '/home/ubuntu/wybcode/.worktrees/univtac-isaac51-r081/envs/utils/data.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    metadata=json.loads((raw/'metadata.json').read_text())
    cfg=yaml.safe_load((REPO/'configs/univtac/autonomous_insert_hole.yaml').read_text())['tactile_history']
    base={'task_goal':'Insert the held object into the hole.', 'examples':[],
          'interpretation':'Historical measured expert motion, not recorded move_to commands. Do not copy historical absolute coordinates into the current scene.',
          'robot_convention':'EE is measured panda_hand pose relative to robot base: metres then quaternion wxyz; not OpenETA world gripper-center TCP. Joint vector: seven arm radians then two finger positions in metres.',
          'time_convention':'Frame step values are recorded. Seconds below are nominal step differences / 120 from published collect timing, not an embedded per-frame clock.',
          'outcome_convention':'Success is the official per-episode metadata result; no checker rerun.',
          'unrecorded':'Original actuator/tool commands and force/grasp-state labels are unavailable.'}
    vision_images=[];touch_images=[];touch_content=[];provenance=[]
    for episode in (0,1):
        info=metadata[str(episode)]
        if info['result']!='success' or info.get('source_seed',info['seed']) in (1000003,1000004,1000005):
            raise ValueError('Fixed official demonstration selection is not eligible')
        with h5py.File(raw/f'{episode}.hdf5') as f:
            ee=f['embodiment/ee'][()];joint=f['embodiment/joint'][()];steps=f['step'][()]
            ids=f['atom/id'][()];tags=f['atom/tag'][()]
            fields={};f.visititems(lambda name,obj,fields=fields:fields.update({name:{'shape':list(obj.shape),'dtype':str(obj.dtype)}}) if isinstance(obj,h5py.Dataset) else None)
            keys={n:f'observation/{n}/rgb' for n in VIEWS[:2]}
            keys.update({n:f'tactile/{n}/rgb_marker' for n in VIEWS[2:]})
            pixels={n:module.HDF5Handler.stream_to_img(f[key][()]) for n,key in keys.items()}
            # Native producer encoded its RGB numeric array with cv2.imencode;
            # native imdecode restores those original numbers. Do not swap again.
        if any(len(x)!=len(steps) for x in [ee,joint,ids,*pixels.values()]):
            raise ValueError('Image/state lengths do not align')
        frames={n:[Image.fromarray(x) for x in arrays] for n,arrays in pixels.items()}
        recorded=[]
        for i,t in enumerate(steps):
            row={'sample_id':i,'step':int(t),'simulation_time_seconds':float((t-steps[0])/120),'features':{}}
            if i:
                row['features']={n:motion_features(pixels[n][i-1],pixels[n][i],cfg) for n in VIEWS[2:]}
            recorded.append(row)
            for n in VIEWS:
                path=out/'historical_expert'/f'episode_{episode}'/f'{i:04d}'/f'{n}.png'
                path.parent.mkdir(parents=True,exist_ok=True);frames[n][i].save(path)
        example={'example_id':f'official_episode_{episode}', 'outcome':'success', 'segments':[]}
        for k,(a,b) in enumerate(segment_bounds(ids,tags),1):
            label=f'example {episode} segment {k}'
            def state(i,ee=ee,joint=joint,steps=steps):return {'ee_xyz_wxyz':ee[i].tolist(),'joint':joint[i].tolist(),'recorded_step':int(steps[i])}
            ra=quaternion_matrix(ee[a,3:][[1,2,3,0]]);rb=quaternion_matrix(ee[b,3:][[1,2,3,0]])
            segment={'segment':k,'before':state(a),'after':state(b),
                'measured_motion':{'translation_in_base_mm':((ee[b,:3]-ee[a,:3])*1000).tolist(),
                    'rotation_vector_in_base_rad':matrix_rotvec(rb@ra.T).tolist(),
                    'finger_position_change_m':(joint[b,-2:]-joint[a,-2:]).tolist(),
                    'nominal_duration_s':float((steps[b]-steps[a])/120)},
                'feedback':'Observed state transition; original execution feedback is not separately recorded.',
                'vision_labels':[]}
            for n in VIEWS[:2]:
                name=f'{label} {n} before/after';path=Path('images')/f'e{episode}_s{k}_{n}.png'
                strip(frames[n],[a,b],steps,out/path,n)
                vision_images.append({'label':name,'path':str(path)})
                segment['vision_labels'].append(name)
            selected=[r['sample_id'] for r in select_times(recorded[a:b+1])]
            touch={'example_id':example['example_id'],'segment':k,'recorded_steps':[int(steps[i]) for i in selected],
                   'selection':'segment-end image/motion change, not slip labels','image_labels':[]}
            for n in VIEWS[2:]:
                name=f'{label} {n} history';path=Path('images')/f'e{episode}_s{k}_{n}.png'
                strip(frames[n],selected,steps,out/path,n)
                touch_images.append({'label':name,'path':str(path)});touch['image_labels'].append(name)
            touch_content.append(touch);example['segments'].append(segment)
        base['examples'].append(example)
        write_json(out/'historical_expert'/f'episode_{episode}'/'recording.json',{
            'recorded_steps':steps.tolist(),'nominal_times_s':((steps-steps[0])/120).tolist(),
            'atom_ids':ids.tolist(),'atom_tags':[x.decode() for x in tags]})
        provenance.append({'episode_id':episode,'published_seed':info['seed'],'source_seed':info.get('source_seed'),
            'metadata_result':info['result'],'file':str((raw/f'{episode}.hdf5').resolve()),'sample_count':len(steps),
            'step_range':[int(steps[0]),int(steps[-1])],'step_deltas':np.unique(np.diff(steps)).tolist(),
            'attrs':{},'fields':fields,'has_raw_action_commands':False,
            'segments':segment_bounds(ids,tags), 'native_outcome_recomputed':False})
    write_json(out/'no_demo.json',{'text':{'examples':[],'message':'No historical demonstrations are provided.'},'images':[]})
    write_json(out/'visual_action_icl.json',{'text':base,'images':vision_images})
    full=copy.deepcopy(base);full['historical_touch']=touch_content
    write_json(out/'tactile_action_icl.json',{'text':full,'images':vision_images+touch_images})
    write_json(out/'provenance.json',{'official_dataset':'byml2024/UniVTAC isaac51/insert_hole',
        'selection':'Numeric episode ID order: 0,1; both metadata successes; no task-specific split manifest found.',
        'producer_commit':'not recorded in release metadata',
        'documentation_revision':'d541e5568227ca3b66104d294f63c80acad7c52c; documentation only, runtime stays pinned',
        'time_basis':'Observed step gap 2; collect/physical 120 Hz in official documentation; no timestamp attributes in HDF5',
        'episodes':provenance,'selection_config':cfg})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();prepare(args.raw.resolve(),args.output.resolve())
