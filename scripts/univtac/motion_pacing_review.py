"""Offline measured-motion comparison and Chinese review videos from retained frames."""
from __future__ import annotations

import argparse
import html
import json
from itertools import pairwise
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sim.envs.univtac.autonomous_operation import matrix_rotvec, quaternion_matrix
from sim.envs.univtac.codex_readonly import read_jsonl
from sim.envs.univtac.tactile_history import VIEWS, encode_recorded_frames
from sim.envs.univtac.trace import write_json

FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'


def text_lines(canvas, lines, start=10, size=20):
    draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype(FONT,size)
    for i,line in enumerate(lines):
        draw.text((10,start+i*(size+7)),line,fill='#172939',font=font)


def panel(root, row, title, multiplier, ended=False):
    canvas=Image.new('RGB',(960,1040),'#f5f7fa')
    state=row['robot']; request=row.get('request',{})
    xyz=lambda a: ', '.join(f'{v*1000:.2f}' for v in a)
    text_lines(canvas,[f"{root.parent.name} · {title}",
        '验证：相同目标，改变参考推进时序后实际运动与接触是否不同',
        f"固定 R1.5 数值目标重放，不是现场 Agent；播放 {multiplier:g}×",
        f"仿真 t={row['simulation_time_seconds']:.4f}s · 动作 {row['action_id']} · control {row['counts']['control_steps']}",
        '目标 xyz(mm): '+xyz(request['xyz_m']) if 'xyz_m' in request else '接管初态 / 无运动请求',
        '实际 xyz(mm): '+xyz(state['xyz_m']),
        f"夹爪 {state['gripper_command']} 目标(mm): {xyz(state['gripper_target_positions_m'])}",
        '实测指开度(mm): '+xyz(state['gripper_finger_positions_m']),
        f"原生终止: {row['native_terminal'] or '未触发'}",
        '此侧已结束，以下为末帧保持' if ended else '本侧仍在执行',
        '只延长已有帧显示，无补造观测；看推进节奏与杆件/夹爪关系'],size=18)
    for i,name in enumerate(VIEWS):
        x,y=(i%2)*480,285+(i//2)*375
        ImageDraw.Draw(canvas).text((x+8,y),name,fill='black',font=ImageFont.truetype(FONT,18))
        with Image.open(root/row['images'][name]) as image:
            canvas.paste(image.convert('RGB').resize((480,350)),(x,y+25))
    return canvas


def metrics(root):
    results=json.loads((root/'replay_results.json').read_text())['rows']
    samples=read_jsonl(root/'samples.jsonl')
    controls=read_jsonl(root/'control_steps.jsonl')
    contact=read_jsonl(root/'replay_host_contact.jsonl')
    outcome=json.loads((root/'episode.json').read_text())
    before=np.array(samples[0]['robot']['xyz_m'])
    actions=[]; last_step=0; last_time=0
    for index,row in enumerate(results,1):
        execution=row['result']['execution']; target=np.array(execution['requested_target']['xyz_m'])
        after=np.array(execution['actual']['xyz_m'])
        group=[c for c in controls if last_step < c['index'] <= last_step+execution['control_steps']]
        duration=execution['counts']['simulation_time_seconds']-last_time
        delta=target-before; distance=np.linalg.norm(delta)
        progress=float(np.dot(after-before,delta)/distance) if distance else 0.0
        positions=[np.array(c['after_tcp']['xyz_m']) for c in group]
        overshoot=max([0.0]+[float(np.dot(p-target,delta)/distance) for p in positions]) if distance else 0.0
        actions.append({'index':index,'source_seq':row['source_seq'],'request':row['request'],
            'arm_reached':execution['arm_reached'],'segment_end_reason':execution['segment_end_reason'],
            'control_steps':execution['control_steps'],'duration_seconds':duration,
            'requested_distance_m':float(distance),'projected_progress_m':progress,
            'remaining_error_m':float(np.linalg.norm(target-after)),
            'remaining_angle_rad':float(np.linalg.norm(execution['remaining_rotation_rad'])),
            'along_request_overshoot_m':overshoot,
            'peak_sampled_speed_m_s':max([0.0]+[c['measured_linear_speed_m_s'] for c in group]),
            'mean_net_speed_m_s':float(np.linalg.norm(after-before)/duration) if duration else 0,
            'peak_sampled_angular_speed_rad_s':max([0.0]+[c['measured_angular_speed_rad_s'] for c in group])})
        before=after;last_step+=execution['control_steps'];last_time=execution['counts']['simulation_time_seconds']
    prefix=[]
    for action in actions:
        if action['request'].get('gripper') is not None: break
        prefix.append(action)
    return {'outcome':outcome,'actions':actions,'before_first_gripper_change':prefix,
            'prefix_maximum_inhand_z_change_m':max(c['inhand_z_change_m'] for c in contact if c['counts']['control_steps'] <= sum(a['control_steps'] for a in prefix)),
            'completed_targets':sum(a['arm_reached'] for a in actions),
            'attempted_targets':len(actions),'samples':len(samples),
            'initial_host':contact[0],'final_host':contact[-1],
            'maximum_inhand_z_change_m':max(c['inhand_z_change_m'] for c in contact),
            'peak_sampled_speed_m_s':max([0.0]+[c['measured_linear_speed_m_s'] for c in controls]),
            'sample_intervals_seconds':np.diff([s['simulation_time_seconds'] for s in samples]).tolist()}


def render_pair(pair, comparison):
    roots=[pair/'original',pair/'paced_candidate']
    streams=[read_jsonl(r/'samples.jsonl') for r in roots]
    # Union of actual times; each side holds its latest causal frame. No length normalization.
    timeline=sorted({round(r['simulation_time_seconds'],9) for rows in streams for r in rows})
    videos=[]
    for rate in (1.0,.05):
        suffix='1x' if rate==1 else '0.05x'
        folder=pair/f'comparison_frames_{suffix}';folder.mkdir(exist_ok=True)
        files=[];mapping=[]
        for i,t in enumerate(timeline):
            canvas=Image.new('RGB',(1920,1040),'white');selected=[]
            for side,(root,rows,title) in enumerate(zip(roots,streams,('左：original 原版','右：paced_candidate 仅改变参考时序'),strict=True)):
                row=max((r for r in rows if r['simulation_time_seconds']<=t+1e-8),key=lambda r:r['simulation_time_seconds'])
                ended=t>=rows[-1]['simulation_time_seconds']-1e-8
                canvas.paste(panel(root,row,title,rate,ended),(side*960,0));selected.append(row['sample_id'])
            file=folder/f'{i:06d}.png';canvas.save(file);files.append(file)
            mapping.append({'comparison_sim_time':t,'sample_ids':selected})
        durations=[(b-a)/rate for a,b in pairwise(timeline)]+[1.0]
        output=pair/f'comparison_{suffix}.mp4'
        encode_recorded_frames(files,durations,output)
        write_json(output.with_suffix('.json'),{'rate':rate,'frames':mapping,'display_durations_s':durations,
                   'final_display_hold_s':1.0,'interpolated_sensor_frames':0})
        videos.append(str(output.name))
        for root,rows,title in zip(roots,streams,('original 原版','paced_candidate 仅改变参考时序'),strict=True):
            folder=root/f'chinese_frames_{suffix}';folder.mkdir(exist_ok=True)
            files=[]
            for row in rows:
                file=folder/f"{row['sample_id']:06d}.png"
                panel(root,row,title,rate,row is rows[-1]).save(file);files.append(file)
            times=[r['simulation_time_seconds'] for r in rows]
            durations=[(b-a)/rate for a,b in pairwise(times)]+[1.0]
            encode_recorded_frames(files,durations,root/f'review_{suffix}.mp4')
            if rate==.05:
                encode_recorded_frames(files,[1.0]*len(files),root/'review_frame_hold.mp4')
    return videos


def expert_video(root):
    rows=json.loads((root/'sparse_review.json').read_text())['frames'];files=[]
    folder=root/'sparse_video_frames';folder.mkdir(exist_ok=True)
    for i,row in enumerate(rows):
        canvas=Image.new('RGB',(960,960),'white')
        text_lines(canvas,[f'{root.name} · 原生 expert 成功示范',
            '稀疏动作边界，非连续过程录像',
            f"动作 {row['step']} · {row['phase']} · 原生步 {row['native_step']} · t={row['simulation_seconds_since_handoff']:.4f}s",
            '每张停留 2 秒仅供查看，不代表物理时长；没有补造动作内帧',
            '原生 collect / planner 执行，不是 OpenETA 或候选控制器'],size=19)
        for j,image in enumerate(row['vision']+row['touch']):
            x,y=(j%2)*480,160+(j//2)*390
            ImageDraw.Draw(canvas).text((x+8,y),image['label'],fill='black',font=ImageFont.truetype(FONT,18))
            with Image.open(root/image['path']) as im:
                canvas.paste(im.convert('RGB').resize((480,360)),(x,y+25))
        file=folder/f'{i:02d}.png';canvas.save(file);files.append(file)
    return encode_recorded_frames(files,[2.0]*len(files),root/'native_expert_sparse_review.mp4')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args();root=args.root.resolve();pairs=[]
    from scripts.univtac.curate_expert_demonstrations import curate
    repo=Path(__file__).resolve().parents[2]
    for seed in (1000000,1000001):
        destination=root/'expert'/f'seed_{seed}'
        if not destination.exists():
            curate(repo/f'outputs/univtac-isaac51-r12/insert_hole/seed_{seed}/expert',destination,seed)
    frozen=json.loads((root/'commands.json').read_text())
    for seed in (1000003,1000004,1000005):
        pair=root/f'seed_{seed}'
        data={v:metrics(pair/v) for v in ('original','paced_candidate')}
        for variant in ('original','paced_candidate'):
            data[variant]['planned_targets']=len(frozen[str(seed)])
        left,right=[data[v]['initial_host'] for v in ('original','paced_candidate')]
        data['initial_difference']={
            'tcp_position_m':(np.array(right['robot']['xyz_m'])-left['robot']['xyz_m']).tolist(),
            'tcp_rotation_angle_rad':float(np.linalg.norm(matrix_rotvec(quaternion_matrix(right['robot']['quat_xyzw']) @ quaternion_matrix(left['robot']['quat_xyzw']).T))),
            'robot_joint_positions_rad':(np.array(right['robot']['joint_positions_rad'])-left['robot']['joint_positions_rad']).tolist(),
            'measured_finger_positions_m':(np.array(right['robot']['gripper_finger_positions_m'])-left['robot']['gripper_finger_positions_m']).tolist(),
            'finger_targets_m':(np.array(right['robot']['gripper_target_positions_m'])-left['robot']['gripper_target_positions_m']).tolist(),
            'prism_in_gripper_native':(np.array(right['prism_in_gripper_native'])-left['prism_in_gripper_native']).tolist()}
        data['seed']=seed
        a,b=data['original'],data['paced_candidate']
        data['commentary']=[
            f'这是 Insert Hole seed {seed} 的固定目标诊断重放，两组使用预先保存的同一组世界目标和夹爪命令，不是现场 Agent。',
            '左侧原版，右侧候选只改变中间参考的时间分配；均按相同仿真时间倍率播放，请先看首次开闭之前的共同前缀。',
            f'实测结果：原版到位 {a["completed_targets"]}/{a["attempted_targets"]}、{a["outcome"]["control_steps"]} 步、{a["outcome"]["termination"]}；候选到位 {b["completed_targets"]}/{b["attempted_targets"]}、{b["outcome"]["control_steps"]} 步、{b["outcome"]["termination"]}。',
            f'采样区间峰值 TCP 速率为 {a["peak_sampled_speed_m_s"]:.3f} / {b["peak_sampled_speed_m_s"]:.3f} m/s；native success 为 {a["outcome"]["task_success"]} / {b["outcome"]["task_success"]}。较短侧结束后显示末帧保持，不能把播放时长误当额外物理。',
            '这些画面不能单独证明滑移或稳定抓握，也不能把未完成目标时较小的触觉变化当作改善；结合初态差异和逐动作进度看结论。']
        (pair/'watch_notes_zh.md').write_text('\n\n'.join(data['commentary'])+'\n')
        write_json(pair/'comparison_metrics.json',data)
        data['videos']=render_pair(pair,data)
        pairs.append(data)
    write_json(root/'comparison_summary.json',{'pairs':pairs,'kind':'fixed_target_diagnostic_not_Agent_or_ICL'})
    for expert in sorted((root/'expert').glob('seed_*')):
        expert_video(expert)
    page=['<html><meta charset="utf-8"><title>R1.6 控制时序对照与 expert 示例</title><style>body{font:18px system-ui;margin:24px;max-width:1600px}video{width:100%;max-height:800px}article{padding:16px;border:1px solid #bbb;margin:20px 0}pre{white-space:pre-wrap}</style>',
        '<a href="/">实验导航</a><h1>R1.6：相同目标，不同参考推进节奏</h1><p>固定 R1.5 记录重放，无现场模型决策。左 original，右 paced_candidate；只改变按控制时间推进的中间参考，默认仍 original。</p>',
        '<p>重点看同一动作的实际运动、杆件与夹爪关系及结束位置。两侧按接管后仿真时间对齐，相同倍率；短侧结束后明确保持末帧。触觉变化不是滑移真值。</p>']
    def link(relative):
        return '/artifact?run='+root.name+'&path='+str(relative)
    for data in pairs:
        seed=data['seed'];pair=Path(f'seed_{seed}')
        page += [f'<article><h2>seed {seed} · 先看并排 0.05×，再用 1×核对真实节奏</h2>',
                 f'<video controls preload="metadata" src="{link(pair/"comparison_0.05x.mp4")}"></video>',
                 f'<p><a href="{link(pair/"comparison_1x.mp4")}">并排 1×</a> · <a href="{link(pair/"comparison_0.05x.mp4")}">并排 0.05×</a></p>']
        for variant in ('original','paced_candidate'):
            m=data[variant];o=m['outcome']
            page.append(f'<p>{variant}：目标到位 {m["completed_targets"]}，已请求 {m["attempted_targets"]} / 预定 {m["planned_targets"]}，control {o["control_steps"]}，physics {o["physics_steps"]}，仿真 {o["simulation_time_seconds"]:.4f}s，native success={o["task_success"]}，结束 {o["termination"]}。'
                + ' · '.join(f'<a href="{link(pair/variant/name)}">{label}</a>' for name,label in [('review_1x.mp4','单独1×'),('review_0.05x.mp4','单独慢放'),('review_frame_hold.mp4','逐帧停留')])+'</p>')
        page += [''.join('<p>'+html.escape(line)+'</p>' for line in data['commentary']),
                 f'<details><summary>逐动作速度、到位误差、初态差异与 host-only 接触记录</summary><pre>{html.escape(json.dumps(data,ensure_ascii=False,indent=2))}</pre></details></article>']
    page.append('<h2>原生 expert 成功示范（单独类别）</h2><p>复用 R1.2 原生 play_once 成功；只有三个动作段前后稀疏帧，不是连续录像。不是本轮固定重放成功，不证明 OpenETA 后端与 expert 等价。</p>')
    for seed in (1000000,1000001):
        path=Path('expert')/f'seed_{seed}'
        page.append(f'<h3>support seed {seed}</h3><video controls src="{link(path/"native_expert_sparse_review.mp4")}"></video><p><a href="{link(path/"manifest.json")}">来源和缺失字段</a></p>')
    (root/'review.html').write_text('\n'.join(page)+'</html>')


if __name__=='__main__':
    main()
