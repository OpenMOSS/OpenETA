"""Offline official-demo previews and clearly labelled autonomous review videos."""
from __future__ import annotations

import argparse
import html
import json
from itertools import pairwise
from pathlib import Path
from urllib.parse import urlencode

import yaml
from PIL import Image, ImageDraw, ImageFont

from sim.envs.univtac.codex_readonly import read_jsonl
from sim.envs.univtac.tactile_history import VIEWS, encode_recorded_frames
from sim.envs.univtac.trace import write_json

FONT = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
NAMES = {'no_demo':'A 无示范', 'visual_action_icl':'B 视觉—运动示范', 'tactile_action_icl':'C 视觉＋触觉—运动示范'}
PAGE = '<!doctype html><meta charset="utf-8"><style>body{font:18px system-ui;max-width:1250px;margin:30px auto;background:#f4f6f8}section{background:white;padding:20px;margin:20px 0}img,video{max-width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#145e9d}</style><a href="/r17-autonomous">← R1.7 实际输入与操作回放</a>'

RULE_NAMES = {r+c: f"{r} {'官方说明＋公开规则' if r=='R' else '官方说明'} · {NAMES[d]}"
              for r in 'UR' for c,d in zip('ABC', NAMES)}


def round_names(root):
    manifest = root/'batch/run_manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()).get('round') == 'R1.8':
        return RULE_NAMES
    return NAMES


def link(run, path):
    return '/artifact?'+urlencode({'run':str(run),'path':str(path)})


def video(url):
    return f'<video controls preload="metadata" src="{html.escape(url)}"></video>'


def describe_request(row):
    request=row['request']
    if row['action_id']=='takeover':
        return '官方初始化结束；尚无本轮 Agent 动作'
    if 'delta_mm' in request:
        move=f"位移 {request['delta_mm']} mm（{request.get('delta_frame','world')}）"
    elif 'xyz_m' in request:
        move=f"目标位置 {[round(v,5) for v in request['xyz_m']]} m"
    elif request.get('execute_preview_id'):
        move='执行先前预览的目标'
    elif 'approach_world' in request or 'jaw_world' in request:
        move='调整工具朝向'
    else:
        move='保持手臂位置'
    grip={'open':'打开夹爪','close':'闭合夹爪'}.get(request.get('gripper'),'保持夹爪命令')
    return f"{row['action_id']}：{move}；{grip}"


def demos(root, package=None, task_instruction=None):
    package = package or root/'demonstrations'
    if not (package/'run_manifest.json').exists():
        write_json(package/'run_manifest.json',{'round':'R1.7 historical expert','source':'official data; not autonomous query'})
    page = PAGE.replace('r17-', 'r18-' if task_instruction else 'r17-').replace('R1.7', 'R1.8' if task_instruction else 'R1.7')+'<h1>历史 expert 示范：B/C 实际输入</h1><p>官方 Isaac51 Insert Hole episode 0/1，metadata 均为 success。视频是历史采集，不计入自主成功率。步号来自 HDF5，时间按公开 collect 配置名义 120 Hz 换算；原始命令未记录，示范使用实测运动。</p>'
    run = package.relative_to(root.parent)
    for episode in (0,1):
        folder = package/'historical_expert'/f'episode_{episode}'
        record = json.loads((folder/'recording.json').read_text())
        out = folder/'review_frames_labelled';out.mkdir(exist_ok=True)
        files=[]
        for i,step in enumerate(record['recorded_steps']):
            path=out/f'{i:04d}.png'
            if not path.exists():
                canvas=Image.new('RGB',(960,880),'white');draw=ImageDraw.Draw(canvas)
                lines=[f'历史原生 expert 成功示范 · episode {episode} · 1× 名义仿真时间',
                       f'row {i} | step {step} | t={record["nominal_times_s"][i]:.3f}s | atom {record["atom_ids"][i]} {record["atom_tags"][i]}',
                       '不是当前 Agent 操作；原始采集状态，不生成中间帧。最终 success 来自官方 metadata。']
                for n,line in enumerate(lines):draw.text((10,6+n*28),line,font=ImageFont.truetype(FONT,20),fill='black')
                for k,name in enumerate(VIEWS):
                    with Image.open(folder/f'{i:04d}'/f'{name}.png') as im:
                        canvas.paste(im.convert('RGB').resize((480,360)),((k%2)*480,120+(k//2)*380))
                    draw.text(((k%2)*480+8,96+(k//2)*380),name,fill='black')
                canvas.save(path)
            files.append(path)
        movie=folder/'official_expert_labelled.mp4'
        if not movie.exists():
            durations=[b-a for a,b in zip(record['nominal_times_s'],record['nominal_times_s'][1:])]+[1/60]
            encode_recorded_frames(files,durations,movie)
        page+=f'<section><h2>历史 expert {episode}</h2>'+video(link(run,movie.relative_to(package)))+'</section>'
    for condition in ('visual_action_icl','tactile_action_icl'):
        projection=json.loads((package/f'{condition}.json').read_text())
        if task_instruction:
            projection['text']['task_goal'] = task_instruction
        page+=f'<section><h2>{NAMES[condition]}</h2><p>B/C 共同视觉和运动内容完全相同；C 下半部额外显示历史双侧触觉。标签中的 row/step 对应真实 HDF5 行。</p>'
        page+='<details><summary>完整模型示范文字</summary><pre>'+html.escape(json.dumps(projection['text'],ensure_ascii=False,indent=2))+'</pre></details>'
        for im in projection['images']:
            page+=f'<figure><figcaption>{html.escape(im["label"])}</figcaption><a href="{link(run,im["path"])}"><img loading="lazy" src="{link(run,im["path"])}"></a></figure>'
        page+='</section>'
    (root/'demonstrations.html').write_text(page)


def episodes(root, rebuild=False):
    names=round_names(root)
    page=PAGE.replace('r17-', 'r18-' if names is RULE_NAMES else 'r17-').replace('R1.7', 'R1.8' if names is RULE_NAMES else 'R1.7')+f'<h1>{len(names)*3} 格自主操作视频</h1><p>所有条件均由现场 Codex 自己决定动作。1× 按仿真时间；慢放重复显示已有帧，不生成观测。模型思考时物理暂停。逐帧及真实 MCP 输入见上方回放入口。</p>'
    for path in sorted((root/'batch').glob('seed_*/*/episode.json')):
        folder=path.parent;state=json.loads(path.read_text());condition=folder.name
        title=f'seed {state["seed"]} · {names[condition]}'
        model_label = 'GPT-6 / low · ' if names is RULE_NAMES else ''
        outcome=state.get('task_success') if state.get('native_success_available') else 'unavailable/pending'
        page+=f'<section><h2>{model_label}{title}</h2><p>Agent 事先看到：{names[condition]}。最终 native outcome={outcome}；结束原因={html.escape(str(state.get("termination","进行中")))}。</p>'
        if state['status'] not in ('completed','infrastructure_issue') or not (folder/'review_frames').exists():
            page+='视频等待本格结束。</section>';continue
        rows=read_jsonl(folder/'samples.jsonl');files=[]
        actions=list({r['action_id']:r for r in rows if r['action_id']!='takeover'}.values())
        page+='<p>本条观察任务信息与历史示范如何影响自主操作。请按动作顺序看：'+html.escape('；'.join(describe_request(r) for r in actions))+'。原生结果以上述记录为准，画面中的接触或动作到位不等于成功。</p>'
        for row in rows:
            dest=folder/'review_labelled'/f'{row["sample_id"]:06d}.png'
            if rebuild or not dest.exists():
                source=Image.open(folder/'review_frames'/f'{row["sample_id"]:06d}.png')
                canvas=Image.new('RGB',(source.width,source.height+120),'white');canvas.paste(source,(0,120));draw=ImageDraw.Draw(canvas)
                for n,line in enumerate([f'seed {state["seed"]} · '+model_label+'现场 Codex 自主操作', names[condition], describe_request(row), f'最终 native={outcome}；注意每帧 terminal 标记，任务结束后无物理推进。']):
                    draw.text((10,4+28*n),line,font=ImageFont.truetype(FONT,20),fill='black')
                dest.parent.mkdir(exist_ok=True);canvas.save(dest)
            files.append(dest)
        times=[r['simulation_time_seconds'] for r in rows]
        durations=[max(.001,b-a) for a,b in pairwise(times)]+[1/60]
        run=folder.relative_to(root.parent)
        for rate in (1, .05 if times[-1]-times[0] < .5 else .1):
            movie=folder/f'autonomous_{rate:g}x.mp4'
            if rebuild or not movie.exists():
                labelled=[]
                for file in files:
                    im=Image.open(file).copy();draw=ImageDraw.Draw(im)
                    draw.rectangle((720,0,960,30),fill='white')
                    draw.text((730,3),f'{rate:g}× 仿真时间',font=ImageFont.truetype(FONT,20),fill='black')
                    frame=folder/f'playback_{rate:g}x'/file.name
                    frame.parent.mkdir(exist_ok=True);im.save(frame);labelled.append(frame)
                encode_recorded_frames(labelled,[d/rate for d in durations],movie)
            page+=f'<h3>{rate:g}× 仿真时间（保持已有帧；原始过程 {times[-1]-times[0]:.3f} 秒）</h3>'+video(link(run,movie.name))
        write_json(folder/'review_video.json',{'path':'autonomous_1x.mp4','clock':'simulation','frame_count':len(rows),'synthetic_observations':0})
        page+='</section>'
    (root/'videos.html').write_text(page)


def summarize(root):
    cells=[]
    names=round_names(root)
    for seed in (1000003,1000004,1000005):
        for condition in names:
            folder=root/'batch'/f'seed_{seed}'/condition
            def read(name,folder=folder):
                p=folder/name
                return json.loads(p.read_text()) if p.exists() else {}
            state=read('episode.json');life=read('codex_lifecycle.json')
            context=read_jsonl(folder/'operator_context.jsonl')
            cells.append({'seed':seed,'condition':condition,'status':state.get('status','not_run'),
                **{k:state.get(k) for k in ('model','reasoning_effort','codex_cli_version','evaluable','task_success','termination','infrastructure_error',
                    'move_request_count','actual_motion_requests','tool_call_count','control_steps','physics_steps','simulation_time_seconds')},
                'codex_wall_seconds':life.get('elapsed_seconds'),
                'worker_wall_seconds':read('worker_lifecycle.json').get('elapsed_seconds'),
                'model_exit':life.get('exit_mode'),'usage':read('codex_trace_summary.json').get('usage'),
                'demonstration_image_counts':[len(x['response_image_paths']) for x in context if x['tool']=='review_demonstrations'],
                'video_1x':str((folder/'autonomous_1x.mp4').relative_to(root)),
                'video_slow':str((folder/('autonomous_0.05x.mp4' if (folder/'autonomous_0.05x.mp4').exists() else 'autonomous_0.1x.mp4')).relative_to(root))})
    groups={c:{'successes':sum(x['evaluable'] is True and x['task_success'] is True for x in cells if x['condition']==c),
               'evaluable':sum(x['evaluable'] is True for x in cells if x['condition']==c),'planned':3} for c in names}
    contrasts = {}
    if names is RULE_NAMES:
        for a,b in [('RA','UA'),('RB','UB'),('RC','UC'),('RB','RA'),('RC','RA'),('RC','RB'),('UC','UB')]:
            contrasts[a+'-'+b] = (groups[a]['successes']-groups[b]['successes'])*100/3 if groups[a]['evaluable']==groups[b]['evaluable']==3 else None
    write_json(root/'results.json',{'cells':cells,'groups':groups,'contrasts_percentage_points':contrasts,
        'interpretation':'Three-seed development pilot; ICL comparisons share original controller and current tactile history.'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--demos',action='store_true')
    parser.add_argument('--demonstrations',type=Path,help='Reuse an existing official demonstration package')
    parser.add_argument('--config',type=Path,help='Task instruction used for this round delivery')
    parser.add_argument('--rebuild',action='store_true',help='Rebuild derived query videos from retained frames; no simulator access')
    args=parser.parse_args()
    if args.demos:
        instruction=yaml.safe_load(args.config.read_text()).get('task_instruction') if args.config else None
        demos(args.root.resolve(),args.demonstrations.resolve() if args.demonstrations else None,instruction)
    episodes(args.root.resolve(),args.rebuild)
    summarize(args.root.resolve())
