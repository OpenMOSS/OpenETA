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


NEW_SEED_NAMES = {k:v for k,v in RULE_NAMES.items() if k.startswith('R')}


def round_manifest(root):
    path = root/'batch/run_manifest.json'
    return json.loads(path.read_text()) if path.exists() else {}


def round_names(root):
    name = round_manifest(root).get('round')
    return {'R1.8':RULE_NAMES, 'R1.9':NEW_SEED_NAMES}.get(name, NAMES)


def round_seeds(root):
    m = round_manifest(root)
    return sorted({seed for seed,_ in m['order']}) if m.get('order') else [1000003,1000004,1000005]


def round_page(root, fallback='R1.7'):
    name = round_manifest(root).get('round', fallback)
    route = {'R1.7':'r17', 'R1.8':'r18', 'R1.9':'r19'}[name]
    return PAGE.replace('R1.7', name).replace('r17-', route+'-')


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
    page = round_page(root, 'R1.8' if task_instruction else 'R1.7')+'<h1>历史 expert 示范：B/C 实际输入</h1><p>官方 Isaac51 Insert Hole episode 0/1，metadata 均为 success。视频是历史采集，不计入自主成功率。步号来自 HDF5，时间按公开 collect 配置名义 120 Hz 换算；原始命令未记录，示范使用实测运动。</p>'
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
    page=round_page(root)+f'<h1>{len(names)*len(round_seeds(root))} 格自主操作视频</h1><p>所有条件均由现场 Codex 自己决定动作。1× 按仿真时间；慢放重复显示已有帧，不生成观测。模型思考时物理暂停。逐帧及真实 MCP 输入见上方回放入口。</p>'
    for path in sorted((root/'batch').glob('seed_*/*/episode.json')):
        folder=path.parent;state=json.loads(path.read_text());condition=folder.name
        title=f'seed {state["seed"]} · {names[condition]}'
        model_label = 'GPT-6 / low · ' if names is not NAMES else ''
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
    for seed in round_seeds(root):
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
               'evaluable':sum(x['evaluable'] is True for x in cells if x['condition']==c),'native_failures':sum(x['evaluable'] is True and x['task_success'] is False for x in cells if x['condition']==c),
               'infrastructure_issues':sum(bool(x['infrastructure_error']) for x in cells if x['condition']==c),
               'planned':len(round_seeds(root))} for c in names}
    contrasts = {}
    if names is RULE_NAMES:
        for a,b in [('RA','UA'),('RB','UB'),('RC','UC'),('RB','RA'),('RC','RA'),('RC','RB'),('UC','UB')]:
            contrasts[a+'-'+b] = (groups[a]['successes']-groups[b]['successes'])*100/3 if groups[a]['evaluable']==groups[b]['evaluable']==3 else None
    pairs = {}
    if names is NEW_SEED_NAMES:
        for a,b in [('RC','RB'),('RC','RA'),('RB','RA')]:
            contrasts[a+'-'+b] = (groups[a]['successes']-groups[b]['successes'])*100/groups[a]['planned'] if groups[a]['evaluable']==groups[b]['evaluable']==groups[a]['planned'] else None
        for b in ('RB','RA'):
            pairs['RC-'+b] = paired_outcomes(cells, 'RC', b)
    write_json(root/'results.json',{'cells':cells,'groups':groups,'contrasts_percentage_points':contrasts,'paired_outcomes':pairs,
        'interpretation':'Project new-seed validation, not an official test split; R1.8 remains separate.' if names is NEW_SEED_NAMES else 'Three-seed development pilot; ICL comparisons share original controller and current tactile history.'})


def paired_timeline(streams):
    """Reuse R1.6's causal frame-hold alignment on actual simulation times."""
    timeline = sorted({round(r['simulation_time_seconds'],9) for rows in streams for r in rows})
    mapping = []
    for t in timeline:
        selected = [max((r for r in rows if r['simulation_time_seconds']<=t+1e-8),
                        key=lambda r:r['simulation_time_seconds']) for rows in streams]
        mapping.append({'simulation_time_seconds':t,'sample_ids':[r['sample_id'] for r in selected],
                        'ended':[t>=rows[-1]['simulation_time_seconds']-1e-8 for rows in streams]})
    return mapping


def review_pairs(root):
    page = round_page(root)+'<h1>RB / RC 同 seed 并排审阅</h1><p>左：视觉—运动示范；右：同一示范增加历史触觉。两侧都是现场 GPT-6 low 自主操作，动作可以不同。按接管后的真实仿真时间对齐，同倍率播放；短侧结束后保持末帧，不拉伸归一化运动时长。历史 expert 不是这里的当前操作。</p>'
    for seed in round_seeds(root):
        pair = root/'batch'/f'seed_{seed}'
        folders = [pair/'RB', pair/'RC']
        if not all((f/'review_video.json').exists() for f in folders):
            page+=f'<section><h2>{seed}</h2><p>等待 RB/RC 两侧完成。</p></section>'
            continue
        streams = [read_jsonl(f/'samples.jsonl') for f in folders]
        mapping = paired_timeline(streams)
        states = [json.loads((f/'episode.json').read_text()) for f in folders]
        page+=f'<section><h2>seed {seed}</h2><p>RB native={states[0].get("task_success")}；RC native={states[1].get("task_success")}。请比较动作方向、姿态、开合与结果；图片变化本身不是滑移或稳定抓取标签。</p>'
        for rate in (1,.05):
            output = pair/f'paired_rb_rc_{rate:g}x.mp4'
            if not output.exists():
                frames=[]
                for i,m in enumerate(mapping):
                    panels=[]
                    for j,(folder,sid,ended) in enumerate(zip(folders,m['sample_ids'],m['ended'],strict=True)):
                        with Image.open(folder/'review_labelled'/f'{sid:06d}.png') as im:
                            panel=Image.new('RGB',(im.width,im.height+62),'white');panel.paste(im,(0,62))
                        draw=ImageDraw.Draw(panel);font=ImageFont.truetype(FONT,20)
                        draw.text((10,3),f'{"左 RB" if j==0 else "右 RC"} · {rate:g}× · 对齐仿真时间 {m["simulation_time_seconds"]:.3f}s',font=font,fill='black')
                        draw.text((10,31),'此侧已结束，以下为末帧保持' if ended else '该侧实际记录；未生成中间观测',font=font,fill='black')
                        panels.append(panel)
                    canvas=Image.new('RGB',(panels[0].width*2,panels[0].height),'white')
                    for j,panel in enumerate(panels):canvas.paste(panel,(j*panel.width,0))
                    frame=pair/f'paired_frames_{rate:g}x'/f'{i:06d}.png';frame.parent.mkdir(exist_ok=True);canvas.save(frame);frames.append(frame)
                times=[m['simulation_time_seconds'] for m in mapping]
                durations=[(b-a)/rate for a,b in pairwise(times)]+[1.0]
                encode_recorded_frames(frames,durations,output)
                write_json(output.with_suffix('.json'),{'rate':rate,'frames':mapping,'final_display_hold_seconds':1,'synthetic_observations':0})
            run=Path('univtac-isaac51-r19/batch')
            page+=f'<h3>{rate:g}×（结尾另停留1秒供审阅）</h3>'+video(link(run,output.relative_to(root/'batch')))
        page+='<p>原始逐帧、完整请求与示范：'+''.join(f'<a href="/r19-autonomous">{c} 实际输入与轨迹</a> ' for c in ('RB','RC'))+'</p></section>'
    (root/'pairs.html').write_text(page)


def paired_outcomes(cells, method, control):
    result = {'method_only_success': 0, 'control_only_success': 0, 'both_success': 0, 'both_failure': 0, 'unavailable': 0}
    by_key = {(c['seed'],c['condition']):c for c in cells}
    for seed in sorted({c['seed'] for c in cells}):
        a,b = by_key[(seed,method)],by_key[(seed,control)]
        if not a['evaluable'] or not b['evaluable']:
            key = 'unavailable'
        elif a['task_success'] and b['task_success']:
            key = 'both_success'
        elif a['task_success']:
            key = 'method_only_success'
        elif b['task_success']:
            key = 'control_only_success'
        else:
            key = 'both_failure'
        result[key] += 1
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--demos',action='store_true')
    parser.add_argument('--demonstrations',type=Path,help='Reuse an existing official demonstration package')
    parser.add_argument('--config',type=Path,help='Task instruction used for this round delivery')
    parser.add_argument('--pairs',action='store_true',help='Render same-seed RB/RC comparisons without simulator access')
    parser.add_argument('--rebuild',action='store_true',help='Rebuild derived query videos from retained frames; no simulator access')
    args=parser.parse_args()
    if args.demos:
        instruction=yaml.safe_load(args.config.read_text()).get('task_instruction') if args.config else None
        demos(args.root.resolve(),args.demonstrations.resolve() if args.demonstrations else None,instruction)
    episodes(args.root.resolve(),args.rebuild)
    summarize(args.root.resolve())
    if args.pairs:review_pairs(args.root.resolve())
