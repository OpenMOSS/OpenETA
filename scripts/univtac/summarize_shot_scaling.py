"""Descriptive shot curves with explicit missing states and paired comparisons."""
from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path

import numpy as np


def wilson(successes, n):
    if not n:
        return None
    z=1.959963984540054;p=successes/n;denom=1+z*z/n
    center=(p+z*z/(2*n))/denom
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return [max(0.,center-half),min(1.,center+half)]


def paired(left,right):
    keys=sorted(set(left)&set(right))
    if not keys:
        return {'valid_pairs':0,'difference_pp':None,'bootstrap_95_pp':None}
    differences=np.array([int(right[k])-int(left[k]) for k in keys])
    rng=np.random.default_rng(20260908)
    means=differences[rng.integers(0,len(keys),size=(2000,len(keys)))].mean(axis=1)*100
    return {'valid_pairs':len(keys),'difference_pp':float(differences.mean()*100),
        'bootstrap_95_pp':np.quantile(means,[.025,.975]).tolist(),
        'right_only_success':int((differences==1).sum()),'left_only_success':int((differences==-1).sum()),
        'both_success':sum(bool(left[k]) and bool(right[k]) for k in keys),
        'both_failure':sum(not left[k] and not right[k] for k in keys),
        'interval_note':'Paired empirical bootstrap over evaluable pairs only; degenerate samples can yield zero-width intervals.'}


def statistics(cells):
    groups=[];comparisons=[];outcomes={}
    tasks=sorted({c['task'] for c in cells})
    for task in tasks:
        for condition in ('B','C'):
            for shot in (1,2,4):
                subset=[c for c in cells if c['task']==task and c['comparison_condition']==condition and c['shot']==shot]
                evaluated=[c for c in subset if c.get('episode',{}).get('evaluable') and not c.get('delivery_or_runner_error')]
                success=sum(c['episode']['task_success'] is True for c in evaluated)
                states={}
                for c in subset:
                    status='delivery_issue' if c.get('delivery_or_runner_error') else c['status']
                    states[status]=states.get(status,0)+1
                outcomes[task,condition,shot]={c['seed']:c['episode']['task_success'] for c in evaluated}
                groups.append({'task':task,'condition':condition,'shot':shot,'planned':len(subset),
                    'evaluable':len(evaluated),'success':success,
                    'unavailable':sum(c['status']=='initialization_unavailable' for c in subset),
                    'states':states,'evaluable_rate':success/len(evaluated) if evaluated else None,
                    'wilson_95_evaluable':wilson(success,len(evaluated)),
                    'success_per_planned':success/len(subset) if subset else None})
        for shot in (1,2,4):
            comparisons.append({'task':task,'comparison':f'C-B_{shot}shot',**paired(outcomes[task,'B',shot],outcomes[task,'C',shot])})
        for condition in ('B','C'):
            for lo,hi in ((1,2),(2,4)):
                comparisons.append({'task':task,'comparison':f'{condition}_{hi}-{lo}shot',**paired(outcomes[task,condition,lo],outcomes[task,condition,hi])})
    averages=[]
    for condition in ('B','C'):
        for shot in (1,2,4):
            rows=[g for g in groups if g['condition']==condition and g['shot']==shot]
            rates=[g['evaluable_rate'] for g in rows]
            averages.append({'condition':condition,'shot':shot,'tasks':len(rows),
                'equal_task_evaluable_rate':sum(rates)/len(rates) if rates and all(x is not None for x in rates) else None,
                'all_planned_resolved':all(not any(g['states'].get(x,0) for x in ('not_run','in_progress')) for g in rows)})
    return {'groups':groups,'paired':comparisons,'equal_task_average':averages,
        'note':'No 0-shot. Wilson intervals condition on evaluability; pending and unavailable are not native failures. Reused main-table cells are not independent of shot selection.'}


def artifact(path):
    parts=Path(path).resolve().parts
    index=parts.index('outputs')
    return '/artifact?run='+parts[index+1]+'&path='+'/'.join(parts[index+2:])


def episode_page(folder,cell):
    def escape(x):return html.escape(str(x))
    p='<meta charset="utf-8"><style>body{max-width:1100px;margin:25px auto;font:18px/1.5 sans-serif}video{width:100%}pre{white-space:pre-wrap}.images{display:flex;flex-wrap:wrap}.images img{width:300px}</style>'
    p+=f'<h1>{cell["task"]} · {cell["comparison_condition"]} · {cell["shot"]}-shot · {cell["seed"]}</h1>'
    p+='<p>GPT-6 low自主操作；当前视觉与触觉均保留。后台判分仅供用户审阅，不曾交付给Agent。视频仅慢放已有帧。</p>'
    p+=f'<video controls src="{artifact(folder/"review.mp4")}"></video>'
    p+='<details><summary>实际示范、Agent Saw和工具请求/反馈</summary>'
    for line in (folder/'operator_context.jsonl').read_text().splitlines():
        c=json.loads(line)
        p+='<h3>'+escape(c['tool'])+'</h3><pre>'+escape(json.dumps(c['arguments'],ensure_ascii=False))+'\n'+escape('\n'.join(c['response_text_blocks']))+'</pre><div class="images">'
        p+=''.join(f'<img loading="lazy" src="{artifact(folder/path)}">' for path in c['response_image_paths'])+'</div>'
    p+='</details><details><summary>Host结果、模型回答与用量</summary>'
    for name in ('episode.json','agent_final.md','codex_trace_summary.json','media_check.json','protocol_delivery_check.json'):
        if (folder/name).exists():
            p+='<h3>'+name+'</h3><pre>'+escape((folder/name).read_text())+'</pre>'
    p+='</details>'
    (folder/'review.html').write_text(p)


def report(root, *, plots=False):
    data=json.loads((root/'results.json').read_text());stats=statistics(data['cells'])
    (root/'statistics.json').write_text(json.dumps(stats,indent=2))
    page='<meta charset="utf-8"><style>body{max-width:1300px;margin:25px auto;font:17px/1.5 sans-serif}td,th{padding:6px;border:1px solid #ddd}table{border-collapse:collapse}img{max-width:100%}</style><h1>四任务 B/C · 1/2/4-shot</h1><p>2400个计划格；共用seed1000000–1000099。缺失不算native失败；已完成格在恢复后保留。仅历史示范数量和触觉投影变化，当前视觉/触觉均完整。</p>'
    page+='<p>完整矩阵未结束前不选择K。四任务参与shot选择，未来引用不是独立测试。旧0-shot不拼入曲线。</p><table><tr><th>任务</th><th>条件</th><th>shot</th><th>planned/evaluable/success/unavailable</th><th>状态</th><th>可评价成功率及95% Wilson区间</th></tr>'
    for g in stats['groups']:
        page+=f'<tr><td>{g["task"]}</td><td>{g["condition"]}</td><td>{g["shot"]}</td><td>{g["planned"]}/{g["evaluable"]}/{g["success"]}/{g["unavailable"]}</td><td>{g["states"]}</td><td>{g["evaluable_rate"]} {g["wilson_95_evaluable"]}</td></tr>'
    page+='</table>'
    if plots:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        for task in sorted({g['task'] for g in stats['groups']}):
            fig,ax=plt.subplots(figsize=(8,5))
            for condition in ('B','C'):
                gs=[g for g in stats['groups'] if g['task']==task and g['condition']==condition]
                gs=[g for g in gs if g['evaluable_rate'] is not None]
                x=[g['shot'] for g in gs];y=[g['evaluable_rate']*100 for g in gs]
                err=[[y[i]-g['wilson_95_evaluable'][0]*100 for i,g in enumerate(gs)],
                     [g['wilson_95_evaluable'][1]*100-y[i] for i,g in enumerate(gs)]]
                ax.errorbar(x,y,yerr=err,label=condition,marker='o',capsize=4)
                for i,g in enumerate(gs):ax.annotate(f'P/E/S/U={g["planned"]}/{g["evaluable"]}/{g["success"]}/{g["unavailable"]}',(x[i],y[i]),xytext=(0,12 if condition=='C' else -25),textcoords='offset points',fontsize=7)
            ax.set(title=task,xlabel='Official expert episodes (shots)',ylabel='Success among evaluable episodes (%)',xticks=[1,2,4],ylim=(-5,105));ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(root/f'{task}_shots.png',dpi=140);plt.close(fig)
    for task in sorted({g['task'] for g in stats['groups']}):
        if (root/f'{task}_shots.png').exists():page+=f'<img src="{artifact(root/f"{task}_shots.png")}">'
    page+=f'<p><a href="{artifact(root/"statistics.json")}">同seed配对差值、有效配对数与等权平均</a></p><h2>全部计划格</h2><table><tr><th>任务/seed/条件</th><th>状态</th><th>Host结果</th><th>审阅</th></tr>'
    for c in data['cells']:
        e=c.get('episode',{});link=''
        if c.get('episode_path'):
            folder=Path(c['episode_path']);target=folder/('review.html' if (folder/'review.html').exists() else 'episode.json');link=f'<a href="{artifact(target)}">输入、动作、结果与慢放</a>'
        page+=f'<tr><td>{c["task"]}/{c["seed"]}/{c["condition"]}</td><td>{c["status"]}</td><td>{e.get("task_success","unavailable")}</td><td>{link}</td></tr>'
    page+='</table>'
    (root/'report.html').write_text(page)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);parser.add_argument('--plots',action='store_true');args=parser.parse_args();report(args.root.resolve(),plots=args.plots)
