"""R1.4 replay using exact operator context and separate host execution records."""
from __future__ import annotations

import json
from pathlib import Path

from sim.envs.univtac.codex_readonly import read_jsonl


def load_autonomous_runs(runs_root: Path, round_name: str = 'R1.4') -> dict:
    batches = []
    pattern = {'R1.11':'univtac-isaac51-r111/**/run_manifest.json','R1.10':'univtac-isaac51-r110/**/run_manifest.json','R1.9':'univtac-isaac51-r19/**/run_manifest.json','R1.8':'univtac-isaac51-r18/**/run_manifest.json','R1.7':'univtac-isaac51-r17/**/run_manifest.json','R1.5':'univtac-isaac51-r15*/**/run_manifest.json'}.get(round_name,'univtac-isaac51-r14*/run_manifest.json')
    for manifest_path in sorted(runs_root.glob(pattern)):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('round') != round_name or manifest.get('mode') == 'observe_only':
            continue
        episodes = []
        for path in sorted(manifest_path.parent.glob('seed_*/*/episode.json' if round_name in ('R1.7','R1.8','R1.9','R1.10','R1.11') else 'seed_*/episode.json')):
            root = path.parent
            def read(name, root=root):
                candidate = root/name
                return json.loads(candidate.read_text()) if candidate.exists() else None
            episodes.append({'run':str(root.relative_to(runs_root)), 'episode':read('episode.json'),
                             'prompt':(root/'prompt.txt').read_text() if (root/'prompt.txt').exists() else None,
                             'context':read_jsonl(root/'operator_context.jsonl'),
                             'execution':read_jsonl(root/'tool_trace.jsonl'),
                             'debug':read('debug_controls.json'),
                             'samples':read_jsonl(root/'samples.jsonl'), 'video':read('review_video.json') or read('video.json'),
                             'recording':read('recording.json'), 'gripper_commands':read_jsonl(root/'gripper_commands.jsonl'),
                             'selections':[json.loads(p.read_text()) for p in sorted(root.glob('history/action_*/selection.json'))],
                             'agent_final': (root/'agent_final.md').read_text() if (root/'agent_final.md').exists() else None,
                             'host_evaluator':read('host_evaluator.json'),
                             'usage':read('codex_trace_summary.json'),
                             'worker_error':read('worker_error.json'), 'codex_lifecycle':read('codex_lifecycle.json')})
        note_path=manifest_path.parent/'validation_note.json'
        note=json.loads(note_path.read_text()) if note_path.exists() else None
        batches.append({'name':manifest_path.parent.name,'manifest':manifest,'episodes':episodes,'validation_note':note})
    if round_name in ('R1.7', 'R1.8', 'R1.9', 'R1.10', 'R1.11'):
        conditions = ('UA','UB','UC','RA','RB','RC') if round_name == 'R1.8' else ('no_demo','visual_action_icl','tactile_action_icl')
        if round_name == 'R1.9':
            conditions = ('RA','RB','RC')
        if round_name == 'R1.11':
            conditions = ('A','B','C')
        if round_name == 'R1.10':
            conditions = ('B_live','C_live','B_no_live','C_no_live')
        batches = [{**b, 'name': b['name']+' / '+c, 'manifest':{**b['manifest'],'mode':'batch'},
                    'episodes':[e for e in b['episodes'] if e['run'].endswith('/'+c)]}
                   for b in batches for c in conditions]
    return {'batches':batches}


HTML = r'''<!doctype html><html><head><meta charset="utf-8"><title>R1.4 Autonomous Insert Hole</title>
<style>body{font:16px system-ui;max-width:1400px;margin:24px auto;padding:0 18px;background:#f5f7fa;color:#142434}article,section{background:white;padding:18px;margin:16px 0;border-radius:12px;border:1px solid #dae1e8}h1,h2,h3{line-height:1.3}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#f2f5f8;padding:12px;font-size:13px}.images{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0;flex:1;min-width:180px;max-width:400px}img{width:100%;background:#eee}summary{cursor:pointer;padding:10px;font-weight:600}.host{border-left:5px solid #ad762b}.debug{color:#865a09}a{color:#1463a3}.metric{font-size:20px}</style></head><body>
<a href="/">← 实验导航</a><h1>R1.4 Autonomous Insert Hole</h1>
<p>Codex 从官方 reset 返回后自己决定位置、朝向与夹爪操作。当前视觉＋双侧触觉＋本体状态，无历史示例。通用工具执行，不运行任务 expert。</p>
<p>调试用于验证接口，不计入成功率。正式 episode 的失败和恢复全部保留；unavailable 不计作原生失败。</p><main id="content"></main>
<script>
const esc=x=>String(x??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
const reason=x=>({native_early_stop:'原生提前终止',native_step_limit:'原生步数用尽',agent_finish:'交互结束',native_success:'原生成功'}[x]??x??'进行中');
const pretty=x=>'<pre>'+esc(JSON.stringify(x,null,2))+'</pre>';
const art=(r,p)=>'/artifact?run='+encodeURIComponent(r)+'&path='+encodeURIComponent(p);
function images(run,paths,labels=[]){return '<div class="images">'+paths.map((p,i)=>'<figure><a target="_blank" href="'+art(run,p)+'"><img loading="lazy" src="'+art(run,p)+'"></a><figcaption>'+esc(labels[i]??p)+'</figcaption></figure>').join('')+'</div>'}
function observation(run,o){if(!o)return '';return '<p>'+esc(o.observation_id)+' · step '+esc(o.counts?.simulator_step)+'</p>'+images(run,(o.images||[]).map(x=>x.path),(o.images||[]).map(x=>x.label));}

const reviews=[];
function review(e){if(!e.samples?.length)return '';const k=reviews.push(e)-1;
return '<section class="review host" data-review="'+k+'"><h3>完整审阅录像 · Host-only 标注，不是 Agent 输入</h3><p>按仿真时间回放；模型思考期间物理暂停。无插值生成的观测。下方逐帧滑杆查看原始四图。</p>'+
(e.video?'<video style="width:100%;max-height:650px" controls preload="metadata" src="'+art(e.run,e.video.path)+'"></video>':'<p>视频尚未生成，原始帧仍可审阅。</p>')+
'<details><summary>两种方法的候选窗口（点编号跳到原始帧）</summary>'+e.selections.map(h=>'<p>'+esc(h.action_id)+' · 主方法 '+h.sample_ids.map(i=>'<button class="jump" data-sample="'+i+'">'+i+'</button>').join(' ')+' · 帧差 '+h.difference_candidate_sample_ids.map(i=>'<button class="jump" data-sample="'+i+'">'+i+'</button>').join(' ')+'</p>').join('')+'</details><label>速度 <select class="speed"><option value="1">1×</option><option value="0.25">0.25×</option><option value="0.1">0.1×</option></select></label> <button class="prev">上一帧</button> <button class="next">下一帧</button><input class="frame" type="range" min="0" max="'+(e.samples.length-1)+'" value="0" step="1" style="width:60%"><p class="frameinfo"></p><div class="curves"></div><div class="raw"></div><details><summary>采样与夹爪命令记录</summary>'+pretty({recording:e.recording,commands:e.gripper_commands})+'</details></section>';
}
function plot(rows,index,series,label){const W=900,H=140,ts=rows.map(r=>r.simulation_time_seconds),start=ts[0],span=Math.max(.001,ts.at(-1)-start),values=series.flatMap(s=>rows.map(s.get)),max=Math.max(.001,...values);let svg='<p>'+esc(label)+'</p><svg viewBox="0 0 '+W+' '+H+'" style="width:100%;background:#fafafa">';
for(const s of series){const points=rows.map(r=>((r.simulation_time_seconds-start)/span*(W-40)+20)+','+(H-20-s.get(r)/max*(H-40))).join(' ');svg+='<polyline fill="none" stroke="'+s.color+'" stroke-width="2" points="'+points+'"/>';}
const x=(ts[index]-start)/span*(W-40)+20;return svg+'<line x1="'+x+'" x2="'+x+'" y1="0" y2="'+H+'" stroke="black"/></svg><small>'+series.map(s=>'<span style="color:'+s.color+'">'+esc(s.name)+': '+s.get(rows[index]).toFixed(4)+'</span>').join(' · ')+'</small>';}
function bindReviews(){document.querySelectorAll('.review').forEach(el=>{const e=reviews[+el.dataset.review],rows=e.samples,slider=el.querySelector('.frame'),video=el.querySelector('video');
const show=(i,seek=false)=>{i=Math.max(0,Math.min(rows.length-1,i));slider.value=i;const r=rows[i];el.querySelector('.frameinfo').textContent='frame '+r.sample_id+' · sim '+r.simulation_time_seconds.toFixed(4)+'s · '+r.action_id+' · request '+JSON.stringify(r.request)+' · terminal '+(r.native_terminal??'none');
el.querySelector('.raw').innerHTML=images(e.run,Object.values(r.images),Object.keys(r.images));
const grip=[{name:'joint1 target (m)',color:'#1261a0',get:r=>r.robot.gripper_target_positions_m[0]},{name:'joint1 measured',color:'#79a7d3',get:r=>r.robot.gripper_finger_positions_m[0]},{name:'joint2 target',color:'#b94427',get:r=>r.robot.gripper_target_positions_m[1]},{name:'joint2 measured',color:'#ec9a6a',get:r=>r.robot.gripper_finger_positions_m[1]}];
const change=[];for(const [side,color] of [['left_tactile','#7e46a8'],['right_tactile','#238764']]){change.push({name:side+' motion/fallback',color,get:r=>r.features?.[side]?.change_strength??0});change.push({name:side+' image difference',color:side==='left_tactile'?'#c1a0d8':'#82bea4',get:r=>r.features?.[side]?.difference_strength??0});}
el.querySelector('.curves').innerHTML=plot(rows,i,grip,'夹爪命令与实测开度（米）')+plot(rows,i,change,'触觉变化强度（各自阈值归一化；不是滑移或力）');if(seek&&video)video.currentTime=r.simulation_time_seconds-rows[0].simulation_time_seconds;};
el.querySelectorAll('.jump').forEach(b=>b.onclick=()=>show(+b.dataset.sample,true));slider.oninput=()=>show(+slider.value,true);el.querySelector('.prev').onclick=()=>show(+slider.value-1,true);el.querySelector('.next').onclick=()=>show(+slider.value+1,true);el.querySelector('.speed').onchange=ev=>{if(video)video.playbackRate=+ev.target.value;};
if(video)video.ontimeupdate=()=>{let i=0;while(i+1<rows.length&&rows[i+1].simulation_time_seconds-rows[0].simulation_time_seconds<=video.currentTime)i++;if(+slider.value!==i)show(i);};show(0);});}

async function render(){const d=await (await fetch('/api/r14-autonomous')).json();document.querySelector('#content').innerHTML=d.batches.map(b=>{
const scored=b.manifest.mode==='batch',valid=b.episodes.filter(e=>e.episode.evaluable),wins=valid.filter(e=>e.episode.task_success);
return '<section><h2>'+esc(b.name)+'</h2><p class="'+(scored?'metric':'debug')+'">'+(scored?(valid.length===3?'自主操作开发成功率 '+wins.length+'/3':'正式开发批次：可评价 '+valid.length+'/3；成功 '+wins.length+'；其余状态见下方'):'独立控制联调 · 不计分')+'</p>'+(b.validation_note?'<details class="debug"><summary>不计分联调结论与配置决定</summary>'+pretty(b.validation_note)+'</details>':'')+b.episodes.map(e=>{
const state=e.episode;return '<article><h3>seed '+esc(state.seed)+' · '+esc(state.status)+'</h3><p>Native outcome: '+esc(state.native_success_available?state.task_success:'unavailable / pending')+' · 结束原因 '+esc(reason(state.termination))+'</p>'+pretty({'工具调用':state.tool_call_count,'收到的非预览请求':e.context.filter(x=>x.tool==='move_to'&&!x.arguments.preview).length||state.move_request_count,'通过终止检查的请求':state.move_request_count,'实际运动请求':state.actual_motion_requests,'控制步':state.control_steps,'物理步':state.physics_steps,'仿真秒':state.simulation_time_seconds,'Codex耗时秒':e.codex_lifecycle?.elapsed_seconds,'token用量':e.usage?.usage??'未返回，不能记为0'})+
review(e)+'<h3>Agent Saw → Agent Requested → Environment Executed</h3>'+(e.context.length?e.context.map(row=>{
const execution=e.execution[row.seq-1];return '<details><summary>'+row.seq+' · '+esc(row.tool)+'</summary><h4>Agent Requested</h4>'+pretty(row.arguments)+'<h4>Agent Saw — 实际 MCP 返回图片</h4>'+images(e.run,row.response_image_paths||[])+ '<details><summary>完整 MCP 文本</summary>'+row.response_text_blocks.map(x=>'<pre>'+esc(x)+'</pre>').join('')+'</details>'+'<h4>Environment Executed</h4><p>'+esc(execution?.result?.text?.execution?.reached===true?'机器人到达请求目标（不代表任务成功）':execution?.result?.text?.execution?.error??'见工具反馈')+'</p><details><summary>实际目标、机器人状态与执行反馈</summary>'+pretty(execution?.result?.text?.execution??execution?.result?.text)+'</details>'+'<h4>Before / After Vision and Touch</h4>'+observation(e.run,execution?.before)+observation(e.run,execution?.result?.text?.observation)+'</details>'}).join(''):'<p>无模型上下文：尚未启动 Codex，或此为纯控制联调。</p>')+
(e.debug?'<details><summary>不计分的调试请求、反馈与前后图片</summary>'+pretty(e.debug)+e.execution.filter(x=>x.tool==='move_to'&&!x.arguments.preview).map(x=>observation(e.run,x.before)+observation(e.run,x.result?.text?.observation)).join('')+'</details>':'')+
'<details><summary>Agent Response — 实际过程发言</summary>'+(e.usage?.visible_assistant_messages||[]).map(x=>'<pre>'+esc(x)+'</pre>').join('')+'</details><h4>Agent 自己的最终回答</h4><pre>'+esc(e.agent_final??'尚无自然收尾回答')+'</pre><details class="host"><summary>Host-only：原生评价、内部错误与成本</summary>'+pretty({episode:state,evaluator:e.host_evaluator,worker_error:e.worker_error,usage:e.usage?.usage??'unavailable',codex_lifecycle:e.codex_lifecycle})+'</details></article>'}).join('')+'</section>'}).join('')||'<p>尚无 R1.4 运行记录。</p>'}
render().then(bindReviews);
</script></body></html>'''

R15_HTML = HTML.replace("R1.4", "R1.5").replace("r14-autonomous", "r15-autonomous")

R17_HTML = HTML.replace('R1.4', 'R1.7').replace('r14-autonomous', 'r17-autonomous').replace(
    '当前视觉＋双侧触觉＋本体状态，无历史示例。',
    '三组当前观测相同：视觉＋触觉历史＋本体状态。A 无示范；B 两条官方 expert 的视觉—运动示范；C 相同示范再加历史触觉。').replace(
    '<main id="content">',
    '<p><a href="/r17-demonstrations">先看历史 expert 与 B/C 实际示范预览</a> · <a href="/r17-videos">九条视频与慢放入口</a></p><main id="content">')

R17_SUMMARY = r'''
function primaryTable(d){
const groups=d.batches.map(b=>({name:b.name.split(' / ').at(-1),entries:b.episodes,
valid:b.episodes.filter(e=>e.episode.evaluable),wins:b.episodes.filter(e=>e.episode.evaluable&&e.episode.task_success).length}));
const names={no_demo:'A 无示范',visual_action_icl:'B 视觉—运动示范',tactile_action_icl:'C 增加历史触觉'};
const sum=(g,get)=>g.entries.length===3&&g.entries.every(e=>get(e)!=null)?g.entries.reduce((a,e)=>a+get(e),0).toLocaleString(): 'pending / unavailable';
let table='<section><h2>主结果：原生任务成功率</h2><p>同一 original 控制器、当前视觉＋触觉历史；三个开发 seed，每格一次。较少动作若以失败结束，不算效率提升。</p><table style="width:100%;text-align:left"><tr><th>条件</th><th>Native success</th><th>实际运动 / 控制步</th><th>Codex 秒</th><th>Input / cached / output tokens</th></tr>';
for(const g of groups)table+='<tr><td>'+esc(names[g.name]??g.name)+'</td><td>'+(g.valid.length===3?g.wins+'/3':g.wins+' 成功；可评价 '+g.valid.length+'/3')+'</td><td>'+sum(g,e=>e.episode.actual_motion_requests)+' / '+sum(g,e=>e.episode.control_steps)+'</td><td>'+sum(g,e=>e.codex_lifecycle?.elapsed_seconds)+'</td><td>'+sum(g,e=>e.usage?.usage?.input_tokens)+' / '+sum(g,e=>e.usage?.usage?.cached_input_tokens)+' / '+sum(g,e=>e.usage?.usage?.output_tokens)+'</td></tr>';
table+='</table><p>token 为 CLI 返回的累计原始字段，cached 单列，不与 input 再相加。成本详单、基础设施状态和原始回答在各格下方。</p>';
if(groups.length===3&&groups.every(g=>g.valid.length===3)){const m=Object.fromEntries(groups.map(g=>[g.name,g.wins]));table+='<p>成功数差（每组 3 次）：C−A = '+(m.tactile_action_icl-m.no_demo)+'；C−B = '+(m.tactile_action_icl-m.visual_action_icl)+'；B−A = '+(m.visual_action_icl-m.no_demo)+'。仅为本次开发 pilot，不能推广为通用收益或通用无效。</p>';}
return table+'</section>';
}
'''
R17_HTML = R17_HTML.replace('async function render()', R17_SUMMARY+'\nasync function render()').replace(
    "document.querySelector('#content').innerHTML=d.batches.map", "document.querySelector('#content').innerHTML=primaryTable(d)+d.batches.map")


R18_SUMMARY = r'''function ruleTable(d){
const names={UA:'U 官方说明 · A 无示范',UB:'U 官方说明 · B 视觉示范',UC:'U 官方说明 · C 增加历史触觉',RA:'R 官方说明＋公开规则 · A 无示范',RB:'R 官方说明＋公开规则 · B 视觉示范',RC:'R 官方说明＋公开规则 · C 增加历史触觉'};
const groups=Object.fromEntries(d.batches.map(b=>[b.name.split(' / ').at(-1),b.episodes]));
const valid=c=>(groups[c]||[]).filter(e=>e.episode.evaluable).length;
const wins=c=>(groups[c]||[]).filter(e=>e.episode.evaluable&&e.episode.task_success).length;
let t='<section><h2>GPT-6 / low：公开规则 × 历史示范</h2><p>R 为事先选定的完整任务信息设置。所有当前观测与 original 控制器相同；每格三个开发 seed。历史 R1.7 使用另一模型，不能作本轮因果对照。</p><table style="width:100%;text-align:left"><tr><th>任务信息</th><th>A 无示范</th><th>B 视觉示范</th><th>C 增加历史触觉</th></tr>';
for(const row of ['U','R']){t+='<tr><th>'+row+'</th>';for(const col of ['A','B','C']){const c=row+col;t+='<td>'+wins(c)+'/3；可评价 '+valid(c)+'/3；基础设施异常 '+(groups[c]||[]).filter(e=>e.episode.infrastructure_error).length+'</td>';}t+='</tr>';}
t+='</table><h3>固定比较（百分点）</h3>';
for(const [a,b,label] of [['RA','UA','规则'],['RB','UB','规则'],['RC','UC','规则'],['RB','RA','视觉示范'],['RC','RA','完整示范'],['RC','RB','R 下历史触觉'],['UC','UB','U 下历史触觉']])t+='<p>'+a+'−'+b+' '+label+'：'+(valid(a)===3&&valid(b)===3?((wins(a)-wins(b))*100/3).toFixed(1)+' pp':'待完成／存在不可评价项')+'</p>';
const prompts=d.batches[0]?.manifest?.prompts??{};for(const c of ['UA','RA'])if(prompts[c])t+='<details><summary>'+c[0]+' 完整冻结初始文本（A/B/C 共用）</summary><pre>'+esc(prompts[c])+'</pre></details>';
t+='<p>cached input 为 input 子集，不重复求和。较早失败的短耗时不算效率提高。各格成本、原始回答与失败原因见下方。</p><h3>六条件导航</h3>'+Object.entries(names).map(([c,n])=>'<p>'+c+'：'+n+'</p>').join('');return t+'</section>';}
'''
R18_HTML = R17_HTML.replace('R1.7', 'R1.8').replace('r17-', 'r18-').replace(
    '九条视频', '十八条视频').replace(R17_SUMMARY, R18_SUMMARY).replace('primaryTable(d)', 'ruleTable(d)').replace(
    "review(e)+'<h3>Agent Saw", "'<details><summary>实际初始 prompt：官方任务说明、共同工具说明与本条件规则全文</summary><pre>'+esc(e.prompt??'尚未启动')+'</pre></details>'+review(e)+'<h3>Agent Saw").replace(
    "esc(state.seed)+' · '+esc(state.status)", "esc(state.seed)+' · '+esc(state.condition)+' · '+esc(state.model)+' / '+esc(state.reasoning_effort)+' · '+esc(state.status)")

R18_HTML = R18_HTML.replace('三组当前观测相同', '六条件当前观测相同').replace(
    "esc(state.condition)", "esc(state.condition??e.run.split('/').at(-1))").replace(
    "wins(c)+'/3；可评价 '", "(valid(c)===3?wins(c)+'/3':wins(c)+' 成功（计划3，未完成）')+'；可评价 '")

R19_SUMMARY = r'''
function newSeedTable(d){
const names={RA:'RA 无示范',RB:'RB 视觉—运动示范',RC:'RC 增加历史触觉'};
const groups=Object.fromEntries(d.batches.map(b=>[b.name.split(' / ').at(-1),b.episodes]));
const valid=c=>(groups[c]||[]).filter(e=>e.episode.evaluable).length;
const wins=c=>(groups[c]||[]).filter(e=>e.episode.evaluable&&e.episode.task_success).length;
let t='<section><h2>R1.9：十二个新 seed，冻结 R 输入设置</h2><p>三个条件共用 GPT-6 / low、官方说明＋公开规则、original 控制器与当前触觉历史。不是官方 test split；R1.8 不并入此表。主比较 RC−RB。</p><table style="width:100%;text-align:left"><tr><th>条件</th><th>成功/计划</th><th>可评价</th><th>原生失败</th><th>基础设施异常</th></tr>';
for(const c of ['RA','RB','RC'])t+='<tr><td>'+names[c]+'</td><td>'+wins(c)+'/12'+(valid(c)<12?'（尚未全部可评价）':'')+'</td><td>'+valid(c)+'/12</td><td>'+((groups[c]||[]).filter(e=>e.episode.evaluable&&!e.episode.task_success).length)+'</td><td>'+((groups[c]||[]).filter(e=>e.episode.infrastructure_error).length)+'</td></tr>';
t+='</table>';
for(const [a,b] of [['RC','RB'],['RC','RA'],['RB','RA']])t+='<p>'+a+'−'+b+'：'+(valid(a)===12&&valid(b)===12?((wins(a)-wins(b))*100/12).toFixed(1)+' pp':'待完成／存在不可评价项')+'</p>';
for(const c of ['RB','RA']){let counts=[0,0,0,0,0];const index=new Map((groups[c]||[]).map(e=>[e.episode.seed,e.episode]));for(let seed=1000006;seed<=1000017;seed++){const a=(groups.RC||[]).find(e=>e.episode.seed===seed)?.episode,b=index.get(seed);counts[!a?.evaluable||!b?.evaluable?4:a.task_success?(b.task_success?2:0):(b.task_success?1:3)]++;}t+='<p>RC 对 '+c+'：仅 RC 成功 / 仅对照成功 / 都成功 / 都失败 / 未完成或不可评价 = '+counts.join(' / ')+'</p>';}
const prompts=d.batches[0]?.manifest?.prompts??{};if(prompts.RA)t+='<details><summary>三组共同的完整冻结 prompt</summary><pre>'+esc(prompts.RA)+'</pre></details>';
return t+'<p>逐 seed 成本与真实 usage 在下方。cached 是 input 子集，reasoning 是 output 子集，不重复求和。较早失败不算效率提高。</p></section>';
}
'''
R19_HTML = R17_HTML.replace(R17_SUMMARY, R19_SUMMARY).replace(
    'primaryTable(d)', 'newSeedTable(d)').replace('R1.7', 'R1.9').replace('r17-', 'r19-').replace(
    '九条视频', '三十六条视频').replace(
    '<main id="content">', '<p><a href="/r19-pairs">同 seed 的 RB/RC 并排慢放 →</a></p><main id="content">').replace(
    'valid.length===3','valid.length===12').replace("wins.length+'/3'", "wins.length+'/12'").replace(
    "valid.length+'/3；成功 '", "valid.length+'/12；成功 '").replace(
    "review(e)+'<h3>Agent Saw", "'<details><summary>实际冻结初始 prompt 全文</summary><pre>'+esc(e.prompt??'尚未启动')+'</pre></details>'+review(e)+'<h3>Agent Saw").replace(
    "esc(state.seed)+' · '+esc(state.status)", "esc(state.seed)+' · '+esc(state.condition??e.run.split('/').at(-1))+' · '+esc(state.model)+' / '+esc(state.reasoning_effort)+' · '+esc(state.status)")

R110_SUMMARY = r'''
function currentTouchTable(d){
const groups=Object.fromEntries(d.batches.map(b=>[b.name.split(' / ').at(-1),b.episodes]));
const valid=c=>(groups[c]||[]).filter(e=>e.episode.evaluable).length;
const wins=c=>(groups[c]||[]).filter(e=>e.episode.evaluable&&e.episode.task_success).length;
const comparisons=[['C_live','B_live'],['C_no_live','B_no_live'],['B_live','B_no_live'],['C_live','C_no_live']];
const delta=(a,b)=>valid(a)===8&&valid(b)===8?(wins(a)-wins(b))*100/8:null;
let t='<section><h2>R1.10：历史触觉 × 当前触觉</h2><p>GPT-6 / low，R 完整规则、同一 original 控制器和两条官方 expert。八个新 seed；历史轮次不并表。no_live 只移除 Agent 操作中的当前触觉，不关闭传感器、初始化自适应抓持、物理或录像。</p><table style="width:100%;text-align:left"><tr><th>历史示范</th><th>有当前触觉 live</th><th>无当前触觉 no_live</th></tr>';
for(const history of ['B','C']){t+='<tr><th>'+history+(history==='B'?' 视觉—运动':' 同示范＋历史触觉')+'</th>';for(const suffix of ['live','no_live']){const c=history+'_'+suffix,es=groups[c]||[];t+='<td>'+wins(c)+'/8；可评价 '+valid(c)+'/8；原生失败 '+es.filter(e=>e.episode.evaluable&&!e.episode.task_success).length+'；基础设施 '+es.filter(e=>e.episode.infrastructure_error).length+'</td>';}t+='</tr>';}
t+='</table><h3>固定比较与逐 seed 配对</h3>';
for(const [a,b] of comparisons){let counts=[0,0,0,0,0];for(let seed=1000018;seed<=1000025;seed++){const x=(groups[a]||[]).find(e=>e.episode.seed===seed)?.episode,y=(groups[b]||[]).find(e=>e.episode.seed===seed)?.episode;counts[!x?.evaluable||!y?.evaluable?4:x.task_success?(y.task_success?2:0):(y.task_success?1:3)]++;}const value=delta(a,b);t+='<p>'+a+'−'+b+'：'+(value===null?'待完成／不可评价':value.toFixed(1)+' pp')+'；仅前者 / 仅后者 / 共同成功 / 共同失败 / 未完成或不可评价：'+counts.join(' / ')+'</p>';}
const a=delta('C_live','B_live'),b=delta('C_no_live','B_no_live');t+='<p>描述性差中差：(C_live−B_live)−(C_no_live−B_no_live) = '+(a===null||b===null?'待完成／不可评价':(a-b).toFixed(1)+' pp')+'；不是模型内部机制证明。</p>';
const prompts=d.batches[0]?.manifest?.prompts??{};if(prompts.B_live)t+='<details><summary>四组共同 prompt 全文</summary><pre>'+esc(prompts.B_live)+'</pre></details>';
return t+'<p>成本详单在下方；cached/ reasoning 是 input/output 子集，不重复求和。移除模态也改变上下文长度，未隔离全部内部机制。较早失败不算效率优势。</p></section>';
}
'''
R110_HTML = R19_HTML.replace(R19_SUMMARY, R110_SUMMARY).replace(
    'newSeedTable(d)', 'currentTouchTable(d)').replace('R1.9', 'R1.10').replace('r19-', 'r110-').replace(
    '三十六条视频', '三十二条视频').replace('同 seed 的 RB/RC 并排慢放', '同 seed 的 live/no_live 并排慢放').replace(
    'valid.length===12', 'valid.length===8').replace("wins.length+'/12'", "wins.length+'/8'").replace(
    "valid.length+'/12；成功 '", "valid.length+'/8；成功 '").replace(
    '三组当前观测相同：视觉＋触觉历史＋本体状态。A 无示范；B 两条官方 expert 的视觉—运动示范；C 相同示范再加历史触觉。',
    '四组均有视觉、本体与两条官方 expert 示范。B/C 只差历史触觉；live/no_live 只差 Agent 操作中的当前触觉。后台四路录像均保留。').replace(
    '调试用于验证接口，不计入成功率。', '本轮不新增联调或模型预试。').replace(
    'Object.keys(r.images));', "Object.keys(r.images).map(n=>n+(e.episode.current_tactile===false&&n.includes('tactile')?' · 仅供用户审阅，本episode未送给Agent':'')));").replace(
    '<h4>Before / After Vision and Touch</h4>',
    '<h4>Host-only 前后记录：可能包含未交付触觉；真实 Agent 输入仅以上方 MCP 为准</h4>')


R111_SUMMARY = R19_SUMMARY.replace('R1.9：十二个新 seed，冻结 R 输入设置', 'R1.11：Grasp & Classify，六个开发 seed').replace(
    '不是官方 test split；R1.8 不并入此表。主比较 RC−RB。',
    '第二任务开发 pilot，Insert Hole 历史结果不并表。主比较 C−B；类别仅用于事后 host 分析。').replace(
    '/12', '/6').replace('===12', '===6').replace('<12', '<6').replace('1000006', '1000026').replace('1000017', '1000031').replace(
    'RA', 'A').replace('RB', 'B').replace('RC', 'C')
R111_SUMMARY = R111_SUMMARY.replace("return t+'<p>逐 seed", "t+='<details class=host><summary>Host-only：实际类别分布与成功数</summary>';for(const c of ['A','B','C'])for(const kind of ['rough','plain']){const es=(groups[c]||[]).filter(e=>e.host_evaluator?.object_class===kind);t+='<p>'+c+' / '+kind+'：'+es.filter(e=>e.episode.evaluable&&e.episode.task_success).length+' 成功 / '+es.length+' 条</p>';}t+='</details>';return t+'<p>逐 seed")
R111_HTML = R19_HTML.replace(R19_SUMMARY, R111_SUMMARY).replace('R1.9','R1.11').replace('r19-', 'r111-').replace(
    '三十六条视频','十八条视频').replace('Autonomous Insert Hole','Autonomous Grasp & Classify').replace(
    'valid.length===12','valid.length===6').replace("wins.length+'/12'", "wins.length+'/6'").replace(
    "valid.length+'/12；成功 '", "valid.length+'/6；成功 '").replace('RB/RC','B/C')
