"""R1.4 replay using exact operator context and separate host execution records."""
from __future__ import annotations

import json
from pathlib import Path

from sim.envs.univtac.codex_readonly import read_jsonl


def load_autonomous_runs(runs_root: Path) -> dict:
    batches = []
    for manifest_path in sorted(runs_root.glob('univtac-isaac51-r14*/run_manifest.json')):
        manifest = json.loads(manifest_path.read_text())
        if manifest.get('round') != 'R1.4':
            continue
        episodes = []
        for path in sorted(manifest_path.parent.glob('seed_*/episode.json')):
            root = path.parent
            def read(name, root=root):
                candidate = root/name
                return json.loads(candidate.read_text()) if candidate.exists() else None
            episodes.append({'run':str(root.relative_to(runs_root)), 'episode':read('episode.json'),
                             'context':read_jsonl(root/'operator_context.jsonl'),
                             'execution':read_jsonl(root/'tool_trace.jsonl'),
                             'debug':read('debug_controls.json'),
                             'host_evaluator':read('host_evaluator.json'),
                             'usage':read('codex_trace_summary.json'),
                             'worker_error':read('worker_error.json'), 'codex_lifecycle':read('codex_lifecycle.json')})
        note_path=manifest_path.parent/'validation_note.json'
        note=json.loads(note_path.read_text()) if note_path.exists() else None
        batches.append({'name':manifest_path.parent.name,'manifest':manifest,'episodes':episodes,'validation_note':note})
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
function images(run,paths,labels=[]){return '<div class="images">'+paths.map((p,i)=>'<figure><img loading="lazy" src="'+art(run,p)+'"><figcaption>'+esc(labels[i]??p)+'</figcaption></figure>').join('')+'</div>'}
function observation(run,o){if(!o)return '';return '<p>'+esc(o.observation_id)+' · step '+esc(o.counts?.simulator_step)+'</p>'+images(run,(o.images||[]).map(x=>x.path),(o.images||[]).map(x=>x.label));}
async function render(){const d=await (await fetch('/api/r14-autonomous')).json();document.querySelector('#content').innerHTML=d.batches.map(b=>{
const scored=b.manifest.mode==='batch',valid=b.episodes.filter(e=>e.episode.evaluable),wins=valid.filter(e=>e.episode.task_success);
return '<section><h2>'+esc(b.name)+'</h2><p class="'+(scored?'metric':'debug')+'">'+(scored?(valid.length===3?'自主操作开发成功率 '+wins.length+'/3':'正式开发批次：可评价 '+valid.length+'/3；成功 '+wins.length+'；其余状态见下方'):'独立控制联调 · 不计分')+'</p>'+(b.validation_note?'<p class="debug">几何复验说明：'+esc(b.validation_note.geometry_validation)+'。腕部几何未通过；正式接口仅开放 head 标点。</p>':'')+b.episodes.map(e=>{
const state=e.episode;return '<article><h3>seed '+esc(state.seed)+' · '+esc(state.status)+'</h3><p>Native outcome: '+esc(state.native_success_available?state.task_success:'unavailable / pending')+' · 结束原因 '+esc(reason(state.termination))+'</p>'+pretty({'工具调用':state.tool_call_count,'非预览请求':state.move_request_count,'实际运动请求':state.actual_motion_requests,'控制步':state.control_steps,'物理步':state.physics_steps,'仿真秒':state.simulation_time_seconds,'Codex耗时秒':e.codex_lifecycle?.elapsed_seconds,'token用量':e.usage?.usage??'未返回，不能记为0'})+
'<h3>Agent Saw → Agent Requested → Environment Executed</h3>'+(e.context.length?e.context.map(row=>{
const execution=e.execution[row.seq-1];return '<details><summary>'+row.seq+' · '+esc(row.tool)+'</summary><h4>Agent Requested</h4>'+pretty(row.arguments)+'<h4>Agent Saw — 实际 MCP 返回图片</h4>'+images(e.run,row.response_image_paths||[])+ '<details><summary>完整 MCP 文本</summary>'+row.response_text_blocks.map(x=>'<pre>'+esc(x)+'</pre>').join('')+'</details>'+'<h4>Environment Executed</h4><p>'+esc(execution?.result?.text?.execution?.reached===true?'机器人到达请求目标（不代表任务成功）':execution?.result?.text?.execution?.error??'见工具反馈')+'</p><details><summary>实际目标、机器人状态与执行反馈</summary>'+pretty(execution?.result?.text?.execution??execution?.result?.text)+'</details>'+'<h4>Before / After Vision and Touch</h4>'+observation(e.run,execution?.before)+observation(e.run,execution?.result?.text?.observation)+'</details>'}).join(''):'<p>无模型上下文：尚未启动 Codex，或此为纯控制联调。</p>')+
(e.debug?'<details><summary>不计分的调试请求、反馈与前后图片</summary>'+pretty(e.debug)+e.execution.filter(x=>x.tool==='move_to'&&!x.arguments.preview).map(x=>observation(e.run,x.before)+observation(e.run,x.result?.text?.observation)).join('')+'</details>':'')+
'<details class="host"><summary>Host-only：原生评价、内部错误与成本</summary>'+pretty({episode:state,evaluator:e.host_evaluator,worker_error:e.worker_error,usage:e.usage?.usage??'unavailable',codex_lifecycle:e.codex_lifecycle})+'</details></article>'}).join('')+'</section>'}).join('')||'<p>尚无 R1.4 运行记录。</p>'}
render();
</script></body></html>'''
