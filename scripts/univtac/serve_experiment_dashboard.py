#!/usr/bin/env python3
"""Serve a read-only replay dashboard for UniVTAC Codex experiments."""

from __future__ import annotations

import argparse
import json
import mimetypes
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

from scripts.univtac.autonomous_dashboard import HTML as R14_HTML
from scripts.univtac.autonomous_dashboard import load_autonomous_runs
from sim.envs.univtac.codex_readonly import read_jsonl, summarize_codex_exec

LIST_HTML = """<!doctype html><meta charset="utf-8"><title>UniVTAC experiments</title>
<style>body{font:14px system-ui;background:#101216;color:#e6e9ef;margin:24px}a{color:#8fd3ff}table{border-collapse:collapse;width:100%;background:#181c22}th,td{padding:9px;border:1px solid #303744;text-align:left}.ok{color:#76db8b}.failed{color:#ff8585}</style>
<h1>UniVTAC experiment runs</h1><p><a href="/progress">项目进展：我和 GPT-Pro 每轮做了什么 →</a></p><p><a href="/r10-operation">R1.0：第一次 Codex 真实操作 →</a></p><p><a href="/r11-icl">R1.1：第一次真实 Tactile-Action ICL →</a></p><p><a href="/r12-branching">R1.2：筛选真正需要触觉分支的任务 →</a></p><p><a href="/r13-icl">R1.3：Insert Hole 接触前后 Tactile ICL →</a></p><p><a href="/r14-autonomous">R1.4 Autonomous Insert Hole →</a></p><p><a href="/r15-autonomous">R1.5：夹爪保持与触觉历史 →</a></p><p><a href="/r16-motion-pacing">R1.6：控制时序对照与 expert 示范 →</a></p><p><a href="/r17-autonomous">R1.7：官方 expert 示范驱动的自主触觉 ICL →</a></p><p><a href="/r18-autonomous">R1.8：GPT-6 low 公开规则 × 触觉 ICL →</a></p><p><a href="/r19-autonomous">R1.9：十二个新 seed 的触觉 ICL 验证 →</a></p><div id="pilots"></div><table><thead><tr><th>round / directory</th><th>task</th><th>seed</th><th>model</th><th>status</th><th>start</th><th>duration</th><th>observe count</th></tr></thead><tbody id="runs"></tbody></table>
<script>const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));async function tick(){const [xs,ps]=await Promise.all([fetch('/api/runs?t='+Date.now()).then(r=>r.json()),fetch('/api/pilots?t='+Date.now()).then(r=>r.json())]);document.querySelector('#pilots').innerHTML=ps.map(x=>`<p><a href="/pilot/${encodeURIComponent(x.directory)}">Causal Pilot: ${esc(x.directory)}</a> · ${esc(x.signal)} · ${esc(x.completed_call_count)} calls</p>`).join('');document.querySelector('#runs').innerHTML=xs.map(x=>`<tr><td><a href="/run/${encodeURIComponent(x.directory)}">${esc(x.round)} / ${esc(x.directory)}</a></td><td>${esc(x.task)}</td><td>${esc(x.seed)}</td><td>${esc(x.model)}</td><td class="${x.status==='completed'?'ok':'failed'}">${esc(x.status)}</td><td>${esc(x.started_at)}</td><td>${esc(x.duration_seconds??'')}</td><td>${esc(x.observe_count)}</td></tr>`).join('')}tick();setInterval(tick,2000)</script>"""

PILOT_HTML = """<!doctype html><meta charset="utf-8"><title>UniVTAC Causal Pilot</title>
<style>body{font:14px system-ui;background:#101216;color:#e6e9ef;margin:20px}a{color:#8fd3ff}.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:10px}.grid.two{grid-template-columns:repeat(2,1fr)}.cell,section{background:#181c22;border:1px solid #303744;border-radius:8px;padding:12px;margin:10px 0}.images{display:grid;grid-template-columns:1fr 1fr;gap:5px}.pair-images{display:grid;grid-template-columns:repeat(3,1fr);gap:8px}.images img,.pair-images img{width:100%;background:#000}pre{white-space:pre-wrap;overflow:auto;background:#0c0e12;padding:8px}.event{border-left:3px solid #5b8bad;padding:7px 10px;margin:8px 0;background:#11161c}.host{border-color:#9a7030}.host-grid{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}.host-card{background:#20252d;padding:10px;border-radius:6px}.bar{height:12px;background:#343b47;margin:4px 0}.bar span{display:block;height:100%;background:#e3a84f}</style>
<a href="/">← runs</a><h1 id="pilotTitle">Pull Out Key · 3 seeds × 3 conditions</h1><section id="top"></section><div class="grid" id="matrix"></div><section id="pairs" hidden></section><section id="humanComparisons" hidden></section><section id="comparison"></section><section class="host"><details><summary><b id="hostTitle">HOST-ONLY EVALUATION — NOT SHOWN TO CODEX</b></summary><div id="hostBars" class="host-grid"></div><pre id="host"></pre></details></section>
<script>
const pilot=decodeURIComponent(location.pathname.slice(7));
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const art=(run,p)=>'/artifact?run='+encodeURIComponent(run)+'&path='+encodeURIComponent(p);
const images=c=>`<div class="images">${(c.image_paths||[]).map((p,i)=>`<figure><img src="${art(c.run,p)}"><figcaption>${esc((c.image_labels||[])[i]||p)}</figcaption></figure>`).join('')}</div>`;
const swapNote=c=>c.condition==='swapped_tactile'?'<p><i>LEFT/RIGHT TACTILE ASSIGNMENT SWAPPED — host-side label, not sent through MCP.</i></p>':c.condition==='swapped_difference'?'<p><i>LEFT/RIGHT DIFFERENCE MAP ASSIGNMENT SWAPPED — host-side label, not sent through MCP.</i></p>':c.condition.includes('swapped')?'<p><i>HOST LABEL: guidance is swapped; this label was not sent through MCP.</i></p>':'';
const legacyCell=c=>`${images(c)}${c.structured_summary?`<h4>Actual structured text seen by Codex</h4><pre>${esc(JSON.stringify(c.structured_summary,null,2))}</pre>`:'<p><i>No structured tactile summary was sent.</i></p>'}<pre>${esc(JSON.stringify(c.prediction,null,2))}</pre><details><summary>raw answer</summary><pre>${esc(c.raw_answer)}</pre></details>`;
const stagedCell=c=>`<div class="event"><h4>1. Agent Saw Images</h4>${images(c)}</div><div class="event"><h4>2. Image-Only Judgment Committed</h4><pre>${esc(JSON.stringify(c.image_judgment,null,2))}</pre></div><div class="event"><h4>3. Structured Guidance Revealed</h4><pre>${esc(JSON.stringify(c.structured_summary,null,2))}</pre></div><div class="event"><h4>4. Final Reaction</h4><pre>${esc(JSON.stringify(c.prediction,null,2))}</pre><p>provisional → final changed: ${esc(c.provisional_changed)} · conflict detected: ${esc(c.conflict_detected)}</p><details><summary>raw answer</summary><pre>${esc(c.raw_answer)}</pre></details></div>`;
const mechanismCell=c=>{const simultaneous=String(c.protocol).startsWith('simultaneous');const committed=String(c.protocol).includes('with_commit');return `<div class="event"><h4>Agent Saw</h4>${images(c)}${simultaneous?`<h4>Structured Guidance Arrived Together</h4><pre>${esc(JSON.stringify(c.structured_summary,null,2))}</pre>`:''}</div>${committed?`<div class="event"><h4>Image-Only Judgment Committed</h4><pre>${esc(JSON.stringify(c.image_judgment,null,2))}</pre></div>`:''}${!simultaneous?`<div class="event"><h4>Structured Guidance Revealed After Images</h4><pre>${esc(JSON.stringify(c.structured_summary,null,2))}</pre></div>`:''}<div class="event"><h4>Final Reaction</h4><pre>${esc(JSON.stringify(c.prediction,null,2))}</pre><p>follow image: ${esc(c.follow_image)} · follow text: ${esc(c.follow_text)} · conflict detected: ${esc(c.conflict_detected)} · evidence basis: ${esc(c.prediction.final_evidence_basis)}</p><details><summary>raw answer</summary><pre>${esc(c.raw_answer)}</pre></details></div>`};
async function tick(){
 const d=await fetch('/api/pilot?pilot='+encodeURIComponent(pilot)+'&t='+Date.now()).then(r=>r.json()),round=d.pilot.round;
 document.querySelector('#pilotTitle').textContent=round==='R0.9.20'?'Pull Out Key · Conflict-Aware Instruction Interaction Ablation':round==='R0.9.19'?'Pull Out Key · Tactile Grounding Mechanism Ablation':round==='R0.9.18'?'Pull Out Key · Image-First Staged Grounding Pilot':round==='R0.9.17'?'Pull Out Key · Unilateral Tactile Guidance Conflict Pilot':round==='R0.9.16'?'Pull Out Key · RGB-marker Structured Guidance Pilot':round==='R0.9.15'?'Pull Out Key · Before/After Tactile Difference Pilot':'Pull Out Key · 3 seeds × 3 conditions';
 const calls=d.summary.completed_semantic_trials??d.summary.completed_call_count,signal=d.summary.minimal_skill_candidate??d.summary.mechanism_result??d.summary.grounding_signal??d.summary.unilateral_signal??d.summary.structured_signal??d.summary.difference_signal??d.summary.pilot_signal;
 document.querySelector('#top').innerHTML=`<b>model:</b> ${esc(d.pilot.model)} · <b>prompt:</b> ${round==='R0.9.20'?'common_conflict_instruction.txt + protocol_prompts/':round==='R0.9.19'?'prompt_common.txt + protocol_prompts/':'prompt.txt'} · <b>completed calls:</b> ${esc(calls)} · <b>signal:</b> ${esc(signal)}`;
 const matrix=document.querySelector('#matrix');matrix.classList.toggle('two',round==='R0.9.18'||round==='R0.9.20');matrix.innerHTML=d.cells.map(c=>`<div class="cell"><h3>seed ${esc(c.seed)} · ${esc(c.condition)}</h3>${swapNote(c)}${round==='R0.9.19'||round==='R0.9.20'?mechanismCell(c):c.phases.length?stagedCell(c):legacyCell(c)}<p>${esc(c.duration_seconds)} s · tool ${esc(c.tool_call_count)} · usage ${esc(JSON.stringify(c.usage||{}))}</p><a href="/run/${encodeURIComponent(c.run)}">detail</a></div>`).join('');
 const pairs=document.querySelector('#pairs');pairs.hidden=!d.pairs.length;pairs.innerHTML='<h2>Baseline | Current | Difference</h2>'+d.pairs.map(x=>`<h3>seed ${esc(x.seed)} · ${esc(x.sensor)}</h3><div class="pair-images">${x.images.map(i=>`<figure><img src="${art(i.run,i.path)}"><figcaption>${esc(i.label)}</figcaption></figure>`).join('')}</div>`).join('');
 const human=document.querySelector('#humanComparisons');human.hidden=!d.human_comparisons.length;human.innerHTML='<h2>每个 seed 用人话对比</h2>'+d.human_comparisons.map(x=>`<div class="host-card"><b>seed ${esc(x.seed)}</b><p>输入里 ${esc(x.baseline_side)} 侧被换成 baseline，真实图片变化侧是 ${esc(x.expected_change_side)}。</p><p>只看图片：${esc(x.image_only_side)}；正确文字：${esc(x.correct_guidance_side)}（是否帮助：${esc(x.correct_guidance_helped)}）；错误文字：${esc(x.swapped_guidance_side)}，因此模型${esc(x.conflict_reaction)}。</p></div>`).join('');
 document.querySelector('#comparison').innerHTML='<h2>Cross-condition comparisons</h2><pre>'+esc(JSON.stringify(d.summary.metrics??d.summary,null,2))+'</pre>'+(round==='R0.9.18'?'<h3>R0.9.17 single-stage swapped vs image-first committed swapped</h3><pre>'+esc(JSON.stringify(d.summary.r0917_comparison||{},null,2))+'</pre>':'')+(round==='R0.9.19'?'<h2>Mechanism funnel</h2><div class="host-grid">'+(d.mechanism_funnel||[]).map(x=>`<div class="host-card"><b>${esc(x.name)}</b><p>follow image ${esc(x.followed)}/3 · threshold reached: ${esc(x.reached)}</p></div>`).join('')+'</div>':'')+(round==='R0.9.20'?'<h2>Conflict-instruction mechanism comparison</h2><div class="host-grid">'+(d.interaction_funnel||[]).map(x=>`<div class="host-card"><b>${esc(x.name)}</b><p>follow image ${esc(x.followed)}/3 · pilot ≥2/3: ${esc(x.pilot_sufficient)} · robust 3/3: ${esc(x.robust)}</p></div>`).join('')+'</div>':'');
 const refs=d.structured_reference?.seeds||{};document.querySelector('#hostBars').innerHTML=Object.values(refs).flatMap(x=>Object.entries(x.sensors||{}).map(([sensor,s])=>{const m=s.structured_metrics||{},v=m.normalized_horizontal_saliency||[];return `<div class="host-card"><b>seed ${esc(x.seed)} · ${esc(sensor)}</b><p>new: ${esc(m.reference_region)} · old centroid: ${esc(s.old_global_centroid_region)} · agree: ${esc(s.old_and_new_region_agree)}</p>${['left','center','right'].map((name,i)=>`<label>${name} ${esc(Number(v[i]||0).toFixed(3))}</label><div class="bar"><span style="width:${Math.max(0,Math.min(100,(v[i]||0)*100))}%"></span></div>`).join('')}</div>`})).join('');
 document.querySelector('#hostTitle').textContent=round==='R0.9.20'?'HOST-ONLY CONFLICT-INSTRUCTION SCORES — NOT SHOWN TO CODEX':round==='R0.9.19'?'HOST-ONLY EXPECTED IMAGE / SWAPPED TEXT SIDES — NOT SHOWN TO CODEX':round==='R0.9.18'?'HOST-ONLY STAGED GROUNDING SCORES — NOT SHOWN TO CODEX':round==='R0.9.17'?'HOST-ONLY COUNTERFACTUAL MAPPING — NOT SHOWN TO CODEX':'HOST-ONLY EVALUATION — NOT SHOWN TO CODEX';
 document.querySelector('#host').textContent=JSON.stringify({reference:d.host_reference,difference_metrics:d.difference_metrics,structured_reference:d.structured_reference,secondary_diagnostics:d.secondary_diagnostics,source_mapping:d.source_mapping,host_expectations:d.host_expectations,condition_manifests:d.condition_manifests},null,2);
}
tick();setInterval(tick,2000)
</script>"""

PROGRESS_HTML = """<!doctype html><meta charset="utf-8"><title>OpenETA-UniVTAC 项目进展</title>
<style>body{font:15px/1.65 system-ui;background:#101216;color:#e6e9ef;margin:24px auto;max-width:1050px;padding:0 18px}a{color:#8fd3ff}.intro{color:#b8c0cc}.entry{background:#181c22;border:1px solid #303744;border-radius:12px;padding:20px 24px;margin:18px 0}.entry h3{margin:18px 0 5px;color:#d9e7f5}.entry p{white-space:pre-wrap;margin:5px 0}.meta{color:#98a3b3;font-size:13px}.status{display:inline-block;border-radius:14px;padding:3px 9px;background:#245f3b;font-size:12px}.warn{background:#805b20}.question{border-left:4px solid #b07cff;padding-left:14px}.method{border-left:4px solid #55b8ff;padding-left:14px}.observed{border-left:4px solid #65c78d;padding-left:14px}.meaning{border-left:4px solid #f0b55a;padding-left:14px}.conclusion{font-size:17px;background:#222934;border-left:4px solid #e8edf5;margin:20px 0 10px;padding:12px 16px}details{background:#11161c;border-radius:8px;padding:10px 12px;margin-top:14px}pre{white-space:pre-wrap}.experiments{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:10px}.card{background:#181c22;border:1px solid #303744;border-radius:8px;padding:12px}</style>
<a href="/">← 实验列表</a><h1>OpenETA-UniVTAC：现在做到哪了</h1><p class="intro">每一轮先讲清楚问题、做法、观察和含义；commit、测试和原始产物收在“技术证据”里。这里只展示可见工作记录，不展示隐藏思维或认证信息。</p><div id="entries"></div><h2>实验可视化</h2><div class="experiments" id="experiments"></div>
<script>const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const list=x=>Array.isArray(x)?x:[];async function tick(){const d=await fetch('/api/progress?t='+Date.now()).then(r=>r.json());document.querySelector('#entries').innerHTML=[...d.entries].reverse().map(x=>{const q=x.pro_question||x.pro_instruction_summary||'';const method=x.method_summary||x.codex_work_summary||'';const observed=x.observed_summary||x.result_summary||'尚未结束';const meaning=x.implication_summary||'';const conclusion=x.one_line_conclusion||x.result_summary||'尚未结束';const evidence=x.evidence||{};const root=evidence.experiment_root||x.experiment_root||'';const commit=evidence.commit||x.commit||'';const push=evidence.push||x.push||'';const tests=list(evidence.tests);const artifacts=list(evidence.artifact_paths);return `<article class="entry"><h2>${esc(x.round)} <span class="status ${x.status==='completed'?'':'warn'}">${esc(x.status)}</span></h2><p class="meta">${esc(x.timestamp)}</p><section class="question"><h3>Pro 想弄清楚什么</h3><p>${esc(q)}</p></section><section class="method"><h3>我们怎么验证的</h3><p>${esc(method)}</p></section><section class="observed"><h3>实际看到了什么</h3><p>${esc(observed)}</p></section><section class="meaning"><h3>这说明什么</h3><p>${esc(meaning||'本轮尚未形成新的研究判断。')}</p></section><blockquote class="conclusion"><b>一句话结论：</b> ${esc(conclusion)}</blockquote>${(x.dashboard_url||root)?`<p><a href="${esc(x.dashboard_url||("/pilot/"+encodeURIComponent(root)))}">打开本轮实验可视化 →</a></p>`:''}<details><summary>技术证据（commit、测试、产物）</summary><p>commit: ${esc(commit||'未记录')}</p><p>push: ${esc(push||'未记录')}</p>${tests.length?`<h4>测试</h4><ul>${tests.map(t=>`<li>${esc(t)}</li>`).join('')}</ul>`:''}${artifacts.length?`<h4>产物</h4><ul>${artifacts.map(p=>`<li>${esc(p)}</li>`).join('')}</ul>`:''}</details></article>`}).join('');document.querySelector('#experiments').innerHTML=d.pilots.map(x=>`<div class="card"><h3>${esc(x.directory)}</h3><p>信号：${esc(x.signal||'运行中')} · 已完成 ${esc(x.completed_call_count)} 次</p><a href="/pilot/${encodeURIComponent(x.directory)}">打开实验图片与回答</a></div>`).join('')}tick();setInterval(tick,2000)</script>"""

DETAIL_HTML = """<!doctype html><meta charset="utf-8"><title>UniVTAC replay</title>
<style>body{font:14px system-ui;background:#101216;color:#e6e9ef;margin:20px}a{color:#8fd3ff}section{background:#181c22;border:1px solid #303744;border-radius:8px;padding:14px;margin:12px 0}pre{white-space:pre-wrap;overflow:auto;background:#0c0e12;padding:10px}.images{display:grid;grid-template-columns:1fr 1fr;gap:10px}.images img{width:100%;max-height:420px;object-fit:contain;background:#000}.badge{display:inline-block;background:#245f3b;padding:4px 8px;border-radius:12px}.event{border-left:3px solid #5b8bad;padding:7px 10px;margin:8px 0;background:#11161c}.host{border-color:#9a7030}</style>
<a href="/">← runs</a><h1 id="title">UniVTAC replay</h1><div class="badge">READ ONLY</div>
<section><h2>Agent Saw</h2><strong>ACTUAL MCP CONTEXT SEEN BY CODEX</strong><div id="saw"></div></section>
<section><h2>Agent Encountered</h2><div id="timeline"></div></section>
<section><h2>Agent Reaction</h2><div id="reaction"></div></section>
<section class="host"><details><summary><b>HOST-ONLY — NOT SHOWN TO AGENT</b></summary><pre id="host"></pre></details></section>
<section><h2>Raw evidence</h2><div id="raw"></div></section>
<script>const run=decodeURIComponent(location.pathname.slice(5));const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const raw=(p)=>'/raw?run='+encodeURIComponent(run)+'&path='+encodeURIComponent(p);const artifact=(p)=>'/artifact?run='+encodeURIComponent(run)+'&path='+encodeURIComponent(p);async function tick(){const d=await fetch('/api/detail?run='+encodeURIComponent(run)+'&t='+Date.now()).then(r=>r.json());document.querySelector('#title').textContent=`${d.episode.round} · ${run} · ${d.episode.status}`;const row=d.operator_context[0]||{},text=(row.response_text_blocks||[]).join('\n'),meta=(()=>{try{return JSON.parse(text)}catch(_){return {}}})();const labels=(meta.images||[]).map(x=>x.label);document.querySelector('#saw').innerHTML=`<pre>${esc(text)}</pre><div class="images">${(row.response_image_paths||[]).map((p,i)=>`<figure><img src="${artifact(p)}"><figcaption>${esc(labels[i]||p)}</figcaption></figure>`).join('')}</div><details><summary>exact operator_context row</summary><pre>${esc(JSON.stringify(row,null,2))}</pre></details>`;document.querySelector('#timeline').innerHTML=(d.timeline||[]).map(x=>`<div class="event"><b>${esc(x.event_type)}</b> · ${esc(x.timestamp||(x.event_index??''))}<pre>${esc(JSON.stringify(x,null,2))}</pre></div>`).join('');document.querySelector('#reaction').innerHTML=`<h3>Original final answer</h3><pre>${esc(d.agent_final)}</pre><h3>Visible assistant messages</h3><pre>${esc(JSON.stringify(d.codex_summary.visible_assistant_messages||[],null,2))}</pre><p>tool calls: ${esc(d.episode.tool_call_count)} · duration: ${esc(d.episode.duration_seconds??'')} s · status: ${esc(d.episode.status)}</p>${d.codex_summary.usage?`<h3>Token usage from original events</h3><pre>${esc(JSON.stringify(d.codex_summary.usage,null,2))}</pre>`:''}`;document.querySelector('#host').textContent=JSON.stringify(d.host_only,null,2);document.querySelector('#raw').innerHTML=d.raw_files.map(p=>`<a target="_blank" href="${raw(p)}">${esc(p)}</a>`).join('<br>')}tick();setInterval(tick,2000)</script>"""

# Keep the JavaScript string escape intact inside the Python triple-quoted HTML.
DETAIL_HTML = DETAIL_HTML.replace("join('\n')", r"join('\n')")

R10_OPERATION_HTML = """<!doctype html><meta charset="utf-8"><title>R1.0 UniVTAC Operation</title>
<style>body{font:15px/1.55 system-ui;background:#101216;color:#e6e9ef;margin:24px auto;max-width:1300px;padding:0 18px}a{color:#8fd3ff}section,.card{background:#181c22;border:1px solid #303744;border-radius:10px;padding:15px;margin:12px 0}table{border-collapse:collapse;width:100%}th,td{border:1px solid #303744;padding:8px;text-align:left}.ok{color:#76db8b}.failed{color:#ff8585}.images{display:grid;grid-template-columns:repeat(2,1fr);gap:8px}.images img{width:100%;max-height:390px;object-fit:contain;background:#000}.timeline{border-left:4px solid #5b8bad}.choice{border-left-color:#b07cff}.execution{border-left-color:#55b8ff}.host{border-color:#9a7030}.host-only{color:#f0b55a}pre{white-space:pre-wrap;overflow:auto;background:#0c0e12;padding:9px}.plain{font-size:17px;background:#222934;border-left:4px solid #76db8b;padding:12px}.attempt{color:#98a3b3}</style>
<a href="/">← 实验列表</a><h1>R1.0：第一次 Codex 真实操作</h1><p class="plain" id="plain">加载中……</p>
<section><h2>Native Expert Development Baseline</h2><p>这不是论文成功率，而是三个固定 seed 的开发基线。</p><h3 id="score"></h3><table><thead><tr><th>seed</th><th>plan</th><th>check_success</th><th>early_stop</th><th>final outcome</th><th>native segments</th></tr></thead><tbody id="experts"></tbody></table></section>
<section><h2>Codex Operation</h2><div id="attempts"></div></section>
<script>
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const art=(run,p)=>'/artifact?run='+encodeURIComponent(run)+'&path='+encodeURIComponent(p);
const imageLabels=['Head RGB','Wrist RGB','Left tactile rgb_marker','Right tactile rgb_marker'];
const images=(run,paths)=>`<div class="images">${(paths||[]).map((p,i)=>`<figure><img src="${art(run,p)}"><figcaption>${esc(imageLabels[i]||p)}</figcaption></figure>`).join('')}</div>`;
const changed=(run,row)=>{const paths=row.response_image_paths||[];return `<div class="images">${paths.flatMap((p,i)=>{const before=p.replace('/post/','/pre/');return [`<figure><img src="${art(run,before)}"><figcaption>Before · ${esc(imageLabels[i])}</figcaption></figure>`,`<figure><img src="${art(run,p)}"><figcaption>After · ${esc(imageLabels[i])}</figcaption></figure>`]}).join('')}</div>`};
async function tick(){
 const d=await fetch('/api/r10-operation?t='+Date.now()).then(r=>r.json()),s=d.expert_summary;
 document.querySelector('#score').textContent=`${s.native_expert_success_count} / 3 native expert episodes succeeded`;
 document.querySelector('#experts').innerHTML=(s.results||[]).map(x=>`<tr><td>${esc(x.seed)}</td><td>${esc(x.plan_success)}</td><td>${esc(x.native_check_success)}</td><td>${esc(x.native_check_early_stop)}</td><td class="${x.expert_episode_success?'ok':'failed'}">${esc(x.expert_episode_success)}</td><td>${esc((x.semantic_segments||[]).join(' → '))}</td></tr>`).join('');
 const completed=(d.agent_attempts||[]).find(x=>x.episode.status==='completed');
 document.querySelector('#plain').textContent=completed?'Codex 先看了相机和双侧触觉，再自己选择三个受限技能；每做一步都重新观察，最后原生任务判定成功。':'尚无完成的 Codex 操作。';
 document.querySelector('#attempts').innerHTML=(d.agent_attempts||[]).map(a=>{const e=a.episode,ctx=a.operator_context||[],exec=ctx.filter(x=>x.tool==='execute_skill'),initial=ctx.find(x=>x.tool==='observe'),final=a.final_result||{};return `<article class="card"><h3>Attempt ${esc(e.attempt)} · seed ${esc(e.seed)} · <span class="${e.status==='completed'?'ok':'failed'}">${esc(e.status)}</span></h3><p class="attempt">model ${esc(e.model)} · tools ${(e.mcp_tools||[]).map(esc).join(', ')}</p>${e.status!=='completed'?`<p>这次在真实动作前失败，world-changing actions=${esc(e.agent_action_count)}；证据保留，但它不是任务结果。</p>`:''}${initial?`<div class="timeline"><h3>1. Agent Saw</h3><p><b>ACTUAL MCP CONTEXT SEEN BY CODEX</b></p><pre>${esc((initial.response_text_blocks||[]).join('\n'))}</pre>${images(a.run,initial.response_image_paths)}</div>`:''}<div class="card choice"><h3>2. Agent Chose</h3><p>${exec.length?exec.map(x=>esc(x.arguments.skill)).join(' → '):'没有执行动作'}</p></div>${exec.map((x,i)=>{const t=(a.action_trace||[])[i]||{};return `<div class="card execution"><h3>Step ${i+1}: ${esc(x.arguments.skill)}</h3><p><b>Environment Executed</b> · simulator steps ${esc((t.simulator_step_range||[]).join(' → '))} · plan ${esc(t.plan_success_after)} · move ${esc(t.move_returned)}</p><details><summary>native action evidence</summary><pre>${esc(JSON.stringify(t.native_actions_after||[],null,2))}</pre></details><h4>What Changed</h4>${changed(a.run,x)}</div>`}).join('')}<div class="card"><h3>3. Agent Reaction</h3><pre>${esc(a.agent_final)}</pre></div><div class="card host"><h3 class="host-only">HOST-ONLY Outcome — NOT SHOWN TO AGENT</h3><p>plan=${esc(final.plan_success)} · native success=${esc(final.native_check_success)} · early stop=${esc(final.native_check_early_stop)} · final=${esc(final.agent_operation_smoke_success)}</p></div></article>`}).join('');
}
tick();setInterval(tick,2000)
</script>"""

# Keep the JavaScript string escape intact inside the Python triple-quoted HTML.
R10_OPERATION_HTML = R10_OPERATION_HTML.replace("join('\n')", r"join('\n')")

R11_ICL_HTML = """<!doctype html><meta charset="utf-8"><title>R1.1 Tactile-Action ICL</title>
<style>body{font:14px/1.5 system-ui;background:#101216;color:#e6e9ef;margin:20px}a{color:#8fd3ff}.summary,.cell{background:#181c22;border:1px solid #303744;border-radius:9px;padding:12px}.grid{display:grid;grid-template-columns:repeat(4,minmax(280px,1fr));gap:10px}.images{display:grid;grid-template-columns:1fr 1fr;gap:5px}.images img{width:100%;background:#000}.ok{color:#76db8b}.failed{color:#ff8585}.host{border:1px solid #9a7030;padding:8px;margin-top:8px}.host b{color:#f0b55a}pre{white-space:pre-wrap;overflow:auto;background:#0c0e12;padding:7px}.metric{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.metric div{background:#222934;padding:10px}</style>
<a href="/">← 实验列表</a><h1>R1.1：第一次真实 Tactile-Action ICL</h1><section class="summary"><p id="plain"></p><div class="metric" id="metrics"></div><pre id="gains"></pre></section><h2>3 个 query 状态 × 4 个条件</h2><div class="grid" id="grid"></div>
<script>
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const art=(run,p)=>'/artifact?run='+encodeURIComponent(run)+'&path='+encodeURIComponent(p);
const imgs=(run,paths)=>`<div class="images">${(paths||[]).map(p=>`<figure><img src="${art(run,p)}"><figcaption>${esc(p.split('/').slice(-2).join('/'))}</figcaption></figure>`).join('')}</div>`;
const labels={no_demo_multimodal:'No Demo',correct_icl_multimodal:'Correct Multimodal ICL',action_swapped_icl_multimodal:'Action-Swapped ICL',correct_icl_visual_only:'Correct Vision-Only ICL'};
async function tick(){const d=await fetch('/api/r11-icl?t='+Date.now()).then(r=>r.json()),s=d.summary,m=s.condition_metrics;
 document.querySelector('#plain').innerHTML=`<b>一句话：</b> 正确 ICL 的首步是 ${m.correct_icl_multimodal.first_skill_correct_count}/3，与无 demo 的 ${m.no_demo_multimodal.first_skill_correct_count}/3 相同；本轮结论为 <b>${esc(s.icl_interpretation)}</b>。`;
 document.querySelector('#metrics').innerHTML=Object.entries(labels).map(([k,v])=>`<div><b>${v}</b><br>first skill ${m[k].first_skill_correct_count}/3<br>exact ${m[k].exact_sequence_count}/3<br>native success ${m[k].native_continuation_success_count}/3</div>`).join('');document.querySelector('#gains').textContent=JSON.stringify(s.gains,null,2);
 document.querySelector('#grid').innerHTML=d.cells.map(c=>{const review=c.operator_context.find(x=>x.tool==='review_demonstrations')||{},obs=c.operator_context.find(x=>x.tool==='observe')||{},exec=c.operator_context.filter(x=>x.tool==='execute_skill'),r=c.result,h=c.condition.host_only;return `<article class="cell"><h3>seed ${c.episode.seed} · ${labels[r.condition]}</h3><p>query: ${esc(r.query_start_state)} · support: ${esc(r.support_seed)}</p><details><summary>Support demonstrations (${review.response_image_paths?.length||0})</summary><pre>${esc((review.response_text_blocks||[]).join('\n'))}</pre>${imgs(c.run,review.response_image_paths)}</details><h4>Agent Saw</h4>${imgs(c.run,obs.response_image_paths)}<h4>Agent chose</h4><p>${esc(r.agent_skill_sequence.join(' → ')||'no action')}</p>${exec.map((x,i)=>`<details><summary>Action ${i+1} after-observation</summary>${imgs(c.run,x.response_image_paths)}</details>`).join('')}<p class="${r.first_skill_correct?'ok':'failed'}">first skill correct: ${r.first_skill_correct}</p><p>exact sequence: ${r.exact_sequence} · native success: ${r.native_continuation_success} · Codex completed: ${r.codex_completed}</p><div class="host"><b>HOST-ONLY — NOT SHOWN TO AGENT</b><p>prefix: ${esc(h.native_prefix.join(' → ')||'none')}<br>expected: ${esc(h.expected_remaining_sequence.join(' → '))}<br>actual native: ${esc((c.final_result.host_only?.selected_native_sequence||[]).join(' → '))}<br>mapping: ${esc(JSON.stringify(h.opaque_to_native))}<br>native evaluator available: ${r.native_evaluation_available}</p></div><details><summary>Raw Codex answer</summary><pre>${esc(c.agent_final)}</pre></details></article>`}).join('');}
tick();setInterval(tick,2000)
</script>"""

R11_ICL_HTML = R11_ICL_HTML.replace("join('\n')", r"join('\n')")

R12_BRANCHING_HTML = """<!doctype html><meta charset="utf-8"><title>R1.2 Branching Qualification</title>
<style>body{font:14px/1.5 system-ui;background:#101216;color:#e6e9ef;margin:20px}a{color:#8fd3ff}.summary,.cell{background:#181c22;border:1px solid #303744;border-radius:9px;padding:12px}.grid{display:grid;grid-template-columns:repeat(3,minmax(280px,1fr));gap:10px}.images{display:grid;grid-template-columns:1fr 1fr;gap:5px}.images img{width:100%;background:#000}.ok{color:#76db8b}.failed{color:#ff8585}.host{border:1px solid #9a7030;padding:8px;margin-top:8px}.host b{color:#f0b55a}pre{white-space:pre-wrap;overflow:auto;background:#0c0e12;padding:7px}.tasks{display:grid;grid-template-columns:1fr 1fr;gap:10px}</style>
<a href="/">← 实验列表</a><h1>R1.2：筛选真正需要触觉分支的任务</h1><section class="summary"><p><b>说人话：</b>我们先让原生专家做 Lift Bottle 和 Insert Hole，检查不同初始状态是否真的需要不同动作，以及选错动作是否会让任务失败。本轮没有 Codex，也没有 ICL。</p><div class="tasks" id="tasks"></div><p id="selection"></p></section><h2>六个固定原生 expert episode</h2><div class="grid" id="experts"></div><h2>同一状态：正确动作 vs 错误动作</h2><div class="grid" id="counterfactual"></div>
<script>
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const art=(run,p)=>'/artifact?run='+encodeURIComponent(run)+'&path='+encodeURIComponent(p);
const imgs=c=>`<div class="images">${(c.image_paths||[]).map((p,i)=>`<figure><img src="${art(c.run,p)}"><figcaption>${esc((c.image_labels||[])[i]||p)}</figcaption></figure>`).join('')}</div>`;
const card=c=>`<article class="cell"><h3>${esc(c.task)} · seed ${esc(c.seed)} · ${esc(c.condition)}</h3><p><b>决策时实际可见的四张图</b></p>${imgs(c)}<p>实际执行：${esc((c.semantic_segments||[]).join(' → '))}</p><p class="${c.expert_episode_success?'ok':'failed'}">native success: ${esc(c.expert_episode_success)}</p><div class="host"><b>HOST-ONLY DECISION CLASS — NOT SHOWN TO AGENT</b><p>${esc(c.host_only?.decision_class)}</p>${c.condition!=='expert'?`<p><b>这次干预：</b>原生建议 x=${esc(c.intervention.native_x_move)}，实际施加 x=${esc(c.intervention.applied_x_move)}；z=${esc(c.intervention.applied_z_move)}。${c.condition==='wrong'?'错误组只把 x 方向反过来。':'正确组照原生建议执行。'}</p>`:''}<pre>${esc(JSON.stringify(c.host_only,null,2))}</pre></div></article>`;
async function tick(){const d=await fetch('/api/r12-branching?t='+Date.now()).then(r=>r.json()),s=d.summary,t=s.task_summaries;
 document.querySelector('#tasks').innerHTML=Object.values(t).map(x=>`<div class="cell"><b>${esc(x.task)}</b><p>expert success ${esc(x.native_expert_success_count)}/3<br>分支种类 ${esc(x.decision_class_count)}：${esc(JSON.stringify(x.decision_class_distribution))}<br>候选任务：${esc(x.branching_candidate)}</p></div>`).join('');
 document.querySelector('#selection').innerHTML=`<b>最终选择：</b> ${esc(s.selected_tactile_icl_task)} · ${esc(s.selection_reason)}`;
 document.querySelector('#experts').innerHTML=d.experts.map(card).join('');document.querySelector('#counterfactual').innerHTML=d.counterfactual.length?d.counterfactual.map(card).join(''):'<p>没有任务通过候选门槛，因此没有运行反事实。</p>';
}tick();setInterval(tick,2000)
</script>"""

R13_ICL_HTML = """<!doctype html><meta charset="utf-8"><title>R1.3 Insert Hole Tactile ICL</title>
<style>body{font:14px/1.5 system-ui;background:#101216;color:#e6e9ef;margin:20px}a{color:#8fd3ff}.summary,.cell{background:#181c22;border:1px solid #303744;border-radius:9px;padding:12px}.grid{display:grid;grid-template-columns:repeat(4,minmax(300px,1fr));gap:10px}.images{display:grid;grid-template-columns:1fr 1fr;gap:4px}.images img{width:100%;background:#000}.ok{color:#76db8b}.failed{color:#ff8585}.host{border:1px solid #9a7030;padding:8px;margin-top:8px}.host b{color:#f0b55a}pre{white-space:pre-wrap;overflow:auto;background:#0c0e12;padding:7px}.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:8px}.metrics div{background:#222934;padding:9px}</style>
<a href="/">← 实验列表</a><h1>R1.3：Insert Hole 接触前后 Tactile-Action ICL</h1><section class="summary"><p><b>说人话：</b>Codex 只能在两个互斥程序里选一个。我们比较不看示范、看正确触觉示范、看动作标签被交换的示范、以及只看视觉示范；选完立即在 fresh simulator 中执行，选错不补救。</p><div class="metrics" id="metrics"></div><p id="conclusion"></p></section><h2>3 个 query seed × 4 个条件</h2><div class="grid" id="grid"></div>
<script>
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const art=(run,p)=>'/artifact?run='+encodeURIComponent(run)+'&path='+encodeURIComponent(p);
const imgs=(run,paths)=>`<div class="images">${(paths||[]).map(p=>`<figure><img src="${art(run,p)}"><figcaption>${esc(p.split('/').slice(-3).join('/'))}</figcaption></figure>`).join('')}</div>`;
const names={no_demo_multimodal:'No Demo',correct_icl_multimodal:'Correct Multimodal ICL',action_swapped_icl_multimodal:'Action-Swapped ICL',correct_icl_vision_only:'Correct Vision-Only ICL'};
async function tick(){const d=await fetch('/api/r13-icl?t='+Date.now()).then(r=>r.json()),s=d.summary,m=s.condition_metrics;
 document.querySelector('#metrics').innerHTML=Object.entries(names).map(([k,v])=>`<div><b>${v}</b><br>选向 ${m[k].correction_selection_correct_count}/3<br>native success ${m[k].native_continuation_success_count}/3</div>`).join('');
 document.querySelector('#conclusion').innerHTML=`<b>结论：</b>${esc(s.action_icl_signal)}；触觉特异性：${esc(s.tactile_icl_signal)}。正确 ICL 相比 no-demo=${esc(s.gains.correct_icl_gain_over_no_demo)}，相比 swapped=${esc(s.gains.correct_icl_gain_over_swapped)}，相比 vision-only=${esc(s.gains.tactile_gain_over_vision_only)}。`;
 document.querySelector('#grid').innerHTML=d.cells.map(c=>`<article class="cell"><h3>seed ${c.seed} · ${names[c.condition]}</h3><details><summary>Agent 真实看到的 support (${c.support_images.length} 张)</summary>${imgs(c.decision_run,c.support_images)}<pre>${esc(c.support_text.join('\n'))}</pre></details><h4>Canonical T0/T1 query (${c.query_images.length} 张)</h4>${imgs(c.decision_run,c.query_images)}<pre>${esc(c.query_text.join('\n'))}</pre><h4>Codex 只选一次</h4><p>${esc(c.selected_skill)} · confidence=${esc(c.confidence)}</p><p>${esc(c.reason)}</p><h4>Fresh physical after</h4>${imgs(c.execution_run,c.after_images)}<p class="${c.selection_correct?'ok':'failed'}">选向正确：${esc(c.selection_correct)}</p><p class="${c.native_success?'ok':'failed'}">native success：${esc(c.native_success)}</p><div class="host"><b>HOST-ONLY — NOT SHOWN TO CODEX</b><p>真实 class=${esc(c.host_only.canonical_decision_class)}；expected=${esc(c.host_only.expected_skill)}；execution mismatch=${esc(c.execution_state_mismatch)}</p><pre>${esc(JSON.stringify(c.intervention,null,2))}</pre></div></article>`).join('');
}tick();setInterval(tick,2000)
</script>"""

R13_ICL_HTML = R13_ICL_HTML.replace("join('\n')", r"join('\n')")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def discover_runs(runs_root: Path) -> list[dict[str, Any]]:
    root = runs_root.expanduser().resolve()
    runs: list[dict[str, Any]] = []
    for episode_path in sorted(root.rglob("episode.json")):
        episode_root = episode_path.parent
        episode = _load_json(episode_path)
        rows = read_jsonl(episode_root / "operator_context.jsonl")
        runs.append(
            {
                "directory": episode_root.relative_to(root).as_posix(),
                "round": episode.get("round", episode_root.name),
                "task": episode.get("task"),
                "seed": episode.get("seed"),
                "model": episode.get("model"),
                "status": episode.get("status", "unknown"),
                "started_at": episode.get("started_at"),
                "duration_seconds": episode.get("duration_seconds"),
                "observe_count": sum(row.get("tool") == "observe" for row in rows),
            }
        )
    return runs


def discover_pilots(runs_root: Path) -> list[dict[str, Any]]:
    root = runs_root.expanduser().resolve()
    pilots = []
    for path in sorted(root.rglob("pilot.json")):
        pilot_root = path.parent
        pilot = _load_json(path)
        summary = _load_json(pilot_root / "summary.json")
        if pilot.get("round") in {
            "R0.9.14",
            "R0.9.15",
            "R0.9.16",
            "R0.9.17",
            "R0.9.18",
            "R0.9.19",
            "R0.9.20",
        }:
            pilots.append(
                {
                    "directory": pilot_root.relative_to(root).as_posix(),
                    "signal": summary.get(
                        "minimal_skill_candidate",
                        summary.get(
                        "mechanism_result",
                        summary.get(
                        "grounding_signal",
                        summary.get(
                            "unilateral_signal",
                            summary.get(
                                "structured_signal",
                                summary.get(
                                    "difference_signal", summary.get("pilot_signal")
                                ),
                            ),
                        ))),
                    ),
                    "completed_call_count": summary.get(
                        "completed_semantic_trials", summary.get("completed_call_count", 0)
                    ),
                }
            )
    return pilots


def load_project_progress(runs_root: Path) -> dict[str, Any]:
    return {
        "entries": read_jsonl(
            runs_root.expanduser().resolve() / "univtac-project-dashboard/progress.jsonl"
        ),
        "pilots": discover_pilots(runs_root),
    }


def load_r10_operation_detail(runs_root: Path) -> dict[str, Any]:
    operation_root = runs_root.expanduser().resolve() / "univtac-isaac51-r10"
    if not (operation_root / "expert_summary.json").is_file():
        raise FileNotFoundError("R1.0 operation evidence is unavailable")
    expert_episodes = []
    for episode_path in sorted((operation_root / "expert").glob("seed_*/episode.json")):
        episode_root = episode_path.parent
        expert_episodes.append(
            {
                "run": episode_root.relative_to(runs_root.resolve()).as_posix(),
                "episode": _load_json(episode_path),
                "action_trace": read_jsonl(episode_root / "action_trace.jsonl"),
                "final_result": _load_json(episode_root / "final_result.json"),
            }
        )
    agent_attempts = []
    for episode_path in (operation_root / "agent").rglob("episode.json"):
        episode_root = episode_path.parent
        episode = _load_json(episode_path)
        episode.setdefault("attempt", 1)
        agent_attempts.append(
            {
                "run": episode_root.relative_to(runs_root.resolve()).as_posix(),
                "episode": episode,
                "operator_context": read_jsonl(episode_root / "operator_context.jsonl"),
                "action_trace": read_jsonl(episode_root / "action_trace.jsonl"),
                "final_result": _load_json(episode_root / "final_result.json"),
                "agent_final": (
                    (episode_root / "agent_final.md").read_text(encoding="utf-8")
                    if (episode_root / "agent_final.md").is_file()
                    else ""
                ),
            }
        )
    agent_attempts.sort(key=lambda row: int(row["episode"].get("attempt", 0)))
    return {
        "run_manifest": _load_json(operation_root / "run_manifest.json"),
        "expert_summary": _load_json(operation_root / "expert_summary.json"),
        "expert_episodes": expert_episodes,
        "agent_attempts": agent_attempts,
    }


def load_r11_icl_detail(runs_root: Path) -> dict[str, Any]:
    root = runs_root.expanduser().resolve()
    pilot_root = root / "univtac-isaac51-r11"
    if not (pilot_root / "summary.json").is_file():
        raise FileNotFoundError("R1.1 ICL evidence is unavailable")
    condition_order = {
        "no_demo_multimodal": 0,
        "correct_icl_multimodal": 1,
        "action_swapped_icl_multimodal": 2,
        "correct_icl_visual_only": 3,
    }
    cells = []
    for result_path in pilot_root.glob("episodes/seed_*/*/attempt_*/result.json"):
        episode_root = result_path.parent
        condition = _load_json(episode_root / "condition.json")
        cells.append(
            {
                "run": episode_root.relative_to(root).as_posix(),
                "episode": _load_json(episode_root / "episode.json"),
                "condition": condition,
                "result": _load_json(result_path),
                "operator_context": read_jsonl(episode_root / "operator_context.jsonl"),
                "action_trace": read_jsonl(episode_root / "action_trace.jsonl"),
                "host_setup": read_jsonl(
                    episode_root / "host_setup/action_trace.jsonl"
                ),
                "final_result": _load_json(episode_root / "final_result.json"),
                "agent_final": (
                    (episode_root / "agent_final.md").read_text(encoding="utf-8")
                    if (episode_root / "agent_final.md").is_file()
                    else "unavailable"
                ),
            }
        )
    cells.sort(
        key=lambda cell: (
            int(cell["episode"]["seed"]),
            condition_order[cell["result"]["condition"]],
        )
    )
    if len(cells) != 12:
        raise ValueError(f"R1.1 dashboard requires 12 cells, found {len(cells)}")
    return {
        "summary": _load_json(pilot_root / "summary.json"),
        "run_manifest": _load_json(pilot_root / "run_manifest.json"),
        "cells": cells,
    }


def _decision_images(operator_visible: dict[str, Any]) -> tuple[list[str], list[str]]:
    paths: list[str] = []
    labels: list[str] = []
    cameras = operator_visible.get("cameras", {})
    for camera in ("head", "wrist"):
        item = cameras.get(camera, {}).get("rgb", {})
        if item.get("path"):
            paths.append(str(item["path"]))
            labels.append(f"camera/{camera}/rgb")
    tactile = operator_visible.get("tactile", {})
    for sensor in sorted(tactile):
        item = tactile[sensor].get("rgb_marker", {})
        if item.get("path"):
            paths.append(str(item["path"]))
            labels.append(f"tactile/{sensor}/rgb_marker")
    return paths, labels


def load_r12_branching_detail(runs_root: Path) -> dict[str, Any]:
    root = runs_root.expanduser().resolve()
    run_root = root / "univtac-isaac51-r12"
    summary = _load_json(run_root / "summary.json")
    if not summary:
        raise FileNotFoundError("R1.2 branching evidence is unavailable")

    def load_cell(episode_root: Path) -> dict[str, Any]:
        decision = _load_json(episode_root / "decision_state.json")
        final = _load_json(episode_root / "final_result.json")
        operator_visible = decision.get("operator_visible", {})
        image_paths, image_labels = _decision_images(operator_visible)
        if len(image_paths) != 4:
            raise ValueError(
                f"R1.2 decision cell requires four images: {episode_root}"
            )
        return {
            "run": episode_root.relative_to(root).as_posix(),
            "task": final.get("task"),
            "seed": final.get("seed"),
            "condition": final.get("condition"),
            "operator_visible": operator_visible,
            "host_only": decision.get("host_only", {}),
            "image_paths": image_paths,
            "image_labels": image_labels,
            "semantic_segments": final.get("semantic_segments", []),
            "expert_episode_success": final.get("expert_episode_success"),
            "intervention": {
                "native_x_move": final.get("native_x_move"),
                "native_z_move": final.get("native_z_move"),
                "applied_x_move": final.get("applied_x_move"),
                "applied_z_move": final.get("applied_z_move"),
                "corrective_moves_executed": final.get("corrective_moves_executed"),
            },
            "final_result": final,
        }

    experts = [
        load_cell(path.parent)
        for path in sorted(run_root.glob("*/seed_*/expert/episode.json"))
    ]
    if len(experts) != 6:
        raise ValueError(f"R1.2 dashboard requires six expert cells, found {len(experts)}")
    counterfactual = [
        load_cell(path.parent)
        for path in sorted(run_root.glob("counterfactual/*/seed_*/*/episode.json"))
    ]
    if len(counterfactual) not in {0, 2}:
        raise ValueError("R1.2 dashboard requires zero or two counterfactual cells")
    return {
        "summary": summary,
        "run_manifest": _load_json(run_root / "run_manifest.json"),
        "experts": experts,
        "counterfactual": counterfactual,
    }


def load_r13_icl_detail(runs_root: Path) -> dict[str, Any]:
    root = runs_root.expanduser().resolve()
    run_root = root / "univtac-isaac51-r13"
    summary = _load_json(run_root / "summary.json")
    if not summary:
        raise FileNotFoundError("R1.3 ICL evidence is unavailable")
    cells = []
    order = {
        "no_demo_multimodal": 0,
        "correct_icl_multimodal": 1,
        "action_swapped_icl_multimodal": 2,
        "correct_icl_vision_only": 3,
    }
    for result_path in run_root.glob("decisions/seed_*/*/decision_result.json"):
        decision_root = result_path.parent
        decision = _load_json(result_path)
        seed = int(decision["seed"])
        condition = str(decision["condition"])
        execution_root = run_root / "executions" / f"seed_{seed}" / condition
        physical = _load_json(execution_root / "r13_result_row.json")
        final_snapshot = _load_json(execution_root / "snapshot_final.json")
        after_images, _ = _decision_images(
            final_snapshot.get("operator_visible", {})
        )
        context = read_jsonl(decision_root / "operator_context.jsonl")
        review = next((row for row in context if row.get("tool") == "review_demonstrations"), {})
        query = next((row for row in context if row.get("tool") == "observe_query"), {})
        choose = next((row for row in context if row.get("tool") == "choose_skill"), {})
        host = _load_json(decision_root / "condition.json").get("host_only", {})
        cells.append(
            {
                "seed": seed,
                "condition": condition,
                "decision_run": decision_root.relative_to(root).as_posix(),
                "execution_run": execution_root.relative_to(root).as_posix(),
                "support_images": review.get("response_image_paths", []),
                "support_text": review.get("response_text_blocks", []),
                "query_images": query.get("response_image_paths", []),
                "query_text": query.get("response_text_blocks", []),
                "selected_skill": choose.get("arguments", {}).get("skill"),
                "confidence": choose.get("arguments", {}).get("confidence"),
                "reason": choose.get("arguments", {}).get("reason"),
                "selection_correct": decision.get("selection_correct"),
                "native_success": physical.get("native_episode_success"),
                "execution_state_mismatch": physical.get("execution_state_mismatch"),
                "host_only": host,
                "intervention": {
                    "native_x_move": physical.get("native_x_move"),
                    "native_z_move": physical.get("native_z_move"),
                    "applied_x_move": physical.get("applied_x_move"),
                    "applied_z_move": physical.get("applied_z_move"),
                },
                "after_images": after_images,
            }
        )
    cells.sort(key=lambda row: (int(row["seed"]), order[str(row["condition"])]))
    if len(cells) != 12:
        raise ValueError(f"R1.3 dashboard requires twelve cells, found {len(cells)}")
    return {
        "summary": summary,
        "run_manifest": _load_json(run_root / "run_manifest.json"),
        "cells": cells,
    }


def build_unilateral_human_comparisons(
    predictions: list[dict[str, Any]], source_mapping: dict[str, Any]
) -> list[dict[str, Any]]:
    seeds = source_mapping.get("seeds")
    if not isinstance(seeds, dict):
        return []
    by_pair = {(row.get("seed"), row.get("condition")): row for row in predictions}
    comparisons = []
    for seed_text, mapping in sorted(seeds.items(), key=lambda item: int(item[0])):
        seed = int(seed_text)
        expected = mapping["expected_image_change_side"]
        opposite = "right" if expected == "left" else "left"
        image = by_pair.get((seed, "image_only"), {})
        correct = by_pair.get((seed, "correct_structured_guidance"), {})
        swapped = by_pair.get((seed, "swapped_structured_guidance"), {})
        swapped_side = swapped.get("tactile_changed_side")
        conflict_reaction = (
            "跟随图片"
            if swapped_side == expected
            else "跟随错误文字"
            if swapped_side == opposite
            else "没有明确跟随任一侧"
        )
        comparisons.append(
            {
                "seed": seed,
                "baseline_side": opposite,
                "expected_change_side": expected,
                "image_only_side": image.get("tactile_changed_side", "unavailable"),
                "correct_guidance_side": correct.get(
                    "tactile_changed_side", "unavailable"
                ),
                "correct_guidance_helped": (
                    correct.get("tactile_changed_side") == expected
                    and image.get("tactile_changed_side") != expected
                ),
                "swapped_guidance_side": swapped_side or "unavailable",
                "conflict_reaction": conflict_reaction,
            }
        )
    return comparisons


def load_pilot_detail(pilot_root: Path) -> dict[str, Any]:
    pilot = _load_json(pilot_root / "pilot.json")
    round_name = pilot.get("round")
    predictions = read_jsonl(
        pilot_root
        / (
            "final_predictions.jsonl"
            if round_name == "R0.9.18"
            else "predictions.jsonl"
        )
    )
    pair_key = "protocol" if round_name in {"R0.9.19", "R0.9.20"} else "condition"
    by_pair = {(row.get("seed"), row.get(pair_key)): row for row in predictions}
    cells = []
    for episode_path in sorted((pilot_root / "runs").glob("seed_*/*/episode.json")):
        run_root = episode_path.parent
        episode = _load_json(episode_path)
        rows = read_jsonl(run_root / "operator_context.jsonl")
        row = rows[0] if rows else {}
        text_blocks = row.get("response_text_blocks", [])
        try:
            visible_payload = json.loads(text_blocks[0]) if text_blocks else {}
        except (TypeError, json.JSONDecodeError):
            visible_payload = {}
        guidance_payload = {}
        guidance_row = next(
            (item for item in rows if item.get("tool") == "observe_structured_guidance"),
            None,
        )
        if guidance_row and guidance_row.get("response_text_blocks"):
            try:
                guidance_payload = json.loads(guidance_row["response_text_blocks"][0])
            except (TypeError, json.JSONDecodeError):
                guidance_payload = {}
        image_judgment = _load_json(run_root / "image_judgment.json")
        condition = episode.get(pair_key)
        prediction = by_pair.get((episode.get("seed"), condition), {})
        staged = round_name == "R0.9.18"
        mechanism = round_name in {"R0.9.19", "R0.9.20"}
        manifest = _load_json(run_root / "condition.json")
        expected_side = manifest.get("expected_image_side")
        text_side = manifest.get("text_indicated_side")
        relative_run = run_root.relative_to(pilot_root.parent).as_posix()
        cells.append(
            {
                "seed": episode.get("seed"),
                "condition": condition,
                "protocol": episode.get("protocol"),
                "run": relative_run,
                "image_paths": row.get("response_image_paths", []),
                "image_labels": [
                    item.get("label") for item in visible_payload.get("images", [])
                ],
                "mcp_text_blocks": [
                    block
                    for trace_row in rows
                    for block in trace_row.get("response_text_blocks", [])
                ],
                "structured_summary": guidance_payload.get(
                    "tactile_change_summary",
                    visible_payload.get("tactile_change_summary"),
                ),
                "image_judgment": image_judgment,
                "prediction": prediction,
                "phases": [
                    {
                        "name": "Agent Saw Images",
                        "timestamp": row.get("timestamp_s"),
                    },
                    {
                        "name": "Image-Only Judgment Committed",
                        "timestamp": rows[1].get("timestamp_s") if len(rows) > 1 else None,
                    },
                    {
                        "name": "Structured Guidance Revealed",
                        "timestamp": rows[2].get("timestamp_s") if len(rows) > 2 else None,
                    },
                    {"name": "Final Reaction", "timestamp": episode.get("ended_at")},
                ]
                if staged
                else [
                    {"name": name, "timestamp": timestamp}
                    for name, timestamp in (
                        ("Agent Saw Images", row.get("timestamp_s")),
                        (
                            "Image-Only Judgment Committed",
                            next(
                                (
                                    item.get("timestamp_s")
                                    for item in rows
                                    if item.get("tool") == "record_image_judgment"
                                ),
                                None,
                            ),
                        ),
                        (
                            "Structured Guidance Revealed",
                            guidance_row.get("timestamp_s") if guidance_row else row.get("timestamp_s"),
                        ),
                        ("Final Reaction", episode.get("ended_at")),
                    )
                    if timestamp is not None
                ]
                if mechanism
                else [],
                "provisional_changed": image_judgment.get("tactile_changed_side")
                != prediction.get("final_changed_side")
                if staged
                else None,
                "conflict_detected": prediction.get("guidance_consistency") == "conflicting"
                if staged or mechanism
                else None,
                "follow_image": prediction.get("final_changed_side") == expected_side
                if mechanism
                else None,
                "follow_text": prediction.get("final_changed_side") == text_side
                if mechanism
                else None,
                "raw_answer": (run_root / "agent_final.md").read_text(encoding="utf-8")
                if (run_root / "agent_final.md").is_file()
                else "",
                "duration_seconds": episode.get("duration_seconds"),
                "usage": episode.get("usage"),
                "tool_call_count": episode.get("tool_call_count"),
            }
        )
    condition_rank = {
        "visual_only": 0,
        "correct_tactile": 1,
        "swapped_tactile": 2,
        "raw_pair": 0,
        "explicit_difference": 1,
        "swapped_difference": 2,
        "difference_only": 0,
        "correct_structured_guidance": 1,
        "swapped_structured_guidance": 2,
        "image_only": 0,
        "correct_guidance": 0,
        "swapped_guidance": 1,
        "simultaneous_fusion": 0,
        "image_first_no_commit": 1,
        "image_first_with_commit": 2,
        "simultaneous_conflict_aware": 0,
        "image_first_no_commit_conflict_aware": 1,
    }
    cells.sort(key=lambda cell: (int(cell["seed"]), condition_rank[str(cell["condition"])]))
    pairs = []
    if pilot.get("round") == "R0.9.15":
        for seed in (1_000_000, 1_000_001, 1_000_002):
            raw_run = pilot_root / "runs" / f"seed_{seed}" / "raw_pair"
            difference_run = pilot_root / "runs" / f"seed_{seed}" / "explicit_difference"
            if not raw_run.is_dir() or not difference_run.is_dir():
                continue
            for sensor in ("left_tactile", "right_tactile"):
                pairs.append(
                    {
                        "seed": seed,
                        "sensor": sensor,
                        "images": [
                            {
                                "label": "Baseline",
                                "run": raw_run.relative_to(pilot_root.parent).as_posix(),
                                "path": f"images/{sensor}_baseline.png",
                            },
                            {
                                "label": "Current",
                                "run": raw_run.relative_to(pilot_root.parent).as_posix(),
                                "path": f"images/{sensor}_current.png",
                            },
                            {
                                "label": "Difference",
                                "run": difference_run.relative_to(pilot_root.parent).as_posix(),
                                "path": f"images/{sensor}_difference.png",
                            },
                        ],
                    }
                )
    source_mapping = _load_json(pilot_root / "source_mapping.json")
    summary = _load_json(pilot_root / "summary.json")
    metrics = summary.get("protocol_metrics", {})
    r0918_followed = (
        summary.get("r0918_reference", {})
        .get("conflict_aware_committed_protocol", {})
        .get("follow_image", {})
        .get("followed", 0)
    )
    mechanism_funnel = [
        {
            "name": "All together",
            "followed": metrics.get("simultaneous_fusion", {})
            .get("final_follow_image_rate", {})
            .get("followed", 0),
        },
        {
            "name": "Images first",
            "followed": metrics.get("image_first_no_commit", {})
            .get("final_follow_image_rate", {})
            .get("followed", 0),
        },
        {
            "name": "Explicit commitment",
            "followed": metrics.get("image_first_with_commit", {})
            .get("final_follow_image_rate", {})
            .get("followed", 0),
        },
        {
            "name": "R0.9.18 conflict-aware instruction",
            "followed": r0918_followed,
        },
    ]
    for item in mechanism_funnel:
        item["reached"] = item["followed"] >= 2
    conflict_metrics = summary.get("protocol_metrics", {})
    r0919_reference = summary.get("r0919_reference", {})
    r0918_reference = summary.get("r0918_reference", {})
    interaction_funnel = [
        {
            "name": "Neutral Simultaneous",
            "followed": r0919_reference.get("simultaneous_neutral", {})
            .get("final_follow_image_rate", {})
            .get("followed", 0),
        },
        {
            "name": "Conflict-Aware Simultaneous",
            "followed": conflict_metrics.get("simultaneous_conflict_aware", {})
            .get("final_follow_image_rate", {})
            .get("followed", 0),
        },
        {
            "name": "Neutral Image-First",
            "followed": r0919_reference.get("image_first_no_commit_neutral", {})
            .get("final_follow_image_rate", {})
            .get("followed", 0),
        },
        {
            "name": "Conflict-Aware Image-First",
            "followed": conflict_metrics.get(
                "image_first_no_commit_conflict_aware", {}
            )
            .get("final_follow_image_rate", {})
            .get("followed", 0),
        },
        {
            "name": "Committed + Conflict-Aware",
            "followed": r0918_reference.get("swapped_final_follow_image_rate", {}).get(
                "followed", 0
            ),
        },
    ]
    for item in interaction_funnel:
        item["pilot_sufficient"] = item["followed"] >= 2
        item["robust"] = item["followed"] == 3
    host_expectations = [
        {
            "seed": cell["seed"],
            "protocol": cell["protocol"],
            "expected_image_side": _load_json(
                pilot_root / "runs" / f"seed_{cell['seed']}" / str(cell["protocol"]) / "condition.json"
            ).get("expected_image_side"),
            "text_indicated_side": _load_json(
                pilot_root / "runs" / f"seed_{cell['seed']}" / str(cell["protocol"]) / "condition.json"
            ).get("text_indicated_side"),
        }
        for cell in cells
        if mechanism
    ]
    return {
        "pilot": pilot,
        "summary": summary,
        "host_reference": _load_json(pilot_root / "host_reference.json"),
        "difference_metrics": _load_json(pilot_root / "difference_metrics.json"),
        "structured_reference": _load_json(pilot_root / "structured_reference.json"),
        "secondary_diagnostics": _load_json(pilot_root / "secondary_diagnostics.json"),
        "source_mapping": source_mapping,
        "human_comparisons": build_unilateral_human_comparisons(
            predictions, source_mapping
        ),
        "condition_manifests": [
            _load_json(path)
            for path in sorted((pilot_root / "runs").glob("seed_*/*/condition.json"))
        ],
        "cells": cells,
        "pairs": pairs,
        "mechanism_funnel": mechanism_funnel if mechanism else [],
        "interaction_funnel": interaction_funnel if round_name == "R0.9.20" else [],
        "host_expectations": host_expectations,
    }


def _event_item(event: dict[str, Any]) -> dict[str, Any]:
    if isinstance(event.get("item"), dict):
        return event["item"]
    params = event.get("params")
    if isinstance(params, dict) and isinstance(params.get("item"), dict):
        return params["item"]
    return {}


def build_timeline(
    episode: dict[str, Any],
    codex_rows: list[dict[str, Any]],
    operator_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    timeline: list[dict[str, Any]] = [
        {"event_type": "Codex started", "timestamp": episode.get("codex_started_at")}
    ]
    for index, event in enumerate(codex_rows):
        item = _event_item(event)
        serialized = json.dumps(item, ensure_ascii=False).lower()
        event_type = str(event.get("type") or event.get("method") or "")
        if "observe" in serialized and ("started" in event_type or "requested" in event_type):
            timeline.append(
                {
                    "event_type": "observe tool requested",
                    "event_index": index,
                    "tool_arguments": item.get("arguments", {}),
                }
            )
        if "error" in event_type.lower() or "failed" in event_type.lower():
            timeline.append(
                {
                    "event_type": "MCP/runtime error",
                    "event_index": index,
                    "error": event.get("message") or event.get("error") or event_type,
                }
            )
    for row in operator_rows:
        timeline.append(
            {
                "event_type": "observe tool returned",
                "timestamp": row.get("timestamp_s"),
                "tool_arguments": row.get("arguments", {}),
                "tool_text": row.get("response_text_blocks", []),
                "images": row.get("response_image_paths", []),
            }
        )
    timeline.append({"event_type": "Codex completed", "timestamp": episode.get("codex_ended_at")})
    return timeline


def load_run_detail(episode_root: Path) -> dict[str, Any]:
    episode = _load_json(episode_root / "episode.json")
    operator_rows = read_jsonl(episode_root / "operator_context.jsonl")
    codex_rows = read_jsonl(episode_root / "codex_exec.jsonl")
    seed_root = episode_root / "simulator/pull_out_key_seed1000000"
    raw_files = [
        path
        for path in (
            "episode.json",
            "operator_context.jsonl",
            "codex_exec.jsonl",
            "agent_final.md",
            "lifecycle/simulator_stdout.log",
        )
        if (episode_root / path).is_file()
    ]
    return {
        "episode": episode,
        "operator_context": operator_rows,
        "codex_summary": summarize_codex_exec(codex_rows),
        "timeline": build_timeline(episode, codex_rows, operator_rows),
        "agent_final": (
            (episode_root / "agent_final.md").read_text(encoding="utf-8")
            if (episode_root / "agent_final.md").is_file()
            else ""
        ),
        "host_only": {
            "simulator": _load_json(episode_root / "simulator/summary.json"),
            "contact_candidate": _load_json(seed_root / "contact_summary.json"),
            "lifecycle": _load_json(episode_root / "lifecycle/simulator.json"),
        },
        "raw_files": raw_files,
    }


def _safe_run_root(runs_root: Path, raw: str) -> Path:
    relative = PurePosixPath(unquote(raw))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("invalid run path")
    root = runs_root.resolve()
    candidate = root.joinpath(*relative.parts).resolve(strict=True)
    if not candidate.is_relative_to(root) or not any((candidate / name).is_file() for name in ("episode.json", "run_manifest.json")):
        raise ValueError("run is outside runs root")
    return candidate


def _safe_pilot_root(runs_root: Path, raw: str) -> Path:
    relative = PurePosixPath(unquote(raw))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("invalid pilot path")
    root = runs_root.resolve()
    candidate = root.joinpath(*relative.parts).resolve(strict=True)
    if not candidate.is_relative_to(root) or not (candidate / "pilot.json").is_file():
        raise ValueError("pilot is outside runs root")
    return candidate


def _safe_artifact(episode_root: Path, raw: str) -> Path:
    relative = PurePosixPath(unquote(raw))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("invalid artifact path")
    candidate = episode_root.joinpath(*relative.parts).resolve(strict=True)
    if not candidate.is_relative_to(episode_root.resolve()) or not candidate.is_file():
        raise ValueError("artifact is outside episode root")
    return candidate


def make_handler(runs_root: Path) -> type[BaseHTTPRequestHandler]:
    root = runs_root.expanduser().resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args: object) -> None:
            return

        def _send(self, data: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _json(self, payload: Any, status: int = 200) -> None:
            self._send(
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                "application/json; charset=utf-8",
                status,
            )

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            query = parse_qs(parsed.query)
            try:
                if parsed.path == "/":
                    return self._send(LIST_HTML.encode(), "text/html; charset=utf-8")
                if parsed.path == "/progress":
                    return self._send(PROGRESS_HTML.encode(), "text/html; charset=utf-8")
                if parsed.path == "/r10-operation":
                    return self._send(
                        R10_OPERATION_HTML.encode(), "text/html; charset=utf-8"
                    )
                if parsed.path == "/r11-icl":
                    return self._send(R11_ICL_HTML.encode(), "text/html; charset=utf-8")
                if parsed.path == "/r12-branching":
                    return self._send(R12_BRANCHING_HTML.encode(), "text/html; charset=utf-8")
                if parsed.path == "/r19-autonomous":
                    from scripts.univtac.autonomous_dashboard import R19_HTML
                    return self._send(R19_HTML.encode(), "text/html; charset=utf-8")
                if parsed.path == "/api/r19-autonomous":
                    return self._json(load_autonomous_runs(root, "R1.9"))
                if parsed.path in ("/r19-demonstrations", "/r19-videos", "/r19-pairs"):
                    name = parsed.path.removeprefix("/r19-")+".html"
                    return self._send((root/'univtac-isaac51-r19'/name).read_bytes(), "text/html; charset=utf-8")
                if parsed.path == "/r18-autonomous":
                    from scripts.univtac.autonomous_dashboard import R18_HTML
                    return self._send(R18_HTML.encode(), "text/html; charset=utf-8")
                if parsed.path == "/api/r18-autonomous":
                    return self._json(load_autonomous_runs(root, "R1.8"))
                if parsed.path in ("/r18-demonstrations", "/r18-videos"):
                    name = "demonstrations.html" if parsed.path.endswith("demonstrations") else "videos.html"
                    return self._send((root/'univtac-isaac51-r18'/name).read_bytes(), "text/html; charset=utf-8")
                if parsed.path == "/r17-autonomous":
                    from scripts.univtac.autonomous_dashboard import R17_HTML
                    return self._send(R17_HTML.encode(), "text/html; charset=utf-8")
                if parsed.path == "/api/r17-autonomous":
                    return self._json(load_autonomous_runs(root, "R1.7"))
                if parsed.path in ("/r17-demonstrations", "/r17-videos"):
                    name = "demonstrations.html" if parsed.path.endswith("demonstrations") else "videos.html"
                    return self._send((root/'univtac-isaac51-r17'/name).read_bytes(), "text/html; charset=utf-8")
                if parsed.path == "/r16-motion-pacing":
                    page = root/'univtac-isaac51-r16/review.html'
                    return self._send(page.read_bytes(), "text/html; charset=utf-8")
                if parsed.path == "/r15-autonomous":
                    from scripts.univtac.autonomous_dashboard import R15_HTML
                    return self._send(R15_HTML.encode(), "text/html; charset=utf-8")
                if parsed.path == "/api/r15-autonomous":
                    return self._json(load_autonomous_runs(root, "R1.5"))
                if parsed.path == "/r14-autonomous":
                    return self._send(R14_HTML.encode(), "text/html; charset=utf-8")
                if parsed.path == "/api/r14-autonomous":
                    return self._json(load_autonomous_runs(root))
                if parsed.path == "/r13-icl":
                    return self._send(R13_ICL_HTML.encode(), "text/html; charset=utf-8")
                if parsed.path.startswith("/run/"):
                    _safe_run_root(root, parsed.path[5:])
                    return self._send(DETAIL_HTML.encode(), "text/html; charset=utf-8")
                if parsed.path.startswith("/pilot/"):
                    _safe_pilot_root(root, parsed.path[7:])
                    return self._send(PILOT_HTML.encode(), "text/html; charset=utf-8")
                if parsed.path == "/api/runs":
                    return self._json(discover_runs(root))
                if parsed.path == "/api/pilots":
                    return self._json(discover_pilots(root))
                if parsed.path == "/api/progress":
                    return self._json(load_project_progress(root))
                if parsed.path == "/api/r10-operation":
                    return self._json(load_r10_operation_detail(root))
                if parsed.path == "/api/r11-icl":
                    return self._json(load_r11_icl_detail(root))
                if parsed.path == "/api/r12-branching":
                    return self._json(load_r12_branching_detail(root))
                if parsed.path == "/api/r13-icl":
                    return self._json(load_r13_icl_detail(root))
                if parsed.path == "/api/pilot":
                    return self._json(
                        load_pilot_detail(_safe_pilot_root(root, query.get("pilot", [""])[0]))
                    )
                run = _safe_run_root(root, query.get("run", [""])[0])
                if parsed.path == "/api/detail":
                    return self._json(load_run_detail(run))
                requested = query.get("path", [""])[0]
                artifact = _safe_artifact(run, requested)
                if parsed.path == "/artifact" and artifact.suffix == ".mp4":
                    size = artifact.stat().st_size
                    start, end = 0, size - 1
                    requested_range = self.headers.get("Range")
                    if requested_range:
                        first, last = requested_range.removeprefix("bytes=").split("-", 1)
                        start = int(first) if first else max(0, size-int(last))
                        end = min(size-1, int(last)) if first and last else size-1
                    self.send_response(206 if requested_range else 200)
                    self.send_header("Content-Type", "video/mp4")
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Length", str(end-start+1))
                    if requested_range:
                        self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
                    self.end_headers()
                    with artifact.open("rb") as stream:
                        stream.seek(start)
                        remaining = end-start+1
                        try:
                            while remaining > 0:
                                block = stream.read(min(1024*1024, remaining))
                                if not block: break
                                self.wfile.write(block)
                                remaining -= len(block)
                        except (BrokenPipeError, ConnectionResetError):
                            pass
                    return
                if parsed.path == "/artifact":
                    return self._send(
                        artifact.read_bytes(),
                        mimetypes.guess_type(artifact.name)[0] or "application/octet-stream",
                    )
                if parsed.path == "/raw":
                    return self._send(
                        artifact.read_bytes(),
                        "text/plain; charset=utf-8",
                    )
            except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
                return self._json({"error": str(exc)}, 404)
            self._json({"error": "not found"}, 404)

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9399)
    args = parser.parse_args(argv)
    if args.host != "127.0.0.1":
        raise ValueError("UniVTAC experiment dashboard is loopback-only")
    server = ThreadingHTTPServer((args.host, args.port), make_handler(args.runs_root))
    print(f"http://{args.host}:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
