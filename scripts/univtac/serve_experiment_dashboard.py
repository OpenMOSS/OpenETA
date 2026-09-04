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

from sim.envs.univtac.codex_readonly import read_jsonl, summarize_codex_exec

LIST_HTML = """<!doctype html><meta charset="utf-8"><title>UniVTAC experiments</title>
<style>body{font:14px system-ui;background:#101216;color:#e6e9ef;margin:24px}a{color:#8fd3ff}table{border-collapse:collapse;width:100%;background:#181c22}th,td{padding:9px;border:1px solid #303744;text-align:left}.ok{color:#76db8b}.failed{color:#ff8585}</style>
<h1>UniVTAC experiment runs</h1><p><a href="/progress">项目进展：我和 GPT-Pro 每轮做了什么 →</a></p><div id="pilots"></div><table><thead><tr><th>round / directory</th><th>task</th><th>seed</th><th>model</th><th>status</th><th>start</th><th>duration</th><th>observe count</th></tr></thead><tbody id="runs"></tbody></table>
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
async function tick(){
 const d=await fetch('/api/pilot?pilot='+encodeURIComponent(pilot)+'&t='+Date.now()).then(r=>r.json()),round=d.pilot.round;
 document.querySelector('#pilotTitle').textContent=round==='R0.9.18'?'Pull Out Key · Image-First Staged Grounding Pilot':round==='R0.9.17'?'Pull Out Key · Unilateral Tactile Guidance Conflict Pilot':round==='R0.9.16'?'Pull Out Key · RGB-marker Structured Guidance Pilot':round==='R0.9.15'?'Pull Out Key · Before/After Tactile Difference Pilot':'Pull Out Key · 3 seeds × 3 conditions';
 const calls=d.summary.completed_semantic_trials??d.summary.completed_call_count,signal=d.summary.grounding_signal??d.summary.unilateral_signal??d.summary.structured_signal??d.summary.difference_signal??d.summary.pilot_signal;
 document.querySelector('#top').innerHTML=`<b>model:</b> ${esc(d.pilot.model)} · <b>prompt:</b> prompt.txt · <b>completed calls:</b> ${esc(calls)} · <b>signal:</b> ${esc(signal)}`;
 const matrix=document.querySelector('#matrix');matrix.classList.toggle('two',round==='R0.9.18');matrix.innerHTML=d.cells.map(c=>`<div class="cell"><h3>seed ${esc(c.seed)} · ${esc(c.condition)}</h3>${swapNote(c)}${c.phases.length?stagedCell(c):legacyCell(c)}<p>${esc(c.duration_seconds)} s · tool ${esc(c.tool_call_count)} · usage ${esc(JSON.stringify(c.usage||{}))}</p><a href="/run/${encodeURIComponent(c.run)}">detail</a></div>`).join('');
 const pairs=document.querySelector('#pairs');pairs.hidden=!d.pairs.length;pairs.innerHTML='<h2>Baseline | Current | Difference</h2>'+d.pairs.map(x=>`<h3>seed ${esc(x.seed)} · ${esc(x.sensor)}</h3><div class="pair-images">${x.images.map(i=>`<figure><img src="${art(i.run,i.path)}"><figcaption>${esc(i.label)}</figcaption></figure>`).join('')}</div>`).join('');
 const human=document.querySelector('#humanComparisons');human.hidden=!d.human_comparisons.length;human.innerHTML='<h2>每个 seed 用人话对比</h2>'+d.human_comparisons.map(x=>`<div class="host-card"><b>seed ${esc(x.seed)}</b><p>输入里 ${esc(x.baseline_side)} 侧被换成 baseline，真实图片变化侧是 ${esc(x.expected_change_side)}。</p><p>只看图片：${esc(x.image_only_side)}；正确文字：${esc(x.correct_guidance_side)}（是否帮助：${esc(x.correct_guidance_helped)}）；错误文字：${esc(x.swapped_guidance_side)}，因此模型${esc(x.conflict_reaction)}。</p></div>`).join('');
 document.querySelector('#comparison').innerHTML='<h2>Cross-condition comparisons</h2><pre>'+esc(JSON.stringify(d.summary.metrics??d.summary,null,2))+'</pre>'+(round==='R0.9.18'?'<h3>R0.9.17 single-stage swapped vs image-first committed swapped</h3><pre>'+esc(JSON.stringify(d.summary.r0917_comparison||{},null,2))+'</pre>':'');
 const refs=d.structured_reference?.seeds||{};document.querySelector('#hostBars').innerHTML=Object.values(refs).flatMap(x=>Object.entries(x.sensors||{}).map(([sensor,s])=>{const m=s.structured_metrics||{},v=m.normalized_horizontal_saliency||[];return `<div class="host-card"><b>seed ${esc(x.seed)} · ${esc(sensor)}</b><p>new: ${esc(m.reference_region)} · old centroid: ${esc(s.old_global_centroid_region)} · agree: ${esc(s.old_and_new_region_agree)}</p>${['left','center','right'].map((name,i)=>`<label>${name} ${esc(Number(v[i]||0).toFixed(3))}</label><div class="bar"><span style="width:${Math.max(0,Math.min(100,(v[i]||0)*100))}%"></span></div>`).join('')}</div>`})).join('');
 document.querySelector('#hostTitle').textContent=round==='R0.9.18'?'HOST-ONLY STAGED GROUNDING SCORES — NOT SHOWN TO CODEX':round==='R0.9.17'?'HOST-ONLY COUNTERFACTUAL MAPPING — NOT SHOWN TO CODEX':'HOST-ONLY EVALUATION — NOT SHOWN TO CODEX';
 document.querySelector('#host').textContent=JSON.stringify({reference:d.host_reference,difference_metrics:d.difference_metrics,structured_reference:d.structured_reference,secondary_diagnostics:d.secondary_diagnostics,source_mapping:d.source_mapping,condition_manifests:d.condition_manifests},null,2);
}
tick();setInterval(tick,2000)
</script>"""

PROGRESS_HTML = """<!doctype html><meta charset="utf-8"><title>OpenETA-UniVTAC 项目进展</title>
<style>body{font:15px/1.65 system-ui;background:#101216;color:#e6e9ef;margin:24px auto;max-width:1050px;padding:0 18px}a{color:#8fd3ff}.intro{color:#b8c0cc}.entry{background:#181c22;border:1px solid #303744;border-radius:12px;padding:20px 24px;margin:18px 0}.entry h3{margin:18px 0 5px;color:#d9e7f5}.entry p{white-space:pre-wrap;margin:5px 0}.meta{color:#98a3b3;font-size:13px}.status{display:inline-block;border-radius:14px;padding:3px 9px;background:#245f3b;font-size:12px}.warn{background:#805b20}.question{border-left:4px solid #b07cff;padding-left:14px}.method{border-left:4px solid #55b8ff;padding-left:14px}.observed{border-left:4px solid #65c78d;padding-left:14px}.meaning{border-left:4px solid #f0b55a;padding-left:14px}.conclusion{font-size:17px;background:#222934;border-left:4px solid #e8edf5;margin:20px 0 10px;padding:12px 16px}details{background:#11161c;border-radius:8px;padding:10px 12px;margin-top:14px}pre{white-space:pre-wrap}.experiments{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:10px}.card{background:#181c22;border:1px solid #303744;border-radius:8px;padding:12px}</style>
<a href="/">← 实验列表</a><h1>OpenETA-UniVTAC：现在做到哪了</h1><p class="intro">每一轮先讲清楚问题、做法、观察和含义；commit、测试和原始产物收在“技术证据”里。这里只展示可见工作记录，不展示隐藏思维或认证信息。</p><div id="entries"></div><h2>实验可视化</h2><div class="experiments" id="experiments"></div>
<script>const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));const list=x=>Array.isArray(x)?x:[];async function tick(){const d=await fetch('/api/progress?t='+Date.now()).then(r=>r.json());document.querySelector('#entries').innerHTML=[...d.entries].reverse().map(x=>{const q=x.pro_question||x.pro_instruction_summary||'';const method=x.method_summary||x.codex_work_summary||'';const observed=x.observed_summary||x.result_summary||'尚未结束';const meaning=x.implication_summary||'';const conclusion=x.one_line_conclusion||x.result_summary||'尚未结束';const evidence=x.evidence||{};const root=evidence.experiment_root||x.experiment_root||'';const commit=evidence.commit||x.commit||'';const push=evidence.push||x.push||'';const tests=list(evidence.tests);const artifacts=list(evidence.artifact_paths);return `<article class="entry"><h2>${esc(x.round)} <span class="status ${x.status==='completed'?'':'warn'}">${esc(x.status)}</span></h2><p class="meta">${esc(x.timestamp)}</p><section class="question"><h3>Pro 想弄清楚什么</h3><p>${esc(q)}</p></section><section class="method"><h3>我们怎么验证的</h3><p>${esc(method)}</p></section><section class="observed"><h3>实际看到了什么</h3><p>${esc(observed)}</p></section><section class="meaning"><h3>这说明什么</h3><p>${esc(meaning||'本轮尚未形成新的研究判断。')}</p></section><blockquote class="conclusion"><b>一句话结论：</b> ${esc(conclusion)}</blockquote>${root?`<p><a href="/pilot/${encodeURIComponent(root)}">打开本轮实验可视化 →</a></p>`:''}<details><summary>技术证据（commit、测试、产物）</summary><p>commit: ${esc(commit||'未记录')}</p><p>push: ${esc(push||'未记录')}</p>${tests.length?`<h4>测试</h4><ul>${tests.map(t=>`<li>${esc(t)}</li>`).join('')}</ul>`:''}${artifacts.length?`<h4>产物</h4><ul>${artifacts.map(p=>`<li>${esc(p)}</li>`).join('')}</ul>`:''}</details></article>`}).join('');document.querySelector('#experiments').innerHTML=d.pilots.map(x=>`<div class="card"><h3>${esc(x.directory)}</h3><p>信号：${esc(x.signal||'运行中')} · 已完成 ${esc(x.completed_call_count)} 次</p><a href="/pilot/${encodeURIComponent(x.directory)}">打开实验图片与回答</a></div>`).join('')}tick();setInterval(tick,2000)</script>"""

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
        }:
            pilots.append(
                {
                    "directory": pilot_root.relative_to(root).as_posix(),
                    "signal": summary.get(
                        "grounding_signal",
                        summary.get(
                            "unilateral_signal",
                            summary.get(
                                "structured_signal",
                                summary.get(
                                    "difference_signal", summary.get("pilot_signal")
                                ),
                            ),
                        ),
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
    predictions = read_jsonl(
        pilot_root
        / (
            "final_predictions.jsonl"
            if pilot.get("round") == "R0.9.18"
            else "predictions.jsonl"
        )
    )
    by_pair = {(row.get("seed"), row.get("condition")): row for row in predictions}
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
        if len(rows) >= 3 and rows[2].get("response_text_blocks"):
            try:
                guidance_payload = json.loads(rows[2]["response_text_blocks"][0])
            except (TypeError, json.JSONDecodeError):
                guidance_payload = {}
        image_judgment = _load_json(run_root / "image_judgment.json")
        prediction = by_pair.get((episode.get("seed"), episode.get("condition")), {})
        staged = pilot.get("round") == "R0.9.18"
        relative_run = run_root.relative_to(pilot_root.parent).as_posix()
        cells.append(
            {
                "seed": episode.get("seed"),
                "condition": episode.get("condition"),
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
                else [],
                "provisional_changed": image_judgment.get("tactile_changed_side")
                != prediction.get("final_changed_side")
                if staged
                else None,
                "conflict_detected": prediction.get("guidance_consistency") == "conflicting"
                if staged
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
    return {
        "pilot": pilot,
        "summary": _load_json(pilot_root / "summary.json"),
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
    if not candidate.is_relative_to(root) or not (candidate / "episode.json").is_file():
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
                if parsed.path == "/api/pilot":
                    return self._json(
                        load_pilot_detail(_safe_pilot_root(root, query.get("pilot", [""])[0]))
                    )
                run = _safe_run_root(root, query.get("run", [""])[0])
                if parsed.path == "/api/detail":
                    return self._json(load_run_detail(run))
                requested = query.get("path", [""])[0]
                artifact = _safe_artifact(run, requested)
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
