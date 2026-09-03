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
<h1>UniVTAC experiment runs</h1><table><thead><tr><th>round / directory</th><th>task</th><th>seed</th><th>model</th><th>status</th><th>start</th><th>duration</th><th>observe count</th></tr></thead><tbody id="runs"></tbody></table>
<script>const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));async function tick(){const xs=await fetch('/api/runs?t='+Date.now()).then(r=>r.json());document.querySelector('#runs').innerHTML=xs.map(x=>`<tr><td><a href="/run/${encodeURIComponent(x.directory)}">${esc(x.round)} / ${esc(x.directory)}</a></td><td>${esc(x.task)}</td><td>${esc(x.seed)}</td><td>${esc(x.model)}</td><td class="${x.status==='completed'?'ok':'failed'}">${esc(x.status)}</td><td>${esc(x.started_at)}</td><td>${esc(x.duration_seconds??'')}</td><td>${esc(x.observe_count)}</td></tr>`).join('')}tick();setInterval(tick,2000)</script>"""

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
                if parsed.path.startswith("/run/"):
                    _safe_run_root(root, parsed.path[5:])
                    return self._send(DETAIL_HTML.encode(), "text/html; charset=utf-8")
                if parsed.path == "/api/runs":
                    return self._json(discover_runs(root))
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
