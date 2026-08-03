#!/usr/bin/env python3
"""Small SSH-friendly browser dashboard for one embodied episode."""

from __future__ import annotations

import argparse
import json
import mimetypes
import socket
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


HTML = """<!doctype html><meta charset=utf-8><title>OpenETA episode</title>
<style>body{font:14px sans-serif;background:#111;color:#ddd;margin:16px}main{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}section{background:#1b1b1b;padding:10px;border:1px solid #333;border-radius:6px}img{max-width:100%;background:#000}.pick{position:relative;display:inline-block;line-height:0;cursor:crosshair}.pick img{display:block;width:100%;height:auto}.pick i{position:absolute;width:10px;height:10px;border:2px solid #0f0;border-radius:50%;display:none;pointer-events:none}.hint{color:#aaa;font-size:12px}.controls{display:flex;gap:6px;flex-wrap:wrap;align-items:center}.controls input{width:72px;background:#222;color:#ddd;border:1px solid #555;padding:4px}.controls button{background:#284b63;color:#fff;border:1px solid #5b8bad;padding:5px 8px;border-radius:4px;cursor:pointer}.controls button:disabled{opacity:.35;cursor:not-allowed}.wide{grid-column:1/-1}pre{white-space:pre-wrap;max-height:420px;overflow:auto;font-size:12px}.trace{max-height:720px;overflow:auto}.thumb{max-height:420px}.tracecard{border-top:1px solid #444;padding:12px 0}.tracehead{display:flex;gap:12px;flex-wrap:wrap;align-items:baseline;color:#bbb}.tracehead b{color:#8fd3ff}.traceimgs{display:grid;grid-template-columns:repeat(3,minmax(220px,1fr));gap:8px;margin-top:8px}.traceimg{width:100%;max-height:360px;object-fit:contain;border:1px solid #444}.tracecard details pre{max-height:360px}.badge{padding:2px 6px;border-radius:10px;background:#444;color:#fff;font-size:11px}.pending{background:#8a5b00}.solved{background:#176b3a}.response{background:#244f73}.live{background:#176b3a}.replay{background:#735224}.offline{background:#753232}.context-text{background:#101820;border-left:3px solid #5b8bad;padding:8px;margin:8px 0;white-space:pre-wrap;font:12px monospace}</style>
<h2>OpenETA embodied episode <span id=modeBadge class=badge>LOADING</span> <span id=contextBadge class=badge>CONTEXT LOADING</span></h2><div id=meta></div><div><a id=sim target=_blank>Simulator dashboard</a> · <a id=viser target=_blank>Viser 3D point cloud</a> <button onclick="startViserReplay()">start / renew 3D replay</button></div>
<section id=manualSection class=wide><h3>Manual Operator harness</h3><div id=controlHint class=hint>Use individual agentview, wrist, top, front, or side images. A pending point returns only the remaining source views that can complete it; solved coordinates are retained in the trace.</div>
<div class=controls><button onclick="sendObserve()">observe / refresh RGB-D</button><label>point ID <input id=pointId value=P0></label><button onclick="sendPoint()">mark_point from selected click</button><span id=manualStatus></span></div>
<div class=controls><b>move_to absolute metres</b><label>x <input id=mx type=number step=0.001></label><label>y <input id=my type=number step=0.001></label><label>z <input id=mz type=number step=0.001></label><b>or world delta mm</b><label>dX <input id=dx type=number step=1></label><label>dY <input id=dy type=number step=1></label><label>dZ <input id=dz type=number step=1></label></div>
<div class=controls><b>optional orientation</b><label>APP x <input id=ax type=number step=0.01></label><label>y <input id=ay type=number step=0.01></label><label>z <input id=az type=number step=0.01></label><label>JAW x <input id=jx type=number step=0.01></label><label>y <input id=jy type=number step=0.01></label><label>z <input id=jz type=number step=0.01></label><button onclick="sendMove(true)">preview</button><button onclick="sendMove(false)">execute</button><button onclick="sendGripper('open')">open gripper</button><button onclick="sendGripper('close')">close gripper</button><span class=hint>Blank controls stay unchanged. APP is grip-site local +Z; world down is [0,0,-1].</span></div></section>
<main><section><h3>Agentview RGB</h3><div class=pick data-view=agentview><img id=agent><i></i></div></section><section><h3>Wrist RGB</h3><img id=wrist></section><section><h3>Latest visualization</h3><img id=viz></section>
<section><h3>Point cloud top</h3><div class=pick data-view=pointcloud_top><img id=top class=thumb><i></i></div></section><section><h3>Point cloud front</h3><div class=pick data-view=pointcloud_front><img id=front class=thumb><i></i></div></section><section><h3>Point cloud side</h3><div class=pick data-view=pointcloud_side><img id=side class=thumb><i></i></div></section>
<section><h3>3D marks / marked pose</h3><pre id=poses></pre></section><section><h3>Authoritative selected point</h3><pre id=pointinfo></pre></section><section><h3>Current projection</h3><pre id=current></pre></section>
<section class=wide><h3>Actual Operator MCP context replay</h3><div class=hint>权威视图：严格按 operator_context.jsonl 顺序显示每次 tool call、返回文字，以及当时真正发送给 Operator 的全部图片。PENDING RAY 卡片中的三张图就是模型看到的跨视图射线。</div><div id=operatorGallery class=trace></div></section>
<section class=wide><h3>Operator initial context / prompt</h3><details><summary>show exact clean-context contract and prompt</summary><pre id=contract></pre></details></section>
<section class=wide><h3>Persistent environment event replay</h3><div class=hint>辅助视图：直接读取 events.jsonl；用于检查 simulator/action event，不代表模型实际收到的 context。</div><div id=traceGallery class=trace></div></section>
<section class=wide><h3>Codex app-server trace</h3><pre id=apptrace class=trace></pre></section></main><script>
  const replayBase=__OPENETA_BASE__;
let selected=null;
const j=async p=>fetch(replayBase+p+'?t='+Date.now()).then(r=>r.json()); const esc=x=>JSON.stringify(x,null,2);
function art(p){return p?replayBase+'/artifact/'+p:''}
function safeText(x){return String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
let traceEventCount=-1, operatorRowCount=-1, traceLastFetch=0, contextLastFetch=0, contractLastFetch=0;
function renderTrace(events){
  const host=document.querySelector('#traceGallery'); if(!host)return;
  host.innerHTML=(events||[]).slice().reverse().map((e,i)=>{
    const p=e.payload||{}, frames=p.frames||[], result=p.result||{};
    const paths=[...frames.map(f=>f.rgb_path),...(p.response_image_paths||[])]
      .filter(Boolean).filter((x,n,a)=>a.indexOf(x)===n).slice(0,4);
    const label=e.kind==='tool_result'?(p.tool||'tool_result'):e.kind;
    const summary=p.observation_id?`observation=${p.observation_id} sim_step=${p.sim_step??''}`:
      p.request?.stage?`stage=${p.request.stage}`:(result.message||p.message||'');
    const details={kind:e.kind,seq:e.seq,action_id:e.action_id,tool_call_id:e.tool_call_id,payload:p};
    return `<article class=tracecard><div class=tracehead><b>${safeText(label)}</b><span>seq ${safeText(e.seq)}</span><span>${safeText(summary)}</span></div><div class=traceimgs>${paths.map(x=>`<img class=traceimg src="${art(x)}">`).join('')}</div><details><summary>event details</summary><pre>${safeText(JSON.stringify(details,null,2))}</pre></details></article>`;
  }).join('')||'<div class=hint>No persisted events yet.</div>';
}
function responseObject(row){
  let blocks=row.response_text_blocks||[];
  if(!blocks.length)return {};
  try{return JSON.parse(blocks[0])}catch(_){return {}}
}
function imageCaption(path){
  let name=String(path||'').split('/').pop();
  if(name.includes('.pending.'))return name.replace('.pending.png','')+' · projected pending ray';
  if(name.includes('.marked.'))return name.replace('.marked.png','')+' · solved point';
  return name;
}
function renderOperatorContext(rows){
  const host=document.querySelector('#operatorGallery'); if(!host)return;
  host.innerHTML=(rows||[]).map(row=>{
    const result=responseObject(row),status=result.status||'',tool=row.tool||'unknown';
    const pending=tool==='mark_point'&&status==='pending';
    const solved=tool==='mark_point'&&status==='solved';
    const badge=pending?'<span class="badge pending">PENDING RAY · ACTUALLY SENT</span>':
      solved?'<span class="badge solved">SOLVED POINT · ACTUALLY SENT</span>':
      '<span class="badge response">ACTUAL MCP RESPONSE</span>';
    const paths=(row.response_image_paths||[]).filter(Boolean);
    const images=paths.map(path=>`<figure><img class=traceimg src="${art(path)}"><figcaption class=hint>${safeText(imageCaption(path))}</figcaption></figure>`).join('');
    const response=(row.response_text_blocks||[]).join(String.fromCharCode(10));
    return `<article class=tracecard data-tool="${safeText(tool)}"><div class=tracehead><b>${safeText(tool)}</b><span>context seq ${safeText(row.seq)}</span>${badge}<span>${safeText(result.observation_id||'')}</span></div><div class=context-text><b>arguments</b>\n${safeText(JSON.stringify(row.arguments||{},null,2))}\n\n<b>response received by Operator</b>\n${safeText(response)}</div><div class=traceimgs>${images}</div><details><summary>exact operator_context row</summary><pre>${safeText(JSON.stringify(row,null,2))}</pre></details></article>`;
  }).join('')||'<div class=hint>No persisted Operator context yet.</div>';
}
function setStatus(x){document.querySelector('#manualStatus').textContent=typeof x==='string'?x:esc(x)}
function sourcePixel(el,e){let r=el.getBoundingClientRect(),img=el.querySelector('img');return {x:Math.round((e.clientX-r.left)*img.naturalWidth/r.width),y:Math.round((e.clientY-r.top)*img.naturalHeight/r.height)}}
document.querySelectorAll('.pick').forEach(el=>{
  el.addEventListener('click',e=>{let p=sourcePixel(el,e);selected={view:el.dataset.view,u:p.x,v:p.y};let i=el.querySelector('i');i.style.left=(e.clientX-el.getBoundingClientRect().left-7)+'px';i.style.top=(e.clientY-el.getBoundingClientRect().top-7)+'px';i.style.display='block';setStatus({selected})})
});
async function post(path,body){let r=await fetch(replayBase+path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});let x=await r.json().catch(()=>({success:false,reason:'invalid_json_response',status:r.status}));if(!r.ok&&x.reason==='manual_control_unavailable')x={...x,reason:'manual_control_http_error',http_status:r.status};setStatus(x);return x}
function sendObserve(){post('/api/manual/observe',{})}
function sendPoint(){if(!selected)return setStatus('click agentview or a point-cloud view first');let point_id=document.querySelector('#pointId').value.trim();if(!point_id)return setStatus('point ID is required');setStatus({pending:'mark_point',point_id,selected});post('/api/manual/mark_point',{view:selected.view,x:selected.u,y:selected.v,point_id}).then(x=>{let t=(x&&x.text)||{};setStatus(t)})}
function optionalTriple(ids,name){let raw=ids.map(id=>document.querySelector('#'+id).value.trim()),used=raw.some(Boolean);if(!used)return null;if(raw.some(x=>!x))throw new Error(name+' requires all three values');let values=raw.map(Number);if(values.some(x=>!Number.isFinite(x)))throw new Error(name+' values must be finite');return values}
function sendMove(preview){try{let xyz=optionalTriple(['mx','my','mz'],'xyz_m'),delta=optionalTriple(['dx','dy','dz'],'delta_mm'),approach=optionalTriple(['ax','ay','az'],'approach_world'),jaw=optionalTriple(['jx','jy','jz'],'jaw_world');if(xyz&&delta)throw new Error('use xyz_m or delta_mm, not both');if(!xyz&&!delta&&!approach&&!jaw)throw new Error('provide at least one move_to control');let body={};if(xyz)body.xyz_m=xyz;if(delta)body.delta_mm=delta;if(approach)body.approach_world=approach;if(jaw)body.jaw_world=jaw;if(preview)body.preview=true;post('/api/manual/move_to',body)}catch(e){setStatus({success:false,reason:'invalid_move_to_input',message:e.message})}}
function sendGripper(action){post('/api/manual/move_to',{gripper:action})}
async function startViserReplay(){let x=await post('/api/replay/viser',{});if(x.success&&x.viser_url){let a=document.querySelector('#viser');a.href=x.viser_url;window.open(x.viser_url,'_blank')}}
async function tick(){
  try{
    let c=await j('/api/current'),cfg=await j('/api/config'),ctl=await j('/api/control_status');
    window.latestCurrent=c;
    let obs=c.record?.observation_id||'',m=c.pointcloud_metrics||{};
    let terminal=ctl.terminal===true,available=ctl.control_available===true,mode=terminal?'TERMINAL · REPLAY':available?'LIVE':'RUNTIME OFFLINE';
    let badge=document.querySelector('#modeBadge');badge.textContent=mode;badge.className='badge '+(terminal?'replay':available?'live':'offline');
    document.querySelectorAll('#manualSection button,#manualSection input').forEach(el=>el.disabled=!available||terminal);
    document.querySelector('#controlHint').textContent=terminal?'Read-only replay: the episode is terminal. Images, exact Operator context, prompt, and event traces remain available.':available?'Click one named source image. A pending point returns only the remaining complementary source views; use the same point ID to solve XYZ.':'Runtime control is offline. Replay remains readable; restart the matching Gateway only if this non-terminal episode should continue.';
    document.querySelector('#meta').textContent=`mode=${mode} status=${c.status} sim_step=${c.sim_step} observation=${obs} cloud=${c.pointcloud_source||'unknown'} cloud_obs=${c.pointcloud_observation_id||m.observation_id||''} cameras=${m.camera_count??'?'} points=${m.points_after_voxel_fusion??m.workspace_points??'?'} voxels5mm=${m.voxels_5mm??'?'} task=${c.task||''}`;
    document.querySelector('#sim').href=c.viewer_url||'#';
    document.querySelector('#viser').href=cfg.viser_url||'#';
    let fs=c.operator?.frames||c.record?.frames||[];
    for(let f of fs){
      let el=f.camera_id==='agentview'?document.querySelector('#agent'):f.camera_id==='wrist'?document.querySelector('#wrist'):null;
      if(el){let src=art(f.rgb_path);if(el.src!==location.origin+src)el.src=src}
    }
    let v=c.pointcloud_views||[],pointMarked=c.point_view_paths||{},activeGrip=c.active_grip_site_view_paths||{};
    for(let x of v){
      let p=pointMarked[x.view]||activeGrip[x.view]||x.image_path;
      let el=x.view==='pointcloud_top'?document.querySelector('#top'):x.view==='pointcloud_front'?document.querySelector('#front'):document.querySelector('#side');
      let src=art(p);if(el.src!==location.origin+src)el.src=src;
    }
    let marks=c.point_marks_3d||{};
    document.querySelector('#poses').textContent=esc({point_marks_3d:marks,pending_point_constraints:c.pending_point_constraints||{}});
    let primary=Object.values(marks)[0]||null;
    document.querySelector('#pointinfo').textContent=esc(primary?{observation_id:primary.observation_id,point_id:primary.point_id,label:primary.label,source_view:primary.view,requested_pixel_xy:primary.requested_pixel_xy,xyz_m:primary.xyz_m}: {message:'No solved point yet. Mark the same point ID in two complementary views.'});
    document.querySelector('#current').textContent=esc({observation_id:obs,sim_step:c.sim_step,pointcloud_source:c.pointcloud_source,pointcloud_mode:c.pointcloud_mode,pointcloud_metrics:m,pointcloud_views:v.map(x=>({view:x.view,width:x.width,height:x.height})),solved_point_ids:Object.keys(marks)});
    if(Date.now()-traceLastFetch>5000){
      traceLastFetch=Date.now();
      let tr=await j('/api/trace');
      if(tr.event_count!==traceEventCount){traceEventCount=tr.event_count;renderTrace(tr.events||[])}
    }
    if(Date.now()-contextLastFetch>2000){
      contextLastFetch=Date.now();
      let op=await j('/api/operator_trace');
      if(op.row_count!==operatorRowCount){operatorRowCount=op.row_count;renderOperatorContext(op.rows||[])}
      let a=await fetch(replayBase+'/api/app_trace?t='+Date.now()).then(r=>r.text());
      if(document.querySelector('#apptrace').textContent!==a)document.querySelector('#apptrace').textContent=a;
    }
    if(Date.now()-contractLastFetch>2000){
      contractLastFetch=Date.now();
      let contract=await j('/api/operator_contract');
      document.querySelector('#contract').textContent=esc(contract);
      let profile=contract.context_profile||{},label=profile.label||'UNVERSIONED';
      let sha=contract.resolved_context_sha256||'';
      document.querySelector('#contextBadge').textContent=`${label}${sha?' · '+sha.slice(0,10):' · resolving'}`;
    }
  }catch(e){document.querySelector('#meta').textContent='dashboard error: '+e}
}
tick();setInterval(tick,1000);
</script>"""


class Handler(BaseHTTPRequestHandler):
    root: Path
    viser_url: str
    control_url: str = ""

    def _send(self, body: bytes, content_type: str) -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        if path == "/":
            return self._send(
                render_dashboard_html("").encode(),
                "text/html; charset=utf-8",
            )
        if path == "/api/current":
            return self._json(self.root / "current.json", {})
        if path == "/api/config":
            return self._send(json.dumps({"viser_url": self.viser_url, "control_url": self.control_url}).encode(), "application/json")
        if path == "/api/control_status":
            return self._send(
                json.dumps(self._control_status()).encode(),
                "application/json",
            )
        if path == "/api/marked_points":
            return self._json(self.root / "perception/marked_points.json", {})
        if path == "/api/poses":
            return self._json(self.root / "perception/pose_candidates.world.json", {})
        if path == "/api/history":
            p = self.root / "operator_context.jsonl"
            return self._send(p.read_bytes() if p.exists() else b"", "application/x-ndjson")
        if path == "/api/operator_trace":
            rows = []
            p = self.root / "operator_context.jsonl"
            if p.exists():
                for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(value, dict):
                        rows.append(value)
            return self._send(
                json.dumps(
                    {"episode_root": str(self.root), "row_count": len(rows), "rows": rows},
                    ensure_ascii=False,
                ).encode(),
                "application/json",
            )
        if path == "/api/operator_contract":
            return self._json(self.root / "operator_context_contract.json", {})
        if path == "/api/app_trace":
            p = self.root / "operator_app_server.jsonl"
            return self._send(p.read_bytes() if p.exists() else b"", "application/x-ndjson")
        if path == "/api/trace":
            events = []
            p = self.root / "events.jsonl"
            if p.exists():
                for line in p.read_text(encoding="utf-8", errors="replace").splitlines():
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(value, dict):
                        events.append(value)
            return self._send(
                json.dumps(
                    {"episode_root": str(self.root), "event_count": len(events), "events": events},
                    ensure_ascii=False,
                ).encode(),
                "application/json",
            )
        if path.startswith("/artifact/"):
            raw = Path(path.removeprefix("/artifact/"))
            # current.json may retain either an episode-relative artifact path
            # or an absolute path from the host process.
            candidate = raw.resolve() if raw.is_absolute() else (self.root / raw).resolve()
            if self.root.resolve() not in candidate.parents and candidate != self.root.resolve():
                return self.send_error(403)
            if candidate.is_file():
                return self._send(candidate.read_bytes(), mimetypes.guess_type(candidate.name)[0] or "application/octet-stream")
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        path = unquote(urlparse(self.path).path)
        if path == "/api/replay/viser":
            result = subprocess.run(
                [
                    "bash",
                    str(
                        Path(__file__).resolve().parent
                        / "start_episode_viser_replay.sh"
                    ),
                    str(self.root),
                ],
                text=True,
                capture_output=True,
                check=False,
            )
            try:
                payload = json.loads(result.stdout.strip().splitlines()[-1])
            except (IndexError, json.JSONDecodeError):
                payload = {
                    "success": False,
                    "reason": "viser_replay_start_failed",
                    "message": result.stderr.strip() or result.stdout.strip(),
                }
            return self._send(
                json.dumps(payload).encode(),
                "application/json",
            )
        if path.startswith("/api/manual/"):
            control = self._control_status()
            if control["terminal"]:
                return self._send(
                    json.dumps(
                        {
                            "success": False,
                            "reason": "episode_read_only",
                            "message": (
                                "This episode is terminal. The dashboard is retained "
                                "as a read-only replay."
                            ),
                        }
                    ).encode(),
                    "application/json",
                )
            if not control["control_available"]:
                return self._send(
                    json.dumps(
                        {
                            "success": False,
                            "reason": "manual_control_unavailable",
                            "message": (
                                "The episode Gateway is offline. Replay remains "
                                "available, but world-changing controls are disabled."
                            ),
                        }
                    ).encode(),
                    "application/json",
                )
            try:
                import urllib.request
                length = int(self.headers.get("Content-Length", "0"))
                payload = self.rfile.read(length)
                req = urllib.request.Request(
                    self.control_url + path.removeprefix("/api/manual"),
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=180) as response:
                    return self._send(response.read(), "application/json")
            except Exception as exc:  # pragma: no cover - browser-facing error path
                return self._send(
                    json.dumps(
                        {
                            "success": False,
                            "reason": "manual_control_unavailable",
                            "message": str(exc),
                        }
                    ).encode(),
                    "application/json",
                )
        self.send_error(404)

    def _json(self, path: Path, default: object) -> None:
        try: body = path.read_bytes()
        except OSError: body = json.dumps(default).encode()
        self._send(body, "application/json")

    @staticmethod
    def _read_json(path: Path) -> dict:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return value if isinstance(value, dict) else {}

    def _control_status(self) -> dict:
        current = self._read_json(self.root / "current.json")
        status = str(current.get("status") or "unknown")
        terminal = status in {
            "completed", "failed", "aborted", "stopped", "terminated", "error"
        }
        parsed = urlparse(self.control_url)
        available = False
        if not terminal and parsed.hostname and parsed.port:
            try:
                with socket.create_connection(
                    (parsed.hostname, parsed.port), timeout=0.15
                ):
                    available = True
            except OSError:
                pass
        return {
            "status": status,
            "terminal": terminal,
            "control_available": available,
            "mode": "replay" if terminal else "live" if available else "offline",
        }

    def log_message(self, *_args: object) -> None:
        pass


def render_dashboard_html(base_path: str) -> str:
    """Bind the episode UI to either a dedicated server or Replay Hub prefix."""

    return HTML.replace(
        "__OPENETA_BASE__",
        json.dumps(str(base_path).rstrip("/")),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8890)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--viser-url", default="http://127.0.0.1:8910")
    parser.add_argument("--control-url", default="http://127.0.0.1:8790")
    args = parser.parse_args()
    Handler.root = args.root.resolve()
    Handler.viser_url = args.viser_url
    Handler.control_url = args.control_url.rstrip("/")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
