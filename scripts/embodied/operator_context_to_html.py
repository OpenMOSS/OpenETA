#!/usr/bin/env python3
"""Render an operator_context.jsonl episode log into a single self-contained HTML file.

All referenced images are embedded as base64 data URIs (deduplicated by path),
so the resulting HTML can be distributed as one file.

Usage:
    python scripts/embodied/operator_context_to_html.py <episode_dir> [output.html]
        [--jpeg-quality N] [--max-width PX]

With --jpeg-quality, images are re-encoded as JPEG (requires Pillow) to
shrink the output file; otherwise the original files are embedded as-is.
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import os
import sys

MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp"}


def load_episode(episode_dir: str) -> dict:
    path = os.path.join(episode_dir, "episode.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def encode_image(path: str, jpeg_quality: int | None, max_width: int) -> str:
    ext = os.path.splitext(path)[1].lower()
    if jpeg_quality is None:
        mime = MIME.get(ext, "application/octet-stream")
        with open(path, "rb") as f:
            return f"data:{mime};base64,{base64.b64encode(f.read()).decode()}"
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        if im.width > max_width:
            im = im.resize((max_width, round(im.height * max_width / im.width)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=jpeg_quality)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def embed_images(records: list[dict], jpeg_quality: int | None, max_width: int) -> dict[str, str]:
    """Base64-embed every existing referenced image, deduplicated by path."""
    cache: dict[str, str] = {}
    for rec in records:
        for p in rec.get("response_image_paths") or []:
            if p in cache:
                continue
            cache[p] = encode_image(p, jpeg_quality, max_width) if os.path.exists(p) else ""
    return cache


def pretty_blocks(blocks: list[str]) -> str:
    out = []
    for b in blocks:
        try:
            out.append(json.dumps(json.loads(b), indent=2, ensure_ascii=False))
        except (ValueError, TypeError):
            out.append(b)
    return "\n".join(out)


def render(episode_dir: str, out_path: str, jpeg_quality: int | None = None, max_width: int = 1280) -> None:
    ctx_path = os.path.join(episode_dir, "operator_context.jsonl")
    with open(ctx_path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    ep = load_episode(episode_dir)
    img_cache = embed_images(records, jpeg_quality, max_width)
    t0 = records[0]["timestamp_s"] if records else 0.0

    cards = []
    for rec in records:
        seq = rec.get("seq", "?")
        tool = html.escape(str(rec.get("tool", "?")))
        dt = rec.get("timestamp_s", t0) - t0
        args = html.escape(json.dumps(rec.get("arguments") or {}, indent=2, ensure_ascii=False))
        resp = html.escape(pretty_blocks(rec.get("response_text_blocks") or []))
        imgs = []
        for p in rec.get("response_image_paths") or []:
            uri = img_cache.get(p, "")
            name = html.escape(os.path.basename(p))
            if uri:
                imgs.append(
                    f'<figure><a href="{uri}" target="_blank">'
                    f'<img src="{uri}" alt="{name}" loading="lazy"></a>'
                    f"<figcaption>{name}</figcaption></figure>"
                )
            else:
                imgs.append(f'<figure class="missing">missing: {html.escape(p)}</figure>')
        imgs_html = f'<div class="images">{"".join(imgs)}</div>' if imgs else ""
        cards.append(
            f'<section class="card" id="step-{seq}">'
            f'<header><span class="seq">#{seq}</span>'
            f'<span class="tool">{tool}</span>'
            f'<span class="time">t+{dt:6.1f}s</span></header>'
            f'<div class="cols"><div class="col"><h4>arguments</h4><pre>{args}</pre></div>'
            f'<div class="col"><h4>response</h4><pre>{resp}</pre></div></div>'
            f"{imgs_html}</section>"
        )

    success = ep.get("success")
    badge = ""
    if success is not None:
        cls = "ok" if success else "fail"
        badge = f'<span class="badge {cls}">{"SUCCESS" if success else "FAILED"}</span>'

    title = ep.get("episode_id") or os.path.basename(episode_dir.rstrip("/"))
    meta_rows = "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>"
        for k, v in [
            ("task", ep.get("task", "")),
            ("env_id", ep.get("env_id", "")),
            ("seed", ep.get("seed", "")),
            ("status", ep.get("status", "")),
            ("steps (tool calls)", len(records)),
            ("sim_step", (ep.get("result") or {}).get("sim_step", "")),
        ]
        if v != ""
    )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
  :root {{ --border:#e2e2e8; --bg:#f6f7f9; --accent:#2563eb; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
         background:var(--bg); color:#1f2430; }}
  .page {{ max-width:1100px; margin:0 auto; padding:24px 16px 64px; }}
  h1 {{ font-size:20px; margin:0 0 4px; word-break:break-all; }}
  .sub {{ color:#667; font-size:13px; margin-bottom:16px; }}
  table.meta {{ border-collapse:collapse; margin:12px 0 24px; font-size:14px; background:#fff;
                border:1px solid var(--border); border-radius:8px; overflow:hidden; }}
  table.meta th, table.meta td {{ padding:6px 14px; border-bottom:1px solid var(--border); text-align:left; }}
  table.meta th {{ color:#667; font-weight:600; white-space:nowrap; }}
  .badge {{ display:inline-block; padding:2px 10px; border-radius:999px; font-size:12px;
            font-weight:700; vertical-align:middle; margin-left:8px; }}
  .badge.ok {{ background:#dcfce7; color:#15803d; }}
  .badge.fail {{ background:#fee2e2; color:#b91c1c; }}
  .card {{ background:#fff; border:1px solid var(--border); border-radius:10px;
           margin-bottom:16px; padding:12px 16px; }}
  .card header {{ display:flex; align-items:baseline; gap:10px; margin-bottom:8px; }}
  .seq {{ color:#99a; font-variant-numeric:tabular-nums; }}
  .tool {{ font-weight:700; color:var(--accent); font-family:ui-monospace,Menlo,Consolas,monospace; }}
  .time {{ margin-left:auto; color:#99a; font-size:12px; font-variant-numeric:tabular-nums; }}
  .cols {{ display:grid; grid-template-columns:1fr 1fr; gap:12px; }}
  @media (max-width:800px) {{ .cols {{ grid-template-columns:1fr; }} }}
  h4 {{ margin:0 0 4px; font-size:11px; text-transform:uppercase; letter-spacing:.06em; color:#99a; }}
  pre {{ margin:0; padding:8px 10px; background:#f1f3f7; border-radius:6px; font-size:12px;
         overflow-x:auto; white-space:pre-wrap; word-break:break-word;
         font-family:ui-monospace,Menlo,Consolas,monospace; }}
  .images {{ display:flex; flex-wrap:wrap; gap:10px; margin-top:12px; }}
  figure {{ margin:0; max-width:340px; }}
  figure img {{ width:100%; border:1px solid var(--border); border-radius:6px; display:block; }}
  figcaption {{ font-size:11px; color:#889; margin-top:3px; word-break:break-all; }}
  figure.missing {{ color:#b91c1c; font-size:12px; }}
</style>
</head>
<body>
<div class="page">
  <h1>{html.escape(title)}{badge}</h1>
  <div class="sub">operator_context.jsonl &mdash; {len(records)} tool calls</div>
  <table class="meta">{meta_rows}</table>
  {''.join(cards)}
</div>
</body>
</html>
"""
    with open(out_path, "w") as f:
        f.write(doc)
    print(f"wrote {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB, "
          f"{len(records)} steps, {sum(1 for v in img_cache.values() if v)} images embedded)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode_dir")
    parser.add_argument("output", nargs="?", help="default: <episode_dir>/operator_context.html")
    parser.add_argument("--jpeg-quality", type=int, default=None,
                        help="re-encode images as JPEG at this quality (e.g. 80) to shrink output")
    parser.add_argument("--max-width", type=int, default=1280,
                        help="max image width when re-encoding (default: 1280)")
    args = parser.parse_args()
    out = args.output or os.path.join(args.episode_dir, "operator_context.html")
    render(args.episode_dir, out, jpeg_quality=args.jpeg_quality, max_width=args.max_width)
