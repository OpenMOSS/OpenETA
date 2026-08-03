"""Local Web GUI for configuring OpenETA planner providers."""

from __future__ import annotations

import argparse
import json
import socket
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from adapter.protocol import JsonDict
from adapter.protocol import CameraFrame, EnvObservation, RobotState
from agent.backends.planner import (
    OpenAICompatiblePlannerBackend,
    OpenAICompatiblePlannerBackendConfig,
    list_openai_compatible_models,
)
from agent.backends.provider_config import (
    PlannerProviderConfig,
    load_planner_provider_config,
    write_env_file,
)
from agent.runtime.planner import ToolCallingPlanner
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.tools.handlers import bind_dummy_tool_handlers
from agent.tools.registry import build_default_tool_registry


class ProviderConfigHandler(BaseHTTPRequestHandler):
    """HTTP handler for the local provider configuration page."""

    server: "ProviderConfigServer"

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path == "/" or self.path.startswith("/?"):
            self._send_html(INDEX_HTML)
            return
        if self.path == "/api/config":
            self._send_json(_config_payload())
            return
        if self.path == "/api/models":
            self._handle_models()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path == "/api/config":
            self._handle_save_config()
            return
        if self.path == "/api/smoke":
            self._handle_smoke()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def log_message(self, format: str, *args: Any) -> None:
        if self.server.quiet:
            return
        super().log_message(format, *args)

    def _handle_models(self) -> None:
        config = load_planner_provider_config()
        backend_config = OpenAICompatiblePlannerBackendConfig.from_provider_config(config)
        try:
            models = list_openai_compatible_models(backend_config)
        except Exception as exc:  # noqa: BLE001 - local GUI returns structured errors.
            self._send_json(
                {
                    "ok": False,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "config": config.redacted(),
                },
                status=HTTPStatus.BAD_REQUEST,
            )
            return
        self._send_json({"ok": True, "models": models, "config": config.redacted()})

    def _handle_save_config(self) -> None:
        payload = self._read_json()
        existing = load_planner_provider_config(dotenv_path=self.server.env_path)
        api_key = str(payload.get("api_key", "")).strip() or existing.api_key
        config = PlannerProviderConfig(
            provider=str(payload.get("provider", "openai-compatible")).strip(),
            model=str(payload.get("model", "")).strip(),
            api_base=str(payload.get("api_base", "")).strip().rstrip("/"),
            api_key=api_key,
            timeout_s=_float_or_default(payload.get("timeout_s"), 60.0),
            max_attempts=_optional_positive_int(payload.get("max_attempts")) or 3,
            retry_backoff_s=max(
                0.0,
                _float_or_default(payload.get("retry_backoff_s"), 0.5),
            ),
            context_window_tokens=_optional_positive_int(
                payload.get("context_window_tokens")
            ),
        )
        missing = config.missing_fields()
        if missing:
            self._send_json(
                {
                    "ok": False,
                    "error": "Missing required provider config fields.",
                    "missing_fields": missing,
                    "config": config.redacted(),
                },
                status=HTTPStatus.BAD_REQUEST,
            )
            return
        write_env_file(config, self.server.env_path)
        self._send_json({"ok": True, "config": config.redacted()})

    def _handle_smoke(self) -> None:
        payload = self._read_json()
        config = load_planner_provider_config(dotenv_path=self.server.env_path)
        model = str(payload.get("model", "")).strip()
        if model:
            config.model = model
        backend_config = OpenAICompatiblePlannerBackendConfig.from_provider_config(config)
        try:
            action = _run_planner_smoke(backend_config)
        except Exception as exc:  # noqa: BLE001 - local GUI returns structured errors.
            self._send_json(
                {
                    "ok": False,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "config": config.redacted(),
                },
                status=HTTPStatus.BAD_REQUEST,
            )
            return
        self._send_json(
            {
                "ok": True,
                "config": config.redacted(),
                "action": _action_summary(action.command),
            }
        )

    def _read_json(self) -> JsonDict:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        body = self.rfile.read(length).decode("utf-8")
        payload = json.loads(body)
        return payload if isinstance(payload, dict) else {}

    def _send_html(self, content: str) -> None:
        data = content.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: JsonDict, *, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class ProviderConfigServer(ThreadingHTTPServer):
    """Threaded local server carrying app configuration."""

    def __init__(
        self,
        server_address: tuple[str, int],
        handler_cls: type[BaseHTTPRequestHandler],
        *,
        env_path: Path,
        quiet: bool,
    ) -> None:
        super().__init__(server_address, handler_cls)
        self.env_path = env_path
        self.quiet = quiet


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--env-path", default=".env")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    port = _pick_port(args.host, args.port)
    server = ProviderConfigServer(
        (args.host, port),
        ProviderConfigHandler,
        env_path=Path(args.env_path),
        quiet=args.quiet,
    )
    print(f"OpenETA provider GUI: http://{args.host}:{port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


def _config_payload() -> JsonDict:
    config = load_planner_provider_config()
    return {"ok": True, "config": config.redacted(), "missing_fields": config.missing_fields()}


def _run_planner_smoke(config: OpenAICompatiblePlannerBackendConfig):
    tools = build_default_tool_registry()
    bind_dummy_tool_handlers(tools, include_dummy_safety=False)
    planner = ToolCallingPlanner(OpenAICompatiblePlannerBackend(config))
    runtime = OpenEtaAgentRuntime(planner=planner, tools=tools)
    runtime.start_session(task=_task())
    return runtime.act(_observation())


def _task() -> str:
    return (
        "Call exactly one available read-only tool to inspect the cube. "
        "Prefer sam3 with image='front' and prompt='cube'."
    )


def _observation() -> EnvObservation:
    return EnvObservation(
        task=_task(),
        cameras=[CameraFrame(frame_id="front", rgb=[[[0, 0, 0]]], depth=[[1.0]])],
        robot=RobotState(end_effector_pose={"xyz": [0.0, 0.0, 0.5]}),
        objects=[{"name": "cube", "position": [0.2, 0.0, 0.0]}],
        metadata={"step_idx": 1},
    )


def _action_summary(command: JsonDict) -> JsonDict:
    request = command.get("request", {})
    tool_calls = command.get("tool_calls", [])
    planner_metadata = command.get("metadata", {}).get("planner_metadata", {})
    return {
        "status": command.get("status"),
        "request": request,
        "tool_calls": [
            {
                "name": call.get("name"),
                "status": call.get("status"),
                "result": call.get("result"),
            }
            for call in tool_calls
            if isinstance(call, dict)
        ],
        "backend": {
            "provider": planner_metadata.get("backend_provider"),
            "model": planner_metadata.get("backend_model"),
            "usage": planner_metadata.get("backend_details", {}).get("usage", {}),
        },
    }


def _pick_port(host: str, preferred: int) -> int:
    for port in range(preferred, preferred + 20):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind((host, port))
            except OSError:
                continue
            return port
    raise RuntimeError(f"No free port found from {preferred} to {preferred + 19}.")


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_positive_int(value: Any) -> int | None:
    try:
        parsed = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>OpenETA Provider Config</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f7f7f4;
      --panel: #ffffff;
      --line: #d8d6ce;
      --text: #202124;
      --muted: #686b70;
      --accent: #246b68;
      --accent-2: #7a4f15;
      --danger: #a33a32;
      --ok: #2f6f3e;
      --shadow: 0 1px 2px rgba(0,0,0,.06);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--text);
    }
    header {
      border-bottom: 1px solid var(--line);
      background: #fbfbf8;
    }
    .wrap {
      max-width: 1120px;
      margin: 0 auto;
      padding: 18px 22px;
    }
    .top {
      display: flex;
      justify-content: space-between;
      gap: 18px;
      align-items: center;
    }
    h1 {
      font-size: 22px;
      line-height: 1.2;
      margin: 0;
      font-weight: 700;
      letter-spacing: 0;
    }
    .sub {
      color: var(--muted);
      font-size: 13px;
      margin-top: 4px;
    }
    main.wrap {
      display: grid;
      grid-template-columns: minmax(340px, 430px) 1fr;
      gap: 18px;
      align-items: start;
      padding-top: 18px;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }
    .section-head {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      padding: 14px 16px;
      border-bottom: 1px solid var(--line);
    }
    h2 {
      font-size: 15px;
      margin: 0;
      letter-spacing: 0;
    }
    form, .body {
      padding: 16px;
    }
    label {
      display: block;
      font-size: 12px;
      font-weight: 650;
      color: #383a3d;
      margin: 0 0 6px;
    }
    input, select {
      width: 100%;
      border: 1px solid #c8c6bf;
      background: #fff;
      color: var(--text);
      border-radius: 6px;
      height: 38px;
      padding: 8px 10px;
      font-size: 14px;
      outline: none;
    }
    input:focus, select:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(36, 107, 104, .14);
    }
    .field { margin-bottom: 13px; }
    .row {
      display: grid;
      grid-template-columns: 1fr 120px;
      gap: 10px;
    }
    .actions {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 16px;
    }
    button {
      appearance: none;
      border: 1px solid #b9b6ad;
      background: #f8f8f5;
      color: var(--text);
      border-radius: 6px;
      height: 36px;
      padding: 0 12px;
      font-size: 13px;
      font-weight: 650;
      cursor: pointer;
    }
    button.primary {
      border-color: var(--accent);
      background: var(--accent);
      color: white;
    }
    button.secondary {
      border-color: #c7b07c;
      color: var(--accent-2);
    }
    button:disabled {
      opacity: .55;
      cursor: not-allowed;
    }
    .status {
      min-height: 36px;
      border-top: 1px solid var(--line);
      padding: 11px 16px;
      font-size: 13px;
      color: var(--muted);
    }
    .status.ok { color: var(--ok); }
    .status.err { color: var(--danger); }
    .kv {
      display: grid;
      grid-template-columns: 150px 1fr;
      gap: 8px 12px;
      font-size: 13px;
    }
    .kv div:nth-child(odd) { color: var(--muted); }
    pre {
      margin: 0;
      min-height: 360px;
      max-height: calc(100vh - 230px);
      overflow: auto;
      background: #202124;
      color: #f1f3f4;
      border-radius: 0 0 8px 8px;
      padding: 14px 16px;
      font-size: 12px;
      line-height: 1.55;
      white-space: pre-wrap;
      word-break: break-word;
    }
    .pill {
      display: inline-flex;
      align-items: center;
      height: 24px;
      padding: 0 8px;
      border-radius: 999px;
      border: 1px solid var(--line);
      background: #fafaf7;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }
    @media (max-width: 820px) {
      main.wrap { grid-template-columns: 1fr; }
      .top { align-items: flex-start; flex-direction: column; }
      .row { grid-template-columns: 1fr; }
      pre { max-height: none; }
    }
  </style>
</head>
<body>
  <header>
    <div class="wrap top">
      <div>
        <h1>OpenETA Provider Config</h1>
        <div class="sub">Local planner backend configuration for closed-loop tool calling</div>
      </div>
      <span id="health" class="pill">Loading</span>
    </div>
  </header>
  <main class="wrap">
    <section>
      <div class="section-head">
        <h2>Provider</h2>
        <span id="missing" class="pill">Checking</span>
      </div>
      <form id="configForm">
        <div class="field">
          <label for="provider">Provider</label>
          <input id="provider" name="provider" value="openai-compatible" autocomplete="off" />
        </div>
        <div class="field">
          <label for="api_base">API Base</label>
          <input id="api_base" name="api_base" placeholder="https://example.com" autocomplete="off" />
        </div>
        <div class="field">
          <label for="api_key">API Key</label>
          <input id="api_key" name="api_key" type="password" placeholder="Stored in local .env" autocomplete="off" />
        </div>
        <div class="row">
          <div class="field">
            <label for="model">Model</label>
            <select id="model" name="model"></select>
          </div>
          <div class="field">
            <label for="timeout_s">Timeout</label>
            <input id="timeout_s" name="timeout_s" value="60" inputmode="decimal" />
          </div>
        </div>
        <div class="field">
          <label for="context_window_tokens">Context Window Tokens</label>
          <input id="context_window_tokens" name="context_window_tokens" inputmode="numeric" placeholder="Optional" />
        </div>
        <div class="row">
          <div class="field">
            <label for="max_attempts">Provider Attempts</label>
            <input id="max_attempts" name="max_attempts" value="3" inputmode="numeric" />
          </div>
          <div class="field">
            <label for="retry_backoff_s">Retry Backoff</label>
            <input id="retry_backoff_s" name="retry_backoff_s" value="0.5" inputmode="decimal" />
          </div>
        </div>
        <div class="actions">
          <button class="primary" type="submit">Save</button>
          <button class="secondary" type="button" id="loadModels">Load Models</button>
          <button type="button" id="runSmoke">Run Smoke</button>
        </div>
      </form>
      <div id="status" class="status">Ready</div>
    </section>
    <section>
      <div class="section-head">
        <h2>Runtime Output</h2>
        <span id="resultStatus" class="pill">Idle</span>
      </div>
      <div class="body">
        <div class="kv" id="summary"></div>
      </div>
      <pre id="output">{}</pre>
    </section>
  </main>
  <script>
    const $ = (id) => document.getElementById(id);
    const state = { config: null, models: [] };

    async function request(path, options = {}) {
      const response = await fetch(path, {
        ...options,
        headers: { 'Content-Type': 'application/json', ...(options.headers || {}) },
      });
      const data = await response.json();
      if (!response.ok || data.ok === false) {
        throw new Error(data.error || response.statusText);
      }
      return data;
    }

    function setStatus(text, kind = '') {
      $('status').textContent = text;
      $('status').className = `status ${kind}`;
    }

    function writeOutput(payload) {
      $('output').textContent = JSON.stringify(payload, null, 2);
    }

    function renderConfig(payload) {
      state.config = payload.config;
      $('provider').value = payload.config.provider || 'openai-compatible';
      $('api_base').value = payload.config.api_base || '';
      $('api_key').value = '';
      $('api_key').placeholder = payload.config.api_key ? payload.config.api_key : 'Stored in local .env';
      $('timeout_s').value = payload.config.timeout_s || 60;
      $('max_attempts').value = payload.config.max_attempts || 3;
      $('retry_backoff_s').value = payload.config.retry_backoff_s ?? 0.5;
      $('context_window_tokens').value = payload.config.context_window_tokens || '';
      if (payload.config.model) setModelOptions([payload.config.model], payload.config.model);
      const missing = payload.missing_fields || [];
      $('missing').textContent = missing.length ? `Missing ${missing.length}` : 'Complete';
      $('health').textContent = 'Ready';
      renderSummary(payload.config);
      writeOutput(payload);
    }

    function renderSummary(config) {
      const rows = [
        ['Provider', config.provider || '-'],
        ['API Base', config.api_base || '-'],
        ['Model', config.model || '-'],
        ['API Key', config.api_key || '-'],
        ['Provider Attempts', config.max_attempts || 3],
        ['Retry Backoff', config.retry_backoff_s ?? 0.5],
        ['Context Window', config.context_window_tokens || '-'],
      ];
      $('summary').innerHTML = rows.map(([k, v]) => `<div>${k}</div><div>${escapeHtml(String(v))}</div>`).join('');
    }

    function setModelOptions(models, selected = '') {
      state.models = Array.from(new Set(models.filter(Boolean)));
      if (selected && !state.models.includes(selected)) state.models.unshift(selected);
      $('model').innerHTML = state.models.map((model) => {
        const flag = model === selected ? ' selected' : '';
        return `<option value="${escapeHtml(model)}"${flag}>${escapeHtml(model)}</option>`;
      }).join('');
      if (!state.models.length) {
        $('model').innerHTML = '<option value="">Load models</option>';
      }
    }

    function formPayload(includeKey = true) {
      return {
        provider: $('provider').value.trim(),
        api_base: $('api_base').value.trim(),
        api_key: includeKey ? $('api_key').value.trim() : '',
        model: $('model').value.trim(),
        timeout_s: Number($('timeout_s').value || 60),
        max_attempts: Number($('max_attempts').value || 3),
        retry_backoff_s: Number($('retry_backoff_s').value || 0),
        context_window_tokens: Number($('context_window_tokens').value || 0) || null,
      };
    }

    function escapeHtml(value) {
      return value
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;');
    }

    async function loadConfig() {
      try {
        const payload = await request('/api/config');
        renderConfig(payload);
        setStatus('Config loaded', 'ok');
      } catch (error) {
        $('health').textContent = 'Error';
        setStatus(error.message, 'err');
      }
    }

    async function saveConfig(event) {
      event.preventDefault();
      setStatus('Saving');
      try {
        const payload = await request('/api/config', {
          method: 'POST',
          body: JSON.stringify(formPayload(true)),
        });
        renderConfig({ ok: true, config: payload.config, missing_fields: [] });
        setStatus('Saved to local .env', 'ok');
      } catch (error) {
        setStatus(error.message, 'err');
      }
    }

    async function loadModels() {
      setStatus('Loading models');
      $('loadModels').disabled = true;
      try {
        const payload = await request('/api/models');
        setModelOptions(payload.models, payload.config.model);
        renderSummary({ ...payload.config, model: $('model').value });
        writeOutput({ models: payload.models, config: payload.config });
        setStatus(`Loaded ${payload.models.length} models`, 'ok');
      } catch (error) {
        setStatus(error.message, 'err');
      } finally {
        $('loadModels').disabled = false;
      }
    }

    async function runSmoke() {
      setStatus('Running smoke test');
      $('runSmoke').disabled = true;
      $('resultStatus').textContent = 'Running';
      try {
        const payload = await request('/api/smoke', {
          method: 'POST',
          body: JSON.stringify({ model: $('model').value.trim() }),
        });
        writeOutput(payload.action);
        renderSummary({ ...payload.config, model: payload.action.backend.model || payload.config.model });
        $('resultStatus').textContent = payload.action.status || 'Done';
        setStatus('Smoke test passed', 'ok');
      } catch (error) {
        $('resultStatus').textContent = 'Error';
        setStatus(error.message, 'err');
      } finally {
        $('runSmoke').disabled = false;
      }
    }

    $('configForm').addEventListener('submit', saveConfig);
    $('loadModels').addEventListener('click', loadModels);
    $('runSmoke').addEventListener('click', runSmoke);
    loadConfig();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
