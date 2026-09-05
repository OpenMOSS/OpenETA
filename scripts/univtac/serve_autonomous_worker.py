#!/usr/bin/env python3
"""One synchronous UniVTAC evaluation session for general OpenETA tools."""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo-root', type=Path, required=True)
    parser.add_argument('--source-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    base, _ = parser.parse_known_args(argv)
    root = base.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    app = task = server = session = None
    try:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args(argv)
        args.enable_cameras = True
        args.device = 'cuda:0'
        app = AppLauncher(args).app
        for p in (args.repo_root, args.source_root):
            sys.path.insert(0, str(p.resolve()))
        import yaml
        from envs.utils.env_parser import build_task_env_cfg, load_task_config

        from sim.envs.univtac.autonomous_session import AutonomousSession
        from sim.envs.univtac.trace import write_json
        config = yaml.safe_load(args.config.read_text())
        native, _ = load_task_config(args.source_root / 'task_config' / f"{config['task_config']}.yml")
        module, cfg, timing, _ = build_task_env_cfg(config['task'], native, config['task_config'], 'eval', device=args.device, save_dir=root/'native')
        # Save observations through OpenETA; the native control/render timing is unchanged.
        task = module.Task(cfg, mode='eval')
        task.reset(seed=args.seed)
        if not task.plan_success:
            raise RuntimeError('official reset/pre_move failed')
        task.mean_steps = cfg.step_lim
        session = AutonomousSession(task, config, root, args.seed)
        write_json(root/'native_configuration.json', {'mode': task.mode, 'step_lim': cfg.step_lim,
                   'dt': cfg.sim.dt, 'decimation': cfg.decimation, 'timing': vars(timing)})

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers.get('Content-Length',0))) or b'{}')
                if self.path == '/host_finalize':
                    session.finished = True
                    payload = {'ok': True}
                elif self.path == '/call':
                    payload = session.call(body['tool'], body.get('arguments', {}))
                else:
                    payload = {'ok': False, 'error': 'unknown endpoint'}
                data = json.dumps(payload, allow_nan=False).encode()
                self.send_response(200)
                self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        server = HTTPServer(('127.0.0.1',0),Handler)
        server.timeout = 1
        write_json(root/'ready.json',{'worker_url':f'http://127.0.0.1:{server.server_port}', 'reset_valid': True})
        deadline = time.monotonic() + config['codex_timeout_seconds'] + config['shutdown_timeout_seconds']
        while not session.finished and time.monotonic() < deadline:
            server.handle_request()
        session.finalize('worker_deadline' if not session.finished else None)
        return 0
    except Exception as exc:  # noqa: BLE001 -- retain runtime failure evidence
        (root/'worker_error.json').write_text(json.dumps({'error': str(exc), 'traceback': traceback.format_exc()}))
        if session:
            session.infrastructure_error = str(exc)
            session.finalize('infrastructure_error')
        return 1
    finally:
        if server:
            server.server_close()
        if task:
            task.close()
        if app:
            app.close()


if __name__ == '__main__':
    raise SystemExit(main())
