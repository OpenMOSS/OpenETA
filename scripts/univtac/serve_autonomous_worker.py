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
    parser.add_argument('--reset-timing-only', action='store_true')
    parser.add_argument('--reset-timing', action='store_true')
    parser.add_argument('--uipc-native-report', action='store_true')
    parser.add_argument('--pad-scene-condition', choices=('O','K','Z'))
    base, _ = parser.parse_known_args(argv)
    if base.uipc_native_report and not base.reset_timing_only:
        parser.error('--uipc-native-report requires --reset-timing-only')
    if base.pad_scene_condition and not (base.reset_timing_only and base.uipc_native_report):
        parser.error('pad scene condition requires reset-only native reports')
    root = base.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    app = task = server = session = timing_probe = pad_probe = None
    if base.reset_timing_only or base.reset_timing:
        sys.path.insert(0, str(base.repo_root.resolve()))
        from sim.envs.univtac.reset_timing import ResetTiming
        timing_probe = ResetTiming(root)
    try:
        from isaaclab.app import AppLauncher
        AppLauncher.add_app_launcher_args(parser)
        args = parser.parse_args(argv)
        args.enable_cameras = True
        args.device = 'cuda:0'
        if timing_probe:
            with timing_probe.span('AppLauncher'):
                app = AppLauncher(args).app
        else:
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
        reset_limit = {'native_default_seconds':float(cfg.reset_time_limit),
                       'override_seconds':config.get('native_reset_time_limit_seconds')}
        if reset_limit['override_seconds'] is not None:
            cfg.reset_time_limit = float(reset_limit['override_seconds'])
        reset_limit['configured_seconds'] = float(cfg.reset_time_limit)
        write_json(root/'native_reset_limit.json', reset_limit)
        # Save observations through OpenETA; the native control/render timing is unchanged.
        if timing_probe:
            from tacex_uipc.sim.uipc_sim import UipcSim
            timing_probe.install_uipc_callbacks(UipcSim)
            if args.uipc_native_report:
                cfg.uipc_sim.logger_level = 'Info'
                timing_probe.enable_native_reports(UipcSim)
            from contextlib import nullcontext
            creation = nullcontext()
            if args.pad_scene_condition:
                from envs.utils.actor import ActorManager

                from sim.envs.univtac.pad_scene_diagnostic import PadSceneDiagnostic
                pad_probe = PadSceneDiagnostic(root, args.pad_scene_condition)
                creation = pad_probe.creation(ActorManager)
            with creation, timing_probe.span('Task.construct'):
                task = module.Task(cfg, mode='eval')
            reset_limit['actual_task_seconds'] = float(task.cfg.reset_time_limit)
            write_json(root/'native_reset_limit.json', reset_limit)
            if pad_probe:
                pad_probe.install(task,config)
            timing_probe.install_task(task)
            if args.uipc_native_report:
                timing_probe.record_uipc_configuration(task)
                timing_probe.export_native_report('after_construct', task.step_count)
            with timing_probe.span('Task.reset'):
                task.reset(seed=args.seed)
            write_json(root/'reset_diagnostic_result.json', {
                'record_scope':'initialization before ready',
                'reset_returned':True, 'seed':args.seed, 'codex_process_count':0,
                'task_body_actions':0, 'native_success_evaluated':False,
                'native_reset_time_limit_s':cfg.reset_time_limit,
                'initialization_steps':task.step_count,
                'initialization_physics_steps':task._physics_step_count})
            if args.uipc_native_report:
                timing_probe.export_native_report('reset_final', task.step_count)
            timing_probe.close()
            timing_probe = None
            if args.reset_timing_only:
                return 0
        else:
            task = module.Task(cfg, mode='eval')
            reset_limit['actual_task_seconds'] = float(task.cfg.reset_time_limit)
            write_json(root/'native_reset_limit.json', reset_limit)
            task.reset(seed=args.seed)
        if not task.plan_success:
            raise RuntimeError('official reset/pre_move failed')
        task.mean_steps = cfg.step_lim
        session = AutonomousSession(task, config, root, args.seed)
        write_json(root/'native_configuration.json', {'mode': task.mode, 'step_lim': cfg.step_lim,
                   'dt': cfg.sim.dt, 'decimation': cfg.decimation, 'timing': vars(timing)})

        released_at = None

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                nonlocal released_at
                body = json.loads(self.rfile.read(int(self.headers.get('Content-Length',0))) or b'{}')
                if self.path == '/host_finalize':
                    session.host_closed = True
                    payload = {'ok': True}
                elif self.path == '/host_release' and config.get('wait_for_operator_release'):
                    released_at = released_at or time.monotonic()
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
        deadline = time.monotonic() + (config['startup_timeout_seconds'] if config.get('wait_for_operator_release') else config['codex_timeout_seconds'] + config['shutdown_timeout_seconds'])
        # finish_episode stops motion but the MCP must stay readable until Codex
        # naturally exits (or its bounded terminal grace expires).
        while not getattr(session, 'host_closed', False) and time.monotonic() < deadline:
            server.handle_request()
            if released_at is not None:
                deadline = released_at + config['codex_timeout_seconds'] + config['shutdown_timeout_seconds']
        session.finalize('worker_deadline' if time.monotonic() >= deadline else None)
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
        if session and session.recorder:
            session.recorder.close()
        if timing_probe and base.uipc_native_report:
            timing_probe.export_native_report('reset_final', getattr(task, 'step_count', None))
        if pad_probe:
            pad_probe.close()
        if task:
            task.close()
        if timing_probe:
            timing_probe.close()
        if app:
            app.close()


if __name__ == '__main__':
    raise SystemExit(main())
