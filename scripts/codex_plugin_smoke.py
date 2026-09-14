"""Install a private plugin snapshot and run one bounded ChatGPT-authenticated Codex episode.

This never changes the user's Codex config or installs into their normal cache.
The simulator URL must name a separately owned simulation service.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
if __package__ in (None, ""):
    sys.path.insert(0, str(REPO))

from tools.codex_controller import CONTROLLER_IDS, DEFAULT_CONTROLLER


def prepare(output, host_args, model, effort, codex, *, model_catalog_json=None):
    output.mkdir(parents=True, exist_ok=True)
    home = output / "codex-home"
    home.mkdir(mode=0o700)
    home.chmod(0o700)
    source = Path(os.environ.get("CODEX_HOME", str(Path.home() / ".codex"))) / "auth.json"
    auth = json.loads(source.read_text())
    if not auth.get("tokens") or auth.get("auth_mode") == "apikey":
        raise RuntimeError("This launcher requires saved ChatGPT login; API-key authentication is not accepted")
    auth_path = home / "auth.json"
    fd = os.open(auth_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as f: json.dump(auth, f)
    work = output / "operator-workspace"
    work.mkdir()
    market = output / "marketplace"
    shutil.copytree(REPO / "plugins/openeta", market / "plugins/openeta")
    if "--tool-profile" in host_args and host_args[host_args.index("--tool-profile")+1] == "atomic":
        shutil.rmtree(market / "plugins/openeta/skills/openeta-pick")
    (market / ".agents/plugins").mkdir(parents=True)
    shutil.copy2(REPO / ".agents/plugins/marketplace.json", market / ".agents/plugins/marketplace.json")
    # A native route can contain multiple normally budgeted motion stages.
    # Let the Host episode deadline stop it rather than severing stdio at 180 s.
    episode_timeout = (float(host_args[host_args.index('--timeout') + 1])
                       if '--timeout' in host_args else 150.)
    native_tool_timeout = max(180., episode_timeout + 30.)
    plugin_mcp = {"openeta": {
        "command": str(REPO / ".venv/bin/python"), "args": ["-m", "tools.codex_mcp_server"],
        "env": {"PYTHONPATH": str(REPO), "PYTHONDONTWRITEBYTECODE": "1",
                "OPENETA_CODEX_HOST_ARGS": json.dumps(host_args)},
        "startup_timeout_sec": 180, "tool_timeout_sec": native_tool_timeout,
    }}
    (market / "plugins/openeta/.mcp.json").write_text(json.dumps(plugin_mcp, indent=2) + "\n")
    config = (f'model = {json.dumps(model)}\nmodel_reasoning_effort = {json.dumps(effort)}\n'
              'model_provider = "openai"\nforced_login_method = "chatgpt"\n'
              # Plugin MCP initialization includes creating/resetting the sim.
              # Codex's default 1 s optional grace can start a tool-less turn.
              # Zero waits for each service's bounded startup_timeout_sec.
              'mcp_optional_startup_grace_ms = 0\n'
              'web_search = "disabled"\n'
              '[features]\nmemories = false\nshell_tool = false\nmulti_agent = false\n'
              '[history]\npersistence = "none"\n'
              '[plugins."openeta@openeta-codex-smoke".mcp_servers.openeta]\n'
              'enabled = true\ndefault_tools_approval_mode = "approve"\n')
    if model_catalog_json is not None:
        catalog = json.loads(Path(model_catalog_json).read_text())
        if not any(m.get("slug") == model for m in catalog.get("models", []) if isinstance(m, dict)):
            raise RuntimeError(f"Pinned model catalog does not contain {model}")
        catalog_path = home / "model-catalog.json"
        catalog_path.write_text(json.dumps(catalog, ensure_ascii=False) + "\n")
        config = f'model_catalog_json = {json.dumps(str(catalog_path))}\n' + config
    (home / "config.toml").write_text(config)
    env = dict(os.environ)
    # Plugin callbacks inherit the Codex process environment. Do not propagate
    # API keys, other provider selection or unrelated hook/plugin configuration.
    for key in list(env):
        if key.startswith(("OPENETA_LLM_", "OPENAI_", "CODEX_")):
            env.pop(key, None)
    env.update(CODEX_HOME=str(home), PYTHONDONTWRITEBYTECODE="1")
    with (output / "plugin-install.log").open("w") as log:
        for command in ([codex, "plugin", "marketplace", "add", str(market)],
                        [codex, "plugin", "add", "openeta@openeta-codex-smoke", "--json"]):
            subprocess.run(command, env=env, cwd=work, stdout=log, stderr=log, timeout=60, check=True)
    return home, work, env


def run(args):
    # Retain compatibility with programmatic callers of the original launcher.
    args.tool_profile = getattr(args, "tool_profile", "pick")
    output = args.output.resolve()
    host_args = ["--sim-url", args.sim_url, "--env-id", args.env_id,
                 "--tool-profile", args.tool_profile,
                 "--expected-controller", args.controller,
                 "--task", args.task, "--seed", str(args.seed),
                 "--output", str(output / "host"), "--timeout", str(args.timeout),
                 "--max-requests", str(args.max_requests),
                 "--sam3-url", args.sam3_url, "--anygrasp-url", args.anygrasp_url]
    process = None
    start = time.monotonic()
    summary = {"model": args.model, "reasoning_effort": args.effort,
               "tool_profile": args.tool_profile,
               "expected_controller": args.controller,
               "authentication": "chatgpt", "auxiliary_model_inference": "disabled",
               "limits": {"episode_timeout_s": args.timeout, "max_requests": args.max_requests,
                          "max_turns": args.max_requests, "max_tool_calls": args.max_requests},
               "status": "preparing", "model_usage": [], "non_mcp_actions": []}
    try:
        catalog_path = getattr(args, "model_catalog_json", None)
        prepare_options = {"model_catalog_json": catalog_path} if catalog_path else {}
        home, work, env = prepare(output, host_args, args.model, args.effort, args.codex, **prepare_options)
        if catalog_path:
            import hashlib
            summary["model_catalog"] = {
                "source_path": str(Path(catalog_path).resolve()),
                "private_copy_sha256": hashlib.sha256((home / "model-catalog.json").read_bytes()).hexdigest(),
            }
        prompt = (
            "Use the installed openeta plugin and openeta-pick skill for this bounded simulator test. "
            "Call episode_status, inspect the images, then work on this task: " + args.task + "\n"
            "Use only the native openeta MCP tools. No shell, web, other model calls, or repository inspection. "
            "After every world-changing command inspect returned evidence. Tool errors may be repaired. "
            "Call finish_episode once you have official success evidence or cannot continue. "
            "Do not claim task success without the Host's official evidence. "
            + ("This is a connection smoke only: inspect the scene, call observe, then finish_episode(success=false) with a concise visual description; do not move the robot."
               if args.observe_only else "")
        )
        if args.tool_profile == "atomic":
            skill = REPO / "plugins/openeta/skills/openeta-atomic/SKILL.md"
            # The restricted operator has no file-reading tool. Deliver the
            # selected skill body explicitly instead of relying on discovery.
            semantics = skill.read_text().split("---", 2)[2].strip()
            prompt = (args.task + "\n\n" + semantics + "\n\n"
                      "Use only the six OpenETA robot operations. Call episode_status first. "
                      "Finish with official success evidence or a concise failure reason.\n"
                      "Tool access: these operations may be exposed through Codex code mode rather than "
                      "direct top-level functions. You may use functions.exec and its tools/ALL_TOOLS "
                      "discovery to invoke only the installed OpenETA operations. Tool namespace wrappers "
                      "are allowed; they are not additional robot capabilities. Forward tool text and "
                      "images so you can inspect them. Do not use shell, filesystem access, network "
                      "access, other model calls, or other application tools. Discover/call episode_status "
                      "through the available wrapper before concluding tools are unavailable.")
            if args.observe_only:
                prompt += "\nConnection check only: observe once then finish_episode(success=false); do not move."
        command = [args.codex, "exec", "-C", str(work), "--skip-git-repo-check",
                   "--ephemeral", "--json", "-s", "read-only", "-m", args.model,
                   "-o", str(output / "final.txt"), prompt]
        (output / "prompt.txt").write_text(prompt + "\n")
        summary["status"] = "running"
        with (output / "codex-events.jsonl").open("w") as stdout, (output / "codex-stderr.log").open("w") as stderr:
            process = subprocess.Popen(command, env=env, cwd=work, stdin=subprocess.DEVNULL,
                                       stdout=stdout, stderr=stderr,
                                       start_new_session=True)
            try:
                summary["returncode"] = process.wait(timeout=args.timeout + 210)
                summary["status"] = "completed" if process.returncode == 0 else "codex_failed"
            except subprocess.TimeoutExpired:
                summary["status"] = "launcher_timeout"
        events = output / "codex-events.jsonl"
        if events.exists():
            for line in events.read_text().splitlines():
                try: event = json.loads(line)
                except json.JSONDecodeError: continue
                if event.get("usage"): summary["model_usage"].append(event["usage"])
                item = event.get("item") or {}
                if item.get("type") in {"command_execution", "web_search", "file_change"}:
                    summary["non_mcp_actions"].append({"id": item.get("id"), "type": item["type"]})
    except Exception as exc:
        summary.update(status="setup_failed", error=f"{type(exc).__name__}: {exc}")
    finally:
        if process is not None:
            # Close descendants even if the parent exited: MCP servers may still
            # own an episode. Group contains only processes launched for this run.
            try: os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError: pass
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=5)
            # Codex can finish before its MCP child's asynchronous close receipt
            # reaches disk. Wait briefly; missing acknowledgement stays unknown.
            status_path = output / "host/host-status.json"
            deadline = time.monotonic() + 10
            while status_path.exists() and time.monotonic() < deadline:
                if json.loads(status_path.read_text()).get("closed"):
                    break
                time.sleep(0.2)
        # The only credential copy belongs to this private, short-lived home.
        (output / "codex-home/auth.json").unlink(missing_ok=True)
        summary["elapsed_s"] = round(time.monotonic() - start, 3)
        p = output / "host/host-status.json"
        if p.exists(): summary["host"] = json.loads(p.read_text())
        host = summary.get("host", {})
        summary["task_success"] = host.get("official_task_success", False)
        summary["integration_passed"] = bool(
            summary["status"] == "completed" and host.get("closed")
            and (host.get("cleanup") or {}).get("ok") is True
            and host.get("tool_calls", 0) > 0 and not summary["non_mcp_actions"]
        )
        output.mkdir(parents=True, exist_ok=True)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["integration_passed"] else 1


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sim-url", required=True)
    p.add_argument("--tool-profile", choices=("pick", "atomic"), default="pick")
    p.add_argument("--controller", choices=CONTROLLER_IDS, default=DEFAULT_CONTROLLER,
                   help="Expected controller of the separately started simulator (default: Mink)")
    p.add_argument("--task", required=True)
    p.add_argument("--env-id", default="openeta/libero_libero_spatial_task0-v0")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sam3-url", default="")
    p.add_argument("--anygrasp-url", default="")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="gpt-5.6-sol")
    p.add_argument("--model-catalog-json", type=Path, default=None,
                   help="Copy a pinned Codex model catalog into this isolated run")
    p.add_argument("--effort", default="medium")
    p.add_argument("--timeout", type=float, default=1200)
    p.add_argument("--max-requests", type=int, default=80)
    p.add_argument("--codex", default="codex")
    p.add_argument("--observe-only", action="store_true")
    return p


def main():
    p = parser()
    args = p.parse_args()
    if args.output.exists(): p.error("--output must be a new directory for isolation")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
