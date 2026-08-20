"""End-to-end R1Pro collision check, self-orchestrating across two interpreters.

cuRobo is installed in the libero venv; BEHAVIOR/Isaac Sim runs under its own
conda env.  They cannot share a process, which mirrors deployment: the MCP
server holds the checker and the bench worker holds the simulator, talking over
HTTP.  So the work is staged, and by default this script drives every stage
itself -- environment preflight, config generation, both interpreters, and
optionally a real MCP server -- so a reviewer runs one command.

  (default)  preflight -> --dump -> --check
  --dump     (behavior env)  boot the real activity, write joint_positions +
                             joint_names through RobotState serialisation
  --check    (libero venv)   load that payload, run the real CollisionChecker
  --via-mcp  (libero venv)   additionally start a real MCP server and call
                             create_env/move_to over the wire, so the assertion
                             covers the deployed path rather than an in-process
                             object

Staging it this way is what makes the result meaningful: --check sees exactly
the bytes the server would receive, so a name dropped at the wire boundary
shows up here rather than passing on a shared in-process object.

The discriminating evidence is the same pose reading clear in an empty world and
in collision inside an enclosing box, plus cuRobo's FK for the eef agreeing with
the pose OmniGibson reports independently -- a mis-permuted q would still be
finite and would still flag the box, but would not agree on the eef.  Two
negative controls guard that agreement: perturbing a right-arm joint must leave
the left eef fixed (proving the check can fail), and reordering the names must
change the mapped vector (proving names drive the permutation, not a stale
cache).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import traceback

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Keep artifacts off /tmp when a job dir is available: parallel runs on the same
# machine otherwise clobber each other's payloads.
_JOB = os.environ.get("CLAUDE_JOB_DIR", "")
OUTDIR = os.environ.get("R1PRO_OUTDIR") or (
    os.path.join(_JOB, "tmp") if _JOB else "/tmp")
PAYLOAD = os.path.join(OUTDIR, "r1pro_payload.json")
OUT = os.path.join(OUTDIR, "r1pro_live.json")
ACTIVITY = os.environ.get("BENCH_ACTIVITY", "picking_up_trash")

BEHAVIOR_PY = os.path.join(REPO, "sim", "venvs", "behavior", "bin", "python")
LIBERO_PY = os.path.join(REPO, "sim", "venvs", "libero", "bin", "python")


def _run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Run a child stage with this repo importable and no inherited PYTHONPATH.

    The two interpreters have incompatible site-packages, so an inherited
    PYTHONPATH from the parent shell is the single most common way this fails.
    """
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["R1PRO_OUTDIR"] = OUTDIR
    return subprocess.run(cmd, env=env, **kw)


def preflight() -> dict:
    """Check both interpreters and the generated cuRobo config before booting.

    Isaac Sim takes minutes to start, so every precondition that can be checked
    in milliseconds is checked first.  The cuRobo config is generated into
    cuRobo's own install tree rather than the repo, so a venv or submodule
    rebuild silently removes it -- that is worth detecting here, and fixing
    automatically, rather than surfacing as a disabled checker much later.
    """
    rep: dict = {"stage": "preflight", "ok": False}
    rep["behavior_python"] = BEHAVIOR_PY
    rep["libero_python"] = LIBERO_PY
    rep["behavior_python_exists"] = os.path.exists(BEHAVIOR_PY)
    rep["libero_python_exists"] = os.path.exists(LIBERO_PY)
    os.makedirs(OUTDIR, exist_ok=True)
    rep["outdir"] = OUTDIR

    if not rep["libero_python_exists"]:
        rep["error"] = (f"libero venv missing at {LIBERO_PY} — "
                        "run scripts/setup_envs.sh")
        return rep
    if not rep["behavior_python_exists"]:
        rep["error"] = (f"behavior env missing at {BEHAVIOR_PY} — "
                        "run scripts/setup_behavior.sh")
        return rep

    # Where cuRobo will look for the robot config, asked of cuRobo itself.
    probe = (
        "from curobo.util_file import get_robot_configs_path, join_path;"
        "import os,sys;"
        "p=join_path(get_robot_configs_path(),'r1pro.yml');"
        "print('CFG',p,os.path.exists(p))"
    )
    r = _run([LIBERO_PY, "-c", probe], capture_output=True, text=True)
    line = [l for l in r.stdout.splitlines() if l.startswith("CFG")]
    if not line:
        rep["error"] = f"could not query cuRobo config path: {r.stderr[-300:]}"
        return rep
    _, path, exists = line[0].split(" ", 2)
    rep["curobo_config_path"] = path
    rep["curobo_config_present"] = exists.strip() == "True"

    if not rep["curobo_config_present"]:
        gen = os.path.join(REPO, "scripts", "gen_r1pro_curobo.py")
        rep["generated_config"] = True
        g = _run([LIBERO_PY, gen], capture_output=True, text=True)
        rep["generator_rc"] = g.returncode
        if g.returncode != 0:
            rep["error"] = f"gen_r1pro_curobo.py failed: {g.stderr[-400:]}"
            return rep
        rep["curobo_config_present"] = os.path.exists(path)
        if not rep["curobo_config_present"]:
            rep["error"] = "generator reported success but config is absent"
            return rep

    rep["nvidia_smi"] = bool(shutil.which("nvidia-smi"))
    rep["ok"] = True
    return rep


def dump() -> int:
    """Stage 1, behavior env: capture a real observation as wire bytes."""
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
    sys.path.insert(0, REPO)

    rep: dict = {"activity": ACTIVITY, "ok": False}
    try:
        import numpy as np

        from adapter.protocol import EnvObservation
        from sim.envs.behavior.direct_env import BehaviorDirectEnv

        env = BehaviorDirectEnv(ACTIVITY, seed=0, render_mode="rgb_array",
                                image_width=64, image_height=64)
        env.reset()

        proprio = env._structured_proprio()
        # Serialise the way the worker does, so a field the wire drops is
        # absent here too rather than being smuggled through in memory.
        obs = EnvObservation.from_dict({"proprio": proprio})
        mcp = obs.to_mcp_dict()
        rep["robot"] = mcp.get("robot", {})
        rep["n_joints"] = len(rep["robot"].get("joint_positions") or [])
        rep["n_names"] = len(rep["robot"].get("joint_names") or [])

        # Independent eef ground truth, in the robot's own frame.
        robot = env._env.robots[0]
        og_pos, _ = robot.get_eef_pose("left")
        base_pos, base_quat = robot.get_position_orientation()
        og = np.asarray([float(v) for v in list(og_pos)[:3]])
        base = np.asarray([float(v) for v in list(base_pos)[:3]])
        x, y, z, w = (float(v) for v in list(base_quat)[:4])
        rot = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        # Full precision: rounding here would cap the FK comparison's resolution
        # at the rounding step and make exact agreement look tautological.
        rep["eef_left_in_base"] = [float(v) for v in rot.T @ (og - base)]

        # get_position_orientation() returns the root prim (base_footprint_x),
        # while cuRobo's root is the URDF base_link.  Those coincide only while
        # the six virtual base joints are ~0, which holds right after reset but
        # is an assumption -- so record the evidence instead of implying it.
        jp_all = rep["robot"].get("joint_positions") or []
        rep["base_joint_values"] = [round(float(v), 6) for v in jp_all[:6]]
        rep["base_at_origin"] = bool(
            jp_all and max(abs(float(v)) for v in jp_all[:6]) < 0.01)
        rep["ok"] = True
    except BaseException as exc:  # noqa: BLE001
        rep["error"] = f"{type(exc).__name__}: {exc}"[:400]
        rep["traceback"] = traceback.format_exc()[-1500:]

    os.makedirs(OUTDIR, exist_ok=True)
    with open(PAYLOAD, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"WROTE {PAYLOAD} ok={rep['ok']} names={rep.get('n_names')} "
          f"{rep.get('error', '')}", flush=True)
    return 0 if rep["ok"] else 1


def check() -> int:
    """Stage 2, libero venv: run the real checker on those bytes."""
    sys.path.insert(0, REPO)
    rep: dict = {"stage": "check", "ok": False}
    try:
        with open(PAYLOAD) as f:
            payload = json.load(f)
        if not payload.get("ok"):
            rep["error"] = f"stage 1 failed: {payload.get('error')}"
            raise RuntimeError(rep["error"])

        from sim.mcp_server.collision import CollisionChecker

        robot = payload["robot"]
        jp = [float(v) for v in robot.get("joint_positions") or []]
        names = [str(n) for n in robot.get("joint_names") or []]
        rep["n_joints"] = len(jp)
        rep["n_names"] = len(names)
        rep["names_survived_wire"] = bool(names)

        checker = CollisionChecker("behavior")
        rep["checker_available"] = bool(checker._available)
        rep["robot_config"] = checker._robot_config

        det_empty, info_empty = checker.check(jp, [], joint_names=names)
        wall = [{"name": "wall", "position": [0.0, 0.0, 0.5],
                 "dims": [6.0, 6.0, 6.0]}]
        det_wall, info_wall = checker.check(jp, wall, joint_names=names)

        rep["empty"] = {"detected": det_empty,
                        "world_pen": info_empty.get("max_world_penetration"),
                        "self_pen": info_empty.get("max_self_penetration"),
                        "world_checked": info_empty.get("world_checked"),
                        "reason": info_empty.get("reason")}
        rep["enclosed"] = {"detected": det_wall,
                           "world_pen": info_wall.get("max_world_penetration"),
                           "world_checked": info_wall.get("world_checked"),
                           "reason": info_wall.get("reason")}
        rep["curobo_dof"] = checker._arm_dof
        rep["permutation"] = checker._joint_permutation

        # Refusals must be demonstrated, not assumed.  Note what is *not*
        # claimed here: a reordered name list is a valid bijection containing
        # every joint cuRobo needs, so it cannot be rejected -- it is caught
        # below by asserting the mapping actually follows it.
        _, info_none = checker.check(jp, [], joint_names=[])
        rep["missing_names_rejected"] = bool(info_none.get("reason"))
        bogus = list(names)
        bogus[names.index("left_arm_joint1")] = "not_a_joint"
        _, info_bogus = checker.check(jp, [], joint_names=bogus)
        rep["absent_joint_rejected"] = bool(info_bogus.get("reason"))
        rep["absent_joint_reason"] = info_bogus.get("reason")

        # FK cross-check against OmniGibson's independent eef reading.
        import torch
        q_mapped, map_err = checker._map_by_name(jp, names)
        rep["map_error"] = map_err
        if q_mapped:
            rw = checker._ensure_robot_world()
            cn = [str(n) for n in rw.kinematics.joint_names]

            def fk(q: list[float]):
                t = torch.tensor([q], device=rw.tensor_args.device,
                                 dtype=rw.tensor_args.dtype)
                s = rw.get_kinematics(t)
                return s.ee_position.detach().cpu().numpy().reshape(-1)[:3]

            ee = fk(q_mapped)
            truth = payload.get("eef_left_in_base") or []
            rep["curobo_left_eef"] = [round(float(v), 6) for v in ee]
            rep["omnigibson_left_eef"] = [round(float(v), 6) for v in truth]
            rep["ee_link"] = str(getattr(rw.kinematics, "ee_link", ""))
            if len(truth) == 3:
                delta = [float(ee[i]) - float(truth[i]) for i in range(3)]
                rep["fk_delta"] = [round(d, 8) for d in delta]
                rep["fk_delta_norm"] = round(sum(d * d for d in delta) ** 0.5, 8)
                # 1 cm, not 5 cm: any residual base_footprint_x->base_link
                # offset lands in the low centimetres, so a 5 cm threshold
                # cannot distinguish agreement from a frame mismatch.  Genuine
                # agreement here is ~1e-5 m.
                rep["fk_agrees_1cm"] = bool(rep["fk_delta_norm"] < 0.01)
            rep["base_at_origin"] = payload.get("base_at_origin")

            # Control 1: the check must be able to fail.  ee_link is on the left
            # chain, so a right-arm joint must not move it -- if it did, the
            # agreement above would be insensitive to a left/right swap and
            # would prove nothing.
            qr = list(q_mapped)
            qr[cn.index("right_arm_joint1")] += 0.3
            rep["right_arm_moves_left_eef_m"] = round(
                float(((fk(qr) - ee) ** 2).sum() ** 0.5), 8)
            ql = list(q_mapped)
            ql[cn.index("left_arm_joint1")] += 0.3
            rep["left_arm_moves_left_eef_m"] = round(
                float(((fk(ql) - ee) ** 2).sum() ** 0.5), 8)
            rep["fk_is_live"] = bool(rep["left_arm_moves_left_eef_m"] > 0.05
                                     and rep["right_arm_moves_left_eef_m"] < 1e-6)

            # Control 2: the permutation must follow the names it was given, not
            # a cache built on the first call.  Swapping two names must move
            # their values with them.
            swapped = list(names)
            i = swapped.index("left_arm_joint1")
            j = swapped.index("right_arm_joint1")
            swapped[i], swapped[j] = swapped[j], swapped[i]
            q_swapped, _ = checker._map_by_name(jp, swapped)
            rep["reorder_remaps"] = bool(q_swapped and q_swapped != q_mapped)
            q_restored, _ = checker._map_by_name(jp, names)
            rep["reorder_is_reversible"] = bool(q_restored == q_mapped)

        rep["ok"] = bool(
            rep["names_survived_wire"] and rep["checker_available"]
            and not det_empty and det_wall
            and info_wall.get("world_checked")
            and rep["missing_names_rejected"] and rep["absent_joint_rejected"]
            and rep.get("fk_agrees_1cm") and rep.get("fk_is_live")
            and rep.get("reorder_remaps") and rep.get("reorder_is_reversible")
            and rep.get("base_at_origin")
        )
    except BaseException as exc:  # noqa: BLE001
        rep.setdefault("error", f"{type(exc).__name__}: {exc}"[:400])
        rep["traceback"] = traceback.format_exc()[-1500:]

    os.makedirs(OUTDIR, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"WROTE {OUT} ok={rep['ok']} {rep.get('error', '')}", flush=True)
    return 0 if rep["ok"] else 1


def via_mcp() -> int:
    """Stage 3, libero venv: drive a real MCP server over the wire.

    ``--check`` proves the checker works on the bytes the server would receive.
    This proves the server actually receives them: it starts the real server,
    which spawns the real BEHAVIOR worker in the other interpreter, and calls
    ``move_to`` with collision checking on.  The distinguishing assertion is
    that ``collision`` comes back with ``world_checked`` true and no
    "joint_names missing" reason -- that string is exactly what appeared when
    the names were dropped at serialisation while every unit test passed.
    """
    import asyncio
    import socket
    import time

    rep: dict = {"stage": "via_mcp", "ok": False}

    def free_port() -> int:
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        p = s.getsockname()[1]
        s.close()
        return p

    port = int(os.environ.get("R1PRO_MCP_PORT") or free_port())
    rep["port"] = port
    log_path = os.path.join(OUTDIR, "r1pro_mcp_server.log")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["MCP_PORT"] = str(port)
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    env.setdefault("OMNIGIBSON_HEADLESS", "1")

    proc = None
    try:
        logf = open(log_path, "w")
        proc = subprocess.Popen(
            [LIBERO_PY, "-m", "sim.mcp_server", "--transport", "sse",
             "--port", str(port)],
            cwd=REPO, env=env, stdout=logf, stderr=subprocess.STDOUT)
        rep["server_log"] = log_path

        # Wait for the port to accept before speaking MCP to it.
        deadline = time.time() + 120
        up = False
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=1):
                    up = True
                    break
            except OSError:
                time.sleep(1)
        rep["server_up"] = up
        if not up:
            rep["error"] = (f"server did not open {port}; see {log_path}")
            raise RuntimeError(rep["error"])

        async def drive() -> dict:
            from mcp import ClientSession
            from mcp.client.streamable_http import streamablehttp_client

            out: dict = {}
            url = f"http://127.0.0.1:{port}/mcp"
            async with streamablehttp_client(url) as (r, w, _):
                async with ClientSession(r, w) as s:
                    await s.initialize()
                    tools = await s.list_tools()
                    out["tools_seen"] = sorted(t.name for t in tools.tools)[:12]

                    res = await s.call_tool("create_env", {
                        "env_id": f"openeta/behavior_{ACTIVITY}-v0",
                        "seed": 0, "image_width": 64, "image_height": 64,
                    })
                    created = _tool_json(res)
                    out["create_env"] = {
                        k: created.get(k) for k in
                        ("handle", "session_id", "backend", "error")}
                    handle = created.get("handle") or ""
                    if not handle:
                        out["error"] = f"create_env gave no handle: {created}"
                        return out
                    # Envs are per-session, and the session id only rides the
                    # context var on the legacy SSE path -- over streamable HTTP
                    # create_env mints one and returns it, so every later call
                    # must pass it back or the handle resolves to "Unknown".
                    sid = created.get("session_id") or ""

                    # A small, deliberately reachable target: the assertion is
                    # about whether collision *ran*, not whether it succeeded.
                    mv = await s.call_tool("move_to", {
                        "handle": handle, "x": 0.4, "y": 0.2, "z": 0.9,
                        "num_steps": 20, "enable_collision_check": True,
                        "session_id": sid,
                    })
                    moved = _tool_json(mv)
                    out["collision"] = moved.get("collision")
                    out["move_error"] = moved.get("error")
                    try:
                        await s.call_tool("close_env",
                                          {"handle": handle, "session_id": sid})
                    except Exception:
                        pass
            return out

        rep.update(asyncio.run(drive()))

        col = rep.get("collision") or {}
        reason = str(col.get("reason") or "")
        rep["collision_reason"] = reason or None
        rep["names_reached_checker"] = "joint_names missing" not in reason
        rep["ok"] = bool(col and rep["names_reached_checker"]
                         and not rep.get("error") and not rep.get("move_error"))
    except BaseException as exc:  # noqa: BLE001
        rep.setdefault("error", f"{type(exc).__name__}: {exc}"[:400])
        rep["traceback"] = traceback.format_exc()[-1500:]
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except Exception:
                proc.kill()

    path = os.path.join(OUTDIR, "r1pro_mcp.json")
    with open(path, "w") as f:
        json.dump(rep, f, indent=2)
    print(f"WROTE {path} ok={rep['ok']} {rep.get('error', '')}", flush=True)
    return 0 if rep["ok"] else 1


def _tool_json(result) -> dict:
    """Pull the JSON payload out of an MCP tool result."""
    data = getattr(result, "structuredContent", None)
    if isinstance(data, dict):
        return data.get("result") if isinstance(data.get("result"), dict) else data
    for item in getattr(result, "content", []) or []:
        text = getattr(item, "text", "")
        if text:
            try:
                return json.loads(text)
            except Exception:
                return {"text": text[:400]}
    return {}


def orchestrate(with_mcp: bool) -> int:
    """Run every stage in the right interpreter, so one command does it all."""
    pre = preflight()
    print(json.dumps(pre, indent=2), flush=True)
    if not pre["ok"]:
        return 1

    print(f"\n=== stage 1: dump (behavior env) — boots Isaac Sim, minutes ===",
          flush=True)
    rc = _run([BEHAVIOR_PY, os.path.abspath(__file__), "--dump"]).returncode
    # Judge stage 1 by its payload, not its exit status: OmniGibson's shutdown
    # hard-exits the interpreter, so a fully successful dump routinely returns
    # nonzero.  The payload is the only trustworthy signal.
    dumped: dict = {}
    try:
        with open(PAYLOAD) as f:
            dumped = json.load(f)
    except Exception:
        pass
    if not dumped.get("ok"):
        print(f"stage 1 failed (rc={rc}); see {PAYLOAD}", file=sys.stderr)
        return rc or 1
    if rc != 0:
        print(f"stage 1 ok despite rc={rc} (OmniGibson shutdown hard-exits)",
              flush=True)

    print("\n=== stage 2: check (libero venv) ===", flush=True)
    rc = _run([LIBERO_PY, os.path.abspath(__file__), "--check"]).returncode
    with open(OUT) as f:
        print(json.dumps(json.load(f), indent=2), flush=True)
    if rc != 0:
        return rc

    if with_mcp:
        print("\n=== stage 3: via MCP server (deployed path) ===", flush=True)
        rc = _run([LIBERO_PY, os.path.abspath(__file__),
                   "--via-mcp-stage"]).returncode
    return rc


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump", action="store_true",
                    help="stage 1 only (must run under the behavior env)")
    ap.add_argument("--check", action="store_true",
                    help="stage 2 only (must run under the libero venv)")
    ap.add_argument("--via-mcp", action="store_true",
                    help="also run stage 3 through a real MCP server")
    ap.add_argument("--via-mcp-stage", action="store_true",
                    help=argparse.SUPPRESS)
    ap.add_argument("--preflight", action="store_true",
                    help="check interpreters and cuRobo config, then stop")
    a = ap.parse_args()

    if a.dump:
        raise SystemExit(dump())
    if a.check:
        raise SystemExit(check())
    if a.via_mcp_stage:
        raise SystemExit(via_mcp())
    if a.preflight:
        rep = preflight()
        print(json.dumps(rep, indent=2))
        raise SystemExit(0 if rep["ok"] else 1)
    raise SystemExit(orchestrate(with_mcp=a.via_mcp))
