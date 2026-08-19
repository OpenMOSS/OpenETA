"""Measure how many BEHAVIOR sims fit on this machine, and what they cost in rate.

The earlier 2-instance number came from an empty ``Scene``.  Real activities
load full interactive houses from the 37 GB asset set, so that figure cannot be
extrapolated -- this drives real activities instead.

Each instance is its own subprocess: ``gm.HEADLESS`` is write-once per process
and ``env.close()`` hard-exits the interpreter, so in-process fan-out is not an
option.  Every child writes its own JSON before shutting down, because anything
after the shutdown call never runs.

Usage:
    bench_behavior_concurrency.py --worker N OUT   # internal, one instance
    bench_behavior_concurrency.py --levels 1,2,4   # driver
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = "/home/yfzhang/nvme1/openeta_fix_wt"
BEHAVIOR_PY = "/home/yfzhang/anaconda3/envs/behavior/bin/python"
ACTIVITY = os.environ.get("BENCH_ACTIVITY", "picking_up_trash")
WARMUP_STEPS = 5
MEASURE_STEPS = 30
# BehaviorDirectEnv maps seed -> activity_instance_id, so a per-worker seed picks
# a different scene instance and most activities ship only instance 0 (seed=1
# died with "expected str, bytes or os.PathLike object, not NoneType" from
# get_task_instance_path).  Every worker therefore loads instance 0 -- which is
# also what makes the throughput numbers comparable: identical work per worker.
INSTANCE_SEED = 0


# ── child: one simulator instance ──────────────────────────────────

def run_worker(out_path: str, worker_id: int, gpu: str | None) -> int:
    os.environ["OMNI_KIT_ACCEPT_EULA"] = "YES"
    os.environ["OMNIGIBSON_HEADLESS"] = "1"
    if gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    sys.path.insert(0, REPO)

    rep: dict = {"worker": worker_id, "gpu": gpu, "activity": ACTIVITY, "ok": False}
    t_start = time.time()
    try:
        import numpy as np
        from sim.envs.behavior.direct_env import BehaviorDirectEnv

        env = BehaviorDirectEnv(ACTIVITY, seed=INSTANCE_SEED, render_mode="rgb_array",
                                image_width=128, image_height=128)
        rep["boot_s"] = round(time.time() - t_start, 1)

        spec = env.openeta_control_spec
        dim = int(spec.get("action_dim") or 0)
        rep["action_dim"] = dim

        env.reset()
        rep["reset_s"] = round(time.time() - t_start, 1)

        zero = np.zeros(dim, dtype=np.float32)
        for _ in range(WARMUP_STEPS):
            env.step(zero)

        # Per-step timings, not just an average: a mean hides the stalls that
        # make a sim unusable interactively even when throughput looks fine.
        times = []
        for _ in range(MEASURE_STEPS):
            t0 = time.perf_counter()
            env.step(zero)
            times.append(time.perf_counter() - t0)

        times_ms = sorted(t * 1000.0 for t in times)
        n = len(times_ms)
        rep["step_ms_median"] = round(times_ms[n // 2], 1)
        rep["step_ms_p95"] = round(times_ms[min(n - 1, int(n * 0.95))], 1)
        rep["step_ms_max"] = round(times_ms[-1], 1)
        rep["hz_median"] = round(1000.0 / times_ms[n // 2], 1)
        rep["total_s"] = round(time.time() - t_start, 1)
        rep["ok"] = True
    except BaseException as exc:  # noqa: BLE001 - report, never mask
        import traceback
        rep["error"] = f"{type(exc).__name__}: {exc}"[:300]
        rep["traceback"] = traceback.format_exc()[-1200:]

    # Write before any shutdown: env.close() / og.shutdown() terminate the
    # interpreter, so a later write is silently lost.
    Path(out_path).write_text(json.dumps(rep, indent=2))
    print(f"[w{worker_id}] ok={rep['ok']} {rep.get('error','')}", flush=True)
    try:
        env.close()  # type: ignore[possibly-undefined]
    except BaseException:
        pass
    return 0 if rep["ok"] else 1


# ── parent: fan out and sample VRAM ────────────────────────────────

def vram_mb() -> list[int]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=20,
        ).stdout
        return [int(x.strip()) for x in out.splitlines() if x.strip()]
    except Exception:
        return []


def run_level(n: int, outdir: Path, spread: bool, stagger: float,
              timeout_s: float) -> dict:
    baseline = vram_mb()
    procs, outs = [], []
    for i in range(n):
        out = outdir / f"n{n}_w{i}.json"
        out.unlink(missing_ok=True)
        outs.append(out)
        cmd = [BEHAVIOR_PY, __file__, "--worker", str(i), str(out)]
        if spread:
            cmd += ["--gpu", str(i % 2)]
        procs.append(subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                      stderr=subprocess.DEVNULL))
        if i < n - 1:
            time.sleep(stagger)

    peak = list(baseline)
    t0 = time.time()
    while any(p.poll() is None for p in procs):
        if time.time() - t0 > timeout_s:
            for p in procs:
                if p.poll() is None:
                    p.kill()
            break
        cur = vram_mb()
        peak = [max(a, b) for a, b in zip(peak, cur)] if len(cur) == len(peak) else peak
        time.sleep(5)

    reports = []
    for out in outs:
        if out.exists():
            try:
                reports.append(json.loads(out.read_text()))
            except Exception:
                pass

    good = [r for r in reports if r.get("ok")]
    hz = sorted(r["hz_median"] for r in good if "hz_median" in r)
    return {
        "n_requested": n,
        "n_survived": len(good),
        "spread_gpus": spread,
        "vram_baseline_mb": baseline,
        "vram_peak_mb": peak,
        "vram_delta_mb": [p - b for p, b in zip(peak, baseline)] if baseline else [],
        "hz_median_per_worker": hz,
        "hz_slowest": hz[0] if hz else None,
        "hz_fastest": hz[-1] if hz else None,
        "boot_s": sorted(r.get("boot_s", 0) for r in good),
        "step_ms_p95": sorted(r.get("step_ms_p95", 0) for r in good),
        "failures": [r.get("error") for r in reports if not r.get("ok")],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int)
    ap.add_argument("--gpu")
    ap.add_argument("out", nargs="?")
    ap.add_argument("--levels", default="1,2,4")
    ap.add_argument("--spread", action="store_true",
                    help="pin workers alternately to GPU0/GPU1")
    ap.add_argument("--stagger", type=float, default=8.0)
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--outdir", default="/tmp/behavior_bench")
    args = ap.parse_args()

    if args.worker is not None:
        if not args.out:
            print("--worker needs an output path", file=sys.stderr)
            return 2
        return run_worker(args.out, args.worker, args.gpu)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    levels = [int(x) for x in args.levels.split(",") if x.strip()]

    results = []
    for n in levels:
        print(f"\n{'=' * 60}\nlevel n={n} spread={args.spread}\n{'=' * 60}", flush=True)
        res = run_level(n, outdir, args.spread, args.stagger, args.timeout)
        results.append(res)
        print(json.dumps(res, indent=2), flush=True)
        summary = outdir / "summary.json"
        summary.write_text(json.dumps(results, indent=2))
        if res["n_survived"] < n:
            print(f"level {n} lost {n - res['n_survived']} instance(s); stopping",
                  flush=True)
            break
        time.sleep(10)  # let VRAM settle before the next level

    print(f"\nsummary -> {outdir / 'summary.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
