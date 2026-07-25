"""Unit tests for the SSE background-render pacing + dirty-tracking logic.

These guard the "A" optimisation in worker_mgr:

  * dirty tracking — an idle env (no step/reset since last render) must NOT be
    re-rendered, but a stepped env must;
  * in-flight dedup + adaptive gap — a worker mid-render or within its cooldown
    must not be re-reserved.

The functions under test are pure state machines over module-level dicts, so
we exercise them directly without a running server or worker.
"""
from __future__ import annotations

import time

import sim.mcp_server.worker_mgr as wm


def _fresh_key(name: str) -> tuple[str, str]:
    """Return an unused obs-key and clear any leftover state for it."""
    key = (f"http://test-{name}", "h0")
    wm._forget_obs_dirty(key)
    return key


# ── dirty tracking ────────────────────────────────────────────────────

def test_new_key_is_dirty():
    """A never-rendered key is dirty so the first frame always renders."""
    key = _fresh_key("new")
    assert wm._obs_is_dirty(key) is True


def test_render_clears_dirty():
    key = _fresh_key("clear")
    gen = wm._mark_obs_dirty(key)      # physics advanced
    assert wm._obs_is_dirty(key) is True
    wm._mark_obs_rendered(key, gen)    # we rendered that generation
    assert wm._obs_is_dirty(key) is False


def test_step_after_render_redirties():
    key = _fresh_key("redirty")
    gen = wm._mark_obs_dirty(key)
    wm._mark_obs_rendered(key, gen)
    assert wm._obs_is_dirty(key) is False
    wm._mark_obs_dirty(key)            # another step
    assert wm._obs_is_dirty(key) is True


def test_step_mid_render_is_not_lost():
    """A step landing between snapshot and record must keep the key dirty."""
    key = _fresh_key("midrender")
    # Snapshot the gen the render will capture (mimics guarded wrapper).
    with wm._obs_dirty_lock:
        pre = wm._obs_dirty_gen.get(key, 1)
    # A step lands *during* the render.
    wm._mark_obs_dirty(key)
    # Render completes and records the (now stale) snapshot gen.
    wm._mark_obs_rendered(key, pre)
    # Still dirty → will re-render next tick.
    assert wm._obs_is_dirty(key) is True


def test_worker_any_dirty():
    wurl = "http://test-anydirty"
    k0, k1 = (wurl, "a"), (wurl, "b")
    for k in (k0, k1):
        wm._forget_obs_dirty(k)
    # both new → dirty
    assert wm._worker_any_dirty(wurl, ["a", "b"]) is True
    # clean both
    for k in (k0, k1):
        g = wm._mark_obs_dirty(k)
        wm._mark_obs_rendered(k, g)
    assert wm._worker_any_dirty(wurl, ["a", "b"]) is False
    # dirty just one
    wm._mark_obs_dirty(k1)
    assert wm._worker_any_dirty(wurl, ["a", "b"]) is True


def test_forget_obs_dirty_resets_state():
    key = _fresh_key("forget")
    g = wm._mark_obs_dirty(key)
    wm._mark_obs_rendered(key, g)
    assert wm._obs_is_dirty(key) is False
    wm._forget_obs_dirty(key)
    # back to "new" → dirty again
    assert wm._obs_is_dirty(key) is True


# ── in-flight dedup + adaptive pacing ─────────────────────────────────

def test_reserve_dedup_and_gap():
    wurl = "http://test-reserve"
    # clear any prior pacing state
    with wm._worker_render_lock:
        wm._worker_render_inflight.pop(wurl, None)
        wm._worker_render_next_ok.pop(wurl, None)

    assert wm._try_reserve_worker_render(wurl) is True, "first reserve wins"
    assert wm._try_reserve_worker_render(wurl) is False, "blocked while in-flight"

    # simulate a 1.2 s render completing
    with wm._worker_render_lock:
        wm._worker_render_inflight[wurl] = False
        wm._worker_render_next_ok[wurl] = time.monotonic() + max(wm._SSE_MIN_REFRESH_GAP, 1.2)
    assert wm._try_reserve_worker_render(wurl) is False, "blocked by adaptive gap"

    # gap elapsed
    with wm._worker_render_lock:
        wm._worker_render_next_ok[wurl] = time.monotonic() - 0.01
    assert wm._try_reserve_worker_render(wurl) is True, "allowed after gap"


def test_guarded_refresh_records_gen_and_paces(monkeypatch):
    """The guarded wrapper clears in-flight, sets a cooldown, and marks rendered."""
    wurl = "http://test-guarded"
    key = (wurl, "h0")
    wm._forget_obs_dirty(key)
    with wm._worker_render_lock:
        wm._worker_render_inflight[wurl] = True  # caller reserved
        wm._worker_render_next_ok.pop(wurl, None)

    gen = wm._mark_obs_dirty(key)  # pending frame to capture
    assert wm._obs_is_dirty(key) is True

    # stub the actual blocking HTTP render
    called = {}
    def fake_refresh(w, hs, sid):
        called["args"] = (w, hs, sid)
    monkeypatch.setattr(wm, "_refresh_cache_for_worker", fake_refresh)

    wm._refresh_cache_for_worker_guarded(wurl, ["h0"], "sid0")

    assert called["args"] == (wurl, ["h0"], "sid0")
    # rendered the captured generation → now clean
    assert wm._obs_is_dirty(key) is False
    # in-flight cleared, cooldown set in the future
    with wm._worker_render_lock:
        assert wm._worker_render_inflight[wurl] is False
        assert wm._worker_render_next_ok[wurl] >= time.monotonic() - 0.001
    _ = gen
