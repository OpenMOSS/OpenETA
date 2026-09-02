from __future__ import annotations

from pathlib import Path

from sim.envs.univtac.elf_loader_closure import (
    legacy_paths,
    loaded_objects,
    parse_proc_maps,
    resolve_needed,
    unexpected_static_missing,
)


def test_maps_and_needed_resolution() -> None:
    records = parse_proc_maps("7f00-7f10 r-xp 00000000 08:01 9 /env/torch/lib/libc10.so\n")
    objects = loaded_objects(records)
    assert objects == ["/env/torch/lib/libc10.so"]
    assert resolve_needed(["libc10.so"], objects)[0]["status"] == "resolved"
    versioned = ["/env/lib/libstdc++.so.6.0.33"]
    assert resolve_needed(["libstdc++.so.6"], versioned, {versioned[0]: "libstdc++.so.6"})[0]["status"] == "resolved"


def test_raw_ldd_only_allows_known_torch_sonames() -> None:
    assert unexpected_static_missing(["libtorch.so", "libc10_cuda.so"]) == []
    assert unexpected_static_missing(["libmystery.so"]) == ["libmystery.so"]


def test_legacy_paths_match_explicit_roots() -> None:
    assert legacy_paths(["/legacy/lib.so", "/target/lib.so"], [Path("/legacy")]) == ["/legacy/lib.so"]
