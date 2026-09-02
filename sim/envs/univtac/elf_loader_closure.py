"""Pure helpers for relating ELF dependencies to a live process loader map."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ALLOWED_TORCH_SONAMES = {
    "libc10.so",
    "libc10_cuda.so",
    "libtorch.so",
    "libtorch_cpu.so",
    "libtorch_cuda.so",
    "libtorch_python.so",
}


def parse_proc_maps(text: str) -> list[dict[str, Any]]:
    records = []
    for line in text.splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) < 6 or not parts[5].startswith("/"):
            continue
        start, end = (int(item, 16) for item in parts[0].split("-", 1))
        path = Path(parts[5])
        records.append(
            {
                "start": start,
                "end": end,
                "perms": parts[1],
                "offset": int(parts[2], 16),
                "device": parts[3],
                "inode": int(parts[4]),
                "path": parts[5],
                "realpath": str(path.resolve()),
            }
        )
    return records


def loaded_objects(records: Iterable[dict[str, Any]]) -> list[str]:
    return sorted({record["realpath"] for record in records if ".so" in Path(record["realpath"]).name})


def resolve_needed(
    needed: Iterable[str],
    objects: Iterable[str],
    sonames: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    by_name: dict[str, list[str]] = defaultdict(list)
    for path in objects:
        by_name[Path(path).name].append(path)
        if sonames and sonames.get(path):
            by_name[sonames[path]].append(path)
    return [
        {
            "soname": name,
            "realpaths": sorted(set(by_name.get(name, []))),
            "status": "missing" if not by_name.get(name) else ("resolved" if len(set(by_name[name])) == 1 else "ambiguous"),
        }
        for name in needed
    ]


def unexpected_static_missing(names: Iterable[str]) -> list[str]:
    return sorted(set(names) - ALLOWED_TORCH_SONAMES)


def legacy_paths(objects: Iterable[str], forbidden_roots: Iterable[Path]) -> list[str]:
    roots = [str(path.resolve()) for path in forbidden_roots]
    return sorted(path for path in objects if any(path == root or path.startswith(root + "/") for root in roots))
