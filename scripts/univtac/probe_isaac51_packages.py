#!/usr/bin/env python3
"""Record installed package versions and import origins without starting Isaac Sim."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path


DISTRIBUTIONS = (
    "torch", "torchvision", "warp-lang", "pyuipc", "nvidia-curobo", "setuptools",
    "setuptools-scm", "wheel", "packaging", "filelock", "isaacsim", "isaaclab",
    "tacex", "tacex-assets",
)
MODULES = ("torch", "torchvision", "warp", "uipc", "curobo", "isaacsim", "isaaclab", "tacex", "tacex_assets")


def version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def module_origin(name: str) -> str | None:
    spec = importlib.util.find_spec(name)
    if spec is None or spec.origin is None:
        return None
    return str(Path(spec.origin).resolve())


def editable_source(python: Path) -> str | None:
    completed = subprocess.run(
        [str(python), "-m", "pip", "show", "nvidia-curobo"],
        check=False, capture_output=True, text=True,
    )
    prefix = "Editable project location:"
    return next((line.split(":", 1)[1].strip() for line in completed.stdout.splitlines() if line.startswith(prefix)), None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    python = Path(sys.executable).resolve()
    payload = {
        "schema_version": "openeta.univtac.isaac51_package_probe.v1",
        "python_executable": str(python),
        "python_version": platform.python_version(),
        "distributions": {name: version(name) for name in DISTRIBUTIONS},
        "module_origins": {name: module_origin(name) for name in MODULES},
        "nvidia_curobo_editable_source": editable_source(python),
        "openeta_imported": any(name == "openeta" or name.startswith("openeta.") for name in sys.modules),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
