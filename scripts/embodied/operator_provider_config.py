#!/usr/bin/env python3
"""Render one model provider into an otherwise isolated Operator config."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


_BARE_KEY = re.compile(r"^[A-Za-z0-9_-]+$")


def _toml_key(value: str) -> str:
    return value if _BARE_KEY.fullmatch(value) else json.dumps(value)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return (
            "{ "
            + ", ".join(
                f"{_toml_key(str(key))} = {_toml_value(item)}"
                for key, item in value.items()
            )
            + " }"
        )
    raise TypeError(f"unsupported provider config value: {type(value).__name__}")


def render_provider_config(source: Path, provider_name: str) -> str:
    """Return only the selected provider and the top-level selection."""

    if not _BARE_KEY.fullmatch(provider_name):
        raise ValueError(f"invalid provider name: {provider_name!r}")
    if provider_name == "openai":
        return 'model_provider = "openai"\n'
    with source.open("rb") as stream:
        config = tomllib.load(stream)
    providers = config.get("model_providers")
    provider = providers.get(provider_name) if isinstance(providers, dict) else None
    if not isinstance(provider, dict):
        raise KeyError(
            f"model provider {provider_name!r} is absent from {source}"
        )
    lines = [
        f"model_provider = {_toml_value(provider_name)}",
        "",
        f"[model_providers.{provider_name}]",
    ]
    lines.extend(
        f"{_toml_key(str(key))} = {_toml_value(value)}"
        for key, value in provider.items()
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--provider", required=True)
    args = parser.parse_args()
    print(
        render_provider_config(args.source.expanduser(), args.provider),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
