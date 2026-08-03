"""Token counting helpers for planner context budget estimates."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from typing import Any

from adapter.protocol import JsonDict


DEFAULT_CONTEXT_WINDOW_TOKENS = 1_000_000


@dataclass(frozen=True, slots=True)
class TokenEstimate:
    """Best-effort token estimate plus estimator metadata."""

    tokens: int
    chars: int
    estimator: JsonDict


def estimate_json_tokens(
    value: JsonDict,
    *,
    model: str | None = None,
    approx_chars_per_token: int = 4,
) -> TokenEstimate:
    """Estimate tokens for a JSON payload.

    If `tiktoken` is installed, use it. Otherwise fall back to a conservative
    character-ratio estimate and report that fallback in metadata.
    """

    text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return estimate_text_tokens(
        text,
        model=model,
        approx_chars_per_token=approx_chars_per_token,
        scope="planner_context_json",
    )


def estimate_text_tokens(
    text: str,
    *,
    model: str | None = None,
    approx_chars_per_token: int = 4,
    scope: str = "text",
) -> TokenEstimate:
    chars = len(text)
    encoding, encoding_name, error = _load_tiktoken_encoding(model)
    if encoding is not None:
        return TokenEstimate(
            tokens=len(encoding.encode(text)),
            chars=chars,
            estimator={
                "method": "tiktoken",
                "model": model or "",
                "encoding": encoding_name,
                "scope": scope,
            },
        )

    chars_per_token = max(1, approx_chars_per_token)
    return TokenEstimate(
        tokens=(chars + chars_per_token - 1) // chars_per_token,
        chars=chars,
        estimator={
            "method": "json_chars_div_approx_chars_per_token",
            "approx_chars_per_token": chars_per_token,
            "scope": scope,
            "fallback_reason": error or "tiktoken_unavailable",
        },
    )


def _load_tiktoken_encoding(model: str | None) -> tuple[Any | None, str, str]:
    try:
        tiktoken = importlib.import_module("tiktoken")
    except ModuleNotFoundError:
        return None, "", "tiktoken_not_installed"

    if model:
        try:
            return tiktoken.encoding_for_model(model), "encoding_for_model", ""
        except Exception as exc:  # noqa: BLE001 - unknown models should fall back gracefully.
            model_error = f"{type(exc).__name__}: {exc}"
    else:
        model_error = "model_not_set"

    try:
        return tiktoken.get_encoding("cl100k_base"), "cl100k_base", ""
    except Exception as exc:  # noqa: BLE001 - optional tokenizer path.
        return None, "", model_error or f"{type(exc).__name__}: {exc}"
