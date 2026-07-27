"""Planner provider configuration helpers for CLI and future GUI setup."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from adapter.protocol import JsonDict
from agent.runtime.token_counting import DEFAULT_CONTEXT_WINDOW_TOKENS


DEFAULT_ENV_PATH = ".env"
DEFAULT_APIKEY_PATH = "apikey.md"


@dataclass(slots=True)
class ProviderEndpointConfig:
    """One OpenAI-compatible provider endpoint used by the planner."""

    provider: str = "openai-compatible"
    model: str = ""
    api_base: str = ""
    api_key: str = ""
    timeout_s: float = 60.0

    def missing_fields(self) -> list[str]:
        missing: list[str] = []
        if not self.provider:
            missing.append("provider")
        if not self.model:
            missing.append("model")
        if not self.api_base:
            missing.append("api_base")
        if not self.api_key:
            missing.append("api_key")
        return missing

    def redacted(self) -> JsonDict:
        return {
            "provider": self.provider,
            "model": self.model,
            "api_base": self.api_base,
            "api_key": _redact_secret(self.api_key),
            "timeout_s": self.timeout_s,
        }


@dataclass(slots=True)
class PlannerProviderConfig:
    """User-editable model provider configuration.

    This object is intentionally UI-friendly: a future GUI can render the
    fields, validate missing values, and write the same values to `.env`.
    """

    provider: str = "openai-compatible"
    model: str = ""
    api_base: str = ""
    api_key: str = ""
    timeout_s: float = 60.0
    max_attempts: int = 3
    retry_backoff_s: float = 0.5
    context_window_tokens: int | None = DEFAULT_CONTEXT_WINDOW_TOKENS
    fallback: ProviderEndpointConfig | None = None
    metadata: JsonDict = field(default_factory=dict)

    def missing_fields(self) -> list[str]:
        missing: list[str] = []
        if not self.provider:
            missing.append("provider")
        if not self.model:
            missing.append("model")
        if not self.api_base:
            missing.append("api_base")
        if not self.api_key:
            missing.append("api_key")
        if self.fallback is not None:
            missing.extend(f"fallback.{field}" for field in self.fallback.missing_fields())
        return missing

    def redacted(self) -> JsonDict:
        return {
            "provider": self.provider,
            "model": self.model,
            "api_base": self.api_base,
            "api_key": _redact_secret(self.api_key),
            "timeout_s": self.timeout_s,
            "max_attempts": self.max_attempts,
            "retry_backoff_s": self.retry_backoff_s,
            "context_window_tokens": self.context_window_tokens,
            "fallback": self.fallback.redacted() if self.fallback is not None else None,
            "metadata": dict(self.metadata),
        }

    def to_env_lines(self) -> list[str]:
        lines = [
            f"OPENETA_LLM_PROVIDER={self.provider}",
            f"OPENETA_LLM_MODEL={self.model}",
            f"OPENETA_LLM_API_BASE={self.api_base}",
            f"OPENETA_LLM_API_KEY={self.api_key}",
            f"OPENETA_LLM_TIMEOUT_S={self.timeout_s}",
            f"OPENETA_LLM_MAX_ATTEMPTS={self.max_attempts}",
            f"OPENETA_LLM_RETRY_BACKOFF_S={self.retry_backoff_s}",
        ]
        if self.context_window_tokens is not None:
            lines.append(f"OPENETA_LLM_CONTEXT_WINDOW_TOKENS={self.context_window_tokens}")
        if self.fallback is not None:
            lines.extend(
                [
                    f"OPENETA_LLM_FALLBACK_PROVIDER={self.fallback.provider}",
                    f"OPENETA_LLM_FALLBACK_MODEL={self.fallback.model}",
                    f"OPENETA_LLM_FALLBACK_API_BASE={self.fallback.api_base}",
                    f"OPENETA_LLM_FALLBACK_API_KEY={self.fallback.api_key}",
                    f"OPENETA_LLM_FALLBACK_TIMEOUT_S={self.fallback.timeout_s}",
                ]
            )
        return lines


def load_planner_provider_config(
    *,
    env: Mapping[str, str] | None = None,
    dotenv_path: str | Path = DEFAULT_ENV_PATH,
    apikey_path: str | Path = DEFAULT_APIKEY_PATH,
) -> PlannerProviderConfig:
    """Load provider config from environment, `.env`, or local `apikey.md`.

    Precedence is environment > `.env` > `apikey.md` > defaults. `apikey.md`
    supports the local newapi channel shape the user provided.
    """

    dotenv = read_env_file(dotenv_path)
    apikey_config = read_apikey_file(apikey_path)
    source_env = dict(env if env is not None else os.environ)

    provider = _first_present(
        source_env.get("OPENETA_LLM_PROVIDER"),
        dotenv.get("OPENETA_LLM_PROVIDER"),
        apikey_config.provider,
        "openai-compatible",
    )
    model = _first_present(
        source_env.get("OPENETA_LLM_MODEL"),
        dotenv.get("OPENETA_LLM_MODEL"),
        apikey_config.model,
    )
    api_base = _first_present(
        source_env.get("OPENETA_LLM_API_BASE"),
        dotenv.get("OPENETA_LLM_API_BASE"),
        apikey_config.api_base,
    )
    api_key = _first_present(
        source_env.get("OPENETA_LLM_API_KEY"),
        dotenv.get("OPENETA_LLM_API_KEY"),
        apikey_config.api_key,
    )
    timeout_raw = _first_present(
        source_env.get("OPENETA_LLM_TIMEOUT_S"),
        dotenv.get("OPENETA_LLM_TIMEOUT_S"),
        str(apikey_config.timeout_s) if apikey_config.timeout_s else "",
        "60",
    )
    try:
        timeout_s = float(timeout_raw)
    except ValueError:
        timeout_s = 60.0
    max_attempts = _first_positive_int(
        source_env.get("OPENETA_LLM_MAX_ATTEMPTS"),
        dotenv.get("OPENETA_LLM_MAX_ATTEMPTS"),
        "3",
    )
    retry_backoff_s = _first_non_negative_float(
        source_env.get("OPENETA_LLM_RETRY_BACKOFF_S"),
        dotenv.get("OPENETA_LLM_RETRY_BACKOFF_S"),
        "0.5",
    )
    context_window_tokens = _first_positive_int(
        source_env.get("OPENETA_LLM_CONTEXT_WINDOW_TOKENS"),
        dotenv.get("OPENETA_LLM_CONTEXT_WINDOW_TOKENS"),
        str(DEFAULT_CONTEXT_WINDOW_TOKENS),
    )
    fallback_provider = _first_present(
        source_env.get("OPENETA_LLM_FALLBACK_PROVIDER"),
        dotenv.get("OPENETA_LLM_FALLBACK_PROVIDER"),
    )
    fallback_model = _first_present(
        source_env.get("OPENETA_LLM_FALLBACK_MODEL"),
        dotenv.get("OPENETA_LLM_FALLBACK_MODEL"),
    )
    fallback_api_base = _first_present(
        source_env.get("OPENETA_LLM_FALLBACK_API_BASE"),
        dotenv.get("OPENETA_LLM_FALLBACK_API_BASE"),
    )
    fallback_api_key = _first_present(
        source_env.get("OPENETA_LLM_FALLBACK_API_KEY"),
        dotenv.get("OPENETA_LLM_FALLBACK_API_KEY"),
    )
    fallback_timeout_raw = _first_present(
        source_env.get("OPENETA_LLM_FALLBACK_TIMEOUT_S"),
        dotenv.get("OPENETA_LLM_FALLBACK_TIMEOUT_S"),
        str(timeout_s),
    )
    try:
        fallback_timeout_s = float(fallback_timeout_raw)
    except ValueError:
        fallback_timeout_s = timeout_s
    fallback = None
    if any((fallback_provider, fallback_model, fallback_api_base, fallback_api_key)):
        fallback = ProviderEndpointConfig(
            provider=fallback_provider,
            model=fallback_model,
            api_base=fallback_api_base.rstrip("/"),
            api_key=fallback_api_key,
            timeout_s=fallback_timeout_s,
        )

    return PlannerProviderConfig(
        provider=provider,
        model=model,
        api_base=api_base.rstrip("/"),
        api_key=api_key,
        timeout_s=timeout_s,
        max_attempts=max_attempts or 3,
        retry_backoff_s=retry_backoff_s,
        context_window_tokens=context_window_tokens,
        fallback=fallback,
        metadata={"sources": {"dotenv_path": str(dotenv_path), "apikey_path": str(apikey_path)}},
    )


def read_env_file(path: str | Path) -> dict[str, str]:
    env_path = Path(path)
    if not env_path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = _strip_env_quotes(value.strip())
    return values


def read_apikey_file(path: str | Path) -> PlannerProviderConfig:
    apikey_path = Path(path)
    if not apikey_path.exists():
        return PlannerProviderConfig()

    text = apikey_path.read_text(encoding="utf-8")
    first_key = ""
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("{"):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if payload.get("_type") == "newapi_channel_conn":
                return PlannerProviderConfig(
                    provider="openai-compatible",
                    api_base=str(payload.get("url", "")).rstrip("/"),
                    api_key=str(payload.get("key", "")),
                    metadata={"source": "newapi_channel_conn"},
                )
        if not first_key:
            first_key = line

    return PlannerProviderConfig(api_key=first_key, metadata={"source": "raw_key"})


def write_env_file(config: PlannerProviderConfig, path: str | Path = DEFAULT_ENV_PATH) -> None:
    env_path = Path(path)
    env_path.write_text("\n".join(config.to_env_lines()) + "\n", encoding="utf-8")


def _first_present(*values: str | None) -> str:
    for value in values:
        if value is not None and value != "":
            return value
    return ""


def _strip_env_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _first_positive_int(*values: str | None) -> int | None:
    for value in values:
        if value is None or value == "":
            continue
        try:
            parsed = int(value)
        except ValueError:
            continue
        if parsed > 0:
            return parsed
    return None


def _first_non_negative_float(*values: str | None) -> float:
    for value in values:
        if value is None or value == "":
            continue
        try:
            parsed = float(value)
        except ValueError:
            continue
        if parsed >= 0:
            return parsed
    return 0.5


def _redact_secret(value: str) -> str:
    if not value:
        return ""
    if len(value) <= 8:
        return "***"
    return f"{value[:3]}...{value[-4:]}"
