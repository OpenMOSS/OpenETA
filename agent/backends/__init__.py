"""Lazy backend exports that keep specialized submodule imports isolated."""

from __future__ import annotations

import importlib
from typing import Any

_EXPORTS = {
    "CallablePlannerBackend": "agent.backends.planner",
    "CodePolicyBackend": "agent.backends.code_policy",
    "CodePolicyGenerationRequest": "agent.backends.code_policy",
    "CodePolicyGenerationResult": "agent.backends.code_policy",
    "CommercialApiBackendConfig": "agent.backends.code_policy",
    "CommercialApiCodePolicyBackend": "agent.backends.code_policy",
    "CommercialApiPlannerBackend": "agent.backends.planner",
    "CommercialApiPlannerBackendConfig": "agent.backends.planner",
    "OpenAICompatibleModelInfo": "agent.backends.planner",
    "OpenAICompatiblePlannerBackend": "agent.backends.planner",
    "OpenAICompatiblePlannerBackendConfig": "agent.backends.planner",
    "PlaceholderCodePolicyBackend": "agent.backends.code_policy",
    "PlaceholderPlannerBackend": "agent.backends.planner",
    "PlannerBackend": "agent.backends.planner",
    "PlannerBackendCallable": "agent.backends.planner",
    "PlannerBackendRequest": "agent.backends.planner",
    "PlannerBackendResult": "agent.backends.planner",
    "PlannerProviderConfig": "agent.backends.provider_config",
    "StaticPlannerBackend": "agent.backends.planner",
    "extract_context_window_tokens": "agent.backends.planner",
    "list_openai_compatible_model_info": "agent.backends.planner",
    "list_openai_compatible_models": "agent.backends.planner",
    "load_planner_provider_config": "agent.backends.provider_config",
    "read_apikey_file": "agent.backends.provider_config",
    "read_env_file": "agent.backends.provider_config",
    "resolve_openai_compatible_context_window_tokens": "agent.backends.planner",
    "write_env_file": "agent.backends.provider_config",
}

__all__ = [
    "CallablePlannerBackend",
    "CodePolicyBackend",
    "CodePolicyGenerationRequest",
    "CodePolicyGenerationResult",
    "CommercialApiBackendConfig",
    "CommercialApiCodePolicyBackend",
    "CommercialApiPlannerBackend",
    "CommercialApiPlannerBackendConfig",
    "OpenAICompatibleModelInfo",
    "OpenAICompatiblePlannerBackend",
    "OpenAICompatiblePlannerBackendConfig",
    "PlaceholderCodePolicyBackend",
    "PlaceholderPlannerBackend",
    "PlannerBackend",
    "PlannerBackendCallable",
    "PlannerBackendRequest",
    "PlannerBackendResult",
    "PlannerProviderConfig",
    "StaticPlannerBackend",
    "extract_context_window_tokens",
    "list_openai_compatible_model_info",
    "list_openai_compatible_models",
    "load_planner_provider_config",
    "read_apikey_file",
    "read_env_file",
    "resolve_openai_compatible_context_window_tokens",
    "write_env_file",
]


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
