"""Interactive terminal console for the OpenETA closed-loop agent."""

from __future__ import annotations

import argparse
import html
import json
import os
import shlex
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.filters import has_completions
from prompt_toolkit.formatted_text import HTML
from prompt_toolkit.history import InMemoryHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.shortcuts import CompleteStyle
from prompt_toolkit.styles import Style
from prompt_toolkit.utils import get_cwidth

from adapter.protocol import EnvAction, JsonDict
from agent.backends.planner import (
    OpenAICompatiblePlannerBackend,
    OpenAICompatiblePlannerBackendConfig,
    list_openai_compatible_model_info,
)
from agent.backends.provider_config import (
    PlannerProviderConfig,
    ProviderEndpointConfig,
    load_planner_provider_config,
    write_env_file,
)
from agent.runtime.checkers import CheckerSubagentConfig
from agent.runtime.episode import (
    DEFAULT_MAX_TURNS,
    EpisodeResult,
    EpisodeStep,
    OpenEtaEpisodeRunner,
    ToolFeedbackEpisodeEnvironment,
)
from agent.runtime.memory import AgentMemory
from agent.runtime.memory_store import JsonMemoryStore
from agent.runtime.mcp_catalog import discover_mcp_tool_catalog
from agent.runtime.promoted_memory import PromotedMemoryStore
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.runtime.runtime_assembly import (
    RuntimeAssemblyConfig,
    RuntimeMcpEndpoints,
    assemble_runtime,
    configure_runtime_self_improvement,
    resolve_runtime_mcp_endpoints,
)
from agent.runtime.self_improvement import SkillReviewProposalStore
from agent.runtime.session_workspace import (
    LEGACY_SESSION_WORKSPACE_ROOT,
    SessionWorkspace,
)
from agent.runtime.supervision import (
    BackendGuidanceResolver,
    SupervisionGate,
    SupervisionProfile,
    SupervisionPolicy,
)
from agent.tools.mcp_registry import (
    compact_mcp_registry,
    load_mcp_server_configs,
    load_mcp_server_url,
)
from agent.tools.registry import (
    ToolExecutionContext,
    ToolRegistry,
)
from agent.tools.sim_mcp import (
    SimulatorMcpToolProxyConfig,
    SseSimulatorMcpTransport,
    close_environment_mcp_env,
    mcp_dashboard_url,
    mcp_server_url_from_endpoint,
)


TOOL_RESULT_MAX_LINES = 5
TOOL_RESULT_FALLBACK_WIDTH = 120
DEFAULT_SIM_MCP_TIMEOUT_S = 300.0


class Theme:
    """Small ANSI theme for the terminal console."""

    enabled = sys.stdout.isatty() and not os.environ.get("NO_COLOR")

    @classmethod
    def color(cls, text: str, code: str) -> str:
        if not cls.enabled:
            return text
        return f"\033[{code}m{text}\033[0m"

    @classmethod
    def dim(cls, text: str) -> str:
        return cls.color(text, "2")

    @classmethod
    def accent(cls, text: str) -> str:
        return cls.color(text, "36")

    @classmethod
    def ok(cls, text: str) -> str:
        return cls.color(text, "32")

    @classmethod
    def warn(cls, text: str) -> str:
        return cls.color(text, "33")

    @classmethod
    def err(cls, text: str) -> str:
        return cls.color(text, "31")

    @classmethod
    def bold(cls, text: str) -> str:
        return cls.color(text, "1")


@dataclass(frozen=True, slots=True)
class SlashCommandSpec:
    """User-facing slash command metadata.

    This mirrors Codex's command registry pattern: presentation order lives in
    data, descriptions are shown in the popup, and aliases are kept out of the
    default visible list.
    """

    name: str
    description: str
    usage: str
    aliases: tuple[str, ...] = ()
    supports_inline_args: bool = False


SLASH_COMMANDS: tuple[SlashCommandSpec, ...] = (
    SlashCommandSpec("provider", "configure provider, base URL, key, and model", "/provider"),
    SlashCommandSpec(
        "model",
        "select or set the active model",
        "/model [name]",
        supports_inline_args=True,
    ),
    SlashCommandSpec("models", "list models returned by the provider", "/models"),
    SlashCommandSpec("tools", "show registered tool handlers", "/tools"),
    SlashCommandSpec(
        "approvement",
        "select the active supervision profile",
        "/approvement [human_gated|standard|reviewed_autonomy]",
        aliases=("approval",),
        supports_inline_args=True,
    ),
    SlashCommandSpec(
        "memory",
        "show local working memory and session paths",
        "/memory [all|facts|artifacts|skill_notes|compact] [--json]",
        supports_inline_args=True,
    ),
    SlashCommandSpec(
        "promote-memory",
        "promote reviewed working memory into agent/memory",
        "/promote-memory <namespace> <key> [--target FILE] [--note TEXT]",
        supports_inline_args=True,
    ),
    SlashCommandSpec(
        "new",
        "start a fresh CLI task session",
        "/new [--clear-working-memory]",
        supports_inline_args=True,
    ),
    SlashCommandSpec("sessions", "list local resumable sessions", "/sessions"),
    SlashCommandSpec(
        "resume",
        "open the session picker or resume by id",
        "/resume [--last|SESSION_ID]",
        supports_inline_args=True,
    ),
    SlashCommandSpec(
        "skill-reviews",
        "list pending skill update proposals",
        "/skill-reviews",
    ),
    SlashCommandSpec(
        "skill-review",
        "show one pending skill update proposal",
        "/skill-review <proposal-id>",
        supports_inline_args=True,
    ),
    SlashCommandSpec(
        "approve-skill-update",
        "approve and apply a pending skill update",
        "/approve-skill-update <proposal-id>",
        aliases=("approve-skill",),
        supports_inline_args=True,
    ),
    SlashCommandSpec(
        "reject-skill-update",
        "reject a pending skill update proposal",
        "/reject-skill-update <proposal-id> [reason]",
        aliases=("reject-skill",),
        supports_inline_args=True,
    ),
    SlashCommandSpec("session", "show current session id and trace path", "/session"),
    SlashCommandSpec("config", "show redacted provider config", "/config"),
    SlashCommandSpec(
        "run",
        "run a closed-loop agent episode",
        "/run [--max-turns N] <task>",
        supports_inline_args=True,
    ),
    SlashCommandSpec("step", "continue the current episode for one turn", "/step"),
    SlashCommandSpec("help", "show available commands", "/help", aliases=("?",)),
    SlashCommandSpec("quit", "exit OpenETA", "/quit", aliases=("exit",)),
)


def _command_lookup(commands: Iterable[SlashCommandSpec]) -> dict[str, SlashCommandSpec]:
    lookup: dict[str, SlashCommandSpec] = {}
    for command in commands:
        lookup[command.name] = command
        for alias in command.aliases:
            lookup[alias] = command
    return lookup


class SlashCommandCompleter(Completer):
    """Slash command popup completer for the OpenETA composer."""

    def __init__(self, commands: tuple[SlashCommandSpec, ...]) -> None:
        self._commands = commands

    def get_completions(self, document: Document, complete_event: Any) -> Iterable[Completion]:
        text = document.text_before_cursor
        first_line = text.splitlines()[0] if text else ""
        if not first_line.startswith("/") or text != first_line:
            return

        stripped = first_line[1:].lstrip()
        token = stripped.split(maxsplit=1)[0] if stripped else ""
        if first_line[1:].startswith(" ") or len(stripped.split(maxsplit=1)) > 1:
            return

        token_lower = token.lower()
        matches = [
            command
            for command in self._commands
            if not token_lower or command.name.startswith(token_lower)
        ]
        for command in matches:
            yield Completion(
                f"/{command.name} ",
                start_position=-len(first_line),
                display=f"/{command.name}",
                display_meta=command.description,
            )


@dataclass(slots=True)
class ConsoleState:
    """Mutable state for one terminal agent session."""

    config: PlannerProviderConfig = field(default_factory=load_planner_provider_config)
    runtime: OpenEtaAgentRuntime | None = None
    episode_runner: OpenEtaEpisodeRunner | None = None
    current_task: str = ""
    step_idx: int = 0
    continue_after_human: bool = False
    simulator_mcp_url: str = ""
    simulator_mcp_transport: SseSimulatorMcpTransport | None = None
    simulator_mcp_config: SimulatorMcpToolProxyConfig = field(
        default_factory=SimulatorMcpToolProxyConfig
    )
    simulator_mcp_tool_catalog: JsonDict = field(default_factory=dict)
    mcp_registry: JsonDict = field(default_factory=dict)
    supervision_profile: SupervisionProfile = field(
        default_factory=lambda: _configured_supervision_profile()
    )
    supervision_gate: SupervisionGate | None = None
    workspace: SessionWorkspace | None = None
    calibration_profile_path: str = ""


class OpenEtaCli:
    """Small Codex-like REPL for OpenETA provider config and agent traces."""

    def __init__(
        self,
        *,
        model_override: str = "",
        calibration_profile: str = "",
    ) -> None:
        self.state = ConsoleState()
        self.commands = SLASH_COMMANDS
        self.command_lookup = _command_lookup(self.commands)
        self.session: PromptSession[str] | None = None
        self._activity_started_at: float | None = None
        self._activity_status = "idle"
        self._tool_started_at: dict[str, list[float]] = {}
        self._shutdown_result: JsonDict | None = None
        if model_override:
            self.state.config.model = model_override
        self.state.calibration_profile_path = calibration_profile
        self._build_runtime()

    def _session(self) -> PromptSession[str]:
        if self.session is None:
            self.session = PromptSession(
                completer=SlashCommandCompleter(self.commands),
                complete_while_typing=True,
                complete_style=CompleteStyle.COLUMN,
                history=InMemoryHistory(),
                key_bindings=self._key_bindings(),
                style=Style.from_dict(
                    {
                        "prompt": "ansicyan bold",
                        "completion-menu.completion": "bg:#1f2937 #d1d5db",
                        "completion-menu.completion.current": "bg:#0e7490 #ffffff",
                        "completion-menu.meta.completion": "bg:#1f2937 #9ca3af",
                        "completion-menu.meta.completion.current": "bg:#0e7490 #e0f2fe",
                        "bottom-toolbar": "bg:#111827 #9ca3af",
                    }
                ),
                bottom_toolbar=self._bottom_toolbar,
            )
        return self.session

    def run(self) -> None:
        self._print_header()
        if not sys.stdin.isatty():
            print(Theme.warn("OpenETA CLI needs an interactive TTY. Run it with: uv run openeta"))
            return
        while True:
            try:
                if self._continue_after_human_answer():
                    continue
                line = self._session().prompt(HTML("<prompt>› </prompt>")).strip()
            except (EOFError, KeyboardInterrupt):
                print()
                print(Theme.dim("exiting (Ctrl-C)"))
                return
            try:
                if not line:
                    self._handle_empty_line()
                    continue
                if line.startswith("/"):
                    should_continue = self._handle_command(line)
                    if not should_continue:
                        print(Theme.dim("exiting"))
                        return
                    continue
                self.run_task(line, max_turns=DEFAULT_MAX_TURNS)
            except KeyboardInterrupt:
                print()
                self._finish_agent_activity("interrupted")
                self._mark_current_episode_interrupted()
                print(Theme.warn("interrupted by Ctrl-C"))
                print(Theme.dim("current run stopped; type /quit to exit"))

    def close(self) -> JsonDict:
        """Close the active MCP environment once before the console exits."""

        if self._shutdown_result is not None:
            return dict(self._shutdown_result)
        config = self.state.simulator_mcp_config
        with config.lifecycle_lock:
            transport = self.state.simulator_mcp_transport
            handle = config.handle
            session_id = config.session_id
            timeout_s = min(config.timeout_s, 30.0)
        if transport is None or not handle:
            self._shutdown_result = {
                "ok": True,
                "closed": False,
                "skipped": True,
            }
            return dict(self._shutdown_result)

        arguments: JsonDict = {"handle": handle}
        if session_id:
            arguments["session_id"] = session_id
        response = close_environment_mcp_env(
            transport,
            handle=handle,
            session_id=session_id,
            timeout_s=timeout_s,
        )
        payload = _load_response_payload(response)
        success = (
            payload.get("success") is not False
            and payload.get("ok") is not False
            and "error" not in payload
        )
        if success:
            try:
                self._sync_simulator_mcp_response("close_env", arguments, response)
            except Exception as exc:  # noqa: BLE001 - shutdown must not mask CLI exit.
                print(Theme.warn(f"MCP environment closed; local state sync failed: {exc}"))
            print(Theme.dim("active MCP environment closed"))
        else:
            error = payload.get("error") or payload.get("content") or "unknown cleanup error"
            print(Theme.warn(f"could not close active MCP environment: {error}"))
        self._shutdown_result = {
            "ok": success,
            "closed": success,
            "handle": handle,
            "session_id": session_id,
        }
        if not success:
            self._shutdown_result["error"] = str(
                payload.get("error") or payload.get("content") or "unknown cleanup error"
            )
        return dict(self._shutdown_result)

    def _key_bindings(self) -> KeyBindings:
        bindings = KeyBindings()

        @bindings.add("enter")
        def _(event: Any) -> None:
            buffer = event.current_buffer
            complete_state = buffer.complete_state
            if complete_state is not None:
                completion = complete_state.current_completion
                if completion is not None:
                    buffer.apply_completion(completion)
            buffer.validate_and_handle()

        @bindings.add("c-c")
        def _(event: Any) -> None:
            event.app.exit(exception=KeyboardInterrupt)

        @bindings.add("escape", filter=has_completions, eager=True)
        def _(event: Any) -> None:
            _cancel_active_completion(event.current_buffer)

        return bindings

    def _prompt_text(self, prompt: str, *, default: str = "", password: bool = False) -> str:
        suffix = f" [{default}]" if default and not password else ""
        prompt_markup = _prompt_html(f"{prompt}{suffix}: ")
        value = (
            self._session()
            .prompt(
                prompt_markup,
                is_password=password,
                complete_while_typing=False,
                bottom_toolbar=None,
            )
            .strip()
        )
        return value or default

    def _bottom_toolbar(self) -> HTML:
        model = self.state.config.model or "model:not-set"
        status = "ready" if not self.state.config.missing_fields() else "needs-config"
        session_id = ""
        if self.state.runtime is not None and self.state.runtime.memory.session_id:
            session_id = f"  session:{self.state.runtime.memory.session_id[:8]}"
        return HTML(
            f" provider:{self.state.config.provider}  model:{model}  "
            f"profile:{self.state.supervision_profile.value}  "
            f"status:{status}{session_id}  Ctrl-C:interrupt/exit "
        )

    def run_task(
        self,
        task: str,
        *,
        max_turns: int = 1,
        raise_on_interrupt: bool = False,
    ) -> None:
        missing = self.state.config.missing_fields()
        if missing:
            print(Theme.err(f"Provider config incomplete: {', '.join(missing)}"))
            if "model" in missing:
                print(Theme.dim("Run /models to inspect available models, then /model <name>."))
            else:
                print(Theme.dim("Run /provider to configure API access."))
            return
        self.state.current_task = task
        self.state.step_idx = 0
        runtime = self._require_runtime()
        self.state.episode_runner = OpenEtaEpisodeRunner(
            runtime=runtime,
            environment=ToolFeedbackEpisodeEnvironment(),
            interaction_resolver=self._interaction_resolver(),
        )
        self._begin_agent_activity()
        try:
            result = self.state.episode_runner.run(
                task=task,
                session_id=(
                    self.state.workspace.session_id
                    if self.state.workspace is not None
                    else None
                ),
                max_turns=max_turns,
                metadata={
                    "source": "OpenEtaCli",
                    "environment_mode": "tool_feedback",
                    "planner_prompt": dict(runtime.planner.prompt_metadata),
                    "workspace": (
                        self.state.workspace.to_dict()
                        if self.state.workspace is not None
                        else {}
                    ),
                },
            )
        except KeyboardInterrupt:
            self._finish_agent_activity("interrupted")
            self._mark_current_episode_interrupted()
            if raise_on_interrupt:
                raise
            print()
            print(Theme.warn("interrupted by Ctrl-C"))
            print(Theme.dim("current run stopped; type /quit to exit"))
            return
        except Exception:
            self._finish_agent_activity("failed")
            raise
        self._finish_agent_activity("worked")
        self._print_episode_result(result)

    def continue_task(
        self,
        *,
        max_turns: int | None = 1,
        raise_on_interrupt: bool = False,
    ) -> None:
        if not self.state.current_task or self.state.episode_runner is None:
            print("No active task. Type a task first.")
            return
        self._begin_agent_activity()
        try:
            result = self.state.episode_runner.continue_run(max_turns=max_turns)
        except KeyboardInterrupt:
            self._finish_agent_activity("interrupted")
            self._mark_current_episode_interrupted()
            if raise_on_interrupt:
                raise
            print()
            print(Theme.warn("interrupted by Ctrl-C"))
            print(Theme.dim("current run stopped; type /quit to exit"))
            return
        except Exception:
            self._finish_agent_activity("failed")
            raise
        self._finish_agent_activity("worked")
        self._print_episode_result(result)

    def _handle_empty_line(self) -> None:
        self._continue_after_human_answer()

    def _continue_after_human_answer(self) -> bool:
        if not self.state.continue_after_human:
            return False
        self.state.continue_after_human = False
        self.continue_task(max_turns=None)
        return True

    def _handle_command(self, line: str) -> bool:
        try:
            parts = shlex.split(line)
        except ValueError as exc:
            print(f"Command parse error: {exc}")
            return True
        if not parts:
            return True
        command = parts[0].lower()
        args = parts[1:]
        spec = self.command_lookup.get(command.removeprefix("/"))
        if spec is not None:
            command = f"/{spec.name}"

        if command in {"/quit", "/exit"}:
            return False
        if command in {"/help", "/?"}:
            self._print_help()
            return True
        if command == "/provider":
            self._configure_provider(args)
            return True
        if command == "/models":
            self._list_models()
            return True
        if command == "/model":
            self._select_model(args)
            return True
        if command == "/config":
            self._print_config()
            return True
        if command == "/tools":
            self._print_tools()
            return True
        if command == "/approvement":
            self._select_approvement(args)
            return True
        if command == "/memory":
            self._print_memory(args)
            return True
        if command == "/promote-memory":
            self._promote_memory(args)
            return True
        if command == "/new":
            self._new_session(args)
            return True
        if command == "/sessions":
            self._print_sessions()
            return True
        if command == "/resume":
            self._resume_session(args)
            return True
        if command == "/skill-reviews":
            self._print_skill_reviews()
            return True
        if command == "/skill-review":
            self._print_skill_review(args)
            return True
        if command == "/approve-skill-update":
            self._approve_skill_update(args)
            return True
        if command == "/reject-skill-update":
            self._reject_skill_update(args)
            return True
        if command == "/session":
            self._print_session()
            return True
        if command == "/step":
            self.state.continue_after_human = False
            self.continue_task(max_turns=1)
            return True
        if command == "/run":
            max_turns, task_args = _parse_run_args(args)
            if not task_args:
                print("Usage: /run [--max-turns N] <task>")
            else:
                self.run_task(" ".join(task_args), max_turns=max_turns)
            return True
        print(Theme.err(f"Unknown command: {command}. Type /help."))
        return True

    def _configure_provider(self, args: list[str]) -> None:
        current = load_planner_provider_config()
        provider = current.provider or "openai-compatible"
        api_base = current.api_base
        api_key = current.api_key
        model = current.model
        timeout_s = current.timeout_s
        max_attempts = current.max_attempts
        retry_backoff_s = current.retry_backoff_s
        context_window_tokens = current.context_window_tokens
        fallback = current.fallback

        if args:
            api_base = args[0]
            if len(args) >= 2:
                api_key = args[1]
            if len(args) >= 3:
                model = args[2]
            if len(args) >= 4:
                context_window_tokens = _parse_optional_positive_int(args[3])
        else:
            print("Configure OpenETA provider. Press Enter to keep current values.")
            provider = self._prompt_text("Provider", default=provider)
            api_base = self._prompt_text("API base", default=api_base)
            entered_key = self._prompt_text("API key (hidden, blank keeps current)", password=True)
            if entered_key:
                api_key = entered_key
            model = self._prompt_text("Model", default=model)
            timeout_raw = self._prompt_text("Timeout seconds", default=str(timeout_s))
            try:
                timeout_s = float(timeout_raw)
            except ValueError:
                timeout_s = current.timeout_s
            context_raw = self._prompt_text(
                "Context window tokens",
                default=str(context_window_tokens or ""),
            )
            context_window_tokens = _parse_optional_positive_int(context_raw)
            max_attempts_raw = self._prompt_text(
                "Provider attempts",
                default=str(max_attempts),
            )
            max_attempts = _parse_optional_positive_int(max_attempts_raw) or current.max_attempts
            retry_backoff_raw = self._prompt_text(
                "Retry backoff seconds",
                default=str(retry_backoff_s),
            )
            try:
                retry_backoff_s = max(0.0, float(retry_backoff_raw))
            except ValueError:
                retry_backoff_s = current.retry_backoff_s

            fallback_enabled = self._prompt_text(
                "Use fallback provider (y/n)",
                default="y" if fallback is not None else "n",
            ).lower()
            if fallback_enabled in {"y", "yes", "true", "1"}:
                existing_fallback = fallback or ProviderEndpointConfig(
                    provider=provider,
                    model=model,
                    timeout_s=timeout_s,
                )
                fallback_provider = self._prompt_text(
                    "Fallback provider",
                    default=existing_fallback.provider or provider,
                )
                fallback_api_base = self._prompt_text(
                    "Fallback API base",
                    default=existing_fallback.api_base,
                )
                fallback_api_key = existing_fallback.api_key
                entered_fallback_key = self._prompt_text(
                    "Fallback API key (hidden, blank keeps current)",
                    password=True,
                )
                if entered_fallback_key:
                    fallback_api_key = entered_fallback_key
                fallback_model = self._prompt_text(
                    "Fallback model",
                    default=existing_fallback.model or model,
                )
                fallback_timeout_raw = self._prompt_text(
                    "Fallback timeout seconds",
                    default=str(existing_fallback.timeout_s),
                )
                try:
                    fallback_timeout_s = float(fallback_timeout_raw)
                except ValueError:
                    fallback_timeout_s = existing_fallback.timeout_s
                fallback = ProviderEndpointConfig(
                    provider=fallback_provider,
                    model=fallback_model,
                    api_base=fallback_api_base.rstrip("/"),
                    api_key=fallback_api_key,
                    timeout_s=fallback_timeout_s,
                )
            else:
                fallback = None

        config = PlannerProviderConfig(
            provider=provider,
            model=model,
            api_base=api_base.rstrip("/"),
            api_key=api_key,
            timeout_s=timeout_s,
            max_attempts=max_attempts,
            retry_backoff_s=retry_backoff_s,
            context_window_tokens=context_window_tokens,
            fallback=fallback,
        )
        missing = config.missing_fields()
        if missing:
            print(Theme.err(f"Missing required fields: {', '.join(missing)}"))
            return
        write_env_file(config)
        self.state.config = config
        self._build_runtime()
        print(Theme.ok("Provider saved to .env"))
        self._print_config()

    def _list_models(self) -> None:
        config = OpenAICompatiblePlannerBackendConfig.from_provider_config(self.state.config)
        try:
            model_info = list_openai_compatible_model_info(config)
        except Exception as exc:  # noqa: BLE001 - CLI reports provider failures.
            print(f"Model listing failed: {type(exc).__name__}: {exc}")
            return
        if not model_info:
            print(Theme.warn("No models returned."))
            return
        print(Theme.dim("available models"))
        for model in model_info:
            marker = Theme.ok("●") if model.id == self.state.config.model else " "
            context = (
                f"  context={model.context_window_tokens}" if model.context_window_tokens else ""
            )
            print(f"  {marker} {model.id}{context}")

    def _select_model(self, args: list[str]) -> None:
        if args:
            self._save_model(args[0])
            return
        config = OpenAICompatiblePlannerBackendConfig.from_provider_config(self.state.config)
        try:
            model_info = list_openai_compatible_model_info(config)
        except Exception as exc:  # noqa: BLE001 - CLI reports provider failures.
            print(Theme.err(f"Model listing failed: {type(exc).__name__}: {exc}"))
            return
        if not model_info:
            print(Theme.warn("No models returned."))
            return
        print(Theme.dim("select model"))
        for idx, model in enumerate(model_info, start=1):
            marker = "*" if model.id == self.state.config.model else " "
            context = (
                f"  context={model.context_window_tokens}" if model.context_window_tokens else ""
            )
            print(f"  {idx:2d}. {marker} {model.id}{context}")
        choice = self._prompt_text("model ›")
        if not choice:
            return
        if choice.isdigit():
            idx = int(choice)
            if idx < 1 or idx > len(model_info):
                print(Theme.err("Model index out of range."))
                return
            selected = model_info[idx - 1]
            self._save_model(selected.id, context_window_tokens=selected.context_window_tokens)
            return
        self._save_model(choice)

    def _save_model(self, model: str, *, context_window_tokens: int | None = None) -> None:
        config = PlannerProviderConfig(
            provider=self.state.config.provider,
            model=model,
            api_base=self.state.config.api_base,
            api_key=self.state.config.api_key,
            timeout_s=self.state.config.timeout_s,
            max_attempts=self.state.config.max_attempts,
            retry_backoff_s=self.state.config.retry_backoff_s,
            context_window_tokens=(
                context_window_tokens
                if context_window_tokens is not None
                else self.state.config.context_window_tokens
            ),
            fallback=self.state.config.fallback,
            metadata=dict(self.state.config.metadata),
        )
        missing = config.missing_fields()
        if missing:
            print(Theme.err(f"Cannot save model, missing: {', '.join(missing)}"))
            return
        write_env_file(config)
        self.state.config = config
        self._build_runtime()
        print(Theme.ok(f"model set: {model}"))

    def _print_tools(self) -> None:
        runtime = self._require_runtime()
        print(Theme.dim("registered tools"))
        for spec in runtime.tools.list():
            handler = Theme.ok("handler") if runtime.tools.can_execute(spec.name) else "pending"
            print(
                f"  {spec.name:28} {spec.effect.value:14} {handler:8} "
                f"{spec.category} - {spec.description}"
            )

    def _select_approvement(self, args: list[str]) -> None:
        profiles = list(SupervisionProfile)
        if args:
            selected = args[0].strip().lower()
        else:
            print(Theme.dim("select supervision profile"))
            for index, profile in enumerate(profiles, start=1):
                marker = "*" if profile == self.state.supervision_profile else " "
                policy = SupervisionPolicy.for_profile(profile)
                print(
                    f"  {index}. {marker} {profile.value:20} "
                    f"actions={policy.world_mutation_mode} "
                    f"skills={policy.skill_change_mode} "
                    f"questions={policy.interaction_mode}"
                )
            selected = self._prompt_text("approvement ›")
            if selected.isdigit():
                index = int(selected)
                if not 1 <= index <= len(profiles):
                    print(Theme.err("Approvement index out of range."))
                    return
                selected = profiles[index - 1].value
        try:
            profile = SupervisionProfile(selected)
        except ValueError:
            print(
                Theme.err(
                    "Unknown supervision profile. Use human_gated, standard, or reviewed_autonomy."
                )
            )
            return
        self.state.supervision_profile = profile
        gate = self.state.supervision_gate
        if gate is not None:
            gate.set_profile(profile)
        if self.state.episode_runner is not None:
            self.state.episode_runner.interaction_resolver = self._interaction_resolver()
        runtime = self.state.runtime
        if runtime is not None:
            configure_runtime_self_improvement(
                runtime,
                policy=SupervisionPolicy.for_profile(profile),
                backend_factory=lambda **kwargs: _new_cli_backend(self, **kwargs),
            )
            runtime.memory.record(
                "supervision_profile_changed",
                SupervisionPolicy.for_profile(profile).to_dict(),
            )
        print(Theme.ok(f"supervision profile set: {profile.value}"))

    def _interaction_resolver(self) -> BackendGuidanceResolver | None:
        if self.state.supervision_profile != SupervisionProfile.REVIEWED_AUTONOMY:
            return None
        return BackendGuidanceResolver(_new_supervision_backend(self))

    def _print_memory(self, args: list[str]) -> None:
        runtime = self._require_runtime()
        namespace = "all"
        as_json = False
        for arg in args:
            if arg == "--json":
                as_json = True
            else:
                namespace = _normalize_memory_namespace(arg)
        payload = _memory_payload(runtime.memory, namespace)
        location = _memory_location(runtime.memory)
        if as_json:
            print_json({"location": location, "memory": payload})
            return

        print(Theme.dim("local memory"))
        print(f"  session_id   {runtime.memory.session_id or '(not started)'}")
        print(f"  session_path {location.get('session_path') or '(not available)'}")
        print(f"  working_dir  {location.get('working_dir') or '(not available)'}")
        working = runtime.memory.planning_context(max_events=0)["working_memory"]
        _print_memory_keys("facts", working.get("facts", {}))
        _print_memory_keys("artifacts", working.get("artifacts", {}))
        _print_memory_keys("skill_notes", working.get("skill_notes", {}))
        summary = working.get("compact_summary") or ""
        print(f"  compact_summary {summary or '(empty)'}")
        if namespace != "all":
            print()
            print_indented_json(payload, indent=2)

    def _promote_memory(self, args: list[str]) -> None:
        try:
            options = _parse_promote_memory_args(args)
        except ValueError as exc:
            print(Theme.err(str(exc)))
            print("Usage: /promote-memory <namespace> <key> [--target FILE] [--note TEXT]")
            return
        runtime = self._require_runtime()
        namespace = str(options["namespace"])
        key = str(options["key"])
        target = str(options["target"])
        note = str(options["note"])
        candidate = _memory_payload(runtime.memory, namespace)
        if namespace in {"facts", "artifacts"}:
            candidate = runtime.memory.get_memory(key, namespace=namespace)
        if namespace == "skill_notes":
            candidate = runtime.memory.get_memory(key, namespace=namespace)
        print(Theme.dim("promote candidate"))
        print_indented_json(candidate, indent=2)
        if not self.confirm(f"Promote {namespace}:{key or 'summary'} to agent/memory/{target}?"):
            print(Theme.warn("promotion cancelled"))
            return
        try:
            result = PromotedMemoryStore().promote(
                runtime.memory,
                namespace=namespace,
                key=key,
                target=target,
                note=note,
                reviewer="cli-user",
            )
        except (KeyError, ValueError) as exc:
            print(Theme.err(f"promotion failed: {exc}"))
            return
        print(Theme.ok(f"promoted memory: {result.entry_id}"))
        print(f"  path {result.path}")

    def _new_session(self, args: list[str]) -> None:
        clear_working_memory = False
        for arg in args:
            if arg == "--clear-working-memory":
                clear_working_memory = True
            else:
                print(Theme.err(f"unknown /new option: {arg}"))
                print("Usage: /new [--clear-working-memory]")
                return
        self.state.current_task = ""
        self.state.step_idx = 0
        self.state.continue_after_human = False
        self.state.episode_runner = None
        if clear_working_memory:
            runtime = self._require_runtime()
            if runtime.memory.session_id is None:
                print(Theme.warn("no active session working memory to clear"))
            elif not self.confirm(
                f"Clear working memory for session {runtime.memory.session_id[:12]}?"
            ):
                print(Theme.warn("new session cancelled"))
                return
            else:
                runtime.memory.clear_working_memory()
                print(Theme.ok("current session working memory cleared"))
        self.state.workspace = None
        self.state.simulator_mcp_tool_catalog = {}
        self._build_runtime()
        print(Theme.ok("new session started"))
        print(Theme.dim("session trace and working memory will start with the next task"))

    def _print_sessions(self) -> None:
        runtime = self._require_runtime()
        sessions = _list_local_sessions()
        if not sessions:
            print(Theme.warn("No local sessions found."))
            return
        self._print_session_picker(sessions, active_session_id=runtime.memory.session_id)
        print(Theme.dim("Use /resume to pick, /resume <session_id>, or /resume --last."))

    def _resume_session(self, args: list[str]) -> None:
        active_runtime = self._require_runtime()
        sessions = _list_local_sessions()
        if not sessions:
            print(Theme.warn("No local sessions found."))
            return
        if not args:
            self._print_session_picker(
                sessions,
                active_session_id=active_runtime.memory.session_id,
            )
            print(Theme.dim("Select a session by number or id. Press Enter to cancel."))
            selection = self._prompt_resume_selection()
            if not selection:
                print(Theme.warn("resume cancelled"))
                return
            try:
                session_id = self._resolve_resume_selection(selection, sessions)
            except ValueError as exc:
                print(Theme.err(str(exc)))
                return
        elif args == ["--last"]:
            session_id = str(sessions[0].get("session_id") or "")
        elif len(args) == 1:
            try:
                session_id = self._resolve_resume_selection(args[0], sessions)
            except ValueError as exc:
                print(Theme.err(str(exc)))
                return
        else:
            print("Usage: /resume [--last|SESSION_ID]")
            return
        selected = next(
            (
                session
                for session in sessions
                if str(session.get("session_id") or "") == session_id
            ),
            None,
        )
        if not session_id or selected is None:
            print(Theme.err(f"Unknown session: {session_id or args}"))
            return
        workspace = SessionWorkspace.create(
            session_id,
            environment_id=_environment_id_for_active_mcp_server(),
        )
        source_memory_root = str(selected.get("_memory_root") or "")
        if source_memory_root and Path(source_memory_root).resolve() != (
            workspace.memory_root.resolve()
        ):
            legacy_workspace_root = Path(source_memory_root).parent
            workspace.import_legacy_roots(
                memory_root=source_memory_root,
                artifact_root=legacy_workspace_root / "artifacts",
            )
        self.state.workspace = workspace
        self.state.simulator_mcp_tool_catalog = {}
        self._build_runtime()
        runtime = self._require_runtime()
        runtime.resume_session(session_id)
        self.state.current_task = runtime.memory.current_user_request or runtime.memory.task or ""
        self.state.step_idx = 0
        self.state.continue_after_human = False
        self.state.episode_runner = None
        self._save_mcp_registry_to_memory()
        self._save_simulator_mcp_tool_catalog_to_memory()
        print(Theme.ok(f"resumed session: {session_id}"))
        if runtime.memory.current_user_request or runtime.memory.task:
            print(f"  task {runtime.memory.current_user_request or runtime.memory.task}")
        location = _memory_location(runtime.memory)
        print(f"  trace {location.get('session_path') or '(not available)'}")
        print(f"  conversation {location.get('conversation_path') or '(not available)'}")
        print(f"  working {location.get('working_dir') or '(not available)'}")

    def _print_session_picker(
        self,
        sessions: list[JsonDict],
        *,
        active_session_id: str | None,
        limit: int = 20,
    ) -> None:
        print(Theme.dim("resumable sessions"))
        for idx, session in enumerate(sessions[:limit], start=1):
            session_id = str(session.get("session_id") or "")
            task = _truncate(_single_line(str(session.get("task") or "(untitled)")), 72)
            updated = _format_timestamp(session.get("updated_at_s"))
            event_count = session.get("event_count") or 0
            marker = "*" if active_session_id == session_id else " "
            print(
                f"  {idx:>2}. {marker} {session_id[:12]}  "
                f"events={event_count!s:>4}  {updated}  {task}"
            )
        if len(sessions) > limit:
            print(Theme.dim(f"showing {limit} of {len(sessions)} sessions"))

    def _prompt_resume_selection(self) -> str:
        if not sys.stdin.isatty():
            return ""
        try:
            return self._session().prompt(HTML("<prompt>resume› </prompt>")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return ""

    def _resolve_resume_selection(self, selection: str, sessions: list[JsonDict]) -> str:
        token = selection.strip()
        if not token:
            raise ValueError("resume selection is empty")
        if token.isdigit():
            index = int(token)
            if 1 <= index <= len(sessions):
                return str(sessions[index - 1].get("session_id") or "")
            raise ValueError(f"resume selection index out of range: {index}")

        matches = [
            str(session.get("session_id") or "")
            for session in sessions
            if str(session.get("session_id") or "").startswith(token)
        ]
        if len(matches) != 1:
            raise ValueError(f"resume needs one matching session id, found {len(matches)}")
        return matches[0]

    def _print_skill_reviews(self) -> None:
        proposals = SkillReviewProposalStore().list(status="pending")
        if not proposals:
            print(Theme.warn("No pending skill review proposals."))
            return
        print(Theme.dim("pending skill reviews"))
        for proposal in proposals:
            print(
                f"  {proposal.get('proposal_id')}  "
                f"{proposal.get('action')}:{proposal.get('skill_name')}  "
                f"{_format_timestamp(proposal.get('created_at_s'))}"
            )
            rationale = str(proposal.get("rationale") or "").strip()
            if rationale:
                print(f"    {Theme.dim(_truncate(_single_line(rationale), 120))}")
        print(Theme.dim("Use /skill-review <id> to inspect a proposal."))

    def _print_skill_review(self, args: list[str]) -> None:
        if not args:
            print("Usage: /skill-review <proposal-id>")
            return
        try:
            proposal = SkillReviewProposalStore().load(args[0])
        except (FileNotFoundError, ValueError) as exc:
            print(Theme.err(str(exc)))
            return
        _print_skill_review_proposal(proposal)

    def _approve_skill_update(self, args: list[str]) -> None:
        if not args:
            print("Usage: /approve-skill-update <proposal-id>")
            return
        store = SkillReviewProposalStore()
        try:
            proposal = store.load(args[0])
        except (FileNotFoundError, ValueError) as exc:
            print(Theme.err(str(exc)))
            return
        _print_skill_review_proposal(proposal)
        target = f"agent/skills/{proposal.get('skill_name')}.md"
        if not self.confirm(f"Apply skill review {proposal.get('proposal_id')} to {target}?"):
            print(Theme.warn("skill update approval cancelled"))
            return
        try:
            updated = store.approve(str(proposal["proposal_id"]), reviewer="cli-user")
        except (FileNotFoundError, FileExistsError, KeyError, ValueError) as exc:
            print(Theme.err(f"skill update failed: {exc}"))
            return
        resolution = updated.get("resolution", {})
        application = resolution.get("application", {}) if isinstance(resolution, dict) else {}
        print(Theme.ok(f"skill update approved: {updated.get('proposal_id')}"))
        if isinstance(application, dict):
            print(f"  path {application.get('target_path')}")
            print(f"  bytes_written {application.get('bytes_written')}")

    def _reject_skill_update(self, args: list[str]) -> None:
        if not args:
            print("Usage: /reject-skill-update <proposal-id> [reason]")
            return
        reason = " ".join(args[1:]).strip()
        try:
            updated = SkillReviewProposalStore().reject(
                args[0],
                reviewer="cli-user",
                reason=reason,
            )
        except (FileNotFoundError, ValueError) as exc:
            print(Theme.err(str(exc)))
            return
        print(Theme.ok(f"skill update rejected: {updated.get('proposal_id')}"))

    def _print_session(self) -> None:
        runtime = self._require_runtime()
        location = _memory_location(runtime.memory)
        print(Theme.dim("session"))
        print(f"  session_id   {runtime.memory.session_id or '(not started)'}")
        print(
            "  task         "
            f"{runtime.memory.current_user_request or self.state.current_task or '(not started)'}"
        )
        print(f"  turn_index   {self.state.step_idx}")
        print(f"  session_path {location.get('session_path') or '(not available)'}")
        print(f"  conversation {location.get('conversation_path') or '(not available)'}")
        print(f"  working_dir  {location.get('working_dir') or '(not available)'}")
        print("  scope        trace and working memory are scoped to this session")

    def _mark_current_episode_interrupted(self) -> None:
        runner = self.state.episode_runner
        worker_idle = True
        if runner is not None:
            runner.interrupt(code="user_interrupt")
            worker_idle = runner.wait_for_idle()
        runtime = self.state.runtime
        if runtime is not None:
            runtime.memory.record(
                "cli_interrupt",
                {
                    "task": self.state.current_task,
                    "turn_index": self.state.step_idx,
                    "execution_id": runner.execution_id if runner is not None else "",
                    "worker_idle": worker_idle,
                    "late_results_fenced": True,
                },
            )
        self.state.continue_after_human = False
        self.state.episode_runner = None

    def _build_runtime(self) -> None:
        if self.state.workspace is None:
            workspace_options: dict[str, str] = {}
            if self.state.calibration_profile_path:
                workspace_options["source_grasp_profile"] = (
                    self.state.calibration_profile_path
                )
            self.state.workspace = SessionWorkspace.create(
                str(uuid4()),
                environment_id=_environment_id_for_active_mcp_server(),
                **workspace_options,
            )
        workspace = self.state.workspace
        proxy_config = self.state.simulator_mcp_config
        proxy_config.timeout_s = max(
            DEFAULT_SIM_MCP_TIMEOUT_S,
            self.state.config.timeout_s,
        )
        proxy_config.image_output_root = workspace.artifacts_dir / "images"
        proxy_config.text_output_root = workspace.artifacts_dir / "text"
        proxy_config.response_output_root = workspace.artifacts_dir / "responses"
        self._refresh_mcp_registry()
        transport = _ensure_simulator_mcp_transport(self)
        policy = SupervisionPolicy.for_profile(self.state.supervision_profile)
        assembly = assemble_runtime(
            RuntimeAssemblyConfig(
                workspace=workspace,
                provider=self.state.config,
                backend_factory=lambda **kwargs: _new_cli_backend(self, **kwargs),
                supervision_policy=policy,
                supervision_policy_provider=lambda: SupervisionPolicy.for_profile(
                    self.state.supervision_profile
                ),
                endpoints=resolve_runtime_mcp_endpoints(
                    RuntimeMcpEndpoints(),
                    loader=_load_mcp_url,
                ),
                simulator_transport=transport,
                simulator_proxy_config=proxy_config,
                mcp_response_callback=self._sync_simulator_mcp_response,
                allow_outside_sandbox=True,
                approve_outside_sandbox=lambda context, mode: self.confirm(
                    f"Approve {mode} python_exec command? "
                    f"{json.dumps(_python_exec_approval_summary(context), ensure_ascii=False)}"
                ),
                human_action_approval=lambda context: self.confirm(
                    f"Approve world-mutating {context.name} command "
                    f"{json.dumps(context.parameters, ensure_ascii=False)}?"
                ),
                calibration_approval=lambda request: self.confirm(
                    "Publish calibration "
                    f"{request.get('calibration_id')} as "
                    f"{request.get('target_status')}?"
                ),
                strategy_approval=lambda request: self.confirm(
                    "Publish grasp strategy "
                    f"{request.get('strategy_id')} as "
                    f"{request.get('target_status')}?"
                ),
                skill_approval=lambda name: self.confirm(
                    f"Approve skill registry change for {name}?"
                ),
                pre_safety_checks=_configured_pre_safety_checks(),
                tool_listeners=(self._print_tool_event,),
                max_validation_retries=2,
            )
        )
        self.state.runtime = assembly.runtime
        self.state.supervision_gate = assembly.supervision_gate
        self.state.runtime.memory.save_fact(
            "session_workspace",
            workspace.to_dict(),
            source="runtime_assembly",
        )
        self.state.runtime.memory.save_fact(
            "supervision",
            policy.to_dict(),
            source="runtime_assembly",
        )
        self._save_mcp_registry_to_memory()
        self._save_simulator_mcp_tool_catalog_to_memory()
        self.state.episode_runner = None

    def _require_runtime(self) -> OpenEtaAgentRuntime:
        if self.state.runtime is None:
            self._build_runtime()
        assert self.state.runtime is not None
        return self.state.runtime

    def _sync_simulator_mcp_response(
        self,
        tool_name: str,
        arguments: JsonDict,
        response: JsonDict,
    ) -> None:
        if tool_name not in {
            "create_env",
            "reset_env",
            "render_env",
            "move_to",
            "close_env",
        }:
            return
        payload = _load_response_payload(response)
        if payload.get("success") is False or payload.get("ok") is False or "error" in payload:
            return
        if tool_name == "close_env":
            with self.state.simulator_mcp_config.lifecycle_lock:
                self.state.simulator_mcp_config.handle = ""
                self.state.simulator_mcp_config.session_id = ""
                self.state.simulator_mcp_config.image_bundle_id = ""
            if self.state.runtime is not None:
                closed = {
                    "type": "simulator_mcp_state",
                    "tool": tool_name,
                    "handle": arguments.get("handle"),
                    "session_id": arguments.get("session_id"),
                    "status": "closed",
                }
                self.state.runtime.memory.save_fact(
                    "simulator_mcp_state",
                    closed,
                    source="simulator_agent_tool",
                )
                self.state.runtime.memory.save_artifact(
                    "simulator_mcp_state",
                    closed,
                    source="simulator_agent_tool",
                )
            return
        handle = payload.get("handle") or arguments.get("handle")
        session_id = payload.get("session_id") or arguments.get("session_id")
        env_id = payload.get("env_id") or arguments.get("env_id")
        changed = False
        if isinstance(handle, str) and handle:
            self.state.simulator_mcp_config.handle = handle
            changed = True
        if isinstance(session_id, str) and session_id:
            self.state.simulator_mcp_config.session_id = session_id
            self.state.simulator_mcp_config.image_bundle_id = session_id
            changed = True
        if changed and self.state.runtime is not None:
            server_url = mcp_server_url_from_endpoint(self.state.simulator_mcp_url)
            dashboard_url = mcp_dashboard_url(
                server_url,
                self.state.simulator_mcp_config.session_id,
            )
            payload: JsonDict = {
                "type": "simulator_mcp_state",
                "tool": tool_name,
                "handle": self.state.simulator_mcp_config.handle,
                "session_id": self.state.simulator_mcp_config.session_id,
            }
            if server_url:
                payload["mcp_server_url"] = server_url
            if dashboard_url:
                payload["dashboard_url"] = dashboard_url
            if isinstance(env_id, str) and env_id:
                payload["env_id"] = env_id
            self.state.runtime.memory.save_fact(
                "simulator_mcp_state",
                payload,
                source="simulator_agent_tool",
            )
            self.state.runtime.memory.save_artifact(
                "simulator_mcp_state",
                payload,
                source="simulator_agent_tool",
            )

    def _save_simulator_mcp_tool_catalog_to_memory(self) -> None:
        if self.state.runtime is None or not self.state.simulator_mcp_tool_catalog:
            return
        self.state.runtime.memory.save_fact(
            "simulator_mcp_tool_catalog",
            self.state.simulator_mcp_tool_catalog,
            source="mcp.list_tools",
        )

    def _refresh_mcp_registry(self) -> None:
        self.state.mcp_registry = compact_mcp_registry()

    def _save_mcp_registry_to_memory(self) -> None:
        if self.state.runtime is None or not self.state.mcp_registry:
            return
        self.state.runtime.memory.save_fact(
            "mcp_registry",
            self.state.mcp_registry,
            source=".mcp.json",
        )

    def _print_episode_result(self, result: EpisodeResult) -> None:
        if not result.steps:
            print(Theme.warn("No episode steps executed."))
            return
        self._print_session_summary()
        for step in result.steps:
            self._print_episode_step(step)
        if result.terminated:
            reason = result.metadata.get("stop_reason") or "done"
            if reason == "status_report":
                print(Theme.warn("episode stopped: status_report"))
            else:
                print(Theme.ok(f"episode terminated: {reason}"))
        if result.truncated:
            reason = result.metadata.get("stop_reason") or "truncated"
            print(Theme.warn(f"episode truncated: {reason}"))
        if (
            result.metadata.get("waiting_for_human")
            and self.state.episode_runner is not None
            and self.state.episode_runner.waiting_for_human
        ):
            print(Theme.warn("episode paused: waiting for human input"))

    def _print_session_summary(self) -> None:
        runtime = self._require_runtime()
        location = _memory_location(runtime.memory)
        print()
        print(
            f"{Theme.dim('session')} id={runtime.memory.session_id or '(not started)'} "
            f"trace={location.get('session_path') or '(not available)'}"
        )

    def _print_episode_step(self, step: EpisodeStep) -> None:
        self.state.step_idx = step.turn_index
        print()
        print(Theme.dim(f"turn {step.turn_index}"))
        if step.turn_index == 1:
            print(f"{Theme.accent('user')} {step.observation.task}")
            print(
                f"  {Theme.dim('observation')} "
                f"step={step.observation.metadata.get('step_idx')} "
                f"objects={len(step.observation.objects)}"
            )
        self._print_action_trace(step.action)
        print(
            f"  {Theme.dim('feedback')} "
            f"reward={step.step_result.reward} "
            f"terminated={step.step_result.terminated} "
            f"truncated={step.step_result.truncated} "
            f"next_step={step.step_result.observation.metadata.get('step_idx')}"
        )
        print(f"  {Theme.dim('memory')} episode_step recorded")
        self._handle_interactive_action(step.action, show_message=False)

    def _print_action_trace(self, action: EnvAction) -> None:
        command = action.command
        request = command.get("request", {})
        metadata = command.get("metadata", {}).get("planner_metadata", {})
        backend = metadata.get("backend", {})
        details = metadata.get("backend_details", {})

        provider = backend.get("provider", metadata.get("backend_provider"))
        model = backend.get("model", metadata.get("backend_model"))
        print(f"{Theme.accent('agent')} planner {provider} / {model}")
        usage = details.get("usage")
        if usage:
            usage_source = str(details.get("usage_source") or "provider")
            print(f"  {Theme.dim('usage')} {compact_usage(usage)} source={usage_source}")
        print(f"  request {request.get('kind')}::{request.get('name')} -> {command.get('status')}")
        reasoning = request.get("reasoning")
        if reasoning:
            print(f"  thinking {reasoning}")
        if request.get("kind") == "response":
            self._print_response_request(request)
            return
        parameters = request.get("parameters", {})
        if parameters:
            detail_width = _tool_detail_width()
            compact_parameters = _compact_tool_parameters(
                str(request.get("name") or ""),
                parameters,
                max_chars=detail_width * 2,
            )
            if compact_parameters:
                compact_parameters = _truncate_display_width(
                    compact_parameters,
                    detail_width,
                )
                print(f"  {Theme.dim('parameters')} {compact_parameters}")

        self._print_calls("safety", command.get("safety_checks", []))
        self._print_calls("tool", command.get("tool_calls", []))
        if command.get("skill_call"):
            self._print_calls("skill", [command["skill_call"]])

    def _print_response_request(self, request: JsonDict) -> None:
        name = str(request.get("name") or "response")
        parameters = request.get("parameters", {})
        if not isinstance(parameters, dict):
            parameters = {"value": parameters}
        message = _response_message(name, parameters)
        if not message:
            return
        label = "ask_human" if name == "ask_human" else "assistant"
        printer = Theme.warn if name == "ask_human" else Theme.accent
        print_wrapped_message(f"  {printer(label)} ", message)

    def _print_calls(self, label: str, calls: list[JsonDict]) -> None:
        for idx, call in enumerate(calls, start=1):
            title = f"{label} call" if label != "tool" else "tool call"
            print(f"  {Theme.accent(title)} {idx}: {call.get('name')} -> {call.get('status')}")
            reason = call.get("reason")
            if reason:
                print(f"    {Theme.dim('reason')} {reason}")
            result = call.get("result")
            if result is not None:
                lines = _compact_tool_result_lines(
                    str(call.get("name") or ""),
                    result,
                    max_lines=TOOL_RESULT_MAX_LINES,
                    line_width=_tool_detail_width(),
                )
                if lines:
                    print(f"    {Theme.dim('result')} {Theme.dim(lines[0])}")
                    for line in lines[1:]:
                        print(f"           {Theme.dim(line)}")

    def _print_tool_event(self, event: JsonDict) -> None:
        name = str(event.get("name") or "(unknown)")
        phase = event.get("phase")
        effect = event.get("effect")
        if phase == "start":
            self._mark_agent_working()
            self._tool_started_at.setdefault(name, []).append(time.monotonic())
            effect_text = f" {Theme.warn(effect)}" if effect == "world_mutating" else ""
            detail_width = _tool_detail_width()
            params = _compact_tool_parameters(
                name,
                event.get("parameters", {}),
                max_chars=detail_width * 2,
            )
            params = _truncate_display_width(params, detail_width)
            suffix = f" {Theme.dim(params)}" if params else ""
            print(f"  {Theme.accent('→')} {Theme.bold(name)}{effect_text}{suffix}", flush=True)
            return
        if phase == "end":
            success = bool(event.get("success", False))
            mark = Theme.ok("✓") if success else Theme.err("✗")
            elapsed = self._pop_tool_elapsed(name)
            elapsed_text = f" {Theme.dim(_format_elapsed(elapsed))}" if elapsed is not None else ""
            content = _single_line(str(event.get("content") or ""))
            suffix = f" {Theme.dim(_truncate(content, 160))}" if content else ""
            print(f"  {mark} {Theme.bold(name)}{elapsed_text}{suffix}", flush=True)

    def _begin_agent_activity(self) -> None:
        self._activity_started_at = time.monotonic()
        self._activity_status = "thinking"
        self._tool_started_at.clear()
        print(f"{Theme.dim('○')} thinking", flush=True)

    def _mark_agent_working(self) -> None:
        if self._activity_status == "working":
            return
        self._activity_status = "working"
        print(f"{Theme.accent('●')} working", flush=True)

    def _finish_agent_activity(self, status: str) -> None:
        started_at = self._activity_started_at
        if started_at is None:
            return
        elapsed = time.monotonic() - started_at
        if status == "worked":
            print(f"{Theme.ok('✓')} worked for {_format_elapsed(elapsed)}", flush=True)
        elif status == "interrupted":
            print(f"{Theme.warn('■')} interrupted after {_format_elapsed(elapsed)}", flush=True)
        else:
            print(f"{Theme.err('✗')} failed after {_format_elapsed(elapsed)}", flush=True)
        self._activity_started_at = None
        self._activity_status = "idle"
        self._tool_started_at.clear()

    def _pop_tool_elapsed(self, name: str) -> float | None:
        starts = self._tool_started_at.get(name)
        if not starts:
            return None
        started_at = starts.pop()
        if not starts:
            self._tool_started_at.pop(name, None)
        return time.monotonic() - started_at

    def _handle_interactive_action(self, action: EnvAction, *, show_message: bool = True) -> None:
        request = action.command.get("request", {})
        if not (request.get("kind") == "response" and request.get("name") == "ask_human"):
            return
        if (
            self.state.episode_runner is not None
            and not self.state.episode_runner.waiting_for_human
        ):
            return
        params = request.get("parameters", {})
        message = (
            params.get("question")
            or params.get("message")
            or request.get("name")
            or "Agent asks for input"
        )
        if show_message:
            print_wrapped_message(f"{Theme.warn('ask_human')} ", message)
        answer = self._prompt_text("›")
        runtime = self._require_runtime()
        runtime.update_memory(
            {
                "type": "human_answer",
                "question": message,
                "answer": answer,
            }
        )
        if self.state.episode_runner is not None:
            self.state.episode_runner.resume_after_human()
        self.state.continue_after_human = True
        print(Theme.dim("human answer recorded; resuming current task"))

    def confirm(self, message: str) -> bool:
        if not sys.stdin.isatty():
            print(Theme.warn(f"permission required: {message}"))
            return False
        answer = self._prompt_text(f"permission {message} [y/N]").lower()
        return answer in {"y", "yes"}

    def _print_header(self) -> None:
        config = self.state.config.redacted()
        model = config.get("model") or "(not set)"
        status = "ready" if not self.state.config.missing_fields() else "needs config"
        width = 72
        print(Theme.accent("╭" + "─" * (width - 2) + "╮"))
        print_box_line("OpenETA Agent Console", width=width, strong=True)
        print_box_line(f"provider  {config.get('provider')}    model  {model}", width=width)
        print_box_line(f"base      {config.get('api_base') or '(not set)'}", width=width)
        print_box_line(
            f"key       {config.get('api_key') or '(not set)'}    status  {status}",
            width=width,
        )
        context_window = config.get("context_window_tokens") or "(not set)"
        print_box_line(f"context  {context_window}", width=width)
        fallback = config.get("fallback")
        if isinstance(fallback, dict):
            print_box_line(
                f"fallback  {fallback.get('provider')}    model  {fallback.get('model')}",
                width=width,
            )
        print_box_line(f"profile  {self.state.supervision_profile.value}", width=width)
        print(Theme.accent("╰" + "─" * (width - 2) + "╯"))
        print(Theme.dim("  Type / to open commands. Press Ctrl-C to exit."))
        if not self.state.config.model:
            print(Theme.warn("  model is not set. Open /models or /model from the command popup."))

    def _print_help(self) -> None:
        print()
        print(Theme.bold("Commands"))
        for command in self.commands:
            alias_text = (
                f" aliases: {', '.join('/' + alias for alias in command.aliases)}"
                if command.aliases
                else ""
            )
            print(f"  {command.usage:18} {command.description}{alias_text}")
        print()
        print("Any non-command line is treated as a task and runs one closed-loop turn.")
        print("Press Ctrl-C to exit immediately; /quit also exits.")

    def _print_config(self) -> None:
        config = self.state.config.redacted()
        print(Theme.dim("provider config"))
        rows = [
            ("provider", config["provider"]),
            ("model", config["model"] or "(not set)"),
            ("api_base", config["api_base"] or "(not set)"),
            ("api_key", config["api_key"] or "(not set)"),
            ("timeout_s", config["timeout_s"]),
            ("max_attempts", config["max_attempts"]),
            ("retry_backoff_s", config["retry_backoff_s"]),
            ("context_window_tokens", config["context_window_tokens"] or "(not set)"),
        ]
        for key, value in rows:
            print(f"  {key:22} {value}")
        fallback = config.get("fallback")
        if isinstance(fallback, dict):
            for key in ("provider", "model", "api_base", "api_key", "timeout_s"):
                print(f"  {'fallback.' + key:22} {fallback.get(key) or '(not set)'}")


def _print_skill_review_proposal(proposal: JsonDict) -> None:
    print(Theme.dim("skill review proposal"))
    rows = [
        ("proposal_id", proposal.get("proposal_id")),
        ("status", proposal.get("status")),
        ("action", proposal.get("action")),
        ("skill_name", proposal.get("skill_name")),
        ("created_at", _format_timestamp(proposal.get("created_at_s"))),
        ("path", proposal.get("path")),
    ]
    for key, value in rows:
        print(f"  {key:11} {value or '(unknown)'}")
    rationale = str(proposal.get("rationale") or "").strip()
    if rationale:
        print(f"  {'rationale':11} {_single_line(rationale)}")
    signals = proposal.get("signals")
    if isinstance(signals, dict) and signals:
        print("  signals")
        print_indented_json(signals, indent=4)
    suggested = str(proposal.get("suggested_markdown") or "").strip()
    if suggested:
        print("  suggested_markdown")
        print_wrapped_message("    ", suggested)


def _format_timestamp(value: object) -> str:
    try:
        timestamp = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "(unknown)"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(timestamp))


def _build_cli_checker_config(tools: ToolRegistry) -> CheckerSubagentConfig:
    """Build CLI hooks without treating dummy checks as real safety gates."""

    pre_safety_checks = _configured_pre_safety_checks()
    for target, checker in pre_safety_checks.items():
        tools.get(target)
        checker_spec = tools.get(checker)
        if checker_spec.category != "safety":
            raise ValueError(f"Configured checker is not a safety tool: {checker}")
        if not tools.can_execute(checker):
            raise ValueError(f"Configured checker has no executable handler: {checker}")

    return CheckerSubagentConfig(
        pre_safety_checks=pre_safety_checks,
        post_failure_checks=tuple(spec.name for spec in tools.list()),
    )


def _configured_pre_safety_checks() -> dict[str, str]:
    raw = os.environ.get("OPENETA_PRE_SAFETY_CHECKS", "").strip()
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("OPENETA_PRE_SAFETY_CHECKS must be a JSON object") from exc
    if not isinstance(parsed, dict):
        raise ValueError("OPENETA_PRE_SAFETY_CHECKS must be a JSON object")
    checks: dict[str, str] = {}
    for target, checker in parsed.items():
        if not isinstance(target, str) or not isinstance(checker, str):
            raise ValueError("OPENETA_PRE_SAFETY_CHECKS keys and values must be strings")
        checks[target] = checker
    return checks


def _ensure_simulator_mcp_transport(cli: OpenEtaCli) -> SseSimulatorMcpTransport | None:
    url = _load_sim_mcp_url()
    if not url:
        return None
    if cli.state.simulator_mcp_transport is None or cli.state.simulator_mcp_url != url:
        cli.state.simulator_mcp_url = url
        cli.state.simulator_mcp_transport = SseSimulatorMcpTransport(url)
        _refresh_simulator_mcp_tool_catalog(cli)
    elif not cli.state.simulator_mcp_tool_catalog:
        _refresh_simulator_mcp_tool_catalog(cli)
    return cli.state.simulator_mcp_transport


def _refresh_simulator_mcp_tool_catalog(cli: OpenEtaCli) -> None:
    transport = cli.state.simulator_mcp_transport
    workspace = cli.state.workspace
    if workspace is None:
        return
    cli.state.simulator_mcp_tool_catalog = discover_mcp_tool_catalog(
        transport,
        endpoint_url=cli.state.simulator_mcp_url,
        output_root=workspace.artifacts_dir / "responses",
    )


def _load_response_payload(response: JsonDict) -> JsonDict:
    payload = dict(response)
    response_path = payload.get("response_path")
    if not isinstance(response_path, str) or not response_path:
        return payload
    try:
        loaded = json.loads(Path(response_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return payload
    if isinstance(loaded, dict):
        merged = dict(loaded)
        merged.update(payload)
        return merged
    return payload


def _python_exec_approval_summary(context: ToolExecutionContext) -> JsonDict:
    code = str(context.parameters.get("code", "") or "")
    return {
        "sandbox": context.parameters.get("sandbox", "sandbox"),
        "code_chars": len(code),
        "first_line": code.strip().splitlines()[0] if code.strip() else "",
    }


# Maps the OPENETA_SIM_MCP_SERVER entry name to the embodiment environment_id
# used for calibration selection. The real UR5e bench reports env_id "real",
# which the calibration registry matches to the robotiq-2f85 profile.
_MCP_SERVER_ENVIRONMENT_IDS = {
    "openeta-ur5e": "real",
}


def _environment_id_for_active_mcp_server() -> str:
    """Return the calibration environment_id implied by OPENETA_SIM_MCP_SERVER."""
    server_name = os.environ.get("OPENETA_SIM_MCP_SERVER", "").strip()
    return _MCP_SERVER_ENVIRONMENT_IDS.get(server_name, "")


def _load_sim_mcp_url(path: str | Path = ".mcp.json") -> str:
    # OPENETA_SIM_MCP_SERVER selects which .mcp.json entry drives the environment
    # MCP (e.g. openeta-ur5e for the real bench); defaults to openeta-sim.
    override = os.environ.get("OPENETA_SIM_MCP_SERVER", "").strip()
    server_name = override or "openeta-sim"
    url = _load_mcp_url(server_name, aliases=("openeta",), path=path)
    if url:
        return url
    # An explicit override that resolves to nothing must fail closed rather than
    # silently connecting to whichever server happens to be listed first.
    if override:
        return ""
    configs = load_mcp_server_configs(path)
    for config in configs.values():
        if config.url:
            return config.url
    return ""


def _load_mcp_url(
    name: str,
    *,
    aliases: tuple[str, ...] = (),
    path: str | Path = ".mcp.json",
) -> str:
    return load_mcp_server_url(name, aliases=aliases, path=path)


def _new_supervision_backend(cli: OpenEtaCli) -> OpenAICompatiblePlannerBackend:
    return _new_cli_backend(cli, max_tokens=512)


def _new_cli_backend(
    cli: OpenEtaCli,
    *,
    max_tokens: int | None = None,
    max_vision_images: int | None = None,
) -> OpenAICompatiblePlannerBackend:
    backend_config = OpenAICompatiblePlannerBackendConfig.from_provider_config(cli.state.config)
    if max_tokens is not None:
        backend_config.max_tokens = max_tokens
    if max_vision_images is not None:
        backend_config.max_vision_images = max(
            backend_config.max_vision_images,
            max_vision_images,
        )
    return OpenAICompatiblePlannerBackend(backend_config)


def _configured_supervision_profile() -> SupervisionProfile:
    raw = os.environ.get("OPENETA_SUPERVISION_PROFILE", "standard").strip().lower()
    try:
        return SupervisionProfile(raw)
    except ValueError:
        return SupervisionProfile.STANDARD


def _memory_payload(memory: AgentMemory, namespace: str) -> JsonDict:
    if namespace == "compact_summary":
        return {"compact_summary": memory.compact_summary}
    return memory.get_memory(namespace=namespace)


def _normalize_memory_namespace(namespace: str) -> str:
    normalized = namespace.strip()
    if normalized == "compact":
        return "compact_summary"
    if normalized in {"all", "facts", "artifacts", "skill_notes", "compact_summary"}:
        return normalized
    return "all"


def _memory_location(memory: AgentMemory) -> JsonDict:
    store = memory.store
    if isinstance(store, JsonMemoryStore):
        session_path = (
            str(store.session_path(memory.session_id)) if memory.session_id is not None else ""
        )
        conversation_path = (
            str(store.conversation_path(memory.session_id))
            if memory.session_id is not None
            else ""
        )
        return {
            "root": str(store.root),
            "working_dir": str(store.working_dir),
            "session_path": session_path,
            "conversation_path": conversation_path,
        }
    return {"root": "", "working_dir": "", "session_path": "", "conversation_path": ""}


def _list_local_sessions() -> list[JsonDict]:
    roots = [Path(".openeta_memory")]
    if LEGACY_SESSION_WORKSPACE_ROOT.exists():
        roots.extend(
            path
            for path in LEGACY_SESSION_WORKSPACE_ROOT.glob("*/memory")
            if path.is_dir()
        )
    sessions_by_id: dict[str, JsonDict] = {}
    for root in roots:
        store = JsonMemoryStore(root=root)
        for session in store.list_sessions():
            session_id = str(session.get("session_id") or "")
            if not session_id:
                continue
            candidate = {**session, "_memory_root": str(root)}
            existing = sessions_by_id.get(session_id)
            if existing is None or float(candidate.get("updated_at_s") or 0.0) > float(
                existing.get("updated_at_s") or 0.0
            ):
                sessions_by_id[session_id] = candidate
    return sorted(
        sessions_by_id.values(),
        key=lambda item: float(item.get("updated_at_s") or 0.0),
        reverse=True,
    )


def _print_memory_keys(namespace: str, values: object) -> None:
    if not isinstance(values, dict) or not values:
        print(f"  {namespace:15} (empty)")
        return
    print(f"  {namespace:15} {', '.join(sorted(str(key) for key in values))}")


def _parse_promote_memory_args(args: list[str]) -> JsonDict:
    if len(args) < 2:
        raise ValueError("promote-memory requires namespace and key")
    options: JsonDict = {
        "namespace": _normalize_memory_namespace(args[0]),
        "key": args[1],
        "target": "project_memory.md",
        "note": "",
    }
    idx = 2
    while idx < len(args):
        arg = args[idx]
        if arg == "--target" and idx + 1 < len(args):
            options["target"] = args[idx + 1]
            idx += 2
            continue
        if arg == "--note" and idx + 1 < len(args):
            options["note"] = args[idx + 1]
            idx += 2
            continue
        raise ValueError(f"unknown promote-memory option: {arg}")
    return options


def _parse_run_args(args: list[str]) -> tuple[int, list[str]]:
    max_turns = DEFAULT_MAX_TURNS
    task_args: list[str] = []
    idx = 0
    while idx < len(args):
        arg = args[idx]
        if arg == "--max-turns" and idx + 1 < len(args):
            try:
                max_turns = max(1, int(args[idx + 1]))
            except ValueError:
                max_turns = DEFAULT_MAX_TURNS
            idx += 2
            continue
        if arg.startswith("--max-turns="):
            raw_value = arg.split("=", 1)[1]
            try:
                max_turns = max(1, int(raw_value))
            except ValueError:
                max_turns = DEFAULT_MAX_TURNS
            idx += 1
            continue
        task_args.append(arg)
        idx += 1
    return max_turns, task_args


def _parse_optional_positive_int(value: str | int | None) -> int | None:
    if value is None:
        return None
    try:
        parsed = int(str(value).strip())
    except ValueError:
        return None
    return parsed if parsed > 0 else None


def print_json(value: Any, *, indent: int = 2) -> None:
    text = json.dumps(value, ensure_ascii=False, indent=indent)
    for line in text.splitlines():
        print(line)


def print_indented_json(value: Any, *, indent: int = 4) -> None:
    prefix = " " * indent
    text = json.dumps(value, ensure_ascii=False, indent=2)
    for line in text.splitlines():
        print(prefix + line)


def print_wrapped_message(prefix: str, message: str) -> None:
    lines = str(message).splitlines() or [""]
    continuation = " " * visible_width(prefix)
    for idx, line in enumerate(lines):
        print((prefix if idx == 0 else continuation) + line)


def visible_width(text: str) -> int:
    return len(strip_ansi(text))


def _response_message(name: str, parameters: JsonDict) -> str:
    for key in ("message", "summary", "answer", "content", "text"):
        value = parameters.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    question = parameters.get("question")
    if isinstance(question, str) and question.strip():
        return question.strip()
    if name == "task_complete":
        success = parameters.get("success")
        if isinstance(success, bool):
            return "任务已完成。" if success else "任务未完成。"
    if parameters:
        return _compact_json(parameters, max_chars=500)
    return ""


def compact_usage(usage: JsonDict) -> str:
    parts: list[str] = []
    for key in ("prompt_tokens", "completion_tokens", "cached_tokens", "total_tokens"):
        if key in usage:
            parts.append(f"{key}={usage[key]}")
    if not parts:
        return json.dumps(usage, ensure_ascii=False)
    return ", ".join(parts)


def _prompt_html(text: str) -> HTML:
    return HTML(f"<prompt>{html.escape(strip_ansi(text))}</prompt>")


def _cancel_active_completion(buffer: Any) -> bool:
    if getattr(buffer, "complete_state", None) is None:
        return False
    buffer.cancel_completion()
    return True


def _compact_json(value: Any, *, max_chars: int) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, sort_keys=True)
    except TypeError:
        text = str(value)
    return _truncate(_single_line(text), max_chars)


def _compact_tool_parameters(name: str, value: Any, *, max_chars: int) -> str:
    if not isinstance(value, dict):
        return _compact_json(value, max_chars=max_chars)
    if not value:
        return ""

    if name == "python_exec":
        parts: list[str] = []
        sandbox = value.get("sandbox")
        if sandbox:
            parts.append(f"sandbox={sandbox}")
        code = str(value.get("code", "") or "").strip()
        if code:
            parts.append(f"code: {_single_line(code.splitlines()[0])}")
        if parts:
            return _truncate("  ".join(parts), max_chars)

    pieces = []
    for key, raw in value.items():
        if _is_empty_param_value(raw):
            continue
        if isinstance(raw, (str, int, float, bool)):
            rendered = str(raw)
        else:
            rendered = _compact_json(raw, max_chars=max_chars)
        pieces.append(f"{key}={rendered}")
    return _truncate("  ".join(pieces), max_chars)


def _compact_tool_result_lines(
    name: str,
    value: Any,
    *,
    max_lines: int = TOOL_RESULT_MAX_LINES,
    line_width: int | None = None,
) -> list[str]:
    """Build a small display-only projection without mutating the tool result."""

    if max_lines <= 0:
        return []
    width = max(12, int(line_width or _tool_detail_width()))
    if not isinstance(value, dict):
        return [_truncate_display_width(_compact_json(value, max_chars=width * 2), width)]

    candidates: list[str] = []
    content = _single_line(str(value.get("content") or ""))
    if content:
        _append_summary_line(candidates, content)

    details = value.get("details")
    if not isinstance(details, dict):
        details = {}
    diagnostics = details.get("diagnostics")
    if isinstance(diagnostics, list):
        for diagnostic in diagnostics[:2]:
            if not isinstance(diagnostic, dict):
                continue
            code = str(diagnostic.get("code") or "diagnostic")
            message = _single_line(str(diagnostic.get("message") or ""))
            rendered = f"{code}: {message}" if message else code
            _append_summary_line(candidates, rendered)

    artifacts = details.get("artifacts")
    if isinstance(artifacts, list) and artifacts:
        paths = _artifact_summary_paths(artifacts, limit=2)
        artifact_line = f"artifacts={len(artifacts)}"
        if paths:
            artifact_line += f"  paths={', '.join(paths)}"
        _append_summary_line(candidates, artifact_line)

    outputs = details.get("outputs")
    if isinstance(outputs, dict):
        for line in _tool_output_summary_lines(outputs, width=width):
            _append_summary_line(candidates, line)

    state_delta = details.get("state_delta")
    state_projection = _project_mapping(
        state_delta,
        ("reward", "terminated", "truncated", "motion", "simulator_environment"),
    )
    if state_projection:
        _append_summary_line(
            candidates,
            f"state {_compact_json(state_projection, max_chars=width * 2)}",
        )

    if not candidates:
        _append_summary_line(
            candidates,
            f"{name} {_compact_json(value, max_chars=width * 2)}".strip(),
        )

    full_chars = len(_json_text(value))
    visible_chars = sum(len(line) for line in candidates)
    display_capacity = width * max_lines
    line_truncated = any(get_cwidth(_single_line(line)) > width for line in candidates)
    omitted = (
        len(candidates) > max_lines
        or line_truncated
        or full_chars > max(visible_chars + 160, display_capacity)
    )
    if omitted:
        marker = "… full result retained in session trace"
        visible = candidates[: max(0, max_lines - 1)]
        visible.append(marker)
    else:
        visible = candidates[:max_lines]
    return [_truncate_display_width(_single_line(line), width) for line in visible]


def _tool_output_summary_lines(outputs: JsonDict, *, width: int) -> list[str]:
    lines: list[str] = []
    scalar_projection = _project_mapping(
        outputs,
        (
            "result_id",
            "detection_count",
            "candidate_count",
            "grasp_count",
            "selection_required",
            "sandbox",
            "executor",
            "returncode",
        ),
    )
    if scalar_projection:
        lines.append(_compact_json(scalar_projection, max_chars=width * 2))

    for key, fields in (
        (
            "environment",
            ("env_id", "handle", "session_id", "dashboard_url", "mcp_server_url"),
        ),
        (
            "selected_detection",
            ("id", "rank", "score", "label", "mask_ref", "overlay_ref"),
        ),
        (
            "active_grasp_candidate",
            ("id", "rank", "score", "frame", "camera_frame", "result_id"),
        ),
        (
            "best_grasp_candidate",
            ("id", "rank", "score", "frame", "camera_frame", "result_id"),
        ),
        (
            "world_pose",
            ("id", "source_grasp_id", "frame", "xyz", "translation_xyz"),
        ),
        (
            "motion_summary",
            ("reached_target", "collision", "position_error", "orientation_error"),
        ),
    ):
        projection = _project_mapping(outputs.get(key), fields)
        if projection:
            lines.append(f"{key} {_compact_json(projection, max_chars=width * 2)}")

    result = outputs.get("result")
    if result is not None:
        lines.append(f"result {_compact_json(result, max_chars=width * 2)}")

    response = outputs.get("response")
    response_projection = _project_mapping(
        response,
        (
            "success",
            "ok",
            "error",
            "handle",
            "session_id",
            "env_id",
            "dashboard_url",
            "response_path",
        ),
    )
    if response_projection:
        lines.append(f"response {_compact_json(response_projection, max_chars=width * 2)}")

    initial = outputs.get("initial_observation")
    if isinstance(initial, dict):
        initial_projection = _project_mapping(
            initial,
            ("success", "ok", "handle", "session_id", "response_path"),
        )
        cameras = initial.get("cameras")
        if isinstance(cameras, list):
            initial_projection["camera_count"] = len(cameras)
        lines.append(
            f"initial_observation {_compact_json(initial_projection, max_chars=width * 2)}"
        )

    stdout = outputs.get("stdout")
    if isinstance(stdout, str) and stdout.strip():
        lines.append(f"stdout {_single_line(stdout)}")

    if not lines and outputs:
        keys = [str(key) for key in outputs][:12]
        lines.append(f"outputs keys={','.join(keys)}")
    return lines


def _project_mapping(value: Any, keys: Iterable[str]) -> JsonDict:
    if not isinstance(value, dict):
        return {}
    return {
        key: value[key] for key in keys if key in value and not _is_empty_param_value(value[key])
    }


def _artifact_summary_paths(artifacts: list[Any], *, limit: int) -> list[str]:
    paths: list[str] = []
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        for key in ("path", "response_path", "mask_ref", "overlay_ref", "crop_ref"):
            path = artifact.get(key)
            if not isinstance(path, str) or not path or path in paths:
                continue
            paths.append(path)
            if len(paths) >= limit:
                return paths
    return paths


def _append_summary_line(lines: list[str], value: str) -> None:
    rendered = _single_line(str(value or ""))
    if rendered and rendered not in lines:
        lines.append(rendered)


def _json_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


def _tool_detail_width() -> int:
    columns = shutil.get_terminal_size(fallback=(TOOL_RESULT_FALLBACK_WIDTH, 24)).columns
    return max(12, min(180, columns - 12))


def _truncate_display_width(text: str, max_cells: int) -> str:
    if max_cells <= 0:
        return ""
    if get_cwidth(text) <= max_cells:
        return text
    suffix = "…"
    suffix_width = get_cwidth(suffix)
    target = max(0, max_cells - suffix_width)
    rendered: list[str] = []
    used = 0
    for char in text:
        char_width = max(0, get_cwidth(char))
        if used + char_width > target:
            break
        rendered.append(char)
        used += char_width
    return "".join(rendered) + suffix


def _is_empty_param_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) == 0
    return False


def _format_elapsed(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 1:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{seconds:.1f}s" if seconds < 10 else f"{int(round(seconds))}s"
    minutes = int(seconds // 60)
    remaining = int(round(seconds - minutes * 60))
    if remaining == 60:
        minutes += 1
        remaining = 0
    return f"{minutes}m {remaining:02d}s"


def _single_line(text: str) -> str:
    return " ".join(text.split())


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)] + "…"


def print_box_line(text: str, *, width: int, strong: bool = False) -> None:
    content_width = width - 4
    clean = text[:content_width]
    if strong:
        clean = Theme.bold(clean)
    padding = max(0, content_width - len(strip_ansi(text[:content_width])))
    print(Theme.accent("│ ") + clean + (" " * padding) + Theme.accent(" │"))


def strip_ansi(text: str) -> str:
    result = []
    in_escape = False
    for char in text:
        if char == "\033":
            in_escape = True
            continue
        if in_escape:
            if char == "m":
                in_escape = False
            continue
        result.append(char)
    return "".join(result)


def main(argv: list[str] | None = None) -> int:
    command_parser = argparse.ArgumentParser(add_help=False)
    command_parser.add_argument(
        "--command",
        choices=("preflight", "run", "iterate", "inspect"),
        default="",
    )
    command_args, remaining = command_parser.parse_known_args(argv)
    if command_args.command:
        from agent.cli.experiment import main as experiment_main

        return experiment_main([command_args.command, *remaining])
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="", help="Override configured model for this run.")
    parser.add_argument(
        "--calibration-profile",
        default="",
        help="Stage this grasp calibration profile in the TUI session workspace.",
    )
    parser.add_argument("--once", default="", help="Run one task and exit.")
    parser.add_argument(
        "--max-turns",
        type=int,
        default=DEFAULT_MAX_TURNS,
        help="Safety limit for one episode run.",
    )
    args = parser.parse_args(argv)

    cli = OpenEtaCli(
        model_override=args.model,
        calibration_profile=args.calibration_profile,
    )
    try:
        if args.once:
            cli.run_task(
                args.once,
                max_turns=max(1, args.max_turns),
                raise_on_interrupt=True,
            )
            return 0
        cli.run()
        return 0
    except KeyboardInterrupt:
        print()
        print(Theme.warn("interrupted by Ctrl-C"))
        return 130
    finally:
        try:
            cli.close()
        except Exception as exc:  # noqa: BLE001 - preserve the original CLI exit result.
            print(Theme.warn(f"unexpected MCP cleanup failure: {exc}"))


if __name__ == "__main__":
    raise SystemExit(main())
