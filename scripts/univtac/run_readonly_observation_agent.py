#!/usr/bin/env python3
"""Run one staged read-only observation request in an isolated process."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context-root", type=Path, required=True)
    args = parser.parse_args(argv)

    from agent.backends.readonly_observation import (
        LOCAL_DUMMY_API_KEY,
        OpenAICompatibleReadOnlyObservationBackend,
    )
    from agent.runtime.observation_input import load_staged_observation_context
    from agent.runtime.observation_runtime import OpenEtaReadOnlyObservationRuntime

    base_url = os.environ["OPENETA_READONLY_BASE_URL"]
    model = os.environ["OPENETA_READONLY_MODEL"]
    api_key = os.environ["OPENETA_READONLY_DUMMY_KEY"]
    if api_key != LOCAL_DUMMY_API_KEY:
        raise RuntimeError("read-only subprocess requires the fixed local dummy key")
    context = load_staged_observation_context(args.context_root)
    backend = OpenAICompatibleReadOnlyObservationBackend(
        base_url=base_url,
        model=model,
        api_key=api_key,
    )
    runtime = OpenEtaReadOnlyObservationRuntime(backend)
    report = runtime.observe(context)
    forbidden_modules = (
        "adapter.bridge",
        "agent.runtime.runtime",
        "agent.runtime.planner",
        "agent.backends.planner",
    )
    payload = {
        "report": report.to_dict(),
        "counters": {
            "agent_observe_call_count": runtime.observe_call_count,
            "agent_act_call_count": 0,
            "planner_decide_call_count": 0,
            "tool_call_count": 0,
            "action_compile_count": 0,
            "env_action_count": 0,
        },
        "module_import_state": {name: name in sys.modules for name in forbidden_modules},
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
