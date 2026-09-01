#!/usr/bin/env python3
"""CLI wrapper for deterministic ToolContract request-readiness evidence."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.evals.tool_contract_readiness import main


if __name__ == "__main__":
    raise SystemExit(main())
