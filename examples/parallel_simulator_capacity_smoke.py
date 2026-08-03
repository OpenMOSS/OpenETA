"""Create/reset/close simulator environments concurrently without model actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from agent.backends.planner import StaticPlannerBackend
from agent.cli.batch_eval import load_parallel_episode_manifest
from agent.runtime.episode import OpenEtaEpisodeRunner
from agent.runtime.parallel import ParallelEpisodeHarness, ParallelEpisodeWorker
from agent.runtime.planner import ToolCallingPlanner
from agent.runtime.runtime import OpenEtaAgentRuntime
from agent.tools.mcp_registry import load_mcp_server_url
from agent.tools.sim_mcp import (
    SimulatorMcpEpisodeConfig,
    SimulatorMcpEpisodeEnvironment,
    SseSimulatorMcpTransport,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Capacity smoke: concurrently create/reset/close manifest environments "
            "without asking a model to move the robot."
        )
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument("--sim-url", default="")
    parser.add_argument("--batch-id", default="capacity-smoke")
    args = parser.parse_args()

    specs = load_parallel_episode_manifest(args.manifest)
    sim_url = args.sim_url or load_mcp_server_url("openeta-sim", aliases=("sim",))
    if not sim_url:
        parser.error("simulator MCP URL is required")

    def factory(spec, batch_id):
        environment = SimulatorMcpEpisodeEnvironment(
            transport=SseSimulatorMcpTransport(sim_url),
            config=SimulatorMcpEpisodeConfig(
                env_id=spec.env_id,
                seed=spec.seed,
                image_output_root=(
                    Path("tmp") / "parallel_eval" / batch_id / spec.episode_id / "images"
                ),
            ),
        )
        runtime = OpenEtaAgentRuntime(
            planner=ToolCallingPlanner(
                StaticPlannerBackend(
                    {
                        "kind": "response",
                        "name": "task_complete",
                        "parameters": {
                            "success": True,
                            "summary": "capacity smoke reset completed",
                        },
                    }
                )
            )
        )
        return ParallelEpisodeWorker(
            runner=OpenEtaEpisodeRunner(runtime=runtime, environment=environment),
            close=environment.close,
        )

    result = ParallelEpisodeHarness(factory, concurrency=args.concurrency).run(
        specs,
        batch_id=args.batch_id,
    )
    summary = {
        "batch_id": result.batch_id,
        "concurrency": result.concurrency,
        "episode_count": len(result.outcomes),
        "success_count": result.success_count,
        "need_human_count": result.need_human_count,
        "fail_count": result.fail_count,
        "duration_s": round(result.duration_s, 3),
        "outcomes": [
            {
                "episode_id": outcome.spec.episode_id,
                "env_id": outcome.spec.env_id,
                "status": outcome.status,
                "duration_s": round(outcome.duration_s, 3),
                "cleanup": outcome.cleanup,
                "error": outcome.error,
            }
            for outcome in result.outcomes
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if result.fail_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
