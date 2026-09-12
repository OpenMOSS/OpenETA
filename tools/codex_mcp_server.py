"""Native MCP transport for the experimental Codex/OpenETA ingress."""
import asyncio
import json
import os
import sys
import signal
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from tools.codex_host import build_host, parser


def server_for(host):
    server = Server("openeta", instructions="Use native tools, not XML. Host owns scene evidence, action admission and episode termination. Call episode_status first. Only reported tools are available. Auxiliary model services are disabled. Inspect images and fresh evidence after motion.")

    @server.list_tools()
    async def list_tools():
        return list(host.schemas.values())

    @server.call_tool(validate_input=False)
    async def call_tool(name, arguments):
        return await asyncio.to_thread(host.call, name, arguments)

    return server


async def serve(host):
    server = server_for(host)
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    loop.add_signal_handler(signal.SIGTERM, task.cancel)
    try:
        async with stdio_server() as (read, write):
            await server.run(read, write, server.create_initialization_options())
    finally:
        await asyncio.to_thread(host.close)


def main():
    argv = sys.argv[1:] or json.loads(os.environ.get("OPENETA_CODEX_HOST_ARGS", "[]"))
    host = build_host(parser().parse_args(argv))
    try:
        asyncio.run(serve(host))
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        host.close()


if __name__ == "__main__":
    main()
