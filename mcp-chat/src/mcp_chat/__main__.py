"""MCP Chat CLI - parse args, wire adapters, run REPL loop."""

import asyncio
import os
import sys
from pathlib import Path

from mcp_chat.adapters.llm.anthropic_adapter import AnthropicAdapter
from mcp_chat.adapters.mcp_client import MCPClientAdapter
from mcp_chat.adapters.transport.stdio_transport import StdioMCPTransport
from mcp_chat.application.agent_service import AgentService
from mcp_chat.application.config import AppConfig
from mcp_chat.application.observability import configure_logging
from mcp_chat.transport.cli import repl_loop


async def main() -> None:
    """Wire adapters, load config, run REPL loop."""
    log_format = os.getenv("LOG_FORMAT", "json")
    configure_logging(log_format=log_format)

    initial_query = None
    if len(sys.argv) > 1:
        query_arg = sys.argv[1]
        if query_arg == "-":
            initial_query = sys.stdin.read().strip()
        else:
            initial_query = query_arg

        if initial_query == "":
            print("Error: empty query", file=sys.stderr)
            sys.exit(1)

    try:
        config = AppConfig.from_env()
    except ValueError as e:
        print(f"Configuration error: {str(e)}", file=sys.stderr)
        sys.exit(1)

    project_root = Path(__file__).parent.parent.parent.parent
    transport = StdioMCPTransport(project_root)
    await transport.connect()

    try:
        llm_adapter = AnthropicAdapter(config)
        mcp_client = MCPClientAdapter(transport)
        agent = AgentService(llm=llm_adapter, mcp_client=mcp_client)
        await repl_loop(agent, initial_query=initial_query)
    finally:
        try:
            await transport.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except SystemExit as e:
        sys.exit(e.code)
