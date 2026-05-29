"""mcp-chat command-line entry point.

Usage:
    python -m mcp_chat "What is the weather in London?"
    echo "What is the weather in London?" | python -m mcp_chat -

Wires up MCP transport, LLM adapters, and streams response through CLI renderer.
"""

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
from mcp_chat.transport.cli import render_to_cli


async def main() -> None:
    """Main entry point for mcp-chat CLI."""
    # Configure logging (JSON by default, console if LOG_FORMAT=console)
    log_format = os.getenv("LOG_FORMAT", "json")
    configure_logging(log_format=log_format)

    # Parse command-line arguments
    if len(sys.argv) < 2:
        print("Usage: python -m mcp_chat <query>", file=sys.stderr)
        sys.exit(1)

    query_arg = sys.argv[1]

    # Read query from stdin if "-" is passed
    if query_arg == "-":
        query = sys.stdin.read().strip()
    else:
        query = query_arg

    if not query:
        print("Error: empty query", file=sys.stderr)
        sys.exit(1)

    # Load configuration - validate API keys at startup
    try:
        config = AppConfig.from_env()
    except ValueError as e:
        print(f"Configuration error: {str(e)}", file=sys.stderr)
        sys.exit(1)

    # Get project root for MCP transport
    project_root = Path(__file__).parent.parent.parent.parent

    # Create MCP transport (spawns server subprocess internally)
    transport = StdioMCPTransport(project_root)
    await transport.connect()

    try:
        # Wire up adapters
        llm_adapter = AnthropicAdapter(config)
        mcp_client = MCPClientAdapter(transport)

        # Create and run agent service
        agent = AgentService(llm=llm_adapter, mcp_client=mcp_client)

        # Stream response through CLI renderer
        await render_to_cli(agent.run(query))

    finally:
        # Disconnect transport
        try:
            await transport.disconnect()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
