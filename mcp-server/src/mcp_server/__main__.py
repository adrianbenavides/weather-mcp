"""MCP server entry point - starts stdio MCP server."""

import anyio
from mcp.server.stdio import stdio_server

from mcp_server.transport.mcp_handler import create_server


async def main() -> None:
    """Start the stdio MCP server.

    Initializes the weather MCP server and runs it over stdio.
    Process exits cleanly on SIGTERM.
    """
    server = create_server()

    async with stdio_server() as (read_stream, write_stream):
        init_options = server.create_initialization_options()
        await server.run(read_stream, write_stream, init_options)


if __name__ == "__main__":
    anyio.run(main)
