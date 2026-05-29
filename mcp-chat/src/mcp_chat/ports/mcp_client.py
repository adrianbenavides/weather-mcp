"""MCP client port - driven port for MCP tool execution."""

from typing import Any, Protocol

from mcp_chat.domain.conversation import ToolSchema


class MCPClientPort(Protocol):
    """Port for communicating with MCP server to list and invoke tools."""

    async def call_tool(self, name: str, args: dict[str, Any]) -> str:
        """Invoke a tool on the MCP server.

        Args:
            name: Tool name.
            args: Tool arguments.

        Returns:
            Tool result as string.
        """
        ...

    async def list_tools(self) -> list[ToolSchema]:
        """List all available tools from MCP server.

        Returns:
            List of tool definitions.
        """
        ...
