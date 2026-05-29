"""MCP client adapter implementation."""

from typing import Any

from mcp_chat.domain.conversation import ToolSchema
from mcp_chat.ports.mcp_client import MCPClientPort
from mcp_chat.ports.mcp_transport import MCPTransportPort


class MCPClientAdapter(MCPClientPort):
    """Adapter for MCP client, translating to/from domain models.

    Uses MCPTransportPort to communicate with MCP server.
    Maps MCP types to domain ToolSchema.
    """

    def __init__(self, transport: MCPTransportPort) -> None:
        """Initialize adapter with transport.

        Args:
            transport: MCPTransportPort for server communication.
        """
        self.transport = transport

    async def list_tools(self) -> list[ToolSchema]:
        """List all available tools from MCP server.

        Returns:
            List of ToolSchema domain models.
        """
        session = self.transport.session
        if session is None:
            return []

        result = await session.list_tools()

        tools = []
        for tool in result.tools:
            tool_schema = ToolSchema(
                name=tool.name,
                description=tool.description,
                input_schema=tool.inputSchema,
            )
            tools.append(tool_schema)

        return tools

    async def call_tool(self, name: str, args: dict[str, Any]) -> str:
        """Invoke a tool on the MCP server.

        Args:
            name: Tool name.
            args: Tool arguments.

        Returns:
            Tool result as string.
        """
        session = self.transport.session
        if session is None:
            return ""

        result = await session.call_tool(name, args)
        return self._extract_text_from_result(result)

    def _extract_text_from_result(self, result: Any) -> str:
        """Extract text content from tool result.

        Args:
            result: MCP tool result with content field.

        Returns:
            Concatenated text from all content blocks.
        """
        if not result.content:
            return ""

        text_parts = []
        for content in result.content:
            if hasattr(content, "text"):
                text_parts.append(content.text)
        return "".join(text_parts)
