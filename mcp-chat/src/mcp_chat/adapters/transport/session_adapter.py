"""Adapter abstracting MCP ClientSession for tool operations."""

from typing import Any

from mcp.client.session import ClientSession

from mcp_chat.domain.conversation import ToolSchema


class MCPSessionAdapter:
    """Wraps MCP ClientSession to provide domain-specific operations.

    Hides infrastructure ClientSession behind a clean interface.
    """

    def __init__(self, session: ClientSession | None) -> None:
        """Initialize adapter with optional session.

        Args:
            session: MCP ClientSession or None if not connected.
        """
        self._session = session

    def is_connected(self) -> bool:
        """Check if connected.

        Returns:
            True if session is available.
        """
        return self._session is not None

    async def call_tool(self, name: str, args: dict[str, Any]) -> str:
        """Call a tool on the MCP server.

        Args:
            name: Tool name.
            args: Tool arguments.

        Returns:
            Tool result as string or empty if not connected.
        """
        if self._session is None:
            return ""

        result = await self._session.call_tool(name, args)
        return self._extract_text_from_result(result)

    async def list_tools(self) -> list[ToolSchema]:
        """List all available tools.

        Returns:
            List of tool schemas or empty if not connected.
        """
        if self._session is None:
            return []

        result = await self._session.list_tools()

        tools = []
        for tool in result.tools:
            description = tool.description or ""
            tool_schema = ToolSchema(
                name=tool.name,
                description=description,
                input_schema=tool.inputSchema,
            )
            tools.append(tool_schema)

        return tools

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
