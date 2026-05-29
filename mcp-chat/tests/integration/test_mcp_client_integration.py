"""Integration tests for MCP client adapter with stdio transport."""

from pathlib import Path

import pytest
from mcp_chat.adapters.mcp_client import MCPClientAdapter
from mcp_chat.adapters.transport.stdio_transport import StdioMCPTransport
from mcp_chat.domain.conversation import ToolSchema


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_tools_returns_get_current_weather() -> None:
    """Integration test: list_tools() returns tools including get_current_weather."""
    project_root = Path(__file__).parent.parent.parent.parent
    transport = StdioMCPTransport(project_root=project_root)
    client = MCPClientAdapter(transport=transport)

    await transport.connect()
    try:
        tools = await client.list_tools()
        tool_names = [tool.name for tool in tools]
        assert "get_current_weather" in tool_names
        # Verify tool has expected structure
        weather_tool = next(t for t in tools if t.name == "get_current_weather")
        assert isinstance(weather_tool, ToolSchema)
        assert weather_tool.description
        assert weather_tool.input_schema
    finally:
        await transport.disconnect()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_call_tool_get_current_weather_returns_weather_data() -> None:
    """Integration test: call_tool() invokes get_current_weather and returns valid result."""
    project_root = Path(__file__).parent.parent.parent.parent
    transport = StdioMCPTransport(project_root=project_root)
    client = MCPClientAdapter(transport=transport)

    await transport.connect()
    try:
        result = await client.call_tool("get_current_weather", args={"location": "Paris"})
        assert isinstance(result, str)
        # Verify result contains expected weather data fields (basic sanity check)
        result_lower = result.lower()
        assert "temp" in result_lower or "weather" in result_lower or "paris" in result_lower
    finally:
        await transport.disconnect()
