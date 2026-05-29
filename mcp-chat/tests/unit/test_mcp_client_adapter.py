"""Unit tests for MCPClientAdapter."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp_chat.adapters.mcp_client import MCPClientAdapter
from mcp_chat.domain.conversation import ToolSchema
from mcp_chat.ports.mcp_transport import MCPTransportPort


class TestMCPClientAdapter:
    """Unit tests for MCPClientAdapter - test through MCPClientPort interface."""

    @pytest.mark.asyncio
    async def test_list_tools_maps_mcp_tools_to_tool_schema(self) -> None:
        """list_tools() maps MCP tool definitions to ToolSchema domain models."""
        # Setup: mock ClientSession with tool definitions
        from types import SimpleNamespace

        tool1 = SimpleNamespace(
            name="get_current_weather",
            description="Get current weather for a location",
            inputSchema={
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        )
        tool2 = SimpleNamespace(
            name="get_forecast",
            description="Get weather forecast",
            inputSchema={
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        )

        mock_session = MagicMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=[tool1, tool2]))

        mock_transport = MagicMock(spec=MCPTransportPort)
        mock_transport.session = mock_session

        adapter = MCPClientAdapter(transport=mock_transport)

        # Act
        tools = await adapter.list_tools()

        # Assert: tools are ToolSchema instances with correct data
        assert len(tools) == 2
        assert all(isinstance(t, ToolSchema) for t in tools)
        assert tools[0].name == "get_current_weather"
        assert tools[0].description == "Get current weather for a location"
        assert tools[0].input_schema["type"] == "object"
        assert tools[1].name == "get_forecast"

    @pytest.mark.asyncio
    async def test_call_tool_invokes_mcp_session_call_tool(self) -> None:
        """call_tool() passes tool name and args to ClientSession.call_tool()."""
        # Setup: mock ClientSession.call_tool
        from types import SimpleNamespace

        content = SimpleNamespace(text="Paris: 15C, sunny")
        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(return_value=SimpleNamespace(content=[content]))

        mock_transport = MagicMock(spec=MCPTransportPort)
        mock_transport.session = mock_session

        adapter = MCPClientAdapter(transport=mock_transport)

        # Act
        result = await adapter.call_tool("get_current_weather", args={"location": "Paris"})

        # Assert: result is string with expected content
        assert isinstance(result, str)
        assert "Paris" in result
        assert "15C" in result

    @pytest.mark.asyncio
    async def test_call_tool_returns_string_result(self) -> None:
        """call_tool() returns result as string."""
        from types import SimpleNamespace

        content = SimpleNamespace(text="Weather data")
        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(return_value=SimpleNamespace(content=[content]))

        mock_transport = MagicMock(spec=MCPTransportPort)
        mock_transport.session = mock_session

        adapter = MCPClientAdapter(transport=mock_transport)

        result = await adapter.call_tool("get_current_weather", args={"location": "London"})

        assert result == "Weather data"

    @pytest.mark.parametrize(
        "location,expected_text",
        [
            ("Paris", "Paris weather"),
            ("London", "London conditions"),
            ("Tokyo", "Tokyo forecast"),
        ],
    )
    @pytest.mark.asyncio
    async def test_call_tool_handles_different_locations(self, location: str, expected_text: str) -> None:
        """call_tool() handles various location arguments."""
        from types import SimpleNamespace

        content = SimpleNamespace(text=expected_text)
        mock_session = MagicMock()
        mock_session.call_tool = AsyncMock(return_value=SimpleNamespace(content=[content]))

        mock_transport = MagicMock(spec=MCPTransportPort)
        mock_transport.session = mock_session

        adapter = MCPClientAdapter(transport=mock_transport)

        result = await adapter.call_tool("get_current_weather", args={"location": location})

        assert result == expected_text
