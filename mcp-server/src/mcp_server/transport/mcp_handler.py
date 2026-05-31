"""MCP server transport - stdio protocol handler with get_current_weather tool."""

import json
from typing import Any

import mcp.server as mcp_server
import mcp.types as mcp_types

from mcp_server.adapters.geocoding import OpenMeteoGeocodingAdapter
from mcp_server.adapters.weather import OpenMeteoWeatherAdapter
from mcp_server.application.weather_service import WeatherService
from mcp_server.domain.errors import WeatherError


def create_server() -> mcp_server.Server:
    """Create and configure MCP server with get_current_weather tool.

    Returns:
        Configured MCP server instance.
    """
    server = mcp_server.Server("weather-mcp")

    # Initialize adapters and service
    geocoding_adapter = OpenMeteoGeocodingAdapter()
    weather_adapter = OpenMeteoWeatherAdapter()
    weather_service = WeatherService(
        geocoding=geocoding_adapter,
        weather=weather_adapter,
    )

    # Register tools
    @server.list_tools()  # type: ignore[no-untyped-call,untyped-decorator]
    async def list_tools() -> list[mcp_types.Tool]:
        """List available tools."""
        return [
            mcp_types.Tool(
                name="get_current_weather",
                description="Get current weather for a location",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name or location (e.g., 'London', 'Paris')",
                        },
                    },
                    "required": ["location"],
                },
            ),
        ]

    @server.call_tool()  # type: ignore[untyped-decorator]
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[mcp_types.TextContent]:
        """Handle tool calls."""
        if name != "get_current_weather":
            raise ValueError(f"Unknown tool: {name}")

        location = arguments.get("location")
        if not location:
            return [
                mcp_types.TextContent(
                    type="text",
                    text=json.dumps({"error": "location argument required"}),
                )
            ]

        try:
            weather_data = await weather_service.run_weather_query(location)
            return [
                mcp_types.TextContent(
                    type="text",
                    text=json.dumps(weather_data.model_dump()),
                )
            ]
        except WeatherError as e:
            return [
                mcp_types.TextContent(
                    type="text",
                    text=json.dumps({"error": e.message}),
                )
            ]
        except Exception as e:
            return [
                mcp_types.TextContent(
                    type="text",
                    text=json.dumps({"error": f"Unexpected error: {str(e)}"}),
                )
            ]

    return server
