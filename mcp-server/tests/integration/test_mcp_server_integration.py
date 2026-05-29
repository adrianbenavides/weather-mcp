"""Integration test for MCP server using subprocess and real Open-Meteo API.

This test validates the full MCP server stack:
- Subprocess spawned fresh per test
- Stdio transport working
- get_current_weather tool exposed and callable
- Real Open-Meteo API integration
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.integration
def test_mcp_server_get_current_weather_london():
    """Test get_current_weather tool via subprocess with real Open-Meteo API.

    Validates:
    - Server starts via `uv run python -m mcp_server`
    - JSON-RPC initialize and tools/call protocol works
    - get_current_weather returns WeatherData with all required fields non-null
    """
    # Spawn fresh subprocess
    mcp_server_dir = Path(__file__).parent.parent.parent
    process = subprocess.Popen(
        [sys.executable, "-m", "mcp_server"],
        cwd=str(mcp_server_dir),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        # Initialize MCP session
        init_message = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "clientInfo": {
                    "name": "test-client",
                    "version": "1.0",
                },
                "capabilities": {},
            },
        }

        process.stdin.write(json.dumps(init_message) + "\n")
        process.stdin.flush()

        # Read initialize response
        response_line = process.stdout.readline()
        init_response = json.loads(response_line)
        assert init_response["id"] == 1
        assert "result" in init_response

        # List tools
        list_tools_message = {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/list",
            "params": {},
        }

        process.stdin.write(json.dumps(list_tools_message) + "\n")
        process.stdin.flush()

        response_line = process.stdout.readline()
        list_tools_response = json.loads(response_line)
        assert list_tools_response["id"] == 2
        assert "result" in list_tools_response

        tools = list_tools_response["result"].get("tools", [])
        tool_names = [tool["name"] for tool in tools]
        assert "get_current_weather" in tool_names, f"Expected get_current_weather in {tool_names}"

        # Call get_current_weather for London
        call_message = {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "get_current_weather",
                "arguments": {"location": "London"},
            },
        }

        process.stdin.write(json.dumps(call_message) + "\n")
        process.stdin.flush()

        response_line = process.stdout.readline()
        call_response = json.loads(response_line)
        assert call_response["id"] == 3
        assert "result" in call_response

        # Parse WeatherData response
        result = call_response["result"]
        assert "content" in result

        content = result["content"]
        assert len(content) > 0

        weather_text = content[0].get("text", "{}")
        weather_data = json.loads(weather_text)

        # Verify all required fields are present and non-null
        assert weather_data.get("temperature_c") is not None
        assert weather_data.get("wind_speed_kmh") is not None
        assert weather_data.get("conditions") is not None
        assert weather_data.get("location_name") is not None

        # Verify data types
        assert isinstance(weather_data["temperature_c"], (int, float))
        assert isinstance(weather_data["wind_speed_kmh"], (int, float))
        assert isinstance(weather_data["conditions"], str)
        assert isinstance(weather_data["location_name"], str)

    finally:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
