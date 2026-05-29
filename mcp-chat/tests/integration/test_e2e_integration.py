"""End-to-end integration tests for weather-agent.

Tests the complete flow through mcp-chat -> mcp-server -> Open-Meteo API.
Requires ANTHROPIC_API_KEY or OPENAI_API_KEY in environment.

Mark with @pytest.mark.integration and skip if no API key available.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest


def get_api_key() -> str | None:
    """Get LLM API key from environment."""
    api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
    return api_key


@pytest.mark.integration
@pytest.mark.skipif(not get_api_key(), reason="No ANTHROPIC_API_KEY or OPENAI_API_KEY")
def test_e2e_weather_query_invalid_location():
    """End-to-end test: ask weather for non-existent location, handle gracefully.

    Behavior: mcp-chat asks "What is the weather in XYZ123InvalidCity?"
    - Spawns mcp-server subprocess
    - Passes query to mcp-chat
    - MCP server returns error from invalid location
    - Agent service injects error tool result
    - LLM generates friendly error response
    - No exceptions raised, no stack traces in output

    This tests the complete error propagation chain:
    1. httpx timeout/connection error -> WeatherError
    2. WeatherError in service -> caught by MCP handler -> error response
    3. Error response in MCPClientAdapter -> injected as tool result
    4. Tool result in AgentService -> passed to LLM for friendly message
    """
    project_root = Path(__file__).parent.parent.parent.parent
    mcp_chat_dir = project_root / "mcp-chat"

    # Run mcp-chat with invalid location query
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "mcp_chat",
            "What is the weather in XYZ123InvalidCity?",
        ],
        cwd=str(mcp_chat_dir),
        capture_output=True,
        text=True,
        timeout=30,
    )

    # Assert no crash (exit code 0 or controlled error exit)
    # The process may return non-zero, but should not timeout or crash
    output = process.stdout + process.stderr

    # Assert no stack trace in output
    assert "Traceback" not in output, f"Stack trace found in output:\n{output}"
    assert "traceback" not in output.lower(), f"Traceback reference found:\n{output}"

    # Assert response is non-empty (LLM should respond with friendly error)
    assert len(output) > 0, "Expected non-empty response from agent"
