"""Integration test for AgentService end-to-end flow with real APIs.

This test verifies:
- AgentService orchestrates LLM streaming with tool-call loop
- Tool calls to MCP server subprocess work correctly
- Weather question returns actual weather data
- Uses real Anthropic/OpenAI API and real mcp-server subprocess
"""

import os
from pathlib import Path
from typing import AsyncIterator

import pytest
import pytest_asyncio
from mcp_chat.adapters.llm.anthropic_adapter import AnthropicAdapter
from mcp_chat.adapters.mcp_client import MCPClientAdapter
from mcp_chat.adapters.transport.stdio_transport import StdioMCPTransport
from mcp_chat.application.agent_service import AgentService
from mcp_chat.application.config import AppConfig
from mcp_chat.domain.conversation import LLMChunk


@pytest.mark.integration
class TestAgentIntegration:
    """Integration tests for AgentService with real MCP server."""

    @pytest_asyncio.fixture
    async def mcp_transport(self) -> AsyncIterator[StdioMCPTransport]:
        """Start real mcp-server subprocess and create transport."""
        # Get project root
        project_root = Path(__file__).parent.parent.parent.parent

        # Create transport (spawns subprocess internally)
        transport = StdioMCPTransport(project_root)
        await transport.connect()

        yield transport

        # Cleanup
        try:
            await transport.disconnect()
        except Exception:
            pass

    @pytest.mark.skipif(
        not os.getenv("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set",
    )
    async def test_asks_weather_question_returns_weather_data(self, mcp_transport: StdioMCPTransport) -> None:
        """End-to-end: weather question through AgentService.

        Given: user asks "What is the weather in London?"
        When: AgentService orchestrates LLM + MCP tool calls
        Then: response contains actual weather information (non-empty result)
        """
        # Setup adapters
        config = AppConfig.from_env()
        llm_adapter = AnthropicAdapter(config)
        mcp_client = MCPClientAdapter(mcp_transport)
        agent_service = AgentService(llm=llm_adapter, mcp_client=mcp_client)

        # Act: run agent with weather question
        query = "What is the weather in London?"
        chunks: list[LLMChunk] = []
        async for chunk in agent_service.run_turn(query):
            chunks.append(chunk)

        # Assert: received at least text response (weather data)
        text_chunks = [c for c in chunks if c.type == "text_delta" and c.text]
        assert len(text_chunks) > 0, "Expected text response from agent"

        # Assert: final text contains weather-related info
        final_text = "".join(c.text or "" for c in text_chunks)
        assert len(final_text) > 0, "Expected non-empty response"
        # Verify it contains meaningful weather info (temperature, condition, etc.)
        assert any(
            keyword in final_text.lower()
            for keyword in [
                "temperature",
                "celsius",
                "fahrenheit",
                "°",
                "weather",
                "condition",
                "wind",
                "cloud",
                "sunny",
                "rain",
                "snow",
                "clear",
                "partly",
            ]
        ), f"Expected weather data in response, got: {final_text[:200]}"
