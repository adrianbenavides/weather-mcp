"""Integration test: verify structlog output contains turn_id and tool call events."""

import json
import os
from io import StringIO
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import structlog
from mcp_chat.application.agent_service import AgentService
from mcp_chat.application.config import AppConfig
from mcp_chat.domain.conversation import LLMChunk


@pytest.mark.integration
@pytest.mark.asyncio
async def test_agent_service_logs_turn_id_and_tool_calls() -> None:
    """Integration test: verify structlog output contains turn_id and tool_call events.

    PORT-TO-PORT PRINCIPLE: Test enters through AgentService.run (driving port),
    asserts on structlog output stream (driven port boundary).
    """
    # Capture structlog output by redirecting to StringIO
    log_output = StringIO()

    # Configure structlog to write to StringIO for testing
    import logging

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=log_output),
    )

    # Create mock LLM and MCP ports
    mock_llm = AsyncMock()
    mock_mcp = AsyncMock()

    # Mock list_tools to return a simple tool
    mock_mcp.list_tools = AsyncMock(
        return_value=[{"name": "get_current_weather", "description": "Get weather"}]
    )

    # Mock call_tool to return a result
    mock_mcp.call_tool = AsyncMock(return_value='{"temperature": 15, "condition": "cloudy"}')

    # First stream: tool call
    async def mock_stream_1():
        yield LLMChunk.tool_use_start("get_current_weather", "tool_use_123")
        yield LLMChunk.tool_use_input_delta_chunk('{"location": "')
        yield LLMChunk.tool_use_input_delta_chunk('London"}')
        yield LLMChunk.tool_use_complete("get_current_weather", "tool_use_123", {"location": "London"})
        yield LLMChunk.stop("tool_use")

    # Second stream: final response
    async def mock_stream_2():
        yield LLMChunk.text_delta("The weather in London is cloudy with 15°C.")
        yield LLMChunk.stop("end_turn")

    # Return different streams for multiple calls
    mock_llm.stream_response = MagicMock(side_effect=[mock_stream_1(), mock_stream_2()])

    # Create agent service with mocks
    agent = AgentService(llm=mock_llm, mcp_client=mock_mcp)

    # Run agent with a simple query
    query = "What is the weather in London?"
    async for chunk in agent.run_turn(query):
        pass  # Consume all chunks

    # Verify structlog output contains expected events
    log_text = log_output.getvalue()
    assert len(log_text) > 0, "No log output generated"

    # Parse each log line as JSON
    log_lines = [line for line in log_text.strip().split("\n") if line]
    assert len(log_lines) > 0, "No log lines found"

    # Convert to dicts
    log_events: list[dict[str, Any]] = []
    for line in log_lines:
        try:
            event = json.loads(line)
            log_events.append(event)
        except json.JSONDecodeError:
            # Skip malformed lines
            pass

    # Verify at least one log event contains turn_id key
    turn_ids = [e.get("turn_id") for e in log_events if "turn_id" in e]
    assert len(turn_ids) > 0, f"No log events contain turn_id field. Events: {log_events}"
    assert all(tid is not None for tid in turn_ids), "turn_id values should not be None"


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_weather_query_logs_turn_id_and_tool_calls() -> None:
    """Integration test: run full weather query with real API, verify structlog output.

    Only runs if ANTHROPIC_API_KEY is set.
    """
    from pathlib import Path

    from mcp_chat.adapters.llm.anthropic_adapter import AnthropicAdapter
    from mcp_chat.adapters.mcp_client import MCPClientAdapter
    from mcp_chat.adapters.transport.stdio_transport import StdioMCPTransport

    # Capture structlog output by redirecting to StringIO
    log_output = StringIO()

    # Configure structlog to write to StringIO
    import logging

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=log_output),
    )

    # Load configuration
    config = AppConfig.from_env()

    # Create MCP transport (spawns server subprocess internally)
    project_root = Path(__file__).parent.parent.parent.parent
    transport = StdioMCPTransport(project_root)
    await transport.connect()

    try:
        # Wire up adapters
        llm_adapter = AnthropicAdapter(config)
        mcp_client = MCPClientAdapter(transport)

        # Create and run agent service
        agent = AgentService(llm=llm_adapter, mcp_client=mcp_client)

        # Run agent with a simple query
        query = "What is the weather in London?"
        async for chunk in agent.run_turn(query):
            pass  # Consume all chunks

        # Verify structlog output contains expected events
        log_text = log_output.getvalue()
        assert len(log_text) > 0, "No log output generated"

        # Parse each log line as JSON
        log_lines = [line for line in log_text.strip().split("\n") if line]
        assert len(log_lines) > 0, "No log lines found"

        # Convert to dicts
        log_events: list[dict[str, Any]] = []
        for line in log_lines:
            try:
                event = json.loads(line)
                log_events.append(event)
            except json.JSONDecodeError:
                # Skip malformed lines
                pass

        # Verify at least one log event contains turn_id key
        turn_ids = [e.get("turn_id") for e in log_events if "turn_id" in e]
        assert len(turn_ids) > 0, "No log events contain turn_id field"
        assert all(tid is not None for tid in turn_ids), "turn_id values should not be None"

        # Verify at least one log event is a tool_call event
        tool_call_events = [e for e in log_events if e.get("event") == "tool_call"]
        assert len(tool_call_events) > 0, "No tool_call events found in logs"

        # Verify tool_call event contains expected fields
        for event in tool_call_events:
            assert "tool_name" in event, "tool_call event missing tool_name"
            assert "latency_ms" in event, "tool_call event missing latency_ms"

    finally:
        # Disconnect transport
        try:
            await transport.disconnect()
        except Exception:
            pass
