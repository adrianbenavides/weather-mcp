"""Unit tests for AgentService tool-call loop orchestration.

This test verifies AgentService:
- Orchestrates LLM streaming with tool definitions
- Detects tool_use chunks and pauses streaming
- Calls MCPClientPort.call_tool() to execute tools
- Injects tool results back into conversation
- Resumes streaming after tool completion
- Yields all chunks to caller (text and tool indicators)

Tests enter through driving port: AgentService.run(user_query: str) -> AsyncIterator[LLMChunk]
"""

from typing import Any

import pytest
from mcp_chat.application.agent_service import AgentService
from mcp_chat.domain.conversation import LLMChunk, ToolSchema


class StubLLMPort:
    """Test double for LLMPort - returns predetermined chunks."""

    def __init__(self, chunks: list[LLMChunk] | list[list[LLMChunk]]) -> None:
        """Initialize with chunks to stream.

        Args:
            chunks: Single stream of LLMChunk, or list of streams for multiple calls.
        """
        # If chunks is a list of lists, treat as multiple streams
        if chunks and isinstance(chunks[0], list):
            self.streams = chunks  # type: ignore
        else:
            self.streams = [chunks]  # type: ignore

        self.call_count = 0
        self.last_conversation: Any = None
        self.last_tools: list[ToolSchema] | None = None

    async def _generate_chunks(self, chunks: list[LLMChunk]) -> Any:
        """Async generator for chunks."""
        for chunk in chunks:
            yield chunk

    def stream_response(self, conversation: Any, tools: list[ToolSchema]) -> Any:
        """Return async generator for chunks."""
        self.last_conversation = conversation
        self.last_tools = tools

        # Return stream at current call_count, or repeat last stream if exhausted
        stream_idx = min(self.call_count, len(self.streams) - 1)
        self.call_count += 1
        return self._generate_chunks(self.streams[stream_idx])


class StubMCPClientPort:
    """Test double for MCPClientPort - returns canned results."""

    def __init__(
        self,
        tools: list[ToolSchema] | None = None,
        tool_results: dict[str, str] | None = None,
    ) -> None:
        """Initialize with available tools and tool result mapping.

        Args:
            tools: List of ToolSchema to return from list_tools().
            tool_results: Map of tool_name -> result_string.
        """
        self.tools = tools or []
        self.tool_results = tool_results or {}
        self.call_count = 0
        self.calls_made: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> list[ToolSchema]:
        """Return predetermined tools."""
        return self.tools

    async def call_tool(self, name: str, args: dict[str, Any]) -> str:
        """Return predetermined result or error."""
        self.call_count += 1
        self.calls_made.append((name, args))

        if name in self.tool_results:
            return self.tool_results[name]
        return f"Tool {name} executed with {args}"


@pytest.mark.asyncio
class TestAgentServiceShould:
    """AgentService behavior."""

    async def test_yields_text_chunks_when_no_tools_called(self) -> None:
        """Given: LLM streams only text (no tool calls)
        When: AgentService.run(query)
        Then: yields all text chunks unchanged
        """
        # Arrange
        text_chunks = [
            LLMChunk.text_delta("The weather in London is "),
            LLMChunk.text_delta("sunny with "),
            LLMChunk.text_delta("15°C."),
            LLMChunk.stop("end_turn"),
        ]
        llm = StubLLMPort(text_chunks)
        mcp = StubMCPClientPort()
        agent = AgentService(llm=llm, mcp_client=mcp)

        # Act
        received_chunks: list[LLMChunk] = []
        async for chunk in agent.run("What is the weather in London?"):
            received_chunks.append(chunk)

        # Assert
        assert len(received_chunks) == 4
        assert received_chunks[0].type == "text_delta"
        assert received_chunks[0].text == "The weather in London is "
        assert received_chunks[3].type == "stop"

    async def test_pauses_on_tool_use_calls_tool_resumes(self) -> None:
        """Given: LLM streams tool_use then text
        When: AgentService detects tool_use_complete
        Then: pauses, calls MCPClientPort.call_tool(), injects result, resumes
        """
        # Arrange: First LLM call emits tool call
        first_stream = [
            LLMChunk.text_delta("I'll check the weather. "),
            LLMChunk.tool_use_start("get_current_weather", "tool_123"),
            LLMChunk.tool_use_complete(
                "get_current_weather",
                "tool_123",
                {"location": "London"},
            ),
            LLMChunk.stop("end_turn"),
        ]

        # Second LLM call (after tool result injected) provides final response
        second_stream = [
            LLMChunk.text_delta("The weather is "),
            LLMChunk.text_delta("sunny."),
            LLMChunk.stop("end_turn"),
        ]

        llm = StubLLMPort([first_stream, second_stream])

        # MCP returns weather data
        weather_data = "Current: Sunny, 15°C, 2m/s wind"
        mcp = StubMCPClientPort(
            tools=[
                ToolSchema(
                    name="get_current_weather",
                    description="Get weather",
                    input_schema={"type": "object"},
                )
            ],
            tool_results={"get_current_weather": weather_data},
        )

        agent = AgentService(llm=llm, mcp_client=mcp)

        # Act
        received_chunks: list[LLMChunk] = []
        async for chunk in agent.run("What is the weather in London?"):
            received_chunks.append(chunk)

        # Assert: tool was called
        assert mcp.call_count == 1
        assert mcp.calls_made[0][0] == "get_current_weather"
        assert mcp.calls_made[0][1] == {"location": "London"}

        # Assert: received all chunks including final response
        assert len(received_chunks) > 0
        text_from_final = "".join(c.text or "" for c in received_chunks if c.type == "text_delta")
        assert "sunny" in text_from_final.lower()

    async def test_injects_tool_result_into_messages_for_continuation(self) -> None:
        """Given: tool is called and returns result
        When: LLM needs to process the result
        Then: tool result message injected into conversation for next LLM call
        """
        # Arrange: first LLM call with tool use
        first_stream = [
            LLMChunk.tool_use_start("get_weather", "id1"),
            LLMChunk.tool_use_complete("get_weather", "id1", {"city": "London"}),
            LLMChunk.stop("end_turn"),
        ]

        # Second LLM call (after tool result) returns final text
        second_stream = [
            LLMChunk.text_delta("Based on the weather data, "),
            LLMChunk.text_delta("it's sunny."),
            LLMChunk.stop("end_turn"),
        ]

        # Create LLM that alternates between first and second stream
        class AlternatingLLM:
            def __init__(self) -> None:
                self.call_num = 0

            async def _generate_chunks(self, chunks: list[LLMChunk]) -> Any:
                """Async generator for chunks."""
                for chunk in chunks:
                    yield chunk

            def stream_response(self, conversation: Any, tools: Any) -> Any:
                """Return async generator for chunks."""
                self.call_num += 1
                chunks = first_stream if self.call_num == 1 else second_stream
                return self._generate_chunks(chunks)

        llm = AlternatingLLM()
        weather_result = "Sunny, 15°C"
        mcp = StubMCPClientPort(
            tools=[
                ToolSchema(
                    name="get_weather",
                    description="Get weather",
                    input_schema={},
                )
            ],
            tool_results={"get_weather": weather_result},
        )

        agent = AgentService(llm=llm, mcp_client=mcp)

        # Act
        chunks: list[LLMChunk] = []
        async for chunk in agent.run("What's the weather?"):
            chunks.append(chunk)

        # Assert: tool was called once
        assert mcp.call_count == 1

        # Assert: second LLM call was made (tool result processed)
        assert llm.call_num == 2, "Expected LLM called twice (tool + continuation)"

        # Assert: final response includes continuation text
        final_text = "".join(c.text or "" for c in chunks if c.type == "text_delta")
        assert "sunny" in final_text.lower()

    async def test_lists_tools_before_llm_call(self) -> None:
        """Given: AgentService.run(query)
        When: starting agent
        Then: lists available tools and passes to LLM
        """
        # Arrange
        tool_schema = ToolSchema(
            name="get_current_weather",
            description="Get current weather",
            input_schema={"type": "object", "properties": {"location": {}}},
        )
        llm = StubLLMPort([LLMChunk.text_delta("Sunny"), LLMChunk.stop("end_turn")])
        mcp = StubMCPClientPort(tools=[tool_schema])

        agent = AgentService(llm=llm, mcp_client=mcp)

        # Act
        _ = [chunk async for chunk in agent.run("What is the weather?")]

        # Assert: tools were retrieved and passed to LLM
        assert llm.last_tools is not None
        assert len(llm.last_tools) == 1
        assert llm.last_tools[0].name == "get_current_weather"

    async def test_starts_with_user_message(self) -> None:
        """Given: user query
        When: AgentService.run(query)
        Then: first message in conversation is user role with query
        """
        # Arrange
        llm = StubLLMPort([LLMChunk.text_delta("Response"), LLMChunk.stop("end_turn")])
        mcp = StubMCPClientPort()
        agent = AgentService(llm=llm, mcp_client=mcp)

        # Act
        _ = [chunk async for chunk in agent.run("What is the weather?")]

        # Assert: conversation contains user message
        assert llm.last_conversation is not None
        assert len(llm.last_conversation.messages) > 0
        assert llm.last_conversation.messages[0].role == "user"
        assert llm.last_conversation.messages[0].content == "What is the weather?"


@pytest.mark.asyncio
class TestAgentServiceErrorHandlingShould:
    """AgentService error handling behavior."""

    async def test_tool_error_injects_error_message_and_continues(self) -> None:
        """Given: tool call raises an exception
        When: AgentService processes the tool_use_complete
        Then: injects error message and continues (does not crash)
        """
        # Arrange: first LLM call emits tool use
        first_stream = [
            LLMChunk.tool_use_start("bad_tool", "id1"),
            LLMChunk.tool_use_complete("bad_tool", "id1", {}),
            LLMChunk.stop("end_turn"),
        ]
        # Second LLM call after error continues
        second_stream = [
            LLMChunk.text_delta("Tool failed, but I can still respond."),
            LLMChunk.stop("end_turn"),
        ]
        llm = StubLLMPort([first_stream, second_stream])

        # MCP client that raises error
        class ErrorMCPClient:
            tools: list[ToolSchema] = [ToolSchema(name="bad_tool", description="d", input_schema={})]
            calls: list[str] = []

            async def list_tools(self) -> list[ToolSchema]:
                return self.tools

            async def call_tool(self, name: str, args: dict[str, Any]) -> str:
                self.calls.append(name)
                raise RuntimeError("network timeout")

        mcp = ErrorMCPClient()  # type: ignore
        agent = AgentService(llm=llm, mcp_client=mcp)  # type: ignore

        # Act
        chunks: list[LLMChunk] = []
        async for chunk in agent.run("query"):
            chunks.append(chunk)

        # Assert: tool was attempted
        assert len(mcp.calls) == 1  # type: ignore
        # Assert: we still got chunks (didn't crash)
        assert any(c.type == "text_delta" for c in chunks)

    async def test_empty_tool_name_does_not_crash(self) -> None:
        """Given: LLM returns tool_use_complete with empty tool name
        When: AgentService processes the chunk
        Then: does not crash
        """
        stream = [
            LLMChunk.tool_use_complete("", "id1", {}),
            LLMChunk.stop("end_turn"),
        ]
        llm = StubLLMPort(stream)
        mcp = StubMCPClientPort()
        agent = AgentService(llm=llm, mcp_client=mcp)

        # Act
        chunks: list[LLMChunk] = []
        async for chunk in agent.run("query"):
            chunks.append(chunk)

        # Assert: got stop chunk (didn't crash)
        assert any(c.type == "stop" for c in chunks)

    async def test_tool_name_none_does_not_crash(self) -> None:
        """Given: tool_use_complete with tool_name=None
        When: AgentService processes it
        Then: does not crash, continues
        """
        stream = [
            LLMChunk(type="tool_use_complete", tool_name=None, tool_use_id="id1"),
            LLMChunk.stop("end_turn"),
        ]
        llm = StubLLMPort(stream)
        mcp = StubMCPClientPort()
        agent = AgentService(llm=llm, mcp_client=mcp)

        # Act
        chunks: list[LLMChunk] = []
        async for chunk in agent.run("query"):
            chunks.append(chunk)

        # Assert: no crash
        assert len(chunks) > 0
