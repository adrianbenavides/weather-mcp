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
from mcp_chat.domain.conversation import LLMChunk, Message, ToolSchema


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
        async for chunk in agent.run_turn("What is the weather in London?"):
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
        async for chunk in agent.run_turn("What is the weather in London?"):
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
        async for chunk in agent.run_turn("What's the weather?"):
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
        _ = [chunk async for chunk in agent.run_turn("What is the weather?")]

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
        _ = [chunk async for chunk in agent.run_turn("What is the weather?")]

        # Assert: conversation contains user message
        assert llm.last_conversation is not None
        assert len(llm.last_conversation.messages) > 0
        assert llm.last_conversation.messages[0].role == "user"
        assert llm.last_conversation.messages[0].content == "What is the weather?"


@pytest.mark.asyncio
class TestAgentServiceSessionStateShould:
    """AgentService session state persistence."""

    async def test_fresh_instance_starts_with_empty_history(self) -> None:
        """Given: newly created AgentService instance
        When: inspecting _messages before any run_turn() call
        Then: _messages list is empty

        PORT-TO-PORT: test enters through AgentService port by calling run_turn().
        """
        # Arrange
        llm = StubLLMPort([LLMChunk.stop("end_turn")])
        mcp = StubMCPClientPort()
        agent = AgentService(llm=llm, mcp_client=mcp)

        # Act: first run_turn call
        _ = [chunk async for chunk in agent.run_turn("test")]

        # Assert: after first call, only the user message from first turn exists
        # (We verify via the conversation passed to LLM, which should have 1 message on first call)
        assert llm.last_conversation is not None
        # On first turn, only 1 user message should exist in conversation
        assert len(llm.last_conversation.messages) == 1

    async def test_run_turn_appends_user_message_to_accumulated_messages(self) -> None:
        """Given: AgentService instance with prior conversation
        When: run_turn() is called with a new query
        Then: user message is appended to existing accumulated messages

        PORT-TO-PORT: test enters through run_turn() driving port, asserts
        at driven port boundary (llm.last_conversation).
        """
        # Arrange
        first_stream = [LLMChunk.text_delta("response1"), LLMChunk.stop("end_turn")]
        second_stream = [LLMChunk.text_delta("response2"), LLMChunk.stop("end_turn")]

        llm = StubLLMPort([first_stream, second_stream])
        mcp = StubMCPClientPort()
        agent = AgentService(llm=llm, mcp_client=mcp)

        # Act: first turn
        _ = [chunk async for chunk in agent.run_turn("first query")]

        # Act: second turn
        _ = [chunk async for chunk in agent.run_turn("second query")]

        # Assert: conversation in second call includes both user messages
        assert llm.last_conversation is not None
        messages = llm.last_conversation.messages

        # Should have at least first query + second query as user messages
        user_messages = [m for m in messages if m.role == "user"]
        assert len(user_messages) >= 2
        assert user_messages[0].content == "first query"
        assert user_messages[-1].content == "second query"

    async def test_second_turn_receives_first_turn_history(self) -> None:
        """Given: two sequential run_turn() calls on same agent instance
        When: second call is made
        Then: LLM receives conversation history from first call

        PORT-TO-PORT: test enters through run_turn() driving port, asserts
        at driven port boundary (llm.last_conversation).
        """
        # Arrange: setup for first turn
        first_stream = [
            LLMChunk.text_delta("Weather is sunny."),
            LLMChunk.stop("end_turn"),
        ]

        # Setup for second turn
        second_stream = [
            LLMChunk.text_delta("That's great!"),
            LLMChunk.stop("end_turn"),
        ]

        llm = StubLLMPort([first_stream, second_stream])
        mcp = StubMCPClientPort()
        agent = AgentService(llm=llm, mcp_client=mcp)

        # Act: First turn
        chunks_turn1: list[LLMChunk] = []
        async for chunk in agent.run_turn("What is the weather?"):
            chunks_turn1.append(chunk)

        # Capture conversation from first turn
        conversation_turn1 = llm.last_conversation

        # Act: Second turn on same instance
        chunks_turn2: list[LLMChunk] = []
        async for chunk in agent.run_turn("Is it cold?"):
            chunks_turn2.append(chunk)

        # Capture conversation from second turn
        conversation_turn2 = llm.last_conversation

        # Assert: first turn has 1 message (user query)
        assert len(conversation_turn1.messages) == 1
        assert conversation_turn1.messages[0].role == "user"
        assert conversation_turn1.messages[0].content == "What is the weather?"

        # Assert: second turn has MORE messages (includes first turn history + new user query)
        # Expected: [user1, assistant1, user2]
        assert len(conversation_turn2.messages) >= 2, (
            f"Expected at least 2 messages in second turn, got {len(conversation_turn2.messages)}"
        )

        # Assert: first message is from first turn
        assert conversation_turn2.messages[0].role == "user"
        assert conversation_turn2.messages[0].content == "What is the weather?"

        # Assert: second turn's new query is appended
        final_message = conversation_turn2.messages[-1]
        assert final_message.role == "user"
        assert final_message.content == "Is it cold?"


@pytest.mark.asyncio
class TestAgentServiceConversationCompactionShould:
    """AgentService conversation compaction via compact() method."""

    async def test_compact_is_noop_if_empty_messages(self) -> None:
        """Given: agent with empty message history
        When: compact() is called
        Then: returns immediately (no-op), no LLM call made
        """
        # Arrange
        llm = StubLLMPort([LLMChunk.stop("end_turn")])
        mcp = StubMCPClientPort()
        agent = AgentService(llm=llm, mcp_client=mcp)

        # Act: compact with no messages
        await agent.compact()

        # Assert: LLM was not called (call_count should be 0)
        assert llm.call_count == 0, "compact() should not call LLM when message history is empty"

    async def test_compact_is_noop_if_one_message(self) -> None:
        """Given: agent with exactly one message in history
        When: compact() is called
        Then: returns immediately (no-op), no LLM call made
        """
        # Arrange
        llm = StubLLMPort([LLMChunk.stop("end_turn")])
        mcp = StubMCPClientPort()
        agent = AgentService(llm=llm, mcp_client=mcp)

        # Act: add one user message and call compact
        _ = [chunk async for chunk in agent.run_turn("Hello")]

        llm_call_count_before = llm.call_count
        await agent.compact()

        # Assert: no additional LLM call made for compact
        assert llm.call_count == llm_call_count_before, (
            "compact() should not call LLM when only 1 message exists"
        )

    async def test_compact_calls_llm_without_tools(self) -> None:
        """Given: agent with >1 messages in history
        When: compact() is called
        Then: calls llm.stream_response() with tools=[] (empty list, no tools)
        """
        # Arrange: two turns to build history
        first_stream = [LLMChunk.text_delta("Response 1."), LLMChunk.stop("end_turn")]
        second_stream = [LLMChunk.text_delta("Response 2."), LLMChunk.stop("end_turn")]
        summary_stream = [LLMChunk.text_delta("Summary."), LLMChunk.stop("end_turn")]

        llm = StubLLMPort([first_stream, second_stream, summary_stream])
        mcp = StubMCPClientPort()
        agent = AgentService(llm=llm, mcp_client=mcp)

        # Act: build history
        _ = [chunk async for chunk in agent.run_turn("Query 1")]
        _ = [chunk async for chunk in agent.run_turn("Query 2")]

        # Act: compact
        await agent.compact()

        # Assert: last LLM call should have tools=[]
        assert llm.last_tools is not None
        assert llm.last_tools == [], f"compact() should call LLM with tools=[], got {llm.last_tools}"

    async def test_compact_replaces_messages_with_summary(self) -> None:
        """Given: agent with >1 accumulated messages
        When: compact() is called and LLM returns summary
        Then: self._messages is replaced with single summary Message
        """
        # Arrange
        first_stream = [LLMChunk.text_delta("First."), LLMChunk.stop("end_turn")]
        second_stream = [LLMChunk.text_delta("Second."), LLMChunk.stop("end_turn")]
        summary_stream = [
            LLMChunk.text_delta("This is "),
            LLMChunk.text_delta("the summary."),
            LLMChunk.stop("end_turn"),
        ]

        llm = StubLLMPort([first_stream, second_stream, summary_stream])
        mcp = StubMCPClientPort()
        agent = AgentService(llm=llm, mcp_client=mcp)

        # Act: build history
        _ = [chunk async for chunk in agent.run_turn("A")]
        _ = [chunk async for chunk in agent.run_turn("B")]

        # Verify we have multiple messages before compact
        conversation_before = llm.last_conversation
        assert len(conversation_before.messages) >= 2

        # Act: compact
        await agent.compact()

        # Assert: after compact, agent should have exactly 1 message (the summary)
        # We verify by making another LLM call and checking conversation passed to it
        follow_up_stream = [LLMChunk.text_delta("Follow-up."), LLMChunk.stop("end_turn")]
        llm.streams.append(follow_up_stream)

        _ = [chunk async for chunk in agent.run_turn("Follow-up query")]

        conversation_after = llm.last_conversation
        # Should have: [summary_message, follow_up_query]
        assert len(conversation_after.messages) == 2
        assert conversation_after.messages[0].role == "user"
        assert "summary" in conversation_after.messages[0].content.lower()
        assert conversation_after.messages[1].content == "Follow-up query"

    async def test_compact_replaces_multiple_messages_with_single_summary(self) -> None:
        """Given: agent with accumulated conversation history (>1 turn)
        When: compact() is called
        Then: self._messages is replaced with single Message containing summary
        And: follow-up run_turn() uses the compacted context

        ACCEPTANCE TEST: exercises compact() through AgentService port,
        verifies summary replaces history, verifies persistence to next turn.
        """
        # Arrange: build conversation history over multiple turns
        first_stream = [LLMChunk.text_delta("London is rainy."), LLMChunk.stop("end_turn")]
        second_stream = [LLMChunk.text_delta("Take an umbrella!"), LLMChunk.stop("end_turn")]

        # Third stream is for LLM summarization during compact()
        summary_stream = [
            LLMChunk.text_delta("User asked about London weather, "),
            LLMChunk.text_delta("got rainy forecast, "),
            LLMChunk.text_delta("received umbrella advice."),
            LLMChunk.stop("end_turn"),
        ]

        # Fourth stream is for follow-up turn after compact
        follow_up_stream = [LLMChunk.text_delta("Yes, sunny now."), LLMChunk.stop("end_turn")]

        llm = StubLLMPort([first_stream, second_stream, summary_stream, follow_up_stream])
        mcp = StubMCPClientPort()
        agent = AgentService(llm=llm, mcp_client=mcp)

        # Act: Build history via two run_turn() calls
        _ = [chunk async for chunk in agent.run_turn("What is the weather in London?")]
        _ = [chunk async for chunk in agent.run_turn("What should I wear?")]

        # Verify history before compact: should have at least 2 user messages
        conversation_before = llm.last_conversation
        assert conversation_before is not None
        user_msgs_before = [m for m in conversation_before.messages if m.role == "user"]
        assert len(user_msgs_before) >= 2, "Expected at least 2 user messages before compact"

        # Act: Call compact()
        await agent.compact()

        # Act: Follow-up turn after compaction
        follow_up_chunks: list[LLMChunk] = []
        async for chunk in agent.run_turn("Is it sunny now?"):
            follow_up_chunks.append(chunk)

        # Assert: LLM received conversation with compacted history
        conversation_after = llm.last_conversation
        assert conversation_after is not None

        # After compact, history should be replaced with summary + new user query
        # Expected messages: [summary_message, new_user_query]
        assert len(conversation_after.messages) == 2, (
            f"Expected 2 messages after compact (summary + new query), got {len(conversation_after.messages)}"
        )

        # First message should be summary (from compact)
        first_msg = conversation_after.messages[0]
        assert first_msg.role == "user", "Summary should be in user role for context"
        assert (
            "weather" in first_msg.content.lower()
            or "london" in first_msg.content.lower()
            or "rainy" in first_msg.content.lower()
            or "umbrella" in first_msg.content.lower()
        ), f"Summary should contain conversation context, got: {first_msg.content}"

        # Second message should be the follow-up query
        second_msg = conversation_after.messages[1]
        assert second_msg.role == "user"
        assert second_msg.content == "Is it sunny now?"

        # Assert: follow-up response was received (no crash)
        assert len(follow_up_chunks) > 0
        assert any(c.type == "text_delta" for c in follow_up_chunks)


@pytest.mark.asyncio
class TestAgentServiceInternalChunkProcessingShould:
    """AgentService _process_chunk internal behavior via run_turn port."""

    async def test_tool_use_id_stored_exactly_from_chunk(self) -> None:
        """Given: tool_use_start chunk with specific tool_use_id
        When: tool call completes and second LLM call is made
        Then: conversation messages contain exact tool_use_id (not None, not mangled)
        """
        first_stream = [
            LLMChunk.tool_use_start("get_weather", "exact-id-abc123"),
            LLMChunk.tool_use_complete("get_weather", "exact-id-abc123", {"city": "London"}),
            LLMChunk.stop("end_turn"),
        ]
        second_stream = [LLMChunk.text_delta("sunny"), LLMChunk.stop("end_turn")]

        llm = StubLLMPort([first_stream, second_stream])
        mcp = StubMCPClientPort(
            tools=[ToolSchema(name="get_weather", description="d", input_schema={})],
            tool_results={"get_weather": "sunny"},
        )
        agent = AgentService(llm=llm, mcp_client=mcp)

        _ = [chunk async for chunk in agent.run_turn("weather?")]

        # Second LLM call receives conversation with assistant + tool messages
        assert llm.last_conversation is not None
        messages = llm.last_conversation.messages
        tool_use_ids = [m.tool_use_id for m in messages if m.tool_use_id is not None]
        assert "exact-id-abc123" in tool_use_ids, (
            f"Expected 'exact-id-abc123' in tool_use_ids, got: {tool_use_ids}"
        )

    async def test_tool_name_empty_string_fallback_not_mangled(self) -> None:
        """Given: tool_use_start chunk with tool_name=None (fallback to '')
        When: chunk processed via _process_chunk
        Then: buffer receives '' (empty string), not 'XXXX'
        """
        first_stream = [
            LLMChunk(type="tool_use_start", tool_name=None, tool_use_id="id1"),
            LLMChunk.tool_use_complete("", "id1", {}),
            LLMChunk.stop("end_turn"),
        ]
        second_stream = [LLMChunk.text_delta("done"), LLMChunk.stop("end_turn")]

        llm = StubLLMPort([first_stream, second_stream])
        mcp = StubMCPClientPort()
        agent = AgentService(llm=llm, mcp_client=mcp)

        _ = [chunk async for chunk in agent.run_turn("q")]

        # If name was mutated to "XXXX", mcp.call_tool would be called with "XXXX"
        # If name was correctly "" (falsy), _execute_tool_and_update_conversation returns early
        assert mcp.call_count == 0, "Empty tool name should cause early return (no tool call)"

    async def test_tool_use_id_empty_string_fallback_not_mangled(self) -> None:
        """Given: tool_use_start chunk with tool_use_id=''
        When: processed, tool call completes
        Then: messages use '' not 'XXXX' for tool_use_id
        """
        first_stream = [
            LLMChunk(type="tool_use_start", tool_name="get_weather", tool_use_id=""),
            LLMChunk.tool_use_complete("get_weather", "", {"city": "X"}),
            LLMChunk.stop("end_turn"),
        ]
        second_stream = [LLMChunk.text_delta("done"), LLMChunk.stop("end_turn")]

        llm = StubLLMPort([first_stream, second_stream])
        mcp = StubMCPClientPort(
            tools=[ToolSchema(name="get_weather", description="d", input_schema={})],
            tool_results={"get_weather": "result"},
        )
        agent = AgentService(llm=llm, mcp_client=mcp)

        _ = [chunk async for chunk in agent.run_turn("q")]

        messages = llm.last_conversation.messages
        for m in messages:
            if m.tool_use_id is not None:
                assert m.tool_use_id != "XXXX", "tool_use_id should not be mangled to 'XXXX'"

    async def test_tool_use_input_delta_chunk_buffered_correctly(self) -> None:
        """Given: tool_use_input chunk with input_chunk=None and tool_use_input_delta='delta_val'
        When: processed, tool call completes with accumulated delta input
        Then: tool is called with correct delta input (not empty string from operator mutation)
        """
        first_stream = [
            LLMChunk.tool_use_start("get_weather", "id1"),
            LLMChunk(
                type="tool_use_input",
                tool_name=None,
                tool_use_id=None,
                input_chunk=None,
                tool_use_input_delta='{"city":',
            ),
            LLMChunk(
                type="tool_use_input",
                tool_name=None,
                tool_use_id=None,
                input_chunk=None,
                tool_use_input_delta='"London"}',
            ),
            LLMChunk.tool_use_complete("get_weather", "id1", {"city": "London"}),
            LLMChunk.stop("end_turn"),
        ]
        second_stream = [LLMChunk.text_delta("done"), LLMChunk.stop("end_turn")]

        llm = StubLLMPort([first_stream, second_stream])
        mcp = StubMCPClientPort(
            tools=[ToolSchema(name="get_weather", description="d", input_schema={})],
            tool_results={"get_weather": "sunny"},
        )
        agent = AgentService(llm=llm, mcp_client=mcp)

        _ = [chunk async for chunk in agent.run_turn("weather?")]

        assert mcp.call_count == 1, "Tool should be called once"

    async def test_tool_use_input_chunk_field_used_when_present(self) -> None:
        """Given: tool_use_input chunk with input_chunk='partial' (not None)
        When: processed
        Then: 'partial' buffered (input_chunk takes precedence over tool_use_input_delta)
        """
        first_stream = [
            LLMChunk.tool_use_start("get_weather", "id1"),
            LLMChunk(
                type="tool_use_input",
                tool_name=None,
                tool_use_id=None,
                input_chunk='{"city": "Paris"}',
                tool_use_input_delta=None,
            ),
            LLMChunk.tool_use_complete("get_weather", "id1", {"city": "Paris"}),
            LLMChunk.stop("end_turn"),
        ]
        second_stream = [LLMChunk.text_delta("done"), LLMChunk.stop("end_turn")]

        llm = StubLLMPort([first_stream, second_stream])
        mcp = StubMCPClientPort(
            tools=[ToolSchema(name="get_weather", description="d", input_schema={})],
            tool_results={"get_weather": "cloudy"},
        )
        agent = AgentService(llm=llm, mcp_client=mcp)

        _ = [chunk async for chunk in agent.run_turn("weather?")]

        assert mcp.call_count == 1

    async def test_second_llm_call_receives_non_none_conversation_after_tool(self) -> None:
        """Given: first LLM stream has tool call
        When: tool executes and second LLM call made
        Then: second call receives valid Conversation with messages (not None/empty)
        Kills run_turn_mutmut_37 (conversation=None) and mutmut_38 (messages=None).
        """
        first_stream = [
            LLMChunk.tool_use_start("get_weather", "id1"),
            LLMChunk.tool_use_complete("get_weather", "id1", {"city": "London"}),
            LLMChunk.stop("end_turn"),
        ]
        second_stream = [LLMChunk.text_delta("sunny"), LLMChunk.stop("end_turn")]

        llm = StubLLMPort([first_stream, second_stream])
        mcp = StubMCPClientPort(
            tools=[ToolSchema(name="get_weather", description="d", input_schema={})],
            tool_results={"get_weather": "sunny 15C"},
        )
        agent = AgentService(llm=llm, mcp_client=mcp)

        _ = [chunk async for chunk in agent.run_turn("weather?")]

        assert llm.call_count == 2, "LLM should be called twice (first call + after tool)"
        # Verify second call received a valid Conversation with messages
        conversation = llm.last_conversation
        assert conversation is not None, "Second LLM call must receive non-None conversation"
        assert conversation.messages is not None, "Conversation messages must not be None"
        assert len(conversation.messages) >= 3, (
            f"Second call conversation should have user+assistant+tool messages, got {len(conversation.messages)}"
        )


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
        async for chunk in agent.run_turn("query"):
            chunks.append(chunk)

        # Assert: tool was attempted
        assert len(mcp.calls) == 1  # type: ignore
        # Assert: error message visible in streamed response
        text_output = "".join(c.text or "" for c in chunks if c.type == "text_delta")
        assert "Tool failed" in text_output

    async def test_tool_error_message_has_exact_format(self) -> None:
        """Given: tool call raises RuntimeError('network timeout')
        When: AgentService handles the exception
        Then: error message stored is exactly 'Error calling tool bad_tool: network timeout'
        Kills _execute_tool_mutmut_11 (mangled format string).
        """
        first_stream = [
            LLMChunk.tool_use_start("bad_tool", "id1"),
            LLMChunk.tool_use_complete("bad_tool", "id1", {}),
            LLMChunk.stop("end_turn"),
        ]
        second_stream = [LLMChunk.text_delta("continuing"), LLMChunk.stop("end_turn")]

        llm = StubLLMPort([first_stream, second_stream])

        class ErrorMCPClient:
            tools: list[ToolSchema] = [ToolSchema(name="bad_tool", description="d", input_schema={})]

            async def list_tools(self) -> list[ToolSchema]:
                return self.tools

            async def call_tool(self, name: str, args: dict[str, Any]) -> str:
                raise RuntimeError("network timeout")

        mcp = ErrorMCPClient()  # type: ignore
        agent = AgentService(llm=llm, mcp_client=mcp)  # type: ignore

        _ = [chunk async for chunk in agent.run_turn("query")]

        # The error message is stored in messages as a tool result
        # Look for the tool message in the second LLM call's conversation
        assert llm.last_conversation is not None
        tool_messages = [m for m in llm.last_conversation.messages if m.role == "tool"]
        assert len(tool_messages) == 1
        assert tool_messages[0].content == "Error calling tool bad_tool: network timeout", (
            f"Exact error message format required, got: {tool_messages[0].content!r}"
        )

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
        async for chunk in agent.run_turn("query"):
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
        async for chunk in agent.run_turn("query"):
            chunks.append(chunk)

        # Assert: no crash
        assert len(chunks) > 0


@pytest.mark.asyncio
class TestAgentServiceMessageFieldsAfterToolCallShould:
    """Exact field values in messages after tool execution."""

    async def test_messages_after_tool_execution_have_correct_fields(self) -> None:
        """Given: tool call with known id/name/input
        When: AgentService processes tool_use_complete + tool result
        Then: messages[1] has exact assistant fields, messages[2] has exact tool fields
        """
        tool_id = "use-id-99"
        first_stream = [
            LLMChunk.tool_use_start("get_weather", tool_id),
            LLMChunk.tool_use_complete("get_weather", tool_id, {"city": "Tokyo"}),
            LLMChunk.stop("end_turn"),
        ]
        second_stream = [
            LLMChunk.text_delta("Tokyo: sunny 28C"),
            LLMChunk.stop("end_turn"),
        ]
        llm = StubLLMPort([first_stream, second_stream])
        mcp = StubMCPClientPort(
            tools=[
                ToolSchema(name="get_weather", description="Get weather", input_schema={"type": "object"})
            ],
            tool_results={"get_weather": "Tokyo: sunny 28C"},
        )
        agent = AgentService(llm=llm, mcp_client=mcp)

        async for _ in agent.run_turn("Weather in Tokyo?"):
            pass

        messages = llm.last_conversation.messages
        # messages[0]=user query, messages[1]=assistant tool_use, messages[2]=tool result
        assert len(messages) >= 3
        assistant_msg = messages[1]
        tool_msg = messages[2]

        assert assistant_msg.role == "assistant"
        assert assistant_msg.content == ""
        assert assistant_msg.tool_use_id == tool_id
        assert assistant_msg.tool_name == "get_weather"
        assert assistant_msg.tool_use_input == {"city": "Tokyo"}

        assert tool_msg.role == "tool"
        assert tool_msg.tool_use_id == tool_id
        assert tool_msg.content == "Tokyo: sunny 28C"


@pytest.mark.asyncio
class TestAgentServiceCompactInternalShould:
    """Internal compact() behavior."""

    async def test_compact_sends_current_messages_to_llm(self) -> None:
        """Given: 3 messages in history
        When: compact() called
        Then: LLM receives all 3 messages plus summarization prompt
        """
        summary_stream: list[LLMChunk] = [
            LLMChunk.text_delta("The summary."),
            LLMChunk.stop("end_turn"),
        ]
        llm = StubLLMPort(summary_stream)
        mcp = StubMCPClientPort()
        agent = AgentService(llm=llm, mcp_client=mcp)
        agent._messages = [  # type: ignore[attr-defined]
            Message(role="user", content="First message"),
            Message(role="assistant", content="First reply"),
            Message(role="user", content="Second message"),
        ]

        await agent.compact()

        # LLM received messages for summarization (3 history + summarization prompt)
        assert llm.last_conversation is not None
        assert len(llm.last_conversation.messages) >= 3

    async def test_compact_summary_starts_empty_accumulates_correctly(self) -> None:
        """Given: compact summary stream yields text chunks
        When: compact() called
        Then: summary_text starts empty (not prefixed) and accumulates all chunk text
        """
        summary_stream: list[LLMChunk] = [
            LLMChunk.text_delta("Part one. "),
            LLMChunk.text_delta("Part two. "),
            LLMChunk.text_delta("Part three."),
            LLMChunk.stop("end_turn"),
        ]
        llm = StubLLMPort(summary_stream)
        mcp = StubMCPClientPort()
        agent = AgentService(llm=llm, mcp_client=mcp)
        # Pre-populate messages to bypass the <=1 guard
        agent._messages = [  # type: ignore[attr-defined]
            Message(role="user", content="User query"),
            Message(role="assistant", content="Assistant reply"),
        ]

        await agent.compact()

        messages = agent._messages  # type: ignore[attr-defined]
        assert len(messages) == 1
        assert messages[0].content == "Part one. Part two. Part three."

    async def test_compact_skips_text_delta_with_none_text(self) -> None:
        """Given: compact summary stream has text_delta with None text
        When: compact() called
        Then: None text skipped, other chunks accumulated
        """
        summary_stream: list[LLMChunk] = [
            LLMChunk(type="text_delta", text=None),
            LLMChunk.text_delta("Real summary."),
            LLMChunk.stop("end_turn"),
        ]
        llm = StubLLMPort(summary_stream)
        mcp = StubMCPClientPort()
        agent = AgentService(llm=llm, mcp_client=mcp)
        agent._messages = [  # type: ignore[attr-defined]
            Message(role="user", content="User query"),
            Message(role="assistant", content="Assistant reply"),
        ]

        await agent.compact()

        messages = agent._messages  # type: ignore[attr-defined]
        assert len(messages) == 1
        assert messages[0].content == "Real summary."

    async def test_compact_sends_tools_empty_list_to_llm(self) -> None:
        """Given: compact is called
        When: LLM is invoked for summarization
        Then: tools=[] passed (not the MCP tools)
        """
        summary_stream: list[LLMChunk] = [
            LLMChunk.text_delta("Summary."),
            LLMChunk.stop("end_turn"),
        ]
        llm = StubLLMPort(summary_stream)
        mcp = StubMCPClientPort(
            tools=[ToolSchema(name="get_weather", description="Get weather", input_schema={"type": "object"})]
        )
        agent = AgentService(llm=llm, mcp_client=mcp)
        agent._messages = [  # type: ignore[attr-defined]
            Message(role="user", content="User query"),
            Message(role="assistant", content="Assistant reply"),
        ]

        await agent.compact()

        assert llm.last_tools == []
