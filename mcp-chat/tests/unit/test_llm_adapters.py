"""Unit tests for LLM adapters with mocked SDK calls.

This test verifies:
- AnthropicAdapter correctly maps streaming events to LLMChunk
- OpenAIAdapter correctly maps streaming chunks to LLMChunk
- All chunk type transformations work correctly
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp_chat.application.config import AppConfig
from mcp_chat.domain.conversation import Conversation, LLMChunk, Message, ToolSchema


def _make_anthropic_adapter() -> "mcp_chat.adapters.llm.anthropic_adapter.AnthropicAdapter":  # type: ignore
    """Create an AnthropicAdapter with test config."""
    from mcp_chat.adapters.llm.anthropic_adapter import AnthropicAdapter

    config = AppConfig(anthropic_api_key="test-key")
    return AnthropicAdapter(config=config)


def _setup_anthropic_stream(adapter, events):  # type: ignore
    """Setup mock stream for AnthropicAdapter."""
    mock_stream_ctx = AsyncMock()
    mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_ctx)
    mock_stream_ctx.__aexit__ = AsyncMock(return_value=None)

    async def async_gen():
        for event in events:
            yield event

    mock_stream_ctx.__aiter__ = lambda self: async_gen().__aiter__()
    adapter._client.messages.stream = MagicMock(return_value=mock_stream_ctx)
    return adapter


class TestAnthropicAdapterUnit:
    """AnthropicAdapter unit tests with mocked Anthropic SDK."""

    @pytest.mark.asyncio
    async def test_anthropic_adapter_maps_text_delta_event_to_chunk(self) -> None:
        """AnthropicAdapter maps ContentBlockDeltaEvent text_delta to LLMChunk."""
        adapter = _make_anthropic_adapter()

        # Mock the Anthropic SDK event
        mock_event = MagicMock()
        mock_event.type = "content_block_delta"
        mock_event.delta.type = "text_delta"
        mock_event.delta.text = "Hello"

        # Create an async context manager mock
        mock_stream_ctx = AsyncMock()
        mock_stream_ctx.__aenter__ = AsyncMock(return_value=mock_stream_ctx)
        mock_stream_ctx.__aexit__ = AsyncMock(return_value=None)

        # Make it async iterable
        async def async_gen():
            yield mock_event

        mock_stream_ctx.__aiter__ = lambda self: async_gen().__aiter__()

        # Mock the client.messages.stream
        adapter._client.messages.stream = MagicMock(return_value=mock_stream_ctx)

        conversation = Conversation(messages=[Message(role="user", content="Say hello")])
        tools: list[ToolSchema] = []

        chunks = []
        async for chunk in adapter.stream_response(conversation, tools):
            chunks.append(chunk)

        # Verify at least one chunk was produced
        assert len(chunks) > 0, "Should produce at least one chunk"

        # Find the text_delta chunk
        text_chunks = [c for c in chunks if c.type == "text_delta"]
        assert len(text_chunks) > 0, "Should have text_delta chunk"

    @pytest.mark.asyncio
    async def test_content_block_start_tool_use_initializes_buffer(self) -> None:
        """Given: content_block_start event with tool_use type
        When: AnthropicAdapter processes it
        Then: tool_use_start chunk is emitted (when complete)
        """
        adapter = _make_anthropic_adapter()

        # Setup events: tool_use_start, input delta, content_block_stop, message_stop
        mock_events = []

        # content_block_start with tool_use
        event1 = MagicMock()
        event1.type = "content_block_start"
        event1.content_block.type = "tool_use"
        event1.content_block.id = "id1"
        event1.content_block.name = "get_weather"
        mock_events.append(event1)

        # content_block_delta with input JSON
        event2 = MagicMock()
        event2.type = "content_block_delta"
        event2.delta.type = "input_json_delta"
        event2.delta.partial_json = '{"location"'
        mock_events.append(event2)

        # content_block_stop
        event3 = MagicMock()
        event3.type = "content_block_stop"
        mock_events.append(event3)

        # message_stop
        event4 = MagicMock()
        event4.type = "message_stop"
        mock_events.append(event4)

        adapter = _setup_anthropic_stream(adapter, mock_events)

        conversation = Conversation(messages=[Message(role="user", content="Test")])
        chunks = [c async for c in adapter.stream_response(conversation, [])]

        # Should have tool_use_start and tool_use_complete
        tool_start = [c for c in chunks if c.type == "tool_use_start"]
        assert len(tool_start) == 1
        assert tool_start[0].tool_name == "get_weather"
        assert tool_start[0].tool_use_id == "id1"

    @pytest.mark.asyncio
    async def test_content_block_start_non_tool_use_does_not_init_buffer(self) -> None:
        """Given: content_block_start event with text type (not tool_use)
        When: AnthropicAdapter processes it
        Then: no tool_use chunks emitted
        """
        adapter = _make_anthropic_adapter()

        mock_events = []

        # content_block_start with text (not tool_use)
        event1 = MagicMock()
        event1.type = "content_block_start"
        event1.content_block.type = "text"
        mock_events.append(event1)

        # content_block_delta with text
        event2 = MagicMock()
        event2.type = "content_block_delta"
        event2.delta.type = "text_delta"
        event2.delta.text = "Hello"
        mock_events.append(event2)

        # message_stop
        event3 = MagicMock()
        event3.type = "message_stop"
        mock_events.append(event3)

        adapter = _setup_anthropic_stream(adapter, mock_events)

        conversation = Conversation(messages=[Message(role="user", content="Test")])
        chunks = [c async for c in adapter.stream_response(conversation, [])]

        # Should have text_delta and stop, no tool chunks
        tool_chunks = [c for c in chunks if c.type.startswith("tool_use")]
        assert len(tool_chunks) == 0

    @pytest.mark.asyncio
    async def test_input_json_delta_yields_tool_input_delta_chunk(self) -> None:
        """Given: content_block_delta with input_json_delta after tool_use_start
        When: processed
        Then: tool_use_input chunk is yielded
        """
        adapter = _make_anthropic_adapter()

        mock_events = []

        # First, start the tool use
        event0 = MagicMock()
        event0.type = "content_block_start"
        event0.content_block.type = "tool_use"
        event0.content_block.id = "id1"
        event0.content_block.name = "tool"
        mock_events.append(event0)

        # Then input delta
        event1 = MagicMock()
        event1.type = "content_block_delta"
        event1.delta.type = "input_json_delta"
        event1.delta.partial_json = '{"loc'
        mock_events.append(event1)

        event2 = MagicMock()
        event2.type = "message_stop"
        mock_events.append(event2)

        adapter = _setup_anthropic_stream(adapter, mock_events)

        conversation = Conversation(messages=[Message(role="user", content="Test")])
        chunks = [c async for c in adapter.stream_response(conversation, [])]

        input_chunks = [c for c in chunks if c.type == "tool_use_input"]
        assert len(input_chunks) == 1
        assert input_chunks[0].tool_use_input_delta == '{"loc'

    @pytest.mark.asyncio
    async def test_content_block_stop_with_buffer_emits_tool_use_chunks(self) -> None:
        """Full sequence: start, delta, stop, message_stop emits complete tool_use chunks."""
        adapter = _make_anthropic_adapter()

        mock_events = []

        event1 = MagicMock()
        event1.type = "content_block_start"
        event1.content_block.type = "tool_use"
        event1.content_block.id = "id1"
        event1.content_block.name = "get_weather"
        mock_events.append(event1)

        event2 = MagicMock()
        event2.type = "content_block_delta"
        event2.delta.type = "input_json_delta"
        event2.delta.partial_json = '{"location": "London"}'
        mock_events.append(event2)

        event3 = MagicMock()
        event3.type = "content_block_stop"
        mock_events.append(event3)

        event4 = MagicMock()
        event4.type = "message_stop"
        mock_events.append(event4)

        adapter = _setup_anthropic_stream(adapter, mock_events)

        conversation = Conversation(messages=[Message(role="user", content="Test")])
        chunks = [c async for c in adapter.stream_response(conversation, [])]

        tool_start = [c for c in chunks if c.type == "tool_use_start"]
        tool_complete = [c for c in chunks if c.type == "tool_use_complete"]

        assert len(tool_start) == 1
        assert len(tool_complete) == 1
        assert tool_complete[0].tool_use_input_complete == {"location": "London"}

    @pytest.mark.asyncio
    async def test_content_block_stop_without_buffer_emits_nothing(self) -> None:
        """Given: content_block_stop without prior tool_use buffer
        When: processed
        Then: no tool chunks (buffer is empty)
        """
        adapter = _make_anthropic_adapter()

        mock_events = []

        event1 = MagicMock()
        event1.type = "content_block_stop"
        mock_events.append(event1)

        event2 = MagicMock()
        event2.type = "message_stop"
        mock_events.append(event2)

        adapter = _setup_anthropic_stream(adapter, mock_events)

        conversation = Conversation(messages=[Message(role="user", content="Test")])
        chunks = [c async for c in adapter.stream_response(conversation, [])]

        tool_chunks = [c for c in chunks if c.type.startswith("tool_use")]
        assert len(tool_chunks) == 0

    @pytest.mark.asyncio
    async def test_message_stop_yields_stop_chunk(self) -> None:
        """Given: message_stop event
        When: processed
        Then: stop chunk with stop_reason='end_turn'
        """
        adapter = _make_anthropic_adapter()

        mock_events = []
        event = MagicMock()
        event.type = "message_stop"
        mock_events.append(event)

        adapter = _setup_anthropic_stream(adapter, mock_events)

        conversation = Conversation(messages=[Message(role="user", content="Test")])
        chunks = [c async for c in adapter.stream_response(conversation, [])]

        stop_chunks = [c for c in chunks if c.type == "stop"]
        assert len(stop_chunks) == 1
        assert stop_chunks[0].stop_reason == "end_turn"

    def test_emit_tool_use_chunks_with_invalid_json_returns_empty_input(self) -> None:
        """Given: tool_use_buffer with invalid JSON
        When: _emit_tool_use_chunks called
        Then: tool_use_complete has empty dict input
        """
        from mcp_chat.adapters.llm.anthropic_adapter import AnthropicAdapter

        adapter = _make_anthropic_adapter()
        buffer = {
            "name": "tool",
            "id": "id1",
            "input_parts": ["{invalid json"],
        }

        chunks = adapter._emit_tool_use_chunks(buffer)

        assert len(chunks) == 2
        assert chunks[0].type == "tool_use_start"
        assert chunks[1].type == "tool_use_complete"
        assert chunks[1].tool_use_input_complete == {}

    def test_emit_tool_use_chunks_with_valid_json_returns_parsed_input(self) -> None:
        """Given: tool_use_buffer with valid JSON parts
        When: _emit_tool_use_chunks called
        Then: tool_use_complete has parsed input dict
        """
        from mcp_chat.adapters.llm.anthropic_adapter import AnthropicAdapter

        adapter = _make_anthropic_adapter()
        buffer = {
            "name": "tool",
            "id": "id1",
            "input_parts": ['{"loc', 'ation": "London"}'],
        }

        chunks = adapter._emit_tool_use_chunks(buffer)

        assert len(chunks) == 2
        assert chunks[1].tool_use_input_complete == {"loc": "ation", "": "London"} or chunks[
            1
        ].tool_use_input_complete == {"location": "London"}


class AsyncIteratorContextManager:
    """Helper: async context manager that is also async iterable."""

    def __init__(self, items):
        self.items = items
        self.index = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

    def __aiter__(self):
        self.index = 0
        return self

    async def __anext__(self):
        if self.index >= len(self.items):
            raise StopAsyncIteration
        item = self.items[self.index]
        self.index += 1
        return item


def _make_openai_adapter() -> "mcp_chat.adapters.llm.openai_adapter.OpenAIAdapter":  # type: ignore
    """Create an OpenAIAdapter with test config."""
    from mcp_chat.adapters.llm.openai_adapter import OpenAIAdapter

    config = AppConfig(openai_api_key="test-key")
    return OpenAIAdapter(config=config)


class TestOpenAIAdapterUnit:
    """OpenAIAdapter unit tests with mocked OpenAI SDK."""

    @pytest.mark.asyncio
    async def test_openai_adapter_maps_delta_to_text_chunk(self) -> None:
        """OpenAIAdapter maps delta.content to text_delta LLMChunk."""
        adapter = _make_openai_adapter()

        # Mock the OpenAI SDK chunk
        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = "Hello"
        mock_chunk.choices[0].delta.tool_calls = None
        mock_chunk.choices[0].finish_reason = "stop"

        mock_stream_ctx = AsyncIteratorContextManager([mock_chunk])

        # Mock the client.chat.completions.create to return the mock context manager
        adapter._client.chat.completions.create = MagicMock(return_value=mock_stream_ctx)

        conversation = Conversation(messages=[Message(role="user", content="Say hello")])
        tools: list[ToolSchema] = []

        chunks = []
        async for chunk in adapter.stream_response(conversation, tools):
            chunks.append(chunk)

        # Verify at least one chunk was produced
        assert len(chunks) > 0, "Should produce at least one chunk"

        # Find the text_delta chunk
        text_chunks = [c for c in chunks if c.type == "text_delta"]
        assert len(text_chunks) > 0, "Should have text_delta chunk"

    @pytest.mark.asyncio
    async def test_tool_call_delta_yields_tool_use_id_chunk(self) -> None:
        """Given: delta with tool_calls containing id
        When: processed
        Then: tool_use_id chunk is yielded
        """
        adapter = _make_openai_adapter()

        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]

        # Create tool_call mock
        tool_call = MagicMock()
        tool_call.id = "call1"
        tool_call.index = 0
        tool_call.function.name = "get_weather"
        tool_call.function.arguments = '{"loc'

        mock_chunk.choices[0].delta.content = None
        mock_chunk.choices[0].delta.tool_calls = [tool_call]
        mock_chunk.choices[0].finish_reason = None

        mock_stream_ctx = AsyncIteratorContextManager([mock_chunk])
        adapter._client.chat.completions.create = MagicMock(return_value=mock_stream_ctx)

        conversation = Conversation(messages=[Message(role="user", content="Test")])
        chunks = [c async for c in adapter.stream_response(conversation, [])]

        id_chunks = [c for c in chunks if c.type == "tool_use_id"]
        assert len(id_chunks) == 1
        assert id_chunks[0].tool_use_id == "call1"

    @pytest.mark.asyncio
    async def test_finish_reason_tool_calls_emits_complete_chunks(self) -> None:
        """Given: finish_reason='tool_calls'
        When: processed
        Then: tool_use_complete chunk emitted
        """
        adapter = _make_openai_adapter()

        # First chunk: tool_calls delta
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        tool_call = MagicMock()
        tool_call.id = "call1"
        tool_call.index = 0
        tool_call.function.name = "get_weather"
        tool_call.function.arguments = '{"location": "London"}'
        mock_chunk1.choices[0].delta.content = None
        mock_chunk1.choices[0].delta.tool_calls = [tool_call]
        mock_chunk1.choices[0].finish_reason = None

        # Second chunk: finish_reason
        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta.content = None
        mock_chunk2.choices[0].delta.tool_calls = None
        mock_chunk2.choices[0].finish_reason = "tool_calls"

        mock_stream_ctx = AsyncIteratorContextManager([mock_chunk1, mock_chunk2])
        adapter._client.chat.completions.create = MagicMock(return_value=mock_stream_ctx)

        conversation = Conversation(messages=[Message(role="user", content="Test")])
        chunks = [c async for c in adapter.stream_response(conversation, [])]

        complete_chunks = [c for c in chunks if c.type == "tool_use_complete"]
        assert len(complete_chunks) == 1
        assert complete_chunks[0].tool_use_input_complete == {"location": "London"}

    @pytest.mark.asyncio
    async def test_finish_reason_stop_yields_stop_chunk(self) -> None:
        """Given: finish_reason='stop'
        When: processed
        Then: stop chunk with stop_reason='end_turn'
        """
        adapter = _make_openai_adapter()

        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = None
        mock_chunk.choices[0].delta.tool_calls = None
        mock_chunk.choices[0].finish_reason = "stop"

        mock_stream_ctx = AsyncIteratorContextManager([mock_chunk])
        adapter._client.chat.completions.create = MagicMock(return_value=mock_stream_ctx)

        conversation = Conversation(messages=[Message(role="user", content="Test")])
        chunks = [c async for c in adapter.stream_response(conversation, [])]

        stop_chunks = [c for c in chunks if c.type == "stop"]
        assert len(stop_chunks) == 1
        assert stop_chunks[0].stop_reason == "end_turn"

    @pytest.mark.asyncio
    async def test_empty_choices_skipped(self) -> None:
        """Given: chunk with empty choices list
        When: processed
        Then: no crash, chunk skipped
        """
        adapter = _make_openai_adapter()

        mock_chunk = MagicMock()
        mock_chunk.choices = []

        mock_stream_ctx = AsyncIteratorContextManager([mock_chunk])
        adapter._client.chat.completions.create = MagicMock(return_value=mock_stream_ctx)

        conversation = Conversation(messages=[Message(role="user", content="Test")])
        chunks = [c async for c in adapter.stream_response(conversation, [])]

        # Should have no chunks from empty choices
        assert len(chunks) == 0

    @pytest.mark.asyncio
    async def test_none_delta_content_no_text_chunk(self) -> None:
        """Given: delta.content=None
        When: processed
        Then: no text_delta chunk
        """
        adapter = _make_openai_adapter()

        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta.content = None
        mock_chunk.choices[0].delta.tool_calls = None
        mock_chunk.choices[0].finish_reason = None

        mock_stream_ctx = AsyncIteratorContextManager([mock_chunk])
        adapter._client.chat.completions.create = MagicMock(return_value=mock_stream_ctx)

        conversation = Conversation(messages=[Message(role="user", content="Test")])
        chunks = [c async for c in adapter.stream_response(conversation, [])]

        text_chunks = [c for c in chunks if c.type == "text_delta"]
        assert len(text_chunks) == 0

    def test_accumulate_tool_calls_builds_buffer_correctly(self) -> None:
        """Given: tool_call with id, name, args
        When: _accumulate_tool_calls called
        Then: buffer entry created with all fields
        """
        adapter = _make_openai_adapter()

        tool_call = MagicMock()
        tool_call.index = 0
        tool_call.id = "call1"
        tool_call.function.name = "get_weather"
        tool_call.function.arguments = '{"x": 1}'

        buffer = {}
        adapter._accumulate_tool_calls([tool_call], buffer)

        assert 0 in buffer
        assert buffer[0]["id"] == "call1"
        assert buffer[0]["name"] == "get_weather"
        assert buffer[0]["args_parts"] == ['{"x": 1}']

    def test_accumulate_tool_calls_appends_args_to_existing(self) -> None:
        """Given: tool_call for existing buffer entry
        When: _accumulate_tool_calls called with args
        Then: args_parts appended to existing list
        """
        adapter = _make_openai_adapter()

        tool_call = MagicMock()
        tool_call.index = 0
        tool_call.id = None
        tool_call.function.name = None
        tool_call.function.arguments = "}"

        buffer = {0: {"id": "call1", "name": "tool", "args_parts": ['{"x']}}
        adapter._accumulate_tool_calls([tool_call], buffer)

        assert buffer[0]["args_parts"] == ['{"x', "}"]

    def test_emit_complete_tool_calls_skips_incomplete_entries(self) -> None:
        """Given: buffer with incomplete tool call (missing name or id)
        When: _emit_complete_tool_calls called
        Then: incomplete entry skipped, no chunk emitted
        """
        adapter = _make_openai_adapter()

        buffer = {
            0: {"id": None, "name": "tool", "args_parts": []},
            1: {"id": "call1", "name": None, "args_parts": []},
        }

        chunks = adapter._emit_complete_tool_calls(buffer)

        assert len(chunks) == 0
